"""On-disk cache for comparison payloads.

Building a payload loads and processes the full observed table (millions of rows
for some cities), so results are cached as JSON keyed by the mtimes of the two
input parquets. A changed input invalidates the entry automatically.

``get_or_build`` also de-duplicates concurrent requests for the same
still-uncached key: without this, two browser tabs (or a page reload racing
its own previous request) hitting the same cold cache would each redundantly
run the full expensive build and both write the same cache file. The
in-flight registry below is per-process (this backend runs as a single
process today, see ``executor.py``), so it only coalesces requests landing
in this same process -- which is exactly where concurrent HTTP requests
collide.
"""

from __future__ import annotations

import asyncio
import json
import threading
from concurrent.futures import Executor, Future
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable

from .config import CACHE_DIR

PAYLOAD_CACHE_VERSION = "v8"
MAX_CACHE_KEY_PREFIX = 120

_inflight: dict[str, Future] = {}
_inflight_lock = threading.Lock()


def _safe_part(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "-" for char in value)


def _key(
    exp_id: str,
    run_id: str,
    synthetic: Path,
    observed: Path | None,
    extra_paths: tuple[Path, ...] = (),
    extra_key: dict[str, Any] | None = None,
) -> str:
    syn_mtime = int(synthetic.stat().st_mtime)
    obs_mtime = int(observed.stat().st_mtime) if observed and observed.exists() else "synthetic-only"
    extra = [
        [str(path), path.stem, int(path.stat().st_mtime) if path.exists() else "missing"]
        for path in extra_paths
    ]
    key_parts = {
        "version": PAYLOAD_CACHE_VERSION,
        "exp_id": exp_id,
        "run_id": run_id,
        "synthetic": [str(synthetic), syn_mtime],
        "observed": [str(observed) if observed else None, obs_mtime],
        "extra": extra,
        "extra_key": extra_key,
    }
    digest = sha256(json.dumps(key_parts, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    prefix = f"{PAYLOAD_CACHE_VERSION}__{_safe_part(exp_id)}__{_safe_part(run_id)}"
    return f"{prefix[:MAX_CACHE_KEY_PREFIX]}__{digest}.json"


async def get_or_build(
    exp_id: str,
    run_id: str,
    synthetic: Path,
    observed: Path | None,
    *,
    build_fn: Callable[..., dict[str, Any]],
    build_args: tuple[Any, ...] = (),
    build_kwargs: dict[str, Any] | None = None,
    executor: Executor | None = None,
    refresh: bool = False,
    extra_paths: tuple[Path, ...] = (),
    extra_key: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Cache-or-build, coalescing concurrent callers for the same key.

    ``build_fn`` must be a picklable top-level function (not a closure/lambda)
    when ``executor`` is a ``ProcessPoolExecutor`` -- pass its arguments via
    ``build_args``/``build_kwargs`` instead of capturing them in a closure.
    Without an ``executor``, ``build_fn`` runs inline (on this async task,
    which for a sync ``build_fn`` means it still blocks the event loop --
    callers on the request path should always pass an executor).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = _key(exp_id, run_id, synthetic, observed, extra_paths, extra_key)
    cache_file = CACHE_DIR / cache_key
    if cache_file.exists() and not refresh:
        return json.loads(cache_file.read_text())

    kwargs = build_kwargs or {}
    with _inflight_lock:
        future = _inflight.get(cache_key)
        owner = future is None
        if owner:
            future = (
                executor.submit(build_fn, *build_args, **kwargs)
                if executor is not None
                else Future()
            )
            _inflight[cache_key] = future

    if owner and executor is None:
        try:
            future.set_result(build_fn(*build_args, **kwargs))
        except Exception as exc:  # noqa: BLE001 - propagated to every waiter below
            future.set_exception(exc)

    try:
        payload = await asyncio.wrap_future(future)
        if owner:
            cache_file.write_text(json.dumps(payload))
    finally:
        if owner:
            with _inflight_lock:
                _inflight.pop(cache_key, None)
    return payload


def get_or_build_parquet(
    cache_name: str,
    key_parts: tuple[str, ...],
    input_path: Path,
    build: Callable[[Path], None],
) -> Path:
    """Like ``get_or_build`` but for a parquet-file cache artifact keyed by a
    single input's mtime (e.g. a derived per-run precomputation), rather than
    the two-mtime JSON comparison-payload cache above.
    """
    subdir = CACHE_DIR / cache_name
    subdir.mkdir(parents=True, exist_ok=True)
    mtime = int(input_path.stat().st_mtime)
    out = subdir / (f"{'__'.join(key_parts)}__{mtime}.parquet")
    if not out.exists():
        build(out)
    return out
