"""On-disk cache for comparison payloads.

Building a payload loads and processes the full observed table (millions of rows
for some cities), so results are cached as JSON keyed by the mtimes of the two
input parquets. A changed input invalidates the entry automatically.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable

from .config import CACHE_DIR

PAYLOAD_CACHE_VERSION = "v5"
MAX_CACHE_KEY_PREFIX = 120


def _safe_part(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "-" for char in value)


def _key(
    exp_id: str,
    run_id: str,
    synthetic: Path,
    observed: Path | None,
    extra_paths: tuple[Path, ...] = (),
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
    }
    digest = sha256(json.dumps(key_parts, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    prefix = f"{PAYLOAD_CACHE_VERSION}__{_safe_part(exp_id)}__{_safe_part(run_id)}"
    return f"{prefix[:MAX_CACHE_KEY_PREFIX]}__{digest}.json"


def get_or_build(
    exp_id: str,
    run_id: str,
    synthetic: Path,
    observed: Path | None,
    build: Callable[[], dict[str, Any]],
    *,
    refresh: bool = False,
    extra_paths: tuple[Path, ...] = (),
) -> dict[str, Any]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / _key(exp_id, run_id, synthetic, observed, extra_paths)
    if cache_file.exists() and not refresh:
        return json.loads(cache_file.read_text())
    payload = build()
    cache_file.write_text(json.dumps(payload))
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
