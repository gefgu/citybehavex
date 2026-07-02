"""On-disk cache for comparison payloads.

Building a payload loads and processes the full observed table (millions of rows
for some cities), so results are cached as JSON keyed by the mtimes of the two
input parquets. A changed input invalidates the entry automatically.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .config import CACHE_DIR


def _key(exp_id: str, run_id: str, synthetic: Path, observed: Path) -> str:
    syn_mtime = int(synthetic.stat().st_mtime)
    obs_mtime = int(observed.stat().st_mtime)
    return f"{exp_id}__{run_id}__{syn_mtime}__{obs_mtime}.json"


def get_or_build(
    exp_id: str,
    run_id: str,
    synthetic: Path,
    observed: Path,
    build: Callable[[], dict[str, Any]],
    *,
    refresh: bool = False,
) -> dict[str, Any]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / _key(exp_id, run_id, synthetic, observed)
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
