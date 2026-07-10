"""Tier-1 (per-file) feature extraction, cached independently of which
comparison a file is used in.

Building a comparison payload combines a synthetic and an observed
trajectory, but most of the expensive work (reading the full trajectory,
computing jump lengths / radius of gyration, deriving the per-user visit
table used by the activity/profile/motif sections) only depends on ONE of
the two files. Caching that per-file work here means a later payload build
that reuses one side unchanged (comparing the same synthetic run against a
different observed dataset, or vice versa) only pays for reading the cached
result plus the cheap cross-file comparison math in ``payload.py``, instead
of re-reading and reprocessing the full raw trajectory again.

Two storage shapes, matching what each result naturally is:
- tabular results (the activity/profile visits table) -> parquet, via the
  existing ``cache.get_or_build_parquet``.
- small per-filter scalar/array bundles (jump lengths, radius of gyration)
  -> a JSON blob next to a parquet-shaped cache key, since forcing a ragged
  per-user array into a parquet schema buys nothing here.
"""

from __future__ import annotations

import json
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Any, Optional

import numpy as np
import polars as pl
import fkmob

from citybehavex.metrics import (
    build_road_network_handle,
    jump_lengths_km as road_jump_lengths_km,
    radius_of_gyration_km as road_radius_of_gyration_km,
)

from .cache import CACHE_DIR, get_or_build_parquet
from .filters import _filter_df
from .reports_bridge import (
    ActivityVisitsResult,
    _adapt_evaluation_dataframe,
    _prepare_activity_visits,
)

FEATURES_CACHE_VERSION = "f2"


def _filters_signature(filters: list[dict[str, Any]]) -> str:
    return sha256(json.dumps(filters, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def _jsonable_config(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        return _jsonable_config(value.model_dump())
    if isinstance(value, dict):
        return {str(k): _jsonable_config(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable_config(v) for v in value]
    if isinstance(value, set):
        return sorted(_jsonable_config(v) for v in value)
    return repr(value)


def _adaptation_signature(config: Optional[object]) -> str:
    return sha256(json.dumps(_jsonable_config(config), sort_keys=True).encode("utf-8")).hexdigest()[:12]


def _path_signature(path: Path) -> str:
    """``get_or_build_parquet``/``_get_or_build_json`` key on ``key_parts +
    mtime`` only -- two different files that happen to share an mtime (very
    plausible for short-lived fixtures/tests, or two experiments' outputs
    written in the same second) would otherwise collide on the same cache
    entry. Fold in a hash of the resolved path so they can't.
    """
    return sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:16]


def _get_or_build_json(
    cache_name: str,
    key_parts: tuple[str, ...],
    input_path: Path,
    build,
) -> dict[str, Any]:
    """Like ``cache.get_or_build_parquet`` but for a JSON-shaped tier-1
    artifact keyed by a single input's mtime, for results too small/ragged
    to be worth a parquet schema."""
    subdir = CACHE_DIR / cache_name
    subdir.mkdir(parents=True, exist_ok=True)
    mtime = int(input_path.stat().st_mtime)
    out = subdir / (f"{'__'.join(key_parts)}__{mtime}.json")
    if out.exists():
        return json.loads(out.read_text())
    result = build()
    out.write_text(json.dumps(result))
    return result


@lru_cache(maxsize=8)
def _cached_road_handle(nodes_path: str, nodes_mtime: int, edges_path: str, edges_mtime: int):
    nodes_df = pl.read_parquet(nodes_path)
    edges_df = pl.read_parquet(edges_path)
    if not len(nodes_df) or not len(edges_df):
        return None, None
    handle = build_road_network_handle(edges_df)
    return handle, nodes_df


def get_road_handle(
    nodes_path: Optional[Path], edges_path: Optional[Path]
) -> tuple[Any, Optional[pl.DataFrame]]:
    """In-process cache of the prepared road-network contraction hierarchy.

    The handle is a live Rust object, not disk-serializable -- preparing the
    contraction hierarchy (not querying it) is the expensive step, so this
    is cached for the lifetime of whichever process built it and reused
    across every payload build that references the same road graph, instead
    of re-preparing it per request.
    """
    if not nodes_path or not edges_path or not nodes_path.exists() or not edges_path.exists():
        return None, None
    return _cached_road_handle(
        str(nodes_path),
        int(nodes_path.stat().st_mtime),
        str(edges_path),
        int(edges_path.stat().st_mtime),
    )


def get_jumps_rog(
    traj_path: Path,
    *,
    uid_col: str,
    lat_col: str,
    lng_col: str,
    datetime_col: str,
    filters: list[dict[str, Any]],
    road_nodes_path: Optional[Path] = None,
    road_edges_path: Optional[Path] = None,
    road_snap_max_distance_m: float = 750.0,
    evaluation_adaptation_config: Optional[object] = None,
    label: str = "trajectory",
) -> dict[str, dict[str, list[float]]]:
    """Per-filter jump-length / radius-of-gyration arrays for one
    trajectory file -- the single most expensive, most duplicated
    computation in the comparison pipeline (previously computed once for the
    ECDF section and again, via a different, non-road-aware code path, for
    the mobility-laws section; both now read this one cached result).

    Road-aware when a road graph is supplied, matching
    ``road_jump_lengths_km``/``road_radius_of_gyration_km``'s behavior --
    the cache key folds in the road graph's own mtimes (not just this
    file's), so a road-graph update invalidates it too.
    """
    road_handle, road_nodes_df = get_road_handle(road_nodes_path, road_edges_path)
    if road_handle is not None:
        road_sig = (
            f"road-{int(road_nodes_path.stat().st_mtime)}"
            f"-{int(road_edges_path.stat().st_mtime)}-{road_snap_max_distance_m:.0f}"
        )
    else:
        road_sig = "no-road"
    key_parts = (
        FEATURES_CACHE_VERSION,
        "jumps_rog",
        _path_signature(traj_path),
        _filters_signature(filters),
        road_sig,
        _adaptation_signature(evaluation_adaptation_config),
    )

    def build() -> dict[str, dict[str, list[float]]]:
        df = pl.read_parquet(traj_path)
        result: dict[str, dict[str, list[float]]] = {}
        for meta in filters:
            filtered = _filter_df(df, datetime_col, meta)
            if filtered.is_empty():
                result[meta["key"]] = {"jumps": [], "rog": []}
                continue
            adapted = _adapt_evaluation_dataframe(
                filtered,
                label=label,
                uid_col=uid_col,
                datetime_col=datetime_col,
                lat_col=lat_col,
                lng_col=lng_col,
                config=evaluation_adaptation_config,
            )
            filtered = adapted.df
            if road_handle is not None:
                jumps = road_jump_lengths_km(
                    filtered,
                    uid_col=uid_col,
                    lat_col=lat_col,
                    lng_col=lng_col,
                    datetime_col=datetime_col,
                    handle=road_handle,
                    nodes_df=road_nodes_df,
                    snap_max_distance_m=road_snap_max_distance_m,
                )
                rog = road_radius_of_gyration_km(
                    filtered,
                    uid_col=uid_col,
                    lat_col=lat_col,
                    lng_col=lng_col,
                    handle=road_handle,
                    nodes_df=road_nodes_df,
                    snap_max_distance_m=road_snap_max_distance_m,
                )["radius_of_gyration"].to_numpy()
            else:
                tr = fkmob.TrajDataFrame(
                    filtered,
                    datetime_col=datetime_col,
                    lat_col=lat_col,
                    lng_col=lng_col,
                    uid_col=uid_col,
                )
                # jump_lengths(merge=True) returns "a backend-appropriate
                # array object" per fkmob's own docs -- for a polars-backed
                # TrajDataFrame that's an Arrow-backed array whose elements
                # are pyarrow scalars, not plain floats.
                jumps = np.asarray(tr.jump_lengths(merge=True), dtype=float)
                rog = tr.radius_of_gyration()["radius_of_gyration"].to_numpy()
            # Zero-length "jumps" between consecutive same-location rows
            # (e.g. repeat check-ins, more common after coordinate rounding)
            # aren't movement -- exclude so the ECDF reflects actual trips.
            jumps = np.asarray(jumps, dtype=float)
            jumps = jumps[jumps > 0]
            result[meta["key"]] = {
                "jumps": np.asarray(jumps, dtype=float).tolist(),
                "rog": np.asarray(rog, dtype=float).tolist(),
            }
        return result

    return _get_or_build_json("features_jumps_rog", key_parts, traj_path, build)


def get_activity_visits(
    traj_path: Path,
    *,
    label: str,
    uid_col: Optional[str],
    datetime_col: Optional[str],
    activity_col: Optional[str],
    location_col: Optional[str],
    lat_col: Optional[str],
    lng_col: Optional[str],
    location_resolution: int = 10,
    end_col: Optional[str] = None,
) -> Optional[ActivityVisitsResult]:
    """Cached per-file activity-visits table (uid/start_timestamp/purpose/
    location_id), the input every activity/profile/motif section builds on.
    Deriving it requires a full pass over the raw trajectory (H3 binning,
    purpose derivation) -- the most expensive part of those sections after
    jumps/RoG, so it's cached the same way, keyed on this file's mtime plus
    the location resolution (which, for the synthetic side, is actually
    detected from the *observed* file -- see ``_location_resolution`` in
    ``citybehavex.reports.comparison`` -- so it must be part of the key, not
    just this file's own mtime).
    """
    if uid_col is None or datetime_col is None:
        return None
    if location_col is None and (lat_col is None or lng_col is None):
        return None

    key_parts = (
        FEATURES_CACHE_VERSION,
        "activity_visits",
        _path_signature(traj_path),
        f"res{location_resolution}",
        f"loc{location_col or 'none'}",
        f"act{activity_col or 'none'}",
    )

    def build(out: Path) -> None:
        df = pl.read_parquet(traj_path)
        resolved_activity_col = activity_col if activity_col and activity_col in df.columns else None
        result = _prepare_activity_visits(
            df,
            label=label,
            uid_col=uid_col,
            datetime_col=datetime_col,
            activity_col=resolved_activity_col,
            location_col=location_col,
            lat_col=lat_col,
            lng_col=lng_col,
            location_resolution=location_resolution,
            end_col=end_col,
        )
        sidecar = out.with_suffix(".json")
        if result is None:
            out.write_bytes(b"")
            sidecar.write_text(json.dumps({"empty": True, "used_heuristic": False, "warning": None}))
            return
        result.visits.write_parquet(out)
        sidecar.write_text(
            json.dumps(
                {"empty": False, "used_heuristic": result.used_heuristic, "warning": result.warning}
            )
        )

    out_path = get_or_build_parquet("features_activity_visits", key_parts, traj_path, build)
    sidecar = json.loads(out_path.with_suffix(".json").read_text())
    if sidecar["empty"]:
        return None
    return ActivityVisitsResult(
        pl.read_parquet(out_path), sidecar["used_heuristic"], sidecar["warning"]
    )
