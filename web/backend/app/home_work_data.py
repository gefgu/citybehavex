"""On-demand home/work agent-location density maps.

Unlike STVD (a time-varying volume metric computed once per day-type filter
and cached in the main ``/charts`` payload), a home/work map is a per-agent
static property: each agent has exactly one home and one work location. The
interesting filter dimension here is agent demographics (gender/age/job),
which has too many combinations to precompute — so this module computes
density on demand with DuckDB (which already has the ``h3`` community
extension loaded elsewhere in this repo, see
``citybehavex.tessellation.builder``) instead of going through the
pandas/fkmob ``citybehavex.reports.comparison`` pipeline.

For each agent we take the *modal* fine-grained H3 cell (resolution 12)
among their HOME- or WORK-tagged rows — collapsing "nights slept at home" or
"days worked" down to a single representative point — then re-bucket that
point into the display resolutions (7 and 9, matching STVD's
``resolutions=[7, 9]``) and count agents per cell.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import duckdb
import h3
import numpy as np

from citybehavex.profiles import ILOSTAT_JOBS

from .datasource import quote_path

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
AGE_BRACKETS: list[dict[str, Any]] = [
    {"key": "16_24", "label": "16–24", "min": 16, "max": 24},
    {"key": "25_34", "label": "25–34", "min": 25, "max": 34},
    {"key": "35_44", "label": "35–44", "min": 35, "max": 44},
    {"key": "45_59", "label": "45–59", "min": 45, "max": 59},
    {"key": "60_80", "label": "60–80", "min": 60, "max": 80},
]
GENDERS: list[str] = ["male", "female"]
JOBS: list[str] = list(ILOSTAT_JOBS)

DISPLAY_RESOLUTIONS: tuple[int, ...] = (7, 9)
_FINE_RESOLUTION = 12

# Sequential 5-class palettes (ColorBrewer Blues/Oranges), echoing the
# blue=synthetic / red=observed convention already used by STVD_COLORS.
SEQ_BLUES = ["#eff3ff", "#bdd7e7", "#6baed6", "#3182bd", "#08519c"]
SEQ_ORANGES = ["#feedde", "#fdbe85", "#fd8d3c", "#e6550d", "#a63603"]

_UID_CANDIDATES = ["uid", "user_id", "user", "agent_id", "userid"]
_LAT_CANDIDATES = ["lat", "latitude"]
_LNG_CANDIDATES = ["lng", "lon", "longitude", "long"]
_DATETIME_CANDIDATES = [
    "datetime", "start_timestamp", "timestamp", "check-in_time",
    "start_time", "_start_time", "checkin_time", "time", "date",
]
_ACTIVITY_CANDIDATES = ["purpose", "activity", "act", "location_type", "category", "purpose_d"]


class _Cols:
    def __init__(self, names: list[str]):
        self.columns = names


def _detect(columns: list[str], candidates: list[str]) -> Optional[str]:
    lower = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def resolve_age_bracket(bracket_key: Optional[str]) -> tuple[Optional[int], Optional[int]]:
    if not bracket_key or bracket_key == "all":
        return None, None
    for bracket in AGE_BRACKETS:
        if bracket["key"] == bracket_key:
            return int(bracket["min"]), int(bracket["max"])
    raise ValueError(f"unknown age bracket: {bracket_key!r}")


@dataclass(frozen=True)
class DemoFilter:
    gender: Optional[str] = None
    age_min: Optional[int] = None
    age_max: Optional[int] = None
    job: Optional[str] = None

    def is_empty(self) -> bool:
        return self.gender is None and self.age_min is None and self.age_max is None and self.job is None


def _parquet_columns(con: duckdb.DuckDBPyConnection, path: Path) -> list[str]:
    rows = con.execute(f"SELECT name FROM parquet_schema('{quote_path(path)}')").fetchall()
    return [r[0] for r in rows if r[0] not in {"schema", "duckdb_schema"}]


def _detect_cols(con: duckdb.DuckDBPyConnection, path: Path) -> dict[str, Optional[str]]:
    columns = _parquet_columns(con, path)
    return {
        "uid": _detect(columns, _UID_CANDIDATES),
        "lat": _detect(columns, _LAT_CANDIDATES),
        "lng": _detect(columns, _LNG_CANDIDATES),
        "datetime": _detect(columns, _DATETIME_CANDIDATES),
        "purpose": _detect(columns, _ACTIVITY_CANDIDATES),
    }


def _load_h3(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("INSTALL h3 FROM community; LOAD h3;")


# --------------------------------------------------------------------------- #
# synthetic density (explicit purpose column, optional demographic filter)
# --------------------------------------------------------------------------- #
def _filter_clause(demo: DemoFilter) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if demo.gender is not None:
        clauses.append("gender = ?")
        params.append(demo.gender)
    if demo.age_min is not None:
        clauses.append("age >= ?")
        params.append(demo.age_min)
    if demo.age_max is not None:
        clauses.append("age <= ?")
        params.append(demo.age_max)
    if demo.job is not None:
        clauses.append("job = ?")
        params.append(demo.job)
    where = " AND ".join(clauses) if clauses else "TRUE"
    return where, params


def _agent_density(
    con: duckdb.DuckDBPyConnection,
    traj_path: Path,
    cols: dict[str, Optional[str]],
    purpose: str,
    resolution: int,
    *,
    profiles_path: Optional[Path] = None,
    demo: Optional[DemoFilter] = None,
) -> list[tuple[str, int]]:
    """Per-purpose agent density at ``resolution``, using an explicit purpose column."""
    uid, lat, lng, purpose_col = cols["uid"], cols["lat"], cols["lng"], cols["purpose"]
    if not (uid and lat and lng and purpose_col):
        return []

    semi_join = ""
    params: list[Any] = [purpose]
    if profiles_path is not None and demo is not None and not demo.is_empty():
        where, filt_params = _filter_clause(demo)
        semi_join = f"""
            SEMI JOIN (
                SELECT uid FROM read_parquet('{quote_path(profiles_path)}') WHERE {where}
            ) filt ON filt.uid = t."{uid}"
        """
        params = filt_params + params

    query = f"""
        WITH rows AS (
            SELECT t."{uid}" AS uid, t."{lat}" AS lat, t."{lng}" AS lng
            FROM read_parquet('{quote_path(traj_path)}') t
            {semi_join}
            WHERE upper(trim(CAST(t."{purpose_col}" AS VARCHAR))) = upper(?)
        ),
        per_agent AS (
            SELECT
                uid,
                h3_latlng_to_cell_string(lat, lng, {_FINE_RESOLUTION}) AS fine_cell,
                any_value(lat) AS lat, any_value(lng) AS lng, count(*) AS cnt
            FROM rows
            GROUP BY uid, fine_cell
        ),
        modal AS (
            SELECT uid, lat, lng,
                   row_number() OVER (PARTITION BY uid ORDER BY cnt DESC, fine_cell) AS rn
            FROM per_agent
        )
        SELECT
            h3_latlng_to_cell_string(lat, lng, {resolution}) AS cell,
            count(*) AS agent_count
        FROM modal
        WHERE rn = 1
        GROUP BY cell
    """
    return con.execute(query, params).fetchall()


def _matched_agent_counts(
    con: duckdb.DuckDBPyConnection,
    traj_path: Path,
    cols: dict[str, Optional[str]],
    profiles_path: Optional[Path],
    demo: DemoFilter,
) -> tuple[int, int]:
    uid, purpose_col = cols["uid"], cols["purpose"]
    if not (uid and purpose_col):
        return 0, 0
    total = con.execute(
        f"""
        SELECT count(DISTINCT "{uid}")
        FROM read_parquet('{quote_path(traj_path)}')
        WHERE upper(trim(CAST("{purpose_col}" AS VARCHAR))) = 'HOME'
        """
    ).fetchone()[0]
    if profiles_path is None or demo.is_empty():
        return int(total), int(total)
    where, params = _filter_clause(demo)
    matched = con.execute(
        f"""
        SELECT count(DISTINCT t."{uid}")
        FROM read_parquet('{quote_path(traj_path)}') t
        SEMI JOIN (
            SELECT uid FROM read_parquet('{quote_path(profiles_path)}') WHERE {where}
        ) filt ON filt.uid = t."{uid}"
        WHERE upper(trim(CAST(t."{purpose_col}" AS VARCHAR))) = 'HOME'
        """,
        params,
    ).fetchone()[0]
    return int(matched), int(total)


# --------------------------------------------------------------------------- #
# observed density (explicit purpose column, else hour-of-day heuristic)
# --------------------------------------------------------------------------- #
def _observed_density_explicit(
    con: duckdb.DuckDBPyConnection,
    obs_path: Path,
    cols: dict[str, Optional[str]],
    purpose: str,
    resolution: int,
) -> list[tuple[str, int]]:
    return _agent_density(con, obs_path, cols, purpose, resolution)


def _observed_density_heuristic(
    con: duckdb.DuckDBPyConnection,
    obs_path: Path,
    cols: dict[str, Optional[str]],
    purpose: str,
    resolution: int,
) -> list[tuple[str, int]]:
    """Mirrors ``_derive_purpose_groups_from_heuristic`` in
    ``citybehavex.reports.comparison`` (home: hour 2-5; work: hour=10 or
    14-16, excluding the agent's home cell) but keeps lat/lng throughout so
    the result can be H3-bucketed at arbitrary display resolutions."""
    uid, lat, lng, dt = cols["uid"], cols["lat"], cols["lng"], cols["datetime"]
    if not (uid and lat and lng and dt):
        return []

    query = f"""
        WITH rows AS (
            SELECT "{uid}" AS uid, "{lat}" AS lat, "{lng}" AS lng,
                   extract('hour' FROM CAST("{dt}" AS TIMESTAMP)) AS hour
            FROM read_parquet('{quote_path(obs_path)}')
        ),
        fine AS (
            SELECT uid, lat, lng, hour,
                   h3_latlng_to_cell_string(lat, lng, {_FINE_RESOLUTION}) AS fine_cell
            FROM rows
        ),
        home_candidates AS (
            SELECT uid, fine_cell, any_value(lat) AS lat, any_value(lng) AS lng, count(*) AS cnt
            FROM fine WHERE hour BETWEEN 2 AND 5
            GROUP BY uid, fine_cell
        ),
        home_modal AS (
            SELECT uid, lat, lng, fine_cell,
                   row_number() OVER (PARTITION BY uid ORDER BY cnt DESC, fine_cell) AS rn
            FROM home_candidates
        ),
        home AS (SELECT uid, lat, lng, fine_cell FROM home_modal WHERE rn = 1),
        work_candidates AS (
            SELECT f.uid, f.fine_cell, any_value(f.lat) AS lat, any_value(f.lng) AS lng, count(*) AS cnt
            FROM fine f
            LEFT JOIN home h ON h.uid = f.uid
            WHERE (f.hour = 10 OR f.hour BETWEEN 14 AND 16)
              AND (h.fine_cell IS NULL OR f.fine_cell != h.fine_cell)
            GROUP BY f.uid, f.fine_cell
        ),
        work_modal AS (
            SELECT uid, lat, lng,
                   row_number() OVER (PARTITION BY uid ORDER BY cnt DESC, fine_cell) AS rn
            FROM work_candidates
        ),
        work AS (SELECT uid, lat, lng FROM work_modal WHERE rn = 1),
        chosen AS (
            SELECT uid, lat, lng FROM home WHERE ? = 'HOME'
            UNION ALL
            SELECT uid, lat, lng FROM work WHERE ? = 'WORK'
        )
        SELECT h3_latlng_to_cell_string(lat, lng, {resolution}) AS cell, count(*) AS agent_count
        FROM chosen
        GROUP BY cell
    """
    return con.execute(query, [purpose, purpose]).fetchall()


def _observed_density(
    con: duckdb.DuckDBPyConnection,
    obs_path: Path,
    cols: dict[str, Optional[str]],
    purpose: str,
    resolution: int,
) -> list[tuple[str, int]]:
    if cols["purpose"]:
        return _observed_density_explicit(con, obs_path, cols, purpose, resolution)
    return _observed_density_heuristic(con, obs_path, cols, purpose, resolution)


# --------------------------------------------------------------------------- #
# GeoJSON + color baking
# --------------------------------------------------------------------------- #
def _feature_collection(rows: list[tuple[str, int]]) -> dict[str, Any]:
    features = []
    for cell, count in rows:
        boundary = h3.cell_to_boundary(cell)
        ring = [[lng, lat] for lat, lng in boundary]
        ring.append(ring[0])
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": {"area": cell, "agent_count": int(count)},
        })
    return {"type": "FeatureCollection", "features": features}


def _annotate_panel(layers: dict[int, dict[str, Any]], ramp: list[str]) -> dict[str, Any]:
    all_counts = [
        f["properties"]["agent_count"]
        for fc in layers.values()
        for f in fc["features"]
        if f["properties"]["agent_count"] > 0
    ]
    total_agents = max((sum(f["properties"]["agent_count"] for f in fc["features"]) for fc in layers.values()), default=0)
    breaks = np.quantile(all_counts, [0.2, 0.4, 0.6, 0.8]).tolist() if all_counts else [0, 0, 0, 0]

    lngs: list[float] = []
    lats: list[float] = []
    out_layers: dict[str, Any] = {}
    for res, fc in layers.items():
        for feature in fc["features"]:
            count = feature["properties"]["agent_count"]
            cls = int(np.searchsorted(breaks, count))
            feature["properties"]["color"] = ramp[min(cls, len(ramp) - 1)]
            feature["properties"]["class"] = cls
            feature["properties"]["agent_pct"] = round(count / total_agents * 100.0, 4) if total_agents else 0.0
            ring = feature["geometry"]["coordinates"][0]
            for lng, lat in ring:
                lngs.append(lng)
                lats.append(lat)
        out_layers[str(res)] = fc

    center = [(min(lngs) + max(lngs)) / 2, (min(lats) + max(lats)) / 2] if lngs and lats else None
    return {
        "center": center,
        "layers": out_layers,
        "colors": ramp,
        "breaks": [round(float(b), 4) for b in breaks],
        "total_agents": int(total_agents),
    }


# --------------------------------------------------------------------------- #
# top-level entry point
# --------------------------------------------------------------------------- #
def build_home_work(
    synthetic_path: Path,
    observed_path: Optional[Path],
    profiles_path: Optional[Path],
    demo: DemoFilter,
) -> dict[str, Any]:
    con = duckdb.connect()
    try:
        _load_h3(con)
        synth_cols = _detect_cols(con, synthetic_path)
        has_profiles = profiles_path is not None and profiles_path.exists()
        effective_profiles = profiles_path if has_profiles else None

        obs_cols: Optional[dict[str, Optional[str]]] = None
        mode = "synthetic_only"
        if observed_path is not None and observed_path.exists():
            obs_cols = _detect_cols(con, observed_path)
            mode = "comparison"

        matched_agents, total_synthetic_agents = _matched_agent_counts(
            con, synthetic_path, synth_cols, effective_profiles, demo
        )

        result: dict[str, Any] = {}
        for purpose, key in (("HOME", "home"), ("WORK", "work")):
            synth_layers = {
                res: _feature_collection(
                    _agent_density(
                        con, synthetic_path, synth_cols, purpose, res,
                        profiles_path=effective_profiles, demo=demo,
                    )
                )
                for res in DISPLAY_RESOLUTIONS
            }
            synthetic_panel = _annotate_panel(synth_layers, SEQ_BLUES)

            real_panel = None
            if obs_cols is not None:
                real_layers = {
                    res: _feature_collection(_observed_density(con, observed_path, obs_cols, purpose, res))
                    for res in DISPLAY_RESOLUTIONS
                }
                real_panel = _annotate_panel(real_layers, SEQ_ORANGES)

            result[key] = {"synthetic": synthetic_panel, "real": real_panel}

        return {
            "mode": mode,
            "has_profiles": has_profiles,
            "matched_agents": matched_agents,
            "total_synthetic_agents": total_synthetic_agents,
            "filter": {
                "gender": demo.gender,
                "age_bracket": _bracket_key_for(demo.age_min, demo.age_max),
                "job": demo.job,
            },
            "filter_options": {
                "genders": GENDERS,
                "age_brackets": AGE_BRACKETS,
                "jobs": JOBS,
            },
            **result,
            "warnings": [],
        }
    finally:
        con.close()


def _bracket_key_for(age_min: Optional[int], age_max: Optional[int]) -> Optional[str]:
    if age_min is None and age_max is None:
        return None
    for bracket in AGE_BRACKETS:
        if bracket["min"] == age_min and bracket["max"] == age_max:
            return bracket["key"]
    return None


async def get_or_build_home_work(
    synthetic_path: Path,
    observed_path: Optional[Path],
    profiles_path: Optional[Path],
    demo: DemoFilter,
    *,
    exp_id: str,
    run_id: str,
    refresh: bool = False,
) -> dict[str, Any]:
    from .cache import get_or_build
    from .executor import get_executor

    return await get_or_build(
        f"{exp_id}__home_work",
        run_id,
        synthetic_path,
        observed_path,
        build_fn=build_home_work,
        build_kwargs=dict(
            synthetic_path=synthetic_path,
            observed_path=observed_path,
            profiles_path=profiles_path,
            demo=demo,
        ),
        executor=get_executor(),
        refresh=refresh,
        extra_paths=(profiles_path,) if profiles_path else (),
        extra_key={
            "demo": {
                "gender": demo.gender,
                "age_min": demo.age_min,
                "age_max": demo.age_max,
                "job": demo.job,
            }
        },
    )
