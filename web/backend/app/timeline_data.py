"""DuckDB query layer for the timeline view.

Trajectory parquets are stop tables (one row per activity episode), not
continuous GPS: ``uid, datetime, lat, lng, arrival, departure,
trip_duration_minutes, dwell_minutes, activity, purpose``. For agent ``uid``
sorted by ``arrival``, the leg from the previous stop to a row travels in a
straight line during ``[arrival - trip_duration_minutes*60s, arrival]``, then
the agent dwells at that row's location during ``[arrival, departure]``.

Rendering a live, viewport/time-filtered view of this directly against the raw
table would mean re-computing "each row's previous stop" (a ``LAG()`` window
function over the whole table) on every request — cheap for gparis (500
agents) but not for yjmob (100k agents / ~29.5M rows). So the window function
is run once per run and cached as a derived, time-sorted parquet (the "legs
index"); all live per-request queries filter that narrower artifact instead.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb

from .cache import get_or_build_parquet
from .config import CACHE_DIR
from .datasource import quote_path

if TYPE_CHECKING:
    from .experiments import Run


def legs_index_path(exp_id: str, run: "Run") -> Path:
    return get_or_build_parquet(
        "timeline_legs",
        (exp_id, run.run_id),
        run.path,
        build=lambda out: _build_legs_index(run.path, out),
    )


def _build_legs_index(trajectory_path: Path, out_path: Path) -> None:
    con = duckdb.connect()
    try:
        con.execute(
            f"""
            COPY (
                WITH ordered AS (
                    SELECT
                        uid, lat, lng, arrival, departure, trip_duration_minutes, purpose,
                        LAG(lat) OVER w AS o_lat,
                        LAG(lng) OVER w AS o_lng
                    FROM read_parquet('{quote_path(trajectory_path)}')
                    WINDOW w AS (PARTITION BY uid ORDER BY arrival)
                ),
                combined AS (
                    SELECT uid, 'dwell' AS kind,
                           arrival AS t_start, departure AS t_end,
                           lat AS o_lat, lng AS o_lng, lat AS d_lat, lng AS d_lng, purpose
                    FROM ordered
                    UNION ALL
                    SELECT uid, 'leg' AS kind,
                           arrival - (trip_duration_minutes * INTERVAL '1 minute') AS t_start,
                           arrival AS t_end,
                           o_lat, o_lng, lat AS d_lat, lng AS d_lng, purpose
                    FROM ordered
                    WHERE o_lat IS NOT NULL
                )
                SELECT * FROM combined ORDER BY t_start
            )
            TO '{quote_path(out_path)}' (FORMAT PARQUET)
            """
        )
    finally:
        con.close()


def run_bbox(exp_id: str, run: "Run") -> dict[str, float] | None:
    """Cached (mtime-keyed) min/max lat/lng across the whole run."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    subdir = CACHE_DIR / "timeline_bbox"
    subdir.mkdir(parents=True, exist_ok=True)
    mtime = int(run.path.stat().st_mtime)
    out = subdir / f"{exp_id}__{run.run_id}__{mtime}.json"
    if out.exists():
        import json

        return json.loads(out.read_text())

    con = duckdb.connect()
    try:
        row = con.execute(
            f"""SELECT min(lat), max(lat), min(lng), max(lng)
                FROM read_parquet('{quote_path(run.path)}')"""
        ).fetchone()
    finally:
        con.close()
    if row is None or row[0] is None:
        return None
    bbox = {"min_lat": row[0], "max_lat": row[1], "min_lng": row[2], "max_lng": row[3]}
    import json

    out.write_text(json.dumps(bbox))
    return bbox


def query_active_legs(
    legs_path: Path,
    since: datetime,
    until: datetime,
    bbox: tuple[float, float, float, float],
    max_agents: int,
) -> tuple[list[dict[str, Any]], bool]:
    """Return (segments, truncated) for agents active in [since, until) and bbox.

    ``bbox`` is (min_lat, min_lng, max_lat, max_lng). Sampling is a deterministic
    hash of ``uid`` (not a plain ORDER BY uid) so repeated requests with the same
    params return the same agent subset without biasing toward low uid values.

    Known v1 approximation: the bbox test only checks leg endpoints (origin or
    destination), not true segment-rectangle intersection — a leg that clips
    through the viewport without either endpoint inside it is missed.
    """
    min_lat, min_lng, max_lat, max_lng = bbox
    con = duckdb.connect()
    try:
        sql = f"""
            WITH candidates AS (
                SELECT DISTINCT uid
                FROM read_parquet('{quote_path(legs_path)}')
                WHERE t_start <= $until AND t_end >= $since
                  AND (
                        (d_lat BETWEEN $min_lat AND $max_lat AND d_lng BETWEEN $min_lng AND $max_lng)
                     OR (o_lat BETWEEN $min_lat AND $max_lat AND o_lng BETWEEN $min_lng AND $max_lng)
                  )
                ORDER BY hash(uid)
                LIMIT $max_agents
            )
            SELECT l.uid, l.kind, l.t_start, l.t_end, l.o_lat, l.o_lng, l.d_lat, l.d_lng, l.purpose
            FROM read_parquet('{quote_path(legs_path)}') l
            JOIN candidates c USING (uid)
            WHERE l.t_start <= $until AND l.t_end >= $since
            ORDER BY l.uid, l.t_start
        """
        rows = con.execute(
            sql,
            {
                "since": since,
                "until": until,
                "min_lat": min_lat,
                "max_lat": max_lat,
                "min_lng": min_lng,
                "max_lng": max_lng,
                "max_agents": max_agents,
            },
        ).fetchall()
        cols = [d[0] for d in con.description]
    finally:
        con.close()
    segments = [dict(zip(cols, r)) for r in rows]
    distinct_uids = {s["uid"] for s in segments}
    truncated = len(distinct_uids) >= max_agents
    return segments, truncated


def query_agent_trips(trajectory_path: Path, uid: int) -> list[dict[str, Any]]:
    con = duckdb.connect()
    try:
        rows = con.execute(
            f"""SELECT arrival, departure, lat, lng, purpose,
                       trip_duration_minutes, dwell_minutes
                FROM read_parquet('{quote_path(trajectory_path)}')
                WHERE uid = $uid ORDER BY arrival""",
            {"uid": uid},
        ).fetchall()
        cols = [d[0] for d in con.description]
    finally:
        con.close()
    return [dict(zip(cols, r)) for r in rows]


def query_agent_encounters(encounters_path: Path, uid: int, limit: int = 20) -> list[dict[str, Any]]:
    con = duckdb.connect()
    try:
        rows = con.execute(
            f"""SELECT CASE WHEN agent = $uid THEN contact ELSE agent END AS contact_uid,
                       to_timestamp(ts)::TIMESTAMP AS ts, tile
                FROM read_parquet('{quote_path(encounters_path)}')
                WHERE agent = $uid OR contact = $uid
                ORDER BY ts DESC LIMIT $limit""",
            {"uid": uid, "limit": limit},
        ).fetchall()
        cols = [d[0] for d in con.description]
    finally:
        con.close()
    return [dict(zip(cols, r)) for r in rows]
