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
        ("v3", exp_id, run.run_id),
        run.path,
        build=lambda out: _build_legs_index(run.path, out),
    )


def moving_index_path(exp_id: str, run: "Run") -> Path | None:
    """Cached, (uid, stop_id, seq)-sorted copy of the run's ``_moving.parquet``.

    Returns ``None`` when the run has no moving parquet (older runs, or runs
    with road routing disabled) — callers fall back to the plain 2-point
    straight-line leg interpolation in that case.
    """
    if not run.moving_path.exists():
        return None
    return get_or_build_parquet(
        "timeline_moving",
        ("v1", exp_id, run.run_id),
        run.moving_path,
        build=lambda out: _build_moving_index(run.moving_path, out),
    )


def _build_moving_index(moving_path: Path, out_path: Path) -> None:
    con = duckdb.connect()
    try:
        con.execute(
            f"""
            COPY (
                SELECT uid, stop_id, seq, lat, lng, t
                FROM read_parquet('{quote_path(moving_path)}')
                ORDER BY uid, stop_id, seq
            )
            TO '{quote_path(out_path)}' (FORMAT PARQUET)
            """
        )
    finally:
        con.close()


def _parquet_columns(con: duckdb.DuckDBPyConnection, path: Path) -> set[str]:
    rows = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{quote_path(path)}')"
    ).fetchall()
    return {str(row[0]) for row in rows}


def _build_legs_index(trajectory_path: Path, out_path: Path) -> None:
    con = duckdb.connect()
    try:
        columns = _parquet_columns(con, trajectory_path)
        category_expr = "category" if "category" in columns else "NULL::VARCHAR AS category"
        stop_id_expr = "stop_id" if "stop_id" in columns else "NULL::BIGINT AS stop_id"
        con.execute(
            f"""
            COPY (
                WITH ordered AS (
                    SELECT
                        uid, lat, lng, arrival, departure, trip_duration_minutes, purpose,
                        {category_expr}, {stop_id_expr},
                        LAG(lat) OVER w AS o_lat,
                        LAG(lng) OVER w AS o_lng
                    FROM read_parquet('{quote_path(trajectory_path)}')
                    WINDOW w AS (PARTITION BY uid ORDER BY arrival)
                ),
                combined AS (
                    SELECT uid, 'dwell' AS kind,
                           arrival AS t_start, departure AS t_end,
                           lat AS o_lat, lng AS o_lng, lat AS d_lat, lng AS d_lng, purpose, category,
                           NULL::BIGINT AS stop_id
                    FROM ordered
                    UNION ALL
                    SELECT uid, 'leg' AS kind,
                           arrival - (trip_duration_minutes * INTERVAL '1 minute') AS t_start,
                           arrival AS t_end,
                           o_lat, o_lng, lat AS d_lat, lng AS d_lng, purpose, category,
                           stop_id
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
    moving_path: Path | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    """Return (segments, truncated) for agents active in [since, until) and bbox.

    ``bbox`` is (min_lat, min_lng, max_lat, max_lng). Sampling is a deterministic
    hash of ``uid`` (not a plain ORDER BY uid) so repeated requests with the same
    params return the same agent subset without biasing toward low uid values.

    Known v1 approximation: the bbox test only checks leg endpoints (origin or
    destination), not true segment-rectangle intersection — a leg that clips
    through the viewport without either endpoint inside it is missed.

    When ``moving_path`` (a cached, sorted copy of the run's road-routing
    waypoints — see ``moving_index_path``) is given, "leg"-kind segments get an
    extra ``waypoints`` field so the frontend can animate movement along the
    actual road path instead of a 2-point straight-line lerp.
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
            SELECT l.uid, l.kind, l.t_start, l.t_end, l.o_lat, l.o_lng, l.d_lat, l.d_lng,
                   l.purpose, l.category, l.stop_id
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

    if moving_path is not None:
        _attach_waypoints(segments, moving_path)
    for s in segments:
        s.pop("stop_id", None)

    return segments, truncated


def _attach_waypoints(segments: list[dict[str, Any]], moving_path: Path) -> None:
    pairs = {
        (s["uid"], s["stop_id"])
        for s in segments
        if s["kind"] == "leg" and s.get("stop_id") is not None
    }
    if not pairs:
        return
    values = ", ".join(f"({uid}, {stop_id})" for uid, stop_id in pairs)
    con = duckdb.connect()
    try:
        rows = con.execute(
            f"""
            SELECT m.uid, m.stop_id, m.lat, m.lng, m.t
            FROM read_parquet('{quote_path(moving_path)}') m
            JOIN (VALUES {values}) AS requested(uid, stop_id)
              ON m.uid = requested.uid AND m.stop_id = requested.stop_id
            ORDER BY m.uid, m.stop_id, m.seq
            """
        ).fetchall()
    finally:
        con.close()

    waypoints_by_key: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for uid, stop_id, lat, lng, t in rows:
        waypoints_by_key.setdefault((uid, stop_id), []).append(
            {"lat": lat, "lng": lng, "t": t.isoformat() if hasattr(t, "isoformat") else t}
        )

    for s in segments:
        if s["kind"] == "leg" and s.get("stop_id") is not None:
            s["waypoints"] = waypoints_by_key.get((s["uid"], s["stop_id"]))


def query_agent_trips(trajectory_path: Path, uid: int) -> list[dict[str, Any]]:
    con = duckdb.connect()
    try:
        columns = _parquet_columns(con, trajectory_path)
        category_expr = "category" if "category" in columns else "NULL::VARCHAR AS category"
        activity_expr = "activity" if "activity" in columns else "NULL::BIGINT AS activity"
        rows = con.execute(
            f"""SELECT arrival, departure, lat, lng, purpose, {category_expr},
                       {activity_expr}, trip_duration_minutes, dwell_minutes
                FROM read_parquet('{quote_path(trajectory_path)}')
                WHERE uid = $uid ORDER BY arrival""",
            {"uid": uid},
        ).fetchall()
        cols = [d[0] for d in con.description]
    finally:
        con.close()
    return [dict(zip(cols, r)) for r in rows]


def query_agent_encounters(
    encounters_path: Path,
    trajectory_path: Path,
    uid: int,
    limit: int = 20,
) -> list[dict[str, Any]]:
    con = duckdb.connect()
    try:
        columns = _parquet_columns(con, trajectory_path)
        category_expr = "t.category" if "category" in columns else "NULL::VARCHAR AS category"
        activity_expr = "t.activity" if "activity" in columns else "NULL::BIGINT AS activity"
        rows = con.execute(
            f"""
                WITH recent AS (
                    SELECT CASE WHEN agent = $uid THEN contact ELSE agent END AS contact_uid,
                           to_timestamp(ts)::TIMESTAMP AS ts, tile
                    FROM read_parquet('{quote_path(encounters_path)}')
                    WHERE agent = $uid OR contact = $uid
                    ORDER BY ts DESC LIMIT $limit
                ),
                matched AS (
                    SELECT recent.contact_uid, recent.ts, recent.tile,
                           t.arrival AS stop_arrival, t.departure AS stop_departure,
                           t.lat, t.lng, t.purpose, {category_expr}, {activity_expr},
                           t.trip_duration_minutes, t.dwell_minutes,
                           row_number() OVER (
                               PARTITION BY recent.contact_uid, recent.ts, recent.tile
                               ORDER BY t.arrival DESC NULLS LAST
                           ) AS rn
                    FROM recent
                    LEFT JOIN read_parquet('{quote_path(trajectory_path)}') t
                      ON t.uid = recent.contact_uid
                     AND t.arrival <= recent.ts
                     AND t.departure >= recent.ts
                )
                SELECT contact_uid, ts, tile, stop_arrival, stop_departure,
                       lat, lng, purpose, category, activity,
                       trip_duration_minutes, dwell_minutes
                FROM matched
                WHERE rn = 1
                ORDER BY ts DESC
            """,
            {"uid": uid, "limit": limit},
        ).fetchall()
        cols = [d[0] for d in con.description]
    finally:
        con.close()
    return [dict(zip(cols, r)) for r in rows]
