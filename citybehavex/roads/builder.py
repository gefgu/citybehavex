"""Build a routable car road graph from Overture Maps transportation data.

Fetches ``theme=transportation/type=segment`` (``subtype='road'``) for a bbox,
splits each segment's geometry into pieces between consecutive connectors
(the routing decision points along the segment), and turns those pieces into
directed graph edges weighted by travel time at 80% of the road's speed limit
(falling back to a class-based default speed when no speed limit is present).

Connector coordinates are derived by linearly interpolating each segment's own
geometry at the connector's ``at`` fraction, so a separate ``type=connector``
fetch isn't needed just to place nodes.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from sklearn.neighbors import NearestNeighbors

from .speeds import CAR_SPEED_FACTOR, DEFAULT_SPEED_KMH_BY_CLASS, DRIVABLE_CLASSES

_MPH_TO_KMH = 1.609344
_EARTH_RADIUS_M = 6371000.0


def haversine_m(lat1: np.ndarray, lng1: np.ndarray, lat2: np.ndarray, lng2: np.ndarray) -> np.ndarray:
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lng2 - lng1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * _EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _is_missing_or_empty(value) -> bool:
    """True for None/NaN/pd.NA and for empty lists/arrays.

    DuckDB NULL LIST columns can surface as ``None``, ``float('nan')``, or
    ``pd.NA`` depending on the pandas conversion path, none of which are
    safely truthy-checkable directly (``bool(pd.NA)`` raises).
    """
    if value is None:
        return True
    if not isinstance(value, (list, tuple)):
        return True  # a scalar missing-value sentinel (NaN/pd.NA), not a real list
    return len(value) == 0


def _speed_kmh_for_pair(speed_limits, road_class: str, from_at: float, to_at: float) -> float:
    default = DEFAULT_SPEED_KMH_BY_CLASS.get(road_class, DEFAULT_SPEED_KMH_BY_CLASS["residential"])
    if _is_missing_or_empty(speed_limits):
        return default

    mid = (from_at + to_at) / 2.0
    whole_segment_kmh: float | None = None
    for entry in speed_limits:
        max_speed = entry.get("max_speed") if entry else None
        if not max_speed or max_speed.get("value") is None:
            continue
        value = float(max_speed["value"])
        unit = (max_speed.get("unit") or "km/h").lower()
        kmh = value * _MPH_TO_KMH if unit == "mph" else value
        between = entry.get("between")
        if between and len(between) == 2:
            lo, hi = float(between[0]), float(between[1])
            if lo <= mid <= hi:
                return kmh
        elif whole_segment_kmh is None:
            whole_segment_kmh = kmh
    return whole_segment_kmh if whole_segment_kmh is not None else default


_NON_CAR_MODES = {"foot", "bicycle", "pedestrian"}


def _direction_for_pair(access_restrictions) -> str | None:
    """Return 'both', 'forward', 'backward', or None (not drivable)."""
    denied_forward = False
    denied_backward = False
    if _is_missing_or_empty(access_restrictions):
        access_restrictions = []
    for entry in access_restrictions:
        if not entry or entry.get("access_type") != "denied":
            continue
        when = entry.get("when") or {}
        mode = when.get("mode")
        if mode and all(str(m).lower() in _NON_CAR_MODES for m in mode):
            continue  # foot/bicycle-only restriction, irrelevant to car routing
        heading = when.get("heading")
        if heading == "forward":
            denied_forward = True
        elif heading == "backward":
            denied_backward = True
        elif heading is None:
            denied_forward = True
            denied_backward = True
    if denied_forward and denied_backward:
        return None
    if denied_forward:
        return "backward"
    if denied_backward:
        return "forward"
    return "both"


def fetch_road_network(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    overture_release: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch and build a car-routable graph from Overture road segments.

    Returns (nodes_df, edges_df):
        nodes_df: node_idx (dense, 0-based), connector_id, lat, lng
        edges_df: from_node, to_node, length_m, speed_kmh, weight_ds, class
    """
    import duckdb

    class_list = ", ".join(f"'{c}'" for c in DRIVABLE_CLASSES)
    typer.echo(f"Fetching Overture Maps {overture_release} road segments ...")
    df = duckdb.sql(f"""
        INSTALL spatial;  LOAD spatial;
        SET s3_region = 'us-west-2';

        WITH segs AS (
            SELECT
                id AS segment_id,
                class,
                speed_limits,
                access_restrictions,
                geometry,
                UNNEST(connectors) AS conn
            FROM read_parquet(
                's3://overturemaps-us-west-2/release/{overture_release}/theme=transportation/type=segment/*',
                filename=true, hive_partitioning=1
            )
            WHERE subtype = 'road'
              AND class IN ({class_list})
              AND bbox.xmin BETWEEN {min_lon} AND {max_lon}
              AND bbox.ymin BETWEEN {min_lat} AND {max_lat}
        ),
        ordered AS (
            SELECT
                segment_id, class, speed_limits, access_restrictions, geometry,
                conn.connector_id AS connector_id,
                conn.at AS at,
                ROW_NUMBER() OVER (PARTITION BY segment_id ORDER BY conn.at) AS rn
            FROM segs
        ),
        pairs AS (
            SELECT
                o1.segment_id,
                o1.class AS road_class,
                o1.speed_limits,
                o1.access_restrictions,
                o1.connector_id AS from_connector,
                o1.at AS from_at,
                o2.connector_id AS to_connector,
                o2.at AS to_at,
                ST_Y(ST_LineInterpolatePoint(o1.geometry, o1.at)) AS from_lat,
                ST_X(ST_LineInterpolatePoint(o1.geometry, o1.at)) AS from_lng,
                ST_Y(ST_LineInterpolatePoint(o1.geometry, o2.at)) AS to_lat,
                ST_X(ST_LineInterpolatePoint(o1.geometry, o2.at)) AS to_lng
            FROM ordered o1
            JOIN ordered o2 ON o1.segment_id = o2.segment_id AND o2.rn = o1.rn + 1
        )
        SELECT
            road_class, speed_limits, access_restrictions,
            from_connector, from_at, to_connector, to_at,
            from_lat, from_lng, to_lat, to_lng,
            2 * 6371000 * ASIN(SQRT(
                POWER(SIN(RADIANS((to_lat - from_lat) / 2)), 2) +
                COS(RADIANS(from_lat)) * COS(RADIANS(to_lat)) *
                POWER(SIN(RADIANS((to_lng - from_lng) / 2)), 2)
            )) AS length_m
        FROM pairs
    """).df()

    typer.echo(f"Fetched {len(df):,} road segment pieces; deriving speeds/direction ...")

    node_coords: dict[str, tuple[float, float]] = {}
    records: list[tuple[str, str, float, float, int, str]] = []

    for row in df.itertuples(index=False):
        speed_kmh = _speed_kmh_for_pair(row.speed_limits, row.road_class, row.from_at, row.to_at)
        direction = _direction_for_pair(row.access_restrictions)
        if direction is None or speed_kmh <= 0:
            continue
        length_m = float(row.length_m) if row.length_m and row.length_m > 0 else 0.1
        speed_mps = (speed_kmh * CAR_SPEED_FACTOR) / 3.6
        weight_ds = max(1, round(length_m / speed_mps * 10))
        node_coords[row.from_connector] = (row.from_lat, row.from_lng)
        node_coords[row.to_connector] = (row.to_lat, row.to_lng)
        if direction in ("both", "forward"):
            records.append((row.from_connector, row.to_connector, length_m, speed_kmh, weight_ds, row.road_class))
        if direction in ("both", "backward"):
            records.append((row.to_connector, row.from_connector, length_m, speed_kmh, weight_ds, row.road_class))

    connector_ids = sorted(node_coords)
    connector_to_idx = {cid: i for i, cid in enumerate(connector_ids)}
    nodes_df = pd.DataFrame(
        {
            "node_idx": np.arange(len(connector_ids), dtype=np.int64),
            "connector_id": connector_ids,
            "lat": [node_coords[c][0] for c in connector_ids],
            "lng": [node_coords[c][1] for c in connector_ids],
        }
    )

    edges_df = pd.DataFrame(
        records,
        columns=["from_connector", "to_connector", "length_m", "speed_kmh", "weight_ds", "class"],
    )
    edges_df["from_node"] = edges_df["from_connector"].map(connector_to_idx).astype(np.int64)
    edges_df["to_node"] = edges_df["to_connector"].map(connector_to_idx).astype(np.int64)
    edges_df = edges_df[["from_node", "to_node", "length_m", "speed_kmh", "weight_ds", "class"]]

    return nodes_df, edges_df


def build_road_graph(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    overture_release: str,
    nodes_output: str,
    edges_output: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a cached road graph from disk, or fetch and cache it."""
    if Path(nodes_output).exists() and Path(edges_output).exists():
        typer.echo(f"Loading cached road graph from {nodes_output} / {edges_output} ...")
        return pd.read_parquet(nodes_output), pd.read_parquet(edges_output)

    nodes_df, edges_df = fetch_road_network(min_lon, min_lat, max_lon, max_lat, overture_release)
    Path(nodes_output).parent.mkdir(parents=True, exist_ok=True)
    Path(edges_output).parent.mkdir(parents=True, exist_ok=True)
    nodes_df.to_parquet(nodes_output, index=False)
    edges_df.to_parquet(edges_output, index=False)
    typer.echo(
        f"Saved road graph: {len(nodes_df):,} nodes, {len(edges_df):,} directed edges "
        f"-> {nodes_output}, {edges_output}"
    )
    return nodes_df, edges_df


def snap_locations_to_graph(
    tessellation_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    max_distance_m: float,
    lat_col: str = "lat",
    lng_col: str = "lng",
) -> np.ndarray:
    """Snap each tessellation row to its nearest road graph node.

    Returns an int64 array aligned 1:1 with ``tessellation_df`` rows; ``-1``
    when the nearest node is farther than ``max_distance_m`` (unsnapped).
    """
    n = len(tessellation_df)
    if len(nodes_df) == 0 or n == 0:
        return np.full(n, -1, dtype=np.int64)

    mean_lat = float(nodes_df["lat"].mean())
    scale = math.cos(math.radians(mean_lat))

    node_lat = nodes_df["lat"].to_numpy(dtype=float)
    node_lng = nodes_df["lng"].to_numpy(dtype=float)
    node_xy = np.column_stack([node_lat, node_lng * scale])

    tree = NearestNeighbors(n_neighbors=1, algorithm="kd_tree", n_jobs=-1)
    tree.fit(node_xy)

    loc_lat = tessellation_df[lat_col].to_numpy(dtype=float)
    loc_lng = tessellation_df[lng_col].to_numpy(dtype=float)
    loc_xy = np.column_stack([loc_lat, loc_lng * scale])
    _, indices = tree.kneighbors(loc_xy)
    nearest_idx = indices[:, 0]

    dist_m = haversine_m(loc_lat, loc_lng, node_lat[nearest_idx], node_lng[nearest_idx])
    node_idx = nodes_df["node_idx"].to_numpy(dtype=np.int64)[nearest_idx]
    return np.where(dist_m <= max_distance_m, node_idx, -1).astype(np.int64)
