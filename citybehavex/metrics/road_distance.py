"""Road-network jump lengths / radius of gyration for mobility comparison.

skmob2's ``jump_lengths``/``radius_of_gyration`` measure straight-line
Haversine distance between stop points. citybehavex's simulation already
routes agents over a real, cached road network (Overture-derived,
contraction-hierarchy routing via ``citybehavex._core.RoadNetworkHandle``);
this module reuses that same graph to recompute the two metrics as actual
road-network distance, for synthetic and real trajectories alike, falling
back to Haversine per-pair wherever a point is unsnapped or the graph is
disconnected between the two points.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from citybehavex.roads import haversine_m, snap_locations_to_graph


def _as_polars(df: pd.DataFrame | pl.DataFrame) -> pl.DataFrame:
    """Accept either a pandas or a polars frame during the pandas->polars
    migration (some callers of this module haven't migrated yet); drop once
    every caller passes polars directly.
    """
    return df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)


def _as_pandas(df: pd.DataFrame | pl.DataFrame) -> pd.DataFrame:
    """``snap_locations_to_graph`` (citybehavex.roads) is pandas-typed and
    out of scope for this migration -- convert right at the call boundary.
    """
    return df if isinstance(df, pd.DataFrame) else df.to_pandas()


def build_road_network_handle(edges_df: pd.DataFrame | pl.DataFrame):
    """Prepare a contraction hierarchy from a cached road graph's edges.

    Build once per report/payload invocation and reuse across every
    ``jump_lengths_km``/``radius_of_gyration_km`` call (CH preparation, not
    the query itself, is the expensive step).
    """
    from citybehavex import _core

    edges_df = _as_polars(edges_df)
    return _core.RoadNetworkHandle(
        edges_df["from_node"].to_numpy().astype(np.int64),
        edges_df["to_node"].to_numpy().astype(np.int64),
        edges_df["weight_ds"].to_numpy().astype(np.int64),
        edges_df["length_m"].to_numpy().astype(np.float64),
    )


def _road_or_haversine_km(
    handle,
    from_node: np.ndarray,
    to_node: np.ndarray,
    from_lat: np.ndarray,
    from_lng: np.ndarray,
    to_lat: np.ndarray,
    to_lng: np.ndarray,
) -> np.ndarray:
    distances_m, connected = handle.batch_distances(
        from_node.astype(np.int64), to_node.astype(np.int64)
    )
    fallback_km = haversine_m(from_lat, from_lng, to_lat, to_lng) / 1000.0
    road_km = np.asarray(distances_m, dtype=np.float64) / 1000.0
    return np.where(np.asarray(connected, dtype=bool), road_km, fallback_km)


def jump_lengths_km(
    df: pd.DataFrame | pl.DataFrame,
    *,
    uid_col: str,
    lat_col: str,
    lng_col: str,
    datetime_col: str,
    handle,
    nodes_df: pd.DataFrame | pl.DataFrame,
    snap_max_distance_m: float = 750.0,
) -> np.ndarray:
    """Road-network jump lengths (km): distance between consecutive stops
    for the same user, sorted by datetime -- mirrors
    ``skmob2.jump_lengths(merge=True)``'s sort key and its inclusion of
    zero-length jumps, but measures along the road network instead of
    straight-line, falling back to Haversine per-pair when unsnapped or
    disconnected.
    """
    sorted_df = _as_polars(df).sort([uid_col, datetime_col], maintain_order=True)
    nodes_pd = _as_pandas(nodes_df)
    node_idx = snap_locations_to_graph(
        _as_pandas(sorted_df.select([lat_col, lng_col])),
        nodes_pd,
        snap_max_distance_m,
        lat_col=lat_col,
        lng_col=lng_col,
    )
    uid_arr = sorted_df[uid_col].to_numpy()
    lat_arr = sorted_df[lat_col].cast(pl.Float64).to_numpy()
    lng_arr = sorted_df[lng_col].cast(pl.Float64).to_numpy()

    same_uid = uid_arr[1:] == uid_arr[:-1]
    from_node = node_idx[:-1][same_uid]
    to_node = node_idx[1:][same_uid]
    if from_node.size == 0:
        return np.empty(0, dtype=np.float64)

    return _road_or_haversine_km(
        handle,
        from_node,
        to_node,
        lat_arr[:-1][same_uid],
        lng_arr[:-1][same_uid],
        lat_arr[1:][same_uid],
        lng_arr[1:][same_uid],
    )


def radius_of_gyration_km(
    df: pd.DataFrame | pl.DataFrame,
    *,
    uid_col: str,
    lat_col: str,
    lng_col: str,
    handle,
    nodes_df: pd.DataFrame | pl.DataFrame,
    snap_max_distance_m: float = 750.0,
) -> pl.DataFrame:
    """Road-network radius of gyration (km) per user: RMS road-network
    distance from each of a user's stops to the arithmetic-mean centroid of
    their stops -- mirrors skmob2's unweighted-centroid formula
    (``r_g(u) = sqrt(mean(d(r_i, r_cm)^2))``), but measures ``d`` along the
    road network instead of straight-line. Returns a
    ``DataFrame[[uid_col, "radius_of_gyration"]]`` matching skmob2's shape.
    """
    df = _as_polars(df)
    if df.is_empty():
        return pl.DataFrame(
            [
                pl.Series(uid_col, [], dtype=df.schema[uid_col]),
                pl.Series("radius_of_gyration", [], dtype=pl.Float64),
            ]
        )

    nodes_pd = _as_pandas(nodes_df)

    centroid = df.group_by(uid_col, maintain_order=True).agg(
        [pl.col(lat_col).mean(), pl.col(lng_col).mean()]
    )
    centroid_node = snap_locations_to_graph(
        _as_pandas(centroid.select([lat_col, lng_col])),
        nodes_pd,
        snap_max_distance_m,
        lat_col=lat_col,
        lng_col=lng_col,
    )
    centroid = centroid.with_columns(
        pl.Series("_centroid_node", centroid_node)
    ).rename({lat_col: "_centroid_lat", lng_col: "_centroid_lng"})

    merged = df.join(centroid, on=uid_col, how="left")
    stop_node = snap_locations_to_graph(
        _as_pandas(merged.select([lat_col, lng_col])),
        nodes_pd,
        snap_max_distance_m,
        lat_col=lat_col,
        lng_col=lng_col,
    )

    dist_km = _road_or_haversine_km(
        handle,
        stop_node,
        merged["_centroid_node"].to_numpy().astype(np.int64),
        merged[lat_col].cast(pl.Float64).to_numpy(),
        merged[lng_col].cast(pl.Float64).to_numpy(),
        merged["_centroid_lat"].cast(pl.Float64).to_numpy(),
        merged["_centroid_lng"].cast(pl.Float64).to_numpy(),
    )

    return (
        pl.DataFrame({uid_col: merged[uid_col], "_dist_km": dist_km})
        .group_by(uid_col, maintain_order=True)
        .agg((pl.col("_dist_km") ** 2).mean().sqrt().alias("radius_of_gyration"))
    )
