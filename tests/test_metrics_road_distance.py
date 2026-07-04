from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from citybehavex.metrics import (
    build_road_network_handle,
    jump_lengths_km,
    radius_of_gyration_km,
)
from citybehavex.roads import haversine_m


def _tiny_road_graph():
    """A 0 -> 1 -> 2 -> 3 chain (cheap by travel time) plus a slower but
    physically shorter direct 0 -> 3 edge, both directions -- mirrors the
    Rust `shortest_path_length_over_a_chain` fixture, so routing must prefer
    the chain (and its length, 3 * 2000m) over the direct edge's length."""
    nodes_df = pd.DataFrame(
        {
            "node_idx": [0, 1, 2, 3],
            "lat": [48.85, 48.85, 48.85, 48.85],
            "lng": [2.35, 2.36, 2.37, 2.38],
        }
    )
    edges_df = pd.DataFrame(
        {
            "from_node": [0, 1, 2, 3, 2, 1, 0, 3],
            "to_node": [1, 2, 3, 2, 1, 0, 3, 0],
            "weight_ds": [100, 100, 100, 100, 100, 100, 1000, 1000],
            "length_m": [2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 500.0, 500.0],
        }
    )
    return nodes_df, edges_df


def test_jump_lengths_km_uses_road_distance_not_haversine():
    nodes_df, edges_df = _tiny_road_graph()
    handle = build_road_network_handle(edges_df)

    df = pd.DataFrame(
        {
            "uid": [1, 1],
            "datetime": pd.to_datetime(["2026-01-01 08:00", "2026-01-01 09:00"]),
            "lat": [48.85, 48.85],
            "lng": [2.35, 2.38],
        }
    )

    jumps_km = jump_lengths_km(
        df, uid_col="uid", lat_col="lat", lng_col="lng", datetime_col="datetime",
        handle=handle, nodes_df=nodes_df, snap_max_distance_m=750.0,
    )
    assert jumps_km.shape == (1,)
    # Routed along the time-optimal chain (3 * 2000m), not the direct-but-slower edge.
    assert jumps_km[0] == pytest.approx(6.0)

    haversine_km = haversine_m(48.85, 2.35, 48.85, 2.38) / 1000.0
    assert jumps_km[0] > haversine_km


def test_jump_lengths_km_falls_back_to_haversine_when_unsnapped():
    nodes_df, edges_df = _tiny_road_graph()
    handle = build_road_network_handle(edges_df)

    # Both stops are far (>> snap_max_distance_m) from any graph node, so
    # both endpoints are unsnapped and the pair must fall back to Haversine.
    df = pd.DataFrame(
        {
            "uid": [1, 1],
            "datetime": pd.to_datetime(["2026-01-01 08:00", "2026-01-01 09:00"]),
            "lat": [10.0, 10.0],
            "lng": [10.0, 10.1],
        }
    )
    jumps_km = jump_lengths_km(
        df, uid_col="uid", lat_col="lat", lng_col="lng", datetime_col="datetime",
        handle=handle, nodes_df=nodes_df, snap_max_distance_m=750.0,
    )
    expected_km = haversine_m(10.0, 10.0, 10.0, 10.1) / 1000.0
    assert jumps_km[0] == pytest.approx(expected_km)


def test_radius_of_gyration_km_uses_road_distance():
    nodes_df, edges_df = _tiny_road_graph()
    handle = build_road_network_handle(edges_df)

    df = pd.DataFrame(
        {
            "uid": [1, 1],
            "lat": [48.85, 48.85],
            "lng": [2.35, 2.38],
        }
    )
    rog = radius_of_gyration_km(
        df, uid_col="uid", lat_col="lat", lng_col="lng",
        handle=handle, nodes_df=nodes_df, snap_max_distance_m=750.0,
    )
    assert list(rog.columns) == ["uid", "radius_of_gyration"]
    assert len(rog) == 1
    assert rog["radius_of_gyration"].iloc[0] > 0.0

    # Road-network RoG should never be smaller than a straight-line RoG
    # computed against the same (unsnapped) points and centroid, since road
    # paths are never shorter than the beeline between the same two points.
    centroid_lat, centroid_lng = df["lat"].mean(), df["lng"].mean()
    haversine_rog = np.sqrt(
        np.mean(
            np.square(
                haversine_m(df["lat"].to_numpy(), df["lng"].to_numpy(), centroid_lat, centroid_lng)
                / 1000.0
            )
        )
    )
    assert rog["radius_of_gyration"].iloc[0] >= haversine_rog
