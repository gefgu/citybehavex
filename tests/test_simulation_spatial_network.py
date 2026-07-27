from __future__ import annotations

import h3
import numpy as np
import pandas as pd

from citybehavex.config.root import CityBehavExConfig
from citybehavex.simulation.network_pipeline import _maybe_snap_to_roads
from citybehavex.simulation.spatial import _resolve_spatial_bounds, h3_cell_strings


def test_h3_cell_strings_match_h3_library_output():
    df = pd.DataFrame(
        {
            "lat": [37.769377, 48.8566],
            "lng": [-122.388519, 2.3522],
        }
    )

    cells = h3_cell_strings(df, resolution=9)

    assert cells.tolist() == [
        h3.latlng_to_cell(lat, lng, 9)
        for lat, lng in zip(df["lat"], df["lng"])
    ]


def test_resolve_spatial_bounds_prefers_config_and_falls_back_to_dataframe():
    df = pd.DataFrame({"lat": [10.0, 12.0], "lng": [20.0, 25.0]})
    configured = CityBehavExConfig.model_validate(
        {
            "simulation": {
                "min_lon": 1.0,
                "min_lat": 2.0,
                "max_lon": 3.0,
                "max_lat": 4.0,
            }
        }
    )
    fallback = CityBehavExConfig.model_validate(
        {
            "simulation": {
                "min_lon": None,
                "min_lat": None,
                "max_lon": None,
                "max_lat": None,
            },
            "tessellation": {
                "min_lon": None,
                "min_lat": None,
                "max_lon": None,
                "max_lat": None,
            },
        }
    )

    assert _resolve_spatial_bounds(configured, df) == (1.0, 2.0, 3.0, 4.0)
    assert _resolve_spatial_bounds(fallback, df) == (20.0, 10.0, 25.0, 12.0)


def test_road_snap_cache_hit_skips_graph_build(monkeypatch, tmp_path):
    snap_path = tmp_path / "road_snap.parquet"
    pd.DataFrame({"road_node": [11, 12]}).to_parquet(snap_path, index=False)
    config = CityBehavExConfig.model_validate(
        {"road_network": {"enabled": True, "snap_output": str(snap_path)}}
    )
    tess = pd.DataFrame({"lat": [1.0, 2.0], "lng": [3.0, 4.0]})

    def fail_build(*_args, **_kwargs):
        raise AssertionError("cache hit should not build the road graph")

    monkeypatch.setattr("citybehavex.simulation.network_pipeline.build_road_graph", fail_build)

    snapped = _maybe_snap_to_roads(config, tess)

    assert snapped["road_node"].tolist() == [11, 12]


def test_road_snap_rebuilds_mismatched_cache(monkeypatch, tmp_path):
    snap_path = tmp_path / "road_snap.parquet"
    pd.DataFrame({"road_node": [11]}).to_parquet(snap_path, index=False)
    config = CityBehavExConfig.model_validate(
        {
            "simulation": {
                "min_lon": 0.0,
                "min_lat": 0.0,
                "max_lon": 10.0,
                "max_lat": 10.0,
            },
            "road_network": {
                "enabled": True,
                "snap_output": str(snap_path),
                "nodes_output": str(tmp_path / "nodes.parquet"),
                "edges_output": str(tmp_path / "edges.parquet"),
            },
        }
    )
    tess = pd.DataFrame({"lat": [1.0, 2.0], "lng": [3.0, 4.0]})

    def fake_build_road_graph(*_args, **_kwargs):
        return (
            pd.DataFrame({"node_idx": [21, 22], "lat": [1.0, 2.0], "lng": [3.0, 4.0]}),
            pd.DataFrame(),
        )

    def fake_snap(*_args, **_kwargs):
        return np.array([21, 22], dtype=np.int64)

    monkeypatch.setattr(
        "citybehavex.simulation.network_pipeline.build_road_graph",
        fake_build_road_graph,
    )
    monkeypatch.setattr(
        "citybehavex.simulation.network_pipeline.snap_locations_to_graph",
        fake_snap,
    )

    snapped = _maybe_snap_to_roads(config, tess)

    assert snapped["road_node"].tolist() == [21, 22]
    assert pd.read_parquet(snap_path)["road_node"].tolist() == [21, 22]
