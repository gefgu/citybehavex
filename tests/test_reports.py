from __future__ import annotations

import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import h3
import pandas as pd
import polars as pl
import pytest
import skmob2

from citybehavex.reports import (
    ALL_REPORT_SECTIONS,
    _activities_sidecar_path,
    _collapse_explicit_purposes,
    _common_part_of_commuters,
    _daily_location_lognormal_dataset,
    _derive_purpose_groups_from_heuristic,
    _mobility_law_visits,
    _motif_visits,
    _prepare_activity_visits,
    _trajectory_od_matrix,
    _visits_for_comparison,
    generate_comparison_report,
    load_trajectory,
    waiting_times_minutes,
)
from citybehavex.reports.network_validation import encounters_sidecar_path
from citybehavex.simulation.core import social_network_sidecar_path


def test_waiting_times_minutes_converts_skmob2_seconds():
    traj = SimpleNamespace(
        df=pd.DataFrame(
            {
                "uid": [1, 1, 1],
                "datetime": pd.to_datetime(
                    [
                        "2026-01-01 00:00:00",
                        "2026-01-01 01:00:00",
                        "2026-01-01 03:00:00",
                    ]
                ),
                "lat": [48.85, 48.86, 48.87],
                "lng": [2.35, 2.36, 2.37],
            }
        ),
        datetime_col="datetime",
        lat_col="lat",
        lng_col="lng",
        uid_col="uid",
    )

    assert waiting_times_minutes(traj) == [60.0, 120.0]


def test_load_trajectory_detects_common_column_names(tmp_path):
    path = tmp_path / "trajectories.parquet"
    pd.DataFrame(
        {
            "user_id": ["u1"],
            "start_timestamp": pd.to_datetime(["2026-01-01 08:00:00"]),
            "latitude": [48.85],
            "longitude": [2.35],
        }
    ).to_parquet(path, index=False)

    traj = load_trajectory(str(path))

    assert traj.uid_col == "user_id"
    assert traj.datetime_col == "start_timestamp"
    assert traj.lat_col == "latitude"
    assert traj.lng_col == "longitude"


def test_trajectory_od_matrix_orders_users_and_excludes_invalid_and_self_loops():
    source = pl.DataFrame(
        {
            "uid": ["u1", "u2", "u1", "u1", "u2", "u1"],
            "datetime": pl.Series(
                [
                    "2026-01-01 10:00:00",
                    "2026-01-01 09:00:00",
                    "2026-01-01 08:00:00",
                    "2026-01-01 09:00:00",
                    "2026-01-01 08:00:00",
                    "2026-01-01 11:00:00",
                ]
            ).str.to_datetime(),
            "lat": [48.90, 48.90, 48.85, 48.85, 48.85, 999.0],
            "lng": [2.45, 2.45, 2.35, 2.35, 2.35, 2.50],
        }
    )

    matrix = _trajectory_od_matrix(
        source,
        uid_col="uid",
        datetime_col="datetime",
        lat_col="lat",
        lng_col="lng",
        resolution=9,
    )

    origin = h3.latlng_to_cell(48.85, 2.35, 9)
    destination = h3.latlng_to_cell(48.90, 2.45, 9)
    row = matrix.filter(pl.col("origin") == origin)
    assert row[destination][0] == 2.0
    assert float(matrix.select(pl.exclude("origin")).sum_horizontal().sum()) == 2.0
    assert origin not in matrix.columns


def test_common_part_of_commuters_uses_trajectory_cpc(monkeypatch):
    traj = SimpleNamespace(
        df=pd.DataFrame(
            {
                "uid": [1, 1],
                "datetime": pd.to_datetime(
                    ["2026-01-01 08:00:00", "2026-01-01 09:00:00"]
                ),
                "lat": [48.85, 48.90],
                "lng": [2.35, 2.45],
            }
        ),
        uid_col="uid",
        datetime_col="datetime",
        lat_col="lat",
        lng_col="lng",
    )
    calls = []

    def cpc_multi(synthetic_traj, observed_traj, *, resolutions):
        calls.append((synthetic_traj, observed_traj, resolutions))
        return [(r, 0.75) for r in resolutions]

    monkeypatch.setattr(
        "citybehavex.reports.comparison.trajectory_common_part_of_commuters_multi",
        cpc_multi,
    )

    values = _common_part_of_commuters(traj, traj)

    assert values == [(7, 0.75), (8, 0.75), (9, 0.75)]
    assert calls == [(traj, traj, (7, 8, 9))]


def test_motif_visits_use_h3_locations_and_binary_purposes():
    source = pl.DataFrame(
        {
            "uid": [1, 1],
            "datetime": pl.Series(
                ["2026-01-01 00:00:00", "2026-01-01 08:00:00"]
            ).str.to_datetime(),
            "lat": [48.85, 48.86],
            "lng": [2.35, 2.36],
            "purpose": ["HOME", "WORK"],
        }
    )

    visits = _visits_for_comparison(
        source,
        uid_col="uid",
        datetime_col="datetime",
        activity_col="purpose",
        location_resolution=10,
    )
    motif_visits = _motif_visits(visits)

    # location_id for lat/lng-derived rows is the H3 cell as a UInt64 (not
    # h3-py's hex-string form) -- callers only group/compare it, and this
    # avoids a per-row Python h3 call at real dataset scale (see _h3_cells).
    expected_cells = [
        h3.str_to_int(h3.latlng_to_cell(lat, lng, 10))
        for lat, lng in zip(source["lat"], source["lng"])
    ]
    assert motif_visits["location_id"].to_list() == expected_cells
    assert motif_visits["purpose"].to_list() == ["HOME", "VISIT"]


def test_explicit_purpose_collapse_uses_home_work_other_only():
    visits = pl.DataFrame(
        {
            "purpose": ["HOME", " work ", "SHOP", "PURCHASE", None, "unknown"],
        }
    )

    grouped = _collapse_explicit_purposes(visits)

    assert grouped["purpose"].to_list() == [
        "HOME",
        "WORK",
        "OTHER",
        "OTHER",
        "OTHER",
        "OTHER",
    ]


def test_heuristic_purpose_derivation_uses_time_location_anchors():
    visits = pl.DataFrame(
        {
            "uid": ["u1"] * 5,
            "start_timestamp": pl.Series(
                [
                    "2026-01-01 02:30",
                    "2026-01-01 05:00",
                    "2026-01-01 10:00",
                    "2026-01-01 15:00",
                    "2026-01-01 20:00",
                ]
            ).str.to_datetime(),
            "location_id": ["home", "home", "work", "work", "shop"],
        }
    )

    derived = _derive_purpose_groups_from_heuristic(visits)

    assert derived["purpose"].to_list() == ["HOME", "HOME", "WORK", "WORK", "OTHER"]


def test_heuristic_purpose_derivation_scopes_masks_per_user():
    """Regression test for per-user mask scoping (the fix for a quadratic-
    blowup bug where the write mask was rebuilt over the whole dataframe once
    per user instead of being scoped to that user's own rows). Rows are
    interleaved across users and use a non-contiguous index to make sure
    per-user grouping and index alignment both hold regardless of row order."""
    visits = pl.DataFrame(
        {
            "uid": ["u1", "u2", "u1", "u2", "u1", "u2"],
            "start_timestamp": pl.Series(
                [
                    "2026-01-01 02:30",
                    "2026-01-01 03:00",
                    "2026-01-01 10:00",
                    "2026-01-01 15:00",
                    "2026-01-01 20:00",
                    "2026-01-01 21:00",
                ]
            ).str.to_datetime(),
            # u2 never visits "home_a"/"work_a" (u1's anchors) and u1 never
            # visits "home_b"/"work_b" (u2's anchors) -- any cross-user mask
            # leak would mislabel these as OTHER instead of HOME/WORK, or
            # vice versa.
            "location_id": ["home_a", "home_b", "work_a", "work_b", "shop", "shop"],
        },
    )

    derived = _derive_purpose_groups_from_heuristic(visits)

    u1 = derived.filter(pl.col("uid") == "u1")["purpose"].to_list()
    u2 = derived.filter(pl.col("uid") == "u2")["purpose"].to_list()
    assert u1 == ["HOME", "WORK", "OTHER"]
    assert u2 == ["HOME", "WORK", "OTHER"]


def test_prepare_activity_visits_warns_when_using_heuristic():
    source = pl.DataFrame(
        {
            "uid": ["u1", "u1", "u1"],
            "datetime": pl.Series(
                ["2026-01-01 03:00", "2026-01-01 10:00", "2026-01-01 18:00"]
            ).str.to_datetime(),
            "location_id": ["home", "work", "shop"],
        }
    )

    result = _prepare_activity_visits(
        source,
        label="survey",
        uid_col="uid",
        datetime_col="datetime",
        activity_col=None,
        location_col="location_id",
        lat_col=None,
        lng_col=None,
    )

    assert result is not None
    assert result.used_heuristic is True
    assert "survey has no explicit purpose column" in result.warning
    assert set(result.visits["purpose"]).issubset({"HOME", "WORK", "OTHER"})


def test_mobility_law_visits_use_existing_locations_or_h3_fallback():
    source = pl.DataFrame(
        {
            "uid": ["u1", "u1"],
            "datetime": pl.Series(
                ["2026-01-01 08:00:00", "2026-01-01 10:00:00"]
            ).str.to_datetime(),
            "lat": [48.85, 48.86],
            "lng": [2.35, 2.36],
            "tile_id": ["home", None],
            "purpose": ["HOME", "WORK"],
        }
    )

    existing = _mobility_law_visits(
        source,
        uid_col="uid",
        datetime_col="datetime",
        lat_col="lat",
        lng_col="lng",
        location_col="tile_id",
        activity_col="purpose",
    )
    fallback = _mobility_law_visits(
        source,
        uid_col="uid",
        datetime_col="datetime",
        lat_col="lat",
        lng_col="lng",
    )

    # Same UInt64-not-hex-string convention as _visits_for_comparison (see
    # _h3_cells): existing["location_id"] stays str-typed (it's built from
    # the string tile_id column with a per-row H3 fallback stringified to
    # match), while the no-location_col path is UInt64 throughout.
    expected_row1_cell = str(h3.str_to_int(h3.latlng_to_cell(source["lat"][1], source["lng"][1], 10)))
    expected_cells = [
        h3.str_to_int(h3.latlng_to_cell(lat, lng, 10))
        for lat, lng in zip(source["lat"], source["lng"])
    ]
    assert existing["location_id"][0] == "home"
    assert existing["location_id"][1] == expected_row1_cell
    assert existing["purpose"].to_list() == ["HOME", "WORK"]
    assert fallback["location_id"].to_list() == expected_cells


def test_daily_location_lognormal_dataset_counts_distinct_locations():
    visits = pl.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u1", "u2", "u2"],
            "timestamp": pl.Series(
                [
                    "2026-01-01 08:00:00",
                    "2026-01-01 09:00:00",
                    "2026-01-01 10:00:00",
                    "2026-01-02 08:00:00",
                    "2026-01-01 08:00:00",
                    "2026-01-01 09:00:00",
                ]
            ).str.to_datetime(),
            "location_id": ["home", "work", "work", "home", "home", "shop"],
        }
    )

    x_points, probabilities, mu, sigma, label = (
        _daily_location_lognormal_dataset(visits, "observed")
    )

    assert x_points.tolist() == [1.0, 2.0]
    assert probabilities.tolist() == [1 / 3, 2 / 3]
    assert mu > 0
    assert sigma > 0
    assert label == "observed"


def test_activities_sidecar_path_uses_synthetic_stem():
    assert (
        _activities_sidecar_path("data/run/synthetic.parquet")
        == "data/run/synthetic_activities.parquet"
    )


def _build_report_fixture(tmp_path):
    """Small but complete synthetic+observed dataset exercising the full
    generate_comparison_report pipeline (jump lengths, RoG, CPC, dwell/trip
    duration, activity JSD, motifs, profiles)."""
    n_agents = 4
    n_stops = 6
    purposes = ["HOME", "WORK", "OTHER"]

    synth_rows = []
    for uid in range(1, n_agents + 1):
        base_lat, base_lng = 48.85 + uid * 0.01, 2.35 + uid * 0.01
        for i in range(n_stops):
            lat, lng = base_lat + (i % 3) * 0.01, base_lng + (i % 3) * 0.01
            ts = datetime(2026, 1, 1) + timedelta(hours=4 * i)
            synth_rows.append(
                {
                    "uid": uid,
                    "datetime": ts,
                    "lat": lat,
                    "lng": lng,
                    "trip_duration_minutes": 15.0,
                    "dwell_minutes": 45.0,
                    "purpose": purposes[i % 3],
                    "location_id": h3.latlng_to_cell(lat, lng, 9),
                }
            )
    synth_df = pl.DataFrame(synth_rows)
    traj = skmob2.TrajDataFrame(
        synth_df, datetime_col="datetime", lat_col="lat", lng_col="lng", uid_col="uid"
    )

    real_rows = []
    for uid in range(101, 101 + n_agents):
        base_lat, base_lng = 48.86 + uid * 0.001, 2.36 + uid * 0.001
        for i in range(n_stops):
            lat, lng = base_lat + (i % 3) * 0.01, base_lng + (i % 3) * 0.01
            ts = datetime(2026, 1, 1) + timedelta(hours=4 * i)
            real_rows.append(
                {
                    "uid": uid,
                    "datetime": ts,
                    "lat": lat,
                    "lng": lng,
                    "duration_minutes": 40.0,
                    "purpose": purposes[i % 3],
                    "location_id": h3.latlng_to_cell(lat, lng, 9),
                }
            )
    real_df = pl.DataFrame(real_rows)
    real_path = tmp_path / "observed.parquet"
    real_df.write_parquet(real_path)
    return traj, str(real_path)


def test_generate_comparison_report_writes_json_metrics(tmp_path):
    traj, real_path = _build_report_fixture(tmp_path)
    synthetic_path = tmp_path / "synthetic.parquet"
    traj.df.write_parquet(synthetic_path)
    social_network_sidecar_path(synthetic_path).write_text(
        json.dumps(
            {
                "kind": "initial_profile_similarity",
                "node_count": 4,
                "edge_count": 3,
                "layout": "profile_svd",
                "directed": True,
                "social_graph_k": 2,
                "nodes": [
                    [0.0, 0.0, 8.0, 1],
                    [1.0, 0.0, 8.0, 2],
                    [0.0, 1.0, 8.0, 3],
                    [1.0, 1.0, 8.0, 4],
                ],
                "edges": [[0, 1, 1.0], [1, 2, 1.0], [2, 3, 1.0]],
                "degrees": [1, 2, 2, 1],
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "agent": [0, 0, 1, 2],
            "contact": [1, 1, 2, 3],
            "tile": [1, 1, 2, 3],
            "ts": [1, 2, 1, 2],
        }
    ).to_parquet(encounters_sidecar_path(synthetic_path), index=False)
    json_path = tmp_path / "metrics.json"

    generate_comparison_report(
        traj=traj,
        synthetic_path=str(synthetic_path),
        real_path=real_path,
        observed_label="observed",
        json_output_path=str(json_path),
        network_validation_config=SimpleNamespace(
            enabled=True,
            synthetic_enabled=True,
            observed_enabled=True,
            location_mode="location_col",
            location_col="purpose",
            h3_resolution=9,
            max_group_size=200,
            random_seed=7,
        ),
    )

    assert json_path.exists()
    payload = json.loads(json_path.read_text())
    assert set(payload["wasserstein"]) >= {
        "jump_lengths_km",
        "visits_per_user",
        "radius_of_gyration_km",
        "dwell_time_min",
    }
    assert set(payload["cpc"]) == {"h3_7", "h3_8", "h3_9"}
    assert set(payload["jsd"]) >= {
        "activity_distribution",
        "activity_transitions",
        "daily_activity_profile",
    }
    assert payload["network_validation"]["synthetic_vs_random"]["comparison"] == "synthetic_vs_random"
    assert payload["network_validation"]["observed_vs_random"]["comparison"] == "observed_vs_random"
    assert set(payload["network_validation"]["synthetic_vs_random"]["wasserstein"]) == {
        "degree",
        "clustering_coefficient",
        "edge_persistence",
        "topological_overlap",
    }
    assert payload["network_validation"]["synthetic_vs_random"]["distributions"]["synthetic"]["edge_persistence"]["count"] == 3
    assert payload["network_validation"]["observed_vs_random"]["distributions"]["observed"]["edge_persistence"]["count"] > 0


def test_generate_comparison_report_adds_transport_spatial_synthetic_only(tmp_path):
    traj, real_path = _build_report_fixture(tmp_path)
    synthetic_path = tmp_path / "synthetic.parquet"
    traj.df.write_parquet(synthetic_path)
    moving_path = tmp_path / "synthetic_moving.parquet"
    pd.DataFrame(
        {
            "uid": [1, 1, 1, 1],
            "stop_id": [2, 2, 3, 3],
            "seq": [0, 1, 0, 1],
            "lat": [48.85, 48.85, 48.86, 48.87],
            "lng": [2.35, 2.36, 2.36, 2.36],
            "t": pd.to_datetime(
                [
                    "2026-01-01 08:00",
                    "2026-01-01 08:10",
                    "2026-01-01 09:00",
                    "2026-01-01 09:20",
                ]
            ),
            "mode": ["car", "car", "walk", "walk"],
        }
    ).to_parquet(moving_path, index=False)
    json_path = tmp_path / "metrics.json"

    generate_comparison_report(
        traj=traj,
        synthetic_path=str(synthetic_path),
        real_path=real_path,
        observed_label="observed",
        json_output_path=str(json_path),
        transport_spatial_config=SimpleNamespace(
            enabled=True,
            observed_enabled=False,
            synthetic_moving_path=str(moving_path),
            mode_map={},
        ),
    )

    payload = json.loads(json_path.read_text())
    modes = {row["mode"]: row for row in payload["transport_spatial"]["synthetic"]["modes"]}
    assert modes["car"]["count"] == 1
    assert modes["walk"]["count"] == 1
    assert modes["car"]["percent"] == pytest.approx(50.0)
    assert modes["walk"]["percent"] == pytest.approx(50.0)
    assert modes["car"]["mean_jump_km"] > 0


def test_generate_comparison_report_transport_spatial_observed_custom_columns(tmp_path):
    traj, real_path = _build_report_fixture(tmp_path)
    synthetic_path = tmp_path / "synthetic.parquet"
    traj.df.write_parquet(synthetic_path)
    moving_path = tmp_path / "synthetic_moving.parquet"
    pd.DataFrame(
        {
            "uid": [1, 1],
            "stop_id": [2, 2],
            "seq": [0, 1],
            "lat": [48.85, 48.86],
            "lng": [2.35, 2.35],
            "t": pd.to_datetime(["2026-01-01 08:00", "2026-01-01 08:12"]),
            "mode": ["rail", "rail"],
        }
    ).to_parquet(moving_path, index=False)

    observed = pl.read_parquet(real_path).with_columns(
        pl.col("uid").alias("person"),
        pl.col("datetime").alias("started_at"),
        pl.col("lat").alias("y"),
        pl.col("lng").alias("x"),
        pl.when(pl.arange(0, pl.len()) % 2 == 0)
        .then(pl.lit("metro"))
        .otherwise(pl.lit("auto"))
        .alias("travel_kind"),
    )
    observed_custom_path = tmp_path / "observed_custom.parquet"
    observed.write_parquet(observed_custom_path)
    json_path = tmp_path / "metrics.json"

    generate_comparison_report(
        traj=traj,
        synthetic_path=str(synthetic_path),
        real_path=str(observed_custom_path),
        observed_label="observed",
        json_output_path=str(json_path),
        transport_spatial_config=SimpleNamespace(
            enabled=True,
            observed_enabled=True,
            synthetic_moving_path=str(moving_path),
            uid_col="person",
            datetime_col="started_at",
            lat_col="y",
            lng_col="x",
            transport_col="travel_kind",
            mode_map={"metro": "rail", "auto": "car"},
        ),
    )

    payload = json.loads(json_path.read_text())
    assert "observed" in payload["transport_spatial"]
    observed_modes = {
        row["mode"]: row for row in payload["transport_spatial"]["observed"]["modes"]
    }
    assert set(observed_modes) == {"car", "rail"}
    assert observed_modes["car"]["count"] > 0
    assert observed_modes["rail"]["count"] > 0


def test_generate_comparison_report_transport_spatial_missing_sidecar_warns(tmp_path, capsys):
    traj, real_path = _build_report_fixture(tmp_path)
    synthetic_path = tmp_path / "synthetic.parquet"
    traj.df.write_parquet(synthetic_path)
    json_path = tmp_path / "metrics.json"

    generate_comparison_report(
        traj=traj,
        synthetic_path=str(synthetic_path),
        real_path=real_path,
        observed_label="observed",
        json_output_path=str(json_path),
        transport_spatial_config=SimpleNamespace(enabled=True, observed_enabled=False),
    )

    assert "moving sidecar not found" in capsys.readouterr().err
    payload = json.loads(json_path.read_text())
    assert "transport_spatial" not in payload


def test_generate_comparison_report_uses_road_network_distance_when_provided(tmp_path):
    """When a cached road graph is supplied, jump_lengths_km must come from
    road-network routing, not skmob2's straight-line Haversine. Build a tiny
    complete graph over every unique fixture coordinate, with each direct
    edge's length set to exactly 2x its Haversine distance (and travel-time
    weight derived consistently from that same length, so the direct edge is
    always the unique time-optimal route by the triangle inequality) -- the
    reported jump_lengths_km Wasserstein distance must then come out at
    exactly 2x the plain-Haversine baseline, since 1-D Wasserstein distance
    scales linearly under a uniform scaling of both distributions.
    """
    import itertools

    import numpy as np

    from citybehavex.roads import haversine_m

    traj, real_path = _build_report_fixture(tmp_path)
    real_df = pl.read_parquet(real_path)

    coords = pl.concat([traj.df.select(["lat", "lng"]), real_df.select(["lat", "lng"])]).unique(
        maintain_order=True
    )
    lat = coords["lat"].to_numpy()
    lng = coords["lng"].to_numpy()
    pairs = list(itertools.permutations(range(len(coords)), 2))
    from_node = np.array([p[0] for p in pairs], dtype=np.int64)
    to_node = np.array([p[1] for p in pairs], dtype=np.int64)
    length_m = haversine_m(lat[from_node], lng[from_node], lat[to_node], lng[to_node]) * 2.0
    weight_ds = np.maximum(1, np.round(length_m)).astype(np.int64)

    road_nodes_df = pl.DataFrame({"node_idx": np.arange(len(coords), dtype=np.int64), "lat": lat, "lng": lng})
    road_edges_df = pl.DataFrame(
        {"from_node": from_node, "to_node": to_node, "length_m": length_m, "weight_ds": weight_ds}
    )

    baseline_json = tmp_path / "baseline_metrics.json"
    generate_comparison_report(
        traj=traj,
        real_path=real_path,
        observed_label="observed",
        json_output_path=str(baseline_json),
    )
    baseline = json.loads(baseline_json.read_text())["wasserstein"]["jump_lengths_km"]

    road_json = tmp_path / "road_metrics.json"
    generate_comparison_report(
        traj=traj,
        real_path=real_path,
        observed_label="observed",
        json_output_path=str(road_json),
        road_nodes_df=road_nodes_df,
        road_edges_df=road_edges_df,
    )
    road = json.loads(road_json.read_text())["wasserstein"]["jump_lengths_km"]

    assert road == pytest.approx(2.0 * baseline, rel=1e-6)


def test_generate_comparison_report_rejects_unknown_section(tmp_path):
    traj, real_path = _build_report_fixture(tmp_path)

    with pytest.raises(ValueError, match="Unknown comparison report section"):
        generate_comparison_report(
            traj=traj,
            real_path=real_path,
            observed_label="observed",
            sections=["bogus"],
        )


def test_generate_comparison_report_sections_skip_expensive_blocks(tmp_path, monkeypatch):
    traj, real_path = _build_report_fixture(tmp_path)
    json_path = tmp_path / "metrics.json"

    import citybehavex.reports.comparison as comparison_module

    called = {"motifs": False}

    original_discover = comparison_module.discover_daily_motifs_from_agents

    def fake_discover(*args, **kwargs):
        called["motifs"] = True
        return original_discover(*args, **kwargs)

    monkeypatch.setattr(comparison_module, "discover_daily_motifs_from_agents", fake_discover)

    generate_comparison_report(
        traj=traj,
        real_path=real_path,
        observed_label="observed",
        json_output_path=str(json_path),
        sections=["cpc"],
    )

    assert not called["motifs"]
    payload = json.loads(json_path.read_text())
    assert set(payload["cpc"]) == {"h3_7", "h3_8", "h3_9"}


def test_all_report_sections_constant_matches_config_validator():
    from citybehavex.reports.config import ComparisonConfig

    # sanity check that the config's validator and the report's own gating
    # agree on the recognized section names.
    cfg = ComparisonConfig(sections=sorted(ALL_REPORT_SECTIONS))
    assert cfg.network_validation.enabled is False
    enabled = ComparisonConfig(
        network_validation={
            "enabled": True,
            "observed_enabled": True,
            "location_mode": "h3",
            "h3_resolution": 9,
        }
    )
    assert enabled.network_validation.enabled is True
    assert enabled.network_validation.observed_enabled is True
