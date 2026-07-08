from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import polars as pl

from citybehavex.reports.network_validation import encounters_sidecar_path
from citybehavex.simulation.core import social_network_sidecar_path
from web.backend.app.payload import (
    TIME_USE_CATEGORIES,
    _filter_df,
    _load_social_network_sidecar,
    _special_day_filters,
    build_chart_base_payload,
    build_chart_filter_payload,
    build_chart_section_payload,
    build_comparison_payload,
    build_metrics_export_payload,
    build_network_validation_payload,
)


def test_load_social_network_sidecar_returns_none_when_absent(tmp_path):
    synthetic = tmp_path / "trajectories_20260101T010203.parquet"
    pd.DataFrame({"uid": [1]}).to_parquet(synthetic, index=False)

    assert _load_social_network_sidecar(str(synthetic)) is None


def test_load_social_network_sidecar_validates_and_returns_payload(tmp_path):
    synthetic = tmp_path / "trajectories_20260101T010203.parquet"
    pd.DataFrame({"uid": [1]}).to_parquet(synthetic, index=False)
    sidecar = social_network_sidecar_path(synthetic)
    payload = {
        "kind": "initial_profile_similarity",
        "node_count": 2,
        "edge_count": 1,
        "layout": "profile_svd",
        "directed": True,
        "social_graph_k": 1,
        "nodes": [[0.0, 0.0, 8.0, 1, "worker"], [1.0, 1.0, 8.0, 2, "student"]],
        "edges": [[0, 1, 0.75]],
        "degrees": [1, 0],
    }
    sidecar.write_text(json.dumps(payload), encoding="utf-8")

    assert _load_social_network_sidecar(str(synthetic)) == payload


def test_load_social_network_sidecar_caps_visible_agents(tmp_path):
    synthetic = tmp_path / "trajectories_20260101T010203.parquet"
    pd.DataFrame({"uid": [1]}).to_parquet(synthetic, index=False)
    sidecar = social_network_sidecar_path(synthetic)
    payload = {
        "kind": "initial_profile_similarity",
        "node_count": 2001,
        "edge_count": 3,
        "layout": "profile_svd",
        "directed": True,
        "social_graph_k": 1,
        "nodes": [[float(i), 0.0, 8.0, i] for i in range(2001)],
        "edges": [[0, 1999, 1.0], [1999, 2000, 1.0], [2000, 1, 1.0]],
        "degrees": [1 for _ in range(2001)],
    }
    sidecar.write_text(json.dumps(payload), encoding="utf-8")

    loaded = _load_social_network_sidecar(str(synthetic))

    assert loaded is not None
    assert loaded["node_count"] == 2001
    assert loaded["edge_count"] == 3
    assert len(loaded["nodes"]) == 2000
    assert loaded["edges"] == [[0, 1999, 1.0]]
    assert len(loaded["degrees"]) == 2000
    assert loaded["nodes_sampled"] is True
    assert loaded["edges_sampled"] is True


def test_build_network_validation_payload_includes_synthetic_validation(tmp_path):
    # network_validation moved to its own endpoint/build function (see
    # web/backend/app/api/charts.py's /network-validation route) so its
    # build time doesn't block the rest of /charts -- these two tests moved
    # from build_comparison_payload accordingly.
    synthetic = tmp_path / "synthetic.parquet"
    pd.DataFrame({"uid": ["u1", "u1"]}).to_parquet(synthetic, index=False)
    social_network_sidecar_path(synthetic).write_text(
        json.dumps(
            {
                "kind": "initial_profile_similarity",
                "node_count": 3,
                "edge_count": 2,
                "layout": "profile_svd",
                "directed": True,
                "social_graph_k": 2,
                "nodes": [[0.0, 0.0, 8.0, 1], [1.0, 0.0, 8.0, 2], [0.0, 1.0, 8.0, 3]],
                "edges": [[0, 1, 1.0], [1, 2, 1.0]],
                "degrees": [1, 2, 1],
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "agent": [0, 0, 1],
            "contact": [1, 1, 2],
            "tile": [1, 1, 2],
            "ts": [1, 2, 1],
        }
    ).to_parquet(encounters_sidecar_path(synthetic), index=False)

    payload = build_network_validation_payload(
        str(synthetic),
        None,
        SimpleNamespace(
            enabled=True,
            synthetic_enabled=True,
            observed_enabled=False,
            random_seed=7,
        ),
    )

    validation = payload["network_validation"]
    assert validation is not None
    assert validation["synthetic_vs_random"]["comparison"] == "synthetic_vs_random"
    assert validation["synthetic_vs_random"]["distributions"]["synthetic"]["edge_persistence"]["count"] == 2
    assert validation["synthetic_vs_random"]["random_network"]["kind"] == "degree_preserving_rnd"


def test_build_network_validation_payload_includes_observed_validation(tmp_path):
    synthetic = tmp_path / "synthetic.parquet"
    observed = tmp_path / "observed.parquet"
    pd.DataFrame({"uid": ["u1"]}).to_parquet(synthetic, index=False)
    pd.DataFrame(
        {
            "uid": ["a", "b", "a", "b"],
            "datetime": pd.to_datetime(["2026-01-01 08:00", "2026-01-01 09:00", "2026-01-02 08:00", "2026-01-02 09:00"]),
            "lat": [48.85, 48.85, 48.85, 48.85],
            "lng": [2.35, 2.35, 2.35, 2.35],
            "location_id": ["x", "x", "x", "x"],
        }
    ).to_parquet(observed, index=False)

    payload = build_network_validation_payload(
        str(synthetic),
        str(observed),
        SimpleNamespace(
            enabled=True,
            synthetic_enabled=False,
            observed_enabled=True,
            location_mode="location_col",
            location_col="location_id",
            max_group_size=200,
            random_seed=7,
        ),
    )

    validation = payload["network_validation"]
    assert validation is not None
    assert validation["observed_vs_random"]["comparison"] == "observed_vs_random"
    assert validation["observed_vs_random"]["distributions"]["observed"]["edge_persistence"]["count"] == 1
    assert validation["observed_vs_random"]["source_network"]["kind"] == "observed_daily_copresence"


def test_build_comparison_payload_groups_activity_and_micro_usage(monkeypatch, tmp_path):
    synthetic = tmp_path / "synthetic.parquet"
    observed = tmp_path / "observed.parquet"
    activities = tmp_path / "synthetic_activities.parquet"

    synthetic_df = pd.DataFrame(
        {
            "uid": ["u1", "u1", "u1"],
            "datetime": pd.to_datetime(
                ["2026-01-01 03:00", "2026-01-01 10:00", "2026-01-01 18:00"]
            ),
            "lat": [48.85, 48.86, 48.87],
            "lng": [2.35, 2.36, 2.37],
            "purpose": ["HOME", "WORK", "SHOP"],
        }
    )
    observed_df = pd.DataFrame(
        {
            "uid": ["u2", "u2", "u2"],
            "datetime": pd.to_datetime(
                ["2026-01-01 03:00", "2026-01-01 10:00", "2026-01-01 18:00"]
            ),
            "lat": [48.85, 48.86, 48.87],
            "lng": [2.35, 2.36, 2.37],
            "location_id": ["home", "work", "shop"],
        }
    )
    activities_df = pd.DataFrame(
        {
            "uid": [1, 1],
            "activity": [0, 3],
            "arrival": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 08:00"]),
            "departure": pd.to_datetime(["2026-01-01 01:00", "2026-01-01 09:00"]),
        }
    )
    synthetic_df.to_parquet(synthetic, index=False)
    observed_df.to_parquet(observed, index=False)
    activities_df.to_parquet(activities, index=False)

    monkeypatch.setattr(
        "web.backend.app.payload.visits_per_user_wasserstein_distance",
        lambda *args, **kwargs: (0.0, None),
    )
    monkeypatch.setattr(
        "web.backend.app.payload._common_part_of_commuters",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "web.backend.app.payload._mobility_law_visits",
        lambda *args, **kwargs: pl.DataFrame(
            {
                "user_id": ["u1", "u1"],
                "timestamp": pl.Series(["2026-01-01 00:00", "2026-01-01 01:00"]).str.to_datetime(),
                "location_id": ["a", "b"],
                "lat": [48.85, 48.86],
                "lng": [2.35, 2.36],
            }
        ),
    )
    payload = build_comparison_payload(
        str(synthetic),
        str(observed),
        "observed",
        synthetic_activities_path=str(activities),
    )

    assert payload["mode"] == "comparison"
    assert payload["activity"] is not None
    all_activity = payload["activity"]["groups"][0]
    assert all_activity["filter_key"] == "all"
    assert all_activity["purpose"]["categories"] == ["HOME", "WORK", "OTHER"]
    assert any("observed has no explicit purpose column" in w for w in payload["warnings"])
    assert payload["micro_activity_usage"] is not None
    names = {
        series["name"]
        for group in payload["micro_activity_usage"]["groups"]
        for series in group["block"]["series"]
    }
    assert {"sleep", "paidwork"}.issubset(names)
    assert {group["filter_key"] for group in payload["ecdf"]["groups"]} >= {
        "all",
        "weekday",
        "weekend",
        "morning",
        "afternoon",
        "evening",
        "night",
    }
    assert any(row["filter_key"] == "all" for row in payload["metrics"]["wasserstein"])


def test_build_comparison_payload_includes_time_use_comparison(monkeypatch, tmp_path):
    synthetic = tmp_path / "synthetic.parquet"
    activities = tmp_path / "synthetic_activities.parquet"
    time_use = tmp_path / "time_use.parquet"

    pd.DataFrame(
        {
            "uid": ["u1"],
            "datetime": pd.to_datetime(["2026-01-01 00:00"]),
            "lat": [48.85],
            "lng": [2.35],
            "purpose": ["HOME"],
        }
    ).to_parquet(synthetic, index=False)
    pd.DataFrame(
        {
            "uid": [1, 1],
            "activity": [0, 3],
            "arrival": pd.to_datetime(["2026-01-05 00:00", "2026-01-05 08:00"]),
            "departure": pd.to_datetime(["2026-01-05 08:00", "2026-01-05 10:00"]),
        }
    ).to_parquet(activities, index=False)
    rows = []
    for day, weight, sleep, paidwork in [
        ("Monday", 1.0, 480.0, 120.0),
        ("Saturday", 3.0, 600.0, 60.0),
    ]:
        row = {
            "country": "France",
            "survey": 2009,
            "day": day,
            "propwt": weight,
            **{category: 0.0 for category in TIME_USE_CATEGORIES},
        }
        row["sleep"] = sleep
        row["paidwork"] = paidwork
        rows.append(row)
    pd.DataFrame(rows).to_parquet(time_use, index=False)

    monkeypatch.setattr(
        "web.backend.app.payload._mobility_law_visits",
        lambda *args, **kwargs: pl.DataFrame(
            {
                "user_id": ["u1"],
                "timestamp": pl.Series(["2026-01-01 00:00"]).str.to_datetime(),
                "location_id": ["a"],
                "lat": [48.85],
                "lng": [2.35],
            }
        ),
    )

    payload = build_comparison_payload(
        str(synthetic),
        None,
        "observed",
        synthetic_activities_path=str(activities),
        time_use_path=str(time_use),
        time_use_label="MTUS France 2009",
        time_use_country="France",
        time_use_survey=2009,
    )

    assert payload["time_use_comparison"] is not None
    groups = {group["filter_key"]: group for group in payload["time_use_comparison"]["groups"]}
    assert set(groups) == {"all", "weekday", "weekend"}
    all_block = groups["all"]["block"]
    assert all_block["categories"] == TIME_USE_CATEGORIES
    assert all_block["labels"] == ["MTUS France 2009", "synthetic"]
    sleep = next(row for row in all_block["rows"] if row["category"] == "sleep")
    assert sleep["observed_minutes"] == 570.0
    assert sleep["synthetic_minutes"] == 480.0
    assert sleep["mtus_minutes"] == 570.0
    assert sleep["simulation_minutes"] == 480.0
    assert sleep["difference_minutes"] == -90.0
    metric = next(
        row
        for row in payload["metrics"]["time_use"]
        if row["filter_key"] == "all"
        and row["metric_name"] == "Mean absolute time-use share difference"
    )
    assert metric["unit"] == "pct points"
    assert metric["value"] > 0


def test_metrics_section_includes_stvd_distances(monkeypatch, tmp_path):
    synthetic = tmp_path / "synthetic.parquet"
    observed = tmp_path / "observed.parquet"
    pd.DataFrame(
        {
            "uid": ["u1", "u1"],
            "datetime": pd.to_datetime(["2026-01-05 08:00", "2026-01-05 09:00"]),
            "lat": [48.85, 48.86],
            "lng": [2.35, 2.36],
            "purpose": ["HOME", "WORK"],
            "dwell_minutes": [60.0, 60.0],
        }
    ).to_parquet(synthetic, index=False)
    pd.DataFrame(
        {
            "uid": ["u2", "u2"],
            "datetime": pd.to_datetime(["2026-01-05 08:00", "2026-01-05 10:00"]),
            "lat": [48.8505, 48.8605],
            "lng": [2.3505, 2.3605],
            "purpose": ["HOME", "WORK"],
            "location_id": ["home", "work"],
        }
    ).to_parquet(observed, index=False)
    monkeypatch.setattr("web.backend.app.payload._common_part_of_commuters", lambda *args, **kwargs: [])
    monkeypatch.setattr("web.backend.app.payload.stvd_emd", lambda *args, **kwargs: 123.456)

    payload = build_chart_section_payload(
        "metrics",
        "all",
        synthetic_path=str(synthetic),
        observed_path=str(observed),
        observed_label="observed",
    )

    stvd_rows = payload["metrics"]["stvd"]
    assert [(row["filter_key"], row["resolution"], row["unit"]) for row in stvd_rows] == [
        ("all", 7, "m"),
        ("all", 8, "m"),
        ("all", 9, "m"),
    ]
    assert {row["metric_name"] for row in stvd_rows} == {"STVD-EMD"}
    assert {row["value"] for row in stvd_rows} == {123.456}


def test_metrics_export_payload_includes_all_metric_groups(monkeypatch, tmp_path):
    synthetic = tmp_path / "synthetic.parquet"
    activities = tmp_path / "synthetic_activities.parquet"
    time_use = tmp_path / "time_use.parquet"
    pd.DataFrame(
        {
            "uid": ["u1"],
            "datetime": pd.to_datetime(["2026-01-05 08:00"]),
            "lat": [48.85],
            "lng": [2.35],
            "purpose": ["HOME"],
        }
    ).to_parquet(synthetic, index=False)
    pd.DataFrame(
        {
            "uid": [1],
            "activity": [0],
            "arrival": pd.to_datetime(["2026-01-05 00:00"]),
            "departure": pd.to_datetime(["2026-01-05 08:00"]),
        }
    ).to_parquet(activities, index=False)
    rows = []
    for day in ["Monday", "Saturday"]:
        rows.append(
            {
                "country": "France",
                "survey": 2009,
                "day": day,
                "propwt": 1.0,
                **{category: 0.0 for category in TIME_USE_CATEGORIES},
                "sleep": 480.0,
            }
        )
    pd.DataFrame(rows).to_parquet(time_use, index=False)
    monkeypatch.setattr(
        "web.backend.app.payload._mobility_law_visits",
        lambda *args, **kwargs: pl.DataFrame(
            {
                "user_id": ["u1"],
                "timestamp": pl.Series(["2026-01-01 00:00"]).str.to_datetime(),
                "location_id": ["a"],
                "lat": [48.85],
                "lng": [2.35],
            }
        ),
    )

    payload = build_metrics_export_payload(
        str(synthetic),
        None,
        "observed",
        synthetic_activities_path=str(activities),
        time_use_path=str(time_use),
        time_use_label="MTUS France 2009",
        time_use_country="France",
        time_use_survey=2009,
    )

    assert set(payload["metrics"]) == {"wasserstein", "jsd", "cpc", "time_use", "stvd"}
    assert any(row["filter_key"] == "weekday" for row in payload["metrics"]["time_use"])
    assert any(row["key"] == "morning" for row in payload["filters"])
    assert {row["category"] for row in payload["time_use_table"]} == set(TIME_USE_CATEGORIES)


def test_build_comparison_payload_supports_synthetic_only(monkeypatch, tmp_path):
    synthetic = tmp_path / "synthetic.parquet"
    synthetic_df = pd.DataFrame(
        {
            "uid": ["u1", "u1", "u1"],
            "datetime": pd.to_datetime(
                ["2026-01-01 03:00", "2026-01-01 10:00", "2026-01-01 18:00"]
            ),
            "lat": [48.85, 48.86, 48.87],
            "lng": [2.35, 2.36, 2.37],
            "purpose": ["HOME", "WORK", "SHOP"],
            "trip_duration_minutes": [10.0, 15.0, 20.0],
            "dwell_minutes": [300.0, 120.0, 90.0],
        }
    )
    synthetic_df.to_parquet(synthetic, index=False)

    monkeypatch.setattr(
        "web.backend.app.payload._mobility_law_visits",
        lambda *args, **kwargs: pl.DataFrame(
            {
                "user_id": ["u1", "u1"],
                "timestamp": pl.Series(["2026-01-01 00:00", "2026-01-01 01:00"]).str.to_datetime(),
                "location_id": ["a", "b"],
                "lat": [48.85, 48.86],
                "lng": [2.35, 2.36],
            }
        ),
    )

    payload = build_comparison_payload(str(synthetic), None, "observed")

    assert payload["mode"] == "synthetic_only"
    assert "observed" not in payload["labels"]
    assert payload["metrics"]["wasserstein"] == []
    assert payload["metrics"]["jsd"] == []
    assert payload["metrics"]["cpc"] == []
    assert payload["metrics"]["time_use"] == []
    assert payload["metrics"]["stvd"] == []
    all_activity = payload["activity"]["groups"][0]
    assert all_activity["transition_difference"]["matrix_mode"] == "raw"
    assert all_activity["daily_activity_difference"]["matrix_mode"] == "raw"
    assert len(all_activity["purpose"]["series"]) == 1


def test_build_chart_base_payload_returns_only_all_filter(monkeypatch, tmp_path):
    synthetic = tmp_path / "synthetic.parquet"
    observed = tmp_path / "observed.parquet"
    pd.DataFrame(
        {
            "uid": ["u1", "u1", "u1"],
            "datetime": pd.to_datetime(["2026-01-01 03:00", "2026-01-01 10:00", "2026-01-01 18:00"]),
            "lat": [48.85, 48.86, 48.87],
            "lng": [2.35, 2.36, 2.37],
            "purpose": ["HOME", "WORK", "SHOP"],
            "dwell_minutes": [300.0, 120.0, 90.0],
        }
    ).to_parquet(synthetic, index=False)
    pd.DataFrame(
        {
            "uid": ["u2", "u2", "u2"],
            "datetime": pd.to_datetime(["2026-01-01 03:00", "2026-01-01 10:00", "2026-01-01 18:00"]),
            "lat": [48.85, 48.86, 48.87],
            "lng": [2.35, 2.36, 2.37],
            "location_id": ["home", "work", "shop"],
        }
    ).to_parquet(observed, index=False)
    monkeypatch.setattr(
        "web.backend.app.payload.visits_per_user_wasserstein_distance",
        lambda *args, **kwargs: (0.0, None),
    )
    monkeypatch.setattr("web.backend.app.payload._common_part_of_commuters", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        "web.backend.app.payload._mobility_law_visits",
        lambda *args, **kwargs: pl.DataFrame(
            {
                "user_id": ["u1", "u1"],
                "timestamp": pl.Series(["2026-01-01 00:00", "2026-01-01 01:00"]).str.to_datetime(),
                "location_id": ["a", "b"],
                "lat": [48.85, 48.86],
                "lng": [2.35, 2.36],
            }
        ),
    )

    payload = build_chart_base_payload(str(synthetic), str(observed), "observed")

    assert payload["loaded_filters"] == []
    assert payload["enabled_sections"] == [
        "distributions",
        "metrics",
        "transport-spatial",
        "activity",
        "mobility-laws",
        "micro-activity",
        "time-use",
        "motifs",
        "stvd",
        "profiles",
        "social-network",
    ]
    assert payload["ecdf"]["groups"] == []
    assert payload["available_filters"] == [
        {"key": "all", "label": "All"},
        {"key": "weekday", "label": "Weekday"},
        {"key": "weekend", "label": "Weekend"},
    ]
    assert any(option["key"] == "morning" for option in payload["distribution_filters"])

    section = build_chart_section_payload(
        "distributions",
        "all",
        synthetic_path=str(synthetic),
        observed_path=str(observed),
        observed_label="observed",
    )
    assert section["loaded_filters"] == ["all"]
    assert [group["filter_key"] for group in section["ecdf"]["groups"]] == ["all"]


def test_build_chart_filter_payload_returns_one_filter(monkeypatch, tmp_path):
    synthetic = tmp_path / "synthetic.parquet"
    pd.DataFrame(
        {
            "uid": ["u1", "u1", "u1"],
            "datetime": pd.to_datetime(["2026-01-03 03:00", "2026-01-05 10:00", "2026-01-05 18:00"]),
            "lat": [48.85, 48.86, 48.87],
            "lng": [2.35, 2.36, 2.37],
            "purpose": ["HOME", "WORK", "SHOP"],
            "trip_duration_minutes": [10.0, 15.0, 20.0],
            "dwell_minutes": [300.0, 120.0, 90.0],
        }
    ).to_parquet(synthetic, index=False)
    monkeypatch.setattr(
        "web.backend.app.payload._mobility_law_visits",
        lambda *args, **kwargs: pl.DataFrame(
            {
                "user_id": ["u1", "u1"],
                "timestamp": pl.Series(["2026-01-01 00:00", "2026-01-01 01:00"]).str.to_datetime(),
                "location_id": ["a", "b"],
                "lat": [48.85, 48.86],
                "lng": [2.35, 2.36],
            }
        ),
    )

    payload = build_chart_filter_payload(str(synthetic), None, "observed", "weekday")

    assert payload["loaded_filters"] == ["weekday"]
    assert [group["filter_key"] for group in payload["ecdf"]["groups"]] == ["weekday"]
    assert payload["activity"] is not None
    assert [group["filter_key"] for group in payload["activity"]["groups"]] == ["weekday"]
    assert payload["profiles"] is None
    assert payload["social_network"] is None


def test_build_chart_filter_payload_time_filter_only_builds_distribution(monkeypatch, tmp_path):
    synthetic = tmp_path / "synthetic.parquet"
    pd.DataFrame(
        {
            "uid": ["u1", "u1"],
            "datetime": pd.to_datetime(["2026-01-01 07:00", "2026-01-01 10:00"]),
            "lat": [48.85, 48.86],
            "lng": [2.35, 2.36],
            "purpose": ["HOME", "WORK"],
            "dwell_minutes": [120.0, 90.0],
        }
    ).to_parquet(synthetic, index=False)
    monkeypatch.setattr(
        "web.backend.app.payload._mobility_law_visits",
        lambda *args, **kwargs: pl.DataFrame(
            {
                "user_id": ["u1"],
                "timestamp": pl.Series(["2026-01-01 00:00"]).str.to_datetime(),
                "location_id": ["a"],
                "lat": [48.85],
                "lng": [2.35],
            }
        ),
    )

    payload = build_chart_filter_payload(str(synthetic), None, "observed", "morning")

    assert payload["loaded_filters"] == ["morning"]
    assert [group["filter_key"] for group in payload["ecdf"]["groups"]] == ["morning"]
    assert payload["activity"] is None
    assert payload["mobility_laws"] is None


def test_special_day_filters_builds_date_range_metadata():
    filters = _special_day_filters(
        [{"name": "emergency", "start_date": "2019-11-14", "end_date": "2019-11-28"}]
    )
    assert filters == [
        {
            "key": "emergency",
            "label": "Emergency",
            "kind": "date_range",
            "start": "2019-11-14",
            "end": "2019-11-28",
        }
    ]
    assert _special_day_filters(None) == []


def test_filter_df_date_range_keeps_only_rows_inside_window():
    df = pl.DataFrame(
        {
            "datetime": pl.Series(
                ["2019-11-13 12:00", "2019-11-14 00:00", "2019-11-20 08:00", "2019-11-29 00:00"]
            ).str.to_datetime(),
            "value": [1, 2, 3, 4],
        }
    )
    meta = {"key": "emergency", "kind": "date_range", "start": "2019-11-14", "end": "2019-11-28"}
    filtered = _filter_df(df, "datetime", meta)
    assert filtered["value"].to_list() == [2, 3]


def test_build_comparison_payload_includes_special_day_filter_group(monkeypatch, tmp_path):
    synthetic = tmp_path / "synthetic.parquet"
    synthetic_df = pd.DataFrame(
        {
            "uid": ["u1", "u1", "u1"],
            "datetime": pd.to_datetime(
                ["2026-01-01 03:00", "2026-01-05 10:00", "2026-01-05 18:00"]
            ),
            "lat": [48.85, 48.86, 48.87],
            "lng": [2.35, 2.36, 2.37],
            "purpose": ["HOME", "WORK", "SHOP"],
            "trip_duration_minutes": [10.0, 15.0, 20.0],
            "dwell_minutes": [300.0, 120.0, 90.0],
        }
    )
    synthetic_df.to_parquet(synthetic, index=False)

    monkeypatch.setattr(
        "web.backend.app.payload._mobility_law_visits",
        lambda *args, **kwargs: pl.DataFrame(
            {
                "user_id": ["u1", "u1"],
                "timestamp": pl.Series(["2026-01-01 00:00", "2026-01-01 01:00"]).str.to_datetime(),
                "location_id": ["a", "b"],
                "lat": [48.85, 48.86],
                "lng": [2.35, 2.36],
            }
        ),
    )

    payload = build_comparison_payload(
        str(synthetic),
        None,
        "observed",
        special_days=[{"name": "emergency", "start_date": "2026-01-05", "end_date": "2026-01-05"}],
    )

    ecdf_keys = {group["filter_key"] for group in payload["ecdf"]["groups"]}
    assert "emergency" in ecdf_keys
    activity_keys = {group["filter_key"] for group in payload["activity"]["groups"]}
    assert "emergency" in activity_keys
