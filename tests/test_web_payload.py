from __future__ import annotations

import json

import pandas as pd

from citybehavex.simulation.core import social_network_sidecar_path
from web.backend.app.payload import (
    TIME_USE_CATEGORIES,
    _filter_df,
    _load_social_network_sidecar,
    _special_day_filters,
    build_comparison_payload,
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
        lambda *args, **kwargs: pd.DataFrame(
            {
                "user_id": ["u1", "u1"],
                "timestamp": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 01:00"]),
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
        lambda *args, **kwargs: pd.DataFrame(
            {
                "user_id": ["u1"],
                "timestamp": pd.to_datetime(["2026-01-01 00:00"]),
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
    assert sleep["difference_minutes"] == -90.0


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
        lambda *args, **kwargs: pd.DataFrame(
            {
                "user_id": ["u1", "u1"],
                "timestamp": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 01:00"]),
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
    all_activity = payload["activity"]["groups"][0]
    assert all_activity["transition_difference"]["matrix_mode"] == "raw"
    assert all_activity["daily_activity_difference"]["matrix_mode"] == "raw"
    assert len(all_activity["purpose"]["series"]) == 1


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
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2019-11-13 12:00", "2019-11-14 00:00", "2019-11-20 08:00", "2019-11-29 00:00"]
            ),
            "value": [1, 2, 3, 4],
        }
    )
    meta = {"key": "emergency", "kind": "date_range", "start": "2019-11-14", "end": "2019-11-28"}
    filtered = _filter_df(df, "datetime", meta)
    assert filtered["value"].tolist() == [2, 3]


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
        lambda *args, **kwargs: pd.DataFrame(
            {
                "user_id": ["u1", "u1"],
                "timestamp": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 01:00"]),
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
