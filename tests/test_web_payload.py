from __future__ import annotations

import json

import pandas as pd

from citybehavex.simulation.core import social_network_sidecar_path
from web.backend.app.payload import _load_social_network_sidecar, build_comparison_payload


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
            "activity": [0, 4],
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

    assert payload["activity"] is not None
    assert payload["activity"]["purpose"]["categories"] == ["HOME", "WORK", "OTHER"]
    assert any("observed has no explicit purpose column" in w for w in payload["warnings"])
    assert payload["micro_activity_usage"] is not None
    names = {series["name"] for series in payload["micro_activity_usage"]["series"]}
    assert {"sleep", "paid_work"}.issubset(names)
