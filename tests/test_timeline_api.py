from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd

from web.backend.app.api import timeline as timeline_mod


def test_timeline_crp_endpoint_maps_display_uid_to_zero_based_agent(monkeypatch, tmp_path):
    crp_path = tmp_path / "run_crp.parquet"
    pd.DataFrame(
        {
            "agent": [0, 1],
            "diary_id": ["wd-0", "wd-1"],
            "day_type": ["weekday", "weekday"],
            "sim": [0.75, 0.25],
            "usage_count": [4, 1],
            "T_a": [0.3, 0.6],
            "alpha_a": [0.1, 0.2],
        }
    ).to_parquet(crp_path, index=False)
    run = SimpleNamespace(run_id="demo", crp_path=crp_path)
    experiment = SimpleNamespace(run=lambda run_id=None: run)
    monkeypatch.setattr(timeline_mod, "get_experiment", lambda exp_id: experiment)

    response = timeline_mod.get_timeline_agent_crp("demo", 1)

    assert response.data["diaries"][0]["diary_id"] == "wd-0"
    assert response.data["T_a"] == 0.3


def test_profile_artifact_warning_mentions_partial_profile_file(tmp_path):
    profiles_path = tmp_path / "profiles.parquet"
    pd.DataFrame(
        {
            "uid": [1, 2],
            "gender": ["female", "male"],
            "name": ["Ana", "Theo"],
            "age": [30, 40],
            "education": ["bachelor", "master or above"],
            "health": [4, 5],
            "household": ["single", "couple"],
            "job": ["professional", "service or sales worker"],
            "has_car": [True, False],
            "has_bike": [False, True],
            "home_tile": [10, 11],
            "work_tile": [12, 13],
        }
    ).to_parquet(profiles_path, index=False)

    warning = timeline_mod._profile_artifact_warning(profiles_path, 64, 500)

    assert warning == "profile artifact has 2 rows for 500 agents; no profile row for uid 64"


def test_timeline_social_endpoint_enriches_friends_and_encounters(monkeypatch, tmp_path):
    social_path = tmp_path / "run_social_network.json"
    social_path.write_text(
        json.dumps(
            {
                "kind": "initial_profile_similarity",
                "node_count": 3,
                "edge_count": 2,
                "layout": "profile_svd",
                "directed": True,
                "social_graph_k": 2,
                "nodes": [[0, 0, 4, 1], [1, 0, 4, 2], [2, 0, 4, 3]],
                "edges": [[0, 1, 0.8], [1, 0, 0.7]],
                "degrees": [1, 1, 0],
            }
        ),
        encoding="utf-8",
    )
    encounters_path = tmp_path / "run_encounters.parquet"
    pd.DataFrame(
        {
            "agent": [1, 2],
            "contact": [2, 1],
            "tile": [10, 10],
            "ts": [1, 2],
        }
    ).to_parquet(encounters_path, index=False)
    profiles_path = tmp_path / "profiles.parquet"
    pd.DataFrame(
        {
            "uid": [2],
            "gender": ["male"],
            "name": ["Theo"],
            "age": [40],
            "education": ["master or above"],
            "health": [5],
            "household": ["couple"],
            "job": ["service or sales worker"],
            "has_car": [False],
            "has_bike": [True],
            "home_tile": [11],
            "work_tile": [13],
        }
    ).to_parquet(profiles_path, index=False)
    run = SimpleNamespace(
        run_id="demo",
        social_network_path=social_path,
        encounters_path=encounters_path,
    )
    experiment = SimpleNamespace(
        run=lambda run_id=None: run,
        profiles_path=profiles_path,
        params={
            "social_graph_k": 2,
            "rho": 0.3,
            "gamma": 0.2,
            "alpha": 0.1,
            "dt_update_mob_sim_hours": 24,
            "indipendency_window_hours": 72,
        },
    )
    monkeypatch.setattr(timeline_mod, "get_experiment", lambda exp_id: experiment)

    response = timeline_mod.get_timeline_agent_social("demo", 1)

    assert response.data["parameters"]["degree"] == 1
    assert response.data["parameters"]["rho"] == 0.3
    assert response.data["friends"][0]["uid"] == 2
    assert response.data["friends"][0]["name"] == "Theo"
    assert response.data["friends"][0]["social_strength"] == 0.8
    assert response.data["friends"][0]["encounter_count"] == 2
    assert response.data["friends"][0]["reciprocated"] is True


def test_timeline_social_endpoint_warns_for_missing_sidecars(monkeypatch, tmp_path):
    run = SimpleNamespace(
        run_id="demo",
        social_network_path=tmp_path / "missing_social.json",
        encounters_path=tmp_path / "missing_encounters.parquet",
    )
    experiment = SimpleNamespace(
        run=lambda run_id=None: run,
        profiles_path=None,
        params={"social_graph_k": 2},
    )
    monkeypatch.setattr(timeline_mod, "get_experiment", lambda exp_id: experiment)

    response = timeline_mod.get_timeline_agent_social("demo", 1)

    assert response.data["friends"] == []
    assert "no social network sidecar available for this run" in response.data["warnings"]
    assert "no encounters data available for this experiment" in response.data["warnings"]
