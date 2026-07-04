from __future__ import annotations

import yaml
import pandas as pd
from types import SimpleNamespace
from fastapi.testclient import TestClient

from web.backend.app.api import charts as charts_mod
from web.backend.app import experiments as experiments_mod
from web.backend.app.main import create_app


def _write_config(configs_dir, data_dir, name: str = "demo") -> tuple[str, object]:
    data_dir.mkdir()
    output = data_dir / "trajectories.parquet"
    observed = data_dir / "observed.parquet"
    profiles = data_dir / "profiles.parquet"
    config_path = configs_dir / f"{name}.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "simulation": {
                    "agents": 10,
                    "days": 2,
                    "output": str(output),
                    "granularity_minutes": 15,
                    "car_speed_kmh": 40.0,
                },
                "comparison": {"label": "before", "path": str(observed)},
                "profiles": {"enabled": True, "output": str(profiles)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return name, output


def _client_for_tmp_configs(monkeypatch, tmp_path) -> TestClient:
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    monkeypatch.setattr(experiments_mod, "CONFIGS_DIR", configs_dir)
    return TestClient(create_app())


def test_patch_experiment_persists_core_fields(monkeypatch, tmp_path):
    client = _client_for_tmp_configs(monkeypatch, tmp_path)
    exp_id, _output = _write_config(tmp_path / "configs", tmp_path / "data")

    response = client.patch(
        f"/api/experiments/{exp_id}",
        json={
            "label": "after",
            "agents": 25,
            "days": 3,
            "start_date": "2026-01-02",
            "granularity_minutes": 30,
            "car_speed_kmh": 35.5,
            "simulation_output": str(tmp_path / "data" / "new_trajectories.parquet"),
            "observed_path": str(tmp_path / "data" / "new_observed.parquet"),
            "profiles_enabled": False,
            "profiles_output": str(tmp_path / "data" / "new_profiles.parquet"),
        },
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["label"] == "after"
    assert data["params"]["agents"] == 25
    assert data["params"]["granularity_minutes"] == 30
    saved = yaml.safe_load((tmp_path / "configs" / f"{exp_id}.yaml").read_text())
    assert saved["simulation"]["output"].endswith("new_trajectories.parquet")
    assert saved["profiles"]["enabled"] is False


def test_patch_experiment_rejects_invalid_edit_without_writing(monkeypatch, tmp_path):
    client = _client_for_tmp_configs(monkeypatch, tmp_path)
    exp_id, _output = _write_config(tmp_path / "configs", tmp_path / "data")
    config_path = tmp_path / "configs" / f"{exp_id}.yaml"
    before = config_path.read_text(encoding="utf-8")

    response = client.patch(f"/api/experiments/{exp_id}", json={"agents": 0})

    assert response.status_code == 400
    assert config_path.read_text(encoding="utf-8") == before


def test_delete_run_removes_only_selected_run_files(monkeypatch, tmp_path):
    client = _client_for_tmp_configs(monkeypatch, tmp_path)
    exp_id, output = _write_config(tmp_path / "configs", tmp_path / "data")
    selected = output.with_name("trajectories_20260101T010203.parquet")
    selected_encounters = output.with_name("trajectories_20260101T010203_encounters.parquet")
    selected_moving = output.with_name("trajectories_20260101T010203_moving.parquet")
    selected_social = output.with_name("trajectories_20260101T010203_social_network.json")
    other_run = output.with_name("trajectories_20260102T010203.parquet")
    for path in (selected, selected_encounters, selected_moving, selected_social, other_run):
        path.write_text("placeholder", encoding="utf-8")

    response = client.delete(f"/api/experiments/{exp_id}/runs/20260101T010203")

    assert response.status_code == 200
    assert not selected.exists()
    assert not selected_encounters.exists()
    assert not selected_moving.exists()
    assert not selected_social.exists()
    assert other_run.exists()
    assert (tmp_path / "configs" / f"{exp_id}.yaml").exists()


def test_archive_experiment_moves_config_out_of_discovery(monkeypatch, tmp_path):
    client = _client_for_tmp_configs(monkeypatch, tmp_path)
    exp_id, _output = _write_config(tmp_path / "configs", tmp_path / "data")

    response = client.post(f"/api/experiments/{exp_id}/archive")

    assert response.status_code == 200
    assert not (tmp_path / "configs" / f"{exp_id}.yaml").exists()
    assert (tmp_path / "configs" / ".archived" / f"{exp_id}.yaml").exists()
    assert client.get("/api/experiments").json()["data"] == []


def test_unknown_experiment_and_run_return_404(monkeypatch, tmp_path):
    client = _client_for_tmp_configs(monkeypatch, tmp_path)
    exp_id, _output = _write_config(tmp_path / "configs", tmp_path / "data")

    assert client.patch("/api/experiments/missing", json={"label": "x"}).status_code == 404
    assert client.post("/api/experiments/missing/archive").status_code == 404
    assert client.delete(f"/api/experiments/{exp_id}/runs/missing").status_code == 404


def test_charts_endpoint_allows_missing_observed_path(monkeypatch, tmp_path):
    run = tmp_path / "trajectories_20260101T010203.parquet"
    pd.DataFrame(
        {
            "uid": [1],
            "datetime": pd.to_datetime(["2026-01-01 00:00"]),
            "lat": [48.85],
            "lng": [2.35],
        }
    ).to_parquet(run, index=False)
    selected = SimpleNamespace(
        run_id="20260101T010203",
        path=run,
        activities_path=tmp_path / "missing_activities.parquet",
        social_network_path=tmp_path / "missing_social.json",
    )
    experiment = SimpleNamespace(
        observed_path=tmp_path / "missing_observed.parquet",
        label="observed",
        run=lambda run_id=None: selected,
    )
    monkeypatch.setattr(charts_mod, "get_experiment", lambda exp_id: experiment)

    def fake_build(synthetic_path, observed_path, observed_label, synthetic_activities_path=None):
        assert observed_path is None
        return {
            "mode": "synthetic_only",
            "labels": {"synthetic": "synthetic"},
            "metrics": {"wasserstein": [], "jsd": [], "cpc": []},
            "ecdf": {"groups": []},
            "mobility_laws": None,
            "activity": None,
            "micro_activity_usage": None,
            "profiles": None,
            "motifs": None,
            "stvd": None,
            "social_network": None,
            "warnings": [],
        }

    monkeypatch.setattr(charts_mod, "build_comparison_payload", fake_build)

    response = charts_mod.get_charts("demo", run="20260101T010203", refresh=True)

    assert response.data["mode"] == "synthetic_only"
    assert response.data["run_id"] == "20260101T010203"
