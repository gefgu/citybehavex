from __future__ import annotations

import pytest

from citybehavex.config import apply_overrides, load_config
from citybehavex.llm import LLMConfig
from citybehavex.llm_diaries import DiariesConfig
from citybehavex.profiles.config import AgentProfilesConfig
from citybehavex.simulation import SimulationConfig


def test_load_config_expands_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("CBX_OUT", "configured.parquet")
    path = tmp_path / "config.yaml"
    path.write_text(
        """
simulation:
  tessellation: input.parquet
  output: ${CBX_OUT}
llm:
  base_url: http://localhost:8000
  api_key: ${CBX_KEY}
  model: test-model
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("CBX_KEY", "secret")
    config = load_config(str(path))
    assert config.simulation.output == "configured.parquet"
    assert config.llm.api_key == "secret"


def test_cli_overrides_config_defaults():
    model = SimulationConfig(tessellation="config.parquet", agents=10)
    updated = apply_overrides(model, {"agents": 20, "output": None})
    assert updated.agents == 20
    assert updated.output == "trajectories.parquet"


def test_simulation_config_rejects_removed_model_field():
    with pytest.raises(ValueError):
        SimulationConfig(model="legacy")


def test_simulation_config_rejects_removed_social_graph_radius():
    with pytest.raises(ValueError):
        SimulationConfig(social_graph_radius=0.5)


def test_simulation_config_accepts_bounded_social_graph_settings():
    config = SimulationConfig(social_graph_k=30, profile_graph_exact_threshold=5000)
    assert config.social_graph_k == 30
    assert config.profile_graph_exact_threshold == 5000


def test_llm_config_defaults_to_thirty_diaries():
    assert LLMConfig().diary_count == 30


def test_profiles_default_to_poi_building_location_inference():
    config = AgentProfilesConfig()
    assert config.location_inference_method == "poi_building"
    assert config.home_poi_inverse_weight == 0.5
    assert config.home_building_weight == 1.0
    assert config.work_poi_weight == 0.75
    assert config.work_building_weight == 1.0
    assert config.work_distance_model == "exponential"
    assert config.work_distance_exponential_lambda == 0.3
    assert config.work_distance_max_km == 60.0
    assert config.work_distance_density_correction_power == 1.0
    assert config.work_from_home_probability == 0.05


def test_simulation_config_rejects_tessellation_and_bbox():
    with pytest.raises(ValueError):
        SimulationConfig(
            tessellation="input.parquet",
            min_lon=0,
            min_lat=0,
            max_lon=1,
            max_lat=1,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"location_count_sigma": 0},
        {"location_count_sigma": -0.5},
        {"max_locations": 7},
        {"max_locations": 0},
    ],
)
def test_diaries_config_rejects_invalid_location_distribution(kwargs):
    with pytest.raises(ValueError):
        DiariesConfig(**kwargs)
