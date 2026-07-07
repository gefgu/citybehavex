from __future__ import annotations

import pytest
from pathlib import Path

from citybehavex.config import apply_overrides, load_config
from citybehavex.llm import LLMConfig
from citybehavex.llm_diaries import DiariesConfig
from citybehavex.activities.config import ActivitiesConfig
from citybehavex.profiles.config import AgentProfilesConfig
from citybehavex.schedules import ScheduleConfig
from citybehavex.simulation import SimulationConfig
from citybehavex.social.config import SocialNetworkConfig


ROOT = Path(__file__).resolve().parents[1]


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


def test_repo_simulator_yaml_configs_validate():
    for path in sorted((ROOT / "configs").glob("*.yaml")):
        config = load_config(str(path))
        assert config.profiles.coherence_alignment_backend in {"none", "rerank"}


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


def test_social_network_config_accepts_bounded_social_graph_settings():
    config = SocialNetworkConfig(social_graph_k=30, profile_graph_exact_threshold=5000)
    assert config.social_graph_k == 30
    assert config.profile_graph_exact_threshold == 5000


def test_social_network_config_rejects_non_positive_settings():
    with pytest.raises(ValueError):
        SocialNetworkConfig(social_graph_k=0)
    with pytest.raises(ValueError):
        SocialNetworkConfig(degree_sigma_ln=0)
    with pytest.raises(ValueError):
        SocialNetworkConfig(home_h3_resolution=16)


def test_llm_config_defaults_to_thirty_diaries():
    assert LLMConfig().diary_count == 30


def test_schedule_alignment_config_defaults_to_embedding_backend():
    config = ScheduleConfig()
    assert config.similarity_backend == "embedding"
    assert config.alignment_base_url is None


def test_schedule_alignment_config_accepts_alignment_backend():
    config = ScheduleConfig(
        similarity_backend="alignment_model",
        alignment_base_url="http://localhost:8082",
        alignment_model="models/modernbert-schedule-aligner",
    )
    assert config.similarity_backend == "alignment_model"
    assert config.alignment_batch_size == 32


def test_activity_alignment_config_defaults_to_disabled_rerank():
    config = ActivitiesConfig()
    assert config.alignment_backend == "none"
    assert config.alignment_base_url is None
    assert config.profile_cluster_similarity_threshold == 0.94


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
    assert config.coherence_alignment_backend == "none"
    assert config.coherence_alignment_base_url is None
    assert config.coherence_profile_cluster_similarity_threshold == 0.94
    assert config.coherence_rerun_rounds == 3
    assert config.coherence_rerun_threshold == 0.6
    assert config.ownership_alignment_backend == "none"
    assert config.ownership_alignment_base_url is None
    assert config.ownership_profile_cluster_similarity_threshold == 0.94


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


def _emergency_diaries_config() -> DiariesConfig:
    return DiariesConfig(
        city_profile="shared",
        city_profile_weekday="weekday text",
        city_profile_weekend="weekend text",
        special_days=[
            {
                "name": "emergency",
                "start_date": "2019-11-14",
                "end_date": "2019-11-28",
                "city_profile": "emergency text",
            }
        ],
    )


def test_profile_for_returns_special_day_profile():
    cfg = _emergency_diaries_config()
    assert cfg.profile_for("emergency") == "emergency text"
    assert cfg.profile_for("weekday") == "weekday text"
    assert cfg.profile_for("weekend") == "weekend text"


def test_profile_for_special_day_falls_back_to_shared_profile():
    cfg = DiariesConfig(
        city_profile="shared",
        special_days=[{"name": "emergency", "start_date": "2019-11-14", "end_date": "2019-11-28"}],
    )
    assert cfg.profile_for("emergency") == "shared"


def test_day_types_for_range_includes_overlapping_special_days():
    cfg = _emergency_diaries_config()
    from datetime import date

    assert cfg.day_types_for_range(date(2019, 9, 15), date(2019, 11, 28)) == [
        "weekday",
        "weekend",
        "emergency",
    ]
    assert cfg.day_types_for_range(date(2019, 9, 15), date(2019, 10, 1)) == ["weekday", "weekend"]


def test_resolve_day_type_prefers_special_day_over_calendar():
    from datetime import date

    cfg = _emergency_diaries_config()
    # 2019-11-14 is a Thursday (a calendar weekday) but inside the emergency range.
    assert cfg.resolve_day_type(date(2019, 11, 14)) == "emergency"
    # A Saturday outside the emergency range still resolves to the calendar rule.
    assert cfg.resolve_day_type(date(2019, 9, 21)) == "weekend"
    assert cfg.resolve_day_type(date(2019, 9, 16)) == "weekday"
