from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from citybehavex.config import CityBehavExConfig
from citybehavex.profiles import (
    AgentProfilesConfig,
    apply_profile_overrides,
    generate_profiles,
    load_profile_overrides,
    reroll_profile_demographics,
    resolve_profile_override_tiles,
    validate_profile_override_tiles,
)
from citybehavex.simulation import profile_pipeline
from citybehavex.simulation.profile_pipeline import maybe_build_profiles


def _tessellation() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tile_id": ["work_1", "work_2", "home_1", "home_2"],
            "lat": [0.1, 0.2, 0.0, 0.0],
            "lng": [0.0, 0.0, 0.1, 0.2],
            "purpose": ["WORK", "WORK", "HOME", "HOME"],
            "relevance": [2.0, 1.0, 1.0, 1.0],
        }
    )


def test_partial_json_overrides_generate_missing_fields_and_support_sparse_uids(tmp_path) -> None:
    path = tmp_path / "partial_profiles.json"
    path.write_text(
        json.dumps(
            [
                {"uid": 1, "home_tile": "home_1", "work_tile": "work_1"},
                {"uid": 3, "home_tile": "home_2"},
            ]
        ),
        encoding="utf-8",
    )
    config = CityBehavExConfig.model_validate(
        {
            "simulation": {"agents": 3, "random_state": 4},
            "profiles": {
                "enabled": True,
                "profiles_path": str(path),
                "output": str(tmp_path / "generated_profiles.parquet"),
            },
        }
    )

    profiles = maybe_build_profiles(config, _tessellation(), "relevance")

    assert profiles is not None
    assert profiles[0].home_tile == 2
    assert profiles[0].work_tile == 0
    assert profiles[2].home_tile == 3
    assert profiles[1].uid == 2
    assert all(profile.name and profile.job for profile in profiles)


def test_partial_parquet_overrides_normalize_nulls(tmp_path) -> None:
    path = tmp_path / "partial_profiles.parquet"
    pd.DataFrame(
        {
            "uid": [1, 2],
            "home_tile": [2, 3],
            "work_tile": [0, None],
        }
    ).to_parquet(path, index=False)

    overrides = load_profile_overrides(str(path), 2)

    assert overrides == {
        1: {"uid": 1, "home_tile": 2, "work_tile": 0},
        2: {"uid": 2, "home_tile": 3},
    }


def test_string_tile_ids_resolve_to_runtime_row_indices() -> None:
    resolved = resolve_profile_override_tiles(
        {
            1: {"uid": 1, "home_tile": "home_1", "work_tile": "work_2"},
            2: {"uid": 2, "home_tile": 2, "work_tile": "work_1"},
        },
        _tessellation(),
    )

    assert resolved == {
        1: {"uid": 1, "home_tile": 2, "work_tile": 1},
        2: {"uid": 2, "home_tile": 2, "work_tile": 0},
    }


def test_string_tile_ids_reject_unknown_ids() -> None:
    with pytest.raises(ValueError, match="unknown work_tile ID"):
        resolve_profile_override_tiles(
            {1: {"uid": 1, "work_tile": "missing-poi-id"}},
            _tessellation(),
        )


@pytest.mark.parametrize(
    ("records", "message"),
    [
        ([{"uid": 0, "home_tile": 1}], "uid 0"),
        ([{"uid": 1, "home_tile": 1}, {"uid": 1, "work_tile": 2}], "duplicate"),
        ([{"uid": 1, "not_a_profile_field": 1}], "invalid profile override"),
    ],
)
def test_profile_override_loader_rejects_invalid_records(tmp_path, records, message) -> None:
    path = tmp_path / "partial_profiles.json"
    path.write_text(json.dumps(records), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_profile_overrides(str(path), 2)


def test_profile_override_tiles_validate_runtime_tessellation_bounds() -> None:
    with pytest.raises(ValueError, match="0..3"):
        validate_profile_override_tiles({1: {"home_tile": 4}}, 4)
    with pytest.raises(ValueError, match="non-integer"):
        validate_profile_override_tiles({1: {"work_tile": True}}, 4)


def test_provided_fields_are_preserved_when_coherence_rerolls() -> None:
    profiles = generate_profiles(
        1,
        AgentProfilesConfig(),
        np.random.default_rng(3),
        _tessellation(),
        "relevance",
    )
    overridden, protected = apply_profile_overrides(
        profiles,
        {1: {"uid": 1, "gender": "female", "age": 61, "home_tile": 3, "work_tile": 0}},
    )

    rerolled = reroll_profile_demographics(
        overridden,
        [0],
        AgentProfilesConfig(),
        np.random.default_rng(7),
        protected_fields=protected,
    )

    assert rerolled[0].gender == "female"
    assert rerolled[0].age == 61
    assert rerolled[0].home_tile == 3
    assert rerolled[0].work_tile == 0


def test_vehicle_alignment_only_fills_unsupplied_fields(monkeypatch, tmp_path) -> None:
    profiles = generate_profiles(
        1,
        AgentProfilesConfig(),
        np.random.default_rng(3),
        _tessellation(),
        "relevance",
    )
    overridden, protected = apply_profile_overrides(
        profiles,
        {1: {"uid": 1, "has_car": True, "bike_ownership_score": 0.9}},
    )
    config = CityBehavExConfig.model_validate(
        {
            "profiles": {
                "enabled": True,
                "output": str(tmp_path / "profiles.parquet"),
                "ownership_alignment_backend": "rerank",
                "ownership_alignment_base_url": "http://example.invalid",
            }
        }
    )
    monkeypatch.setattr(profile_pipeline, "embed_profiles", lambda *_args: None)
    monkeypatch.setattr(
        profile_pipeline,
        "cluster_profile_embeddings",
        lambda narratives, *_args: type(
            "Clusters", (), {"narratives": narratives, "labels": np.array([0], dtype=np.int64)}
        )(),
    )
    monkeypatch.setattr(
        profile_pipeline,
        "score_vehicle_ownership_alignment",
        lambda *_args, **_kwargs: (np.array([[0.0, 0.25]]), pd.DataFrame()),
    )
    monkeypatch.setattr(
        profile_pipeline,
        "expand_vehicle_scores",
        lambda scores, _clusters: scores,
    )

    updated = profile_pipeline._apply_vehicle_ownership_alignment(
        overridden, config, protected
    )

    assert updated[0].has_car is True
    assert updated[0].bike_ownership_score == 0.9
    assert updated[0].car_ownership_score == 0.0
    assert updated[0].has_bike is False
