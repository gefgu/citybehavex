from __future__ import annotations

import types

import numpy as np
import pytest
from pydantic import ValidationError

from citybehavex.activities import ProfileClusters, activity_duration_arrays, build_catalog
from citybehavex.activities.poi_semantic import (
    UNKNOWN_SEMANTIC_CLUSTER,
    build_poi_semantic_activity_data,
    load_poi_activity_mask,
    semantic_cluster_for_category,
)
from citybehavex.config.root import CityBehavExConfig
from citybehavex.simulation.runner import _build_activity_data


@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
def test_act_dur_scale_shifts_mean_duration_uniformly(scale: float) -> None:
    config = CityBehavExConfig()
    config.activities.enabled = True
    config.activities.act_dur_scale = scale

    _, act_dur_mu, act_dur_sigma, *_ = _build_activity_data(config)

    base_mu, base_sigma = activity_duration_arrays()
    assert np.allclose(np.exp(act_dur_mu), np.exp(base_mu) * scale)
    assert np.array_equal(act_dur_sigma, base_sigma)


def test_act_dur_sigma_scale_leaves_mu_untouched() -> None:
    config = CityBehavExConfig()
    config.activities.enabled = True
    config.activities.act_dur_sigma_scale = 1.5

    _, act_dur_mu, act_dur_sigma, *_ = _build_activity_data(config)

    base_mu, base_sigma = activity_duration_arrays()
    assert np.array_equal(act_dur_mu, base_mu)
    assert np.allclose(act_dur_sigma, base_sigma * 1.5)


def test_per_activity_duration_scale_shifts_only_named_activity() -> None:
    config = CityBehavExConfig(
        activities={
            "enabled": True,
            "durations": {
                "sleep": {"scale": 2.0},
            },
        }
    )

    _, act_dur_mu, act_dur_sigma, *_ = _build_activity_data(config)

    base_mu, base_sigma = activity_duration_arrays()
    sleep_idx = {activity.name: activity.idx for activity in build_catalog()}["sleep"]
    expected_mu = base_mu.copy()
    expected_mu[sleep_idx] += np.log(2.0)
    assert np.allclose(act_dur_mu, expected_mu)
    assert np.array_equal(act_dur_sigma, base_sigma)


def test_per_activity_mu_replacement_is_then_scaled() -> None:
    config = CityBehavExConfig(
        activities={
            "enabled": True,
            "durations": {
                "sleep": {"mu_ln": 1.0, "scale": 2.0, "sigma_ln": 0.2, "sigma_scale": 1.5},
            },
        }
    )

    _, act_dur_mu, act_dur_sigma, *_ = _build_activity_data(config)

    base_mu, base_sigma = activity_duration_arrays()
    sleep_idx = {activity.name: activity.idx for activity in build_catalog()}["sleep"]
    expected_mu = base_mu.copy()
    expected_sigma = base_sigma.copy()
    expected_mu[sleep_idx] = 1.0 + np.log(2.0)
    expected_sigma[sleep_idx] = 0.2 * 1.5
    assert np.allclose(act_dur_mu, expected_mu)
    assert np.allclose(act_dur_sigma, expected_sigma)


def test_unknown_activity_duration_override_is_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown activity name"):
        CityBehavExConfig(
            activities={
                "enabled": True,
                "durations": {
                    "napquest": {"scale": 2.0},
                },
            }
        )


def test_poi_semantic_mapping_and_mask_package_data() -> None:
    data = build_poi_semantic_activity_data()
    assert semantic_cluster_for_category("coffee_shop", data.category_to_cluster) == "food_drink"
    assert semantic_cluster_for_category(float("nan"), data.category_to_cluster) == UNKNOWN_SEMANTIC_CLUSTER
    assert data.cluster_to_id[UNKNOWN_SEMANTIC_CLUSTER] == 0

    mask = load_poi_activity_mask()
    assert not mask["travel"].any()
    assert not mask["commute"].any()


def test_build_activity_data_calls_activity_alignment_without_probe_inputs(monkeypatch):
    captured: dict = {}

    def fake_score_activity_alignment(narratives, diaries, config):
        captured["args"] = (narratives, diaries, config)
        return None

    monkeypatch.setattr(
        "citybehavex.simulation.runner.score_activity_alignment",
        fake_score_activity_alignment,
    )

    config = CityBehavExConfig()
    config.activities.enabled = True
    config.activities.alignment_backend = "rerank"
    config.activities.alignment_base_url = "http://tei.local"

    # A minimal stand-in for DiaryBank: _build_activity_data only reads
    # `bank.diaries` before forwarding it untouched to score_activity_alignment.
    bank = types.SimpleNamespace(diaries=["placeholder-diary"])
    clusters = ProfileClusters(
        labels=np.array([0], dtype=np.int64),
        narratives=["profile"],
        representative_indices=np.array([0], dtype=np.int64),
    )

    _build_activity_data(config, bank=bank, profile_clusters=clusters)
    assert captured["args"][0] == ["profile"]
    assert captured["args"][1] == ["placeholder-diary"]
    assert captured["args"][2] is config.activities
