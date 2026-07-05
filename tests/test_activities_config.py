from __future__ import annotations

import numpy as np
import pytest

from citybehavex.activities import activity_duration_arrays
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
