from __future__ import annotations

import numpy as np

from citybehavex.profiles import AgentProfile, AgentProfilesConfig, reroll_profile_demographics
from citybehavex.simulation.runner import _coherence_rerun_indices


def _profile(uid: int = 1) -> AgentProfile:
    return AgentProfile(
        uid=uid,
        gender="female",
        name="Alice",
        age=35,
        education="bachelor",
        health=4,
        household="living alone",
        job="professional",
        has_car=True,
        has_bike=False,
        home_tile=10,
        work_tile=20,
    )


def test_hybrid_coherence_rerun_indices_are_deterministic():
    scores = np.array([0.2, 0.7, 1.0], dtype=np.float64)

    first = _coherence_rerun_indices(scores, 0.6, np.random.default_rng(7))
    second = _coherence_rerun_indices(scores, 0.6, np.random.default_rng(7))

    assert first.tolist() == second.tolist()
    assert 0 in first
    assert 2 not in first


def test_reroll_profile_demographics_preserves_home_work_and_transport():
    profiles = [_profile(1), _profile(2)]
    rerolled = reroll_profile_demographics(
        profiles,
        [0],
        AgentProfilesConfig(),
        np.random.default_rng(12),
    )

    assert rerolled[0].uid == profiles[0].uid
    assert rerolled[0].home_tile == profiles[0].home_tile
    assert rerolled[0].work_tile == profiles[0].work_tile
    assert rerolled[0].has_car == profiles[0].has_car
    assert rerolled[0].has_bike == profiles[0].has_bike
    assert rerolled[1] == profiles[1]
