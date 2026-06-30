from __future__ import annotations

import numpy as np
import pandas as pd

import citybehavex._core as core
from citybehavex.activities import (
    N_ACTIVITIES,
    activity_duration_arrays,
    build_eligibility_csr,
)
from citybehavex.simulation_core import simulate_agents

_SLOT = 900
_SPEED = 50.0


def _run(lats, lngs, abs_locs, slot_times, *, end_ts, rho=1.0, gamma=0.21):
    diary_ts = np.asarray(slot_times, dtype=np.int64)
    diary_loc = np.asarray(abs_locs, dtype=np.int32)
    starts = np.array([0], dtype=np.int64)
    ends = np.array([len(diary_ts)], dtype=np.int64)
    # Returns 11-tuple: agents, lats, lngs, arrival, departure, duration,
    #                   enc_agent, enc_contact, enc_tile, enc_ts, activity
    return core.simulation_core_simulate_agents(
        np.asarray(lats, dtype=float),
        np.asarray(lngs, dtype=float),
        np.ones(len(lats), dtype=float),
        np.empty(0, dtype=np.float64),
        np.array([0, 0], dtype=np.int64),
        np.empty(0, dtype=np.int64),
        diary_ts,
        diary_loc,
        starts,
        ends,
        rho,
        gamma,
        0.0,
        0,
        end_ts,
        1800,
        3600,
        _SLOT,
        _SPEED,
        1,
        42,
        np.array([0], dtype=np.int64),
        False,
    )


def test_simulation_core_long_trip_is_centered_on_slot_boundary():
    lats = [48.8566, 48.95]
    lngs = [2.3522, 2.55]
    slot_times = [0, 8 * 3600, 18 * 3600]
    abs_locs = [0, 1, 0]
    ag, _, _, arr, dep, dur, *_ = _run(lats, lngs, abs_locs, slot_times, end_ts=86400)

    assert len(ag) == 3
    arr, dep, dur = np.asarray(arr), np.asarray(dep), np.asarray(dur)
    assert np.all(np.diff(arr) >= 0)
    assert np.all((dep - arr) >= 0)
    assert dur[1] > _SLOT
    assert dep[0] < 8 * 3600
    assert arr[1] > 8 * 3600
    assert abs((8 * 3600 - dep[0]) - (arr[1] - 8 * 3600)) <= 1


def test_simulation_core_short_trip_arrives_within_the_slot():
    lats = [48.8566, 48.8580]
    lngs = [2.3522, 2.3540]
    slot_times = [0, 8 * 3600, 18 * 3600]
    abs_locs = [0, 1, 0]
    _, _, _, arr, dep, dur, *_ = _run(lats, lngs, abs_locs, slot_times, end_ts=86400)
    arr, dep, dur = np.asarray(arr), np.asarray(dep), np.asarray(dur)

    assert dur[1] < _SLOT
    assert dep[0] == 8 * 3600
    assert 8 * 3600 <= arr[1] < 8 * 3600 + _SLOT


def test_simulation_core_trip_durations_are_off_the_hourly_grid():
    lats = [48.8566, 48.95]
    lngs = [2.3522, 2.55]
    slot_times = [0, 8 * 3600, 18 * 3600]
    abs_locs = [0, 1, 0]
    _, _, _, arr, *_ = _run(lats, lngs, abs_locs, slot_times, end_ts=86400)
    assert any(int(a) % _SLOT != 0 for a in np.asarray(arr))


def test_simulation_core_keeps_one_location_for_continuous_abstract_block():
    lats = [48.8566, 48.8580, 48.8610, 48.8640]
    lngs = [2.3522, 2.3540, 2.3580, 2.3620]
    slot_times = [0, 8 * 3600, 8 * 3600 + _SLOT, 8 * 3600 + 2 * _SLOT, 18 * 3600]
    abs_locs = [0, 1, 1, 1, 0]

    ag, *_ = _run(
        lats,
        lngs,
        abs_locs,
        slot_times,
        end_ts=86400,
        rho=1.0,
        gamma=0.0,
    )

    assert len(ag) == 3


def test_simulation_core_reuses_same_day_location_for_abstract_code():
    lats = [48.8566, 48.8580, 48.8610, 48.8640]
    lngs = [2.3522, 2.3540, 2.3580, 2.3620]
    slot_times = [
        0,
        8 * 3600,
        8 * 3600 + _SLOT,
        10 * 3600,
        14 * 3600,
        14 * 3600 + _SLOT,
        18 * 3600,
    ]
    abs_locs = [0, 1, 1, 0, 1, 1, 0]

    _, out_lats, out_lngs, *_ = _run(
        lats,
        lngs,
        abs_locs,
        slot_times,
        end_ts=86400,
        rho=1.0,
        gamma=0.0,
    )

    out_lats = np.asarray(out_lats)
    out_lngs = np.asarray(out_lngs)
    assert len(out_lats) == 5
    assert out_lats[1] == out_lats[3]
    assert out_lngs[1] == out_lngs[3]


def test_simulate_agents_returns_trip_columns():
    tess = pd.DataFrame(
        {
            "tile_id": [0, 1],
            "lat": [48.8566, 48.95],
            "lng": [2.3522, 2.55],
            "relevance": [1.0, 1.0],
        }
    )
    diary_arrays = (
        np.array([0, 8 * 3600, 18 * 3600], dtype=np.int64),
        np.array([0, 1, 0], dtype=np.int32),
        np.array([0], dtype=np.int64),
        np.array([3], dtype=np.int64),
    )
    df, encounters = simulate_agents(
        tess,
        "relevance",
        diary_arrays,
        start_ts=0,
        end_ts=86400,
        slot_seconds=_SLOT,
        car_speed_kmh=_SPEED,
        n_agents=1,
        random_state=42,
    )
    for column in (
        "uid",
        "datetime",
        "lat",
        "lng",
        "arrival",
        "departure",
        "trip_duration_minutes",
        "dwell_minutes",
    ):
        assert column in df.columns
    assert (df["dwell_minutes"] >= 0).all()
    assert (df["trip_duration_minutes"] >= 0).all()
    assert pd.api.types.is_datetime64_any_dtype(df["arrival"])


def test_simulate_agents_encounters_has_expected_columns():
    tess = pd.DataFrame(
        {
            "tile_id": [0, 1],
            "lat": [48.8566, 48.95],
            "lng": [2.3522, 2.55],
            "relevance": [1.0, 1.0],
        }
    )
    diary_arrays = (
        np.array([0, 8 * 3600, 18 * 3600], dtype=np.int64),
        np.array([0, 1, 0], dtype=np.int32),
        np.array([0, 0], dtype=np.int64),
        np.array([3, 3], dtype=np.int64),
    )
    _, encounters = simulate_agents(
        tess,
        "relevance",
        diary_arrays,
        start_ts=0,
        end_ts=86400,
        slot_seconds=_SLOT,
        car_speed_kmh=_SPEED,
        n_agents=2,
        random_state=42,
    )
    assert isinstance(encounters, pd.DataFrame)
    for col in ("agent", "contact", "tile", "ts"):
        assert col in encounters.columns


# --- Phase 4: activity CRP + early/late exit timing -------------------------

def _diary_arrays_single(abs_locs, slot_times):
    return (
        np.asarray(slot_times, dtype=np.int64),
        np.asarray(abs_locs, dtype=np.int32),
        np.array([0], dtype=np.int64),
        np.array([len(slot_times)], dtype=np.int64),
    )


def test_activity_column_present_when_enabled():
    """activities.enabled → trajectory DataFrame has 'activity' column."""
    tess = pd.DataFrame({
        "tile_id": [0, 1],
        "lat": [48.8566, 48.95],
        "lng": [2.3522, 2.55],
        "relevance": [1.0, 1.0],
    })
    diary_arrays = _diary_arrays_single([0, 1, 0], [0, 8 * 3600, 18 * 3600])
    act_dur_mu, act_dur_sigma = activity_duration_arrays()
    purpose_act_starts, purpose_acts = build_eligibility_csr()
    df, _ = simulate_agents(
        tess, "relevance", diary_arrays,
        start_ts=0, end_ts=86400,
        slot_seconds=_SLOT, car_speed_kmh=_SPEED,
        n_agents=1, random_state=42,
        act_dur_mu=act_dur_mu,
        act_dur_sigma=act_dur_sigma,
        purpose_act_starts=purpose_act_starts,
        purpose_acts=purpose_acts,
    )
    assert "activity" in df.columns
    assert (df["activity"] >= 0).all()
    assert (df["activity"] < N_ACTIVITIES).all()


def test_activity_column_absent_when_disabled():
    """Without activity params, 'activity' column is all zeros (disabled)."""
    tess = pd.DataFrame({
        "tile_id": [0, 1],
        "lat": [48.8566, 48.95],
        "lng": [2.3522, 2.55],
        "relevance": [1.0, 1.0],
    })
    diary_arrays = _diary_arrays_single([0, 1, 0], [0, 8 * 3600, 18 * 3600])
    df, _ = simulate_agents(
        tess, "relevance", diary_arrays,
        start_ts=0, end_ts=86400,
        slot_seconds=_SLOT, car_speed_kmh=_SPEED,
        n_agents=1, random_state=42,
    )
    # No activity params → all zeros
    assert (df["activity"] == 0).all()


def test_activities_produce_non_trivial_dwell():
    """With activity CRP, departure should differ from slot timestamps for at least some records."""
    tess = pd.DataFrame({
        "tile_id": [0, 1],
        "lat": [48.8566, 48.95],
        "lng": [2.3522, 2.55],
        "relevance": [1.0, 1.0],
    })
    # 3 days of HOME/WORK/HOME pattern
    slots, locs = [], []
    for d in range(3):
        base = d * 86400
        slots += [base, base + 9 * 3600, base + 17 * 3600]
        locs += [0, 1, 0]
    diary_arrays = _diary_arrays_single(locs, slots)
    act_dur_mu, act_dur_sigma = activity_duration_arrays()
    purpose_act_starts, purpose_acts = build_eligibility_csr()
    df, _ = simulate_agents(
        tess, "relevance", diary_arrays,
        start_ts=0, end_ts=3 * 86400,
        slot_seconds=_SLOT, car_speed_kmh=_SPEED,
        n_agents=1, random_state=7,
        act_dur_mu=act_dur_mu,
        act_dur_sigma=act_dur_sigma,
        purpose_act_starts=purpose_act_starts,
        purpose_acts=purpose_acts,
    )
    assert (df["dwell_minutes"] >= 0).all()
    # dwell_minutes should vary (not all zero)
    assert df["dwell_minutes"].std() > 0


def test_activities_catalog_coverage():
    """Every purpose code 0-6 has at least one eligible activity."""
    purpose_act_starts, purpose_acts = build_eligibility_csr()
    assert len(purpose_act_starts) == 8  # 7 purposes + sentinel
    for p in range(7):
        n_eligible = int(purpose_act_starts[p + 1] - purpose_act_starts[p])
        assert n_eligible > 0, f"Purpose {p} has no eligible activities"
