from __future__ import annotations

import numpy as np
import pandas as pd

import citybehavex._core as core
from citybehavex.activities import (
    N_ACTIVITIES,
    activity_duration_arrays,
    build_eligibility_csr,
)
from citybehavex.simulation.core import simulate_agents

_SLOT = 900
_SPEED = 50.0


def _run(
    lats,
    lngs,
    abs_locs,
    slot_times,
    *,
    end_ts,
    rho=1.0,
    gamma=0.21,
    relevances=None,
    act_dur_mu=None,
    act_dur_sigma=None,
    purpose_act_starts=None,
    purpose_acts=None,
):
    diary_ts = np.asarray(slot_times, dtype=np.int64)
    diary_loc = np.asarray(abs_locs, dtype=np.int32)
    starts = np.array([0], dtype=np.int64)
    ends = np.array([len(diary_ts)], dtype=np.int64)
    rels = np.ones(len(lats), dtype=float) if relevances is None else np.asarray(relevances, dtype=float)
    # Returns a 3-tuple of tuples: (10 trip arrays), (7 path arrays), (6 activity arrays).
    # Trip: agents, lats, lngs, arrival, departure, duration,
    #       enc_agent, enc_contact, enc_tile, enc_ts
    # Paths: stop_id, path_agent, path_stop_id, path_seq, path_lat, path_lng, path_t
    # Activities: act_agent, act_stop_id, act_seq, act_activity, act_arrival, act_departure
    return core.simulation_core_simulate_agents(
        latitudes=np.asarray(lats, dtype=float),
        longitudes=np.asarray(lngs, dtype=float),
        relevances=rels,
        distances=np.empty(0, dtype=np.float64),
        neighbor_starts=np.array([0, 0], dtype=np.int64),
        neighbors=np.empty(0, dtype=np.int64),
        diary_timestamps=diary_ts,
        diary_abs_locs=diary_loc,
        diary_starts=starts,
        diary_ends=ends,
        rho=rho,
        gamma=gamma,
        alpha=0.0,
        start_ts=0,
        end_ts=end_ts,
        indipendency_window_s=1800,
        dt_update_mob_sim_s=3600,
        slot_seconds=_SLOT,
        car_speed_kmh=_SPEED,
        n_agents=1,
        master_seed=42,
        starting_locs=np.array([0], dtype=np.int64),
        starting_locs_mode_relevance=False,
        act_dur_mu=act_dur_mu,
        act_dur_sigma=act_dur_sigma,
        purpose_act_starts=purpose_act_starts,
        purpose_acts=purpose_acts,
    )


def test_simulation_core_long_trip_is_centered_on_slot_boundary():
    lats = [48.8566, 48.95]
    lngs = [2.3522, 2.55]
    slot_times = [0, 8 * 3600, 18 * 3600]
    abs_locs = [0, 1, 0]
    trip, _, _ = _run(lats, lngs, abs_locs, slot_times, end_ts=86400)
    ag, _, _, arr, dep, dur, *_ = trip

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
    trip, _, _ = _run(lats, lngs, abs_locs, slot_times, end_ts=86400)
    _, _, _, arr, dep, dur, *_ = trip
    arr, dep, dur = np.asarray(arr), np.asarray(dep), np.asarray(dur)

    assert dur[1] < _SLOT
    assert dep[0] == 8 * 3600
    assert 8 * 3600 <= arr[1] < 8 * 3600 + _SLOT


def test_simulation_core_trip_durations_are_off_the_hourly_grid():
    lats = [48.8566, 48.95]
    lngs = [2.3522, 2.55]
    slot_times = [0, 8 * 3600, 18 * 3600]
    abs_locs = [0, 1, 0]
    trip, _, _ = _run(lats, lngs, abs_locs, slot_times, end_ts=86400)
    _, _, _, arr, *_ = trip
    assert any(int(a) % _SLOT != 0 for a in np.asarray(arr))


def test_simulation_core_keeps_one_location_for_continuous_abstract_block():
    lats = [48.8566, 48.8580, 48.8610, 48.8640]
    lngs = [2.3522, 2.3540, 2.3580, 2.3620]
    slot_times = [0, 8 * 3600, 8 * 3600 + _SLOT, 8 * 3600 + 2 * _SLOT, 18 * 3600]
    abs_locs = [0, 1, 1, 1, 0]

    trip, _, _ = _run(
        lats,
        lngs,
        abs_locs,
        slot_times,
        end_ts=86400,
        rho=1.0,
        gamma=0.0,
    )

    ag = trip[0]
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

    trip, _, _ = _run(
        lats,
        lngs,
        abs_locs,
        slot_times,
        end_ts=86400,
        rho=1.0,
        gamma=0.0,
    )

    out_lats = np.asarray(trip[1])
    out_lngs = np.asarray(trip[2])
    assert len(out_lats) == 5
    assert out_lats[1] == out_lats[3]
    assert out_lngs[1] == out_lngs[3]


def test_same_physical_location_across_abstract_codes_yields_one_stop():
    """Different abstract-location codes that resolve (e.g. via preferential
    return, rho=0) to the agent's *current* physical tile must not fragment
    the stop table -- a new stop row only appears on a real relocation."""
    lats = [48.8566, 48.8580, 48.8700]
    lngs = [2.3522, 2.3540, 2.4000]
    slot_times = [0, 8 * 3600, 12 * 3600, 18 * 3600]
    abs_locs = [0, 1, 2, 0]

    trip, _, _ = _run(
        lats,
        lngs,
        abs_locs,
        slot_times,
        end_ts=86400,
        rho=0.0,
        gamma=0.0,
    )
    ag, out_lats, _, arr, dep, *_ = trip

    assert len(ag) == 1
    assert arr[0] == 0
    assert dep[0] == 86400
    assert out_lats[0] == lats[0]


def test_same_physical_location_still_samples_multiple_activities():
    """Even though the stop table collapses to one row, each abstract-
    location change that lands back on the same tile should still record its
    own micro-activity in the separate activities table, in order, with
    non-overlapping [arrival, departure) windows spanning the whole stay."""
    lats = [48.8566, 48.8580, 48.8700]
    lngs = [2.3522, 2.3540, 2.4000]
    slot_times = [0, 8 * 3600, 12 * 3600, 18 * 3600]
    abs_locs = [0, 1, 2, 0]
    act_dur_mu, act_dur_sigma = activity_duration_arrays()
    purpose_act_starts, purpose_acts = build_eligibility_csr()

    trip, _, acts = _run(
        lats,
        lngs,
        abs_locs,
        slot_times,
        end_ts=86400,
        rho=0.0,
        gamma=0.0,
        act_dur_mu=act_dur_mu,
        act_dur_sigma=act_dur_sigma,
        purpose_act_starts=purpose_act_starts,
        purpose_acts=purpose_acts,
    )
    ag = trip[0]
    assert len(ag) == 1  # still one physical stop

    act_agent, act_stop_id, act_seq, act_activity, act_arrival, act_departure = (
        np.asarray(a) for a in acts
    )
    # One activity per abstract-location-change event: the initial bootstrap
    # plus the three diary moves (0 -> 1 -> 2 -> 0), all landing on tile 0.
    assert len(act_agent) == 4
    assert (act_stop_id == 0).all()
    assert list(act_seq) == [0, 1, 2, 3]
    assert (act_activity >= 0).all()
    assert (act_activity < N_ACTIVITIES).all()
    # Contiguous, non-overlapping, covering the whole simulated window.
    assert act_arrival[0] == 0
    assert act_departure[-1] == 86400
    assert list(act_arrival[1:]) == list(act_departure[:-1])
    assert (act_departure >= act_arrival).all()


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
    df, encounters, moving, activities = simulate_agents(
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
    assert "activity" not in df.columns
    assert (df["dwell_minutes"] >= 0).all()
    assert (df["trip_duration_minutes"] >= 0).all()
    assert pd.api.types.is_datetime64_any_dtype(df["arrival"])
    assert isinstance(moving, pd.DataFrame)
    assert isinstance(activities, pd.DataFrame)


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
    _, encounters, _, _ = simulate_agents(
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
    """activities.enabled -> a non-empty activities DataFrame is returned,
    with the stop table itself left untouched (no inline activity column)."""
    tess = pd.DataFrame({
        "tile_id": [0, 1],
        "lat": [48.8566, 48.95],
        "lng": [2.3522, 2.55],
        "relevance": [1.0, 1.0],
    })
    diary_arrays = _diary_arrays_single([0, 1, 0], [0, 8 * 3600, 18 * 3600])
    act_dur_mu, act_dur_sigma = activity_duration_arrays()
    purpose_act_starts, purpose_acts = build_eligibility_csr()
    df, _, _, activities = simulate_agents(
        tess, "relevance", diary_arrays,
        start_ts=0, end_ts=86400,
        slot_seconds=_SLOT, car_speed_kmh=_SPEED,
        n_agents=1, random_state=42,
        act_dur_mu=act_dur_mu,
        act_dur_sigma=act_dur_sigma,
        purpose_act_starts=purpose_act_starts,
        purpose_acts=purpose_acts,
    )
    assert "activity" not in df.columns
    assert len(activities) > 0
    assert (activities["activity"] >= 0).all()
    assert (activities["activity"] < N_ACTIVITIES).all()
    assert set(activities["stop_id"]).issubset(set(range(len(df))))


def test_activity_column_absent_when_disabled():
    """Without activity params, no activities are sampled -> empty table."""
    tess = pd.DataFrame({
        "tile_id": [0, 1],
        "lat": [48.8566, 48.95],
        "lng": [2.3522, 2.55],
        "relevance": [1.0, 1.0],
    })
    diary_arrays = _diary_arrays_single([0, 1, 0], [0, 8 * 3600, 18 * 3600])
    df, _, _, activities = simulate_agents(
        tess, "relevance", diary_arrays,
        start_ts=0, end_ts=86400,
        slot_seconds=_SLOT, car_speed_kmh=_SPEED,
        n_agents=1, random_state=42,
    )
    assert "activity" not in df.columns
    assert len(activities) == 0


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
    df, _, _, _ = simulate_agents(
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
    """Every purpose code 0-2 has at least one eligible activity."""
    purpose_act_starts, purpose_acts = build_eligibility_csr()
    assert len(purpose_act_starts) == 4  # 3 purposes + sentinel
    for p in range(3):
        n_eligible = int(purpose_act_starts[p + 1] - purpose_act_starts[p])
        assert n_eligible > 0, f"Purpose {p} has no eligible activities"
