from __future__ import annotations

import numpy as np
import pandas as pd

import citybehavex._core as core
from citybehavex.trip_ditras import simulate_trip_ditras

# Power-law gravity matching skmob2's singly-constrained Gravity default.
_DETERRENCE = ("power_law", -2.0, 1.0, 1.0)
_SLOT = 900  # 15 minutes
_SPEED = 50.0  # km/h


def _run(lats, lngs, abs_locs, slot_times, *, end_ts):
    diary_ts = np.asarray(slot_times, dtype=np.int64)
    diary_loc = np.asarray(abs_locs, dtype=np.int32)
    starts = np.array([0], dtype=np.int64)
    ends = np.array([len(diary_ts)], dtype=np.int64)
    return core.trip_ditras_simulate_agents(
        np.asarray(lats, dtype=float),
        np.asarray(lngs, dtype=float),
        np.ones(len(lats), dtype=float),
        diary_ts,
        diary_loc,
        starts,
        ends,
        _DETERRENCE[0],
        _DETERRENCE[1],
        _DETERRENCE[2],
        _DETERRENCE[3],
        0.3,
        0.21,
        0,
        end_ts,
        _SLOT,
        _SPEED,
        1,
        42,
        np.array([0], dtype=np.int64),  # force home = index 0
    )


def test_long_trip_is_centered_on_slot_boundary():
    # Two locations only, so the single away choice is forced to the far one
    # (~18 km -> ~21 min car trip, longer than a 15-min slot).
    lats = [48.8566, 48.95]
    lngs = [2.3522, 2.55]
    slot_times = [0, 8 * 3600, 18 * 3600]
    abs_locs = [0, 1, 0]
    ag, la, lo, arr, dep, dur = _run(lats, lngs, abs_locs, slot_times, end_ts=86400)

    assert len(ag) == 3
    arr, dep, dur = np.asarray(arr), np.asarray(dep), np.asarray(dur)
    assert np.all(np.diff(arr) >= 0)
    assert np.all((dep - arr) >= 0)
    assert dur[0] == 0.0 and arr[0] == 0

    # Inbound long trip is centered on T=08:00: depart before, arrive after.
    assert dur[1] > _SLOT
    assert dep[0] < 8 * 3600
    assert arr[1] > 8 * 3600
    # Symmetric placement around the boundary.
    assert abs((8 * 3600 - dep[0]) - (arr[1] - 8 * 3600)) <= 1


def test_short_trip_arrives_within_the_slot():
    # Two locations a short hop apart (~0.2 km -> well under a 15-min slot).
    lats = [48.8566, 48.8580]
    lngs = [2.3522, 2.3540]
    slot_times = [0, 8 * 3600, 18 * 3600]
    abs_locs = [0, 1, 0]
    _, _, _, arr, dep, dur = _run(lats, lngs, abs_locs, slot_times, end_ts=86400)
    arr, dep, dur = np.asarray(arr), np.asarray(dep), np.asarray(dur)

    # Short inbound trip: depart at the scheduled boundary, arrive within the slot.
    assert dur[1] < _SLOT
    assert dep[0] == 8 * 3600
    assert 8 * 3600 <= arr[1] < 8 * 3600 + _SLOT


def test_trip_durations_are_off_the_hourly_grid():
    lats = [48.8566, 48.95]
    lngs = [2.3522, 2.55]
    slot_times = [0, 8 * 3600, 18 * 3600]
    abs_locs = [0, 1, 0]
    _, _, _, arr, dep, dur = _run(lats, lngs, abs_locs, slot_times, end_ts=86400)
    # The centered long trip shifts arrival off the 15-min grid.
    assert any(int(a) % _SLOT != 0 for a in np.asarray(arr))


def test_simulate_trip_ditras_returns_trip_columns():
    tess = pd.DataFrame(
        {
            "tile_id": [0, 1, 2],
            "lat": [48.8566, 48.95, 48.8580],
            "lng": [2.3522, 2.55, 2.3540],
            "relevance": [1.0, 1.0, 1.0],
        }
    )
    diary_arrays = (
        np.array([0, 8 * 3600, 18 * 3600], dtype=np.int64),
        np.array([0, 1, 0], dtype=np.int32),
        np.array([0], dtype=np.int64),
        np.array([3], dtype=np.int64),
    )
    df = simulate_trip_ditras(
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
    for column in ("uid", "datetime", "lat", "lng", "arrival", "departure",
                   "trip_duration_minutes", "dwell_minutes"):
        assert column in df.columns
    assert (df["dwell_minutes"] >= 0).all()
    assert (df["trip_duration_minutes"] >= 0).all()
    assert pd.api.types.is_datetime64_any_dtype(df["arrival"])
