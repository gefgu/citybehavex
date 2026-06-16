from __future__ import annotations

import numpy as np
import pandas as pd

import citybehavex._core as core
from citybehavex.trip_sts_epr import simulate_trip_sts_epr

_SLOT = 900
_SPEED = 50.0


def _run(lats, lngs, abs_locs, slot_times, *, end_ts):
    diary_ts = np.asarray(slot_times, dtype=np.int64)
    diary_loc = np.asarray(abs_locs, dtype=np.int32)
    starts = np.array([0], dtype=np.int64)
    ends = np.array([len(diary_ts)], dtype=np.int64)
    return core.trip_sts_epr_simulate_agents(
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
        1.0,
        0.21,
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


def test_sts_epr_long_trip_is_centered_on_slot_boundary():
    lats = [48.8566, 48.95]
    lngs = [2.3522, 2.55]
    slot_times = [0, 8 * 3600, 18 * 3600]
    abs_locs = [0, 1, 0]
    ag, _, _, arr, dep, dur = _run(lats, lngs, abs_locs, slot_times, end_ts=86400)

    assert len(ag) == 3
    arr, dep, dur = np.asarray(arr), np.asarray(dep), np.asarray(dur)
    assert np.all(np.diff(arr) >= 0)
    assert np.all((dep - arr) >= 0)
    assert dur[1] > _SLOT
    assert dep[0] < 8 * 3600
    assert arr[1] > 8 * 3600
    assert abs((8 * 3600 - dep[0]) - (arr[1] - 8 * 3600)) <= 1


def test_sts_epr_short_trip_arrives_within_the_slot():
    lats = [48.8566, 48.8580]
    lngs = [2.3522, 2.3540]
    slot_times = [0, 8 * 3600, 18 * 3600]
    abs_locs = [0, 1, 0]
    _, _, _, arr, dep, dur = _run(lats, lngs, abs_locs, slot_times, end_ts=86400)
    arr, dep, dur = np.asarray(arr), np.asarray(dep), np.asarray(dur)

    assert dur[1] < _SLOT
    assert dep[0] == 8 * 3600
    assert 8 * 3600 <= arr[1] < 8 * 3600 + _SLOT


def test_sts_epr_trip_durations_are_off_the_hourly_grid():
    lats = [48.8566, 48.95]
    lngs = [2.3522, 2.55]
    slot_times = [0, 8 * 3600, 18 * 3600]
    abs_locs = [0, 1, 0]
    _, _, _, arr, _, _ = _run(lats, lngs, abs_locs, slot_times, end_ts=86400)
    assert any(int(a) % _SLOT != 0 for a in np.asarray(arr))


def test_simulate_trip_sts_epr_returns_trip_columns():
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
    df = simulate_trip_sts_epr(
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
