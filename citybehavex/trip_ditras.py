"""Trip-duration-aware DITRAS driver for citybehavex.

Builds a multi-day mobility diary by stitching per-day Markov diaries (selecting a
weekday or weekend chain per calendar day) and feeds it to the citybehavex Rust
extension (`citybehavex._core.trip_ditras_simulate_agents`), which assigns physical
locations via the same gravity/EPR mechanism as skmob2's DITRAS but additionally
derives a car trip duration per leg and shifts arrival/departure off the slot grid.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

import citybehavex._core as _cbx_core
from skmob2 import _core as _skmob_core
from skmob2.models.gravity import Gravity
from skmob2.models.markov_diary_generator import MarkovDiaryGenerator

DiaryArrays = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _day_seed(random_state: int, day_index: int) -> int:
    return (int(random_state) * 1_000_003 + day_index) & 0x7FFFFFFF


def build_daily_diary(
    generators: Mapping[str, MarkovDiaryGenerator],
    start_date: pd.Timestamp,
    days: int,
    n_agents: int,
    random_state: int,
) -> DiaryArrays:
    """Generate and stitch per-day diaries into flat per-agent arrays.

    For each calendar day, the weekend chain is used on Sat/Sun and the weekday
    chain otherwise (falling back to the weekday chain if no weekend one exists).
    Abstract location ids only signal home (0) vs away (non-zero), so per-day
    restarts stitch cleanly: the agent returns home at each midnight.
    """
    weekday_gen = generators["weekday"]
    slots_per_day = weekday_gen._slots_per_day

    per_ts: list[list[np.ndarray]] = [[] for _ in range(n_agents)]
    per_loc: list[list[np.ndarray]] = [[] for _ in range(n_agents)]

    for day_index in range(days):
        day = pd.Timestamp(start_date) + pd.Timedelta(days=day_index)
        day_type = "weekend" if day.dayofweek >= 5 else "weekday"
        gen = generators.get(day_type, weekday_gen)
        day_ts = int(day.timestamp())
        seed = _day_seed(random_state, day_index)
        ts, locs, starts, ends = _skmob_core.markov_diary_batch_generate(
            gen._cdf_matrix_flat,
            slots_per_day,
            day_ts,
            n_agents,
            seed,
            slots_per_day,
        )
        ts = np.asarray(ts, dtype=np.int64)
        locs = np.asarray(locs, dtype=np.int32)
        for agent in range(n_agents):
            per_ts[agent].append(ts[starts[agent] : ends[agent]])
            per_loc[agent].append(locs[starts[agent] : ends[agent]])

    flat_ts: list[np.ndarray] = []
    flat_loc: list[np.ndarray] = []
    d_starts: list[int] = []
    d_ends: list[int] = []
    offset = 0
    for agent in range(n_agents):
        ts_a = (
            np.concatenate(per_ts[agent])
            if per_ts[agent]
            else np.empty(0, dtype=np.int64)
        )
        loc_a = (
            np.concatenate(per_loc[agent])
            if per_loc[agent]
            else np.empty(0, dtype=np.int32)
        )
        d_starts.append(offset)
        offset += len(ts_a)
        d_ends.append(offset)
        flat_ts.append(ts_a)
        flat_loc.append(loc_a)

    diary_timestamps = (
        np.concatenate(flat_ts) if flat_ts else np.empty(0, dtype=np.int64)
    ).astype(np.int64)
    diary_abs_locs = (
        np.concatenate(flat_loc) if flat_loc else np.empty(0, dtype=np.int32)
    ).astype(np.int32)
    return (
        diary_timestamps,
        diary_abs_locs,
        np.asarray(d_starts, dtype=np.int64),
        np.asarray(d_ends, dtype=np.int64),
    )


def simulate_trip_ditras(
    tessellation_df: pd.DataFrame,
    relevance_column: str | None,
    diary_arrays: DiaryArrays,
    *,
    start_ts: int,
    end_ts: int,
    slot_seconds: int,
    car_speed_kmh: float,
    n_agents: int,
    random_state: int,
    rho: float = 0.3,
    gamma: float = 0.21,
) -> pd.DataFrame:
    """Run the trip-duration-aware DITRAS and return a record-per-stay DataFrame.

    Columns: ``uid, datetime (=arrival), lat, lng, arrival, departure,
    trip_duration_minutes, dwell_minutes``.
    """
    lats = np.ascontiguousarray(tessellation_df["lat"].to_numpy(dtype=float))
    lng_col = "lng" if "lng" in tessellation_df.columns else "lon"
    lngs = np.ascontiguousarray(tessellation_df[lng_col].to_numpy(dtype=float))
    if relevance_column and relevance_column in tessellation_df.columns:
        relevances = np.ascontiguousarray(
            tessellation_df[relevance_column].fillna(0).to_numpy(dtype=float)
        )
    else:
        relevances = np.ones(len(tessellation_df), dtype=float)

    diary_timestamps, diary_abs_locs, diary_starts, diary_ends = diary_arrays
    gravity = Gravity(gravity_type="singly constrained")

    agent_ids, out_lats, out_lngs, arrival, departure, trip_dur = (
        _cbx_core.trip_ditras_simulate_agents(
            lats,
            lngs,
            relevances,
            diary_timestamps,
            diary_abs_locs,
            diary_starts,
            diary_ends,
            gravity.deterrence_func_type,
            float(gravity.deterrence_func_args[0]),
            float(gravity.origin_exp),
            float(gravity.destination_exp),
            float(rho),
            float(gamma),
            int(start_ts),
            int(end_ts),
            int(slot_seconds),
            float(car_speed_kmh),
            int(n_agents),
            int(random_state),
            None,
        )
    )

    arrival = np.asarray(arrival, dtype=np.int64)
    departure = np.asarray(departure, dtype=np.int64)
    trip_dur = np.asarray(trip_dur, dtype=float)
    return pd.DataFrame(
        {
            "uid": np.asarray(agent_ids, dtype=np.int64),
            "datetime": arrival.astype("datetime64[s]"),
            "lat": np.asarray(out_lats, dtype=float),
            "lng": np.asarray(out_lngs, dtype=float),
            "arrival": arrival.astype("datetime64[s]"),
            "departure": departure.astype("datetime64[s]"),
            "trip_duration_minutes": trip_dur / 60.0,
            "dwell_minutes": (departure - arrival) / 60.0,
        }
    )
