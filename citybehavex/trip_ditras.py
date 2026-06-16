"""Trip-duration-aware DITRAS driver for citybehavex.

Feeds a multi-day mobility diary (built by the ddCRP schedule selector in
`citybehavex.schedule_ddcrp`) to the citybehavex Rust extension
(`citybehavex._core.trip_ditras_simulate_agents`), which assigns physical
locations via the same gravity/EPR mechanism as skmob2's DITRAS but additionally
derives a car trip duration per leg and shifts arrival/departure off the slot grid.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

import citybehavex._core as _cbx_core
from skmob2.models.gravity import Gravity


DiaryArrays = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


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
    timing: Any | None = None,
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

    start = time.perf_counter()
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
    elapsed = time.perf_counter() - start
    if timing is not None:
        timing.seconds += elapsed

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
