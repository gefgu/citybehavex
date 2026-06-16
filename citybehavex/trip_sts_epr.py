"""Trip-duration-aware STS-EPR driver for citybehavex."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

import citybehavex._core as _cbx_core
from skmob2 import _core as _skmob_core

from .trip_ditras import DiaryArrays


@dataclass
class RustTiming:
    seconds: float = 0.0


def simulate_trip_sts_epr(
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
    rho: float = 0.6,
    gamma: float = 0.21,
    alpha: float = 0.2,
    social_graph_radius: float = 0.5,
    dt_update_mob_sim_hours: float = 24 * 7,
    indipendency_window_hours: float = 0.5,
    rsl: bool = False,
    timing: RustTiming | None = None,
) -> pd.DataFrame:
    """Run trip-duration-aware STS-EPR and return a record-per-stay DataFrame."""
    lats = np.ascontiguousarray(tessellation_df["lat"].to_numpy(dtype=float))
    lng_col = "lng" if "lng" in tessellation_df.columns else "lon"
    lngs = np.ascontiguousarray(tessellation_df[lng_col].to_numpy(dtype=float))
    if relevance_column and relevance_column in tessellation_df.columns:
        relevances = np.asarray(tessellation_df[relevance_column].fillna(0), dtype=float)
        relevances = np.where(relevances == 0, 0.1, relevances)
        relevances = np.ascontiguousarray(relevances)
    else:
        relevances = np.ones(len(tessellation_df), dtype=float)

    diary_timestamps, diary_abs_locs, diary_starts, diary_ends = diary_arrays
    neighbor_starts, neighbors = _skmob_core.model_social_graph_random_geometric(
        int(n_agents),
        float(social_graph_radius),
        int(random_state),
    )
    neighbor_starts = np.asarray(neighbor_starts, dtype=np.int64)
    neighbors = np.asarray(neighbors, dtype=np.int64)
    flat_distances = np.empty(0, dtype=np.float64)

    start = time.perf_counter()
    agent_ids, out_lats, out_lngs, arrival, departure, trip_dur = (
        _cbx_core.trip_sts_epr_simulate_agents(
            lats,
            lngs,
            relevances,
            flat_distances,
            neighbor_starts,
            neighbors,
            diary_timestamps,
            diary_abs_locs,
            diary_starts,
            diary_ends,
            float(rho),
            float(gamma),
            float(alpha),
            int(start_ts),
            int(end_ts),
            int(indipendency_window_hours * 3600),
            int(dt_update_mob_sim_hours * 3600),
            int(slot_seconds),
            float(car_speed_kmh),
            int(n_agents),
            int(random_state),
            None,
            bool(rsl),
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
