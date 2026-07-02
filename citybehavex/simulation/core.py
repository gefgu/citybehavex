"""Project-owned simulation core driver for citybehavex."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

import citybehavex._core as _cbx_core

from citybehavex.schedules import DiaryArrays
from citybehavex.simulation.social_graph import (
    build_knn_fallback_social_graph,
    build_profile_social_graph,
)


@dataclass
class CoreTiming:
    seconds: float = 0.0


def simulate_agents(
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
    social_graph_k: int = 20,
    profile_graph_exact_threshold: int = 10_000,
    dt_update_mob_sim_hours: float = 24 * 7,
    indipendency_window_hours: float = 0.5,
    rsl: bool = False,
    timing: CoreTiming | None = None,
    starting_locs: np.ndarray | None = None,
    work_tiles: np.ndarray | None = None,
    profile_embeddings: np.ndarray | None = None,
    act_embs: np.ndarray | None = None,
    act_dur_mu: np.ndarray | None = None,
    act_dur_sigma: np.ndarray | None = None,
    purpose_act_starts: np.ndarray | None = None,
    purpose_acts: np.ndarray | None = None,
    act_kappa: float = 1.0,
    act_temp: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the CityBehavEx simulation core.

    Returns:
        (trajectories_df, encounters_df) where encounters_df records
        (agent, contact, tile, ts) for each social interaction.
    """
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

    if profile_embeddings is not None:
        neighbor_starts, neighbors, edge_weights = build_profile_social_graph(
            profile_embeddings,
            k=social_graph_k,
            random_state=random_state,
            exact_threshold=profile_graph_exact_threshold,
        )
    else:
        neighbor_starts, neighbors, edge_weights = build_knn_fallback_social_graph(
            n_agents, social_graph_k, random_state
        )

    neighbor_starts = np.ascontiguousarray(neighbor_starts, dtype=np.int64)
    neighbors = np.ascontiguousarray(neighbors, dtype=np.int64)
    edge_weights = np.ascontiguousarray(edge_weights, dtype=np.float64)
    flat_distances = np.empty(0, dtype=np.float64)

    sl = (
        np.ascontiguousarray(starting_locs, dtype=np.int64)
        if starting_locs is not None
        else None
    )
    wt = (
        np.ascontiguousarray(work_tiles, dtype=np.int64)
        if work_tiles is not None
        else None
    )
    eps = edge_weights if len(edge_weights) == len(neighbors) else None

    # Flatten activity embedding matrix if provided.
    emb_dim = 0
    act_embs_flat: np.ndarray | None = None
    prof_embs_flat: np.ndarray | None = None
    profile_act_sims_flat: np.ndarray | None = None
    if act_dur_mu is not None and act_embs is not None and profile_embeddings is not None:
        emb_dim = act_embs.shape[1] if act_embs.ndim == 2 else 0
        if emb_dim > 0:
            act_embs_flat = np.ascontiguousarray(act_embs.flatten(), dtype=np.float64)
            prof_embs_flat = np.ascontiguousarray(profile_embeddings.flatten(), dtype=np.float64)
            # Precompute profile×activity cosine sims: embeddings are L2-normalized,
            # so dot product == cosine similarity. Shape: [n_agents * n_acts].
            profile_act_sims_flat = np.ascontiguousarray(
                (profile_embeddings.astype(np.float64) @ act_embs.astype(np.float64).T).flatten(),
                dtype=np.float64,
            )

    start = time.perf_counter()
    (
        agent_ids, out_lats, out_lngs, arrival, departure, trip_dur,
        enc_agent, enc_contact, enc_tile, enc_ts, out_activity,
    ) = _cbx_core.simulation_core_simulate_agents(
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
        sl,
        bool(rsl),
        wt,
        eps,
        act_embs_flat,
        act_dur_mu.astype(np.float64) if act_dur_mu is not None else None,
        act_dur_sigma.astype(np.float64) if act_dur_sigma is not None else None,
        purpose_act_starts.astype(np.int64) if purpose_act_starts is not None else None,
        purpose_acts.astype(np.int64) if purpose_acts is not None else None,
        prof_embs_flat,
        emb_dim,
        float(act_kappa),
        float(act_temp),
        profile_act_sims_flat,
    )
    elapsed = time.perf_counter() - start
    if timing is not None:
        timing.seconds += elapsed

    arrival = np.asarray(arrival, dtype=np.int64)
    departure = np.asarray(departure, dtype=np.int64)
    trip_dur = np.asarray(trip_dur, dtype=float)
    trajectories = pd.DataFrame(
        {
            "uid": np.asarray(agent_ids, dtype=np.int64),
            "datetime": arrival.astype("datetime64[s]"),
            "lat": np.asarray(out_lats, dtype=float),
            "lng": np.asarray(out_lngs, dtype=float),
            "arrival": arrival.astype("datetime64[s]"),
            "departure": departure.astype("datetime64[s]"),
            "trip_duration_minutes": trip_dur / 60.0,
            "dwell_minutes": (departure - arrival) / 60.0,
            "activity": np.asarray(out_activity, dtype=np.int64),
        }
    )

    encounters = pd.DataFrame(
        {
            "agent": np.asarray(enc_agent, dtype=np.int64),
            "contact": np.asarray(enc_contact, dtype=np.int64),
            "tile": np.asarray(enc_tile, dtype=np.int64),
            "ts": np.asarray(enc_ts, dtype=np.int64),
        }
    )

    return trajectories, encounters
