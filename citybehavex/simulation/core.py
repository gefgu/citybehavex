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
    road_edge_from: np.ndarray | None = None,
    road_edge_to: np.ndarray | None = None,
    road_edge_weight_ds: np.ndarray | None = None,
    road_node_lats: np.ndarray | None = None,
    road_node_lngs: np.ndarray | None = None,
    location_road_node: np.ndarray | None = None,
    max_leg_waypoints: int = 16,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the CityBehavEx simulation core.

    Returns:
        (trajectories_df, encounters_df, moving_df, activities_df).
        ``trajectories_df`` is a stop table: one row per real physical
        location visit (a new row only appears when the agent actually
        relocates). ``encounters_df`` records (agent, contact, tile, ts) for
        each social interaction. ``moving_df`` has one row per waypoint along
        each trip's road-following path (or a 2-point origin/destination pair
        when road routing is disabled/unsnapped). ``activities_df`` has one
        row per sampled micro-activity, keyed by ``stop_id`` — a single stop
        can span several micro-activities (e.g. sleep -> breakfast -> get
        ready, all at HOME), kept separate so the stop table itself stays a
        clean one-row-per-visit table.
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

    road_enabled = (
        road_edge_from is not None
        and road_edge_to is not None
        and road_edge_weight_ds is not None
        and len(road_edge_from) > 0
    )
    if road_enabled:
        r_edge_from = np.ascontiguousarray(road_edge_from, dtype=np.int64)
        r_edge_to = np.ascontiguousarray(road_edge_to, dtype=np.int64)
        r_edge_weight = np.ascontiguousarray(road_edge_weight_ds, dtype=np.int64)
        r_node_lats = np.ascontiguousarray(road_node_lats, dtype=np.float64)
        r_node_lngs = np.ascontiguousarray(road_node_lngs, dtype=np.float64)
        r_location_node = (
            np.ascontiguousarray(location_road_node, dtype=np.int64)
            if location_road_node is not None
            else np.full(len(tessellation_df), -1, dtype=np.int64)
        )
    else:
        r_edge_from = np.empty(0, dtype=np.int64)
        r_edge_to = np.empty(0, dtype=np.int64)
        r_edge_weight = np.empty(0, dtype=np.int64)
        r_node_lats = np.empty(0, dtype=np.float64)
        r_node_lngs = np.empty(0, dtype=np.float64)
        r_location_node = np.empty(0, dtype=np.int64)

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
        (
            agent_ids, out_lats, out_lngs, arrival, departure, trip_dur,
            enc_agent, enc_contact, enc_tile, enc_ts,
        ),
        (
            stop_id, path_agent, path_stop_id, path_seq, path_lat, path_lng, path_t,
        ),
        (
            act_agent, act_stop_id, act_seq, act_activity, act_arrival, act_departure,
        ),
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
        r_edge_from,
        r_edge_to,
        r_edge_weight,
        r_node_lats,
        r_node_lngs,
        r_location_node,
        int(max_leg_waypoints),
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
            "stop_id": np.asarray(stop_id, dtype=np.int64),
            "datetime": arrival.astype("datetime64[s]"),
            "lat": np.asarray(out_lats, dtype=float),
            "lng": np.asarray(out_lngs, dtype=float),
            "arrival": arrival.astype("datetime64[s]"),
            "departure": departure.astype("datetime64[s]"),
            "trip_duration_minutes": trip_dur / 60.0,
            "dwell_minutes": (departure - arrival) / 60.0,
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

    path_t_arr = np.asarray(path_t, dtype=np.int64)
    moving = pd.DataFrame(
        {
            "uid": np.asarray(path_agent, dtype=np.int64),
            "stop_id": np.asarray(path_stop_id, dtype=np.int64),
            "seq": np.asarray(path_seq, dtype=np.int32),
            "lat": np.asarray(path_lat, dtype=float),
            "lng": np.asarray(path_lng, dtype=float),
            "t": path_t_arr.astype("datetime64[s]"),
        }
    )

    act_arrival_arr = np.asarray(act_arrival, dtype=np.int64)
    act_departure_arr = np.asarray(act_departure, dtype=np.int64)
    activities = pd.DataFrame(
        {
            "uid": np.asarray(act_agent, dtype=np.int64),
            "stop_id": np.asarray(act_stop_id, dtype=np.int64),
            "seq": np.asarray(act_seq, dtype=np.int32),
            "activity": np.asarray(act_activity, dtype=np.int64),
            "arrival": act_arrival_arr.astype("datetime64[s]"),
            "departure": act_departure_arr.astype("datetime64[s]"),
        }
    )

    return trajectories, encounters, moving, activities
