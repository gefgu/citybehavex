"""Project-owned simulation core driver for citybehavex."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

import citybehavex._core as _cbx_core

from citybehavex.math import sample_weighted_indices
from citybehavex.schedules import DiaryArrays
from citybehavex.simulation.social_graph import (
    build_knn_fallback_social_graph,
    build_profile_social_graph,
)


@dataclass
class CoreTiming:
    seconds: float = 0.0


@dataclass(frozen=True)
class SocialGraphArtifact:
    nodes: list[list[Any]]
    edges: list[list[float]]
    degrees: list[int]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.metadata,
            "nodes": self.nodes,
            "edges": self.edges,
            "degrees": self.degrees,
        }

    def write_json(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), separators=(",", ":")), encoding="utf-8")


def social_network_sidecar_path(output_path: str | Path) -> Path:
    p = Path(output_path)
    return p.with_name(f"{p.stem}_social_network.json")


def _social_layout(
    n_agents: int,
    random_state: int,
    profile_embeddings: np.ndarray | None,
) -> np.ndarray:
    if profile_embeddings is not None and n_agents >= 2 and profile_embeddings.ndim == 2:
        embeddings = np.ascontiguousarray(profile_embeddings, dtype=np.float64)
        if embeddings.shape[1] >= 2:
            coords = TruncatedSVD(n_components=2, random_state=random_state).fit_transform(embeddings)
        elif embeddings.shape[1] == 1:
            coords = np.column_stack([embeddings[:, 0], np.zeros(n_agents, dtype=np.float64)])
        else:
            coords = np.empty((n_agents, 0), dtype=np.float64)
        if coords.shape == (n_agents, 2) and np.isfinite(coords).all():
            scale = np.std(coords, axis=0)
            scale = np.where(scale == 0, 1.0, scale)
            return np.round((coords / scale) * 100.0, 1)

    rng = np.random.default_rng(random_state)
    return np.round((rng.random((n_agents, 2), dtype=np.float64) - 0.5) * 1000.0, 1)


def build_social_graph_artifact(
    neighbor_starts: np.ndarray,
    neighbors: np.ndarray,
    edge_weights: np.ndarray,
    *,
    n_agents: int,
    random_state: int,
    social_graph_k: int,
    profile_embeddings: np.ndarray | None = None,
    profile_types: list[str] | None = None,
) -> SocialGraphArtifact:
    """Pack the initial simulation social graph for WebGL rendering."""
    starts = np.asarray(neighbor_starts, dtype=np.int64)
    neigh = np.asarray(neighbors, dtype=np.int64)
    weights = np.asarray(edge_weights, dtype=np.float64)
    coords = _social_layout(n_agents, random_state, profile_embeddings)
    degrees = np.diff(starts).astype(np.int64)

    weighted = np.zeros(n_agents, dtype=np.float64)
    for i in range(n_agents):
        start, end = int(starts[i]), int(starts[i + 1])
        if end > start and len(weights) == len(neigh):
            weighted[i] = float(np.abs(weights[start:end]).sum())
        else:
            weighted[i] = float(end - start)
    max_weighted = float(weighted.max()) if weighted.size else 0.0
    if max_weighted > 0:
        sizes = 3.0 + 13.0 * np.sqrt(weighted / max_weighted)
    else:
        sizes = np.full(n_agents, 3.0, dtype=np.float64)

    nodes: list[list[Any]] = []
    for i in range(n_agents):
        row: list[Any] = [
            float(coords[i, 0]),
            float(coords[i, 1]),
            round(float(np.clip(sizes[i], 3.0, 16.0)), 1),
            i + 1,
        ]
        if profile_types is not None and i < len(profile_types):
            row.append(str(profile_types[i]))
        nodes.append(row)

    edges: list[list[float]] = []
    for source in range(n_agents):
        for edge_idx in range(int(starts[source]), int(starts[source + 1])):
            target = int(neigh[edge_idx])
            weight = float(weights[edge_idx]) if len(weights) == len(neigh) else 1.0
            edges.append([source, target, round(weight, 4)])

    return SocialGraphArtifact(
        nodes=nodes,
        edges=edges,
        degrees=[int(d) for d in degrees],
        metadata={
            "kind": "initial_profile_similarity",
            "node_count": int(n_agents),
            "edge_count": int(len(edges)),
            "layout": "profile_svd" if profile_embeddings is not None else "fallback_seeded_random",
            "directed": True,
            "social_graph_k": int(social_graph_k),
        },
    )


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
    return_social_graph: bool = False,
    social_node_profiles: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame] | tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, SocialGraphArtifact
]:
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
    social_graph_artifact = None
    if return_social_graph:
        social_graph_artifact = build_social_graph_artifact(
            neighbor_starts,
            neighbors,
            edge_weights,
            n_agents=n_agents,
            random_state=random_state,
            social_graph_k=social_graph_k,
            profile_embeddings=profile_embeddings,
            profile_types=social_node_profiles,
        )

    sl = (
        np.ascontiguousarray(starting_locs, dtype=np.int64)
        if starting_locs is not None
        else None
    )
    # WORK is pinned to a single tile per agent for the whole simulation (the
    # Rust core requires it), so when the caller hasn't supplied one (e.g. no
    # agent profiles were generated), sample it here the same way
    # `citybehavex.profiles.agents` does: relevance-weighted (commercial bias).
    wt = (
        np.ascontiguousarray(work_tiles, dtype=np.int64)
        if work_tiles is not None
        else np.ascontiguousarray(
            sample_weighted_indices(relevances, n_agents, np.random.default_rng(random_state)),
            dtype=np.int64,
        )
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
            enc_agent, enc_contact, enc_tile, enc_ts, stop_abstract_loc,
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
    # Purpose comes straight from the abstract-location code that drove each
    # stop in the Rust engine (0=HOME, 1=WORK, everything else=OTHER) --
    # matches citybehavex.schedules.ddcrp._PURPOSE_CODE exactly, so it always
    # reflects the actual routing decision instead of being re-derived from
    # a stop's (possibly slot-shifted) arrival timestamp.
    abstract_loc_arr = np.asarray(stop_abstract_loc, dtype=np.int32)
    purpose = np.where(
        abstract_loc_arr == 0, "HOME", np.where(abstract_loc_arr == 1, "WORK", "OTHER")
    )
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
            "purpose": purpose,
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

    if social_graph_artifact is not None:
        return trajectories, encounters, moving, activities, social_graph_artifact
    return trajectories, encounters, moving, activities
