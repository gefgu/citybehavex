"""Project-owned simulation core driver for citybehavex."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

import citybehavex._core as _cbx_core

from citybehavex.math import sample_weighted_indices
from citybehavex.schedules import DiaryArrays
from citybehavex.simulation.social_graph import (
    build_colocation_social_graph,
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
    edge_sources: np.ndarray | None = None,
    edge_targets: np.ndarray | None = None,
    edge_kinds: np.ndarray | None = None,
) -> SocialGraphArtifact:
    """Pack the simulation social graph for WebGL rendering."""
    starts = np.asarray(neighbor_starts, dtype=np.int64)
    neigh = np.asarray(neighbors, dtype=np.int64)
    weights = np.asarray(edge_weights, dtype=np.float64)
    coords = _social_layout(n_agents, random_state, profile_embeddings)
    if edge_sources is not None and edge_targets is not None:
        sources = np.asarray(edge_sources, dtype=np.int64)
        targets = np.asarray(edge_targets, dtype=np.int64)
        final_weights = np.asarray(edge_weights, dtype=np.float64)
        degrees = np.bincount(sources[(0 <= sources) & (sources < n_agents)], minlength=n_agents)
    else:
        sources = None
        targets = None
        final_weights = weights
        degrees = np.diff(starts).astype(np.int64)

    weighted = np.zeros(n_agents, dtype=np.float64)
    if sources is not None and targets is not None and len(final_weights) == len(sources):
        for source, weight in zip(sources, final_weights):
            if 0 <= source < n_agents:
                weighted[int(source)] += abs(float(weight))
    else:
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
    if sources is not None and targets is not None:
        kinds = np.asarray(edge_kinds, dtype=np.uint8) if edge_kinds is not None else np.zeros(len(sources), dtype=np.uint8)
        for edge_idx, (source, target) in enumerate(zip(sources, targets)):
            weight = float(final_weights[edge_idx]) if edge_idx < len(final_weights) else 1.0
            kind = int(kinds[edge_idx]) if edge_idx < len(kinds) else 0
            edges.append([int(source), int(target), round(weight, 4), kind])
    else:
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
            "kind": "final_dynamic_social" if sources is not None else "initial_profile_similarity",
            "node_count": int(n_agents),
            "edge_count": int(len(edges)),
            "layout": "profile_svd" if profile_embeddings is not None else "fallback_seeded_random",
            "directed": True,
            "social_graph_k": int(social_graph_k),
            "edge_kind_labels": {"0": "initial", "1": "dynamic"} if sources is not None else None,
        },
    )


def _build_moving_frame(
    path_agent: np.ndarray,
    path_stop_id: np.ndarray,
    path_seq: np.ndarray,
    path_lat: np.ndarray,
    path_lng: np.ndarray,
    path_t: np.ndarray,
    path_mode: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build the one-row-per-waypoint `moving` frame from the raw arrays the
    Rust core returns -- shared by the final one-shot build and the per-day
    streaming callback so both produce identically-shaped chunks."""
    path_t_arr = np.asarray(path_t, dtype=np.int64)
    mode_arr = (
        np.asarray(path_mode, dtype=np.uint8)
        if path_mode is not None
        else np.ones(len(path_agent), dtype=np.uint8)
    )
    mode = np.select(
        [mode_arr == 2, mode_arr == 3, mode_arr == 4],
        ["walk", "bike", "rail"],
        default="car",
    )
    return pd.DataFrame(
        {
            "uid": np.asarray(path_agent, dtype=np.int64),
            "stop_id": np.asarray(path_stop_id, dtype=np.int64),
            "seq": np.asarray(path_seq, dtype=np.int32),
            "lat": np.asarray(path_lat, dtype=float),
            "lng": np.asarray(path_lng, dtype=float),
            "t": path_t_arr.astype("datetime64[s]"),
            "mode": mode,
        }
    )


def _build_encounters_frame(
    agent: np.ndarray,
    contact: np.ndarray,
    tile: np.ndarray,
    ts: np.ndarray,
) -> pd.DataFrame:
    """Build the one-row-per-encounter `encounters` frame from the raw arrays
    the Rust core returns -- shared by the final one-shot build and the
    per-day streaming callback so both produce identically-shaped chunks."""
    return pd.DataFrame(
        {
            "agent": np.asarray(agent, dtype=np.int64),
            "contact": np.asarray(contact, dtype=np.int64),
            "tile": np.asarray(tile, dtype=np.int64),
            "ts": np.asarray(ts, dtype=np.int64),
        }
    )


def _build_activity_frame(
    agent: np.ndarray,
    stop_id: np.ndarray,
    seq: np.ndarray,
    activity: np.ndarray,
    arrival: np.ndarray,
    departure: np.ndarray,
    block_id: np.ndarray,
) -> pd.DataFrame:
    """Build the one-row-per-micro-activity `activities` frame from the raw
    arrays the Rust core returns -- shared by the final one-shot build and
    the per-day streaming callback so both produce identically-shaped
    chunks.

    ``block_id`` is the diary block that drove this activity's contextual
    alignment lookup (see ``citybehavex.activities.alignment``); it's exposed
    here purely so Python-side reachability analysis can tell which
    (cluster, block) pairs a run actually visited.
    """
    arrival_arr = np.asarray(arrival, dtype=np.int64)
    departure_arr = np.asarray(departure, dtype=np.int64)
    return pd.DataFrame(
        {
            "uid": np.asarray(agent, dtype=np.int64),
            "stop_id": np.asarray(stop_id, dtype=np.int64),
            "seq": np.asarray(seq, dtype=np.int32),
            "activity": np.asarray(activity, dtype=np.int64),
            "arrival": arrival_arr.astype("datetime64[s]"),
            "departure": departure_arr.astype("datetime64[s]"),
            "block_id": np.asarray(block_id, dtype=np.int64),
        }
    )


def _build_trip_frame(
    agent: np.ndarray,
    loc_id: np.ndarray,
    arrival: np.ndarray,
    departure: np.ndarray,
    duration: np.ndarray,
    stop_id: np.ndarray,
    abstract_loc: np.ndarray,
    lats: np.ndarray,
    lngs: np.ndarray,
) -> pd.DataFrame:
    """Build the one-row-per-stop `trajectories` frame from the raw arrays the
    Rust core returns. The Rust side stores a `loc_id` (tessellation row
    index) instead of a lat/lng copy per stop to keep the per-row footprint
    small; the join back to actual coordinates happens here, against the
    small O(n_locations) `lats`/`lngs` tables already built from
    `tessellation_df`."""
    loc_idx = np.asarray(loc_id, dtype=np.int64)
    arrival_arr = np.asarray(arrival, dtype=np.int64)
    departure_arr = np.asarray(departure, dtype=np.int64)
    # Purpose comes straight from the abstract-location code that drove each
    # stop in the Rust engine (0=HOME, 1=WORK, everything else=OTHER) --
    # matches citybehavex.schedules.ddcrp._PURPOSE_CODE exactly, so it always
    # reflects the actual routing decision instead of being re-derived from
    # a stop's (possibly slot-shifted) arrival timestamp.
    abstract_loc_arr = np.asarray(abstract_loc, dtype=np.int32)
    purpose = np.where(
        abstract_loc_arr == 0, "HOME", np.where(abstract_loc_arr == 1, "WORK", "OTHER")
    )
    return pd.DataFrame(
        {
            "uid": np.asarray(agent, dtype=np.int64),
            "stop_id": np.asarray(stop_id, dtype=np.int64),
            "datetime": arrival_arr.astype("datetime64[s]"),
            "lat": lats[loc_idx],
            "lng": lngs[loc_idx],
            "arrival": arrival_arr.astype("datetime64[s]"),
            "departure": departure_arr.astype("datetime64[s]"),
            "trip_duration_minutes": np.asarray(duration, dtype=np.float64) / 60.0,
            "dwell_minutes": (departure_arr - arrival_arr) / 60.0,
            "purpose": purpose,
        }
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
    walking_speed_kmh: float = 4.8,
    bike_speed_kmh: float = 15.0,
    walking_threshold_mu_ln_km: float = -0.35,
    walking_threshold_sigma_ln: float = 0.45,
    bike_threshold_mu_ln_km: float = 1.4,
    bike_threshold_sigma_ln: float = 0.55,
    rho: float = 0.6,
    gamma: float = 0.21,
    alpha: float = 0.2,
    social_graph_k: int = 20,
    profile_graph_exact_threshold: int = 10_000,
    home_h3_resolution: int = 7,
    work_h3_resolution: int = 7,
    degree_mu_ln: float = 2.1776,
    degree_sigma_ln: float = 0.5,
    max_degree: int = 200,
    similarity_temperature: float = 0.3,
    max_candidate_pool: int = 2000,
    max_ring_expansion: int = 2,
    dynamic_friendships_enabled: bool = True,
    friendship_update_interval_hours: float = 24.0,
    encounter_window_hours: float = 24.0 * 7,
    regularity_threshold: float = 0.3,
    topological_overlap_threshold: float = 0.05,
    recast_random_baseline_samples: int = 256,
    recast_random_chance_probability: float = 1.0e-3,
    strength_initial: float = 0.1,
    strength_growth_mu_ln: float = -2.3,
    strength_growth_sigma_ln: float = 0.5,
    strength_decay_rate: float = 0.05,
    max_dynamic_degree: int = 200,
    max_colocation_group_size: int = 50,
    dt_update_mob_sim_hours: float = 24 * 7,
    indipendency_window_hours: float = 0.5,
    gravity_deterrence_exponent: float = -2.0,
    gravity_origin_exponent: float = 1.0,
    gravity_destination_exponent: float = 1.0,
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
    activity_alignment_scores: np.ndarray | None = None,
    activity_cluster_labels: np.ndarray | None = None,
    poi_semantic_scores: np.ndarray | None = None,
    location_semantic_cluster_ids: np.ndarray | None = None,
    poi_mask_starts: np.ndarray | None = None,
    poi_mask_activities: np.ndarray | None = None,
    poi_type_choice_enabled: bool = False,
    poi_type_alignment_scores: np.ndarray | None = None,
    poi_type_choice_temperature: float = 0.5,
    poi_type_choice_alpha: float = 1.0,
    activity_history_weight: float = 1.0,
    materialize_travel: bool = True,
    road_edge_from: np.ndarray | None = None,
    road_edge_to: np.ndarray | None = None,
    road_edge_weight_ds: np.ndarray | None = None,
    road_node_lats: np.ndarray | None = None,
    road_node_lngs: np.ndarray | None = None,
    location_road_node: np.ndarray | None = None,
    max_leg_waypoints: int = 16,
    rail_edge_from: np.ndarray | None = None,
    rail_edge_to: np.ndarray | None = None,
    rail_edge_weight_ds: np.ndarray | None = None,
    rail_node_lats: np.ndarray | None = None,
    rail_node_lngs: np.ndarray | None = None,
    location_rail_node: np.ndarray | None = None,
    max_rail_leg_waypoints: int = 16,
    has_car: np.ndarray | None = None,
    has_bike: np.ndarray | None = None,
    return_social_graph: bool = False,
    social_node_profiles: list[str] | None = None,
    on_day_flush: Callable[[pd.DataFrame], None] | None = None,
    on_encounter_day_flush: Callable[[pd.DataFrame], None] | None = None,
    on_trip_day_flush: Callable[[pd.DataFrame], None] | None = None,
    on_activity_day_flush: Callable[[pd.DataFrame], None] | None = None,
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

    if len(diary_arrays) == 5:
        diary_timestamps, diary_abs_locs, diary_starts, diary_ends, diary_block_ids = diary_arrays
    else:
        diary_timestamps, diary_abs_locs, diary_starts, diary_ends = diary_arrays
        diary_block_ids = np.zeros_like(diary_abs_locs, dtype=np.int32)

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

    if profile_embeddings is not None and sl is not None:
        home_cells = _cbx_core.batch_latlng_to_cells(lats[sl], lngs[sl], home_h3_resolution)
        work_cells = _cbx_core.batch_latlng_to_cells(lats[wt], lngs[wt], work_h3_resolution)
        neighbor_starts, neighbors, edge_weights = build_colocation_social_graph(
            profile_embeddings,
            home_cells,
            work_cells,
            degree_mu_ln=degree_mu_ln,
            degree_sigma_ln=degree_sigma_ln,
            max_degree=max_degree,
            temperature=similarity_temperature,
            max_candidate_pool=max_candidate_pool,
            max_ring_expansion=max_ring_expansion,
            random_state=random_state,
        )
    elif profile_embeddings is not None:
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

    rail_enabled = (
        rail_edge_from is not None
        and rail_edge_to is not None
        and rail_edge_weight_ds is not None
        and len(rail_edge_from) > 0
    )
    if rail_enabled:
        rail_from = np.ascontiguousarray(rail_edge_from, dtype=np.int64)
        rail_to = np.ascontiguousarray(rail_edge_to, dtype=np.int64)
        rail_weight = np.ascontiguousarray(rail_edge_weight_ds, dtype=np.int64)
        rail_lats = np.ascontiguousarray(rail_node_lats, dtype=np.float64)
        rail_lngs = np.ascontiguousarray(rail_node_lngs, dtype=np.float64)
        rail_location_node = (
            np.ascontiguousarray(location_rail_node, dtype=np.int64)
            if location_rail_node is not None
            else np.full(len(tessellation_df), -1, dtype=np.int64)
        )
    else:
        rail_from = np.empty(0, dtype=np.int64)
        rail_to = np.empty(0, dtype=np.int64)
        rail_weight = np.empty(0, dtype=np.int64)
        rail_lats = np.empty(0, dtype=np.float64)
        rail_lngs = np.empty(0, dtype=np.float64)
        rail_location_node = np.empty(0, dtype=np.int64)

    rng = np.random.default_rng(random_state)
    has_car_arr = (
        np.ascontiguousarray(has_car, dtype=np.bool_)
        if has_car is not None
        else np.ones(n_agents, dtype=np.bool_)
    )
    has_bike_arr = (
        np.ascontiguousarray(has_bike, dtype=np.bool_)
        if has_bike is not None
        else np.zeros(n_agents, dtype=np.bool_)
    )
    walking_threshold = rng.lognormal(
        mean=float(walking_threshold_mu_ln_km),
        sigma=float(walking_threshold_sigma_ln),
        size=n_agents,
    )
    bike_threshold = rng.lognormal(
        mean=float(bike_threshold_mu_ln_km),
        sigma=float(bike_threshold_sigma_ln),
        size=n_agents,
    )
    bike_threshold = np.maximum(bike_threshold, walking_threshold)
    walking_threshold = np.ascontiguousarray(walking_threshold, dtype=np.float64)
    bike_threshold = np.ascontiguousarray(bike_threshold, dtype=np.float64)

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
    activity_alignment_flat = None
    poi_semantic_scores_flat = None
    poi_type_alignment_flat = None
    n_activity_clusters = 0
    n_activity_blocks = 0
    n_activity_prev = 0
    n_poi_semantic_clusters = 0
    n_poi_type_blocks = 0
    n_poi_type_clusters = 0
    activity_cluster_labels_arr = None
    if activity_alignment_scores is not None:
        activity_alignment_scores = np.asarray(activity_alignment_scores, dtype=np.float64)
        if activity_alignment_scores.ndim != 4:
            raise ValueError("activity_alignment_scores must have shape [clusters, blocks, previous, activities]")
        n_activity_clusters = int(activity_alignment_scores.shape[0])
        n_activity_blocks = int(activity_alignment_scores.shape[1])
        n_activity_prev = int(activity_alignment_scores.shape[2])
        activity_alignment_flat = np.ascontiguousarray(activity_alignment_scores.flatten(), dtype=np.float64)
        if activity_cluster_labels is None:
            raise ValueError("activity_cluster_labels is required when activity_alignment_scores is provided")
        activity_cluster_labels_arr = np.ascontiguousarray(activity_cluster_labels, dtype=np.int64)
        if len(activity_cluster_labels_arr) != n_agents:
            raise ValueError("activity_cluster_labels must have one label per agent")
    if poi_semantic_scores is not None:
        poi_semantic_scores = np.asarray(poi_semantic_scores, dtype=np.float64)
        if poi_semantic_scores.ndim != 3:
            raise ValueError("poi_semantic_scores must have shape [clusters, semantic_clusters, activities]")
        if activity_cluster_labels is None:
            raise ValueError("activity_cluster_labels is required when poi_semantic_scores is provided")
        if activity_cluster_labels_arr is None:
            activity_cluster_labels_arr = np.ascontiguousarray(activity_cluster_labels, dtype=np.int64)
            if len(activity_cluster_labels_arr) != n_agents:
                raise ValueError("activity_cluster_labels must have one label per agent")
        if int(poi_semantic_scores.shape[0]) != n_activity_clusters and n_activity_clusters != 0:
            raise ValueError("poi_semantic_scores cluster dimension must match activity_alignment_scores")
        n_activity_clusters = int(poi_semantic_scores.shape[0])
        n_poi_semantic_clusters = int(poi_semantic_scores.shape[1])
        poi_semantic_scores_flat = np.ascontiguousarray(poi_semantic_scores.flatten(), dtype=np.float64)
    if poi_type_alignment_scores is not None:
        poi_type_alignment_scores = np.asarray(poi_type_alignment_scores, dtype=np.float64)
        if poi_type_alignment_scores.ndim != 3:
            raise ValueError("poi_type_alignment_scores must have shape [clusters, blocks, semantic_clusters]")
        if activity_cluster_labels is None:
            raise ValueError("activity_cluster_labels is required when poi_type_alignment_scores is provided")
        if activity_cluster_labels_arr is None:
            activity_cluster_labels_arr = np.ascontiguousarray(activity_cluster_labels, dtype=np.int64)
            if len(activity_cluster_labels_arr) != n_agents:
                raise ValueError("activity_cluster_labels must have one label per agent")
        if int(poi_type_alignment_scores.shape[0]) != n_activity_clusters and n_activity_clusters != 0:
            raise ValueError("poi_type_alignment_scores cluster dimension must match other alignment tensors")
        n_activity_clusters = int(poi_type_alignment_scores.shape[0])
        n_poi_type_blocks = int(poi_type_alignment_scores.shape[1])
        n_poi_type_clusters = int(poi_type_alignment_scores.shape[2])
        poi_type_alignment_flat = np.ascontiguousarray(poi_type_alignment_scores.flatten(), dtype=np.float64)
    location_semantic_cluster_ids_arr = (
        np.ascontiguousarray(location_semantic_cluster_ids, dtype=np.int64)
        if location_semantic_cluster_ids is not None
        else None
    )
    poi_mask_starts_arr = (
        np.ascontiguousarray(poi_mask_starts, dtype=np.int64)
        if poi_mask_starts is not None
        else None
    )
    poi_mask_activities_arr = (
        np.ascontiguousarray(poi_mask_activities, dtype=np.int64)
        if poi_mask_activities is not None
        else None
    )
    if poi_type_choice_enabled:
        if poi_type_alignment_flat is None:
            raise ValueError("poi_type_alignment_scores is required when poi_type_choice_enabled=True")
        if location_semantic_cluster_ids_arr is None:
            raise ValueError("location_semantic_cluster_ids is required when poi_type_choice_enabled=True")
        if len(location_semantic_cluster_ids_arr) != len(tessellation_df):
            raise ValueError("location_semantic_cluster_ids must have one value per tessellation row")
        if n_poi_type_clusters <= 0:
            raise ValueError("poi_type_alignment_scores must include at least one semantic cluster")

    rust_on_day_flush = None
    if on_day_flush is not None:

        def rust_on_day_flush(agent, dest_stop_id, seq, lat, lng, t, mode):
            on_day_flush(_build_moving_frame(agent, dest_stop_id, seq, lat, lng, t, mode))

    rust_on_encounter_day_flush = None
    if on_encounter_day_flush is not None:

        def rust_on_encounter_day_flush(agent, contact, tile, ts):
            on_encounter_day_flush(_build_encounters_frame(agent, contact, tile, ts))

    rust_on_trip_day_flush = None
    if on_trip_day_flush is not None:

        def rust_on_trip_day_flush(agent, loc_id, arrival, departure, duration, stop_id, abstract_loc):
            on_trip_day_flush(
                _build_trip_frame(
                    agent, loc_id, arrival, departure, duration, stop_id, abstract_loc, lats, lngs
                )
            )

    rust_on_activity_day_flush = None
    if on_activity_day_flush is not None:

        def rust_on_activity_day_flush(agent, stop_id, seq, activity, arrival, departure, block_id):
            on_activity_day_flush(
                _build_activity_frame(agent, stop_id, seq, activity, arrival, departure, block_id)
            )

    start = time.perf_counter()
    rust_result = _cbx_core.simulation_core_simulate_agents(
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
        np.ascontiguousarray(diary_block_ids, dtype=np.int32),
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
        activity_alignment_flat,
        activity_cluster_labels_arr,
        int(n_activity_clusters),
        int(n_activity_blocks),
        int(n_activity_prev),
        poi_semantic_scores_flat,
        location_semantic_cluster_ids_arr,
        poi_mask_starts_arr,
        poi_mask_activities_arr,
        int(n_poi_semantic_clusters),
        bool(poi_type_choice_enabled),
        poi_type_alignment_flat,
        int(n_poi_type_blocks),
        int(n_poi_type_clusters),
        float(poi_type_choice_temperature),
        float(poi_type_choice_alpha),
        float(activity_history_weight),
        bool(materialize_travel),
        r_edge_from,
        r_edge_to,
        r_edge_weight,
        r_node_lats,
        r_node_lngs,
        r_location_node,
        int(max_leg_waypoints),
        float(gravity_deterrence_exponent),
        float(gravity_origin_exponent),
        float(gravity_destination_exponent),
        float(walking_speed_kmh),
        float(bike_speed_kmh),
        has_car_arr,
        has_bike_arr,
        walking_threshold,
        bike_threshold,
        rail_from,
        rail_to,
        rail_weight,
        rail_lats,
        rail_lngs,
        rail_location_node,
        int(max_rail_leg_waypoints),
        rust_on_day_flush,
        rust_on_encounter_day_flush,
        rust_on_trip_day_flush,
        rust_on_activity_day_flush,
        bool(dynamic_friendships_enabled),
        int(friendship_update_interval_hours * 3600),
        int(encounter_window_hours * 3600),
        float(regularity_threshold),
        float(topological_overlap_threshold),
        int(recast_random_baseline_samples),
        float(recast_random_chance_probability),
        float(strength_initial),
        float(strength_growth_mu_ln),
        float(strength_growth_sigma_ln),
        float(strength_decay_rate),
        int(max_dynamic_degree),
        int(max_colocation_group_size),
        bool(return_social_graph),
    )
    if return_social_graph:
        (
            (
                agent_ids, loc_id, arrival, departure, trip_dur,
                enc_agent, enc_contact, enc_tile, enc_ts, stop_abstract_loc,
            ),
            (
                stop_id, path_agent, path_stop_id, path_seq, path_lat, path_lng, path_t, path_mode,
            ),
            (
                act_agent, act_stop_id, act_seq, act_activity, act_arrival, act_departure, act_block_id,
            ),
            (social_source, social_target, social_weight, social_kind),
        ) = rust_result
    else:
        social_source = social_target = social_weight = social_kind = None
        (
        (
            agent_ids, loc_id, arrival, departure, trip_dur,
            enc_agent, enc_contact, enc_tile, enc_ts, stop_abstract_loc,
        ),
        (
            stop_id, path_agent, path_stop_id, path_seq, path_lat, path_lng, path_t, path_mode,
        ),
        (
            act_agent, act_stop_id, act_seq, act_activity, act_arrival, act_departure, act_block_id,
        ),
        ) = rust_result
    elapsed = time.perf_counter() - start
    if timing is not None:
        timing.seconds += elapsed

    trajectories = _build_trip_frame(
        agent_ids, loc_id, arrival, departure, trip_dur, stop_id, stop_abstract_loc, lats, lngs
    )

    encounters = _build_encounters_frame(enc_agent, enc_contact, enc_tile, enc_ts)

    moving = _build_moving_frame(path_agent, path_stop_id, path_seq, path_lat, path_lng, path_t, path_mode)

    activities = _build_activity_frame(
        act_agent, act_stop_id, act_seq, act_activity, act_arrival, act_departure, act_block_id
    )

    if social_graph_artifact is not None:
        if social_source is not None and social_target is not None and social_weight is not None:
            social_graph_artifact = build_social_graph_artifact(
                neighbor_starts,
                neighbors,
                np.asarray(social_weight, dtype=np.float64),
                n_agents=n_agents,
                random_state=random_state,
                social_graph_k=social_graph_k,
                profile_embeddings=profile_embeddings,
                profile_types=social_node_profiles,
                edge_sources=np.asarray(social_source, dtype=np.int64),
                edge_targets=np.asarray(social_target, dtype=np.int64),
                edge_kinds=np.asarray(social_kind, dtype=np.uint8) if social_kind is not None else None,
            )
        return trajectories, encounters, moving, activities, social_graph_artifact
    return trajectories, encounters, moving, activities
