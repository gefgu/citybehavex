"""Prepared Python inputs for the Rust simulation core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import polars as pl
from fastmob._core import latlng_to_h3_numpy

from citybehavex.math import sample_weighted_indices
from citybehavex.schedules import DiaryArrays
from citybehavex.social.social_graph import (
    build_colocation_social_graph,
    build_knn_fallback_social_graph,
    build_profile_social_graph,
)

__all__ = [
    "ActivityInputs",
    "CoreTiming",
    "DiaryInputs",
    "InitialLocationInputs",
    "LocationInputs",
    "NetworkInputs",
    "OutputHooks",
    "SimulationRunParams",
    "SocialGraphInputs",
    "TransportInputs",
]


@dataclass
class CoreTiming:
    seconds: float = 0.0


def _empty(dtype: np.dtype | type) -> np.ndarray:
    return np.empty(0, dtype=dtype)


def _contig(value: np.ndarray | None, dtype: np.dtype | type) -> np.ndarray:
    if value is None:
        return _empty(dtype)
    return np.ascontiguousarray(value, dtype=dtype)


@dataclass(slots=True)
class SimulationRunParams:
    start_ts: int
    end_ts: int
    slot_seconds: int
    car_speed_kmh: float
    n_agents: int
    master_seed: int | None
    walking_speed_kmh: float = 4.8
    bike_speed_kmh: float = 15.0
    rho: float = 0.6
    gamma: float = 0.21
    alpha: float = 0.2
    dt_update_mob_sim_s: int = 24 * 7 * 3600
    indipendency_window_s: int = 1800
    gravity_deterrence_exponent: float = -2.0
    gravity_origin_exponent: float = 1.0
    gravity_destination_exponent: float = 1.0
    dynamic_friendships_enabled: bool = True
    friendship_update_interval_s: int = 86400
    encounter_window_s: int = 604800
    regularity_threshold: float = 0.3
    topological_overlap_threshold: float = 0.05
    recast_random_baseline_samples: int = 256
    recast_random_chance_probability: float = 1.0e-3
    strength_initial: float = 0.1
    strength_growth_mu_ln: float = -2.3
    strength_growth_sigma_ln: float = 0.5
    strength_decay_rate: float = 0.05
    max_dynamic_degree: int = 200
    max_colocation_group_size: int = 50

    @classmethod
    def from_hours(
        cls,
        *,
        start_ts: int,
        end_ts: int,
        slot_seconds: int,
        car_speed_kmh: float,
        n_agents: int,
        random_state: int,
        walking_speed_kmh: float = 4.8,
        bike_speed_kmh: float = 15.0,
        rho: float = 0.6,
        gamma: float = 0.21,
        alpha: float = 0.2,
        dt_update_mob_sim_hours: float = 24 * 7,
        indipendency_window_hours: float = 0.5,
        gravity_deterrence_exponent: float = -2.0,
        gravity_origin_exponent: float = 1.0,
        gravity_destination_exponent: float = 1.0,
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
    ) -> SimulationRunParams:
        return cls(
            start_ts=int(start_ts),
            end_ts=int(end_ts),
            slot_seconds=int(slot_seconds),
            car_speed_kmh=float(car_speed_kmh),
            n_agents=int(n_agents),
            master_seed=int(random_state),
            walking_speed_kmh=float(walking_speed_kmh),
            bike_speed_kmh=float(bike_speed_kmh),
            rho=float(rho),
            gamma=float(gamma),
            alpha=float(alpha),
            dt_update_mob_sim_s=int(dt_update_mob_sim_hours * 3600),
            indipendency_window_s=int(indipendency_window_hours * 3600),
            gravity_deterrence_exponent=float(gravity_deterrence_exponent),
            gravity_origin_exponent=float(gravity_origin_exponent),
            gravity_destination_exponent=float(gravity_destination_exponent),
            dynamic_friendships_enabled=bool(dynamic_friendships_enabled),
            friendship_update_interval_s=int(friendship_update_interval_hours * 3600),
            encounter_window_s=int(encounter_window_hours * 3600),
            regularity_threshold=float(regularity_threshold),
            topological_overlap_threshold=float(topological_overlap_threshold),
            recast_random_baseline_samples=int(recast_random_baseline_samples),
            recast_random_chance_probability=float(recast_random_chance_probability),
            strength_initial=float(strength_initial),
            strength_growth_mu_ln=float(strength_growth_mu_ln),
            strength_growth_sigma_ln=float(strength_growth_sigma_ln),
            strength_decay_rate=float(strength_decay_rate),
            max_dynamic_degree=int(max_dynamic_degree),
            max_colocation_group_size=int(max_colocation_group_size),
        )


@dataclass(slots=True)
class LocationInputs:
    latitudes: np.ndarray
    longitudes: np.ndarray
    relevances: np.ndarray
    distances: np.ndarray | None = None
    location_semantic_cluster_ids: np.ndarray | None = None
    poi_type_choice_enabled: bool = False
    poi_type_alignment_scores: np.ndarray | None = None
    poi_type_alignment_blocks: int = 0
    poi_type_alignment_clusters: int = 0
    poi_type_choice_temperature: float = 0.5
    poi_type_choice_alpha: float = 1.0

    def __post_init__(self) -> None:
        self.latitudes = _contig(self.latitudes, np.float64)
        self.longitudes = _contig(self.longitudes, np.float64)
        self.relevances = _contig(self.relevances, np.float64)
        self.distances = _contig(self.distances, np.float64)
        self.location_semantic_cluster_ids = _contig(
            self.location_semantic_cluster_ids, np.int64
        )
        if self.poi_type_alignment_scores is None:
            self.poi_type_alignment_scores = _empty(np.float64)
        else:
            arr = np.asarray(self.poi_type_alignment_scores, dtype=np.float64)
            if arr.ndim != 3:
                raise ValueError(
                    "poi_type_alignment_scores must have shape "
                    "[clusters, blocks, semantic_clusters]"
                )
            self.poi_type_alignment_blocks = int(arr.shape[1])
            self.poi_type_alignment_clusters = int(arr.shape[2])
            self.poi_type_alignment_scores = np.ascontiguousarray(
                arr.flatten(), dtype=np.float64
            )
        if self.poi_type_choice_enabled:
            if len(self.poi_type_alignment_scores) == 0:
                raise ValueError(
                    "poi_type_alignment_scores is required when "
                    "poi_type_choice_enabled=True"
                )
            if len(self.location_semantic_cluster_ids) != len(self.latitudes):
                raise ValueError(
                    "location_semantic_cluster_ids must have one value per tessellation row"
                )

    @classmethod
    def from_tessellation(
        cls,
        tessellation_df: pd.DataFrame,
        relevance_column: str | None,
        *,
        location_semantic_cluster_ids: np.ndarray | None = None,
        poi_type_choice_enabled: bool = False,
        poi_type_alignment_scores: np.ndarray | None = None,
        poi_type_choice_temperature: float = 0.5,
        poi_type_choice_alpha: float = 1.0,
    ) -> LocationInputs:
        lng_col = "lng" if "lng" in tessellation_df.columns else "lon"
        if relevance_column and relevance_column in tessellation_df.columns:
            relevances = np.asarray(tessellation_df[relevance_column].fillna(0), dtype=float)
            relevances = np.where(relevances == 0, 0.1, relevances)
        else:
            relevances = np.ones(len(tessellation_df), dtype=float)
        return cls(
            latitudes=tessellation_df["lat"].to_numpy(dtype=float),
            longitudes=tessellation_df[lng_col].to_numpy(dtype=float),
            relevances=relevances,
            location_semantic_cluster_ids=location_semantic_cluster_ids,
            poi_type_choice_enabled=poi_type_choice_enabled,
            poi_type_alignment_scores=poi_type_alignment_scores,
            poi_type_choice_temperature=poi_type_choice_temperature,
            poi_type_choice_alpha=poi_type_choice_alpha,
        )


@dataclass(slots=True)
class DiaryInputs:
    diary_timestamps: np.ndarray
    diary_abs_locs: np.ndarray
    diary_starts: np.ndarray
    diary_ends: np.ndarray
    diary_block_ids: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.diary_timestamps = _contig(self.diary_timestamps, np.int64)
        self.diary_abs_locs = _contig(self.diary_abs_locs, np.int32)
        self.diary_starts = _contig(self.diary_starts, np.int64)
        self.diary_ends = _contig(self.diary_ends, np.int64)
        if self.diary_block_ids is None:
            self.diary_block_ids = np.zeros_like(self.diary_abs_locs, dtype=np.int32)
        else:
            self.diary_block_ids = _contig(self.diary_block_ids, np.int32)

    @classmethod
    def from_arrays(cls, diary_arrays: DiaryArrays) -> DiaryInputs:
        if len(diary_arrays) == 5:
            timestamps, abs_locs, starts, ends, block_ids = diary_arrays
        else:
            timestamps, abs_locs, starts, ends = diary_arrays
            block_ids = None
        return cls(timestamps, abs_locs, starts, ends, block_ids)


@dataclass(slots=True)
class InitialLocationInputs:
    starting_locs: np.ndarray | None
    work_tiles: np.ndarray | None
    starting_locs_mode_relevance: bool = False

    def __post_init__(self) -> None:
        self.starting_locs = (
            np.ascontiguousarray(self.starting_locs, dtype=np.int64)
            if self.starting_locs is not None
            else None
        )
        self.work_tiles = _contig(self.work_tiles, np.int64)

    @classmethod
    def build(
        cls,
        *,
        starting_locs: np.ndarray | None,
        work_tiles: np.ndarray | None,
        locations: LocationInputs,
        n_agents: int,
        random_state: int,
        starting_locs_mode_relevance: bool = False,
    ) -> InitialLocationInputs:
        if work_tiles is None:
            work_tiles = sample_weighted_indices(
                locations.relevances, n_agents, np.random.default_rng(random_state)
            )
        return cls(starting_locs, work_tiles, starting_locs_mode_relevance)


@dataclass(slots=True)
class SocialGraphInputs:
    neighbor_starts: np.ndarray
    neighbors: np.ndarray
    edge_profile_sim: np.ndarray | None = None
    profile_embeddings: np.ndarray | None = None
    social_graph_k: int = 20
    edge_weights: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.neighbor_starts = _contig(self.neighbor_starts, np.int64)
        self.neighbors = _contig(self.neighbors, np.int64)
        edge_weights = self.edge_profile_sim if self.edge_weights is None else self.edge_weights
        self.edge_weights = _contig(edge_weights, np.float64)
        self.edge_profile_sim = (
            self.edge_weights if len(self.edge_weights) == len(self.neighbors) else _empty(np.float64)
        )
        self.profile_embeddings = (
            np.ascontiguousarray(self.profile_embeddings, dtype=np.float64)
            if self.profile_embeddings is not None
            else None
        )

    @classmethod
    def build(
        cls,
        *,
        n_agents: int,
        random_state: int,
        locations: LocationInputs,
        initial_locations: InitialLocationInputs,
        profile_embeddings: np.ndarray | None = None,
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
    ) -> SocialGraphInputs:
        if profile_embeddings is not None and initial_locations.starting_locs is not None:
            home_cells = latlng_to_h3_numpy(
                locations.latitudes[initial_locations.starting_locs],
                locations.longitudes[initial_locations.starting_locs],
                home_h3_resolution,
            )
            work_cells = latlng_to_h3_numpy(
                locations.latitudes[initial_locations.work_tiles],
                locations.longitudes[initial_locations.work_tiles],
                work_h3_resolution,
            )
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
        return cls(
            neighbor_starts,
            neighbors,
            edge_profile_sim=edge_weights,
            profile_embeddings=profile_embeddings,
            social_graph_k=social_graph_k,
            edge_weights=edge_weights,
        )


@dataclass(slots=True)
class ActivityInputs:
    act_embs: np.ndarray | None = None
    act_dur_mu: np.ndarray | None = None
    act_dur_sigma: np.ndarray | None = None
    purpose_act_starts: np.ndarray | None = None
    purpose_acts: np.ndarray | None = None
    profile_embs: np.ndarray | None = None
    profile_act_sims: np.ndarray | None = None
    emb_dim: int = 0
    act_kappa: float = 1.0
    act_temp: float = 0.5
    activity_alignment_scores: np.ndarray | None = None
    activity_cluster_labels: np.ndarray | None = None
    activity_alignment_clusters: int = 0
    activity_alignment_blocks: int = 0
    activity_alignment_previous: int = 0
    poi_semantic_scores: np.ndarray | None = None
    poi_mask_starts: np.ndarray | None = None
    poi_mask_activities: np.ndarray | None = None
    poi_semantic_clusters: int = 0
    activity_history_weight: float = 1.0
    materialize_travel: bool = True

    def __post_init__(self) -> None:
        self.act_dur_mu = _contig(self.act_dur_mu, np.float64)
        self.act_dur_sigma = _contig(self.act_dur_sigma, np.float64)
        self.purpose_act_starts = _contig(self.purpose_act_starts, np.int64)
        self.purpose_acts = _contig(self.purpose_acts, np.int64)

        if self.act_embs is not None and self.profile_embs is not None and len(self.act_dur_mu) > 0:
            act_embs = np.asarray(self.act_embs, dtype=np.float64)
            profile_embs = np.asarray(self.profile_embs, dtype=np.float64)
            self.emb_dim = act_embs.shape[1] if act_embs.ndim == 2 else 0
            if self.emb_dim > 0:
                self.act_embs = np.ascontiguousarray(act_embs.flatten(), dtype=np.float64)
                self.profile_embs = np.ascontiguousarray(profile_embs.flatten(), dtype=np.float64)
                self.profile_act_sims = np.ascontiguousarray(
                    (profile_embs.astype(np.float64) @ act_embs.astype(np.float64).T).flatten(),
                    dtype=np.float64,
                )
            else:
                self.act_embs = _empty(np.float64)
                self.profile_embs = _empty(np.float64)
                self.profile_act_sims = _empty(np.float64)
        else:
            self.act_embs = _empty(np.float64)
            self.profile_embs = _empty(np.float64)
            self.profile_act_sims = _empty(np.float64)

        labels = self.activity_cluster_labels
        self.activity_cluster_labels = _empty(np.int64)
        if self.activity_alignment_scores is None:
            self.activity_alignment_scores = _empty(np.float64)
            if labels is not None:
                self.activity_cluster_labels = _contig(labels, np.int64)
        else:
            arr = np.asarray(self.activity_alignment_scores, dtype=np.float64)
            if arr.ndim != 4:
                raise ValueError(
                    "activity_alignment_scores must have shape "
                    "[clusters, blocks, previous, activities]"
                )
            if labels is None:
                raise ValueError(
                    "activity_cluster_labels is required when "
                    "activity_alignment_scores is provided"
                )
            self.activity_alignment_clusters = int(arr.shape[0])
            self.activity_alignment_blocks = int(arr.shape[1])
            self.activity_alignment_previous = int(arr.shape[2])
            self.activity_alignment_scores = np.ascontiguousarray(
                arr.flatten(), dtype=np.float64
            )
            self.activity_cluster_labels = _contig(labels, np.int64)

        if self.poi_semantic_scores is None:
            self.poi_semantic_scores = _empty(np.float64)
        else:
            arr = np.asarray(self.poi_semantic_scores, dtype=np.float64)
            if arr.ndim != 3:
                raise ValueError(
                    "poi_semantic_scores must have shape "
                    "[clusters, semantic_clusters, activities]"
                )
            if labels is None:
                raise ValueError(
                    "activity_cluster_labels is required when poi_semantic_scores is provided"
                )
            if self.activity_alignment_clusters not in (0, int(arr.shape[0])):
                raise ValueError(
                    "poi_semantic_scores cluster dimension must match "
                    "activity_alignment_scores"
                )
            self.activity_alignment_clusters = int(arr.shape[0])
            self.poi_semantic_clusters = int(arr.shape[1])
            self.poi_semantic_scores = np.ascontiguousarray(arr.flatten(), dtype=np.float64)
            if len(self.activity_cluster_labels) == 0:
                self.activity_cluster_labels = _contig(labels, np.int64)
        self.poi_mask_starts = _contig(self.poi_mask_starts, np.int64)
        self.poi_mask_activities = _contig(self.poi_mask_activities, np.int64)

    @classmethod
    def build(
        cls,
        *,
        n_agents: int,
        act_embs: np.ndarray | None = None,
        act_dur_mu: np.ndarray | None = None,
        act_dur_sigma: np.ndarray | None = None,
        purpose_act_starts: np.ndarray | None = None,
        purpose_acts: np.ndarray | None = None,
        profile_embeddings: np.ndarray | None = None,
        act_kappa: float = 1.0,
        act_temp: float = 0.5,
        activity_alignment_scores: np.ndarray | None = None,
        activity_cluster_labels: np.ndarray | None = None,
        poi_semantic_scores: np.ndarray | None = None,
        poi_mask_starts: np.ndarray | None = None,
        poi_mask_activities: np.ndarray | None = None,
        activity_history_weight: float = 1.0,
        materialize_travel: bool = True,
    ) -> ActivityInputs:
        result = cls(
            act_embs=act_embs,
            act_dur_mu=act_dur_mu,
            act_dur_sigma=act_dur_sigma,
            purpose_act_starts=purpose_act_starts,
            purpose_acts=purpose_acts,
            profile_embs=profile_embeddings,
            act_kappa=act_kappa,
            act_temp=act_temp,
            activity_alignment_scores=activity_alignment_scores,
            activity_cluster_labels=activity_cluster_labels,
            poi_semantic_scores=poi_semantic_scores,
            poi_mask_starts=poi_mask_starts,
            poi_mask_activities=poi_mask_activities,
            activity_history_weight=activity_history_weight,
            materialize_travel=materialize_travel,
        )
        if len(result.activity_cluster_labels) not in (0, n_agents):
            raise ValueError("activity_cluster_labels must have one label per agent")
        return result


@dataclass(slots=True)
class NetworkInputs:
    edge_from: np.ndarray | None = None
    edge_to: np.ndarray | None = None
    edge_weight_ds: np.ndarray | None = None
    node_lats: np.ndarray | None = None
    node_lngs: np.ndarray | None = None
    location_node: np.ndarray | None = None
    max_leg_waypoints: int = 16
    n_locations: int = 0

    def __post_init__(self) -> None:
        enabled = (
            self.edge_from is not None
            and self.edge_to is not None
            and self.edge_weight_ds is not None
            and len(self.edge_from) > 0
        )
        if enabled:
            self.edge_from = _contig(self.edge_from, np.int64)
            self.edge_to = _contig(self.edge_to, np.int64)
            self.edge_weight_ds = _contig(self.edge_weight_ds, np.int64)
            self.node_lats = _contig(self.node_lats, np.float64)
            self.node_lngs = _contig(self.node_lngs, np.float64)
            self.location_node = (
                np.full(self.n_locations, -1, dtype=np.int64)
                if self.location_node is None
                else _contig(self.location_node, np.int64)
            )
        else:
            self.edge_from = _empty(np.int64)
            self.edge_to = _empty(np.int64)
            self.edge_weight_ds = _empty(np.int64)
            self.node_lats = _empty(np.float64)
            self.node_lngs = _empty(np.float64)
            self.location_node = _empty(np.int64)
        self.max_leg_waypoints = int(self.max_leg_waypoints)

    @classmethod
    def from_arrays(
        cls,
        *,
        n_locations: int,
        edge_from: np.ndarray | None = None,
        edge_to: np.ndarray | None = None,
        edge_weight_ds: np.ndarray | None = None,
        node_lats: np.ndarray | None = None,
        node_lngs: np.ndarray | None = None,
        location_node: np.ndarray | None = None,
        max_leg_waypoints: int = 16,
    ) -> NetworkInputs:
        return cls(
            edge_from=edge_from,
            edge_to=edge_to,
            edge_weight_ds=edge_weight_ds,
            node_lats=node_lats,
            node_lngs=node_lngs,
            location_node=location_node,
            max_leg_waypoints=max_leg_waypoints,
            n_locations=n_locations,
        )


@dataclass(slots=True)
class TransportInputs:
    has_car: np.ndarray | None
    has_bike: np.ndarray | None
    walking_threshold_km: np.ndarray
    bike_threshold_km: np.ndarray

    def __post_init__(self) -> None:
        self.has_car = _contig(self.has_car, np.bool_)
        self.has_bike = _contig(self.has_bike, np.bool_)
        self.walking_threshold_km = _contig(self.walking_threshold_km, np.float64)
        self.bike_threshold_km = _contig(self.bike_threshold_km, np.float64)

    @classmethod
    def build(
        cls,
        *,
        n_agents: int,
        random_state: int,
        has_car: np.ndarray | None = None,
        has_bike: np.ndarray | None = None,
        walking_threshold_mu_ln_km: float = -0.35,
        walking_threshold_sigma_ln: float = 0.45,
        bike_threshold_mu_ln_km: float = 1.4,
        bike_threshold_sigma_ln: float = 0.55,
    ) -> TransportInputs:
        rng = np.random.default_rng(random_state)
        walking = rng.lognormal(
            mean=float(walking_threshold_mu_ln_km),
            sigma=float(walking_threshold_sigma_ln),
            size=n_agents,
        )
        bike = rng.lognormal(
            mean=float(bike_threshold_mu_ln_km),
            sigma=float(bike_threshold_sigma_ln),
            size=n_agents,
        )
        return cls(
            np.ones(n_agents, dtype=np.bool_) if has_car is None else has_car,
            np.zeros(n_agents, dtype=np.bool_) if has_bike is None else has_bike,
            walking,
            np.maximum(bike, walking),
        )


@dataclass(slots=True)
class OutputHooks:
    on_day_flush: Callable[[pl.DataFrame], None] | None = None
    on_encounter_day_flush: Callable[[pl.DataFrame], None] | None = None
    on_trip_day_flush: Callable[[pl.DataFrame], None] | None = None
    on_activity_day_flush: Callable[[pl.DataFrame], None] | None = None
