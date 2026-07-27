"""Project-owned simulation core driver for citybehavex."""

from __future__ import annotations

import time
import numpy as np
import polars as pl
import citybehavex._core as _cbx_core
from citybehavex.simulation.inputs import (
    ActivityInputs,
    CoreTiming,
    DiaryInputs,
    InitialLocationInputs,
    LocationInputs,
    NetworkInputs,
    OutputHooks,
    SimulationRunParams,
    SocialGraphInputs,
    TransportInputs,
)
from citybehavex.simulation.frames import (
    _build_activity_frame,
    _build_encounters_frame,
    _build_moving_frame,
    _build_trip_frame,
)
from citybehavex.social.social_artifact import (
    SocialGraphArtifact,
    build_social_graph_artifact,
    social_network_sidecar_path,
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
    "SocialGraphArtifact",
    "SocialGraphInputs",
    "TransportInputs",
    "build_social_graph_artifact",
    "simulate_agents",
    "social_network_sidecar_path",
]


def simulate_agents(
    locations: LocationInputs,
    diary: DiaryInputs,
    params: SimulationRunParams,
    *,
    initial_locations: InitialLocationInputs | None = None,
    social_graph: SocialGraphInputs | None = None,
    activities: ActivityInputs | None = None,
    transport: TransportInputs | None = None,
    road_network: NetworkInputs | None = None,
    rail_network: NetworkInputs | None = None,
    output_hooks: OutputHooks | None = None,
    timing: CoreTiming | None = None,
    return_social_graph: bool = False,
    social_node_profiles: list[str] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame] | tuple[
    pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, SocialGraphArtifact
]:
    """Run the CityBehavEx simulation core."""
    random_state = int(params.master_seed or 0)
    if initial_locations is None:
        initial_locations = InitialLocationInputs.build(
            starting_locs=None,
            work_tiles=None,
            locations=locations,
            n_agents=params.n_agents,
            random_state=random_state,
        )
    if social_graph is None:
        social_graph = SocialGraphInputs.build(
            n_agents=params.n_agents,
            random_state=random_state,
            locations=locations,
            initial_locations=initial_locations,
        )
    if activities is None:
        activities = ActivityInputs()
    if transport is None:
        transport = TransportInputs.build(n_agents=params.n_agents, random_state=random_state)
    if road_network is None:
        road_network = NetworkInputs(n_locations=len(locations.latitudes))
    if rail_network is None:
        rail_network = NetworkInputs(n_locations=len(locations.latitudes))
    if output_hooks is None:
        output_hooks = OutputHooks()

    social_graph_artifact = None
    if return_social_graph:
        social_graph_artifact = build_social_graph_artifact(
            social_graph.neighbor_starts,
            social_graph.neighbors,
            social_graph.edge_weights,
            n_agents=params.n_agents,
            random_state=random_state,
            social_graph_k=social_graph.social_graph_k,
            profile_embeddings=social_graph.profile_embeddings,
            profile_types=social_node_profiles,
        )

    rust_on_day_flush = None
    if output_hooks.on_day_flush is not None:

        def rust_on_day_flush(agent, dest_stop_id, seq, lat, lng, t, mode):
            output_hooks.on_day_flush(
                _build_moving_frame(agent, dest_stop_id, seq, lat, lng, t, mode)
            )

    rust_on_encounter_day_flush = None
    if output_hooks.on_encounter_day_flush is not None:

        def rust_on_encounter_day_flush(agent, contact, tile, ts):
            output_hooks.on_encounter_day_flush(
                _build_encounters_frame(agent, contact, tile, ts)
            )

    rust_on_trip_day_flush = None
    if output_hooks.on_trip_day_flush is not None:

        def rust_on_trip_day_flush(agent, loc_id, arrival, departure, duration, stop_id, abstract_loc):
            output_hooks.on_trip_day_flush(
                _build_trip_frame(
                    agent,
                    loc_id,
                    arrival,
                    departure,
                    duration,
                    stop_id,
                    abstract_loc,
                    locations.latitudes,
                    locations.longitudes,
                )
            )

    rust_on_activity_day_flush = None
    if output_hooks.on_activity_day_flush is not None:

        def rust_on_activity_day_flush(agent, stop_id, seq, activity, arrival, departure, block_id):
            output_hooks.on_activity_day_flush(
                _build_activity_frame(agent, stop_id, seq, activity, arrival, departure, block_id)
            )

    start = time.perf_counter()
    rust_result = _cbx_core.simulation_core_simulate_agents(
        locations,
        social_graph,
        diary,
        params,
        initial_locations,
        activities,
        road_network,
        rail_network,
        transport,
        rust_on_day_flush,
        rust_on_encounter_day_flush,
        rust_on_trip_day_flush,
        rust_on_activity_day_flush,
        bool(return_social_graph),
    )
    if return_social_graph:
        (
            (
                agent_ids,
                loc_id,
                arrival,
                departure,
                trip_dur,
                enc_agent,
                enc_contact,
                enc_tile,
                enc_ts,
                stop_abstract_loc,
            ),
            (
                stop_id,
                path_agent,
                path_stop_id,
                path_seq,
                path_lat,
                path_lng,
                path_t,
                path_mode,
            ),
            (
                act_agent,
                act_stop_id,
                act_seq,
                act_activity,
                act_arrival,
                act_departure,
                act_block_id,
            ),
            (social_source, social_target, social_weight, social_kind),
        ) = rust_result
    else:
        social_source = social_target = social_weight = social_kind = None
        (
            (
                agent_ids,
                loc_id,
                arrival,
                departure,
                trip_dur,
                enc_agent,
                enc_contact,
                enc_tile,
                enc_ts,
                stop_abstract_loc,
            ),
            (
                stop_id,
                path_agent,
                path_stop_id,
                path_seq,
                path_lat,
                path_lng,
                path_t,
                path_mode,
            ),
            (
                act_agent,
                act_stop_id,
                act_seq,
                act_activity,
                act_arrival,
                act_departure,
                act_block_id,
            ),
        ) = rust_result
    elapsed = time.perf_counter() - start
    if timing is not None:
        timing.seconds += elapsed

    trajectories = _build_trip_frame(
        agent_ids,
        loc_id,
        arrival,
        departure,
        trip_dur,
        stop_id,
        stop_abstract_loc,
        locations.latitudes,
        locations.longitudes,
    )
    encounters = _build_encounters_frame(enc_agent, enc_contact, enc_tile, enc_ts)
    moving = _build_moving_frame(
        path_agent, path_stop_id, path_seq, path_lat, path_lng, path_t, path_mode
    )
    activity_frame = _build_activity_frame(
        act_agent,
        act_stop_id,
        act_seq,
        act_activity,
        act_arrival,
        act_departure,
        act_block_id,
    )

    if social_graph_artifact is not None:
        if social_source is not None and social_target is not None and social_weight is not None:
            social_graph_artifact = build_social_graph_artifact(
                social_graph.neighbor_starts,
                social_graph.neighbors,
                np.asarray(social_weight, dtype=np.float64),
                n_agents=params.n_agents,
                random_state=random_state,
                social_graph_k=social_graph.social_graph_k,
                profile_embeddings=social_graph.profile_embeddings,
                profile_types=social_node_profiles,
                edge_sources=np.asarray(social_source, dtype=np.int64),
                edge_targets=np.asarray(social_target, dtype=np.int64),
                edge_kinds=np.asarray(social_kind, dtype=np.uint8) if social_kind is not None else None,
            )
        return trajectories, encounters, moving, activity_frame, social_graph_artifact
    return trajectories, encounters, moving, activity_frame
