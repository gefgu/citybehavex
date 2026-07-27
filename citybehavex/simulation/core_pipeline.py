from __future__ import annotations

from typing import Optional

import fastmob
import numpy as np
import pandas as pd
import typer
from fastmob.models import DensityEPR

from citybehavex.activities import ProfileClusters
from citybehavex.config import CityBehavExConfig
from citybehavex.profiles import AgentProfile
from citybehavex.schedules import DiaryBank
from citybehavex.simulation.activity_pipeline import _build_activity_data
from citybehavex.simulation.core import simulate_agents, social_network_sidecar_path
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
from citybehavex.simulation.network_pipeline import build_rail_network_kwargs, build_road_network_kwargs
from citybehavex.simulation.output import _IncrementalParquetWriter


def _run_density_epr(
    config: CityBehavExConfig,
    tessellation_df: pd.DataFrame,
    relevance_column: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> tuple[fastmob.TrajDataFrame, Optional[str]]:
    typer.echo(
        f"Running DensityEPR: {config.simulation.agents} agents x {config.simulation.days} days "
        f"({start_date.date()} -> {end_date.date()})"
    )
    model = DensityEPR()
    traj = model.generate(
        start_date=start_date,
        end_date=end_date,
        spatial_tessellation=tessellation_df,
        n_agents=config.simulation.agents,
        relevance_column=relevance_column,
        random_state=config.simulation.random_state,
    )
    traj = fastmob.TrajDataFrame(traj)
    synth_activity_col = None
    if "purpose" in tessellation_df.columns:
        traj.df = _merge_tessellation_metadata(
            traj.df,
            tessellation_df,
            ["tile_id", "purpose", "category"],
        )
        synth_activity_col = "purpose"
    return traj, synth_activity_col


def _merge_tessellation_metadata(
    df: pd.DataFrame,
    tessellation_df: pd.DataFrame,
    candidate_cols: list[str],
) -> pd.DataFrame:
    extra_cols = [c for c in candidate_cols if c in tessellation_df.columns and c not in df.columns]
    if not extra_cols:
        return df
    lookup = tessellation_df[["lat", "lng"] + extra_cols].drop_duplicates(["lat", "lng"])
    return df.merge(lookup, on=["lat", "lng"], how="left")


def _run_simulation_core(
    config: CityBehavExConfig,
    tessellation_df: pd.DataFrame,
    relevance_column: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    diary_arrays: tuple,
    timing: CoreTiming,
    profiles: Optional[list[AgentProfile]] = None,
    profile_embeddings: Optional[np.ndarray] = None,
    bank: Optional[DiaryBank] = None,
    profile_clusters: Optional[ProfileClusters] = None,
    output_path: Optional[str] = None,
) -> tuple[fastmob.TrajDataFrame, Optional[str], bool]:
    granularity = config.simulation.granularity_minutes
    typer.echo(
        f"Running simulation core: {config.simulation.agents} agents x {config.simulation.days} days "
        f"@ {granularity}-min slots, {config.simulation.car_speed_kmh:.0f} km/h car "
        f"({start_date.date()} -> {end_date.date()})"
    )
    home_tiles = (
        np.array([p.home_tile for p in profiles], dtype=np.int64)
        if profiles is not None
        else None
    )
    work_tiles = (
        np.array([p.work_tile for p in profiles], dtype=np.int64)
        if profiles is not None
        else None
    )
    (
        act_embs,
        act_dur_mu,
        act_dur_sigma,
        purpose_act_starts,
        purpose_acts,
        activity_alignment_scores,
        activity_cluster_labels,
        poi_semantic_scores,
        location_semantic_cluster_ids,
        poi_mask_starts,
        poi_mask_activities,
        poi_type_alignment_scores,
    ) = _build_activity_data(
        config,
        tessellation_df=tessellation_df,
        bank=bank,
        profile_clusters=profile_clusters,
        output_path=output_path,
    )

    road_kwargs = build_road_network_kwargs(config, tessellation_df)
    rail_kwargs = build_rail_network_kwargs(config, tessellation_df)

    base = output_path or config.simulation.output
    moving_path = base.replace(".parquet", "_moving.parquet")
    transport_paths_enabled = config.road_network.enabled or config.rail_network.enabled
    stream_moving = config.simulation.stream_output and transport_paths_enabled
    moving_writer = _IncrementalParquetWriter(moving_path) if stream_moving else None
    on_day_flush = moving_writer.write if moving_writer is not None else None

    enc_path = base.replace(".parquet", "_encounters.parquet")
    stream_encounters = config.simulation.stream_output
    encounters_writer = _IncrementalParquetWriter(enc_path) if stream_encounters else None
    on_encounter_day_flush = encounters_writer.write if encounters_writer is not None else None

    # Streaming the main stop table means this function -- not run_simulation
    # -- owns writing `base` incrementally; run_simulation skips its own
    # to_parquet call in that case (nothing else consumes the returned
    # TrajDataFrame's `.df`, confirmed via callers).
    stream_trip = config.simulation.stream_output
    trip_writer = _IncrementalParquetWriter(base) if stream_trip else None
    on_trip_day_flush = None
    if trip_writer is not None:

        def on_trip_day_flush(chunk_df):
            trip_writer.write(
                _merge_tessellation_metadata(
                    chunk_df.to_pandas(), tessellation_df, ["tile_id", "category"]
                )
            )

    act_path = base.replace(".parquet", "_activities.parquet")
    stream_activities = config.simulation.stream_output
    activities_writer = _IncrementalParquetWriter(act_path) if stream_activities else None
    on_activity_day_flush = activities_writer.write if activities_writer is not None else None

    profile_types = [p.job for p in profiles] if profiles is not None else None
    has_car = np.asarray([p.has_car for p in profiles], dtype=np.bool_) if profiles is not None else None
    has_bike = np.asarray([p.has_bike for p in profiles], dtype=np.bool_) if profiles is not None else None
    locations = LocationInputs.from_tessellation(
        tessellation_df,
        relevance_column,
        location_semantic_cluster_ids=location_semantic_cluster_ids,
        poi_type_choice_enabled=config.activities.poi_type_choice_enabled
        and poi_type_alignment_scores is not None,
        poi_type_alignment_scores=poi_type_alignment_scores,
        poi_type_choice_temperature=config.activities.poi_type_choice_temperature,
        poi_type_choice_alpha=config.activities.poi_type_choice_alpha,
    )
    params = SimulationRunParams.from_hours(
        start_ts=int(start_date.timestamp()),
        end_ts=int(end_date.timestamp()),
        slot_seconds=granularity * 60,
        car_speed_kmh=config.simulation.car_speed_kmh,
        walking_speed_kmh=config.simulation.walking_speed_kmh,
        bike_speed_kmh=config.simulation.bike_speed_kmh,
        n_agents=config.simulation.agents,
        random_state=config.simulation.random_state,
        rho=config.simulation.rho,
        gamma=config.simulation.gamma,
        alpha=config.simulation.alpha,
        dt_update_mob_sim_hours=config.simulation.dt_update_mob_sim_hours,
        indipendency_window_hours=config.simulation.indipendency_window_hours,
        gravity_deterrence_exponent=config.simulation.gravity_deterrence_exponent,
        gravity_origin_exponent=config.simulation.gravity_origin_exponent,
        gravity_destination_exponent=config.simulation.gravity_destination_exponent,
        dynamic_friendships_enabled=config.social.dynamic_friendships_enabled,
        friendship_update_interval_hours=config.social.friendship_update_interval_hours,
        encounter_window_hours=config.social.encounter_window_hours,
        regularity_threshold=config.social.regularity_threshold,
        topological_overlap_threshold=config.social.topological_overlap_threshold,
        recast_random_baseline_samples=config.social.recast_random_baseline_samples,
        recast_random_chance_probability=config.social.recast_random_chance_probability,
        strength_initial=config.social.strength_initial,
        strength_growth_mu_ln=config.social.strength_growth_mu_ln,
        strength_growth_sigma_ln=config.social.strength_growth_sigma_ln,
        strength_decay_rate=config.social.strength_decay_rate,
        max_dynamic_degree=config.social.max_dynamic_degree,
        max_colocation_group_size=config.social.max_colocation_group_size,
    )
    initial_locations = InitialLocationInputs.build(
        starting_locs=home_tiles,
        work_tiles=work_tiles,
        locations=locations,
        n_agents=config.simulation.agents,
        random_state=config.simulation.random_state,
    )
    social_graph_inputs = SocialGraphInputs.build(
        n_agents=config.simulation.agents,
        random_state=config.simulation.random_state,
        locations=locations,
        initial_locations=initial_locations,
        profile_embeddings=profile_embeddings,
        social_graph_k=config.social.social_graph_k,
        profile_graph_exact_threshold=config.social.profile_graph_exact_threshold,
        home_h3_resolution=config.social.home_h3_resolution,
        work_h3_resolution=config.social.work_h3_resolution,
        degree_mu_ln=config.social.degree_mu_ln,
        degree_sigma_ln=config.social.degree_sigma_ln,
        max_degree=config.social.max_degree,
        similarity_temperature=config.social.similarity_temperature,
        max_candidate_pool=config.social.max_candidate_pool,
        max_ring_expansion=config.social.max_ring_expansion,
    )
    activity_inputs = ActivityInputs.build(
        n_agents=config.simulation.agents,
        act_embs=act_embs,
        act_dur_mu=act_dur_mu,
        act_dur_sigma=act_dur_sigma,
        purpose_act_starts=purpose_act_starts,
        purpose_acts=purpose_acts,
        profile_embeddings=profile_embeddings,
        act_kappa=config.activities.kappa,
        act_temp=config.activities.temperature,
        activity_alignment_scores=activity_alignment_scores,
        activity_cluster_labels=activity_cluster_labels,
        poi_semantic_scores=poi_semantic_scores,
        poi_mask_starts=poi_mask_starts,
        poi_mask_activities=poi_mask_activities,
        activity_history_weight=config.activities.history_weight,
        materialize_travel=config.activities.materialize_travel,
    )
    transport_inputs = TransportInputs.build(
        n_agents=config.simulation.agents,
        random_state=config.simulation.random_state,
        has_car=has_car,
        has_bike=has_bike,
        walking_threshold_mu_ln_km=config.simulation.walking_threshold_mu_ln_km,
        walking_threshold_sigma_ln=config.simulation.walking_threshold_sigma_ln,
        bike_threshold_mu_ln_km=config.simulation.bike_threshold_mu_ln_km,
        bike_threshold_sigma_ln=config.simulation.bike_threshold_sigma_ln,
    )
    df, encounters, moving, activities, social_graph = simulate_agents(
        locations,
        DiaryInputs.from_arrays(diary_arrays),
        params,
        initial_locations=initial_locations,
        social_graph=social_graph_inputs,
        activities=activity_inputs,
        transport=transport_inputs,
        road_network=NetworkInputs.from_arrays(n_locations=len(tessellation_df), **road_kwargs),
        rail_network=NetworkInputs.from_arrays(n_locations=len(tessellation_df), **rail_kwargs),
        output_hooks=OutputHooks(
            on_day_flush=on_day_flush,
            on_encounter_day_flush=on_encounter_day_flush,
            on_trip_day_flush=on_trip_day_flush,
            on_activity_day_flush=on_activity_day_flush,
        ),
        timing=timing,
        return_social_graph=True,
        social_node_profiles=profile_types,
    )
    social_path = social_network_sidecar_path(base)
    social_graph.write_json(social_path)
    typer.echo(
        f"Saved social network ({social_graph.metadata['node_count']:,} nodes, "
        f"{social_graph.metadata['edge_count']:,} edges) -> {social_path}"
    )
    if encounters_writer is not None:
        # `encounters` here is only the tail since the last flush -- earlier
        # days were already streamed to disk via on_encounter_day_flush.
        encounters_writer.write(encounters)
        encounters_writer.close()
        if encounters_writer.rows_written > 0:
            typer.echo(
                f"Saved {encounters_writer.rows_written:,} encounters (streamed) -> {enc_path}"
            )
    elif len(encounters) > 0:
        encounters.write_parquet(enc_path)
        typer.echo(f"Saved {len(encounters):,} encounters -> {enc_path}")
    if moving_writer is not None:
        # `moving` here is only the final day's still-open tail -- everything
        # closed before it was already streamed to disk via on_day_flush.
        moving_writer.write(moving)
        moving_writer.close()
        if moving_writer.rows_written > 0:
            typer.echo(f"Saved {moving_writer.rows_written:,} waypoints (streamed) -> {moving_path}")
    elif transport_paths_enabled and len(moving) > 0:
        moving.write_parquet(moving_path)
        typer.echo(f"Saved {len(moving):,} waypoints -> {moving_path}")
    if activities_writer is not None:
        # `activities` here is only the tail since the last flush -- earlier
        # days were already streamed to disk via on_activity_day_flush.
        activities_writer.write(activities)
        activities_writer.close()
        if activities_writer.rows_written > 0:
            typer.echo(f"Saved {activities_writer.rows_written:,} activities (streamed) -> {act_path}")
    elif config.activities.enabled and len(activities) > 0:
        activities.write_parquet(act_path)
        typer.echo(f"Saved {len(activities):,} activities -> {act_path}")

    df = _merge_tessellation_metadata(df.to_pandas(), tessellation_df, ["tile_id", "category"])
    if trip_writer is not None:
        # `df` here is only the tail (the final, possibly-partial day plus
        # every agent's still-open stop) -- earlier days were already
        # streamed to disk via on_trip_day_flush.
        trip_writer.write(df)
        trip_writer.close()
        typer.echo(
            f"Saved {trip_writer.rows_written:,} records "
            f"({config.simulation.agents} agents) -> {base}"
        )
    traj = fastmob.TrajDataFrame(
        df, datetime_col="datetime", lat_col="lat", lng_col="lng", uid_col="uid"
    )
    return traj, "purpose", trip_writer is not None
