from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import skmob2
import typer
from skmob2.models import DensityEPR

from citybehavex.activities import activity_descriptions, activity_duration_arrays, build_eligibility_csr
from citybehavex.config import CityBehavExConfig
from citybehavex.embedding import embed_profiles, embed_texts
from citybehavex.llm_diaries import DiaryBatch, LLMStats, allocate_location_counts, fetch_diary_batch
from citybehavex.profiles import AgentProfile, generate_profiles, load_profiles, profile_to_narrative, profiles_to_frame
from citybehavex.roads import build_road_graph, snap_locations_to_graph
from citybehavex.schedules import DiaryBank, build_ddcrp_diary, build_diary_bank
from citybehavex.simulation.core import CoreTiming, simulate_agents, social_network_sidecar_path
from citybehavex.tessellation import build_poi_tessellation, build_tessellation, purpose_distribution


def load_or_build_tessellation(config: CityBehavExConfig) -> tuple[pd.DataFrame, str]:
    tessellation_df, relevance_column = _load_or_build_tessellation_df(config)
    tessellation_df = _maybe_snap_to_roads(config, tessellation_df)
    return tessellation_df, relevance_column


def _maybe_snap_to_roads(config: CityBehavExConfig, tessellation_df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``road_node`` column mapping each location to its nearest road graph node.

    The snapping result is cached separately from the tessellation file itself,
    since ``sim.tessellation``/``tess.path`` may point at a file we don't own.
    """
    rn = config.road_network
    if not rn.enabled:
        return tessellation_df

    if Path(rn.snap_output).exists():
        snap_df = pd.read_parquet(rn.snap_output)
        if len(snap_df) == len(tessellation_df):
            typer.echo(f"Loading cached road-node snapping from {rn.snap_output} ...")
            return tessellation_df.assign(road_node=snap_df["road_node"].to_numpy())
        typer.echo(
            f"Warning: cached road-node snapping at {rn.snap_output} has "
            f"{len(snap_df):,} rows but tessellation has {len(tessellation_df):,} — rebuilding"
        )

    sim = config.simulation
    tess = config.tessellation
    min_lon = sim.min_lon if sim.min_lon is not None else tess.min_lon
    min_lat = sim.min_lat if sim.min_lat is not None else tess.min_lat
    max_lon = sim.max_lon if sim.max_lon is not None else tess.max_lon
    max_lat = sim.max_lat if sim.max_lat is not None else tess.max_lat
    lng_col = "lng" if "lng" in tessellation_df.columns else "lon"
    if None in [min_lon, min_lat, max_lon, max_lat]:
        min_lon, max_lon = float(tessellation_df[lng_col].min()), float(tessellation_df[lng_col].max())
        min_lat, max_lat = float(tessellation_df["lat"].min()), float(tessellation_df["lat"].max())

    overture_release = rn.overture_release or tess.overture_release
    nodes_df, _edges_df = build_road_graph(
        min_lon, min_lat, max_lon, max_lat, overture_release, rn.nodes_output, rn.edges_output
    )
    road_node = snap_locations_to_graph(
        tessellation_df, nodes_df, rn.snap_max_distance_m, lat_col="lat", lng_col=lng_col
    )
    n_unsnapped = int((road_node < 0).sum())
    if n_unsnapped:
        typer.echo(
            f"Warning: {n_unsnapped:,}/{len(road_node):,} locations are farther than "
            f"{rn.snap_max_distance_m:.0f}m from the road graph and will fall back to "
            "straight-line routing for trips touching them"
        )

    Path(rn.snap_output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"road_node": road_node}).to_parquet(rn.snap_output, index=False)
    return tessellation_df.assign(road_node=road_node)


def _load_or_build_tessellation_df(config: CityBehavExConfig) -> tuple[pd.DataFrame, str]:
    sim = config.simulation
    tess = config.tessellation
    tessellation_path = sim.tessellation or tess.path

    if tessellation_path:
        typer.echo(f"Loading tessellation from {tessellation_path} ...")
        tessellation_df = pd.read_parquet(tessellation_path)
        relevance_column = sim.relevance_column or tess.relevance_column
        if tess.min_poi_count > 0 and relevance_column in tessellation_df.columns:
            n_before = len(tessellation_df)
            tessellation_df = tessellation_df[
                tessellation_df[relevance_column] >= tess.min_poi_count
            ].reset_index(drop=True)
            n_dropped = n_before - len(tessellation_df)
            if n_dropped:
                typer.echo(
                    f"Dropped {n_dropped:,} cells with {relevance_column} < {tess.min_poi_count} "
                    f"({len(tessellation_df):,} remaining)"
                )
        return tessellation_df, relevance_column

    if tess.output and Path(tess.output).exists():
        typer.echo(f"Loading cached generated tessellation from {tess.output} ...")
        tessellation_df = pd.read_parquet(tess.output)
        relevance_column = sim.relevance_column or tess.relevance_column
        if tess.poi_tessellation and relevance_column == "total_poi_count" and "relevance" in tessellation_df.columns:
            relevance_column = "relevance"
        return tessellation_df, relevance_column

    min_lon = sim.min_lon if sim.min_lon is not None else tess.min_lon
    min_lat = sim.min_lat if sim.min_lat is not None else tess.min_lat
    max_lon = sim.max_lon if sim.max_lon is not None else tess.max_lon
    max_lat = sim.max_lat if sim.max_lat is not None else tess.max_lat
    if None in [min_lon, min_lat, max_lon, max_lat]:
        raise ValueError(
            "provide a tessellation path or all four bbox values "
            "(min_lon, min_lat, max_lon, max_lat)"
        )

    if tess.poi_tessellation:
        tessellation_df = build_poi_tessellation(
            min_lon, min_lat, max_lon, max_lat, tess.overture_release
        )
        typer.echo(f"Generated {len(tessellation_df):,} POI tiles from bbox")
    else:
        tessellation_df = build_tessellation(
            min_lon,
            min_lat,
            max_lon,
            max_lat,
            tess.resolution,
            tess.enrich_overture,
            tess.overture_release,
            min_poi_count=tess.min_poi_count,
        )
        typer.echo(f"Generated {len(tessellation_df):,} H3 cells from bbox")

    if tess.output:
        Path(tess.output).parent.mkdir(parents=True, exist_ok=True)
        tessellation_df.to_parquet(tess.output, index=False)
        typer.echo(f"Saved generated tessellation -> {tess.output}")

    relevance_column = sim.relevance_column or tess.relevance_column
    if tess.poi_tessellation and relevance_column == "total_poi_count" and "relevance" in tessellation_df.columns:
        relevance_column = "relevance"
    return tessellation_df, relevance_column


def simulation_dates(config: CityBehavExConfig) -> tuple[pd.Timestamp, pd.Timestamp]:
    if config.simulation.start_date:
        start_date = pd.Timestamp(config.simulation.start_date)
    else:
        start_date = pd.Timestamp(
            datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        )
    return start_date, start_date + timedelta(days=config.simulation.days)


def maybe_build_diaries(
    config: CityBehavExConfig,
    tessellation_df: pd.DataFrame,
) -> Optional[tuple[dict[str, DiaryBatch], LLMStats, float]]:
    """Fetch one diary batch per day type (weekday/weekend), or None if no LLM
    client and no validated cache are configured."""
    valid_cache = config.llm.validated_diaries_path
    has_llm_client = all([config.llm.base_url, config.llm.api_key, config.llm.model])
    if not has_llm_client and not valid_cache:
        return None

    started = time.perf_counter()
    stats = LLMStats()
    distribution = purpose_distribution(tessellation_df)
    location_counts = allocate_location_counts(
        config.diaries.location_count_mu,
        config.diaries.location_count_sigma,
        config.diaries.max_locations,
        config.llm.diary_count,
    )
    batches: dict[str, DiaryBatch] = {}
    for day_type in ("weekday", "weekend"):
        batches[day_type] = fetch_diary_batch(
            config.llm,
            city_profile=config.diaries.profile_for(day_type),
            representative_day=config.diaries.representative_day,
            purpose_distribution=distribution,
            location_counts=location_counts,
            location_count_mu=config.diaries.location_count_mu,
            location_count_sigma=config.diaries.location_count_sigma,
            max_locations=config.diaries.max_locations,
            variant=day_type,
            stats=stats,
        )
    return batches, stats, time.perf_counter() - started


def _run_density_epr(
    config: CityBehavExConfig,
    tessellation_df: pd.DataFrame,
    relevance_column: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> tuple[skmob2.TrajDataFrame, Optional[str]]:
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
    traj = skmob2.TrajDataFrame(traj)
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


def _build_schedule(
    config: CityBehavExConfig,
    diary_batches: dict[str, DiaryBatch],
    start_date: pd.Timestamp,
    profiles: Optional[list[AgentProfile]] = None,
) -> tuple[DiaryBank, tuple, np.ndarray, Optional[np.ndarray]]:
    """Build the diary bank and run profile-driven CRP schedule selection.

    Returns (bank, diary_arrays, chosen, profile_embeddings).
    profile_embeddings is None when embeddings are disabled or unavailable.
    """
    bank = build_diary_bank(
        diary_batches,
        config.embedding,
        config.simulation.granularity_minutes,
    )
    typer.echo(
        f"ddCRP schedule bank: {len(bank.diaries)} diaries "
        f"({int((~bank.is_weekend).sum())} weekday / {int(bank.is_weekend.sum())} weekend), "
        f"embeddings={'on' if bank.embedded else 'off (popularity CRP, no profile similarity)'}"
    )

    profile_embeddings = None
    if profiles is not None:
        narratives = [profile_to_narrative(p) for p in profiles]
        profile_embeddings = embed_profiles(narratives, config.embedding)
        if profile_embeddings is not None:
            typer.echo(f"Profile embeddings: {profile_embeddings.shape}")
        else:
            typer.echo("Profile embeddings unavailable — falling back to popularity CRP")

    diary_arrays, chosen = build_ddcrp_diary(
        bank,
        start_date,
        config.simulation.days,
        config.simulation.agents,
        config.simulation.random_state,
        config.schedule,
        profile_embeddings=profile_embeddings,
    )
    return bank, diary_arrays, chosen, profile_embeddings


def _build_activity_data(
    config: CityBehavExConfig,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (act_embs, act_dur_mu, act_dur_sigma, purpose_act_starts, purpose_acts) when enabled."""
    if not config.activities.enabled:
        return None, None, None, None, None
    act_dur_mu, act_dur_sigma = activity_duration_arrays()
    purpose_act_starts, purpose_acts = build_eligibility_csr()
    act_embs = None
    if config.activities.embed_activities:
        descriptions = activity_descriptions()
        act_embs = embed_texts(descriptions, config.embedding)
        if act_embs is not None:
            typer.echo(f"Activity embeddings: {act_embs.shape}")
        else:
            typer.echo("Activity embeddings unavailable — using count-only CRP")
    typer.echo(f"Activities enabled: {len(act_dur_mu)} activities, kappa={config.activities.kappa}, T={config.activities.temperature}")
    return act_embs, act_dur_mu, act_dur_sigma, purpose_act_starts, purpose_acts


def _stamp_path(path: str, ts: str) -> str:
    p = Path(path)
    return str(p.with_name(f"{p.stem}_{ts}{p.suffix}"))


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
    output_path: Optional[str] = None,
) -> tuple[skmob2.TrajDataFrame, Optional[str]]:
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
    act_embs, act_dur_mu, act_dur_sigma, purpose_act_starts, purpose_acts = _build_activity_data(config)

    road_kwargs: dict = {}
    rn = config.road_network
    if rn.enabled and "road_node" in tessellation_df.columns:
        lng_col = "lng" if "lng" in tessellation_df.columns else "lon"
        min_lon = config.simulation.min_lon if config.simulation.min_lon is not None else config.tessellation.min_lon
        min_lat = config.simulation.min_lat if config.simulation.min_lat is not None else config.tessellation.min_lat
        max_lon = config.simulation.max_lon if config.simulation.max_lon is not None else config.tessellation.max_lon
        max_lat = config.simulation.max_lat if config.simulation.max_lat is not None else config.tessellation.max_lat
        if None in [min_lon, min_lat, max_lon, max_lat]:
            min_lon, max_lon = float(tessellation_df[lng_col].min()), float(tessellation_df[lng_col].max())
            min_lat, max_lat = float(tessellation_df["lat"].min()), float(tessellation_df["lat"].max())
        overture_release = rn.overture_release or config.tessellation.overture_release
        nodes_df, edges_df = build_road_graph(
            min_lon, min_lat, max_lon, max_lat, overture_release, rn.nodes_output, rn.edges_output
        )
        typer.echo(
            f"Road routing enabled: {len(nodes_df):,} nodes, {len(edges_df):,} directed edges, "
            f"max {rn.max_leg_waypoints} waypoints/leg"
        )
        road_kwargs = dict(
            road_edge_from=edges_df["from_node"].to_numpy(dtype=np.int64),
            road_edge_to=edges_df["to_node"].to_numpy(dtype=np.int64),
            road_edge_weight_ds=edges_df["weight_ds"].to_numpy(dtype=np.int64),
            road_node_lats=nodes_df["lat"].to_numpy(dtype=np.float64),
            road_node_lngs=nodes_df["lng"].to_numpy(dtype=np.float64),
            location_road_node=tessellation_df["road_node"].to_numpy(dtype=np.int64),
            max_leg_waypoints=rn.max_leg_waypoints,
        )

    profile_types = [p.job for p in profiles] if profiles is not None else None
    df, encounters, moving, activities, social_graph = simulate_agents(
        tessellation_df,
        relevance_column,
        diary_arrays,
        start_ts=int(start_date.timestamp()),
        end_ts=int(end_date.timestamp()),
        slot_seconds=granularity * 60,
        car_speed_kmh=config.simulation.car_speed_kmh,
        n_agents=config.simulation.agents,
        random_state=config.simulation.random_state,
        social_graph_k=config.simulation.social_graph_k,
        profile_graph_exact_threshold=config.simulation.profile_graph_exact_threshold,
        timing=timing,
        starting_locs=home_tiles,
        work_tiles=work_tiles,
        profile_embeddings=profile_embeddings,
        act_embs=act_embs,
        act_dur_mu=act_dur_mu,
        act_dur_sigma=act_dur_sigma,
        purpose_act_starts=purpose_act_starts,
        purpose_acts=purpose_acts,
        act_kappa=config.activities.kappa,
        act_temp=config.activities.temperature,
        return_social_graph=True,
        social_node_profiles=profile_types,
        **road_kwargs,
    )
    base = output_path or config.simulation.output
    social_path = social_network_sidecar_path(base)
    social_graph.write_json(social_path)
    typer.echo(
        f"Saved social network ({social_graph.metadata['node_count']:,} nodes, "
        f"{social_graph.metadata['edge_count']:,} edges) -> {social_path}"
    )
    if len(encounters) > 0:
        enc_path = base.replace(".parquet", "_encounters.parquet")
        encounters.to_parquet(enc_path, index=False)
        typer.echo(f"Saved {len(encounters):,} encounters -> {enc_path}")
    if rn.enabled and len(moving) > 0:
        moving_path = base.replace(".parquet", "_moving.parquet")
        moving.to_parquet(moving_path, index=False)
        typer.echo(f"Saved {len(moving):,} waypoints -> {moving_path}")
    if config.activities.enabled and len(activities) > 0:
        act_path = base.replace(".parquet", "_activities.parquet")
        activities.to_parquet(act_path, index=False)
        typer.echo(f"Saved {len(activities):,} activities -> {act_path}")
    df = _merge_tessellation_metadata(df, tessellation_df, ["tile_id", "category"])
    traj = skmob2.TrajDataFrame(
        df, datetime_col="datetime", lat_col="lat", lng_col="lng", uid_col="uid"
    )
    return traj, "purpose"


def maybe_build_profiles(
    config: CityBehavExConfig,
    tessellation_df: pd.DataFrame,
    relevance_column: str,
) -> Optional[list[AgentProfile]]:
    """Generate or load agent profiles when ``profiles.enabled`` is true."""
    if not config.profiles.enabled:
        return None
    n = config.simulation.agents
    pc = config.profiles
    if pc.profiles_path:
        loaded = load_profiles(pc.profiles_path, n)
        if loaded is not None:
            typer.echo(f"Loaded {len(loaded)} agent profiles from {pc.profiles_path}")
            return loaded
        typer.echo(f"Warning: profiles_path {pc.profiles_path!r} not usable — generating")
    rng = np.random.default_rng(config.simulation.random_state)
    profiles = generate_profiles(n, pc, rng, tessellation_df, relevance_column)
    typer.echo(f"Generated {len(profiles)} agent profiles")
    if pc.output:
        from pathlib import Path
        out = Path(pc.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        profiles_to_frame(profiles).to_parquet(str(out), index=False)
        typer.echo(f"Saved agent profiles -> {pc.output}")
    return profiles


def run_simulation(config: CityBehavExConfig) -> skmob2.TrajDataFrame:
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    stamped_output = _stamp_path(config.simulation.output, ts)
    stamped_html = _stamp_path(config.comparison.html, ts)

    tessellation_df, relevance_column = load_or_build_tessellation(config)
    start_date, end_date = simulation_dates(config)
    profiles = maybe_build_profiles(config, tessellation_df, relevance_column)
    diary_result = maybe_build_diaries(config, tessellation_df)
    core_timing = CoreTiming()

    if diary_result is None:
        traj, synth_activity_col = _run_density_epr(
            config, tessellation_df, relevance_column, start_date, end_date
        )
    else:
        diary_batches, llm_stats, llm_seconds = diary_result
        cache_text = (
            f", {llm_stats.cache_hits:,} cached diary batches"
            if llm_stats.cache_hits
            else ""
        )
        typer.echo(
            f"LLM diary phase: {llm_seconds:.2f}s, {llm_stats.calls:,} chat completion calls"
            f"{cache_text}"
        )
        _, diary_arrays, _, profile_embeddings = _build_schedule(
            config, diary_batches, start_date, profiles=profiles
        )
        traj, synth_activity_col = _run_simulation_core(
            config,
            tessellation_df,
            relevance_column,
            start_date,
            end_date,
            diary_arrays,
            core_timing,
            profiles=profiles,
            profile_embeddings=profile_embeddings,
            output_path=stamped_output,
        )
        typer.echo(f"Rust simulation phase: {core_timing.seconds:.2f}s")

    traj.df.to_parquet(stamped_output, index=False)
    typer.echo(
        f"Saved {len(traj.df):,} records "
        f"({traj.df[traj.uid_col].nunique()} agents) -> {stamped_output}"
    )

    if config.comparison.path:
        from citybehavex.reports import generate_comparison_report

        generate_comparison_report(
            traj=traj,
            real_path=config.comparison.path,
            observed_label=config.comparison.label,
            output_path=stamped_html,
            synth_activity_col=synth_activity_col,
            synthetic_activities_path=stamped_output.replace(
                ".parquet",
                "_activities.parquet",
            ),
        )
    return traj
