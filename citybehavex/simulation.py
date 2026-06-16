from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import skmob2
import typer
from skmob2.models import DensityEPR, MarkovDiaryGenerator

from .config import CityBehavExConfig
from .diaries import annotate_trajectory_purposes, diary_batch_to_markov_training
from .llm_diaries import DiaryBatch, LLMStats, allocate_location_counts, fetch_diary_batch
from .tessellation import build_poi_tessellation, build_tessellation, purpose_distribution
from .trip_ditras import build_daily_diary, simulate_trip_ditras
from .trip_sts_epr import RustTiming, simulate_trip_sts_epr


def load_or_build_tessellation(config: CityBehavExConfig) -> tuple[pd.DataFrame, str]:
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
        extra_cols = [c for c in ["tile_id", "purpose"] if c in tessellation_df.columns]
        lookup = tessellation_df[["lat", "lng"] + extra_cols].drop_duplicates(["lat", "lng"])
        traj.df = traj.df.merge(lookup, on=["lat", "lng"], how="left")
        synth_activity_col = "purpose"
    return traj, synth_activity_col


def _run_trip_ditras(
    config: CityBehavExConfig,
    tessellation_df: pd.DataFrame,
    relevance_column: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    diary_batches: dict[str, DiaryBatch],
    timing: RustTiming,
) -> tuple[skmob2.TrajDataFrame, Optional[str]]:
    granularity = config.simulation.granularity_minutes
    generators: dict[str, MarkovDiaryGenerator] = {}
    for day_type, batch in diary_batches.items():
        training = diary_batch_to_markov_training(
            batch,
            representative_day=config.diaries.representative_day,
            granularity_minutes=granularity,
        )
        generator = MarkovDiaryGenerator(granularity_minutes=granularity)
        generator.fit(training, len(batch.diaries), lid="location")
        generators[day_type] = generator

    typer.echo(
        f"Running trip-DITRAS: {config.simulation.agents} agents x {config.simulation.days} days "
        f"@ {granularity}-min slots, {config.simulation.car_speed_kmh:.0f} km/h car "
        f"({start_date.date()} -> {end_date.date()})"
    )
    diary_arrays = build_daily_diary(
        generators,
        start_date,
        config.simulation.days,
        config.simulation.agents,
        config.simulation.random_state,
    )
    df = simulate_trip_ditras(
        tessellation_df,
        relevance_column,
        diary_arrays,
        start_ts=int(start_date.timestamp()),
        end_ts=int(end_date.timestamp()),
        slot_seconds=granularity * 60,
        car_speed_kmh=config.simulation.car_speed_kmh,
        n_agents=config.simulation.agents,
        random_state=config.simulation.random_state,
        timing=timing,
    )
    df = annotate_trajectory_purposes(
        df,
        diary_batches["weekday"],
        weekend_batch=diary_batches.get("weekend"),
    )
    traj = skmob2.TrajDataFrame(
        df, datetime_col="datetime", lat_col="lat", lng_col="lng", uid_col="uid"
    )
    return traj, "purpose"


def _run_trip_sts_epr(
    config: CityBehavExConfig,
    tessellation_df: pd.DataFrame,
    relevance_column: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    diary_batches: dict[str, DiaryBatch],
    timing: RustTiming,
) -> tuple[skmob2.TrajDataFrame, Optional[str]]:
    granularity = config.simulation.granularity_minutes
    generators: dict[str, MarkovDiaryGenerator] = {}
    for day_type, batch in diary_batches.items():
        training = diary_batch_to_markov_training(
            batch,
            representative_day=config.diaries.representative_day,
            granularity_minutes=granularity,
        )
        generator = MarkovDiaryGenerator(granularity_minutes=granularity)
        generator.fit(training, len(batch.diaries), lid="location")
        generators[day_type] = generator

    typer.echo(
        f"Running trip-STS-EPR: {config.simulation.agents} agents x {config.simulation.days} days "
        f"@ {granularity}-min slots, {config.simulation.car_speed_kmh:.0f} km/h car "
        f"({start_date.date()} -> {end_date.date()})"
    )
    diary_arrays = build_daily_diary(
        generators,
        start_date,
        config.simulation.days,
        config.simulation.agents,
        config.simulation.random_state,
    )
    df = simulate_trip_sts_epr(
        tessellation_df,
        relevance_column,
        diary_arrays,
        start_ts=int(start_date.timestamp()),
        end_ts=int(end_date.timestamp()),
        slot_seconds=granularity * 60,
        car_speed_kmh=config.simulation.car_speed_kmh,
        n_agents=config.simulation.agents,
        random_state=config.simulation.random_state,
        timing=timing,
    )
    df = annotate_trajectory_purposes(
        df,
        diary_batches["weekday"],
        weekend_batch=diary_batches.get("weekend"),
    )
    traj = skmob2.TrajDataFrame(
        df, datetime_col="datetime", lat_col="lat", lng_col="lng", uid_col="uid"
    )
    return traj, "purpose"


def run_simulation(config: CityBehavExConfig) -> skmob2.TrajDataFrame:
    tessellation_df, relevance_column = load_or_build_tessellation(config)
    start_date, end_date = simulation_dates(config)
    diary_result = maybe_build_diaries(config, tessellation_df)
    rust_timing = RustTiming()

    if diary_result is None:
        traj, synth_activity_col = _run_density_epr(
            config, tessellation_df, relevance_column, start_date, end_date
        )
    else:
        diary_batches, llm_stats, llm_seconds = diary_result
        typer.echo(
            f"LLM diary phase: {llm_seconds:.2f}s, {llm_stats.calls:,} chat completion calls"
        )
        if config.simulation.model == "ditras":
            traj, synth_activity_col = _run_trip_ditras(
                config,
                tessellation_df,
                relevance_column,
                start_date,
                end_date,
                diary_batches,
                rust_timing,
            )
        else:
            traj, synth_activity_col = _run_trip_sts_epr(
                config,
                tessellation_df,
                relevance_column,
                start_date,
                end_date,
                diary_batches,
                rust_timing,
            )
        typer.echo(f"Rust simulation phase: {rust_timing.seconds:.2f}s")

    traj.df.to_parquet(config.simulation.output, index=False)
    typer.echo(
        f"Saved {len(traj.df):,} records "
        f"({traj.df[traj.uid_col].nunique()} agents) -> {config.simulation.output}"
    )

    if config.comparison.path:
        from .reports import generate_comparison_report

        generate_comparison_report(
            traj=traj,
            real_path=config.comparison.path,
            observed_label=config.comparison.label,
            output_path=config.comparison.html,
            synth_activity_col=synth_activity_col,
        )
    return traj
