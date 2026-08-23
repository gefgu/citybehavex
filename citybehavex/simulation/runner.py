from __future__ import annotations

from datetime import datetime
from pathlib import Path

import fastmob
import typer

from citybehavex.config import CityBehavExConfig
from citybehavex.simulation.core import CoreTiming
from citybehavex.simulation.core_pipeline import _run_density_epr, _run_simulation_core
from citybehavex.simulation.diary_pipeline import maybe_build_diaries, simulation_dates
from citybehavex.simulation.output import _stamp_path
from citybehavex.simulation.profile_pipeline import maybe_build_profiles
from citybehavex.simulation.schedule_pipeline import _build_schedule, _save_crp_artifact
from citybehavex.simulation.tessellation_pipeline import load_or_build_tessellation


def run_simulation(config: CityBehavExConfig) -> fastmob.TrajDataFrame:
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    stamped_output = _stamp_path(config.simulation.output, ts)
    # All sidecar artifacts for this run are written next to stamped_output.
    Path(stamped_output).parent.mkdir(parents=True, exist_ok=True)

    tessellation_df, relevance_column, home_tile_pool = load_or_build_tessellation(config)
    start_date, end_date = simulation_dates(config)
    profiles = maybe_build_profiles(config, tessellation_df, relevance_column, home_tile_pool)
    diary_result = maybe_build_diaries(config, tessellation_df, start_date, end_date)
    core_timing = CoreTiming()

    if diary_result is None:
        traj, _synth_activity_col = _run_density_epr(
            config, tessellation_df, relevance_column, start_date, end_date
        )
        already_written = False
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
        bank, diary_arrays, chosen, profile_embeddings, crp_info, profile_clusters = _build_schedule(
            config, diary_batches, start_date, profiles=profiles
        )
        crp_path = stamped_output.replace(".parquet", "_crp.parquet")
        _save_crp_artifact(crp_path, bank, chosen, crp_info)
        typer.echo(
            f"Saved SW-CRP diary selection state "
            f"({config.simulation.agents} agents x {len(bank.diaries)} diaries) -> {crp_path}"
        )
        traj, _synth_activity_col, already_written = _run_simulation_core(
            config,
            tessellation_df,
            relevance_column,
            start_date,
            end_date,
            diary_arrays,
            core_timing,
            profiles=profiles,
            profile_embeddings=profile_embeddings,
            bank=bank,
            profile_clusters=profile_clusters,
            output_path=stamped_output,
        )
        typer.echo(f"Rust simulation phase: {core_timing.seconds:.2f}s")

    if not already_written:
        traj.df.to_parquet(stamped_output, index=False)
        typer.echo(
            f"Saved {len(traj.df):,} records "
            f"({traj.df[traj.uid_col].nunique()} agents) -> {stamped_output}"
        )

    if config.comparison.path:
        typer.echo("Comparison data configured; view this run in the CityBehavEx web UI.")
    return traj
