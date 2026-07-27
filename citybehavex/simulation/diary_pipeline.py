from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import typer

from citybehavex.config import CityBehavExConfig
from citybehavex.llm_diaries import DiaryBatch, LLMStats, allocate_location_counts, fetch_diary_batch
from citybehavex.tessellation import purpose_distribution
from citybehavex.utils import ProgressReporter

_PROGRESS_INTERVAL_SECONDS = 5.0

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
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> Optional[tuple[dict[str, DiaryBatch], LLMStats, float]]:
    """Fetch one diary batch per day type needed for [start_date, end_date)
    (weekday/weekend plus any overlapping special days), or None if no LLM
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
        max_one_location=config.diaries.max_one_location_diaries,
    )
    day_types = config.diaries.day_types_for_range(
        start_date.date(), (end_date - timedelta(days=1)).date()
    )
    batches: dict[str, DiaryBatch] = {}
    total_diaries = len(day_types) * config.llm.diary_count
    completed_by_day_type = {day_type: 0 for day_type in day_types}
    diary_progress = ProgressReporter(
        "LLM diary generation",
        total_diaries,
        "diaries",
        rate_precision=2,
        min_interval_seconds=_PROGRESS_INTERVAL_SECONDS,
        emit=typer.echo,
        started=started,
    )

    def report_diary_generation_progress(
        day_type: str,
        completed: int,
        total: int,
        current_stats: LLMStats,
    ) -> None:
        completed_by_day_type[day_type] = completed
        done = sum(completed_by_day_type.values())
        diary_progress.report(
            done,
            detail=f"({day_type} {completed}/{total}), {current_stats.calls:,} chat calls",
        )

    for day_type in day_types:
        batches[day_type] = fetch_diary_batch(
            config.llm,
            city_profile=config.diaries.profile_for(day_type),
            representative_day=config.diaries.representative_day,
            purpose_distribution=distribution,
            location_counts=location_counts,
            location_count_mu=config.diaries.location_count_mu,
            location_count_sigma=config.diaries.location_count_sigma,
            max_locations=config.diaries.max_locations,
            motif_exploration_rate=config.diaries.motif_exploration_rate,
            random_state=config.simulation.random_state,
            variant=day_type,
            stats=stats,
            progress_callback=report_diary_generation_progress,
        )
    return batches, stats, time.perf_counter() - started
