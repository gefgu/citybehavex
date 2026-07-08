from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import requests
from pydantic import ValidationError

from citybehavex.llm.config import LLMConfig
from citybehavex.llm.server import resolve_llm_server
from citybehavex.math import allocate_location_counts

from .cache import apply_variant, cache_path, load_cache_with_fallback, save_validated_diary_cache
from .models import Diary, DiaryBatch, DiaryValidationError, LLMStats, LocationCountDistribution
from .motifs import MOTIF_EXCURSION_PATTERNS, build_motif_rule, sample_motif
from .parsing import parse_single_diary_response
from .prompts import build_single_diary_prompt


def fetch_diary_batch(
    config: LLMConfig,
    *,
    city_profile: str,
    representative_day: str,
    purpose_distribution: Optional[dict[str, float]] = None,
    location_counts: Optional[list[int]] = None,
    location_count_mu: float = 1.0,
    location_count_sigma: float = 0.5,
    max_locations: int = 6,
    motif_exploration_rate: float = 1.0,
    random_state: int = 0,
    variant: str = "",
    stats: Optional[LLMStats] = None,
    requests_module=requests,
) -> DiaryBatch:
    base_valid_path = cache_path(config)
    valid_path = apply_variant(base_valid_path, variant)

    distribution_metadata = LocationCountDistribution(
        mu=location_count_mu,
        sigma=location_count_sigma,
        max_locations=max_locations,
    )
    expected_location_counts = location_counts or allocate_location_counts(
        location_count_mu,
        location_count_sigma,
        max_locations,
        config.diary_count,
    )
    if len(expected_location_counts) != config.diary_count:
        raise ValueError("location_counts must have one entry per configured diary")
    if any(
        count < 1 or count > distribution_metadata.max_locations
        for count in expected_location_counts
    ):
        raise ValueError("location_counts must be within the configured range")

    def load_cached_batch() -> DiaryBatch:
        batch = load_cache_with_fallback(
            valid_path,
            base_valid_path,
            expected_distribution=distribution_metadata,
            expected_location_counts=expected_location_counts,
            expected_motif_exploration_rate=motif_exploration_rate,
        )
        if stats is not None:
            stats.cache_hits += 1
        return batch

    can_generate = all([config.base_url, config.api_key, config.model]) or (
        config.auto_launch and config.model
    )
    if not can_generate:
        return load_cached_batch()

    if config.reuse_cache:
        try:
            return load_cached_batch()
        except DiaryValidationError:
            pass  # No usable cache (missing or config changed) -> generate below.

    with resolve_llm_server(config, log_dir=config.cache_dir) as effective_url:
        from citybehavex.llm import OpenAICompatibleDiaryClient

        client = OpenAICompatibleDiaryClient(
            config, base_url=effective_url, requests_module=requests_module
        )
        try:
            client.preflight()
        except DiaryValidationError as exc:
            try:
                return load_cached_batch()
            except DiaryValidationError as cache_error:
                raise DiaryValidationError(
                    f"LLM diary generation failed and no valid cache was available: {exc}"
                ) from cache_error

        diaries: list[Diary] = []
        generated_by_count: dict[int, list[Diary]] = {}
        last_error: Exception | None = None

        for diary_number, diary_location_count in enumerate(expected_location_counts, start=1):
            diary_rng = np.random.default_rng(
                np.random.SeedSequence([int(random_state), diary_number])
            )
            motif_rule = ""
            if diary_location_count > 1 and diary_rng.random() >= motif_exploration_rate:
                motif_ordinal = sample_motif(diary_location_count, diary_rng)
                if motif_ordinal is not None:
                    motif_rule = build_motif_rule(MOTIF_EXCURSION_PATTERNS[motif_ordinal])

            prompt = build_single_diary_prompt(
                diary_number=diary_number,
                diary_count=config.diary_count,
                city_profile=city_profile,
                representative_day=representative_day,
                purpose_distribution=purpose_distribution,
                location_count=diary_location_count,
                previous_diaries=generated_by_count.get(diary_location_count),
                motif_rule=motif_rule,
            )

            diary: Diary | None = None
            for _ in range(max(config.retries, 1)):
                try:
                    payload = client.generate_json(prompt, stats=stats)
                    diary = parse_single_diary_response(payload)
                    if diary_location_count == 1 and any(
                        episode.purpose != "HOME" for episode in diary.episodes
                    ):
                        raise DiaryValidationError(
                            "one-location diary must contain only HOME episodes"
                        )
                    diary.diary_id = f"routine-{diary_number:03d}"
                    break
                except Exception as exc:  # noqa: BLE001 - converted to cache fallback or domain error.
                    last_error = exc
            if diary is None:
                break
            diaries.append(diary)
            generated_by_count.setdefault(diary_location_count, []).append(diary)

        if len(diaries) == config.diary_count:
            try:
                batch = DiaryBatch.model_validate(
                    {
                        "representative_day": representative_day,
                        "location_count_distribution": distribution_metadata.model_dump(),
                        "target_location_counts": expected_location_counts,
                        "motif_exploration_rate": motif_exploration_rate,
                        "diaries": diaries,
                    }
                )
            except ValidationError as exc:
                last_error = DiaryValidationError(f"invalid combined diary batch: {exc}")
            else:
                save_validated_diary_cache(batch, valid_path)
                return batch

        try:
            return load_cached_batch()
        except DiaryValidationError as cache_error:
            raise DiaryValidationError(
                f"LLM diary generation failed and no valid cache was available: {last_error}"
            ) from cache_error
