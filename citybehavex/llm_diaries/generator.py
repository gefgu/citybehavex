from __future__ import annotations

from typing import Optional

import requests
from pydantic import ValidationError

from citybehavex.config import LLMConfig

from .cache import apply_variant, cache_path, load_cache_with_fallback, save_validated_diary_cache
from .client import OpenAICompatibleDiaryClient
from .distribution import allocate_location_counts
from .models import Diary, DiaryBatch, DiaryValidationError, LLMStats, LocationCountDistribution
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

    if not all([config.base_url, config.api_key, config.model]):
        return load_cache_with_fallback(
            valid_path,
            base_valid_path,
            expected_distribution=distribution_metadata,
            expected_location_counts=expected_location_counts,
        )

    if config.reuse_cache:
        try:
            return load_cache_with_fallback(
                valid_path,
                base_valid_path,
                expected_distribution=distribution_metadata,
                expected_location_counts=expected_location_counts,
            )
        except DiaryValidationError:
            pass  # No usable cache (missing or config changed) -> generate below.

    client = OpenAICompatibleDiaryClient(config, requests_module=requests_module)
    try:
        client.preflight()
    except DiaryValidationError as exc:
        try:
            return load_cache_with_fallback(
                valid_path,
                base_valid_path,
                expected_distribution=distribution_metadata,
                expected_location_counts=expected_location_counts,
            )
        except DiaryValidationError as cache_error:
            raise DiaryValidationError(
                f"LLM diary generation failed and no valid cache was available: {exc}"
            ) from cache_error

    diaries: list[Diary] = []
    generated_by_count: dict[int, list[Diary]] = {}
    last_error: Exception | None = None

    for diary_number, diary_location_count in enumerate(expected_location_counts, start=1):
        prompt = build_single_diary_prompt(
            diary_number=diary_number,
            diary_count=config.diary_count,
            city_profile=city_profile,
            representative_day=representative_day,
            purpose_distribution=purpose_distribution,
            location_count=diary_location_count,
            previous_diaries=generated_by_count.get(diary_location_count),
        )

        diary: Diary | None = None
        for _ in range(max(config.retries, 1)):
            try:
                payload = client.generate_json(prompt, stats=stats)
                diary = parse_single_diary_response(payload)
                if diary_location_count == 1 and any(
                    episode.purpose != "HOME" for episode in diary.episodes
                ):
                    raise DiaryValidationError("one-location diary must contain only HOME episodes")
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
                    "diaries": diaries,
                }
            )
        except ValidationError as exc:
            last_error = DiaryValidationError(f"invalid combined diary batch: {exc}")
        else:
            save_validated_diary_cache(batch, valid_path)
            return batch

    try:
        return load_cache_with_fallback(
            valid_path,
            base_valid_path,
            expected_distribution=distribution_metadata,
            expected_location_counts=expected_location_counts,
        )
    except DiaryValidationError as cache_error:
        raise DiaryValidationError(
            f"LLM diary generation failed and no valid cache was available: {last_error}"
        ) from cache_error
