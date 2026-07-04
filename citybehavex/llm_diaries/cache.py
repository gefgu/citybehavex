from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from citybehavex.llm.config import LLMConfig

from .models import DiaryBatch, DiaryValidationError, LocationCountDistribution


def cache_path(config: LLMConfig) -> Path:
    cache_dir = Path(config.cache_dir)
    return (
        Path(config.validated_diaries_path)
        if config.validated_diaries_path
        else cache_dir / "validated_diaries.json"
    )


def apply_variant(path: Path, variant: str) -> Path:
    if not variant:
        return path
    return path.with_name(f"{path.stem}_{variant}{path.suffix}")


def load_validated_diary_cache(
    path: Path,
    *,
    expected_distribution: Optional[LocationCountDistribution] = None,
    expected_location_counts: Optional[list[int]] = None,
    expected_motif_exploration_rate: Optional[float] = None,
) -> DiaryBatch:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        batch = DiaryBatch.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise DiaryValidationError(f"invalid diary cache at {path}: {exc}") from exc
    if (
        expected_distribution is not None
        and batch.location_count_distribution != expected_distribution
    ):
        raise DiaryValidationError(
            f"invalid diary cache at {path}: location-count distribution does not match configuration"
        )
    if (
        expected_location_counts is not None
        and batch.target_location_counts != expected_location_counts
    ):
        raise DiaryValidationError(
            f"invalid diary cache at {path}: target location counts do not match configuration"
        )
    if (
        expected_motif_exploration_rate is not None
        and batch.motif_exploration_rate != expected_motif_exploration_rate
    ):
        raise DiaryValidationError(
            f"invalid diary cache at {path}: motif exploration rate does not match configuration"
        )
    return batch


def save_validated_diary_cache(batch: DiaryBatch, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(batch.model_dump_json(indent=2), encoding="utf-8")


def load_cache_with_fallback(
    valid_path: Path,
    base_valid_path: Path,
    *,
    expected_distribution: LocationCountDistribution,
    expected_location_counts: list[int],
    expected_motif_exploration_rate: Optional[float] = None,
) -> DiaryBatch:
    try:
        return load_validated_diary_cache(
            valid_path,
            expected_distribution=expected_distribution,
            expected_location_counts=expected_location_counts,
            expected_motif_exploration_rate=expected_motif_exploration_rate,
        )
    except DiaryValidationError:
        if base_valid_path != valid_path:
            return load_validated_diary_cache(
                base_valid_path,
                expected_distribution=expected_distribution,
                expected_location_counts=expected_location_counts,
                expected_motif_exploration_rate=expected_motif_exploration_rate,
            )
        raise
