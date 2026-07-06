from __future__ import annotations

import requests

from .config import DiariesConfig
from .cache import (
    apply_variant,
    cache_path,
    load_cache_with_fallback,
    load_validated_diary_cache,
    save_validated_diary_cache,
)
from citybehavex.math import allocate_location_counts, lognormal_location_probabilities
from .generator import fetch_diary_batch as _fetch_diary_batch
from .motifs import (
    MOTIF_EXCURSION_PATTERNS,
    MOTIF_LOCATION_COUNTS,
    build_motif_rule,
    sample_motif,
)
from .models import (
    ChatChoice,
    ChatCompletionResponse,
    ChatMessage,
    Diary,
    DiaryBatch,
    DiaryEpisode,
    DiaryValidationError,
    LLMStats,
    LocationCountDistribution,
    Purpose,
    parse_clock_minutes,
)
from .parsing import (
    diary_schema,
    loads_model_json,
    parse_chat_completion_response,
    parse_diary_content,
    parse_diary_response,
    parse_single_diary_content,
    parse_single_diary_response,
)
from .prompts import build_single_diary_prompt, diary_episode_summary

_cache_paths = cache_path
_apply_variant = apply_variant
_load_cache_with_fallback = load_cache_with_fallback


def fetch_diary_batch(*args, **kwargs):
    """Compatibility wrapper that keeps ``citybehavex.llm_diaries.requests`` patchable."""
    kwargs.setdefault("requests_module", requests)
    return _fetch_diary_batch(*args, **kwargs)


def __getattr__(name: str):
    if name == "OpenAICompatibleDiaryClient":
        from citybehavex.llm import OpenAICompatibleDiaryClient

        return OpenAICompatibleDiaryClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ChatChoice",
    "ChatCompletionResponse",
    "ChatMessage",
    "DiariesConfig",
    "Diary",
    "DiaryBatch",
    "DiaryEpisode",
    "DiaryValidationError",
    "LLMStats",
    "LocationCountDistribution",
    "MOTIF_EXCURSION_PATTERNS",
    "MOTIF_LOCATION_COUNTS",
    "OpenAICompatibleDiaryClient",
    "Purpose",
    "allocate_location_counts",
    "build_motif_rule",
    "build_single_diary_prompt",
    "diary_episode_summary",
    "diary_schema",
    "fetch_diary_batch",
    "load_validated_diary_cache",
    "loads_model_json",
    "lognormal_location_probabilities",
    "parse_chat_completion_response",
    "parse_clock_minutes",
    "parse_diary_content",
    "parse_diary_response",
    "parse_single_diary_content",
    "parse_single_diary_response",
    "sample_motif",
    "save_validated_diary_cache",
]
