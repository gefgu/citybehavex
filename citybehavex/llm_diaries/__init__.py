from __future__ import annotations

import requests

from .cache import (
    apply_variant,
    cache_path,
    load_cache_with_fallback,
    load_validated_diary_cache,
    save_validated_diary_cache,
)
from .client import OpenAICompatibleDiaryClient
from .distribution import allocate_location_counts, lognormal_location_probabilities
from .generator import fetch_diary_batch as _fetch_diary_batch
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


__all__ = [
    "ChatChoice",
    "ChatCompletionResponse",
    "ChatMessage",
    "Diary",
    "DiaryBatch",
    "DiaryEpisode",
    "DiaryValidationError",
    "LLMStats",
    "LocationCountDistribution",
    "OpenAICompatibleDiaryClient",
    "Purpose",
    "allocate_location_counts",
    "build_single_diary_prompt",
    "diary_episode_summary",
    "diary_schema",
    "fetch_diary_batch",
    "load_validated_diary_cache",
    "lognormal_location_probabilities",
    "parse_chat_completion_response",
    "parse_clock_minutes",
    "parse_diary_content",
    "parse_diary_response",
    "parse_single_diary_content",
    "parse_single_diary_response",
    "save_validated_diary_cache",
]
