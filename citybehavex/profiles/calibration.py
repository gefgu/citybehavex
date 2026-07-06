"""LLM-based calibration of demographic distribution weights for a city.

When ``AgentProfilesConfig.llm_override`` is enabled, the configured LLM is asked
to propose education/health/household/job category weights from a free-text city
profile (the same description used for LLM diary generation), instead of relying
on the fixed defaults in ``AgentProfilesConfig``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import requests

from citybehavex.llm.config import LLMConfig
from citybehavex.llm.server import resolve_llm_server
from citybehavex.llm_diaries import (
    DiaryValidationError,
    LLMStats,
    loads_model_json,
    parse_chat_completion_response,
)

from .agents import EDUCATION_LEVELS, HEALTH_LEVELS, HOUSEHOLD_TYPES, ILOSTAT_JOBS

# Maps each `AgentProfilesConfig` weight field to its ordered category labels.
WEIGHT_GROUPS: dict[str, list[str]] = {
    "education_weights": EDUCATION_LEVELS,
    "health_weights": [str(level) for level in HEALTH_LEVELS],
    "household_weights": HOUSEHOLD_TYPES,
    "job_weights": ILOSTAT_JOBS,
}


def _build_prompt(city_profile: str) -> str:
    groups = "\n".join(
        f'- "{field}": {len(labels)} values, one per category in this exact order: {labels}'
        for field, labels in WEIGHT_GROUPS.items()
    )
    return (
        "Return JSON only with calibrated demographic distribution weights for the "
        "population of simulated mobility agents in the described city.\n"
        f"City profile: {city_profile or 'No additional city profile provided.'}\n"
        "For each group below, return an array of non-negative weights matching the "
        "category order exactly. Weights need not sum to 1; they will be renormalized.\n"
        f"{groups}\n"
        'Shape: {"education_weights": [...], "health_weights": [...], '
        '"household_weights": [...], "job_weights": [...]}\n'
        "Base the weights on the city's real socioeconomic profile: education "
        "attainment, health outcomes, household composition, and occupational "
        "structure. No commentary, markdown, or extra keys.\n"
    )


def _normalized_weights(payload: object, field: str, labels: list[str]) -> list[float]:
    if not isinstance(payload, dict) or field not in payload:
        raise DiaryValidationError(f"calibration response is missing {field!r}")
    values = payload[field]
    if not isinstance(values, list) or len(values) != len(labels):
        raise DiaryValidationError(f"{field} must be a list of {len(labels)} numbers")
    try:
        weights = [float(v) for v in values]
    except (TypeError, ValueError) as exc:
        raise DiaryValidationError(f"{field} must contain only numbers") from exc
    if any(w < 0 for w in weights):
        raise DiaryValidationError(f"{field} values must be non-negative")
    total = sum(weights)
    if total <= 0:
        raise DiaryValidationError(f"{field} values must sum to a positive number")
    return [w / total for w in weights]


def calibrate_demographic_weights(
    config: LLMConfig,
    *,
    city_profile: str,
    stats: Optional[LLMStats] = None,
    requests_module=requests,
) -> Optional[dict[str, list[float]]]:
    """Ask the configured LLM for calibrated demographic weights.

    Returns ``None`` (defaults should be left untouched) when no LLM server is
    configured. Raises ``DiaryValidationError`` if a server is configured but its
    response is unusable, since falling back silently would hide a broken config.
    """
    can_generate = all([config.base_url, config.api_key, config.model]) or (
        config.auto_launch and config.model
    )
    if not can_generate:
        return None

    log_dir = Path(config.cache_dir) / "profile_calibration"
    prompt = _build_prompt(city_profile)
    with resolve_llm_server(config, log_dir=log_dir) as effective_url:
        from citybehavex.llm import OpenAICompatibleDiaryClient

        client = OpenAICompatibleDiaryClient(
            config, base_url=effective_url, requests_module=requests_module
        )
        client.preflight()

        last_error: Exception | None = None
        for _ in range(max(config.retries, 1)):
            try:
                raw_payload = client.generate_json(prompt, stats=stats)
                response = parse_chat_completion_response(raw_payload)
                content = loads_model_json(response.choices[0].message.content)
                return {
                    field: _normalized_weights(content, field, labels)
                    for field, labels in WEIGHT_GROUPS.items()
                }
            except Exception as exc:  # noqa: BLE001 - retried below, raised if exhausted
                last_error = exc

    raise DiaryValidationError(
        f"LLM demographic weight calibration failed after {max(config.retries, 1)} attempt(s): "
        f"{last_error}"
    )
