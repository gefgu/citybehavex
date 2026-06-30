from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import requests
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .config import LLMConfig

Purpose = Literal["HOME", "WORK", "STUDIES", "PURCHASE", "LEISURE", "HEALTH", "OTHER"]


class DiaryValidationError(ValueError):
    """Raised when an LLM response or diary artifact fails validation."""


@dataclass
class LLMStats:
    calls: int = 0


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Optional[str] = None
    content: str


class ChatChoice(BaseModel):
    model_config = ConfigDict(extra="allow")

    message: ChatMessage


class ChatCompletionResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    choices: list[ChatChoice] = Field(min_length=1)


def parse_clock_minutes(value: str) -> int:
    if not isinstance(value, str):
        raise ValueError("time must be a string")
    parts = value.strip().split(":")
    if len(parts) != 2:
        raise ValueError("time must use HH:MM format")
    hour, minute = (int(part) for part in parts)
    if hour == 24 and minute == 0:
        return 24 * 60
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise ValueError("time must be between 00:00 and 24:00")
    return hour * 60 + minute


class DiaryEpisode(BaseModel):
    model_config = ConfigDict(extra="ignore")

    start: str
    end: str
    purpose: Purpose

    @field_validator("start", "end")
    @classmethod
    def validate_clock(cls, value: str) -> str:
        parse_clock_minutes(value)
        return value

    @model_validator(mode="after")
    def validate_end(self) -> DiaryEpisode:
        if self.end_minutes <= self.start_minutes:
            raise ValueError("episode end must be after start")
        return self

    @property
    def start_minutes(self) -> int:
        return parse_clock_minutes(self.start)

    @property
    def end_minutes(self) -> int:
        return parse_clock_minutes(self.end)


class Diary(BaseModel):
    model_config = ConfigDict(extra="ignore")

    diary_id: str
    episodes: list[DiaryEpisode] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_episodes(self) -> Diary:
        if not any(episode.purpose == "HOME" for episode in self.episodes):
            raise ValueError("each diary must contain HOME")

        previous_end = 0
        for index, episode in enumerate(self.episodes):
            if index == 0 and episode.start_minutes != 0:
                raise ValueError("diary must start at 00:00")
            if episode.start_minutes != previous_end:
                raise ValueError("episodes must be ordered, non-overlapping, and cover the day")
            previous_end = episode.end_minutes

        if previous_end != 24 * 60:
            raise ValueError("diary must cover the representative day through 24:00")
        return self


class LocationCountDistribution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mu: float
    sigma: float = Field(gt=0)
    max_locations: int = Field(ge=1, le=6)

    @field_validator("mu", "sigma")
    @classmethod
    def finite_parameters(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("location-count distribution parameters must be finite")
        return value


class DiaryBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    representative_day: Optional[str] = None
    location_count_distribution: LocationCountDistribution
    target_location_counts: list[int] = Field(min_length=10, max_length=30)
    diaries: list[Diary] = Field(min_length=10, max_length=30)

    @model_validator(mode="after")
    def validate_location_count_metadata(self) -> DiaryBatch:
        if len(self.target_location_counts) != len(self.diaries):
            raise ValueError("target_location_counts must have one entry per diary")
        if any(
            count < 1 or count > self.location_count_distribution.max_locations
            for count in self.target_location_counts
        ):
            raise ValueError("target location counts must be within the configured range")
        for count, diary in zip(self.target_location_counts, self.diaries):
            if count == 1 and any(
                episode.purpose != "HOME" for episode in diary.episodes
            ):
                raise ValueError("one-location diaries must contain only HOME episodes")
        return self


def parse_chat_completion_response(payload: Any) -> ChatCompletionResponse:
    try:
        return ChatCompletionResponse.model_validate(payload)
    except ValidationError as exc:
        raise DiaryValidationError(f"invalid OpenAI-compatible response: {exc}") from exc


def parse_diary_content(content: str) -> DiaryBatch:
    try:
        payload = _loads_model_json(content)
    except json.JSONDecodeError as exc:
        raise DiaryValidationError(f"diary content is not valid JSON: {exc}") from exc
    try:
        return DiaryBatch.model_validate(payload)
    except ValidationError as exc:
        raise DiaryValidationError(f"invalid diary payload: {exc}") from exc


def parse_diary_response(payload: Any) -> DiaryBatch:
    response = parse_chat_completion_response(payload)
    return parse_diary_content(response.choices[0].message.content)


def parse_single_diary_content(content: str) -> Diary:
    try:
        payload = _loads_model_json(content)
    except json.JSONDecodeError as exc:
        raise DiaryValidationError(f"diary content is not valid JSON: {exc}") from exc
    if isinstance(payload, dict) and "diary" in payload:
        payload = payload["diary"]
    elif isinstance(payload, dict) and "diaries" in payload:
        diaries = payload["diaries"]
        if not isinstance(diaries, list) or len(diaries) != 1:
            raise DiaryValidationError("single-diary response must contain exactly one diary")
        payload = diaries[0]
    try:
        return Diary.model_validate(payload)
    except ValidationError as exc:
        raise DiaryValidationError(f"invalid diary payload: {exc}") from exc


def parse_single_diary_response(payload: Any) -> Diary:
    response = parse_chat_completion_response(payload)
    return parse_single_diary_content(response.choices[0].message.content)


def diary_schema() -> dict[str, Any]:
    return DiaryBatch.model_json_schema()


def _loads_model_json(content: str) -> Any:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    return json.loads(text)


def diary_episode_summary(diary: Diary) -> str:
    """Compact one-line ``HH:MM-HH:MM PURPOSE | ...`` summary of a diary."""
    return " | ".join(f"{ep.start}-{ep.end} {ep.purpose}" for ep in diary.episodes)


# Cap how many prior schedules are echoed back into a prompt (token budget).
_MAX_PREVIOUS_SCHEDULES = 20


def build_single_diary_prompt(
    *,
    diary_number: int,
    diary_count: int,
    city_profile: str,
    representative_day: str,
    purpose_distribution: Optional[dict[str, float]] = None,
    location_count: Optional[int] = None,
    previous_diaries: Optional[list[Diary]] = None,
) -> str:
    distribution = purpose_distribution or {}
    location_rule = ""
    if location_count == 1:
        location_rule = (
            "This is a one-location schedule: HOME is the only visited location. "
            "Return exactly one episode from 00:00 to 24:00 with purpose HOME. "
            "Do not include any non-HOME purpose.\n"
        )
    elif location_count is not None:
        location_rule = (
            f"This schedule should visit exactly {location_count} distinct "
            f"location(s) counting HOME (so {max(location_count - 1, 0)} non-home "
            "place(s)); returning to a place already visited does not add to that count.\n"
        )

    # One-location schedules are necessarily identical (a single HOME episode), so
    # there is nothing to differentiate; only de-duplicate multi-location ones.
    dedup_rule = ""
    if location_count != 1 and previous_diaries:
        shown = previous_diaries[-_MAX_PREVIOUS_SCHEDULES:]
        listing = "".join(
            f"  {i}. {diary_episode_summary(d)}\n" for i, d in enumerate(shown, start=1)
        )
        dedup_rule = (
            f"The following {len(shown)} schedule(s) with this same location count "
            "have already been generated. Produce a clearly different routine (vary "
            "wake/sleep and activity timing and the non-home activity mix); do NOT "
            "repeat or trivially rephrase any of them:\n"
            f"{listing}"
        )

    return (
        "Return JSON only for one synthetic daily mobility diary.\n"
        "Shape: {\"diary_id\":\"d1\",\"episodes\":[{\"start\":\"00:00\",\"end\":\"07:00\","
        "\"purpose\":\"HOME\"}]}.\n"
        f"Representative day: {representative_day}\n"
        f"Diary number: {diary_number} of {diary_count}\n"
        f"City profile: {city_profile or 'No additional city profile provided.'}\n"
        f"Purpose distribution hints: {json.dumps(distribution, sort_keys=True)}\n"
        f"{location_rule}"
        f"{dedup_rule}"
        "Allowed purposes: HOME, WORK, STUDIES, PURCHASE, LEISURE, HEALTH, OTHER.\n"
        "Rules: start at 00:00, end at 24:00, use contiguous non-overlapping episodes, "
        "include HOME, and use only start/end times in HH:MM.\n"
        "Vary this diary from the others by routine timing and non-home activity mix.\n"
        "No descriptions, notes, markdown, or commentary.\n"
    )


def lognormal_location_probabilities(
    mu: float,
    sigma: float,
    max_locations: int,
) -> dict[int, float]:
    """Return rounded log-normal probabilities truncated to ``1..max_locations``."""
    distribution = LocationCountDistribution(
        mu=mu,
        sigma=sigma,
        max_locations=max_locations,
    )

    def cdf(value: float) -> float:
        if value <= 0:
            return 0.0
        z = (math.log(value) - distribution.mu) / (
            distribution.sigma * math.sqrt(2.0)
        )
        return 0.5 * (1.0 + math.erf(z))

    probabilities = {
        count: cdf(count + 0.5) - cdf(count - 0.5)
        for count in range(1, distribution.max_locations + 1)
    }
    total = sum(probabilities.values())
    if total <= 0:
        raise ValueError("location-count distribution has no probability in range")
    return {count: probability / total for count, probability in probabilities.items()}


def allocate_location_counts(
    mu: float,
    sigma: float,
    max_locations: int,
    n: int,
) -> list[int]:
    """Allocate ``n`` diaries to a truncated rounded log-normal distribution."""
    if n <= 0:
        return []
    probabilities = lognormal_location_probabilities(mu, sigma, max_locations)
    raw = {count: probability * n for count, probability in probabilities.items()}
    counts = {k: int(value) for k, value in raw.items()}
    assigned = sum(counts.values())
    remainder = sorted(
        probabilities, key=lambda k: (raw[k] - counts[k], probabilities[k]), reverse=True
    )
    for count in remainder[: n - assigned]:
        counts[count] += 1
        assigned += 1
    out: list[int] = []
    for k in sorted(counts):
        out.extend([k] * counts[k])
    return out


def _cache_paths(config: LLMConfig) -> Path:
    cache_dir = Path(config.cache_dir)
    valid_path = (
        Path(config.validated_diaries_path)
        if config.validated_diaries_path
        else cache_dir / "validated_diaries.json"
    )
    return valid_path


def _apply_variant(path: Path, variant: str) -> Path:
    if not variant:
        return path
    return path.with_name(f"{path.stem}_{variant}{path.suffix}")


def load_validated_diary_cache(
    path: Path,
    *,
    expected_distribution: Optional[LocationCountDistribution] = None,
    expected_location_counts: Optional[list[int]] = None,
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
    return batch


def save_validated_diary_cache(batch: DiaryBatch, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(batch.model_dump_json(indent=2), encoding="utf-8")


def _load_cache_with_fallback(
    valid_path: Path,
    base_valid_path: Path,
    *,
    expected_distribution: LocationCountDistribution,
    expected_location_counts: list[int],
) -> DiaryBatch:
    try:
        return load_validated_diary_cache(
            valid_path,
            expected_distribution=expected_distribution,
            expected_location_counts=expected_location_counts,
        )
    except DiaryValidationError:
        if base_valid_path != valid_path:
            return load_validated_diary_cache(
                base_valid_path,
                expected_distribution=expected_distribution,
                expected_location_counts=expected_location_counts,
            )
        raise


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
) -> DiaryBatch:
    base_valid_path = _cache_paths(config)
    valid_path = _apply_variant(base_valid_path, variant)

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

    def location_count_for(diary_number: int) -> int:
        return expected_location_counts[diary_number - 1]

    if not all([config.base_url, config.api_key, config.model]):
        return _load_cache_with_fallback(
            valid_path,
            base_valid_path,
            expected_distribution=distribution_metadata,
            expected_location_counts=expected_location_counts,
        )

    # Cache-first: with a chat client configured we would otherwise re-query the
    # LLM on every run. Reuse a config-matching cache when present so the same
    # config reuses its diaries instead of regenerating them.
    if config.reuse_cache:
        try:
            return _load_cache_with_fallback(
                valid_path,
                base_valid_path,
                expected_distribution=distribution_metadata,
                expected_location_counts=expected_location_counts,
            )
        except DiaryValidationError:
            pass  # No usable cache (missing or config changed) -> generate below.

    url = config.base_url.rstrip("/") + "/v1/chat/completions"
    headers = {"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"}

    last_error: Exception | None = None
    try:
        models_response = requests.get(
            config.base_url.rstrip("/") + "/v1/models",
            headers=headers,
            timeout=min(config.timeout_seconds, 10.0),
        )
        models_response.raise_for_status()
    except Exception as exc:  # noqa: BLE001 - converted to cache fallback or domain error.
        last_error = DiaryValidationError(
            f"LLM server preflight failed at {config.base_url.rstrip('/')}/v1/models: {exc}"
        )
        try:
            return _load_cache_with_fallback(
                valid_path,
                base_valid_path,
                expected_distribution=distribution_metadata,
                expected_location_counts=expected_location_counts,
            )
        except DiaryValidationError as cache_error:
            raise DiaryValidationError(
                f"LLM diary generation failed and no valid cache was available: {last_error}"
            ) from cache_error

    diaries: list[Diary] = []
    # Previously generated diaries grouped by their location count, so each new
    # prompt can show prior schedules of the same count and avoid duplicates.
    generated_by_count: dict[int, list[Diary]] = {}
    for diary_number in range(1, config.diary_count + 1):
        diary_location_count = location_count_for(diary_number)
        prompt = build_single_diary_prompt(
            diary_number=diary_number,
            diary_count=config.diary_count,
            city_profile=city_profile,
            representative_day=representative_day,
            purpose_distribution=purpose_distribution,
            location_count=diary_location_count,
            previous_diaries=generated_by_count.get(diary_location_count),
        )
        request_payload = {
            "model": config.model,
            "temperature": config.temperature,
            "messages": [
                {"role": "system", "content": "You generate strictly valid JSON for mobility simulation."},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        if config.max_tokens is not None:
            request_payload["max_tokens"] = config.max_tokens

        diary: Diary | None = None
        for _ in range(max(config.retries, 1)):
            try:
                if stats is not None:
                    stats.calls += 1
                response = requests.post(
                    url,
                    headers=headers,
                    json=request_payload,
                    timeout=config.timeout_seconds,
                )
                response.raise_for_status()
                try:
                    payload = response.json()
                except ValueError as exc:
                    raise DiaryValidationError(
                        f"LLM server returned non-JSON response at {url}: {response.text[:500]}"
                    ) from exc
                diary = parse_single_diary_response(payload)
                if location_count_for(diary_number) == 1 and any(
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
                    "diaries": diaries,
                }
            )
        except ValidationError as exc:
            last_error = DiaryValidationError(f"invalid combined diary batch: {exc}")
        else:
            save_validated_diary_cache(batch, valid_path)
            return batch

    try:
        return _load_cache_with_fallback(
            valid_path,
            base_valid_path,
            expected_distribution=distribution_metadata,
            expected_location_counts=expected_location_counts,
        )
    except DiaryValidationError as cache_error:
        raise DiaryValidationError(
            f"LLM diary generation failed and no valid cache was available: {last_error}"
        ) from cache_error
