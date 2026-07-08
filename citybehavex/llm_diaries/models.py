from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Purpose = Literal["HOME", "WORK", "OTHER"]

_PURPOSE_ALIASES = {
    "HOME": "HOME",
    "HOUSE": "HOME",
    "RESIDENTIAL": "HOME",
    "WORK": "WORK",
    "OFFICE": "WORK",
    "SCHOOL": "OTHER",
    "STUDY": "OTHER",
    "EDUCATION": "OTHER",
    "PURCHASE": "OTHER",
    "SHOPPING": "OTHER",
    "ERRAND": "OTHER",
    "ERRANDS": "OTHER",
    "LEISURE": "OTHER",
    "RECREATION": "OTHER",
    "HEALTHCARE": "OTHER",
    "SOCIAL": "OTHER",
    "OTHER": "OTHER",
}


class DiaryValidationError(ValueError):
    """Raised when an LLM response or diary artifact fails validation."""


@dataclass
class LLMStats:
    calls: int = 0
    cache_hits: int = 0


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

    @field_validator("purpose", mode="before")
    @classmethod
    def normalize_purpose(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        return _PURPOSE_ALIASES.get(value.strip().upper(), value)

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
    max_locations: int = Field(ge=1, le=10)

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
    motif_exploration_rate: float = Field(default=1.0, ge=0.0, le=1.0)
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
            if count == 1 and any(episode.purpose != "HOME" for episode in diary.episodes):
                raise ValueError("one-location diaries must contain only HOME episodes")
        return self
