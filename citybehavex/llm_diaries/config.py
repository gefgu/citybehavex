from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class SpecialDayConfig(BaseModel):
    """An explicit date range that overrides the weekday/weekend calendar rule.

    Used for scenarios like a disaster/emergency period that needs its own
    city profile and diary pool, regardless of which day of the week it falls
    on.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    start_date: str
    end_date: str
    city_profile: str = ""

    def _start(self) -> date:
        return date.fromisoformat(self.start_date)

    def _end(self) -> date:
        return date.fromisoformat(self.end_date)

    def contains(self, day: date) -> bool:
        return self._start() <= day <= self._end()

    def overlaps(self, start: date, end: date) -> bool:
        return self._start() <= end and self._end() >= start


class DiariesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    city_profile: str = ""
    city_profile_weekday: str = ""
    city_profile_weekend: str = ""
    representative_day: str = "2026-01-01"
    special_days: list[SpecialDayConfig] = Field(default_factory=list)
    allowed_purposes: list[str] = Field(
        default_factory=lambda: [
            "HOME",
            "WORK",
            "OTHER",
        ]
    )
    location_count_mu: float = 1.0
    location_count_sigma: float = Field(default=0.5, gt=0)
    max_locations: int = Field(default=6, ge=1, le=6)
    max_one_location_diaries: int | None = Field(default=None, ge=0)
    motif_exploration_rate: float = Field(default=1.0, ge=0.0, le=1.0)

    def profile_for(self, day_type: str) -> str:
        """City profile for a day type, falling back to the shared one."""
        for special_day in self.special_days:
            if special_day.name == day_type:
                return special_day.city_profile or self.city_profile
        specific = self.city_profile_weekday if day_type == "weekday" else self.city_profile_weekend
        return specific or self.city_profile

    def day_types_for_range(self, start: date, end: date) -> list[str]:
        """Day types needed to cover a date range: weekday/weekend plus any
        special days whose range overlaps ``[start, end]``."""
        day_types = ["weekday", "weekend"]
        day_types.extend(
            special_day.name for special_day in self.special_days if special_day.overlaps(start, end)
        )
        return day_types

    def resolve_day_type(self, day: date) -> str:
        """Day type for a single calendar date: a matching special day's name
        if ``day`` falls in its range, else the weekday/weekend calendar rule."""
        for special_day in self.special_days:
            if special_day.contains(day):
                return special_day.name
        return "weekend" if day.weekday() >= 5 else "weekday"
