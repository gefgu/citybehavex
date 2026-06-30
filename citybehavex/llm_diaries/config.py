from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class DiariesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    city_profile: str = ""
    city_profile_weekday: str = ""
    city_profile_weekend: str = ""
    representative_day: str = "2026-01-01"
    allowed_purposes: list[str] = Field(
        default_factory=lambda: [
            "HOME",
            "WORK",
            "STUDIES",
            "PURCHASE",
            "LEISURE",
            "HEALTH",
            "OTHER",
        ]
    )
    location_count_mu: float = 1.0
    location_count_sigma: float = Field(default=0.5, gt=0)
    max_locations: int = Field(default=6, ge=1, le=6)

    def profile_for(self, day_type: str) -> str:
        """City profile for weekday/weekend, falling back to the shared one."""
        specific = self.city_profile_weekday if day_type == "weekday" else self.city_profile_weekend
        return specific or self.city_profile
