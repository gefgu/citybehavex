from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class SimulationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tessellation: Optional[str] = None
    min_lon: Optional[float] = None
    min_lat: Optional[float] = None
    max_lon: Optional[float] = None
    max_lat: Optional[float] = None
    agents: int = 500
    days: int = 7
    start_date: Optional[str] = None
    output: str = "trajectories.parquet"
    random_state: int = 42
    relevance_column: str = "total_poi_count"
    granularity_minutes: int = 15
    car_speed_kmh: float = 50.0

    @field_validator("agents", "days")
    @classmethod
    def positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be positive")
        return value

    @field_validator("granularity_minutes")
    @classmethod
    def valid_granularity(cls, value: int) -> int:
        if value <= 0 or 1440 % value != 0:
            raise ValueError("granularity_minutes must be a positive divisor of 1440")
        return value

    @field_validator("car_speed_kmh")
    @classmethod
    def positive_speed(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("car_speed_kmh must be positive")
        return value

    @model_validator(mode="after")
    def validate_source(self) -> SimulationConfig:
        bbox_values = [self.min_lon, self.min_lat, self.max_lon, self.max_lat]
        has_any_bbox = any(v is not None for v in bbox_values)
        has_full_bbox = all(v is not None for v in bbox_values)
        if self.tessellation and has_any_bbox:
            raise ValueError("provide either tessellation or bbox, not both")
        if has_any_bbox and not has_full_bbox:
            raise ValueError("bbox requires min_lon, min_lat, max_lon, and max_lat")
        return self
