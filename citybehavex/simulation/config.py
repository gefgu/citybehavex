from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
    social_graph_k: int = 20
    profile_graph_exact_threshold: int = 10_000

    # When true, road-path waypoints are flushed to the `_moving.parquet`
    # sidecar incrementally, once per simulated day, instead of being held in
    # memory (and returned from the Rust core) for the entire run. This is
    # the single largest driver of RSS growth on long/large-agent-count runs
    # (up to `max_leg_waypoints` rows per stop, vs. one row per stop for the
    # main trajectory table). Default off to keep today's single-shot output
    # byte-for-byte unchanged.
    stream_output: bool = False

    # EPR (Exploration and Preferential Return) mobility model. `rho`/`gamma`
    # set the per-step explore-vs-return probability (p_explore = rho * S^-gamma,
    # S = agent's current number of distinct visited locations); `alpha` mixes
    # in social (profile-similar neighbor) location choice vs. individual EPR.
    rho: float = Field(default=0.6, gt=0)
    gamma: float = Field(default=0.21, gt=0)
    alpha: float = Field(default=0.2, ge=0.0, le=1.0)
    dt_update_mob_sim_hours: float = Field(default=24 * 7, gt=0)
    indipendency_window_hours: float = Field(default=0.5, gt=0)

    # Gravity/OD model for destination choice among candidate locations:
    # T_ij ~ O_i^origin_exponent * D_j^destination_exponent * distance_km^deterrence_exponent.
    gravity_deterrence_exponent: float = -2.0
    gravity_origin_exponent: float = 1.0
    gravity_destination_exponent: float = 1.0

    @field_validator("agents", "days", "social_graph_k", "profile_graph_exact_threshold")
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
