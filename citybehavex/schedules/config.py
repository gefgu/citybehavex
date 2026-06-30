from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ScheduleConfig(BaseModel):
    """Profile-driven CRP schedule selection."""

    model_config = ConfigDict(extra="forbid")

    temperature_beta_a: float = Field(default=2.0, gt=0)
    temperature_beta_b: float = Field(default=5.0, gt=0)
    alpha_beta_a: float = Field(default=2.0, gt=0)
    alpha_beta_b: float = Field(default=5.0, gt=0)
