from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ActivitiesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    kappa: float = Field(default=1.0, gt=0)
    temperature: float = Field(default=0.5, gt=0)
    embed_activities: bool = False
