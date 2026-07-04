from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ActivitiesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    kappa: float = Field(default=1.0, gt=0)
    temperature: float = Field(default=0.5, gt=0)
    embed_activities: bool = False

    # Uniform tuning knobs over the MTUS catalog's per-activity log-normal
    # duration params (citybehavex.activities.catalog._CATALOG_RAW). `mu_ln`
    # is signed, so `act_dur_scale` is applied as an additive shift in
    # log-space (mu_ln + ln(scale)) rather than a multiplicative one on
    # mu_ln directly -- that's what actually scales every activity's duration
    # by the same factor regardless of its mu_ln's sign.
    act_dur_scale: float = Field(default=1.0, gt=0)
    act_dur_sigma_scale: float = Field(default=1.0, gt=0)
