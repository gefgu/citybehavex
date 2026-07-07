from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ActivityDurationOverride(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mu_ln: Optional[float] = None
    sigma_ln: Optional[float] = Field(default=None, gt=0)
    scale: Optional[float] = Field(default=None, gt=0)
    sigma_scale: Optional[float] = Field(default=None, gt=0)


class ActivitiesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    kappa: float = Field(default=1.0, gt=0)
    temperature: float = Field(default=0.5, gt=0)
    embed_activities: bool = False
    alignment_backend: Literal["none", "rerank"] = "none"
    alignment_base_url: Optional[str] = None
    alignment_model: Optional[str] = None
    alignment_timeout_seconds: float = Field(default=120.0, gt=0)
    alignment_batch_size: int = Field(default=32, gt=0)
    alignment_cache_path: Optional[str] = None
    alignment_concurrency: int = Field(default=4, ge=1)
    alignment_retries: int = Field(default=2, ge=1)
    alignment_checkpoint_every: int = Field(default=20, ge=1)
    profile_cluster_similarity_threshold: float = Field(default=0.94, ge=-1.0, le=1.0)
    history_weight: float = Field(default=1.0, ge=0.0)
    materialize_travel: bool = True
    poi_type_choice_enabled: bool = False
    poi_type_choice_temperature: float = Field(default=0.5, gt=0)
    poi_type_choice_alpha: float = Field(default=1.0, ge=0.0)
    # When true, run a cheap disposable simulation pass first (no contextual
    # alignment, no road/rail routing) to discover which (cluster, block)
    # pairs are actually reachable, and only score those through the
    # reranker -- see citybehavex.simulation.runner._probe_visited_activity_blocks.
    # Validated on the Greater Paris config (~51% fewer pairs scored, ~2x
    # faster alignment phase, correct end-to-end output), so it's the default
    # now; set false to restore the exact (unpruned) behavior.
    prune_to_reachable: bool = True

    # Uniform tuning knobs over the MTUS catalog's per-activity log-normal
    # duration params (citybehavex.activities.catalog._CATALOG_RAW). `mu_ln`
    # is signed, so `act_dur_scale` is applied as an additive shift in
    # log-space (mu_ln + ln(scale)) rather than a multiplicative one on
    # mu_ln directly -- that's what actually scales every activity's duration
    # by the same factor regardless of its mu_ln's sign.
    act_dur_scale: float = Field(default=1.0, gt=0)
    act_dur_sigma_scale: float = Field(default=1.0, gt=0)
    durations: dict[str, ActivityDurationOverride] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_duration_activity_names(self) -> ActivitiesConfig:
        if not self.durations:
            return self
        from citybehavex.activities.catalog import build_catalog

        known = {activity.name for activity in build_catalog()}
        unknown = sorted(set(self.durations) - known)
        if unknown:
            raise ValueError(
                "activities.durations contains unknown activity name(s): "
                + ", ".join(unknown)
            )
        return self
