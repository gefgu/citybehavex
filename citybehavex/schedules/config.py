from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class ScheduleConfig(BaseModel):
    """Profile-driven CRP schedule selection."""

    model_config = ConfigDict(extra="forbid")

    similarity_backend: Literal["embedding", "alignment_model"] = "embedding"
    alignment_base_url: Optional[str] = None
    alignment_model: Optional[str] = None
    alignment_timeout_seconds: float = Field(default=120.0, gt=0)
    alignment_batch_size: int = Field(default=32, gt=0)
    alignment_cache_path: Optional[str] = None
    alignment_concurrency: int = Field(default=4, ge=1)
    alignment_retries: int = Field(default=2, ge=1)
    alignment_checkpoint_every: int = Field(default=5, ge=1)
    temperature_beta_a: float = Field(default=2.0, gt=0)
    temperature_beta_b: float = Field(default=5.0, gt=0)
    alpha_beta_a: float = Field(default=2.0, gt=0)
    alpha_beta_b: float = Field(default=5.0, gt=0)
