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

    # SW-CRP temperature T and exploration weight alpha, sampled per-agent
    # (see citybehavex/schedules/crp.py). Each is drawn from either a Beta or
    # a LogNormal distribution; the *_beta_a/b fields apply to the "beta"
    # choice, the *_mu_ln/sigma_ln fields to the "lognormal" one.
    temperature_distribution: Literal["beta", "lognormal"] = "lognormal"
    temperature_beta_a: float = Field(default=2.0, gt=0)
    temperature_beta_b: float = Field(default=5.0, gt=0)
    # Target median T ~= 0.3: mu_ln = ln(0.3) - sigma_ln**2 / 2.
    temperature_mu_ln: float = Field(default=-1.329)
    temperature_sigma_ln: float = Field(default=0.5, gt=0)

    alpha_distribution: Literal["beta", "lognormal"] = "lognormal"
    alpha_beta_a: float = Field(default=2.0, gt=0)
    alpha_beta_b: float = Field(default=5.0, gt=0)
    # Target median alpha ~= 1.0: mu_ln = ln(1.0) - sigma_ln**2 / 2.
    alpha_mu_ln: float = Field(default=-0.125)
    alpha_sigma_ln: float = Field(default=0.5, gt=0)
