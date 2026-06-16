from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TessellationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Optional[str] = None
    min_lon: Optional[float] = None
    min_lat: Optional[float] = None
    max_lon: Optional[float] = None
    max_lat: Optional[float] = None
    resolution: int = 10
    enrich_overture: bool = False
    overture_release: str = "2026-05-20.0"
    min_poi_count: int = 1
    poi_tessellation: bool = False
    output: str = "tessellation.parquet"
    relevance_column: str = "total_poi_count"

    @model_validator(mode="after")
    def validate_source(self) -> TessellationConfig:
        bbox_values = [self.min_lon, self.min_lat, self.max_lon, self.max_lat]
        has_any_bbox = any(v is not None for v in bbox_values)
        has_full_bbox = all(v is not None for v in bbox_values)
        if has_any_bbox and not has_full_bbox:
            raise ValueError("bbox requires min_lon, min_lat, max_lon, and max_lat")
        return self


class SimulationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tessellation: Optional[str] = None
    model: Literal["sts_epr", "ditras"] = "sts_epr"
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


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None
    temperature: float = 0.4
    max_tokens: Optional[int] = None
    timeout_seconds: float = 60.0
    retries: int = 1
    diary_count: int = Field(default=30, ge=10, le=30)
    cache_dir: str = ".citybehavex/llm_diaries"
    prompt_path: Optional[str] = None
    raw_response_path: Optional[str] = None
    validated_diaries_path: Optional[str] = None

    @model_validator(mode="after")
    def validate_client_fields(self) -> LLMConfig:
        if any([self.base_url, self.api_key, self.model]) and not all(
            [self.base_url, self.api_key, self.model]
        ):
            raise ValueError("llm base_url, api_key, and model must be provided together")
        return self


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
    # Rounded log-normal distribution for distinct daily locations, including HOME.
    location_count_mu: float = 1.0
    location_count_sigma: float = Field(default=0.5, gt=0)
    max_locations: int = Field(default=6, ge=1, le=6)

    def profile_for(self, day_type: str) -> str:
        """City profile for ``"weekday"`` / ``"weekend"``, falling back to the shared one."""
        specific = self.city_profile_weekday if day_type == "weekday" else self.city_profile_weekend
        return specific or self.city_profile


class EmbeddingConfig(BaseModel):
    """Diary-embedding backend for the ddCRP schedule selector.

    Embeddings are served over an OpenAI-compatible ``/v1/embeddings`` endpoint.
    If ``base_url`` is unset and ``auto_launch`` is true, a local vLLM server is
    spawned on demand (only when uncached diaries need embedding) and shut down
    afterwards. Computed vectors are cached to ``cache_path`` so the server runs
    rarely. When ``enabled`` is false (or every backend fails), the selector falls
    back to identity similarity (exact preferential return, no semantic smoothing).
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    model: str = "nomic-ai/nomic-embed-text-v2-moe"
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    task_prefix: str = "clustering: "
    dimensions: int = Field(default=768, gt=0)
    timeout_seconds: float = 120.0
    auto_launch: bool = True
    vllm_port: int = 8001
    vllm_startup_timeout_seconds: float = 600.0
    vllm_extra_args: list[str] = Field(default_factory=list)
    cache_dir: str = ".citybehavex/embeddings"
    cache_path: Optional[str] = None

    def resolved_cache_path(self) -> str:
        return self.cache_path or str(Path(self.cache_dir) / "diary_embeddings.npz")


class ScheduleConfig(BaseModel):
    """Distance-dependent Chinese Restaurant Process (ddCRP) schedule selection.

    Each simulated day an agent picks one whole LLM diary, weighted by calendar
    recency (habit) and the semantic similarity of candidate diaries to the diaries
    it used on past same-type days. Weekday/weekend are hard-separated banks.
    """

    model_config = ConfigDict(extra="forbid")

    lam: float = Field(default=0.15, ge=0)  # recency decay per day
    rho: float = Field(default=0.6, ge=0)  # exploration coefficient
    gamma: float = Field(default=0.21, ge=0)  # exploration decay exponent
    memory_window_days: int = Field(default=60, ge=1)  # memory truncation
    implant_memory: bool = True  # seed cold-start memory per agent
    semantic_temperature: float = Field(default=1.0, gt=0)  # scales similarity


class ComparisonConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Optional[str] = None
    label: str = "observed"
    html: str = "comparison.html"


class CityBehavExConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tessellation: TessellationConfig = Field(default_factory=TessellationConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    diaries: DiariesConfig = Field(default_factory=DiariesConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig)
    comparison: ComparisonConfig = Field(default_factory=ComparisonConfig)


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def load_config(path: Optional[str]) -> CityBehavExConfig:
    if path is None:
        return CityBehavExConfig()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("config file must contain a YAML mapping")
    return CityBehavExConfig.model_validate(_expand_env(raw))


def apply_overrides(model: BaseModel, overrides: dict[str, Any]) -> BaseModel:
    clean = {key: value for key, value in overrides.items() if value is not None}
    if not clean:
        return model
    data = model.model_dump()
    data.update(clean)
    return model.__class__.model_validate(data)
