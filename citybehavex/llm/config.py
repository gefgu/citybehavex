from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None
    temperature: float = 0.4
    max_tokens: Optional[int] = None
    timeout_seconds: float = 60.0
    retries: int = 1
    diary_count: int = Field(default=30, ge=10, le=50)
    reuse_cache: bool = True
    cache_dir: str = ".citybehavex/llm_diaries"
    prompt_path: Optional[str] = None
    raw_response_path: Optional[str] = None
    validated_diaries_path: Optional[str] = None
    auto_launch: bool = False
    vllm_port: int = 8080
    vllm_startup_timeout_seconds: float = 600.0
    vllm_extra_args: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_client_fields(self) -> LLMConfig:
        if self.auto_launch:
            if not self.model:
                raise ValueError("llm model must be provided when auto_launch is enabled")
            return self
        if any([self.base_url, self.api_key, self.model]) and not all(
            [self.base_url, self.api_key, self.model]
        ):
            raise ValueError("llm base_url, api_key, and model must be provided together")
        return self
