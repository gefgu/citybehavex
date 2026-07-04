from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class EmbeddingConfig(BaseModel):
    """Diary-embedding backend for the ddCRP schedule selector."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    model: str = "nomic-ai/nomic-embed-text-v2-moe"
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    task_prefix: str = "clustering: "
    dimensions: int = Field(default=768, gt=0)
    timeout_seconds: float = 120.0
    auto_launch: bool = True
    vllm_executable: str = "vllm"
    vllm_port: int = 8001
    vllm_startup_timeout_seconds: float = 600.0
    vllm_extra_args: list[str] = Field(default_factory=list)
    cache_dir: str = ".citybehavex/embeddings"
    cache_path: Optional[str] = None

    def resolved_cache_path(self) -> str:
        return self.cache_path or str(Path(self.cache_dir) / "diary_embeddings.npz")
