from __future__ import annotations

from .config import EmbeddingConfig
from .service import (
    cosine_sim_matrix,
    cross_cosine,
    diary_to_prose,
    diary_to_text,
    embed_diaries,
    embed_profiles,
    embed_texts,
)

__all__ = [
    "EmbeddingConfig",
    "cosine_sim_matrix",
    "cross_cosine",
    "diary_to_prose",
    "diary_to_text",
    "embed_diaries",
    "embed_profiles",
    "embed_texts",
]
