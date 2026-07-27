from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from citybehavex.activities.alignment import (
    ProfileClusters,
    _score_chunk_with_retries,
)
from citybehavex.utils.alignment import (
    alignment_cache_key,
    alignment_query_text,
    score_cached_alignment_pairs,
)
from citybehavex.profiles.config import AgentProfilesConfig

COHERENCE_CANDIDATE_TEXT = "demographically coherent and valid synthetic agent profile"
COHERENCE_QUERY_INSTRUCTION = (
    "Score whether this synthetic agent profile is demographically coherent "
    "and plausible. Use 0 for impossible or highly inconsistent profiles and "
    "1 for fully coherent profiles."
)


def score_profile_coherence_alignment(
    cluster_narratives: Sequence[str],
    config: AgentProfilesConfig,
    *,
    city_profile: str | None = None,
) -> tuple[np.ndarray, pd.DataFrame] | None:
    """Return demographic coherence scores for representative profiles."""
    if (
        not cluster_narratives
        or config.coherence_alignment_backend != "rerank"
        or not config.coherence_alignment_base_url
    ):
        return None

    scores = np.zeros(len(cluster_narratives), dtype=np.float64)
    rows: list[dict[str, object]] = []

    try:
        pairs: list[tuple[str, str, str]] = []
        for profile_text in cluster_narratives:
            query = alignment_query_text(profile_text, city_profile, COHERENCE_QUERY_INSTRUCTION)
            key = alignment_cache_key(
                config.coherence_alignment_model,
                profile_text,
                city_profile,
                COHERENCE_CANDIDATE_TEXT,
            )
            pairs.append((key, query, COHERENCE_CANDIDATE_TEXT))

        cache = score_cached_alignment_pairs(
            pairs,
            base_url=config.coherence_alignment_base_url,
            model=config.coherence_alignment_model,
            batch_size=config.coherence_alignment_batch_size,
            cache_path=config.coherence_alignment_cache_path,
            concurrency=config.coherence_alignment_concurrency,
            timeout_seconds=config.coherence_alignment_timeout_seconds,
            retries=config.coherence_alignment_retries,
            checkpoint_every=config.coherence_alignment_checkpoint_every,
            progress_label="Profile coherence alignment",
            score_chunk=_score_chunk_with_retries,
        )

        for cluster_id, profile_text in enumerate(cluster_narratives):
            query = alignment_query_text(profile_text, city_profile, COHERENCE_QUERY_INSTRUCTION)
            key = alignment_cache_key(
                config.coherence_alignment_model,
                profile_text,
                city_profile,
                COHERENCE_CANDIDATE_TEXT,
            )
            score = float(cache[key])
            scores[cluster_id] = score
            rows.append(
                {
                    "cluster": cluster_id,
                    "profile_text": profile_text,
                    "query_text": query,
                    "candidate_text": COHERENCE_CANDIDATE_TEXT,
                    "score": score,
                }
            )
    except Exception:  # noqa: BLE001 - callers intentionally fall back.
        return None

    return np.clip(scores, 0.0, 1.0), pd.DataFrame(rows)


def expand_coherence_scores(scores: np.ndarray, clusters: ProfileClusters) -> np.ndarray:
    if len(clusters.labels) == 0:
        return np.empty(0, dtype=np.float64)
    if int(clusters.labels.max()) >= scores.shape[0]:
        raise ValueError("cluster labels reference a missing coherence score row")
    return np.asarray(scores, dtype=np.float64)[clusters.labels]
