from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from citybehavex.activities.alignment import (
    ProfileClusters,
    _score_chunk_with_retries,
)
from citybehavex.profiles._common import (
    alignment_cache_key,
    alignment_query_text,
    score_cached_alignment_pairs,
)
from citybehavex.profiles.config import AgentProfilesConfig

VEHICLE_CANDIDATES: tuple[tuple[str, str], ...] = (
    (
        "car",
        "owns or has reliable access to a private car for everyday travel",
    ),
    (
        "bike",
        "owns or has reliable access to a bicycle, e-bike, or equivalent personal cycle",
    ),
)
OWNERSHIP_QUERY_INSTRUCTION = (
    "Score how likely this person is to have the listed transport option. "
    "Use 0 for very unlikely and 1 for very likely."
)


def _query_text(profile_text: str, city_profile: str | None) -> str:
    return alignment_query_text(profile_text, city_profile, OWNERSHIP_QUERY_INSTRUCTION)


def _cache_key(
    model: str | None,
    profile_text: str,
    city_profile: str | None,
    vehicle: str,
    candidate_text: str,
) -> str:
    return alignment_cache_key(model, profile_text, city_profile, vehicle, candidate_text)


def score_vehicle_ownership_alignment(
    cluster_narratives: Sequence[str],
    config: AgentProfilesConfig,
    *,
    city_profile: str | None = None,
) -> tuple[np.ndarray, pd.DataFrame] | None:
    """Return car/bike ownership probabilities for representative profiles.

    The returned score matrix has shape ``[n_clusters, 2]`` in
    ``VEHICLE_CANDIDATES`` order: car, bike.
    """
    if (
        not cluster_narratives
        or config.ownership_alignment_backend != "rerank"
        or not config.ownership_alignment_base_url
    ):
        return None

    scores = np.zeros((len(cluster_narratives), len(VEHICLE_CANDIDATES)), dtype=np.float64)
    rows: list[dict[str, object]] = []

    try:
        pairs: list[tuple[str, str, str]] = []
        for profile_text in cluster_narratives:
            query = _query_text(profile_text, city_profile)
            for vehicle, candidate_text in VEHICLE_CANDIDATES:
                key = _cache_key(
                    config.ownership_alignment_model,
                    profile_text,
                    city_profile,
                    vehicle,
                    candidate_text,
                )
                pairs.append((key, query, candidate_text))

        cache = score_cached_alignment_pairs(
            pairs,
            base_url=config.ownership_alignment_base_url,
            model=config.ownership_alignment_model,
            batch_size=config.ownership_alignment_batch_size,
            cache_path=config.ownership_alignment_cache_path,
            concurrency=config.ownership_alignment_concurrency,
            timeout_seconds=config.ownership_alignment_timeout_seconds,
            retries=config.ownership_alignment_retries,
            checkpoint_every=config.ownership_alignment_checkpoint_every,
            progress_label="Vehicle ownership alignment",
            score_chunk=_score_chunk_with_retries,
        )

        for cluster_id, profile_text in enumerate(cluster_narratives):
            query = _query_text(profile_text, city_profile)
            for vehicle_idx, (vehicle, candidate_text) in enumerate(VEHICLE_CANDIDATES):
                key = _cache_key(
                    config.ownership_alignment_model,
                    profile_text,
                    city_profile,
                    vehicle,
                    candidate_text,
                )
                score = float(cache[key])
                scores[cluster_id, vehicle_idx] = score
                rows.append(
                    {
                        "cluster": cluster_id,
                        "vehicle": vehicle,
                        "profile_text": profile_text,
                        "query_text": query,
                        "candidate_text": candidate_text,
                        "score": score,
                    }
                )
    except Exception:  # noqa: BLE001 - callers intentionally fall back.
        return None

    return np.clip(scores, 0.0, 1.0), pd.DataFrame(rows)


def expand_vehicle_scores(scores: np.ndarray, clusters: ProfileClusters) -> np.ndarray:
    if len(clusters.labels) == 0:
        return np.empty((0, scores.shape[1]), dtype=np.float64)
    if int(clusters.labels.max()) >= scores.shape[0]:
        raise ValueError("cluster labels reference a missing vehicle score row")
    return np.asarray(scores, dtype=np.float64)[clusters.labels]
