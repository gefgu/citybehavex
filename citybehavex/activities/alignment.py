from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import requests

from citybehavex.activities.catalog import N_PURPOSES, Activity, build_catalog
from citybehavex.activities.config import ActivitiesConfig
from citybehavex.activities.poi_semantic import (
    PoiSemanticActivityData,
    build_poi_semantic_activity_data,
    example_categories_by_semantic_cluster,
)
from citybehavex.llm_diaries import Diary
from citybehavex.utils.alignment import (
    alignment_cache_key,
    score_cached_alignment_pairs,
    score_chunk_with_retries,
)

START_PREVIOUS_ACTIVITY = -1


@dataclass(frozen=True)
class ActivityBlock:
    block_id: int
    diary_id: str
    episode_index: int
    purpose: str
    start: str
    end: str


@dataclass(frozen=True)
class ProfileClusters:
    labels: np.ndarray
    narratives: list[str]
    representative_indices: np.ndarray


@dataclass(frozen=True)
class ActivityAlignmentScores:
    scores: np.ndarray
    cluster_labels: np.ndarray
    clusters: ProfileClusters
    blocks: list[ActivityBlock]
    metadata: pd.DataFrame


def cluster_profile_embeddings(
    narratives: Sequence[str],
    profile_embeddings: np.ndarray | None,
    threshold: float,
) -> ProfileClusters:
    """Greedily cluster profiles by cosine similarity for scorer reuse."""
    n = len(narratives)
    if n == 0:
        return ProfileClusters(
            labels=np.empty(0, dtype=np.int64),
            narratives=[],
            representative_indices=np.empty(0, dtype=np.int64),
        )
    if profile_embeddings is None or profile_embeddings.shape[0] != n:
        return ProfileClusters(
            labels=np.arange(n, dtype=np.int64),
            narratives=list(narratives),
            representative_indices=np.arange(n, dtype=np.int64),
        )

    embeddings = np.asarray(profile_embeddings, dtype=np.float64)
    labels = np.full(n, -1, dtype=np.int64)
    representatives: list[int] = []
    for idx in range(n):
        if labels[idx] >= 0:
            continue
        cluster_id = len(representatives)
        representatives.append(idx)
        labels[idx] = cluster_id
        sims = embeddings @ embeddings[idx]
        labels[(labels < 0) & (sims >= threshold)] = cluster_id

    representative_indices = np.asarray(representatives, dtype=np.int64)
    return ProfileClusters(
        labels=labels,
        narratives=[str(narratives[i]) for i in representative_indices],
        representative_indices=representative_indices,
    )


def expand_cluster_scores(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Expand a cluster-level matrix/tensor to one row per agent."""
    if len(labels) == 0:
        return np.empty((0, *scores.shape[1:]), dtype=np.float64)
    if int(labels.max()) >= scores.shape[0]:
        raise ValueError("cluster labels reference a missing score row")
    return np.asarray(scores, dtype=np.float64)[labels]


def diary_activity_blocks(diaries: Sequence[Diary]) -> list[ActivityBlock]:
    blocks: list[ActivityBlock] = []
    for diary in diaries:
        for episode_index, episode in enumerate(diary.episodes):
            blocks.append(
                ActivityBlock(
                    block_id=len(blocks),
                    diary_id=diary.diary_id,
                    episode_index=episode_index,
                    purpose=episode.purpose,
                    start=episode.start,
                    end=episode.end,
                )
            )
    return blocks


def _purpose_code(purpose: str) -> int:
    if purpose == "HOME":
        return 0
    if purpose == "WORK":
        return 1
    return 2


def _eligible_activity_indices(block: ActivityBlock, catalog: Sequence[Activity]) -> list[int]:
    purpose = _purpose_code(block.purpose)
    return [activity.idx for activity in catalog if purpose in activity.eligible_purposes]


def _uses_contextual_block_alignment(block: ActivityBlock) -> bool:
    return block.purpose in {"HOME", "WORK"}


def _activity_text(activity: Activity) -> str:
    return f"{activity.name}: {activity.description}"


def _previous_activity_text(previous: int, catalog: Sequence[Activity]) -> str:
    if previous == START_PREVIOUS_ACTIVITY:
        return "no previous micro-activity in this block"
    if 0 <= previous < len(catalog):
        activity = catalog[previous]
        return f"previous micro-activity was {activity.name}: {activity.description}"
    return "previous micro-activity is unknown"


def _query_text(
    profile_text: str,
    block: ActivityBlock,
    previous: int,
    catalog: Sequence[Activity],
) -> str:
    return (
        f"{profile_text}\n"
        f"Schedule block: diary {block.diary_id}, block {block.episode_index}, "
        f"{block.purpose} from {block.start} to {block.end}.\n"
        f"Transition/history context: {_previous_activity_text(previous, catalog)}.\n"
        "Score which valid time-use activity best fits this person, block, time, and history."
    )


_PERIODS: tuple[tuple[int, int, str], ...] = (
    (0, 6 * 60, "00-06"),
    (6 * 60, 12 * 60, "06-12"),
    (12 * 60, 18 * 60, "12-18"),
    (18 * 60, 24 * 60, "18-24"),
)


def _time_to_minutes(value: str) -> int:
    hour, minute = (int(part) for part in value.split(":", maxsplit=1))
    return 24 * 60 if hour == 24 else hour * 60 + minute


def _period_index_for_block(block: ActivityBlock) -> int:
    start = _time_to_minutes(block.start)
    end = _time_to_minutes(block.end)
    overlaps = [
        max(0, min(end, period_end) - max(start, period_start))
        for period_start, period_end, _label in _PERIODS
    ]
    return max(range(len(overlaps)), key=lambda idx: (overlaps[idx], -idx))


def _period_label(period_index: int) -> str:
    return _PERIODS[period_index][2]


def _period_query_text(
    profile_text: str,
    purpose: str,
    period_index: int,
    previous: int,
    catalog: Sequence[Activity],
) -> str:
    period_start, period_end, period_label = _PERIODS[period_index]
    return (
        f"{profile_text}\n"
        f"Schedule block group: {purpose} blocks mostly in the {period_label} "
        f"period ({period_start // 60:02d}:00-{period_end // 60:02d}:00).\n"
        f"Transition/history context: {_previous_activity_text(previous, catalog)}.\n"
        "Score which valid time-use activity best fits this person, block period, time, and history."
    )


def _cache_key(
    model: str | None,
    profile_text: str,
    block: ActivityBlock,
    previous: int,
    activity_text: str,
) -> str:
    return alignment_cache_key(
        model,
        profile_text,
        block.diary_id,
        str(block.episode_index),
        block.purpose,
        block.start,
        block.end,
        str(previous),
        activity_text,
    )


def _period_cache_key(
    model: str | None,
    profile_text: str,
    purpose: str,
    period_index: int,
    previous: int,
    activity_text: str,
) -> str:
    return alignment_cache_key(
        model,
        profile_text,
        "PERIOD_BLOCK",
        purpose,
        str(period_index),
        str(previous),
        activity_text,
    )


def _poi_query_text(
    profile_text: str,
    semantic_cluster: str,
    example_categories: Sequence[str],
) -> str:
    examples = ", ".join(example_categories[:12]) if example_categories else "unknown POI types"
    return (
        f"{profile_text}\n"
        f"Public POI context: semantic cluster {semantic_cluster}. "
        f"Example POI types: {examples}.\n"
        "Score which valid time-use activity best fits this person at this kind of public place."
    )


def _poi_type_query_text(profile_text: str, block: ActivityBlock) -> str:
    return (
        f"{profile_text}\n"
        f"Schedule block: diary {block.diary_id}, block {block.episode_index}, "
        f"{block.purpose} from {block.start} to {block.end}.\n"
        "Score which kind of public place best fits this person and schedule block."
    )


def _poi_type_candidate_text(semantic_cluster: str, example_categories: Sequence[str]) -> str:
    examples = ", ".join(example_categories[:12]) if example_categories else "unknown POI types"
    return f"{semantic_cluster}: public place type with example Overture categories {examples}"


def _poi_type_cache_key(
    model: str | None,
    profile_text: str,
    block: ActivityBlock,
    semantic_cluster: str,
) -> str:
    return alignment_cache_key(
        model,
        profile_text,
        "POI_TYPE",
        block.diary_id,
        str(block.episode_index),
        block.purpose,
        block.start,
        block.end,
        semantic_cluster,
    )


def _score_chunk_with_retries(
    base_url: str,
    model: str | None,
    pairs: Sequence[tuple[str, str]],
    *,
    timeout: float,
    retries: int,
) -> list[float]:
    return score_chunk_with_retries(
        base_url,
        model,
        pairs,
        timeout=timeout,
        retries=retries,
        requests_module=requests,
    )


def score_activity_alignment(
    cluster_narratives: Sequence[str],
    diaries: Sequence[Diary],
    config: ActivitiesConfig,
) -> Optional[tuple[np.ndarray, list[ActivityBlock], pd.DataFrame]]:
    """Return contextual micro-activity alignment scores or ``None`` on failure.

    Shape is ``[n_clusters, n_blocks, n_activities + 1, n_activities]``. The
    third dimension reserves index 0 for no previous activity and activity ``a``
    at index ``a + 1``.

    Only HOME and WORK blocks are scored by this contextual tensor. OTHER/POI
    blocks are handled by ``score_poi_semantic_alignment`` and remain
    zero-initialized here. Reranker requests are reused across raw diary blocks
    by scoring canonical ``(purpose, 6-hour period)`` groups, then copying those
    canonical scores into every matching raw block slot expected by Rust.
    """
    if (
        not cluster_narratives
        or not diaries
        or config.alignment_backend != "rerank"
        or not config.alignment_base_url
    ):
        return None

    catalog = build_catalog()
    n_activities = len(catalog)
    blocks = diary_activity_blocks(diaries)
    scores = np.zeros(
        (len(cluster_narratives), len(blocks), n_activities + 1, n_activities),
        dtype=np.float64,
    )
    rows: list[dict[str, object]] = []

    # A block's true previous activity is whichever activity the agent last
    # performed, which is always either START (nothing simulated yet) or an
    # activity eligible for the *actual preceding episode's* purpose: the same
    # diary's prior episode, or -- for a diary's first episode -- any diary's
    # closing episode, since days chain onto each other across the week.
    # Restricting to that set (rather than every catalog activity) is provably
    # safe because the Rust engine can never present any other previous state
    # to this block; scoring the rest would be wasted work. Diary ids repeat
    # across weekday/weekend banks, so predecessors are computed by mirroring
    # diary_activity_blocks' own (diary, episode) iteration rather than a
    # diary_id lookup, which would silently collide.
    day_boundary_purposes = sorted(
        {_purpose_code(diary.episodes[-1].purpose) for diary in diaries}
    )
    eligible_by_purpose = {
        purpose: [activity.idx for activity in catalog if purpose in activity.eligible_purposes]
        for purpose in range(N_PURPOSES)
    }

    block_previous_candidates: list[list[int]] = []
    for diary in diaries:
        for episode_index, episode in enumerate(diary.episodes):
            if episode_index > 0:
                purpose_codes = [_purpose_code(diary.episodes[episode_index - 1].purpose)]
            else:
                purpose_codes = day_boundary_purposes
            candidates = {START_PREVIOUS_ACTIVITY}
            for purpose in purpose_codes:
                candidates.update(eligible_by_purpose[purpose])
            block_previous_candidates.append(sorted(candidates))

    block_eligible = [
        _eligible_activity_indices(block, catalog) if _uses_contextual_block_alignment(block) else []
        for block in blocks
    ]

    block_periods = [
        _period_index_for_block(block) if eligible else -1
        for block, eligible in zip(blocks, block_eligible)
    ]
    group_previous_candidates: dict[tuple[str, int], list[int]] = {}
    grouped: dict[tuple[str, int], set[int]] = {}
    for block, eligible, period_index, previous_candidates in zip(
        blocks, block_eligible, block_periods, block_previous_candidates
    ):
        if not eligible:
            continue
        grouped.setdefault((block.purpose, period_index), set()).update(previous_candidates)
    group_previous_candidates = {
        group: sorted(previous) for group, previous in grouped.items()
    }

    try:
        pairs: list[tuple[str, str, str]] = []
        for profile_text in cluster_narratives:
            for purpose, period_index in sorted(group_previous_candidates):
                eligible = [
                    activity.idx
                    for activity in catalog
                    if _purpose_code(purpose) in activity.eligible_purposes
                ]
                texts = [_activity_text(catalog[idx]) for idx in eligible]
                for previous in group_previous_candidates[(purpose, period_index)]:
                    query = _period_query_text(profile_text, purpose, period_index, previous, catalog)
                    for text in texts:
                        key = _period_cache_key(
                            config.alignment_model,
                            profile_text,
                            purpose,
                            period_index,
                            previous,
                            text,
                        )
                        pairs.append((key, query, text))

        cache = score_cached_alignment_pairs(
            pairs,
            base_url=config.alignment_base_url,
            model=config.alignment_model,
            batch_size=config.alignment_batch_size,
            cache_path=config.alignment_cache_path,
            concurrency=config.alignment_concurrency,
            timeout_seconds=config.alignment_timeout_seconds,
            retries=config.alignment_retries,
            checkpoint_every=config.alignment_checkpoint_every,
            progress_label="Activity alignment",
            score_chunk=_score_chunk_with_retries,
        )

        for cluster_id, profile_text in enumerate(cluster_narratives):
            for block, eligible, previous_candidates, period_index in zip(
                blocks, block_eligible, block_previous_candidates, block_periods
            ):
                if not eligible:
                    continue
                texts = [_activity_text(catalog[idx]) for idx in eligible]
                for previous in previous_candidates:
                    prev_pos = 0 if previous == START_PREVIOUS_ACTIVITY else previous + 1
                    for local_idx, activity_idx in enumerate(eligible):
                        key = _period_cache_key(
                            config.alignment_model,
                            profile_text,
                            block.purpose,
                            period_index,
                            previous,
                            texts[local_idx],
                        )
                        score = float(cache[key])
                        scores[cluster_id, block.block_id, prev_pos, activity_idx] = score
                        rows.append(
                            {
                                "cluster": cluster_id,
                                "diary_id": block.diary_id,
                                "block_index": block.episode_index,
                                "block_id": block.block_id,
                                "purpose": block.purpose,
                                "start": block.start,
                                "end": block.end,
                                "period_index": period_index,
                                "period_label": _period_label(period_index),
                                "previous_activity": previous,
                                "activity": catalog[activity_idx].name,
                                "activity_idx": activity_idx,
                                "score": score,
                            }
                        )
    except Exception:  # noqa: BLE001 - callers intentionally fall back.
        return None

    return np.clip(scores, 0.0, 1.0), blocks, pd.DataFrame(rows)


def score_poi_semantic_alignment(
    cluster_narratives: Sequence[str],
    config: ActivitiesConfig,
    poi_data: PoiSemanticActivityData | None = None,
) -> Optional[tuple[np.ndarray, pd.DataFrame]]:
    """Return POI semantic-cluster alignment scores or ``None`` on failure.

    Shape is ``[n_profile_clusters, n_semantic_clusters, n_activities]``.
    Only activities allowed by the hard semantic-cluster mask are sent to the
    reranker; every other slot remains zero and is ignored by Rust's mask.
    """
    if (
        not cluster_narratives
        or config.alignment_backend != "rerank"
        or not config.alignment_base_url
    ):
        return None

    catalog = build_catalog()
    n_activities = len(catalog)
    poi_data = poi_data or build_poi_semantic_activity_data()
    scores = np.zeros(
        (len(cluster_narratives), len(poi_data.semantic_clusters), n_activities),
        dtype=np.float64,
    )
    rows: list[dict[str, object]] = []
    examples_by_cluster: dict[str, list[str]] = {cluster: [] for cluster in poi_data.semantic_clusters}
    for category, cluster in poi_data.category_to_cluster.items():
        if cluster in examples_by_cluster and len(examples_by_cluster[cluster]) < 12:
            examples_by_cluster[cluster].append(category)

    try:
        pairs: list[tuple[str, str, str]] = []
        for profile_text in cluster_narratives:
            for semantic_cluster_id, semantic_cluster in enumerate(poi_data.semantic_clusters):
                start = int(poi_data.mask_starts[semantic_cluster_id])
                end = int(poi_data.mask_starts[semantic_cluster_id + 1])
                allowed = poi_data.mask_activities[start:end]
                if len(allowed) == 0:
                    continue
                query = _poi_query_text(
                    profile_text,
                    semantic_cluster,
                    examples_by_cluster.get(semantic_cluster, []),
                )
                for activity_idx in allowed:
                    text = _activity_text(catalog[int(activity_idx)])
                    key = alignment_cache_key(
                        config.alignment_model,
                        profile_text,
                        "POI",
                        semantic_cluster,
                        text,
                    )
                    pairs.append((key, query, text))

        cache = score_cached_alignment_pairs(
            pairs,
            base_url=config.alignment_base_url,
            model=config.alignment_model,
            batch_size=config.alignment_batch_size,
            cache_path=config.alignment_cache_path,
            concurrency=config.alignment_concurrency,
            timeout_seconds=config.alignment_timeout_seconds,
            retries=config.alignment_retries,
            checkpoint_every=config.alignment_checkpoint_every,
            progress_label="POI activity alignment",
            score_chunk=_score_chunk_with_retries,
        )

        for cluster_id, profile_text in enumerate(cluster_narratives):
            for semantic_cluster_id, semantic_cluster in enumerate(poi_data.semantic_clusters):
                start = int(poi_data.mask_starts[semantic_cluster_id])
                end = int(poi_data.mask_starts[semantic_cluster_id + 1])
                for activity_idx in poi_data.mask_activities[start:end]:
                    activity = catalog[int(activity_idx)]
                    text = _activity_text(activity)
                    key = alignment_cache_key(
                        config.alignment_model,
                        profile_text,
                        "POI",
                        semantic_cluster,
                        text,
                    )
                    score = float(cache[key])
                    scores[cluster_id, semantic_cluster_id, int(activity_idx)] = score
                    rows.append(
                        {
                            "cluster": cluster_id,
                            "semantic_cluster_id": semantic_cluster_id,
                            "semantic_cluster": semantic_cluster,
                            "activity": activity.name,
                            "activity_idx": int(activity.idx),
                            "score": score,
                        }
                    )
    except Exception:  # noqa: BLE001 - callers intentionally fall back.
        return None

    return np.clip(scores, 0.0, 1.0), pd.DataFrame(rows)


def score_poi_type_alignment(
    cluster_narratives: Sequence[str],
    diaries: Sequence[Diary],
    config: ActivitiesConfig,
    poi_data: PoiSemanticActivityData | None = None,
    available_cluster_ids: Sequence[int] | None = None,
) -> Optional[tuple[np.ndarray, list[ActivityBlock], pd.DataFrame]]:
    """Return OTHER-block POI type alignment scores or ``None`` on failure.

    Shape is ``[n_profile_clusters, n_blocks, n_semantic_clusters]``. Only
    OTHER schedule blocks and semantic clusters present in the tessellation
    are sent to the reranker; all other slots remain zero.
    """
    if (
        not cluster_narratives
        or not diaries
        or not config.poi_type_choice_enabled
        or config.alignment_backend != "rerank"
        or not config.alignment_base_url
    ):
        return None

    poi_data = poi_data or build_poi_semantic_activity_data()
    blocks = diary_activity_blocks(diaries)
    n_clusters = len(poi_data.semantic_clusters)
    scores = np.zeros((len(cluster_narratives), len(blocks), n_clusters), dtype=np.float64)
    rows: list[dict[str, object]] = []
    examples = example_categories_by_semantic_cluster(poi_data)

    if available_cluster_ids is None:
        cluster_ids = list(range(n_clusters))
    else:
        cluster_ids = sorted({
            int(cluster_id)
            for cluster_id in available_cluster_ids
            if 0 <= int(cluster_id) < n_clusters
        })
    if not cluster_ids:
        return None

    try:
        pairs: list[tuple[str, str, str]] = []
        for profile_text in cluster_narratives:
            for block in blocks:
                if block.purpose != "OTHER":
                    continue
                query = _poi_type_query_text(profile_text, block)
                for semantic_cluster_id in cluster_ids:
                    semantic_cluster = poi_data.semantic_clusters[semantic_cluster_id]
                    text = _poi_type_candidate_text(
                        semantic_cluster,
                        examples.get(semantic_cluster, []),
                    )
                    key = _poi_type_cache_key(
                        config.alignment_model,
                        profile_text,
                        block,
                        semantic_cluster,
                    )
                    pairs.append((key, query, text))

        cache = score_cached_alignment_pairs(
            pairs,
            base_url=config.alignment_base_url,
            model=config.alignment_model,
            batch_size=config.alignment_batch_size,
            cache_path=config.alignment_cache_path,
            concurrency=config.alignment_concurrency,
            timeout_seconds=config.alignment_timeout_seconds,
            retries=config.alignment_retries,
            checkpoint_every=config.alignment_checkpoint_every,
            progress_label="POI type alignment",
            score_chunk=_score_chunk_with_retries,
        )

        for cluster_id, profile_text in enumerate(cluster_narratives):
            for block in blocks:
                if block.purpose != "OTHER":
                    continue
                for semantic_cluster_id in cluster_ids:
                    semantic_cluster = poi_data.semantic_clusters[semantic_cluster_id]
                    key = _poi_type_cache_key(
                        config.alignment_model,
                        profile_text,
                        block,
                        semantic_cluster,
                    )
                    score = float(cache[key])
                    scores[cluster_id, block.block_id, semantic_cluster_id] = score
                    rows.append(
                        {
                            "cluster": cluster_id,
                            "diary_id": block.diary_id,
                            "block_index": block.episode_index,
                            "block_id": block.block_id,
                            "purpose": block.purpose,
                            "start": block.start,
                            "end": block.end,
                            "semantic_cluster": semantic_cluster,
                            "semantic_cluster_id": semantic_cluster_id,
                            "score": score,
                        }
                    )
    except Exception:  # noqa: BLE001 - callers intentionally fall back.
        return None

    return np.clip(scores, 0.0, 1.0), blocks, pd.DataFrame(rows)
