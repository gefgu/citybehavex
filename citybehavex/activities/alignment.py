from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import requests

from citybehavex.activities.catalog import Activity, build_catalog
from citybehavex.activities.config import ActivitiesConfig
from citybehavex.llm_diaries import Diary

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


def _cache_key(
    model: str | None,
    profile_text: str,
    block: ActivityBlock,
    previous: int,
    activity_text: str,
) -> str:
    raw = (
        f"{model or ''}\x00{profile_text}\x00{block.diary_id}\x00"
        f"{block.episode_index}\x00{block.purpose}\x00{block.start}\x00"
        f"{block.end}\x00{previous}\x00{activity_text}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_cache(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    try:
        data = np.load(path, allow_pickle=False)
        keys = data["keys"]
        scores = data["scores"]
    except Exception:  # noqa: BLE001 - corrupt cache should not break a run.
        return {}
    return {str(k): float(scores[i]) for i, k in enumerate(keys)}


def _save_cache(path: Path, cache: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = np.array(list(cache.keys()))
    scores = np.array([cache[k] for k in cache], dtype=np.float32)
    np.savez(path, keys=keys, scores=scores)


def _extract_scores(payload: Any, expected: int) -> Optional[list[float]]:
    rows: Any
    if isinstance(payload, dict):
        rows = (
            payload.get("data")
            or payload.get("results")
            or payload.get("scores")
            or payload.get("rerank")
        )
    else:
        rows = payload
    if rows is None:
        return None
    if isinstance(rows, list) and all(isinstance(x, (int, float)) for x in rows):
        if len(rows) != expected:
            return None
        return [float(x) for x in rows]
    if isinstance(rows, list) and all(isinstance(x, dict) for x in rows):
        scores = [0.0] * expected
        seen = set()
        for pos, row in enumerate(rows):
            idx = int(row.get("index", row.get("corpus_id", pos)))
            if idx < 0 or idx >= expected:
                return None
            raw_score = row.get("score", row.get("relevance_score"))
            if raw_score is None:
                return None
            scores[idx] = float(raw_score)
            seen.add(idx)
        if len(seen) != expected:
            return None
        return scores
    return None


def _post_rerank(
    base_url: str,
    model: str | None,
    query: str,
    texts: Sequence[str],
    *,
    timeout: float,
) -> Optional[list[float]]:
    payload: dict[str, Any] = {
        "query": query,
        "texts": list(texts),
        "raw_scores": False,
        "truncate": True,
    }
    if model:
        payload["model"] = model
    resp = requests.post(
        base_url.rstrip("/") + "/rerank",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return _extract_scores(resp.json(), len(texts))


def score_activity_alignment(
    cluster_narratives: Sequence[str],
    diaries: Sequence[Diary],
    config: ActivitiesConfig,
) -> Optional[tuple[np.ndarray, list[ActivityBlock], pd.DataFrame]]:
    """Return contextual micro-activity alignment scores or ``None`` on failure.

    Shape is ``[n_clusters, n_blocks, n_activities + 1, n_activities]``. The
    third dimension reserves index 0 for no previous activity and activity ``a``
    at index ``a + 1``.
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
    previous_values = [START_PREVIOUS_ACTIVITY, *range(n_activities)]
    blocks = diary_activity_blocks(diaries)
    scores = np.zeros(
        (len(cluster_narratives), len(blocks), n_activities + 1, n_activities),
        dtype=np.float64,
    )
    rows: list[dict[str, object]] = []

    cache: dict[str, float] = {}
    cache_path = Path(config.alignment_cache_path) if config.alignment_cache_path else None
    if cache_path is not None:
        cache = _load_cache(cache_path)

    try:
        for cluster_id, profile_text in enumerate(cluster_narratives):
            for block in blocks:
                eligible = _eligible_activity_indices(block, catalog)
                if not eligible:
                    continue
                texts = [_activity_text(catalog[idx]) for idx in eligible]
                for prev_pos, previous in enumerate(previous_values):
                    keys = [
                        _cache_key(config.alignment_model, profile_text, block, previous, text)
                        for text in texts
                    ]
                    missing = [idx for idx, key in enumerate(keys) if key not in cache]
                    query = _query_text(profile_text, block, previous, catalog)
                    for start in range(0, len(missing), config.alignment_batch_size):
                        chunk_idx = missing[start : start + config.alignment_batch_size]
                        chunk_scores = _post_rerank(
                            config.alignment_base_url,
                            config.alignment_model,
                            query,
                            [texts[i] for i in chunk_idx],
                            timeout=config.alignment_timeout_seconds,
                        )
                        if chunk_scores is None:
                            return None
                        for idx, score in zip(chunk_idx, chunk_scores):
                            cache[keys[idx]] = float(np.clip(score, 0.0, 1.0))
                    for local_idx, activity_idx in enumerate(eligible):
                        score = float(cache[keys[local_idx]])
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
                                "previous_activity": previous,
                                "activity": catalog[activity_idx].name,
                                "activity_idx": activity_idx,
                                "score": score,
                            }
                        )
    except Exception:  # noqa: BLE001 - callers intentionally fall back.
        return None

    if cache_path is not None:
        _save_cache(cache_path, cache)
    return np.clip(scores, 0.0, 1.0), blocks, pd.DataFrame(rows)
