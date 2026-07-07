from __future__ import annotations

import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

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

START_PREVIOUS_ACTIVITY = -1


def _format_duration(seconds: float) -> str:
    if not np.isfinite(seconds):
        return "unknown"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.1f}h"


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


def _poi_cache_key(
    model: str | None,
    profile_text: str,
    semantic_cluster: str,
    activity_text: str,
) -> str:
    raw = f"{model or ''}\x00{profile_text}\x00POI\x00{semantic_cluster}\x00{activity_text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


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
    raw = (
        f"{model or ''}\x00{profile_text}\x00POI_TYPE\x00{block.diary_id}\x00"
        f"{block.episode_index}\x00{block.purpose}\x00{block.start}\x00"
        f"{block.end}\x00{semantic_cluster}"
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
    """Write the cache atomically: a crash or interrupt mid-write must never
    leave `path` holding a truncated/corrupt ``.npz``, since it's reused
    across runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = np.array(list(cache.keys()))
    scores = np.array([cache[k] for k in cache], dtype=np.float32)
    # Write via an explicit file handle (rather than a bare path) so numpy
    # doesn't append its own ".npz" suffix to the temp name, which would
    # break the atomic rename below.
    tmp_path = path.parent / (path.name + ".tmp")
    try:
        with open(tmp_path, "wb") as fh:
            np.savez(fh, keys=keys, scores=scores)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


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


def _post_pair_scores(
    base_url: str,
    model: str | None,
    pairs: Sequence[tuple[str, str]],
    *,
    timeout: float,
) -> Optional[list[float]]:
    payload: dict[str, Any] = {
        "pairs": [[query, text] for query, text in pairs],
        "raw_scores": False,
        "truncate": True,
    }
    if model:
        payload["model"] = model
    resp = requests.post(
        base_url.rstrip("/") + "/score_pairs",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return _extract_scores(resp.json(), len(pairs))


def _score_chunk_with_retries(
    base_url: str,
    model: str | None,
    pairs: Sequence[tuple[str, str]],
    *,
    timeout: float,
    retries: int,
) -> list[float]:
    """Score one batch, retrying transient failures. Raises (rather than
    returning ``None``) once retries are exhausted, so callers' existing
    broad ``except Exception: return None`` fallback still applies."""
    last_error: Exception | None = None
    for _attempt in range(max(1, retries)):
        try:
            scores = _post_pair_scores(base_url, model, pairs, timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - retry, raise on exhaustion.
            last_error = exc
            continue
        if scores is None:
            last_error = ValueError("reranker response could not be parsed")
            continue
        return scores
    raise RuntimeError(f"failed to score a batch of {len(pairs)} pairs") from last_error


def score_activity_alignment(
    cluster_narratives: Sequence[str],
    diaries: Sequence[Diary],
    config: ActivitiesConfig,
    visited_pairs: Optional[set[tuple[int, int]]] = None,
) -> Optional[tuple[np.ndarray, list[ActivityBlock], pd.DataFrame]]:
    """Return contextual micro-activity alignment scores or ``None`` on failure.

    Shape is ``[n_clusters, n_blocks, n_activities + 1, n_activities]``. The
    third dimension reserves index 0 for no previous activity and activity ``a``
    at index ``a + 1``.

    Only HOME and WORK blocks are scored by this legacy contextual tensor.
    OTHER/POI blocks are handled by ``score_poi_semantic_alignment`` and remain
    zero-initialized here.

    ``visited_pairs``, when given, is a set of ``(cluster_id, block_id)`` pairs
    a cheap reachability probe found were actually visited; any other pair is
    skipped entirely (both the network request and the tensor slot, which
    stays at its zero-initialized default -- a documented graceful-degradation
    fallback toward base-rate weighting, not a crash). Pruning at this
    (cluster, block) granularity rather than per previous-activity is
    deliberate: block visitation is driven by diary structure, identical
    between the probe run and the final run, while which previous activity
    gets sampled within an already-visited block depends on weights that
    differ between the two runs.
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

    cache: dict[str, float] = {}
    cache_path = Path(config.alignment_cache_path) if config.alignment_cache_path else None
    if cache_path is not None:
        cache = _load_cache(cache_path)

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

    def should_score(cluster_id: int, block: ActivityBlock) -> bool:
        return visited_pairs is None or (cluster_id, block.block_id) in visited_pairs

    try:
        pending: list[tuple[str, str, str]] = []
        pending_seen: set[str] = set()
        for cluster_id, profile_text in enumerate(cluster_narratives):
            for block, eligible, previous_candidates in zip(
                blocks, block_eligible, block_previous_candidates
            ):
                if not eligible or not should_score(cluster_id, block):
                    continue
                texts = [_activity_text(catalog[idx]) for idx in eligible]
                for previous in previous_candidates:
                    query = _query_text(profile_text, block, previous, catalog)
                    for text in texts:
                        key = _cache_key(config.alignment_model, profile_text, block, previous, text)
                        if key in cache or key in pending_seen:
                            continue
                        pending_seen.add(key)
                        pending.append((key, query, text))

        chunks = [
            pending[start : start + config.alignment_batch_size]
            for start in range(0, len(pending), config.alignment_batch_size)
        ]
        total_chunks = len(chunks)
        total_pairs = len(pending)
        start_time = time.perf_counter()
        checkpoint_every = config.alignment_checkpoint_every

        def _apply_chunk_scores(chunk: list[tuple[str, str, str]], chunk_scores: list[float]) -> None:
            for (key, _query, _text), score in zip(chunk, chunk_scores):
                cache[key] = float(np.clip(score, 0.0, 1.0))

        def _report_progress(done_chunks: int, done_pairs: int) -> None:
            if checkpoint_every <= 0 or done_chunks % checkpoint_every != 0:
                return
            elapsed = time.perf_counter() - start_time
            rate = done_pairs / elapsed if elapsed > 0 else 0.0
            remaining_pairs = total_pairs - done_pairs
            eta = remaining_pairs / rate if rate > 0 else float("nan")
            print(
                f"Activity alignment: {done_chunks}/{total_chunks} batches, "
                f"{done_pairs}/{total_pairs} pairs, {rate:.0f} pairs/sec, "
                f"{elapsed:.1f}s elapsed, ETA {_format_duration(eta)}",
                flush=True,
            )
            if cache_path is not None:
                _save_cache(cache_path, cache)

        if config.alignment_concurrency <= 1:
            done_pairs = 0
            for done_chunks, chunk in enumerate(chunks, start=1):
                chunk_scores = _score_chunk_with_retries(
                    config.alignment_base_url,
                    config.alignment_model,
                    [(query, text) for _key, query, text in chunk],
                    timeout=config.alignment_timeout_seconds,
                    retries=config.alignment_retries,
                )
                _apply_chunk_scores(chunk, chunk_scores)
                done_pairs += len(chunk)
                _report_progress(done_chunks, done_pairs)
        else:
            done_pairs = 0
            with ThreadPoolExecutor(max_workers=config.alignment_concurrency) as executor:
                futures = {
                    executor.submit(
                        _score_chunk_with_retries,
                        config.alignment_base_url,
                        config.alignment_model,
                        [(query, text) for _key, query, text in chunk],
                        timeout=config.alignment_timeout_seconds,
                        retries=config.alignment_retries,
                    ): chunk
                    for chunk in chunks
                }
                for done_chunks, future in enumerate(as_completed(futures), start=1):
                    chunk = futures[future]
                    chunk_scores = future.result()
                    _apply_chunk_scores(chunk, chunk_scores)
                    done_pairs += len(chunk)
                    _report_progress(done_chunks, done_pairs)

        for cluster_id, profile_text in enumerate(cluster_narratives):
            for block, eligible, previous_candidates in zip(
                blocks, block_eligible, block_previous_candidates
            ):
                if not eligible or not should_score(cluster_id, block):
                    continue
                texts = [_activity_text(catalog[idx]) for idx in eligible]
                for previous in previous_candidates:
                    prev_pos = 0 if previous == START_PREVIOUS_ACTIVITY else previous + 1
                    for local_idx, activity_idx in enumerate(eligible):
                        key = _cache_key(config.alignment_model, profile_text, block, previous, texts[local_idx])
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

    cache: dict[str, float] = {}
    cache_path = Path(config.alignment_cache_path) if config.alignment_cache_path else None
    if cache_path is not None:
        cache = _load_cache(cache_path)

    try:
        pending: list[tuple[str, str, str]] = []
        pending_seen: set[str] = set()
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
                    key = _poi_cache_key(config.alignment_model, profile_text, semantic_cluster, text)
                    if key in cache or key in pending_seen:
                        continue
                    pending_seen.add(key)
                    pending.append((key, query, text))

        chunks = [
            pending[start : start + config.alignment_batch_size]
            for start in range(0, len(pending), config.alignment_batch_size)
        ]
        total_chunks = len(chunks)
        total_pairs = len(pending)
        start_time = time.perf_counter()
        checkpoint_every = config.alignment_checkpoint_every

        def _apply_chunk_scores(chunk: list[tuple[str, str, str]], chunk_scores: list[float]) -> None:
            for (key, _query, _text), score in zip(chunk, chunk_scores):
                cache[key] = float(np.clip(score, 0.0, 1.0))

        def _report_progress(done_chunks: int, done_pairs: int) -> None:
            if checkpoint_every <= 0 or done_chunks % checkpoint_every != 0:
                return
            elapsed = time.perf_counter() - start_time
            rate = done_pairs / elapsed if elapsed > 0 else 0.0
            remaining_pairs = total_pairs - done_pairs
            eta = remaining_pairs / rate if rate > 0 else float("nan")
            print(
                f"POI activity alignment: {done_chunks}/{total_chunks} batches, "
                f"{done_pairs}/{total_pairs} pairs, {rate:.0f} pairs/sec, "
                f"{elapsed:.1f}s elapsed, ETA {_format_duration(eta)}",
                flush=True,
            )
            if cache_path is not None:
                _save_cache(cache_path, cache)

        if config.alignment_concurrency <= 1:
            done_pairs = 0
            for done_chunks, chunk in enumerate(chunks, start=1):
                chunk_scores = _score_chunk_with_retries(
                    config.alignment_base_url,
                    config.alignment_model,
                    [(query, text) for _key, query, text in chunk],
                    timeout=config.alignment_timeout_seconds,
                    retries=config.alignment_retries,
                )
                _apply_chunk_scores(chunk, chunk_scores)
                done_pairs += len(chunk)
                _report_progress(done_chunks, done_pairs)
        else:
            done_pairs = 0
            with ThreadPoolExecutor(max_workers=config.alignment_concurrency) as executor:
                futures = {
                    executor.submit(
                        _score_chunk_with_retries,
                        config.alignment_base_url,
                        config.alignment_model,
                        [(query, text) for _key, query, text in chunk],
                        timeout=config.alignment_timeout_seconds,
                        retries=config.alignment_retries,
                    ): chunk
                    for chunk in chunks
                }
                for done_chunks, future in enumerate(as_completed(futures), start=1):
                    chunk = futures[future]
                    chunk_scores = future.result()
                    _apply_chunk_scores(chunk, chunk_scores)
                    done_pairs += len(chunk)
                    _report_progress(done_chunks, done_pairs)

        for cluster_id, profile_text in enumerate(cluster_narratives):
            for semantic_cluster_id, semantic_cluster in enumerate(poi_data.semantic_clusters):
                start = int(poi_data.mask_starts[semantic_cluster_id])
                end = int(poi_data.mask_starts[semantic_cluster_id + 1])
                for activity_idx in poi_data.mask_activities[start:end]:
                    activity = catalog[int(activity_idx)]
                    text = _activity_text(activity)
                    key = _poi_cache_key(config.alignment_model, profile_text, semantic_cluster, text)
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

    if cache_path is not None:
        _save_cache(cache_path, cache)
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

    cache: dict[str, float] = {}
    cache_path = Path(config.alignment_cache_path) if config.alignment_cache_path else None
    if cache_path is not None:
        cache = _load_cache(cache_path)

    try:
        pending: list[tuple[str, str, str]] = []
        pending_seen: set[str] = set()
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
                    if key in cache or key in pending_seen:
                        continue
                    pending_seen.add(key)
                    pending.append((key, query, text))

        chunks = [
            pending[start : start + config.alignment_batch_size]
            for start in range(0, len(pending), config.alignment_batch_size)
        ]
        total_chunks = len(chunks)
        total_pairs = len(pending)
        start_time = time.perf_counter()
        checkpoint_every = config.alignment_checkpoint_every

        def _apply_chunk_scores(chunk: list[tuple[str, str, str]], chunk_scores: list[float]) -> None:
            for (key, _query, _text), score in zip(chunk, chunk_scores):
                cache[key] = float(np.clip(score, 0.0, 1.0))

        def _report_progress(done_chunks: int, done_pairs: int) -> None:
            if checkpoint_every <= 0 or done_chunks % checkpoint_every != 0:
                return
            elapsed = time.perf_counter() - start_time
            rate = done_pairs / elapsed if elapsed > 0 else 0.0
            remaining_pairs = total_pairs - done_pairs
            eta = remaining_pairs / rate if rate > 0 else float("nan")
            print(
                f"POI type alignment: {done_chunks}/{total_chunks} batches, "
                f"{done_pairs}/{total_pairs} pairs, {rate:.0f} pairs/sec, "
                f"{elapsed:.1f}s elapsed, ETA {_format_duration(eta)}",
                flush=True,
            )
            if cache_path is not None:
                _save_cache(cache_path, cache)

        if config.alignment_concurrency <= 1:
            done_pairs = 0
            for done_chunks, chunk in enumerate(chunks, start=1):
                chunk_scores = _score_chunk_with_retries(
                    config.alignment_base_url,
                    config.alignment_model,
                    [(query, text) for _key, query, text in chunk],
                    timeout=config.alignment_timeout_seconds,
                    retries=config.alignment_retries,
                )
                _apply_chunk_scores(chunk, chunk_scores)
                done_pairs += len(chunk)
                _report_progress(done_chunks, done_pairs)
        else:
            done_pairs = 0
            with ThreadPoolExecutor(max_workers=config.alignment_concurrency) as executor:
                futures = {
                    executor.submit(
                        _score_chunk_with_retries,
                        config.alignment_base_url,
                        config.alignment_model,
                        [(query, text) for _key, query, text in chunk],
                        timeout=config.alignment_timeout_seconds,
                        retries=config.alignment_retries,
                    ): chunk
                    for chunk in chunks
                }
                for done_chunks, future in enumerate(as_completed(futures), start=1):
                    chunk = futures[future]
                    chunk_scores = future.result()
                    _apply_chunk_scores(chunk, chunk_scores)
                    done_pairs += len(chunk)
                    _report_progress(done_chunks, done_pairs)

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

    if cache_path is not None:
        _save_cache(cache_path, cache)
    return np.clip(scores, 0.0, 1.0), blocks, pd.DataFrame(rows)
