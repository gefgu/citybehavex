from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from citybehavex.activities.alignment import (
    ProfileClusters,
    _format_duration,
    _load_cache,
    _save_cache,
    _score_chunk_with_retries,
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


def _query_text(profile_text: str, city_profile: str | None) -> str:
    city_context = f"\nCity context: {city_profile}" if city_profile else ""
    return (
        f"{profile_text}{city_context}\n"
        "Score how likely this person is to have the listed transport option. "
        "Use 0 for very unlikely and 1 for very likely."
    )


def _cache_key(
    model: str | None,
    profile_text: str,
    city_profile: str | None,
    vehicle: str,
    candidate_text: str,
) -> str:
    raw = f"{model or ''}\x00{profile_text}\x00{city_profile or ''}\x00{vehicle}\x00{candidate_text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


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

    cache: dict[str, float] = {}
    cache_path = (
        Path(config.ownership_alignment_cache_path)
        if config.ownership_alignment_cache_path
        else None
    )
    if cache_path is not None:
        cache = _load_cache(cache_path)

    try:
        pending: list[tuple[str, str, str]] = []
        pending_seen: set[str] = set()
        for cluster_id, profile_text in enumerate(cluster_narratives):
            query = _query_text(profile_text, city_profile)
            for vehicle, candidate_text in VEHICLE_CANDIDATES:
                key = _cache_key(
                    config.ownership_alignment_model,
                    profile_text,
                    city_profile,
                    vehicle,
                    candidate_text,
                )
                if key in cache or key in pending_seen:
                    continue
                pending_seen.add(key)
                pending.append((key, query, candidate_text))

        chunks = [
            pending[start : start + config.ownership_alignment_batch_size]
            for start in range(0, len(pending), config.ownership_alignment_batch_size)
        ]
        total_chunks = len(chunks)
        total_pairs = len(pending)
        start_time = time.perf_counter()
        checkpoint_every = config.ownership_alignment_checkpoint_every

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
                f"Vehicle ownership alignment: {done_chunks}/{total_chunks} batches, "
                f"{done_pairs}/{total_pairs} pairs, {rate:.0f} pairs/sec, "
                f"{elapsed:.1f}s elapsed, ETA {_format_duration(eta)}",
                flush=True,
            )
            if cache_path is not None:
                _save_cache(cache_path, cache)

        if config.ownership_alignment_concurrency <= 1:
            done_pairs = 0
            for done_chunks, chunk in enumerate(chunks, start=1):
                chunk_scores = _score_chunk_with_retries(
                    config.ownership_alignment_base_url,
                    config.ownership_alignment_model,
                    [(query, text) for _key, query, text in chunk],
                    timeout=config.ownership_alignment_timeout_seconds,
                    retries=config.ownership_alignment_retries,
                )
                _apply_chunk_scores(chunk, chunk_scores)
                done_pairs += len(chunk)
                _report_progress(done_chunks, done_pairs)
        else:
            done_pairs = 0
            with ThreadPoolExecutor(max_workers=config.ownership_alignment_concurrency) as executor:
                futures = {
                    executor.submit(
                        _score_chunk_with_retries,
                        config.ownership_alignment_base_url,
                        config.ownership_alignment_model,
                        [(query, text) for _key, query, text in chunk],
                        timeout=config.ownership_alignment_timeout_seconds,
                        retries=config.ownership_alignment_retries,
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

    if cache_path is not None:
        _save_cache(cache_path, cache)
    return np.clip(scores, 0.0, 1.0), pd.DataFrame(rows)


def expand_vehicle_scores(scores: np.ndarray, clusters: ProfileClusters) -> np.ndarray:
    if len(clusters.labels) == 0:
        return np.empty((0, scores.shape[1]), dtype=np.float64)
    if int(clusters.labels.max()) >= scores.shape[0]:
        raise ValueError("cluster labels reference a missing vehicle score row")
    return np.asarray(scores, dtype=np.float64)[clusters.labels]
