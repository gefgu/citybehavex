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

COHERENCE_CANDIDATE_TEXT = "demographically coherent and valid synthetic agent profile"


def _query_text(profile_text: str, city_profile: str | None) -> str:
    city_context = f"\nCity context: {city_profile}" if city_profile else ""
    return (
        f"{profile_text}{city_context}\n"
        "Score whether this synthetic agent profile is demographically coherent "
        "and plausible. Use 0 for impossible or highly inconsistent profiles and "
        "1 for fully coherent profiles."
    )


def _cache_key(
    model: str | None,
    profile_text: str,
    city_profile: str | None,
    candidate_text: str,
) -> str:
    raw = f"{model or ''}\x00{profile_text}\x00{city_profile or ''}\x00{candidate_text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


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
    cache: dict[str, float] = {}
    cache_path = (
        Path(config.coherence_alignment_cache_path)
        if config.coherence_alignment_cache_path
        else None
    )
    if cache_path is not None:
        cache = _load_cache(cache_path)

    try:
        pending: list[tuple[str, str, str]] = []
        pending_seen: set[str] = set()
        for profile_text in cluster_narratives:
            query = _query_text(profile_text, city_profile)
            key = _cache_key(
                config.coherence_alignment_model,
                profile_text,
                city_profile,
                COHERENCE_CANDIDATE_TEXT,
            )
            if key in cache or key in pending_seen:
                continue
            pending_seen.add(key)
            pending.append((key, query, COHERENCE_CANDIDATE_TEXT))

        chunks = [
            pending[start : start + config.coherence_alignment_batch_size]
            for start in range(0, len(pending), config.coherence_alignment_batch_size)
        ]
        total_chunks = len(chunks)
        total_pairs = len(pending)
        start_time = time.perf_counter()
        checkpoint_every = config.coherence_alignment_checkpoint_every

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
                f"Profile coherence alignment: {done_chunks}/{total_chunks} batches, "
                f"{done_pairs}/{total_pairs} pairs, {rate:.0f} pairs/sec, "
                f"{elapsed:.1f}s elapsed, ETA {_format_duration(eta)}",
                flush=True,
            )
            if cache_path is not None:
                _save_cache(cache_path, cache)

        if config.coherence_alignment_concurrency <= 1:
            done_pairs = 0
            for done_chunks, chunk in enumerate(chunks, start=1):
                chunk_scores = _score_chunk_with_retries(
                    config.coherence_alignment_base_url,
                    config.coherence_alignment_model,
                    [(query, text) for _key, query, text in chunk],
                    timeout=config.coherence_alignment_timeout_seconds,
                    retries=config.coherence_alignment_retries,
                )
                _apply_chunk_scores(chunk, chunk_scores)
                done_pairs += len(chunk)
                _report_progress(done_chunks, done_pairs)
        else:
            done_pairs = 0
            with ThreadPoolExecutor(max_workers=config.coherence_alignment_concurrency) as executor:
                futures = {
                    executor.submit(
                        _score_chunk_with_retries,
                        config.coherence_alignment_base_url,
                        config.coherence_alignment_model,
                        [(query, text) for _key, query, text in chunk],
                        timeout=config.coherence_alignment_timeout_seconds,
                        retries=config.coherence_alignment_retries,
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
            key = _cache_key(
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

    if cache_path is not None:
        _save_cache(cache_path, cache)
    return np.clip(scores, 0.0, 1.0), pd.DataFrame(rows)


def expand_coherence_scores(scores: np.ndarray, clusters: ProfileClusters) -> np.ndarray:
    if len(clusters.labels) == 0:
        return np.empty(0, dtype=np.float64)
    if int(clusters.labels.max()) >= scores.shape[0]:
        raise ValueError("cluster labels reference a missing coherence score row")
    return np.asarray(scores, dtype=np.float64)[clusters.labels]
