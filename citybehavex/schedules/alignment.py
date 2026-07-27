from __future__ import annotations

import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import requests

from citybehavex.embedding import diary_to_prose
from citybehavex.llm_diaries import Diary
from citybehavex.schedules.config import ScheduleConfig
from citybehavex.utils.alignment import (
    alignment_cache_key,
    post_rerank_scores,
)
from citybehavex.utils.cache import load_score_cache, save_score_cache


def _rerank_chunk_with_retries(
    base_url: str,
    model: str | None,
    query: str,
    texts: Sequence[str],
    *,
    timeout: float,
    retries: int,
) -> list[float]:
    return post_rerank_scores(
        base_url,
        model,
        query,
        texts,
        timeout=timeout,
        retries=retries,
        requests_module=requests,
    )


def score_alignment_matrix(
    profile_texts: Sequence[str],
    diaries: Sequence[Diary],
    config: ScheduleConfig,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> Optional[np.ndarray]:
    """Return learned macro-schedule alignment scores, or ``None`` on failure.

    The expected inference server is a TEI reranker/sequence-classification
    endpoint accepting ``/rerank`` requests with one profile query and many diary
    texts. Scores are clipped to [0, 1] before ddCRP consumes them.

    Diary chunks are scored with up to ``config.alignment_concurrency`` requests
    in flight at once (one request per profile row per diary chunk), mirroring
    the batching used by the activity/ownership/coherence alignment scorers.
    """
    if not profile_texts or not diaries or not config.alignment_base_url:
        return None

    diary_texts = [diary_to_prose(d) for d in diaries]
    cache: dict[str, float] = {}
    cache_path = Path(config.alignment_cache_path) if config.alignment_cache_path else None
    if cache_path is not None:
        cache = load_score_cache(cache_path)

    matrix = np.empty((len(profile_texts), len(diary_texts)), dtype=np.float64)
    start_time = time.perf_counter()

    row_keys = [
        [alignment_cache_key(config.alignment_model, profile_text, text) for text in diary_texts]
        for profile_text in profile_texts
    ]
    work_items: list[tuple[int, list[int]]] = []
    for row, keys in enumerate(row_keys):
        missing = [idx for idx, key in enumerate(keys) if key not in cache]
        for start in range(0, len(missing), config.alignment_batch_size):
            work_items.append((row, missing[start : start + config.alignment_batch_size]))

    total_rows = len(profile_texts)
    rows_remaining_chunks = Counter(row for row, _ in work_items)
    done_rows = total_rows - len(rows_remaining_chunks)
    checkpoint_every = config.alignment_checkpoint_every

    def _run_item(row: int, chunk_idx: list[int]) -> tuple[int, list[int], list[float]]:
        scores = _rerank_chunk_with_retries(
            config.alignment_base_url,
            config.alignment_model,
            profile_texts[row],
            [diary_texts[i] for i in chunk_idx],
            timeout=config.alignment_timeout_seconds,
            retries=config.alignment_retries,
        )
        return row, chunk_idx, scores

    def _apply(row: int, chunk_idx: list[int], scores: list[float]) -> None:
        nonlocal done_rows
        for idx, score in zip(chunk_idx, scores):
            cache[row_keys[row][idx]] = float(np.clip(score, 0.0, 1.0))
        rows_remaining_chunks[row] -= 1
        if rows_remaining_chunks[row] > 0:
            return
        done_rows += 1
        if progress_callback is not None:
            progress_callback(done_rows, total_rows, time.perf_counter() - start_time)
        if cache_path is not None and checkpoint_every > 0 and done_rows % checkpoint_every == 0:
            save_score_cache(cache_path, cache)

    try:
        with ThreadPoolExecutor(max_workers=max(config.alignment_concurrency, 1)) as executor:
            futures = {
                executor.submit(_run_item, row, chunk_idx): (row, chunk_idx)
                for row, chunk_idx in work_items
            }
            for future in as_completed(futures):
                row, chunk_idx, scores = future.result()
                _apply(row, chunk_idx, scores)
    except Exception:  # noqa: BLE001 - callers intentionally fall back.
        if cache_path is not None:
            save_score_cache(cache_path, cache)
        return None

    if progress_callback is not None and done_rows < total_rows:
        progress_callback(total_rows, total_rows, time.perf_counter() - start_time)

    for row, keys in enumerate(row_keys):
        matrix[row] = [cache[key] for key in keys]

    if cache_path is not None:
        save_score_cache(cache_path, cache)
    return np.clip(matrix, 0.0, 1.0)
