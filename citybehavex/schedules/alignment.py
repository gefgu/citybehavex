from __future__ import annotations

import hashlib
import os
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time
from typing import Any, Callable, Optional, Sequence

import numpy as np
import requests

from citybehavex.embedding import diary_to_prose
from citybehavex.llm_diaries import Diary
from citybehavex.schedules.config import ScheduleConfig


def _cache_key(model: str | None, profile_text: str, diary_text: str) -> str:
    raw = f"{model or ''}\x00{profile_text}\x00{diary_text}"
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
    across runs -- only entries missing from it get re-sent to the reranker."""
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = np.array(list(cache.keys()))
    scores = np.array([cache[k] for k in cache], dtype=np.float32)
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as fh:
            np.savez(fh, keys=keys, scores=scores)
            tmp_name = fh.name
        os.replace(tmp_name, path)
    except BaseException:
        if tmp_name is not None:
            Path(tmp_name).unlink(missing_ok=True)
        raise


def _extract_scores(payload: Any, expected: int) -> Optional[list[float]]:
    """Parse common TEI rerank/sequence-classification response shapes."""
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


def _rerank_chunk_with_retries(
    base_url: str,
    model: str | None,
    query: str,
    texts: Sequence[str],
    *,
    timeout: float,
    retries: int,
) -> list[float]:
    """Score one profile row's diary chunk, retrying transient failures. Raises
    (rather than returning ``None``) once retries are exhausted, so callers'
    existing broad ``except Exception: return None`` fallback still applies."""
    last_error: Exception | None = None
    for _attempt in range(max(1, retries)):
        try:
            scores = _post_rerank(base_url, model, query, texts, timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - retry, raise on exhaustion.
            last_error = exc
            continue
        if scores is None:
            last_error = ValueError("reranker response could not be parsed")
            continue
        return scores
    raise RuntimeError(f"failed to score a batch of {len(texts)} diaries") from last_error


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
        cache = _load_cache(cache_path)

    matrix = np.empty((len(profile_texts), len(diary_texts)), dtype=np.float64)
    start_time = time.perf_counter()

    row_keys = [
        [_cache_key(config.alignment_model, profile_text, text) for text in diary_texts]
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
            _save_cache(cache_path, cache)

    try:
        if config.alignment_concurrency <= 1:
            for row, chunk_idx in work_items:
                _, _, scores = _run_item(row, chunk_idx)
                _apply(row, chunk_idx, scores)
        else:
            with ThreadPoolExecutor(max_workers=config.alignment_concurrency) as executor:
                futures = {
                    executor.submit(_run_item, row, chunk_idx): (row, chunk_idx)
                    for row, chunk_idx in work_items
                }
                for future in as_completed(futures):
                    row, chunk_idx, scores = future.result()
                    _apply(row, chunk_idx, scores)
    except Exception:  # noqa: BLE001 - callers intentionally fall back.
        if cache_path is not None:
            _save_cache(cache_path, cache)
        return None

    if progress_callback is not None and done_rows < total_rows:
        progress_callback(total_rows, total_rows, time.perf_counter() - start_time)

    for row, keys in enumerate(row_keys):
        matrix[row] = [cache[key] for key in keys]

    if cache_path is not None:
        _save_cache(cache_path, cache)
    return np.clip(matrix, 0.0, 1.0)
