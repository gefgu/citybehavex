from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Optional, Sequence

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
    tmp_path = path.parent / (path.name + ".tmp")
    try:
        with open(tmp_path, "wb") as fh:
            np.savez(fh, keys=keys, scores=scores)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
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


def score_alignment_matrix(
    profile_texts: Sequence[str],
    diaries: Sequence[Diary],
    config: ScheduleConfig,
) -> Optional[np.ndarray]:
    """Return learned macro-schedule alignment scores, or ``None`` on failure.

    The expected inference server is a TEI reranker/sequence-classification
    endpoint accepting ``/rerank`` requests with one profile query and many diary
    texts. Scores are clipped to [0, 1] before ddCRP consumes them.
    """
    if not profile_texts or not diaries or not config.alignment_base_url:
        return None

    diary_texts = [diary_to_prose(d) for d in diaries]
    cache: dict[str, float] = {}
    cache_path = Path(config.alignment_cache_path) if config.alignment_cache_path else None
    if cache_path is not None:
        cache = _load_cache(cache_path)

    matrix = np.empty((len(profile_texts), len(diary_texts)), dtype=np.float64)
    try:
        for row, profile_text in enumerate(profile_texts):
            keys = [_cache_key(config.alignment_model, profile_text, text) for text in diary_texts]
            missing = [idx for idx, key in enumerate(keys) if key not in cache]
            for start in range(0, len(missing), config.alignment_batch_size):
                chunk_idx = missing[start : start + config.alignment_batch_size]
                scores = _post_rerank(
                    config.alignment_base_url,
                    config.alignment_model,
                    profile_text,
                    [diary_texts[i] for i in chunk_idx],
                    timeout=config.alignment_timeout_seconds,
                )
                if scores is None:
                    if cache_path is not None:
                        _save_cache(cache_path, cache)
                    return None
                for idx, score in zip(chunk_idx, scores):
                    cache[keys[idx]] = float(np.clip(score, 0.0, 1.0))
            matrix[row] = [cache[key] for key in keys]
            checkpoint_every = config.alignment_checkpoint_every
            if cache_path is not None and checkpoint_every > 0 and (row + 1) % checkpoint_every == 0:
                _save_cache(cache_path, cache)
    except Exception:  # noqa: BLE001 - callers intentionally fall back.
        if cache_path is not None:
            _save_cache(cache_path, cache)
        return None

    if cache_path is not None:
        _save_cache(cache_path, cache)
    return np.clip(matrix, 0.0, 1.0)
