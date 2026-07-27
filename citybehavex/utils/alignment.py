from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional, Protocol, Sequence

import numpy as np
import requests

from citybehavex.utils.cache import load_score_cache, save_score_cache
from citybehavex.utils.http import post_json_with_retries
from citybehavex.utils.progress import ProgressReporter

AlignmentPair = tuple[str, str, str]


class ScoreChunkFn(Protocol):
    def __call__(
        self,
        base_url: str,
        model: str | None,
        pairs: Sequence[tuple[str, str]],
        *,
        timeout: float,
        retries: int,
    ) -> list[float]:
        ...


def alignment_cache_key(*parts: str | None) -> str:
    raw = "\x00".join(part or "" for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def alignment_query_text(profile_text: str, city_profile: str | None, instruction: str) -> str:
    city_context = f"\nCity context: {city_profile}" if city_profile else ""
    return f"{profile_text}{city_context}\n{instruction}"


def extract_rerank_scores(payload: Any, expected: int) -> Optional[list[float]]:
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


def post_rerank_scores(
    base_url: str,
    model: str | None,
    query: str,
    texts: Sequence[str],
    *,
    timeout: float,
    retries: int = 2,
    requests_module=requests,
) -> list[float]:
    payload: dict[str, Any] = {
        "query": query,
        "texts": list(texts),
        "raw_scores": False,
        "truncate": True,
    }
    if model:
        payload["model"] = model
    response = post_json_with_retries(
        base_url.rstrip("/") + "/rerank",
        headers={"Content-Type": "application/json"},
        payload=payload,
        timeout=timeout,
        retries=retries,
        requests_module=requests_module,
    )
    scores = extract_rerank_scores(response, len(texts))
    if scores is None:
        raise ValueError("reranker response could not be parsed")
    return scores


def post_pair_scores(
    base_url: str,
    model: str | None,
    pairs: Sequence[tuple[str, str]],
    *,
    timeout: float,
    retries: int = 2,
    requests_module=requests,
) -> list[float]:
    payload: dict[str, Any] = {
        "pairs": [[query, text] for query, text in pairs],
        "raw_scores": False,
        "truncate": True,
    }
    if model:
        payload["model"] = model
    response = post_json_with_retries(
        base_url.rstrip("/") + "/score_pairs",
        headers={"Content-Type": "application/json"},
        payload=payload,
        timeout=timeout,
        retries=retries,
        requests_module=requests_module,
    )
    scores = extract_rerank_scores(response, len(pairs))
    if scores is None:
        raise ValueError("reranker response could not be parsed")
    return scores


def score_chunk_with_retries(
    base_url: str,
    model: str | None,
    pairs: Sequence[tuple[str, str]],
    *,
    timeout: float,
    retries: int,
    requests_module=requests,
) -> list[float]:
    return post_pair_scores(
        base_url,
        model,
        pairs,
        timeout=timeout,
        retries=retries,
        requests_module=requests_module,
    )


def score_cached_alignment_pairs(
    pairs: Sequence[AlignmentPair],
    *,
    base_url: str,
    model: str | None,
    batch_size: int,
    cache_path: str | None,
    concurrency: int,
    timeout_seconds: float,
    retries: int,
    checkpoint_every: int,
    progress_label: str,
    score_chunk: ScoreChunkFn = score_chunk_with_retries,
) -> dict[str, float]:
    cache_file = Path(cache_path) if cache_path else None
    cache: dict[str, float] = load_score_cache(cache_file) if cache_file is not None else {}

    try:
        pending: list[AlignmentPair] = []
        pending_seen: set[str] = set()
        for key, query, text in pairs:
            if key in cache or key in pending_seen:
                continue
            pending_seen.add(key)
            pending.append((key, query, text))

        chunks = [
            pending[start : start + batch_size]
            for start in range(0, len(pending), batch_size)
        ]
        progress = ProgressReporter(
            progress_label,
            len(pending),
            "pairs",
            rate_precision=0,
            checkpoint_every=checkpoint_every,
            emit=lambda message: print(message, flush=True),
            on_report=(lambda: save_score_cache(cache_file, cache))
            if cache_file is not None
            else None,
        )

        def apply_chunk_scores(chunk: list[AlignmentPair], chunk_scores: list[float]) -> None:
            for (key, _query, _text), score in zip(chunk, chunk_scores):
                cache[key] = float(np.clip(score, 0.0, 1.0))

        done_pairs = 0
        with ThreadPoolExecutor(max_workers=max(concurrency, 1)) as executor:
            futures = {
                executor.submit(
                    score_chunk,
                    base_url,
                    model,
                    [(query, text) for _key, query, text in chunk],
                    timeout=timeout_seconds,
                    retries=retries,
                ): chunk
                for chunk in chunks
            }
            for done_chunks, future in enumerate(as_completed(futures), start=1):
                chunk = futures[future]
                apply_chunk_scores(chunk, future.result())
                done_pairs += len(chunk)
                progress.report(
                    done_pairs,
                    checkpoint_count=done_chunks,
                    done_batches=done_chunks,
                    total_batches=len(chunks),
                )
    finally:
        if cache_file is not None:
            save_score_cache(cache_file, cache)

    return cache
