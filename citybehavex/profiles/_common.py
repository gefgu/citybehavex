from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from citybehavex.activities.alignment import _load_cache, _save_cache
from citybehavex.utils import ProgressReporter

AlignmentPair = tuple[str, str, str]


class ScoreChunkFn(Protocol):
    def __call__(
        self,
        base_url: str,
        model: str | None,
        pairs: list[tuple[str, str]],
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
    score_chunk: ScoreChunkFn,
) -> dict[str, float]:
    cache_file = Path(cache_path) if cache_path else None
    cache: dict[str, float] = _load_cache(cache_file) if cache_file is not None else {}

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
        total_chunks = len(chunks)
        progress = ProgressReporter(
            progress_label,
            len(pending),
            "pairs",
            rate_precision=0,
            checkpoint_every=checkpoint_every,
            emit=lambda message: print(message, flush=True),
            on_report=(lambda: _save_cache(cache_file, cache)) if cache_file is not None else None,
        )

        def apply_chunk_scores(chunk: list[AlignmentPair], chunk_scores: list[float]) -> None:
            for (key, _query, _text), score in zip(chunk, chunk_scores):
                cache[key] = float(np.clip(score, 0.0, 1.0))

        if concurrency <= 1:
            done_pairs = 0
            for done_chunks, chunk in enumerate(chunks, start=1):
                chunk_scores = score_chunk(
                    base_url,
                    model,
                    [(query, text) for _key, query, text in chunk],
                    timeout=timeout_seconds,
                    retries=retries,
                )
                apply_chunk_scores(chunk, chunk_scores)
                done_pairs += len(chunk)
                progress.report(
                    done_pairs,
                    checkpoint_count=done_chunks,
                    done_batches=done_chunks,
                    total_batches=total_chunks,
                )
        else:
            done_pairs = 0
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
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
                        total_batches=total_chunks,
                    )
    finally:
        if cache_file is not None:
            _save_cache(cache_file, cache)

    return cache
