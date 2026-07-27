from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np


def _atomic_npz_save(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as fh:
            np.savez(fh, **arrays)
            tmp_name = fh.name
        os.replace(tmp_name, path)
    except BaseException:
        if tmp_name is not None:
            Path(tmp_name).unlink(missing_ok=True)
        raise


def load_score_cache(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    try:
        data = np.load(path, allow_pickle=False)
        keys = data["keys"]
        scores = data["scores"]
    except Exception:  # noqa: BLE001 - corrupt cache should not break a run.
        return {}
    return {str(k): float(scores[i]) for i, k in enumerate(keys)}


def save_score_cache(path: Path, cache: dict[str, float]) -> None:
    keys = np.array(list(cache.keys()))
    scores = np.array([cache[k] for k in cache], dtype=np.float32)
    _atomic_npz_save(path, keys=keys, scores=scores)


def load_vector_cache(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    try:
        data = np.load(path, allow_pickle=False)
        keys = data["keys"]
        vectors = data["vectors"]
    except Exception:  # noqa: BLE001 - corrupt cache should not break a run.
        return {}
    return {str(k): vectors[i] for i, k in enumerate(keys)}


def save_vector_cache(path: Path, cache: dict[str, np.ndarray]) -> None:
    keys = np.array(list(cache.keys()))
    vectors = (
        np.stack([cache[k] for k in cache]) if cache else np.empty((0, 0), dtype=np.float32)
    )
    _atomic_npz_save(path, keys=keys, vectors=vectors)


def write_text_atomic(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding=encoding,
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as fh:
            fh.write(text)
            tmp_name = fh.name
        os.replace(tmp_name, path)
    except BaseException:
        if tmp_name is not None:
            Path(tmp_name).unlink(missing_ok=True)
        raise
