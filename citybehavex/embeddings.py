"""Diary embeddings for the ddCRP schedule selector.

Serves one vector per whole LLM diary via an OpenAI-compatible ``/v1/embeddings``
endpoint (``nomic-embed-text-v2-moe`` by default). Behaviour, in order:

1. Load any cached vectors from ``config.embedding.cache_path``.
2. For cache misses: if a server is reachable at ``base_url`` use it; otherwise, if
   ``auto_launch`` is set, spawn a local vLLM ``--task embed`` server on demand and
   shut it down afterwards.
3. Persist freshly computed vectors back to the cache.

If embeddings are disabled or every backend fails, :func:`embed_diaries` returns
``None`` and the caller falls back to identity similarity (exact preferential
return, no semantic smoothing).
"""

from __future__ import annotations

import contextlib
import hashlib
import subprocess
import time
from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np
import requests

from .config import EmbeddingConfig
from .llm_diaries import Diary


def diary_to_text(diary: Diary) -> str:
    """Canonical, deterministic text serialization of a diary's episodes.

    Example: ``"00:00-07:00 HOME | 07:00-09:00 WORK | 09:00-24:00 HOME"``.
    Episodes are already validated to be ordered and gap-free, so the natural
    order is stable and a good basis for a content-addressed cache key.
    """
    parts = [
        f"{ep.start}-{ep.end if ep.end is not None else f'+{ep.duration_minutes}m'} {ep.purpose}"
        for ep in diary.episodes
    ]
    return " | ".join(parts)


def _cache_key(text: str, model: str, dim: int) -> str:
    digest = hashlib.sha256(f"{model}\x00{dim}\x00{text}".encode("utf-8")).hexdigest()
    return digest


def _load_cache(cache_path: Path) -> dict[str, np.ndarray]:
    if not cache_path.exists():
        return {}
    try:
        data = np.load(cache_path, allow_pickle=False)
        keys = data["keys"]
        vectors = data["vectors"]
    except Exception:  # noqa: BLE001 - a corrupt cache should not be fatal.
        return {}
    return {str(k): vectors[i] for i, k in enumerate(keys)}


def _save_cache(cache_path: Path, cache: dict[str, np.ndarray]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    keys = np.array(list(cache.keys()))
    vectors = (
        np.stack([cache[k] for k in cache]) if cache else np.empty((0, 0), dtype=np.float32)
    )
    np.savez(cache_path, keys=keys, vectors=vectors)


def _server_reachable(base_url: str, timeout: float) -> bool:
    for path in ("/health", "/v1/models"):
        try:
            resp = requests.get(base_url.rstrip("/") + path, timeout=timeout)
            if resp.ok:
                return True
        except Exception:  # noqa: BLE001 - probing; failure just means "not ready".
            continue
    return False


@contextlib.contextmanager
def _vllm_server(config: EmbeddingConfig) -> Iterator[str]:
    """Spawn a local vLLM embedding server, yield its base_url, then shut it down."""
    port = config.vllm_port
    base_url = f"http://127.0.0.1:{port}"
    log_dir = Path(config.cache_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "vllm_embed.log"

    cmd = [
        "vllm",
        "serve",
        config.model,
        "--task",
        "embed",
        "--trust-remote-code",
        "--port",
        str(port),
        *config.vllm_extra_args,
    ]
    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + config.vllm_startup_timeout_seconds
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"vllm exited early (code {proc.returncode}); see {log_path}"
                    )
                if _server_reachable(base_url, timeout=2.0):
                    break
                time.sleep(2.0)
            else:
                raise TimeoutError(
                    f"vllm did not become ready within "
                    f"{config.vllm_startup_timeout_seconds:.0f}s; see {log_path}"
                )
            yield base_url
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)


def _post_embeddings(
    base_url: str,
    model: str,
    texts: Sequence[str],
    *,
    api_key: Optional[str],
    timeout: float,
) -> np.ndarray:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = requests.post(
        base_url.rstrip("/") + "/v1/embeddings",
        headers=headers,
        json={"model": model, "input": list(texts), "encoding_format": "float"},
        timeout=timeout,
    )
    resp.raise_for_status()
    payload = resp.json()
    rows = sorted(payload["data"], key=lambda d: d.get("index", 0))
    return np.asarray([row["embedding"] for row in rows], dtype=np.float32)


def _finalize(vectors: np.ndarray, dim: int) -> np.ndarray:
    """Matryoshka-truncate to ``dim`` and L2-normalize rows."""
    if vectors.shape[1] > dim:
        vectors = vectors[:, :dim]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (vectors / norms).astype(np.float32)


def embed_diaries(
    diaries: Sequence[Diary], config: EmbeddingConfig
) -> Optional[np.ndarray]:
    """Return an ``[K, dim]`` L2-normalized embedding matrix, or ``None`` on failure.

    Cache-first; computes only the misses, optionally auto-launching vLLM. A return
    value of ``None`` signals the caller to fall back to identity similarity.
    """
    if not config.enabled or not diaries:
        return None

    texts = [config.task_prefix + diary_to_text(d) for d in diaries]
    keys = [_cache_key(t, config.model, config.dimensions) for t in texts]

    cache_path = Path(config.resolved_cache_path())
    cache = _load_cache(cache_path)

    missing_idx = [i for i, k in enumerate(keys) if k not in cache]
    if missing_idx:
        missing_texts = [texts[i] for i in missing_idx]
        computed: Optional[np.ndarray] = None
        try:
            if config.base_url and _server_reachable(config.base_url, timeout=5.0):
                computed = _post_embeddings(
                    config.base_url,
                    config.model,
                    missing_texts,
                    api_key=config.api_key,
                    timeout=config.timeout_seconds,
                )
            elif config.auto_launch:
                with _vllm_server(config) as base_url:
                    computed = _post_embeddings(
                        base_url,
                        config.model,
                        missing_texts,
                        api_key=config.api_key,
                        timeout=config.timeout_seconds,
                    )
        except Exception:  # noqa: BLE001 - any failure falls back to identity sim.
            return None
        if computed is None:
            return None
        computed = _finalize(computed, config.dimensions)
        for slot, vec in zip(missing_idx, computed):
            cache[keys[slot]] = vec
        _save_cache(cache_path, cache)

    return np.stack([cache[k] for k in keys]).astype(np.float32)


def cosine_sim_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Cosine-similarity matrix for already-L2-normalized row vectors."""
    return np.clip(embeddings @ embeddings.T, -1.0, 1.0)
