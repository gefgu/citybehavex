"""Text embeddings for the ddCRP schedule selector and profile similarity graph.

All text types (diary schedules, agent profile narratives, activity descriptions)
are embedded by a single shared model via an OpenAI-compatible ``/v1/embeddings``
endpoint (``nomic-embed-text-v2-moe`` by default). The shared cache and endpoint
keep the embedding space consistent so cross-cosine comparisons are meaningful.

Behaviour, in order:
1. Load any cached vectors from ``config.embedding.cache_path``.
2. For cache misses: if a server is reachable at ``base_url`` use it; otherwise, if
   ``auto_launch`` is set, spawn a local vLLM ``--task embed`` server on demand and
   shut it down afterwards.
3. Persist freshly computed vectors back to the cache.

If embeddings are disabled or every backend fails, :func:`embed_texts` returns
``None`` and callers fall back to identity / uniform similarity.
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

from citybehavex.embedding.config import EmbeddingConfig
from citybehavex.llm_diaries import Diary


def diary_to_text(diary: Diary) -> str:
    """Canonical, deterministic text serialization of a diary's episodes.

    Example: ``"00:00-07:00 HOME | 07:00-09:00 WORK | 09:00-24:00 HOME"``.
    Episodes are already validated to be ordered and gap-free, so the natural
    order is stable and a good basis for a content-addressed cache key.
    """
    parts = [f"{ep.start}-{ep.end} {ep.purpose}" for ep in diary.episodes]
    return " | ".join(parts)


def diary_to_prose(diary: Diary) -> str:
    """Natural-language prose description of a diary for richer cross-embedding.

    Profiles are described as prose; matching diaries as prose too makes
    profile↔schedule cosine more discriminative than raw time-codes.
    """
    home_minutes = sum(
        ep.end_minutes - ep.start_minutes
        for ep in diary.episodes
        if ep.purpose == "HOME"
    )
    away_episodes = [ep for ep in diary.episodes if ep.purpose != "HOME"]
    away_parts = [
        f"{ep.purpose.lower()} from {ep.start} to {ep.end}"
        for ep in away_episodes
    ]
    home_hours = home_minutes // 60
    if away_parts:
        return f"Spends {home_hours} hours at home; {', '.join(away_parts)}."
    return f"Stays at home the entire day ({home_hours} hours)."


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
        config.vllm_executable,
        "serve",
        config.model,
        # vLLM >=0.11 renamed the old `--task embed` flag to `--runner`; "pooling"
        # auto-resolves to `--convert embed` for embedding-native models like
        # nomic-embed-text. `--task embed` is a hard CLI error on newer vLLM.
        "--runner",
        "pooling",
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


def embed_texts(
    texts: Sequence[str], config: EmbeddingConfig
) -> Optional[np.ndarray]:
    """Return an ``[N, dim]`` L2-normalized embedding matrix for ``texts``, or ``None``.

    Cache-first: only missing texts are sent to the embedding server. Vectors are
    keyed by SHA256(model, dim, text) so the shared cache is safe across call sites
    (diaries, profiles, activities all share one cache file).

    Returns ``None`` when embeddings are disabled or every backend fails; callers
    should fall back to identity / uniform similarity.
    """
    if not config.enabled or not texts:
        return None

    prefixed = [config.task_prefix + t for t in texts]
    keys = [_cache_key(t, config.model, config.dimensions) for t in prefixed]

    cache_path = Path(config.resolved_cache_path())
    cache = _load_cache(cache_path)

    missing_idx = [i for i, k in enumerate(keys) if k not in cache]
    if missing_idx:
        missing_texts = [prefixed[i] for i in missing_idx]
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
            else:
                print(
                    "Embedding backend unavailable: no reachable base_url "
                    f"({config.base_url!r}) and auto_launch is disabled.",
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001 - any failure falls back to caller default.
            print(f"Embedding backend failed: {exc}", flush=True)
            return None
        if computed is None:
            return None
        computed = _finalize(computed, config.dimensions)
        for slot, vec in zip(missing_idx, computed):
            cache[keys[slot]] = vec
        _save_cache(cache_path, cache)

    return np.stack([cache[k] for k in keys]).astype(np.float32)


def embed_diaries(
    diaries: Sequence[Diary], config: EmbeddingConfig
) -> Optional[np.ndarray]:
    """Embed diary schedules as prose for cross-modal similarity with profiles."""
    return embed_texts([diary_to_prose(d) for d in diaries], config)


def embed_profiles(
    narratives: Sequence[str], config: EmbeddingConfig
) -> Optional[np.ndarray]:
    """Embed agent profile narratives (output of ``profile_to_narrative``)."""
    return embed_texts(list(narratives), config)


def cosine_sim_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Cosine-similarity matrix for already-L2-normalized row vectors."""
    return np.clip(embeddings @ embeddings.T, -1.0, 1.0)


def cross_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cross-cosine similarity: ``[M, N]`` matrix, ``a`` is ``[M,d]``, ``b`` is ``[N,d]``."""
    return np.clip(a @ b.T, -1.0, 1.0)
