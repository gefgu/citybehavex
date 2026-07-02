from __future__ import annotations

import numpy as np

from citybehavex.simulation.social_graph import (
    build_knn_fallback_social_graph,
    build_profile_social_graph,
)


def _assert_valid_bounded_csr(
    starts: np.ndarray,
    neighbors: np.ndarray,
    weights: np.ndarray,
    *,
    n: int,
    k: int,
) -> None:
    assert starts.dtype == np.int64
    assert neighbors.dtype == np.int64
    assert weights.dtype == np.float64
    assert starts.shape == (n + 1,)
    assert starts[0] == 0
    assert starts[-1] == len(neighbors) == len(weights)
    assert np.all(np.diff(starts) >= 0)
    assert len(neighbors) <= n * k
    for i in range(n):
        row = neighbors[starts[i] : starts[i + 1]]
        assert len(row) <= min(k, max(0, n - 1))
        assert i not in row
        assert np.all((0 <= row) & (row < n))


def _normalized_embeddings(n: int, dim: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    embeddings = rng.normal(size=(n, dim))
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


def test_knn_fallback_empty_and_single_agent_graphs():
    for n in (0, 1):
        starts, neighbors, weights = build_knn_fallback_social_graph(n, k=20, random_state=7)
        _assert_valid_bounded_csr(starts, neighbors, weights, n=n, k=20)
        assert len(neighbors) == 0


def test_knn_fallback_handles_n_smaller_than_k():
    n, k = 4, 20
    starts, neighbors, weights = build_knn_fallback_social_graph(n, k=k, random_state=7)
    _assert_valid_bounded_csr(starts, neighbors, weights, n=n, k=k)
    assert len(neighbors) == n * (n - 1)
    assert np.all(weights == 1.0)


def test_knn_fallback_is_deterministic_and_bounded():
    first = build_knn_fallback_social_graph(100, k=20, random_state=9)
    second = build_knn_fallback_social_graph(100, k=20, random_state=9)
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
    _assert_valid_bounded_csr(*first, n=100, k=20)


def test_profile_graph_exact_knn_is_bounded_with_cosine_weights():
    embeddings = _normalized_embeddings(40, 8)
    starts, neighbors, weights = build_profile_social_graph(
        embeddings,
        k=10,
        random_state=3,
        exact_threshold=100,
    )
    _assert_valid_bounded_csr(starts, neighbors, weights, n=40, k=10)
    assert np.all((-1.0 <= weights) & (weights <= 1.0))


def test_profile_graph_cluster_sampling_is_deterministic_and_bounded():
    embeddings = _normalized_embeddings(60, 12)
    first = build_profile_social_graph(
        embeddings,
        k=7,
        random_state=5,
        exact_threshold=1,
    )
    second = build_profile_social_graph(
        embeddings,
        k=7,
        random_state=5,
        exact_threshold=1,
    )
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
    _assert_valid_bounded_csr(*first, n=60, k=7)
    assert np.all((-1.0 <= first[2]) & (first[2] <= 1.0))
