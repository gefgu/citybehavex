from __future__ import annotations

import h3
import numpy as np

from citybehavex.simulation.social_graph import (
    build_colocation_social_graph,
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


def _colocation_kwargs(**overrides):
    kwargs = dict(
        degree_mu_ln=np.log(10),
        degree_sigma_ln=0.4,
        max_degree=50,
        temperature=0.3,
        max_candidate_pool=500,
        max_ring_expansion=2,
        random_state=42,
    )
    kwargs.update(overrides)
    return kwargs


def test_colocation_graph_is_empty_for_zero_agents():
    starts, neighbors, weights = build_colocation_social_graph(
        np.empty((0, 4)),
        np.empty(0, dtype=np.uint64),
        np.empty(0, dtype=np.uint64),
        **_colocation_kwargs(),
    )
    assert starts.shape == (1,)
    assert len(neighbors) == 0
    assert len(weights) == 0


def test_colocation_graph_is_deterministic():
    n, dim = 150, 8
    embeddings = _normalized_embeddings(n, dim)
    rng = np.random.default_rng(3)
    home_cells = rng.integers(100, 106, size=n).astype(np.uint64)
    work_cells = rng.integers(200, 204, size=n).astype(np.uint64)

    first = build_colocation_social_graph(embeddings, home_cells, work_cells, **_colocation_kwargs())
    second = build_colocation_social_graph(embeddings, home_cells, work_cells, **_colocation_kwargs())
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)


def test_colocation_graph_respects_max_degree_and_edge_weight_bounds():
    n, dim = 150, 8
    embeddings = _normalized_embeddings(n, dim)
    rng = np.random.default_rng(4)
    home_cells = rng.integers(100, 103, size=n).astype(np.uint64)
    work_cells = rng.integers(200, 202, size=n).astype(np.uint64)

    starts, neighbors, weights = build_colocation_social_graph(
        embeddings, home_cells, work_cells, **_colocation_kwargs(max_degree=15)
    )
    degrees = np.diff(starts)
    assert np.all(degrees <= 15)
    assert np.all((-1.0 <= weights) & (weights <= 1.0))
    assert starts.dtype == np.int64
    assert neighbors.dtype == np.int64
    assert weights.dtype == np.float64
    for i in range(n):
        row = neighbors[starts[i] : starts[i + 1]]
        assert i not in row
        assert np.all((0 <= row) & (row < n))


def test_colocation_graph_only_connects_shared_home_or_work_cells():
    n, dim = 200, 6
    embeddings = _normalized_embeddings(n, dim)
    rng = np.random.default_rng(5)
    home_cells = rng.integers(100, 105, size=n).astype(np.uint64)
    work_cells = rng.integers(200, 203, size=n).astype(np.uint64)

    starts, neighbors, _weights = build_colocation_social_graph(
        embeddings, home_cells, work_cells, **_colocation_kwargs(max_ring_expansion=0)
    )
    for i in range(n):
        for j in neighbors[starts[i] : starts[i + 1]]:
            assert home_cells[i] == home_cells[j] or work_cells[i] == work_cells[j]


def test_colocation_graph_expands_through_h3_rings_when_cell_is_isolated():
    n, dim = 20, 4
    embeddings = _normalized_embeddings(n, dim)
    base_cell = h3.latlng_to_cell(37.7749, -122.4194, 9)
    base_int = h3.str_to_int(base_cell)
    neighbor_int = h3.str_to_int(next(iter(h3.grid_ring(base_cell, 1))))

    home_cells = np.full(n, base_int, dtype=np.uint64)
    work_cells = np.full(n, base_int, dtype=np.uint64)
    home_cells[0] = neighbor_int
    work_cells[0] = neighbor_int

    kwargs = _colocation_kwargs(degree_mu_ln=np.log(5), degree_sigma_ln=0.3, max_degree=20)

    starts, neighbors, _weights = build_colocation_social_graph(
        embeddings, home_cells, work_cells, **{**kwargs, "max_ring_expansion": 2}
    )
    assert len(neighbors[starts[0] : starts[1]]) > 0

    starts_no_expansion, neighbors_no_expansion, _ = build_colocation_social_graph(
        embeddings, home_cells, work_cells, **{**kwargs, "max_ring_expansion": 0}
    )
    assert len(neighbors_no_expansion[starts_no_expansion[0] : starts_no_expansion[1]]) == 0


def test_colocation_graph_rejects_non_positive_temperature_and_max_degree():
    embeddings = _normalized_embeddings(5, 4)
    cells = np.zeros(5, dtype=np.uint64)
    try:
        build_colocation_social_graph(embeddings, cells, cells, **_colocation_kwargs(temperature=0))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    try:
        build_colocation_social_graph(embeddings, cells, cells, **_colocation_kwargs(max_degree=0))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
