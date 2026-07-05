"""Profile-similarity social graph for the STS-EPR simulation.

Replaces the random-geometric graph (spatial proximity) with a graph built from
cosine similarity between agent profile embeddings. Agents with similar profiles
are more likely to be friends — seeding the social influence mechanism in the
Rust sim with a semantically meaningful initial edge weight.

The graph is output in the same CSR (compressed sparse row) format that the Rust
core already consumes:
  ``neighbor_starts[i] .. neighbor_starts[i+1]`` indexes into ``neighbors``
  to give the list of agent indices adjacent to agent ``i``.
  ``edge_weights[j]`` is the cosine similarity for the j-th edge in ``neighbors``.

Over simulation time the Rust loop blends this initial weight with mobility-cosine
(visit-pattern cosine) updates — so the graph evolves toward topological overlap.
"""

from __future__ import annotations

import math

import h3
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors


def _empty_graph(n_nodes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros(n_nodes + 1, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.float64),
    )


def _rows_to_csr(
    rows: list[np.ndarray],
    edge_weights: list[np.ndarray],
    n_nodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    neighbor_starts = np.zeros(n_nodes + 1, dtype=np.int64)
    for i, row in enumerate(rows):
        neighbor_starts[i + 1] = neighbor_starts[i] + len(row)
    if rows:
        neighbors = np.concatenate(rows).astype(np.int64, copy=False)
        weights = np.concatenate(edge_weights).astype(np.float64, copy=False)
    else:
        neighbors = np.empty(0, dtype=np.int64)
        weights = np.empty(0, dtype=np.float64)
    return neighbor_starts, neighbors, weights


def _bounded_k(k: int, n: int) -> int:
    if k <= 0:
        raise ValueError("k must be positive")
    return min(int(k), max(0, n - 1))


def build_knn_fallback_social_graph(
    n_agents: int,
    k: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a bounded synthetic social graph with at most ``n_agents * k`` edges.

    Agents are placed deterministically in a synthetic unit square, then each row
    connects to its k nearest peers. This replaces the old fixed-radius random
    geometric graph whose expected degree grew linearly with population size.
    """
    if n_agents < 0:
        raise ValueError("n_agents must be non-negative")
    k = _bounded_k(k, n_agents)
    if n_agents == 0 or k == 0:
        return _empty_graph(n_agents)

    rng = np.random.default_rng(random_state)
    coordinates = rng.random((n_agents, 2), dtype=np.float64)
    n_neighbors = min(k + 1, n_agents)
    tree = NearestNeighbors(
        n_neighbors=n_neighbors,
        algorithm="kd_tree",
        metric="euclidean",
        n_jobs=-1,
    )
    tree.fit(coordinates)
    indices = tree.kneighbors(coordinates, return_distance=False)

    rows: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for i, row in enumerate(indices):
        peers = row[row != i][:k].astype(np.int64, copy=False)
        rows.append(peers)
        weights.append(np.ones(len(peers), dtype=np.float64))
    return _rows_to_csr(rows, weights, n_agents)


def build_profile_social_graph(
    profile_embeddings: np.ndarray,
    k: int = 20,
    random_state: int = 42,
    exact_threshold: int = 10_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a KNN social graph from L2-normalized profile embeddings.

    Each agent connects to its ``k`` most-similar peers (by cosine similarity).
    Self-loops are excluded. Edges are directed so the edge count is capped by
    ``n_agents * k``.

    Args:
        profile_embeddings: ``[n_agents, dim]`` L2-normalized embedding matrix.
        k: Number of nearest neighbours per agent.
        random_state: Seed for deterministic large-graph cluster sampling.
        exact_threshold: Use exact cosine kNN up to this population size; above
            it, use MiniBatchKMeans clusters and sample bounded peer sets.

    Returns:
        ``(neighbor_starts, neighbors, edge_weights)`` where:
        - ``neighbor_starts`` is int64[n_agents + 1] (CSR indptr)
        - ``neighbors`` is int64[n_edges] (CSR indices)
        - ``edge_weights`` is float64[n_edges] (cosine similarities)
    """
    n = len(profile_embeddings)
    k = _bounded_k(k, n)
    if n == 0 or k == 0:
        return _empty_graph(n)
    if exact_threshold <= 0:
        raise ValueError("exact_threshold must be positive")

    embeddings = np.ascontiguousarray(profile_embeddings, dtype=np.float64)
    if n <= exact_threshold:
        return _exact_profile_knn(embeddings, k)
    return _cluster_sample_profile_graph(embeddings, k, random_state)


def _exact_profile_knn(
    embeddings: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(embeddings)
    nn = NearestNeighbors(
        n_neighbors=min(k + 1, n),
        algorithm="brute",
        metric="cosine",
        n_jobs=-1,
    )
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings, return_distance=True)

    rows: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for i, row in enumerate(indices):
        mask = row != i
        peers = row[mask][:k].astype(np.int64, copy=False)
        peer_distances = distances[i][mask][:k]
        rows.append(peers)
        weights.append(np.clip(1.0 - peer_distances, -1.0, 1.0))
    return _rows_to_csr(rows, weights, n)


def _cluster_sample_profile_graph(
    embeddings: np.ndarray,
    k: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(embeddings)
    n_clusters = min(n, max(2, math.ceil(n / 500)))
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=min(4096, n),
        n_init="auto",
    )
    labels = kmeans.fit_predict(embeddings)
    clusters = [np.flatnonzero(labels == c).astype(np.int64) for c in range(n_clusters)]

    centroids = np.ascontiguousarray(kmeans.cluster_centers_, dtype=np.float64)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    centroids = centroids / norms
    centroid_order = np.argsort(-(centroids @ centroids.T), axis=1)

    rows: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    all_agents = np.arange(n, dtype=np.int64)
    for i in range(n):
        label = int(labels[i])
        candidate_parts: list[np.ndarray] = []
        total = 0
        for cluster_id in centroid_order[label]:
            cluster_agents = clusters[int(cluster_id)]
            if len(cluster_agents) == 0:
                continue
            candidate_parts.append(cluster_agents)
            total += len(cluster_agents)
            if total > k:
                break
        candidates = (
            np.concatenate(candidate_parts)
            if candidate_parts
            else all_agents
        )
        candidates = candidates[candidates != i]
        if len(candidates) <= k:
            peers = candidates.astype(np.int64, copy=False)
        else:
            rng = np.random.default_rng(np.random.SeedSequence([int(random_state), i]))
            peers = rng.choice(candidates, size=k, replace=False).astype(np.int64, copy=False)
            peers.sort()
        rows.append(peers)
        if len(peers) == 0:
            weights.append(np.empty(0, dtype=np.float64))
        else:
            sims = embeddings[i] @ embeddings[peers].T
            weights.append(np.clip(sims, -1.0, 1.0).astype(np.float64, copy=False))
    return _rows_to_csr(rows, weights, n)


def _group_by_cell(cells: np.ndarray) -> dict[int, np.ndarray]:
    """Group agent indices by H3 cell id in O(n log n), no Python loop."""
    if cells.size == 0:
        return {}
    order = np.argsort(cells, kind="stable")
    sorted_cells = cells[order]
    boundaries = np.flatnonzero(np.diff(sorted_cells)) + 1
    groups = np.split(order, boundaries)
    starts = np.concatenate(([0], boundaries))
    return {int(sorted_cells[start]): group.astype(np.int64) for start, group in zip(starts, groups)}


_EMPTY_POOL = np.empty(0, dtype=np.int64)


def _pool_from_groups(cells: list[int], groups: dict[int, np.ndarray], exclude: int) -> np.ndarray:
    parts = [groups[c] for c in cells if c in groups]
    if not parts:
        return _EMPTY_POOL
    pool = np.unique(np.concatenate(parts))
    return pool[pool != exclude]


def _colocation_pool(
    agent: int,
    home_cell: int,
    work_cell: int,
    home_groups: dict[int, np.ndarray],
    work_groups: dict[int, np.ndarray],
    max_ring_expansion: int,
) -> np.ndarray:
    pool = np.union1d(
        _pool_from_groups([home_cell], home_groups, agent),
        _pool_from_groups([work_cell], work_groups, agent),
    )
    if pool.size > 0 or max_ring_expansion <= 0:
        return pool
    home_str = h3.int_to_str(int(home_cell))
    work_str = h3.int_to_str(int(work_cell))
    for ring in range(1, max_ring_expansion + 1):
        ring_cells = [
            h3.str_to_int(c) for c in {*h3.grid_disk(home_str, ring), *h3.grid_disk(work_str, ring)}
        ]
        home_ring_pool = _pool_from_groups(ring_cells, home_groups, agent)
        work_ring_pool = _pool_from_groups(ring_cells, work_groups, agent)
        pool = np.union1d(home_ring_pool, work_ring_pool)
        if pool.size > 0:
            return pool
    return pool


def build_colocation_social_graph(
    profile_embeddings: np.ndarray,
    home_cells: np.ndarray,
    work_cells: np.ndarray,
    degree_mu_ln: float,
    degree_sigma_ln: float,
    max_degree: int,
    temperature: float,
    max_candidate_pool: int,
    max_ring_expansion: int,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a friendship graph from home/work H3-cell colocation.

    Each agent's target degree is an independent draw from
    ``lognormal(degree_mu_ln, degree_sigma_ln)`` (clipped to
    ``[0, max_degree]``). Friends are then sampled -- without replacement,
    weighted by ``exp(cosine_similarity / temperature)`` -- from the pool of
    agents who share the agent's home or work H3 cell. If that pool is
    empty, it's expanded through H3 rings around home and work (up to
    ``max_ring_expansion``); if still empty, the agent gets zero edges (see
    ``citybehavex-py`` social.rs for the runtime "casual encounter"
    mechanism that can still connect such agents during simulation).

    Returns:
        ``(neighbor_starts, neighbors, edge_weights)`` in the same CSR
        format as ``build_profile_social_graph``.
    """
    n = len(profile_embeddings)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if max_degree <= 0:
        raise ValueError("max_degree must be positive")
    if n == 0:
        return _empty_graph(n)

    embeddings = np.ascontiguousarray(profile_embeddings, dtype=np.float64)
    home_cells = np.asarray(home_cells, dtype=np.uint64)
    work_cells = np.asarray(work_cells, dtype=np.uint64)
    home_groups = _group_by_cell(home_cells)
    work_groups = _group_by_cell(work_cells)

    rng = np.random.default_rng(random_state)
    degrees = np.clip(
        np.round(rng.lognormal(degree_mu_ln, degree_sigma_ln, size=n)),
        0,
        max_degree,
    ).astype(np.int64)

    rows: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for i in range(n):
        degree_i = int(degrees[i])
        pool = (
            _colocation_pool(i, int(home_cells[i]), int(work_cells[i]), home_groups, work_groups, max_ring_expansion)
            if degree_i > 0
            else _EMPTY_POOL
        )
        if pool.size == 0:
            rows.append(_EMPTY_POOL)
            weights.append(np.empty(0, dtype=np.float64))
            continue

        sub_seed, sample_seed = np.random.SeedSequence([int(random_state), i]).spawn(2)
        if pool.size > max_candidate_pool:
            pool = np.random.default_rng(sub_seed).choice(pool, size=max_candidate_pool, replace=False)

        sims = np.clip(embeddings[i] @ embeddings[pool].T, -1.0, 1.0)
        logits = (sims - sims.max()) / temperature
        w = np.exp(logits)
        probs = w / w.sum()

        k = min(degree_i, pool.size)
        chosen = np.random.default_rng(sample_seed).choice(pool.size, size=k, replace=False, p=probs)
        order = np.argsort(pool[chosen])
        rows.append(pool[chosen][order].astype(np.int64, copy=False))
        weights.append(sims[chosen][order].astype(np.float64, copy=False))

    return _rows_to_csr(rows, weights, n)
