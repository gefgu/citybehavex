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

import numpy as np


def build_profile_social_graph(
    profile_embeddings: np.ndarray,
    k: int = 10,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a KNN social graph from L2-normalized profile embeddings.

    Each agent connects to its ``k`` most-similar peers (by cosine similarity).
    Self-loops are excluded. When two agents would connect, the edge is stored
    for both directions (symmetric graph).

    Args:
        profile_embeddings: ``[n_agents, dim]`` L2-normalized embedding matrix.
        k: Number of nearest neighbours per agent.
        rng: Optional RNG for breaking ties (not used currently, reserved).

    Returns:
        ``(neighbor_starts, neighbors, edge_weights)`` where:
        - ``neighbor_starts`` is int64[n_agents + 1] (CSR indptr)
        - ``neighbors`` is int64[n_edges] (CSR indices)
        - ``edge_weights`` is float64[n_edges] (cosine similarities)
    """
    n = len(profile_embeddings)
    if n == 0:
        empty = np.zeros(1, dtype=np.int64)
        return empty, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    k = min(k, n - 1)  # can't have more neighbours than other agents

    # Cosine similarity matrix (already L2-normalized → dot product).
    sim = np.clip(
        profile_embeddings.astype(np.float64) @ profile_embeddings.astype(np.float64).T,
        -1.0, 1.0,
    )
    np.fill_diagonal(sim, -2.0)  # exclude self-loops

    # For each agent, pick top-k neighbours.
    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for i in range(n):
        top_k = np.argpartition(sim[i], -k)[-k:]
        for j in top_k:
            s = float(sim[i, j])
            adj[i].append((int(j), s))
            # Ensure symmetry: add reverse edge too.
            adj[j].append((i, s))

    # Deduplicate and sort each adjacency list.
    neighbour_starts = np.zeros(n + 1, dtype=np.int64)
    nb_list: list[int] = []
    ew_list: list[float] = []
    for i in range(n):
        # Deduplicate by keeping max similarity for repeated edges.
        seen: dict[int, float] = {}
        for j, s in adj[i]:
            if j in seen:
                seen[j] = max(seen[j], s)
            else:
                seen[j] = s
        sorted_nb = sorted(seen.items())
        neighbour_starts[i + 1] = neighbour_starts[i] + len(sorted_nb)
        for j, s in sorted_nb:
            nb_list.append(j)
            ew_list.append(s)

    return (
        neighbour_starts,
        np.asarray(nb_list, dtype=np.int64),
        np.asarray(ew_list, dtype=np.float64),
    )


def random_geometric_fallback(
    n_agents: int,
    radius: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Thin wrapper around skmob2 random-geometric graph returning the same triple.

    Used as a fallback when profile embeddings are unavailable. Edge weights are
    all 1.0 (no semantic information).
    """
    from skmob2 import _core as _skmob_core

    ns, nb = _skmob_core.model_social_graph_random_geometric(
        int(n_agents), float(radius), int(random_state)
    )
    ns = np.asarray(ns, dtype=np.int64)
    nb = np.asarray(nb, dtype=np.int64)
    ew = np.ones(len(nb), dtype=np.float64)
    return ns, nb, ew
