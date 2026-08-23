"""Lightweight undirected-graph primitives for social-network validation.

fastmob's earlier ``fastmob.measures.collective.contact_network`` module
(``NetworkGraph``, ``co_presence_graph_from_visits``,
``degree_preserving_random_graph``, ``clustering_coefficients``,
``topological_overlap``, ``random_persistence``, ``safe_wasserstein``,
``distribution_summary``) was removed when fastmob's social-network tooling
was redesigned around the RECAST framework (``fastmob.social``), which
classifies *which* co-presence edges are genuine ties rather than exposing
generic graph-comparison primitives. citybehavex's network-validation report
still needs those generic primitives (build a graph from raw co-presence
pairs, compare its degree/clustering/edge-persistence/topological-overlap
distributions against a degree-preserving random baseline), so they're
reimplemented here on top of ``networkx`` -- already a transitive dependency
-- and fastmob's still-available generic ``wasserstein_distance``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import networkx as nx
import numpy as np
from fastmob import wasserstein_distance


@dataclass(frozen=True)
class NetworkGraph:
    """A simple undirected graph: integer node ids ``0..node_count-1``."""

    node_count: int
    edge_from: np.ndarray
    edge_to: np.ndarray

    @property
    def edge_count(self) -> int:
        return len(self.edge_from)

    @property
    def edges(self) -> list[tuple[int, int]]:
        return list(zip(self.edge_from.tolist(), self.edge_to.tolist()))

    def degrees(self) -> np.ndarray:
        deg = np.zeros(self.node_count, dtype=np.int64)
        if self.edge_count:
            np.add.at(deg, self.edge_from, 1)
            np.add.at(deg, self.edge_to, 1)
        return deg

    def to_networkx(self) -> nx.Graph:
        g: nx.Graph = nx.Graph()
        g.add_nodes_from(range(self.node_count))
        if self.edge_count:
            g.add_edges_from(zip(self.edge_from.tolist(), self.edge_to.tolist()))
        return g


def graph_from_edges(node_count: int, edges: Iterable[tuple[int, int]]) -> NetworkGraph:
    dedup = {(int(u), int(v)) if u < v else (int(v), int(u)) for u, v in edges if u != v}
    if not dedup:
        empty = np.empty(0, dtype=np.uint32)
        return NetworkGraph(node_count, empty, empty)
    arr = np.asarray(sorted(dedup), dtype=np.uint32)
    return NetworkGraph(node_count, arr[:, 0], arr[:, 1])


def clustering_coefficients(graph: NetworkGraph) -> np.ndarray:
    g = graph.to_networkx()
    coeffs = nx.clustering(g)
    return np.asarray([coeffs[i] for i in range(graph.node_count)], dtype=float)


def topological_overlap(graph: NetworkGraph) -> np.ndarray:
    """Per-edge neighborhood (Jaccard) overlap between an edge's two
    endpoints' neighbor sets (each endpoint is its own neighbor's set
    member via the edge itself, so a shared triangle counts) -- the
    "overlap" measure from Onnela et al. (2007), which the RECAST paper
    also uses as an edge-strength proxy.
    """
    g = graph.to_networkx()
    values = []
    for u, v in g.edges():
        nu = set(g.neighbors(u))
        nv = set(g.neighbors(v))
        union = nu | nv
        values.append(len(nu & nv) / len(union) if union else 0.0)
    return np.asarray(values, dtype=float)


def degree_preserving_random_graph(degrees: np.ndarray, *, seed: int) -> NetworkGraph:
    """A random graph whose degree sequence matches ``degrees`` in
    expectation, via the configuration model (stub-matching, then
    collapsing self-loops/parallel edges into a simple graph -- the
    standard, fast approach; exact degree preservation isn't guaranteed
    after simplification, same tradeoff the old degree-preserving generator
    documented)."""
    degree_sequence = [int(d) for d in degrees]
    node_count = len(degree_sequence)
    if node_count == 0 or sum(degree_sequence) == 0:
        empty = np.empty(0, dtype=np.uint32)
        return NetworkGraph(node_count, empty, empty)
    multigraph = nx.configuration_model(degree_sequence, seed=seed)
    simple = nx.Graph(multigraph)
    simple.remove_edges_from(nx.selfloop_edges(simple))
    if simple.number_of_edges() == 0:
        empty = np.empty(0, dtype=np.uint32)
        return NetworkGraph(node_count, empty, empty)
    edges = np.asarray(sorted(simple.edges()), dtype=np.uint32)
    return NetworkGraph(node_count, edges[:, 0], edges[:, 1])


def random_persistence(
    random_graph: NetworkGraph,
    source_degrees: np.ndarray,
    *,
    time_steps: int,
    seed: int,
) -> np.ndarray:
    """Synthetic per-edge persistence for a random baseline graph.

    There's no real co-presence timeline for a random graph, so persistence
    is modeled as repeated independent contact draws: each random edge gets
    a Binomial(time_steps, p) / time_steps persistence rate, with p set to
    the source graph's mean observed persistence (moment-matched null
    model -- same expected rate as the real network, no structure).
    """
    if random_graph.edge_count == 0 or time_steps <= 0:
        return np.asarray([], dtype=float)
    p = float(np.clip(np.mean(source_degrees) / max(source_degrees.size, 1), 0.0, 1.0)) if source_degrees.size else 0.0
    rng = np.random.default_rng(seed)
    draws = rng.binomial(time_steps, p, size=random_graph.edge_count)
    return draws.astype(float) / time_steps


def safe_wasserstein(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return None
    return float(wasserstein_distance(a, b))


def distribution_summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
            "median": 0.0, "p10": 0.0, "p90": 0.0,
        }
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


def co_presence_graph_from_visits(
    visits: Any,
    *,
    user_id_col: str,
    day_col: str,
    location_id_col: str,
    max_group_size: int,
) -> tuple[NetworkGraph, np.ndarray, int, dict[str, int]]:
    """Build a co-presence graph: an edge between two users for every
    (location, day) they were both observed at, deduplicated across
    days, plus per-edge persistence (fraction of days co-present out of
    the total days either could have been) and the total day count.

    Groups larger than ``max_group_size`` are skipped (a busy
    location/day pair would otherwise contribute O(group_size^2) edges).
    """
    df = visits.select([user_id_col, day_col, location_id_col]).drop_nulls()
    if df.is_empty():
        empty = np.empty(0, dtype=np.uint32)
        return NetworkGraph(0, empty, empty), np.asarray([], dtype=float), 0, {"skipped_groups": 0, "skipped_rows": 0}

    users = df[user_id_col].unique().sort().to_list()
    user_index = {u: i for i, u in enumerate(users)}
    node_count = len(users)
    days = df[day_col].unique().to_list()
    time_steps = len(days)

    # Pure-Python per-group pairing -- fastmob's Rust-backed
    # co_presence_graph_from_visits no longer exists (removed with the
    # RECAST-based social-network redesign); this is correct but
    # meaningfully slower at real dataset scale (tens of millions of raw
    # pair-instances for shanghai/yjmob-sized inputs).
    pair_days: dict[tuple[int, int], set[Any]] = {}
    skipped_groups = 0
    skipped_rows = 0
    for (_loc, day), group in df.group_by([location_id_col, day_col]):
        group_users = group[user_id_col].unique().to_list()
        if len(group_users) < 2:
            continue
        if len(group_users) > max_group_size:
            skipped_groups += 1
            skipped_rows += len(group_users)
            continue
        indices = sorted(user_index[u] for u in group_users)
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                pair_days.setdefault((indices[i], indices[j]), set()).add(day)

    if not pair_days:
        empty = np.empty(0, dtype=np.uint32)
        return (
            NetworkGraph(node_count, empty, empty),
            np.asarray([], dtype=float),
            time_steps,
            {"skipped_groups": skipped_groups, "skipped_rows": skipped_rows},
        )

    edges = sorted(pair_days)
    edge_arr = np.asarray(edges, dtype=np.uint32)
    persistence = np.asarray(
        [len(pair_days[edge]) / time_steps if time_steps else 0.0 for edge in edges],
        dtype=float,
    )
    graph = NetworkGraph(node_count, edge_arr[:, 0], edge_arr[:, 1])
    return graph, persistence, time_steps, {"skipped_groups": skipped_groups, "skipped_rows": skipped_rows}
