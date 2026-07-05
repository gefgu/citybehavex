from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from citybehavex.simulation.core import social_network_sidecar_path

NETWORK_METRIC_LABELS = {
    "clustering_coefficient": "Clustering coefficient",
    "edge_persistence": "Edge persistence",
    "topological_overlap": "Topological overlap",
}


@dataclass(frozen=True)
class NetworkGraph:
    node_count: int
    edges: set[tuple[int, int]]
    adjacency: list[set[int]]


def encounters_sidecar_path(output_path: str | Path) -> Path:
    p = Path(output_path)
    return p.with_name(f"{p.stem}_encounters{p.suffix}")


def _empty_graph(node_count: int) -> NetworkGraph:
    return NetworkGraph(node_count=node_count, edges=set(), adjacency=[set() for _ in range(node_count)])


def _normal_edge(a: Any, b: Any, node_count: int) -> tuple[int, int] | None:
    try:
        u, v = int(a), int(b)
    except (TypeError, ValueError):
        return None
    if u == v or u < 0 or v < 0 or u >= node_count or v >= node_count:
        return None
    return (u, v) if u < v else (v, u)


def graph_from_edges(node_count: int, edges: set[tuple[int, int]]) -> NetworkGraph:
    graph = _empty_graph(node_count)
    for u, v in edges:
        edge = _normal_edge(u, v, node_count)
        if edge is None:
            continue
        a, b = edge
        graph.edges.add(edge)
        graph.adjacency[a].add(b)
        graph.adjacency[b].add(a)
    return graph


def _load_social_sidecar(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("nodes"), list) or not isinstance(data.get("edges"), list):
        raise ValueError(f"invalid social network sidecar arrays: {path}")
    node_count = int(data.get("node_count", len(data["nodes"])))
    if node_count != len(data["nodes"]):
        raise ValueError(f"social network sidecar node count mismatch: {path}")
    return data


def _social_edges(data: dict[str, Any]) -> set[tuple[int, int]]:
    node_count = int(data["node_count"])
    edges: set[tuple[int, int]] = set()
    for row in data.get("edges", []):
        if isinstance(row, list) and len(row) >= 2:
            edge = _normal_edge(row[0], row[1], node_count)
            if edge is not None:
                edges.add(edge)
    return edges


def _encounter_edges_and_persistence(
    encounters: pd.DataFrame,
    *,
    node_count: int,
) -> tuple[set[tuple[int, int]], np.ndarray, int]:
    required = {"agent", "contact", "ts"}
    missing = required - set(encounters.columns)
    if missing:
        raise ValueError(f"encounters table missing columns: {', '.join(sorted(missing))}")
    if encounters.empty:
        return set(), np.asarray([], dtype=float), 0

    work = encounters[["agent", "contact", "ts"]].dropna().copy()
    if work.empty:
        return set(), np.asarray([], dtype=float), 0
    work["agent"] = pd.to_numeric(work["agent"], errors="coerce")
    work["contact"] = pd.to_numeric(work["contact"], errors="coerce")
    work = work.dropna(subset=["agent", "contact", "ts"])
    if work.empty:
        return set(), np.asarray([], dtype=float), 0

    time_steps = int(work["ts"].nunique())
    if time_steps <= 0:
        return set(), np.asarray([], dtype=float), 0

    pair_steps: dict[tuple[int, int], set[Any]] = {}
    for row in work.itertuples(index=False):
        edge = _normal_edge(row.agent, row.contact, node_count)
        if edge is None:
            continue
        pair_steps.setdefault(edge, set()).add(row.ts)

    edges = set(pair_steps)
    persistence = np.asarray(
        [len(steps) / time_steps for steps in pair_steps.values()],
        dtype=float,
    )
    return edges, persistence, time_steps


def clustering_coefficients(graph: NetworkGraph) -> np.ndarray:
    values = np.zeros(graph.node_count, dtype=float)
    for node, neighbors in enumerate(graph.adjacency):
        degree = len(neighbors)
        if degree < 2:
            continue
        links = 0
        ordered = list(neighbors)
        for idx, u in enumerate(ordered[:-1]):
            u_neighbors = graph.adjacency[u]
            for v in ordered[idx + 1 :]:
                if v in u_neighbors:
                    links += 1
        values[node] = (2.0 * links) / (degree * (degree - 1))
    return values


def topological_overlap(graph: NetworkGraph) -> np.ndarray:
    values: list[float] = []
    for u, v in sorted(graph.edges):
        left = graph.adjacency[u]
        right = graph.adjacency[v]
        union = left | right
        values.append(float(len(left & right) / len(union)) if union else 0.0)
    return np.asarray(values, dtype=float)


def _distribution_summary(values: np.ndarray) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "mean": None}
    return {"count": int(arr.size), "mean": float(arr.mean())}


def _safe_wasserstein(left: np.ndarray, right: np.ndarray) -> float | None:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return None
    return float(wasserstein_distance(a, b))


def degree_preserving_random_graph(
    degrees: np.ndarray,
    *,
    seed: int = 42,
) -> NetworkGraph:
    deg = np.asarray(degrees, dtype=float)
    n = int(len(deg))
    total_degree = float(deg.sum())
    if n <= 1 or total_degree <= 0:
        return _empty_graph(n)

    rng = np.random.default_rng(seed)
    edges: set[tuple[int, int]] = set()
    for i in range(n - 1):
        if deg[i] <= 0:
            continue
        probs = np.clip((deg[i] * deg[i + 1 :]) / total_degree, 0.0, 1.0)
        if probs.size == 0:
            continue
        draws = rng.random(probs.size) < probs
        for offset in np.flatnonzero(draws):
            edges.add((i, i + 1 + int(offset)))
    return graph_from_edges(n, edges)


def _random_persistence(
    edges: set[tuple[int, int]],
    degrees: np.ndarray,
    *,
    time_steps: int,
    seed: int,
) -> np.ndarray:
    if time_steps <= 0 or not edges:
        return np.asarray([], dtype=float)
    total_degree = float(np.asarray(degrees, dtype=float).sum())
    if total_degree <= 0:
        return np.asarray([], dtype=float)
    rng = np.random.default_rng(seed)
    values = []
    for u, v in edges:
        p = float(np.clip((degrees[u] * degrees[v]) / total_degree, 0.0, 1.0))
        values.append(float(rng.binomial(time_steps, p) / time_steps))
    return np.asarray(values, dtype=float)


def _metric_bundle(
    graph: NetworkGraph,
    persistence: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "clustering_coefficient": clustering_coefficients(graph),
        "edge_persistence": persistence,
        "topological_overlap": topological_overlap(graph),
    }


def _network_block_from_graph(
    graph: NetworkGraph,
    *,
    source_sidecar: dict[str, Any],
    kind: str,
) -> dict[str, Any]:
    source_nodes = source_sidecar.get("nodes", [])
    nodes: list[list[Any]] = []
    degrees = [len(graph.adjacency[i]) for i in range(graph.node_count)]
    max_degree = max(degrees) if degrees else 0
    for i in range(graph.node_count):
        if i < len(source_nodes) and isinstance(source_nodes[i], list) and len(source_nodes[i]) >= 4:
            row = list(source_nodes[i])
            if max_degree > 0:
                row[2] = round(float(3.0 + 13.0 * np.sqrt(degrees[i] / max_degree)), 1)
            nodes.append(row)
        else:
            nodes.append([float(i), 0.0, 3.0, i + 1])
    return {
        "kind": kind,
        "node_count": graph.node_count,
        "edge_count": len(graph.edges),
        "layout": source_sidecar.get("layout", "source_layout"),
        "directed": False,
        "social_graph_k": source_sidecar.get("social_graph_k", 0),
        "nodes": nodes,
        "edges": [[u, v, 1.0] for u, v in sorted(graph.edges)],
        "degrees": degrees,
    }


def build_network_validation(
    synthetic_path: str | Path,
    *,
    seed: int = 42,
) -> tuple[dict[str, Any] | None, list[str]]:
    warnings: list[str] = []
    synthetic = Path(synthetic_path)
    social_path = social_network_sidecar_path(synthetic)
    if not social_path.exists():
        return None, [f"social network sidecar not found: {social_path}"]

    social_data = _load_social_sidecar(social_path)
    node_count = int(social_data["node_count"])
    edges = _social_edges(social_data)

    persistence = np.asarray([], dtype=float)
    time_steps = 0
    enc_path = encounters_sidecar_path(synthetic)
    if enc_path.exists():
        encounter_edges, persistence, time_steps = _encounter_edges_and_persistence(
            pd.read_parquet(enc_path),
            node_count=node_count,
        )
        edges |= encounter_edges
    else:
        warnings.append(f"encounters sidecar not found: {enc_path}; edge persistence unavailable")

    synthetic_graph = graph_from_edges(node_count, edges)
    degrees = np.asarray([len(n) for n in synthetic_graph.adjacency], dtype=float)
    random_graph = degree_preserving_random_graph(degrees, seed=seed)
    random_persistence = _random_persistence(
        random_graph.edges,
        degrees,
        time_steps=time_steps,
        seed=seed + 1,
    )

    synthetic_metrics = _metric_bundle(synthetic_graph, persistence)
    random_metrics = _metric_bundle(random_graph, random_persistence)
    wasserstein = {
        name: _safe_wasserstein(synthetic_metrics[name], random_metrics[name])
        for name in NETWORK_METRIC_LABELS
    }
    for name, value in wasserstein.items():
        if value is None:
            warnings.append(f"{NETWORK_METRIC_LABELS[name]} distribution is empty; Wasserstein unavailable")

    return (
        {
            "comparison": "synthetic_vs_random",
            "random_model": "degree_preserving_rnd",
            "wasserstein": wasserstein,
            "distributions": {
                "synthetic": {
                    name: _distribution_summary(values)
                    for name, values in synthetic_metrics.items()
                },
                "random": {
                    name: _distribution_summary(values)
                    for name, values in random_metrics.items()
                },
            },
            "synthetic_network": _network_block_from_graph(
                synthetic_graph,
                source_sidecar=social_data,
                kind="synthetic_social_encounter_union",
            ),
            "random_network": _network_block_from_graph(
                random_graph,
                source_sidecar=social_data,
                kind="degree_preserving_rnd",
            ),
        },
        warnings,
    )
