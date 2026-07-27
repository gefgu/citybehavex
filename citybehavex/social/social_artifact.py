"""Social graph artifact helpers for simulation outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import TruncatedSVD


@dataclass(frozen=True)
class SocialGraphArtifact:
    nodes: list[list[Any]]
    edges: list[list[float]]
    degrees: list[int]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.metadata,
            "nodes": self.nodes,
            "edges": self.edges,
            "degrees": self.degrees,
        }

    def write_json(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), separators=(",", ":")), encoding="utf-8")


def social_network_sidecar_path(output_path: str | Path) -> Path:
    p = Path(output_path)
    return p.with_name(f"{p.stem}_social_network.json")


def _social_layout(
    n_agents: int,
    random_state: int,
    profile_embeddings: np.ndarray | None,
) -> np.ndarray:
    if profile_embeddings is not None and n_agents >= 2 and profile_embeddings.ndim == 2:
        embeddings = np.ascontiguousarray(profile_embeddings, dtype=np.float64)
        if embeddings.shape[1] >= 2:
            coords = TruncatedSVD(n_components=2, random_state=random_state).fit_transform(
                embeddings
            )
        elif embeddings.shape[1] == 1:
            coords = np.column_stack([embeddings[:, 0], np.zeros(n_agents, dtype=np.float64)])
        else:
            coords = np.empty((n_agents, 0), dtype=np.float64)
        if coords.shape == (n_agents, 2) and np.isfinite(coords).all():
            scale = np.std(coords, axis=0)
            scale = np.where(scale == 0, 1.0, scale)
            return np.round((coords / scale) * 100.0, 1)

    rng = np.random.default_rng(random_state)
    return np.round((rng.random((n_agents, 2), dtype=np.float64) - 0.5) * 1000.0, 1)


def build_social_graph_artifact(
    neighbor_starts: np.ndarray,
    neighbors: np.ndarray,
    edge_weights: np.ndarray,
    *,
    n_agents: int,
    random_state: int,
    social_graph_k: int,
    profile_embeddings: np.ndarray | None = None,
    profile_types: list[str] | None = None,
    edge_sources: np.ndarray | None = None,
    edge_targets: np.ndarray | None = None,
    edge_kinds: np.ndarray | None = None,
) -> SocialGraphArtifact:
    """Pack the simulation social graph for WebGL rendering."""
    starts = np.asarray(neighbor_starts, dtype=np.int64)
    neigh = np.asarray(neighbors, dtype=np.int64)
    weights = np.asarray(edge_weights, dtype=np.float64)
    coords = _social_layout(n_agents, random_state, profile_embeddings)
    if edge_sources is not None and edge_targets is not None:
        sources = np.asarray(edge_sources, dtype=np.int64)
        targets = np.asarray(edge_targets, dtype=np.int64)
        final_weights = np.asarray(edge_weights, dtype=np.float64)
        degrees = np.bincount(sources[(0 <= sources) & (sources < n_agents)], minlength=n_agents)
    else:
        sources = None
        targets = None
        final_weights = weights
        degrees = np.diff(starts).astype(np.int64)

    weighted = np.zeros(n_agents, dtype=np.float64)
    if sources is not None and targets is not None and len(final_weights) == len(sources):
        for source, weight in zip(sources, final_weights):
            if 0 <= source < n_agents:
                weighted[int(source)] += abs(float(weight))
    else:
        for i in range(n_agents):
            start, end = int(starts[i]), int(starts[i + 1])
            if end > start and len(weights) == len(neigh):
                weighted[i] = float(np.abs(weights[start:end]).sum())
            else:
                weighted[i] = float(end - start)
    max_weighted = float(weighted.max()) if weighted.size else 0.0
    if max_weighted > 0:
        sizes = 3.0 + 13.0 * np.sqrt(weighted / max_weighted)
    else:
        sizes = np.full(n_agents, 3.0, dtype=np.float64)

    nodes: list[list[Any]] = []
    for i in range(n_agents):
        row: list[Any] = [
            float(coords[i, 0]),
            float(coords[i, 1]),
            round(float(np.clip(sizes[i], 3.0, 16.0)), 1),
            i + 1,
        ]
        if profile_types is not None and i < len(profile_types):
            row.append(str(profile_types[i]))
        nodes.append(row)

    edges: list[list[float]] = []
    if sources is not None and targets is not None:
        kinds = (
            np.asarray(edge_kinds, dtype=np.uint8)
            if edge_kinds is not None
            else np.zeros(len(sources), dtype=np.uint8)
        )
        for edge_idx, (source, target) in enumerate(zip(sources, targets)):
            weight = float(final_weights[edge_idx]) if edge_idx < len(final_weights) else 1.0
            kind = int(kinds[edge_idx]) if edge_idx < len(kinds) else 0
            edges.append([int(source), int(target), round(weight, 4), kind])
    else:
        for source in range(n_agents):
            for edge_idx in range(int(starts[source]), int(starts[source + 1])):
                target = int(neigh[edge_idx])
                weight = float(weights[edge_idx]) if len(weights) == len(neigh) else 1.0
                edges.append([source, target, round(weight, 4)])

    return SocialGraphArtifact(
        nodes=nodes,
        edges=edges,
        degrees=[int(d) for d in degrees],
        metadata={
            "kind": "final_dynamic_social" if sources is not None else "initial_profile_similarity",
            "node_count": int(n_agents),
            "edge_count": int(len(edges)),
            "layout": "profile_svd" if profile_embeddings is not None else "fallback_seeded_random",
            "directed": True,
            "social_graph_k": int(social_graph_k),
            "edge_kind_labels": {"0": "initial", "1": "dynamic"} if sources is not None else None,
        },
    )
