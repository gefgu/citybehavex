"""Compatibility exports for fastmob's social-network primitives."""

from fastmob.social import (
    NetworkGraph,
    clustering_coefficients,
    degree_preserving_random_graph,
    distribution_summary,
    graph_from_edges,
    random_persistence,
    safe_wasserstein,
    topological_overlap,
)

__all__ = [
    "NetworkGraph",
    "clustering_coefficients",
    "degree_preserving_random_graph",
    "distribution_summary",
    "graph_from_edges",
    "random_persistence",
    "safe_wasserstein",
    "topological_overlap",
]
