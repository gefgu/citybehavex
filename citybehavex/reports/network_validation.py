from __future__ import annotations

import json
from itertools import combinations
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h3
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from citybehavex.simulation.core import social_network_sidecar_path

NETWORK_METRIC_LABELS = {
    "clustering_coefficient": "Clustering coefficient",
    "edge_persistence": "Edge persistence",
    "topological_overlap": "Topological overlap",
}

_DATETIME_CANDIDATES = ["datetime", "start_timestamp", "timestamp", "check-in_time", "start_time", "time", "date"]
_UID_CANDIDATES = ["uid", "user_id", "user", "agent_id", "userid"]
_LAT_CANDIDATES = ["lat", "latitude"]
_LNG_CANDIDATES = ["lng", "lon", "longitude", "long"]
_LOCATION_CANDIDATES = ["location_id", "tile_id", "venueId", "venue_id", "area", "location"]


@dataclass(frozen=True)
class NetworkGraph:
    node_count: int
    edges: set[tuple[int, int]]
    adjacency: list[set[int]]


def encounters_sidecar_path(output_path: str | Path) -> Path:
    p = Path(output_path)
    return p.with_name(f"{p.stem}_encounters{p.suffix}")


def _detect_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols_lower = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in cols_lower:
            return cols_lower[candidate.lower()]
    return None


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
    source_sidecar: dict[str, Any] | None = None,
    kind: str,
    seed: int = 42,
) -> dict[str, Any]:
    source_sidecar = source_sidecar or {}
    source_nodes = source_sidecar.get("nodes", [])
    nodes: list[list[Any]] = []
    degrees = [len(graph.adjacency[i]) for i in range(graph.node_count)]
    max_degree = max(degrees) if degrees else 0
    rng = np.random.default_rng(seed)
    fallback_coords = (
        np.round((rng.random((graph.node_count, 2), dtype=np.float64) - 0.5) * 1000.0, 1)
        if graph.node_count
        else np.empty((0, 2), dtype=float)
    )
    for i in range(graph.node_count):
        if i < len(source_nodes) and isinstance(source_nodes[i], list) and len(source_nodes[i]) >= 4:
            row = list(source_nodes[i])
            if max_degree > 0:
                row[2] = round(float(3.0 + 13.0 * np.sqrt(degrees[i] / max_degree)), 1)
            nodes.append(row)
        else:
            size = round(float(3.0 + 13.0 * np.sqrt(degrees[i] / max_degree)), 1) if max_degree > 0 else 3.0
            nodes.append([float(fallback_coords[i, 0]), float(fallback_coords[i, 1]), size, i + 1])
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


def _validation_block(
    *,
    comparison: str,
    source_label: str,
    source_graph: NetworkGraph,
    source_persistence: np.ndarray,
    time_steps: int,
    source_kind: str,
    random_seed: int,
    source_sidecar: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    degrees = np.asarray([len(n) for n in source_graph.adjacency], dtype=float)
    random_graph = degree_preserving_random_graph(degrees, seed=random_seed)
    random_persistence = _random_persistence(
        random_graph.edges,
        degrees,
        time_steps=time_steps,
        seed=random_seed + 1,
    )

    source_metrics = _metric_bundle(source_graph, source_persistence)
    random_metrics = _metric_bundle(random_graph, random_persistence)
    wasserstein = {
        name: _safe_wasserstein(source_metrics[name], random_metrics[name])
        for name in NETWORK_METRIC_LABELS
    }
    for name, value in wasserstein.items():
        if value is None:
            warnings.append(f"{NETWORK_METRIC_LABELS[name]} distribution is empty; Wasserstein unavailable")

    return (
        {
            "comparison": comparison,
            "random_model": "degree_preserving_rnd",
            "wasserstein": wasserstein,
            "distributions": {
                source_label: {
                    name: _distribution_summary(values)
                    for name, values in source_metrics.items()
                },
                "random": {
                    name: _distribution_summary(values)
                    for name, values in random_metrics.items()
                },
            },
            "source_network": _network_block_from_graph(
                source_graph,
                source_sidecar=source_sidecar,
                kind=source_kind,
                seed=random_seed,
            ),
            "random_network": _network_block_from_graph(
                random_graph,
                source_sidecar=source_sidecar,
                kind="degree_preserving_rnd",
                seed=random_seed + 1,
            ),
        },
        warnings,
    )


def _synthetic_validation_block(
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
    block, block_warnings = _validation_block(
        comparison="synthetic_vs_random",
        source_label="synthetic",
        source_graph=synthetic_graph,
        source_persistence=persistence,
        time_steps=time_steps,
        source_kind="synthetic_social_encounter_union",
        random_seed=seed,
        source_sidecar=social_data,
    )
    return block, [*warnings, *block_warnings]


def _resolve_observed_location(
    df: pd.DataFrame,
    *,
    location_mode: str,
    location_col: str | None,
    h3_resolution: int,
) -> tuple[pd.Series, str]:
    if location_mode not in {"auto", "location_col", "h3"}:
        raise ValueError(f"unsupported network validation location_mode: {location_mode}")

    chosen = location_col if location_col and location_col in df.columns else None
    if chosen is None and location_mode == "auto":
        chosen = _detect_column(df, _LOCATION_CANDIDATES)
    if location_mode == "location_col" and chosen is None:
        raise ValueError(f"network validation location_col not found: {location_col!r}")
    if chosen is not None and location_mode != "h3":
        return df[chosen].astype(str), chosen

    lat_col = _detect_column(df, _LAT_CANDIDATES)
    lng_col = _detect_column(df, _LNG_CANDIDATES)
    if lat_col is None or lng_col is None:
        raise ValueError("h3 network validation requires latitude/longitude columns")
    lat = pd.to_numeric(df[lat_col], errors="coerce")
    lng = pd.to_numeric(df[lng_col], errors="coerce")
    valid = lat.between(-90, 90) & lng.between(-180, 180)
    cells = pd.Series(pd.NA, index=df.index, dtype="object")
    cells.loc[valid] = [
        h3.latlng_to_cell(float(a), float(b), int(h3_resolution))
        for a, b in zip(lat.loc[valid], lng.loc[valid])
    ]
    return cells, f"h3_{h3_resolution}"


def _observed_edges_and_persistence(
    df: pd.DataFrame,
    *,
    uid_col: str,
    datetime_col: str,
    location_mode: str,
    location_col: str | None,
    h3_resolution: int,
    max_group_size: int,
) -> tuple[NetworkGraph, np.ndarray, int, list[str]]:
    if max_group_size < 2:
        raise ValueError("network validation max_group_size must be at least 2")
    required = [uid_col, datetime_col]
    missing = [col for col in required if col is None or col not in df.columns]
    if missing:
        raise ValueError(f"observed network validation missing columns: {', '.join(map(str, missing))}")

    work = pd.DataFrame(
        {
            "uid": df[uid_col],
            "day": pd.to_datetime(df[datetime_col], errors="coerce").dt.normalize(),
        }
    )
    work["location"], location_source = _resolve_observed_location(
        df,
        location_mode=location_mode,
        location_col=location_col,
        h3_resolution=h3_resolution,
    )
    work = work.dropna(subset=["uid", "day", "location"])
    if work.empty:
        return _empty_graph(0), np.asarray([], dtype=float), 0, [f"observed network has no valid rows using {location_source}"]

    uid_codes, uid_values = pd.factorize(work["uid"], sort=True)
    work = work.assign(node=uid_codes.astype(np.int64))
    node_count = int(len(uid_values))
    time_steps = int(work["day"].nunique())

    pair_days: dict[tuple[int, int], set[pd.Timestamp]] = {}
    skipped_groups = 0
    skipped_rows = 0
    grouped = work.drop_duplicates(["day", "location", "node"]).groupby(["day", "location"], sort=False)["node"]
    for (day, _location), nodes in grouped:
        unique_nodes = np.asarray(nodes, dtype=np.int64)
        group_size = int(len(unique_nodes))
        if group_size < 2:
            continue
        if group_size > max_group_size:
            skipped_groups += 1
            skipped_rows += group_size
            continue
        for u, v in combinations(unique_nodes.tolist(), 2):
            edge = (u, v) if u < v else (v, u)
            pair_days.setdefault(edge, set()).add(day)

    edges = set(pair_days)
    persistence = np.asarray(
        [len(days) / time_steps for days in pair_days.values()],
        dtype=float,
    )
    warnings: list[str] = []
    if skipped_groups:
        warnings.append(
            f"observed network skipped {skipped_groups} {location_source}/day groups "
            f"({skipped_rows} user-presences) larger than max_group_size={max_group_size}"
        )
    return graph_from_edges(node_count, edges), persistence, time_steps, warnings


def _observed_validation_block(
    observed_df: pd.DataFrame,
    *,
    uid_col: str | None = None,
    datetime_col: str | None = None,
    location_mode: str = "auto",
    location_col: str | None = None,
    h3_resolution: int = 9,
    max_group_size: int = 200,
    seed: int = 42,
) -> tuple[dict[str, Any] | None, list[str]]:
    uid_name = uid_col or _detect_column(observed_df, _UID_CANDIDATES)
    datetime_name = datetime_col or _detect_column(observed_df, _DATETIME_CANDIDATES)
    if uid_name is None or datetime_name is None:
        return None, ["observed network validation requires user and datetime columns"]

    graph, persistence, time_steps, warnings = _observed_edges_and_persistence(
        observed_df,
        uid_col=uid_name,
        datetime_col=datetime_name,
        location_mode=location_mode,
        location_col=location_col,
        h3_resolution=h3_resolution,
        max_group_size=max_group_size,
    )
    block, block_warnings = _validation_block(
        comparison="observed_vs_random",
        source_label="observed",
        source_graph=graph,
        source_persistence=persistence,
        time_steps=time_steps,
        source_kind="observed_daily_copresence",
        random_seed=seed,
        source_sidecar=None,
    )
    return block, [*warnings, *block_warnings]


def build_network_validation(
    synthetic_path: str | Path,
    *,
    observed_df: pd.DataFrame | None = None,
    observed_uid_col: str | None = None,
    observed_datetime_col: str | None = None,
    enabled: bool = True,
    synthetic_enabled: bool = True,
    observed_enabled: bool = False,
    location_mode: str = "auto",
    location_col: str | None = None,
    h3_resolution: int = 9,
    max_group_size: int = 200,
    seed: int = 42,
) -> tuple[dict[str, Any] | None, list[str]]:
    if not enabled:
        return None, []

    payload: dict[str, Any] = {}
    warnings: list[str] = []
    if synthetic_enabled:
        block, block_warnings = _synthetic_validation_block(synthetic_path, seed=seed)
        if block is not None:
            payload["synthetic_vs_random"] = block
        warnings.extend(f"synthetic_vs_random: {warning}" for warning in block_warnings)

    if observed_enabled:
        if observed_df is None:
            warnings.append("observed_vs_random: observed dataframe unavailable")
        else:
            block, block_warnings = _observed_validation_block(
                observed_df,
                uid_col=observed_uid_col,
                datetime_col=observed_datetime_col,
                location_mode=location_mode,
                location_col=location_col,
                h3_resolution=h3_resolution,
                max_group_size=max_group_size,
                seed=seed,
            )
            if block is not None:
                payload["observed_vs_random"] = block
            warnings.extend(f"observed_vs_random: {warning}" for warning in block_warnings)

    return (payload or None), warnings
