from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.stats import wasserstein_distance

from citybehavex import _core as _cbx_core
from citybehavex.simulation.core import social_network_sidecar_path

_H3_INVALID_CELL = np.uint64(2**64 - 1)


def _h3_cells(lat: pl.Series, lng: pl.Series, resolution: int) -> pl.Series:
    """Vectorized lat/lng -> H3 cell index (nullable ``UInt64``), via the Rust
    extension instead of a per-row ``h3.latlng_to_cell`` Python loop -- see
    the twin helper in ``citybehavex.reports.comparison`` (duplicated rather
    than imported to avoid a circular import between the two report
    modules). Only used as a groupby/comparison key here, never displayed,
    so the numeric form is fine.
    """
    lat_arr = lat.cast(pl.Float64, strict=False).to_numpy()
    lng_arr = lng.cast(pl.Float64, strict=False).to_numpy()
    cells = _cbx_core.batch_latlng_to_cells(lat_arr, lng_arr, resolution)
    result = pl.Series(cells, dtype=pl.UInt64)
    invalid = pl.Series(cells == _H3_INVALID_CELL)
    if invalid.any():
        result = result.set(invalid, None)
    return result


NETWORK_METRIC_LABELS = {
    "degree": "Degree",
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
    """Undirected graph as a plain edge list (``u < v``, sorted, deduped),
    not per-node adjacency ``set``s -- for the observed co-presence graph
    (tens of millions of edges for shanghai/yjmob), materializing a Python
    `set`/`set`-of-`set`s costs seconds of object construction and gigabytes
    of memory on its own, on top of the O(sum of degree^2) metric loops that
    used to run against it. ``clustering_coefficients``/``topological_overlap``
    consume these arrays directly via the Rust extension; ``.edges`` is a
    convenience for small graphs (tests, the synthetic path) and should not
    be used in a hot path over the observed graph.
    """

    node_count: int
    edge_from: np.ndarray  # uint32[E], edge_from < edge_to elementwise
    edge_to: np.ndarray  # uint32[E]

    @property
    def edge_count(self) -> int:
        return int(self.edge_from.shape[0])

    @property
    def edges(self) -> set[tuple[int, int]]:
        return set(zip(self.edge_from.tolist(), self.edge_to.tolist()))

    def degrees(self) -> np.ndarray:
        return np.bincount(
            np.concatenate([self.edge_from, self.edge_to]),
            minlength=self.node_count,
        ) if self.edge_from.size else np.zeros(self.node_count, dtype=np.int64)


def encounters_sidecar_path(output_path: str | Path) -> Path:
    p = Path(output_path)
    return p.with_name(f"{p.stem}_encounters{p.suffix}")


def _detect_column(df: pl.DataFrame, candidates: list[str]) -> str | None:
    cols_lower = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in cols_lower:
            return cols_lower[candidate.lower()]
    return None


def _empty_graph(node_count: int) -> NetworkGraph:
    empty = np.empty(0, dtype=np.uint32)
    return NetworkGraph(node_count=node_count, edge_from=empty, edge_to=empty)


def _normal_edge(a: Any, b: Any, node_count: int) -> tuple[int, int] | None:
    try:
        u, v = int(a), int(b)
    except (TypeError, ValueError):
        return None
    if u == v or u < 0 or v < 0 or u >= node_count or v >= node_count:
        return None
    return (u, v) if u < v else (v, u)


def graph_from_edges(node_count: int, edges: set[tuple[int, int]]) -> NetworkGraph:
    """Build a graph from a small/moderate edge collection (synthetic-scale
    social + encounter graphs, and test fixtures) -- normalizes/dedupes via
    a Python ``set`` since that's cheap at this scale. The observed
    co-presence graph is built directly from Rust arrays instead
    (``_observed_edges_and_persistence``), bypassing this entirely.
    """
    normalized: set[tuple[int, int]] = set()
    for u, v in edges:
        edge = _normal_edge(u, v, node_count)
        if edge is not None:
            normalized.add(edge)
    if not normalized:
        return _empty_graph(node_count)
    ordered = sorted(normalized)
    edge_from = np.ascontiguousarray([u for u, _ in ordered], dtype=np.uint32)
    edge_to = np.ascontiguousarray([v for _, v in ordered], dtype=np.uint32)
    return NetworkGraph(node_count=node_count, edge_from=edge_from, edge_to=edge_to)


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
    encounters: pl.DataFrame,
    *,
    node_count: int,
) -> tuple[set[tuple[int, int]], np.ndarray, int]:
    required = {"agent", "contact", "ts"}
    missing = required - set(encounters.columns)
    if missing:
        raise ValueError(f"encounters table missing columns: {', '.join(sorted(missing))}")
    if encounters.is_empty():
        return set(), np.asarray([], dtype=float), 0

    work = encounters.select(["agent", "contact", "ts"]).drop_nulls()
    if work.is_empty():
        return set(), np.asarray([], dtype=float), 0
    work = work.with_columns(
        pl.col("agent").cast(pl.Float64, strict=False),
        pl.col("contact").cast(pl.Float64, strict=False),
    ).drop_nulls(subset=["agent", "contact", "ts"])
    if work.is_empty():
        return set(), np.asarray([], dtype=float), 0

    time_steps = work["ts"].n_unique()
    if time_steps <= 0:
        return set(), np.asarray([], dtype=float), 0

    pair_steps: dict[tuple[int, int], set[Any]] = {}
    for agent, contact, ts in work.iter_rows():
        edge = _normal_edge(agent, contact, node_count)
        if edge is None:
            continue
        pair_steps.setdefault(edge, set()).add(ts)

    edges = set(pair_steps)
    persistence = np.asarray(
        [len(steps) / time_steps for steps in pair_steps.values()],
        dtype=float,
    )
    return edges, persistence, time_steps


def _graph_metrics(graph: NetworkGraph) -> tuple[np.ndarray, np.ndarray]:
    """Per-node clustering coefficient and per-edge topological overlap
    (Jaccard similarity of endpoint neighborhoods), computed once by the
    Rust extension and shared by both public accessors below (and by
    ``_metric_bundle``, which needs both) -- see
    ``citybehavex-py/src/simulation_core/network_graph.rs`` for why this
    used to be a pure-Python `O(sum of degree^2)` cost (measured: ~51
    minutes extrapolated for shanghai's dense observed co-presence graph).
    """
    return _cbx_core.graph_metrics(graph.node_count, graph.edge_from, graph.edge_to)


def clustering_coefficients(graph: NetworkGraph) -> np.ndarray:
    clustering, _overlap = _graph_metrics(graph)
    return clustering


def topological_overlap(graph: NetworkGraph) -> np.ndarray:
    _clustering, overlap = _graph_metrics(graph)
    return overlap


def _distribution_summary(values: np.ndarray) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "mean": None, "median": None, "std": None, "p10": None, "p90": None}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


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
    # Collected as arrays of (i, j) pairs per outer iteration rather than
    # inserted into a Python set one at a time -- each i produces at most
    # n-1-i pairs and, since j always comes from i+1.., no (i, j) pair can
    # recur across iterations, so no dedup is needed, only a final sort.
    from_chunks: list[np.ndarray] = []
    to_chunks: list[np.ndarray] = []
    for i in range(n - 1):
        if deg[i] <= 0:
            continue
        probs = np.clip((deg[i] * deg[i + 1 :]) / total_degree, 0.0, 1.0)
        if probs.size == 0:
            continue
        offsets = np.flatnonzero(rng.random(probs.size) < probs)
        if offsets.size == 0:
            continue
        from_chunks.append(np.full(offsets.size, i, dtype=np.uint32))
        to_chunks.append((i + 1 + offsets).astype(np.uint32))

    if not from_chunks:
        return _empty_graph(n)
    edge_from = np.concatenate(from_chunks)
    edge_to = np.concatenate(to_chunks)
    order = np.lexsort((edge_to, edge_from))
    return NetworkGraph(node_count=n, edge_from=edge_from[order], edge_to=edge_to[order])


def _random_persistence(
    graph: NetworkGraph,
    degrees: np.ndarray,
    *,
    time_steps: int,
    seed: int,
) -> np.ndarray:
    if time_steps <= 0 or graph.edge_count == 0:
        return np.asarray([], dtype=float)
    deg = np.asarray(degrees, dtype=float)
    total_degree = float(deg.sum())
    if total_degree <= 0:
        return np.asarray([], dtype=float)
    rng = np.random.default_rng(seed)
    probs = np.clip((deg[graph.edge_from] * deg[graph.edge_to]) / total_degree, 0.0, 1.0)
    return rng.binomial(time_steps, probs) / time_steps


def _metric_bundle(
    graph: NetworkGraph,
    persistence: np.ndarray,
) -> dict[str, np.ndarray]:
    clustering, overlap = _graph_metrics(graph)
    return {
        "degree": graph.degrees().astype(float),
        "clustering_coefficient": clustering,
        "edge_persistence": persistence,
        "topological_overlap": overlap,
    }


_MAX_VISUALIZED_EDGES = 20_000


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
    degrees = graph.degrees()
    max_degree = int(degrees.max()) if degrees.size else 0
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

    # A force-directed graph render (and the JSON payload carrying it) isn't
    # viable at the observed co-presence graph's real scale (tens of
    # millions of edges for shanghai/yjmob) -- cap what's actually sent for
    # visualization while keeping edge_count/degrees/metrics reflecting the
    # true full graph. Sampled rather than truncated to the first N so the
    # visualization isn't biased toward whatever ordering the edges happen
    # to be in.
    edge_count = graph.edge_count
    if edge_count > _MAX_VISUALIZED_EDGES:
        sample_idx = rng.choice(edge_count, size=_MAX_VISUALIZED_EDGES, replace=False)
        sample_idx.sort()
        edge_from, edge_to = graph.edge_from[sample_idx], graph.edge_to[sample_idx]
    else:
        edge_from, edge_to = graph.edge_from, graph.edge_to

    return {
        "kind": kind,
        "node_count": graph.node_count,
        "edge_count": edge_count,
        "layout": source_sidecar.get("layout", "source_layout"),
        "directed": False,
        "social_graph_k": source_sidecar.get("social_graph_k", 0),
        "nodes": nodes,
        "edges": [[int(u), int(v), 1.0] for u, v in zip(edge_from.tolist(), edge_to.tolist())],
        "edges_sampled": edge_count > _MAX_VISUALIZED_EDGES,
        "degrees": degrees.tolist(),
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
    degrees = source_graph.degrees().astype(float)
    random_graph = degree_preserving_random_graph(degrees, seed=random_seed)
    random_persistence = _random_persistence(
        random_graph,
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
            pl.read_parquet(enc_path),
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
    df: pl.DataFrame,
    *,
    location_mode: str,
    location_col: str | None,
    h3_resolution: int,
) -> tuple[pl.Series, str]:
    if location_mode not in {"auto", "location_col", "h3"}:
        raise ValueError(f"unsupported network validation location_mode: {location_mode}")

    chosen = location_col if location_col and location_col in df.columns else None
    if chosen is None and location_mode == "auto":
        chosen = _detect_column(df, _LOCATION_CANDIDATES)
    if location_mode == "location_col" and chosen is None:
        raise ValueError(f"network validation location_col not found: {location_col!r}")
    if chosen is not None and location_mode != "h3":
        return df[chosen].cast(pl.Utf8), chosen

    lat_col = _detect_column(df, _LAT_CANDIDATES)
    lng_col = _detect_column(df, _LNG_CANDIDATES)
    if lat_col is None or lng_col is None:
        raise ValueError("h3 network validation requires latitude/longitude columns")
    lat = df[lat_col].cast(pl.Float64, strict=False)
    lng = df[lng_col].cast(pl.Float64, strict=False)
    valid = lat.is_between(-90, 90) & lng.is_between(-180, 180)
    cells = pl.Series([None] * len(df), dtype=pl.UInt64)
    valid_idx = valid.arg_true()
    if len(valid_idx):
        computed = _h3_cells(lat.filter(valid), lng.filter(valid), int(h3_resolution))
        cells = cells.scatter(valid_idx, computed)
    return cells, f"h3_{h3_resolution}"


def _to_day(col: pl.Series) -> pl.Series:
    """Coerce a datetime-ish column (string or already-parsed) to midnight-truncated datetimes."""
    if col.dtype == pl.Utf8:
        col = col.str.to_datetime(strict=False)
    elif not isinstance(col.dtype, pl.Datetime):
        col = col.cast(pl.Datetime, strict=False)
    return col.dt.truncate("1d")


def _observed_edges_and_persistence(
    df: pl.DataFrame,
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

    location, location_source = _resolve_observed_location(
        df,
        location_mode=location_mode,
        location_col=location_col,
        h3_resolution=h3_resolution,
    )
    work = pl.DataFrame(
        {"uid": df[uid_col], "day": _to_day(df[datetime_col]), "location": location}
    ).drop_nulls(subset=["uid", "day", "location"])
    if work.is_empty():
        return _empty_graph(0), np.asarray([], dtype=float), 0, [f"observed network has no valid rows using {location_source}"]

    uid_map = work.select(pl.col("uid").unique().sort()).with_row_index("node")
    work = work.join(uid_map, on="uid", how="left")
    node_count = uid_map.height

    day_map = work.select(pl.col("day").unique().sort()).with_row_index("day_code")
    work = work.join(day_map, on="day", how="left")
    time_steps = day_map.height

    location_map = work.select(pl.col("location").unique()).with_row_index("location_code")
    work = work.join(location_map, on="location", how="left")

    dedup = work.unique(subset=["day_code", "location_code", "node"])
    # Pair generation + per-edge day-persistence via the Rust extension --
    # was an itertools.combinations loop into a dict[edge, set[day]]
    # (measured: 150s on shanghai's ~65M raw pair-instances, plus the
    # O(sum of degree^2) metric computation that followed from it).
    edge_from, edge_to, persistence, skipped_groups, skipped_rows = _cbx_core.build_co_presence_edges(
        dedup["day_code"].cast(pl.Int64).to_numpy(),
        dedup["location_code"].cast(pl.Int64).to_numpy(),
        dedup["node"].cast(pl.Int64).to_numpy(),
        max_group_size,
        time_steps,
    )
    graph = NetworkGraph(node_count=node_count, edge_from=edge_from, edge_to=edge_to)

    warnings: list[str] = []
    if skipped_groups:
        warnings.append(
            f"observed network skipped {skipped_groups} {location_source}/day groups "
            f"({skipped_rows} user-presences) larger than max_group_size={max_group_size}"
        )
    return graph, persistence, time_steps, warnings


def _observed_validation_block(
    observed_df: pl.DataFrame,
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
    observed_df: pl.DataFrame | None = None,
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
