from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import fastmob
import numpy as np
import polars as pl
import pyarrow.compute as pc
from fastmob.core import Locations, Staypoints
from fastmob.social import co_presence_graph_from_staypoints

from citybehavex.reports._graph_metrics import (
    NetworkGraph,
    clustering_coefficients,
    degree_preserving_random_graph,
    distribution_summary,
    graph_from_edges,
    random_persistence,
    safe_wasserstein,
    topological_overlap,
)
from citybehavex.simulation.core import social_network_sidecar_path

__all__ = [
    "NetworkGraph",
    "graph_from_edges",
    "clustering_coefficients",
    "topological_overlap",
    "degree_preserving_random_graph",
    "encounters_sidecar_path",
    "build_network_validation",
    "build_observed_pair_network_validation",
    "social_network_sidecar_path",
]

NETWORK_METRIC_LABELS = {
    "degree": "Degree",
    "clustering_coefficient": "Clustering coefficient",
    "edge_persistence": "Edge persistence",
    "topological_overlap": "Topological overlap",
}

_DATETIME_CANDIDATES = ["datetime", "start_timestamp", "timestamp", "check-in_time", "start_time", "time", "date"]
_END_DATETIME_CANDIDATES = ["end_timestamp", "finished_at", "end_time", "end_datetime"]
_DURATION_MINUTE_CANDIDATES = ["duration_minutes", "dwell_minutes", "dwell_time_min", "duration_min"]
_UID_CANDIDATES = ["uid", "user_id", "user", "agent_id", "userid"]
_LAT_CANDIDATES = ["lat", "latitude"]
_LNG_CANDIDATES = ["lng", "lon", "longitude", "long"]
_LOCATION_CANDIDATES = ["location_id", "tile_id", "venueId", "venue_id", "area", "location"]


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


def _metric_bundle(
    graph: NetworkGraph,
    persistence: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "degree": graph.degrees().astype(float),
        "clustering_coefficient": clustering_coefficients(graph),
        "edge_persistence": persistence,
        "topological_overlap": topological_overlap(graph),
    }


def _recast_friends_edges_and_persistence(
    staypoints: Staypoints,
    locations: Locations,
    uid_values: pl.Series,
    *,
    seed: int = 42,
) -> tuple[NetworkGraph, np.ndarray, int]:
    """Build a RECAST ``FRIENDS``-only graph from raw staypoints.

    Used for both the synthetic and observed sides so the comparison is
    methodologically symmetric (same classifier, same thresholds, same
    interval-overlap rule on both sides) -- this replaced comparing raw
    co-presence (any spatiotemporal overlap counts as an edge, which
    conflates genuine repeated contact with incidental crowding and made the
    degree metric ~20-120x off between synthetic and real data) against a
    RECAST-filtered real side, or a pre-assigned synthetic social-network
    sidecar against a RECAST-filtered real side -- neither was apples-to-apples.

    ``RecastResult.source_users``/``target_users`` are the *original* uid
    values (fastmob maps its internal compact node indices back to them for
    interpretability), not compact ``0..N-1`` graph indices -- remap through
    the same sorted-unique factorization fastmob's internal ``_prepare()``
    uses internally, matching ``co_presence_graph_from_staypoints``'s node
    ordering. Getting this wrong silently drops any edge whose endpoint uid
    value is >= node_count (found the hard way: real check-in uids aren't
    sequential, so this dropped ~75% of real friend edges).
    """
    from fastmob.social.recast import RecastClass, recast_from_staypoints

    unique_sorted_uids = uid_values.unique().sort().to_list()
    uid_to_index = {u: i for i, u in enumerate(unique_sorted_uids)}
    node_count = len(unique_sorted_uids)

    recast = recast_from_staypoints(staypoints, locations, seed=seed)
    mask = recast.mask(RecastClass.FRIENDS)
    src = pc.filter(recast.source_users, mask).to_pylist()
    dst = pc.filter(recast.target_users, mask).to_pylist()
    persistence = pc.filter(recast.edge_persistence, mask).to_numpy(zero_copy_only=False)
    edges = [(uid_to_index[u], uid_to_index[v]) for u, v in zip(src, dst)]
    graph = graph_from_edges(node_count, edges)
    return graph, np.asarray(persistence, dtype=float), int(recast.time_steps)


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
    random_baseline: bool = True,
) -> tuple[dict[str, Any] | None, list[str], dict[str, np.ndarray]]:
    warnings: list[str] = []
    source_metrics = _metric_bundle(source_graph, source_persistence)

    if not random_baseline:
        # The ablation table only reads the synthetic_vs_observed /
        # observed_vs_observed comparisons (see aggregate_ablation_results.py's
        # NETWORK_METRICS lookup) -- it never reads this comparison's own
        # payload. Building the degree-preserving random graph and computing
        # clustering_coefficients/topological_overlap on it is roughly half
        # the total network-validation cost (measured: ~56% on yjmob's dense
        # co-presence graph), so skip it entirely when the caller doesn't
        # need this block, keeping only source_metrics (which IS needed
        # downstream for synthetic_vs_observed).
        return None, warnings, source_metrics

    degrees = source_graph.degrees().astype(float)
    random_graph = degree_preserving_random_graph(degrees, seed=random_seed)
    random_pers = random_persistence(
        random_graph,
        degrees,
        time_steps=time_steps,
        seed=random_seed + 1,
    )

    random_metrics = _metric_bundle(random_graph, random_pers)
    wasserstein = {
        name: safe_wasserstein(source_metrics[name], random_metrics[name])
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
                    name: distribution_summary(values)
                    for name, values in source_metrics.items()
                },
                "random": {
                    name: distribution_summary(values)
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
        source_metrics,
    )


def _metric_wasserstein_block(
    *,
    comparison: str,
    left_label: str,
    left_metrics: dict[str, np.ndarray],
    right_label: str,
    right_metrics: dict[str, np.ndarray],
) -> tuple[dict[str, Any], list[str]]:
    wasserstein = {
        name: safe_wasserstein(left_metrics[name], right_metrics[name])
        for name in NETWORK_METRIC_LABELS
    }
    warnings = [
        f"{NETWORK_METRIC_LABELS[name]} distribution is empty; Wasserstein unavailable"
        for name, value in wasserstein.items()
        if value is None
    ]
    return (
        {
            "comparison": comparison,
            "wasserstein": wasserstein,
            "distributions": {
                left_label: {
                    name: distribution_summary(values)
                    for name, values in left_metrics.items()
                },
                right_label: {
                    name: distribution_summary(values)
                    for name, values in right_metrics.items()
                },
            },
        },
        warnings,
    )


def _synthetic_validation_block(
    synthetic_df: pl.DataFrame,
    *,
    seed: int = 42,
    random_baseline: bool = True,
) -> tuple[dict[str, Any] | None, list[str], dict[str, np.ndarray] | None]:
    """Build the synthetic side from the RAW SIMULATED TRAJECTORY (arrival/
    departure per stop), through the exact same RECAST FRIENDS pipeline as
    the observed side -- not the simulation's own pre-assigned social-network
    sidecar (``social.*``-config-driven, a designed graph) compared against
    an empirically-derived real graph, which was an apples-to-oranges
    methodology mismatch on top of the raw-co-presence-vs-real mismatch.
    """
    uid_col = _detect_column(synthetic_df, _UID_CANDIDATES)
    lat_col = _detect_column(synthetic_df, _LAT_CANDIDATES)
    lng_col = _detect_column(synthetic_df, _LNG_CANDIDATES)
    location_col = _detect_column(synthetic_df, _LOCATION_CANDIDATES)
    start_col = _detect_column(synthetic_df, ["arrival", *_DATETIME_CANDIDATES])
    end_col = _detect_column(synthetic_df, ["departure", *_END_DATETIME_CANDIDATES])
    if uid_col is None or lat_col is None or lng_col is None or start_col is None or end_col is None:
        return None, ["synthetic network validation requires uid/lat/lng/arrival/departure-like columns"], None

    work = synthetic_df.select(
        pl.col(uid_col).alias("uid"),
        (pl.col(location_col).cast(pl.Utf8) if location_col else pl.lit(None, dtype=pl.Utf8)).alias("location_id"),
        pl.col(lat_col).cast(pl.Float64, strict=False).alias("lat"),
        pl.col(lng_col).cast(pl.Float64, strict=False).alias("lng"),
        _to_datetime(synthetic_df[start_col]).alias("started_at"),
        _to_datetime(synthetic_df[end_col]).alias("finished_at"),
    ).drop_nulls(subset=["uid", "started_at", "finished_at"])

    if location_col is None:
        work = work.with_columns(_h3_cells(work["lat"], work["lng"], 9).cast(pl.Utf8).alias("location_id"))
    work = work.drop_nulls(subset=["location_id"]).filter(pl.col("finished_at") >= pl.col("started_at"))
    if work.is_empty():
        return None, ["synthetic network validation: no valid rows after filtering"], None

    location_catalogue = work.select(
        "location_id", pl.col("lat").alias("center_lat"), pl.col("lng").alias("center_lng")
    ).unique(subset=["location_id"], keep="first", maintain_order=True)
    locations = Locations(location_catalogue, scope="global")
    staypoints = Staypoints(
        work,
        uid_col="uid",
        lat_col="lat",
        lng_col="lng",
        started_at_col="started_at",
        finished_at_col="finished_at",
    )
    graph, persistence, time_steps = _recast_friends_edges_and_persistence(staypoints, locations, work["uid"], seed=seed)

    block, block_warnings, metrics = _validation_block(
        comparison="synthetic_vs_random",
        source_label="synthetic",
        source_graph=graph,
        source_persistence=persistence,
        time_steps=time_steps,
        source_kind="synthetic_recast_friends",
        random_seed=seed,
        source_sidecar=None,
        random_baseline=random_baseline,
    )
    return block, block_warnings, metrics


def _h3_cells(lat: pl.Series, lng: pl.Series, resolution: int) -> pl.Series:
    """Vectorized lat/lng -> H3 cell index (nullable ``UInt64``), via
    fastmob's Rust-accelerated ``latlng_to_h3``. Only used as a
    groupby/comparison key here, never displayed, so the numeric form is fine.
    """
    tmp = pl.DataFrame({"lat": lat, "lng": lng})
    result = fastmob.preprocessing.latlng_to_h3(tmp, resolution, lat_col="lat", lng_col="lng", output_col="h3_cell")
    return result["h3_cell"]


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


def _to_datetime(col: pl.Series) -> pl.Series:
    if col.dtype == pl.Utf8:
        return col.str.to_datetime(strict=False)
    if not isinstance(col.dtype, pl.Datetime):
        return col.cast(pl.Datetime, strict=False)
    return col


def _observed_edges_and_persistence(
    df: pl.DataFrame,
    *,
    uid_col: str,
    datetime_col: str,
    location_mode: str,
    location_col: str | None,
    h3_resolution: int,
) -> tuple[NetworkGraph, np.ndarray, int, list[str]]:
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
    lat_col = _detect_column(df, _LAT_CANDIDATES)
    lng_col = _detect_column(df, _LNG_CANDIDATES)
    lat = (
        df[lat_col].cast(pl.Float64, strict=False).fill_nan(0.0).fill_null(0.0).clip(-90.0, 90.0)
        if lat_col is not None
        else pl.Series("_lat", np.zeros(len(df), dtype=np.float64))
    )
    lng = (
        df[lng_col].cast(pl.Float64, strict=False).fill_nan(0.0).fill_null(0.0).clip(-180.0, 180.0)
        if lng_col is not None
        else pl.Series("_lng", np.zeros(len(df), dtype=np.float64))
    )
    end_col = _detect_column(df, _END_DATETIME_CANDIDATES)
    duration_col = _detect_column(df, _DURATION_MINUTE_CANDIDATES)
    columns: dict[str, pl.Series] = {
        "uid": df[uid_col],
        "started_at": _to_datetime(df[datetime_col]),
        "location_id": location,
        "lat": lat,
        "lng": lng,
    }
    if end_col is not None:
        columns["finished_at"] = _to_datetime(df[end_col])
    elif duration_col is not None:
        columns["duration_minutes"] = df[duration_col].cast(pl.Float64, strict=False)
    work = pl.DataFrame(columns).drop_nulls(subset=["uid", "started_at", "location_id"])
    if work.is_empty():
        return _empty_graph(0), np.asarray([], dtype=float), 0, [f"observed network has no valid rows using {location_source}"]

    work = work.sort(["uid", "started_at"]).with_columns(
        pl.col("started_at").shift(-1).over("uid").alias("_inferred_finished_at")
    )
    work = work.with_columns(
        pl.col("_inferred_finished_at")
        .fill_null(pl.col("started_at").dt.truncate("1d") + pl.duration(days=1))
        .alias("_inferred_finished_at")
    )
    if end_col is None:
        if duration_col is None:
            work = work.with_columns(pl.col("_inferred_finished_at").alias("finished_at"))
        else:
            explicit_duration_end = pl.col("started_at") + pl.duration(
                microseconds=(pl.col("duration_minutes") * 60_000_000).cast(pl.Int64)
            )
            work = work.with_columns(
                pl.when(pl.col("duration_minutes").is_finite() & (pl.col("duration_minutes") >= 0.0))
                .then(explicit_duration_end)
                .otherwise(pl.col("_inferred_finished_at"))
                .alias("finished_at")
            )
    else:
        work = work.with_columns(
            pl.coalesce("finished_at", "_inferred_finished_at").alias("finished_at")
        )
    work = work.drop("_inferred_finished_at")
    if duration_col is not None and end_col is None:
        work = work.drop("duration_minutes")
    work = work.filter(pl.col("finished_at").is_not_null() & (pl.col("finished_at") >= pl.col("started_at")))
    if work.is_empty():
        return _empty_graph(0), np.asarray([], dtype=float), 0, [f"observed network has no valid intervals using {location_source}"]

    location_catalogue = (
        work.select(
            "location_id",
            pl.col("lat").alias("center_lat"),
            pl.col("lng").alias("center_lng"),
        )
        .unique(subset=["location_id"], keep="first", maintain_order=True)
    )
    locations = Locations(location_catalogue, scope="global")
    staypoints = Staypoints(
        work,
        uid_col="uid",
        lat_col="lat",
        lng_col="lng",
        started_at_col="started_at",
        finished_at_col="finished_at",
    )
    graph, persistence, time_steps = _recast_friends_edges_and_persistence(staypoints, locations, work["uid"])
    return (graph, persistence, time_steps, [])


def observed_network_validation_cache_path(
    real_path: str | Path,
    *,
    location_mode: str,
    location_col: str | None,
    h3_resolution: int,
    seed: int,
) -> Path:
    """Deterministic cache path for the observed-side co-presence metrics.

    The observed comparison dataset (and these params) are identical across
    every ablation variant/round for a given dataset, but
    ``_observed_validation_block`` used to recompute the whole co-presence
    graph plus clustering-coefficient/topological-overlap from scratch on
    every single ``citybehavex report`` call -- for yjmob's dense graph
    that's ~13 minutes of redundant work repeated on every one of 15 combos.
    Cache keyed by path+params so it's computed once and reused.
    """
    p = Path(real_path)
    # "recastfriends" tags the classification method into the cache key so a
    # cache built under the old raw-co-presence method (pre-RECAST-friends
    # switch) is never silently reused -- it'd have completely different
    # (much smaller) edge counts and degree values.
    suffix = f"_nv_cache_recastfriends_h3{h3_resolution}_{location_mode}_{location_col or 'auto'}_seed{seed}.npz"
    return p.with_name(p.stem + suffix)


def _load_observed_metrics_cache(cache_path: Path) -> tuple[dict[str, np.ndarray], int] | None:
    if not cache_path.exists():
        return None
    data = np.load(cache_path)
    metrics = {
        "degree": data["degree"],
        "clustering_coefficient": data["clustering_coefficient"],
        "edge_persistence": data["edge_persistence"],
        "topological_overlap": data["topological_overlap"],
    }
    return metrics, int(data["time_steps"])


def _save_observed_metrics_cache(cache_path: Path, metrics: dict[str, np.ndarray], time_steps: int) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Name already ends in .npz so numpy won't append a second .npz suffix;
    # write-then-rename so a concurrent reader (another combo's report
    # process racing to build the same cache) never sees a partial file.
    tmp_path = cache_path.parent / f"{cache_path.stem}.tmp{cache_path.suffix}"
    np.savez_compressed(
        tmp_path,
        degree=metrics["degree"],
        clustering_coefficient=metrics["clustering_coefficient"],
        edge_persistence=metrics["edge_persistence"],
        topological_overlap=metrics["topological_overlap"],
        time_steps=np.asarray(time_steps),
    )
    tmp_path.replace(cache_path)


def _observed_validation_block(
    observed_df: pl.DataFrame,
    *,
    uid_col: str | None = None,
    datetime_col: str | None = None,
    location_mode: str = "auto",
    location_col: str | None = None,
    h3_resolution: int = 9,
    seed: int = 42,
    random_baseline: bool = True,
    cache_path: Path | None = None,
) -> tuple[dict[str, Any] | None, list[str], dict[str, np.ndarray] | None]:
    uid_name = uid_col or _detect_column(observed_df, _UID_CANDIDATES)
    datetime_name = datetime_col or _detect_column(observed_df, _DATETIME_CANDIDATES)
    if uid_name is None or datetime_name is None:
        return None, ["observed network validation requires user and datetime columns"], None

    cached = _load_observed_metrics_cache(cache_path) if cache_path is not None else None
    if cached is not None:
        metrics, time_steps = cached
        warnings: list[str] = []
        graph = persistence = None
    else:
        graph, persistence, time_steps, warnings = _observed_edges_and_persistence(
            observed_df,
            uid_col=uid_name,
            datetime_col=datetime_name,
            location_mode=location_mode,
            location_col=location_col,
            h3_resolution=h3_resolution,
        )
        metrics = _metric_bundle(graph, persistence)
        if cache_path is not None:
            _save_observed_metrics_cache(cache_path, metrics, time_steps)

    if not random_baseline:
        return None, warnings, metrics

    if graph is None:
        # Cache hit: no full graph object available (only its derived
        # per-node/per-edge metrics were cached), so the random-baseline
        # comparison's source/random network visualization sub-blocks can't
        # be built. This combination isn't used by the ablation pipeline
        # (which always disables random_baseline) -- build the wasserstein
        # numbers from the cached metrics' degree sequence without them.
        degrees = metrics["degree"]
        random_graph = degree_preserving_random_graph(degrees, seed=seed)
        random_pers = random_persistence(random_graph, degrees, time_steps=time_steps, seed=seed + 1)
        random_metrics = _metric_bundle(random_graph, random_pers)
        wasserstein = {
            name: safe_wasserstein(metrics[name], random_metrics[name]) for name in NETWORK_METRIC_LABELS
        }
        block = {
            "comparison": "observed_vs_random",
            "random_model": "degree_preserving_rnd",
            "wasserstein": wasserstein,
            "distributions": {
                "observed": {name: distribution_summary(v) for name, v in metrics.items()},
                "random": {name: distribution_summary(v) for name, v in random_metrics.items()},
            },
        }
        return block, warnings, metrics

    block, block_warnings, metrics = _validation_block(
        comparison="observed_vs_random",
        source_label="observed",
        source_graph=graph,
        source_persistence=persistence,
        time_steps=time_steps,
        source_kind="observed_daily_copresence",
        random_seed=seed,
        source_sidecar=None,
        random_baseline=random_baseline,
    )
    return block, [*warnings, *block_warnings], metrics


def build_observed_pair_network_validation(
    df_a: pl.DataFrame,
    df_b: pl.DataFrame,
    *,
    label_a: str = "half_a",
    label_b: str = "half_b",
    uid_col: str | None = None,
    datetime_col: str | None = None,
    location_mode: str = "auto",
    location_col: str | None = None,
    h3_resolution: int = 9,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Compare two real RECAST-friends networks (e.g. a population-halved
    Ref. split) directly against each other -- no random null graph. Reuses
    ``_observed_edges_and_persistence`` (RECAST FRIENDS classification, same
    as ``_observed_validation_block``), just diffing two real graphs' metric
    distributions instead of diffing one real graph against a synthetic
    degree-preserving random baseline (``degree_preserving_random_graph``,
    which is a slow pure-Python O(n^2) generator not needed here).
    """
    uid_name = uid_col or _detect_column(df_a, _UID_CANDIDATES)
    datetime_name = datetime_col or _detect_column(df_a, _DATETIME_CANDIDATES)
    if uid_name is None or datetime_name is None:
        return None, ["observed-pair network validation requires user and datetime columns"]

    graph_a, persistence_a, _, warnings_a = _observed_edges_and_persistence(
        df_a,
        uid_col=uid_name,
        datetime_col=datetime_name,
        location_mode=location_mode,
        location_col=location_col,
        h3_resolution=h3_resolution,
    )
    graph_b, persistence_b, _, warnings_b = _observed_edges_and_persistence(
        df_b,
        uid_col=uid_name,
        datetime_col=datetime_name,
        location_mode=location_mode,
        location_col=location_col,
        h3_resolution=h3_resolution,
    )
    metrics_a = _metric_bundle(graph_a, persistence_a)
    metrics_b = _metric_bundle(graph_b, persistence_b)
    block, block_warnings = _metric_wasserstein_block(
        comparison="observed_vs_observed",
        left_label=label_a,
        left_metrics=metrics_a,
        right_label=label_b,
        right_metrics=metrics_b,
    )
    return block, [*warnings_a, *warnings_b, *block_warnings]


def build_network_validation(
    synthetic_df: pl.DataFrame,
    *,
    observed_df: pl.DataFrame | None = None,
    observed_uid_col: str | None = None,
    observed_datetime_col: str | None = None,
    observed_source_path: str | Path | None = None,
    enabled: bool = True,
    synthetic_enabled: bool = True,
    observed_enabled: bool = False,
    location_mode: str = "auto",
    location_col: str | None = None,
    h3_resolution: int = 9,
    seed: int = 42,
    random_baseline: bool = True,
    cache_observed: bool = True,
) -> tuple[dict[str, Any] | None, list[str]]:
    if not enabled:
        return None, []

    payload: dict[str, Any] = {}
    warnings: list[str] = []
    synthetic_metrics: dict[str, np.ndarray] | None = None
    if synthetic_enabled:
        block, block_warnings, synthetic_metrics = _synthetic_validation_block(
            synthetic_df, seed=seed, random_baseline=random_baseline
        )
        if block is not None:
            payload["synthetic_vs_random"] = block
        warnings.extend(f"synthetic_vs_random: {warning}" for warning in block_warnings)

    if observed_enabled:
        if observed_df is None:
            warnings.append("observed_vs_random: observed dataframe unavailable")
        else:
            cache_path = (
                observed_network_validation_cache_path(
                    observed_source_path,
                    location_mode=location_mode,
                    location_col=location_col,
                    h3_resolution=h3_resolution,
                    seed=seed,
                )
                if cache_observed and observed_source_path is not None
                else None
            )
            block, block_warnings, observed_metrics = _observed_validation_block(
                observed_df,
                uid_col=observed_uid_col,
                datetime_col=observed_datetime_col,
                location_mode=location_mode,
                location_col=location_col,
                h3_resolution=h3_resolution,
                seed=seed,
                random_baseline=random_baseline,
                cache_path=cache_path,
            )
            if block is not None:
                payload["observed_vs_random"] = block
            warnings.extend(f"observed_vs_random: {warning}" for warning in block_warnings)
            if synthetic_metrics is not None and observed_metrics is not None:
                observed_block, observed_warnings = _metric_wasserstein_block(
                    comparison="synthetic_vs_observed",
                    left_label="synthetic",
                    left_metrics=synthetic_metrics,
                    right_label="observed",
                    right_metrics=observed_metrics,
                )
                payload["synthetic_vs_observed"] = observed_block
                warnings.extend(
                    f"synthetic_vs_observed: {warning}" for warning in observed_warnings
                )

    return (payload or None), warnings
