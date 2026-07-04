from __future__ import annotations

import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import duckdb
import h3
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import skmob2
import typer
from skmob2.models import DensityEPR

from citybehavex.activities import activity_descriptions, activity_duration_arrays, build_eligibility_csr
from citybehavex.config import CityBehavExConfig
from citybehavex.embedding import embed_profiles, embed_texts
from citybehavex.llm_diaries import DiaryBatch, LLMStats, allocate_location_counts, fetch_diary_batch
from citybehavex.profiles import AgentProfile, generate_profiles, load_profiles, profile_to_narrative, profiles_to_frame
from citybehavex.roads import build_road_graph, snap_locations_to_graph
from citybehavex.schedules import (
    DdcrpAgentInfo,
    DiaryBank,
    build_ddcrp_diary,
    build_diary_bank,
    score_alignment_matrix,
)
from citybehavex.simulation.core import CoreTiming, simulate_agents, social_network_sidecar_path
from citybehavex.tessellation import build_poi_tessellation, build_tessellation, purpose_distribution

_WORK_SCORE_COLUMN = "work_score"


def _minmax(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if len(arr) == 0:
        return arr
    lo = float(arr.min())
    hi = float(arr.max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


def _lng_column(df: pd.DataFrame) -> str:
    return "lng" if "lng" in df.columns else "lon"


def _resolve_spatial_bounds(config: CityBehavExConfig, tessellation_df: pd.DataFrame) -> tuple[float, float, float, float]:
    sim = config.simulation
    tess = config.tessellation
    min_lon = sim.min_lon if sim.min_lon is not None else tess.min_lon
    min_lat = sim.min_lat if sim.min_lat is not None else tess.min_lat
    max_lon = sim.max_lon if sim.max_lon is not None else tess.max_lon
    max_lat = sim.max_lat if sim.max_lat is not None else tess.max_lat
    if None not in [min_lon, min_lat, max_lon, max_lat]:
        return float(min_lon), float(min_lat), float(max_lon), float(max_lat)

    lng_col = _lng_column(tessellation_df)
    if {"lat", lng_col}.issubset(tessellation_df.columns) and len(tessellation_df) > 0:
        return (
            float(tessellation_df[lng_col].min()),
            float(tessellation_df["lat"].min()),
            float(tessellation_df[lng_col].max()),
            float(tessellation_df["lat"].max()),
        )
    raise ValueError(
        "POI + building location inference requires a configured bbox, a tessellation "
        "with lat/lng columns, or cached Overture building features"
    )


def _building_features_output_path(config: CityBehavExConfig, resolution: int) -> Path:
    configured = config.profiles.overture_building_features_output
    if configured:
        return Path(configured)
    profile_out = Path(config.profiles.output)
    return profile_out.with_name(f"{profile_out.stem}_overture_buildings_h3r{resolution}.parquet")


def _read_building_features(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "h3_cell" not in df.columns and "tile_id" in df.columns:
        df = df.rename(columns={"tile_id": "h3_cell"})
    if "building_count" not in df.columns:
        raise ValueError(f"building features at {path} must contain a building_count column")
    if "h3_cell" not in df.columns:
        raise ValueError(f"building features at {path} must contain an h3_cell or tile_id column")
    out = df[["h3_cell", "building_count"]].copy()
    out["h3_cell"] = out["h3_cell"].astype(str)
    out["building_count"] = pd.to_numeric(out["building_count"], errors="coerce").fillna(0.0)
    return out.groupby("h3_cell", as_index=False)["building_count"].sum()


def _fetch_overture_building_features(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    resolution: int,
    overture_release: str,
) -> pd.DataFrame:
    typer.echo(
        f"Fetching Overture Maps {overture_release} building counts "
        f"by H3 cell (res={resolution}) ..."
    )
    return duckdb.sql(f"""
        INSTALL spatial; LOAD spatial;
        INSTALL h3 FROM community; LOAD h3;
        INSTALL httpfs; LOAD httpfs;
        SET s3_region = 'us-west-2';

        SELECT
            h3_latlng_to_cell_string(
                ST_Y(ST_Centroid(geometry)),
                ST_X(ST_Centroid(geometry)),
                {resolution}
            ) AS h3_cell,
            COUNT(*) AS building_count
        FROM read_parquet(
            's3://overturemaps-us-west-2/release/{overture_release}/theme=buildings/type=*/*',
            filename=true,
            hive_partitioning=1
        )
        WHERE bbox.xmax >= {min_lon}
          AND bbox.xmin <= {max_lon}
          AND bbox.ymax >= {min_lat}
          AND bbox.ymin <= {max_lat}
        GROUP BY h3_cell
    """).df()


def _load_or_build_building_features(
    config: CityBehavExConfig,
    tessellation_df: pd.DataFrame,
    resolution: int,
) -> pd.DataFrame:
    if config.profiles.overture_building_features_path:
        path = Path(config.profiles.overture_building_features_path)
        if path.exists():
            typer.echo(f"Loading Overture building features from {path} ...")
            return _read_building_features(path)

    out = _building_features_output_path(config, resolution)
    if out.exists():
        typer.echo(f"Loading cached Overture building features from {out} ...")
        return _read_building_features(out)

    min_lon, min_lat, max_lon, max_lat = _resolve_spatial_bounds(config, tessellation_df)
    features = _fetch_overture_building_features(
        min_lon,
        min_lat,
        max_lon,
        max_lat,
        resolution,
        config.tessellation.overture_release,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out, index=False)
    typer.echo(f"Saved {len(features):,} Overture building feature cells -> {out}")
    return _read_building_features(out)


def _base_relevance_column(
    config: CityBehavExConfig,
    tessellation_df: pd.DataFrame,
    relevance_column: str,
) -> str | None:
    candidates = [
        relevance_column if relevance_column != _WORK_SCORE_COLUMN else None,
        config.simulation.relevance_column,
        config.tessellation.relevance_column,
        "total_poi_count",
        "relevance",
    ]
    for candidate in candidates:
        if candidate and candidate in tessellation_df.columns:
            return candidate
    return None


def _poi_counts_by_h3(
    tessellation_df: pd.DataFrame,
    resolution: int,
    relevance_column: str | None,
) -> pd.Series:
    lng_col = _lng_column(tessellation_df)
    cells = [
        h3.latlng_to_cell(lat, lng, resolution)
        for lat, lng in zip(tessellation_df["lat"], tessellation_df[lng_col])
    ]
    if relevance_column and relevance_column in tessellation_df.columns:
        weights = pd.to_numeric(tessellation_df[relevance_column], errors="coerce").fillna(0.0)
    else:
        weights = pd.Series(1.0, index=tessellation_df.index)
    return pd.DataFrame({"h3_cell": cells, "weight": weights}).groupby("h3_cell")["weight"].sum()


def _jitter_h3_cell_centers(
    sampled_cells: np.ndarray,
    resolution: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    centers = np.array([h3.cell_to_latlng(c) for c in sampled_cells])
    edge_len_m = h3.average_hexagon_edge_length(resolution, unit="m")
    lat_jitter_deg = 0.4 * edge_len_m / 111_320.0
    lng_scale_m = 111_320.0 * np.cos(np.radians(centers[:, 0]))
    lng_jitter_deg = 0.4 * edge_len_m / np.where(lng_scale_m > 0, lng_scale_m, 111_320.0)

    lat = centers[:, 0].copy()
    lng = centers[:, 1].copy()
    for i, cell in enumerate(sampled_cells):
        for _attempt in range(12):
            candidate_lat = centers[i, 0] + rng.uniform(-lat_jitter_deg, lat_jitter_deg)
            candidate_lng = centers[i, 1] + rng.uniform(-lng_jitter_deg[i], lng_jitter_deg[i])
            if h3.latlng_to_cell(candidate_lat, candidate_lng, resolution) == cell:
                lat[i] = candidate_lat
                lng[i] = candidate_lng
                break
    return pd.DataFrame({"lat": lat, "lng": lng})


def _append_work_scores(
    config: CityBehavExConfig,
    tessellation_df: pd.DataFrame,
    relevance_column: str,
) -> tuple[pd.DataFrame, str]:
    if not config.profiles.enabled:
        return tessellation_df, relevance_column
    if not {"lat", _lng_column(tessellation_df)}.issubset(tessellation_df.columns):
        raise ValueError("POI + building work scoring requires tessellation lat/lng columns")

    base_column = _base_relevance_column(config, tessellation_df, relevance_column)
    resolution = config.profiles.overture_feature_h3_resolution or config.tessellation.resolution
    buildings = _load_or_build_building_features(config, tessellation_df, resolution)
    building_counts = dict(zip(buildings["h3_cell"], buildings["building_count"]))

    lng_col = _lng_column(tessellation_df)
    tile_cells = [
        h3.latlng_to_cell(lat, lng, resolution)
        for lat, lng in zip(tessellation_df["lat"], tessellation_df[lng_col])
    ]
    poi = (
        pd.to_numeric(tessellation_df[base_column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if base_column
        else np.ones(len(tessellation_df), dtype=float)
    )
    building = np.array([building_counts.get(cell, 0.0) for cell in tile_cells], dtype=float)

    pc = config.profiles
    enriched = tessellation_df.copy()
    enriched["building_count"] = building
    enriched[_WORK_SCORE_COLUMN] = (
        pc.work_building_weight * _minmax(building)
        + pc.work_poi_weight * _minmax(poi)
    )
    if float(enriched[_WORK_SCORE_COLUMN].sum()) <= 0:
        enriched[_WORK_SCORE_COLUMN] = 1.0
    typer.echo("Using POI + Overture building work scores for profile work tiles")
    return enriched, _WORK_SCORE_COLUMN


def load_or_build_tessellation(config: CityBehavExConfig) -> tuple[pd.DataFrame, str, np.ndarray | None]:
    tessellation_df, relevance_column = _load_or_build_tessellation_df(config)
    tessellation_df, relevance_column = _append_work_scores(config, tessellation_df, relevance_column)
    tessellation_df, home_tile_pool = _append_home_anchors(config, tessellation_df, relevance_column)
    tessellation_df = _maybe_snap_to_roads(config, tessellation_df)
    return tessellation_df, relevance_column, home_tile_pool


def _home_anchors_output_path(config: CityBehavExConfig) -> Path:
    configured = config.profiles.home_anchors_output
    if configured:
        return Path(configured)
    profile_out = Path(config.profiles.output)
    method = config.profiles.location_inference_method
    resolution = config.profiles.home_anchor_h3_resolution
    return profile_out.with_name(f"{profile_out.stem}_home_anchors_{method}_v3_h3r{resolution}.parquet")


def _read_home_anchor_candidates(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if {"lat", "lng"}.issubset(df.columns):
        return df[["lat", "lng"]].copy()

    if "geometry" in df.columns:
        try:
            import geopandas as gpd

            gdf = gpd.read_parquet(path)
            centroids = gdf.geometry.centroid
            return pd.DataFrame({"lat": centroids.y, "lng": centroids.x})
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"could not extract home-anchor centroids from {path}") from exc

    raise ValueError(f"home anchors at {path} must have lat/lng or geometry columns")


def _derive_legacy_home_anchor_candidates_from_tessellation(
    config: CityBehavExConfig,
    tessellation_df: pd.DataFrame,
    limit: int,
) -> pd.DataFrame:
    """Approximate residential HOME anchors from the local POI tessellation.

    There is no ground-truth "residential building" signal in the POI-derived
    tessellation (Overture ``places`` only covers businesses/amenities), and
    fetching Overture's ``buildings`` theme over the network for this is
    prohibitively slow for a large metro bbox. Instead, tile the bbox with H3
    cells and weight sampling toward cells with fewer nearby POIs, since dense
    POI clusters are commercial/downtown cores while sparser cells are more
    likely residential.
    """
    sim = config.simulation
    tess = config.tessellation
    min_lon = sim.min_lon if sim.min_lon is not None else tess.min_lon
    min_lat = sim.min_lat if sim.min_lat is not None else tess.min_lat
    max_lon = sim.max_lon if sim.max_lon is not None else tess.max_lon
    max_lat = sim.max_lat if sim.max_lat is not None else tess.max_lat
    lng_col = "lng" if "lng" in tessellation_df.columns else "lon"
    if None in [min_lon, min_lat, max_lon, max_lat]:
        min_lon, max_lon = float(tessellation_df[lng_col].min()), float(tessellation_df[lng_col].max())
        min_lat, max_lat = float(tessellation_df["lat"].min()), float(tessellation_df["lat"].max())

    resolution = config.profiles.home_anchor_h3_resolution
    boundary = h3.LatLngPoly(
        [
            (min_lat, min_lon),
            (min_lat, max_lon),
            (max_lat, max_lon),
            (max_lat, min_lon),
        ]
    )
    cells = list(h3.polygon_to_cells(boundary, resolution))
    if not cells:
        raise ValueError("no H3 cells found for the configured bounding box")

    poi_cells = [
        h3.latlng_to_cell(lat, lng, resolution)
        for lat, lng in zip(tessellation_df["lat"], tessellation_df[lng_col])
    ]
    poi_counts = pd.Series(poi_cells).value_counts()

    weights = np.array([1.0 / (1.0 + poi_counts.get(cell, 0)) for cell in cells], dtype=float)
    weights /= weights.sum()

    rng = np.random.default_rng(config.simulation.random_state)
    sampled_cells = rng.choice(np.asarray(cells), size=limit, p=weights, replace=True)
    return _jitter_h3_cell_centers(sampled_cells, resolution, rng)


def _derive_home_anchor_candidates_from_tessellation(
    config: CityBehavExConfig,
    tessellation_df: pd.DataFrame,
    relevance_column: str,
    limit: int,
) -> pd.DataFrame:
    min_lon, min_lat, max_lon, max_lat = _resolve_spatial_bounds(config, tessellation_df)
    resolution = config.profiles.home_anchor_h3_resolution
    boundary = h3.LatLngPoly(
        [
            (min_lat, min_lon),
            (min_lat, max_lon),
            (max_lat, max_lon),
            (max_lat, min_lon),
        ]
    )
    cells = list(h3.polygon_to_cells(boundary, resolution))
    if not cells:
        raise ValueError("no H3 cells found for the configured bounding box")

    base_column = _base_relevance_column(config, tessellation_df, relevance_column)
    poi_counts = _poi_counts_by_h3(tessellation_df, resolution, base_column)
    buildings = _load_or_build_building_features(config, tessellation_df, resolution)
    building_counts = buildings.set_index("h3_cell")["building_count"]
    cells = sorted(cell for cell in cells if float(building_counts.get(cell, 0.0)) > 0)
    if not cells:
        raise ValueError(
            "POI + building HOME inference found no building cells inside the configured bbox; "
            "provide a matching overture_building_features_path/cache or check the bbox/resolution"
        )

    poi = np.array([poi_counts.get(cell, 0.0) for cell in cells], dtype=float)
    building = np.array([building_counts.get(cell, 0.0) for cell in cells], dtype=float)
    poi_scaled = _minmax(poi)
    building_scaled = _minmax(np.log1p(building))
    pc = config.profiles
    weights = building_scaled * (
        pc.home_building_weight
        + pc.home_poi_inverse_weight * (1.0 - poi_scaled)
    )
    if float(weights.sum()) <= 0:
        weights = np.ones(len(cells), dtype=float)
    weights /= weights.sum()

    rng = np.random.default_rng(config.simulation.random_state)
    sampled_cells = rng.choice(np.asarray(cells), size=limit, p=weights, replace=True)
    typer.echo("Derived residential HOME anchors from POI + Overture building scores")
    return _jitter_h3_cell_centers(sampled_cells, resolution, rng)


def _load_or_build_home_anchor_candidates(
    config: CityBehavExConfig,
    tessellation_df: pd.DataFrame,
    relevance_column: str,
) -> pd.DataFrame:
    if config.profiles.home_anchors_path:
        path = Path(config.profiles.home_anchors_path)
        if path.exists():
            typer.echo(f"Loading residential HOME anchors from {path} ...")
            return _read_home_anchor_candidates(path)

    out = _home_anchors_output_path(config)
    if out.exists():
        typer.echo(f"Loading cached residential HOME anchors from {out} ...")
        return _read_home_anchor_candidates(out)

    typer.echo("Deriving residential HOME anchors from POI + Overture building scores ...")
    anchors = _derive_home_anchor_candidates_from_tessellation(
        config,
        tessellation_df,
        relevance_column,
        config.simulation.agents,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    anchors.to_parquet(out, index=False)
    typer.echo(f"Saved {len(anchors):,} HOME anchor candidates -> {out}")
    return anchors


def _append_home_anchors(
    config: CityBehavExConfig,
    tessellation_df: pd.DataFrame,
    relevance_column: str,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    if not config.profiles.enabled:
        return tessellation_df, None

    anchors = _load_or_build_home_anchor_candidates(config, tessellation_df, relevance_column)
    anchors = anchors.replace([np.inf, -np.inf], np.nan).dropna(subset=["lat", "lng"]).reset_index(drop=True)
    if len(anchors) == 0:
        raise ValueError("no valid residential HOME anchors are available")

    n_agents = config.simulation.agents
    rng = np.random.default_rng(config.simulation.random_state)
    chosen = anchors.iloc[rng.choice(len(anchors), size=n_agents, replace=len(anchors) < n_agents)].reset_index(drop=True)

    start_idx = len(tessellation_df)
    rows = pd.DataFrame({col: [pd.NA] * n_agents for col in tessellation_df.columns})
    rows["lat"] = chosen["lat"].to_numpy(dtype=float)
    lng_col = "lng" if "lng" in tessellation_df.columns else "lon"
    rows[lng_col] = chosen["lng"].to_numpy(dtype=float)
    if "lng" in tessellation_df.columns and "lon" in rows.columns:
        rows["lon"] = rows["lng"]
    rows["tile_id"] = [f"home_anchor_{i + 1}" for i in range(n_agents)]
    rows["category"] = "residential"
    rows["purpose"] = "HOME"
    if relevance_column in rows.columns:
        rows[relevance_column] = float(config.profiles.home_anchor_relevance)
    elif "relevance" in rows.columns:
        rows["relevance"] = float(config.profiles.home_anchor_relevance)

    augmented = pd.concat([tessellation_df, rows], ignore_index=True)
    home_tile_pool = np.arange(start_idx, start_idx + n_agents, dtype=np.int64)
    typer.echo(f"Appended {n_agents:,} synthetic residential HOME anchors")
    return augmented, home_tile_pool


def _maybe_snap_to_roads(config: CityBehavExConfig, tessellation_df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``road_node`` column mapping each location to its nearest road graph node.

    The snapping result is cached separately from the tessellation file itself,
    since ``sim.tessellation``/``tess.path`` may point at a file we don't own.
    """
    rn = config.road_network
    if not rn.enabled:
        return tessellation_df

    if Path(rn.snap_output).exists():
        snap_df = pd.read_parquet(rn.snap_output)
        if len(snap_df) == len(tessellation_df):
            typer.echo(f"Loading cached road-node snapping from {rn.snap_output} ...")
            return tessellation_df.assign(road_node=snap_df["road_node"].to_numpy())
        typer.echo(
            f"Warning: cached road-node snapping at {rn.snap_output} has "
            f"{len(snap_df):,} rows but tessellation has {len(tessellation_df):,} — rebuilding"
        )

    sim = config.simulation
    tess = config.tessellation
    min_lon = sim.min_lon if sim.min_lon is not None else tess.min_lon
    min_lat = sim.min_lat if sim.min_lat is not None else tess.min_lat
    max_lon = sim.max_lon if sim.max_lon is not None else tess.max_lon
    max_lat = sim.max_lat if sim.max_lat is not None else tess.max_lat
    lng_col = "lng" if "lng" in tessellation_df.columns else "lon"
    if None in [min_lon, min_lat, max_lon, max_lat]:
        min_lon, max_lon = float(tessellation_df[lng_col].min()), float(tessellation_df[lng_col].max())
        min_lat, max_lat = float(tessellation_df["lat"].min()), float(tessellation_df["lat"].max())

    overture_release = rn.overture_release or tess.overture_release
    nodes_df, _edges_df = build_road_graph(
        min_lon, min_lat, max_lon, max_lat, overture_release, rn.nodes_output, rn.edges_output
    )
    road_node = snap_locations_to_graph(
        tessellation_df, nodes_df, rn.snap_max_distance_m, lat_col="lat", lng_col=lng_col
    )
    n_unsnapped = int((road_node < 0).sum())
    if n_unsnapped:
        typer.echo(
            f"Warning: {n_unsnapped:,}/{len(road_node):,} locations are farther than "
            f"{rn.snap_max_distance_m:.0f}m from the road graph and will fall back to "
            "straight-line routing for trips touching them"
        )

    Path(rn.snap_output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"road_node": road_node}).to_parquet(rn.snap_output, index=False)
    return tessellation_df.assign(road_node=road_node)


def _load_or_build_tessellation_df(config: CityBehavExConfig) -> tuple[pd.DataFrame, str]:
    sim = config.simulation
    tess = config.tessellation
    tessellation_path = sim.tessellation or tess.path

    if tessellation_path:
        typer.echo(f"Loading tessellation from {tessellation_path} ...")
        tessellation_df = pd.read_parquet(tessellation_path)
        relevance_column = sim.relevance_column or tess.relevance_column
        if tess.min_poi_count > 0 and relevance_column in tessellation_df.columns:
            n_before = len(tessellation_df)
            tessellation_df = tessellation_df[
                tessellation_df[relevance_column] >= tess.min_poi_count
            ].reset_index(drop=True)
            n_dropped = n_before - len(tessellation_df)
            if n_dropped:
                typer.echo(
                    f"Dropped {n_dropped:,} cells with {relevance_column} < {tess.min_poi_count} "
                    f"({len(tessellation_df):,} remaining)"
                )
        return tessellation_df, relevance_column

    if tess.output and Path(tess.output).exists():
        typer.echo(f"Loading cached generated tessellation from {tess.output} ...")
        tessellation_df = pd.read_parquet(tess.output)
        relevance_column = sim.relevance_column or tess.relevance_column
        if tess.poi_tessellation and relevance_column == "total_poi_count" and "relevance" in tessellation_df.columns:
            relevance_column = "relevance"
        return tessellation_df, relevance_column

    min_lon = sim.min_lon if sim.min_lon is not None else tess.min_lon
    min_lat = sim.min_lat if sim.min_lat is not None else tess.min_lat
    max_lon = sim.max_lon if sim.max_lon is not None else tess.max_lon
    max_lat = sim.max_lat if sim.max_lat is not None else tess.max_lat
    if None in [min_lon, min_lat, max_lon, max_lat]:
        raise ValueError(
            "provide a tessellation path or all four bbox values "
            "(min_lon, min_lat, max_lon, max_lat)"
        )

    if tess.poi_tessellation:
        tessellation_df = build_poi_tessellation(
            min_lon, min_lat, max_lon, max_lat, tess.overture_release
        )
        typer.echo(f"Generated {len(tessellation_df):,} POI tiles from bbox")
    else:
        tessellation_df = build_tessellation(
            min_lon,
            min_lat,
            max_lon,
            max_lat,
            tess.resolution,
            tess.enrich_overture,
            tess.overture_release,
            min_poi_count=tess.min_poi_count,
        )
        typer.echo(f"Generated {len(tessellation_df):,} H3 cells from bbox")

    if tess.output:
        Path(tess.output).parent.mkdir(parents=True, exist_ok=True)
        tessellation_df.to_parquet(tess.output, index=False)
        typer.echo(f"Saved generated tessellation -> {tess.output}")

    relevance_column = sim.relevance_column or tess.relevance_column
    if tess.poi_tessellation and relevance_column == "total_poi_count" and "relevance" in tessellation_df.columns:
        relevance_column = "relevance"
    return tessellation_df, relevance_column


def simulation_dates(config: CityBehavExConfig) -> tuple[pd.Timestamp, pd.Timestamp]:
    if config.simulation.start_date:
        start_date = pd.Timestamp(config.simulation.start_date)
    else:
        start_date = pd.Timestamp(
            datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        )
    return start_date, start_date + timedelta(days=config.simulation.days)


def maybe_build_diaries(
    config: CityBehavExConfig,
    tessellation_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> Optional[tuple[dict[str, DiaryBatch], LLMStats, float]]:
    """Fetch one diary batch per day type needed for [start_date, end_date)
    (weekday/weekend plus any overlapping special days), or None if no LLM
    client and no validated cache are configured."""
    valid_cache = config.llm.validated_diaries_path
    has_llm_client = all([config.llm.base_url, config.llm.api_key, config.llm.model])
    if not has_llm_client and not valid_cache:
        return None

    started = time.perf_counter()
    stats = LLMStats()
    distribution = purpose_distribution(tessellation_df)
    location_counts = allocate_location_counts(
        config.diaries.location_count_mu,
        config.diaries.location_count_sigma,
        config.diaries.max_locations,
        config.llm.diary_count,
    )
    day_types = config.diaries.day_types_for_range(
        start_date.date(), (end_date - timedelta(days=1)).date()
    )
    batches: dict[str, DiaryBatch] = {}
    for day_type in day_types:
        batches[day_type] = fetch_diary_batch(
            config.llm,
            city_profile=config.diaries.profile_for(day_type),
            representative_day=config.diaries.representative_day,
            purpose_distribution=distribution,
            location_counts=location_counts,
            location_count_mu=config.diaries.location_count_mu,
            location_count_sigma=config.diaries.location_count_sigma,
            max_locations=config.diaries.max_locations,
            motif_exploration_rate=config.diaries.motif_exploration_rate,
            random_state=config.simulation.random_state,
            variant=day_type,
            stats=stats,
        )
    return batches, stats, time.perf_counter() - started


def _run_density_epr(
    config: CityBehavExConfig,
    tessellation_df: pd.DataFrame,
    relevance_column: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> tuple[skmob2.TrajDataFrame, Optional[str]]:
    typer.echo(
        f"Running DensityEPR: {config.simulation.agents} agents x {config.simulation.days} days "
        f"({start_date.date()} -> {end_date.date()})"
    )
    model = DensityEPR()
    traj = model.generate(
        start_date=start_date,
        end_date=end_date,
        spatial_tessellation=tessellation_df,
        n_agents=config.simulation.agents,
        relevance_column=relevance_column,
        random_state=config.simulation.random_state,
    )
    traj = skmob2.TrajDataFrame(traj)
    synth_activity_col = None
    if "purpose" in tessellation_df.columns:
        traj.df = _merge_tessellation_metadata(
            traj.df,
            tessellation_df,
            ["tile_id", "purpose", "category"],
        )
        synth_activity_col = "purpose"
    return traj, synth_activity_col


def _merge_tessellation_metadata(
    df: pd.DataFrame,
    tessellation_df: pd.DataFrame,
    candidate_cols: list[str],
) -> pd.DataFrame:
    extra_cols = [c for c in candidate_cols if c in tessellation_df.columns and c not in df.columns]
    if not extra_cols:
        return df
    lookup = tessellation_df[["lat", "lng"] + extra_cols].drop_duplicates(["lat", "lng"])
    return df.merge(lookup, on=["lat", "lng"], how="left")


def _build_schedule(
    config: CityBehavExConfig,
    diary_batches: dict[str, DiaryBatch],
    start_date: pd.Timestamp,
    profiles: Optional[list[AgentProfile]] = None,
) -> tuple[DiaryBank, tuple, np.ndarray, Optional[np.ndarray], DdcrpAgentInfo]:
    """Build the diary bank and run profile-driven CRP schedule selection.

    Returns (bank, diary_arrays, chosen, profile_embeddings, crp_info).
    profile_embeddings is None when embeddings are disabled or unavailable.
    """
    bank = build_diary_bank(
        diary_batches,
        config.embedding,
        config.simulation.granularity_minutes,
    )
    counts = Counter(bank.day_type.tolist())
    typer.echo(
        f"ddCRP schedule bank: {len(bank.diaries)} diaries "
        f"({', '.join(f'{n} {t}' for t, n in counts.items())}), "
        f"embeddings={'on' if bank.embedded else 'off (popularity CRP, no profile similarity)'}"
    )

    profile_embeddings = None
    narratives = None
    if profiles is not None:
        narratives = [profile_to_narrative(p) for p in profiles]
        profile_embeddings = embed_profiles(narratives, config.embedding)
        if profile_embeddings is not None:
            typer.echo(f"Profile embeddings: {profile_embeddings.shape}")
        else:
            typer.echo("Profile embeddings unavailable — falling back to popularity CRP")

    agent_diary_sim = None
    if (
        profiles is not None
        and narratives is not None
        and config.schedule.similarity_backend == "alignment_model"
    ):
        agent_diary_sim = score_alignment_matrix(narratives, bank.diaries, config.schedule)
        if agent_diary_sim is not None:
            typer.echo(f"Macro-schedule alignment scores: {agent_diary_sim.shape}")
        else:
            typer.echo("Alignment scorer unavailable — falling back to embedding cosine")

    day_types = [
        config.diaries.resolve_day_type((start_date + pd.Timedelta(days=d)).date())
        for d in range(config.simulation.days)
    ]
    diary_arrays, chosen, crp_info = build_ddcrp_diary(
        bank,
        start_date,
        config.simulation.days,
        day_types,
        config.simulation.agents,
        config.simulation.random_state,
        config.schedule,
        profile_embeddings=profile_embeddings,
        agent_diary_sim=agent_diary_sim,
    )
    return bank, diary_arrays, chosen, profile_embeddings, crp_info


def _save_crp_artifact(
    path: str,
    bank: DiaryBank,
    chosen: np.ndarray,
    crp_info: DdcrpAgentInfo,
) -> None:
    """Persist per-(agent, diary) ddCRP state next to the trajectory output.

    ``build_ddcrp_diary`` computes T_a/alpha_a/similarity/usage-counts and then
    throws them away once the diary picks are baked into ``diary_arrays`` — the
    web UI's diary-selection debug panel needs them to reconstruct "what would
    this agent pick next", so they're written out in long form (one row per
    agent x bank diary) alongside the run.
    """
    n_agents, _days = chosen.shape
    K = len(bank.diaries)
    usage_counts = np.stack([np.bincount(chosen[a], minlength=K) for a in range(n_agents)])

    diary_ids = np.array([d.diary_id for d in bank.diaries])
    df = pd.DataFrame(
        {
            "agent": np.repeat(np.arange(n_agents, dtype=np.int64), K),
            "diary_id": np.tile(diary_ids, n_agents),
            "day_type": np.tile(bank.day_type, n_agents),
            "sim": crp_info.agent_diary_sim.reshape(-1),
            "usage_count": usage_counts.reshape(-1),
            "T_a": np.repeat(crp_info.T_per_agent, K),
            "alpha_a": np.repeat(crp_info.alpha_per_agent, K),
        }
    )
    df.to_parquet(path, index=False)


def _build_activity_data(
    config: CityBehavExConfig,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (act_embs, act_dur_mu, act_dur_sigma, purpose_act_starts, purpose_acts) when enabled."""
    if not config.activities.enabled:
        return None, None, None, None, None
    act_dur_mu, act_dur_sigma = activity_duration_arrays()
    if config.activities.act_dur_scale != 1.0:
        act_dur_mu = act_dur_mu + np.log(config.activities.act_dur_scale)
    if config.activities.act_dur_sigma_scale != 1.0:
        act_dur_sigma = act_dur_sigma * config.activities.act_dur_sigma_scale
    purpose_act_starts, purpose_acts = build_eligibility_csr()
    act_embs = None
    if config.activities.embed_activities:
        descriptions = activity_descriptions()
        act_embs = embed_texts(descriptions, config.embedding)
        if act_embs is not None:
            typer.echo(f"Activity embeddings: {act_embs.shape}")
        else:
            typer.echo("Activity embeddings unavailable — using count-only CRP")
    typer.echo(f"Activities enabled: {len(act_dur_mu)} activities, kappa={config.activities.kappa}, T={config.activities.temperature}")
    return act_embs, act_dur_mu, act_dur_sigma, purpose_act_starts, purpose_acts


def _stamp_path(path: str, ts: str) -> str:
    p = Path(path)
    return str(p.with_name(f"{p.stem}_{ts}{p.suffix}"))


class _IncrementalParquetWriter:
    """Appends DataFrame chunks to a parquet file as they arrive, opening the
    writer lazily on the first non-empty chunk (parquet needs a schema up
    front). Used to stream per-day waypoint chunks straight to disk instead
    of accumulating the whole run's waypoints in memory."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._writer: pq.ParquetWriter | None = None
        self.rows_written = 0

    def write(self, chunk: pd.DataFrame) -> None:
        if chunk.empty:
            return
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self._path, table.schema)
        self._writer.write_table(table)
        self.rows_written += len(chunk)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()


def _run_simulation_core(
    config: CityBehavExConfig,
    tessellation_df: pd.DataFrame,
    relevance_column: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    diary_arrays: tuple,
    timing: CoreTiming,
    profiles: Optional[list[AgentProfile]] = None,
    profile_embeddings: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
) -> tuple[skmob2.TrajDataFrame, Optional[str]]:
    granularity = config.simulation.granularity_minutes
    typer.echo(
        f"Running simulation core: {config.simulation.agents} agents x {config.simulation.days} days "
        f"@ {granularity}-min slots, {config.simulation.car_speed_kmh:.0f} km/h car "
        f"({start_date.date()} -> {end_date.date()})"
    )
    home_tiles = (
        np.array([p.home_tile for p in profiles], dtype=np.int64)
        if profiles is not None
        else None
    )
    work_tiles = (
        np.array([p.work_tile for p in profiles], dtype=np.int64)
        if profiles is not None
        else None
    )
    act_embs, act_dur_mu, act_dur_sigma, purpose_act_starts, purpose_acts = _build_activity_data(config)

    road_kwargs: dict = {}
    rn = config.road_network
    if rn.enabled and "road_node" in tessellation_df.columns:
        lng_col = "lng" if "lng" in tessellation_df.columns else "lon"
        min_lon = config.simulation.min_lon if config.simulation.min_lon is not None else config.tessellation.min_lon
        min_lat = config.simulation.min_lat if config.simulation.min_lat is not None else config.tessellation.min_lat
        max_lon = config.simulation.max_lon if config.simulation.max_lon is not None else config.tessellation.max_lon
        max_lat = config.simulation.max_lat if config.simulation.max_lat is not None else config.tessellation.max_lat
        if None in [min_lon, min_lat, max_lon, max_lat]:
            min_lon, max_lon = float(tessellation_df[lng_col].min()), float(tessellation_df[lng_col].max())
            min_lat, max_lat = float(tessellation_df["lat"].min()), float(tessellation_df["lat"].max())
        overture_release = rn.overture_release or config.tessellation.overture_release
        nodes_df, edges_df = build_road_graph(
            min_lon, min_lat, max_lon, max_lat, overture_release, rn.nodes_output, rn.edges_output
        )
        typer.echo(
            f"Road routing enabled: {len(nodes_df):,} nodes, {len(edges_df):,} directed edges, "
            f"max {rn.max_leg_waypoints} waypoints/leg"
        )
        road_kwargs = dict(
            road_edge_from=edges_df["from_node"].to_numpy(dtype=np.int64),
            road_edge_to=edges_df["to_node"].to_numpy(dtype=np.int64),
            road_edge_weight_ds=edges_df["weight_ds"].to_numpy(dtype=np.int64),
            road_node_lats=nodes_df["lat"].to_numpy(dtype=np.float64),
            road_node_lngs=nodes_df["lng"].to_numpy(dtype=np.float64),
            location_road_node=tessellation_df["road_node"].to_numpy(dtype=np.int64),
            max_leg_waypoints=rn.max_leg_waypoints,
        )

    base = output_path or config.simulation.output
    moving_path = base.replace(".parquet", "_moving.parquet")
    stream_moving = config.simulation.stream_output and rn.enabled
    moving_writer = _IncrementalParquetWriter(moving_path) if stream_moving else None
    on_day_flush = moving_writer.write if moving_writer is not None else None

    profile_types = [p.job for p in profiles] if profiles is not None else None
    df, encounters, moving, activities, social_graph = simulate_agents(
        tessellation_df,
        relevance_column,
        diary_arrays,
        start_ts=int(start_date.timestamp()),
        end_ts=int(end_date.timestamp()),
        slot_seconds=granularity * 60,
        car_speed_kmh=config.simulation.car_speed_kmh,
        n_agents=config.simulation.agents,
        random_state=config.simulation.random_state,
        social_graph_k=config.simulation.social_graph_k,
        profile_graph_exact_threshold=config.simulation.profile_graph_exact_threshold,
        rho=config.simulation.rho,
        gamma=config.simulation.gamma,
        alpha=config.simulation.alpha,
        dt_update_mob_sim_hours=config.simulation.dt_update_mob_sim_hours,
        indipendency_window_hours=config.simulation.indipendency_window_hours,
        gravity_deterrence_exponent=config.simulation.gravity_deterrence_exponent,
        gravity_origin_exponent=config.simulation.gravity_origin_exponent,
        gravity_destination_exponent=config.simulation.gravity_destination_exponent,
        timing=timing,
        starting_locs=home_tiles,
        work_tiles=work_tiles,
        profile_embeddings=profile_embeddings,
        act_embs=act_embs,
        act_dur_mu=act_dur_mu,
        act_dur_sigma=act_dur_sigma,
        purpose_act_starts=purpose_act_starts,
        purpose_acts=purpose_acts,
        act_kappa=config.activities.kappa,
        act_temp=config.activities.temperature,
        return_social_graph=True,
        social_node_profiles=profile_types,
        on_day_flush=on_day_flush,
        **road_kwargs,
    )
    social_path = social_network_sidecar_path(base)
    social_graph.write_json(social_path)
    typer.echo(
        f"Saved social network ({social_graph.metadata['node_count']:,} nodes, "
        f"{social_graph.metadata['edge_count']:,} edges) -> {social_path}"
    )
    if len(encounters) > 0:
        enc_path = base.replace(".parquet", "_encounters.parquet")
        encounters.to_parquet(enc_path, index=False)
        typer.echo(f"Saved {len(encounters):,} encounters -> {enc_path}")
    if moving_writer is not None:
        # `moving` here is only the final day's still-open tail -- everything
        # closed before it was already streamed to disk via on_day_flush.
        moving_writer.write(moving)
        moving_writer.close()
        if moving_writer.rows_written > 0:
            typer.echo(f"Saved {moving_writer.rows_written:,} waypoints (streamed) -> {moving_path}")
    elif rn.enabled and len(moving) > 0:
        moving.to_parquet(moving_path, index=False)
        typer.echo(f"Saved {len(moving):,} waypoints -> {moving_path}")
    if config.activities.enabled and len(activities) > 0:
        act_path = base.replace(".parquet", "_activities.parquet")
        activities.to_parquet(act_path, index=False)
        typer.echo(f"Saved {len(activities):,} activities -> {act_path}")
    df = _merge_tessellation_metadata(df, tessellation_df, ["tile_id", "category"])
    traj = skmob2.TrajDataFrame(
        df, datetime_col="datetime", lat_col="lat", lng_col="lng", uid_col="uid"
    )
    return traj, "purpose"


def maybe_build_profiles(
    config: CityBehavExConfig,
    tessellation_df: pd.DataFrame,
    relevance_column: str,
    home_tile_pool: np.ndarray | None = None,
) -> Optional[list[AgentProfile]]:
    """Generate or load agent profiles when ``profiles.enabled`` is true."""
    if not config.profiles.enabled:
        return None
    n = config.simulation.agents
    pc = config.profiles
    if pc.profiles_path:
        loaded = load_profiles(pc.profiles_path, n)
        if loaded is not None:
            typer.echo(f"Loaded {len(loaded)} agent profiles from {pc.profiles_path}")
            return loaded
        typer.echo(f"Warning: profiles_path {pc.profiles_path!r} not usable — generating")
    rng = np.random.default_rng(config.simulation.random_state)
    profiles = generate_profiles(n, pc, rng, tessellation_df, relevance_column, home_tile_pool=home_tile_pool)
    typer.echo(f"Generated {len(profiles)} agent profiles")
    if pc.output:
        from pathlib import Path
        out = Path(pc.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        profiles_to_frame(profiles).to_parquet(str(out), index=False)
        typer.echo(f"Saved agent profiles -> {pc.output}")
    return profiles


def run_simulation(config: CityBehavExConfig) -> skmob2.TrajDataFrame:
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    stamped_output = _stamp_path(config.simulation.output, ts)

    tessellation_df, relevance_column, home_tile_pool = load_or_build_tessellation(config)
    start_date, end_date = simulation_dates(config)
    profiles = maybe_build_profiles(config, tessellation_df, relevance_column, home_tile_pool)
    diary_result = maybe_build_diaries(config, tessellation_df, start_date, end_date)
    core_timing = CoreTiming()

    if diary_result is None:
        traj, synth_activity_col = _run_density_epr(
            config, tessellation_df, relevance_column, start_date, end_date
        )
    else:
        diary_batches, llm_stats, llm_seconds = diary_result
        cache_text = (
            f", {llm_stats.cache_hits:,} cached diary batches"
            if llm_stats.cache_hits
            else ""
        )
        typer.echo(
            f"LLM diary phase: {llm_seconds:.2f}s, {llm_stats.calls:,} chat completion calls"
            f"{cache_text}"
        )
        bank, diary_arrays, chosen, profile_embeddings, crp_info = _build_schedule(
            config, diary_batches, start_date, profiles=profiles
        )
        crp_path = stamped_output.replace(".parquet", "_crp.parquet")
        _save_crp_artifact(crp_path, bank, chosen, crp_info)
        typer.echo(
            f"Saved ddCRP diary selection state "
            f"({config.simulation.agents} agents x {len(bank.diaries)} diaries) -> {crp_path}"
        )
        traj, synth_activity_col = _run_simulation_core(
            config,
            tessellation_df,
            relevance_column,
            start_date,
            end_date,
            diary_arrays,
            core_timing,
            profiles=profiles,
            profile_embeddings=profile_embeddings,
            output_path=stamped_output,
        )
        typer.echo(f"Rust simulation phase: {core_timing.seconds:.2f}s")

    traj.df.to_parquet(stamped_output, index=False)
    typer.echo(
        f"Saved {len(traj.df):,} records "
        f"({traj.df[traj.uid_col].nunique()} agents) -> {stamped_output}"
    )

    if config.comparison.path:
        typer.echo("Comparison data configured; view this run in the CityBehavEx web UI.")
    return traj
