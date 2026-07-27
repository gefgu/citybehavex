from __future__ import annotations

from pathlib import Path

import duckdb
import h3
import numpy as np
import pandas as pd
import typer

from citybehavex.config import CityBehavExConfig
from citybehavex.simulation.network_pipeline import _maybe_snap_to_rail, _maybe_snap_to_roads
from citybehavex.simulation.spatial import _lng_column, _minmax, _resolve_spatial_bounds, h3_cell_strings
from citybehavex.tessellation import build_poi_tessellation, build_tessellation

_WORK_SCORE_COLUMN = "work_score"


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
    cells = h3_cell_strings(tessellation_df, resolution, lng_col=lng_col)
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
    tile_cells = h3_cell_strings(tessellation_df, resolution, lng_col=lng_col)
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
    tessellation_df = _maybe_snap_to_rail(config, tessellation_df)
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


def load_or_build_tessellation(config: CityBehavExConfig) -> tuple[pd.DataFrame, str, np.ndarray | None]:
    tessellation_df, relevance_column = _load_or_build_tessellation_df(config)
    tessellation_df, relevance_column = _append_work_scores(config, tessellation_df, relevance_column)
    tessellation_df, home_tile_pool = _append_home_anchors(config, tessellation_df, relevance_column)
    tessellation_df = _maybe_snap_to_roads(config, tessellation_df)
    tessellation_df = _maybe_snap_to_rail(config, tessellation_df)
    return tessellation_df, relevance_column, home_tile_pool
