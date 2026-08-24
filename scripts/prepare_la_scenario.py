#!/usr/bin/env python
"""Prepare Los Angeles inputs for the 1%-population CityBehavEx scenario.

The source data intentionally stays in the sibling ``scenes_stvd`` project.
This script creates only local, regenerable simulator inputs: a POI tessellation
weighted by observed July 2021 visits and population-proportional HOME anchors.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import polars as pl
from shapely.geometry import Point


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENES_ROOT = PROJECT_ROOT.parent / "scenes_stvd"
LA_SOURCE = SCENES_ROOT / "data" / "Visitor_census" / "LA"
LA_ANALYSIS = SCENES_ROOT / "data" / "la"
ACS_TOTALS = SCENES_ROOT / "data" / "acs" / "acsdt5y2021-b02001.dat"

COUNTIES = ("06037", "06059")
POPULATION_FRACTION = 0.01
RANDOM_STATE = 20210719

CANONICAL_CATEGORY = {
    "Food & Leisure": "restaurant",
    "Outdoors & Landmarks": "park",
    "Shopping": "supermarket",
    "Services & Institutions": "school",
    "Mobility": "car_dealer",
    "Unmapped": "other",
}
PURPOSE_BY_TOP_LEVEL = {
    "Services & Institutions": "WORK",
}


def _top_level_categories() -> dict[str, str]:
    mapping = pd.read_csv(LA_SOURCE / "category_mapping.csv")
    return dict(zip(mapping["category"].astype(str), mapping["top_level"].astype(str)))


def _classify_categories(value: object, category_map: dict[str, str]) -> tuple[str, str]:
    labels = [part.strip() for part in str(value or "").split("|") if part.strip()]
    top_levels = [category_map.get(label, "Unmapped") for label in labels]
    top_level = next((item for item in top_levels if item != "Unmapped"), "Unmapped")
    return CANONICAL_CATEGORY[top_level], PURPOSE_BY_TOP_LEVEL.get(top_level, "OTHER")


def _load_cbgs() -> gpd.GeoDataFrame:
    cbgs = gpd.read_file(LA_SOURCE / "LA_cbg.geojson")
    cbgs = cbgs.rename(columns={"GEOID": "cgb"})
    cbgs["cgb"] = cbgs["cgb"].astype(str)
    return cbgs.loc[cbgs["cgb"].str[:5].isin(COUNTIES), ["cgb", "geometry"]].copy()


def _weekly_visits() -> pd.Series:
    visits = pl.read_parquet(LA_ANALYSIS / "la_visits_hourly.parquet")
    totals = visits.group_by("cgb").agg(pl.col("num_visit").sum().alias("weekly_visits"))
    return totals.to_pandas().set_index("cgb")["weekly_visits"]


def build_tessellation(output: Path, cbgs: gpd.GeoDataFrame) -> pd.DataFrame:
    raw = pd.read_parquet(
        LA_SOURCE / "gplaces_pois_la.parquet",
        columns=["gmap_id", "latitude", "longitude", "categories"],
    ).dropna(subset=["gmap_id", "latitude", "longitude"])
    raw = raw.loc[raw["latitude"].between(-90, 90) & raw["longitude"].between(-180, 180)]
    raw = raw.drop_duplicates("gmap_id").copy()
    points = gpd.GeoDataFrame(
        raw,
        geometry=gpd.points_from_xy(raw["longitude"], raw["latitude"]),
        crs="EPSG:4326",
    )
    joined = points.sjoin(cbgs, how="left", predicate="within")
    category_map = _top_level_categories()
    classified = joined["categories"].map(lambda value: _classify_categories(value, category_map))
    joined["category"] = classified.map(lambda item: item[0])
    joined["purpose"] = classified.map(lambda item: item[1])

    weekly_visits = _weekly_visits()
    joined["weekly_visits"] = joined["cgb"].map(weekly_visits).fillna(0.0)
    poi_per_cbg = joined.groupby("cgb", dropna=False)["gmap_id"].transform("size")
    joined["relevance"] = (joined["weekly_visits"] / poi_per_cbg).fillna(0.0).clip(lower=1.0)
    tessellation = pd.DataFrame(
        {
            "tile_id": "la_poi_" + joined["gmap_id"].astype(str),
            "lat": joined["latitude"].astype(float),
            "lng": joined["longitude"].astype(float),
            "category": joined["category"],
            "purpose": joined["purpose"],
            "relevance": joined["relevance"].astype(float),
        }
    ).sort_values("tile_id", kind="stable").reset_index(drop=True)
    if tessellation["tile_id"].duplicated().any() or not np.isfinite(tessellation[["lat", "lng", "relevance"]]).all().all():
        raise ValueError("LA POI tessellation contains invalid IDs, coordinates, or relevance")
    output.parent.mkdir(parents=True, exist_ok=True)
    tessellation.to_parquet(output, index=False)
    return tessellation


def _population_by_cbg() -> pd.Series:
    acs = pl.read_csv(
        ACS_TOTALS,
        separator="|",
        columns=["GEO_ID", "B02001_E001"],
        schema_overrides={"B02001_E001": pl.Int64},
        null_values=["", "."],
    )
    population = (
        acs.filter(pl.col("GEO_ID").str.starts_with("1500000US"))
        .with_columns(pl.col("GEO_ID").str.slice(9).alias("cgb"))
        .filter(pl.col("cgb").str.slice(0, 5).is_in(COUNTIES))
        .select("cgb", "B02001_E001")
        .to_pandas()
        .set_index("cgb")["B02001_E001"]
    )
    return population.clip(lower=0).astype(np.int64)


def _allocate_one_percent(population: pd.Series) -> pd.Series:
    exact = population.astype(float) * POPULATION_FRACTION
    allocated = np.floor(exact).astype(np.int64)
    target = int(round(float(exact.sum())))
    remaining = target - int(allocated.sum())
    if remaining > 0:
        order = (exact - allocated).sort_values(ascending=False, kind="stable").index[:remaining]
        allocated.loc[order] += 1
    if int(allocated.sum()) != target:
        raise AssertionError("population allocation does not match the requested 1% total")
    return allocated


def _sample_points(geometry, count: int, rng: np.random.Generator) -> list[tuple[float, float]]:
    if count == 0:
        return []
    min_x, min_y, max_x, max_y = geometry.bounds
    points: list[tuple[float, float]] = []
    while len(points) < count:
        batch_size = max(32, (count - len(points)) * 3)
        xs = rng.uniform(min_x, max_x, batch_size)
        ys = rng.uniform(min_y, max_y, batch_size)
        for x, y in zip(xs, ys, strict=True):
            if geometry.contains(Point(float(x), float(y))):
                points.append((float(y), float(x)))
                if len(points) == count:
                    return points
    return points


def build_home_anchors(output: Path, cbgs: gpd.GeoDataFrame) -> pd.DataFrame:
    population = _population_by_cbg()
    allocated = _allocate_one_percent(population)
    joined = cbgs.join(allocated.rename("count"), on="cgb", how="inner")
    rng = np.random.default_rng(RANDOM_STATE)
    rows: list[tuple[float, float]] = []
    for row in joined.itertuples(index=False):
        rows.extend(_sample_points(row.geometry, int(row.count), rng))
    anchors = pd.DataFrame(rows, columns=["lat", "lng"])
    expected = int(allocated.sum())
    if len(anchors) != expected or not np.isfinite(anchors[["lat", "lng"]]).all().all():
        raise ValueError("LA HOME anchors are incomplete or contain invalid coordinates")
    output.parent.mkdir(parents=True, exist_ok=True)
    anchors.to_parquet(output, index=False)
    return anchors


def build_building_features(output: Path, tessellation: pd.DataFrame) -> pd.DataFrame:
    """Provide a local building proxy so the scenario never fetches Overture."""
    cells = [h3.latlng_to_cell(lat, lng, 10) for lat, lng in zip(tessellation["lat"], tessellation["lng"], strict=True)]
    features = (
        pd.DataFrame({"h3_cell": cells, "building_count": 1})
        .groupby("h3_cell", as_index=False)["building_count"]
        .sum()
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output, index=False)
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "la")
    args = parser.parse_args()
    cbgs = _load_cbgs()
    tessellation = build_tessellation(args.output_dir / "la_poi_tessellation.parquet", cbgs)
    anchors = build_home_anchors(args.output_dir / "la_home_anchors_1pct.parquet", cbgs)
    features = build_building_features(args.output_dir / "la_building_features_h3r10.parquet", tessellation)
    print(
        f"Wrote {len(tessellation):,} LA POIs, {len(anchors):,} population-weighted HOME anchors, "
        f"and {len(features):,} local building-feature cells."
    )


if __name__ == "__main__":
    main()
