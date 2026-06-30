from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import typer

_CATEGORY_CSV = Path(__file__).parents[1] / "category" / "unique_categories.csv"


def load_category_mapping() -> dict[str, str]:
    mapping_df = pd.read_csv(_CATEGORY_CSV)
    return dict(zip(mapping_df["primary_category"], mapping_df["purpose"]))


def build_tessellation(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    resolution: int,
    enrich_overture: bool,
    overture_release: str,
    min_poi_count: int = 1,
) -> pd.DataFrame:
    bbox_wkt = (
        f"POLYGON(({min_lon} {min_lat}, {max_lon} {min_lat}, "
        f"{max_lon} {max_lat}, {min_lon} {max_lat}, {min_lon} {min_lat}))"
    )

    if enrich_overture:
        typer.echo(
            f"Fetching H3 cells (res={resolution}) and enriching with "
            f"Overture Maps {overture_release} POI data ..."
        )
        df = duckdb.sql(f"""
            INSTALL spatial;  LOAD spatial;
            INSTALL h3 FROM community;  LOAD h3;
            SET s3_region = 'us-west-2';

            WITH grouped_pois AS (
                SELECT
                    h3_latlng_to_cell_string(
                        ST_Y(geometry), ST_X(geometry), {resolution}
                    ) AS h3_str,
                    COUNT(*) AS total_poi_count
                FROM read_parquet(
                    's3://overturemaps-us-west-2/release/{overture_release}/theme=places/type=place/*',
                    filename=true, hive_partitioning=1
                )
                WHERE bbox.xmin BETWEEN {min_lon} AND {max_lon}
                  AND bbox.ymin BETWEEN {min_lat} AND {max_lat}
                GROUP BY h3_str
                HAVING COUNT(*) >= {min_poi_count}
            )
            SELECT
                h3_str                                          AS tile_id,
                h3_cell_to_lat(h3_string_to_h3(h3_str))       AS lat,
                h3_cell_to_lng(h3_string_to_h3(h3_str))       AS lng,
                h3_cell_to_boundary_wkt(h3_string_to_h3(h3_str)) AS cell_polygon_wkt,
                total_poi_count
            FROM grouped_pois
            ORDER BY total_poi_count DESC
        """).df()
    else:
        typer.echo(f"Generating H3 tessellation (res={resolution}) from bbox ...")
        df = duckdb.sql(f"""
            INSTALL h3 FROM community;  LOAD h3;

            WITH cells AS (
                SELECT UNNEST(
                    h3_polygon_wkt_to_cells('{bbox_wkt}', {resolution})
                ) AS h3_cell
            )
            SELECT
                h3_h3_to_string(h3_cell)         AS tile_id,
                h3_cell_to_lat(h3_cell)          AS lat,
                h3_cell_to_lng(h3_cell)          AS lng,
                h3_cell_to_boundary_wkt(h3_cell) AS cell_polygon_wkt,
                1                                AS total_poi_count
            FROM cells
            ORDER BY h3_cell
        """).df()

    if min_poi_count > 0:
        df = df[df["total_poi_count"] >= min_poi_count].reset_index(drop=True)
    return df


def build_poi_tessellation(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    overture_release: str,
) -> pd.DataFrame:
    typer.echo(
        f"Fetching individual POIs from Overture Maps {overture_release} "
        "and computing 500 m relevance ..."
    )
    df = duckdb.sql(f"""
        INSTALL spatial;  LOAD spatial;
        SET s3_region = 'us-west-2';

        WITH raw_pois AS (
            SELECT
                id                        AS poi_id,
                ST_Y(geometry)            AS lat,
                ST_X(geometry)            AS lng,
                categories.primary        AS category
            FROM read_parquet(
                's3://overturemaps-us-west-2/release/{overture_release}/theme=places/type=place/*',
                filename=true, hive_partitioning=1
            )
            WHERE bbox.xmin BETWEEN {min_lon} AND {max_lon}
              AND bbox.ymin BETWEEN {min_lat} AND {max_lat}
        ),
        relevance_counts AS (
            SELECT
                p1.poi_id,
                p1.lat,
                p1.lng,
                p1.category,
                COUNT(p2.poi_id) AS relevance
            FROM raw_pois p1
            LEFT JOIN raw_pois p2
                ON  p1.poi_id != p2.poi_id
                AND ABS(p2.lat - p1.lat) <= 0.0045
                AND ABS(p2.lng - p1.lng) <= 0.0045
                AND 2 * 6371000 * ASIN(SQRT(
                        POWER(SIN(RADIANS((p2.lat - p1.lat) / 2)), 2) +
                        COS(RADIANS(p1.lat)) * COS(RADIANS(p2.lat)) *
                        POWER(SIN(RADIANS((p2.lng - p1.lng) / 2)), 2)
                    )) <= 500
            GROUP BY p1.poi_id, p1.lat, p1.lng, p1.category
        )
        SELECT
            poi_id      AS tile_id,
            lat,
            lng,
            category,
            relevance
        FROM relevance_counts
        ORDER BY relevance DESC
    """).df()
    category_map = load_category_mapping()
    df["purpose"] = df["category"].map(category_map).fillna("OTHER")
    return df


def purpose_distribution(tessellation_df: pd.DataFrame) -> dict[str, float]:
    if "purpose" not in tessellation_df.columns or len(tessellation_df) == 0:
        return {}
    counts = tessellation_df["purpose"].value_counts(normalize=True)
    return {str(key): float(value) for key, value in counts.items()}
