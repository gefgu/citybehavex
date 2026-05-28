from datetime import datetime, timedelta
from typing import Optional

import duckdb
import pandas as pd
import skmob2
import typer
from skmob2.models import DensityEPR

app = typer.Typer(help="CityBehavEx – synthetic urban mobility toolkit.")


def _build_tessellation(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    resolution: int,
    enrich_overture: bool,
    overture_release: str,
) -> pd.DataFrame:
    bbox_wkt = (
        f"POLYGON(({min_lon} {min_lat}, {max_lon} {min_lat}, "
        f"{max_lon} {max_lat}, {min_lon} {max_lat}, {min_lon} {min_lat}))"
    )

    if enrich_overture:
        typer.echo(
            f"Fetching H3 cells (res={resolution}) and enriching with "
            f"Overture Maps {overture_release} POI data …"
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
        typer.echo(f"Generating H3 tessellation (res={resolution}) from bbox …")
        df = duckdb.sql(f"""
            INSTALL h3 FROM community;  LOAD h3;

            WITH cells AS (
                SELECT UNNEST(
                    h3_polygon_wkt_to_cells('{bbox_wkt}', {resolution})
                ) AS h3_cell
            )
            SELECT
                h3_cell_to_string(h3_cell)       AS tile_id,
                h3_cell_to_lat(h3_cell)          AS lat,
                h3_cell_to_lng(h3_cell)          AS lng,
                h3_cell_to_boundary_wkt(h3_cell) AS cell_polygon_wkt,
                1                                AS total_poi_count
            FROM cells
            ORDER BY h3_cell
        """).df()

    return df


@app.command()
def tessellate(
    min_lon: float = typer.Option(..., help="Bounding box west longitude"),
    min_lat: float = typer.Option(..., help="Bounding box south latitude"),
    max_lon: float = typer.Option(..., help="Bounding box east longitude"),
    max_lat: float = typer.Option(..., help="Bounding box north latitude"),
    resolution: int = typer.Option(
        10, help="H3 resolution (0–15). Resolution 10 ≈ 65 m edge length."
    ),
    enrich_overture: bool = typer.Option(
        False,
        "--enrich-overture/--no-enrich-overture",
        help="Enrich cells with Overture Maps place (POI) counts via S3.",
    ),
    overture_release: str = typer.Option(
        "2026-05-20.0", help="Overture Maps release tag (used only with --enrich-overture)."
    ),
    output: str = typer.Option("tessellation.parquet", help="Output parquet path"),
):
    """Generate an H3 tessellation from a bounding box."""
    df = _build_tessellation(
        min_lon, min_lat, max_lon, max_lat, resolution, enrich_overture, overture_release
    )
    df.to_parquet(output, index=False)
    typer.echo(f"Saved {len(df):,} H3 cells → {output}")


@app.command()
def simulate(
    tessellation: Optional[str] = typer.Option(
        None, help="Path to an existing tessellation parquet. Mutually exclusive with bbox options."
    ),
    min_lon: Optional[float] = typer.Option(None, help="Bounding box west longitude"),
    min_lat: Optional[float] = typer.Option(None, help="Bounding box south latitude"),
    max_lon: Optional[float] = typer.Option(None, help="Bounding box east longitude"),
    max_lat: Optional[float] = typer.Option(None, help="Bounding box north latitude"),
    resolution: int = typer.Option(10, help="H3 resolution when building tessellation from bbox"),
    enrich_overture: bool = typer.Option(
        False,
        "--enrich-overture/--no-enrich-overture",
        help="Enrich bbox-generated tessellation with Overture Maps POI counts.",
    ),
    overture_release: str = typer.Option(
        "2026-05-20.0", help="Overture Maps release tag (used only with --enrich-overture)."
    ),
    agents: int = typer.Option(500, help="Number of synthetic agents"),
    days: int = typer.Option(7, help="Simulation duration in days"),
    relevance_column: str = typer.Option(
        "total_poi_count", help="Column used as location attractiveness weight"
    ),
    output: str = typer.Option("trajectories.parquet", help="Output parquet path"),
    random_state: int = typer.Option(42, help="Random seed"),
):
    """Run DensityEPR simulation on a tessellation file or a bbox."""
    has_bbox = all(v is not None for v in [min_lon, min_lat, max_lon, max_lat])

    if tessellation and has_bbox:
        typer.echo(
            "Error: provide either --tessellation or bbox options, not both.", err=True
        )
        raise typer.Exit(1)

    if tessellation:
        typer.echo(f"Loading tessellation from {tessellation} …")
        tessellation_df = pd.read_parquet(tessellation)
    elif has_bbox:
        tessellation_df = _build_tessellation(
            min_lon, min_lat, max_lon, max_lat, resolution, enrich_overture, overture_release
        )
        typer.echo(f"Generated {len(tessellation_df):,} H3 cells from bbox")
    else:
        typer.echo(
            "Error: provide --tessellation or all four bbox options "
            "(--min-lon, --min-lat, --max-lon, --max-lat).",
            err=True,
        )
        raise typer.Exit(1)

    start_date = pd.Timestamp(
        datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    )
    end_date = start_date + timedelta(days=days)

    typer.echo(
        f"Running DensityEPR: {agents} agents × {days} days "
        f"({start_date.date()} → {end_date.date()})"
    )
    model = DensityEPR()
    traj = model.generate(
        start_date=start_date,
        end_date=end_date,
        spatial_tessellation=tessellation_df,
        n_agents=agents,
        relevance_column=relevance_column,
        random_state=random_state,
    )

    traj = skmob2.TrajDataFrame(traj)
    traj.df.to_parquet(output, index=False)
    typer.echo(
        f"Saved {len(traj.df):,} records "
        f"({traj.df['uid'].nunique()} agents) → {output}"
    )


if __name__ == "__main__":
    app()
