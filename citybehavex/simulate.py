from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
import skmob2
import typer
from skmob2.comparison import wasserstein_distance
from skmob2.models import DensityEPR
from skmob_vis import plot_jump_lengths_ecdf, plot_visits_frequency_ecdf

app = typer.Typer(help="CityBehavEx – synthetic urban mobility toolkit.")

_DATETIME_CANDIDATES = [
    "datetime", "start_timestamp", "timestamp", "check-in_time",
    "start_time", "checkin_time", "time", "date",
]
_LAT_CANDIDATES = ["lat", "latitude"]
_LNG_CANDIDATES = ["lng", "lon", "longitude", "long"]
_UID_CANDIDATES = ["uid", "user_id", "user", "agent_id", "userid"]


def _detect_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in cols_lower:
            return cols_lower[candidate.lower()]
    return None


def _build_tessellation(
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
        typer.echo(f"Generating H3 tessellation (res={resolution}) from bbox …")
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
    min_poi_count: int = typer.Option(
        1, help="Minimum POI count per cell; cells below this threshold are dropped."
    ),
    output: str = typer.Option("tessellation.parquet", help="Output parquet path"),
):
    """Generate an H3 tessellation from a bounding box."""
    df = _build_tessellation(
        min_lon, min_lat, max_lon, max_lat, resolution, enrich_overture, overture_release,
        min_poi_count=min_poi_count,
    )
    df.to_parquet(output, index=False)
    typer.echo(f"Saved {len(df):,} H3 cells → {output}")


def _generate_comparison_report(
    traj: skmob2.TrajDataFrame,
    real_path: str,
    observed_label: str,
    output_path: str,
) -> None:
    real_df = pd.read_parquet(real_path)
    real_traj = skmob2.TrajDataFrame(
        real_df,
        datetime_col=_detect_column(real_df, _DATETIME_CANDIDATES),
        lat_col=_detect_column(real_df, _LAT_CANDIDATES),
        lng_col=_detect_column(real_df, _LNG_CANDIDATES),
        uid_col=_detect_column(real_df, _UID_CANDIDATES),
    )

    synth_jumps = traj.jump_lengths(merge=True)
    real_jumps = real_traj.jump_lengths(merge=True)

    synth_visits = traj.df[traj.uid_col].value_counts().to_list()
    real_visits = real_traj.df[real_traj.uid_col].value_counts().to_list()

    w_jump = wasserstein_distance(synth_jumps, real_jumps)
    w_visits = wasserstein_distance(synth_visits, real_visits)

    fig_jump = plot_jump_lengths_ecdf(
        synth_jumps, real_jumps,
        labels=("synthetic", observed_label),
    )
    fig_visits = plot_visits_frequency_ecdf(
        synth_visits, real_visits,
        labels=("synthetic", observed_label),
        title="Visits frequency ECDF",
    )

    metrics_html = (
        f"<div style=\"font-family:'IBM Plex Mono',monospace;font-size:14px;"
        f"padding:16px 24px;background:#fbf8f1;color:#14110d;"
        f"border-top:1px solid #dcd5c4;\">"
        f"<strong>Wasserstein distances</strong><br>"
        f"Jump lengths: {w_jump:.4f} km &nbsp;|&nbsp; Visits frequency: {w_visits:.4f}"
        f"</div>"
    )
    full_html = (
        "<!doctype html>\n<html>\n<head>\n"
        "  <meta charset=\"utf-8\">\n"
        "  <title>CityBehavEx Comparison Report</title>\n"
        "  <style>\n"
        "    html,body{margin:0;padding:0;background:#fbf8f1;}\n"
        "    .charts{display:flex;flex-wrap:wrap;}\n"
        "    .charts iframe{flex:1 1 600px;border:0;min-height:420px;}\n"
        "  </style>\n"
        "</head>\n<body>\n"
        f"  <div class=\"charts\">{fig_jump._repr_html_()}{fig_visits._repr_html_()}</div>\n"
        f"  {metrics_html}\n"
        "</body>\n</html>\n"
    )
    Path(output_path).write_text(full_html, encoding="utf-8")
    typer.echo(
        f"Comparison report → {output_path}  "
        f"(W-dist: jump={w_jump:.4f}, visits={w_visits:.4f})"
    )


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
    min_poi_count: int = typer.Option(
        1, help="Minimum value of --relevance-column per cell; cells below this threshold are dropped."
    ),
    agents: int = typer.Option(500, help="Number of synthetic agents"),
    days: int = typer.Option(7, help="Simulation duration in days"),
    relevance_column: str = typer.Option(
        "total_poi_count", help="Column used as location attractiveness weight"
    ),
    output: str = typer.Option("trajectories.parquet", help="Output parquet path"),
    random_state: int = typer.Option(42, help="Random seed"),
    comparison: Optional[str] = typer.Option(
        None, "--comparison",
        help="Path to a trajectories parquet to compare against. Triggers HTML report.",
    ),
    comparison_label: str = typer.Option(
        "observed", help="Legend label for the comparison series in plots."
    ),
    comparison_html: str = typer.Option(
        "comparison.html", help="Output path for the comparison HTML report."
    ),
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
        if min_poi_count > 0 and relevance_column in tessellation_df.columns:
            n_before = len(tessellation_df)
            tessellation_df = tessellation_df[
                tessellation_df[relevance_column] >= min_poi_count
            ].reset_index(drop=True)
            n_dropped = n_before - len(tessellation_df)
            if n_dropped:
                typer.echo(
                    f"Dropped {n_dropped:,} cells with {relevance_column} < {min_poi_count} "
                    f"({len(tessellation_df):,} remaining)"
                )
    elif has_bbox:
        tessellation_df = _build_tessellation(
            min_lon, min_lat, max_lon, max_lat, resolution, enrich_overture, overture_release,
            min_poi_count=min_poi_count,
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
        f"({traj.df[traj.uid_col].nunique()} agents) → {output}"
    )

    if comparison:
        _generate_comparison_report(
            traj=traj,
            real_path=comparison,
            observed_label=comparison_label,
            output_path=comparison_html,
        )


if __name__ == "__main__":
    app()
