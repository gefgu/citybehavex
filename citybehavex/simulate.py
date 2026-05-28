from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
import skmob2
import typer
from skmob2.comparison import wasserstein_distance
from skmob2.measures.spatial import waiting_times as _waiting_times
from skmob2.measures.visits.motifs import discover_daily_motifs_from_agents
from skmob2.models import DensityEPR
from skmob_vis import (
    plot_activity_transition_matrix,
    plot_daily_activity_distribution,
    plot_dwell_time_ecdf,
    plot_jump_lengths_ecdf,
    plot_motif_literature_comparison,
    plot_radius_of_gyration_ecdf,
    plot_trip_duration_ecdf,
    plot_visit_purpose_distribution,
    plot_visits_frequency_ecdf,
)

app = typer.Typer(help="CityBehavEx – synthetic urban mobility toolkit.")

_DATETIME_CANDIDATES = [
    "datetime", "start_timestamp", "timestamp", "check-in_time",
    "start_time", "_start_time", "checkin_time", "time", "date",
]
_LAT_CANDIDATES = ["lat", "latitude"]
_LNG_CANDIDATES = ["lng", "lon", "longitude", "long"]
_UID_CANDIDATES = ["uid", "user_id", "user", "agent_id", "userid"]
_DURATION_CANDIDATES = ["duration_minutes", "duration", "trip_duration_minutes", "duration_hours"]
_ACTIVITY_CANDIDATES = ["purpose", "activity", "act", "location_type", "category", "purpose_d"]
_LOCATION_CANDIDATES = ["location_id", "tile_id", "Code_INSEE_D", "area", "venueId", "location"]
_END_TS_CANDIDATES = ["end_timestamp", "_end_time", "end_time"]


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


def _build_poi_tessellation(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    overture_release: str,
) -> pd.DataFrame:
    typer.echo(
        f"Fetching individual POIs from Overture Maps {overture_release} "
        f"and computing 500 m relevance …"
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
            category    AS purpose,
            relevance
        FROM relevance_counts
        ORDER BY relevance DESC
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
    min_poi_count: int = typer.Option(
        1, help="Minimum POI count per cell; cells below this threshold are dropped."
    ),
    poi_tessellation: bool = typer.Option(
        False,
        "--poi-tessellation/--no-poi-tessellation",
        help=(
            "Use individual Overture POIs as tiles instead of H3 cells. "
            "tile_id = POI id, purpose = primary category, "
            "relevance = POI count within 500 m radius."
        ),
    ),
    output: str = typer.Option("tessellation.parquet", help="Output parquet path"),
):
    """Generate an H3 tessellation from a bounding box."""
    if poi_tessellation:
        df = _build_poi_tessellation(
            min_lon, min_lat, max_lon, max_lat, overture_release
        )
        typer.echo(f"Saved {len(df):,} POI tiles → {output}")
    else:
        df = _build_tessellation(
            min_lon, min_lat, max_lon, max_lat, resolution, enrich_overture, overture_release,
            min_poi_count=min_poi_count,
        )
        typer.echo(f"Saved {len(df):,} H3 cells → {output}")
    df.to_parquet(output, index=False)


def _waiting_times_minutes(traj: skmob2.TrajDataFrame) -> list:
    secs = _waiting_times(
        traj.df,
        merge=True,
        datetime_col=traj.datetime_col,
        lat_col=traj.lat_col,
        lng_col=traj.lng_col,
        uid_col=traj.uid_col,
    )
    return [s / 60 for s in secs]


def _generate_comparison_report(
    traj: skmob2.TrajDataFrame,
    real_path: str,
    observed_label: str,
    output_path: str,
    synth_activity_col: Optional[str] = None,
) -> None:
    from datetime import datetime as _dt

    real_df = pd.read_parquet(real_path)
    real_traj = skmob2.TrajDataFrame(
        real_df,
        datetime_col=_detect_column(real_df, _DATETIME_CANDIDATES),
        lat_col=_detect_column(real_df, _LAT_CANDIDATES),
        lng_col=_detect_column(real_df, _LNG_CANDIDATES),
        uid_col=_detect_column(real_df, _UID_CANDIDATES),
    )

    labels = ("synthetic", observed_label)

    # Jump lengths
    synth_jumps = traj.jump_lengths(merge=True)
    real_jumps = real_traj.jump_lengths(merge=True)
    w_jump = wasserstein_distance(synth_jumps, real_jumps)

    # Visits frequency
    synth_visits = traj.df[traj.uid_col].value_counts().to_list()
    real_visits = real_traj.df[real_traj.uid_col].value_counts().to_list()
    w_visits = wasserstein_distance(synth_visits, real_visits)

    # Radius of gyration
    synth_rog = traj.radius_of_gyration()["radius_of_gyration"].to_numpy()
    real_rog = real_traj.radius_of_gyration()["radius_of_gyration"].to_numpy()
    w_rog = wasserstein_distance(synth_rog, real_rog)

    # Dwell time (waiting times between consecutive visits, in minutes)
    synth_dwell = _waiting_times_minutes(traj)
    real_dwell = _waiting_times_minutes(real_traj)
    w_dwell = wasserstein_distance(synth_dwell, real_dwell)

    # Trip duration — use explicit duration column if available in real data
    duration_col = _detect_column(real_df, _DURATION_CANDIDATES)
    if duration_col:
        real_trip = real_df[duration_col].dropna().tolist()
        synth_trip = _waiting_times_minutes(traj)
        w_trip = wasserstein_distance(synth_trip, real_trip)
    else:
        real_trip = synth_trip = w_trip = None

    # Build ECDF plots
    fig_jump = plot_jump_lengths_ecdf(synth_jumps, real_jumps, labels=labels)
    fig_visits = plot_visits_frequency_ecdf(synth_visits, real_visits, labels=labels)
    fig_rog = plot_radius_of_gyration_ecdf(synth_rog, real_rog, labels=labels)
    fig_dwell = plot_dwell_time_ecdf(synth_dwell, real_dwell, labels=labels)
    fig_trip = (
        plot_trip_duration_ecdf(synth_trip, real_trip, labels=labels)
        if real_trip is not None
        else None
    )

    ecdf_charts_html = "".join(
        f._repr_html_()
        for f in [fig_jump, fig_visits, fig_rog, fig_dwell]
    )
    if fig_trip:
        ecdf_charts_html += fig_trip._repr_html_()

    # Activity profile plots
    activity_col = _detect_column(real_df, _ACTIVITY_CANDIDATES)
    activity_section_html = ""
    if activity_col:
        fig_purpose = plot_visit_purpose_distribution(real_df)
        fig_transition = plot_activity_transition_matrix(real_df)
        fig_daily = plot_daily_activity_distribution(real_df)
        activity_charts_html = (
            fig_purpose._repr_html_()
            + fig_transition._repr_html_()
            + fig_daily._repr_html_()
        )
        activity_section_html = f"""
  <div class="section-header">
    <span>Activity profile &mdash; {observed_label}</span>
  </div>
  <div class="charts">{activity_charts_html}</div>"""

    if synth_activity_col and synth_activity_col in traj.df.columns:
        fig_synth_purpose = plot_visit_purpose_distribution(traj.df)
        fig_synth_transition = plot_activity_transition_matrix(traj.df)
        fig_synth_daily = plot_daily_activity_distribution(traj.df)
        synth_activity_charts_html = (
            fig_synth_purpose._repr_html_()
            + fig_synth_transition._repr_html_()
            + fig_synth_daily._repr_html_()
        )
        activity_section_html += f"""
  <div class="section-header">
    <span>Activity profile &mdash; synthetic</span>
  </div>
  <div class="charts">{synth_activity_charts_html}</div>"""

    # Motif comparison chart
    motif_section_html = ""
    try:
        real_uid = _detect_column(real_df, _UID_CANDIDATES + ["id"])
        real_loc = _detect_column(real_df, _LOCATION_CANDIDATES)
        real_purpose = _detect_column(real_df, _ACTIVITY_CANDIDATES)
        real_start = _detect_column(real_df, _DATETIME_CANDIDATES)
        real_end = _detect_column(real_df, _END_TS_CANDIDATES)
        if all(c is not None for c in [real_uid, real_loc, real_purpose, real_start, real_end]):
            _, real_motif_dist = discover_daily_motifs_from_agents(
                real_df,
                user_id_col=real_uid,
                location_id_col=real_loc,
                purpose_col=real_purpose,
                timestamp_col=real_start,
                end_timestamp_col=real_end,
            )
        else:
            real_motif_dist = None

        synth_motif_dist = None
        if synth_activity_col and "tile_id" in traj.df.columns:
            synth_df = traj.df.sort_values([traj.uid_col, traj.datetime_col]).copy()
            synth_df["_end_ts"] = synth_df.groupby(traj.uid_col)[traj.datetime_col].shift(-1)
            synth_df = synth_df.dropna(subset=["_end_ts"]).rename(
                columns={traj.datetime_col: "start_timestamp", "_end_ts": "end_timestamp"}
            )
            _, synth_motif_dist = discover_daily_motifs_from_agents(
                synth_df,
                user_id_col=traj.uid_col,
                location_id_col="tile_id",
                purpose_col=synth_activity_col,
                timestamp_col="start_timestamp",
                end_timestamp_col="end_timestamp",
            )

        if real_motif_dist is not None or synth_motif_dist is not None:
            fig_motif = plot_motif_literature_comparison(
                reference_distribution=real_motif_dist,
                comparison_distribution=synth_motif_dist,
                labels=(observed_label, "synthetic"),
            )
            motif_section_html = f"""
  <div class="section-header">
    <span>Motif comparison &mdash; literature vs {observed_label}{" vs synthetic" if synth_motif_dist is not None else ""}</span>
  </div>
  <div class="charts">{fig_motif._repr_html_()}</div>"""
    except Exception:
        pass

    # Wasserstein table rows
    w_rows = [
        ("Jump lengths", f"{w_jump:.4f}", "km"),
        ("Visits frequency", f"{w_visits:.4f}", ""),
        ("Radius of gyration", f"{w_rog:.4f}", "km"),
        ("Dwell time", f"{w_dwell:.4f}", "min"),
    ]
    if w_trip is not None:
        w_rows.append(("Trip duration", f"{w_trip:.4f}", "min"))

    table_rows = "".join(
        f"<tr><td>{name}</td><td>{val}</td><td>{unit}</td></tr>"
        for name, val, unit in w_rows
    )

    generated_at = _dt.now().strftime("%Y-%m-%d %H:%M")

    full_html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>CityBehavEx Comparison Report</title>
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&family=IBM+Plex+Sans:wght@400;600&display=swap">
  <style>
    html,body{{margin:0;padding:0;background:#fbf8f1;color:#14110d;font-family:'IBM Plex Sans',sans-serif;}}
    .header{{padding:32px 32px 24px;border-bottom:1px solid #dcd5c4;}}
    .header h1{{margin:0;font-size:20px;font-weight:600;letter-spacing:-0.01em;}}
    .header p{{margin:6px 0 0;font-size:13px;color:#6b5e4c;}}
    .section-header{{padding:20px 32px 0;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:#6b5e4c;}}
    .charts{{display:flex;flex-wrap:wrap;padding:24px;gap:24px;}}
    .charts iframe{{flex:1 1 580px;border:0;min-height:420px;}}
    .metrics{{padding:24px 32px 32px;border-top:1px solid #dcd5c4;}}
    .metrics h2{{margin:0 0 14px;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:#6b5e4c;}}
    .metrics table{{border-collapse:collapse;font-family:'IBM Plex Mono',monospace;font-size:13px;}}
    .metrics td{{padding:3px 20px 3px 0;}}
    .metrics td:nth-child(2){{font-weight:500;font-variant-numeric:tabular-nums;}}
    .metrics td:nth-child(3){{color:#6b5e4c;}}
  </style>
</head>
<body>
  <div class="header">
    <h1>synthetic &nbsp;vs&nbsp; {observed_label}</h1>
    <p>Generated {generated_at}</p>
  </div>
  <div class="section-header">Distribution comparisons</div>
  <div class="charts">{ecdf_charts_html}</div>{activity_section_html}{motif_section_html}
  <div class="metrics">
    <h2>Wasserstein distances</h2>
    <table>{table_rows}</table>
  </div>
</body>
</html>
"""
    Path(output_path).write_text(full_html, encoding="utf-8")

    summary = "  ".join(f"{n}: {v}" for n, v, _ in w_rows)
    typer.echo(f"Comparison report → {output_path}  ({summary})")


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
    poi_tessellation: bool = typer.Option(
        False,
        "--poi-tessellation/--no-poi-tessellation",
        help=(
            "Use individual Overture POIs as tiles instead of H3 cells. "
            "tile_id = POI id, purpose = primary category, "
            "relevance = POI count within 500 m radius."
        ),
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
        if poi_tessellation:
            tessellation_df = _build_poi_tessellation(
                min_lon, min_lat, max_lon, max_lat, overture_release
            )
            typer.echo(f"Generated {len(tessellation_df):,} POI tiles from bbox")
        else:
            tessellation_df = _build_tessellation(
                min_lon, min_lat, max_lon, max_lat, resolution, enrich_overture,
                overture_release, min_poi_count=min_poi_count,
            )
            typer.echo(f"Generated {len(tessellation_df):,} H3 cells from bbox")
    else:
        typer.echo(
            "Error: provide --tessellation or all four bbox options "
            "(--min-lon, --min-lat, --max-lon, --max-lat).",
            err=True,
        )
        raise typer.Exit(1)

    if poi_tessellation and relevance_column == "total_poi_count" and "relevance" in tessellation_df.columns:
        relevance_column = "relevance"

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

    synth_activity_col = None
    if "purpose" in tessellation_df.columns:
        extra_cols = [c for c in ["tile_id", "purpose"] if c in tessellation_df.columns]
        lookup = tessellation_df[["lat", "lng"] + extra_cols].drop_duplicates(["lat", "lng"])
        traj.df = traj.df.merge(lookup, on=["lat", "lng"], how="left")
        synth_activity_col = "purpose"

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
            synth_activity_col=synth_activity_col,
        )


if __name__ == "__main__":
    app()
