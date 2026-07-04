from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from .config import CityBehavExConfig, apply_overrides, load_config
from .llm import LLMConfig
from .reports import ComparisonConfig
from .reports.comparison import (
    _activities_sidecar_path,
    generate_comparison_report,
    load_trajectory,
)
from .roads import RoadNetworkConfig
from .simulation import run_simulation
from .simulation import SimulationConfig
from .tessellation import build_poi_tessellation, build_tessellation
from .tessellation import TessellationConfig

app = typer.Typer(help="CityBehavEx - synthetic urban mobility toolkit.")


@app.command()
def tessellate(
    config: Optional[str] = typer.Option(None, "--config", help="YAML config path"),
    min_lon: Optional[float] = typer.Option(None, help="Bounding box west longitude"),
    min_lat: Optional[float] = typer.Option(None, help="Bounding box south latitude"),
    max_lon: Optional[float] = typer.Option(None, help="Bounding box east longitude"),
    max_lat: Optional[float] = typer.Option(None, help="Bounding box north latitude"),
    resolution: Optional[int] = typer.Option(None, help="H3 resolution (0-15)"),
    enrich_overture: Optional[bool] = typer.Option(
        None,
        "--enrich-overture/--no-enrich-overture",
        help="Enrich cells with Overture Maps place counts via S3.",
    ),
    overture_release: Optional[str] = typer.Option(None, help="Overture Maps release tag"),
    min_poi_count: Optional[int] = typer.Option(None, help="Minimum POI count per cell"),
    poi_tessellation: Optional[bool] = typer.Option(
        None,
        "--poi-tessellation/--no-poi-tessellation",
        help="Use individual Overture POIs as tiles instead of H3 cells.",
    ),
    output: Optional[str] = typer.Option(None, help="Output parquet path"),
):
    """Generate an H3 or POI tessellation from a bounding box."""
    loaded = load_config(config)
    tess = apply_overrides(
        loaded.tessellation,
        {
            "min_lon": min_lon,
            "min_lat": min_lat,
            "max_lon": max_lon,
            "max_lat": max_lat,
            "resolution": resolution,
            "enrich_overture": enrich_overture,
            "overture_release": overture_release,
            "min_poi_count": min_poi_count,
            "poi_tessellation": poi_tessellation,
            "output": output,
        },
    )
    assert isinstance(tess, TessellationConfig)
    if None in [tess.min_lon, tess.min_lat, tess.max_lon, tess.max_lat]:
        typer.echo(
            "Error: provide bbox values in config or CLI "
            "(--min-lon, --min-lat, --max-lon, --max-lat).",
            err=True,
        )
        raise typer.Exit(1)

    if tess.poi_tessellation:
        df = build_poi_tessellation(
            tess.min_lon,
            tess.min_lat,
            tess.max_lon,
            tess.max_lat,
            tess.overture_release,
        )
        typer.echo(f"Saved {len(df):,} POI tiles -> {tess.output}")
    else:
        df = build_tessellation(
            tess.min_lon,
            tess.min_lat,
            tess.max_lon,
            tess.max_lat,
            tess.resolution,
            tess.enrich_overture,
            tess.overture_release,
            min_poi_count=tess.min_poi_count,
        )
        typer.echo(f"Saved {len(df):,} H3 cells -> {tess.output}")
    df.to_parquet(tess.output, index=False)


@app.command()
def report(
    config: Optional[str] = typer.Option(None, "--config", help="YAML config path"),
    synthetic: Optional[str] = typer.Option(
        None,
        "--synthetic",
        help="Synthetic trajectories parquet. Defaults to simulation.output.",
    ),
    comparison: Optional[str] = typer.Option(
        None,
        "--comparison",
        help="Observed trajectories parquet. Defaults to comparison.path.",
    ),
    comparison_label: Optional[str] = typer.Option(
        None,
        help="Observed series label. Defaults to comparison.label.",
    ),
    output: Optional[str] = typer.Option(
        "report.html",
        help="HTML report output path.",
    ),
    json_output: Optional[str] = typer.Option(
        None,
        "--json",
        help="Metrics JSON output path.",
    ),
):
    """Generate an HTML + JSON mobility comparison report.

    Jump lengths / radius of gyration are recomputed as road-network distance
    (instead of straight-line Haversine) when the config's road_network is
    enabled and its cached graph parquet files exist -- otherwise falls back
    to the plain Haversine-based metrics.
    """
    loaded = load_config(config)
    synthetic_path = synthetic or loaded.simulation.output
    real_path = comparison or loaded.comparison.path
    label = comparison_label or loaded.comparison.label

    rn = loaded.road_network
    road_nodes_df = road_edges_df = None
    if (
        rn.enabled
        and loaded.comparison.road_network_distance
        and Path(rn.nodes_output).exists()
        and Path(rn.edges_output).exists()
    ):
        typer.echo(f"Loading cached road graph from {rn.nodes_output} / {rn.edges_output} ...")
        road_nodes_df = pd.read_parquet(rn.nodes_output)
        road_edges_df = pd.read_parquet(rn.edges_output)

    traj = load_trajectory(synthetic_path)
    generate_comparison_report(
        traj=traj,
        real_path=real_path,
        observed_label=label,
        output_path=output,
        synthetic_activities_path=_activities_sidecar_path(synthetic_path),
        json_output_path=json_output,
        road_nodes_df=road_nodes_df,
        road_edges_df=road_edges_df,
        road_snap_max_distance_m=rn.snap_max_distance_m,
    )


@app.command()
def simulate(
    config: Optional[str] = typer.Option(None, "--config", help="YAML config path"),
    tessellation: Optional[str] = typer.Option(
        None, help="Path to an existing tessellation parquet. Mutually exclusive with bbox options."
    ),
    min_lon: Optional[float] = typer.Option(None, help="Bounding box west longitude"),
    min_lat: Optional[float] = typer.Option(None, help="Bounding box south latitude"),
    max_lon: Optional[float] = typer.Option(None, help="Bounding box east longitude"),
    max_lat: Optional[float] = typer.Option(None, help="Bounding box north latitude"),
    resolution: Optional[int] = typer.Option(None, help="H3 resolution when building tessellation from bbox"),
    enrich_overture: Optional[bool] = typer.Option(
        None,
        "--enrich-overture/--no-enrich-overture",
        help="Enrich bbox-generated tessellation with Overture Maps POI counts.",
    ),
    overture_release: Optional[str] = typer.Option(None, help="Overture Maps release tag"),
    min_poi_count: Optional[int] = typer.Option(
        None, help="Minimum value of --relevance-column per cell"
    ),
    poi_tessellation: Optional[bool] = typer.Option(
        None,
        "--poi-tessellation/--no-poi-tessellation",
        help="Use individual Overture POIs as tiles instead of H3 cells.",
    ),
    agents: Optional[int] = typer.Option(None, help="Number of synthetic agents"),
    days: Optional[int] = typer.Option(None, help="Simulation duration in days"),
    start_date: Optional[str] = typer.Option(None, help="Simulation start timestamp/date"),
    relevance_column: Optional[str] = typer.Option(None, help="Location attractiveness column"),
    output: Optional[str] = typer.Option(None, help="Output parquet path"),
    random_state: Optional[int] = typer.Option(None, help="Random seed"),
    social_graph_k: Optional[int] = typer.Option(
        None,
        "--social-graph-k",
        min=1,
        help="Maximum social neighbors per agent.",
    ),
    profile_graph_exact_threshold: Optional[int] = typer.Option(
        None,
        "--profile-graph-exact-threshold",
        min=1,
        help="Maximum agent count for exact profile kNN before cluster sampling.",
    ),
    diary_count: Optional[int] = typer.Option(
        None,
        "--diary-count",
        min=10,
        max=30,
        help="Number of weekday and weekend LLM diaries to generate (default: 30).",
    ),
    comparison: Optional[str] = typer.Option(
        None, "--comparison", help="Path to trajectories parquet to compare against."
    ),
    comparison_label: Optional[str] = typer.Option(None, help="Comparison series label"),
    enable_road_routing: Optional[bool] = typer.Option(
        None,
        "--enable-road-routing/--no-enable-road-routing",
        help="Route car trips over the Overture Maps road graph instead of straight-line haversine.",
    ),
):
    """Run DensityEPR fallback or config-driven simulation core."""
    loaded = load_config(config)
    tess = apply_overrides(
        loaded.tessellation,
        {
            "resolution": resolution,
            "enrich_overture": enrich_overture,
            "overture_release": overture_release,
            "min_poi_count": min_poi_count,
            "poi_tessellation": poi_tessellation,
            "relevance_column": relevance_column,
        },
    )
    road_network = apply_overrides(
        loaded.road_network,
        {
            "enabled": enable_road_routing,
        },
    )
    sim = apply_overrides(
        loaded.simulation,
        {
            "tessellation": tessellation,
            "min_lon": min_lon,
            "min_lat": min_lat,
            "max_lon": max_lon,
            "max_lat": max_lat,
            "agents": agents,
            "days": days,
            "start_date": start_date,
            "relevance_column": relevance_column,
            "output": output,
            "random_state": random_state,
            "social_graph_k": social_graph_k,
            "profile_graph_exact_threshold": profile_graph_exact_threshold,
        },
    )
    comp = apply_overrides(
        loaded.comparison,
        {
            "path": comparison,
            "label": comparison_label,
        },
    )
    llm = apply_overrides(
        loaded.llm,
        {
            "diary_count": diary_count,
        },
    )
    assert isinstance(tess, TessellationConfig)
    assert isinstance(sim, SimulationConfig)
    assert isinstance(comp, ComparisonConfig)
    assert isinstance(llm, LLMConfig)
    assert isinstance(road_network, RoadNetworkConfig)
    effective = CityBehavExConfig(
        tessellation=tess,
        simulation=sim,
        road_network=road_network,
        llm=llm,
        diaries=loaded.diaries,
        embedding=loaded.embedding,
        schedule=loaded.schedule,
        profiles=loaded.profiles,
        activities=loaded.activities,
        comparison=comp,
    )
    try:
        run_simulation(effective)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


if __name__ == "__main__":
    app()
