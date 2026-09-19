from __future__ import annotations

import enum
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from .config import CityBehavExConfig, apply_overrides, load_config
from .llm import LLMConfig
from .project import doctor as run_doctor
from .project import download_yjmob, init_project
from .reports import ComparisonConfig
from .reports.comparison import (
    _ACTIVITY_CANDIDATES,
    _activities_sidecar_path,
    detect_column,
    generate_comparison_report,
    load_trajectory,
)
from .roads import RoadNetworkConfig
from .services import route_to_local_aligners, temporary_aligners
from .simulation import SimulationConfig, run_simulation
from .social.config import SocialNetworkConfig
from .tessellation import TessellationConfig, build_poi_tessellation, build_tessellation


class AlignerDevice(str, enum.Enum):
    cuda = "cuda"
    cpu = "cpu"


app = typer.Typer(help="CityBehavEx - synthetic urban mobility toolkit.")
data_app = typer.Typer(help="Download public CityBehavEx example data.")
app.add_typer(data_app, name="data")


@app.command()
def init(destination: str = typer.Argument(..., help="New empty project directory.")):
    """Create an editable public YJMOB-1k example project."""
    try:
        path = init_project(Path(destination))
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Created project -> {path}")


@data_app.command("download")
def data_download(
    dataset: str = typer.Argument(..., help="Public dataset name (currently: yjmob)."),
    project: str = typer.Option(".", "--project", help="Initialized project directory."),
):
    """Download the versioned prepared public sample data."""
    if dataset != "yjmob":
        typer.echo("Error: only the yjmob sample is currently available.", err=True)
        raise typer.Exit(1)
    try:
        download_yjmob(Path(project))
    except Exception as exc:  # noqa: BLE001 - present a CLI error, retain cause.
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo("YJMOB-1k sample data: OK")


@app.command()
def doctor(config: str = typer.Option(..., "--config", help="YAML config path.")):
    """Check configured local inputs and external service endpoints."""
    for message in run_doctor(config):
        typer.echo(message)


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
    json_output: Optional[str] = typer.Option(
        None,
        "--json",
        help="Metrics JSON output path.",
    ),
    skip_network_validation: bool = typer.Option(
        False,
        "--skip-network-validation",
        help=(
            "Skip the social/contact-network validation section (co-presence graph "
            "construction + degree/clustering/persistence/topological-overlap "
            "metrics) even if the config enables it. Useful while that section is "
            "a known slow path -- rerun without this flag later to backfill it."
        ),
    ),
    skip_network_validation_random_baseline: bool = typer.Option(
        False,
        "--skip-network-validation-random-baseline",
        help=(
            "Skip building the degree-preserving random null-model graph and its "
            "clustering/topological-overlap metrics (the synthetic_vs_random / "
            "observed_vs_random comparisons) -- roughly half of network validation's "
            "cost on dense graphs. Only synthetic_vs_observed/observed_vs_observed "
            "(what the ablation table reads) are unaffected; the web UI's random-"
            "baseline charts will be empty for reports generated with this flag."
        ),
    ),
):
    """Compute mobility comparison metrics and optionally write JSON.

    Jump lengths / radius of gyration are recomputed as road-network distance
    (instead of straight-line Haversine) when the config's road_network is
    enabled and its cached graph parquet files exist -- otherwise falls back
    to the plain Haversine-based metrics.
    """
    loaded = load_config(config)
    synthetic_path = synthetic or loaded.simulation.output
    real_path = comparison or loaded.comparison.path
    label = comparison_label or loaded.comparison.label

    if skip_network_validation and loaded.comparison.network_validation.enabled:
        typer.echo("Skipping network validation (--skip-network-validation) ...")
        loaded.comparison.network_validation.enabled = False

    if skip_network_validation_random_baseline and loaded.comparison.network_validation.random_baseline:
        typer.echo("Skipping network validation random baseline (--skip-network-validation-random-baseline) ...")
        loaded.comparison.network_validation.random_baseline = False

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
    synth_activity_col = detect_column(traj.df, _ACTIVITY_CANDIDATES)
    generate_comparison_report(
        traj=traj,
        synthetic_path=synthetic_path,
        real_path=real_path,
        observed_label=label,
        synth_activity_col=synth_activity_col,
        synthetic_activities_path=_activities_sidecar_path(synthetic_path),
        json_output_path=json_output,
        road_nodes_df=road_nodes_df,
        road_edges_df=road_edges_df,
        road_snap_max_distance_m=rn.snap_max_distance_m,
        network_validation_config=loaded.comparison.network_validation,
        transport_spatial_config=loaded.comparison.transport_spatial,
        evaluation_adaptation_config=loaded.comparison.evaluation_adaptation,
        sections=loaded.comparison.sections,
        distance_h3_resolution=loaded.comparison.distance_h3_resolution,
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
    start_aligners: bool = typer.Option(
        False,
        "--start-aligners",
        help="Temporarily start the local aligner/embedding service for this simulation.",
    ),
    aligner_device: AlignerDevice = typer.Option(
        AlignerDevice.cuda, "--aligner-device", help="Local aligner device."
    ),
    aligner_port: int = typer.Option(8090, "--aligner-port", min=1, max=65535),
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
        },
    )
    soc = apply_overrides(
        loaded.social,
        {
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
    assert isinstance(soc, SocialNetworkConfig)
    assert isinstance(comp, ComparisonConfig)
    assert isinstance(llm, LLMConfig)
    assert isinstance(road_network, RoadNetworkConfig)
    effective = CityBehavExConfig(
        tessellation=tess,
        simulation=sim,
        road_network=road_network,
        rail_network=loaded.rail_network,
        llm=llm,
        diaries=loaded.diaries,
        embedding=loaded.embedding,
        schedule=loaded.schedule,
        profiles=loaded.profiles,
        activities=loaded.activities,
        social=soc,
        comparison=comp,
    )
    if aligner_device is AlignerDevice.cpu and start_aligners:
        typer.echo("Warning: CPU aligner inference is substantially slower than CUDA.", err=True)
    try:
        if start_aligners:
            with temporary_aligners(port=aligner_port, device=aligner_device.value) as base_url:
                run_simulation(route_to_local_aligners(effective, base_url))
        else:
            run_simulation(effective)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


if __name__ == "__main__":
    app()
