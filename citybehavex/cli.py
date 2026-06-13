from __future__ import annotations

from typing import Optional

import typer

from .config import (
    CityBehavExConfig,
    ComparisonConfig,
    LLMConfig,
    SimulationConfig,
    TessellationConfig,
    apply_overrides,
    load_config,
)
from .cityview import build_cityview_file
from .simulation import run_simulation
from .tessellation import build_poi_tessellation, build_tessellation

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
        None,
        help="Output HTML path. Defaults to comparison.html.",
    ),
):
    """Generate a comparison report from existing trajectory data."""
    loaded = load_config(config)
    synthetic_path = synthetic or loaded.simulation.output
    comparison_path = comparison or loaded.comparison.path
    observed_label = comparison_label or loaded.comparison.label
    output_path = output or loaded.comparison.html

    if comparison_path is None:
        typer.echo(
            "Error: provide --comparison or set comparison.path in the config.",
            err=True,
        )
        raise typer.Exit(1)

    from .reports import generate_comparison_report_from_paths

    try:
        generate_comparison_report_from_paths(
            synthetic_path=synthetic_path,
            real_path=comparison_path,
            observed_label=observed_label,
            output_path=output_path,
        )
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


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
    comparison_html: Optional[str] = typer.Option(None, help="Output comparison HTML path"),
):
    """Run DensityEPR or config-driven DITRAS simulation."""
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
    comp = apply_overrides(
        loaded.comparison,
        {
            "path": comparison,
            "label": comparison_label,
            "html": comparison_html,
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
    effective = CityBehavExConfig(
        tessellation=tess,
        simulation=sim,
        llm=llm,
        diaries=loaded.diaries,
        comparison=comp,
    )
    try:
        run_simulation(effective)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command()
def cityview(
    config: Optional[str] = typer.Option(None, "--config", help="YAML config path"),
    min_lon: Optional[float] = typer.Option(None, help="Bounding box west longitude"),
    min_lat: Optional[float] = typer.Option(None, help="Bounding box south latitude"),
    max_lon: Optional[float] = typer.Option(None, help="Bounding box east longitude"),
    max_lat: Optional[float] = typer.Option(None, help="Bounding box north latitude"),
    overture_release: Optional[str] = typer.Option(None, help="Overture Maps release tag"),
    output: Optional[str] = typer.Option(
        None, help="Output FlatGeobuf path (default: cityview.fgb)"
    ),
):
    """Export pre-triangulated buildings/roads/green spaces to FlatGeobuf for the Bevy viewer."""
    loaded = load_config(config)
    tess = apply_overrides(
        loaded.tessellation,
        {
            "min_lon": min_lon,
            "min_lat": min_lat,
            "max_lon": max_lon,
            "max_lat": max_lat,
            "overture_release": overture_release,
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

    out = output or "cityview.fgb"
    try:
        build_cityview_file(
            tess.min_lon,
            tess.min_lat,
            tess.max_lon,
            tess.max_lat,
            tess.overture_release,
            out,
        )
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


if __name__ == "__main__":
    app()
