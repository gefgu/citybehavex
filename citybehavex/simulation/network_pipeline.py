from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import typer
from fastmob.network import build_rail_graph, build_road_graph, snap_locations_to_graph

from citybehavex.config import CityBehavExConfig
from citybehavex.simulation.spatial import _lng_column, _resolve_spatial_bounds


def _as_numpy(values: object, dtype: np.dtype) -> np.ndarray:
    """Normalize pandas and Arrow columns at the simulator boundary."""
    to_numpy = getattr(values, "to_numpy", None)
    if to_numpy is not None:
        try:
            values = to_numpy(zero_copy_only=False)
        except TypeError:
            values = to_numpy()
    return np.asarray(values, dtype=dtype)


def _build_road_graph_for_config(config: CityBehavExConfig, tessellation_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rn = config.road_network
    min_lon, min_lat, max_lon, max_lat = _resolve_spatial_bounds(config, tessellation_df)
    overture_release = rn.overture_release or config.tessellation.overture_release
    return build_road_graph(
        min_lon, min_lat, max_lon, max_lat, overture_release, rn.nodes_output, rn.edges_output
    )


def _build_rail_graph_for_config(config: CityBehavExConfig, tessellation_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rn = config.rail_network
    min_lon, min_lat, max_lon, max_lat = _resolve_spatial_bounds(config, tessellation_df)
    overture_release = rn.overture_release or config.tessellation.overture_release
    return build_rail_graph(
        min_lon,
        min_lat,
        max_lon,
        max_lat,
        overture_release,
        rn.nodes_output,
        rn.edges_output,
        rn.classes,
        rn.speed_kmh_by_class,
        rn.default_speed_kmh,
    )


def _maybe_snap_to_network(
    config: CityBehavExConfig,
    tessellation_df: pd.DataFrame,
    *,
    network_name: str,
    column_name: str,
    network_config,
    graph_builder: Callable[[CityBehavExConfig, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]],
    fallback_message: str,
) -> pd.DataFrame:
    if not network_config.enabled:
        return tessellation_df

    snap_path = Path(network_config.snap_output)
    if snap_path.exists():
        snap_df = pd.read_parquet(snap_path)
        if len(snap_df) == len(tessellation_df):
            typer.echo(f"Loading cached {network_name}-node snapping from {network_config.snap_output} ...")
            return tessellation_df.assign(**{column_name: snap_df[column_name].to_numpy()})
        typer.echo(
            f"Warning: cached {network_name}-node snapping at {network_config.snap_output} has "
            f"{len(snap_df):,} rows but tessellation has {len(tessellation_df):,} — rebuilding"
        )

    lng_col = _lng_column(tessellation_df)
    nodes_df, _edges_df = graph_builder(config, tessellation_df)
    snapped = snap_locations_to_graph(
        tessellation_df,
        nodes_df,
        network_config.snap_max_distance_m,
        lat_col="lat",
        lng_col=lng_col,
    )
    snapped = _as_numpy(snapped, np.dtype(np.int64))
    n_unsnapped = int((snapped < 0).sum())
    if n_unsnapped:
        typer.echo(
            f"Warning: {n_unsnapped:,}/{len(snapped):,} locations are farther than "
            f"{network_config.snap_max_distance_m:.0f}m from the {network_name} graph and {fallback_message}"
        )

    snap_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({column_name: snapped}).to_parquet(snap_path, index=False)
    return tessellation_df.assign(**{column_name: snapped})


def _maybe_snap_to_roads(config: CityBehavExConfig, tessellation_df: pd.DataFrame) -> pd.DataFrame:
    return _maybe_snap_to_network(
        config,
        tessellation_df,
        network_name="road",
        column_name="road_node",
        network_config=config.road_network,
        graph_builder=_build_road_graph_for_config,
        fallback_message="will fall back to straight-line routing for trips touching them",
    )


def _maybe_snap_to_rail(config: CityBehavExConfig, tessellation_df: pd.DataFrame) -> pd.DataFrame:
    return _maybe_snap_to_network(
        config,
        tessellation_df,
        network_name="rail",
        column_name="rail_node",
        network_config=config.rail_network,
        graph_builder=_build_rail_graph_for_config,
        fallback_message="will fall back to car for rail-eligible trips touching them",
    )


def build_road_network_kwargs(config: CityBehavExConfig, tessellation_df: pd.DataFrame) -> dict:
    rn = config.road_network
    if not rn.enabled or "road_node" not in tessellation_df.columns:
        return {}
    nodes_df, edges_df = _build_road_graph_for_config(config, tessellation_df)
    typer.echo(
        f"Road routing enabled: {len(nodes_df):,} nodes, {len(edges_df):,} directed edges, "
        f"max {rn.max_leg_waypoints} waypoints/leg"
    )
    return dict(
        edge_from=_as_numpy(edges_df["from_node"], np.dtype(np.int64)),
        edge_to=_as_numpy(edges_df["to_node"], np.dtype(np.int64)),
        edge_weight_ds=_as_numpy(edges_df["weight_ds"], np.dtype(np.int64)),
        node_lats=_as_numpy(nodes_df["lat"], np.dtype(np.float64)),
        node_lngs=_as_numpy(nodes_df["lng"], np.dtype(np.float64)),
        location_node=_as_numpy(tessellation_df["road_node"], np.dtype(np.int64)),
        max_leg_waypoints=rn.max_leg_waypoints,
    )


def build_rail_network_kwargs(config: CityBehavExConfig, tessellation_df: pd.DataFrame) -> dict:
    rn = config.rail_network
    if not rn.enabled or "rail_node" not in tessellation_df.columns:
        return {}
    nodes_df, edges_df = _build_rail_graph_for_config(config, tessellation_df)
    typer.echo(
        f"Rail routing enabled: {len(nodes_df):,} nodes, {len(edges_df):,} directed edges, "
        f"max {rn.max_leg_waypoints} waypoints/leg"
    )
    return dict(
        edge_from=_as_numpy(edges_df["from_node"], np.dtype(np.int64)),
        edge_to=_as_numpy(edges_df["to_node"], np.dtype(np.int64)),
        edge_weight_ds=_as_numpy(edges_df["weight_ds"], np.dtype(np.int64)),
        node_lats=_as_numpy(nodes_df["lat"], np.dtype(np.float64)),
        node_lngs=_as_numpy(nodes_df["lng"], np.dtype(np.float64)),
        location_node=_as_numpy(tessellation_df["rail_node"], np.dtype(np.int64)),
        max_leg_waypoints=rn.max_leg_waypoints,
    )
