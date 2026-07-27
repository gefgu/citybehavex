"""Road-network jump lengths / radius of gyration for mobility comparison.

Thin wrapper around ``fastmob.network``/``fastmob.measures.individual.network_distance``
(migrated there since none of this is citybehavex-specific: routing over a
real road/rail network instead of straight-line Haversine applies to any
mobility-analysis project, not just citybehavex's simulation).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from fastmob.measures.individual.network_distance import (
    jump_lengths_km as _fastmob_jump_lengths_km,
)
from fastmob.measures.individual.network_distance import (
    radius_of_gyration_km as _fastmob_radius_of_gyration_km,
)
from fastmob.network import RoadNetwork


def build_road_network_handle(
    edges_df: pd.DataFrame | pl.DataFrame,
    nodes_df: pd.DataFrame | pl.DataFrame,
) -> RoadNetwork:
    """Prepare a contraction hierarchy from a cached road graph's edges+nodes.

    Build once per report/payload invocation and reuse across every
    ``jump_lengths_km``/``radius_of_gyration_km`` call (CH preparation, not
    the query itself, is the expensive step).
    """
    return RoadNetwork.build(edges_df, nodes_df)


def jump_lengths_km(
    df: pd.DataFrame | pl.DataFrame,
    *,
    uid_col: str,
    lat_col: str,
    lng_col: str,
    datetime_col: str,
    network: RoadNetwork,
    snap_max_distance_m: float = 750.0,
) -> np.ndarray:
    return _fastmob_jump_lengths_km(
        df,
        network=network,
        uid_col=uid_col,
        lat_col=lat_col,
        lng_col=lng_col,
        datetime_col=datetime_col,
        snap_max_distance_m=snap_max_distance_m,
    )


def radius_of_gyration_km(
    df: pd.DataFrame | pl.DataFrame,
    *,
    uid_col: str,
    lat_col: str,
    lng_col: str,
    network: RoadNetwork,
    snap_max_distance_m: float = 750.0,
) -> Any:
    return _fastmob_radius_of_gyration_km(
        df,
        network=network,
        uid_col=uid_col,
        lat_col=lat_col,
        lng_col=lng_col,
        snap_max_distance_m=snap_max_distance_m,
    )
