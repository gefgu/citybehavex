from __future__ import annotations

import h3
import numpy as np
import pandas as pd
from fastmob.preprocessing import latlng_to_h3

from citybehavex.config import CityBehavExConfig


def _minmax(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if len(arr) == 0:
        return arr
    lo = float(arr.min())
    hi = float(arr.max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


def _lng_column(df: pd.DataFrame) -> str:
    return "lng" if "lng" in df.columns else "lon"


def _resolve_spatial_bounds(config: CityBehavExConfig, tessellation_df: pd.DataFrame) -> tuple[float, float, float, float]:
    sim = config.simulation
    tess = config.tessellation
    min_lon = sim.min_lon if sim.min_lon is not None else tess.min_lon
    min_lat = sim.min_lat if sim.min_lat is not None else tess.min_lat
    max_lon = sim.max_lon if sim.max_lon is not None else tess.max_lon
    max_lat = sim.max_lat if sim.max_lat is not None else tess.max_lat
    if None not in [min_lon, min_lat, max_lon, max_lat]:
        return float(min_lon), float(min_lat), float(max_lon), float(max_lat)

    lng_col = _lng_column(tessellation_df)
    if {"lat", lng_col}.issubset(tessellation_df.columns) and len(tessellation_df) > 0:
        return (
            float(tessellation_df[lng_col].min()),
            float(tessellation_df["lat"].min()),
            float(tessellation_df[lng_col].max()),
            float(tessellation_df["lat"].max()),
        )
    raise ValueError(
        "POI + building location inference requires a configured bbox, a tessellation "
        "with lat/lng columns, or cached Overture building features"
    )


def h3_cell_strings(
    df: pd.DataFrame,
    resolution: int,
    *,
    lat_col: str = "lat",
    lng_col: str | None = None,
) -> np.ndarray:
    lng_col = lng_col or _lng_column(df)
    converted = latlng_to_h3(
        df[[lat_col, lng_col]],
        resolution=resolution,
        lat_col=lat_col,
        lng_col=lng_col,
        output_col="h3_cell",
    )
    raw = np.asarray(converted["h3_cell"])
    return np.array([h3.int_to_str(int(cell)) for cell in raw], dtype=object)
