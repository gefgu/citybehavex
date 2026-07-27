"""Polars frame builders for arrays returned by the Rust simulation core."""

from __future__ import annotations

import numpy as np
import polars as pl


def _datetime_seconds(values: np.ndarray) -> pl.Series:
    return pl.Series(np.asarray(values, dtype=np.int64) * 1_000_000).cast(pl.Datetime("us"))


def _build_moving_frame(
    path_agent: np.ndarray,
    path_stop_id: np.ndarray,
    path_seq: np.ndarray,
    path_lat: np.ndarray,
    path_lng: np.ndarray,
    path_t: np.ndarray,
    path_mode: np.ndarray | None = None,
) -> pl.DataFrame:
    """Build the one-row-per-waypoint `moving` frame from Rust arrays."""
    mode_arr = (
        np.asarray(path_mode, dtype=np.uint8)
        if path_mode is not None
        else np.ones(len(path_agent), dtype=np.uint8)
    )
    mode = np.select(
        [mode_arr == 2, mode_arr == 3, mode_arr == 4],
        ["walk", "bike", "rail"],
        default="car",
    )
    return pl.DataFrame(
        {
            "uid": np.asarray(path_agent, dtype=np.int64),
            "stop_id": np.asarray(path_stop_id, dtype=np.int64),
            "seq": np.asarray(path_seq, dtype=np.int32),
            "lat": np.asarray(path_lat, dtype=float),
            "lng": np.asarray(path_lng, dtype=float),
            "t": _datetime_seconds(path_t),
            "mode": mode,
        }
    )


def _build_encounters_frame(
    agent: np.ndarray,
    contact: np.ndarray,
    tile: np.ndarray,
    ts: np.ndarray,
) -> pl.DataFrame:
    """Build the one-row-per-encounter `encounters` frame from Rust arrays."""
    return pl.DataFrame(
        {
            "agent": np.asarray(agent, dtype=np.int64),
            "contact": np.asarray(contact, dtype=np.int64),
            "tile": np.asarray(tile, dtype=np.int64),
            "ts": np.asarray(ts, dtype=np.int64),
        }
    )


def _build_activity_frame(
    agent: np.ndarray,
    stop_id: np.ndarray,
    seq: np.ndarray,
    activity: np.ndarray,
    arrival: np.ndarray,
    departure: np.ndarray,
    block_id: np.ndarray,
) -> pl.DataFrame:
    """Build the one-row-per-micro-activity `activities` frame from Rust arrays."""
    return pl.DataFrame(
        {
            "uid": np.asarray(agent, dtype=np.int64),
            "stop_id": np.asarray(stop_id, dtype=np.int64),
            "seq": np.asarray(seq, dtype=np.int32),
            "activity": np.asarray(activity, dtype=np.int64),
            "arrival": _datetime_seconds(arrival),
            "departure": _datetime_seconds(departure),
            "block_id": np.asarray(block_id, dtype=np.int64),
        }
    )


def _build_trip_frame(
    agent: np.ndarray,
    loc_id: np.ndarray,
    arrival: np.ndarray,
    departure: np.ndarray,
    duration: np.ndarray,
    stop_id: np.ndarray,
    abstract_loc: np.ndarray,
    lats: np.ndarray,
    lngs: np.ndarray,
) -> pl.DataFrame:
    """Build the one-row-per-stop `trajectories` frame from Rust arrays."""
    loc_idx = np.asarray(loc_id, dtype=np.int64)
    arrival_arr = np.asarray(arrival, dtype=np.int64)
    departure_arr = np.asarray(departure, dtype=np.int64)
    abstract_loc_arr = np.asarray(abstract_loc, dtype=np.int32)
    purpose = np.where(
        abstract_loc_arr == 0, "HOME", np.where(abstract_loc_arr == 1, "WORK", "OTHER")
    )
    return pl.DataFrame(
        {
            "uid": np.asarray(agent, dtype=np.int64),
            "stop_id": np.asarray(stop_id, dtype=np.int64),
            "datetime": _datetime_seconds(arrival_arr),
            "lat": lats[loc_idx],
            "lng": lngs[loc_idx],
            "arrival": _datetime_seconds(arrival_arr),
            "departure": _datetime_seconds(departure_arr),
            "trip_duration_minutes": np.asarray(duration, dtype=np.float64) / 60.0,
            "dwell_minutes": (departure_arr - arrival_arr) / 60.0,
            "purpose": purpose,
        }
    )
