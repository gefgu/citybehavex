from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime as _dt
from html import escape
from pathlib import Path
from typing import Any, Optional

import h3
import numpy as np
import polars as pl
from citybehavex import _core as _cbx_core
import plotly.graph_objects as go
import skmob2
import typer
from skmob2 import (
    activity_distribution_jensen_shannon_divergence,
    activity_transition_matrix,
    activity_transition_matrix_jensen_shannon_divergence,
    bin_visitation_law_data,
    compute_visitation_law_data,
    daily_activity_distribution,
    discover_daily_motifs_from_agents,
    fit_values_to_truncated_powerlaw,
    fit_visitation_law,
    jensen_shannon_divergence,
    time_bin_matrix_jensen_shannon_divergence,
    trajectory_common_part_of_commuters_multi,
    visits_per_user_wasserstein_distance,
    waiting_times,
    wasserstein_distance,
)
from skmob_vis import (
    get_resource_bundle,
    plot_activity_transition_difference,
    plot_daily_activity_difference,
    plot_distance_frequency_law,
    plot_dwell_time_ecdf,
    plot_jump_lengths_ecdf,
    plot_lognormal_fits,
    plot_mobility_profiles,
    plot_motif_literature_comparison,
    plot_profile_metrics,
    plot_radius_of_gyration_ecdf,
    plot_stvd_comparison,
    plot_trip_duration_ecdf,
    plot_truncated_powerlaw_fits,
    plot_visit_purpose_comparison,
    plot_visits_frequency_ecdf,
)

from citybehavex.activities import build_catalog
from citybehavex.metrics import (
    build_road_network_handle,
    jump_lengths_km as road_jump_lengths_km,
    radius_of_gyration_km as road_radius_of_gyration_km,
)
from citybehavex.profiles import PROFILE_METRICS, compute_profiles
from citybehavex.reports.network_validation import (
    NETWORK_METRIC_LABELS,
    build_network_validation,
)

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
_TRANSPORT_CANDIDATES = [
    "mode", "transport_mode", "transport", "travel_mode", "trip_mode", "vehicle_mode"
]
_DEFAULT_MODE_ORDER = ["walk", "bike", "car", "rail"]

# Speed used to turn real jump lengths into a car travel-time proxy for the trip
# duration comparison. Matches the synthetic SimulationConfig.car_speed_kmh default.
CAR_SPEED_KMH = 50.0
CPC_H3_RESOLUTIONS = (7, 8, 9)

# Report sections that can be individually disabled via `ComparisonConfig.sections`
# (None/omitted = run all of them, the historical/default behavior). Wasserstein
# jump/visits/RoG/dwell/trip-duration metrics and their ECDF charts are always
# computed -- they're cheap and feed the always-on Distribution-comparisons
# section, so gating them would either be a no-op or break that section.
ACTIVITY_JSD_SECTIONS = {"activity_jsd", "activity_comparison", "motifs", "mobility_profiles"}
ALL_REPORT_SECTIONS = ACTIVITY_JSD_SECTIONS | {"cpc", "stvd", "micro_activity", "mobility_laws"}


@dataclass(frozen=True)
class ActivityVisitsResult:
    visits: pl.DataFrame
    used_heuristic: bool
    warning: Optional[str] = None


def detect_column(df: pl.DataFrame, candidates: list[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in cols_lower:
            return cols_lower[candidate.lower()]
    return None


def waiting_times_minutes(traj: skmob2.TrajDataFrame) -> list:
    secs = waiting_times(
        traj.df,
        merge=True,
        datetime_col=traj.datetime_col,
        lat_col=traj.lat_col,
        lng_col=traj.lng_col,
        uid_col=traj.uid_col,
    )
    return [s / 60 for s in secs]


def _to_datetime(col: pl.Series) -> pl.Series:
    """Coerce a datetime-ish column (string or already-parsed) to polars
    ``Datetime``, coercing unparsable values to null."""
    if col.dtype == pl.Utf8:
        return col.str.to_datetime(strict=False)
    if isinstance(col.dtype, pl.Datetime):
        return col
    return col.cast(pl.Datetime, strict=False)


def _haversine_km_np(lat1, lng1, lat2, lng2) -> np.ndarray:
    lat1_arr = np.radians(np.asarray(lat1, dtype=float))
    lng1_arr = np.radians(np.asarray(lng1, dtype=float))
    lat2_arr = np.radians(np.asarray(lat2, dtype=float))
    lng2_arr = np.radians(np.asarray(lng2, dtype=float))
    dlat = lat2_arr - lat1_arr
    dlng = lng2_arr - lng1_arr
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_arr) * np.cos(lat2_arr) * np.sin(dlng / 2.0) ** 2
    )
    return 6371.0088 * 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def _default_synthetic_moving_path(synthetic_path: Optional[str]) -> Optional[Path]:
    if not synthetic_path:
        return None
    path = Path(synthetic_path)
    return path.with_name(f"{path.stem}_moving{path.suffix}")


def _normalize_transport_mode(value: Any, mode_map: dict[str, str]) -> Optional[str]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw or raw.lower() in {"nan", "none", "null"}:
        return None
    lowered = raw.lower()
    mapped = mode_map.get(raw, mode_map.get(lowered, lowered))
    mapped = str(mapped).strip().lower()
    return mapped or None


def _transport_mode_map(config: Optional[object]) -> dict[str, str]:
    raw = getattr(config, "mode_map", {}) if config is not None else {}
    return {str(k).strip().lower(): str(v).strip().lower() for k, v in dict(raw).items()}


def _synthetic_transport_leg_records(
    moving_path: Path,
    *,
    mode_map: dict[str, str],
) -> pl.DataFrame:
    moving = pl.read_parquet(moving_path)
    required = {"uid", "stop_id", "seq", "lat", "lng", "t", "mode"}
    missing = required - set(moving.columns)
    if missing:
        raise ValueError(f"synthetic moving sidecar missing columns: {sorted(missing)}")
    if moving.is_empty():
        return pl.DataFrame(
            schema={
                "source": pl.Utf8,
                "mode": pl.Utf8,
                "jump_km": pl.Float64,
                "duration_min": pl.Float64,
            }
        )

    pdf = (
        moving.select(["uid", "stop_id", "seq", "lat", "lng", "t", "mode"])
        .with_columns(_to_datetime(moving["t"]).alias("t"))
        .drop_nulls(subset=["uid", "stop_id", "seq", "lat", "lng", "t", "mode"])
        .to_pandas()
    )
    if pdf.empty:
        return pl.DataFrame(
            schema={
                "source": pl.Utf8,
                "mode": pl.Utf8,
                "jump_km": pl.Float64,
                "duration_min": pl.Float64,
            }
        )
    rows: list[dict[str, Any]] = []
    for (_uid, _stop_id), group in pdf.sort_values(["uid", "stop_id", "seq"]).groupby(
        ["uid", "stop_id"],
        sort=False,
    ):
        if len(group) < 2:
            continue
        mode = _normalize_transport_mode(group["mode"].dropna().iloc[0], mode_map)
        if mode is None:
            continue
        lat = group["lat"].to_numpy(dtype=float)
        lng = group["lng"].to_numpy(dtype=float)
        valid = np.isfinite(lat) & np.isfinite(lng)
        if valid.sum() < 2:
            continue
        lat = lat[valid]
        lng = lng[valid]
        jump_km = float(np.nansum(_haversine_km_np(lat[:-1], lng[:-1], lat[1:], lng[1:])))
        t = group["t"].dropna()
        duration_min = None
        if len(t) >= 2:
            duration_min = float((t.max() - t.min()).total_seconds() / 60.0)
        rows.append(
            {
                "source": "synthetic",
                "mode": mode,
                "jump_km": jump_km,
                "duration_min": duration_min,
            }
        )
    return (
        pl.DataFrame(rows)
        if rows
        else pl.DataFrame(
            schema={
                "source": pl.Utf8,
                "mode": pl.Utf8,
                "jump_km": pl.Float64,
                "duration_min": pl.Float64,
            }
        )
    )


def _observed_transport_leg_records(
    observed_df: pl.DataFrame,
    *,
    uid_col: Optional[str],
    datetime_col: Optional[str],
    lat_col: Optional[str],
    lng_col: Optional[str],
    transport_col: Optional[str],
    duration_col: Optional[str],
    mode_map: dict[str, str],
) -> pl.DataFrame:
    uid = uid_col or detect_column(observed_df, _UID_CANDIDATES)
    dt = datetime_col or detect_column(observed_df, _DATETIME_CANDIDATES)
    lat = lat_col or detect_column(observed_df, _LAT_CANDIDATES)
    lng = lng_col or detect_column(observed_df, _LNG_CANDIDATES)
    mode_col = transport_col or detect_column(observed_df, _TRANSPORT_CANDIDATES)
    missing = [
        name
        for name, value in {
            "uid_col": uid,
            "datetime_col": dt,
            "lat_col": lat,
            "lng_col": lng,
            "transport_col": mode_col,
        }.items()
        if not value or value not in observed_df.columns
    ]
    if missing:
        raise ValueError(f"observed transport comparison missing columns: {', '.join(missing)}")

    select_cols = [uid, dt, lat, lng, mode_col]
    dur = duration_col if duration_col and duration_col in observed_df.columns else None
    if dur:
        select_cols.append(dur)
    pdf = (
        observed_df.select(select_cols)
        .with_columns(_to_datetime(observed_df[dt]).alias(dt))
        .drop_nulls(subset=[uid, dt, lat, lng, mode_col])
        .sort([uid, dt])
        .to_pandas()
    )
    if pdf.empty:
        return pl.DataFrame(
            schema={
                "source": pl.Utf8,
                "mode": pl.Utf8,
                "jump_km": pl.Float64,
                "duration_min": pl.Float64,
            }
        )

    rows: list[dict[str, Any]] = []
    for _uid, group in pdf.groupby(uid, sort=False):
        group = group.sort_values(dt)
        if len(group) < 2:
            continue
        prev_lat = group[lat].shift(1)
        prev_lng = group[lng].shift(1)
        prev_t = group[dt].shift(1)
        distances = _haversine_km_np(
            prev_lat.iloc[1:],
            prev_lng.iloc[1:],
            group[lat].iloc[1:],
            group[lng].iloc[1:],
        )
        for idx, jump_km in zip(group.index[1:], distances):
            if not np.isfinite(jump_km):
                continue
            mode = _normalize_transport_mode(group.at[idx, mode_col], mode_map)
            if mode is None:
                continue
            duration_min = None
            if dur:
                raw_duration = group.at[idx, dur]
                if raw_duration is not None and np.isfinite(float(raw_duration)):
                    duration_min = float(raw_duration)
            else:
                delta = group.at[idx, dt] - prev_t.loc[idx]
                if delta is not None:
                    duration_min = float(delta.total_seconds() / 60.0)
            rows.append(
                {
                    "source": "observed",
                    "mode": mode,
                    "jump_km": float(jump_km),
                    "duration_min": duration_min,
                }
            )
    return (
        pl.DataFrame(rows)
        if rows
        else pl.DataFrame(
            schema={
                "source": pl.Utf8,
                "mode": pl.Utf8,
                "jump_km": pl.Float64,
                "duration_min": pl.Float64,
            }
        )
    )


def _transport_spatial_summary(records: pl.DataFrame) -> dict[str, Any]:
    if records.is_empty():
        return {}
    summary: dict[str, Any] = {}
    for source in records["source"].unique().to_list():
        src = records.filter(pl.col("source") == source)
        total = int(len(src))
        mode_rows = []
        for mode in sorted(
            src["mode"].unique().to_list(),
            key=lambda m: (
                _DEFAULT_MODE_ORDER.index(m) if m in _DEFAULT_MODE_ORDER else 99,
                m,
            ),
        ):
            mode_df = src.filter(pl.col("mode") == mode)
            durations = mode_df["duration_min"].drop_nulls()
            mode_rows.append(
                {
                    "mode": mode,
                    "count": int(len(mode_df)),
                    "percent": float(len(mode_df) / total * 100.0) if total else 0.0,
                    "mean_jump_km": float(mode_df["jump_km"].mean()) if len(mode_df) else None,
                    "mean_duration_min": float(durations.mean()) if len(durations) else None,
                }
            )
        summary[source] = {"total_trips": total, "modes": mode_rows}
    return summary


def _transport_spatial_section_html(records: pl.DataFrame, *, observed_label: str) -> str:
    if records.is_empty():
        return ""
    summary = _transport_spatial_summary(records)
    sources = ["synthetic"]
    if "observed" in summary:
        sources.append("observed")
    mode_order = sorted(
        set(records["mode"].to_list()),
        key=lambda m: (_DEFAULT_MODE_ORDER.index(m) if m in _DEFAULT_MODE_ORDER else 99, m),
    )
    source_labels = {"synthetic": "synthetic", "observed": observed_label}

    fig_share = go.Figure()
    for source in sources:
        mode_to_percent = {row["mode"]: row["percent"] for row in summary.get(source, {}).get("modes", [])}
        fig_share.add_trace(
            go.Bar(
                name=source_labels[source],
                x=mode_order,
                y=[mode_to_percent.get(mode, 0.0) for mode in mode_order],
                hovertemplate="%{x}<br>%{y:.2f}%<extra></extra>",
            )
        )
    fig_share.update_layout(
        title="Trip share by transport mode",
        template="plotly_white",
        barmode="group",
        yaxis_title="% of trips",
        xaxis_title="Transport mode",
        margin=dict(l=48, r=20, t=56, b=48),
        width=640,
        height=420,
    )

    fig_ecdf = go.Figure()
    for source in sources:
        source_df = records.filter(pl.col("source") == source)
        for mode in mode_order:
            values = np.sort(source_df.filter(pl.col("mode") == mode)["jump_km"].to_numpy())
            if len(values) == 0:
                continue
            y = np.arange(1, len(values) + 1, dtype=float) / len(values)
            fig_ecdf.add_trace(
                go.Scatter(
                    x=values,
                    y=y,
                    mode="lines",
                    name=f"{source_labels[source]} · {mode}",
                    hovertemplate="%{x:.3f} km<br>%{y:.2%}<extra></extra>",
                )
            )
    fig_ecdf.update_layout(
        title="Jump length ECDF by transport mode",
        template="plotly_white",
        xaxis_title="Jump length (km)",
        yaxis_title="ECDF",
        margin=dict(l=48, r=20, t=56, b=48),
        width=640,
        height=420,
    )

    rows = []
    for source in sources:
        for row in summary.get(source, {}).get("modes", []):
            duration = row["mean_duration_min"]
            duration_text = f"{duration:.2f}" if duration is not None else "n/a"
            rows.append(
                "<tr>"
                f"<td>{escape(source_labels[source])}</td>"
                f"<td>{escape(row['mode'])}</td>"
                f"<td>{row['count']}</td>"
                f"<td>{row['percent']:.2f}%</td>"
                f"<td>{row['mean_jump_km']:.4f}</td>"
                f"<td>{duration_text}</td>"
                "</tr>"
            )
    table = f"""
  <div class="metrics">
    <div>
      <h2>Transport spatial mobility</h2>
      <table>
        <tr><td>source</td><td>mode</td><td>trips</td><td>share</td><td>mean jump km</td><td>mean duration min</td></tr>
        {"".join(rows)}
      </table>
    </div>
  </div>"""
    charts = (
        fig_share.to_html(full_html=False, include_plotlyjs=True)
        + fig_ecdf.to_html(full_html=False, include_plotlyjs=False)
    )
    return f"""
  <div class="section-header">
    <span>Transport-conditioned spatial mobility</span>
  </div>{table}
  <div class="charts">{charts}</div>"""


def _trajectory_od_matrix(
    df: pl.DataFrame,
    *,
    uid_col: str,
    datetime_col: str,
    lat_col: str,
    lng_col: str,
    resolution: int,
) -> pl.DataFrame:
    points = df.select([uid_col, datetime_col, lat_col, lng_col]).with_columns(
        _to_datetime(df[datetime_col]).alias("_datetime"),
        pl.col(lat_col).cast(pl.Float64, strict=False).alias("_lat"),
        pl.col(lng_col).cast(pl.Float64, strict=False).alias("_lng"),
    )
    points = points.drop_nulls(subset=[uid_col, "_datetime", "_lat", "_lng"])
    points = points.filter(
        pl.col("_lat").is_between(-90, 90) & pl.col("_lng").is_between(-180, 180)
    )
    points = points.sort([uid_col, "_datetime"])
    points = points.with_columns(
        pl.struct(["_lat", "_lng"])
        .map_elements(
            lambda row: h3.latlng_to_cell(row["_lat"], row["_lng"], resolution),
            return_dtype=pl.Utf8,
        )
        .alias("origin")
    )
    points = points.with_columns(pl.col("origin").shift(-1).over(uid_col).alias("destination"))
    trips = points.drop_nulls(subset=["destination"])
    trips = trips.filter(pl.col("origin") != pl.col("destination"))

    if trips.is_empty():
        return pl.DataFrame()

    flows = (
        trips.group_by(["origin", "destination"])
        .agg(pl.len().cast(pl.Float64).alias("count"))
        .pivot(on="destination", index="origin", values="count")
        .fill_null(0.0)
    )
    return flows


def _common_part_of_commuters(
    traj: skmob2.TrajDataFrame,
    real_traj: skmob2.TrajDataFrame,
    resolutions: tuple[int, ...] = CPC_H3_RESOLUTIONS,
) -> list[tuple[int, float]]:
    return trajectory_common_part_of_commuters_multi(traj, real_traj, resolutions=resolutions)


def _metrics_section_html(
    wasserstein_rows: list[tuple[str, str, str]],
    jsd_rows: list[tuple[str, str, str]],
    cpc_rows: list[tuple[int, float]],
) -> str:
    def table_rows(rows: list[tuple[str, str, str]]) -> str:
        return "".join(
            f"<tr><td>{name}</td><td>{value}</td><td>{unit}</td></tr>"
            for name, value, unit in rows
        )

    cpc_table_rows = "".join(
        f"<tr><td>H3 {resolution}</td><td>{value:.4f}</td><td></td></tr>"
        for resolution, value in cpc_rows
    )
    return f"""
  <div class="metrics">
    <div>
      <h2>Wasserstein distances</h2>
      <table>{table_rows(wasserstein_rows)}</table>
    </div>
    <div>
      <h2>Jensen-Shannon divergences</h2>
      <table>{table_rows(jsd_rows)}</table>
    </div>
    <div>
      <h2>Common Part of Commuters</h2>
      <table>{cpc_table_rows}</table>
    </div>
  </div>"""


def _plot_network_block(block: dict, *, title: str) -> go.Figure:
    nodes = block.get("nodes", [])
    edges = block.get("edges", [])
    x = [float(row[0]) for row in nodes if isinstance(row, list) and len(row) >= 2]
    y = [float(row[1]) for row in nodes if isinstance(row, list) and len(row) >= 2]
    degree = block.get("degrees") or [0] * len(x)

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for row in edges:
        if not isinstance(row, list) or len(row) < 2:
            continue
        source, target = int(row[0]), int(row[1])
        if source >= len(nodes) or target >= len(nodes):
            continue
        edge_x.extend([float(nodes[source][0]), float(nodes[target][0]), None])
        edge_y.extend([float(nodes[source][1]), float(nodes[target][1]), None])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=0.6, color="rgba(20,17,13,0.18)"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(
                size=[float(row[2]) if isinstance(row, list) and len(row) > 2 else 5 for row in nodes],
                color=degree,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="degree"),
                line=dict(width=0),
            ),
            text=[f"agent {row[3]}<br>degree {degree[i] if i < len(degree) else 0}" for i, row in enumerate(nodes)],
            hoverinfo="text",
            showlegend=False,
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        margin=dict(l=20, r=20, t=44, b=20),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        width=640,
        height=480,
    )
    return fig


def _network_validation_section_html(network_validation: Optional[dict]) -> str:
    if not network_validation:
        return ""
    sections: list[str] = []
    titles = {
        "synthetic_vs_random": ("Synthetic vs random", "Synthetic social + encounters"),
        "observed_vs_random": ("Observed vs random", "Observed daily co-presence"),
    }
    for block_key, block in network_validation.items():
        if not isinstance(block, dict):
            continue
        table_title, source_title = titles.get(block_key, (block_key.replace("_", " "), "Source network"))
        rows = []
        for key, label in NETWORK_METRIC_LABELS.items():
            value = block.get("wasserstein", {}).get(key)
            rows.append(
                f"<tr><td>{label}</td><td>{value:.4f}</td><td></td></tr>"
                if value is not None
                else f"<tr><td>{label}</td><td>n/a</td><td></td></tr>"
            )
        source_network = block.get("source_network")
        random_network = block.get("random_network")
        source_plot = (
            _plot_network_block(source_network, title=source_title).to_html(
                full_html=False,
                include_plotlyjs=True,
            )
            if isinstance(source_network, dict)
            else ""
        )
        random_plot = (
            _plot_network_block(random_network, title="Degree-preserving random network").to_html(
                full_html=False,
                include_plotlyjs=True,
            )
            if isinstance(random_network, dict)
            else ""
        )
        sections.append(
            f"""
  <div class="section-header">
    <span>Random network validation &mdash; {table_title}</span>
  </div>
  <div class="metrics">
    <div>
      <h2>{table_title} Wasserstein</h2>
      <table>{"".join(rows)}</table>
    </div>
  </div>
  <div class="charts">{source_plot}{random_plot}</div>"""
        )
    return "".join(sections)


_H3_INVALID_CELL = np.uint64(2**64 - 1)


def _h3_cells(lat: pl.Series, lng: pl.Series, resolution: int) -> pl.Series:
    """Vectorized lat/lng -> H3 cell index, via the Rust extension instead of
    a per-row ``h3.latlng_to_cell`` Python loop -- the difference is
    meaningful at real dataset scale (~100x measured on 100M+ rows). Returns
    a nullable ``UInt64`` series (not the hex-string form ``h3.latlng_to_cell``
    returns) since callers only group/compare locations, never display them;
    invalid/non-finite coordinates map to null.
    """
    lat_arr = lat.cast(pl.Float64, strict=False).to_numpy()
    lng_arr = lng.cast(pl.Float64, strict=False).to_numpy()
    cells = _cbx_core.batch_latlng_to_cells(lat_arr, lng_arr, resolution)
    result = pl.Series(cells, dtype=pl.UInt64)
    invalid = pl.Series(cells == _H3_INVALID_CELL)
    if invalid.any():
        result = result.set(invalid, None)
    return result


def _visits_for_comparison(
    df: pl.DataFrame,
    *,
    uid_col: str,
    datetime_col: str,
    activity_col: Optional[str] = None,
    location_col: Optional[str] = None,
    location_resolution: int = 10,
    end_col: Optional[str] = None,
    lat_col: Optional[str] = None,
    lng_col: Optional[str] = None,
) -> pl.DataFrame:
    visits = pl.DataFrame(
        {
            "uid": df[uid_col],
            "start_timestamp": _to_datetime(df[datetime_col]),
        }
    )
    if activity_col:
        visits = visits.with_columns(df[activity_col].alias("purpose"))

    if location_col:
        visits = visits.with_columns(df[location_col].cast(pl.Utf8).alias("location_id"))
    else:
        lat_name = lat_col or detect_column(df, _LAT_CANDIDATES) or "lat"
        lng_name = lng_col or detect_column(df, _LNG_CANDIDATES) or "lng"
        visits = visits.with_columns(
            _h3_cells(df[lat_name], df[lng_name], location_resolution).alias("location_id")
        )

    if end_col:
        visits = visits.with_columns(_to_datetime(df[end_col]).alias("end_timestamp"))
    else:
        visits = visits.sort(["uid", "start_timestamp"])
        visits = visits.with_columns(
            pl.col("start_timestamp").shift(-1).over("uid").alias("end_timestamp")
        )
        visits = visits.with_columns(
            pl.col("end_timestamp").fill_null(
                pl.col("start_timestamp").dt.truncate("1d") + pl.duration(days=1)
            )
        )
    return visits


def _collapse_purpose_group(value: object) -> str:
    if isinstance(value, str):
        normalized = value.strip().upper()
        if normalized in {"HOME", "WORK"}:
            return normalized
    return "OTHER"


def _collapse_explicit_purposes(visits: pl.DataFrame) -> pl.DataFrame:
    return visits.with_columns(
        pl.col("purpose").map_elements(
            _collapse_purpose_group, return_dtype=pl.Utf8, skip_nulls=False
        )
    )


def _modal_location_per_user(candidates: pl.DataFrame) -> pl.DataFrame:
    """Per-``uid`` most-frequent ``location_id`` among ``candidates`` rows.

    Ties (equal visit counts) are broken by ascending ``location_id`` for a
    deterministic result, matching the ``ORDER BY cnt DESC, fine_cell``
    convention already used by the equivalent DuckDB heuristic in
    ``web/backend/app/home_work_data.py`` (``_observed_density_heuristic``).
    Returns a ``[uid, location_id]`` lookup table; users with no candidate
    rows are absent from the result (callers should treat a missing ``uid``
    as "no home/work location found", not as a match against null).
    """
    if candidates.is_empty():
        return candidates.select(["uid", "location_id"])
    counts = (
        candidates.group_by(["uid", "location_id"], maintain_order=True)
        .agg(pl.len().alias("_count"))
        .sort(["uid", "_count", "location_id"], descending=[False, True, False])
    )
    return counts.unique(subset=["uid"], keep="first", maintain_order=True).select(
        ["uid", "location_id"]
    )


def _derive_purpose_groups_from_heuristic(visits: pl.DataFrame) -> pl.DataFrame:
    """Assign HOME/WORK/OTHER per row from time-of-day + repeated-location
    anchors, vectorized across all users at once (no per-user Python loop):
    HOME is a user's most-visited location during hour 2-5, WORK is their
    most-visited location (other than HOME) during hour 10 or 14-16.
    """
    derived = visits.with_row_index("_row")
    hour = derived["start_timestamp"].dt.hour()

    home_loc = _modal_location_per_user(derived.filter(hour.is_between(2, 5))).rename(
        {"location_id": "_home_loc"}
    )

    work_mask = hour.eq(10) | hour.is_between(14, 16)
    work_candidates = derived.filter(work_mask).join(home_loc, on="uid", how="left")
    work_candidates = work_candidates.filter(
        pl.col("_home_loc").is_null() | (pl.col("location_id") != pl.col("_home_loc"))
    )
    work_loc = _modal_location_per_user(
        work_candidates.select(["uid", "location_id"])
    ).rename({"location_id": "_work_loc"})

    derived = derived.join(home_loc, on="uid", how="left").join(work_loc, on="uid", how="left")
    is_home = pl.col("_home_loc").is_not_null() & (pl.col("location_id") == pl.col("_home_loc"))
    is_work = pl.col("_work_loc").is_not_null() & (pl.col("location_id") == pl.col("_work_loc"))
    derived = (
        derived.with_columns(
            pl.when(is_home)
            .then(pl.lit("HOME"))
            .when(is_work)
            .then(pl.lit("WORK"))
            .otherwise(pl.lit("OTHER"))
            .alias("purpose")
        )
        .sort("_row")
        .drop(["_row", "_home_loc", "_work_loc"])
    )
    return derived


def _prepare_activity_visits(
    df: pl.DataFrame,
    *,
    label: str,
    uid_col: Optional[str],
    datetime_col: Optional[str],
    activity_col: Optional[str],
    location_col: Optional[str],
    lat_col: Optional[str],
    lng_col: Optional[str],
    location_resolution: int = 10,
    end_col: Optional[str] = None,
) -> Optional[ActivityVisitsResult]:
    if uid_col is None or datetime_col is None:
        return None
    if location_col is None and (lat_col is None or lng_col is None):
        return None

    resolved_activity_col = (
        activity_col if activity_col is not None and activity_col in df.columns else None
    )
    visits = _visits_for_comparison(
        df,
        uid_col=uid_col,
        datetime_col=datetime_col,
        activity_col=resolved_activity_col,
        location_col=location_col,
        location_resolution=location_resolution,
        end_col=end_col,
        lat_col=lat_col,
        lng_col=lng_col,
    )
    visits = visits.drop_nulls(subset=["uid", "start_timestamp", "location_id"])
    if visits.is_empty():
        return None

    if resolved_activity_col:
        return ActivityVisitsResult(_collapse_explicit_purposes(visits), False)

    warning = (
        f"{label} has no explicit purpose column; derived HOME/WORK/OTHER "
        "with time-of-day and repeated-location heuristics."
    )
    return ActivityVisitsResult(
        _derive_purpose_groups_from_heuristic(visits),
        True,
        warning,
    )


def _collapse_to_stays(
    df: pl.DataFrame,
    *,
    uid_col: str,
    lat_col: str,
    lng_col: str,
    datetime_col: str,
) -> pl.DataFrame:
    """Collapse a slot-by-slot trajectory into one row per stay episode.

    The synthetic trajectory emits a record per time slot, so consecutive slots
    at the same location are the same visit. Keeping only the first row of each
    maximal same-location run per user makes "visits per user" count distinct
    stays, comparable to the observed stay-event table instead of slot density.
    """
    ordered = df.sort([uid_col, datetime_col])
    same_user = ordered[uid_col].eq(ordered[uid_col].shift())
    same_loc = ordered[lat_col].eq(ordered[lat_col].shift()) & ordered[lng_col].eq(
        ordered[lng_col].shift()
    )
    new_stay = ~(same_user & same_loc).fill_null(False)
    return ordered.filter(new_stay)


def _motif_visits(visits: pl.DataFrame) -> pl.DataFrame:
    return visits.with_columns(
        pl.when(pl.col("purpose") == "HOME").then(pl.col("purpose")).otherwise(pl.lit("VISIT")).alias("purpose")
    )


def _location_resolution(
    df: pl.DataFrame,
    location_col: Optional[str],
    default: int = 10,
) -> int:
    if location_col:
        for value in df[location_col].drop_nulls().cast(pl.Utf8):
            try:
                return h3.get_resolution(value)
            except ValueError:
                break
    return default


_STVD_ALL_HOURS = list(range(24))
_STVD_HOUR_COLS = [str(h) for h in _STVD_ALL_HOURS]


def _stvd_hourly_histogram(
    df: pl.DataFrame,
    *,
    lat_col: str,
    lng_col: str,
    datetime_col: str,
    resolutions: list[int],
) -> dict[int, pl.DataFrame]:
    """Per-H3-cell, per-hour-of-day row count, one table per resolution and
    per trajectory -- the tier-1 half of the STVD computation (pure
    per-trajectory binning, reusable across every comparison this trajectory
    participates in). ``_diff_stvd_layers`` is the tier-2 half: diffing two
    already-binned tables into the GeoJSON volume-diff/peak-shift map.
    """
    work = df.select([lat_col, lng_col, datetime_col]).with_columns(
        _to_datetime(df[datetime_col]).alias("_dt")
    )
    work = work.drop_nulls(subset=["_dt", lat_col, lng_col])
    work = work.with_columns(pl.col("_dt").dt.hour().alias("_hour"))

    layers: dict[int, pl.DataFrame] = {}
    for res in resolutions:
        # Per-row H3 binning via the Rust extension (see ``_h3_cells``) --
        # only the small number of *unique* cells below pay the h3-py string
        # round-trip (for ``cell_to_boundary``/the "area" property), not
        # every row.
        cells = _h3_cells(work[lat_col], work[lng_col], res)
        binned = work.with_columns(pl.Series("_cell", cells)).select(["_cell", "_hour"])
        binned = binned.drop_nulls(subset=["_cell"])

        hourly = (
            binned.group_by(["_cell", "_hour"])
            .agg(pl.len().alias("_count"))
            .pivot(on="_hour", index="_cell", values="_count")
        )
        missing = [h for h in _STVD_HOUR_COLS if h not in hourly.columns]
        if missing:
            hourly = hourly.with_columns([pl.lit(0).alias(h) for h in missing])
        layers[res] = hourly.select(["_cell", *_STVD_HOUR_COLS]).fill_null(0)

    return layers


def _diff_stvd_layers(
    syn_hourly: dict[int, pl.DataFrame],
    real_hourly: dict[int, pl.DataFrame],
    resolutions: list[int],
) -> dict[int, dict]:
    """Volume-diff / peak-shift classification + GeoJSON emission from two
    already-binned per-trajectory hourly tables (see ``_stvd_hourly_histogram``).
    """
    zero_row = {h: 0 for h in _STVD_HOUR_COLS}
    layers: dict[int, dict] = {}
    for res in resolutions:
        syn_lookup = {row["_cell"]: row for row in syn_hourly[res].iter_rows(named=True)}
        real_lookup = {row["_cell"]: row for row in real_hourly[res].iter_rows(named=True)}
        all_cells = set(syn_lookup) | set(real_lookup)

        features = []
        for cell in all_cells:
            syn_row = syn_lookup.get(cell, zero_row)
            real_row = real_lookup.get(cell, zero_row)

            syn_vol = float(sum(syn_row[h] for h in _STVD_HOUR_COLS))
            real_vol = float(sum(real_row[h] for h in _STVD_HOUR_COLS))
            syn_peak = max(_STVD_ALL_HOURS, key=lambda h: syn_row[str(h)]) if syn_vol > 0 else 0
            real_peak = max(_STVD_ALL_HOURS, key=lambda h: real_row[str(h)]) if real_vol > 0 else 0

            volume_diff_pct = (syn_vol - real_vol) / max(real_vol, 1.0) * 100.0
            raw_shift = abs(syn_peak - real_peak)
            peak_shift_hours = float(min(raw_shift, 12 - raw_shift if raw_shift <= 12 else raw_shift))
            peak_shift_hours = min(peak_shift_hours, 12.0)

            cell_hex = format(int(cell), "x")
            boundary = h3.cell_to_boundary(cell_hex)
            ring = [[lng, lat] for lat, lng in boundary]
            ring.append(ring[0])

            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {
                    "area": cell_hex,
                    "volume_diff_pct": round(volume_diff_pct, 4),
                    "peak_shift_hours": round(peak_shift_hours, 4),
                },
            })

        layers[res] = {"type": "FeatureCollection", "features": features}

    return layers


def _compute_stvd_layers(
    traj: skmob2.TrajDataFrame,
    real_traj: skmob2.TrajDataFrame,
    resolutions: list[int],
) -> dict[int, dict]:
    """Compute per-H3-zone volume diff and peak shift for the STVD
    visualisation -- thin composition of ``_stvd_hourly_histogram`` (tier-1)
    and ``_diff_stvd_layers`` (tier-2), kept as a single call for the
    standalone HTML report path (``generate_comparison_report``), which has
    no per-filter-group caching need.
    """
    syn_hourly = _stvd_hourly_histogram(
        traj.df,
        lat_col=traj.lat_col,
        lng_col=traj.lng_col,
        datetime_col=traj.datetime_col,
        resolutions=resolutions,
    )
    real_hourly = _stvd_hourly_histogram(
        real_traj.df,
        lat_col=real_traj.lat_col,
        lng_col=real_traj.lng_col,
        datetime_col=real_traj.datetime_col,
        resolutions=resolutions,
    )
    return _diff_stvd_layers(syn_hourly, real_hourly, resolutions)


def _split_transition_matrix_categories(matrix: Any) -> tuple[Any, list[Any] | None]:
    """``skmob2.activity_transition_matrix`` returns activity labels in the
    index for a pandas result, but embeds them in an explicit ``activity``
    column for other backends (its own documented behavior) -- split that
    column out here so ``activity_transition_matrix_jensen_shannon_divergence``
    gets a pure numeric matrix (with categories passed explicitly) either way.
    """
    if isinstance(matrix, pl.DataFrame) and "activity" in matrix.columns:
        return matrix.drop("activity"), matrix["activity"].to_list()
    return matrix, None


def _motif_distribution_jsd(
    left: pl.DataFrame,
    right: pl.DataFrame,
) -> float:
    left_counts = dict(zip(left["motif_id"], left["count"]))
    right_counts = dict(zip(right["motif_id"], right["count"]))
    labels = sorted(set(left_counts) | set(right_counts), key=str)
    return jensen_shannon_divergence(
        [left_counts.get(label, 0) for label in labels],
        [right_counts.get(label, 0) for label in labels],
    )


def _activity_comparison_section_html(
    observed_visits: Optional[pl.DataFrame],
    synthetic_visits: Optional[pl.DataFrame],
    observed_label: str,
    warnings: Optional[list[str]] = None,
) -> str:
    if observed_visits is None or synthetic_visits is None:
        return ""

    labels = (observed_label, "synthetic")
    charts_html = (
        plot_visit_purpose_comparison(
            {
                observed_label: observed_visits,
                "synthetic": synthetic_visits,
            },
            bundle_libs=False,
        )._repr_html_()
        + plot_activity_transition_difference(
            observed_visits,
            synthetic_visits,
            labels=labels,
            bundle_libs=False,
        )._repr_html_()
        + plot_daily_activity_difference(
            observed_visits,
            synthetic_visits,
            labels=labels,
            bundle_libs=False,
        )._repr_html_()
    )
    warning_html = _activity_warning_html(warnings or [])
    return f"""
  <div class="section-header">
    <span>Activity comparison &mdash; {observed_label} vs synthetic</span>
  </div>
  {warning_html}
  <div class="charts">{charts_html}</div>"""


def _activity_warning_html(warnings: list[str]) -> str:
    if not warnings:
        return ""
    items = "".join(f"<li>{escape(message)}</li>" for message in warnings)
    return f"""
  <div class="report-warning">
    <strong>Purpose heuristic warning</strong>
    <ul>{items}</ul>
  </div>"""


def _activities_sidecar_path(synthetic_path: str) -> str:
    path = Path(synthetic_path)
    return str(path.with_name(f"{path.stem}_activities{path.suffix}"))


def _micro_activity_daily_usage_figure(
    activities: pl.DataFrame,
    *,
    bin_size_minutes: int = 10,
) -> go.Figure:
    usage = _micro_activity_daily_usage_data(
        activities,
        bin_size_minutes=bin_size_minutes,
    )
    fig = go.Figure()
    for series in usage["series"]:
        fig.add_trace(
            go.Scatter(
                x=usage["x"],
                y=series["values"],
                mode="lines",
                stackgroup="micro_activity_usage",
                name=series["name"],
            )
        )
    fig.update_layout(
        title="Synthetic micro-activity mean daily usage",
        xaxis_title="Time of day",
        yaxis_title="Share of synthetic micro-activity time (%)",
        hovermode="x unified",
        template="plotly_white",
        legend_title_text="Micro-activity",
        height=460,
        margin=dict(l=48, r=24, t=64, b=48),
    )
    fig.update_xaxes(tickmode="array", tickvals=usage["x"][::6], tickangle=0)
    return fig


def _micro_activity_daily_usage_data(
    activities: pl.DataFrame,
    *,
    bin_size_minutes: int = 10,
) -> dict[str, object]:
    from datetime import timedelta

    required = ["uid", "activity", "arrival", "departure"]
    missing = sorted(set(required) - set(activities.columns))
    if missing:
        raise ValueError(f"activities table missing columns: {', '.join(missing)}")
    if bin_size_minutes <= 0 or 1440 % bin_size_minutes != 0:
        raise ValueError("bin_size_minutes must be a positive divisor of 1440")

    work = activities.select(required).with_columns(
        _to_datetime(activities["arrival"]).alias("arrival"),
        _to_datetime(activities["departure"]).alias("departure"),
        pl.col("activity").cast(pl.Float64, strict=False),
    )
    work = work.drop_nulls(subset=["arrival", "departure", "activity"])
    work = work.filter(pl.col("departure") > pl.col("arrival"))
    if work.is_empty():
        raise ValueError("activities table has no valid intervals")

    catalog = build_catalog()
    labels = {activity.idx: activity.name for activity in catalog}
    activity_ids = [activity.idx for activity in catalog]
    n_bins = 1440 // bin_size_minutes
    seconds = np.zeros((len(activity_ids), n_bins), dtype=float)
    id_to_row = {activity_id: row for row, activity_id in enumerate(activity_ids)}

    bin_seconds = bin_size_minutes * 60
    one_day = timedelta(days=1)
    for row in work.iter_rows(named=True):
        activity_id = int(row["activity"])
        if activity_id not in id_to_row:
            continue
        current = row["arrival"]
        end = row["departure"]
        while current < end:
            midnight = current.replace(hour=0, minute=0, second=0, microsecond=0)
            next_midnight = midnight + one_day
            segment_end = min(end, next_midnight)
            start_second = int((current - midnight).total_seconds())
            end_second = int((segment_end - midnight).total_seconds())
            start_bin = start_second // bin_seconds
            end_bin = max(start_bin, (end_second - 1) // bin_seconds)
            for bin_idx in range(start_bin, min(end_bin + 1, n_bins)):
                bin_start = midnight + timedelta(seconds=bin_idx * bin_seconds)
                bin_end = bin_start + timedelta(seconds=bin_seconds)
                overlap = (min(segment_end, bin_end) - max(current, bin_start)).total_seconds()
                if overlap > 0:
                    seconds[id_to_row[activity_id], bin_idx] += overlap
            current = segment_end

    totals = seconds.sum(axis=0)
    percentages = np.divide(
        seconds * 100.0,
        totals,
        out=np.zeros_like(seconds),
        where=totals > 0,
    )
    x = [
        f"{minute // 60:02d}:{minute % 60:02d}"
        for minute in range(0, 1440, bin_size_minutes)
    ]
    return {
        "bin_size_minutes": bin_size_minutes,
        "n_bins": n_bins,
        "x": x,
        "series": [
            {
                "activity_id": activity_id,
                "name": labels[activity_id],
                "values": percentages[id_to_row[activity_id]].round(6).tolist(),
            }
            for activity_id in activity_ids
        ],
    }


def _micro_activity_section_html(activities_path: Optional[str]) -> str:
    if not activities_path:
        return ""
    path = Path(activities_path)
    if not path.exists():
        typer.echo(f"Warning: micro-activity chart skipped: {path} not found", err=True)
        return ""
    try:
        activities = pl.read_parquet(path)
        if activities.is_empty():
            raise ValueError("activities table is empty")
        fig = _micro_activity_daily_usage_figure(activities)
    except Exception as exc:
        typer.echo(f"Warning: micro-activity chart skipped: {exc}", err=True)
        return ""

    return f"""
  <div class="section-header">
    <span>Synthetic micro-activity usage</span>
  </div>
  <div class="charts">{fig.to_html(full_html=False, include_plotlyjs=True)}</div>"""


def _mobility_law_visits(
    df: pl.DataFrame,
    *,
    uid_col: str,
    datetime_col: str,
    lat_col: str,
    lng_col: str,
    location_col: Optional[str] = None,
    activity_col: Optional[str] = None,
    location_resolution: int = 10,
) -> pl.DataFrame:
    columns = [uid_col, datetime_col, lat_col, lng_col]
    if location_col:
        columns.append(location_col)
    if activity_col:
        columns.append(activity_col)

    source = df.select(columns).with_columns(
        _to_datetime(df[datetime_col]).alias(datetime_col),
        pl.col(lat_col).cast(pl.Float64, strict=False),
        pl.col(lng_col).cast(pl.Float64, strict=False),
    )
    source = source.drop_nulls(subset=[uid_col, datetime_col, lat_col, lng_col])
    source = source.filter(
        pl.col(lat_col).is_between(-90, 90) & pl.col(lng_col).is_between(-180, 180)
    )

    visits = pl.DataFrame(
        {
            "user_id": source[uid_col],
            "timestamp": source[datetime_col],
            "lat": source[lat_col],
            "lng": source[lng_col],
        }
    )
    if location_col:
        missing = source[location_col].is_null()
        location_id = source[location_col].cast(pl.Utf8)
        # Only pay for H3 conversion on rows that actually need the
        # fallback -- e.g. shanghai/yjmob always have a populated location
        # column here, so this is skipped entirely for them.
        if missing.any():
            fallback = _h3_cells(
                visits["lat"].filter(missing), visits["lng"].filter(missing), location_resolution
            ).cast(pl.Utf8)
            location_id = location_id.scatter(missing.arg_true(), fallback)
        visits = visits.with_columns(location_id.alias("location_id"))
    else:
        visits = visits.with_columns(
            _h3_cells(visits["lat"], visits["lng"], location_resolution).alias("location_id")
        )
    if activity_col:
        visits = visits.with_columns(source[activity_col].alias("purpose"))
    return visits


def _daily_location_lognormal_dataset(
    visits: pl.DataFrame,
    label: str,
) -> tuple[np.ndarray, np.ndarray, float, float, str]:
    daily = (
        visits.with_columns(pl.col("timestamp").dt.truncate("1d").alias("date"))
        .group_by(["user_id", "date"])
        .agg(pl.col("location_id").n_unique().alias("_count"))
    )
    values = daily["_count"].cast(pl.Float64).to_numpy()
    values = values[np.isfinite(values) & (values > 0)]
    if values.size < 2:
        raise ValueError("at least two daily location counts are required")

    log_values = np.log(values)
    mu = float(log_values.mean())
    sigma = float(log_values.std())
    if not np.isfinite(sigma) or sigma <= 1e-12:
        raise ValueError("daily location counts must have positive log variance")

    x_points, counts = np.unique(values, return_counts=True)
    y_points = counts / counts.sum()
    return x_points, y_points, mu, sigma, label


def _truncated_powerlaw_dataset(
    values: list | np.ndarray,
    label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    filtered = np.asarray(values, dtype=float)
    filtered = filtered[np.isfinite(filtered) & (filtered > 0)]
    if filtered.size < 2 or np.unique(filtered).size < 2:
        raise ValueError("at least two distinct positive values are required")
    parameters, x_points, y_points = fit_values_to_truncated_powerlaw(
        filtered.tolist()
    )
    return parameters, x_points, y_points, label


def _distance_frequency_dataset(
    visits: pl.DataFrame,
    label: str,
) -> tuple[np.ndarray, np.ndarray, float, float, str]:
    purpose_col = "purpose" if "purpose" in visits.columns else None
    law_data = compute_visitation_law_data(
        visits,
        user_id_col="user_id",
        location_id_col="location_id",
        timestamp_col="timestamp",
        purpose_col=purpose_col,
        lat_col="lat",
        lng_col="lng",
    )
    rf_points, rho_points, _ = bin_visitation_law_data(
        law_data,
        user_id_col="user_id",
        location_id_col="location_id",
    )
    eta, mu, _ = fit_visitation_law(rf_points, rho_points)
    if eta <= 0 or mu <= 0:
        raise ValueError("distance-frequency fit parameters must be positive")
    return rf_points, rho_points, eta, mu, label


def _fit_parameters_html(
    formula: str,
    rows: list[tuple[str, list[tuple[str, float]]]],
) -> str:
    parameter_rows = "".join(
        "<tr>"
        f"<td>{label}</td>"
        f"<td>{', '.join(f'{name}={value:.4g}' for name, value in parameters)}</td>"
        "</tr>"
        for label, parameters in rows
    )
    return f"""
    <div class="fit-parameters">
      <div class="fit-formula">{formula}</div>
      <table>{parameter_rows}</table>
    </div>"""


def _mobility_laws_section_html(
    *,
    observed_visits: pl.DataFrame,
    synthetic_visits: pl.DataFrame,
    observed_jumps: list | np.ndarray,
    synthetic_jumps: list | np.ndarray,
    observed_rog: list | np.ndarray,
    synthetic_rog: list | np.ndarray,
    observed_label: str,
) -> str:
    chart_html: list[str] = []

    def render(name: str, build_chart) -> None:
        try:
            figure, parameters_html = build_chart()
            chart_html.append(
                '<div class="mobility-law-chart">'
                f"{figure._repr_html_()}{parameters_html}"
                "</div>"
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            typer.echo(f"Warning: {name} mobility-law chart skipped: {exc}", err=True)

    def truncated_powerlaw_chart(values_observed, values_synthetic, **plot_kwargs):
        observed = _truncated_powerlaw_dataset(values_observed, observed_label)
        synthetic = _truncated_powerlaw_dataset(values_synthetic, "synthetic")
        figure = plot_truncated_powerlaw_fits(
            observed,
            synthetic,
            bundle_libs=False,
            **plot_kwargs,
        )
        return figure, _fit_parameters_html(
            "p(x) = c (x + r0)<sup>-beta</sup> exp(-x / kappa)",
            [
                (
                    observed[3],
                    list(zip(("c", "r0", "beta", "kappa"), observed[0])),
                ),
                (
                    synthetic[3],
                    list(zip(("c", "r0", "beta", "kappa"), synthetic[0])),
                ),
            ],
        )

    render(
        "travel-distance",
        lambda: truncated_powerlaw_chart(
            observed_jumps,
            synthetic_jumps,
            title="Travel-distance mobility law",
        ),
    )
    render(
        "radius-of-gyration",
        lambda: truncated_powerlaw_chart(
            observed_rog,
            synthetic_rog,
            title="Radius-of-gyration mobility law",
            x_label="radius of gyration · km",
            y_label="P(r_g)",
        ),
    )

    def lognormal_chart():
        observed = _daily_location_lognormal_dataset(
            observed_visits,
            observed_label,
        )
        synthetic = _daily_location_lognormal_dataset(
            synthetic_visits,
            "synthetic",
        )
        figure = plot_lognormal_fits(observed, synthetic, bundle_libs=False)
        return figure, _fit_parameters_html(
            "f(N) = exp(-(ln N - mu)<sup>2</sup> / (2 sigma<sup>2</sup>)) "
            "/ (N sigma sqrt(2 pi))",
            [
                (observed[4], [("mu", observed[2]), ("sigma", observed[3])]),
                (synthetic[4], [("mu", synthetic[2]), ("sigma", synthetic[3])]),
            ],
        )

    render(
        "daily-locations log-normal",
        lognormal_chart,
    )

    def distance_frequency_chart():
        observed = _distance_frequency_dataset(observed_visits, observed_label)
        synthetic = _distance_frequency_dataset(synthetic_visits, "synthetic")
        figure = plot_distance_frequency_law(observed, synthetic, bundle_libs=False)
        return figure, _fit_parameters_html(
            "rho(r, f) = mu (r f)<sup>-eta</sup>",
            [
                (observed[4], [("eta", observed[2]), ("mu", observed[3])]),
                (synthetic[4], [("eta", synthetic[2]), ("mu", synthetic[3])]),
            ],
        )

    render(
        "distance-frequency",
        distance_frequency_chart,
    )

    if not chart_html:
        return ""
    return f"""
  <div class="section-header">
    <span>Mobility laws &mdash; {observed_label} vs synthetic</span>
  </div>
  <div class="charts mobility-law-charts">{"".join(chart_html)}</div>"""


def load_trajectory(path: str) -> skmob2.TrajDataFrame:
    df = pl.read_parquet(path)
    datetime_col = detect_column(df, _DATETIME_CANDIDATES)
    lat_col = detect_column(df, _LAT_CANDIDATES)
    lng_col = detect_column(df, _LNG_CANDIDATES)
    uid_col = detect_column(df, _UID_CANDIDATES)
    missing = [
        name
        for name, column in [
            ("datetime", datetime_col),
            ("latitude", lat_col),
            ("longitude", lng_col),
            ("user ID", uid_col),
        ]
        if column is None
    ]
    if missing:
        raise ValueError(
            f"{path} is missing recognizable columns for: {', '.join(missing)}"
        )
    return skmob2.TrajDataFrame(
        df,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
    )


def generate_comparison_report_from_paths(
    synthetic_path: str,
    real_path: str,
    observed_label: str,
    output_path: str,
    json_output_path: Optional[str] = None,
    sections: Optional[list[str]] = None,
) -> None:
    typer.echo(f"Loading synthetic trajectories from {synthetic_path} ...")
    traj = load_trajectory(synthetic_path)
    synth_activity_col = detect_column(traj.df, _ACTIVITY_CANDIDATES)
    generate_comparison_report(
        traj=traj,
        synthetic_path=synthetic_path,
        real_path=real_path,
        observed_label=observed_label,
        output_path=output_path,
        synth_activity_col=synth_activity_col,
        synthetic_activities_path=_activities_sidecar_path(synthetic_path),
        json_output_path=json_output_path,
        sections=sections,
    )


def generate_comparison_report(
    traj: skmob2.TrajDataFrame,
    real_path: str,
    observed_label: str,
    output_path: str,
    synthetic_path: Optional[str] = None,
    synth_activity_col: Optional[str] = None,
    synthetic_activities_path: Optional[str] = None,
    json_output_path: Optional[str] = None,
    sections: Optional[list[str]] = None,
    road_nodes_df: Optional[pl.DataFrame] = None,
    road_edges_df: Optional[pl.DataFrame] = None,
    road_snap_max_distance_m: float = 750.0,
    network_validation_config: Optional[object] = None,
    transport_spatial_config: Optional[object] = None,
) -> None:
    if sections is not None:
        unknown = set(sections) - ALL_REPORT_SECTIONS
        if unknown:
            raise ValueError(
                f"Unknown comparison report section(s): {sorted(unknown)}. "
                f"Valid sections: {sorted(ALL_REPORT_SECTIONS)}"
            )
    enabled_sections = set(sections) if sections is not None else set(ALL_REPORT_SECTIONS)
    need_activity_visits = bool(enabled_sections & ACTIVITY_JSD_SECTIONS)
    metrics: dict = {"wasserstein": {}, "jsd": {}}
    typer.echo(f"Loading observed trajectories from {real_path} ...")
    real_df = pl.read_parquet(real_path)
    _dt_col = detect_column(real_df, _DATETIME_CANDIDATES)
    if _dt_col and not isinstance(real_df.schema[_dt_col], pl.Datetime):
        real_df = real_df.with_columns(_to_datetime(real_df[_dt_col]).alias(_dt_col))
    real_traj = skmob2.TrajDataFrame(
        real_df,
        datetime_col=_dt_col,
        lat_col=detect_column(real_df, _LAT_CANDIDATES),
        lng_col=detect_column(real_df, _LNG_CANDIDATES),
        uid_col=detect_column(real_df, _UID_CANDIDATES),
    )

    typer.echo("Computing mobility metrics ...")
    labels = ("synthetic", observed_label)

    # When a cached road graph is supplied, recompute jump lengths / radius of
    # gyration as road-network distance (instead of skmob2's straight-line
    # Haversine) for both synthetic and real trajectories -- otherwise fall
    # back to the plain skmob2 calls unchanged.
    road_handle = (
        build_road_network_handle(road_edges_df)
        if road_nodes_df is not None and road_edges_df is not None and len(road_nodes_df) and len(road_edges_df)
        else None
    )
    if road_handle is not None:
        synth_jumps = road_jump_lengths_km(
            traj.df,
            uid_col=traj.uid_col,
            lat_col=traj.lat_col,
            lng_col=traj.lng_col,
            datetime_col=traj.datetime_col,
            handle=road_handle,
            nodes_df=road_nodes_df,
            snap_max_distance_m=road_snap_max_distance_m,
        )
        real_jumps = road_jump_lengths_km(
            real_traj.df,
            uid_col=real_traj.uid_col,
            lat_col=real_traj.lat_col,
            lng_col=real_traj.lng_col,
            datetime_col=real_traj.datetime_col,
            handle=road_handle,
            nodes_df=road_nodes_df,
            snap_max_distance_m=road_snap_max_distance_m,
        )
    else:
        # jump_lengths(merge=True) returns "a backend-appropriate array
        # object" per skmob2's own docs -- for a polars-backed TrajDataFrame
        # that's an Arrow-backed array whose elements are pyarrow scalars,
        # not plain floats, so normalize to a numpy array before any
        # downstream arithmetic/comparisons.
        synth_jumps = np.asarray(traj.jump_lengths(merge=True), dtype=float)
        real_jumps = np.asarray(real_traj.jump_lengths(merge=True), dtype=float)
    w_jump = wasserstein_distance(synth_jumps, real_jumps)
    metrics["wasserstein"]["jump_lengths_km"] = w_jump

    # Collapse the slot-by-slot synthetic trajectory into distinct stay episodes
    # so visits-per-user counts visits (not 15-min slots), comparable to the
    # observed stay-event table.
    synth_stays = _collapse_to_stays(
        traj.df,
        uid_col=traj.uid_col,
        lat_col=traj.lat_col,
        lng_col=traj.lng_col,
        datetime_col=traj.datetime_col,
    )
    synth_visits = synth_stays[traj.uid_col].value_counts()["count"].to_list()
    real_visits = real_traj.df[real_traj.uid_col].value_counts()["count"].to_list()
    w_visits, _ = visits_per_user_wasserstein_distance(
        synth_stays,
        real_df,
        user_id_col1=traj.uid_col,
        user_id_col2=real_traj.uid_col,
    )
    metrics["wasserstein"]["visits_per_user"] = w_visits

    if road_handle is not None:
        synth_rog = road_radius_of_gyration_km(
            traj.df,
            uid_col=traj.uid_col,
            lat_col=traj.lat_col,
            lng_col=traj.lng_col,
            handle=road_handle,
            nodes_df=road_nodes_df,
            snap_max_distance_m=road_snap_max_distance_m,
        )["radius_of_gyration"].to_numpy()
        real_rog = road_radius_of_gyration_km(
            real_traj.df,
            uid_col=real_traj.uid_col,
            lat_col=real_traj.lat_col,
            lng_col=real_traj.lng_col,
            handle=road_handle,
            nodes_df=road_nodes_df,
            snap_max_distance_m=road_snap_max_distance_m,
        )["radius_of_gyration"].to_numpy()
    else:
        synth_rog = traj.radius_of_gyration()["radius_of_gyration"].to_numpy()
        real_rog = real_traj.radius_of_gyration()["radius_of_gyration"].to_numpy()
    w_rog = wasserstein_distance(synth_rog, real_rog)
    metrics["wasserstein"]["radius_of_gyration_km"] = w_rog

    if "cpc" in enabled_sections:
        typer.echo("Computing Common Part of Commuters ...")
        cpc_rows = _common_part_of_commuters(traj, real_traj)
    else:
        cpc_rows = []
    metrics["cpc"] = {f"h3_{resolution}": value for resolution, value in cpc_rows}

    # Dwell time = time spent at a location. The synthetic simulation records this
    # directly as departure - arrival (`dwell_minutes`); otherwise fall back to
    # inter-event gaps. The observed side uses the real stay-duration column when
    # present (NOT inter-event gaps, which on a sparse visit table span days).
    duration_col = detect_column(real_df, _DURATION_CANDIDATES)
    if "dwell_minutes" in traj.df.columns:
        synth_dwell = [d for d in traj.df["dwell_minutes"].drop_nulls().to_list() if d >= 0]
    else:
        synth_dwell = waiting_times_minutes(traj)
    if duration_col:
        real_dwell = real_df[duration_col].drop_nulls().to_list()
    else:
        real_dwell = waiting_times_minutes(real_traj)
    w_dwell = wasserstein_distance(synth_dwell, real_dwell)
    metrics["wasserstein"]["dwell_time_min"] = w_dwell

    # Trip (travel) duration. The synthetic side carries a genuine car trip
    # duration per leg; the observed visit table has no travel-time ground truth,
    # so the real comparator is a car-time proxy from real jump lengths at the same
    # speed (km / CAR_SPEED_KMH * 60), making both sides directly comparable.
    if "trip_duration_minutes" in traj.df.columns:
        synth_trip = [t for t in traj.df["trip_duration_minutes"].drop_nulls().to_list() if t > 0]
        real_trip = [(j / CAR_SPEED_KMH) * 60.0 for j in real_jumps if j > 0]
        w_trip = wasserstein_distance(synth_trip, real_trip) if synth_trip and real_trip else None
    elif duration_col:
        real_trip = real_df[duration_col].drop_nulls().to_list()
        synth_trip = waiting_times_minutes(traj)
        w_trip = wasserstein_distance(synth_trip, real_trip)
    else:
        real_trip = synth_trip = w_trip = None
    if w_trip is not None:
        metrics["wasserstein"]["trip_duration_min"] = w_trip

    network_validation = None
    nv_cfg = network_validation_config
    nv_enabled = bool(getattr(nv_cfg, "enabled", False)) if nv_cfg is not None else False
    if synthetic_path is not None and nv_enabled:
        try:
            network_validation, network_warnings = build_network_validation(
                synthetic_path,
                observed_df=real_df,
                observed_uid_col=real_traj.uid_col,
                observed_datetime_col=real_traj.datetime_col,
                enabled=True,
                synthetic_enabled=bool(getattr(nv_cfg, "synthetic_enabled", True)),
                observed_enabled=bool(getattr(nv_cfg, "observed_enabled", False)),
                location_mode=str(getattr(nv_cfg, "location_mode", "auto")),
                location_col=getattr(nv_cfg, "location_col", None),
                h3_resolution=int(getattr(nv_cfg, "h3_resolution", 9)),
                max_group_size=int(getattr(nv_cfg, "max_group_size", 200)),
                seed=int(getattr(nv_cfg, "random_seed", 42)),
            )
            if network_validation is not None:
                metrics["network_validation"] = network_validation
            for warning in network_warnings:
                typer.echo(f"Warning: network validation: {warning}", err=True)
        except Exception as exc:
            typer.echo(f"Warning: network validation skipped: {exc}", err=True)

    js_rows: list[tuple[str, str, str]] = []
    synthetic_visits = None
    observed_visits = None
    activity_warnings: list[str] = []
    real_activity_col = detect_column(real_df, _ACTIVITY_CANDIDATES)
    real_start_col = detect_column(real_df, _DATETIME_CANDIDATES)
    real_end_col = detect_column(real_df, _END_TS_CANDIDATES)
    real_location_col = detect_column(real_df, _LOCATION_CANDIDATES)
    synth_location_col = detect_column(traj.df, _LOCATION_CANDIDATES)
    location_resolution = _location_resolution(real_df, real_location_col)

    if need_activity_visits:
        synthetic_visit_result = _prepare_activity_visits(
            traj.df,
            label="synthetic",
            uid_col=traj.uid_col,
            datetime_col=traj.datetime_col,
            activity_col=(
                synth_activity_col
                if synth_activity_col and synth_activity_col in traj.df.columns
                else None
            ),
            location_col=synth_location_col,
            lat_col=traj.lat_col,
            lng_col=traj.lng_col,
            location_resolution=location_resolution,
        )
        observed_visit_result = _prepare_activity_visits(
            real_df,
            label=observed_label,
            uid_col=real_traj.uid_col,
            datetime_col=real_start_col,
            activity_col=real_activity_col,
            location_col=real_location_col,
            lat_col=real_traj.lat_col,
            lng_col=real_traj.lng_col,
            location_resolution=location_resolution,
            end_col=real_end_col,
        )
        if synthetic_visit_result is not None:
            synthetic_visits = synthetic_visit_result.visits
            if synthetic_visit_result.warning:
                activity_warnings.append(synthetic_visit_result.warning)
        if observed_visit_result is not None:
            observed_visits = observed_visit_result.visits
            if observed_visit_result.warning:
                activity_warnings.append(observed_visit_result.warning)
        for warning in activity_warnings:
            typer.echo(f"Warning: {warning}", err=True)

    if (
        "activity_jsd" in enabled_sections
        and synthetic_visits is not None
        and observed_visits is not None
    ):
        activity_distribution_jsd = activity_distribution_jensen_shannon_divergence(
            synthetic_visits, observed_visits
        )
        metrics["jsd"]["activity_distribution"] = activity_distribution_jsd
        js_rows.append(
            (
                "Activity distribution",
                f"{activity_distribution_jsd:.4f}",
                "",
            )
        )
        synth_transition, synth_transition_categories = _split_transition_matrix_categories(
            activity_transition_matrix(synthetic_visits)
        )
        real_transition, real_transition_categories = _split_transition_matrix_categories(
            activity_transition_matrix(observed_visits)
        )
        activity_transitions_jsd = activity_transition_matrix_jensen_shannon_divergence(
            synth_transition,
            real_transition,
            categories1=synth_transition_categories,
            categories2=real_transition_categories,
        )
        metrics["jsd"]["activity_transitions"] = activity_transitions_jsd
        js_rows.append(
            (
                "Activity transitions",
                f"{activity_transitions_jsd:.4f}",
                "",
            )
        )
        synth_daily, synth_categories, _ = daily_activity_distribution(
            synthetic_visits
        )
        real_daily, real_categories, _ = daily_activity_distribution(
            observed_visits
        )
        daily_activity_profile_jsd = time_bin_matrix_jensen_shannon_divergence(
            synth_daily, real_daily, synth_categories, real_categories
        )
        metrics["jsd"]["daily_activity_profile"] = daily_activity_profile_jsd
        js_rows.append(
            (
                "Daily activity profile",
                f"{daily_activity_profile_jsd:.4f}",
                "",
            )
        )

    typer.echo("Rendering distribution charts ...")
    fig_jump = plot_jump_lengths_ecdf(synth_jumps, real_jumps, labels=labels, bundle_libs=False)
    fig_visits = plot_visits_frequency_ecdf(synth_visits, real_visits, labels=labels, bundle_libs=False)
    fig_rog = plot_radius_of_gyration_ecdf(synth_rog, real_rog, labels=labels, bundle_libs=False)
    fig_dwell = plot_dwell_time_ecdf(synth_dwell, real_dwell, labels=labels, bundle_libs=False)
    fig_trip = (
        plot_trip_duration_ecdf(synth_trip, real_trip, labels=labels, bundle_libs=False)
        if real_trip is not None
        else None
    )

    ecdf_charts_html = "".join(
        f._repr_html_()
        for f in [fig_jump, fig_visits, fig_rog, fig_dwell]
    )
    if fig_trip:
        ecdf_charts_html += fig_trip._repr_html_()

    if "mobility_laws" in enabled_sections:
        typer.echo("Rendering mobility-law charts ...")
        real_location_col = detect_column(real_df, _LOCATION_CANDIDATES)
        real_activity_col = detect_column(real_df, _ACTIVITY_CANDIDATES)
        mobility_observed_visits = _mobility_law_visits(
            real_df,
            uid_col=real_traj.uid_col,
            datetime_col=real_traj.datetime_col,
            lat_col=real_traj.lat_col,
            lng_col=real_traj.lng_col,
            location_col=real_location_col,
            activity_col=real_activity_col,
        )
        mobility_synthetic_visits = _mobility_law_visits(
            traj.df,
            uid_col=traj.uid_col,
            datetime_col=traj.datetime_col,
            lat_col=traj.lat_col,
            lng_col=traj.lng_col,
            location_col=synth_location_col,
            activity_col=(
                synth_activity_col
                if synth_activity_col and synth_activity_col in traj.df.columns
                else None
            ),
        )
        mobility_laws_section_html = _mobility_laws_section_html(
            observed_visits=mobility_observed_visits,
            synthetic_visits=mobility_synthetic_visits,
            observed_jumps=real_jumps,
            synthetic_jumps=synth_jumps,
            observed_rog=real_rog,
            synthetic_rog=synth_rog,
            observed_label=observed_label,
        )
    else:
        mobility_laws_section_html = ""

    if "activity_comparison" in enabled_sections:
        if observed_visits is not None and synthetic_visits is not None:
            typer.echo(
                f"Rendering activity comparison for {observed_label} and synthetic trajectories ..."
            )
        activity_section_html = _activity_comparison_section_html(
            observed_visits,
            synthetic_visits,
            observed_label,
            activity_warnings,
        )
    else:
        activity_section_html = ""

    if "micro_activity" in enabled_sections:
        micro_activity_section_html = _micro_activity_section_html(synthetic_activities_path)
    else:
        micro_activity_section_html = ""

    motif_section_html = ""
    if "motifs" in enabled_sections:
        try:
            if observed_visits is not None:
                observed_motif_visits = _motif_visits(observed_visits)
                _, real_motif_dist = discover_daily_motifs_from_agents(
                    observed_motif_visits,
                    user_id_col="uid",
                    location_id_col="location_id",
                    purpose_col="purpose",
                    timestamp_col="start_timestamp",
                    end_timestamp_col="end_timestamp",
                )
            else:
                real_motif_dist = None

            synth_motif_dist = None
            if synthetic_visits is not None:
                synthetic_motif_visits = _motif_visits(synthetic_visits)
                _, synth_motif_dist = discover_daily_motifs_from_agents(
                    synthetic_motif_visits,
                    user_id_col="uid",
                    location_id_col="location_id",
                    purpose_col="purpose",
                    timestamp_col="start_timestamp",
                    end_timestamp_col="end_timestamp",
                )
                if real_motif_dist is not None:
                    daily_motifs_jsd = _motif_distribution_jsd(synth_motif_dist, real_motif_dist)
                    metrics["jsd"]["daily_motifs"] = daily_motifs_jsd
                    js_rows.append(
                        (
                            "Daily motifs",
                            f"{daily_motifs_jsd:.4f}",
                            "",
                        )
                    )

            if real_motif_dist is not None or synth_motif_dist is not None:
                fig_motif = plot_motif_literature_comparison(
                    reference_distribution=real_motif_dist,
                    comparison_distribution=synth_motif_dist,
                    labels=(observed_label, "synthetic"),
                    bundle_libs=False,
                )
                motif_section_html = f"""
  <div class="section-header">
    <span>Motif comparison &mdash; literature vs {observed_label}{" vs synthetic" if synth_motif_dist is not None else ""}</span>
  </div>
  <div class="charts">{fig_motif._repr_html_()}</div>"""
        except Exception as exc:
            typer.echo(f"Warning: motif chart skipped: {exc}", err=True)

    stvd_section_html = ""
    if "stvd" in enabled_sections and traj.lat_col and traj.lng_col and real_traj.lat_col and real_traj.lng_col:
        try:
            typer.echo("Rendering STVD map ...")
            stvd_layers = _compute_stvd_layers(traj, real_traj, resolutions=[7, 9])
            fig_stvd = plot_stvd_comparison(
                stvd_layers,
                title=f"STVD — {observed_label} vs synthetic",
                bundle_libs=False,
            )
            stvd_section_html = f"""
  <div class="section-header">
    <span>Spatial-temporal volume difference &mdash; {observed_label} vs synthetic</span>
  </div>
  <div class="charts stvd-section">{fig_stvd._repr_html_()}</div>"""
        except Exception as exc:
            typer.echo(f"Warning: STVD chart skipped: {exc}", err=True)

    profiles_section_html = ""
    if "mobility_profiles" in enabled_sections and observed_visits is not None and synthetic_visits is not None:
        try:
            typer.echo("Rendering mobility profiles ...")
            obs_profiles = compute_profiles(observed_visits)
            synth_profiles = compute_profiles(synthetic_visits)
            fig_profiles_obs = plot_mobility_profiles(
                obs_profiles, title=observed_label, bundle_libs=False
            )
            fig_profiles_synth = plot_mobility_profiles(
                synth_profiles, title="synthetic", bundle_libs=False
            )
            fig_profile_metrics = plot_profile_metrics(
                {"synthetic": synth_profiles, observed_label: obs_profiles},
                metrics=PROFILE_METRICS,
                bundle_libs=False,
            )
            profiles_section_html = f"""
  <div class="section-header">
    <span>Mobility profiles &mdash; {observed_label} vs synthetic</span>
  </div>
  <div class="charts">{fig_profiles_obs._repr_html_()}{fig_profiles_synth._repr_html_()}</div>
  <div class="charts">{fig_profile_metrics._repr_html_()}</div>"""
        except Exception as exc:
            typer.echo(f"Warning: mobility profiles skipped: {exc}", err=True)

    transport_spatial_section_html = ""
    transport_cfg = transport_spatial_config
    if bool(getattr(transport_cfg, "enabled", True)):
        synthetic_moving_path = getattr(transport_cfg, "synthetic_moving_path", None)
        moving_path = (
            Path(synthetic_moving_path)
            if synthetic_moving_path
            else _default_synthetic_moving_path(synthetic_path)
        )
        if moving_path is None:
            typer.echo("Warning: transport spatial mobility skipped: synthetic_path was not provided", err=True)
        elif not moving_path.exists():
            typer.echo(
                f"Warning: transport spatial mobility skipped: moving sidecar not found: {moving_path}",
                err=True,
            )
        else:
            try:
                mode_map = _transport_mode_map(transport_cfg)
                transport_records = _synthetic_transport_leg_records(moving_path, mode_map=mode_map)
                if transport_records.is_empty():
                    typer.echo("Warning: transport spatial mobility skipped: no synthetic transport legs", err=True)
                else:
                    if bool(getattr(transport_cfg, "observed_enabled", False)):
                        try:
                            observed_records = _observed_transport_leg_records(
                                real_df,
                                uid_col=getattr(transport_cfg, "uid_col", None),
                                datetime_col=getattr(transport_cfg, "datetime_col", None),
                                lat_col=getattr(transport_cfg, "lat_col", None),
                                lng_col=getattr(transport_cfg, "lng_col", None),
                                transport_col=getattr(transport_cfg, "transport_col", None),
                                duration_col=duration_col,
                                mode_map=mode_map,
                            )
                            if observed_records.is_empty():
                                typer.echo(
                                    "Warning: observed transport spatial comparison skipped: no observed transport legs",
                                    err=True,
                                )
                            else:
                                transport_records = pl.concat(
                                    [transport_records, observed_records],
                                    how="diagonal",
                                )
                        except Exception as exc:
                            typer.echo(
                                f"Warning: observed transport spatial comparison skipped: {exc}",
                                err=True,
                            )
                    transport_summary = _transport_spatial_summary(transport_records)
                    if transport_summary:
                        metrics["transport_spatial"] = transport_summary
                        transport_spatial_section_html = _transport_spatial_section_html(
                            transport_records,
                            observed_label=observed_label,
                        )
            except Exception as exc:
                typer.echo(f"Warning: transport spatial mobility skipped: {exc}", err=True)

    w_rows = [
        ("Jump lengths", f"{w_jump:.4f}", "km"),
        ("Visits per user", f"{w_visits:.4f}", "visits"),
        ("Radius of gyration", f"{w_rog:.4f}", "km"),
        ("Dwell time", f"{w_dwell:.4f}", "min"),
    ]
    if w_trip is not None:
        w_rows.append(("Trip duration (car)", f"{w_trip:.4f}", "min"))

    metrics_html = _metrics_section_html(w_rows, js_rows, cpc_rows)
    network_validation_section_html = _network_validation_section_html(network_validation)
    generated_at = _dt.now().strftime("%Y-%m-%d %H:%M")
    resource_bundle = get_resource_bundle(echarts=True, leaflet=True)
    full_html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>CityBehavEx Comparison Report</title>
  {resource_bundle}
  <style>
    html,body{{margin:0;padding:0;background:#fbf8f1;color:#14110d;font-family:sans-serif;}}
    .header{{padding:32px 32px 24px;border-bottom:1px solid #dcd5c4;}}
    .header h1{{margin:0;font-size:20px;font-weight:600;}}
    .header p{{margin:6px 0 0;font-size:13px;color:#6b5e4c;}}
    .section-header{{padding:20px 32px 0;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:#6b5e4c;}}
    .charts{{display:flex;flex-wrap:wrap;padding:24px;gap:24px;}}
    .charts iframe{{flex:1 1 580px;border:0;min-height:420px;}}
    .charts .skmob-vis-widget{{min-height:420px;}}
    .mobility-law-chart{{flex:1 1 580px;min-width:0;}}
    .mobility-law-chart iframe{{width:100%;}}
    .stvd-section .skmob-vis-widget{{min-height:600px;}}
    .fit-parameters{{margin:8px 12px 0;padding:12px 16px;border:1px solid #dcd5c4;background:#fffdf8;font-family:monospace;font-size:12px;}}
    .fit-formula{{margin-bottom:8px;color:#6b5e4c;}}
    .fit-parameters table{{border-collapse:collapse;}}
    .fit-parameters td{{padding:2px 18px 2px 0;}}
    .fit-parameters td:first-child{{font-weight:600;}}
    .report-warning{{margin:16px 32px 0;padding:12px 16px;border:1px solid #d5a33d;background:#fff7df;color:#4b3511;font-size:13px;}}
    .report-warning strong{{display:block;margin-bottom:6px;}}
    .report-warning ul{{margin:0;padding-left:18px;}}
    .metrics{{display:flex;flex-wrap:wrap;gap:64px;padding:24px 32px 32px;border-bottom:1px solid #dcd5c4;}}
    .metrics h2{{margin:0 0 14px;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:#6b5e4c;}}
    .metrics table{{border-collapse:collapse;font-family:monospace;font-size:13px;}}
    .metrics td{{padding:3px 20px 3px 0;}}
  </style>
</head>
<body>
  <div class="header">
    <h1>synthetic &nbsp;vs&nbsp; {observed_label}</h1>
    <p>Generated {generated_at}</p>
  </div>{metrics_html}
  <div class="section-header">Distribution comparisons</div>
  <div class="charts">{ecdf_charts_html}</div>{network_validation_section_html}{transport_spatial_section_html}{mobility_laws_section_html}{activity_section_html}{micro_activity_section_html}{profiles_section_html}{motif_section_html}{stvd_section_html}
</body>
</html>
"""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(full_html, encoding="utf-8")
    summary = "  ".join(f"{n}: {v}" for n, v, _ in w_rows)
    typer.echo(f"Comparison report -> {output_path}  ({summary})")

    if json_output_path:
        json_out = Path(json_output_path)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        typer.echo(f"Comparison metrics -> {json_output_path}")
