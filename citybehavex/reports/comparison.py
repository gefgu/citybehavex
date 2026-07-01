from __future__ import annotations

from datetime import datetime as _dt
from pathlib import Path
from typing import Optional

import h3
import numpy as np
import pandas as pd
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
    od_matrix_common_part_of_commuters,
    time_bin_matrix_jensen_shannon_divergence,
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

from citybehavex.profiles import PROFILE_METRICS, compute_profiles

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

# Speed used to turn real jump lengths into a car travel-time proxy for the trip
# duration comparison. Matches the synthetic SimulationConfig.car_speed_kmh default.
CAR_SPEED_KMH = 50.0
CPC_H3_RESOLUTIONS = (7, 8, 9)


def detect_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
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


def _trajectory_od_matrix(
    df: pd.DataFrame,
    *,
    uid_col: str,
    datetime_col: str,
    lat_col: str,
    lng_col: str,
    resolution: int,
) -> pd.DataFrame:
    points = df[[uid_col, datetime_col, lat_col, lng_col]].copy()
    points["_datetime"] = pd.to_datetime(points[datetime_col], errors="coerce")
    points["_lat"] = pd.to_numeric(points[lat_col], errors="coerce")
    points["_lng"] = pd.to_numeric(points[lng_col], errors="coerce")
    points = points.dropna(subset=[uid_col, "_datetime", "_lat", "_lng"])
    points = points[
        points["_lat"].between(-90, 90)
        & points["_lng"].between(-180, 180)
    ]
    points = points.sort_values([uid_col, "_datetime"], kind="mergesort")
    points["origin"] = [
        h3.latlng_to_cell(lat, lng, resolution)
        for lat, lng in zip(points["_lat"], points["_lng"])
    ]
    points["destination"] = points.groupby(uid_col)["origin"].shift(-1)
    trips = points.dropna(subset=["destination"])
    trips = trips[trips["origin"] != trips["destination"]]

    if trips.empty:
        return pd.DataFrame(dtype=float)

    flows = (
        trips.groupby(["origin", "destination"])
        .size()
        .unstack(fill_value=0)
        .astype(float)
    )
    return flows


def _common_part_of_commuters(
    traj: skmob2.TrajDataFrame,
    real_traj: skmob2.TrajDataFrame,
    resolutions: tuple[int, ...] = CPC_H3_RESOLUTIONS,
) -> list[tuple[int, float]]:
    values = []
    for resolution in resolutions:
        synthetic_od = _trajectory_od_matrix(
            traj.df,
            uid_col=traj.uid_col,
            datetime_col=traj.datetime_col,
            lat_col=traj.lat_col,
            lng_col=traj.lng_col,
            resolution=resolution,
        )
        observed_od = _trajectory_od_matrix(
            real_traj.df,
            uid_col=real_traj.uid_col,
            datetime_col=real_traj.datetime_col,
            lat_col=real_traj.lat_col,
            lng_col=real_traj.lng_col,
            resolution=resolution,
        )
        values.append(
            (
                resolution,
                od_matrix_common_part_of_commuters(synthetic_od, observed_od),
            )
        )
    return values


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


def _visits_for_comparison(
    df: pd.DataFrame,
    *,
    uid_col: str,
    datetime_col: str,
    activity_col: str,
    location_col: Optional[str] = None,
    location_resolution: int = 10,
    end_col: Optional[str] = None,
) -> pd.DataFrame:
    visits = pd.DataFrame(
        {
            "uid": df[uid_col],
            "start_timestamp": pd.to_datetime(df[datetime_col]),
            "purpose": df[activity_col],
        }
    )
    if location_col:
        visits["location_id"] = df[location_col].astype(str)
    else:
        visits["location_id"] = [
            h3.latlng_to_cell(lat, lng, location_resolution)
            for lat, lng in zip(df["lat"], df["lng"])
        ]

    if end_col:
        visits["end_timestamp"] = pd.to_datetime(df[end_col])
    else:
        visits = visits.sort_values(["uid", "start_timestamp"]).reset_index(drop=True)
        visits["end_timestamp"] = visits.groupby("uid")["start_timestamp"].shift(-1)
        visits["end_timestamp"] = visits["end_timestamp"].fillna(
            visits["start_timestamp"].dt.normalize() + pd.Timedelta(days=1)
        )
    return visits


def _collapse_to_stays(
    df: pd.DataFrame,
    *,
    uid_col: str,
    lat_col: str,
    lng_col: str,
    datetime_col: str,
) -> pd.DataFrame:
    """Collapse a slot-by-slot trajectory into one row per stay episode.

    The synthetic trajectory emits a record per time slot, so consecutive slots
    at the same location are the same visit. Keeping only the first row of each
    maximal same-location run per user makes "visits per user" count distinct
    stays, comparable to the observed stay-event table instead of slot density.
    """
    ordered = df.sort_values([uid_col, datetime_col])
    same_user = ordered[uid_col].eq(ordered[uid_col].shift())
    same_loc = ordered[lat_col].eq(ordered[lat_col].shift()) & ordered[lng_col].eq(
        ordered[lng_col].shift()
    )
    new_stay = ~(same_user & same_loc)
    return ordered[new_stay].reset_index(drop=True)


def _motif_visits(visits: pd.DataFrame) -> pd.DataFrame:
    motif_visits = visits.copy()
    motif_visits["purpose"] = motif_visits["purpose"].where(
        motif_visits["purpose"].eq("HOME"),
        "VISIT",
    )
    return motif_visits


def _location_resolution(
    df: pd.DataFrame,
    location_col: Optional[str],
    default: int = 10,
) -> int:
    if location_col:
        for value in df[location_col].dropna().astype(str):
            try:
                return h3.get_resolution(value)
            except ValueError:
                break
    return default


def _compute_stvd_layers(
    traj: skmob2.TrajDataFrame,
    real_traj: skmob2.TrajDataFrame,
    resolutions: list[int],
) -> dict[int, dict]:
    """Compute per-H3-zone volume diff and peak shift for the STVD visualisation."""
    syn = traj.df[[traj.uid_col, traj.lat_col, traj.lng_col, traj.datetime_col]].copy()
    real = real_traj.df[[real_traj.uid_col, real_traj.lat_col, real_traj.lng_col, real_traj.datetime_col]].copy()

    syn["_dt"] = pd.to_datetime(syn[traj.datetime_col], errors="coerce")
    real["_dt"] = pd.to_datetime(real[real_traj.datetime_col], errors="coerce")
    syn = syn.dropna(subset=["_dt", traj.lat_col, traj.lng_col])
    real = real.dropna(subset=["_dt", real_traj.lat_col, real_traj.lng_col])
    syn["_hour"] = syn["_dt"].dt.hour
    real["_hour"] = real["_dt"].dt.hour

    layers: dict[int, dict] = {}
    for res in resolutions:
        syn["_cell"] = [
            h3.latlng_to_cell(lat, lng, res)
            for lat, lng in zip(syn[traj.lat_col], syn[traj.lng_col])
        ]
        real["_cell"] = [
            h3.latlng_to_cell(lat, lng, res)
            for lat, lng in zip(real[real_traj.lat_col], real[real_traj.lng_col])
        ]

        all_hours = list(range(24))
        syn_hourly = (
            syn.groupby(["_cell", "_hour"]).size().unstack(fill_value=0).reindex(columns=all_hours, fill_value=0)
        )
        real_hourly = (
            real.groupby(["_cell", "_hour"]).size().unstack(fill_value=0).reindex(columns=all_hours, fill_value=0)
        )

        all_cells = set(syn_hourly.index) | set(real_hourly.index)
        features = []
        for cell in all_cells:
            syn_row = syn_hourly.loc[cell] if cell in syn_hourly.index else pd.Series(0, index=all_hours)
            real_row = real_hourly.loc[cell] if cell in real_hourly.index else pd.Series(0, index=all_hours)

            syn_vol = float(syn_row.sum())
            real_vol = float(real_row.sum())
            syn_peak = int(syn_row.idxmax()) if syn_vol > 0 else 0
            real_peak = int(real_row.idxmax()) if real_vol > 0 else 0

            volume_diff_pct = (syn_vol - real_vol) / max(real_vol, 1.0) * 100.0
            raw_shift = abs(syn_peak - real_peak)
            peak_shift_hours = float(min(raw_shift, 12 - raw_shift if raw_shift <= 12 else raw_shift))
            peak_shift_hours = min(peak_shift_hours, 12.0)

            boundary = h3.cell_to_boundary(cell)
            ring = [[lng, lat] for lat, lng in boundary]
            ring.append(ring[0])

            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {
                    "area": cell,
                    "volume_diff_pct": round(volume_diff_pct, 4),
                    "peak_shift_hours": round(peak_shift_hours, 4),
                },
            })

        layers[res] = {"type": "FeatureCollection", "features": features}

    return layers


def _motif_distribution_jsd(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> float:
    left_counts = dict(zip(left["motif_id"], left["count"]))
    right_counts = dict(zip(right["motif_id"], right["count"]))
    labels = sorted(set(left_counts) | set(right_counts), key=str)
    return jensen_shannon_divergence(
        [left_counts.get(label, 0) for label in labels],
        [right_counts.get(label, 0) for label in labels],
    )


def _activity_comparison_section_html(
    observed_visits: Optional[pd.DataFrame],
    synthetic_visits: Optional[pd.DataFrame],
    observed_label: str,
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
    return f"""
  <div class="section-header">
    <span>Activity comparison &mdash; {observed_label} vs synthetic</span>
  </div>
  <div class="charts">{charts_html}</div>"""


def _mobility_law_visits(
    df: pd.DataFrame,
    *,
    uid_col: str,
    datetime_col: str,
    lat_col: str,
    lng_col: str,
    location_col: Optional[str] = None,
    activity_col: Optional[str] = None,
    location_resolution: int = 10,
) -> pd.DataFrame:
    columns = [uid_col, datetime_col, lat_col, lng_col]
    if location_col:
        columns.append(location_col)
    if activity_col:
        columns.append(activity_col)

    source = df[columns].copy()
    source[datetime_col] = pd.to_datetime(source[datetime_col], errors="coerce")
    source[lat_col] = pd.to_numeric(source[lat_col], errors="coerce")
    source[lng_col] = pd.to_numeric(source[lng_col], errors="coerce")
    source = source.dropna(subset=[uid_col, datetime_col, lat_col, lng_col])
    source = source[
        source[lat_col].between(-90, 90) & source[lng_col].between(-180, 180)
    ]

    visits = pd.DataFrame(
        {
            "user_id": source[uid_col],
            "timestamp": source[datetime_col],
            "lat": source[lat_col],
            "lng": source[lng_col],
        }
    )
    fallback_locations = pd.Series(
        [
            h3.latlng_to_cell(lat, lng, location_resolution)
            for lat, lng in zip(visits["lat"], visits["lng"])
        ],
        index=visits.index,
    )
    if location_col:
        visits["location_id"] = source[location_col].where(
            source[location_col].notna(),
            fallback_locations,
        ).astype(str)
    else:
        visits["location_id"] = fallback_locations
    if activity_col:
        visits["purpose"] = source[activity_col].to_numpy()
    return visits.reset_index(drop=True)


def _daily_location_lognormal_dataset(
    visits: pd.DataFrame,
    label: str,
) -> tuple[np.ndarray, np.ndarray, float, float, str]:
    daily = (
        visits.assign(date=visits["timestamp"].dt.normalize())
        .groupby(["user_id", "date"])["location_id"]
        .nunique()
    )
    values = daily.to_numpy(dtype=float)
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
    visits: pd.DataFrame,
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
    observed_visits: pd.DataFrame,
    synthetic_visits: pd.DataFrame,
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
    df = pd.read_parquet(path)
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
) -> None:
    typer.echo(f"Loading synthetic trajectories from {synthetic_path} ...")
    traj = load_trajectory(synthetic_path)
    synth_activity_col = detect_column(traj.df, _ACTIVITY_CANDIDATES)
    generate_comparison_report(
        traj=traj,
        real_path=real_path,
        observed_label=observed_label,
        output_path=output_path,
        synth_activity_col=synth_activity_col,
    )


def generate_comparison_report(
    traj: skmob2.TrajDataFrame,
    real_path: str,
    observed_label: str,
    output_path: str,
    synth_activity_col: Optional[str] = None,
) -> None:
    typer.echo(f"Loading observed trajectories from {real_path} ...")
    real_df = pd.read_parquet(real_path)
    _dt_col = detect_column(real_df, _DATETIME_CANDIDATES)
    if _dt_col and not pd.api.types.is_datetime64_any_dtype(real_df[_dt_col]):
        real_df[_dt_col] = pd.to_datetime(real_df[_dt_col])
    real_traj = skmob2.TrajDataFrame(
        real_df,
        datetime_col=_dt_col,
        lat_col=detect_column(real_df, _LAT_CANDIDATES),
        lng_col=detect_column(real_df, _LNG_CANDIDATES),
        uid_col=detect_column(real_df, _UID_CANDIDATES),
    )

    typer.echo("Computing mobility metrics ...")
    labels = ("synthetic", observed_label)
    synth_jumps = traj.jump_lengths(merge=True)
    real_jumps = real_traj.jump_lengths(merge=True)
    w_jump = wasserstein_distance(synth_jumps, real_jumps)

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
    synth_visits = synth_stays[traj.uid_col].value_counts().to_list()
    real_visits = real_traj.df[real_traj.uid_col].value_counts().to_list()
    w_visits, _ = visits_per_user_wasserstein_distance(
        synth_stays,
        real_df,
        user_id_col1=traj.uid_col,
        user_id_col2=real_traj.uid_col,
    )

    synth_rog = traj.radius_of_gyration()["radius_of_gyration"].to_numpy()
    real_rog = real_traj.radius_of_gyration()["radius_of_gyration"].to_numpy()
    w_rog = wasserstein_distance(synth_rog, real_rog)

    typer.echo("Computing Common Part of Commuters ...")
    cpc_rows = _common_part_of_commuters(traj, real_traj)

    # Dwell time = time spent at a location. The synthetic simulation records this
    # directly as departure - arrival (`dwell_minutes`); otherwise fall back to
    # inter-event gaps. The observed side uses the real stay-duration column when
    # present (NOT inter-event gaps, which on a sparse visit table span days).
    duration_col = detect_column(real_df, _DURATION_CANDIDATES)
    if "dwell_minutes" in traj.df.columns:
        synth_dwell = [d for d in traj.df["dwell_minutes"].dropna().tolist() if d >= 0]
    else:
        synth_dwell = waiting_times_minutes(traj)
    if duration_col:
        real_dwell = real_df[duration_col].dropna().tolist()
    else:
        real_dwell = waiting_times_minutes(real_traj)
    w_dwell = wasserstein_distance(synth_dwell, real_dwell)

    # Trip (travel) duration. The synthetic side carries a genuine car trip
    # duration per leg; the observed visit table has no travel-time ground truth,
    # so the real comparator is a car-time proxy from real jump lengths at the same
    # speed (km / CAR_SPEED_KMH * 60), making both sides directly comparable.
    if "trip_duration_minutes" in traj.df.columns:
        synth_trip = [t for t in traj.df["trip_duration_minutes"].dropna().tolist() if t > 0]
        real_trip = [(j / CAR_SPEED_KMH) * 60.0 for j in real_jumps if j > 0]
        w_trip = wasserstein_distance(synth_trip, real_trip) if synth_trip and real_trip else None
    elif duration_col:
        real_trip = real_df[duration_col].dropna().tolist()
        synth_trip = waiting_times_minutes(traj)
        w_trip = wasserstein_distance(synth_trip, real_trip)
    else:
        real_trip = synth_trip = w_trip = None

    js_rows: list[tuple[str, str, str]] = []
    synthetic_visits = None
    observed_visits = None
    if synth_activity_col and synth_activity_col in traj.df.columns:
        real_activity_col = detect_column(real_df, _ACTIVITY_CANDIDATES)
        real_start_col = detect_column(real_df, _DATETIME_CANDIDATES)
        real_end_col = detect_column(real_df, _END_TS_CANDIDATES)
        real_location_col = detect_column(real_df, _LOCATION_CANDIDATES)
        if real_activity_col and real_start_col:
            location_resolution = _location_resolution(real_df, real_location_col)
            synthetic_visits = _visits_for_comparison(
                traj.df,
                uid_col=traj.uid_col,
                datetime_col=traj.datetime_col,
                activity_col=synth_activity_col,
                location_resolution=location_resolution,
            )
            observed_visits = _visits_for_comparison(
                real_df,
                uid_col=real_traj.uid_col,
                datetime_col=real_start_col,
                activity_col=real_activity_col,
                location_col=real_location_col,
                end_col=real_end_col,
            )
            js_rows.append(
                (
                    "Activity distribution",
                    f"{activity_distribution_jensen_shannon_divergence(synthetic_visits, observed_visits):.4f}",
                    "",
                )
            )
            synth_transition = activity_transition_matrix(synthetic_visits)
            real_transition = activity_transition_matrix(observed_visits)
            js_rows.append(
                (
                    "Activity transitions",
                    f"{activity_transition_matrix_jensen_shannon_divergence(synth_transition, real_transition):.4f}",
                    "",
                )
            )
            synth_daily, synth_categories, _ = daily_activity_distribution(
                synthetic_visits
            )
            real_daily, real_categories, _ = daily_activity_distribution(
                observed_visits
            )
            js_rows.append(
                (
                    "Daily activity profile",
                    f"{time_bin_matrix_jensen_shannon_divergence(synth_daily, real_daily, synth_categories, real_categories):.4f}",
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

    typer.echo("Rendering mobility-law charts ...")
    real_location_col = detect_column(real_df, _LOCATION_CANDIDATES)
    synth_location_col = detect_column(traj.df, _LOCATION_CANDIDATES)
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

    if observed_visits is not None and synthetic_visits is not None:
        typer.echo(
            f"Rendering activity comparison for {observed_label} and synthetic trajectories ..."
        )
    activity_section_html = _activity_comparison_section_html(
        observed_visits,
        synthetic_visits,
        observed_label,
    )

    motif_section_html = ""
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
                js_rows.append(
                    (
                        "Daily motifs",
                        f"{_motif_distribution_jsd(synth_motif_dist, real_motif_dist):.4f}",
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
    if traj.lat_col and traj.lng_col and real_traj.lat_col and real_traj.lng_col:
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
    if observed_visits is not None and synthetic_visits is not None:
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

    w_rows = [
        ("Jump lengths", f"{w_jump:.4f}", "km"),
        ("Visits per user", f"{w_visits:.4f}", "visits"),
        ("Radius of gyration", f"{w_rog:.4f}", "km"),
        ("Dwell time", f"{w_dwell:.4f}", "min"),
    ]
    if w_trip is not None:
        w_rows.append(("Trip duration (car)", f"{w_trip:.4f}", "min"))

    metrics_html = _metrics_section_html(w_rows, js_rows, cpc_rows)
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
  <div class="charts">{ecdf_charts_html}</div>{mobility_laws_section_html}{activity_section_html}{profiles_section_html}{motif_section_html}{stvd_section_html}
</body>
</html>
"""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(full_html, encoding="utf-8")
    summary = "  ".join(f"{n}: {v}" for n, v, _ in w_rows)
    typer.echo(f"Comparison report -> {output_path}  ({summary})")
