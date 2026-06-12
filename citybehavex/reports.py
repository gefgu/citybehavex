from __future__ import annotations

from datetime import datetime as _dt
from pathlib import Path
from typing import Optional

import h3
import pandas as pd
import skmob2
import typer
from skmob2 import (
    activity_distribution_jensen_shannon_divergence,
    activity_transition_matrix,
    activity_transition_matrix_jensen_shannon_divergence,
    daily_activity_distribution,
    discover_daily_motifs_from_agents,
    jensen_shannon_divergence,
    time_bin_matrix_jensen_shannon_divergence,
    visits_per_user_wasserstein_distance,
    waiting_times,
    wasserstein_distance,
)
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
    real_traj = skmob2.TrajDataFrame(
        real_df,
        datetime_col=detect_column(real_df, _DATETIME_CANDIDATES),
        lat_col=detect_column(real_df, _LAT_CANDIDATES),
        lng_col=detect_column(real_df, _LNG_CANDIDATES),
        uid_col=detect_column(real_df, _UID_CANDIDATES),
    )

    typer.echo("Computing mobility metrics ...")
    labels = ("synthetic", observed_label)
    synth_jumps = traj.jump_lengths(merge=True)
    real_jumps = real_traj.jump_lengths(merge=True)
    w_jump = wasserstein_distance(synth_jumps, real_jumps)

    synth_visits = traj.df[traj.uid_col].value_counts().to_list()
    real_visits = real_traj.df[real_traj.uid_col].value_counts().to_list()
    w_visits, _ = visits_per_user_wasserstein_distance(
        traj.df,
        real_df,
        user_id_col1=traj.uid_col,
        user_id_col2=real_traj.uid_col,
    )

    synth_rog = traj.radius_of_gyration()["radius_of_gyration"].to_numpy()
    real_rog = real_traj.radius_of_gyration()["radius_of_gyration"].to_numpy()
    w_rog = wasserstein_distance(synth_rog, real_rog)

    # Dwell time = time spent at a location. The synthetic trip-DITRAS records this
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

    activity_col = detect_column(real_df, _ACTIVITY_CANDIDATES)
    activity_section_html = ""
    if activity_col:
        typer.echo(f"Rendering activity charts for {observed_label} ...")
        activity_charts_html = (
            plot_visit_purpose_distribution(real_df)._repr_html_()
            + plot_activity_transition_matrix(real_df)._repr_html_()
            + plot_daily_activity_distribution(real_df)._repr_html_()
        )
        activity_section_html = f"""
  <div class="section-header">
    <span>Activity profile &mdash; {observed_label}</span>
  </div>
  <div class="charts">{activity_charts_html}</div>"""

    if synth_activity_col and synth_activity_col in traj.df.columns:
        typer.echo("Rendering activity charts for synthetic trajectories ...")
        synth_activity_charts_html = (
            plot_visit_purpose_distribution(traj.df)._repr_html_()
            + plot_activity_transition_matrix(traj.df)._repr_html_()
            + plot_daily_activity_distribution(traj.df)._repr_html_()
        )
        activity_section_html += f"""
  <div class="section-header">
    <span>Activity profile &mdash; synthetic</span>
  </div>
  <div class="charts">{synth_activity_charts_html}</div>"""

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
            )
            motif_section_html = f"""
  <div class="section-header">
    <span>Motif comparison &mdash; literature vs {observed_label}{" vs synthetic" if synth_motif_dist is not None else ""}</span>
  </div>
  <div class="charts">{fig_motif._repr_html_()}</div>"""
    except Exception as exc:
        typer.echo(f"Warning: motif chart skipped: {exc}", err=True)

    w_rows = [
        ("Jump lengths", f"{w_jump:.4f}", "km"),
        ("Visits per user", f"{w_visits:.4f}", "visits"),
        ("Radius of gyration", f"{w_rog:.4f}", "km"),
        ("Dwell time", f"{w_dwell:.4f}", "min"),
    ]
    if w_trip is not None:
        w_rows.append(("Trip duration (car)", f"{w_trip:.4f}", "min"))

    wasserstein_rows = "".join(
        f"<tr><td>{name}</td><td>{val}</td><td>{unit}</td></tr>"
        for name, val, unit in w_rows
    )
    jsd_rows = "".join(
        f"<tr><td>{name}</td><td>{val}</td><td>{unit}</td></tr>"
        for name, val, unit in js_rows
    )
    metrics_html = f"""
  <div class="metrics">
    <div>
      <h2>Wasserstein distances</h2>
      <table>{wasserstein_rows}</table>
    </div>
    <div>
      <h2>Jensen-Shannon divergences</h2>
      <table>{jsd_rows}</table>
    </div>
  </div>"""
    generated_at = _dt.now().strftime("%Y-%m-%d %H:%M")
    full_html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>CityBehavEx Comparison Report</title>
  <style>
    html,body{{margin:0;padding:0;background:#fbf8f1;color:#14110d;font-family:sans-serif;}}
    .header{{padding:32px 32px 24px;border-bottom:1px solid #dcd5c4;}}
    .header h1{{margin:0;font-size:20px;font-weight:600;}}
    .header p{{margin:6px 0 0;font-size:13px;color:#6b5e4c;}}
    .section-header{{padding:20px 32px 0;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:#6b5e4c;}}
    .charts{{display:flex;flex-wrap:wrap;padding:24px;gap:24px;}}
    .charts iframe{{flex:1 1 580px;border:0;min-height:420px;}}
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
  <div class="charts">{ecdf_charts_html}</div>{activity_section_html}{motif_section_html}
</body>
</html>
"""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(full_html, encoding="utf-8")
    summary = "  ".join(f"{n}: {v}" for n, v, _ in w_rows)
    typer.echo(f"Comparison report -> {output_path}  ({summary})")
