"""Build the comparison-report payload as raw JSON plot data.

This mirrors the compute pipeline of
``citybehavex.reports.comparison.generate_comparison_report`` but, instead of
feeding the results into ``skmob_vis`` widgets and writing an HTML file, it emits
plain arrays / matrices / GeoJSON. The React frontend turns those into themed
ECharts options and a Leaflet map.

Numeric work that would otherwise be duplicated in the frontend (ECDF curves,
mobility-law fit and reference curves, profile box statistics, motif mapping,
STVD bivariate classification) is done here so the frontend only has to plot
arrays.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import skmob2
from skmob2 import (
    activity_distribution_jensen_shannon_divergence,
    activity_transition_matrix,
    activity_transition_matrix_jensen_shannon_divergence,
    daily_activity_distribution,
    discover_daily_motifs_from_agents,
    jensen_shannon_divergence,
    time_bin_matrix_jensen_shannon_divergence,
    visits_per_user_wasserstein_distance,
    wasserstein_distance,
)
from skmob_vis._core import compute_ecdf
from skmob_vis.motifs import (
    _literature_distribution_rows,
    _motif_axis_label_styles,
    map_motif_distribution_to_literature_basis,
)

from .reports_bridge import (
    CAR_SPEED_KMH,
    PROFILE_METRICS,
    _ACTIVITY_CANDIDATES,
    _DATETIME_CANDIDATES,
    _DURATION_CANDIDATES,
    _END_TS_CANDIDATES,
    _LOCATION_CANDIDATES,
    _collapse_to_stays,
    _common_part_of_commuters,
    _compute_stvd_layers,
    _daily_location_lognormal_dataset,
    _distance_frequency_dataset,
    _location_resolution,
    _micro_activity_daily_usage_data,
    _mobility_law_visits,
    _motif_visits,
    _prepare_activity_visits,
    _truncated_powerlaw_dataset,
    compute_profiles,
    detect_column,
    load_trajectory,
    waiting_times_minutes,
)
from citybehavex.simulation.core import social_network_sidecar_path

# STVD bivariate palette (volume-diff bin x peak-shift bin), matching
# skmob_vis.stvd.STVD_COLORS so the map reads identically to the HTML report.
STVD_COLORS = [
    ["#91bfdb", "#f7f7f7", "#f4a582"],
    ["#4393c3", "#bdbdbd", "#d6604d"],
    ["#2166ac", "#6e6e6e", "#b2182b"],
]
STVD_VOLUME_THRESHOLD = 3.0

_MAX_ECDF_POINTS = 400
_MAX_SCATTER_POINTS = 4000


# --------------------------------------------------------------------------- #
# small numeric helpers
# --------------------------------------------------------------------------- #
def _downsample(points: list[list[float]], max_points: int) -> list[list[float]]:
    n = len(points)
    if n <= max_points:
        return points
    idx = np.linspace(0, n - 1, max_points).round().astype(int)
    idx = np.unique(idx)
    return [points[i] for i in idx]


def _ecdf(values: Any, cutoff: float = 0.98) -> list[list[float]]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return []
    points = compute_ecdf(np.ascontiguousarray(arr), cutoff)
    return _downsample([[float(x), float(y)] for x, y in points], _MAX_ECDF_POINTS)


def _ecdf_block(
    label_syn: str,
    syn_values: Any,
    label_obs: str,
    obs_values: Any,
    x_label: str,
    x_unit: str,
) -> dict[str, Any]:
    return {
        "x_label": x_label,
        "x_unit": x_unit,
        "series": [
            {"name": label_syn, "role": "synthetic", "points": _ecdf(syn_values)},
            {"name": label_obs, "role": "observed", "points": _ecdf(obs_values)},
        ],
    }


def _geometric_scale(y: np.ndarray, shape: np.ndarray) -> float:
    valid = np.isfinite(shape) & (shape > 0) & np.isfinite(y) & (y > 0)
    if not valid.any():
        return 1.0
    return float(np.exp(np.mean(np.log(y[valid]) - np.log(shape[valid]))))


def _curve_x(all_x: list[np.ndarray], *, logarithmic: bool, n: int = 200) -> np.ndarray:
    x_min = min(float(x.min()) for x in all_x)
    x_max = max(float(x.max()) for x in all_x)
    if x_min == x_max:
        return np.asarray([x_min], dtype=float)
    if logarithmic:
        x_min = max(x_min, 1e-9)
        return np.logspace(np.log10(x_min), np.log10(x_max), n)
    return np.linspace(x_min, x_max, n)


def _xy(x: np.ndarray, y: np.ndarray) -> list[list[float]]:
    return [[float(a), float(b)] for a, b in zip(x, y)]


# --------------------------------------------------------------------------- #
# mobility laws
# --------------------------------------------------------------------------- #
def _truncated_powerlaw_series(
    observed_values: Any,
    synthetic_values: Any,
    label_obs: str,
    reference: tuple[float, float, float] = (1.5, 1.75, 400.0),
) -> dict[str, Any]:
    obs = _truncated_powerlaw_dataset(observed_values, label_obs)   # (params, x, y, label)
    syn = _truncated_powerlaw_dataset(synthetic_values, "synthetic")
    all_x = [np.asarray(obs[1], float), np.asarray(syn[1], float)]
    curve_x = _curve_x(all_x, logarithmic=True)

    series = []
    fits = []
    for params, x_pts, y_pts, label, role in (
        (*obs, "observed"),
        (*syn, "synthetic"),
    ):
        c, r0, beta, kappa = (float(v) for v in params)
        curve_y = c * np.power(curve_x + r0, -beta) * np.exp(-curve_x / kappa)
        series.append({"name": label, "role": role, "type": "scatter",
                       "points": _xy(np.asarray(x_pts, float), np.asarray(y_pts, float))})
        series.append({"name": f"{label} fit", "role": role, "type": "line",
                       "points": _xy(curve_x, curve_y)})
        fits.append({"label": label, "params": {"c": c, "r0": r0, "beta": beta, "kappa": kappa}})

    r0, beta, kappa = reference
    joined_x = np.concatenate(all_x)
    joined_y = np.concatenate([np.asarray(obs[2], float), np.asarray(syn[2], float)])
    shape = np.power(joined_x + r0, -beta) * np.exp(-joined_x / kappa)
    c = _geometric_scale(joined_y, shape)
    series.append({"name": "Gonzalez reference", "role": "reference", "type": "line",
                   "points": _xy(curve_x, c * np.power(curve_x + r0, -beta) * np.exp(-curve_x / kappa))})

    return {
        "x_log": True,
        "formula": "p(x) = c (x + r0)^-beta exp(-x / kappa)",
        "series": series,
        "fits": fits,
    }


def _lognormal_series(observed_visits, synthetic_visits, label_obs) -> dict[str, Any]:
    obs = _daily_location_lognormal_dataset(observed_visits, label_obs)   # (x, y, mu, sigma, label)
    syn = _daily_location_lognormal_dataset(synthetic_visits, "synthetic")
    all_x = [np.asarray(obs[0], float), np.asarray(syn[0], float)]
    curve_x = _curve_x(all_x, logarithmic=False)

    series, fits = [], []
    for x_pts, y_pts, mu, sigma, label, role in ((*obs, "observed"), (*syn, "synthetic")):
        curve_y = np.exp(-((np.log(curve_x) - mu) ** 2) / (2 * sigma**2)) / (
            curve_x * sigma * np.sqrt(2 * np.pi)
        )
        series.append({"name": label, "role": role, "type": "scatter",
                       "points": _xy(np.asarray(x_pts, float), np.asarray(y_pts, float))})
        series.append({"name": f"{label} fit", "role": role, "type": "line",
                       "points": _xy(curve_x, curve_y)})
        fits.append({"label": label, "params": {"mu": float(mu), "sigma": float(sigma)}})

    mu, sigma = 1.0, 0.5
    ref_y = np.exp(-((np.log(curve_x) - mu) ** 2) / (2 * sigma**2)) / (
        curve_x * sigma * np.sqrt(2 * np.pi)
    )
    series.append({"name": "Log-normal reference", "role": "reference", "type": "line",
                   "points": _xy(curve_x, ref_y)})
    return {
        "x_log": False,
        "formula": "f(N) = exp(-(ln N - mu)^2 / (2 sigma^2)) / (N sigma sqrt(2 pi))",
        "series": series,
        "fits": fits,
    }


def _distance_frequency_series(observed_visits, synthetic_visits, label_obs) -> dict[str, Any]:
    obs = _distance_frequency_dataset(observed_visits, label_obs)   # (rf, rho, eta, mu, label)
    syn = _distance_frequency_dataset(synthetic_visits, "synthetic")
    all_x = [np.asarray(obs[0], float), np.asarray(syn[0], float)]
    curve_x = _curve_x(all_x, logarithmic=True)

    series, fits = [], []
    for rf, rho, eta, mu, label, role in ((*obs, "observed"), (*syn, "synthetic")):
        series.append({"name": label, "role": role, "type": "scatter",
                       "points": _xy(np.asarray(rf, float), np.asarray(rho, float))})
        series.append({"name": f"{label} fit", "role": role, "type": "line",
                       "points": _xy(curve_x, mu * np.power(curve_x, -eta))})
        fits.append({"label": label, "params": {"eta": float(eta), "mu": float(mu)}})

    alpha = -2.0
    joined_x = np.concatenate(all_x)
    joined_y = np.concatenate([np.asarray(obs[1], float), np.asarray(syn[1], float)])
    scale = _geometric_scale(joined_y, np.power(joined_x, alpha))
    series.append({"name": "Schlapfer reference", "role": "reference", "type": "line",
                   "points": _xy(curve_x, scale * np.power(curve_x, alpha))})
    return {
        "x_log": True,
        "formula": "rho(r, f) = mu (r f)^-eta",
        "series": series,
        "fits": fits,
    }


# --------------------------------------------------------------------------- #
# STVD
# --------------------------------------------------------------------------- #
def _classify(volume_diff: float, peak_shift: float, threshold: float) -> tuple[int, int]:
    if volume_diff < -threshold:
        x_bin = 0
    elif volume_diff <= threshold:
        x_bin = 1
    else:
        x_bin = 2
    if peak_shift <= 2:
        y_bin = 0
    elif peak_shift <= 5:
        y_bin = 1
    else:
        y_bin = 2
    return x_bin, y_bin


def _annotate_stvd(layers: dict[int, dict]) -> dict[str, Any]:
    out_layers: dict[str, Any] = {}
    lngs: list[float] = []
    lats: list[float] = []
    for res, fc in layers.items():
        for feature in fc.get("features", []):
            props = feature.get("properties", {})
            x_bin, y_bin = _classify(
                float(props.get("volume_diff_pct", 0.0)),
                float(props.get("peak_shift_hours", 0.0)),
                STVD_VOLUME_THRESHOLD,
            )
            props["color"] = STVD_COLORS[y_bin][x_bin]
            props["class"] = y_bin * 3 + x_bin
            ring = feature.get("geometry", {}).get("coordinates", [[]])[0]
            for lng, lat in ring:
                lngs.append(lng)
                lats.append(lat)
        out_layers[str(res)] = fc

    center = None
    if lngs and lats:
        center = [(min(lngs) + max(lngs)) / 2, (min(lats) + max(lats)) / 2]
    return {"center": center, "layers": out_layers, "colors": STVD_COLORS,
            "threshold": STVD_VOLUME_THRESHOLD}


def _load_social_network_sidecar(synthetic_path: str) -> dict[str, Any] | None:
    path = social_network_sidecar_path(synthetic_path)
    if not path.exists():
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    nodes = data.get("nodes")
    edges = data.get("edges")
    degrees = data.get("degrees")
    node_count = int(data.get("node_count", -1))
    edge_count = int(data.get("edge_count", -1))
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError(f"invalid social network sidecar arrays: {path}")
    if node_count != len(nodes) or edge_count != len(edges):
        raise ValueError(f"social network sidecar count mismatch: {path}")
    if degrees is not None and (not isinstance(degrees, list) or len(degrees) != len(nodes)):
        raise ValueError(f"social network sidecar degree count mismatch: {path}")
    for row in nodes[:10]:
        if not isinstance(row, list) or len(row) < 4:
            raise ValueError(f"invalid social network node row: {path}")
    for row in edges[:10]:
        if not isinstance(row, list) or len(row) < 2:
            raise ValueError(f"invalid social network edge row: {path}")
    return data


# --------------------------------------------------------------------------- #
# main entry
# --------------------------------------------------------------------------- #
def build_comparison_payload(
    synthetic_path: str,
    observed_path: str,
    observed_label: str,
    synthetic_activities_path: Optional[str] = None,
) -> dict[str, Any]:
    warnings: list[str] = []

    def guard(section: str, fn):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - degrade gracefully per section
            warnings.append(f"{section}: {exc}")
            return None

    traj = load_trajectory(synthetic_path)
    synth_activity_col = detect_column(traj.df, _ACTIVITY_CANDIDATES)

    real_df = pd.read_parquet(observed_path)
    real_dt_col = detect_column(real_df, _DATETIME_CANDIDATES)
    if real_dt_col and not pd.api.types.is_datetime64_any_dtype(real_df[real_dt_col]):
        real_df[real_dt_col] = pd.to_datetime(real_df[real_dt_col])
    real_traj = skmob2.TrajDataFrame(
        real_df,
        datetime_col=real_dt_col,
        lat_col=detect_column(real_df, ["lat", "latitude"]),
        lng_col=detect_column(real_df, ["lng", "lon", "longitude", "long"]),
        uid_col=detect_column(real_df, ["uid", "user_id", "user", "agent_id", "userid"]),
    )

    labels = {"synthetic": "synthetic", "observed": observed_label}

    # ---- base arrays / metrics ------------------------------------------- #
    synth_jumps = traj.jump_lengths(merge=True)
    real_jumps = real_traj.jump_lengths(merge=True)
    w_jump = wasserstein_distance(synth_jumps, real_jumps)

    synth_stays = _collapse_to_stays(
        traj.df, uid_col=traj.uid_col, lat_col=traj.lat_col,
        lng_col=traj.lng_col, datetime_col=traj.datetime_col,
    )
    synth_visits_count = synth_stays[traj.uid_col].value_counts().to_list()
    real_visits_count = real_traj.df[real_traj.uid_col].value_counts().to_list()
    w_visits, _ = visits_per_user_wasserstein_distance(
        synth_stays, real_df,
        user_id_col1=traj.uid_col, user_id_col2=real_traj.uid_col,
    )

    synth_rog = traj.radius_of_gyration()["radius_of_gyration"].to_numpy()
    real_rog = real_traj.radius_of_gyration()["radius_of_gyration"].to_numpy()
    w_rog = wasserstein_distance(synth_rog, real_rog)

    cpc_rows = guard("cpc", lambda: _common_part_of_commuters(traj, real_traj)) or []

    duration_col = detect_column(real_df, _DURATION_CANDIDATES)
    if "dwell_minutes" in traj.df.columns:
        synth_dwell = [d for d in traj.df["dwell_minutes"].dropna().tolist() if d >= 0]
    else:
        synth_dwell = waiting_times_minutes(traj)
    real_dwell = real_df[duration_col].dropna().tolist() if duration_col else waiting_times_minutes(real_traj)
    w_dwell = wasserstein_distance(synth_dwell, real_dwell)

    synth_trip = real_trip = None
    w_trip = None
    if "trip_duration_minutes" in traj.df.columns:
        synth_trip = [t for t in traj.df["trip_duration_minutes"].dropna().tolist() if t > 0]
        real_trip = [(j / CAR_SPEED_KMH) * 60.0 for j in real_jumps if j > 0]
        if synth_trip and real_trip:
            w_trip = wasserstein_distance(synth_trip, real_trip)
    elif duration_col:
        real_trip = real_df[duration_col].dropna().tolist()
        synth_trip = waiting_times_minutes(traj)
        w_trip = wasserstein_distance(synth_trip, real_trip)

    wasserstein = [
        {"name": "Jump lengths", "value": float(w_jump), "unit": "km"},
        {"name": "Visits per user", "value": float(w_visits), "unit": "visits"},
        {"name": "Radius of gyration", "value": float(w_rog), "unit": "km"},
        {"name": "Dwell time", "value": float(w_dwell), "unit": "min"},
    ]
    if w_trip is not None:
        wasserstein.append({"name": "Trip duration (car)", "value": float(w_trip), "unit": "min"})

    # ---- ECDFs ----------------------------------------------------------- #
    ecdf = {
        "jump_lengths": _ecdf_block("synthetic", synth_jumps, observed_label, real_jumps, "jump length", "km"),
        "visits_per_user": _ecdf_block("synthetic", synth_visits_count, observed_label, real_visits_count, "number of visits", ""),
        "radius_of_gyration": _ecdf_block("synthetic", synth_rog, observed_label, real_rog, "radius of gyration", "km"),
        "dwell_time": _ecdf_block("synthetic", synth_dwell, observed_label, real_dwell, "dwell time", "min"),
    }
    if synth_trip and real_trip:
        ecdf["trip_duration"] = _ecdf_block("synthetic", synth_trip, observed_label, real_trip, "trip duration", "min")

    # ---- activity visits (shared by activity/profiles/motifs/JSD) -------- #
    synthetic_visits = observed_visits = None
    jsd: list[dict[str, Any]] = []
    real_activity_col = detect_column(real_df, _ACTIVITY_CANDIDATES)
    real_start_col = detect_column(real_df, _DATETIME_CANDIDATES)
    real_end_col = detect_column(real_df, _END_TS_CANDIDATES)
    real_location_col = detect_column(real_df, _LOCATION_CANDIDATES)
    synth_location_col = detect_column(traj.df, _LOCATION_CANDIDATES)
    resolution = _location_resolution(real_df, real_location_col)

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
        location_resolution=resolution,
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
        location_resolution=resolution,
        end_col=real_end_col,
    )
    if synthetic_visit_result is not None:
        synthetic_visits = synthetic_visit_result.visits
        if synthetic_visit_result.warning:
            warnings.append(synthetic_visit_result.warning)
    if observed_visit_result is not None:
        observed_visits = observed_visit_result.visits
        if observed_visit_result.warning:
            warnings.append(observed_visit_result.warning)

    activity = None
    if synthetic_visits is not None and observed_visits is not None:
        def _activity():
            jsd.append({"name": "Activity distribution",
                        "value": float(activity_distribution_jensen_shannon_divergence(synthetic_visits, observed_visits))})
            synth_transition = activity_transition_matrix(synthetic_visits)
            real_transition = activity_transition_matrix(observed_visits)
            jsd.append({"name": "Activity transitions",
                        "value": float(activity_transition_matrix_jensen_shannon_divergence(synth_transition, real_transition))})
            synth_daily, synth_cats, synth_bins = daily_activity_distribution(synthetic_visits)
            real_daily, real_cats, real_bins = daily_activity_distribution(observed_visits)
            jsd.append({"name": "Daily activity profile",
                        "value": float(time_bin_matrix_jensen_shannon_divergence(synth_daily, real_daily, synth_cats, real_cats))})

            return _build_activity_block(
                observed_label,
                synthetic_visits, observed_visits,
                synth_transition, real_transition,
                (synth_daily, synth_cats, synth_bins), (real_daily, real_cats, real_bins),
            )
        activity = guard("activity", _activity)

    def _micro_activity_usage():
        if synthetic_activities_path is None:
            return None
        path = Path(synthetic_activities_path)
        if not path.exists():
            return None
        activities = pd.read_parquet(path)
        if activities.empty:
            return None
        return _micro_activity_daily_usage_data(activities)

    micro_activity_usage = guard("micro_activity_usage", _micro_activity_usage)

    # ---- mobility laws --------------------------------------------------- #
    def _mobility_laws():
        real_location_col = detect_column(real_df, _LOCATION_CANDIDATES)
        real_activity_col = detect_column(real_df, _ACTIVITY_CANDIDATES)
        obs_law_visits = _mobility_law_visits(
            real_df, uid_col=real_traj.uid_col, datetime_col=real_traj.datetime_col,
            lat_col=real_traj.lat_col, lng_col=real_traj.lng_col,
            location_col=real_location_col, activity_col=real_activity_col,
        )
        syn_law_visits = _mobility_law_visits(
            traj.df, uid_col=traj.uid_col, datetime_col=traj.datetime_col,
            lat_col=traj.lat_col, lng_col=traj.lng_col, location_col=synth_location_col,
            activity_col=(synth_activity_col if synth_activity_col in traj.df.columns else None),
        )
        block: dict[str, Any] = {}
        block["travel_distance"] = guard(
            "mobility_laws.travel_distance",
            lambda: {"title": "Travel-distance mobility law", "x_label": "travel distance", "x_unit": "km",
                     **_truncated_powerlaw_series(real_jumps, synth_jumps, observed_label)})
        block["radius_of_gyration"] = guard(
            "mobility_laws.radius_of_gyration",
            lambda: {"title": "Radius-of-gyration mobility law", "x_label": "radius of gyration", "x_unit": "km",
                     **_truncated_powerlaw_series(real_rog, synth_rog, observed_label)})
        block["daily_locations"] = guard(
            "mobility_laws.daily_locations",
            lambda: {"title": "Daily visited locations", "x_label": "number of locations (N)", "x_unit": "",
                     **_lognormal_series(obs_law_visits, syn_law_visits, observed_label)})
        block["distance_frequency"] = guard(
            "mobility_laws.distance_frequency",
            lambda: {"title": "Distance-frequency visitation law", "x_label": "r · f", "x_unit": "km",
                     **_distance_frequency_series(obs_law_visits, syn_law_visits, observed_label)})
        return {k: v for k, v in block.items() if v is not None} or None
    mobility_laws = guard("mobility_laws", _mobility_laws)

    # ---- profiles -------------------------------------------------------- #
    profiles = None
    if synthetic_visits is not None and observed_visits is not None:
        profiles = guard("profiles", lambda: _build_profiles_block(
            observed_label, compute_profiles(observed_visits), compute_profiles(synthetic_visits)))

    # ---- motifs ---------------------------------------------------------- #
    motifs = None
    if observed_visits is not None or synthetic_visits is not None:
        motifs = guard("motifs", lambda: _build_motifs_block(observed_label, observed_visits, synthetic_visits, jsd))

    # ---- STVD ------------------------------------------------------------ #
    stvd = None
    if traj.lat_col and traj.lng_col and real_traj.lat_col and real_traj.lng_col:
        stvd = guard("stvd", lambda: _annotate_stvd(_compute_stvd_layers(traj, real_traj, resolutions=[7, 9])))

    # ---- social network --------------------------------------------------- #
    social_network = guard("social_network", lambda: _load_social_network_sidecar(synthetic_path))

    return {
        "labels": labels,
        "metrics": {"wasserstein": wasserstein, "jsd": jsd,
                    "cpc": [{"resolution": r, "value": float(v)} for r, v in cpc_rows]},
        "ecdf": ecdf,
        "mobility_laws": mobility_laws,
        "activity": activity,
        "micro_activity_usage": micro_activity_usage,
        "profiles": profiles,
        "motifs": motifs,
        "stvd": stvd,
        "social_network": social_network,
        "warnings": warnings,
    }


# --------------------------------------------------------------------------- #
# activity / profiles / motifs blocks
# --------------------------------------------------------------------------- #
def _purpose_distribution(visits: pd.DataFrame) -> dict[str, float]:
    counts = visits["purpose"].astype(str).value_counts()
    total = float(counts.sum()) or 1.0
    return {str(k): round(float(v) / total * 100.0, 2) for k, v in counts.items()}


def _matrix_to_categories(matrix: Any) -> tuple[list[str], np.ndarray]:
    if hasattr(matrix, "index") and hasattr(matrix, "columns"):
        cats = [str(c) for c in matrix.index.tolist()]
        return cats, np.asarray(matrix.values, dtype=float)
    arr = np.asarray(matrix, dtype=float)
    return [str(i) for i in range(arr.shape[0])], arr


def _build_activity_block(
    observed_label, synthetic_visits, observed_visits,
    synth_transition, real_transition, synth_daily, real_daily,
) -> dict[str, Any]:
    # purpose distribution (grouped bar)
    syn_dist = _purpose_distribution(synthetic_visits)
    obs_dist = _purpose_distribution(observed_visits)
    categories = list(dict.fromkeys([*syn_dist, *obs_dist]))
    purpose = {
        "categories": categories,
        "series": [
            {"name": "synthetic", "role": "synthetic", "values": [syn_dist.get(c, 0.0) for c in categories]},
            {"name": observed_label, "role": "observed", "values": [obs_dist.get(c, 0.0) for c in categories]},
        ],
    }

    # transition difference heatmap (observed - synthetic), percentage points
    syn_cats, syn_mat = _matrix_to_categories(synth_transition)
    obs_cats, obs_mat = _matrix_to_categories(real_transition)
    trans_cats = list(dict.fromkeys([*syn_cats, *obs_cats]))

    def _align_sq(cats, mat):
        idx = {c: i for i, c in enumerate(trans_cats)}
        out = np.zeros((len(trans_cats), len(trans_cats)))
        src = [idx[c] for c in cats]
        out[np.ix_(src, src)] = mat
        return out

    diff = _align_sq(obs_cats, obs_mat) - _align_sq(syn_cats, syn_mat)
    transition = {
        "categories": trans_cats,
        "labels": ["synthetic", observed_label],
        "matrix": diff.round(3).tolist(),
        "limit": max(float(np.abs(diff[np.isfinite(diff)]).max()) if np.isfinite(diff).any() else 0.0, 1.0),
    }

    # daily activity difference heatmap
    syn_mat_d, syn_cats_d, syn_bins = synth_daily
    real_mat_d, real_cats_d, real_bins = real_daily
    daily = None
    if syn_bins == real_bins:
        dcats = list(dict.fromkeys([*syn_cats_d, *real_cats_d]))

        def _align_daily(cats, mat):
            mat = np.asarray(mat, float)
            idx = {c: i for i, c in enumerate(dcats)}
            out = np.zeros((len(dcats), mat.shape[1]))
            for i, c in enumerate(cats):
                out[idx[c]] = mat[i]
            return out

        ddiff = _align_daily(real_cats_d, real_mat_d) - _align_daily(syn_cats_d, syn_mat_d)
        daily = {
            "categories": dcats,
            "n_bins": int(syn_bins),
            "labels": ["synthetic", observed_label],
            "matrix": ddiff.round(3).tolist(),
            "limit": max(float(np.abs(ddiff[np.isfinite(ddiff)]).max()) if np.isfinite(ddiff).any() else 0.0, 1.0),
        }

    return {"purpose": purpose, "transition_difference": transition, "daily_activity_difference": daily}


def _profile_scatter(profiles_df: pd.DataFrame, name: str) -> dict[str, Any]:
    df = profiles_df[["degree_of_return", "intermittency", "agent_type"]].dropna()
    if len(df) > _MAX_SCATTER_POINTS:
        df = df.sample(_MAX_SCATTER_POINTS, random_state=0)
    return {
        "name": name,
        "points": [
            {"x": float(r.degree_of_return), "y": float(r.intermittency), "profile": str(r.agent_type)}
            for r in df.itertuples()
        ],
    }


def _box_stats(values: np.ndarray) -> Optional[list[float]]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return [
        float(np.min(values)), float(np.percentile(values, 25)), float(np.percentile(values, 50)),
        float(np.percentile(values, 75)), float(np.max(values)),
    ]


def _build_profiles_block(observed_label, obs_profiles, synth_profiles) -> dict[str, Any]:
    profile_order = ["Scouter", "Regular", "Routiner"]
    datasets = {"synthetic": synth_profiles, observed_label: obs_profiles}
    box: dict[str, Any] = {}
    for metric in PROFILE_METRICS:
        box[metric] = {}
        for name, df in datasets.items():
            by_profile = {}
            for profile in profile_order:
                vals = df.loc[df["agent_type"] == profile, metric].to_numpy(dtype=float)
                by_profile[profile] = _box_stats(vals)
            box[metric][name] = by_profile
    return {
        "scatter": [
            _profile_scatter(synth_profiles, "synthetic"),
            _profile_scatter(obs_profiles, observed_label),
        ],
        "profile_order": profile_order,
        "metrics": list(PROFILE_METRICS),
        "datasets": ["synthetic", observed_label],
        "box": box,
    }


def _motif_distribution(visits: pd.DataFrame):
    _, dist = discover_daily_motifs_from_agents(
        _motif_visits(visits),
        user_id_col="uid", location_id_col="location_id", purpose_col="purpose",
        timestamp_col="start_timestamp", end_timestamp_col="end_timestamp",
    )
    return dist


def _build_motifs_block(observed_label, observed_visits, synthetic_visits, jsd) -> dict[str, Any]:
    literature_rows = _literature_distribution_rows(None)
    categories = [row["hex_id"] for row in literature_rows]
    motif_label_keys, motif_label_styles = _motif_axis_label_styles()

    def _values(rows):
        by_hex = {row["hex_id"]: round(float(row["percentage"]), 2) for row in rows}
        return [by_hex.get(hexid, 0.0) for hexid in categories]

    def _series(name: str, role: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "name": name,
            "role": role,
            "values": _values(rows),
            "rows": rows,
        }

    series = [_series("Literature", "reference", literature_rows)]

    obs_dist = _motif_distribution(observed_visits) if observed_visits is not None else None
    synth_dist = _motif_distribution(synthetic_visits) if synthetic_visits is not None else None

    if obs_dist is not None:
        series.append(_series(observed_label, "observed",
                              map_motif_distribution_to_literature_basis(obs_dist)))
    if synth_dist is not None:
        series.append(_series("synthetic", "synthetic",
                              map_motif_distribution_to_literature_basis(synth_dist)))
        if obs_dist is not None:
            left = dict(zip(synth_dist["motif_id"], synth_dist["count"]))
            right = dict(zip(obs_dist["motif_id"], obs_dist["count"]))
            keys = sorted(set(left) | set(right), key=str)
            jsd.append({"name": "Daily motifs", "value": float(jensen_shannon_divergence(
                [left.get(k, 0) for k in keys], [right.get(k, 0) for k in keys]))})

    return {
        "categories": categories,
        "series": series,
        "motif_label_keys": motif_label_keys,
        "motif_label_styles": motif_label_styles,
    }
