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

from citybehavex.activities import build_catalog
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
from citybehavex.metrics import (
    build_road_network_handle,
    jump_lengths_km as road_jump_lengths_km,
    radius_of_gyration_km as road_radius_of_gyration_km,
)
from citybehavex.reports.network_validation import build_network_validation
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
TIME_USE_CATEGORIES = [
    "sleep",
    "eatdrink",
    "selfcare",
    "paidwork",
    "educatn",
    "foodprep",
    "cleanetc",
    "maintain",
    "shopserv",
    "garden",
    "petcare",
    "eldcare",
    "pkidcare",
    "ikidcare",
    "religion",
    "volorgwk",
    "commute",
    "travel",
    "sportex",
    "tvradio",
    "read",
    "compint",
    "goout",
    "leisure",
    "missing",
]


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
    label_obs: str | None,
    obs_values: Any | None,
    x_label: str,
    x_unit: str,
) -> dict[str, Any]:
    series = [{"name": label_syn, "role": "synthetic", "points": _ecdf(syn_values)}]
    if label_obs is not None and obs_values is not None:
        series.append({"name": label_obs, "role": "observed", "points": _ecdf(obs_values)})
    return {"x_label": x_label, "x_unit": x_unit, "series": series}


FILTERS = [
    {"key": "all", "label": "All", "kind": "base"},
    {"key": "weekday", "label": "Weekday", "kind": "day"},
    {"key": "weekend", "label": "Weekend", "kind": "day"},
]

_TIME_FILTERS = [
    {"key": "morning", "label": "Morning", "kind": "time", "start": 6, "end": 12},
    {"key": "afternoon", "label": "Afternoon", "kind": "time", "start": 12, "end": 18},
    {"key": "evening", "label": "Evening", "kind": "time", "start": 18, "end": 24},
    {"key": "night", "label": "Night", "kind": "time", "start": 0, "end": 6},
]


def _special_day_filters(special_days: Optional[list[dict[str, str]]]) -> list[dict[str, Any]]:
    """Turn config-declared special days (e.g. an "emergency" date range) into
    the same filter-metadata shape as the built-in weekday/weekend filters."""
    return [
        {
            "key": sd["name"],
            "label": sd["name"].replace("_", " ").title(),
            "kind": "date_range",
            "start": sd["start_date"],
            "end": sd["end_date"],
        }
        for sd in (special_days or [])
    ]


def _empty_group(meta: dict[str, Any], blocks_key: str = "blocks") -> dict[str, Any]:
    return {"filter_key": meta["key"], "filter_label": meta["label"], blocks_key: {}}


def _filter_df(df: pd.DataFrame, datetime_col: str | None, meta: dict[str, Any]) -> pd.DataFrame:
    if meta["key"] == "all" or not datetime_col or datetime_col not in df.columns:
        return df
    dt = pd.to_datetime(df[datetime_col], errors="coerce")
    if meta["kind"] == "day":
        mask = dt.dt.dayofweek < 5
        if meta["key"] == "weekend":
            mask = ~mask
    elif meta["kind"] == "date_range":
        day = dt.dt.normalize()
        mask = (day >= pd.Timestamp(meta["start"])) & (day <= pd.Timestamp(meta["end"]))
    else:
        hour = dt.dt.hour
        mask = (hour >= int(meta["start"])) & (hour < int(meta["end"]))
    return df.loc[mask.fillna(False)].copy()


def _filter_visits(visits: pd.DataFrame | None, meta: dict[str, Any]) -> pd.DataFrame | None:
    if visits is None:
        return None
    return _filter_df(visits, "start_timestamp", meta)


def _read_time_use_table(path: Path, required_columns: list[str]) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".dta":
        return pd.read_stata(path, columns=required_columns)
    if suffix == ".parquet":
        return pd.read_parquet(path, columns=required_columns)
    if suffix == ".csv":
        return pd.read_csv(path, usecols=required_columns)
    raise ValueError(f"unsupported time-use file extension: {path.suffix}")


def _load_mtus_time_use(
    path: Path,
    *,
    country: str | None,
    survey: int | None,
    weight_col: str,
) -> pd.DataFrame:
    optional_columns = ["country", "survey", "day", weight_col]
    columns = list(dict.fromkeys([*optional_columns, *TIME_USE_CATEGORIES]))
    df = _read_time_use_table(path, columns)
    missing = sorted(set(TIME_USE_CATEGORIES) - set(df.columns))
    if missing:
        raise ValueError(f"time-use file missing columns: {', '.join(missing)}")
    if weight_col not in df.columns:
        raise ValueError(f"time-use file missing weight column: {weight_col}")
    if "day" not in df.columns:
        raise ValueError("time-use file missing day column")

    if country is not None:
        if "country" not in df.columns:
            raise ValueError("time-use country filter configured but file has no country column")
        df = df[df["country"].astype(str) == str(country)]
    if survey is not None:
        if "survey" not in df.columns:
            raise ValueError("time-use survey filter configured but file has no survey column")
        df = df[pd.to_numeric(df["survey"], errors="coerce") == int(survey)]

    if df.empty:
        raise ValueError("time-use file has no rows after filters")

    df = df.copy()
    df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0).astype(float)
    if df[weight_col].sum() <= 0:
        raise ValueError(f"time-use weight column {weight_col!r} has no positive total weight")
    for col in TIME_USE_CATEGORIES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)
    df["day_group"] = np.where(df["day"].astype(str).isin(["Saturday", "Sunday"]), "Weekend", "Weekday")
    return df


def _weighted_time_use_mean(df: pd.DataFrame, category: str, weight_col: str) -> float:
    weights = df[weight_col].astype(float)
    if weights.sum() <= 0:
        return 0.0
    return float(np.average(df[category].astype(float), weights=weights))


def _time_use_observed_group(df: pd.DataFrame, meta: dict[str, Any], weight_col: str) -> dict[str, float]:
    group = df if meta["key"] == "all" else df[df["day_group"].str.lower() == meta["key"]]
    if group.empty:
        return {category: 0.0 for category in TIME_USE_CATEGORIES}
    return {
        category: _weighted_time_use_mean(group, category, weight_col)
        for category in TIME_USE_CATEGORIES
    }


def _split_activity_segments(activities: pd.DataFrame) -> pd.DataFrame:
    required = ["uid", "activity", "arrival", "departure"]
    missing = sorted(set(required) - set(activities.columns))
    if missing:
        raise ValueError(f"activities table missing columns: {', '.join(missing)}")

    work = activities[required].copy()
    work["arrival"] = pd.to_datetime(work["arrival"], errors="coerce")
    work["departure"] = pd.to_datetime(work["departure"], errors="coerce")
    work["activity"] = pd.to_numeric(work["activity"], errors="coerce")
    work = work.dropna(subset=["uid", "activity", "arrival", "departure"])
    work = work[work["departure"] > work["arrival"]]
    if work.empty:
        raise ValueError("activities table has no valid intervals")

    labels = {activity.idx: activity.name for activity in build_catalog()}
    rows: list[dict[str, Any]] = []
    for row in work.itertuples(index=False):
        activity_name = labels.get(int(row.activity))
        if activity_name not in TIME_USE_CATEGORIES:
            continue
        current = pd.Timestamp(row.arrival)
        end = pd.Timestamp(row.departure)
        while current < end:
            next_midnight = current.normalize() + pd.Timedelta(days=1)
            segment_end = min(end, next_midnight)
            rows.append(
                {
                    "uid": row.uid,
                    "date": current.date(),
                    "day_group": "Weekend" if current.dayofweek >= 5 else "Weekday",
                    "category": activity_name,
                    "minutes": (segment_end - current).total_seconds() / 60.0,
                }
            )
            current = segment_end
    if not rows:
        raise ValueError("activities table has no mappable time-use segments")
    return pd.DataFrame(rows)


def _time_use_synthetic_group(segments: pd.DataFrame, meta: dict[str, Any]) -> dict[str, float]:
    group = segments if meta["key"] == "all" else segments[segments["day_group"].str.lower() == meta["key"]]
    agent_days = group[["uid", "date"]].drop_duplicates()
    if agent_days.empty:
        return {category: 0.0 for category in TIME_USE_CATEGORIES}
    minutes = group.groupby("category")["minutes"].sum()
    n_agent_days = float(len(agent_days))
    return {
        category: float(minutes.get(category, 0.0) / n_agent_days)
        for category in TIME_USE_CATEGORIES
    }


def _build_time_use_comparison_block(
    *,
    time_use_path: str | None,
    synthetic_activities_path: str | None,
    observed_label: str,
    country: str | None,
    survey: int | None,
    weight_col: str,
) -> dict[str, Any] | None:
    if time_use_path is None or synthetic_activities_path is None:
        return None
    observed_file = Path(time_use_path)
    activities_file = Path(synthetic_activities_path)
    if not observed_file.exists():
        raise ValueError(f"time-use file not found: {observed_file}")
    if not activities_file.exists():
        raise ValueError(f"synthetic activities file not found: {activities_file}")

    observed = _load_mtus_time_use(
        observed_file,
        country=country,
        survey=survey,
        weight_col=weight_col,
    )
    segments = _split_activity_segments(pd.read_parquet(activities_file))

    groups = []
    for meta in FILTERS:
        observed_minutes = _time_use_observed_group(observed, meta, weight_col)
        synthetic_minutes = _time_use_synthetic_group(segments, meta)
        rows = []
        for category in TIME_USE_CATEGORIES:
            obs = observed_minutes[category]
            syn = synthetic_minutes[category]
            diff = syn - obs
            rows.append(
                {
                    "category": category,
                    "observed_minutes": round(obs, 6),
                    "synthetic_minutes": round(syn, 6),
                    "difference_minutes": round(diff, 6),
                    "percent_difference": round(diff / obs * 100.0, 6) if obs else None,
                    "share_of_day_difference_pct_points": round(diff / 1440.0 * 100.0, 6),
                }
            )
        groups.append(
            {
                "filter_key": meta["key"],
                "filter_label": meta["label"],
                "block": {
                    "categories": TIME_USE_CATEGORIES,
                    "labels": [observed_label, "synthetic"],
                    "rows": rows,
                },
            }
        )
    return {"groups": groups}


def _traj_like(source: skmob2.TrajDataFrame, df: pd.DataFrame) -> skmob2.TrajDataFrame:
    return skmob2.TrajDataFrame(
        df,
        datetime_col=source.datetime_col,
        lat_col=source.lat_col,
        lng_col=source.lng_col,
        uid_col=source.uid_col,
    )


def _metric_row(
    meta: dict[str, Any],
    metric_name: str,
    value: float | None,
    unit: str = "",
) -> dict[str, Any] | None:
    if value is None or not np.isfinite(value):
        return None
    row: dict[str, Any] = {
        "filter_key": meta["key"],
        "filter_label": meta["label"],
        "metric_name": metric_name,
        "name": metric_name,
        "value": float(value),
    }
    if unit:
        row["unit"] = unit
    return row


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
    observed_values: Any | None,
    synthetic_values: Any,
    label_obs: str | None,
    reference: tuple[float, float, float] = (1.5, 1.75, 400.0),
) -> dict[str, Any]:
    syn = _truncated_powerlaw_dataset(synthetic_values, "synthetic")
    datasets = [(*syn, "synthetic")]
    if observed_values is not None and label_obs is not None:
        obs = _truncated_powerlaw_dataset(observed_values, label_obs)   # (params, x, y, label)
        datasets.insert(0, (*obs, "observed"))
    all_x = [np.asarray(row[1], float) for row in datasets]
    curve_x = _curve_x(all_x, logarithmic=True)

    series = []
    fits = []
    for params, x_pts, y_pts, label, role in datasets:
        c, r0, beta, kappa = (float(v) for v in params)
        curve_y = c * np.power(curve_x + r0, -beta) * np.exp(-curve_x / kappa)
        series.append({"name": label, "role": role, "type": "scatter",
                       "points": _xy(np.asarray(x_pts, float), np.asarray(y_pts, float))})
        series.append({"name": f"{label} fit", "role": role, "type": "line",
                       "points": _xy(curve_x, curve_y)})
        fits.append({"label": label, "params": {"c": c, "r0": r0, "beta": beta, "kappa": kappa}})

    r0, beta, kappa = reference
    joined_x = np.concatenate(all_x)
    joined_y = np.concatenate([np.asarray(row[2], float) for row in datasets])
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


def _lognormal_series(observed_visits, synthetic_visits, label_obs: str | None) -> dict[str, Any]:
    syn = _daily_location_lognormal_dataset(synthetic_visits, "synthetic")
    datasets = [(*syn, "synthetic")]
    if observed_visits is not None and label_obs is not None:
        obs = _daily_location_lognormal_dataset(observed_visits, label_obs)   # (x, y, mu, sigma, label)
        datasets.insert(0, (*obs, "observed"))
    all_x = [np.asarray(row[0], float) for row in datasets]
    curve_x = _curve_x(all_x, logarithmic=False)

    series, fits = [], []
    for x_pts, y_pts, mu, sigma, label, role in datasets:
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


def _distance_frequency_series(observed_visits, synthetic_visits, label_obs: str | None) -> dict[str, Any]:
    syn = _distance_frequency_dataset(synthetic_visits, "synthetic")
    datasets = [(*syn, "synthetic")]
    if observed_visits is not None and label_obs is not None:
        obs = _distance_frequency_dataset(observed_visits, label_obs)   # (rf, rho, eta, mu, label)
        datasets.insert(0, (*obs, "observed"))
    all_x = [np.asarray(row[0], float) for row in datasets]
    curve_x = _curve_x(all_x, logarithmic=True)

    series, fits = [], []
    for rf, rho, eta, mu, label, role in datasets:
        series.append({"name": label, "role": role, "type": "scatter",
                       "points": _xy(np.asarray(rf, float), np.asarray(rho, float))})
        series.append({"name": f"{label} fit", "role": role, "type": "line",
                       "points": _xy(curve_x, mu * np.power(curve_x, -eta))})
        fits.append({"label": label, "params": {"eta": float(eta), "mu": float(mu)}})

    alpha = -2.0
    joined_x = np.concatenate(all_x)
    joined_y = np.concatenate([np.asarray(row[1], float) for row in datasets])
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
    observed_path: Optional[str],
    observed_label: str,
    synthetic_activities_path: Optional[str] = None,
    time_use_path: Optional[str] = None,
    time_use_label: str = "time-use",
    time_use_country: Optional[str] = None,
    time_use_survey: Optional[int] = None,
    time_use_weight_col: str = "propwt",
    road_nodes_path: Optional[str] = None,
    road_edges_path: Optional[str] = None,
    road_snap_max_distance_m: float = 750.0,
    special_days: Optional[list[dict[str, str]]] = None,
) -> dict[str, Any]:
    warnings: list[str] = []
    filters = [*FILTERS, *_special_day_filters(special_days)]
    distribution_filters = [*filters, *_TIME_FILTERS]

    def guard(section: str, fn):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - degrade gracefully per section
            warnings.append(f"{section}: {exc}")
            return None

    # When a cached road graph is supplied, jump lengths / radius of gyration
    # are recomputed as road-network distance (instead of skmob2's
    # straight-line Haversine) for both synthetic and real trajectories --
    # built once here and reused across every distribution-filter group below,
    # since preparing the contraction hierarchy (not querying it) is the
    # expensive step. Falls back to plain skmob2 calls when absent/missing.
    road_nodes_df = None
    road_handle = None
    if road_nodes_path and road_edges_path and Path(road_nodes_path).exists() and Path(road_edges_path).exists():
        road_nodes_df = pd.read_parquet(road_nodes_path)
        road_edges_df = pd.read_parquet(road_edges_path)
        if len(road_nodes_df) and len(road_edges_df):
            road_handle = build_road_network_handle(road_edges_df)
        else:
            road_nodes_df = None

    traj = load_trajectory(synthetic_path)
    synth_activity_col = detect_column(traj.df, _ACTIVITY_CANDIDATES)
    real_df = None
    real_traj = None
    real_dt_col = None
    real_activity_col = None
    real_start_col = None
    real_end_col = None
    real_location_col = None
    if observed_path and Path(observed_path).exists():
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
        real_activity_col = detect_column(real_df, _ACTIVITY_CANDIDATES)
        real_start_col = detect_column(real_df, _DATETIME_CANDIDATES)
        real_end_col = detect_column(real_df, _END_TS_CANDIDATES)
        real_location_col = detect_column(real_df, _LOCATION_CANDIDATES)
    elif observed_path:
        warnings.append(f"observed comparison parquet not found: {observed_path}")

    mode = "comparison" if real_df is not None and real_traj is not None else "synthetic_only"
    labels = {"synthetic": "synthetic"}
    if mode == "comparison":
        labels["observed"] = observed_label

    duration_col = detect_column(real_df, _DURATION_CANDIDATES) if real_df is not None else None
    synth_location_col = detect_column(traj.df, _LOCATION_CANDIDATES)
    resolution = _location_resolution(real_df, real_location_col) if real_df is not None else 10
    wasserstein: list[dict[str, Any]] = []
    jsd: list[dict[str, Any]] = []
    cpc_metrics: list[dict[str, Any]] = []

    def _jumps_for(tr) -> np.ndarray:
        if road_handle is not None:
            return road_jump_lengths_km(
                tr.df,
                uid_col=tr.uid_col,
                lat_col=tr.lat_col,
                lng_col=tr.lng_col,
                datetime_col=tr.datetime_col,
                handle=road_handle,
                nodes_df=road_nodes_df,
                snap_max_distance_m=road_snap_max_distance_m,
            )
        return tr.jump_lengths(merge=True)

    def _rog_for(tr) -> np.ndarray:
        if road_handle is not None:
            return road_radius_of_gyration_km(
                tr.df,
                uid_col=tr.uid_col,
                lat_col=tr.lat_col,
                lng_col=tr.lng_col,
                handle=road_handle,
                nodes_df=road_nodes_df,
                snap_max_distance_m=road_snap_max_distance_m,
            )["radius_of_gyration"].to_numpy()
        return tr.radius_of_gyration()["radius_of_gyration"].to_numpy()

    def distribution_group(meta: dict[str, Any]) -> dict[str, Any]:
        group = _empty_group(meta)
        synth_df = _filter_df(traj.df, traj.datetime_col, meta)
        if synth_df.empty:
            warnings.append(f"{meta['label']} distribution filter has no synthetic rows")
            return group
        synth_traj = _traj_like(traj, synth_df)
        real_group_df = _filter_df(real_df, real_dt_col, meta) if real_df is not None else None
        real_group_traj = (
            _traj_like(real_traj, real_group_df)
            if real_group_df is not None and real_traj is not None and not real_group_df.empty
            else None
        )

        synth_jumps = _jumps_for(synth_traj)
        real_jumps = _jumps_for(real_group_traj) if real_group_traj is not None else None
        synth_stays = _collapse_to_stays(
            synth_df, uid_col=traj.uid_col, lat_col=traj.lat_col,
            lng_col=traj.lng_col, datetime_col=traj.datetime_col,
        )
        synth_visits_count = synth_stays[traj.uid_col].value_counts().to_list()
        real_visits_count = real_group_df[real_traj.uid_col].value_counts().to_list() if real_group_df is not None and real_traj is not None else None
        synth_rog = _rog_for(synth_traj)
        real_rog = _rog_for(real_group_traj) if real_group_traj is not None else None
        synth_dwell = (
            [d for d in synth_df["dwell_minutes"].dropna().tolist() if d >= 0]
            if "dwell_minutes" in synth_df.columns
            else waiting_times_minutes(synth_traj)
        )
        real_dwell = None
        if real_group_df is not None and real_group_traj is not None:
            real_dwell = real_group_df[duration_col].dropna().tolist() if duration_col else waiting_times_minutes(real_group_traj)
        synth_trip = real_trip = None
        if "trip_duration_minutes" in synth_df.columns:
            synth_trip = [t for t in synth_df["trip_duration_minutes"].dropna().tolist() if t > 0]
            real_trip = [(j / CAR_SPEED_KMH) * 60.0 for j in (real_jumps if real_jumps is not None else []) if j > 0]
        elif duration_col and real_group_df is not None:
            synth_trip = waiting_times_minutes(synth_traj)
            real_trip = real_group_df[duration_col].dropna().tolist()

        blocks = {
            "jump_lengths": _ecdf_block("synthetic", synth_jumps, observed_label if real_jumps is not None else None, real_jumps, "jump length", "km"),
            "visits_per_user": _ecdf_block("synthetic", synth_visits_count, observed_label if real_visits_count is not None else None, real_visits_count, "number of visits", ""),
            "radius_of_gyration": _ecdf_block("synthetic", synth_rog, observed_label if real_rog is not None else None, real_rog, "radius of gyration", "km"),
            "dwell_time": _ecdf_block("synthetic", synth_dwell, observed_label if real_dwell is not None else None, real_dwell, "dwell time", "min"),
        }
        if synth_trip and (mode == "synthetic_only" or real_trip):
            blocks["trip_duration"] = _ecdf_block("synthetic", synth_trip, observed_label if real_trip else None, real_trip, "trip duration", "min")
        group["blocks"] = blocks

        if real_group_df is not None and real_group_traj is not None:
            for row in (
                _metric_row(meta, "Jump lengths", wasserstein_distance(synth_jumps, real_jumps), "km") if real_jumps is not None and len(real_jumps) else None,
                _metric_row(meta, "Visits per user", visits_per_user_wasserstein_distance(
                    synth_stays, real_group_df,
                    user_id_col1=traj.uid_col, user_id_col2=real_traj.uid_col,
                )[0], "visits"),
                _metric_row(meta, "Radius of gyration", wasserstein_distance(synth_rog, real_rog), "km") if real_rog is not None and len(real_rog) else None,
                _metric_row(meta, "Dwell time", wasserstein_distance(synth_dwell, real_dwell), "min") if real_dwell else None,
                _metric_row(meta, "Trip duration (car)", wasserstein_distance(synth_trip, real_trip), "min") if synth_trip and real_trip else None,
            ):
                if row is not None:
                    wasserstein.append(row)
        return group

    ecdf = {"groups": [guard(f"ecdf.{m['key']}", lambda m=m: distribution_group(m)) for m in distribution_filters]}
    ecdf["groups"] = [g for g in ecdf["groups"] if g is not None]

    synthetic_visits = None
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
    if synthetic_visit_result is not None:
        synthetic_visits = synthetic_visit_result.visits
        if synthetic_visit_result.warning:
            warnings.append(synthetic_visit_result.warning)
    observed_visits = None
    if real_df is not None and real_traj is not None:
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
        if observed_visit_result is not None:
            observed_visits = observed_visit_result.visits
            if observed_visit_result.warning:
                warnings.append(observed_visit_result.warning)

    activity = None
    if synthetic_visits is not None:
        def _activity_group(meta: dict[str, Any]):
            syn_v = _filter_visits(synthetic_visits, meta)
            obs_v = _filter_visits(observed_visits, meta)
            if syn_v is None or syn_v.empty:
                warnings.append(f"{meta['label']} activity filter has no synthetic visits")
                return None
            synth_transition = activity_transition_matrix(syn_v)
            synth_daily, synth_cats, synth_bins = daily_activity_distribution(syn_v)
            real_transition = real_daily_tuple = None
            if obs_v is not None and not obs_v.empty:
                jsd.extend([
                    row for row in (
                        _metric_row(meta, "Activity distribution", activity_distribution_jensen_shannon_divergence(syn_v, obs_v)),
                        _metric_row(meta, "Activity transitions", activity_transition_matrix_jensen_shannon_divergence(
                            synth_transition, activity_transition_matrix(obs_v)
                        )),
                    ) if row is not None
                ])
                real_transition = activity_transition_matrix(obs_v)
                real_daily, real_cats, real_bins = daily_activity_distribution(obs_v)
                jsd_row = _metric_row(meta, "Daily activity profile", time_bin_matrix_jensen_shannon_divergence(
                    synth_daily, real_daily, synth_cats, real_cats
                ))
                if jsd_row is not None:
                    jsd.append(jsd_row)
                real_daily_tuple = (real_daily, real_cats, real_bins)
            block = _build_activity_block(
                observed_label if obs_v is not None and not obs_v.empty else None,
                syn_v, obs_v if obs_v is not None and not obs_v.empty else None,
                synth_transition, real_transition,
                (synth_daily, synth_cats, synth_bins), real_daily_tuple,
            )
            return {"filter_key": meta["key"], "filter_label": meta["label"], **block}
        activity_groups = [guard(f"activity.{m['key']}", lambda m=m: _activity_group(m)) for m in filters]
        activity_groups = [g for g in activity_groups if g is not None]
        activity = {"groups": activity_groups} if activity_groups else None

    def _micro_activity_usage():
        if synthetic_activities_path is None:
            return None
        path = Path(synthetic_activities_path)
        if not path.exists():
            return None
        activities = pd.read_parquet(path)
        if activities.empty:
            return None
        groups = []
        dt_col = detect_column(activities, ["arrival", "start_timestamp", "datetime"])
        for meta in filters:
            filtered = _filter_df(activities, dt_col, meta)
            if filtered.empty:
                continue
            groups.append({
                "filter_key": meta["key"],
                "filter_label": meta["label"],
                "block": _micro_activity_daily_usage_data(filtered),
            })
        return {"groups": groups} if groups else None

    micro_activity_usage = guard("micro_activity_usage", _micro_activity_usage)

    time_use_comparison = guard(
        "time_use_comparison",
        lambda: _build_time_use_comparison_block(
            time_use_path=time_use_path,
            synthetic_activities_path=synthetic_activities_path,
            observed_label=time_use_label,
            country=time_use_country,
            survey=time_use_survey,
            weight_col=time_use_weight_col,
        ),
    )

    # ---- mobility laws --------------------------------------------------- #
    def _mobility_laws_group(meta: dict[str, Any]):
        synth_df = _filter_df(traj.df, traj.datetime_col, meta)
        real_group_df = _filter_df(real_df, real_dt_col, meta) if real_df is not None else None
        if synth_df.empty:
            return None
        synth_traj = _traj_like(traj, synth_df)
        real_group_traj = _traj_like(real_traj, real_group_df) if real_group_df is not None and real_traj is not None and not real_group_df.empty else None
        synth_jumps = synth_traj.jump_lengths(merge=True)
        real_jumps = real_group_traj.jump_lengths(merge=True) if real_group_traj is not None else None
        synth_rog = synth_traj.radius_of_gyration()["radius_of_gyration"].to_numpy()
        real_rog = real_group_traj.radius_of_gyration()["radius_of_gyration"].to_numpy() if real_group_traj is not None else None
        obs_law_visits = (
            _mobility_law_visits(
                real_group_df, uid_col=real_traj.uid_col, datetime_col=real_traj.datetime_col,
                lat_col=real_traj.lat_col, lng_col=real_traj.lng_col,
                location_col=real_location_col, activity_col=real_activity_col,
            )
            if real_group_df is not None and real_traj is not None and not real_group_df.empty
            else None
        )
        syn_law_visits = _mobility_law_visits(
            synth_df, uid_col=traj.uid_col, datetime_col=traj.datetime_col,
            lat_col=traj.lat_col, lng_col=traj.lng_col, location_col=synth_location_col,
            activity_col=(synth_activity_col if synth_activity_col in traj.df.columns else None),
        )
        block: dict[str, Any] = {}
        block["travel_distance"] = guard(
            f"mobility_laws.{meta['key']}.travel_distance",
            lambda: {"title": "Travel-distance mobility law", "x_label": "travel distance", "x_unit": "km",
                     **_truncated_powerlaw_series(real_jumps, synth_jumps, observed_label if real_jumps is not None else None)})
        block["radius_of_gyration"] = guard(
            f"mobility_laws.{meta['key']}.radius_of_gyration",
            lambda: {"title": "Radius-of-gyration mobility law", "x_label": "radius of gyration", "x_unit": "km",
                     **_truncated_powerlaw_series(real_rog, synth_rog, observed_label if real_rog is not None else None)})
        block["daily_locations"] = guard(
            f"mobility_laws.{meta['key']}.daily_locations",
            lambda: {"title": "Daily visited locations", "x_label": "number of locations (N)", "x_unit": "",
                     **_lognormal_series(obs_law_visits, syn_law_visits, observed_label if obs_law_visits is not None else None)})
        block["distance_frequency"] = guard(
            f"mobility_laws.{meta['key']}.distance_frequency",
            lambda: {"title": "Distance-frequency visitation law", "x_label": "r · f", "x_unit": "km",
                     **_distance_frequency_series(obs_law_visits, syn_law_visits, observed_label if obs_law_visits is not None else None)})
        blocks = {k: v for k, v in block.items() if v is not None}
        return {"filter_key": meta["key"], "filter_label": meta["label"], "blocks": blocks} if blocks else None
    mobility_groups = [guard(f"mobility_laws.{m['key']}", lambda m=m: _mobility_laws_group(m)) for m in filters]
    mobility_groups = [g for g in mobility_groups if g is not None]
    mobility_laws = {"groups": mobility_groups} if mobility_groups else None

    # ---- profiles -------------------------------------------------------- #
    profiles = None
    if synthetic_visits is not None and observed_visits is not None:
        profiles = guard("profiles", lambda: _build_profiles_block(
            observed_label, compute_profiles(observed_visits), compute_profiles(synthetic_visits)))

    # ---- motifs ---------------------------------------------------------- #
    motifs = None
    if observed_visits is not None or synthetic_visits is not None:
        motif_groups = []
        for meta in filters:
            motif = guard(
                f"motifs.{meta['key']}",
                lambda meta=meta: _build_motifs_block(
                    observed_label,
                    _filter_visits(observed_visits, meta),
                    _filter_visits(synthetic_visits, meta),
                    jsd,
                    meta,
                ),
            )
            if motif is not None:
                motif_groups.append({"filter_key": meta["key"], "filter_label": meta["label"], "block": motif})
        motifs = {"groups": motif_groups} if motif_groups else None

    # ---- STVD ------------------------------------------------------------ #
    stvd = None
    if mode == "comparison" and traj.lat_col and traj.lng_col and real_traj and real_traj.lat_col and real_traj.lng_col:
        stvd_groups = []
        for meta in filters:
            def _stvd(meta=meta):
                synth_df = _filter_df(traj.df, traj.datetime_col, meta)
                real_group_df = _filter_df(real_df, real_dt_col, meta)
                if synth_df.empty or real_group_df.empty:
                    return None
                return {
                    "filter_key": meta["key"],
                    "filter_label": meta["label"],
                    "block": _annotate_stvd(_compute_stvd_layers(
                        _traj_like(traj, synth_df),
                        _traj_like(real_traj, real_group_df),
                        resolutions=[7, 9],
                    )),
                }
            group = guard(f"stvd.{meta['key']}", _stvd)
            if group is not None:
                stvd_groups.append(group)
        stvd = {"groups": stvd_groups} if stvd_groups else None

    if mode == "comparison" and real_traj is not None:
        for meta in filters:
            def _cpc(meta=meta):
                synth_df = _filter_df(traj.df, traj.datetime_col, meta)
                real_group_df = _filter_df(real_df, real_dt_col, meta)
                if synth_df.empty or real_group_df.empty:
                    return []
                return _common_part_of_commuters(_traj_like(traj, synth_df), _traj_like(real_traj, real_group_df))
            for resolution_value, value in guard(f"cpc.{meta['key']}", _cpc) or []:
                cpc_metrics.append({
                    "filter_key": meta["key"],
                    "filter_label": meta["label"],
                    "resolution": resolution_value,
                    "value": float(value),
                })

    # ---- social network --------------------------------------------------- #
    # network_validation is served by its own endpoint/cache entry now (see
    # build_network_validation_payload below and the /network-validation
    # route in web/backend/app/api/charts.py) so its build time -- still the
    # largest single section for shanghai/yjmob even after the Rust port --
    # doesn't block first paint of the rest of the charts.
    social_network = guard("social_network", lambda: _load_social_network_sidecar(synthetic_path))

    return {
        "mode": mode,
        "labels": labels,
        "metrics": {"wasserstein": wasserstein, "jsd": jsd, "cpc": cpc_metrics},
        "ecdf": ecdf,
        "mobility_laws": mobility_laws,
        "activity": activity,
        "micro_activity_usage": micro_activity_usage,
        "time_use_comparison": time_use_comparison,
        "profiles": profiles,
        "motifs": motifs,
        "stvd": stvd,
        "social_network": social_network,
        "warnings": warnings,
    }


def build_network_validation_payload(
    synthetic_path: str,
    observed_path: Optional[str],
    network_validation_config: Optional[object],
) -> dict[str, Any]:
    """The ``network_validation`` section on its own, split out of
    ``build_comparison_payload`` so the frontend can fetch/render it
    independently (see the ``/network-validation`` route in
    ``web/backend/app/api/charts.py``) instead of blocking first paint of
    the rest of the charts on this section's build time -- even after the
    Rust-accelerated graph computation, this section's cache-miss cost is
    still the largest single piece of the comparison payload for the denser
    observed-network cities (shanghai/yjmob).

    Only reads/loads what ``build_network_validation`` itself needs (the raw
    observed parquet, uid/datetime auto-detected the same way
    ``_observed_validation_block`` already does internally) rather than the
    full trajectory-loading/column-detection machinery
    ``build_comparison_payload`` runs for every other section.
    """
    real_df = pd.read_parquet(observed_path) if observed_path and Path(observed_path).exists() else None

    nv_cfg = network_validation_config
    nv_enabled = bool(getattr(nv_cfg, "enabled", False)) if nv_cfg is not None else False
    try:
        network_validation, network_warnings = build_network_validation(
            synthetic_path,
            observed_df=real_df,
            observed_uid_col=None,
            observed_datetime_col=None,
            enabled=nv_enabled,
            synthetic_enabled=bool(getattr(nv_cfg, "synthetic_enabled", True)),
            observed_enabled=bool(getattr(nv_cfg, "observed_enabled", False)),
            location_mode=str(getattr(nv_cfg, "location_mode", "auto")),
            location_col=getattr(nv_cfg, "location_col", None),
            h3_resolution=int(getattr(nv_cfg, "h3_resolution", 9)),
            max_group_size=int(getattr(nv_cfg, "max_group_size", 200)),
            seed=int(getattr(nv_cfg, "random_seed", 42)),
        )
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, matching build_comparison_payload's guard()
        return {"network_validation": None, "warnings": [f"network_validation: {exc}"]}

    return {
        "network_validation": network_validation,
        "warnings": [f"network_validation: {warning}" for warning in network_warnings],
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
    obs_dist = _purpose_distribution(observed_visits) if observed_visits is not None else {}
    categories = list(dict.fromkeys([*syn_dist, *obs_dist]))
    purpose_series = [
        {"name": "synthetic", "role": "synthetic", "values": [syn_dist.get(c, 0.0) for c in categories]},
    ]
    if observed_label is not None and observed_visits is not None:
        purpose_series.append(
            {"name": observed_label, "role": "observed", "values": [obs_dist.get(c, 0.0) for c in categories]}
        )
    purpose = {
        "categories": categories,
        "series": purpose_series,
    }

    syn_cats, syn_mat = _matrix_to_categories(synth_transition)
    obs_cats, obs_mat = _matrix_to_categories(real_transition) if real_transition is not None else ([], None)
    trans_cats = list(dict.fromkeys([*syn_cats, *obs_cats]))

    def _align_sq(cats, mat):
        idx = {c: i for i, c in enumerate(trans_cats)}
        out = np.zeros((len(trans_cats), len(trans_cats)))
        src = [idx[c] for c in cats]
        out[np.ix_(src, src)] = mat
        return out

    if real_transition is not None and observed_label is not None:
        matrix = _align_sq(obs_cats, obs_mat) - _align_sq(syn_cats, syn_mat)
        matrix_mode = "difference"
        labels = ["synthetic", observed_label]
    else:
        matrix = _align_sq(syn_cats, syn_mat)
        matrix_mode = "raw"
        labels = ["synthetic"]
    transition = {
        "categories": trans_cats,
        "labels": labels,
        "matrix_mode": matrix_mode,
        "matrix": matrix.round(3).tolist(),
        "limit": max(float(np.abs(matrix[np.isfinite(matrix)]).max()) if np.isfinite(matrix).any() else 0.0, 1.0),
    }

    syn_mat_d, syn_cats_d, syn_bins = synth_daily
    daily = None
    if real_daily is not None:
        real_mat_d, real_cats_d, real_bins = real_daily
        can_build_daily = syn_bins == real_bins
        dcats = list(dict.fromkeys([*syn_cats_d, *real_cats_d])) if can_build_daily else []
    else:
        real_mat_d = real_cats_d = None
        can_build_daily = True
        dcats = list(syn_cats_d)
    if can_build_daily:

        def _align_daily(cats, mat):
            mat = np.asarray(mat, float)
            idx = {c: i for i, c in enumerate(dcats)}
            out = np.zeros((len(dcats), mat.shape[1]))
            for i, c in enumerate(cats):
                out[idx[c]] = mat[i]
            return out

        if real_daily is not None and observed_label is not None:
            daily_matrix = _align_daily(real_cats_d, real_mat_d) - _align_daily(syn_cats_d, syn_mat_d)
            daily_mode = "difference"
            daily_labels = ["synthetic", observed_label]
        else:
            daily_matrix = _align_daily(syn_cats_d, syn_mat_d)
            daily_mode = "raw"
            daily_labels = ["synthetic"]
        daily = {
            "categories": dcats,
            "n_bins": int(syn_bins),
            "labels": daily_labels,
            "matrix_mode": daily_mode,
            "matrix": daily_matrix.round(3).tolist(),
            "limit": max(float(np.abs(daily_matrix[np.isfinite(daily_matrix)]).max()) if np.isfinite(daily_matrix).any() else 0.0, 1.0),
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


def _build_motifs_block(
    observed_label,
    observed_visits,
    synthetic_visits,
    jsd,
    filter_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
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

    obs_dist = _motif_distribution(observed_visits) if observed_visits is not None and not observed_visits.empty else None
    synth_dist = _motif_distribution(synthetic_visits) if synthetic_visits is not None and not synthetic_visits.empty else None

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
            value = float(jensen_shannon_divergence([left.get(k, 0) for k in keys], [right.get(k, 0) for k in keys]))
            if filter_meta is None:
                jsd.append({"name": "Daily motifs", "metric_name": "Daily motifs", "value": value})
            else:
                row = _metric_row(filter_meta, "Daily motifs", value)
                if row is not None:
                    jsd.append(row)

    return {
        "categories": categories,
        "series": series,
        "motif_label_keys": motif_label_keys,
        "motif_label_styles": motif_label_styles,
    }
