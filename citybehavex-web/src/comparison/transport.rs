//! Mirrors `comparison.py`'s transport-leg extraction: `_synthetic_transport_leg_records`,
//! `_observed_transport_leg_records`, `_transport_spatial_summary`, and the
//! small mode-normalization helpers they share.

use super::DEFAULT_MODE_ORDER;
use super::util::{haversine_km, to_datetime_expr};
use crate::columns::{DATETIME_CANDIDATES, LAT_CANDIDATES, LNG_CANDIDATES, TRANSPORT_CANDIDATES, UID_CANDIDATES, detect_in};
use polars::prelude::*;
use serde::Serialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Mirrors `comparison.py::_default_synthetic_moving_path`.
pub fn default_synthetic_moving_path(synthetic_path: &Path) -> PathBuf {
    let stem = synthetic_path.file_stem().unwrap_or_default().to_string_lossy();
    let ext = synthetic_path.extension().map(|e| format!(".{}", e.to_string_lossy())).unwrap_or_default();
    synthetic_path.with_file_name(format!("{stem}_moving{ext}"))
}

/// Mirrors `comparison.py::_normalize_transport_mode`.
fn normalize_transport_mode(value: Option<&str>, mode_map: &HashMap<String, String>) -> Option<String> {
    let raw = value?.trim();
    if raw.is_empty() {
        return None;
    }
    let lowered_raw = raw.to_lowercase();
    if matches!(lowered_raw.as_str(), "nan" | "none" | "null") {
        return None;
    }
    let mapped = mode_map
        .get(raw)
        .or_else(|| mode_map.get(&lowered_raw))
        .cloned()
        .unwrap_or(lowered_raw);
    let mapped = mapped.trim().to_lowercase();
    if mapped.is_empty() { None } else { Some(mapped) }
}

/// Mirrors `comparison.py::_transport_mode_map`: lowercase every key/value
/// of the configured `mode_map`.
pub fn transport_mode_map(raw: &HashMap<String, String>) -> HashMap<String, String> {
    raw.iter()
        .map(|(k, v)| (k.trim().to_lowercase(), v.trim().to_lowercase()))
        .collect()
}

fn empty_transport_records() -> anyhow::Result<DataFrame> {
    Ok(df![
        "source" => Vec::<String>::new(),
        "mode" => Vec::<String>::new(),
        "jump_km" => Vec::<f64>::new(),
        "duration_min" => Vec::<f64>::new(),
    ]?)
}

/// Builds a `(raw_mode, normalized_mode)` two-column lookup frame over only
/// the *distinct* raw mode values present, then left-joins it onto `df` --
/// matches the Python source's "normalize over distinct raw values only,
/// not every row" performance note, without needing an exact port of
/// Polars-Python's `Expr.replace_strict` (no equivalent Rust-side API).
fn join_normalized_mode(df: LazyFrame, mode_col: &str, mode_map: &HashMap<String, String>) -> anyhow::Result<LazyFrame> {
    let raw_modes: Vec<Option<String>> = df
        .clone()
        .select([col(mode_col)])
        .unique(None, UniqueKeepStrategy::First)
        .drop_nulls(None)
        .collect()?
        .column(mode_col)?
        .as_materialized_series()
        .cast(&DataType::String)?
        .str()?
        .into_iter()
        .map(|v| v.map(str::to_string))
        .collect();
    let normalized: Vec<Option<String>> = raw_modes
        .iter()
        .map(|raw| normalize_transport_mode(raw.as_deref(), mode_map))
        .collect();
    let lookup = df![
        "_raw" => raw_modes,
        "mode" => normalized,
    ]?
    .lazy();
    Ok(df.join(
        lookup,
        [col(mode_col).cast(DataType::String)],
        [col("_raw")],
        JoinArgs::new(JoinType::Left),
    ))
}

/// Mirrors `comparison.py::_synthetic_transport_leg_records`: one row per
/// `(uid, stop_id)` transport leg (total path length + time span), fully
/// lazy/streaming since the `_moving.parquet` sidecar is 100M-700M+ rows at
/// yjmob/yjmob2 scale.
pub fn synthetic_transport_leg_records(moving_path: &Path, mode_map: &HashMap<String, String>) -> anyhow::Result<DataFrame> {
    let lf = LazyFrame::scan_parquet(PlPath::new(&moving_path.to_string_lossy()), ScanArgsParquet::default())?;
    let schema = lf.clone().collect_schema()?;
    let required = ["uid", "stop_id", "seq", "lat", "lng", "t", "mode"];
    let missing: Vec<&str> = required.iter().copied().filter(|c| schema.get(c).is_none()).collect();
    if !missing.is_empty() {
        anyhow::bail!("synthetic moving sidecar missing columns: {:?}", missing);
    }

    let t_expr = to_datetime_expr(&schema, "t");
    let work = lf
        .select([col("uid"), col("stop_id"), col("seq"), col("lat"), col("lng"), col("t"), col("mode")])
        .with_columns([
            t_expr.alias("t"),
            col("lat").cast(DataType::Float64),
            col("lng").cast(DataType::Float64),
        ])
        .drop_nulls(Some(cols(["uid", "stop_id", "seq", "lat", "lng", "t", "mode"])))
        .sort(["uid", "stop_id", "seq"], SortMultipleOptions::default());

    let legs = work.clone().group_by([col("uid"), col("stop_id")]).agg([
        len().alias("_n"),
        col("t").max().alias("_t_max"),
        col("t").min().alias("_t_min"),
        col("mode").drop_nulls().first().alias("_raw_mode"),
    ]);

    let finite = work
        .filter(col("lat").is_finite().and(col("lng").is_finite()))
        .with_columns([
            col("lat").shift(lit(1)).over([col("uid"), col("stop_id")]).alias("_prev_lat"),
            col("lng").shift(lit(1)).over([col("uid"), col("stop_id")]).alias("_prev_lng"),
        ])
        .with_columns([super::util::haversine_km_expr(col("_prev_lat"), col("_prev_lng"), col("lat"), col("lng")).alias("_step_km")]);
    let distances = finite.group_by([col("uid"), col("stop_id")]).agg([
        col("_step_km").sum().alias("jump_km"),
        len().alias("_valid_n"),
    ]);

    let legs = legs
        .join(distances, [col("uid"), col("stop_id")], [col("uid"), col("stop_id")], JoinArgs::new(JoinType::Left))
        .filter(col("_n").gt_eq(lit(2)).and(col("_valid_n").fill_null(lit(0)).gt_eq(lit(2))));

    let legs = legs.collect_with_engine(Engine::Streaming)?;
    if legs.height() == 0 {
        return empty_transport_records();
    }

    let legs = join_normalized_mode(legs.lazy(), "_raw_mode", mode_map)?
        .with_columns([
            ((col("_t_max") - col("_t_min")).dt().total_seconds().cast(DataType::Float64) / lit(60.0)).alias("duration_min"),
        ])
        .filter(col("mode").is_not_null())
        .collect()?;
    if legs.height() == 0 {
        return empty_transport_records();
    }

    Ok(legs
        .lazy()
        .select([
            lit("synthetic").alias("source"),
            col("mode"),
            col("jump_km").fill_null(lit(0.0)),
            col("duration_min"),
        ])
        .collect()?)
}

/// Mirrors `comparison.py::_observed_transport_leg_records`: one row per
/// consecutive pair of observations for the same uid, via columnar window-
/// function shifts instead of a per-user Python loop.
#[allow(clippy::too_many_arguments)]
pub fn observed_transport_leg_records(
    observed_df: &DataFrame,
    uid_col: Option<&str>,
    datetime_col: Option<&str>,
    lat_col: Option<&str>,
    lng_col: Option<&str>,
    transport_col: Option<&str>,
    duration_col: Option<&str>,
    mode_map: &HashMap<String, String>,
) -> anyhow::Result<DataFrame> {
    let col_names: Vec<&str> = observed_df.get_column_names().iter().map(|s| s.as_str()).collect();
    let uid = uid_col.map(str::to_string).or_else(|| detect_in(&col_names, UID_CANDIDATES));
    let dt = datetime_col.map(str::to_string).or_else(|| detect_in(&col_names, DATETIME_CANDIDATES));
    let lat = lat_col.map(str::to_string).or_else(|| detect_in(&col_names, LAT_CANDIDATES));
    let lng = lng_col.map(str::to_string).or_else(|| detect_in(&col_names, LNG_CANDIDATES));
    let mode_col = transport_col.map(str::to_string).or_else(|| detect_in(&col_names, TRANSPORT_CANDIDATES));

    let mut missing = Vec::new();
    for (name, value) in [("uid_col", &uid), ("datetime_col", &dt), ("lat_col", &lat), ("lng_col", &lng), ("transport_col", &mode_col)] {
        match value {
            Some(v) if col_names.contains(&v.as_str()) => {}
            _ => missing.push(name),
        }
    }
    if !missing.is_empty() {
        anyhow::bail!("observed transport comparison missing columns: {}", missing.join(", "));
    }
    let (uid, dt, lat, lng, mode_col) = (uid.unwrap(), dt.unwrap(), lat.unwrap(), lng.unwrap(), mode_col.unwrap());

    let schema = observed_df.schema();
    let dt_expr = to_datetime_expr(&schema, &dt);
    let dur = duration_col.filter(|c| col_names.contains(c));

    let mut select_cols = vec![col(&uid), col(&dt), col(&lat), col(&lng), col(&mode_col)];
    if let Some(dur) = dur {
        select_cols.push(col(dur));
    }

    let work = observed_df
        .clone()
        .lazy()
        .select(select_cols)
        .with_columns([
            dt_expr.alias(&dt),
            col(&lat).cast(DataType::Float64),
            col(&lng).cast(DataType::Float64),
        ])
        .drop_nulls(Some(cols([uid.as_str(), dt.as_str(), lat.as_str(), lng.as_str(), mode_col.as_str()])))
        .sort([uid.as_str(), dt.as_str()], SortMultipleOptions::default());

    let mut work_df = work.collect()?;
    if work_df.height() == 0 {
        return empty_transport_records();
    }

    work_df = work_df
        .lazy()
        .with_columns([
            col(&lat).shift(lit(1)).over([col(&uid)]).alias("_prev_lat"),
            col(&lng).shift(lit(1)).over([col(&uid)]).alias("_prev_lng"),
            col(&dt).shift(lit(1)).over([col(&uid)]).alias("_prev_t"),
        ])
        .filter(col("_prev_lat").is_not_null())
        .collect()?;
    if work_df.height() == 0 {
        return empty_transport_records();
    }

    // Haversine over materialized arrays (matches `_haversine_km_np`; this
    // path isn't lazy/streaming in the Python source either since observed
    // data is much smaller than the synthetic `_moving.parquet` sidecar).
    let prev_lat = work_df.column("_prev_lat")?.f64()?.clone();
    let prev_lng = work_df.column("_prev_lng")?.f64()?.clone();
    let this_lat = work_df.column(&lat)?.f64()?.clone();
    let this_lng = work_df.column(&lng)?.f64()?.clone();
    let jump_km: Vec<f64> = (0..work_df.height())
        .map(|i| {
            match (prev_lat.get(i), prev_lng.get(i), this_lat.get(i), this_lng.get(i)) {
                (Some(a), Some(b), Some(c), Some(d)) => haversine_km(a, b, c, d),
                _ => f64::NAN,
            }
        })
        .collect();
    work_df.with_column(Series::new("jump_km".into(), jump_km))?;
    work_df = work_df.lazy().filter(col("jump_km").is_finite()).collect()?;
    if work_df.height() == 0 {
        return empty_transport_records();
    }

    work_df = if let Some(dur) = dur {
        work_df
            .lazy()
            .with_columns([{
                let dur_f = col(dur).cast(DataType::Float64);
                let valid = dur_f.clone().is_not_null().and(dur_f.clone().is_finite());
                when(valid).then(dur_f).otherwise(lit(NULL)).alias("duration_min")
            }])
            .collect()?
    } else {
        work_df
            .lazy()
            .with_columns([((col(&dt) - col("_prev_t")).dt().total_seconds().cast(DataType::Float64) / lit(60.0)).alias("duration_min")])
            .collect()?
    };

    work_df = join_normalized_mode(work_df.lazy(), &mode_col, mode_map)?.collect()?;
    work_df = work_df.lazy().filter(col("mode").is_not_null()).collect()?;
    if work_df.height() == 0 {
        return empty_transport_records();
    }

    Ok(work_df
        .lazy()
        .select([lit("observed").alias("source"), col("mode"), col("jump_km"), col("duration_min")])
        .collect()?)
}

#[derive(Debug, Clone, Serialize)]
pub struct ModeSummary {
    pub mode: String,
    pub count: i64,
    pub percent: f64,
    pub mean_jump_km: Option<f64>,
    pub mean_duration_min: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SourceSummary {
    pub total_trips: i64,
    pub modes: Vec<ModeSummary>,
}

/// Mirrors `comparison.py::_transport_spatial_summary`.
pub fn transport_spatial_summary(records: &DataFrame) -> anyhow::Result<HashMap<String, SourceSummary>> {
    let mut out = HashMap::new();
    if records.height() == 0 {
        return Ok(out);
    }
    let sources: Vec<String> = records
        .column("source")?
        .as_materialized_series()
        .str()?
        .into_iter()
        .flatten()
        .map(str::to_string)
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();

    for source in sources {
        let src = records
            .clone()
            .lazy()
            .filter(col("source").eq(lit(source.clone())))
            .collect()?;
        let total = src.height() as i64;
        let mut modes: Vec<String> = src
            .column("mode")?
            .as_materialized_series()
            .str()?
            .into_iter()
            .flatten()
            .map(str::to_string)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        modes.sort_by_key(|m| {
            let order = DEFAULT_MODE_ORDER.iter().position(|d| d == m).unwrap_or(99);
            (order, m.clone())
        });

        let mut mode_rows = Vec::new();
        for mode in modes {
            let mode_df = src.clone().lazy().filter(col("mode").eq(lit(mode.clone()))).collect()?;
            let n = mode_df.height() as i64;
            let mean_jump_km = if n > 0 {
                mode_df.column("jump_km")?.as_materialized_series().mean()
            } else {
                None
            };
            let durations = mode_df.column("duration_min")?.as_materialized_series().drop_nulls();
            let mean_duration_min = if durations.len() > 0 { durations.mean() } else { None };
            mode_rows.push(ModeSummary {
                mode,
                count: n,
                percent: if total > 0 { n as f64 / total as f64 * 100.0 } else { 0.0 },
                mean_jump_km,
                mean_duration_min,
            });
        }
        out.insert(source, SourceSummary { total_trips: total, modes: mode_rows });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_mode_lowercases_and_maps() {
        let mut map = HashMap::new();
        map.insert("driving".to_string(), "car".to_string());
        assert_eq!(normalize_transport_mode(Some("Driving"), &map), Some("car".to_string()));
        assert_eq!(normalize_transport_mode(Some("  Walk  "), &map), Some("walk".to_string()));
        assert_eq!(normalize_transport_mode(Some("NaN"), &map), None);
        assert_eq!(normalize_transport_mode(Some(""), &map), None);
        assert_eq!(normalize_transport_mode(None, &map), None);
    }

    #[test]
    fn default_synthetic_moving_path_appends_suffix() {
        let p = default_synthetic_moving_path(Path::new("data/gparis/results/trajectories.parquet"));
        assert_eq!(p, PathBuf::from("data/gparis/results/trajectories_moving.parquet"));
    }

    #[test]
    fn transport_spatial_summary_computes_percentages() {
        let df = df![
            "source" => ["synthetic", "synthetic", "synthetic"],
            "mode" => ["car", "car", "walk"],
            "jump_km" => [10.0, 20.0, 1.0],
            "duration_min" => [15.0, 25.0, 5.0],
        ]
        .unwrap();
        let summary = transport_spatial_summary(&df).unwrap();
        let syn = &summary["synthetic"];
        assert_eq!(syn.total_trips, 3);
        let car = syn.modes.iter().find(|m| m.mode == "car").unwrap();
        assert_eq!(car.count, 2);
        assert!((car.percent - 66.666666).abs() < 1e-3);
        assert!((car.mean_jump_km.unwrap() - 15.0).abs() < 1e-9);
    }

    /// Cross-checked against the live Python backend's
    /// `_synthetic_transport_leg_records(path, mode_map={})` +
    /// `_transport_spatial_summary(legs)` on the same file: leg counts,
    /// mode split, and durations match exactly (37934 total legs;
    /// walk=10291/27.13%, bike=3010/7.93%, car=20601/54.31%, rail=4032/10.63%).
    ///
    /// `mean_jump_km` is a **deliberate exception**: Python's
    /// `_haversine_km_expr` clamps via `pl.min_horizontal(a.sqrt(), lit(1.0))`,
    /// and `min_horizontal` silently *skips* nulls instead of propagating
    /// them (confirmed directly against the installed polars: `min_horizontal(None,
    /// 1.0) == 1.0`, not null). Since `a` is null for every leg's first
    /// waypoint (no predecessor to diff against), every single leg picks up
    /// a spurious `arcsin(1.0) -> 2*R*(pi/2) ~= 20015.09 km` "jump" baked
    /// into its total -- confirmed on this file: Python reports
    /// `car.mean_jump_km == 20028.458052213467`, i.e. exactly this port's
    /// physically-correct `13.343610177541207` plus that constant offset.
    /// This port intentionally does NOT reproduce that bug (nulls propagate
    /// correctly through `.clip_max()`), matching the physically-correct
    /// value instead -- confirmed by independently recomputing the Python
    /// side with a null-safe clamp, which reproduces this port's numbers
    /// exactly for all four modes. Flagged upstream; not fixed in
    /// `citybehavex/reports/comparison.py` as part of this port.
    #[test]
    #[ignore = "requires repo data at data/gparis/results/gparis_simulation_core_trajectories_20260710T073952_moving.parquet"]
    fn gparis_moving_sidecar_matches_python_reference() {
        let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf();
        let path = repo_root.join(
            "data/gparis/results/gparis_simulation_core_trajectories_20260710T073952_moving.parquet",
        );
        let legs = synthetic_transport_leg_records(&path, &HashMap::new()).unwrap();
        assert_eq!(legs.height(), 37934);

        let summary = transport_spatial_summary(&legs).unwrap();
        let syn = &summary["synthetic"];
        assert_eq!(syn.total_trips, 37934);

        // (mode, count, percent, physically-correct mean_jump_km, mean_duration_min)
        let expected = [
            ("walk", 10291i64, 27.128697210945322f64, 0.2996713489659243f64, 3.7459041881255457f64),
            ("bike", 3010, 7.934834185690937, 2.460197088424463, 9.840537098560354),
            ("car", 20601, 54.30748141508936, 13.343610177541207, 13.903950455479507),
            ("rail", 4032, 10.628987188274372, 23.325795465925367, 32.95834986772487),
        ];
        assert_eq!(syn.modes.len(), expected.len());
        for (row, (mode, count, percent, mean_jump_km, mean_duration_min)) in syn.modes.iter().zip(expected.iter()) {
            assert_eq!(&row.mode, mode);
            assert_eq!(row.count, *count);
            assert!((row.percent - percent).abs() < 1e-6, "{mode} percent: got {}", row.percent);
            assert!(
                (row.mean_jump_km.unwrap() - mean_jump_km).abs() < 1e-6,
                "{mode} mean_jump_km: got {:?}",
                row.mean_jump_km
            );
            assert!(
                (row.mean_duration_min.unwrap() - mean_duration_min).abs() < 1e-6,
                "{mode} mean_duration_min: got {:?}",
                row.mean_duration_min
            );
        }
    }
}
