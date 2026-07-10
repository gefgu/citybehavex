//! Mirrors `comparison.py::_looks_like_panel_observations`,
//! `_adapt_evaluation_dataframe`, and `_collapse_to_stays`.

use super::EvaluationAdaptationResult;
use super::h3::h3_cells;
use super::util::fill_null_false;
use crate::columns::{DURATION_CANDIDATES, END_TS_CANDIDATES, detect_in};
use polars::prelude::*;

fn sorted(df: &DataFrame, uid_col: &str, datetime_col: &str) -> anyhow::Result<DataFrame> {
    Ok(df.sort([uid_col, datetime_col], SortMultipleOptions::default())?)
}

/// `same_user &amp; same_key`, matching consecutive-row-pair comparisons via a
/// plain (not per-partition) `.shift(1)` -- boundary rows are guarded by the
/// separate `same_user` check rather than a partitioned shift's implicit
/// null, matching the Python source's own (non-`.over()`) implementation.
fn same_as_previous(series: &Series) -> anyhow::Result<BooleanChunked> {
    Ok(series.equal(&series.shift(1))?)
}

/// Mirrors `comparison.py::_collapse_to_stays`: collapse a slot-by-slot
/// trajectory into one row per stay episode (first row of each maximal
/// same-user/same-location run).
pub fn collapse_to_stays(
    df: &DataFrame,
    uid_col: &str,
    lat_col: &str,
    lng_col: &str,
    datetime_col: &str,
) -> anyhow::Result<DataFrame> {
    let ordered = sorted(df, uid_col, datetime_col)?;
    let same_user = same_as_previous(ordered.column(uid_col)?.as_materialized_series())?;
    let same_lat = same_as_previous(ordered.column(lat_col)?.as_materialized_series())?;
    let same_lng = same_as_previous(ordered.column(lng_col)?.as_materialized_series())?;
    let same_loc = &same_lat & &same_lng;
    let new_stay = !fill_null_false(&(&same_user & &same_loc));
    Ok(ordered.filter(&new_stay)?)
}

/// Mirrors `comparison.py::_looks_like_panel_observations`: detects
/// timestamp-only panel/ping-like data (no explicit duration/end-timestamp
/// column, and a high share of consecutive same-user/same-location rows).
pub fn looks_like_panel_observations(
    df: &DataFrame,
    uid_col: &str,
    datetime_col: &str,
    lat_col: &str,
    lng_col: &str,
    location_col: Option<&str>,
    h3_resolution: u8,
) -> anyhow::Result<bool> {
    let cols: Vec<&str> = df.get_column_names().iter().map(|s| s.as_str()).collect();
    if detect_in(&cols, END_TS_CANDIDATES).is_some() || detect_in(&cols, DURATION_CANDIDATES).is_some() {
        return Ok(false);
    }
    if df.height() == 0 {
        return Ok(false);
    }

    let ordered = sorted(df, uid_col, datetime_col)?;
    let same_user = same_as_previous(ordered.column(uid_col)?.as_materialized_series())?;

    let same_loc = if let Some(location_col) = location_col.filter(|c| cols.contains(c)) {
        same_as_previous(ordered.column(location_col)?.as_materialized_series())?
    } else {
        let cells = h3_cells(
            ordered.column(lat_col)?.as_materialized_series(),
            ordered.column(lng_col)?.as_materialized_series(),
            h3_resolution,
        )?;
        same_as_previous(&cells)?
    };

    let comparable = fill_null_false(&same_user);
    let denominator = comparable.into_iter().filter(|v| v.unwrap_or(false)).count() as i64;
    if denominator <= 0 {
        return Ok(false);
    }
    let duplicate = fill_null_false(&(&same_user & &same_loc));
    let duplicate_count = duplicate.into_iter().filter(|v| v.unwrap_or(false)).count() as i64;
    let duplicate_share = duplicate_count as f64 / denominator as f64;
    Ok(duplicate_share >= 0.2)
}

pub enum AdaptationMode {
    Auto,
    Force,
    Off,
}

/// Mirrors `comparison.py::_adapt_evaluation_dataframe`.
#[allow(clippy::too_many_arguments)]
pub fn adapt_evaluation_dataframe(
    df: &DataFrame,
    label: &str,
    uid_col: &str,
    datetime_col: &str,
    lat_col: &str,
    lng_col: &str,
    mode: AdaptationMode,
    configured_location_col: Option<&str>,
    h3_resolution: u8,
) -> anyhow::Result<EvaluationAdaptationResult> {
    if matches!(mode, AdaptationMode::Off) {
        return Ok(EvaluationAdaptationResult {
            df: df.clone(),
            adapted: false,
            warning: None,
            location_col: None,
            h3_resolution: None,
        });
    }

    let cols: Vec<&str> = df.get_column_names().iter().map(|s| s.as_str()).collect();
    let location_col = configured_location_col.filter(|c| cols.contains(c));

    let should_adapt = matches!(mode, AdaptationMode::Force)
        || looks_like_panel_observations(df, uid_col, datetime_col, lat_col, lng_col, location_col, h3_resolution)?;
    if !should_adapt {
        return Ok(EvaluationAdaptationResult {
            df: df.clone(),
            adapted: false,
            warning: None,
            location_col: None,
            h3_resolution: None,
        });
    }

    let ordered = sorted(df, uid_col, datetime_col)?;
    let (key, key_source) = if let Some(location_col) = location_col {
        (
            ordered.column(location_col)?.as_materialized_series().cast(&DataType::String)?,
            format!("location column '{location_col}'"),
        )
    } else {
        let cells = h3_cells(
            ordered.column(lat_col)?.as_materialized_series(),
            ordered.column(lng_col)?.as_materialized_series(),
            h3_resolution,
        )?;
        (cells.cast(&DataType::String)?, format!("H3 resolution {h3_resolution}"))
    };

    let same_user = same_as_previous(ordered.column(uid_col)?.as_materialized_series())?;
    let same_key = same_as_previous(&key)?;
    let new_stay = !fill_null_false(&(&same_user & &same_key));

    let before = ordered.height();
    let adapted = ordered.filter(&new_stay)?;
    let after = adapted.height();

    let warning = format!(
        "{label} comparison data looks like timestamp-only panel observations; \
         evaluation metrics were adapted by collapsing consecutive rows with the same \
         {key_source} into stays ({before} rows -> {after} stays)."
    );

    Ok(EvaluationAdaptationResult {
        df: adapted,
        adapted: true,
        warning: Some(warning),
        location_col: location_col.map(str::to_string),
        h3_resolution: if location_col.is_some() { None } else { Some(h3_resolution as i64) },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn df_panel_like() -> DataFrame {
        df![
            "uid" => [1i64, 1, 1, 1, 2, 2],
            "dt" => [
                "2026-01-01T00:00:00", "2026-01-01T00:15:00", "2026-01-01T00:30:00",
                "2026-01-01T02:00:00", "2026-01-01T00:00:00", "2026-01-01T00:15:00",
            ],
            "lat" => [1.0, 1.0, 1.0, 2.0, 5.0, 5.0],
            "lng" => [1.0, 1.0, 1.0, 2.0, 5.0, 5.0],
        ]
        .unwrap()
    }

    #[test]
    fn panel_like_data_is_detected() {
        let df = df_panel_like();
        let result =
            looks_like_panel_observations(&df, "uid", "dt", "lat", "lng", None, 10).unwrap();
        assert!(result, "4/5 comparable consecutive rows share a location -> panel-like");
    }

    #[test]
    fn empty_dataframe_is_not_panel_like() {
        let df = df![
            "uid" => Vec::<i64>::new(),
            "dt" => Vec::<&str>::new(),
            "lat" => Vec::<f64>::new(),
            "lng" => Vec::<f64>::new(),
        ]
        .unwrap();
        assert!(!looks_like_panel_observations(&df, "uid", "dt", "lat", "lng", None, 10).unwrap());
    }

    #[test]
    fn collapse_to_stays_keeps_first_row_of_each_run() {
        let df = df_panel_like();
        let stays = collapse_to_stays(&df, "uid", "lat", "lng", "dt").unwrap();
        // uid 1: 3 consecutive (1.0,1.0) rows collapse to 1, plus the (2.0,2.0) row = 2 stays.
        // uid 2: 2 consecutive (5.0,5.0) rows collapse to 1 stay.
        assert_eq!(stays.height(), 3);
    }

    /// Cross-checked against the live Python backend
    /// (`_looks_like_panel_observations(obs, uid_col="user_id",
    /// datetime_col="start_timestamp", lat_col="lat", lng_col="lon",
    /// location_col=None, h3_resolution=10)` on the same file returned
    /// `False`) -- requires repo data, so `#[ignore]`d by default.
    #[test]
    #[ignore = "requires repo data at data/gparis/gparis_visitation_df.parquet"]
    fn gparis_observed_data_is_not_panel_like() {
        let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf();
        let path = repo_root.join("data/gparis/gparis_visitation_df.parquet");
        let df = super::super::trajectory::read_parquet(&path).unwrap();
        let result =
            looks_like_panel_observations(&df, "user_id", "start_timestamp", "lat", "lon", None, 10)
                .unwrap();
        assert!(!result);
    }
}
