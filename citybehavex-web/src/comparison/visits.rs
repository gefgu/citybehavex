//! Mirrors `comparison.py::_visits_for_comparison`, `_collapse_explicit_purposes`,
//! `_modal_location_per_user`, `_derive_purpose_groups_from_heuristic`,
//! `_prepare_activity_visits`, and `_motif_visits`.

use super::h3::h3_cells;
use super::util::to_datetime_expr;
use super::{ActivityVisitsResult};
use polars::prelude::*;

/// Mirrors `comparison.py::_visits_for_comparison`. Produces
/// `[uid, start_timestamp, end_timestamp, location_id, purpose?]`; when no
/// explicit `end_col` is given, `end_timestamp` is derived per-user as "the
/// next visit's start" (or end-of-day for a user's last visit).
#[allow(clippy::too_many_arguments)]
pub fn visits_for_comparison(
    df: &DataFrame,
    uid_col: &str,
    datetime_col: &str,
    activity_col: Option<&str>,
    location_col: Option<&str>,
    location_resolution: u8,
    end_col: Option<&str>,
    lat_col: Option<&str>,
    lng_col: Option<&str>,
) -> anyhow::Result<DataFrame> {
    let schema = df.schema();
    let dt_expr = to_datetime_expr(&schema, datetime_col);
    let mut visits = df
        .clone()
        .lazy()
        .select([col(uid_col).alias("uid"), dt_expr.alias("start_timestamp")])
        .collect()?;

    if let Some(activity_col) = activity_col {
        visits.with_column(df.column(activity_col)?.as_materialized_series().clone().with_name("purpose".into()))?;
    }

    if let Some(location_col) = location_col {
        visits.with_column(
            df.column(location_col)?.as_materialized_series().cast(&DataType::String)?.with_name("location_id".into()),
        )?;
    } else {
        let lat_name = lat_col.ok_or_else(|| anyhow::anyhow!("lat_col required when location_col is absent"))?;
        let lng_name = lng_col.ok_or_else(|| anyhow::anyhow!("lng_col required when location_col is absent"))?;
        let cells = h3_cells(df.column(lat_name)?.as_materialized_series(), df.column(lng_name)?.as_materialized_series(), location_resolution)?;
        visits.with_column(cells.with_name("location_id".into()))?;
    }

    if let Some(end_col) = end_col {
        let end_expr = to_datetime_expr(&schema, end_col);
        let end_series = df.clone().lazy().select([end_expr.alias("end_timestamp")]).collect()?;
        visits.with_column(end_series.column("end_timestamp")?.as_materialized_series().clone())?;
    } else {
        visits = visits
            .lazy()
            .sort(["uid", "start_timestamp"], SortMultipleOptions::default())
            .with_columns([col("start_timestamp").shift(lit(-1)).over([col("uid")]).alias("end_timestamp")])
            .with_columns([col("end_timestamp")
                .fill_null(col("start_timestamp").dt().truncate(lit("1d")) + lit(Duration::parse("1d")))
                .alias("end_timestamp")])
            .collect()?;
    }
    Ok(visits)
}

/// Mirrors `comparison.py::_collapse_purpose_group` + `_collapse_explicit_purposes`:
/// any purpose other than `HOME`/`WORK` (case/whitespace-insensitive) collapses to `OTHER`.
pub fn collapse_explicit_purposes(visits: &DataFrame) -> anyhow::Result<DataFrame> {
    let purpose = visits.column("purpose")?.as_materialized_series();
    let collapsed: StringChunked = purpose
        .cast(&DataType::String)?
        .str()?
        .into_iter()
        .map(|v| {
            let normalized = v.map(|s| s.trim().to_uppercase());
            match normalized.as_deref() {
                Some("HOME") => "HOME",
                Some("WORK") => "WORK",
                _ => "OTHER",
            }
        })
        .collect();
    let mut out = visits.clone();
    out.with_column(collapsed.into_series().with_name("purpose".into()))?;
    Ok(out)
}

/// Mirrors `comparison.py::_modal_location_per_user`: per-`uid` most-frequent
/// `location_id`, ties broken by ascending `location_id`.
pub fn modal_location_per_user(candidates: &DataFrame) -> anyhow::Result<DataFrame> {
    if candidates.height() == 0 {
        return Ok(candidates.select(["uid", "location_id"])?);
    }
    let counts = candidates
        .clone()
        .lazy()
        .group_by([col("uid"), col("location_id")])
        .agg([len().alias("_count")])
        .sort(
            ["uid", "_count", "location_id"],
            SortMultipleOptions::default().with_order_descending_multi([false, true, false]),
        )
        .collect()?;
    Ok(counts
        .lazy()
        .unique(Some(cols(["uid"])), UniqueKeepStrategy::First)
        .select([col("uid"), col("location_id")])
        .collect()?)
}

/// Mirrors `comparison.py::_derive_purpose_groups_from_heuristic`: HOME/WORK/OTHER
/// per row from time-of-day + repeated-location anchors (no explicit purpose column).
pub fn derive_purpose_groups_from_heuristic(visits: &DataFrame) -> anyhow::Result<DataFrame> {
    let derived = visits
        .clone()
        .lazy()
        .with_row_index("_row", None)
        .with_columns([col("start_timestamp").dt().hour().cast(DataType::Int32).alias("_hour")])
        .collect()?;

    let home_candidates = derived.clone().lazy().filter(col("_hour").is_between(lit(2), lit(5), ClosedInterval::Both)).collect()?;
    let home_loc = modal_location_per_user(&home_candidates)?
        .lazy()
        .rename(["location_id"], ["_home_loc"], true)
        .collect()?;

    let work_mask_expr = col("_hour").eq(lit(10)).or(col("_hour").is_between(lit(14), lit(16), ClosedInterval::Both));
    let work_candidates_joined = derived
        .clone()
        .lazy()
        .filter(work_mask_expr)
        .join(home_loc.clone().lazy(), [col("uid")], [col("uid")], JoinArgs::new(JoinType::Left))
        .filter(col("_home_loc").is_null().or(col("location_id").neq(col("_home_loc"))))
        .select([col("uid"), col("location_id")])
        .collect()?;
    let work_loc = modal_location_per_user(&work_candidates_joined)?
        .lazy()
        .rename(["location_id"], ["_work_loc"], true)
        .collect()?;

    let result = derived
        .lazy()
        .join(home_loc.lazy(), [col("uid")], [col("uid")], JoinArgs::new(JoinType::Left))
        .join(work_loc.lazy(), [col("uid")], [col("uid")], JoinArgs::new(JoinType::Left))
        .with_columns([{
            let is_home = col("_home_loc").is_not_null().and(col("location_id").eq(col("_home_loc")));
            let is_work = col("_work_loc").is_not_null().and(col("location_id").eq(col("_work_loc")));
            when(is_home).then(lit("HOME")).when(is_work).then(lit("WORK")).otherwise(lit("OTHER")).alias("purpose")
        }])
        .sort(["_row"], SortMultipleOptions::default())
        .drop(cols(["_row", "_home_loc", "_work_loc", "_hour"]))
        .collect()?;
    Ok(result)
}

/// Mirrors `comparison.py::_prepare_activity_visits`.
#[allow(clippy::too_many_arguments)]
pub fn prepare_activity_visits(
    df: &DataFrame,
    label: &str,
    uid_col: Option<&str>,
    datetime_col: Option<&str>,
    activity_col: Option<&str>,
    location_col: Option<&str>,
    lat_col: Option<&str>,
    lng_col: Option<&str>,
    location_resolution: u8,
    end_col: Option<&str>,
) -> anyhow::Result<Option<ActivityVisitsResult>> {
    let (Some(uid_col), Some(datetime_col)) = (uid_col, datetime_col) else {
        return Ok(None);
    };
    if location_col.is_none() && (lat_col.is_none() || lng_col.is_none()) {
        return Ok(None);
    }

    let col_names: Vec<&str> = df.get_column_names().iter().map(|s| s.as_str()).collect();
    let resolved_activity_col = activity_col.filter(|c| col_names.contains(c));

    let visits = visits_for_comparison(
        df, uid_col, datetime_col, resolved_activity_col, location_col, location_resolution, end_col, lat_col, lng_col,
    )?;
    let visits = visits
        .lazy()
        .drop_nulls(Some(cols(["uid", "start_timestamp", "location_id"])))
        .collect()?;
    if visits.height() == 0 {
        return Ok(None);
    }

    if resolved_activity_col.is_some() {
        Ok(Some(ActivityVisitsResult {
            visits: collapse_explicit_purposes(&visits)?,
            used_heuristic: false,
            warning: None,
        }))
    } else {
        let warning = format!(
            "{label} has no explicit purpose column; derived HOME/WORK/OTHER with time-of-day and repeated-location heuristics."
        );
        Ok(Some(ActivityVisitsResult {
            visits: derive_purpose_groups_from_heuristic(&visits)?,
            used_heuristic: true,
            warning: Some(warning),
        }))
    }
}

/// Mirrors `comparison.py::_motif_visits`: collapses purpose to `HOME`/`VISIT`
/// (everything non-`HOME` becomes `VISIT`) for motif-graph node labeling.
pub fn motif_visits(visits: &DataFrame) -> anyhow::Result<DataFrame> {
    Ok(visits
        .clone()
        .lazy()
        .with_columns([when(col("purpose").eq(lit("HOME"))).then(col("purpose")).otherwise(lit("VISIT")).alias("purpose")])
        .collect()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_df() -> DataFrame {
        df![
            "uid" => [1i64, 1, 1, 2, 2],
            "dt" => [
                "2026-01-01T08:00:00", "2026-01-01T12:00:00", "2026-01-01T18:00:00",
                "2026-01-01T09:00:00", "2026-01-01T20:00:00",
            ],
            "purpose" => ["home", "  Work ", "shopping", "HOME", "gym"],
            "location_id" => ["a", "b", "c", "x", "y"],
        ]
        .unwrap()
    }

    #[test]
    fn visits_for_comparison_derives_end_timestamp_from_next_start() {
        let df = sample_df();
        let visits = visits_for_comparison(&df, "uid", "dt", Some("purpose"), Some("location_id"), 10, None, None, None).unwrap();
        assert_eq!(visits.height(), 5);
        assert!(visits.get_column_names().iter().any(|c| c.as_str() == "end_timestamp"));
    }

    #[test]
    fn collapse_explicit_purposes_normalizes_case_and_whitespace() {
        let df = sample_df();
        let visits = visits_for_comparison(&df, "uid", "dt", Some("purpose"), Some("location_id"), 10, None, None, None).unwrap();
        let collapsed = collapse_explicit_purposes(&visits).unwrap();
        let purposes: Vec<String> = collapsed.column("purpose").unwrap().str().unwrap().into_iter().flatten().map(str::to_string).collect();
        assert_eq!(purposes, vec!["HOME", "WORK", "OTHER", "HOME", "OTHER"]);
    }

    #[test]
    fn motif_visits_collapses_to_home_or_visit() {
        let df = sample_df();
        let visits = visits_for_comparison(&df, "uid", "dt", Some("purpose"), Some("location_id"), 10, None, None, None).unwrap();
        let collapsed = collapse_explicit_purposes(&visits).unwrap();
        let motif = motif_visits(&collapsed).unwrap();
        let purposes: Vec<String> = motif.column("purpose").unwrap().str().unwrap().into_iter().flatten().map(str::to_string).collect();
        assert_eq!(purposes, vec!["HOME", "VISIT", "VISIT", "HOME", "VISIT"]);
    }
}
