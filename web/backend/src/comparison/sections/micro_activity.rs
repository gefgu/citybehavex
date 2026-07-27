//! `micro-activity` chart section: per-activity daily-usage-percentage
//! matrix from the synthetic micro-activity sidecar. Mirrors
//! `legacy Python backend/payload/legacy.py`'s micro-activity section.

use crate::columns::detect_in;
use crate::comparison::filters::filter_df;
use crate::comparison::micro_activity::micro_activity_daily_usage_data;
use crate::comparison::trajectory::read_parquet;
use crate::payload::{ComparisonContext, choose_regular_filter, empty_chart_payload};
use polars::prelude::*;
use serde_json::json;

fn micro_activity_datetime_col(df: &DataFrame) -> Option<String> {
    let cols: Vec<&str> = df.get_column_names().iter().map(|s| s.as_str()).collect();
    detect_in(&cols, &["arrival", "start_timestamp", "datetime"])
}

pub fn micro_activity_section_payload(
    ctx: &ComparisonContext,
    filter_key: &str,
) -> anyhow::Result<serde_json::Value> {
    let Some(filter) = choose_regular_filter(ctx, filter_key)? else {
        return Ok(empty_chart_payload(ctx, vec![filter_key.to_string()]));
    };
    let mut payload = empty_chart_payload(ctx, vec![filter_key.to_string()]);
    let Some(path) = &ctx.synthetic_activities_path else {
        return Ok(payload);
    };
    if !path.exists() {
        return Ok(payload);
    }
    let activities = read_parquet(path)?;
    if activities.height() == 0 {
        return Ok(payload);
    }
    let dt_col = micro_activity_datetime_col(&activities);
    let filtered = filter_df(&activities, dt_col.as_deref(), &filter)?;
    if filtered.height() == 0 {
        return Ok(payload);
    }
    let block = micro_activity_daily_usage_data(&filtered, 10)?;
    payload["micro_activity_usage"] = json!({
        "groups": [{
            "filter_key": filter.key,
            "filter_label": filter.label,
            "block": block,
        }],
    });
    Ok(payload)
}
