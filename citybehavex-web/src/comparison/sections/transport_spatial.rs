//! `transport-spatial` chart section: mode-share summary and jump-length
//! ECDF over synthetic (and optionally observed) transport legs. Mirrors
//! `web/backend/app/payload/legacy.py`'s transport-spatial section.

use crate::comparison::ecdf::transport_ecdf_block;
use crate::comparison::trajectory::load_trajectory;
use crate::comparison::transport::{
    default_synthetic_moving_path, observed_transport_leg_records, synthetic_transport_leg_records,
    transport_spatial_summary,
};
use crate::payload::{ComparisonContext, duration_col, empty_chart_payload};
use serde_json::{Value, json};
use std::collections::{BTreeSet, HashMap};

pub fn transport_spatial_section_payload(ctx: &ComparisonContext) -> anyhow::Result<Value> {
    let mut payload = empty_chart_payload(ctx, Vec::new());
    let mut warnings = Vec::<String>::new();
    if !ctx.transport_enabled {
        return Ok(payload);
    }
    let moving_path = ctx
        .transport_synthetic_moving_path
        .clone()
        .unwrap_or_else(|| default_synthetic_moving_path(&ctx.synthetic_path));
    if !moving_path.exists() {
        payload["warnings"] = json!([format!(
            "transport_spatial: moving sidecar not found: {}",
            moving_path.display()
        )]);
        return Ok(payload);
    }

    let mut records = synthetic_transport_leg_records(&moving_path, &ctx.transport_mode_map)?;
    if records.height() == 0 {
        payload["warnings"] = json!(["transport_spatial: no synthetic transport legs"]);
        return Ok(payload);
    }

    if ctx.transport_observed_enabled {
        if let Some(observed_path) = &ctx.observed_path {
            let observed_traj = load_trajectory(observed_path)?;
            let duration = duration_col(&observed_traj.df);
            match observed_transport_leg_records(
                &observed_traj.df,
                ctx.transport_uid_col.as_deref(),
                ctx.transport_datetime_col.as_deref(),
                ctx.transport_lat_col.as_deref(),
                ctx.transport_lng_col.as_deref(),
                ctx.transport_col.as_deref(),
                duration.as_deref(),
                &ctx.transport_mode_map,
            ) {
                Ok(observed_records) if observed_records.height() > 0 => {
                    records.vstack_mut(&observed_records)?;
                }
                Ok(_) => warnings.push("transport_spatial: no observed transport legs".to_string()),
                Err(err) => warnings.push(format!("transport_spatial.observed: {err}")),
            }
        }
    }

    let summary = transport_spatial_summary(&records)?;
    let mut modes = BTreeSet::<String>::new();
    for source in summary.values() {
        for row in &source.modes {
            modes.insert(row.mode.clone());
        }
    }
    let mut mode_order: Vec<String> = modes.into_iter().collect();
    mode_order.sort_by_key(|m| {
        let order = crate::comparison::DEFAULT_MODE_ORDER
            .iter()
            .position(|d| d == m)
            .unwrap_or(99);
        (order, m.clone())
    });

    let mut share_series = Vec::new();
    for (source, label) in [
        ("synthetic", "synthetic"),
        ("observed", ctx.observed_label.as_str()),
    ] {
        let Some(source_summary) = summary.get(source) else {
            continue;
        };
        let by_mode: HashMap<&str, f64> = source_summary
            .modes
            .iter()
            .map(|row| (row.mode.as_str(), row.percent))
            .collect();
        share_series.push(json!({
            "name": label,
            "role": source,
            "values": mode_order.iter().map(|mode| by_mode.get(mode.as_str()).copied().unwrap_or(0.0)).collect::<Vec<_>>(),
        }));
    }

    payload["transport_spatial"] = json!({
        "summary": summary,
        "share": {
            "categories": mode_order,
            "series": share_series,
        },
        "jump_ecdf": transport_ecdf_block(&records, &ctx.observed_label)?,
    });
    if !warnings.is_empty() {
        payload["warnings"] = json!(warnings);
    }
    Ok(payload)
}
