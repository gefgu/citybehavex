//! `stvd` chart section: spatio-temporal visit-difference GeoJSON layers.
//! Mirrors `web/backend/app/payload/legacy.py`'s STVD section, built on top
//! of `comparison::stvd::compute_stvd_layers`.

use crate::comparison::filters::filter_df;
use crate::comparison::stvd::compute_stvd_layers;
use crate::comparison::trajectory::load_trajectory;
use crate::payload::{ComparisonContext, choose_regular_filter, empty_chart_payload};
use serde_json::{Value, json};

fn classify_stvd(volume_diff: f64, peak_shift: f64, threshold: f64) -> (usize, usize) {
    let x_bin = if volume_diff < -threshold {
        0
    } else if volume_diff <= threshold {
        1
    } else {
        2
    };
    let y_bin = if peak_shift <= 2.0 {
        0
    } else if peak_shift <= 5.0 {
        1
    } else {
        2
    };
    (x_bin, y_bin)
}

pub fn stvd_section_payload(ctx: &ComparisonContext, filter_key: &str) -> anyhow::Result<Value> {
    let Some(filter) = choose_regular_filter(ctx, filter_key)? else {
        return Ok(empty_chart_payload(ctx, vec![filter_key.to_string()]));
    };
    let mut payload = empty_chart_payload(ctx, vec![filter_key.to_string()]);
    let Some(observed_path) = &ctx.observed_path else {
        return Ok(payload);
    };
    let synthetic = load_trajectory(&ctx.synthetic_path)?;
    let observed = load_trajectory(observed_path)?;
    let syn_df = filter_df(&synthetic.df, Some(&synthetic.datetime_col), &filter)?;
    let obs_df = filter_df(&observed.df, Some(&observed.datetime_col), &filter)?;
    if syn_df.height() == 0 || obs_df.height() == 0 {
        return Ok(payload);
    }
    let layers = compute_stvd_layers(
        &syn_df,
        &synthetic.lat_col,
        &synthetic.lng_col,
        &synthetic.datetime_col,
        &obs_df,
        &observed.lat_col,
        &observed.lng_col,
        &observed.datetime_col,
        &[7, 9],
    )?;
    const COLORS: [[&str; 3]; 3] = [
        ["#2c7bb6", "#abd9e9", "#ffffbf"],
        ["#74add1", "#f7f7f7", "#fdae61"],
        ["#ffffbf", "#fdae61", "#d7191c"],
    ];
    let threshold = 25.0;
    let mut out_layers = serde_json::Map::new();
    let mut lngs = Vec::new();
    let mut lats = Vec::new();
    for (res, features) in layers {
        let mut geo_features = Vec::new();
        for feature in features {
            let (x_bin, y_bin) =
                classify_stvd(feature.volume_diff_pct, feature.peak_shift_hours, threshold);
            for [lng, lat] in &feature.ring {
                lngs.push(*lng);
                lats.push(*lat);
            }
            geo_features.push(json!({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [feature.ring]},
                "properties": {
                    "area": feature.cell_hex,
                    "volume_diff_pct": feature.volume_diff_pct,
                    "peak_shift_hours": feature.peak_shift_hours,
                    "color": COLORS[y_bin][x_bin],
                    "class": y_bin * 3 + x_bin,
                },
            }));
        }
        out_layers.insert(
            res.to_string(),
            json!({"type": "FeatureCollection", "features": geo_features}),
        );
    }
    let center = if lngs.is_empty() {
        Value::Null
    } else {
        json!([
            (lngs.iter().copied().fold(f64::INFINITY, f64::min)
                + lngs.iter().copied().fold(f64::NEG_INFINITY, f64::max))
                / 2.0,
            (lats.iter().copied().fold(f64::INFINITY, f64::min)
                + lats.iter().copied().fold(f64::NEG_INFINITY, f64::max))
                / 2.0
        ])
    };
    payload["stvd"] = json!({"groups": [{
        "filter_key": filter.key,
        "filter_label": filter.label,
        "block": {
            "center": center,
            "layers": out_layers,
            "colors": COLORS,
            "threshold": threshold,
        },
    }]});
    Ok(payload)
}
