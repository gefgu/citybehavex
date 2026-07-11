//! Mirrors `/home/gustavo/skmob-vis/skmob_vis/motifs.py`'s literature-basis
//! constants/remap and `web/backend/app/payload/legacy.py:1797-1857`'s
//! `_motif_distribution`/`_build_motifs_block`. The hard part (motif
//! discovery itself, `discover_daily_motifs_from_agents`) is already ported
//! in `comparison::activity`; this module only adds the literature-basis
//! remap and JSON assembly on top of it.

use crate::comparison::activity::{MotifDistributionRow, discover_daily_motifs_from_agents};
use crate::comparison::filters::FilterMeta;
use crate::comparison::metric_row::{MetricRow, metric_row};
use crate::comparison::metrics::jensen_shannon_divergence;
use crate::comparison::visits::motif_visits;
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use polars::prelude::*;
use serde::Serialize;
use serde_json::{Value, json};
use std::collections::BTreeMap;

pub const OTHER_MOTIF_ID: &str = "other";

/// `(literature_motif_id, percentage)`, mirrors `LITERATURE_MOTIF_PERCENTAGES`.
const LITERATURE_MOTIF_PERCENTAGES: &[(i64, f64)] = &[
    (1, 10.10),
    (2, 30.80),
    (3, 12.70),
    (4, 9.40),
    (5, 0.60),
    (6, 7.30),
    (7, 5.00),
    (8, 1.70),
    (9, 0.70),
    (10, 3.00),
    (11, 2.30),
    (12, 1.30),
    (13, 1.00),
    (14, 1.30),
    (15, 1.30),
    (16, 0.72),
    (17, 0.86),
];

/// `(literature_motif_id, fkmob_motif_id)`, mirrors `LITERATURE_TO_FKMOB_MOTIF_ID`.
/// Order matches the Python dict's insertion order (ordinals 1..17), which is
/// also the canonical `categories` order for the chart.
const LITERATURE_TO_FKMOB_MOTIF_ID: &[(i64, i64)] = &[
    (1, 0x1000000000),
    (2, 0x2000000006),
    (3, 0x30000000E4),
    (4, 0x300000008C),
    (5, 0x30000000E2),
    (6, 0x4000006818),
    (7, 0x4000004218),
    (8, 0x4000007888),
    (9, 0x4000006984),
    (10, 0x5000C80830),
    (11, 0x5000820830),
    (12, 0x5000E84030),
    (13, 0x5000C10610),
    (14, 0x6620102060),
    (15, 0x6408102060),
    (16, 0x6720802060),
    (17, 0x66040A0060),
];

/// Embedded SVG glyphs for each literature motif ID, copied from
/// `skmob-vis`'s package assets so this crate doesn't need a runtime/build
/// dependency on a sibling checkout. Keyed by `fkmob_motif_id`.
const MOTIF_SVG_BYTES: &[(i64, &[u8])] = &[
    (
        0x1000000000,
        include_bytes!("../../../assets/motifs/0x1000000000.svg"),
    ),
    (
        0x2000000006,
        include_bytes!("../../../assets/motifs/0x2000000006.svg"),
    ),
    (
        0x30000000E4,
        include_bytes!("../../../assets/motifs/0x30000000e4.svg"),
    ),
    (
        0x300000008C,
        include_bytes!("../../../assets/motifs/0x300000008c.svg"),
    ),
    (
        0x30000000E2,
        include_bytes!("../../../assets/motifs/0x30000000e2.svg"),
    ),
    (
        0x4000006818,
        include_bytes!("../../../assets/motifs/0x4000006818.svg"),
    ),
    (
        0x4000004218,
        include_bytes!("../../../assets/motifs/0x4000004218.svg"),
    ),
    (
        0x4000007888,
        include_bytes!("../../../assets/motifs/0x4000007888.svg"),
    ),
    (
        0x4000006984,
        include_bytes!("../../../assets/motifs/0x4000006984.svg"),
    ),
    (
        0x5000C80830,
        include_bytes!("../../../assets/motifs/0x5000c80830.svg"),
    ),
    (
        0x5000820830,
        include_bytes!("../../../assets/motifs/0x5000820830.svg"),
    ),
    (
        0x5000E84030,
        include_bytes!("../../../assets/motifs/0x5000e84030.svg"),
    ),
    (
        0x5000C10610,
        include_bytes!("../../../assets/motifs/0x5000c10610.svg"),
    ),
    (
        0x6620102060,
        include_bytes!("../../../assets/motifs/0x6620102060.svg"),
    ),
    (
        0x6408102060,
        include_bytes!("../../../assets/motifs/0x6408102060.svg"),
    ),
    (
        0x6720802060,
        include_bytes!("../../../assets/motifs/0x6720802060.svg"),
    ),
    (
        0x66040A0060,
        include_bytes!("../../../assets/motifs/0x66040a0060.svg"),
    ),
];

#[derive(Debug, Clone, Serialize)]
pub struct MotifBasisRow {
    pub literature_motif_id: Value,
    pub motif_id: Value,
    pub hex_id: String,
    pub percentage: f64,
    pub count: i64,
}

/// Mirrors `format_motif_hex_id` for the packed-integer case (the `None`/
/// `"other"` sentinel case is handled separately by callers here since Rust
/// doesn't need the string/int union Python's version accepts).
fn format_motif_hex_id(motif_id: i64) -> String {
    format!("{motif_id:#x}")
}

/// Rounds to 2 decimal places. Known deviation from Python's `round()`
/// (banker's/round-half-to-even) on exact `.xx5` ties -- same documented
/// tradeoff as `ecdf::downsample`; doesn't change which literature bucket a
/// value lands in, only the last digit on a rare tie.
fn round2(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}

/// Mirrors `_empty_basis`.
fn empty_basis() -> Vec<MotifBasisRow> {
    let mut rows: Vec<MotifBasisRow> = LITERATURE_TO_FKMOB_MOTIF_ID
        .iter()
        .map(|&(literature_motif_id, fkmob_motif_id)| MotifBasisRow {
            literature_motif_id: json!(literature_motif_id),
            motif_id: json!(fkmob_motif_id),
            hex_id: format_motif_hex_id(fkmob_motif_id),
            percentage: 0.0,
            count: 0,
        })
        .collect();
    rows.push(MotifBasisRow {
        literature_motif_id: json!(OTHER_MOTIF_ID),
        motif_id: json!(OTHER_MOTIF_ID),
        hex_id: "Other".to_string(),
        percentage: 0.0,
        count: 0,
    });
    rows
}

/// Mirrors `_literature_distribution_rows(None)` -- the canonical reference
/// series is always built from the hardcoded percentages, this port never
/// takes an external `literature_df` (Python's non-`None` branch is dead
/// code on every call site in `legacy.py`).
pub fn literature_distribution_rows() -> Vec<MotifBasisRow> {
    let mut rows = empty_basis();
    let total: f64 = LITERATURE_MOTIF_PERCENTAGES.iter().map(|&(_, p)| p).sum();
    let n = rows.len();
    for (row, &(_, percentage)) in rows[..n - 1].iter_mut().zip(LITERATURE_MOTIF_PERCENTAGES) {
        row.percentage = percentage;
    }
    rows[n - 1].percentage = (100.0 - total).max(0.0);
    rows
}

/// Mirrors `map_motif_distribution_to_literature_basis`.
pub fn map_motif_distribution_to_literature_basis(
    distribution: &[MotifDistributionRow],
) -> Vec<MotifBasisRow> {
    let mut rows = empty_basis();
    let n = rows.len();
    let mut by_motif_id: BTreeMap<i64, usize> = BTreeMap::new();
    for (idx, row) in rows[..n - 1].iter().enumerate() {
        by_motif_id.insert(row.motif_id.as_i64().unwrap(), idx);
    }
    let other_idx = n - 1;

    for row in distribution {
        let target_idx = by_motif_id.get(&row.motif_id).copied().unwrap_or(other_idx);
        let target = &mut rows[target_idx];
        target.percentage = round2(target.percentage + row.percentage);
        target.count += row.count;
    }
    rows
}

/// Mirrors `_motif_axis_label_styles`: `label_keys` maps each hex ID to a
/// style key, `rich_styles` maps that style key to an ECharts rich-text
/// style embedding the motif's SVG glyph as a base64 data URI.
pub fn motif_axis_label_styles() -> (BTreeMap<String, String>, BTreeMap<String, Value>) {
    let mut label_keys = BTreeMap::new();
    let mut rich_styles = BTreeMap::new();
    for (ordinal, &(_, fkmob_motif_id)) in LITERATURE_TO_FKMOB_MOTIF_ID.iter().enumerate() {
        let ordinal = ordinal + 1;
        let hex_id = format_motif_hex_id(fkmob_motif_id);
        let style_key = format!("motif_{ordinal}");
        let svg_bytes = MOTIF_SVG_BYTES
            .iter()
            .find(|&&(id, _)| id == fkmob_motif_id)
            .map(|&(_, bytes)| bytes)
            .unwrap_or(&[]);
        let data_uri = format!("data:image/svg+xml;base64,{}", BASE64.encode(svg_bytes));
        label_keys.insert(hex_id, style_key.clone());
        rich_styles.insert(
            style_key,
            json!({
                "width": 88,
                "height": 88,
                "backgroundColor": {"image": data_uri},
            }),
        );
    }
    (label_keys, rich_styles)
}

/// Mirrors `_motif_distribution`: runs the already-ported motif discovery
/// over the HOME/VISIT-collapsed visits table. `visits` must be sorted by
/// `[uid, start_timestamp]` (same precondition as
/// `discover_daily_motifs_from_agents`).
pub fn motif_distribution(visits: &DataFrame) -> anyhow::Result<Vec<MotifDistributionRow>> {
    let collapsed = motif_visits(visits)?;
    let (_, distribution) = discover_daily_motifs_from_agents(&collapsed)?;
    Ok(distribution)
}

/// Mirrors `_series`'s `"rows": rows` field: the raw row dicts, not
/// re-rounded here. `literature_distribution_rows`'s "Other" bucket is
/// deliberately unrounded (`100.0 - total`, matching Python's
/// `_literature_distribution_rows`); `map_motif_distribution_to_literature_basis`
/// rows are already rounded during accumulation.
fn basis_row_json(row: &MotifBasisRow) -> Value {
    json!({
        "literature_motif_id": row.literature_motif_id,
        "motif_id": row.motif_id,
        "hex_id": row.hex_id,
        "percentage": row.percentage,
        "count": row.count,
    })
}

fn series_values(categories: &[String], rows: &[MotifBasisRow]) -> Vec<f64> {
    let by_hex: BTreeMap<&str, f64> = rows
        .iter()
        .map(|r| (r.hex_id.as_str(), r.percentage))
        .collect();
    categories
        .iter()
        .map(|c| round2(by_hex.get(c.as_str()).copied().unwrap_or(0.0)))
        .collect()
}

fn series_json(name: &str, role: &str, categories: &[String], rows: &[MotifBasisRow]) -> Value {
    json!({
        "name": name,
        "role": role,
        "values": series_values(categories, rows),
        "rows": rows.iter().map(basis_row_json).collect::<Vec<_>>(),
    })
}

/// Mirrors `_build_motifs_block`. `jsd` accumulates a `MetricRow` for the
/// "Daily motifs" Jensen-Shannon divergence when both sides are present
/// (folded into `metrics.jsd` by the caller).
pub fn build_motifs_block(
    observed_label: &str,
    observed_visits: Option<&DataFrame>,
    synthetic_visits: Option<&DataFrame>,
    filter_meta: &FilterMeta,
    jsd: &mut Vec<MetricRow>,
) -> anyhow::Result<Value> {
    let literature_rows = literature_distribution_rows();
    let categories: Vec<String> = literature_rows.iter().map(|r| r.hex_id.clone()).collect();
    let (motif_label_keys, motif_label_styles) = motif_axis_label_styles();

    let mut series = vec![series_json(
        "Literature",
        "reference",
        &categories,
        &literature_rows,
    )];

    let obs_dist = match observed_visits {
        Some(v) if v.height() > 0 => Some(motif_distribution(v)?),
        _ => None,
    };
    let synth_dist = match synthetic_visits {
        Some(v) if v.height() > 0 => Some(motif_distribution(v)?),
        _ => None,
    };

    if let Some(obs_dist) = &obs_dist {
        series.push(series_json(
            observed_label,
            "observed",
            &categories,
            &map_motif_distribution_to_literature_basis(obs_dist),
        ));
    }
    if let Some(synth_dist) = &synth_dist {
        series.push(series_json(
            "synthetic",
            "synthetic",
            &categories,
            &map_motif_distribution_to_literature_basis(synth_dist),
        ));
        if let Some(obs_dist) = &obs_dist {
            let mut keys: Vec<i64> = synth_dist
                .iter()
                .map(|r| r.motif_id)
                .chain(obs_dist.iter().map(|r| r.motif_id))
                .collect();
            keys.sort_unstable();
            keys.dedup();
            let left_counts: BTreeMap<i64, i64> =
                synth_dist.iter().map(|r| (r.motif_id, r.count)).collect();
            let right_counts: BTreeMap<i64, i64> =
                obs_dist.iter().map(|r| (r.motif_id, r.count)).collect();
            let left: Vec<f64> = keys
                .iter()
                .map(|k| left_counts.get(k).copied().unwrap_or(0) as f64)
                .collect();
            let right: Vec<f64> = keys
                .iter()
                .map(|k| right_counts.get(k).copied().unwrap_or(0) as f64)
                .collect();
            let value = jensen_shannon_divergence(&left, &right)?;
            if let Some(row) = metric_row(filter_meta, "Daily motifs", Some(value), "") {
                jsd.push(row);
            }
        }
    }

    Ok(json!({
        "categories": categories,
        "series": series,
        "motif_label_keys": motif_label_keys,
        "motif_label_styles": motif_label_styles,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cross-checked against a live run of
    /// `skmob_vis.motifs._literature_distribution_rows(None)`.
    #[test]
    fn literature_distribution_rows_matches_python_reference() {
        let rows = literature_distribution_rows();
        assert_eq!(rows.len(), 18);
        assert_eq!(rows[0].hex_id, "0x1000000000");
        assert_eq!(rows[0].literature_motif_id, json!(1));
        assert_eq!(rows[0].motif_id, json!(68719476736i64));
        assert!((rows[0].percentage - 10.1).abs() < 1e-9);
        assert_eq!(rows[16].hex_id, "0x66040a0060");
        assert!((rows[16].percentage - 0.86).abs() < 1e-9);
        let other = rows.last().unwrap();
        assert_eq!(other.hex_id, "Other");
        assert_eq!(other.literature_motif_id, json!(OTHER_MOTIF_ID));
        // Deliberately unrounded, matching Python's `max(0.0, 100.0 - total)`.
        assert!((other.percentage - 9.920000000000002).abs() < 1e-12);
    }

    /// Cross-checked against a live run of
    /// `skmob_vis.motifs.map_motif_distribution_to_literature_basis({"motif_id": [0x1000000000, 0x2000000006, 999], "count": [5, 3, 2]})`.
    #[test]
    fn map_motif_distribution_to_literature_basis_matches_python_reference() {
        let distribution = vec![
            MotifDistributionRow {
                motif_id: 0x1000000000,
                count: 5,
                percentage: 50.0,
            },
            MotifDistributionRow {
                motif_id: 0x2000000006,
                count: 3,
                percentage: 30.0,
            },
            MotifDistributionRow {
                motif_id: 999,
                count: 2,
                percentage: 20.0,
            },
        ];
        let rows = map_motif_distribution_to_literature_basis(&distribution);
        assert_eq!(rows[0].hex_id, "0x1000000000");
        assert_eq!(rows[0].percentage, 50.0);
        assert_eq!(rows[0].count, 5);
        assert_eq!(rows[1].hex_id, "0x2000000006");
        assert_eq!(rows[1].percentage, 30.0);
        assert_eq!(rows[1].count, 3);
        for row in &rows[2..17] {
            assert_eq!(row.percentage, 0.0);
            assert_eq!(row.count, 0);
        }
        let other = rows.last().unwrap();
        assert_eq!(other.percentage, 20.0);
        assert_eq!(other.count, 2);
    }

    #[test]
    fn motif_axis_label_styles_covers_all_17_and_embeds_svg() {
        let (label_keys, rich_styles) = motif_axis_label_styles();
        assert_eq!(label_keys.len(), 17);
        assert_eq!(rich_styles.len(), 17);
        let style_key = label_keys.get("0x1000000000").unwrap();
        assert_eq!(style_key, "motif_1");
        let style = rich_styles.get(style_key).unwrap();
        let image = style["backgroundColor"]["image"].as_str().unwrap();
        assert!(image.starts_with("data:image/svg+xml;base64,"));
        assert!(image.len() > "data:image/svg+xml;base64,".len());
    }

    #[test]
    fn format_motif_hex_id_matches_python_hex_builtin() {
        assert_eq!(format_motif_hex_id(0x1000000000), "0x1000000000");
        assert_eq!(format_motif_hex_id(0x66040A0060), "0x66040a0060");
    }
}
