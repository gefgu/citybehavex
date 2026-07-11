//! On-demand home/work density maps for the axum backend.
//!
//! Mirrors `web/backend/app/home_work_data.py`, but uses Rust `h3o` for H3
//! bucketing/polygon generation instead of DuckDB's community H3 extension.

use crate::columns::{
    ACTIVITY_CANDIDATES, LAT_CANDIDATES, LNG_CANDIDATES, UID_CANDIDATES, detect_in,
};
use crate::datasource::{parquet_columns, quote_path};
use h3o::{CellIndex, LatLng, Resolution};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::Path;

const DISPLAY_RESOLUTIONS: [u8; 2] = [7, 9];
const SEQ_BLUES: [&str; 5] = ["#eff3ff", "#bdd7e7", "#6baed6", "#3182bd", "#08519c"];
const SEQ_ORANGES: [&str; 5] = ["#feedde", "#fdbe85", "#fd8d3c", "#e6550d", "#a63603"];

const AGE_BRACKETS: [(&str, &str, i64, i64); 5] = [
    ("16_24", "16-24", 16, 24),
    ("25_34", "25-34", 25, 34),
    ("35_44", "35-44", 35, 44),
    ("45_59", "45-59", 45, 59),
    ("60_80", "60-80", 60, 80),
];

const JOBS: [&str; 10] = [
    "Managers",
    "Professionals",
    "Technicians and associate professionals",
    "Clerical support workers",
    "Service and sales workers",
    "Skilled agricultural, forestry and fishery workers",
    "Craft and related trades workers",
    "Plant and machine operators, and assemblers",
    "Elementary occupations",
    "Armed forces occupations",
];

#[derive(Debug, Clone)]
pub struct DemoFilter {
    pub gender: Option<String>,
    pub age_bracket: Option<String>,
    pub job: Option<String>,
}

impl DemoFilter {
    fn is_empty(&self) -> bool {
        self.gender.is_none() && self.age_bracket.is_none() && self.job.is_none()
    }
}

fn detect_cols(path: &Path) -> anyhow::Result<HashMap<&'static str, Option<String>>> {
    let columns = parquet_columns(path)?;
    Ok(HashMap::from([
        (
            "uid",
            detect_in(
                &columns.iter().map(String::as_str).collect::<Vec<_>>(),
                UID_CANDIDATES,
            ),
        ),
        (
            "lat",
            detect_in(
                &columns.iter().map(String::as_str).collect::<Vec<_>>(),
                LAT_CANDIDATES,
            ),
        ),
        (
            "lng",
            detect_in(
                &columns.iter().map(String::as_str).collect::<Vec<_>>(),
                LNG_CANDIDATES,
            ),
        ),
        (
            "purpose",
            detect_in(
                &columns.iter().map(String::as_str).collect::<Vec<_>>(),
                ACTIVITY_CANDIDATES,
            ),
        ),
    ]))
}

fn profile_filter_sql(profiles_path: Option<&Path>, demo: &DemoFilter) -> String {
    let Some(path) = profiles_path.filter(|p| p.exists()) else {
        return String::new();
    };
    if demo.is_empty() {
        return String::new();
    }
    let mut clauses = Vec::new();
    if let Some(gender) = &demo.gender {
        clauses.push(format!("gender = '{}'", gender.replace('\'', "''")));
    }
    if let Some(age_key) = &demo.age_bracket {
        if let Some((_, _, min, max)) = AGE_BRACKETS.iter().find(|(key, _, _, _)| key == age_key) {
            clauses.push(format!("age >= {min} AND age <= {max}"));
        }
    }
    if let Some(job) = &demo.job {
        clauses.push(format!("job = '{}'", job.replace('\'', "''")));
    }
    if clauses.is_empty() {
        return String::new();
    }
    format!(
        " SEMI JOIN (SELECT uid FROM read_parquet('{}') WHERE {}) filt ON filt.uid = t.uid ",
        quote_path(path),
        clauses.join(" AND ")
    )
}

fn modal_points(
    path: &Path,
    cols: &HashMap<&str, Option<String>>,
    purpose: &str,
    profiles_path: Option<&Path>,
    demo: &DemoFilter,
) -> anyhow::Result<Vec<(f64, f64)>> {
    let (Some(uid), Some(lat), Some(lng), Some(purpose_col)) = (
        cols.get("uid").and_then(|v| v.as_deref()),
        cols.get("lat").and_then(|v| v.as_deref()),
        cols.get("lng").and_then(|v| v.as_deref()),
        cols.get("purpose").and_then(|v| v.as_deref()),
    ) else {
        return Ok(Vec::new());
    };
    let join = profile_filter_sql(profiles_path, demo);
    let sql = format!(
        r#"
        WITH rows AS (
            SELECT t."{uid}" AS uid, t."{lat}" AS lat, t."{lng}" AS lng
            FROM read_parquet('{path}') t
            {join}
            WHERE upper(trim(CAST(t."{purpose_col}" AS VARCHAR))) = '{purpose}'
        ),
        rounded AS (
            SELECT uid, round(lat, 6) AS lat, round(lng, 6) AS lng, count(*) AS cnt
            FROM rows
            WHERE lat BETWEEN -90 AND 90 AND lng BETWEEN -180 AND 180
            GROUP BY uid, round(lat, 6), round(lng, 6)
        ),
        modal AS (
            SELECT uid, lat, lng,
                   row_number() OVER (PARTITION BY uid ORDER BY cnt DESC, lat, lng) AS rn
            FROM rounded
        )
        SELECT lat, lng FROM modal WHERE rn = 1
        "#,
        path = quote_path(path),
        purpose = purpose.replace('\'', "''"),
    );
    let conn = duckdb::Connection::open_in_memory()?;
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map([], |row| Ok((row.get::<_, f64>(0)?, row.get::<_, f64>(1)?)))?;
    let mut out = Vec::new();
    for row in rows {
        out.push(row?);
    }
    Ok(out)
}

fn counts_by_cell(points: &[(f64, f64)], resolution: u8) -> anyhow::Result<HashMap<u64, i64>> {
    let res = Resolution::try_from(resolution)
        .map_err(|e| anyhow::anyhow!("invalid H3 resolution {resolution}: {e}"))?;
    let mut counts = HashMap::<u64, i64>::new();
    for &(lat, lng) in points {
        if let Ok(ll) = LatLng::new(lat, lng) {
            *counts.entry(u64::from(ll.to_cell(res))).or_insert(0) += 1;
        }
    }
    Ok(counts)
}

fn feature_collection(counts: &HashMap<u64, i64>) -> anyhow::Result<Value> {
    let mut features = Vec::new();
    let mut cells: Vec<_> = counts.iter().collect();
    cells.sort_by_key(|(cell, _)| **cell);
    for (&cell, &count) in cells {
        let cell_index = CellIndex::try_from(cell)
            .map_err(|e| anyhow::anyhow!("invalid H3 cell {cell}: {e}"))?;
        let boundary = cell_index.boundary();
        let mut ring: Vec<[f64; 2]> = boundary.iter().map(|ll| [ll.lng(), ll.lat()]).collect();
        if let Some(&first) = ring.first() {
            ring.push(first);
        }
        features.push(json!({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": {"area": format!("{cell:x}"), "agent_count": count},
        }));
    }
    Ok(json!({"type": "FeatureCollection", "features": features}))
}

fn quantile(sorted: &[i64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let pos = q * (sorted.len().saturating_sub(1) as f64);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo] as f64
    } else {
        let w = pos - lo as f64;
        sorted[lo] as f64 * (1.0 - w) + sorted[hi] as f64 * w
    }
}

fn annotate_panel(mut layers: HashMap<String, Value>, ramp: &[&str]) -> Value {
    let mut all_counts = Vec::new();
    let mut total_agents = 0i64;
    for fc in layers.values() {
        let counts: Vec<i64> = fc
            .get("features")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|f| f.pointer("/properties/agent_count").and_then(Value::as_i64))
            .filter(|c| *c > 0)
            .collect();
        total_agents = total_agents.max(counts.iter().sum());
        all_counts.extend(counts);
    }
    all_counts.sort_unstable();
    let breaks = [
        quantile(&all_counts, 0.2),
        quantile(&all_counts, 0.4),
        quantile(&all_counts, 0.6),
        quantile(&all_counts, 0.8),
    ];
    let mut lngs = Vec::new();
    let mut lats = Vec::new();
    for fc in layers.values_mut() {
        if let Some(features) = fc.get_mut("features").and_then(Value::as_array_mut) {
            for feature in features {
                let count = feature
                    .pointer("/properties/agent_count")
                    .and_then(Value::as_i64)
                    .unwrap_or(0);
                let class = breaks.iter().take_while(|b| count as f64 > **b).count();
                if let Some(props) = feature.get_mut("properties").and_then(Value::as_object_mut) {
                    props.insert("color".to_string(), json!(ramp[class.min(ramp.len() - 1)]));
                    props.insert("class".to_string(), json!(class));
                    props.insert(
                        "agent_pct".to_string(),
                        json!(if total_agents > 0 {
                            ((count as f64 / total_agents as f64 * 100.0) * 10000.0).round()
                                / 10000.0
                        } else {
                            0.0
                        }),
                    );
                }
                if let Some(ring) = feature
                    .pointer("/geometry/coordinates/0")
                    .and_then(Value::as_array)
                {
                    for coord in ring {
                        if let Some(pair) = coord.as_array() {
                            if pair.len() >= 2 {
                                if let (Some(lng), Some(lat)) = (pair[0].as_f64(), pair[1].as_f64())
                                {
                                    lngs.push(lng);
                                    lats.push(lat);
                                }
                            }
                        }
                    }
                }
            }
        }
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
    json!({
        "center": center,
        "layers": layers,
        "colors": ramp,
        "breaks": breaks.map(|v| (v * 10000.0).round() / 10000.0),
        "total_agents": total_agents,
    })
}

fn panel_for(
    path: &Path,
    cols: &HashMap<&str, Option<String>>,
    purpose: &str,
    profiles_path: Option<&Path>,
    demo: &DemoFilter,
    ramp: &[&str],
) -> anyhow::Result<Value> {
    let points = modal_points(path, cols, purpose, profiles_path, demo)?;
    let mut layers = HashMap::new();
    for res in DISPLAY_RESOLUTIONS {
        layers.insert(
            res.to_string(),
            feature_collection(&counts_by_cell(&points, res)?)?,
        );
    }
    Ok(annotate_panel(layers, ramp))
}

fn matched_counts(
    synthetic_path: &Path,
    cols: &HashMap<&str, Option<String>>,
    profiles_path: Option<&Path>,
    demo: &DemoFilter,
) -> anyhow::Result<(i64, i64)> {
    let total = modal_points(
        synthetic_path,
        cols,
        "HOME",
        None,
        &DemoFilter {
            gender: None,
            age_bracket: None,
            job: None,
        },
    )?
    .len() as i64;
    if profiles_path.is_none() || demo.is_empty() {
        return Ok((total, total));
    }
    let matched = modal_points(synthetic_path, cols, "HOME", profiles_path, demo)?.len() as i64;
    Ok((matched, total))
}

pub fn build_home_work(
    synthetic_path: &Path,
    observed_path: Option<&Path>,
    profiles_path: Option<&Path>,
    demo: &DemoFilter,
) -> anyhow::Result<Value> {
    let synth_cols = detect_cols(synthetic_path)?;
    let obs_cols = match observed_path.filter(|p| p.exists()) {
        Some(path) => Some(detect_cols(path)?),
        None => None,
    };
    let effective_profiles = profiles_path.filter(|p| p.exists());
    let (matched_agents, total_synthetic_agents) =
        matched_counts(synthetic_path, &synth_cols, effective_profiles, demo)?;

    let mut result = serde_json::Map::new();
    for (purpose, key) in [("HOME", "home"), ("WORK", "work")] {
        let synthetic = panel_for(
            synthetic_path,
            &synth_cols,
            purpose,
            effective_profiles,
            demo,
            &SEQ_BLUES,
        )?;
        let real = match (observed_path.filter(|p| p.exists()), &obs_cols) {
            (Some(path), Some(cols)) => Some(panel_for(
                path,
                cols,
                purpose,
                None,
                &DemoFilter {
                    gender: None,
                    age_bracket: None,
                    job: None,
                },
                &SEQ_ORANGES,
            )?),
            _ => None,
        };
        result.insert(
            key.to_string(),
            json!({"synthetic": synthetic, "real": real}),
        );
    }

    let age_brackets: Vec<Value> = AGE_BRACKETS
        .iter()
        .map(|(key, label, min, max)| json!({"key": key, "label": label, "min": min, "max": max}))
        .collect();
    Ok(json!({
        "mode": if observed_path.is_some_and(|p| p.exists()) { "comparison" } else { "synthetic_only" },
        "has_profiles": effective_profiles.is_some(),
        "matched_agents": matched_agents,
        "total_synthetic_agents": total_synthetic_agents,
        "filter": {
            "gender": demo.gender,
            "age_bracket": demo.age_bracket,
            "job": demo.job,
        },
        "filter_options": {
            "genders": ["male", "female"],
            "age_brackets": age_brackets,
            "jobs": JOBS,
        },
        "warnings": [],
        "home": result.remove("home").unwrap_or(Value::Null),
        "work": result.remove("work").unwrap_or(Value::Null),
    }))
}
