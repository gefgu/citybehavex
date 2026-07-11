//! Timeline-view API routes.

use crate::cache::Cache;
use crate::config;
use crate::datasource::{quote_path, run_summary};
use crate::experiments::{self, get_experiment};
use crate::models::{ApiError, ApiResponse, ApiResult};
use axum::extract::{Path, Query};
use chrono::{DateTime, NaiveDateTime};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::{Path as FsPath, PathBuf};

#[derive(Debug, Deserialize)]
pub struct TimelineRunQuery {
    pub run: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct TimelineLegsQuery {
    pub since: String,
    pub until: String,
    pub min_lat: f64,
    pub min_lng: f64,
    pub max_lat: f64,
    pub max_lng: f64,
    pub run: Option<String>,
    #[serde(default = "default_max_agents")]
    pub max_agents: i64,
}

fn default_max_agents() -> i64 {
    2000
}

fn selected(
    exp_id: &str,
    run_id: Option<&str>,
) -> Result<(experiments::Experiment, experiments::Run), ApiError> {
    let exp = get_experiment(exp_id)
        .ok_or_else(|| ApiError::not_found(format!("unknown experiment {exp_id:?}")))?;
    let run = exp
        .run(run_id)
        .cloned()
        .ok_or_else(|| ApiError::not_found(format!("no runs found for experiment {exp_id:?}")))?;
    Ok((exp, run))
}

fn parse_datetime(value: &str) -> Result<NaiveDateTime, ApiError> {
    NaiveDateTime::parse_from_str(value, "%Y-%m-%dT%H:%M:%S")
        .or_else(|_| DateTime::parse_from_rfc3339(value).map(|dt| dt.naive_local()))
        .map_err(|e| ApiError::unprocessable(format!("invalid datetime: {e}")))
}

fn parquet_columns(path: &FsPath) -> anyhow::Result<HashSet<String>> {
    Ok(crate::datasource::parquet_columns(path)?
        .into_iter()
        .collect())
}

fn cache() -> Cache {
    Cache::new(config::cache_dir())
}

fn build_moving_index(moving_path: &FsPath, out: &FsPath) -> anyhow::Result<()> {
    let columns = parquet_columns(moving_path)?;
    let mode_expr = if columns.contains("mode") {
        "mode"
    } else {
        "'car' AS mode"
    };
    let sql = format!(
        r#"
        COPY (
            SELECT uid, stop_id, seq, lat, lng, t, {mode_expr}
            FROM read_parquet('{input}')
            ORDER BY uid, stop_id, seq
        )
        TO '{out}' (FORMAT PARQUET)
        "#,
        input = quote_path(moving_path),
        out = quote_path(out)
    );
    duckdb::Connection::open_in_memory()?.execute_batch(&sql)?;
    Ok(())
}

fn build_legs_index(trajectory_path: &FsPath, out: &FsPath) -> anyhow::Result<()> {
    let columns = parquet_columns(trajectory_path)?;
    let category_expr = if columns.contains("category") {
        "category"
    } else {
        "NULL::VARCHAR AS category"
    };
    let stop_id_expr = if columns.contains("stop_id") {
        "stop_id"
    } else {
        "NULL::BIGINT AS stop_id"
    };
    let sql = format!(
        r#"
        COPY (
            WITH ordered AS (
                SELECT uid, lat, lng, arrival, departure, trip_duration_minutes, purpose,
                       {category_expr}, {stop_id_expr},
                       LAG(lat) OVER w AS o_lat,
                       LAG(lng) OVER w AS o_lng
                FROM read_parquet('{input}')
                WINDOW w AS (PARTITION BY uid ORDER BY arrival)
            ),
            combined AS (
                SELECT uid, 'dwell' AS kind,
                       arrival AS t_start, departure AS t_end,
                       lat AS o_lat, lng AS o_lng, lat AS d_lat, lng AS d_lng, purpose, category,
                       NULL::BIGINT AS stop_id
                FROM ordered
                UNION ALL
                SELECT uid, 'leg' AS kind,
                       arrival - (trip_duration_minutes * INTERVAL '1 minute') AS t_start,
                       arrival AS t_end,
                       o_lat, o_lng, lat AS d_lat, lng AS d_lng, purpose, category,
                       stop_id
                FROM ordered
                WHERE o_lat IS NOT NULL
            )
            SELECT * FROM combined ORDER BY t_start
        )
        TO '{out}' (FORMAT PARQUET)
        "#,
        input = quote_path(trajectory_path),
        out = quote_path(out)
    );
    duckdb::Connection::open_in_memory()?.execute_batch(&sql)?;
    Ok(())
}

fn legs_index_path(exp_id: &str, run: &experiments::Run) -> anyhow::Result<PathBuf> {
    cache().get_or_build_parquet(
        "timeline_legs",
        &["v3", exp_id, &run.run_id],
        &run.path,
        |out| build_legs_index(&run.path, out),
    )
}

fn moving_index_path(exp_id: &str, run: &experiments::Run) -> anyhow::Result<Option<PathBuf>> {
    let moving_path = run.moving_path();
    if !moving_path.exists() {
        return Ok(None);
    }
    Ok(Some(cache().get_or_build_parquet(
        "timeline_moving",
        &["v2", exp_id, &run.run_id],
        &moving_path,
        |out| build_moving_index(&moving_path, out),
    )?))
}

fn run_bbox(path: &FsPath) -> anyhow::Result<Option<Value>> {
    let conn = duckdb::Connection::open_in_memory()?;
    let sql = format!(
        "SELECT min(lat), max(lat), min(lng), max(lng) FROM read_parquet('{}')",
        quote_path(path)
    );
    let row = conn.query_row(&sql, [], |row| {
        Ok((
            row.get::<_, Option<f64>>(0)?,
            row.get::<_, Option<f64>>(1)?,
            row.get::<_, Option<f64>>(2)?,
            row.get::<_, Option<f64>>(3)?,
        ))
    })?;
    Ok(match row {
        (Some(min_lat), Some(max_lat), Some(min_lng), Some(max_lng)) => Some(
            json!({"min_lat": min_lat, "max_lat": max_lat, "min_lng": min_lng, "max_lng": max_lng}),
        ),
        _ => None,
    })
}

fn query_active_legs(
    legs_path: &FsPath,
    since: NaiveDateTime,
    until: NaiveDateTime,
    bbox: (f64, f64, f64, f64),
    max_agents: i64,
    moving_path: Option<&FsPath>,
    profiles_path: Option<&FsPath>,
) -> anyhow::Result<(Vec<Value>, bool)> {
    let min_lat = bbox.0;
    let min_lng = bbox.1;
    let max_lat = bbox.2;
    let max_lng = bbox.3;
    let sql = format!(
        r#"
        WITH candidates AS (
            SELECT DISTINCT uid
            FROM read_parquet('{path}')
            WHERE t_start <= ? AND t_end >= ?
              AND (
                    (d_lat BETWEEN ? AND ? AND d_lng BETWEEN ? AND ?)
                 OR (o_lat BETWEEN ? AND ? AND o_lng BETWEEN ? AND ?)
              )
            ORDER BY hash(uid)
            LIMIT ?
        )
        SELECT l.uid, l.kind, CAST(l.t_start AS VARCHAR), CAST(l.t_end AS VARCHAR),
               l.o_lat, l.o_lng, l.d_lat, l.d_lng, l.purpose, l.category, l.stop_id
        FROM read_parquet('{path}') l
        JOIN candidates c USING (uid)
        WHERE l.t_start <= ? AND l.t_end >= ?
        ORDER BY l.uid, l.t_start
        "#,
        path = quote_path(legs_path)
    );
    let conn = duckdb::Connection::open_in_memory()?;
    let mut stmt = conn.prepare(&sql)?;
    let mut rows = stmt.query(duckdb::params![
        until.to_string(),
        since.to_string(),
        min_lat,
        max_lat,
        min_lng,
        max_lng,
        min_lat,
        max_lat,
        min_lng,
        max_lng,
        max_agents,
        until.to_string(),
        since.to_string(),
    ])?;
    let mut out = Vec::new();
    let mut uids = HashSet::new();
    while let Some(row) = rows.next()? {
        let uid: i64 = row.get(0)?;
        uids.insert(uid);
        let kind: String = row.get(1)?;
        let stop_id = row.get::<_, Option<i64>>(10)?;
        out.push(json!({
            "uid": uid,
            "kind": kind,
            "t_start": row.get::<_, String>(2)?,
            "t_end": row.get::<_, String>(3)?,
            "o_lat": row.get::<_, f64>(4)?,
            "o_lng": row.get::<_, f64>(5)?,
            "d_lat": row.get::<_, f64>(6)?,
            "d_lng": row.get::<_, f64>(7)?,
            "purpose": row.get::<_, Option<String>>(8)?.unwrap_or_default(),
            "category": row.get::<_, Option<String>>(9)?,
            "stop_id": stop_id,
            "mode": if kind == "dwell" { "stay" } else { "car" },
        }));
    }
    if let Some(path) = moving_path {
        attach_waypoints(&mut out, path)?;
    }
    if let Some(path) = profiles_path.filter(|p| p.exists()) {
        attach_profile_character_fields(&mut out, path)?;
    }
    for segment in &mut out {
        if let Some(obj) = segment.as_object_mut() {
            obj.remove("stop_id");
        }
    }
    Ok((out, uids.len() as i64 >= max_agents))
}

fn normalize_character_gender(value: Option<&str>) -> &'static str {
    match value.unwrap_or("").trim().to_ascii_lowercase().as_str() {
        "female" | "f" | "woman" | "women" => "female",
        "male" | "m" | "man" | "men" => "man",
        _ => "unknown",
    }
}

fn job_matches(job: &str, keywords: &[&str]) -> bool {
    let raw = job.trim().to_ascii_lowercase();
    keywords.iter().any(|keyword| raw.contains(keyword))
}

fn profile_character_sprite(uid: i64, gender: &str, age: Option<i64>, job: &str) -> String {
    if gender == "unknown" {
        return "unknown".to_string();
    }
    if age.is_some_and(|age| age >= 65) {
        return if gender == "female" {
            "woman_3"
        } else {
            "men_4"
        }
        .to_string();
    }
    if job_matches(
        job,
        &[
            "construction",
            "machine",
            "operator",
            "craft",
            "repair",
            "transport",
            "agricultural",
        ],
    ) {
        return if gender == "female" {
            "woman_4"
        } else {
            "men_5"
        }
        .to_string();
    }
    if job_matches(
        job,
        &[
            "manager",
            "professional",
            "technician",
            "associate",
            "clerical",
        ],
    ) {
        return if gender == "female" {
            "woman_2"
        } else {
            "men_3"
        }
        .to_string();
    }
    if job_matches(job, &["service", "sales", "care", "health", "education"]) {
        return if gender == "female" {
            "woman_5"
        } else {
            "men_2"
        }
        .to_string();
    }
    if age.is_some_and(|age| age <= 25) {
        return if gender == "female" { "female" } else { "man" }.to_string();
    }
    let fallbacks = if gender == "female" {
        ["female", "woman_4", "woman_5"]
    } else {
        ["man", "men_2", "men_6"]
    };
    fallbacks[uid.rem_euclid(fallbacks.len() as i64) as usize].to_string()
}

fn attach_profile_character_fields(
    segments: &mut [Value],
    profiles_path: &FsPath,
) -> anyhow::Result<()> {
    let uids: BTreeSet<i64> = segments
        .iter()
        .filter_map(|s| s.get("uid").and_then(Value::as_i64))
        .collect();
    if uids.is_empty() {
        return Ok(());
    }
    let columns = parquet_columns(profiles_path)?;
    if !columns.contains("uid") || !columns.contains("gender") {
        return Ok(());
    }
    let values = uids
        .iter()
        .map(|uid| format!("({uid})"))
        .collect::<Vec<_>>()
        .join(", ");
    let age_expr = if columns.contains("age") {
        "p.age"
    } else {
        "NULL AS age"
    };
    let job_expr = if columns.contains("job") {
        "p.job"
    } else {
        "NULL AS job"
    };
    let sql = format!(
        r#"
        SELECT p.uid, CAST(p.gender AS VARCHAR), TRY_CAST({age_expr} AS BIGINT), CAST({job_expr} AS VARCHAR)
        FROM read_parquet('{path}') p
        JOIN (VALUES {values}) AS requested(uid) USING (uid)
        "#,
        path = quote_path(profiles_path)
    );
    let conn = duckdb::Connection::open_in_memory()?;
    let mut stmt = conn.prepare(&sql)?;
    let mut rows = stmt.query([])?;
    let mut by_uid = HashMap::<i64, (String, String)>::new();
    while let Some(row) = rows.next()? {
        let uid: i64 = row.get(0)?;
        let gender_raw = row.get::<_, Option<String>>(1)?;
        let gender = normalize_character_gender(gender_raw.as_deref()).to_string();
        let age = row.get::<_, Option<i64>>(2)?;
        let job = row.get::<_, Option<String>>(3)?.unwrap_or_default();
        let sprite = profile_character_sprite(uid, &gender, age, &job);
        by_uid.insert(uid, (gender, sprite));
    }
    for segment in segments {
        let uid = segment.get("uid").and_then(Value::as_i64).unwrap_or(0);
        let (gender, sprite) = by_uid
            .get(&uid)
            .cloned()
            .unwrap_or_else(|| ("unknown".to_string(), "unknown".to_string()));
        if let Some(obj) = segment.as_object_mut() {
            obj.insert("gender".to_string(), json!(gender));
            obj.insert("character_sprite".to_string(), json!(sprite));
        }
    }
    Ok(())
}

fn attach_waypoints(segments: &mut [Value], moving_path: &FsPath) -> anyhow::Result<()> {
    let pairs: BTreeSet<(i64, i64)> = segments
        .iter()
        .filter(|s| s.get("kind").and_then(Value::as_str) == Some("leg"))
        .filter_map(|s| Some((s.get("uid")?.as_i64()?, s.get("stop_id")?.as_i64()?)))
        .collect();
    if pairs.is_empty() {
        return Ok(());
    }
    let values = pairs
        .iter()
        .map(|(uid, stop_id)| format!("({uid}, {stop_id})"))
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!(
        r#"
        SELECT m.uid, m.stop_id, m.lat, m.lng, CAST(m.t AS VARCHAR), m.mode
        FROM read_parquet('{path}') m
        JOIN (VALUES {values}) AS requested(uid, stop_id)
          ON m.uid = requested.uid AND m.stop_id = requested.stop_id
        ORDER BY m.uid, m.stop_id, m.seq
        "#,
        path = quote_path(moving_path)
    );
    let conn = duckdb::Connection::open_in_memory()?;
    let mut stmt = conn.prepare(&sql)?;
    let mut rows = stmt.query([])?;
    let mut waypoints = HashMap::<(i64, i64), Vec<Value>>::new();
    let mut modes = HashMap::<(i64, i64), String>::new();
    while let Some(row) = rows.next()? {
        let key = (row.get::<_, i64>(0)?, row.get::<_, i64>(1)?);
        waypoints.entry(key).or_default().push(json!({
            "lat": row.get::<_, f64>(2)?,
            "lng": row.get::<_, f64>(3)?,
            "t": row.get::<_, Option<String>>(4)?.unwrap_or_default(),
        }));
        modes.insert(
            key,
            row.get::<_, Option<String>>(5)?
                .unwrap_or_else(|| "car".to_string()),
        );
    }
    for segment in segments {
        if segment.get("kind").and_then(Value::as_str) != Some("leg") {
            continue;
        }
        let Some(key) = segment
            .get("uid")
            .and_then(Value::as_i64)
            .zip(segment.get("stop_id").and_then(Value::as_i64))
        else {
            continue;
        };
        if let Some(obj) = segment.as_object_mut() {
            obj.insert(
                "waypoints".to_string(),
                waypoints
                    .get(&key)
                    .cloned()
                    .map(Value::Array)
                    .unwrap_or(Value::Null),
            );
            obj.insert(
                "mode".to_string(),
                json!(
                    modes
                        .get(&key)
                        .cloned()
                        .unwrap_or_else(|| "car".to_string())
                ),
            );
        }
    }
    Ok(())
}

fn query_profile(path: Option<&FsPath>, uid: i64) -> anyhow::Result<Option<Value>> {
    let Some(path) = path.filter(|p| p.exists()) else {
        return Ok(None);
    };
    let col_names = crate::datasource::parquet_columns(path)?;
    if !col_names.iter().any(|name| name == "uid") {
        return Ok(None);
    }
    let select = col_names
        .iter()
        .map(|name| format!(r#""{}""#, name.replace('"', "\"\"")))
        .collect::<Vec<_>>()
        .join(", ");
    let conn = duckdb::Connection::open_in_memory()?;
    let sql = format!(
        "SELECT {select} FROM read_parquet('{}') WHERE uid = ? LIMIT 1",
        quote_path(path),
    );
    let mut stmt = conn.prepare(&sql)?;
    let mut rows = stmt.query(duckdb::params![uid])?;
    let Some(row) = rows.next()? else {
        return Ok(None);
    };
    let mut obj = serde_json::Map::new();
    for (idx, name) in col_names.iter().enumerate() {
        let value: duckdb::types::Value = row.get(idx)?;
        obj.insert(name.clone(), duck_value_to_json(value));
    }
    Ok(Some(Value::Object(obj)))
}

fn duck_value_to_json(value: duckdb::types::Value) -> Value {
    use duckdb::types::Value as Dv;
    match value {
        Dv::Null => Value::Null,
        Dv::Boolean(v) => json!(v),
        Dv::TinyInt(v) => json!(v),
        Dv::SmallInt(v) => json!(v),
        Dv::Int(v) => json!(v),
        Dv::BigInt(v) => json!(v),
        Dv::UTinyInt(v) => json!(v),
        Dv::USmallInt(v) => json!(v),
        Dv::UInt(v) => json!(v),
        Dv::UBigInt(v) => json!(v),
        Dv::Float(v) => json!(v),
        Dv::Double(v) => json!(v),
        Dv::Text(v) => json!(v),
        other => json!(format!("{other:?}")),
    }
}

fn activity_fields(activity_id: Option<i64>) -> Value {
    let Some(id) = activity_id else {
        return json!({"activity_name": null, "activity_description": null});
    };
    match crate::settings::catalog::by_id(id) {
        Some(def) => json!({"activity_name": def.name, "activity_description": def.description}),
        None => json!({"activity_name": null, "activity_description": null}),
    }
}

fn query_agent_activities(path: &FsPath, uid: i64) -> anyhow::Result<HashMap<i64, Vec<Value>>> {
    if !path.exists() {
        return Ok(HashMap::new());
    }
    let columns = parquet_columns(path)?;
    if !columns.contains("uid")
        || !columns.contains("stop_id")
        || !columns.contains("activity")
        || !columns.contains("arrival")
        || !columns.contains("departure")
    {
        return Ok(HashMap::new());
    }
    let seq_expr = if columns.contains("seq") {
        "seq"
    } else {
        "0 AS seq"
    };
    let sql = format!(
        r#"SELECT stop_id, CAST(arrival AS VARCHAR), CAST(departure AS VARCHAR),
                  TRY_CAST(activity AS BIGINT), {seq_expr},
                  date_diff('second', arrival, departure) / 60.0
           FROM read_parquet('{path}') WHERE uid = ? ORDER BY stop_id, seq"#,
        path = quote_path(path)
    );
    let conn = duckdb::Connection::open_in_memory()?;
    let mut stmt = conn.prepare(&sql)?;
    let mut rows = stmt.query(duckdb::params![uid])?;
    let mut by_stop = HashMap::<i64, Vec<Value>>::new();
    while let Some(row) = rows.next()? {
        let stop_id = row.get::<_, i64>(0)?;
        let activity = row.get::<_, Option<i64>>(3)?;
        let fields = activity_fields(activity);
        by_stop.entry(stop_id).or_default().push(json!({
            "arrival": row.get::<_, String>(1)?,
            "departure": row.get::<_, String>(2)?,
            "purpose": null,
            "category": null,
            "activity": activity,
            "activity_name": fields["activity_name"].clone(),
            "activity_description": fields["activity_description"].clone(),
            "trip_duration_minutes": 0.0,
            "dwell_minutes": row.get::<_, Option<f64>>(5)?.unwrap_or(0.0),
        }));
    }
    Ok(by_stop)
}

fn query_agent_trips(
    path: &FsPath,
    activities_path: &FsPath,
    uid: i64,
) -> anyhow::Result<Vec<Value>> {
    let columns = parquet_columns(path)?;
    let category_expr = if columns.contains("category") {
        "category"
    } else {
        "NULL::VARCHAR AS category"
    };
    let stop_id_expr = if columns.contains("stop_id") {
        "stop_id"
    } else {
        "NULL::BIGINT AS stop_id"
    };
    let sidecar_activities = query_agent_activities(activities_path, uid)?;
    let sql = format!(
        r#"SELECT CAST(arrival AS VARCHAR), CAST(departure AS VARCHAR), lat, lng, purpose,
                  {category_expr}, {stop_id_expr}, trip_duration_minutes, dwell_minutes
           FROM read_parquet('{path}') WHERE uid = $uid ORDER BY arrival"#,
        path = quote_path(path)
    );
    let conn = duckdb::Connection::open_in_memory()?;
    let mut stmt = conn.prepare(&sql)?;
    let mut rows = stmt.query(duckdb::params![uid])?;
    let mut out = Vec::new();
    while let Some(row) = rows.next()? {
        let stop_id = row.get::<_, Option<i64>>(6)?;
        let activities = stop_id
            .and_then(|id| sidecar_activities.get(&id).cloned())
            .unwrap_or_else(|| {
                vec![json!({
                    "arrival": row.get::<_, String>(0).unwrap_or_default(),
                    "departure": row.get::<_, String>(1).unwrap_or_default(),
                    "purpose": row.get::<_, Option<String>>(4).ok().flatten().unwrap_or_default(),
                    "category": row.get::<_, Option<String>>(5).ok().flatten(),
                    "activity": null,
                    "activity_name": null,
                    "activity_description": null,
                    "trip_duration_minutes": row.get::<_, Option<f64>>(7).ok().flatten().unwrap_or(0.0),
                    "dwell_minutes": row.get::<_, Option<f64>>(8).ok().flatten().unwrap_or(0.0),
                })]
            });
        out.push(json!({
            "arrival": row.get::<_, String>(0)?,
            "departure": row.get::<_, String>(1)?,
            "lat": row.get::<_, f64>(2)?,
            "lng": row.get::<_, f64>(3)?,
            "purpose": row.get::<_, Option<String>>(4)?.unwrap_or_default(),
            "category": row.get::<_, Option<String>>(5)?,
            "trip_duration_minutes": row.get::<_, Option<f64>>(7)?.unwrap_or(0.0),
            "dwell_minutes": row.get::<_, Option<f64>>(8)?.unwrap_or(0.0),
            "activities": activities,
        }));
    }
    Ok(out)
}

fn diary_cache_variant_path(base: &FsPath, variant: &str) -> PathBuf {
    if variant.is_empty() {
        return base.to_path_buf();
    }
    let stem = base.file_stem().unwrap_or_default().to_string_lossy();
    let ext = base
        .extension()
        .map(|e| format!(".{}", e.to_string_lossy()))
        .unwrap_or_default();
    base.with_file_name(format!("{stem}_{variant}{ext}"))
}

fn load_diary_cache(
    base_path: Option<&FsPath>,
    day_types: &[String],
) -> HashMap<(String, String), Value> {
    let Some(base_path) = base_path else {
        return HashMap::new();
    };
    let mut paths = vec![base_path.to_path_buf()];
    for day_type in day_types {
        paths.push(diary_cache_variant_path(base_path, day_type));
    }
    let mut out = HashMap::new();
    for path in paths {
        if !path.exists() {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Ok(payload) = serde_json::from_str::<Value>(&text) else {
            continue;
        };
        let day_type = path
            .file_stem()
            .and_then(|s| s.to_str())
            .and_then(|stem| stem.rsplit_once('_').map(|(_, suffix)| suffix.to_string()));
        let Some(diaries) = payload.get("diaries").and_then(Value::as_array) else {
            continue;
        };
        for diary in diaries {
            let Some(diary_id) = diary.get("diary_id").and_then(Value::as_str) else {
                continue;
            };
            if let Some(day_type) = &day_type {
                out.insert((day_type.clone(), diary_id.to_string()), diary.clone());
            }
            out.entry(("".to_string(), diary_id.to_string()))
                .or_insert_with(|| diary.clone());
        }
    }
    out
}

fn query_agent_crp_rows(
    path: &FsPath,
    diary_cache_path: Option<&FsPath>,
    uid: i64,
) -> anyhow::Result<(Option<f64>, Option<f64>, Vec<Value>)> {
    if !path.exists() {
        return Ok((None, None, Vec::new()));
    }
    let agent = uid - 1;
    let sql = format!(
        "SELECT diary_id, day_type, sim, usage_count, T_a, alpha_a FROM read_parquet('{}') WHERE agent = $uid ORDER BY usage_count DESC",
        quote_path(path)
    );
    let conn = duckdb::Connection::open_in_memory()?;
    let mut stmt = conn.prepare(&sql)?;
    let mut rows = stmt.query(duckdb::params![agent])?;
    let mut diaries = Vec::new();
    let mut t_a = None;
    let mut alpha_a = None;
    while let Some(row) = rows.next()? {
        if t_a.is_none() {
            t_a = row.get::<_, Option<f64>>(4)?;
            alpha_a = row.get::<_, Option<f64>>(5)?;
        }
        diaries.push(json!({
            "diary_id": row.get::<_, Option<String>>(0)?.unwrap_or_default(),
            "day_type": row.get::<_, Option<String>>(1)?.unwrap_or_default(),
            "sim": row.get::<_, Option<f64>>(2)?.unwrap_or(0.0),
            "usage_count": row.get::<_, Option<i64>>(3)?.unwrap_or(0),
        }));
    }
    let day_types: Vec<String> = diaries
        .iter()
        .filter_map(|d| {
            d.get("day_type")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .collect();
    let cache = load_diary_cache(diary_cache_path, &day_types);
    for diary in &mut diaries {
        let day_type = diary
            .get("day_type")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let diary_id = diary
            .get("diary_id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let cached = cache
            .get(&(day_type, diary_id.clone()))
            .or_else(|| cache.get(&("".to_string(), diary_id)));
        if let (Some(obj), Some(cached)) = (diary.as_object_mut(), cached) {
            if let Some(description) = cached.get("description") {
                obj.insert("description".to_string(), description.clone());
            }
            if let Some(episodes) = cached.get("episodes") {
                obj.insert("episodes".to_string(), episodes.clone());
            }
        }
    }
    Ok((t_a, alpha_a, diaries))
}

fn query_encounter_counts(path: &FsPath, uid: i64) -> anyhow::Result<HashMap<i64, i64>> {
    if !path.exists() {
        return Ok(HashMap::new());
    }
    let sql = format!(
        "SELECT CASE WHEN agent = $uid THEN contact ELSE agent END AS contact_uid, count(*) FROM read_parquet('{}') WHERE agent = $uid OR contact = $uid GROUP BY 1",
        quote_path(path)
    );
    let conn = duckdb::Connection::open_in_memory()?;
    let mut stmt = conn.prepare(&sql)?;
    let mut rows = stmt.query(duckdb::params![uid])?;
    let mut out = HashMap::new();
    while let Some(row) = rows.next()? {
        out.insert(row.get::<_, i64>(0)?, row.get::<_, i64>(1)?);
    }
    Ok(out)
}

pub async fn timeline_meta_route(
    Path(exp_id): Path<String>,
    Query(q): Query<TimelineRunQuery>,
) -> ApiResult<Value> {
    let (exp, run) = selected(&exp_id, q.run.as_deref())?;
    let summary = run_summary(&run.path).map_err(|e| ApiError::internal(e.to_string()))?;
    let bbox = run_bbox(&run.path).map_err(|e| ApiError::internal(e.to_string()))?;
    let payload = json!({
        "run_id": run.run_id,
        "date_start": summary.date_start,
        "date_end": summary.date_end,
        "bbox": bbox,
        "agents_total": summary.uids,
        "has_profiles": exp.profiles_path.as_ref().is_some_and(|p| p.exists()),
        "has_encounters": run.encounters_path().exists(),
        "car_speed_kmh": exp.params.car_speed_kmh,
    });
    Ok(ApiResponse::new(payload))
}

pub async fn timeline_legs_route(
    Path(exp_id): Path<String>,
    Query(q): Query<TimelineLegsQuery>,
) -> ApiResult<Value> {
    let (exp, run) = selected(&exp_id, q.run.as_deref())?;
    if q.max_agents < 1 || q.max_agents > 5000 {
        return Err(ApiError::unprocessable(
            "max_agents must be between 1 and 5000",
        ));
    }
    let since = parse_datetime(&q.since)?;
    let until = parse_datetime(&q.until)?;
    if until <= since {
        return Err(ApiError::unprocessable("until must be after since"));
    }
    if until - since > chrono::Duration::hours(6) {
        return Err(ApiError::unprocessable(
            "requested window too large (max 6h of sim time per request)",
        ));
    }
    let legs_path =
        legs_index_path(&exp_id, &run).map_err(|e| ApiError::internal(e.to_string()))?;
    let moving_path =
        moving_index_path(&exp_id, &run).map_err(|e| ApiError::internal(e.to_string()))?;
    let (segments, truncated) = query_active_legs(
        &legs_path,
        since,
        until,
        (q.min_lat, q.min_lng, q.max_lat, q.max_lng),
        q.max_agents,
        moving_path.as_deref(),
        exp.profiles_path.as_deref(),
    )
    .map_err(|e| ApiError::internal(e.to_string()))?;
    let agent_count = segments
        .iter()
        .filter_map(|s| s.get("uid").and_then(Value::as_i64))
        .collect::<HashSet<_>>()
        .len();
    let payload = json!({
        "run_id": run.run_id,
        "since": q.since,
        "until": q.until,
        "agent_count": agent_count,
        "truncated": truncated,
        "segments": segments,
    });
    Ok(ApiResponse::new(payload))
}

pub async fn timeline_agent_route(
    Path((exp_id, uid)): Path<(String, i64)>,
    Query(q): Query<TimelineRunQuery>,
) -> ApiResult<Value> {
    let (exp, run) = selected(&exp_id, q.run.as_deref())?;
    let mut warnings = Vec::new();
    let profile = query_profile(exp.profiles_path.as_deref(), uid)
        .map_err(|e| ApiError::internal(e.to_string()))?;
    if profile.is_none() {
        warnings.push("no agent profile available for this uid".to_string());
    }
    let trips = query_agent_trips(&run.path, &run.activities_path(), uid)
        .map_err(|e| ApiError::internal(e.to_string()))?;
    if !run.encounters_path().exists() {
        warnings.push("no encounters data available for this experiment".to_string());
    }
    Ok(ApiResponse::new(json!({
        "uid": uid,
        "run_id": run.run_id,
        "profile": profile,
        "narrative": null,
        "trips": trips,
        "encounters": [],
        "warnings": warnings,
    })))
}

pub async fn timeline_agent_crp_route(
    Path((exp_id, uid)): Path<(String, i64)>,
    Query(q): Query<TimelineRunQuery>,
) -> ApiResult<Value> {
    let (exp, run) = selected(&exp_id, q.run.as_deref())?;
    let mut warnings = Vec::new();
    let (t_a, alpha_a, diaries) =
        query_agent_crp_rows(&run.crp_path(), exp.diary_cache_path.as_deref(), uid)
            .map_err(|e| ApiError::internal(e.to_string()))?;
    if !run.crp_path().exists() {
        warnings.push("no ddCRP diary selection data available for this run".to_string());
    } else if diaries.is_empty() {
        warnings.push("uid not found in ddCRP diary selection data".to_string());
    }
    Ok(ApiResponse::new(json!({
        "uid": uid,
        "run_id": run.run_id,
        "T_a": t_a,
        "alpha_a": alpha_a,
        "diaries": diaries,
        "warnings": warnings,
    })))
}

pub async fn timeline_agent_social_route(
    Path((exp_id, uid)): Path<(String, i64)>,
    Query(q): Query<TimelineRunQuery>,
) -> ApiResult<Value> {
    let (exp, run) = selected(&exp_id, q.run.as_deref())?;
    let mut warnings = Vec::new();
    let mut parameters = json!({
        "degree": 0,
        "total_social_strength": 0.0,
        "social_graph_k": exp.params.social_graph_k,
        "layout": null,
        "kind": null,
        "directed": null,
        "rho": exp.params.rho,
        "gamma": exp.params.gamma,
        "alpha": exp.params.alpha,
        "dt_update_mob_sim_hours": exp.params.dt_update_mob_sim_hours,
        "indipendency_window_hours": exp.params.indipendency_window_hours,
    });
    let mut friends = Vec::new();
    if run.social_network_path().exists() {
        let data: Value = serde_json::from_slice(
            &std::fs::read(run.social_network_path())
                .map_err(|e| ApiError::internal(e.to_string()))?,
        )
        .map_err(|e| ApiError::internal(e.to_string()))?;
        if let Some(metadata) = data.get("metadata").and_then(Value::as_object) {
            for key in ["social_graph_k", "layout", "kind", "directed"] {
                if let Some(value) = metadata.get(key) {
                    parameters[key] = value.clone();
                }
            }
        }
        let agent_idx = uid - 1;
        let mut reverse = HashSet::new();
        if let Some(edges) = data.get("edges").and_then(Value::as_array) {
            for edge in edges {
                let Some(row) = edge.as_array() else { continue };
                if row.len() < 2 {
                    continue;
                }
                let source = row[0].as_i64().unwrap_or(-1);
                let target = row[1].as_i64().unwrap_or(-1);
                if target == agent_idx {
                    reverse.insert(source);
                }
            }
            for edge in edges {
                let Some(row) = edge.as_array() else { continue };
                if row.len() < 2 {
                    continue;
                }
                let source = row[0].as_i64().unwrap_or(-1);
                let target = row[1].as_i64().unwrap_or(-1);
                if source != agent_idx {
                    continue;
                }
                let weight = row.get(2).and_then(Value::as_f64).unwrap_or(1.0);
                friends.push(json!({
                    "uid": target + 1,
                    "name": null,
                    "profile": null,
                    "social_strength": weight,
                    "embedding_similarity": weight,
                    "encounter_count": 0,
                    "reciprocated": reverse.contains(&target),
                }));
            }
        }
        friends.sort_by(|a, b| {
            let aw = a
                .get("social_strength")
                .and_then(Value::as_f64)
                .unwrap_or(0.0);
            let bw = b
                .get("social_strength")
                .and_then(Value::as_f64)
                .unwrap_or(0.0);
            bw.total_cmp(&aw)
        });
        parameters["degree"] = json!(friends.len());
        parameters["total_social_strength"] = json!(
            friends
                .iter()
                .map(|f| f
                    .get("social_strength")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0))
                .sum::<f64>()
        );
    } else {
        warnings.push("no social network sidecar available for this run".to_string());
    }
    let encounter_counts = query_encounter_counts(&run.encounters_path(), uid)
        .map_err(|e| ApiError::internal(e.to_string()))?;
    for friend in &mut friends {
        let friend_uid = friend.get("uid").and_then(Value::as_i64).unwrap_or(0);
        friend["encounter_count"] = json!(encounter_counts.get(&friend_uid).copied().unwrap_or(0));
        if let Some(profile) = query_profile(exp.profiles_path.as_deref(), friend_uid)
            .map_err(|e| ApiError::internal(e.to_string()))?
        {
            friend["name"] = profile.get("name").cloned().unwrap_or(Value::Null);
            friend["profile"] = profile;
        }
    }
    Ok(ApiResponse::new(json!({
        "uid": uid,
        "run_id": run.run_id,
        "parameters": parameters,
        "friends": friends,
        "warnings": warnings,
    })))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_character_sprite_uses_profile_fields() {
        assert_eq!(
            profile_character_sprite(7, "female", Some(70), "retired"),
            "woman_3"
        );
        assert_eq!(
            profile_character_sprite(8, "man", Some(35), "construction worker"),
            "men_5"
        );
        assert_eq!(profile_character_sprite(1, "unknown", None, ""), "unknown");
    }

    #[test]
    fn normalize_character_gender_matches_python_aliases() {
        assert_eq!(normalize_character_gender(Some("F")), "female");
        assert_eq!(normalize_character_gender(Some("men")), "man");
        assert_eq!(normalize_character_gender(Some("not supplied")), "unknown");
    }

    #[test]
    fn query_agent_trips_uses_activities_sidecar() {
        let dir = std::env::temp_dir().join(format!(
            "citybehavex-timeline-activities-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let traj = dir.join("traj.parquet");
        let activities = dir.join("traj_activities.parquet");
        let conn = duckdb::Connection::open_in_memory().unwrap();
        conn.execute_batch(&format!(
            r#"
            COPY (
                SELECT 7::BIGINT AS uid, 42::BIGINT AS stop_id,
                       TIMESTAMP '2026-01-01 00:00:00' AS arrival,
                       TIMESTAMP '2026-01-01 01:00:00' AS departure,
                       TIMESTAMP '2026-01-01 00:00:00' AS datetime,
                       48.0::DOUBLE AS lat, 2.0::DOUBLE AS lng,
                       0.0::DOUBLE AS trip_duration_minutes,
                       60.0::DOUBLE AS dwell_minutes,
                       'HOME'::VARCHAR AS purpose,
                       NULL::VARCHAR AS category
            ) TO '{}' (FORMAT PARQUET);
            COPY (
                SELECT 7::BIGINT AS uid, 42::BIGINT AS stop_id, 0::INTEGER AS seq,
                       13::BIGINT AS activity,
                       TIMESTAMP '2026-01-01 00:00:00' AS arrival,
                       TIMESTAMP '2026-01-01 00:15:00' AS departure,
                       1::BIGINT AS block_id
                UNION ALL
                SELECT 7::BIGINT, 42::BIGINT, 1::INTEGER, 0::BIGINT,
                       TIMESTAMP '2026-01-01 00:15:00',
                       TIMESTAMP '2026-01-01 01:00:00',
                       1::BIGINT
            ) TO '{}' (FORMAT PARQUET);
            "#,
            crate::datasource::quote_path(&traj),
            crate::datasource::quote_path(&activities)
        ))
        .unwrap();

        let trips = query_agent_trips(&traj, &activities, 7).unwrap();
        let acts = trips[0].get("activities").unwrap().as_array().unwrap();
        assert_eq!(acts.len(), 2);
        assert_eq!(acts[0]["activity"], json!(13));
        assert_eq!(acts[0]["activity_name"], json!("ikidcare"));
        assert_eq!(acts[0]["dwell_minutes"], json!(15.0));
        assert_eq!(acts[1]["activity_name"], json!("sleep"));

        let _ = std::fs::remove_file(traj);
        let _ = std::fs::remove_file(activities);
        let _ = std::fs::remove_dir(dir);
    }

    #[test]
    fn query_agent_crp_rows_enriches_diary_episodes() {
        let dir =
            std::env::temp_dir().join(format!("citybehavex-timeline-crp-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let crp = dir.join("run_crp.parquet");
        let cache_base = dir.join("validated_diaries.json");
        let cache_weekday = dir.join("validated_diaries_weekday.json");
        let conn = duckdb::Connection::open_in_memory().unwrap();
        conn.execute_batch(&format!(
            r#"
            COPY (
                SELECT 6::BIGINT AS agent,
                       'routine-017'::VARCHAR AS diary_id,
                       'weekday'::VARCHAR AS day_type,
                       0.4::DOUBLE AS sim,
                       3::BIGINT AS usage_count,
                       0.2::DOUBLE AS T_a,
                       0.1::DOUBLE AS alpha_a
            ) TO '{}' (FORMAT PARQUET);
            "#,
            crate::datasource::quote_path(&crp)
        ))
        .unwrap();
        std::fs::write(
            &cache_weekday,
            r#"{"diaries":[{"diary_id":"routine-017","episodes":[{"start":"00:00","end":"07:00","purpose":"HOME"},{"start":"07:00","end":"09:00","purpose":"WORK"}]}]}"#,
        )
        .unwrap();

        let (_t, _alpha, diaries) = query_agent_crp_rows(&crp, Some(&cache_base), 7).unwrap();
        assert_eq!(diaries[0]["diary_id"], json!("routine-017"));
        assert_eq!(diaries[0]["episodes"][0]["purpose"], json!("HOME"));
        assert_eq!(diaries[0]["episodes"][1]["start"], json!("07:00"));

        let _ = std::fs::remove_file(crp);
        let _ = std::fs::remove_file(cache_weekday);
        let _ = std::fs::remove_dir(dir);
    }
}
