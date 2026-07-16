//! Export a static, GitHub Pages-friendly copy of the web demo data.
//!
//! Rust successor to `scripts/export_static_web_demo.py`, which called
//! `web/backend/app`'s payload builders directly (in-process, not over
//! HTTP) to materialize the same endpoint-shaped JSON under
//! `web/frontend/public/demo-data` -- ported here the same way, calling
//! `citybehavex_web`'s library functions directly rather than the Python
//! ones, now that `web/backend/` has been retired.

use citybehavex_web::comparison::network_validation::{self, ObservedGraph};
use citybehavex_web::config::repo_root;
use citybehavex_web::datasource::run_summary;
use citybehavex_web::experiments::{Experiment, Run, get_experiment};
use citybehavex_web::home_work::{self, DemoFilter};
use citybehavex_web::payload::{self, ComparisonContext};
use citybehavex_web::routes::timeline as timeline_routes;

use chrono::{NaiveDateTime, TimeDelta};
use serde::Deserialize;
use serde_json::{Value, json};
use std::path::Path;

const DEFAULT_SECTIONS: &[(&str, &str)] = &[
    ("micro-activity", "all"),
    ("time-use", "all"),
    ("activity", "all"),
    ("motifs", "all"),
    ("profiles", "all"),
    ("social-network", "all"),
    ("metrics", "all"),
    ("transport-spatial", "all"),
    ("distributions", "all"),
    ("mobility-laws", "all"),
    ("stvd", "all"),
];

#[derive(Debug, Deserialize)]
struct ManifestExperiment {
    id: String,
    run_id: String,
    label: Option<String>,
    #[serde(default)]
    allow_observed: bool,
    sample_agents: Option<i64>,
    observed_sample_agents: Option<i64>,
    expected_agents: Option<i64>,
    timeline_days: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct Manifest {
    #[serde(default = "default_output_dir")]
    output_dir: String,
    #[serde(default = "default_chunk_hours")]
    timeline_chunk_hours: i64,
    #[serde(default = "default_max_agents")]
    timeline_max_agents: i64,
    #[serde(default = "default_true")]
    export_agent_details: bool,
    chart_sections: Option<Vec<(String, String)>>,
    experiments: Vec<ManifestExperiment>,
}

fn default_output_dir() -> String {
    "web/frontend/public/demo-data".to_string()
}
fn default_chunk_hours() -> i64 {
    6
}
fn default_max_agents() -> i64 {
    500
}
fn default_true() -> bool {
    true
}

fn write_json(path: &Path, data: &Value) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_vec(data)?)?;
    Ok(())
}

fn wrapped(data: Value) -> Value {
    json!({"data": data})
}

fn quote_path(path: &Path) -> String {
    path.display().to_string().replace('\'', "''")
}

fn duckdb_parquet_columns(path: &Path) -> anyhow::Result<Vec<String>> {
    let conn = duckdb::Connection::open_in_memory()?;
    let mut stmt = conn.prepare(&format!(
        "SELECT name FROM parquet_schema('{}')",
        quote_path(path)
    ))?;
    let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
    let mut out = Vec::new();
    for row in rows {
        let name = row?;
        if name != "schema" && name != "duckdb_schema" {
            out.push(name);
        }
    }
    Ok(out)
}

fn copy_user_sample(
    src: &Path,
    dst: &Path,
    uid_col: &str,
    max_uid: i64,
    zero_based: bool,
) -> anyhow::Result<()> {
    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let (lower, upper) = if zero_based {
        (0, max_uid - 1)
    } else {
        (1, max_uid)
    };
    let conn = duckdb::Connection::open_in_memory()?;
    conn.execute_batch(&format!(
        r#"
        COPY (
            SELECT * FROM read_parquet('{src}')
            WHERE "{uid_col}" BETWEEN {lower} AND {upper}
        ) TO '{dst}' (FORMAT PARQUET)
        "#,
        src = quote_path(src),
        dst = quote_path(dst),
    ))?;
    Ok(())
}

fn copy_encounter_sample(src: &Path, dst: &Path, max_uid: i64) -> anyhow::Result<()> {
    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let conn = duckdb::Connection::open_in_memory()?;
    conn.execute_batch(&format!(
        r#"
        COPY (
            SELECT * FROM read_parquet('{src}')
            WHERE agent BETWEEN 1 AND {max_uid} AND contact BETWEEN 1 AND {max_uid}
        ) TO '{dst}' (FORMAT PARQUET)
        "#,
        src = quote_path(src),
        dst = quote_path(dst),
    ))?;
    Ok(())
}

fn copy_social_sample(src: &Path, dst: &Path, max_uid: i64) -> anyhow::Result<()> {
    let payload: Value = serde_json::from_slice(&std::fs::read(src)?)?;
    let nodes: Vec<Value> = payload
        .get("nodes")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter(|node| {
            node.as_array()
                .is_some_and(|arr| arr.len() >= 4 && arr[3].as_i64().unwrap_or(i64::MAX) <= max_uid)
        })
        .collect();
    let kept_zero_based: std::collections::HashSet<i64> = nodes
        .iter()
        .filter_map(|node| node.as_array().and_then(|arr| arr[3].as_i64()).map(|v| v - 1))
        .collect();
    let edges: Vec<Value> = payload
        .get("edges")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter(|edge| {
            edge.as_array().is_some_and(|arr| {
                arr.len() >= 2
                    && arr[0].as_i64().is_some_and(|v| kept_zero_based.contains(&v))
                    && arr[1].as_i64().is_some_and(|v| kept_zero_based.contains(&v))
            })
        })
        .collect();
    let mut degrees = vec![0i64; max_uid as usize];
    for edge in &edges {
        let arr = edge.as_array().unwrap();
        let source = arr[0].as_i64().unwrap_or(-1);
        let target = arr[1].as_i64().unwrap_or(-1);
        if source >= 0 && (source as usize) < degrees.len() {
            degrees[source as usize] += 1;
        }
        if target >= 0 && (target as usize) < degrees.len() {
            degrees[target as usize] += 1;
        }
    }
    let mut sampled = payload;
    sampled["node_count"] = json!(nodes.len());
    sampled["edge_count"] = json!(edges.len());
    sampled["nodes"] = json!(nodes);
    sampled["edges"] = json!(edges);
    sampled["degrees"] = json!(degrees);
    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(dst, serde_json::to_vec(&sampled)?)?;
    Ok(())
}

fn sample_run(experiment: &Experiment, max_uid: i64) -> anyhow::Result<Experiment> {
    let selected = experiment.runs.first().expect("at least one run selected");
    let sample_dir = repo_root()
        .join("data")
        .join("static_demo_samples")
        .join(&experiment.id)
        .join(&selected.run_id)
        .join(format!("first_{max_uid}"));
    let stem = selected.path.file_stem().unwrap_or_default().to_string_lossy();
    let ext = selected
        .path
        .extension()
        .map(|e| e.to_string_lossy().to_string())
        .unwrap_or_default();
    let sample_path = sample_dir.join(format!("{stem}_first{max_uid}.{ext}"));
    println!(
        "[{}] materializing first {max_uid} users -> {}",
        experiment.id,
        sample_path.display()
    );
    copy_user_sample(&selected.path, &sample_path, "uid", max_uid, false)?;

    let sibling = |suffix: &str| sample_path.with_file_name(format!("{stem}_first{max_uid}{suffix}.{ext}"));
    for (source, target, uid_col, zero_based) in [
        (selected.activities_path(), sibling("_activities"), "uid", false),
        (selected.moving_path(), sibling("_moving"), "uid", false),
        (selected.crp_path(), sibling("_crp"), "agent", true),
    ] {
        if source.exists() {
            copy_user_sample(&source, &target, uid_col, max_uid, zero_based)?;
        }
    }
    if selected.encounters_path().exists() {
        copy_encounter_sample(
            &selected.encounters_path(),
            &sample_path.with_file_name(format!("{stem}_first{max_uid}_encounters.{ext}")),
            max_uid,
        )?;
    }
    if selected.social_network_path().exists() {
        copy_social_sample(
            &selected.social_network_path(),
            &sample_path.with_file_name(format!("{stem}_first{max_uid}_social_network.json")),
            max_uid,
        )?;
    }

    let mut sampled_profiles_path = experiment.profiles_path.clone();
    if let Some(profiles_path) = &experiment.profiles_path {
        if profiles_path.exists() && duckdb_parquet_columns(profiles_path)?.iter().any(|c| c == "uid") {
            let p_stem = profiles_path.file_stem().unwrap_or_default().to_string_lossy();
            let p_ext = profiles_path
                .extension()
                .map(|e| e.to_string_lossy().to_string())
                .unwrap_or_default();
            let target = sample_dir.join(format!("{p_stem}_first{max_uid}.{p_ext}"));
            copy_user_sample(profiles_path, &target, "uid", max_uid, false)?;
            sampled_profiles_path = Some(target);
        }
    }

    let mtime = sample_path.metadata()?.modified()?.duration_since(std::time::UNIX_EPOCH)?.as_secs_f64();
    let sampled_run = Run {
        run_id: format!("{}_first{max_uid}", selected.run_id),
        path: sample_path,
        mtime,
    };
    let mut out = experiment.clone();
    out.runs = vec![sampled_run];
    out.profiles_path = sampled_profiles_path.clone();
    out.profiles_output = sampled_profiles_path;
    Ok(out)
}

fn sample_observed(experiment: &Experiment, max_users: i64) -> anyhow::Result<Experiment> {
    let observed_path = experiment
        .observed_path
        .as_ref()
        .filter(|p| p.exists())
        .ok_or_else(|| anyhow::anyhow!("{}: cannot sample missing observed path", experiment.id))?;
    let columns = duckdb_parquet_columns(observed_path)?;
    let uid_col = if columns.iter().any(|c| c == "uid") {
        "uid"
    } else if columns.iter().any(|c| c == "user_id") {
        "user_id"
    } else {
        anyhow::bail!("{}: observed path has no uid/user_id column", experiment.id);
    };
    let sample_dir = repo_root()
        .join("data")
        .join("static_demo_samples")
        .join(&experiment.id)
        .join("observed")
        .join(format!("first_{max_users}"));
    let stem = observed_path.file_stem().unwrap_or_default().to_string_lossy();
    let ext = observed_path
        .extension()
        .map(|e| e.to_string_lossy().to_string())
        .unwrap_or_default();
    let sample_path = sample_dir.join(format!("{stem}_first{max_users}.{ext}"));
    println!(
        "[{}] materializing first {max_users} observed users -> {}",
        experiment.id,
        sample_path.display()
    );
    std::fs::create_dir_all(&sample_dir)?;
    let conn = duckdb::Connection::open_in_memory()?;
    conn.execute_batch(&format!(
        r#"
        COPY (
            WITH first_users AS (
                SELECT DISTINCT "{uid_col}" AS sampled_uid
                FROM read_parquet('{src}')
                ORDER BY sampled_uid
                LIMIT {max_users}
            )
            SELECT obs.*
            FROM read_parquet('{src}') obs
            JOIN first_users ON obs."{uid_col}" = first_users.sampled_uid
        ) TO '{dst}' (FORMAT PARQUET)
        "#,
        src = quote_path(observed_path),
        dst = quote_path(&sample_path),
    ))?;
    let mut out = experiment.clone();
    out.observed_path = Some(sample_path);
    Ok(out)
}

fn validate_expected_agents(experiment: &Experiment, expected: Option<i64>) -> anyhow::Result<()> {
    let Some(expected) = expected else { return Ok(()) };
    let selected = experiment.runs.first().expect("at least one run selected");
    let summary = run_summary(&selected.path)?;
    if summary.uids != Some(expected) {
        anyhow::bail!(
            "{} run {}: has {:?} agents; manifest expected {expected}. Generate/pin the intended demo run first.",
            experiment.id,
            selected.run_id,
            summary.uids,
        );
    }
    Ok(())
}

fn build_observed_graph_for_export(
    experiment: &Experiment,
) -> Option<Result<ObservedGraph, String>> {
    if !experiment.network_validation_config.observed_enabled {
        return None;
    }
    let Some(path) = experiment.observed_path.as_deref().filter(|p| p.exists()) else {
        return Some(Err("observed dataframe unavailable".to_string()));
    };
    Some((|| {
        let df = citybehavex_web::comparison::trajectory::read_parquet(path).map_err(|e| e.to_string())?;
        let cols: Vec<&str> = df.get_column_names().iter().map(|s| s.as_str()).collect();
        let uid_col = citybehavex_web::columns::detect_in(&cols, citybehavex_web::columns::UID_CANDIDATES);
        let datetime_col =
            citybehavex_web::columns::detect_in(&cols, citybehavex_web::columns::DATETIME_CANDIDATES);
        let (Some(uid_col), Some(datetime_col)) = (uid_col, datetime_col) else {
            return Err("observed network validation requires user and datetime columns".to_string());
        };
        network_validation::observed_edges_and_persistence(
            &df,
            &uid_col,
            &datetime_col,
            experiment.network_validation_config.location_mode,
            experiment.network_validation_config.location_col.as_deref(),
            experiment.network_validation_config.h3_resolution as u8,
            experiment.network_validation_config.max_group_size as usize,
        )
        .map_err(|e| e.to_string())
    })())
}

fn build_chart_payloads(
    out_dir: &Path,
    experiment: &Experiment,
    sections: &[(String, String)],
) -> anyhow::Result<()> {
    let selected = experiment.runs.first().expect("at least one run selected").clone();
    println!("[{}] exporting chart payloads", experiment.id);
    let ctx = ComparisonContext::from_experiment(experiment, &selected);

    let mut base = payload::chart_base_payload(&ctx);
    base["run_id"] = json!(selected.run_id);
    write_json(&out_dir.join("charts").join("base.json"), &wrapped(base))?;

    for (section, filter_key) in sections {
        let Ok(mut section_payload) = payload::chart_section_payload(&ctx, section, filter_key) else {
            continue;
        };
        section_payload["run_id"] = json!(selected.run_id);
        write_json(
            &out_dir
                .join("charts")
                .join("sections")
                .join(section)
                .join(format!("{filter_key}.json")),
            &wrapped(section_payload),
        )?;
    }

    let metrics = payload::metrics_export_artifact(&ctx)?;
    let mut metrics_out = json!({"experiment_id": experiment.id, "run_id": selected.run_id});
    if let (Some(target), Some(source)) = (metrics_out.as_object_mut(), metrics.as_object()) {
        for (key, value) in source {
            target.insert(key.clone(), value.clone());
        }
    }
    write_json(&out_dir.join("metrics-export.json"), &metrics_out)?;

    let observed_graph = build_observed_graph_for_export(experiment);
    let mut network_validation = payload::network_validation_payload(
        experiment.network_validation_config.enabled,
        experiment.network_validation_config.synthetic_enabled,
        Some(&selected.social_network_path()),
        observed_graph.as_ref().map(|r| r.as_ref().map_err(|e| e.as_str())),
        experiment.network_validation_config.random_seed,
    );
    network_validation["run_id"] = json!(selected.run_id);
    write_json(&out_dir.join("network-validation.json"), &wrapped(network_validation))?;

    let demo = DemoFilter { gender: None, age_bracket: None, job: None };
    let mut home_work_payload = home_work::build_home_work(
        &selected.path,
        ctx.observed_path.as_deref(),
        experiment.profiles_path.as_deref(),
        &demo,
    )?;
    home_work_payload["run_id"] = json!(selected.run_id);
    write_json(&out_dir.join("home-work").join("all.json"), &wrapped(home_work_payload))?;

    println!("[{}] chart payloads complete", experiment.id);
    Ok(())
}

fn timeline_meta_json(experiment: &Experiment, run: &Run) -> anyhow::Result<Value> {
    timeline_routes::build_timeline_meta(experiment, run)
}

fn export_timeline_chunks(
    out_dir: &Path,
    experiment: &Experiment,
    chunk_hours: i64,
    max_agents: i64,
    max_days: Option<i64>,
) -> anyhow::Result<()> {
    let selected = experiment.runs.first().expect("at least one run selected");
    println!("[{}] exporting timeline chunks", experiment.id);
    let meta = timeline_meta_json(experiment, selected)?;
    write_json(&out_dir.join("timeline").join("meta.json"), &wrapped(meta.clone()))?;

    let (Some(date_start), Some(date_end), Some(bbox)) = (
        meta.get("date_start").and_then(Value::as_str),
        meta.get("date_end").and_then(Value::as_str),
        meta.get("bbox").filter(|b| !b.is_null()),
    ) else {
        write_json(
            &out_dir.join("timeline").join("chunks.json"),
            &wrapped(json!({"chunks": []})),
        )?;
        return Ok(());
    };
    let start = NaiveDateTime::parse_from_str(date_start, "%Y-%m-%d %H:%M:%S")
        .or_else(|_| NaiveDateTime::parse_from_str(date_start, "%Y-%m-%dT%H:%M:%S"))?;
    let mut end = NaiveDateTime::parse_from_str(date_end, "%Y-%m-%d %H:%M:%S")
        .or_else(|_| NaiveDateTime::parse_from_str(date_end, "%Y-%m-%dT%H:%M:%S"))?;
    if let Some(max_days) = max_days {
        end = end.min(start + TimeDelta::days(max_days));
    }
    let step = TimeDelta::hours(chunk_hours);
    let bbox_tuple = (
        bbox.get("min_lat").and_then(Value::as_f64).unwrap_or(-90.0),
        bbox.get("min_lng").and_then(Value::as_f64).unwrap_or(-180.0),
        bbox.get("max_lat").and_then(Value::as_f64).unwrap_or(90.0),
        bbox.get("max_lng").and_then(Value::as_f64).unwrap_or(180.0),
    );

    let mut chunks = Vec::new();
    let mut current = start;
    let mut index = 0usize;
    while current < end {
        let until = (current + step).min(end);
        let mut chunk_payload = timeline_routes::build_timeline_legs(
            &experiment.id,
            experiment,
            selected,
            current,
            until,
            bbox_tuple,
            max_agents,
        )?;
        chunk_payload["since"] = json!(current.and_utc().to_rfc3339());
        chunk_payload["until"] = json!(until.and_utc().to_rfc3339());
        let chunk_name = format!("{index:05}.json");
        write_json(
            &out_dir.join("timeline").join("legs").join(&chunk_name),
            &wrapped(chunk_payload.clone()),
        )?;
        chunks.push(json!({
            "file": chunk_name,
            "since": chunk_payload["since"],
            "until": chunk_payload["until"],
        }));
        current = until;
        index += 1;
    }
    write_json(
        &out_dir.join("timeline").join("chunks.json"),
        &wrapped(json!({"chunks": chunks})),
    )?;
    println!("[{}] exported {} timeline chunks", experiment.id, chunks.len());
    Ok(())
}

fn export_agent_details(out_dir: &Path, experiment: &Experiment, max_agents: i64) -> anyhow::Result<()> {
    let selected = experiment.runs.first().expect("at least one run selected");
    println!("[{}] exporting agent details for {max_agents} agents", experiment.id);
    for uid in 1..=max_agents {
        let agent = timeline_routes::build_agent_payload(experiment, selected, uid)?;
        let crp = timeline_routes::build_agent_crp_payload(experiment, selected, uid)?;
        let social = timeline_routes::build_agent_social_payload(experiment, selected, uid)?;
        let agent_dir = out_dir.join("timeline").join("agents").join(uid.to_string());
        write_json(&agent_dir.join("profile.json"), &wrapped(agent))?;
        write_json(&agent_dir.join("crp.json"), &wrapped(crp))?;
        write_json(&agent_dir.join("social.json"), &wrapped(social))?;
    }
    println!("[{}] agent details complete", experiment.id);
    Ok(())
}

fn sanitize_experiment(mut experiment: Experiment, allow_observed: bool, label: Option<&str>) -> Experiment {
    if let Some(label) = label {
        experiment.label = label.to_string();
    }
    if !allow_observed {
        experiment.observed_path = None;
    }
    experiment
}

fn filter_runs(mut experiment: Experiment, run_id: &str) -> anyhow::Result<Experiment> {
    let selected = experiment
        .run(Some(run_id))
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("{}: run {run_id:?} not found", experiment.id))?;
    experiment.runs = vec![selected];
    Ok(experiment)
}

fn export_static_demo(manifest_path: &Path) -> anyhow::Result<()> {
    let manifest: Manifest = serde_yaml::from_slice(&std::fs::read(manifest_path)?)?;
    let output_dir = repo_root().join(&manifest.output_dir);
    let chunk_hours = manifest.timeline_chunk_hours;
    let max_agents = manifest.timeline_max_agents;
    let export_agent_details_flag = manifest.export_agent_details;
    let sections: Vec<(String, String)> = manifest.chart_sections.clone().unwrap_or_else(|| {
        DEFAULT_SECTIONS
            .iter()
            .map(|(s, f)| (s.to_string(), f.to_string()))
            .collect()
    });

    let mut prepared: Vec<(&ManifestExperiment, Experiment)> = Vec::new();
    for entry in &manifest.experiments {
        let base_experiment = get_experiment(&entry.id)
            .ok_or_else(|| anyhow::anyhow!("unknown experiment {:?}", entry.id))?;
        let mut experiment = sanitize_experiment(
            filter_runs(base_experiment, &entry.run_id)?,
            entry.allow_observed,
            entry.label.as_deref(),
        );
        if let Some(sample_agents) = entry.sample_agents {
            experiment = sample_run(&experiment, sample_agents)?;
        }
        if let Some(observed_sample_agents) = entry.observed_sample_agents {
            experiment = sample_observed(&experiment, observed_sample_agents)?;
        }
        validate_expected_agents(&experiment, entry.expected_agents)?;
        prepared.push((entry, experiment));
    }

    if output_dir.exists() {
        std::fs::remove_dir_all(&output_dir)?;
    }
    std::fs::create_dir_all(&output_dir)?;

    let mut experiments_payload = Vec::new();
    let mut detail_payloads: std::collections::HashMap<String, Value> = std::collections::HashMap::new();

    for (entry, experiment) in &prepared {
        let selected = experiment.runs.first().expect("at least one run selected");
        let exp_out = output_dir.join(&entry.id).join(&selected.run_id);
        let mut detail = serde_json::to_value(experiment.to_json(true))?;
        detail["static_demo"] = json!(true);
        detail["observed_sanitized"] = json!(!entry.allow_observed);
        experiments_payload.push(detail.clone());
        detail_payloads.insert(entry.id.clone(), detail);

        build_chart_payloads(&exp_out, experiment, &sections)?;
        export_timeline_chunks(&exp_out, experiment, chunk_hours, max_agents, entry.timeline_days)?;
        if export_agent_details_flag {
            export_agent_details(&exp_out, experiment, max_agents)?;
        }
    }

    write_json(&output_dir.join("experiments.json"), &wrapped(json!(experiments_payload)))?;
    for (exp_id, detail) in &detail_payloads {
        write_json(&output_dir.join("experiments").join(format!("{exp_id}.json")), &wrapped(detail.clone()))?;
    }
    let manifest_experiments: Vec<Value> = manifest
        .experiments
        .iter()
        .map(|e| {
            let run_id = detail_payloads[&e.id]["runs"][0]["run_id"].clone();
            json!({
                "id": e.id,
                "run_id": run_id,
                "source_run_id": e.run_id,
                "allow_observed": e.allow_observed,
            })
        })
        .collect();
    write_json(
        &output_dir.join("manifest.json"),
        &json!({
            "generated_from": manifest_path.strip_prefix(repo_root()).unwrap_or(manifest_path).display().to_string(),
            "timeline_chunk_hours": chunk_hours,
            "timeline_max_agents": max_agents,
            "experiments": manifest_experiments,
        }),
    )?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let manifest_arg = args
        .iter()
        .position(|a| a == "--manifest")
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
        .unwrap_or("web/demo_export.yaml");
    let manifest_path = repo_root().join(manifest_arg);
    if let Err(err) = export_static_demo(&manifest_path) {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
    Ok(())
}
