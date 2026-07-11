//! Chart, metrics-export, network-validation, and home/work API routes.

use crate::cache::Cache;
use crate::config;
use crate::experiments::{self, get_experiment};
use crate::models::{ApiError, ApiResponse, ApiResult};
use crate::payload::{self, ComparisonContext};
use axum::extract::{Path, Query};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use serde_json::{Value, json};
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
pub struct RunRefreshQuery {
    pub run: Option<String>,
    #[serde(default)]
    pub refresh: bool,
}

#[derive(Debug, Deserialize)]
pub struct SectionQuery {
    #[serde(default = "default_filter")]
    pub filter: String,
    pub run: Option<String>,
    #[serde(default)]
    pub refresh: bool,
}

fn default_filter() -> String {
    "all".to_string()
}

#[derive(Debug, Deserialize)]
pub struct MetricsExportQuery {
    pub run: Option<String>,
    #[serde(default = "default_format")]
    pub format: String,
    #[serde(default)]
    pub refresh: bool,
}

fn default_format() -> String {
    "json".to_string()
}

#[derive(Debug, Deserialize)]
pub struct HomeWorkQuery {
    pub run: Option<String>,
    pub gender: Option<String>,
    pub age_bracket: Option<String>,
    pub job: Option<String>,
    #[serde(default)]
    pub refresh: bool,
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

fn cache() -> Cache {
    Cache::new(config::cache_dir())
}

fn cache_extra_paths(paths: &[PathBuf]) -> Vec<&std::path::Path> {
    paths
        .iter()
        .filter(|p| p.exists())
        .map(|p| p.as_path())
        .collect()
}

pub async fn get_charts_route(
    Path(exp_id): Path<String>,
    Query(q): Query<RunRefreshQuery>,
) -> ApiResult<Value> {
    let (exp, run) = selected(&exp_id, q.run.as_deref())?;
    let ctx = ComparisonContext::from_experiment(&exp, &run);
    let observed = exp
        .observed_path
        .as_ref()
        .filter(|p| p.exists())
        .map(|p| p.as_path());
    let extra_owned = vec![
        run.social_network_path(),
        run.encounters_path(),
        run.activities_path(),
        run.moving_path(),
    ];
    let extra = cache_extra_paths(&extra_owned);
    let extra_key = json!({
        "transport_spatial": exp.transport_spatial_config,
        "evaluation_adaptation": exp.evaluation_adaptation_config,
    });
    let run_id = run.run_id.clone();
    let synthetic = run.path.clone();
    let value = cache()
        .get_or_build(
            &exp_id,
            &run_id,
            &synthetic,
            observed,
            &extra,
            Some(&extra_key),
            q.refresh,
            || async move { Ok(payload::chart_base_payload(&ctx)) },
        )
        .await
        .map_err(|e| ApiError::internal(e.to_string()))?;
    let mut value = value;
    value["run_id"] = json!(run_id);
    Ok(ApiResponse::new(value))
}

pub async fn get_chart_section_route(
    Path((exp_id, section)): Path<(String, String)>,
    Query(q): Query<SectionQuery>,
) -> ApiResult<Value> {
    let (exp, run) = selected(&exp_id, q.run.as_deref())?;
    let ctx = ComparisonContext::from_experiment(&exp, &run);
    let observed = exp
        .observed_path
        .as_ref()
        .filter(|p| p.exists())
        .map(|p| p.as_path());
    let mut extra_owned = vec![run.activities_path(), run.moving_path()];
    if let Some(path) = exp.time_use_path.clone() {
        extra_owned.push(path.with_extension("parquet"));
        extra_owned.push(path);
    }
    let extra = cache_extra_paths(&extra_owned);
    let extra_key = json!({
        "section": section,
        "filter": q.filter,
        "transport_spatial": exp.transport_spatial_config,
        "evaluation_adaptation": exp.evaluation_adaptation_config,
    });
    let run_id = run.run_id.clone();
    let synthetic = run.path.clone();
    let section_for_build = section.clone();
    let filter_for_build = q.filter.clone();
    let value = cache()
        .get_or_build(
            &exp_id,
            &run_id,
            &synthetic,
            observed,
            &extra,
            Some(&extra_key),
            q.refresh,
            || async move {
                payload::chart_section_payload(&ctx, &section_for_build, &filter_for_build)
            },
        )
        .await
        .map_err(|e| match e {
            crate::cache::CacheError::Build(inner)
                if inner.to_string().starts_with("unknown chart section:") =>
            {
                ApiError::not_found(inner.to_string())
            }
            other => ApiError::internal(other.to_string()),
        })?;
    let mut value = value;
    value["run_id"] = json!(run_id);
    Ok(ApiResponse::new(value))
}

pub async fn metrics_export_route(
    Path(exp_id): Path<String>,
    Query(q): Query<MetricsExportQuery>,
) -> Result<Response, ApiError> {
    if q.format.to_lowercase() != "json" {
        return Err(ApiError::bad_request(
            "only json metrics export is supported",
        ));
    }
    let (exp, run) = selected(&exp_id, q.run.as_deref())?;
    let ctx = ComparisonContext::from_experiment(&exp, &run);
    let observed = exp
        .observed_path
        .as_ref()
        .filter(|p| p.exists())
        .map(|p| p.as_path());
    let mut extra_owned = vec![run.activities_path()];
    if let Some(path) = exp.time_use_path.clone() {
        extra_owned.push(path.with_extension("parquet"));
        extra_owned.push(path);
    }
    let extra = cache_extra_paths(&extra_owned);
    let extra_key = json!({
        "format": "json",
        "evaluation_adaptation": exp.evaluation_adaptation_config,
    });
    let run_id = run.run_id.clone();
    let synthetic = run.path.clone();
    let artifact = cache()
        .get_or_build(
            &format!("{exp_id}__metrics_export"),
            &run_id,
            &synthetic,
            observed,
            &extra,
            Some(&extra_key),
            q.refresh,
            || async move { Ok(payload::chart_section_payload(&ctx, "metrics", "all")?) },
        )
        .await
        .map_err(|e| ApiError::internal(e.to_string()))?;
    let mut out =
        payload::metrics_export_payload(&ComparisonContext::from_experiment(&exp, &run), &artifact);
    out["experiment_id"] = json!(exp_id);
    out["run_id"] = json!(run_id);
    let filename = format!("citybehavex-{}-{}-metrics.json", exp.id, run.run_id);
    let mut headers = HeaderMap::new();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    headers.insert(
        header::CONTENT_DISPOSITION,
        HeaderValue::from_str(&format!("attachment; filename=\"{filename}\""))
            .map_err(|e| ApiError::internal(e.to_string()))?,
    );
    let body = serde_json::to_string_pretty(&out).map_err(|e| ApiError::internal(e.to_string()))?;
    Ok((StatusCode::OK, headers, body).into_response())
}

pub async fn network_validation_route(
    Path(exp_id): Path<String>,
    Query(q): Query<RunRefreshQuery>,
) -> ApiResult<Value> {
    let (exp, run) = selected(&exp_id, q.run.as_deref())?;
    let observed = exp
        .observed_path
        .as_ref()
        .filter(|p| p.exists())
        .map(|p| p.as_path());
    let extra_owned = vec![run.social_network_path(), run.encounters_path()];
    let extra = cache_extra_paths(&extra_owned);
    let run_id = run.run_id.clone();
    let synthetic = run.path.clone();
    let social_path = run.social_network_path();
    let nv_seed = exp.network_validation_config.random_seed;
    let nv_enabled = exp.network_validation_config.enabled;
    let value = cache()
        .get_or_build(
            &format!("{exp_id}__network_validation"),
            &run_id,
            &synthetic,
            observed,
            &extra,
            None,
            q.refresh,
            || async move {
                Ok(payload::network_validation_payload(
                    nv_enabled,
                    Some(&social_path),
                    nv_seed,
                ))
            },
        )
        .await
        .map_err(|e| ApiError::internal(e.to_string()))?;
    let mut value = value;
    value["run_id"] = json!(run_id);
    Ok(ApiResponse::new(value))
}

pub async fn home_work_route(
    Path(exp_id): Path<String>,
    Query(q): Query<HomeWorkQuery>,
) -> ApiResult<Value> {
    let (exp, run) = selected(&exp_id, q.run.as_deref())?;
    let observed = exp
        .observed_path
        .as_ref()
        .filter(|p| p.exists())
        .map(|p| p.as_path());
    let extra_owned: Vec<PathBuf> = exp.profiles_path.iter().cloned().collect();
    let extra = cache_extra_paths(&extra_owned);
    let demo = crate::home_work::DemoFilter {
        gender: q.gender.clone(),
        age_bracket: q.age_bracket.clone(),
        job: q.job.clone(),
    };
    let extra_key = json!({
        "demo": {
            "gender": q.gender,
            "age_bracket": q.age_bracket,
            "job": q.job,
        }
    });
    let run_id = run.run_id.clone();
    let synthetic = run.path.clone();
    let synthetic_for_build = synthetic.clone();
    let observed_for_build = observed.map(|p| p.to_path_buf());
    let profiles_path = exp.profiles_path.clone();
    let value = cache()
        .get_or_build(
            &format!("{exp_id}__home_work"),
            &run_id,
            &synthetic,
            observed,
            &extra,
            Some(&extra_key),
            q.refresh,
            || async move {
                crate::home_work::build_home_work(
                    &synthetic_for_build,
                    observed_for_build.as_deref(),
                    profiles_path.as_deref(),
                    &demo,
                )
            },
        )
        .await
        .map_err(|e| ApiError::internal(e.to_string()))?;
    let mut payload = value;
    payload["run_id"] = json!(run_id);
    Ok(ApiResponse::new(payload))
}
