//! `/api/experiments*` routes, mirroring `web/backend/app/api/experiments.py`.

use crate::experiments::{
    self, ExperimentError, ExperimentUpdate, archive_experiment, delete_run, get_experiment,
    list_experiments, update_experiment,
};
use crate::models::{ApiError, ApiResponse, ApiResult};
use axum::Json;
use axum::extract::rejection::JsonRejection;
use axum::extract::{Path, Query};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct WithSummaryQuery {
    #[serde(default)]
    with_summary: bool,
}

pub async fn list_experiments_route(
    Query(q): Query<WithSummaryQuery>,
) -> ApiResult<Vec<experiments::ExperimentJson>> {
    let out = list_experiments()
        .iter()
        .map(|e| e.to_json(q.with_summary))
        .collect();
    Ok(ApiResponse::new(out))
}

pub async fn get_experiment_route(
    Path(exp_id): Path<String>,
) -> ApiResult<experiments::ExperimentJson> {
    let experiment = get_experiment(&exp_id)
        .ok_or_else(|| ApiError::not_found(format!("unknown experiment {exp_id:?}")))?;
    Ok(ApiResponse::new(experiment.to_json(true)))
}

fn mutation_error_response(exp_id: &str, err: ExperimentError) -> ApiError {
    match err {
        ExperimentError::NotFound(_) => ApiError::not_found(format!("unknown experiment {exp_id:?}")),
        ExperimentError::RunNotFound(run_id) => {
            ApiError::not_found(format!("unknown run {run_id:?}"))
        }
        ExperimentError::Mutation(msg) => ApiError::bad_request(msg),
        ExperimentError::Io(e) => ApiError::internal(e.to_string()),
    }
}

pub async fn patch_experiment_route(
    Path(exp_id): Path<String>,
    body: Result<Json<ExperimentUpdate>, JsonRejection>,
) -> ApiResult<experiments::ExperimentJson> {
    // FastAPI/Pydantic returns 422 (not axum's default 400 plain-text
    // rejection) for a request body that fails to parse/validate against
    // the expected shape.
    let Json(update) = body.map_err(|e| ApiError::unprocessable(e.body_text()))?;
    let experiment = update_experiment(&exp_id, &update)
        .map_err(|e| mutation_error_response(&exp_id, e))?;
    Ok(ApiResponse::new(experiment.to_json(true)))
}

#[derive(Debug, Serialize)]
pub struct ArchivedConfig {
    archived_config: String,
}

pub async fn archive_experiment_route(
    Path(exp_id): Path<String>,
) -> ApiResult<ArchivedConfig> {
    let archived_path =
        archive_experiment(&exp_id).map_err(|e| mutation_error_response(&exp_id, e))?;
    Ok(ApiResponse::new(ArchivedConfig {
        archived_config: archived_path.file_name().unwrap().to_string_lossy().to_string(),
    }))
}

#[derive(Debug, Serialize)]
pub struct DeletedRun {
    deleted: Vec<String>,
}

pub async fn delete_run_route(
    Path((exp_id, run_id)): Path<(String, String)>,
) -> ApiResult<DeletedRun> {
    let deleted = delete_run(&exp_id, &run_id).map_err(|e| mutation_error_response(&exp_id, e))?;
    Ok(ApiResponse::new(DeletedRun {
        deleted: deleted.into_iter().map(|p| p.display().to_string()).collect(),
    }))
}
