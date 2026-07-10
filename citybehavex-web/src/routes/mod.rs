pub mod experiments;

use axum::Router;
use axum::routing::{delete, get, post};

pub fn experiments_router() -> Router {
    Router::new()
        .route("/experiments", get(experiments::list_experiments_route))
        .route(
            "/experiments/{exp_id}",
            get(experiments::get_experiment_route).patch(experiments::patch_experiment_route),
        )
        .route(
            "/experiments/{exp_id}/archive",
            post(experiments::archive_experiment_route),
        )
        .route(
            "/experiments/{exp_id}/runs/{run_id}",
            delete(experiments::delete_run_route),
        )
}
