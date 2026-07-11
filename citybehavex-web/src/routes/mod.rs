pub mod charts;
pub mod experiments;
pub mod timeline;

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

pub fn charts_router() -> Router {
    Router::new()
        .route(
            "/experiments/{exp_id}/charts",
            get(charts::get_charts_route),
        )
        .route(
            "/experiments/{exp_id}/charts/{section}",
            get(charts::get_chart_section_route),
        )
        .route(
            "/experiments/{exp_id}/metrics-export",
            get(charts::metrics_export_route),
        )
        .route(
            "/experiments/{exp_id}/network-validation",
            get(charts::network_validation_route),
        )
        .route(
            "/experiments/{exp_id}/home-work",
            get(charts::home_work_route),
        )
}

pub fn timeline_router() -> Router {
    Router::new()
        .route(
            "/experiments/{exp_id}/timeline/meta",
            get(timeline::timeline_meta_route),
        )
        .route(
            "/experiments/{exp_id}/timeline/legs",
            get(timeline::timeline_legs_route),
        )
        .route(
            "/experiments/{exp_id}/timeline/agents/{uid}",
            get(timeline::timeline_agent_route),
        )
        .route(
            "/experiments/{exp_id}/timeline/agents/{uid}/crp",
            get(timeline::timeline_agent_crp_route),
        )
        .route(
            "/experiments/{exp_id}/timeline/agents/{uid}/social",
            get(timeline::timeline_agent_social_route),
        )
}
