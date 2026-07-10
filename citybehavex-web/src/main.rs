mod cache;
mod config;
mod datasource;
mod experiments;
mod models;
mod routes;
mod settings;

use axum::Router;
use axum::body::Body;
use axum::extract::Request;
use axum::http::{HeaderValue, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use regex::Regex;
use serde_json::json;
use std::path::PathBuf;
use std::sync::Arc;
use tower::ServiceExt;
use tower_http::compression::CompressionLayer;
use tower_http::cors::{AllowHeaders, AllowMethods, AllowOrigin, CorsLayer};
use tower_http::services::ServeDir;
use tower_http::trace::TraceLayer;

/// Mirrors `main.py::create_app`'s `allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?"`,
/// `allow_credentials=True`, `allow_methods=["*"]`, `allow_headers=["*"]`.
/// Starlette's `CORSMiddleware` implements "allow everything, with
/// credentials" by reflecting the request's actual `Access-Control-Request-
/// Method`/`-Headers` back rather than literally sending `*` -- the CORS
/// spec forbids a literal wildcard alongside `Allow-Credentials: true`, and
/// tower-http enforces that at layer-construction time (panics otherwise), so
/// `mirror_request()` is the equivalent here.
fn cors_layer() -> CorsLayer {
    let origin_re = Arc::new(Regex::new(r"^http://(localhost|127\.0\.0\.1)(:\d+)?$").unwrap());
    CorsLayer::new()
        .allow_credentials(true)
        .allow_methods(AllowMethods::mirror_request())
        .allow_headers(AllowHeaders::mirror_request())
        .allow_origin(AllowOrigin::predicate(move |origin: &HeaderValue, _| {
            origin
                .to_str()
                .map(|s| origin_re.is_match(s))
                .unwrap_or(false)
        }))
}

/// JSON 404 for unmatched `/api/*` paths, matching FastAPI's default
/// `HTTPException(404)` body shape.
async fn api_not_found() -> impl IntoResponse {
    (StatusCode::NOT_FOUND, axum::Json(json!({"detail": "Not Found"})))
}

/// Routes under `/api` only -- nested into the top-level router below so its
/// own `.fallback` (JSON 404) only applies to unmatched `/api/*` paths,
/// leaving the top-level router's fallback free for the SPA static files.
/// (A router has exactly one fallback slot; setting it twice on the same
/// router silently overwrites the first, which is what nesting avoids here.)
fn api_router() -> Router {
    Router::new()
        .route("/health", get(|| async { axum::Json(json!({"status": "ok"})) }))
        .merge(routes::experiments_router())
        .fallback(api_not_found)
}

/// Serves a real static asset when the request matches one, otherwise falls
/// back to `index.html` with a plain **200** (SPA client-side routing) --
/// matching `main.py::_mount_frontend`'s `@app.exception_handler(404)`,
/// which returns `FileResponse(index)` uncustomized (Starlette defaults that
/// to 200), not a 404. This request never sees `/api/*` paths: those are
/// fully handled (matched route or JSON 404) by the nested `/api` router
/// before the top-level fallback is ever reached.
async fn spa_fallback(dist: PathBuf, req: Request<Body>) -> Response {
    let serve_dir = ServeDir::new(&dist);
    match serve_dir.oneshot(req).await {
        Ok(res) if res.status() != StatusCode::NOT_FOUND => res.into_response(),
        _ => match tokio::fs::read(dist.join("index.html")).await {
            Ok(bytes) => (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
                bytes,
            )
                .into_response(),
            Err(_) => (StatusCode::NOT_FOUND, axum::Json(json!({"detail": "Not Found"})))
                .into_response(),
        },
    }
}

/// Collapses **every** `/api/*` 404 (whether from an unmatched route or a
/// handler's own `HTTPException(404, detail="...")`-equivalent) down to the
/// generic `{"detail":"Not Found"}` body -- reproducing a real, easily
/// overlooked quirk of `main.py::_mount_frontend`: its
/// `@app.exception_handler(404)` is a Starlette/FastAPI exception handler
/// registered by *status code*, so it catches every 404 response app-wide,
/// not just genuinely-unmatched routes, and for any `/api` path it always
/// substitutes the generic body regardless of what detail the raising code
/// provided. That handler is only registered at all when `web/frontend/dist`
/// exists (`_mount_frontend` is conditional on `_FRONTEND_DIST.is_dir()`), so
/// this middleware is gated the same way -- confirmed against the live
/// Python backend: `GET /api/experiments/{unknown}` returns
/// `{"detail":"unknown experiment ..."}` when no frontend build exists, but
/// `{"detail":"Not Found"}` once one does.
async fn collapse_api_404(dist_mounted: bool, req: Request, next: Next) -> Response {
    let is_api_path = req.uri().path().starts_with("/api");
    let response = next.run(req).await;
    if dist_mounted && is_api_path && response.status() == StatusCode::NOT_FOUND {
        return (StatusCode::NOT_FOUND, axum::Json(json!({"detail": "Not Found"}))).into_response();
    }
    response
}

fn build_app() -> Router {
    let app = Router::new().nest("/api", api_router());

    let dist = config::frontend_dist_dir();
    let dist_mounted = dist.is_dir();
    let app = if dist_mounted {
        tracing::info!(path = %dist.display(), "serving frontend SPA build");
        app.fallback_service(tower::service_fn(move |req: Request<Body>| {
            let dist = dist.clone();
            async move { Ok::<_, std::convert::Infallible>(spa_fallback(dist, req).await) }
        }))
    } else {
        tracing::warn!(path = %dist.display(), "frontend dist/ not found, SPA not mounted");
        app
    };

    app.layer(TraceLayer::new_for_http())
        .layer(CompressionLayer::new())
        .layer(cors_layer())
        .layer(middleware::from_fn(move |req, next| {
            collapse_api_404(dist_mounted, req, next)
        }))
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let port: u16 = std::env::var("CBX_WEB_RS_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8001);

    let app = build_app();
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port))
        .await
        .unwrap_or_else(|e| panic!("failed to bind 0.0.0.0:{port}: {e}"));
    tracing::info!(port, "citybehavex-web listening");
    axum::serve(listener, app).await.unwrap();
}
