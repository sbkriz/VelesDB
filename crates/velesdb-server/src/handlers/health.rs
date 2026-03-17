//! Health and readiness check handlers.

use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use std::sync::atomic::Ordering;
use std::sync::Arc;

use crate::AppState;

/// Liveness probe — always returns 200 OK.
#[utoipa::path(
    get,
    path = "/health",
    tag = "health",
    responses(
        (status = 200, description = "Server is alive", body = Object)
    )
)]
pub async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION")
    }))
}

/// Readiness probe — returns 200 when the database is fully loaded, 503 otherwise.
#[utoipa::path(
    get,
    path = "/ready",
    tag = "health",
    responses(
        (status = 200, description = "Server is ready to accept requests", body = Object),
        (status = 503, description = "Server is not yet ready", body = Object)
    )
)]
pub async fn readiness_check(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if state.ready.load(Ordering::Relaxed) {
        (
            StatusCode::OK,
            Json(serde_json::json!({
                "status": "ready",
                "version": env!("CARGO_PKG_VERSION")
            })),
        )
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "status": "not_ready",
                "version": env!("CARGO_PKG_VERSION")
            })),
        )
    }
}
