//! Prometheus metrics handler for VelesDB REST API.
//!
//! Provides a `/metrics` endpoint for Prometheus scraping.
//! Requires the `prometheus` feature flag to be enabled.
//! [EPIC-016/US-034, US-035]
//!
//! Metrics exposed:
//! - `velesdb_info`: Server version info
//! - `velesdb_up`: Server availability gauge

#![allow(dead_code)] // Functions exposed via feature flag, used when prometheus feature enabled

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};
use std::fmt::Write;

fn formatting_error_response() -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        [("content-type", "text/plain; version=0.0.4; charset=utf-8")],
        "Failed to format metrics".to_string(),
    )
        .into_response()
}
/// Prometheus text format metrics response.
///
/// Returns metrics in Prometheus exposition format.
#[utoipa::path(
    get,
    path = "/metrics",
    responses(
        (status = 200, description = "Prometheus metrics", content_type = "text/plain"),
        (status = 500, description = "Internal server error")
    ),
    tag = "metrics"
)]
pub async fn prometheus_metrics() -> impl IntoResponse {
    let mut output = String::new();

    // Write header comments
    if writeln!(output, "# VelesDB Prometheus Metrics").is_err() || writeln!(output).is_err() {
        return formatting_error_response();
    }

    // Server info
    if writeln!(output, "# HELP velesdb_info VelesDB server information").is_err()
        || writeln!(output, "# TYPE velesdb_info gauge").is_err()
        || writeln!(
            output,
            "velesdb_info{{version=\"{}\"}} 1",
            env!("CARGO_PKG_VERSION")
        )
        .is_err()
        || writeln!(output).is_err()
    {
        return formatting_error_response();
    }

    // velesdb_up gauge
    if writeln!(output, "# HELP velesdb_up VelesDB server is up and running").is_err()
        || writeln!(output, "# TYPE velesdb_up gauge").is_err()
        || writeln!(output, "velesdb_up 1").is_err()
    {
        return formatting_error_response();
    }

    (
        StatusCode::OK,
        [("content-type", "text/plain; version=0.0.4; charset=utf-8")],
        output,
    )
        .into_response()
}

/// Simple health metrics for lightweight monitoring.
pub async fn health_metrics() -> impl IntoResponse {
    let mut output = String::new();

    if writeln!(output, "# HELP velesdb_up VelesDB server is up").is_err()
        || writeln!(output, "# TYPE velesdb_up gauge").is_err()
        || writeln!(output, "velesdb_up 1").is_err()
    {
        return formatting_error_response();
    }

    (
        StatusCode::OK,
        [("content-type", "text/plain; version=0.0.4; charset=utf-8")],
        output,
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_prometheus_metrics_response_shape() {
        let response = prometheus_metrics().await.into_response();
        assert_eq!(response.status(), StatusCode::OK);

        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|h| h.to_str().ok());
        assert_eq!(
            content_type,
            Some("text/plain; version=0.0.4; charset=utf-8")
        );
    }

    #[tokio::test]
    async fn test_health_metrics_response_shape() {
        let response = health_metrics().await.into_response();
        assert_eq!(response.status(), StatusCode::OK);

        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|h| h.to_str().ok());
        assert_eq!(
            content_type,
            Some("text/plain; version=0.0.4; charset=utf-8")
        );
    }
}
