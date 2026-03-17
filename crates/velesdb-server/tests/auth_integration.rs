#![allow(clippy::doc_markdown)]
//! Integration tests for API key authentication middleware.

mod common;

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use common::{create_test_app, create_test_app_with_auth};
use serde_json::Value;
use tempfile::TempDir;
use tower::ServiceExt;

// ============================================================================
// Auth enabled — valid key returns 200
// ============================================================================

#[tokio::test]
async fn test_auth_valid_key_returns_200() {
    let temp_dir = TempDir::new().unwrap();
    let app = create_test_app_with_auth(&temp_dir, vec!["secret-key".to_string()]);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/collections")
                .header("Authorization", "Bearer secret-key")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

// ============================================================================
// Auth enabled — invalid key returns 401
// ============================================================================

#[tokio::test]
async fn test_auth_invalid_key_returns_401() {
    let temp_dir = TempDir::new().unwrap();
    let app = create_test_app_with_auth(&temp_dir, vec!["secret-key".to_string()]);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/collections")
                .header("Authorization", "Bearer wrong-key")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["error"], "Unauthorized");
}

// ============================================================================
// Auth enabled — missing header returns 401
// ============================================================================

#[tokio::test]
async fn test_auth_missing_header_returns_401() {
    let temp_dir = TempDir::new().unwrap();
    let app = create_test_app_with_auth(&temp_dir, vec!["secret-key".to_string()]);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/collections")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

// ============================================================================
// Auth enabled — /health bypasses auth (always public)
// ============================================================================

#[tokio::test]
async fn test_auth_health_endpoint_bypasses_auth() {
    let temp_dir = TempDir::new().unwrap();
    let app = create_test_app_with_auth(&temp_dir, vec!["secret-key".to_string()]);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["status"], "ok");
}

// ============================================================================
// Auth disabled — no keys configured, all endpoints accessible
// ============================================================================

#[tokio::test]
async fn test_auth_disabled_when_no_keys() {
    let temp_dir = TempDir::new().unwrap();
    let app = create_test_app(&temp_dir);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/collections")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

// ============================================================================
// Auth enabled — multiple keys supported
// ============================================================================

#[tokio::test]
async fn test_auth_multiple_keys_all_valid() {
    let temp_dir = TempDir::new().unwrap();
    let keys = vec!["key-alpha".to_string(), "key-beta".to_string()];

    // First key works
    let app = create_test_app_with_auth(&temp_dir, keys.clone());
    let response = app
        .oneshot(
            Request::builder()
                .uri("/collections")
                .header("Authorization", "Bearer key-alpha")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    // Second key works too
    let app = create_test_app_with_auth(&temp_dir, keys);
    let response = app
        .oneshot(
            Request::builder()
                .uri("/collections")
                .header("Authorization", "Bearer key-beta")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

// ============================================================================
// Auth enabled — invalid format (Basic instead of Bearer) returns 401
// ============================================================================

#[tokio::test]
async fn test_auth_basic_scheme_rejected() {
    let temp_dir = TempDir::new().unwrap();
    let app = create_test_app_with_auth(&temp_dir, vec!["secret-key".to_string()]);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/collections")
                .header("Authorization", "Basic dXNlcjpwYXNz")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}
