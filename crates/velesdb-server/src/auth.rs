//! API key authentication middleware.
//!
//! When `api_keys` is non-empty, all requests except those to public paths
//! (e.g. `GET /health`) must include a valid `Authorization: Bearer <key>` header.
//! When `api_keys` is empty, authentication is disabled (local dev mode).

use axum::{
    body::Body,
    extract::Request,
    http::{header, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use std::sync::Arc;

/// Shared authentication state injected into the middleware.
#[derive(Debug, Clone)]
pub struct AuthState {
    /// Allowed API keys. Empty means auth is disabled.
    pub api_keys: Arc<Vec<String>>,
}

impl AuthState {
    /// Create a new `AuthState` from a list of API keys.
    pub fn new(api_keys: Vec<String>) -> Self {
        Self {
            api_keys: Arc::new(api_keys),
        }
    }

    /// Returns `true` when authentication is enabled.
    pub fn auth_enabled(&self) -> bool {
        !self.api_keys.is_empty()
    }
}

/// Paths that bypass authentication.
fn is_public_path(path: &str) -> bool {
    path == "/health" || path == "/ready"
}

/// Extract the Bearer token from the Authorization header value.
fn extract_bearer_token(header_value: &str) -> Option<&str> {
    let trimmed = header_value.trim();
    if trimmed.len() > 7 && trimmed[..7].eq_ignore_ascii_case("bearer ") {
        Some(trimmed[7..].trim())
    } else {
        None
    }
}

/// Axum middleware function for API key authentication.
///
/// Use with `axum::middleware::from_fn_with_state`.
pub async fn auth_middleware(
    axum::extract::State(state): axum::extract::State<AuthState>,
    request: Request<Body>,
    next: Next,
) -> Response {
    // Skip auth if disabled (no keys configured)
    if !state.auth_enabled() {
        return next.run(request).await;
    }

    // Skip auth for public paths
    if is_public_path(request.uri().path()) {
        return next.run(request).await;
    }

    // Extract and validate Bearer token
    let auth_header = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok());

    match auth_header {
        Some(value) => match extract_bearer_token(value) {
            Some(token) if state.api_keys.contains(&token.to_string()) => next.run(request).await,
            Some(_) => unauthorized_response("invalid API key"),
            None => {
                unauthorized_response("invalid Authorization header format, expected: Bearer <key>")
            }
        },
        None => unauthorized_response("missing Authorization header"),
    }
}

/// Build a 401 Unauthorized JSON response.
fn unauthorized_response(message: &str) -> Response {
    (
        StatusCode::UNAUTHORIZED,
        Json(serde_json::json!({
            "error": "Unauthorized",
            "message": message
        })),
    )
        .into_response()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auth_state_disabled_when_empty() {
        let state = AuthState::new(vec![]);
        assert!(!state.auth_enabled());
    }

    #[test]
    fn test_auth_state_enabled_with_keys() {
        let state = AuthState::new(vec!["key1".to_string()]);
        assert!(state.auth_enabled());
    }

    #[test]
    fn test_is_public_path_health() {
        assert!(is_public_path("/health"));
    }

    #[test]
    fn test_is_public_path_ready() {
        assert!(is_public_path("/ready"));
    }

    #[test]
    fn test_is_public_path_other() {
        assert!(!is_public_path("/collections"));
        assert!(!is_public_path("/query"));
        assert!(!is_public_path("/health/extra"));
    }

    #[test]
    fn test_extract_bearer_token_valid() {
        assert_eq!(extract_bearer_token("Bearer my-key"), Some("my-key"));
        assert_eq!(extract_bearer_token("bearer my-key"), Some("my-key"));
        assert_eq!(extract_bearer_token("BEARER my-key"), Some("my-key"));
        assert_eq!(extract_bearer_token("  Bearer  my-key  "), Some("my-key"));
    }

    #[test]
    fn test_extract_bearer_token_invalid() {
        assert_eq!(extract_bearer_token("Basic abc123"), None);
        assert_eq!(extract_bearer_token("my-key"), None);
        assert_eq!(extract_bearer_token("Bearer"), None);
        assert_eq!(extract_bearer_token(""), None);
    }
}
