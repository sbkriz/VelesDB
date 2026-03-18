//! Shared handler helpers to reduce duplication across endpoint modules.

use axum::{http::StatusCode, response::IntoResponse, Json};

use crate::types::ErrorResponse;
use crate::AppState;

/// Look up a vector collection by name, returning a 404 response on miss.
#[allow(clippy::result_large_err)]
pub(crate) fn get_vector_collection_or_404(
    state: &AppState,
    name: &str,
) -> Result<velesdb_core::collection::VectorCollection, axum::response::Response> {
    state.db.get_vector_collection(name).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!(
                    "Collection '{}' not found or is not a vector collection",
                    name
                ),
            }),
        )
            .into_response()
    })
}
