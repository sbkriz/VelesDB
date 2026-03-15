//! Collection management handlers.

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use std::sync::Arc;

use crate::types::{
    CollectionConfigResponse, CollectionResponse, CollectionStatsResponse, ColumnStatsResponse,
    CreateCollectionRequest, ErrorResponse, GuardRailsConfigRequest, GuardRailsConfigResponse,
    IndexStatsResponse,
};
use crate::AppState;
use velesdb_core::{DistanceMetric, StorageMode};

/// List all collections.
#[utoipa::path(
    get,
    path = "/collections",
    tag = "collections",
    responses(
        (status = 200, description = "List of collections", body = Object)
    )
)]
pub async fn list_collections(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let collections = state.db.list_collections();
    Json(serde_json::json!({ "collections": collections }))
}

/// Create a new collection.
#[utoipa::path(
    post,
    path = "/collections",
    tag = "collections",
    request_body = CreateCollectionRequest,
    responses(
        (status = 201, description = "Collection created", body = Object),
        (status = 400, description = "Invalid request", body = ErrorResponse)
    )
)]
pub async fn create_collection(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateCollectionRequest>,
) -> impl IntoResponse {
    let metric = match req.metric.to_lowercase().as_str() {
        "cosine" => DistanceMetric::Cosine,
        "euclidean" | "l2" => DistanceMetric::Euclidean,
        "dot" | "dotproduct" | "ip" => DistanceMetric::DotProduct,
        "hamming" => DistanceMetric::Hamming,
        "jaccard" => DistanceMetric::Jaccard,
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: format!(
                        "Invalid metric: {}. Valid: cosine, euclidean, dot, hamming, jaccard",
                        req.metric
                    ),
                }),
            )
                .into_response()
        }
    };

    let storage_mode = match req.storage_mode.to_lowercase().as_str() {
        "full" | "f32" => StorageMode::Full,
        "sq8" | "int8" => StorageMode::SQ8,
        "binary" | "bit" => StorageMode::Binary,
        "pq" | "product_quantization" => StorageMode::ProductQuantization,
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: format!(
                        "Invalid storage_mode: {}. Valid: full, sq8, binary, pq",
                        req.storage_mode
                    ),
                }),
            )
                .into_response()
        }
    };

    let result = match req.collection_type.to_lowercase().as_str() {
        "metadata_only" | "metadata-only" | "metadata" => {
            state.db.create_metadata_collection(&req.name)
        }
        "graph" | "knowledge_graph" | "kg" => {
            use velesdb_core::GraphSchema;
            state
                .db
                .create_graph_collection(&req.name, GraphSchema::schemaless())
        }
        "vector" | "" => {
            let dimension = match req.dimension {
                Some(d) => d,
                None => {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(ErrorResponse {
                            error: "dimension is required for vector collections".to_string(),
                        }),
                    )
                        .into_response()
                }
            };
            if req.hnsw_m.is_some() || req.hnsw_ef_construction.is_some() {
                state.db.create_vector_collection_with_hnsw(
                    &req.name,
                    dimension,
                    metric,
                    storage_mode,
                    req.hnsw_m,
                    req.hnsw_ef_construction,
                )
            } else {
                state.db.create_vector_collection_with_options(
                    &req.name,
                    dimension,
                    metric,
                    storage_mode,
                )
            }
        }
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: format!(
                        "Invalid collection_type: {}. Valid: vector, graph, metadata_only",
                        req.collection_type
                    ),
                }),
            )
                .into_response()
        }
    };

    match result {
        Ok(()) => {
            let mut warnings = Vec::new();
            let is_vector = matches!(req.collection_type.to_lowercase().as_str(), "vector" | "");
            if is_vector {
                warnings.push("Collection dimension and metric are immutable after creation. If your embedding model changes, create a new collection and reindex data.");
                warnings.push("For first queries, start without strict filters/thresholds, then tighten progressively.");
            }

            (
                StatusCode::CREATED,
                Json(serde_json::json!({
                    "message": "Collection created",
                    "name": req.name,
                    "type": req.collection_type,
                    "warnings": warnings
                })),
            )
                .into_response()
        }
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
            .into_response(),
    }
}

/// Get collection information.
#[utoipa::path(
    get,
    path = "/collections/{name}",
    tag = "collections",
    params(
        ("name" = String, Path, description = "Collection name")
    ),
    responses(
        (status = 200, description = "Collection details", body = CollectionResponse),
        (status = 404, description = "Collection not found", body = ErrorResponse)
    )
)]
pub async fn get_collection(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    match state.db.get_collection(&name) {
        Some(collection) => {
            let config = collection.config();
            Json(CollectionResponse {
                name: config.name,
                dimension: config.dimension,
                metric: format!("{:?}", config.metric).to_lowercase(),
                point_count: config.point_count,
                storage_mode: format!("{:?}", config.storage_mode).to_lowercase(),
            })
            .into_response()
        }
        None => (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("Collection '{}' not found", name),
            }),
        )
            .into_response(),
    }
}

/// Get detailed collection configuration (HNSW params, storage mode, schema, etc.).
#[utoipa::path(
    get,
    path = "/collections/{name}/config",
    tag = "collections",
    params(
        ("name" = String, Path, description = "Collection name")
    ),
    responses(
        (status = 200, description = "Collection configuration", body = CollectionConfigResponse),
        (status = 404, description = "Collection not found", body = ErrorResponse)
    )
)]
pub async fn get_collection_config(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    match state.db.get_collection(&name) {
        Some(collection) => {
            let config = collection.config();
            let graph_schema = config
                .graph_schema
                .as_ref()
                .and_then(|gs| serde_json::to_value(gs).ok());

            Json(CollectionConfigResponse {
                name: config.name,
                dimension: config.dimension,
                metric: format!("{:?}", config.metric).to_lowercase(),
                storage_mode: format!("{:?}", config.storage_mode).to_lowercase(),
                point_count: config.point_count,
                metadata_only: config.metadata_only,
                graph_schema,
                embedding_dimension: config.embedding_dimension,
            })
            .into_response()
        }
        None => (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("Collection '{}' not found", name),
            }),
        )
            .into_response(),
    }
}

/// Run a quick sanity check for onboarding and troubleshooting.
#[utoipa::path(
    get,
    path = "/collections/{name}/sanity",
    tag = "collections",
    params(
        ("name" = String, Path, description = "Collection name")
    ),
    responses(
        (status = 200, description = "Collection sanity status", body = Object),
        (status = 404, description = "Collection not found", body = ErrorResponse)
    )
)]
pub async fn collection_sanity(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    match state.db.get_collection(&name) {
        Some(collection) => {
            let config = collection.config();
            let has_data = config.point_count > 0;
            Json(serde_json::json!({
                "collection": config.name,
                "dimension": config.dimension,
                "metric": format!("{:?}", config.metric).to_lowercase(),
                "point_count": config.point_count,
                "is_empty": collection.is_empty(),
                "checks": {
                    "has_vectors": has_data,
                    "search_ready": has_data,
                    "dimension_configured": config.dimension > 0
                },
                "diagnostics": {
                    "search_requests_total": state.onboarding_metrics.search_requests_total.load(std::sync::atomic::Ordering::Relaxed),
                    "dimension_mismatch_total": state.onboarding_metrics.dimension_mismatch_total.load(std::sync::atomic::Ordering::Relaxed),
                    "empty_search_results_total": state.onboarding_metrics.empty_search_results_total.load(std::sync::atomic::Ordering::Relaxed),
                    "filter_parse_errors_total": state.onboarding_metrics.filter_parse_errors_total.load(std::sync::atomic::Ordering::Relaxed)
                },
                "hints": if has_data {
                    vec![
                        "Run a search without strict filters first, then tighten filters progressively."
                    ]
                } else {
                    vec![
                        "Insert at least one known vector before evaluating search quality.",
                        "Verify you are querying the intended collection."
                    ]
                }
            }))
            .into_response()
        }
        None => (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("Collection '{}' not found", name),
            }),
        )
            .into_response(),
    }
}

/// Delete a collection.
#[utoipa::path(
    delete,
    path = "/collections/{name}",
    tag = "collections",
    params(
        ("name" = String, Path, description = "Collection name")
    ),
    responses(
        (status = 200, description = "Collection deleted", body = Object),
        (status = 404, description = "Collection not found", body = ErrorResponse)
    )
)]
pub async fn delete_collection(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    match state.db.delete_collection(&name) {
        Ok(()) => Json(serde_json::json!({
            "message": "Collection deleted",
            "name": name
        }))
        .into_response(),
        Err(e) => (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
            .into_response(),
    }
}

/// Check if a collection is empty.
#[utoipa::path(
    get,
    path = "/collections/{name}/empty",
    tag = "collections",
    params(
        ("name" = String, Path, description = "Collection name")
    ),
    responses(
        (status = 200, description = "Empty status", body = Object),
        (status = 404, description = "Collection not found", body = ErrorResponse)
    )
)]
pub async fn is_empty(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    match state.db.get_collection(&name) {
        Some(collection) => Json(serde_json::json!({
            "is_empty": collection.is_empty()
        }))
        .into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("Collection '{}' not found", name),
            }),
        )
            .into_response(),
    }
}

/// Flush pending changes to disk.
#[utoipa::path(
    post,
    path = "/collections/{name}/flush",
    tag = "collections",
    params(
        ("name" = String, Path, description = "Collection name")
    ),
    responses(
        (status = 200, description = "Flushed successfully", body = Object),
        (status = 404, description = "Collection not found", body = ErrorResponse),
        (status = 500, description = "Flush failed", body = ErrorResponse)
    )
)]
pub async fn flush_collection(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    match state.db.get_collection(&name) {
        Some(collection) => match collection.flush() {
            Ok(()) => Json(serde_json::json!({
                "message": "Flushed successfully",
                "collection": name
            }))
            .into_response(),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("Flush failed: {}", e),
                }),
            )
                .into_response(),
        },
        None => (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("Collection '{}' not found", name),
            }),
        )
            .into_response(),
    }
}

/// Analyze a collection, computing and persisting statistics.
#[utoipa::path(
    post,
    path = "/collections/{name}/analyze",
    tag = "collections",
    params(
        ("name" = String, Path, description = "Collection name")
    ),
    responses(
        (status = 200, description = "Collection analyzed", body = CollectionStatsResponse),
        (status = 404, description = "Collection not found", body = ErrorResponse),
        (status = 500, description = "Analysis failed", body = ErrorResponse)
    )
)]
pub async fn analyze_collection(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    match state.db.analyze_collection(&name) {
        Ok(stats) => {
            let response = map_stats_to_response(&stats);
            (StatusCode::OK, Json(response)).into_response()
        }
        Err(e) => {
            let status = if e.to_string().contains("not found") {
                StatusCode::NOT_FOUND
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            };
            (
                status,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            )
                .into_response()
        }
    }
}

/// Get cached collection statistics (returns 404 if never analyzed).
#[utoipa::path(
    get,
    path = "/collections/{name}/stats",
    tag = "collections",
    params(
        ("name" = String, Path, description = "Collection name")
    ),
    responses(
        (status = 200, description = "Collection statistics", body = CollectionStatsResponse),
        (status = 404, description = "No statistics available", body = ErrorResponse),
        (status = 500, description = "Failed to read stats", body = ErrorResponse)
    )
)]
pub async fn get_collection_stats(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    match state.db.get_collection_stats(&name) {
        Ok(Some(stats)) => {
            let response = map_stats_to_response(&stats);
            (StatusCode::OK, Json(response)).into_response()
        }
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!(
                    "No stats for '{name}'. Run POST /collections/{name}/analyze first."
                ),
            }),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
            .into_response(),
    }
}

/// Get current guard-rails configuration.
#[utoipa::path(
    get,
    path = "/guardrails",
    tag = "guardrails",
    responses(
        (status = 200, description = "Current guard-rails config", body = GuardRailsConfigResponse)
    )
)]
pub async fn get_guardrails(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let limits = state.query_limits.read();
    Json(limits_to_response(&limits))
}

/// Update guard-rails configuration (partial update).
#[utoipa::path(
    put,
    path = "/guardrails",
    tag = "guardrails",
    request_body = GuardRailsConfigRequest,
    responses(
        (status = 200, description = "Updated guard-rails config", body = GuardRailsConfigResponse)
    )
)]
pub async fn update_guardrails(
    State(state): State<Arc<AppState>>,
    Json(req): Json<GuardRailsConfigRequest>,
) -> impl IntoResponse {
    let mut limits = state.query_limits.write();
    apply_guardrails_update(&mut limits, &req);
    Json(limits_to_response(&limits))
}

/// Convert `QueryLimits` to the REST response type.
fn limits_to_response(limits: &velesdb_core::guardrails::QueryLimits) -> GuardRailsConfigResponse {
    GuardRailsConfigResponse {
        max_depth: limits.max_depth,
        max_cardinality: limits.max_cardinality,
        memory_limit_bytes: limits.memory_limit_bytes,
        timeout_ms: limits.timeout_ms,
        rate_limit_qps: limits.rate_limit_qps,
        circuit_failure_threshold: limits.circuit_failure_threshold,
        circuit_recovery_seconds: limits.circuit_recovery_seconds,
    }
}

/// Convert core `CollectionStats` to the REST response type.
fn map_stats_to_response(
    stats: &velesdb_core::collection::stats::CollectionStats,
) -> CollectionStatsResponse {
    let column_stats = stats
        .column_stats
        .iter()
        .map(|(k, v)| {
            (
                k.clone(),
                ColumnStatsResponse {
                    name: v.name.clone(),
                    null_count: v.null_count,
                    distinct_count: v.distinct_count,
                    min_value: v.min_value.clone(),
                    max_value: v.max_value.clone(),
                    avg_size_bytes: v.avg_size_bytes,
                },
            )
        })
        .collect();

    let index_stats = stats
        .index_stats
        .iter()
        .map(|(k, v)| {
            (
                k.clone(),
                IndexStatsResponse {
                    name: v.name.clone(),
                    index_type: v.index_type.clone(),
                    entry_count: v.entry_count,
                    depth: v.depth,
                    size_bytes: v.size_bytes,
                },
            )
        })
        .collect();

    CollectionStatsResponse {
        total_points: stats.total_points,
        total_size_bytes: stats.total_size_bytes,
        row_count: stats.row_count,
        deleted_count: stats.deleted_count,
        avg_row_size_bytes: stats.avg_row_size_bytes,
        payload_size_bytes: stats.payload_size_bytes,
        last_analyzed_epoch_ms: stats.last_analyzed_epoch_ms,
        column_stats,
        index_stats,
    }
}

/// Apply partial update fields to query limits.
fn apply_guardrails_update(
    limits: &mut velesdb_core::guardrails::QueryLimits,
    req: &GuardRailsConfigRequest,
) {
    if let Some(v) = req.max_depth {
        limits.max_depth = v;
    }
    if let Some(v) = req.max_cardinality {
        limits.max_cardinality = v;
    }
    if let Some(v) = req.memory_limit_bytes {
        limits.memory_limit_bytes = v;
    }
    if let Some(v) = req.timeout_ms {
        limits.timeout_ms = v;
    }
    if let Some(v) = req.rate_limit_qps {
        limits.rate_limit_qps = v;
    }
    if let Some(v) = req.circuit_failure_threshold {
        limits.circuit_failure_threshold = v;
    }
    if let Some(v) = req.circuit_recovery_seconds {
        limits.circuit_recovery_seconds = v;
    }
}
