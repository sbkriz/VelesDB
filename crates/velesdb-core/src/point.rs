//! Point data structure representing a vector with metadata.

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::index::sparse::SparseVector;

/// A point in the vector database.
///
/// A point consists of:
/// - A unique identifier
/// - A dense vector (embedding)
/// - Optional payload (metadata)
/// - Optional sparse vector (e.g., SPLADE, BM25 term weights)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Point {
    /// Unique identifier for the point.
    pub id: u64,

    /// The dense vector embedding.
    pub vector: Vec<f32>,

    /// Optional JSON payload containing metadata.
    #[serde(default)]
    pub payload: Option<JsonValue>,

    /// Optional sparse vector for hybrid dense+sparse search.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sparse_vector: Option<SparseVector>,
}

impl Point {
    /// Creates a new point with the given ID, vector, and optional payload.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier
    /// * `vector` - Vector embedding
    /// * `payload` - Optional metadata
    #[must_use]
    pub fn new(id: u64, vector: Vec<f32>, payload: Option<JsonValue>) -> Self {
        Self {
            id,
            vector,
            payload,
            sparse_vector: None,
        }
    }

    /// Creates a new point without payload.
    #[must_use]
    pub fn without_payload(id: u64, vector: Vec<f32>) -> Self {
        Self::new(id, vector, None)
    }

    /// Creates a metadata-only point (no vector, only payload).
    ///
    /// Used for metadata-only collections that don't store vectors.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier
    /// * `payload` - Metadata (JSON value)
    #[must_use]
    pub fn metadata_only(id: u64, payload: JsonValue) -> Self {
        Self {
            id,
            vector: Vec::new(), // Empty vector
            payload: Some(payload),
            sparse_vector: None,
        }
    }

    /// Creates a point with both dense and sparse vectors.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier
    /// * `vector` - Dense vector embedding
    /// * `payload` - Optional metadata
    /// * `sparse_vector` - Optional sparse vector
    #[must_use]
    pub fn with_sparse(
        id: u64,
        vector: Vec<f32>,
        payload: Option<JsonValue>,
        sparse_vector: Option<SparseVector>,
    ) -> Self {
        Self {
            id,
            vector,
            payload,
            sparse_vector,
        }
    }

    /// Creates a sparse-only point (no dense vector).
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier
    /// * `sparse_vector` - The sparse vector
    /// * `payload` - Optional metadata
    #[must_use]
    pub fn sparse_only(id: u64, sparse_vector: SparseVector, payload: Option<JsonValue>) -> Self {
        Self {
            id,
            vector: Vec::new(),
            payload,
            sparse_vector: Some(sparse_vector),
        }
    }

    /// Returns `true` if this point has a sparse vector.
    #[must_use]
    pub const fn has_sparse_vector(&self) -> bool {
        self.sparse_vector.is_some()
    }

    /// Returns the dimension of the vector.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.vector.len()
    }

    /// Returns true if this point has no vector (metadata-only).
    #[must_use]
    pub fn is_metadata_only(&self) -> bool {
        self.vector.is_empty()
    }
}

/// A search result containing a point and its similarity score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// The matching point.
    pub point: Point,

    /// Similarity score (interpretation depends on the distance metric).
    pub score: f32,
}

impl SearchResult {
    /// Creates a new search result.
    #[must_use]
    pub const fn new(point: Point, score: f32) -> Self {
        Self { point, score }
    }
}
