//! Shared upsert-mapping logic for HNSW index variants.
//!
//! Both `HnswIndex` and `NativeHnswIndex` use identical mapping upsert
//! semantics. This module provides a single implementation to avoid
//! duplication.

use super::sharded_mappings::ShardedMappings;
use super::sharded_vectors::ShardedVectors;

/// Result of an upsert mapping operation, carrying rollback information.
///
/// On success the caller uses `idx` as the internal HNSW index for the new
/// graph node. On graph-insert failure, the caller passes this struct to
/// [`rollback_upsert`] to restore the previous state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct UpsertResult {
    /// Newly allocated internal index for the vector.
    pub idx: usize,
    /// Previous internal index if this was an update (not a fresh insert).
    pub old_idx: Option<usize>,
}

/// Registers an ID with upsert semantics and cleans up stale vector data.
///
/// If the ID already existed, the old mapping is replaced and the stale
/// sidecar vector (if stored) is removed from `ShardedVectors`.
///
/// Returns an [`UpsertResult`] containing the new index and optional old
/// index for rollback purposes.
#[must_use]
pub(crate) fn upsert_mapping(
    mappings: &ShardedMappings,
    vectors: &ShardedVectors,
    enable_vector_storage: bool,
    id: u64,
) -> UpsertResult {
    let (idx, old_idx) = mappings.register_or_replace(id);
    if let Some(old) = old_idx {
        if enable_vector_storage {
            vectors.remove(old);
        }
    }
    UpsertResult { idx, old_idx }
}

/// Rolls back mapping state after a failed graph insertion.
///
/// Removes the newly-allocated mapping and, if this was an update,
/// restores the previous mapping so the point remains searchable
/// through its old graph node.
///
/// **Transient gap**: Between `remove` and `restore`, the ID has no
/// mapping for a brief window (nanoseconds). A concurrent search during
/// this window will not find the point. This only occurs on graph-insert
/// failure, which is rare (allocation error).
///
/// **Sidecar loss**: The old sidecar vector (in `ShardedVectors`) was
/// already removed by [`upsert_mapping`] and cannot be cheaply restored.
/// The HNSW graph still holds the vector data in `ContiguousVectors` for
/// traversal, so the point remains searchable -- only sidecar reranking
/// precision is lost for the affected point until the next successful
/// upsert.
pub(crate) fn rollback_upsert(mappings: &ShardedMappings, id: u64, result: &UpsertResult) {
    mappings.remove(id);
    if let Some(old_idx) = result.old_idx {
        mappings.restore(id, old_idx);
    }
}
