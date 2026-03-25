//! Crash recovery: gap detection between vector storage and HNSW index.
//!
//! On [`Collection::open()`](super::super::Collection::open), vectors may
//! exist in storage but not in HNSW if a crash occurred between the storage
//! write and the HNSW batch insert (deferred indexer gap, delta buffer gap,
//! or normal insert gap).
//!
//! This module detects such gaps and re-indexes the missing vectors.
//!
//! ## Known limitation
//!
//! If a crash occurs between the HNSW delete and the storage delete being
//! persisted, a previously deleted vector may appear in storage but not in
//! HNSW — indistinguishable from an insert gap. Recovery will re-index the
//! deleted vector. This is an inherent trade-off without two-phase commit
//! and is acceptable because (a) the window is very small, and (b) a
//! resurrected vector is preferable to a silently lost one.

use crate::index::HnswIndex;
use crate::storage::{MmapStorage, VectorStorage};
use parking_lot::RwLock;
use std::sync::Arc;

/// Detects vectors in storage that are missing from the HNSW index and
/// re-indexes them.
///
/// Returns the number of recovered (re-indexed) vectors.
///
/// # Early exit
///
/// Returns `0` immediately if storage is empty or its count matches HNSW.
/// This heuristic may miss gaps in the theoretical case where a gap and an
/// HNSW orphan cancel out (e.g., one inserted + one deleted during the same
/// crash). This scenario requires two complementary failure modes and is
/// extremely unlikely in practice.
///
/// # Errors
///
/// Returns an error if vector retrieval from storage fails.
pub(crate) fn recover_hnsw_gap(
    vector_storage: &Arc<RwLock<MmapStorage>>,
    index: &Arc<HnswIndex>,
    dimension: usize,
) -> crate::error::Result<usize> {
    let storage = vector_storage.read();
    let storage_count = storage.len();
    let hnsw_count = index.len();

    if storage_count == 0 || storage_count == hnsw_count {
        return Ok(0);
    }

    let gap_ids = find_gap_ids(&storage, index);
    if gap_ids.is_empty() {
        return Ok(0);
    }

    let vectors = retrieve_valid_vectors(&storage, &gap_ids, dimension)?;
    let gap_total = gap_ids.len();
    drop(storage);

    let recovered = reindex_vectors(index, &vectors);
    tracing::warn!(
        recovered,
        gap_total,
        "Crash recovery: re-indexed gap vectors into HNSW"
    );
    Ok(recovered)
}

/// Returns storage IDs not present in the HNSW index.
fn find_gap_ids(storage: &MmapStorage, index: &HnswIndex) -> Vec<u64> {
    storage
        .ids()
        .into_iter()
        .filter(|id| !index.mappings.contains(*id))
        .collect()
}

/// Retrieves vectors for gap IDs, propagating IO errors.
///
/// Skips vectors with wrong dimension (corruption) or missing data
/// (concurrent deletion between `ids()` and `retrieve()`).
fn retrieve_valid_vectors(
    storage: &MmapStorage,
    gap_ids: &[u64],
    dimension: usize,
) -> crate::error::Result<Vec<(u64, Vec<f32>)>> {
    let mut vectors = Vec::with_capacity(gap_ids.len());
    for &id in gap_ids {
        match storage.retrieve(id) {
            Ok(Some(v)) if v.len() == dimension => vectors.push((id, v)),
            Ok(Some(v)) => tracing::warn!(
                id,
                expected = dimension,
                actual = v.len(),
                "Skipping gap vector with mismatched dimension"
            ),
            Ok(None) => {} // Deleted between ids() and retrieve()
            Err(e) => return Err(crate::error::Error::Storage(format!(
                "failed to retrieve gap vector {id}: {e}"
            ))),
        }
    }
    Ok(vectors)
}

/// Batch-inserts recovered vectors into the HNSW index.
fn reindex_vectors(index: &HnswIndex, vectors: &[(u64, Vec<f32>)]) -> usize {
    if vectors.is_empty() {
        return 0;
    }
    let refs: Vec<(u64, &[f32])> = vectors
        .iter()
        .map(|(id, v)| (*id, v.as_slice()))
        .collect();
    index.insert_batch_parallel(refs)
}
