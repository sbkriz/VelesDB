//! VectorIndex trait implementation for HnswIndex.

use super::HnswIndex;
use crate::distance::DistanceMetric;
use crate::index::hnsw::params::SearchQuality;
use crate::index::VectorIndex;
use crate::scored_result::ScoredResult;

impl VectorIndex for HnswIndex {
    #[inline]
    fn insert(&self, id: u64, vector: &[f32]) {
        assert_eq!(
            vector.len(),
            self.dimension,
            "Vector dimension mismatch: expected {}, got {}",
            self.dimension,
            vector.len()
        );

        let result = self.upsert_mapping(id);

        if let Err(e) = self.inner.write().insert((vector, result.idx)) {
            self.rollback_upsert(id, &result);
            tracing::error!("HnswIndex::insert failed for id={id}: {e}");
            return;
        }

        if self.enable_vector_storage {
            self.vectors.insert(result.idx, vector);
        }
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<ScoredResult> {
        self.search_with_quality(query, k, SearchQuality::Balanced)
    }

    /// Performs a **soft delete** of the vector.
    ///
    /// Removes the ID from mappings and cleans up stored vector data.
    /// The HNSW graph node becomes a tombstone, filtered out during search.
    ///
    /// For workloads with many deletions, consider periodic `vacuum()` to
    /// rebuild the graph and reclaim memory.
    fn remove(&self, id: u64) -> bool {
        if let Some(old_idx) = self.mappings.remove(id) {
            if self.enable_vector_storage {
                self.vectors.remove(old_idx);
            }
            true
        } else {
            false
        }
    }

    fn len(&self) -> usize {
        self.mappings.len()
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn metric(&self) -> DistanceMetric {
        self.metric
    }
}
