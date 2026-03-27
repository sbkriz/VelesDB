//! Bulk CRUD operations for Collection (`upsert_bulk`, `upsert_bulk_from_raw`).
//!
//! Extracted from `crud.rs` (Issue #425) to keep each file under 500 NLOC.
//! These methods are optimized for high-throughput import with parallel I/O.

use crate::collection::types::Collection;
use crate::error::{Error, Result};
use crate::point::Point;
use crate::storage::VectorStorage;
use crate::validation::validate_dimension_match;

use std::collections::BTreeMap;

impl Collection {
    /// Bulk insert optimized for high-throughput import.
    ///
    /// # Performance
    ///
    /// This method is optimized for bulk loading:
    /// - Uses parallel HNSW insertion (rayon)
    /// - Parallel payload + vector I/O via `rayon::join` (Issue #424)
    /// - Single flush at the end (not per-point)
    /// - No HNSW index save (deferred for performance)
    /// - ~15x faster than previous sequential approach on large batches (5000+)
    /// - Benchmark: 25-30 Kvec/s on 768D vectors
    ///
    /// # Errors
    ///
    /// Returns an error if any point has a mismatched dimension.
    pub fn upsert_bulk(&self, points: &[Point]) -> Result<usize> {
        if points.is_empty() {
            return Ok(0);
        }

        let dimension = self.config.read().dimension;
        for point in points {
            validate_dimension_match(dimension, point.dimension())?;
        }

        let vector_refs: Vec<(u64, &[f32])> =
            points.iter().map(|p| (p.id, p.vector.as_slice())).collect();
        let sparse_batch = Self::collect_sparse_batch(points);

        self.store_vectors_and_payloads(&vector_refs, points)?;

        let inserted = self.bulk_index_or_defer(vector_refs);
        self.config.write().point_count = self.vector_storage.read().len();

        self.apply_sparse_batch_bulk(&sparse_batch)?;
        self.invalidate_caches_and_bump_generation();

        Ok(inserted)
    }

    /// Writes vectors and payloads to storage (parallel with rayon when available).
    fn store_vectors_and_payloads(
        &self,
        vector_refs: &[(u64, &[f32])],
        points: &[Point],
    ) -> Result<()> {
        #[cfg(feature = "persistence")]
        {
            let (vec_result, pay_result) = rayon::join(
                || self.bulk_store_vectors(vector_refs),
                || self.bulk_store_payloads(points),
            );
            vec_result?;
            pay_result?;
        }

        #[cfg(not(feature = "persistence"))]
        {
            self.bulk_store_vectors(vector_refs)?;
            self.bulk_store_payloads(points)?;
        }

        Ok(())
    }

    /// Bulk insert from contiguous flat slices (zero-copy from numpy / FFI).
    ///
    /// Accepts a flat `f32` slice of shape `(n, dimension)` in row-major order
    /// plus a matching `u64` ID slice of length `n`. This avoids per-row
    /// `Vec<f32>` allocation that `upsert_bulk` requires through `Point`.
    ///
    /// # Performance
    ///
    /// Eliminates `n * dimension * 4` bytes of intermediate copies compared
    /// to the `Point`-based `upsert_bulk` path. For 100K vectors at 768D
    /// this saves ~293 MB of heap allocations.
    ///
    /// # Errors
    ///
    /// - Returns [`Error::InvalidVector`] if `vectors.len() != ids.len() * dimension`.
    /// - Returns [`Error::DimensionMismatch`] if `dimension` does not match the collection.
    pub fn upsert_bulk_from_raw(
        &self,
        vectors: &[f32],
        ids: &[u64],
        dimension: usize,
        payloads: Option<&[Option<serde_json::Value>]>,
    ) -> Result<usize> {
        let n = ids.len();
        if n == 0 {
            return Ok(0);
        }

        // Validate inputs BEFORE any state mutation.
        self.validate_raw_inputs(vectors, ids, dimension, payloads)?;

        // Build (id, &[f32]) pairs by slicing the flat buffer — zero copy.
        let vector_refs: Vec<(u64, &[f32])> = ids
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, &vectors[i * dimension..(i + 1) * dimension]))
            .collect();

        // Payload entries for batch WAL write (only ids that have payloads).
        let payload_entries: Vec<(u64, &serde_json::Value)> = payloads
            .into_iter()
            .flat_map(|ps| {
                ps.iter()
                    .enumerate()
                    .filter_map(|(i, opt)| opt.as_ref().map(|val| (ids[i], val)))
            })
            .collect();

        self.store_vectors_and_payload_entries(&vector_refs, &payload_entries)?;

        self.update_text_index_from_raw(ids, payloads);

        let inserted = self.bulk_index_or_defer(vector_refs);
        self.config.write().point_count = self.vector_storage.read().len();
        self.invalidate_caches_and_bump_generation();

        Ok(inserted)
    }

    /// Validates raw bulk-insert inputs before any state mutation.
    fn validate_raw_inputs(
        &self,
        vectors: &[f32],
        ids: &[u64],
        dimension: usize,
        payloads: Option<&[Option<serde_json::Value>]>,
    ) -> Result<()> {
        let n = ids.len();
        let expected_len = n.checked_mul(dimension).ok_or_else(|| {
            Error::InvalidVector(format!(
                "overflow computing {n} * {dimension} for flat vector length"
            ))
        })?;
        if vectors.len() != expected_len {
            return Err(Error::InvalidVector(format!(
                "flat vectors length {} != ids.len() ({n}) * dimension ({dimension}) = {expected_len}",
                vectors.len()
            )));
        }
        if let Some(ps) = payloads {
            if ps.len() != n {
                return Err(Error::InvalidVector(format!(
                    "payloads length ({}) must match ids length ({n})",
                    ps.len()
                )));
            }
        }
        let collection_dim = self.config.read().dimension;
        validate_dimension_match(collection_dim, dimension)?;
        Ok(())
    }

    /// Stores pre-built payload entries via batch WAL write + flush.
    ///
    /// Extracted from `bulk_store_payloads` to accept `(u64, &Value)` pairs
    /// directly, avoiding the need to reconstruct `Point` structs.
    fn bulk_store_payload_entries(&self, entries: &[(u64, &serde_json::Value)]) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }
        self.payload_storage.write().store_batch(entries)?;
        Ok(())
    }

    /// Writes vectors and raw payload entries to storage (parallel when available).
    fn store_vectors_and_payload_entries(
        &self,
        vector_refs: &[(u64, &[f32])],
        payload_entries: &[(u64, &serde_json::Value)],
    ) -> Result<()> {
        #[cfg(feature = "persistence")]
        {
            let (vec_result, pay_result) = rayon::join(
                || self.bulk_store_vectors(vector_refs),
                || self.bulk_store_payload_entries(payload_entries),
            );
            vec_result?;
            pay_result?;
        }

        #[cfg(not(feature = "persistence"))]
        {
            self.bulk_store_vectors(vector_refs)?;
            self.bulk_store_payload_entries(payload_entries)?;
        }

        Ok(())
    }

    /// Updates BM25 text index from raw payload slices.
    ///
    /// Points with `Some(payload)` get their text indexed.
    /// Points with `None` payload get their stale BM25 entry removed
    /// (consistent with `update_text_index` in `crud.rs`).
    fn update_text_index_from_raw(
        &self,
        ids: &[u64],
        payloads: Option<&[Option<serde_json::Value>]>,
    ) {
        let Some(ps) = payloads else { return };
        for (i, opt) in ps.iter().enumerate() {
            if let Some(payload) = opt {
                let text = Self::extract_text_from_payload(payload);
                if !text.is_empty() {
                    self.text_index.add_document(ids[i], &text);
                }
            } else {
                self.text_index.remove_document(ids[i]);
            }
        }
    }

    /// Collects sparse vectors grouped by index name for batch insert.
    fn collect_sparse_batch(
        points: &[Point],
    ) -> BTreeMap<String, Vec<(u64, crate::index::sparse::SparseVector)>> {
        let mut batch: BTreeMap<String, Vec<(u64, crate::index::sparse::SparseVector)>> =
            BTreeMap::new();
        for point in points {
            if let Some(sv_map) = &point.sparse_vectors {
                for (name, sv) in sv_map {
                    batch
                        .entry(name.clone())
                        .or_default()
                        .push((point.id, sv.clone()));
                }
            }
        }
        batch
    }

    /// Stores vectors in bulk via batch WAL + mmap write.
    fn bulk_store_vectors(&self, vectors: &[(u64, &[f32])]) -> Result<()> {
        let mut storage = self.vector_storage.write();
        storage.store_batch(vectors)?;
        storage.flush()?;
        Ok(())
    }

    /// Stores payloads and updates BM25 text index in bulk.
    ///
    /// Uses `LogPayloadStorage::store_batch()` for a single WAL sync instead
    /// of per-point fsync, improving bulk insert throughput by 10-50x.
    fn bulk_store_payloads(&self, points: &[Point]) -> Result<()> {
        let entries: Vec<(u64, &serde_json::Value)> = points
            .iter()
            .filter_map(|p| p.payload.as_ref().map(|pl| (p.id, pl)))
            .collect();

        self.payload_storage.write().store_batch(&entries)?;

        // Issue #425: BM25 skip — when no point has a payload AND the BM25
        // index is empty, skip the text index loop entirely. The bulk path
        // inserts fresh points (no old documents to remove), so the loop
        // body would be a no-op for every point.
        if !entries.is_empty() || !self.text_index.is_empty() {
            for point in points {
                Self::update_text_index(&self.text_index, point);
            }
        }

        Ok(())
    }

    /// Applies sparse batch with WAL-before-apply for bulk insert.
    fn apply_sparse_batch_bulk(
        &self,
        sparse_batch: &BTreeMap<String, Vec<(u64, crate::index::sparse::SparseVector)>>,
    ) -> Result<()> {
        if sparse_batch.is_empty() {
            return Ok(());
        }
        #[cfg(feature = "persistence")]
        {
            for (name, docs) in sparse_batch {
                let wal_path =
                    crate::index::sparse::persistence::wal_path_for_name(&self.path, name);
                for (point_id, sv) in docs {
                    crate::index::sparse::persistence::wal_append_upsert(&wal_path, *point_id, sv)?;
                }
            }
        }
        let mut indexes = self.sparse_indexes.write();
        for (name, docs) in sparse_batch {
            let idx = indexes.entry(name.clone()).or_default();
            idx.insert_batch_chunk(docs);
        }
        Ok(())
    }
}
