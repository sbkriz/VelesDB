//! `VectorCollection`: newtype wrapper around `Collection` for vector workloads.
//!
//! This type provides a stable, typed API for vector collections.
//! Internally it delegates 100% to the `Collection` executor to avoid
//! any data synchronisation issues between separate storage layers.

use std::collections::HashMap;
use std::path::PathBuf;

use crate::collection::types::{Collection, CollectionConfig};
use crate::distance::DistanceMetric;
use crate::error::Result;
use crate::point::{Point, SearchResult};
use crate::quantization::StorageMode;

/// A vector collection combining HNSW search, payload storage, and full-text search.
///
/// `VectorCollection` is a typed newtype over `Collection` that provides
/// a stable public API for vector workloads. All storage operations delegate
/// to the single `inner: Collection` instance — no dual-storage desync.
///
/// # Examples
///
/// ```rust,no_run
/// use velesdb_core::{VectorCollection, DistanceMetric, Point, StorageMode};
/// use serde_json::json;
///
/// let coll = VectorCollection::create(
///     "./data/docs".into(),
///     "docs",
///     768,
///     DistanceMetric::Cosine,
///     StorageMode::Full,
/// )?;
///
/// coll.upsert(vec![
///     Point::new(1, vec![0.1; 768], Some(json!({"title": "Hello"}))),
/// ])?;
///
/// let results = coll.search(&vec![0.1; 768], 10)?;
/// # Ok::<(), velesdb_core::Error>(())
/// ```
#[derive(Clone)]
pub struct VectorCollection {
    /// Single source of truth — all operations delegate here.
    pub(crate) inner: Collection,
}

impl VectorCollection {
    // -------------------------------------------------------------------------
    // Lifecycle
    // -------------------------------------------------------------------------

    /// Creates a new `VectorCollection` at the given path.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be created or storage fails.
    pub fn create(
        path: PathBuf,
        _name: &str,
        dimension: usize,
        metric: DistanceMetric,
        storage_mode: StorageMode,
    ) -> Result<Self> {
        Ok(Self {
            inner: Collection::create_with_options(path, dimension, metric, storage_mode)?,
        })
    }

    /// Opens an existing `VectorCollection` from disk.
    ///
    /// # Errors
    ///
    /// Returns an error if the config file cannot be read or storage cannot be opened.
    pub fn open(path: PathBuf) -> Result<Self> {
        Ok(Self {
            inner: Collection::open(path)?,
        })
    }

    /// Flushes all engines to disk and saves the config.
    ///
    /// # Errors
    ///
    /// Returns an error if any flush operation fails.
    pub fn flush(&self) -> Result<()> {
        self.inner.flush()
    }

    // -------------------------------------------------------------------------
    // Collection metadata — all delegate to inner
    // -------------------------------------------------------------------------

    /// Returns the collection name.
    #[must_use]
    pub fn name(&self) -> String {
        self.inner.config().name
    }

    /// Returns the vector dimension.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.inner.config().dimension
    }

    /// Returns the distance metric.
    #[must_use]
    pub fn metric(&self) -> DistanceMetric {
        self.inner.config().metric
    }

    /// Returns the storage mode.
    #[must_use]
    pub fn storage_mode(&self) -> StorageMode {
        self.inner.config().storage_mode
    }

    /// Returns the number of points in the collection.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if the collection is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns all point IDs.
    #[must_use]
    pub fn all_ids(&self) -> Vec<u64> {
        self.inner.all_ids()
    }

    /// Returns the current collection config.
    #[must_use]
    pub fn config(&self) -> CollectionConfig {
        self.inner.config()
    }

    // -------------------------------------------------------------------------
    // CRUD — all delegate to inner
    // -------------------------------------------------------------------------

    /// Bulk insert optimized for high-throughput import.
    ///
    /// # Errors
    ///
    /// Returns an error if any point has a mismatched dimension.
    pub fn upsert_bulk(&self, points: &[Point]) -> Result<usize> {
        self.inner.upsert_bulk(points)
    }

    /// Inserts or updates points.
    ///
    /// # Errors
    ///
    /// Returns an error if the dimension mismatches or storage fails.
    pub fn upsert(&self, points: impl IntoIterator<Item = Point>) -> Result<()> {
        self.inner.upsert(points)
    }

    /// Retrieves points by IDs.
    #[must_use]
    pub fn get(&self, ids: &[u64]) -> Vec<Option<Point>> {
        self.inner.get(ids)
    }

    /// Deletes points by IDs.
    ///
    /// # Errors
    ///
    /// Returns an error if storage operations fail.
    pub fn delete(&self, ids: &[u64]) -> Result<()> {
        self.inner.delete(ids)
    }

    // -------------------------------------------------------------------------
    // Search — all delegate to inner
    // -------------------------------------------------------------------------

    /// Performs kNN vector search.
    /// # Errors
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        self.inner.search(query, k)
    }

    /// Performs full-text BM25 search.
    #[must_use]
    pub fn text_search(&self, query: &str, k: usize) -> Vec<SearchResult> {
        self.inner.text_search(query, k)
    }

    /// kNN search with explicit ef_search override.
    /// # Errors
    pub fn search_with_ef(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<SearchResult>> {
        self.inner.search_with_ef(query, k, ef_search)
    }

    /// kNN search with metadata filter.
    /// # Errors
    pub fn search_with_filter(
        &self,
        query: &[f32],
        k: usize,
        filter: &crate::filter::Filter,
    ) -> Result<Vec<SearchResult>> {
        self.inner.search_with_filter(query, k, filter)
    }

    /// Returns `(id, score)` pairs without payload hydration.
    /// # Errors
    pub fn search_ids(&self, query: &[f32], k: usize) -> Result<Vec<(u64, f32)>> {
        self.inner.search_ids(query, k)
    }

    /// Full-text search with metadata filter.
    #[must_use]
    pub fn text_search_with_filter(
        &self,
        query: &str,
        k: usize,
        filter: &crate::filter::Filter,
    ) -> Vec<SearchResult> {
        self.inner.text_search_with_filter(query, k, filter)
    }

    /// Hybrid search (vector + BM25 with RRF fusion).
    /// # Errors
    pub fn hybrid_search(
        &self,
        vector: &[f32],
        text: &str,
        k: usize,
        alpha: Option<f32>,
    ) -> Result<Vec<SearchResult>> {
        self.inner.hybrid_search(vector, text, k, alpha)
    }

    /// Hybrid search with metadata filter.
    /// # Errors
    pub fn hybrid_search_with_filter(
        &self,
        vector: &[f32],
        text: &str,
        k: usize,
        alpha: Option<f32>,
        filter: &crate::filter::Filter,
    ) -> Result<Vec<SearchResult>> {
        self.inner
            .hybrid_search_with_filter(vector, text, k, alpha, filter)
    }

    /// Batch search with per-query filters.
    /// # Errors
    pub fn search_batch_with_filters(
        &self,
        queries: &[&[f32]],
        k: usize,
        filters: &[Option<crate::filter::Filter>],
    ) -> Result<Vec<Vec<SearchResult>>> {
        self.inner.search_batch_with_filters(queries, k, filters)
    }

    /// Multi-query search (multiple vectors fused).
    /// # Errors
    pub fn multi_query_search(
        &self,
        queries: &[&[f32]],
        k: usize,
        strategy: crate::fusion::FusionStrategy,
        filter: Option<&crate::filter::Filter>,
    ) -> Result<Vec<SearchResult>> {
        self.inner.multi_query_search(queries, k, strategy, filter)
    }

    // -------------------------------------------------------------------------
    // Statistics / misc
    // -------------------------------------------------------------------------

    /// Returns CBO statistics.
    #[must_use]
    pub fn get_stats(&self) -> crate::collection::stats::CollectionStats {
        self.inner.get_stats()
    }

    /// Returns `true` if the collection is a metadata-only collection.
    #[must_use]
    pub fn is_metadata_only(&self) -> bool {
        self.inner.is_metadata_only()
    }

    /// Returns a reference to the inner `Collection`.
    ///
    /// Provides access to methods not yet promoted to `VectorCollection`'s public API.
    #[doc(hidden)]
    #[must_use]
    pub fn as_collection(&self) -> &crate::collection::Collection {
        &self.inner
    }

    // -------------------------------------------------------------------------
    // VelesQL
    // -------------------------------------------------------------------------

    /// Executes a `VelesQL` query.
    /// # Errors
    pub fn execute_query(
        &self,
        query: &crate::velesql::Query,
        params: &HashMap<String, serde_json::Value>,
    ) -> Result<Vec<SearchResult>> {
        self.inner.execute_query(query, params)
    }

    /// Executes a raw VelesQL string.
    /// # Errors
    pub fn execute_query_str(
        &self,
        sql: &str,
        params: &HashMap<String, serde_json::Value>,
    ) -> Result<Vec<SearchResult>> {
        let query = self
            .inner
            .query_cache
            .parse(sql)
            .map_err(|e| crate::error::Error::Query(e.to_string()))?;
        self.inner.execute_query(&query, params)
    }
}
