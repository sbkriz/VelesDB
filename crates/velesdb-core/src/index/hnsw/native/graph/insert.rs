//! HNSW insert operations.

use super::super::distance::DistanceEngine;
use super::super::layer::{Layer, NodeId};
use super::NativeHnsw;
use std::borrow::Cow;
use std::sync::atomic::Ordering;

/// Result of [`NativeHnsw::allocate_batch`]: `(NodeId, layer)` pairs and
/// the pre-processed query vectors (normalized for cosine, borrowed otherwise).
type BatchAllocation<'a> = (Vec<(NodeId, usize)>, Vec<Cow<'a, [f32]>>);

impl<D: DistanceEngine> NativeHnsw<D> {
    /// Allocates vector storage if needed and pushes the vector, returning its node ID.
    ///
    /// # Errors
    ///
    /// Returns an error if storage allocation or push fails.
    fn allocate_and_store_vector(&self, vector: &[f32]) -> crate::error::Result<NodeId> {
        let mut guard = self.vectors.write();
        if guard.is_none() {
            *guard = Some(crate::perf_optimizations::ContiguousVectors::new(
                vector.len(),
                16,
            )?);
        }
        let storage = guard.as_mut().ok_or_else(|| {
            crate::error::Error::Internal("Vector storage missing after init".to_string())
        })?;
        let id = storage.len();
        storage.push(vector)?;
        Ok(id)
    }

    /// Pre-creates layers and allocates node capacity for an upcoming batch.
    ///
    /// Uses a statistical upper bound for the max expected layer:
    /// `ceil(log_M(total_nodes)) + 2`, capped at 15.
    // Reason: cast_precision_loss acceptable for statistical bound calculation
    // Reason: cast_possible_truncation result is capped at 15
    // Reason: cast_sign_loss log of positive numbers is positive
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub(in crate::index::hnsw::native) fn pre_expand_layers(&self, total_nodes: usize) {
        let max_layer = if self.max_connections > 1 && total_nodes > 1 {
            let log_m =
                (total_nodes as f64).ln() / (self.max_connections as f64).ln();
            (log_m.ceil() as usize + 2).min(15)
        } else {
            15
        };

        let mut layers = self.layers.write();
        while layers.len() <= max_layer {
            layers.push(Layer::new(total_nodes));
        }
        for layer in layers.iter_mut() {
            layer.ensure_capacity(total_nodes.saturating_sub(1));
        }
        self.pre_allocated_capacity
            .store(total_nodes, Ordering::Relaxed);
    }

    /// Ensures all layers up to `node_layer` exist and have capacity for `node_id`.
    fn expand_layers(&self, node_id: NodeId, node_layer: usize) {
        // Fast path: skip write lock if pre-allocation covers this insert
        if node_id < self.pre_allocated_capacity.load(Ordering::Relaxed)
            && node_layer < self.layers.read().len()
        {
            return;
        }
        // Slow path: acquire write lock (rare after pre-allocation)
        let mut layers = self.layers.write();
        while layers.len() <= node_layer {
            layers.push(Layer::new(node_id + 1));
        }
        for layer in layers.iter_mut() {
            layer.ensure_capacity(node_id);
        }
    }

    /// Inserts a vector into the index.
    ///
    /// Accepts a borrowed slice to avoid forcing callers to clone. For cosine
    /// metric with pre-normalization, a temporary copy is made internally;
    /// for all other metrics the slice is used directly (zero-copy).
    ///
    /// # Errors
    ///
    /// Returns an error if vector storage allocation or insertion fails.
    pub fn insert(&self, vector: &[f32]) -> crate::error::Result<NodeId> {
        let query = self.prepare_query(vector);

        let node_id = self.allocate_and_store_vector(&query)?;
        let node_layer = self.random_layer();
        self.expand_layers(node_id, node_layer);

        let entry_point = *self.entry_point.read();
        if let Some(ep) = entry_point {
            self.insert_with_entry_point(node_id, &query, node_layer, ep);
        }

        self.promote_entry_point(node_id, node_layer);
        self.count.fetch_add(1, Ordering::Relaxed);
        Ok(node_id)
    }

    /// Atomically updates the entry point if the index is empty or the node
    /// reaches a higher layer than the current maximum.
    ///
    /// Reads the true current state under the `entry_point` write lock,
    /// eliminating the TOCTOU race where two concurrent first-inserts both
    /// think the index is empty.
    pub(in crate::index::hnsw::native) fn promote_entry_point(
        &self,
        node_id: NodeId,
        node_layer: usize,
    ) {
        // Hold write lock before reading max_layer to serialize concurrent updates.
        // The lock release provides a happens-before guarantee for the Relaxed
        // max_layer store, ensuring the next acquirer sees the updated value.
        let mut ep_guard = self.entry_point.write();
        let current_max = self.max_layer.load(Ordering::Relaxed);
        if ep_guard.is_none() || node_layer > current_max {
            *ep_guard = Some(node_id);
            if node_layer > current_max {
                self.max_layer.store(node_layer, Ordering::Relaxed);
            }
        }
    }

    /// Batch-allocates vectors and assigns random layers.
    ///
    /// Returns `(assignments, processed_queries)` where:
    /// - `assignments`: `(NodeId, layer)` pairs for each vector
    /// - `processed_queries`: normalized query vectors (reusable in Phase B)
    ///
    /// This is Phase A of the two-phase batch insertion: all vectors are stored
    /// and layers expanded in single lock scopes. The caller is responsible for
    /// connecting nodes (Phase B) and updating `entry_point`/`count` (Phase C).
    ///
    /// # Errors
    ///
    /// Returns an error if vector storage allocation fails.
    pub(in crate::index::hnsw::native) fn allocate_batch<'a>(
        &self,
        vectors: &[&'a [f32]],
    ) -> crate::error::Result<BatchAllocation<'a>> {
        if vectors.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        // Normalize cosine vectors (Cow::Borrowed for non-cosine = zero-alloc)
        let processed: Vec<Cow<'a, [f32]>> =
            vectors.iter().map(|v| self.prepare_query(v)).collect();

        // Batch-store all vectors (single write lock)
        let first_id = {
            let mut guard = self.vectors.write();
            if guard.is_none() {
                *guard = Some(crate::perf_optimizations::ContiguousVectors::new(
                    processed[0].len(),
                    vectors.len().max(16),
                )?);
            }
            let storage = guard.as_mut().ok_or_else(|| {
                crate::error::Error::Internal("Vector storage missing after init".to_string())
            })?;
            let first = storage.len();
            let slices: Vec<&[f32]> = processed.iter().map(AsRef::as_ref).collect();
            storage.push_batch(&slices)?;
            first
        };

        // Assign random layers and expand (single write lock via pre_expand_layers)
        let total = first_id + vectors.len();
        self.pre_expand_layers(total);

        let assignments: Vec<(NodeId, usize)> = (0..vectors.len())
            .map(|i| (first_id + i, self.random_layer()))
            .collect();

        Ok((assignments, processed))
    }

    /// Greedy descent through upper HNSW layers above `node_layer` to find
    /// the best entry point for the target layers.
    pub(in crate::index::hnsw::native) fn greedy_descent_upper_layers(
        &self,
        query: &[f32],
        node_layer: usize,
        mut entry_point: NodeId,
    ) -> NodeId {
        let max_layer = self.max_layer.load(Ordering::Relaxed);
        for layer_idx in (node_layer + 1..=max_layer).rev() {
            entry_point = self.search_layer_single(query, entry_point, layer_idx);
        }
        entry_point
    }

    /// Connects a node into the HNSW graph at layers 0..=`node_layer`.
    ///
    /// Searches for neighbors at each layer, selects the best candidates,
    /// and creates bidirectional connections.
    pub(in crate::index::hnsw::native) fn connect_node(
        &self,
        node_id: NodeId,
        query: &[f32],
        node_layer: usize,
        mut entry_point: NodeId,
    ) {
        for layer_idx in (0..=node_layer).rev() {
            let max_conn = if layer_idx == 0 {
                self.max_connections_0
            } else {
                self.max_connections
            };
            // Stagnation disabled during construction (0) to ensure
            // optimal neighbor selection — see Devin review PR #336.
            let neighbors =
                self.search_layer(query, &[entry_point], self.ef_construction, layer_idx, 0);
            let selected = self.select_neighbors(&neighbors, max_conn);
            self.connect_neighbors_batch(node_id, &selected, layer_idx, max_conn);
            if !neighbors.is_empty() {
                entry_point = neighbors[0].0;
            }
        }
    }

    /// Performs the two-phase HNSW insertion when an entry point exists:
    /// 1. Greedy descent through upper layers above `node_layer`
    /// 2. Neighbor selection and bidirectional connection at layers 0..=`node_layer`
    #[inline]
    fn insert_with_entry_point(
        &self,
        node_id: NodeId,
        query: &[f32],
        node_layer: usize,
        ep: NodeId,
    ) {
        let current_ep = self.greedy_descent_upper_layers(query, node_layer, ep);
        self.connect_node(node_id, query, node_layer, current_ep);
    }
}
