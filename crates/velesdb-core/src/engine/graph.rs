//! Graph engine: edge store + property/range indexes + BFS/DFS traversal.
//!
//! # Lock ordering (internal, ascending)
//!
//!   1. `edge_store`      (`RwLock<EdgeStore>`)
//!   2. `property_index`  (`RwLock<PropertyIndex>`)
//!   3. `range_index`     (`RwLock<RangeIndex>`)
//!
//! Never acquire in reverse order.

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;

use parking_lot::RwLock;

use crate::collection::graph::{
    bfs_stream, EdgeStore, GraphEdge, PropertyIndex, RangeIndex, StreamingConfig, TraversalConfig,
    TraversalResult,
};
use crate::error::Result;

/// Encapsulates edge storage, property/range indexes, and graph traversal.
///
/// This is `pub(crate)` — consumers use it through `GraphCollection`.
#[derive(Clone)]
pub(crate) struct GraphEngine {
    /// Bidirectional edge store.
    pub(crate) edge_store: Arc<RwLock<EdgeStore>>,
    /// O(1) equality lookups on node properties.
    pub(crate) property_index: Arc<RwLock<PropertyIndex>>,
    /// O(log n) range queries on node properties.
    pub(crate) range_index: Arc<RwLock<RangeIndex>>,
}

impl GraphEngine {
    /// Creates a new empty `GraphEngine`.
    pub(crate) fn new() -> Self {
        Self {
            edge_store: Arc::new(RwLock::new(EdgeStore::new())),
            property_index: Arc::new(RwLock::new(PropertyIndex::new())),
            range_index: Arc::new(RwLock::new(RangeIndex::new())),
        }
    }

    /// Opens an existing `GraphEngine`, loading indexes from disk if present.
    pub(crate) fn open(path: &Path) -> Self {
        let property_index = Self::load_property_index(path);
        let range_index = Self::load_range_index(path);
        Self {
            edge_store: Arc::new(RwLock::new(EdgeStore::new())),
            property_index: Arc::new(RwLock::new(property_index)),
            range_index: Arc::new(RwLock::new(range_index)),
        }
    }

    /// Adds an edge to the store.
    ///
    /// # Errors
    ///
    /// Returns `Error::EdgeExists` if an edge with the same ID already exists.
    pub(crate) fn add_edge(&self, edge: GraphEdge) -> Result<()> {
        self.edge_store.write().add_edge(edge)
    }

    /// Returns all edges, optionally filtered by label.
    pub(crate) fn get_edges(&self, label: Option<&str>) -> Vec<GraphEdge> {
        let store = self.edge_store.read();
        if let Some(lbl) = label {
            store.get_edges_by_label(lbl).into_iter().cloned().collect()
        } else {
            store.all_edges().into_iter().cloned().collect()
        }
    }

    /// Returns all edges for a source node.
    pub(crate) fn get_outgoing(&self, node_id: u64) -> Vec<GraphEdge> {
        self.edge_store
            .read()
            .get_outgoing(node_id)
            .into_iter()
            .cloned()
            .collect()
    }

    /// Returns all edges targeting a node.
    pub(crate) fn get_incoming(&self, node_id: u64) -> Vec<GraphEdge> {
        self.edge_store
            .read()
            .get_incoming(node_id)
            .into_iter()
            .cloned()
            .collect()
    }

    /// Returns `(in_degree, out_degree)` for a node.
    pub(crate) fn node_degree(&self, node_id: u64) -> (usize, usize) {
        let store = self.edge_store.read();
        (
            store.get_incoming(node_id).len(),
            store.get_outgoing(node_id).len(),
        )
    }

    /// BFS traversal using the core `bfs_stream` iterator.
    pub(crate) fn traverse_bfs(
        &self,
        source_id: u64,
        config: &TraversalConfig,
    ) -> Vec<TraversalResult> {
        let store = self.edge_store.read();
        let streaming = StreamingConfig {
            max_depth: config.max_depth,
            rel_types: config.rel_types.clone(),
            limit: Some(config.limit),
            max_visited_size: 100_000,
        };
        bfs_stream(&store, source_id, streaming)
            .take(config.limit)
            .collect()
    }

    /// DFS traversal (iterative, avoids stack overflow on deep graphs).
    pub(crate) fn traverse_dfs(
        &self,
        source_id: u64,
        config: &TraversalConfig,
    ) -> Vec<TraversalResult> {
        let store = self.edge_store.read();
        let rel_filter: HashSet<&str> = config.rel_types.iter().map(String::as_str).collect();

        let mut results = Vec::new();
        let mut visited: HashSet<u64> = HashSet::new();
        let mut stack: Vec<(u64, u32, Vec<u64>)> = vec![(source_id, 0, Vec::new())];

        while let Some((node_id, depth, path)) = stack.pop() {
            if results.len() >= config.limit {
                break;
            }
            if visited.contains(&node_id) {
                continue;
            }
            visited.insert(node_id);

            if depth >= config.min_depth && depth > 0 {
                results.push(TraversalResult::new(node_id, path.clone(), depth));
                if results.len() >= config.limit {
                    break;
                }
            }

            if depth < config.max_depth {
                let edges = store.get_outgoing(node_id);
                for edge in edges.into_iter().rev() {
                    if !rel_filter.is_empty() && !rel_filter.contains(edge.label()) {
                        continue;
                    }
                    if visited.contains(&edge.target()) {
                        continue;
                    }
                    let mut new_path = path.clone();
                    new_path.push(edge.id());
                    stack.push((edge.target(), depth + 1, new_path));
                }
            }
        }
        results
    }

    /// Flushes property and range indexes to disk.
    ///
    /// # Errors
    ///
    /// Returns an error if any save operation fails.
    pub(crate) fn flush(&self, path: &Path) -> Result<()> {
        let property_path = path.join("property_index.bin");
        self.property_index
            .read()
            .save_to_file(&property_path)
            .map_err(crate::error::Error::Io)?;

        let range_path = path.join("range_index.bin");
        self.range_index
            .read()
            .save_to_file(&range_path)
            .map_err(crate::error::Error::Io)?;

        Ok(())
    }

    // -------------------------------------------------------------------------
    // Disk loading helpers
    // -------------------------------------------------------------------------

    fn load_property_index(path: &Path) -> PropertyIndex {
        let index_path = path.join("property_index.bin");
        if index_path.exists() {
            match PropertyIndex::load_from_file(&index_path) {
                Ok(idx) => return idx,
                Err(e) => tracing::warn!(
                    "Failed to load PropertyIndex from {:?}: {}. Starting empty.",
                    index_path,
                    e
                ),
            }
        }
        PropertyIndex::new()
    }

    fn load_range_index(path: &Path) -> RangeIndex {
        let index_path = path.join("range_index.bin");
        if index_path.exists() {
            match RangeIndex::load_from_file(&index_path) {
                Ok(idx) => return idx,
                Err(e) => tracing::warn!(
                    "Failed to load RangeIndex from {:?}: {}. Starting empty.",
                    index_path,
                    e
                ),
            }
        }
        RangeIndex::new()
    }
}
