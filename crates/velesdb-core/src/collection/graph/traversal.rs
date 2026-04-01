//! Graph traversal algorithms for multi-hop queries.
//!
//! This module provides BFS-based traversal for variable-length path patterns
//! like `(a)-[*1..3]->(b)` in MATCH clauses.
//!
//! # Streaming Mode (EPIC-019 US-005)
//!
//! For large graphs, the module provides streaming iterators that yield results
//! lazily without loading all visited nodes into memory at once.

#![allow(dead_code)] // WIP: Will be used by MATCH clause execution

use super::EdgeStore;
use smallvec::SmallVec;
use std::collections::{HashSet, VecDeque};

/// Default maximum depth for unbounded traversals.
pub const DEFAULT_MAX_DEPTH: u32 = 3;

/// Safety cap for maximum depth to prevent runaway traversals.
/// Only applied when user requests unbounded traversal (*).
///
/// Note: Neo4j and ArangoDB do NOT impose hard limits.
/// 100 is chosen to cover most real-world use cases:
/// - Social networks (6 degrees of separation)
/// - Dependency graphs (deep npm/cargo trees)
/// - Organizational hierarchies
/// - Knowledge graphs
pub const SAFETY_MAX_DEPTH: u32 = 100;

/// Stack-allocated path type used internally by BFS/DFS traversal.
///
/// Most graph queries are depth 1-3 (social networks, knowledge graphs).
/// `SmallVec<[u64; 4]>` stores up to 4 edge IDs inline (32 bytes on stack)
/// and only heap-allocates for deeper traversals, eliminating per-path
/// allocation overhead in the common case.
///
/// Note: [`TraversalResult::path`] uses `Vec<u64>` at the public API
/// boundary. This type is exposed for advanced internal use only.
pub type TraversalPath = SmallVec<[u64; 4]>;

/// Result of a graph traversal operation.
///
/// The `path` field is `Vec<u64>` at the public API boundary. Internally,
/// BFS state uses `SmallVec<[u64; 4]>` to avoid per-path heap allocation
/// for typical depth 1-3 traversals, but this is converted to `Vec` when
/// building the result.
#[derive(Debug, Clone)]
pub struct TraversalResult {
    /// The target node ID reached.
    pub target_id: u64,
    /// The path taken (list of edge IDs).
    pub path: Vec<u64>,
    /// Depth of the traversal (number of hops).
    pub depth: u32,
}

impl TraversalResult {
    /// Creates a new traversal result.
    #[must_use]
    pub fn new(target_id: u64, path: Vec<u64>, depth: u32) -> Self {
        Self {
            target_id,
            path,
            depth,
        }
    }

    /// Creates a traversal result from an internal `SmallVec` path.
    ///
    /// Converts the stack-allocated path to a heap-allocated `Vec` at the
    /// public API boundary.
    #[must_use]
    #[allow(clippy::needless_pass_by_value)]
    // Reason: Call sites pass owned SmallVecs (often via move or clone); taking
    // by value avoids requiring callers to borrow-then-clone at call sites that
    // already own the value.
    pub(crate) fn from_smallvec(target_id: u64, path: TraversalPath, depth: u32) -> Self {
        Self {
            target_id,
            path: path.to_vec(),
            depth,
        }
    }
}

/// Configuration for graph traversal.
#[derive(Debug, Clone)]
pub struct TraversalConfig {
    /// Minimum number of hops (inclusive).
    pub min_depth: u32,
    /// Maximum number of hops (inclusive).
    pub max_depth: u32,
    /// Maximum number of results to return.
    pub limit: usize,
    /// Filter by relationship types (empty = all types).
    pub rel_types: Vec<String>,
}

impl Default for TraversalConfig {
    fn default() -> Self {
        Self {
            min_depth: 1,
            max_depth: DEFAULT_MAX_DEPTH,
            limit: 100,
            rel_types: Vec::new(),
        }
    }
}

impl TraversalConfig {
    /// Creates a config for a specific range (e.g., *1..3).
    ///
    /// Respects the caller's max_depth without artificial capping.
    /// For unbounded traversals, use `with_unbounded_range()` instead.
    #[must_use]
    pub fn with_range(min: u32, max: u32) -> Self {
        Self {
            min_depth: min,
            max_depth: max,
            ..Self::default()
        }
    }

    /// Creates a config for unbounded traversal (e.g., *1..).
    ///
    /// Applies SAFETY_MAX_DEPTH cap to prevent runaway traversals.
    #[must_use]
    pub fn with_unbounded_range(min: u32) -> Self {
        Self {
            min_depth: min,
            max_depth: SAFETY_MAX_DEPTH,
            ..Self::default()
        }
    }

    /// Sets the result limit.
    #[must_use]
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Filters by relationship types.
    #[must_use]
    pub fn with_rel_types(mut self, types: Vec<String>) -> Self {
        self.rel_types = types;
        self
    }

    /// Sets a custom max depth (for advanced use cases).
    #[must_use]
    pub fn with_max_depth(mut self, max_depth: u32) -> Self {
        self.max_depth = max_depth;
        self
    }
}

/// BFS state for traversal.
#[derive(Debug)]
pub(super) struct BfsState {
    /// Current node ID.
    pub(super) node_id: u64,
    /// Path taken to reach this node (edge IDs).
    /// Uses `SmallVec` to avoid heap allocation for typical depth 1-3 traversals.
    pub(super) path: TraversalPath,
    /// Current depth.
    pub(super) depth: u32,
}

/// Direction of edge traversal used by the shared BFS helper.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BfsDirection {
    /// Follow outgoing edges (source -> target).
    Forward,
    /// Follow incoming edges (target -> source).
    Reverse,
}

/// Core BFS loop shared by forward and reverse traversal.
///
/// The `direction` parameter controls which edges are followed and which
/// endpoint is used as the next hop. All other logic is identical.
#[must_use]
fn bfs_traverse_directed(
    edge_store: &EdgeStore,
    source_id: u64,
    config: &TraversalConfig,
    direction: BfsDirection,
) -> Vec<TraversalResult> {
    let mut results = Vec::new();
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();

    // Pre-build a HashSet<&str> once for the entire traversal, not per-node.
    // Vec<String>::contains(&to_string()) allocated a String per edge in the inner loop.
    let rel_filter: HashSet<&str> = config.rel_types.iter().map(String::as_str).collect();

    // CRITICAL FIX: Mark source node as visited before traversal
    // to prevent cycles back to source causing duplicate work
    visited.insert(source_id);

    queue.push_back(BfsState {
        node_id: source_id,
        path: SmallVec::new(),
        depth: 0,
    });

    while let Some(state) = queue.pop_front() {
        if results.len() >= config.limit {
            break;
        }
        let edges = match direction {
            BfsDirection::Forward => edge_store.get_outgoing(state.node_id),
            BfsDirection::Reverse => edge_store.get_incoming(state.node_id),
        };
        process_bfs_neighbors(
            &edges,
            &state,
            config,
            &rel_filter,
            direction,
            &mut results,
            &mut visited,
            &mut queue,
        );
    }

    results
}

/// Processes neighbors for a single BFS level: filters edges, records results,
/// and enqueues unvisited nodes for the next hop.
#[inline]
#[allow(clippy::too_many_arguments)] // Reason: BFS helper passes pre-built filter set alongside traversal state; bundling into a context struct adds complexity without clarity for a private fn
fn process_bfs_neighbors(
    edges: &[&super::GraphEdge],
    state: &BfsState,
    config: &TraversalConfig,
    rel_filter: &HashSet<&str>,
    direction: BfsDirection,
    results: &mut Vec<TraversalResult>,
    visited: &mut HashSet<u64>,
    queue: &mut VecDeque<BfsState>,
) {
    for edge in edges {
        if results.len() >= config.limit {
            break;
        }
        if !rel_filter.is_empty() && !rel_filter.contains(edge.label()) {
            continue;
        }
        let next_node = match direction {
            BfsDirection::Forward => edge.target(),
            BfsDirection::Reverse => edge.source(),
        };
        let new_depth = state.depth + 1;
        if new_depth > config.max_depth {
            continue;
        }
        let mut new_path = state.path.clone();
        new_path.push(edge.id());

        if new_depth >= config.min_depth {
            results.push(TraversalResult::from_smallvec(
                next_node,
                new_path.clone(),
                new_depth,
            ));
        }
        if new_depth < config.max_depth && visited.insert(next_node) {
            queue.push_back(BfsState {
                node_id: next_node,
                path: new_path,
                depth: new_depth,
            });
        }
    }
}

/// Performs BFS traversal from a source node.
///
/// Finds all paths from `source_id` within the configured depth range.
/// Uses iterative BFS with `VecDeque` for better cache locality.
///
/// # Arguments
///
/// * `edge_store` - The edge storage to traverse.
/// * `source_id` - Starting node ID.
/// * `config` - Traversal configuration.
///
/// # Returns
///
/// Vector of traversal results, limited by `config.limit`.
#[must_use]
pub fn bfs_traverse(
    edge_store: &EdgeStore,
    source_id: u64,
    config: &TraversalConfig,
) -> Vec<TraversalResult> {
    bfs_traverse_directed(edge_store, source_id, config, BfsDirection::Forward)
}

/// Performs BFS traversal in the reverse direction (following incoming edges).
#[must_use]
pub fn bfs_traverse_reverse(
    edge_store: &EdgeStore,
    source_id: u64,
    config: &TraversalConfig,
) -> Vec<TraversalResult> {
    bfs_traverse_directed(edge_store, source_id, config, BfsDirection::Reverse)
}

/// Performs bidirectional BFS (follows both directions).
#[must_use]
pub fn bfs_traverse_both(
    edge_store: &EdgeStore,
    source_id: u64,
    config: &TraversalConfig,
) -> Vec<TraversalResult> {
    let mut results = Vec::new();
    let half_limit = config.limit / 2 + 1;

    let config_half = TraversalConfig {
        limit: half_limit,
        ..config.clone()
    };

    // Forward traversal
    let forward = bfs_traverse(edge_store, source_id, &config_half);
    // Build O(1) dedup set from forward results to avoid O(n) linear scan per reverse result.
    let seen: HashSet<u64> = forward.iter().map(|r| r.target_id).collect();
    results.extend(forward);

    // Reverse traversal — skip targets already reached by forward BFS
    if results.len() < config.limit {
        let reverse = bfs_traverse_reverse(edge_store, source_id, &config_half);
        for r in reverse {
            if results.len() >= config.limit {
                break;
            }
            if !seen.contains(&r.target_id) {
                results.push(r);
            }
        }
    }

    results.truncate(config.limit);
    results
}

// Tests moved to traversal_tests.rs per project rules
