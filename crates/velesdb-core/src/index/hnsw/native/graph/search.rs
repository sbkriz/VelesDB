//! HNSW search operations.

use super::super::distance::DistanceEngine;
use super::super::layer::NodeId;
use super::super::ordered_float::OrderedFloat;
use super::NativeHnsw;
use crate::perf_optimizations::ContiguousVectors;
use smallvec::SmallVec;
use std::borrow::Cow;
use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::atomic::Ordering;

// =============================================================================
// BitVecVisited — compact visited-node tracker (Issue #420, Component 2)
// =============================================================================

/// Compact visited-node tracker using one bit per node ID.
///
/// For 10K nodes this uses 1.25 KB (fits in L1 cache), compared to
/// ~80 KB for `FxHashSet<usize>`. The bitset is stored as `Vec<u64>`
/// for efficient 64-bit word operations.
///
/// # Usage
///
/// ```text
/// let mut visited = BitVecVisited::with_capacity(10_000);
/// visited.insert(42);
/// assert!(visited.contains(42));
/// visited.clear();     // O(n/64) memset, preserves allocation
/// ```
#[derive(Default)]
pub(crate) struct BitVecVisited {
    /// Each bit at position `i` indicates whether node `i` has been visited.
    words: Vec<u64>,
}

impl BitVecVisited {
    /// Creates a new `BitVecVisited` with enough capacity for node IDs in `[0, capacity)`.
    ///
    /// All bits are initially unset (not visited).
    #[inline]
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        let word_count = capacity.div_ceil(64);
        Self {
            words: vec![0_u64; word_count],
        }
    }

    /// Returns `true` if the given node ID has been marked as visited.
    ///
    /// Returns `false` for IDs beyond the current capacity (no panic).
    #[inline]
    pub(crate) fn contains(&self, id: usize) -> bool {
        let word_idx = id / 64;
        let bit_idx = id % 64;
        self.words
            .get(word_idx)
            .is_some_and(|word| word & (1_u64 << bit_idx) != 0)
    }

    /// Marks a node ID as visited.
    ///
    /// Returns `true` if the node was **not** previously visited (newly inserted),
    /// matching the `HashSet::insert` contract for drop-in replacement.
    ///
    /// Grows the internal storage if `id` exceeds the current capacity.
    #[inline]
    pub(crate) fn insert(&mut self, id: usize) -> bool {
        self.ensure_capacity(id);
        let word_idx = id / 64;
        let bit_idx = id % 64;
        let mask = 1_u64 << bit_idx;
        let was_unset = self.words[word_idx] & mask == 0;
        self.words[word_idx] |= mask;
        was_unset
    }

    /// Resets all bits to zero without deallocating.
    ///
    /// Uses `fill(0)` which compiles to a single `memset` — O(n/64)
    /// and far cheaper than dropping and reallocating.
    #[inline]
    pub(crate) fn clear(&mut self) {
        self.words.fill(0);
    }

    /// Grows the internal storage so that `id` fits, if it does not already.
    #[inline]
    fn ensure_capacity(&mut self, id: usize) {
        let required = id / 64 + 1;
        if required > self.words.len() {
            self.words.resize(required, 0);
        }
    }
}

/// Returns whether prefetch hints should be used for vectors of the given dimension.
///
/// Threshold: vector must span at least 2 cache lines (128 bytes = 32 f32 elements).
/// Below this, prefetch overhead exceeds the benefit.
#[inline]
fn should_prefetch(dimension: usize) -> bool {
    let vector_bytes = dimension * std::mem::size_of::<f32>();
    vector_bytes >= 2 * crate::simd_native::L2_CACHE_LINE_BYTES
}

/// Maximum number of pooled instances retained per thread.
///
/// Applies to visited bitsets, candidate heaps, and result heaps.
const POOL_MAX: usize = 4;

/// Type alias for the candidate min-heap (closest candidate first).
type CandidateHeap = BinaryHeap<Reverse<(OrderedFloat, NodeId)>>;

/// Type alias for the result max-heap (furthest result first for eviction).
type ResultHeap = BinaryHeap<(OrderedFloat, NodeId)>;

// Thread-local pools of reusable search data structures to avoid repeated
// allocations during batch HNSW searches. Each thread keeps up to `POOL_MAX`
// instances of each type.
thread_local! {
    static VISITED_POOL: RefCell<Vec<BitVecVisited>> = const { RefCell::new(Vec::new()) };
    static CANDIDATE_HEAP_POOL: RefCell<Vec<CandidateHeap>> = const { RefCell::new(Vec::new()) };
    static RESULT_HEAP_POOL: RefCell<Vec<ResultHeap>> = const { RefCell::new(Vec::new()) };
}

/// Borrows a visited bitset from the thread-local pool, or creates a new one.
///
/// `capacity_hint` is the current node count — the returned bitset is
/// guaranteed to hold at least that many bits. Pooled bitsets are grown
/// to `capacity_hint` if the index has expanded since they were returned.
#[inline]
fn acquire_visited_set(capacity_hint: usize) -> BitVecVisited {
    VISITED_POOL.with(|pool| {
        let mut set = pool
            .borrow_mut()
            .pop()
            .unwrap_or_else(|| BitVecVisited::with_capacity(capacity_hint));
        // Ensure pooled bitsets are large enough for the current index size.
        if capacity_hint > 0 {
            set.ensure_capacity(capacity_hint.saturating_sub(1));
        }
        set
    })
}

/// Returns a visited bitset to the thread-local pool after clearing it.
///
/// Keeps at most [`POOL_MAX`] bitsets per thread to bound memory usage.
#[inline]
fn release_visited_set(mut set: BitVecVisited) {
    set.clear();
    VISITED_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        if pool.len() < POOL_MAX {
            pool.push(set);
        }
    });
}

/// Borrows a candidate heap from the thread-local pool, or creates a new one.
///
/// The returned heap is guaranteed to be empty (cleared before pooling).
#[inline]
fn acquire_candidate_heap() -> CandidateHeap {
    CANDIDATE_HEAP_POOL.with(|pool| pool.borrow_mut().pop().unwrap_or_default())
}

/// Returns a candidate heap to the thread-local pool after clearing it.
///
/// Keeps at most [`POOL_MAX`] heaps per thread to bound memory usage.
#[inline]
fn release_candidate_heap(mut heap: CandidateHeap) {
    heap.clear();
    CANDIDATE_HEAP_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        if pool.len() < POOL_MAX {
            pool.push(heap);
        }
    });
}

/// Borrows a result heap from the thread-local pool, or creates a new one.
///
/// The returned heap is guaranteed to be empty (cleared before pooling).
#[inline]
fn acquire_result_heap() -> ResultHeap {
    RESULT_HEAP_POOL.with(|pool| pool.borrow_mut().pop().unwrap_or_default())
}

/// Returns a result heap to the thread-local pool after clearing it.
///
/// Keeps at most [`POOL_MAX`] heaps per thread to bound memory usage.
#[inline]
fn release_result_heap(mut heap: ResultHeap) {
    heap.clear();
    RESULT_HEAP_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        if pool.len() < POOL_MAX {
            pool.push(heap);
        }
    });
}

// =============================================================================
// Extracted search helpers (Issue #366, Phase A.1 GREEN)
// =============================================================================

/// Encapsulates the mutable state of a `search_layer` traversal.
///
/// Bundles the candidate/result heaps, visited set, and stagnation
/// counter into a single struct to reduce argument counts and keep
/// helper functions under the Codacy complexity limit.
struct SearchState {
    candidates: BinaryHeap<Reverse<(OrderedFloat, NodeId)>>,
    results: BinaryHeap<(OrderedFloat, NodeId)>,
    visited: BitVecVisited,
    stagnation_count: usize,
}

impl SearchState {
    /// Creates a fresh search state, acquiring pooled data structures.
    ///
    /// `capacity_hint` is the current node count of the HNSW index, used
    /// to size the bitset so most inserts avoid reallocation. Candidate
    /// and result heaps are also acquired from thread-local pools to
    /// avoid repeated allocation/deallocation during batch searches.
    fn new(capacity_hint: usize) -> Self {
        Self {
            candidates: acquire_candidate_heap(),
            results: acquire_result_heap(),
            visited: acquire_visited_set(capacity_hint),
            stagnation_count: 0,
        }
    }

    /// Pushes a candidate node into both heaps and marks it visited.
    #[inline]
    fn push_candidate(&mut self, node: NodeId, dist: f32) {
        self.candidates.push(Reverse((OrderedFloat(dist), node)));
        self.results.push((OrderedFloat(dist), node));
        self.visited.insert(node);
    }

    /// Returns `true` if the search should terminate.
    ///
    /// Termination conditions:
    /// 1. The current candidate distance exceeds the furthest result and
    ///    the result set has reached `ef` capacity.
    /// 2. Stagnation limit is enabled and the counter has reached it.
    #[inline]
    fn should_terminate(&self, c_dist: f32, ef: usize, stagnation_limit: usize) -> bool {
        let furthest = self.results.peek().map_or(f32::MAX, |r| r.0 .0);
        if c_dist > furthest && self.results.len() >= ef {
            return true;
        }
        stagnation_limit > 0 && self.stagnation_count >= stagnation_limit
    }

    /// Updates the stagnation counter: resets on improvement, increments otherwise.
    #[inline]
    fn update_stagnation(&mut self, improved: bool) {
        if improved {
            self.stagnation_count = 0;
        } else {
            self.stagnation_count += 1;
        }
    }

    /// Consumes the state and returns results sorted by distance ascending.
    ///
    /// When `limit` is `Some(k)`, uses partial sort (`select_nth_unstable_by`)
    /// to efficiently return only the top-k nearest results in O(n + k log k)
    /// instead of O(n log n).
    ///
    /// The visited set is released back to the thread-local pool by the
    /// [`Drop`] impl when `self` goes out of scope.
    fn into_sorted_results(mut self, limit: Option<usize>) -> Vec<(NodeId, f32)> {
        let results = std::mem::take(&mut self.results);
        let mut result_vec: Vec<(NodeId, f32)> =
            results.into_iter().map(|(d, n)| (n, d.0)).collect();
        let cmp = |a: &(NodeId, f32), b: &(NodeId, f32)| a.1.total_cmp(&b.1);
        if let Some(k) = limit {
            crate::index::top_k_partial_sort(&mut result_vec, k, cmp);
        } else {
            result_vec.sort_by(cmp);
        }
        result_vec
    }
}

impl Drop for SearchState {
    fn drop(&mut self) {
        // Return pooled data structures. `std::mem::take` leaves a Default
        // (empty) value in `self`, so the subsequent Drop of `self` fields
        // is a no-op.

        let visited = std::mem::take(&mut self.visited);
        if !visited.words.is_empty() {
            release_visited_set(visited);
        }

        let candidates = std::mem::take(&mut self.candidates);
        if candidates.capacity() > 0 {
            release_candidate_heap(candidates);
        }

        let results = std::mem::take(&mut self.results);
        if results.capacity() > 0 {
            release_result_heap(results);
        }
    }
}

/// Speculative prefetch lookahead distance for `gather_unvisited_neighbors`.
///
/// Prefetches this many neighbors ahead in the list. Speculative: a prefetched
/// neighbor may turn out to be already visited, in which case the prefetch is
/// harmless (only wastes a cache line slot, ~64 bytes on x86_64).
const GATHER_PREFETCH_AHEAD: usize = 2;

/// Gathers vector slices for unvisited neighbors, marking them visited.
///
/// When `use_prefetch` is `true`, speculatively prefetches neighbor vectors
/// [`GATHER_PREFETCH_AHEAD`] positions ahead to hide memory latency during
/// the `get_unchecked` calls.
#[inline]
fn gather_unvisited_neighbors<'a>(
    neighbors: &[NodeId],
    visited: &mut BitVecVisited,
    vectors: &'a ContiguousVectors,
    use_prefetch: bool,
) -> SmallVec<[(NodeId, &'a [f32]); 32]> {
    let mut batch = SmallVec::new();

    // Speculatively prefetch the first GATHER_PREFETCH_AHEAD neighbor vectors.
    if use_prefetch {
        for &neighbor in neighbors.iter().take(GATHER_PREFETCH_AHEAD) {
            vectors.prefetch(neighbor);
        }
    }

    for (i, &neighbor) in neighbors.iter().enumerate() {
        // Speculatively prefetch the vector GATHER_PREFETCH_AHEAD positions
        // ahead while processing the current neighbor.
        if use_prefetch {
            if let Some(&ahead) = neighbors.get(i + GATHER_PREFETCH_AHEAD) {
                vectors.prefetch(ahead);
            }
        }

        if visited.insert(neighbor) {
            // SAFETY: neighbor is a valid node_id from the graph's neighbor list,
            // only containing IDs of successfully inserted nodes.
            // - Condition 1: neighbor < vectors.len().
            // Reason: Batch gathering of unvisited neighbor vectors.
            let vec = unsafe { vectors.get_unchecked(neighbor) };
            batch.push((neighbor, vec));
        }
    }
    batch
}

/// Processes batch distance results into the search state heaps.
///
/// Returns `true` if any neighbor improved the result set.
#[inline]
fn process_batch_results(
    batch: &[(NodeId, &[f32])],
    distances: &[f32],
    ef: usize,
    state: &mut SearchState,
) -> bool {
    let mut improved = false;
    for (&(node_id, _), &dist) in batch.iter().zip(distances.iter()) {
        let furthest = state.results.peek().map_or(f32::MAX, |r| r.0 .0);
        if dist < furthest || state.results.len() < ef {
            state
                .candidates
                .push(Reverse((OrderedFloat(dist), node_id)));
            state.results.push((OrderedFloat(dist), node_id));
            if state.results.len() > ef {
                state.results.pop();
            }
            improved = true;
        }
    }
    improved
}

impl<D: DistanceEngine> NativeHnsw<D> {
    /// Searches for k nearest neighbors.
    ///
    /// # Distance semantics
    ///
    /// Returned distances are **raw engine distances** from `D::distance()`.
    /// When `D = CachedSimdDistance`, Euclidean values are squared L2 (no
    /// sqrt). Callers that expose results to users must apply
    /// [`transform_score()`] to convert to the user-visible metric.
    ///
    /// [`transform_score()`]: super::backend_adapter::NativeHnsw::transform_score
    #[inline]
    #[must_use]
    pub fn search(&self, query: &[f32], k: usize, ef_search: usize) -> Vec<(NodeId, f32)> {
        let prepared_query = self.prepare_query(query);
        let query: &[f32] = &prepared_query;
        let entry_point = *self.entry_point.read();
        let Some(ep) = entry_point else {
            return Vec::new();
        };

        let max_layer = self.max_layer.load(Ordering::Relaxed);

        let mut current_ep = ep;
        for layer_idx in (1..=max_layer).rev() {
            current_ep = self.search_layer_single(query, current_ep, layer_idx);
        }

        let count = self.count.load(Ordering::Relaxed);
        let probes = self.adaptive_num_probes(count, ef_search, k);

        if probes > 1 {
            return self.search_multi_entry(query, k, ef_search, probes);
        }

        self.search_layer(
            query,
            &[current_ep],
            ef_search,
            0,
            self.stagnation_limit,
            Some(k),
        )
    }

    /// Adaptive number of entry-point probes for high-recall searches.
    #[inline]
    fn adaptive_num_probes(&self, count: usize, ef_search: usize, k: usize) -> usize {
        if count < 10_000 || ef_search <= (k * 4).max(64) {
            return 1;
        }

        if ef_search >= 1024 {
            4
        } else if ef_search >= 512 {
            3
        } else {
            2
        }
    }

    /// Multi-entry point search for improved recall on hard queries.
    #[must_use]
    pub fn search_multi_entry(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
        num_probes: usize,
    ) -> Vec<(NodeId, f32)> {
        let prepared_query = self.prepare_query(query);
        let query: &[f32] = &prepared_query;
        let entry_point = *self.entry_point.read();
        let Some(ep) = entry_point else {
            return Vec::new();
        };

        let count = self.count.load(Ordering::Relaxed);
        if count == 0 {
            return Vec::new();
        }

        let max_layer = self.max_layer.load(Ordering::Relaxed);

        let mut current_ep = ep;
        for layer_idx in (1..=max_layer).rev() {
            current_ep = self.search_layer_single(query, current_ep, layer_idx);
        }

        let mut entry_points = vec![current_ep];
        if num_probes > 1 && count > 10 {
            for _ in 1..num_probes.min(4) {
                let old_state = self
                    .rng_state
                    .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |mut state| {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        Some(state)
                    })
                    .unwrap_or_else(|s| s);

                let mut state = old_state;
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;

                let random_id = (state as usize) % count;
                if !entry_points.contains(&random_id) {
                    entry_points.push(random_id);
                }
            }
        }

        self.search_layer(
            query,
            &entry_points,
            ef_search,
            0,
            self.stagnation_limit,
            Some(k),
        )
    }

    // =========================================================================
    // Layer-level search helpers
    // =========================================================================

    /// F-04 optimization: acquires both vectors and layers read locks once
    /// before the greedy descent loop, avoiding repeated lock cycles per hop.
    ///
    /// Includes software prefetch hints for upcoming neighbor vectors to
    /// reduce memory latency in upper HNSW layers (mirrors `search_layer`).
    #[inline]
    pub(in crate::index::hnsw::native::graph) fn search_layer_single(
        &self,
        query: &[f32],
        entry: NodeId,
        layer: usize,
    ) -> NodeId {
        self.with_vectors_and_layers_read(|vectors, layers| {
            let dimension = vectors.dimension();
            let prefetch_dist = crate::simd_native::calculate_prefetch_distance(dimension);
            let mut best = entry;
            // SAFETY: `entry` is a valid node_id from entry_point (always a
            // successfully inserted node).
            // - Condition 1: entry < vectors.len() (from a prior successful insert).
            // Reason: Hot-path greedy descent — bounds check eliminated.
            let entry_vec = unsafe { vectors.get_unchecked(entry) };
            let mut best_dist = self.distance.distance(query, entry_vec);

            loop {
                let improved = layers[layer]
                    .with_neighbors(best, |neighbors| {
                        self.greedy_scan_with_prefetch(
                            query,
                            neighbors,
                            vectors,
                            dimension,
                            prefetch_dist,
                            &mut best,
                            &mut best_dist,
                        )
                    })
                    .unwrap_or(false);

                if !improved {
                    break;
                }
            }

            best
        })
    }

    /// Prefetch neighbor vectors into CPU cache ahead of access.
    #[inline]
    fn prefetch_neighbors(
        neighbors: &[NodeId],
        vectors: &crate::perf_optimizations::ContiguousVectors,
        start: usize,
        count: usize,
    ) {
        for &neighbor_id in neighbors.iter().skip(start).take(count) {
            if neighbor_id < vectors.len() {
                vectors.prefetch(neighbor_id);
            }
        }
    }

    /// Scans a neighbor list with software prefetch, updating best node/dist.
    ///
    /// Returns `true` if a closer neighbor was found during the scan.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn greedy_scan_with_prefetch(
        &self,
        query: &[f32],
        neighbors: &[NodeId],
        vectors: &crate::perf_optimizations::ContiguousVectors,
        dimension: usize,
        prefetch_dist: usize,
        best: &mut NodeId,
        best_dist: &mut f32,
    ) -> bool {
        let use_prefetch = should_prefetch(dimension);

        // Prefetch the first batch of neighbor vectors into cache.
        if use_prefetch && neighbors.len() > prefetch_dist {
            Self::prefetch_neighbors(neighbors, vectors, 0, prefetch_dist);
        }

        let mut improved = false;
        for (i, &neighbor) in neighbors.iter().enumerate() {
            // Prefetch upcoming neighbor vectors while processing the current one.
            if use_prefetch && i + prefetch_dist < neighbors.len() {
                Self::prefetch_neighbors(neighbors, vectors, i + prefetch_dist, 1);
            }

            // SAFETY: neighbor is a valid node_id from the graph's
            // neighbor list, only containing IDs of inserted nodes.
            // - Condition 1: neighbor < vectors.len().
            // Reason: Inner loop of greedy descent — bounds check eliminated.
            let neighbor_vec = unsafe { vectors.get_unchecked(neighbor) };
            let dist = self.distance.distance(query, neighbor_vec);
            if dist < *best_dist {
                *best = neighbor;
                *best_dist = dist;
                improved = true;
            }
        }

        improved
    }

    /// Search a single layer with ef candidates.
    ///
    /// Delegates to [`SearchState`], [`gather_unvisited_neighbors`], and
    /// [`process_batch_results`] to keep each helper under Codacy limits
    /// (CC <= 8, NLOC <= 50).
    ///
    /// F-03 optimization: acquires both vectors and layers read locks once
    /// before the search loop, avoiding ~ef lock acquire/release cycles.
    ///
    /// `stagnation_limit` controls early termination: 0 disables it (use
    /// during index construction to avoid degrading neighbor quality).
    /// For search queries, pass `self.stagnation_limit`.
    ///
    /// `result_limit` controls partial sort optimization: when `Some(k)`,
    /// uses `select_nth_unstable_by` to return only the top-k nearest
    /// results in O(n + k log k) instead of sorting all ef candidates
    /// in O(ef log ef). Pass `None` during construction to get all
    /// candidates sorted (needed for VAMANA neighbor selection).
    #[inline]
    pub(in crate::index::hnsw::native::graph) fn search_layer(
        &self,
        query: &[f32],
        entry_points: &[NodeId],
        ef: usize,
        layer: usize,
        stagnation_limit: usize,
        result_limit: Option<usize>,
    ) -> Vec<(NodeId, f32)> {
        let capacity_hint = self.count.load(Ordering::Relaxed);
        let mut state = SearchState::new(capacity_hint);

        self.with_vectors_and_layers_read(|vectors, layers| {
            let use_prefetch = should_prefetch(vectors.dimension());

            // Initialize entry points
            for &ep in entry_points {
                // SAFETY: ep is a valid node_id from entry_point or random probe,
                // always IDs of successfully inserted nodes.
                // - Condition 1: ep < vectors.len().
                // Reason: Entry-point initialization in search hot path.
                let ep_vec = unsafe { vectors.get_unchecked(ep) };
                let dist = self.distance.distance(query, ep_vec);
                state.push_candidate(ep, dist);
            }

            // Main search loop
            while let Some(Reverse((OrderedFloat(c_dist), c_node))) = state.candidates.pop() {
                if state.should_terminate(c_dist, ef, stagnation_limit) {
                    break;
                }

                let improved = layers[layer]
                    .with_neighbors(c_node, |neighbors| {
                        let batch = gather_unvisited_neighbors(
                            neighbors,
                            &mut state.visited,
                            vectors,
                            use_prefetch,
                        );
                        if batch.is_empty() {
                            return false;
                        }
                        let vecs: SmallVec<[&[f32]; 32]> = batch.iter().map(|(_, v)| *v).collect();
                        let distances = self.distance.batch_distance(query, &vecs);
                        process_batch_results(&batch, &distances, ef, &mut state)
                    })
                    .unwrap_or(false);

                state.update_stagnation(improved);
            }
        });

        state.into_sorted_results(result_limit)
    }

    /// Prepares a query vector for search or insertion. Returns `Cow::Borrowed`
    /// for non-cosine metrics (zero-allocation) or `Cow::Owned` with normalized
    /// copy for cosine.
    #[inline]
    pub(in crate::index::hnsw::native) fn prepare_query<'a>(
        &self,
        query: &'a [f32],
    ) -> Cow<'a, [f32]> {
        if self.distance.is_pre_normalized()
            && self.distance.metric() == crate::DistanceMetric::Cosine
        {
            let mut prepared = query.to_vec();
            crate::simd_native::normalize_inplace_native(&mut prepared);
            Cow::Owned(prepared)
        } else {
            Cow::Borrowed(query)
        }
    }
}

// =============================================================================
// Contract tests for SearchState, gather_unvisited_neighbors, and
// process_batch_results helpers extracted from search_layer (Issue #366).
// =============================================================================

#[cfg(test)]
mod search_refactor_tests {
    use super::super::super::distance::SimdDistance;
    use super::super::super::layer::NodeId;
    use super::super::super::ordered_float::OrderedFloat;
    use super::super::NativeHnsw;
    use super::{gather_unvisited_neighbors, process_batch_results};
    use super::{BitVecVisited, SearchState};
    use crate::distance::DistanceMetric;
    use rustc_hash::FxHashSet;
    use smallvec::SmallVec;
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    // =========================================================================
    // Helpers
    // =========================================================================

    /// Brute-force cosine distance for ground-truth computation.
    #[allow(dead_code)]
    fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            1.0
        } else {
            1.0 - (dot / (norm_a * norm_b))
        }
    }

    // =========================================================================
    // 1. SearchState::new + push_candidate
    // =========================================================================

    #[test]
    fn test_search_state_new_and_push() {
        let mut state = SearchState::new(0);

        // Push three candidates with known distances
        state.push_candidate(10, 0.5);
        state.push_candidate(20, 0.1);
        state.push_candidate(30, 0.9);

        // Candidates min-heap: closest first (0.1 at top)
        let (OrderedFloat(top_dist), top_node) = state
            .candidates
            .peek()
            .map(|Reverse(item)| item)
            .copied()
            .expect("candidates should not be empty");
        assert_eq!(top_node, 20, "min-heap should surface closest candidate");
        assert!((top_dist - 0.1).abs() < f32::EPSILON);

        // Results max-heap: furthest first (0.9 at top)
        let &(OrderedFloat(furthest_dist), furthest_node) =
            state.results.peek().expect("results should not be empty");
        assert_eq!(furthest_node, 30, "max-heap should surface furthest result");
        assert!((furthest_dist - 0.9).abs() < f32::EPSILON);

        // Visited set should contain all three
        assert!(state.visited.contains(10));
        assert!(state.visited.contains(20));
        assert!(state.visited.contains(30));
    }

    // =========================================================================
    // 2. SearchState::should_terminate
    // =========================================================================

    #[test]
    fn test_search_state_should_terminate() {
        let mut state = SearchState::new(0);

        // Fill ef=3 results with distances 0.1, 0.3, 0.5
        state.push_candidate(1, 0.1);
        state.push_candidate(2, 0.3);
        state.push_candidate(3, 0.5);

        let ef = 3;
        let stagnation_limit = 10;

        // c_dist=0.6 > furthest(0.5) AND results.len()>=ef => should terminate
        assert!(
            state.should_terminate(0.6, ef, stagnation_limit),
            "should terminate: c_dist > furthest and results full"
        );

        // c_dist=0.4 < furthest(0.5) => should NOT terminate
        assert!(
            !state.should_terminate(0.4, ef, stagnation_limit),
            "should not terminate: c_dist < furthest"
        );

        // c_dist=0.6 > furthest but results.len() < ef => should NOT terminate
        // (simulate by creating state with only 2 results)
        let mut state2 = SearchState::new(0);
        state2.push_candidate(1, 0.1);
        state2.push_candidate(2, 0.3);
        assert!(
            !state2.should_terminate(0.6, ef, stagnation_limit),
            "should not terminate: results not yet full"
        );
    }

    // =========================================================================
    // 3. SearchState stagnation tracking
    // =========================================================================

    #[test]
    fn test_search_state_stagnation() {
        let mut state = SearchState::new(0);

        // Fill ef=2 results
        state.push_candidate(1, 0.1);
        state.push_candidate(2, 0.3);

        let ef = 2;
        let stagnation_limit = 3;

        // Initial stagnation count is 0
        assert_eq!(state.stagnation_count, 0);

        // Three rounds without improvement
        state.update_stagnation(false); // count -> 1
        assert_eq!(state.stagnation_count, 1);
        assert!(!state.should_terminate(0.0, ef, stagnation_limit));

        state.update_stagnation(false); // count -> 2
        assert_eq!(state.stagnation_count, 2);
        assert!(!state.should_terminate(0.0, ef, stagnation_limit));

        state.update_stagnation(false); // count -> 3 >= limit
        assert_eq!(state.stagnation_count, 3);
        assert!(
            state.should_terminate(0.0, ef, stagnation_limit),
            "should terminate after reaching stagnation limit"
        );

        // Improvement resets stagnation
        state.update_stagnation(true); // count -> 0
        assert_eq!(state.stagnation_count, 0);
        assert!(!state.should_terminate(0.0, ef, stagnation_limit));
    }

    // =========================================================================
    // 4. SearchState::into_sorted_results
    // =========================================================================

    #[test]
    fn test_search_state_into_sorted_results() {
        let mut state = SearchState::new(0);

        // Insert results in non-sorted order
        state.push_candidate(10, 0.7);
        state.push_candidate(20, 0.2);
        state.push_candidate(30, 0.5);
        state.push_candidate(40, 0.1);
        state.push_candidate(50, 0.9);

        let sorted = state.into_sorted_results(None);

        // Should be sorted ascending by distance
        assert_eq!(sorted.len(), 5);
        assert_eq!(sorted[0].0, 40); // dist 0.1
        assert_eq!(sorted[1].0, 20); // dist 0.2
        assert_eq!(sorted[2].0, 30); // dist 0.5
        assert_eq!(sorted[3].0, 10); // dist 0.7
        assert_eq!(sorted[4].0, 50); // dist 0.9

        // Verify distances are monotonically non-decreasing
        for window in sorted.windows(2) {
            assert!(
                window[0].1 <= window[1].1,
                "results must be sorted by distance ascending: {} <= {}",
                window[0].1,
                window[1].1,
            );
        }
    }

    // =========================================================================
    // 5. gather_unvisited_neighbors filters visited nodes
    // =========================================================================

    #[test]
    fn test_gather_unvisited_neighbors_filters_visited() {
        let dim = 4;
        let mut vectors = crate::perf_optimizations::ContiguousVectors::new(dim, 10)
            .expect("alloc should succeed");
        for i in 0..5_usize {
            #[allow(clippy::cast_precision_loss)]
            let v: Vec<f32> = (0..dim).map(|j| (i * dim + j) as f32).collect();
            vectors.push(&v).expect("push should succeed");
        }

        let neighbors: Vec<NodeId> = vec![0, 1, 2, 3, 4];
        let mut visited = BitVecVisited::with_capacity(5);
        // Pre-mark nodes 1 and 3 as visited
        visited.insert(1);
        visited.insert(3);

        let unvisited: SmallVec<[(NodeId, &[f32]); 32]> =
            gather_unvisited_neighbors(&neighbors, &mut visited, &vectors, false);

        let ids: Vec<NodeId> = unvisited.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids.len(), 3, "should exclude 2 visited nodes");
        assert!(ids.contains(&0));
        assert!(ids.contains(&2));
        assert!(ids.contains(&4));
        assert!(!ids.contains(&1), "visited node 1 must be excluded");
        assert!(!ids.contains(&3), "visited node 3 must be excluded");
    }

    // =========================================================================
    // 6. gather_unvisited_neighbors marks returned nodes as visited
    // =========================================================================

    #[test]
    fn test_gather_unvisited_neighbors_marks_visited() {
        let dim = 4;
        let mut vectors = crate::perf_optimizations::ContiguousVectors::new(dim, 10)
            .expect("alloc should succeed");
        for i in 0..3_usize {
            #[allow(clippy::cast_precision_loss)]
            let v: Vec<f32> = (0..dim).map(|j| (i * dim + j) as f32).collect();
            vectors.push(&v).expect("push should succeed");
        }

        let neighbors: Vec<NodeId> = vec![0, 1, 2];
        let mut visited = BitVecVisited::with_capacity(3);

        let _unvisited = gather_unvisited_neighbors(&neighbors, &mut visited, &vectors, false);

        // All returned neighbors should now be in the visited set
        assert!(visited.contains(0), "node 0 should be marked visited");
        assert!(visited.contains(1), "node 1 should be marked visited");
        assert!(visited.contains(2), "node 2 should be marked visited");
    }

    // =========================================================================
    // 7. process_batch_results updates candidate and result heaps
    // =========================================================================

    #[test]
    fn test_process_batch_results_updates_heaps() {
        let mut state = SearchState::new(0);
        let ef = 10;

        // Simulate a batch of 3 neighbors with their pre-computed distances
        let dim = 4;
        let vecs: Vec<Vec<f32>> = (0..3)
            .map(|i| (0..dim).map(|j| (i * dim + j) as f32).collect())
            .collect();
        let batch: Vec<(NodeId, &[f32])> = vecs
            .iter()
            .enumerate()
            .map(|(i, v)| (i, v.as_slice()))
            .collect();
        let distances = vec![0.3_f32, 0.1, 0.5];

        let improved = process_batch_results(&batch, &distances, ef, &mut state);

        assert!(improved, "first batch should improve empty state");

        // Candidates should contain all 3
        assert_eq!(state.candidates.len(), 3);

        // Results should contain all 3
        assert_eq!(state.results.len(), 3);

        // Min-candidate should be node 1 (dist 0.1)
        let Reverse((OrderedFloat(min_dist), min_node)) =
            *state.candidates.peek().expect("non-empty");
        assert_eq!(min_node, 1);
        assert!((min_dist - 0.1).abs() < f32::EPSILON);
    }

    // =========================================================================
    // 8. process_batch_results evicts furthest when results exceed ef
    // =========================================================================

    #[test]
    fn test_process_batch_results_evicts_furthest_when_full() {
        let mut state = SearchState::new(0);
        let ef = 3;

        // Pre-fill with 3 results (ef is full)
        state.push_candidate(10, 0.2);
        state.push_candidate(20, 0.4);
        state.push_candidate(30, 0.6);

        // New batch: one candidate closer than the furthest (0.6), one farther
        let dim = 4;
        let v_close: Vec<f32> = vec![1.0; dim];
        let v_far: Vec<f32> = vec![2.0; dim];
        let batch: Vec<(NodeId, &[f32])> = vec![(40, v_close.as_slice()), (50, v_far.as_slice())];
        let distances = vec![0.3_f32, 0.8];

        let improved = process_batch_results(&batch, &distances, ef, &mut state);

        assert!(
            improved,
            "batch with closer candidate should improve results"
        );

        // Results should still be capped at ef=3
        assert_eq!(state.results.len(), ef, "results must not exceed ef");

        // The furthest result should now be 0.4 (node 20), since:
        //   - 0.6 was evicted when 0.3 was inserted
        //   - 0.8 was rejected (> furthest after eviction)
        let result_ids: Vec<NodeId> = state.results.iter().map(|(_, id)| *id).collect();
        assert!(
            !result_ids.contains(&30),
            "node 30 (dist 0.6) should have been evicted"
        );
        assert!(
            result_ids.contains(&40),
            "node 40 (dist 0.3) should have been admitted"
        );
    }

    // =========================================================================
    // 9. Refactored search recall matches original (regression guard)
    //
    // This test uses only the existing public API and compiles TODAY.
    // After the refactoring, search results must remain identical.
    // =========================================================================

    #[test]
    fn test_refactored_search_recall_matches_original() {
        let dim = 32;
        let n = 200;
        let k = 10;
        let ef_search = 64;
        let n_queries = 10;

        // Build index
        let engine = SimdDistance::new(DistanceMetric::Euclidean);
        let hnsw = NativeHnsw::new(engine, 16, 100, n);

        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                (0..dim)
                    .map(|j| ((i * dim + j) as f32 * 0.001).sin())
                    .collect()
            })
            .collect();

        for v in &vectors {
            hnsw.insert(v).expect("insert should succeed in test");
        }

        // Verify recall against brute-force ground truth
        let mut total_recall = 0.0_f64;

        for q_idx in 0..n_queries {
            let query = &vectors[q_idx * (n / n_queries)];

            let hnsw_results: Vec<NodeId> = hnsw
                .search(query, k, ef_search)
                .iter()
                .map(|(id, _)| *id)
                .collect();

            // Brute-force ground truth (euclidean)
            let mut brute: Vec<(NodeId, f32)> = vectors
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let dist: f32 = v
                        .iter()
                        .zip(query.iter())
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum();
                    (i, dist)
                })
                .collect();
            brute.sort_by(|a, b| a.1.total_cmp(&b.1));
            let ground_truth: Vec<NodeId> = brute.iter().take(k).map(|(id, _)| *id).collect();

            let hits = hnsw_results
                .iter()
                .filter(|id| ground_truth.contains(id))
                .count();

            #[allow(clippy::cast_precision_loss)]
            {
                total_recall += hits as f64 / k as f64;
            }
        }

        #[allow(clippy::cast_precision_loss)]
        let avg_recall = total_recall / n_queries as f64;

        assert!(
            avg_recall >= 0.90,
            "search recall must be >= 90% (got {:.1}%); \
             if this fails after refactoring, the extraction broke correctness",
            avg_recall * 100.0,
        );
    }

    // =========================================================================
    // 10. into_sorted_results with partial sort limit (Issue #373)
    // =========================================================================

    #[test]
    fn test_into_sorted_results_with_limit() {
        let mut state = SearchState::new(0);

        // Insert 10 results with known distances
        for i in 0..10_usize {
            #[allow(clippy::cast_precision_loss)]
            state.push_candidate(i, (10 - i) as f32 * 0.1);
        }
        assert_eq!(state.results.len(), 10);

        let sorted = state.into_sorted_results(Some(3));

        // Exactly 3 results returned
        assert_eq!(sorted.len(), 3, "limit=3 should return exactly 3 results");

        // Must be sorted by distance ascending
        for window in sorted.windows(2) {
            assert!(
                window[0].1 <= window[1].1,
                "results must be sorted ascending: {} <= {}",
                window[0].1,
                window[1].1,
            );
        }

        // Must contain the 3 nearest (smallest distances)
        // Distances were: 0.1, 0.2, ..., 1.0 — top-3 are 0.1, 0.2, 0.3
        assert!(
            (sorted[0].1 - 0.1).abs() < f32::EPSILON,
            "first result should be dist 0.1, got {}",
            sorted[0].1,
        );
        assert!(
            (sorted[1].1 - 0.2).abs() < f32::EPSILON,
            "second result should be dist 0.2, got {}",
            sorted[1].1,
        );
        assert!(
            (sorted[2].1 - 0.3).abs() < f32::EPSILON,
            "third result should be dist 0.3, got {}",
            sorted[2].1,
        );
    }

    #[test]
    fn test_into_sorted_results_without_limit() {
        let mut state = SearchState::new(0);

        // Insert 10 results
        for i in 0..10_usize {
            #[allow(clippy::cast_precision_loss)]
            state.push_candidate(i, (10 - i) as f32 * 0.1);
        }

        let sorted = state.into_sorted_results(None);

        // All 10 results returned
        assert_eq!(sorted.len(), 10, "None limit should return all results");

        // Must be sorted by distance ascending
        for window in sorted.windows(2) {
            assert!(
                window[0].1 <= window[1].1,
                "results must be sorted ascending: {} <= {}",
                window[0].1,
                window[1].1,
            );
        }
    }

    #[test]
    fn test_into_sorted_results_limit_greater_than_results() {
        let mut state = SearchState::new(0);

        // Insert only 5 results
        for i in 0..5_usize {
            #[allow(clippy::cast_precision_loss)]
            state.push_candidate(i, (5 - i) as f32 * 0.1);
        }

        // Request limit=10, but only 5 exist — should not panic
        let sorted = state.into_sorted_results(Some(10));

        assert_eq!(
            sorted.len(),
            5,
            "limit > len should return all available results"
        );

        // Must still be sorted ascending
        for window in sorted.windows(2) {
            assert!(
                window[0].1 <= window[1].1,
                "results must be sorted ascending: {} <= {}",
                window[0].1,
                window[1].1,
            );
        }
    }

    #[test]
    fn test_into_sorted_results_limit_zero() {
        let mut state = SearchState::new(0);

        state.push_candidate(0, 0.5);
        state.push_candidate(1, 0.1);
        state.push_candidate(2, 0.9);

        // limit=0 should return empty (truncate to 0)
        let sorted = state.into_sorted_results(Some(0));
        assert!(sorted.is_empty(), "limit=0 should return empty vec");
    }

    #[test]
    fn test_into_sorted_results_empty_state() {
        let state = SearchState::new(0);

        // None limit on empty state
        let sorted = state.into_sorted_results(None);
        assert!(
            sorted.is_empty(),
            "empty state with None should return empty vec"
        );

        let state2 = SearchState::new(0);

        // Some limit on empty state
        let sorted2 = state2.into_sorted_results(Some(5));
        assert!(
            sorted2.is_empty(),
            "empty state with Some(5) should return empty vec"
        );
    }

    // =========================================================================
    // 11. BitVecVisited unit tests (Issue #420, Component 2)
    // =========================================================================

    #[test]
    fn test_bitvec_visited_insert_and_contains() {
        let mut visited = BitVecVisited::with_capacity(1000);
        assert!(!visited.contains(42));
        visited.insert(42);
        assert!(visited.contains(42));
        assert!(!visited.contains(43));
    }

    #[test]
    fn test_bitvec_visited_insert_returns_newly_inserted() {
        let mut visited = BitVecVisited::with_capacity(100);
        // First insert returns true (newly inserted)
        assert!(visited.insert(10));
        // Second insert of same ID returns false (already present)
        assert!(!visited.insert(10));
        // Different ID returns true
        assert!(visited.insert(11));
    }

    #[test]
    fn test_bitvec_visited_clear_resets() {
        let mut visited = BitVecVisited::with_capacity(100);
        visited.insert(50);
        assert!(visited.contains(50));
        visited.clear();
        assert!(!visited.contains(50));
    }

    #[test]
    fn test_bitvec_visited_clear_preserves_capacity() {
        let mut visited = BitVecVisited::with_capacity(1000);
        let words_before = visited.words.len();
        visited.insert(999);
        visited.clear();
        // Capacity (word count) must not shrink after clear
        assert_eq!(visited.words.len(), words_before);
    }

    #[test]
    fn test_bitvec_visited_out_of_bounds_grows() {
        let mut visited = BitVecVisited::with_capacity(10);
        // Inserting beyond capacity should grow, not panic
        visited.insert(100);
        assert!(visited.contains(100));
        assert!(!visited.contains(99));
    }

    #[test]
    fn test_bitvec_visited_zero_capacity() {
        let mut visited = BitVecVisited::with_capacity(0);
        // Should handle zero capacity gracefully
        assert!(!visited.contains(0));
        visited.insert(0);
        assert!(visited.contains(0));
    }

    #[test]
    fn test_bitvec_visited_word_boundary() {
        let mut visited = BitVecVisited::with_capacity(128);
        // Test around the 64-bit word boundary
        for id in [0, 1, 62, 63, 64, 65, 126, 127] {
            visited.insert(id);
        }
        for id in [0, 1, 62, 63, 64, 65, 126, 127] {
            assert!(visited.contains(id), "should contain {id}");
        }
        // IDs NOT inserted should be absent
        for id in [2, 32, 66, 100] {
            assert!(!visited.contains(id), "should not contain {id}");
        }
    }

    #[test]
    fn test_bitvec_visited_identical_to_hashset() {
        // Same sequence of operations must produce same contains() results
        let mut bv = BitVecVisited::with_capacity(10_000);
        let mut hs = FxHashSet::default();
        let ids = [0, 1, 42, 999, 5000, 9999, 7, 128, 255, 256, 1023];
        for &id in &ids {
            let bv_new = bv.insert(id);
            let hs_new = hs.insert(id);
            assert_eq!(
                bv_new, hs_new,
                "insert return mismatch at {id}: bv={bv_new}, hs={hs_new}"
            );
        }
        for i in 0..10_000 {
            assert_eq!(bv.contains(i), hs.contains(&i), "contains mismatch at {i}");
        }
    }

    #[test]
    fn test_bitvec_visited_recall_regression() {
        // End-to-end recall test: build HNSW, search, verify >= 0.95
        let dim = 32;
        let n = 500;
        let k = 10;
        let ef_search = 128;
        let n_queries = 20;

        let engine = SimdDistance::new(DistanceMetric::Euclidean);
        let hnsw = NativeHnsw::new(engine, 16, 200, n);

        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                (0..dim)
                    .map(|j| ((i * dim + j) as f32 * 0.001).sin())
                    .collect()
            })
            .collect();

        for v in &vectors {
            hnsw.insert(v).expect("insert should succeed");
        }

        let mut total_recall = 0.0_f64;
        for q_idx in 0..n_queries {
            let query = &vectors[q_idx * (n / n_queries)];

            let hnsw_ids: Vec<NodeId> = hnsw
                .search(query, k, ef_search)
                .iter()
                .map(|(id, _)| *id)
                .collect();

            // Brute-force ground truth
            let mut brute: Vec<(NodeId, f32)> = vectors
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let dist: f32 = v
                        .iter()
                        .zip(query.iter())
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum();
                    (i, dist)
                })
                .collect();
            brute.sort_by(|a, b| a.1.total_cmp(&b.1));
            let gt: Vec<NodeId> = brute.iter().take(k).map(|(id, _)| *id).collect();

            let hits = hnsw_ids.iter().filter(|id| gt.contains(id)).count();
            #[allow(clippy::cast_precision_loss)]
            {
                total_recall += hits as f64 / k as f64;
            }
        }

        #[allow(clippy::cast_precision_loss)]
        let avg_recall = total_recall / n_queries as f64;

        assert!(
            avg_recall >= 0.95,
            "BitVecVisited recall@{k} must be >= 95% (got {:.1}%); \
             bitvec visited set regression detected",
            avg_recall * 100.0,
        );
    }

    // =========================================================================
    // 12. Heap pool tests (Issue #421, Component A)
    // =========================================================================

    #[test]
    fn test_heap_pool_reuses_allocations() {
        // First search state allocates fresh heaps, Drop returns them to pool.
        {
            let mut state = SearchState::new(100);
            state.push_candidate(1, 0.5);
            state.push_candidate(2, 0.3);
            // Drop returns heaps to pool
        }

        // Second search state should reuse pooled heaps (no allocation).
        {
            let mut state = SearchState::new(100);
            // Reused heaps must be empty (cleared on return to pool).
            assert!(
                state.candidates.is_empty(),
                "pooled candidate heap must be empty on acquire"
            );
            assert!(
                state.results.is_empty(),
                "pooled result heap must be empty on acquire"
            );

            // Must still function correctly after pool reuse.
            state.push_candidate(10, 0.1);
            state.push_candidate(20, 0.9);
            assert_eq!(state.candidates.len(), 2);
            assert_eq!(state.results.len(), 2);
        }
    }

    #[test]
    fn test_heap_pool_bounded_size() {
        // Create POOL_MAX + 2 states to exceed the pool limit.
        for _ in 0..(super::POOL_MAX + 2) {
            let mut state = SearchState::new(50);
            state.push_candidate(1, 0.5);
            // Drop returns heaps to pool (bounded at POOL_MAX).
        }

        // Drain the candidate pool to verify it doesn't exceed POOL_MAX.
        let mut count = 0_usize;
        super::CANDIDATE_HEAP_POOL.with(|pool| {
            count = pool.borrow().len();
        });
        assert!(
            count <= super::POOL_MAX,
            "candidate pool must not exceed POOL_MAX ({count} > {})",
            super::POOL_MAX,
        );

        let mut result_count = 0_usize;
        super::RESULT_HEAP_POOL.with(|pool| {
            result_count = pool.borrow().len();
        });
        assert!(
            result_count <= super::POOL_MAX,
            "result pool must not exceed POOL_MAX ({result_count} > {})",
            super::POOL_MAX,
        );
    }

    #[test]
    fn test_heap_pool_recall_regression() {
        // End-to-end recall test with pooled heaps: build HNSW, search
        // multiple times (triggering pool reuse), verify recall >= 0.95.
        let dim = 32;
        let n = 500;
        let k = 10;
        let ef_search = 128;
        let n_queries = 20;

        let engine = SimdDistance::new(DistanceMetric::Euclidean);
        let hnsw = NativeHnsw::new(engine, 16, 200, n);

        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                (0..dim)
                    .map(|j| ((i * dim + j) as f32 * 0.001).sin())
                    .collect()
            })
            .collect();

        for v in &vectors {
            hnsw.insert(v).expect("insert should succeed");
        }

        let mut total_recall = 0.0_f64;
        for q_idx in 0..n_queries {
            let query = &vectors[q_idx * (n / n_queries)];

            let hnsw_ids: Vec<NodeId> = hnsw
                .search(query, k, ef_search)
                .iter()
                .map(|(id, _)| *id)
                .collect();

            // Brute-force ground truth
            let mut brute: Vec<(NodeId, f32)> = vectors
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let dist: f32 = v
                        .iter()
                        .zip(query.iter())
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum();
                    (i, dist)
                })
                .collect();
            brute.sort_by(|a, b| a.1.total_cmp(&b.1));
            let gt: Vec<NodeId> = brute.iter().take(k).map(|(id, _)| *id).collect();

            let hits = hnsw_ids.iter().filter(|id| gt.contains(id)).count();
            #[allow(clippy::cast_precision_loss)]
            {
                total_recall += hits as f64 / k as f64;
            }
        }

        #[allow(clippy::cast_precision_loss)]
        let avg_recall = total_recall / n_queries as f64;

        assert!(
            avg_recall >= 0.95,
            "Pooled heap recall@{k} must be >= 95% (got {:.1}%); \
             heap pool regression detected",
            avg_recall * 100.0,
        );
    }

    // =========================================================================
    // 13. Prefetch in gather_unvisited_neighbors tests (Issue #421, Component C)
    // =========================================================================

    #[test]
    fn test_gather_unvisited_neighbors_with_prefetch() {
        // Verify that enabling prefetch does not alter the results of
        // gather_unvisited_neighbors (correctness, not performance).
        let dim = 64; // >= 2 cache lines (128B) so should_prefetch returns true
        let n = 10_usize;
        let mut vectors = crate::perf_optimizations::ContiguousVectors::new(dim, n)
            .expect("alloc should succeed");
        for i in 0..n {
            #[allow(clippy::cast_precision_loss)]
            let v: Vec<f32> = (0..dim).map(|j| (i * dim + j) as f32).collect();
            vectors.push(&v).expect("push should succeed");
        }

        let neighbors: Vec<NodeId> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut visited_no_pf = BitVecVisited::with_capacity(n);
        let mut visited_pf = BitVecVisited::with_capacity(n);

        // Pre-mark some nodes visited in both
        visited_no_pf.insert(2);
        visited_no_pf.insert(5);
        visited_no_pf.insert(8);
        visited_pf.insert(2);
        visited_pf.insert(5);
        visited_pf.insert(8);

        let batch_no_pf =
            gather_unvisited_neighbors(&neighbors, &mut visited_no_pf, &vectors, false);
        let batch_pf = gather_unvisited_neighbors(&neighbors, &mut visited_pf, &vectors, true);

        // Both must return the same node IDs in the same order.
        let ids_no_pf: Vec<NodeId> = batch_no_pf.iter().map(|(id, _)| *id).collect();
        let ids_pf: Vec<NodeId> = batch_pf.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids_no_pf, ids_pf, "prefetch must not alter gather results");

        // Verify the expected unvisited nodes: 0,1,3,4,6,7,9
        assert_eq!(ids_pf, vec![0, 1, 3, 4, 6, 7, 9]);
    }

    #[test]
    fn test_prefetch_recall_regression() {
        // End-to-end recall test with prefetch enabled via higher dimension.
        // Uses dim=64 (256 bytes per vector, > 2 cache lines) so
        // should_prefetch(64) == true in the search path.
        let dim = 64;
        let n = 500;
        let k = 10;
        let ef_search = 128;
        let n_queries = 20;

        let engine = SimdDistance::new(DistanceMetric::Euclidean);
        let hnsw = NativeHnsw::new(engine, 16, 200, n);

        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                (0..dim)
                    .map(|j| ((i * dim + j) as f32 * 0.001).sin())
                    .collect()
            })
            .collect();

        for v in &vectors {
            hnsw.insert(v).expect("insert should succeed");
        }

        let mut total_recall = 0.0_f64;
        for q_idx in 0..n_queries {
            let query = &vectors[q_idx * (n / n_queries)];

            let hnsw_ids: Vec<NodeId> = hnsw
                .search(query, k, ef_search)
                .iter()
                .map(|(id, _)| *id)
                .collect();

            // Brute-force ground truth
            let mut brute: Vec<(NodeId, f32)> = vectors
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let dist: f32 = v
                        .iter()
                        .zip(query.iter())
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum();
                    (i, dist)
                })
                .collect();
            brute.sort_by(|a, b| a.1.total_cmp(&b.1));
            let gt: Vec<NodeId> = brute.iter().take(k).map(|(id, _)| *id).collect();

            let hits = hnsw_ids.iter().filter(|id| gt.contains(id)).count();
            #[allow(clippy::cast_precision_loss)]
            {
                total_recall += hits as f64 / k as f64;
            }
        }

        #[allow(clippy::cast_precision_loss)]
        let avg_recall = total_recall / n_queries as f64;

        assert!(
            avg_recall >= 0.95,
            "Prefetch recall@{k} must be >= 95% (got {:.1}%); \
             prefetch regression detected",
            avg_recall * 100.0,
        );
    }
}
