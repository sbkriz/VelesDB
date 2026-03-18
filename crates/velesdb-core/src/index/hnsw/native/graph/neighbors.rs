//! HNSW neighbor selection and bidirectional connection management.

use super::super::distance::DistanceEngine;
use super::super::layer::NodeId;
use super::NativeHnsw;
use rustc_hash::FxHashSet;

impl<D: DistanceEngine> NativeHnsw<D> {
    /// VAMANA-style neighbor selection with alpha diversification.
    #[inline]
    pub(crate) fn select_neighbors(
        &self,
        candidates: &[(NodeId, f32)],
        max_neighbors: usize,
    ) -> Vec<NodeId> {
        if candidates.is_empty() {
            return Vec::new();
        }

        if candidates.len() <= max_neighbors {
            return candidates.iter().map(|(id, _)| *id).collect();
        }

        let mut selected: Vec<NodeId> = Vec::with_capacity(max_neighbors);
        let mut selected_set: FxHashSet<NodeId> = FxHashSet::default();

        self.with_vectors_read(|vectors| {
            for &(candidate_id, candidate_dist) in candidates {
                if selected.len() >= max_neighbors {
                    break;
                }

                // SAFETY: candidate_id is a valid node_id from the search results,
                // which only contains IDs of successfully inserted nodes.
                // - Condition 1: candidate_id < vectors.len().
                // Reason: Neighbor selection after search — bounds check eliminated.
                let candidate_vec = unsafe { vectors.get_unchecked(candidate_id) };

                let is_diverse = selected.iter().all(|&selected_id| {
                    // SAFETY: selected_id is a valid node_id already confirmed above.
                    // - Condition 1: selected_id < vectors.len().
                    // Reason: Pairwise diversity check in neighbor selection.
                    let dist_to_selected = self
                        .distance
                        .distance(candidate_vec, unsafe { vectors.get_unchecked(selected_id) });
                    self.alpha * candidate_dist <= dist_to_selected
                });

                if is_diverse || selected.is_empty() {
                    selected.push(candidate_id);
                    selected_set.insert(candidate_id);
                }
            }
        });

        if selected.len() < max_neighbors {
            for &(candidate_id, _) in candidates {
                if selected.len() >= max_neighbors {
                    break;
                }
                if selected_set.insert(candidate_id) {
                    selected.push(candidate_id);
                }
            }
        }

        selected
    }

    /// Adds a bidirectional connection between nodes.
    ///
    /// # Lock Ordering (BUG-CORE-001 fix)
    ///
    /// This method respects the global lock order: `vectors` → `layers` → `neighbors`
    /// to prevent deadlocks with `search_layer()` which also follows this order.
    #[inline]
    pub(in crate::index::hnsw::native::graph) fn add_bidirectional_connection(
        &self,
        new_node: NodeId,
        neighbor: NodeId,
        layer: usize,
        max_conn: usize,
    ) {
        // Phase 1: Check neighbor count without cloning the full list (F-15)
        let neighbor_count = self.with_layers_read(|layers| {
            layers[layer]
                .with_neighbors(neighbor, <[usize]>::len)
                .unwrap_or(0)
        });

        if neighbor_count < max_conn {
            // Simple case: append if absent under a single node write lock
            self.with_layers_read(|layers| {
                let _ = layers[layer].with_neighbors_mut(neighbor, |neighbors| {
                    if !neighbors.contains(&new_node) {
                        neighbors.push(new_node);
                    }
                });
            });
        } else {
            // Pruning case: get current neighbors + new_node, compute distances
            let mut all_neighbors =
                self.with_layers_read(|layers| layers[layer].get_neighbors(neighbor));
            all_neighbors.push(new_node);

            let mut with_dist: Vec<(NodeId, f32)> = self.with_vectors_read(|vectors| {
                // SAFETY: neighbor is a valid node_id from the graph's neighbor list.
                // - Condition 1: neighbor < vectors.len().
                // Reason: Distance computation for neighbor pruning.
                let neighbor_vec = unsafe { vectors.get_unchecked(neighbor) };
                all_neighbors
                    .iter()
                    .map(|&n| {
                        // SAFETY: n is a valid node_id from the graph's neighbor list
                        // or a just-inserted node_id.
                        // - Condition 1: n < vectors.len().
                        // Reason: Pairwise distance for pruning decision.
                        (
                            n,
                            self.distance
                                .distance(neighbor_vec, unsafe { vectors.get_unchecked(n) }),
                        )
                    })
                    .collect()
            });

            with_dist.sort_by(|a, b| a.1.total_cmp(&b.1));
            let pruned: Vec<NodeId> = with_dist
                .into_iter()
                .take(max_conn)
                .map(|(n, _)| n)
                .collect();

            // Phase 3: Write pruned neighbors under single node write lock
            self.with_layers_read(|layers| {
                let _ = layers[layer].with_neighbors_mut(neighbor, |neighbors| {
                    *neighbors = pruned;
                });
            });
        }
    }
}
