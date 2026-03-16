//! Extended tests for [`ClusteredIndex`]: insert/remove/get_neighbors,
//! compaction, and fragmentation ratio calculation.

use super::clustered_index::ClusteredIndex;

// ─────────────────────────────────────────────────────────────────────────────
// Insert / remove / get_neighbors
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn insert_creates_new_entry() {
    let mut idx = ClusteredIndex::new();
    idx.insert(100, 200);
    assert_eq!(idx.get_neighbors(100), &[200]);
    assert_eq!(idx.node_count(), 1);
    assert_eq!(idx.edge_count(), 1);
}

#[test]
fn insert_appends_to_existing_node() {
    let mut idx = ClusteredIndex::new();
    idx.insert(1, 10);
    idx.insert(1, 20);
    idx.insert(1, 30);

    let neighbors = idx.get_neighbors(1);
    assert_eq!(neighbors.len(), 3);
    assert!(neighbors.contains(&10));
    assert!(neighbors.contains(&20));
    assert!(neighbors.contains(&30));
}

#[test]
fn insert_deduplicates() {
    let mut idx = ClusteredIndex::new();
    idx.insert(1, 10);
    idx.insert(1, 10);
    assert_eq!(idx.neighbor_count(1), 1);
}

#[test]
fn remove_returns_true_when_present() {
    let mut idx = ClusteredIndex::new();
    idx.insert(1, 10);
    idx.insert(1, 20);
    assert!(idx.remove(1, 10));
    assert!(!idx.contains(1, 10));
    assert!(idx.contains(1, 20));
}

#[test]
fn remove_returns_false_when_absent() {
    let mut idx = ClusteredIndex::new();
    idx.insert(1, 10);
    assert!(!idx.remove(1, 999));
    assert!(!idx.remove(999, 10));
}

#[test]
fn remove_last_neighbor_deletes_node() {
    let mut idx = ClusteredIndex::new();
    idx.insert(1, 10);
    assert!(idx.remove(1, 10));
    assert_eq!(idx.node_count(), 0);
    assert!(idx.get_neighbors(1).is_empty());
}

#[test]
fn remove_node_clears_all_neighbors() {
    let mut idx = ClusteredIndex::new();
    idx.insert(1, 10);
    idx.insert(1, 20);
    idx.insert(1, 30);
    idx.remove_node(1);

    assert_eq!(idx.node_count(), 0);
    assert!(idx.get_neighbors(1).is_empty());
}

#[test]
fn get_neighbors_unknown_node_returns_empty() {
    let idx = ClusteredIndex::new();
    assert!(idx.get_neighbors(42).is_empty());
}

// ─────────────────────────────────────────────────────────────────────────────
// Compact reduces fragmentation
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn compact_reduces_fragmentation_to_zero() {
    let mut idx = ClusteredIndex::new();
    for node in 0..10 {
        for target in 0..5 {
            idx.insert(node, target * 100);
        }
    }
    // Remove half the nodes to create fragmentation
    for node in 0..5 {
        idx.remove_node(node);
    }
    assert!(idx.fragmentation() > 0.0, "should have fragmentation");

    idx.compact();
    assert!(
        idx.fragmentation().abs() < f64::EPSILON,
        "compaction should eliminate fragmentation"
    );
    assert_eq!(idx.node_count(), 5);
    assert_eq!(idx.edge_count(), 25);
}

#[test]
fn compact_preserves_all_data() {
    let mut idx = ClusteredIndex::new();
    idx.insert(1, 10);
    idx.insert(1, 20);
    idx.insert(2, 100);
    idx.insert(3, 200);

    // Remove node 2 to create a gap, then compact
    idx.remove_node(2);
    idx.compact();

    assert_eq!(idx.node_count(), 2);
    assert!(idx.contains(1, 10));
    assert!(idx.contains(1, 20));
    assert!(idx.contains(3, 200));
    assert!(!idx.contains(2, 100));
}

#[test]
fn compact_noop_when_no_fragmentation() {
    let mut idx = ClusteredIndex::new();
    idx.insert(1, 10);
    assert!(idx.fragmentation().abs() < f64::EPSILON);
    idx.compact(); // should not crash
    assert_eq!(idx.neighbor_count(1), 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// Fragmentation ratio calculation
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn fragmentation_empty_index_is_zero() {
    let idx = ClusteredIndex::new();
    assert!(idx.fragmentation().abs() < f64::EPSILON);
}

#[test]
fn fragmentation_no_deletes_is_zero() {
    let mut idx = ClusteredIndex::new();
    idx.insert(1, 10);
    idx.insert(2, 20);
    assert!(idx.fragmentation().abs() < f64::EPSILON);
}

#[test]
fn fragmentation_increases_after_removal() {
    let mut idx = ClusteredIndex::new();
    for i in 0..10 {
        idx.insert(i, i * 10);
    }
    let before = idx.fragmentation();
    idx.remove_node(0);
    let after = idx.fragmentation();
    assert!(after > before);
}

// ─────────────────────────────────────────────────────────────────────────────
// with_capacity
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn with_capacity_works() {
    let mut idx = ClusteredIndex::with_capacity(100, 500);
    idx.insert(1, 10);
    assert_eq!(idx.node_count(), 1);
}
