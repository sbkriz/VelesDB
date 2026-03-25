//! Tests for `upsert` module — shared upsert-mapping logic.

use super::sharded_mappings::ShardedMappings;
use super::sharded_vectors::ShardedVectors;
use super::upsert::{rollback_upsert, upsert_mapping, upsert_mapping_batch};

// -------------------------------------------------------------------------
// upsert_mapping_batch tests (issue #375)
// -------------------------------------------------------------------------

#[test]
fn test_upsert_mapping_batch_empty() {
    let mappings = ShardedMappings::new();
    let vectors = ShardedVectors::new(4);
    let results = upsert_mapping_batch(&mappings, &vectors, true, &[]);
    assert!(results.is_empty());
    assert!(mappings.is_empty());
}

#[test]
fn test_upsert_mapping_batch_all_new() {
    let mappings = ShardedMappings::new();
    let vectors = ShardedVectors::new(4);
    let ids = [10, 20, 30];
    let results = upsert_mapping_batch(&mappings, &vectors, true, &ids);

    assert_eq!(results.len(), 3);
    for result in &results {
        assert_eq!(result.old_idx, None, "new IDs should have no old index");
    }
    assert_eq!(mappings.len(), 3);
}

#[test]
fn test_upsert_mapping_batch_all_existing_cleans_stale_vectors() {
    let mappings = ShardedMappings::new();
    let vectors = ShardedVectors::new(4);

    // Pre-insert IDs and store sidecar vectors
    for id in [10, 20, 30] {
        let idx = mappings.register(id).expect("register");
        vectors.insert(idx, &[1.0, 0.0, 0.0, 0.0]);
    }
    assert_eq!(vectors.len(), 3);

    // Batch upsert same IDs
    let results = upsert_mapping_batch(&mappings, &vectors, true, &[10, 20, 30]);

    assert_eq!(results.len(), 3);
    for result in &results {
        assert!(
            result.old_idx.is_some(),
            "existing IDs should have old index"
        );
    }
    // Old sidecar vectors removed
    assert_eq!(vectors.len(), 0, "stale vectors should be cleaned up");
    // Mapping count unchanged
    assert_eq!(mappings.len(), 3);
}

#[test]
fn test_upsert_mapping_batch_no_stale_cleanup_when_storage_disabled() {
    let mappings = ShardedMappings::new();
    let vectors = ShardedVectors::new(4);

    // Pre-insert and store vectors
    for id in [10, 20] {
        let idx = mappings.register(id).expect("register");
        vectors.insert(idx, &[1.0, 0.0, 0.0, 0.0]);
    }

    // Batch upsert with storage disabled: vectors should NOT be removed
    let results = upsert_mapping_batch(&mappings, &vectors, false, &[10, 20]);

    assert_eq!(results.len(), 2);
    assert_eq!(
        vectors.len(),
        2,
        "vectors should not be removed when storage disabled"
    );
}

#[test]
fn test_upsert_mapping_batch_mixed_new_and_existing() {
    let mappings = ShardedMappings::new();
    let vectors = ShardedVectors::new(4);

    // Pre-insert one ID
    let old_idx = mappings.register(20).expect("register");
    vectors.insert(old_idx, &[1.0, 0.0, 0.0, 0.0]);

    let results = upsert_mapping_batch(&mappings, &vectors, true, &[10, 20, 30]);

    assert_eq!(results.len(), 3);
    // ID 10: new
    assert_eq!(results[0].old_idx, None);
    // ID 20: replaced
    assert_eq!(results[1].old_idx, Some(old_idx));
    // ID 30: new
    assert_eq!(results[2].old_idx, None);

    assert_eq!(mappings.len(), 3);
    // Stale vector for old_idx removed
    assert!(vectors.get(old_idx).is_none());
}

#[test]
fn test_upsert_mapping_batch_single_element() {
    let mappings = ShardedMappings::new();
    let vectors = ShardedVectors::new(4);

    let results = upsert_mapping_batch(&mappings, &vectors, true, &[42]);

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].old_idx, None);
    assert_eq!(mappings.len(), 1);
}

// -------------------------------------------------------------------------
// upsert_mapping_batch + rollback integration
// -------------------------------------------------------------------------

#[test]
fn test_upsert_mapping_batch_rollback_restores_old_mappings() {
    let mappings = ShardedMappings::new();
    let vectors = ShardedVectors::new(4);

    // Pre-insert ID 10
    let original_idx = mappings.register(10).expect("register");

    // Batch upsert replaces ID 10
    let results = upsert_mapping_batch(&mappings, &vectors, false, &[10]);
    assert_eq!(results[0].old_idx, Some(original_idx));
    let new_idx = results[0].idx;
    assert_ne!(new_idx, original_idx);

    // Simulate failure: rollback
    rollback_upsert(&mappings, 10, &results[0]);

    // Old mapping restored
    assert_eq!(mappings.get_idx(10), Some(original_idx));
    assert_eq!(mappings.get_id(original_idx), Some(10));
    // New mapping gone
    assert_eq!(mappings.get_id(new_idx), None);
}

#[test]
fn test_upsert_mapping_batch_rollback_reverse_order() {
    let mappings = ShardedMappings::new();
    let vectors = ShardedVectors::new(4);

    // Pre-insert IDs 10 and 20
    let idx_10 = mappings.register(10).expect("register");
    let idx_20 = mappings.register(20).expect("register");

    // Batch upsert replaces both
    let results = upsert_mapping_batch(&mappings, &vectors, false, &[10, 20]);

    // Rollback in reverse order (as the production code does)
    for (id, result) in [10_u64, 20].iter().zip(results.iter()).rev() {
        rollback_upsert(&mappings, *id, result);
    }

    // Both original mappings restored
    assert_eq!(mappings.get_idx(10), Some(idx_10));
    assert_eq!(mappings.get_idx(20), Some(idx_20));
}

// -------------------------------------------------------------------------
// Consistency: batch result matches sequential upsert_mapping calls
// -------------------------------------------------------------------------

#[test]
fn test_upsert_mapping_batch_matches_sequential() {
    let ids = [100, 200, 300];

    // Sequential path
    let seq_mappings = ShardedMappings::new();
    let seq_vectors = ShardedVectors::new(4);
    let seq_results: Vec<_> = ids
        .iter()
        .map(|&id| upsert_mapping(&seq_mappings, &seq_vectors, true, id))
        .collect();

    // Batch path
    let batch_mappings = ShardedMappings::new();
    let batch_vectors = ShardedVectors::new(4);
    let batch_results = upsert_mapping_batch(&batch_mappings, &batch_vectors, true, &ids);

    // Results should be identical for all-new IDs
    assert_eq!(seq_results.len(), batch_results.len());
    for (seq, batch) in seq_results.iter().zip(batch_results.iter()) {
        assert_eq!(seq.idx, batch.idx, "indices should match");
        assert_eq!(seq.old_idx, batch.old_idx, "old_idx should match");
    }
}
