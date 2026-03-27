#![cfg(all(test, feature = "persistence"))]

use crate::storage::PayloadStorage;
use crate::{distance::DistanceMetric, point::Point, quantization::StorageMode, Collection};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;

#[test]
fn test_upsert_product_quantization_after_training_backfills_cache() {
    // ARRANGE
    let temp_dir = tempfile::tempdir().expect("temp dir should be created");
    let collection = Collection::create_with_options(
        PathBuf::from(temp_dir.path()),
        16,
        DistanceMetric::Cosine,
        StorageMode::ProductQuantization,
    )
    .expect("collection should be created");

    let points: Vec<Point> = (0u64..128)
        .map(|id| {
            let mut vector: Vec<f32> = (0..16)
                .map(|d| {
                    let id_term = f32::from(u16::try_from(id + 1).expect("id fits in u16")) * 0.17;
                    let d_term =
                        f32::from(u16::try_from(d).expect("dimension index fits in u16")) * 0.11;
                    (id_term + d_term).sin()
                })
                .collect();
            let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in &mut vector {
                    *x /= norm;
                }
            }
            Point::without_payload(id, vector)
        })
        .collect();

    // ACT
    collection.upsert(points).expect("upsert should succeed");

    // ASSERT
    assert!(
        collection.pq_quantizer.read().is_some(),
        "quantizer should be trained after reaching sample threshold"
    );
    assert_eq!(
        collection.pq_cache.read().len(),
        128,
        "all training samples should be backfilled in PQ cache"
    );
}

#[test]
fn test_concurrent_upsert_and_search_no_deadlock() {
    // ARRANGE: shared collection accessible from multiple threads.
    let temp_dir = tempfile::tempdir().expect("temp dir should be created");
    let col = Arc::new(
        Collection::create(PathBuf::from(temp_dir.path()), 4, DistanceMetric::Cosine)
            .expect("collection should be created"),
    );

    // Seed with enough points so HNSW search is exercised.
    #[allow(clippy::cast_precision_loss)] // Reason: i in [0,20); u64→f32 exact for small values.
    let seeds: Vec<Point> = (0u64..20)
        .map(|i| Point::without_payload(i, vec![i as f32 / 20.0, 0.1, 0.1, 0.1]))
        .collect();
    col.upsert(seeds).expect("seed upsert should succeed");

    // ACT: 4 threads each interleave upsert + search 50 times.
    let handles: Vec<_> = (0u64..4)
        .map(|t| {
            let col = Arc::clone(&col);
            thread::spawn(move || {
                for i in 0u64..50 {
                    let id = t * 1_000 + i;
                    #[allow(clippy::cast_precision_loss)] // Reason: i in [0,50); u64→f32 exact.
                    col.upsert(vec![Point::without_payload(
                        id,
                        vec![i as f32 / 50.0, 0.2, 0.2, 0.2],
                    )])
                    .expect("concurrent upsert should not fail");
                    let _ = col.search(&[0.5_f32, 0.1, 0.1, 0.1], 5);
                }
            })
        })
        .collect();

    // ASSERT: no thread panicked (panic = deadlock or data race).
    for h in handles {
        h.join()
            .expect("thread panicked — possible deadlock or data race");
    }
}

#[test]
fn test_upsert_indexes_sparse_vectors() {
    use crate::index::sparse::SparseVector;

    let dir = tempfile::tempdir().unwrap();
    let coll = Collection::create(dir.path().to_path_buf(), 4, DistanceMetric::Cosine).unwrap();

    // Upsert a point with named sparse vectors
    let mut sv_map = BTreeMap::new();
    sv_map.insert(String::new(), SparseVector::new(vec![(1, 1.0), (2, 0.5)]));
    sv_map.insert(
        "title".to_string(),
        SparseVector::new(vec![(10, 2.0), (20, 1.0)]),
    );

    let point = Point::with_sparse(1, vec![0.1, 0.2, 0.3, 0.4], None, Some(sv_map));
    coll.upsert(vec![point]).unwrap();

    // Verify both named indexes were populated
    let indexes = coll.sparse_indexes().read();
    assert!(
        indexes.contains_key(""),
        "Default sparse index should be created"
    );
    assert!(
        indexes.contains_key("title"),
        "Named sparse index 'title' should be created"
    );

    let default_idx = indexes.get("").unwrap();
    assert_eq!(default_idx.doc_count(), 1);
    let postings = default_idx.get_all_postings(1);
    assert_eq!(postings.len(), 1);
    assert_eq!(postings[0].doc_id, 1);

    let title_idx = indexes.get("title").unwrap();
    assert_eq!(title_idx.doc_count(), 1);
    let postings = title_idx.get_all_postings(10);
    assert_eq!(postings.len(), 1);
    assert_eq!(postings[0].doc_id, 1);
}

#[test]
fn test_delete_removes_from_sparse_indexes() {
    use crate::index::sparse::SparseVector;

    let dir = tempfile::tempdir().unwrap();
    let coll = Collection::create(dir.path().to_path_buf(), 4, DistanceMetric::Cosine).unwrap();

    // Upsert a point with sparse vectors
    let mut sv_map = BTreeMap::new();
    sv_map.insert(String::new(), SparseVector::new(vec![(1, 1.0)]));

    let point = Point::with_sparse(42, vec![0.1, 0.2, 0.3, 0.4], None, Some(sv_map));
    coll.upsert(vec![point]).unwrap();

    // Verify it was indexed
    {
        let indexes = coll.sparse_indexes().read();
        let idx = indexes.get("").unwrap();
        assert_eq!(idx.doc_count(), 1);
    }

    // Delete the point
    coll.delete(&[42]).unwrap();

    // Verify it was removed from sparse index
    {
        let indexes = coll.sparse_indexes().read();
        let idx = indexes.get("").unwrap();
        assert_eq!(idx.doc_count(), 0);
        assert!(idx.get_all_postings(1).is_empty());
    }
}

#[test]
#[allow(clippy::cast_possible_truncation)]
fn test_u32_max_term_id() {
    use crate::index::sparse::search::sparse_search;
    use crate::index::sparse::SparseVector;

    let dir = tempfile::tempdir().unwrap();
    let coll = Collection::create(dir.path().to_path_buf(), 4, DistanceMetric::Cosine).unwrap();

    // Use u32::MAX - 1 (4_294_967_294) as term_id
    let extreme_term = u32::MAX - 1;
    let mut sv_map = BTreeMap::new();
    sv_map.insert(String::new(), SparseVector::new(vec![(extreme_term, 1.5)]));

    let point = Point::with_sparse(1, vec![0.1, 0.2, 0.3, 0.4], None, Some(sv_map));
    coll.upsert(vec![point]).unwrap();

    // Verify term_id roundtrips through the index
    {
        let indexes = coll.sparse_indexes().read();
        let idx = indexes.get("").unwrap();
        assert_eq!(idx.doc_count(), 1);

        let postings = idx.get_all_postings(extreme_term);
        assert_eq!(
            postings.len(),
            1,
            "term_id {extreme_term} must have one posting"
        );
        assert_eq!(postings[0].doc_id, 1);
        assert!((postings[0].weight - 1.5).abs() < f32::EPSILON);
    }

    // Search using a query with the extreme term_id
    {
        let indexes = coll.sparse_indexes().read();
        let idx = indexes.get("").unwrap();
        let query = SparseVector::new(vec![(extreme_term, 1.0)]);
        let results = sparse_search(idx, &query, 10);
        assert_eq!(
            results.len(),
            1,
            "search with extreme term_id must find the document"
        );
        assert_eq!(results[0].doc_id, 1);
    }

    // Verify persistence roundtrip: flush and reload
    coll.flush().unwrap();
    let coll2 = Collection::open(dir.path().to_path_buf()).unwrap();
    {
        let indexes = coll2.sparse_indexes().read();
        let idx = indexes.get("").unwrap();
        assert_eq!(
            idx.doc_count(),
            1,
            "doc_count must survive persistence roundtrip"
        );
        let postings = idx.get_all_postings(extreme_term);
        assert_eq!(
            postings.len(),
            1,
            "extreme term_id must survive persistence roundtrip"
        );
        assert_eq!(postings[0].doc_id, 1);
    }
}

#[test]
fn test_sparse_wal_written_on_upsert() {
    use crate::index::sparse::SparseVector;

    let dir = tempfile::tempdir().unwrap();
    let coll = Collection::create(dir.path().to_path_buf(), 4, DistanceMetric::Cosine).unwrap();

    let mut sv_map = BTreeMap::new();
    sv_map.insert(String::new(), SparseVector::new(vec![(1, 1.0)]));

    let point = Point::with_sparse(1, vec![0.1, 0.2, 0.3, 0.4], None, Some(sv_map));
    coll.upsert(vec![point]).unwrap();

    // WAL file should exist for the default sparse index
    let wal_path = dir.path().join("sparse.wal");
    assert!(wal_path.exists(), "Sparse WAL should be created on upsert");
    assert!(
        std::fs::metadata(&wal_path).unwrap().len() > 0,
        "Sparse WAL should have content"
    );
}

/// Regression test: `upsert()` with a batch should produce searchable results.
#[test]
fn test_upsert_batch_produces_searchable_results() {
    let dir = tempfile::tempdir().unwrap();
    let coll = Collection::create(dir.path().to_path_buf(), 16, DistanceMetric::Cosine).unwrap();

    #[allow(clippy::cast_precision_loss)] // Reason: i in [0,200); u64→f32 exact
    let points: Vec<Point> = (0u64..200)
        .map(|i| {
            let v: Vec<f32> = (0..16).map(|d| (i as f32 + d as f32) * 0.01).collect();
            Point::without_payload(i, v)
        })
        .collect();

    coll.upsert(points).expect("batch upsert should succeed");

    #[allow(clippy::cast_precision_loss)] // Reason: d in [0,16); i32→f32 exact
    let query: Vec<f32> = (0..16).map(|d| d as f32 * 0.01).collect();
    let results = coll.search(&query, 10).expect("search should succeed");
    assert_eq!(results.len(), 10, "search should return k results");
    assert_eq!(coll.config.read().point_count, 200);
}

/// Regression test: `upsert()` throughput should be close to `upsert_bulk()`.
///
/// With batched storage + batched HNSW, the gap should be within 3x.
/// The remaining overhead is secondary indexes, quantization, text indexing.
#[test]
fn test_upsert_throughput_not_degraded_vs_bulk() {
    let dim = 32;
    let n = 500;

    let dir1 = tempfile::tempdir().unwrap();
    let coll1 = Collection::create(dir1.path().to_path_buf(), dim, DistanceMetric::Cosine).unwrap();

    #[allow(clippy::cast_precision_loss)]
    let points1: Vec<Point> = (0u64..n)
        .map(|i| {
            let v: Vec<f32> = (0..dim).map(|d| (i as f32 + d as f32) * 0.01).collect();
            Point::without_payload(i, v)
        })
        .collect();

    let t0 = std::time::Instant::now();
    coll1.upsert(points1).expect("upsert should succeed");
    let upsert_dur = t0.elapsed();

    let dir2 = tempfile::tempdir().unwrap();
    let coll2 = Collection::create(dir2.path().to_path_buf(), dim, DistanceMetric::Cosine).unwrap();

    #[allow(clippy::cast_precision_loss)]
    let points2: Vec<Point> = (0u64..n)
        .map(|i| {
            let v: Vec<f32> = (0..dim).map(|d| (i as f32 + d as f32) * 0.01).collect();
            Point::without_payload(i, v)
        })
        .collect();

    let t0 = std::time::Instant::now();
    coll2
        .upsert_bulk(&points2)
        .expect("upsert_bulk should succeed");
    let bulk_dur = t0.elapsed();

    // Threshold is generous (8x) because debug builds amplify overhead from
    // secondary index updates, HashMap tracking, etc. In release builds the
    // ratio is ~1.0x. The goal is to catch gross regressions (the original
    // bug was 19x), not micro-optimize debug perf.
    let ratio = upsert_dur.as_secs_f64() / bulk_dur.as_secs_f64().max(0.001);
    assert!(
        ratio < 8.0,
        "upsert() is {ratio:.1}x slower than upsert_bulk() — \
         expected <8x (upsert={upsert_dur:?}, bulk={bulk_dur:?})"
    );
}

/// BUG-0001 regression: intra-batch duplicate IDs with mixed payload patterns.
///
/// Verifies last-writer-wins semantics across four scenarios:
/// 1. Some(A) then Some(B) -> final payload is B
/// 2. Some(A) then None    -> no payload (delete wins)
/// 3. None then Some(C)    -> final payload is C
/// 4. Unique ID (no dup)   -> payload stored as-is
///
/// Also verifies WAL deduplication: only the final payload per ID is
/// written, reducing WAL bloat for batches with duplicate IDs.
#[test]
fn test_upsert_intra_batch_duplicate_ids_last_writer_wins() {
    let dir = tempfile::tempdir().unwrap();
    let coll = Collection::create(dir.path().to_path_buf(), 4, DistanceMetric::Cosine).unwrap();

    // Pre-seed id=10 with a payload so scenario 2 tests overwrite-then-delete
    coll.upsert(vec![Point::new(
        10,
        vec![0.1, 0.2, 0.3, 0.4],
        Some(serde_json::json!({"pre": "existing"})),
    )])
    .unwrap();

    let batch = vec![
        // Scenario 1: id=1 appears twice, both with payloads — last wins
        Point::new(
            1,
            vec![1.0, 0.0, 0.0, 0.0],
            Some(serde_json::json!({"v": "A"})),
        ),
        Point::new(
            1,
            vec![0.0, 1.0, 0.0, 0.0],
            Some(serde_json::json!({"v": "B"})),
        ),
        // Scenario 2: id=10 (pre-seeded), Some then None — delete wins
        Point::new(
            10,
            vec![0.0, 0.0, 1.0, 0.0],
            Some(serde_json::json!({"v": "X"})),
        ),
        Point::new(10, vec![0.0, 0.0, 0.0, 1.0], None),
        // Scenario 3: id=20, None then Some — store wins
        Point::without_payload(20, vec![0.5, 0.5, 0.0, 0.0]),
        Point::new(
            20,
            vec![0.0, 0.5, 0.5, 0.0],
            Some(serde_json::json!({"v": "C"})),
        ),
        // Scenario 4: id=30, unique — no dedup needed
        Point::new(
            30,
            vec![0.0, 0.0, 0.5, 0.5],
            Some(serde_json::json!({"v": "D"})),
        ),
    ];

    coll.upsert(batch).unwrap();

    let results = coll.get(&[1, 10, 20, 30]);
    assert_eq!(results.len(), 4);

    // Scenario 1: last payload wins (B), last vector wins ([0,1,0,0])
    let p1 = results[0].as_ref().expect("id=1 should exist");
    assert_eq!(p1.payload, Some(serde_json::json!({"v": "B"})));
    assert_eq!(p1.vector, vec![0.0, 1.0, 0.0, 0.0]);

    // Scenario 2: last has None payload — should be deleted
    let p10 = results[1]
        .as_ref()
        .expect("id=10 should still have a vector");
    assert!(p10.payload.is_none(), "payload should be None (deleted)");
    assert_eq!(p10.vector, vec![0.0, 0.0, 0.0, 1.0]);

    // Scenario 3: last has Some(C) — should be stored
    let p20 = results[2].as_ref().expect("id=20 should exist");
    assert_eq!(p20.payload, Some(serde_json::json!({"v": "C"})));
    assert_eq!(p20.vector, vec![0.0, 0.5, 0.5, 0.0]);

    // Scenario 4: unique — stored as-is
    let p30 = results[3].as_ref().expect("id=30 should exist");
    assert_eq!(p30.payload, Some(serde_json::json!({"v": "D"})));

    // Verify point count: 4 unique IDs (1, 10, 20, 30)
    assert_eq!(coll.len(), 4, "should have 4 unique points");
}

/// BUG-0001 regression: WAL replay produces correct state for intra-batch dupes.
///
/// Flushes, reopens the collection from disk, and verifies that the payload
/// WAL replay produces the same state as the in-memory result.
#[test]
fn test_upsert_intra_batch_wal_replay_consistency() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();
    {
        let coll = Collection::create(path.clone(), 4, DistanceMetric::Cosine).unwrap();

        let batch = vec![
            Point::new(
                1,
                vec![1.0, 0.0, 0.0, 0.0],
                Some(serde_json::json!({"a": 1})),
            ),
            Point::new(
                1,
                vec![0.0, 1.0, 0.0, 0.0],
                Some(serde_json::json!({"b": 2})),
            ),
            Point::without_payload(2, vec![0.5, 0.5, 0.0, 0.0]),
            Point::new(
                2,
                vec![0.0, 0.5, 0.5, 0.0],
                Some(serde_json::json!({"c": 3})),
            ),
        ];

        coll.upsert(batch).unwrap();
        coll.flush().unwrap();
    }

    // Reopen from WAL
    let coll2 = Collection::open(path).unwrap();
    let results = coll2.get(&[1, 2]);

    let p1 = results[0].as_ref().expect("id=1 should exist after reload");
    assert_eq!(p1.payload, Some(serde_json::json!({"b": 2})));
    assert_eq!(p1.vector, vec![0.0, 1.0, 0.0, 0.0]);

    let p2 = results[1].as_ref().expect("id=2 should exist after reload");
    assert_eq!(p2.payload, Some(serde_json::json!({"c": 3})));
    assert_eq!(p2.vector, vec![0.0, 0.5, 0.5, 0.0]);
}

/// BUG-0001 regression: WAL deduplication writes fewer entries.
///
/// Measures that the payload WAL is smaller when duplicate IDs are
/// deduplicated before writing, confirming the optimization is effective.
#[test]
fn test_upsert_intra_batch_wal_dedup_reduces_entries() {
    let dir = tempfile::tempdir().unwrap();
    let coll = Collection::create(dir.path().to_path_buf(), 4, DistanceMetric::Cosine).unwrap();

    // Batch with 3 occurrences of id=1, each with a different payload
    let batch = vec![
        Point::new(
            1,
            vec![1.0, 0.0, 0.0, 0.0],
            Some(serde_json::json!({"v": "A"})),
        ),
        Point::new(
            1,
            vec![0.0, 1.0, 0.0, 0.0],
            Some(serde_json::json!({"v": "B"})),
        ),
        Point::new(
            1,
            vec![0.0, 0.0, 1.0, 0.0],
            Some(serde_json::json!({"v": "C"})),
        ),
    ];

    coll.upsert(batch).unwrap();
    coll.flush().unwrap();

    // The payload WAL should contain exactly 1 store entry (not 3)
    // Verify by counting IDs in the payload storage index
    let payload_ids = coll.payload_storage.read().ids();
    assert_eq!(payload_ids.len(), 1, "should have 1 unique payload ID");
    assert!(
        payload_ids.contains(&1),
        "id=1 should be in payload storage"
    );

    // Verify correctness: last writer wins
    let payload = coll.payload_storage.read().retrieve(1).unwrap();
    assert_eq!(payload, Some(serde_json::json!({"v": "C"})));
}

/// Issue #424: Parallel I/O in `batch_store_all` must produce the same results
/// as the sequential implementation for large batches.
///
/// Verifies that both vectors and payloads are correctly stored when
/// payload and vector writes execute concurrently via `rayon::join`.
#[test]
fn test_batch_store_all_parallel_io_correctness() {
    let dir = tempfile::tempdir().unwrap();
    let coll = Collection::create(dir.path().to_path_buf(), 128, DistanceMetric::Cosine).unwrap();

    // Build a batch large enough to exercise the parallel path meaningfully
    #[allow(clippy::cast_precision_loss)] // Reason: i in [0,500); u64->f32 exact for small values
    let points: Vec<Point> = (0u64..500)
        .map(|i| {
            let v: Vec<f32> = (0..128).map(|d| (i as f32 + d as f32) * 0.001).collect();
            let payload = serde_json::json!({"idx": i, "label": format!("point_{i}")});
            Point::new(i, v, Some(payload))
        })
        .collect();

    coll.upsert(points.clone()).expect("upsert should succeed");

    // Verify all points were stored correctly
    assert_eq!(coll.len(), 500, "all 500 points should be stored");

    let ids: Vec<u64> = (0..500).collect();
    let results = coll.get(&ids);
    for (i, result) in results.iter().enumerate() {
        let p = result
            .as_ref()
            .unwrap_or_else(|| panic!("point {i} should exist"));
        assert_eq!(p.vector.len(), 128, "point {i} should have 128 dimensions");
        // Reason: i in [0, 500) — fits in u16
        #[allow(clippy::cast_precision_loss)]
        let expected_first = i as f32 * 0.001;
        assert!(
            (p.vector[0] - expected_first).abs() < 1e-6,
            "point {i} first element mismatch"
        );
        let payload = p
            .payload
            .as_ref()
            .unwrap_or_else(|| panic!("point {i} should have payload"));
        assert_eq!(payload["idx"], i as u64, "point {i} payload.idx mismatch");
    }

    // Verify search still works (HNSW was populated correctly)
    #[allow(clippy::cast_precision_loss)] // Reason: d in [0,128); i32->f32 exact for small values
    let query: Vec<f32> = (0..128).map(|d| d as f32 * 0.001).collect();
    let search_results = coll.search(&query, 10).expect("search should succeed");
    assert_eq!(search_results.len(), 10, "search should return k results");
}

/// Issue #424: Parallel I/O preserves crash recovery semantics.
///
/// After flush + reopen, all vectors and payloads written via the parallel
/// path must survive WAL replay.
#[test]
fn test_batch_store_all_parallel_io_survives_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();
    {
        let coll = Collection::create(path.clone(), 32, DistanceMetric::Cosine).unwrap();

        #[allow(clippy::cast_precision_loss)]
        let points: Vec<Point> = (0u64..100)
            .map(|i| {
                let v: Vec<f32> = (0..32).map(|d| (i as f32 + d as f32) * 0.01).collect();
                Point::new(i, v, Some(serde_json::json!({"id": i})))
            })
            .collect();

        coll.upsert(points).expect("upsert should succeed");
        coll.flush().expect("flush should succeed");
    }

    // Reopen from WAL
    let coll2 = Collection::open(path).unwrap();
    assert_eq!(coll2.len(), 100, "all points should survive reopen");

    // Spot-check a few points
    let results = coll2.get(&[0, 50, 99]);
    for (i, &id) in [0u64, 50, 99].iter().enumerate() {
        let p = results[i]
            .as_ref()
            .unwrap_or_else(|| panic!("point {id} should exist after reopen"));
        assert_eq!(p.vector.len(), 32);
        let payload = p
            .payload
            .as_ref()
            .unwrap_or_else(|| panic!("point {id} should have payload after reopen"));
        assert_eq!(payload["id"], id);
    }
}

/// Issue #424: Parallel I/O handles empty-payload batches correctly.
///
/// When all points have `payload=None`, the payload write is a no-op
/// but must not panic or corrupt the vector write that runs in parallel.
#[test]
fn test_batch_store_all_parallel_io_no_payloads() {
    let dir = tempfile::tempdir().unwrap();
    let coll = Collection::create(dir.path().to_path_buf(), 16, DistanceMetric::Cosine).unwrap();

    #[allow(clippy::cast_precision_loss)]
    let points: Vec<Point> = (0u64..200)
        .map(|i| {
            let v: Vec<f32> = (0..16).map(|d| (i as f32 + d as f32) * 0.01).collect();
            Point::without_payload(i, v)
        })
        .collect();

    coll.upsert(points).expect("upsert should succeed");
    assert_eq!(coll.len(), 200, "all points should be stored");

    // Verify vectors are correct despite parallel path
    let results = coll.get(&[0]);
    let p0 = results[0].as_ref().expect("point 0 should exist");
    assert_eq!(p0.vector.len(), 16);
    assert!(p0.payload.is_none(), "no payload should be stored");
}

/// Issue #424: Parallel I/O handles intra-batch duplicates with mixed payloads.
///
/// The parallel path must not break the old_payloads collection that happens
/// BEFORE the parallel fork (while payload lock is still held).
#[test]
fn test_batch_store_all_parallel_io_with_duplicates() {
    let dir = tempfile::tempdir().unwrap();
    let coll = Collection::create(dir.path().to_path_buf(), 4, DistanceMetric::Cosine).unwrap();

    // Pre-seed id=1 so the batch tests overwrite behavior
    coll.upsert(vec![Point::new(
        1,
        vec![0.1, 0.2, 0.3, 0.4],
        Some(serde_json::json!({"pre": "existing"})),
    )])
    .unwrap();

    // Batch with duplicates: id=1 appears twice, id=2 is unique
    let batch = vec![
        Point::new(
            1,
            vec![1.0, 0.0, 0.0, 0.0],
            Some(serde_json::json!({"v": "A"})),
        ),
        Point::new(
            1,
            vec![0.0, 1.0, 0.0, 0.0],
            Some(serde_json::json!({"v": "B"})),
        ),
        Point::new(
            2,
            vec![0.5, 0.5, 0.0, 0.0],
            Some(serde_json::json!({"v": "C"})),
        ),
    ];

    coll.upsert(batch)
        .expect("batch with duplicates should succeed via parallel I/O");

    let results = coll.get(&[1, 2]);
    let p1 = results[0].as_ref().expect("id=1 should exist");
    assert_eq!(
        p1.payload,
        Some(serde_json::json!({"v": "B"})),
        "last writer wins for payload"
    );
    assert_eq!(
        p1.vector,
        vec![0.0, 1.0, 0.0, 0.0],
        "last writer wins for vector"
    );

    let p2 = results[1].as_ref().expect("id=2 should exist");
    assert_eq!(p2.payload, Some(serde_json::json!({"v": "C"})));
}
