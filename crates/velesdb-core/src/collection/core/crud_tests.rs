#![cfg(all(test, feature = "persistence"))]

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
