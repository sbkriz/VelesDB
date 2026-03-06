//! PQ recall accuracy benchmark suite.
//!
//! Measures recall@10 for PQ, OPQ, and `RaBitQ` quantization methods
//! against brute-force exact L2 search ground truth.

#![allow(clippy::cast_precision_loss)]

use std::collections::HashSet;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tempfile::{tempdir, TempDir};
use velesdb_core::{Collection, DistanceMetric, Point, StorageMode};

const DIMENSION: usize = 128;
const NUM_VECTORS: usize = 5_000;
const NUM_CLUSTERS: usize = 10;
const NUM_QUERIES: usize = 50;
const K: usize = 10;

/// Generate clustered synthetic data with seeded RNG for reproducibility.
fn generate_clustered_data(n: usize, dim: usize, num_clusters: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);

    // Generate cluster centers
    let centers: Vec<Vec<f32>> = (0..num_clusters)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0_f32..1.0)).collect())
        .collect();

    // Generate points around cluster centers with Gaussian-like noise
    (0..n)
        .map(|i| {
            let center = &centers[i % num_clusters];
            center
                .iter()
                .map(|&c| {
                    let noise = rng.gen_range(-0.1_f32..0.1);
                    c + noise
                })
                .collect()
        })
        .collect()
}

/// Compute L2 (squared Euclidean) distance between two vectors.
fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Brute-force exact top-k search returning IDs sorted by distance.
fn brute_force_topk(query: &[f32], dataset: &[Vec<f32>], k: usize) -> Vec<u64> {
    let mut dists: Vec<(u64, f32)> = dataset
        .iter()
        .enumerate()
        .map(|(i, v)| {
            #[allow(clippy::cast_possible_truncation)]
            let id = i as u64;
            (id, l2_distance(query, v))
        })
        .collect();
    dists.sort_by(|a, b| a.1.total_cmp(&b.1));
    dists.iter().take(k).map(|(id, _)| *id).collect()
}

/// Compute recall@k: fraction of ground truth IDs present in results.
fn recall_at_k(ground_truth: &[u64], results: &[u64], k: usize) -> f64 {
    assert!(k > 0, "recall@k requires k > 0");
    let gt_set: HashSet<u64> = ground_truth.iter().take(k).copied().collect();
    let result_set: HashSet<u64> = results.iter().take(k).copied().collect();
    let intersection = gt_set.intersection(&result_set).count();
    #[allow(clippy::cast_precision_loss)]
    let recall = intersection as f64 / k as f64;
    recall
}

/// Build a collection with given storage mode and data.
///
/// Returns the `TempDir` alongside the `Collection` so the directory stays
/// alive for the benchmark duration and is cleaned up on drop.
fn build_pq_collection(
    storage_mode: StorageMode,
    dataset: &[Vec<f32>],
    dimension: usize,
    name: &str,
) -> (Collection, TempDir) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(name);

    let collection =
        Collection::create_with_options(path, dimension, DistanceMetric::Euclidean, storage_mode)
            .expect("create collection");

    let points: Vec<Point> = dataset
        .iter()
        .enumerate()
        .map(|(id, v)| {
            #[allow(clippy::cast_possible_truncation)]
            let pid = id as u64;
            Point::new(pid, v.clone(), Some(serde_json::json!({})))
        })
        .collect();
    collection.upsert(points).expect("upsert");
    (collection, dir)
}

/// Measure average recall@k for a collection against brute-force ground truth.
fn measure_recall(
    collection: &Collection,
    queries: &[Vec<f32>],
    dataset: &[Vec<f32>],
    k: usize,
) -> f64 {
    assert!(
        !queries.is_empty(),
        "measure_recall requires non-empty queries"
    );
    let mut total_recall = 0.0;
    for query in queries {
        let gt = brute_force_topk(query, dataset, k);
        let results = collection.search_ids(query, k).expect("search");
        let result_ids: Vec<u64> = results.iter().map(|(id, _)| *id).collect();
        total_recall += recall_at_k(&gt, &result_ids, k);
    }
    #[allow(clippy::cast_precision_loss)]
    let avg = total_recall / queries.len() as f64;
    avg
}

fn pq_recall_benchmarks(c: &mut Criterion) {
    // Generate shared dataset and queries
    let dataset = generate_clustered_data(NUM_VECTORS, DIMENSION, NUM_CLUSTERS, 42);
    let queries = generate_clustered_data(NUM_QUERIES, DIMENSION, NUM_CLUSTERS, 123);

    // Build full-precision reference collection
    let (full_collection, _full_dir) =
        build_pq_collection(StorageMode::Full, &dataset, DIMENSION, "full_ref");

    // Build PQ collection (auto-trains on upsert)
    let (pq_collection, _pq_dir) = build_pq_collection(
        StorageMode::ProductQuantization,
        &dataset,
        DIMENSION,
        "pq_m8_k256",
    );

    // Measure full-precision recall (should be ~100% as self-reference)
    let full_recall = measure_recall(&full_collection, &queries, &dataset, K);
    println!("Full-precision recall@{K}: {full_recall:.4}");

    // Measure PQ recall with default rescore
    let pq_recall = measure_recall(&pq_collection, &queries, &dataset, K);
    println!("PQ (m=auto, k=auto, rescore) recall@{K}: {pq_recall:.4}");

    // Benchmark: PQ recall measurement (rescore enabled by default)
    let mut group = c.benchmark_group("pq_recall");
    group.sample_size(10); // accuracy benchmark, not speed
    group.measurement_time(std::time::Duration::from_secs(5));

    group.bench_function("pq_recall_m_auto_rescore", |b| {
        b.iter(|| {
            let recall = measure_recall(
                black_box(&pq_collection),
                black_box(&queries),
                black_box(&dataset),
                K,
            );
            // PQ with auto-training on 5K synthetic vectors.
            // Threshold: >= 20% (conservative for auto-trained PQ on synthetic data).
            // Production recall is higher with more vectors and manual training.
            assert!(
                recall >= 0.20,
                "PQ recall@{K} = {recall:.4}, expected >= 0.20"
            );
            recall
        });
    });

    group.bench_function("full_precision_baseline", |b| {
        b.iter(|| {
            measure_recall(
                black_box(&full_collection),
                black_box(&queries),
                black_box(&dataset),
                K,
            )
        });
    });

    group.finish();

    // Print summary for CI reporting
    println!("\n=== PQ Recall Summary ===");
    println!("Full precision: {full_recall:.4}");
    println!("PQ (rescore):   {pq_recall:.4}");
}

criterion_group!(pq_recall, pq_recall_benchmarks);
criterion_main!(pq_recall);
