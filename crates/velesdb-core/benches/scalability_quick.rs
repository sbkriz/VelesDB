//! Quick scalability benchmark for CI (100K vectors, minimal config).
//!
//! Runs weekly and on push to main to detect scalability regressions
//! without the multi-hour cost of the full scalability benchmark.
//!
//! # Usage
//!
//! ```bash
//! cargo bench -p velesdb-core --bench scalability_quick --features internal-bench -- --noplot
//! ```

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

#[path = "bench_helpers.rs"]
mod bench_helpers;

use bench_helpers::generate_normalized_vector;
use criterion::{black_box, criterion_group, criterion_main, Criterion, SamplingMode};
use std::time::{Duration, Instant};
use velesdb_core::{DistanceMetric, HnswIndex, VectorIndex};

/// 100K vectors -- large enough to surface O(n log n) regressions,
/// small enough to finish in a few minutes on CI.
const DATASET_SIZE: usize = 100_000;

/// 128-d is the lightest common embedding dimension.
/// Keeps wall-clock time down while still exercising SIMD paths.
const DIMENSION: usize = 128;

/// Number of search queries per measurement iteration.
const QUERIES: usize = 20;

/// Top-K results requested per search.
const TOP_K: usize = 10;

/// Builds and returns a pre-populated HNSW index with `DATASET_SIZE` vectors.
fn build_index() -> HnswIndex {
    let index = HnswIndex::new(DIMENSION, DistanceMetric::Cosine).unwrap();
    for i in 0..DATASET_SIZE {
        let vector = generate_normalized_vector(DIMENSION, i as u64);
        index.insert(i as u64, &vector);
    }
    index
}

/// Measures insert throughput: build a fresh 100K-vector HNSW index from scratch.
fn bench_insert_100k(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalability_quick_insert");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    group.bench_function("100k_128d", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let index = HnswIndex::new(DIMENSION, DistanceMetric::Cosine).unwrap();
                let start = Instant::now();
                for i in 0..DATASET_SIZE {
                    let vector = generate_normalized_vector(DIMENSION, i as u64);
                    index.insert(i as u64, &vector);
                }
                total += start.elapsed();
                black_box(index.len());
            }
            total
        });
    });

    group.finish();
}

/// Measures search latency against a pre-built 100K index.
fn bench_search_100k(c: &mut Criterion) {
    let index = build_index();

    let mut group = c.benchmark_group("scalability_quick_search");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    group.bench_function("top10_100k_128d", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for iter in 0..iters {
                let start = Instant::now();
                for q in 0..QUERIES {
                    let seed = iter * (QUERIES as u64) + (q as u64) + 1_000_000;
                    let query = generate_normalized_vector(DIMENSION, seed);
                    black_box(index.search(&query, TOP_K));
                }
                total += start.elapsed();
            }
            total
        });
    });

    group.finish();
}

criterion_group! {
    name = scalability_quick;
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(Duration::from_secs(10));
    targets = bench_insert_100k, bench_search_100k
}
criterion_main!(scalability_quick);
