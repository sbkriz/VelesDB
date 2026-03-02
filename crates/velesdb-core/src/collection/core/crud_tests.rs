#![cfg(all(test, feature = "persistence"))]

use crate::{distance::DistanceMetric, point::Point, quantization::StorageMode, Collection};
use std::path::PathBuf;

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
