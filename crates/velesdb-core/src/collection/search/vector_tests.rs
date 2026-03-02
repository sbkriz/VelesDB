#![cfg(all(test, feature = "persistence"))]

use crate::{distance::DistanceMetric, point::Point, quantization::StorageMode, Collection};
use std::path::PathBuf;

#[test]
fn test_search_ids_product_quantization_cosine_scores_stay_in_similarity_domain() {
    // ARRANGE
    let temp_dir = tempfile::tempdir().expect("temp dir should be created");
    let collection = Collection::create_with_options(
        PathBuf::from(temp_dir.path()),
        16,
        DistanceMetric::Cosine,
        StorageMode::ProductQuantization,
    )
    .expect("collection should be created");

    let points: Vec<Point> = (0u64..160)
        .map(|id| {
            let mut vector: Vec<f32> = (0..16)
                .map(|d| ((id + 1) as f32 * 0.13 + d as f32 * 0.07).cos())
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

    let query = points[0].vector.clone();
    collection.upsert(points).expect("upsert should succeed");

    // ACT
    let results = collection
        .search_ids(&query, 10)
        .expect("search_ids should succeed");

    // ASSERT
    assert!(!results.is_empty(), "search should return candidates");
    assert!(
        results
            .iter()
            .all(|(_, score)| (-1.0..=1.0).contains(score)),
        "cosine metric scores must remain in similarity domain [-1, 1]"
    );
}
