//! Tests for PDX columnar vector storage and block-parallel distance kernels.

#![allow(clippy::cast_precision_loss)]
// Reason: test loops need the index for both distance-array lookup and cv.get(i)
#![allow(clippy::needless_range_loop)]

use super::columnar_distance::{block_cosine_distance, block_dot_product, block_squared_l2};
use super::columnar_vectors::{ColumnarVectors, PDX_BLOCK_SIZE};
use crate::perf_optimizations::ContiguousVectors;

// ---------------------------------------------------------------------------
// ColumnarVectors structure tests
// ---------------------------------------------------------------------------

#[test]
fn test_empty_vectors() {
    let cv = ContiguousVectors::new(128, 16).unwrap();
    let pdx = ColumnarVectors::from_contiguous(&cv);

    assert_eq!(pdx.len(), 0);
    assert!(pdx.is_empty());
    assert_eq!(pdx.block_count(), 0);
    assert_eq!(pdx.dimension(), 128);
}

#[test]
fn test_block_count_exact_multiple() {
    let dim = 4;
    let n = PDX_BLOCK_SIZE * 3; // 192 vectors = exactly 3 blocks
    let mut cv = ContiguousVectors::new(dim, n).unwrap();
    for i in 0..n {
        let v: Vec<f32> = (0..dim).map(|d| (i * dim + d) as f32).collect();
        cv.push(&v).unwrap();
    }

    let pdx = ColumnarVectors::from_contiguous(&cv);
    assert_eq!(pdx.block_count(), 3);
    assert_eq!(pdx.len(), n);
    assert!(!pdx.is_empty());
    assert_eq!(pdx.block_size(0), PDX_BLOCK_SIZE);
    assert_eq!(pdx.block_size(1), PDX_BLOCK_SIZE);
    assert_eq!(pdx.block_size(2), PDX_BLOCK_SIZE);
}

#[test]
fn test_block_count_partial_last_block() {
    let dim = 4;
    let n = PDX_BLOCK_SIZE + 10; // 74 vectors = 2 blocks, last has 10
    let mut cv = ContiguousVectors::new(dim, n).unwrap();
    for i in 0..n {
        let v: Vec<f32> = (0..dim).map(|d| (i * dim + d) as f32).collect();
        cv.push(&v).unwrap();
    }

    let pdx = ColumnarVectors::from_contiguous(&cv);
    assert_eq!(pdx.block_count(), 2);
    assert_eq!(pdx.block_size(0), PDX_BLOCK_SIZE);
    assert_eq!(pdx.block_size(1), 10);
}

#[test]
fn test_single_vector() {
    let dim = 8;
    let mut cv = ContiguousVectors::new(dim, 16).unwrap();
    let v: Vec<f32> = (0..dim).map(|d| d as f32 + 1.0).collect();
    cv.push(&v).unwrap();

    let pdx = ColumnarVectors::from_contiguous(&cv);
    assert_eq!(pdx.block_count(), 1);
    assert_eq!(pdx.block_size(0), 1);
    assert_eq!(pdx.len(), 1);
}

#[test]
fn test_from_contiguous_roundtrip() {
    let dim = 16;
    let n = 100;
    let mut cv = ContiguousVectors::new(dim, n).unwrap();
    for i in 0..n {
        let v: Vec<f32> = (0..dim)
            .map(|d| ((i * dim + d) as f32 * 0.1).sin())
            .collect();
        cv.push(&v).unwrap();
    }

    let pdx = ColumnarVectors::from_contiguous(&cv);

    // Verify every vector can be reconstructed from PDX layout
    for vec_idx in 0..n {
        let block_idx = vec_idx / PDX_BLOCK_SIZE;
        let local = vec_idx % PDX_BLOCK_SIZE;
        let block = pdx.block_ptr(block_idx);
        let original = cv.get(vec_idx).unwrap();

        for d in 0..dim {
            let pdx_val = block[d * PDX_BLOCK_SIZE + local];
            let delta = (pdx_val - original[d]).abs();
            assert!(
                delta < f32::EPSILON,
                "Mismatch at vec={vec_idx}, dim={d}: pdx={pdx_val}, orig={}",
                original[d],
            );
        }
    }
}

#[test]
fn test_block_ptr_length() {
    let dim = 32;
    let n = PDX_BLOCK_SIZE + 5;
    let mut cv = ContiguousVectors::new(dim, n).unwrap();
    for i in 0..n {
        let v: Vec<f32> = vec![i as f32; dim];
        cv.push(&v).unwrap();
    }

    let pdx = ColumnarVectors::from_contiguous(&cv);
    // Both blocks have the same slice length (full block, zero-padded)
    assert_eq!(pdx.block_ptr(0).len(), PDX_BLOCK_SIZE * dim);
    assert_eq!(pdx.block_ptr(1).len(), PDX_BLOCK_SIZE * dim);
}

// ---------------------------------------------------------------------------
// Block distance kernel tests
// ---------------------------------------------------------------------------

/// Helper: build a `ContiguousVectors` + PDX pair for distance tests.
fn build_test_data(dim: usize, n: usize) -> (ContiguousVectors, ColumnarVectors) {
    let mut cv = ContiguousVectors::new(dim, n).unwrap();
    for i in 0..n {
        let v: Vec<f32> = (0..dim)
            .map(|d| ((i * dim + d) as f32 * 0.017).sin())
            .collect();
        cv.push(&v).unwrap();
    }
    let pdx = ColumnarVectors::from_contiguous(&cv);
    (cv, pdx)
}

/// Helper: scalar squared L2 between two slices.
fn reference_squared_l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Helper: scalar dot product between two slices.
fn reference_dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Helper: scalar cosine distance (1 - similarity) between two slices.
fn reference_cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < f32::EPSILON || norm_b < f32::EPSILON {
        return 1.0;
    }
    1.0 - (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

/// Helper: assert that a distance value matches expected within tolerance.
fn assert_distance_eq(actual: f32, expected: f32, tol: f32, context: &str) {
    let delta = (actual - expected).abs();
    assert!(
        delta < tol,
        "{context}: actual={actual}, expected={expected}, delta={delta}",
    );
}

/// Helper: assert that a padding slot is effectively zero.
fn assert_padding_zero(value: f32, slot: usize) {
    assert!(
        value.abs() < f32::EPSILON,
        "Padding slot {slot} should be zero, got {value}",
    );
}

#[test]
fn test_block_squared_l2_matches_sequential() {
    let dim = 128;
    let n = PDX_BLOCK_SIZE; // One full block
    let (cv, pdx) = build_test_data(dim, n);

    let query: Vec<f32> = (0..dim).map(|d| (d as f32 * 0.03).cos()).collect();
    let block = pdx.block_ptr(0);
    let distances = block_squared_l2(&query, block, dim, n);

    for i in 0..n {
        let vec = cv.get(i).unwrap();
        let expected = reference_squared_l2(&query, vec);
        assert_distance_eq(
            distances[i],
            expected,
            1e-3,
            &format!("Squared L2 at vec {i}"),
        );
    }
}

#[test]
fn test_block_dot_product_matches_sequential() {
    let dim = 128;
    let n = PDX_BLOCK_SIZE;
    let (cv, pdx) = build_test_data(dim, n);

    let query: Vec<f32> = (0..dim).map(|d| (d as f32 * 0.03).cos()).collect();
    let block = pdx.block_ptr(0);
    let neg_dots = block_dot_product(&query, block, dim, n);

    for i in 0..n {
        let vec = cv.get(i).unwrap();
        let expected = -reference_dot_product(&query, vec);
        assert_distance_eq(
            neg_dots[i],
            expected,
            1e-3,
            &format!("Dot product at vec {i}"),
        );
    }
}

#[test]
fn test_block_cosine_matches_sequential() {
    let dim = 128;
    let n = PDX_BLOCK_SIZE;
    let (cv, pdx) = build_test_data(dim, n);

    let query: Vec<f32> = (0..dim).map(|d| (d as f32 * 0.03).cos()).collect();
    let block = pdx.block_ptr(0);
    let cosine_dists = block_cosine_distance(&query, block, dim, n);

    for i in 0..n {
        let vec = cv.get(i).unwrap();
        let expected = reference_cosine_distance(&query, vec);
        assert_distance_eq(
            cosine_dists[i],
            expected,
            1e-4,
            &format!("Cosine distance at vec {i}"),
        );
    }
}

#[test]
fn test_block_squared_l2_partial_block() {
    let dim = 32;
    let n = 17; // Partial block: only 17 vectors
    let (cv, pdx) = build_test_data(dim, n);

    let query: Vec<f32> = vec![1.0; dim];
    let block = pdx.block_ptr(0);
    let distances = block_squared_l2(&query, block, dim, n);

    // Valid slots match reference
    for i in 0..n {
        let vec = cv.get(i).unwrap();
        let expected = reference_squared_l2(&query, vec);
        assert_distance_eq(
            distances[i],
            expected,
            1e-3,
            &format!("Partial block L2 at vec {i}"),
        );
    }

    // Padding slots are zeroed
    for i in n..PDX_BLOCK_SIZE {
        assert_padding_zero(distances[i], i);
    }
}

#[test]
fn test_block_dot_product_partial_block() {
    let dim = 32;
    let n = 17;
    let (cv, pdx) = build_test_data(dim, n);

    let query: Vec<f32> = vec![1.0; dim];
    let block = pdx.block_ptr(0);
    let neg_dots = block_dot_product(&query, block, dim, n);

    for i in 0..n {
        let vec = cv.get(i).unwrap();
        let expected = -reference_dot_product(&query, vec);
        assert_distance_eq(
            neg_dots[i],
            expected,
            1e-3,
            &format!("Partial block dot at vec {i}"),
        );
    }

    for i in n..PDX_BLOCK_SIZE {
        assert_padding_zero(neg_dots[i], i);
    }
}

#[test]
fn test_block_cosine_partial_block() {
    let dim = 32;
    let n = 17;
    let (cv, pdx) = build_test_data(dim, n);

    let query: Vec<f32> = vec![1.0; dim];
    let block = pdx.block_ptr(0);
    let cosine_dists = block_cosine_distance(&query, block, dim, n);

    for i in 0..n {
        let vec = cv.get(i).unwrap();
        let expected = reference_cosine_distance(&query, vec);
        assert_distance_eq(
            cosine_dists[i],
            expected,
            1e-4,
            &format!("Partial block cosine at vec {i}"),
        );
    }
}

#[test]
fn test_multi_block_squared_l2() {
    let dim = 64;
    let n = PDX_BLOCK_SIZE * 2 + 7; // 135 vectors across 3 blocks
    let (cv, pdx) = build_test_data(dim, n);

    let query: Vec<f32> = (0..dim).map(|d| (d as f32 * 0.05).sin()).collect();

    for block_idx in 0..pdx.block_count() {
        let bs = pdx.block_size(block_idx);
        let block = pdx.block_ptr(block_idx);
        let distances = block_squared_l2(&query, block, dim, bs);

        for local in 0..bs {
            let vec_idx = block_idx * PDX_BLOCK_SIZE + local;
            let vec = cv.get(vec_idx).unwrap();
            let expected = reference_squared_l2(&query, vec);
            assert_distance_eq(
                distances[local],
                expected,
                1e-3,
                &format!("Multi-block L2 at block={block_idx}, local={local}"),
            );
        }
    }
}

#[test]
fn test_high_dimension_768d() {
    let dim = 768;
    let n = PDX_BLOCK_SIZE;
    let (cv, pdx) = build_test_data(dim, n);

    let query: Vec<f32> = (0..dim).map(|d| (d as f32 * 0.001).sin()).collect();
    let block = pdx.block_ptr(0);

    let l2_dists = block_squared_l2(&query, block, dim, n);
    let neg_dots = block_dot_product(&query, block, dim, n);
    let cos_dists = block_cosine_distance(&query, block, dim, n);

    // Spot-check a few vectors
    for i in [0, 1, 31, 63] {
        let vec = cv.get(i).unwrap();

        let expected_l2 = reference_squared_l2(&query, vec);
        assert_distance_eq(l2_dists[i], expected_l2, 1e-2, &format!("768D L2 at {i}"));

        let expected_dot = -reference_dot_product(&query, vec);
        assert_distance_eq(neg_dots[i], expected_dot, 1e-2, &format!("768D dot at {i}"));

        let expected_cos = reference_cosine_distance(&query, vec);
        assert_distance_eq(
            cos_dists[i],
            expected_cos,
            1e-3,
            &format!("768D cosine at {i}"),
        );
    }
}
