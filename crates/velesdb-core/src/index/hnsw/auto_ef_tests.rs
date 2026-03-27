//! Tests for auto-tuned ef_search range computation.

use super::auto_ef::auto_ef_range;

// =============================================================================
// Small collection tests (≤1K vectors)
// =============================================================================

#[test]
fn test_auto_ef_range_small_collection_low_dim() {
    let (min_ef, max_ef) = auto_ef_range(500, 128, 10);
    // base = k * 2 = 20, dim_factor = 1.0 => min_ef = 20, max_ef = 80
    assert_eq!(min_ef, 20);
    assert_eq!(max_ef, 80);
}

#[test]
fn test_auto_ef_range_small_collection_high_dim() {
    let (min_ef, max_ef) = auto_ef_range(500, 768, 10);
    // base = k * 2 = 20, dim_factor = 1.5 => min_ef = 30, max_ef = 120
    assert_eq!(min_ef, 30);
    assert_eq!(max_ef, 120);
}

// =============================================================================
// Medium collection tests (1K–10K vectors)
// =============================================================================

#[test]
fn test_auto_ef_range_medium_collection_low_dim() {
    let (min_ef, max_ef) = auto_ef_range(5_000, 256, 10);
    // base = k * 4 = 40, dim_factor = 1.0 => min_ef = 40, max_ef = 160
    assert_eq!(min_ef, 40);
    assert_eq!(max_ef, 160);
}

#[test]
fn test_auto_ef_range_medium_collection_high_dim() {
    let (min_ef, max_ef) = auto_ef_range(5_000, 768, 10);
    // base = k * 4 = 40, dim_factor = 1.5 => min_ef = 60, max_ef = 240
    assert_eq!(min_ef, 60);
    assert_eq!(max_ef, 240);
}

// =============================================================================
// Large collection tests (10K–100K vectors)
// =============================================================================

#[test]
fn test_auto_ef_range_large_collection() {
    let (min_ef, max_ef) = auto_ef_range(50_000, 768, 10);
    // base = k * 8 = 80, dim_factor = 1.5 => min_ef = 120, max_ef = 480
    assert_eq!(min_ef, 120);
    assert_eq!(max_ef, 480);
}

// =============================================================================
// Very large collection tests (100K+ vectors)
// =============================================================================

#[test]
fn test_auto_ef_range_very_large_collection() {
    let (min_ef, max_ef) = auto_ef_range(500_000, 768, 10);
    // base = k * 12 = 120, dim_factor = 1.5 => min_ef = 180, max_ef = 720
    assert_eq!(min_ef, 180);
    assert_eq!(max_ef, 720);
}

// =============================================================================
// Edge cases
// =============================================================================

#[test]
fn test_auto_ef_range_empty_collection() {
    let (min_ef, max_ef) = auto_ef_range(0, 128, 10);
    // base = k * 2 = 20, dim_factor = 1.0 => min_ef = 20, max_ef = 80
    assert_eq!(min_ef, 20);
    assert_eq!(max_ef, 80);
}

#[test]
fn test_auto_ef_range_min_ef_at_least_k() {
    // With k=100 and a small collection, base = 200. min_ef should be >= k.
    let (min_ef, _) = auto_ef_range(100, 128, 100);
    assert!(min_ef >= 100, "min_ef must be at least k");
}

#[test]
fn test_auto_ef_range_k_larger_than_base() {
    // k=500, count=100 => base = 500*2 = 1000, min_ef = max(1000, 500) = 1000
    let (min_ef, max_ef) = auto_ef_range(100, 128, 500);
    assert_eq!(min_ef, 1000);
    assert_eq!(max_ef, 4000);
}

#[test]
fn test_auto_ef_range_dimension_boundary() {
    // dim=512 is NOT high-dimensional (factor 1.0), dim=513 IS (factor 1.5)
    let (min_512, _) = auto_ef_range(5_000, 512, 10);
    let (min_513, _) = auto_ef_range(5_000, 513, 10);
    assert!(
        min_513 > min_512,
        "dim > 512 should produce a higher min_ef"
    );
}

#[test]
fn test_auto_ef_range_max_ef_is_four_times_min() {
    for &(count, dim, k) in &[
        (100, 128, 10),
        (5_000, 768, 20),
        (50_000, 1536, 50),
        (500_000, 384, 10),
    ] {
        let (min_ef, max_ef) = auto_ef_range(count, dim, k);
        assert_eq!(
            max_ef,
            min_ef * 4,
            "max_ef should be 4x min_ef for count={count}, dim={dim}, k={k}"
        );
    }
}
