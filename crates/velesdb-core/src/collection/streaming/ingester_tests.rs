//! Extended tests for [`StreamIngester`] backpressure, shutdown drain,
//! and configuration defaults.

use super::ingester::{BackpressureError, StreamIngester, StreamingConfig};
use crate::collection::types::Collection;
use crate::distance::DistanceMetric;
use crate::point::Point;
use tempfile::TempDir;

/// Helper: create a test collection in a temp directory.
fn test_collection(dim: usize) -> (TempDir, Collection) {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("test_ingester_coll");
    let coll = Collection::create(path, dim, DistanceMetric::Cosine).expect("create collection");
    (dir, coll)
}

/// Helper: create a point with the given id and dimension.
fn make_point(id: u64, dim: usize) -> Point {
    Point {
        id,
        vector: vec![0.1_f32; dim],
        payload: None,
        sparse_vectors: None,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Config defaults
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn streaming_config_defaults() {
    let cfg = StreamingConfig::default();
    assert_eq!(cfg.buffer_size, 10_000);
    assert_eq!(cfg.batch_size, 128);
    assert_eq!(cfg.flush_interval_ms, 50);
}

#[test]
fn streaming_config_clone_is_identical() {
    let cfg = StreamingConfig {
        buffer_size: 42,
        batch_size: 7,
        flush_interval_ms: 100,
    };
    let cloned = cfg.clone();
    assert_eq!(cloned.buffer_size, 42);
    assert_eq!(cloned.batch_size, 7);
    assert_eq!(cloned.flush_interval_ms, 100);
}

// ─────────────────────────────────────────────────────────────────────────────
// try_send with backpressure
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn ingester_try_send_buffer_full_error_display() {
    let (_dir, coll) = test_collection(4);
    let config = StreamingConfig {
        buffer_size: 1,
        batch_size: 1000,
        flush_interval_ms: 60_000,
    };
    let ingester = StreamIngester::new(coll, config);

    // Fill the single-slot channel
    assert!(ingester.try_send(make_point(1, 4)).is_ok());

    let err = ingester.try_send(make_point(2, 4)).unwrap_err();
    assert!(
        matches!(err, BackpressureError::BufferFull),
        "expected BufferFull, got: {err}"
    );
    // Verify Display impl
    let msg = format!("{err}");
    assert!(
        msg.contains("full"),
        "error message should mention 'full': {msg}"
    );

    ingester.shutdown().await;
}

#[tokio::test]
async fn ingester_config_accessor() {
    let (_dir, coll) = test_collection(4);
    let config = StreamingConfig {
        buffer_size: 42,
        batch_size: 7,
        flush_interval_ms: 99,
    };
    let ingester = StreamIngester::new(coll, config);

    let c = ingester.config();
    assert_eq!(c.buffer_size, 42);
    assert_eq!(c.batch_size, 7);
    assert_eq!(c.flush_interval_ms, 99);

    ingester.shutdown().await;
}

// ─────────────────────────────────────────────────────────────────────────────
// Shutdown drains pending items
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn ingester_shutdown_drains_all_pending() {
    let (_dir, coll) = test_collection(4);
    let config = StreamingConfig {
        buffer_size: 100,
        batch_size: 10_000,        // huge — won't auto-flush by batch
        flush_interval_ms: 60_000, // huge — won't auto-flush by timer
    };
    let coll_clone = coll.clone();
    let ingester = StreamIngester::new(coll, config);

    for i in 1..=5 {
        ingester.try_send(make_point(i, 4)).expect("send ok");
    }

    // Brief yield so drain loop receives points into its channel buffer
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // Shutdown flushes everything
    ingester.shutdown().await;

    let results = coll_clone.get(&[1, 2, 3, 4, 5]);
    let found = results.iter().filter(|r| r.is_some()).count();
    assert_eq!(found, 5, "shutdown must flush all pending points");
}

// ─────────────────────────────────────────────────────────────────────────────
// BackpressureError variants
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn backpressure_error_not_configured_display() {
    let err = BackpressureError::NotConfigured;
    let msg = format!("{err}");
    assert!(msg.contains("not configured"));
}

#[test]
fn backpressure_error_drain_task_dead_display() {
    let err = BackpressureError::DrainTaskDead;
    let msg = format!("{err}");
    assert!(msg.contains("dead"));
}
