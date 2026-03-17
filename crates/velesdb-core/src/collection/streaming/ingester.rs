//! StreamIngester: bounded-channel ingestion with micro-batch drain.

use crate::collection::types::Collection;
use crate::point::Point;

use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::sync::Notify;

/// Configuration for the streaming ingestion pipeline.
///
/// Controls channel capacity, micro-batch sizing, and flush timing.
///
/// # Defaults
///
/// | Parameter          | Default  |
/// |--------------------|----------|
/// | `buffer_size`      | 10 000   |
/// | `batch_size`       | 128      |
/// | `flush_interval_ms`| 50       |
///
/// Future: persist StreamingConfig in CollectionConfig (STREAM-04)
///
/// `StreamingConfig` is currently runtime-only. A future pass should
/// serialize it into `CollectionConfig` so the pipeline is automatically
/// restored on `Collection::open`.
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Capacity of the bounded mpsc channel (backpressure threshold).
    pub buffer_size: usize,

    /// Number of points that trigger an immediate micro-batch flush.
    pub batch_size: usize,

    /// Maximum time (ms) before a partial batch is flushed.
    pub flush_interval_ms: u64,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            buffer_size: 10_000,
            batch_size: 128,
            flush_interval_ms: 50,
        }
    }
}

/// Internal write mode discriminator (not exposed to users).
///
/// Distinguishes between API-driven writes (synchronous upsert) and
/// streaming-driven writes (micro-batch drain).
///
/// Future: integrate WriteMode into StreamingConfig (STREAM-03)
///
/// `WriteMode` is currently unused. Once streaming-specific write paths
/// (e.g., bypass WAL for low-latency inserts) are implemented, wire this
/// into the flush pipeline.
#[allow(dead_code)] // Tracked: STREAM-03
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WriteMode {
    /// Standard synchronous API upsert.
    Api,
    /// Streaming micro-batch drain.
    Streaming,
}

/// Error returned when the streaming channel cannot accept a point.
#[derive(Debug, thiserror::Error)]
pub enum BackpressureError {
    /// The ingestion buffer is full; the caller should retry after a short delay.
    #[error("streaming buffer is full (backpressure)")]
    BufferFull,

    /// Streaming is not configured on this collection.
    #[error("streaming is not configured on this collection")]
    NotConfigured,

    /// The drain task has exited; the streaming pipeline is no longer functional.
    ///
    /// This is a fatal condition — the server should respond 503 Service Unavailable
    /// and the collection may need to be reconfigured.
    #[error("streaming drain task has exited; the ingestion pipeline is dead")]
    DrainTaskDead,
}

/// Streaming ingestion handle for a single collection.
///
/// Owns a bounded mpsc sender and a background drain task. Points sent via
/// [`try_send`](Self::try_send) are accumulated into micro-batches and flushed
/// to the collection's existing `upsert` pipeline.
///
/// # Shutdown
///
/// Call [`shutdown`](Self::shutdown) to gracefully drain remaining points.
/// If dropped without shutdown, the drain task is aborted (points in the
/// channel may be lost).
pub struct StreamIngester {
    sender: mpsc::Sender<Point>,
    config: StreamingConfig,
    drain_handle: Option<tokio::task::JoinHandle<()>>,
    shutdown: Arc<Notify>,
}

impl StreamIngester {
    /// Creates a new streaming ingester for the given collection.
    ///
    /// Spawns a background drain task that accumulates points and flushes
    /// micro-batches via `Collection::upsert`.
    #[must_use]
    pub fn new(collection: Collection, config: StreamingConfig) -> Self {
        let (tx, rx) = mpsc::channel(config.buffer_size);
        let shutdown = Arc::new(Notify::new());

        let drain_handle = tokio::spawn(drain_loop(
            collection,
            rx,
            config.batch_size,
            config.flush_interval_ms,
            Arc::clone(&shutdown),
        ));

        Self {
            sender: tx,
            config,
            drain_handle: Some(drain_handle),
            shutdown,
        }
    }

    /// Attempts to send a point into the streaming channel.
    ///
    /// Returns immediately. If the channel is at capacity, returns
    /// [`BackpressureError::BufferFull`]. If the drain task has exited
    /// (channel closed), returns [`BackpressureError::DrainTaskDead`].
    ///
    /// # Errors
    ///
    /// - [`BackpressureError::BufferFull`] — the bounded channel is at capacity.
    /// - [`BackpressureError::DrainTaskDead`] — the drain task exited unexpectedly.
    pub fn try_send(&self, point: Point) -> Result<(), BackpressureError> {
        self.sender.try_send(point).map_err(|e| match e {
            mpsc::error::TrySendError::Full(_) => BackpressureError::BufferFull,
            mpsc::error::TrySendError::Closed(_) => BackpressureError::DrainTaskDead,
        })
    }

    /// Returns a reference to the streaming configuration.
    #[must_use]
    pub fn config(&self) -> &StreamingConfig {
        &self.config
    }

    /// Gracefully shuts down the ingester, flushing any remaining buffered points.
    ///
    /// This notifies the drain loop to exit and awaits its completion.
    pub async fn shutdown(mut self) {
        self.shutdown.notify_one();
        if let Some(handle) = self.drain_handle.take() {
            // Ignore JoinError — the drain loop should not panic.
            let _ = handle.await;
        }
    }
}

impl Drop for StreamIngester {
    fn drop(&mut self) {
        // Abort the drain task to prevent orphaned background tasks.
        // For graceful shutdown with flush, call `shutdown()` before dropping.
        if let Some(handle) = self.drain_handle.take() {
            handle.abort();
        }
    }
}

/// Background drain loop that accumulates points and flushes micro-batches.
///
/// Uses `tokio::select!` with three branches:
/// 1. Shutdown notification — flush remaining batch and exit.
/// 2. Timer tick — flush partial batch if non-empty.
/// 3. Channel receive — push to batch; flush when `batch_size` reached.
async fn drain_loop(
    collection: Collection,
    mut rx: mpsc::Receiver<Point>,
    batch_size: usize,
    flush_interval_ms: u64,
    shutdown: Arc<Notify>,
) {
    let mut batch: Vec<Point> = Vec::with_capacity(batch_size);
    let mut interval = tokio::time::interval(std::time::Duration::from_millis(flush_interval_ms));
    // The first tick completes immediately; consume it.
    interval.tick().await;

    loop {
        tokio::select! {
            // Branch 1: shutdown signal — drain remaining channel items in
            // micro-batches (M-1: flush at batch_size to bound memory usage).
            () = shutdown.notified() => {
                while let Ok(point) = rx.try_recv() {
                    batch.push(point);
                    // Flush at batch_size boundaries to avoid unbounded accumulation.
                    if batch.len() >= batch_size {
                        flush_batch(&collection, &mut batch).await;
                    }
                }
                if !batch.is_empty() {
                    flush_batch(&collection, &mut batch).await;
                }
                break;
            }

            // Branch 2: timer tick — flush partial batch
            _ = interval.tick() => {
                if !batch.is_empty() {
                    flush_batch(&collection, &mut batch).await;
                }
            }

            // Branch 3: receive point from channel
            msg = rx.recv() => {
                if let Some(point) = msg {
                    batch.push(point);
                    if batch.len() >= batch_size {
                        flush_batch(&collection, &mut batch).await;
                        // Reset the interval so the timer doesn't fire
                        // immediately after a batch-size flush.
                        interval.reset();
                    }
                } else {
                    // Channel closed (all senders dropped).
                    if !batch.is_empty() {
                        flush_batch(&collection, &mut batch).await;
                    }
                    break;
                }
            }
        }
    }
}

/// Flushes the accumulated batch via the collection's existing upsert pipeline.
///
/// Runs the blocking upsert on Tokio's blocking thread pool to avoid stalling
/// the async runtime. If the delta buffer is active (HNSW rebuild in progress),
/// also pushes the batch vectors into the delta buffer for immediate searchability.
async fn flush_batch(collection: &Collection, batch: &mut Vec<Point>) {
    let points: Vec<Point> = std::mem::take(batch);

    // Snapshot vectors for delta buffer before moving points into upsert.
    // Only allocate if delta is active (common case: delta is inactive).
    let delta_entries: Vec<(u64, Vec<f32>)> = if collection.delta_buffer.is_active() {
        points.iter().map(|p| (p.id, p.vector.clone())).collect()
    } else {
        Vec::new()
    };

    let coll = collection.clone();
    // spawn_blocking wraps the synchronous upsert call (which acquires
    // multiple RwLocks and does mmap I/O) to prevent blocking the async runtime.
    let result = tokio::task::spawn_blocking(move || coll.upsert(points)).await;
    match result {
        Ok(Ok(())) => {
            // After successful upsert, push to delta buffer if active.
            // The upsert wrote to storage+WAL; delta is an additional runtime
            // copy so search can find these vectors before HNSW is rebuilt.
            if !delta_entries.is_empty() {
                collection.delta_buffer.extend(delta_entries);
            }
        }
        Ok(Err(e)) => {
            tracing::error!("Streaming drain flush failed: {e}");
        }
        Err(e) => {
            tracing::error!("Streaming drain task panicked: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::DistanceMetric;
    use std::time::Duration;
    use tempfile::TempDir;

    /// Helper: create a test collection in a temp directory.
    fn test_collection(dim: usize) -> (TempDir, Collection) {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("test_stream_coll");
        let coll =
            Collection::create(path, dim, DistanceMetric::Cosine).expect("create collection");
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

    #[tokio::test]
    async fn test_stream_try_send_succeeds_when_capacity_available() {
        let (_dir, coll) = test_collection(4);
        let config = StreamingConfig {
            buffer_size: 10,
            batch_size: 100,
            flush_interval_ms: 5000,
        };
        let ingester = StreamIngester::new(coll, config);

        let result = ingester.try_send(make_point(1, 4));
        assert!(
            result.is_ok(),
            "try_send should succeed when channel has capacity"
        );

        ingester.shutdown().await;
    }

    #[tokio::test]
    async fn test_stream_try_send_returns_buffer_full_when_at_capacity() {
        let (_dir, coll) = test_collection(4);
        let config = StreamingConfig {
            buffer_size: 2,
            batch_size: 100, // Large batch_size so drain doesn't flush quickly
            flush_interval_ms: 60_000, // Very long interval to avoid timer flush
        };
        let ingester = StreamIngester::new(coll, config);

        // Fill the channel (capacity = 2)
        assert!(ingester.try_send(make_point(1, 4)).is_ok());
        assert!(ingester.try_send(make_point(2, 4)).is_ok());

        // Third send should fail with BufferFull
        let result = ingester.try_send(make_point(3, 4));
        assert!(result.is_err(), "should return error when buffer full");
        match result.unwrap_err() {
            BackpressureError::BufferFull => {} // expected
            other => panic!("expected BufferFull, got: {other}"),
        }

        ingester.shutdown().await;
    }

    #[tokio::test]
    async fn test_stream_drain_flushes_at_batch_size() {
        let (_dir, coll) = test_collection(4);
        let batch_size = 4;
        let config = StreamingConfig {
            buffer_size: 100,
            batch_size,
            flush_interval_ms: 60_000, // Long interval — batch_size should trigger flush
        };
        let coll_clone = coll.clone();
        let ingester = StreamIngester::new(coll, config);

        // Send exactly batch_size points
        for i in 0..batch_size {
            ingester
                .try_send(make_point(i as u64 + 1, 4))
                .expect("send should succeed");
        }

        // Give the drain loop time to process
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Verify points were upserted
        let results = coll_clone.get(&[1, 2, 3, 4]);
        let found_count = results.iter().filter(|r| r.is_some()).count();
        assert_eq!(
            found_count, 4,
            "all {batch_size} points should be flushed via upsert"
        );

        ingester.shutdown().await;
    }

    #[tokio::test]
    async fn test_stream_drain_flushes_partial_batch_after_timeout() {
        let (_dir, coll) = test_collection(4);
        let config = StreamingConfig {
            buffer_size: 100,
            batch_size: 100, // Very large batch — timer should trigger flush
            flush_interval_ms: 50,
        };
        let coll_clone = coll.clone();
        let ingester = StreamIngester::new(coll, config);

        // Send only 2 points (well below batch_size of 100)
        ingester.try_send(make_point(1, 4)).expect("send 1");
        ingester.try_send(make_point(2, 4)).expect("send 2");

        // Wait longer than flush_interval_ms
        tokio::time::sleep(Duration::from_millis(300)).await;

        // Verify points were flushed by the timer
        let results = coll_clone.get(&[1, 2]);
        let found_count = results.iter().filter(|r| r.is_some()).count();
        assert_eq!(
            found_count, 2,
            "partial batch should be flushed after flush_interval_ms"
        );

        ingester.shutdown().await;
    }

    #[tokio::test]
    async fn test_stream_shutdown_flushes_remaining_points() {
        let (_dir, coll) = test_collection(4);
        let config = StreamingConfig {
            buffer_size: 100,
            batch_size: 1000, // Very large batch — won't trigger batch flush
            flush_interval_ms: 60_000, // Very long — won't trigger timer flush
        };
        let coll_clone = coll.clone();
        let ingester = StreamIngester::new(coll, config);

        // Send a few points
        ingester.try_send(make_point(10, 4)).expect("send");
        ingester.try_send(make_point(11, 4)).expect("send");

        // Sleep long enough to ensure the drain loop has received the points
        // from the channel before shutdown is signalled (N-5: yield_now()
        // is not sufficient — the drain loop may not have been scheduled yet).
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Shutdown should flush remaining
        ingester.shutdown().await;

        // Verify points were flushed during shutdown
        let results = coll_clone.get(&[10, 11]);
        let found_count = results.iter().filter(|r| r.is_some()).count();
        assert_eq!(
            found_count, 2,
            "shutdown should flush remaining buffered points"
        );
    }

    #[tokio::test]
    async fn test_stream_delta_drain_loop_routes_to_delta_when_active() {
        let (_dir, coll) = test_collection(4);
        let config = StreamingConfig {
            buffer_size: 100,
            batch_size: 4,
            flush_interval_ms: 50,
        };
        let coll_clone = coll.clone();

        // Activate delta buffer (simulating HNSW rebuild)
        coll.delta_buffer.activate();

        let ingester = StreamIngester::new(coll, config);

        // Send points via streaming
        for i in 1..=4 {
            ingester
                .try_send(make_point(i, 4))
                .expect("send should succeed");
        }

        // Wait for drain loop to flush
        tokio::time::sleep(Duration::from_millis(300)).await;

        // Verify points are in storage (upsert always writes)
        let results = coll_clone.get(&[1, 2, 3, 4]);
        let found = results.iter().filter(|r| r.is_some()).count();
        assert_eq!(found, 4, "upsert should write all points to storage");

        // Verify points are also in the delta buffer
        assert_eq!(
            coll_clone.delta_buffer.len(),
            4,
            "delta buffer should contain the streamed points when active"
        );

        ingester.shutdown().await;
    }

    #[tokio::test]
    #[allow(clippy::cast_possible_truncation)]
    async fn test_stream_searchable_immediately() {
        let (_dir, coll) = test_collection(4);
        let config = StreamingConfig {
            buffer_size: 100,
            batch_size: 4,
            flush_interval_ms: 50,
        };
        let coll_clone = coll.clone();
        let ingester = StreamIngester::new(coll, config);

        // Insert points with distinct vectors for search
        for i in 1..=4u64 {
            let mut vec = vec![0.0_f32; 4];
            vec[(i as usize - 1) % 4] = 1.0;
            let p = Point {
                id: i,
                vector: vec,
                payload: None,
                sparse_vectors: None,
            };
            ingester.try_send(p).expect("send should succeed");
        }

        // Wait for drain to flush
        tokio::time::sleep(Duration::from_millis(300)).await;

        // Search for the first vector — should find it via HNSW (indexed by upsert)
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = coll_clone.search(&query, 4).expect("search should succeed");
        assert!(
            !results.is_empty(),
            "inserted points should be searchable after drain"
        );
        // The closest result to [1,0,0,0] should be id=1
        assert_eq!(results[0].point.id, 1, "closest match should be id=1");

        ingester.shutdown().await;
    }

    #[tokio::test]
    #[allow(clippy::cast_precision_loss)]
    async fn test_stream_delta_rebuild_no_data_loss() {
        let (_dir, coll) = test_collection(4);
        // Pre-insert some points via direct upsert (these go into HNSW)
        let initial_points: Vec<Point> = (1..=5u64)
            .map(|i| {
                let mut vec = vec![0.0_f32; 4];
                vec[0] = i as f32;
                Point {
                    id: i,
                    vector: vec,
                    payload: None,
                    sparse_vectors: None,
                }
            })
            .collect();
        coll.upsert(initial_points).expect("upsert initial points");

        // Simulate rebuild: activate delta buffer
        coll.delta_buffer.activate();
        assert!(coll.delta_buffer.is_active());

        let config = StreamingConfig {
            buffer_size: 100,
            batch_size: 4,
            flush_interval_ms: 50,
        };
        let coll_clone = coll.clone();
        let ingester = StreamIngester::new(coll, config);

        // Insert new points during "rebuild"
        for i in 6..=10u64 {
            let mut vec = vec![0.0_f32; 4];
            vec[0] = i as f32;
            let p = Point {
                id: i,
                vector: vec,
                payload: None,
                sparse_vectors: None,
            };
            ingester.try_send(p).expect("send should succeed");
        }

        // Wait for drain to flush
        tokio::time::sleep(Duration::from_millis(300)).await;

        // Search — should find both HNSW (old) and delta (new) points
        let query = vec![10.0, 0.0, 0.0, 0.0];
        let results = coll_clone
            .search_ids(&query, 10)
            .expect("search_ids should succeed");
        let found_ids: std::collections::HashSet<u64> = results.iter().map(|sr| sr.id).collect();

        // All 10 points should be found (5 HNSW + 5 delta)
        for id in 1..=10 {
            assert!(
                found_ids.contains(&id),
                "point id={id} should be in search results"
            );
        }

        // Deactivate and drain
        let drained = coll_clone.delta_buffer.deactivate_and_drain();
        assert!(!coll_clone.delta_buffer.is_active());
        assert_eq!(drained.len(), 5, "delta should have had 5 entries");

        ingester.shutdown().await;
    }
}
