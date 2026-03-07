//! Delta buffer for accumulating vectors during HNSW rebuilds.
//!
//! The [`DeltaBuffer`] holds recently inserted vectors that have not yet been
//! indexed into the HNSW graph (e.g., because a rebuild is in progress).
//! In Plan 02, the search pipeline will brute-force scan this buffer and
//! merge results with HNSW results for immediate searchability.
//!
//! # Lock ordering
//!
//! `DeltaBuffer` is at position **10** in the collection lock order
//! (after `sparse_indexes` at 9). Code must never hold a delta buffer lock
//! while acquiring a lower-numbered lock.

use parking_lot::RwLock;
use std::sync::atomic::{AtomicBool, Ordering};

/// Stub delta buffer for streaming inserts during HNSW rebuilds.
///
/// Populated in Plan 02 with actual insert/search methods.
/// For now, provides construction and an `is_active` check.
pub struct DeltaBuffer {
    /// Buffered `(point_id, vector)` pairs awaiting index insertion.
    #[allow(dead_code)]
    points: RwLock<Vec<(u64, Vec<f32>)>>,

    /// Whether the delta buffer is actively accumulating (rebuild in progress).
    active: AtomicBool,
}

impl DeltaBuffer {
    /// Creates an empty, inactive delta buffer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            points: RwLock::new(Vec::new()),
            active: AtomicBool::new(false),
        }
    }

    /// Returns `true` if the delta buffer is actively accumulating vectors
    /// (i.e., an HNSW rebuild is in progress).
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }
}

impl Default for DeltaBuffer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_delta_buffer_compiles_and_defaults_inactive() {
        let buf = DeltaBuffer::new();
        assert!(
            !buf.is_active(),
            "new DeltaBuffer should be inactive by default"
        );
    }

    #[test]
    fn test_stream_delta_buffer_default_trait() {
        let buf = DeltaBuffer::default();
        assert!(!buf.is_active());
    }
}
