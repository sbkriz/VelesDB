#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp
)]
//! Tests for Issue #423 Component 2: Deferred `vectors.idx` serialization.
//!
//! Verifies that `flush()` can skip writing `vectors.idx` while still
//! maintaining full crash recovery via WAL replay, and that `flush_full()`
//! and `flush_index()` write the index file when explicitly requested.

use super::mmap::MmapStorage;
use super::traits::VectorStorage;
use tempfile::tempdir;

/// Helper: returns the modification time of `vectors.idx` (if it exists).
fn idx_mtime(path: &std::path::Path) -> Option<std::time::SystemTime> {
    let idx_path = path.join("vectors.idx");
    std::fs::metadata(&idx_path)
        .ok()
        .map(|m| m.modified().unwrap())
}

// -----------------------------------------------------------------------
// Issue #423: Deferred index serialization tests
// -----------------------------------------------------------------------

#[test]
fn test_flush_skips_index_write_by_default() {
    let dir = tempdir().unwrap();
    let path = dir.path().to_path_buf();

    let mut storage = MmapStorage::new(&path, 3).unwrap();
    storage.store(1, &[1.0, 2.0, 3.0]).unwrap();

    // flush() should NOT create/update vectors.idx
    storage.flush().unwrap();

    // vectors.idx should NOT exist after a fast flush
    assert!(
        !path.join("vectors.idx").exists(),
        "flush() should skip vectors.idx serialization"
    );
}

#[test]
fn test_flush_full_writes_index() {
    let dir = tempdir().unwrap();
    let path = dir.path().to_path_buf();

    let mut storage = MmapStorage::new(&path, 3).unwrap();
    storage.store(1, &[1.0, 2.0, 3.0]).unwrap();

    // flush_full() MUST write vectors.idx
    storage.flush_full().unwrap();

    assert!(
        path.join("vectors.idx").exists(),
        "flush_full() must write vectors.idx"
    );
}

#[test]
fn test_flush_index_writes_only_index() {
    let dir = tempdir().unwrap();
    let path = dir.path().to_path_buf();

    let mut storage = MmapStorage::new(&path, 3).unwrap();
    storage.store(1, &[1.0, 2.0, 3.0]).unwrap();

    // First do a regular flush (WAL + mmap)
    storage.flush().unwrap();

    // Then write just the index
    storage.flush_index().unwrap();

    assert!(
        path.join("vectors.idx").exists(),
        "flush_index() must write vectors.idx"
    );

    // Verify the index is correct by reopening
    drop(storage);
    let storage2 = MmapStorage::new(&path, 3).unwrap();
    assert_eq!(storage2.retrieve(1).unwrap(), Some(vec![1.0, 2.0, 3.0]));
    assert_eq!(storage2.len(), 1);
}

#[test]
fn test_recovery_after_flush_without_index() {
    let dir = tempdir().unwrap();
    let path = dir.path().to_path_buf();

    {
        let mut storage = MmapStorage::new(&path, 3).unwrap();
        storage.store(1, &[1.0, 2.0, 3.0]).unwrap();
        storage.store(2, &[4.0, 5.0, 6.0]).unwrap();
        storage.store(3, &[7.0, 8.0, 9.0]).unwrap();

        // Fast flush: WAL + mmap only, no index
        storage.flush().unwrap();
    } // storage dropped — best-effort shutdown sync (no index write)

    // Reopen: WAL replay must reconstruct the index from scratch
    let storage2 = MmapStorage::new(&path, 3).unwrap();
    assert_eq!(storage2.len(), 3);
    assert_eq!(storage2.retrieve(1).unwrap(), Some(vec![1.0, 2.0, 3.0]));
    assert_eq!(storage2.retrieve(2).unwrap(), Some(vec![4.0, 5.0, 6.0]));
    assert_eq!(storage2.retrieve(3).unwrap(), Some(vec![7.0, 8.0, 9.0]));
}

#[test]
fn test_recovery_with_delete_after_flush_without_index() {
    let dir = tempdir().unwrap();
    let path = dir.path().to_path_buf();

    {
        let mut storage = MmapStorage::new(&path, 3).unwrap();
        storage.store(1, &[1.0, 2.0, 3.0]).unwrap();
        storage.store(2, &[4.0, 5.0, 6.0]).unwrap();
        storage.delete(1).unwrap();

        // Fast flush: WAL + mmap only, no index
        storage.flush().unwrap();
    }

    // Reopen: WAL replay must reconstruct the correct state (id=1 deleted)
    let storage2 = MmapStorage::new(&path, 3).unwrap();
    assert_eq!(storage2.len(), 1);
    assert_eq!(storage2.retrieve(1).unwrap(), None);
    assert_eq!(storage2.retrieve(2).unwrap(), Some(vec![4.0, 5.0, 6.0]));
}

#[test]
fn test_flush_full_then_reopen_no_wal_replay_needed() {
    let dir = tempdir().unwrap();
    let path = dir.path().to_path_buf();

    {
        let mut storage = MmapStorage::new(&path, 3).unwrap();
        storage.store(1, &[1.0, 2.0, 3.0]).unwrap();

        // flush_full writes index — WAL is truncated on next open
        storage.flush_full().unwrap();
    }

    // Reopen: index file is fresh, WAL replay should be a no-op
    let storage2 = MmapStorage::new(&path, 3).unwrap();
    assert_eq!(storage2.len(), 1);
    assert_eq!(storage2.retrieve(1).unwrap(), Some(vec![1.0, 2.0, 3.0]));
}

#[test]
fn test_mixed_flush_then_flush_full() {
    let dir = tempdir().unwrap();
    let path = dir.path().to_path_buf();

    let mut storage = MmapStorage::new(&path, 3).unwrap();

    // Phase 1: fast flush (no index)
    storage.store(1, &[1.0, 2.0, 3.0]).unwrap();
    storage.flush().unwrap();
    assert!(!path.join("vectors.idx").exists());

    // Phase 2: more writes, then full flush
    storage.store(2, &[4.0, 5.0, 6.0]).unwrap();
    storage.flush_full().unwrap();
    assert!(path.join("vectors.idx").exists());

    // Reopen and verify both vectors survived
    drop(storage);
    let storage2 = MmapStorage::new(&path, 3).unwrap();
    assert_eq!(storage2.len(), 2);
    assert_eq!(storage2.retrieve(1).unwrap(), Some(vec![1.0, 2.0, 3.0]));
    assert_eq!(storage2.retrieve(2).unwrap(), Some(vec![4.0, 5.0, 6.0]));
}

#[test]
fn test_flush_index_updates_stale_idx() {
    let dir = tempdir().unwrap();
    let path = dir.path().to_path_buf();

    let mut storage = MmapStorage::new(&path, 3).unwrap();
    storage.store(1, &[1.0, 2.0, 3.0]).unwrap();
    storage.flush_full().unwrap();

    let mtime_before = idx_mtime(&path);

    // Small delay to ensure filesystem timestamp changes
    std::thread::sleep(std::time::Duration::from_millis(50));

    // More writes + fast flush (index stale now)
    storage.store(2, &[4.0, 5.0, 6.0]).unwrap();
    storage.flush().unwrap();

    // Index should NOT have been updated by the fast flush
    let mtime_after_fast = idx_mtime(&path);
    assert_eq!(
        mtime_before, mtime_after_fast,
        "fast flush must not update vectors.idx"
    );

    // Now write index explicitly
    storage.flush_index().unwrap();
    let mtime_after_index = idx_mtime(&path);
    assert_ne!(
        mtime_before, mtime_after_index,
        "flush_index() must update vectors.idx"
    );
}
