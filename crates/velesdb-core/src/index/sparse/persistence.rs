//! Sparse index persistence: WAL, compaction, and mmap-based loading.
//!
//! All types and functions in this module are gated behind `#[cfg(feature = "persistence")]`.
//!
//! ## On-disk layout
//!
//! ```text
//! <collection_dir>/
//!   sparse.wal        # Write-ahead log (length-prefixed entries)
//!   sparse.idx        # Compacted posting lists (raw PostingEntry bytes)
//!   sparse.terms      # Term dictionary (postcard-serialized Vec<TermEntry>)
//!   sparse.meta       # Metadata (postcard-serialized SparseMeta)
//! ```

use std::io::{BufWriter, Write};
use std::path::Path;

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use super::inverted_index::{FrozenSegment, SparseInvertedIndex};
use super::types::{PostingEntry, SparseVector};
use crate::error::{Error, Result};

// ---------------------------------------------------------------------------
// WAL constants
// ---------------------------------------------------------------------------

const WAL_OP_UPSERT: u8 = 0x01;
const WAL_OP_DELETE: u8 = 0x02;

/// Number of replayed WAL entries that triggers automatic compaction on load.
const COMPACTION_REPLAY_THRESHOLD: u64 = 10_000;

// ---------------------------------------------------------------------------
// On-disk structures
// ---------------------------------------------------------------------------

/// Metadata header for the compacted sparse index.
#[derive(Debug, Serialize, Deserialize)]
struct SparseMeta {
    version: u32,
    doc_count: u64,
    term_count: u32,
}

/// Term dictionary entry mapping `term_id` to its posting range in `sparse.idx`.
#[derive(Debug, Serialize, Deserialize)]
struct TermEntry {
    term_id: u32,
    offset: u64,
    len: u32,
    max_weight: f32,
}

// ---------------------------------------------------------------------------
// WAL operations
// ---------------------------------------------------------------------------

/// Size of a single `PostingEntry` on disk (`u64` `doc_id` + `f32` weight, no padding).
///
/// Note: `size_of::<PostingEntry>()` is 16 due to alignment padding on `#[repr(C)]`,
/// but on disk we write the fields individually without padding (packed layout).
const POSTING_DISK_SIZE: usize = 12; // 8 + 4, packed

// Compile-time guard: POSTING_DISK_SIZE must equal the sum of the two constituent field sizes.
// If u64 or f32 ever change width (e.g., on exotic targets), this fires at compile time.
const _: () = assert!(
    std::mem::size_of::<u64>() + std::mem::size_of::<f32>() == POSTING_DISK_SIZE,
    "POSTING_DISK_SIZE must match u64 + f32 packed size"
);

// ---------------------------------------------------------------------------
// Byte-parsing helpers
// ---------------------------------------------------------------------------

/// Reads a little-endian `u64` from `data[pos..pos+8]`.
///
/// The caller is responsible for bounds-checking before calling. The `try_into()` can only
/// fail if the slice is not exactly 8 bytes — which the upstream bounds checks prevent.
/// We propagate rather than panic to fail-fast if an upstream refactor breaks the invariant.
#[inline]
fn read_le_u64(data: &[u8], pos: usize, context: &str) -> Result<u64> {
    data[pos..pos + 8]
        .try_into()
        .map(u64::from_le_bytes)
        .map_err(|_| Error::SparseIndexError(format!("{context} at offset {pos}")))
}

/// Reads a little-endian `u32` from `data[pos..pos+4]`.
///
/// See [`read_le_u64`] for the invariant reasoning.
#[inline]
fn read_le_u32(data: &[u8], pos: usize, context: &str) -> Result<u32> {
    data[pos..pos + 4]
        .try_into()
        .map(u32::from_le_bytes)
        .map_err(|_| Error::SparseIndexError(format!("{context} at offset {pos}")))
}

/// Reads a little-endian `f32` from `data[pos..pos+4]`.
///
/// See [`read_le_u64`] for the invariant reasoning.
#[inline]
fn read_le_f32(data: &[u8], pos: usize, context: &str) -> Result<f32> {
    data[pos..pos + 4]
        .try_into()
        .map(f32::from_le_bytes)
        .map_err(|_| Error::SparseIndexError(format!("{context} at offset {pos}")))
}

/// Appends an upsert entry to the sparse WAL.
///
/// # Errors
///
/// Returns an error if the WAL file cannot be opened or written.
pub fn wal_append_upsert(wal_path: &Path, point_id: u64, vector: &SparseVector) -> Result<()> {
    #[allow(clippy::cast_possible_truncation)] // nnz bounded by sparse vector dimension count
    let nnz = vector.nnz() as u32;
    // total_len = op(1) + point_id(8) + nnz(4) + pairs(nnz * 8).
    // Use checked arithmetic to guard against pathologically large sparse vectors.
    let total_len: u32 = nnz
        .checked_mul(8)
        .and_then(|pairs_len| {
            1u32.checked_add(8)
                .and_then(|h| h.checked_add(4))
                .and_then(|h| h.checked_add(pairs_len))
        })
        .ok_or_else(|| {
            Error::SparseIndexError(format!(
                "WAL entry too large: nnz={nnz} would overflow u32 length prefix"
            ))
        })?;

    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(wal_path)
        .map_err(|e| Error::SparseIndexError(format!("WAL open failed: {e}")))?;
    let mut w = BufWriter::new(file);

    w.write_all(&total_len.to_le_bytes())
        .map_err(|e| Error::SparseIndexError(format!("WAL write failed: {e}")))?;
    w.write_all(&[WAL_OP_UPSERT])
        .map_err(|e| Error::SparseIndexError(format!("WAL write failed: {e}")))?;
    w.write_all(&point_id.to_le_bytes())
        .map_err(|e| Error::SparseIndexError(format!("WAL write failed: {e}")))?;
    w.write_all(&nnz.to_le_bytes())
        .map_err(|e| Error::SparseIndexError(format!("WAL write failed: {e}")))?;

    for (&idx, &val) in vector.indices.iter().zip(vector.values.iter()) {
        w.write_all(&idx.to_le_bytes())
            .map_err(|e| Error::SparseIndexError(format!("WAL write failed: {e}")))?;
        w.write_all(&val.to_le_bytes())
            .map_err(|e| Error::SparseIndexError(format!("WAL write failed: {e}")))?;
    }

    w.flush()
        .map_err(|e| Error::SparseIndexError(format!("WAL flush failed: {e}")))?;
    Ok(())
}

/// Appends a delete entry to the sparse WAL.
///
/// # Errors
///
/// Returns an error if the WAL file cannot be opened or written.
pub fn wal_append_delete(wal_path: &Path, point_id: u64) -> Result<()> {
    // total_len = op(1) + point_id(8) = 9
    let total_len: u32 = 1 + 8;

    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(wal_path)
        .map_err(|e| Error::SparseIndexError(format!("WAL open failed: {e}")))?;
    let mut w = BufWriter::new(file);

    w.write_all(&total_len.to_le_bytes())
        .map_err(|e| Error::SparseIndexError(format!("WAL write failed: {e}")))?;
    w.write_all(&[WAL_OP_DELETE])
        .map_err(|e| Error::SparseIndexError(format!("WAL write failed: {e}")))?;
    w.write_all(&point_id.to_le_bytes())
        .map_err(|e| Error::SparseIndexError(format!("WAL write failed: {e}")))?;

    w.flush()
        .map_err(|e| Error::SparseIndexError(format!("WAL flush failed: {e}")))?;
    Ok(())
}

/// Replays a sparse WAL into the given index. Returns the number of entries replayed.
///
/// On truncated entry (remaining bytes < declared `total_len`), logs a warning and stops.
/// On missing WAL file, returns `Ok(0)`.
///
/// # Errors
///
/// Returns an error if the WAL file cannot be read (other than not-found), or if
/// byte sequences that should be exactly 4 or 8 bytes long are corrupt.
pub fn wal_replay(wal_path: &Path, index: &SparseInvertedIndex) -> Result<u64> {
    let data = match std::fs::read(wal_path) {
        Ok(d) => d,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(e) => return Err(Error::SparseIndexError(format!("WAL read failed: {e}"))),
    };

    let mut pos = 0usize;
    let mut count = 0u64;

    while pos < data.len() {
        if pos + 4 > data.len() {
            tracing::warn!(
                "Sparse WAL truncated at offset {pos}: not enough bytes for length prefix"
            );
            break;
        }
        let total_len =
            read_le_u32(&data, pos, "WAL entry corrupted: bad length-prefix bytes")? as usize;
        pos += 4;

        if pos + total_len > data.len() {
            tracing::warn!(
                "Sparse WAL truncated at offset {}: declared {total_len} bytes but only {} remain",
                pos - 4,
                data.len() - pos
            );
            break;
        }

        let entry_start = pos;
        let op = data[pos];
        pos += 1;

        match op {
            WAL_OP_UPSERT => {
                if total_len < 1 + 8 + 4 {
                    tracing::warn!("Sparse WAL upsert entry too short at offset {entry_start}");
                    break;
                }
                let point_id = read_le_u64(&data, pos, "WAL entry corrupted: bad point_id bytes")?;
                pos += 8;
                let nnz = read_le_u32(&data, pos, "WAL entry corrupted: bad nnz bytes")? as usize;
                pos += 4;

                let expected_pairs_len = nnz * 8;
                if entry_start + total_len < pos + expected_pairs_len {
                    tracing::warn!("Sparse WAL upsert entry truncated at offset {entry_start}");
                    break;
                }

                let mut pairs = Vec::with_capacity(nnz);
                for _ in 0..nnz {
                    let idx = read_le_u32(&data, pos, "WAL entry corrupted: bad term-index bytes")?;
                    pos += 4;
                    let val = read_le_f32(&data, pos, "WAL entry corrupted: bad weight bytes")?;
                    pos += 4;
                    pairs.push((idx, val));
                }

                let vector = SparseVector::new(pairs);
                index.insert(point_id, &vector);
                count += 1;
            }
            WAL_OP_DELETE => {
                let point_id = read_le_u64(&data, pos, "WAL entry corrupted: bad point_id bytes")?;
                pos += 8;
                index.delete(point_id);
                count += 1;
            }
            unknown => {
                tracing::warn!("Sparse WAL unknown op 0x{unknown:02x} at offset {entry_start}");
                pos = entry_start + total_len;
            }
        }

        // Ensure pos advances to end of entry in case of internal padding
        let expected_end = entry_start + total_len;
        if pos < expected_end {
            pos = expected_end;
        }
    }

    Ok(count)
}

// ---------------------------------------------------------------------------
// Named sparse index helpers
// ---------------------------------------------------------------------------

/// Returns the file prefix for a named sparse index.
///
/// - Empty name `""` -> `"sparse"` (backward compat with unprefixed files)
/// - Named `"title"` -> `"sparse-title"`
fn sparse_file_prefix(name: &str) -> String {
    if name.is_empty() {
        "sparse".to_string()
    } else {
        format!("sparse-{name}")
    }
}

/// Compacts a named sparse index to disk using name-prefixed files.
///
/// Default name `""` uses unprefixed `sparse.*` files for backward compat.
///
/// # Errors
///
/// Returns an error if disk writes fail.
pub fn compact_named(dir: &Path, name: &str, index: &SparseInvertedIndex) -> Result<()> {
    let prefix = sparse_file_prefix(name);
    compact_with_prefix(dir, &prefix, index)
}

/// Loads a named sparse index from disk using name-prefixed files.
///
/// # Errors
///
/// Returns an error if files exist but are corrupt.
pub fn load_named_from_disk(dir: &Path, name: &str) -> Result<Option<SparseInvertedIndex>> {
    let prefix = sparse_file_prefix(name);
    load_from_disk_with_prefix(dir, &prefix)
}

/// Returns the WAL path for a named sparse index.
#[must_use]
pub fn wal_path_for_name(dir: &Path, name: &str) -> std::path::PathBuf {
    let prefix = sparse_file_prefix(name);
    dir.join(format!("{prefix}.wal"))
}

// ---------------------------------------------------------------------------
// Compaction
// ---------------------------------------------------------------------------

/// Compacts the in-memory index to disk using the default (unprefixed) file names.
///
/// Delegates to `compact_with_prefix` with prefix `"sparse"`.
///
/// # Errors
///
/// Returns an error if disk writes fail or if an internal index invariant is violated.
pub fn compact(dir: &Path, index: &SparseInvertedIndex) -> Result<()> {
    compact_with_prefix(dir, "sparse", index)
}

/// Compacts the in-memory index to disk using the given file prefix.
///
/// Files written: `{prefix}.idx`, `{prefix}.terms`, `{prefix}.meta`.
/// Truncates `{prefix}.wal` after successful compaction.
///
/// # Errors
///
/// Returns an error if disk writes fail or if the index's internal posting map is
/// inconsistent (a term ID present in the sorted key list is absent from the map).
#[allow(clippy::too_many_lines)]
fn compact_with_prefix(dir: &Path, prefix: &str, index: &SparseInvertedIndex) -> Result<()> {
    let merged = index.get_merged_postings_for_compaction();

    // Sort terms for deterministic output
    let mut term_ids: Vec<u32> = merged.keys().copied().collect();
    term_ids.sort_unstable();

    // --- Write {prefix}.idx.tmp ---
    let idx_tmp = dir.join(format!("{prefix}.idx.tmp"));
    let mut idx_file = BufWriter::new(
        std::fs::File::create(&idx_tmp)
            .map_err(|e| Error::SparseIndexError(format!("compact idx create: {e}")))?,
    );

    let mut term_entries: Vec<TermEntry> = Vec::with_capacity(term_ids.len());
    let mut current_offset: u64 = 0;

    for &term_id in &term_ids {
        let (postings, max_weight) = merged.get(&term_id).ok_or_else(|| {
            Error::SparseIndexError(format!(
                "compact: term_id {term_id} present in sorted key list \
                 but absent from merged postings map — index state is inconsistent"
            ))
        })?;

        // Write each PostingEntry as packed bytes: doc_id(u64 LE) + weight(f32 LE)
        for entry in postings {
            idx_file
                .write_all(&entry.doc_id.to_le_bytes())
                .map_err(|e| Error::SparseIndexError(format!("compact idx write: {e}")))?;
            idx_file
                .write_all(&entry.weight.to_le_bytes())
                .map_err(|e| Error::SparseIndexError(format!("compact idx write: {e}")))?;
        }

        let byte_len = (postings.len() * POSTING_DISK_SIZE) as u64;
        term_entries.push(TermEntry {
            term_id,
            offset: current_offset,
            #[allow(clippy::cast_possible_truncation)] // posting count bounded by doc count
            len: postings.len() as u32,
            max_weight: *max_weight,
        });
        current_offset += byte_len;
    }
    idx_file
        .flush()
        .map_err(|e| Error::SparseIndexError(format!("compact idx flush: {e}")))?;
    drop(idx_file);

    // --- Write {prefix}.terms.tmp ---
    let terms_tmp = dir.join(format!("{prefix}.terms.tmp"));
    let terms_data = postcard::to_allocvec(&term_entries)
        .map_err(|e| Error::SparseIndexError(format!("compact terms serialize: {e}")))?;
    std::fs::write(&terms_tmp, &terms_data)
        .map_err(|e| Error::SparseIndexError(format!("compact terms write: {e}")))?;

    // --- Write {prefix}.meta.tmp ---
    let meta_tmp = dir.join(format!("{prefix}.meta.tmp"));
    let meta = SparseMeta {
        version: 1,
        doc_count: index.doc_count(),
        #[allow(clippy::cast_possible_truncation)] // term count bounded by vocabulary size
        term_count: term_ids.len() as u32,
    };
    let meta_data = postcard::to_allocvec(&meta)
        .map_err(|e| Error::SparseIndexError(format!("compact meta serialize: {e}")))?;
    std::fs::write(&meta_tmp, &meta_data)
        .map_err(|e| Error::SparseIndexError(format!("compact meta write: {e}")))?;

    // --- Atomic rename ---
    std::fs::rename(&idx_tmp, dir.join(format!("{prefix}.idx")))
        .map_err(|e| Error::SparseIndexError(format!("compact idx rename: {e}")))?;
    std::fs::rename(&terms_tmp, dir.join(format!("{prefix}.terms")))
        .map_err(|e| Error::SparseIndexError(format!("compact terms rename: {e}")))?;
    std::fs::rename(&meta_tmp, dir.join(format!("{prefix}.meta")))
        .map_err(|e| Error::SparseIndexError(format!("compact meta rename: {e}")))?;

    // --- Truncate WAL ---
    let wal_path = dir.join(format!("{prefix}.wal"));
    if wal_path.exists() {
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(&wal_path)
            .map_err(|e| Error::SparseIndexError(format!("compact wal truncate: {e}")))?;
        file.set_len(0)
            .map_err(|e| Error::SparseIndexError(format!("compact wal truncate: {e}")))?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Loading from disk
// ---------------------------------------------------------------------------

/// Loads a sparse index from disk using default (unprefixed) file names.
///
/// Delegates to `load_from_disk_with_prefix` with prefix `"sparse"`.
///
/// # Errors
///
/// Returns an error if files exist but are corrupt.
pub fn load_from_disk(dir: &Path) -> Result<Option<SparseInvertedIndex>> {
    load_from_disk_with_prefix(dir, "sparse")
}

/// Loads a sparse index from disk using the given file prefix.
///
/// Returns `Ok(None)` if no `{prefix}.meta` file is found (empty collection).
/// If `{prefix}.wal` exists, replays it after loading compacted data.
/// If replayed entries exceed the compaction threshold, triggers automatic compaction.
///
/// # Errors
///
/// Returns an error if files exist but cannot be read, deserialized, or contain
/// corrupt byte sequences that cannot be converted to the expected fixed-size arrays.
fn load_from_disk_with_prefix(dir: &Path, prefix: &str) -> Result<Option<SparseInvertedIndex>> {
    let meta_path = dir.join(format!("{prefix}.meta"));
    if !meta_path.exists() {
        // No sparse data -- check for WAL-only scenario
        let wal_path = dir.join(format!("{prefix}.wal"));
        if wal_path.exists() {
            let index = SparseInvertedIndex::new();
            let replayed = wal_replay(&wal_path, &index)?;
            if replayed > 0 {
                if replayed >= COMPACTION_REPLAY_THRESHOLD {
                    compact_with_prefix(dir, prefix, &index)?;
                }
                return Ok(Some(index));
            }
        }
        return Ok(None);
    }

    // Read and deserialize meta
    let meta_data = std::fs::read(&meta_path)
        .map_err(|e| Error::SparseIndexError(format!("load meta read: {e}")))?;
    let meta: SparseMeta = postcard::from_bytes(&meta_data)
        .map_err(|e| Error::SparseIndexError(format!("load meta deserialize: {e}")))?;

    if meta.version != 1 {
        return Err(Error::SparseIndexError(format!(
            "unsupported sparse meta version: {}",
            meta.version
        )));
    }

    // Read and deserialize term dictionary
    let terms_path = dir.join(format!("{prefix}.terms"));
    let terms_data = std::fs::read(&terms_path)
        .map_err(|e| Error::SparseIndexError(format!("load terms read: {e}")))?;
    let term_entries: Vec<TermEntry> = postcard::from_bytes(&terms_data)
        .map_err(|e| Error::SparseIndexError(format!("load terms deserialize: {e}")))?;

    // Read the posting index file
    let idx_path = dir.join(format!("{prefix}.idx"));
    let idx_data = std::fs::read(&idx_path)
        .map_err(|e| Error::SparseIndexError(format!("load idx read: {e}")))?;

    let postings = build_postings_from_idx(&idx_data, &term_entries)?;

    // Use doc_count from meta (more accurate than counting postings)
    #[allow(clippy::cast_possible_truncation)] // doc_count fits in usize on supported platforms
    let frozen = FrozenSegment::new(postings, meta.doc_count as usize);
    let index = SparseInvertedIndex::from_frozen_segment(frozen);

    // Replay WAL if exists
    let wal_path = dir.join(format!("{prefix}.wal"));
    let replayed = wal_replay(&wal_path, &index)?;

    if replayed >= COMPACTION_REPLAY_THRESHOLD {
        compact_with_prefix(dir, prefix, &index)?;
    }

    Ok(Some(index))
}

/// Deserializes the posting lists from a raw index buffer and its term dictionary.
///
/// Extracted to keep `load_from_disk` within the pedantic line-count budget.
fn build_postings_from_idx(
    idx_data: &[u8],
    term_entries: &[TermEntry],
) -> Result<FxHashMap<u32, (Vec<PostingEntry>, f32)>> {
    let mut postings: FxHashMap<u32, (Vec<PostingEntry>, f32)> = FxHashMap::default();

    for te in term_entries {
        #[allow(clippy::cast_possible_truncation)] // 32-bit target: file offsets won't exceed usize
        let start = te.offset as usize;
        let byte_count = (te.len as usize) * POSTING_DISK_SIZE;
        let end = start + byte_count;

        if end > idx_data.len() {
            return Err(Error::SparseIndexError(format!(
                "load idx: term {} offset {start}+{byte_count} exceeds file size {}",
                te.term_id,
                idx_data.len()
            )));
        }

        let mut entries = Vec::with_capacity(te.len as usize);
        let mut pos = start;
        for _ in 0..te.len {
            // We verified `end <= idx_data.len()` above, so every 12-byte window is in-bounds.
            // read_le_u64/f32 propagate rather than panic to catch future refactor regressions.
            let doc_id = read_le_u64(idx_data, pos, "load idx: corrupt doc_id bytes")?;
            pos += 8;
            let weight = read_le_f32(idx_data, pos, "load idx: corrupt weight bytes")?;
            pos += 4;
            entries.push(PostingEntry { doc_id, weight });
        }

        postings.insert(te.term_id, (entries, te.max_weight));
    }

    Ok(postings)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn make_vector(pairs: Vec<(u32, f32)>) -> SparseVector {
        SparseVector::new(pairs)
    }

    #[test]
    fn test_wal_write_and_replay() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("sparse.wal");

        let index1 = SparseInvertedIndex::new();
        // Insert 100 vectors and write WAL entries
        for i in 0..100u64 {
            let v = make_vector(vec![(1, 1.0), (2, 0.5 + i as f32 * 0.01)]);
            index1.insert(i, &v);
            wal_append_upsert(&wal_path, i, &v).unwrap();
        }

        // Create fresh index and replay
        let index2 = SparseInvertedIndex::new();
        let count = wal_replay(&wal_path, &index2).unwrap();
        assert_eq!(count, 100);
        assert_eq!(index2.doc_count(), 100);

        // Verify postings match
        let p1 = index1.get_all_postings(1);
        let p2 = index2.get_all_postings(1);
        assert_eq!(p1.len(), p2.len());
        for (a, b) in p1.iter().zip(p2.iter()) {
            assert_eq!(a.doc_id, b.doc_id);
            assert!((a.weight - b.weight).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_wal_truncated_entry() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("sparse.wal");

        // Write one valid entry
        let v = make_vector(vec![(1, 1.0)]);
        wal_append_upsert(&wal_path, 42, &v).unwrap();

        // Append 5 random bytes (simulating truncation)
        {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(&wal_path)
                .unwrap();
            f.write_all(&[0xFF, 0x00, 0xAA, 0xBB, 0xCC]).unwrap();
        }

        // Replay should recover the valid entry
        let index = SparseInvertedIndex::new();
        let count = wal_replay(&wal_path, &index).unwrap();
        assert_eq!(count, 1);
        assert_eq!(index.doc_count(), 1);

        let postings = index.get_all_postings(1);
        assert_eq!(postings.len(), 1);
        assert_eq!(postings[0].doc_id, 42);
    }

    #[test]
    fn test_wal_delete_replay() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("sparse.wal");

        let v = make_vector(vec![(1, 1.0)]);
        wal_append_upsert(&wal_path, 1, &v).unwrap();
        wal_append_upsert(&wal_path, 2, &v).unwrap();
        wal_append_delete(&wal_path, 1).unwrap();

        let index = SparseInvertedIndex::new();
        let count = wal_replay(&wal_path, &index).unwrap();
        assert_eq!(count, 3);
        // doc_count is 1 (2 inserts - 1 delete)
        assert_eq!(index.doc_count(), 1);

        let postings = index.get_all_postings(1);
        assert_eq!(postings.len(), 1);
        assert_eq!(postings[0].doc_id, 2);
    }

    #[test]
    fn test_compaction_round_trip() {
        let dir = tempdir().unwrap();

        let index1 = SparseInvertedIndex::new();
        for i in 0..500u64 {
            let v = make_vector(vec![
                (i as u32 % 50, 1.0 + (i as f32) * 0.001),
                (100 + i as u32 % 20, 0.5),
            ]);
            index1.insert(i, &v);
        }

        // Compact to disk
        compact(dir.path(), &index1).unwrap();

        // Load from disk
        let loaded = load_from_disk(dir.path()).unwrap();
        assert!(loaded.is_some());
        let index2 = loaded.unwrap();

        assert_eq!(index2.doc_count(), 500);

        // Verify search results match for a sample term
        let p1 = index1.get_all_postings(5);
        let p2 = index2.get_all_postings(5);
        assert_eq!(p1.len(), p2.len());
        for (a, b) in p1.iter().zip(p2.iter()) {
            assert_eq!(a.doc_id, b.doc_id);
            assert!((a.weight - b.weight).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_empty_directory_returns_none() {
        let dir = tempdir().unwrap();
        let result = load_from_disk(dir.path()).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_full_restart_simulation() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("sparse.wal");

        // Phase 1: Insert and compact some vectors
        let index1 = SparseInvertedIndex::new();
        for i in 0..50u64 {
            let v = make_vector(vec![(1, 1.0), (2, 2.0)]);
            index1.insert(i, &v);
        }
        compact(dir.path(), &index1).unwrap();

        // Phase 2: More inserts via WAL only (simulating in-flight mutations)
        for i in 50..60u64 {
            let v = make_vector(vec![(1, 3.0), (3, 1.0)]);
            wal_append_upsert(&wal_path, i, &v).unwrap();
        }

        // Phase 3: Simulate restart — load from disk + replay WAL
        let loaded = load_from_disk(dir.path()).unwrap();
        assert!(loaded.is_some());
        let index2 = loaded.unwrap();

        // Should have 50 compacted + 10 replayed = 60
        assert_eq!(index2.doc_count(), 60);

        // Term 1: all 60 docs
        let p1 = index2.get_all_postings(1);
        assert_eq!(p1.len(), 60);

        // Term 3: only docs 50..60
        let p3 = index2.get_all_postings(3);
        assert_eq!(p3.len(), 10);
    }

    #[test]
    fn test_meta_contains_correct_values() {
        let dir = tempdir().unwrap();

        let index = SparseInvertedIndex::new();
        for i in 0..25u64 {
            let v = make_vector(vec![(i as u32 % 5, 1.0), (10, 0.5)]);
            index.insert(i, &v);
        }
        compact(dir.path(), &index).unwrap();

        // Read meta directly
        let meta_data = std::fs::read(dir.path().join("sparse.meta")).unwrap();
        let meta: SparseMeta = postcard::from_bytes(&meta_data).unwrap();
        assert_eq!(meta.version, 1);
        assert_eq!(meta.doc_count, 25);
        // 5 terms (0..4) + term 10 = 6 terms
        assert_eq!(meta.term_count, 6);
    }

    #[test]
    fn test_wal_missing_file_returns_zero() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("nonexistent.wal");
        let index = SparseInvertedIndex::new();
        let count = wal_replay(&wal_path, &index).unwrap();
        assert_eq!(count, 0);
    }

    /// Simulates a crash that left a `.tmp` file behind from a previous interrupted compaction.
    ///
    /// Verifies that `load_from_disk` ignores stale `.tmp` artefacts and correctly recovers
    /// state from the WAL alone (no `sparse.meta` present — WAL-only scenario).
    #[test]
    fn test_partial_compaction_crash_recovery() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("sparse.wal");

        // Insert 5 distinct vectors and record them only in the WAL (no compaction).
        for i in 0..5u64 {
            let v = make_vector(vec![(i as u32, 1.0 + i as f32 * 0.1)]);
            wal_append_upsert(&wal_path, i, &v).unwrap();
        }

        // Simulate an interrupted compaction: the .tmp file exists but the final sparse.idx
        // and sparse.meta files were never atomically renamed into place.
        let tmp_path = dir.path().join("sparse.idx.tmp");
        std::fs::write(&tmp_path, b"garbage partial write").unwrap();

        // sparse.meta must NOT exist so load_from_disk follows the WAL-only path.
        assert!(!dir.path().join("sparse.meta").exists());

        // load_from_disk must ignore the .tmp file and recover from the WAL.
        let loaded = load_from_disk(dir.path()).unwrap();
        assert!(
            loaded.is_some(),
            "WAL-only load should return Some after partial compaction crash"
        );
        let index = loaded.unwrap();

        // All 5 WAL-inserted documents must be present.
        assert_eq!(index.doc_count(), 5);

        // Verify each term has exactly one posting with the correct doc_id.
        for i in 0..5u64 {
            let postings = index.get_all_postings(i as u32);
            assert_eq!(
                postings.len(),
                1,
                "term {i} should have exactly one posting"
            );
            assert_eq!(postings[0].doc_id, i);
        }

        // The stale .tmp file must still be present (load_from_disk must not delete it).
        assert!(
            tmp_path.exists(),
            "load_from_disk must not remove stale .tmp artefacts"
        );
    }

    #[test]
    fn test_compaction_truncates_wal() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("sparse.wal");

        let index = SparseInvertedIndex::new();
        let v = make_vector(vec![(1, 1.0)]);
        index.insert(0, &v);
        wal_append_upsert(&wal_path, 0, &v).unwrap();

        // WAL should have content
        assert!(std::fs::metadata(&wal_path).unwrap().len() > 0);

        compact(dir.path(), &index).unwrap();

        // WAL should be truncated
        assert_eq!(std::fs::metadata(&wal_path).unwrap().len(), 0);
    }
}
