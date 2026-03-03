//! Payload engine: log-structured payload storage + BM25 text index.
//!
//! # Lock ordering (internal)
//!
//! Within `PayloadEngine`, the only lock is `storage` (`RwLock<LogPayloadStorage>`).
//! The `text_index` (`Bm25Index`) uses internal atomics — no external lock needed.

use std::path::Path;
use std::sync::Arc;

use parking_lot::RwLock;

use crate::error::{Error, Result};
use crate::index::Bm25Index;
use crate::storage::{LogPayloadStorage, PayloadStorage};

/// Encapsulates payload storage and full-text search.
///
/// This is `pub(crate)` — consumers use it through `VectorCollection`,
/// `GraphCollection`, or `MetadataCollection`.
#[derive(Clone)]
pub(crate) struct PayloadEngine {
    /// Log-structured on-disk payload storage.
    pub(crate) storage: Arc<RwLock<LogPayloadStorage>>,
    /// BM25 full-text index (uses internal atomics, no external lock).
    pub(crate) text_index: Arc<Bm25Index>,
}

impl PayloadEngine {
    /// Creates a new `PayloadEngine` at the given path.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage cannot be created.
    pub(crate) fn create(path: &Path) -> Result<Self> {
        let storage = Arc::new(RwLock::new(
            LogPayloadStorage::new(path).map_err(Error::Io)?,
        ));
        let text_index = Arc::new(Bm25Index::new());
        Ok(Self {
            storage,
            text_index,
        })
    }

    /// Opens an existing `PayloadEngine` at the given path and rebuilds the BM25 index.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage cannot be opened.
    pub(crate) fn open(path: &Path) -> Result<Self> {
        let storage = Arc::new(RwLock::new(
            LogPayloadStorage::new(path).map_err(Error::Io)?,
        ));
        let text_index = Arc::new(Bm25Index::new());

        // Rebuild BM25 from persisted payloads
        {
            let s = storage.read();
            for id in s.ids() {
                if let Ok(Some(payload)) = s.retrieve(id) {
                    let text = extract_text(&payload);
                    if !text.is_empty() {
                        text_index.add_document(id, &text);
                    }
                }
            }
        }

        Ok(Self {
            storage,
            text_index,
        })
    }

    /// Stores a payload and updates the BM25 index.
    ///
    /// # Errors
    ///
    /// Returns an error if storage fails.
    pub(crate) fn store(
        &self,
        id: u64,
        payload: Option<&serde_json::Value>,
        old_payload: Option<&serde_json::Value>,
    ) -> Result<()> {
        let mut s = self.storage.write();
        if let Some(p) = payload {
            s.store(id, p).map_err(Error::Io)?;
            let text = extract_text(p);
            if !text.is_empty() {
                self.text_index.add_document(id, &text);
            } else if old_payload.is_some() {
                self.text_index.remove_document(id);
            }
        } else {
            let _ = s.delete(id);
            self.text_index.remove_document(id);
        }
        Ok(())
    }

    /// Retrieves a payload by ID.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage operation fails.
    pub(crate) fn retrieve(&self, id: u64) -> Result<Option<serde_json::Value>> {
        self.storage.read().retrieve(id).map_err(Error::Io)
    }

    /// Deletes a payload by ID.
    // Reason: callers use `?` operator; keeping Result for API consistency even though
    // LogPayloadStorage::delete currently never fails.
    #[allow(clippy::unnecessary_wraps)]
    pub(crate) fn delete(&self, id: u64) -> Result<()> {
        let _ = self.storage.write().delete(id);
        self.text_index.remove_document(id);
        Ok(())
    }

    /// Returns all stored IDs.
    pub(crate) fn ids(&self) -> Vec<u64> {
        self.storage.read().ids()
    }

    /// Returns the number of stored payloads.
    pub(crate) fn len(&self) -> usize {
        self.storage.read().ids().len()
    }

    /// Flushes payload storage to disk.
    ///
    /// # Errors
    ///
    /// Returns an error if the flush fails.
    pub(crate) fn flush(&self) -> Result<()> {
        self.storage.write().flush().map_err(Error::Io)
    }

    /// Performs a BM25 full-text search.
    pub(crate) fn text_search(&self, query: &str, k: usize) -> Vec<(u64, f32)> {
        self.text_index.search(query, k)
    }
}

// ---------------------------------------------------------------------------
// Internal helper: extracts all string values from a JSON payload for BM25.
// Mirrors `Collection::extract_text_from_payload` — kept here to avoid
// a dependency on the private `collection::types` module.
// ---------------------------------------------------------------------------

fn extract_text(value: &serde_json::Value) -> String {
    let mut texts = Vec::new();
    collect_strings(value, &mut texts);
    texts.join(" ")
}

fn collect_strings(value: &serde_json::Value, out: &mut Vec<String>) {
    match value {
        serde_json::Value::String(s) => out.push(s.clone()),
        serde_json::Value::Array(arr) => {
            for item in arr {
                collect_strings(item, out);
            }
        }
        serde_json::Value::Object(obj) => {
            for v in obj.values() {
                collect_strings(v, out);
            }
        }
        _ => {}
    }
}
