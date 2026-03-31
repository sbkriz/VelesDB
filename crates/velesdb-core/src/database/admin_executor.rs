//! Admin executor for `VelesQL` v3.6.
//!
//! Handles FLUSH statements by delegating to existing [`Database`]
//! and collection flush APIs.

use crate::velesql::{AdminStatement, FlushStatement};
use crate::{Error, Result, SearchResult};

use super::Database;

/// Applies a flush function (fast or full) to a collection reference.
///
/// This helper avoids repeating `if full { flush_full() } else { flush() }`
/// across each registry loop.
fn flush_with<T, F, G>(coll: &T, full: bool, fast: F, full_fn: G) -> Result<()>
where
    F: FnOnce(&T) -> Result<()>,
    G: FnOnce(&T) -> Result<()>,
{
    if full { full_fn(coll) } else { fast(coll) }
}

impl Database {
    /// Dispatches an admin statement to the appropriate executor.
    ///
    /// # Errors
    ///
    /// Returns an error if the target collection does not exist (named FLUSH)
    /// or if the flush operation itself fails.
    pub(super) fn execute_admin(&self, stmt: &AdminStatement) -> Result<Vec<SearchResult>> {
        match stmt {
            AdminStatement::Flush(flush) => self.execute_flush(flush),
        }
    }

    /// Executes a `FLUSH [FULL] [collection]` statement.
    ///
    /// - No collection: flushes all collections in every registry.
    /// - Named collection: resolves and flushes that single collection.
    /// - `full = true`: includes index serialization (`flush_full`).
    /// - `full = false`: WAL-only fast flush.
    ///
    /// # Errors
    ///
    /// Returns an error if a named collection does not exist or if any
    /// individual flush operation fails.
    fn execute_flush(&self, stmt: &FlushStatement) -> Result<Vec<SearchResult>> {
        if let Some(ref name) = stmt.collection {
            self.flush_single_collection(name, stmt.full)?;
        } else {
            self.flush_all_collections(stmt.full)?;
        }

        let payload = serde_json::json!({
            "status": "flushed",
            "full": stmt.full,
        });
        let result = SearchResult::new(crate::Point::metadata_only(0, payload), 0.0);
        Ok(vec![result])
    }

    /// Flushes a single collection by name, choosing fast or full mode.
    ///
    /// Checks vector, graph, and metadata registries in order.
    ///
    /// # Errors
    ///
    /// Returns `CollectionNotFound` if the name does not match any registry.
    fn flush_single_collection(&self, name: &str, full: bool) -> Result<()> {
        if let Some(vc) = self.get_vector_collection(name) {
            return if full { vc.flush_full() } else { vc.flush() };
        }
        if let Some(gc) = self.get_graph_collection(name) {
            return if full { gc.flush_full() } else { gc.flush() };
        }
        if let Some(mc) = self.get_metadata_collection(name) {
            return if full { mc.flush_full() } else { mc.flush() };
        }
        Err(Error::CollectionNotFound(name.to_string()))
    }

    /// Flushes all collections across all registries.
    ///
    /// Iterates vector, graph, and metadata registries. Uses fast or full
    /// mode based on the `full` flag. Fails on the first error encountered.
    ///
    /// # Errors
    ///
    /// Returns the first flush error encountered.
    fn flush_all_collections(&self, full: bool) -> Result<()> {
        for (_, vc) in self.vector_colls.read().iter() {
            flush_with(vc, full, |c| c.flush(), |c| c.flush_full())?;
        }
        for (_, gc) in self.graph_colls.read().iter() {
            flush_with(gc, full, |c| c.flush(), |c| c.flush_full())?;
        }
        for (_, mc) in self.metadata_colls.read().iter() {
            flush_with(mc, full, |c| c.flush(), |c| c.flush_full())?;
        }
        Ok(())
    }
}
