//! Sparse vector index using inverted index with segment isolation.
//!
//! Provides `SparseVector` for sparse embeddings (e.g., SPLADE, BM25 term weights)
//! and `SparseInvertedIndex` for efficient sparse-to-sparse similarity search.

pub mod inverted_index;
#[cfg(feature = "persistence")]
pub mod persistence;
pub mod search;
pub mod types;

pub use inverted_index::SparseInvertedIndex;
pub use search::sparse_search;
pub use types::{PostingEntry, ScoredDoc, SparseVector};
