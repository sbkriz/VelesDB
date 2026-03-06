//! Sparse vector index using inverted index with segment isolation.
//!
//! Provides `SparseVector` for sparse embeddings (e.g., SPLADE, BM25 term weights)
//! and `SparseInvertedIndex` for efficient sparse-to-sparse similarity search.

pub mod inverted_index;
pub mod types;
// Modules enabled in later plans:
// pub mod search;      [EPIC-062/US-001]
// pub mod persistence; [EPIC-062/US-002]

pub use inverted_index::SparseInvertedIndex;
pub use types::{PostingEntry, ScoredDoc, SparseVector};
