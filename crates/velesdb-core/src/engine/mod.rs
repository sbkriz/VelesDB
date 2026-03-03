//! Internal engine layer for VelesDB collection subsystems.
//!
//! Each engine encapsulates a distinct domain with its own lock ordering:
//! - [`VectorEngine`]: HNSW index + quantization caches + mmap storage
//! - [`PayloadEngine`]: log-structured payload storage + BM25 text index
//! - [`GraphEngine`]: edge store + property/range indexes + traversal
//!
//! All engines are `pub(crate)` — they are implementation details, not public API.

pub(crate) mod graph;
pub(crate) mod payload;
pub(crate) mod vector;

#[allow(unused_imports)]
pub(crate) use graph::GraphEngine;
#[allow(unused_imports)]
pub(crate) use payload::PayloadEngine;
#[allow(unused_imports)]
pub(crate) use vector::VectorEngine;
