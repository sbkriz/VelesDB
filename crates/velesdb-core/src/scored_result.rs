//! Unified scored result type for vector search and graph traversal.
//!
//! `ScoredResult` replaces scattered `(u64, f32)` tuple patterns across search
//! paths, providing a named, self-documenting type with bidirectional conversions.

use crate::sparse_index::types::ScoredDoc;

/// A search result pairing an item identifier with a relevance score.
///
/// Used as the canonical return type across vector search, sparse search,
/// and hybrid fusion pipelines.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoredResult {
    /// Unique identifier of the matched item.
    pub id: u64,
    /// Relevance score (interpretation depends on metric: higher may be better
    /// for similarity, lower for distance).
    pub score: f32,
}

impl ScoredResult {
    /// Creates a new scored result.
    #[must_use]
    #[inline]
    pub fn new(id: u64, score: f32) -> Self {
        Self { id, score }
    }
}

// --- Tuple conversions ---

impl From<(u64, f32)> for ScoredResult {
    #[inline]
    fn from((id, score): (u64, f32)) -> Self {
        Self { id, score }
    }
}

impl From<ScoredResult> for (u64, f32) {
    #[inline]
    fn from(sr: ScoredResult) -> Self {
        (sr.id, sr.score)
    }
}

// --- ScoredDoc conversions ---

impl From<ScoredDoc> for ScoredResult {
    #[inline]
    fn from(sd: ScoredDoc) -> Self {
        Self {
            id: sd.doc_id,
            score: sd.score,
        }
    }
}

impl From<ScoredResult> for ScoredDoc {
    #[inline]
    fn from(sr: ScoredResult) -> Self {
        Self {
            doc_id: sr.id,
            score: sr.score,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let sr = ScoredResult::new(42, 0.95);
        assert_eq!(sr.id, 42);
        assert!((sr.score - 0.95).abs() < f32::EPSILON);
    }

    #[test]
    fn test_from_tuple() {
        let sr: ScoredResult = (10, 0.5).into();
        assert_eq!(sr.id, 10);
        assert!((sr.score - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_into_tuple() {
        let sr = ScoredResult::new(7, 0.3);
        let (id, score): (u64, f32) = sr.into();
        assert_eq!(id, 7);
        assert!((score - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn test_from_scored_doc() {
        let sd = ScoredDoc {
            doc_id: 99,
            score: 1.5,
        };
        let sr: ScoredResult = sd.into();
        assert_eq!(sr.id, 99);
        assert!((sr.score - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_into_scored_doc() {
        let sr = ScoredResult::new(55, 2.0);
        let sd: ScoredDoc = sr.into();
        assert_eq!(sd.doc_id, 55);
        assert!((sd.score - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_vec_conversion() {
        let tuples: Vec<(u64, f32)> = vec![(1, 0.1), (2, 0.2), (3, 0.3)];
        let results: Vec<ScoredResult> = tuples.into_iter().map(ScoredResult::from).collect();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].id, 1);

        let back: Vec<(u64, f32)> = results.into_iter().map(Into::into).collect();
        assert_eq!(back.len(), 3);
        assert_eq!(back[2].0, 3);
    }
}
