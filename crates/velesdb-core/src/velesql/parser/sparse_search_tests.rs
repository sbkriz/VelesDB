//! Tests for sparse vector search parsing (SPARSE_NEAR).

use crate::velesql::ast::condition::{SparseVectorExpr, SparseVectorSearch};
use crate::velesql::{Condition, Parser};

#[test]
fn test_parse_sparse_near_param() {
    let query = Parser::parse("SELECT * FROM docs WHERE vector SPARSE_NEAR $sv LIMIT 10").unwrap();
    let where_clause = query.select.where_clause.as_ref().unwrap();
    match where_clause {
        Condition::SparseVectorSearch(svs) => {
            assert!(matches!(&svs.vector, SparseVectorExpr::Parameter(name) if name == "sv"));
            assert!(svs.index_name.is_none());
        }
        other => panic!("Expected SparseVectorSearch, got {other:?}"),
    }
}

#[test]
fn test_parse_sparse_near_literal() {
    let query =
        Parser::parse("SELECT * FROM docs WHERE vector SPARSE_NEAR {12: 0.8, 45: 0.3} LIMIT 10")
            .unwrap();
    let where_clause = query.select.where_clause.as_ref().unwrap();
    match where_clause {
        Condition::SparseVectorSearch(svs) => {
            match &svs.vector {
                SparseVectorExpr::Literal(sv) => {
                    assert_eq!(sv.indices, vec![12, 45]);
                    assert!((sv.values[0] - 0.8).abs() < 1e-5);
                    assert!((sv.values[1] - 0.3).abs() < 1e-5);
                }
                other @ SparseVectorExpr::Parameter(_) => panic!("Expected Literal, got {other:?}"),
            }
            assert!(svs.index_name.is_none());
        }
        other => panic!("Expected SparseVectorSearch, got {other:?}"),
    }
}

#[test]
fn test_parse_sparse_near_using() {
    let query = Parser::parse(
        "SELECT * FROM docs WHERE vector SPARSE_NEAR $sv USING 'title-sparse' LIMIT 10",
    )
    .unwrap();
    let where_clause = query.select.where_clause.as_ref().unwrap();
    match where_clause {
        Condition::SparseVectorSearch(svs) => {
            assert!(matches!(&svs.vector, SparseVectorExpr::Parameter(name) if name == "sv"));
            assert_eq!(svs.index_name.as_deref(), Some("title-sparse"));
        }
        other => panic!("Expected SparseVectorSearch, got {other:?}"),
    }
}

/// Documents the accepted quote style for the USING clause.
///
/// The `string` grammar rule is defined as an atomic single-quoted rule:
///   `string = @{ "'" ~ (!"'" ~ ANY)* ~ "'" }`
///
/// Single-quoted index names (`USING 'body'`) are therefore the only accepted
/// form. The parser strips the surrounding quotes via `trim_matches('\'')`,
/// which is correct and non-trivial (the quotes ARE present in `pair.as_str()`
/// because `@{ ... }` is an atomic rule that captures the full span including
/// delimiters).
#[test]
fn test_parse_sparse_near_using_single_quoted() {
    // Canonical form: single-quoted string.
    let query =
        Parser::parse("SELECT * FROM docs WHERE vector SPARSE_NEAR $sv USING 'body' LIMIT 10")
            .unwrap();
    let where_clause = query.select.where_clause.as_ref().unwrap();
    match where_clause {
        Condition::SparseVectorSearch(svs) => {
            // Quotes must be stripped: the index_name is "body", not "'body'".
            assert_eq!(
                svs.index_name.as_deref(),
                Some("body"),
                "single-quoted USING value must be stripped of quotes"
            );
        }
        other => panic!("Expected SparseVectorSearch, got {other:?}"),
    }
}

/// Documents that double-quoted USING values are NOT accepted by the grammar.
///
/// The `string` rule only matches `'...'` (single-quoted). Attempting to use
/// `"body"` will fail at the pest parse stage, before the condition parser is
/// ever reached.
#[test]
fn test_parse_sparse_near_using_double_quoted_rejected() {
    // Double-quoted strings are not part of the `string` grammar rule and
    // must produce a parse error.
    let result =
        Parser::parse("SELECT * FROM docs WHERE vector SPARSE_NEAR $sv USING \"body\" LIMIT 10");
    assert!(
        result.is_err(),
        "double-quoted USING value must be rejected by the grammar"
    );
}

#[test]
fn test_parse_hybrid_near_and_sparse() {
    let query = Parser::parse(
        "SELECT * FROM docs WHERE vector NEAR $dv AND vector SPARSE_NEAR $sv LIMIT 10",
    )
    .unwrap();
    let where_clause = query.select.where_clause.as_ref().unwrap();
    match where_clause {
        Condition::And(left, right) => {
            assert!(
                matches!(left.as_ref(), Condition::VectorSearch(_)),
                "Left should be VectorSearch"
            );
            assert!(
                matches!(right.as_ref(), Condition::SparseVectorSearch(_)),
                "Right should be SparseVectorSearch"
            );
        }
        other => panic!("Expected And(VectorSearch, SparseVectorSearch), got {other:?}"),
    }
}

#[test]
fn test_parse_sparse_with_filter() {
    let query = Parser::parse(
        "SELECT * FROM docs WHERE vector SPARSE_NEAR $sv AND category = 'tech' LIMIT 10",
    )
    .unwrap();
    let where_clause = query.select.where_clause.as_ref().unwrap();
    match where_clause {
        Condition::And(left, right) => {
            assert!(
                matches!(left.as_ref(), Condition::SparseVectorSearch(_)),
                "Left should be SparseVectorSearch"
            );
            assert!(
                matches!(right.as_ref(), Condition::Comparison(_)),
                "Right should be Comparison"
            );
        }
        other => panic!("Expected And(SparseVectorSearch, Comparison), got {other:?}"),
    }
}

#[test]
fn test_parse_rsf_fusion() {
    let query = Parser::parse(
        "SELECT * FROM docs WHERE vector NEAR $dv AND vector SPARSE_NEAR $sv LIMIT 10 USING FUSION(strategy = 'rsf', dense_w = 0.7, sparse_w = 0.3)",
    )
    .unwrap();
    // Verify the fusion clause
    let fusion = query.select.fusion_clause.as_ref().unwrap();
    assert_eq!(fusion.strategy, crate::velesql::FusionStrategyType::Rsf);
    assert!((fusion.dense_weight.unwrap() - 0.7).abs() < 0.01);
    assert!((fusion.sparse_weight.unwrap() - 0.3).abs() < 0.01);
}

#[test]
fn test_parse_sparse_near_case_insensitive() {
    // SPARSE_NEAR is case-insensitive
    let query = Parser::parse("SELECT * FROM docs WHERE vector sparse_near $sv LIMIT 5").unwrap();
    let where_clause = query.select.where_clause.as_ref().unwrap();
    assert!(matches!(where_clause, Condition::SparseVectorSearch(_)));
}

#[test]
fn test_parse_sparse_literal_sorted() {
    // Sparse literal with out-of-order indices should be sorted
    let query =
        Parser::parse("SELECT * FROM docs WHERE vector SPARSE_NEAR {45: 0.3, 12: 0.8} LIMIT 5")
            .unwrap();
    let where_clause = query.select.where_clause.as_ref().unwrap();
    match where_clause {
        Condition::SparseVectorSearch(SparseVectorSearch {
            vector: SparseVectorExpr::Literal(sv),
            ..
        }) => {
            // SparseVector::new sorts by index
            assert_eq!(sv.indices, vec![12, 45]);
        }
        other => panic!("Expected SparseVectorSearch with Literal, got {other:?}"),
    }
}

/// Verifies that term indices with leading zeros in the sparse literal format
/// are parsed as their decimal u32 values (not as octal or any other base).
///
/// Leading zeros are significant only in the textual representation; the parser
/// calls `str::parse::<u32>()` which always uses base-10, so `"012"` → `12`
/// (not 10 as it would be in octal). This means `"012"` and `"12"` denote the
/// *same* dimension index.
///
/// When both appear in the same literal (`{012: 0.5, 12: 0.8}`), `SparseVector::new`
/// detects the duplicate index during construction and sums their weights (merge
/// semantics), then filters the result if the sum is effectively zero. Callers
/// should avoid such ambiguous literals; this test documents the current behaviour
/// so any future change to the merging strategy is caught immediately.
#[test]
fn test_parse_sparse_literal_leading_zero_index() {
    // "012" parses as u32 12 (decimal, base-10). This is identical to "12".
    let query =
        Parser::parse("SELECT * FROM docs WHERE vector SPARSE_NEAR {012: 0.5} LIMIT 5").unwrap();
    let where_clause = query.select.where_clause.as_ref().unwrap();
    match where_clause {
        Condition::SparseVectorSearch(SparseVectorSearch {
            vector: SparseVectorExpr::Literal(sv),
            ..
        }) => {
            // "012" is parsed as the decimal integer 12.
            assert_eq!(
                sv.indices,
                vec![12],
                "leading-zero index must parse as decimal 12"
            );
            assert!(
                (sv.values[0] - 0.5).abs() < 1e-5,
                "weight must be preserved exactly"
            );
        }
        other => panic!("Expected SparseVectorSearch with Literal, got {other:?}"),
    }
}

/// Verifies the last-value-wins (sum) behaviour when a sparse literal contains
/// both `"012"` and `"12"` as keys — they map to the same u32 dimension index.
///
/// `SparseVector::new` sums duplicate indices, so `{012: 0.5, 12: 0.8}` yields
/// a single entry at index 12 with weight 1.3 (0.5 + 0.8).
#[test]
fn test_parse_sparse_literal_leading_zero_duplicate_sums() {
    let query =
        Parser::parse("SELECT * FROM docs WHERE vector SPARSE_NEAR {012: 0.5, 12: 0.8} LIMIT 5")
            .unwrap();
    let where_clause = query.select.where_clause.as_ref().unwrap();
    match where_clause {
        Condition::SparseVectorSearch(SparseVectorSearch {
            vector: SparseVectorExpr::Literal(sv),
            ..
        }) => {
            // Both keys resolve to index 12; SparseVector::new sums them.
            assert_eq!(
                sv.indices,
                vec![12],
                "\"012\" and \"12\" must collapse to the same index"
            );
            assert!(
                (sv.values[0] - 1.3).abs() < 1e-5,
                "weights must be summed: 0.5 + 0.8 = 1.3, got {}",
                sv.values[0]
            );
        }
        other => panic!("Expected SparseVectorSearch with Literal, got {other:?}"),
    }
}
