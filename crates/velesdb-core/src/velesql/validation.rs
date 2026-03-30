//! Query validation for VelesQL (EPIC-044 US-007).
//!
//! Type definitions (`ValidationError`, `ValidationErrorKind`, `ValidationConfig`,
//! `ComplexityStats`) live in `validation_types.rs` to keep each file under
//! the 500 NLOC limit.

use super::ast::{ArithmeticExpr, Condition, OrderByExpr, Query, SelectColumns};
use super::error::{ParseError, ParseErrorKind};

// Re-export types so that existing `use crate::velesql::validation::*` paths
// continue to work without changes.
pub use super::validation_types::{
    ComplexityStats, ValidationConfig, ValidationError, ValidationErrorKind,
};

/// Stateless validator for semantic and complexity checks.
pub struct QueryValidator;

impl QueryValidator {
    /// Validates a query with default configuration.
    ///
    /// # Errors
    ///
    /// Returns `ValidationError` if the query fails semantic validation.
    pub fn validate(query: &Query) -> Result<(), ValidationError> {
        Self::validate_with_config(query, &ValidationConfig::default())
    }

    /// Validates a query with custom semantic configuration.
    ///
    /// # Errors
    ///
    /// Returns `ValidationError` if the query fails semantic validation.
    pub fn validate_with_config(
        query: &Query,
        config: &ValidationConfig,
    ) -> Result<(), ValidationError> {
        // LET bindings are only meaningful for SELECT/MATCH queries.
        // Reject them on DDL, DML, and introspection statements where they
        // are nonsensical.
        if (query.is_ddl_query() || query.is_dml_query() || query.is_introspection_query())
            && !query.let_bindings.is_empty()
        {
            return Err(ValidationError::new(
                ValidationErrorKind::InvalidLetBinding,
                None,
                "LET clause",
                "LET bindings are not supported with DDL, DML, or introspection statements",
            ));
        }

        // DDL statements (CREATE/DROP COLLECTION) bypass SELECT-specific
        // validation -- they have no FROM clause, no similarity conditions, etc.
        if query.is_ddl_query() {
            return Ok(());
        }

        // Introspection statements (SHOW/DESCRIBE/EXPLAIN) bypass SELECT-specific
        // validation -- they have no FROM clause or similarity conditions.
        if query.is_introspection_query() {
            return Ok(());
        }

        // DML statements bypass SELECT-specific validation.
        if query.is_dml_query() {
            return Ok(());
        }

        if let Some(ref condition) = query.select.where_clause {
            Self::validate_condition(condition, query.select.limit, config)?;
        }
        if let Some(ref compound) = query.compound {
            for (_, right_select) in &compound.operations {
                if let Some(ref condition) = right_select.where_clause {
                    Self::validate_condition(condition, right_select.limit, config)?;
                }
            }
        }

        Self::validate_similarity_context(&query.select)?;
        Self::validate_qualified_wildcards(&query.select)?;

        if let Some(ref compound) = query.compound {
            for (_, right_select) in &compound.operations {
                Self::validate_similarity_context(right_select)?;
                Self::validate_qualified_wildcards(right_select)?;
            }
        }

        Ok(())
    }

    /// Validates that `similarity()` in SELECT or ORDER BY has a score context.
    fn validate_similarity_context(
        stmt: &super::ast::SelectStatement,
    ) -> Result<(), ValidationError> {
        let has_score_context = stmt
            .where_clause
            .as_ref()
            .is_some_and(Self::has_score_producing_condition);

        if !has_score_context && Self::select_uses_similarity(&stmt.columns) {
            return Err(ValidationError::new(
                ValidationErrorKind::SimilarityWithoutContext,
                None,
                "similarity()",
                "Add a vector NEAR or similarity() predicate in WHERE to provide a score context",
            ));
        }

        if let Some(ref order_by) = stmt.order_by {
            for ob in order_by {
                Self::validate_order_by_expr(&ob.expr, has_score_context)?;
            }
        }

        Ok(())
    }

    /// Validates a single ORDER BY expression for similarity context issues.
    fn validate_order_by_expr(
        expr: &OrderByExpr,
        has_score_context: bool,
    ) -> Result<(), ValidationError> {
        match expr {
            OrderByExpr::SimilarityBare if !has_score_context => Err(ValidationError::new(
                ValidationErrorKind::SimilarityWithoutContext,
                None,
                "ORDER BY similarity()",
                "Add a vector NEAR or similarity() predicate in WHERE to provide a score context",
            )),
            OrderByExpr::Arithmetic(arith) => {
                Self::validate_arithmetic_similarity(arith, has_score_context)
            }
            _ => Ok(()),
        }
    }

    /// Recursively validates similarity() usage inside arithmetic expressions.
    fn validate_arithmetic_similarity(
        expr: &ArithmeticExpr,
        has_score_context: bool,
    ) -> Result<(), ValidationError> {
        match expr {
            ArithmeticExpr::Similarity(inner) => match inner.as_ref() {
                OrderByExpr::Similarity(_) => Err(ValidationError::new(
                    ValidationErrorKind::UnsupportedArithmeticSimilarity,
                    None,
                    "similarity(field, $vec) in arithmetic",
                    "Use bare similarity() instead; parameterized similarity inside arithmetic is not yet supported",
                )),
                OrderByExpr::SimilarityBare if !has_score_context => Err(ValidationError::new(
                    ValidationErrorKind::SimilarityWithoutContext,
                    None,
                    "similarity() in arithmetic",
                    "Add a vector NEAR or similarity() predicate in WHERE to provide a score context",
                )),
                _ => Ok(()),
            },
            ArithmeticExpr::BinaryOp { left, right, .. } => {
                Self::validate_arithmetic_similarity(left, has_score_context)?;
                Self::validate_arithmetic_similarity(right, has_score_context)
            }
            ArithmeticExpr::Literal(_) | ArithmeticExpr::Variable(_) => Ok(()),
        }
    }

    /// Returns true if `SelectColumns` references `similarity()`.
    fn select_uses_similarity(columns: &SelectColumns) -> bool {
        match columns {
            SelectColumns::SimilarityScore(_) => true,
            SelectColumns::Mixed {
                similarity_scores, ..
            } => !similarity_scores.is_empty(),
            _ => false,
        }
    }

    /// Validates that qualified wildcard aliases are declared in FROM/JOIN.
    fn validate_qualified_wildcards(
        stmt: &super::ast::SelectStatement,
    ) -> Result<(), ValidationError> {
        let aliases = &stmt.from_alias;
        let from_name = &stmt.from;

        let check_alias = |alias: &str| -> Result<(), ValidationError> {
            let is_declared = aliases.iter().any(|a| a == alias) || alias == from_name;
            if !is_declared {
                return Err(ValidationError::new(
                    ValidationErrorKind::UndeclaredAlias,
                    None,
                    format!("{alias}.*"),
                    format!(
                        "Alias '{alias}' is not declared in FROM or JOIN. Use FROM ... AS {alias}"
                    ),
                ));
            }
            Ok(())
        };

        match &stmt.columns {
            SelectColumns::QualifiedWildcard(alias) => check_alias(alias)?,
            SelectColumns::Mixed {
                qualified_wildcards,
                ..
            } => {
                for alias in qualified_wildcards {
                    check_alias(alias)?;
                }
            }
            _ => {}
        }

        Ok(())
    }

    /// Enforces complexity budgets and returns parse errors on overflow.
    ///
    /// # Errors
    ///
    /// Returns `ParseError` if the query exceeds configured complexity limits.
    pub fn enforce_query_complexity(
        query: &Query,
        raw_query: &str,
        config: &ValidationConfig,
    ) -> Result<(), ParseError> {
        if raw_query.len() > config.max_query_length {
            return Err(ParseError::new(
                ParseErrorKind::ComplexityLimit,
                config.max_query_length,
                raw_query.chars().take(128).collect::<String>(),
                format!(
                    "Query length exceeded: max={}, actual={}",
                    config.max_query_length,
                    raw_query.len()
                ),
            ));
        }

        let stats = Self::analyze_query_complexity(query);
        if stats.ast_depth > config.max_ast_depth {
            return Err(ParseError::new(
                ParseErrorKind::ComplexityLimit,
                0,
                "WHERE",
                format!(
                    "AST depth exceeded: max={}, actual={}",
                    config.max_ast_depth, stats.ast_depth
                ),
            ));
        }
        if stats.like_ilike_terms > config.max_like_ilike_terms {
            return Err(ParseError::new(
                ParseErrorKind::ComplexityLimit,
                0,
                "LIKE/ILIKE",
                format!(
                    "LIKE/ILIKE budget exceeded: max={}, actual={}",
                    config.max_like_ilike_terms, stats.like_ilike_terms
                ),
            ));
        }
        if stats.max_graph_hops > config.max_graph_expansion {
            return Err(ParseError::new(
                ParseErrorKind::ComplexityLimit,
                0,
                "MATCH",
                format!(
                    "Graph expansion exceeded: max={}, actual={}",
                    config.max_graph_expansion, stats.max_graph_hops
                ),
            ));
        }

        Ok(())
    }

    #[must_use]
    /// Extracts complexity statistics from a parsed query.
    pub fn analyze_query_complexity(query: &Query) -> ComplexityStats {
        let mut stats = ComplexityStats {
            ast_depth: 0,
            like_ilike_terms: 0,
            max_graph_hops: 0,
        };

        if let Some(ref condition) = query.select.where_clause {
            let (depth, like_count) = Self::analyze_condition(condition);
            stats.ast_depth = stats.ast_depth.max(depth);
            stats.like_ilike_terms += like_count;
        }

        if let Some(ref compound) = query.compound {
            for (_, right_select) in &compound.operations {
                if let Some(ref condition) = right_select.where_clause {
                    let (depth, like_count) = Self::analyze_condition(condition);
                    stats.ast_depth = stats.ast_depth.max(depth);
                    stats.like_ilike_terms += like_count;
                }
            }
        }

        if let Some(ref m) = query.match_clause {
            for rel in m.patterns.iter().flat_map(|p| p.relationships.iter()) {
                if let Some((_, max)) = rel.range {
                    stats.max_graph_hops = stats.max_graph_hops.max(max);
                }
            }
        }

        stats
    }

    fn validate_condition(
        condition: &Condition,
        _limit: Option<u64>,
        _config: &ValidationConfig,
    ) -> Result<(), ValidationError> {
        let similarity_count = Self::count_similarity_conditions(condition);
        if similarity_count > 1 && Self::has_multiple_similarity_in_or(condition) {
            return Err(ValidationError::multiple_similarity(
                "Multiple similarity() in OR are not supported. Use AND instead.",
            ));
        }
        Ok(())
    }

    fn analyze_condition(condition: &Condition) -> (usize, usize) {
        match condition {
            Condition::Like(_) => (1, 1),
            Condition::And(l, r) | Condition::Or(l, r) => {
                let (ld, ll) = Self::analyze_condition(l);
                let (rd, rl) = Self::analyze_condition(r);
                (1 + ld.max(rd), ll + rl)
            }
            Condition::Not(inner) | Condition::Group(inner) => {
                let (d, l) = Self::analyze_condition(inner);
                (1 + d, l)
            }
            _ => (1, 0),
        }
    }

    /// Returns true if the condition contains any score-producing search
    /// (vector, similarity, fused, or sparse).
    fn has_score_producing_condition(condition: &Condition) -> bool {
        match condition {
            Condition::Similarity(_)
            | Condition::VectorSearch(_)
            | Condition::VectorFusedSearch(_)
            | Condition::SparseVectorSearch(_) => true,
            Condition::And(l, r) | Condition::Or(l, r) => {
                Self::has_score_producing_condition(l) || Self::has_score_producing_condition(r)
            }
            Condition::Not(inner) | Condition::Group(inner) => {
                Self::has_score_producing_condition(inner)
            }
            _ => false,
        }
    }

    pub(crate) fn count_similarity_conditions(condition: &Condition) -> usize {
        match condition {
            Condition::Similarity(_)
            | Condition::VectorSearch(_)
            | Condition::VectorFusedSearch(_) => 1,
            Condition::And(l, r) | Condition::Or(l, r) => {
                Self::count_similarity_conditions(l) + Self::count_similarity_conditions(r)
            }
            Condition::Not(inner) | Condition::Group(inner) => {
                Self::count_similarity_conditions(inner)
            }
            _ => 0,
        }
    }

    #[cfg(test)]
    pub(crate) fn contains_similarity(condition: &Condition) -> bool {
        Self::count_similarity_conditions(condition) > 0
    }

    #[cfg(test)]
    pub(crate) fn has_not_similarity(condition: &Condition) -> bool {
        match condition {
            Condition::Not(inner) => Self::contains_similarity(inner),
            Condition::And(l, r) | Condition::Or(l, r) => {
                Self::has_not_similarity(l) || Self::has_not_similarity(r)
            }
            Condition::Group(inner) => Self::has_not_similarity(inner),
            _ => false,
        }
    }

    fn has_multiple_similarity_in_or(condition: &Condition) -> bool {
        match condition {
            Condition::Or(l, r) => {
                Self::count_similarity_conditions(l) > 0 && Self::count_similarity_conditions(r) > 0
                    || Self::has_multiple_similarity_in_or(l)
                    || Self::has_multiple_similarity_in_or(r)
            }
            Condition::And(l, r) => {
                Self::has_multiple_similarity_in_or(l) || Self::has_multiple_similarity_in_or(r)
            }
            Condition::Not(inner) | Condition::Group(inner) => {
                Self::has_multiple_similarity_in_or(inner)
            }
            _ => false,
        }
    }
}
