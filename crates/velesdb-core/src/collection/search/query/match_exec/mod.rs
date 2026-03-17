//! MATCH query execution for graph pattern matching (EPIC-045 US-002).
//!
//! This module implements the `execute_match()` method for executing
//! Cypher-like MATCH queries on VelesDB collections.

// SAFETY: Numeric casts in MATCH query execution are intentional:
// - u64->usize for result limits: limits are small (< 1M) and bounded
// - f64->f32 for embedding vectors: precision sufficient for similarity search
// - u32->f32 for depth scoring: depth values are small (< 1000)
// - All casts are for internal query execution, not user data validation
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]

mod similarity;
mod where_eval;

use crate::collection::graph::{bfs_stream, StreamingConfig};
use crate::collection::types::Collection;
use crate::error::{Error, Result};
use crate::guardrails::QueryContext;
use crate::storage::{PayloadStorage, VectorStorage};
use crate::velesql::{GraphPattern, MatchClause};
use std::collections::HashMap;

/// Result of a MATCH query traversal.
#[derive(Debug, Clone)]
pub struct MatchResult {
    /// Node ID that was matched.
    pub node_id: u64,
    /// Depth in the traversal (0 = start node).
    pub depth: u32,
    /// Path of edge IDs from start to this node.
    pub path: Vec<u64>,
    /// Bound variables from the pattern (alias -> node_id).
    pub bindings: HashMap<String, u64>,
    /// Similarity score if combined with vector search.
    pub score: Option<f32>,
    /// Projected properties from RETURN clause (EPIC-058 US-007).
    /// Key format: "alias.property" (e.g., "author.name").
    pub projected: HashMap<String, serde_json::Value>,
}

impl MatchResult {
    /// Creates a new match result.
    #[must_use]
    pub fn new(node_id: u64, depth: u32, path: Vec<u64>) -> Self {
        Self {
            node_id,
            depth,
            path,
            bindings: HashMap::new(),
            score: None,
            projected: HashMap::new(),
        }
    }

    /// Adds a variable binding.
    #[must_use]
    pub fn with_binding(mut self, alias: String, node_id: u64) -> Self {
        self.bindings.insert(alias, node_id);
        self
    }

    /// Adds projected properties (EPIC-058 US-007).
    #[must_use]
    pub fn with_projected(mut self, projected: HashMap<String, serde_json::Value>) -> Self {
        self.projected = projected;
        self
    }
}

/// Parses a property path expression like "alias.property" (EPIC-058 US-007).
///
/// Returns `Some((alias, property))` if valid, `None` otherwise.
/// For nested paths like "doc.metadata.category", returns `("doc", "metadata.category")`.
#[must_use]
pub fn parse_property_path(expression: &str) -> Option<(&str, &str)> {
    // Skip special cases
    if expression == "*" || expression.contains('(') {
        return None;
    }

    // Split on first dot
    let dot_pos = expression.find('.')?;
    if dot_pos == 0 || dot_pos == expression.len() - 1 {
        return None;
    }

    let alias = &expression[..dot_pos];
    let property = &expression[dot_pos + 1..];
    Some((alias, property))
}

impl Collection {
    /// Executes a MATCH query on this collection (EPIC-045 US-002).
    ///
    /// This method performs graph pattern matching by:
    /// 1. Finding start nodes matching the first node pattern
    /// 2. Traversing relationships according to the pattern
    /// 3. Filtering results by WHERE clause conditions
    /// 4. Returning results according to RETURN clause
    ///
    /// # Arguments
    ///
    /// * `match_clause` - The parsed MATCH clause
    /// * `params` - Query parameters for resolving placeholders
    ///
    /// # Returns
    ///
    /// Vector of `MatchResult` containing matched nodes and their bindings.
    ///
    /// # Errors
    ///
    /// Returns an error if the query cannot be executed.
    /// Executes a MATCH query without guard-rail context (backward-compatible entry point).
    pub fn execute_match(
        &self,
        match_clause: &MatchClause,
        params: &HashMap<String, serde_json::Value>,
    ) -> Result<Vec<MatchResult>> {
        self.execute_match_with_context(match_clause, params, None)
    }

    /// Executes a MATCH query on this collection (EPIC-045 US-002, EPIC-048).
    ///
    /// This method performs graph pattern matching by:
    /// 1. Finding start nodes matching the first node pattern
    /// 2. Traversing relationships according to the pattern
    /// 3. Enforcing guard-rail limits (depth, cardinality, timeout) if a context is provided
    /// 4. Filtering results by WHERE clause conditions
    /// 5. Returning results according to RETURN clause
    ///
    /// # Arguments
    ///
    /// * `match_clause` - The parsed MATCH clause
    /// * `params` - Query parameters for resolving placeholders
    /// * `ctx` - Optional guard-rail context for enforcing limits
    ///
    /// # Errors
    ///
    /// Returns an error if the query cannot be executed or a guard-rail is violated.
    /// Executes a MATCH query on this collection (EPIC-045 US-002, EPIC-048).
    pub fn execute_match_with_context(
        &self,
        match_clause: &MatchClause,
        params: &HashMap<String, serde_json::Value>,
        ctx: Option<&QueryContext>,
    ) -> Result<Vec<MatchResult>> {
        let limit = match_clause.return_clause.limit.map_or(100, |l| l as usize);

        if match_clause.patterns.is_empty() {
            return Err(Error::Config(
                "MATCH query must have at least one pattern".to_string(),
            ));
        }

        let mut all_results: Vec<MatchResult> = Vec::new();
        let mut iteration_count: u32 = 0;
        let mut reported_cardinality: usize = 0;
        let edge_store = self.edge_store.read();

        for pattern in &match_clause.patterns {
            if all_results.len() >= limit {
                break;
            }

            let mut seen_pairs: std::collections::HashSet<(u64, u64)> =
                std::collections::HashSet::new();

            let start_nodes = self.find_start_nodes(pattern)?;
            if start_nodes.is_empty() {
                continue;
            }

            if pattern.relationships.is_empty() {
                self.collect_single_node_results(
                    &start_nodes,
                    match_clause,
                    params,
                    &mut seen_pairs,
                    &mut all_results,
                    limit,
                )?;
                continue;
            }

            let max_depth = Self::compute_max_depth(pattern);
            let rel_types = Self::extract_rel_types(pattern);

            for (start_id, start_bindings) in start_nodes {
                if all_results.len() >= limit {
                    break;
                }

                let config = StreamingConfig::default()
                    .with_limit(limit.saturating_sub(all_results.len()))
                    .with_max_depth(max_depth)
                    .with_rel_types(rel_types.clone());

                for traversal_result in bfs_stream(&edge_store, start_id, config) {
                    if all_results.len() >= limit {
                        break;
                    }

                    iteration_count += 1;
                    self.check_periodic_guardrails(
                        ctx,
                        iteration_count,
                        &all_results,
                        &mut reported_cardinality,
                    )?;

                    let match_result = self.build_traversal_match_result(
                        &traversal_result,
                        &start_bindings,
                        pattern,
                        ctx,
                    )?;

                    if let Some(ref where_clause) = match_clause.where_clause {
                        if !self.evaluate_where_condition(
                            traversal_result.target_id,
                            Some(&match_result.bindings),
                            where_clause,
                            params,
                        )? {
                            continue;
                        }
                    }

                    if seen_pairs.contains(&(start_id, traversal_result.target_id)) {
                        continue;
                    }
                    seen_pairs.insert((start_id, traversal_result.target_id));

                    let mut final_result = match_result;
                    final_result.projected = self
                        .project_properties(&final_result.bindings, &match_clause.return_clause);

                    all_results.push(final_result);
                }
            }
        }

        Ok(all_results)
    }

    /// Collects results for single-node patterns (no relationships).
    fn collect_single_node_results(
        &self,
        start_nodes: &[(u64, HashMap<String, u64>)],
        match_clause: &MatchClause,
        params: &HashMap<String, serde_json::Value>,
        seen_pairs: &mut std::collections::HashSet<(u64, u64)>,
        all_results: &mut Vec<MatchResult>,
        limit: usize,
    ) -> Result<()> {
        for (node_id, bindings) in start_nodes {
            if all_results.len() >= limit {
                break;
            }
            if let Some(ref where_clause) = match_clause.where_clause {
                if !self.evaluate_where_condition(*node_id, Some(bindings), where_clause, params)? {
                    continue;
                }
            }
            if seen_pairs.contains(&(*node_id, *node_id)) {
                continue;
            }
            seen_pairs.insert((*node_id, *node_id));

            let mut result = MatchResult::new(*node_id, 0, Vec::new());
            result.bindings.clone_from(bindings);
            result.projected = self.project_properties(bindings, &match_clause.return_clause);
            all_results.push(result);
        }
        Ok(())
    }

    /// Periodic guard-rail checks every 100 iterations (EPIC-048).
    #[allow(clippy::unused_self)]
    fn check_periodic_guardrails(
        &self,
        ctx: Option<&QueryContext>,
        iteration_count: u32,
        all_results: &[MatchResult],
        reported_cardinality: &mut usize,
    ) -> Result<()> {
        if iteration_count % 100 != 0 {
            return Ok(());
        }
        let Some(ctx) = ctx else { return Ok(()) };
        ctx.check_timeout()
            .map_err(|e| Error::GuardRail(e.to_string()))?;
        let new_results = all_results.len().saturating_sub(*reported_cardinality);
        if new_results > 0 {
            ctx.check_cardinality(new_results)
                .map_err(|e| Error::GuardRail(e.to_string()))?;
            *reported_cardinality = all_results.len();
        }
        Ok(())
    }

    /// Builds a `MatchResult` from a traversal result with bindings and depth check.
    #[allow(clippy::unused_self)]
    fn build_traversal_match_result(
        &self,
        traversal_result: &crate::collection::graph::TraversalResult,
        start_bindings: &HashMap<String, u64>,
        pattern: &GraphPattern,
        ctx: Option<&QueryContext>,
    ) -> Result<MatchResult> {
        let mut match_result = MatchResult::new(
            traversal_result.target_id,
            traversal_result.depth,
            traversal_result.path.clone(),
        );
        match_result.bindings.clone_from(start_bindings);

        if let Some(ctx) = ctx {
            ctx.check_depth(traversal_result.depth)
                .map_err(|e| Error::GuardRail(e.to_string()))?;
        }

        if let Some(target_pattern) = pattern.nodes.get(traversal_result.depth as usize) {
            if let Some(ref alias) = target_pattern.alias {
                match_result
                    .bindings
                    .insert(alias.clone(), traversal_result.target_id);
            }
        }

        Ok(match_result)
    }

    /// Finds start nodes matching the first node pattern.
    fn find_start_nodes(&self, pattern: &GraphPattern) -> Result<Vec<(u64, HashMap<String, u64>)>> {
        let first_node = pattern
            .nodes
            .first()
            .ok_or_else(|| Error::Config("Pattern must have at least one node".to_string()))?;

        let mut results = Vec::new();
        let payload_storage = self.payload_storage.read();
        let vector_storage = self.vector_storage.read();

        let has_label_filter = !first_node.labels.is_empty();
        let has_property_filter = !first_node.properties.is_empty();

        for id in vector_storage.ids() {
            let payload_opt = if has_label_filter || has_property_filter {
                payload_storage.retrieve(id).ok().flatten()
            } else {
                None
            };

            if has_label_filter
                && !Self::node_matches_labels(payload_opt.as_ref(), &first_node.labels)
            {
                continue;
            }
            if has_property_filter
                && !Self::node_matches_properties(payload_opt.as_ref(), &first_node.properties)
            {
                continue;
            }

            let mut bindings: HashMap<String, u64> = HashMap::new();
            if let Some(ref alias) = first_node.alias {
                bindings.insert(alias.clone(), id);
            }
            results.push((id, bindings));
        }

        Ok(results)
    }

    /// Checks if a node's payload matches the required labels.
    fn node_matches_labels(payload: Option<&serde_json::Value>, required: &[String]) -> bool {
        let Some(payload) = payload else { return false };
        let Some(labels) = payload.get("_labels").and_then(|v| v.as_array()) else {
            return false;
        };
        let node_labels: Vec<&str> = labels.iter().filter_map(|v| v.as_str()).collect();
        required.iter().all(|r| node_labels.contains(&r.as_str()))
    }

    /// Checks if a node's payload matches the required properties.
    fn node_matches_properties(
        payload: Option<&serde_json::Value>,
        properties: &HashMap<String, crate::velesql::Value>,
    ) -> bool {
        let Some(payload) = payload else { return false };
        properties.iter().all(|(key, expected)| {
            payload
                .get(key)
                .is_some_and(|actual| Self::values_match(expected, actual))
        })
    }

    /// Computes maximum traversal depth from pattern.
    fn compute_max_depth(pattern: &GraphPattern) -> u32 {
        let mut max_depth = 0u32;

        for rel in &pattern.relationships {
            if let Some((_, end)) = rel.range {
                max_depth = max_depth.saturating_add(end.min(10)); // Cap at 10
            } else {
                max_depth = max_depth.saturating_add(1);
            }
        }

        // Default to at least 1 if we have relationships
        if max_depth == 0 && !pattern.relationships.is_empty() {
            // SAFETY: Pattern relationships count is typically < 10, capped at 10 anyway
            max_depth = u32::try_from(pattern.relationships.len()).unwrap_or(10);
        }

        max_depth.min(10) // Cap at 10 for safety
    }

    /// Extracts relationship type filters from pattern.
    fn extract_rel_types(pattern: &GraphPattern) -> Vec<String> {
        let mut types = Vec::new();
        for rel in &pattern.relationships {
            types.extend(rel.types.clone());
        }
        types
    }

    /// Compares a VelesQL Value with a JSON value.
    fn values_match(velesql_value: &crate::velesql::Value, json_value: &serde_json::Value) -> bool {
        use crate::velesql::Value;

        match (velesql_value, json_value) {
            (Value::String(s), serde_json::Value::String(js)) => s == js,
            (Value::Integer(i), serde_json::Value::Number(n)) => {
                n.as_i64().is_some_and(|ni| *i == ni)
            }
            (Value::Float(f), serde_json::Value::Number(n)) => {
                n.as_f64().is_some_and(|nf| (*f - nf).abs() < 0.001)
            }
            (Value::Boolean(b), serde_json::Value::Bool(jb)) => b == jb,
            (Value::Null, serde_json::Value::Null) => true,
            _ => false,
        }
    }
}

// Tests moved to match_exec_tests.rs per project rules
