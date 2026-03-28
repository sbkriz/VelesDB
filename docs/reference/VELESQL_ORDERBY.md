# VelesQL ORDER BY Reference

*Version 0.3.0 — January 2026*

This document describes the ORDER BY clause for custom result sorting in VelesDB, including support for **arithmetic expressions** combining multiple scores.

---

## Overview

VelesQL supports ORDER BY with arithmetic expressions, enabling custom scoring formulas:

```sql
SELECT * FROM documents
WHERE vector NEAR $query
ORDER BY 0.7 * vector_score + 0.3 * graph_score DESC
LIMIT 10
```

---

## Syntax

### Basic ORDER BY

```bnf
<order_clause> ::= "ORDER" "BY" <order_expr> ["ASC" | "DESC"]
<order_expr>   ::= <arithmetic_expr> | <identifier>
<arithmetic_expr> ::= <term> (("+" | "-") <term>)*
<term>         ::= <factor> (("*" | "/") <factor>)*
<factor>       ::= <number> | <identifier> | "(" <arithmetic_expr> ")"
```

### Direction

| Keyword | Description | Default |
|---------|-------------|---------|
| `DESC` | Descending (highest first) | ✅ Default for scores |
| `ASC` | Ascending (lowest first) | |

---

## Score Variables

### Available Variables

| Variable | Type | Description |
|----------|------|-------------|
| `vector_score` | f32 | Vector similarity score (0.0 - 1.0 for cosine) |
| `graph_score` | f32 | Graph relevance score |
| `fused_score` | f32 | Combined score (default fusion) |
| `bm25_score` | f32 | BM25 text relevance score |

### Variable Availability

| Query Type | vector_score | graph_score | bm25_score |
|------------|--------------|-------------|------------|
| Vector only | ✅ | ❌ | ❌ |
| Graph only | ❌ | ✅ | ❌ |
| Hybrid (vector+graph) | ✅ | ✅ | ❌ |
| Text search | ❌ | ❌ | ✅ |
| Vector + text | ✅ | ❌ | ✅ |

---

## Arithmetic Expressions

### Operators

| Operator | Precedence | Description |
|----------|------------|-------------|
| `*` | 2 | Multiplication |
| `/` | 2 | Division |
| `+` | 1 | Addition |
| `-` | 1 | Subtraction |

### Examples

```sql
-- Weighted combination
ORDER BY 0.7 * vector_score + 0.3 * graph_score DESC

-- Boost text matches
ORDER BY vector_score + 0.5 * bm25_score DESC

-- Normalize and combine
ORDER BY (vector_score + graph_score) / 2 DESC

-- Complex formula
ORDER BY 0.5 * vector_score + 0.3 * graph_score + 0.2 * bm25_score DESC
```

---

## Common Patterns

### 1. Vector-First Ranking

Prioritize vector similarity:

```sql
SELECT * FROM docs
WHERE vector NEAR $v
ORDER BY vector_score DESC
LIMIT 10
```

### 2. Balanced Hybrid

Equal weight to vector and graph:

```sql
SELECT * FROM docs
WHERE vector NEAR $v
ORDER BY (vector_score + graph_score) / 2 DESC
LIMIT 10
```

### 3. Graph-Boosted

Prefer well-connected nodes:

```sql
SELECT * FROM docs
WHERE vector NEAR $v
ORDER BY 0.4 * vector_score + 0.6 * graph_score DESC
LIMIT 10
```

### 4. Text-Enhanced

Boost keyword matches:

```sql
SELECT * FROM docs
WHERE vector NEAR $v AND content MATCH 'rust'
ORDER BY 0.6 * vector_score + 0.4 * bm25_score DESC
LIMIT 10
```

### 5. Recency Boost

Combine with metadata (requires column reference):

```sql
SELECT * FROM articles
WHERE vector NEAR $v
ORDER BY vector_score * (1 + 0.1 * recency_factor) DESC
LIMIT 10
```

---

## Score Evaluation

### Engine: `ordering.rs`

VelesDB uses a recursive expression evaluator for ORDER BY arithmetic expressions.
The `ScoreContext` holds the pre-computed search score and optional payload, and
`evaluate_arithmetic` walks the `ArithmeticExpr` tree:

```rust
pub(crate) struct ScoreContext<'a> {
    search_score: f32,
    payload: Option<&'a serde_json::Value>,
}

pub(crate) fn evaluate_arithmetic(expr: &ArithmeticExpr, ctx: &ScoreContext<'_>) -> f32 {
    // Recursively evaluates arithmetic expression with score variables
    // Division by zero returns 0.0 (safe default for sorting)
}
```

### Performance

| Expression Complexity | Evaluation Time |
|----------------------|-----------------|
| Single variable | ~5 ns |
| Simple binary op | ~10 ns |
| Complex (5+ ops) | ~25 ns |

---

## Best Practices

### 1. Use Meaningful Weights

Choose weights based on your use case:

```sql
-- RAG: prioritize semantic match
ORDER BY 0.8 * vector_score + 0.2 * bm25_score DESC

-- Social: prioritize connections
ORDER BY 0.3 * vector_score + 0.7 * graph_score DESC
```

### 2. Normalize Scores

When combining different score types:

```sql
-- Scores may have different ranges
ORDER BY 0.5 * vector_score + 0.5 * (graph_score / max_graph_score) DESC
```

### 3. Test with EXPLAIN

Verify scoring behavior:

```sql
EXPLAIN SELECT * FROM docs
WHERE vector NEAR $v
ORDER BY 0.7 * vector_score + 0.3 * graph_score DESC
LIMIT 10
```

### 4. Division by Zero is Safe

Division by zero in ORDER BY arithmetic expressions returns `0.0` automatically.
No special handling is needed:

```sql
-- Safe: returns 0.0 if divisor is zero
ORDER BY vector_score / divisor DESC
```

---

## Limitations

| Feature | Status | Notes |
|---------|--------|-------|
| Arithmetic expressions | ✅ | +, -, *, / |
| Score variables | ✅ | vector_score, graph_score, etc. |
| Column references | ✅ | Payload field values resolved as f32 |
| Multiple ORDER BY | ✅ | Comma-separated, each with own direction |
| Functions (ABS, SQRT) | ❌ | Not supported |
| CASE expressions | ❌ | Not supported |

---

## Troubleshooting

### Unknown score variable resolves to 0.0

Unknown variable names (not a built-in like `vector_score` and not present in the
payload) silently resolve to `0.0`. This means misspelled variable names will not
produce errors but will effectively contribute nothing to the combined score.

**Tip**: Use valid built-in names (`vector_score`, `fused_score`, `similarity`) or
ensure the variable corresponds to a numeric payload field.

### Division by zero returns 0.0

Division by zero in arithmetic expressions returns `0.0` (safe default for sorting).
No error is raised.

---

## SDK Examples

### Python

```python
results = collection.query(
    """
    SELECT * FROM docs 
    WHERE vector NEAR $v 
    ORDER BY 0.7 * vector_score + 0.3 * graph_score DESC
    LIMIT 10
    """,
    params={"v": query_vector}
)
```

### TypeScript

```typescript
const results = await db.query('docs',
    `SELECT * FROM docs 
     WHERE vector NEAR $v 
     ORDER BY 0.7 * vector_score + 0.3 * graph_score DESC
     LIMIT 10`,
    { v: queryVector }
);
```

### Rust

```rust
let query = Parser::parse(
    "SELECT * FROM docs WHERE vector NEAR $v \
     ORDER BY 0.7 * vector_score + 0.3 * graph_score DESC \
     LIMIT 10"
)?;
let results = collection.execute_query(&query, &params)?;
```

---

## See Also

- [VelesQL Specification](./VELESQL_SPEC.md)
- [JOIN Reference](./VELESQL_JOIN.md)
- [Multi-Model Queries Guide](../guides/MULTIMODEL_QUERIES.md)
