# VelesDB REST API Reference

Complete REST API documentation for `velesdb-server`. All endpoints are served from the base URL `http://localhost:8080` by default.

> See also: [OpenAPI specification](../openapi.yaml) | [Lightweight API reference](api-reference.md)

---

## Collections

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/collections` | `GET` | List all collections |
| `/collections` | `POST` | Create a collection |
| `/collections/{name}` | `GET` | Get collection info |
| `/collections/{name}` | `DELETE` | Delete a collection |
| `/collections/{name}/empty` | `GET` | Check if a collection has no points |
| `/collections/{name}/flush` | `POST` | Flush collection data to disk |
| `/collections/{name}/sanity` | `GET` | Quick collection diagnostics (readiness and counters) |
| `/collections/{name}/config` | `GET` | Get collection configuration (dimensions, distance metric, HNSW params) |
| `/collections/{name}/analyze` | `POST` | Run collection analysis (index quality, fragmentation, optimization hints) |
| `/collections/{name}/stats` | `GET` | Get collection statistics (point count, memory usage, index stats) |

## Points

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/collections/{name}/points` | `POST` | Upsert points |
| `/collections/{name}/points/{id}` | `GET` | Get a point by ID |
| `/collections/{name}/points/{id}` | `DELETE` | Delete a point |
| `/collections/{name}/stream/insert` | `POST` | Stream insert a single point (bounded channel, backpressure 429) |

## Search (Vector)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/collections/{name}/search` | `POST` | Vector similarity search |
| `/collections/{name}/search/batch` | `POST` | Batch search (multiple queries) |
| `/collections/{name}/search/multi` | `POST` | Multi-query search |
| `/collections/{name}/search/text` | `POST` | BM25 full-text search |
| `/collections/{name}/search/hybrid` | `POST` | Hybrid vector + text search |
| `/collections/{name}/search/ids` | `POST` | Search by point IDs (batch ID lookup with optional filtering) |

> **Sparse & hybrid search:** Use `/collections/{name}/search` with a `sparse_vector` field for sparse-only search, or both `vector` and `sparse_vector` for hybrid dense+sparse search (auto-detected, fused via RRF/RSF).

## Graph

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/collections/{name}/graph/edges` | `GET` | Get edges for a node |
| `/collections/{name}/graph/edges` | `POST` | Add edge between nodes |
| `/collections/{name}/graph/traverse` | `POST` | BFS/DFS graph traversal |
| `/collections/{name}/graph/nodes/{node_id}/degree` | `GET` | Get node degree (in/out) |

## Indexes

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/collections/{name}/indexes` | `GET` | List indexes |
| `/collections/{name}/indexes` | `POST` | Create index on property |
| `/collections/{name}/indexes/{label}/{property}` | `DELETE` | Delete index |

## VelesQL v3.0.0 (Unified Query)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query` | `POST` | Execute VelesQL (Vector + Graph + ColumnStore queries) |
| `/aggregate` | `POST` | Execute aggregation-only VelesQL queries (`GROUP BY`/`HAVING`) |
| `/query/explain` | `POST` | Return query execution plan (EXPLAIN) |

**VelesQL v3.0.0 Features:**
- `GROUP BY` / `HAVING` with AND/OR operators
- `ORDER BY` multi-column + `similarity()` function
- `JOIN ... ON` across collections (inner join runtime support)
- `JOIN ... USING (...)` runtime supports single-column only (`USING (a, b)` rejected)
- `UNION` / `INTERSECT` / `EXCEPT` set operations
- `USING FUSION(strategy='rrf')` hybrid search
- `SPARSE_NEAR` clause for sparse vector similarity search
- `TRAIN QUANTIZER ON <collection> WITH (m=8, k=256)` for explicit PQ training
- `FUSE BY` / `USING FUSION` with `dense_weight`/`sparse_weight` for RSF fusion
- `WITH (max_groups=100)` query-time config

```sql
-- Example: Analytics with aggregation
SELECT category, COUNT(*), AVG(price) FROM products
GROUP BY category HAVING COUNT(*) > 5

-- Example: Hybrid search with fusion
SELECT * FROM docs USING FUSION(strategy='rrf', k=60) LIMIT 20

-- Example: Set operations
SELECT * FROM active UNION SELECT * FROM archived
```

> **Note:** ColumnStore operations (INSERT, UPDATE, SELECT on structured data) are performed via the `/query` endpoint using VelesQL syntax.
> For top-level `MATCH` on `/query`, include `collection` in the JSON body (or use `/collections/{name}/match`).

## Administration

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/guardrails` | `GET` | Get current query guardrails configuration (limits, timeouts) |
| `/guardrails` | `PUT` | Update query guardrails configuration |

## Health

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | `GET` | Health check |
| `/ready` | `GET` | Readiness check |
| `/metrics` | `GET` | Prometheus metrics endpoint |

---

## Request/Response Examples

### Create Collection

```bash
curl -X POST http://localhost:8080/collections \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my_vectors",
    "dimension": 768,
    "metric": "cosine"  # Options: cosine, euclidean, dot, hamming, jaccard
  }'
```

**Response:**
```json
{"message": "Collection created", "name": "my_vectors"}
```

### Upsert Points

Inserts new points or **replaces existing ones** if the ID already exists. Since v1.7, upsert is the default behavior: inserting a point with an existing ID updates the vector and payload in-place, and the HNSW graph edges are automatically reconnected. No separate delete is needed.

```bash
curl -X POST http://localhost:8080/collections/my_vectors/points \
  -H "Content-Type: application/json" \
  -d '{
    "points": [
      {
        "id": 1,
        "vector": [0.1, 0.2, 0.3, ...],
        "payload": {"title": "Document 1", "tags": ["ai", "ml"]}
      }
    ]
  }'
```

**Response:**
```json
{"message": "Points upserted", "count": 1}
```

### Streaming Insert

```bash
curl -X POST http://localhost:8080/collections/my_vectors/stream/insert \
  -H "Content-Type: application/json" \
  -d '{
    "id": 1,
    "vector": [0.1, 0.2, 0.3, 0.4],
    "payload": {"title": "Doc 1"}
  }'
```

**Response (202 Accepted):**
```json
{"message": "Point accepted into streaming buffer"}
```

> Returns `429 Too Many Requests` when the streaming buffer is full. Retry after 1 second.

### Vector Search

```bash
curl -X POST http://localhost:8080/collections/my_vectors/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, 0.3, ...],
    "top_k": 10
  }'
```

**Response:**
```json
{
  "results": [
    {"id": 1, "score": 0.95, "payload": {"title": "Document 1"}},
    {"id": 42, "score": 0.87, "payload": {"title": "Document 42"}}
  ]
}
```

### Sparse Search

```bash
curl -X POST http://localhost:8080/collections/my_vectors/search \
  -H "Content-Type: application/json" \
  -d '{
    "sparse_vector": {"42": 0.8, "137": 0.6, "891": 0.3},
    "top_k": 10
  }'
```

**Response:**
```json
{
  "results": [
    {"id": 7, "score": 1.42, "payload": {"title": "Sparse Match 1"}},
    {"id": 19, "score": 0.91, "payload": {"title": "Sparse Match 2"}}
  ]
}
```

### Batch Search

```bash
curl -X POST http://localhost:8080/collections/my_vectors/search/batch \
  -H "Content-Type: application/json" \
  -d '{
    "searches": [
      {"vector": [0.1, 0.2, ...], "top_k": 5},
      {"vector": [0.3, 0.4, ...], "top_k": 5}
    ]
  }'
```

**Response:**
```json
{
  "results": [
    {"results": [{"id": 1, "score": 0.95, "payload": {...}}]},
    {"results": [{"id": 2, "score": 0.89, "payload": {...}}]}
  ],
  "timing_ms": 1.23
}
```

### VelesQL Query

```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "SELECT * FROM my_vectors WHERE vector NEAR $v LIMIT 10",
    "params": {"v": [0.1, 0.2, 0.3, ...]}
  }'
```

**Response:**
```json
{
  "results": [
    {"id": 1, "score": 0.95, "payload": {"title": "Document 1"}}
  ],
  "timing_ms": 2.34,
  "rows_returned": 1
}
```
