# VelesDB Performance Benchmarks

*Last updated: March 7, 2026 (v1.5 -- PQ Quantization, Sparse Search, Hybrid Search)*

---

## Test Environment

| Parameter | Value |
|-----------|-------|
| **CPU** | Intel Core i9-14900KF (24 cores, 32 threads, AVX2) |
| **RAM** | 64 GB DDR5 |
| **OS** | Microsoft Windows 11 Professionnel |
| **Rust** | rustc 1.92.0 (ded5c06cf 2025-12-08) |
| **Build** | `--release`, `target-cpu=native`, LTO thin, codegen-units=1 |
| **Framework** | Criterion.rs with `--noplot` |

Hardware configuration captured in `benchmarks/machine-config.json`.

---

## 1. Dense Search Baseline (SIMD Kernels)

SIMD kernels use AVX2 4-accumulator pipelines with runtime feature detection via `simd_dispatch`. These numbers are unchanged from v1.4.1 (EPIC-052/077/081) as the SIMD layer was not modified in v1.5.

### SIMD Kernel Latency

| Operation | 128D | 384D | 768D | 1536D | 3072D |
|-----------|------|------|------|-------|-------|
| **Dot Product** | 4.05 ns | 9.71 ns | 18.68 ns | 32.91 ns | 70.73 ns |
| **Euclidean** | 8.59 ns | 11.56 ns | 20.88 ns | 43.80 ns | 81.69 ns |
| **Cosine** | 7.87 ns | 19.67 ns | 37.26 ns | 58.09 ns | 110.13 ns |
| **Hamming** | 6.25 ns | 9.78 ns | 18.99 ns | 38.35 ns | 82.01 ns |
| **Jaccard** | 5.00 ns | 11.61 ns | 22.81 ns | 47.72 ns | 93.63 ns |

*Run `cargo bench -p velesdb-core --bench simd_benchmark -- --noplot` to regenerate.*

### Cosine Similarity (v1.5 Fresh Run)

Confirmed v1.5 cosine performance via `simd_benchmark` (March 2026):

| Dimension | Native Kernel | Engine Dispatch | Overhead |
|-----------|---------------|-----------------|----------|
| 384D | 18.0 ns | 18.8 ns | 4.6% |
| 768D | 35.1 ns | 35.6 ns | 1.4% |
| 1536D | 56.0 ns | 63.3 ns | 13.0% |

Engine dispatch overhead is minimal at typical embedding dimensions (384D--768D).

### Throughput

| Dimension | Dot Product | Throughput |
|-----------|-------------|------------|
| 768D | 18.68 ns | 41.1 Gelem/s |
| 1536D | 32.91 ns | 46.6 Gelem/s |
| 3072D | 70.73 ns | 43.4 Gelem/s |

### Batch Distance Computation

| Benchmark | Latency | Per-Vector |
|-----------|---------|------------|
| Native 1000x768D | 38.3 us | 38.3 ns |
| Engine 1000x768D | 38.8 us | 38.8 ns |

---

## 2. PQ Recall and Latency

Product Quantization (PQ) trades recall for memory compression and faster approximate search. Benchmarked with Criterion.

### PQ Recall (pq_recall_benchmark)

**Setup:** 5,000 vectors, 128D, L2 distance, 10 clusters, 50 queries, k=10.

| Mode | Recall@10 | Search Latency (50 queries) | Per-Query |
|------|-----------|----------------------------|-----------|
| **Full Precision** | 87.6% | 19.1 ms | 382 us |
| **PQ (m=auto, rescore)** | 30.6% | 30.6 ms | 612 us |

Notes:
- Full-precision recall is 87.6% (not 100%) because HNSW is approximate search.
- PQ recall@10 of 30.6% on 128D/5K vectors is expected for standard PQ without OPQ -- low dimensionality limits subspace quality.
- Rescore oversampling (default 4x) is applied.

### PQ vs SQ8 vs Full HNSW Latency (pq_hnsw_benchmark)

**Setup:** 2,000 vectors, 64D, L2 distance, top-20 search.

| Storage Mode | Search Latency | Recall@50 | Compression |
|--------------|---------------|-----------|-------------|
| **Full Precision** | 24.9 us | baseline | 1x |
| **SQ8** | 25.2 us | 100% | 4x |
| **PQ** | 257.6 us | 68.0% | ~16-32x |

Key findings:
- **SQ8 is the best general-purpose mode**: zero recall loss with 4x compression and identical latency.
- PQ search is slower due to ADC (Asymmetric Distance Computation) table lookups, but delivers 16--32x compression for memory-constrained deployments.
- PQ recall improves significantly with higher dimensionality (256D+) and OPQ rotation.

*Run `cargo bench -p velesdb-core --bench pq_recall_benchmark -- --noplot` to regenerate recall numbers.*
*Run `cargo bench -p velesdb-core --bench pq_hnsw_benchmark -- --noplot` to regenerate latency comparison.*

---

## 3. Sparse Search Latency

Sparse vector search uses an inverted index with MaxScore optimization for early termination.

### Sparse Search (sparse_benchmark)

**Setup:** 10,000 documents, BM25-style sparse vectors, Criterion.

| Benchmark | Latency (estimate) | Notes |
|-----------|--------------------|-------|
| **Insert 10K sequential** | 69.3 ms | 6.93 us/doc |
| **Insert 10K parallel (4 threads)** | 159.4 ms | Concurrent, includes contention |
| **Search top-10, 10K corpus** | 757 us | MaxScore pruning active |
| **Search top-100, 10K corpus** | 762 us | Minimal cost for larger k |
| **Concurrent 16-thread (8 insert + 8 search)** | 189.3 ms | Mixed read/write workload |

Latency percentiles are not separately measured by Criterion (which reports confidence intervals). The estimates above represent the mean of the sampling distribution.

| Metric | Value |
|--------|-------|
| **MaxScore threshold** | 30% coverage (total postings > 0.3 * doc_count * query_nnz) |
| **Accumulator strategy** | Dense array up to 10M doc IDs, FxHashMap above |
| **Linear scan fallback** | When coverage exceeds threshold |

*Run `cargo bench -p velesdb-core --bench sparse_benchmark -- --noplot` to regenerate.*

---

## 4. Hybrid Search

Hybrid search combines dense vector similarity with sparse keyword matching using Reciprocal Rank Fusion (RRF, k=60) or Relative Score Fusion (RSF).

No dedicated hybrid benchmark suite exists yet. Performance can be estimated from the individual components:

| Component | Latency (10K corpus) | Source |
|-----------|---------------------|--------|
| Dense HNSW search (k=10, 768D) | ~57 us | hnsw_benchmark |
| Sparse search (top-10, 10K) | ~757 us | sparse_benchmark |
| RRF fusion overhead | negligible (score merging) | -- |
| **Estimated hybrid total** | **~815 us** | Dense + Sparse + fusion |

The RRF fusion step is a simple score merge with no distance computation, so hybrid latency is dominated by the sparse search branch. For workloads where sparse search is the bottleneck, the MaxScore optimization provides early termination on high-selectivity queries.

To run a hybrid search benchmark when available:
```bash
cargo bench -p velesdb-core --bench hybrid_benchmark -- --noplot
```

---

## 5. HNSW Vector Search

| Operation | Latency | Throughput |
|-----------|---------|------------|
| **Search k=10** (10K/768D) | 57 us | 9.2K qps |
| **Search k=50** | 90 us | -- |
| **Search k=100** | 174 us | -- |
| **Insert 1K x 768D** | 696 ms | 1.4K elem/s |
| **Parallel Insert 1K x 768D** | 443 ms | 2,259 vec/s |

### HNSW Recall Profiles (10K/128D)

| Profile | Recall@10 | Latency P50 |
|---------|-----------|-------------|
| Fast (ef=64) | 92.2% | 36 us |
| Balanced (ef=128) | 98.8% | 57 us |
| Accurate (ef=256) | 100.0% | 130 us |
| Perfect (ef=2048) | 100% | 200 us |

Recall@10 >= 95% is guaranteed for Balanced mode and above. Use `HnswParams::for_dataset_size()` for automatic parameter tuning.

---

## 6. ColumnStore Filtering

| Scale | ColumnStore | JSON Scan | Speedup |
|-------|-------------|-----------|---------|
| 10K rows | 8.6 us | 397 us | 46x |
| 100K rows | 88 us | 3.9 ms | 44x |
| 500K rows | 136 us | 18.6 ms | 137x |

---

## 7. VelesQL Parser

| Mode | Latency | Throughput |
|------|---------|------------|
| Simple Parse | 1.4 us | 707K qps |
| Vector Query | 2.0 us | 490K qps |
| Complex Query | 7.9 us | 122K qps |
| **Cache Hit** | **84 ns** | **12M qps** |
| EXPLAIN Plan | 61 ns | 16M qps |

---

## 8. Graph (EdgeStore)

| Operation | Latency |
|-----------|---------|
| **get_neighbors (degree 10)** | 155 ns |
| **get_neighbors (degree 50)** | 508 ns |
| **add_edge** | 278 ns |
| **BFS depth 3** | 3.6 us |
| **Parallel reads (8 threads)** | 346 us |

---

## 9. Competitive Analysis

### SIMD Distance Kernels

| Library | Dot Product 1536D | Notes |
|---------|-------------------|-------|
| **VelesDB** | **32 ns** | AVX2 4-acc, native Rust |
| SimSIMD | ~25-30 ns | AVX-512, C library |
| NumPy | ~200-400 ns | BLAS backend |
| SciPy | ~300-500 ns | No SIMD optimization |

### Vector Database Search Latency

| Database | Search Latency | Scale | Notes |
|----------|---------------|-------|-------|
| **VelesDB** | **< 1 ms** | 10K | Local, in-memory HNSW |
| Milvus | < 10 ms p50 | 1M+ | Distributed |
| Qdrant | 20-50 ms | 1M+ | Cloud/distributed |
| pgvector | 45-100 ms | 100K+ | PostgreSQL extension |
| Redis | ~5 ms | 1M+ | In-memory |

---

## 10. Performance Targets by Scale

| Dataset Size | Search P99 | Recall@10 | Status |
|--------------|------------|-----------|--------|
| 10K vectors | < 1 ms | >= 98% | Achieved |
| 100K vectors | < 5 ms | >= 95% | Achieved (96.1%) |
| 1M vectors | < 50 ms | >= 95% | Target |

---

## Methodology

- **Hardware**: See Test Environment section above
- **Framework**: Criterion.rs (`--release`, `--noplot`)
- **Concurrency**: Tests run with `--test-threads=1` for isolation
- **Reproducibility**: Seeded RNG for synthetic data generation
- **Reporting**: Criterion `estimate` value (center of confidence interval)

All benchmarks can be reproduced with:
```bash
cargo bench -p velesdb-core --bench <benchmark_name> -- --noplot
```
