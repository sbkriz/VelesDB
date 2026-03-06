# Requirements: VelesDB v1.5 Open-Core

**Defined:** 2026-03-05
**Core Value:** Un seul moteur de connaissance pour les agents IA — Vector + Graph + ColumnStore, sub-milliseconde, offline, 15 Mo — sans glue code ni dépendances cloud.

## v1 Requirements

Requirements pour la release v1.5. Chaque requirement mappe à une phase du roadmap.

### Quality & Security (QUAL)

- [x] **QUAL-01**: bincode RUSTSEC-2025-0141 migré vers une alternative sound (bitcode ou postcard) — wire-format compatible si possible
- [ ] **QUAL-02**: BUG-8 corrigé — multi-alias FROM dans VelesQL produit des résultats incorrects
- [ ] **QUAL-03**: `ProductQuantizer::train()` — `assert!` / `panic!` remplacés par `Result<_, VelesError>` — input invalide ne tue plus le serveur
- [ ] **QUAL-04**: k-means++ init implémenté pour PQ — remplace l'init séquentielle déterministe qui produit des codebooks dégénérés sur données réelles
- [x] **QUAL-05**: `cargo audit || true` retiré du CI — advisory réel = CI rouge
- [ ] **QUAL-06**: Criterion baseline v1.5 enregistré dans `benchmarks/baseline.json` — seuil 15% enforced sur toutes les 35+ suites
- [ ] **QUAL-07**: Coverage code ≥ 82% maintenue après toutes les additions v1.5

### Product Quantization — EPIC-063 (PQ)

- [ ] **PQ-01**: Codebook PQ entraînable avec k-means++ (m sous-espaces, k centroids configurables)
- [ ] **PQ-02**: ADC (Asymmetric Distance Computation) avec SIMD — lookup table tient en cache L1 (m × k × 4 bytes, ~8KB pour m=8 k=256)
- [ ] **PQ-03**: OPQ pre-rotation optionnelle via `ndarray` — améliore recall ~5-15% sur données groupées
- [ ] **PQ-04**: Phase de rescore — oversampling + rerank f32 activé par défaut (évite le silent recall collapse)
- [ ] **PQ-05**: Commande VelesQL `TRAIN QUANTIZER ON <collection> WITH (m=8, k=256)` — entraînement explicite, pas automatique
- [ ] **PQ-06**: `QuantizationConfig` étendu avec variante PQ — rétrocompatible avec SQ8/Binary existants, pas de breaking change
- [ ] **PQ-07**: Suite Criterion dédiée `pq_recall` — seuils de précision (recall@10 ≥ 92% pour m=8) enregistrés dans baseline

### Sparse Vectors — EPIC-062 (SPARSE)

- [ ] **SPARSE-01**: `WeightedPostingList` — inverted index avec poids f32 par terme, SPLADE-v2/BM42-compatible (format `{term_id: u32 → weight: f32}`)
- [ ] **SPARSE-02**: Index sparse persisté sur disque — nouveau fichier `sparse.idx` dans le répertoire collection, survit aux redémarrages
- [ ] **SPARSE-03**: Recherche sparse ANN — inner product sur posting lists, support WAND ou scan linéaire selon densité
- [ ] **SPARSE-04**: Hybrid dense+sparse via RRF existant — `fusion/rrf_merge()` non modifié, fonctionne out-of-the-box
- [ ] **SPARSE-05**: Extension grammaire VelesQL — keyword `SPARSE` et clause `vector SPARSE_NEAR $sv` dans `velesql.pest`
- [ ] **SPARSE-06**: API REST — endpoints upsert avec champ `sparse_vector` + search sparse endpoint
- [ ] **SPARSE-07**: term_id u32 (4 milliards de termes — couvre tous les vocabulaires LLM modernes)

### Query Plan Cache — EPIC-065 (CACHE)

- [ ] **CACHE-01**: `CompiledPlanCache` deux niveaux — AST cache inchangé + nouveau tier plan compilé avec invalidation collection
- [ ] **CACHE-02**: `write_generation: AtomicU64` par collection — tout write incrémente, invalide tous les plans cached pour cette collection
- [ ] **CACHE-03**: Invalidation lifecycle — drop ou recreate d'une collection invalide immédiatement tous les plans associés
- [ ] **CACHE-04**: Métriques cache exposées — hit rate, miss rate, evictions accessibles via `/metrics` (Prometheus)

### Streaming Inserts — EPIC-064 (STREAM)

- [ ] **STREAM-01**: `StreamIngester` — bounded `tokio::sync::mpsc` channel, backpressure HTTP 429 quand buffer plein
- [ ] **STREAM-02**: Micro-batches drainés dans HNSW — taille configurable (défaut 128 vecteurs), calibrée contre le coût d'acquisition write lock HNSW
- [ ] **STREAM-03**: Delta buffer pour inserts pendant HNSW rebuild — vecteurs reçus mid-rebuild incorporés sans perte
- [ ] **STREAM-04**: Insert-and-immediately-searchable garanti — recherche inclut le buffer en attente de drain
- [ ] **STREAM-05**: Exclusion WASM automatique via `#[cfg(feature = "persistence")]` — pas de tokio runtime en WASM

### SDK Parity (SDK)

- [ ] **SDK-01**: Python SDK — sparse upsert/search, PQ train/config, streaming insert propagés dans velesdb-python
- [ ] **SDK-02**: TypeScript SDK — sparse vectors, PQ config, streaming insert dans `sdks/typescript`
- [ ] **SDK-03**: WASM module — sparse search sans persistence, plan cache actif (features compatibles no-persistence)
- [ ] **SDK-04**: Mobile iOS/Android — bindings UniFFI mis à jour pour API v1.5 (sparse + PQ)
- [ ] **SDK-05**: LangChain VectorStore — hybrid dense+sparse supporté nativement via le VectorStore officiel
- [ ] **SDK-06**: LlamaIndex integration — sparse + PQ config exposés dans le VectorStore
- [ ] **SDK-07**: Tauri plugin — synchronisé avec API core v1.5

### Documentation (DOC)

- [ ] **DOC-01**: README v1.5 — métriques recalculées (PQ recall, sparse latency), features v1.5, exemples mis à jour
- [ ] **DOC-02**: rustdoc complet API publique `velesdb-core` — tous les types/fonctions publics ont doc comment
- [ ] **DOC-03**: OpenAPI spec v1.5 — nouveaux endpoints sparse + streaming documentés, générée depuis annotations
- [ ] **DOC-04**: Guide migration v1.4 → v1.5 — breaking changes `QuantizationConfig`, VelesQL `SPARSE`, bincode wire-format
- [ ] **DOC-05**: `BENCHMARKS.md` v1.5 — résultats réels PQ recall@k, sparse search latency, streaming throughput
- [ ] **DOC-06**: `CHANGELOG.md` v1.5 — complet avec toutes les features, fixes, breaking changes

### Release (REL)

- [ ] **REL-01**: Tous crates versionnés `1.5.0`, publiés sur crates.io dans l'ordre des dépendances (velesdb-core en premier)
- [ ] **REL-02**: PyPI — wheels cross-platform via maturin CI matrix (linux-x86_64, linux-aarch64, macos-arm64, windows-x86_64)
- [ ] **REL-03**: npm — `@wiscale/velesdb` et `@wiscale/velesdb-wasm` publiés avec version 1.5.0
- [ ] **REL-04**: GitHub Release — notes de release structurées + artefacts binaires (Linux, macOS ARM, macOS Intel, Windows)
- [ ] **REL-05**: CI release matrix — validation cross-platform automatique avant toute publication

## v2 Requirements

Deferred à une version ultérieure (v1.6+ ou premium).

### Engine

- **DIST-01**: Distributed Mode / multi-node clustering — réservé Premium
- **SPARSE-ADV-01**: WAND exact top-k traversal pour vocabulaires > 100K termes — v1.6
- **PQ-ADV-01**: RaBitQ (arXiv:2405.12497) — algorithme 2024 prometteur, maturité à confirmer — v1.6
- **QUANT-ADV-01**: Product Quantization GPU acceleration — complexité trop élevée pour v1.5

### Premium boundary

- **PREM-01**: RBAC multi-tenant — via `DatabaseObserver`, repo premium
- **PREM-02**: Audit log — via `DatabaseObserver`, repo premium
- **PREM-03**: Admin UI — repo premium
- **PREM-04**: HA / failover — repo premium

## Out of Scope

| Feature | Reason |
|---------|--------|
| Distributed Mode open-core | Réservé Premium — architecture séparée via observer |
| Cloud-hosted / SaaS | Non-goal open-core — local-first par design |
| Sparse GPU acceleration | Complexité trop élevée, yield faible pour DB locale |
| Real-time collaborative editing | Hors du domaine vector DB |
| Full SQL compliance (DDL, transactions) | VelesQL est intentionnellement SQL-like, pas SQL-complet |
| RBAC / multi-tenancy dans open-core | Observer pattern uniquement — pas de leak de logique premium |

## Traceability

Mapping requirements → phases. Updated 2026-03-05 after roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| QUAL-01 | Phase 1: Quality Baseline & Security | Complete |
| QUAL-02 | Phase 1: Quality Baseline & Security | Pending |
| QUAL-03 | Phase 1: Quality Baseline & Security | Pending |
| QUAL-04 | Phase 1: Quality Baseline & Security | Pending |
| QUAL-05 | Phase 1: Quality Baseline & Security | Complete |
| QUAL-06 | Phase 1: Quality Baseline & Security | Pending |
| QUAL-07 | Phase 1: Quality Baseline & Security | Pending |
| PQ-01 | Phase 2: PQ Core Engine | Pending |
| PQ-02 | Phase 2: PQ Core Engine | Pending |
| PQ-03 | Phase 2: PQ Core Engine | Pending |
| PQ-04 | Phase 2: PQ Core Engine | Pending |
| PQ-05 | Phase 3: PQ Integration | Pending |
| PQ-06 | Phase 3: PQ Integration | Pending |
| PQ-07 | Phase 3: PQ Integration | Pending |
| SPARSE-01 | Phase 4: Sparse Vector Engine | Pending |
| SPARSE-02 | Phase 4: Sparse Vector Engine | Pending |
| SPARSE-03 | Phase 4: Sparse Vector Engine | Pending |
| SPARSE-04 | Phase 5: Sparse Integration | Pending |
| SPARSE-05 | Phase 5: Sparse Integration | Pending |
| SPARSE-06 | Phase 5: Sparse Integration | Pending |
| SPARSE-07 | Phase 5: Sparse Integration | Pending |
| CACHE-01 | Phase 6: Query Plan Cache | Pending |
| CACHE-02 | Phase 6: Query Plan Cache | Pending |
| CACHE-03 | Phase 6: Query Plan Cache | Pending |
| CACHE-04 | Phase 6: Query Plan Cache | Pending |
| STREAM-01 | Phase 7: Streaming Inserts | Pending |
| STREAM-02 | Phase 7: Streaming Inserts | Pending |
| STREAM-03 | Phase 7: Streaming Inserts | Pending |
| STREAM-04 | Phase 7: Streaming Inserts | Pending |
| STREAM-05 | Phase 7: Streaming Inserts | Pending |
| SDK-01 | Phase 8: SDK Parity | Pending |
| SDK-02 | Phase 8: SDK Parity | Pending |
| SDK-03 | Phase 8: SDK Parity | Pending |
| SDK-04 | Phase 8: SDK Parity | Pending |
| SDK-05 | Phase 8: SDK Parity | Pending |
| SDK-06 | Phase 8: SDK Parity | Pending |
| SDK-07 | Phase 8: SDK Parity | Pending |
| DOC-01 | Phase 9: Documentation | Pending |
| DOC-02 | Phase 9: Documentation | Pending |
| DOC-03 | Phase 9: Documentation | Pending |
| DOC-04 | Phase 9: Documentation | Pending |
| DOC-05 | Phase 9: Documentation | Pending |
| DOC-06 | Phase 9: Documentation | Pending |
| REL-01 | Phase 10: Release Readiness | Pending |
| REL-02 | Phase 10: Release Readiness | Pending |
| REL-03 | Phase 10: Release Readiness | Pending |
| REL-04 | Phase 10: Release Readiness | Pending |
| REL-05 | Phase 10: Release Readiness | Pending |

**Coverage:**
- v1 requirements: 48 total
- Mapped to phases: 48
- Unmapped: 0

---
*Requirements defined: 2026-03-05*
*Last updated: 2026-03-05 after roadmap creation*
