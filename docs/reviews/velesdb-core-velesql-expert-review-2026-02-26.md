# VelesDB Core + VelesQL — Architecture actuelle, revue experte Rust et architecture cible

Date: 2026-02-26
Périmètre: `crates/velesdb-core` + `crates/velesdb-server` (surface VelesQL)

---

## 1) Architecture actuelle (AS-IS)

## 1.1 Positionnement réel du cœur produit

Le noyau de VelesDB est déjà **hybride** et combine:

- **Vectoriel** via HNSW + SIMD + quantization.
- **Graph** (nœuds/arêtes, traversals, index par labels).
- **Multi-column** via `ColumnStore` typé + bitmap filtering.
- **Recherche hybride** (vector + texte/BM25 + fusion).
- **Langage VelesQL** (parser, AST, validation, planner, agrégations, clauses avancées).

Cette combinaison correspond bien à la cible “graph + multicolumn + vector” et doit être traitée comme un seul moteur logique, pas trois moteurs isolés.

## 1.2 Découpage modulaire actuel

### A. Cœur `velesdb-core`

- `database`: ouverture/gestion des collections.
- `collection`: runtime principal (types de collections, search, graph ops, stats).
- `index`: HNSW (et composants ANN).
- `column_store`: stockage colonne + filtrage bitmap + string table.
- `distance` + `simd_native`: 5 métriques (Cosine, Euclidean, DotProduct, Hamming, Jaccard).
- `quantization`: SQ/Binary/PQ selon modes.
- `velesql`: parser + planification + validation + cache de parsing.
- `metrics`: instrumentation opérationnelle/retrieval/latence.

### B. Surface d’exécution

- `velesdb-server` expose les handlers API (query/search/graph/etc.).
- Les bindings (Python/WASM/mobile) consomment le moteur cœur.

## 1.3 Flux d’exécution (simplifié)

1. Entrée requête (API/SDK) → VelesQL parser.
2. Validation + planning (rules/cost simplifiées).
3. Exécution orientée collection:
   - préfiltrage metadata (`ColumnStore`),
   - ANN (`HNSW`) pour la partie vectorielle,
   - opérations graph (match/traversal),
   - fusion/ranking final.
4. Retour résultats + métriques.

## 1.4 Forces techniques déjà visibles

- Très bonne base Rust modulaire.
- SIMD natif assumé et multi-métriques.
- Socle graph et column déjà présent dans le cœur (pas seulement “plugin”).
- Surface de tests riche (unit, intégration, benchmarks, fuzz targets).

## 1.5 Écarts d’architecture observés

1. **Contrat “hybride natif” incomplet dans la planification**: l’optimiseur reste surtout orienté exécution par blocs, pas encore unifié cost-based cross-domain (graph + vector + column).
2. **Gestion du cache VelesQL à durcir** (collisions hash/LRU invariants) pour un niveau production multi-tenant strict.
3. **Guardrails de complexité requête** à harmoniser (LIKE/ILIKE, profondeur parser, budgets traversal/joins).
4. **Chemins d’erreur à normaliser** pour éviter tout panic exploitable en corruption/entrée malveillante.
5. **Doc d’architecture produit**: certaines descriptions ne reflètent pas partout la réalité 5 métriques + cœur hybride.

---

## 2) Revue experte (architecte logiciel Rust)

## 2.1 Diagnostic global

- Le projet est sur une **bonne trajectoire de moteur unifié**.
- Les briques techniques critiques existent déjà, mais l’architecture doit évoluer d’un assemblage performant vers une **plateforme query-first gouvernée par SLO**.

## 2.2 Risques prioritaires (ordre d’impact)

### R1 — Cohérence et sûreté du cache de parsing VelesQL

Si le cache ne valide pas strictement la requête originale et ses invariants LRU, risque de confusion de requêtes, baisse de hit-rate réel, et comportements non déterministes sous charge.

### R2 — Budget de complexité requête non global

Sans budget central (tokens, profondeur AST, coût LIKE/ILIKE, expansion graph), les charges adversariales peuvent saturer CPU/RAM.

### R3 — Optimisation inter-domaines insuffisante

Le vrai différenciateur de VelesDB (graph + multicolumn + vector) nécessite un optimiseur unifié avec cardinalité/coût inter-opérateurs; sinon on perd les gains de composition.

### R4 — Résilience I/O et corruption

Tous les chemins critiques de lecture mmap/index doivent être strictement fallibles (erreurs structurées) et jamais reposer sur des assertions fatales en prod.

### R5 — “Contract drift” documentation ↔ implémentation

Les docs doivent refléter explicitement le cœur hybride + 5 métriques + choix d’optimisation, sinon dette de compréhension pour clients/ops/contributeurs.

---

## 3) Architecture cible (TO-BE)

## 3.1 Principes directeurs

1. **Hybrid-first by design**: chaque requête peut combiner graph, colonne, vectoriel sans rupture.
2. **Cost-based unifié**: un seul modèle de coût multi-opérateurs.
3. **Fail-safe runtime**: aucune panique non récupérable sur input utilisateur/données corrompues.
4. **SLO-driven**: observabilité, budgets, admission control.
5. **Performance portable**: SIMD natif + fallback propre + capacités explicites.

## 3.2 Blueprint cible

### A. Query Control Plane

- Parser VelesQL versionné.
- Validator avec politiques de complexité.
- Plan cache sûr (clé robuste, collision-safe, LRU exacte).
- Logical optimizer (rewrites).
- Physical optimizer (coûts cross-domain).

### B. Data Execution Plane

- **Vector operator**: ANN/HNSW + métriques (Cosine, Euclidean, DotProduct, Hamming, Jaccard).
- **Column operator**: pushdown filtres typed + bitmap.
- **Graph operator**: traversals + pattern matching + index label/property.
- **Fusion operator**: ranking multi-sources déterministe.
- **Pipeline executor**: exécution vectorisée + parallelisme contrôlé.

### C. Storage & Reliability Plane

- Formats versionnés (index/vector/column/graph metadata).
- WAL/recovery homogènes pour toutes les structures.
- Validation stricte à la lecture (pas de panic).
- Outils de repair/audit.

### D. Observability & Governance Plane

- Métriques cardinales par opérateur (latence, recall, rejets guardrails, collisions cache).
- Traces requête → plan → opérateurs.
- SLO dashboards + error budgets.

## 3.3 Contrat produit explicite à stabiliser

Le contrat officiel doit déclarer clairement:

- **Cœur hybride natif**: graph + multi-column + vector.
- **5 métriques vectorielles supportées en production**.
- **Optimisation de bout en bout**: planning + SIMD + index + fusion.

---

## 4) Plan de réalisation (tâches + critères d’acceptation)

## EPIC 1 — Sécurisation Query Control Plane

### T1.1 Cache VelesQL collision-safe
- Remplacer clé/hash-only par entrée robuste incluant texte canonique.
- Vérifier égalité stricte sur hit avant réutilisation AST.

**Critères d’acceptation**
- Tests de collision forcée: jamais de mauvais AST retourné.
- Invariant `cache_size == order_size` maintenu.
- Benchmark parse-cache: régression ≤ 2% vs baseline.

### T1.2 LRU stricte et déterministe
- Move-to-MRU sur hit.
- Suppression des doublons d’ordre.

**Critères d’acceptation**
- Tests d’invariants LRU concurrentes verts.
- Aucune éviction prématurée de clé chaude.

### T1.3 Guardrails de complexité unifiés
- Limites: longueur requête, profondeur AST, budget LIKE/ILIKE, limite expansion graph.

**Critères d’acceptation**
- Requêtes hors budget rejetées avec erreurs explicites.
- Pas de panic ni OOM sur corpus adversarial.

## EPIC 2 — Optimiseur hybride unifié

### T2.1 Statistiques et cardinalités multi-domaines
- Collecte stats vector/column/graph.
- Estimation cardinalité join/traversal/filter/ANN.

**Critères d’acceptation**
- Plans choisis améliorent p95 latence sur workloads mixtes.
- Explications de plan (`EXPLAIN`) incluent coûts par opérateur.

### T2.2 Pushdown et ordering d’opérateurs
- Pushdown filtres colonne avant ANN quand rentable.
- Heuristiques traversal-first vs vector-first selon sélectivité.

**Critères d’acceptation**
- Jeux de tests golden plan stables.
- Gain mesuré sur au moins 3 profils (RAG, graph analytics léger, recherche hybride).

## EPIC 3 — Résilience stockage et exécution

### T3.1 Suppression des chemins panic en lecture critique
- Convertir assertions runtime en erreurs `InvalidData` contextualisées.

**Critères d’acceptation**
- Tests corruption index/mmap: erreur propre, process vivant.
- Crash-recovery tests enrichis pour fichiers altérés.

### T3.2 Versionnement format + compatibilité
- Header/version/checksum sur structures persistées critiques.

**Critères d’acceptation**
- Lecture backward-compatible (N-1) prouvée.
- Outil de migration vérifié en CI.

## EPIC 4 — Performance “totalement optimisée” mesurable

### T4.1 Profiling continu SIMD/index/fusion
- Bench automatiques sur 5 métriques.
- Rapport perf par architecture CPU.

**Critères d’acceptation**
- Tableau perf officiel mis à jour par release.
- Aucune régression > seuil défini (ex: 5%) sans waivers documentés.

### T4.2 Workload packs représentatifs
- Pack RAG, recommandation, knowledge graph, multi-tenant API.

**Critères d’acceptation**
- p50/p95/p99 + recall publiés par pack.
- Garde CI sur indicateurs clés.

## EPIC 5 — Documentation d’architecture et gouvernance

### T5.1 Alignement docs “code-truth”
- Mettre à jour architecture de référence avec composantes réellement livrées.
- Mention explicite du cœur hybride et des 5 métriques.

**Critères d’acceptation**
- Matrice “doc ↔ modules code” validée.
- Revue architecture approuvée et versionnée.

### T5.2 Runbooks opératoires
- Limites de requêtes, tuning HNSW/graph/column, incident playbooks.

**Critères d’acceptation**
- Runbook incident testé en exercice.
- Onboarding ops < 1 journée pour exécuter les procédures critiques.

---

## 5) Roadmap conseillée (90 jours)

- **J0-J30**: EPIC 1 + T3.1 (sécurité/fiabilité immédiate).
- **J31-J60**: EPIC 2 (optimiseur hybride) + T4.1 instrumentation perf.
- **J61-J90**: EPIC 3.2 + EPIC 5 + industrialisation bench/SLO.

---

## 6) Décision d’architecture recommandée

**Décider officiellement que VelesDB est un moteur “Hybrid Query Engine”** (et non “vector DB + modules”), avec VelesQL comme plan de contrôle unique.
C’est la décision qui maximise la valeur du cœur Rust déjà en place et aligne le produit avec l’objectif: **graph + multi-column + vector, 5 métriques, performances optimisées de bout en bout**.
