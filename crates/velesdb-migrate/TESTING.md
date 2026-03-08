# Testing VelesDB-Migrate with Real Data

Ce guide explique comment tester `velesdb-migrate` avec vos vraies données.

## 🔧 Configuration

### Variables d'environnement requises

```powershell
# Supabase - Required
$env:SUPABASE_URL = "https://YOUR_PROJECT.supabase.co"
$env:SUPABASE_SERVICE_KEY = "your-service-role-key"
$env:SUPABASE_TABLE = "your_table_name"

# Optional - Column names (defaults shown)
$env:SUPABASE_VECTOR_COL = "embedding"
$env:SUPABASE_ID_COL = "id"
```

## 🧪 Tests d'intégration

### Exécuter les tests avec données réelles

```powershell
# Depuis le dossier velesdb-core
cd /path/to/velesdb-core

# Tests E2E locaux déterministes contre velesdb-core (JSON/CSV + checkpoint + workers)
cargo test -p velesdb-migrate --test pipeline_e2e

# Tous les tests d'intégration
cargo test -p velesdb-migrate --test integration_test -- --ignored --nocapture

# Test spécifique
cargo test -p velesdb-migrate --test integration_test test_supabase_connection -- --ignored --nocapture
cargo test -p velesdb-migrate --test integration_test test_dimension_detection_accuracy -- --ignored --nocapture
```

### Tests disponibles

### Tests E2E locaux

| Test | Description |
|------|-------------|
| `pipeline_e2e` | Valide l'écriture réelle dans `velesdb-core`, la reprise par checkpoint, `dry_run`, `continue_on_error` et la cohérence `workers` |

### Tests avec sources réelles

| Test | Description |
|------|-------------|
| `test_supabase_connection` | Vérifie la connexion et détection de schéma |
| `test_supabase_extract_batch` | Extrait un batch de vecteurs |
| `test_full_migration_to_velesdb` | Migration complète (100 vecteurs) |
| `test_dimension_detection_accuracy` | Vérifie la précision de détection de dimension |

## 📊 Benchmarks

### Exécuter les benchmarks

```powershell
# Benchmarks locaux (sans connexion réseau)
cargo bench -p velesdb-migrate

# Avec données réelles Supabase (nécessite env vars)
$env:SUPABASE_URL = "https://..."
$env:SUPABASE_SERVICE_KEY = "..."
cargo bench -p velesdb-migrate
```

### Benchmarks disponibles

| Benchmark | Description |
|-----------|-------------|
| `parse_pgvector_1536d` | Parsing d'un vecteur pgvector 1536D |
| `pgvector_parse_by_dimension` | Parsing pour différentes dimensions (384-3072) |
| `vector_normalize_1536d` | Normalisation d'un vecteur |
| `vector_dot_product_1536d` | Produit scalaire |
| `process_batch_100x1536d` | Traitement d'un batch de 100 vecteurs |
| `batch_size_impact` | Impact de la taille de batch (10-1000) |
| `supabase_schema_detection` | Détection de schéma (réseau) |
| `supabase_batch_extraction` | Extraction de batch (réseau) |

### Consulter les résultats

```powershell
# Les résultats sont dans target/criterion/
# Ouvrir le rapport HTML
start target\criterion\report\index.html
```

## 🚀 Script de test complet

### Utilisation

```powershell
# Configurer les variables
$env:SUPABASE_URL = "https://YOUR_PROJECT.supabase.co"
$env:SUPABASE_SERVICE_KEY = "your-service-role-key"

# Exécuter le script de test
.\crates\velesdb-migrate\scripts\test-with-real-data.ps1 -All

# Ou options individuelles
.\crates\velesdb-migrate\scripts\test-with-real-data.ps1 -IntegrationTests
.\crates\velesdb-migrate\scripts\test-with-real-data.ps1 -Benchmarks
.\crates\velesdb-migrate\scripts\test-with-real-data.ps1 -FullMigration
```

## 📈 Exemple de résultats attendus

### Test de connexion Supabase

```
✅ Connected to Supabase!
   Collection: your_table_name
   Dimension: 1536
   Total count: Some(10000)
   Fields: 8
```

### Benchmark pgvector parsing

```
parse_pgvector_1536d    time:   [150.32 µs 151.45 µs 152.67 µs]

pgvector_parse_by_dimension/dimension/384
                        time:   [38.21 µs 38.56 µs 38.93 µs]
pgvector_parse_by_dimension/dimension/768
                        time:   [76.45 µs 77.12 µs 77.84 µs]
pgvector_parse_by_dimension/dimension/1536
                        time:   [152.34 µs 153.21 µs 154.12 µs]
```

### Benchmark extraction Supabase

```
supabase_schema_detection
                        time:   [245.3 ms 267.8 ms 291.2 ms]

supabase_batch_extraction/batch_size/10
                        time:   [312.5 ms 334.2 ms 356.8 ms]
supabase_batch_extraction/batch_size/100
                        time:   [456.7 ms 489.3 ms 523.1 ms]
```

## 🔍 Debugging

### Verbose output

```powershell
# Ajouter RUST_LOG pour plus de détails
$env:RUST_LOG = "debug"
cargo test -p velesdb-migrate --test integration_test -- --ignored --nocapture
```

### Vérifier la connexion manuellement

```powershell
# Tester avec detect
.\target\release\velesdb-migrate.exe detect `
    --source supabase `
    --url $env:SUPABASE_URL `
    --collection $env:SUPABASE_TABLE `
    --api-key $env:SUPABASE_SERVICE_KEY `
    --output test.yaml
```

## 📋 Checklist avant release

- [ ] Tests unitaires passent: `cargo test -p velesdb-migrate`
- [ ] Tests d'intégration passent avec données réelles
- [ ] Benchmarks exécutés et résultats documentés
- [ ] Détection dimension fonctionne pour toutes les sources
- [ ] Migration complète testée de bout en bout
