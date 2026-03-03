//! Vector engine: HNSW index + quantization caches + mmap vector storage.
//!
//! # Lock ordering (internal, ascending)
//!
//!   1. `storage`           (`RwLock<MmapStorage>`)
//!   2. `sq8_cache`         (`RwLock<HashMap<…>>`)
//!   3. `binary_cache`      (`RwLock<HashMap<…>>`)
//!   4. `pq_cache`          (`RwLock<HashMap<…>>`)
//!   5. `pq_quantizer`      (`RwLock<Option<ProductQuantizer>>`)
//!   6. `pq_training_buffer`(`RwLock<VecDeque<…>>`)
//!
//! Callers must never acquire these locks in reverse order.

use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::Arc;

use parking_lot::RwLock;

use crate::distance::DistanceMetric;
use crate::error::{Error, Result};
use crate::index::{HnswIndex, VectorIndex};
use crate::quantization::{
    BinaryQuantizedVector, PQVector, ProductQuantizer, QuantizedVector, StorageMode,
};
use crate::storage::{MmapStorage, VectorStorage};

type PqSample = (u64, Vec<f32>);

const PQ_TRAINING_SAMPLES: usize = 128;

/// Encapsulates HNSW index, mmap vector storage, and all quantization caches.
///
/// This is `pub(crate)` — consumers use it through `VectorCollection`.
#[derive(Clone)]
pub(crate) struct VectorEngine {
    /// Memory-mapped vector storage (on-disk).
    pub(crate) storage: Arc<RwLock<MmapStorage>>,
    /// HNSW approximate nearest-neighbor index.
    pub(crate) index: Arc<HnswIndex>,
    /// SQ8 quantized vectors cache.
    pub(crate) sq8_cache: Arc<RwLock<HashMap<u64, QuantizedVector>>>,
    /// Binary quantized vectors cache.
    pub(crate) binary_cache: Arc<RwLock<HashMap<u64, BinaryQuantizedVector>>>,
    /// Product-quantized vectors cache.
    pub(crate) pq_cache: Arc<RwLock<HashMap<u64, PQVector>>>,
    /// Trained ProductQuantizer (lazy-trained).
    pub(crate) pq_quantizer: Arc<RwLock<Option<ProductQuantizer>>>,
    /// Buffer of samples used to train the PQ codebook.
    pub(crate) pq_training_buffer: Arc<RwLock<VecDeque<PqSample>>>,
    /// Vector dimension (stored for re-open, accessed via config).
    #[allow(dead_code)]
    pub(crate) dimension: usize,
    /// Distance metric (stored for re-open, accessed via config).
    #[allow(dead_code)]
    pub(crate) metric: DistanceMetric,
    /// Storage mode (Full / SQ8 / Binary / ProductQuantization).
    pub(crate) storage_mode: StorageMode,
}

impl VectorEngine {
    /// Creates a new `VectorEngine` at the given path.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage cannot be created.
    pub(crate) fn create(
        path: &Path,
        dimension: usize,
        metric: DistanceMetric,
        storage_mode: StorageMode,
    ) -> Result<Self> {
        let storage = Arc::new(RwLock::new(
            MmapStorage::new(path, dimension).map_err(Error::Io)?,
        ));
        let index = Arc::new(HnswIndex::new(dimension, metric));
        Ok(Self {
            storage,
            index,
            sq8_cache: Arc::new(RwLock::new(HashMap::new())),
            binary_cache: Arc::new(RwLock::new(HashMap::new())),
            pq_cache: Arc::new(RwLock::new(HashMap::new())),
            pq_quantizer: Arc::new(RwLock::new(None)),
            pq_training_buffer: Arc::new(RwLock::new(VecDeque::new())),
            dimension,
            metric,
            storage_mode,
        })
    }

    /// Opens an existing `VectorEngine` at the given path.
    ///
    /// Loads the HNSW index from disk if present.
    ///
    /// # Errors
    ///
    /// Returns an error if storage or index cannot be opened.
    pub(crate) fn open(
        path: &Path,
        dimension: usize,
        metric: DistanceMetric,
        storage_mode: StorageMode,
    ) -> Result<Self> {
        let storage = Arc::new(RwLock::new(
            MmapStorage::new(path, dimension).map_err(Error::Io)?,
        ));
        let index = if path.join("hnsw.bin").exists() {
            Arc::new(HnswIndex::load(path, dimension, metric).map_err(Error::Io)?)
        } else {
            Arc::new(HnswIndex::new(dimension, metric))
        };
        Ok(Self {
            storage,
            index,
            sq8_cache: Arc::new(RwLock::new(HashMap::new())),
            binary_cache: Arc::new(RwLock::new(HashMap::new())),
            pq_cache: Arc::new(RwLock::new(HashMap::new())),
            pq_quantizer: Arc::new(RwLock::new(None)),
            pq_training_buffer: Arc::new(RwLock::new(VecDeque::new())),
            dimension,
            metric,
            storage_mode,
        })
    }

    /// Returns the number of stored vectors.
    pub(crate) fn len(&self) -> usize {
        self.storage.read().len()
    }

    /// Stores a vector and updates the HNSW index.
    ///
    /// # Errors
    ///
    /// Returns an error if the dimension mismatches or storage fails.
    pub(crate) fn store_vector(&self, id: u64, vector: &[f32]) -> Result<()> {
        self.storage.write().store(id, vector).map_err(Error::Io)?;
        self.index.insert(id, vector);
        self.cache_quantized(id, vector);
        Ok(())
    }

    /// Retrieves a raw vector by ID.
    pub(crate) fn retrieve_vector(&self, id: u64) -> Option<Vec<f32>> {
        self.storage.read().retrieve(id).ok().flatten()
    }

    /// Performs kNN vector search, returning `(id, score)` pairs.
    pub(crate) fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        self.index.search(query, k)
    }

    /// Flushes storage and saves the HNSW index.
    ///
    /// # Errors
    ///
    /// Returns an error if any flush operation fails.
    pub(crate) fn flush(&self, path: &Path) -> crate::error::Result<()> {
        self.storage
            .write()
            .flush()
            .map_err(crate::error::Error::Io)?;
        self.index.save(path).map_err(crate::error::Error::Io)?;
        Ok(())
    }

    /// Deletes a vector from storage caches (HNSW soft-delete not available).
    pub(crate) fn delete_vector(&self, id: u64) {
        let _ = self.storage.write().delete(id);
        self.sq8_cache.write().remove(&id);
        self.binary_cache.write().remove(&id);
        self.pq_cache.write().remove(&id);
    }

    // -------------------------------------------------------------------------
    // Quantization helpers (internal)
    // -------------------------------------------------------------------------

    fn cache_quantized(&self, id: u64, vector: &[f32]) {
        match self.storage_mode {
            StorageMode::SQ8 => {
                let quantized = QuantizedVector::from_f32(vector);
                self.sq8_cache.write().insert(id, quantized);
            }
            StorageMode::Binary => {
                let quantized = BinaryQuantizedVector::from_f32(vector);
                self.binary_cache.write().insert(id, quantized);
            }
            StorageMode::ProductQuantization => {
                self.cache_pq_vector(id, vector);
            }
            StorageMode::Full => {}
        }
    }

    fn cache_pq_vector(&self, id: u64, vector: &[f32]) {
        let mut quantizer_guard = self.pq_quantizer.write();
        let mut backfill: Vec<PqSample> = Vec::new();

        if quantizer_guard.is_none() {
            let mut buffer = self.pq_training_buffer.write();
            buffer.push_back((id, vector.to_vec()));
            if buffer.len() >= PQ_TRAINING_SAMPLES {
                let training: Vec<Vec<f32>> = buffer.iter().map(|(_, v)| v.clone()).collect();
                let num_centroids = 256_usize.min(training.len().max(2));
                let num_subspaces = auto_num_subspaces(vector.len());
                *quantizer_guard = Some(ProductQuantizer::train(
                    &training,
                    num_subspaces,
                    num_centroids,
                ));
                backfill = buffer.drain(..).collect();
            }
        }

        if let Some(quantizer) = quantizer_guard.as_ref() {
            let mut cache = self.pq_cache.write();
            for (bid, bvec) in backfill {
                cache.insert(bid, quantizer.quantize(&bvec));
            }
            cache.insert(id, quantizer.quantize(vector));
        }
    }
}

fn auto_num_subspaces(dimension: usize) -> usize {
    let mut n = 8_usize;
    while n > 1 && dimension % n != 0 {
        n /= 2;
    }
    n.max(1)
}
