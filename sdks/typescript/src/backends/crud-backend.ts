/**
 * CRUD Backend operations for VelesDB REST API.
 *
 * Extracted from rest.ts to keep file size manageable.
 * Implements: createCollection, deleteCollection, getCollection,
 * listCollections, insert, insertBatch, delete, get, isEmpty, flush.
 */

import type {
  CollectionConfig,
  Collection,
  VectorDocument,
  RestPointId,
  SparseVector,
} from '../types';
import { NotFoundError, ValidationError, VelesDBError } from '../types';

/** Minimal transport interface for CRUD operations. */
export interface CrudTransport {
  requestJson<T>(
    method: string,
    path: string,
    body?: unknown
  ): Promise<{ data?: T; error?: { code: string; message: string } }>;
}

export function parseRestPointId(id: string | number): RestPointId {
  if (
    typeof id !== 'number' ||
    !Number.isFinite(id) ||
    id < 0 ||
    !Number.isInteger(id) ||
    id > Number.MAX_SAFE_INTEGER
  ) {
    throw new ValidationError(
      `REST backend requires numeric u64-compatible IDs in JS safe integer range (0..${Number.MAX_SAFE_INTEGER}). Received: ${String(id)}`
    );
  }
  return id;
}

export function sparseVectorToRestFormat(sv: SparseVector): Record<string, number> {
  const result: Record<string, number> = {};
  for (const [k, v] of Object.entries(sv)) {
    result[String(k)] = v;
  }
  return result;
}

export async function createCollection(
  transport: CrudTransport,
  name: string,
  config: CollectionConfig
): Promise<void> {
  const response = await transport.requestJson('POST', '/collections', {
    name,
    dimension: config.dimension,
    metric: config.metric ?? 'cosine',
    storage_mode: config.storageMode ?? 'full',
    collection_type: config.collectionType ?? 'vector',
    description: config.description,
    hnsw_m: config.hnsw?.m,
    hnsw_ef_construction: config.hnsw?.efConstruction,
  });
  if (response.error) {
    throw new VelesDBError(response.error.message, response.error.code);
  }
}

export async function deleteCollection(
  transport: CrudTransport,
  name: string
): Promise<void> {
  const response = await transport.requestJson(
    'DELETE',
    `/collections/${encodeURIComponent(name)}`
  );
  if (response.error) {
    if (response.error.code === 'NOT_FOUND') {
      throw new NotFoundError(`Collection '${name}'`);
    }
    throw new VelesDBError(response.error.message, response.error.code);
  }
}

export async function getCollection(
  transport: CrudTransport,
  name: string
): Promise<Collection | null> {
  const response = await transport.requestJson<Collection>(
    'GET',
    `/collections/${encodeURIComponent(name)}`
  );
  if (response.error) {
    if (response.error.code === 'NOT_FOUND') {
      return null;
    }
    throw new VelesDBError(response.error.message, response.error.code);
  }
  return response.data ?? null;
}

export async function listCollections(
  transport: CrudTransport
): Promise<Collection[]> {
  const response = await transport.requestJson<Collection[]>('GET', '/collections');
  if (response.error) {
    throw new VelesDBError(response.error.message, response.error.code);
  }
  return response.data ?? [];
}

export async function insert(
  transport: CrudTransport,
  collection: string,
  doc: VectorDocument
): Promise<void> {
  const restId = parseRestPointId(doc.id);
  const vector = doc.vector instanceof Float32Array
    ? Array.from(doc.vector)
    : doc.vector;

  const response = await transport.requestJson(
    'POST',
    `/collections/${encodeURIComponent(collection)}/points`,
    { points: [{ id: restId, vector, payload: doc.payload }] }
  );
  if (response.error) {
    if (response.error.code === 'NOT_FOUND') {
      throw new NotFoundError(`Collection '${collection}'`);
    }
    throw new VelesDBError(response.error.message, response.error.code);
  }
}

export async function insertBatch(
  transport: CrudTransport,
  collection: string,
  docs: VectorDocument[]
): Promise<void> {
  const vectors = docs.map(doc => ({
    id: parseRestPointId(doc.id),
    vector: doc.vector instanceof Float32Array ? Array.from(doc.vector) : doc.vector,
    payload: doc.payload,
  }));

  const response = await transport.requestJson(
    'POST',
    `/collections/${encodeURIComponent(collection)}/points`,
    { points: vectors }
  );
  if (response.error) {
    if (response.error.code === 'NOT_FOUND') {
      throw new NotFoundError(`Collection '${collection}'`);
    }
    throw new VelesDBError(response.error.message, response.error.code);
  }
}

export async function deletePoint(
  transport: CrudTransport,
  collection: string,
  id: string | number
): Promise<boolean> {
  const restId = parseRestPointId(id);
  const response = await transport.requestJson<{ deleted: boolean }>(
    'DELETE',
    `/collections/${encodeURIComponent(collection)}/points/${encodeURIComponent(String(restId))}`
  );
  if (response.error) {
    if (response.error.code === 'NOT_FOUND') {
      return false;
    }
    throw new VelesDBError(response.error.message, response.error.code);
  }
  return response.data?.deleted ?? false;
}

export async function get(
  transport: CrudTransport,
  collection: string,
  id: string | number
): Promise<VectorDocument | null> {
  const restId = parseRestPointId(id);
  const response = await transport.requestJson<VectorDocument>(
    'GET',
    `/collections/${encodeURIComponent(collection)}/points/${encodeURIComponent(String(restId))}`
  );
  if (response.error) {
    if (response.error.code === 'NOT_FOUND') {
      return null;
    }
    throw new VelesDBError(response.error.message, response.error.code);
  }
  return response.data ?? null;
}

export async function isEmpty(
  transport: CrudTransport,
  collection: string
): Promise<boolean> {
  const response = await transport.requestJson<{ is_empty: boolean }>(
    'GET',
    `/collections/${encodeURIComponent(collection)}/empty`
  );
  if (response.error) {
    if (response.error.code === 'NOT_FOUND') {
      throw new NotFoundError(`Collection '${collection}'`);
    }
    throw new VelesDBError(response.error.message, response.error.code);
  }
  return response.data?.is_empty ?? true;
}

export async function flush(
  transport: CrudTransport,
  collection: string
): Promise<void> {
  const response = await transport.requestJson(
    'POST',
    `/collections/${encodeURIComponent(collection)}/flush`
  );
  if (response.error) {
    if (response.error.code === 'NOT_FOUND') {
      throw new NotFoundError(`Collection '${collection}'`);
    }
    throw new VelesDBError(response.error.message, response.error.code);
  }
}
