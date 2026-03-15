/**
 * Index Backend operations for VelesDB REST API.
 *
 * Extracted from rest.ts to keep file size manageable.
 * Implements: createIndex, listIndexes, hasIndex, dropIndex.
 */

import type {
  CreateIndexOptions,
  IndexInfo,
} from '../types';
import { NotFoundError, VelesDBError } from '../types';

/** Minimal transport interface for index operations. */
export interface IndexTransport {
  requestJson<T>(
    method: string,
    path: string,
    body?: unknown
  ): Promise<{ data?: T; error?: { code: string; message: string } }>;
}

export async function createIndex(
  transport: IndexTransport,
  collection: string,
  options: CreateIndexOptions
): Promise<void> {
  const response = await transport.requestJson(
    'POST',
    `/collections/${encodeURIComponent(collection)}/indexes`,
    {
      label: options.label,
      property: options.property,
      index_type: options.indexType ?? 'hash',
    }
  );

  if (response.error) {
    if (response.error.code === 'NOT_FOUND') {
      throw new NotFoundError(`Collection '${collection}'`);
    }
    throw new VelesDBError(response.error.message, response.error.code);
  }
}

export async function listIndexes(
  transport: IndexTransport,
  collection: string
): Promise<IndexInfo[]> {
  const response = await transport.requestJson<{ indexes: Array<{
    label: string;
    property: string;
    index_type: string;
    cardinality: number;
    memory_bytes: number;
  }>; total: number }>(
    'GET',
    `/collections/${encodeURIComponent(collection)}/indexes`
  );

  if (response.error) {
    if (response.error.code === 'NOT_FOUND') {
      throw new NotFoundError(`Collection '${collection}'`);
    }
    throw new VelesDBError(response.error.message, response.error.code);
  }

  return (response.data?.indexes ?? []).map(idx => ({
    label: idx.label,
    property: idx.property,
    indexType: idx.index_type as 'hash' | 'range',
    cardinality: idx.cardinality,
    memoryBytes: idx.memory_bytes,
  }));
}

export async function hasIndex(
  transport: IndexTransport,
  collection: string,
  label: string,
  property: string
): Promise<boolean> {
  const indexes = await listIndexes(transport, collection);
  return indexes.some(idx => idx.label === label && idx.property === property);
}

export async function dropIndex(
  transport: IndexTransport,
  collection: string,
  label: string,
  property: string
): Promise<boolean> {
  const response = await transport.requestJson<{ dropped: boolean }>(
    'DELETE',
    `/collections/${encodeURIComponent(collection)}/indexes/${encodeURIComponent(label)}/${encodeURIComponent(property)}`
  );

  if (response.error) {
    if (response.error.code === 'NOT_FOUND') {
      return false;  // Index didn't exist
    }
    throw new VelesDBError(response.error.message, response.error.code);
  }

  // BUG-2 FIX: Success without error = index was dropped
  // API may return 200/204 without body, so default to true on success
  return response.data?.dropped ?? true;
}
