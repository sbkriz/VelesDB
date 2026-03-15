/**
 * Search Backend operations for VelesDB REST API.
 *
 * Extracted from rest.ts to keep file size manageable.
 * Implements: search, searchBatch, textSearch, hybridSearch,
 * multiQuerySearch, and searchIds.
 */

import type {
  SearchOptions,
  SearchResult,
  MultiQuerySearchOptions,
  SparseVector,
} from '../types';
import { NotFoundError, VelesDBError } from '../types';

/** Batch search response structure (mirrors rest.ts private type). */
interface BatchSearchResponse {
  results: Array<{ results: SearchResult[] }>;
}

/** Minimal transport interface for search operations. */
export interface SearchTransport {
  requestJson<T>(
    method: string,
    path: string,
    body?: unknown
  ): Promise<{ data?: T; error?: { code: string; message: string } }>;

  sparseToRest(sv: SparseVector): Record<string, number>;
}

export async function search(
  transport: SearchTransport,
  collection: string,
  query: number[] | Float32Array,
  options?: SearchOptions
): Promise<SearchResult[]> {
  const queryVector = query instanceof Float32Array ? Array.from(query) : query;

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const body: Record<string, any> = {
    vector: queryVector,
    top_k: options?.k ?? 10,
    filter: options?.filter,
    include_vectors: options?.includeVectors ?? false,
  };

  if (options?.sparseVector) {
    body.sparse_vector = transport.sparseToRest(options.sparseVector);
  }

  const response = await transport.requestJson<{ results: SearchResult[] }>(
    'POST',
    `/collections/${encodeURIComponent(collection)}/search`,
    body
  );

  if (response.error) {
    if (response.error.code === 'NOT_FOUND') {
      throw new NotFoundError(`Collection '${collection}'`);
    }
    throw new VelesDBError(response.error.message, response.error.code);
  }

  return response.data?.results ?? [];
}

export async function searchBatch(
  transport: SearchTransport,
  collection: string,
  searches: Array<{
    vector: number[] | Float32Array;
    k?: number;
    filter?: Record<string, unknown>;
  }>
): Promise<SearchResult[][]> {
  const formattedSearches = searches.map(s => ({
    vector: s.vector instanceof Float32Array ? Array.from(s.vector) : s.vector,
    top_k: s.k ?? 10,
    filter: s.filter,
  }));

  const response = await transport.requestJson<BatchSearchResponse>(
    'POST',
    `/collections/${encodeURIComponent(collection)}/search/batch`,
    { searches: formattedSearches }
  );

  if (response.error) {
    if (response.error.code === 'NOT_FOUND') {
      throw new NotFoundError(`Collection '${collection}'`);
    }
    throw new VelesDBError(response.error.message, response.error.code);
  }

  return response.data?.results.map(r => r.results) ?? [];
}

export async function textSearch(
  transport: SearchTransport,
  collection: string,
  query: string,
  options?: { k?: number; filter?: Record<string, unknown> }
): Promise<SearchResult[]> {
  const response = await transport.requestJson<{ results: SearchResult[] }>(
    'POST',
    `/collections/${encodeURIComponent(collection)}/search/text`,
    {
      query,
      top_k: options?.k ?? 10,
      filter: options?.filter,
    }
  );

  if (response.error) {
    if (response.error.code === 'NOT_FOUND') {
      throw new NotFoundError(`Collection '${collection}'`);
    }
    throw new VelesDBError(response.error.message, response.error.code);
  }

  return response.data?.results ?? [];
}

export async function hybridSearch(
  transport: SearchTransport,
  collection: string,
  vector: number[] | Float32Array,
  textQuery: string,
  options?: { k?: number; vectorWeight?: number; filter?: Record<string, unknown> }
): Promise<SearchResult[]> {
  const queryVector = vector instanceof Float32Array ? Array.from(vector) : vector;

  const response = await transport.requestJson<{ results: SearchResult[] }>(
    'POST',
    `/collections/${encodeURIComponent(collection)}/search/hybrid`,
    {
      vector: queryVector,
      query: textQuery,
      top_k: options?.k ?? 10,
      vector_weight: options?.vectorWeight ?? 0.5,
      filter: options?.filter,
    }
  );

  if (response.error) {
    if (response.error.code === 'NOT_FOUND') {
      throw new NotFoundError(`Collection '${collection}'`);
    }
    throw new VelesDBError(response.error.message, response.error.code);
  }

  return response.data?.results ?? [];
}

export async function multiQuerySearch(
  transport: SearchTransport,
  collection: string,
  vectors: Array<number[] | Float32Array>,
  options?: MultiQuerySearchOptions
): Promise<SearchResult[]> {
  const formattedVectors = vectors.map(v =>
    v instanceof Float32Array ? Array.from(v) : v
  );

  const response = await transport.requestJson<{ results: SearchResult[] }>(
    'POST',
    `/collections/${encodeURIComponent(collection)}/search/multi`,
    {
      vectors: formattedVectors,
      top_k: options?.k ?? 10,
      strategy: options?.fusion ?? 'rrf',
      rrf_k: options?.fusionParams?.k ?? 60,
      avg_weight: options?.fusionParams?.avgWeight,
      max_weight: options?.fusionParams?.maxWeight,
      hit_weight: options?.fusionParams?.hitWeight,
      filter: options?.filter,
    }
  );

  if (response.error) {
    if (response.error.code === 'NOT_FOUND') {
      throw new NotFoundError(`Collection '${collection}'`);
    }
    throw new VelesDBError(response.error.message, response.error.code);
  }

  return response.data?.results ?? [];
}

export async function searchIds(
  transport: SearchTransport,
  collection: string,
  query: number[] | Float32Array,
  options?: SearchOptions
): Promise<Array<{ id: number; score: number }>> {
  const queryVector = query instanceof Float32Array ? Array.from(query) : query;

  const response = await transport.requestJson<{
    results: Array<{ id: number; score: number }>;
  }>(
    'POST',
    `/collections/${encodeURIComponent(collection)}/search/ids`,
    {
      vector: queryVector,
      top_k: options?.k ?? 10,
      filter: options?.filter,
    }
  );

  if (response.error) {
    if (response.error.code === 'NOT_FOUND') {
      throw new NotFoundError(`Collection '${collection}'`);
    }
    throw new VelesDBError(response.error.message, response.error.code);
  }

  return response.data?.results ?? [];
}
