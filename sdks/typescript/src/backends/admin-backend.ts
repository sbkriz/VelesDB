/**
 * Admin Backend operations for VelesDB REST API.
 *
 * Extracted from rest.ts to keep file size manageable.
 * Implements: getCollectionStats, analyzeCollection, getCollectionConfig.
 */

import type {
  CollectionStatsResponse,
  CollectionConfigResponse,
} from '../types';
import { NotFoundError, VelesDBError } from '../types';

/** Minimal transport interface for admin operations. */
export interface AdminTransport {
  requestJson<T>(
    method: string,
    path: string,
    body?: unknown
  ): Promise<{ data?: T; error?: { code: string; message: string } }>;
}

/** Raw stats shape returned by the REST API. */
interface StatsApiResponse {
  total_points: number;
  total_size_bytes: number;
  row_count: number;
  deleted_count: number;
  avg_row_size_bytes: number;
  payload_size_bytes: number;
  last_analyzed_epoch_ms: number;
}

export function mapStatsResponse(data: StatsApiResponse): CollectionStatsResponse {
  return {
    totalPoints: data.total_points,
    totalSizeBytes: data.total_size_bytes,
    rowCount: data.row_count,
    deletedCount: data.deleted_count,
    avgRowSizeBytes: data.avg_row_size_bytes,
    payloadSizeBytes: data.payload_size_bytes,
    lastAnalyzedEpochMs: data.last_analyzed_epoch_ms,
  };
}

export async function getCollectionStats(
  transport: AdminTransport,
  collection: string
): Promise<CollectionStatsResponse | null> {
  const response = await transport.requestJson<StatsApiResponse>(
    'GET',
    `/collections/${encodeURIComponent(collection)}/stats`
  );

  if (response.error) {
    if (response.error.code === 'NOT_FOUND') {
      return null;
    }
    throw new VelesDBError(response.error.message, response.error.code);
  }

  return mapStatsResponse(response.data!);
}

export async function analyzeCollection(
  transport: AdminTransport,
  collection: string
): Promise<CollectionStatsResponse> {
  const response = await transport.requestJson<StatsApiResponse>(
    'POST',
    `/collections/${encodeURIComponent(collection)}/analyze`
  );

  if (response.error) {
    if (response.error.code === 'NOT_FOUND') {
      throw new NotFoundError(`Collection '${collection}'`);
    }
    throw new VelesDBError(response.error.message, response.error.code);
  }

  return mapStatsResponse(response.data!);
}

export async function getCollectionConfig(
  transport: AdminTransport,
  collection: string
): Promise<CollectionConfigResponse> {
  const response = await transport.requestJson<{
    name: string;
    dimension: number;
    metric: string;
    storage_mode: string;
    point_count: number;
    metadata_only: boolean;
    graph_schema?: Record<string, unknown>;
    embedding_dimension?: number;
  }>('GET', `/collections/${encodeURIComponent(collection)}/config`);

  if (response.error) {
    if (response.error.code === 'NOT_FOUND') {
      throw new NotFoundError(`Collection '${collection}'`);
    }
    throw new VelesDBError(response.error.message, response.error.code);
  }

  const data = response.data!;
  return {
    name: data.name,
    dimension: data.dimension,
    metric: data.metric,
    storageMode: data.storage_mode,
    pointCount: data.point_count,
    metadataOnly: data.metadata_only,
    graphSchema: data.graph_schema,
    embeddingDimension: data.embedding_dimension,
  };
}
