/**
 * Streaming Backend operations for VelesDB REST API.
 *
 * Extracted from rest.ts to keep file size manageable.
 * Implements: trainPq, streamInsert.
 */

import type {
  VectorDocument,
  PqTrainOptions,
  SparseVector,
  RestPointId,
} from '../types';
import { BackpressureError, ConnectionError, VelesDBError } from '../types';

/** Minimal transport interface for streaming operations. */
export interface StreamingTransport {
  requestJson<T>(
    method: string,
    path: string,
    body?: unknown
  ): Promise<{ data?: T; error?: { code: string; message: string } }>;

  readonly baseUrl: string;
  readonly apiKey: string | undefined;
  readonly timeout: number;

  parseRestPointId(id: string | number): RestPointId;
  sparseVectorToRestFormat(sv: SparseVector): Record<string, number>;
  mapStatusToErrorCode(status: number): string;
  extractErrorPayload(data: unknown): { code?: string; message?: string };
}

export async function trainPq(
  transport: StreamingTransport,
  collection: string,
  options?: PqTrainOptions
): Promise<string> {
  const m = options?.m ?? 8;
  const k = options?.k ?? 256;
  const withClause = options?.opq
    ? `WITH (m=${m}, k=${k}, opq=true)`
    : `WITH (m=${m}, k=${k})`;
  const queryString = `TRAIN QUANTIZER ON ${collection} ${withClause}`;

  const response = await transport.requestJson<{ message: string }>(
    'POST',
    '/query',
    { query: queryString }
  );

  if (response.error) {
    throw new VelesDBError(response.error.message, response.error.code);
  }

  return response.data?.message ?? 'PQ training initiated';
}

export async function streamInsert(
  transport: StreamingTransport,
  collection: string,
  docs: VectorDocument[]
): Promise<void> {
  for (const doc of docs) {
    const restId = transport.parseRestPointId(doc.id);
    const vector = doc.vector instanceof Float32Array
      ? Array.from(doc.vector)
      : doc.vector;

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const body: Record<string, any> = {
      id: restId,
      vector,
      payload: doc.payload,
    };

    if (doc.sparseVector) {
      body.sparse_vector = transport.sparseVectorToRestFormat(doc.sparseVector);
    }

    const url = `${transport.baseUrl}/collections/${encodeURIComponent(collection)}/stream/insert`;
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };

    if (transport.apiKey) {
      headers['Authorization'] = `Bearer ${transport.apiKey}`;
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), transport.timeout);

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers,
        body: JSON.stringify(body),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (response.status === 429) {
        throw new BackpressureError();
      }

      if (!response.ok && response.status !== 202) {
        const data = await response.json().catch(() => ({}));
        const errorPayload = transport.extractErrorPayload(data);
        throw new VelesDBError(
          errorPayload.message ?? `HTTP ${response.status}`,
          errorPayload.code ?? transport.mapStatusToErrorCode(response.status)
        );
      }
    } catch (error) {
      clearTimeout(timeoutId);

      if (error instanceof BackpressureError || error instanceof VelesDBError) {
        throw error;
      }

      if (error instanceof Error && error.name === 'AbortError') {
        throw new ConnectionError('Request timeout');
      }

      throw new ConnectionError(
        `Stream insert failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
        error instanceof Error ? error : undefined
      );
    }
  }
}
