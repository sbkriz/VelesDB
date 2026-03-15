/**
 * Graph Backend operations for VelesDB REST API.
 *
 * Extracted from rest.ts to keep file size manageable.
 * Implements: addEdge, getEdges, traverseGraph, getNodeDegree,
 * createGraphCollection.
 */

import type {
  AddEdgeRequest,
  GetEdgesOptions,
  GraphEdge,
  TraverseRequest,
  TraverseResponse,
  DegreeResponse,
  GraphCollectionConfig,
} from '../types';
import { NotFoundError, VelesDBError } from '../types';

/** Minimal transport interface for graph operations. */
export interface GraphTransport {
  requestJson<T>(
    method: string,
    path: string,
    body?: unknown
  ): Promise<{ data?: T; error?: { code: string; message: string } }>;
}

export async function addEdge(
  transport: GraphTransport,
  collection: string,
  edge: AddEdgeRequest
): Promise<void> {
  const response = await transport.requestJson(
    'POST',
    `/collections/${encodeURIComponent(collection)}/graph/edges`,
    {
      id: edge.id,
      source: edge.source,
      target: edge.target,
      label: edge.label,
      properties: edge.properties ?? {},
    }
  );

  if (response.error) {
    if (response.error.code === 'NOT_FOUND') {
      throw new NotFoundError(`Collection '${collection}'`);
    }
    throw new VelesDBError(response.error.message, response.error.code);
  }
}

export async function getEdges(
  transport: GraphTransport,
  collection: string,
  options?: GetEdgesOptions
): Promise<GraphEdge[]> {
  const queryParams = options?.label ? `?label=${encodeURIComponent(options.label)}` : '';

  const response = await transport.requestJson<{ edges: GraphEdge[]; count: number }>(
    'GET',
    `/collections/${encodeURIComponent(collection)}/graph/edges${queryParams}`
  );

  if (response.error) {
    if (response.error.code === 'NOT_FOUND') {
      throw new NotFoundError(`Collection '${collection}'`);
    }
    throw new VelesDBError(response.error.message, response.error.code);
  }

  return response.data?.edges ?? [];
}

export async function traverseGraph(
  transport: GraphTransport,
  collection: string,
  request: TraverseRequest
): Promise<TraverseResponse> {
  const response = await transport.requestJson<{
    results: Array<{ target_id: number; depth: number; path: number[] }>;
    next_cursor: string | null;
    has_more: boolean;
    stats: { visited: number; depth_reached: number };
  }>(
    'POST',
    `/collections/${encodeURIComponent(collection)}/graph/traverse`,
    {
      source: request.source,
      strategy: request.strategy ?? 'bfs',
      max_depth: request.maxDepth ?? 3,
      limit: request.limit ?? 100,
      cursor: request.cursor,
      rel_types: request.relTypes ?? [],
    }
  );

  if (response.error) {
    if (response.error.code === 'NOT_FOUND') {
      throw new NotFoundError(`Collection '${collection}'`);
    }
    throw new VelesDBError(response.error.message, response.error.code);
  }

  const data = response.data!;
  return {
    results: data.results.map(r => ({
      targetId: r.target_id,
      depth: r.depth,
      path: r.path,
    })),
    nextCursor: data.next_cursor ?? undefined,
    hasMore: data.has_more,
    stats: {
      visited: data.stats.visited,
      depthReached: data.stats.depth_reached,
    },
  };
}

export async function getNodeDegree(
  transport: GraphTransport,
  collection: string,
  nodeId: number
): Promise<DegreeResponse> {
  const response = await transport.requestJson<{ in_degree: number; out_degree: number }>(
    'GET',
    `/collections/${encodeURIComponent(collection)}/graph/nodes/${nodeId}/degree`
  );

  if (response.error) {
    if (response.error.code === 'NOT_FOUND') {
      throw new NotFoundError(`Collection '${collection}'`);
    }
    throw new VelesDBError(response.error.message, response.error.code);
  }

  return {
    inDegree: response.data?.in_degree ?? 0,
    outDegree: response.data?.out_degree ?? 0,
  };
}

export async function createGraphCollection(
  transport: GraphTransport,
  name: string,
  config?: GraphCollectionConfig
): Promise<void> {
  const response = await transport.requestJson('POST', '/collections', {
    name,
    collection_type: 'graph',
    dimension: config?.dimension,
    metric: config?.metric ?? 'cosine',
    schema_mode: config?.schemaMode ?? 'schemaless',
  });

  if (response.error) {
    throw new VelesDBError(response.error.message, response.error.code);
  }
}
