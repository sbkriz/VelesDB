/**
 * Agent Memory facade for VelesDB.
 *
 * Provides semantic, episodic, and procedural memory abstractions
 * on top of the VelesDB backend interface.
 */

import type {
  IVelesDBBackend,
  AgentMemoryConfig,
  SemanticEntry,
  EpisodicEvent,
  ProceduralPattern,
  SearchResult,
} from './types';

/**
 * Agent Memory client for semantic, episodic, and procedural memory
 */
export class AgentMemoryClient {
  constructor(
    private readonly backend: IVelesDBBackend,
    private readonly config?: AgentMemoryConfig
  ) {}

  /** Configured embedding dimension (default: 384) */
  get dimension(): number {
    return this.config?.dimension ?? 384;
  }

  /** Store a semantic fact */
  async storeFact(collection: string, entry: SemanticEntry): Promise<void> {
    return this.backend.storeSemanticFact(collection, entry);
  }

  /** Search semantic memory */
  async searchFacts(collection: string, embedding: number[], k = 5): Promise<SearchResult[]> {
    return this.backend.searchSemanticMemory(collection, embedding, k);
  }

  /** Record an episodic event */
  async recordEvent(collection: string, event: EpisodicEvent): Promise<void> {
    return this.backend.recordEpisodicEvent(collection, event);
  }

  /** Recall episodic events */
  async recallEvents(collection: string, embedding: number[], k = 5): Promise<SearchResult[]> {
    return this.backend.recallEpisodicEvents(collection, embedding, k);
  }

  /** Store a procedural pattern */
  async learnProcedure(collection: string, pattern: ProceduralPattern): Promise<void> {
    return this.backend.storeProceduralPattern(collection, pattern);
  }

  /** Match procedural patterns */
  async recallProcedures(collection: string, embedding: number[], k = 5): Promise<SearchResult[]> {
    return this.backend.matchProceduralPatterns(collection, embedding, k);
  }
}
