"""Semantic chunking with entity-aware boundaries.

Provides intelligent text chunking that respects entity boundaries
for better knowledge graph construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Callable


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""

    text: str
    start_idx: int
    end_idx: int
    entities: List[str]

    @property
    def length(self) -> int:
        """Length of the chunk text."""
        return len(self.text)


class SemanticChunker:
    """Chunks text with semantic awareness and entity boundaries.

    Uses recursive splitting with overlap, preferring to split at
    sentence boundaries and avoiding splitting entities.

    Args:
        chunk_size: Target chunk size in characters (default: 1000).
        chunk_overlap: Overlap between chunks (default: 200).
        separators: List of separators to try, in order of preference.

    Example:
        >>> chunker = SemanticChunker(chunk_size=500)
        >>> chunks = chunker.chunk("Long document text...")
        >>> for chunk in chunks:
        ...     print(f"Chunk: {chunk.text[:50]}...")
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]

    def chunk(self, text: str, entities: Optional[List[str]] = None) -> List[Chunk]:
        """Split text into semantic chunks.

        Args:
            text: Input text to chunk.
            entities: Optional list of entity names to preserve.

        Returns:
            List of Chunk objects.
        """
        if not text.strip():
            return []

        if len(text) <= self.chunk_size:
            return [
                Chunk(
                    text=text,
                    start_idx=0,
                    end_idx=len(text),
                    entities=self._find_entities(text, entities or []),
                )
            ]

        return self._recursive_split(text, entities or [], 0)

    def _recursive_split(
        self,
        text: str,
        entities: List[str],
        offset: int,
    ) -> List[Chunk]:
        """Recursively split text using separators."""
        if len(text) <= self.chunk_size:
            return [
                Chunk(
                    text=text,
                    start_idx=offset,
                    end_idx=offset + len(text),
                    entities=self._find_entities(text, entities),
                )
            ]

        for separator in self.separators:
            chunks = self._try_split_with_separator(text, separator, entities, offset)
            if chunks:
                return chunks

        mid = len(text) // 2
        return self._recursive_split(text[:mid], entities, offset) + self._recursive_split(
            text[mid:], entities, offset + mid
        )

    def _try_split_with_separator(
        self,
        text: str,
        separator: str,
        entities: List[str],
        offset: int,
    ) -> List[Chunk]:
        """Attempt to split text using a single separator, returning chunks or empty list."""
        splits = text.split(separator)
        if len(splits) == 1:
            return []

        chunks: List[Chunk] = []
        current_chunk = ""
        current_offset = offset

        for split in splits:
            current_chunk, current_offset = self._accumulate_split(
                chunks, current_chunk, current_offset, split,
                separator, entities,
            )

        if current_chunk:
            chunks.append(Chunk(
                text=current_chunk,
                start_idx=current_offset,
                end_idx=current_offset + len(current_chunk),
                entities=self._find_entities(current_chunk, entities),
            ))

        return chunks

    def _accumulate_split(
        self,
        chunks: List[Chunk],
        current_chunk: str,
        current_offset: int,
        split: str,
        separator: str,
        entities: List[str],
    ) -> tuple[str, int]:
        """Add a split fragment to the current chunk, flushing when over size."""
        potential = current_chunk + (separator if current_chunk else "") + split

        if len(potential) <= self.chunk_size:
            return potential, current_offset

        if not current_chunk:
            return split, current_offset

        if self._would_split_entity(current_chunk, entities):
            return potential, current_offset

        chunks.append(Chunk(
            text=current_chunk,
            start_idx=current_offset,
            end_idx=current_offset + len(current_chunk),
            entities=self._find_entities(current_chunk, entities),
        ))
        effective_overlap = min(self.chunk_overlap, len(current_chunk))
        new_offset = current_offset + len(current_chunk) - effective_overlap
        overlap_text = current_chunk[-effective_overlap:] if effective_overlap > 0 else ""
        return overlap_text + separator + split, new_offset

    def _find_entities(self, text: str, entities: List[str]) -> List[str]:
        """Find which entities appear in the text."""
        return [e for e in entities if e.lower() in text.lower()]

    def _would_split_entity(self, text: str, entities: List[str]) -> bool:
        """Check if the chunk boundary would split an entity name."""
        boundary_text = text[-50:] if len(text) > 50 else text

        for entity in entities:
            if entity.lower() in boundary_text.lower():
                entity_start = boundary_text.lower().find(entity.lower())
                entity_end = entity_start + len(entity)
                if entity_end > len(boundary_text) - 5:
                    return True

        return False


def chunk_with_entities(
    text: str,
    extractor_fn: Callable[[str], List[str]],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Chunk]:
    """Convenience function to chunk text with entity extraction.

    Args:
        text: Input text.
        extractor_fn: Function that extracts entity names from text.
        chunk_size: Target chunk size.
        chunk_overlap: Overlap between chunks.

    Returns:
        List of Chunk objects with entities populated.
    """
    entities = extractor_fn(text)
    chunker = SemanticChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.chunk(text, entities)
