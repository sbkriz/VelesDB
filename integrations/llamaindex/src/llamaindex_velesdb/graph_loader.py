"""VelesDB Graph Loader for LlamaIndex.

Load entities and relations into VelesDB's Knowledge Graph from LlamaIndex nodes.

Example:
    >>> from llamaindex_velesdb import VelesDBVectorStore, GraphLoader
    >>> from llama_index.core.schema import TextNode
    >>>
    >>> vector_store = VelesDBVectorStore(path="./db")
    >>> loader = GraphLoader(vector_store)
    >>>
    >>> # Add nodes and edges
    >>> loader.add_node(id=1, label="PERSON", metadata={"name": "John"})
    >>> loader.add_edge(id=1, source=1, target=2, label="KNOWS")
    >>>
    >>> # Query edges
    >>> edges = loader.get_edges(label="KNOWS")
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING
import hashlib
import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from llamaindex_velesdb.vectorstore import VelesDBVectorStore


def _generate_id(name: str, entity_type: str) -> int:
    """Generate a deterministic ID from entity name and type."""
    hash_input = f"{entity_type}:{name}".encode("utf-8")
    return int(hashlib.sha256(hash_input).hexdigest()[:15], 16)


class GraphLoader:
    """Load entities and relations into VelesDB's Knowledge Graph.

    Provides methods to add nodes and edges to a VelesDB collection's
    graph layer, enabling Knowledge Graph construction from LlamaIndex data.

    Args:
        vector_store: VelesDBVectorStore instance.

    Example:
        >>> loader = GraphLoader(vector_store)
        >>> loader.add_node(id=1, label="PERSON", metadata={"name": "John"})
        >>> loader.add_edge(id=1, source=1, target=2, label="KNOWS")
        >>> edges = loader.get_edges(label="KNOWS")
    """

    def __init__(self, vector_store: "VelesDBVectorStore") -> None:
        """Initialize GraphLoader with a VelesDBVectorStore."""
        self._vector_store = vector_store
        self._default_dimension = 384
        self._edges: List[Dict[str, Any]] = []

        self._graph_store = getattr(vector_store, "_graph_store", None)
        if self._graph_store is None:
            try:
                import velesdb

                self._graph_store = velesdb.GraphStore()
            except Exception:
                self._graph_store = None

    def add_node(
        self,
        id: int,
        label: str,
        metadata: Optional[Dict[str, Any]] = None,
        vector: Optional[List[float]] = None,
    ) -> None:
        """Add a node to the graph.

        Args:
            id: Unique node ID.
            label: Node label (type, e.g., "PERSON", "DOCUMENT").
            metadata: Optional node properties.
            vector: Optional embedding vector for the node.

        Example:
            >>> loader.add_node(
            ...     id=1,
            ...     label="PERSON",
            ...     metadata={"name": "John", "age": 30}
            ... )
        """
        collection = self._get_collection(dimension=len(vector) if vector else None)

        payload = {"label": label, **(metadata or {})}

        if vector is not None:
            collection.upsert([{
                "id": id,
                "vector": vector,
                "payload": payload,
            }])
        else:
            info = collection.info()
            if info.get("metadata_only", False):
                collection.upsert_metadata([{
                    "id": id,
                    "payload": payload,
                }])
            else:
                raise ValueError(
                    "Collection requires vectors for node insertion. "
                    "Provide `vector` or initialize a metadata-only collection."
                )

    def add_edge(
        self,
        id: int,
        source: int,
        target: int,
        label: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add an edge to the graph.

        Args:
            id: Unique edge ID.
            source: Source node ID.
            target: Target node ID.
            label: Edge label (relationship type, e.g., "KNOWS", "WORKS_AT").
            metadata: Optional edge properties.

        Example:
            >>> loader.add_edge(
            ...     id=1,
            ...     source=100,
            ...     target=200,
            ...     label="KNOWS",
            ...     metadata={"since": "2024-01-01"}
            ... )
        """
        edge = {
            "id": id,
            "source": source,
            "target": target,
            "label": label,
            "properties": metadata or {},
        }
        self._edges.append(edge)
        if self._graph_store is not None:
            self._graph_store.add_edge(edge)

    def get_edges(
        self,
        label: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get edges from the graph.

        Args:
            label: Optional filter by edge label.

        Returns:
            List of edge dictionaries with id, source, target, label, properties.

        Example:
            >>> edges = loader.get_edges(label="KNOWS")
            >>> for edge in edges:
            ...     print(f"{edge['source']} -> {edge['target']}")
        """
        if label:
            edges = [e for e in self._edges if e.get("label") == label]
        else:
            edges = list(self._edges)

        return [
            {
                "id": e.get("id", 0),
                "source": e.get("source", 0),
                "target": e.get("target", 0),
                "label": e.get("label", ""),
                "properties": e.get("properties", {}),
            }
            for e in edges
        ]

    def load_from_nodes(
        self,
        nodes: List[Any],
        node_label: str = "DOCUMENT",
        extract_relations: bool = False,
    ) -> Dict[str, int]:
        """Load LlamaIndex nodes as graph nodes.

        Args:
            nodes: List of LlamaIndex BaseNode objects.
            node_label: Label to assign to all nodes. Defaults to "DOCUMENT".
            extract_relations: Whether to extract relations (requires NLP).

        Returns:
            Dictionary with counts: {"nodes": n, "edges": m}.

        Example:
            >>> from llama_index.core.schema import TextNode
            >>> nodes = [TextNode(text="Hello", id_="1")]
            >>> counts = loader.load_from_nodes(nodes)
        """
        nodes_added = 0

        for node in nodes:
            # Use deterministic SHA256-based ID (not Python hash() which is randomized)
            node_id = _generate_id(node.node_id, node_label)

            content = node.get_content() if hasattr(node, "get_content") else None
            metadata = {
                "node_id": node.node_id,
                "text_preview": content[:200] if content else "",
            }

            if hasattr(node, "metadata") and node.metadata:
                metadata.update({
                    k: v for k, v in node.metadata.items()
                    if isinstance(v, (str, int, float, bool))
                })

            try:
                self.add_node(
                    id=node_id,
                    label=node_label,
                    metadata=metadata,
                )
                nodes_added += 1
            except Exception as e:
                logger.warning(f"Failed to add node {node.node_id}: {e}")

        return {"nodes": nodes_added, "edges": 0}

    def _get_collection(self, dimension: Optional[int] = None):
        """Get or lazily initialize the underlying collection."""
        collection = getattr(self._vector_store, "_collection", None)
        if collection is not None:
            return collection

        get_collection = getattr(self._vector_store, "_get_collection", None)
        if get_collection is None:
            raise ValueError(
                "Vector store does not expose _get_collection; cannot initialize collection."
            )

        resolved_dimension = dimension or getattr(self._vector_store, "_dimension", None)
        if resolved_dimension is None:
            resolved_dimension = self._default_dimension

        collection = get_collection(int(resolved_dimension))
        if collection is None:
            raise ValueError("Failed to initialize collection from vector store")
        return collection
