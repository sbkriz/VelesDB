"""
Regression tests for GitHub issues #356 and #357.

- #357: __init__.py imports PyGraphCollection/PyGraphSchema but native
        module exports GraphCollection/GraphSchema
- #356: agent_memory() fails with VELES-031 on same-process re-entrant open

Run with: pytest tests/test_issue_fixes.py -v
"""

import tempfile
import shutil

import pytest

try:
    from velesdb import Database, PyGraphCollection, PyGraphSchema
    from velesdb.velesdb import GraphCollection, GraphSchema  # native names

    VELESDB_AVAILABLE = True
except ImportError:
    VELESDB_AVAILABLE = False
    Database = None

pytestmark = pytest.mark.skipif(
    not VELESDB_AVAILABLE,
    reason="VelesDB Python bindings not installed. Run: maturin develop",
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    temp_dir = tempfile.mkdtemp()
    db = Database(temp_dir)
    yield db
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestIssue357GraphImports:
    """Regression tests for #357: PyGraphCollection/PyGraphSchema import fix."""

    def test_pygraphcollection_is_importable(self):
        """PyGraphCollection must be importable from velesdb."""
        if not VELESDB_AVAILABLE:
            pytest.skip("velesdb not available")
        assert PyGraphCollection is not None

    def test_pygraphschema_is_importable(self):
        """PyGraphSchema must be importable from velesdb."""
        if not VELESDB_AVAILABLE:
            pytest.skip("velesdb not available")
        assert PyGraphSchema is not None

    def test_pygraphcollection_is_same_as_native(self):
        """PyGraphCollection alias must point to the native GraphCollection."""
        if not VELESDB_AVAILABLE:
            pytest.skip("velesdb not available")
        assert PyGraphCollection is GraphCollection

    def test_pygraphschema_is_same_as_native(self):
        """PyGraphSchema alias must point to the native GraphSchema."""
        if not VELESDB_AVAILABLE:
            pytest.skip("velesdb not available")
        assert PyGraphSchema is GraphSchema

    def test_graphschema_schemaless(self):
        """GraphSchema.schemaless() must work through the alias."""
        if not VELESDB_AVAILABLE:
            pytest.skip("velesdb not available")
        schema = PyGraphSchema.schemaless()
        assert schema is not None


class TestIssue356AgentMemoryLock:
    """Regression tests for #356: agent_memory() VELES-031 re-entrant lock."""

    def test_agent_memory_on_open_db(self, temp_db):
        """agent_memory() must succeed on an already-opened Database."""
        memory = temp_db.agent_memory(dimension=4)
        assert memory is not None
        assert memory.dimension == 4

    def test_agent_memory_default_dimension(self, temp_db):
        """agent_memory() with default dimension must not raise."""
        memory = temp_db.agent_memory()
        assert memory is not None

    def test_agent_memory_semantic_store_and_query(self, temp_db):
        """Full semantic memory round-trip through agent_memory()."""
        memory = temp_db.agent_memory(dimension=4)
        embedding = [0.1, 0.2, 0.3, 0.4]
        memory.semantic.store(1, "Paris is the capital of France", embedding)
        results = memory.semantic.query(embedding, top_k=1)
        assert len(results) >= 1
        assert results[0]["content"] == "Paris is the capital of France"

    def test_multiple_agent_memory_calls(self, temp_db):
        """Calling agent_memory() multiple times must not fail."""
        mem1 = temp_db.agent_memory(dimension=4)
        mem2 = temp_db.agent_memory(dimension=4)
        assert mem1 is not None
        assert mem2 is not None
