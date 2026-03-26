"""
Regression tests for GitHub issues #356, #357, #412, and #413.

- #357: __init__.py imports PyGraphCollection/PyGraphSchema but native
        module exports GraphCollection/GraphSchema
- #356: agent_memory() fails with VELES-031 on same-process re-entrant open
- #412: bool→int silent conversion in python_to_json (True stored as 1)
- #413: unsupported payload types silently dropped instead of raising

Run with: pytest tests/test_issue_fixes.py -v
"""

import datetime

import pytest

from conftest import _SKIP_NO_BINDINGS

# temp_db fixture is provided by conftest.py auto-discovery.
pytestmark = _SKIP_NO_BINDINGS

try:
    from velesdb import PyGraphCollection, PyGraphSchema
    from velesdb.velesdb import GraphCollection, GraphSchema  # native names
except (ImportError, AttributeError):
    PyGraphCollection = None  # type: ignore[assignment]
    PyGraphSchema = None  # type: ignore[assignment]
    GraphCollection = None  # type: ignore[assignment]
    GraphSchema = None  # type: ignore[assignment]


class TestIssue357GraphImports:
    """Regression tests for #357: PyGraphCollection/PyGraphSchema import fix.

    The two *_is_importable tests were removed: they only asserted `is not None`,
    which is strictly weaker than the identity tests below.  A passing identity
    test implies importability.
    """

    def test_pygraphcollection_is_same_as_native(self):
        """PyGraphCollection alias must point to the native GraphCollection."""
        assert PyGraphCollection is GraphCollection

    def test_pygraphschema_is_same_as_native(self):
        """PyGraphSchema alias must point to the native GraphSchema."""
        assert PyGraphSchema is GraphSchema

    def test_graphschema_schemaless(self):
        """GraphSchema.schemaless() must work through the alias."""
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
        assert results[0]["score"] > 0.9

    def test_multiple_agent_memory_calls(self, temp_db):
        """Calling agent_memory() multiple times must not fail.

        Also verifies that two handles on the same DB share state (store via
        mem1, query via mem2) and that both report the same dimension.
        """
        mem1 = temp_db.agent_memory(dimension=4)
        mem2 = temp_db.agent_memory(dimension=4)
        assert mem1 is not None
        assert mem2 is not None
        # Dimension must be consistent across handles.
        assert mem1.dimension == mem2.dimension == 4
        # Data written through mem1 must be visible through mem2.
        embedding = [0.1, 0.2, 0.3, 0.4]
        mem1.semantic.store(99, "cross-handle fact", embedding)
        results = mem2.semantic.query(embedding, top_k=1)
        assert len(results) >= 1
        assert results[0]["content"] == "cross-handle fact"


class TestIssue412BoolIntConversion:
    """Regression tests for #412: bool→int silent conversion in python_to_json.

    Python bool is a subclass of int. If i64 extraction is checked before bool,
    True is stored as 1 and False as 0 instead of proper JSON booleans.
    """

    @pytest.mark.parametrize(
        "flag, expected",
        [
            (True, True),
            (False, False),
        ],
        ids=["true", "false"],
    )
    def test_bool_roundtrip(self, temp_db, flag, expected):
        """bool values must round-trip as bool, not as int."""
        col = temp_db.create_collection(f"bool_{flag}_test", dimension=4)
        col.upsert([{
            "id": 1,
            "vector": [1.0, 0.0, 0.0, 0.0],
            "payload": {"flag": flag},
        }])
        payload = col.get([1])[0]["payload"]
        assert payload["flag"] is expected, (
            f"Expected {expected!r} (bool), got {payload['flag']!r}"
            f" ({type(payload['flag']).__name__})"
        )
        assert isinstance(payload["flag"], bool)

    def test_bool_and_int_coexist(self, temp_db):
        """Booleans and integers in the same payload must preserve their types.

        Includes `one=1` to cover the True/1 ambiguity from the integer side:
        the int literal 1 must NOT be returned as bool True.
        """
        col = temp_db.create_collection("bool_int_test", dimension=4)
        col.upsert([{
            "id": 1,
            "vector": [1.0, 0.0, 0.0, 0.0],
            "payload": {
                "active": True,
                "disabled": False,
                "count": 42,
                "zero": 0,
                "one": 1,
            },
        }])
        p = col.get([1])[0]["payload"]
        assert p["active"] is True
        assert p["disabled"] is False
        assert p["count"] == 42
        assert isinstance(p["count"], int) and not isinstance(p["count"], bool)
        assert p["zero"] == 0
        assert isinstance(p["zero"], int) and not isinstance(p["zero"], bool)
        assert p["one"] == 1
        assert isinstance(p["one"], int) and not isinstance(p["one"], bool)

    def test_bool_in_nested_payload(self, temp_db):
        """Booleans nested in dicts and lists must preserve their type."""
        col = temp_db.create_collection("bool_nested_test", dimension=4)
        col.upsert([{
            "id": 1,
            "vector": [1.0, 0.0, 0.0, 0.0],
            "payload": {
                "nested": {"flag": True},
                "list": [True, False, 1, 0],
            },
        }])
        p = col.get([1])[0]["payload"]
        assert p["nested"]["flag"] is True
        assert p["list"][0] is True
        assert p["list"][1] is False
        assert p["list"][2] == 1
        assert isinstance(p["list"][2], int) and not isinstance(p["list"][2], bool)
        # list[3] is the int 0 — must not come back as bool False
        assert p["list"][3] == 0
        assert isinstance(p["list"][3], int) and not isinstance(p["list"][3], bool)


class TestIssue413SilentDataLoss:
    """Regression tests for #413: unsupported types in payload must raise, not drop."""

    def test_unsupported_type_raises_valueerror(self, temp_db):
        """Unsupported types in payload must raise ValueError, not silently drop.

        The Rust side raises PyValueError::new_err(...), which maps to Python
        ValueError — not TypeError.  The union was overly broad.
        """
        col = temp_db.create_collection("data_loss_test", dimension=4)
        with pytest.raises(ValueError):
            col.upsert([{
                "id": 1,
                "vector": [1.0, 0.0, 0.0, 0.0],
                "payload": {"ts": datetime.datetime.now()},
            }])

    def test_unsupported_type_in_list_raises(self, temp_db):
        """Unsupported types inside lists must also raise ValueError."""
        col = temp_db.create_collection("list_loss_test", dimension=4)
        with pytest.raises(ValueError):
            col.upsert([{
                "id": 1,
                "vector": [1.0, 0.0, 0.0, 0.0],
                "payload": {"items": [1, "ok", datetime.datetime.now()]},
            }])

    def test_supported_types_still_work(self, temp_db):
        """All JSON-compatible types must still work after the fix."""
        col = temp_db.create_collection("types_test", dimension=4)
        col.upsert([{
            "id": 1,
            "vector": [1.0, 0.0, 0.0, 0.0],
            "payload": {
                "str": "hello",
                "int": 42,
                "float": 3.14,
                "bool_t": True,
                "bool_f": False,
                "null": None,
                "list": [1, "two", True, None],
                "nested": {"a": 1, "b": [2, 3]},
                "empty_list": [],
                "empty_dict": {},
            },
        }])
        p = col.get([1])[0]["payload"]
        assert p["str"] == "hello"
        assert p["int"] == 42
        assert p["float"] == pytest.approx(3.14)
        assert p["bool_t"] is True
        assert p["bool_f"] is False
        assert p["null"] is None
        assert p["list"] == [1, "two", True, None]
        assert p["nested"] == {"a": 1, "b": [2, 3]}
        assert p["empty_list"] == []
        assert p["empty_dict"] == {}
