"""
Shared pytest fixtures and configuration for the velesdb test suite.

Why a conftest.py?
  The `temp_db` fixture was duplicated verbatim across test_issue_fixes.py,
  test_agent_memory.py, and test_e2e_complete.py.  pytest auto-discovers this
  file in the tests/ directory, so every test module gets the shared fixtures
  without explicit imports.

  ImportError AND AttributeError are both caught: AttributeError fires when the
  native extension is present on sys.path but the top-level __init__.py re-export
  is broken (e.g. the PyO3 symbol was renamed while __init__.py still referenced
  the old name).
"""

import shutil
import tempfile

import pytest

try:
    from velesdb import Database

    VELESDB_AVAILABLE = True
except (ImportError, AttributeError):
    VELESDB_AVAILABLE = False
    Database = None  # type: ignore[assignment,misc]

# _SKIP_NO_BINDINGS is the canonical skip mark shared across all test modules.
# Each module must assign it to their own `pytestmark` — pytest does NOT
# propagate conftest.pytestmark automatically to collected test files.
_SKIP_NO_BINDINGS = pytest.mark.skipif(
    not VELESDB_AVAILABLE,
    reason="VelesDB Python bindings not installed. Run: maturin develop",
)


@pytest.fixture
def temp_db():
    """Yield a Database opened in a fresh temporary directory.

    Chosen as function-scoped (the default) because each test must start with
    an empty, isolated database — collection names collide otherwise.
    The temporary directory is deleted unconditionally in the finalizer even
    when the test itself raises.
    """
    temp_dir = tempfile.mkdtemp()
    db = Database(temp_dir)
    yield db
    shutil.rmtree(temp_dir, ignore_errors=True)
