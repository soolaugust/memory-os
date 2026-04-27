"""
pytest global configuration — automatic tmpfs setup for all tests
This ensures MEMORY_OS_DIR and MEMORY_OS_DB env vars are set before any modules are imported.
"""
import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

print(f"[conftest] initializing tmpfs", file=sys.stderr)

# Create tmpfs directory
_tmpdir = tempfile.mkdtemp(prefix="memory_os_test_")
os.environ["MEMORY_OS_DIR"] = _tmpdir
os.environ["MEMORY_OS_DB"] = str(Path(_tmpdir) / "store.db")
print(f"[conftest] MEMORY_OS_DIR={_tmpdir}", file=sys.stderr)

def _cleanup():
    """Cleanup on process exit"""
    try:
        shutil.rmtree(_tmpdir, ignore_errors=True)
    except Exception:
        pass
    os.environ.pop("MEMORY_OS_DIR", None)
    os.environ.pop("MEMORY_OS_DB", None)

atexit.register(_cleanup)

def pytest_configure(config):
    """Ensure tmpfs is set up before any test imports happen"""
    print(f"[pytest_configure] tmpfs already set: MEMORY_OS_DIR={os.environ.get('MEMORY_OS_DIR', 'NOT SET')[-30:]}", file=sys.stderr)
