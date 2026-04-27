"""
store_core.py — Backward-compatible facade (post-split shim)

All functions are re-exported from sub-modules for zero-change import compatibility.
Original 1815-line monolith split into:
  store_vfs.py  — VFS/CRUD/FTS5/evict (constants + DB ops)
  store_proc.py — /proc stats + dmesg ring buffer
  store_swap.py — OOM constants + swap out/in/fault
  store_criu.py — checkpoint/restore + recall counts
  store_sched.py — scheduler CRUD
"""
from store_vfs import *    # noqa: F401,F403
from store_proc import *   # noqa: F401,F403
from store_swap import *   # noqa: F401,F403
from store_criu import *   # noqa: F401,F403
from store_sched import *  # noqa: F401,F403

# Private symbols needed by store_mm.py (not exported by wildcard import)
from store_vfs import _safe_add_column, _ensure_fts5, _fts5_escape  # noqa: F401
from store_proc import _LEVEL_ORDER  # noqa: F401
from store_criu import _ensure_checkpoint_schema, _checkpoint_cleanup, _CRIU_CHECKPOINT_DIR  # noqa: F401
