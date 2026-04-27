"""
memory-os tmpfs — 测试隔离模块

迭代54：OS 类比 — Linux tmpfs (2000, Christoph Rohland)
tmpfs 是内存文件系统，挂载在 /dev/shm，进程退出后自动销毁。
测试文件在 import store 前先 import tmpfs，自动将 MEMORY_OS_DIR/MEMORY_OS_DB
指向临时目录，避免测试数据污染生产 store.db。

用法（在 test_*.py 头部、import store 之前）：
    import tmpfs  # 必须在 import store 之前
    from store import open_db, ensure_schema, ...

迭代90：Conftest Integration — 当 conftest.py 已设置 tmpfs 时重用，避免重复创建
"""
import atexit
import os
import shutil
import tempfile
from pathlib import Path

# 迭代90：如果 conftest 已设置，重用；否则创建新 tmpdir
_tmpdir = os.environ.get("MEMORY_OS_DIR")
if not _tmpdir or "memory_os_test_" not in _tmpdir:
    # 创建临时目录（等价于 mount -t tmpfs）
    _tmpdir = tempfile.mkdtemp(prefix="memory_os_test_")
    # 设置环境变量，store.py import 时会读取
    os.environ["MEMORY_OS_DIR"] = _tmpdir
    os.environ["MEMORY_OS_DB"] = str(Path(_tmpdir) / "store.db")
# 若已设置，什么都不做（由 conftest 或先前的 import tmpfs 管理）


def _cleanup():
    """进程退出时自动清理（等价于 umount tmpfs）"""
    try:
        shutil.rmtree(_tmpdir, ignore_errors=True)
    except Exception:
        pass
    os.environ.pop("MEMORY_OS_DIR", None)
    os.environ.pop("MEMORY_OS_DB", None)


atexit.register(_cleanup)
