"""
iter535: deferred_initcall — Conditional Boot Subsystem Bypass

OS 类比：Linux deferred_struct_pages (Mel Gorman, 2015, kernel 4.2)
  CONFIG_DEFERRED_STRUCT_PAGE_INIT 在大内存系统中跳过 boot node 以外的 memmap 初始化。
  小内存系统不触发，大系统 boot time 缩短 80%+。

测试：验证 _should_defer_reclaim() 健康探针的各条件判定逻辑。
"""
import sys
import os
import sqlite3
import json
import tempfile
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta

# ── tmpfs 测试隔离 ──
_tmpdir = tempfile.mkdtemp(prefix="test_iter535_")
os.environ["MEMORY_OS_DIR"] = _tmpdir
os.environ["MEMORY_OS_DB"] = os.path.join(_tmpdir, "store.db")

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "hooks"))

from config import get as _sysctl
from store import open_db, ensure_schema, insert_chunk, dmesg_log, DMESG_INFO
from schema import MemoryChunk


def _setup_db():
    """Create fresh DB with schema — wipe existing data for isolation."""
    conn = open_db()
    ensure_schema(conn)
    # Wipe data for test isolation
    conn.execute("DELETE FROM memory_chunks")
    conn.execute("DELETE FROM dmesg")
    conn.commit()
    return conn


def _insert_chunks(conn, n, project="test_proj", importance=0.7, access_count=1):
    """Helper: insert N chunks."""
    for i in range(n):
        chunk = MemoryChunk(
            project=project,
            chunk_type="decision",
            summary=f"Test chunk {i} for deferred_initcall",
            importance=importance,
            content=f"Content {i}",
        )
        insert_chunk(conn, chunk.to_dict())
        if access_count > 0:
            conn.execute(
                "UPDATE memory_chunks SET access_count=? WHERE id=?",
                (access_count, chunk.id)
            )
    conn.commit()


def _insert_recent_reclaim_dmesg(conn, hours_ago=0.5):
    """Insert a dmesg entry simulating recent reclaim."""
    ts = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    conn.execute(
        "INSERT INTO dmesg (timestamp, level, subsystem, message) VALUES (?, ?, ?, ?)",
        (ts, 3, "kfree_rcu", "freed=0 dead=0 skip_prot=0 0.1ms")
    )
    conn.commit()


# Import the function under test
from loader import _should_defer_reclaim


def test_empty_db_defers():
    """T1: Empty DB (0 chunks) should defer — trivially healthy."""
    conn = _setup_db()
    _insert_recent_reclaim_dmesg(conn)
    should_defer, reason, metrics = _should_defer_reclaim(conn, "test_proj")
    # 0 < 150 (defer_max_chunks), zero_pct=0 < 0.30, 0 zombies, recent reclaim
    assert should_defer is True, f"Expected defer, got {reason}"
    assert reason == "healthy"
    conn.close()


def test_small_healthy_db_defers():
    """T2: Small DB with low zero-access rate should defer."""
    conn = _setup_db()
    _insert_chunks(conn, 50, importance=0.7, access_count=2)
    _insert_recent_reclaim_dmesg(conn)
    should_defer, reason, metrics = _should_defer_reclaim(conn, "test_proj")
    assert should_defer is True, f"Expected defer, got {reason}"
    assert metrics["total"] == 50
    conn.close()


def test_large_db_does_not_defer():
    """T3: DB with >= defer_max_chunks (150) should NOT defer."""
    conn = _setup_db()
    _insert_chunks(conn, 160, importance=0.7, access_count=1)
    should_defer, reason, metrics = _should_defer_reclaim(conn, "test_proj")
    assert should_defer is False, f"Expected no defer, got {reason}"
    assert "large_db" in reason
    conn.close()


def test_high_zero_access_does_not_defer():
    """T4: Zero access rate >= 30% should NOT defer."""
    conn = _setup_db()
    # 40 chunks: 15 with access, 25 zero access → 25/40 = 62.5% > 30%
    _insert_chunks(conn, 15, importance=0.7, access_count=3)
    _insert_chunks(conn, 25, importance=0.7, access_count=0)
    _insert_recent_reclaim_dmesg(conn)
    should_defer, reason, metrics = _should_defer_reclaim(conn, "test_proj")
    assert should_defer is False, f"Expected no defer, got {reason}"
    assert "high_zero" in reason
    conn.close()


def test_zombies_prevent_defer():
    """T5: Presence of zombie chunks (imp<0.2, acc=0) prevents deferral."""
    conn = _setup_db()
    _insert_chunks(conn, 30, importance=0.7, access_count=2)
    # Add 3 zombies
    _insert_chunks(conn, 3, importance=0.15, access_count=0)
    _insert_recent_reclaim_dmesg(conn)
    should_defer, reason, metrics = _should_defer_reclaim(conn, "test_proj")
    assert should_defer is False, f"Expected no defer, got {reason}"
    assert "zombies" in reason
    conn.close()


def test_cooldown_expired_does_not_defer():
    """T6: If last reclaim was > defer_cooldown_hours ago, should NOT defer."""
    conn = _setup_db()
    _insert_chunks(conn, 30, importance=0.7, access_count=2)
    # Insert old reclaim dmesg (3 hours ago > default 2h cooldown)
    _insert_recent_reclaim_dmesg(conn, hours_ago=3.0)
    should_defer, reason, metrics = _should_defer_reclaim(conn, "test_proj")
    assert should_defer is False, f"Expected no defer, got {reason}"
    assert "cooldown_expired" in reason
    conn.close()


def test_no_reclaim_history_still_defers():
    """T7: No dmesg reclaim history — should still defer (no evidence of need)."""
    conn = _setup_db()
    _insert_chunks(conn, 30, importance=0.7, access_count=2)
    # No reclaim dmesg inserted
    should_defer, reason, metrics = _should_defer_reclaim(conn, "test_proj")
    assert should_defer is True, f"Expected defer, got {reason}"
    conn.close()


def test_global_project_counts():
    """T8: Global project chunks are included in the count."""
    conn = _setup_db()
    _insert_chunks(conn, 20, project="test_proj", importance=0.7, access_count=2)
    _insert_chunks(conn, 100, project="global", importance=0.7, access_count=2)
    _insert_recent_reclaim_dmesg(conn)
    should_defer, reason, metrics = _should_defer_reclaim(conn, "test_proj")
    # total = 20 + 100 = 120 < 150
    assert should_defer is True, f"Expected defer, got {reason}"
    assert metrics["total"] == 120
    conn.close()


def test_global_overflow_triggers_large_db():
    """T9: Global + project > threshold should not defer."""
    conn = _setup_db()
    _insert_chunks(conn, 80, project="test_proj", importance=0.7, access_count=2)
    _insert_chunks(conn, 80, project="global", importance=0.7, access_count=2)
    should_defer, reason, metrics = _should_defer_reclaim(conn, "test_proj")
    # total = 160 >= 150
    assert should_defer is False, f"Expected no defer, got {reason}"
    assert "large_db" in reason
    conn.close()


def test_boundary_zero_pct():
    """T10: Exactly at 30% zero access should NOT defer (>= threshold)."""
    conn = _setup_db()
    # 100 chunks: 70 with access, 30 zero → 30/100 = 30% == threshold
    _insert_chunks(conn, 70, importance=0.7, access_count=1)
    _insert_chunks(conn, 30, importance=0.7, access_count=0)
    _insert_recent_reclaim_dmesg(conn)
    should_defer, reason, metrics = _should_defer_reclaim(conn, "test_proj")
    assert should_defer is False, f"Expected no defer, got {reason}"
    assert "high_zero" in reason
    conn.close()


def test_config_tunables_exist():
    """T11: Verify the 3 new sysctl tunables are registered."""
    assert _sysctl("loader.defer_max_chunks") == 150
    assert _sysctl("loader.defer_zero_pct") == 0.30
    assert _sysctl("loader.defer_cooldown_hours") == 2.0


def test_performance():
    """T12: _should_defer_reclaim should complete in < 5ms."""
    conn = _setup_db()
    _insert_chunks(conn, 100, importance=0.7, access_count=2)
    _insert_recent_reclaim_dmesg(conn)

    t0 = time.time()
    for _ in range(100):
        _should_defer_reclaim(conn, "test_proj")
    elapsed = (time.time() - t0) * 1000 / 100  # ms per call
    assert elapsed < 5.0, f"Too slow: {elapsed:.2f}ms/call"
    print(f"  Performance: {elapsed:.3f}ms/call")
    conn.close()


# ── Run all tests ──
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")

    # Cleanup
    import shutil
    shutil.rmtree(_tmpdir, ignore_errors=True)
