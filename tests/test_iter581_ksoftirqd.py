"""
iter581: ksoftirqd — Runtime Reclaim Trigger on Write Path

验证 raise_softirq / consume_softirq 工作流：
  - 写入路径检测 DB 不健康 → 写标志文件
  - loader 启动时消费标志 → 强制 reclaim bypass deferred_initcall
"""
import json
import os
import sys
import sqlite3
import time
import tempfile
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["MEMORY_OS_TESTING"] = "1"

import tmpfs  # noqa: F401 — 自动 mount tmpfs

from store_core import open_db, ensure_schema, MEMORY_OS_DIR
from store_mm import raise_softirq, consume_softirq, _SOFTIRQ_FLAG


import pytest


@pytest.fixture(autouse=True)
def clean_db():
    """每个测试前清理 DB 和标志文件"""
    conn = open_db()
    ensure_schema(conn)
    conn.execute("DELETE FROM memory_chunks")
    conn.commit()
    conn.close()
    _SOFTIRQ_FLAG.unlink(missing_ok=True)
    yield
    _SOFTIRQ_FLAG.unlink(missing_ok=True)


def _raw_insert(conn, cid, project="test_proj", importance=0.5, access_count=1, chunk_type="decision"):
    """直接 SQL INSERT 绕过 VFS 质量门控"""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO memory_chunks (id, project, chunk_type, importance, access_count, "
        "summary, content, created_at, updated_at, last_accessed, oom_adj) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,0)",
        (cid, project, chunk_type, importance, access_count,
         f"test summary {cid}", f"test content {cid}", now, now, now),
    )


def _setup_db_with_chunks(n_total=10, n_zero_access=0, n_zombies=0, project="test_proj"):
    """创建测试 DB 并插入 chunks，直接 SQL 写入"""
    conn = open_db()
    ensure_schema(conn)

    for i in range(n_total):
        imp = 0.15 if i < n_zombies else 0.5
        acc = 0 if i < n_zero_access else 3
        _raw_insert(conn, f"chunk_{i}_{time.time()}", project=project,
                    importance=imp, access_count=acc)

    conn.commit()
    return conn


def test_raise_softirq_zombies():
    """T1: zombies > 0 时应 raise softirq"""
    conn = _setup_db_with_chunks(n_total=10, n_zero_access=3, n_zombies=2, project="test_proj")
    result = raise_softirq(conn, "test_proj")
    conn.close()

    assert result["raised"] is True
    assert "zombies" in result["reason"]
    assert _SOFTIRQ_FLAG.exists()
    _SOFTIRQ_FLAG.unlink(missing_ok=True)


def test_raise_softirq_high_zero_pct():
    """T2: 零访问率 >= 40% 时应 raise softirq"""
    conn = _setup_db_with_chunks(n_total=10, n_zero_access=5, n_zombies=0, project="test_proj")
    result = raise_softirq(conn, "test_proj")
    conn.close()

    assert result["raised"] is True
    assert "zero_pct" in result["reason"]
    assert _SOFTIRQ_FLAG.exists()
    _SOFTIRQ_FLAG.unlink(missing_ok=True)


def test_raise_softirq_healthy():
    """T3: 健康 DB（低零访问率 + 无 zombies）不应 raise"""
    conn = _setup_db_with_chunks(n_total=10, n_zero_access=2, n_zombies=0, project="test_proj")
    result = raise_softirq(conn, "test_proj")
    conn.close()

    assert result["raised"] is False
    assert result["reason"] == "healthy"
    assert not _SOFTIRQ_FLAG.exists()


def test_raise_softirq_disabled():
    """T4: config 禁用时不 raise"""
    conn = _setup_db_with_chunks(n_total=10, n_zero_access=8, n_zombies=5, project="test_proj")

    # 临时修改 config
    import config
    orig = config._REGISTRY.get("ksoftirqd.enabled")
    config._REGISTRY["ksoftirqd.enabled"] = (False, bool, None, None, None, "test")
    try:
        result = raise_softirq(conn, "test_proj")
        assert result["raised"] is False
        assert result["reason"] == "disabled"
    finally:
        if orig:
            config._REGISTRY["ksoftirqd.enabled"] = orig
        else:
            del config._REGISTRY["ksoftirqd.enabled"]
    conn.close()


def test_raise_softirq_empty_db():
    """T5: 空 DB 不 raise"""
    conn = open_db()
    ensure_schema(conn)
    result = raise_softirq(conn, "test_proj")
    conn.close()

    assert result["raised"] is False
    assert result["reason"] == "empty_db"


def test_consume_softirq_pending():
    """T6: 标志文件存在时应消费并返回 pending=True"""
    # 手动创建标志文件
    info = {"raised_at": "2026-05-02T15:13:00+00:00", "reason": "zombies=5",
            "project": "test_proj", "total": 50, "zero_acc": 20, "zombies": 5}
    _SOFTIRQ_FLAG.write_text(json.dumps(info), encoding="utf-8")

    result = consume_softirq()
    assert result["pending"] is True
    assert result["info"]["reason"] == "zombies=5"
    assert not _SOFTIRQ_FLAG.exists()  # 消费后删除


def test_consume_softirq_no_flag():
    """T7: 无标志文件时应返回 pending=False"""
    _SOFTIRQ_FLAG.unlink(missing_ok=True)
    result = consume_softirq()
    assert result["pending"] is False


def test_consume_softirq_corrupt_file():
    """T8: 标志文件格式错误时应清理并返回 pending=False"""
    _SOFTIRQ_FLAG.write_text("not json {{{", encoding="utf-8")
    result = consume_softirq()
    assert result["pending"] is False
    assert not _SOFTIRQ_FLAG.exists()


def test_raise_then_consume_roundtrip():
    """T9: 完整 raise→consume 循环验证"""
    conn = _setup_db_with_chunks(n_total=10, n_zero_access=3, n_zombies=2, project="test_proj")

    # raise
    r1 = raise_softirq(conn, "test_proj")
    assert r1["raised"] is True
    assert _SOFTIRQ_FLAG.exists()

    # consume
    r2 = consume_softirq()
    assert r2["pending"] is True
    assert "zombies" in r2["info"]["reason"]
    assert not _SOFTIRQ_FLAG.exists()

    # 再次 consume → 幂等
    r3 = consume_softirq()
    assert r3["pending"] is False

    conn.close()


def test_raise_softirq_global_project():
    """T10: project=None 时全局扫描"""
    conn = _setup_db_with_chunks(n_total=10, n_zero_access=6, n_zombies=0, project="global")
    result = raise_softirq(conn, None)
    conn.close()

    assert result["raised"] is True
    assert "zero_pct" in result["reason"]
    _SOFTIRQ_FLAG.unlink(missing_ok=True)


def test_raise_softirq_cross_project():
    """T11: project 过滤只看指定 project + global"""
    conn = open_db()
    ensure_schema(conn)

    # 插入 other_proj 的 zombies（对 test_proj 不可见）
    for i in range(5):
        _raw_insert(conn, f"other_{i}", project="other_proj", importance=0.1, access_count=0)

    # 插入 test_proj 的健康 chunks
    for i in range(10):
        _raw_insert(conn, f"healthy_{i}", project="test_proj", importance=0.8, access_count=5)
    conn.commit()

    # test_proj 视角应该是健康的（other_proj 的 zombies 不影响）
    result = raise_softirq(conn, "test_proj")
    conn.close()
    assert result["raised"] is False
    assert result["reason"] == "healthy"


def test_raise_softirq_threshold_tunable():
    """T12: zero_threshold sysctl 可调"""
    conn = _setup_db_with_chunks(n_total=10, n_zero_access=3, n_zombies=0, project="test_proj")

    import config
    # 默认 0.40 → 30% 零访问不触发
    result1 = raise_softirq(conn, "test_proj")
    assert result1["raised"] is False

    # 改为 0.25 → 30% 零访问应触发
    orig = config._REGISTRY.get("ksoftirqd.zero_threshold")
    config._REGISTRY["ksoftirqd.zero_threshold"] = (0.25, float, 0.20, 0.80, None, "test")
    try:
        result2 = raise_softirq(conn, "test_proj")
        assert result2["raised"] is True
    finally:
        if orig:
            config._REGISTRY["ksoftirqd.zero_threshold"] = orig
        else:
            del config._REGISTRY["ksoftirqd.zero_threshold"]
    conn.close()
    _SOFTIRQ_FLAG.unlink(missing_ok=True)


def test_flag_file_content():
    """T13: 标志文件包含正确的 JSON 结构"""
    conn = _setup_db_with_chunks(n_total=10, n_zero_access=3, n_zombies=2, project="test_proj")
    raise_softirq(conn, "test_proj")
    conn.close()

    assert _SOFTIRQ_FLAG.exists()
    data = json.loads(_SOFTIRQ_FLAG.read_text(encoding="utf-8"))
    assert "raised_at" in data
    assert "reason" in data
    assert "project" in data
    assert "total" in data
    assert "zero_acc" in data
    assert "zombies" in data
    assert data["project"] == "test_proj"
    _SOFTIRQ_FLAG.unlink(missing_ok=True)


def test_idempotent_raise():
    """T14: 多次 raise 只保留最新标志"""
    conn = _setup_db_with_chunks(n_total=10, n_zero_access=3, n_zombies=2, project="test_proj")

    r1 = raise_softirq(conn, "test_proj")
    time.sleep(0.01)
    r2 = raise_softirq(conn, "test_proj")
    conn.close()

    assert r1["raised"] is True
    assert r2["raised"] is True
    # 只有一个标志文件（最新的覆盖旧的）
    assert _SOFTIRQ_FLAG.exists()
    data = json.loads(_SOFTIRQ_FLAG.read_text(encoding="utf-8"))
    assert data["project"] == "test_proj"
    _SOFTIRQ_FLAG.unlink(missing_ok=True)


def test_performance():
    """T15: raise_softirq 应在 5ms 内完成"""
    conn = _setup_db_with_chunks(n_total=100, n_zero_access=50, n_zombies=10, project="test_proj")

    times = []
    for _ in range(10):
        t0 = time.time()
        raise_softirq(conn, "test_proj")
        times.append((time.time() - t0) * 1000)
        _SOFTIRQ_FLAG.unlink(missing_ok=True)

    conn.close()
    avg = sum(times) / len(times)
    assert avg < 5.0, f"avg={avg:.2f}ms exceeds 5ms budget"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
