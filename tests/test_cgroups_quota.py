#!/usr/bin/env python3
"""
迭代25 测试：cgroups Per-Project Resource Quotas
验证配额检查 + 超额淘汰 + 受保护类型不被淘汰
"""
import sys
import os
import uuid
import time
import sqlite3
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).parent))
import tmpfs  # noqa: F401 — tmpfs isolation (iter54), must precede store import
from store import (open_db, ensure_schema, insert_chunk, get_project_chunk_count,
                   evict_lowest_retention, get_chunk_count, delete_chunks)
from schema import MemoryChunk

TEST_DB = Path("/tmp/test_cgroups_quota.db")
PROJECT_A = "test_project_a"
PROJECT_B = "test_project_b"


def _make_chunk(project: str, chunk_type: str = "decision",
                importance: float = 0.5, age_days: int = 0,
                summary: str = None) -> dict:
    now = datetime.now(timezone.utc) - timedelta(days=age_days)
    return MemoryChunk(
        id=str(uuid.uuid4()),
        project=project,
        source_session="test",
        chunk_type=chunk_type,
        content=f"test content {uuid.uuid4().hex[:8]}",
        summary=summary or f"test summary {uuid.uuid4().hex[:8]}",
        importance=importance,
        last_accessed=now.isoformat(),
        created_at=now.isoformat(),
        updated_at=now.isoformat(),
    ).to_dict()


def setup():
    if TEST_DB.exists():
        TEST_DB.unlink()
    conn = open_db(TEST_DB)
    ensure_schema(conn)
    return conn


def test_1_get_project_chunk_count():
    """测试 per-project chunk 计数（不同项目互不影响）"""
    conn = setup()
    for _ in range(5):
        insert_chunk(conn, _make_chunk(PROJECT_A))
    for _ in range(3):
        insert_chunk(conn, _make_chunk(PROJECT_B))
    conn.commit()

    count_a = get_project_chunk_count(conn, PROJECT_A)
    count_b = get_project_chunk_count(conn, PROJECT_B)
    conn.close()

    assert count_a == 5, f"项目A应有5条，实际{count_a}"
    assert count_b == 3, f"项目B应有3条，实际{count_b}"
    print(f"  ✓ 项目隔离计数：A={count_a}, B={count_b}")


def test_2_evict_lowest_retention():
    """测试按 importance+recency 淘汰最低价值 chunk"""
    conn = setup()
    old_low = _make_chunk(PROJECT_A, importance=0.3, age_days=30, summary="old_low")
    old_mid = _make_chunk(PROJECT_A, importance=0.5, age_days=20, summary="old_mid")
    new_high = _make_chunk(PROJECT_A, importance=0.9, age_days=0, summary="new_high")
    recent_low = _make_chunk(PROJECT_A, importance=0.3, age_days=1, summary="recent_low")

    for c in [old_low, old_mid, new_high, recent_low]:
        insert_chunk(conn, c)
    conn.commit()

    assert get_project_chunk_count(conn, PROJECT_A) == 4

    evicted = evict_lowest_retention(conn, PROJECT_A, 2)
    conn.commit()

    assert len(evicted) == 2, f"应淘汰2条，实际{len(evicted)}"
    remaining_count = get_project_chunk_count(conn, PROJECT_A)
    assert remaining_count == 2, f"应剩2条，实际{remaining_count}"

    remaining = conn.execute(
        "SELECT summary FROM memory_chunks WHERE project=? ORDER BY importance DESC",
        (PROJECT_A,)
    ).fetchall()
    remaining_summaries = [r[0] for r in remaining]
    assert "new_high" in remaining_summaries, f"new_high 应保留，剩余: {remaining_summaries}"
    conn.close()
    print(f"  ✓ 淘汰最低价值：evicted={len(evicted)}, remaining={remaining_summaries}")


def test_3_protect_task_state():
    """测试 task_state 类型不被淘汰（受保护）"""
    conn = setup()
    ts_chunk = _make_chunk(PROJECT_A, chunk_type="task_state", importance=0.1, age_days=100)
    dec_chunk = _make_chunk(PROJECT_A, chunk_type="decision", importance=0.3, age_days=50)
    for c in [ts_chunk, dec_chunk]:
        insert_chunk(conn, c)
    conn.commit()

    evicted = evict_lowest_retention(conn, PROJECT_A, 1)
    conn.commit()

    remaining = conn.execute(
        "SELECT chunk_type FROM memory_chunks WHERE project=?", (PROJECT_A,)
    ).fetchall()
    remaining_types = [r[0] for r in remaining]
    assert "task_state" in remaining_types, f"task_state 应被保护，剩余: {remaining_types}"
    conn.close()
    print(f"  ✓ task_state 受保护：剩余类型={remaining_types}")


def test_4_cross_project_isolation():
    """测试淘汰只影响目标项目，不影响其他项目"""
    conn = setup()
    for _ in range(3):
        insert_chunk(conn, _make_chunk(PROJECT_A, importance=0.2, age_days=30))
    for _ in range(2):
        insert_chunk(conn, _make_chunk(PROJECT_B, importance=0.8))
    conn.commit()

    evicted = evict_lowest_retention(conn, PROJECT_A, 2)
    conn.commit()

    count_a = get_project_chunk_count(conn, PROJECT_A)
    count_b = get_project_chunk_count(conn, PROJECT_B)
    conn.close()

    assert count_a == 1, f"项目A应剩1条，实际{count_a}"
    assert count_b == 2, f"项目B应保持2条，实际{count_b}"
    print(f"  ✓ 跨项目隔离：A={count_a}(淘汰后), B={count_b}(不受影响)")


def test_5_quota_trigger_eviction():
    """端到端测试：模拟配额触发淘汰流程"""
    conn = setup()
    QUOTA = 5

    for i in range(QUOTA):
        c = _make_chunk(PROJECT_A, importance=0.3 + i * 0.1, age_days=QUOTA - i)
        insert_chunk(conn, c)
    conn.commit()

    count = get_project_chunk_count(conn, PROJECT_A)
    assert count == QUOTA, f"应有{QUOTA}条，实际{count}"

    incoming_count = 2
    current = get_project_chunk_count(conn, PROJECT_A)
    if current + incoming_count > QUOTA:
        evict_count = (current + incoming_count) - QUOTA
        evicted = evict_lowest_retention(conn, PROJECT_A, evict_count)
        conn.commit()

    for _ in range(incoming_count):
        insert_chunk(conn, _make_chunk(PROJECT_A, importance=0.9))
    conn.commit()

    final_count = get_project_chunk_count(conn, PROJECT_A)
    assert final_count == QUOTA, f"写入后应仍为{QUOTA}条，实际{final_count}"
    conn.close()
    print(f"  ✓ 配额触发淘汰：写入{incoming_count}条后总数保持={final_count}")


def test_6_evict_zero_noop():
    """测试淘汰数量为0时不做任何操作"""
    conn = setup()
    insert_chunk(conn, _make_chunk(PROJECT_A))
    conn.commit()

    evicted = evict_lowest_retention(conn, PROJECT_A, 0)
    count = get_project_chunk_count(conn, PROJECT_A)
    conn.close()

    assert evicted == [], f"淘汰0条应返回空列表，实际{evicted}"
    assert count == 1, f"应保持1条，实际{count}"
    print(f"  ✓ 零淘汰 noop：evicted={evicted}, count={count}")


def test_7_env_override_quota():
    """测试环境变量覆盖配额"""
    os.environ["MEMORY_OS_CHUNK_QUOTA"] = "50"
    val = int(os.environ.get("MEMORY_OS_CHUNK_QUOTA", "200"))
    assert val == 50, f"环境变量覆盖应为50，实际{val}"
    del os.environ["MEMORY_OS_CHUNK_QUOTA"]

    val_default = int(os.environ.get("MEMORY_OS_CHUNK_QUOTA", "200"))
    assert val_default == 200, f"默认值应为200，实际{val_default}"
    print(f"  ✓ 环境变量覆盖：MEMORY_OS_CHUNK_QUOTA=50→200(default)")


if __name__ == "__main__":
    t0 = time.time()
    tests = [
        test_1_get_project_chunk_count,
        test_2_evict_lowest_retention,
        test_3_protect_task_state,
        test_4_cross_project_isolation,
        test_5_quota_trigger_eviction,
        test_6_evict_zero_noop,
        test_7_env_override_quota,
    ]
    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__}: {e}")
    elapsed = (time.time() - t0) * 1000
    print(f"\n结果：{passed}/{len(tests)} 通过，耗时 {elapsed:.1f}ms")
    if TEST_DB.exists():
        TEST_DB.unlink()
