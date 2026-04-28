#!/usr/bin/env python3
"""
Task13 测试：Optimistic Locking — Compare-And-Swap 多 Agent 并发写保护

OS 类比：Linux seqlock + atomic_cmpxchg — 仅当版本号匹配时才允许写入。

测试矩阵：
  T1: 新 chunk 默认 row_version=1
  T2: cas_update() 版本匹配 → 写入成功，版本+1
  T3: cas_update() 版本冲突 → 返回 {"ok": False, "reason": "version_conflict"}
  T4: cas_update() chunk 不存在 → {"ok": False, "reason": "not_found"}
  T5: 连续 cas_update() — 每次成功后版本递增
  T6: 模拟两 agent 竞争 — 后写的 agent 检测到冲突
  T7: cas_update() 只更新 allowed_keys 字段（安全字段过滤）
  T8: broadcast_invalidate() 写入 ipc_msgq 广播消息
  T9: get_chunk_version() 返回准确版本号
  T10: cas_update() 空 updates 返回错误
"""
import sys
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone, timedelta

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

import tmpfs  # noqa: F401
from store import (
    open_db, ensure_schema, insert_chunk,
    cas_update, get_chunk_version, broadcast_invalidate,
)

PROJECT = f"test_cas_{uuid.uuid4().hex[:6]}"


def _chunk(importance=0.5) -> dict:
    now = datetime.now(timezone.utc)
    ts = now.isoformat()
    return {
        "id": str(uuid.uuid4()),
        "created_at": ts,
        "updated_at": ts,
        "project": PROJECT,
        "source_session": "test",
        "chunk_type": "decision",
        "content": "test content",
        "summary": f"summary {uuid.uuid4().hex[:8]}",
        "tags": json.dumps(["test"]),
        "importance": importance,
        "retrievability": 0.3,
        "last_accessed": ts,
        "feishu_url": None,
    }


def _setup():
    conn = open_db()
    ensure_schema(conn)
    conn.execute("DELETE FROM memory_chunks WHERE project=?", (PROJECT,))
    conn.commit()
    return conn


def _teardown(conn):
    conn.execute("DELETE FROM memory_chunks WHERE project=?", (PROJECT,))
    conn.commit()
    conn.close()


def test_01_default_version():
    """T1: 新 chunk 默认 row_version=1"""
    conn = _setup()
    c = _chunk()
    insert_chunk(conn, c)
    conn.commit()

    v = get_chunk_version(conn, c["id"])
    assert v == 1, f"Expected 1, got {v}"

    _teardown(conn)
    print("  T1 ✓ new chunk defaults to row_version=1")


def test_02_cas_success():
    """T2: cas_update() 版本匹配 → 写入成功，版本+1"""
    conn = _setup()
    c = _chunk()
    insert_chunk(conn, c)
    conn.commit()

    result = cas_update(conn, c["id"], expected_version=1,
                        updates={"summary": "updated summary"})
    conn.commit()

    assert result["ok"] is True, f"Expected ok=True, got {result}"
    assert result["new_version"] == 2, f"Expected v=2, got {result['new_version']}"

    # 验证实际写入
    row = conn.execute(
        "SELECT summary, row_version FROM memory_chunks WHERE id=?", (c["id"],)
    ).fetchone()
    assert row[0] == "updated summary"
    assert row[1] == 2

    _teardown(conn)
    print(f"  T2 ✓ cas_update success: v1→v2, summary updated")


def test_03_cas_version_conflict():
    """T3: cas_update() 版本冲突 → ok=False"""
    conn = _setup()
    c = _chunk()
    insert_chunk(conn, c)
    conn.commit()

    # 先写一次（版本变为 2）
    cas_update(conn, c["id"], expected_version=1, updates={"summary": "first update"})
    conn.commit()

    # 再用旧版本写（应该冲突）
    result = cas_update(conn, c["id"], expected_version=1,
                        updates={"summary": "conflict update"})

    assert result["ok"] is False
    assert result["reason"] == "version_conflict"
    assert result["actual_version"] == 2

    # 验证数据未被第二次写入覆盖
    row = conn.execute("SELECT summary FROM memory_chunks WHERE id=?", (c["id"],)).fetchone()
    assert row[0] == "first update", f"Data should not be overwritten: {row[0]}"

    _teardown(conn)
    print(f"  T3 ✓ cas_update conflict: reason={result['reason']}, actual_v={result['actual_version']}")


def test_04_cas_not_found():
    """T4: cas_update() chunk 不存在 → not_found"""
    conn = _setup()

    result = cas_update(conn, "nonexistent-id", expected_version=1,
                        updates={"summary": "ghost"})
    assert result["ok"] is False
    assert result["reason"] == "not_found"

    _teardown(conn)
    print("  T4 ✓ cas_update not_found")


def test_05_sequential_cas():
    """T5: 连续 cas_update() — 每次成功后版本递增"""
    conn = _setup()
    c = _chunk()
    insert_chunk(conn, c)
    conn.commit()

    for i in range(5):
        expected_v = i + 1
        result = cas_update(conn, c["id"], expected_version=expected_v,
                            updates={"importance": 0.1 * (i + 1)})
        conn.commit()
        assert result["ok"] is True, f"Step {i} failed: {result}"
        assert result["new_version"] == expected_v + 1

    final_v = get_chunk_version(conn, c["id"])
    assert final_v == 6, f"Expected v=6, got {final_v}"

    _teardown(conn)
    print(f"  T5 ✓ sequential CAS: v1→v6 (5 successful updates)")


def test_06_two_agent_competition():
    """T6: 两 agent 竞争 — 后写的 agent 检测到冲突"""
    conn = _setup()
    c = _chunk()
    insert_chunk(conn, c)
    conn.commit()

    # Agent A 读版本 v=1
    v_a = get_chunk_version(conn, c["id"])  # = 1
    # Agent B 读版本 v=1（同时）
    v_b = get_chunk_version(conn, c["id"])  # = 1

    # Agent A 先写成功
    result_a = cas_update(conn, c["id"], expected_version=v_a,
                          updates={"summary": "Agent A update"})
    conn.commit()
    assert result_a["ok"] is True

    # Agent B 后写 → 冲突
    result_b = cas_update(conn, c["id"], expected_version=v_b,
                          updates={"summary": "Agent B update"})

    assert result_b["ok"] is False
    assert result_b["reason"] == "version_conflict"
    assert result_b["actual_version"] == 2

    # Agent B 重读版本后可以重试
    v_b2 = get_chunk_version(conn, c["id"])  # = 2
    result_b2 = cas_update(conn, c["id"], expected_version=v_b2,
                           updates={"summary": "Agent B retry"})
    conn.commit()
    assert result_b2["ok"] is True
    assert result_b2["new_version"] == 3

    _teardown(conn)
    print("  T6 ✓ two-agent competition: A wins, B detects conflict, B retries successfully")


def test_07_safe_field_filter():
    """T7: cas_update() 只更新 allowed_keys，不允许写 id/row_version/project"""
    conn = _setup()
    c = _chunk()
    insert_chunk(conn, c)
    conn.commit()

    # 尝试修改 id 和 project（应被过滤）
    result = cas_update(conn, c["id"], expected_version=1,
                        updates={
                            "id": "hacked-id",
                            "project": "hacked-project",
                            "summary": "safe update",
                        })
    conn.commit()

    assert result["ok"] is True  # 因为 summary 是 safe field
    row = conn.execute(
        "SELECT id, project, summary FROM memory_chunks WHERE id=?", (c["id"],)
    ).fetchone()
    assert row[0] == c["id"], "id should not be modified"
    assert row[1] == PROJECT, "project should not be modified"
    assert row[2] == "safe update"

    _teardown(conn)
    print("  T7 ✓ safe field filter: id/project unchanged, summary updated")


def test_08_broadcast_invalidate():
    """T8: broadcast_invalidate() 写入 ipc_msgq"""
    conn = _setup()
    c1 = _chunk()
    c2 = _chunk()
    for c in (c1, c2):
        insert_chunk(conn, c)
    conn.commit()

    count = broadcast_invalidate(conn, [c1["id"], c2["id"]], agent_id="agent_a")
    conn.commit()

    assert count == 2, f"Expected 2 messages, got {count}"

    # 验证消息在 ipc_msgq 中
    msgs = conn.execute(
        "SELECT source_agent, msg_type, payload FROM ipc_msgq WHERE source_agent='agent_a'"
    ).fetchall()
    assert len(msgs) == 2
    for source, mtype, payload in msgs:
        assert mtype == "INVALIDATE"
        data = json.loads(payload)
        assert "chunk_id" in data

    # cleanup before teardown
    conn.execute("DELETE FROM ipc_msgq WHERE source_agent='agent_a'")
    _teardown(conn)
    print(f"  T8 ✓ broadcast_invalidate: {count} INVALIDATE msgs in ipc_msgq")


def test_09_get_chunk_version():
    """T9: get_chunk_version() 返回准确版本"""
    conn = _setup()
    c = _chunk()
    insert_chunk(conn, c)
    conn.commit()

    assert get_chunk_version(conn, c["id"]) == 1
    assert get_chunk_version(conn, "nonexistent") is None

    cas_update(conn, c["id"], 1, {"summary": "v2"})
    conn.commit()
    assert get_chunk_version(conn, c["id"]) == 2

    _teardown(conn)
    print("  T9 ✓ get_chunk_version accurate")


def test_10_empty_updates():
    """T10: cas_update() 空 updates 返回错误"""
    conn = _setup()
    c = _chunk()
    insert_chunk(conn, c)
    conn.commit()

    result = cas_update(conn, c["id"], expected_version=1, updates={})
    assert result["ok"] is False
    assert result["reason"] in ("empty_updates", "no_safe_fields")

    # 只有不安全字段也失败
    result2 = cas_update(conn, c["id"], expected_version=1, updates={"id": "hack"})
    assert result2["ok"] is False

    _teardown(conn)
    print("  T10 ✓ empty/unsafe updates return error")


if __name__ == "__main__":
    print("Task13 测试：Optimistic Locking — CAS 多 Agent 并发写")
    print("=" * 60)

    tests = [
        test_01_default_version,
        test_02_cas_success,
        test_03_cas_version_conflict,
        test_04_cas_not_found,
        test_05_sequential_cas,
        test_06_two_agent_competition,
        test_07_safe_field_filter,
        test_08_broadcast_invalidate,
        test_09_get_chunk_version,
        test_10_empty_updates,
    ]

    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            failed += 1
            import traceback
            print(f"  FAIL {t.__name__}: {e}")
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"结果：{passed}/{passed+failed} 通过")
    if failed:
        import sys
        sys.exit(1)
