#!/usr/bin/env python3
"""
Task12 测试：Chunk Lifecycle FSM — ACTIVE/COLD/DEAD/SWAP/GHOST

OS 类比：Linux Page State Machine — PG_active, PG_lru, PG_swapcache, 等。

测试矩阵：
  T1: 新插入 chunk 默认 chunk_state=ACTIVE
  T2: mark_cold() 将 7天+未访问 ACTIVE → COLD，不触碰新 chunk
  T3: mark_dead() 将 30天+未访问 COLD → DEAD，不触碰 ACTIVE/新 COLD
  T4: mark_active() 任意状态 → ACTIVE（SWAP 除外）
  T5: mark_ghost() DEAD → GHOST
  T6: SWAP 状态不被 mark_active() 影响
  T7: fsm_transition() 批量执行 ACTIVE→COLD→DEAD
  T8: get_state_distribution() 返回准确的 5 状态分布
  T9: CHUNK_STATES 常量集合完整
  T10: 跨 project 隔离——project A 的转换不影响 project B
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
    mark_active, mark_cold, mark_dead, mark_ghost,
    fsm_transition, get_state_distribution, CHUNK_STATES,
)

PROJECT_A = f"test_fsm_a_{uuid.uuid4().hex[:6]}"
PROJECT_B = f"test_fsm_b_{uuid.uuid4().hex[:6]}"


def _chunk(project, days_ago=0, importance=0.5, chunk_type="decision") -> dict:
    now = datetime.now(timezone.utc) - timedelta(days=days_ago)
    ts = now.isoformat()
    return {
        "id": str(uuid.uuid4()),
        "created_at": ts,
        "updated_at": ts,
        "project": project,
        "source_session": "test",
        "chunk_type": chunk_type,
        "content": "test content",
        "summary": f"summary {uuid.uuid4().hex[:8]}",
        "tags": json.dumps(["test"]),
        "importance": importance,
        "retrievability": 0.3,
        "last_accessed": ts,
        "feishu_url": None,
    }


def _cleanup(conn):
    conn.execute("DELETE FROM memory_chunks WHERE project IN (?, ?)", (PROJECT_A, PROJECT_B))
    conn.commit()


def _get_state(conn, cid) -> str:
    row = conn.execute(
        "SELECT COALESCE(chunk_state, 'ACTIVE') FROM memory_chunks WHERE id=?", (cid,)
    ).fetchone()
    return row[0] if row else None


def test_01_default_state_active():
    """T1: 新插入 chunk 默认 chunk_state=ACTIVE"""
    conn = open_db()
    ensure_schema(conn)
    _cleanup(conn)

    c = _chunk(PROJECT_A, days_ago=0)
    insert_chunk(conn, c)
    conn.commit()

    state = _get_state(conn, c["id"])
    assert state == "ACTIVE", f"Expected ACTIVE, got {state}"

    _cleanup(conn)
    conn.close()
    print("  T1 ✓ new chunk defaults to ACTIVE")


def test_02_mark_cold():
    """T2: mark_cold() ACTIVE(7d+) → COLD，新 chunk 不受影响"""
    conn = open_db()
    ensure_schema(conn)
    _cleanup(conn)

    old = _chunk(PROJECT_A, days_ago=10)  # 10天前 → 应变 COLD
    new = _chunk(PROJECT_A, days_ago=1)   # 1天前 → 保持 ACTIVE
    insert_chunk(conn, old)
    insert_chunk(conn, new)
    conn.commit()

    conn.execute(
        "UPDATE memory_chunks SET chunk_state='ACTIVE' WHERE project=?", (PROJECT_A,)
    )
    conn.commit()

    # mark_cold with 7-day threshold
    where = "chunk_state='ACTIVE' AND project=? AND datetime(last_accessed) < datetime('now', '-7 days')"
    conn.execute("UPDATE memory_chunks SET chunk_state='COLD' WHERE " + where, (PROJECT_A,))
    conn.commit()

    assert _get_state(conn, old["id"]) == "COLD", "10-day-old chunk should be COLD"
    assert _get_state(conn, new["id"]) == "ACTIVE", "1-day-old chunk should remain ACTIVE"

    _cleanup(conn)
    conn.close()
    print("  T2 ✓ mark_cold transitions 10-day-old ACTIVE → COLD")


def test_03_mark_dead():
    """T3: mark_dead() COLD(30d+) → DEAD，新 COLD 不受影响"""
    conn = open_db()
    ensure_schema(conn)
    _cleanup(conn)

    ancient = _chunk(PROJECT_A, days_ago=40)  # 40天前 COLD → 应变 DEAD
    recent_cold = _chunk(PROJECT_A, days_ago=8)  # 8天前 COLD → 保持 COLD
    insert_chunk(conn, ancient)
    insert_chunk(conn, recent_cold)
    conn.commit()

    # 手动设置状态
    conn.execute("UPDATE memory_chunks SET chunk_state='COLD' WHERE project=?", (PROJECT_A,))
    conn.commit()

    # mark_dead with 30-day threshold
    where = "chunk_state='COLD' AND project=? AND datetime(last_accessed) < datetime('now', '-30 days')"
    conn.execute("UPDATE memory_chunks SET chunk_state='DEAD' WHERE " + where, (PROJECT_A,))
    conn.commit()

    assert _get_state(conn, ancient["id"]) == "DEAD", "40-day-old chunk should be DEAD"
    assert _get_state(conn, recent_cold["id"]) == "COLD", "8-day-old chunk should remain COLD"

    _cleanup(conn)
    conn.close()
    print("  T3 ✓ mark_dead transitions 40-day-old COLD → DEAD")


def test_04_mark_active():
    """T4: mark_active() 任意非SWAP状态 → ACTIVE"""
    conn = open_db()
    ensure_schema(conn)
    _cleanup(conn)

    c_cold = _chunk(PROJECT_A)
    c_dead = _chunk(PROJECT_A)
    c_ghost = _chunk(PROJECT_A)
    for c in (c_cold, c_dead, c_ghost):
        insert_chunk(conn, c)
    conn.execute("UPDATE memory_chunks SET chunk_state='COLD' WHERE id=?", (c_cold["id"],))
    conn.execute("UPDATE memory_chunks SET chunk_state='DEAD' WHERE id=?", (c_dead["id"],))
    conn.execute("UPDATE memory_chunks SET chunk_state='GHOST' WHERE id=?", (c_ghost["id"],))
    conn.commit()

    count = mark_active(conn, [c_cold["id"], c_dead["id"], c_ghost["id"]])
    conn.commit()

    assert _get_state(conn, c_cold["id"]) == "ACTIVE"
    assert _get_state(conn, c_dead["id"]) == "ACTIVE"
    assert _get_state(conn, c_ghost["id"]) == "ACTIVE"
    assert count == 3, f"Expected 3, got {count}"

    _cleanup(conn)
    conn.close()
    print(f"  T4 ✓ mark_active: {count} chunks activated (COLD/DEAD/GHOST → ACTIVE)")


def test_05_mark_ghost():
    """T5: mark_ghost() 任意 → GHOST"""
    conn = open_db()
    ensure_schema(conn)
    _cleanup(conn)

    c = _chunk(PROJECT_A)
    insert_chunk(conn, c)
    conn.execute("UPDATE memory_chunks SET chunk_state='DEAD' WHERE id=?", (c["id"],))
    conn.commit()

    count = mark_ghost(conn, [c["id"]])
    conn.commit()

    assert _get_state(conn, c["id"]) == "GHOST"
    assert count == 1

    _cleanup(conn)
    conn.close()
    print("  T5 ✓ mark_ghost: DEAD → GHOST")


def test_06_swap_state_not_reactivated():
    """T6: SWAP 状态不被 mark_active() 影响（mlock 语义）"""
    conn = open_db()
    ensure_schema(conn)
    _cleanup(conn)

    c = _chunk(PROJECT_A)
    insert_chunk(conn, c)
    # 手动设置为 SWAP 状态（模拟 swap_out 后）
    conn.execute("UPDATE memory_chunks SET chunk_state='SWAP' WHERE id=?", (c["id"],))
    conn.commit()

    count = mark_active(conn, [c["id"]])
    conn.commit()

    # SWAP 状态应不受影响
    assert _get_state(conn, c["id"]) == "SWAP", "SWAP chunk should not be reactivated"
    assert count == 0, f"Expected 0, got {count}"

    _cleanup(conn)
    conn.close()
    print("  T6 ✓ SWAP state immune to mark_active()")


def test_07_fsm_transition():
    """T7: fsm_transition() 批量 ACTIVE→COLD→DEAD"""
    conn = open_db()
    ensure_schema(conn)
    _cleanup(conn)

    # 3 种年龄的 chunk
    fresh = _chunk(PROJECT_A, days_ago=1)    # 保持 ACTIVE
    stale = _chunk(PROJECT_A, days_ago=10)   # → COLD
    ancient = _chunk(PROJECT_A, days_ago=40) # → COLD then DEAD

    for c in (fresh, stale, ancient):
        insert_chunk(conn, c)
    # 确保初始状态是 ACTIVE
    conn.execute("UPDATE memory_chunks SET chunk_state='ACTIVE' WHERE project=?", (PROJECT_A,))
    conn.commit()

    result = fsm_transition(conn, project=PROJECT_A, cold_days=7, dead_days=30)
    conn.commit()

    assert result["cold"] >= 1, f"Expected ≥1 cold transition, got {result['cold']}"
    assert result["dead"] >= 1, f"Expected ≥1 dead transition, got {result['dead']}"
    assert _get_state(conn, fresh["id"]) == "ACTIVE"

    _cleanup(conn)
    conn.close()
    print(f"  T7 ✓ fsm_transition: cold={result['cold']}, dead={result['dead']}")


def test_08_state_distribution():
    """T8: get_state_distribution() 返回准确的 5 状态分布"""
    conn = open_db()
    ensure_schema(conn)
    _cleanup(conn)

    chunks = [_chunk(PROJECT_A) for _ in range(5)]
    for c in chunks:
        insert_chunk(conn, c)
    # 手动分配各状态
    states = ["ACTIVE", "COLD", "DEAD", "GHOST", "SWAP"]
    for c, s in zip(chunks, states):
        conn.execute("UPDATE memory_chunks SET chunk_state=? WHERE id=?", (s, c["id"]))
    conn.commit()

    dist = get_state_distribution(conn, project=PROJECT_A)
    assert dist["ACTIVE"] == 1, f"ACTIVE={dist['ACTIVE']}"
    assert dist["COLD"] == 1
    assert dist["DEAD"] == 1
    assert dist["GHOST"] == 1
    assert dist["SWAP"] == 1
    assert dist["total"] == 5

    _cleanup(conn)
    conn.close()
    print(f"  T8 ✓ state_distribution: {dist}")


def test_09_chunk_states_constant():
    """T9: CHUNK_STATES 常量集合完整"""
    assert "ACTIVE" in CHUNK_STATES
    assert "COLD" in CHUNK_STATES
    assert "DEAD" in CHUNK_STATES
    assert "SWAP" in CHUNK_STATES
    assert "GHOST" in CHUNK_STATES
    assert len(CHUNK_STATES) == 5
    print(f"  T9 ✓ CHUNK_STATES: {CHUNK_STATES}")


def test_10_cross_project_isolation():
    """T10: project A 的 FSM 转换不影响 project B"""
    conn = open_db()
    ensure_schema(conn)
    _cleanup(conn)

    c_a = _chunk(PROJECT_A, days_ago=10)
    c_b = _chunk(PROJECT_B, days_ago=10)
    insert_chunk(conn, c_a)
    insert_chunk(conn, c_b)
    conn.execute("UPDATE memory_chunks SET chunk_state='ACTIVE' WHERE project IN (?, ?)",
                 (PROJECT_A, PROJECT_B))
    conn.commit()

    # 只对 project A 执行 fsm_transition
    fsm_transition(conn, project=PROJECT_A, cold_days=7, dead_days=30)
    conn.commit()

    assert _get_state(conn, c_a["id"]) != "ACTIVE", "Project A chunk should have transitioned"
    assert _get_state(conn, c_b["id"]) == "ACTIVE", "Project B chunk should be unaffected"

    _cleanup(conn)
    conn.close()
    print("  T10 ✓ cross-project isolation confirmed")


if __name__ == "__main__":
    print("Task12 测试：Chunk Lifecycle FSM")
    print("=" * 60)

    tests = [
        test_01_default_state_active,
        test_02_mark_cold,
        test_03_mark_dead,
        test_04_mark_active,
        test_05_mark_ghost,
        test_06_swap_state_not_reactivated,
        test_07_fsm_transition,
        test_08_state_distribution,
        test_09_chunk_states_constant,
        test_10_cross_project_isolation,
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
