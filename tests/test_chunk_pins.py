#!/usr/bin/env python3
"""
迭代104 测试：chunk_pins — 项目级 pin（VMA per-process mlock 语义）

OS 类比：Linux mlock()/munlock() — 同一物理页在不同进程的 VMA 中有独立锁状态。
同一 chunk 在 project A 中 pinned，project B 中不受保护。

测试矩阵：
  T1: pin_chunk / unpin_chunk / is_pinned — 基本 CRUD
  T2: get_pinned_chunks — 列出 pinned chunks（含元数据）
  T3: get_pinned_ids — 高效批量查询（set）
  T4: stale reclaim 跳过 pinned chunks（soft + hard 均保护）
  T5: damon_scan 跳过 pinned chunks（DEAD/COLD）
  T6: evict_lowest_retention 跳过 hard pin，不跳过 soft pin
  T7: 跨 project 隔离 — project A pin 不影响 project B 淘汰
  T8: pin 不存在的 chunk 返回 False
  T9: unpin 未 pin 的 chunk 返回 False
  T10: extractor import 验证（pin_chunk 导入正常）
"""
import sys
import json
import os
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

import tmpfs  # noqa: F401
from store import (
    open_db, ensure_schema, insert_chunk, delete_chunks,
    pin_chunk, unpin_chunk, is_pinned, get_pinned_chunks, get_pinned_ids,
    evict_lowest_retention,
)
from store_mm import _reclaim_stale_chunks, damon_scan
from config import get as _sysctl

TEST_PROJECT_A = f"test_pins_a_{uuid.uuid4().hex[:6]}"
TEST_PROJECT_B = f"test_pins_b_{uuid.uuid4().hex[:6]}"


def _chunk(project: str, importance: float = 0.5, days_ago: int = 0,
           chunk_type: str = "decision") -> dict:
    now = datetime.now(timezone.utc)
    ts = (now - timedelta(days=days_ago)).isoformat()
    return {
        "id": str(uuid.uuid4()),
        "created_at": ts,
        "updated_at": ts,
        "project": project,
        "source_session": "test",
        "chunk_type": chunk_type,
        "content": f"content_{uuid.uuid4().hex[:8]}",
        "summary": f"summary_{uuid.uuid4().hex}",
        "tags": json.dumps(["test"]),
        "importance": importance,
        "retrievability": 0.3,
        "last_accessed": ts,
        "feishu_url": None,
    }


def _cleanup(conn, *projects):
    for p in projects:
        conn.execute("DELETE FROM memory_chunks WHERE project=?", (p,))
        conn.execute("DELETE FROM chunk_pins WHERE project=?", (p,))
    conn.commit()


# ── T1: 基本 CRUD ──
def test_basic_crud():
    conn = open_db()
    ensure_schema(conn)
    _cleanup(conn, TEST_PROJECT_A)

    c = _chunk(TEST_PROJECT_A)
    insert_chunk(conn, c)
    conn.commit()

    # 初始：未 pin
    assert is_pinned(conn, c["id"], TEST_PROJECT_A) is None

    # soft pin
    r = pin_chunk(conn, c["id"], TEST_PROJECT_A, "soft")
    conn.commit()
    assert r is True
    assert is_pinned(conn, c["id"], TEST_PROJECT_A) == "soft"

    # upsert → hard pin
    pin_chunk(conn, c["id"], TEST_PROJECT_A, "hard")
    conn.commit()
    assert is_pinned(conn, c["id"], TEST_PROJECT_A) == "hard"

    # unpin
    r2 = unpin_chunk(conn, c["id"], TEST_PROJECT_A)
    conn.commit()
    assert r2 is True
    assert is_pinned(conn, c["id"], TEST_PROJECT_A) is None

    _cleanup(conn, TEST_PROJECT_A)
    conn.close()
    print("  T1 basic CRUD ✓")


# ── T2: get_pinned_chunks ──
def test_get_pinned_chunks():
    conn = open_db()
    ensure_schema(conn)
    _cleanup(conn, TEST_PROJECT_A)

    c1 = _chunk(TEST_PROJECT_A, importance=0.9, chunk_type="design_constraint")
    c2 = _chunk(TEST_PROJECT_A, importance=0.7, chunk_type="decision")
    c3 = _chunk(TEST_PROJECT_A, importance=0.5)
    for c in (c1, c2, c3):
        insert_chunk(conn, c)
    pin_chunk(conn, c1["id"], TEST_PROJECT_A, "hard")
    pin_chunk(conn, c2["id"], TEST_PROJECT_A, "soft")
    conn.commit()

    all_pinned = get_pinned_chunks(conn, TEST_PROJECT_A)
    assert len(all_pinned) == 2, f"Expected 2, got {len(all_pinned)}"

    hard_only = get_pinned_chunks(conn, TEST_PROJECT_A, pin_type="hard")
    assert len(hard_only) == 1
    assert hard_only[0]["chunk_id"] == c1["id"]
    assert hard_only[0]["chunk_type"] == "design_constraint"

    soft_only = get_pinned_chunks(conn, TEST_PROJECT_A, pin_type="soft")
    assert len(soft_only) == 1

    _cleanup(conn, TEST_PROJECT_A)
    conn.close()
    print("  T2 get_pinned_chunks ✓")


# ── T3: get_pinned_ids ──
def test_get_pinned_ids():
    conn = open_db()
    ensure_schema(conn)
    _cleanup(conn, TEST_PROJECT_A)

    ids = []
    for _ in range(4):
        c = _chunk(TEST_PROJECT_A)
        insert_chunk(conn, c)
        ids.append(c["id"])

    pin_chunk(conn, ids[0], TEST_PROJECT_A, "hard")
    pin_chunk(conn, ids[1], TEST_PROJECT_A, "soft")
    pin_chunk(conn, ids[2], TEST_PROJECT_A, "hard")
    conn.commit()

    all_ids = get_pinned_ids(conn, TEST_PROJECT_A)
    assert all_ids == {ids[0], ids[1], ids[2]}
    assert ids[3] not in all_ids

    hard_ids = get_pinned_ids(conn, TEST_PROJECT_A, pin_type="hard")
    assert hard_ids == {ids[0], ids[2]}

    soft_ids = get_pinned_ids(conn, TEST_PROJECT_A, pin_type="soft")
    assert soft_ids == {ids[1]}

    _cleanup(conn, TEST_PROJECT_A)
    conn.close()
    print("  T3 get_pinned_ids ✓")


# ── T4: stale reclaim 跳过 pinned chunks ──
def test_stale_reclaim_skips_pinned():
    conn = open_db()
    ensure_schema(conn)
    _cleanup(conn, TEST_PROJECT_A)

    stale_days = _sysctl("kswapd.stale_days")

    # 创建 stale + pinned（soft pin 也保护 stale reclaim）
    pinned_stale = _chunk(TEST_PROJECT_A, importance=0.3, days_ago=stale_days + 20)
    insert_chunk(conn, pinned_stale)
    pin_chunk(conn, pinned_stale["id"], TEST_PROJECT_A, "soft")

    # 创建 stale + unpinned（应被回收）
    unpinned_stale = _chunk(TEST_PROJECT_A, importance=0.3, days_ago=stale_days + 20)
    insert_chunk(conn, unpinned_stale)

    conn.commit()

    reclaimed = _reclaim_stale_chunks(conn, TEST_PROJECT_A, stale_days, max_reclaim=10)

    assert unpinned_stale["id"] in reclaimed, "Unpinned stale chunk should be reclaimed"
    assert pinned_stale["id"] not in reclaimed, "Pinned stale chunk should survive"

    # 验证 pinned chunk 仍在主表
    row = conn.execute(
        "SELECT id FROM memory_chunks WHERE id=?", (pinned_stale["id"],)
    ).fetchone()
    assert row is not None, "Pinned chunk should still be in memory_chunks"

    _cleanup(conn, TEST_PROJECT_A)
    conn.close()
    print("  T4 stale reclaim skips pinned ✓")


# ── T5: damon_scan 跳过 pinned chunks ──
def test_damon_skips_pinned():
    conn = open_db()
    ensure_schema(conn)
    _cleanup(conn, TEST_PROJECT_A)

    dead_age_days = _sysctl("damon.dead_age_days")
    dead_imp_max = _sysctl("damon.dead_importance_max")

    # 创建 DEAD 条件 chunk（access=0, old, low importance）+ pinned
    pinned_dead = _chunk(TEST_PROJECT_A,
                         importance=dead_imp_max - 0.1,
                         days_ago=dead_age_days + 5)
    insert_chunk(conn, pinned_dead)
    pin_chunk(conn, pinned_dead["id"], TEST_PROJECT_A, "hard")

    # 创建 DEAD + unpinned
    unpinned_dead = _chunk(TEST_PROJECT_A,
                           importance=dead_imp_max - 0.1,
                           days_ago=dead_age_days + 5)
    insert_chunk(conn, unpinned_dead)

    conn.commit()

    result = damon_scan(conn, TEST_PROJECT_A)

    # pinned DEAD chunk 不应被 swap out（仍在主表）
    row = conn.execute(
        "SELECT id FROM memory_chunks WHERE id=?", (pinned_dead["id"],)
    ).fetchone()
    assert row is not None, "Hard-pinned DEAD chunk should not be swapped out"

    _cleanup(conn, TEST_PROJECT_A)
    conn.close()
    print(f"  T5 damon skips pinned ✓ (swapped_dead={result['actions']['swapped_dead']})")


# ── T6: evict_lowest_retention — hard pin 保护，soft pin 不保护 ──
def test_evict_respects_pin_types():
    conn = open_db()
    ensure_schema(conn)
    _cleanup(conn, TEST_PROJECT_A)

    # 创建 3 个低分 chunk
    hard_pinned = _chunk(TEST_PROJECT_A, importance=0.1)
    soft_pinned = _chunk(TEST_PROJECT_A, importance=0.1)
    unpinned = _chunk(TEST_PROJECT_A, importance=0.1)

    for c in (hard_pinned, soft_pinned, unpinned):
        insert_chunk(conn, c)

    pin_chunk(conn, hard_pinned["id"], TEST_PROJECT_A, "hard")
    pin_chunk(conn, soft_pinned["id"], TEST_PROJECT_A, "soft")
    conn.commit()

    evicted = evict_lowest_retention(conn, TEST_PROJECT_A, count=3)

    assert hard_pinned["id"] not in evicted, "Hard-pinned chunk must not be evicted"
    # soft pin 不保护 kswapd 硬淘汰
    # unpinned or soft_pinned should be evicted (at least one)
    evicted_set = set(evicted)
    assert (soft_pinned["id"] in evicted_set or unpinned["id"] in evicted_set), \
        "At least one of soft_pinned/unpinned should be evicted"

    _cleanup(conn, TEST_PROJECT_A)
    conn.close()
    print(f"  T6 evict respects pin types ✓ (evicted={len(evicted)}, hard_pin_survived=True)")


# ── T7: 跨 project 隔离 ──
def test_cross_project_isolation():
    conn = open_db()
    ensure_schema(conn)
    _cleanup(conn, TEST_PROJECT_A, TEST_PROJECT_B)

    # 同一 chunk 在 A 中 pinned，B 中不受保护
    c = _chunk(TEST_PROJECT_A, importance=0.1)
    # 需要同一个 chunk 也在 project B 存在（但实际场景 chunk 属于特定 project）
    # 这里测试 pin 隔离：project A 的 pin 不影响 project B 的淘汰查询
    cB = _chunk(TEST_PROJECT_B, importance=0.1)
    insert_chunk(conn, c)
    insert_chunk(conn, cB)

    pin_chunk(conn, c["id"], TEST_PROJECT_A, "hard")
    conn.commit()

    # project A 的 pinned_ids 不包含 cB
    a_pinned = get_pinned_ids(conn, TEST_PROJECT_A)
    assert c["id"] in a_pinned
    assert cB["id"] not in a_pinned

    # project B 的 pinned_ids 为空
    b_pinned = get_pinned_ids(conn, TEST_PROJECT_B)
    assert c["id"] not in b_pinned, "Project A pin should not bleed into project B"
    assert len(b_pinned) == 0

    _cleanup(conn, TEST_PROJECT_A, TEST_PROJECT_B)
    conn.close()
    print("  T7 cross-project isolation ✓")


# ── T8: pin 不存在的 chunk ──
def test_pin_nonexistent_chunk():
    conn = open_db()
    ensure_schema(conn)

    fake_id = str(uuid.uuid4())
    r = pin_chunk(conn, fake_id, TEST_PROJECT_A)
    assert r is False, "Pinning nonexistent chunk should return False"

    conn.close()
    print("  T8 pin nonexistent chunk returns False ✓")


# ── T9: unpin 未 pin 的 chunk ──
def test_unpin_not_pinned():
    conn = open_db()
    ensure_schema(conn)
    _cleanup(conn, TEST_PROJECT_A)

    c = _chunk(TEST_PROJECT_A)
    insert_chunk(conn, c)
    conn.commit()

    r = unpin_chunk(conn, c["id"], TEST_PROJECT_A)
    assert r is False, "Unpinning not-pinned chunk should return False"

    _cleanup(conn, TEST_PROJECT_A)
    conn.close()
    print("  T9 unpin not-pinned chunk returns False ✓")


# ── T10: extractor import 验证 ──
def test_extractor_imports_pin():
    ext_path = _ROOT / "hooks" / "extractor.py"
    code = ext_path.read_text()
    assert "pin_chunk" in code, "extractor.py should import pin_chunk"
    assert "pin_chunk(conn, cid, project" in code, "extractor.py should call pin_chunk"
    print("  T10 extractor imports pin_chunk ✓")


if __name__ == "__main__":
    print("迭代104 测试：chunk_pins — 项目级 pin")
    print("=" * 60)

    tests = [
        test_basic_crud,
        test_get_pinned_chunks,
        test_get_pinned_ids,
        test_stale_reclaim_skips_pinned,
        test_damon_skips_pinned,
        test_evict_respects_pin_types,
        test_cross_project_isolation,
        test_pin_nonexistent_chunk,
        test_unpin_not_pinned,
        test_extractor_imports_pin,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            import traceback
            print(f"  FAIL {test.__name__}: {e}")
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"结果：{passed}/{passed + failed} 通过")
    if failed:
        sys.exit(1)
