#!/usr/bin/env python3
"""
迭代38 测试：OOM Score — Per-Chunk 淘汰优先级调控

OS 类比：Linux OOM Killer oom_score_adj (2003→2010)
验证：
  T1 set/get oom_adj 基本 CRUD
  T2 batch_set_oom_adj 批量设置
  T3 oom_adj=-1000 的 chunk 不被 evict_lowest_retention 淘汰
  T4 oom_adj=+1000 的 chunk 在淘汰排序中最低（优先淘汰）
  T5 oom_adj=-1000 的 chunk 不被 swap_out（mlock 语义）
  T6 _reclaim_stale_chunks 尊重 oom_adj 保护
  T7 get_protected_chunks 列出所有受保护 chunk
  T8 proc_stats 包含 oom_score 统计
  T9 oom_adj 范围钳位（-1000 ~ +1000）
  T10 默认 oom_adj=0 时行为完全向后兼容
"""
import sys
import json
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).parent))

import tmpfs  # noqa: F401 — tmpfs isolation (iter54), must precede store import
from store import (
    open_db, ensure_schema, insert_chunk, delete_chunks,
    set_oom_adj, get_oom_adj, batch_set_oom_adj, get_protected_chunks,
    evict_lowest_retention, swap_out, swap_in, proc_stats,
    get_project_chunk_count, OOM_ADJ_MIN, OOM_ADJ_PROTECTED,
    OOM_ADJ_DEFAULT, OOM_ADJ_PREFER, OOM_ADJ_MAX,
    _reclaim_stale_chunks,
)
from schema import MemoryChunk

PROJECT = "test_oom_score"


def _make_chunk(suffix: str, importance: float = 0.5, chunk_type: str = "decision",
                age_days: int = 0) -> dict:
    """创建测试用 chunk dict。"""
    now = datetime.now(timezone.utc) - timedelta(days=age_days)
    chunk = MemoryChunk(
        project=PROJECT,
        source_session="test",
        chunk_type=chunk_type,
        content=f"[{chunk_type}] test content {suffix}",
        summary=f"test summary {suffix}",
        tags=[chunk_type, PROJECT],
        importance=importance,
        retrievability=0.5,
    )
    d = chunk.to_dict()
    d["created_at"] = now.isoformat()
    d["last_accessed"] = now.isoformat()
    d["updated_at"] = now.isoformat()
    return d


def _setup_db():
    """创建内存数据库，初始化 schema。"""
    conn = open_db(Path(":memory:"))
    ensure_schema(conn)
    return conn


def _cleanup(conn, project=PROJECT):
    """清理测试数据。"""
    conn.execute("DELETE FROM memory_chunks WHERE project=?", (project,))
    conn.execute("DELETE FROM swap_chunks WHERE project=?", (project,))
    conn.commit()


def test_01_set_get_oom_adj():
    """T1: set/get oom_adj 基本 CRUD"""
    conn = _setup_db()
    chunk = _make_chunk("t1")
    insert_chunk(conn, chunk)
    conn.commit()

    # 默认 oom_adj = 0
    assert get_oom_adj(conn, chunk["id"]) == 0

    # 设置 oom_adj = -500
    assert set_oom_adj(conn, chunk["id"], -500) is True
    assert get_oom_adj(conn, chunk["id"]) == -500

    # 设置 oom_adj = 1000
    assert set_oom_adj(conn, chunk["id"], 1000) is True
    assert get_oom_adj(conn, chunk["id"]) == 1000

    # 不存在的 chunk 返回 False / None
    assert set_oom_adj(conn, "nonexistent_id_xxx", 100) is False
    assert get_oom_adj(conn, "nonexistent_id_xxx") is None

    _cleanup(conn)
    conn.close()
    print("  T1 ✅ set/get oom_adj CRUD")


def test_02_batch_set():
    """T2: batch_set_oom_adj 批量设置"""
    conn = _setup_db()
    chunks = [_make_chunk(f"t2_{i}") for i in range(5)]
    for c in chunks:
        insert_chunk(conn, c)
    conn.commit()

    ids = [c["id"] for c in chunks]
    count = batch_set_oom_adj(conn, ids[:3], -500)
    conn.commit()
    assert count == 3

    # 验证前 3 个被设置，后 2 个仍为 0
    for i in range(3):
        assert get_oom_adj(conn, ids[i]) == -500
    for i in range(3, 5):
        assert get_oom_adj(conn, ids[i]) == 0

    # 空列表返回 0
    assert batch_set_oom_adj(conn, [], 100) == 0

    _cleanup(conn)
    conn.close()
    print("  T2 ✅ batch_set_oom_adj")


def test_03_evict_respects_oom_min():
    """T3: oom_adj=-1000 的 chunk 不被 evict_lowest_retention 淘汰"""
    conn = _setup_db()

    # 创建 5 个 chunk：2 个 oom_adj=-1000（保护），3 个默认
    protected = [_make_chunk(f"t3_prot_{i}", importance=0.3) for i in range(2)]
    normal = [_make_chunk(f"t3_norm_{i}", importance=0.3) for i in range(3)]
    for c in protected + normal:
        insert_chunk(conn, c)
    conn.commit()

    # 保护 2 个 chunk
    for c in protected:
        set_oom_adj(conn, c["id"], OOM_ADJ_MIN)
    conn.commit()

    # 淘汰 4 个（但只有 3 个可淘汰）
    evicted = evict_lowest_retention(conn, PROJECT, 4)
    conn.commit()

    # 被淘汰的只有 normal 的（最多 3 个）
    protected_ids = {c["id"] for c in protected}
    for eid in evicted:
        assert eid not in protected_ids, f"protected chunk {eid} was evicted!"

    assert len(evicted) == 3

    _cleanup(conn)
    conn.close()
    print("  T3 ✅ evict respects oom_adj=-1000")


def test_04_oom_positive_priority_eviction():
    """T4: oom_adj=+1000 的 chunk 在淘汰排序中优先被淘汰"""
    conn = _setup_db()

    # 3 个 chunk：相同 importance，但 oom_adj 不同
    high_prio = _make_chunk("t4_high", importance=0.6)  # oom_adj=+1000 → 优先淘汰
    normal = _make_chunk("t4_norm", importance=0.6)      # oom_adj=0
    low_prio = _make_chunk("t4_low", importance=0.6)     # oom_adj=-500 → 高保护

    for c in [high_prio, normal, low_prio]:
        insert_chunk(conn, c)
    conn.commit()

    set_oom_adj(conn, high_prio["id"], OOM_ADJ_MAX)
    set_oom_adj(conn, low_prio["id"], OOM_ADJ_PROTECTED)
    conn.commit()

    # 只淘汰 1 个——应该是 oom_adj=+1000 的
    evicted = evict_lowest_retention(conn, PROJECT, 1)
    conn.commit()

    assert len(evicted) == 1
    assert evicted[0] == high_prio["id"], \
        f"Expected {high_prio['id']} (oom_adj=+1000) to be evicted first, got {evicted[0]}"

    _cleanup(conn)
    conn.close()
    print("  T4 ✅ oom_adj=+1000 prioritized for eviction")


def test_05_swap_out_respects_mlock():
    """T5: oom_adj=-1000 的 chunk 不被 swap_out（mlock 语义）"""
    conn = _setup_db()

    # 创建 2 个 chunk：1 个 mlock，1 个正常
    locked = _make_chunk("t5_locked", importance=0.7)
    normal = _make_chunk("t5_normal", importance=0.7)
    for c in [locked, normal]:
        insert_chunk(conn, c)
    conn.commit()

    set_oom_adj(conn, locked["id"], OOM_ADJ_MIN)
    conn.commit()

    # swap_out 两个
    result = swap_out(conn, [locked["id"], normal["id"]])
    conn.commit()

    # locked 应该还在主表，normal 应该在 swap
    assert result["swapped_count"] == 1
    locked_exists = conn.execute(
        "SELECT id FROM memory_chunks WHERE id=?", (locked["id"],)
    ).fetchone()
    assert locked_exists is not None, "mlock chunk was swapped out!"

    normal_in_swap = conn.execute(
        "SELECT id FROM swap_chunks WHERE id=?", (normal["id"],)
    ).fetchone()
    assert normal_in_swap is not None

    _cleanup(conn)
    conn.close()
    print("  T5 ✅ swap_out respects mlock (oom_adj=-1000)")


def test_06_stale_reclaim_respects_oom():
    """T6: _reclaim_stale_chunks 尊重 oom_adj=-1000 保护"""
    conn = _setup_db()

    # 创建 3 个 stale chunk（100天前）：1 个受保护
    protected = _make_chunk("t6_prot", importance=0.5, age_days=100)
    stale1 = _make_chunk("t6_stale1", importance=0.5, age_days=100)
    stale2 = _make_chunk("t6_stale2", importance=0.5, age_days=100)
    for c in [protected, stale1, stale2]:
        insert_chunk(conn, c)
    conn.commit()

    set_oom_adj(conn, protected["id"], OOM_ADJ_MIN)
    conn.commit()

    evicted = _reclaim_stale_chunks(conn, PROJECT, stale_days=90, max_reclaim=10)
    conn.commit()

    # 受保护的不应被回收
    assert protected["id"] not in evicted
    assert len(evicted) == 2

    _cleanup(conn)
    conn.close()
    print("  T6 ✅ stale reclaim respects oom_adj protection")


def test_07_get_protected_chunks():
    """T7: get_protected_chunks 列出所有受保护 chunk"""
    conn = _setup_db()

    protected1 = _make_chunk("t7_p1", importance=0.8)
    protected2 = _make_chunk("t7_p2", importance=0.9)
    normal = _make_chunk("t7_n", importance=0.5)
    for c in [protected1, protected2, normal]:
        insert_chunk(conn, c)
    conn.commit()

    set_oom_adj(conn, protected1["id"], OOM_ADJ_MIN)
    set_oom_adj(conn, protected2["id"], OOM_ADJ_PROTECTED)
    conn.commit()

    protected = get_protected_chunks(conn, PROJECT)
    assert len(protected) == 2
    ids = {p["id"] for p in protected}
    assert protected1["id"] in ids
    assert protected2["id"] in ids
    assert normal["id"] not in ids

    # 按 oom_adj 排序（-1000 在前）
    assert protected[0]["oom_adj"] <= protected[1]["oom_adj"]

    _cleanup(conn)
    conn.close()
    print("  T7 ✅ get_protected_chunks")


def test_08_proc_stats_oom_score():
    """T8: proc_stats 包含 oom_score 统计"""
    conn = _setup_db()

    c1 = _make_chunk("t8_locked", importance=0.9)
    c2 = _make_chunk("t8_protected", importance=0.8)
    c3 = _make_chunk("t8_normal", importance=0.5)
    c4 = _make_chunk("t8_disposable", importance=0.3)
    for c in [c1, c2, c3, c4]:
        insert_chunk(conn, c)
    conn.commit()

    set_oom_adj(conn, c1["id"], OOM_ADJ_MIN)      # locked
    set_oom_adj(conn, c2["id"], OOM_ADJ_PROTECTED) # protected
    set_oom_adj(conn, c4["id"], OOM_ADJ_PREFER)    # disposable
    conn.commit()

    stats = proc_stats(conn)
    oom = stats.get("oom_score", {})
    assert oom["locked"] == 1       # oom_adj <= -1000
    assert oom["protected"] == 2    # oom_adj < 0 (包含 locked)
    assert oom["disposable"] == 1   # oom_adj > 0
    assert oom["default"] == 1      # oom_adj = 0

    _cleanup(conn)
    conn.close()
    print("  T8 ✅ proc_stats oom_score")


def test_09_oom_adj_clamping():
    """T9: oom_adj 范围钳位（-1000 ~ +1000）"""
    conn = _setup_db()
    chunk = _make_chunk("t9")
    insert_chunk(conn, chunk)
    conn.commit()

    # 超出范围应被钳位
    set_oom_adj(conn, chunk["id"], -2000)
    assert get_oom_adj(conn, chunk["id"]) == -1000

    set_oom_adj(conn, chunk["id"], 5000)
    assert get_oom_adj(conn, chunk["id"]) == 1000

    _cleanup(conn)
    conn.close()
    print("  T9 ✅ oom_adj clamping")


def test_10_backward_compatible():
    """T10: 默认 oom_adj=0 时行为完全向后兼容"""
    conn = _setup_db()

    # 创建 5 个 chunk，全部默认 oom_adj=0
    chunks = [_make_chunk(f"t10_{i}", importance=0.3 + i * 0.1) for i in range(5)]
    for c in chunks:
        insert_chunk(conn, c)
    conn.commit()

    # 验证所有 oom_adj 默认为 0
    for c in chunks:
        assert get_oom_adj(conn, c["id"]) == 0

    # 淘汰 2 个——应该与无 oom_adj 时行为一致（按 retention_score 升序）
    evicted = evict_lowest_retention(conn, PROJECT, 2)
    conn.commit()

    # 被淘汰的应该是 importance 最低的
    assert len(evicted) == 2
    evicted_set = set(evicted)
    # importance 0.3 和 0.4 的应该被淘汰
    assert chunks[0]["id"] in evicted_set or chunks[1]["id"] in evicted_set

    _cleanup(conn)
    conn.close()
    print("  T10 ✅ backward compatible (default oom_adj=0)")


# ── 常量验证 ──
def test_constants():
    """验证 OOM_ADJ 常量值正确"""
    assert OOM_ADJ_MIN == -1000
    assert OOM_ADJ_PROTECTED == -500
    assert OOM_ADJ_DEFAULT == 0
    assert OOM_ADJ_PREFER == 500
    assert OOM_ADJ_MAX == 1000
    print("  T0 ✅ OOM_ADJ constants")


if __name__ == "__main__":
    t_start = time.time()
    print("迭代38: OOM Score — Per-Chunk 淘汰优先级调控")
    print("=" * 50)
    test_constants()
    test_01_set_get_oom_adj()
    test_02_batch_set()
    test_03_evict_respects_oom_min()
    test_04_oom_positive_priority_eviction()
    test_05_swap_out_respects_mlock()
    test_06_stale_reclaim_respects_oom()
    test_07_get_protected_chunks()
    test_08_proc_stats_oom_score()
    test_09_oom_adj_clamping()
    test_10_backward_compatible()
    elapsed = (time.time() - t_start) * 1000
    print(f"\n全部 11/11 测试通过 ✅  avg {elapsed/11:.2f}ms/test  total {elapsed:.1f}ms")
