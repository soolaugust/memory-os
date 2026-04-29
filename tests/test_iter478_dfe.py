"""
test_iter478_dfe.py — iter478: Directed Forgetting Effect 单元测试

覆盖：
  DF1: importance 下降 >= dfe_min_importance_drop(0.20) + 高相似邻居 → stability 降低
  DF2: importance 降幅 < dfe_min_importance_drop → 不触发 DFE
  DF3: dfe_enabled=False → 无遗忘传播
  DF4: 邻居相似度 < dfe_min_similarity(0.30) → 不受影响
  DF5: 衰减比例受 dfe_max_decay(0.10) 保护
  DF6: 受影响邻居数受 dfe_max_neighbors(8) 限制
  DF7: 衰减后 stability 不低于 0.1（下限保护）
  DF8: 直接调用 apply_directed_forgetting_effect → dfe_propagated=True

认知科学依据：
  Bjork (1972) 定向遗忘 — "忘记这项"指令使相关记忆也受到抑制（category inhibition）。
  Bjork & Woodward (1973): TBF 条目及其关联条目均受抑制。

OS 类比：Linux MADV_FREE — 进程标记页面为"懒惰释放"→ 内存压力时被驱逐。
"""
import sys
import sqlite3
import datetime
import unittest.mock as mock
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import tmpfs  # noqa

from store_vfs import ensure_schema, apply_directed_forgetting_effect
import config


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _utcnow():
    return datetime.datetime.now(datetime.timezone.utc)


def _insert_raw(conn, cid, content="", project="test", importance=0.6, stability=5.0):
    now_iso = _utcnow().isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count,
            encode_context, session_type_history)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (cid, project, "observation", content, "summary", importance, stability,
         now_iso, now_iso, 0.5, now_iso, 0, "test_ctx", "")
    )
    conn.commit()


def _get_stability(conn, cid: str) -> float:
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


# ── DF1: 高降幅 + 高相似邻居 → stability 降低 ───────────────────────────────────────────────

def test_df1_propagates_to_similar_chunk(conn):
    """DF1: importance 大幅下降 + 相似邻居 → 邻居 stability 降低。"""
    content_src = "memory allocator kernel slab buddy page frame vmalloc"
    content_nb = "memory allocator kernel slab buddy page cache mmap"  # 高 Jaccard

    _insert_raw(conn, "df1_src", content=content_src, importance=0.6)
    _insert_raw(conn, "df1_nb", content=content_nb, importance=0.6, stability=5.0)

    stab_nb_before = _get_stability(conn, "df1_nb")
    result = apply_directed_forgetting_effect(conn, "df1_src", "test",
                                               old_importance=0.8, new_importance=0.2)
    stab_nb_after = _get_stability(conn, "df1_nb")

    assert stab_nb_after < stab_nb_before, (
        f"DF1: 相似邻居 stability 应降低，before={stab_nb_before:.4f} after={stab_nb_after:.4f}"
    )
    assert result["dfe_propagated"] is True, f"DF1: dfe_propagated 应为 True，got {result}"


# ── DF2: 降幅不足 → 不触发 DFE ──────────────────────────────────────────────────────────────

def test_df2_small_drop_no_effect(conn):
    """DF2: importance 降幅 < dfe_min_importance_drop(0.20) → 不触发 DFE。"""
    content_src = "memory allocator kernel slab buddy page frame vmalloc"
    content_nb = "memory allocator kernel slab buddy page cache mmap"

    _insert_raw(conn, "df2_src", content=content_src)
    _insert_raw(conn, "df2_nb", content=content_nb, stability=5.0)

    stab_nb_before = _get_stability(conn, "df2_nb")
    result = apply_directed_forgetting_effect(conn, "df2_src", "test",
                                               old_importance=0.7, new_importance=0.6)  # 降幅 0.10 < 0.20
    stab_nb_after = _get_stability(conn, "df2_nb")

    assert abs(stab_nb_after - stab_nb_before) < 0.001, (
        f"DF2: 降幅不足时不应触发 DFE，before={stab_nb_before:.4f} after={stab_nb_after:.4f}"
    )
    assert result["dfe_propagated"] is False, f"DF2: dfe_propagated 应为 False"


# ── DF3: dfe_enabled=False → 无传播 ─────────────────────────────────────────────────────────

def test_df3_disabled_no_propagation(conn):
    """DF3: dfe_enabled=False → 无遗忘传播。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.dfe_enabled":
            return False
        return original_get(key, project=project)

    content_src = "memory allocator kernel slab buddy page frame vmalloc"
    content_nb = "memory allocator kernel slab buddy page cache mmap"

    _insert_raw(conn, "df3_src", content=content_src)
    _insert_raw(conn, "df3_nb", content=content_nb, stability=5.0)

    stab_nb_before = _get_stability(conn, "df3_nb")
    with mock.patch.object(config, 'get', side_effect=patched_get):
        result = apply_directed_forgetting_effect(conn, "df3_src", "test",
                                                   old_importance=0.8, new_importance=0.2)
    stab_nb_after = _get_stability(conn, "df3_nb")

    assert abs(stab_nb_after - stab_nb_before) < 0.001, (
        f"DF3: disabled 时不应传播，before={stab_nb_before:.4f} after={stab_nb_after:.4f}"
    )
    assert result["dfe_propagated"] is False, f"DF3: dfe_propagated 应为 False"


# ── DF4: 低相似度邻居不受影响 ────────────────────────────────────────────────────────────────

def test_df4_dissimilar_chunk_not_affected(conn):
    """DF4: Jaccard < dfe_min_similarity(0.30) 的邻居不受 DFE 影响。"""
    content_src = "memory allocator kernel slab buddy page frame vmalloc"
    content_nb = "neural network deep learning gradient descent backpropagation"  # 无重叠

    _insert_raw(conn, "df4_src", content=content_src)
    _insert_raw(conn, "df4_nb", content=content_nb, stability=5.0)

    stab_nb_before = _get_stability(conn, "df4_nb")
    apply_directed_forgetting_effect(conn, "df4_src", "test",
                                      old_importance=0.8, new_importance=0.1)
    stab_nb_after = _get_stability(conn, "df4_nb")

    assert abs(stab_nb_after - stab_nb_before) < 0.001, (
        f"DF4: 低相似度邻居不应受 DFE 影响，before={stab_nb_before:.4f} after={stab_nb_after:.4f}"
    )


# ── DF5: 衰减受 dfe_max_decay 保护 ──────────────────────────────────────────────────────────

def test_df5_max_decay_cap(conn):
    """DF5: DFE 衰减不超过 base × dfe_max_decay(0.10)。"""
    dfe_max_decay = config.get("store_vfs.dfe_max_decay")  # 0.10
    base = 5.0

    content_src = "memory allocator kernel slab buddy page frame vmalloc"
    content_nb = "memory allocator kernel slab buddy page cache mmap"

    _insert_raw(conn, "df5_src", content=content_src)
    _insert_raw(conn, "df5_nb", content=content_nb, stability=base)

    stab_before = _get_stability(conn, "df5_nb")
    apply_directed_forgetting_effect(conn, "df5_src", "test",
                                      old_importance=0.9, new_importance=0.1)
    stab_after = _get_stability(conn, "df5_nb")

    decay = stab_before - stab_after
    max_allowed = base * dfe_max_decay + 0.01
    assert decay <= max_allowed, (
        f"DF5: 衰减 {decay:.4f} 不应超过 max_decay 允许的 {max_allowed:.4f}，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert stab_after < stab_before, (
        f"DF5: 应有 DFE 衰减，before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── DF6: 受影响邻居数受 dfe_max_neighbors 限制 ──────────────────────────────────────────────

def test_df6_max_neighbors_limit(conn):
    """DF6: DFE 影响的邻居数不超过 dfe_max_neighbors(8)。"""
    content_src = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    _insert_raw(conn, "df6_src", content=content_src)

    # 插入 12 个高相似度邻居（超过 max_neighbors=8）
    for i in range(12):
        content_nb = f"alpha beta gamma delta epsilon zeta eta theta iota lambda_{i}"
        _insert_raw(conn, f"df6_nb_{i}", content=content_nb, stability=5.0)

    stab_before = {f"df6_nb_{i}": _get_stability(conn, f"df6_nb_{i}") for i in range(12)}
    apply_directed_forgetting_effect(conn, "df6_src", "test",
                                      old_importance=0.8, new_importance=0.1)
    stab_after = {f"df6_nb_{i}": _get_stability(conn, f"df6_nb_{i}") for i in range(12)}

    decayed_count = sum(1 for i in range(12)
                        if stab_after[f"df6_nb_{i}"] < stab_before[f"df6_nb_{i}"] - 0.001)

    max_neighbors = config.get("store_vfs.dfe_max_neighbors")
    assert decayed_count <= max_neighbors, (
        f"DF6: 受影响邻居数 {decayed_count} 不应超过 max_neighbors={max_neighbors}"
    )


# ── DF7: stability 下限 0.1 ──────────────────────────────────────────────────────────────────

def test_df7_stability_floor_01(conn):
    """DF7: DFE 衰减后邻居 stability 不低于 0.1。"""
    content_src = "memory allocator kernel slab buddy page frame vmalloc"
    content_nb = "memory allocator kernel slab buddy page cache mmap"

    _insert_raw(conn, "df7_src", content=content_src)
    _insert_raw(conn, "df7_nb", content=content_nb, stability=0.12)  # 接近下限

    apply_directed_forgetting_effect(conn, "df7_src", "test",
                                      old_importance=0.9, new_importance=0.0)
    stab = _get_stability(conn, "df7_nb")
    assert stab >= 0.1, f"DF7: stability 不应低于 0.1，got {stab:.4f}"


# ── DF8: 直接调用返回 dfe_propagated=True ────────────────────────────────────────────────────

def test_df8_direct_function_propagated(conn):
    """DF8: apply_directed_forgetting_effect 直接调用返回 dfe_propagated=True。"""
    content_src = "memory allocator kernel slab buddy page frame vmalloc"
    content_nb = "memory allocator kernel slab buddy page cache mmap"

    _insert_raw(conn, "df8_src", content=content_src, importance=0.6)
    _insert_raw(conn, "df8_nb", content=content_nb, importance=0.6, stability=5.0)

    stab_before = _get_stability(conn, "df8_nb")
    result = apply_directed_forgetting_effect(conn, "df8_src", "test",
                                               old_importance=0.9, new_importance=0.1)
    stab_after = _get_stability(conn, "df8_nb")

    assert result["dfe_propagated"] is True, f"DF8: dfe_propagated 应为 True，got {result}"
    assert result["dfe_neighbors_decayed"] >= 1, (
        f"DF8: dfe_neighbors_decayed 应 >= 1，got {result}"
    )
    assert stab_after < stab_before, (
        f"DF8: 邻居 stability 应降低，before={stab_before:.4f} after={stab_after:.4f}"
    )
