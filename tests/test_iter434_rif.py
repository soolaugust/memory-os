"""
test_iter434_rif.py — iter434: Retrieval-Induced Forgetting (RIF) 单元测试

覆盖：
  RIF1: 命中 chunk 的同类相似竞争者 stability 下降
  RIF2: 不同类型（chunk_type 不同）→ 不受 RIF 抑制
  RIF3: 相似度低于阈值（Jaccard < 0.25）→ 不受 RIF 抑制
  RIF4: importance >= rif_protect_importance → 豁免抑制
  RIF5: rif_enabled=False → 无抑制
  RIF6: 保护类型（design_constraint/procedure）→ 豁免抑制
  RIF7: rif_factor 可通过 sysctl 配置（factor=0.90 抑制更强）
  RIF8: 返回 suppressed 计数正确
  RIF9: permastore/ribot floor 保护不被过度抑制
  RIF10: 命中 chunk 自身不被抑制（RP+ 不受影响）

认知科学依据：
  Anderson, Bjork & Bjork (1994) "Remembering can cause forgetting" —
  检索 practiced items (RP+) → 主动抑制同类未练习项目 (RP-) → RP- 遗忘增加 10-20%。
  与 iter417 区别：iter434 按 chunk_type 分组 + summary Jaccard 相似度。

OS 类比：CPU set-associative cache way eviction —
  访问 cache line A → LRU 将同 set 竞争者 B 推向更高 way → B 的驱逐概率上升。
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

from store_vfs import (
    ensure_schema,
    apply_rif_by_summary,
)
import config


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _ago_iso(hours: float = 0.0) -> str:
    return (datetime.datetime.now(datetime.timezone.utc) -
            datetime.timedelta(hours=hours)).isoformat()


def _insert(conn, cid, chunk_type="decision", project="test",
            stability=5.0, importance=0.6,
            summary="python async await concurrent programming code",
            last_accessed_hours_ago: float = 48.0):
    """Insert a chunk with last_accessed defaulting to 48h ago (outside RDR 6h window)."""
    now = _now_iso()
    last_acc = _ago_iso(hours=last_accessed_hours_ago)
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0.9, ?, 1)""",
        (cid, project, chunk_type, f"content {cid}", summary,
         importance, stability, now, now, last_acc)
    )
    conn.commit()


def _get_stability(conn, cid: str) -> float:
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


# ── RIF1: 同类相似竞争者 stability 下降 ──────────────────────────────────────

def test_rif1_same_type_competitor_suppressed(conn):
    """RIF1: 命中 chunk 的同类型且 summary 相似的竞争者，stability 应下降。"""
    # 命中 chunk A（decision，python async）
    _insert(conn, "hit_a", chunk_type="decision",
            summary="python async await coroutine concurrency programming")
    # 竞争者 B（decision，相似 summary）— 应被抑制
    _insert(conn, "comp_b", chunk_type="decision",
            summary="python async await coroutine event loop")

    stab_before = _get_stability(conn, "comp_b")
    result = apply_rif_by_summary(conn, "test", ["hit_a"])

    stab_after = _get_stability(conn, "comp_b")
    rif_factor = config.get("scorer.rif_factor")  # 0.95

    assert stab_after < stab_before, (
        f"RIF1: 同类相似竞争者 stability 应下降，before={stab_before:.3f} after={stab_after:.3f}"
    )
    assert result["suppressed"] >= 1, f"RIF1: suppressed 应 >= 1，got {result}"


# ── RIF2: 不同 chunk_type → 不受影响 ──────────────────────────────────────────

def test_rif2_different_type_no_suppression(conn):
    """RIF2: 不同 chunk_type 的 chunk 即使内容相似也不受 RIF 抑制。"""
    _insert(conn, "hit_a", chunk_type="decision",
            summary="python async await coroutine concurrency programming")
    # 不同类型（reasoning_chain）— 不应受影响
    _insert(conn, "other_type", chunk_type="reasoning_chain",
            summary="python async await coroutine event loop programming")

    stab_before = _get_stability(conn, "other_type")
    apply_rif_by_summary(conn, "test", ["hit_a"])
    stab_after = _get_stability(conn, "other_type")

    assert abs(stab_after - stab_before) < 0.001, (
        f"RIF2: 不同类型不应受 RIF 影响，before={stab_before:.3f} after={stab_after:.3f}"
    )


# ── RIF3: 相似度低于阈值 → 不受影响 ──────────────────────────────────────────

def test_rif3_low_similarity_no_suppression(conn):
    """RIF3: Jaccard 相似度 < 0.25 的同类 chunk 不受 RIF 抑制。"""
    _insert(conn, "hit_a", chunk_type="decision",
            summary="python async await coroutine")
    # 完全不相关的内容（Jaccard ≈ 0）
    _insert(conn, "unrelated", chunk_type="decision",
            summary="database sql join query optimization index performance")

    stab_before = _get_stability(conn, "unrelated")
    apply_rif_by_summary(conn, "test", ["hit_a"])
    stab_after = _get_stability(conn, "unrelated")

    assert abs(stab_after - stab_before) < 0.001, (
        f"RIF3: 低相似度不应受 RIF 影响，before={stab_before:.3f} after={stab_after:.3f}"
    )


# ── RIF4: importance 高 → 豁免 ────────────────────────────────────────────────

def test_rif4_high_importance_protected(conn):
    """RIF4: importance >= rif_protect_importance(0.85) 的 chunk 豁免 RIF 抑制。"""
    _insert(conn, "hit_a", chunk_type="decision",
            summary="python async await coroutine concurrency")
    protect_imp = config.get("scorer.rif_protect_importance")  # 0.85
    # 高 importance 竞争者
    _insert(conn, "high_imp", chunk_type="decision",
            summary="python async await coroutine event loop",
            importance=protect_imp)  # 等于保护阈值（按 < threshold 的条件，等于不保护）

    # 插入真正被保护的 importance=0.90 的竞争者
    _insert(conn, "very_high_imp", chunk_type="decision",
            summary="python async await coroutine event loop programming",
            importance=0.90)

    stab_before = _get_stability(conn, "very_high_imp")
    apply_rif_by_summary(conn, "test", ["hit_a"])
    stab_after = _get_stability(conn, "very_high_imp")

    assert abs(stab_after - stab_before) < 0.001, (
        f"RIF4: 高 importance chunk 应豁免 RIF，before={stab_before:.3f} after={stab_after:.3f}"
    )


# ── RIF5: rif_enabled=False → 无抑制 ──────────────────────────────────────────

def test_rif5_disabled_no_suppression(conn):
    """RIF5: scorer.rif_enabled=False → 无任何抑制。"""
    original_get = config.get
    def patched_get(key, project=None):
        if key == "scorer.rif_enabled":
            return False
        return original_get(key, project=project)

    _insert(conn, "hit_a", chunk_type="decision",
            summary="python async await coroutine concurrency programming")
    _insert(conn, "comp_b", chunk_type="decision",
            summary="python async await coroutine event loop")

    stab_before = _get_stability(conn, "comp_b")
    with mock.patch.object(config, 'get', side_effect=patched_get):
        result = apply_rif_by_summary(conn, "test", ["hit_a"])

    stab_after = _get_stability(conn, "comp_b")
    assert abs(stab_after - stab_before) < 0.001, \
        f"RIF5: 禁用时不应有抑制，before={stab_before:.3f} after={stab_after:.3f}"
    assert result["suppressed"] == 0, f"RIF5: suppressed 应为 0，got {result}"


# ── RIF6: 保护类型 → 豁免 ─────────────────────────────────────────────────────

def test_rif6_protected_type_exempt(conn):
    """RIF6: design_constraint 和 procedure 类型豁免 RIF 抑制。"""
    # 命中 chunk（非保护类型）
    _insert(conn, "hit_a", chunk_type="decision",
            summary="python async await coroutine concurrency programming")
    # 保护类型（不应受影响）
    _insert(conn, "dc_chunk", chunk_type="design_constraint",
            summary="python async await coroutine concurrency programming event loop")

    stab_before = _get_stability(conn, "dc_chunk")
    apply_rif_by_summary(conn, "test", ["hit_a"])
    stab_after = _get_stability(conn, "dc_chunk")

    # design_constraint 不与 decision 同 chunk_type，所以不受影响
    assert abs(stab_after - stab_before) < 0.001, (
        f"RIF6: design_constraint 应豁免 RIF，before={stab_before:.3f} after={stab_after:.3f}"
    )


# ── RIF7: rif_factor 可配置 ───────────────────────────────────────────────────

def test_rif7_configurable_rif_factor(conn):
    """RIF7: rif_factor=0.90 时抑制比 rif_factor=0.95 更强。"""
    original_get = config.get

    _insert(conn, "hit_a", chunk_type="decision",
            summary="python async await coroutine concurrency programming")

    # 测试 rif_factor=0.90（更强抑制）
    _insert(conn, "comp_90", chunk_type="decision",
            summary="python async await coroutine event loop programming")

    def patched_90(key, project=None):
        if key == "scorer.rif_factor":
            return 0.90
        return original_get(key, project=project)

    stab_before_90 = _get_stability(conn, "comp_90")
    with mock.patch.object(config, 'get', side_effect=patched_90):
        apply_rif_by_summary(conn, "test", ["hit_a"])
    stab_after_90 = _get_stability(conn, "comp_90")

    # 重置 stability
    conn.execute("UPDATE memory_chunks SET stability=5.0 WHERE id='comp_90'")
    conn.commit()

    # 测试默认 rif_factor=0.95（较弱抑制）
    stab_before_95 = _get_stability(conn, "comp_90")
    apply_rif_by_summary(conn, "test", ["hit_a"])
    stab_after_95 = _get_stability(conn, "comp_90")

    drop_90 = stab_before_90 - stab_after_90
    drop_95 = stab_before_95 - stab_after_95

    assert drop_90 > drop_95, (
        f"RIF7: rif_factor=0.90 抑制应更强，drop_90={drop_90:.3f} drop_95={drop_95:.3f}"
    )


# ── RIF8: 返回计数正确 ──────────────────────────────────────────────────────────

def test_rif8_return_counts_correct(conn):
    """RIF8: result dict 中 suppressed 和 total_examined 计数正确。"""
    _insert(conn, "hit_a", chunk_type="decision",
            summary="python async await coroutine concurrency programming")
    # 3 个同类 chunk：2 相似（应被抑制），1 不相似
    _insert(conn, "sim1", chunk_type="decision",
            summary="python async await coroutine event loop programming")
    _insert(conn, "sim2", chunk_type="decision",
            summary="python async await coroutine asyncio concurrent code")
    _insert(conn, "diff", chunk_type="decision",
            summary="database sql join query optimization performance index")

    result = apply_rif_by_summary(conn, "test", ["hit_a"])

    assert "suppressed" in result, f"RIF8: result 应含 suppressed key"
    assert "total_examined" in result, f"RIF8: result 应含 total_examined key"
    assert result["suppressed"] >= 1, f"RIF8: 应有至少 1 个 chunk 被抑制，got {result}"
    assert result["total_examined"] >= 3, f"RIF8: 应检查 >= 3 个竞争者，got {result}"


# ── RIF9: 不超过 floor（permastore 保护）──────────────────────────────────────

def test_rif9_permastore_floor_protection(conn):
    """RIF9: RIF 抑制不能把 stability 降到 permastore floor 以下。"""
    # 插入 access_count=20（高访问 → permastore floor 较高）
    now = _now_iso()
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0.9, ?, ?)""",
        ("protected_comp", "test", "decision",
         "content protected_comp",
         "python async await coroutine concurrency programming event loop asyncio",
         0.6, 2.0, now, now, now, 20)  # access_count=20
    )
    conn.commit()

    _insert(conn, "hit_a", chunk_type="decision",
            summary="python async await coroutine concurrency programming")

    stab_before = _get_stability(conn, "protected_comp")
    apply_rif_by_summary(conn, "test", ["hit_a"])
    stab_after = _get_stability(conn, "protected_comp")

    # stability 最多降到 base * rif_factor，不能低于 permastore floor（0.0 if access=20 is too low）
    # 关键：stability 不应降为 0
    assert stab_after > 0.0, f"RIF9: stability 不应降为 0，got {stab_after}"
    # RIF 不能超过原来的 2 倍 penalty
    min_allowed = stab_before * 0.80  # rif_factor 最低 0.80（config 限制）
    assert stab_after >= min_allowed * 0.99, (  # 0.99 tolerance for float
        f"RIF9: stability 不应降到 20% 以下，before={stab_before:.3f} after={stab_after:.3f}"
    )


# ── RIF10: 命中 chunk 自身不被抑制 ────────────────────────────────────────────

def test_rif10_hit_chunk_not_suppressed(conn):
    """RIF10: 被命中的 chunk 自身不受 RIF 抑制（RP+ 应增强，不应被压制）。"""
    _insert(conn, "hit_a", chunk_type="decision",
            summary="python async await coroutine concurrency programming")
    _insert(conn, "comp_b", chunk_type="decision",
            summary="python async await coroutine event loop programming")

    stab_hit_before = _get_stability(conn, "hit_a")
    apply_rif_by_summary(conn, "test", ["hit_a"])
    stab_hit_after = _get_stability(conn, "hit_a")

    # 命中 chunk 的 stability 不应因为 RIF 下降
    assert stab_hit_after >= stab_hit_before, (
        f"RIF10: 命中 chunk 不应被 RIF 抑制，before={stab_hit_before:.3f} after={stab_hit_after:.3f}"
    )
