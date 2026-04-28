"""
test_iter435_rdr.py — iter435: Recency-Induced Decay Resistance (RDR) 单元测试

覆盖：
  RDR1: 近期访问 chunk（last_accessed < 6h）在 RIF 抑制中受保护
  RDR2: 旧访问 chunk（last_accessed > 6h）在 RIF 中正常被抑制
  RDR3: 近期访问但 importance < rdr_min_importance(0.5) → 不受 RDR 保护
  RDR4: rdr_enabled=False → 近期访问 chunk 也可被 RIF 抑制
  RDR5: rdr_window_hours 可通过 sysctl 配置（2h 窗口 vs 12h 窗口）
  RDR6: 保护边界精确（恰好在窗口内 vs 恰好在窗口外）
  RDR7: RDR 不影响 importance >= rif_protect_importance 的高重要性 chunk（已由 RIF 保护）
  RDR8: 跨 chunk_type 的 RDR 保护（decision + reasoning_chain 均受保护）
  RDR9: rdr_min_importance 边界（恰好等于阈值的 chunk 受保护）

认知科学依据：
  McGaugh (2000) Memory consolidation window —
  学习/检索后数小时内，海马体持续重放记忆痕迹（retrograde amnesia gradient 基础）。
  近期访问记忆处于 consolidation window，对遗忘干扰（包括 RIF 竞争性抑制）抵抗力最强。

OS 类比：Linux MGLRU young generation minimum age (min_lru_age) —
  刚被访问/提升到 young generation 的页面有 grace period（min_lru_age），
  kswapd 在此期间不执行 aging，防止"刚提升就被驱逐"的 LRU thrashing。
  memory-os：last_accessed < rdr_window_hours 的重要 chunk 豁免 RIF 的竞争性抑制。
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
            last_accessed_hours_ago=48.0,  # default: 48h ago (outside RDR window)
            summary="python async await coroutine event loop"):
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


# ── RDR1: 近期访问 chunk 在 RIF 中受保护 ──────────────────────────────────────

def test_rdr1_recent_chunk_protected_from_rif(conn):
    """RDR1: last_accessed < 6h 且 importance >= 0.5 → 豁免 RIF 竞争性抑制。"""
    # 命中 chunk
    _insert(conn, "hit_a", chunk_type="decision",
            summary="python async await coroutine concurrency programming",
            last_accessed_hours_ago=0.5, importance=0.7)
    # 近期访问的竞争者（should be protected by RDR）
    _insert(conn, "recent_comp", chunk_type="decision",
            summary="python async await coroutine event loop concurrent",
            last_accessed_hours_ago=1.0, importance=0.6, stability=5.0)

    stab_before = _get_stability(conn, "recent_comp")
    apply_rif_by_summary(conn, "test", ["hit_a"])
    stab_after = _get_stability(conn, "recent_comp")

    assert abs(stab_after - stab_before) < 0.001, (
        f"RDR1: 近期访问 chunk 应豁免 RIF 抑制，before={stab_before:.3f} after={stab_after:.3f}"
    )


# ── RDR2: 旧访问 chunk 正常被 RIF 抑制 ───────────────────────────────────────

def test_rdr2_old_chunk_rif_suppressed(conn):
    """RDR2: last_accessed > rdr_window_hours(6h) → 正常参与 RIF 竞争性抑制。"""
    _insert(conn, "hit_a", chunk_type="decision",
            summary="python async await coroutine concurrency programming",
            last_accessed_hours_ago=0.5)
    # 旧访问竞争者（48h 前，超出 6h 窗口）
    _insert(conn, "old_comp", chunk_type="decision",
            summary="python async await coroutine event loop concurrent",
            last_accessed_hours_ago=48.0, importance=0.6, stability=5.0)

    stab_before = _get_stability(conn, "old_comp")
    apply_rif_by_summary(conn, "test", ["hit_a"])
    stab_after = _get_stability(conn, "old_comp")

    assert stab_after < stab_before, (
        f"RDR2: 旧访问 chunk 应被 RIF 抑制，before={stab_before:.3f} after={stab_after:.3f}"
    )


# ── RDR3: 近期访问但 importance 低 → 不受 RDR 保护 ───────────────────────────

def test_rdr3_low_importance_recent_not_protected(conn):
    """RDR3: last_accessed < 6h 但 importance < rdr_min_importance(0.5) → 不受 RDR 保护。"""
    _insert(conn, "hit_a", chunk_type="decision",
            summary="python async await coroutine concurrency programming",
            last_accessed_hours_ago=0.5)
    # 近期访问但低 importance（0.3 < 0.5）→ 不受 RDR 保护
    _insert(conn, "low_imp_recent", chunk_type="decision",
            summary="python async await coroutine event loop concurrent",
            last_accessed_hours_ago=1.0, importance=0.3, stability=5.0)

    stab_before = _get_stability(conn, "low_imp_recent")
    apply_rif_by_summary(conn, "test", ["hit_a"])
    stab_after = _get_stability(conn, "low_imp_recent")

    assert stab_after < stab_before, (
        f"RDR3: 低 importance 近期 chunk 不应受 RDR 保护，before={stab_before:.3f} after={stab_after:.3f}"
    )


# ── RDR4: rdr_enabled=False → 近期访问 chunk 也被 RIF 抑制 ───────────────────

def test_rdr4_disabled_recent_suppressed(conn):
    """RDR4: store_vfs.rdr_enabled=False → 近期访问 chunk 也受 RIF 竞争性抑制。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.rdr_enabled":
            return False
        return original_get(key, project=project)

    _insert(conn, "hit_a", chunk_type="decision",
            summary="python async await coroutine concurrency programming",
            last_accessed_hours_ago=0.5)
    _insert(conn, "recent_no_rdr", chunk_type="decision",
            summary="python async await coroutine event loop concurrent",
            last_accessed_hours_ago=1.0, importance=0.6, stability=5.0)

    stab_before = _get_stability(conn, "recent_no_rdr")
    with mock.patch.object(config, 'get', side_effect=patched_get):
        apply_rif_by_summary(conn, "test", ["hit_a"])
    stab_after = _get_stability(conn, "recent_no_rdr")

    assert stab_after < stab_before, (
        f"RDR4: 禁用 RDR 时近期 chunk 应被 RIF 抑制，before={stab_before:.3f} after={stab_after:.3f}"
    )


# ── RDR5: rdr_window_hours 可配置 ────────────────────────────────────────────

def test_rdr5_configurable_window(conn):
    """RDR5: rdr_window_hours=2h 时，3h 前访问的 chunk 不受 RDR 保护（窗口外）。"""
    original_get = config.get

    def patched_2h(key, project=None):
        if key == "store_vfs.rdr_window_hours":
            return 2.0  # 缩小保护窗口到 2 小时
        return original_get(key, project=project)

    _insert(conn, "hit_a", chunk_type="decision",
            summary="python async await coroutine concurrency programming",
            last_accessed_hours_ago=0.5)
    # 3h 前访问 → 超出 2h 窗口 → 不受 RDR 保护
    _insert(conn, "outside_2h", chunk_type="decision",
            summary="python async await coroutine event loop concurrent",
            last_accessed_hours_ago=3.0, importance=0.6, stability=5.0)
    # 1h 前访问 → 在 2h 窗口内 → 受 RDR 保护
    _insert(conn, "inside_2h", chunk_type="decision",
            summary="python async await coroutine event loop concurrent execution",
            last_accessed_hours_ago=1.0, importance=0.6, stability=5.0)

    stab_out_before = _get_stability(conn, "outside_2h")
    stab_in_before = _get_stability(conn, "inside_2h")

    with mock.patch.object(config, 'get', side_effect=patched_2h):
        apply_rif_by_summary(conn, "test", ["hit_a"])

    stab_out_after = _get_stability(conn, "outside_2h")
    stab_in_after = _get_stability(conn, "inside_2h")

    assert stab_out_after < stab_out_before, (
        f"RDR5: 窗口外 chunk 应被 RIF 抑制，before={stab_out_before:.3f} after={stab_out_after:.3f}"
    )
    assert abs(stab_in_after - stab_in_before) < 0.001, (
        f"RDR5: 窗口内 chunk 应受 RDR 保护，before={stab_in_before:.3f} after={stab_in_after:.3f}"
    )


# ── RDR6: 边界精确测试 ────────────────────────────────────────────────────────

def test_rdr6_boundary_precision(conn):
    """RDR6: 访问时间恰好在窗口内(5.9h) vs 窗口外(6.1h)的精确边界行为。"""
    _insert(conn, "hit_a", chunk_type="decision",
            summary="python async await coroutine concurrency programming",
            last_accessed_hours_ago=0.5)
    # 5.9h 前访问（窗口 6h 内，应受 RDR 保护）
    _insert(conn, "border_in", chunk_type="decision",
            summary="python async await coroutine event loop concurrent",
            last_accessed_hours_ago=5.9, importance=0.6, stability=5.0)
    # 6.1h 前访问（窗口 6h 外，应被 RIF 抑制）
    _insert(conn, "border_out", chunk_type="decision",
            summary="python async await coroutine event loop concurrent code",
            last_accessed_hours_ago=6.1, importance=0.6, stability=5.0)

    stab_in_before = _get_stability(conn, "border_in")
    stab_out_before = _get_stability(conn, "border_out")

    apply_rif_by_summary(conn, "test", ["hit_a"])

    stab_in_after = _get_stability(conn, "border_in")
    stab_out_after = _get_stability(conn, "border_out")

    assert abs(stab_in_after - stab_in_before) < 0.001, (
        f"RDR6: 窗口内边界 chunk 应受保护，before={stab_in_before:.3f} after={stab_in_after:.3f}"
    )
    assert stab_out_after < stab_out_before, (
        f"RDR6: 窗口外边界 chunk 应被抑制，before={stab_out_before:.3f} after={stab_out_after:.3f}"
    )


# ── RDR7: 高 importance 已由 RIF protect 保护（RDR 不影响） ──────────────────

def test_rdr7_high_importance_rif_protected(conn):
    """RDR7: importance >= rif_protect_importance(0.85) → 已由 RIF 保护，RDR 不产生额外效果。"""
    _insert(conn, "hit_a", chunk_type="decision",
            summary="python async await coroutine concurrency programming",
            last_accessed_hours_ago=0.5)
    # 高 importance（0.90 > 0.85 = rif_protect_importance）→ RIF 保护已覆盖
    _insert(conn, "high_imp_old", chunk_type="decision",
            summary="python async await coroutine event loop concurrent",
            last_accessed_hours_ago=48.0, importance=0.90, stability=5.0)

    stab_before = _get_stability(conn, "high_imp_old")
    apply_rif_by_summary(conn, "test", ["hit_a"])
    stab_after = _get_stability(conn, "high_imp_old")

    # 高 importance 由 RIF importance 保护（WHERE importance < protect_imp）
    assert abs(stab_after - stab_before) < 0.001, (
        f"RDR7: 高 importance chunk 应由 RIF 保护（无关 RDR），before={stab_before:.3f} after={stab_after:.3f}"
    )


# ── RDR8: 跨 chunk_type 均受 RDR 保护 ────────────────────────────────────────

def test_rdr8_cross_type_protection(conn):
    """RDR8: decision 和 reasoning_chain 类型的近期 chunk 均受 RDR 保护。"""
    # 命中 chunks（不同类型各一个）
    _insert(conn, "hit_dec", chunk_type="decision",
            summary="python async await coroutine concurrency programming",
            last_accessed_hours_ago=0.5)
    _insert(conn, "hit_rc", chunk_type="reasoning_chain",
            summary="python async await coroutine concurrency programming design",
            last_accessed_hours_ago=0.5)

    # 近期访问的同类型竞争者
    _insert(conn, "recent_dec", chunk_type="decision",
            summary="python async await coroutine event loop concurrent",
            last_accessed_hours_ago=2.0, importance=0.6, stability=5.0)
    _insert(conn, "recent_rc", chunk_type="reasoning_chain",
            summary="python async await coroutine event loop concurrent design",
            last_accessed_hours_ago=2.0, importance=0.6, stability=5.0)

    stab_dec_before = _get_stability(conn, "recent_dec")
    stab_rc_before = _get_stability(conn, "recent_rc")

    apply_rif_by_summary(conn, "test", ["hit_dec", "hit_rc"])

    stab_dec_after = _get_stability(conn, "recent_dec")
    stab_rc_after = _get_stability(conn, "recent_rc")

    assert abs(stab_dec_after - stab_dec_before) < 0.001, (
        f"RDR8: 近期 decision chunk 应受 RDR 保护，before={stab_dec_before:.3f} after={stab_dec_after:.3f}"
    )
    assert abs(stab_rc_after - stab_rc_before) < 0.001, (
        f"RDR8: 近期 reasoning_chain chunk 应受 RDR 保护，before={stab_rc_before:.3f} after={stab_rc_after:.3f}"
    )


# ── RDR9: rdr_min_importance 边界 ────────────────────────────────────────────

def test_rdr9_rdr_min_importance_boundary(conn):
    """RDR9: importance 恰好等于 rdr_min_importance(0.5) → 受 RDR 保护（>= 阈值）。"""
    rdr_min_imp = config.get("store_vfs.rdr_min_importance")  # 0.5

    _insert(conn, "hit_a", chunk_type="decision",
            summary="python async await coroutine concurrency programming",
            last_accessed_hours_ago=0.5)
    # importance 恰好等于阈值
    _insert(conn, "boundary_imp", chunk_type="decision",
            summary="python async await coroutine event loop concurrent",
            last_accessed_hours_ago=1.0, importance=rdr_min_imp, stability=5.0)

    stab_before = _get_stability(conn, "boundary_imp")
    apply_rif_by_summary(conn, "test", ["hit_a"])
    stab_after = _get_stability(conn, "boundary_imp")

    assert abs(stab_after - stab_before) < 0.001, (
        f"RDR9: importance={rdr_min_imp} 等于阈值应受 RDR 保护，"
        f"before={stab_before:.3f} after={stab_after:.3f}"
    )
