"""
test_iter438_jost.py — iter438: Jost's Law 单元测试

覆盖：
  J1: 高龄 + 高 importance + 未访问 chunk 在衰减后获得 stability 修复（减速）
  J2: 低龄（< jost_min_age_days=14）chunk 不获得 Jost 修复
  J3: importance < jost_min_importance(0.50) → 不修复
  J4: jost_enabled=False → 不修复
  J5: access_count >= 2 的 chunk 不受修复（活跃 chunk 不触发）
  J6: 高龄 chunk 比低龄 chunk 获得更多修复量（age-gradient）
  J7: jost_scale 越大 → 修复量越大（可配置）
  J8: 修复后 stability 不超过 365.0（上限保护）
  J9: 返回计数正确（adjusted, total_examined）

认知科学依据：
  Jost (1897) — 若两个记忆强度相同，较老的记忆遗忘得更慢。
  Baddeley (1997) Human Memory: Theory and Practice —
    Jost's Law 与 Ribot's Law 互补：Ribot = floor 保护，Jost = decay 减速。

OS 类比：Linux MGLRU old generation protection —
  经历多个 aging interval 的 old gen page 获得更弱的 reclaim pressure。
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
    apply_jost_law,
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


def _ago_iso(days: float = 0.0) -> str:
    return (datetime.datetime.now(datetime.timezone.utc) -
            datetime.timedelta(days=days)).isoformat()


def _insert_chunk(conn, cid, project="test", stability=2.0, importance=0.7,
                  age_days=30.0, access_count=0, chunk_type="decision",
                  last_accessed_days_ago=60.0):
    """
    Insert a chunk with controlled age (created_at) and last_accessed.
    age_days: how old the chunk is (days since creation)
    last_accessed_days_ago: how long ago it was last accessed
    """
    created_at = _ago_iso(days=age_days)
    last_accessed = _ago_iso(days=last_accessed_days_ago)
    now = _now_iso()
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0.9, ?, ?)""",
        (cid, project, chunk_type, f"content {cid}", f"summary {cid}",
         importance, stability, created_at, now, last_accessed, access_count)
    )
    conn.commit()


def _get_stability(conn, cid: str) -> float:
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


# ── J1: 高龄 + 高 importance + 未访问 → 获得 Jost 衰减减速修复 ──────────────────

def test_j1_old_important_chunk_gets_restored(conn):
    """J1: age > jost_min_age_days(14) + importance >= 0.50 + stale → stability 增加。"""
    # age=60 days, importance=0.7, last_accessed=60 days ago (stale), access_count=0
    _insert_chunk(conn, "old_chunk", age_days=60.0, importance=0.7,
                  stability=2.0, access_count=0, last_accessed_days_ago=60.0)

    stab_before = _get_stability(conn, "old_chunk")
    result = apply_jost_law(conn, "test", stale_days=30)
    stab_after = _get_stability(conn, "old_chunk")

    assert stab_after > stab_before, (
        f"J1: 高龄重要 chunk 应获得 Jost 减速修复，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["adjusted"] >= 1, f"J1: adjusted 应 >= 1，got {result}"


# ── J2: 低龄 chunk → 不修复 ──────────────────────────────────────────────────────

def test_j2_young_chunk_not_adjusted(conn):
    """J2: age < jost_min_age_days(14) → 不受 Jost 修复。"""
    # age=5 days, 比 jost_min_age_days(14) 小
    _insert_chunk(conn, "young_chunk", age_days=5.0, importance=0.7,
                  stability=2.0, access_count=0, last_accessed_days_ago=40.0)

    stab_before = _get_stability(conn, "young_chunk")
    apply_jost_law(conn, "test", stale_days=30)
    stab_after = _get_stability(conn, "young_chunk")

    assert abs(stab_after - stab_before) < 0.001, (
        f"J2: 低龄 chunk 不应受 Jost 修复，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── J3: importance 低 → 不修复 ──────────────────────────────────────────────────

def test_j3_low_importance_not_adjusted(conn):
    """J3: importance < jost_min_importance(0.50) → 不修复。"""
    _insert_chunk(conn, "low_imp", age_days=60.0, importance=0.3,
                  stability=2.0, access_count=0, last_accessed_days_ago=60.0)

    stab_before = _get_stability(conn, "low_imp")
    apply_jost_law(conn, "test", stale_days=30)
    stab_after = _get_stability(conn, "low_imp")

    assert abs(stab_after - stab_before) < 0.001, (
        f"J3: 低 importance chunk 不应受 Jost 修复，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── J4: jost_enabled=False → 不修复 ──────────────────────────────────────────────

def test_j4_disabled_no_adjustment(conn):
    """J4: store_vfs.jost_enabled=False → 无任何修复。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.jost_enabled":
            return False
        return original_get(key, project=project)

    _insert_chunk(conn, "disabled_jost", age_days=60.0, importance=0.7,
                  stability=2.0, access_count=0, last_accessed_days_ago=60.0)

    stab_before = _get_stability(conn, "disabled_jost")
    with mock.patch.object(config, 'get', side_effect=patched_get):
        result = apply_jost_law(conn, "test", stale_days=30)
    stab_after = _get_stability(conn, "disabled_jost")

    assert abs(stab_after - stab_before) < 0.001, (
        f"J4: disabled 时不应修复，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["adjusted"] == 0, f"J4: adjusted 应为 0，got {result}"


# ── J5: access_count >= 2 → 不触发（活跃 chunk 无需修复）──────────────────────────

def test_j5_active_chunk_not_adjusted(conn):
    """J5: access_count >= 2 的 chunk 不触发 Jost 修复（活跃 chunk 未被衰减）。"""
    _insert_chunk(conn, "active_chunk", age_days=60.0, importance=0.7,
                  stability=2.0, access_count=3,  # 活跃
                  last_accessed_days_ago=60.0)

    stab_before = _get_stability(conn, "active_chunk")
    apply_jost_law(conn, "test", stale_days=30)
    stab_after = _get_stability(conn, "active_chunk")

    assert abs(stab_after - stab_before) < 0.001, (
        f"J5: 活跃 chunk（access_count >= 2）不应受 Jost 修复，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── J6: 高龄 chunk 比低龄 chunk 修复量更大（age gradient）──────────────────────────

def test_j6_older_chunk_more_adjustment(conn):
    """J6: age=365d 的 chunk 修复量应多于 age=30d（Jost's Law age gradient）。"""
    # 两个 chunk 相同 stability/importance，不同年龄
    _insert_chunk(conn, "very_old", age_days=365.0, importance=0.7,
                  stability=2.0, access_count=0, last_accessed_days_ago=60.0)
    _insert_chunk(conn, "moderately_old", age_days=30.0, importance=0.7,
                  stability=2.0, access_count=0, last_accessed_days_ago=60.0)

    stab_old_before = _get_stability(conn, "very_old")
    stab_mod_before = _get_stability(conn, "moderately_old")
    apply_jost_law(conn, "test", stale_days=30)
    stab_old_after = _get_stability(conn, "very_old")
    stab_mod_after = _get_stability(conn, "moderately_old")

    delta_old = stab_old_after - stab_old_before
    delta_mod = stab_mod_after - stab_mod_before

    assert delta_old >= delta_mod, (
        f"J6: 高龄 chunk 修复量应 >= 低龄，"
        f"delta_old={delta_old:.5f} delta_mod={delta_mod:.5f}"
    )


# ── J7: jost_scale 越大 → 修复量越大 ────────────────────────────────────────────

def test_j7_larger_scale_more_adjustment(conn):
    """J7: jost_scale=0.30 时修复量比默认 0.10 更大。"""
    original_get = config.get

    # Test with larger scale (0.30)
    _insert_chunk(conn, "scale_30", age_days=60.0, importance=0.7,
                  stability=2.0, access_count=0, last_accessed_days_ago=60.0)

    def patched_scale_30(key, project=None):
        if key == "store_vfs.jost_scale":
            return 0.30
        return original_get(key, project=project)

    stab_before_30 = _get_stability(conn, "scale_30")
    with mock.patch.object(config, 'get', side_effect=patched_scale_30):
        apply_jost_law(conn, "test", stale_days=30)
    stab_after_30 = _get_stability(conn, "scale_30")
    delta_30 = stab_after_30 - stab_before_30

    # Reset stability
    conn.execute("UPDATE memory_chunks SET stability=2.0 WHERE id='scale_30'")
    conn.commit()

    # Test with default scale (0.10)
    stab_before_10 = _get_stability(conn, "scale_30")
    apply_jost_law(conn, "test", stale_days=30)
    stab_after_10 = _get_stability(conn, "scale_30")
    delta_10 = stab_after_10 - stab_before_10

    assert delta_30 > delta_10, (
        f"J7: jost_scale=0.30 修复量应大于 0.10，"
        f"delta_30={delta_30:.5f} delta_10={delta_10:.5f}"
    )


# ── J8: 修复后 stability 不超过 365.0 ───────────────────────────────────────────

def test_j8_stability_cap_365(conn):
    """J8: Jost 修复后 stability 不超过 365.0。"""
    # 极高 stability + 高龄：确保修复后仍不超上限
    _insert_chunk(conn, "near_cap", age_days=365.0, importance=0.8,
                  stability=364.9, access_count=0, last_accessed_days_ago=60.0)

    apply_jost_law(conn, "test", stale_days=30)
    stab_after = _get_stability(conn, "near_cap")

    assert stab_after <= 365.0, f"J8: stability 不应超过 365.0，got {stab_after}"


# ── J9: 返回计数正确 ──────────────────────────────────────────────────────────────

def test_j9_return_counts_correct(conn):
    """J9: result dict 中 adjusted 和 total_examined 计数正确。"""
    min_age = config.get("store_vfs.jost_min_age_days")   # 14
    min_imp = config.get("store_vfs.jost_min_importance")  # 0.50

    # 符合条件的 2 个
    _insert_chunk(conn, "q1", age_days=30.0, importance=0.7,
                  stability=2.0, access_count=0, last_accessed_days_ago=60.0)
    _insert_chunk(conn, "q2", age_days=60.0, importance=0.8,
                  stability=3.0, access_count=0, last_accessed_days_ago=60.0)
    # 太年轻（age < 14d）
    _insert_chunk(conn, "too_young", age_days=5.0, importance=0.7,
                  stability=2.0, access_count=0, last_accessed_days_ago=40.0)
    # importance 太低
    _insert_chunk(conn, "low_imp", age_days=60.0, importance=0.3,
                  stability=2.0, access_count=0, last_accessed_days_ago=60.0)

    result = apply_jost_law(conn, "test", stale_days=30)

    assert "adjusted" in result, "J9: result 应含 adjusted key"
    assert "total_examined" in result, "J9: result 应含 total_examined key"
    # q1, q2 符合（adjusted >= 2，但 total_examined 是查询返回的所有候选数，不含 too_young/low_imp）
    assert result["adjusted"] >= 2, f"J9: 应有 >= 2 个 chunk 被修复，got {result}"
    assert result["total_examined"] >= 2, f"J9: total_examined 应 >= 2，got {result}"
