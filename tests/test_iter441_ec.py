"""
test_iter441_ec.py — iter441: Emotional Consolidation 单元测试

覆盖：
  EC1: emotional_weight >= ec_min_weight(0.40) + importance >= 0.40 → stability 增加
  EC2: emotional_weight < ec_min_weight → 无 consolidation
  EC3: importance < ec_min_importance(0.40) → 无 consolidation（低重要性情绪 chunk 不受保护）
  EC4: ec_enabled=False → 无任何 consolidation
  EC5: 高 emotional_weight 比低 emotional_weight 获得更多加成（weight-proportional）
  EC6: consolidation 后 stability 不超过 365.0
  EC7: ec_scale 越大 → 加成越大（可配置）
  EC8: 非 stale 的 chunk（recent access）也受 consolidation（不限于 stale）
  EC9: 返回计数正确（consolidated, total_examined）

认知科学依据：
  McGaugh (2000) "Memory — a century of consolidation" Science 287 —
  情绪唤醒通过杏仁核-海马交互在睡眠期间优先巩固记忆，与 iter409（Flashbulb 写入加成）互补：
  iter409 = encoding 阶段；iter441 = consolidation 阶段（每次 sleep_consolidate 持续加成）。

OS 类比：Linux writeback priority —
  高优先级 dirty page 被 pdflush 优先刷写（优先巩固）。
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
    apply_emotional_consolidation,
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


def _insert_chunk(conn, cid, project="test", stability=5.0, importance=0.6,
                  emotional_weight=0.5, access_count=0, last_accessed_days_ago=1.0):
    now = _now_iso()
    last_accessed = _ago_iso(days=last_accessed_days_ago)
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count,
            emotional_weight)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0.9, ?, ?, ?)""",
        (cid, project, "decision", f"content {cid}", f"summary {cid}",
         importance, stability, now, now, last_accessed, access_count, emotional_weight)
    )
    conn.commit()


def _get_stability(conn, cid: str) -> float:
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


# ── EC1: 高 emotional_weight 且 importance 足够 → stability 增加 ─────────────────

def test_ec1_emotional_chunk_consolidated(conn):
    """EC1: emotional_weight >= ec_min_weight(0.40) + importance >= 0.40 → stability 增加。"""
    ec_min_weight = config.get("store_vfs.ec_min_weight")  # 0.40
    ec_min_imp = config.get("store_vfs.ec_min_importance")  # 0.40
    _insert_chunk(conn, "emotional", emotional_weight=0.80, importance=ec_min_imp,
                  stability=5.0)

    stab_before = _get_stability(conn, "emotional")
    result = apply_emotional_consolidation(conn, "test")
    stab_after = _get_stability(conn, "emotional")

    assert stab_after > stab_before, (
        f"EC1: 情绪显著性 chunk 应获得 sleep 巩固加成，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["consolidated"] >= 1, f"EC1: consolidated 应 >= 1，got {result}"


# ── EC2: emotional_weight 不足 → 无 consolidation ────────────────────────────────

def test_ec2_low_emotional_weight_no_consolidation(conn):
    """EC2: emotional_weight < ec_min_weight(0.40) → 无 Emotional Consolidation。"""
    ec_min_weight = config.get("store_vfs.ec_min_weight")  # 0.40
    _insert_chunk(conn, "low_ew", emotional_weight=0.20,  # < 0.40
                  importance=0.6, stability=5.0)

    stab_before = _get_stability(conn, "low_ew")
    apply_emotional_consolidation(conn, "test")
    stab_after = _get_stability(conn, "low_ew")

    assert abs(stab_after - stab_before) < 0.001, (
        f"EC2: 低 emotional_weight chunk 不应受巩固加成，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── EC3: importance 不足 → 无 consolidation ──────────────────────────────────────

def test_ec3_low_importance_no_consolidation(conn):
    """EC3: importance < ec_min_importance(0.40) → 无 consolidation。"""
    _insert_chunk(conn, "low_imp", emotional_weight=0.80, importance=0.20,  # < 0.40
                  stability=5.0)

    stab_before = _get_stability(conn, "low_imp")
    apply_emotional_consolidation(conn, "test")
    stab_after = _get_stability(conn, "low_imp")

    assert abs(stab_after - stab_before) < 0.001, (
        f"EC3: 低 importance chunk 不应受巩固加成，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── EC4: ec_enabled=False → 无 consolidation ─────────────────────────────────────

def test_ec4_disabled_no_consolidation(conn):
    """EC4: store_vfs.ec_enabled=False → 无任何 consolidation。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.ec_enabled":
            return False
        return original_get(key, project=project)

    _insert_chunk(conn, "disabled_ec", emotional_weight=0.90, importance=0.7, stability=5.0)

    stab_before = _get_stability(conn, "disabled_ec")
    with mock.patch.object(config, 'get', side_effect=patched_get):
        result = apply_emotional_consolidation(conn, "test")
    stab_after = _get_stability(conn, "disabled_ec")

    assert abs(stab_after - stab_before) < 0.001, (
        f"EC4: disabled 时不应有 consolidation，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["consolidated"] == 0, f"EC4: consolidated 应为 0，got {result}"


# ── EC5: 高 emotional_weight 比低 emotional_weight 加成更大 ─────────────────────

def test_ec5_higher_weight_more_boost(conn):
    """EC5: emotional_weight=0.90 的 chunk 加成应多于 emotional_weight=0.50。"""
    _insert_chunk(conn, "high_ew", emotional_weight=0.90, importance=0.6, stability=5.0)
    _insert_chunk(conn, "medium_ew", emotional_weight=0.50, importance=0.6, stability=5.0)

    stab_high_before = _get_stability(conn, "high_ew")
    stab_med_before = _get_stability(conn, "medium_ew")
    apply_emotional_consolidation(conn, "test")
    stab_high_after = _get_stability(conn, "high_ew")
    stab_med_after = _get_stability(conn, "medium_ew")

    delta_high = stab_high_after - stab_high_before
    delta_med = stab_med_after - stab_med_before

    assert delta_high > delta_med, (
        f"EC5: 高 emotional_weight chunk 加成应多于低 emotional_weight，"
        f"delta_high={delta_high:.5f} delta_med={delta_med:.5f}"
    )


# ── EC6: consolidation 后 stability 不超过 365.0 ─────────────────────────────────

def test_ec6_stability_cap_365(conn):
    """EC6: Emotional Consolidation 后 stability 不超过 365.0。"""
    _insert_chunk(conn, "near_cap", emotional_weight=1.0, importance=0.8,
                  stability=364.9)

    apply_emotional_consolidation(conn, "test")
    stab_after = _get_stability(conn, "near_cap")

    assert stab_after <= 365.0, f"EC6: stability 不应超过 365.0，got {stab_after}"


# ── EC7: ec_scale 越大 → 加成越大（可配置） ──────────────────────────────────────

def test_ec7_configurable_scale(conn):
    """EC7: ec_scale=0.20 时加成比默认 0.08 更大。"""
    original_get = config.get

    _insert_chunk(conn, "scale_chunk", emotional_weight=0.80, importance=0.6, stability=5.0)

    def patched_20(key, project=None):
        if key == "store_vfs.ec_scale":
            return 0.20
        return original_get(key, project=project)

    stab_before = _get_stability(conn, "scale_chunk")
    with mock.patch.object(config, 'get', side_effect=patched_20):
        apply_emotional_consolidation(conn, "test")
    stab_after_20 = _get_stability(conn, "scale_chunk")
    delta_20 = stab_after_20 - stab_before

    # 重置
    conn.execute("UPDATE memory_chunks SET stability=5.0 WHERE id='scale_chunk'")
    conn.commit()

    stab_before_default = _get_stability(conn, "scale_chunk")
    apply_emotional_consolidation(conn, "test")  # 使用默认 scale=0.08
    stab_after_default = _get_stability(conn, "scale_chunk")
    delta_default = stab_after_default - stab_before_default

    assert delta_20 > delta_default, (
        f"EC7: ec_scale=0.20 加成应大于默认 0.08，"
        f"delta_20={delta_20:.5f} delta_default={delta_default:.5f}"
    )


# ── EC8: 非 stale 的 chunk 也受 consolidation ────────────────────────────────────

def test_ec8_non_stale_chunk_also_consolidated(conn):
    """EC8: emotional_weight 高的 chunk 无论访问时间（非 stale）都受 consolidation。"""
    # 近期访问的 chunk（非 stale）
    _insert_chunk(conn, "recent_emotional", emotional_weight=0.80, importance=0.6,
                  stability=5.0, last_accessed_days_ago=0.5)  # 12小时前访问

    stab_before = _get_stability(conn, "recent_emotional")
    result = apply_emotional_consolidation(conn, "test")
    stab_after = _get_stability(conn, "recent_emotional")

    assert stab_after > stab_before, (
        f"EC8: 近期访问的情绪显著性 chunk 也应受巩固加成（不限 stale），"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── EC9: 返回计数正确 ─────────────────────────────────────────────────────────────

def test_ec9_return_counts_correct(conn):
    """EC9: result dict 中 consolidated 和 total_examined 计数正确。"""
    ec_min_weight = config.get("store_vfs.ec_min_weight")  # 0.40

    # 2 个满足条件
    _insert_chunk(conn, "e1", emotional_weight=0.60, importance=0.6, stability=4.0)
    _insert_chunk(conn, "e2", emotional_weight=0.90, importance=0.7, stability=5.0)
    # emotional_weight 不足
    _insert_chunk(conn, "e3", emotional_weight=0.10, importance=0.6, stability=4.0)
    # importance 不足
    _insert_chunk(conn, "e4", emotional_weight=0.80, importance=0.20, stability=4.0)

    result = apply_emotional_consolidation(conn, "test")

    assert "consolidated" in result, "EC9: result 应含 consolidated key"
    assert "total_examined" in result, "EC9: result 应含 total_examined key"
    assert result["consolidated"] >= 2, f"EC9: 应有 >= 2 个 chunk 被巩固，got {result}"
    assert result["total_examined"] >= 2, f"EC9: total_examined 应 >= 2，got {result}"
