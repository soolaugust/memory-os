"""
test_iter445_rtmc.py — iter445: Reward-Tagged Memory Consolidation 单元测试

覆盖：
  RT1: access_count >= rtmc_min_access(3) + last_accessed 在 rtmc_recency_hours(48h) 内 → stability 增加
  RT2: access_count 不足（< 3）→ 无 consolidation
  RT3: last_accessed 超出 recency_hours 窗口 → 无 consolidation（奖励信号不新鲜）
  RT4: rtmc_enabled=False → 无任何 consolidation
  RT5: access_count 越高 → reward_signal 越大 → bonus 越大（对数正比）
  RT6: last_accessed 越近 → recency_factor 越大 → bonus 越大（线性正比）
  RT7: consolidation 后 stability 不超过 365.0
  RT8: rtmc_scale 越大 → 奖励加成越大（可配置）
  RT9: importance < rtmc_min_importance(0.35) → 无 consolidation
  RT10: 返回计数正确（rtmc_boosted, total_examined）

认知科学依据：
  Murty & Adcock (2014) — 多巴胺奖励信号在 SWS 期优先强化高奖励预期记忆。
  Hennies et al. (2015) — reward × sleep 交互效应大于单独效应之和。
  Patil et al. (2017) — 频繁被提取的记忆被视为"高价值"，睡眠期海马优先回放。

OS 类比：Linux workingset_activation —
  refcount × recency = 工作集优先级；高频近期访问 page = 最高 protection。
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
    apply_reward_tagged_memory_consolidation,
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


def _insert_chunk(conn, cid, project="test", stability=5.0, importance=0.6,
                  access_count=5, last_accessed_hours_ago=1.0):
    """Insert chunk with controlled access_count and last_accessed."""
    now = _now_iso()
    last_accessed = _ago_iso(hours=last_accessed_hours_ago)
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count,
            encode_context)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0.8, ?, ?, ?)""",
        (cid, project, "decision", f"content {cid}", f"summary {cid}",
         importance, stability, now, now, last_accessed, access_count, "")
    )
    conn.commit()


def _get_stability(conn, cid: str) -> float:
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


# ── RT1: 高访问 + 近期访问 → stability 增加 ─────────────────────────────────────

def test_rt1_reward_tagged_chunk_boosted(conn):
    """RT1: access_count >= 3 + last_accessed 在 48h 内 → RTMC 奖励巩固。"""
    rtmc_min_access = config.get("store_vfs.rtmc_min_access")       # 3
    rtmc_recency = config.get("store_vfs.rtmc_recency_hours")       # 48.0

    _insert_chunk(conn, "rewarded", importance=0.5, access_count=rtmc_min_access,
                  stability=5.0, last_accessed_hours_ago=1.0)  # 1小时前访问，在 48h 内

    stab_before = _get_stability(conn, "rewarded")
    result = apply_reward_tagged_memory_consolidation(conn, "test")
    stab_after = _get_stability(conn, "rewarded")

    assert stab_after > stab_before, (
        f"RT1: 高访问+近期访问的 chunk 应获得 RTMC 奖励巩固，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["rtmc_boosted"] >= 1, f"RT1: rtmc_boosted 应 >= 1，got {result}"


# ── RT2: access_count 不足 → 无 consolidation ────────────────────────────────────

def test_rt2_low_access_no_consolidation(conn):
    """RT2: access_count < rtmc_min_access(3) → 无奖励巩固（访问不足 = 无奖励标签）。"""
    _insert_chunk(conn, "low_acc", importance=0.6, access_count=1,  # < 3
                  stability=5.0, last_accessed_hours_ago=1.0)

    stab_before = _get_stability(conn, "low_acc")
    apply_reward_tagged_memory_consolidation(conn, "test")
    stab_after = _get_stability(conn, "low_acc")

    assert abs(stab_after - stab_before) < 0.001, (
        f"RT2: access_count 不足的 chunk 不应获得奖励巩固，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── RT3: last_accessed 超出 recency 窗口 → 无 consolidation ─────────────────────

def test_rt3_stale_access_no_consolidation(conn):
    """RT3: last_accessed 超出 rtmc_recency_hours(48h) 窗口 → 奖励信号不新鲜 → 无巩固。"""
    rtmc_recency = config.get("store_vfs.rtmc_recency_hours")  # 48.0

    _insert_chunk(conn, "stale_acc", importance=0.6, access_count=10,
                  stability=5.0,
                  last_accessed_hours_ago=rtmc_recency + 1.0)  # 49h 前 > 48h 窗口

    stab_before = _get_stability(conn, "stale_acc")
    apply_reward_tagged_memory_consolidation(conn, "test")
    stab_after = _get_stability(conn, "stale_acc")

    assert abs(stab_after - stab_before) < 0.001, (
        f"RT3: 超出 recency 窗口的 chunk 不应获得奖励巩固，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── RT4: rtmc_enabled=False → 无 consolidation ──────────────────────────────────

def test_rt4_disabled_no_consolidation(conn):
    """RT4: store_vfs.rtmc_enabled=False → 无任何 RTMC 巩固。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.rtmc_enabled":
            return False
        return original_get(key, project=project)

    _insert_chunk(conn, "disabled_rtmc", importance=0.6, access_count=10,
                  stability=5.0, last_accessed_hours_ago=1.0)

    stab_before = _get_stability(conn, "disabled_rtmc")
    with mock.patch.object(config, 'get', side_effect=patched_get):
        result = apply_reward_tagged_memory_consolidation(conn, "test")
    stab_after = _get_stability(conn, "disabled_rtmc")

    assert abs(stab_after - stab_before) < 0.001, (
        f"RT4: disabled 时不应巩固，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["rtmc_boosted"] == 0, f"RT4: rtmc_boosted 应为 0，got {result}"


# ── RT5: access_count 越高 → reward_signal 越大 → bonus 越大 ────────────────────

def test_rt5_higher_access_more_boost(conn):
    """RT5: access_count 越高 → reward_signal 对数增长 → bonus 越大。"""
    # 高访问 chunk（access=20，接近 rtmc_acc_ref=10 参考值）
    _insert_chunk(conn, "high_acc", importance=0.6, access_count=20,
                  stability=5.0, last_accessed_hours_ago=1.0)
    # 低访问 chunk（access=3，仅刚过最低阈值）
    _insert_chunk(conn, "min_acc", importance=0.6, access_count=3,
                  stability=5.0, last_accessed_hours_ago=1.0)

    stab_high_before = _get_stability(conn, "high_acc")
    stab_min_before = _get_stability(conn, "min_acc")
    apply_reward_tagged_memory_consolidation(conn, "test")
    stab_high_after = _get_stability(conn, "high_acc")
    stab_min_after = _get_stability(conn, "min_acc")

    delta_high = stab_high_after - stab_high_before
    delta_min = stab_min_after - stab_min_before

    assert delta_high > delta_min, (
        f"RT5: 高访问 chunk 的 bonus 应大于低访问，"
        f"delta_high={delta_high:.5f} delta_min={delta_min:.5f}"
    )


# ── RT6: last_accessed 越近 → recency_factor 越大 → bonus 越大 ─────────────────

def test_rt6_fresher_access_more_boost(conn):
    """RT6: last_accessed 越近 → recency_factor 越大（线性）→ bonus 越大。"""
    _insert_chunk(conn, "very_fresh", importance=0.6, access_count=10,
                  stability=5.0, last_accessed_hours_ago=0.5)   # 30分钟前
    _insert_chunk(conn, "less_fresh", importance=0.6, access_count=10,
                  stability=5.0, last_accessed_hours_ago=24.0)  # 24小时前（在 48h 窗口内）

    stab_vf_before = _get_stability(conn, "very_fresh")
    stab_lf_before = _get_stability(conn, "less_fresh")
    apply_reward_tagged_memory_consolidation(conn, "test")
    stab_vf_after = _get_stability(conn, "very_fresh")
    stab_lf_after = _get_stability(conn, "less_fresh")

    delta_fresh = stab_vf_after - stab_vf_before
    delta_less = stab_lf_after - stab_lf_before

    assert delta_fresh > delta_less, (
        f"RT6: 更近期访问的 chunk bonus 应更大，"
        f"delta_fresh={delta_fresh:.5f} delta_less={delta_less:.5f}"
    )


# ── RT7: consolidation 后 stability 不超过 365.0 ────────────────────────────────

def test_rt7_stability_cap_365(conn):
    """RT7: RTMC 巩固后 stability 不超过 365.0。"""
    _insert_chunk(conn, "near_cap", importance=0.7, access_count=50,
                  stability=364.9, last_accessed_hours_ago=0.1)

    apply_reward_tagged_memory_consolidation(conn, "test")
    stab_after = _get_stability(conn, "near_cap")

    assert stab_after <= 365.0, f"RT7: stability 不应超过 365.0，got {stab_after}"


# ── RT8: rtmc_scale 越大 → 加成越大（可配置）────────────────────────────────────

def test_rt8_configurable_scale(conn):
    """RT8: rtmc_scale=0.25 时加成比默认 0.08 更大。"""
    original_get = config.get

    _insert_chunk(conn, "scale_chunk", importance=0.6, access_count=10,
                  stability=5.0, last_accessed_hours_ago=1.0)

    def patched_25(key, project=None):
        if key == "store_vfs.rtmc_scale":
            return 0.25
        return original_get(key, project=project)

    stab_before = _get_stability(conn, "scale_chunk")
    with mock.patch.object(config, 'get', side_effect=patched_25):
        apply_reward_tagged_memory_consolidation(conn, "test")
    stab_after_25 = _get_stability(conn, "scale_chunk")
    delta_25 = stab_after_25 - stab_before

    # 重置
    conn.execute("UPDATE memory_chunks SET stability=5.0 WHERE id='scale_chunk'")
    conn.commit()

    stab_before_default = _get_stability(conn, "scale_chunk")
    apply_reward_tagged_memory_consolidation(conn, "test")  # 默认 scale=0.08
    stab_after_default = _get_stability(conn, "scale_chunk")
    delta_default = stab_after_default - stab_before_default

    assert delta_25 > delta_default, (
        f"RT8: rtmc_scale=0.25 加成应大于默认 0.08，"
        f"delta_25={delta_25:.5f} delta_default={delta_default:.5f}"
    )


# ── RT9: importance < rtmc_min_importance → 无 consolidation ────────────────────

def test_rt9_low_importance_no_consolidation(conn):
    """RT9: importance < rtmc_min_importance(0.35) → 无奖励巩固。"""
    _insert_chunk(conn, "low_imp", importance=0.20,  # < 0.35
                  access_count=10, stability=5.0, last_accessed_hours_ago=1.0)

    stab_before = _get_stability(conn, "low_imp")
    apply_reward_tagged_memory_consolidation(conn, "test")
    stab_after = _get_stability(conn, "low_imp")

    assert abs(stab_after - stab_before) < 0.001, (
        f"RT9: 低 importance chunk 不应获得奖励巩固，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── RT10: 返回计数正确 ─────────────────────────────────────────────────────────

def test_rt10_return_counts_correct(conn):
    """RT10: result dict 中 rtmc_boosted 和 total_examined 计数正确。"""
    # 2 个满足条件（高访问 + 近期 + 高 importance）
    _insert_chunk(conn, "r1", importance=0.6, access_count=5,
                  stability=5.0, last_accessed_hours_ago=1.0)
    _insert_chunk(conn, "r2", importance=0.5, access_count=8,
                  stability=4.0, last_accessed_hours_ago=2.0)
    # access_count 不足
    _insert_chunk(conn, "r3", importance=0.6, access_count=1,
                  stability=5.0, last_accessed_hours_ago=1.0)
    # 超出 recency 窗口
    rtmc_recency = config.get("store_vfs.rtmc_recency_hours")
    _insert_chunk(conn, "r4", importance=0.6, access_count=10,
                  stability=5.0, last_accessed_hours_ago=rtmc_recency + 5.0)
    # importance 不足
    _insert_chunk(conn, "r5", importance=0.10, access_count=10,
                  stability=5.0, last_accessed_hours_ago=1.0)

    result = apply_reward_tagged_memory_consolidation(conn, "test")

    assert "rtmc_boosted" in result, "RT10: result 应含 rtmc_boosted key"
    assert "total_examined" in result, "RT10: result 应含 total_examined key"
    assert result["rtmc_boosted"] >= 2, f"RT10: 应有 >= 2 个 chunk 被奖励巩固，got {result}"
    assert result["total_examined"] >= 2, f"RT10: total_examined 应 >= 2，got {result}"
