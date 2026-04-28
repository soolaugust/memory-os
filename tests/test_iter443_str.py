"""
test_iter443_str.py — iter443: Sleep-Targeted Reactivation 单元测试

覆盖：
  ST1: importance >= str_min_importance(0.65) + retrievability <= str_max_retrievability(0.40) → stability 增加
  ST2: importance 不足（< 0.65）→ 无 reactivation
  ST3: retrievability 过高（> 0.40）→ 无 reactivation（记忆仍健康，不需要抢救）
  ST4: str_enabled=False → 无任何 reactivation
  ST5: retrievability 越低 → rescue_bonus 越大（正比于衰退程度）
  ST6: reactivation 后 stability 不超过 365.0
  ST7: str_scale 越大 → 修复越大（可配置）
  ST8: 非 stale chunk 也受 reactivation（不限于旧/未访问 chunk）
  ST9: 返回计数正确（rescued, total_examined）

认知科学依据：
  Stickgold (2005) "Sleep-dependent memory consolidation" (Nature) —
  睡眠期海马 sharp-wave ripples 优先重放高价值但 retrievability 下降的记忆（memory rescue）。
  Stickgold & Walker (2013) "Sleep-dependent memory triage" —
  优先级 = importance × (1 - retrievability)：高价值 + 正在衰退 = 最需要抢救。

OS 类比：Linux dirty page "expire" writeback scan —
  即将超时（接近 dirty_expire_centisecs）的脏页被 flusher 优先抢救写回，防止数据丢失。
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
    apply_sleep_targeted_reactivation,
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


def _insert_chunk(conn, cid, project="test", stability=5.0, importance=0.7,
                  retrievability=0.20, access_count=0, last_accessed_days_ago=10.0):
    """Insert chunk with specified retrievability (simulating decay state)."""
    now = _now_iso()
    last_accessed = _ago_iso(days=last_accessed_days_ago)
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count,
            encode_context)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (cid, project, "decision", f"content {cid}", f"summary {cid}",
         importance, stability, now, now, retrievability, last_accessed, access_count, "")
    )
    conn.commit()


def _get_stability(conn, cid: str) -> float:
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


# ── ST1: 高 importance + 低 retrievability → stability 增加 ─────────────────────

def test_st1_targeted_chunk_rescued(conn):
    """ST1: importance >= 0.65 + retrievability <= 0.40 → sleep-targeted reactivation。"""
    str_min_imp = config.get("store_vfs.str_min_importance")      # 0.65
    str_max_ret = config.get("store_vfs.str_max_retrievability")  # 0.40

    _insert_chunk(conn, "decaying", importance=str_min_imp,
                  retrievability=0.20, stability=5.0)  # 低 retrievability，正在衰退

    stab_before = _get_stability(conn, "decaying")
    result = apply_sleep_targeted_reactivation(conn, "test")
    stab_after = _get_stability(conn, "decaying")

    assert stab_after > stab_before, (
        f"ST1: 高 importance + 低 retrievability chunk 应获得 stability 修复，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["rescued"] >= 1, f"ST1: rescued 应 >= 1，got {result}"


# ── ST2: importance 不足 → 无 reactivation ───────────────────────────────────────

def test_st2_low_importance_no_reactivation(conn):
    """ST2: importance < str_min_importance(0.65) → 无 reactivation（低价值记忆不值得抢救）。"""
    _insert_chunk(conn, "low_imp", importance=0.40,  # < 0.65
                  retrievability=0.20, stability=5.0)

    stab_before = _get_stability(conn, "low_imp")
    apply_sleep_targeted_reactivation(conn, "test")
    stab_after = _get_stability(conn, "low_imp")

    assert abs(stab_after - stab_before) < 0.001, (
        f"ST2: 低 importance chunk 不应被抢救，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── ST3: retrievability 过高 → 无 reactivation ───────────────────────────────────

def test_st3_high_retrievability_no_reactivation(conn):
    """ST3: retrievability > str_max_retrievability(0.40) → 无 reactivation（记忆仍健康）。"""
    _insert_chunk(conn, "healthy", importance=0.80,
                  retrievability=0.70, stability=5.0)  # retrievability=0.70 > 0.40

    stab_before = _get_stability(conn, "healthy")
    apply_sleep_targeted_reactivation(conn, "test")
    stab_after = _get_stability(conn, "healthy")

    assert abs(stab_after - stab_before) < 0.001, (
        f"ST3: 高 retrievability chunk 不需要抢救，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── ST4: str_enabled=False → 无 reactivation ─────────────────────────────────────

def test_st4_disabled_no_reactivation(conn):
    """ST4: store_vfs.str_enabled=False → 无任何 reactivation。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.str_enabled":
            return False
        return original_get(key, project=project)

    _insert_chunk(conn, "disabled_str", importance=0.80, retrievability=0.20, stability=5.0)

    stab_before = _get_stability(conn, "disabled_str")
    with mock.patch.object(config, 'get', side_effect=patched_get):
        result = apply_sleep_targeted_reactivation(conn, "test")
    stab_after = _get_stability(conn, "disabled_str")

    assert abs(stab_after - stab_before) < 0.001, (
        f"ST4: disabled 时不应 reactivate，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["rescued"] == 0, f"ST4: rescued 应为 0，got {result}"


# ── ST5: retrievability 越低 → rescue_bonus 越大 ─────────────────────────────────

def test_st5_lower_retrievability_more_boost(conn):
    """ST5: retrievability=0.05（接近零）比 retrievability=0.35 获得更大修复。"""
    _insert_chunk(conn, "near_zero_ret", importance=0.75,
                  retrievability=0.05, stability=5.0)  # 几乎遗忘
    _insert_chunk(conn, "low_ret", importance=0.75,
                  retrievability=0.35, stability=5.0)   # 低但未到边缘

    stab_nz_before = _get_stability(conn, "near_zero_ret")
    stab_lr_before = _get_stability(conn, "low_ret")
    apply_sleep_targeted_reactivation(conn, "test")
    stab_nz_after = _get_stability(conn, "near_zero_ret")
    stab_lr_after = _get_stability(conn, "low_ret")

    delta_nz = stab_nz_after - stab_nz_before
    delta_lr = stab_lr_after - stab_lr_before

    assert delta_nz > delta_lr, (
        f"ST5: retrievability 越低修复越大，"
        f"delta_nz={delta_nz:.5f} delta_lr={delta_lr:.5f}"
    )


# ── ST6: reactivation 后 stability 不超过 365.0 ───────────────────────────────────

def test_st6_stability_cap_365(conn):
    """ST6: Sleep-Targeted Reactivation 后 stability 不超过 365.0。"""
    _insert_chunk(conn, "near_cap", importance=0.90,
                  retrievability=0.05, stability=364.9)  # 接近上限

    apply_sleep_targeted_reactivation(conn, "test")
    stab_after = _get_stability(conn, "near_cap")

    assert stab_after <= 365.0, f"ST6: stability 不应超过 365.0，got {stab_after}"


# ── ST7: str_scale 越大 → 修复越大（可配置）──────────────────────────────────────

def test_st7_configurable_scale(conn):
    """ST7: str_scale=0.30 时修复量比默认 0.12 更大。"""
    original_get = config.get

    _insert_chunk(conn, "scale_chunk", importance=0.80, retrievability=0.10, stability=5.0)

    def patched_30(key, project=None):
        if key == "store_vfs.str_scale":
            return 0.30
        return original_get(key, project=project)

    stab_before = _get_stability(conn, "scale_chunk")
    with mock.patch.object(config, 'get', side_effect=patched_30):
        apply_sleep_targeted_reactivation(conn, "test")
    stab_after_30 = _get_stability(conn, "scale_chunk")
    delta_30 = stab_after_30 - stab_before

    # 重置
    conn.execute("UPDATE memory_chunks SET stability=5.0 WHERE id='scale_chunk'")
    conn.commit()

    stab_before_default = _get_stability(conn, "scale_chunk")
    apply_sleep_targeted_reactivation(conn, "test")  # 默认 scale=0.12
    stab_after_default = _get_stability(conn, "scale_chunk")
    delta_default = stab_after_default - stab_before_default

    assert delta_30 > delta_default, (
        f"ST7: str_scale=0.30 修复量应大于默认 0.12，"
        f"delta_30={delta_30:.5f} delta_default={delta_default:.5f}"
    )


# ── ST8: 非 stale chunk（近期访问）也受 reactivation ─────────────────────────────

def test_st8_recent_chunk_also_rescued(conn):
    """ST8: 近期访问但 retrievability 已低的 chunk 也受 reactivation（不限于 stale）。"""
    _insert_chunk(conn, "recent_decaying", importance=0.80,
                  retrievability=0.25, stability=5.0,
                  last_accessed_days_ago=0.5)  # 12小时前访问，但 retrievability 已低

    stab_before = _get_stability(conn, "recent_decaying")
    result = apply_sleep_targeted_reactivation(conn, "test")
    stab_after = _get_stability(conn, "recent_decaying")

    assert stab_after > stab_before, (
        f"ST8: 近期访问但 retrievability 低的 chunk 也应被抢救，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── ST9: 返回计数正确 ─────────────────────────────────────────────────────────────

def test_st9_return_counts_correct(conn):
    """ST9: result dict 中 rescued 和 total_examined 计数正确。"""
    # 2 个满足条件（高 importance + 低 retrievability）
    _insert_chunk(conn, "r1", importance=0.75, retrievability=0.15, stability=5.0)
    _insert_chunk(conn, "r2", importance=0.80, retrievability=0.30, stability=4.0)
    # importance 不足
    _insert_chunk(conn, "r3", importance=0.40, retrievability=0.10, stability=5.0)
    # retrievability 过高
    _insert_chunk(conn, "r4", importance=0.80, retrievability=0.80, stability=5.0)

    result = apply_sleep_targeted_reactivation(conn, "test")

    assert "rescued" in result, "ST9: result 应含 rescued key"
    assert "total_examined" in result, "ST9: result 应含 total_examined key"
    assert result["rescued"] >= 2, f"ST9: 应有 >= 2 个 chunk 被抢救，got {result}"
    assert result["total_examined"] >= 2, f"ST9: total_examined 应 >= 2，got {result}"
