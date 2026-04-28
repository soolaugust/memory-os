"""
test_iter437_hypermnesia.py — iter437: Hypermnesia 单元测试

覆盖：
  HM1: spaced_access_count >= threshold(4) 且 importance >= 0.55 → stability boost
  HM2: spaced_access_count < threshold → 无 boost
  HM3: importance < hypermnesia_min_importance(0.55) → 无 boost
  HM4: hypermnesia_enabled=False → 无 boost
  HM5: cooldown 内已 boost 的 chunk → 不再 boost（防止反复触发）
  HM6: cooldown 外的 chunk → 可再次 boost
  HM7: stability 上限 365.0（boost 后不超过遗忘曲线最大值）
  HM8: hypermnesia_threshold 可通过 sysctl 配置
  HM9: apply_hypermnesia 返回计数正确（boosted, total_examined）

认知科学依据：
  Erdelyi & Becker (1974) Hypermnesia — 多轮分布式自由回忆中总召回量净增长。
  Payne (1987) Meta-analysis: +15-25% improvement across 3-5 test sessions。
  条件：必须是间隔分布（spaced）检索，不是集中（massed）检索。

OS 类比：Linux khugepaged Transparent HugePage —
  多 epoch 热访问的页面被合并为 2MB hugepage，降低 TLB miss rate（净效率提升）。
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
    apply_hypermnesia,
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


def _insert_chunk(conn, cid, project="test", stability=3.0, importance=0.7,
                  spaced_access_count=0, chunk_type="decision",
                  hypermnesia_last_boost=None):
    now = _now_iso()
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count,
            spaced_access_count, hypermnesia_last_boost)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0.9, ?, 5, ?, ?)""",
        (cid, project, chunk_type, f"content {cid}", f"summary {cid}",
         importance, stability, now, now, now,
         spaced_access_count, hypermnesia_last_boost)
    )
    conn.commit()


def _get_stability(conn, cid: str) -> float:
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


def _get_hm_last_boost(conn, cid: str):
    row = conn.execute("SELECT hypermnesia_last_boost FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return row[0] if row else None


# ── HM1: 满足条件 → stability boost ──────────────────────────────────────────

def test_hm1_qualified_chunk_gets_boost(conn):
    """HM1: spaced_access_count >= 4 且 importance >= 0.55 → stability boost。"""
    threshold = config.get("store_vfs.hypermnesia_threshold")  # 4
    _insert_chunk(conn, "hm_chunk", spaced_access_count=threshold, importance=0.7, stability=3.0)

    stab_before = _get_stability(conn, "hm_chunk")
    result = apply_hypermnesia(conn, "test")
    stab_after = _get_stability(conn, "hm_chunk")

    hm_boost = config.get("store_vfs.hypermnesia_boost")  # 1.10

    assert stab_after > stab_before, (
        f"HM1: 满足条件的 chunk 应获得 stability boost，"
        f"before={stab_before:.3f} after={stab_after:.3f}"
    )
    assert abs(stab_after - stab_before * hm_boost) < 0.01, (
        f"HM1: stability 应为 {stab_before:.3f} × {hm_boost} = {stab_before * hm_boost:.3f}，"
        f"got {stab_after:.3f}"
    )
    assert result["boosted"] >= 1, f"HM1: boosted 应 >= 1，got {result}"


# ── HM2: spaced_access_count 不足 → 无 boost ─────────────────────────────────

def test_hm2_insufficient_spaced_count_no_boost(conn):
    """HM2: spaced_access_count < threshold(4) → 无 Hypermnesia boost。"""
    _insert_chunk(conn, "low_spaced", spaced_access_count=2, importance=0.7, stability=3.0)

    stab_before = _get_stability(conn, "low_spaced")
    apply_hypermnesia(conn, "test")
    stab_after = _get_stability(conn, "low_spaced")

    assert abs(stab_after - stab_before) < 0.001, (
        f"HM2: spaced_access_count 不足时不应 boost，"
        f"before={stab_before:.3f} after={stab_after:.3f}"
    )


# ── HM3: importance 不足 → 无 boost ──────────────────────────────────────────

def test_hm3_low_importance_no_boost(conn):
    """HM3: importance < hypermnesia_min_importance(0.55) → 无 boost。"""
    threshold = config.get("store_vfs.hypermnesia_threshold")
    _insert_chunk(conn, "low_imp", spaced_access_count=threshold, importance=0.40, stability=3.0)

    stab_before = _get_stability(conn, "low_imp")
    apply_hypermnesia(conn, "test")
    stab_after = _get_stability(conn, "low_imp")

    assert abs(stab_after - stab_before) < 0.001, (
        f"HM3: 低 importance chunk 不应 boost，"
        f"before={stab_before:.3f} after={stab_after:.3f}"
    )


# ── HM4: hypermnesia_enabled=False → 无 boost ────────────────────────────────

def test_hm4_disabled_no_boost(conn):
    """HM4: store_vfs.hypermnesia_enabled=False → 无任何 boost。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.hypermnesia_enabled":
            return False
        return original_get(key, project=project)

    threshold = config.get("store_vfs.hypermnesia_threshold")
    _insert_chunk(conn, "disabled_hm", spaced_access_count=threshold, importance=0.7, stability=3.0)

    stab_before = _get_stability(conn, "disabled_hm")
    with mock.patch.object(config, 'get', side_effect=patched_get):
        result = apply_hypermnesia(conn, "test")
    stab_after = _get_stability(conn, "disabled_hm")

    assert abs(stab_after - stab_before) < 0.001, (
        f"HM4: disabled 时不应 boost，before={stab_before:.3f} after={stab_after:.3f}"
    )
    assert result["boosted"] == 0, f"HM4: boosted 应为 0，got {result}"


# ── HM5: cooldown 内已 boost → 不再 boost ────────────────────────────────────

def test_hm5_cooldown_prevents_double_boost(conn):
    """HM5: hypermnesia_last_boost 在 cooldown_days(7) 内 → 不再触发 boost。"""
    threshold = config.get("store_vfs.hypermnesia_threshold")
    recent_boost = _ago_iso(days=1.0)  # 1 天前 boost 过（< 7 天冷却期）
    _insert_chunk(conn, "already_boosted", spaced_access_count=threshold,
                  importance=0.7, stability=3.0,
                  hypermnesia_last_boost=recent_boost)

    stab_before = _get_stability(conn, "already_boosted")
    result = apply_hypermnesia(conn, "test")
    stab_after = _get_stability(conn, "already_boosted")

    assert abs(stab_after - stab_before) < 0.001, (
        f"HM5: 冷却期内不应再次 boost，before={stab_before:.3f} after={stab_after:.3f}"
    )
    assert result["boosted"] == 0, f"HM5: 冷却期内 boosted 应为 0，got {result}"


# ── HM6: cooldown 外 → 可再次 boost ─────────────────────────────────────────

def test_hm6_post_cooldown_reboostable(conn):
    """HM6: hypermnesia_last_boost 在 cooldown_days(7) 外 → 可再次 boost。"""
    threshold = config.get("store_vfs.hypermnesia_threshold")
    old_boost = _ago_iso(days=10.0)  # 10 天前 boost 过（> 7 天冷却期）
    _insert_chunk(conn, "old_boosted", spaced_access_count=threshold,
                  importance=0.7, stability=3.0,
                  hypermnesia_last_boost=old_boost)

    stab_before = _get_stability(conn, "old_boosted")
    result = apply_hypermnesia(conn, "test")
    stab_after = _get_stability(conn, "old_boosted")

    assert stab_after > stab_before, (
        f"HM6: 冷却期外应可再次 boost，before={stab_before:.3f} after={stab_after:.3f}"
    )
    assert result["boosted"] >= 1, f"HM6: boosted 应 >= 1，got {result}"


# ── HM7: stability 上限 365.0 ────────────────────────────────────────────────

def test_hm7_stability_capped_at_365(conn):
    """HM7: boost 后 stability 不超过 365.0。"""
    threshold = config.get("store_vfs.hypermnesia_threshold")
    _insert_chunk(conn, "high_stab", spaced_access_count=threshold,
                  importance=0.8, stability=350.0)  # close to cap

    apply_hypermnesia(conn, "test")
    stab_after = _get_stability(conn, "high_stab")

    assert stab_after <= 365.0, f"HM7: stability 不应超过 365.0，got {stab_after}"
    assert stab_after >= 350.0, f"HM7: stability 应保持在 350 以上，got {stab_after}"


# ── HM8: hypermnesia_threshold 可配置 ─────────────────────────────────────────

def test_hm8_configurable_threshold(conn):
    """HM8: hypermnesia_threshold=2 时，spaced_access_count=2 的 chunk 也可被 boost。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.hypermnesia_threshold":
            return 2  # 降低阈值到 2
        return original_get(key, project=project)

    _insert_chunk(conn, "low_threshold", spaced_access_count=2,
                  importance=0.7, stability=3.0)

    stab_before = _get_stability(conn, "low_threshold")
    with mock.patch.object(config, 'get', side_effect=patched_get):
        result = apply_hypermnesia(conn, "test")
    stab_after = _get_stability(conn, "low_threshold")

    assert stab_after > stab_before, (
        f"HM8: threshold=2 时 spaced_access=2 的 chunk 应 boost，"
        f"before={stab_before:.3f} after={stab_after:.3f}"
    )


# ── HM9: 返回计数正确 ─────────────────────────────────────────────────────────

def test_hm9_return_counts_correct(conn):
    """HM9: result dict 中 boosted 和 total_examined 计数正确。"""
    threshold = config.get("store_vfs.hypermnesia_threshold")
    # 2 个满足条件，1 个 importance 不足，1 个 spaced_count 不足
    _insert_chunk(conn, "q1", spaced_access_count=threshold, importance=0.7, stability=2.0)
    _insert_chunk(conn, "q2", spaced_access_count=threshold + 2, importance=0.8, stability=3.0)
    _insert_chunk(conn, "no_imp", spaced_access_count=threshold, importance=0.3, stability=2.0)
    _insert_chunk(conn, "no_spaced", spaced_access_count=1, importance=0.7, stability=2.0)

    result = apply_hypermnesia(conn, "test")

    assert "boosted" in result, "HM9: result 应含 boosted key"
    assert "total_examined" in result, "HM9: result 应含 total_examined key"
    assert result["boosted"] == 2, f"HM9: 应有 2 个 chunk 被 boost，got {result}"
    assert result["total_examined"] >= 2, f"HM9: total_examined 应 >= 2，got {result}"
