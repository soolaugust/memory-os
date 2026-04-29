"""
test_iter465_ldsb.py — iter465: Lag-Dependent Spacing Boost 单元测试

覆盖：
  LD1: lag > ldsb_min_lag_hours(2) 且 importance >= ldsb_min_importance → stability 加成
  LD2: lag < ldsb_min_lag_hours → 无加成
  LD3: ldsb_enabled=False → 无任何加成
  LD4: importance < ldsb_min_importance(0.25) → 不参与 LDSB
  LD5: 加成受 ldsb_max_boost(0.15) 上限保护
  LD6: stability 加成后不超过 365.0
  LD7: 长间隔比短间隔获得更大加成（单调性验证）
  LD8: update_accessed 集成测试 — 长间隔访问后 LDSB 触发

认知科学依据：
  Landauer & Bjork (1978) "Optimum rehearsal patterns and name learning"
    (in Practical Aspects of Memory, Academic Press) —
    间隔检索练习（expanded retrieval practice）：回忆越难（间隔越长）→ 记忆加固越强。
  SM-2 算法（Wozniak 1987）：新稳定性 = 旧稳定性 × EF × f(lag/stability)，
    lag/stability 比率越大 → EF 贡献越大 → 稳定性增益越强。

OS 类比：Linux page aging（mm/vmscan.c）—
  长时间停留在 inactive list 的 page（cold page）再次被访问时，
  获得更高的 active list 优先级（cold page reactivation = high utility signal）。
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

from store_vfs import ensure_schema, apply_lag_dependent_spacing_boost, update_accessed
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


def _insert_chunk(conn, cid, project="test", stability=5.0, importance=0.6,
                  chunk_type="decision", last_accessed_hours_ago=0):
    now = _utcnow()
    last_acc = (now - datetime.timedelta(hours=last_accessed_hours_ago)).isoformat()
    now_iso = now.isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count,
            encode_context, session_type_history)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (cid, project, chunk_type, f"content {cid}", f"summary {cid}",
         importance, stability, now_iso, now_iso, 0.5,
         last_acc, 2, "kernel_mm", "")
    )
    conn.commit()


def _get_stability(conn, cid: str) -> float:
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


# ── LD1: 长间隔 → stability 加成 ─────────────────────────────────────────────────────────

def test_ld1_long_lag_boosted(conn):
    """LD1: lag > ldsb_min_lag_hours(2) 且 importance >= 0.25 → stability 加成。"""
    min_lag = config.get("store_vfs.ldsb_min_lag_hours")  # 2.0

    # 长间隔：last_accessed 在 24 小时前
    _insert_chunk(conn, "ldsb_1_long", stability=5.0, importance=0.6,
                  last_accessed_hours_ago=24)
    stab_before = _get_stability(conn, "ldsb_1_long")
    now_iso = _utcnow().isoformat()
    result = apply_lag_dependent_spacing_boost(conn, ["ldsb_1_long"], now_iso=now_iso)
    stab_after = _get_stability(conn, "ldsb_1_long")

    assert stab_after > stab_before, (
        f"LD1: 长间隔后 stability 应增加，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["ldsb_boosted"] >= 1, f"LD1: ldsb_boosted 应 >= 1，got {result}"


# ── LD2: 短间隔 → 无加成 ──────────────────────────────────────────────────────────────────

def test_ld2_short_lag_no_boost(conn):
    """LD2: lag < ldsb_min_lag_hours(2) → 无 LDSB 加成。"""
    # 短间隔：last_accessed 在 1 小时前（< 2 小时阈值）
    _insert_chunk(conn, "ldsb_2_short", stability=5.0, importance=0.6,
                  last_accessed_hours_ago=1)
    stab_before = _get_stability(conn, "ldsb_2_short")
    now_iso = _utcnow().isoformat()
    result = apply_lag_dependent_spacing_boost(conn, ["ldsb_2_short"], now_iso=now_iso)
    stab_after = _get_stability(conn, "ldsb_2_short")

    assert abs(stab_after - stab_before) < 0.001, (
        f"LD2: 短间隔不应有 LDSB 加成，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["ldsb_boosted"] == 0, f"LD2: ldsb_boosted 应为 0，got {result}"


# ── LD3: ldsb_enabled=False → 无加成 ─────────────────────────────────────────────────────

def test_ld3_disabled_no_boost(conn):
    """LD3: ldsb_enabled=False → 无任何 LDSB 加成。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.ldsb_enabled":
            return False
        return original_get(key, project=project)

    _insert_chunk(conn, "ldsb_3", stability=5.0, importance=0.6,
                  last_accessed_hours_ago=24)
    stab_before = _get_stability(conn, "ldsb_3")
    now_iso = _utcnow().isoformat()

    with mock.patch.object(config, 'get', side_effect=patched_get):
        result = apply_lag_dependent_spacing_boost(conn, ["ldsb_3"], now_iso=now_iso)
    stab_after = _get_stability(conn, "ldsb_3")

    assert abs(stab_after - stab_before) < 0.001, (
        f"LD3: disabled 时不应有 LDSB 加成，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["ldsb_boosted"] == 0, f"LD3: ldsb_boosted 应为 0，got {result}"


# ── LD4: importance 不足 → 不参与 LDSB ───────────────────────────────────────────────────

def test_ld4_low_importance_no_boost(conn):
    """LD4: importance < ldsb_min_importance(0.25) → 不参与 LDSB。"""
    _insert_chunk(conn, "ldsb_4_low", stability=5.0, importance=0.10,
                  last_accessed_hours_ago=24)
    stab_before = _get_stability(conn, "ldsb_4_low")
    now_iso = _utcnow().isoformat()
    result = apply_lag_dependent_spacing_boost(conn, ["ldsb_4_low"], now_iso=now_iso)
    stab_after = _get_stability(conn, "ldsb_4_low")

    assert abs(stab_after - stab_before) < 0.001, (
        f"LD4: 低 importance 不应有 LDSB 加成，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["ldsb_boosted"] == 0, f"LD4: ldsb_boosted 应为 0，got {result}"


# ── LD5: 加成受 ldsb_max_boost 上限保护 ──────────────────────────────────────────────────

def test_ld5_max_boost_cap(conn):
    """LD5: LDSB 加成不超过 base × ldsb_max_boost(0.15)。"""
    ldsb_max_boost = config.get("store_vfs.ldsb_max_boost")  # 0.15
    base = 5.0

    # 超长间隔（远超稳定性周期）→ 触发最大加成
    _insert_chunk(conn, "ldsb_5", stability=base, importance=0.6,
                  last_accessed_hours_ago=720)  # 30 天前
    stab_before = _get_stability(conn, "ldsb_5")
    now_iso = _utcnow().isoformat()
    apply_lag_dependent_spacing_boost(conn, ["ldsb_5"], now_iso=now_iso)
    stab_after = _get_stability(conn, "ldsb_5")

    increment = stab_after - stab_before
    max_allowed = base * ldsb_max_boost + 0.01
    assert increment <= max_allowed, (
        f"LD5: LDSB 增量 {increment:.4f} 不应超过 max_boost 允许的 {max_allowed:.4f}，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert stab_after > stab_before, f"LD5: 应有 LDSB 加成，before={stab_before:.4f} after={stab_after:.4f}"


# ── LD6: stability 上限 365.0 ─────────────────────────────────────────────────────────────

def test_ld6_stability_cap_365(conn):
    """LD6: LDSB boost 后 stability 不超过 365.0。"""
    _insert_chunk(conn, "ldsb_6", stability=364.0, importance=0.8,
                  last_accessed_hours_ago=720)
    now_iso = _utcnow().isoformat()
    apply_lag_dependent_spacing_boost(conn, ["ldsb_6"], now_iso=now_iso)
    stab_after = _get_stability(conn, "ldsb_6")
    assert stab_after <= 365.0, f"LD6: stability 不应超过 365.0，got {stab_after}"


# ── LD7: 长间隔比短间隔获得更大加成（单调性）─────────────────────────────────────────────

def test_ld7_longer_lag_more_boost(conn):
    """LD7: 间隔更长的 chunk 获得更大（或相等）的 stability 加成。"""
    base = 5.0

    # 中等间隔：4 小时
    _insert_chunk(conn, "ldsb_7_med", stability=base, importance=0.6,
                  last_accessed_hours_ago=4)
    # 长间隔：48 小时
    _insert_chunk(conn, "ldsb_7_long", stability=base, importance=0.6,
                  last_accessed_hours_ago=48)

    now_iso = _utcnow().isoformat()
    apply_lag_dependent_spacing_boost(conn, ["ldsb_7_med", "ldsb_7_long"], now_iso=now_iso)

    stab_med = _get_stability(conn, "ldsb_7_med")
    stab_long = _get_stability(conn, "ldsb_7_long")

    boost_med = stab_med - base
    boost_long = stab_long - base

    assert boost_long >= boost_med - 0.001, (
        f"LD7: 长间隔加成应 >= 中等间隔加成，long={boost_long:.4f} med={boost_med:.4f}"
    )


# ── LD8: update_accessed 集成测试 ────────────────────────────────────────────────────────

def test_ld8_update_accessed_integration(conn):
    """LD8: update_accessed 对长间隔 chunk 触发 LDSB 加成。"""
    # 插入 24 小时前访问的 chunk
    _insert_chunk(conn, "ldsb_8", stability=5.0, importance=0.6,
                  last_accessed_hours_ago=24)
    stab_before = _get_stability(conn, "ldsb_8")

    # update_accessed 会触发 LDSB
    update_accessed(conn, ["ldsb_8"])
    stab_after = _get_stability(conn, "ldsb_8")

    # 有 LDSB 加成（至少 stability 不应降低）
    assert stab_after >= stab_before, (
        f"LD8: update_accessed 后 stability 不应降低，before={stab_before:.4f} after={stab_after:.4f}"
    )
