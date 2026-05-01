"""
test_iter482_seb.py — iter482: Spacing Effect Bonus 单元测试

覆盖：
  SE1: 间隔 >= seb_min_gap_hours(4) 后访问 → stability 加成
  SE2: 间隔 < seb_min_gap_hours → 无 SEB 加成
  SE3: seb_enabled=False → 无加成
  SE4: importance < seb_min_importance(0.25) → 不参与 SEB
  SE5: 间隔越长加成越大（对数增长）
  SE6: 加成受 seb_max_bonus(0.12) 上限保护
  SE7: stability 上限 365.0 保护
  SE8: 直接调用 apply_spacing_effect_bonus → seb_boosted > 0

认知科学依据：
  Ebbinghaus (1885) + Cepeda et al. (2006) meta-analysis (n=317, d=0.70) —
    间隔越长，每次访问带来的 long-term retention 增益越大。

OS 类比：Linux page access bit TLB aging — 距上次访问越久，下次命中优先级越高。
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

from store_vfs import ensure_schema, apply_spacing_effect_bonus
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


def _insert_with_last_accessed(conn, cid, hours_ago=0, importance=0.6, stability=5.0):
    """插入 chunk，last_accessed 设为 hours_ago 小时前。"""
    import datetime as dt
    now = dt.datetime.now(dt.timezone.utc)
    last_acc = (now - dt.timedelta(hours=hours_ago)).isoformat()
    now_iso = now.isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count,
            encode_context, session_type_history)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (cid, "test", "observation", "content " + cid, "summary", importance, stability,
         last_acc, last_acc, 0.5, last_acc, 1, "test_ctx", "")
    )
    conn.commit()


def _get_stability(conn, cid: str) -> float:
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


# ── SE1: 间隔 >= min_gap → stability 加成 ───────────────────────────────────────────────────

def test_se1_long_gap_boosted(conn):
    """SE1: 间隔 >= seb_min_gap_hours(4) → SEB stability 加成。"""
    _insert_with_last_accessed(conn, "se1", hours_ago=8)  # 8小时前

    stab_before = _get_stability(conn, "se1")
    result = apply_spacing_effect_bonus(conn, ["se1"])
    stab_after = _get_stability(conn, "se1")

    assert stab_after > stab_before, (
        f"SE1: 长间隔后访问应获得 SEB 加成，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["seb_boosted"] > 0, f"SE1: seb_boosted 应 > 0，got {result}"


# ── SE2: 间隔 < min_gap → 无加成 ────────────────────────────────────────────────────────────

def test_se2_short_gap_no_boost(conn):
    """SE2: 间隔 < seb_min_gap_hours(4) → 无 SEB 加成。"""
    _insert_with_last_accessed(conn, "se2", hours_ago=1)  # 1小时前（< 4h阈值）

    stab_before = _get_stability(conn, "se2")
    result = apply_spacing_effect_bonus(conn, ["se2"])
    stab_after = _get_stability(conn, "se2")

    assert abs(stab_after - stab_before) < 0.001, (
        f"SE2: 短间隔不应有 SEB 加成，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["seb_boosted"] == 0, f"SE2: seb_boosted 应为 0"


# ── SE3: seb_enabled=False → 无加成 ─────────────────────────────────────────────────────────

def test_se3_disabled_no_boost(conn):
    """SE3: seb_enabled=False → 无 SEB 加成。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.seb_enabled":
            return False
        return original_get(key, project=project)

    _insert_with_last_accessed(conn, "se3", hours_ago=12)

    stab_before = _get_stability(conn, "se3")
    with mock.patch.object(config, 'get', side_effect=patched_get):
        result = apply_spacing_effect_bonus(conn, ["se3"])
    stab_after = _get_stability(conn, "se3")

    assert abs(stab_after - stab_before) < 0.001, (
        f"SE3: disabled 时不应有 SEB 加成，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["seb_boosted"] == 0


# ── SE4: importance 不足 → 不参与 SEB ────────────────────────────────────────────────────────

def test_se4_low_importance_no_boost(conn):
    """SE4: importance < seb_min_importance(0.25) → 不参与 SEB。"""
    _insert_with_last_accessed(conn, "se4", hours_ago=8, importance=0.10)

    stab_before = _get_stability(conn, "se4")
    result = apply_spacing_effect_bonus(conn, ["se4"])
    stab_after = _get_stability(conn, "se4")

    assert abs(stab_after - stab_before) < 0.001, (
        f"SE4: 低 importance 不应触发 SEB，before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── SE5: 间隔越长加成越大 ────────────────────────────────────────────────────────────────────

def test_se5_longer_gap_more_boost(conn):
    """SE5: 间隔越长，SEB 加成越大（对数增长关系）。"""
    _insert_with_last_accessed(conn, "se5_8h", hours_ago=8)    # 8小时
    _insert_with_last_accessed(conn, "se5_48h", hours_ago=48)  # 48小时

    stab_8h_before = _get_stability(conn, "se5_8h")
    stab_48h_before = _get_stability(conn, "se5_48h")

    apply_spacing_effect_bonus(conn, ["se5_8h"])
    apply_spacing_effect_bonus(conn, ["se5_48h"])

    stab_8h_after = _get_stability(conn, "se5_8h")
    stab_48h_after = _get_stability(conn, "se5_48h")

    gain_8h = stab_8h_after - stab_8h_before
    gain_48h = stab_48h_after - stab_48h_before

    assert gain_48h >= gain_8h - 0.001, (
        f"SE5: 48h 间隔加成应 >= 8h 间隔，gain_8h={gain_8h:.4f} gain_48h={gain_48h:.4f}"
    )


# ── SE6: 加成受 seb_max_bonus 保护 ──────────────────────────────────────────────────────────

def test_se6_max_bonus_cap(conn):
    """SE6: SEB 加成受 seb_max_bonus(0.12) 上限保护。"""
    seb_max_bonus = config.get("store_vfs.seb_max_bonus")  # 0.12
    base = 5.0

    _insert_with_last_accessed(conn, "se6", hours_ago=10000, stability=base)  # 极长间隔

    stab_before = _get_stability(conn, "se6")
    apply_spacing_effect_bonus(conn, ["se6"])
    stab_after = _get_stability(conn, "se6")

    increment = stab_after - stab_before
    max_allowed = base * seb_max_bonus + 0.01
    assert increment <= max_allowed, (
        f"SE6: SEB 增量 {increment:.4f} 不应超过 max_bonus 允许的 {max_allowed:.4f}"
    )
    assert stab_after > stab_before, f"SE6: 应有 SEB 加成"


# ── SE7: stability 上限 365.0 ────────────────────────────────────────────────────────────────

def test_se7_stability_cap_365(conn):
    """SE7: SEB 加成后 stability 不超过 365.0。"""
    _insert_with_last_accessed(conn, "se7", hours_ago=100, stability=364.0)

    apply_spacing_effect_bonus(conn, ["se7"])
    stab = _get_stability(conn, "se7")
    assert stab <= 365.0, f"SE7: stability 不应超过 365.0，got {stab}"


# ── SE8: 直接调用返回 seb_boosted > 0 ────────────────────────────────────────────────────────

def test_se8_direct_function_boost(conn):
    """SE8: apply_spacing_effect_bonus 直接调用返回 seb_boosted > 0。"""
    _insert_with_last_accessed(conn, "se8", hours_ago=24)

    stab_before = _get_stability(conn, "se8")
    result = apply_spacing_effect_bonus(conn, ["se8"])
    stab_after = _get_stability(conn, "se8")

    assert result["seb_boosted"] > 0, f"SE8: seb_boosted 应 > 0，got {result}"
    assert stab_after > stab_before, (
        f"SE8: 应有 SEB 加成，before={stab_before:.4f} after={stab_after:.4f}"
    )
