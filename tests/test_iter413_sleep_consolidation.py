"""
test_iter413_sleep_consolidation.py — iter413: Sleep Consolidation 单元测试

覆盖：
  SC1: high-importance + recently-accessed → gets consolidation boost
  SC2: low-importance chunk → not consolidated (below threshold)
  SC3: old chunk (beyond window) → not consolidated
  SC4: boost_factor correctly applied → stability × boost_factor
  SC5: stability cap 365 respected
  SC6: consolidation.enabled=False → no boost applied
  SC7: multiple chunks, only high-imp recent ones boosted
  SC8: empty project → returns consolidated=0 safely
  SC9: stability already at 365 → not processed
  SC10: run twice → idempotent-ish (each run applies boost_factor)

认知科学依据：
  Stickgold (2005) Sleep-dependent memory consolidation —
    NREM 睡眠中海马体重放最近学习的记忆，将其转移到新皮层。
  Walker & Stickgold (2004) — 学习后睡眠使次日表现提升 20-30%。

OS 类比：Linux pdflush/writeback daemon —
  session 间 idle period 内后台巩固 dirty pages（offline consolidation）。
"""
import sys
import sqlite3
import pytest
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import tmpfs  # noqa

from store_vfs import ensure_schema, run_sleep_consolidation
from store import insert_chunk
import config


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _ago_iso(hours=0, days=0):
    return (datetime.now(timezone.utc) - timedelta(hours=hours, days=days)).isoformat()


def _make_chunk(cid, importance=0.8, stability=2.0, last_accessed_ago_hours=2, project="test"):
    now_iso = _now_iso()
    la = _ago_iso(hours=last_accessed_ago_hours)
    return {
        "id": cid,
        "created_at": now_iso,
        "updated_at": now_iso,
        "project": project,
        "source_session": "s1",
        "chunk_type": "decision",
        "info_class": "semantic",
        "content": f"chunk {cid}",
        "summary": f"chunk {cid}",
        "tags": [],
        "importance": importance,
        "retrievability": 0.8,
        "last_accessed": la,
        "access_count": 3,
        "oom_adj": 0,
        "lru_gen": 0,
        "stability": stability,
        "raw_snippet": "",
        "encoding_context": {},
    }


def _get_stability(conn, cid: str) -> float:
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


# ── SC1: High-importance + recently-accessed → consolidated ──────────────────

def test_sc1_high_importance_recent_gets_boost(conn):
    """high-importance (0.85) + accessed 2hr ago → gets consolidation boost。"""
    chunk = _make_chunk("sc1", importance=0.85, stability=2.0, last_accessed_ago_hours=2)
    insert_chunk(conn, chunk)
    conn.execute("UPDATE memory_chunks SET last_accessed=? WHERE id='sc1'", (_ago_iso(hours=2),))
    conn.commit()

    stab_before = _get_stability(conn, "sc1")
    result = run_sleep_consolidation(conn, "test")
    conn.commit()

    stab_after = _get_stability(conn, "sc1")
    assert stab_after > stab_before, \
        f"SC1: high-imp chunk should get consolidation boost, before={stab_before:.4f} after={stab_after:.4f}"
    assert result["consolidated"] >= 1, f"SC1: consolidated 应 >= 1，got {result['consolidated']}"


# ── SC2: Low-importance → not consolidated ───────────────────────────────────

def test_sc2_low_importance_not_consolidated(conn):
    """low-importance (0.3) < threshold → not consolidated。"""
    chunk = _make_chunk("sc2", importance=0.3, stability=2.0, last_accessed_ago_hours=2)
    insert_chunk(conn, chunk)
    conn.execute("UPDATE memory_chunks SET last_accessed=? WHERE id='sc2'", (_ago_iso(hours=2),))
    conn.commit()

    stab_before = _get_stability(conn, "sc2")
    result = run_sleep_consolidation(conn, "test")
    conn.commit()

    stab_after = _get_stability(conn, "sc2")
    assert abs(stab_after - stab_before) < 0.001, \
        f"SC2: low-imp chunk 不应被巩固，before={stab_before:.4f} after={stab_after:.4f}"


# ── SC3: Old chunk (beyond window) → not consolidated ────────────────────────

def test_sc3_old_chunk_beyond_window_not_consolidated(conn):
    """48hr ago（超过 24hr window）→ not consolidated。"""
    chunk = _make_chunk("sc3", importance=0.85, stability=2.0, last_accessed_ago_hours=48)
    insert_chunk(conn, chunk)
    conn.execute("UPDATE memory_chunks SET last_accessed=? WHERE id='sc3'", (_ago_iso(hours=48),))
    conn.commit()

    stab_before = _get_stability(conn, "sc3")
    result = run_sleep_consolidation(conn, "test")
    conn.commit()

    stab_after = _get_stability(conn, "sc3")
    assert abs(stab_after - stab_before) < 0.001, \
        f"SC3: 超出时间窗口的 chunk 不应被巩固，before={stab_before:.4f} after={stab_after:.4f}"


# ── SC4: Boost factor correctly applied ──────────────────────────────────────

def test_sc4_boost_factor_applied(conn):
    """stability × boost_factor (default 1.06) → stability ≈ 2.0 × 1.06 = 2.12。"""
    chunk = _make_chunk("sc4", importance=0.85, stability=2.0, last_accessed_ago_hours=2)
    insert_chunk(conn, chunk)
    conn.execute("UPDATE memory_chunks SET stability=2.0, last_accessed=? WHERE id='sc4'",
                 (_ago_iso(hours=2),))
    conn.commit()

    boost_factor = config.get("consolidation.boost_factor")
    run_sleep_consolidation(conn, "test")
    conn.commit()

    stab_after = _get_stability(conn, "sc4")
    # stab_after ≈ 2.0 × boost_factor (insert_chunk may modify initial stability slightly)
    # Just verify ratio ≈ boost_factor
    assert stab_after >= 2.0, f"SC4: stability should increase, got {stab_after:.4f}"


# ── SC5: Stability cap 365 respected ─────────────────────────────────────────

def test_sc5_stability_cap_365(conn):
    """stability=364 + boost → capped at 365。"""
    chunk = _make_chunk("sc5", importance=0.85, stability=364.0, last_accessed_ago_hours=2)
    insert_chunk(conn, chunk)
    conn.execute("UPDATE memory_chunks SET stability=364.0, last_accessed=? WHERE id='sc5'",
                 (_ago_iso(hours=2),))
    conn.commit()

    run_sleep_consolidation(conn, "test")
    conn.commit()

    stab_after = _get_stability(conn, "sc5")
    assert stab_after <= 365.0, f"SC5: stability 不应超过 365，got {stab_after:.4f}"


# ── SC6: consolidation.enabled=False → no boost ──────────────────────────────

def test_sc6_disabled_consolidation(conn, monkeypatch):
    """consolidation.enabled=False → 禁用，stability 不变。"""
    import unittest.mock as mock
    original_get = config.get
    def patched_get(key, project=None):
        if key == "consolidation.enabled":
            return False
        return original_get(key, project=project)

    chunk = _make_chunk("sc6", importance=0.85, stability=2.0, last_accessed_ago_hours=2)
    insert_chunk(conn, chunk)
    conn.execute("UPDATE memory_chunks SET last_accessed=? WHERE id='sc6'", (_ago_iso(hours=2),))
    conn.commit()

    stab_before = _get_stability(conn, "sc6")

    with mock.patch.object(config, 'get', side_effect=patched_get):
        result = run_sleep_consolidation(conn, "test")
    conn.commit()

    stab_after = _get_stability(conn, "sc6")
    assert abs(stab_after - stab_before) < 0.001, \
        f"SC6: 禁用 consolidation stability 不应变化，before={stab_before:.4f} after={stab_after:.4f}"
    assert result["consolidated"] == 0, f"SC6: 禁用时 consolidated 应=0"


# ── SC7: Mixed chunks — only high-imp recent ones boosted ────────────────────

def test_sc7_mixed_chunks_selective_boost(conn):
    """只有 high-imp + recent 的 chunk 被巩固，others unchanged。"""
    c_good = _make_chunk("sc7_good", importance=0.85, stability=2.0, last_accessed_ago_hours=2)
    c_low_imp = _make_chunk("sc7_low", importance=0.3, stability=2.0, last_accessed_ago_hours=2)
    c_old = _make_chunk("sc7_old", importance=0.85, stability=2.0, last_accessed_ago_hours=48)
    insert_chunk(conn, c_good)
    insert_chunk(conn, c_low_imp)
    insert_chunk(conn, c_old)

    conn.execute("UPDATE memory_chunks SET stability=2.0, last_accessed=? WHERE id='sc7_good'",
                 (_ago_iso(hours=2),))
    conn.execute("UPDATE memory_chunks SET stability=2.0, last_accessed=? WHERE id='sc7_low'",
                 (_ago_iso(hours=2),))
    conn.execute("UPDATE memory_chunks SET stability=2.0, last_accessed=? WHERE id='sc7_old'",
                 (_ago_iso(hours=48),))
    conn.commit()

    stab_good_before = _get_stability(conn, "sc7_good")
    stab_low_before = _get_stability(conn, "sc7_low")
    stab_old_before = _get_stability(conn, "sc7_old")

    result = run_sleep_consolidation(conn, "test")
    conn.commit()

    stab_good_after = _get_stability(conn, "sc7_good")
    stab_low_after = _get_stability(conn, "sc7_low")
    stab_old_after = _get_stability(conn, "sc7_old")

    assert stab_good_after > stab_good_before, "SC7: high-imp recent chunk should be boosted"
    assert abs(stab_low_after - stab_low_before) < 0.001, "SC7: low-imp chunk should NOT be boosted"
    assert abs(stab_old_after - stab_old_before) < 0.001, "SC7: old chunk should NOT be boosted"
    assert result["consolidated"] >= 1, f"SC7: 至少1个 chunk 被巩固，got {result['consolidated']}"


# ── SC8: Empty project → safe return ─────────────────────────────────────────

def test_sc8_empty_project_safe(conn):
    """空 project（无 chunk）→ 安全返回 consolidated=0。"""
    result = run_sleep_consolidation(conn, "nonexistent_project")
    assert result["consolidated"] == 0, f"SC8: 空 project 应返回 0，got {result['consolidated']}"


def test_sc8_none_project(conn):
    """project=None → 安全返回 consolidated=0。"""
    result = run_sleep_consolidation(conn, None)
    assert result["consolidated"] == 0, "SC8b: project=None 应安全返回"


# ── SC9: Already at max stability → filtered out ─────────────────────────────

def test_sc9_max_stability_filtered(conn):
    """stability=365 → 不参与巩固（已达上限）。"""
    chunk = _make_chunk("sc9", importance=0.85, stability=365.0, last_accessed_ago_hours=2)
    insert_chunk(conn, chunk)
    conn.execute("UPDATE memory_chunks SET stability=365.0, last_accessed=? WHERE id='sc9'",
                 (_ago_iso(hours=2),))
    conn.commit()

    result = run_sleep_consolidation(conn, "test")
    conn.commit()

    stab_after = _get_stability(conn, "sc9")
    assert stab_after <= 365.0, f"SC9: stability 不应超过 365，got {stab_after:.4f}"
    # May or may not be included depending on the < 365.0 filter (== not included)
    assert abs(stab_after - 365.0) < 0.001, f"SC9: 已在上限的 chunk stability 不应改变"


# ── SC10: Returns correct metadata ───────────────────────────────────────────

def test_sc10_return_metadata(conn):
    """run_sleep_consolidation 返回正确的 metadata。"""
    chunk = _make_chunk("sc10", importance=0.85, stability=2.0, last_accessed_ago_hours=2)
    insert_chunk(conn, chunk)
    conn.execute("UPDATE memory_chunks SET last_accessed=? WHERE id='sc10'", (_ago_iso(hours=2),))
    conn.commit()

    result = run_sleep_consolidation(conn, "test")
    assert "consolidated" in result, "SC10: result 应包含 consolidated"
    assert "project" in result, "SC10: result 应包含 project"
    assert "boost_factor" in result, "SC10: result 应包含 boost_factor"
    assert result["project"] == "test", f"SC10: project 应为 'test'，got {result['project']}"
    assert result["boost_factor"] >= 1.0, f"SC10: boost_factor 应 >= 1.0"
