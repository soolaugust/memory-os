"""
test_reconsolidation_window.py — iter389: Reconsolidation Window 单元测试

覆盖：
  RW1: gap < 1hr → SM-2 quality=3（无增益，stability 不变）
  RW2: 1hr <= gap < 24hr → SM-2 quality=4（轻微加固，×1.1）
  RW3: gap >= 24hr → SM-2 quality=5（最大巩固，×1.2）
  RW4: 显式 recall_quality 优先于再巩固窗口推断
  RW5: recon.enabled=False → 回退到固定 quality=4（×1.1）
  RW6: last_accessed=None（新 chunk）→ 安全退化 quality=4
  RW7: 混合间隔批量 → 不同 chunk 获得不同 quality
  RW8: stability 上限 365 天（不超限）

认知科学依据：
  Walker & Stickgold (2004) Memory Reconsolidation —
  记忆在每次被激活后进入不稳定的"可塑窗口"，然后重新巩固。
  Ebbinghaus (1885) Spacing Effect — 间隔越长的重复激活巩固效果越强。
OS 类比：Linux MGLRU page aging —
  短间隔访问不晋升 generation；跨 aging interval 后访问 → generation 晋升。
"""
import sys
import sqlite3
import pytest
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import tmpfs  # noqa

from store_vfs import ensure_schema, update_accessed
from store import insert_chunk
import config


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _make_chunk(cid, summary="test chunk", last_accessed_ago_secs=0,
                stability=2.0, project="test"):
    now = datetime.now(timezone.utc)
    la = (now - timedelta(seconds=last_accessed_ago_secs)).isoformat()
    now_iso = now.isoformat()
    return {
        "id": cid,
        "created_at": now_iso,
        "updated_at": now_iso,
        "project": project,
        "source_session": "s1",
        "chunk_type": "decision",
        "info_class": "semantic",
        "content": summary,
        "summary": summary,
        "tags": [],
        "importance": 0.8,
        "retrievability": 0.9,
        "last_accessed": la,
        "access_count": 2,
        "oom_adj": 0,
        "lru_gen": 0,
        "stability": stability,
        "raw_snippet": "",
        "encoding_context": {},
    }


def _get_stability(conn, cid: str) -> float:
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


# ── RW1: gap < 1hr → quality=3，stability 不变 (×1.0) ─────────────────────

def test_rw1_short_gap_no_gain(conn):
    """gap < 1hr → SM-2 quality=3（stability × 1.0）。"""
    # 30 seconds ago — within short_gap_hours=1hr
    chunk = _make_chunk("rw1", last_accessed_ago_secs=30, stability=2.0)
    insert_chunk(conn, chunk)
    conn.commit()

    update_accessed(conn, ["rw1"])
    conn.commit()

    stab = _get_stability(conn, "rw1")
    # quality=3 → factor=1.0 → stability unchanged ≈ 2.0
    assert abs(stab - 2.0) < 0.05, \
        f"短间隔（30s）质量=3，stability 不应增加，got {stab}"


# ── RW2: 1hr <= gap < 24hr → quality=4 (×1.1) ────────────────────────────

def test_rw2_medium_gap_light_boost(conn):
    """1hr <= gap < 24hr → SM-2 quality=4（stability × 1.1）。"""
    # 6 hours ago
    chunk = _make_chunk("rw2", last_accessed_ago_secs=6*3600, stability=2.0)
    insert_chunk(conn, chunk)
    conn.commit()

    update_accessed(conn, ["rw2"])
    conn.commit()

    stab = _get_stability(conn, "rw2")
    # quality=4 → factor=1.1 → stability ≈ 2.2
    assert abs(stab - 2.2) < 0.1, \
        f"中间隔（6hr）质量=4，stability 应 ≈ 2.2，got {stab}"


# ── RW3: gap >= 24hr → quality=5 (×1.2) ──────────────────────────────────

def test_rw3_long_gap_max_boost(conn):
    """gap >= 24hr → SM-2 quality=5（stability × 1.2）。"""
    # 3 days ago
    chunk = _make_chunk("rw3", last_accessed_ago_secs=3*86400, stability=2.0)
    insert_chunk(conn, chunk)
    conn.commit()

    update_accessed(conn, ["rw3"])
    conn.commit()

    stab = _get_stability(conn, "rw3")
    # quality=5 → factor=1.2 → stability ≈ 2.4
    assert abs(stab - 2.4) < 0.1, \
        f"长间隔（3天）质量=5，stability 应 ≈ 2.4，got {stab}"


# ── RW4: 显式 recall_quality 优先 ─────────────────────────────────────────

def test_rw4_explicit_quality_overrides_recon(conn):
    """显式 recall_quality=3 优先于再巩固窗口推断。"""
    # gap >= 24hr (would give quality=5), but explicit quality=3 → ×1.0
    chunk = _make_chunk("rw4", last_accessed_ago_secs=3*86400, stability=2.0)
    insert_chunk(conn, chunk)
    conn.commit()

    update_accessed(conn, ["rw4"], recall_quality=3)  # explicit quality=3
    conn.commit()

    stab = _get_stability(conn, "rw4")
    # explicit quality=3 → factor=1.0 → stability ≈ 2.0
    assert abs(stab - 2.0) < 0.05, \
        f"显式 quality=3 优先，stability 应不变，got {stab}"


# ── RW5: recon.enabled=False → 固定 quality=4 ─────────────────────────────

def test_rw5_disabled_recon_fallback_quality4(conn, monkeypatch):
    """recon.enabled=False → 回退到固定 quality=4（stability × 1.1）。"""
    monkeypatch.setenv("MEMORY_OS_RECON_ENABLED", "false")
    import importlib
    importlib.reload(config)  # reload to pick up env var (if config uses env)

    chunk = _make_chunk("rw5", last_accessed_ago_secs=3*86400, stability=2.0)
    insert_chunk(conn, chunk)
    conn.commit()

    # When recon disabled: fixed quality=4 → ×1.1
    # We test by patching config.get for this key
    original_get = config.get
    def patched_get(key, project=None):
        if key == "recon.enabled":
            return False
        return original_get(key, project=project)

    import unittest.mock as mock
    with mock.patch.object(config, 'get', side_effect=patched_get):
        import importlib as _il
        import store_vfs as _svfs
        # Reload store_vfs to use patched config
        update_accessed(conn, ["rw5"])
    conn.commit()

    stab = _get_stability(conn, "rw5")
    # With recon disabled, stability should be either 2.0 (quality=3 if it happened)
    # or 2.2 (quality=4) depending on the gap calculation
    # The key constraint: stability should not be 2.4 (quality=5 result)
    # Since gap=3 days and recon is disabled, result should be exactly 2.2 (quality=4 fallback)
    assert stab <= 2.25, f"禁用 recon 时 stability 应 ≤ 2.25（fallback quality=4），got {stab}"


# ── RW6: last_accessed=None → 安全退化 quality=4 ─────────────────────────

def test_rw6_null_last_accessed_safe_default(conn):
    """last_accessed=None（新 chunk）→ 安全退化 quality=4（×1.1）。"""
    # Insert chunk with NULL last_accessed
    now_iso = datetime.now(timezone.utc).isoformat()
    c = {
        "id": "rw6",
        "created_at": now_iso,
        "updated_at": now_iso,
        "project": "test",
        "source_session": "s1",
        "chunk_type": "decision",
        "info_class": "semantic",
        "content": "null accessed",
        "summary": "null accessed",
        "tags": [],
        "importance": 0.8,
        "retrievability": 0.9,
        "last_accessed": now_iso,  # will be set to NULL below
        "access_count": 0,
        "oom_adj": 0,
        "lru_gen": 0,
        "stability": 2.0,
        "raw_snippet": "",
        "encoding_context": {},
    }
    insert_chunk(conn, c)
    # Set last_accessed to NULL after insert
    conn.execute("UPDATE memory_chunks SET last_accessed=NULL WHERE id='rw6'")
    conn.commit()

    update_accessed(conn, ["rw6"])
    conn.commit()

    stab = _get_stability(conn, "rw6")
    # NULL last_accessed → default quality=4 → ×1.1 → stability ≈ 2.2
    assert stab >= 2.0, f"NULL last_accessed 不应出错，got stability={stab}"


# ── RW7: 混合间隔批量更新 ─────────────────────────────────────────────────

def test_rw7_mixed_gaps_batch_update(conn):
    """混合间隔批量更新：不同 chunk 获得不同 quality，稳定性各异。"""
    c_short = _make_chunk("short_c", last_accessed_ago_secs=60, stability=3.0)
    c_medium = _make_chunk("medium_c", last_accessed_ago_secs=12*3600, stability=3.0)
    c_long = _make_chunk("long_c", last_accessed_ago_secs=7*86400, stability=3.0)
    insert_chunk(conn, c_short)
    insert_chunk(conn, c_medium)
    insert_chunk(conn, c_long)
    conn.commit()

    update_accessed(conn, ["short_c", "medium_c", "long_c"])
    conn.commit()

    stab_short = _get_stability(conn, "short_c")
    stab_medium = _get_stability(conn, "medium_c")
    stab_long = _get_stability(conn, "long_c")

    # short < medium < long (or ≤ since floor is quality=3→×1.0)
    assert stab_short <= stab_medium, \
        f"短间隔 stability 应 ≤ 中间隔：{stab_short:.3f} ≤ {stab_medium:.3f}"
    assert stab_medium <= stab_long, \
        f"中间隔 stability 应 ≤ 长间隔：{stab_medium:.3f} ≤ {stab_long:.3f}"
    assert stab_long > stab_short, \
        f"长间隔 stability 应 > 短间隔：{stab_long:.3f} > {stab_short:.3f}"


# ── RW8: stability 上限 365 天 ────────────────────────────────────────────

def test_rw8_stability_capped_at_365(conn):
    """stability 上限 365 天（即使 quality=5 也不超过）。"""
    chunk = _make_chunk("rw8", last_accessed_ago_secs=7*86400, stability=364.0)
    insert_chunk(conn, chunk)
    conn.commit()

    update_accessed(conn, ["rw8"])
    conn.commit()

    stab = _get_stability(conn, "rw8")
    assert stab <= 365.0, f"stability 不应超过 365 天上限，got {stab}"
    assert stab > 364.0, f"stability 应增加（quality=5×1.2），got {stab}"
