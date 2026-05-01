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
import config


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _insert_raw(conn, cid, stability=2.0, last_accessed_ago_secs=0, project="test"):
    """直接 SQL 插入，bypass insert_chunk 写入效应，只测 reconsolidation window 逻辑。"""
    now = datetime.now(timezone.utc)
    la = (now - timedelta(seconds=last_accessed_ago_secs)).isoformat()
    now_iso = now.isoformat()
    conn.execute("""
        INSERT OR REPLACE INTO memory_chunks
            (id, created_at, updated_at, project, source_session,
             chunk_type, info_class, content, summary, tags,
             importance, retrievability, last_accessed, access_count,
             oom_adj, lru_gen, stability, raw_snippet, encoding_context)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (cid, now_iso, now_iso, project, "s1",
          "decision", "semantic",
          f"test chunk {cid}", f"test chunk {cid}", "[]",
          0.8, 0.9, la, 2,
          0, 0, stability, "", "{}"))
    conn.commit()


def _get_stability(conn, cid: str) -> float:
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


# ── RW1: gap < 1hr → quality=3，stability 不变 (×1.0) ─────────────────────

def test_rw1_short_gap_no_gain(conn):
    """gap < 1hr → SM-2 quality=3（stability × 1.0）。"""
    _insert_raw(conn, "rw1", last_accessed_ago_secs=30, stability=2.0)
    stab_before = _get_stability(conn, "rw1")
    update_accessed(conn, ["rw1"], _sm2_only=True)
    conn.commit()
    stab = _get_stability(conn, "rw1")
    ratio = stab / stab_before if stab_before else 0
    assert abs(ratio - 1.0) < 0.05, \
        f"短间隔（30s）质量=3，stability ratio 应 ≈ 1.0，got ratio={ratio:.4f}"


# ── RW2: 1hr <= gap < 24hr → quality=4 (×1.1) ────────────────────────────

def test_rw2_medium_gap_light_boost(conn):
    """1hr <= gap < 24hr → SM-2 quality=4（stability × 1.1）。"""
    _insert_raw(conn, "rw2", last_accessed_ago_secs=6*3600, stability=2.0)
    stab_before = _get_stability(conn, "rw2")
    update_accessed(conn, ["rw2"], _sm2_only=True)
    conn.commit()
    stab = _get_stability(conn, "rw2")
    ratio = stab / stab_before if stab_before else 0
    assert abs(ratio - 1.1) < 0.1, \
        f"中间隔（6hr）质量=4，stability ratio 应 ≈ 1.1，got ratio={ratio:.4f}"


# ── RW3: gap >= 24hr → quality=5 (×1.2) ──────────────────────────────────

def test_rw3_long_gap_max_boost(conn):
    """gap >= 24hr → SM-2 quality=5（stability × 1.2）。"""
    _insert_raw(conn, "rw3", last_accessed_ago_secs=3*86400, stability=2.0)
    stab_before = _get_stability(conn, "rw3")
    update_accessed(conn, ["rw3"], _sm2_only=True)
    conn.commit()
    stab = _get_stability(conn, "rw3")
    ratio = stab / stab_before if stab_before else 0
    assert abs(ratio - 1.2) < 0.1, \
        f"长间隔（3天）质量=5，stability ratio 应 ≈ 1.2，got ratio={ratio:.4f}"


# ── RW4: 显式 recall_quality 优先 ─────────────────────────────────────────

def test_rw4_explicit_quality_overrides_recon(conn):
    """显式 recall_quality=3 优先于再巩固窗口推断。"""
    _insert_raw(conn, "rw4", last_accessed_ago_secs=3*86400, stability=2.0)
    stab_before = _get_stability(conn, "rw4")
    update_accessed(conn, ["rw4"], recall_quality=3, _sm2_only=True)
    conn.commit()
    stab = _get_stability(conn, "rw4")
    ratio = stab / stab_before if stab_before else 0
    assert abs(ratio - 1.0) < 0.05, \
        f"显式 quality=3 优先，stability ratio 应不变，got ratio={ratio:.4f}"


# ── RW5: recon.enabled=False → 固定 quality=4 ─────────────────────────────

def test_rw5_disabled_recon_fallback_quality4(conn, monkeypatch):
    """recon.enabled=False → 回退到固定 quality=4（stability × 1.1）。"""
    _insert_raw(conn, "rw5", last_accessed_ago_secs=3*86400, stability=2.0)
    stab_before = _get_stability(conn, "rw5")

    original_get = config.get
    def patched_get(key, project=None):
        if key == "recon.enabled":
            return False
        return original_get(key, project=project)

    import unittest.mock as mock
    with mock.patch.object(config, 'get', side_effect=patched_get):
        update_accessed(conn, ["rw5"], _sm2_only=True)
    conn.commit()

    stab = _get_stability(conn, "rw5")
    ratio = stab / stab_before if stab_before else 0
    assert abs(ratio - 1.1) < 0.1, \
        f"禁用 recon 时 quality=4 fallback，ratio 应 ≈ 1.1，got ratio={ratio:.4f}"


# ── RW6: last_accessed=None → 安全退化 quality=4 ─────────────────────────

def test_rw6_null_last_accessed_safe_default(conn):
    """last_accessed=None（新 chunk）→ 安全退化 quality=4（×1.1）。"""
    _insert_raw(conn, "rw6", stability=2.0, last_accessed_ago_secs=0)
    conn.execute("UPDATE memory_chunks SET last_accessed=NULL WHERE id='rw6'")
    conn.commit()

    update_accessed(conn, ["rw6"], _sm2_only=True)
    conn.commit()

    stab = _get_stability(conn, "rw6")
    # NULL last_accessed → default quality=4 → ×1.1 → stability ≈ 2.2
    assert stab >= 2.0, f"NULL last_accessed 不应出错，got stability={stab}"


# ── RW7: 混合间隔批量更新 ─────────────────────────────────────────────────

def test_rw7_mixed_gaps_batch_update(conn):
    """混合间隔批量更新：不同 chunk 获得不同 quality，稳定性各异。"""
    _insert_raw(conn, "short_c",  stability=3.0, last_accessed_ago_secs=60)
    _insert_raw(conn, "medium_c", stability=3.0, last_accessed_ago_secs=12*3600)
    _insert_raw(conn, "long_c",   stability=3.0, last_accessed_ago_secs=7*86400)

    update_accessed(conn, ["short_c", "medium_c", "long_c"], _sm2_only=True)
    conn.commit()

    stab_short  = _get_stability(conn, "short_c")
    stab_medium = _get_stability(conn, "medium_c")
    stab_long   = _get_stability(conn, "long_c")

    assert stab_short <= stab_medium, \
        f"短间隔 stability 应 ≤ 中间隔：{stab_short:.3f} ≤ {stab_medium:.3f}"
    assert stab_medium <= stab_long, \
        f"中间隔 stability 应 ≤ 长间隔：{stab_medium:.3f} ≤ {stab_long:.3f}"
    assert stab_long > stab_short, \
        f"长间隔 stability 应 > 短间隔：{stab_long:.3f} > {stab_short:.3f}"


# ── RW8: stability 上限 365 天 ────────────────────────────────────────────

def test_rw8_stability_capped_at_365(conn):
    """stability 上限 365 天（即使 quality=5 也不超过）。"""
    _insert_raw(conn, "rw8", stability=364.0, last_accessed_ago_secs=7*86400)

    update_accessed(conn, ["rw8"], _sm2_only=True)
    conn.commit()

    stab = _get_stability(conn, "rw8")
    assert stab <= 365.0, f"stability 不应超过 365 天上限，got {stab}"
    assert stab > 364.0, f"stability 应增加（quality=5×1.2），got {stab}"
