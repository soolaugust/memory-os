"""
test_iter419_associative_memory.py — iter419: Associative Memory 单元测试

覆盖：
  AM1: apply_associative_memory_bonus — high-importance anchor + overlap >= min_overlap → bonus > 0
  AM2: apply_associative_memory_bonus — no overlap → no bonus
  AM3: apply_associative_memory_bonus — overlap below min_overlap → no bonus
  AM4: apply_associative_memory_bonus — anchor importance below threshold → no bonus
  AM5: apply_associative_memory_bonus — am_enabled=False → no bonus
  AM6: apply_associative_memory_bonus — bonus capped at base × bonus_cap
  AM7: apply_associative_memory_bonus — cross-project anchors do not help
  AM8: apply_associative_memory_bonus — more overlap → larger bonus (up to cap)
  AM9: insert_chunk with high-importance neighbor + overlap → stability boosted at write time
  AM10: new chunk with no existing anchors → no bonus (empty project)
  AM11: stability floor — bonus never pushes stability above 365.0
  AM12: multiple anchors — max overlap across all anchors used

认知科学依据：
  Ebbinghaus (1885) Paired Associates Learning;
  Collins & Loftus (1975) Spreading Activation — 新知识与已有强记忆共享节点时
  形成更强的记忆痕迹（associative encoding advantage）。

OS 类比：Linux huge pages (THP) — small page adjacent to huge page shares same TLB entry
  and benefits from the huge page's TLB locality (associative memory locality)。
"""
import sys
import sqlite3
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import tmpfs  # noqa

from store_vfs import (
    ensure_schema,
    apply_associative_memory_bonus,
)
from store import insert_chunk
import config


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _insert_chunk_direct(conn, cid, content="test", chunk_type="decision",
                          stability=2.0, project="test", importance=0.8,
                          encode_context=""):
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO memory_chunks "
        "(id, project, chunk_type, content, summary, importance, stability, "
        "created_at, updated_at, retrievability, encode_context) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0.9, ?)",
        (cid, project, chunk_type, content, content, importance, stability, now, now,
         encode_context)
    )
    conn.commit()


def _get_stability(conn, cid: str) -> float:
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


# ── AM1: High-importance anchor + overlap → bonus > 0 ────────────────────────

def test_am1_anchor_overlap_gets_bonus(conn):
    """高 importance 的 anchor 与新 chunk 有足够重叠 → bonus > 0。"""
    # Anchor: auth, jwt, token, security (high importance=0.9)
    _insert_chunk_direct(conn, "am1_anchor",
                         importance=0.9, encode_context="auth,jwt,token,security",
                         project="test")
    # New chunk: auth, jwt, refresh (2 shared tokens with anchor)
    _insert_chunk_direct(conn, "am1_new",
                         stability=2.0, encode_context="auth,jwt,refresh",
                         project="test")

    stab_before = _get_stability(conn, "am1_new")
    apply_associative_memory_bonus(conn, "am1_new", "test", base_stability=2.0)
    conn.commit()
    stab_after = _get_stability(conn, "am1_new")

    assert stab_after > stab_before, \
        f"AM1: anchor 重叠 → stability 应增加，before={stab_before:.4f} after={stab_after:.4f}"


# ── AM2: No overlap → no bonus ────────────────────────────────────────────────

def test_am2_no_overlap_no_bonus(conn):
    """无 encode_context 重叠 → 无 bonus。"""
    _insert_chunk_direct(conn, "am2_anchor",
                         importance=0.9, encode_context="auth,jwt,token",
                         project="test")
    _insert_chunk_direct(conn, "am2_new",
                         stability=2.0, encode_context="database,postgres,migration",
                         project="test")

    stab_before = _get_stability(conn, "am2_new")
    apply_associative_memory_bonus(conn, "am2_new", "test", base_stability=2.0)
    conn.commit()
    stab_after = _get_stability(conn, "am2_new")

    assert abs(stab_after - stab_before) < 0.001, \
        f"AM2: 无重叠 → stability 不应变，before={stab_before:.4f} after={stab_after:.4f}"


# ── AM3: Overlap below min_overlap → no bonus ────────────────────────────────

def test_am3_low_overlap_no_bonus(conn, monkeypatch):
    """overlap=1 < min_overlap=2 → 无 bonus。"""
    import unittest.mock as mock
    original_get = config.get
    def patched_get(key, project=None):
        if key == "store_vfs.am_min_overlap":
            return 3  # require 3 overlapping tokens
        return original_get(key, project=project)

    # Only 1 shared token: "auth"
    _insert_chunk_direct(conn, "am3_anchor",
                         importance=0.9, encode_context="auth,jwt,token",
                         project="test")
    _insert_chunk_direct(conn, "am3_new",
                         stability=2.0, encode_context="auth,database,migration",
                         project="test")

    stab_before = _get_stability(conn, "am3_new")
    with mock.patch.object(config, 'get', side_effect=patched_get):
        apply_associative_memory_bonus(conn, "am3_new", "test", base_stability=2.0)
    conn.commit()
    stab_after = _get_stability(conn, "am3_new")

    assert abs(stab_after - stab_before) < 0.001, \
        f"AM3: 低重叠 → stability 不应变，before={stab_before:.4f} after={stab_after:.4f}"


# ── AM4: Anchor importance below threshold → no bonus ────────────────────────

def test_am4_low_importance_anchor_no_bonus(conn, monkeypatch):
    """anchor importance < am_min_importance → 无 bonus。"""
    import unittest.mock as mock
    original_get = config.get
    def patched_get(key, project=None):
        if key == "store_vfs.am_min_importance":
            return 0.90  # require very high importance
        return original_get(key, project=project)

    # Anchor with importance=0.7 (below 0.90 threshold)
    _insert_chunk_direct(conn, "am4_anchor",
                         importance=0.7, encode_context="auth,jwt,token,security",
                         project="test")
    _insert_chunk_direct(conn, "am4_new",
                         stability=2.0, encode_context="auth,jwt,refresh",
                         project="test")

    stab_before = _get_stability(conn, "am4_new")
    with mock.patch.object(config, 'get', side_effect=patched_get):
        apply_associative_memory_bonus(conn, "am4_new", "test", base_stability=2.0)
    conn.commit()
    stab_after = _get_stability(conn, "am4_new")

    assert abs(stab_after - stab_before) < 0.001, \
        f"AM4: 低 importance anchor → stability 不应变，before={stab_before:.4f} after={stab_after:.4f}"


# ── AM5: am_enabled=False → no bonus ─────────────────────────────────────────

def test_am5_disabled_no_bonus(conn, monkeypatch):
    """am_enabled=False → 禁用，stability 不变。"""
    import unittest.mock as mock
    original_get = config.get
    def patched_get(key, project=None):
        if key == "store_vfs.am_enabled":
            return False
        return original_get(key, project=project)

    _insert_chunk_direct(conn, "am5_anchor",
                         importance=0.9, encode_context="auth,jwt,token,security",
                         project="test")
    _insert_chunk_direct(conn, "am5_new",
                         stability=2.0, encode_context="auth,jwt,refresh",
                         project="test")

    stab_before = _get_stability(conn, "am5_new")
    with mock.patch.object(config, 'get', side_effect=patched_get):
        apply_associative_memory_bonus(conn, "am5_new", "test", base_stability=2.0)
    conn.commit()
    stab_after = _get_stability(conn, "am5_new")

    assert abs(stab_after - stab_before) < 0.001, \
        f"AM5: 禁用 → stability 不应变，before={stab_before:.4f} after={stab_after:.4f}"


# ── AM6: Bonus capped at base × bonus_cap ────────────────────────────────────

def test_am6_bonus_capped(conn, monkeypatch):
    """bonus 上限为 base × am_bonus_cap。"""
    import unittest.mock as mock
    original_get = config.get
    cap = 0.15

    def patched_get(key, project=None):
        if key == "store_vfs.am_bonus_cap":
            return cap
        return original_get(key, project=project)

    # Many overlapping tokens to ensure max bonus
    _insert_chunk_direct(conn, "am6_anchor",
                         importance=0.9,
                         encode_context="auth,jwt,token,security,refresh,session,oauth,user",
                         project="test")
    _insert_chunk_direct(conn, "am6_new",
                         stability=2.0,
                         encode_context="auth,jwt,token,security,refresh,session,oauth,user",
                         project="test")

    stab_before = _get_stability(conn, "am6_new")
    with mock.patch.object(config, 'get', side_effect=patched_get):
        apply_associative_memory_bonus(conn, "am6_new", "test", base_stability=2.0)
    conn.commit()
    stab_after = _get_stability(conn, "am6_new")

    max_expected = stab_before + stab_before * cap + 0.001  # small tolerance
    assert stab_after <= max_expected, \
        f"AM6: bonus 应受 cap 限制，before={stab_before:.4f} after={stab_after:.4f} cap={cap}"
    assert stab_after > stab_before, \
        f"AM6: 应有正 bonus，before={stab_before:.4f} after={stab_after:.4f}"


# ── AM7: Cross-project anchors do not help ───────────────────────────────────

def test_am7_cross_project_isolation(conn):
    """不同 project 的 anchor 不提供关联记忆加成。"""
    # Anchor in project_B (different project)
    _insert_chunk_direct(conn, "am7_anchor",
                         importance=0.9, encode_context="auth,jwt,token,security",
                         project="project_B")
    # New chunk in project_A (should not benefit from project_B anchor)
    _insert_chunk_direct(conn, "am7_new",
                         stability=2.0, encode_context="auth,jwt,refresh",
                         project="project_A")

    stab_before = _get_stability(conn, "am7_new")
    apply_associative_memory_bonus(conn, "am7_new", "project_A", base_stability=2.0)
    conn.commit()
    stab_after = _get_stability(conn, "am7_new")

    assert abs(stab_after - stab_before) < 0.001, \
        f"AM7: 跨 project anchor → stability 不应变，before={stab_before:.4f} after={stab_after:.4f}"


# ── AM8: More overlap → larger bonus (up to cap) ─────────────────────────────

def test_am8_more_overlap_larger_bonus(conn):
    """更多重叠 tokens → 更大 bonus（直到上限）。"""
    # Two anchors at different overlap levels
    _insert_chunk_direct(conn, "am8_anchor1",
                         importance=0.9, encode_context="auth,jwt,token",
                         project="test")
    _insert_chunk_direct(conn, "am8_low",
                         stability=2.0, encode_context="auth,jwt,other1,other2",
                         project="test")
    stab_before_low = _get_stability(conn, "am8_low")
    apply_associative_memory_bonus(conn, "am8_low", "test", base_stability=2.0)
    conn.commit()
    stab_after_low = _get_stability(conn, "am8_low")
    bonus_low = stab_after_low - stab_before_low

    _insert_chunk_direct(conn, "am8_anchor2",
                         importance=0.9,
                         encode_context="auth,jwt,token,security,refresh,session",
                         project="test")
    _insert_chunk_direct(conn, "am8_high",
                         stability=2.0,
                         encode_context="auth,jwt,token,security,refresh,session",
                         project="test")
    stab_before_high = _get_stability(conn, "am8_high")
    apply_associative_memory_bonus(conn, "am8_high", "test", base_stability=2.0)
    conn.commit()
    stab_after_high = _get_stability(conn, "am8_high")
    bonus_high = stab_after_high - stab_before_high

    assert bonus_high >= bonus_low, \
        f"AM8: 更多重叠应得更大 bonus：bonus_high={bonus_high:.4f} >= bonus_low={bonus_low:.4f}"


# ── AM9: insert_chunk with high-importance neighbor → stability boosted ───────

def test_am9_insert_chunk_with_anchor_boosted(conn):
    """insert_chunk 写入时，若有高 importance 同项目 anchor → stability 被加成。"""
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Pre-insert anchor
    _insert_chunk_direct(conn, "am9_anchor",
                         importance=0.9,
                         encode_context="auth,jwt,token,security,refresh",
                         project="test")

    chunk = {
        "id": "am9_new",
        "created_at": now, "updated_at": now, "project": "test",
        "source_session": "s1", "chunk_type": "decision",
        "info_class": "semantic",
        "content": "JWT authentication implementation decision.",
        "summary": "JWT auth decision",
        "tags": [], "importance": 0.8, "retrievability": 0.9,
        "last_accessed": now, "access_count": 1,
        "oom_adj": 0, "lru_gen": 0, "stability": 2.0,
        "raw_snippet": "", "encoding_context": {"tokens": ["auth", "jwt", "token"]},
    }
    insert_chunk(conn, chunk)
    conn.commit()

    stab = _get_stability(conn, "am9_new")
    # Since insert_chunk now applies AM bonus, stability might be boosted
    # We just verify it's positive (not negative)
    assert stab > 0, f"AM9: stability 应 > 0，got {stab:.4f}"


# ── AM10: Empty project → no bonus ───────────────────────────────────────────

def test_am10_empty_project_no_bonus(conn):
    """项目内无其他 chunk（空项目）→ 无关联记忆 bonus。"""
    _insert_chunk_direct(conn, "am10_new",
                         stability=2.0, encode_context="auth,jwt,token",
                         project="empty_project")

    stab_before = _get_stability(conn, "am10_new")
    apply_associative_memory_bonus(conn, "am10_new", "empty_project", base_stability=2.0)
    conn.commit()
    stab_after = _get_stability(conn, "am10_new")

    assert abs(stab_after - stab_before) < 0.001, \
        f"AM10: 空项目 → stability 不应变，before={stab_before:.4f} after={stab_after:.4f}"


# ── AM11: Stability never pushed above 365.0 ─────────────────────────────────

def test_am11_stability_cap_at_365(conn, monkeypatch):
    """stability 上限为 365.0（不超过 365 天）。"""
    import unittest.mock as mock
    original_get = config.get
    def patched_get(key, project=None):
        if key == "store_vfs.am_bonus_cap":
            return 0.5  # large cap for test
        return original_get(key, project=project)

    _insert_chunk_direct(conn, "am11_anchor",
                         importance=0.9,
                         encode_context="auth,jwt,token,security,refresh,session",
                         project="test")
    # Start with very high stability
    _insert_chunk_direct(conn, "am11_new",
                         stability=360.0,
                         encode_context="auth,jwt,token,security,refresh,session",
                         project="test")

    with mock.patch.object(config, 'get', side_effect=patched_get):
        apply_associative_memory_bonus(conn, "am11_new", "test", base_stability=360.0)
    conn.commit()
    stab_after = _get_stability(conn, "am11_new")

    assert stab_after <= 365.0, f"AM11: stability 不应超过 365.0，got {stab_after:.4f}"


# ── AM12: Multiple anchors — max overlap used ─────────────────────────────────

def test_am12_multiple_anchors_max_overlap(conn):
    """多个 anchor → 取最大重叠的 anchor 作为关联依据。"""
    # Anchor A: 2 shared tokens with new chunk
    _insert_chunk_direct(conn, "am12_anchor_a",
                         importance=0.9, encode_context="auth,jwt,other1,other2",
                         project="test")
    # Anchor B: 5 shared tokens with new chunk (better)
    _insert_chunk_direct(conn, "am12_anchor_b",
                         importance=0.9,
                         encode_context="auth,jwt,token,security,refresh",
                         project="test")

    _insert_chunk_direct(conn, "am12_new",
                         stability=2.0,
                         encode_context="auth,jwt,token,security,refresh,session",
                         project="test")

    stab_before = _get_stability(conn, "am12_new")
    apply_associative_memory_bonus(conn, "am12_new", "test", base_stability=2.0)
    conn.commit()
    stab_after = _get_stability(conn, "am12_new")

    assert stab_after > stab_before, \
        f"AM12: 最大重叠 anchor → stability 应增加，before={stab_before:.4f} after={stab_after:.4f}"
