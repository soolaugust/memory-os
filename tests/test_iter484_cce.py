"""
test_iter484_cce.py — iter484: Cross-Session Consolidation Effect 单元测试

覆盖：
  CC1: 跨 session 访问 + 间隔 >= cce_min_gap_hours(6) → stability 加成
  CC2: 同 session 访问 → 无 CCE 加成
  CC3: 跨 session 但间隔 < cce_min_gap_hours → 无加成
  CC4: cce_enabled=False → 无加成
  CC5: importance < cce_min_importance(0.25) → 不参与 CCE
  CC6: 间隔越长加成越大（最大在 24h 时）
  CC7: 加成受 cce_max_boost(0.10) 保护
  CC8: update_accessed 集成 — 跨 session 访问触发 CCE

认知科学依据：
  Walker & Stickgold (2004) Neuron — 睡眠期海马-皮质巩固，睡眠后记忆提升 6-12%。
  Stickgold (2005) Science: 睡眠是记忆巩固的积极过程。

OS 类比：Linux kswapd background reclaim — session 间隔期整理 page，下次访问效率提升。
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

from store_vfs import ensure_schema, apply_cross_session_consolidation, update_accessed
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


def _insert_with_session_and_access(conn, cid, source_session="sess1",
                                     hours_ago=0, importance=0.6, stability=5.0):
    import datetime as dt
    now = dt.datetime.now(dt.timezone.utc)
    last_acc = (now - dt.timedelta(hours=hours_ago)).isoformat()
    now_iso = now.isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count,
            encode_context, session_type_history, source_session)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (cid, "test", "observation", "content " + cid, "summary", importance, stability,
         last_acc, last_acc, 0.5, last_acc, 1, "test_ctx", "coding", source_session)
    )
    conn.commit()


def _get_stability(conn, cid: str) -> float:
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


# ── CC1: 跨 session + 长间隔 → stability 加成 ───────────────────────────────────────────────

def test_cc1_cross_session_long_gap_boosted(conn):
    """CC1: 跨 session 访问 + 间隔 >= cce_min_gap_hours(6) → CCE stability 加成。"""
    # chunk 来自 sess1，8小时前最后访问
    _insert_with_session_and_access(conn, "cc1", source_session="sess1", hours_ago=8)

    stab_before = _get_stability(conn, "cc1")
    # 现在从 sess2 访问
    result = apply_cross_session_consolidation(conn, ["cc1"], session_id="sess2")
    stab_after = _get_stability(conn, "cc1")

    assert stab_after > stab_before, (
        f"CC1: 跨 session 长间隔访问应获得 CCE 加成，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["cce_boosted"] > 0, f"CC1: cce_boosted 应 > 0，got {result}"


# ── CC2: 同 session → 无加成 ────────────────────────────────────────────────────────────────

def test_cc2_same_session_no_boost(conn):
    """CC2: 同 session 访问（source_session == session_id）→ 无 CCE 加成。"""
    _insert_with_session_and_access(conn, "cc2", source_session="sess1", hours_ago=8)

    stab_before = _get_stability(conn, "cc2")
    result = apply_cross_session_consolidation(conn, ["cc2"], session_id="sess1")  # 同 session
    stab_after = _get_stability(conn, "cc2")

    assert abs(stab_after - stab_before) < 0.001, (
        f"CC2: 同 session 访问不应有 CCE 加成，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["cce_boosted"] == 0, f"CC2: cce_boosted 应为 0"


# ── CC3: 跨 session 但间隔不足 → 无加成 ─────────────────────────────────────────────────────

def test_cc3_cross_session_short_gap_no_boost(conn):
    """CC3: 跨 session 但间隔 < cce_min_gap_hours(6) → 无 CCE 加成。"""
    _insert_with_session_and_access(conn, "cc3", source_session="sess1", hours_ago=2)  # 2h < 6h

    stab_before = _get_stability(conn, "cc3")
    result = apply_cross_session_consolidation(conn, ["cc3"], session_id="sess2")
    stab_after = _get_stability(conn, "cc3")

    assert abs(stab_after - stab_before) < 0.001, (
        f"CC3: 间隔不足时不应有 CCE 加成，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["cce_boosted"] == 0


# ── CC4: cce_enabled=False → 无加成 ─────────────────────────────────────────────────────────

def test_cc4_disabled_no_boost(conn):
    """CC4: cce_enabled=False → 无 CCE 加成。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.cce_enabled":
            return False
        return original_get(key, project=project)

    _insert_with_session_and_access(conn, "cc4", source_session="sess1", hours_ago=12)

    stab_before = _get_stability(conn, "cc4")
    with mock.patch.object(config, 'get', side_effect=patched_get):
        result = apply_cross_session_consolidation(conn, ["cc4"], session_id="sess2")
    stab_after = _get_stability(conn, "cc4")

    assert abs(stab_after - stab_before) < 0.001, (
        f"CC4: disabled 时不应有 CCE 加成，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["cce_boosted"] == 0


# ── CC5: importance 不足 → 不参与 CCE ────────────────────────────────────────────────────────

def test_cc5_low_importance_no_boost(conn):
    """CC5: importance < cce_min_importance(0.25) → 不参与 CCE。"""
    _insert_with_session_and_access(conn, "cc5", source_session="sess1",
                                     hours_ago=8, importance=0.10)

    stab_before = _get_stability(conn, "cc5")
    result = apply_cross_session_consolidation(conn, ["cc5"], session_id="sess2")
    stab_after = _get_stability(conn, "cc5")

    assert abs(stab_after - stab_before) < 0.001, (
        f"CC5: 低 importance 不应触发 CCE，before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── CC6: 间隔越长加成越大（最大在 24h）────────────────────────────────────────────────────────

def test_cc6_longer_gap_more_boost(conn):
    """CC6: 间隔越长，CCE 加成越大（6h < 24h）。"""
    _insert_with_session_and_access(conn, "cc6_6h", source_session="sess1", hours_ago=6)
    _insert_with_session_and_access(conn, "cc6_24h", source_session="sess1", hours_ago=24)

    stab_6h_before = _get_stability(conn, "cc6_6h")
    stab_24h_before = _get_stability(conn, "cc6_24h")

    apply_cross_session_consolidation(conn, ["cc6_6h"], session_id="sess2")
    apply_cross_session_consolidation(conn, ["cc6_24h"], session_id="sess2")

    stab_6h_after = _get_stability(conn, "cc6_6h")
    stab_24h_after = _get_stability(conn, "cc6_24h")

    gain_6h = stab_6h_after - stab_6h_before
    gain_24h = stab_24h_after - stab_24h_before

    assert gain_24h >= gain_6h - 0.001, (
        f"CC6: 24h 加成应 >= 6h 加成，gain_6h={gain_6h:.4f} gain_24h={gain_24h:.4f}"
    )


# ── CC7: 加成受 cce_max_boost 保护 ──────────────────────────────────────────────────────────

def test_cc7_max_boost_cap(conn):
    """CC7: CCE 加成受 cce_max_boost(0.10) 保护。"""
    cce_max_boost = config.get("store_vfs.cce_max_boost")  # 0.10
    base = 5.0

    _insert_with_session_and_access(conn, "cc7", source_session="sess1",
                                     hours_ago=1000, stability=base)

    stab_before = _get_stability(conn, "cc7")
    apply_cross_session_consolidation(conn, ["cc7"], session_id="sess2")
    stab_after = _get_stability(conn, "cc7")

    increment = stab_after - stab_before
    max_allowed = base * cce_max_boost + 0.01
    assert increment <= max_allowed, (
        f"CC7: CCE 增量 {increment:.4f} 不应超过 max_boost 允许的 {max_allowed:.4f}"
    )
    assert stab_after > stab_before, f"CC7: 应有 CCE 加成"


# ── CC8: update_accessed 集成 ─────────────────────────────────────────────────────────────────

def test_cc8_update_accessed_integration(conn):
    """CC8: update_accessed 跨 session 访问触发 CCE，stability 不降低。"""
    import datetime as dt
    long_ago = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=8)).isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count,
            encode_context, session_type_history, source_session)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("cc8", "test", "observation", "content cc8", "summary", 0.6, 5.0,
         long_ago, long_ago, 0.5, long_ago, 1, "test_ctx", "coding", "sess_old")
    )
    conn.commit()

    stab_before = _get_stability(conn, "cc8")
    # 从新 session 访问
    update_accessed(conn, ["cc8"], session_id="sess_new", project="test")
    stab_after = _get_stability(conn, "cc8")

    assert stab_after >= stab_before, (
        f"CC8: 跨 session update_accessed 后 stability 不应降低，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )
