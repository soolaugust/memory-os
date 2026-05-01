"""test_iter501_nbe.py — iter501: Negative Bias Effect 单元测试

NBE1: 含 bug/error 关键词 → stability 提升
NBE2: 正面内容无负面词 → 不触发
NBE3: nbe_enabled=False → 不触发
NBE4: importance < min → 不触发
NBE5: 多个负面关键词 → bonus 更大
NBE6: summary 中的关键词也触发
NBE7: stability 上限 365
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
from store_vfs import ensure_schema, apply_negative_bias_effect
import config


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _insert(conn, cid, stability=10.0, importance=0.5, content="neutral", summary="sum"):
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id,project,chunk_type,content,summary,importance,stability,
            created_at,updated_at,retrievability,last_accessed,access_count,
            encode_context,session_type_history)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (cid, "test", "decision", content, summary, importance, stability,
         now, now, 0.8, now, 1, "ctx", "")
    )
    conn.commit()


def _stab(conn, cid):
    r = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(r[0]) if r else 0.0


def test_nbe1_negative_keyword_boosts(conn):
    """NBE1: 含 bug + error → stability 提升。"""
    _insert(conn, "nbe1", stability=10.0, importance=0.5,
            content="Found a critical bug causing memory leak in the parser")
    result = apply_negative_bias_effect(conn, ["nbe1"])
    assert result["nbe_boosted"] == 1
    assert _stab(conn, "nbe1") > 10.0


def test_nbe2_positive_no_boost(conn):
    """NBE2: 正面内容 → 不触发。"""
    _insert(conn, "nbe2", stability=10.0, importance=0.5,
            content="Great improvement in latency, team celebrated the success")
    result = apply_negative_bias_effect(conn, ["nbe2"])
    assert result["nbe_boosted"] == 0
    assert _stab(conn, "nbe2") == 10.0


def test_nbe3_disabled(conn):
    """NBE3: nbe_enabled=False → 不触发。"""
    _insert(conn, "nbe3", stability=10.0, importance=0.5,
            content="critical bug error crash")
    with mock.patch.object(config, 'get', side_effect=lambda k: False if k == "store_vfs.nbe_enabled" else config.get(k)):
        result = apply_negative_bias_effect(conn, ["nbe3"])
    assert result["nbe_boosted"] == 0


def test_nbe4_low_importance(conn):
    """NBE4: importance < min → 不触发。"""
    _insert(conn, "nbe4", stability=10.0, importance=0.1,
            content="some bug found")
    result = apply_negative_bias_effect(conn, ["nbe4"])
    assert result["nbe_boosted"] == 0


def test_nbe5_multiple_keywords_bigger_bonus(conn):
    """NBE5: 多个负面关键词 → bonus 更大。"""
    _insert(conn, "nbe5a", stability=10.0, importance=0.5,
            content="There was a bug in the system")
    _insert(conn, "nbe5b", stability=10.0, importance=0.5,
            content="Critical bug caused crash and memory leak with timeout failure")
    apply_negative_bias_effect(conn, ["nbe5a", "nbe5b"])
    stab_a = _stab(conn, "nbe5a")
    stab_b = _stab(conn, "nbe5b")
    assert stab_b > stab_a, f"Multiple negatives ({stab_b}) should > single ({stab_a})"


def test_nbe6_summary_keywords_trigger(conn):
    """NBE6: summary 中的负面关键词也触发。"""
    _insert(conn, "nbe6", stability=10.0, importance=0.5,
            content="investigation into the recent incident",
            summary="regression found in auth module")
    result = apply_negative_bias_effect(conn, ["nbe6"])
    assert result["nbe_boosted"] == 1
    assert _stab(conn, "nbe6") > 10.0


def test_nbe7_stability_cap_365(conn):
    """NBE7: stability 不超过 365。"""
    _insert(conn, "nbe7", stability=360.0, importance=0.5,
            content="bug error crash failure")
    apply_negative_bias_effect(conn, ["nbe7"])
    assert _stab(conn, "nbe7") <= 365.0
