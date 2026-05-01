"""test_iter496_spe.py — iter496: Survival Processing Effect 单元测试

SPE1: 含 survival 关键词 → stability + importance 提升
SPE2: 无 survival 关键词 → 不触发
SPE3: spe_enabled=False → 不触发
SPE4: importance < min → 不触发
SPE5: 多个 survival 关键词 → bonus 更大
SPE6: importance 不超过 1.0
SPE7: stability 上限 365
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
from store_vfs import ensure_schema, apply_survival_processing_effect
import config


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _insert(conn, cid, stability=10.0, importance=0.5, content="normal content", summary="sum"):
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


def _imp(conn, cid):
    r = conn.execute("SELECT importance FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(r[0]) if r else 0.0


def test_spe1_survival_keyword_boosts(conn):
    """SPE1: 含 critical/urgent → stability + importance 均提升。"""
    _insert(conn, "spe1", stability=10.0, importance=0.5,
            content="This is a critical security vulnerability that caused a crash")
    result = apply_survival_processing_effect(conn, ["spe1"])
    assert result["spe_boosted"] == 1
    assert _stab(conn, "spe1") > 10.0
    assert _imp(conn, "spe1") > 0.5


def test_spe2_no_keyword_no_boost(conn):
    """SPE2: 无 survival 关键词 → 不触发。"""
    _insert(conn, "spe2", stability=10.0, importance=0.5,
            content="We discussed the color scheme for the new dashboard")
    result = apply_survival_processing_effect(conn, ["spe2"])
    assert result["spe_boosted"] == 0
    assert _stab(conn, "spe2") == 10.0


def test_spe3_disabled(conn):
    """SPE3: spe_enabled=False → 不触发。"""
    _insert(conn, "spe3", stability=10.0, importance=0.5,
            content="critical crash fatal security breach")
    with mock.patch.object(config, 'get', side_effect=lambda k: False if k == "store_vfs.spe_enabled" else config.get(k)):
        result = apply_survival_processing_effect(conn, ["spe3"])
    assert result["spe_boosted"] == 0


def test_spe4_low_importance(conn):
    """SPE4: importance < min → 不触发。"""
    _insert(conn, "spe4", stability=10.0, importance=0.05,
            content="critical crash bug")
    result = apply_survival_processing_effect(conn, ["spe4"])
    assert result["spe_boosted"] == 0


def test_spe5_multiple_keywords_bigger_bonus(conn):
    """SPE5: 多个 survival 关键词 → bonus 更大。"""
    _insert(conn, "spe5a", stability=10.0, importance=0.5,
            content="There was a crash in production")
    _insert(conn, "spe5b", stability=10.0, importance=0.5,
            content="critical crash with fatal security incident and urgent rollback needed")
    apply_survival_processing_effect(conn, ["spe5a", "spe5b"])
    stab_a = _stab(conn, "spe5a")
    stab_b = _stab(conn, "spe5b")
    assert stab_b > stab_a, f"Multiple keywords ({stab_b}) should > single ({stab_a})"


def test_spe6_importance_cap_1(conn):
    """SPE6: importance 不超过 1.0。"""
    _insert(conn, "spe6", stability=10.0, importance=0.98,
            content="critical fatal crash emergency")
    apply_survival_processing_effect(conn, ["spe6"])
    assert _imp(conn, "spe6") <= 1.0


def test_spe7_stability_cap_365(conn):
    """SPE7: stability 不超过 365。"""
    _insert(conn, "spe7", stability=360.0, importance=0.5,
            content="critical fatal crash emergency")
    apply_survival_processing_effect(conn, ["spe7"])
    assert _stab(conn, "spe7") <= 365.0
