"""test_iter499_pse.py — iter499: Picture Superiority Effect 单元测试

PSE1: 含结构化标记（table/list）→ stability 提升
PSE2: 纯 prose → 不触发
PSE3: pse_enabled=False → 不触发
PSE4: importance < min → 不触发
PSE5: 指标数不足 → 不触发
PSE6: 更多结构化指标 → bonus 更大
PSE7: stability 上限 365
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
from store_vfs import ensure_schema, apply_picture_superiority_effect
import config


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _insert(conn, cid, stability=10.0, importance=0.5, content="plain text"):
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id,project,chunk_type,content,summary,importance,stability,
            created_at,updated_at,retrievability,last_accessed,access_count,
            encode_context,session_type_history)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (cid, "test", "decision", content, "s", importance, stability,
         now, now, 0.8, now, 1, "ctx", "")
    )
    conn.commit()


def _stab(conn, cid):
    r = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(r[0]) if r else 0.0


def test_pse1_structured_content_boosts(conn):
    """PSE1: 含 table marker (| ) + list (1. ) → stability 提升。"""
    content = "Task list:\n1. Fix bug\n2. Deploy\n| col1 | col2 |\n| a | b |"
    _insert(conn, "pse1", stability=10.0, importance=0.6, content=content)
    result = apply_picture_superiority_effect(conn, ["pse1"])
    assert result["pse_boosted"] == 1
    assert _stab(conn, "pse1") > 10.0


def test_pse2_prose_no_boost(conn):
    """PSE2: 纯 prose 无结构化标记 → 不触发。"""
    content = "We had a long discussion about the project direction and decided to move forward"
    _insert(conn, "pse2", stability=10.0, importance=0.6, content=content)
    result = apply_picture_superiority_effect(conn, ["pse2"])
    assert result["pse_boosted"] == 0
    assert _stab(conn, "pse2") == 10.0


def test_pse3_disabled(conn):
    """PSE3: pse_enabled=False → 不触发。"""
    content = "1. item\n2. item\n| col | val |"
    _insert(conn, "pse3", stability=10.0, importance=0.6, content=content)
    with mock.patch.object(config, 'get', side_effect=lambda k: False if k == "store_vfs.pse_enabled" else config.get(k)):
        result = apply_picture_superiority_effect(conn, ["pse3"])
    assert result["pse_boosted"] == 0


def test_pse4_low_importance(conn):
    """PSE4: importance < min → 不触发。"""
    content = "1. item\n2. item\n| col | val |"
    _insert(conn, "pse4", stability=10.0, importance=0.1, content=content)
    result = apply_picture_superiority_effect(conn, ["pse4"])
    assert result["pse_boosted"] == 0


def test_pse5_insufficient_indicators(conn):
    """PSE5: 只有 1 个指标 → 不触发。"""
    content = "The flow goes like this: A -> B and then we continue"
    # Only "->" matches, "- " is not in indicators
    _insert(conn, "pse5", stability=10.0, importance=0.6, content=content)
    result = apply_picture_superiority_effect(conn, ["pse5"])
    assert result["pse_boosted"] == 0


def test_pse6_more_indicators_bigger_bonus(conn):
    """PSE6: 更多结构化指标 → bonus 更大。"""
    content_2 = "Steps:\n1. Do X\n2. Do Y"
    content_4 = "```\n1. Do X\n2. Do Y\n| col | val |\n| --- | --- |"
    _insert(conn, "pse6a", stability=10.0, importance=0.6, content=content_2)
    _insert(conn, "pse6b", stability=10.0, importance=0.6, content=content_4)
    apply_picture_superiority_effect(conn, ["pse6a", "pse6b"])
    stab_a = _stab(conn, "pse6a")
    stab_b = _stab(conn, "pse6b")
    assert stab_b > stab_a, f"4 indicators ({stab_b}) should > 2 indicators ({stab_a})"


def test_pse7_stability_cap_365(conn):
    """PSE7: stability 不超过 365。"""
    content = "```\n1. item\n2. item\n| a | b |\n| --- | --- |"
    _insert(conn, "pse7", stability=360.0, importance=0.6, content=content)
    apply_picture_superiority_effect(conn, ["pse7"])
    assert _stab(conn, "pse7") <= 365.0
