"""test_iter495_dce.py — iter495: Dual-Coding Effect 单元测试

DCE1: 含代码+文字的 chunk → stability 提升
DCE2: 纯文字 chunk（无结构化指标）→ 不触发
DCE3: dce_enabled=False → 不触发
DCE4: importance < min → 不触发
DCE5: content 太短 → 不触发
DCE6: 多指标 → bonus 更大（上限 max_boost）
DCE7: stability 上限 365
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
from store_vfs import ensure_schema, apply_dual_coding_effect
import config


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _insert(conn, cid, stability=10.0, importance=0.5, content="short"):
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


def test_dce1_code_plus_text_boosts(conn):
    """DCE1: chunk 含 def + import + .py → 双编码触发，stability 提升。"""
    content = "We decided to use def process_data in utils.py with import json for the parser"
    _insert(conn, "dce1", stability=10.0, importance=0.6, content=content)
    with mock.patch.object(config, 'get', wraps=config.get):
        result = apply_dual_coding_effect(conn, ["dce1"])
    assert result["dce_boosted"] == 1
    assert _stab(conn, "dce1") > 10.0


def test_dce2_pure_text_no_boost(conn):
    """DCE2: 纯文字 chunk 无结构化指标 → 不触发。"""
    content = "We should consider improving the performance of the retrieval system in the future"
    _insert(conn, "dce2", stability=10.0, importance=0.6, content=content)
    result = apply_dual_coding_effect(conn, ["dce2"])
    assert result["dce_boosted"] == 0
    assert _stab(conn, "dce2") == 10.0


def test_dce3_disabled(conn):
    """DCE3: dce_enabled=False → 不触发。"""
    content = "def foo(): import bar from baz.py"
    _insert(conn, "dce3", stability=10.0, importance=0.6, content=content)
    with mock.patch.object(config, 'get', side_effect=lambda k: False if k == "store_vfs.dce_enabled" else config.get(k)):
        result = apply_dual_coding_effect(conn, ["dce3"])
    assert result["dce_boosted"] == 0


def test_dce4_low_importance(conn):
    """DCE4: importance < min → 不触发。"""
    content = "def foo(): import bar from baz.py with https://example.com"
    _insert(conn, "dce4", stability=10.0, importance=0.1, content=content)
    result = apply_dual_coding_effect(conn, ["dce4"])
    assert result["dce_boosted"] == 0


def test_dce5_short_content(conn):
    """DCE5: content 长度 < min_content_len → 不触发。"""
    content = "def foo"
    _insert(conn, "dce5", stability=10.0, importance=0.6, content=content)
    result = apply_dual_coding_effect(conn, ["dce5"])
    assert result["dce_boosted"] == 0


def test_dce6_more_indicators_bigger_bonus(conn):
    """DCE6: 更多指标 → bonus 更大。"""
    content_2 = "Use def process and import json to handle the data pipeline correctly"
    content_4 = "Use def process and import json from utils.py with SELECT * FROM table and https://api.example.com"
    _insert(conn, "dce6a", stability=10.0, importance=0.6, content=content_2)
    _insert(conn, "dce6b", stability=10.0, importance=0.6, content=content_4)
    apply_dual_coding_effect(conn, ["dce6a", "dce6b"])
    stab_a = _stab(conn, "dce6a")
    stab_b = _stab(conn, "dce6b")
    assert stab_b > stab_a, f"4 indicators ({stab_b}) should > 2 indicators ({stab_a})"


def test_dce7_stability_cap_365(conn):
    """DCE7: stability 不超过 365。"""
    content = "def foo import bar class Baz https://x.com with .py extension"
    _insert(conn, "dce7", stability=360.0, importance=0.6, content=content)
    apply_dual_coding_effect(conn, ["dce7"])
    assert _stab(conn, "dce7") <= 365.0
