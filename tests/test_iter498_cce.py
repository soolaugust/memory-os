"""test_iter498_cce.py — iter498: Concreteness Effect 单元测试

CCE1: 含具体性指标（数字+路径等）→ stability 提升
CCE2: 纯抽象内容 → 不触发
CCE3: cce_enabled=False → 不触发
CCE4: importance < min → 不触发
CCE5: 指标数不足 min_indicators → 不触发
CCE6: 更多指标 → bonus 更大
CCE7: stability 上限 365
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
from store_vfs import ensure_schema, apply_concreteness_effect
import config


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _insert(conn, cid, stability=10.0, importance=0.5, content="abstract content"):
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


def test_cce1_concrete_indicators_boost(conn):
    """CCE1: 含具体数据（数字 + 路径）→ stability 提升。"""
    content = "Response latency improved from 200ms to 85ms after adding index on path:/api/users"
    _insert(conn, "cce1", stability=10.0, importance=0.6, content=content)
    result = apply_concreteness_effect(conn, ["cce1"])
    assert result["cce_boosted"] == 1
    assert _stab(conn, "cce1") > 10.0


def test_cce2_abstract_no_boost(conn):
    """CCE2: 纯抽象内容 → 不触发。"""
    content = "We should consider improving the overall system architecture in the future"
    _insert(conn, "cce2", stability=10.0, importance=0.6, content=content)
    result = apply_concreteness_effect(conn, ["cce2"])
    assert result["cce_boosted"] == 0
    assert _stab(conn, "cce2") == 10.0


def test_cce3_disabled(conn):
    """CCE3: cce_enabled=False → 不触发。"""
    content = "Latency 200ms on path:/api with 50% improvement"
    _insert(conn, "cce3", stability=10.0, importance=0.6, content=content)
    with mock.patch.object(config, 'get', side_effect=lambda k: False if k == "store_vfs.cce_enabled" else config.get(k)):
        result = apply_concreteness_effect(conn, ["cce3"])
    assert result["cce_boosted"] == 0


def test_cce4_low_importance(conn):
    """CCE4: importance < min → 不触发。"""
    content = "Response time 200ms on path:/api/v2 with 3GB memory usage"
    _insert(conn, "cce4", stability=10.0, importance=0.1, content=content)
    result = apply_concreteness_effect(conn, ["cce4"])
    assert result["cce_boosted"] == 0


def test_cce5_insufficient_indicators(conn):
    """CCE5: 指标数不足 min_indicators → 不触发。"""
    content = "The system uses about 50% of available resources on average"
    # 只有 1 个指标 (%)
    _insert(conn, "cce5", stability=10.0, importance=0.6, content=content)
    result = apply_concreteness_effect(conn, ["cce5"])
    assert result["cce_boosted"] == 0


def test_cce6_more_indicators_bigger_bonus(conn):
    """CCE6: 更多具体指标 → bonus 更大。"""
    content_2 = "Latency is 200ms with 2GB memory usage"
    content_5 = "Latency 200ms on http://api.example.com path:/api/users with 2GB and 95% hit rate for example"
    _insert(conn, "cce6a", stability=10.0, importance=0.6, content=content_2)
    _insert(conn, "cce6b", stability=10.0, importance=0.6, content=content_5)
    apply_concreteness_effect(conn, ["cce6a", "cce6b"])
    stab_a = _stab(conn, "cce6a")
    stab_b = _stab(conn, "cce6b")
    assert stab_b > stab_a, f"5 indicators ({stab_b}) should > 2 indicators ({stab_a})"


def test_cce7_stability_cap_365(conn):
    """CCE7: stability 不超过 365。"""
    content = "Latency 200ms on http://x.com path:/api with 5GB and 99% at line:42"
    _insert(conn, "cce7", stability=360.0, importance=0.6, content=content)
    apply_concreteness_effect(conn, ["cce7"])
    assert _stab(conn, "cce7") <= 365.0
