"""test_iter500_ae.py — iter500: Anchoring Effect 单元测试

AE1: 项目最早 chunk（anchor）→ stability 提升
AE2: 项目中后期 chunk（非 anchor）→ 不触发
AE3: ae_enabled=False → 不触发
AE4: importance < min → 不触发
AE5: 项目 chunk 太少（< min_project_chunks）→ 不触发
AE6: 只有 early_percentile 内的 chunk 获得 boost
AE7: stability 上限 365
"""
import sys
import sqlite3
import datetime
import unittest.mock as mock
import pytest
from pathlib import Path
from datetime import timedelta

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import tmpfs  # noqa
from store_vfs import ensure_schema, apply_anchoring_effect
import config


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _insert(conn, cid, project="test", stability=10.0, importance=0.5, days_ago=0):
    now = datetime.datetime.now(datetime.timezone.utc)
    created = (now - timedelta(days=days_ago)).isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id,project,chunk_type,content,summary,importance,stability,
            created_at,updated_at,retrievability,last_accessed,access_count,
            encode_context,session_type_history)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (cid, project, "decision", f"content {cid}", "s", importance, stability,
         created, created, 0.8, created, 1, "ctx", "")
    )
    conn.commit()


def _stab(conn, cid):
    r = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(r[0]) if r else 0.0


def _populate(conn, project="test", count=20):
    """插入 20 个 chunk，days_ago 从 count-1 到 0。"""
    for i in range(count):
        _insert(conn, f"pop_{i}", project=project, days_ago=count - i, importance=0.5)


def test_ae1_anchor_chunk_boosts(conn):
    """AE1: 项目最早的 chunk（前 10%）→ stability 提升。"""
    _populate(conn, count=20)
    # pop_0 是最早的（days_ago=20），属于前 10% (2个)
    _insert(conn, "ae1", project="test", stability=10.0, importance=0.6, days_ago=100)
    result = apply_anchoring_effect(conn, ["ae1"], project="test")
    assert result["ae_boosted"] == 1
    assert _stab(conn, "ae1") > 10.0


def test_ae2_non_anchor_no_boost(conn):
    """AE2: 中后期 chunk（不在前 10%）→ 不触发。"""
    _populate(conn, count=20)
    # 插入一个较新的 chunk（days_ago=1，不在前 10%）
    _insert(conn, "ae2", project="test", stability=10.0, importance=0.6, days_ago=1)
    result = apply_anchoring_effect(conn, ["ae2"], project="test")
    assert result["ae_boosted"] == 0
    assert _stab(conn, "ae2") == 10.0


def test_ae3_disabled(conn):
    """AE3: ae_enabled=False → 不触发。"""
    _populate(conn, count=20)
    _insert(conn, "ae3", project="test", stability=10.0, importance=0.6, days_ago=100)
    with mock.patch.object(config, 'get', side_effect=lambda k: False if k == "store_vfs.ae_enabled" else config.get(k)):
        result = apply_anchoring_effect(conn, ["ae3"], project="test")
    assert result["ae_boosted"] == 0


def test_ae4_low_importance(conn):
    """AE4: importance < min → 不触发。"""
    _populate(conn, count=20)
    _insert(conn, "ae4", project="test", stability=10.0, importance=0.1, days_ago=100)
    result = apply_anchoring_effect(conn, ["ae4"], project="test")
    assert result["ae_boosted"] == 0


def test_ae5_too_few_chunks(conn):
    """AE5: 项目 chunk < min_project_chunks → 不触发。"""
    # 只有 3 个 chunk（< 10）
    for i in range(3):
        _insert(conn, f"few_{i}", project="test", days_ago=10 - i)
    result = apply_anchoring_effect(conn, [f"few_0"], project="test")
    assert result["ae_boosted"] == 0


def test_ae6_only_early_percentile_boosted(conn):
    """AE6: 只有 early_percentile 内的 chunk 获得 boost。"""
    _populate(conn, count=20)
    # pop_0(最早), pop_10(中间)
    result = apply_anchoring_effect(conn, ["pop_0", "pop_10"], project="test")
    # pop_0 是最早（前 10% = 2个），但 importance=0.5 < min=0.25... wait check
    # Default min_importance = 0.25, pop has 0.5, so should work
    stab_early = _stab(conn, "pop_0")
    stab_mid = _stab(conn, "pop_10")
    # pop_0 should be boosted (earliest 10%), pop_10 should not
    assert stab_early > 10.0, f"early chunk should be boosted: {stab_early}"
    assert stab_mid == 10.0, f"mid chunk should not be boosted: {stab_mid}"


def test_ae7_stability_cap_365(conn):
    """AE7: stability 不超过 365。"""
    _populate(conn, count=20)
    _insert(conn, "ae7", project="test", stability=360.0, importance=0.6, days_ago=200)
    apply_anchoring_effect(conn, ["ae7"], project="test")
    assert _stab(conn, "ae7") <= 365.0
