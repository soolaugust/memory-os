"""test_iter497_bze.py — iter497: Bizarreness Effect 单元测试

BZE1: 稀有 chunk_type（频率 < threshold）→ stability 提升
BZE2: 常见 chunk_type（频率 >= threshold）→ 不触发
BZE3: bze_enabled=False → 不触发
BZE4: importance < min → 不触发
BZE5: 项目 chunk 太少（< 5）→ 不触发
BZE6: 越稀有 bonus 越大
BZE7: stability 上限 365
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
from store_vfs import ensure_schema, apply_bizarreness_effect
import config


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _insert(conn, cid, project="test", chunk_type="decision", stability=10.0, importance=0.5):
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id,project,chunk_type,content,summary,importance,stability,
            created_at,updated_at,retrievability,last_accessed,access_count,
            encode_context,session_type_history)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (cid, project, chunk_type, f"content {cid}", "s", importance, stability,
         now, now, 0.8, now, 1, "ctx", "")
    )
    conn.commit()


def _stab(conn, cid):
    r = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(r[0]) if r else 0.0


def _populate_common(conn, project="test", count=20):
    """插入 20 个 decision 类型的 chunk（占主导）。"""
    for i in range(count):
        _insert(conn, f"common_{i}", project=project, chunk_type="decision")


def test_bze1_rare_type_boosts(conn):
    """BZE1: 稀有 chunk_type → stability 提升。"""
    _populate_common(conn, count=20)  # 20 个 decision
    # 插入 1 个 rare_insight 类型（频率 = 1/21 ≈ 4.7% < 10%）
    _insert(conn, "bze1", chunk_type="rare_insight", stability=10.0, importance=0.6)
    result = apply_bizarreness_effect(conn, ["bze1"], project="test")
    assert result["bze_boosted"] == 1
    assert _stab(conn, "bze1") > 10.0


def test_bze2_common_type_no_boost(conn):
    """BZE2: 常见 chunk_type（decision 占多数）→ 不触发。"""
    _populate_common(conn, count=20)
    _insert(conn, "bze2", chunk_type="decision", stability=10.0, importance=0.6)
    result = apply_bizarreness_effect(conn, ["bze2"], project="test")
    assert result["bze_boosted"] == 0
    assert _stab(conn, "bze2") == 10.0


def test_bze3_disabled(conn):
    """BZE3: bze_enabled=False → 不触发。"""
    _populate_common(conn, count=20)
    _insert(conn, "bze3", chunk_type="rare_insight", stability=10.0, importance=0.6)
    with mock.patch.object(config, 'get', side_effect=lambda k: False if k == "store_vfs.bze_enabled" else config.get(k)):
        result = apply_bizarreness_effect(conn, ["bze3"], project="test")
    assert result["bze_boosted"] == 0


def test_bze4_low_importance(conn):
    """BZE4: importance < min → 不触发。"""
    _populate_common(conn, count=20)
    _insert(conn, "bze4", chunk_type="rare_insight", stability=10.0, importance=0.1)
    result = apply_bizarreness_effect(conn, ["bze4"], project="test")
    assert result["bze_boosted"] == 0


def test_bze5_too_few_chunks(conn):
    """BZE5: 项目 chunk 太少（< 5）→ 不触发。"""
    _insert(conn, "bze5a", chunk_type="decision", stability=10.0, importance=0.6)
    _insert(conn, "bze5b", chunk_type="rare_insight", stability=10.0, importance=0.6)
    result = apply_bizarreness_effect(conn, ["bze5b"], project="test")
    assert result["bze_boosted"] == 0


def test_bze6_rarer_gives_bigger_bonus(conn):
    """BZE6: 越稀有（频率越低）→ bonus 越大。"""
    # 30 个 decision + 2 个 uncommon (6.25%) + 1 个 very_rare (3.0%)
    for i in range(30):
        _insert(conn, f"bg_{i}", chunk_type="decision")
    _insert(conn, "unc_a", chunk_type="uncommon")
    _insert(conn, "unc_b", chunk_type="uncommon")
    _insert(conn, "bze6_rare", chunk_type="very_rare", stability=10.0, importance=0.6)
    _insert(conn, "bze6_uncommon", chunk_type="uncommon", stability=10.0, importance=0.6)

    apply_bizarreness_effect(conn, ["bze6_rare", "bze6_uncommon"], project="test")
    stab_rare = _stab(conn, "bze6_rare")
    stab_uncommon = _stab(conn, "bze6_uncommon")
    # very_rare (1/34 ≈ 2.9%) should get bigger bonus than uncommon (3/34 ≈ 8.8%)
    assert stab_rare > stab_uncommon, f"rarer ({stab_rare}) should > uncommon ({stab_uncommon})"


def test_bze7_stability_cap_365(conn):
    """BZE7: stability 不超过 365。"""
    _populate_common(conn, count=20)
    _insert(conn, "bze7", chunk_type="rare_insight", stability=360.0, importance=0.6)
    apply_bizarreness_effect(conn, ["bze7"], project="test")
    assert _stab(conn, "bze7") <= 365.0
