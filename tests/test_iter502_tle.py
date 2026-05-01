"""test_iter502_tle.py — iter502: Temporal Landmark Effect 单元测试

TLE1: 含 deploy/release 关键词 → stability 提升
TLE2: 无 landmark 关键词 → 不触发
TLE3: tle_enabled=False → 不触发
TLE4: importance < min → 不触发
TLE5: 多个 landmark 关键词 → bonus 更大（cap at 2）
TLE6: 中文 landmark 关键词也触发
TLE7: stability 上限 365
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
from store_vfs import ensure_schema, apply_temporal_landmark_effect
import config


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _insert(conn, cid, stability=10.0, importance=0.5, content="normal", summary="sum"):
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


def test_tle1_landmark_keyword_boosts(conn):
    """TLE1: 含 deploy + release → stability 提升。"""
    _insert(conn, "tle1", stability=10.0, importance=0.5,
            content="Completed deploy of v2.1 release to production")
    result = apply_temporal_landmark_effect(conn, ["tle1"])
    assert result["tle_boosted"] == 1
    assert _stab(conn, "tle1") > 10.0


def test_tle2_no_landmark_no_boost(conn):
    """TLE2: 无 landmark 关键词 → 不触发。"""
    _insert(conn, "tle2", stability=10.0, importance=0.5,
            content="Discussed refactoring plans for next quarter")
    result = apply_temporal_landmark_effect(conn, ["tle2"])
    assert result["tle_boosted"] == 0
    assert _stab(conn, "tle2") == 10.0


def test_tle3_disabled(conn):
    """TLE3: tle_enabled=False → 不触发。"""
    _insert(conn, "tle3", stability=10.0, importance=0.5,
            content="deploy release launch milestone")
    with mock.patch.object(config, 'get', side_effect=lambda k: False if k == "store_vfs.tle_enabled" else config.get(k)):
        result = apply_temporal_landmark_effect(conn, ["tle3"])
    assert result["tle_boosted"] == 0


def test_tle4_low_importance(conn):
    """TLE4: importance < min → 不触发。"""
    _insert(conn, "tle4", stability=10.0, importance=0.1,
            content="deploy release")
    result = apply_temporal_landmark_effect(conn, ["tle4"])
    assert result["tle_boosted"] == 0


def test_tle5_multiple_landmarks_capped(conn):
    """TLE5: 多个 landmark 关键词 bonus 上限 = 2×stability_bonus。"""
    _insert(conn, "tle5a", stability=10.0, importance=0.5,
            content="deploy to production complete")
    _insert(conn, "tle5b", stability=10.0, importance=0.5,
            content="deploy release launch milestone hotfix go-live")
    apply_temporal_landmark_effect(conn, ["tle5a", "tle5b"])
    stab_a = _stab(conn, "tle5a")
    stab_b = _stab(conn, "tle5b")
    # tle5b has more keywords but capped at min(hit, 2), so max = 2*bonus
    assert stab_b >= stab_a, f"More landmarks should >= single: {stab_b} vs {stab_a}"
    # Both should be boosted
    assert stab_a > 10.0
    assert stab_b > 10.0


def test_tle6_chinese_keywords_trigger(conn):
    """TLE6: 中文 landmark 关键词（上线/发布/部署）也触发。"""
    _insert(conn, "tle6", stability=10.0, importance=0.5,
            content="项目已完成上线部署，进入稳定运行阶段")
    result = apply_temporal_landmark_effect(conn, ["tle6"])
    assert result["tle_boosted"] == 1
    assert _stab(conn, "tle6") > 10.0


def test_tle7_stability_cap_365(conn):
    """TLE7: stability 不超过 365。"""
    _insert(conn, "tle7", stability=360.0, importance=0.5,
            content="deploy release milestone launch")
    apply_temporal_landmark_effect(conn, ["tle7"])
    assert _stab(conn, "tle7") <= 365.0
