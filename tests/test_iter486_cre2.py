"""test_iter486_cre2.py — iter486: Contextual Reinstatement Effect 单元测试

CR1: project(namespace) 匹配 → retrievability +ns_bonus
CR2: project 不匹配 → 不提升
CR3: tag Jaccard >= threshold → +tag_bonus
CR4: tag Jaccard < threshold → 不提升
CR5: cre2_enabled=False → 不触发
CR6: importance < min_importance → 不触发
CR7: ns+tag 组合不超 max_boost
CR8: retrievability 不超 1.0
"""
import sys, sqlite3, datetime, json, unittest.mock as mock, pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import tmpfs  # noqa
from store_vfs import ensure_schema, apply_contextual_reinstatement
import config


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _insert(conn, cid, importance=0.6, retrievability=0.5, project="default", tags=None):
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    if tags is None:
        tags = []
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id,project,chunk_type,content,summary,importance,stability,
            created_at,updated_at,retrievability,last_accessed,access_count,
            encode_context,session_type_history,tags)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (cid, project, "observation","c"+cid,"s",importance,5.0,
         now,now,retrievability,now,1,"ctx","",json.dumps(tags))
    )
    conn.commit()


def _ret(conn, cid):
    r = conn.execute("SELECT retrievability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(r[0]) if r else 0.0


def test_cr1_namespace_match_boosts(conn):
    """project(namespace) 匹配 → retrievability 提升。"""
    _insert(conn, "cr1", importance=0.6, retrievability=0.5, project="project_A")
    r0 = _ret(conn, "cr1")
    result = apply_contextual_reinstatement(conn, ["cr1"], query_namespace="project_A")
    conn.commit()
    r1 = _ret(conn, "cr1")
    assert result["cre2_boosted"] == 1
    assert r1 > r0
    assert abs(r1 - r0 - config.get("store_vfs.cre2_namespace_match_bonus")) < 0.01


def test_cr2_namespace_mismatch_no_boost(conn):
    """project 不匹配 → 不提升。"""
    _insert(conn, "cr2", importance=0.6, project="project_B")
    r0 = _ret(conn, "cr2")
    result = apply_contextual_reinstatement(conn, ["cr2"], query_namespace="project_A")
    conn.commit()
    assert result["cre2_boosted"] == 0
    assert abs(_ret(conn, "cr2") - r0) < 0.01


def test_cr3_tag_overlap_above_threshold_boosts(conn):
    """Jaccard=0.5 >= threshold(0.5) → +tag_bonus。"""
    # tags ["a","b","c"], query ["a","b","d"] → J=2/4=0.5
    _insert(conn, "cr3", importance=0.6, retrievability=0.5, tags=["a","b","c"])
    orig = config.get

    def pg(k, project=None):
        if k == "store_vfs.cre2_namespace_match_bonus":
            return 0.0
        return orig(k, project=project)

    with mock.patch.object(config, 'get', side_effect=pg):
        result = apply_contextual_reinstatement(conn, ["cr3"], query_namespace="", query_tags=["a","b","d"])
    conn.commit()
    assert result["cre2_boosted"] == 1
    assert _ret(conn, "cr3") > 0.5


def test_cr4_tag_overlap_below_threshold_no_boost(conn):
    """Jaccard=1/5=0.2 < threshold(0.5) → 不触发。"""
    _insert(conn, "cr4", importance=0.6, tags=["a","b","c","d","e"])
    orig = config.get

    def pg(k, project=None):
        if k == "store_vfs.cre2_namespace_match_bonus":
            return 0.0
        return orig(k, project=project)

    with mock.patch.object(config, 'get', side_effect=pg):
        result = apply_contextual_reinstatement(conn, ["cr4"], query_namespace="", query_tags=["a"])
    conn.commit()
    assert result["cre2_boosted"] == 0


def test_cr5_disabled_no_boost(conn):
    """cre2_enabled=False → 不触发。"""
    _insert(conn, "cr5", importance=0.6, project="project_A")
    r0 = _ret(conn, "cr5")
    orig = config.get

    def pg(k, project=None):
        return False if k == "store_vfs.cre2_enabled" else orig(k, project=project)

    with mock.patch.object(config, 'get', side_effect=pg):
        result = apply_contextual_reinstatement(conn, ["cr5"], query_namespace="project_A")
    conn.commit()
    assert result["cre2_boosted"] == 0
    assert abs(_ret(conn, "cr5") - r0) < 0.01


def test_cr6_low_importance_no_boost(conn):
    """importance=0.10 < min_importance(0.20) → 不触发。"""
    _insert(conn, "cr6", importance=0.10, project="project_A")
    result = apply_contextual_reinstatement(conn, ["cr6"], query_namespace="project_A")
    conn.commit()
    assert result["cre2_boosted"] == 0


def test_cr7_max_boost_cap(conn):
    """ns+tag 组合不超 max_boost(0.12)。"""
    _insert(conn, "cr7", importance=0.8, retrievability=0.5, project="projA", tags=["x","y"])
    orig = config.get

    def pg(k, project=None):
        if k == "store_vfs.cre2_namespace_match_bonus":
            return 0.10
        if k == "store_vfs.cre2_tag_overlap_bonus":
            return 0.10
        if k == "store_vfs.cre2_tag_overlap_threshold":
            return 0.30
        if k == "store_vfs.cre2_max_boost":
            return 0.12
        return orig(k, project=project)

    with mock.patch.object(config, 'get', side_effect=pg):
        apply_contextual_reinstatement(conn, ["cr7"], query_namespace="projA", query_tags=["x","y","z"])
    conn.commit()
    boost = _ret(conn, "cr7") - 0.5
    assert boost <= 0.12 + 0.01, f"max_boost 违反: boost={boost:.4f}"


def test_cr8_retrievability_capped_at_1(conn):
    """retrievability 不超过 1.0。"""
    _insert(conn, "cr8", importance=0.9, retrievability=0.95, project="proj")
    apply_contextual_reinstatement(conn, ["cr8"], query_namespace="proj")
    conn.commit()
    assert _ret(conn, "cr8") <= 1.0 + 0.001
