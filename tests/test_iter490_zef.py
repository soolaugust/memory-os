"""test_iter490_zef.py — iter490: Zeigarnik Effect 单元测试

ZF1: content 含 TODO → stability 提升
ZF2: content 无未完成关键词 → 不触发
ZF3: zef_enabled=False → 不触发
ZF4: importance < min → 不触发
ZF5: summary 含关键词也触发
ZF6: max_boost 上限
ZF7: stability 上限 365
"""
import sys, sqlite3, datetime, unittest.mock as mock, pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import tmpfs  # noqa
from store_vfs import ensure_schema, apply_zeigarnik_effect_zef as apply_zeigarnik_effect
import config


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _insert(conn, cid, stability=10.0, importance=0.5, content="plain", summary="summary"):
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id,project,chunk_type,content,summary,importance,stability,
            created_at,updated_at,retrievability,last_accessed,access_count,
            encode_context,session_type_history)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (cid,"test","observation",content,summary,importance,stability,
         now,now,0.8,now,1,"ctx","")
    )
    conn.commit()


def _stab(conn, cid):
    r = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(r[0]) if r else 0.0


def test_zf1_todo_in_content_boosts(conn):
    """content 含 TODO → 触发 ZEF。"""
    _insert(conn, "zf1", stability=10.0, importance=0.5,
            content="TODO: implement this feature later")
    s0 = _stab(conn, "zf1")
    r = apply_zeigarnik_effect(conn, ["zf1"])
    conn.commit()
    assert r["zef_boosted"] == 1
    assert _stab(conn, "zf1") > s0


def test_zf2_no_keyword_no_boost(conn):
    """无未完成关键词 → 不触发。"""
    _insert(conn, "zf2", stability=10.0, importance=0.5,
            content="This task is completed and done.")
    s0 = _stab(conn, "zf2")
    r = apply_zeigarnik_effect(conn, ["zf2"])
    conn.commit()
    assert r["zef_boosted"] == 0
    assert abs(_stab(conn, "zf2") - s0) < 0.01


def test_zf3_disabled_no_boost(conn):
    """zef_enabled=False → 不触发。"""
    _insert(conn, "zf3", stability=10.0, importance=0.5,
            content="TODO: this needs to be done")
    s0 = _stab(conn, "zf3")
    orig = config.get

    def pg(k, project=None):
        return False if k == "store_vfs.zef_enabled" else orig(k, project=project)

    with mock.patch.object(config, 'get', side_effect=pg):
        r = apply_zeigarnik_effect(conn, ["zf3"])
    assert r["zef_boosted"] == 0
    assert abs(_stab(conn, "zf3") - s0) < 0.01


def test_zf4_low_importance_no_boost(conn):
    """importance < min_importance → 不触发。"""
    _insert(conn, "zf4", stability=10.0, importance=0.05,
            content="TODO: this is important")
    r = apply_zeigarnik_effect(conn, ["zf4"])
    assert r["zef_boosted"] == 0


def test_zf5_keyword_in_summary_also_triggers(conn):
    """summary 含 FIXME → 也触发 ZEF。"""
    _insert(conn, "zf5", stability=10.0, importance=0.5,
            content="some content", summary="FIXME: need to fix this")
    s0 = _stab(conn, "zf5")
    r = apply_zeigarnik_effect(conn, ["zf5"])
    conn.commit()
    assert r["zef_boosted"] == 1
    assert _stab(conn, "zf5") > s0


def test_zf6_max_boost_cap(conn):
    """增益不超 max_boost(0.20)。"""
    _insert(conn, "zf6", stability=10.0, importance=0.8,
            content="TODO FIXME PENDING WIP BLOCKED")
    s0 = _stab(conn, "zf6")
    orig = config.get

    def pg(k, project=None):
        if k == "store_vfs.zef_stability_bonus":
            return 0.50  # 超出上限
        if k == "store_vfs.zef_max_boost":
            return 0.20
        return orig(k, project=project)

    with mock.patch.object(config, 'get', side_effect=pg):
        apply_zeigarnik_effect(conn, ["zf6"])
    conn.commit()
    ratio = _stab(conn, "zf6") / s0
    assert ratio <= 1.20 + 0.01, f"max_boost 违反: ×{ratio:.4f}"


def test_zf7_stability_cap_365(conn):
    """stability 不超 365。"""
    _insert(conn, "zf7", stability=360.0, importance=0.8,
            content="TODO: important pending work")
    apply_zeigarnik_effect(conn, ["zf7"])
    conn.commit()
    assert _stab(conn, "zf7") <= 365.01
