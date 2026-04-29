"""test_iter492_pef.py — iter492: Production Effect 单元测试

PE1: decision 类型 → stability 提升
PE2: observation 类型（非生产型）→ 不触发
PE3: pef_enabled=False → 不触发
PE4: importance < min → 不触发
PE5: 自定义 production_types
PE6: max_boost 上限
PE7: stability 上限 365
"""
import sys, sqlite3, datetime, unittest.mock as mock, pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import tmpfs  # noqa
from store_vfs import ensure_schema, apply_production_effect
import config


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _insert(conn, cid, stability=10.0, importance=0.5, chunk_type="observation"):
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id,project,chunk_type,content,summary,importance,stability,
            created_at,updated_at,retrievability,last_accessed,access_count,
            encode_context,session_type_history)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (cid,"test",chunk_type,"c"+cid,"s",importance,stability,
         now,now,0.8,now,1,"ctx","")
    )
    conn.commit()


def _stab(conn, cid):
    r = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(r[0]) if r else 0.0


def test_pe1_decision_type_boosts(conn):
    """chunk_type=decision（默认生产型）→ stability 提升。"""
    _insert(conn, "pe1", stability=10.0, importance=0.5, chunk_type="decision")
    s0 = _stab(conn, "pe1")
    r = apply_production_effect(conn, ["pe1"])
    conn.commit()
    assert r["pef_boosted"] == 1
    bonus = config.get("store_vfs.pef_stability_bonus")
    ratio = _stab(conn, "pe1") / s0
    assert abs(ratio - (1.0 + bonus)) < 0.01, f"预期 ×{1+bonus:.2f}，实际 ×{ratio:.4f}"


def test_pe2_observation_no_boost(conn):
    """chunk_type=observation（非生产型）→ 不触发。"""
    _insert(conn, "pe2", stability=10.0, importance=0.5, chunk_type="observation")
    s0 = _stab(conn, "pe2")
    r = apply_production_effect(conn, ["pe2"])
    conn.commit()
    assert r["pef_boosted"] == 0
    assert abs(_stab(conn, "pe2") - s0) < 0.01


def test_pe3_disabled_no_boost(conn):
    """pef_enabled=False → 不触发。"""
    _insert(conn, "pe3", stability=10.0, importance=0.5, chunk_type="decision")
    s0 = _stab(conn, "pe3")
    orig = config.get

    def pg(k, project=None):
        return False if k == "store_vfs.pef_enabled" else orig(k, project=project)

    with mock.patch.object(config, 'get', side_effect=pg):
        r = apply_production_effect(conn, ["pe3"])
    assert r["pef_boosted"] == 0
    assert abs(_stab(conn, "pe3") - s0) < 0.01


def test_pe4_low_importance_no_boost(conn):
    """importance=0.05 < min(0.15) → 不触发。"""
    _insert(conn, "pe4", stability=10.0, importance=0.05, chunk_type="decision")
    r = apply_production_effect(conn, ["pe4"])
    assert r["pef_boosted"] == 0


def test_pe5_custom_production_types(conn):
    """自定义 production_types 包含 'custom_type'。"""
    _insert(conn, "pe5", stability=10.0, importance=0.5, chunk_type="custom_type")
    s0 = _stab(conn, "pe5")
    orig = config.get

    def pg(k, project=None):
        if k == "store_vfs.pef_production_types":
            return ["custom_type", "decision"]
        return orig(k, project=project)

    with mock.patch.object(config, 'get', side_effect=pg):
        r = apply_production_effect(conn, ["pe5"])
    conn.commit()
    assert r["pef_boosted"] == 1
    assert _stab(conn, "pe5") > s0


def test_pe6_max_boost_cap(conn):
    """增益不超 max_boost(0.15)。"""
    _insert(conn, "pe6", stability=10.0, importance=0.8, chunk_type="reflection")
    s0 = _stab(conn, "pe6")
    orig = config.get

    def pg(k, project=None):
        if k == "store_vfs.pef_stability_bonus":
            return 0.50
        if k == "store_vfs.pef_max_boost":
            return 0.15
        return orig(k, project=project)

    with mock.patch.object(config, 'get', side_effect=pg):
        apply_production_effect(conn, ["pe6"])
    conn.commit()
    ratio = _stab(conn, "pe6") / s0
    assert ratio <= 1.15 + 0.01, f"max_boost 违反: ×{ratio:.4f}"


def test_pe7_stability_cap_365(conn):
    """stability 不超 365。"""
    _insert(conn, "pe7", stability=362.0, importance=0.8, chunk_type="insight")
    apply_production_effect(conn, ["pe7"])
    conn.commit()
    assert _stab(conn, "pe7") <= 365.01
