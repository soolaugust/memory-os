"""test_iter489_eve.py — iter489: Encoding Variability Effect 单元测试

EV1: unique_session_types >= min → stability 提升
EV2: unique_session_types < min → 不触发
EV3: eve_enabled=False → 不触发
EV4: importance < min → 不触发
EV5: 更多不同 session_type → 更大增益
EV6: max_boost 上限
EV7: stability 上限 365
EV8: 空 session_type_history → 不触发
"""
import sys, sqlite3, datetime, unittest.mock as mock, pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import tmpfs  # noqa
from store_vfs import ensure_schema, apply_encoding_variability
import config


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _insert(conn, cid, stability=5.0, importance=0.5, session_type_history=""):
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id,project,chunk_type,content,summary,importance,stability,
            created_at,updated_at,retrievability,last_accessed,access_count,
            encode_context,session_type_history)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (cid,"test","observation","c"+cid,"s",importance,stability,
         now,now,0.8,now,1,"ctx",session_type_history)
    )
    conn.commit()


def _stab(conn, cid):
    r = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(r[0]) if r else 0.0


def test_ev1_multiple_types_boosts(conn):
    """3 种不同 session_type >= min(2) → 触发 EVE。"""
    _insert(conn, "ev1", stability=5.0, importance=0.5,
            session_type_history="chat,search,write")
    s0 = _stab(conn, "ev1")
    r = apply_encoding_variability(conn, ["ev1"])
    conn.commit()
    assert r["eve_boosted"] == 1
    assert _stab(conn, "ev1") > s0


def test_ev2_single_type_no_boost(conn):
    """只有 1 种 session_type < min(2) → 不触发。"""
    _insert(conn, "ev2", stability=5.0, importance=0.5,
            session_type_history="chat,chat,chat")
    s0 = _stab(conn, "ev2")
    r = apply_encoding_variability(conn, ["ev2"])
    conn.commit()
    assert r["eve_boosted"] == 0
    assert abs(_stab(conn, "ev2") - s0) < 0.01


def test_ev3_disabled_no_boost(conn):
    """eve_enabled=False → 不触发。"""
    _insert(conn, "ev3", stability=5.0, importance=0.5,
            session_type_history="chat,search,write")
    s0 = _stab(conn, "ev3")
    orig = config.get

    def pg(k, project=None):
        return False if k == "store_vfs.eve_enabled" else orig(k, project=project)

    with mock.patch.object(config, 'get', side_effect=pg):
        r = apply_encoding_variability(conn, ["ev3"])
    assert r["eve_boosted"] == 0
    assert abs(_stab(conn, "ev3") - s0) < 0.01


def test_ev4_low_importance_no_boost(conn):
    """importance=0.05 < min(0.20) → 不触发。"""
    _insert(conn, "ev4", stability=5.0, importance=0.05,
            session_type_history="chat,search,write")
    r = apply_encoding_variability(conn, ["ev4"])
    assert r["eve_boosted"] == 0


def test_ev5_more_types_more_boost(conn):
    """更多不同 session_type → 更大增益（2种 vs 4种）。"""
    _insert(conn, "ev5a", stability=5.0, importance=0.6,
            session_type_history="chat,search")
    _insert(conn, "ev5b", stability=5.0, importance=0.6,
            session_type_history="chat,search,write,review")
    s0a, s0b = _stab(conn, "ev5a"), _stab(conn, "ev5b")
    apply_encoding_variability(conn, ["ev5a", "ev5b"])
    conn.commit()
    ra = _stab(conn, "ev5a") / s0a
    rb = _stab(conn, "ev5b") / s0b
    assert rb >= ra, f"更多类型应获更大增益: 2types={ra:.4f}, 4types={rb:.4f}"


def test_ev6_max_boost_cap(conn):
    """增益不超 max_boost(0.15)。"""
    # 10 种类型，bonus_per_type=0.04 → 理论增益 = 0.04*10 = 0.40 > 0.15
    _insert(conn, "ev6", stability=5.0, importance=0.8,
            session_type_history="a,b,c,d,e,f,g,h,i,j")
    s0 = _stab(conn, "ev6")
    apply_encoding_variability(conn, ["ev6"])
    conn.commit()
    ratio = _stab(conn, "ev6") / s0
    assert ratio <= 1.15 + 0.02, f"max_boost=0.15 上限, 实际 ×{ratio:.4f}"


def test_ev7_stability_cap_365(conn):
    """stability 不超 365。"""
    _insert(conn, "ev7", stability=362.0, importance=0.8,
            session_type_history="chat,search,write")
    apply_encoding_variability(conn, ["ev7"])
    conn.commit()
    assert _stab(conn, "ev7") <= 365.01


def test_ev8_empty_history_no_boost(conn):
    """空 session_type_history → 不触发。"""
    _insert(conn, "ev8", stability=5.0, importance=0.6, session_type_history="")
    s0 = _stab(conn, "ev8")
    r = apply_encoding_variability(conn, ["ev8"])
    assert r["eve_boosted"] == 0
    assert abs(_stab(conn, "ev8") - s0) < 0.01
