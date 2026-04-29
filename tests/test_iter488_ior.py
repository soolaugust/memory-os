"""test_iter488_ior.py — iter488: Inhibition of Return 单元测试

IOR1: 窗口内短间隔（30s）→ stability 下降（penalty_factor）
IOR2: 超出窗口（600s > 300s）→ 不惩罚
IOR3: ior_enabled=False → 不惩罚
IOR4: importance < min_importance → 不惩罚
IOR5: 窗口内但接近边界 → 惩罚轻于短间隔
IOR6: stability floor=1.0
IOR7: 空列表 → 无操作
"""
import sys, sqlite3, datetime, unittest.mock as mock, pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import tmpfs  # noqa
from store_vfs import ensure_schema, apply_inhibition_of_return
import config


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _insert(conn, cid, stability=10.0, importance=0.5, last_accessed_ago_secs=60):
    now = datetime.datetime.now(datetime.timezone.utc)
    last = (now - datetime.timedelta(seconds=last_accessed_ago_secs)).isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id,project,chunk_type,content,summary,importance,stability,
            created_at,updated_at,retrievability,last_accessed,access_count,
            encode_context,session_type_history)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (cid,"test","observation","c"+cid,"s",importance,stability,
         last,last,0.8,last,1,"ctx","")
    )
    conn.commit()


def _stab(conn, cid):
    r = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(r[0]) if r else 0.0


def test_ior1_within_window_penalized(conn):
    """30s 前访问（窗口 300s，min_interval 60s）→ 最重惩罚（×0.50）。"""
    _insert(conn, "ior1", stability=10.0, importance=0.5, last_accessed_ago_secs=30)
    s0 = _stab(conn, "ior1")
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    r = apply_inhibition_of_return(conn, ["ior1"], now_iso=now)
    conn.commit()
    assert r["ior_penalized"] == 1, "短时重访应触发 IOR"
    assert _stab(conn, "ior1") < s0
    # penalty_factor=0.50 → stability * 0.50 (约)
    pf = config.get("store_vfs.ior_penalty_factor")
    assert abs(_stab(conn, "ior1") - s0 * pf) < 1.0


def test_ior2_outside_window_no_penalty(conn):
    """600s 前（窗口 300s）→ 不惩罚。"""
    _insert(conn, "ior2", stability=10.0, importance=0.5, last_accessed_ago_secs=600)
    s0 = _stab(conn, "ior2")
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    r = apply_inhibition_of_return(conn, ["ior2"], now_iso=now)
    conn.commit()
    assert r["ior_penalized"] == 0
    assert abs(_stab(conn, "ior2") - s0) < 0.01


def test_ior3_disabled_no_penalty(conn):
    """ior_enabled=False → 不惩罚。"""
    _insert(conn, "ior3", stability=10.0, importance=0.5, last_accessed_ago_secs=30)
    s0 = _stab(conn, "ior3")
    orig = config.get

    def pg(k, project=None):
        return False if k == "store_vfs.ior_enabled" else orig(k, project=project)

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with mock.patch.object(config, 'get', side_effect=pg):
        r = apply_inhibition_of_return(conn, ["ior3"], now_iso=now)
    assert r["ior_penalized"] == 0
    assert abs(_stab(conn, "ior3") - s0) < 0.01


def test_ior4_low_importance_no_penalty(conn):
    """importance=0.05 < min_importance(0.15) → 不惩罚。"""
    _insert(conn, "ior4", stability=10.0, importance=0.05, last_accessed_ago_secs=30)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    r = apply_inhibition_of_return(conn, ["ior4"], now_iso=now)
    assert r["ior_penalized"] == 0


def test_ior5_longer_gap_less_penalty(conn):
    """窗口内 30s vs 250s — 短间隔惩罚更重。"""
    _insert(conn, "ior5s", stability=10.0, importance=0.5, last_accessed_ago_secs=30)
    _insert(conn, "ior5l", stability=10.0, importance=0.5, last_accessed_ago_secs=250)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    apply_inhibition_of_return(conn, ["ior5s", "ior5l"], now_iso=now)
    conn.commit()
    ss = _stab(conn, "ior5s")
    sl = _stab(conn, "ior5l")
    assert ss <= sl, f"短间隔惩罚应重: 30s_stab={ss:.3f}, 250s_stab={sl:.3f}"


def test_ior6_stability_floor_1(conn):
    """极重惩罚后 stability 不低于 1.0。"""
    _insert(conn, "ior6", stability=2.0, importance=0.5, last_accessed_ago_secs=10)
    orig = config.get

    def pg(k, project=None):
        if k == "store_vfs.ior_penalty_factor":
            return 0.01
        if k == "store_vfs.ior_min_interval_secs":
            return 60
        return orig(k, project=project)

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with mock.patch.object(config, 'get', side_effect=pg):
        apply_inhibition_of_return(conn, ["ior6"], now_iso=now)
    conn.commit()
    assert _stab(conn, "ior6") >= 1.0


def test_ior7_empty_chunk_ids(conn):
    """空列表 → 无操作。"""
    r = apply_inhibition_of_return(conn, [])
    assert r["ior_penalized"] == 0
