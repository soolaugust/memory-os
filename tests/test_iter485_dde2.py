"""test_iter485_dde2.py — iter485: Desirable Difficulty Effect 单元测试

覆盖：
  DD1: 低 R + 低 stability → 触发 DDE2，stability 提升
  DD2: 高 R（刚访问）→ 不触发
  DD3: stability > threshold → 不触发
  DD4: dde2_enabled=False → 不触发
  DD5: importance < min_importance → 不触发
  DD6: max_boost 上限保护
  DD7: R 越低→ 增益越大
  DD8: stability 上限 365
"""
import sys, sqlite3, datetime, math, unittest.mock as mock, pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import tmpfs  # noqa
from store_vfs import ensure_schema, apply_desirable_difficulty
import config


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _insert(conn, cid, stability=2.0, importance=0.5, last_accessed_ago_secs=3*86400):
    now = datetime.datetime.now(datetime.timezone.utc)
    last = (now - datetime.timedelta(seconds=last_accessed_ago_secs)).isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id,project,chunk_type,content,summary,importance,stability,
            created_at,updated_at,retrievability,last_accessed,access_count,
            encode_context,session_type_history)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (cid,"test","observation","c"+cid,"s",importance,stability,
         last,last,math.exp(-last_accessed_ago_secs/86400/max(stability,0.1)),
         last,1,"ctx","")
    )
    conn.commit()


def _stab(conn, cid):
    r = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(r[0]) if r else 0.0


def test_dd1_low_retrievability_low_stability_boosts(conn):
    """低 R + 低 stability → 触发 DDE2。"""
    _insert(conn, "dd1", stability=2.0, importance=0.5, last_accessed_ago_secs=3*86400)
    s0 = _stab(conn, "dd1")
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    r = apply_desirable_difficulty(conn, ["dd1"], now_iso=now)
    conn.commit()
    assert r["dde2_boosted"] == 1, f"应触发 DDE2，got {r}"
    assert _stab(conn, "dd1") > s0


def test_dd2_high_retrievability_no_boost(conn):
    """高 R（60s 前访问）→ 不触发。"""
    _insert(conn, "dd2", stability=5.0, importance=0.5, last_accessed_ago_secs=60)
    s0 = _stab(conn, "dd2")
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    r = apply_desirable_difficulty(conn, ["dd2"], now_iso=now)
    conn.commit()
    assert r["dde2_boosted"] == 0
    assert abs(_stab(conn, "dd2") - s0) < 0.01


def test_dd3_high_stability_no_boost(conn):
    """stability=50 > 默认 threshold(10) → 不触发。"""
    _insert(conn, "dd3", stability=50.0, importance=0.5, last_accessed_ago_secs=200*86400)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    r = apply_desirable_difficulty(conn, ["dd3"], now_iso=now)
    assert r["dde2_boosted"] == 0


def test_dd4_disabled_no_boost(conn):
    """dde2_enabled=False → 不触发。"""
    _insert(conn, "dd4", stability=1.0, importance=0.5, last_accessed_ago_secs=5*86400)
    s0 = _stab(conn, "dd4")
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    orig = config.get

    def pg(k, project=None):
        return False if k == "store_vfs.dde2_enabled" else orig(k, project=project)

    with mock.patch.object(config, 'get', side_effect=pg):
        r = apply_desirable_difficulty(conn, ["dd4"], now_iso=now)
    assert r["dde2_boosted"] == 0
    assert abs(_stab(conn, "dd4") - s0) < 0.01


def test_dd5_low_importance_no_boost(conn):
    """importance=0.05 < min_importance(0.20) → 不触发。"""
    _insert(conn, "dd5", stability=1.0, importance=0.05, last_accessed_ago_secs=5*86400)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    r = apply_desirable_difficulty(conn, ["dd5"], now_iso=now)
    assert r["dde2_boosted"] == 0


def test_dd6_max_boost_cap(conn):
    """增益不超 max_boost (默认 0.20)。"""
    _insert(conn, "dd6", stability=2.0, importance=0.8, last_accessed_ago_secs=90*86400)
    s0 = _stab(conn, "dd6")
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    apply_desirable_difficulty(conn, ["dd6"], now_iso=now)
    conn.commit()
    ratio = _stab(conn, "dd6") / s0
    assert ratio <= 1.20 + 0.02, f"max_boost 违反: ×{ratio:.4f}"


def test_dd7_higher_difficulty_more_boost(conn):
    """R 越低→ 增益越大（30天 > 1天）。"""
    _insert(conn, "dd7_e", stability=2.0, importance=0.8, last_accessed_ago_secs=86400)
    _insert(conn, "dd7_h", stability=2.0, importance=0.8, last_accessed_ago_secs=30*86400)
    s0e, s0h = _stab(conn, "dd7_e"), _stab(conn, "dd7_h")
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    apply_desirable_difficulty(conn, ["dd7_e", "dd7_h"], now_iso=now)
    conn.commit()
    re = _stab(conn, "dd7_e") / s0e
    rh = _stab(conn, "dd7_h") / s0h
    assert rh >= re, f"高难度应获更大增益: easy={re:.4f}, hard={rh:.4f}"


def test_dd8_stability_cap_365(conn):
    """stability 不超 365。"""
    _insert(conn, "dd8", stability=360.0, importance=0.9, last_accessed_ago_secs=5*86400)
    orig = config.get

    def pg(k, project=None):
        return 1000.0 if k == "store_vfs.dde2_stability_threshold" else orig(k, project=project)

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with mock.patch.object(config, 'get', side_effect=pg):
        apply_desirable_difficulty(conn, ["dd8"], now_iso=now)
    conn.commit()
    assert _stab(conn, "dd8") <= 365.01
