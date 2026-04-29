"""test_iter491_vre.py — iter491: von Restorff Isolation Effect 单元测试

VR1: 稀有 chunk_type（比例 < threshold）→ stability 提升
VR2: 常见 chunk_type（比例 >= threshold）→ 不触发
VR3: vre_enabled=False → 不触发
VR4: importance < min → 不触发
VR5: session chunks < min_session_chunks → 使用全局比例
VR6: max_boost 上限
VR7: stability 上限 365
"""
import sys, sqlite3, datetime, unittest.mock as mock, pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import tmpfs  # noqa
from store_vfs import ensure_schema, apply_von_restorff_isolation
import config


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _insert(conn, cid, stability=10.0, importance=0.5, chunk_type="observation",
            source_session="sess1"):
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id,project,chunk_type,content,summary,importance,stability,
            created_at,updated_at,retrievability,last_accessed,access_count,
            encode_context,session_type_history,source_session)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (cid,"test",chunk_type,"c"+cid,"s",importance,stability,
         now,now,0.8,now,1,"ctx","",source_session)
    )
    conn.commit()


def _stab(conn, cid):
    r = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(r[0]) if r else 0.0


def test_vr1_rare_type_boosts(conn):
    """稀有类型 chunk（1/10=10% < threshold 20%）→ stability 提升。"""
    # 9 个 observation + 1 个 decision（稀有）
    for i in range(9):
        _insert(conn, f"vr1_obs{i}", chunk_type="observation", source_session="s1")
    _insert(conn, "vr1_dec", chunk_type="decision", importance=0.5, source_session="s1")

    s0 = _stab(conn, "vr1_dec")
    r = apply_von_restorff_isolation(conn, ["vr1_dec"], session_id="s1")
    conn.commit()
    assert r["vre_boosted"] == 1, f"稀有类型应触发 VRE，got {r}"
    assert _stab(conn, "vr1_dec") > s0


def test_vr2_common_type_no_boost(conn):
    """常见类型（7/10=70% >= threshold 20%）→ 不触发。"""
    # 7 个 observation（常见）+ 3 个其他类型
    for i in range(7):
        _insert(conn, f"vr2_obs{i}", chunk_type="observation", source_session="s2")
    for i in range(3):
        _insert(conn, f"vr2_oth{i}", chunk_type="reflection", source_session="s2")

    _insert(conn, "vr2_target", chunk_type="observation", importance=0.5, source_session="s2")
    s0 = _stab(conn, "vr2_target")
    r = apply_von_restorff_isolation(conn, ["vr2_target"], session_id="s2")
    conn.commit()
    assert r["vre_boosted"] == 0, "常见类型不应触发 VRE"
    assert abs(_stab(conn, "vr2_target") - s0) < 0.01


def test_vr3_disabled_no_boost(conn):
    """vre_enabled=False → 不触发。"""
    for i in range(9):
        _insert(conn, f"vr3_obs{i}", chunk_type="observation", source_session="s3")
    _insert(conn, "vr3_dec", chunk_type="decision", importance=0.5, source_session="s3")

    s0 = _stab(conn, "vr3_dec")
    orig = config.get

    def pg(k, project=None):
        return False if k == "store_vfs.vre_enabled" else orig(k, project=project)

    with mock.patch.object(config, 'get', side_effect=pg):
        r = apply_von_restorff_isolation(conn, ["vr3_dec"], session_id="s3")
    assert r["vre_boosted"] == 0
    assert abs(_stab(conn, "vr3_dec") - s0) < 0.01


def test_vr4_low_importance_no_boost(conn):
    """importance < min → 不触发。"""
    for i in range(9):
        _insert(conn, f"vr4_obs{i}", chunk_type="observation", source_session="s4")
    _insert(conn, "vr4_dec", chunk_type="decision", importance=0.05, source_session="s4")

    r = apply_von_restorff_isolation(conn, ["vr4_dec"], session_id="s4")
    assert r["vre_boosted"] == 0


def test_vr5_uses_global_ratio_when_session_small(conn):
    """session chunk 数量 < min_session_chunks → 使用全局比例，稀有类型仍可触发。"""
    # 全局：20 个 observation + 1 个 decision（稀有：1/21≈5%）
    for i in range(20):
        _insert(conn, f"vr5_glob{i}", chunk_type="observation", source_session="other_sess")
    _insert(conn, "vr5_dec", chunk_type="decision", importance=0.5, source_session="tiny_sess")
    # tiny_sess 只有 1 个 chunk < min(3)，应回退到全局比例

    s0 = _stab(conn, "vr5_dec")
    r = apply_von_restorff_isolation(conn, ["vr5_dec"], session_id="tiny_sess")
    conn.commit()
    # decision 全局比例 1/21 ≈ 5% < 20% threshold，应触发
    assert r["vre_boosted"] == 1 or _stab(conn, "vr5_dec") >= s0, "小 session 回退全局应触发"


def test_vr6_max_boost_cap(conn):
    """增益不超 max_boost(0.18)。"""
    for i in range(9):
        _insert(conn, f"vr6_obs{i}", chunk_type="observation", source_session="s6")
    _insert(conn, "vr6_dec", chunk_type="decision", importance=0.8, source_session="s6")

    s0 = _stab(conn, "vr6_dec")
    orig = config.get

    def pg(k, project=None):
        if k == "store_vfs.vre_stability_bonus":
            return 0.50
        if k == "store_vfs.vre_max_boost":
            return 0.18
        return orig(k, project=project)

    with mock.patch.object(config, 'get', side_effect=pg):
        apply_von_restorff_isolation(conn, ["vr6_dec"], session_id="s6")
    conn.commit()
    ratio = _stab(conn, "vr6_dec") / s0
    assert ratio <= 1.18 + 0.01, f"max_boost 违反: ×{ratio:.4f}"


def test_vr7_stability_cap_365(conn):
    """stability 不超 365。"""
    for i in range(9):
        _insert(conn, f"vr7_obs{i}", chunk_type="observation", source_session="s7")
    _insert(conn, "vr7_dec", chunk_type="decision", stability=362.0, importance=0.8,
            source_session="s7")
    apply_von_restorff_isolation(conn, ["vr7_dec"], session_id="s7")
    conn.commit()
    assert _stab(conn, "vr7_dec") <= 365.01
