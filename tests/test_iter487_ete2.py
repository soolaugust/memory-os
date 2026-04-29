"""test_iter487_ete2.py — iter487: Emotion Tagging Effect 单元测试

ET1: importance >= threshold → stability × (1 + decay_reduction)
ET2: importance < threshold → 不触发
ET3: 含情绪关键词 → +keyword_bonus
ET4: ete2_enabled=False → 不触发
ET5: max_decay_reduction 上限
ET6: stability 上限 365
ET7: 多 chunk，只高 importance 被触发
"""
import sys, sqlite3, datetime, unittest.mock as mock, pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import tmpfs  # noqa
from store_vfs import ensure_schema, apply_emotion_tagging_decay_reduction
import config


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _insert(conn, cid, stability=10.0, importance=0.8, content="plain text"):
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id,project,chunk_type,content,summary,importance,stability,
            created_at,updated_at,retrievability,last_accessed,access_count,
            encode_context,session_type_history)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (cid,"test","observation",content,"s",importance,stability,
         now,now,0.8,now,1,"ctx","")
    )
    conn.commit()


def _stab(conn, cid):
    r = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(r[0]) if r else 0.0


def test_et1_high_importance_boosts_stability(conn):
    """importance=0.80 >= threshold(0.70) → stability × 1.15。"""
    _insert(conn, "et1", stability=10.0, importance=0.80, content="plain text")
    s0 = _stab(conn, "et1")
    r = apply_emotion_tagging_decay_reduction(conn, ["et1"])
    conn.commit()
    assert r["ete2_boosted"] == 1
    ratio = _stab(conn, "et1") / s0
    decay_red = config.get("store_vfs.ete2_stability_decay_reduction")
    assert abs(ratio - (1.0 + decay_red)) < 0.01, f"预期 ×{1+decay_red:.2f}，实际 ×{ratio:.4f}"


def test_et2_low_importance_no_boost(conn):
    """importance=0.50 < threshold(0.70) → 不触发。"""
    _insert(conn, "et2", stability=10.0, importance=0.50)
    s0 = _stab(conn, "et2")
    r = apply_emotion_tagging_decay_reduction(conn, ["et2"])
    conn.commit()
    assert r["ete2_boosted"] == 0
    assert abs(_stab(conn, "et2") - s0) < 0.01


def test_et3_emotion_keyword_extra_bonus(conn):
    """含情绪关键词 → +keyword_bonus。"""
    _insert(conn, "et3", stability=10.0, importance=0.85,
            content="This is a critical error that needs urgent attention")
    s0 = _stab(conn, "et3")
    r = apply_emotion_tagging_decay_reduction(conn, ["et3"])
    conn.commit()
    ratio = _stab(conn, "et3") / s0
    dr = config.get("store_vfs.ete2_stability_decay_reduction")
    kb = config.get("store_vfs.ete2_keyword_bonus")
    expected = 1.0 + dr + kb
    assert abs(ratio - expected) < 0.01, f"含关键词预期 ×{expected:.3f}，实际 ×{ratio:.4f}"


def test_et4_disabled_no_boost(conn):
    """ete2_enabled=False → 不触发。"""
    _insert(conn, "et4", stability=10.0, importance=0.90)
    s0 = _stab(conn, "et4")
    orig = config.get

    def pg(k, project=None):
        return False if k == "store_vfs.ete2_enabled" else orig(k, project=project)

    with mock.patch.object(config, 'get', side_effect=pg):
        r = apply_emotion_tagging_decay_reduction(conn, ["et4"])
    assert r["ete2_boosted"] == 0
    assert abs(_stab(conn, "et4") - s0) < 0.01


def test_et5_max_decay_reduction_cap(conn):
    """总减免不超 max_decay_reduction(0.30)。"""
    _insert(conn, "et5", stability=10.0, importance=0.9, content="urgent critical task")
    s0 = _stab(conn, "et5")
    orig = config.get

    def pg(k, project=None):
        if k == "store_vfs.ete2_stability_decay_reduction":
            return 0.40  # 超出上限
        if k == "store_vfs.ete2_keyword_bonus":
            return 0.20
        if k == "store_vfs.ete2_max_decay_reduction":
            return 0.30
        return orig(k, project=project)

    with mock.patch.object(config, 'get', side_effect=pg):
        apply_emotion_tagging_decay_reduction(conn, ["et5"])
    conn.commit()
    ratio = _stab(conn, "et5") / s0
    assert ratio <= 1.30 + 0.01, f"max_decay_reduction=0.30 上限，实际 ×{ratio:.4f}"


def test_et6_stability_cap_365(conn):
    """stability 不超 365。"""
    _insert(conn, "et6", stability=355.0, importance=0.90)
    apply_emotion_tagging_decay_reduction(conn, ["et6"])
    conn.commit()
    assert _stab(conn, "et6") <= 365.01


def test_et7_multiple_chunks_selective(conn):
    """多 chunk，只 importance >= threshold 的被触发。"""
    _insert(conn, "et7h", stability=10.0, importance=0.85)
    _insert(conn, "et7l", stability=10.0, importance=0.50)
    r = apply_emotion_tagging_decay_reduction(conn, ["et7h", "et7l"])
    conn.commit()
    assert r["ete2_boosted"] == 1
    assert _stab(conn, "et7h") > 10.0
    assert abs(_stab(conn, "et7l") - 10.0) < 0.01
