"""
test_iter480_fap.py — iter480: Forward Association Primacy 单元测试

覆盖：
  FA1: 访问较晚的 session chunk → 较早的 sibling retrievability 提升
  FA2: session 中最早的 chunk → 无前向联想来源，不触发 FAP
  FA3: fap_enabled=False → 无前向联想提升
  FA4: importance < fap_min_importance(0.25) → 不触发 FAP
  FA5: 提升受 fap_max_boost(0.12) 保护
  FA6: session 大小 < fap_min_session_size(3) → 不触发 FAP
  FA7: retrievability 提升后不超过 1.0
  FA8: update_accessed 集成 — 访问触发前向联想提升

认知科学依据：
  Kahana (2002) 前向联想比后向联想强 ~1.5:1 —
    访问后来的 chunk 时，较早的 session sibling 的可达性提升（正向联想方向）。
  Howard & Kahana (1999): 前向联想在情节记忆中持续稳定（d ≈ 0.4）。

OS 类比：CPU 指令流水线预取 — 执行当前指令时预取相关指令到 fetch buffer。
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

from store_vfs import ensure_schema, apply_forward_association_primacy, update_accessed
import config


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _utcnow():
    return datetime.datetime.now(datetime.timezone.utc)


def _insert_raw(conn, cid, project="test", importance=0.6, retrievability=0.5,
                source_session="sess1", created_offset_seconds=0):
    import datetime as dt
    ts = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=created_offset_seconds)
    now_iso = ts.isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count,
            encode_context, session_type_history, source_session)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (cid, project, "observation", "content " + cid, "summary", importance, 5.0,
         now_iso, now_iso, retrievability, now_iso, 0, "test_ctx", "coding", source_session)
    )
    conn.commit()


def _get_retrievability(conn, cid: str) -> float:
    row = conn.execute("SELECT retrievability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


# ── FA1: 访问较晚 chunk → 较早 sibling retrievability 提升 ─────────────────────────────────

def test_fa1_later_access_boosts_earlier_chunks(conn):
    """FA1: 访问 session 中较晚的 chunk → 较早的 sibling retrievability 提升。"""
    for i in range(5):
        _insert_raw(conn, f"fa1_{i}", source_session="sess_fa1",
                    created_offset_seconds=i * 10)

    # 访问最新的 chunk（fa1_4），应提升之前的 chunk（fa1_0 ~ fa1_3）
    retr_before = [_get_retrievability(conn, f"fa1_{i}") for i in range(4)]
    result = apply_forward_association_primacy(conn, ["fa1_4"])
    retr_after = [_get_retrievability(conn, f"fa1_{i}") for i in range(4)]

    any_boosted = any(retr_after[i] > retr_before[i] for i in range(4))
    assert any_boosted, (
        f"FA1: 访问较晚 chunk 应提升较早 sibling retrievability，"
        f"before={retr_before} after={retr_after}"
    )
    assert result["fap_boosted"] > 0, f"FA1: fap_boosted 应 > 0，got {result}"


# ── FA2: 最早 chunk 无前向联想来源 ──────────────────────────────────────────────────────────

def test_fa2_earliest_chunk_no_fap(conn):
    """FA2: 访问 session 中最早的 chunk → 没有更早的 sibling，不触发 FAP。"""
    for i in range(5):
        _insert_raw(conn, f"fa2_{i}", source_session="sess_fa2",
                    created_offset_seconds=i * 10)

    # 访问最早的 chunk（fa2_0），没有更早的 sibling
    retr_others_before = [_get_retrievability(conn, f"fa2_{i}") for i in range(1, 5)]
    result = apply_forward_association_primacy(conn, ["fa2_0"])
    retr_others_after = [_get_retrievability(conn, f"fa2_{i}") for i in range(1, 5)]

    assert result["fap_boosted"] == 0, (
        f"FA2: 访问最早 chunk 不应触发 FAP，got {result}"
    )
    # 其他 chunk 不变
    for i in range(4):
        assert abs(retr_others_after[i] - retr_others_before[i]) < 0.001, (
            f"FA2: fa2_{i+1} retrievability 不应改变"
        )


# ── FA3: fap_enabled=False → 无提升 ─────────────────────────────────────────────────────────

def test_fa3_disabled_no_boost(conn):
    """FA3: fap_enabled=False → 无前向联想提升。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.fap_enabled":
            return False
        return original_get(key, project=project)

    for i in range(5):
        _insert_raw(conn, f"fa3_{i}", source_session="sess_fa3",
                    created_offset_seconds=i * 10)

    retr_before = _get_retrievability(conn, "fa3_0")
    with mock.patch.object(config, 'get', side_effect=patched_get):
        result = apply_forward_association_primacy(conn, ["fa3_4"])
    retr_after = _get_retrievability(conn, "fa3_0")

    assert abs(retr_after - retr_before) < 0.001, (
        f"FA3: disabled 时不应有前向联想提升，before={retr_before:.4f} after={retr_after:.4f}"
    )
    assert result["fap_boosted"] == 0, f"FA3: fap_boosted 应为 0"


# ── FA4: importance 不足 → 不触发 FAP ────────────────────────────────────────────────────────

def test_fa4_low_importance_no_fap(conn):
    """FA4: importance < fap_min_importance(0.25) → 不触发 FAP。"""
    for i in range(5):
        _insert_raw(conn, f"fa4_{i}", source_session="sess_fa4",
                    created_offset_seconds=i * 10, importance=0.10)

    retr_before = _get_retrievability(conn, "fa4_0")
    result = apply_forward_association_primacy(conn, ["fa4_4"])
    retr_after = _get_retrievability(conn, "fa4_0")

    assert abs(retr_after - retr_before) < 0.001, (
        f"FA4: 低 importance 不应触发 FAP，before={retr_before:.4f} after={retr_after:.4f}"
    )


# ── FA5: 提升受 fap_max_boost 保护 ──────────────────────────────────────────────────────────

def test_fa5_max_boost_cap(conn):
    """FA5: FAP retrievability 提升受 fap_max_boost(0.12) 保护。"""
    fap_max_boost = config.get("store_vfs.fap_max_boost")  # 0.12
    fap_lookback = config.get("store_vfs.fap_lookback_window")  # 10

    # 插入足够多的 chunk，测试多次累积不超上限
    for i in range(15):
        _insert_raw(conn, f"fa5_{i}", source_session="sess_fa5",
                    created_offset_seconds=i * 5, retrievability=0.0)

    # 访问最后一个，应对之前的 chunk 产生提升
    # 注意：fap_retr_boost=0.04，lookback=10，最多提升 10 个前面的 chunk
    result = apply_forward_association_primacy(conn, ["fa5_14"])

    # 验证每个被提升的 chunk 提升量不超过 max_boost
    for i in range(4, 14):  # 最近 lookback_window=10 个
        retr = _get_retrievability(conn, f"fa5_{i}")
        # 从 0.0 开始，提升量 = retr - 0.0 = retr
        assert retr <= fap_max_boost + 0.01, (
            f"FA5: fa5_{i} retrievability 提升 {retr:.4f} 不应超过 max_boost={fap_max_boost}"
        )


# ── FA6: session 大小不足 → 不触发 FAP ───────────────────────────────────────────────────────

def test_fa6_small_session_no_fap(conn):
    """FA6: session 大小 < fap_min_session_size(3) → 不触发 FAP。"""
    # 只有 2 个 chunk（< min_session_size=3）
    for i in range(2):
        _insert_raw(conn, f"fa6_{i}", source_session="sess_fa6_small",
                    created_offset_seconds=i * 5)

    retr_before = _get_retrievability(conn, "fa6_0")
    result = apply_forward_association_primacy(conn, ["fa6_1"])
    retr_after = _get_retrievability(conn, "fa6_0")

    assert abs(retr_after - retr_before) < 0.001, (
        f"FA6: session 过小不应触发 FAP，before={retr_before:.4f} after={retr_after:.4f}"
    )
    assert result["fap_boosted"] == 0, f"FA6: fap_boosted 应为 0"


# ── FA7: retrievability 上限 1.0 ─────────────────────────────────────────────────────────────

def test_fa7_retrievability_cap_1(conn):
    """FA7: FAP 提升后 retrievability 不超过 1.0。"""
    for i in range(5):
        _insert_raw(conn, f"fa7_{i}", source_session="sess_fa7",
                    created_offset_seconds=i * 5, retrievability=0.98)

    apply_forward_association_primacy(conn, ["fa7_4"])
    for i in range(4):
        retr = _get_retrievability(conn, f"fa7_{i}")
        assert retr <= 1.0, f"FA7: fa7_{i} retrievability 不应超过 1.0，got {retr:.4f}"


# ── FA8: update_accessed 集成测试 ────────────────────────────────────────────────────────────

def test_fa8_update_accessed_integration(conn):
    """FA8: update_accessed 触发 FAP，较早 session sibling retrievability 不降低。"""
    now_iso = _utcnow().isoformat()
    import datetime as dt
    for i in range(5):
        ts = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=i * 10)
        conn.execute(
            """INSERT OR REPLACE INTO memory_chunks
               (id, project, chunk_type, content, summary, importance, stability,
                created_at, updated_at, retrievability, last_accessed, access_count,
                encode_context, session_type_history, source_session)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (f"fa8_{i}", "test", "observation", f"content fa8_{i}", "summary",
             0.6, 5.0, ts.isoformat(), ts.isoformat(), 0.5, ts.isoformat(),
             0, "test_ctx", "coding", "sess_fa8")
        )
    conn.commit()

    # 访问最后一个 chunk（fa8_4），较早的 sibling retrievability 不应降低
    retr_before = [_get_retrievability(conn, f"fa8_{i}") for i in range(4)]
    update_accessed(conn, ["fa8_4"], session_id="sess_fa8", project="test")
    retr_after = [_get_retrievability(conn, f"fa8_{i}") for i in range(4)]

    for i in range(4):
        assert retr_after[i] >= retr_before[i] - 0.001, (
            f"FA8: fa8_{i} retrievability 不应降低，"
            f"before={retr_before[i]:.4f} after={retr_after[i]:.4f}"
        )
