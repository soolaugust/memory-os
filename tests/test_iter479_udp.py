"""
test_iter479_udp.py — iter479: Use-Dependent Plasticity 单元测试

覆盖：
  UD1: 多个 chunk 共同访问（batch >= 2）→ 各 chunk stability 提升
  UD2: 单独访问（batch = 1）→ 无 UDP 加成
  UD3: udp_enabled=False → 无加成
  UD4: importance < udp_min_importance(0.25) → 不参与 UDP
  UD5: 加成随对等数增加而增加（但受 udp_max_boost=0.08 保护）
  UD6: stability 上限 365.0 保护
  UD7: 直接调用 apply_use_dependent_plasticity → udp_boosted > 0
  UD8: update_accessed 集成 — 批量访问触发 UDP

认知科学依据：
  Hebb (1949) "Neurons that fire together wire together" —
    共同激活的记忆节点间连接加强（Hebbian plasticity）。
  Bhattacharya & Bhattacharya (2009) LTP — 重复共激活 → AMPA receptor density 上调。

OS 类比：Linux working set model — 共同访问 page 各自 refcount++，更难被回收。
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

from store_vfs import ensure_schema, apply_use_dependent_plasticity, update_accessed
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


def _insert_raw(conn, cid, project="test", importance=0.6, stability=5.0,
                source_session="sess1"):
    now_iso = _utcnow().isoformat()
    import datetime as _dt
    la_iso = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(minutes=10)).isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count,
            encode_context, session_type_history, source_session)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (cid, project, "observation", "content " + cid, "summary", importance, stability,
         now_iso, now_iso, 0.5, la_iso, 0, "test_ctx", "coding", source_session)
    )
    conn.commit()


def _get_stability(conn, cid: str) -> float:
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


# ── UD1: 批量共同访问 → stability 提升 ──────────────────────────────────────────────────────

def test_ud1_co_access_boosts_stability(conn):
    """UD1: batch >= 2 的 chunk 共同访问 → 各 chunk stability 提升。"""
    for i in range(3):
        _insert_raw(conn, f"ud1_{i}")

    stab_before = {f"ud1_{i}": _get_stability(conn, f"ud1_{i}") for i in range(3)}
    result = apply_use_dependent_plasticity(conn, ["ud1_0", "ud1_1", "ud1_2"])
    stab_after = {f"ud1_{i}": _get_stability(conn, f"ud1_{i}") for i in range(3)}

    for i in range(3):
        assert stab_after[f"ud1_{i}"] > stab_before[f"ud1_{i}"], (
            f"UD1: ud1_{i} stability 应提升，"
            f"before={stab_before[f'ud1_{i}']:.4f} after={stab_after[f'ud1_{i}']:.4f}"
        )
    assert result["udp_boosted"] >= 1, f"UD1: udp_boosted 应 >= 1，got {result}"


# ── UD2: 单独访问 → 无 UDP 加成 ─────────────────────────────────────────────────────────────

def test_ud2_single_access_no_boost(conn):
    """UD2: 单独访问（batch = 1）→ 无 UDP 加成（需要至少 2 个 chunk）。"""
    _insert_raw(conn, "ud2_0")

    stab_before = _get_stability(conn, "ud2_0")
    result = apply_use_dependent_plasticity(conn, ["ud2_0"])
    stab_after = _get_stability(conn, "ud2_0")

    assert abs(stab_after - stab_before) < 0.001, (
        f"UD2: 单独访问不应有 UDP 加成，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["udp_boosted"] == 0, f"UD2: udp_boosted 应为 0，got {result}"


# ── UD3: udp_enabled=False → 无加成 ─────────────────────────────────────────────────────────

def test_ud3_disabled_no_boost(conn):
    """UD3: udp_enabled=False → 无 UDP 加成。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.udp_enabled":
            return False
        return original_get(key, project=project)

    for i in range(3):
        _insert_raw(conn, f"ud3_{i}")

    stab_before = _get_stability(conn, "ud3_0")
    with mock.patch.object(config, 'get', side_effect=patched_get):
        result = apply_use_dependent_plasticity(conn, ["ud3_0", "ud3_1", "ud3_2"])
    stab_after = _get_stability(conn, "ud3_0")

    assert abs(stab_after - stab_before) < 0.001, (
        f"UD3: disabled 时不应有加成，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["udp_boosted"] == 0, f"UD3: udp_boosted 应为 0，got {result}"


# ── UD4: importance 不足 → 不参与 UDP ────────────────────────────────────────────────────────

def test_ud4_low_importance_no_boost(conn):
    """UD4: importance < udp_min_importance(0.25) → 不触发 UDP。"""
    for i in range(3):
        _insert_raw(conn, f"ud4_{i}", importance=0.10)

    stab_before = _get_stability(conn, "ud4_0")
    result = apply_use_dependent_plasticity(conn, ["ud4_0", "ud4_1", "ud4_2"])
    stab_after = _get_stability(conn, "ud4_0")

    assert abs(stab_after - stab_before) < 0.001, (
        f"UD4: 低 importance 不应触发 UDP，before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── UD5: 加成受 udp_max_boost 保护 ──────────────────────────────────────────────────────────

def test_ud5_max_boost_cap(conn):
    """UD5: UDP 加成受 udp_max_boost(0.08) 保护（peer 数量超出 max_peers=5 时）。"""
    udp_max_boost = config.get("store_vfs.udp_max_boost")  # 0.08
    base = 5.0

    # 插入 8 个 chunk（超过 max_peers=5）
    for i in range(8):
        _insert_raw(conn, f"ud5_{i}", stability=base)

    stab_before = _get_stability(conn, "ud5_0")
    apply_use_dependent_plasticity(conn, [f"ud5_{i}" for i in range(8)])
    stab_after = _get_stability(conn, "ud5_0")

    increment = stab_after - stab_before
    max_allowed = base * udp_max_boost + 0.01
    assert increment <= max_allowed, (
        f"UD5: UDP 增量 {increment:.4f} 不应超过 max_boost 允许的 {max_allowed:.4f}，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert stab_after > stab_before, (
        f"UD5: 应有 UDP 加成，before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── UD6: stability 上限 365.0 ────────────────────────────────────────────────────────────────

def test_ud6_stability_cap_365(conn):
    """UD6: UDP 加成后 stability 不超过 365.0。"""
    for i in range(3):
        _insert_raw(conn, f"ud6_{i}", stability=364.0)

    apply_use_dependent_plasticity(conn, ["ud6_0", "ud6_1", "ud6_2"])
    stab = _get_stability(conn, "ud6_0")
    assert stab <= 365.0, f"UD6: stability 不应超过 365.0，got {stab}"


# ── UD7: 直接调用返回 udp_boosted > 0 ────────────────────────────────────────────────────────

def test_ud7_direct_function_boost(conn):
    """UD7: apply_use_dependent_plasticity 直接调用返回 udp_boosted > 0。"""
    for i in range(4):
        _insert_raw(conn, f"ud7_{i}")

    stabs_before = [_get_stability(conn, f"ud7_{i}") for i in range(4)]
    result = apply_use_dependent_plasticity(conn, ["ud7_0", "ud7_1", "ud7_2", "ud7_3"])
    stabs_after = [_get_stability(conn, f"ud7_{i}") for i in range(4)]

    assert result["udp_boosted"] > 0, f"UD7: udp_boosted 应 > 0，got {result}"
    assert any(stabs_after[i] > stabs_before[i] for i in range(4)), (
        "UD7: 至少有一个 chunk 应获得 UDP 加成"
    )


# ── UD8: update_accessed 集成 ─────────────────────────────────────────────────────────────────

def test_ud8_update_accessed_integration(conn):
    """UD8: update_accessed 批量访问 >= 2 个 chunk → UDP 触发，stability 提升。"""
    now_iso = _utcnow().isoformat()
    import datetime as _dt
    la_iso = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(minutes=10)).isoformat()
    for cid in ["ud8_a", "ud8_b", "ud8_c"]:
        conn.execute(
            """INSERT OR REPLACE INTO memory_chunks
               (id, project, chunk_type, content, summary, importance, stability,
                created_at, updated_at, retrievability, last_accessed, access_count,
                encode_context, session_type_history, source_session)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (cid, "test", "observation", "content " + cid, "summary", 0.6, 5.0,
             now_iso, now_iso, 0.5, la_iso, 0, "test_ctx", "coding", "sess_ud8")
        )
    conn.commit()

    stabs_before = {cid: _get_stability(conn, cid) for cid in ["ud8_a", "ud8_b", "ud8_c"]}
    update_accessed(conn, ["ud8_a", "ud8_b", "ud8_c"], session_id="sess_ud8", project="test")
    stabs_after = {cid: _get_stability(conn, cid) for cid in ["ud8_a", "ud8_b", "ud8_c"]}

    # 至少有一个 chunk 稳定性提升
    any_boosted = any(stabs_after[cid] > stabs_before[cid] for cid in ["ud8_a", "ud8_b", "ud8_c"])
    assert any_boosted, (
        f"UD8: update_accessed 批量访问后应触发 UDP，"
        f"before={stabs_before} after={stabs_after}"
    )
