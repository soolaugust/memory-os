"""
test_iter477_mbe.py — iter477: Memory Binding Effect 单元测试

覆盖：
  MB1: 同 session 同批编码的 chunk → stability 加成
  MB2: 不同 session 的 chunk → 无 MBE 加成
  MB3: mbe_enabled=False → 无加成
  MB4: importance < mbe_min_importance(0.25) → 不参与 MBE
  MB5: 加成随邻居数线性增长，受 mbe_max_boost(0.10) 保护
  MB6: stability 加成后不超过 365.0
  MB7: 直接调用 apply_memory_binding_effect → mbe_boosted > 0
  MB8: 时间窗口外的同 session chunk → 无加成

认知科学依据：
  Eichenbaum (2004) 海马体情节绑定 — 同一事件内的记忆元素被绑定在一起，
    共同激活使各部分稳定性相互增强（episodic binding）。
  Norman & Eichenbaum (2014): 绑定程度 ∝ 编码时的共同激活强度。

OS 类比：Linux THP（Transparent Huge Pages）—
  相邻 4K page 合并为 2MB 大页；合并后 TLB coverage 更大，整体稳定性提升。
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

from store_vfs import ensure_schema, apply_memory_binding_effect
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


def _insert_raw(conn, cid, project="test", chunk_type="observation",
                importance=0.6, stability=5.0, source_session="sess1",
                created_offset_seconds=0):
    import datetime as dt
    ts = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=created_offset_seconds)
    now_iso = ts.isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count,
            encode_context, session_type_history, source_session)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (cid, project, chunk_type, "content " + cid, "summary", importance, stability,
         now_iso, now_iso, 0.5, now_iso, 0, "test_ctx", "coding", source_session)
    )
    conn.commit()


def _get_stability(conn, cid: str) -> float:
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


# ── MB1: 同 session 同批 chunk → stability 加成 ─────────────────────────────────────────────

def test_mb1_same_session_binding(conn):
    """MB1: 同 session 同批（时间窗口内）编码的 chunk → MBE stability 加成。"""
    # 在同一 session 内插入 3 个 chunk，间隔 10 秒（远在 mbe_window_seconds=300 内）
    for i in range(3):
        _insert_raw(conn, f"mb1_{i}", source_session="sess_mb1", created_offset_seconds=i*10)

    stab_before = _get_stability(conn, "mb1_0")
    result = apply_memory_binding_effect(conn, ["mb1_0"])
    stab_after = _get_stability(conn, "mb1_0")

    assert stab_after > stab_before, (
        f"MB1: 同 session 同批 chunk 应获得 MBE 加成，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["mbe_boosted"] > 0, f"MB1: mbe_boosted 应 > 0，got {result}"


# ── MB2: 不同 session → 无加成 ──────────────────────────────────────────────────────────────

def test_mb2_different_session_no_binding(conn):
    """MB2: 不同 session 的 chunk → 无 MBE 加成。"""
    _insert_raw(conn, "mb2_a", source_session="sess_mb2_A")
    _insert_raw(conn, "mb2_b", source_session="sess_mb2_B")  # 不同 session

    stab_before = _get_stability(conn, "mb2_a")
    result = apply_memory_binding_effect(conn, ["mb2_a"])
    stab_after = _get_stability(conn, "mb2_a")

    # mb2_b 是不同 session，不应参与绑定
    assert abs(stab_after - stab_before) < 0.001, (
        f"MB2: 不同 session 不应有 MBE 加成，before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── MB3: mbe_enabled=False → 无加成 ─────────────────────────────────────────────────────────

def test_mb3_disabled_no_boost(conn):
    """MB3: mbe_enabled=False → 无 MBE 加成。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.mbe_enabled":
            return False
        return original_get(key, project=project)

    for i in range(3):
        _insert_raw(conn, f"mb3_{i}", source_session="sess_mb3", created_offset_seconds=i*5)

    stab_before = _get_stability(conn, "mb3_0")
    with mock.patch.object(config, 'get', side_effect=patched_get):
        result = apply_memory_binding_effect(conn, ["mb3_0"])
    stab_after = _get_stability(conn, "mb3_0")

    assert abs(stab_after - stab_before) < 0.001, (
        f"MB3: disabled 时不应有加成，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["mbe_boosted"] == 0, f"MB3: mbe_boosted 应为 0，got {result}"


# ── MB4: importance < mbe_min_importance → 不参与 MBE ────────────────────────────────────────

def test_mb4_low_importance_no_boost(conn):
    """MB4: importance < mbe_min_importance(0.25) → 不参与 MBE。"""
    for i in range(3):
        _insert_raw(conn, f"mb4_{i}", source_session="sess_mb4",
                    created_offset_seconds=i*5, importance=0.10)

    stab_before = _get_stability(conn, "mb4_0")
    result = apply_memory_binding_effect(conn, ["mb4_0"])
    stab_after = _get_stability(conn, "mb4_0")

    assert abs(stab_after - stab_before) < 0.001, (
        f"MB4: 低 importance 不应触发 MBE，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["mbe_boosted"] == 0, f"MB4: mbe_boosted 应为 0"


# ── MB5: 加成受 mbe_max_boost 保护 ──────────────────────────────────────────────────────────

def test_mb5_max_boost_cap(conn):
    """MB5: MBE 加成受 mbe_max_boost(0.10) 保护（即使邻居数很多）。"""
    mbe_max_boost = config.get("store_vfs.mbe_max_boost")  # 0.10
    base = 5.0

    # 插入 10 个同 session chunk（多于 mbe_max_neighbors=5）
    for i in range(10):
        _insert_raw(conn, f"mb5_{i}", source_session="sess_mb5",
                    created_offset_seconds=i*5, stability=base)

    stab_before = _get_stability(conn, "mb5_0")
    apply_memory_binding_effect(conn, ["mb5_0"])
    stab_after = _get_stability(conn, "mb5_0")

    increment = stab_after - stab_before
    max_allowed = base * mbe_max_boost + 0.01
    assert increment <= max_allowed, (
        f"MB5: MBE 增量 {increment:.4f} 不应超过 max_boost 允许的 {max_allowed:.4f}，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert stab_after > stab_before, (
        f"MB5: 应有 MBE 加成，before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── MB6: stability 上限 365.0 ────────────────────────────────────────────────────────────────

def test_mb6_stability_cap_365(conn):
    """MB6: MBE boost 后 stability 不超过 365.0。"""
    base = 364.0
    for i in range(3):
        _insert_raw(conn, f"mb6_{i}", source_session="sess_mb6",
                    created_offset_seconds=i*5, stability=base)

    apply_memory_binding_effect(conn, ["mb6_0"])
    stab = _get_stability(conn, "mb6_0")
    assert stab <= 365.0, f"MB6: stability 不应超过 365.0，got {stab}"


# ── MB7: 直接调用返回 mbe_boosted > 0 ────────────────────────────────────────────────────────

def test_mb7_direct_function_boost(conn):
    """MB7: 直接调用 apply_memory_binding_effect，验证返回 mbe_boosted > 0。"""
    for i in range(4):
        _insert_raw(conn, f"mb7_{i}", source_session="sess_mb7", created_offset_seconds=i*5)

    stab_before = _get_stability(conn, "mb7_0")
    result = apply_memory_binding_effect(conn, ["mb7_0", "mb7_1", "mb7_2"])
    stab_after = _get_stability(conn, "mb7_0")

    assert result["mbe_boosted"] > 0, f"MB7: mbe_boosted 应 > 0，got {result}"
    assert stab_after > stab_before, (
        f"MB7: 应有 MBE 加成，before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── MB8: 时间窗口外的 chunk → 无加成 ─────────────────────────────────────────────────────────

def test_mb8_outside_time_window_no_binding(conn):
    """MB8: 同 session 但超过 mbe_window_seconds(300) 的 chunk → 无 MBE 绑定。"""
    # 插入两个 chunk，时间差 > 300 秒（超出窗口）
    import datetime as dt
    now = dt.datetime.now(dt.timezone.utc)
    far_past = now - dt.timedelta(seconds=600)  # 10 分钟前（超出 300s 窗口）

    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count,
            encode_context, session_type_history, source_session)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("mb8_old", "test", "observation", "content old", "summary", 0.6, 5.0,
         far_past.isoformat(), far_past.isoformat(), 0.5, far_past.isoformat(),
         0, "test_ctx", "coding", "sess_mb8")
    )
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count,
            encode_context, session_type_history, source_session)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("mb8_new", "test", "observation", "content new", "summary", 0.6, 5.0,
         now.isoformat(), now.isoformat(), 0.5, now.isoformat(),
         0, "test_ctx", "coding", "sess_mb8")
    )
    conn.commit()

    # mb8_new 和 mb8_old 时间差 600s > 300s，不应绑定
    stab_before = _get_stability(conn, "mb8_new")
    result = apply_memory_binding_effect(conn, ["mb8_new"])
    stab_after = _get_stability(conn, "mb8_new")

    assert abs(stab_after - stab_before) < 0.001, (
        f"MB8: 时间窗口外 chunk 不应绑定，before={stab_before:.4f} after={stab_after:.4f}"
    )
