"""
test_iter463_oie.py — iter463: Output Interference Effect 单元测试

覆盖：
  OI1: 列表末尾 chunk（高 position_ratio）stability 降低
  OI2: 列表开头 chunk（position_ratio=0）无惩罚
  OI3: 列表长度 < oie_min_list_len(3) → 无 OIE 惩罚
  OI4: oie_enabled=False → 无任何惩罚
  OI5: importance < oie_min_importance(0.25) → 不受 OIE 惩罚
  OI6: 惩罚量受 oie_max_penalty(0.05) 上限保护
  OI7: stability 惩罚后不低于 0.1（floor 保护）
  OI8: 末尾 chunk 的 stability 低于开头 chunk（串行位置效应验证）

认知科学依据：
  Postman & Underwood (1973) "Critical issues in interference theory" —
    顺序回忆（串行检索）中，后位项目受前位项目的输出干扰（output interference）。
  Roediger (1974): 自由回忆中，回忆第 N 个词后，第 N+1 个词可及性下降约 5-8%。
  Smith et al. (1978): 顺序输出干扰随列表长度增大而累积（serial position effect）。

OS 类比：Linux TLB invalidation cascade —
  顺序 shootdown 多个 TLB entry 时，后续 entry 因 pipeline stall 累积而
  经历更高的 invalidation latency（IPI broadcast collision 递增）。
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

from store_vfs import ensure_schema, apply_output_interference_effect
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


def _insert_chunk(conn, cid, project="test", stability=5.0, importance=0.6,
                  chunk_type="decision"):
    now_iso = _utcnow().isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count,
            encode_context, session_type_history)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (cid, project, chunk_type, f"content {cid}", f"summary {cid}",
         importance, stability, now_iso, now_iso, 0.5,
         now_iso, 2, "kernel_mm", "")
    )
    conn.commit()


def _get_stability(conn, cid: str) -> float:
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


# ── OI1: 末尾 chunk → stability 降低 ──────────────────────────────────────────────────

def test_oi1_last_chunk_penalized(conn):
    """OI1: 列表末尾 chunk（position_ratio=1.0）stability 应降低。"""
    ids = ["oie_1a", "oie_1b", "oie_1c"]
    for cid in ids:
        _insert_chunk(conn, cid, stability=5.0, importance=0.6)

    stab_last_before = _get_stability(conn, ids[-1])
    result = apply_output_interference_effect(conn, ids, "test")
    stab_last_after = _get_stability(conn, ids[-1])

    assert stab_last_after < stab_last_before, (
        f"OI1: 末尾 chunk stability 应降低，"
        f"before={stab_last_before:.4f} after={stab_last_after:.4f}"
    )
    assert result["oie_penalized"] >= 1, f"OI1: oie_penalized 应 >= 1，got {result}"


# ── OI2: 开头 chunk → 无惩罚 ──────────────────────────────────────────────────────────

def test_oi2_first_chunk_no_penalty(conn):
    """OI2: 列表开头 chunk（position_ratio=0）无 OIE 惩罚。"""
    ids = ["oie_2a", "oie_2b", "oie_2c"]
    for cid in ids:
        _insert_chunk(conn, cid, stability=5.0, importance=0.6)

    stab_first_before = _get_stability(conn, ids[0])
    apply_output_interference_effect(conn, ids, "test")
    stab_first_after = _get_stability(conn, ids[0])

    assert abs(stab_first_after - stab_first_before) < 0.001, (
        f"OI2: 开头 chunk 不应受 OIE 惩罚，"
        f"before={stab_first_before:.4f} after={stab_first_after:.4f}"
    )


# ── OI3: 列表长度不足 → 无惩罚 ───────────────────────────────────────────────────────

def test_oi3_short_list_no_penalty(conn):
    """OI3: 列表长度 < oie_min_list_len(3) → 无 OIE 惩罚。"""
    ids = ["oie_3a", "oie_3b"]  # 只有 2 个，< min_list_len=3
    for cid in ids:
        _insert_chunk(conn, cid, stability=5.0, importance=0.6)

    stabs_before = {cid: _get_stability(conn, cid) for cid in ids}
    result = apply_output_interference_effect(conn, ids, "test")
    stabs_after = {cid: _get_stability(conn, cid) for cid in ids}

    for cid in ids:
        assert abs(stabs_after[cid] - stabs_before[cid]) < 0.001, (
            f"OI3: 短列表不应有 OIE 惩罚，{cid}: before={stabs_before[cid]:.4f} after={stabs_after[cid]:.4f}"
        )
    assert result["oie_penalized"] == 0, f"OI3: oie_penalized 应为 0，got {result}"


# ── OI4: oie_enabled=False → 无惩罚 ──────────────────────────────────────────────────

def test_oi4_disabled_no_penalty(conn):
    """OI4: oie_enabled=False → 无任何 OIE 惩罚。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.oie_enabled":
            return False
        return original_get(key, project=project)

    ids = ["oie_4a", "oie_4b", "oie_4c", "oie_4d"]
    for cid in ids:
        _insert_chunk(conn, cid, stability=5.0, importance=0.6)

    stabs_before = {cid: _get_stability(conn, cid) for cid in ids}
    with mock.patch.object(config, 'get', side_effect=patched_get):
        result = apply_output_interference_effect(conn, ids, "test")
    stabs_after = {cid: _get_stability(conn, cid) for cid in ids}

    for cid in ids:
        assert abs(stabs_after[cid] - stabs_before[cid]) < 0.001, (
            f"OI4: disabled 时 {cid} 不应有 OIE 惩罚"
        )
    assert result["oie_penalized"] == 0, f"OI4: oie_penalized 应为 0，got {result}"


# ── OI5: importance 不足 → 不受 OIE 惩罚 ────────────────────────────────────────────

def test_oi5_low_importance_no_penalty(conn):
    """OI5: importance < oie_min_importance(0.25) → 不受 OIE 惩罚。"""
    ids = ["oie_5a", "oie_5b", "oie_5c"]
    for cid in ids:
        _insert_chunk(conn, cid, stability=5.0, importance=0.10)  # < 0.25

    stabs_before = {cid: _get_stability(conn, cid) for cid in ids}
    apply_output_interference_effect(conn, ids, "test")
    stabs_after = {cid: _get_stability(conn, cid) for cid in ids}

    for cid in ids:
        assert abs(stabs_after[cid] - stabs_before[cid]) < 0.001, (
            f"OI5: 低 importance 不应受 OIE 惩罚，{cid}: before={stabs_before[cid]:.4f}"
        )


# ── OI6: 惩罚受 oie_max_penalty 保护 ─────────────────────────────────────────────────

def test_oi6_max_penalty_cap(conn):
    """OI6: 末尾 chunk 的惩罚不超过 oie_max_penalty(0.05)。"""
    oie_max_penalty = config.get("store_vfs.oie_max_penalty")  # 0.05
    base = 5.0

    ids = ["oie_6a", "oie_6b", "oie_6c", "oie_6d"]
    for cid in ids:
        _insert_chunk(conn, cid, stability=base, importance=0.6)

    stab_last_before = _get_stability(conn, ids[-1])
    apply_output_interference_effect(conn, ids, "test")
    stab_last_after = _get_stability(conn, ids[-1])

    penalty = (stab_last_before - stab_last_after) / stab_last_before
    assert penalty <= oie_max_penalty + 0.001, (
        f"OI6: 惩罚比例 {penalty:.5f} 不应超过 max_penalty={oie_max_penalty}，"
        f"before={stab_last_before:.4f} after={stab_last_after:.4f}"
    )


# ── OI7: stability 惩罚后不低于 0.1 ──────────────────────────────────────────────────

def test_oi7_stability_floor_0_1(conn):
    """OI7: OIE 惩罚后 stability 不低于 0.1。"""
    ids = ["oie_7a", "oie_7b", "oie_7c"]
    for cid in ids:
        _insert_chunk(conn, cid, stability=0.11, importance=0.6)

    apply_output_interference_effect(conn, ids, "test")

    for cid in ids:
        stab = _get_stability(conn, cid)
        assert stab >= 0.1, f"OI7: OIE 后 {cid} stability 不应低于 0.1，got {stab}"


# ── OI8: 末尾 chunk stability < 开头 chunk ─────────────────────────────────────────

def test_oi8_serial_position_effect(conn):
    """OI8: 串行位置效应验证：末尾 chunk stability <= 开头 chunk stability。"""
    ids = ["oie_8a", "oie_8b", "oie_8c", "oie_8d", "oie_8e"]
    for cid in ids:
        _insert_chunk(conn, cid, stability=5.0, importance=0.6)

    apply_output_interference_effect(conn, ids, "test")

    stab_first = _get_stability(conn, ids[0])
    stab_last = _get_stability(conn, ids[-1])

    assert stab_first >= stab_last, (
        f"OI8: 串行位置效应：开头 stability 应 >= 末尾，"
        f"first={stab_first:.4f} last={stab_last:.4f}"
    )
