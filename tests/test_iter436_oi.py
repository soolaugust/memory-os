"""
test_iter436_oi.py — iter436: Output Interference (OI) 单元测试

覆盖：
  OI1: 同轮注入多 chunk，position >= 1 的 chunk stability 下降
  OI2: position=0 (第一个 chunk) 不受 OI 惩罚
  OI3: 单独注入（单 chunk）→ 无 OI 惩罚（无竞争）
  OI4: importance >= oi_protect_importance(0.85) → 豁免 OI 惩罚
  OI5: oi_enabled=False → 无任何惩罚
  OI6: position 越靠后惩罚越强（factor^pos 递增）
  OI7: oi_decay_factor 可通过 sysctl 配置（0.95 vs 0.99）
  OI8: 跨多条 trace — 同一 chunk 多次出现时取最大 position（最严重干扰）
  OI9: 返回计数正确（penalized, total_examined）

认知科学依据：
  Roediger (1978) "Recall as a self-limiting process" —
  同一回忆测试中，早期项目的输出（output）干扰后续项目的工作记忆占用，
  导致越靠后的序列位置的项目巩固效果越差（cumulative output interference）。

OS 类比：Linux BFQ dispatch batch budget 消耗 —
  同一 dispatch batch 中，第一个 I/O 消耗大部分 budget，后续请求完成的 I/O 减少。
"""
import sys
import sqlite3
import datetime
import json
import unittest.mock as mock
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import tmpfs  # noqa

from store_vfs import (
    ensure_schema,
    apply_output_interference,
)
import config


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _ago_iso(hours: float = 0.0) -> str:
    return (datetime.datetime.now(datetime.timezone.utc) -
            datetime.timedelta(hours=hours)).isoformat()


def _insert_chunk(conn, cid, project="test", stability=5.0, importance=0.6,
                  chunk_type="decision"):
    now = _now_iso()
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0.9, ?, 1)""",
        (cid, project, chunk_type, f"content {cid}", f"summary {cid}",
         importance, stability, now, now, now)
    )
    conn.commit()


def _insert_trace(conn, project, chunk_ids: list, injected=1, hours_ago=0.5):
    """Insert a recall_trace with multiple co-injected chunks."""
    tid = f"trace_{len(chunk_ids)}_{hours_ago}"
    top_k = json.dumps([{"id": cid, "summary": f"summary {cid}", "score": 0.8}
                        for cid in chunk_ids])
    ts = _ago_iso(hours=hours_ago)
    conn.execute(
        """INSERT OR REPLACE INTO recall_traces
           (id, timestamp, session_id, project, prompt_hash,
            candidates_count, top_k_json, injected, reason, duration_ms)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (tid, ts, "test_session", project, "hash123",
         len(chunk_ids), top_k, injected, "test", 10.0)
    )
    conn.commit()


def _get_stability(conn, cid: str) -> float:
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


# ── OI1: 同轮注入多 chunk，后序 chunk stability 下降 ─────────────────────────

def test_oi1_coinjected_chunks_penalized(conn):
    """OI1: position >= 1 的同轮注入 chunk 应受 OI 惩罚。"""
    _insert_chunk(conn, "c0", stability=5.0, importance=0.6)
    _insert_chunk(conn, "c1", stability=5.0, importance=0.6)
    _insert_chunk(conn, "c2", stability=5.0, importance=0.6)
    _insert_trace(conn, "test", ["c0", "c1", "c2"])

    stab_c1_before = _get_stability(conn, "c1")
    stab_c2_before = _get_stability(conn, "c2")
    result = apply_output_interference(conn, "test")

    stab_c1_after = _get_stability(conn, "c1")
    stab_c2_after = _get_stability(conn, "c2")

    assert stab_c1_after < stab_c1_before, (
        f"OI1: position=1 chunk stability 应下降，"
        f"before={stab_c1_before:.3f} after={stab_c1_after:.3f}"
    )
    assert stab_c2_after < stab_c2_before, (
        f"OI1: position=2 chunk stability 应下降，"
        f"before={stab_c2_before:.3f} after={stab_c2_after:.3f}"
    )
    assert result["penalized"] >= 2, f"OI1: 应有 >= 2 个 chunk 受惩罚，got {result}"


# ── OI2: position=0 的 chunk 不受惩罚 ─────────────────────────────────────────

def test_oi2_first_chunk_not_penalized(conn):
    """OI2: 同轮注入的第一个 chunk（position=0）不受 OI 惩罚。"""
    _insert_chunk(conn, "first", stability=5.0, importance=0.6)
    _insert_chunk(conn, "second", stability=5.0, importance=0.6)
    _insert_trace(conn, "test", ["first", "second"])

    stab_before = _get_stability(conn, "first")
    apply_output_interference(conn, "test")
    stab_after = _get_stability(conn, "first")

    assert abs(stab_after - stab_before) < 0.001, (
        f"OI2: position=0 chunk 不应受 OI 惩罚，"
        f"before={stab_before:.3f} after={stab_after:.3f}"
    )


# ── OI3: 单 chunk 注入 → 无惩罚 ───────────────────────────────────────────────

def test_oi3_single_injection_no_penalty(conn):
    """OI3: 只注入了 1 个 chunk 的 trace 无 OI 效应（无竞争）。"""
    _insert_chunk(conn, "solo", stability=5.0, importance=0.6)
    _insert_trace(conn, "test", ["solo"])  # single chunk

    stab_before = _get_stability(conn, "solo")
    result = apply_output_interference(conn, "test")
    stab_after = _get_stability(conn, "solo")

    assert abs(stab_after - stab_before) < 0.001, (
        f"OI3: 单 chunk 注入不应有 OI 惩罚，"
        f"before={stab_before:.3f} after={stab_after:.3f}"
    )
    assert result["penalized"] == 0, f"OI3: penalized 应为 0，got {result}"


# ── OI4: importance >= protect_importance → 豁免 ──────────────────────────────

def test_oi4_high_importance_protected(conn):
    """OI4: importance >= oi_protect_importance(0.85) → 豁免 OI 惩罚。"""
    protect_imp = config.get("store_vfs.oi_protect_importance")  # 0.85
    _insert_chunk(conn, "first", stability=5.0, importance=0.6)
    _insert_chunk(conn, "important_second", stability=5.0, importance=0.90)  # > 0.85
    _insert_trace(conn, "test", ["first", "important_second"])

    stab_before = _get_stability(conn, "important_second")
    apply_output_interference(conn, "test")
    stab_after = _get_stability(conn, "important_second")

    assert abs(stab_after - stab_before) < 0.001, (
        f"OI4: 高 importance chunk 应豁免 OI，"
        f"before={stab_before:.3f} after={stab_after:.3f}"
    )


# ── OI5: oi_enabled=False → 无惩罚 ────────────────────────────────────────────

def test_oi5_disabled_no_penalty(conn):
    """OI5: store_vfs.oi_enabled=False → 无任何 OI 惩罚。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.oi_enabled":
            return False
        return original_get(key, project=project)

    _insert_chunk(conn, "c0", stability=5.0, importance=0.6)
    _insert_chunk(conn, "c1", stability=5.0, importance=0.6)
    _insert_trace(conn, "test", ["c0", "c1"])

    stab_before = _get_stability(conn, "c1")
    with mock.patch.object(config, 'get', side_effect=patched_get):
        result = apply_output_interference(conn, "test")
    stab_after = _get_stability(conn, "c1")

    assert abs(stab_after - stab_before) < 0.001, (
        f"OI5: oi_enabled=False 时不应有惩罚，"
        f"before={stab_before:.3f} after={stab_after:.3f}"
    )
    assert result["penalized"] == 0, f"OI5: penalized 应为 0，got {result}"


# ── OI6: position 越靠后惩罚越强 ──────────────────────────────────────────────

def test_oi6_later_position_more_penalty(conn):
    """OI6: position=2 的 chunk 受惩罚比 position=1 更强（factor^2 < factor^1）。"""
    _insert_chunk(conn, "pos0", stability=5.0, importance=0.6)
    _insert_chunk(conn, "pos1", stability=5.0, importance=0.6)
    _insert_chunk(conn, "pos2", stability=5.0, importance=0.6)
    _insert_trace(conn, "test", ["pos0", "pos1", "pos2"])

    stab_p1_before = _get_stability(conn, "pos1")
    stab_p2_before = _get_stability(conn, "pos2")
    apply_output_interference(conn, "test")
    stab_p1_after = _get_stability(conn, "pos1")
    stab_p2_after = _get_stability(conn, "pos2")

    drop_p1 = stab_p1_before - stab_p1_after
    drop_p2 = stab_p2_before - stab_p2_after

    assert drop_p2 > drop_p1, (
        f"OI6: position=2 应比 position=1 惩罚更强，"
        f"drop_p1={drop_p1:.4f} drop_p2={drop_p2:.4f}"
    )


# ── OI7: oi_decay_factor 可配置 ───────────────────────────────────────────────

def test_oi7_configurable_decay_factor(conn):
    """OI7: oi_decay_factor=0.95 时惩罚比默认 0.99 更强。"""
    original_get = config.get

    # Test with stronger factor (0.95)
    _insert_chunk(conn, "first_95", stability=5.0, importance=0.6)
    _insert_chunk(conn, "second_95", stability=5.0, importance=0.6)
    _insert_trace(conn, "test", ["first_95", "second_95"])

    def patched_95(key, project=None):
        if key == "store_vfs.oi_decay_factor":
            return 0.95
        return original_get(key, project=project)

    stab_before_95 = _get_stability(conn, "second_95")
    with mock.patch.object(config, 'get', side_effect=patched_95):
        apply_output_interference(conn, "test")
    stab_after_95 = _get_stability(conn, "second_95")
    drop_95 = stab_before_95 - stab_after_95

    # Reset stability
    conn.execute("UPDATE memory_chunks SET stability=5.0 WHERE id='second_95'")
    conn.commit()

    # Test with default factor (0.99)
    stab_before_99 = _get_stability(conn, "second_95")
    apply_output_interference(conn, "test")
    stab_after_99 = _get_stability(conn, "second_95")
    drop_99 = stab_before_99 - stab_after_99

    assert drop_95 > drop_99, (
        f"OI7: factor=0.95 惩罚应比 0.99 更强，"
        f"drop_95={drop_95:.4f} drop_99={drop_99:.4f}"
    )


# ── OI8: 同 chunk 出现在多 trace 中，取最大 position ────────────────────────────

def test_oi8_max_position_across_traces(conn):
    """OI8: 同一 chunk 在多条 trace 中，应使用最大 position（最严重干扰）。"""
    _insert_chunk(conn, "anchor", stability=5.0, importance=0.6)
    _insert_chunk(conn, "target", stability=5.0, importance=0.6)
    _insert_chunk(conn, "other1", stability=5.0, importance=0.6)
    _insert_chunk(conn, "other2", stability=5.0, importance=0.6)

    # Trace 1: target at position=1 (mild penalty)
    _insert_trace(conn, "test", ["anchor", "target"], hours_ago=2.0)
    # Trace 2: target at position=3 (stronger penalty)
    # Need unique trace ids, so use different approach
    top_k = json.dumps([
        {"id": "other1", "summary": "s", "score": 0.9},
        {"id": "other2", "summary": "s", "score": 0.8},
        {"id": "anchor", "summary": "s", "score": 0.7},
        {"id": "target", "summary": "s", "score": 0.6},
    ])
    conn.execute(
        """INSERT OR REPLACE INTO recall_traces
           (id, timestamp, session_id, project, prompt_hash,
            candidates_count, top_k_json, injected, reason, duration_ms)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("trace_pos3", _ago_iso(hours=1.0), "test_session", "test", "hash456",
         4, top_k, 1, "test", 10.0)
    )
    conn.commit()

    stab_before = _get_stability(conn, "target")
    apply_output_interference(conn, "test")
    stab_after = _get_stability(conn, "target")

    # Should be penalized with factor^3 (max position across traces)
    oi_factor = config.get("store_vfs.oi_decay_factor")  # 0.99
    expected_min_drop = stab_before * (1 - oi_factor ** 3)
    actual_drop = stab_before - stab_after

    assert actual_drop >= expected_min_drop * 0.9, (
        f"OI8: target 应用最大 position=3 的惩罚，"
        f"expected_min_drop={expected_min_drop:.4f} actual={actual_drop:.4f}"
    )


# ── OI9: 返回计数正确 ─────────────────────────────────────────────────────────

def test_oi9_return_counts_correct(conn):
    """OI9: result dict 中 penalized 和 total_examined 计数正确。"""
    _insert_chunk(conn, "r0", stability=5.0, importance=0.6)
    _insert_chunk(conn, "r1", stability=5.0, importance=0.6)
    _insert_chunk(conn, "r2", stability=5.0, importance=0.6)
    _insert_chunk(conn, "r3", stability=5.0, importance=0.9)  # protected (high imp)
    _insert_trace(conn, "test", ["r0", "r1", "r2", "r3"])

    result = apply_output_interference(conn, "test")

    assert "penalized" in result, "OI9: result 应含 penalized key"
    assert "total_examined" in result, "OI9: result 应含 total_examined key"
    # r1, r2 at pos 1,2 (not protected); r3 at pos 3 (protected by importance=0.9)
    assert result["total_examined"] >= 3, f"OI9: 应检查 >= 3 个 chunk，got {result}"
    assert result["penalized"] >= 2, f"OI9: 应有 >= 2 个 chunk 被惩罚，got {result}"
