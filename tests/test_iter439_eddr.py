"""
test_iter439_eddr.py — iter439: Encoding Depth Decay Resistance 单元测试

覆盖：
  ED1: entity_count >= deep_threshold(5) + stale → stability 增加（深度编码保护）
  ED2: entity_count <= shallow_threshold(1) + stale → stability 轻微下降（浅层编码加速）
  ED3: 中等 entity_count (2-4) → stability 不变（不干预中间态）
  ED4: eddr_enabled=False → 无任何调整
  ED5: access_count >= 2 的 chunk 不受干预（活跃 chunk 无需修复）
  ED6: 深度越高（更多 entities）→ 修复量越大（entity gradient）
  ED7: 深度编码修复后 stability 不超过 365.0
  ED8: shallow chunk stability 不低于 0.1（下限保护）
  ED9: 返回计数正确（deep_boosted, shallow_penalized, total_examined）

认知科学依据：
  Craik & Tulving (1975) "Depth of processing and the retention of words in episodic memory"
    — 深度语义加工（entity-rich encoding）比浅层加工产生更强的记忆痕迹，
    遗忘速率显著降低（深层编码 24h 保留率比浅层高 50-80%）。
  Craik & Lockhart (1972) Levels of Processing framework —
    编码深度 = 语义分析深度；encode_context entity 数量是 depth 的代理指标。

OS 类比：Linux ext4 extent tree depth —
  深层 extent tree（多 entity）的 inode kswapd 驱逐代价更高（更抗 reclaim）。
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

from store_vfs import (
    ensure_schema,
    apply_encoding_depth_decay_resistance,
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


def _ago_iso(days: float = 0.0) -> str:
    return (datetime.datetime.now(datetime.timezone.utc) -
            datetime.timedelta(days=days)).isoformat()


def _make_encode_context(entity_count: int) -> str:
    """生成指定数量 entity 的 encode_context 字符串。"""
    return ", ".join(f"entity_{i}" for i in range(entity_count))


def _insert_chunk(conn, cid, project="test", stability=5.0, importance=0.6,
                  access_count=0, entity_count=5, last_accessed_days_ago=40.0):
    """
    Insert a chunk with controlled encode_context entity count and stale state.
    last_accessed_days_ago > stale_days(30) → stale（被 decay 扫描过）
    """
    now = _now_iso()
    last_accessed = _ago_iso(days=last_accessed_days_ago)
    encode_ctx = _make_encode_context(entity_count)
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count,
            encode_context)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0.9, ?, ?, ?)""",
        (cid, project, "decision", f"content {cid}", f"summary {cid}",
         importance, stability, now, now, last_accessed, access_count, encode_ctx)
    )
    conn.commit()


def _get_stability(conn, cid: str) -> float:
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


# ── ED1: 深度编码 chunk 获得 stability 修复 ──────────────────────────────────────

def test_ed1_deep_encoding_gets_boost(conn):
    """ED1: entity_count >= deep_threshold(5) + stale → stability 增加（深度编码保护）。"""
    deep_threshold = config.get("store_vfs.eddr_deep_threshold")  # 5
    _insert_chunk(conn, "deep_chunk", entity_count=deep_threshold,
                  stability=5.0, last_accessed_days_ago=40.0)

    stab_before = _get_stability(conn, "deep_chunk")
    result = apply_encoding_depth_decay_resistance(conn, "test", stale_days=30)
    stab_after = _get_stability(conn, "deep_chunk")

    assert stab_after > stab_before, (
        f"ED1: 深度编码 chunk 应获得 stability 修复，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["deep_boosted"] >= 1, f"ED1: deep_boosted 应 >= 1，got {result}"


# ── ED2: 浅层编码 chunk 轻微加速衰减 ─────────────────────────────────────────────

def test_ed2_shallow_encoding_penalized(conn):
    """ED2: entity_count <= shallow_threshold(1) + stale → stability 轻微下降。"""
    shallow_threshold = config.get("store_vfs.eddr_shallow_threshold")  # 1
    _insert_chunk(conn, "shallow_chunk", entity_count=shallow_threshold,
                  stability=5.0, last_accessed_days_ago=40.0)

    stab_before = _get_stability(conn, "shallow_chunk")
    result = apply_encoding_depth_decay_resistance(conn, "test", stale_days=30)
    stab_after = _get_stability(conn, "shallow_chunk")

    assert stab_after < stab_before, (
        f"ED2: 浅层编码 chunk 应受轻微惩罚，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["shallow_penalized"] >= 1, f"ED2: shallow_penalized 应 >= 1，got {result}"


# ── ED3: 中等深度 chunk 不干预 ─────────────────────────────────────────────────────

def test_ed3_medium_depth_not_adjusted(conn):
    """ED3: entity_count in (shallow_threshold, deep_threshold) → stability 不变。"""
    _insert_chunk(conn, "medium_chunk", entity_count=3,  # 2 < 3 < 5
                  stability=5.0, last_accessed_days_ago=40.0)

    stab_before = _get_stability(conn, "medium_chunk")
    apply_encoding_depth_decay_resistance(conn, "test", stale_days=30)
    stab_after = _get_stability(conn, "medium_chunk")

    assert abs(stab_after - stab_before) < 0.001, (
        f"ED3: 中等深度 chunk 不应被调整，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── ED4: eddr_enabled=False → 无调整 ─────────────────────────────────────────────

def test_ed4_disabled_no_adjustment(conn):
    """ED4: store_vfs.eddr_enabled=False → 无任何调整。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.eddr_enabled":
            return False
        return original_get(key, project=project)

    _insert_chunk(conn, "disabled_eddr", entity_count=8,
                  stability=5.0, last_accessed_days_ago=40.0)

    stab_before = _get_stability(conn, "disabled_eddr")
    with mock.patch.object(config, 'get', side_effect=patched_get):
        result = apply_encoding_depth_decay_resistance(conn, "test", stale_days=30)
    stab_after = _get_stability(conn, "disabled_eddr")

    assert abs(stab_after - stab_before) < 0.001, (
        f"ED4: disabled 时不应调整，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["deep_boosted"] == 0, f"ED4: deep_boosted 应为 0，got {result}"
    assert result["shallow_penalized"] == 0, f"ED4: shallow_penalized 应为 0，got {result}"


# ── ED5: access_count >= 2 的 chunk 不受干预 ─────────────────────────────────────

def test_ed5_active_chunk_not_adjusted(conn):
    """ED5: access_count >= 2 的 chunk 不触发 EDDR（活跃 chunk 不被衰减扫描）。"""
    _insert_chunk(conn, "active_chunk", entity_count=8,  # deep
                  stability=5.0, access_count=3, last_accessed_days_ago=40.0)

    stab_before = _get_stability(conn, "active_chunk")
    apply_encoding_depth_decay_resistance(conn, "test", stale_days=30)
    stab_after = _get_stability(conn, "active_chunk")

    assert abs(stab_after - stab_before) < 0.001, (
        f"ED5: 活跃 chunk（access_count >= 2）不应受 EDDR 调整，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── ED6: entity 越多 → 修复量越大（entity gradient）──────────────────────────────

def test_ed6_deeper_encoding_more_boost(conn):
    """ED6: entity_count=10 的 chunk 修复量应多于 entity_count=5。"""
    _insert_chunk(conn, "very_deep", entity_count=10,
                  stability=5.0, last_accessed_days_ago=40.0)
    _insert_chunk(conn, "moderately_deep", entity_count=5,
                  stability=5.0, last_accessed_days_ago=40.0)

    stab_vd_before = _get_stability(conn, "very_deep")
    stab_md_before = _get_stability(conn, "moderately_deep")
    apply_encoding_depth_decay_resistance(conn, "test", stale_days=30)
    stab_vd_after = _get_stability(conn, "very_deep")
    stab_md_after = _get_stability(conn, "moderately_deep")

    delta_vd = stab_vd_after - stab_vd_before
    delta_md = stab_md_after - stab_md_before

    assert delta_vd >= delta_md, (
        f"ED6: 更深编码（entity=10）修复量应 >= 较浅编码（entity=5），"
        f"delta_vd={delta_vd:.5f} delta_md={delta_md:.5f}"
    )


# ── ED7: 深度编码修复后 stability 不超过 365.0 ───────────────────────────────────

def test_ed7_stability_cap_365(conn):
    """ED7: 深度编码修复后 stability 不超过 365.0。"""
    _insert_chunk(conn, "near_cap", entity_count=10,
                  stability=364.9, last_accessed_days_ago=40.0)

    apply_encoding_depth_decay_resistance(conn, "test", stale_days=30)
    stab_after = _get_stability(conn, "near_cap")

    assert stab_after <= 365.0, f"ED7: stability 不应超过 365.0，got {stab_after}"


# ── ED8: 浅层编码惩罚后 stability 不低于 0.1 ─────────────────────────────────────

def test_ed8_shallow_penalty_floor(conn):
    """ED8: 浅层编码惩罚后 stability 不低于 0.1（下限保护）。"""
    _insert_chunk(conn, "near_floor", entity_count=0,  # zero entities = most shallow
                  stability=0.15, last_accessed_days_ago=40.0)

    apply_encoding_depth_decay_resistance(conn, "test", stale_days=30)
    stab_after = _get_stability(conn, "near_floor")

    assert stab_after >= 0.1, f"ED8: stability 不应低于 0.1，got {stab_after}"


# ── ED9: 返回计数正确 ─────────────────────────────────────────────────────────────

def test_ed9_return_counts_correct(conn):
    """ED9: result dict 中 deep_boosted, shallow_penalized, total_examined 计数正确。"""
    deep_threshold = config.get("store_vfs.eddr_deep_threshold")   # 5
    shallow_threshold = config.get("store_vfs.eddr_shallow_threshold")  # 1

    # 2 deep chunks
    _insert_chunk(conn, "d1", entity_count=deep_threshold, stability=5.0)
    _insert_chunk(conn, "d2", entity_count=deep_threshold + 3, stability=5.0)
    # 1 shallow chunk
    _insert_chunk(conn, "s1", entity_count=shallow_threshold, stability=5.0)
    # 1 medium chunk (no intervention)
    _insert_chunk(conn, "m1", entity_count=3, stability=5.0)

    result = apply_encoding_depth_decay_resistance(conn, "test", stale_days=30)

    assert "deep_boosted" in result, "ED9: result 应含 deep_boosted key"
    assert "shallow_penalized" in result, "ED9: result 应含 shallow_penalized key"
    assert "total_examined" in result, "ED9: result 应含 total_examined key"
    assert result["deep_boosted"] >= 2, f"ED9: 应有 >= 2 个深度编码 chunk，got {result}"
    assert result["shallow_penalized"] >= 1, f"ED9: 应有 >= 1 个浅层编码 chunk，got {result}"
    assert result["total_examined"] >= 4, f"ED9: total_examined 应 >= 4，got {result}"
