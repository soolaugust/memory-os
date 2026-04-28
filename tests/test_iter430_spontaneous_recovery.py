"""
test_iter430_spontaneous_recovery.py — iter430: Spontaneous Recovery 单元测试

覆盖：
  SR1: 满足所有条件（days>=3, access>=3, imp>=0.65）→ chunk 从 swap 恢复
  SR2: swap 时间不足（< sr_min_swap_days）→ 不恢复
  SR3: access_count 不足（< sr_min_access_count）→ 不恢复
  SR4: importance 不足（< sr_min_importance）→ 不恢复
  SR5: sr_enabled=False → 不恢复
  SR6: 恢复后 stability × sr_recovery_boost（默认 1.15）
  SR7: max_recover_per_run 限制（每次最多恢复 N 个）
  SR8: 恢复后 chunk 在主表中重新可检索
  SR9: 返回 dict 包含正确的 recovered 和 boosted 计数
  SR10: 多个满足条件的 chunk 按 importance DESC 优先恢复

认知科学依据：
  Pavlov (1927) — 条件反射被抑制（extinction）后经过"休息"可自发恢复。
  Rescorla (1997) — 恢复程度与休息时间和初始重要性正相关。

OS 类比：Linux MGLRU active 列表晋升 —
  swap 分区中高历史访问的页面经过一段时间后被提升回 active generation。
"""
import sys
import sqlite3
import pytest
import json
import zlib
import base64
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import tmpfs  # noqa
from store_vfs import ensure_schema
from store_swap import run_spontaneous_recovery


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    # 确保 swap_chunks 表存在
    c.execute("""
        CREATE TABLE IF NOT EXISTS swap_chunks (
            id TEXT PRIMARY KEY,
            swapped_at TEXT NOT NULL,
            project TEXT NOT NULL,
            chunk_type TEXT NOT NULL,
            original_importance REAL DEFAULT 0.5,
            access_count_at_swap INTEGER DEFAULT 0,
            compressed_data TEXT NOT NULL
        )
    """)
    c.commit()
    yield c
    c.close()


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _make_swap_entry(conn, cid, project="test", importance=0.8,
                     access_count=5, days_ago=5.0, chunk_type="decision",
                     stability=1.0):
    """在 swap_chunks 中插入测试数据，模拟 chunk 已被 swap out。"""
    swapped_at = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    now = _now_iso()
    chunk_data = {
        "id": cid,
        "created_at": now,
        "updated_at": now,
        "project": project,
        "source_session": "test_session",
        "chunk_type": chunk_type,
        "content": f"content of {cid}",
        "summary": f"summary of {cid}",
        "tags": "",
        "importance": importance,
        "retrievability": 0.9,
        "last_accessed": now,
        "access_count": access_count,
        "stability": stability,
    }
    raw_json = json.dumps(chunk_data, ensure_ascii=False)
    compressed = base64.b64encode(zlib.compress(raw_json.encode("utf-8"))).decode("ascii")
    conn.execute(
        """INSERT OR REPLACE INTO swap_chunks
           (id, swapped_at, project, chunk_type, original_importance,
            access_count_at_swap, compressed_data)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (cid, swapped_at, project, chunk_type, importance, access_count, compressed),
    )
    conn.commit()


def _get_main_chunk(conn, cid):
    return conn.execute(
        "SELECT id, stability FROM memory_chunks WHERE id=?", (cid,)
    ).fetchone()


def _in_swap(conn, cid):
    row = conn.execute("SELECT id FROM swap_chunks WHERE id=?", (cid,)).fetchone()
    return row is not None


# ── SR1: 满足所有条件 → chunk 从 swap 恢复 ─────────────────────────────────────

def test_sr1_fully_qualified_chunk_recovers(conn):
    """SR1: 满足 days>=3, access>=3, imp>=0.65 的 chunk 应从 swap 恢复到主表。"""
    _make_swap_entry(conn, "sr1_chunk", importance=0.8, access_count=5, days_ago=5.0)

    result = run_spontaneous_recovery(conn, "test")
    assert result["recovered"] >= 1, f"SR1: 应恢复 1 个 chunk，got {result}"
    assert _get_main_chunk(conn, "sr1_chunk") is not None, "SR1: chunk 应在主表中"
    assert not _in_swap(conn, "sr1_chunk"), "SR1: chunk 应从 swap 中移除"


# ── SR2: swap 时间不足 → 不恢复 ────────────────────────────────────────────────

def test_sr2_too_recent_swap_not_recovered(conn):
    """SR2: 刚 swap out（< 3 天）的 chunk 不应恢复（需要足够休息时间）。"""
    _make_swap_entry(conn, "sr2_chunk", importance=0.8, access_count=5, days_ago=1.0)

    result = run_spontaneous_recovery(conn, "test")
    assert result["recovered"] == 0, \
        f"SR2: 不足 3 天的 chunk 不应恢复，got {result}"
    assert _in_swap(conn, "sr2_chunk"), "SR2: chunk 应仍在 swap 中"


# ── SR3: access_count 不足 → 不恢复 ────────────────────────────────────────────

def test_sr3_low_access_count_not_recovered(conn):
    """SR3: 历史 access_count < 3 的 chunk 不触发自发恢复。"""
    _make_swap_entry(conn, "sr3_chunk", importance=0.8, access_count=1, days_ago=5.0)

    result = run_spontaneous_recovery(conn, "test")
    assert result["recovered"] == 0, \
        f"SR3: access_count<3 的 chunk 不应恢复，got {result}"


# ── SR4: importance 不足 → 不恢复 ─────────────────────────────────────────────

def test_sr4_low_importance_not_recovered(conn):
    """SR4: importance < 0.65 的 chunk 不参与自发恢复（不够重要）。"""
    _make_swap_entry(conn, "sr4_chunk", importance=0.50, access_count=5, days_ago=5.0)

    result = run_spontaneous_recovery(conn, "test")
    assert result["recovered"] == 0, \
        f"SR4: importance<0.65 的 chunk 不应恢复，got {result}"


# ── SR5: sr_enabled=False → 不恢复 ─────────────────────────────────────────────

def test_sr5_disabled_no_recovery(conn):
    """SR5: swap.sr_enabled=False 时，不执行任何自发恢复。"""
    import unittest.mock as mock
    import config as _config

    _make_swap_entry(conn, "sr5_chunk", importance=0.8, access_count=5, days_ago=5.0)

    original_get = _config.get
    def patched_get(key, project=None):
        if key == "swap.sr_enabled":
            return False
        return original_get(key, project=project)

    with mock.patch.object(_config, 'get', side_effect=patched_get):
        result = run_spontaneous_recovery(conn, "test")

    assert result["recovered"] == 0, \
        f"SR5: 禁用 SR 后不应恢复，got {result}"
    assert _in_swap(conn, "sr5_chunk"), "SR5: chunk 应仍在 swap 中"


# ── SR6: 恢复后 stability × sr_recovery_boost ─────────────────────────────────

def test_sr6_stability_boosted_after_recovery(conn):
    """SR6: 恢复后 chunk stability 应乘以 recovery_boost（默认 1.15）。"""
    _make_swap_entry(conn, "sr6_chunk", importance=0.8, access_count=5, days_ago=5.0,
                     stability=2.0)

    result = run_spontaneous_recovery(conn, "test")
    assert result["recovered"] >= 1, f"SR6: 应恢复 chunk，got {result}"
    assert result["boosted"] >= 1, f"SR6: 应有 boosted 计数，got {result}"

    row = _get_main_chunk(conn, "sr6_chunk")
    assert row is not None, "SR6: chunk 应在主表中"
    stab = float(row[1]) if row[1] is not None else 0.0
    # 2.0 × 1.15 = 2.3
    assert stab > 2.0, f"SR6: stability 应被提升，got {stab}"
    assert abs(stab - 2.3) < 0.05, f"SR6: 应约为 2.0×1.15=2.3，got {stab}"


# ── SR7: max_recover_per_run 限制 ────────────────────────────────────────────

def test_sr7_max_recover_per_run_limit(conn):
    """SR7: max_recover_per_run=2 时，即使有 5 个满足条件，也只恢复 2 个。"""
    import unittest.mock as mock
    import config as _config

    for i in range(5):
        _make_swap_entry(conn, f"sr7_chunk_{i}", importance=0.8,
                         access_count=5, days_ago=5.0)

    original_get = _config.get
    def patched_get(key, project=None):
        if key == "swap.sr_max_recover_per_run":
            return 2
        return original_get(key, project=project)

    with mock.patch.object(_config, 'get', side_effect=patched_get):
        result = run_spontaneous_recovery(conn, "test")

    assert result["recovered"] == 2, \
        f"SR7: max=2 时应只恢复 2 个，got {result['recovered']}"


# ── SR8: 恢复后 chunk 在主表中可检索 ─────────────────────────────────────────

def test_sr8_chunk_queryable_after_recovery(conn):
    """SR8: 恢复后 chunk 在主表中可查询（project、chunk_type 正确）。"""
    _make_swap_entry(conn, "sr8_chunk", project="test", importance=0.9,
                     access_count=10, days_ago=7.0, chunk_type="design_constraint")

    run_spontaneous_recovery(conn, "test")

    row = conn.execute(
        "SELECT id, project, chunk_type FROM memory_chunks WHERE id=?", ("sr8_chunk",)
    ).fetchone()
    assert row is not None, "SR8: 恢复后 chunk 应在主表"
    assert row[1] == "test", f"SR8: project 应为 test，got {row[1]}"
    assert row[2] == "design_constraint", f"SR8: chunk_type 应为 design_constraint，got {row[2]}"


# ── SR9: 返回 dict 包含正确的 recovered/boosted 计数 ─────────────────────────

def test_sr9_return_dict_accurate(conn):
    """SR9: run_spontaneous_recovery 返回 dict 中 recovered 和 boosted 计数准确。"""
    for i in range(3):
        _make_swap_entry(conn, f"sr9_chunk_{i}", importance=0.8,
                         access_count=5, days_ago=5.0)

    result = run_spontaneous_recovery(conn, "test")

    assert "recovered" in result, "SR9: 应有 recovered 字段"
    assert "boosted" in result, "SR9: 应有 boosted 字段"
    assert result["recovered"] == 3, \
        f"SR9: recovered 应为 3，got {result['recovered']}"
    assert result["boosted"] == 3, \
        f"SR9: boosted 应为 3，got {result['boosted']}"


# ── SR10: 多个满足条件的 chunk 按 importance DESC 优先恢复 ────────────────────

def test_sr10_highest_importance_recovered_first(conn):
    """SR10: max_recover=1 时，importance 最高的 chunk 被优先恢复。"""
    import unittest.mock as mock
    import config as _config

    _make_swap_entry(conn, "low_imp_chunk", importance=0.70, access_count=5, days_ago=5.0)
    _make_swap_entry(conn, "high_imp_chunk", importance=0.95, access_count=5, days_ago=5.0)

    original_get = _config.get
    def patched_get(key, project=None):
        if key == "swap.sr_max_recover_per_run":
            return 1
        return original_get(key, project=project)

    with mock.patch.object(_config, 'get', side_effect=patched_get):
        result = run_spontaneous_recovery(conn, "test")

    assert result["recovered"] == 1, "SR10: 应恢复 1 个"
    # 高 importance 应被优先恢复
    assert _get_main_chunk(conn, "high_imp_chunk") is not None, \
        "SR10: 高 importance chunk 应被优先恢复"
    assert _get_main_chunk(conn, "low_imp_chunk") is None, \
        "SR10: 低 importance chunk 不应被恢复（max=1）"
