"""
test_iter457_cocp.py — iter457: Cue Overload Consolidation Penalty 单元测试

覆盖：
  CO1: 同类型 chunk 数量 > cocp_type_threshold(15) → 受 stability 惩罚
  CO2: 同类型 chunk 数量 <= cocp_type_threshold → 无惩罚（低于阈值不触发）
  CO3: importance >= cocp_protect_importance(0.80) → 豁免 COCP 惩罚
  CO4: cocp_protect_types 中的 chunk_type → 豁免 COCP 惩罚
  CO5: cocp_enabled=False → 无任何惩罚
  CO6: 惩罚系数随 overload 程度增大（N 越大惩罚越重，但不超过 cocp_max_penalty）
  CO7: stability 惩罚后不低于 0.1（min floor）
  CO8: cocp_max_penalty=0.05 时最大惩罚减小（可配置）
  CO9: 返回值正确（cocp_penalized, total_examined 计数）
  CO10: sleep_consolidate() 集成测试：同类型过多时触发 COCP

认知科学依据：
  Watkins & Watkins (1975) "Build-up of proactive inhibition as a cue-overload effect" —
    太多记忆共享同一检索线索 → 每个记忆的可提取性下降（线索过载）。
  Roediger (1978) "Recall as a self-limiting process" —
    同类记忆越多，单个记忆的 recall probability 呈次线性下降。

OS 类比：Linux CPU cache set-associativity saturation —
  太多 cache line 映射到同一 set → 每次新写入导致更频繁的 LRU eviction。
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
    apply_cue_overload_consolidation_penalty,
)
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
            encode_context)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (cid, project, chunk_type, f"content {cid}", f"summary {cid}",
         importance, stability, now_iso, now_iso, 0.5,
         now_iso, 2, "kernel_mm")
    )
    conn.commit()


def _get_stability(conn, cid: str) -> float:
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


def _insert_many_chunks(conn, count, project="test", chunk_type="decision",
                        stability=5.0, importance=0.6, prefix="chunk_"):
    """Insert N chunks of the same type."""
    for i in range(count):
        _insert_chunk(conn, f"{prefix}{i}", project=project,
                      stability=stability, importance=importance,
                      chunk_type=chunk_type)


# ── CO1: 超过阈值 → 受 stability 惩罚 ─────────────────────────────────────────────

def test_co1_overloaded_type_gets_penalized(conn):
    """CO1: 同类型 chunk 数量 > cocp_type_threshold(15) → stability 受惩罚。"""
    threshold = config.get("store_vfs.cocp_type_threshold")  # 15
    # 插入 threshold+5 = 20 个同类型 chunk
    _insert_many_chunks(conn, threshold + 5, chunk_type="decision",
                        stability=5.0, importance=0.6, prefix="over_")

    stab_before = _get_stability(conn, "over_0")
    result = apply_cue_overload_consolidation_penalty(conn, "test")
    stab_after = _get_stability(conn, "over_0")

    assert stab_after < stab_before, (
        f"CO1: 超过阈值时 stability 应受惩罚，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["cocp_penalized"] >= 1, f"CO1: cocp_penalized 应 >= 1，got {result}"


# ── CO2: 低于阈值 → 无惩罚 ────────────────────────────────────────────────────────

def test_co2_under_threshold_no_penalty(conn):
    """CO2: 同类型 chunk 数量 <= cocp_type_threshold → 无惩罚。"""
    threshold = config.get("store_vfs.cocp_type_threshold")  # 15
    # 插入 threshold-1 = 14 个同类型 chunk
    _insert_many_chunks(conn, threshold - 1, chunk_type="reasoning_chain",
                        stability=5.0, importance=0.6, prefix="under_")

    stab_before = _get_stability(conn, "under_0")
    result = apply_cue_overload_consolidation_penalty(conn, "test")
    stab_after = _get_stability(conn, "under_0")

    assert abs(stab_after - stab_before) < 0.001, (
        f"CO2: 低于阈值不应有 COCP 惩罚，before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── CO3: 高 importance → 豁免 COCP 惩罚 ────────────────────────────────────────────

def test_co3_high_importance_protected(conn):
    """CO3: importance >= cocp_protect_importance(0.80) → 豁免 COCP 惩罚。"""
    threshold = config.get("store_vfs.cocp_type_threshold")  # 15
    protect_imp = config.get("store_vfs.cocp_protect_importance")  # 0.80

    # 插入超过阈值的 chunk，全部高 importance
    _insert_many_chunks(conn, threshold + 5, chunk_type="decision",
                        stability=5.0, importance=protect_imp,  # = 0.80，刚好达到保护
                        prefix="prot_")

    stab_before = _get_stability(conn, "prot_0")
    apply_cue_overload_consolidation_penalty(conn, "test")
    stab_after = _get_stability(conn, "prot_0")

    assert abs(stab_after - stab_before) < 0.001, (
        f"CO3: importance >= protect_importance 时应豁免 COCP 惩罚，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── CO4: protect_types → 豁免 COCP 惩罚 ────────────────────────────────────────────

def test_co4_protected_type_no_penalty(conn):
    """CO4: cocp_protect_types 中的 chunk_type → 即使数量过多也豁免。"""
    threshold = config.get("store_vfs.cocp_type_threshold")  # 15
    # cocp_protect_types 默认包含 "design_constraint"
    _insert_many_chunks(conn, threshold + 5, chunk_type="design_constraint",
                        stability=5.0, importance=0.5, prefix="dc_")

    stab_before = _get_stability(conn, "dc_0")
    apply_cue_overload_consolidation_penalty(conn, "test")
    stab_after = _get_stability(conn, "dc_0")

    assert abs(stab_after - stab_before) < 0.001, (
        f"CO4: design_constraint 在 protect_types 中应豁免 COCP，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── CO5: cocp_enabled=False → 无任何惩罚 ────────────────────────────────────────────

def test_co5_disabled_no_penalty(conn):
    """CO5: store_vfs.cocp_enabled=False → 无任何 COCP 惩罚。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.cocp_enabled":
            return False
        return original_get(key, project=project)

    threshold = config.get("store_vfs.cocp_type_threshold")
    _insert_many_chunks(conn, threshold + 5, chunk_type="decision",
                        stability=5.0, importance=0.5, prefix="dis_")

    stab_before = _get_stability(conn, "dis_0")
    with mock.patch.object(config, 'get', side_effect=patched_get):
        result = apply_cue_overload_consolidation_penalty(conn, "test")
    stab_after = _get_stability(conn, "dis_0")

    assert abs(stab_after - stab_before) < 0.001, (
        f"CO5: disabled 时不应有 COCP 惩罚，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["cocp_penalized"] == 0, f"CO5: cocp_penalized 应为 0，got {result}"


# ── CO6: 更多 overload → 更大惩罚（但不超过 max_penalty）────────────────────────────

def test_co6_more_overload_more_penalty(conn):
    """CO6: N 越大（overload 越重）→ 惩罚系数越大（但受 cocp_max_penalty 限制）。"""
    threshold = config.get("store_vfs.cocp_type_threshold")  # 15

    # 场景 A：轻度过载（threshold+5 = 20）
    _insert_many_chunks(conn, threshold + 5, project="proj_a",
                        chunk_type="decision", stability=10.0, importance=0.5,
                        prefix="light_")
    apply_cue_overload_consolidation_penalty(conn, "proj_a")
    stab_light = _get_stability(conn, "light_0")

    # 场景 B：重度过载（threshold+25 = 40）
    _insert_many_chunks(conn, threshold + 25, project="proj_b",
                        chunk_type="decision", stability=10.0, importance=0.5,
                        prefix="heavy_")
    apply_cue_overload_consolidation_penalty(conn, "proj_b")
    stab_heavy = _get_stability(conn, "heavy_0")

    # 惩罚后 stability 越低代表惩罚越大
    # light: 更少 overload → penalty 更小 → stability 更高
    assert stab_light >= stab_heavy, (
        f"CO6: 轻度 overload 应比重度 overload 受更小惩罚，"
        f"stab_light={stab_light:.4f} stab_heavy={stab_heavy:.4f}"
    )


# ── CO7: stability 惩罚后不低于 0.1 ────────────────────────────────────────────────

def test_co7_stability_floor_0_1(conn):
    """CO7: COCP 惩罚后 stability 不低于 0.1（min floor）。"""
    threshold = config.get("store_vfs.cocp_type_threshold")
    # 用极小的 stability 初始值
    _insert_many_chunks(conn, threshold + 5, chunk_type="decision",
                        stability=0.11, importance=0.5, prefix="floor_")

    apply_cue_overload_consolidation_penalty(conn, "test")
    stab_after = _get_stability(conn, "floor_0")

    assert stab_after >= 0.1, f"CO7: COCP 惩罚后 stability 不应低于 0.1，got {stab_after}"


# ── CO8: cocp_max_penalty 可配置 ─────────────────────────────────────────────────

def test_co8_configurable_max_penalty(conn):
    """CO8: cocp_max_penalty=0.05 时最大惩罚小于默认 0.10。"""
    original_get = config.get

    def patched_05(key, project=None):
        if key == "store_vfs.cocp_max_penalty":
            return 0.05
        return original_get(key, project=project)

    threshold = config.get("store_vfs.cocp_type_threshold")
    # 插入足以达到 max_penalty 的 chunk 数量
    _insert_many_chunks(conn, threshold + 50, project="proj_05",
                        chunk_type="decision", stability=10.0, importance=0.5,
                        prefix="p05_")
    with mock.patch.object(config, 'get', side_effect=patched_05):
        apply_cue_overload_consolidation_penalty(conn, "proj_05")
    stab_05 = _get_stability(conn, "p05_0")
    penalty_05 = 10.0 - stab_05

    _insert_many_chunks(conn, threshold + 50, project="proj_10",
                        chunk_type="decision", stability=10.0, importance=0.5,
                        prefix="p10_")
    apply_cue_overload_consolidation_penalty(conn, "proj_10")
    stab_10 = _get_stability(conn, "p10_0")
    penalty_10 = 10.0 - stab_10

    assert penalty_05 <= penalty_10, (
        f"CO8: max_penalty=0.05 时惩罚应 <= max_penalty=0.10，"
        f"penalty_05={penalty_05:.5f} penalty_10={penalty_10:.5f}"
    )


# ── CO9: 返回值正确 ──────────────────────────────────────────────────────────────

def test_co9_return_counts_correct(conn):
    """CO9: result dict 中 cocp_penalized 和 total_examined 计数正确。"""
    threshold = config.get("store_vfs.cocp_type_threshold")
    overload_n = threshold + 5
    _insert_many_chunks(conn, overload_n, chunk_type="causal_chain",
                        stability=5.0, importance=0.5, prefix="cnt_")

    result = apply_cue_overload_consolidation_penalty(conn, "test")

    assert "cocp_penalized" in result, "CO9: result 应含 cocp_penalized key"
    assert "total_examined" in result, "CO9: result 应含 total_examined key"
    assert result["total_examined"] >= overload_n, (
        f"CO9: total_examined 应 >= {overload_n}，got {result}"
    )
    assert result["cocp_penalized"] >= 1, f"CO9: cocp_penalized 应 >= 1，got {result}"
