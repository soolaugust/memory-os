"""
test_iter460_ssde.py — iter460: Sleep Spindle Density Effect 单元测试

覆盖：
  SD1: declarative 类型（decision）在 sleep 时 stability 增加（× ssde_declarative_multiplier=1.20）
  SD2: procedural 类型（procedure）在 sleep 时 stability 降低（× ssde_procedural_multiplier=0.85）
  SD3: 中性类型（conversation_summary）不受 SSDE 影响（× 1.0）
  SD4: ssde_enabled=False → 无差异化加成/减成
  SD5: importance < ssde_min_importance(0.45) → 不参与 SSDE
  SD6: declarative 类型的最终 stability > 中性类型 > procedural 类型（三级分层）
  SD7: stability 加成后不超过 365.0（cap 保护）
  SD8: stability 减成后不低于 0.1（floor 保护）
  SD9: 返回值正确（ssde_boosted, ssde_reduced, total_examined 计数）
  SD10: 多种 declarative 类型均触发（design_constraint/reasoning_chain/causal_chain）

认知科学依据：
  Stickgold (2005) "Sleep-dependent memory consolidation" (Nature 437) —
    NREM Stage 2 sleep spindle 密度与陈述性记忆巩固量正相关（r=0.71）。
  Gais et al. (2002) "Learning-dependent increases in sleep spindle density" —
    学习后睡眠 spindle 密度增加 +17%，与次日记忆保留率高度相关。
  Walker & Stickgold (2004) — 不同记忆类型有不同睡眠阶段偏好：
    陈述性 → NREM SWS + spindles；程序性 → REM。

OS 类比：Linux NUMA-aware writeback priority —
  data page（陈述性）→ pdflush 优先处理；
  journal page（程序性）→ jbd2 单独管理；
  spindle-preferred 类型在 sleep_consolidate 中获得更强优先级。
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
    apply_sleep_spindle_density_effect,
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


# ── SD1: declarative 类型 → stability 增加（× 1.20）──────────────────────────

def test_sd1_declarative_stability_boosted(conn):
    """SD1: decision（陈述性）chunk 在 SSDE 后 stability 增加（× 1.20）。"""
    _insert_chunk(conn, "ssde_dec", chunk_type="decision",
                  stability=5.0, importance=0.6)
    stab_before = _get_stability(conn, "ssde_dec")

    result = apply_sleep_spindle_density_effect(conn, "test")
    stab_after = _get_stability(conn, "ssde_dec")

    assert stab_after > stab_before, (
        f"SD1: declarative chunk 应在 SSDE 后 stability 增加，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["ssde_boosted"] >= 1, f"SD1: ssde_boosted 应 >= 1，got {result}"


# ── SD2: procedural 类型 → stability 降低（× 0.85）───────────────────────────

def test_sd2_procedural_stability_reduced(conn):
    """SD2: procedure（程序性）chunk 在 SSDE 后 stability 降低（× 0.85）。"""
    _insert_chunk(conn, "ssde_proc", chunk_type="procedure",
                  stability=5.0, importance=0.6)
    stab_before = _get_stability(conn, "ssde_proc")

    result = apply_sleep_spindle_density_effect(conn, "test")
    stab_after = _get_stability(conn, "ssde_proc")

    assert stab_after < stab_before, (
        f"SD2: procedural chunk 应在 SSDE 后 stability 降低，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["ssde_reduced"] >= 1, f"SD2: ssde_reduced 应 >= 1，got {result}"


# ── SD3: 中性类型不受 SSDE 影响 ──────────────────────────────────────────────

def test_sd3_neutral_type_unchanged(conn):
    """SD3: 中性类型（conversation_summary）不受 SSDE 差异化影响。"""
    _insert_chunk(conn, "ssde_neu", chunk_type="conversation_summary",
                  stability=5.0, importance=0.6)
    stab_before = _get_stability(conn, "ssde_neu")

    apply_sleep_spindle_density_effect(conn, "test")
    stab_after = _get_stability(conn, "ssde_neu")

    # 中性类型的 stability 不应被 SSDE 改变
    assert abs(stab_after - stab_before) < 0.01, (
        f"SD3: 中性类型不应受 SSDE 差异化影响，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── SD4: ssde_enabled=False → 无差异化 ────────────────────────────────────────

def test_sd4_disabled_no_effect(conn):
    """SD4: ssde_enabled=False → 无任何 SSDE 差异化加成/减成。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.ssde_enabled":
            return False
        return original_get(key, project=project)

    _insert_chunk(conn, "ssde_dis_dec", chunk_type="decision",
                  stability=5.0, importance=0.6)
    _insert_chunk(conn, "ssde_dis_proc", chunk_type="procedure",
                  stability=5.0, importance=0.6)
    stab_dec_before = _get_stability(conn, "ssde_dis_dec")
    stab_proc_before = _get_stability(conn, "ssde_dis_proc")

    with mock.patch.object(config, 'get', side_effect=patched_get):
        result = apply_sleep_spindle_density_effect(conn, "test")
    stab_dec_after = _get_stability(conn, "ssde_dis_dec")
    stab_proc_after = _get_stability(conn, "ssde_dis_proc")

    assert abs(stab_dec_after - stab_dec_before) < 0.01, (
        f"SD4: disabled 时 declarative 不应改变，"
        f"before={stab_dec_before:.4f} after={stab_dec_after:.4f}"
    )
    assert abs(stab_proc_after - stab_proc_before) < 0.01, (
        f"SD4: disabled 时 procedural 不应改变，"
        f"before={stab_proc_before:.4f} after={stab_proc_after:.4f}"
    )
    assert result["ssde_boosted"] == 0 and result["ssde_reduced"] == 0, (
        f"SD4: disabled 时 ssde_boosted 和 ssde_reduced 均应为 0，got {result}"
    )


# ── SD5: importance 不足 → 不参与 SSDE ───────────────────────────────────────

def test_sd5_low_importance_no_effect(conn):
    """SD5: importance < ssde_min_importance(0.45) → 不参与 SSDE。"""
    _insert_chunk(conn, "ssde_low_imp", chunk_type="decision",
                  stability=5.0, importance=0.30)  # < 0.45
    stab_before = _get_stability(conn, "ssde_low_imp")

    result = apply_sleep_spindle_density_effect(conn, "test")
    stab_after = _get_stability(conn, "ssde_low_imp")

    assert abs(stab_after - stab_before) < 0.01, (
        f"SD5: 低 importance 的 declarative chunk 不应受 SSDE 影响，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── SD6: 三级分层验证 ─────────────────────────────────────────────────────────

def test_sd6_three_tier_ordering(conn):
    """SD6: 相同初始条件下：declarative > neutral > procedural stability。"""
    _insert_chunk(conn, "ssde_t_dec", chunk_type="decision",
                  stability=5.0, importance=0.6)
    _insert_chunk(conn, "ssde_t_neu", chunk_type="conversation_summary",
                  stability=5.0, importance=0.6)
    _insert_chunk(conn, "ssde_t_proc", chunk_type="procedure",
                  stability=5.0, importance=0.6)

    apply_sleep_spindle_density_effect(conn, "test")

    stab_dec = _get_stability(conn, "ssde_t_dec")
    stab_neu = _get_stability(conn, "ssde_t_neu")
    stab_proc = _get_stability(conn, "ssde_t_proc")

    assert stab_dec >= stab_neu, (
        f"SD6: declarative stability 应 >= neutral，"
        f"dec={stab_dec:.4f} neu={stab_neu:.4f}"
    )
    assert stab_neu >= stab_proc, (
        f"SD6: neutral stability 应 >= procedural，"
        f"neu={stab_neu:.4f} proc={stab_proc:.4f}"
    )


# ── SD7: stability 上限 365.0 ─────────────────────────────────────────────────

def test_sd7_stability_cap_365(conn):
    """SD7: SSDE boost 后 stability 不超过 365.0。"""
    _insert_chunk(conn, "ssde_cap", chunk_type="decision",
                  stability=364.0, importance=0.8)
    apply_sleep_spindle_density_effect(conn, "test")
    stab_after = _get_stability(conn, "ssde_cap")

    assert stab_after <= 365.0, f"SD7: stability 不应超过 365.0，got {stab_after}"


# ── SD8: stability 减成后不低于 0.1 ──────────────────────────────────────────

def test_sd8_stability_floor_0_1(conn):
    """SD8: SSDE 对 procedural 的减成后 stability 不低于 0.1。"""
    _insert_chunk(conn, "ssde_floor", chunk_type="procedure",
                  stability=0.12, importance=0.6)  # 0.12 × 0.85 = 0.102 > 0.1
    apply_sleep_spindle_density_effect(conn, "test")
    stab_after = _get_stability(conn, "ssde_floor")

    assert stab_after >= 0.1, (
        f"SD8: SSDE 减成后 stability 不应低于 0.1，got {stab_after}"
    )


# ── SD9: 返回值正确 ────────────────────────────────────────────────────────────

def test_sd9_return_counts_correct(conn):
    """SD9: result dict 中 ssde_boosted、ssde_reduced、total_examined 计数正确。"""
    _insert_chunk(conn, "ssde_r1", chunk_type="decision",
                  stability=5.0, importance=0.6)
    _insert_chunk(conn, "ssde_r2", chunk_type="reasoning_chain",
                  stability=5.0, importance=0.6)
    _insert_chunk(conn, "ssde_r3", chunk_type="procedure",
                  stability=5.0, importance=0.6)
    _insert_chunk(conn, "ssde_r4", chunk_type="task_state",
                  stability=5.0, importance=0.6)

    result = apply_sleep_spindle_density_effect(conn, "test")

    assert "ssde_boosted" in result, "SD9: result 应含 ssde_boosted key"
    assert "ssde_reduced" in result, "SD9: result 应含 ssde_reduced key"
    assert "total_examined" in result, "SD9: result 应含 total_examined key"
    assert result["ssde_boosted"] >= 2, (
        f"SD9: 至少 2 个 declarative chunk 应被 boost，got {result}"
    )
    assert result["ssde_reduced"] >= 2, (
        f"SD9: 至少 2 个 procedural chunk 应被 reduce，got {result}"
    )


# ── SD10: 多种 declarative 类型均触发 ────────────────────────────────────────

def test_sd10_all_declarative_types_boosted(conn):
    """SD10: decision/design_constraint/reasoning_chain/causal_chain 均触发 SSDE 加成。"""
    declarative_types = [
        "decision", "design_constraint", "reasoning_chain",
        "quantitative_evidence", "causal_chain"
    ]
    for i, ctype in enumerate(declarative_types):
        _insert_chunk(conn, f"ssde_d{i}", chunk_type=ctype,
                      stability=5.0, importance=0.6)

    apply_sleep_spindle_density_effect(conn, "test")

    for i, ctype in enumerate(declarative_types):
        stab = _get_stability(conn, f"ssde_d{i}")
        assert stab > 5.0, (
            f"SD10: {ctype} 应触发 SSDE boost，expected > 5.0 got {stab:.4f}"
        )
