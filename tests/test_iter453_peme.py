"""
test_iter453_peme.py — iter453: Prediction Error Memory Enhancement 单元测试

覆盖：
  PE1: access_count_before <= peme_max_access(5) + retrievability < peme_low_retrievability(0.50)
       + importance >= peme_min_importance(0.45) → stability 增加（意外命中触发多巴胺强化）
  PE2: access_count_before > peme_max_access → 无 PEME 加成（高历史召回 = 非意外）
  PE3: retrievability >= peme_low_retrievability → 无 PEME 加成（记忆未遗忘 = 非低预期）
  PE4: peme_enabled=False → 无任何加成
  PE5: importance < peme_min_importance → 不参与 PEME
  PE6: 更低 retrievability + 更低 access_count → 更大 surprise_score → 更大加成（正比关系）
  PE7: stability 加成后不超过 365.0（cap 保护）
  PE8: peme_scale 可配置（更大 scale → 更大加成）
  PE9: 返回值正确（更新后的 stability）
  PE10: access_count_before=0（完全新 chunk）+ 低 retrievability → 最大 surprise_score
  PE11: access_count_before == peme_max_access（边界条件） → 仍可触发（a_surprise > 0）
  PE12: update_accessed() 集成测试：低访问 + 低 retrievability chunk 被检索后 stability 增加

认知科学依据：
  Rescorla & Wagner (1972) "A theory of Pavlovian conditioning" —
    强化量与预测误差正比：λ > V（结果优于预期）→ 最大强化。
  Schultz, Dayan & Montague (1997) Science "A Neural Substrate of Prediction and Reward" —
    VTA 多巴胺神经元精确编码 TD 预测误差：意外奖励 → dopamine burst → 强化关联。
  Lisman & Grace (2005) Neuron "The hippocampal-VTA loop" —
    海马发现新奇/意外信号 → 激活 VTA 多巴胺 → 强化当前检索路径的海马-新皮层突触（LTP）。

OS 类比：CPU branch predictor misprediction → forced L1 cache line promotion —
  预测 not-taken（低历史相关）但实际 taken（被检索到）→ pipeline flush +
  强制将目标路径 cache line 提升到 L1；类比：低预期 chunk 被意外命中 → stability 强制提升。
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
    apply_prediction_error_enhancement,
    update_accessed,
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
                  retrievability=0.3, access_count=0):
    """Insert a chunk with controlled state for PEME testing."""
    now_iso = _utcnow().isoformat()
    # Set last_accessed to 72h ago so retrievability will be low when computed
    last_acc = (_utcnow() - datetime.timedelta(hours=72)).isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count,
            encode_context)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (cid, project, "decision", f"content {cid}", f"summary {cid}",
         importance, stability, now_iso, now_iso, retrievability,
         last_acc, access_count, "kernel_mm")
    )
    conn.commit()


def _get_stability(conn, cid: str) -> float:
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


# ── PE1: 意外命中条件全满足 → stability 增加 ──────────────────────────────────────────

def test_pe1_surprise_hit_boosts_stability(conn):
    """PE1: access_count_before<=5, retrievability<0.50, importance>=0.45 → stability 增加。"""
    _insert_chunk(conn, "peme_hit", importance=0.6, stability=5.0,
                  retrievability=0.2, access_count=1)

    stab_before = _get_stability(conn, "peme_hit")
    new_stab = apply_prediction_error_enhancement(
        conn, "peme_hit", "test",
        access_count_before=1,
        retrievability=0.2,
        importance=0.6,
        stability=5.0,
    )
    stab_after = _get_stability(conn, "peme_hit")

    assert stab_after > stab_before, (
        f"PE1: 意外命中应触发 PEME 加成，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert new_stab > stab_before, f"PE1: 返回值应大于原始 stability，got {new_stab:.4f}"


# ── PE2: access_count 过高 → 无 PEME 加成 ────────────────────────────────────────────

def test_pe2_high_access_count_no_boost(conn):
    """PE2: access_count_before > peme_max_access(5) → 高历史召回 = 非意外 → 无加成。"""
    peme_max = config.get("store_vfs.peme_max_access")  # 5

    _insert_chunk(conn, "peme_high_acc", importance=0.6, stability=5.0,
                  retrievability=0.1, access_count=peme_max + 1)

    stab_before = _get_stability(conn, "peme_high_acc")
    new_stab = apply_prediction_error_enhancement(
        conn, "peme_high_acc", "test",
        access_count_before=peme_max + 1,
        retrievability=0.1,
        importance=0.6,
        stability=5.0,
    )
    stab_after = _get_stability(conn, "peme_high_acc")

    assert abs(stab_after - stab_before) < 0.001, (
        f"PE2: access_count 过高时不应触发 PEME，before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── PE3: retrievability 过高 → 无 PEME 加成 ──────────────────────────────────────────

def test_pe3_high_retrievability_no_boost(conn):
    """PE3: retrievability >= peme_low_retrievability(0.50) → 记忆未遗忘 → 无加成。"""
    peme_low_ret = config.get("store_vfs.peme_low_retrievability")  # 0.50

    _insert_chunk(conn, "peme_high_ret", importance=0.6, stability=5.0,
                  retrievability=peme_low_ret + 0.1, access_count=1)

    stab_before = _get_stability(conn, "peme_high_ret")
    new_stab = apply_prediction_error_enhancement(
        conn, "peme_high_ret", "test",
        access_count_before=1,
        retrievability=peme_low_ret + 0.1,
        importance=0.6,
        stability=5.0,
    )
    stab_after = _get_stability(conn, "peme_high_ret")

    assert abs(stab_after - stab_before) < 0.001, (
        f"PE3: retrievability 过高时不应触发 PEME，before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── PE4: peme_enabled=False → 无加成 ─────────────────────────────────────────────────

def test_pe4_disabled_no_boost(conn):
    """PE4: store_vfs.peme_enabled=False → 无任何 PEME 加成。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.peme_enabled":
            return False
        return original_get(key, project=project)

    _insert_chunk(conn, "peme_disabled", importance=0.6, stability=5.0,
                  retrievability=0.1, access_count=1)

    stab_before = _get_stability(conn, "peme_disabled")
    with mock.patch.object(config, 'get', side_effect=patched_get):
        new_stab = apply_prediction_error_enhancement(
            conn, "peme_disabled", "test",
            access_count_before=1,
            retrievability=0.1,
            importance=0.6,
            stability=5.0,
        )
    stab_after = _get_stability(conn, "peme_disabled")

    assert abs(stab_after - stab_before) < 0.001, (
        f"PE4: disabled 时不应有 PEME 加成，before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── PE5: importance 不足 → 不参与 PEME ────────────────────────────────────────────────

def test_pe5_low_importance_excluded(conn):
    """PE5: importance < peme_min_importance(0.45) → 不参与 PEME。"""
    _insert_chunk(conn, "peme_low_imp", importance=0.20, stability=5.0,
                  retrievability=0.1, access_count=1)

    stab_before = _get_stability(conn, "peme_low_imp")
    new_stab = apply_prediction_error_enhancement(
        conn, "peme_low_imp", "test",
        access_count_before=1,
        retrievability=0.1,
        importance=0.20,
        stability=5.0,
    )
    stab_after = _get_stability(conn, "peme_low_imp")

    assert abs(stab_after - stab_before) < 0.001, (
        f"PE5: 低 importance 不应触发 PEME，before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── PE6: 更低 ret + 更低 acc → 更大 surprise_score → 更大加成 ────────────────────────

def test_pe6_more_surprise_more_boost(conn):
    """PE6: 更低 retrievability + 更低 access_count → 更大 surprise_score → 更大加成。"""
    # 场景 A：低惊喜（acc=4, ret=0.4 → surprise = 0.6 × 0.2 = 0.12）
    _insert_chunk(conn, "low_surprise", importance=0.6, stability=5.0,
                  retrievability=0.4, access_count=4)
    apply_prediction_error_enhancement(
        conn, "low_surprise", "test",
        access_count_before=4,
        retrievability=0.4,
        importance=0.6,
        stability=5.0,
    )
    stab_low = _get_stability(conn, "low_surprise")

    # 场景 B：高惊喜（acc=0, ret=0.05 → surprise = 0.95 × 1.0 = 0.95）
    _insert_chunk(conn, "high_surprise", importance=0.6, stability=5.0,
                  retrievability=0.05, access_count=0)
    apply_prediction_error_enhancement(
        conn, "high_surprise", "test",
        access_count_before=0,
        retrievability=0.05,
        importance=0.6,
        stability=5.0,
    )
    stab_high = _get_stability(conn, "high_surprise")

    assert stab_high > stab_low, (
        f"PE6: 更高惊喜度应获得更大加成，stab_low={stab_low:.4f} stab_high={stab_high:.4f}"
    )


# ── PE7: stability 上限 365.0 ─────────────────────────────────────────────────────────

def test_pe7_stability_cap_365(conn):
    """PE7: PEME boost 后 stability 不超过 365.0。"""
    _insert_chunk(conn, "peme_cap", importance=0.8, stability=364.9,
                  retrievability=0.05, access_count=0)

    apply_prediction_error_enhancement(
        conn, "peme_cap", "test",
        access_count_before=0,
        retrievability=0.05,
        importance=0.8,
        stability=364.9,
    )
    stab_after = _get_stability(conn, "peme_cap")

    assert stab_after <= 365.0, f"PE7: stability 不应超过 365.0，got {stab_after}"


# ── PE8: peme_scale 可配置 ────────────────────────────────────────────────────────────

def test_pe8_configurable_scale(conn):
    """PE8: peme_scale=0.30 时加成比默认 0.15 更大。"""
    original_get = config.get

    def patched_30(key, project=None):
        if key == "store_vfs.peme_scale":
            return 0.30
        return original_get(key, project=project)

    _insert_chunk(conn, "scale_chunk", importance=0.6, stability=5.0,
                  retrievability=0.1, access_count=1)
    stab_before = _get_stability(conn, "scale_chunk")

    with mock.patch.object(config, 'get', side_effect=patched_30):
        apply_prediction_error_enhancement(
            conn, "scale_chunk", "test",
            access_count_before=1,
            retrievability=0.1,
            importance=0.6,
            stability=5.0,
        )
    stab_after_30 = _get_stability(conn, "scale_chunk")
    delta_30 = stab_after_30 - stab_before

    # 重置
    conn.execute("UPDATE memory_chunks SET stability=5.0 WHERE id='scale_chunk'")
    conn.commit()

    apply_prediction_error_enhancement(
        conn, "scale_chunk", "test",
        access_count_before=1,
        retrievability=0.1,
        importance=0.6,
        stability=5.0,
    )
    stab_after_default = _get_stability(conn, "scale_chunk")
    delta_default = stab_after_default - stab_before

    assert delta_30 > delta_default, (
        f"PE8: peme_scale=0.30 加成应大于默认 0.15，"
        f"delta_30={delta_30:.5f} delta_default={delta_default:.5f}"
    )


# ── PE9: 返回值正确 ────────────────────────────────────────────────────────────────────

def test_pe9_return_value_correct(conn):
    """PE9: apply_prediction_error_enhancement 返回更新后的 stability。"""
    _insert_chunk(conn, "peme_ret_val", importance=0.6, stability=5.0,
                  retrievability=0.1, access_count=1)

    returned = apply_prediction_error_enhancement(
        conn, "peme_ret_val", "test",
        access_count_before=1,
        retrievability=0.1,
        importance=0.6,
        stability=5.0,
    )
    stab_db = _get_stability(conn, "peme_ret_val")

    assert abs(returned - stab_db) < 0.001, (
        f"PE9: 返回值应等于数据库中的新 stability，returned={returned:.4f} db={stab_db:.4f}"
    )
    assert returned > 5.0, f"PE9: 触发了 PEME 应返回增加后的值，got {returned:.4f}"


# ── PE10: access_count=0 → 最大 a_surprise ───────────────────────────────────────────

def test_pe10_zero_access_count_max_surprise(conn):
    """PE10: access_count_before=0（完全新 chunk） + 低 retrievability → 最大 a_surprise=1.0。"""
    _insert_chunk(conn, "zero_acc", importance=0.6, stability=5.0,
                  retrievability=0.05, access_count=0)

    # 计算预期的加成
    # a_surprise = 1.0 - 0/5 = 1.0
    # r_surprise = 1.0 - 0.05 = 0.95
    # surprise_score = 1.0 * 0.95 = 0.95
    # bonus = 0.95 * 0.15 = 0.1425
    # new_stab = 5.0 * 1.1425 = 5.7125

    new_stab = apply_prediction_error_enhancement(
        conn, "zero_acc", "test",
        access_count_before=0,
        retrievability=0.05,
        importance=0.6,
        stability=5.0,
    )
    stab_db = _get_stability(conn, "zero_acc")

    assert stab_db > 5.5, (
        f"PE10: access_count=0 + ret=0.05 应获得接近最大加成，got {stab_db:.4f}"
    )
    assert abs(stab_db - 5.0 * (1.0 + 0.95 * 0.15)) < 0.02, (
        f"PE10: 加成计算应接近理论值 5.7125，got {stab_db:.4f}"
    )


# ── PE11: access_count == peme_max_access（边界条件）───────────────────────────────────

def test_pe11_access_count_equals_max_boundary(conn):
    """PE11: access_count_before == peme_max_access(5) → a_surprise = 1 - 5/5 = 0 → 无加成。"""
    peme_max = config.get("store_vfs.peme_max_access")  # 5

    _insert_chunk(conn, "boundary_acc", importance=0.6, stability=5.0,
                  retrievability=0.1, access_count=peme_max)

    stab_before = _get_stability(conn, "boundary_acc")
    new_stab = apply_prediction_error_enhancement(
        conn, "boundary_acc", "test",
        access_count_before=peme_max,
        retrievability=0.1,
        importance=0.6,
        stability=5.0,
    )
    stab_after = _get_stability(conn, "boundary_acc")

    # access_count == peme_max → a_surprise = 1 - max_access/max_access = 0 → surprise_score = 0
    # → 无加成
    assert abs(stab_after - stab_before) < 0.001, (
        f"PE11: access_count={peme_max} == peme_max_access={peme_max}时 a_surprise=0 → 无加成，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── PE12: update_accessed() 集成测试 ─────────────────────────────────────────────────

def test_pe12_update_accessed_integration(conn):
    """PE12: update_accessed() 对低访问 + 低 retrievability chunk 检索后 stability 增加。"""
    # 插入一个低访问、低 retrievability 的 chunk
    _insert_chunk(conn, "peme_integ", importance=0.6, stability=5.0,
                  retrievability=0.1,   # < peme_low_retrievability(0.50)
                  access_count=1)       # <= peme_max_access(5)

    stab_before = _get_stability(conn, "peme_integ")

    # 触发 update_accessed → 内部调用 apply_prediction_error_enhancement
    update_accessed(conn, ["peme_integ"])

    stab_after = _get_stability(conn, "peme_integ")

    assert stab_after > stab_before, (
        f"PE12: update_accessed 对低预期 chunk 应触发 PEME 加成，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )
