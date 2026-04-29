"""
test_iter455_gsie.py — iter455: Generation-Spacing Interaction Effect 单元测试

覆盖：
  GS1: high effort + high streak → stability 增加（双因子交互触发）
  GS2: low effort（retrievability ≈ 1.0）→ 无 GSIE 加成
  GS3: low streak（spaced_access_count < gsie_min_streak=2）→ 无 GSIE 加成
  GS4: gsie_enabled=False → 无任何加成
  GS5: importance < gsie_min_importance(0.40) → 不参与 GSIE
  GS6: 更大 effort × 更大 streak → 更大 interaction_score → 更大加成（乘法正比）
  GS7: stability 加成后不超过 365.0（cap 保护）
  GS8: gsie_scale 可配置（更大 scale → 更大加成）
  GS9: 返回值正确（gsie_boosted, total_examined 计数）
  GS10: effort_score 边界——gap=0（刚访问）→ effort≈0 → 无加成
  GS11: streak_factor 正确计算（spaced_access_count=gsie_ref_streak → streak_factor=1.0）
  GS12: update_accessed() 集成测试：高 gap + 高 streak chunk 被检索后 stability 增加

认知科学依据：
  Pyc & Rawson (2009) "Testing the retrieval effort hypothesis: Does greater difficulty
    correctly recalling information lead to higher levels of memory?" (JML 60:437-447) —
    检索巩固效益 = 检索难度 × 间隔成功历史的乘积（双因子乘法交互）。
  Carrier & Pashler (1992) — 突触标记需要先前 LTP 历史（streak）+ 当前去极化充分（effort）。
  Frey & Morris (1997) synaptic tagging hypothesis — 两者缺一不可。

OS 类比：Linux ARC ghost list + frequency-weighted promotion —
  ghost list page re-fault（检索努力）× T2 history weight（间隔历史）= 晋升力度。
"""
import sys
import math
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
    apply_generation_spacing_interaction_effect,
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
                  chunk_type="decision", retrievability=0.5, access_count=3,
                  spaced_access_count=3, last_accessed_hours_ago=48.0):
    """Insert a chunk with controlled state for GSIE testing."""
    now_iso = _utcnow().isoformat()
    last_acc = (_utcnow() - datetime.timedelta(hours=last_accessed_hours_ago)).isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count,
            spaced_access_count, encode_context)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (cid, project, chunk_type, f"content {cid}", f"summary {cid}",
         importance, stability, now_iso, now_iso, retrievability,
         last_acc, access_count, spaced_access_count, "kernel_mm")
    )
    conn.commit()


def _get_stability(conn, cid: str) -> float:
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


def _compute_r_at_recall(pre_stability: float, gap_hours: float) -> float:
    """Helper: compute R_at_recall = exp(-gap_hours / (stability * 24))."""
    return math.exp(-gap_hours / max(0.01, pre_stability * 24.0))


# ── GS1: 高 effort + 高 streak → stability 增加 ────────────────────────────────────────

def test_gs1_high_effort_high_streak_boosts_stability(conn):
    """GS1: effort 高（gap=48h，stability=5 → R≈0.66，effort≈0.34）+ streak≥2 → stability 增加。"""
    # gap=48h, stability=5 → R = exp(-48/(5*24)) = exp(-0.4) ≈ 0.67 → effort ≈ 0.33
    # spaced_access_count=4 >= gsie_min_streak(2) → streak_factor = 4/6 = 0.67
    last_acc_iso = (_utcnow() - datetime.timedelta(hours=48)).isoformat()
    _insert_chunk(conn, "gsie_hit", project="test",
                  stability=5.0, importance=0.6,
                  spaced_access_count=4, last_accessed_hours_ago=48.0)

    pre_stability_map = {"gsie_hit": 5.0}
    pre_spaced_access_map = {"gsie_hit": 4}
    pre_last_accessed_map = {"gsie_hit": last_acc_iso}
    now_iso = _utcnow().isoformat()

    stab_before = _get_stability(conn, "gsie_hit")
    result = apply_generation_spacing_interaction_effect(
        conn, ["gsie_hit"], "test",
        pre_stability_map, pre_spaced_access_map, pre_last_accessed_map, now_iso
    )
    stab_after = _get_stability(conn, "gsie_hit")

    assert stab_after > stab_before, (
        f"GS1: 高 effort + 高 streak 应触发 GSIE，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["gsie_boosted"] >= 1, f"GS1: gsie_boosted 应 >= 1，got {result}"


# ── GS2: 低 effort（gap≈0）→ 无 GSIE 加成 ───────────────────────────────────────────────

def test_gs2_low_effort_no_boost(conn):
    """GS2: gap≈0（刚访问，R≈1.0）→ effort_score≈0 → 无 GSIE 加成。"""
    # gap=0.01h, stability=5 → R = exp(-0.01/(5*24)) ≈ 1.0 → effort ≈ 0 < gsie_min_effort(0.10)
    last_acc_iso = (_utcnow() - datetime.timedelta(minutes=1)).isoformat()
    _insert_chunk(conn, "low_effort", project="test",
                  stability=5.0, importance=0.6,
                  spaced_access_count=5, last_accessed_hours_ago=0.016)

    pre_stability_map = {"low_effort": 5.0}
    pre_spaced_access_map = {"low_effort": 5}
    pre_last_accessed_map = {"low_effort": last_acc_iso}
    now_iso = _utcnow().isoformat()

    stab_before = _get_stability(conn, "low_effort")
    result = apply_generation_spacing_interaction_effect(
        conn, ["low_effort"], "test",
        pre_stability_map, pre_spaced_access_map, pre_last_accessed_map, now_iso
    )
    stab_after = _get_stability(conn, "low_effort")

    assert abs(stab_after - stab_before) < 0.001, (
        f"GS2: 低 effort 不应触发 GSIE，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["gsie_boosted"] == 0, f"GS2: gsie_boosted 应为 0，got {result}"


# ── GS3: 低 streak → 无 GSIE 加成 ───────────────────────────────────────────────────────

def test_gs3_low_streak_no_boost(conn):
    """GS3: spaced_access_count=1 < gsie_min_streak(2) → 无 GSIE 加成。"""
    gsie_min_streak = config.get("store_vfs.gsie_min_streak")  # 2

    last_acc_iso = (_utcnow() - datetime.timedelta(hours=48)).isoformat()
    _insert_chunk(conn, "low_streak", project="test",
                  stability=5.0, importance=0.6,
                  spaced_access_count=gsie_min_streak - 1,  # = 1 < min
                  last_accessed_hours_ago=48.0)

    pre_stability_map = {"low_streak": 5.0}
    pre_spaced_access_map = {"low_streak": gsie_min_streak - 1}
    pre_last_accessed_map = {"low_streak": last_acc_iso}
    now_iso = _utcnow().isoformat()

    stab_before = _get_stability(conn, "low_streak")
    result = apply_generation_spacing_interaction_effect(
        conn, ["low_streak"], "test",
        pre_stability_map, pre_spaced_access_map, pre_last_accessed_map, now_iso
    )
    stab_after = _get_stability(conn, "low_streak")

    assert abs(stab_after - stab_before) < 0.001, (
        f"GS3: 低 streak 不应触发 GSIE，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["gsie_boosted"] == 0, f"GS3: gsie_boosted 应为 0，got {result}"


# ── GS4: gsie_enabled=False → 无加成 ──────────────────────────────────────────────────

def test_gs4_disabled_no_boost(conn):
    """GS4: store_vfs.gsie_enabled=False → 无任何 GSIE 加成。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.gsie_enabled":
            return False
        return original_get(key, project=project)

    last_acc_iso = (_utcnow() - datetime.timedelta(hours=48)).isoformat()
    _insert_chunk(conn, "gsie_disabled", project="test",
                  stability=5.0, importance=0.6,
                  spaced_access_count=5, last_accessed_hours_ago=48.0)

    pre_stability_map = {"gsie_disabled": 5.0}
    pre_spaced_access_map = {"gsie_disabled": 5}
    pre_last_accessed_map = {"gsie_disabled": last_acc_iso}
    now_iso = _utcnow().isoformat()

    stab_before = _get_stability(conn, "gsie_disabled")
    with mock.patch.object(config, 'get', side_effect=patched_get):
        result = apply_generation_spacing_interaction_effect(
            conn, ["gsie_disabled"], "test",
            pre_stability_map, pre_spaced_access_map, pre_last_accessed_map, now_iso
        )
    stab_after = _get_stability(conn, "gsie_disabled")

    assert abs(stab_after - stab_before) < 0.001, (
        f"GS4: disabled 时不应有 GSIE 加成，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["gsie_boosted"] == 0, f"GS4: gsie_boosted 应为 0，got {result}"


# ── GS5: importance 不足 → 不参与 GSIE ────────────────────────────────────────────────

def test_gs5_low_importance_excluded(conn):
    """GS5: importance < gsie_min_importance(0.40) → 不参与 GSIE。"""
    last_acc_iso = (_utcnow() - datetime.timedelta(hours=48)).isoformat()
    _insert_chunk(conn, "low_imp", project="test",
                  stability=5.0, importance=0.20,  # < 0.40
                  spaced_access_count=5, last_accessed_hours_ago=48.0)

    pre_stability_map = {"low_imp": 5.0}
    pre_spaced_access_map = {"low_imp": 5}
    pre_last_accessed_map = {"low_imp": last_acc_iso}
    now_iso = _utcnow().isoformat()

    stab_before = _get_stability(conn, "low_imp")
    result = apply_generation_spacing_interaction_effect(
        conn, ["low_imp"], "test",
        pre_stability_map, pre_spaced_access_map, pre_last_accessed_map, now_iso
    )
    stab_after = _get_stability(conn, "low_imp")

    assert abs(stab_after - stab_before) < 0.001, (
        f"GS5: 低 importance 不应触发 GSIE，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["gsie_boosted"] == 0, f"GS5: gsie_boosted 应为 0，got {result}"


# ── GS6: 更大 effort × 更大 streak → 更大加成 ────────────────────────────────────────

def test_gs6_multiplicative_interaction(conn):
    """GS6: 更大 effort × 更大 streak → 更大 interaction_score → 更大加成（乘法正比）。"""
    # 场景 A：中等 effort（gap=24h, stab=5 → R≈0.82, effort≈0.18）+ 小 streak(2)
    last_acc_a = (_utcnow() - datetime.timedelta(hours=24)).isoformat()
    _insert_chunk(conn, "low_inter", project="proj_a",
                  stability=5.0, importance=0.6,
                  spaced_access_count=2, last_accessed_hours_ago=24.0)
    result_a = apply_generation_spacing_interaction_effect(
        conn, ["low_inter"], "proj_a",
        {"low_inter": 5.0}, {"low_inter": 2}, {"low_inter": last_acc_a},
        _utcnow().isoformat()
    )
    stab_a = _get_stability(conn, "low_inter")

    # 场景 B：高 effort（gap=168h, stab=5 → R≈0.25, effort≈0.75）+ 大 streak(6)
    last_acc_b = (_utcnow() - datetime.timedelta(hours=168)).isoformat()
    _insert_chunk(conn, "high_inter", project="proj_b",
                  stability=5.0, importance=0.6,
                  spaced_access_count=6, last_accessed_hours_ago=168.0)
    result_b = apply_generation_spacing_interaction_effect(
        conn, ["high_inter"], "proj_b",
        {"high_inter": 5.0}, {"high_inter": 6}, {"high_inter": last_acc_b},
        _utcnow().isoformat()
    )
    stab_b = _get_stability(conn, "high_inter")

    assert stab_b > stab_a, (
        f"GS6: 更大 effort×streak 应获得更大加成，stab_a={stab_a:.4f} stab_b={stab_b:.4f}"
    )


# ── GS7: stability 上限 365.0 ────────────────────────────────────────────────────────

def test_gs7_stability_cap_365(conn):
    """GS7: GSIE boost 后 stability 不超过 365.0。"""
    last_acc_iso = (_utcnow() - datetime.timedelta(hours=200)).isoformat()
    _insert_chunk(conn, "gsie_cap", project="test",
                  stability=364.9, importance=0.8,
                  spaced_access_count=10, last_accessed_hours_ago=200.0)

    pre_stability_map = {"gsie_cap": 364.9}
    pre_spaced_access_map = {"gsie_cap": 10}
    pre_last_accessed_map = {"gsie_cap": last_acc_iso}
    now_iso = _utcnow().isoformat()

    apply_generation_spacing_interaction_effect(
        conn, ["gsie_cap"], "test",
        pre_stability_map, pre_spaced_access_map, pre_last_accessed_map, now_iso
    )
    stab_after = _get_stability(conn, "gsie_cap")

    assert stab_after <= 365.0, f"GS7: stability 不应超过 365.0，got {stab_after}"


# ── GS8: gsie_scale 可配置 ──────────────────────────────────────────────────────────

def test_gs8_configurable_scale(conn):
    """GS8: gsie_scale=0.30 时加成比默认 0.12 更大。"""
    original_get = config.get

    def patched_30(key, project=None):
        if key == "store_vfs.gsie_scale":
            return 0.30
        return original_get(key, project=project)

    last_acc_iso = (_utcnow() - datetime.timedelta(hours=72)).isoformat()
    _insert_chunk(conn, "scale_gsie", project="proj_scale",
                  stability=5.0, importance=0.6,
                  spaced_access_count=5, last_accessed_hours_ago=72.0)

    pre_m = {"scale_gsie": 5.0}
    pre_s = {"scale_gsie": 5}
    pre_a = {"scale_gsie": last_acc_iso}
    now_iso = _utcnow().isoformat()

    stab_before = _get_stability(conn, "scale_gsie")
    with mock.patch.object(config, 'get', side_effect=patched_30):
        apply_generation_spacing_interaction_effect(
            conn, ["scale_gsie"], "proj_scale",
            pre_m, pre_s, pre_a, now_iso
        )
    stab_after_30 = _get_stability(conn, "scale_gsie")
    delta_30 = stab_after_30 - stab_before

    # 重置
    conn.execute("UPDATE memory_chunks SET stability=5.0 WHERE id='scale_gsie'")
    conn.commit()

    apply_generation_spacing_interaction_effect(
        conn, ["scale_gsie"], "proj_scale",
        pre_m, pre_s, pre_a, now_iso
    )
    stab_after_default = _get_stability(conn, "scale_gsie")
    delta_default = stab_after_default - 5.0

    assert delta_30 > delta_default, (
        f"GS8: gsie_scale=0.30 加成应大于默认 0.12，"
        f"delta_30={delta_30:.5f} delta_default={delta_default:.5f}"
    )


# ── GS9: 返回值正确 ─────────────────────────────────────────────────────────────────

def test_gs9_return_counts_correct(conn):
    """GS9: result dict 中 gsie_boosted 和 total_examined 计数正确。"""
    # 3 个满足条件的 chunk（高 effort + 高 streak + 高 importance）
    for i in range(3):
        last_acc = (_utcnow() - datetime.timedelta(hours=72)).isoformat()
        _insert_chunk(conn, f"ret_{i}", project="test",
                      stability=5.0, importance=0.6,
                      spaced_access_count=4, last_accessed_hours_ago=72.0)

    # 1 个 importance 不足（应被 examined 但不 boosted）
    _insert_chunk(conn, "ret_low_imp", project="test",
                  stability=5.0, importance=0.10,
                  spaced_access_count=5, last_accessed_hours_ago=72.0)

    pre_stability_map = {f"ret_{i}": 5.0 for i in range(3)}
    pre_stability_map["ret_low_imp"] = 5.0
    pre_spaced_access_map = {f"ret_{i}": 4 for i in range(3)}
    pre_spaced_access_map["ret_low_imp"] = 5
    last_acc_iso = (_utcnow() - datetime.timedelta(hours=72)).isoformat()
    pre_last_accessed_map = {f"ret_{i}": last_acc_iso for i in range(3)}
    pre_last_accessed_map["ret_low_imp"] = last_acc_iso
    now_iso = _utcnow().isoformat()

    all_ids = [f"ret_{i}" for i in range(3)] + ["ret_low_imp"]
    result = apply_generation_spacing_interaction_effect(
        conn, all_ids, "test",
        pre_stability_map, pre_spaced_access_map, pre_last_accessed_map, now_iso
    )

    assert "gsie_boosted" in result, "GS9: result 应含 gsie_boosted key"
    assert "total_examined" in result, "GS9: result 应含 total_examined key"
    assert result["gsie_boosted"] >= 3, f"GS9: gsie_boosted 应 >= 3，got {result}"
    assert result["total_examined"] >= 4, f"GS9: total_examined 应 >= 4，got {result}"


# ── GS10: effort_score 边界——gap=0 → effort≈0 → 无加成 ──────────────────────────────

def test_gs10_zero_gap_no_boost(conn):
    """GS10: gap=0（pre_last_accessed == now）→ R≈1.0 → effort≈0 < gsie_min_effort → 无加成。"""
    now_iso = _utcnow().isoformat()
    _insert_chunk(conn, "zero_gap", project="test",
                  stability=5.0, importance=0.6,
                  spaced_access_count=6, last_accessed_hours_ago=0.0)

    pre_stability_map = {"zero_gap": 5.0}
    pre_spaced_access_map = {"zero_gap": 6}
    # pre_last_accessed_map == now → gap = 0
    pre_last_accessed_map = {"zero_gap": now_iso}

    stab_before = _get_stability(conn, "zero_gap")
    result = apply_generation_spacing_interaction_effect(
        conn, ["zero_gap"], "test",
        pre_stability_map, pre_spaced_access_map, pre_last_accessed_map, now_iso
    )
    stab_after = _get_stability(conn, "zero_gap")

    # effort = 1 - exp(0) = 0 < 0.10 → 无加成
    assert abs(stab_after - stab_before) < 0.001, (
        f"GS10: gap=0 时 effort=0 不应触发 GSIE，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["gsie_boosted"] == 0, f"GS10: gsie_boosted 应为 0，got {result}"


# ── GS11: streak_factor 正确计算 ───────────────────────────────────────────────────

def test_gs11_streak_factor_calculation(conn):
    """GS11: spaced_access_count=gsie_ref_streak(6) → streak_factor=1.0 → 最大 GSIE 加成。"""
    gsie_ref_streak = config.get("store_vfs.gsie_ref_streak")  # 6
    gsie_scale = config.get("store_vfs.gsie_scale")             # 0.12

    # gap=168h, stability=5 → R = exp(-168/120) = exp(-1.4) ≈ 0.247 → effort ≈ 0.753
    last_acc_iso = (_utcnow() - datetime.timedelta(hours=168)).isoformat()
    _insert_chunk(conn, "max_streak", project="test",
                  stability=5.0, importance=0.6,
                  spaced_access_count=gsie_ref_streak,  # = 6
                  last_accessed_hours_ago=168.0)

    pre_stability_map = {"max_streak": 5.0}
    pre_spaced_access_map = {"max_streak": gsie_ref_streak}
    pre_last_accessed_map = {"max_streak": last_acc_iso}
    now_iso = _utcnow().isoformat()

    apply_generation_spacing_interaction_effect(
        conn, ["max_streak"], "test",
        pre_stability_map, pre_spaced_access_map, pre_last_accessed_map, now_iso
    )
    stab_after = _get_stability(conn, "max_streak")

    # effort = 1 - exp(-168/120) ≈ 0.753, streak_factor = 1.0 (= ref_streak/ref_streak)
    # interaction = 0.753, gsie_bonus = 0.753 * 0.12 ≈ 0.090
    # new_stab = 5.0 * 1.090 = 5.45
    effort = 1.0 - math.exp(-168.0 / (5.0 * 24.0))
    expected_bonus = effort * 1.0 * gsie_scale
    expected_stab = 5.0 * (1.0 + expected_bonus)

    assert abs(stab_after - expected_stab) < 0.05, (
        f"GS11: streak_factor=1.0 时加成应精确，expected={expected_stab:.4f} got={stab_after:.4f}"
    )


# ── GS12: update_accessed() 集成测试 ────────────────────────────────────────────────

def test_gs12_update_accessed_integration(conn):
    """GS12: update_accessed() 对高 gap + 高 streak chunk 被检索后 stability 增加。"""
    # 插入一个高间隔历史、低 retrievability 的 chunk
    _insert_chunk(conn, "gsie_integ", project="test",
                  stability=5.0, importance=0.6,
                  spaced_access_count=5,      # >= gsie_min_streak(2)
                  last_accessed_hours_ago=96.0,  # 4 days ago → high effort
                  retrievability=0.3)

    stab_before = _get_stability(conn, "gsie_integ")

    # 触发 update_accessed → 内部读取 pre_spaced_access_map + 调用 apply_generation_spacing_interaction_effect
    update_accessed(conn, ["gsie_integ"])

    stab_after = _get_stability(conn, "gsie_integ")

    assert stab_after > stab_before, (
        f"GS12: update_accessed 对高 gap + 高 streak chunk 应触发 GSIE 加成，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )
