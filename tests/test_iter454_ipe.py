"""
test_iter454_ipe.py — iter454: Interleaved Practice Effect 单元测试

覆盖：
  IP1: chunk_ids 含多种 chunk_type（interleaved）→ 每个 chunk stability 增加
  IP2: 所有 chunk 同一 chunk_type → 无 IPE 加成（集中练习，非混合）
  IP3: chunk_ids 数量 < ipe_min_chunks(2) → 无 IPE 加成
  IP4: ipe_enabled=False → 无任何加成
  IP5: importance < ipe_min_importance(0.40) → 不参与 IPE
  IP6: 更高 diversity_factor（类型数/总数比例更大）→ 更大加成（正比关系）
  IP7: stability 加成后不超过 365.0（cap 保护）
  IP8: ipe_scale 可配置（更大 scale → 更大加成）
  IP9: 返回值正确（ipe_boosted, total_examined 计数）
  IP10: 恰好 ipe_min_types(2) 种类型 → 触发（边界条件）
  IP11: unique_type_count == len(chunk_ids)（完全混合，diversity_factor=1.0）→ 最大加成
  IP12: update_accessed() 集成测试：混合类型 chunk 被检索后 stability 增加

认知科学依据：
  Kornell & Bjork (2008) Psychological Science "Learning concepts and categories" —
    混合练习（不同类别交替）比集中练习（同类别连续）在延迟测试中表现好 43-57%。
    机制：迫使大脑重建区分性特征，形成更丰富的多维检索线索集。
  Rohrer & Taylor (2007) "The shuffling of mathematics problems improves learning" —
    混合练习使分类识别能力提升 +43%（边界特征编码强化）。

OS 类比：CPU cross-stride interleaved access → multi-stream prefetch trigger —
  跨 chunk_type 的混合检索 = 多维语义访问模式 → prefetcher 提升预取优先级
  = memory-os 给予混合检索的每个 chunk 额外 stability 加成。
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
    apply_interleaved_practice_effect,
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
                  chunk_type="decision", retrievability=0.5, access_count=2):
    """Insert a chunk with controlled state for IPE testing."""
    import datetime as _dt
    now_iso = _utcnow().isoformat()
    # last_accessed 10min ago to avoid IOR penalty (IOR window=300s)
    la_iso = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(minutes=10)).isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count,
            encode_context)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (cid, project, chunk_type, f"content {cid}", f"summary {cid}",
         importance, stability, now_iso, now_iso, retrievability,
         la_iso, access_count, "kernel_mm")
    )
    conn.commit()


def _get_stability(conn, cid: str) -> float:
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


# ── IP1: 混合类型检索 → stability 增加 ──────────────────────────────────────────────────

def test_ip1_mixed_types_boost_stability(conn):
    """IP1: chunk_ids 含多种 chunk_type（interleaved）→ 每个 chunk stability 增加。"""
    _insert_chunk(conn, "chunk_dec", chunk_type="decision", stability=5.0, importance=0.6)
    _insert_chunk(conn, "chunk_dc", chunk_type="design_constraint", stability=5.0, importance=0.6)
    _insert_chunk(conn, "chunk_rc", chunk_type="reasoning_chain", stability=5.0, importance=0.6)

    stabs_before = {
        "chunk_dec": _get_stability(conn, "chunk_dec"),
        "chunk_dc": _get_stability(conn, "chunk_dc"),
        "chunk_rc": _get_stability(conn, "chunk_rc"),
    }

    result = apply_interleaved_practice_effect(
        conn, ["chunk_dec", "chunk_dc", "chunk_rc"], "test"
    )

    for cid in ["chunk_dec", "chunk_dc", "chunk_rc"]:
        stab_after = _get_stability(conn, cid)
        assert stab_after > stabs_before[cid], (
            f"IP1: 混合类型检索应触发 IPE，{cid} before={stabs_before[cid]:.4f} after={stab_after:.4f}"
        )

    assert result["ipe_boosted"] >= 3, f"IP1: ipe_boosted 应 >= 3，got {result}"


# ── IP2: 同一 chunk_type → 无 IPE 加成 ───────────────────────────────────────────────────

def test_ip2_same_type_no_boost(conn):
    """IP2: 所有 chunk 同一 chunk_type → 集中练习，不触发 IPE。"""
    _insert_chunk(conn, "same1", chunk_type="decision", stability=5.0, importance=0.6)
    _insert_chunk(conn, "same2", chunk_type="decision", stability=5.0, importance=0.6)
    _insert_chunk(conn, "same3", chunk_type="decision", stability=5.0, importance=0.6)

    stab_before = _get_stability(conn, "same1")
    result = apply_interleaved_practice_effect(
        conn, ["same1", "same2", "same3"], "test"
    )
    stab_after = _get_stability(conn, "same1")

    assert abs(stab_after - stab_before) < 0.001, (
        f"IP2: 同类型不应触发 IPE，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["ipe_boosted"] == 0, f"IP2: ipe_boosted 应为 0，got {result}"


# ── IP3: chunk_ids 数量不足 → 无 IPE 加成 ────────────────────────────────────────────────

def test_ip3_too_few_chunks_no_boost(conn):
    """IP3: chunk_ids 数量 < ipe_min_chunks(2) → 无 IPE 加成。"""
    _insert_chunk(conn, "single", chunk_type="decision", stability=5.0, importance=0.6)

    stab_before = _get_stability(conn, "single")
    result = apply_interleaved_practice_effect(
        conn, ["single"], "test"  # 只有 1 个 chunk
    )
    stab_after = _get_stability(conn, "single")

    assert abs(stab_after - stab_before) < 0.001, (
        f"IP3: chunk 数量不足不应触发 IPE，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["ipe_boosted"] == 0, f"IP3: ipe_boosted 应为 0，got {result}"


# ── IP4: ipe_enabled=False → 无加成 ─────────────────────────────────────────────────────

def test_ip4_disabled_no_boost(conn):
    """IP4: store_vfs.ipe_enabled=False → 无任何 IPE 加成。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.ipe_enabled":
            return False
        return original_get(key, project=project)

    _insert_chunk(conn, "dis1", chunk_type="decision", stability=5.0, importance=0.6)
    _insert_chunk(conn, "dis2", chunk_type="design_constraint", stability=5.0, importance=0.6)

    stab_before = _get_stability(conn, "dis1")
    with mock.patch.object(config, 'get', side_effect=patched_get):
        result = apply_interleaved_practice_effect(conn, ["dis1", "dis2"], "test")
    stab_after = _get_stability(conn, "dis1")

    assert abs(stab_after - stab_before) < 0.001, (
        f"IP4: disabled 时不应有 IPE 加成，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["ipe_boosted"] == 0, f"IP4: ipe_boosted 应为 0，got {result}"


# ── IP5: importance 不足 → 不参与 IPE ────────────────────────────────────────────────────

def test_ip5_low_importance_excluded(conn):
    """IP5: importance < ipe_min_importance(0.40) → 不参与 IPE。"""
    _insert_chunk(conn, "low_imp1", chunk_type="decision", stability=5.0, importance=0.20)
    _insert_chunk(conn, "low_imp2", chunk_type="design_constraint", stability=5.0, importance=0.20)

    stab_before = _get_stability(conn, "low_imp1")
    result = apply_interleaved_practice_effect(
        conn, ["low_imp1", "low_imp2"], "test"
    )
    stab_after = _get_stability(conn, "low_imp1")

    assert abs(stab_after - stab_before) < 0.001, (
        f"IP5: 低 importance 不应触发 IPE，before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── IP6: 更高 diversity_factor → 更大加成 ────────────────────────────────────────────────

def test_ip6_higher_diversity_more_boost(conn):
    """IP6: 更高 diversity_factor（类型数/总数比例更大）→ 更大 interleave_bonus → 更大加成。"""
    # 场景 A：2 种类型 / 4 个 chunk = diversity_factor = 0.5
    for i in range(2):
        _insert_chunk(conn, f"low_div_dec_{i}", project="proj_a",
                      chunk_type="decision", stability=5.0, importance=0.6)
        _insert_chunk(conn, f"low_div_dc_{i}", project="proj_a",
                      chunk_type="design_constraint", stability=5.0, importance=0.6)
    apply_interleaved_practice_effect(
        conn, [f"low_div_dec_{i}" for i in range(2)] + [f"low_div_dc_{i}" for i in range(2)],
        "proj_a"
    )
    stab_low_div = _get_stability(conn, "low_div_dec_0")

    # 场景 B：4 种类型 / 4 个 chunk = diversity_factor = 1.0
    types_b = ["decision", "design_constraint", "reasoning_chain", "causal_chain"]
    for t in types_b:
        _insert_chunk(conn, f"high_div_{t}", project="proj_b",
                      chunk_type=t, stability=5.0, importance=0.6)
    apply_interleaved_practice_effect(
        conn, [f"high_div_{t}" for t in types_b], "proj_b"
    )
    stab_high_div = _get_stability(conn, "high_div_decision")

    assert stab_high_div > stab_low_div, (
        f"IP6: 更高 diversity_factor 应获得更大加成，"
        f"stab_low_div={stab_low_div:.4f} stab_high_div={stab_high_div:.4f}"
    )


# ── IP7: stability 上限 365.0 ─────────────────────────────────────────────────────────────

def test_ip7_stability_cap_365(conn):
    """IP7: IPE boost 后 stability 不超过 365.0。"""
    _insert_chunk(conn, "cap1", chunk_type="decision", stability=364.9, importance=0.8)
    _insert_chunk(conn, "cap2", chunk_type="design_constraint", stability=364.9, importance=0.8)

    apply_interleaved_practice_effect(conn, ["cap1", "cap2"], "test")
    stab_after = _get_stability(conn, "cap1")

    assert stab_after <= 365.0, f"IP7: stability 不应超过 365.0，got {stab_after}"


# ── IP8: ipe_scale 可配置 ─────────────────────────────────────────────────────────────────

def test_ip8_configurable_scale(conn):
    """IP8: ipe_scale=0.30 时加成比默认 0.08 更大。"""
    original_get = config.get

    def patched_30(key, project=None):
        if key == "store_vfs.ipe_scale":
            return 0.30
        return original_get(key, project=project)

    _insert_chunk(conn, "scale1", project="proj_scale", chunk_type="decision",
                  stability=5.0, importance=0.6)
    _insert_chunk(conn, "scale2", project="proj_scale", chunk_type="design_constraint",
                  stability=5.0, importance=0.6)
    stab_before = _get_stability(conn, "scale1")

    with mock.patch.object(config, 'get', side_effect=patched_30):
        apply_interleaved_practice_effect(conn, ["scale1", "scale2"], "proj_scale")
    stab_after_30 = _get_stability(conn, "scale1")
    delta_30 = stab_after_30 - stab_before

    # 重置
    conn.execute("UPDATE memory_chunks SET stability=5.0 WHERE id='scale1'")
    conn.commit()

    apply_interleaved_practice_effect(conn, ["scale1", "scale2"], "proj_scale")
    stab_after_default = _get_stability(conn, "scale1")
    delta_default = stab_after_default - 5.0  # 用原始值 5.0 计算

    assert delta_30 > delta_default, (
        f"IP8: ipe_scale=0.30 加成应大于默认 0.08，"
        f"delta_30={delta_30:.5f} delta_default={delta_default:.5f}"
    )


# ── IP9: 返回值正确 ──────────────────────────────────────────────────────────────────────

def test_ip9_return_counts_correct(conn):
    """IP9: result dict 中 ipe_boosted 和 total_examined 计数正确。"""
    types_ok = ["decision", "design_constraint", "reasoning_chain"]
    for t in types_ok:
        _insert_chunk(conn, f"ret_{t}", chunk_type=t, stability=5.0, importance=0.6)

    # 一个 importance 不足的 chunk（应被 examined 但不 boosted）
    _insert_chunk(conn, "ret_low", chunk_type="causal_chain", stability=5.0, importance=0.10)

    chunk_ids = [f"ret_{t}" for t in types_ok] + ["ret_low"]
    result = apply_interleaved_practice_effect(conn, chunk_ids, "test")

    assert "ipe_boosted" in result, "IP9: result 应含 ipe_boosted key"
    assert "total_examined" in result, "IP9: result 应含 total_examined key"
    assert result["ipe_boosted"] >= 3, f"IP9: ipe_boosted 应 >= 3，got {result}"
    assert result["total_examined"] >= 4, f"IP9: total_examined 应 >= 4，got {result}"


# ── IP10: 恰好 ipe_min_types(2) 种类型 → 触发（边界条件）────────────────────────────────

def test_ip10_exactly_min_types_triggers(conn):
    """IP10: unique_type_count == ipe_min_types(2) → 边界条件，应触发 IPE。"""
    _insert_chunk(conn, "bound1", chunk_type="decision", stability=5.0, importance=0.6)
    _insert_chunk(conn, "bound2", chunk_type="design_constraint", stability=5.0, importance=0.6)

    stab_before = _get_stability(conn, "bound1")
    result = apply_interleaved_practice_effect(conn, ["bound1", "bound2"], "test")
    stab_after = _get_stability(conn, "bound1")

    assert stab_after > stab_before, (
        f"IP10: 恰好 2 种类型时应触发 IPE，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["ipe_boosted"] >= 1, f"IP10: ipe_boosted 应 >= 1，got {result}"


# ── IP11: 完全混合（diversity_factor=1.0）→ 最大加成 ─────────────────────────────────────

def test_ip11_full_diversity_max_bonus(conn):
    """IP11: unique_type_count == len(chunk_ids) → diversity_factor=1.0 → 最大 interleave_bonus。"""
    # 每个 chunk 都是不同类型（完全混合）
    unique_types = ["decision", "design_constraint", "reasoning_chain",
                    "causal_chain", "quantitative_evidence"]
    for t in unique_types:
        _insert_chunk(conn, f"full_{t}", chunk_type=t, stability=5.0, importance=0.6)

    stab_before = _get_stability(conn, "full_decision")
    result = apply_interleaved_practice_effect(
        conn, [f"full_{t}" for t in unique_types], "test"
    )
    stab_after = _get_stability(conn, "full_decision")

    ipe_scale = config.get("store_vfs.ipe_scale")  # 0.08
    expected_bonus = 1.0 * ipe_scale  # diversity_factor=1.0
    expected_stab = 5.0 * (1.0 + expected_bonus)

    assert stab_after > stab_before, f"IP11: 完全混合应触发 IPE，after={stab_after:.4f}"
    assert abs(stab_after - expected_stab) < 0.01, (
        f"IP11: 完全混合时应达到最大加成，expected={expected_stab:.4f} got={stab_after:.4f}"
    )


# ── IP12: update_accessed() 集成测试 ─────────────────────────────────────────────────────

def test_ip12_update_accessed_integration(conn):
    """IP12: update_accessed() 对混合类型 chunk 检索后 stability 增加。"""
    # 插入不同类型的 chunk
    _insert_chunk(conn, "integ_dec", chunk_type="decision",
                  stability=5.0, importance=0.6, retrievability=0.6, access_count=3)
    _insert_chunk(conn, "integ_dc", chunk_type="design_constraint",
                  stability=5.0, importance=0.6, retrievability=0.6, access_count=3)

    stab_before_dec = _get_stability(conn, "integ_dec")
    stab_before_dc = _get_stability(conn, "integ_dc")

    # 触发 update_accessed → 内部调用 apply_interleaved_practice_effect
    update_accessed(conn, ["integ_dec", "integ_dc"])

    stab_after_dec = _get_stability(conn, "integ_dec")
    stab_after_dc = _get_stability(conn, "integ_dc")

    # 混合类型检索应触发 IPE
    assert stab_after_dec > stab_before_dec or stab_after_dc > stab_before_dc, (
        f"IP12: update_accessed 对混合类型 chunk 应触发 IPE，"
        f"dec: {stab_before_dec:.4f}→{stab_after_dec:.4f}, "
        f"dc: {stab_before_dc:.4f}→{stab_after_dc:.4f}"
    )
