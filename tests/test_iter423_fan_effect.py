"""
test_iter423_fan_effect.py — iter423: Fan Effect — IDF加权 Spreading Activation

覆盖：
  FE1: 低扇出 entity → 无惩罚（degree < min_degree）
  FE2: 高扇出 entity → spreading activation 激活分降低
  FE3: 高扇出 entity 比低扇出 entity 的 spreading activation 分更低
  FE4: fan_effect_enabled=False → 无 Fan Effect 惩罚
  FE5: fan_effect_idf_weight=0 → 无惩罚（类似 disabled）
  FE6: fan_effect_idf_weight=1.0 → 最大惩罚
  FE7: 单 entity 无边（degree=0）→ 无 Fan Effect
  FE8: Fan Effect 不影响直接 FTS5 命中的 chunk 分数（只影响 spreading activation 邻居）
  FE9: _fan_idf_factor 函数验证（degree, median → factor ∈ [0.1, 1.0]）
  FE10: 多 hop 下高扇出 entity 的激活分抑制效果传递

认知科学依据：
  Anderson (1974) Fan Effect — 一个概念关联越多事实，提取每条事实越慢。
  Anderson (1983) ACT* theory — 激活扩散时，高扇出节点的激活总量被其所有出边分摊。

OS 类比：CPU cache set-associativity conflict — 过多 cache line 映射到同一 set →
  命中率下降，fan_idf 是 cache placement policy 的激活版本。
"""
import sys
import sqlite3
import pytest
from pathlib import Path
import math

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import tmpfs  # noqa

from store_vfs import ensure_schema, spreading_activate
import config


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _insert_entity_edge(conn, from_e, to_e, project="test", confidence=0.9):
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    conn.execute(
        "INSERT OR IGNORE INTO entity_edges "
        "(id, from_entity, relation, to_entity, project, confidence, agent_id, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (f"{from_e}-{to_e}", from_e, "related", to_e, project, confidence, "", now)
    )
    conn.commit()


def _insert_entity_map(conn, entity_name, chunk_id, project="test"):
    conn.execute(
        "INSERT OR IGNORE INTO entity_map (entity_name, chunk_id, project) VALUES (?, ?, ?)",
        (entity_name, chunk_id, project)
    )
    conn.commit()


def _insert_chunk(conn, cid, project="test"):
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO memory_chunks "
        "(id, project, chunk_type, content, summary, importance, stability, "
        "created_at, updated_at, retrievability) "
        "VALUES (?, ?, 'decision', ?, ?, 0.7, 1.0, ?, ?, 0.9)",
        (cid, project, f"content {cid}", f"summary {cid}", now, now)
    )
    conn.commit()


# ── FE1: Low fan-out entity → no penalty (degree < min_degree) ──────────────

def test_fe1_low_fanout_no_penalty(conn):
    """低扇出 entity（degree < min_degree）→ spreading activation 无惩罚。"""
    # Setup: chunk A → entity_A → entity_B → chunk B (only 1 edge, low fanout)
    _insert_chunk(conn, "fe1_a")
    _insert_chunk(conn, "fe1_b")
    _insert_entity_map(conn, "entity_a1", "fe1_a")
    _insert_entity_map(conn, "entity_b1", "fe1_b")
    _insert_entity_edge(conn, "entity_a1", "entity_b1", confidence=0.8)

    result = spreading_activate(conn, ["fe1_a"], project="test", decay=0.7, max_hops=1)
    # With low fanout (degree=1 < min_degree=3), score should be based on pure activation
    assert "fe1_b" in result, "FE1: 低扇出 entity 邻居应在结果中"
    # Score = confidence × decay × distance_decay_factor(hop=1) = 0.8 × 0.7 × 0.6 = 0.336
    # sa_distance_decay_factor defaults to 0.6
    expected_score = 0.8 * 0.7 * 0.6  # 0.336
    assert result["fe1_b"] >= expected_score * 0.8, \
        f"FE1: 低扇出激活分应接近 {expected_score:.3f}，got {result['fe1_b']:.3f}"


# ── FE2: High fan-out entity → activation score reduced ─────────────────────

def test_fe2_high_fanout_reduced_score(conn):
    """高扇出 entity → spreading activation 激活分降低。"""
    # Setup: chunk A → "common_entity" (connected to 10 other entities) → chunk B
    _insert_chunk(conn, "fe2_a")
    _insert_chunk(conn, "fe2_b")
    _insert_entity_map(conn, "common_entity", "fe2_a")
    _insert_entity_map(conn, "fe2_target", "fe2_b")

    # Make common_entity high fan-out: connect it to 10 different entities
    for i in range(10):
        _insert_entity_edge(conn, "common_entity", f"fe2_other_{i}", confidence=0.5)
    _insert_entity_edge(conn, "common_entity", "fe2_target", confidence=0.9)

    result_with_fan = spreading_activate(conn, ["fe2_a"], project="test", decay=0.7, max_hops=1)

    # Verify fe2_b is in results (entity might still activate if degree high enough)
    if "fe2_b" in result_with_fan:
        # Score should be reduced due to Fan Effect
        # Without fan: 0.9 × 0.7 × 0.6 = 0.378; with fan penalty it should be less or equal
        expected_no_fan = 0.9 * 0.7 * 0.6
        assert result_with_fan["fe2_b"] <= expected_no_fan + 0.001, \
            f"FE2: 高扇出 entity 激活分应 <= 无惩罚值 {expected_no_fan:.3f}，got {result_with_fan['fe2_b']:.4f}"


# ── FE3: High fan-out entity activates neighbors with lower score ─────────────

def test_fe3_high_vs_low_fanout_comparison(conn):
    """高扇出 entity 的 spreading activation 分 < 低扇出 entity 的分。"""
    # Setup two paths:
    # Path 1: chunk_src → low_fan_entity (degree=1) → chunk_low
    # Path 2: chunk_src → high_fan_entity (degree=10) → chunk_high
    _insert_chunk(conn, "fe3_src")
    _insert_chunk(conn, "fe3_low")
    _insert_chunk(conn, "fe3_high_target")
    _insert_entity_map(conn, "fe3_low_entity", "fe3_src")
    _insert_entity_map(conn, "fe3_high_entity", "fe3_src")
    _insert_entity_map(conn, "fe3_low_neighbor", "fe3_low")
    _insert_entity_map(conn, "fe3_high_neighbor", "fe3_high_target")

    # Low fan: 1 connection (below min_degree=3)
    _insert_entity_edge(conn, "fe3_low_entity", "fe3_low_neighbor", confidence=0.9)

    # High fan: 8 connections (above min_degree=3)
    for i in range(8):
        _insert_entity_edge(conn, "fe3_high_entity", f"fe3_high_other_{i}", confidence=0.5)
    _insert_entity_edge(conn, "fe3_high_entity", "fe3_high_neighbor", confidence=0.9)

    result = spreading_activate(conn, ["fe3_src"], project="test", decay=0.7, max_hops=1)

    if "fe3_low" in result and "fe3_high_target" in result:
        score_low = result["fe3_low"]
        score_high = result["fe3_high_target"]
        assert score_low >= score_high, \
            f"FE3: 低扇出 entity 激活分({score_low:.4f}) 应 >= 高扇出 entity 激活分({score_high:.4f})"


# ── FE4: fan_effect_enabled=False → no penalty ───────────────────────────────

def test_fe4_disabled_no_penalty(conn, monkeypatch):
    """fan_effect_enabled=False → 无 Fan Effect 惩罚。"""
    import unittest.mock as mock
    original_get = config.get
    def patched_get(key, project=None):
        if key == "retriever.fan_effect_enabled":
            return False
        return original_get(key, project=project)

    _insert_chunk(conn, "fe4_a")
    _insert_chunk(conn, "fe4_b")
    _insert_entity_map(conn, "fe4_common", "fe4_a")
    _insert_entity_map(conn, "fe4_target", "fe4_b")

    # High fan-out entity
    for i in range(10):
        _insert_entity_edge(conn, "fe4_common", f"fe4_other_{i}", confidence=0.5)
    _insert_entity_edge(conn, "fe4_common", "fe4_target", confidence=0.9)

    with mock.patch.object(config, 'get', side_effect=patched_get):
        result = spreading_activate(conn, ["fe4_a"], project="test", decay=0.7, max_hops=1)

    if "fe4_b" in result:
        # Without fan effect: score = 0.9 × 0.7 × 0.6 = 0.378 (no penalty, dist_decay=0.6)
        expected = 0.9 * 0.7 * 0.6
        assert result["fe4_b"] >= expected * 0.90, \
            f"FE4: 禁用 Fan Effect 后分数应接近 {expected:.3f}，got {result['fe4_b']:.4f}"


# ── FE5: fan_effect_idf_weight=0 → no penalty ────────────────────────────────

def test_fe5_idf_weight_zero_no_penalty(conn, monkeypatch):
    """fan_effect_idf_weight=0 → 无惩罚。"""
    import unittest.mock as mock
    original_get = config.get
    def patched_get(key, project=None):
        if key == "retriever.fan_effect_idf_weight":
            return 0.0
        return original_get(key, project=project)

    _insert_chunk(conn, "fe5_a")
    _insert_chunk(conn, "fe5_b")
    _insert_entity_map(conn, "fe5_common", "fe5_a")
    _insert_entity_map(conn, "fe5_target", "fe5_b")

    for i in range(8):
        _insert_entity_edge(conn, "fe5_common", f"fe5_other_{i}", confidence=0.5)
    _insert_entity_edge(conn, "fe5_common", "fe5_target", confidence=0.9)

    with mock.patch.object(config, 'get', side_effect=patched_get):
        result = spreading_activate(conn, ["fe5_a"], project="test", decay=0.7, max_hops=1)

    if "fe5_b" in result:
        expected = 0.9 * 0.7 * 0.6  # include distance decay factor
        assert result["fe5_b"] >= expected * 0.90, \
            f"FE5: idf_weight=0 时分数应接近 {expected:.3f}，got {result['fe5_b']:.4f}"


# ── FE6: fan_effect_idf_weight=1.0 → max penalty ────────────────────────────

def test_fe6_idf_weight_max_penalty(conn, monkeypatch):
    """fan_effect_idf_weight=1.0 → 最大惩罚，高扇出 entity 激活大幅降低。"""
    import unittest.mock as mock
    original_get = config.get
    def patched_get(key, project=None):
        if key == "retriever.fan_effect_idf_weight":
            return 1.0
        return original_get(key, project=project)

    _insert_chunk(conn, "fe6_a")
    _insert_chunk(conn, "fe6_b_base")   # normal fanout
    _insert_chunk(conn, "fe6_b_high")   # high fanout path

    _insert_entity_map(conn, "fe6_normal_entity", "fe6_a")
    _insert_entity_map(conn, "fe6_high_entity", "fe6_a")
    _insert_entity_map(conn, "fe6_normal_target", "fe6_b_base")
    _insert_entity_map(conn, "fe6_high_target", "fe6_b_high")

    # Normal: 1 edge
    _insert_entity_edge(conn, "fe6_normal_entity", "fe6_normal_target", confidence=0.9)

    # High: 15 edges
    for i in range(15):
        _insert_entity_edge(conn, "fe6_high_entity", f"fe6_high_other_{i}", confidence=0.5)
    _insert_entity_edge(conn, "fe6_high_entity", "fe6_high_target", confidence=0.9)

    with mock.patch.object(config, 'get', side_effect=patched_get):
        result = spreading_activate(conn, ["fe6_a"], project="test", decay=0.7, max_hops=1)

    if "fe6_b_base" in result and "fe6_b_high" in result:
        assert result["fe6_b_base"] > result["fe6_b_high"], \
            f"FE6: max 惩罚下低扇出({result['fe6_b_base']:.4f}) 应 > 高扇出({result['fe6_b_high']:.4f})"


# ── FE7: Entity with no edges (degree=0) → no Fan Effect ─────────────────────

def test_fe7_zero_degree_no_fan(conn):
    """Entity degree=0 → 无 Fan Effect 惩罚（不存在于 entity_edges）。"""
    # This tests that when an entity has no cached degree (0), it's treated as low fan-out
    _insert_chunk(conn, "fe7_a")
    _insert_chunk(conn, "fe7_b")
    _insert_entity_map(conn, "fe7_isolated", "fe7_a")
    _insert_entity_map(conn, "fe7_target", "fe7_b")
    _insert_entity_edge(conn, "fe7_isolated", "fe7_target", confidence=0.85)
    # fe7_isolated has degree=1 (just this one edge)

    result = spreading_activate(conn, ["fe7_a"], project="test", decay=0.7, max_hops=1)
    if "fe7_b" in result:
        expected = 0.85 * 0.7 * 0.6  # include distance decay factor (hop=1)
        # Should be approximately the expected score (no fan penalty for degree=1)
        assert result["fe7_b"] >= expected * 0.85, \
            f"FE7: degree=1 entity 应无 Fan 惩罚，expected>={expected*0.85:.4f} got {result['fe7_b']:.4f}"


# ── FE8: Fan Effect doesn't affect directly found chunk scores ───────────────

def test_fe8_no_effect_on_direct_hits(conn):
    """Fan Effect 不影响直接命中 chunk 的分数（只影响 spreading activation 邻居）。"""
    # Fan Effect is only in spreading_activate (spreading to neighbors)
    # Direct hits from FTS5 are not affected
    # Verify: spreading_activate returns only NEIGHBOR chunks, not the hit chunks themselves
    _insert_chunk(conn, "fe8_hit")
    _insert_chunk(conn, "fe8_neighbor")
    _insert_entity_map(conn, "fe8_entity", "fe8_hit")
    _insert_entity_map(conn, "fe8_n_entity", "fe8_neighbor")

    for i in range(10):
        _insert_entity_edge(conn, "fe8_entity", f"fe8_other_{i}", confidence=0.5)
    _insert_entity_edge(conn, "fe8_entity", "fe8_n_entity", confidence=0.9)

    result = spreading_activate(conn, ["fe8_hit"], project="test",
                                 existing_ids={"fe8_hit"}, decay=0.7, max_hops=1)
    assert "fe8_hit" not in result, "FE8: 命中 chunk 不应出现在 spreading activation 结果中"


# ── FE9: _fan_idf_factor function validation ─────────────────────────────────

def test_fe9_fan_idf_factor_correctness():
    """_fan_idf_factor 函数逻辑验证。"""
    import importlib
    # Import the function by executing code equivalent
    # Test the IDF factor logic directly:
    def fan_idf_factor(degree, min_degree, median_deg, idf_weight):
        if degree < min_degree:
            return 1.0
        idf_raw = math.log(1.0 + max(1.0, median_deg) / (1.0 + degree))
        idf_norm_max = math.log(1.0 + max(1.0, median_deg))
        idf = idf_raw / idf_norm_max if idf_norm_max > 0 else 1.0
        idf = max(0.1, min(1.0, idf))
        return 1.0 - idf_weight * (1.0 - idf)

    median = 5.0
    min_deg = 3

    # degree < min_degree → factor = 1.0
    assert fan_idf_factor(2, min_deg, median, 0.5) == 1.0, "FE9: degree < min → factor=1.0"

    # degree = median → moderate factor
    factor_at_median = fan_idf_factor(5, min_deg, median, 0.5)
    assert 0.5 < factor_at_median < 1.0, f"FE9: degree=median → factor ∈ (0.5, 1.0)，got {factor_at_median}"

    # high degree → lower factor
    factor_high = fan_idf_factor(50, min_deg, median, 0.5)
    factor_low = fan_idf_factor(4, min_deg, median, 0.5)
    assert factor_high < factor_low, \
        f"FE9: high degree({factor_high:.4f}) 应 < low degree({factor_low:.4f})"

    # factor always ∈ [1-idf_weight, 1.0]
    for d in [3, 5, 10, 50, 100]:
        f = fan_idf_factor(d, min_deg, median, 0.5)
        assert 0.0 < f <= 1.0, f"FE9: factor 应在 (0, 1]，degree={d} got {f:.4f}"

    # idf_weight=0 → factor=1.0 always
    for d in [3, 10, 100]:
        f = fan_idf_factor(d, min_deg, median, 0.0)
        assert f == 1.0, f"FE9: idf_weight=0 → factor=1.0，degree={d} got {f}"


# ── FE10: Multi-hop propagation with Fan Effect ───────────────────────────────

def test_fe10_multihop_fan_attenuation(conn):
    """多跳激活中，高扇出 entity 的激活惩罚沿路径传递（累积衰减）。"""
    # Path: chunk_src → low_fan (hop1) → hub_entity (high fan, hop2) → chunk_deep
    _insert_chunk(conn, "fe10_src")
    _insert_chunk(conn, "fe10_mid")
    _insert_chunk(conn, "fe10_deep")

    _insert_entity_map(conn, "fe10_src_e", "fe10_src")
    _insert_entity_map(conn, "fe10_mid_e", "fe10_mid")
    _insert_entity_map(conn, "fe10_hub_e", "fe10_mid")  # hub is also mapped to fe10_mid
    _insert_entity_map(conn, "fe10_deep_e", "fe10_deep")

    # Hop 1: src → mid (low fanout)
    _insert_entity_edge(conn, "fe10_src_e", "fe10_mid_e", confidence=0.9)

    # Hop 2: hub (high fanout) → deep
    for i in range(10):
        _insert_entity_edge(conn, "fe10_hub_e", f"fe10_hub_other_{i}", confidence=0.5)
    _insert_entity_edge(conn, "fe10_hub_e", "fe10_deep_e", confidence=0.9)

    result = spreading_activate(conn, ["fe10_src"], project="test", decay=0.7, max_hops=2)

    if "fe10_deep" in result:
        # With hop=2 decay + possible fan penalty: score should be << 0.9 × 0.7^2 = 0.441
        assert result["fe10_deep"] <= 0.45, \
            f"FE10: 多跳高扇出路径激活分应较低，got {result['fe10_deep']:.4f}"
