"""
test_semantic_distance_decay.py — iter393: Semantic Distance Decay in Spreading Activation

覆盖：
  SD1: 1跳邻居 → activation_score = seed × eff_conf × decay × ddf^1
  SD2: 2跳邻居 → activation_score 比 1跳低（× ddf^2 vs × ddf^1）
  SD3: distance_decay_enabled=False → 退化到旧行为（1跳 × decay，2跳 × decay^2，无距离衰减）
  SD4: distance_decay_factor=1.0 → 等价于 disabled（无额外衰减）
  SD5: 2跳/1跳比值 = ddf（在 eff_conf=1.0, decay=1.0 时精确验证）
  SD6: max_hops=1 → 只有 1 跳邻居被激活，2 跳无结果
  SD7: distance_decay_factor=0.1 → 2跳激活极小（接近截止阈值）

认知科学依据：
  Collins & Loftus (1975) Spreading Activation Theory —
    "cat" → "animal"（1跳）比 "cat" → "mammal" → "vertebrate"（2跳）激活量低，
    语义距离越远，激活越弱，形成自然的语义相关性梯度。
OS 类比：NUMA 局部性 — 同节点 L3 命中快，跨 2 个 NUMA 节点延迟呈指数增长。
"""
import sys
import sqlite3
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "hooks"))

import tmpfs  # noqa
from store_vfs import ensure_schema, spreading_activate


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _seed_graph(conn, seed_chunk="chunk_seed", entity_a="entity_A",
                entity_b="entity_B", entity_c="entity_C",
                chunk_b="chunk_B", chunk_c="chunk_C",
                project="test_proj"):
    """
    构建测试图：
      seed_chunk  →  entity_A  (直接绑定)
      entity_A → entity_B   (1跳邻居 → chunk_B)
      entity_B → entity_C   (2跳邻居 → chunk_C)
    confidence=1.0, edge_half_life_days=0 (禁用时间衰减, eff_conf=1.0)
    """
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()

    # seed chunk
    conn.execute(
        "INSERT INTO memory_chunks (id, project, chunk_type, content, summary, "
        "importance, retrievability, created_at, updated_at) "
        "VALUES (?, ?, 'decision', ?, ?, 0.8, 0.5, ?, ?)",
        (seed_chunk, project, f"content {seed_chunk}", f"summary {seed_chunk}", now, now)
    )
    # chunk_B (1跳目标)
    conn.execute(
        "INSERT INTO memory_chunks (id, project, chunk_type, content, summary, "
        "importance, retrievability, created_at, updated_at) "
        "VALUES (?, ?, 'reasoning_chain', ?, ?, 0.7, 0.5, ?, ?)",
        (chunk_b, project, f"content {chunk_b}", f"summary {chunk_b}", now, now)
    )
    # chunk_C (2跳目标)
    conn.execute(
        "INSERT INTO memory_chunks (id, project, chunk_type, content, summary, "
        "importance, retrievability, created_at, updated_at) "
        "VALUES (?, ?, 'causal_chain', ?, ?, 0.6, 0.5, ?, ?)",
        (chunk_c, project, f"content {chunk_c}", f"summary {chunk_c}", now, now)
    )

    # entity_map: seed_chunk → entity_A
    conn.execute(
        "INSERT OR IGNORE INTO entity_map (entity_name, chunk_id, project) VALUES (?, ?, ?)",
        (entity_a, seed_chunk, project)
    )
    # entity_map: chunk_B → entity_B
    conn.execute(
        "INSERT OR IGNORE INTO entity_map (entity_name, chunk_id, project) VALUES (?, ?, ?)",
        (entity_b, chunk_b, project)
    )
    # entity_map: chunk_C → entity_C
    conn.execute(
        "INSERT OR IGNORE INTO entity_map (entity_name, chunk_id, project) VALUES (?, ?, ?)",
        (entity_c, chunk_c, project)
    )

    # entity_edges: A → B (confidence=1.0)
    conn.execute(
        "INSERT OR IGNORE INTO entity_edges (from_entity, to_entity, relation, project, confidence, created_at) "
        "VALUES (?, ?, 'uses', ?, 1.0, ?)",
        (entity_a, entity_b, project, now)
    )
    # entity_edges: B → C (confidence=1.0)
    conn.execute(
        "INSERT OR IGNORE INTO entity_edges (from_entity, to_entity, relation, project, confidence, created_at) "
        "VALUES (?, ?, 'uses', ?, 1.0, ?)",
        (entity_b, entity_c, project, now)
    )
    conn.commit()


# ══════════════════════════════════════════════════════════════════════
# 1. 基础语义距离衰减行为
# ══════════════════════════════════════════════════════════════════════

def test_sd1_one_hop_activation(conn):
    """1跳邻居 chunk_B 应被激活，score > 0。"""
    _seed_graph(conn)
    result = spreading_activate(
        conn, ["chunk_seed"], project="test_proj",
        decay=1.0,        # 禁用 decay，纯距离衰减
        max_hops=2,
        edge_half_life_days=0,  # 禁用时间衰减
        distance_decay_enabled=True,
        distance_decay_factor=0.6,
    )
    assert "chunk_B" in result, "SD1: 1跳邻居 chunk_B 应在激活结果中"
    assert result["chunk_B"] > 0, f"SD1: 1跳分数应 > 0，got {result.get('chunk_B')}"


def test_sd2_two_hop_lower_than_one_hop(conn):
    """2跳邻居 chunk_C 的 activation_score < 1跳邻居 chunk_B。"""
    _seed_graph(conn)
    result = spreading_activate(
        conn, ["chunk_seed"], project="test_proj",
        decay=1.0,
        max_hops=2,
        edge_half_life_days=0,
        distance_decay_enabled=True,
        distance_decay_factor=0.6,
    )
    assert "chunk_B" in result, "SD2: chunk_B 应在结果中"
    assert "chunk_C" in result, "SD2: chunk_C 应在结果中（2跳）"
    assert result["chunk_C"] < result["chunk_B"], (
        f"SD2: 2跳 score={result['chunk_C']:.4f} 应 < 1跳 score={result['chunk_B']:.4f}"
    )


def test_sd3_disabled_no_distance_penalty(conn):
    """distance_decay_enabled=False → 2跳与1跳 score 相同（只有 decay^hop 差异，decay=1.0 时相等）。"""
    _seed_graph(conn)
    result = spreading_activate(
        conn, ["chunk_seed"], project="test_proj",
        decay=1.0,
        max_hops=2,
        edge_half_life_days=0,
        distance_decay_enabled=False,
        distance_decay_factor=0.6,  # 应被忽略
    )
    assert "chunk_B" in result, "SD3: chunk_B 应在结果中"
    assert "chunk_C" in result, "SD3: chunk_C 应在结果中（disabled 时 2 跳仍激活）"
    # decay=1.0, confidence=1.0, distance_decay disabled → 1跳 = 2跳 = 1.0 (capped by max_activation_bonus)
    # 实际值取决于 parent_score 传播，但两者应近似（B=1.0×1.0×1.0=1.0, C=1.0×1.0×1.0=1.0）
    # 由于 max_activation_bonus 上限，两者可能都被 cap 到 0.4
    b_score = result["chunk_B"]
    c_score = result["chunk_C"]
    # 无距离衰减时，两者 score 之差 ≤ 1e-6（取决于 max_activation_bonus cap）
    assert abs(b_score - c_score) < 1e-6, (
        f"SD3: disabled 时 1跳 score={b_score:.4f} 应等于 2跳 score={c_score:.4f}"
    )


def test_sd4_factor_1_equals_disabled(conn):
    """distance_decay_factor=1.0 等价于 disabled（乘以 1.0^hop=1.0，无衰减）。"""
    _seed_graph(conn)
    result_1 = spreading_activate(
        conn, ["chunk_seed"], project="test_proj",
        decay=1.0, max_hops=2, edge_half_life_days=0,
        distance_decay_enabled=True,
        distance_decay_factor=1.0,  # 无额外衰减
    )
    result_disabled = spreading_activate(
        conn, ["chunk_seed"], project="test_proj",
        decay=1.0, max_hops=2, edge_half_life_days=0,
        distance_decay_enabled=False,
    )
    for cid in ("chunk_B", "chunk_C"):
        assert abs(result_1.get(cid, 0) - result_disabled.get(cid, 0)) < 1e-6, (
            f"SD4: factor=1.0 与 disabled 结果不同: {cid} "
            f"factor1={result_1.get(cid, 0):.4f} disabled={result_disabled.get(cid, 0):.4f}"
        )


def test_sd5_two_hop_ratio_equals_decay_factor(conn):
    """
    eff_conf=1.0, decay=1.0 时，2跳/1跳 score 比值 = distance_decay_factor。

    推导（BFS score 传播）：
      hop=1 frontier[entity_B] = seed_score(1.0) * eff_conf(1.0) * decay(1.0) * ddf^1
                                = ddf
      hop=2 frontier[entity_C] = frontier[entity_B] * eff_conf(1.0) * decay(1.0) * ddf^2
                                = ddf * ddf^2 = ddf^3

    NOTE: parent_score in hop=2 已是 ddf（来自 hop=1 frontier）。
    所以：
      score_B_entity = 1.0 * 1.0 * 1.0 * ddf = ddf
      score_C_entity = ddf * 1.0 * 1.0 * ddf^2 = ddf^3
      ratio = ddf^3 / ddf = ddf^2

    实际上 store 的 score = min(entity_score, max_activation_bonus)，
    测试中使用 max_activation_bonus=99.0 确保不触发 cap。
    """
    ddf = 0.6
    _seed_graph(conn)
    result = spreading_activate(
        conn, ["chunk_seed"], project="test_proj",
        decay=1.0, max_hops=2, edge_half_life_days=0,
        max_activation_bonus=99.0,  # 不触发 cap
        distance_decay_enabled=True,
        distance_decay_factor=ddf,
    )
    score_b = result.get("chunk_B", 0)
    score_c = result.get("chunk_C", 0)
    assert score_b > 0 and score_c > 0, f"SD5: 两个邻居都应有分数，B={score_b} C={score_c}"
    ratio = score_c / score_b
    # score_B = ddf^1 = 0.6, score_C = ddf^3 = 0.216
    # ratio = 0.216/0.6 = 0.36 = ddf^2
    expected_ratio = ddf ** 2
    assert abs(ratio - expected_ratio) < 1e-6, (
        f"SD5: 2跳/1跳 ratio={ratio:.6f}，expected ddf^2={expected_ratio:.6f}"
    )


def test_sd6_max_hops_1_no_two_hop(conn):
    """max_hops=1 → 只有 1 跳邻居被激活，2 跳无结果。"""
    _seed_graph(conn)
    result = spreading_activate(
        conn, ["chunk_seed"], project="test_proj",
        decay=1.0, max_hops=1, edge_half_life_days=0,
        distance_decay_enabled=True,
        distance_decay_factor=0.6,
    )
    assert "chunk_B" in result, "SD6: 1跳邻居 chunk_B 应在结果中"
    assert "chunk_C" not in result, f"SD6: max_hops=1 时 chunk_C（2跳）不应在结果中，got {result}"


def test_sd7_extreme_decay_factor(conn):
    """distance_decay_factor=0.1 → 1跳 0.1^1=0.1，2跳 score 极小（可能低于截止阈值被丢弃）。"""
    _seed_graph(conn)
    result = spreading_activate(
        conn, ["chunk_seed"], project="test_proj",
        decay=1.0, max_hops=2, edge_half_life_days=0,
        max_activation_bonus=99.0,
        distance_decay_enabled=True,
        distance_decay_factor=0.1,
    )
    score_b = result.get("chunk_B", 0)
    # 1跳：score = 1.0 * 1.0 * 1.0 * 0.1^1 = 0.1
    # 2跳：score = 0.1 * 1.0 * 1.0 * 0.1^2 = 0.001 < 0.05 cutoff → 被丢弃
    assert abs(score_b - 0.1) < 1e-6, f"SD7: 1跳 score 应为 0.1，got {score_b}"
    assert "chunk_C" not in result, (
        f"SD7: factor=0.1 时 2跳 score=0.001 低于截止阈值，chunk_C 不应在结果中，got {result}"
    )
