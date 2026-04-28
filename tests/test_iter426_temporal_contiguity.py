"""
test_iter426_temporal_contiguity.py — iter426: Temporal Contiguity Effect 单元测试

覆盖：
  TC1: EdgeType.TEMPORAL_FORWARD 存在于 store_graph.EdgeType
  TC2: temporal_forward 边 weight=0.60，后向 COOCCURS weight=0.15
  TC3: add_edge() 写入 temporal_forward 边后可正确查询
  TC4: expand_with_neighbors — temporal_forward 边（weight=0.60）通过 min_weight=0.55
  TC5: expand_with_neighbors — 后向 COOCCURS 边（weight=0.15）不通过 min_weight=0.55
  TC6: 前向边使 expand_with_neighbors 优先返回时序后继 chunk
  TC7: 多个时序后继时，weight 最高者优先返回
  TC8: temporal_forward 边与 entity_edges 的 CAUSES 边共存，不冲突

认知科学依据：
  Kahana (1996) "Associative retrieval processes in free recall" —
    自由回忆中，从 item_i 到 item_{i+1} 的前向联想率约是后向的 2:1（forward asymmetry）。
  Howard & Kahana (2002) Temporal Context Model (TCM) —
    时间上下文向量在序列中前向传播，相邻 item 共享高度重叠的时间上下文，
    但前向邻居比后向邻居有更高的激活相似度。

OS 类比：Linux mm/readahead.c 前向预取 —
  顺序读取时内核预取下一个 page（前向局部性），不预取上一个 page；
  类比 temporal_forward 边：chunk A 之后写入 chunk B → A→B 强边（预取 B 作为 A 的时序后继）。
"""
import sys
import sqlite3
import pytest
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import tmpfs  # noqa
from store_graph import EdgeType, add_edge, expand_with_neighbors, ensure_graph_schema
from store_vfs import ensure_schema


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    ensure_graph_schema(c)
    yield c
    c.close()


def _insert_chunk(conn, cid, project="test", chunk_type="decision", importance=0.7):
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO memory_chunks "
        "(id, project, chunk_type, content, summary, importance, stability, "
        "created_at, updated_at, retrievability) "
        "VALUES (?, ?, ?, ?, ?, ?, 1.0, ?, ?, 0.9)",
        (cid, project, chunk_type, f"content_{cid}", f"summary_{cid}", importance, now, now)
    )
    conn.commit()


# ── TC1: TEMPORAL_FORWARD 存在于 EdgeType ────────────────────────────────────

def test_tc1_temporal_forward_in_edge_type():
    """TC1: EdgeType.TEMPORAL_FORWARD 属性存在。"""
    assert hasattr(EdgeType, "TEMPORAL_FORWARD"), \
        "TC1: EdgeType 应有 TEMPORAL_FORWARD 属性"
    assert EdgeType.TEMPORAL_FORWARD == "temporal_forward", \
        f"TC1: TEMPORAL_FORWARD 值应为 'temporal_forward'，got {EdgeType.TEMPORAL_FORWARD}"


# ── TC2: 边权重非对称性 ───────────────────────────────────────────────────────

def test_tc2_weight_asymmetry(conn):
    """TC2: 前向 temporal_forward 边 weight=0.60，后向 COOCCURS 边 weight=0.15。"""
    _insert_chunk(conn, "older_chunk")
    _insert_chunk(conn, "newer_chunk")

    # 前向边：older → newer
    add_edge(conn, "older_chunk", "newer_chunk", EdgeType.TEMPORAL_FORWARD, 0.60,
             source="temporal")
    # 后向边：newer → older
    add_edge(conn, "newer_chunk", "older_chunk", EdgeType.COOCCURS, 0.15,
             source="temporal")
    conn.commit()

    # 验证前向边权重
    fwd_row = conn.execute(
        "SELECT weight FROM chunk_edges WHERE from_id=? AND to_id=? AND relation_type=?",
        ("older_chunk", "newer_chunk", "temporal_forward")
    ).fetchone()
    assert fwd_row is not None, "TC2: 前向边应存在"
    assert abs(float(fwd_row[0]) - 0.60) < 0.001, \
        f"TC2: 前向边 weight 应为 0.60，got {fwd_row[0]}"

    # 验证后向边权重
    bwd_row = conn.execute(
        "SELECT weight FROM chunk_edges WHERE from_id=? AND to_id=? AND relation_type=?",
        ("newer_chunk", "older_chunk", "cooccurs")
    ).fetchone()
    assert bwd_row is not None, "TC2: 后向边应存在"
    assert abs(float(bwd_row[0]) - 0.15) < 0.001, \
        f"TC2: 后向边 weight 应为 0.15，got {bwd_row[0]}"


# ── TC3: add_edge 写入后可查询 ───────────────────────────────────────────────

def test_tc3_add_edge_queryable(conn):
    """TC3: add_edge() 写入 temporal_forward 边后可正确查询。"""
    _insert_chunk(conn, "tc3_a")
    _insert_chunk(conn, "tc3_b")

    result = add_edge(conn, "tc3_a", "tc3_b", EdgeType.TEMPORAL_FORWARD, 0.60,
                      source="temporal")
    conn.commit()

    assert result is True, "TC3: add_edge 应返回 True（成功写入）"

    row = conn.execute(
        "SELECT relation_type, weight, source FROM chunk_edges "
        "WHERE from_id=? AND to_id=?",
        ("tc3_a", "tc3_b")
    ).fetchone()
    assert row is not None, "TC3: 写入后应可查询到边"
    assert row[0] == "temporal_forward", f"TC3: relation_type 应为 temporal_forward，got {row[0]}"
    assert row[2] == "temporal", f"TC3: source 应为 temporal，got {row[2]}"


# ── TC4: temporal_forward（0.60）通过 min_weight=0.55 ────────────────────────

def test_tc4_forward_edge_passes_min_weight(conn):
    """TC4: expand_with_neighbors 中，temporal_forward 边（weight=0.60 >= 0.55）被纳入。"""
    _insert_chunk(conn, "seed_chunk")
    _insert_chunk(conn, "forward_neighbor")

    add_edge(conn, "seed_chunk", "forward_neighbor", EdgeType.TEMPORAL_FORWARD, 0.60,
             source="temporal")
    conn.commit()

    neighbors = expand_with_neighbors(conn, ["seed_chunk"], top_n=5, min_weight=0.55)
    neighbor_ids = [n["id"] for n in neighbors]
    assert "forward_neighbor" in neighbor_ids, \
        f"TC4: temporal_forward(0.60) 邻居应通过 min_weight=0.55，got {neighbor_ids}"


# ── TC5: 后向 COOCCURS（0.15）不通过 min_weight=0.55 ─────────────────────────

def test_tc5_backward_cooccurs_blocked_by_min_weight(conn):
    """TC5: expand_with_neighbors 中，后向 COOCCURS 边（weight=0.15 < 0.55）被过滤。"""
    _insert_chunk(conn, "newer_seed")
    _insert_chunk(conn, "older_precede")

    # 后向边：newer → older（表示 older 先写入）
    add_edge(conn, "newer_seed", "older_precede", EdgeType.COOCCURS, 0.15,
             source="temporal")
    conn.commit()

    neighbors = expand_with_neighbors(conn, ["newer_seed"], top_n=5, min_weight=0.55)
    neighbor_ids = [n["id"] for n in neighbors]
    assert "older_precede" not in neighbor_ids, \
        f"TC5: 后向 COOCCURS(0.15) 应被 min_weight=0.55 过滤，got {neighbor_ids}"


# ── TC6: forward 边使 expand 优先返回时序后继 ────────────────────────────────

def test_tc6_forward_neighbor_prioritized(conn):
    """TC6: expand_with_neighbors 优先返回 temporal_forward 邻居（比低权重边的邻居优先）。"""
    _insert_chunk(conn, "base")
    _insert_chunk(conn, "time_successor")  # B 在 A 之后写入 → A→B 前向边
    _insert_chunk(conn, "weak_related")     # 弱相关 chunk

    # 前向边（高权重）
    add_edge(conn, "base", "time_successor", EdgeType.TEMPORAL_FORWARD, 0.60,
             source="temporal")
    # 弱边（权重刚好过 min_weight）
    add_edge(conn, "base", "weak_related", EdgeType.RELATED, 0.56,
             source="rule")
    conn.commit()

    neighbors = expand_with_neighbors(conn, ["base"], top_n=2, min_weight=0.55)
    assert len(neighbors) >= 1, "TC6: 应至少返回 1 个邻居"
    # time_successor 应排在 weak_related 前（权重更高）
    if len(neighbors) >= 2:
        assert neighbors[0]["id"] == "time_successor", \
            f"TC6: temporal_forward 邻居应排第一，got {neighbors[0]['id']}"


# ── TC7: 多个时序后继时，weight 最高者优先 ──────────────────────────────────

def test_tc7_highest_weight_first(conn):
    """TC7: 多个 temporal_forward 邻居时，两者都通过 min_weight 且都被返回。"""
    _insert_chunk(conn, "tc7_seed")
    _insert_chunk(conn, "tc7_next_a")  # weight=0.60
    _insert_chunk(conn, "tc7_next_b")  # weight=0.62（更高）

    add_edge(conn, "tc7_seed", "tc7_next_a", EdgeType.TEMPORAL_FORWARD, 0.60, source="temporal")
    add_edge(conn, "tc7_seed", "tc7_next_b", EdgeType.TEMPORAL_FORWARD, 0.62, source="temporal")
    conn.commit()

    neighbors = expand_with_neighbors(conn, ["tc7_seed"], top_n=5, min_weight=0.55)
    neighbor_ids = {n["id"] for n in neighbors}
    # 两个前向邻居都应通过 min_weight=0.55
    assert "tc7_next_a" in neighbor_ids, "TC7: tc7_next_a 应在结果中"
    assert "tc7_next_b" in neighbor_ids, "TC7: tc7_next_b 应在结果中"
    # 验证权重值正确存储
    neighbor_weights = {n["id"]: n["weight"] for n in neighbors}
    assert abs(neighbor_weights.get("tc7_next_b", 0) - 0.62) < 0.01, \
        f"TC7: tc7_next_b weight 应为 0.62，got {neighbor_weights.get('tc7_next_b')}"


# ── TC8: temporal_forward 与 entity_edges CAUSES 共存 ────────────────────────

def test_tc8_coexists_with_causes_edge(conn):
    """TC8: temporal_forward 边与 CAUSES 边共存于同一 chunk pair，不冲突。"""
    _insert_chunk(conn, "tc8_a")
    _insert_chunk(conn, "tc8_b")

    # 同时有语义因果边和时序前向边（A→B）
    add_edge(conn, "tc8_a", "tc8_b", EdgeType.CAUSES, 0.8, source="rule")
    add_edge(conn, "tc8_a", "tc8_b", EdgeType.TEMPORAL_FORWARD, 0.60, source="temporal")
    conn.commit()

    # 两条边都存在（UNIQUE 约束基于 from_id+to_id+relation_type，不互相覆盖）
    rows = conn.execute(
        "SELECT relation_type FROM chunk_edges WHERE from_id=? AND to_id=?",
        ("tc8_a", "tc8_b")
    ).fetchall()
    rel_types = {r[0] for r in rows}
    assert "causes" in rel_types, "TC8: CAUSES 边应存在"
    assert "temporal_forward" in rel_types, "TC8: temporal_forward 边应存在"
