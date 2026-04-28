"""
test_chunk_graph.py — iter366 知识图谱关联扩散测试

覆盖：
  G1: add_edge — 基本写入 + 幂等（重复不新增行）
  G2: add_edge — weight 更新取最大值
  G3: add_cooccurrence_edges — N 个 chunk 建立双向共现边
  G4: infer_edges_from_summaries — decision supersedes excluded_path
  G5: expand_with_neighbors — 从种子找 1-hop 邻居
  G6: expand_with_neighbors — 不返回种子本身
  G7: expand_with_neighbors — exclude_types 过滤
  G8: graph_stats — 统计边数和类型
  G9: min_weight 过滤 — 低权重边不被扩散
"""
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


@pytest.fixture()
def tmpdb(tmp_path):
    db_path = tmp_path / "test_store.db"
    os.environ["MEMORY_OS_DB"] = str(db_path)
    os.environ["MEMORY_OS_DIR"] = str(tmp_path)
    yield db_path
    os.environ.pop("MEMORY_OS_DB", None)
    os.environ.pop("MEMORY_OS_DIR", None)


@pytest.fixture()
def conn(tmpdb):
    from store_vfs import open_db
    from store_graph import ensure_graph_schema
    c = open_db(tmpdb)
    ensure_graph_schema(c)
    # 创建 memory_chunks 表（expand_with_neighbors 需要 JOIN）
    c.execute("""
        CREATE TABLE IF NOT EXISTS memory_chunks (
            id TEXT PRIMARY KEY, summary TEXT, chunk_type TEXT,
            importance REAL DEFAULT 0.5,
            created_at TEXT, updated_at TEXT, project TEXT,
            source_session TEXT, content TEXT, tags TEXT,
            retrievability REAL DEFAULT 1.0, last_accessed TEXT,
            feishu_url TEXT, access_count INTEGER DEFAULT 0,
            oom_adj INTEGER DEFAULT 0, lru_gen INTEGER DEFAULT 0
        )
    """)
    c.commit()
    yield c
    c.close()


def _insert_chunk(conn, chunk_id, chunk_type="decision", summary="summary"):
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT OR IGNORE INTO memory_chunks
        (id, chunk_type, summary, importance, created_at, updated_at, project,
         source_session, content)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, (chunk_id, chunk_type, summary, 0.7, now, now, "p", "s", summary))
    conn.commit()


# ── G1: add_edge 基本 + 幂等 ─────────────────────────────────────────────────

def test_g1_add_edge_basic(conn):
    from store_graph import add_edge, EdgeType
    result = add_edge(conn, "c1", "c2", EdgeType.RELATED, 0.8)
    assert result is True  # 新建
    row = conn.execute(
        "SELECT weight FROM chunk_edges WHERE from_id='c1' AND to_id='c2'"
    ).fetchone()
    assert row is not None
    assert abs(row[0] - 0.8) < 0.01


def test_g1_add_edge_idempotent(conn):
    from store_graph import add_edge, EdgeType
    add_edge(conn, "c1", "c2", EdgeType.RELATED, 0.8)
    result = add_edge(conn, "c1", "c2", EdgeType.RELATED, 0.8)
    assert result is False  # 已存在
    count = conn.execute("SELECT COUNT(*) FROM chunk_edges").fetchone()[0]
    assert count == 1


# ── G2: weight 更新取最大值 ───────────────────────────────────────────────────

def test_g2_weight_update_takes_max(conn):
    from store_graph import add_edge, EdgeType
    add_edge(conn, "c1", "c2", EdgeType.RELATED, 0.5)
    add_edge(conn, "c1", "c2", EdgeType.RELATED, 0.9)
    row = conn.execute(
        "SELECT weight FROM chunk_edges WHERE from_id='c1' AND to_id='c2'"
    ).fetchone()
    assert abs(row[0] - 0.9) < 0.01


def test_g2_weight_does_not_decrease(conn):
    from store_graph import add_edge, EdgeType
    add_edge(conn, "c1", "c2", EdgeType.RELATED, 0.8)
    add_edge(conn, "c1", "c2", EdgeType.RELATED, 0.3)
    row = conn.execute(
        "SELECT weight FROM chunk_edges WHERE from_id='c1' AND to_id='c2'"
    ).fetchone()
    assert abs(row[0] - 0.8) < 0.01  # 未降低


# ── G3: add_cooccurrence_edges ────────────────────────────────────────────────

def test_g3_cooccurrence_edges(conn):
    from store_graph import add_cooccurrence_edges
    count = add_cooccurrence_edges(conn, ["c1", "c2", "c3"], weight=0.5)
    # 3 个 chunk → 3 对 × 2 方向 = 6 条边（新建）
    assert count == 6


def test_g3_cooccurrence_single_no_edges(conn):
    from store_graph import add_cooccurrence_edges
    count = add_cooccurrence_edges(conn, ["c1"], weight=0.5)
    assert count == 0


# ── G4: infer_edges_from_summaries ────────────────────────────────────────────

def test_g4_decision_supersedes_excluded(conn):
    from store_graph import infer_edges_from_summaries, EdgeType
    chunks = [
        {"id": "d1", "summary": "选择 Docker 部署", "chunk_type": "decision"},
        {"id": "e1", "summary": "放弃直接 systemd 部署", "chunk_type": "excluded_path"},
    ]
    count = infer_edges_from_summaries(conn, chunks)
    assert count > 0
    edge = conn.execute(
        "SELECT relation_type FROM chunk_edges WHERE from_id='d1' AND to_id='e1'"
    ).fetchone()
    assert edge is not None
    assert edge[0] == EdgeType.SUPERSEDES


# ── G5: expand_with_neighbors ────────────────────────────────────────────────

def test_g5_expand_finds_neighbors(conn):
    from store_graph import add_edge, expand_with_neighbors, EdgeType
    _insert_chunk(conn, "seed", "decision", "种子 chunk")
    _insert_chunk(conn, "neighbor1", "reasoning_chain", "相关推理 1")
    _insert_chunk(conn, "neighbor2", "decision", "相关决策 2")
    add_edge(conn, "seed", "neighbor1", EdgeType.CAUSES, 0.8)
    add_edge(conn, "seed", "neighbor2", EdgeType.RELATED, 0.7)
    neighbors = expand_with_neighbors(conn, ["seed"], top_n=3)
    neighbor_ids = {n["id"] for n in neighbors}
    assert "neighbor1" in neighbor_ids or "neighbor2" in neighbor_ids


# ── G6: 不返回种子本身 ────────────────────────────────────────────────────────

def test_g6_seed_not_in_result(conn):
    from store_graph import add_edge, expand_with_neighbors, EdgeType
    _insert_chunk(conn, "a", "decision", "A")
    _insert_chunk(conn, "b", "decision", "B")
    add_edge(conn, "a", "b", EdgeType.RELATED, 0.9)
    neighbors = expand_with_neighbors(conn, ["a", "b"], top_n=5)
    neighbor_ids = {n["id"] for n in neighbors}
    assert "a" not in neighbor_ids
    assert "b" not in neighbor_ids


# ── G7: exclude_types 过滤 ────────────────────────────────────────────────────

def test_g7_exclude_types(conn):
    from store_graph import add_edge, expand_with_neighbors, EdgeType
    _insert_chunk(conn, "seed", "decision", "种子")
    _insert_chunk(conn, "stub", "entity_stub", "实体存根")
    add_edge(conn, "seed", "stub", EdgeType.RELATED, 0.9)
    neighbors = expand_with_neighbors(
        conn, ["seed"], top_n=5,
        exclude_types=["entity_stub"]
    )
    assert not any(n["id"] == "stub" for n in neighbors)


# ── G8: graph_stats ───────────────────────────────────────────────────────────

def test_g8_graph_stats(conn):
    from store_graph import add_edge, graph_stats, EdgeType
    add_edge(conn, "a", "b", EdgeType.CAUSES, 0.8)
    add_edge(conn, "b", "c", EdgeType.RELATED, 0.7)
    stats = graph_stats(conn)
    assert stats["total_edges"] == 2
    assert stats["by_type"].get(EdgeType.CAUSES) == 1


# ── G9: min_weight 过滤 ───────────────────────────────────────────────────────

def test_g9_min_weight_filter(conn):
    from store_graph import add_edge, expand_with_neighbors, EdgeType
    _insert_chunk(conn, "seed", "decision", "种子")
    _insert_chunk(conn, "weak", "decision", "弱关联")
    _insert_chunk(conn, "strong", "decision", "强关联")
    add_edge(conn, "seed", "weak", EdgeType.RELATED, 0.3)    # 低权重
    add_edge(conn, "seed", "strong", EdgeType.RELATED, 0.9)  # 高权重
    neighbors = expand_with_neighbors(conn, ["seed"], top_n=5, min_weight=0.5)
    ids = {n["id"] for n in neighbors}
    assert "strong" in ids
    assert "weak" not in ids
