"""
test_entity_graph.py — 迭代304：知识图谱关系边单元测试

测试点：
  1. ensure_schema 后 entity_edges 表存在
  2. insert_edge 幂等（重复插入不报错，confidence 更新）
  3. query_neighbors 能按 from/to/both 方向查询
  4. _extract_entity_relations 从 "memory-os 使用 SQLite 作为后端存储" 提取到 (memory-os, uses, SQLite)
  5. _extract_entity_relations 从 "retriever.py 依赖 bm25.py" 提取到 (retriever.py, depends_on, bm25.py)

OS 类比：内核模块依赖图测试 — insmod / modprobe --show-depends 的正确性验证。
"""
import sys
import os
import sqlite3
import tempfile
from pathlib import Path

# conftest.py 已通过 tmpfs 设置了 MEMORY_OS_DIR / MEMORY_OS_DB 环境变量

import pytest
import sys
sys.path.insert(0, str(Path(__file__).parent))

from store_vfs import (
    open_db,
    ensure_schema,
    insert_edge,
    query_neighbors,
)

# ─── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def conn():
    """每个测试用一个独立的 in-memory SQLite 连接。"""
    c = sqlite3.connect(":memory:")
    ensure_schema(c)
    yield c
    c.close()


# ─── 测试 1：ensure_schema 后 entity_edges 表存在 ────────────────────────────

def test_schema_entity_edges_table_exists(conn):
    """
    迭代304：ensure_schema() 必须创建 entity_edges 表。
    OS 类比：fill_super() 挂载后 /proc/fs/ext4/<dev>/ 目录可读。
    """
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='entity_edges'"
    ).fetchall()
    assert rows, "entity_edges 表不存在"


def test_schema_entity_edges_columns(conn):
    """entity_edges 表包含必要列。"""
    info = {row[1]: row[2] for row in conn.execute("PRAGMA table_info(entity_edges)")}
    required = {'id', 'from_entity', 'relation', 'to_entity', 'project',
                'source_chunk_id', 'confidence', 'created_at'}
    assert required <= set(info.keys()), f"缺少列: {required - set(info.keys())}"


def test_schema_entity_edges_indexes(conn):
    """entity_edges 有 from_entity / to_entity / project 索引。"""
    indexes = {row[1] for row in conn.execute("PRAGMA index_list(entity_edges)")}
    assert any('from' in idx for idx in indexes), f"缺少 from_entity 索引，现有: {indexes}"
    assert any('to' in idx for idx in indexes), f"缺少 to_entity 索引，现有: {indexes}"
    assert any('project' in idx for idx in indexes), f"缺少 project 索引，现有: {indexes}"


# ─── 测试 2：insert_edge 幂等 ─────────────────────────────────────────────────

def test_insert_edge_basic(conn):
    """基础插入并能查到。"""
    eid = insert_edge(conn, "memory-os", "uses", "SQLite",
                      project="test", confidence=0.8)
    assert eid.startswith("ee_"), f"edge_id 格式错误: {eid}"

    row = conn.execute("SELECT * FROM entity_edges WHERE id=?", (eid,)).fetchone()
    assert row is not None, "插入的边查不到"
    # (id, from_entity, relation, to_entity, project, source_chunk_id, confidence, created_at)
    assert row[1] == "memory-os"
    assert row[2] == "uses"
    assert row[3] == "SQLite"
    assert row[6] == 0.8


def test_insert_edge_idempotent_no_error(conn):
    """重复插入相同三元组不报错。"""
    insert_edge(conn, "A", "uses", "B", project="p1", confidence=0.7)
    insert_edge(conn, "A", "uses", "B", project="p1", confidence=0.7)  # 不应抛出
    count = conn.execute(
        "SELECT COUNT(*) FROM entity_edges WHERE from_entity='A' AND relation='uses' AND to_entity='B'"
    ).fetchone()[0]
    assert count == 1, f"幂等失败，期望 1 条边，实际 {count}"


def test_insert_edge_idempotent_confidence_update(conn):
    """重复插入相同三元组时 confidence 被更新为新值。"""
    insert_edge(conn, "X", "depends_on", "Y", confidence=0.6)
    insert_edge(conn, "X", "depends_on", "Y", confidence=0.9)  # 更新 confidence
    row = conn.execute(
        "SELECT confidence FROM entity_edges WHERE from_entity='X' AND relation='depends_on' AND to_entity='Y'"
    ).fetchone()
    assert row is not None
    assert abs(row[0] - 0.9) < 1e-6, f"confidence 未更新，期望 0.9 实际 {row[0]}"


def test_insert_edge_different_relations_are_different_edges(conn):
    """相同 from/to 但不同 relation 应为独立的边。"""
    insert_edge(conn, "A", "uses", "B")
    insert_edge(conn, "A", "depends_on", "B")
    count = conn.execute(
        "SELECT COUNT(*) FROM entity_edges WHERE from_entity='A' AND to_entity='B'"
    ).fetchone()[0]
    assert count == 2, f"期望 2 条边（不同 relation），实际 {count}"


# ─── 测试 3：query_neighbors ──────────────────────────────────────────────────

def test_query_neighbors_out(conn):
    """direction='out' 返回出边（entity 是 from）。"""
    insert_edge(conn, "A", "uses", "B", project="p")
    insert_edge(conn, "A", "depends_on", "C", project="p")
    insert_edge(conn, "X", "uses", "A", project="p")  # A 是 to，不应出现

    results = query_neighbors(conn, "A", project="p", direction="out")
    neighbors = {(r[0], r[1]) for r in results}
    assert ("uses", "B") in neighbors, f"出边 uses→B 未找到，结果: {neighbors}"
    assert ("depends_on", "C") in neighbors, f"出边 depends_on→C 未找到"
    # X→A 不应出现在 out 结果
    neighbor_entities = {r[1] for r in results}
    assert "X" not in neighbor_entities, "入边 X→A 不应出现在 direction=out 结果中"


def test_query_neighbors_in(conn):
    """direction='in' 返回入边（entity 是 to）。"""
    insert_edge(conn, "A", "uses", "B", project="p")
    insert_edge(conn, "X", "uses", "B", project="p")
    insert_edge(conn, "B", "depends_on", "Y", project="p")  # B 是 from，不应出现

    results = query_neighbors(conn, "B", project="p", direction="in")
    neighbors = {(r[0], r[1]) for r in results}
    assert ("uses", "A") in neighbors, f"入边 A→B 未找到"
    assert ("uses", "X") in neighbors, f"入边 X→B 未找到"
    neighbor_entities = {r[1] for r in results}
    assert "Y" not in neighbor_entities, "出边 B→Y 不应出现在 direction=in 结果中"


def test_query_neighbors_both(conn):
    """direction='both'（默认）返回双向邻居。"""
    insert_edge(conn, "A", "uses", "B")
    insert_edge(conn, "C", "depends_on", "A")

    results = query_neighbors(conn, "A", direction="both")
    neighbor_entities = {r[1] for r in results}
    assert "B" in neighbor_entities, "出边目标 B 未找到"
    assert "C" in neighbor_entities, "入边来源 C 未找到"


def test_query_neighbors_project_filter(conn):
    """project 过滤有效。"""
    insert_edge(conn, "A", "uses", "B", project="proj1")
    insert_edge(conn, "A", "uses", "C", project="proj2")

    results = query_neighbors(conn, "A", project="proj1", direction="out")
    neighbors = {r[1] for r in results}
    assert "B" in neighbors
    assert "C" not in neighbors, "project=proj2 的边不应在 proj1 结果中"


def test_query_neighbors_empty(conn):
    """查询不存在的实体返回空列表。"""
    results = query_neighbors(conn, "nonexistent_entity")
    assert results == []


def test_query_neighbors_returns_confidence(conn):
    """返回结果包含 confidence。"""
    insert_edge(conn, "A", "uses", "B", confidence=0.75)
    results = query_neighbors(conn, "A", direction="out")
    assert results, "结果不应为空"
    rel, neighbor, conf = results[0]
    assert abs(conf - 0.75) < 1e-6, f"confidence 不匹配: {conf}"


# ─── 测试 4 & 5：_extract_entity_relations ───────────────────────────────────

# 延迟导入（避免在 conftest 设置前导入）
@pytest.fixture
def extract_fn():
    from hooks.extractor import _extract_entity_relations
    return _extract_entity_relations


def test_extract_uses_chinese(conn, extract_fn):
    """
    测试 4：从中文 "memory-os 使用 SQLite 作为后端存储" 提取 (memory-os, uses, SQLite)。
    """
    text = "memory-os 使用 SQLite 作为后端存储"
    count = extract_fn(text, "test_proj", "sess1", conn)
    assert count > 0, "未提取到任何边"

    rows = conn.execute(
        "SELECT from_entity, relation, to_entity FROM entity_edges "
        "WHERE relation='uses'"
    ).fetchall()
    triples = {(r[0], r[1], r[2]) for r in rows}
    assert ("memory-os", "uses", "SQLite") in triples, \
        f"未找到 (memory-os, uses, SQLite)，实际: {triples}"


def test_extract_depends_on_chinese(conn, extract_fn):
    """
    测试 5：从 "retriever.py 依赖 bm25.py" 提取 (retriever.py, depends_on, bm25.py)。
    """
    text = "retriever.py 依赖 bm25.py"
    count = extract_fn(text, "test_proj", "sess1", conn)
    assert count > 0, "未提取到任何边"

    rows = conn.execute(
        "SELECT from_entity, relation, to_entity FROM entity_edges "
        "WHERE relation='depends_on'"
    ).fetchall()
    triples = {(r[0], r[1], r[2]) for r in rows}
    assert ("retriever.py", "depends_on", "bm25.py") in triples, \
        f"未找到 (retriever.py, depends_on, bm25.py)，实际: {triples}"


def test_extract_uses_english(conn, extract_fn):
    """英文 uses 模式：X uses Y。"""
    text = "The kernel uses ftrace for tracing"
    extract_fn(text, "test_proj", "sess1", conn)
    rows = conn.execute(
        "SELECT from_entity, relation, to_entity FROM entity_edges WHERE relation='uses'"
    ).fetchall()
    triples = {(r[0], r[1], r[2]) for r in rows}
    assert any(t[2] == "ftrace" for t in triples), \
        f"未找到 uses ftrace 的边，实际: {triples}"


def test_extract_implements(conn, extract_fn):
    """implements 模式：X 实现了 Y。"""
    text = "store_vfs.py 实现了 VFS 接口"
    extract_fn(text, "test_proj", "sess1", conn)
    rows = conn.execute(
        "SELECT from_entity, relation, to_entity FROM entity_edges WHERE relation='implements'"
    ).fetchall()
    assert rows, f"未找到 implements 关系边"


def test_extract_idempotent(conn, extract_fn):
    """同一文本调用两次，边数不翻倍（insert_edge 幂等）。"""
    text = "memory-os 使用 SQLite 作为后端存储"
    extract_fn(text, "test_proj", "sess1", conn)
    extract_fn(text, "test_proj", "sess2", conn)
    count = conn.execute(
        "SELECT COUNT(*) FROM entity_edges WHERE from_entity='memory-os' AND relation='uses' AND to_entity='SQLite'"
    ).fetchone()[0]
    assert count == 1, f"幂等失败，期望 1 条边，实际 {count}"


def test_extract_no_self_edge(conn, extract_fn):
    """不应产生自指边（X uses X）。"""
    text = "SQLite 使用 SQLite 存储数据"  # 故意构造自指
    extract_fn(text, "test_proj", "sess1", conn)
    rows = conn.execute(
        "SELECT * FROM entity_edges WHERE from_entity=to_entity"
    ).fetchall()
    assert not rows, f"存在自指边: {rows}"


def test_extract_empty_text(conn, extract_fn):
    """空文本不应写入任何边，也不报错。"""
    count = extract_fn("", "test_proj", "sess1", conn)
    assert count == 0
