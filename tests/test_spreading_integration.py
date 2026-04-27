"""
test_spreading_integration.py — 迭代310：Spreading Activation 端到端集成测试

验证 entity_map 自动填充 + spreading_activate 实际接入检索流程两条路径：
  1. insert_chunk 时自动关联 entity_map（chunk summary 含 entity 名 → 写入映射）
  2. insert_edge 时反向关联 entity_map（新 edge 两端 entity → 查 chunks 写入映射）
  3. spreading_activate 能从 entity_map 找到实际 chunk 并扩散激活
  4. retriever._spreading_activate 接口正常调用（不抛异常，返回 dict）
"""
import sys
import os
import sqlite3
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "hooks"))

import tmpfs  # noqa

from store_vfs import (
    open_db, ensure_schema, insert_chunk, insert_edge,
    spreading_activate,
)


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _make_chunk(cid, summary, chunk_type="decision", importance=0.7, project="test"):
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    return {
        "id": cid,
        "created_at": now,
        "updated_at": now,
        "project": project,
        "source_session": "s1",
        "chunk_type": chunk_type,
        "info_class": "world",
        "content": f"[{chunk_type}] {summary}",
        "summary": summary,
        "tags": [chunk_type, project],
        "importance": importance,
        "retrievability": 0.5,
        "last_accessed": now,
        "access_count": 0,
        "oom_adj": 0,
        "lru_gen": 0,
        "stability": 1.0,
        "raw_snippet": "",
    }


# ─── 测试 1：insert_chunk 时自动填充 entity_map ─────────────────────────────

def test_insert_chunk_auto_entity_map(conn):
    """
    先有 entity_edges，再 insert_chunk：
    chunk summary 包含 entity 名 → insert_chunk 后 entity_map 自动建立。
    """
    # 先插入 edge（让 entity 先存在于 entity_edges）
    insert_edge(conn, "retriever", "depends_on", "store", project="test", confidence=0.9)

    # 插入一个 summary 包含 "retriever" 的 chunk
    insert_chunk(conn, _make_chunk("c_retriever", "retriever 检索主流程实现"))

    row = conn.execute(
        "SELECT chunk_id FROM entity_map WHERE entity_name='retriever' AND project='test'"
    ).fetchone()
    assert row is not None, "entity_map 应自动关联 chunk: c_retriever"
    assert row[0] == "c_retriever"


# ─── 测试 2：insert_edge 时反向填充 entity_map ──────────────────────────────

def test_insert_edge_reverse_entity_map(conn):
    """
    先有 chunk，再 insert_edge：
    edge 两端 entity 名出现在 chunk summary → insert_edge 后 entity_map 反向建立。
    """
    # 先插入 chunk
    insert_chunk(conn, _make_chunk("c_store", "store 持久化层设计"))
    conn.commit()

    # 再插入 edge（store 出现在 entity_edges 后，反向关联已有 chunk）
    insert_edge(conn, "retriever", "depends_on", "store", project="test", confidence=0.8)

    row = conn.execute(
        "SELECT chunk_id FROM entity_map WHERE entity_name='store' AND project='test'"
    ).fetchone()
    assert row is not None, "insert_edge 后应反向关联 store chunk"
    assert row[0] == "c_store"


# ─── 测试 3：端到端扩散激活 ─────────────────────────────────────────────────

def test_end_to_end_spreading(conn):
    """
    完整端到端：
    1. insert_chunk for retriever + store
    2. insert_edge retriever→store
    3. entity_map 自动建立（通过 insert_edge 反向关联）
    4. spreading_activate 从 retriever chunk 扩散到 store chunk
    """
    insert_chunk(conn, _make_chunk("c_retriever", "retriever 检索模块"))
    insert_chunk(conn, _make_chunk("c_store", "store 存储模块"))
    conn.commit()

    insert_edge(conn, "retriever", "depends_on", "store", project="test", confidence=1.0)
    conn.commit()

    result = spreading_activate(
        conn, ["c_retriever"], project="test",
        decay=0.7, max_hops=1, max_activation_bonus=1.0,
    )

    assert "c_store" in result, \
        f"store chunk 应通过 entity_edges 被激活，got keys={list(result.keys())}"
    assert 0.5 <= result["c_store"] <= 0.8, \
        f"激活分应约 0.7（decay=0.7, conf=1.0），got {result['c_store']}"


# ─── 测试 4：entity_map 无关联时 spreading 返回空 ──────────────────────────

def test_no_entity_map_empty_result(conn):
    """
    chunk 没有 entity_map 绑定时，spreading_activate 返回空 dict。
    """
    insert_chunk(conn, _make_chunk("c_orphan", "completely unrelated chunk xyz"))
    insert_edge(conn, "retriever", "depends_on", "store", project="test")
    conn.commit()

    result = spreading_activate(conn, ["c_orphan"], project="test", decay=0.7, max_hops=2)
    assert result == {}, f"孤立 chunk 无 entity 关联，应返回空，got {result}"


# ─── 测试 5：_spreading_activate retriever 接口不抛异常 ─────────────────────

def test_retriever_interface_no_exception(conn):
    """
    retriever._spreading_activate 接口正常调用：不抛异常，返回 dict。
    """
    from retriever import _spreading_activate
    result = _spreading_activate(conn, ["nonexistent_id"], project="test")
    assert isinstance(result, dict), "应返回 dict"


# ─── 测试 6：多 chunk 对应同一 entity ──────────────────────────────────────

def test_multiple_chunks_same_entity(conn):
    """
    entity_map PRIMARY KEY (entity_name, project) — 一个 entity 对应最新写入的 chunk。
    insert_chunk 自动更新映射（INSERT OR REPLACE）。
    """
    insert_edge(conn, "bm25", "used_by", "retriever", project="test")
    conn.commit()

    insert_chunk(conn, _make_chunk("c1", "bm25 原始算法实现"))
    insert_chunk(conn, _make_chunk("c2", "bm25 优化版本 v2"))
    conn.commit()

    rows = conn.execute(
        "SELECT chunk_id FROM entity_map WHERE entity_name='bm25' AND project='test'"
    ).fetchall()
    # 只有一条记录（最后一次 INSERT OR REPLACE 的结果）
    assert len(rows) == 1, f"entity_map 应只保留最新映射，got {[r[0] for r in rows]}"
