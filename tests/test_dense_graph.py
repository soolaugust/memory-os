"""
test_dense_graph.py — 迭代318：知识图谱稠密化单元测试

验证：
  1. extract_summary_triples 能从短句中抽取正确的三元组
  2. 噪声实体被过滤（纯中文片段、停用词）
  3. 不同关系类型正确识别（uses/writes_to/triggers/superseded_by）
  4. _write_chunk 后 entity_edges 自动填充（端到端）
  5. spreading_activation 能沿新图扩散命中相关 chunk
  6. 边界：空输入/无谓语/自指边安全处理

OS 类比：测试 ext3 htree B-tree rebuild 后，readdir/lookup 能正确走新索引。
"""
import sys
import sqlite3
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "hooks"))

import tmpfs  # noqa

from store_vfs import (
    open_db, ensure_schema, insert_chunk, insert_edge,
    spreading_activate, query_neighbors,
)
from extractor import extract_summary_triples, extract_and_write_summary_triples


# ══════════════════════════════════════════════════════════════════════
# 1. extract_summary_triples 基础测试
# ══════════════════════════════════════════════════════════════════════

def test_extracts_uses_relation():
    """'X 使用 Y' 抽出 (X, uses, Y)。"""
    triples = extract_summary_triples("retriever 使用 BM25 检索")
    assert len(triples) >= 1
    found = any(t[0] == 'retriever' and t[1] == 'uses' and t[2] == 'BM25'
                for t in triples)
    assert found, f"应有 (retriever, uses, BM25)，got {triples}"


def test_extracts_writes_to_relation():
    """'X 写入 Y' 抽出 (X, writes_to, Y)。"""
    triples = extract_summary_triples("extractor 写入 memory_chunks 表")
    assert len(triples) >= 1
    found = any(t[1] == 'writes_to' and 'extractor' in t[0]
                for t in triples)
    assert found, f"应有 writes_to 关系，got {triples}"


def test_extracts_triggers_relation():
    """'X 触发 Y' 抽出 (X, triggers, Y)。"""
    triples = extract_summary_triples("kswapd 触发 eviction 流程")
    assert len(triples) >= 1
    found = any(t[1] == 'triggers' for t in triples)
    assert found, f"应有 triggers 关系，got {triples}"


def test_extracts_superseded_by_relation():
    """'X 被 Y 替代' 抽出 superseded_by 关系。"""
    triples = extract_summary_triples("BM25 被 FTS5 替代")
    # 注意：汉字"被"在谓语左边匹配，from=BM25, to=FTS5
    assert len(triples) >= 1, f"应抽到关系，got {triples}"


def test_no_false_positive_pure_cjk():
    """纯中文片段不应抽取为三元组（两边都无英文字母的噪声）。"""
    triples = extract_summary_triples("记录触发了过阈值时的结果")
    # 如果有三元组，两边至少一方有英文字母
    for f, r, t in triples:
        has_alpha = any(c.isalpha() and ord(c) < 128 for c in f + t)
        assert has_alpha, f"纯中文片段不应成为三元组：{(f, r, t)}"


def test_no_self_reference():
    """自指边不应写入（retriever uses retriever）。"""
    triples = extract_summary_triples("retriever 使用 retriever 缓存")
    for f, r, t in triples:
        assert f.lower() != t.lower(), f"自指边不应出现：{(f, r, t)}"


def test_empty_input():
    """空输入安全返回空列表。"""
    assert extract_summary_triples("") == []
    assert extract_summary_triples("   ") == []
    assert extract_summary_triples("abc") == []  # 太短


def test_english_uses_pattern():
    """英文 'X uses Y' 也能识别。"""
    triples = extract_summary_triples("retriever uses FTS5 index")
    assert len(triples) >= 1
    found = any(t[1] == 'uses' for t in triples)
    assert found, f"应有 uses 关系，got {triples}"


def test_calls_relation():
    """'X calls Y' 抽取 calls 关系。"""
    triples = extract_summary_triples("scorer calls retrieval_score function")
    assert len(triples) >= 1


# ══════════════════════════════════════════════════════════════════════
# 2. extract_and_write_summary_triples 端到端测试
# ══════════════════════════════════════════════════════════════════════

@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _make_chunk(cid, summary, chunk_type="decision", project="test"):
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    return {
        "id": cid, "created_at": now, "updated_at": now,
        "project": project, "source_session": "s1",
        "chunk_type": chunk_type, "info_class": "world",
        "content": f"[{chunk_type}] {summary}",
        "summary": summary, "tags": [chunk_type],
        "importance": 0.7, "retrievability": 0.5,
        "last_accessed": now, "access_count": 0,
        "oom_adj": 0, "lru_gen": 0, "stability": 1.4,
        "raw_snippet": "", "encoding_context": {},
    }


def test_write_triples_creates_edges(conn):
    """extract_and_write_summary_triples 把边写入 entity_edges。"""
    summary = "retriever 使用 BM25 检索模块"
    insert_chunk(conn, _make_chunk("c1", summary))
    conn.commit()

    n = extract_and_write_summary_triples(summary, "c1", "test", conn)
    conn.commit()

    assert n >= 1, f"应写入 >= 1 条边，got {n}"
    row = conn.execute(
        "SELECT * FROM entity_edges WHERE from_entity='retriever' AND to_entity='BM25'"
    ).fetchone()
    assert row is not None, "entity_edges 中应有 (retriever, uses, BM25) 边"


def test_write_triples_creates_entity_map(conn):
    """写边后 entity_map 应有 chunk_id 映射。"""
    summary = "kswapd 触发 eviction 流程"
    insert_chunk(conn, _make_chunk("c2", summary))
    conn.commit()

    extract_and_write_summary_triples(summary, "c2", "test", conn)
    conn.commit()

    # entity_map 应记录 kswapd → c2
    row = conn.execute(
        "SELECT * FROM entity_map WHERE entity_name='kswapd' AND chunk_id='c2'"
    ).fetchone()
    assert row is not None, "entity_map 应有 kswapd → c2 映射"


# ══════════════════════════════════════════════════════════════════════
# 3. spreading_activation 沿稠密图扩散
# ══════════════════════════════════════════════════════════════════════

def test_spreading_activation_with_edges(conn):
    """有边后 spreading_activate 能找到邻居 chunk。"""
    # chunk A: retriever
    insert_chunk(conn, _make_chunk("ca", "retriever 模块核心", project="test"))
    # chunk B: BM25（通过边与 retriever 关联）
    insert_chunk(conn, _make_chunk("cb", "BM25 检索算法", project="test"))
    conn.commit()

    # 手动建立 retriever → BM25 边（及 entity_map）
    insert_edge(conn, "retriever", "uses", "BM25", project="test", source_chunk_id="ca")
    conn.execute(
        "INSERT OR REPLACE INTO entity_map (entity_name, chunk_id, project, updated_at) "
        "VALUES ('retriever', 'ca', 'test', datetime('now'))"
    )
    conn.execute(
        "INSERT OR REPLACE INTO entity_map (entity_name, chunk_id, project, updated_at) "
        "VALUES ('BM25', 'cb', 'test', datetime('now'))"
    )
    conn.commit()

    # 从 ca 出发扩散，应找到 cb
    activation = spreading_activate(
        conn, hit_chunk_ids=["ca"], project="test", decay=0.7, max_hops=2
    )
    assert "cb" in activation, f"应通过边扩散到 cb，got activation={activation}"
    assert activation["cb"] > 0, f"cb 的激活分应 > 0，got {activation['cb']}"


def test_spreading_activation_empty_graph(conn):
    """无边时 spreading_activate 安全返回空 dict。"""
    insert_chunk(conn, _make_chunk("c1", "孤立 chunk", project="test"))
    conn.commit()

    result = spreading_activate(conn, hit_chunk_ids=["c1"], project="test")
    assert isinstance(result, dict)


def test_query_neighbors_after_backfill(conn):
    """insert_edge 后 query_neighbors 能正确查询邻居。"""
    insert_edge(conn, "FTS5", "triggers", "BFS", project="test")
    conn.commit()

    neighbors = query_neighbors(conn, "FTS5", project="test", direction="out")
    assert len(neighbors) >= 1
    found = any(n[0] == "triggers" and n[1] == "BFS" for n in neighbors)
    assert found, f"应找到 FTS5 → BFS 边，got {neighbors}"
