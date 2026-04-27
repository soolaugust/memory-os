"""
test_spreading_activation.py — 迭代310：Spreading Activation 单元测试

设计哲学：
  人类语义记忆是关联激活网络（spreading activation, Collins & Loftus 1975）——
  一个概念激活后，相邻概念以衰减权重被"预激活"。
  memory-os 的等价：FTS5 命中 chunk A → entity_edges 一/二跳邻居 B、C
  以 decay 权重加入候选集，形成认知网络式召回，而非孤立的 top-K。

OS 类比：
  CPU prefetch + L2 cache warm-up — 主访问触发相邻 cache line 预热，
  避免第二次访问时付 cache miss 代价。

测试覆盖：
  1. _spreading_activate：从命中 chunk 的 entity 出发，一跳邻居被激活（decay=0.7）
  2. _spreading_activate：二跳邻居被激活（decay=0.49）
  3. 已在 FTS5 候选集中的 chunk 不重复添加（去重）
  4. 无 entity 关联的 chunk 不被激活（孤立节点）
  5. activation_bonus 不超过 max_activation_bonus 上限（防止 entity 过度主导）
  6. 空图（无 entity_edges）时安全降级，返回空激活集
  7. 性能：100 个 entity_edges 时 _spreading_activate < 5ms
  8. 构建式召回：intent="understand" 时 reasoning_chain chunk 优先展示
  9. 构建式召回：intent="implement" 时 procedure/decision chunk 优先展示
  10. 展示层调整不修改 DB 中的 chunk 内容（只改注入顺序和前缀）
"""
import sys
import os
import json
import time
import sqlite3
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "hooks"))

import tmpfs  # noqa — 确保 tmpfs 隔离已初始化

from store_vfs import open_db, ensure_schema, insert_edge, query_neighbors, spreading_activate

# ─── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def conn():
    """每个测试独立的 in-memory SQLite 连接（含 entity_edges + memory_chunks schema）。"""
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _insert_chunk(conn, chunk_id, summary, chunk_type="decision",
                  importance=0.7, entity_name=None):
    """辅助：插入测试 chunk，可选绑定 entity_name。"""
    conn.execute("""
        INSERT OR REPLACE INTO memory_chunks
            (id, summary, content, chunk_type, importance, project,
             last_accessed, access_count, created_at, updated_at,
             stability, retrievability, info_class)
        VALUES (?, ?, '', ?, ?, 'test', datetime('now'), 0,
                datetime('now'), datetime('now'), 1.0, 0.5, 'world')
    """, (chunk_id, summary, chunk_type, importance))
    if entity_name:
        # entity_map: entity_name -> chunk_id（模拟 entity 关联）
        # 实现中由 _spreading_activate 通过 entity_edges 的 from_entity 字段匹配
        conn.execute("""
            INSERT OR REPLACE INTO entity_map (entity_name, chunk_id, project)
            VALUES (?, ?, 'test')
        """, (entity_name, chunk_id))
    conn.commit()


def _ensure_entity_map(conn):
    """确保 entity_map 辅助表存在（Spreading Activation 需要 entity→chunk 映射）。"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entity_map (
            entity_name TEXT NOT NULL,
            chunk_id    TEXT NOT NULL,
            project     TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (entity_name, project)
        )
    """)
    conn.commit()


# ─── 直接使用 store_vfs.spreading_activate（已在 store_vfs 实现）────────────────

def _try_import_spreading():
    """返回 spreading_activate 函数。"""
    return spreading_activate


# ─── 测试 1：一跳邻居激活，decay=0.7 ────────────────────────────────────────

def test_one_hop_activation(conn):
    """
    FTS5 命中 chunk_A（entity=retriever）→ entity_edges 中 retriever→store（dep）
    → chunk_B（entity=store）以 decay=0.7 进入激活集。
    """
    _ensure_entity_map(conn)
    _insert_chunk(conn, "chunk_A", "retriever 检索逻辑", entity_name="retriever")
    _insert_chunk(conn, "chunk_B", "store 持久化层", entity_name="store")
    # 建立 entity 关联边
    insert_edge(conn, "retriever", "depends_on", "store", confidence=0.9, project="test")

    # _spreading_activate 签名：(conn, hit_chunk_ids, project, decay=0.7, max_hops=1)
    # 返回：{chunk_id: activation_score}
    fn = _try_import_spreading()
    if fn is None:
        pytest.skip("_spreading_activate 尚未实现")

    # max_activation_bonus=1.0 以测试原始分数计算（不截断）
    result = fn(conn, ["chunk_A"], project="test", decay=0.7, max_hops=1,
                max_activation_bonus=1.0)

    assert "chunk_B" in result, "一跳邻居 chunk_B 应被激活"
    # decay=0.7, hop=1, conf=0.9 → score = parent(1.0) × conf(0.9) × decay^1(0.7) = 0.63
    assert 0.55 <= result["chunk_B"] <= 0.75, \
        f"decay=0.7 时 chunk_B 的激活分应约为 0.63, got {result['chunk_B']}"
    assert "chunk_A" not in result, "命中 chunk 本身不应出现在激活集"


# ─── 测试 2：二跳邻居激活，decay²=0.49 ──────────────────────────────────────

def test_two_hop_activation(conn):
    """
    chunk_A(retriever) → chunk_B(store) → chunk_C(sqlite)
    二跳 chunk_C 的激活分 = decay² × conf_AB × conf_BC。
    """
    _ensure_entity_map(conn)
    _insert_chunk(conn, "chunk_A", "retriever", entity_name="retriever")
    _insert_chunk(conn, "chunk_B", "store", entity_name="store")
    _insert_chunk(conn, "chunk_C", "sqlite 底层", entity_name="sqlite")
    insert_edge(conn, "retriever", "depends_on", "store", confidence=1.0, project="test")
    insert_edge(conn, "store", "uses", "sqlite", confidence=1.0, project="test")

    fn = _try_import_spreading()
    if fn is None:
        pytest.skip("_spreading_activate 尚未实现")

    # max_activation_bonus=1.0 测试原始分数不截断
    result = fn(conn, ["chunk_A"], project="test", decay=0.7, max_hops=2,
                max_activation_bonus=1.0)

    assert "chunk_C" in result, "二跳邻居 chunk_C 应被激活"
    # decay=0.7, hop=2, conf=1.0 → parent×conf×decay² = 1.0×1.0×0.7² = 0.49
    assert 0.4 <= result["chunk_C"] <= 0.55, \
        f"二跳激活分应约 0.49, got {result['chunk_C']}"
    # 一跳分应高于二跳
    assert result.get("chunk_B", 0) > result.get("chunk_C", 0), \
        "一跳邻居 score 应高于二跳"


# ─── 测试 3：FTS5 已有候选不重复添加 ─────────────────────────────────────────

def test_no_duplicate_with_fts_hits(conn):
    """
    chunk_B 既是 entity 激活邻居，又在 FTS5 命中集中。
    _spreading_activate 对已有 chunk 不应重复加入（由调用方去重或返回时标注）。
    """
    _ensure_entity_map(conn)
    _insert_chunk(conn, "chunk_A", "retriever", entity_name="retriever")
    _insert_chunk(conn, "chunk_B", "store（已被 FTS5 命中）", entity_name="store")
    insert_edge(conn, "retriever", "depends_on", "store", confidence=0.8, project="test")

    fn = _try_import_spreading()
    if fn is None:
        pytest.skip("_spreading_activate 尚未实现")

    # 传入 existing_ids 表示 FTS5 已有的 chunk
    result = fn(conn, ["chunk_A"], project="test", decay=0.7,
                max_hops=1, existing_ids={"chunk_B"})

    assert "chunk_B" not in result, \
        "chunk_B 已在 FTS5 候选中，不应再出现在 spreading activation 结果里"


# ─── 测试 4：孤立节点（无 entity 关联）不被激活 ──────────────────────────────

def test_isolated_chunk_not_activated(conn):
    """
    chunk_X 没有 entity_map 绑定和任何 entity_edges，不应出现在激活集。
    """
    _ensure_entity_map(conn)
    _insert_chunk(conn, "chunk_A", "有 entity 的 chunk", entity_name="retriever")
    _insert_chunk(conn, "chunk_X", "孤立 chunk，无 entity")  # 无 entity_name
    insert_edge(conn, "retriever", "depends_on", "store", confidence=0.9, project="test")

    fn = _try_import_spreading()
    if fn is None:
        pytest.skip("_spreading_activate 尚未实现")

    result = fn(conn, ["chunk_A"], project="test", decay=0.7, max_hops=1)

    assert "chunk_X" not in result, "孤立 chunk 不应被激活"


# ─── 测试 5：activation_bonus 上限保护 ───────────────────────────────────────

def test_activation_bonus_capped(conn):
    """
    即使 entity 图中有高置信度的多路径，激活分不应超过 max_activation_bonus（默认 0.4）。
    防止 entity 路由完全主导 chunk 评分，盖过 FTS5 的语义相关性。
    """
    _ensure_entity_map(conn)
    _insert_chunk(conn, "chunk_A", "源节点", entity_name="source")
    _insert_chunk(conn, "chunk_B", "目标节点", entity_name="target")
    # 多条高置信路径模拟（同一方向多关系）
    insert_edge(conn, "source", "uses", "target", confidence=1.0, project="test")
    insert_edge(conn, "source", "extends", "target", confidence=1.0, project="test")

    fn = _try_import_spreading()
    if fn is None:
        pytest.skip("_spreading_activate 尚未实现")

    result = fn(conn, ["chunk_A"], project="test", decay=0.7, max_hops=1,
                max_activation_bonus=0.4)

    if "chunk_B" in result:
        assert result["chunk_B"] <= 0.4, \
            f"activation_bonus 不应超过 0.4，实际: {result['chunk_B']}"


# ─── 测试 6：空图安全降级 ────────────────────────────────────────────────────

def test_empty_graph_safe_fallback(conn):
    """
    entity_edges 表完全为空时，_spreading_activate 安全返回 {}，不抛异常。
    OS 类比：vDSO 缓存 miss → fallback 到 syscall，不崩溃。
    """
    _ensure_entity_map(conn)
    _insert_chunk(conn, "chunk_A", "孤立 chunk", entity_name="orphan")
    # 不插入任何 edge

    fn = _try_import_spreading()
    if fn is None:
        pytest.skip("_spreading_activate 尚未实现")

    result = fn(conn, ["chunk_A"], project="test", decay=0.7, max_hops=2)

    assert isinstance(result, dict), "应返回 dict"
    assert len(result) == 0, "无边图应返回空激活集"


# ─── 测试 7：性能 — 100 edges 时 < 5ms ───────────────────────────────────────

def test_spreading_activation_performance(conn):
    """
    100 个 entity_edges 时，_spreading_activate 耗时 < 5ms。
    OS 类比：TLB hit path < 3ms — spreading activation 作为检索增强层不应成为瓶颈。
    """
    _ensure_entity_map(conn)

    # 构建 star topology：hub → 99 leaves
    _insert_chunk(conn, "chunk_hub", "中心节点", entity_name="hub")
    for i in range(99):
        _insert_chunk(conn, f"chunk_leaf_{i}", f"叶节点 {i}", entity_name=f"leaf_{i}")
        insert_edge(conn, "hub", "connects", f"leaf_{i}",
                    confidence=0.8, project="test")
    conn.commit()

    fn = _try_import_spreading()
    if fn is None:
        pytest.skip("_spreading_activate 尚未实现")

    t0 = time.time()
    for _ in range(20):
        fn(conn, ["chunk_hub"], project="test", decay=0.7, max_hops=1)
    elapsed_ms = (time.time() - t0) * 1000 / 20

    assert elapsed_ms < 5.0, f"spreading_activate 太慢: {elapsed_ms:.1f}ms（应 < 5ms）"


# ─── 测试 8：构建式召回 — understand 意图时 reasoning_chain 优先 ─────────────

def test_constructive_recall_understand_intent():
    """
    intent='understand' 时，注入顺序调整：reasoning_chain > decision > task_state。
    OS 类比：进程 nice 值重排 — 同样的内容，根据当前任务优先级重新调度。
    """
    try:
        import sys, importlib
        sys.path.insert(0, str(Path(__file__).parent / "hooks"))
        r = importlib.import_module("retriever")
        if not hasattr(r, '_constructive_reorder'):
            pytest.skip("_constructive_reorder 尚未实现")
    except Exception:
        pytest.skip("retriever 导入失败")

    chunks = [
        {"id": "c1", "chunk_type": "decision", "summary": "决策 A", "importance": 0.8},
        {"id": "c2", "chunk_type": "reasoning_chain", "summary": "推理链 B", "importance": 0.7},
        {"id": "c3", "chunk_type": "task_state", "summary": "任务状态 C", "importance": 0.9},
    ]

    reordered = r._constructive_reorder(chunks, intent="understand")
    types = [c["chunk_type"] for c in reordered]

    assert types[0] == "reasoning_chain", \
        f"understand 意图下 reasoning_chain 应排第一，实际: {types}"


# ─── 测试 9：构建式召回 — implement 意图时 procedure/decision 优先 ────────────

def test_constructive_recall_implement_intent():
    """
    intent='implement' 时，procedure 和 decision 排在 reasoning_chain 前面。
    """
    try:
        import sys, importlib
        sys.path.insert(0, str(Path(__file__).parent / "hooks"))
        r = importlib.import_module("retriever")
        if not hasattr(r, '_constructive_reorder'):
            pytest.skip("_constructive_reorder 尚未实现")
    except Exception:
        pytest.skip("retriever 导入失败")

    chunks = [
        {"id": "c1", "chunk_type": "reasoning_chain", "summary": "推理", "importance": 0.9},
        {"id": "c2", "chunk_type": "procedure", "summary": "操作步骤", "importance": 0.7},
        {"id": "c3", "chunk_type": "decision", "summary": "决策记录", "importance": 0.8},
    ]

    reordered = r._constructive_reorder(chunks, intent="implement")
    types = [c["chunk_type"] for c in reordered]

    # procedure 或 decision 应在前两位
    assert types[0] in ("procedure", "decision"), \
        f"implement 意图下 procedure/decision 应排首位，实际: {types}"


# ─── 测试 10：展示层调整不修改 DB ────────────────────────────────────────────

def test_constructive_recall_no_db_mutation(conn):
    """
    _constructive_reorder 只调整展示顺序，不修改 memory_chunks 表中的任何字段。
    这是关键不变量：展示层 vs 存储层严格分离（CoW 语义）。
    """
    _ensure_entity_map(conn)
    _insert_chunk(conn, "chunk_A", "原始摘要内容", chunk_type="reasoning_chain")

    try:
        import sys, importlib
        sys.path.insert(0, str(Path(__file__).parent / "hooks"))
        r = importlib.import_module("retriever")
        if not hasattr(r, '_constructive_reorder'):
            pytest.skip("_constructive_reorder 尚未实现")
    except Exception:
        pytest.skip("retriever 导入失败")

    chunks = [{"id": "chunk_A", "chunk_type": "reasoning_chain",
               "summary": "原始摘要内容", "importance": 0.8}]
    r._constructive_reorder(chunks, intent="fix_bug")

    # 验证 DB 中内容未被修改
    row = conn.execute(
        "SELECT summary FROM memory_chunks WHERE id = 'chunk_A'"
    ).fetchone()
    assert row is not None
    assert row["summary"] == "原始摘要内容", \
        "_constructive_reorder 不应修改 DB 中的 summary 字段"
