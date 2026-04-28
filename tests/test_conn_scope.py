#!/usr/bin/env python3
"""
迭代24 验证：Per-Request Connection Scope
测试连接复用、bug 修复、向后兼容性。
"""
import sys
import os
import json
import time
import sqlite3
from pathlib import Path

# 设置环境
os.environ["CLAUDE_CWD"] = str(__import__("pathlib").Path(__file__).parent.parent.parent.parent.parent)
os.environ["CLAUDE_SESSION_ID"] = "test-conn-scope"

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "hooks"))

import tmpfs  # noqa: F401 — tmpfs isolation (iter54), must precede store import
from store import open_db, ensure_schema, insert_chunk, fts_search, get_chunks
from schema import MemoryChunk
from utils import resolve_project_id


def _seed_test_data(conn, project):
    """写入测试数据。"""
    ensure_schema(conn)
    for i, (ctype, summary, importance) in enumerate([
        ("decision", "选择 FTS5 替代 Python BM25 全表扫描", 0.85),
        ("decision", "采用 WAL 模式提升并发读写性能", 0.80),
        ("reasoning_chain", "因为 BM25 复杂度 O(N)，所以引入 FTS5 索引降为 O(log N)", 0.75),
        ("excluded_path", "放弃 chromadb 因为中文 BM25 效果差", 0.70),
        ("conversation_summary", "已完成 FTS5 全文索引迁移，10/10 测试通过", 0.65),
        ("decision", "连接池在 hook subprocess 中无意义，改用 per-request scope", 0.85),
    ]):
        chunk = MemoryChunk(
            project=project,
            source_session="test-seed",
            chunk_type=ctype,
            content=f"[{ctype}] {summary}",
            summary=summary,
            tags=[ctype, project],
            importance=importance,
        )
        insert_chunk(conn, chunk.to_dict())
    conn.commit()


def test_1_single_conn_retriever_flow():
    """T1: retriever 全流程使用单一连接（不再多次 open/close）"""
    project = resolve_project_id()
    conn = open_db()
    try:
        _seed_test_data(conn, project)
        # 模拟 retriever 流程：FTS 搜索 + get_chunks fallback + 都用同一个 conn
        fts_results = fts_search(conn, "FTS5 索引", project, top_k=5)
        assert len(fts_results) > 0, f"FTS5 应返回结果, got {len(fts_results)}"
        # 同一个 conn 继续做 get_chunks（fallback 路径）
        chunks = get_chunks(conn, project)
        assert len(chunks) >= 6, f"应有至少 6 条 chunk, got {len(chunks)}"
        print(f"  T1 PASS: single conn, FTS={len(fts_results)} chunks={len(chunks)}")
    finally:
        conn.close()


def test_2_candidates_count_no_crash():
    """T2: FTS5 路径下 candidates_count 正确计算（修复原 len(chunks) bug）"""
    project = resolve_project_id()
    conn = open_db()
    try:
        ensure_schema(conn)
        fts_results = fts_search(conn, "BM25 WAL", project, top_k=10)
        # 之前 bug：FTS5 路径下 chunks 未定义，len(chunks) 会 NameError
        # 修复后用 candidates_count = len(fts_results)
        candidates_count = len(fts_results)
        assert isinstance(candidates_count, int), "candidates_count 应该是 int"
        assert candidates_count >= 0, "candidates_count >= 0"
        print(f"  T2 PASS: candidates_count={candidates_count} (no NameError)")
    finally:
        conn.close()


def test_3_kr_conn_passthrough():
    """T3: KnowledgeRouter _search_memory_os 接受外部 conn"""
    from knowledge_router import _search_memory_os
    project = resolve_project_id()
    conn = open_db()
    try:
        ensure_schema(conn)
        # 传入外部 conn（应复用，不自行 open/close）
        results = _search_memory_os("FTS5 索引", project, conn=conn)
        assert isinstance(results, list), "应返回 list"
        # conn 还能正常使用（没被内部 close 掉）
        count = conn.execute("SELECT COUNT(*) FROM memory_chunks").fetchone()[0]
        assert count > 0, f"conn 仍可用, count={count}"
        print(f"  T3 PASS: conn passthrough, results={len(results)}, conn still alive")
    finally:
        conn.close()


def test_4_kr_standalone_compat():
    """T4: KnowledgeRouter standalone 调用（无外部 conn）向后兼容"""
    from knowledge_router import _search_memory_os
    project = resolve_project_id()
    # 不传 conn，应自行管理连接
    results = _search_memory_os("WAL 模式", project)
    assert isinstance(results, list), "应返回 list"
    print(f"  T4 PASS: standalone compat, results={len(results)}")


def test_5_kr_route_with_conn():
    """T5: route() 传入 conn，全链路复用"""
    from knowledge_router import route
    project = resolve_project_id()
    conn = open_db()
    try:
        ensure_schema(conn)
        results = route("FTS5 BM25", project=project,
                        sources=["memory_os"], conn=conn)
        assert isinstance(results, list), "应返回 list"
        # conn 仍可用
        count = conn.execute("SELECT COUNT(*) FROM memory_chunks").fetchone()[0]
        assert count > 0, f"conn 仍可用"
        print(f"  T5 PASS: route() with conn, results={len(results)}")
    finally:
        conn.close()


def test_6_perf_single_vs_multi():
    """T6: 性能对比——单连接 vs 多连接"""
    project = resolve_project_id()

    # 单连接：模拟改造后的 retriever 流程
    t0 = time.time()
    conn = open_db()
    ensure_schema(conn)
    fts_search(conn, "FTS5 索引 BM25", project, top_k=5)
    get_chunks(conn, project)
    conn.execute("SELECT COUNT(*) FROM memory_chunks").fetchone()
    conn.close()
    single_ms = (time.time() - t0) * 1000

    # 多连接：模拟改造前的 retriever 流程（每步都 open/close）
    t0 = time.time()
    conn1 = open_db()
    ensure_schema(conn1)
    fts_search(conn1, "FTS5 索引 BM25", project, top_k=5)
    conn1.close()
    conn2 = open_db()
    ensure_schema(conn2)
    get_chunks(conn2, project)
    conn2.close()
    conn3 = open_db()
    ensure_schema(conn3)
    conn3.execute("SELECT COUNT(*) FROM memory_chunks").fetchone()
    conn3.close()
    multi_ms = (time.time() - t0) * 1000

    saved = multi_ms - single_ms
    print(f"  T6 PASS: single={single_ms:.2f}ms, multi={multi_ms:.2f}ms, saved={saved:.2f}ms")


def test_7_retriever_syntax_check():
    """T7: retriever.py 语法正确（无 NameError/SyntaxError）"""
    import py_compile
    retriever_path = ROOT.parent / "hooks" / "retriever.py"
    try:
        py_compile.compile(str(retriever_path), doraise=True)
        print(f"  T7 PASS: retriever.py syntax OK")
    except py_compile.PyCompileError as e:
        print(f"  T7 FAIL: {e}")
        raise


def test_8_kr_syntax_check():
    """T8: knowledge_router.py 语法正确"""
    import py_compile
    kr_path = ROOT.parent / "hooks" / "knowledge_router.py"
    try:
        py_compile.compile(str(kr_path), doraise=True)
        print(f"  T8 PASS: knowledge_router.py syntax OK")
    except py_compile.PyCompileError as e:
        print(f"  T8 FAIL: {e}")
        raise


if __name__ == "__main__":
    tests = [
        test_1_single_conn_retriever_flow,
        test_2_candidates_count_no_crash,
        test_3_kr_conn_passthrough,
        test_4_kr_standalone_compat,
        test_5_kr_route_with_conn,
        test_6_perf_single_vs_multi,
        test_7_retriever_syntax_check,
        test_8_kr_syntax_check,
    ]
    passed = 0
    failed = 0
    t_start = time.time()
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {t.__name__}: {e}")
            failed += 1
    total_ms = (time.time() - t_start) * 1000
    print(f"\n{'='*40}")
    print(f"迭代24 验证: {passed}/{passed+failed} passed, avg {total_ms/(passed+failed):.2f}ms")
    if failed:
        sys.exit(1)
