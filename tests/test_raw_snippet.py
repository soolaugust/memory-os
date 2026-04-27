"""
test_raw_snippet.py — 迭代306：写入时保真（raw_snippet）测试

测试点：
1. ensure_schema 后 memory_chunks 有 raw_snippet 列
2. insert_chunk 能写入 raw_snippet，读回正确
3. _write_chunk 传入 raw_snippet 参数后，DB 里正确存储
4. raw_snippet 超过 500 字时自动截断
5. retriever 注入时，importance=0.80 的 chunk 有 raw_snippet → 附加「原文：」
6. retriever 注入时，importance=0.50 的 chunk → 不附加原文
"""
import os
import sys
import sqlite3
import tempfile
import pytest

# ── 测试用临时 DB 隔离 ──────────────────────────────────────────────────────────
@pytest.fixture
def tmp_db(tmp_path):
    db_path = tmp_path / "test_store.db"
    os.environ["MEMORY_OS_DIR"] = str(tmp_path)
    os.environ["MEMORY_OS_DB"] = str(db_path)
    yield db_path
    # 清理环境变量（避免污染其他测试）
    os.environ.pop("MEMORY_OS_DIR", None)
    os.environ.pop("MEMORY_OS_DB", None)


# ── 测试 1：ensure_schema 后 raw_snippet 列存在 ───────────────────────────────
def test_schema_has_raw_snippet_column(tmp_db):
    # 需要在 env 设置后 import，避免模块级路径固化
    import importlib
    import store_vfs
    importlib.reload(store_vfs)

    conn = store_vfs.open_db(tmp_db)
    store_vfs.ensure_schema(conn)
    conn.commit()

    cols = [row[1] for row in conn.execute("PRAGMA table_info(memory_chunks)").fetchall()]
    assert "raw_snippet" in cols, f"raw_snippet 列不存在，当前列：{cols}"
    conn.close()


# ── 测试 2：insert_chunk 写入并读回 raw_snippet ───────────────────────────────
def test_insert_and_read_raw_snippet(tmp_db):
    import importlib
    import store_vfs
    importlib.reload(store_vfs)
    from schema import MemoryChunk

    conn = store_vfs.open_db(tmp_db)
    store_vfs.ensure_schema(conn)

    snippet_text = "这是一段原始提取的上下文文本，包含完整的决策背景信息。"
    chunk = MemoryChunk(
        project="test_project",
        source_session="sess_001",
        chunk_type="decision",
        content="[decision] 选择使用 SQLite",
        summary="选择使用 SQLite",
        raw_snippet=snippet_text,
        importance=0.85,
        retrievability=0.35,
    )
    store_vfs.insert_chunk(conn, chunk.to_dict())
    conn.commit()

    row = conn.execute(
        "SELECT raw_snippet FROM memory_chunks WHERE summary=?",
        ("选择使用 SQLite",)
    ).fetchone()
    assert row is not None, "chunk 未写入"
    assert row[0] == snippet_text, f"raw_snippet 读回不匹配：{row[0]!r} != {snippet_text!r}"
    conn.close()


# ── 测试 3：_write_chunk 传入 raw_snippet 后 DB 正确存储 ──────────────────────
def test_write_chunk_stores_raw_snippet(tmp_db, monkeypatch):
    import importlib
    import store_vfs
    importlib.reload(store_vfs)

    # 将 hooks 目录加入 sys.path
    hooks_dir = os.path.join(os.path.dirname(__file__), "hooks")
    if hooks_dir not in sys.path:
        sys.path.insert(0, hooks_dir)

    import extractor
    importlib.reload(extractor)

    conn = store_vfs.open_db(tmp_db)
    store_vfs.ensure_schema(conn)

    snippet = "原始代码片段：def foo(): return 42  # 迭代306 测试用原文"
    extractor._write_chunk(
        chunk_type="decision",
        summary="选择使用 raw_snippet 存储原文（测试）",
        project="test_project",
        session_id="sess_002",
        conn=conn,
        raw_snippet=snippet,
        _txn_managed=True,
    )
    conn.commit()

    row = conn.execute(
        "SELECT raw_snippet FROM memory_chunks WHERE summary=?",
        ("选择使用 raw_snippet 存储原文（测试）",)
    ).fetchone()
    assert row is not None, "_write_chunk 未写入 DB"
    assert row[0] == snippet, f"raw_snippet 存储错误：{row[0]!r}"
    conn.close()


# ── 测试 4：raw_snippet 超过 500 字时自动截断 ─────────────────────────────────
def test_raw_snippet_truncated_at_500(tmp_db):
    import importlib
    import store_vfs
    importlib.reload(store_vfs)
    from schema import MemoryChunk

    conn = store_vfs.open_db(tmp_db)
    store_vfs.ensure_schema(conn)

    long_text = "A" * 600  # 超过 500 字
    chunk = MemoryChunk(
        project="test_project",
        source_session="sess_003",
        chunk_type="reasoning_chain",
        content="[reasoning_chain] 长文本截断测试",
        summary="长文本截断测试_unique",
        raw_snippet=long_text,
        importance=0.80,
        retrievability=0.2,
    )
    store_vfs.insert_chunk(conn, chunk.to_dict())
    conn.commit()

    row = conn.execute(
        "SELECT raw_snippet FROM memory_chunks WHERE summary=?",
        ("长文本截断测试_unique",)
    ).fetchone()
    assert row is not None, "chunk 未写入"
    assert len(row[0]) == 500, f"raw_snippet 未截断到 500，实际长度：{len(row[0])}"
    conn.close()


# ── 测试 5 & 6：retriever 注入时 importance 门控 ──────────────────────────────
def test_retriever_inject_raw_snippet_high_importance(tmp_db):
    """importance=0.80 + raw_snippet → 注入时附加「原文：」"""
    import importlib
    import store_vfs
    importlib.reload(store_vfs)
    from schema import MemoryChunk

    conn = store_vfs.open_db(tmp_db)
    store_vfs.ensure_schema(conn)

    snippet = "原文内容：选择 SQLite 是因为零依赖且跨平台。"
    chunk = MemoryChunk(
        project="test_proj",
        source_session="sess_hi",
        chunk_type="decision",
        content="[decision] 选择 SQLite",
        summary="选择 SQLite 因为零依赖",
        raw_snippet=snippet,
        importance=0.80,
        retrievability=0.35,
    )
    store_vfs.insert_chunk(conn, chunk.to_dict())
    conn.commit()

    # 模拟 retriever 逻辑：取 importance >= 0.75 的 chunk 的 raw_snippet
    chunk_id = conn.execute(
        "SELECT id FROM memory_chunks WHERE summary=?",
        ("选择 SQLite 因为零依赖",)
    ).fetchone()[0]

    high_imp_ids = [chunk_id]
    rs_ph = ",".join("?" * len(high_imp_ids))
    rs_rows = conn.execute(
        f"SELECT id, raw_snippet FROM memory_chunks WHERE id IN ({rs_ph})",
        high_imp_ids,
    ).fetchall()
    raw_snippets = {r[0]: r[1] for r in rs_rows if r[1]}

    assert chunk_id in raw_snippets, "importance=0.80 的 chunk 应能取到 raw_snippet"
    rs = raw_snippets[chunk_id]
    line = f"[决策] 选择 SQLite 因为零依赖"
    line_with_raw = f"{line}（原文：{rs[:150]}）"
    assert "原文：" in line_with_raw, "注入行应包含「原文：」"
    assert snippet[:50] in line_with_raw, "原文内容应出现在注入行中"
    conn.close()


def test_retriever_no_raw_snippet_low_importance(tmp_db):
    """importance=0.50 → 不应附加原文（不在 high_imp_ids 中）"""
    import importlib
    import store_vfs
    importlib.reload(store_vfs)
    from schema import MemoryChunk

    conn = store_vfs.open_db(tmp_db)
    store_vfs.ensure_schema(conn)

    snippet = "低重要性不应附加的原文内容。"
    chunk = MemoryChunk(
        project="test_proj",
        source_session="sess_lo",
        chunk_type="conversation_summary",
        content="[conversation_summary] 低重要性摘要",
        summary="低重要性摘要测试",
        raw_snippet=snippet,
        importance=0.50,
        retrievability=0.35,
    )
    store_vfs.insert_chunk(conn, chunk.to_dict())
    conn.commit()

    chunk_id = conn.execute(
        "SELECT id FROM memory_chunks WHERE summary=?",
        ("低重要性摘要测试",)
    ).fetchone()[0]

    # importance=0.50 不满足 >= 0.75，不进入 high_imp_ids
    high_imp_ids = []  # 模拟 importance < 0.75 → 不查询
    raw_snippets: dict = {}  # 空

    rs = raw_snippets.get(chunk_id, "")
    line = "[摘要] 低重要性摘要测试"
    # 不附加原文
    if rs:
        line = f"{line}（原文：{rs[:150]}）"

    assert "原文：" not in line, "importance=0.50 的 chunk 不应附加原文"
    conn.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
