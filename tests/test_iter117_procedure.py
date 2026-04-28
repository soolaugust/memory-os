"""
test_iter117_procedure — iter117 procedure chunk visibility

验证:
  1. procedure 在 _ALL_RETRIEVE_TYPES 中（retriever 可见）
  2. procedure 在 WORKING_SET_TYPES 中（loader 预加载）
  3. procedure 在 code_review 和 implement 意图中（intent routing）
  4. procedure chunks 在 fts_search 中可被检索
"""
import sys
import re
import os
import pytest
import sqlite3
import json
from pathlib import Path

# Setup path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def test_procedure_in_all_retrieve_types():
    """iter117: procedure 必须在 _ALL_RETRIEVE_TYPES 中。"""
    import hooks.retriever as retriever_module
    # Read retriever.py source and check _ALL_RETRIEVE_TYPES contains procedure
    src = (ROOT.parent / "hooks" / "retriever.py").read_text()
    # Find _ALL_RETRIEVE_TYPES block
    idx = src.find("_ALL_RETRIEVE_TYPES = (")
    assert idx >= 0, "_ALL_RETRIEVE_TYPES not found"
    block_end = src.find(")", idx)
    block = src[idx:block_end+1]
    assert "procedure" in block, f"'procedure' missing from _ALL_RETRIEVE_TYPES: {block}"


def test_procedure_in_working_set_types():
    """iter117: procedure 必须在 loader.py WORKING_SET_TYPES 中。"""
    src = (ROOT.parent / "hooks" / "loader.py").read_text()
    idx = src.find("WORKING_SET_TYPES = (")
    assert idx >= 0, "WORKING_SET_TYPES not found"
    block_end = src.find(")", idx)
    block = src[idx:block_end+1]
    assert "procedure" in block, f"'procedure' missing from WORKING_SET_TYPES: {block}"


def test_procedure_in_implement_intent():
    """iter117: implement 意图应包含 procedure 类型。"""
    src = (ROOT.parent / "hooks" / "retriever.py").read_text()
    # Find _INTENT_MAP block
    m = re.search(r'_INTENT_MAP\s*=\s*\{(.+?)\}', src, re.DOTALL)
    assert m, "_INTENT_MAP not found"
    block = m.group(1)
    # Find implement line within the block
    for line in block.split("\n"):
        if '"implement"' in line and "procedure" in line:
            return  # Found and has procedure
    pytest.fail(f"'procedure' missing from implement intent in _INTENT_MAP block:\n{block}")


def test_procedure_in_code_review_intent():
    """iter117: code_review 意图应包含 procedure 类型。"""
    src = (ROOT.parent / "hooks" / "retriever.py").read_text()
    # Find _INTENT_MAP block
    m = re.search(r'_INTENT_MAP\s*=\s*\{(.+?)\}', src, re.DOTALL)
    assert m, "_INTENT_MAP not found"
    block = m.group(1)
    for line in block.split("\n"):
        if '"code_review"' in line and "procedure" in line:
            return  # Found and has procedure
    pytest.fail(f"'procedure' missing from code_review intent in _INTENT_MAP block:\n{block}")


def test_procedure_fts_searchable(tmp_path, monkeypatch):
    """iter117: procedure chunks 可通过 fts_search 检索。"""
    # Set up temp DB
    db_path = tmp_path / "store.db"
    monkeypatch.setenv("MEMORY_OS_DIR", str(tmp_path))
    monkeypatch.setenv("MEMORY_OS_DB", str(db_path))

    from store import open_db, ensure_schema, insert_chunk, fts_search
    import hashlib, datetime

    conn = open_db()
    ensure_schema(conn)

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    chunk = {
        "id": "test-proc-001",
        "created_at": now,
        "updated_at": now,
        "project": "global",
        "source_session": "test",
        "chunk_type": "procedure",
        "content": "[capabilities] 锁分析协议 > 使用 lockdep 检查死锁，记录持锁路径",
        "summary": "[capabilities] 锁分析协议",
        "tags": json.dumps(["procedure", "capabilities"]),
        "importance": 0.85,
        "retrievability": 1.0,
        "last_accessed": now,
        "access_count": 0,
        "oom_adj": 0,
        "lru_gen": 0,
    }
    insert_chunk(conn, chunk)
    conn.commit()

    # FTS search should find it
    results = fts_search(conn, "锁分析协议", "global", top_k=5)
    ids = [r["id"] for r in results]
    assert "test-proc-001" in ids, f"procedure chunk not found in fts_search results: {ids}"

    # Also check chunk_type
    found = next(r for r in results if r["id"] == "test-proc-001")
    assert found["chunk_type"] == "procedure"
    conn.close()


def test_procedure_not_excluded_by_default(tmp_path, monkeypatch):
    """iter117: 默认 exclude_types 不包含 procedure。"""
    db_path = tmp_path / "store.db"
    monkeypatch.setenv("MEMORY_OS_DIR", str(tmp_path))
    monkeypatch.setenv("MEMORY_OS_DB", str(db_path))

    from config import get as sysctl
    exclude_str = sysctl("retriever.exclude_types")
    excluded = set(t.strip() for t in exclude_str.split(",") if t.strip()) if exclude_str else set()
    assert "procedure" not in excluded, f"procedure is in default exclude_types: {excluded}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
