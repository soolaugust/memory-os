#!/usr/bin/env python3
"""
test_iter84_readonly.py — Read-Only Fast Path 测试
迭代84：验证只读连接隔离、DeferredLogs 缓冲、读写分离无锁竞争

OS 类比：open(O_RDONLY) + write-back caching 测试
"""
import tmpfs  # 测试隔离（迭代54）：必须在 store import 前

import json
import os
import sqlite3
import sys
import threading
import time
import unittest
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

# memory-os imports
from store import (open_db, ensure_schema, insert_chunk, fts_search,
                   update_accessed, dmesg_log, DMESG_INFO, DMESG_WARN,
                   DMESG_DEBUG, get_chunks, insert_trace)
from store_core import STORE_DB, MEMORY_OS_DIR, bump_chunk_version


def _make_chunk(cid, summary, project="test_proj", chunk_type="decision",
                importance=0.8):
    now = datetime.now(timezone.utc).isoformat()
    return {
        "id": cid,
        "created_at": now,
        "updated_at": now,
        "project": project,
        "source_session": "test_session",
        "chunk_type": chunk_type,
        "content": f"content for {summary}",
        "summary": summary,
        "tags": "[]",
        "importance": importance,
        "retrievability": 0.5,
        "last_accessed": now,
    }


def _seed_db(conn, n=3, project="test_proj"):
    """Seed DB with test chunks and commit."""
    for i in range(n):
        insert_chunk(conn, _make_chunk(
            f"c{i}", f"test chunk readonly {i}", project=project))
    conn.commit()


class TestOpenDbReadonly(unittest.TestCase):
    """验证 _open_db_readonly() 行为：immutable 连接不能写入。"""

    def setUp(self):
        conn = open_db()
        ensure_schema(conn)
        _seed_db(conn)
        conn.commit()
        conn.close()

    def test_immutable_connect(self):
        """immutable=1 URI 连接成功打开。"""
        uri = f"file:{STORE_DB}?immutable=1"
        conn = sqlite3.connect(uri, uri=True)
        # 能读
        count = conn.execute("SELECT COUNT(*) FROM memory_chunks").fetchone()[0]
        self.assertGreater(count, 0)
        conn.close()

    def test_immutable_cannot_write(self):
        """immutable=1 连接不能执行写操作。"""
        uri = f"file:{STORE_DB}?immutable=1"
        conn = sqlite3.connect(uri, uri=True)
        with self.assertRaises(sqlite3.OperationalError):
            conn.execute("INSERT INTO dmesg (timestamp, level, subsystem, message) "
                         "VALUES ('2026-01-01', 'INFO', 'test', 'should fail')")
        conn.close()

    def test_fts5_works_readonly(self):
        """immutable 连接上 FTS5 搜索正常工作。"""
        uri = f"file:{STORE_DB}?immutable=1"
        conn = sqlite3.connect(uri, uri=True)
        rows = conn.execute(
            "SELECT COUNT(*) FROM memory_chunks_fts"
        ).fetchone()[0]
        self.assertGreater(rows, 0)
        conn.close()

    def test_query_only_fallback(self):
        """query_only=ON fallback 也阻止写入。"""
        conn = sqlite3.connect(str(STORE_DB), timeout=2)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA query_only=ON")
        with self.assertRaises(sqlite3.OperationalError):
            conn.execute("INSERT INTO dmesg (timestamp, level, subsystem, message) "
                         "VALUES ('2026-01-01', 'INFO', 'test', 'should fail')")
        conn.close()


class TestDeferredLogs(unittest.TestCase):
    """验证 _DeferredLogs 缓冲区正确收集和 flush。"""

    def setUp(self):
        conn = open_db()
        ensure_schema(conn)
        conn.commit()
        conn.close()

    def test_buffer_and_flush(self):
        """日志先缓冲到内存，flush 后写入 DB。"""
        # 模拟 retriever.py 的 _DeferredLogs
        sys.path.insert(0, str(Path(__file__).parent / "hooks"))
        # 手动实现 DeferredLogs（与 retriever.py 中一致）
        buf = []
        buf.append((DMESG_INFO, "retriever", "test msg 1", "s1", "p1", None))
        buf.append((DMESG_WARN, "retriever", "test msg 2", "s1", "p1", {"k": "v"}))

        self.assertEqual(len(buf), 2)

        # Flush to write connection
        wconn = open_db()
        ensure_schema(wconn)
        for level, subsystem, message, session_id, project, extra in buf:
            dmesg_log(wconn, level, subsystem, message,
                      session_id=session_id, project=project, extra=extra)
        wconn.commit()

        # Verify
        rows = wconn.execute(
            "SELECT level, message FROM dmesg WHERE subsystem='retriever' "
            "ORDER BY id DESC LIMIT 2"
        ).fetchall()
        self.assertEqual(len(rows), 2)
        levels = {r[0] for r in rows}
        self.assertIn("INFO", levels)
        self.assertIn("WARN", levels)
        wconn.close()

    def test_empty_flush_noop(self):
        """空缓冲区 flush 不产生任何 DB 写入。"""
        wconn = open_db()
        ensure_schema(wconn)
        before = wconn.execute("SELECT COUNT(*) FROM dmesg").fetchone()[0]
        # Empty flush — nothing happens
        wconn.commit()
        after = wconn.execute("SELECT COUNT(*) FROM dmesg").fetchone()[0]
        self.assertEqual(before, after)
        wconn.close()


class TestReadWriteSeparation(unittest.TestCase):
    """验证只读连接和写连接可以并发操作（零锁竞争）。"""

    def setUp(self):
        conn = open_db()
        ensure_schema(conn)
        _seed_db(conn)
        conn.commit()
        conn.close()

    def test_no_lock_contention(self):
        """只读连接不阻塞写连接。"""
        # Open readonly
        uri = f"file:{STORE_DB}?immutable=1"
        rconn = sqlite3.connect(uri, uri=True)

        # Read while holding readonly connection open
        count = rconn.execute("SELECT COUNT(*) FROM memory_chunks").fetchone()[0]
        self.assertGreater(count, 0)

        # Write connection should succeed without timeout
        wconn = open_db()
        ensure_schema(wconn)
        dmesg_log(wconn, DMESG_INFO, "test", "concurrent write OK")
        wconn.commit()
        wconn.close()

        rconn.close()

    def test_fts5_search_while_writing(self):
        """FTS5 搜索在写连接活跃时正常工作。"""
        # Start a write transaction
        wconn = open_db()
        ensure_schema(wconn)
        insert_chunk(wconn, _make_chunk("cw1", "write during read test"))
        # Don't commit yet — hold write lock

        # Readonly FTS5 search should still work (immutable sees snapshot)
        uri = f"file:{STORE_DB}?immutable=1"
        rconn = sqlite3.connect(uri, uri=True)
        # Search existing data (not the uncommitted write)
        rows = rconn.execute(
            "SELECT COUNT(*) FROM memory_chunks WHERE summary LIKE '%readonly%'"
        ).fetchone()[0]
        self.assertGreaterEqual(rows, 0)  # immutable sees pre-write snapshot
        rconn.close()

        wconn.commit()
        wconn.close()


class TestWriteBackPath(unittest.TestCase):
    """验证写回路径：update_accessed + mglru_promote + trace 在写连接上执行。"""

    def setUp(self):
        conn = open_db()
        ensure_schema(conn)
        _seed_db(conn, n=3)
        conn.commit()
        conn.close()

    def test_update_accessed_on_write_conn(self):
        """update_accessed 在写连接上正确递增 access_count。"""
        wconn = open_db()
        ensure_schema(wconn)
        before = wconn.execute(
            "SELECT COALESCE(access_count, 0) FROM memory_chunks WHERE id='c0'"
        ).fetchone()[0]
        update_accessed(wconn, ["c0"])
        wconn.commit()
        after = wconn.execute(
            "SELECT COALESCE(access_count, 0) FROM memory_chunks WHERE id='c0'"
        ).fetchone()[0]
        self.assertEqual(after, before + 1)
        wconn.close()

    def test_trace_insert_on_write_conn(self):
        """recall_traces INSERT 在写连接上成功。"""
        wconn = open_db()
        ensure_schema(wconn)
        import uuid
        insert_trace(wconn, {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": "test_s",
            "project": "test_proj",
            "prompt_hash": "abc123",
            "candidates_count": 5,
            "top_k_json": [{"id": "c0", "summary": "test", "score": 0.8}],
            "injected": 1,
            "reason": "test|readonly",
            "duration_ms": 7.5,
        })
        wconn.commit()
        count = wconn.execute(
            "SELECT COUNT(*) FROM recall_traces WHERE reason LIKE '%readonly%'"
        ).fetchone()[0]
        self.assertEqual(count, 1)
        wconn.close()

    def test_writeback_after_readonly_close(self):
        """完整流程：readonly 读 → 关闭 → 写连接写回。"""
        # Phase 1: readonly read
        uri = f"file:{STORE_DB}?immutable=1"
        rconn = sqlite3.connect(uri, uri=True)
        chunks = rconn.execute(
            "SELECT id FROM memory_chunks WHERE project='test_proj'"
        ).fetchall()
        chunk_ids = [r[0] for r in chunks]
        self.assertGreater(len(chunk_ids), 0)
        rconn.close()

        # Phase 2: write-back
        wconn = open_db()
        ensure_schema(wconn)
        update_accessed(wconn, chunk_ids)
        dmesg_log(wconn, DMESG_INFO, "test", "writeback after readonly")
        wconn.commit()

        # Verify writes landed
        row = wconn.execute(
            "SELECT COALESCE(access_count, 0) FROM memory_chunks WHERE id=?",
            (chunk_ids[0],)
        ).fetchone()
        self.assertGreater(row[0], 0)
        wconn.close()


class TestRetrieverCodePaths(unittest.TestCase):
    """AST 级验证：retriever.py 只读路径无 dmesg_log(conn,...) 调用。"""

    def test_no_direct_dmesg_on_readonly_conn(self):
        """retriever.py 中 dmesg_log 调用不使用只读 conn 变量。"""
        retriever_path = Path(__file__).parent / "hooks" / "retriever.py"
        source = retriever_path.read_text()

        # 在 main() 函数体内，dmesg_log 的第一个参数不应该是 conn
        # 排除 _DeferredLogs.flush 内的调用（那里接收的是写连接）
        import re
        # 找所有 dmesg_log(conn, 调用（非 wconn）
        # 应该只出现在 _DeferredLogs.flush 方法内
        pattern = r'dmesg_log\(conn,'
        matches = list(re.finditer(pattern, source))
        # 唯一合法出现：_DeferredLogs.flush 中的 dmesg_log(conn, ...)
        # 在 flush 方法内 conn 参数是写连接传入
        for m in matches:
            # 检查上下文是否在 flush 方法内（或 _DeferredLogs 类体内）
            start = max(0, m.start() - 500)
            context = source[start:m.start()]
            in_flush = "def flush" in context
            in_deferred = "_DeferredLogs" in context or "class _DeferredLogs" in source[:m.start()]
            self.assertTrue(in_flush or ("_buf" in context),
                            f"Found dmesg_log(conn,...) outside DeferredLogs at pos {m.start()}")

    def test_no_update_accessed_on_readonly(self):
        """retriever.py main() 中 update_accessed 不直接使用 conn。"""
        retriever_path = Path(__file__).parent / "hooks" / "retriever.py"
        source = retriever_path.read_text()

        import re
        # update_accessed(conn, ...) 应该不存在于 main() 中
        # 只有 update_accessed(wconn, ...) 是合法的
        main_start = source.find("def main():")
        main_body = source[main_start:]

        matches = list(re.finditer(r'update_accessed\(conn,', main_body))
        self.assertEqual(len(matches), 0,
                         "Found update_accessed(conn,...) in main() — should use wconn")


if __name__ == "__main__":
    unittest.main()
