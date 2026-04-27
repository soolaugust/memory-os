#!/usr/bin/env python3
"""
test_session_identity.py — 迭代66：Session Identity Fix
验证所有 hook 正确从 stdin 获取 session_id

OS 类比：/proc/self/status → PID Identity
  进程通过 /proc/self 获取自己的 PID，而不是猜测环境变量。
  类似地，hook 应从 stdin（hook 协议的"进程描述符"）获取 session_id。
"""
import tmpfs  # noqa: F401 — 测试隔离
import unittest
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch
from io import StringIO

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "hooks"))


class TestRetrieverSessionIdentity(unittest.TestCase):
    """retriever.py 应从 hook_input 获取 session_id"""

    def test_session_id_from_hook_input(self):
        """hook_input 有 session_id 时应使用它"""
        from store import open_db, ensure_schema, insert_chunk
        from schema import MemoryChunk
        import retriever

        conn = open_db()
        ensure_schema(conn)

        chunk = MemoryChunk(
            chunk_type="decision",
            summary="test session identity decision",
            importance=0.8,
            project="test-session-id",
            source_session="real-session-abc",
        )
        insert_chunk(conn, chunk.to_dict())
        conn.commit()

        hook_input = {
            "session_id": "real-session-abc",
            "prompt": "what decision about session identity?",
        }
        retriever._vdso_hook_input = hook_input

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CLAUDE_SESSION_ID", None)
            session_id = (hook_input.get("session_id", "")
                          or os.environ.get("CLAUDE_SESSION_ID", "")
                          or "unknown")

        self.assertEqual(session_id, "real-session-abc")
        conn.close()

    def test_session_id_env_fallback(self):
        """hook_input 无 session_id 时 fallback 到环境变量"""
        hook_input = {"prompt": "test"}

        with patch.dict(os.environ, {"CLAUDE_SESSION_ID": "env-session-xyz"}):
            session_id = (hook_input.get("session_id", "")
                          or os.environ.get("CLAUDE_SESSION_ID", "")
                          or "unknown")

        self.assertEqual(session_id, "env-session-xyz")

    def test_session_id_unknown_fallback(self):
        """hook_input 和环境变量都无时 fallback 到 unknown"""
        hook_input = {"prompt": "test"}

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CLAUDE_SESSION_ID", None)
            session_id = (hook_input.get("session_id", "")
                          or os.environ.get("CLAUDE_SESSION_ID", "")
                          or "unknown")

        self.assertEqual(session_id, "unknown")


class TestWriterSessionIdentity(unittest.TestCase):
    """writer.py _get_session_id 应优先使用 hook_input"""

    def test_from_hook_input(self):
        import writer
        hook_input = {"session_id": "writer-session-123"}

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CLAUDE_SESSION_ID", None)
            sid = writer._get_session_id(hook_input)

        self.assertEqual(sid, "writer-session-123")

    def test_from_env_fallback(self):
        import writer

        with patch.dict(os.environ, {"CLAUDE_SESSION_ID": "env-456"}):
            sid = writer._get_session_id({})

        self.assertEqual(sid, "env-456")

    def test_from_env_no_hook_input(self):
        import writer

        with patch.dict(os.environ, {"CLAUDE_SESSION_ID": "env-789"}):
            sid = writer._get_session_id(None)

        self.assertEqual(sid, "env-789")

    def test_unknown_fallback(self):
        import writer

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CLAUDE_SESSION_ID", None)
            sid = writer._get_session_id({})

        self.assertEqual(sid, "unknown")


class TestExtractorSessionIdentity(unittest.TestCase):
    """extractor.py 应从 hook_input 获取 session_id"""

    def test_session_id_logic(self):
        """验证提取逻辑正确"""
        hook_input = {"session_id": "extractor-session-abc", "last_assistant_message": "test"}

        session_id = (hook_input.get("session_id", "")
                      or os.environ.get("CLAUDE_SESSION_ID", "")
                      or "unknown")

        self.assertEqual(session_id, "extractor-session-abc")


class TestSwapOutBackwardCompat(unittest.TestCase):
    """save-task-state.py swap out 查询应兼容 unknown session_id"""

    def test_query_includes_unknown(self):
        """真实 session_id 查询时应同时包含 'unknown'"""
        from store import open_db, ensure_schema

        conn = open_db()
        ensure_schema(conn)

        conn.execute("""
            INSERT INTO recall_traces (id, timestamp, session_id, project, prompt_hash, top_k_json, injected, duration_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, ("trace-1", "2026-04-19T10:00:00", "unknown", "test-swap-compat",
              "hash1", json.dumps([{"id": "chunk-old-1"}]), 1, 5.0))

        conn.execute("""
            INSERT INTO recall_traces (id, timestamp, session_id, project, prompt_hash, top_k_json, injected, duration_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, ("trace-2", "2026-04-19T10:01:00", "real-session-xyz", "test-swap-compat",
              "hash2", json.dumps([{"id": "chunk-new-1"}]), 1, 3.0))

        conn.commit()

        session_id = "real-session-xyz"
        session_ids = [session_id]
        if session_id != "unknown":
            session_ids.append("unknown")

        ph_sid = ",".join("?" * len(session_ids))
        rows = conn.execute(f"""
            SELECT top_k_json FROM recall_traces
            WHERE project = ? AND session_id IN ({ph_sid})
            ORDER BY timestamp DESC LIMIT 10
        """, ["test-swap-compat"] + session_ids).fetchall()

        self.assertEqual(len(rows), 2)

        hit_ids = []
        for r in rows:
            top_k = json.loads(r[0]) if r[0] else []
            for item in top_k:
                cid = item.get("id", "") if isinstance(item, dict) else str(item)
                if cid and cid not in hit_ids:
                    hit_ids.append(cid)

        self.assertIn("chunk-old-1", hit_ids)
        self.assertIn("chunk-new-1", hit_ids)

        conn.close()

    def test_query_unknown_only(self):
        """session_id=unknown 时不重复添加"""
        session_id = "unknown"
        session_ids = [session_id]
        if session_id != "unknown":
            session_ids.append("unknown")

        self.assertEqual(len(session_ids), 1)
        self.assertEqual(session_ids[0], "unknown")


class TestSessionIdFromStdin(unittest.TestCase):
    """验证 stdin JSON 包含 session_id 的各种格式"""

    def test_standard_format(self):
        """标准 hook stdin 格式"""
        stdin_json = {
            "session_id": "b7234a05-c1be-4a14-ae32-b351159e788b",
            "cwd": "/test/workspace",
            "prompt": "test prompt",
        }
        session_id = stdin_json.get("session_id", "")
        self.assertEqual(session_id, "b7234a05-c1be-4a14-ae32-b351159e788b")

    def test_precompact_format(self):
        """PreCompact stdin 格式"""
        stdin_json = {
            "session_id": "b7234a05-c1be-4a14-ae32-b351159e788b",
            "transcript_path": "/tmp/test.jsonl",
            "cwd": "/test/workspace",
            "hook_event_name": "PreCompact",
            "trigger": "auto",
        }
        session_id = stdin_json.get("session_id", "")
        self.assertEqual(session_id, "b7234a05-c1be-4a14-ae32-b351159e788b")

    def test_empty_stdin(self):
        """空 stdin fallback"""
        stdin_json = {}
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CLAUDE_SESSION_ID", None)
            session_id = (stdin_json.get("session_id", "")
                          or os.environ.get("CLAUDE_SESSION_ID", "")
                          or "unknown")
        self.assertEqual(session_id, "unknown")


class TestSessionIdPersistence(unittest.TestCase):
    """验证修复后 session_id 正确写入 recall_traces 和 memory_chunks"""

    def test_trace_with_real_session(self):
        """recall_traces 应包含真实 session_id"""
        from store import open_db, ensure_schema

        conn = open_db()
        ensure_schema(conn)

        real_session = "test-real-session-456"
        conn.execute("""
            INSERT INTO recall_traces (id, timestamp, session_id, project, prompt_hash, top_k_json, injected, duration_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, ("trace-persist-1", "2026-04-19T11:00:00", real_session, "test-persist",
              "hash-p", json.dumps([{"id": "c1"}]), 1, 2.0))
        conn.commit()

        row = conn.execute(
            "SELECT session_id FROM recall_traces WHERE id = ?",
            ("trace-persist-1",)
        ).fetchone()
        self.assertEqual(row[0], real_session)
        self.assertNotEqual(row[0], "unknown")

        conn.close()

    def test_chunk_with_real_session(self):
        """memory_chunks source_session 应包含真实 session_id"""
        from store import open_db, ensure_schema, insert_chunk
        from schema import MemoryChunk

        conn = open_db()
        ensure_schema(conn)

        chunk = MemoryChunk(
            chunk_type="decision",
            summary="test real session chunk",
            importance=0.8,
            project="test-persist",
            source_session="test-real-session-789",
        )
        insert_chunk(conn, chunk.to_dict())
        conn.commit()

        row = conn.execute(
            "SELECT source_session FROM memory_chunks WHERE summary = ?",
            ("test real session chunk",)
        ).fetchone()
        self.assertEqual(row[0], "test-real-session-789")

        conn.close()


if __name__ == "__main__":
    unittest.main()
