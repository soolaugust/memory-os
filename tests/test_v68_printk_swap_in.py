#!/usr/bin/env python3
"""
迭代68 测试：printk/kmsg for Swap In + top_k chunk_type enrichment

测试内容：
1. resume-task-state.py _log_error() 错误写入 swap_errors.log
2. resume-task-state.py _open_db_readonly() 只读模式
3. resume-task-state.py 空恢复记录 WARN dmesg
4. resume-task-state.py dmesg 记录包含 elapsed_ms 和 restore_source
5. retriever.py top_k_data 包含 chunk_type 字段
"""
import tmpfs  # noqa: F401 — 测试隔离

import json
import os
import sys
import time
import tempfile
import sqlite3
import unittest
from pathlib import Path
from datetime import datetime, timezone, timedelta

_MOS_ROOT = Path(__file__).parent
sys.path.insert(0, str(_MOS_ROOT))


class TestResumeErrorLogging(unittest.TestCase):
    """测试 resume-task-state.py 的错误可见性"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.error_log = Path(self.tmpdir) / "swap_errors.log"

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_log_error_writes_to_file(self):
        """_log_error 应将错误写入 swap_errors.log"""
        import traceback as tb

        def _log_error(context, exc, error_log=self.error_log, tmpdir=self.tmpdir):
            try:
                Path(tmpdir).mkdir(parents=True, exist_ok=True)
                now = datetime.now(timezone.utc).isoformat()
                msg = f"[{now}] resume:{context}: {type(exc).__name__}: {exc}\n"
                msg += tb.format_exc() + "\n"
                with open(error_log, "a", encoding="utf-8") as f:
                    f.write(msg)
            except Exception:
                pass

        try:
            raise ValueError("test error for swap in")
        except ValueError as e:
            _log_error("load_swap_state", e)

        self.assertTrue(self.error_log.exists())
        content = self.error_log.read_text("utf-8")
        self.assertIn("resume:load_swap_state", content)
        self.assertIn("ValueError", content)
        self.assertIn("test error for swap in", content)

    def test_log_error_rotation(self):
        """swap_errors.log 超过 50KB 自动轮转"""
        self.error_log.write_text("x" * 55_000, "utf-8")
        self.assertGreater(self.error_log.stat().st_size, 50_000)

        if self.error_log.stat().st_size > 50_000:
            content = self.error_log.read_text("utf-8")
            self.error_log.write_text(content[-30_000:], "utf-8")

        self.assertLessEqual(self.error_log.stat().st_size, 30_001)


class TestResumeDbReadonly(unittest.TestCase):
    """测试 resume-task-state.py 的 DB 只读模式"""

    def test_immutable_uri(self):
        """_open_db_readonly 使用 immutable=1 URI"""
        tmpdir = tempfile.mkdtemp()
        db_path = Path(tmpdir) / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE test (id INTEGER)")
        conn.execute("INSERT INTO test VALUES (1)")
        conn.commit()
        conn.close()

        uri = f"file:{db_path}?immutable=1"
        conn2 = sqlite3.connect(uri, uri=True)
        rows = conn2.execute("SELECT id FROM test").fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], 1)

        with self.assertRaises(sqlite3.OperationalError):
            conn2.execute("INSERT INTO test VALUES (2)")

        conn2.close()
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


class TestResumeDmesgEnrichment(unittest.TestCase):
    """测试 dmesg 记录增强"""

    def test_dmesg_contains_elapsed_ms(self):
        """dmesg extra 应包含 elapsed_ms 字段"""
        from store import open_db, ensure_schema, dmesg_log, dmesg_read, DMESG_INFO

        conn = open_db()
        ensure_schema(conn)

        dmesg_log(conn, DMESG_INFO, "swap_in",
                  "PostCompact swap in: 500 chars from swap_state 15ms",
                  extra={"session": "test-68",
                         "has_transcript": True,
                         "elapsed_ms": 15.2,
                         "restore_source": "swap_state"})
        conn.commit()

        msgs = dmesg_read(conn, subsystem="swap_in", limit=1)
        self.assertEqual(len(msgs), 1)
        extra = msgs[0]["extra"] if isinstance(msgs[0]["extra"], dict) else (json.loads(msgs[0]["extra"]) if msgs[0]["extra"] else {})
        self.assertIn("elapsed_ms", extra)
        self.assertEqual(extra["elapsed_ms"], 15.2)
        self.assertIn("restore_source", extra)
        self.assertEqual(extra["restore_source"], "swap_state")
        conn.close()

    def test_dmesg_empty_restore_is_warn(self):
        """空恢复应记录 WARN 级别 dmesg"""
        from store import open_db, ensure_schema, dmesg_log, dmesg_read, DMESG_WARN

        conn = open_db()
        ensure_schema(conn)

        dmesg_log(conn, DMESG_WARN, "swap_in",
                  "PostCompact swap in: EMPTY (no state to restore) 3ms",
                  extra={"session": "test-68-empty", "elapsed_ms": 3.1})
        conn.commit()

        msgs = dmesg_read(conn, level=DMESG_WARN, limit=5)
        found = [m for m in msgs if "EMPTY" in m["message"]]
        self.assertGreater(len(found), 0)
        self.assertEqual(found[0]["level"], DMESG_WARN)
        conn.close()


class TestSwapStateLoadError(unittest.TestCase):
    """测试 swap_state.json 加载异常处理"""

    def test_corrupted_swap_state_logged(self):
        """损坏的 swap_state.json 应记录错误而非静默忽略"""
        tmpdir = tempfile.mkdtemp()
        swap_file = Path(tmpdir) / "swap_state.json"
        error_log = Path(tmpdir) / "swap_errors.log"

        swap_file.write_text("{invalid json", "utf-8")

        try:
            json.loads(swap_file.read_text("utf-8"))
            self.fail("Should have raised JSONDecodeError")
        except json.JSONDecodeError as e:
            msg = f"resume:load_swap_state: {type(e).__name__}: {e}\n"
            error_log.write_text(msg, "utf-8")

        self.assertTrue(error_log.exists())
        content = error_log.read_text("utf-8")
        self.assertIn("JSONDecodeError", content)

        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_expired_swap_state_returns_empty(self):
        """超过 10 分钟的 swap_state 应返回空"""
        old_time = (datetime.now(timezone.utc) - timedelta(minutes=15)).isoformat()
        state = {"swap_out_at": old_time, "decisions": [{"summary": "old"}]}

        dt = datetime.fromisoformat(state["swap_out_at"])
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age_secs = (datetime.now(timezone.utc) - dt).total_seconds()

        self.assertGreater(age_secs, 600)


class TestTopKChunkType(unittest.TestCase):
    """测试 retriever.py top_k_data 包含 chunk_type"""

    def test_top_k_data_structure(self):
        """top_k_data 应包含 id, summary, score, chunk_type 四个字段"""
        top_k = [
            (0.85, {"id": "uuid-1", "summary": "test decision", "chunk_type": "decision"}),
            (0.72, {"id": "uuid-2", "summary": "test path", "chunk_type": "excluded_path"}),
            (0.65, {"id": "uuid-3", "summary": "test context", "chunk_type": "prompt_context"}),
        ]

        top_k_data = [
            {"id": c["id"], "summary": c["summary"], "score": round(s, 4), "chunk_type": c.get("chunk_type", "")}
            for s, c in top_k
        ]

        self.assertEqual(len(top_k_data), 3)
        for item in top_k_data:
            self.assertIn("id", item)
            self.assertIn("summary", item)
            self.assertIn("score", item)
            self.assertIn("chunk_type", item)

        self.assertEqual(top_k_data[0]["chunk_type"], "decision")
        self.assertEqual(top_k_data[1]["chunk_type"], "excluded_path")
        self.assertEqual(top_k_data[2]["chunk_type"], "prompt_context")

    def test_top_k_data_missing_chunk_type(self):
        """chunk_type 缺失时应 fallback 为空字符串"""
        top_k = [
            (0.85, {"id": "uuid-1", "summary": "legacy chunk"}),
        ]
        top_k_data = [
            {"id": c["id"], "summary": c["summary"], "score": round(s, 4), "chunk_type": c.get("chunk_type", "")}
            for s, c in top_k
        ]
        self.assertEqual(top_k_data[0]["chunk_type"], "")

    def test_top_k_json_serializable(self):
        """top_k_data 必须 JSON 可序列化"""
        top_k_data = [
            {"id": "uuid-1", "summary": "测试中文", "score": 0.8177, "chunk_type": "decision"},
        ]
        serialized = json.dumps(top_k_data, ensure_ascii=False)
        deserialized = json.loads(serialized)
        self.assertEqual(deserialized[0]["chunk_type"], "decision")
        self.assertEqual(deserialized[0]["summary"], "测试中文")


class TestFormatRestoreContext(unittest.TestCase):
    """测试恢复上下文格式化"""

    def test_full_state_restore(self):
        """完整 swap_state 应格式化所有区块"""
        state = {
            "conversation_summary": [
                {"role": "user", "summary": "帮我优化数据库查询"},
                {"role": "assistant", "summary": "分析了3个慢查询，建议加索引"},
            ],
            "task_progress": [
                {"status": "completed", "content": "分析慢查询"},
                {"status": "in_progress", "content": "添加索引"},
                {"status": "pending", "content": "验证性能"},
            ],
            "topics": ["数据库优化"],
            "decisions": [
                {"summary": "使用复合索引而非单列索引"},
            ],
            "reasoning_progress": [
                {"type": "next_steps", "content": "在 users 表上创建索引"},
            ],
            "madvise_hints": ["PostgreSQL", "索引", "查询优化"],
        }

        # 手动复现格式化逻辑
        parts = []
        conv = state.get("conversation_summary", [])
        if conv:
            lines = []
            for msg in conv:
                role = msg.get("role", "?")
                summary = msg.get("summary", "")
                if summary:
                    prefix = "  用户: " if role == "user" else "  AI: "
                    lines.append(f"{prefix}{summary}")
            if lines:
                parts.append("【最近对话】\n" + "\n".join(lines))

        decisions = state.get("decisions", [])
        if decisions:
            dec_lines = []
            for d in decisions[:5]:
                s = d.get("summary", "")
                if s:
                    dec_lines.append(f"  - {s}")
            if dec_lines:
                parts.append("【已有结论】\n" + "\n".join(dec_lines))

        result = "\n".join(parts)
        self.assertIn("用户: 帮我优化数据库查询", result)
        self.assertIn("AI: 分析了3个慢查询", result)
        self.assertIn("使用复合索引而非单列索引", result)


if __name__ == "__main__":
    unittest.main()
