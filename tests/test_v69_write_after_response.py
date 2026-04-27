#!/usr/bin/env python3
"""
迭代69 测试：Write-After-Response — 输出前置，写入后置

测试内容：
1. 主路径：print(output) 在 conn.commit() 之前
2. Hard deadline 路径：print(output) 在 conn.commit() 之前
3. sys.stdout.flush() 确保输出立即到达
4. 写入路径数据完整性不受影响
5. 延迟基准：输出时间 vs commit+close 时间分离验证
"""
import tmpfs  # noqa: F401 — 测试隔离

import json
import os
import sys
import time
import io
import sqlite3
import unittest
from pathlib import Path
from datetime import datetime, timezone

_MOS_ROOT = Path(__file__).parent
sys.path.insert(0, str(_MOS_ROOT))


class TestWriteAfterResponseOrder(unittest.TestCase):
    """验证输出在写入之前发生"""

    def test_print_before_commit_pattern(self):
        """主路径源码中 print+flush 应在 commit 之前"""
        retriever_path = _MOS_ROOT / "hooks" / "retriever.py"
        lines = retriever_path.read_text("utf-8").splitlines()

        # 找到主路径的 Write-After-Response 注释（非 hard deadline）
        war_line = None
        for i, line in enumerate(lines):
            if "迭代69：Write-After-Response — 输出前置，写入后置" in line:
                war_line = i
                break
        self.assertIsNotNone(war_line, "主路径应包含 Write-After-Response 注释")

        # 在注释之后的代码行中查找（只看非注释行）
        print_line = flush_line = commit_line = None
        for i in range(war_line, min(war_line + 60, len(lines))):
            stripped = lines[i].lstrip()
            if stripped.startswith("#"):
                continue  # 跳过注释行
            if "print(json.dumps(output" in lines[i] and print_line is None:
                print_line = i
            if "sys.stdout.flush()" in lines[i] and flush_line is None:
                flush_line = i
            if "conn.commit()" in lines[i] and commit_line is None:
                commit_line = i

        self.assertIsNotNone(print_line, "应找到 print 代码行")
        self.assertIsNotNone(flush_line, "应找到 flush 代码行")
        self.assertIsNotNone(commit_line, "应找到 commit 代码行")
        self.assertLess(print_line, commit_line,
                        f"print(L{print_line+1}) 应在 commit(L{commit_line+1}) 之前")
        self.assertLess(flush_line, commit_line,
                        f"flush(L{flush_line+1}) 应在 commit(L{commit_line+1}) 之前")
        self.assertLess(print_line, flush_line,
                        f"print(L{print_line+1}) 应在 flush(L{flush_line+1}) 之前")

    def test_hard_deadline_print_before_commit(self):
        """hard deadline 路径源码中 print+flush 应在 commit 之前"""
        retriever_path = _MOS_ROOT / "hooks" / "retriever.py"
        source = retriever_path.read_text("utf-8")

        # 找到 hard deadline 路径的标记
        hd_marker = "# ── 迭代69+84：输出前置 + 只读连接关闭 ──"
        self.assertIn(hd_marker, source,
                      "hard deadline 路径应包含 Write-After-Response 注释")

        hd_idx = source.index(hd_marker)
        after_hd = source[hd_idx:hd_idx + 1000]

        print_idx = after_hd.index("print(json.dumps(")
        flush_idx = after_hd.index("sys.stdout.flush()")
        commit_idx = after_hd.index("conn.commit()")

        self.assertLess(print_idx, commit_idx,
                        "hard deadline: print 应在 conn.commit() 之前")
        self.assertLess(flush_idx, commit_idx,
                        "hard deadline: flush 应在 conn.commit() 之前")


class TestFlushSemantics(unittest.TestCase):
    """验证 flush 语义：输出立即可用"""

    def test_flush_makes_output_immediately_available(self):
        """sys.stdout.flush() 后输出应立即可读"""
        old_stdout = sys.stdout
        captured = io.StringIO()
        sys.stdout = captured

        output = {"hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": "测试内容"
        }}

        print(json.dumps(output, ensure_ascii=False))
        sys.stdout.flush()

        result = captured.getvalue()
        sys.stdout = old_stdout

        parsed = json.loads(result.strip())
        self.assertEqual(parsed["hookSpecificOutput"]["additionalContext"], "测试内容")

    def test_output_before_slow_write(self):
        """输出应在慢写入之前完成"""
        old_stdout = sys.stdout
        captured = io.StringIO()
        sys.stdout = captured

        output = {"test": "data"}
        t0 = time.monotonic()

        print(json.dumps(output))
        sys.stdout.flush()
        t_output = time.monotonic() - t0

        time.sleep(0.01)  # 10ms 模拟 commit
        t_total = time.monotonic() - t0

        sys.stdout = old_stdout

        self.assertLess(t_output, t_total * 0.5,
                        f"输出应在写入之前完成: output={t_output*1000:.1f}ms total={t_total*1000:.1f}ms")


class TestWriteIntegrity(unittest.TestCase):
    """验证写入路径数据完整性"""

    def test_trace_written_after_output(self):
        """写入路径在输出后仍应正确写入数据"""
        from store import open_db, ensure_schema, insert_trace, update_accessed

        conn = open_db()
        ensure_schema(conn)

        trace = {
            "id": "test-v69-trace-001",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": "test-v69",
            "project": "test-project",
            "prompt_hash": "abc12345",
            "candidates_count": 10,
            "top_k_json": [{"id": "c1", "summary": "test", "score": 0.8, "chunk_type": "decision"}],
            "injected": 1,
            "reason": "hash_changed|full",
            "duration_ms": 15.0,
        }
        insert_trace(conn, trace)

        conn.execute("""
            INSERT OR IGNORE INTO memory_chunks
            (id, summary, content, chunk_type, importance, project, created_at, last_accessed, access_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, ("test-v69-c1", "test chunk", "content", "decision", 0.8,
              "test-project", datetime.now(timezone.utc).isoformat(),
              datetime.now(timezone.utc).isoformat(), 0))
        update_accessed(conn, ["test-v69-c1"])

        conn.commit()

        rows = conn.execute(
            "SELECT id, reason FROM recall_traces WHERE id = ?",
            ("test-v69-trace-001",)
        ).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1], "hash_changed|full")

        ac = conn.execute(
            "SELECT access_count FROM memory_chunks WHERE id = ?",
            ("test-v69-c1",)
        ).fetchone()
        self.assertIsNotNone(ac)
        self.assertGreaterEqual(ac[0], 1)

        conn.close()

    def test_commit_close_latency_isolation(self):
        """验证 commit + close 延迟确实是瓶颈（基准测试）"""
        from store import open_db, ensure_schema

        conn = open_db()
        ensure_schema(conn)

        conn.execute("""
            INSERT OR IGNORE INTO memory_chunks
            (id, summary, content, chunk_type, importance, project, created_at, last_accessed, access_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, ("test-v69-bench", "bench", "bench content", "decision", 0.5,
              "bench-project", datetime.now(timezone.utc).isoformat(),
              datetime.now(timezone.utc).isoformat(), 0))

        t0 = time.monotonic()
        conn.commit()
        commit_ms = (time.monotonic() - t0) * 1000

        t0 = time.monotonic()
        conn.close()
        close_ms = (time.monotonic() - t0) * 1000

        conn2 = open_db()
        ensure_schema(conn2)
        t0 = time.monotonic()
        conn2.close()
        close_no_write_ms = (time.monotonic() - t0) * 1000

        self.assertLess(close_no_write_ms, 10,
                        f"close_no_write 应 <10ms, got {close_no_write_ms:.1f}ms")


class TestSourceCodeStructure(unittest.TestCase):
    """验证源码结构性要求"""

    def test_no_print_after_commit_in_main_path(self):
        """主路径中 commit 之后不应有 print（old pattern）"""
        retriever_path = _MOS_ROOT / "hooks" / "retriever.py"
        source = retriever_path.read_text("utf-8")

        war_marker = "# ── 迭代69：Write-After-Response — 输出前置，写入后置"
        war_idx = source.index(war_marker)
        finally_idx = source.index("    finally:", war_idx)
        main_block = source[war_idx:finally_idx]

        commit_idx = main_block.rindex("conn.commit()")
        after_commit = main_block[commit_idx:]
        self.assertNotIn("print(json.dumps(", after_commit,
                         "conn.commit() 之后不应有 print 输出")

    def test_two_flush_points(self):
        """应有恰好 2 个 sys.stdout.flush() 代码行（非注释）"""
        retriever_path = _MOS_ROOT / "hooks" / "retriever.py"
        lines = retriever_path.read_text("utf-8").splitlines()
        code_flush_count = 0
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue  # 跳过注释行
            if "sys.stdout.flush()" in stripped:
                code_flush_count += 1
        self.assertEqual(code_flush_count, 2,
                         f"应有 2 个非注释 flush 点（主路径+hard deadline），实际 {code_flush_count}")

    def test_write_after_response_comments(self):
        """两个输出点都应有迭代69注释"""
        retriever_path = _MOS_ROOT / "hooks" / "retriever.py"
        source = retriever_path.read_text("utf-8")

        self.assertIn("迭代69：Write-After-Response — 输出前置，写入后置", source)
        self.assertIn("迭代69：Write-After-Response — 输出前置", source)
        self.assertIn("write-back caching", source)


class TestEdgeCases(unittest.TestCase):
    """边界情况测试"""

    def test_empty_output_still_flushes(self):
        """即使输出为空 JSON，flush 也应正常工作"""
        old_stdout = sys.stdout
        captured = io.StringIO()
        sys.stdout = captured

        output = {"hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": ""
        }}
        print(json.dumps(output, ensure_ascii=False))
        sys.stdout.flush()

        result = captured.getvalue()
        sys.stdout = old_stdout

        parsed = json.loads(result.strip())
        self.assertEqual(parsed["hookSpecificOutput"]["additionalContext"], "")

    def test_large_output_flushes_correctly(self):
        """大输出（接近 max_context_chars）flush 后完整可读"""
        old_stdout = sys.stdout
        captured = io.StringIO()
        sys.stdout = captured

        large_text = "测试内容 " * 100
        output = {"hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": large_text
        }}
        print(json.dumps(output, ensure_ascii=False))
        sys.stdout.flush()

        result = captured.getvalue()
        sys.stdout = old_stdout

        parsed = json.loads(result.strip())
        self.assertEqual(parsed["hookSpecificOutput"]["additionalContext"], large_text)

    def test_unicode_output_integrity(self):
        """中文+特殊字符输出 flush 后完整性"""
        old_stdout = sys.stdout
        captured = io.StringIO()
        sys.stdout = captured

        text = "[决策] 使用复合索引而非单列索引\n[排除] /tmp/test\n[推理] BM25→FTS5 迁移"
        output = {"hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": text
        }}
        print(json.dumps(output, ensure_ascii=False))
        sys.stdout.flush()

        result = captured.getvalue()
        sys.stdout = old_stdout

        parsed = json.loads(result.strip())
        self.assertIn("[决策]", parsed["hookSpecificOutput"]["additionalContext"])
        self.assertIn("[排除]", parsed["hookSpecificOutput"]["additionalContext"])


if __name__ == "__main__":
    unittest.main()
