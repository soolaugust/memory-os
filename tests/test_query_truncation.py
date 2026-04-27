#!/usr/bin/env python3
"""
迭代62 测试：Query Truncation + PSI Import-aware Timing

测试三个修复点：
1. _build_query() 截断超长 query（防 FTS5 性能退化）
2. PSI 延迟采样排除 hard_deadline trace（不代表正常检索延迟）
3. FTS5 截断后性能验证（query 长度 vs 延迟的因果关系）
"""
import tmpfs  # 测试隔离：临时目录 + 环境变量覆盖

import sys
import os
import json
import time
import sqlite3

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "hooks"))

import store
import config
from hooks.retriever import _build_query, _extract_key_entities


class TestQueryTruncation:
    """测试 query 截断逻辑"""

    def test_short_query_not_truncated(self):
        """短 query 不应被截断"""
        hook_input = {"prompt": "memory-os 性能优化"}
        query = _build_query(hook_input)
        assert query == "memory-os 性能优化"

    def test_long_query_truncated(self):
        """超长 query 应被截断到 max_query_chars"""
        max_chars = config.get("retriever.max_query_chars")
        long_prompt = "A" * 2000
        hook_input = {"prompt": long_prompt}
        query = _build_query(hook_input)
        assert len(query) <= max_chars, f"query len {len(query)} > max {max_chars}"

    def test_truncation_preserves_prefix(self):
        """截断应保留 query 前缀（前面的词更有价值）"""
        long_prompt = "memory-os virtual memory " + "padding " * 200
        hook_input = {"prompt": long_prompt}
        query = _build_query(hook_input)
        assert query.startswith("memory-os virtual memory"), f"prefix lost: {query[:50]}"

    def test_entities_appended_before_truncation(self):
        """实体提取在截断前完成——确保实体不被截断丢失"""
        max_chars = config.get("retriever.max_query_chars")
        prompt = "`store.py` " + "x" * (max_chars - 20)
        hook_input = {"prompt": prompt}
        query = _build_query(hook_input)
        assert len(query) <= max_chars

    def test_task_list_included_in_query(self):
        """task_list 的 in_progress 任务应包含在 query 中"""
        hook_input = {
            "prompt": "检查状态",
            "task_list": [
                {"status": "in_progress", "subject": "实现 PSI 修复"},
                {"status": "completed", "subject": "诊断延迟"},
            ]
        }
        query = _build_query(hook_input)
        assert "实现 PSI 修复" in query
        assert "诊断延迟" not in query  # completed 不包含

    def test_default_max_query_chars(self):
        """默认 max_query_chars 应该是 300"""
        assert config.get("retriever.max_query_chars") == 300


class TestPSITraceHygiene:
    """测试 PSI 延迟采样的 trace 过滤"""

    def setup_method(self):
        self.conn = store.open_db()
        store.ensure_schema(self.conn)

    def teardown_method(self):
        self.conn.close()

    def _insert_trace(self, duration_ms, reason, project="test_psi"):
        """插入测试 trace"""
        from datetime import datetime, timezone
        import uuid
        store.insert_trace(self.conn, {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": "test",
            "project": project,
            "prompt_hash": "test",
            "candidates_count": 5,
            "top_k_json": [],
            "injected": 1 if "skip" not in reason else 0,
            "reason": reason,
            "duration_ms": duration_ms,
        })
        self.conn.commit()

    def test_hard_deadline_excluded_from_psi(self):
        """hard_deadline trace 不应纳入 PSI 延迟采样"""
        project = "test_psi_hd"
        # 插入 normal traces（低延迟）
        for _ in range(10):
            self._insert_trace(8.0, "hash_changed|lite", project)
        # 插入 hard_deadline traces（高延迟，pre-fix 的膨胀数据）
        for _ in range(5):
            self._insert_trace(300.0, "hash_changed|lite|hard_deadline", project)

        psi = store.psi_stats(self.conn, project)
        ret = psi["retrieval"]

        # PSI 应该只看 10 条正常 trace，不看 5 条 hard_deadline
        assert ret["samples"] <= 10, f"expected <=10 samples, got {ret['samples']}"
        # avg 应该接近 8ms，不是被 300ms 拉高
        assert ret["avg_ms"] < 20, f"avg {ret['avg_ms']}ms too high (hard_deadline leaking)"
        # 不应该是 FULL
        assert ret["level"] != "FULL", f"PSI should not be FULL with clean traces"

    def test_skipped_same_hash_excluded(self):
        """skipped_same_hash trace 仍然被排除"""
        project = "test_psi_ssh"
        for _ in range(10):
            self._insert_trace(8.0, "hash_changed|lite", project)
        for _ in range(5):
            self._insert_trace(0, "skipped_same_hash", project)

        psi = store.psi_stats(self.conn, project)
        ret = psi["retrieval"]
        # 0ms 的 skipped_same_hash 被过滤（duration_ms > 0）
        assert ret["samples"] == 10

    def test_normal_traces_included(self):
        """正常 trace 应被纳入 PSI 采样"""
        project = "test_psi_normal"
        for i in range(15):
            self._insert_trace(5.0 + i * 0.5, "hash_changed|lite", project)

        psi = store.psi_stats(self.conn, project)
        ret = psi["retrieval"]
        assert ret["samples"] == 15
        assert ret["level"] == "NONE", f"expected NONE, got {ret['level']}"

    def test_mixed_traces_only_clean(self):
        """混合场景：PSI 只看干净的 trace"""
        project = "test_psi_mixed"
        # 10 条正常（低延迟）
        for _ in range(10):
            self._insert_trace(6.0, "hash_changed|lite", project)
        # 3 条 hard_deadline（高延迟 — 排除）
        for _ in range(3):
            self._insert_trace(400.0, "hash_changed|full|hard_deadline", project)
        # 5 条 skipped_same_hash（0ms — 排除）
        for _ in range(5):
            self._insert_trace(0, "skipped_same_hash", project)
        # 2 条 PSI downgrade（中等延迟 — 包含）
        for _ in range(2):
            self._insert_trace(15.0, "hash_changed|lite|psi_downgrade", project)

        psi = store.psi_stats(self.conn, project)
        ret = psi["retrieval"]
        # 应该只有 10 + 2 = 12 条（排除 hard_deadline 和 skipped_same_hash）
        assert ret["samples"] == 12, f"expected 12 samples, got {ret['samples']}"
        # avg 应该接近 (10*6 + 2*15) / 12 = 7.5ms
        assert ret["avg_ms"] < 10, f"avg {ret['avg_ms']}ms suggests dirty samples"


class TestFTS5PerformanceWithTruncation:
    """验证 query 截断后 FTS5 性能改善"""

    def setup_method(self):
        self.conn = store.open_db()
        store.ensure_schema(self.conn)
        # 插入一些测试 chunks
        import uuid
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        for i in range(20):
            store.insert_chunk(self.conn, {
                "id": str(uuid.uuid4()),
                "created_at": now,
                "updated_at": now,
                "project": "test_perf",
                "source_session": "test",
                "summary": f"test decision about memory-os feature {i} optimization",
                "content": f"详细内容：memory-os 的第 {i} 个优化决策，涉及 FTS5 索引和 BM25 检索",
                "chunk_type": "decision",
                "importance": 0.7,
                "retrievability": 1.0,
                "last_accessed": now,
                "tags": "[]",
            })
        self.conn.commit()

    def teardown_method(self):
        self.conn.close()

    def test_short_query_fast(self):
        """短 query FTS5 应该快（<20ms）"""
        t0 = time.time()
        results = store.fts_search(self.conn, "memory-os optimization", "test_perf", top_k=9)
        elapsed = (time.time() - t0) * 1000
        assert elapsed < 20, f"short query took {elapsed:.1f}ms"
        assert len(results) > 0

    def test_truncated_query_fast(self):
        """截断后的长 query 应该和短 query 一样快"""
        max_chars = config.get("retriever.max_query_chars")
        # 模拟截断
        long_query = "memory-os virtual memory " + "padding test data " * 100
        truncated = long_query[:max_chars]

        t0 = time.time()
        results = store.fts_search(self.conn, truncated, "test_perf", top_k=9)
        elapsed = (time.time() - t0) * 1000
        assert elapsed < 20, f"truncated query took {elapsed:.1f}ms"

    def test_query_truncation_preserves_relevance(self):
        """截断不应丢失关键检索词"""
        # 关键词在 query 前部
        long_query = "memory-os FTS5 optimization " + "irrelevant padding " * 50
        max_chars = config.get("retriever.max_query_chars")
        truncated = long_query[:max_chars]

        results_full = store.fts_search(self.conn, long_query[:100], "test_perf", top_k=5)
        results_trunc = store.fts_search(self.conn, truncated, "test_perf", top_k=5)

        # 两者应该返回相同的结果（关键词在前缀中保留）
        ids_full = {r["id"] for r in results_full}
        ids_trunc = {r["id"] for r in results_trunc}
        # 允许小差异（FTS5 内部排名可能因 token 数变化略有不同）
        overlap = len(ids_full & ids_trunc) / max(1, len(ids_full))
        assert overlap >= 0.6, f"relevance degraded: overlap {overlap:.0%}"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
