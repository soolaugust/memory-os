#!/usr/bin/env python3
"""
迭代48 测试：Readahead — Co-Access Prefetch
OS 类比：Linux readahead (generic_file_readahead, 2002→2004)
"""
import json
import os
import sys
import uuid
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path

# 确保能 import memory-os 模块
sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("CLAUDE_CWD", str(Path(__file__).parent))

import tmpfs  # noqa: F401 — tmpfs isolation (iter54), must precede store import
from store import open_db, ensure_schema, insert_chunk, insert_trace, readahead_pairs
from schema import MemoryChunk
from config import get as sysctl


def _make_chunk(conn, project, chunk_type="decision", summary="test", importance=0.8):
    chunk_id = str(uuid.uuid4())
    chunk = MemoryChunk(
        id=chunk_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        project=project,
        source_session="test-session",
        chunk_type=chunk_type,
        content=summary,
        summary=summary,
        tags=[],
        importance=importance,
    )
    insert_chunk(conn, chunk.to_dict())
    return chunk_id


def _make_trace(conn, project, chunk_ids, session_id="test-session"):
    """模拟一条 recall_trace，top_k_json 包含指定的 chunk_ids。"""
    top_k = [{"id": cid, "summary": f"chunk-{cid[:8]}", "score": 0.9} for cid in chunk_ids]
    insert_trace(conn, {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "project": project,
        "prompt_hash": "test_hash",
        "candidates_count": len(chunk_ids),
        "top_k_json": top_k,
        "injected": 1,
        "reason": "test",
        "duration_ms": 1.0,
    })


class TestReadahead(unittest.TestCase):
    """Readahead Co-Access Prefetch 测试套件"""

    def setUp(self):
        self.conn = open_db(Path(":memory:"))
        ensure_schema(self.conn)
        self.project = "test-readahead"

    def tearDown(self):
        self.conn.close()

    def test_01_no_traces_returns_empty(self):
        """无 recall_traces 时 readahead_pairs 返回空 dict"""
        result = readahead_pairs(self.conn, self.project)
        self.assertEqual(result, {})

    def test_02_single_trace_no_pairs(self):
        """只有一条 trace 且只有 1 个 chunk，不产生 pair"""
        cid = _make_chunk(self.conn, self.project, summary="single chunk")
        self.conn.commit()
        _make_trace(self.conn, self.project, [cid])
        self.conn.commit()
        result = readahead_pairs(self.conn, self.project)
        self.assertEqual(result, {})

    def test_03_cooccurrence_below_threshold(self):
        """共现次数未达 min_cooccurrence 阈值时不建立 pair"""
        c1 = _make_chunk(self.conn, self.project, summary="chunk A")
        c2 = _make_chunk(self.conn, self.project, summary="chunk B")
        self.conn.commit()
        # 只共现 1 次（默认 min_cooccurrence=2）
        _make_trace(self.conn, self.project, [c1, c2])
        self.conn.commit()
        result = readahead_pairs(self.conn, self.project)
        self.assertEqual(result, {})

    def test_04_cooccurrence_meets_threshold(self):
        """共现次数达到阈值时成功建立 pair"""
        c1 = _make_chunk(self.conn, self.project, summary="选择 React 框架")
        c2 = _make_chunk(self.conn, self.project, summary="排除 Vue 框架")
        self.conn.commit()
        # 共现 2 次（达到默认 min_cooccurrence=2）
        _make_trace(self.conn, self.project, [c1, c2])
        _make_trace(self.conn, self.project, [c1, c2])
        self.conn.commit()
        result = readahead_pairs(self.conn, self.project)
        # c1 应该有 c2 作为 partner，反之亦然
        self.assertIn(c1, result)
        self.assertIn(c2, result)
        self.assertEqual(result[c1][0][0], c2)
        self.assertEqual(result[c2][0][0], c1)
        self.assertEqual(result[c1][0][1], 2)  # cooccurrence count = 2

    def test_05_three_chunk_cluster(self):
        """3 个 chunk 频繁共现，形成完整的 pair 网络"""
        c1 = _make_chunk(self.conn, self.project, summary="arch: microservices")
        c2 = _make_chunk(self.conn, self.project, summary="tech: gRPC")
        c3 = _make_chunk(self.conn, self.project, summary="rejected: REST for internal")
        self.conn.commit()
        # 3 次共现
        for _ in range(3):
            _make_trace(self.conn, self.project, [c1, c2, c3])
        self.conn.commit()
        result = readahead_pairs(self.conn, self.project)
        # 每个 chunk 应有 2 个 partner
        self.assertEqual(len(result[c1]), 2)
        self.assertEqual(len(result[c2]), 2)
        self.assertEqual(len(result[c3]), 2)
        # 共现次数都是 3
        for partners in result.values():
            for _, cnt in partners:
                self.assertEqual(cnt, 3)

    def test_06_hit_ids_filter(self):
        """指定 hit_ids 时只返回与 hit_ids 相关的 pair"""
        c1 = _make_chunk(self.conn, self.project, summary="hit chunk")
        c2 = _make_chunk(self.conn, self.project, summary="partner chunk")
        c3 = _make_chunk(self.conn, self.project, summary="unrelated chunk")
        self.conn.commit()
        # c1-c2 共现 3 次，c3 与 c1 共现 0 次
        for _ in range(3):
            _make_trace(self.conn, self.project, [c1, c2])
        self.conn.commit()
        # 只查 hit_ids=[c1] 的 pair
        result = readahead_pairs(self.conn, self.project, hit_ids=[c1])
        self.assertIn(c1, result)
        self.assertEqual(result[c1][0][0], c2)
        self.assertNotIn(c3, result)

    def test_07_hit_ids_excludes_already_hit(self):
        """hit_ids 中已包含的 partner 不出现在 prefetch 列表"""
        c1 = _make_chunk(self.conn, self.project, summary="chunk 1")
        c2 = _make_chunk(self.conn, self.project, summary="chunk 2")
        self.conn.commit()
        for _ in range(3):
            _make_trace(self.conn, self.project, [c1, c2])
        self.conn.commit()
        # 两个都在 hit_ids 中 — 不需要 prefetch
        result = readahead_pairs(self.conn, self.project, hit_ids=[c1, c2])
        # c1 的 partner c2 已在 hit_ids 中 → 被过滤
        # c2 的 partner c1 已在 hit_ids 中 → 被过滤
        self.assertEqual(result, {})

    def test_08_project_isolation(self):
        """不同 project 的 traces 不互相影响"""
        c1 = _make_chunk(self.conn, "project-A", summary="A chunk 1")
        c2 = _make_chunk(self.conn, "project-A", summary="A chunk 2")
        c3 = _make_chunk(self.conn, "project-B", summary="B chunk 1")
        c4 = _make_chunk(self.conn, "project-B", summary="B chunk 2")
        self.conn.commit()
        for _ in range(3):
            _make_trace(self.conn, "project-A", [c1, c2])
            _make_trace(self.conn, "project-B", [c3, c4])
        self.conn.commit()
        result_a = readahead_pairs(self.conn, "project-A")
        result_b = readahead_pairs(self.conn, "project-B")
        # A 只看到 A 的 pair
        self.assertIn(c1, result_a)
        self.assertNotIn(c3, result_a)
        # B 只看到 B 的 pair
        self.assertIn(c3, result_b)
        self.assertNotIn(c1, result_b)

    def test_09_sorted_by_cooccurrence(self):
        """partner 列表按共现次数降序排列"""
        c1 = _make_chunk(self.conn, self.project, summary="center chunk")
        c2 = _make_chunk(self.conn, self.project, summary="frequent partner")
        c3 = _make_chunk(self.conn, self.project, summary="less frequent partner")
        self.conn.commit()
        # c1-c2 共现 5 次
        for _ in range(5):
            _make_trace(self.conn, self.project, [c1, c2])
        # c1-c3 共现 2 次
        for _ in range(2):
            _make_trace(self.conn, self.project, [c1, c3])
        self.conn.commit()
        result = readahead_pairs(self.conn, self.project)
        partners = result[c1]
        # c2(5次) 应排在 c3(2次) 前面
        self.assertEqual(partners[0][0], c2)
        self.assertEqual(partners[0][1], 5)
        self.assertEqual(partners[1][0], c3)
        self.assertEqual(partners[1][1], 2)

    def test_10_performance(self):
        """性能测试：50 条 trace × 3 chunks → readahead_pairs < 10ms"""
        import time
        chunks = [_make_chunk(self.conn, self.project, summary=f"chunk-{i}") for i in range(10)]
        self.conn.commit()
        # 50 条 trace，每条随机 3 个 chunk
        import random
        random.seed(42)
        for _ in range(50):
            sample = random.sample(chunks, 3)
            _make_trace(self.conn, self.project, sample)
        self.conn.commit()

        t0 = time.time()
        for _ in range(100):
            readahead_pairs(self.conn, self.project, hit_ids=chunks[:3])
        elapsed = (time.time() - t0) * 1000 / 100
        print(f"  readahead_pairs avg: {elapsed:.2f}ms")
        self.assertLess(elapsed, 10.0, f"readahead_pairs too slow: {elapsed:.2f}ms")

    def test_11_sysctl_tunables_registered(self):
        """验证 4 个 readahead.* tunable 已注册"""
        self.assertIsInstance(sysctl("readahead.min_cooccurrence"), int)
        self.assertIsInstance(sysctl("readahead.prefetch_bonus"), float)
        self.assertIsInstance(sysctl("readahead.max_prefetch"), int)
        self.assertIsInstance(sysctl("readahead.window_traces"), int)


if __name__ == "__main__":
    unittest.main(verbosity=2)
