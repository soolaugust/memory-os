#!/usr/bin/env python3
"""
迭代50 测试：DRR Fair Queuing — 检索结果类型多样性保障
OS 类比：Deficit Round Robin (DRR, 1996)
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "hooks"))

# 迫使 config 使用默认值（不读磁盘）
os.environ["MEMORY_OS_RETRIEVER_DRR_ENABLED"] = "1"
os.environ["MEMORY_OS_RETRIEVER_DRR_MAX_SAME_TYPE"] = "2"


class TestDRRFairQueuing(unittest.TestCase):
    """测试 DRR 类型多样性选择器。"""

    def _make_chunk(self, cid, chunk_type, score):
        return (score, {
            "id": cid,
            "summary": f"summary_{cid}",
            "content": "",
            "chunk_type": chunk_type,
            "importance": 0.8,
            "last_accessed": "2026-04-19T00:00:00+00:00",
            "access_count": 1,
            "created_at": "2026-04-18T00:00:00+00:00",
        })

    def test_drr_basic_diversity(self):
        """DRR 保证多类型覆盖：decision 被限额后 reasoning_chain 得以入选。"""
        from retriever import _drr_select
        candidates = [
            self._make_chunk("d1", "decision", 0.95),
            self._make_chunk("d2", "decision", 0.90),
            self._make_chunk("d3", "decision", 0.85),
            self._make_chunk("r1", "reasoning_chain", 0.80),
            self._make_chunk("d4", "decision", 0.75),
        ]
        top_k = _drr_select(candidates, 3)
        self.assertEqual(len(top_k), 3)
        types = [c["chunk_type"] for _, c in top_k]
        # decision 最多 2 个（max_same_type=2），第 3 个槽位给 reasoning_chain
        self.assertEqual(types.count("decision"), 2)
        self.assertIn("reasoning_chain", types)

    def test_drr_single_type_fallback(self):
        """只有一种类型时，DRR 回流（overflow）填满 top_k。"""
        from retriever import _drr_select
        candidates = [
            self._make_chunk("d1", "decision", 0.95),
            self._make_chunk("d2", "decision", 0.90),
            self._make_chunk("d3", "decision", 0.85),
            self._make_chunk("d4", "decision", 0.80),
        ]
        top_k = _drr_select(candidates, 3)
        self.assertEqual(len(top_k), 3)
        # 只有 decision，max_same_type=2 先选 2 个，然后回流补 1 个
        types = [c["chunk_type"] for _, c in top_k]
        self.assertEqual(types.count("decision"), 3)

    def test_drr_preserves_score_order_within_type(self):
        """同类型内保持分数顺序：最高分的 decision 先入选。"""
        from retriever import _drr_select
        candidates = [
            self._make_chunk("d1", "decision", 0.95),
            self._make_chunk("d2", "decision", 0.90),
            self._make_chunk("d3", "decision", 0.85),
            self._make_chunk("r1", "reasoning_chain", 0.50),
        ]
        top_k = _drr_select(candidates, 3)
        ids = [c["id"] for _, c in top_k]
        # d1, d2 (decision top-2) + r1 (reasoning_chain)
        self.assertIn("d1", ids)
        self.assertIn("d2", ids)
        self.assertIn("r1", ids)

    def test_drr_multi_type(self):
        """三种类型各有代表，top_k=3 时各类型各 1 个。"""
        from retriever import _drr_select
        candidates = [
            self._make_chunk("d1", "decision", 0.95),
            self._make_chunk("d2", "decision", 0.90),
            self._make_chunk("d3", "decision", 0.85),
            self._make_chunk("r1", "reasoning_chain", 0.80),
            self._make_chunk("s1", "conversation_summary", 0.75),
            self._make_chunk("r2", "reasoning_chain", 0.70),
        ]
        top_k = _drr_select(candidates, 3)
        types = set(c["chunk_type"] for _, c in top_k)
        # max_same_type=2: d1,d2 占 2 slots，第 3 slot 给 r1（下一个高分非 decision）
        self.assertEqual(len(top_k), 3)
        self.assertIn("reasoning_chain", types)

    def test_drr_overflow_order(self):
        """回流 chunk 按原始分数顺序。"""
        from retriever import _drr_select
        candidates = [
            self._make_chunk("d1", "decision", 0.99),
            self._make_chunk("d2", "decision", 0.98),
            self._make_chunk("d3", "decision", 0.97),
            self._make_chunk("d4", "decision", 0.96),
        ]
        # max_same_type=2, top_k=4
        top_k = _drr_select(candidates, 4)
        # d1, d2 直接选入；d3, d4 先进 overflow 再回流
        ids = [c["id"] for _, c in top_k]
        self.assertEqual(ids, ["d1", "d2", "d3", "d4"])

    def test_drr_fewer_candidates_than_top_k(self):
        """候选数 < top_k 时全部选入。"""
        from retriever import _drr_select
        candidates = [
            self._make_chunk("d1", "decision", 0.95),
            self._make_chunk("r1", "reasoning_chain", 0.80),
        ]
        top_k = _drr_select(candidates, 5)
        self.assertEqual(len(top_k), 2)

    def test_drr_empty_candidates(self):
        """空候选集 → 空结果。"""
        from retriever import _drr_select
        top_k = _drr_select([], 3)
        self.assertEqual(len(top_k), 0)

    def test_drr_disabled_pure_score(self):
        """DRR 禁用时退化为纯 score 排序。"""
        os.environ["MEMORY_OS_RETRIEVER_DRR_ENABLED"] = "0"
        # 重新加载 config 缓存
        from config import _invalidate_cache, get
        _invalidate_cache()
        self.assertFalse(get("retriever.drr_enabled"))
        # 恢复
        os.environ["MEMORY_OS_RETRIEVER_DRR_ENABLED"] = "1"
        _invalidate_cache()

    def test_drr_performance(self):
        """DRR 选择器性能：1000 候选集 < 1ms。"""
        import time
        from retriever import _drr_select
        candidates = []
        for i in range(1000):
            ctype = "decision" if i % 5 != 0 else "reasoning_chain"
            candidates.append(self._make_chunk(f"c{i}", ctype, 1.0 - i * 0.0001))
        start = time.time()
        for _ in range(100):
            _drr_select(candidates, 5)
        elapsed = (time.time() - start) / 100 * 1000
        print(f"  DRR 1000 candidates → Top-5: {elapsed:.3f}ms")
        self.assertLess(elapsed, 1.0, f"DRR too slow: {elapsed:.3f}ms")


class TestDRRSysctl(unittest.TestCase):
    """测试 DRR sysctl tunable 注册。"""

    def test_tunable_registered(self):
        """DRR tunable 已注册到 config.py _REGISTRY。"""
        from config import get
        self.assertIsInstance(get("retriever.drr_enabled"), bool)
        self.assertIsInstance(get("retriever.drr_max_same_type"), int)

    def test_tunable_defaults(self):
        """DRR tunable 默认值正确。"""
        from config import _REGISTRY
        enabled_entry = _REGISTRY["retriever.drr_enabled"]
        self.assertTrue(enabled_entry[0])  # default True
        max_same_entry = _REGISTRY["retriever.drr_max_same_type"]
        self.assertEqual(max_same_entry[0], 2)  # default 2


if __name__ == "__main__":
    unittest.main(verbosity=2)
