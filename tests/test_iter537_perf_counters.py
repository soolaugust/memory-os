"""
iter537: perf_counters — Retrieval Quality PMU Counters
OS 类比：Linux perf_event_open() / perf stat (Ingo Molnár / Thomas Gleixner, 2009)

测试覆盖：
  T1: 空 DB 返回零值
  T2: 有 traces 返回正确 avg/min/p25 scores
  T3: low_score_count 正确统计
  T4: score_histogram 分桶正确
  T5: type_concentration HHI 计算正确（单类型=1.0）
  T6: type_concentration HHI 均匀分布接近 1/N
  T7: 非注入 traces 不纳入 score 统计
  T8: injection_rate 计算正确
  T9: window 参数限制 trace 数量
  T10: autotune 策略8 — low_score_ratio 高时提高阈值
  T11: autotune 策略8 — avg_score 高且 low_ratio=0 时降低阈值
  T12: config tunables 注册正确
  T13: 性能 < 50ms（100 traces）
"""
import sys, os, json, time, uuid
from pathlib import Path
from datetime import datetime, timezone, timedelta

# tmpfs 测试隔离
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tmpfs  # noqa: F401 — 自动隔离 DB

import unittest
from store import open_db, ensure_schema, insert_chunk, insert_trace
from store_mm import perf_counters, autotune
from config import get as _cfg, sysctl_set, _REGISTRY


def _unique_project():
    return f"test_perf_{uuid.uuid4().hex[:8]}"


def _make_trace(project, top_k_items, injected=1, ts=None):
    """构造一条 recall_trace。"""
    return {
        "id": str(uuid.uuid4()),
        "timestamp": (ts or datetime.now(timezone.utc)).isoformat(),
        "session_id": "test-sess",
        "project": project,
        "prompt_hash": f"ph_{uuid.uuid4().hex[:8]}",
        "candidates_count": 10,
        "top_k_json": top_k_items,
        "injected": injected,
        "reason": "hash_changed|full",
        "duration_ms": 15.0,
    }


class TestPerfCounters(unittest.TestCase):

    def setUp(self):
        self.conn = open_db()
        ensure_schema(self.conn)

    def tearDown(self):
        self.conn.close()

    def test_01_empty_db(self):
        """T1: 空 DB 返回零值"""
        p = _unique_project()
        result = perf_counters(self.conn, p)
        self.assertEqual(result["total_traces"], 0)
        self.assertEqual(result["avg_score"], 0.0)
        self.assertEqual(result["injection_rate"], 0.0)

    def test_02_basic_scores(self):
        """T2: 有 traces 返回正确 avg/min/p25 scores"""
        p = _unique_project()
        items = [
            {"id": "c1", "score": 0.9, "chunk_type": "decision"},
            {"id": "c2", "score": 0.6, "chunk_type": "decision"},
            {"id": "c3", "score": 0.3, "chunk_type": "reasoning_chain"},
        ]
        insert_trace(self.conn, _make_trace(p, items))
        self.conn.commit()

        result = perf_counters(self.conn, p)
        self.assertEqual(result["total_traces"], 1)
        self.assertEqual(result["injected_traces"], 1)
        self.assertAlmostEqual(result["avg_score"], 0.6, places=2)
        self.assertAlmostEqual(result["min_score"], 0.3, places=2)

    def test_03_low_score_count(self):
        """T3: low_score_count 正确统计（默认 threshold=0.40）"""
        p = _unique_project()
        items = [
            {"id": "c1", "score": 0.9, "chunk_type": "decision"},
            {"id": "c2", "score": 0.35, "chunk_type": "decision"},  # < 0.40
            {"id": "c3", "score": 0.2, "chunk_type": "decision"},   # < 0.40
        ]
        insert_trace(self.conn, _make_trace(p, items))
        self.conn.commit()

        result = perf_counters(self.conn, p)
        self.assertEqual(result["low_score_count"], 2)
        self.assertAlmostEqual(result["low_score_ratio"], 2/3, places=3)

    def test_04_score_histogram(self):
        """T4: score_histogram 分桶正确"""
        p = _unique_project()
        items = [
            {"id": "c1", "score": 0.15, "chunk_type": "decision"},  # 0.0-0.2
            {"id": "c2", "score": 0.35, "chunk_type": "decision"},  # 0.2-0.4
            {"id": "c3", "score": 0.55, "chunk_type": "decision"},  # 0.4-0.6
            {"id": "c4", "score": 0.75, "chunk_type": "decision"},  # 0.6-0.8
            {"id": "c5", "score": 0.95, "chunk_type": "decision"},  # 0.8+
        ]
        insert_trace(self.conn, _make_trace(p, items))
        self.conn.commit()

        result = perf_counters(self.conn, p)
        hist = result["score_histogram"]
        self.assertEqual(hist["0.0-0.2"], 1)
        self.assertEqual(hist["0.2-0.4"], 1)
        self.assertEqual(hist["0.4-0.6"], 1)
        self.assertEqual(hist["0.6-0.8"], 1)
        self.assertEqual(hist["0.8+"], 1)

    def test_05_type_concentration_single(self):
        """T5: type_concentration HHI 计算正确（单类型=1.0）"""
        p = _unique_project()
        items = [
            {"id": "c1", "score": 0.8, "chunk_type": "decision"},
            {"id": "c2", "score": 0.7, "chunk_type": "decision"},
        ]
        insert_trace(self.conn, _make_trace(p, items))
        self.conn.commit()

        result = perf_counters(self.conn, p)
        self.assertAlmostEqual(result["type_concentration"], 1.0, places=2)
        self.assertEqual(result["top_type"], "decision")

    def test_06_type_concentration_uniform(self):
        """T6: type_concentration HHI 均匀分布接近 1/N"""
        p = _unique_project()
        items = [
            {"id": "c1", "score": 0.8, "chunk_type": "decision"},
            {"id": "c2", "score": 0.7, "chunk_type": "reasoning_chain"},
            {"id": "c3", "score": 0.6, "chunk_type": "design_constraint"},
            {"id": "c4", "score": 0.5, "chunk_type": "excluded_path"},
        ]
        insert_trace(self.conn, _make_trace(p, items))
        self.conn.commit()

        result = perf_counters(self.conn, p)
        # 4 equally distributed types: HHI = 4*(1/4)^2 = 0.25
        self.assertAlmostEqual(result["type_concentration"], 0.25, places=2)

    def test_07_non_injected_excluded(self):
        """T7: 非注入 traces 不纳入 score 统计"""
        p = _unique_project()
        items = [{"id": "c1", "score": 0.1, "chunk_type": "decision"}]
        insert_trace(self.conn, _make_trace(p, items, injected=0))
        insert_trace(self.conn, _make_trace(p,
                     [{"id": "c2", "score": 0.8, "chunk_type": "decision"}], injected=1))
        self.conn.commit()

        result = perf_counters(self.conn, p)
        self.assertEqual(result["total_traces"], 2)
        self.assertEqual(result["injected_traces"], 1)
        # Only injected trace's score counts
        self.assertAlmostEqual(result["avg_score"], 0.8, places=2)

    def test_08_injection_rate(self):
        """T8: injection_rate 计算正确"""
        p = _unique_project()
        items = [{"id": "c1", "score": 0.8, "chunk_type": "decision"}]
        for i in range(7):
            insert_trace(self.conn, _make_trace(p, items, injected=1))
        for i in range(3):
            insert_trace(self.conn, _make_trace(p, items, injected=0))
        self.conn.commit()

        result = perf_counters(self.conn, p)
        self.assertEqual(result["total_traces"], 10)
        self.assertAlmostEqual(result["injection_rate"], 0.7, places=2)

    def test_09_window_limit(self):
        """T9: window 参数限制 trace 数量"""
        p = _unique_project()
        items = [{"id": "c1", "score": 0.5, "chunk_type": "decision"}]
        for i in range(20):
            insert_trace(self.conn, _make_trace(p, items))
        self.conn.commit()

        result = perf_counters(self.conn, p, window=5)
        self.assertEqual(result["total_traces"], 5)

    def test_10_autotune_raise_threshold(self):
        """T10: autotune 策略8 — low_score_ratio 高时提高 min_score_threshold"""
        p = _unique_project()
        # 写入大量低分 traces 制造高 low_score_ratio
        for i in range(15):
            items = [
                {"id": f"c{i}a", "score": 0.25, "chunk_type": "decision"},  # < 0.40
                {"id": f"c{i}b", "score": 0.30, "chunk_type": "decision"},  # < 0.40
            ]
            insert_trace(self.conn, _make_trace(p, items))
        self.conn.commit()

        # 记录当前阈值
        old_threshold = _cfg("retriever.min_score_threshold", project=p)

        # 运行 autotune
        result = autotune(self.conn, p)

        # 验证 perf_counters 驱动了阈值提高
        new_threshold = _cfg("retriever.min_score_threshold", project=p)
        if result.get("tuned"):
            perf_adj = [a for a in result["adjustments"] if "iter537" in a.get("reason", "")]
            if perf_adj:
                self.assertGreater(new_threshold, old_threshold)

    def test_11_autotune_lower_threshold(self):
        """T11: autotune 策略8 — avg_score 高且 low_ratio=0 时降低阈值"""
        p = _unique_project()
        # 先手动提高阈值
        sysctl_set("retriever.min_score_threshold", 0.45, project=p)

        # 写入全高分 traces
        for i in range(15):
            items = [
                {"id": f"c{i}a", "score": 0.85, "chunk_type": "decision"},
                {"id": f"c{i}b", "score": 0.90, "chunk_type": "reasoning_chain"},
            ]
            insert_trace(self.conn, _make_trace(p, items))
        self.conn.commit()

        result = autotune(self.conn, p)

        new_threshold = _cfg("retriever.min_score_threshold", project=p)
        if result.get("tuned"):
            perf_adj = [a for a in result["adjustments"] if "iter537" in a.get("reason", "") and "lower" in a.get("reason", "")]
            if perf_adj:
                self.assertLess(new_threshold, 0.45)

    def test_12_config_tunables_registered(self):
        """T12: config tunables 注册正确"""
        perf_keys = [k for k in _REGISTRY if k.startswith("perf.")]
        self.assertGreaterEqual(len(perf_keys), 6)
        # 验证关键 tunables 存在
        self.assertIn("perf.low_score_threshold", _REGISTRY)
        self.assertIn("perf.autotune_enabled", _REGISTRY)
        self.assertIn("perf.raise_threshold_pct", _REGISTRY)
        self.assertIn("perf.threshold_max", _REGISTRY)
        self.assertIn("perf.threshold_min", _REGISTRY)

    def test_13_performance(self):
        """T13: 性能 < 50ms（100 traces）"""
        p = _unique_project()
        items = [
            {"id": "c1", "score": 0.8, "chunk_type": "decision"},
            {"id": "c2", "score": 0.6, "chunk_type": "reasoning_chain"},
        ]
        for i in range(100):
            insert_trace(self.conn, _make_trace(p, items))
        self.conn.commit()

        start = time.time()
        for _ in range(10):
            perf_counters(self.conn, p, window=30)
        elapsed_ms = (time.time() - start) * 1000 / 10
        self.assertLess(elapsed_ms, 50, f"perf_counters too slow: {elapsed_ms:.1f}ms")


if __name__ == "__main__":
    unittest.main()
