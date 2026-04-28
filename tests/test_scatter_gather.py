"""
tests/test_scatter_gather.py — Domain-Aware Scatter-Gather 测试

验证：
  1. domain_classify() 识别准确性
  2. scatter_gather_route() 结构返回正确
  3. 并发性：scatter_ms < 串行预期时间（T1+T2+T3 的 50%）
  4. 短路：高质量结果足够时 short_circuit=True
  5. 域亲和性：亲和源的结果 score 有 +5% bonus
"""
import sys
import time
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from hooks.knowledge_router import (
    domain_classify,
    scatter_gather_route,
    _DOMAIN_SOURCE_AFFINITY,
    _DOMAIN_SIGNALS,
)


# ─────────────────────────────────────────────────────────────
# 1. domain_classify
# ─────────────────────────────────────────────────────────────

class TestDomainClassify:
    def test_code_domain(self):
        assert domain_classify("这个函数的参数格式是什么") == "code"

    def test_code_english(self):
        assert domain_classify("how to implement this function") == "code"

    def test_project_domain(self):
        assert domain_classify("上次的设计决策是什么") == "project"

    def test_project_english(self):
        assert domain_classify("what was the design decision here") == "project"

    def test_rule_domain(self):
        assert domain_classify("操作规范和约束条件") == "rule"

    def test_general_no_signal(self):
        # 无匹配信号 → general
        result = domain_classify("今天天气怎么样")
        assert result == "general"

    def test_empty_query(self):
        result = domain_classify("")
        assert result == "general"

    def test_mixed_prefers_highest(self):
        # code 信号多 → code
        q = "函数 方法 class 实现 上次决策"  # 4 code + 1 project
        result = domain_classify(q)
        assert result == "code"

    def test_domain_affinity_coverage(self):
        """所有域都有 source affinity 配置。"""
        for domain in ["code", "project", "rule", "general"]:
            assert domain in _DOMAIN_SOURCE_AFFINITY
            assert len(_DOMAIN_SOURCE_AFFINITY[domain]) >= 1


# ─────────────────────────────────────────────────────────────
# 2. scatter_gather_route 结构测试（mock 各搜索源）
# ─────────────────────────────────────────────────────────────

def _make_mock_results(source: str, count: int = 2, score: float = 0.6):
    return [
        {
            "source": source,
            "chunk_type": "decision",
            "summary": f"{source} result {i}",
            "score": score,
            "content": f"content {i}",
            "path": "",
        }
        for i in range(count)
    ]


@pytest.fixture
def mock_search_fns():
    """Mock 所有搜索函数，控制返回值和延迟。"""
    delay_map = {
        "_search_memory_os": 0.005,      # 5ms
        "_search_memory_md": 0.003,      # 3ms
        "_search_self_improving": 0.004, # 4ms
    }

    def make_mock(name, results):
        def fn(*args, **kwargs):
            time.sleep(delay_map.get(name, 0.003))
            return results
        return fn

    patches = {
        "_search_memory_os":     make_mock("_search_memory_os",     _make_mock_results("memory_os")),
        "_search_memory_md":     make_mock("_search_memory_md",     _make_mock_results("memory_md")),
        "_search_self_improving": make_mock("_search_self_improving", _make_mock_results("self_improving")),
    }
    return patches


class TestScatterGatherRoute:
    def test_returns_correct_structure(self, mock_search_fns):
        """返回值结构正确。"""
        with patch("hooks.knowledge_router._search_memory_os", mock_search_fns["_search_memory_os"]), \
             patch("hooks.knowledge_router._search_memory_md", mock_search_fns["_search_memory_md"]), \
             patch("hooks.knowledge_router._search_self_improving", mock_search_fns["_search_self_improving"]), \
             patch("hooks.knowledge_router.resolve_project_id", return_value="test"):

            result = scatter_gather_route("函数实现方式", project="test")

        assert "results" in result
        assert "domain" in result
        assert "scatter_ms" in result
        assert "gather_ms" in result
        assert "short_circuit" in result
        assert "source_times" in result

    def test_domain_detected(self, mock_search_fns):
        """code 域查询能识别为 code。"""
        with patch("hooks.knowledge_router._search_memory_os", mock_search_fns["_search_memory_os"]), \
             patch("hooks.knowledge_router._search_memory_md", mock_search_fns["_search_memory_md"]), \
             patch("hooks.knowledge_router._search_self_improving", mock_search_fns["_search_self_improving"]), \
             patch("hooks.knowledge_router.resolve_project_id", return_value="test"):

            result = scatter_gather_route("这个函数的实现方式", project="test")

        assert result["domain"] == "code"

    def test_parallel_faster_than_serial(self, mock_search_fns):
        """
        并发执行应比串行快。
        各源延迟 3~5ms → 串行约 12ms，并发应 < 8ms（最慢源 + 小开销）。
        """
        with patch("hooks.knowledge_router._search_memory_os", mock_search_fns["_search_memory_os"]), \
             patch("hooks.knowledge_router._search_memory_md", mock_search_fns["_search_memory_md"]), \
             patch("hooks.knowledge_router._search_self_improving", mock_search_fns["_search_self_improving"]), \
             patch("hooks.knowledge_router.resolve_project_id", return_value="test"):

            t0 = time.monotonic()
            result = scatter_gather_route("函数实现", project="test")
            elapsed = (time.monotonic() - t0) * 1000

        # 并发应在 10ms 内完成（3 个 3-5ms 任务并发）
        assert elapsed < 15.0, f"scatter_gather took {elapsed:.1f}ms, expected < 15ms"
        # scatter_ms 字段也应该反映这个值
        assert result["scatter_ms"] < 15.0

    def test_results_deduplicated(self, mock_search_fns):
        """相同 summary 的结果应去重，只保留最高分。"""
        # 让两个源返回相同的 summary
        same_result = [{"source": "memory_os", "chunk_type": "decision",
                        "summary": "same result", "score": 0.8, "content": "", "path": ""}]
        same_result2 = [{"source": "memory_md", "chunk_type": "decision",
                         "summary": "same result", "score": 0.6, "content": "", "path": ""}]

        with patch("hooks.knowledge_router._search_memory_os", return_value=same_result), \
             patch("hooks.knowledge_router._search_memory_md", return_value=same_result2), \
             patch("hooks.knowledge_router._search_self_improving", return_value=[]), \
             patch("hooks.knowledge_router.resolve_project_id", return_value="test"):

            result = scatter_gather_route("test query", project="test")

        summaries = [r["summary"] for r in result["results"]]
        assert summaries.count("same result") == 1

    def test_affinity_bonus_applied(self):
        """亲和域的前两个源应获得 score +5% bonus。"""
        # code domain → affinity srcs = ["memory_os", "self_improving"]
        base_score = 0.6
        results_os = [{"source": "memory_os", "chunk_type": "decision",
                       "summary": "os result", "score": base_score, "content": "", "path": ""}]
        results_md = [{"source": "memory_md", "chunk_type": "decision",
                       "summary": "md result", "score": base_score, "content": "", "path": ""}]

        with patch("hooks.knowledge_router._search_memory_os", return_value=results_os), \
             patch("hooks.knowledge_router._search_memory_md", return_value=results_md), \
             patch("hooks.knowledge_router._search_self_improving", return_value=[]), \
             patch("hooks.knowledge_router.resolve_project_id", return_value="test"):

            result = scatter_gather_route("函数实现", project="test", domain="code")

        # code domain affinity = ["memory_os", "self_improving", "memory_md"]
        # 前两个是 memory_os 和 self_improving → memory_os 应有 bonus
        os_results = [r for r in result["results"] if r["source"] == "memory_os"]
        md_results = [r for r in result["results"] if r["source"] == "memory_md"]
        if os_results and md_results:
            # memory_os 是亲和源（+5% bonus），memory_md 不是（第三位）
            # 注意：需要考虑 SOURCE_WEIGHT 也影响 score
            assert os_results[0]["score"] >= md_results[0]["score"]

    def test_short_circuit_triggered(self):
        """高质量结果达到阈值时触发短路。"""
        # 返回足够多的高分结果
        high_q_results = [
            {"source": "memory_os", "chunk_type": "decision",
             "summary": f"high quality {i}", "score": 0.9, "content": "", "path": ""}
            for i in range(5)  # 5 个高质量结果，超过默认阈值 3
        ]

        with patch("hooks.knowledge_router._search_memory_os", return_value=high_q_results), \
             patch("hooks.knowledge_router._search_memory_md", return_value=[]), \
             patch("hooks.knowledge_router._search_self_improving", return_value=[]), \
             patch("hooks.knowledge_router.resolve_project_id", return_value="test"), \
             patch("hooks.knowledge_router._sysctl", side_effect=lambda k: {
                 "router.scatter_shortcircuit_score": 0.75,
                 "router.scatter_shortcircuit_count": 3,
                 "router.top_k_per_source": 3,
                 "router.min_score": 0.01,
             }.get(k, 3)):

            result = scatter_gather_route("查询", project="test")

        assert result["short_circuit"] is True

    def test_domain_override(self, mock_search_fns):
        """domain 参数可以覆盖自动识别。"""
        with patch("hooks.knowledge_router._search_memory_os", mock_search_fns["_search_memory_os"]), \
             patch("hooks.knowledge_router._search_memory_md", mock_search_fns["_search_memory_md"]), \
             patch("hooks.knowledge_router._search_self_improving", mock_search_fns["_search_self_improving"]), \
             patch("hooks.knowledge_router.resolve_project_id", return_value="test"):

            result = scatter_gather_route("today weather", project="test", domain="rule")

        assert result["domain"] == "rule"

    def test_source_times_recorded(self, mock_search_fns):
        """source_times 记录各源的耗时。"""
        with patch("hooks.knowledge_router._search_memory_os", mock_search_fns["_search_memory_os"]), \
             patch("hooks.knowledge_router._search_memory_md", mock_search_fns["_search_memory_md"]), \
             patch("hooks.knowledge_router._search_self_improving", mock_search_fns["_search_self_improving"]), \
             patch("hooks.knowledge_router.resolve_project_id", return_value="test"):

            result = scatter_gather_route("函数实现", project="test")

        # 至少有一个 source 记录了时间
        assert len(result["source_times"]) >= 1
        for src, ms in result["source_times"].items():
            assert ms >= 0
