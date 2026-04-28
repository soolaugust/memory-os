"""
tests/test_prefetch.py — Prefetch Engine 测试

验证：
  1. tool → domain 映射
  2. query 提取（_extract_prefetch_query）
  3. trigger_prefetch 非阻塞性（< 5ms 返回）
  4. 去重（相同 session+tool+query 不重复提交）
  5. 自适应控制器（hit rate 影响 scale）
  6. 预取统计
  7. disabled 状态下不触发
"""
import sys
import time
import threading
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from hooks.prefetch_engine import (
    _extract_prefetch_query,
    _TOOL_PATTERN_TABLE,
    _dedup,
    _PrefetchDeduplicator,
    _AdaptivePrefetchController,
    _stats,
    trigger_prefetch,
    record_query_outcome,
    get_prefetch_stats,
)


# ─────────────────────────────────────────────────────────────
# 1. Tool → Domain 映射完整性
# ─────────────────────────────────────────────────────────────

class TestToolPatternTable:
    def test_all_write_tools_covered(self):
        """写操作工具应有覆盖。"""
        for tool in ["Edit", "Write", "MultiEdit"]:
            assert tool in _TOOL_PATTERN_TABLE

    def test_read_tools_covered(self):
        for tool in ["Read", "Glob", "Grep"]:
            assert tool in _TOOL_PATTERN_TABLE

    def test_default_fallback_exists(self):
        assert "__default__" in _TOOL_PATTERN_TABLE

    def test_domains_are_valid(self):
        valid = {"code", "project", "rule", "general"}
        for tool, (domains, boost) in _TOOL_PATTERN_TABLE.items():
            for d in domains:
                assert d in valid, f"Tool {tool} has invalid domain {d}"

    def test_boost_positive(self):
        for tool, (domains, boost) in _TOOL_PATTERN_TABLE.items():
            assert boost > 0, f"Tool {tool} has non-positive boost {boost}"

    def test_write_tools_have_higher_boost(self):
        """写操作工具预取强度应比只读工具高。"""
        edit_boost = _TOOL_PATTERN_TABLE["Edit"][1]
        read_boost = _TOOL_PATTERN_TABLE["Read"][1]
        assert edit_boost > read_boost


# ─────────────────────────────────────────────────────────────
# 2. Query 提取
# ─────────────────────────────────────────────────────────────

class TestExtractPrefetchQuery:
    def test_edit_extracts_filename(self):
        q = _extract_prefetch_query("Edit", {"file_path": "/home/user/store_vfs.py"})
        assert "store_vfs.py" in q

    def test_bash_extracts_command(self):
        q = _extract_prefetch_query("Bash", {"command": "git push origin main"})
        assert "git" in q or "push" in q

    def test_agent_extracts_description(self):
        q = _extract_prefetch_query("Agent", {
            "description": "代码审查",
            "prompt": "review memory-os architecture"
        })
        assert "代码审查" in q or "review" in q

    def test_empty_input_uses_tool_name(self):
        q = _extract_prefetch_query("WebFetch", {})
        assert q == "WebFetch"

    def test_unknown_tool_uses_tool_name(self):
        q = _extract_prefetch_query("UnknownTool", {"some_field": "value"})
        assert q == "UnknownTool"

    def test_max_length(self):
        long_command = "a" * 1000
        q = _extract_prefetch_query("Bash", {"command": long_command})
        assert len(q) <= 200

    def test_file_path_basename_extracted(self):
        """文件路径应提取 basename，不含完整路径。"""
        q = _extract_prefetch_query("Read", {
            "file_path": "/very/long/path/to/knowledge_router.py"
        })
        assert "knowledge_router.py" in q
        # 不应包含完整路径
        assert "/very/long/path/to/" not in q


# ─────────────────────────────────────────────────────────────
# 3. trigger_prefetch 非阻塞性
# ─────────────────────────────────────────────────────────────

class TestTriggerPrefetchNonBlocking:
    def test_returns_immediately(self):
        """trigger_prefetch 应在 5ms 内返回。"""
        executed = threading.Event()

        def slow_prefetch(*args, **kwargs):
            time.sleep(0.1)  # 模拟 100ms 预取
            executed.set()

        with patch("hooks.prefetch_engine._do_prefetch", side_effect=slow_prefetch), \
             patch("hooks.prefetch_engine._sysctl", side_effect=lambda k: {
                 "prefetch.enabled": True,
                 "prefetch.max_chunks": 10,
             }.get(k, True)):

            t0 = time.monotonic()
            result = trigger_prefetch(
                "sess-nb-1", "proj-1", "Edit",
                {"file_path": "test.py"}
            )
            elapsed = (time.monotonic() - t0) * 1000

        assert elapsed < 10.0, f"trigger_prefetch blocked for {elapsed:.1f}ms"
        assert result is True  # 已提交

    def test_disabled_returns_false(self):
        """prefetch.enabled=False 时立即返回 False。"""
        with patch("hooks.prefetch_engine._sysctl", side_effect=lambda k: {
            "prefetch.enabled": False,
            "prefetch.max_chunks": 10,
        }.get(k, False)):
            result = trigger_prefetch("sess-dis-1", "proj-1", "Edit", {})
        assert result is False


# ─────────────────────────────────────────────────────────────
# 4. 去重
# ─────────────────────────────────────────────────────────────

class TestPrefetchDeduplication:
    def test_dedup_blocks_second_call(self):
        """相同 session + query 不触发第二次预取。"""
        dedup = _PrefetchDeduplicator(max_size=100)
        assert dedup.is_dup("sess-1", "query-A") is False  # 第一次：不重复
        assert dedup.is_dup("sess-1", "query-A") is True   # 第二次：重复

    def test_different_sessions_not_dedup(self):
        """不同 session 的相同 query 不去重。"""
        dedup = _PrefetchDeduplicator(max_size=100)
        assert dedup.is_dup("sess-1", "same-query") is False
        assert dedup.is_dup("sess-2", "same-query") is False  # 不同 session

    def test_different_queries_not_dedup(self):
        """相同 session 的不同 query 不去重。"""
        dedup = _PrefetchDeduplicator(max_size=100)
        assert dedup.is_dup("sess-1", "query-A") is False
        assert dedup.is_dup("sess-1", "query-B") is False

    def test_max_size_eviction(self):
        """超过 max_size 时自动驱逐一半。"""
        dedup = _PrefetchDeduplicator(max_size=4)
        for i in range(4):
            dedup.is_dup("sess-1", f"query-{i}")
        assert len(dedup._seen) == 4
        # 再加一个触发驱逐
        dedup.is_dup("sess-1", "query-new")
        assert len(dedup._seen) <= 4  # 应保持在 max_size 以内


# ─────────────────────────────────────────────────────────────
# 5. 自适应控制器
# ─────────────────────────────────────────────────────────────

class TestAdaptivePrefetchController:
    def test_initial_scale_is_one(self):
        ctrl = _AdaptivePrefetchController()
        assert ctrl.scale() == 1.0

    def test_low_hit_rate_decreases_scale(self):
        """命中率 < 20% 时 scale 降低。"""
        ctrl = _AdaptivePrefetchController()
        ctrl._eval_interval = 5  # 缩短评估间隔便于测试

        # 模拟低命中率
        with patch("hooks.prefetch_engine._stats") as mock_stats:
            mock_stats.hit_rate.return_value = 10.0  # 10% 命中率
            for _ in range(ctrl._eval_interval):
                ctrl.update()

        assert ctrl.scale() < 1.0

    def test_high_hit_rate_increases_scale(self):
        """命中率 > 60% 时 scale 提升。"""
        ctrl = _AdaptivePrefetchController()
        ctrl._eval_interval = 5
        ctrl._scale = 1.0

        with patch("hooks.prefetch_engine._stats") as mock_stats:
            mock_stats.hit_rate.return_value = 80.0  # 80% 命中率
            for _ in range(ctrl._eval_interval):
                ctrl.update()

        assert ctrl.scale() > 1.0

    def test_normal_hit_rate_resets_scale(self):
        """命中率在正常范围时 scale 重置为 1.0。"""
        ctrl = _AdaptivePrefetchController()
        ctrl._eval_interval = 5
        ctrl._scale = 0.5  # 从低 scale 开始

        with patch("hooks.prefetch_engine._stats") as mock_stats:
            mock_stats.hit_rate.return_value = 40.0  # 40% 正常命中率
            for _ in range(ctrl._eval_interval):
                ctrl.update()

        assert ctrl.scale() == 1.0

    def test_effective_top_k(self):
        ctrl = _AdaptivePrefetchController()
        ctrl._scale = 1.0
        assert ctrl.effective_top_k(10) == 10
        ctrl._scale = 0.5
        assert ctrl.effective_top_k(10) == 5
        ctrl._scale = 2.0
        assert ctrl.effective_top_k(10) == 20

    def test_effective_top_k_min_one(self):
        """effective_top_k 最小值为 1。"""
        ctrl = _AdaptivePrefetchController()
        ctrl._scale = 0.01
        assert ctrl.effective_top_k(10) >= 1


# ─────────────────────────────────────────────────────────────
# 6. 统计
# ─────────────────────────────────────────────────────────────

class TestPrefetchStats:
    def test_stats_structure(self):
        s = get_prefetch_stats()
        expected = [
            "total_prefetches", "hits", "misses", "errors",
            "total_chunks", "avg_ms", "hit_rate_pct", "adaptive_scale",
        ]
        for key in expected:
            assert key in s, f"Missing key: {key}"

    def test_record_outcome(self):
        """record_query_outcome 更新命中/未命中统计。"""
        from hooks.prefetch_engine import _PrefetchStats
        local_stats = _PrefetchStats()
        with patch("hooks.prefetch_engine._stats", local_stats):
            record_query_outcome("sess-1", "query-A", was_cache_hit=True)
            record_query_outcome("sess-1", "query-B", was_cache_hit=False)
        assert local_stats.hits == 1
        assert local_stats.misses == 1

    def test_hit_rate_calculation(self):
        from hooks.prefetch_engine import _PrefetchStats
        local_stats = _PrefetchStats()
        local_stats.hits = 3
        local_stats.misses = 1
        assert local_stats.hit_rate() == 75.0

    def test_hit_rate_no_data(self):
        from hooks.prefetch_engine import _PrefetchStats
        local_stats = _PrefetchStats()
        assert local_stats.hit_rate() == 0.0


# ─────────────────────────────────────────────────────────────
# 7. do_prefetch 集成（mock store.db）
# ─────────────────────────────────────────────────────────────

class TestDoPrefetchIntegration:
    def test_do_prefetch_no_crash(self):
        """
        _do_prefetch 使用 lazy import，测试触发后无异常。
        通过 trigger_prefetch + mock 验证整体流程不崩溃。
        """
        triggered = threading.Event()
        completed = threading.Event()

        def mock_do_prefetch(*args, **kwargs):
            triggered.set()
            completed.set()

        with patch("hooks.prefetch_engine._do_prefetch", side_effect=mock_do_prefetch), \
             patch("hooks.prefetch_engine._sysctl", side_effect=lambda k: {
                 "prefetch.enabled": True,
                 "prefetch.max_chunks": 5,
             }.get(k, True)):

            result = trigger_prefetch(
                "sess-integration", "proj-integration",
                "Edit", {"file_path": "test.py"}
            )
            assert result is True

        # 等待后台线程完成（最多 1 秒）
        completed.wait(timeout=1.0)
        assert triggered.is_set(), "_do_prefetch 应被后台线程调用"
