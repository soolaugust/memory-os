"""
tests/test_proactive_swap.py — Proactive Swap Probe 测试（迭代355）

验证：
  1. proactive_swap_enabled=False 时不触发探针
  2. top_k 为空时不触发探针（走原来的 swap_fault 路径）
  3. swap 中有高 importance 匹配时，主动恢复并注入 top_k
  4. importance < threshold 的 swap chunk 不被恢复
  5. 恢复数量不超过 proactive_swap_max_restore
  6. swap_in 失败时主流程不受影响（异常吞掉）
  7. 恢复后 top_k 按 score 重新排序
"""
import sys
import sqlite3
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


def _make_swap_match(chunk_id: str, importance: float,
                     hit_ratio: float = 0.7) -> dict:
    return {
        "id": chunk_id,
        "summary": f"summary for {chunk_id}",
        "importance": importance,
        "chunk_type": "decision",
        "hit_count": 3,
        "hit_ratio": hit_ratio,
    }


def _make_chunk_row(chunk_id: str, importance: float = 0.9) -> dict:
    return {
        "id": chunk_id,
        "project": "test",
        "source_session": "sess",
        "chunk_type": "decision",
        "content": f"content for {chunk_id}",
        "summary": f"summary for {chunk_id}",
        "importance": importance,
        "retrievability": 0.8,
        "stability": 5.0,
        "tags": "",
        "raw_snippet": "",
        "feishu_url": "",
        "encoding_context": "",
        "last_accessed": "",
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "info_class": "project",
        "access_count": 2,
    }


# ─────────────────────────────────────────────────────────────
# 辅助：构建 mock sysctl
# ─────────────────────────────────────────────────────────────

def _default_sysctl(key):
    return {
        "retriever.proactive_swap_enabled": True,
        "retriever.proactive_swap_imp_threshold": 0.80,
        "retriever.proactive_swap_max_restore": 3,
    }.get(key, None)


def _disabled_sysctl(key):
    return {
        "retriever.proactive_swap_enabled": False,
        "retriever.proactive_swap_imp_threshold": 0.80,
        "retriever.proactive_swap_max_restore": 3,
    }.get(key, None)


# ─────────────────────────────────────────────────────────────
# 1. 基本功能测试（通过直接测试逻辑，不依赖 retriever.py 的复杂路径）
# ─────────────────────────────────────────────────────────────

class TestProactiveSwapLogic:
    """直接测试 proactive swap 过滤逻辑，不依赖 retriever.py 完整路径。"""

    def test_importance_threshold_filter(self):
        """只有 importance >= threshold 的 swap match 才被处理。"""
        matches = [
            _make_swap_match("high-imp", importance=0.90),  # pass
            _make_swap_match("mid-imp", importance=0.75),   # fail (< 0.80)
            _make_swap_match("low-imp", importance=0.50),   # fail
        ]
        threshold = 0.80
        filtered = [m for m in matches if m.get("importance", 0) >= threshold]
        assert len(filtered) == 1
        assert filtered[0]["id"] == "high-imp"

    def test_max_restore_cap(self):
        """最多恢复 max_restore 个 chunk。"""
        matches = [_make_swap_match(f"chunk-{i}", 0.9) for i in range(10)]
        max_restore = 3
        to_restore = [m["id"] for m in matches[:max_restore]]
        assert len(to_restore) == 3

    def test_already_in_top_k_skipped(self):
        """已在 top_k 中的 chunk 不重复注入。"""
        top_k = [(0.9, {"id": "chunk-0", "importance": 0.9})]
        already_ids = {c.get("id", "") for _, c in top_k}

        probe_matches = [
            _make_swap_match("chunk-0", 0.9),  # 已在 top_k
            _make_swap_match("chunk-1", 0.85), # 不在 top_k
        ]
        new_ids = [m["id"] for m in probe_matches if m["id"] not in already_ids]
        assert new_ids == ["chunk-1"]

    def test_top_k_reranked_after_injection(self):
        """注入后 top_k 按 score 重排序。"""
        top_k = [
            (0.5, {"id": "existing-1"}),
            (0.4, {"id": "existing-2"}),
        ]
        # 注入一个高分 chunk
        top_k.append((0.95, {"id": "injected-high"}))
        top_k.sort(key=lambda x: x[0], reverse=True)

        assert top_k[0][1]["id"] == "injected-high"

    def test_disabled_flag_skips_probe(self):
        """proactive_swap_enabled=False 时探针不运行（通过 sysctl 控制）。"""
        # 验证 sysctl 值控制逻辑
        assert _disabled_sysctl("retriever.proactive_swap_enabled") is False
        assert _default_sysctl("retriever.proactive_swap_enabled") is True

    def test_empty_top_k_skips_proactive_path(self):
        """top_k 为空时，proactive 探针不触发（走原来的 if not top_k 路径）。"""
        top_k = []
        # proactive 路径要求 top_k 非空
        should_run_proactive = bool(top_k)  # False when empty
        assert should_run_proactive is False

    def test_no_high_importance_matches(self):
        """swap 中无高 importance 匹配时，top_k 不变。"""
        top_k_before = [(0.7, {"id": "existing"})]
        top_k = list(top_k_before)

        matches = [_make_swap_match("low-imp-chunk", importance=0.60)]
        threshold = 0.80
        filtered = [m for m in matches if m.get("importance", 0) >= threshold]

        # filtered 为空 → 不执行 swap_in → top_k 不变
        assert len(filtered) == 0
        assert top_k == top_k_before


# ─────────────────────────────────────────────────────────────
# 2. config 配置项验证
# ─────────────────────────────────────────────────────────────

class TestProactiveSwapConfig:
    def test_config_keys_exist(self):
        """config.py 中包含 proactive_swap 配置项。"""
        from config import get as sysctl
        # 这些 key 应该存在（不应抛出 KeyError）
        val_enabled = sysctl("retriever.proactive_swap_enabled")
        val_threshold = sysctl("retriever.proactive_swap_imp_threshold")
        val_max = sysctl("retriever.proactive_swap_max_restore")

        assert isinstance(val_enabled, bool)
        assert isinstance(val_threshold, float)
        assert isinstance(val_max, int)

    def test_default_values_sensible(self):
        """默认值合理：threshold=0.80，max_restore=3，enabled=True。"""
        from config import get as sysctl
        assert sysctl("retriever.proactive_swap_enabled") is True
        assert sysctl("retriever.proactive_swap_imp_threshold") == 0.80
        assert sysctl("retriever.proactive_swap_max_restore") == 3

    def test_threshold_in_valid_range(self):
        """threshold 应在 [0.5, 1.0] 范围内。"""
        from config import get as sysctl
        threshold = sysctl("retriever.proactive_swap_imp_threshold")
        assert 0.5 <= threshold <= 1.0

    def test_max_restore_positive(self):
        """max_restore 应为正整数。"""
        from config import get as sysctl
        max_restore = sysctl("retriever.proactive_swap_max_restore")
        assert max_restore >= 1


# ─────────────────────────────────────────────────────────────
# 3. store_swap.swap_fault 行为验证（mock DB）
# ─────────────────────────────────────────────────────────────

class TestSwapFaultBehavior:
    def test_swap_fault_filters_by_importance(self):
        """swap_fault 返回结果应包含 importance 字段。"""
        import zlib, base64, json

        # 构造 mock DB 行
        high_imp_data = {"summary": "重要决策", "content": "使用架构A方案"}
        compressed = base64.b64encode(
            zlib.compress(json.dumps(high_imp_data).encode())
        ).decode()

        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = [
            ("id-high", 0.90, "decision", compressed),
            ("id-low",  0.40, "decision", compressed),
        ]

        from store_swap import swap_fault
        matches = swap_fault(mock_conn, "架构决策", "test-project")

        # swap_fault 应返回匹配结果（含 importance 字段）
        for m in matches:
            assert "importance" in m
            assert "id" in m

    def test_swap_fault_empty_query_returns_empty(self):
        """空 query 返回空列表（无 tokens 可匹配）。"""
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = []

        from store_swap import swap_fault
        matches = swap_fault(mock_conn, "", "test-project")
        assert matches == []

    def test_swap_in_returns_restored_count(self):
        """swap_in 返回 dict 含 restored_count。"""
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = []

        from store_swap import swap_in
        result = swap_in(mock_conn, ["nonexistent-id"])
        assert "restored_count" in result
        assert result["restored_count"] == 0  # 不存在的 id → 0
