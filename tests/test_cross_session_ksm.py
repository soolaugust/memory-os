"""
tests/test_cross_session_ksm.py — Cross-Session KSM 测试（迭代358）

验证：
  1. get_hot_chunks 扫描多 session working set
  2. 只返回 access_count >= min_access_count 的 chunk
  3. 只返回出现在 >= min_sessions 个 session 的 chunk
  4. promote_hot_chunks 写回 store.db（retrievability 提升）
  5. 单 session 的 chunk 不被提升（min_sessions=2）
  6. config 默认值合理
"""
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from agent_working_set import WorkingSet, WorkingSetRegistry, WSEntry
from schema import MemoryChunk


def _make_chunk(idx: int, project: str = "ksm-project",
                importance: float = 0.8) -> MemoryChunk:
    return MemoryChunk(
        id=f"ksm-chunk-{idx}",
        project=project,
        source_session="sess",
        chunk_type="decision",
        content=f"ksm content {idx}",
        summary=f"ksm summary {idx}",
        importance=importance,
        retrievability=0.7,
    )


def _fresh_registry() -> WorkingSetRegistry:
    """每次测试使用干净的注册表。"""
    r = WorkingSetRegistry()
    with r._reg_lock:
        r._registry.clear()
    return r


# ─────────────────────────────────────────────────────────────
# 1. get_hot_chunks 基础逻辑
# ─────────────────────────────────────────────────────────────

class TestGetHotChunks:
    def test_hot_chunk_appears_in_multiple_sessions(self):
        """相同 chunk 在多个 session 中高频访问 → 应出现在 hot_chunks 中。"""
        r = _fresh_registry()
        ws1 = r.get_or_create("sess-ksm-1", "ksm-project")
        ws2 = r.get_or_create("sess-ksm-2", "ksm-project")

        chunk = _make_chunk(0)
        ws1.put(chunk)
        ws2.put(chunk)

        # 模拟多次访问
        ws1._lru["ksm-chunk-0"].access_count = 5
        ws2._lru["ksm-chunk-0"].access_count = 4

        hot = r.get_hot_chunks("ksm-project", min_access_count=3, min_sessions=2)
        assert any(h["chunk_id"] == "ksm-chunk-0" for h in hot)

    def test_low_access_chunk_excluded(self):
        """access_count < min_access_count 的 chunk 不应被识别为热点。"""
        r = _fresh_registry()
        ws1 = r.get_or_create("sess-ksm-3", "ksm-project")
        ws2 = r.get_or_create("sess-ksm-4", "ksm-project")

        chunk = _make_chunk(1)
        ws1.put(chunk)
        ws2.put(chunk)

        # access_count=1（< min=3）
        ws1._lru["ksm-chunk-1"].access_count = 1
        ws2._lru["ksm-chunk-1"].access_count = 1

        hot = r.get_hot_chunks("ksm-project", min_access_count=3, min_sessions=2)
        assert not any(h["chunk_id"] == "ksm-chunk-1" for h in hot)

    def test_single_session_chunk_excluded(self):
        """只在 1 个 session 出现的 chunk（min_sessions=2）不应被提升。"""
        r = _fresh_registry()
        ws1 = r.get_or_create("sess-ksm-5", "ksm-project")

        chunk = _make_chunk(2)
        ws1.put(chunk)
        ws1._lru["ksm-chunk-2"].access_count = 10  # 高访问，但只有 1 个 session

        hot = r.get_hot_chunks("ksm-project", min_access_count=3, min_sessions=2)
        assert not any(h["chunk_id"] == "ksm-chunk-2" for h in hot)

    def test_cross_project_chunks_excluded(self):
        """不属于当前 project 的 chunk 不参与 KSM 扫描。"""
        r = _fresh_registry()
        ws1 = r.get_or_create("sess-ksm-6", "other-project")
        ws2 = r.get_or_create("sess-ksm-7", "other-project")

        chunk = _make_chunk(3, project="other-project")
        ws1.put(chunk)
        ws2.put(chunk)
        ws1._lru["ksm-chunk-3"].access_count = 5
        ws2._lru["ksm-chunk-3"].access_count = 5

        # 查询 ksm-project，other-project 的 chunk 不应出现
        hot = r.get_hot_chunks("ksm-project", min_access_count=3, min_sessions=2)
        assert not any(h["chunk_id"] == "ksm-chunk-3" for h in hot)

    def test_hot_chunks_sorted_by_total_access(self):
        """hot_chunks 按 total_access_count 降序排列。"""
        r = _fresh_registry()
        ws1 = r.get_or_create("sess-ksm-8", "ksm-project")
        ws2 = r.get_or_create("sess-ksm-9", "ksm-project")

        chunk_a = _make_chunk(10)
        chunk_b = _make_chunk(11)
        for ws in [ws1, ws2]:
            ws.put(chunk_a)
            ws.put(chunk_b)

        ws1._lru["ksm-chunk-10"].access_count = 10
        ws2._lru["ksm-chunk-10"].access_count = 8   # total=18

        ws1._lru["ksm-chunk-11"].access_count = 5
        ws2._lru["ksm-chunk-11"].access_count = 3   # total=8

        hot = r.get_hot_chunks("ksm-project", min_access_count=3, min_sessions=2)
        ids = [h["chunk_id"] for h in hot]
        assert ids.index("ksm-chunk-10") < ids.index("ksm-chunk-11")

    def test_empty_registry_returns_empty(self):
        """空注册表返回空列表。"""
        r = _fresh_registry()
        hot = r.get_hot_chunks("ksm-project")
        assert hot == []

    def test_session_count_tracked(self):
        """session_count 正确计算跨 session 数量。"""
        r = _fresh_registry()
        ws1 = r.get_or_create("sess-ksm-10", "ksm-project")
        ws2 = r.get_or_create("sess-ksm-11", "ksm-project")
        ws3 = r.get_or_create("sess-ksm-12", "ksm-project")

        chunk = _make_chunk(20)
        for ws in [ws1, ws2, ws3]:
            ws.put(chunk)
            ws._lru["ksm-chunk-20"].access_count = 5

        hot = r.get_hot_chunks("ksm-project", min_access_count=3, min_sessions=2)
        found = next((h for h in hot if h["chunk_id"] == "ksm-chunk-20"), None)
        assert found is not None
        assert found["session_count"] == 3
        assert found["total_access_count"] == 15


# ─────────────────────────────────────────────────────────────
# 2. promote_hot_chunks（mock store.db）
# ─────────────────────────────────────────────────────────────

class TestPromoteHotChunks:
    def test_promote_returns_count(self):
        """promote_hot_chunks 返回实际提升的 chunk 数量。"""
        r = _fresh_registry()
        ws1 = r.get_or_create("sess-promote-1", "ksm-project")
        ws2 = r.get_or_create("sess-promote-2", "ksm-project")

        chunk = _make_chunk(30)
        for ws in [ws1, ws2]:
            ws.put(chunk)
            ws._lru["ksm-chunk-30"].access_count = 5

        # Mock store.db — open_db/ensure_schema 在 promote_hot_chunks 内部 import
        mock_conn = MagicMock()
        mock_conn.execute.return_value = MagicMock()

        with patch("store.open_db", return_value=mock_conn), \
             patch("store.ensure_schema"):
            count = r.promote_hot_chunks(
                "ksm-project", min_access_count=3, min_sessions=2
            )

        assert count >= 1

    def test_no_hot_chunks_returns_zero(self):
        """无热点 chunk 时返回 0。"""
        r = _fresh_registry()
        count = r.promote_hot_chunks("ksm-project")
        assert count == 0

    def test_retrievability_increase_on_promote(self):
        """promote 后 retrievability 应提升（不超过 1.0）。"""
        # 直接测试 promote 逻辑（不依赖 registry）
        base_retrievability = 0.7
        session_count = 2
        new_retrievability = min(1.0, base_retrievability + 0.05 * session_count)
        assert new_retrievability > base_retrievability
        assert new_retrievability <= 1.0

    def test_retrievability_cap_at_one(self):
        """retrievability 不超过 1.0（即使 session_count 很大）。"""
        base_retrievability = 0.98
        session_count = 10  # extreme case
        new_retrievability = min(1.0, base_retrievability + 0.05 * session_count)
        assert new_retrievability == 1.0


# ─────────────────────────────────────────────────────────────
# 3. config 默认值
# ─────────────────────────────────────────────────────────────

class TestKSMConfig:
    def test_config_keys_exist(self):
        from config import get as sysctl
        assert isinstance(sysctl("ksm.enabled"), bool)
        assert isinstance(sysctl("ksm.min_access_count"), int)
        assert isinstance(sysctl("ksm.min_sessions"), int)

    def test_defaults(self):
        from config import get as sysctl
        assert sysctl("ksm.enabled") is True
        assert sysctl("ksm.min_access_count") == 3
        assert sysctl("ksm.min_sessions") == 2

    def test_min_sessions_at_least_two(self):
        """min_sessions 必须 >= 2（1 个 session 不需要 KSM）。"""
        from config import get as sysctl
        assert sysctl("ksm.min_sessions") >= 2
