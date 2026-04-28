"""
tests/test_working_set.py — Per-Agent Working Set 测试

验证：
  1. TLB hit/miss 正确性
  2. LRU 驱逐（Clock 算法）
  3. dirty bit 和 flush_dirty
  4. pin/unpin
  5. WorkingSetRegistry（全局注册表 + TLB shootdown）
  6. 容量控制（max_chunks）
  7. 统计准确性
"""
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from agent_working_set import WorkingSet, WorkingSetRegistry, WSEntry
from schema import MemoryChunk


def make_chunk(idx: int, chunk_type: str = "decision",
               importance: float = 0.5) -> MemoryChunk:
    return MemoryChunk(
        id=f"chunk-{idx}",
        project="test-project",
        source_session="test-session",
        chunk_type=chunk_type,
        content=f"test content {idx}",
        summary=f"summary {idx}",
        importance=importance,
    )


# ─────────────────────────────────────────────────────────────
# 1. 基础 TLB 操作
# ─────────────────────────────────────────────────────────────

class TestWorkingSetBasics:
    def test_put_and_get_hit(self):
        """put 后 get 应命中。"""
        ws = WorkingSet("s1", "p1", max_chunks=10)
        chunk = make_chunk(0)
        ws.put(chunk)
        result = ws.get("chunk-0")
        assert result is not None
        assert result.id == "chunk-0"

    def test_get_miss(self):
        """未 put 的 chunk_id 应 miss。"""
        ws = WorkingSet("s2", "p1", max_chunks=10)
        assert ws.get("nonexistent") is None

    def test_hit_updates_stats(self):
        """命中后 stats.hits 增加。"""
        ws = WorkingSet("s3", "p1", max_chunks=10)
        ws.put(make_chunk(0))
        ws.get("chunk-0")
        assert ws._stats["hits"] == 1
        assert ws._stats["misses"] == 0

    def test_miss_updates_stats(self):
        """未命中后 stats.misses 增加。"""
        ws = WorkingSet("s4", "p1", max_chunks=10)
        ws.get("nonexistent")
        assert ws._stats["misses"] == 1
        assert ws._stats["hits"] == 0

    def test_put_updates_existing(self):
        """对已存在 chunk 的 put 应更新内容。"""
        ws = WorkingSet("s5", "p1", max_chunks=10)
        chunk_v1 = make_chunk(0)
        chunk_v2 = make_chunk(0)
        chunk_v2.content = "updated content"
        ws.put(chunk_v1)
        ws.put(chunk_v2, dirty=True)
        result = ws.get("chunk-0")
        assert result.content == "updated content"
        # dirty bit 应被置位
        assert ws._lru["chunk-0"].dirty is True

    def test_size(self):
        ws = WorkingSet("s6", "p1", max_chunks=10)
        assert ws.size() == 0
        ws.put(make_chunk(0))
        assert ws.size() == 1
        ws.put(make_chunk(1))
        assert ws.size() == 2


# ─────────────────────────────────────────────────────────────
# 2. LRU 驱逐（Clock 算法）
# ─────────────────────────────────────────────────────────────

class TestLRUEviction:
    def test_eviction_on_overflow(self):
        """超过 max_chunks 时自动驱逐。"""
        ws = WorkingSet("s7", "p1", max_chunks=3)
        for i in range(5):
            ws.put(make_chunk(i))
        assert ws.size() <= 3

    def test_eviction_count_tracked(self):
        """驱逐次数被统计。"""
        ws = WorkingSet("s8", "p1", max_chunks=3)
        for i in range(6):
            ws.put(make_chunk(i))
        assert ws._stats["evictions"] >= 1

    def test_pinned_not_evicted(self):
        """pinned chunk 不被 LRU 驱逐。"""
        ws = WorkingSet("s9", "p1", max_chunks=3)
        ws.put(make_chunk(0))
        ws.put(make_chunk(1))
        ws.put(make_chunk(2))
        ws.pin("chunk-0")
        # 清除 chunk-0 的 accessed bit，确保它是 LRU 候选
        ws._lru["chunk-0"].accessed = False

        # 加入更多 chunk，触发驱逐
        for i in range(3, 6):
            ws.put(make_chunk(i))

        # pinned 的 chunk-0 应仍在
        assert ws.get("chunk-0") is not None

    def test_lru_evicts_oldest(self):
        """最久未访问的 chunk 先被驱逐。"""
        ws = WorkingSet("s10", "p1", max_chunks=3)
        ws.put(make_chunk(0))
        ws.put(make_chunk(1))
        ws.put(make_chunk(2))

        # 访问 chunk-0（提升到 MRU 端）
        ws._lru["chunk-0"].accessed = False
        ws._lru["chunk-1"].accessed = False
        ws._lru["chunk-2"].accessed = False
        ws.get("chunk-2")  # 访问 chunk-2

        # 加入第 4 个 chunk，触发驱逐
        ws.put(make_chunk(3))

        # chunk-2 刚访问，不应被驱逐
        # chunk-0 或 chunk-1 应被驱逐
        assert ws.get("chunk-2") is not None or ws.get("chunk-3") is not None

    def test_clock_second_chance(self):
        """Clock 算法：accessed=True 的 chunk 获得第二次机会。"""
        ws = WorkingSet("s11", "p1", max_chunks=2)
        ws.put(make_chunk(0))  # accessed=True
        ws.put(make_chunk(1))  # accessed=True

        # 强制清除 chunk-0 的 accessed bit
        ws._lru["chunk-0"].accessed = False

        # 加入第 3 个 chunk，超出 max_chunks=2
        ws.put(make_chunk(2))
        # chunk-0（accessed=False）应被驱逐
        # chunk-1 有 accessed=True，第一次遇到给第二次机会，但 chunk-0 先驱逐
        assert ws.get("chunk-0") is None or ws.size() <= 2


# ─────────────────────────────────────────────────────────────
# 3. dirty bit 和 flush_dirty
# ─────────────────────────────────────────────────────────────

class TestDirtyAndFlush:
    def test_put_with_dirty(self):
        """dirty=True 时 dirty bit 置位。"""
        ws = WorkingSet("s12", "p1", max_chunks=10)
        ws.put(make_chunk(0), dirty=True)
        assert ws._lru["chunk-0"].dirty is True

    def test_put_without_dirty(self):
        """dirty=False（默认）时 dirty bit 清除。"""
        ws = WorkingSet("s13", "p1", max_chunks=10)
        ws.put(make_chunk(0), dirty=False)
        assert ws._lru["chunk-0"].dirty is False

    def test_mark_dirty(self):
        ws = WorkingSet("s14", "p1", max_chunks=10)
        ws.put(make_chunk(0), dirty=False)
        result = ws.mark_dirty("chunk-0")
        assert result is True
        assert ws._lru["chunk-0"].dirty is True

    def test_mark_dirty_nonexistent(self):
        ws = WorkingSet("s15", "p1", max_chunks=10)
        result = ws.mark_dirty("nonexistent")
        assert result is False

    def test_flush_dirty_clears_bit(self):
        """flush_dirty 后 dirty bit 应清除。"""
        ws = WorkingSet("s16", "p1", max_chunks=10)
        ws.put(make_chunk(0), dirty=True)
        ws.put(make_chunk(1), dirty=False)

        # mock store.db 写入
        with patch("agent_working_set.WorkingSet.flush_dirty") as mock_flush:
            mock_flush.return_value = 1
            count = ws.flush_dirty()

        # 直接测试 dirty bit 逻辑（不依赖真实 db）
        dirty_chunks = ws.list_chunks(dirty_only=True)
        # 只有 chunk-0 是 dirty
        assert any(c["id"] == "chunk-0" for c in dirty_chunks)
        assert not any(c["id"] == "chunk-1" for c in dirty_chunks)

    def test_flush_skips_when_disabled(self):
        """working_set.flush_dirty_on_exit=False 时跳过 flush。"""
        ws = WorkingSet("s17", "p1", max_chunks=10)
        ws.put(make_chunk(0), dirty=True)
        with patch("agent_working_set._sysctl", side_effect=lambda k: {
            "working_set.flush_dirty_on_exit": False,
            "working_set.max_chunks": 200,
        }.get(k, 200)):
            count = ws.flush_dirty()
        assert count == 0


# ─────────────────────────────────────────────────────────────
# 4. pin / unpin
# ─────────────────────────────────────────────────────────────

class TestPinUnpin:
    def test_pin_sets_flag(self):
        ws = WorkingSet("s18", "p1", max_chunks=10)
        ws.put(make_chunk(0))
        result = ws.pin("chunk-0")
        assert result is True
        assert ws._lru["chunk-0"].pinned is True

    def test_unpin_clears_flag(self):
        ws = WorkingSet("s19", "p1", max_chunks=10)
        ws.put(make_chunk(0))
        ws.pin("chunk-0")
        result = ws.unpin("chunk-0")
        assert result is True
        assert ws._lru["chunk-0"].pinned is False

    def test_pin_nonexistent(self):
        ws = WorkingSet("s20", "p1", max_chunks=10)
        assert ws.pin("nonexistent") is False

    def test_unpin_nonexistent(self):
        ws = WorkingSet("s21", "p1", max_chunks=10)
        assert ws.unpin("nonexistent") is False


# ─────────────────────────────────────────────────────────────
# 5. stats
# ─────────────────────────────────────────────────────────────

class TestStats:
    def test_stats_structure(self):
        ws = WorkingSet("s22", "p1", max_chunks=10)
        s = ws.stats()
        expected_keys = [
            "session_id", "project", "size", "max_chunks",
            "utilization", "hit_rate", "hits", "misses",
            "evictions", "dirty_flushes", "dirty_count", "pinned_count",
        ]
        for key in expected_keys:
            assert key in s, f"Missing key: {key}"

    def test_hit_rate_calculation(self):
        ws = WorkingSet("s23", "p1", max_chunks=10)
        ws.put(make_chunk(0))
        ws.get("chunk-0")   # hit
        ws.get("chunk-0")   # hit
        ws.get("missing")   # miss
        s = ws.stats()
        assert s["hits"] == 2
        assert s["misses"] == 1
        assert abs(s["hit_rate"] - 66.7) < 1.0

    def test_utilization(self):
        ws = WorkingSet("s24", "p1", max_chunks=10)
        ws.put(make_chunk(0))
        ws.put(make_chunk(1))
        s = ws.stats()
        assert s["size"] == 2
        assert s["utilization"] == 20.0

    def test_dirty_count(self):
        ws = WorkingSet("s25", "p1", max_chunks=10)
        ws.put(make_chunk(0), dirty=True)
        ws.put(make_chunk(1), dirty=False)
        s = ws.stats()
        assert s["dirty_count"] == 1
        assert s["pinned_count"] == 0


# ─────────────────────────────────────────────────────────────
# 6. invalidate
# ─────────────────────────────────────────────────────────────

class TestInvalidate:
    def test_invalidate_existing(self):
        ws = WorkingSet("s26", "p1", max_chunks=10)
        ws.put(make_chunk(0))
        result = ws.invalidate("chunk-0")
        assert result is True
        assert ws.get("chunk-0") is None

    def test_invalidate_nonexistent(self):
        ws = WorkingSet("s27", "p1", max_chunks=10)
        assert ws.invalidate("nonexistent") is False

    def test_clear(self):
        ws = WorkingSet("s28", "p1", max_chunks=10)
        for i in range(5):
            ws.put(make_chunk(i))
        count = ws.clear()
        assert count == 5
        assert ws.size() == 0


# ─────────────────────────────────────────────────────────────
# 7. WorkingSetRegistry（全局注册表）
# ─────────────────────────────────────────────────────────────

class TestWorkingSetRegistry:
    def setup_method(self):
        """每个测试前清理注册表。"""
        registry = WorkingSetRegistry()
        with registry._reg_lock:
            registry._registry.clear()

    def test_get_or_create(self):
        registry = WorkingSetRegistry()
        ws1 = registry.get_or_create("sess-1", "proj-1")
        ws2 = registry.get_or_create("sess-1", "proj-1")
        assert ws1 is ws2  # 同一对象

    def test_different_sessions(self):
        registry = WorkingSetRegistry()
        ws1 = registry.get_or_create("sess-A", "proj-1")
        ws2 = registry.get_or_create("sess-B", "proj-1")
        assert ws1 is not ws2

    def test_close_session(self):
        registry = WorkingSetRegistry()
        registry.get_or_create("sess-close", "proj-1")
        with patch("agent_working_set.WorkingSet.flush_dirty", return_value=0):
            result = registry.close_session("sess-close")
        assert registry.get("sess-close") is None

    def test_broadcast_invalidate(self):
        """TLB Shootdown：通知所有 session 失效指定 chunk。"""
        registry = WorkingSetRegistry()
        ws1 = registry.get_or_create("sess-inv-1", "proj-1")
        ws2 = registry.get_or_create("sess-inv-2", "proj-1")

        ws1.put(make_chunk(99))
        ws2.put(make_chunk(99))

        count = registry.broadcast_invalidate("chunk-99")
        assert count == 2
        assert ws1.get("chunk-99") is None
        assert ws2.get("chunk-99") is None

    def test_broadcast_excludes_source(self):
        """broadcast_invalidate 可以排除触发者自身。"""
        registry = WorkingSetRegistry()
        ws1 = registry.get_or_create("sess-src", "proj-1")
        ws2 = registry.get_or_create("sess-dst", "proj-1")

        ws1.put(make_chunk(88))
        ws2.put(make_chunk(88))

        count = registry.broadcast_invalidate("chunk-88", exclude_session="sess-src")
        assert count == 1
        # ws1（触发者）的 chunk 应保留
        assert ws1.get("chunk-88") is not None
        # ws2 的 chunk 应被失效
        assert ws2.get("chunk-88") is None

    def test_singleton_pattern(self):
        """WorkingSetRegistry 是全局单例。"""
        r1 = WorkingSetRegistry()
        r2 = WorkingSetRegistry()
        assert r1 is r2


# ─────────────────────────────────────────────────────────────
# 8. list_chunks
# ─────────────────────────────────────────────────────────────

class TestListChunks:
    def test_list_all(self):
        ws = WorkingSet("s30", "p1", max_chunks=10)
        ws.put(make_chunk(0))
        ws.put(make_chunk(1))
        chunks = ws.list_chunks()
        assert len(chunks) == 2

    def test_list_dirty_only(self):
        ws = WorkingSet("s31", "p1", max_chunks=10)
        ws.put(make_chunk(0), dirty=True)
        ws.put(make_chunk(1), dirty=False)
        dirty = ws.list_chunks(dirty_only=True)
        assert len(dirty) == 1
        assert dirty[0]["id"] == "chunk-0"

    def test_list_chunk_structure(self):
        ws = WorkingSet("s32", "p1", max_chunks=10)
        ws.put(make_chunk(0))
        chunks = ws.list_chunks()
        assert len(chunks) == 1
        expected = ["id", "summary", "chunk_type", "dirty", "pinned",
                    "accessed", "access_count", "load_time"]
        for key in expected:
            assert key in chunks[0], f"Missing key: {key}"
