#!/usr/bin/env python3
"""
test_vfs_adapter.py — Gap 3: VFS 外部适配器注册中心测试

验证：
1. VFSAdapterRegistry.register() 能注册 backend
2. VFSAdapterRegistry.unregister() 能注销 backend
3. VFSAdapterRegistry.list_backends() 按 priority 排序
4. VFSAdapterRegistry.source_weight() 正确映射 priority → weight
5. KnowledgeVFS.register_external_backend() 同步到本实例 + 全局注册表
6. KnowledgeVFS.search() 自动包含已注册外部 backend 的结果
7. Mock backend 结果正确合并到 search 输出（分数 + 去重）
8. 注销后 search() 不再返回该 backend 的结果
9. 注册失败（非 VFSBackend 实例）抛出 TypeError
10. 并发注册/注销不引起竞态
"""
import tmpfs  # noqa: F401 — must be first to isolate test DB

import sys
import unittest
import threading
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from vfs_core import (
    VFSBackend, VFSItem, VFSMetadata,
    VFSItemType, VFSSource, VFSScope,
)
from vfs_adapter_registry import VFSAdapterRegistry
from vfs import KnowledgeVFS


# ── Mock Backend（测试用）──────────────────────────────────────────────────────

class MockBackend(VFSBackend):
    """可控的 mock backend，用于测试注册/搜索集成。"""

    def __init__(self, name: str, items: List[VFSItem] = None):
        self._name = name
        self._items = items or []

    @property
    def name(self) -> str:
        return self._name

    @property
    def source_type(self) -> str:
        return self._name  # 用 name 作为 source_type

    def read(self, path: str) -> Optional[VFSItem]:
        for item in self._items:
            if item.path == path:
                return item
        return None

    def search(self, query: str, top_k: int = 5) -> List[VFSItem]:
        # 简单：返回所有 item（测试用，不做实际搜索）
        results = []
        for item in self._items:
            if query.lower() in item.summary.lower() or query.lower() in item.content.lower():
                results.append(item)
        return results[:top_k]

    def write(self, item: VFSItem) -> bool:
        self._items.append(item)
        return True

    def delete(self, path: str) -> bool:
        before = len(self._items)
        self._items = [i for i in self._items if i.path != path]
        return len(self._items) < before


def _make_item(name: str, summary: str, score: float = 0.8) -> VFSItem:
    """创建测试用 VFSItem。"""
    import uuid
    now = "2026-04-21T00:00:00+00:00"
    return VFSItem(
        id=uuid.uuid4().hex[:12],
        path=f"mock://{name}/{summary[:20].replace(' ', '_')}",
        content=f"Content about {summary}",
        summary=summary,
        source=name,
        score=score,
        type=VFSItemType.DECISION.value,
        metadata=VFSMetadata(
            created_at=now,
            updated_at=now,
            last_accessed=now,
            importance=0.8,
            retrievability=0.5,
            access_count=0,
            source_session="test-session",
            scope=VFSScope.PROJECT.value,
            tags=["mock", name],
            project=name,
            content_hash="",
        ),
    )


# ── Test Classes ──────────────────────────────────────────────────────────────

class TestVFSAdapterRegistryBasic(unittest.TestCase):
    """验证 VFSAdapterRegistry 的基本注册/注销/查询功能。"""

    def setUp(self):
        VFSAdapterRegistry.clear()

    def tearDown(self):
        VFSAdapterRegistry.clear()

    def test_register_and_get_backend(self):
        """注册 backend 后能按名获取。"""
        backend = MockBackend("test-mock")
        VFSAdapterRegistry.register("test-mock", backend, priority=60)
        retrieved = VFSAdapterRegistry.get_backend("test-mock")
        self.assertIs(retrieved, backend)

    def test_unregister_backend(self):
        """注销后无法再获取。"""
        backend = MockBackend("test-unreg")
        VFSAdapterRegistry.register("test-unreg", backend)
        result = VFSAdapterRegistry.unregister("test-unreg")
        self.assertTrue(result)
        self.assertIsNone(VFSAdapterRegistry.get_backend("test-unreg"))

    def test_unregister_nonexistent_returns_false(self):
        """注销不存在的 backend 返回 False。"""
        result = VFSAdapterRegistry.unregister("does-not-exist")
        self.assertFalse(result)

    def test_list_backends_sorted_by_priority(self):
        """list_backends() 按 priority 降序排列。"""
        VFSAdapterRegistry.register("low-prio", MockBackend("low-prio"), priority=10)
        VFSAdapterRegistry.register("high-prio", MockBackend("high-prio"), priority=90)
        VFSAdapterRegistry.register("mid-prio", MockBackend("mid-prio"), priority=50)

        entries = VFSAdapterRegistry.list_backends()
        priorities = [e.priority for e in entries]
        self.assertEqual(priorities, sorted(priorities, reverse=True),
                         "list_backends() should be sorted by priority descending")

    def test_register_type_error_on_non_backend(self):
        """非 VFSBackend 实例注册时抛出 TypeError。"""
        with self.assertRaises(TypeError):
            VFSAdapterRegistry.register("bad", "not-a-backend", priority=50)

    def test_register_overwrites_same_name(self):
        """同名注册会覆盖旧 backend。"""
        b1 = MockBackend("overwrite-test")
        b2 = MockBackend("overwrite-test")
        VFSAdapterRegistry.register("overwrite-test", b1)
        VFSAdapterRegistry.register("overwrite-test", b2)
        retrieved = VFSAdapterRegistry.get_backend("overwrite-test")
        self.assertIs(retrieved, b2, "Second register should overwrite first")

    def test_priority_clamped_to_0_100(self):
        """priority 被钳制到 [0, 100]。"""
        VFSAdapterRegistry.register("clamp-test", MockBackend("clamp-test"), priority=999)
        entry = VFSAdapterRegistry.get_entry("clamp-test")
        self.assertLessEqual(entry.priority, 100)

        VFSAdapterRegistry.register("clamp-neg", MockBackend("clamp-neg"), priority=-50)
        entry2 = VFSAdapterRegistry.get_entry("clamp-neg")
        self.assertGreaterEqual(entry2.priority, 0)

    def test_source_weight_mapping(self):
        """source_weight() 正确将 priority 映射到 [0.5, 1.0]。"""
        VFSAdapterRegistry.register("w0", MockBackend("w0"), priority=0)
        VFSAdapterRegistry.register("w100", MockBackend("w100"), priority=100)
        VFSAdapterRegistry.register("w50", MockBackend("w50"), priority=50)

        self.assertAlmostEqual(VFSAdapterRegistry.source_weight("w0"), 0.5)
        self.assertAlmostEqual(VFSAdapterRegistry.source_weight("w100"), 1.0)
        self.assertAlmostEqual(VFSAdapterRegistry.source_weight("w50"), 0.75)

    def test_source_weight_nonexistent_returns_0_5(self):
        """未注册 backend 的权重返回 0.5（默认值）。"""
        weight = VFSAdapterRegistry.source_weight("nonexistent-backend")
        self.assertAlmostEqual(weight, 0.5)

    def test_list_names(self):
        """list_names() 返回所有注册名称。"""
        VFSAdapterRegistry.register("a", MockBackend("a"))
        VFSAdapterRegistry.register("b", MockBackend("b"))
        names = VFSAdapterRegistry.list_names()
        self.assertIn("a", names)
        self.assertIn("b", names)


class TestKnowledgeVFSExternalBackend(unittest.TestCase):
    """验证 KnowledgeVFS 与外部 backend 注册的集成。"""

    def setUp(self):
        VFSAdapterRegistry.clear()
        self.vfs = KnowledgeVFS()

    def tearDown(self):
        VFSAdapterRegistry.clear()

    def test_register_external_backend_in_vfs(self):
        """register_external_backend() 成功注册到 vfs 实例。"""
        backend = MockBackend("ext-test")
        self.vfs.register_external_backend("ext-test", backend, priority=70)
        # 全局注册表也应该有
        retrieved = VFSAdapterRegistry.get_backend("ext-test")
        self.assertIs(retrieved, backend)

    def test_unregister_external_backend(self):
        """unregister_external_backend() 从 vfs 和全局注册表都移除。"""
        backend = MockBackend("ext-unreg")
        self.vfs.register_external_backend("ext-unreg", backend)
        result = self.vfs.unregister_external_backend("ext-unreg")
        self.assertTrue(result)
        self.assertIsNone(VFSAdapterRegistry.get_backend("ext-unreg"))

    def test_search_includes_external_backend_results(self):
        """search() 自动包含已注册外部 backend 的结果。"""
        # 创建含唯一 summary 的 mock backend item（确保 top_k 足够大时能被找到）
        # 使用极高分数（score=1.0）保证排在前面，以及极具体的关键词避免被其他 backend 压倒
        unique_summary = "xyzzy_unique_vfs_adapter_test_phrase_42"
        item = _make_item("ext-search", unique_summary, score=1.0)
        backend = MockBackend("ext-search", items=[item])
        self.vfs.register_external_backend("ext-search", backend, priority=100)

        results = self.vfs.search(unique_summary, top_k=20)
        # 结果中应包含 mock backend 的 item
        found = any(r.source == "ext-search" for r in results)
        self.assertTrue(found,
                        f"Mock backend results should be in search output. "
                        f"Got sources: {[r.source for r in results]}")

    def test_search_without_external_backend_still_works(self):
        """无外部 backend 时 search() 正常工作。"""
        results = self.vfs.search("test query", top_k=5)
        self.assertIsInstance(results, list)  # 不报错，返回列表

    def test_external_backend_results_deduplicated(self):
        """相同 summary 的结果只保留最高分（全局去重）。"""
        # 两个 backend 返回相同 summary 的 item
        item1 = _make_item("ext-dup-a", "dedup test summary", score=0.7)
        item2 = _make_item("ext-dup-b", "dedup test summary", score=0.9)

        b1 = MockBackend("ext-dup-a", items=[item1])
        b2 = MockBackend("ext-dup-b", items=[item2])
        self.vfs.register_external_backend("ext-dup-a", b1)
        self.vfs.register_external_backend("ext-dup-b", b2)

        results = self.vfs.search("dedup test", top_k=10)
        # 相同 summary 只出现一次
        summaries = [r.summary for r in results if r.summary == "dedup test summary"]
        self.assertLessEqual(len(summaries), 1,
                            "Duplicate summaries should be deduplicated")

    def test_global_registry_synced_to_new_vfs_instance(self):
        """全局注册表的 backend 在新 KnowledgeVFS 实例中也能发现。"""
        item = _make_item("global-reg", "cross instance knowledge", score=0.85)
        backend = MockBackend("global-reg", items=[item])
        # 直接注册到全局注册表（不通过 vfs 实例）
        VFSAdapterRegistry.register("global-reg", backend, priority=70)

        # 新 vfs 实例的 search() 应该能感知全局注册
        new_vfs = KnowledgeVFS()
        results = new_vfs.search("cross instance", top_k=10)
        found = any(r.source == "global-reg" for r in results)
        self.assertTrue(found,
                        "New VFS instance should auto-discover globally registered backends")


class TestVFSAdapterConcurrency(unittest.TestCase):
    """验证并发注册/注销不引起竞态（线程安全）。"""

    def setUp(self):
        VFSAdapterRegistry.clear()

    def tearDown(self):
        VFSAdapterRegistry.clear()

    def test_concurrent_register_no_crash(self):
        """并发注册 10 个 backend 不引起竞态或崩溃。"""
        errors = []

        def register_worker(i):
            try:
                name = f"concurrent-{i}"
                backend = MockBackend(name)
                VFSAdapterRegistry.register(name, backend, priority=i)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_worker, args=(i,))
                   for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Concurrent register errors: {errors}")
        self.assertEqual(len(VFSAdapterRegistry.list_names()), 10)

    def test_concurrent_register_unregister_no_crash(self):
        """并发注册和注销不引起崩溃。"""
        errors = []

        def worker(i):
            try:
                name = f"conc-rw-{i}"
                VFSAdapterRegistry.register(name, MockBackend(name))
                VFSAdapterRegistry.unregister(name)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,))
                   for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Concurrent errors: {errors}")


class TestVFSAdapterRegistryEdgeCases(unittest.TestCase):
    """边界情况和优雅降级测试。"""

    def setUp(self):
        VFSAdapterRegistry.clear()

    def tearDown(self):
        VFSAdapterRegistry.clear()

    def test_clear_returns_count(self):
        """clear() 返回清除条目数。"""
        VFSAdapterRegistry.register("c1", MockBackend("c1"))
        VFSAdapterRegistry.register("c2", MockBackend("c2"))
        count = VFSAdapterRegistry.clear()
        self.assertEqual(count, 2)

    def test_empty_registry_list_returns_empty(self):
        """空注册表 list_backends() 返回空列表。"""
        entries = VFSAdapterRegistry.list_backends()
        self.assertIsInstance(entries, list)
        self.assertEqual(len(entries), 0)

    def test_search_with_failing_backend_still_returns_others(self):
        """某个 backend 的 search() 抛异常，不影响其他 backend 结果。"""

        class BrokenBackend(VFSBackend):
            @property
            def name(self): return "broken"
            @property
            def source_type(self): return "broken"
            def read(self, path): return None
            def search(self, query, top_k=5): raise RuntimeError("backend broken!")
            def write(self, item): return False
            def delete(self, path): return False

        vfs = KnowledgeVFS()
        vfs.register_external_backend("broken", BrokenBackend(), priority=50)
        # search() 不应因 broken backend 而崩溃
        try:
            results = vfs.search("anything", top_k=5)
            self.assertIsInstance(results, list)
        except Exception as e:
            self.fail(f"search() should not raise when backend is broken: {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
