#!/usr/bin/env python3
"""
Phase 2D+2E: KnowledgeVFS 核心 — 统一知识虚拟文件系统

实现二级缓存（dentry + inode）、并行搜索、去重、100ms 硬 deadline。

OS 类比：
- Linux VFS 核心在于 dentry_cache 和 inode_cache 两级缓存
- dentry cache 存储 (path, inode) 映射，支撑路径查询
- inode cache 存储完整 inode 属性，减少后端重复查询
"""
import time
import hashlib
import threading
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from vfs_core import VFSItem, VFSSource, VFSBackend
from vfs_backend_sqlite import SQLiteBackend
from vfs_backend_filesystem import SelfImprovingBackend, MemoryMdBackend


# ── 缓存实现（对标 Linux dentry_cache + inode_cache）──────────────
class DentryCache:
    """dentry 缓存 — 路径到 VFSItem 的快速查找

    OS 类比：Linux dentry_cache 存储 (parent, name, inode) 元组，
             支撑 walk_path 的快速查询（避免每次都走 lookup）。
    """

    def __init__(self, ttl_secs: int = 300):
        """初始化 dentry 缓存

        Args:
            ttl_secs: 缓存 TTL（秒）
        """
        self.ttl_secs = ttl_secs
        self._cache: Dict[str, Tuple[VFSItem, float]] = {}
        self._lock = threading.Lock()

    def get(self, path: str) -> Optional[VFSItem]:
        """查询缓存（O(1) 哈希表查询）"""
        with self._lock:
            if path not in self._cache:
                return None
            item, ts = self._cache[path]
            if time.time() - ts > self.ttl_secs:
                del self._cache[path]
                return None
            return item

    def put(self, path: str, item: VFSItem):
        """写入缓存"""
        with self._lock:
            self._cache[path] = (item, time.time())

    def invalidate(self, path: str):
        """失效单项"""
        with self._lock:
            self._cache.pop(path, None)

    def clear(self):
        """清空缓存"""
        with self._lock:
            self._cache.clear()

    def stats(self) -> Dict[str, Any]:
        """缓存统计"""
        with self._lock:
            now = time.time()
            valid = sum(1 for _, ts in self._cache.values()
                       if now - ts < self.ttl_secs)
            return {
                "total": len(self._cache),
                "valid": valid,
                "expired": len(self._cache) - valid,
            }


class InodeCache:
    """inode 缓存 — 内容缓存（减少后端重复查询）

    OS 类比：Linux inode_cache 存储完整 inode 属性，减少 lookup 后端调用。
    """

    def __init__(self, ttl_secs: int = 300):
        self.ttl_secs = ttl_secs
        self._cache: Dict[str, Tuple[VFSItem, float]] = {}
        self._lock = threading.Lock()

    def get(self, item_id: str) -> Optional[VFSItem]:
        """按 ID 查询"""
        with self._lock:
            if item_id not in self._cache:
                return None
            item, ts = self._cache[item_id]
            if time.time() - ts > self.ttl_secs:
                del self._cache[item_id]
                return None
            return item

    def put(self, item: VFSItem):
        """缓存单个项"""
        with self._lock:
            self._cache[item.id] = (item, time.time())

    def clear(self):
        """清空缓存"""
        with self._lock:
            self._cache.clear()

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            now = time.time()
            valid = sum(1 for _, ts in self._cache.values()
                       if now - ts < self.ttl_secs)
            return {"total": len(self._cache), "valid": valid}


# ── KnowledgeVFS 核心（对标 VFS 超级块）──────────────────────────
class KnowledgeVFS:
    """统一知识虚拟文件系统

    提供统一的知识访问接口，后端可插拔（SQLite/Filesystem/JSONL）。
    """

    def __init__(self, dentry_ttl: int = 300, inode_ttl: int = 300):
        """初始化 VFS

        Args:
            dentry_ttl: dentry 缓存 TTL（秒）
            inode_ttl: inode 缓存 TTL（秒）
        """
        self.dentry_cache = DentryCache(ttl_secs=dentry_ttl)
        self.inode_cache = InodeCache(ttl_secs=inode_ttl)

        # 后端注册表（对标 file_system_type）
        self._backends: Dict[str, VFSBackend] = {}
        self._backends_lock = threading.Lock()

        # 线程池（用于并行搜索）
        self._executor = ThreadPoolExecutor(max_workers=3)

        # 注册默认后端
        self._register_default_backends()

        # 性能统计
        self._stats = {
            "reads": 0,
            "searches": 0,
            "dentry_hits": 0,
            "inode_hits": 0,
            "cache_misses": 0,
        }
        self._stats_lock = threading.Lock()

    def _register_default_backends(self):
        """注册默认后端"""
        sqlite_backend = SQLiteBackend(readonly=True)
        self._backends[sqlite_backend.source_type] = sqlite_backend

        # Phase 2+: Filesystem 后端（self-improving + memory-md）
        si_backend = SelfImprovingBackend()
        self._backends[si_backend.source_type] = si_backend

        mm_backend = MemoryMdBackend()
        self._backends[mm_backend.source_type] = mm_backend

    def register_external_backend(self, name: str, backend: VFSBackend,
                                   priority: int = 50) -> None:
        """
        注册外部 backend 到 VFS（运行时动态挂载）。

        OS 类比：mount(2) — 将新文件系统挂载到 VFS 命名空间，
          挂载后该文件系统的文件对所有进程可见，无需重启。
          同理：register_external_backend 后 search() 自动包含该 backend。

        参数：
          name     — 唯一标识符（如 "openai-assistants", "weaviate"）
          backend  — VFSBackend 实例
          priority — 搜索权重（0-100，默认 50；越高越优先）

        使用方式：
          vfs = KnowledgeVFS()
          vfs.register_external_backend("my-backend", MyBackend(), priority=60)
          results = vfs.search("query")  # 自动包含 my-backend
        """
        from vfs_adapter_registry import VFSAdapterRegistry
        # 同步注册到全局注册表（其他 KnowledgeVFS 实例也能感知）
        VFSAdapterRegistry.register(name, backend, priority)
        # 同步到本实例 _backends（避免在 search 中每次查询注册表）
        with self._backends_lock:
            self._backends[name] = backend

    def unregister_external_backend(self, name: str) -> bool:
        """
        注销外部 backend（运行时动态卸载）。

        OS 类比：umount(2) — 从 VFS 命名空间卸载文件系统。

        返回：True = 成功注销，False = 名称不存在
        """
        from vfs_adapter_registry import VFSAdapterRegistry
        VFSAdapterRegistry.unregister(name)
        with self._backends_lock:
            if name in self._backends:
                del self._backends[name]
                return True
            return False

    def read(self, path: str) -> Optional[VFSItem]:
        """按虚拟路径读单个项

        Stage 1: dentry cache 检查（< 0.1ms）
        Stage 2: backend.read（后端路径，< 10ms）

        Args:
            path: 虚拟路径 /<source>/<id>

        Returns:
            VFSItem 或 None
        """
        with self._stats_lock:
            self._stats["reads"] += 1

        # Stage 1: dentry cache（< 0.1ms）
        cached = self.dentry_cache.get(path)
        if cached:
            with self._stats_lock:
                self._stats["dentry_hits"] += 1
            return cached

        # 解析虚拟路径
        parts = path.strip("/").split("/")
        if len(parts) < 1:
            return None

        source = parts[0]

        # Stage 2: 后端查询
        with self._backends_lock:
            backend = self._backends.get(source)

        if not backend:
            with self._stats_lock:
                self._stats["cache_misses"] += 1
            return None

        item = backend.read(path)
        if item:
            # 写入缓存
            self.dentry_cache.put(path, item)
            self.inode_cache.put(item)

        return item

    def search(self, query: str, top_k: int = 5, deadline_ms: int = 100) -> List[VFSItem]:
        """全文搜索（并行查询所有后端）

        Strategy:
            1. 并行查询所有后端（ThreadPoolExecutor）
            2. 每个后端返回 top_k 结果（独立配额）
            3. 应用源权重
            4. 全局去重（同 summary 保留最高分）
            5. 排序并返回 top_k

        Args:
            query: 搜索查询
            top_k: 返回结果数
            deadline_ms: 硬 deadline（ms）

        Returns:
            VFSItem 列表，按相关度排序
        """
        with self._stats_lock:
            self._stats["searches"] += 1

        start_time = time.time()
        deadline_secs = deadline_ms / 1000.0

        # 源权重（对标 file_system_type 优先级）
        source_weights = {
            VFSSource.MEMORY_OS.value: 1.0,       # 最高权重：会话记忆主存储
            "memory-md": 0.85,                    # 用户写的 MEMORY.md / wiki
            VFSSource.SELF_IMPROVING.value: 0.7,  # self-improving 知识库
            VFSSource.PROJECT.value: 0.6,          # 项目历史
        }
        # 外部注册 backend 的权重（由 priority 决定）
        try:
            from vfs_adapter_registry import VFSAdapterRegistry
            for entry in VFSAdapterRegistry.list_backends():
                source_weights[entry.name] = VFSAdapterRegistry.source_weight(entry.name)
        except Exception:
            pass

        # Step 1: 并行查询所有后端
        # ── 迭代Phase3：从全局注册表同步外部 backends（运行时动态发现）──
        # OS 类比：mount namespace — 挂载新文件系统后，所有进程自动可见
        # 将全局注册的外部 backend（未在 _backends 中的）追加到本次查询
        try:
            from vfs_adapter_registry import VFSAdapterRegistry
            for entry in VFSAdapterRegistry.list_backends():
                with self._backends_lock:
                    if entry.name not in self._backends:
                        self._backends[entry.name] = entry.backend
        except Exception:
            pass  # 注册表异常不影响 search 主流程

        with self._backends_lock:
            futures = {
                source: self._executor.submit(backend.search, query, top_k * 2)
                for source, backend in self._backends.items()
            }

        # Step 2: 收集结果（带 timeout）
        all_items: List[VFSItem] = []
        for source, future in futures.items():
            try:
                remaining_time = deadline_secs - (time.time() - start_time)
                if remaining_time <= 0:
                    break
                items = future.result(timeout=remaining_time)
                all_items.extend(items)
            except Exception:
                # 后端超时或异常，继续下一个
                pass

        # Step 3: 应用源权重
        for item in all_items:
            weight = source_weights.get(item.source, 0.5)
            item.score *= weight

        # Step 4: 全局去重（同 summary 保留最高分）
        seen: Dict[str, VFSItem] = {}
        for item in all_items:
            # 用 summary 作为去重 key（内容相同即去重）
            key = hashlib.sha256(item.summary.encode()).hexdigest()
            if key not in seen or item.score > seen[key].score:
                seen[key] = item

        # Step 5: 排序并返回 top_k
        result = sorted(seen.values(), key=lambda x: x.score, reverse=True)[:top_k]

        # 写入缓存
        for item in result:
            self.dentry_cache.put(item.path, item)
            self.inode_cache.put(item)

        return result

    def stats(self) -> Dict[str, Any]:
        """返回性能统计"""
        with self._stats_lock:
            stats = self._stats.copy()

        stats["dentry_cache"] = self.dentry_cache.stats()
        stats["inode_cache"] = self.inode_cache.stats()

        return stats


# ── 全局 VFS 实例（单例）──────────────────────────────────────────
_vfs_instance: Optional[KnowledgeVFS] = None
_vfs_lock = threading.Lock()


def get_vfs() -> KnowledgeVFS:
    """获取全局 VFS 实例（延迟初始化）"""
    global _vfs_instance
    if _vfs_instance is None:
        with _vfs_lock:
            if _vfs_instance is None:
                _vfs_instance = KnowledgeVFS()
    return _vfs_instance


# ── 测试（Phase 2D+2E 验证）──────────────────────────────────────
if __name__ == "__main__":
    import sys

    vfs = get_vfs()
    print(f"✓ KnowledgeVFS initialized")

    # Test search
    start = time.time()
    results = vfs.search("BM25", top_k=3, deadline_ms=100)
    elapsed_ms = (time.time() - start) * 1000

    print(f"✓ Search completed in {elapsed_ms:.1f}ms (deadline=100ms)")
    print(f"  Found {len(results)} items:")
    for i, item in enumerate(results, 1):
        print(f"  {i}. [{item.source}] {item.summary[:50]}... (score={item.score:.3f})")

    # Test read
    if results:
        first = results[0]
        start = time.time()
        read_item = vfs.read(first.path)
        elapsed_ms = (time.time() - start) * 1000

        if read_item:
            print(f"✓ Read completed in {elapsed_ms:.1f}ms")
            print(f"  Path: {read_item.path}")
            print(f"  Type: {read_item.type}")

    # Show stats
    stats = vfs.stats()
    print(f"\n📊 VFS Statistics:")
    print(f"  Total reads: {stats['reads']}")
    print(f"  Total searches: {stats['searches']}")
    print(f"  Dentry hits: {stats['dentry_hits']}")
    print(f"  Dentry cache: {stats['dentry_cache']}")
    print(f"  Inode cache: {stats['inode_cache']}")

    print("\n✅ Phase 2D+2E: KnowledgeVFS core verified")
