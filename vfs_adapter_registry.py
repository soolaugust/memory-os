#!/usr/bin/env python3
"""
vfs_adapter_registry.py — VFS 外部适配器注册中心

OS 类比：Linux /dev 字符设备注册（register_chrdev / cdev_add）
  内核驱动通过 register_chrdev_region() 注册到全局设备表，
  用户空间通过 /dev/xxx 访问，不需要知道具体驱动实现。
  VFS 适配器注册同理：第三方 LLM 框架实现 VFSBackend ABC，
  通过 VFSAdapterRegistry.register() 注册到全局注册表，
  KnowledgeVFS.search() 自动发现并并行查询，无需修改核心代码。

使用方式：
  # 注册外部 backend（如 OpenAI Assistants）
  from vfs_adapter_registry import VFSAdapterRegistry
  from vfs_adapter_openai import OpenAIAssistantsBackend

  backend = OpenAIAssistantsBackend(api_key="...", assistant_id="...")
  VFSAdapterRegistry.register("openai-assistants", backend, priority=60)

  # KnowledgeVFS 自动使用注册的 backend
  from vfs import KnowledgeVFS
  vfs = KnowledgeVFS()
  results = vfs.search("query")  # 自动包含 openai-assistants 后端

架构约束：
  - 注册表是进程级单例（不跨进程）
  - backend 实例必须是线程安全的
  - priority 值越高，权重越高（0-100，默认 50）
  - 注册/注销在 _registry_lock 保护下进行
"""
import threading
from typing import Optional, List, Dict, Any

from vfs_core import VFSBackend


class BackendEntry:
    """注册的 backend 条目（含元数据）。"""

    def __init__(self, name: str, backend: VFSBackend, priority: int = 50):
        self.name = name
        self.backend = backend
        self.priority = max(0, min(100, priority))  # 钳制到 [0, 100]
        self.registered_at = __import__('datetime').datetime.now(
            __import__('datetime').timezone.utc
        ).isoformat()

    def __repr__(self) -> str:
        return (f"BackendEntry(name={self.name!r}, "
                f"priority={self.priority}, "
                f"source_type={self.backend.source_type!r})")


class VFSAdapterRegistry:
    """
    全局 VFS 适配器注册中心（线程安全单例）。

    OS 类比：Linux 字符设备注册表（chrdev_map/cdev_map）
      chrdev_map 是内核全局哈希表，key=major number，value=驱动操作集。
      VFSAdapterRegistry 是进程全局字典，key=backend 名称，value=BackendEntry。
    """

    _registry: Dict[str, BackendEntry] = {}
    _registry_lock = threading.Lock()

    @classmethod
    def register(cls, name: str, backend: VFSBackend, priority: int = 50) -> None:
        """
        注册外部 backend。

        OS 类比：register_chrdev_region() — 向内核注册字符设备驱动。

        参数：
          name     — 唯一标识符（如 "openai-assistants", "weaviate", "pinecone"）
          backend  — 实现了 VFSBackend ABC 的实例
          priority — 搜索权重（0-100，默认 50；越高越优先）

        若同名已存在，则覆盖（相当于驱动模块重新加载）。
        """
        if not isinstance(backend, VFSBackend):
            raise TypeError(
                f"backend must be a VFSBackend instance, got {type(backend).__name__}"
            )
        entry = BackendEntry(name, backend, priority)
        with cls._registry_lock:
            cls._registry[name] = entry

    @classmethod
    def unregister(cls, name: str) -> bool:
        """
        注销 backend。

        OS 类比：unregister_chrdev_region() — 驱动卸载时注销。

        返回：True = 成功注销，False = 名称不存在
        """
        with cls._registry_lock:
            if name in cls._registry:
                del cls._registry[name]
                return True
            return False

    @classmethod
    def get_backend(cls, name: str) -> Optional[VFSBackend]:
        """
        按名获取 backend 实例。

        OS 类比：lookup_bdev() — 按路径查找块设备。

        返回：VFSBackend 实例，或 None（不存在）
        """
        with cls._registry_lock:
            entry = cls._registry.get(name)
            return entry.backend if entry else None

    @classmethod
    def get_entry(cls, name: str) -> Optional[BackendEntry]:
        """按名获取完整的 BackendEntry（含 priority 等元数据）。"""
        with cls._registry_lock:
            return cls._registry.get(name)

    @classmethod
    def list_backends(cls) -> List[BackendEntry]:
        """
        列出所有注册的 backends，按 priority 降序排列。

        OS 类比：/proc/devices — 列出所有注册的字符/块设备。

        返回：BackendEntry 列表（降序优先级）
        """
        with cls._registry_lock:
            entries = list(cls._registry.values())
        return sorted(entries, key=lambda e: e.priority, reverse=True)

    @classmethod
    def list_names(cls) -> List[str]:
        """返回所有已注册的 backend 名称列表。"""
        with cls._registry_lock:
            return list(cls._registry.keys())

    @classmethod
    def clear(cls) -> int:
        """
        清空注册表（仅用于测试）。

        OS 类比：rmmod — 卸载所有驱动。

        返回：清除的条目数
        """
        with cls._registry_lock:
            count = len(cls._registry)
            cls._registry.clear()
            return count

    @classmethod
    def source_weight(cls, name: str) -> float:
        """
        计算 backend 的搜索权重（将 priority 映射到 [0.5, 1.0] 区间）。

        priority 0  → 0.5（最低，与未知 source 同等）
        priority 50 → 0.75（中等）
        priority 100 → 1.0（最高，与 memory-os 同等）
        """
        entry = cls.get_entry(name)
        if entry is None:
            return 0.5
        # 线性映射：priority [0, 100] → weight [0.5, 1.0]
        return 0.5 + (entry.priority / 100.0) * 0.5
