#!/usr/bin/env python3
"""
KnowledgeVFS — 统一知识虚拟文件系统（fs/ — Phase 2）

对应 OS 历史节点：Unix VFS (1988) 和 Linux VFS (1996)
- Unix VFS：Linux 采纳 4.4BSD VFS，统一访问不同文件系统（ext2、NFS 等）
- Linux VFS：inode 和 dentry 缓存层，支持任意文件系统后端

KnowledgeVFS 的理念：
  三套存储后端（SQLite、Filesystem、Project）就像三套不同的文件系统，
  统一 VFS 层让上层应用（retriever/loader/writer）无需知道底层差异。

核心设计：
  1. VFSItem — 统一的知识表示（对应 VFS inode）
  2. KnowledgeVFS — 核心路由接口（对应 VFS switch_table）
  3. Backend 适配器 — 三套后端的实现类
  4. 缓存系统 — dentry cache + inode cache（两级缓存）
  5. 垂直路由 — 路径→后端的自动映射

路径格式：/<source>/<id>
  - /memory-os/chunk-uuid — SQLite 后端的 chunk
  - /memory-md/reference-key — MEMORY.md 的索引条
  - /self-improving/filepath — self-improving 的文件
  - /project/history-key — 项目级 JSONL

缓存分层：
  L1 dentry cache — (path, metadata) 快速查找（<0.1ms，进程内）
  L2 inode cache — (id, full_content) 完整数据（<1ms，PCID 跨进程）
  L3 后端存储 — 冷路径（10-100ms）

API 超时：100ms hard deadline（与 retriever 对齐）
"""

import sys
import time
import sqlite3
import json
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple, Union
from pathlib import Path
from enum import Enum
from datetime import datetime, timezone

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from config import get as _sysctl
from bm25 import bm25_normalized as _bm25_norm


# ─────────────────────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────────────────────

class VFSItemType(Enum):
    """知识项类型（对应 chunk_type + 跨系统类型）"""
    DECISION = "decision"
    RULE = "rule"
    TRACE = "trace"
    REFERENCE = "reference"
    FEEDBACK = "feedback"
    REASONING_CHAIN = "reasoning_chain"
    TASK_STATE = "task_state"
    CONVERSATION_SUMMARY = "conversation_summary"
    PROMPT_CONTEXT = "prompt_context"
    EXCLUDED_PATH = "excluded_path"
    TOOL_INSIGHT = "tool_insight"


class VFSSource(Enum):
    """知识来源"""
    MEMORY_OS = "memory-os"
    MEMORY_MD = "memory-md"
    SELF_IMPROVING = "self-improving"
    PROJECT = "project"


@dataclass
class VFSMetadata:
    """VFS 元数据（对应 inode 属性）"""
    created_at: str  # ISO 8601
    updated_at: str
    importance: int = 0  # 0-10 scale
    scope: str = "session"  # session | project | global
    source: str = ""  # 完整来源路径
    tags: List[str] = field(default_factory=list)
    retrievability: float = 1.0  # 可检索性 0-1
    mtime: float = 0.0  # 源文件 mtime（PCID 校验）
    hash: str = ""  # 内容哈希（变更检测）


@dataclass
class VFSItem:
    """
    统一知识表示（对应 VFS inode）
    """
    id: str  # UUID 或路径哈希
    type: VFSItemType
    content: str  # 完整内容
    summary: str = ""  # 摘要（<120 字符）
    source: VFSSource = VFSSource.MEMORY_OS
    metadata: Optional[VFSMetadata] = None
    score: float = 0.0  # 检索时的相关度分数
    path: str = ""  # 虚拟路径 /<source>/<id>

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "summary": self.summary,
            "source": self.source.value,
            "metadata": asdict(self.metadata) if self.metadata else {},
            "score": round(self.score, 4),
            "path": self.path,
        }

    @staticmethod
    def from_dict(d: dict) -> "VFSItem":
        """从字典反序列化"""
        meta = d.get("metadata", {})
        metadata = VFSMetadata(**meta) if meta else None
        return VFSItem(
            id=d["id"],
            type=VFSItemType(d["type"]),
            content=d["content"],
            summary=d.get("summary", ""),
            source=VFSSource(d.get("source", "memory-os")),
            metadata=metadata,
            score=d.get("score", 0.0),
            path=d.get("path", ""),
        )


# ─────────────────────────────────────────────────────────────
# 缓存层（L1 dentry + L2 inode）
# ─────────────────────────────────────────────────────────────

class VFSCache:
    """
    两级缓存系统
    L1: dentry cache — 元数据快速查找（进程内 TTL）
    L2: inode cache — 完整内容缓存（PCID 跨进程）
    """

    def __init__(self, ttl_secs: int = 300):
        self.ttl = ttl_secs
        self._dentry_cache: Dict[str, Tuple[float, VFSItem]] = {}
        self._inode_cache: Dict[str, Tuple[float, VFSItem]] = {}

    def dentry_get(self, path: str) -> Optional[VFSItem]:
        """L1 快速查找"""
        entry = self._dentry_cache.get(path)
        if entry and (time.time() - entry[0]) < self.ttl:
            return entry[1]
        if entry:
            del self._dentry_cache[path]
        return None

    def dentry_set(self, path: str, item: VFSItem):
        """L1 写入"""
        self._dentry_cache[path] = (time.time(), item)

    def inode_get(self, inode_id: str) -> Optional[VFSItem]:
        """L2 内容缓存查找"""
        entry = self._inode_cache.get(inode_id)
        if entry and (time.time() - entry[0]) < self.ttl:
            return entry[1]
        if entry:
            del self._inode_cache[inode_id]
        return None

    def inode_set(self, inode_id: str, item: VFSItem):
        """L2 写入"""
        self._inode_cache[inode_id] = (time.time(), item)

    def invalidate(self, scope: str = "all"):
        """缓存失效：all | dentry | inode"""
        if scope in ("all", "dentry"):
            self._dentry_cache.clear()
        if scope in ("all", "inode"):
            self._inode_cache.clear()


# ─────────────────────────────────────────────────────────────
# 后端适配器接口
# ─────────────────────────────────────────────────────────────

class VFSBackend(ABC):
    """
    VFS 后端适配器（对应 file_system_type）
    每个后端必须实现统一的读写搜索接口
    """

    @abstractmethod
    def name(self) -> str:
        """后端名称"""
        pass

    @abstractmethod
    def read(self, path: str, recursive: bool = False) -> List[VFSItem]:
        """读取路径下的项目"""
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 3, timeout_ms: int = 100) -> List[VFSItem]:
        """全文搜索"""
        pass

    @abstractmethod
    def write(self, item: VFSItem) -> str:
        """写入项目，返回新 ID"""
        pass

    @abstractmethod
    def delete(self, item_id: str, force: bool = False) -> bool:
        """删除项目"""
        pass

    @abstractmethod
    def invalidate_cache(self):
        """手动失效缓存"""
        pass


# ─────────────────────────────────────────────────────────────
# 核心 KnowledgeVFS 类
# ─────────────────────────────────────────────────────────────

class KnowledgeVFS:
    """
    统一知识虚拟文件系统
    OS 类比：Linux VFS 的 super_block 角色
    """

    def __init__(self, backends: Dict[str, VFSBackend], cache: Optional[VFSCache] = None):
        """
        参数：
          backends — {source_name: backend_instance}
          cache — 缓存系统（默认创建）
        """
        self.backends = backends
        self.cache = cache or VFSCache(ttl_secs=_sysctl("router.cache_ttl_secs"))
        self._route_cache: Dict[str, str] = {}  # 路径→后端映射缓存

    def _parse_path(self, path: str) -> Tuple[str, str]:
        """
        解析虚拟路径 /<source>/<id> → (source, id)
        """
        if not path.startswith("/"):
            path = "/" + path
        parts = path.strip("/").split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid VFS path: {path}")
        return parts[0], parts[1]

    def _get_backend(self, source: str) -> VFSBackend:
        """
        根据源名称获取后端
        OS 类比：inode→superblock 查找
        """
        if source not in self.backends:
            raise KeyError(f"Unknown VFS source: {source}")
        return self.backends[source]

    def read(self, path: str, recursive: bool = False, timeout_ms: int = 100) -> List[VFSItem]:
        """
        读取 VFS 路径下的项目
        """
        try:
            source, item_id = self._parse_path(path)
            backend = self._get_backend(source)

            # L1 dentry cache 检查
            cached = self.cache.dentry_get(path)
            if cached:
                return [cached]

            # 后端读取
            start = time.time()
            results = backend.read(path, recursive=recursive)
            elapsed_ms = (time.time() - start) * 1000

            if elapsed_ms > timeout_ms:
                raise TimeoutError(f"VFS read exceeded {timeout_ms}ms")

            # 缓存结果
            for item in results:
                self.cache.dentry_set(item.path, item)

            return results
        except Exception as e:
            print(f"VFS read error: {e}")
            return []

    def search(
        self,
        query: str,
        sources: Optional[List[str]] = None,
        top_k: int = 3,
        timeout_ms: int = 100,
    ) -> List[VFSItem]:
        """
        跨后端全文搜索
        OS 类比：read() syscall 对多个文件系统的路由
        """
        if sources is None:
            sources = list(self.backends.keys())

        all_results: List[VFSItem] = []
        start_time = time.time()

        # 源权重（归一化不同后端的分数）
        SOURCE_WEIGHT = {
            "memory-os": 1.0,
            "memory-md": 0.8,
            "self-improving": 0.7,
            "project": 0.6,
        }

        for source in sources:
            # 超时检查
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > timeout_ms * 0.9:  # 留 10% 缓冲
                break

            try:
                backend = self._get_backend(source)
                remaining_ms = timeout_ms - int(elapsed_ms)

                results = backend.search(query, top_k=top_k, timeout_ms=remaining_ms)

                # 应用源权重
                weight = SOURCE_WEIGHT.get(source, 0.7)
                for r in results:
                    r.score = round(r.score * weight, 4)

                all_results.extend(results)
            except Exception as e:
                print(f"VFS search error in {source}: {e}")
                continue

        # 全局去重（同 summary 保留最高分）
        seen: Dict[str, VFSItem] = {}
        for r in all_results:
            key = r.summary.lower().strip()
            if key not in seen or r.score > seen[key].score:
                seen[key] = r

        # 排序并返回 Top-K
        deduped = sorted(seen.values(), key=lambda x: x.score, reverse=True)
        return deduped[:top_k * len(sources)]

    def write(self, item: VFSItem, scope: str = "session") -> str:
        """
        写入项目到适当的后端
        返回新 ID
        """
        backend = self._get_backend(item.source.value)
        new_id = backend.write(item)

        # 更新项目 path 和缓存
        item.id = new_id
        item.path = f"/{item.source.value}/{new_id}"
        self.cache.inode_set(new_id, item)

        return new_id

    def delete(self, path: str, force: bool = False) -> bool:
        """
        删除项目
        """
        try:
            source, item_id = self._parse_path(path)
            backend = self._get_backend(source)

            # 后端删除
            result = backend.delete(item_id, force=force)

            # 失效缓存
            if result:
                self.cache.dentry_get(path)
                self.cache.invalidate("dentry")

            return result
        except Exception as e:
            print(f"VFS delete error: {e}")
            return False

    def invalidate_cache(self, scope: str = "all"):
        """
        手动失效缓存
        scope: all | dentry | inode
        """
        self.cache.invalidate(scope)
        for backend in self.backends.values():
            try:
                backend.invalidate_cache()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────
# 全局单例
# ─────────────────────────────────────────────────────────────

_vfs_instance: Optional[KnowledgeVFS] = None


def init_vfs(backends: Dict[str, VFSBackend]) -> KnowledgeVFS:
    """初始化全局 VFS 实例"""
    global _vfs_instance
    _vfs_instance = KnowledgeVFS(backends)
    return _vfs_instance


def get_vfs() -> KnowledgeVFS:
    """获取全局 VFS 实例（需要先调用 init_vfs）"""
    if _vfs_instance is None:
        raise RuntimeError("VFS not initialized. Call init_vfs() first.")
    return _vfs_instance


if __name__ == "__main__":
    print("KnowledgeVFS module loaded")
