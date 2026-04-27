#!/usr/bin/env python3
"""
KnowledgeVFS — 统一知识虚拟文件系统

Phase 2: fs/ — 统一知识虚拟文件系统（对标 Linux VFS 1996）
OS 类比：Linux VFS 提供统一的文件系统抽象，隐藏 ext4/NFS/tmpfs 等细节。
        KnowledgeVFS 类似地统一 memory-os/memory-md/self-improving 三套存储后端。

迭代56：VFS 概念框架 + 路由设计
迭代73-74：Phase 2A-B 实现阶段

Phase 2A: VFSItem + VFSMetadata 定义和序列化
"""
import json
import hashlib
from enum import Enum
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from abc import ABC, abstractmethod


# ── 枚举定义 ──────────────────────────────────────────────────────
class VFSItemType(str, Enum):
    """VFS 项目类型（对标 Linux inode mode 的不同文件类型）"""
    # 决策层知识
    DECISION = "decision"                    # 决策链片段
    EXCLUDED_PATH = "excluded_path"          # 排除路径
    REASONING_CHAIN = "reasoning_chain"      # 推理链
    COMPARISON = "comparison"                # 对比论述

    # 上下文和总结
    CONVERSATION_SUMMARY = "conversation_summary"  # 对话摘要
    PROMPT_CONTEXT = "prompt_context"        # 提示上下文

    # 规则和模式
    PATTERN = "pattern"                      # 提取的模式（迭代94）
    RULE = "rule"                            # 可复用规则
    CORRECTION = "correction"                # 错误纠正记录

    # 工具和系统
    TOOL_INSIGHT = "tool_insight"            # 工具输出洞察
    PERFORMANCE_DATA = "performance_data"    # 性能数据

    # 元数据
    METADATA = "metadata"                    # 元数据项（标签、分类等）
    REFERENCE = "reference"                  # 参考链接


class VFSSource(str, Enum):
    """VFS 数据源（对标 Linux 文件系统类型）"""
    MEMORY_OS = "memory-os"                  # SQLite 主存储
    MEMORY_MD = "memory-md"                  # ~/self-improving/memory.md
    SELF_IMPROVING = "self-improving"        # ~/self-improving/**/*.md
    PROJECT = "project"                      # 项目本地 JSONL 历史
    GLOBAL = "global"                        # 全局跨项目知识


class VFSScope(str, Enum):
    """VFS 项目可见范围（对标 Linux 文件权限 rwx）"""
    SESSION = "session"                      # 当前会话内可见
    PROJECT = "project"                      # 项目内可见
    GLOBAL = "global"                        # 全局（所有项目）跨可见


# ── 数据结构 ──────────────────────────────────────────────────────
@dataclass
class VFSMetadata:
    """VFS 元数据（对标 Linux inode 属性）"""
    # 时间戳（对标 inode ctime/mtime/atime）
    created_at: str                          # ISO 8601，UTC
    updated_at: str                          # ISO 8601，UTC
    last_accessed: str                       # ISO 8601，UTC

    # 重要性和可检索性
    importance: float                        # 0.0-1.0，importance decay 计算结果
    retrievability: float                    # 0.0-1.0，被检索到的难度
    access_count: int                        # 被访问次数（对标 inode 链接计数）

    # 来源和作用域
    source_session: str                      # 源会话 ID（对标 inode owner）
    scope: str                               # "session" | "project" | "global"

    # 标签和分类
    tags: List[str]                          # 标签列表（对标 xattr）
    project: str                             # 所属项目（"global" 表示跨项目）

    # 内容完整性校验（对标 inode checksum）
    content_hash: str                        # SHA-256(content)

    # 额外元数据（可扩展）
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VFSItem:
    """VFS 统一知识项（对标 Linux inode）"""
    # 身份和定位（对标 inode number）
    id: str                                  # UUID 或路径哈希
    type: str                                # VFSItemType
    source: str                              # VFSSource

    # 内容（对标 inode 指向的数据块）
    content: str                             # 完整内容
    summary: str                             # 摘要（< 120 字符，用于 UI 展示）

    # 元数据（对标 inode 属性块）
    metadata: VFSMetadata                    # 完整元数据

    # 虚拟路径（对标 dentry 中的名称）
    path: str                                # 虚拟路径 /<source>/<id>

    # 检索相关度（额外属性，非 Linux 原型）
    score: float                             # 0.0-1.0，BM25 + scorer 结果

    @classmethod
    def from_chunk(cls, chunk_dict: dict, score: float = 1.0) -> "VFSItem":
        """从 memory-os chunk 转换为 VFSItem"""
        chunk_id = chunk_dict.get("id", "unknown")
        chunk_type = chunk_dict.get("chunk_type", "decision")
        project = chunk_dict.get("project", "unknown")

        # 计算内容哈希（对标 ext4 inode checksum）
        content = chunk_dict.get("content", "")
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        # 构建元数据
        metadata = VFSMetadata(
            created_at=chunk_dict.get("created_at", datetime.now(timezone.utc).isoformat()),
            updated_at=chunk_dict.get("updated_at", datetime.now(timezone.utc).isoformat()),
            last_accessed=chunk_dict.get("last_accessed", datetime.now(timezone.utc).isoformat()),
            importance=chunk_dict.get("importance", 0.5),
            retrievability=chunk_dict.get("retrievability", 0.5),
            access_count=chunk_dict.get("access_count", 0),
            source_session=chunk_dict.get("source_session", "unknown"),
            scope=VFSScope.GLOBAL.value if project == "global" else VFSScope.PROJECT.value,
            tags=chunk_dict.get("tags", []) if isinstance(chunk_dict.get("tags"), list)
                  else json.loads(chunk_dict.get("tags", "[]")),
            project=project,
            content_hash=content_hash,
        )

        # 构建虚拟路径
        vfs_path = f"/{VFSSource.MEMORY_OS.value}/{chunk_id}"

        return cls(
            id=chunk_id,
            type=chunk_type,
            source=VFSSource.MEMORY_OS.value,
            content=content,
            summary=chunk_dict.get("summary", ""),
            metadata=metadata,
            path=vfs_path,
            score=score,
        )

    def to_dict(self) -> dict:
        """序列化为 dict"""
        return {
            "id": self.id,
            "type": self.type,
            "source": self.source,
            "content": self.content,
            "summary": self.summary,
            "metadata": asdict(self.metadata),
            "path": self.path,
            "score": self.score,
        }

    def to_json(self) -> str:
        """序列化为 JSON"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "VFSItem":
        """从 dict 反序列化"""
        metadata_dict = data.get("metadata", {})
        metadata = VFSMetadata(
            created_at=metadata_dict.get("created_at", ""),
            updated_at=metadata_dict.get("updated_at", ""),
            last_accessed=metadata_dict.get("last_accessed", ""),
            importance=metadata_dict.get("importance", 0.5),
            retrievability=metadata_dict.get("retrievability", 0.5),
            access_count=metadata_dict.get("access_count", 0),
            source_session=metadata_dict.get("source_session", ""),
            scope=metadata_dict.get("scope", "project"),
            tags=metadata_dict.get("tags", []),
            project=metadata_dict.get("project", ""),
            content_hash=metadata_dict.get("content_hash", ""),
            extra=metadata_dict.get("extra", {}),
        )
        return cls(
            id=data.get("id", ""),
            type=data.get("type", "decision"),
            source=data.get("source", "memory-os"),
            content=data.get("content", ""),
            summary=data.get("summary", ""),
            metadata=metadata,
            path=data.get("path", ""),
            score=data.get("score", 1.0),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "VFSItem":
        """从 JSON 反序列化"""
        data = json.loads(json_str)
        return cls.from_dict(data)


# ── VFS 后端接口（对标 file_system_type）──────────────────────────
class VFSBackend(ABC):
    """VFS 后端抽象接口（对标 Linux file_system_type）

    OS 类比：Linux VFS 定义 file_system_type 结构体，ext4/NFS 等通过实现
             具体的 mount/read/write 等 operations 来接入 VFS。
    """

    @abstractmethod
    def read(self, path: str) -> Optional[VFSItem]:
        """按虚拟路径读单个项（对标 inode_operations.lookup）"""
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[VFSItem]:
        """全文搜索（对标 file_operations.read + inode_operations.lookup）"""
        pass

    @abstractmethod
    def write(self, item: VFSItem) -> bool:
        """写入新项（对标 inode_operations.create）"""
        pass

    @abstractmethod
    def delete(self, path: str) -> bool:
        """删除项（对标 inode_operations.unlink）"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """后端名称"""
        pass

    @property
    @abstractmethod
    def source_type(self) -> str:
        """数据源类型（VFSSource 值）"""
        pass


# ── 测试用例（Phase 2A 验证）──────────────────────────────────────
if __name__ == "__main__":
    # Test VFSMetadata serialization
    meta = VFSMetadata(
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
        last_accessed=datetime.now(timezone.utc).isoformat(),
        importance=0.85,
        retrievability=0.45,
        access_count=5,
        source_session="sess-001",
        scope="project",
        tags=["bm25", "optimization"],
        project="git:abc123",
        content_hash="abc123def456",
    )
    print("✓ VFSMetadata created")

    # Test VFSItem creation from chunk
    chunk_dict = {
        "id": "chunk-001",
        "chunk_type": "decision",
        "content": "使用 BM25 作为检索引擎",
        "summary": "选择 BM25 而非 chromadb",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "last_accessed": datetime.now(timezone.utc).isoformat(),
        "importance": 0.85,
        "retrievability": 0.45,
        "access_count": 5,
        "source_session": "sess-001",
        "tags": json.dumps(["bm25", "decision"]),
        "project": "git:abc123",
    }

    item = VFSItem.from_chunk(chunk_dict, score=0.92)
    print(f"✓ VFSItem created: {item.path}")

    # Test serialization
    json_str = item.to_json()
    print(f"✓ Serialized to JSON ({len(json_str)} bytes)")

    # Test deserialization
    item2 = VFSItem.from_json(json_str)
    assert item2.id == item.id
    assert item2.summary == item.summary
    assert item2.metadata.importance == item.metadata.importance
    print("✓ Deserialization round-trip successful")

    print("\n✅ Phase 2A: VFSItem + VFSMetadata verified")
