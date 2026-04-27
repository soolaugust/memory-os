#!/usr/bin/env python3
"""
KnowledgeVFS 初始化模块

负责：
1. 实例化所有后端适配器
2. 初始化全局 VFS 实例
3. 提供高级 API（兼容 knowledge_router 接口）
"""

import sys
from pathlib import Path
from typing import Optional, List

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from knowledge_vfs import KnowledgeVFS, get_vfs, init_vfs as _init_vfs
from knowledge_vfs_backends import SQLiteBackend, FilesystemBackend, ProjectBackend


def init_knowledge_vfs() -> KnowledgeVFS:
    """
    初始化全局 KnowledgeVFS 实例

    创建三个后端适配器并注册到 VFS
    返回初始化后的 VFS 实例
    """
    backends = {
        "memory-os": SQLiteBackend(),
        "memory-md": SQLiteBackend(),  # 临时：由 SQLiteBackend 处理 memory-md
        "self-improving": FilesystemBackend(),
        "project": ProjectBackend(),
    }

    vfs = _init_vfs(backends)
    return vfs


# ─────────────────────────────────────────────────────────────
# 高级 API（兼容 knowledge_router 接口）
# ─────────────────────────────────────────────────────────────

def search(
    query: str,
    sources: Optional[List[str]] = None,
    top_k: int = 3,
    timeout_ms: int = 100,
) -> List[dict]:
    """
    统一搜索接口（兼容 knowledge_router.route）

    返回格式与 knowledge_router 兼容：
      [{
        "source": "memory-os",
        "chunk_type": "decision",
        "summary": "...",
        "score": 0.95,
        "content": "...",
        "path": "..."
      }, ...]
    """
    vfs = get_vfs()

    if sources is None:
        sources = ["memory-os", "self-improving", "project"]

    # VFS 搜索
    items = vfs.search(query, sources=sources, top_k=top_k, timeout_ms=timeout_ms)

    # 转换为 knowledge_router 兼容格式
    results = []
    for item in items:
        results.append({
            "source": item.source.value,
            "chunk_type": item.type.value,
            "summary": item.summary,
            "score": item.score,
            "content": item.content[:300] if item.content else "",
            "path": item.path,
        })

    return results


def read(path: str) -> Optional[dict]:
    """
    读取单个项目
    返回与 VFSItem 兼容的字典格式
    """
    vfs = get_vfs()
    results = vfs.read(path, timeout_ms=100)

    if not results:
        return None

    item = results[0]
    return {
        "id": item.id,
        "type": item.type.value,
        "content": item.content,
        "summary": item.summary,
        "source": item.source.value,
        "path": item.path,
        "metadata": {
            "created_at": item.metadata.created_at if item.metadata else "",
            "updated_at": item.metadata.updated_at if item.metadata else "",
            "importance": item.metadata.importance if item.metadata else 0,
        } if item.metadata else {},
    }


def write(item_dict: dict, source: str = "memory-os") -> str:
    """
    写入项目

    参数：
      item_dict — {type, summary, content, metadata?}
      source — "memory-os" | "self-improving" | "project"

    返回：新项目 ID
    """
    from knowledge_vfs import VFSItem, VFSItemType, VFSSource, VFSMetadata
    from datetime import datetime, timezone

    vfs = get_vfs()

    # 构建 VFSItem
    now = datetime.now(timezone.utc).isoformat()
    metadata = VFSMetadata(
        created_at=now,
        updated_at=now,
        importance=item_dict.get("importance", 0),
        scope=item_dict.get("scope", "session"),
        source=f"/{source}",
        tags=item_dict.get("tags", []),
    )

    item = VFSItem(
        id="",  # 由后端生成
        type=VFSItemType(item_dict.get("type", "trace")),
        content=item_dict.get("content", ""),
        summary=item_dict.get("summary", "")[:120],
        source=VFSSource(source),
        metadata=metadata,
    )

    return vfs.write(item)


def format_for_context(results: List[dict]) -> str:
    """
    将搜索结果格式化为 context 注入文本
    兼容 knowledge_router.format_for_context
    """
    if not results:
        return ""

    _PREFIX = {
        "decision": "[决策]", "excluded_path": "[排除]",
        "reasoning_chain": "[推理]", "rule": "[规则]",
        "reference": "[索引]", "knowledge": "[知识]", "task_state": "",
    }

    lines = ["【知识路由召回】"]
    for r in results:
        prefix = _PREFIX.get(r["chunk_type"], "")
        src_tag = f"({r['source']})"
        line = f"{prefix} {r['summary']} {src_tag}".strip()
        lines.append(f"- {line}")

    return "\n".join(lines)


if __name__ == "__main__":
    vfs = init_knowledge_vfs()
    print(f"KnowledgeVFS initialized: {len(vfs.backends)} backends")
    print(f"Backends: {list(vfs.backends.keys())}")
