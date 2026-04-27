#!/usr/bin/env python3
"""
vfs_adapter_openai.py — OpenAI Assistants API VFS 适配器（示例）

OS 类比：NFS 网络文件系统驱动
  NFS 通过实现 file_operations 接口将远程文件系统挂载到 VFS，
  对上层 VFS 完全透明。OpenAI Assistants 同理：
  通过实现 VFSBackend ABC 将 OpenAI 的 file_search 能力挂载到 AIOS VFS，
  KnowledgeVFS.search() 无需知道底层是 SQLite 还是 OpenAI。

使用方式（需要 openai 包 + API key）：
  from vfs_adapter_openai import OpenAIAssistantsBackend
  from vfs_adapter_registry import VFSAdapterRegistry

  backend = OpenAIAssistantsBackend(
      api_key="sk-...",
      assistant_id="asst_...",
      vector_store_id="vs_...",   # 可选
  )
  VFSAdapterRegistry.register("openai-assistants", backend, priority=60)

注意：此文件是示例实现，演示 VFSBackend ABC 如何对接 HTTP API。
     生产环境使用需要实际的 OpenAI API key 和 Assistant 配置。
     不依赖 openai 包即可导入（懒导入），未安装时 search() 返回空列表。
"""
import json
from typing import List, Optional
from datetime import datetime, timezone

from vfs_core import (
    VFSBackend, VFSItem, VFSMetadata,
    VFSItemType, VFSSource, VFSScope,
)


class OpenAIAssistantsBackend(VFSBackend):
    """
    OpenAI Assistants API file_search 适配器。

    OS 类比：NFS 客户端驱动 — 将远程 RPC 调用映射到 VFS 操作。
    通过 OpenAI Assistants API 执行 file_search，结果转换为 VFSItem。

    特性：
    - 懒加载 openai 包（未安装时 search() 优雅降级返回空列表）
    - 线程安全（客户端实例无内部状态，每次调用独立）
    - 支持 vector_store_id 或 assistant_id 两种查询模式
    """

    def __init__(
        self,
        api_key: str,
        assistant_id: Optional[str] = None,
        vector_store_id: Optional[str] = None,
        timeout_secs: float = 10.0,
        base_url: Optional[str] = None,
    ):
        """
        初始化 OpenAI Assistants 后端。

        参数：
          api_key         — OpenAI API key（必填）
          assistant_id    — Assistant ID（assistant 查询模式）
          vector_store_id — Vector Store ID（直接向量搜索模式，更快）
          timeout_secs    — HTTP 超时秒数（默认 10s）
          base_url        — 自定义 API base URL（用于 proxy/兼容 API）
        """
        self._api_key = api_key
        self._assistant_id = assistant_id
        self._vector_store_id = vector_store_id
        self._timeout_secs = timeout_secs
        self._base_url = base_url
        self._client = None  # 懒加载

    def _get_client(self):
        """懒加载 OpenAI 客户端（避免在未安装时导入失败）。"""
        if self._client is None:
            try:
                import openai
                kwargs = {"api_key": self._api_key, "timeout": self._timeout_secs}
                if self._base_url:
                    kwargs["base_url"] = self._base_url
                self._client = openai.OpenAI(**kwargs)
            except ImportError:
                return None
        return self._client

    @property
    def name(self) -> str:
        return "openai-assistants"

    @property
    def source_type(self) -> str:
        return VFSSource.EXTERNAL.value if hasattr(VFSSource, "EXTERNAL") else "external"

    def read(self, path: str) -> Optional[VFSItem]:
        """
        按路径读取文件（此适配器不支持路径直接读取）。

        OS 类比：NFS open() — 远程文件需要先 lookup 才能 open。
        OpenAI Assistants 不支持直接按路径读取，返回 None。
        """
        return None

    def search(self, query: str, top_k: int = 5) -> List[VFSItem]:
        """
        通过 OpenAI Assistants file_search 搜索。

        OS 类比：NFS readdir + lookup — 遍历远程目录查找匹配项。

        策略：
        1. 优先使用 vector_store_id（直接向量搜索，低延迟）
        2. 降级到 assistant_id（通过 Run 接口，较高延迟）
        3. openai 包未安装时返回空列表（优雅降级）

        返回：VFSItem 列表，score 来自 OpenAI 的相关度分数
        """
        client = self._get_client()
        if client is None:
            return []  # openai 包未安装，优雅降级

        try:
            if self._vector_store_id:
                return self._search_vector_store(client, query, top_k)
            elif self._assistant_id:
                return self._search_via_assistant(client, query, top_k)
            return []
        except Exception:
            return []  # 网络错误 / API 错误，优雅降级

    def _search_vector_store(self, client, query: str, top_k: int) -> List[VFSItem]:
        """直接查询 Vector Store（Beta API）。"""
        try:
            results = client.beta.vector_stores.file_search(
                vector_store_id=self._vector_store_id,
                query=query,
                max_num_results=top_k,
            )
            items = []
            for r in results.data:
                item = VFSItem(
                    path=f"openai://vs/{self._vector_store_id}/{r.id}",
                    content=r.content[0].text if r.content else "",
                    summary=r.content[0].text[:120] if r.content else "",
                    source=self.source_type,
                    score=r.score if hasattr(r, "score") else 0.5,
                    item_type=VFSItemType.DOCUMENT,
                    metadata=VFSMetadata(
                        created_at=datetime.now(timezone.utc).isoformat(),
                        tags=["openai", "vector_store"],
                        scope=VFSScope.GLOBAL,
                        extra={"file_id": r.id},
                    ),
                )
                items.append(item)
            return items
        except Exception:
            return []

    def _search_via_assistant(self, client, query: str, top_k: int) -> List[VFSItem]:
        """
        通过 Assistant Run 搜索（较高延迟，但更准确）。

        注意：此模式会消耗 token，延迟较高（通常 2-5s）。
             生产环境建议使用 vector_store_id 直接搜索。
        """
        try:
            # 创建临时 thread 和 message
            thread = client.beta.threads.create()
            client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=f"Search for: {query}",
            )
            # 创建 run（同步等待）
            run = client.beta.threads.runs.create_and_poll(
                thread_id=thread.id,
                assistant_id=self._assistant_id,
                max_prompt_tokens=1000,
                max_completion_tokens=500,
            )
            if run.status != "completed":
                return []

            # 读取 assistant 回复
            messages = client.beta.threads.messages.list(thread_id=thread.id)
            for msg in messages.data:
                if msg.role == "assistant":
                    for content_block in msg.content:
                        if content_block.type == "text":
                            text = content_block.text.value
                            item = VFSItem(
                                path=f"openai://assistant/{self._assistant_id}/{msg.id}",
                                content=text,
                                summary=text[:120],
                                source=self.source_type,
                                score=0.7,  # Assistant 搜索无显式分数
                                item_type=VFSItemType.DOCUMENT,
                                metadata=VFSMetadata(
                                    created_at=datetime.now(timezone.utc).isoformat(),
                                    tags=["openai", "assistant"],
                                    scope=VFSScope.GLOBAL,
                                    extra={"run_id": run.id},
                                ),
                            )
                            return [item]
            return []
        except Exception:
            return []

    def write(self, item: VFSItem) -> bool:
        """写入（此适配器不支持写入）。"""
        return False

    def delete(self, path: str) -> bool:
        """删除（此适配器不支持删除）。"""
        return False
