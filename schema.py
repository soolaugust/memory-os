"""
memory-os 统一数据结构
所有层（L2-L5）共用 MemoryChunk，保证接口一致，后续迭代只填充实现不改接口。

迭代300：info_class — 三层路由（借鉴 GBrain brain-vs-memory 分层）
  world      : 关于外部世界的事实（人/项目/技术/决策）
  operational: 关于 agent 自身操作的配置（偏好/规则/工具设置）
  ephemeral  : 当前会话临时状态，低优先保留

迭代301：stability — Ebbinghaus 遗忘曲线记忆稳定性
  每次被检索命中时 stability *= 2.0（间隔重复加固）
  eviction 优先驱逐 stability 低且 age 大的 chunk
  初始值由 importance 决定（0.5 importance → stability=1.0 天）
"""
from dataclasses import dataclass, field, asdict
from typing import Optional
import uuid
import json
from datetime import datetime, timezone


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class MemoryChunk:
    # 元信息
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    project: str = ""
    source_session: str = ""

    # 分类
    chunk_type: str = "task_state"  # "task_state" | "decision" | "discussion" | "excluded_path"
    # 迭代300：三层路由 — world / operational / ephemeral
    info_class: str = "world"       # "world" | "operational" | "ephemeral"

    # 内容
    content: str = ""          # 完整正文
    summary: str = ""          # ≤100字摘要，用于 L2 注入
    # 迭代306：写入时保真 — 保留原始片段，读取时 on-demand 附加原文（≤500字）
    raw_snippet: str = ""      # 原始提取文本片段，不参与 FTS5 索引

    # 标签与权重
    tags: list = field(default_factory=list)
    importance: float = 0.5    # 0-1，置换算法使用
    retrievability: float = 0.5  # 0-1，越低越不能换出（不可重建的推理状态）

    # 访问记录（LRU）
    last_accessed: str = field(default_factory=_now_iso)
    # 迭代301：Ebbinghaus 记忆稳定性（单位：天）
    # 初始值 = importance * 2.0；每次检索命中 *= 2.0（间隔重复加固）
    stability: float = 1.0

    # L3 向量（MVP 阶段为空列表，迭代 2 填充）
    embedding: list = field(default_factory=list)

    # 可选外部链接
    feishu_url: Optional[str] = None

    # 迭代315：编码情境（Encoding Specificity, Tulving 1973）
    # 存储 chunk 写入时的情境特征，检索时与 query_context 比对匹配
    encoding_context: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryChunk":
        return cls(
            id=d.get("id", str(uuid.uuid4())),
            created_at=d.get("created_at", _now_iso()),
            updated_at=d.get("updated_at", _now_iso()),
            project=d.get("project", ""),
            source_session=d.get("source_session", ""),
            chunk_type=d.get("chunk_type", "task_state"),
            info_class=d.get("info_class", "world"),
            content=d.get("content", ""),
            summary=d.get("summary", ""),
            raw_snippet=d.get("raw_snippet", ""),
            tags=d.get("tags", []),
            importance=d.get("importance", 0.5),
            retrievability=d.get("retrievability", 0.5),
            last_accessed=d.get("last_accessed", _now_iso()),
            stability=d.get("stability", 1.0),
            embedding=d.get("embedding", []),
            feishu_url=d.get("feishu_url"),
            encoding_context=d.get("encoding_context", {}),
        )

    @classmethod
    def from_json(cls, s: str) -> "MemoryChunk":
        return cls.from_dict(json.loads(s))
