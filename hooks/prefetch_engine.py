#!/usr/bin/env python3
"""
prefetch_engine.py — PreTool Context-Aware Prefetch（硬件预取类比）

OS 类比：Intel HW Prefetcher + Linux readahead(2)

背景：
  CPU 硬件预取器（Haswell+ Next-Line Prefetcher, Stride Prefetcher）：
    - 观察到连续的 cache miss pattern（如顺序读 arr[0], arr[1], arr[2]）
    - 提前将后续数据加载到 L2/L3 cache，消除 cache miss 延迟
    - 核心思路：不等 miss 发生，预测 miss 提前 fetch

  Linux readahead(2) / page_cache_readahead_unbounded()：
    - 顺序读文件时，内核自动预取后续 pages（readahead window）
    - struct file_ra_state 追踪预取状态：prev_pos, size, async_size
    - PreTool ≈ "即将发生的 access pattern" 信号

  memory-os 映射：
    PreToolUse event  ≈ prefetch trigger signal（预测即将需要哪类知识）
    tool_name         ≈ memory access pattern（不同 tool 暗示不同知识需求）
    prefetch_engine   ≈ hardware prefetcher（后台异步预取到 working set）
    working_set.put() ≈ L2 cache fill（预取完成，数据就位）
    PostTool query    ≈ actual memory access（此时直接 working set hit）

Tool → 知识域映射（Pattern Table，类比 HW Prefetcher 历史表）：
  Edit / Write / MultiEdit → code / project（代码修改需要决策历史和架构约束）
  Bash                     → rule / code（命令执行需要操作约束和工具规范）
  Read / Glob              → project（文件读取暗示在理解项目结构）
  WebFetch / WebSearch     → general（网络查询通常需要通用背景知识）
  Agent                    → project / rule（启动子 agent 需要项目约束）

预取流程（时序）：
  T=0ms   PreTool hook 触发，tool_name 已知
  T=1ms   prefetch_engine 异步启动（non-blocking）
  T=0ms   PreTool hook 立即返回（不阻塞主路径）
  T=50ms  (后台) scatter_gather_route 完成，chunks 填入 working_set
  T=200ms 主对话工具返回，PostTool hook 触发
  T=200ms query_with_working_set → working_set HIT（节省 ~5ms store.db 查询）

关键设计：
  1. 非阻塞：asyncio + fire-and-forget，PreTool hook 立即返回
  2. 幂等：同一 (session_id, tool_name, query_hash) 不重复预取
  3. 容量控制：预取不超过 working_set.max_chunks 的 20%
  4. 自适应：预取命中率低时降低预取强度（类比 Intel LLC Late Prefetch Penalty）

与 extractor_pool 的区别：
  extractor_pool: PostTool → 写路径异步化（Stop hook 提交 extract_task）
  prefetch_engine: PreTool → 读路径预加载（PreTool hook 触发预取）
  两者都是 "hook → async worker" 模式，但方向相反（写 vs 读）。
"""
from __future__ import annotations

import sys
import os
import json
import time
import zlib
import threading
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
from config import get as _sysctl

# ── 常量 ──────────────────────────────────────────────────────────────────────
MEMORY_OS_DIR = Path.home() / ".claude" / "memory-os"
PREFETCH_LOG = MEMORY_OS_DIR / "prefetch.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [prefetch] %(message)s",
    handlers=[logging.FileHandler(str(PREFETCH_LOG), encoding="utf-8")],
)
_log = logging.getLogger("prefetch")


# ─────────────────────────────────────────────────────────────
# Tool → Domain 映射表（Pattern Table）
# OS 类比：HW Prefetcher 的 Stream Detector + Stride Prefetcher 历史表
# ─────────────────────────────────────────────────────────────

# (domains, priority_boost)
# domains: 预取时优先检索的知识域列表
# priority_boost: 预取强度系数（1.0=正常，>1 加大预取量）
_TOOL_PATTERN_TABLE: dict = {
    # 写操作：最需要代码决策和架构约束
    "Edit":       (["code", "project"], 1.5),
    "MultiEdit":  (["code", "project"], 1.5),
    "Write":      (["code", "project"], 1.2),

    # 命令执行：需要操作规范和命令约束
    "Bash":       (["rule", "code"],    1.2),

    # 读操作：理解项目结构
    "Read":       (["project", "code"], 0.8),
    "Glob":       (["project"],         0.6),
    "Grep":       (["code", "project"], 0.8),

    # 网络查询：通用背景知识
    "WebFetch":   (["general"],         0.7),
    "WebSearch":  (["general"],         0.7),

    # 子 Agent：项目约束和规则
    "Agent":      (["project", "rule"], 1.0),

    # 默认（未匹配 tool）
    "__default__": (["general"],         0.5),
}

# 工具输入字段提取（用于构建更精准的 prefetch query）
_TOOL_INPUT_QUERY_FIELDS: dict = {
    "Edit":      ["file_path", "old_string"],
    "MultiEdit": ["file_path"],
    "Write":     ["file_path"],
    "Bash":      ["command"],
    "Read":      ["file_path"],
    "Glob":      ["pattern"],
    "Grep":      ["pattern", "path"],
    "Agent":     ["description", "prompt"],
}


def _extract_prefetch_query(tool_name: str, tool_input: dict) -> str:
    """
    从 tool_input 提取预取 query。

    OS 类比：HW Stride Prefetcher 从访问地址序列推算下一个 stride：
      addr[t], addr[t+1], addr[t+2] → stride = addr[t+1] - addr[t]
      预取目标 = addr[t+2] + stride

    这里从 tool 参数推算"即将需要什么知识"：
      Edit file_path=store_vfs.py → 预取"store_vfs 相关决策和约束"
      Bash command="git push" → 预取"git 操作规范"
    """
    fields = _TOOL_INPUT_QUERY_FIELDS.get(tool_name, [])
    parts = []
    for f in fields:
        val = tool_input.get(f, "")
        if val and isinstance(val, str):
            # 只取有意义的片段（文件名、关键词），避免过长
            val = val.strip()[:100]
            # 文件路径只取 basename 和父目录名
            if f in ("file_path",):
                val = Path(val).name + " " + Path(val).parent.name
            parts.append(val)

    base_query = " ".join(parts).strip()
    if not base_query:
        base_query = tool_name  # 最少以 tool 名为 query

    return base_query[:200]


# ─────────────────────────────────────────────────────────────
# 预取任务去重（避免同一 session 重复预取相同内容）
# OS 类比：HW Prefetcher 的 prefetch queue + duplicate filter
# ─────────────────────────────────────────────────────────────

class _PrefetchDeduplicator:
    """
    预取去重器。
    使用 (session_id, query_hash) 防止在同一 session 内对相同内容重复预取。

    OS 类比：HW Prefetcher 内部的 prefetch queue（PQ），
    防止同一 cache line 被重复 prefetch（PQ deduplication）。
    """
    def __init__(self, max_size: int = 100):
        self._seen: set = set()
        self._max = max_size
        self._lock = threading.Lock()

    def is_dup(self, session_id: str, query: str) -> bool:
        key = f"{session_id}:{zlib.crc32(query.encode()) & 0xffffffff:08x}"
        with self._lock:
            if key in self._seen:
                return True
            if len(self._seen) >= self._max:
                # 随机清除一半（LRU 太贵，随机足够用）
                self._seen = set(list(self._seen)[self._max // 2:])
            self._seen.add(key)
            return False


_dedup = _PrefetchDeduplicator()


# ─────────────────────────────────────────────────────────────
# 预取统计（类比 /proc/vmstat nr_prefetch_hit/miss）
# ─────────────────────────────────────────────────────────────

@dataclass
class _PrefetchStats:
    """全局预取统计。"""
    total_prefetches: int = 0
    hits: int = 0       # 预取后 query 命中 working_set 的次数
    misses: int = 0     # 预取后 query 仍然 miss 的次数
    errors: int = 0
    total_chunks: int = 0
    total_ms: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def hit(self) -> None:
        with self._lock:
            self.hits += 1

    def miss(self) -> None:
        with self._lock:
            self.misses += 1

    def record(self, chunks: int, ms: float, error: bool = False) -> None:
        with self._lock:
            self.total_prefetches += 1
            if error:
                self.errors += 1
            else:
                self.total_chunks += chunks
                self.total_ms += ms

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "total_prefetches": self.total_prefetches,
                "hits": self.hits,
                "misses": self.misses,
                "errors": self.errors,
                "total_chunks": self.total_chunks,
                "avg_ms": round(self.total_ms / max(1, self.total_prefetches - self.errors), 1),
                "hit_rate_pct": round(self.hit_rate(), 1),
            }


_stats = _PrefetchStats()


# ─────────────────────────────────────────────────────────────
# 自适应预取强度（Late Prefetch Penalty 类比）
# ─────────────────────────────────────────────────────────────

class _AdaptivePrefetchController:
    """
    自适应预取控制器。

    OS 类比：Intel CPU 的 Adaptive Prefetch：
      当 LLC（Last Level Cache）miss 率高时降低预取强度（避免带宽浪费）；
      命中率高时增加预取深度（积极预取）。
      实现：每 N 次操作评估命中率，动态调整 prefetch_enabled 和 depth。

    这里：
      hit_rate < 20% → 降低预取强度（scale=0.5，减少预取数量）
      hit_rate 20-60% → 正常（scale=1.0）
      hit_rate > 60% → 增加预取（scale=1.5）
    """
    def __init__(self):
        self._scale = 1.0
        self._eval_interval = 20  # 每 20 次操作重新评估
        self._op_count = 0
        self._lock = threading.Lock()

    def scale(self) -> float:
        with self._lock:
            return self._scale

    def update(self) -> None:
        with self._lock:
            self._op_count += 1
            if self._op_count % self._eval_interval != 0:
                return
            hr = _stats.hit_rate()
            if hr < 20.0:
                self._scale = max(0.3, self._scale * 0.8)  # 降低
            elif hr > 60.0:
                self._scale = min(2.0, self._scale * 1.1)  # 提升
            else:
                self._scale = 1.0  # 重置到正常

    def effective_top_k(self, base_k: int) -> int:
        return max(1, int(base_k * self.scale()))


_adaptive = _AdaptivePrefetchController()


# ─────────────────────────────────────────────────────────────
# 核心预取函数（后台线程执行）
# ─────────────────────────────────────────────────────────────

def _do_prefetch(
    session_id: str,
    project: str,
    tool_name: str,
    query: str,
    domains: list,
    top_k: int,
) -> None:
    """
    实际预取执行（在后台线程中运行，不阻塞主路径）。

    OS 类比：__do_page_cache_readahead() — 内核 readahead 实际 I/O 函数：
      1. 调用 add_to_page_cache_lru()（对应 working_set.put()）
      2. 提交 bio（对应 scatter_gather_route()）
      3. 等待 I/O 完成（对应 gather 阶段）
    """
    t0 = time.monotonic()
    chunks_loaded = 0
    error = False

    try:
        from agent_working_set import registry
        from hooks.knowledge_router import scatter_gather_route
        from store import open_db, ensure_schema
        from schema import MemoryChunk

        ws = registry.get_or_create(session_id, project)

        conn = open_db()
        ensure_schema(conn)

        for domain in domains:
            if chunks_loaded >= top_k:
                break

            try:
                sg = scatter_gather_route(
                    query=query,
                    project=project,
                    timeout_ms=_sysctl("prefetch.timeout_ms"),
                    conn=conn,
                    domain=domain,
                )
                for r in sg["results"]:
                    if chunks_loaded >= top_k:
                        break
                    if r.get("source") == "memory_os" and r.get("summary"):
                        # 从 store.db 取完整 chunk
                        row = conn.execute(
                            "SELECT * FROM memory_chunks "
                            "WHERE summary=? AND project=? LIMIT 1",
                            (r["summary"][:120], project)
                        ).fetchone()
                        if row:
                            chunk = MemoryChunk.from_dict(dict(row))
                            ws.put(chunk, dirty=False)  # 预取 = clean（只读副本）
                            chunks_loaded += 1
            except Exception as e:
                _log.debug("scatter_gather error domain=%s: %s", domain, e)

        conn.close()

    except Exception as e:
        error = True
        _log.warning("prefetch error session=%s tool=%s: %s", session_id, tool_name, e)

    elapsed = round((time.monotonic() - t0) * 1000, 1)
    _stats.record(chunks_loaded, elapsed, error)
    _adaptive.update()

    if not error and chunks_loaded > 0:
        _log.info("prefetch done session=%s tool=%s domain=%s chunks=%d elapsed=%.1fms",
                  session_id[:8], tool_name, domains[0], chunks_loaded, elapsed)


# ─────────────────────────────────────────────────────────────
# 公开接口：PreTool hook 入口
# ─────────────────────────────────────────────────────────────

def trigger_prefetch(
    session_id: str,
    project: str,
    tool_name: str,
    tool_input: dict,
) -> bool:
    """
    PreTool hook 调用此函数触发异步预取。

    设计约束：
      - 必须立即返回（非阻塞），不能增加 PreTool hook 延迟
      - OS 类比：queue_work(prefetch_wq, &ra->work) — 提交到 workqueue 后立即返回

    返回 True 表示预取任务已提交，False 表示跳过（disabled/dedup）。
    """
    if not _sysctl("prefetch.enabled"):
        return False

    # 从 pattern table 获取域配置
    domains, priority_boost = _TOOL_PATTERN_TABLE.get(
        tool_name, _TOOL_PATTERN_TABLE["__default__"]
    )

    # 构建 query
    query = _extract_prefetch_query(tool_name, tool_input)

    # 去重（同 session 相同内容不重复预取）
    if _dedup.is_dup(session_id, f"{tool_name}:{query}"):
        return False

    # 计算预取数量（adaptive scaling）
    base_k = _sysctl("prefetch.max_chunks")
    effective_k = _adaptive.effective_top_k(int(base_k * priority_boost))

    # 后台线程 fire-and-forget（类比 async_schedule()）
    t = threading.Thread(
        target=_do_prefetch,
        args=(session_id, project, tool_name, query, domains, effective_k),
        name=f"prefetch-{tool_name}-{session_id[:6]}",
        daemon=True,  # daemon=True 确保主进程退出时不等待
    )
    t.start()
    return True


def record_query_outcome(session_id: str, query: str,
                         was_cache_hit: bool) -> None:
    """
    PostTool 查询结果反馈给预取控制器。

    OS 类比：cpu_cache_miss() / cpu_cache_hit() — 让 prefetcher 学习
    这个函数应该在 query_with_working_set() 后调用，用于自适应调整。
    """
    if was_cache_hit:
        _stats.hit()
    else:
        _stats.miss()
    _adaptive.update()


def get_prefetch_stats() -> dict:
    """预取统计（/proc/vmstat nr_prefetch_* 类比）。"""
    return {
        **_stats.to_dict(),
        "adaptive_scale": round(_adaptive.scale(), 2),
    }


# ─────────────────────────────────────────────────────────────
# PreTool hook main（独立调用时的入口）
# ─────────────────────────────────────────────────────────────

def main():
    """
    作为 PreTool hook 独立运行时的入口。

    Claude Code 会将 hook_input JSON 写入 stdin：
      {
        "session_id": "...",
        "tool_name": "Edit",
        "tool_input": {"file_path": "...", ...},
        ...
      }

    hook 要求：立即退出（exit 0），不阻塞工具执行。
    """
    import sys
    try:
        raw = sys.stdin.read(512 * 1024)
        hook_input = json.loads(raw) if raw.strip() else {}
    except Exception:
        hook_input = {}

    if not hook_input:
        sys.exit(0)

    tool_name  = hook_input.get("tool_name", "")
    tool_input = hook_input.get("tool_input", {}) or {}
    session_id = hook_input.get("session_id", "")

    if not tool_name or not session_id:
        sys.exit(0)

    # 推导 project
    try:
        from utils import resolve_project_id
        project = resolve_project_id()
    except Exception:
        project = "unknown"

    # 触发异步预取（非阻塞）
    trigger_prefetch(session_id, project, tool_name, tool_input)

    # 立即退出，不等待预取完成
    sys.exit(0)


if __name__ == "__main__":
    # 测试模式
    import json as _json

    print("=== Prefetch Engine 测试 ===")

    # 模拟 tool 触发
    test_cases = [
        ("Edit",    {"file_path": "store_vfs.py", "old_string": "def insert_chunk"}),
        ("Bash",    {"command": "git push origin main"}),
        ("Read",    {"file_path": "hooks/knowledge_router.py"}),
        ("Agent",   {"description": "代码审查", "prompt": "review memory-os arch"}),
    ]

    for tool_name, tool_input in test_cases:
        q = _extract_prefetch_query(tool_name, tool_input)
        domains, boost = _TOOL_PATTERN_TABLE.get(tool_name, _TOOL_PATTERN_TABLE["__default__"])
        effective_k = _adaptive.effective_top_k(int(_sysctl("prefetch.max_chunks") * boost))
        print(f"  {tool_name:12s} → domain={domains}, k={effective_k}, query='{q[:50]}'")

    print()
    print("Pattern Table 完整性：", list(_TOOL_PATTERN_TABLE.keys()))
    print("Adaptive scale:", _adaptive.scale())
    print("Stats:", _json.dumps(_stats.to_dict(), ensure_ascii=False))
