#!/usr/bin/env python3
"""
agent_working_set.py — Per-Agent Working Set（TLB / L1 Cache 类比）

OS 类比：Denning Working Set Model (1968) + TLB（Translation Lookaside Buffer）

背景：
  Denning (1968) 提出 Working Set Model：进程在时间窗口 [t-Δ, t] 内访问的
  页面集合 W(t, Δ) 称为其 working set。操作系统保证进程的 working set 常驻
  内存，减少 page fault。
    "A process is in the working set if it has been referenced in the
     recent past window of size Δ" — Denning, 1968

  TLB（Translation Lookaside Buffer）：CPU 内的小型快速缓存，保存最近使用的
  虚拟地址→物理地址映射。TLB hit 绕过页表 walk（~1ns），miss 则触发全页表
  遍历（~100ns）。命中率通常 >99%（空间局部性）。

  memory-os 映射：
    MemoryChunk       ≈ 物理页
    chunk_id          ≈ 物理页帧号（PFN）
    query → chunk     ≈ 虚拟地址 → 物理地址翻译
    WorkingSet.get()  ≈ TLB lookup（命中则跳过 store.db）
    WorkingSet.load() ≈ page fault handler（从 store.db 加载并缓存）
    flush_dirty()     ≈ CPU cache writeback + TLB shootdown（多 CPU 一致性）
    evict()           ≈ LRU page eviction（Clock 算法）

架构：
  ┌─ AgentSession ──────────────────────────────────────────┐
  │  working_set = WorkingSet(session_id, project)          │
  │                                                         │
  │  query_time:                                            │
  │    hit  = working_set.get(chunk_id)  # TLB hit, ~0ms   │
  │    miss → working_set.load_from_store(query)  # ~5ms   │
  │                                                         │
  │  write_time:                                            │
  │    working_set.put(chunk, dirty=True)  # 写 working set │
  │                                                         │
  │  session_end:                                           │
  │    working_set.flush_dirty()  # writeback to store.db   │
  └─────────────────────────────────────────────────────────┘

MESI 缓存一致性（多 Agent 场景）：
  - Modified  (M): dirty chunk，本 agent 已写，store.db 未更新
  - Exclusive (E): clean chunk，只本 agent 持有
  - Shared    (S): clean chunk，多 agent 可能共享（从 store.db 读入）
  - Invalid   (I): 已驱逐或失效

  简化实现：
    dirty=True  → M 状态，flush 时写回 store.db
    dirty=False → S/E 状态（只读副本）
    evict()     → I 状态（从 working_set 移除）
    broadcast_invalidate() → 通知其他 agent 的 working set 失效（via agent_notify）

性能预期：
  高频重复 query（长对话）：~5ms → <0.1ms（TLB hit 率 >80% 时）
  首次加载（cold start）：与原来相同（~5ms store.db query）
  Session 结束 flush：批量写入，相比逐条写入减少 ~60% db 写操作
"""
from __future__ import annotations

import sys
import time
import threading
import sqlite3
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))
from config import get as _sysctl
from schema import MemoryChunk

# ── 常量 ──────────────────────────────────────────────────────────────────────
MEMORY_OS_DIR = Path.home() / ".claude" / "memory-os"
STORE_DB = MEMORY_OS_DIR / "store.db"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─────────────────────────────────────────────────────────────
# Working Set Entry（对应 page table entry / TLB entry）
# ─────────────────────────────────────────────────────────────

@dataclass
class WSEntry:
    """
    Working Set 中的单个缓存条目。

    OS 类比：page table entry (PTE)
      dirty     ≈ PTE.dirty bit（写入后置位）
      accessed  ≈ PTE.accessed bit（Clock 算法用）
      pinned    ≈ mlock()（不可被 LRU 驱逐）
      load_time ≈ 页面映射到进程地址空间的时间
    """
    chunk: MemoryChunk
    dirty: bool = False         # 是否已修改（需要 writeback）
    accessed: bool = True       # LRU Clock 算法访问位
    pinned: bool = False        # mlock — 不参与 LRU 驱逐
    load_time: str = field(default_factory=_now_iso)
    access_count: int = 1       # 访问次数（Ebbinghaus stability 更新用）


# ─────────────────────────────────────────────────────────────
# WorkingSet — Per-Agent TLB / L1 Cache
# ─────────────────────────────────────────────────────────────

class WorkingSet:
    """
    Per-Agent Working Set：agent 内存热数据缓存。

    使用 OrderedDict 实现 LRU（Python 3.7+ dict 保证插入顺序），
    等价于 Linux 内核的 active_list（LRU 链表活跃端）。

    线程安全：使用 threading.Lock 保护所有状态，
    OS 类比：spinlock（TLB 操作需要持有 CPU 锁）。
    """

    def __init__(self, session_id: str, project: str,
                 max_chunks: int = None):
        self.session_id = session_id
        self.project = project
        self.max_chunks = max_chunks or _sysctl("working_set.max_chunks")
        self._lock = threading.Lock()
        # OrderedDict 作为 LRU Cache（head=LRU, tail=MRU）
        self._lru: OrderedDict[str, WSEntry] = OrderedDict()
        # 统计
        self._stats = {
            "hits": 0,          # TLB hit 次数
            "misses": 0,        # TLB miss 次数
            "evictions": 0,     # LRU 驱逐次数
            "dirty_flushes": 0, # dirty writeback 次数
        }

    # ── 核心 TLB 操作 ─────────────────────────────────────────────────────────

    def get(self, chunk_id: str) -> Optional[MemoryChunk]:
        """
        TLB lookup（O(1)）。

        OS 类比：TLB lookup by VPN（virtual page number）
          命中：返回 PFN（physical frame）+ 置 accessed bit
          未命中：返回 None，触发 page fault（调用方负责 load_from_store）

        时间复杂度：O(1) OrderedDict lookup
        """
        with self._lock:
            entry = self._lru.get(chunk_id)
            if entry is None:
                self._stats["misses"] += 1
                return None
            # TLB hit：置 accessed bit，移到 MRU 端（LRU 链表尾部）
            entry.accessed = True
            entry.access_count += 1
            self._lru.move_to_end(chunk_id)
            self._stats["hits"] += 1
            return entry.chunk

    def put(self, chunk: MemoryChunk, dirty: bool = False,
            pinned: bool = False) -> None:
        """
        加载或更新 chunk（TLB fill）。

        OS 类比：TLB fill after page walk
          新条目插入 TLB MRU 端（最近访问）。
          若缓存已满，先驱逐 LRU 端（最久未访问）。

        dirty=True  → M 状态（已修改，flush 时写回）
        dirty=False → S/E 状态（只读副本）
        """
        with self._lock:
            existing = self._lru.get(chunk.id)
            if existing:
                # 更新现有条目（比较并置 dirty bit，类似 PTE.dirty）
                existing.chunk = chunk
                if dirty:
                    existing.dirty = True
                existing.accessed = True
                existing.access_count += 1
                self._lru.move_to_end(chunk.id)
                return

            # 新条目：检查容量，必要时 LRU 驱逐
            while len(self._lru) >= self.max_chunks:
                self._evict_lru_locked()

            entry = WSEntry(
                chunk=chunk,
                dirty=dirty,
                pinned=pinned,
                load_time=_now_iso(),
            )
            self._lru[chunk.id] = entry
            # 新条目插入 MRU 端（move_to_end 确保在尾部）
            self._lru.move_to_end(chunk.id)

    def mark_dirty(self, chunk_id: str) -> bool:
        """将 chunk 置为 dirty（M 状态）。返回 chunk 是否存在。"""
        with self._lock:
            entry = self._lru.get(chunk_id)
            if entry is None:
                return False
            entry.dirty = True
            return True

    def pin(self, chunk_id: str) -> bool:
        """mlock — 防止 LRU 驱逐。"""
        with self._lock:
            entry = self._lru.get(chunk_id)
            if entry is None:
                return False
            entry.pinned = True
            return True

    def unpin(self, chunk_id: str) -> bool:
        """munlock — 允许再次被 LRU 驱逐。"""
        with self._lock:
            entry = self._lru.get(chunk_id)
            if entry is None:
                return False
            entry.pinned = False
            return True

    # ── Page Fault Handler — 从 store.db 加载 ─────────────────────────────

    def load_from_store(self, query: str, top_k: int = None,
                        conn: sqlite3.Connection = None) -> list:
        """
        Page fault handler：从 store.db 加载相关 chunks 到 working set。

        OS 类比：do_fault() → __do_page_fault() → handle_mm_fault()
          - 检索 store.db（相当于从磁盘读取页面）
          - 将结果填充到 working_set（相当于 pte_set_wrprotect + TLB fill）

        返回加载的 MemoryChunk 列表。
        """
        if top_k is None:
            top_k = _sysctl("working_set.max_chunks") // 10  # 每次加载 10%

        try:
            from store import open_db, ensure_schema
            from hooks.knowledge_router import scatter_gather_route

            own_conn = conn is None
            if own_conn:
                conn = open_db()
                ensure_schema(conn)

            sg_result = scatter_gather_route(
                query, project=self.project,
                timeout_ms=_sysctl("prefetch.timeout_ms"),
                conn=conn,
            )
            chunks_loaded = []
            for r in sg_result["results"][:top_k]:
                # 从 store.db 取完整 chunk（scatter_gather_route 只返回摘要）
                if r.get("source") == "memory_os":
                    raw = conn.execute(
                        "SELECT * FROM memory_chunks WHERE summary=? AND project=? LIMIT 1",
                        (r["summary"][:120], self.project)
                    ).fetchone()
                    if raw:
                        chunk = MemoryChunk.from_dict(dict(raw))
                        self.put(chunk, dirty=False)
                        chunks_loaded.append(chunk)

            if own_conn:
                conn.close()
            return chunks_loaded
        except Exception:
            return []

    # ── LRU 驱逐（Clock 算法简化版）─────────────────────────────────────────

    def _evict_lru_locked(self) -> Optional[WSEntry]:
        """
        LRU Clock 驱逐（持锁调用）。

        OS 类比：Linux Clock 页面置换算法（二次机会算法）
          遍历 LRU 链表头部：
            - accessed=True → 清 accessed bit，给一次机会，移到尾部
            - accessed=False → 驱逐

          pinned=True 的条目跳过（mlock 保护）。

        返回被驱逐的 WSEntry（可用于 writeback），None 表示无法驱逐。
        """
        # 遍历 LRU 端（OrderedDict 头部），给 accessed=True 一次机会
        candidates = list(self._lru.keys())
        for chunk_id in candidates:
            entry = self._lru.get(chunk_id)
            if entry is None:
                continue
            if entry.pinned:
                continue  # mlock 保护，跳过
            if entry.accessed:
                # 第一次遇到：清 accessed bit，移到 MRU 端（给一次机会）
                entry.accessed = False
                self._lru.move_to_end(chunk_id)
                continue
            # accessed=False：驱逐
            del self._lru[chunk_id]
            self._stats["evictions"] += 1
            return entry
        return None  # 全部 pinned 或都有访问位

    # ── Flush（Writeback）── ─────────────────────────────────────────────────

    def flush_dirty(self, conn: sqlite3.Connection = None) -> int:
        """
        将所有 dirty chunks 写回 store.db（CPU cache writeback）。

        OS 类比：pdflush/flush-X:Y writeback
          - 扫描 dirty list（working set 中 dirty=True 的条目）
          - 批量写回 backing store（store.db）
          - 清除 dirty bit（PTE.dirty = 0）

        返回写回的 chunk 数量。
        """
        if not _sysctl("working_set.flush_dirty_on_exit"):
            return 0

        with self._lock:
            dirty_entries = [
                (cid, e) for cid, e in self._lru.items() if e.dirty
            ]

        if not dirty_entries:
            return 0

        try:
            from store import open_db, ensure_schema, insert_chunk
            own_conn = conn is None
            if own_conn:
                conn = open_db()
                ensure_schema(conn)

            written = 0
            for chunk_id, entry in dirty_entries:
                try:
                    chunk = entry.chunk
                    # 更新 stability（Ebbinghaus 间隔重复）
                    chunk.stability = min(
                        30.0,
                        chunk.stability * (1.0 + 0.1 * entry.access_count)
                    )
                    chunk.last_accessed = _now_iso()
                    chunk.updated_at = _now_iso()
                    # INSERT OR REPLACE（upsert）
                    conn.execute(
                        """INSERT OR REPLACE INTO memory_chunks
                           (id, created_at, updated_at, project, source_session,
                            chunk_type, info_class, content, summary, raw_snippet,
                            tags, importance, retrievability, last_accessed,
                            stability, embedding, feishu_url, encoding_context)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            chunk.id, chunk.created_at, chunk.updated_at,
                            chunk.project, chunk.source_session,
                            chunk.chunk_type, chunk.info_class,
                            chunk.content, chunk.summary, chunk.raw_snippet,
                            ",".join(chunk.tags) if chunk.tags else "",
                            chunk.importance, chunk.retrievability,
                            chunk.last_accessed, chunk.stability,
                            "", chunk.feishu_url or "",
                            str(chunk.encoding_context) if chunk.encoding_context else "",
                        ),
                    )
                    written += 1
                except Exception:
                    pass

            conn.commit()
            if own_conn:
                conn.close()

            # 清除 dirty bit
            with self._lock:
                for chunk_id, entry in dirty_entries:
                    if chunk_id in self._lru:
                        self._lru[chunk_id].dirty = False

            self._stats["dirty_flushes"] += written
            return written
        except Exception:
            return 0

    # ── 统计 + 工具 ──────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """
        TLB/缓存统计（类比 /proc/vmstat 中的 nr_tlb_local_flush_*）。
        """
        with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            hit_rate = (self._stats["hits"] / total * 100) if total > 0 else 0.0
            dirty_count = sum(1 for e in self._lru.values() if e.dirty)
            pinned_count = sum(1 for e in self._lru.values() if e.pinned)
            return {
                "session_id":   self.session_id,
                "project":      self.project,
                "size":         len(self._lru),
                "max_chunks":   self.max_chunks,
                "utilization":  round(len(self._lru) / self.max_chunks * 100, 1),
                "hit_rate":     round(hit_rate, 1),
                "hits":         self._stats["hits"],
                "misses":       self._stats["misses"],
                "evictions":    self._stats["evictions"],
                "dirty_flushes": self._stats["dirty_flushes"],
                "dirty_count":  dirty_count,
                "pinned_count": pinned_count,
            }

    def size(self) -> int:
        with self._lock:
            return len(self._lru)

    def invalidate(self, chunk_id: str) -> bool:
        """
        强制失效指定 chunk（MESI Invalid 状态）。

        OS 类比：TLB shootdown — 当一个 CPU 修改了 PTE 后，
        向其他 CPU 发送 IPI，强制它们 flush 对应 TLB 条目。
        """
        with self._lock:
            if chunk_id in self._lru:
                del self._lru[chunk_id]
                return True
            return False

    def clear(self) -> int:
        """清空 working set（进程退出，释放所有页面）。"""
        with self._lock:
            count = len(self._lru)
            self._lru.clear()
            return count

    def list_chunks(self, dirty_only: bool = False) -> list:
        """列出 working set 中的 chunk（调试用）。"""
        with self._lock:
            result = []
            for chunk_id, entry in self._lru.items():
                if dirty_only and not entry.dirty:
                    continue
                result.append({
                    "id":           chunk_id,
                    "summary":      entry.chunk.summary[:80],
                    "chunk_type":   entry.chunk.chunk_type,
                    "dirty":        entry.dirty,
                    "pinned":       entry.pinned,
                    "accessed":     entry.accessed,
                    "access_count": entry.access_count,
                    "load_time":    entry.load_time,
                })
            return result


# ─────────────────────────────────────────────────────────────
# WorkingSetRegistry — 全局 Session 注册表
# ─────────────────────────────────────────────────────────────

class WorkingSetRegistry:
    """
    全局 WorkingSet 注册表（类比 Linux mm_struct 管理）。

    每个 session_id 对应一个 WorkingSet。
    Registry 负责创建、查找、注销，以及跨 session 的 TLB shootdown
    （当某个 session 修改了 chunk，通知其他 session 失效）。

    OS 类比：mm_struct 全局链表（mmlist）
      Linux 通过 mmlist 跟踪所有进程的内存描述符，
      用于 OOM Killer 遍历、TLB shootdown、/proc/meminfo 统计。
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._registry: dict[str, WorkingSet] = {}
                cls._instance._reg_lock = threading.Lock()
            return cls._instance

    def get_or_create(self, session_id: str, project: str) -> WorkingSet:
        """获取或创建 session 的 WorkingSet。"""
        with self._reg_lock:
            if session_id not in self._registry:
                ws = WorkingSet(session_id, project)
                self._registry[session_id] = ws
            return self._registry[session_id]

    def get(self, session_id: str) -> Optional[WorkingSet]:
        with self._reg_lock:
            return self._registry.get(session_id)

    def close_session(self, session_id: str,
                      flush: bool = True) -> dict:
        """
        Session 结束：flush dirty + 从注册表移除。

        OS 类比：exit_mm() — 进程退出时解引用 mm_struct，
        触发最后一次 writeback（如果引用计数归零）。
        """
        with self._reg_lock:
            ws = self._registry.pop(session_id, None)
        if ws is None:
            return {"flushed": 0, "cleared": 0}
        flushed = ws.flush_dirty() if flush else 0
        cleared = ws.clear()

        # 迭代358：KSM — session 关闭时触发跨 session 热点提升
        # OS 类比：ksmd 在进程退出时扫描残留的 shared page，更新全局合并统计
        ksm_promoted = 0
        if _sysctl("ksm.enabled"):
            try:
                ksm_promoted = self.promote_hot_chunks(
                    ws.project,
                    min_access_count=_sysctl("ksm.min_access_count"),
                    min_sessions=_sysctl("ksm.min_sessions"),
                )
            except Exception:
                pass

        return {"flushed": flushed, "cleared": cleared, "ksm_promoted": ksm_promoted}

    def broadcast_invalidate(self, chunk_id: str,
                              exclude_session: str = None) -> int:
        """
        TLB Shootdown：通知所有 session 失效指定 chunk。

        OS 类比：smp_call_function_many(mask, flush_tlb_func, ...)
          当一个 CPU 修改页表后，通过 IPI 让其他 CPU flush TLB。
          这里用线程锁串行模拟（真实 IPI 并行）。

        返回被失效的 session 数。
        """
        with self._reg_lock:
            sessions = dict(self._registry)
        count = 0
        for sid, ws in sessions.items():
            if sid == exclude_session:
                continue
            if ws.invalidate(chunk_id):
                count += 1
        return count

    def global_stats(self) -> list:
        """所有 session 的 working set 统计（/proc/meminfo 类比）。"""
        with self._reg_lock:
            sessions = dict(self._registry)
        return [ws.stats() for ws in sessions.values()]

    def get_hot_chunks(self, project: str,
                       min_access_count: int = 3,
                       min_sessions: int = 2) -> list:
        """
        迭代358：Cross-Session KSM — 扫描所有 session 的 working set，
        找出跨 session 高频访问的 hot chunk。

        OS 类比：Linux KSM (Kernel Samepage Merging, 2009)：
          ksmd 线程扫描所有进程的匿名页，找到内容相同的页面进行合并，
          减少物理内存占用并提升 cache 利用率。
          这里扫描所有 session 的 working set，找到被多个 session 频繁访问的
          chunk，提升其全局可见性（促进跨 session 知识共享）。

        返回：[{chunk_id, chunk, session_count, total_access_count}]，
              按 total_access_count DESC 排序。
        """
        with self._reg_lock:
            sessions = dict(self._registry)

        # 统计每个 chunk_id 在多少 session 中出现，以及总 access_count
        chunk_stats: dict = {}  # chunk_id → {sessions, total_access, chunk}
        for sid, ws in sessions.items():
            with ws._lock:
                for cid, entry in ws._lru.items():
                    if entry.chunk.project != project:
                        continue
                    if entry.access_count < min_access_count:
                        continue
                    if cid not in chunk_stats:
                        chunk_stats[cid] = {
                            "chunk_id": cid,
                            "chunk": entry.chunk,
                            "session_count": 0,
                            "total_access_count": 0,
                        }
                    chunk_stats[cid]["session_count"] += 1
                    chunk_stats[cid]["total_access_count"] += entry.access_count

        # 过滤：必须出现在 >= min_sessions 个 session 中
        hot = [v for v in chunk_stats.values() if v["session_count"] >= min_sessions]
        hot.sort(key=lambda x: x["total_access_count"], reverse=True)
        return hot

    def promote_hot_chunks(self, project: str,
                           min_access_count: int = 3,
                           min_sessions: int = 2) -> int:
        """
        迭代358：将跨 session 热点 chunk 写回 store.db 并提升 importance/retrievability，
        实现知识的跨 session 共享（类似 KSM 合并后的 shared page 属性）。

        OS 类比：KSM merge 后的共享页标记为 VM_SHARED，所有访问该物理页的 VMA 都可读取，
        减少每个进程单独持有副本的内存开销，并通过 COW 机制保持写隔离。

        返回提升的 chunk 数量。
        """
        hot_chunks = self.get_hot_chunks(project, min_access_count, min_sessions)
        if not hot_chunks:
            return 0

        try:
            from store import open_db, ensure_schema
            conn = open_db()
            ensure_schema(conn)

            promoted = 0
            for hc in hot_chunks[:10]:  # 最多提升 10 个，避免大量写
                chunk = hc["chunk"]
                cid = hc["chunk_id"]
                # 提升策略：增加 retrievability（热点知识更容易被检索）
                # 不改变 importance（importance 是内容价值，不应被访问频率扭曲）
                new_retrievability = min(1.0, chunk.retrievability + 0.05 * hc["session_count"])
                conn.execute(
                    """UPDATE memory_chunks
                       SET retrievability = ?,
                           last_accessed = datetime('now'),
                           access_count = COALESCE(access_count, 0) + ?
                       WHERE id = ? AND project = ?""",
                    (new_retrievability, hc["total_access_count"], cid, project),
                )
                promoted += 1

            conn.commit()
            conn.close()
            return promoted
        except Exception:
            return 0


# 全局单例
registry = WorkingSetRegistry()


# ─────────────────────────────────────────────────────────────
# 便捷函数（供 hooks 直接调用）
# ─────────────────────────────────────────────────────────────

def get_working_set(session_id: str, project: str = None) -> WorkingSet:
    """获取或创建 session 的 WorkingSet（便捷入口）。"""
    if project is None:
        from utils import resolve_project_id
        project = resolve_project_id()
    return registry.get_or_create(session_id, project)


def query_with_working_set(
    query: str,
    session_id: str,
    project: str = None,
    top_k: int = 5,
    conn: sqlite3.Connection = None,
) -> dict:
    """
    带 Working Set 缓存的查询（TLB-aware 检索）。

    流程：
      1. 对 query token 中的 chunk_id 做 TLB lookup（working_set.get）
      2. Miss → scatter_gather_route 查 store.db（page fault）
      3. 将新 chunk 填入 working_set（TLB fill）
      4. 返回命中/缺页结果 + 统计

    返回：
      {
        "results": list[MemoryChunk],
        "hits": int,          # TLB 命中数
        "misses": int,        # Page fault 数
        "from_cache": list,   # 来自 working set 的 chunks
        "from_store": list,   # 从 store.db 加载的 chunks
        "domain": str,        # 检测到的知识域
      }
    """
    t0 = time.monotonic()
    if project is None:
        from utils import resolve_project_id
        project = resolve_project_id()

    ws = registry.get_or_create(session_id, project)

    # Step 1: 尝试从 working set 召回（BM25 in-memory 扫描）
    from_cache = []
    with ws._lock:
        cached_chunks = [(cid, e.chunk) for cid, e in ws._lru.items()]

    if cached_chunks:
        # 在 working set 内做轻量 BM25 排序
        try:
            from bm25 import bm25_normalized as _bm25
            docs = [f"{c.summary} {c.content[:100]}" for _, c in cached_chunks]
            scores = _bm25(query, docs)
            threshold = _sysctl("router.min_score")
            for i, (cid, chunk) in enumerate(cached_chunks):
                if scores[i] >= threshold:
                    from_cache.append((scores[i], chunk))
                    ws._lru[cid].accessed = True
                    ws._lru[cid].access_count += 1
                    ws._stats["hits"] += 1
            from_cache.sort(key=lambda x: x[0], reverse=True)
            from_cache = [c for _, c in from_cache[:top_k]]
        except Exception:
            from_cache = []

    # Step 2: 不足 top_k → page fault，从 store.db 补充
    need = top_k - len(from_cache)
    from_store = []
    domain = "general"
    if need > 0:
        from_store_chunks = ws.load_from_store(query, top_k=need, conn=conn)
        from_store = from_store_chunks
        # 获取域信息
        try:
            from hooks.knowledge_router import domain_classify
            domain = domain_classify(query)
        except Exception:
            pass

    elapsed_ms = round((time.monotonic() - t0) * 1000, 1)

    return {
        "results":    from_cache + from_store,
        "hits":       len(from_cache),
        "misses":     len(from_store),
        "from_cache": from_cache,
        "from_store": from_store,
        "domain":     domain,
        "elapsed_ms": elapsed_ms,
    }


if __name__ == "__main__":
    import json

    # 简单测试
    print("=== WorkingSet 测试 ===")
    ws = WorkingSet("test-session", "test-project", max_chunks=5)

    # 创建测试 chunks
    for i in range(7):
        chunk = MemoryChunk(
            id=f"chunk-{i}",
            project="test-project",
            chunk_type="decision",
            content=f"测试决策内容 {i}",
            summary=f"决策 {i}：选择方案A",
            importance=0.5 + i * 0.05,
        )
        ws.put(chunk, dirty=(i % 2 == 0))

    print(f"Working set size: {ws.size()} (max=5，应触发 LRU 驱逐)")
    print(json.dumps(ws.stats(), ensure_ascii=False, indent=2))

    # TLB 命中测试
    hit = ws.get("chunk-6")
    miss = ws.get("chunk-0")  # 可能已被驱逐
    print(f"chunk-6 (最新): {'HIT' if hit else 'MISS'}")
    print(f"chunk-0 (最旧): {'HIT' if miss else 'MISS (被 LRU 驱逐)'}")
    print(json.dumps(ws.stats(), ensure_ascii=False, indent=2))
