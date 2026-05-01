#!/usr/bin/env python3
"""
memory_lookup MCP Server — 推理中断模型（Inference Interrupt Model）

迭代 99：AI 主动缺页中断（Active Page Fault）

OS 类比：
  传统 batch I/O（polling）：进程定期检查设备状态（busy wait）
  中断模型（interrupt）：设备就绪时主动通知 CPU（interrupt → ISR → resume）

  memory-os 演进路径：
    v1（轮询）：UserPromptSubmit hook → batch inject top-K chunks → 推理开始
      问题：固定 top-K 无法按推理中途发现的新需求动态调整
      类比：进程开始前读取所有可能需要的数据（预读 + 批处理）

    v2（中断）：AI 推理中途感知到知识缺口 → 调用 memory_lookup() → 按需注入
      优势：
        1. 查询粒度更细（"SCX_ENQ_IMMED 的约束" vs 宽泛的初始 prompt）
        2. 多轮检索（发现 A → A 引导发现 B）
        3. 零冗余（只注入真正需要的知识）
      类比：demand paging（需要时才加载页面，而不是启动时全部加载）

使用方式：
  AI 在推理时遇到知识缺口时，直接调用此 MCP 工具：
    memory_lookup("SCX_ENQ_IMMED 约束")
    memory_lookup("checkpoint_restore 返回格式")
    memory_lookup("BM25 scorer 参数", top_k=5)

  工具返回格式化的检索结果，AI 继续推理。

架构：
  FastMCP stdio server → 通过 MCP 协议与 Claude Code 通信
  检索管道：fts_search → retrieval_score → format → return
  DB 连接：只读连接（避免与 writer 锁竞争）
"""

import sys
import os
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

# FastMCP — Model Context Protocol Python SDK
from mcp.server.fastmcp import FastMCP

# ── AIOS memory-os 路径 ───────────────────────────────────────────────────────
_MOS_ROOT = Path(__file__).resolve().parent
if str(_MOS_ROOT) not in sys.path:
    sys.path.insert(0, str(_MOS_ROOT))

from store_vfs import fts_search, open_db, pin_chunk, unpin_chunk, is_pinned, get_pinned_chunks, ensure_schema
from scorer import retrieval_score, recency_score
from utils import resolve_project_id

# ── MCP Server 初始化 ────────────────────────────────────────────────────────
mcp = FastMCP(
    name="memory-os",
    instructions=(
        "AIOS Memory OS — 主动知识检索工具。\n"
        "当你在推理过程中发现需要查找特定知识时（不确定某个决策/设计/约束），\n"
        "调用 memory_lookup 按需检索。这是推理中断模型（demand paging）：\n"
        "不需要在推理前预载所有知识，在需要时精确查询即可。\n\n"
        "适用场景：\n"
        "  - '我不确定这个函数的参数格式' → memory_lookup('函数名 参数')\n"
        "  - '上次关于这个模块的决策是什么' → memory_lookup('模块名 决策')\n"
        "  - '这里有什么设计约束' → memory_lookup('约束 限制', chunk_types=['design_constraint'])\n"
    )
)

# ── DB 连接（只读，避免锁竞争）──────────────────────────────────────────────
def _open_readonly() -> sqlite3.Connection:
    """打开只读 DB 连接（类比 O_RDONLY，零写锁竞争）"""
    db_path = Path.home() / ".claude" / "memory-os" / "store.db"
    env_path = os.environ.get("MEMORY_OS_DB")
    if env_path:
        db_path = Path(env_path)
    if not db_path.exists():
        raise FileNotFoundError(f"store.db not found: {db_path}")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _format_chunk(chunk: dict, rank: int) -> str:
    """格式化单个 chunk 为可读文本（注入格式与 retriever 保持一致）"""
    chunk_type = chunk.get("chunk_type", "")
    summary = chunk.get("summary", "").strip()
    content = chunk.get("content", "").strip()
    importance = chunk.get("importance", 0.5)

    # chunk_type 符号
    type_icons = {
        "decision": "💡",
        "reasoning_chain": "🔗",
        "design_constraint": "⚠️",
        "prompt_context": "📋",
        "code_snippet": "💻",
        "task_state": "📌",
        "quantitative_evidence": "📊",
        "causal_chain": "🔗",
    }
    icon = type_icons.get(chunk_type, "📎")
    if chunk_type == "semantic_memory":
        icon = "🌐"  # 语义记忆：跨项目通用知识

    # 语义层来源标记
    semantic_tag = " [跨项目语义记忆]" if chunk.get("_from_semantic_layer") else ""
    lines = [f"{icon} [{rank}] [{chunk_type}]{semantic_tag} (importance={importance:.2f})"]
    lines.append(f"  {summary}")
    if content and len(content) < 500:
        lines.append(f"  ---")
        lines.append(f"  {content[:400]}{'...' if len(content) > 400 else ''}")

    return "\n".join(lines)


# ── MCP 工具定义 ─────────────────────────────────────────────────────────────

@mcp.tool()
def memory_lookup(
    query: str,
    top_k: int = 5,
    chunk_types: list[str] | None = None,
    project: str | None = None,
) -> str:
    """
    在 AIOS 知识库中主动检索相关记忆（推理中断 / demand paging）。

    当你推理时发现需要某类知识但不确定时，调用此工具：
    - 上次关于某模块的决策
    - 某功能的已知设计约束
    - 某段代码的参数格式
    - 之前总结的性能数据

    Args:
        query: 查询字符串（自然语言或关键词均可）
        top_k: 返回结果数量（默认 5）
        chunk_types: 可选，过滤 chunk 类型（如 ["design_constraint", "decision"]）
        project: 可选，指定项目 ID（默认自动推断）

    Returns:
        格式化的检索结果，包含 summary 和 content 摘要
    """
    if not query or not query.strip():
        return "❌ 查询为空，请提供检索关键词。"

    # 推断项目 ID
    if not project:
        try:
            project = resolve_project_id()
        except Exception:
            project = "default"

    try:
        conn = _open_readonly()
    except FileNotFoundError as e:
        return f"❌ 知识库未初始化：{e}"

    try:
        # ── FTS5 检索 ─────────────────────────────────────────────────────────
        ct_tuple = tuple(chunk_types) if chunk_types else None
        candidates = fts_search(conn, query, project, top_k=top_k * 3, chunk_types=ct_tuple)

        if not candidates:
            # fallback：也尝试 global 层
            global_candidates = fts_search(conn, query, "global", top_k=top_k, chunk_types=ct_tuple)
            candidates = global_candidates

        # ── 语义记忆层激活（跨项目通用知识，__semantic__ project）──────────────
        # OS 类比：TLB 命中后补充 shared memory page — 语义记忆跨 project 共享，
        # 任何查询都自动激活相关的通用知识，不受 project 边界限制。
        _SEMANTIC_PROJECT = "__semantic__"
        try:
            semantic_ct = tuple(["semantic_memory"]) if not chunk_types else ct_tuple
            semantic_candidates = fts_search(
                conn, query, _SEMANTIC_PROJECT,
                top_k=max(2, top_k // 2),
                chunk_types=semantic_ct,
            )
            if semantic_candidates:
                # 标记来源，避免与 project 内 chunk 混淆
                for c in semantic_candidates:
                    c["_from_semantic_layer"] = True
                candidates = candidates + semantic_candidates
        except Exception:
            pass  # 语义层激活失败不影响主检索

        if not candidates:
            return f"💭 未找到与 '{query}' 相关的记忆。\n  提示：知识库可能还没有这方面的内容，或查询词可以换个角度。"

        # ── 评分排序 ────────────────────────────────────────────────────────
        scored = []
        for c in candidates:
            fts_rank = c.get("fts_rank", 0.0) or 0.0
            score = retrieval_score(
                relevance=min(fts_rank / 10.0, 1.0),  # normalize fts_rank
                importance=c.get("importance", 0.5),
                last_accessed=c.get("last_accessed", ""),
                created_at=c.get("created_at", ""),
                access_count=c.get("access_count", 0),
            )
            scored.append((score, c))

        # design_constraint 优先（类比 mlock 保护页优先服务缺页中断）
        scored.sort(key=lambda x: (
            -(1 if x[1].get("chunk_type") == "design_constraint" else 0),
            -x[0]
        ))

        top_results = scored[:top_k]

        # ── 格式化输出 ────────────────────────────────────────────────────────
        lines = [f"🔍 memory_lookup: '{query}' → {len(top_results)} 条结果\n"]

        # 分离约束和普通知识（类比 retriever 的强制注入逻辑）
        constraints = [(s, c) for s, c in top_results if c.get("chunk_type") == "design_constraint"]
        others = [(s, c) for s, c in top_results if c.get("chunk_type") != "design_constraint"]

        if constraints:
            lines.append("【已知约束（系统级设计限制）】")
            for i, (score, c) in enumerate(constraints, 1):
                lines.append(_format_chunk(c, i))
                lines.append(f"  (score={score:.3f})")
            lines.append("")

        if others:
            lines.append("【相关知识】")
            offset = len(constraints)
            for i, (score, c) in enumerate(others, 1):
                lines.append(_format_chunk(c, offset + i))
                lines.append(f"  (score={score:.3f})")
                lines.append("")

        return "\n".join(lines)

    except Exception as e:
        return f"❌ 检索失败：{type(e).__name__}: {e}"
    finally:
        conn.close()


@mcp.tool()
def memory_stats(project: str | None = None) -> str:
    """
    查询 AIOS 知识库的统计信息（chunk 数量、类型分布、近期活跃度）。

    Returns:
        知识库统计摘要
    """
    if not project:
        try:
            project = resolve_project_id()
        except Exception:
            project = "default"

    try:
        conn = _open_readonly()
    except FileNotFoundError as e:
        return f"❌ 知识库未初始化：{e}"

    try:
        # 总量
        total = conn.execute(
            "SELECT COUNT(*) FROM memory_chunks WHERE project=?", (project,)
        ).fetchone()[0]

        # 类型分布
        type_rows = conn.execute(
            "SELECT chunk_type, COUNT(*) FROM memory_chunks WHERE project=? GROUP BY chunk_type",
            (project,)
        ).fetchall()

        # 近期活跃 (access_count > 0)
        active = conn.execute(
            "SELECT COUNT(*) FROM memory_chunks WHERE project=? AND access_count > 0",
            (project,)
        ).fetchone()[0]

        # 最近写入
        recent = conn.execute(
            """SELECT id, chunk_type, summary FROM memory_chunks
               WHERE project=? ORDER BY created_at DESC LIMIT 3""",
            (project,)
        ).fetchall()

        lines = [f"📊 Memory OS 知识库统计 (project={project})\n"]
        lines.append(f"  总量: {total} chunks，活跃: {active} ({active/total*100:.1f}% 被引用)" if total > 0 else "  总量: 0 chunks")
        lines.append("")
        lines.append("  类型分布:")
        for row in sorted(type_rows, key=lambda r: -r[1]):
            lines.append(f"    {row[0]:25s}: {row[1]}")
        lines.append("")
        lines.append("  最近写入:")
        for row in recent:
            lines.append(f"    [{row[1]}] {row[2][:80]}")

        return "\n".join(lines)

    except Exception as e:
        return f"❌ 统计失败：{type(e).__name__}: {e}"
    finally:
        conn.close()


# ── 迭代104：pin/unpin MCP 工具（OS 类比：mlock/munlock per-VMA）──────────────

def _open_readwrite() -> sqlite3.Connection:
    """打开读写 DB 连接（pin/unpin 需要写权限）"""
    db_path = Path.home() / ".claude" / "memory-os" / "store.db"
    env_path = os.environ.get("MEMORY_OS_DB")
    if env_path:
        db_path = Path(env_path)
    if not db_path.exists():
        raise FileNotFoundError(f"知识库不存在：{db_path}")
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


@mcp.tool()
def pin_memory(
    chunk_id: str,
    pin_type: str = "soft",
    project: str | None = None,
) -> str:
    """
    将指定 chunk 锁定到当前项目，阻止被自动淘汰。
    OS 类比：mlock(addr, len) — 将页面锁定在进程地址空间，阻止 swap out。

    pin_type:
      'hard' — 所有淘汰路径跳过该 chunk（stale reclaim、DAMON DEAD、kswapd ZONE_MIN）
               适用：设计约束、不可变决策、关键架构知识
      'soft' — 保护 stale reclaim 和 DAMON DEAD，但 kswapd 内存极度紧张时仍可淘汰
               适用：重要但非关键的量化证据、近期决策

    Args:
        chunk_id: 要锁定的 chunk ID（可从 memory_lookup 结果中获取）
        pin_type: 'hard' 或 'soft'（默认 'soft'）
        project: 项目 ID（默认自动解析当前目录）

    Returns:
        操作结果描述
    """
    if not project:
        try:
            project = resolve_project_id()
        except Exception:
            project = "default"

    if pin_type not in ("hard", "soft"):
        return f"❌ pin_type 必须是 'hard' 或 'soft'，得到：{pin_type!r}"

    try:
        conn = _open_readwrite()
    except FileNotFoundError as e:
        return f"❌ 知识库未初始化：{e}"

    try:
        ensure_schema(conn)
        success = pin_chunk(conn, chunk_id, project, pin_type)
        conn.commit()
        if success:
            return (
                f"✅ chunk {chunk_id[:12]}... 已 {pin_type} pin 到 project={project}\n"
                f"  {'🔒 hard pin: 所有淘汰路径均跳过' if pin_type == 'hard' else '🔐 soft pin: 保护 stale/DAMON，不挡 kswapd 硬淘汰'}"
            )
        else:
            return f"❌ chunk {chunk_id[:12]}... 不存在（pin 失败）"
    except Exception as e:
        return f"❌ pin 失败：{type(e).__name__}: {e}"
    finally:
        conn.close()


@mcp.tool()
def unpin_memory(
    chunk_id: str,
    project: str | None = None,
) -> str:
    """
    解除 chunk 在当前项目中的 pin，允许被自动淘汰。
    OS 类比：munlock(addr, len) — 解除内存锁定，页面重新可被 swap out。

    Args:
        chunk_id: 要解锁的 chunk ID
        project: 项目 ID（默认自动解析当前目录）

    Returns:
        操作结果描述
    """
    if not project:
        try:
            project = resolve_project_id()
        except Exception:
            project = "default"

    try:
        conn = _open_readwrite()
    except FileNotFoundError as e:
        return f"❌ 知识库未初始化：{e}"

    try:
        ensure_schema(conn)
        success = unpin_chunk(conn, chunk_id, project)
        conn.commit()
        if success:
            return f"✅ chunk {chunk_id[:12]}... pin 已解除 (project={project})"
        else:
            return f"⚠️ chunk {chunk_id[:12]}... 在 project={project} 中未被 pin"
    except Exception as e:
        return f"❌ unpin 失败：{type(e).__name__}: {e}"
    finally:
        conn.close()


@mcp.tool()
def list_pinned(
    pin_type: str | None = None,
    project: str | None = None,
) -> str:
    """
    列出当前项目中所有 pinned chunks。
    OS 类比：/proc/[pid]/smaps 中 Locked: 字段 — 查看进程的 mlock 区域。

    Args:
        pin_type: 过滤类型 'hard'/'soft'（默认显示全部）
        project: 项目 ID（默认自动解析当前目录）

    Returns:
        格式化的 pinned chunk 列表
    """
    if not project:
        try:
            project = resolve_project_id()
        except Exception:
            project = "default"

    try:
        conn = _open_readonly()
    except FileNotFoundError as e:
        return f"❌ 知识库未初始化：{e}"

    try:
        pinned = get_pinned_chunks(conn, project, pin_type=pin_type)
        if not pinned:
            label = f" ({pin_type} pin)" if pin_type else ""
            return f"📌 project={project} 没有 pinned chunks{label}"

        lines = [f"📌 Pinned chunks (project={project}" + (f", type={pin_type}" if pin_type else "") + f") — {len(pinned)} 条\n"]
        for i, c in enumerate(pinned, 1):
            icon = "🔒" if c["pin_type"] == "hard" else "🔐"
            lines.append(
                f"  {i}. {icon}[{c['pin_type']}] {c['chunk_type']:20s} imp={c['importance']:.2f}\n"
                f"     ID: {c['chunk_id'][:16]}...\n"
                f"     {c['summary'][:100]}"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"❌ 查询失败：{type(e).__name__}: {e}"
    finally:
        conn.close()


# ── 入口 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
