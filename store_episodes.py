"""
store_episodes.py — 情节时间线（iter364）

OS 类比：ftrace ring buffer — 带时间戳的事件流，可以重放
  - 每个 session = 一次进程运行记录
  - session_episodes 表 = /sys/kernel/debug/tracing/trace 文件
  - 情节注入 = strace 重放：新 session 直接看到上次"做了什么"

人的记忆类比：情节记忆（Episodic Memory，Tulving 1972）
  - 语义记忆：稳定事实（"这个服务用 gRPC"） → 现有 memory_chunks
  - 情节记忆：带时间戳的事件（"上周五我在这里改了 X，然后出了 Y 问题"）← 本模块
  - 区别：情节记忆有"我当时"的视角，语义记忆是去个人化的知识

核心用途：
  新 session 进入工作区时，注入"上 N 次 session 的行为摘要"，
  比 CRIU intent（只保留意图）更完整，回答"上次到底干了什么"。
"""
import json
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from store_vfs import open_db, _safe_add_column


# ── Schema ────────────────────────────────────────────────────────────────────

def ensure_episodes_schema(conn: sqlite3.Connection) -> None:
    """幂等 schema — CREATE IF NOT EXISTS。"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS session_episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            workspace_id TEXT,            -- 可为 NULL（无工作区 context 时）
            project TEXT NOT NULL,
            started_at TEXT,
            ended_at TEXT NOT NULL,
            duration_secs REAL DEFAULT 0,
            -- 行为摘要
            summary TEXT NOT NULL,        -- 1-3 句自然语言摘要
            actions_json TEXT,            -- [{type, desc}] 执行的关键操作
            chunks_created INTEGER DEFAULT 0,    -- 本 session 写入的 chunk 数
            files_modified_json TEXT,     -- ["path1", "path2"] 修改的文件列表
            tools_used_json TEXT,         -- {"Bash": 5, "Edit": 3} 工具调用统计
            -- 注入控制
            injected_count INTEGER DEFAULT 0     -- 被注入到新 session 的次数
        )
    """)
    _safe_add_column(conn, "session_episodes", "workspace_id", "TEXT")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ep_workspace ON session_episodes(workspace_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ep_project ON session_episodes(project, ended_at DESC)"
    )
    conn.commit()


# ── Write ─────────────────────────────────────────────────────────────────────

def write_episode(
    conn: sqlite3.Connection,
    session_id: str,
    project: str,
    summary: str,
    *,
    workspace_id: Optional[str] = None,
    started_at: Optional[str] = None,
    ended_at: Optional[str] = None,
    duration_secs: float = 0.0,
    actions: Optional[list] = None,
    chunks_created: int = 0,
    files_modified: Optional[list] = None,
    tools_used: Optional[dict] = None,
) -> int:
    """
    写入一条 session 情节记录，返回 rowid。
    每个 session 调用一次（Stop hook 末尾）。
    同一 session_id 重复调用会 INSERT 新行（一次 session 可有多个片段）。
    """
    ensure_episodes_schema(conn)
    now = datetime.now(timezone.utc).isoformat()
    row = conn.execute("""
        INSERT INTO session_episodes
        (session_id, workspace_id, project, started_at, ended_at,
         duration_secs, summary, actions_json,
         chunks_created, files_modified_json, tools_used_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        session_id,
        workspace_id,
        project,
        started_at or now,
        ended_at or now,
        duration_secs,
        summary[:500],  # 硬截断防止过长
        json.dumps(actions or [], ensure_ascii=False),
        chunks_created,
        json.dumps(files_modified or [], ensure_ascii=False),
        json.dumps(tools_used or {}, ensure_ascii=False),
    ))
    conn.commit()
    return row.lastrowid


# ── Read ──────────────────────────────────────────────────────────────────────

def get_recent_episodes(
    conn: sqlite3.Connection,
    project: str,
    *,
    workspace_id: Optional[str] = None,
    limit: int = 3,
) -> list:
    """
    获取最近 N 条 session 情节，按时间倒序。
    workspace_id 不为空时优先返回该工作区的情节。

    Returns:
        [{"session_id", "ended_at", "summary", "actions", "chunks_created",
          "files_modified", "duration_secs"}, ...]
    """
    ensure_episodes_schema(conn)

    if workspace_id:
        rows = conn.execute("""
            SELECT session_id, ended_at, summary, actions_json,
                   chunks_created, files_modified_json, duration_secs
            FROM session_episodes
            WHERE workspace_id = ?
            ORDER BY ended_at DESC
            LIMIT ?
        """, (workspace_id, limit)).fetchall()
    else:
        rows = conn.execute("""
            SELECT session_id, ended_at, summary, actions_json,
                   chunks_created, files_modified_json, duration_secs
            FROM session_episodes
            WHERE project = ?
            ORDER BY ended_at DESC
            LIMIT ?
        """, (project, limit)).fetchall()

    result = []
    for r in rows:
        try:
            actions = json.loads(r[3]) if r[3] else []
        except (json.JSONDecodeError, TypeError):
            actions = []
        try:
            files = json.loads(r[5]) if r[5] else []
        except (json.JSONDecodeError, TypeError):
            files = []
        result.append({
            "session_id": r[0],
            "ended_at": r[1],
            "summary": r[2],
            "actions": actions,
            "chunks_created": r[4],
            "files_modified": files,
            "duration_secs": r[6] or 0,
        })
    return result


def mark_episode_injected(
    conn: sqlite3.Connection,
    session_id: str,
) -> None:
    """记录情节被注入到新 session（用于注入频次统计）。"""
    conn.execute("""
        UPDATE session_episodes
        SET injected_count = injected_count + 1
        WHERE session_id = ?
    """, (session_id,))
    conn.commit()


# ── Summarization helper ──────────────────────────────────────────────────────

def build_episode_summary(
    last_message: str,
    chunks_created: int,
    files_modified: list,
    tools_used: dict,
) -> str:
    """
    从 Stop hook 可见数据构建情节摘要（不调用 LLM，纯规则）。
    目标：1-2 句，描述"这个 session 做了什么"。
    """
    parts = []

    # 1. 从 last_assistant_message 提取摘要句
    #    取前 3 行中最长的有意义句子
    if last_message:
        lines = [l.strip() for l in last_message.splitlines() if l.strip()]
        # 过滤掉 markdown 符号行、代码行
        import re
        text_lines = [
            l for l in lines
            if not l.startswith(('```', '#', '|', '---'))
               and not re.match(r'^[a-z_]+\s*=', l)  # 代码赋值
               and len(l) > 20
        ]
        if text_lines:
            # 取最长的前 120 个字符
            best = max(text_lines[:5], key=len)
            parts.append(best[:120])

    # 2. 行为摘要（文件/chunk）
    action_parts = []
    if chunks_created > 0:
        action_parts.append(f"写入 {chunks_created} 条知识")
    if files_modified:
        n = len(files_modified)
        if n <= 3:
            import os
            fnames = [os.path.basename(f) for f in files_modified]
            action_parts.append(f"修改 {', '.join(fnames)}")
        else:
            action_parts.append(f"修改 {n} 个文件")
    if tools_used:
        top_tools = sorted(tools_used.items(), key=lambda x: -x[1])[:3]
        tool_str = " / ".join(f"{t}×{c}" for t, c in top_tools)
        action_parts.append(f"工具: {tool_str}")

    if action_parts:
        parts.append("；".join(action_parts))

    summary = "。".join(parts) if parts else "session 结束（无显著操作）"
    return summary[:400]


def format_episodes_for_injection(episodes: list, max_chars: int = 300) -> str:
    """
    将情节列表格式化为可注入 context 的文本。
    OS 类比：ftrace 输出格式 — 紧凑但可读。

    Freshness gate（迭代12）：
      < 24h: 全文注入（高鲜度，完整上下文）
      24h–7d: 只注入日期+简短摘要（中等鲜度，减少 token）
      > 7d: 跳过（过期信息，信噪比太低）
    OS 类比：Linux page age tiering — hot/warm/cold 分层，cold page 不预取。
    """
    if not episodes:
        return ""

    import os as _os
    from datetime import datetime, timezone, timedelta
    _now = datetime.now(timezone.utc)
    _24H = timedelta(hours=24)
    _7D = timedelta(days=7)

    lines = ["【历史 Session 轨迹】"]
    total = len("【历史 Session 轨迹】")

    for ep in episodes:
        ended_str = ep.get("ended_at", "")
        # 解析 episode 时间，计算 age
        try:
            _dt = datetime.fromisoformat(ended_str.replace("Z", "+00:00"))
            if _dt.tzinfo is None:
                _dt = _dt.replace(tzinfo=timezone.utc)
            _age = _now - _dt
        except Exception:
            _age = timedelta(days=30)  # 解析失败视为很旧

        # freshness gate: > 7天跳过
        if _age > _7D:
            continue

        ended = ended_str[:10]  # 日期
        summary = ep.get("summary", "")
        chunks = ep.get("chunks_created", 0)
        files = ep.get("files_modified", [])

        if _age <= _24H:
            # 热区：完整注入
            summary_trunc = summary[:100]
            line_parts = [f"[{ended}] {summary_trunc}"]
            if chunks:
                line_parts.append(f"+{chunks}知识")
            if files:
                fnames = [_os.path.basename(f) for f in files[:2]]
                line_parts.append(" ".join(fnames))
        else:
            # 温区（24h–7d）：简短摘要（只取前40字）
            summary_trunc = summary[:40]
            line_parts = [f"[{ended}] {summary_trunc}"]
            if chunks:
                line_parts.append(f"+{chunks}知识")

        line = " · ".join(line_parts)
        if total + len(line) + 1 > max_chars:
            break
        lines.append(f"  {line}")
        total += len(line) + 1

    return "\n".join(lines) if len(lines) > 1 else ""
