"""
store_todos.py — Workspace 级前瞻记忆（iter365）

OS 类比：inotify_add_watch() + IN_CLOSE_WRITE — "当我回到这个目录时提醒我"
  - workspace_todos 表 = at/cron job，但绑定到 workspace 生命周期
  - 进入 workspace = 触发待办事项注入（类似 cron 检查到达时间）
  - TTL = 无限（直到完成），与 session_intent 的 24h TTL 不同

人的记忆类比：前瞻性记忆（Prospective Memory）
  - "我记得要回来检查这个" — 带有"未来行动"意向的记忆
  - 不同于 CRIU intent（会话级，24h）：workspace todo 跨多个 session 保持有效
  - 典型场景："等 X 合并后来处理 Y"/"下次进这个项目要先看日志"

与 CRIU session intent 的区别：
  | session_intent | workspace_todos |
  |---|---|
  | 24h TTL，自动过期 | 无限，直到 done/cancel |
  | 会话级（我上次做到哪） | 工作区级（我记得要做什么） |
  | 被动恢复（CRIU restore） | 主动提醒（cron + inotify） |
"""
import json
import re
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from store_vfs import _safe_add_column


# ── Schema ────────────────────────────────────────────────────────────────────

def ensure_todos_schema(conn: sqlite3.Connection) -> None:
    """幂等 schema 初始化。"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS workspace_todos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_id TEXT NOT NULL,
            project TEXT NOT NULL,
            content TEXT NOT NULL,          -- 待办内容（自然语言）
            created_at TEXT NOT NULL,
            source_session TEXT,            -- 创建时的 session_id
            due_hint TEXT,                  -- 触发条件提示（如 "等X合并后"）
            status TEXT NOT NULL DEFAULT 'pending',  -- pending / done / cancelled
            completed_at TEXT,
            injected_count INTEGER DEFAULT 0  -- 被注入次数
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_todos_workspace ON workspace_todos(workspace_id, status)"
    )
    conn.commit()


# ── Write ─────────────────────────────────────────────────────────────────────

def add_todo(
    conn: sqlite3.Connection,
    workspace_id: str,
    project: str,
    content: str,
    *,
    source_session: str = "",
    due_hint: str = "",
) -> int:
    """添加一条 workspace 级待办，返回 id。"""
    ensure_todos_schema(conn)
    now = datetime.now(timezone.utc).isoformat()
    row = conn.execute("""
        INSERT INTO workspace_todos
        (workspace_id, project, content, created_at, source_session, due_hint)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (workspace_id, project, content[:300], now, source_session, due_hint[:100]))
    conn.commit()
    return row.lastrowid


def complete_todo(conn: sqlite3.Connection, todo_id: int) -> None:
    """标记待办为完成。"""
    conn.execute("""
        UPDATE workspace_todos SET status='done', completed_at=?
        WHERE id=?
    """, (datetime.now(timezone.utc).isoformat(), todo_id))
    conn.commit()


def cancel_todo(conn: sqlite3.Connection, todo_id: int) -> None:
    conn.execute(
        "UPDATE workspace_todos SET status='cancelled' WHERE id=?",
        (todo_id,)
    )
    conn.commit()


# ── Read ──────────────────────────────────────────────────────────────────────

def get_pending_todos(
    conn: sqlite3.Connection,
    workspace_id: str,
    limit: int = 5,
) -> list:
    """获取 workspace 的 pending 待办列表。"""
    ensure_todos_schema(conn)
    rows = conn.execute("""
        SELECT id, content, due_hint, created_at, injected_count
        FROM workspace_todos
        WHERE workspace_id = ? AND status = 'pending'
        ORDER BY created_at ASC
        LIMIT ?
    """, (workspace_id, limit)).fetchall()
    return [
        {"id": r[0], "content": r[1], "due_hint": r[2],
         "created_at": r[3], "injected_count": r[4]}
        for r in rows
    ]


def mark_todo_injected(conn: sqlite3.Connection, todo_id: int) -> None:
    conn.execute(
        "UPDATE workspace_todos SET injected_count = injected_count + 1 WHERE id=?",
        (todo_id,)
    )
    conn.commit()


def format_todos_for_injection(todos: list, max_chars: int = 200) -> str:
    """格式化为可注入 context 的文本。"""
    if not todos:
        return ""
    lines = ["【工作区待办】"]
    total = len(lines[0])
    for t in todos:
        due = f"（{t['due_hint']}）" if t.get("due_hint") else ""
        line = f"  - {t['content'][:80]}{due}"
        if total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line)
    return "\n".join(lines) if len(lines) > 1 else ""


# ── Extraction from conversation ──────────────────────────────────────────────

# 前瞻性记忆触发模式 — 识别"下次/以后/等X之后"类型的表达
TODO_PATTERNS = [
    # 显式 TODO/待办
    r'(?:TODO|FIXME|HACK|NOTE)[：:\s]+(.{10,120})',
    r'(?:待办|记得|提醒)[：:\s]+(.{10,100})',
    # 条件性未来行动
    r'等\s*(.{3,30})\s*(?:合并|发布|完成|上线|解决)后[，,]?\s*(?:再|需要|来|回来)(.{5,80})',
    r'(?:下次|以后|将来)\s*(?:进来|进入|回来)?\s*(?:要|需要|记得)\s*(.{5,100})',
    r'(?:暂时|先|目前)\s*(?:跳过|不处理|搁置)\s*(.{5,80})',
    # 英文
    r'(?:will need to|need to remember|TODO later)[：:\s]+(.{10,100})',
    r'(?:defer|postpone|skip for now)[：:\s]+(.{5,80})',
    # 带条件的
    r'(?:once|after|when)\s+(.{5,40})\s+(?:is done|merges|completes)[,，]?\s+(.{5,80})',
]

_TODO_COMPILED = [re.compile(p, re.IGNORECASE) for p in TODO_PATTERNS]


def extract_todos_from_text(text: str) -> list:
    """
    从对话文本中提取前瞻性记忆信号，返回 [(content, due_hint)] 列表。
    """
    results = []
    seen = set()
    for pat in _TODO_COMPILED:
        for m in pat.finditer(text):
            groups = [g.strip() for g in m.groups() if g and g.strip()]
            if len(groups) == 1:
                content = groups[0]
                due_hint = ""
            elif len(groups) >= 2:
                # 条件性：group1 = 条件，group2 = 行动
                due_hint = groups[0]
                content = groups[1]
            else:
                continue
            # 过滤过短或纯标点
            if len(content) < 8:
                continue
            # 去重
            key = content[:50].lower()
            if key in seen:
                continue
            seen.add(key)
            results.append({"content": content[:200], "due_hint": due_hint[:80]})
    return results[:5]  # 每 session 最多提取 5 条
