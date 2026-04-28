"""
store_workspace.py — Workspace-aware memory layer (iter363)

OS 类比：进程地址空间切换（mm_struct switch）
  - 进入工作区 = context switch：整体加载该 workspace 的知识集
  - 离开工作区 = 地址空间卸载：清理热路径 cache
  - 文件变更感知 = inotify：扫描本地文件提取结构化 facts

核心思想：记忆的激活粒度是工作区（场景），而非词语（检索词）。
  BM25 是 demand paging（缺页后按需加载），
  Workspace activation 是 exec() 后整体加载程序地址空间 — 两者互补。
"""
import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from store_vfs import open_db, STORE_DB, _safe_add_column


# ── Schema ────────────────────────────────────────────────────────────────────

def ensure_workspace_schema(conn: sqlite3.Connection) -> None:
    """
    幂等 schema 初始化 — CREATE IF NOT EXISTS + ADD COLUMN.
    """
    # 工作区注册表 — 每个 cwd 路径对应一个 workspace
    conn.execute("""
        CREATE TABLE IF NOT EXISTS workspaces (
            id TEXT PRIMARY KEY,          -- sha256(cwd)[:16]
            path TEXT NOT NULL,           -- 绝对路径
            name TEXT NOT NULL,           -- basename
            created_at TEXT NOT NULL,
            last_entered TEXT,
            entry_count INTEGER DEFAULT 0
        )
    """)

    # workspace ↔ memory_chunk 关联表
    conn.execute("""
        CREATE TABLE IF NOT EXISTS workspace_knowledge (
            workspace_id TEXT NOT NULL,
            chunk_id TEXT NOT NULL,
            source TEXT NOT NULL,         -- 'conversation' | 'file_scan' | 'manual'
            pinned INTEGER DEFAULT 0,     -- 1 = 永不自动解除
            linked_at TEXT NOT NULL,
            PRIMARY KEY (workspace_id, chunk_id)
        )
    """)

    # 文件快照表 — 记录已扫描文件的 hash，用于增量更新
    conn.execute("""
        CREATE TABLE IF NOT EXISTS workspace_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_id TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_hash TEXT NOT NULL,      -- sha256 前16字节，用于变更检测
            scanned_at TEXT NOT NULL,
            facts_json TEXT,              -- 从文件提取的结构化 facts (JSON list)
            UNIQUE(workspace_id, file_path)
        )
    """)

    conn.execute("CREATE INDEX IF NOT EXISTS idx_wk_workspace ON workspace_knowledge(workspace_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_wf_workspace ON workspace_files(workspace_id)")
    conn.commit()


# ── Workspace resolution ───────────────────────────────────────────────────────

def _workspace_id(cwd: str) -> str:
    """cwd 路径 → 确定性 ID（sha256 前16字符）"""
    return hashlib.sha256(cwd.encode()).hexdigest()[:16]


def resolve_workspace(conn: sqlite3.Connection, cwd: str) -> str:
    """
    查找或创建 workspace 记录，返回 workspace_id。
    OS 类比：do_execve() 中的 mm_alloc() — 首次进入时分配新地址空间结构。
    """
    ensure_workspace_schema(conn)
    ws_id = _workspace_id(cwd)
    name = Path(cwd).name or cwd

    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT INTO workspaces (id, path, name, created_at, last_entered, entry_count)
        VALUES (?, ?, ?, ?, ?, 1)
        ON CONFLICT(id) DO UPDATE SET
            last_entered = excluded.last_entered,
            entry_count = entry_count + 1
    """, (ws_id, cwd, name, now, now))
    conn.commit()
    return ws_id


# ── Workspace activation / deactivation ───────────────────────────────────────

def activate_workspace(conn: sqlite3.Connection, workspace_id: str) -> dict:
    """
    激活工作区：返回该工作区关联的所有知识 chunks 和文件 facts。

    OS 类比：switch_mm() — 将目标进程的地址空间加载到 CPU，
      使后续的 demand paging 能在正确的地址空间中命中。

    Returns:
        {
            "workspace_id": str,
            "workspace_name": str,
            "workspace_path": str,
            "kb_chunks": [{"id": ..., "summary": ..., "content": ..., "chunk_type": ...}],
            "file_facts": [{"file": ..., "facts": [...]}]
        }
    """
    ensure_workspace_schema(conn)

    ws_row = conn.execute(
        "SELECT id, name, path FROM workspaces WHERE id = ?", (workspace_id,)
    ).fetchone()
    if not ws_row:
        return {"workspace_id": workspace_id, "kb_chunks": [], "file_facts": []}

    ws_id, ws_name, ws_path = ws_row

    # 加载关联的 KB chunks
    kb_rows = conn.execute("""
        SELECT mc.id, mc.summary, mc.content, mc.chunk_type, mc.importance,
               wk.pinned, wk.source
        FROM workspace_knowledge wk
        JOIN memory_chunks mc ON mc.id = wk.chunk_id
        WHERE wk.workspace_id = ?
        ORDER BY wk.pinned DESC, mc.importance DESC
    """, (ws_id,)).fetchall()

    kb_chunks = [
        {
            "id": r[0],
            "summary": r[1],
            "content": r[2],
            "chunk_type": r[3],
            "importance": r[4],
            "pinned": bool(r[5]),
            "source": r[6],
        }
        for r in kb_rows
    ]

    # 加载文件 facts
    file_rows = conn.execute("""
        SELECT file_path, facts_json
        FROM workspace_files
        WHERE workspace_id = ? AND facts_json IS NOT NULL AND facts_json != '[]'
        ORDER BY scanned_at DESC
    """, (ws_id,)).fetchall()

    file_facts = []
    for file_path, facts_json in file_rows:
        try:
            facts = json.loads(facts_json) if facts_json else []
        except (json.JSONDecodeError, TypeError):
            facts = []
        if facts:
            file_facts.append({"file": file_path, "facts": facts})

    return {
        "workspace_id": ws_id,
        "workspace_name": ws_name,
        "workspace_path": ws_path,
        "kb_chunks": kb_chunks,
        "file_facts": file_facts,
    }


# ── Workspace ↔ chunk linking ──────────────────────────────────────────────────

def link_chunk_to_workspace(
    conn: sqlite3.Connection,
    workspace_id: str,
    chunk_id: str,
    source: str = "conversation",
    pinned: bool = False,
) -> None:
    """
    将 memory chunk 关联到 workspace。
    OS 类比：mmap() — 将一个内存区域映射到地址空间。
    """
    ensure_workspace_schema(conn)
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT INTO workspace_knowledge (workspace_id, chunk_id, source, pinned, linked_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(workspace_id, chunk_id) DO UPDATE SET
            source = excluded.source,
            pinned = CASE WHEN excluded.pinned = 1 THEN 1 ELSE workspace_knowledge.pinned END,
            linked_at = excluded.linked_at
    """, (workspace_id, chunk_id, source, int(pinned), now))
    conn.commit()


def unlink_chunk_from_workspace(
    conn: sqlite3.Connection,
    workspace_id: str,
    chunk_id: str,
) -> None:
    """解除 chunk 与 workspace 的关联（仅解除非 pinned 的）。"""
    conn.execute("""
        DELETE FROM workspace_knowledge
        WHERE workspace_id = ? AND chunk_id = ? AND pinned = 0
    """, (workspace_id, chunk_id))
    conn.commit()


def get_workspace_knowledge(
    conn: sqlite3.Connection,
    workspace_id: str,
    chunk_type: Optional[str] = None,
) -> list:
    """
    获取 workspace 关联的所有 chunks（可按 type 过滤）。
    """
    ensure_workspace_schema(conn)
    if chunk_type:
        rows = conn.execute("""
            SELECT mc.id, mc.summary, mc.content, mc.chunk_type, mc.importance
            FROM workspace_knowledge wk
            JOIN memory_chunks mc ON mc.id = wk.chunk_id
            WHERE wk.workspace_id = ? AND mc.chunk_type = ?
            ORDER BY wk.pinned DESC, mc.importance DESC
        """, (workspace_id, chunk_type)).fetchall()
    else:
        rows = conn.execute("""
            SELECT mc.id, mc.summary, mc.content, mc.chunk_type, mc.importance
            FROM workspace_knowledge wk
            JOIN memory_chunks mc ON mc.id = wk.chunk_id
            WHERE wk.workspace_id = ?
            ORDER BY wk.pinned DESC, mc.importance DESC
        """, (workspace_id,)).fetchall()

    return [
        {"id": r[0], "summary": r[1], "content": r[2],
         "chunk_type": r[3], "importance": r[4]}
        for r in rows
    ]


# ── File snapshot (for change detection) ──────────────────────────────────────

def _file_hash(path: str) -> str:
    """计算文件内容的 sha256 前16字符。不存在时返回空字符串。"""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except OSError:
        return ""


def upsert_workspace_file(
    conn: sqlite3.Connection,
    workspace_id: str,
    file_path: str,
    facts: list,
) -> bool:
    """
    更新文件快照。仅当文件内容变更时返回 True（已扫描 & 无变化 → False）。
    OS 类比：inotify_add_watch() + IN_MODIFY 事件 — 跟踪文件变更。
    """
    ensure_workspace_schema(conn)
    current_hash = _file_hash(file_path)
    if not current_hash:
        return False  # 文件不存在

    existing = conn.execute("""
        SELECT file_hash FROM workspace_files
        WHERE workspace_id = ? AND file_path = ?
    """, (workspace_id, file_path)).fetchone()

    if existing and existing[0] == current_hash:
        return False  # 未变更

    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT INTO workspace_files (workspace_id, file_path, file_hash, scanned_at, facts_json)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(workspace_id, file_path) DO UPDATE SET
            file_hash = excluded.file_hash,
            scanned_at = excluded.scanned_at,
            facts_json = excluded.facts_json
    """, (workspace_id, file_path, current_hash, now, json.dumps(facts, ensure_ascii=False)))
    conn.commit()
    return True


def get_workspace_files(conn: sqlite3.Connection, workspace_id: str) -> list:
    """返回 workspace 的所有文件快照记录。"""
    ensure_workspace_schema(conn)
    rows = conn.execute("""
        SELECT file_path, file_hash, scanned_at, facts_json
        FROM workspace_files
        WHERE workspace_id = ?
        ORDER BY scanned_at DESC
    """, (workspace_id,)).fetchall()
    result = []
    for file_path, file_hash, scanned_at, facts_json in rows:
        try:
            facts = json.loads(facts_json) if facts_json else []
        except (json.JSONDecodeError, TypeError):
            facts = []
        result.append({
            "file_path": file_path,
            "file_hash": file_hash,
            "scanned_at": scanned_at,
            "facts": facts,
        })
    return result


# ── Workspace listing ──────────────────────────────────────────────────────────

def list_workspaces(conn: sqlite3.Connection) -> list:
    """列出所有注册的 workspaces。"""
    ensure_workspace_schema(conn)
    rows = conn.execute("""
        SELECT id, path, name, last_entered, entry_count
        FROM workspaces
        ORDER BY last_entered DESC
    """).fetchall()
    return [
        {"id": r[0], "path": r[1], "name": r[2],
         "last_entered": r[3], "entry_count": r[4]}
        for r in rows
    ]


def get_workspace_by_path(conn: sqlite3.Connection, cwd: str) -> Optional[dict]:
    """按路径查找 workspace，不存在返回 None。"""
    ensure_workspace_schema(conn)
    ws_id = _workspace_id(cwd)
    row = conn.execute(
        "SELECT id, path, name, last_entered, entry_count FROM workspaces WHERE id = ?",
        (ws_id,)
    ).fetchone()
    if not row:
        return None
    return {"id": row[0], "path": row[1], "name": row[2],
            "last_entered": row[3], "entry_count": row[4]}
