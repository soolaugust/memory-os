#!/usr/bin/env python3
"""
memory-os tool_profiler — PostToolUse hook
迭代110 P3: eBPF-style Tool Call Observability

OS 类比: Linux eBPF (2014+) + perf_event — 在内核关键路径上挂探针，
  以极低开销收集 per-syscall 延迟、调用频率、参数分布。
  不修改内核代码，只观测。

AIOS 类比: 在每个工具调用后记录：
  - tool_name, duration_ms, output_size_bytes
  - 是否重复调用相同文件/命令（ineffective call detection）
  - 写入 tool_profile.db (SQLite, 轻量，不影响 store.db)

用途:
  1. 识别高频低效工具调用（同一文件 Read 3+ 次 → hint）
  2. 找出 context 占用热点（哪个工具产出最大输出）
  3. 为 P1 output_compressor 提供阈值校准数据

ineffective call 检测规则:
  - 同一 file_path Read > 2 次（应该 cache 或精确查询）
  - 同一 Bash command 执行 > 1 次（应该保存输出）
  - Read 后 Grep 同一文件（应该直接 Read + 内存处理）

输出: additionalContext（仅在检测到低效时注入提示，避免噪音）
持久化: ~/.claude/memory-os/tool_profile.db
"""

import sys
import json
import sqlite3
import time
import re
from pathlib import Path
from datetime import datetime, timezone

MEMORY_OS_DIR = Path.home() / ".claude" / "memory-os"
PROFILE_DB = MEMORY_OS_DIR / "tool_profile.db"

# 触发 ineffective call 警告的阈值
READ_REPEAT_THRESHOLD = 3     # 同一文件 Read 次数
BASH_REPEAT_THRESHOLD = 2     # 同一命令 Bash 次数
SESSION_WINDOW_SECS = 3600    # 只统计最近 1h 内的调用

# 最大 DB 行数（防止无限增长）
MAX_DB_ROWS = 10_000


def _open_profile_db() -> sqlite3.Connection:
    MEMORY_OS_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(PROFILE_DB))
    # ── 性能优化：profiler 数据丢失可接受（观测数据非关键），最大化写入速度 ──
    # OS 类比：Linux O_SYNC vs buffered write — profiler 不需要 fsync 保证，
    #   每次 commit 的 fsync 是 ~50ms 的固定税，synchronous=OFF 去掉它。
    # journal_mode=WAL 保留（读写并发），synchronous=OFF（不 fsync WAL header）
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tool_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            session_id TEXT DEFAULT '',
            tool_name TEXT NOT NULL,
            tool_key TEXT NOT NULL,   -- normalized key for dedup detection
            output_bytes INTEGER DEFAULT 0,
            duration_ms REAL DEFAULT 0,
            flagged INTEGER DEFAULT 0  -- 1 = detected as ineffective
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ts ON tool_calls(ts)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_key ON tool_calls(tool_name, tool_key)")
    # 不在此处 commit：schema DDL 在 WAL 模式下非事务，下次写入时一并提交
    return conn


def _normalize_key(tool_name: str, tool_input: dict) -> str:
    """提取工具调用的规范化 key，用于重复检测。"""
    if tool_name == "Read":
        path = tool_input.get("file_path", "")
        # 去掉 offset/limit 参数，只用文件路径
        return f"read:{path}"
    elif tool_name == "Bash":
        cmd = tool_input.get("command", "")
        # 规范化：去掉多余空白、注释
        cmd = re.sub(r'\s+', ' ', cmd.strip())[:200]
        return f"bash:{cmd}"
    elif tool_name == "Grep":
        pattern = tool_input.get("pattern", "")
        path = tool_input.get("path", "")
        return f"grep:{path}:{pattern[:50]}"
    elif tool_name in ("Edit", "Write", "MultiEdit"):
        path = tool_input.get("file_path", "")
        return f"write:{path}"
    else:
        return f"{tool_name.lower()}:_"


def _get_output_size(tool_response) -> int:
    """计算工具输出大小（bytes）。"""
    if isinstance(tool_response, str):
        return len(tool_response.encode("utf-8", errors="replace"))
    elif isinstance(tool_response, dict):
        content = tool_response.get("content", "")
        if isinstance(content, str):
            return len(content.encode("utf-8", errors="replace"))
        elif isinstance(content, list):
            total = 0
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text":
                    total += len(c.get("text", "").encode("utf-8", errors="replace"))
            return total
    return 0


def _check_ineffective(conn: sqlite3.Connection, tool_name: str,
                       tool_key: str, session_id: str) -> tuple[bool, str]:
    """
    检查当前调用是否属于低效调用。
    返回 (is_ineffective, reason_str)。
    """
    if tool_name not in ("Read", "Bash", "Grep"):
        return False, ""

    threshold = READ_REPEAT_THRESHOLD if tool_name != "Bash" else BASH_REPEAT_THRESHOLD

    cutoff = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    # 统计近 1h 内同一 tool_key 的调用次数
    row = conn.execute(
        """SELECT COUNT(*) FROM tool_calls
           WHERE tool_name = ? AND tool_key = ?
             AND ts >= datetime('now', '-1 hour')""",
        (tool_name, tool_key)
    ).fetchone()
    count = row[0] if row else 0

    if count >= threshold:
        entity = tool_key.split(":", 1)[-1][:60]
        return True, f"[eBPF] {tool_name} × {count+1}次: '{entity}' — 建议缓存结果或精确查询"

    return False, ""


def _evict_old_rows(conn: sqlite3.Connection) -> None:
    """Keep DB size in check — delete oldest rows when over limit."""
    row = conn.execute("SELECT COUNT(*) FROM tool_calls").fetchone()
    if row and row[0] > MAX_DB_ROWS:
        overflow = row[0] - MAX_DB_ROWS
        conn.execute(
            "DELETE FROM tool_calls WHERE id IN "
            "(SELECT id FROM tool_calls ORDER BY id ASC LIMIT ?)",
            (overflow,)
        )


def main():
    t_start = time.monotonic()

    try:
        raw = sys.stdin.read()
        hook_input = json.loads(raw) if raw.strip() else {}
    except Exception:
        sys.exit(0)

    tool_name = hook_input.get("tool_name", "")
    tool_input = hook_input.get("tool_input", {}) or {}
    tool_response = hook_input.get("tool_response", {})
    session_id = hook_input.get("session_id", "")

    if not tool_name:
        sys.exit(0)

    # 只 profile 高频/高影响工具
    PROFILED_TOOLS = {"Read", "Bash", "Grep", "Glob", "Write", "Edit", "MultiEdit"}
    if tool_name not in PROFILED_TOOLS:
        sys.exit(0)

    tool_key = _normalize_key(tool_name, tool_input)
    output_bytes = _get_output_size(tool_response)
    duration_ms = (time.monotonic() - t_start) * 1000  # hook 自身耗时（近似）

    try:
        conn = _open_profile_db()

        # 检测低效调用（在写入本次记录之前）
        is_ineffective, ineffective_msg = _check_ineffective(conn, tool_name, tool_key, session_id)

        # 写入本次调用记录
        conn.execute(
            "INSERT INTO tool_calls (ts, session_id, tool_name, tool_key, output_bytes, duration_ms, flagged) "
            "VALUES (datetime('now'), ?, ?, ?, ?, ?, ?)",
            (session_id, tool_name, tool_key, output_bytes, round(duration_ms, 2),
             1 if is_ineffective else 0)
        )
        _evict_old_rows(conn)
        conn.commit()
        conn.close()

        if is_ineffective:
            print(json.dumps({
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": ineffective_msg,
                }
            }, ensure_ascii=False))

    except Exception:
        pass  # 永远不阻塞工具执行

    sys.exit(0)


if __name__ == "__main__":
    main()
