#!/usr/bin/env python3
"""
posttool_guard.py — PostToolUse 合并守卫（output_compressor + thrashing_detector）

迭代合并：将两个同步 PostToolUse Python 进程合并为一个。
  原来：output_compressor(~38ms) + thrashing_detector(~38ms) = ~76ms
  现在：posttool_guard(~40ms) = 单次进程启动，两个功能

OS 类比：Linux interrupt coalescing（NAPI, 2001）—
  高频网卡中断合并为批量处理，减少 context switch 次数。
  每次 PostToolUse 触发两次 Python 进程 = 两次 fork/exec/exit（固定税）。
  合并后只有一次。

功能：
  1. zram 压缩提示（原 output_compressor）：Bash/Read 大输出时注入注意力指引
  2. thrashing 检测（原 thrashing_detector）：context 增长过快时注入告警

合并策略：
  - zram 提示 优先级高（直接可操作的建议）
  - thrashing 告警 追加在后（如果 zram 已提示，thrashing 不重复）
  - 共用一次 JSON state 读写（thrashing 需要）
"""

import sys
import json
import os
import sqlite3
import time
import re
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

MEMORY_OS_DIR = Path.home() / ".claude" / "memory-os"
PROFILE_DB = MEMORY_OS_DIR / "tool_profile.db"
STATE_FILE = MEMORY_OS_DIR / "thrashing_state.json"

# ── zram 压缩阈值 ──────────────────────────────────────────────────────────────
BASH_THRESHOLD_BYTES = 3 * 1024
READ_THRESHOLD_BYTES = 4 * 1024
BASH_HEAD_LINES = 20
BASH_TAIL_LINES = 40
READ_HEAD_LINES = 30
READ_TAIL_LINES = 30
MAX_NOTICE_LEN = 300

# ── thrashing 检测阈值 ──────────────────────────────────────────────────────────
WARN_MB = 2.0
HOT_MB = 5.0
CRIT_MB = 10.0
LARGE_FILE_CRIT_KB = 500
WINDOW_CALLS = 20
WARN_COOLDOWN_SECS = 120


# ══════════════════════════════════════════════════════════════════════════════
# zram 压缩功能
# ══════════════════════════════════════════════════════════════════════════════

def _is_error_line(line: str) -> bool:
    lower = line.lower()
    return any(kw in lower for kw in (
        'error', 'exception', 'traceback', 'failed', 'fatal',
        'warning', 'warn:', 'critical', 'assert',
        '错误', '异常', '失败', '警告',
    ))


def _compress_bash_output(output: str, tool_input_cmd: str) -> str | None:
    if len(output.encode('utf-8', errors='replace')) < BASH_THRESHOLD_BYTES:
        return None
    lines = output.splitlines()
    if len(lines) <= BASH_HEAD_LINES + BASH_TAIL_LINES + 10:
        return None
    omitted = len(lines) - BASH_HEAD_LINES - BASH_TAIL_LINES
    error_lines = [l for l in lines[BASH_HEAD_LINES:-BASH_TAIL_LINES] if _is_error_line(l)]
    parts = [
        f"[zram:Bash] 输出 {len(lines)} 行，已压缩（保留首{BASH_HEAD_LINES}行、末{BASH_TAIL_LINES}行）。",
        f"中间省略 {omitted} 行。",
    ]
    if error_lines:
        parts.append(f"发现 {len(error_lines)} 个错误/警告行（已包含在末尾区域）。")
    parts.append("请优先关注最后几行（实际结果）和错误行。完整输出可用 Read 读取。")
    return " ".join(parts)[:MAX_NOTICE_LEN]


def _compress_read_output(output: str, file_path: str) -> str | None:
    if len(output.encode('utf-8', errors='replace')) < READ_THRESHOLD_BYTES:
        return None
    lines = output.splitlines()
    if len(lines) <= READ_HEAD_LINES + READ_TAIL_LINES + 10:
        return None
    omitted = len(lines) - READ_HEAD_LINES - READ_TAIL_LINES
    fname = Path(file_path).name if file_path else '(unknown)'
    return (
        f"[zram:Read] {fname} 共 {len(lines)} 行，已压缩（首{READ_HEAD_LINES}+末{READ_TAIL_LINES}行）。"
        f"省略中间 {omitted} 行。如需读取特定区域，用 offset/limit 参数。"
    )[:MAX_NOTICE_LEN]


def _extract_output_text(tool_response) -> str:
    """从 tool_response 中提取文本内容"""
    if isinstance(tool_response, str):
        return tool_response
    if isinstance(tool_response, dict):
        content = tool_response.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                c.get("text", "") for c in content
                if isinstance(c, dict) and c.get("type") == "text"
            )
    return ""


# ══════════════════════════════════════════════════════════════════════════════
# thrashing 检测功能
# ══════════════════════════════════════════════════════════════════════════════

def _load_state() -> dict:
    try:
        if STATE_FILE.exists():
            return json.loads(STATE_FILE.read_text())
    except Exception:
        pass
    return {"session_bytes": 0, "last_warn_ts": 0, "compact_count": 0,
            "window_bytes_history": []}


def _save_state(state: dict):
    try:
        STATE_FILE.write_text(json.dumps(state, ensure_ascii=False))
    except Exception:
        pass


def _get_file_size(tool_input: dict, tool_name: str) -> int:
    if tool_name != "Read":
        return 0
    fp = tool_input.get("file_path", "") if isinstance(tool_input, dict) else ""
    if not fp:
        return 0
    try:
        return Path(fp).stat().st_size
    except Exception:
        return 0


def _build_thrashing_notice(level: str, window_mb: float, session_mb: float,
                             top_tools: list, file_size_kb: float, file_name: str) -> str:
    if level == "critical":
        base = (
            f"[thrashing_detector:critical] ⚠⚠ context 极度膨胀！"
            f"近 {WINDOW_CALLS} 次输出 {window_mb:.1f}MB，session 累计 {session_mb:.1f}MB。"
            f" 强烈建议立即 /clear，memory-os 已记录关键知识。"
        )
    elif level == "hot":
        top_str = ""
        if top_tools:
            top_str = "高负载来源：" + "；".join(
                f"{t[0]}({t[2]//1024}KB)" for t in top_tools[:3]
            ) + "。"
        file_hint = f" {file_name}({file_size_kb:.0f}KB) 是高危文件——请只读取需要的行段。" if file_name and file_size_kb >= LARGE_FILE_CRIT_KB else ""
        base = (
            f"[thrashing_detector:hot] ⚠ context 增长过快：近 {WINDOW_CALLS} 次输出 {window_mb:.1f}MB。"
            f" {top_str}强烈建议：停止整体 Read 大文件，改用 Grep pattern/LSP goToDefinition。{file_hint}"
        )
    else:  # warn
        base = (
            f"[thrashing_detector:warn] 近 {WINDOW_CALLS} 次工具调用累计输出 {window_mb:.1f}MB，"
            f"建议改用 Grep/LSP 精确查找，减少整体 Read。"
        )
    return base[:500]


def _open_profile_db() -> sqlite3.Connection | None:
    try:
        if not PROFILE_DB.exists():
            return None
        conn = sqlite3.connect(str(PROFILE_DB), timeout=3)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception:
        return None


def _run_thrashing(tool_name: str, tool_input: dict, effective_bytes: int,
                   session_id: str, file_size: int, file_name: str,
                   state: dict, now_ts: float) -> str | None:
    """执行 thrashing 检测，返回告警文本或 None。"""
    # 更新 session 累计
    state["session_bytes"] = state.get("session_bytes", 0) + effective_bytes

    # 更新滑动窗口历史
    history = state.get("window_bytes_history", [])
    history.append([now_ts, effective_bytes])
    history = [h for h in history if now_ts - h[0] < 3600]
    state["window_bytes_history"] = history

    window_bytes = sum(h[1] for h in history[-WINDOW_CALLS:])
    window_mb = window_bytes / 1024 / 1024
    session_mb = state["session_bytes"] / 1024 / 1024
    file_size_kb = file_size / 1024

    since_last_warn = now_ts - state.get("last_warn_ts", 0)
    in_cooldown = since_last_warn < WARN_COOLDOWN_SECS

    level = None
    if window_mb >= CRIT_MB:
        level = "critical"
    elif window_mb >= HOT_MB:
        level = "hot"
    elif window_mb >= WARN_MB and not in_cooldown:
        level = "warn"
    elif file_size_kb >= LARGE_FILE_CRIT_KB and not in_cooldown:
        level = "hot"

    if not level:
        return None

    # 超阈值时才开 DB 查 top_tools
    top_tools = []
    _db_threshold_mb = WARN_MB / 2
    if window_mb >= _db_threshold_mb or file_size_kb >= LARGE_FILE_CRIT_KB:
        conn = _open_profile_db()
        if conn:
            try:
                rows = conn.execute("""
                    SELECT tool_name, tool_key, SUM(output_bytes) as total
                    FROM tool_calls WHERE session_id = ?
                    GROUP BY tool_name, tool_key
                    ORDER BY total DESC LIMIT 5
                """, (session_id,)).fetchall()
                top_tools = [(r["tool_name"], r["tool_key"], r["total"]) for r in rows]
            except Exception:
                pass
            conn.close()

    state["last_warn_ts"] = now_ts
    return _build_thrashing_notice(level, window_mb, session_mb, top_tools, file_size_kb, file_name)


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

def main():
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

    notices = []

    # ── Phase 1: zram 压缩（仅 Bash/Read）──────────────────────────────────
    output_text = _extract_output_text(tool_response) if tool_name in ("Bash", "Read") else ""
    zram_notice = None
    if tool_name == "Bash":
        cmd = tool_input.get("command", "") if isinstance(tool_input, dict) else str(tool_input)
        zram_notice = _compress_bash_output(output_text, cmd)
    elif tool_name == "Read":
        file_path = tool_input.get("file_path", "") if isinstance(tool_input, dict) else ""
        zram_notice = _compress_read_output(output_text, file_path)

    if zram_notice:
        notices.append(zram_notice)

    # ── Phase 2: thrashing 检测（所有工具）──────────────────────────────────
    output_bytes = len(output_text.encode('utf-8', errors='replace')) if output_text else 0
    file_size = _get_file_size(tool_input, tool_name)
    file_name = ""
    if tool_name == "Read" and isinstance(tool_input, dict):
        fp = tool_input.get("file_path", "")
        file_name = Path(fp).name if fp else ""
    effective_bytes = max(output_bytes, file_size)

    state = _load_state()
    now_ts = time.time()

    # thrashing 告警只在 zram 未告警（或 zram 未触发）时注入（避免重复打扰）
    thrashing_notice = _run_thrashing(
        tool_name, tool_input, effective_bytes,
        session_id, file_size, file_name, state, now_ts
    )
    if thrashing_notice and not zram_notice:
        notices.append(thrashing_notice)

    _save_state(state)

    if notices:
        combined = " | ".join(notices)
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": combined[:600],
            }
        }, ensure_ascii=False))

    sys.exit(0)


if __name__ == "__main__":
    main()
