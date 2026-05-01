#!/usr/bin/env python3
"""
thrashing_detector.py — PostToolUse Autocompact Thrashing Detector

迭代 B6：OS 类比 — Linux kswapd thrashing detection (2.6.32+)

背景：
  Linux 通过 vm.swappiness 和 kswapd 的 scan/reclaim 速率检测 thrashing：
  如果 page reclaim 速率 < page fault 速率，系统进入 thrashing 状态。
  内核记录 pswpin/pswpout 速率，超阈值时触发 OOM killer 或 cgroup throttle。

  AIOS 类比：
    autocompact = kswapd（将 context 压缩/换出）
    tool output 注入 = page fault（context 再填充）
    thrashing = 压缩速率 < 填充速率 → context 在 3 轮内重新填满

检测原理（滑动窗口）：
  1. PostToolUse 记录每次工具输出大小到 SQLite（复用 tool_profile.db）
  2. 计算滑动窗口（最近 N 次调用）的 context 增长速率
  3. 分级干预：
     - warn  (>WARN_MB)   : 注入警告，建议避免重复读大文件
     - block_hint (>HOT_MB): 强烈建议改用 Grep/LSP
     - critical (>CRIT_MB) : 强制建议 /clear + 存档关键信息

状态持久化：
  ~/.claude/memory-os/thrashing_state.json
  记录：
    - compact_times: 最近 compact 的时间戳列表
    - session_bytes: 本 session 累计 context 字节数（估算）
    - last_warn_ts: 上次 warn 的时间（防止 warn 风暴）

OS 类比 mapping：
  WARN_MB      ~ vm.vfs_cache_pressure (压力开始)
  HOT_MB       ~ /proc/pressure/memory some (PSI moderate)
  CRIT_MB      ~ /proc/pressure/memory full (PSI severe → OOM)
  WINDOW_CALLS ~ kswapd scan window size
"""

import sys
import json
import os
import sqlite3
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta

# 阈值配置
WARN_MB = 2.0        # 近 N 次调用累计输出 > 2MB → warn
HOT_MB = 5.0         # > 5MB → 强烈建议 Grep/LSP
CRIT_MB = 10.0       # > 10MB → 强制建议 /clear
WINDOW_CALLS = 20    # 滑动窗口大小（最近 N 次调用）
WARN_COOLDOWN_SECS = 120  # warn 冷却期（防止每次都 warn）

# 单个文件大小警戒线
LARGE_FILE_WARN_KB = 50   # > 50KB 的文件被 Read 时追加警告
LARGE_FILE_CRIT_KB = 200  # > 200KB 视为 thrashing 高危

MEMORY_OS_DIR = Path.home() / ".claude" / "memory-os"
PROFILE_DB = MEMORY_OS_DIR / "tool_profile.db"
STATE_FILE = MEMORY_OS_DIR / "thrashing_state.json"


def _load_state() -> dict:
    try:
        if STATE_FILE.exists():
            return json.loads(STATE_FILE.read_text())
    except Exception:
        pass
    return {
        "session_bytes": 0,
        "last_warn_ts": 0,
        "compact_count": 0,
        "window_bytes_history": [],  # list of (ts, bytes) tuples
    }


def _save_state(state: dict):
    try:
        MEMORY_OS_DIR.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(state, ensure_ascii=False))
    except Exception:
        pass


def _open_db() -> sqlite3.Connection | None:
    try:
        if not PROFILE_DB.exists():
            return None
        conn = sqlite3.connect(str(PROFILE_DB), timeout=3)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception:
        return None


def _get_window_bytes(conn: sqlite3.Connection, session_id: str, window: int) -> tuple[int, list]:
    """
    获取当前 session 最近 window 次调用的累计输出字节数。
    返回 (total_bytes, per_call_list)
    """
    try:
        rows = conn.execute("""
            SELECT output_bytes, tool_name, tool_key, ts
            FROM tool_calls
            WHERE session_id = ?
              AND output_bytes > 0
            ORDER BY id DESC
            LIMIT ?
        """, (session_id, window)).fetchall()
        total = sum(r["output_bytes"] for r in rows)
        details = [(r["tool_name"], r["tool_key"][:60], r["output_bytes"]) for r in rows]
        return total, details
    except Exception:
        return 0, []


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


def _record_call(conn: sqlite3.Connection, session_id: str,
                 tool_name: str, tool_key: str, output_bytes: int):
    """写入 tool_profile.db（复用 profiler schema）。"""
    try:
        conn.execute("""
            INSERT INTO tool_calls (ts, session_id, tool_name, tool_key, output_bytes, duration_ms, flagged)
            VALUES (datetime('now'), ?, ?, ?, ?, 0, 0)
        """, (session_id, tool_name, tool_key, output_bytes))
        conn.commit()
    except Exception:
        pass


def _get_file_size(tool_input: dict, tool_name: str) -> int:
    """对 Read 工具，获取目标文件的实际磁盘大小。"""
    if tool_name != "Read":
        return 0
    file_path = tool_input.get("file_path", "") if isinstance(tool_input, dict) else ""
    if not file_path:
        return 0
    try:
        return os.path.getsize(file_path)
    except Exception:
        return 0


def _build_notice(level: str, window_mb: float, session_mb: float,
                  top_tools: list, file_size_kb: float = 0,
                  file_name: str = "") -> str:
    """构造 additionalContext 提示。"""

    if level == "warn":
        msg = (
            f"[thrashing_detector:warn] 近 {WINDOW_CALLS} 次工具调用累计输出 {window_mb:.1f}MB，"
            f"本 session 累计 {session_mb:.1f}MB。"
        )
        if file_size_kb > LARGE_FILE_WARN_KB:
            msg += f" 当前读取文件 {file_name} 大小 {file_size_kb:.0f}KB，建议改用 Grep/LSP 精确查询。"
        else:
            msg += " 建议：优先用 Grep/LSP 替代整体 Read，避免 context thrashing。"

    elif level == "hot":
        top_str = "；".join(f"{t[0]}({t[2]//1024}KB)" for t in top_tools[:3] if t[2] > 0)
        msg = (
            f"[thrashing_detector:hot] ⚠ context 增长过快：近 {WINDOW_CALLS} 次输出 {window_mb:.1f}MB。"
            f" 高负载来源：{top_str or '未知'}。"
            f" 强烈建议：停止整体 Read 大文件，改用 Grep pattern/LSP goToDefinition。"
        )
        if file_size_kb > LARGE_FILE_CRIT_KB:
            msg += f" {file_name}({file_size_kb:.0f}KB) 是高危文件——请只读取需要的行段。"

    else:  # critical
        msg = (
            f"[thrashing_detector:critical] 🚨 Thrashing 风险极高！"
            f"近 {WINDOW_CALLS} 次输出 {window_mb:.1f}MB，session 累计 {session_mb:.1f}MB。"
            f" 极可能触发 Autocompact 循环。"
            f" 建议立即：1) 执行 /clear 重置 context；2) 将关键信息存入 memory；"
            f" 3) 只用 Grep/LSP/mcp__memory-os__memory_lookup 获取所需信息。"
        )

    return msg[:500]  # 限制长度防止 notice 本身膨胀 context


def main():
    try:
        raw = sys.stdin.read()
        hook_input = json.loads(raw) if raw.strip() else {}
    except Exception:
        sys.exit(0)

    tool_name = hook_input.get("tool_name", "")
    tool_input = hook_input.get("tool_input", {})
    tool_response = hook_input.get("tool_response", {})
    session_id = hook_input.get("session_id", "")

    if not tool_name:
        sys.exit(0)

    # 计算本次输出大小
    output_bytes = _get_output_size(tool_response)

    # 对 Read 工具，也检查文件磁盘大小（output_bytes 可能因 limit 参数偏小）
    file_size = _get_file_size(tool_input, tool_name)
    file_name = ""
    if tool_name == "Read" and isinstance(tool_input, dict):
        fp = tool_input.get("file_path", "")
        file_name = Path(fp).name if fp else ""
    # 取两者较大值作为 context 压力估算
    effective_bytes = max(output_bytes, file_size)

    # 加载状态
    state = _load_state()
    now_ts = time.time()

    # 更新 session 累计（使用 effective_bytes）
    state["session_bytes"] = state.get("session_bytes", 0) + effective_bytes

    # 更新滑动窗口历史
    history = state.get("window_bytes_history", [])
    history.append([now_ts, effective_bytes])
    # 只保留最近 WINDOW_CALLS * 2 条记录（时间窗口 + buffer）
    history = [h for h in history if now_ts - h[0] < 3600]  # 保留 1h 内
    state["window_bytes_history"] = history

    # 计算滑动窗口总字节（最近 WINDOW_CALLS 条）— 先算出压力等级再决定是否开 DB
    window_bytes = sum(h[1] for h in history[-WINDOW_CALLS:])
    window_mb = window_bytes / 1024 / 1024
    session_mb = state["session_bytes"] / 1024 / 1024
    file_size_kb = file_size / 1024

    # 冷却检查
    since_last_warn = now_ts - state.get("last_warn_ts", 0)
    in_cooldown = since_last_warn < WARN_COOLDOWN_SECS

    notice = None
    level = None

    # ── 懒开 SQLite（OS 类比：Linux writeback dirty ratio 门控）─────────────────
    # window_bytes < WARN_MB/2 时不开 DB：无告警可能，DB 写入纯浪费（每次 ~30ms）。
    # 超过 WARN_MB/2 时才打开，写入 tool_calls 供 top_tools 聚合。
    # 代价：warn 时 top_tools 数据少（低压力期间的记录被跳过），可接受。
    _db_threshold_mb = WARN_MB / 2  # 1.0 MB
    conn = None
    if window_mb >= _db_threshold_mb or file_size_kb >= LARGE_FILE_CRIT_KB:
        conn = _open_db()
        if conn and effective_bytes > 0:
            tool_key = f"{tool_name.lower()}:{tool_input.get('file_path', '') or tool_input.get('command', '')[:100]}"
            _record_call(conn, session_id, tool_name, tool_key, effective_bytes)

    if window_mb >= CRIT_MB:
        level = "critical"
    elif window_mb >= HOT_MB:
        level = "hot"
    elif window_mb >= WARN_MB and not in_cooldown:
        level = "warn"
    elif file_size_kb >= LARGE_FILE_CRIT_KB and not in_cooldown:
        # 单文件超大，即使窗口还没超阈值也警告
        level = "hot"

    if level:
        # 获取 top 工具（从历史中聚合）
        top_tools: list[tuple] = []
        if conn:
            try:
                rows = conn.execute("""
                    SELECT tool_name, tool_key, SUM(output_bytes) as total
                    FROM tool_calls
                    WHERE session_id = ?
                    GROUP BY tool_name, tool_key
                    ORDER BY total DESC
                    LIMIT 5
                """, (session_id,)).fetchall()
                top_tools = [(r["tool_name"], r["tool_key"], r["total"]) for r in rows]
            except Exception:
                pass

        notice = _build_notice(level, window_mb, session_mb, top_tools, file_size_kb, file_name)
        state["last_warn_ts"] = now_ts

    _save_state(state)
    if conn:
        conn.close()

    if notice:
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": notice,
            }
        }, ensure_ascii=False))

    sys.exit(0)


if __name__ == "__main__":
    main()
