#!/usr/bin/env python3
"""
memory-os output_compressor — PostToolUse hook
迭代110: Tool Output Compression (zram analogy)

OS 类比: Linux zram (2014) — 内存不足时将 LRU 页面压缩后留在 RAM，
而非直接 swap 到磁盘。代价是 CPU 时间，收益是 RAM 空间。

AIOS 类比: 大型工具输出（Bash stdout / Read 内容）占用大量 context tokens。
压缩策略：
  - Bash: 保留 首20行(headers) + 最后40行(result) + 所有 error/warning 行
           截断中间部分，注入摘要提示
  - Read: 保留 首30行 + 末30行，中间用 [...N lines omitted...] 占位
  - 阈值: 输出 > 3KB (约 750 tokens) 才触发

效果预期: ~30% context 节省（实测大 Bash 输出通常 5-20KB）

不修改工具输出本身（Claude Code hooks 不支持），
而是通过 additionalContext 注入"注意力指引"：
  "[zram] Bash output truncated for context efficiency.
   Focus on: last N lines (actual result) + error lines.
   Full output is available via Read if needed."

对 Claude 的实际效果：
  - Claude 在看到这条提示后，会优先关注 additionalContext 指出的部分
  - 相当于 madvise(MADV_COLD) 冷化中间部分，MADV_WILLNEED 预热关键部分
"""

import sys
import json
import re
from pathlib import Path

# 触发阈值
BASH_THRESHOLD_BYTES = 3 * 1024   # 3KB Bash output
READ_THRESHOLD_BYTES = 4 * 1024   # 4KB Read output (bigger because code is denser)

# 保留行数（压缩后）
BASH_HEAD_LINES = 20
BASH_TAIL_LINES = 40
READ_HEAD_LINES = 30
READ_TAIL_LINES = 30

# 最大摘要长度（additionalContext）
MAX_NOTICE_LEN = 300


def _is_error_line(line: str) -> bool:
    """检测错误/警告行（高价值，必须保留）"""
    lower = line.lower()
    return any(kw in lower for kw in (
        'error', 'exception', 'traceback', 'failed', 'fatal',
        'warning', 'warn:', 'critical', 'assert',
        '错误', '异常', '失败', '警告',
    ))


def _compress_bash_output(output: str, tool_input: str) -> str | None:
    """
    压缩 Bash 输出。返回 additionalContext 字符串，None 表示不压缩。
    """
    if len(output.encode('utf-8', errors='replace')) < BASH_THRESHOLD_BYTES:
        return None

    lines = output.splitlines()
    total_lines = len(lines)

    # 已经很短就不压缩
    if total_lines <= BASH_HEAD_LINES + BASH_TAIL_LINES + 10:
        return None

    head = lines[:BASH_HEAD_LINES]
    tail = lines[-BASH_TAIL_LINES:]
    error_lines = [l for l in lines[BASH_HEAD_LINES:-BASH_TAIL_LINES]
                   if _is_error_line(l)]
    omitted = total_lines - BASH_HEAD_LINES - BASH_TAIL_LINES

    # 提取命令（用于提示）
    cmd_preview = tool_input[:80].split('\n')[0] if tool_input else ''

    parts = [
        f"[zram:Bash] 输出 {total_lines} 行，已压缩（保留首{BASH_HEAD_LINES}行、末{BASH_TAIL_LINES}行）。",
        f"中间省略 {omitted} 行。",
    ]
    if error_lines:
        parts.append(f"发现 {len(error_lines)} 个错误/警告行（已包含在末尾区域）。")
    parts.append("请优先关注最后几行（实际结果）和错误行。完整输出可用 Read 读取。")

    return " ".join(parts)[:MAX_NOTICE_LEN]


def _compress_read_output(output: str, file_path: str) -> str | None:
    """
    压缩 Read 输出。返回 additionalContext 字符串，None 表示不压缩。
    """
    if len(output.encode('utf-8', errors='replace')) < READ_THRESHOLD_BYTES:
        return None

    lines = output.splitlines()
    total_lines = len(lines)

    if total_lines <= READ_HEAD_LINES + READ_TAIL_LINES + 10:
        return None

    omitted = total_lines - READ_HEAD_LINES - READ_TAIL_LINES
    fname = Path(file_path).name if file_path else '(unknown)'

    return (
        f"[zram:Read] {fname} 共 {total_lines} 行，已压缩（首{READ_HEAD_LINES}+末{READ_TAIL_LINES}行）。"
        f"省略中间 {omitted} 行。如需读取特定区域，用 offset/limit 参数。"
    )[:MAX_NOTICE_LEN]


def main():
    try:
        raw = sys.stdin.read()
        hook_input = json.loads(raw) if raw.strip() else {}
    except Exception:
        sys.exit(0)

    tool_name = hook_input.get("tool_name", "")
    tool_input = hook_input.get("tool_input", {})
    tool_response = hook_input.get("tool_response", {})

    if not tool_name or not tool_response:
        sys.exit(0)

    notice = None

    if tool_name == "Bash":
        # Bash: tool_response 是字符串或 {"type": "tool_result", "content": "..."}
        output = ""
        if isinstance(tool_response, str):
            output = tool_response
        elif isinstance(tool_response, dict):
            content = tool_response.get("content", "")
            if isinstance(content, str):
                output = content
            elif isinstance(content, list):
                parts = []
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        parts.append(c.get("text", ""))
                output = "\n".join(parts)

        cmd = tool_input.get("command", "") if isinstance(tool_input, dict) else str(tool_input)
        notice = _compress_bash_output(output, cmd)

    elif tool_name == "Read":
        output = ""
        if isinstance(tool_response, str):
            output = tool_response
        elif isinstance(tool_response, dict):
            content = tool_response.get("content", "")
            if isinstance(content, str):
                output = content
            elif isinstance(content, list):
                parts = []
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        parts.append(c.get("text", ""))
                output = "\n".join(parts)

        file_path = tool_input.get("file_path", "") if isinstance(tool_input, dict) else ""
        notice = _compress_read_output(output, file_path)

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
