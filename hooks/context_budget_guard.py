#!/usr/bin/env python3
"""
context_budget_guard.py — SessionStart 时自动检查 context budget

迭代 B5：OS 类比 — Linux Early OOM (earlyoom, 2017)

背景：
  Linux OOM killer 在内存完全耗尽时才触发——此时系统已严重卡顿。
  earlyoom (2017) 在内存压力到达阈值时就提前 kill 低优先级进程，
  避免系统进入 thrashing 状态。

  AIOS 类比：
    "Prompt is too long" = OOM（系统提示已超限，session 无法启动）。
    context_budget_guard = earlyoom（在 SessionStart 时检测并提前回收）。

集成方式：
  在 settings.json → hooks.SessionStart 中添加：
    {"type": "command", "command": "python3 .../context_budget_guard.py", "timeout": 10}

  如果 pressure >= "some"：
    1. 自动 reclaim 低优先级组件
    2. 通过 additionalContext 注入警告信息
    3. 记录 dmesg 日志
"""

import sys
import json
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from context_cgroup import scan, reclaim


def main():
    try:
        _raw = sys.stdin.read()
    except Exception:
        _raw = ""

    report = scan()

    if report.pressure == "none":
        # 在预算内，不输出任何 additionalContext（节省 tokens）
        sys.exit(0)

    # pressure = "some" or "full" → 自动回收
    report = reclaim(report, dry_run=False)

    # 构造注入信息
    lines = [f"⚠ Context Budget: {report.total_chars:,}/{report.max_chars:,} chars ({report.usage_pct:.0f}%) pressure={report.pressure}"]

    actions = [a for a in report.actions_taken if isinstance(a, dict)]
    if actions:
        freed = sum(a.get("chars_freed", 0) for a in actions if a.get("executed", False))
        lines.append(f"Auto-reclaimed {len(actions)} components ({freed:,} chars freed)")

    # 如果仍超限，建议手动清理
    if report.pressure == "full":
        lines.append("仍超限！建议：python3 aios/memory-os/context_cgroup.py scan")

    context_text = " | ".join(lines)

    # dmesg 日志
    try:
        from store import open_db, ensure_schema, dmesg_log, DMESG_WARN
        conn = open_db()
        ensure_schema(conn)
        dmesg_log(conn, DMESG_WARN, "context_cgroup",
                  f"budget_guard: pressure={report.pressure} total={report.total_chars} reclaimed={len(actions)}",
                  extra={"actions": [a.get("name") for a in actions if isinstance(a, dict)]})
        conn.commit()
        conn.close()
    except Exception:
        pass

    output = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": context_text,
        }
    }
    print(json.dumps(output, ensure_ascii=False))
    sys.exit(0)


if __name__ == "__main__":
    main()
