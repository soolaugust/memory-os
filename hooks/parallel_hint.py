#!/usr/bin/env python3
"""
memory-os parallel_hint — UserPromptSubmit hook
迭代110 P4: Multi-Agent Parallel Scheduling Hint

OS 类比: Linux CFS Work-Stealing Scheduler (2007) + fork-exec 并行
  当 CPU 空闲时，从其他 CPU 的运行队列"偷取"就绪任务并行执行。
  目标：消除顺序等待，最大化 CPU 利用率。

AIOS 类比: 检测用户 prompt 中的独立并行子任务，
  注入 additionalContext 提示 Claude 使用 Agent tool 并行执行，
  而非默认的顺序串行处理。

检测信号（3 类）：
  P0 显式并行请求："分别"/"各自"/"同时"/"并行"/"一起" + 多对象
  P1 列表型任务：编号列表（1. 2. 3.）或破折号列表中的 3+ 独立项
  P2 比较型任务："A 和 B 分别"/"对比 X 和 Y"/"分析 X、Y、Z"

注入策略：
  - 只在检测到 2+ 独立可并行任务时才注入（避免噪音）
  - 注入一条简短提示："[CFS] 检测到 N 个独立子任务，可用 Agent tool 并行执行"
  - 如果任务明显串行（含"然后"/"之后"/"先...再"），跳过注入

约束：
  - 不修改用户 prompt（hooks 不支持）
  - additionalContext 最多 200 字
  - 全程 try/except，失败不影响 UserPromptSubmit
"""

import sys
import json
import re

MAX_NOTICE_LEN = 200

# ── 串行依赖信号（存在则不建议并行）──────────────────────────
SERIAL_SIGNALS = re.compile(
    r'(?:然后|之后|接着|再(?:去|做|看|分析|检查)|先.*?(?:再|然后)|完成.*?(?:再|然后)|等.*?(?:再|完成))',
    re.IGNORECASE
)

# ── P0: 显式并行词 + 多对象 ────────────────────────────────
PARALLEL_EXPLICIT = re.compile(
    r'(?:分别|各自|同时|并行|一并|同步)(?:.{0,30}(?:和|与|、|，|,))',
    re.IGNORECASE
)

# ── P1: 编号列表（3+ 项）──────────────────────────────────
NUMBERED_LIST = re.compile(
    r'(?:^|\n)\s*(?:[①②③④⑤]|\d+[.。、）)]\s+.{5,})',
    re.MULTILINE
)

# ── P2: 对比型任务 ────────────────────────────────────────
COMPARISON = re.compile(
    r'(?:对比|比较|分析)\s*.{2,20}(?:和|与|、)\s*.{2,20}'
    r'|.{2,20}(?:和|与|、).{2,20}(?:分别|各自|哪个|哪种)',
    re.IGNORECASE
)

# ── P3: 枚举对象（3+ 个用顿号/逗号连接的项）────────────────
ENUM_OBJECTS = re.compile(
    r'(?:[^\n，,、]{2,15}(?:[，,、])){2,}[^\n，,、]{2,15}',
    re.IGNORECASE
)


def _count_parallel_signals(text: str) -> tuple[int, list[str]]:
    """
    检测 prompt 中的并行化信号。
    返回 (signal_count, reasons) — signal_count >= 2 时建议注入。
    """
    # 只看前 500 字（prompt 通常较短，避免分析全文）
    sample = text[:500]
    signals = []

    # 串行依赖短路：有明显依赖关系时直接返回 0
    if SERIAL_SIGNALS.search(sample):
        # 只有 1 处串行信号时还允许通过（可能只是部分串行）
        serial_count = len(SERIAL_SIGNALS.findall(sample))
        if serial_count >= 2:
            return 0, []

    # P0: 显式并行词
    if PARALLEL_EXPLICIT.search(sample):
        signals.append("显式并行")

    # P1: 编号列表（需要 3+ 项）
    numbered_items = NUMBERED_LIST.findall(text[:1000])
    if len(numbered_items) >= 3:
        signals.append(f"列表{len(numbered_items)}项")

    # P2: 对比型
    if COMPARISON.search(sample):
        signals.append("对比分析")

    # P3: 枚举 3+ 对象
    enum_matches = ENUM_OBJECTS.findall(sample)
    if enum_matches:
        # 验证枚举中的项确实是独立对象（非数字序列）
        for m in enum_matches:
            parts = re.split(r'[，,、]', m)
            if len(parts) >= 3 and all(2 <= len(p.strip()) <= 15 for p in parts):
                signals.append(f"枚举{len(parts)}个对象")
                break

    return len(signals), signals


def _extract_task_count(text: str) -> int:
    """粗略估计独立子任务数量。"""
    # 编号列表最准确
    numbered = NUMBERED_LIST.findall(text[:1000])
    if len(numbered) >= 2:
        return len(numbered)
    # 顿号枚举
    enum_matches = ENUM_OBJECTS.findall(text[:500])
    for m in enum_matches:
        parts = re.split(r'[，,、]', m)
        if len(parts) >= 2:
            return len(parts)
    return 2  # 默认 2


def main():
    try:
        raw = sys.stdin.read()
        hook_input = json.loads(raw) if raw.strip() else {}
    except Exception:
        sys.exit(0)

    prompt = hook_input.get("prompt", "")
    if not prompt or len(prompt) < 10:
        sys.exit(0)

    try:
        signal_count, reasons = _count_parallel_signals(prompt)
        if signal_count < 2:
            sys.exit(0)

        task_count = _extract_task_count(prompt)
        reason_str = "、".join(reasons[:2])

        notice = (
            f"[CFS] 检测到 {task_count} 个独立子任务（{reason_str}）。"
            f"可用 Agent tool 并行执行，消除顺序等待。"
            f"注意：只在任务间无数据依赖时并行。"
        )

        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": notice[:MAX_NOTICE_LEN],
            }
        }, ensure_ascii=False))

    except Exception:
        pass  # 永远不阻塞用户输入

    sys.exit(0)


if __name__ == "__main__":
    main()
