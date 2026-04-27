#!/usr/bin/env python3
"""
sched_hook.py — SubagentStart / SubagentStop Hook Wrapper

迭代 B2：将 sched/ 子系统挂载到 hooks 执行链。

OS 类比：fork()/exit() 时内核通知 scheduler 分配/释放调度实体。
  - SubagentStart → scheduler.submit(task) — 新 agent 注册到运行队列
  - SubagentStop  → scheduler.complete(task_id) — agent 退出，释放调度资源

输入（stdin JSON）：
  SubagentStart: {"session_id": "...", "subagent_id": "...", ...}
  SubagentStop:  {"session_id": "...", "subagent_id": "...", ...}

向后兼容：所有逻辑包裹在 try/except 中，失败不影响 hook 链。
"""

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


def _get_hook_event(hook_input: dict) -> str:
    """从 hook 输入推断事件类型"""
    # Claude Code hook 输入中可能包含 hookEventName
    event = hook_input.get("hookEventName", "")
    if event:
        return event
    # fallback: 从环境变量获取
    return os.environ.get("CLAUDE_HOOK_EVENT", "")


def _handle_subagent_start(hook_input: dict) -> None:
    """SubagentStart: 提交新 agent task 到调度器"""
    from sched.agent_scheduler import Scheduler, AgentTask, NiceLevel

    session_id = hook_input.get("session_id", "") or os.environ.get("CLAUDE_SESSION_ID", "")
    subagent_id = hook_input.get("subagent_id", "") or "unknown"
    # 从输入提取 agent 元信息（如果有）
    agent_name = hook_input.get("agent_name", "") or f"subagent-{subagent_id[:8]}"
    agent_type = hook_input.get("agent_type", "worker")
    # 默认 nice=0（普通优先级），调用方可通过输入覆盖
    nice = int(hook_input.get("nice", NiceLevel.NORMAL))

    sched = Scheduler()
    try:
        task = AgentTask(
            name=agent_name,
            agent_type=agent_type,
            nice=nice,
            session_id=session_id,
            project=hook_input.get("project", ""),
            cgroup_name=hook_input.get("cgroup", "foreground"),
        )
        # 将 subagent_id 存入 extra，便于 SubagentStop 时查找
        task.extra = {"subagent_id": subagent_id}
        submitted = sched.submit(task)

        # 将 task_id 写入临时映射文件，供 SubagentStop 查找
        _write_task_mapping(subagent_id, submitted.task_id)
    finally:
        sched.close()


def _handle_subagent_stop(hook_input: dict) -> None:
    """SubagentStop: 标记 agent task 为 completed"""
    from sched.agent_scheduler import Scheduler

    subagent_id = hook_input.get("subagent_id", "") or "unknown"
    tokens_used = int(hook_input.get("tokens_used", 0))

    # 从映射文件找到 task_id
    task_id = _read_task_mapping(subagent_id)
    if not task_id:
        return  # 没有对应的 task 记录，静默跳过

    sched = Scheduler()
    try:
        # 如果有 token 使用量，先更新 vruntime
        if tokens_used > 0:
            sched.update_vruntime(task_id, tokens_used)

        # 检查是否有错误
        error = hook_input.get("error", "")
        if error:
            sched.error(task_id, reason=str(error)[:200])
        else:
            sched.complete(task_id, tokens_final=0)

        # 清理映射
        _remove_task_mapping(subagent_id)
    finally:
        sched.close()


# ── Task ID 映射管理（subagent_id → task_id）─────────────────────────────
# 使用简单的 JSON 文件，避免引入额外依赖

_MAPPING_DIR = Path.home() / ".claude" / "memory-os" / ".sched_mappings"


def _write_task_mapping(subagent_id: str, task_id: str) -> None:
    _MAPPING_DIR.mkdir(parents=True, exist_ok=True)
    mapping_file = _MAPPING_DIR / f"{subagent_id}.json"
    mapping_file.write_text(json.dumps({"task_id": task_id}), encoding="utf-8")


def _read_task_mapping(subagent_id: str) -> str:
    mapping_file = _MAPPING_DIR / f"{subagent_id}.json"
    if not mapping_file.exists():
        return ""
    try:
        data = json.loads(mapping_file.read_text(encoding="utf-8"))
        return data.get("task_id", "")
    except Exception:
        return ""


def _remove_task_mapping(subagent_id: str) -> None:
    mapping_file = _MAPPING_DIR / f"{subagent_id}.json"
    try:
        mapping_file.unlink(missing_ok=True)
    except Exception:
        pass


def main():
    # 读取 stdin hook 输入
    try:
        raw = sys.stdin.read()
        hook_input = json.loads(raw) if raw.strip() else {}
    except Exception:
        hook_input = {}

    event = _get_hook_event(hook_input)

    try:
        if event == "SubagentStart" or "start" in sys.argv:
            _handle_subagent_start(hook_input)
        elif event == "SubagentStop" or "stop" in sys.argv:
            _handle_subagent_stop(hook_input)
        else:
            # 无法确定事件类型时，尝试从命令行参数判断
            if len(sys.argv) > 1:
                arg = sys.argv[1].lower()
                if "start" in arg:
                    _handle_subagent_start(hook_input)
                elif "stop" in arg:
                    _handle_subagent_stop(hook_input)
    except Exception:
        # 向后兼容：sched hook 失败不影响正常流程
        pass


if __name__ == "__main__":
    main()
