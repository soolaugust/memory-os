"""
EXPERIMENTAL — 未接入生产。无 hook 注册，无实际调用者。
保留供未来 agent 调度需求参考。

sched/ — Memory OS Agent 调度器子系统

迭代 81：OS 类比 — Linux CFS (Completely Fair Scheduler, Ingo Molnár, 2007)

Linux CFS 核心思想：
  - 每个进程维护 vruntime（虚拟运行时间），调度器总是选择 vruntime 最小的进程
  - 通过 nice 值调整权重，高优先级进程的 vruntime 增长更慢
  - cgroup CPU quota 为一组进程分配整体 CPU 时间上限

映射到 Agent 调度：
  - Agent = Process — 每个 spawn 的 agent 是一个"进程"
  - Token 消耗 = CPU 时间 — agent 消耗的 token 是它的"运行时间"
  - vruntime = 已消耗 token / 权重 — 公平性度量
  - nice 值 = 优先级 — 高优先级 agent 获得更多 token 预算
  - cgroup = team — 一个 team 下的所有 agent 共享 token 配额

模块：
  agent_scheduler.py — 核心调度器（CFS vruntime + 红黑树调度）
  agent_cgroup.py   — 资源组管理（类似 cgroup v2 cpu.max）
  agent_monitor.py  — 运行时监控（类似 /proc/sched_debug）
"""

from .agent_scheduler import Scheduler, AgentTask, TaskStatus, NiceLevel
from .agent_cgroup import CGroupManager, CGroup
from .agent_monitor import AgentMonitor

__all__ = [
    "Scheduler", "AgentTask", "TaskStatus", "NiceLevel",
    "CGroupManager", "CGroup",
    "AgentMonitor",
]
