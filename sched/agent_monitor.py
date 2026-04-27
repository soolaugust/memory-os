"""
agent_monitor.py — Memory OS Agent 运行时监控

迭代 81：OS 类比 — Linux /proc/sched_debug + schedstat (2.6.23, 2007)

Linux 调度监控接口：
  /proc/sched_debug  — 每个 CPU 的调度域状态、运行队列统计、各进程 vruntime
  /proc/schedstat    — 每 CPU 调度统计（调度次数、等待时间、运行时间）
  /proc/<pid>/sched  — 单个进程调度信息（nr_voluntary_switches, se.sum_exec_runtime 等）
  top/htop           — 实时调度状态可视化

Memory OS 映射：
  AgentMonitor.sched_debug()    ≈ cat /proc/sched_debug
  AgentMonitor.proc_task()      ≈ cat /proc/<pid>/sched
  AgentMonitor.detect_timeouts()≈ watchdog/hung_task_timeout_secs
  AgentMonitor.summary()        ≈ 综合 top 视图

超时检测（类比 hung_task_timeout_secs）：
  Linux 默认 120s 未调度的任务触发 WARN（kernel/hung_task.c）。
  Memory OS：任务 running 状态超过 timeout_tokens（token 估算运行时间）
  或超过 timeout_seconds 实际时间，触发自动 preempt。
"""

import json
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

from .agent_scheduler import (
    Scheduler, AgentTask, TaskStatus,
    SCHED_DB_PATH, _open_sched_db,
)
from .agent_cgroup import CGroupManager


# 默认超时阈值
DEFAULT_TIMEOUT_SECONDS = 300       # 5 分钟未完成的 running 任务触发超时
DEFAULT_MAX_TOKEN_THRESHOLD = 50000 # 超过此 token 消耗量发出 WARN（非抢占）


class AgentMonitor:
    """
    Agent 调度运行时监控器

    主要职责：
      1. sched_debug()      — 全局调度状态快照（类 /proc/sched_debug）
      2. proc_task(task_id) — 单任务详细状态（类 /proc/<pid>/sched）
      3. detect_timeouts()  — 超时检测并自动 preempt（类 hung_task watchdog）
      4. summary()          — 人类可读的 top 视图
      5. cgroup_stats()     — cgroup 资源使用统计

    与 Scheduler/CGroupManager 的关系：
      AgentMonitor 持有独立的只读连接用于监控查询，
      超时 preempt 通过调用 Scheduler.preempt() 执行（写操作）。
    """

    def __init__(self, db_path: Path = None,
                 timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
                 max_token_warn: int = DEFAULT_MAX_TOKEN_THRESHOLD):
        self.db_path = db_path or SCHED_DB_PATH
        self._conn = _open_sched_db(self.db_path)
        self.timeout_seconds = timeout_seconds
        self.max_token_warn = max_token_warn
        # Scheduler 实例（用于 preempt 写操作）
        self._scheduler = Scheduler(db_path=self.db_path)
        self._cgroup_mgr = CGroupManager(db_path=self.db_path)

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    def _now_iso(self) -> str:
        return self._now().isoformat()

    # ── /proc/sched_debug 等价 ─────────────────────────────────────────────

    def sched_debug(self) -> Dict[str, Any]:
        """
        全局调度状态快照。
        类比 cat /proc/sched_debug — 输出每个运行队列的状态。

        返回结构：
          scheduler_stats:  调度器全局统计（来自 Scheduler.get_stats()）
          runqueue:         按 cgroup 分组的运行队列视图
          cgroup_stats:     各 cgroup token 配额使用情况
          top_consumers:    token 消耗最多的前 10 任务
          recent_preempts:  最近 5 次抢占记录
        """
        stats = self._scheduler.get_stats()

        # 按 cgroup 分组的运行队列
        runqueue = self._build_runqueue_view()

        # cgroup 统计
        cgroups = self._cgroup_mgr.list_cgroups()
        cgroup_stats = [cg.to_dict() for cg in cgroups]

        # token 消耗排行
        top_rows = self._conn.execute(
            """SELECT task_id, name, agent_type, nice, vruntime, token_used,
                      token_budget, status, cgroup_name
               FROM agent_tasks
               ORDER BY token_used DESC LIMIT 10"""
        ).fetchall()
        top_consumers = [
            {
                "task_id": r["task_id"][:8],
                "name": r["name"],
                "nice": r["nice"],
                "vruntime": round(r["vruntime"], 2),
                "token_used": r["token_used"],
                "token_budget": r["token_budget"],
                "budget_pct": round(r["token_used"] / r["token_budget"] * 100, 1)
                              if r["token_budget"] > 0 else 0.0,
                "status": r["status"],
                "cgroup": r["cgroup_name"],
            }
            for r in top_rows
        ]

        # 最近抢占
        preempt_rows = self._conn.execute(
            """SELECT task_id, name, token_used, preempt_reason, ended_at
               FROM agent_tasks
               WHERE status='preempted'
               ORDER BY ended_at DESC NULLS LAST LIMIT 5"""
        ).fetchall()
        recent_preempts = [
            {
                "task_id": r["task_id"][:8],
                "name": r["name"],
                "token_used": r["token_used"],
                "reason": r["preempt_reason"],
                "ended_at": r["ended_at"],
            }
            for r in preempt_rows
        ]

        return {
            "timestamp": self._now_iso(),
            "scheduler_stats": stats,
            "runqueue": runqueue,
            "cgroup_stats": cgroup_stats,
            "top_consumers": top_consumers,
            "recent_preempts": recent_preempts,
        }

    def proc_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        单任务详细状态。
        类比 cat /proc/<pid>/sched — 输出该进程的调度统计。

        返回字段（对应 Linux /proc/<pid>/sched 输出）：
          se.sum_exec_runtime  ≈ token_used（累计 CPU 时间）
          se.vruntime          ≈ vruntime
          nr_voluntary_switches ≈ preempt 次数（简化：0 或 1）
          prio                 ≈ nice + 120（Linux 内部优先级）
          wait_start           ≈ created_at（等待开始时间）
          exec_start           ≈ started_at（执行开始时间）
        """
        row = self._conn.execute(
            "SELECT * FROM agent_tasks WHERE task_id=?", (task_id,)
        ).fetchone()
        if row is None:
            return None

        # 计算等待时间和运行时间
        now = self._now()
        created = datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
        wait_seconds = None
        run_seconds = None

        if row["started_at"]:
            started = datetime.fromisoformat(row["started_at"].replace("Z", "+00:00"))
            wait_seconds = (started - created).total_seconds()
            if row["ended_at"]:
                ended = datetime.fromisoformat(row["ended_at"].replace("Z", "+00:00"))
                run_seconds = (ended - started).total_seconds()
            elif row["status"] == "running":
                run_seconds = (now - started).total_seconds()
        else:
            wait_seconds = (now - created).total_seconds()

        prio = row["nice"] + 120  # Linux 内部优先级（NORMAL=120）

        return {
            "task_id": row["task_id"],
            "name": row["name"],
            "agent_type": row["agent_type"],
            "status": row["status"],
            "cgroup": row["cgroup_name"],
            "session_id": row["session_id"],
            "project": row["project"],
            # CFS 调度字段
            "se.vruntime": round(row["vruntime"], 4),
            "se.sum_exec_tokens": row["token_used"],  # ≈ sum_exec_runtime
            "token_budget": row["token_budget"],
            "budget_pct": round(row["token_used"] / row["token_budget"] * 100, 1)
                          if row["token_budget"] > 0 else 0.0,
            "nice": row["nice"],
            "prio": prio,
            # 时间统计
            "wait_seconds": round(wait_seconds, 2) if wait_seconds is not None else None,
            "run_seconds": round(run_seconds, 2) if run_seconds is not None else None,
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "ended_at": row["ended_at"],
            "preempt_reason": row["preempt_reason"],
        }

    def cgroup_stats(self) -> List[Dict[str, Any]]:
        """
        各 cgroup 资源使用统计（类比 cat /sys/fs/cgroup/*/cpu.stat）
        """
        cgroups = self._cgroup_mgr.list_cgroups()
        result = []
        for cg in cgroups:
            # 统计该 cgroup 内各状态任务数
            rows = self._conn.execute(
                """SELECT status, COUNT(*) as cnt, SUM(token_used) as tokens
                   FROM agent_tasks WHERE cgroup_name=?
                   GROUP BY status""",
                (cg.name,),
            ).fetchall()
            status_breakdown = {r["status"]: r["cnt"] for r in rows}
            total_tokens_tasks = sum(r["tokens"] or 0 for r in rows)

            result.append({
                **cg.to_dict(),
                "task_breakdown": status_breakdown,
                "total_tokens_tasks": total_tokens_tasks,
                "agent_count_active": self._cgroup_mgr.get_agent_count(
                    cg.name, active_only=True
                ),
                "agent_count_total": self._cgroup_mgr.get_agent_count(cg.name),
            })
        return result

    # ── 超时检测（hung_task watchdog 等价）──────────────────────────────────

    def detect_timeouts(self) -> List[Dict[str, Any]]:
        """
        检测并处理超时任务。
        类比 Linux kernel/hung_task.c — check_hung_uninterruptible_tasks()

        策略：
          1. 查找 status=running 且 started_at 超过 timeout_seconds 的任务
          2. 对超时任务调用 Scheduler.preempt(reason="timeout")
          3. 同时检查 token_used > max_token_warn 的任务发出 WARN

        返回：超时被抢占的任务列表
        """
        now = self._now()
        cutoff = (now - timedelta(seconds=self.timeout_seconds)).isoformat()

        # 查找超时的 running 任务
        timeout_rows = self._conn.execute(
            """SELECT * FROM agent_tasks
               WHERE status='running'
                 AND started_at IS NOT NULL
                 AND started_at < ?""",
            (cutoff,),
        ).fetchall()

        preempted = []
        for row in timeout_rows:
            started = datetime.fromisoformat(
                row["started_at"].replace("Z", "+00:00")
            )
            elapsed = (now - started).total_seconds()
            task = self._scheduler.preempt(
                row["task_id"],
                reason=f"timeout:{elapsed:.0f}s>{self.timeout_seconds}s",
            )
            if task:
                preempted.append({
                    "task_id": row["task_id"][:8],
                    "name": row["name"],
                    "elapsed_seconds": round(elapsed, 1),
                    "token_used": row["token_used"],
                    "reason": f"timeout after {elapsed:.0f}s",
                })

        # 检查高 token 消耗但未超时的任务（WARN，不 preempt）
        warn_rows = self._conn.execute(
            """SELECT task_id, name, token_used, token_budget FROM agent_tasks
               WHERE status='running'
                 AND token_used > ?""",
            (self.max_token_warn,),
        ).fetchall()
        for row in warn_rows:
            # 通过 dmesg_bridge 发出警告（非阻塞）
            self._scheduler._dmesg_bridge(
                "WARN",
                f"sched: task {row['name']}({row['task_id'][:8]}) "
                f"high token usage {row['token_used']}/{row['token_budget']}",
                {"task_id": row["task_id"], "token_used": row["token_used"]},
            )

        return preempted

    # ── 可读摘要（top 视图）──────────────────────────────────────────────────

    def summary(self) -> str:
        """
        人类可读的调度状态摘要（类比 top 命令输出）。

        格式：
          == Memory OS Agent Scheduler (sched_debug) ==
          Tasks: N total, P pending, R running, C completed, X preempted, E error
          Tokens: N total consumed | min_vruntime: V

          CGroup         Quota      Used    Pct   Active
          foreground  100000    12345   12.3%     3/5
          background   50000     8000   16.0%     2/10

          Running Queue (sorted by vruntime):
          [vruntime]  [name]            [nice] [tokens] [budget%]
          1234.56     review_pr_agent      0    5000     25.0%
          ...
        """
        stats = self._scheduler.get_stats()
        cgroups = self._cgroup_mgr.list_cgroups()

        lines = [
            "== Memory OS Agent Scheduler ==",
            f"Timestamp: {self._now_iso()}",
            f"Tasks: {stats['total_tasks']} total | "
            f"{stats['nr_pending']} pending | "
            f"{stats['nr_running']} running | "
            f"{stats['nr_completed']} completed | "
            f"{stats['nr_preempted']} preempted | "
            f"{stats['nr_error']} error",
            f"Tokens: {stats['total_tokens']} total | "
            f"min_vruntime: {stats['min_vruntime']:.2f}",
            "",
            f"{'CGroup':<16} {'Quota':>10} {'Used':>10} {'Pct':>8} {'Active':>10}",
            "-" * 58,
        ]
        for cg in cgroups:
            quota_str = str(cg.token_quota) if cg.token_quota >= 0 else "unlimited"
            active = self._cgroup_mgr.get_agent_count(cg.name, active_only=True)
            max_str = str(cg.max_agents) if cg.max_agents >= 0 else "∞"
            pct_str = f"{cg.quota_pct:.1f}%" if cg.token_quota >= 0 else "n/a"
            lines.append(
                f"{cg.name:<16} {quota_str:>10} {cg.token_used:>10} "
                f"{pct_str:>8} {active:>4}/{max_str:<5}"
            )

        # 运行队列（pending + running，按 vruntime 排序）
        queue_rows = self._conn.execute(
            """SELECT name, nice, vruntime, token_used, token_budget, status, cgroup_name
               FROM agent_tasks
               WHERE status IN ('pending', 'running')
               ORDER BY vruntime ASC LIMIT 20"""
        ).fetchall()

        if queue_rows:
            lines.extend([
                "",
                "Running Queue (sorted by vruntime):",
                f"{'vruntime':>12} {'status':>10} {'name':<24} {'nice':>5} "
                f"{'tokens':>8} {'budget%':>8} {'cgroup':<12}",
                "-" * 84,
            ])
            for r in queue_rows:
                pct = r["token_used"] / r["token_budget"] * 100 if r["token_budget"] > 0 else 0
                lines.append(
                    f"{r['vruntime']:>12.2f} {r['status']:>10} {r['name']:<24} "
                    f"{r['nice']:>5} {r['token_used']:>8} {pct:>7.1f}% "
                    f"{r['cgroup_name']:<12}"
                )
        else:
            lines.append("\n(no pending/running tasks)")

        return "\n".join(lines)

    def close(self) -> None:
        self._conn.close()
        self._scheduler.close()
        self._cgroup_mgr.close()

    # ── 私有 ─────────────────────────────────────────────────────────────────

    def _build_runqueue_view(self) -> Dict[str, List[Dict]]:
        """
        按 cgroup 构建运行队列视图（类比 /proc/sched_debug 中每 CPU 的 runqueue）
        """
        rows = self._conn.execute(
            """SELECT cgroup_name, task_id, name, nice, vruntime,
                      token_used, token_budget, status
               FROM agent_tasks
               WHERE status IN ('pending', 'running')
               ORDER BY cgroup_name, vruntime ASC"""
        ).fetchall()

        runqueue: Dict[str, List[Dict]] = {}
        for r in rows:
            cg = r["cgroup_name"]
            if cg not in runqueue:
                runqueue[cg] = []
            runqueue[cg].append({
                "task_id": r["task_id"][:8],
                "name": r["name"],
                "nice": r["nice"],
                "vruntime": round(r["vruntime"], 2),
                "token_used": r["token_used"],
                "budget_pct": round(r["token_used"] / r["token_budget"] * 100, 1)
                              if r["token_budget"] > 0 else 0.0,
                "status": r["status"],
            })
        return runqueue
