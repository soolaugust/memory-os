"""
agent_scheduler.py — Memory OS Agent 调度器核心

迭代 81：OS 类比 — Linux CFS (Completely Fair Scheduler, Ingo Molnár, 2007)

CFS 设计理念（来自 Documentation/scheduler/sched-design-CFS.rst）：
  "CFS basically models an 'ideal, precise multi-tasking CPU' on real hardware."
  - 核心数据结构：红黑树，按 vruntime 排序（最左节点 = 下一个调度的进程）
  - vruntime 计算：delta_exec * (NICE_0_WEIGHT / weight)
    高优先级进程权重大 → vruntime 增长慢 → 更频繁被调度
  - 公平性保证：所有进程的 vruntime 之差有上界（sched_latency_ns）

Agent 调度映射：
  - AgentTask  ≈ task_struct（进程描述符）
  - vruntime   ≈ se.vruntime（调度实体虚拟运行时间）
  - token_used ≈ sum(delta_exec)（累计 CPU 时间）
  - nice       ≈ p->static_prio - 120（nice 值，-20~19）
  - token_budget ≈ RLIMIT_CPU（进程 CPU 时间上限）
  - Scheduler.pick_next() ≈ __pick_first_entity()（选红黑树最左节点）

数据库模式（sched.db）：
  agent_tasks:
    task_id     TEXT PRIMARY KEY    — 任务唯一标识
    name        TEXT                — 任务名称
    agent_type  TEXT                — agent 类型（worker/planner/reviewer等）
    nice        INTEGER             — 优先级值（-20 ~ 19）
    vruntime    REAL                — 虚拟运行时间（公平性度量）
    token_used  INTEGER             — 已消耗 token 数
    token_budget INTEGER            — token 预算上限
    status      TEXT                — pending/running/completed/preempted/error
    cgroup_name TEXT                — 所属 cgroup 名
    created_at  TEXT                — ISO8601 UTC 时间戳
    started_at  TEXT                — 开始运行时间（可 NULL）
    ended_at    TEXT                — 结束时间（可 NULL）
    preempt_reason TEXT             — 抢占原因（可 NULL）
    session_id  TEXT                — 关联会话
    project     TEXT                — 关联项目
"""

import json
import sqlite3
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any
import os

# ── 环境配置（与 store_core.py 保持一致）─────────────────────────────────────
SCHED_OS_DIR = (
    Path(os.environ["MEMORY_OS_DIR"])
    if os.environ.get("MEMORY_OS_DIR")
    else Path.home() / ".claude" / "memory-os"
)
SCHED_DB_PATH = (
    Path(os.environ["SCHED_DB"])
    if os.environ.get("SCHED_DB")
    else SCHED_OS_DIR / "sched.db"
)

# ── CFS 权重表（对应 Linux sched_prio_to_weight[] in kernel/sched/core.c）────
# Linux 每 nice 值相差约 1.25 倍权重（NICE_0_WEIGHT = 1024）
NICE_WEIGHT_TABLE: Dict[int, int] = {
    -20: 88761, -19: 71755, -18: 56483, -17: 46273, -16: 36291,
    -15: 29154, -14: 23254, -13: 18705, -12: 14949, -11: 11916,
    -10: 9548,  -9: 7620,   -8: 6100,   -7: 4904,   -6: 3906,
    -5: 3121,   -4: 2501,   -3: 1991,   -2: 1586,   -1: 1277,
     0: 1024,    1: 820,     2: 655,     3: 526,     4: 423,
     5: 335,     6: 272,     7: 215,     8: 172,     9: 137,
    10: 110,    11: 87,     12: 70,     13: 56,     14: 45,
    15: 36,     16: 29,     17: 23,     18: 18,     19: 15,
}
NICE_0_WEIGHT = 1024  # nice=0 的标准权重


class NiceLevel(int, Enum):
    """
    Agent 优先级等级（对应 Linux nice 范围 -20 ~ 19）

    REALTIME: -20  实时任务（用户立即等待结果）
    HIGH:     -10  高优先级（前台重要任务）
    NORMAL:     0  默认优先级（普通任务）
    LOW:       10  低优先级（后台任务）
    IDLE:      19  空闲级（可被任何任务抢占）
    """
    REALTIME = -20
    HIGH = -10
    NORMAL = 0
    LOW = 10
    IDLE = 19


class TaskStatus(str, Enum):
    PENDING   = "pending"    # 已提交，等待调度
    RUNNING   = "running"    # 正在执行
    COMPLETED = "completed"  # 正常完成
    PREEMPTED = "preempted"  # 被抢占（超时/超预算/更高优先级）
    ERROR     = "error"      # 错误终止


# 默认 token 预算（按 nice 等级分档）
DEFAULT_TOKEN_BUDGET: Dict[int, int] = {
    NiceLevel.REALTIME: 50000,
    NiceLevel.HIGH:     30000,
    NiceLevel.NORMAL:   20000,
    NiceLevel.LOW:      10000,
    NiceLevel.IDLE:      5000,
}


def _default_budget(nice: int) -> int:
    """根据 nice 值返回默认 token 预算，找最近档位"""
    for level in [NiceLevel.REALTIME, NiceLevel.HIGH, NiceLevel.NORMAL,
                  NiceLevel.LOW, NiceLevel.IDLE]:
        if nice <= level.value:
            return DEFAULT_TOKEN_BUDGET[level]
    return DEFAULT_TOKEN_BUDGET[NiceLevel.IDLE]


@dataclass
class AgentTask:
    """
    Agent 任务描述符（类比 Linux task_struct）

    字段设计遵循 CFS 调度模型：
      vruntime 是公平性的核心度量，初始值设为当前调度队列中最小 vruntime
      （防止新任务被"饿死"，对应 CFS 的 place_entity() 逻辑）
    """
    name: str                              # 任务名称（e.g. "review_pr_agent"）
    agent_type: str = "worker"             # agent 类型
    nice: int = 0                          # nice 值（-20 ~ 19）
    token_budget: int = 0                  # token 预算（0 = 用 nice 默认值）
    session_id: str = ""                   # 关联会话
    project: str = ""                      # 关联项目
    cgroup_name: str = "foreground"        # 所属 cgroup

    # 以下字段由调度器管理，不由调用方设置
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    vruntime: float = 0.0                  # 虚拟运行时间（调度器维护）
    token_used: int = 0                    # 已消耗 token
    status: TaskStatus = TaskStatus.PENDING
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    preempt_reason: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None  # 附加元数据

    def __post_init__(self):
        # nice 值 clamp 到 [-20, 19]
        self.nice = max(-20, min(19, self.nice))
        # token_budget = 0 时用 nice 级别默认值
        if self.token_budget <= 0:
            self.token_budget = _default_budget(self.nice)

    @property
    def weight(self) -> int:
        """CFS 权重（从 NICE_WEIGHT_TABLE 查表）"""
        return NICE_WEIGHT_TABLE.get(self.nice, NICE_0_WEIGHT)

    @property
    def budget_remaining(self) -> int:
        return max(0, self.token_budget - self.token_used)

    @property
    def budget_pct(self) -> float:
        if self.token_budget <= 0:
            return 0.0
        return self.token_used / self.token_budget * 100

    def is_over_budget(self) -> bool:
        return self.token_used >= self.token_budget

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value if isinstance(self.status, TaskStatus) else self.status
        return d


def _open_sched_db(db_path: Path = None) -> sqlite3.Connection:
    """打开 sched.db，WAL 模式（与 store_core.open_db 保持一致）"""
    if db_path is None:
        db_path = SCHED_DB_PATH
    SCHED_OS_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_sched_schema(conn: sqlite3.Connection) -> None:
    """幂等建表（对应 ensure_schema 模式）"""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS agent_tasks (
            task_id       TEXT PRIMARY KEY,
            name          TEXT NOT NULL,
            agent_type    TEXT NOT NULL DEFAULT 'worker',
            nice          INTEGER NOT NULL DEFAULT 0,
            vruntime      REAL NOT NULL DEFAULT 0.0,
            token_used    INTEGER NOT NULL DEFAULT 0,
            token_budget  INTEGER NOT NULL DEFAULT 20000,
            status        TEXT NOT NULL DEFAULT 'pending',
            cgroup_name   TEXT NOT NULL DEFAULT 'foreground',
            created_at    TEXT NOT NULL,
            started_at    TEXT,
            ended_at      TEXT,
            preempt_reason TEXT,
            session_id    TEXT NOT NULL DEFAULT '',
            project       TEXT NOT NULL DEFAULT '',
            extra         TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_agent_tasks_status
            ON agent_tasks(status);
        CREATE INDEX IF NOT EXISTS idx_agent_tasks_vruntime
            ON agent_tasks(vruntime, status);
        CREATE INDEX IF NOT EXISTS idx_agent_tasks_cgroup
            ON agent_tasks(cgroup_name, status);

        -- 调度事件日志（类似 dmesg，用于可观测性）
        CREATE TABLE IF NOT EXISTS sched_events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            event_type  TEXT NOT NULL,   -- submit/pick/update/preempt/complete/error
            task_id     TEXT,
            task_name   TEXT,
            details     TEXT             -- JSON
        );
        CREATE INDEX IF NOT EXISTS idx_sched_events_task
            ON sched_events(task_id);
        CREATE INDEX IF NOT EXISTS idx_sched_events_time
            ON sched_events(timestamp DESC);
    """)
    conn.commit()


class Scheduler:
    """
    CFS-inspired Agent 调度器

    核心算法：
      1. submit(task)    — 将 task 加入运行队列，初始化 vruntime
      2. pick_next()     — 选取 vruntime 最小且 status=pending 的 task
      3. update_vruntime — 消耗 token 后更新 vruntime（delta / weight * NICE_0_WEIGHT）
      4. preempt()       — 标记任务为 preempted（超时/超预算）
      5. complete()      — 标记任务完成
      6. get_stats()     — 输出调度器统计（类似 /proc/sched_debug）

    数据库持久化：
      所有状态写入 sched.db agent_tasks 表，支持跨会话查询。
      调度事件写入 sched_events 表（环形缓冲，保留最新 1000 条）。

    与 store_core.dmesg_log 集成：
      重要调度事件（WARN/ERR）同步写入主 store.db 的 dmesg 表。
    """

    SCHED_EVENTS_MAX = 1000  # 调度事件环形缓冲大小

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or SCHED_DB_PATH
        self._conn = _open_sched_db(self.db_path)
        _ensure_sched_schema(self._conn)

    # ── 内部工具 ─────────────────────────────────────────────────────────────

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _min_vruntime(self) -> float:
        """
        返回当前运行队列中最小 vruntime（pending/running 状态）
        对应 CFS 的 update_curr() + min_vruntime 跟踪逻辑
        """
        row = self._conn.execute(
            "SELECT MIN(vruntime) FROM agent_tasks WHERE status IN ('pending','running')"
        ).fetchone()
        v = row[0]
        return v if v is not None else 0.0

    def _record_event(self, event_type: str, task_id: str = "",
                      task_name: str = "", details: dict = None) -> None:
        """
        写入调度事件（类似 printk 到 sched_events 环形缓冲）
        """
        self._conn.execute(
            """INSERT INTO sched_events (timestamp, event_type, task_id, task_name, details)
               VALUES (?, ?, ?, ?, ?)""",
            (
                self._now(), event_type, task_id, task_name,
                json.dumps(details, ensure_ascii=False) if details else None,
            ),
        )
        # 环形缓冲裁剪
        self._conn.execute(
            """DELETE FROM sched_events WHERE id NOT IN (
               SELECT id FROM sched_events ORDER BY id DESC LIMIT ?
            )""",
            (self.SCHED_EVENTS_MAX,),
        )
        self._conn.commit()

    def _dmesg_bridge(self, level: str, message: str,
                      extra: dict = None) -> None:
        """
        将重要调度事件桥接写入 store_core.dmesg 表。
        如果 store.db 不存在或 dmesg 表不存在则静默跳过（非阻塞）。
        """
        try:
            store_db = (
                Path(os.environ["MEMORY_OS_DB"])
                if os.environ.get("MEMORY_OS_DB")
                else SCHED_OS_DIR / "store.db"
            )
            if not store_db.exists():
                return
            store_conn = sqlite3.connect(str(store_db))
            # 检查 dmesg 表存在
            row = store_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='dmesg'"
            ).fetchone()
            if row is None:
                store_conn.close()
                return
            now_iso = self._now()
            extra_json = json.dumps(extra, ensure_ascii=False) if extra else None
            store_conn.execute(
                """INSERT INTO dmesg (timestamp, level, subsystem, message,
                   session_id, project, extra)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (now_iso, level, "sched", message[:500], "", "", extra_json),
            )
            store_conn.commit()
            store_conn.close()
        except Exception:
            pass  # 调度器写 dmesg 桥接不能阻断调度流程

    # ── 核心调度 API ──────────────────────────────────────────────────────────

    def submit(self, task: AgentTask) -> AgentTask:
        """
        提交新 agent 任务到运行队列。

        CFS place_entity() 等价操作：
          新任务 vruntime = max(task.vruntime, min_vruntime - sched_latency)
          防止新任务饿死老任务，也防止老任务长期被忽略。
          简化实现：新任务 vruntime = min_vruntime（加入时公平起点）
        """
        min_v = self._min_vruntime()
        # 新任务从当前最小 vruntime 出发（CFS place_entity 简化版）
        task.vruntime = max(task.vruntime, min_v)
        task.status = TaskStatus.PENDING

        self._conn.execute(
            """INSERT OR REPLACE INTO agent_tasks
               (task_id, name, agent_type, nice, vruntime, token_used,
                token_budget, status, cgroup_name, created_at, started_at,
                ended_at, preempt_reason, session_id, project, extra)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                task.task_id, task.name, task.agent_type, task.nice,
                task.vruntime, task.token_used, task.token_budget,
                task.status.value, task.cgroup_name, task.created_at,
                task.started_at, task.ended_at, task.preempt_reason,
                task.session_id, task.project,
                json.dumps(task.extra) if task.extra else None,
            ),
        )
        self._conn.commit()
        self._record_event("submit", task.task_id, task.name, {
            "nice": task.nice,
            "vruntime": task.vruntime,
            "token_budget": task.token_budget,
            "cgroup": task.cgroup_name,
        })
        return task

    def pick_next(self) -> Optional[AgentTask]:
        """
        选择下一个运行的 agent（vruntime 最小的 pending 任务）。
        对应 CFS __pick_first_entity()：红黑树最左节点。

        返回 None 表示无可调度任务。
        """
        row = self._conn.execute(
            """SELECT * FROM agent_tasks
               WHERE status = 'pending'
               ORDER BY vruntime ASC, created_at ASC
               LIMIT 1"""
        ).fetchone()
        if row is None:
            return None

        now = self._now()
        self._conn.execute(
            "UPDATE agent_tasks SET status=?, started_at=? WHERE task_id=?",
            (TaskStatus.RUNNING.value, now, row["task_id"]),
        )
        self._conn.commit()

        task = self._row_to_task(row)
        task.status = TaskStatus.RUNNING
        task.started_at = now
        self._record_event("pick", task.task_id, task.name, {
            "vruntime": task.vruntime,
            "nice": task.nice,
        })
        return task

    def update_vruntime(self, task_id: str, tokens_used: int) -> Optional[AgentTask]:
        """
        Agent 消耗 token 后更新 vruntime。

        CFS update_curr() 等价：
          delta_exec = tokens_used（此次增量，非累计）
          vruntime += delta_exec * (NICE_0_WEIGHT / weight)
          高优先级（大 weight）→ vruntime 增量小 → 下次更早被调度

        同时检查 token budget，超预算发出 WARN 并触发降级。
        """
        row = self._conn.execute(
            "SELECT * FROM agent_tasks WHERE task_id=?", (task_id,)
        ).fetchone()
        if row is None:
            return None

        task = self._row_to_task(row)
        weight = NICE_WEIGHT_TABLE.get(task.nice, NICE_0_WEIGHT)

        # CFS vruntime 公式（整数 tokens 映射到 float vruntime）
        delta_vruntime = tokens_used * (NICE_0_WEIGHT / weight)
        new_vruntime = task.vruntime + delta_vruntime
        new_token_used = task.token_used + tokens_used

        # 检查预算
        over_budget = new_token_used >= task.token_budget
        new_status = task.status

        if over_budget and task.status == TaskStatus.RUNNING:
            # 超预算 → 自动降级为 preempted（类似 cgroup cpu.max 到达上限）
            new_status = TaskStatus.PREEMPTED
            preempt_reason = f"token_budget_exceeded:{new_token_used}/{task.token_budget}"
            ended_at = self._now()
            self._conn.execute(
                """UPDATE agent_tasks
                   SET vruntime=?, token_used=?, status=?, preempt_reason=?, ended_at=?
                   WHERE task_id=?""",
                (new_vruntime, new_token_used, new_status.value,
                 preempt_reason, ended_at, task_id),
            )
            self._conn.commit()
            self._record_event("preempt", task_id, task.name, {
                "reason": "budget_exceeded",
                "token_used": new_token_used,
                "token_budget": task.token_budget,
            })
            self._dmesg_bridge("WARN",
                f"sched: task {task.name}({task_id[:8]}) exceeded budget "
                f"{new_token_used}/{task.token_budget} tokens",
                {"task_id": task_id, "nice": task.nice})
            task.status = new_status
            task.preempt_reason = preempt_reason
        else:
            self._conn.execute(
                "UPDATE agent_tasks SET vruntime=?, token_used=? WHERE task_id=?",
                (new_vruntime, new_token_used, task_id),
            )
            self._conn.commit()

        task.vruntime = new_vruntime
        task.token_used = new_token_used
        return task

    def preempt(self, task_id: str, reason: str = "") -> Optional[AgentTask]:
        """
        强制抢占正在运行或等待中的 task。
        对应 Linux resched_curr()（设置 TIF_NEED_RESCHED 标志）。

        场景：
          - 更高优先级任务提交时，抢占低优先级运行任务
          - 超时检测器检测到任务运行超时
          - 外部中断（用户取消、系统关闭）
        """
        row = self._conn.execute(
            "SELECT * FROM agent_tasks WHERE task_id=?", (task_id,)
        ).fetchone()
        if row is None:
            return None

        task = self._row_to_task(row)
        if task.status in (TaskStatus.COMPLETED, TaskStatus.ERROR):
            return task  # 已终态，不可抢占

        now = self._now()
        self._conn.execute(
            """UPDATE agent_tasks
               SET status=?, preempt_reason=?, ended_at=?
               WHERE task_id=?""",
            (TaskStatus.PREEMPTED.value, reason or "manual_preempt", now, task_id),
        )
        self._conn.commit()
        self._record_event("preempt", task_id, task.name, {
            "reason": reason,
            "token_used": task.token_used,
        })
        self._dmesg_bridge("WARN",
            f"sched: task {task.name}({task_id[:8]}) preempted: {reason}",
            {"task_id": task_id})
        task.status = TaskStatus.PREEMPTED
        task.preempt_reason = reason
        return task

    def complete(self, task_id: str,
                 tokens_final: int = 0) -> Optional[AgentTask]:
        """
        标记任务完成（正常退出）。
        对应 do_exit() → release 调度实体。
        """
        row = self._conn.execute(
            "SELECT * FROM agent_tasks WHERE task_id=?", (task_id,)
        ).fetchone()
        if row is None:
            return None

        task = self._row_to_task(row)
        now = self._now()
        final_tokens = task.token_used + tokens_final

        # 最后一次 vruntime 更新
        weight = NICE_WEIGHT_TABLE.get(task.nice, NICE_0_WEIGHT)
        final_vruntime = task.vruntime + tokens_final * (NICE_0_WEIGHT / weight)

        self._conn.execute(
            """UPDATE agent_tasks
               SET status=?, token_used=?, vruntime=?, ended_at=?
               WHERE task_id=?""",
            (TaskStatus.COMPLETED.value, final_tokens, final_vruntime, now, task_id),
        )
        self._conn.commit()
        self._record_event("complete", task_id, task.name, {
            "token_used": final_tokens,
            "vruntime": final_vruntime,
        })
        task.status = TaskStatus.COMPLETED
        task.token_used = final_tokens
        task.vruntime = final_vruntime
        task.ended_at = now
        return task

    def error(self, task_id: str, reason: str = "") -> Optional[AgentTask]:
        """标记任务异常终止"""
        row = self._conn.execute(
            "SELECT * FROM agent_tasks WHERE task_id=?", (task_id,)
        ).fetchone()
        if row is None:
            return None
        task = self._row_to_task(row)
        now = self._now()
        self._conn.execute(
            """UPDATE agent_tasks
               SET status=?, preempt_reason=?, ended_at=?
               WHERE task_id=?""",
            (TaskStatus.ERROR.value, reason, now, task_id),
        )
        self._conn.commit()
        self._record_event("error", task_id, task.name, {"reason": reason})
        self._dmesg_bridge("ERR",
            f"sched: task {task.name}({task_id[:8]}) error: {reason}",
            {"task_id": task_id})
        task.status = TaskStatus.ERROR
        return task

    # ── 查询接口 ──────────────────────────────────────────────────────────────

    def get_task(self, task_id: str) -> Optional[AgentTask]:
        row = self._conn.execute(
            "SELECT * FROM agent_tasks WHERE task_id=?", (task_id,)
        ).fetchone()
        return self._row_to_task(row) if row else None

    def list_tasks(self, status: Optional[TaskStatus] = None,
                   cgroup_name: str = None,
                   limit: int = 100) -> List[AgentTask]:
        """列出任务，支持按 status 和 cgroup 过滤"""
        where, params = [], []
        if status:
            where.append("status=?")
            params.append(status.value)
        if cgroup_name:
            where.append("cgroup_name=?")
            params.append(cgroup_name)
        sql = "SELECT * FROM agent_tasks"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY vruntime ASC, created_at ASC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_task(r) for r in rows]

    def get_stats(self) -> Dict[str, Any]:
        """
        调度器统计输出（类比 /proc/sched_debug 和 /proc/schedstat）

        返回字段：
          nr_pending     — 等待调度的任务数
          nr_running     — 正在运行的任务数
          nr_completed   — 已完成任务数
          nr_preempted   — 被抢占任务数
          nr_error       — 错误任务数
          min_vruntime   — 当前最小 vruntime（调度队列前沿）
          total_tokens   — 所有任务累计 token 消耗
          avg_vruntime   — 活跃任务平均 vruntime
          recent_events  — 最近 10 条调度事件
        """
        stats_rows = self._conn.execute("""
            SELECT
                SUM(CASE WHEN status='pending'   THEN 1 ELSE 0 END) AS nr_pending,
                SUM(CASE WHEN status='running'   THEN 1 ELSE 0 END) AS nr_running,
                SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) AS nr_completed,
                SUM(CASE WHEN status='preempted' THEN 1 ELSE 0 END) AS nr_preempted,
                SUM(CASE WHEN status='error'     THEN 1 ELSE 0 END) AS nr_error,
                SUM(token_used)                                       AS total_tokens,
                AVG(CASE WHEN status IN ('pending','running')
                    THEN vruntime END)                                AS avg_vruntime
            FROM agent_tasks
        """).fetchone()

        min_v = self._min_vruntime()

        events = self._conn.execute(
            """SELECT timestamp, event_type, task_id, task_name, details
               FROM sched_events ORDER BY id DESC LIMIT 10"""
        ).fetchall()
        recent = [
            {
                "timestamp": e["timestamp"],
                "event": e["event_type"],
                "task_id": (e["task_id"] or "")[:8],
                "task_name": e["task_name"],
                "details": json.loads(e["details"]) if e["details"] else None,
            }
            for e in events
        ]

        return {
            "nr_pending":   stats_rows["nr_pending"] or 0,
            "nr_running":   stats_rows["nr_running"] or 0,
            "nr_completed": stats_rows["nr_completed"] or 0,
            "nr_preempted": stats_rows["nr_preempted"] or 0,
            "nr_error":     stats_rows["nr_error"] or 0,
            "total_tasks":  sum([
                stats_rows["nr_pending"] or 0,
                stats_rows["nr_running"] or 0,
                stats_rows["nr_completed"] or 0,
                stats_rows["nr_preempted"] or 0,
                stats_rows["nr_error"] or 0,
            ]),
            "total_tokens":  stats_rows["total_tokens"] or 0,
            "min_vruntime":  min_v,
            "avg_vruntime":  stats_rows["avg_vruntime"],
            "recent_events": recent,
        }

    def close(self) -> None:
        self._conn.close()

    # ── 私有转换 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_task(row: sqlite3.Row) -> AgentTask:
        extra = None
        if row["extra"]:
            try:
                extra = json.loads(row["extra"])
            except Exception:
                pass
        t = AgentTask(
            name=row["name"],
            agent_type=row["agent_type"],
            nice=row["nice"],
            token_budget=row["token_budget"],
            session_id=row["session_id"] or "",
            project=row["project"] or "",
            cgroup_name=row["cgroup_name"],
            task_id=row["task_id"],
            vruntime=row["vruntime"],
            token_used=row["token_used"],
            status=TaskStatus(row["status"]),
            created_at=row["created_at"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            preempt_reason=row["preempt_reason"],
            extra=extra,
        )
        return t
