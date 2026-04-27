"""
agent_cgroup.py — Memory OS Agent 资源组管理

迭代 81：OS 类比 — Linux cgroup v2 (Unified Hierarchy, Tejun Heo, 2015)

cgroup v2 cpu 子系统关键接口：
  cpu.max        — "quota period" 配置（e.g., "50000 100000" = 50ms/100ms）
  cpu.weight     — 相对权重（取代 v1 的 cpu.shares）
  cpu.stat       — 统计输出（usage_usec, user_usec, throttled_usec 等）

Memory OS 映射：
  CGroup.token_quota  ≈ cpu.max quota    — 组内所有 agent 共享的 token 上限
  CGroup.token_period ≈ cpu.max period   — 配额重置周期（未来扩展）
  CGroup.weight       ≈ cpu.weight       — 组间调度权重
  check_quota()       ≈ throttle_cfs_rq() — 检查是否达到配额上限

预定义 cgroup：
  foreground — 用户交互型 agent（quota=100000, weight=1024）
  background — 后台异步 agent（quota=50000, weight=256）
  system     — 系统维护 agent（quota=20000, weight=128）

数据库模式（sched.db agent_cgroups 表）：
  cgroup_name  TEXT PRIMARY KEY  — 资源组名称
  token_quota  INTEGER           — token 配额上限（-1 = 无限）
  token_used   INTEGER           — 当前周期已消耗
  weight       INTEGER           — 调度权重（影响组间 vruntime 计算）
  max_agents   INTEGER           — 最大并发 agent 数（-1 = 无限）
  created_at   TEXT              — ISO8601 UTC
  description  TEXT              — 描述
"""

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any
import os

from .agent_scheduler import SCHED_DB_PATH, SCHED_OS_DIR, _open_sched_db


@dataclass
class CGroup:
    """
    资源组描述符（类比 Linux struct cgroup + css_set）

    token_quota = -1 表示无配额限制（类似 cgroup cpu.max = "max 100000"）
    max_agents  = -1 表示无并发 agent 数限制
    """
    name: str
    token_quota: int = -1            # -1 = 无限
    token_used: int = 0              # 当前已用 token（跨活跃任务累计）
    weight: int = 1024               # 组间调度权重（NICE_0_WEIGHT = 1024）
    max_agents: int = -1             # 最大并发 agent 数（-1 = 无限）
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    description: str = ""

    @property
    def quota_remaining(self) -> int:
        if self.token_quota < 0:
            return -1  # 无限
        return max(0, self.token_quota - self.token_used)

    @property
    def quota_pct(self) -> float:
        if self.token_quota <= 0:
            return 0.0
        return self.token_used / self.token_quota * 100

    def is_throttled(self) -> bool:
        """是否已达到配额上限（类比 throttle_cfs_rq）"""
        return self.token_quota >= 0 and self.token_used >= self.token_quota

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "token_quota": self.token_quota,
            "token_used": self.token_used,
            "weight": self.weight,
            "max_agents": self.max_agents,
            "created_at": self.created_at,
            "description": self.description,
            "quota_pct": self.quota_pct,
            "is_throttled": self.is_throttled(),
        }


# 预定义 cgroup 配置（在 CGroupManager.__init__ 中自动创建）
BUILTIN_CGROUPS = [
    CGroup(
        name="foreground",
        token_quota=100_000,
        weight=1024,
        max_agents=5,
        description="用户交互型 agent（实时响应）",
    ),
    CGroup(
        name="background",
        token_quota=50_000,
        weight=256,
        max_agents=10,
        description="后台异步 agent（低优先级批量任务）",
    ),
    CGroup(
        name="system",
        token_quota=20_000,
        weight=128,
        max_agents=3,
        description="系统维护 agent（memory GC / index rebuild）",
    ),
    CGroup(
        name="unlimited",
        token_quota=-1,
        weight=512,
        max_agents=-1,
        description="无配额限制组（超级任务，需显式指定）",
    ),
]


def _ensure_cgroup_schema(conn: sqlite3.Connection) -> None:
    """幂等建 cgroup 和 agent_tasks 表（agent_tasks 由 Scheduler 主建，这里做 fallback）"""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS agent_cgroups (
            cgroup_name  TEXT PRIMARY KEY,
            token_quota  INTEGER NOT NULL DEFAULT -1,
            token_used   INTEGER NOT NULL DEFAULT 0,
            weight       INTEGER NOT NULL DEFAULT 1024,
            max_agents   INTEGER NOT NULL DEFAULT -1,
            created_at   TEXT NOT NULL,
            description  TEXT NOT NULL DEFAULT ''
        );

        -- agent_tasks 在 Scheduler 中建立，CGroupManager 也需要读取它
        -- 如果 Scheduler 先创建则此处为 no-op
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
    """)
    conn.commit()


class CGroupManager:
    """
    cgroup 资源组管理器

    职责：
      1. create_cgroup()    — 创建资源组（幂等）
      2. get_cgroup()       — 查询资源组
      3. add_to_cgroup()    — 将 agent task 归入某 cgroup
      4. check_quota()      — 检查 cgroup 是否已达配额（调度器提交前调用）
      5. charge_tokens()    — 扣减 cgroup token 配额（与 update_vruntime 配套）
      6. reset_quota()      — 重置 token_used（周期性调用，类似 cgroup 配额周期）
      7. get_agent_count()  — 查询 cgroup 内当前活跃 agent 数量
      8. list_cgroups()     — 列出所有 cgroup

    与 Scheduler 的协作：
      Scheduler.submit() 前调用 check_quota()
      Scheduler.update_vruntime() 后调用 charge_tokens()
      两者共享同一个 sched.db 连接（同进程）或各自连接（跨进程）
    """

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or SCHED_DB_PATH
        self._conn = _open_sched_db(self.db_path)
        _ensure_cgroup_schema(self._conn)
        self._init_builtins()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _init_builtins(self) -> None:
        """创建内置 cgroup（幂等，已存在则跳过）"""
        for cg in BUILTIN_CGROUPS:
            self._conn.execute(
                """INSERT OR IGNORE INTO agent_cgroups
                   (cgroup_name, token_quota, token_used, weight,
                    max_agents, created_at, description)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (cg.name, cg.token_quota, cg.token_used, cg.weight,
                 cg.max_agents, cg.created_at, cg.description),
            )
        self._conn.commit()

    # ── 核心 API ──────────────────────────────────────────────────────────────

    def create_cgroup(self, name: str, token_quota: int = -1,
                      weight: int = 1024, max_agents: int = -1,
                      description: str = "") -> CGroup:
        """
        创建资源组（幂等：若已存在则返回现有）
        类比 mkdir /sys/fs/cgroup/<name>
        """
        existing = self.get_cgroup(name)
        if existing:
            return existing

        cg = CGroup(
            name=name,
            token_quota=token_quota,
            weight=weight,
            max_agents=max_agents,
            description=description,
        )
        self._conn.execute(
            """INSERT INTO agent_cgroups
               (cgroup_name, token_quota, token_used, weight,
                max_agents, created_at, description)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (cg.name, cg.token_quota, cg.token_used, cg.weight,
             cg.max_agents, cg.created_at, cg.description),
        )
        self._conn.commit()
        return cg

    def get_cgroup(self, name: str) -> Optional[CGroup]:
        """查询 cgroup（类比 cat /sys/fs/cgroup/<name>/cpu.max）"""
        row = self._conn.execute(
            "SELECT * FROM agent_cgroups WHERE cgroup_name=?", (name,)
        ).fetchone()
        return self._row_to_cgroup(row) if row else None

    def add_to_cgroup(self, task_id: str, cgroup_name: str) -> bool:
        """
        将 agent task 归入 cgroup。
        类比 echo <pid> > /sys/fs/cgroup/<name>/cgroup.procs

        返回 True = 成功，False = cgroup 不存在或 agent 不存在
        """
        cg = self.get_cgroup(cgroup_name)
        if cg is None:
            return False

        result = self._conn.execute(
            "UPDATE agent_tasks SET cgroup_name=? WHERE task_id=?",
            (cgroup_name, task_id),
        )
        self._conn.commit()
        return result.rowcount > 0

    def check_quota(self, cgroup_name: str,
                    new_agent: bool = False) -> Dict[str, Any]:
        """
        检查 cgroup 是否允许新任务提交。
        类比 throttle_cfs_rq() + cgroup cpu bandwidth 检查。

        返回：
          allowed: bool         — True = 可提交
          reason:  str          — 拒绝原因（若 allowed=False）
          quota_remaining: int  — 剩余 token 配额
          agents_remaining: int — 剩余 agent 槽位（-1 = 无限）
        """
        cg = self.get_cgroup(cgroup_name)
        if cg is None:
            return {
                "allowed": False,
                "reason": f"cgroup '{cgroup_name}' not found",
                "quota_remaining": 0,
                "agents_remaining": 0,
            }

        # 检查 token 配额
        if cg.is_throttled():
            return {
                "allowed": False,
                "reason": f"cgroup '{cgroup_name}' token quota exhausted "
                          f"({cg.token_used}/{cg.token_quota})",
                "quota_remaining": 0,
                "agents_remaining": self._agents_remaining(cg),
            }

        # 检查并发 agent 数
        if new_agent and cg.max_agents >= 0:
            active = self.get_agent_count(cgroup_name, active_only=True)
            if active >= cg.max_agents:
                return {
                    "allowed": False,
                    "reason": f"cgroup '{cgroup_name}' agent limit reached "
                              f"({active}/{cg.max_agents})",
                    "quota_remaining": cg.quota_remaining,
                    "agents_remaining": 0,
                }

        return {
            "allowed": True,
            "reason": "",
            "quota_remaining": cg.quota_remaining,
            "agents_remaining": self._agents_remaining(cg),
        }

    def charge_tokens(self, cgroup_name: str, tokens: int) -> Optional[CGroup]:
        """
        从 cgroup token 配额中扣减 tokens。
        类比 account_cfs_rq_runtime() — 调度器在 update_curr() 后调用。

        与 Scheduler.update_vruntime() 配套使用。
        """
        result = self._conn.execute(
            """UPDATE agent_cgroups
               SET token_used = token_used + ?
               WHERE cgroup_name=?""",
            (tokens, cgroup_name),
        )
        self._conn.commit()
        if result.rowcount == 0:
            return None
        return self.get_cgroup(cgroup_name)

    def reset_quota(self, cgroup_name: str) -> Optional[CGroup]:
        """
        重置 token_used 到 0（类比 cgroup CPU bandwidth period 到期重置）。
        通常由定时任务调用（每 session 或每日重置）。
        """
        result = self._conn.execute(
            "UPDATE agent_cgroups SET token_used=0 WHERE cgroup_name=?",
            (cgroup_name,),
        )
        self._conn.commit()
        return self.get_cgroup(cgroup_name) if result.rowcount > 0 else None

    def reset_all_quotas(self) -> int:
        """重置所有 cgroup 的 token_used，返回重置的 cgroup 数量"""
        result = self._conn.execute("UPDATE agent_cgroups SET token_used=0")
        self._conn.commit()
        return result.rowcount

    def get_agent_count(self, cgroup_name: str,
                        active_only: bool = False) -> int:
        """
        查询 cgroup 内 agent 数量。
        类比 cat /sys/fs/cgroup/<name>/cgroup.procs（只读进程列表长度）
        """
        sql = "SELECT COUNT(*) FROM agent_tasks WHERE cgroup_name=?"
        params = [cgroup_name]
        if active_only:
            sql += " AND status IN ('pending', 'running')"
        row = self._conn.execute(sql, params).fetchone()
        return row[0] if row else 0

    def list_cgroups(self) -> List[CGroup]:
        """列出所有 cgroup（类比 ls /sys/fs/cgroup/）"""
        rows = self._conn.execute(
            "SELECT * FROM agent_cgroups ORDER BY weight DESC"
        ).fetchall()
        return [self._row_to_cgroup(r) for r in rows]

    def update_cgroup(self, name: str, token_quota: int = None,
                      weight: int = None, max_agents: int = None,
                      description: str = None) -> Optional[CGroup]:
        """
        更新 cgroup 配置（类比 echo <val> > /sys/fs/cgroup/<name>/cpu.max）
        """
        sets, params = [], []
        if token_quota is not None:
            sets.append("token_quota=?"); params.append(token_quota)
        if weight is not None:
            sets.append("weight=?"); params.append(weight)
        if max_agents is not None:
            sets.append("max_agents=?"); params.append(max_agents)
        if description is not None:
            sets.append("description=?"); params.append(description)
        if not sets:
            return self.get_cgroup(name)
        params.append(name)
        self._conn.execute(
            f"UPDATE agent_cgroups SET {', '.join(sets)} WHERE cgroup_name=?",
            params,
        )
        self._conn.commit()
        return self.get_cgroup(name)

    def delete_cgroup(self, name: str) -> bool:
        """
        删除 cgroup（类比 rmdir /sys/fs/cgroup/<name>）
        若 cgroup 内还有活跃 agent，拒绝删除（类比非空 cgroup 不可删）
        """
        if name in {cg.name for cg in BUILTIN_CGROUPS}:
            return False  # 内置 cgroup 不可删除
        active = self.get_agent_count(name, active_only=True)
        if active > 0:
            return False
        result = self._conn.execute(
            "DELETE FROM agent_cgroups WHERE cgroup_name=?", (name,)
        )
        self._conn.commit()
        return result.rowcount > 0

    def aimd_adjust(self, cgroup_name: str,
                    queue_depth: int,
                    threshold: int = 3,
                    increase_rate: float = 0.10,
                    min_quota: int = 5_000,
                    max_quota: int = 500_000) -> Optional[CGroup]:
        """
        AIMD（Additive Increase Multiplicative Decrease）自适应配额调整。

        OS 类比：TCP AIMD 拥塞控制（Van Jacobson, 1988）
          - 慢启动/加性增：cwnd += 1 MSS / RTT（无拥塞时线性增大）
          - 乘性减：拥塞时 cwnd *= 0.5（指数退避，快速释放资源）
          - 效果：在无拥塞时充分利用带宽，拥塞时迅速让步（防振荡）

        Memory OS 映射：
          - cwnd         → token_quota（资源窗口）
          - queue_depth  → pending task 数（类比 ACK 延迟 = 拥塞信号）
          - threshold    → 队列深度阈值（类比 ssthresh）
          - 加性增：queue_depth == 0 → quota += quota * increase_rate（空闲扩容）
          - 乘性减：queue_depth > threshold → quota *= 0.5（过载收缩）

        参数：
          cgroup_name:   目标 cgroup
          queue_depth:   当前待处理 task 数（0 = 空闲，>threshold = 过载）
          threshold:     触发乘性减的队列深度阈值（默认 3）
          increase_rate: 加性增比例（默认 10%）
          min_quota:     配额下限，防止过度收缩（默认 5000 tokens）
          max_quota:     配额上限，防止无限扩张（默认 500000 tokens）

        返回：更新后的 CGroup，或 None（cgroup 不存在 / 无限配额不调整）
        """
        cg = self.get_cgroup(cgroup_name)
        if cg is None:
            return None
        if cg.token_quota < 0:
            return cg  # 无限配额组不参与 AIMD 调整

        current_quota = cg.token_quota

        if queue_depth == 0:
            # 加性增：空闲时扩容 increase_rate
            new_quota = int(current_quota * (1.0 + increase_rate))
            new_quota = min(new_quota, max_quota)
        elif queue_depth > threshold:
            # 乘性减：过载时减半（指数退避）
            new_quota = int(current_quota * 0.5)
            new_quota = max(new_quota, min_quota)
        else:
            # 正常区间（1 <= queue_depth <= threshold）：不调整
            return cg

        return self.update_cgroup(cgroup_name, token_quota=new_quota)

    def close(self) -> None:
        self._conn.close()

    # ── 私有 ─────────────────────────────────────────────────────────────────

    def _agents_remaining(self, cg: CGroup) -> int:
        if cg.max_agents < 0:
            return -1
        active = self.get_agent_count(cg.name, active_only=True)
        return max(0, cg.max_agents - active)

    @staticmethod
    def _row_to_cgroup(row: sqlite3.Row) -> CGroup:
        return CGroup(
            name=row["cgroup_name"],
            token_quota=row["token_quota"],
            token_used=row["token_used"],
            weight=row["weight"],
            max_agents=row["max_agents"],
            created_at=row["created_at"],
            description=row["description"],
        )
