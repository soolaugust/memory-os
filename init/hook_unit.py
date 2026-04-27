"""
hook_unit.py — Hook Unit 定义
OS 类比：systemd unit file (.service/.timer/.target)

每个 HookUnit 对应 settings.json 中一条 hook 配置，
加上依赖关系注解（after/requires/wants）和运行时状态。

状态机（类比 systemd unit 生命周期）：
  inactive → activating → active → deactivating → inactive
                                 ↘ failed
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class HookStatus(str, Enum):
    """
    OS 类比：systemctl status 返回的 ActiveState
      inactive   → 未运行（初始态）
      activating → 正在启动（命令已 spawn，等待结果）
      active     → 上次执行成功
      failed     → 上次执行失败（exit_code != 0 或超时）
      deactivating → 正在停止（async hook 收到 SIGTERM）
    """
    INACTIVE     = "inactive"
    ACTIVATING   = "activating"
    ACTIVE       = "active"
    FAILED       = "failed"
    DEACTIVATING = "deactivating"


class HookDependency(str, Enum):
    """
    依赖类型（类比 systemd 依赖指令）
      AFTER    — After=    顺序约束，本 unit 必须在目标 unit 完成后才开始
      REQUIRES — Requires= 强依赖：目标失败则本 unit 也标记 failed，不执行
      WANTS    — Wants=    弱依赖：目标失败不阻止本 unit 执行
    """
    AFTER    = "after"
    REQUIRES = "requires"
    WANTS    = "wants"


@dataclass
class HookUnit:
    """
    Hook Unit — 最小可管理单元
    OS 类比：systemd service unit

    字段说明：
      name       — 单元名（全局唯一，格式：<subsystem>-<function>，类比 nginx.service）
      event      — 触发事件（类比 WantedBy=multi-user.target）
      command    — 执行命令（类比 ExecStart=）
      matcher    — Claude hook matcher 模式（"*" / "Bash" / regex）
      timeout_ms — 超时毫秒（类比 TimeoutStartSec=）
      is_async   — 异步执行（类比 Type=forking + RemainAfterExit=no）

    依赖字段（类比 After= / Requires= / Wants=）：
      after      — 顺序依赖（必须在这些 unit 执行完成后）
      requires   — 强依赖（任一失败则本 unit 跳过并标记 failed）
      wants      — 弱依赖（失败不阻止本 unit）

    运行时状态（类比 systemd active state）：
      status        — 当前生命周期状态
      last_exit_code — 上次退出码（None = 未执行过）
      last_duration_ms — 上次执行耗时
      last_run_at   — 上次执行时间戳（epoch）
      run_count     — 累计执行次数
      fail_count    — 累计失败次数
    """

    # 核心字段
    name: str = ""
    event: str = ""
    command: str = ""
    matcher: str = "*"
    timeout_ms: int = 30_000
    is_async: bool = False

    # 依赖关系（类比 systemd 依赖指令）
    after: list = field(default_factory=list)
    requires: list = field(default_factory=list)
    wants: list = field(default_factory=list)

    # 运行时状态
    status: HookStatus = HookStatus.INACTIVE
    last_exit_code: Optional[int] = None
    last_duration_ms: Optional[float] = None
    last_run_at: Optional[float] = None   # time.time()
    run_count: int = 0
    fail_count: int = 0

    def __post_init__(self):
        # 自动规范化：枚举值允许字符串传入
        if isinstance(self.status, str):
            self.status = HookStatus(self.status)

    # ── 状态转换方法（类比 systemd state machine）─────────────────

    def mark_activating(self):
        """开始执行前调用（类比 systemd → activating）"""
        self.status = HookStatus.ACTIVATING
        self.last_run_at = time.time()

    def mark_active(self, exit_code: int, duration_ms: float):
        """执行成功后调用（类比 systemd → active）"""
        self.status = HookStatus.ACTIVE
        self.last_exit_code = exit_code
        self.last_duration_ms = duration_ms
        self.run_count += 1

    def mark_failed(self, exit_code: int, duration_ms: float):
        """执行失败/超时后调用（类比 systemd → failed）"""
        self.status = HookStatus.FAILED
        self.last_exit_code = exit_code
        self.last_duration_ms = duration_ms
        self.run_count += 1
        self.fail_count += 1

    def mark_inactive(self):
        """重置到初始态（类比 systemctl reset-failed）"""
        self.status = HookStatus.INACTIVE

    # ── 查询方法 ──────────────────────────────────────────────────

    @property
    def is_healthy(self) -> bool:
        """上次执行是否成功（status == active 且 exit_code == 0）"""
        return self.status == HookStatus.ACTIVE and self.last_exit_code == 0

    @property
    def reliability(self) -> float:
        """可靠性分数 0.0-1.0（类比 systemd service success rate）"""
        if self.run_count == 0:
            return 1.0
        return (self.run_count - self.fail_count) / self.run_count

    def to_status_dict(self) -> dict:
        """
        类比 systemctl status <unit> 的结构化输出
        """
        return {
            "name":            self.name,
            "event":           self.event,
            "matcher":         self.matcher,
            "status":          self.status.value,
            "is_async":        self.is_async,
            "timeout_ms":      self.timeout_ms,
            "run_count":       self.run_count,
            "fail_count":      self.fail_count,
            "reliability":     round(self.reliability, 3),
            "last_exit_code":  self.last_exit_code,
            "last_duration_ms": self.last_duration_ms,
            "dependencies": {
                "after":    self.after,
                "requires": self.requires,
                "wants":    self.wants,
            },
        }

    def __repr__(self) -> str:
        return (
            f"HookUnit(name={self.name!r}, event={self.event!r}, "
            f"status={self.status.value}, async={self.is_async})"
        )
