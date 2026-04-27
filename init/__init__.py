"""
Memory OS — init/ Hook 编排子系统
OS 类比：Linux systemd (2010, Lennart Poettering)

模块：
  hook_unit     — HookUnit 定义（类比 systemd unit）
  hook_manager  — 编排管理器（类比 systemd manager + 拓扑排序）
  hook_journal  — 执行日志（类比 journald）
  hook_analyzer — 静态分析工具（类比 systemd-analyze）
"""

from .hook_unit import HookUnit, HookStatus, HookDependency
from .hook_manager import HookManager
from .hook_journal import HookJournal
from .hook_analyzer import HookAnalyzer

__all__ = [
    "HookUnit",
    "HookStatus",
    "HookDependency",
    "HookManager",
    "HookJournal",
    "HookAnalyzer",
]
