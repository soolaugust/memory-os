"""
hook_manager.py — Hook 编排管理器
OS 类比：systemd PID 1 manager

职责：
  1. load_units()        — 解析 settings.json → HookUnit 列表（类比 systemd daemon-reload）
  2. resolve_order()     — 拓扑排序，生成执行序列（类比 systemd-analyze）
  3. execute_target()    — 按依赖图执行一个 event 下的所有 hooks（类比 systemctl start <target>）
  4. get_status()        — 所有 unit 当前状态（类比 systemctl list-units）
  5. get_dependency_graph() — 依赖树文本（类比 systemctl list-dependencies）

并行策略：
  无相互依赖的 unit 在同一"执行波次（wave）"内并发运行（concurrent.futures.ThreadPoolExecutor）。
  有 After= 约束的 unit 必须等待前置 wave 完成。

循环依赖检测：
  拓扑排序使用 Kahn 算法，发现入度永不归零的节点时抛出 CyclicDependencyError。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

# 允许在 init/ 子包外直接运行时也能导入
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from init.hook_unit import HookUnit, HookStatus


# ── 异常类型 ──────────────────────────────────────────────────────────────


class CyclicDependencyError(Exception):
    """检测到循环依赖，类比 systemd 的 job loop detection"""
    pass


class UnknownUnitError(Exception):
    """依赖声明中引用了未知 unit 名称"""
    pass


# ── 执行结果 ──────────────────────────────────────────────────────────────


@dataclass
class UnitResult:
    """单条 unit 执行结果（类比 systemd job result）"""
    unit_name: str
    success: bool
    exit_code: int
    duration_ms: float
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class TargetResult:
    """一个 event target 的整体执行结果（类比 systemd target activation result）"""
    event: str
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    total_duration_ms: float = 0.0
    unit_results: List[UnitResult] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.failed == 0


# ── 内置依赖规则（启发式，按子系统职责推导）─────────────────────────────


# 每个 event 内，按子系统建议的执行关键词顺序（越靠前越早执行）
_BUILTIN_ORDER: Dict[str, List[str]] = {
    "SessionStart": [
        "sleep-session-start",
        "snarc-session-start",
        "ecc-session-start",
        "memory-os-loader",
    ],
    "UserPromptSubmit": [
        "sleep-activity-touch",
        "snarc-user-prompt",
        "task-create-reminder",
        "memory-os-writer",
        "memory-os-retriever",
    ],
    "Stop": [
        "memory-os-extractor",
        "snarc-session-end",
        "notify",
        "memory-os-stop-coalesced",
    ],
    "PostToolUse": [
        "ote-auto-record",
        "observagent-relay",
        "ecc-pr-created",
        "ecc-build-complete",
        "ecc-quality-gate",
        "ecc-governance-capture",
        "memory-os-posttool-observers",
    ],
    "PreToolUse": [
        "git-no-verify-block",
        "ecc-doc-file-warning",
        "ecc-suggest-compact",
        "ecc-mcp-health-check",
        "memory-os-pretool-coalesced",
    ],
    "PreCompact": [
        "save-task-state",
        "ecc-pre-compact",
    ],
    "PostCompact": [
        "resume-task-state",
        "snarc-post-compact",
    ],
    "SessionEnd": [
        "ote-flush-queue",
        "ecc-session-end-marker",
    ],
}


def _infer_unit_name(event: str, index: int, command: str, matcher: str) -> str:
    """
    从命令推断 unit 名称（类比 systemd 用 unit file 文件名作为 unit name）。
    """
    import re

    # 内联 node -e "..." → 从代码内容提取关键词
    if command.strip().startswith("node -e"):
        for keyword in [
            "session-start", "session-end", "session:start", "session:end",
            "user-prompt", "post-compact", "pre-compact",
            "pr-created", "build-complete", "quality-gate",
            "governance-capture", "mcp-health-check",
            "doc-file-warning", "suggest-compact",
            "no-verify", "git",
        ]:
            if keyword in command:
                safe = keyword.replace(":", "-").replace("_", "-")
                return f"{event.lower()}-{safe}-{index}"
        return f"{event.lower()}-inline-{index}"

    # 提取脚本文件名（不含扩展名）
    parts = command.split()
    script = None
    for p in reversed(parts):
        p = p.strip('"\'')
        if "/" in p or p.endswith((".py", ".js", ".sh")):
            script = os.path.basename(p)
            script = re.sub(r'\.(py|js|sh)$', '', script)
            break

    if not script:
        script = os.path.basename(parts[0]) if parts else "unknown"

    if matcher and matcher != "*":
        safe_matcher = re.sub(r'[^a-zA-Z0-9]', '-', matcher)[:20].strip('-')
        return f"{script}-{safe_matcher}-{index}"

    return f"{script}-{index}"


# ── HookManager ───────────────────────────────────────────────────────────


class HookManager:
    """
    Hook 编排管理器
    OS 类比：systemd PID 1

    使用流程：
      mgr = HookManager(Path.home() / ".claude" / "settings.json")
      mgr.load_units()
      result = mgr.execute_target("SessionStart", stdin_data="{...}")
      print(mgr.get_status())
    """

    def __init__(self, settings_path: str = None, db_path: str = None,
                 max_workers: int = 8):
        self.settings_path = settings_path or os.path.expanduser("~/.claude/settings.json")
        self.db_path = db_path
        self.max_workers = max_workers

        self._units: Dict[str, HookUnit] = {}
        self._event_index: Dict[str, List[str]] = defaultdict(list)
        self._lock = threading.Lock()
        self._loaded = False

    # ── 加载 ─────────────────────────────────────────────────────────────

    def load_units(self) -> int:
        """
        解析 settings.json 中所有 hooks → HookUnit。
        类比：systemd daemon-reload。
        返回：加载的 unit 总数。
        """
        with open(self.settings_path, "r", encoding="utf-8") as f:
            settings = json.load(f)

        hooks_cfg = settings.get("hooks", {})
        new_units: Dict[str, HookUnit] = {}
        new_event_index: Dict[str, List[str]] = defaultdict(list)

        for event, event_hooks in hooks_cfg.items():
            for idx, hook_group in enumerate(event_hooks):
                matcher = hook_group.get("matcher", "*")
                for hook_def in hook_group.get("hooks", []):
                    command = hook_def.get("command", "")
                    timeout_sec = hook_def.get("timeout", 30)
                    is_async = bool(hook_def.get("async", False))

                    name = _infer_unit_name(event, idx, command, matcher)

                    # 防重名
                    base_name = name
                    suffix = 0
                    while name in new_units:
                        suffix += 1
                        name = f"{base_name}-dup{suffix}"

                    unit = HookUnit(
                        name=name,
                        event=event,
                        command=command,
                        matcher=matcher,
                        timeout_ms=int(timeout_sec * 1000),
                        is_async=is_async,
                    )
                    self._apply_builtin_deps(unit)

                    new_units[name] = unit
                    new_event_index[event].append(name)

        with self._lock:
            self._units = new_units
            self._event_index = new_event_index
            self._loaded = True

        return len(new_units)

    def _apply_builtin_deps(self, unit: HookUnit) -> None:
        """根据 _BUILTIN_ORDER 推断 After= 依赖（启发式）"""
        order_list = _BUILTIN_ORDER.get(unit.event, [])
        my_pos = None
        for pos, keyword in enumerate(order_list):
            if keyword in unit.name:
                my_pos = pos
                break
        if my_pos is None or my_pos == 0:
            return
        for prev_pos in range(my_pos - 1, -1, -1):
            unit.after.append(f"keyword:{order_list[prev_pos]}")

    # ── 手动依赖注册 ──────────────────────────────────────────────────────

    def add_dependency(self, unit_name: str, dep_type: str, dep_target: str) -> None:
        """手动添加依赖。dep_type: 'after' | 'requires' | 'wants'"""
        with self._lock:
            unit = self._units.get(unit_name)
            if unit is None:
                raise UnknownUnitError(f"Unit '{unit_name}' not found")
            target_list = getattr(unit, dep_type, None)
            if target_list is None:
                raise ValueError(f"Unknown dep_type: {dep_type}")
            if dep_target not in target_list:
                target_list.append(dep_target)

    # ── 拓扑排序 ─────────────────────────────────────────────────────────

    def resolve_order(self, event: str) -> List[List[str]]:
        """
        为指定 event 计算执行波次（waves）。
        算法：Kahn's BFS 拓扑排序 + 循环依赖检测。
        返回：List[List[unit_name]]，同一 wave 内可并行执行。
        类比：systemd transaction 构建。
        """
        with self._lock:
            unit_names = list(self._event_index.get(event, []))
            units_snapshot = {n: self._units[n] for n in unit_names if n in self._units}

        if not unit_names:
            return []

        name_set = set(unit_names)

        def resolve_dep(dep: str) -> Optional[str]:
            if dep.startswith("keyword:"):
                keyword = dep[8:]
                candidates = [n for n in unit_names if keyword in n]
                return candidates[-1] if candidates else None
            return dep if dep in name_set else None

        # 构建图
        in_degree: Dict[str, int] = {n: 0 for n in unit_names}
        successors: Dict[str, Set[str]] = {n: set() for n in unit_names}

        for name in unit_names:
            unit = units_snapshot[name]
            all_deps = list(unit.after) + list(unit.requires) + list(unit.wants)
            seen_preds = set()
            for dep in all_deps:
                pred = resolve_dep(dep)
                if pred and pred != name and pred not in seen_preds:
                    seen_preds.add(pred)
                    successors[pred].add(name)
                    in_degree[name] += 1

        # Kahn BFS
        waves: List[List[str]] = []
        remaining = set(unit_names)

        while remaining:
            wave = sorted(n for n in remaining if in_degree[n] == 0)
            if not wave:
                raise CyclicDependencyError(
                    f"Cyclic dependency in event '{event}' among: {sorted(remaining)}"
                )
            waves.append(wave)
            for n in wave:
                remaining.remove(n)
                for succ in successors[n]:
                    in_degree[succ] -= 1

        return waves

    # ── 执行 ─────────────────────────────────────────────────────────────

    def execute_target(self, event: str, stdin_data: str = "",
                       env: dict = None) -> TargetResult:
        """
        执行指定 event 下所有 hooks（按依赖顺序，同波次并行）。
        类比：systemctl start <event>.target
        """
        result = TargetResult(event=event)
        t_start = time.time()

        try:
            waves = self.resolve_order(event)
        except CyclicDependencyError as e:
            self._journal("ERR", "hook-manager",
                          f"Cyclic dependency in {event}: {e}")
            result.total_duration_ms = (time.time() - t_start) * 1000
            return result

        failed_requires: Set[str] = set()

        for wave in waves:
            wave_futures = {}
            skip_list = []

            for unit_name in wave:
                with self._lock:
                    unit = self._units.get(unit_name)
                if unit is None:
                    continue

                skip_reason = self._check_requires(unit, failed_requires)
                if skip_reason:
                    skip_list.append(UnitResult(
                        unit_name=unit_name, success=False,
                        exit_code=-1, duration_ms=0,
                        skipped=True, skip_reason=skip_reason,
                    ))
                    failed_requires.add(unit_name)
                    continue

                with ThreadPoolExecutor(max_workers=1) as pool:
                    wave_futures[unit_name] = pool.submit(
                        self._run_unit, unit, stdin_data, env or {}
                    )

            # 并行提交本 wave 所有非 skip 的 unit
            if wave_futures:
                with ThreadPoolExecutor(max_workers=min(self.max_workers, len(wave_futures))) as pool:
                    futures = {pool.submit(self._run_unit,
                                           self._units[n], stdin_data, env or {}): n
                               for n in wave_futures}
                    for future in as_completed(futures):
                        unit_name = futures[future]
                        try:
                            ur = future.result()
                        except Exception as exc:
                            ur = UnitResult(unit_name=unit_name, success=False,
                                            exit_code=-1, duration_ms=0,
                                            skip_reason=str(exc))
                        result.unit_results.append(ur)
                        result.total += 1
                        if ur.success:
                            result.succeeded += 1
                        else:
                            result.failed += 1
                            failed_requires.add(unit_name)

            for ur in skip_list:
                result.unit_results.append(ur)
                result.total += 1
                result.skipped += 1

        result.total_duration_ms = (time.time() - t_start) * 1000
        self._journal(
            "INFO", "hook-manager",
            f"target {event}: {result.succeeded}/{result.total} ok, "
            f"{result.failed} failed, {result.skipped} skipped, "
            f"{result.total_duration_ms:.0f}ms",
        )
        return result

    def _check_requires(self, unit: HookUnit, failed: Set[str]) -> str:
        with self._lock:
            requires = list(unit.requires)
        for dep in requires:
            if dep in failed:
                return f"required unit '{dep}' failed"
        return ""

    def _run_unit(self, unit: HookUnit, stdin_data: str, extra_env: dict) -> UnitResult:
        """
        执行单个 HookUnit。
        async unit: spawn 后不等待，直接返回 success=True。
        sync unit: 等待进程结束，检查 exit_code，带超时。
        """
        env = os.environ.copy()
        env.update(extra_env)

        unit.mark_activating()
        t0 = time.time()

        try:
            proc = subprocess.Popen(
                unit.command,
                shell=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )

            if unit.is_async:
                duration_ms = (time.time() - t0) * 1000
                unit.mark_active(0, duration_ms)
                self._journal("DEBUG", "hook-manager",
                              f"async spawn: {unit.name} ({duration_ms:.0f}ms)")
                return UnitResult(unit_name=unit.name, success=True,
                                  exit_code=0, duration_ms=duration_ms)

            timeout_sec = unit.timeout_ms / 1000.0
            try:
                stdout, stderr = proc.communicate(
                    input=stdin_data.encode("utf-8"),
                    timeout=timeout_sec,
                )
                exit_code = proc.returncode
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                duration_ms = (time.time() - t0) * 1000
                unit.mark_failed(-1, duration_ms)
                self._journal("WARN", "hook-manager",
                              f"timeout: {unit.name} after {unit.timeout_ms}ms")
                return UnitResult(unit_name=unit.name, success=False,
                                  exit_code=-1, duration_ms=duration_ms)

            duration_ms = (time.time() - t0) * 1000
            success = (exit_code == 0)

            if success:
                unit.mark_active(exit_code, duration_ms)
                self._journal("DEBUG", "hook-manager",
                              f"ok: {unit.name} exit={exit_code} {duration_ms:.0f}ms")
            else:
                unit.mark_failed(exit_code, duration_ms)
                stderr_str = stderr.decode("utf-8", errors="replace")[:200]
                self._journal("WARN", "hook-manager",
                              f"failed: {unit.name} exit={exit_code} | {stderr_str}")

            return UnitResult(unit_name=unit.name, success=success,
                              exit_code=exit_code, duration_ms=duration_ms)

        except Exception as exc:
            duration_ms = (time.time() - t0) * 1000
            unit.mark_failed(-1, duration_ms)
            self._journal("ERR", "hook-manager", f"exception: {unit.name}: {exc}")
            return UnitResult(unit_name=unit.name, success=False,
                              exit_code=-1, duration_ms=duration_ms,
                              skip_reason=str(exc))

    # ── 状态查询 ─────────────────────────────────────────────────────────

    def get_status(self, event: str = None) -> List[dict]:
        """
        返回所有（或指定 event 的）unit 状态列表。
        类比：systemctl list-units
        """
        with self._lock:
            names = self._event_index.get(event, []) if event else list(self._units.keys())
            return [self._units[n].to_status_dict() for n in names if n in self._units]

    def get_dependency_graph(self, event: str = None) -> str:
        """
        返回依赖图 ASCII 文本。
        类比：systemctl list-dependencies [<target>]
        """
        lines = []
        events = [event] if event else sorted(self._event_index.keys())

        for ev in events:
            lines.append(f"{ev}.target")
            with self._lock:
                names = list(self._event_index.get(ev, []))

            try:
                waves = self.resolve_order(ev)
                wave_map = {n: wi for wi, wave in enumerate(waves) for n in wave}
            except CyclicDependencyError:
                wave_map = {}

            for i, name in enumerate(names):
                is_last = (i == len(names) - 1)
                prefix = "└── " if is_last else "├── "
                child_prefix = "    " if is_last else "│   "

                with self._lock:
                    unit = self._units.get(name)
                if not unit:
                    continue

                wave_label = f"wave={wave_map.get(name, '?')}" if wave_map else ""
                lines.append(
                    f"  {prefix}{name:<45} [{unit.status.value:<11}] "
                    f"{'async' if unit.is_async else 'sync '} {wave_label}"
                )

                deps = (
                    [f"After: {d}" for d in unit.after] +
                    [f"Requires: {d}" for d in unit.requires] +
                    [f"Wants: {d}" for d in unit.wants]
                )
                for j, dep in enumerate(deps):
                    dp = "└── " if j == len(deps) - 1 else "├── "
                    lines.append(f"  {child_prefix}  {dp}{dep}")

            lines.append("")

        return "\n".join(lines)

    def list_events(self) -> List[str]:
        """返回所有已知 event 名称"""
        with self._lock:
            return sorted(self._event_index.keys())

    def get_unit(self, name: str) -> Optional[HookUnit]:
        """按名称获取 unit（类比 systemctl show <unit>）"""
        with self._lock:
            return self._units.get(name)

    def reset_failed(self, event: str = None) -> int:
        """重置所有 failed 状态的 unit 为 inactive（类比 systemctl reset-failed）"""
        count = 0
        with self._lock:
            names = self._event_index.get(event, []) if event else list(self._units.keys())
            for name in names:
                unit = self._units.get(name)
                if unit and unit.status == HookStatus.FAILED:
                    unit.mark_inactive()
                    count += 1
        return count

    # ── Journal 写入 ──────────────────────────────────────────────────────

    def _journal(self, level: str, subsystem: str, message: str) -> None:
        """写入 dmesg journal（若 db_path 已配置）"""
        if not self.db_path:
            return
        try:
            import sqlite3
            from store_core import dmesg_log
            conn = sqlite3.connect(self.db_path)
            dmesg_log(conn, level, subsystem, message)
            conn.commit()
            conn.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        with self._lock:
            n = len(self._units)
            evs = len(self._event_index)
        return f"HookManager(units={n}, events={evs}, loaded={self._loaded})"
