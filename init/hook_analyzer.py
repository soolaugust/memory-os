"""
hook_analyzer.py — Hook 静态分析工具
OS 类比：systemd-analyze（启动分析）+ systemd-analyze verify（配置校验）

功能：
  1. 解析 settings.json 中全部 34 个 hooks → HookUnit
  2. 输出依赖图可视化（ASCII art，类比 systemd-analyze dot）
  3. 检测潜在竞争条件（同 event 内同 wave 内访问共同资源的 hook 对）
  4. 检测超时风险（sync hook 超时过长 / async hook 失败无感知）
  5. 输出并行化建议（可合并为同一 wave 的 hooks）
  6. 生成可读报告

可直接运行：
  python3 init/hook_analyzer.py [--settings ~/.claude/settings.json] [--event SessionStart]
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from init.hook_unit import HookUnit
from init.hook_manager import HookManager, CyclicDependencyError


# ── 分析结果数据类 ────────────────────────────────────────────────────────


@dataclass
class RaceCondition:
    """
    潜在竞争条件：两个 hook 在同一 wave 内访问同一资源但无顺序约束。
    OS 类比：两个进程无锁并发写同一文件。
    """
    event: str
    unit_a: str
    unit_b: str
    shared_resource: str
    severity: str        # "HIGH" | "MEDIUM" | "LOW"
    description: str


@dataclass
class TimeoutRisk:
    """
    超时风险：sync hook 阻塞整个 wave，或 timeout 设置不合理。
    OS 类比：TimeoutStartSec 过长导致启动卡住。
    """
    unit_name: str
    event: str
    timeout_ms: int
    is_async: bool
    risk_level: str      # "HIGH" | "MEDIUM" | "LOW"
    description: str


@dataclass
class ParallelGroup:
    """
    可并行执行的 hook 组（同一 wave 内无依赖）。
    OS 类比：systemd parallel activation。
    """
    event: str
    wave: int
    units: List[str]
    estimated_saving_ms: float


@dataclass
class AnalysisReport:
    """完整分析报告"""
    settings_path: str
    total_hooks: int
    events: List[str]
    hooks_per_event: Dict[str, int]
    waves_per_event: Dict[str, List[List[str]]]
    race_conditions: List[RaceCondition]
    timeout_risks: List[TimeoutRisk]
    parallel_groups: List[ParallelGroup]
    cycle_errors: Dict[str, str]


# ── 资源访问模式（启发式竞争检测）──────────────────────────────────────────


_RESOURCE_PATTERNS: List[Tuple[str, str]] = [
    ("store.db",       "memory-os-sqlite"),
    ("store_core",     "memory-os-sqlite"),
    ("extractor",      "memory-os-sqlite"),
    ("loader",         "memory-os-sqlite"),
    ("writer",         "memory-os-sqlite"),
    ("retriever",      "memory-os-sqlite"),
    ("observagent",    "observagent-relay"),
    ("snarc",          "snarc-db"),
    ("ote",            "ote-queue"),
    ("task-state",     "task-state-file"),
    ("sleep-activity", "activity-timestamp"),
    ("governance",     "governance-log"),
]

_TIMEOUT_HIGH_MS   = 20_000
_TIMEOUT_MEDIUM_MS = 10_000
_DEFAULT_TIMEOUT_MS = 30_000


def _infer_resources(unit: HookUnit) -> Set[str]:
    """推断 hook 访问的资源集合（命令文本模式匹配）"""
    cmd_lower = unit.command.lower()
    return {res for kw, res in _RESOURCE_PATTERNS if kw in cmd_lower}


# ── HookAnalyzer ──────────────────────────────────────────────────────────


class HookAnalyzer:
    """
    Hook 静态分析工具
    OS 类比：systemd-analyze verify + systemd-analyze critical-chain

    使用方式：
      analyzer = HookAnalyzer(Path.home() / ".claude" / "settings.json")
      report = analyzer.analyze()
      print(analyzer.format_report(report))
    """

    def __init__(self, settings_path: str = None):
        self.settings_path = settings_path or os.path.expanduser("~/.claude/settings.json")
        self._manager = HookManager(settings_path=self.settings_path)

    def analyze(self) -> AnalysisReport:
        """执行完整静态分析（类比：systemd-analyze verify）"""
        n = self._manager.load_units()
        events = self._manager.list_events()

        hooks_per_event = {ev: len(self._manager.get_status(ev)) for ev in events}

        waves_per_event: Dict[str, List[List[str]]] = {}
        cycle_errors: Dict[str, str] = {}
        for ev in events:
            try:
                waves_per_event[ev] = self._manager.resolve_order(ev)
            except CyclicDependencyError as e:
                cycle_errors[ev] = str(e)
                waves_per_event[ev] = []

        return AnalysisReport(
            settings_path=self.settings_path,
            total_hooks=n,
            events=events,
            hooks_per_event=hooks_per_event,
            waves_per_event=waves_per_event,
            race_conditions=self._detect_race_conditions(events),
            timeout_risks=self._detect_timeout_risks(events),
            parallel_groups=self._analyze_parallel_groups(waves_per_event),
            cycle_errors=cycle_errors,
        )

    # ── 竞争条件检测 ──────────────────────────────────────────────────────

    def _detect_race_conditions(self, events: List[str]) -> List[RaceCondition]:
        """
        检测同 event 同 wave 内、访问共同资源的 hook 对。
        OS 类比：inotifywait 检测并发写同一文件。
        """
        races = []
        for ev in events:
            try:
                waves = self._manager.resolve_order(ev)
            except CyclicDependencyError:
                continue

            for wave in waves:
                if len(wave) < 2:
                    continue

                unit_resources: Dict[str, Set[str]] = {}
                for name in wave:
                    unit = self._manager.get_unit(name)
                    if unit:
                        unit_resources[name] = _infer_resources(unit)

                names = list(unit_resources.keys())
                for i in range(len(names)):
                    for j in range(i + 1, len(names)):
                        a, b = names[i], names[j]
                        shared = unit_resources[a] & unit_resources[b]
                        if not shared:
                            continue

                        resource_str = ", ".join(sorted(shared))
                        severity = (
                            "HIGH"   if "memory-os-sqlite" in shared else
                            "MEDIUM" if ("snarc-db" in shared or "ote-queue" in shared) else
                            "LOW"
                        )
                        races.append(RaceCondition(
                            event=ev, unit_a=a, unit_b=b,
                            shared_resource=resource_str,
                            severity=severity,
                            description=(
                                f"Both access [{resource_str}] in the same execution wave "
                                f"with no ordering constraint. "
                                f"Fix: add After={a} to {b} (or vice versa)."
                            ),
                        ))
        return races

    # ── 超时风险检测 ──────────────────────────────────────────────────────

    def _detect_timeout_risks(self, events: List[str]) -> List[TimeoutRisk]:
        """
        检测超时风险：sync hook 超时过长 / async hook 无健康检查。
        OS 类比：TimeoutStartSec 检查。
        """
        risks = []
        for ev in events:
            for u in self._manager.get_status(ev):
                name       = u["name"]
                is_async   = u["is_async"]
                timeout_ms = u["timeout_ms"]

                if not is_async:
                    if timeout_ms >= _TIMEOUT_HIGH_MS:
                        risks.append(TimeoutRisk(
                            unit_name=name, event=ev,
                            timeout_ms=timeout_ms, is_async=False,
                            risk_level="HIGH",
                            description=(
                                f"Sync hook with {timeout_ms/1000:.0f}s timeout "
                                f"blocks the entire wave. Consider async=true or reduce timeout."
                            ),
                        ))
                    elif timeout_ms >= _TIMEOUT_MEDIUM_MS:
                        risks.append(TimeoutRisk(
                            unit_name=name, event=ev,
                            timeout_ms=timeout_ms, is_async=False,
                            risk_level="MEDIUM",
                            description=(
                                f"Sync hook with {timeout_ms/1000:.0f}s timeout "
                                f"may delay dependent units."
                            ),
                        ))
                else:
                    # async：spawn 后不等待，失败无感知
                    if timeout_ms >= _DEFAULT_TIMEOUT_MS:
                        risks.append(TimeoutRisk(
                            unit_name=name, event=ev,
                            timeout_ms=timeout_ms, is_async=True,
                            risk_level="LOW",
                            description=(
                                f"Async hook failures are silently ignored. "
                                f"Add health check if this hook is critical."
                            ),
                        ))
        return risks

    # ── 并行化机会分析 ────────────────────────────────────────────────────

    def _analyze_parallel_groups(
        self, waves_per_event: Dict[str, List[List[str]]]
    ) -> List[ParallelGroup]:
        """
        分析每个 wave 内可并行执行的 hook 组。
        OS 类比：systemd parallel activation。
        """
        groups = []
        for ev, waves in waves_per_event.items():
            for wi, wave in enumerate(waves):
                if len(wave) >= 2:
                    groups.append(ParallelGroup(
                        event=ev, wave=wi, units=wave,
                        estimated_saving_ms=(len(wave) - 1) * 200.0,
                    ))
        groups.sort(key=lambda g: g.estimated_saving_ms, reverse=True)
        return groups

    # ── 报告格式化 ────────────────────────────────────────────────────────

    def format_report(self, report: AnalysisReport, detail: bool = True) -> str:
        """
        格式化完整分析报告。
        类比：systemd-analyze blame + systemd-analyze critical-chain 综合输出。
        """
        lines = []
        SEP = "=" * 70

        lines += [
            SEP,
            "  Memory OS Hook Orchestration Analysis",
            "  OS analogy: systemd-analyze verify + plot",
            SEP,
            f"  Settings : {report.settings_path}",
            f"  Hooks    : {report.total_hooks}",
            f"  Events   : {len(report.events)}",
            "",
        ]

        # 1. Hook 分布
        lines.append("── [1] Hook 分布 (systemctl list-units --all)")
        lines.append("")
        lines.append(f"  {'Event':<25} {'Hooks':>5}  {'Waves':>5}  {'Parallel':>8}")
        lines.append(f"  {'-'*25} {'-'*5}  {'-'*5}  {'-'*8}")
        for ev in sorted(report.events):
            count  = report.hooks_per_event.get(ev, 0)
            waves  = report.waves_per_event.get(ev, [])
            wcount = len(waves)
            par    = sum(len(w) for w in waves if len(w) > 1)
            lines.append(f"  {ev:<25} {count:>5}  {wcount:>5}  {par:>8}")
        lines.append("")

        # 2. 依赖图
        lines.append("── [2] 依赖图 ASCII (systemctl list-dependencies)")
        lines.append("")
        lines.append(self._manager.get_dependency_graph())

        # 3. 循环依赖
        if report.cycle_errors:
            lines.append("── [3] CYCLE ERRORS (systemd job loop detection)")
            lines.append("")
            for ev, err in report.cycle_errors.items():
                lines.append(f"  [CYCLE] {ev}: {err}")
            lines.append("")
        else:
            lines.append("── [3] 循环依赖检测: OK (无循环)")
            lines.append("")

        # 4. 竞争条件
        lines.append("── [4] 竞争条件分析 (并发写入检测)")
        lines.append("")
        if not report.race_conditions:
            lines.append("  [OK] 未检测到竞争条件")
        else:
            by_sev: Dict[str, list] = defaultdict(list)
            for rc in report.race_conditions:
                by_sev[rc.severity].append(rc)
            for sev in ["HIGH", "MEDIUM", "LOW"]:
                for rc in by_sev.get(sev, []):
                    lines.append(f"  [{sev:<6}] {rc.event}")
                    lines.append(f"           A: {rc.unit_a}")
                    lines.append(f"           B: {rc.unit_b}")
                    lines.append(f"           shared: {rc.shared_resource}")
                    if detail:
                        lines.append(f"           → {rc.description}")
                    lines.append("")
        lines.append("")

        # 5. 超时风险
        lines.append("── [5] 超时风险分析 (TimeoutStartSec 检查)")
        lines.append("")
        if not report.timeout_risks:
            lines.append("  [OK] 未检测到超时风险")
        else:
            high   = [r for r in report.timeout_risks if r.risk_level == "HIGH"]
            medium = [r for r in report.timeout_risks if r.risk_level == "MEDIUM"]
            low    = [r for r in report.timeout_risks if r.risk_level == "LOW"]
            for risk in high + medium + low:
                mode = "async" if risk.is_async else "sync "
                lines.append(
                    f"  [{risk.risk_level:<6}] {risk.event:<20} "
                    f"{risk.unit_name:<42} {mode} {risk.timeout_ms}ms"
                )
                if detail:
                    lines.append(f"           → {risk.description}")
            lines.append("")

        # 6. 并行化机会
        lines.append("── [6] 并行化机会 (systemd parallel activation)")
        lines.append("")
        if not report.parallel_groups:
            lines.append("  无可并行组（所有 hooks 均为串行依赖）")
        else:
            total_saving = sum(g.estimated_saving_ms for g in report.parallel_groups)
            lines.append(
                f"  共 {len(report.parallel_groups)} 个并行组，"
                f"预估总节省 {total_saving:.0f}ms（假设每 hook 均值 200ms）"
            )
            lines.append("")
            for g in report.parallel_groups[:10]:
                lines.append(
                    f"  {g.event:<20} wave={g.wave}  "
                    f"{len(g.units)} units  ~{g.estimated_saving_ms:.0f}ms saving"
                )
                for u in g.units:
                    lines.append(f"    ║ {u}")
            lines.append("")

        # 7. 建议摘要
        lines.append("── [7] 优化建议摘要")
        lines.append("")
        high_races = [r for r in report.race_conditions if r.severity == "HIGH"]
        high_risks = [r for r in report.timeout_risks   if r.risk_level == "HIGH"]
        if report.cycle_errors:
            lines.append(f"  [!!!] {len(report.cycle_errors)} 个 event 存在循环依赖，必须立即修复")
        if high_races:
            lines.append(f"  [!]   {len(high_races)} 个 HIGH 竞争条件 → 添加 After= 约束")
        if high_risks:
            lines.append(f"  [!]   {len(high_risks)} 个 HIGH 超时风险 → 改为 async=true 或缩短 timeout")
        medium_races = [r for r in report.race_conditions if r.severity == "MEDIUM"]
        if medium_races:
            lines.append(f"  [~]   {len(medium_races)} 个 MEDIUM 竞争条件，建议添加顺序约束")
        if not (report.cycle_errors or high_races or high_risks):
            lines.append("  [OK]  无高优先级问题")
        lines.append("")
        lines.append(SEP)

        return "\n".join(lines)

    def print_report(self, detail: bool = True) -> AnalysisReport:
        """一键分析并打印（类比：直接运行 systemd-analyze）"""
        report = self.analyze()
        print(self.format_report(report, detail=detail))
        return report


# ── 命令行入口 ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Memory OS Hook Orchestration Analyzer (systemd-analyze analogy)"
    )
    parser.add_argument(
        "--settings",
        default=os.path.expanduser("~/.claude/settings.json"),
        help="Path to settings.json",
    )
    parser.add_argument(
        "--event",
        default=None,
        help="Only analyze a specific event (e.g. SessionStart)",
    )
    parser.add_argument(
        "--no-detail",
        action="store_true",
        help="Suppress detailed descriptions in report",
    )
    args = parser.parse_args()

    analyzer = HookAnalyzer(settings_path=args.settings)

    if args.event:
        n = analyzer._manager.load_units()
        print(f"\n{args.event}.target  ({n} total hooks loaded)\n")
        print(analyzer._manager.get_dependency_graph(args.event))
    else:
        report = analyzer.analyze()
        print(analyzer.format_report(report, detail=not args.no_detail))


if __name__ == "__main__":
    main()
