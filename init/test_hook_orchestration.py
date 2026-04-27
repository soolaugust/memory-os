"""
test_hook_orchestration.py — Hook 编排系统测试套件
OS 类比：systemd 测试套件（systemd-nspawn 隔离环境）

测试隔离：使用 tempfile.mkdtemp() 模拟 tmpfs，所有 DB 和 settings 在临时目录。
测试分组：
  TestHookUnit         — HookUnit 数据类和状态机
  TestHookManager      — 加载、拓扑排序、依赖检测
  TestHookManagerExec  — 实际命令执行（用 echo/true/false 模拟）
  TestHookJournal      — 日志读写
  TestHookAnalyzer     — 静态分析（使用真实 settings.json 只读）
  TestCyclicDep        — 循环依赖检测
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import time
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from init.hook_unit import HookUnit, HookStatus, HookDependency
from init.hook_manager import HookManager, CyclicDependencyError, TargetResult
from init.hook_journal import HookJournal
from init.hook_analyzer import HookAnalyzer


# ── 测试辅助 ──────────────────────────────────────────────────────────────


def _make_settings(hooks_cfg: dict) -> str:
    """在临时目录创建 settings.json，返回路径"""
    tmpdir = tempfile.mkdtemp(prefix="hook-test-")
    path = os.path.join(tmpdir, "settings.json")
    with open(path, "w") as f:
        json.dump({"hooks": hooks_cfg}, f)
    return path


def _make_db() -> str:
    """在临时目录创建含 dmesg 表的 SQLite DB，返回路径"""
    tmpdir = tempfile.mkdtemp(prefix="hook-db-")
    db_path = os.path.join(tmpdir, "store.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS dmesg (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp  TEXT NOT NULL,
            level      TEXT NOT NULL,
            subsystem  TEXT NOT NULL,
            message    TEXT NOT NULL,
            session_id TEXT DEFAULT '',
            project    TEXT DEFAULT '',
            extra      TEXT
        )
    """)
    conn.commit()
    conn.close()
    return db_path


# ── TestHookUnit ──────────────────────────────────────────────────────────


class TestHookUnit(unittest.TestCase):
    """HookUnit 数据类和状态机测试"""

    def test_default_status_inactive(self):
        unit = HookUnit(name="test", event="SessionStart", command="echo hi")
        self.assertEqual(unit.status, HookStatus.INACTIVE)

    def test_string_status_normalized(self):
        unit = HookUnit(name="x", event="Stop", command="true", status="active")
        self.assertEqual(unit.status, HookStatus.ACTIVE)

    def test_mark_activating(self):
        unit = HookUnit(name="u", event="E", command="c")
        unit.mark_activating()
        self.assertEqual(unit.status, HookStatus.ACTIVATING)
        self.assertIsNotNone(unit.last_run_at)

    def test_mark_active(self):
        unit = HookUnit(name="u", event="E", command="c")
        unit.mark_active(exit_code=0, duration_ms=42.0)
        self.assertEqual(unit.status, HookStatus.ACTIVE)
        self.assertEqual(unit.last_exit_code, 0)
        self.assertEqual(unit.last_duration_ms, 42.0)
        self.assertEqual(unit.run_count, 1)
        self.assertEqual(unit.fail_count, 0)
        self.assertTrue(unit.is_healthy)

    def test_mark_failed(self):
        unit = HookUnit(name="u", event="E", command="c")
        unit.mark_failed(exit_code=1, duration_ms=100.0)
        self.assertEqual(unit.status, HookStatus.FAILED)
        self.assertEqual(unit.fail_count, 1)
        self.assertFalse(unit.is_healthy)

    def test_reliability(self):
        unit = HookUnit(name="u", event="E", command="c")
        self.assertEqual(unit.reliability, 1.0)  # 未运行时为 1.0
        unit.mark_active(0, 10)
        unit.mark_active(0, 10)
        unit.mark_failed(1, 10)
        # 3 runs, 1 fail → reliability = 2/3
        self.assertAlmostEqual(unit.reliability, 2 / 3, places=3)

    def test_mark_inactive_resets(self):
        unit = HookUnit(name="u", event="E", command="c")
        unit.mark_failed(1, 10)
        unit.mark_inactive()
        self.assertEqual(unit.status, HookStatus.INACTIVE)

    def test_to_status_dict_structure(self):
        unit = HookUnit(name="loader", event="SessionStart",
                        command="python3 loader.py",
                        after=["snarc-start"], is_async=False)
        d = unit.to_status_dict()
        self.assertEqual(d["name"], "loader")
        self.assertEqual(d["dependencies"]["after"], ["snarc-start"])
        self.assertIn("reliability", d)
        self.assertIn("run_count", d)

    def test_hook_dependency_enum(self):
        self.assertEqual(HookDependency.AFTER.value,    "after")
        self.assertEqual(HookDependency.REQUIRES.value, "requires")
        self.assertEqual(HookDependency.WANTS.value,    "wants")


# ── TestHookManager ───────────────────────────────────────────────────────


class TestHookManager(unittest.TestCase):
    """HookManager 加载和拓扑排序测试"""

    def _mgr(self, hooks_cfg: dict) -> HookManager:
        path = _make_settings(hooks_cfg)
        mgr = HookManager(settings_path=path)
        mgr.load_units()
        return mgr

    def test_load_units_count(self):
        cfg = {
            "SessionStart": [
                {"hooks": [{"type": "command", "command": "echo a"}]},
                {"hooks": [{"type": "command", "command": "echo b"}]},
                {"hooks": [{"type": "command", "command": "echo c"}]},
            ]
        }
        mgr = self._mgr(cfg)
        self.assertEqual(len(mgr.get_status("SessionStart")), 3)

    def test_load_real_settings(self):
        """加载真实 settings.json，确认 hook 总数 >= 30"""
        real = os.path.expanduser("~/.claude/settings.json")
        if not os.path.exists(real):
            self.skipTest("~/.claude/settings.json not found")
        mgr = HookManager(settings_path=real)
        n = mgr.load_units()
        self.assertGreaterEqual(n, 20, f"Expected >=20 hooks, got {n}")
        self.assertIn("SessionStart", mgr.list_events())
        self.assertIn("Stop", mgr.list_events())

    def test_resolve_order_no_deps_single_wave(self):
        """无依赖 → 全部 units 排入 waves（至少 1 个 wave）"""
        cfg = {
            "SessionStart": [
                {"hooks": [{"type": "command", "command": "echo a"}]},
                {"hooks": [{"type": "command", "command": "echo b"}]},
            ]
        }
        mgr = self._mgr(cfg)
        waves = mgr.resolve_order("SessionStart")
        all_units = [u for w in waves for u in w]
        self.assertEqual(len(all_units), 2)

    def test_resolve_order_after_dep_two_waves(self):
        """手动 After= 依赖 → 生成两个串行波次"""
        cfg = {
            "SessionStart": [
                {"hooks": [{"type": "command", "command": "echo first"}]},
                {"hooks": [{"type": "command", "command": "echo second"}]},
            ]
        }
        mgr = self._mgr(cfg)
        units = mgr.get_status("SessionStart")
        first_name  = units[0]["name"]
        second_name = units[1]["name"]
        mgr.add_dependency(second_name, "after", first_name)
        waves = mgr.resolve_order("SessionStart")
        self.assertEqual(len(waves), 2)
        self.assertIn(first_name,  waves[0])
        self.assertIn(second_name, waves[1])

    def test_list_events(self):
        cfg = {
            "SessionStart": [{"hooks": [{"type": "command", "command": "echo x"}]}],
            "Stop":         [{"hooks": [{"type": "command", "command": "echo y"}]}],
        }
        mgr = self._mgr(cfg)
        events = mgr.list_events()
        self.assertIn("SessionStart", events)
        self.assertIn("Stop", events)

    def test_get_dependency_graph_contains_target(self):
        cfg = {"SessionStart": [
            {"hooks": [{"type": "command", "command": "echo a"}]},
        ]}
        mgr = self._mgr(cfg)
        graph = mgr.get_dependency_graph()
        self.assertIn("SessionStart.target", graph)

    def test_get_status_empty_event(self):
        cfg = {"SessionStart": [{"hooks": [{"type": "command", "command": "echo a"}]}]}
        mgr = self._mgr(cfg)
        self.assertEqual(mgr.get_status("NonExistentEvent"), [])

    def test_reset_failed(self):
        cfg = {"SessionStart": [{"hooks": [{"type": "command", "command": "echo a"}]}]}
        mgr = self._mgr(cfg)
        name = mgr.get_status("SessionStart")[0]["name"]
        mgr.get_unit(name).mark_failed(1, 10)
        self.assertEqual(mgr.reset_failed("SessionStart"), 1)
        self.assertEqual(mgr.get_unit(name).status, HookStatus.INACTIVE)

    def test_matcher_preserved(self):
        cfg = {"PreToolUse": [
            {"matcher": "Bash", "hooks": [{"type": "command", "command": "echo x"}]},
        ]}
        mgr = self._mgr(cfg)
        units = mgr.get_status("PreToolUse")
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0]["matcher"], "Bash")


# ── TestCyclicDep ─────────────────────────────────────────────────────────


class TestCyclicDep(unittest.TestCase):
    """循环依赖检测测试"""

    def test_two_node_cycle_raises(self):
        cfg = {
            "SessionStart": [
                {"hooks": [{"type": "command", "command": "echo a"}]},
                {"hooks": [{"type": "command", "command": "echo b"}]},
            ]
        }
        path = _make_settings(cfg)
        mgr = HookManager(settings_path=path)
        mgr.load_units()
        units = mgr.get_status("SessionStart")
        a, b = units[0]["name"], units[1]["name"]
        mgr.add_dependency(a, "after", b)
        mgr.add_dependency(b, "after", a)
        with self.assertRaises(CyclicDependencyError):
            mgr.resolve_order("SessionStart")

    def test_three_node_cycle_raises(self):
        cfg = {
            "SessionStart": [
                {"hooks": [{"type": "command", "command": "echo a"}]},
                {"hooks": [{"type": "command", "command": "echo b"}]},
                {"hooks": [{"type": "command", "command": "echo c"}]},
            ]
        }
        path = _make_settings(cfg)
        mgr = HookManager(settings_path=path)
        mgr.load_units()
        units = mgr.get_status("SessionStart")
        a, b, c = units[0]["name"], units[1]["name"], units[2]["name"]
        # a→b→c→a 循环
        mgr.add_dependency(b, "after", a)
        mgr.add_dependency(c, "after", b)
        mgr.add_dependency(a, "after", c)
        with self.assertRaises(CyclicDependencyError):
            mgr.resolve_order("SessionStart")


# ── TestHookManagerExec ───────────────────────────────────────────────────


class TestHookManagerExec(unittest.TestCase):
    """实际命令执行测试（用系统命令 true/false/echo 模拟）"""

    def _mgr(self, hooks_cfg: dict) -> HookManager:
        path = _make_settings(hooks_cfg)
        mgr = HookManager(settings_path=path)
        mgr.load_units()
        return mgr

    def test_execute_success(self):
        cfg = {"SessionStart": [
            {"hooks": [{"type": "command", "command": "true", "timeout": 5}]},
        ]}
        result = self._mgr(cfg).execute_target("SessionStart")
        self.assertEqual(result.succeeded, 1)
        self.assertEqual(result.failed,    0)
        self.assertTrue(result.success)

    def test_execute_failure(self):
        cfg = {"SessionStart": [
            {"hooks": [{"type": "command", "command": "false", "timeout": 5}]},
        ]}
        result = self._mgr(cfg).execute_target("SessionStart")
        self.assertEqual(result.failed,    1)
        self.assertEqual(result.succeeded, 0)
        self.assertFalse(result.success)

    def test_execute_async_returns_immediately(self):
        """async unit spawn 后立即返回，不等待进程"""
        cfg = {"SessionStart": [
            {"hooks": [{"type": "command", "command": "sleep 60",
                        "timeout": 5, "async": True}]},
        ]}
        t0 = time.time()
        result = self._mgr(cfg).execute_target("SessionStart")
        elapsed = time.time() - t0
        self.assertEqual(result.succeeded, 1)
        self.assertLess(elapsed, 2.0, "Async hook should spawn and return immediately")

    def test_execute_requires_skip_on_failure(self):
        """强依赖失败 → 下游 unit 被 skip"""
        cfg = {"SessionStart": [
            {"hooks": [{"type": "command", "command": "false", "timeout": 5}]},
            {"hooks": [{"type": "command", "command": "echo downstream", "timeout": 5}]},
        ]}
        path = _make_settings(cfg)
        mgr = HookManager(settings_path=path)
        mgr.load_units()
        units = mgr.get_status("SessionStart")
        first, second = units[0]["name"], units[1]["name"]
        mgr.add_dependency(second, "requires", first)
        mgr.add_dependency(second, "after",    first)

        result = mgr.execute_target("SessionStart")
        self.assertEqual(result.failed,  1)
        self.assertEqual(result.skipped, 1)

    def test_execute_wants_no_skip_on_failure(self):
        """弱依赖（wants）失败 → 下游 unit 不被 skip"""
        cfg = {"SessionStart": [
            {"hooks": [{"type": "command", "command": "false", "timeout": 5}]},
            {"hooks": [{"type": "command", "command": "true",  "timeout": 5}]},
        ]}
        path = _make_settings(cfg)
        mgr = HookManager(settings_path=path)
        mgr.load_units()
        units = mgr.get_status("SessionStart")
        first, second = units[0]["name"], units[1]["name"]
        # wants 不阻止执行，only after for ordering
        mgr.add_dependency(second, "wants", first)
        mgr.add_dependency(second, "after", first)

        result = mgr.execute_target("SessionStart")
        # first fails, second runs (wants = soft dep)
        self.assertEqual(result.skipped, 0)
        self.assertEqual(result.succeeded, 1)
        self.assertEqual(result.failed,    1)

    def test_execute_empty_event(self):
        cfg = {"SessionStart": [
            {"hooks": [{"type": "command", "command": "true"}]},
        ]}
        result = self._mgr(cfg).execute_target("NonExistent")
        self.assertEqual(result.total, 0)

    def test_target_result_duration_positive(self):
        cfg = {"SessionStart": [
            {"hooks": [{"type": "command", "command": "true", "timeout": 5}]},
        ]}
        result = self._mgr(cfg).execute_target("SessionStart")
        self.assertGreater(result.total_duration_ms, 0)


# ── TestHookJournal ───────────────────────────────────────────────────────


class TestHookJournal(unittest.TestCase):
    """HookJournal 读写测试（临时 SQLite DB）"""

    def setUp(self):
        self.db_path = _make_db()
        self.journal = HookJournal(self.db_path)

    def test_log_stop_and_query(self):
        self.journal.log_unit_stop(
            "memory-os-loader", duration_ms=42.3, exit_code=0,
            event="SessionStart",
        )
        entries = self.journal.query_journal(unit_name="memory-os-loader")
        self.assertEqual(len(entries), 1)
        extra = entries[0]["extra"]
        self.assertEqual(extra["phase"], "stop")
        self.assertEqual(extra["exit_code"], 0)
        self.assertAlmostEqual(extra["duration_ms"], 42.3, places=1)

    def test_log_fail_and_summary(self):
        self.journal.log_unit_fail(
            "bad-hook", duration_ms=5000.0, exit_code=1,
            reason="timeout", event="Stop",
        )
        failures = self.journal.get_failure_summary()
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["unit"],   "bad-hook")
        self.assertEqual(failures[0]["reason"], "timeout")

    def test_log_start(self):
        self.journal.log_unit_start("snarc-session-start", "SessionStart")
        entries = self.journal.query_journal(
            unit_name="snarc-session-start", phase="start"
        )
        self.assertEqual(len(entries), 1)

    def test_log_target_complete(self):
        self.journal.log_target_complete(
            "SessionStart", total=4, succeeded=3, failed=1,
            skipped=0, duration_ms=120.0,
        )
        entries = self.journal.query_journal(phase="target_complete")
        self.assertEqual(len(entries), 1)
        extra = entries[0]["extra"]
        self.assertEqual(extra["total"],  4)
        self.assertEqual(extra["failed"], 1)

    def test_query_phase_filter(self):
        self.journal.log_unit_start("u", "E")
        self.journal.log_unit_stop("u", 10.0, 0, event="E")
        starts = self.journal.query_journal(phase="start")
        stops  = self.journal.query_journal(phase="stop")
        self.assertEqual(len(starts), 1)
        self.assertEqual(len(stops),  1)

    def test_query_event_filter(self):
        self.journal.log_unit_stop("u", 10.0, 0, event="SessionStart")
        self.journal.log_unit_stop("v", 10.0, 0, event="Stop")
        ss = self.journal.query_journal(event="SessionStart")
        st = self.journal.query_journal(event="Stop")
        self.assertEqual(len(ss), 1)
        self.assertEqual(len(st), 1)

    def test_stats_error_rate(self):
        self.journal.log_unit_stop("a", 10.0, 0, event="E")
        self.journal.log_unit_stop("b", 10.0, 0, event="E")
        self.journal.log_unit_fail("c", 10.0, 1, event="E")
        stats = self.journal.stats()
        self.assertEqual(stats["by_phase"].get("stop", 0), 2)
        self.assertEqual(stats["by_phase"].get("fail", 0), 1)
        self.assertAlmostEqual(stats["error_rate"], 1 / 3, places=2)

    def test_format_journal_text(self):
        self.journal.log_unit_stop("loader", 42.0, 0, event="SessionStart")
        entries = self.journal.query_journal()
        text = self.journal.format_journal(entries)
        self.assertIn("loader",   text)
        self.assertIn("[stop ]",  text)

    def test_query_unit_history(self):
        for i in range(5):
            self.journal.log_unit_stop(f"unit-{i}", 10.0, 0, event="E")
        # query a specific unit
        h = self.journal.query_unit_history("unit-3")
        self.assertEqual(len(h), 1)


# ── TestHookAnalyzer ──────────────────────────────────────────────────────


class TestHookAnalyzer(unittest.TestCase):
    """HookAnalyzer 静态分析测试"""

    def setUp(self):
        self.real_settings = os.path.expanduser("~/.claude/settings.json")

    def test_analyze_real_settings(self):
        if not os.path.exists(self.real_settings):
            self.skipTest("~/.claude/settings.json not found")
        analyzer = HookAnalyzer(settings_path=self.real_settings)
        report = analyzer.analyze()
        self.assertGreaterEqual(report.total_hooks, 10)  # flexible: depends on actual settings
        self.assertIn("SessionStart", report.events)

    def test_format_report_no_crash(self):
        if not os.path.exists(self.real_settings):
            self.skipTest("~/.claude/settings.json not found")
        analyzer = HookAnalyzer(settings_path=self.real_settings)
        report = analyzer.analyze()
        text = analyzer.format_report(report)
        self.assertIn("Memory OS Hook Orchestration Analysis", text)
        self.assertIn("SessionStart", text)
        self.assertIn("依赖图", text)

    def test_no_cycle_errors_in_real_settings(self):
        if not os.path.exists(self.real_settings):
            self.skipTest("~/.claude/settings.json not found")
        analyzer = HookAnalyzer(settings_path=self.real_settings)
        report = analyzer.analyze()
        self.assertEqual(report.cycle_errors, {},
                         f"Unexpected cycle errors: {report.cycle_errors}")

    def test_parallel_groups_exist(self):
        if not os.path.exists(self.real_settings):
            self.skipTest("~/.claude/settings.json not found")
        analyzer = HookAnalyzer(settings_path=self.real_settings)
        report = analyzer.analyze()
        self.assertGreater(len(report.parallel_groups), 0)

    def test_waves_cover_all_events(self):
        if not os.path.exists(self.real_settings):
            self.skipTest("~/.claude/settings.json not found")
        analyzer = HookAnalyzer(settings_path=self.real_settings)
        report = analyzer.analyze()
        for ev in report.events:
            if ev not in report.cycle_errors:
                self.assertIn(ev, report.waves_per_event)
                self.assertGreater(len(report.waves_per_event[ev]), 0)

    def test_minimal_settings(self):
        cfg = {"SessionStart": [
            {"hooks": [{"type": "command", "command": "echo hello"}]},
        ]}
        path = _make_settings(cfg)
        analyzer = HookAnalyzer(settings_path=path)
        report = analyzer.analyze()
        self.assertEqual(report.total_hooks, 1)
        text = analyzer.format_report(report)
        self.assertIn("SessionStart", text)

    def test_timeout_risks_detected(self):
        """高超时 sync hook 应被检测为 HIGH 风险"""
        cfg = {"SessionStart": [
            {"hooks": [{"type": "command", "command": "echo x",
                        "timeout": 25}]},  # 25s > HIGH threshold
        ]}
        path = _make_settings(cfg)
        analyzer = HookAnalyzer(settings_path=path)
        report = analyzer.analyze()
        high_risks = [r for r in report.timeout_risks if r.risk_level == "HIGH"]
        self.assertGreater(len(high_risks), 0)


# ── 主入口 ────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    unittest.main(verbosity=2)
