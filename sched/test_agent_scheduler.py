#!/usr/bin/env python3
"""
test_agent_scheduler.py — sched/ 子系统完整测试套件

迭代 81：Memory OS Agent 调度器测试

测试覆盖：
  1. AgentTask 数据结构（nice clamp, budget 默认值, weight 查表）
  2. Scheduler.submit() — vruntime 公平起点（CFS place_entity）
  3. Scheduler.pick_next() — 选 vruntime 最小任务（CFS 红黑树）
  4. Scheduler.update_vruntime() — token→vruntime 换算（按 weight）
  5. Scheduler.preempt() — 强制抢占
  6. Scheduler.complete() — 正常完成
  7. 超预算自动 preempt（token_budget 触发）
  8. get_stats() — 调度器统计
  9. CGroupManager.create_cgroup() — 资源组创建（幂等）
  10. CGroupManager.check_quota() — 配额检查
  11. CGroupManager.charge_tokens() — 扣减配额
  12. CGroupManager.get_agent_count() — 活跃 agent 数
  13. AgentMonitor.sched_debug() — 全局快照
  14. AgentMonitor.proc_task() — 单任务详情
  15. AgentMonitor.detect_timeouts() — 超时检测与自动 preempt
  16. AgentMonitor.summary() — 人类可读输出
  17. 公平性验证：相同 budget 下高优先级 vruntime 增量 < 低优先级
  18. 多 cgroup 隔离：foreground agent 不影响 background 配额
  19. dmesg 桥接（store.db 存在时写入）
  20. tmpfs 隔离（测试用独立数据库，不污染生产数据）

隔离策略：
  所有测试使用 tmpfs 目录（MEMORY_OS_DIR + SCHED_DB 环境变量覆盖）
  测试完成后自动清理
"""

import os
import sys
import tempfile
import shutil
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ── 测试环境隔离（必须在 import sched 之前设置）──────────────────────────────
_TMPDIR = tempfile.mkdtemp(prefix="memoryos_sched_test_")
os.environ["MEMORY_OS_DIR"] = _TMPDIR
os.environ["SCHED_DB"] = str(Path(_TMPDIR) / "sched.db")
os.environ["MEMORY_OS_DB"] = str(Path(_TMPDIR) / "store.db")

# 加入父目录到 sys.path，以便 from sched import ... 正确工作
sys.path.insert(0, str(Path(__file__).parent.parent))

from sched.agent_scheduler import (
    Scheduler, AgentTask, TaskStatus, NiceLevel,
    NICE_WEIGHT_TABLE, NICE_0_WEIGHT, _default_budget,
)
from sched.agent_cgroup import CGroupManager, CGroup, BUILTIN_CGROUPS
from sched.agent_monitor import AgentMonitor

# ── 测试统计 ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    passed = 0
    failed = 0
    total = 0


    def check(name: str, cond: bool, detail: str = "") -> bool:
        global passed, failed, total
        total += 1
        if cond:
            passed += 1
            print(f"  ✅ [{total:02d}] {name}")
            return True
        else:
            failed += 1
            print(f"  ❌ [{total:02d}] {name}" + (f": {detail}" if detail else ""))
            return False


    def section(title: str) -> None:
        print(f"\n{'─' * 60}")
        print(f"  {title}")
        print(f"{'─' * 60}")


    # ── 辅助：每测试组独立数据库（隔离状态污染）─────────────────────────────────
    _db_counter = [0]


    def fresh_db() -> Path:
        """每次调用返回新的独立数据库路径（防止测试间状态污染）"""
        _db_counter[0] += 1
        return Path(_TMPDIR) / f"sched_{_db_counter[0]}.db"


    # 全局共享 db（仅用于 monitor/cgroup 需与 scheduler 共享同一 db 的测试）
    _shared_db: Path = None


    def make_scheduler(db: Path = None) -> Scheduler:
        return Scheduler(db_path=db or fresh_db())


    def make_scheduler_shared() -> Scheduler:
        global _shared_db
        if _shared_db is None:
            _shared_db = fresh_db()
        return Scheduler(db_path=_shared_db)


    def make_cgroup_mgr(db: Path = None) -> CGroupManager:
        return CGroupManager(db_path=db or _shared_db or fresh_db())


    def make_monitor(db: Path = None) -> AgentMonitor:
        return AgentMonitor(db_path=db or _shared_db or fresh_db(), timeout_seconds=1)


    # ═══════════════════════════════════════════════════════════════════════════════
    # 测试组 1：AgentTask 数据结构
    # ═══════════════════════════════════════════════════════════════════════════════
    section("1. AgentTask 数据结构")

    t = AgentTask(name="test_task", nice=0)
    check("task_id 自动生成 UUID", len(t.task_id) == 36)
    check("默认 status = PENDING", t.status == TaskStatus.PENDING)
    check("nice=0 时 weight=1024", t.weight == NICE_0_WEIGHT)
    check("nice=0 默认 budget = 20000", t.token_budget == 20000)

    t_rt = AgentTask(name="realtime", nice=-20)
    check("REALTIME nice=-20 weight=88761", t_rt.weight == 88761)
    check("REALTIME budget=50000", t_rt.token_budget == 50000)

    t_idle = AgentTask(name="idle", nice=19)
    check("IDLE nice=19 weight=15", t_idle.weight == 15)
    check("IDLE budget=5000", t_idle.token_budget == 5000)

    # nice clamp
    t_oob = AgentTask(name="out_of_range", nice=99)
    check("nice 超出范围自动 clamp 到 19", t_oob.nice == 19)

    t_oob2 = AgentTask(name="out_of_range2", nice=-99)
    check("nice 低于范围自动 clamp 到 -20", t_oob2.nice == -20)

    # budget_remaining
    t2 = AgentTask(name="budget_test", nice=0, token_budget=1000)
    t2.token_used = 600
    check("budget_remaining = 400", t2.budget_remaining == 400)
    check("budget_pct ≈ 60%", abs(t2.budget_pct - 60.0) < 0.01)
    check("is_over_budget False (600<1000)", not t2.is_over_budget())
    t2.token_used = 1000
    check("is_over_budget True (1000>=1000)", t2.is_over_budget())

    # ═══════════════════════════════════════════════════════════════════════════════
    # 测试组 2：Scheduler.submit() — vruntime 公平起点
    # ═══════════════════════════════════════════════════════════════════════════════
    section("2. Scheduler.submit() — CFS place_entity")

    _db2 = fresh_db()
    sched = Scheduler(db_path=_db2)

    # 空队列提交，vruntime 从 0 开始
    task_a = AgentTask(name="agent_A", nice=0)
    sched.submit(task_a)
    check("空队列 submit：vruntime = 0.0", task_a.vruntime == 0.0)

    # 第二个任务继承当前 min_vruntime
    sched.update_vruntime(task_a.task_id, 500)  # A 消耗 500 tokens
    task_b = AgentTask(name="agent_B", nice=0)
    sched.submit(task_b)
    # B 的 vruntime 应当等于 A 的当前 vruntime（min_vruntime）
    a_after = sched.get_task(task_a.task_id)
    b_after = sched.get_task(task_b.task_id)
    check(
        "新任务 vruntime = min_vruntime（不饿死老任务）",
        abs(b_after.vruntime - a_after.vruntime) < 1.0,
        f"B.vruntime={b_after.vruntime:.2f}, A.vruntime={a_after.vruntime:.2f}",
    )
    sched.close()

    # ═══════════════════════════════════════════════════════════════════════════════
    # 测试组 3：Scheduler.pick_next() — 选 vruntime 最小任务
    # ═══════════════════════════════════════════════════════════════════════════════
    section("3. Scheduler.pick_next() — CFS 红黑树最左节点")

    _db3 = fresh_db()
    sched = Scheduler(db_path=_db3)

    # 提交 3 个不同 vruntime 的任务
    t_low = AgentTask(name="low_vruntime", nice=0)
    t_low.vruntime = 100.0
    t_mid = AgentTask(name="mid_vruntime", nice=0)
    t_mid.vruntime = 500.0
    t_high = AgentTask(name="high_vruntime", nice=0)
    t_high.vruntime = 1000.0

    # 故意乱序提交
    sched.submit(t_high)
    sched.submit(t_low)
    sched.submit(t_mid)

    # pick_next 应选 vruntime=100 的任务
    picked = sched.pick_next()
    check("pick_next 选择 vruntime 最小任务", picked is not None and picked.name == "low_vruntime",
          f"picked={picked.name if picked else None}")
    check("pick_next 后状态变为 RUNNING", picked is not None and picked.status == TaskStatus.RUNNING)

    # 再次 pick_next 选第二小
    picked2 = sched.pick_next()
    check("第二次 pick_next 选次小 vruntime", picked2 is not None and picked2.name == "mid_vruntime")

    # 空队列检查（remaining: high_vruntime 还是 pending）
    sched.complete(t_high.task_id)
    sched.complete(t_mid.task_id)
    sched.complete(picked.task_id)
    empty = sched.pick_next()
    check("所有任务完成后 pick_next 返回 None", empty is None)
    sched.close()

    # ═══════════════════════════════════════════════════════════════════════════════
    # 测试组 4：update_vruntime — vruntime 换算验证
    # ═══════════════════════════════════════════════════════════════════════════════
    section("4. update_vruntime — CFS delta 换算（delta * NICE_0_WEIGHT / weight）")

    _db4 = fresh_db()
    sched = Scheduler(db_path=_db4)

    # nice=0（weight=1024）：消耗 1024 tokens → vruntime += 1024 * 1024/1024 = 1024.0
    t_n0 = AgentTask(name="nice0_task", nice=0, token_budget=100000)
    sched.submit(t_n0)
    sched.pick_next()  # 设置为 running
    v_n0_before = sched.get_task(t_n0.task_id).vruntime
    sched.update_vruntime(t_n0.task_id, 1024)
    t_n0_after = sched.get_task(t_n0.task_id)
    delta_n0 = t_n0_after.vruntime - v_n0_before
    check(
        "nice=0: delta 1024 → vruntime += 1024.0",
        abs(delta_n0 - 1024.0) < 0.01,
        f"delta={delta_n0:.4f}",
    )

    # nice=-10（weight=9548）：消耗 1024 tokens → vruntime += 1024 * 1024/9548 ≈ 109.9
    # 使用独立数据库确保 vruntime 从 0 开始
    _db4b = fresh_db()
    sched2 = Scheduler(db_path=_db4b)
    t_nm10 = AgentTask(name="nice-10_task", nice=-10, token_budget=100000)
    sched2.submit(t_nm10)
    sched2.pick_next()
    v_nm10_before = sched2.get_task(t_nm10.task_id).vruntime
    sched2.update_vruntime(t_nm10.task_id, 1024)
    t_nm10_after = sched2.get_task(t_nm10.task_id)
    delta_nm10 = t_nm10_after.vruntime - v_nm10_before
    expected_delta_nm10 = 1024 * (NICE_0_WEIGHT / NICE_WEIGHT_TABLE[-10])
    check(
        "nice=-10: vruntime 增量按权重缩放",
        abs(delta_nm10 - expected_delta_nm10) < 1.0,
        f"expected≈{expected_delta_nm10:.2f}, got_delta={delta_nm10:.2f}",
    )

    # 公平性验证：相同 token 消耗，高优先级 vruntime 增量更小
    check(
        "公平性：nice=-10 vruntime 增量 < nice=0",
        delta_nm10 < delta_n0,
        f"nice=-10 delta={delta_nm10:.2f}, nice=0 delta={delta_n0:.2f}",
    )
    sched.close()
    sched2.close()

    # ═══════════════════════════════════════════════════════════════════════════════
    # 测试组 5：超预算自动 preempt
    # ═══════════════════════════════════════════════════════════════════════════════
    section("5. 超预算自动 preempt（token_budget_exceeded）")

    _db5 = fresh_db()
    sched = Scheduler(db_path=_db5)
    t_budget = AgentTask(name="budget_task", nice=0, token_budget=500)
    sched.submit(t_budget)
    sched.pick_next()

    # 消耗到接近预算（450/500，不触发）
    sched.update_vruntime(t_budget.task_id, 450)
    after_450 = sched.get_task(t_budget.task_id)
    check("450/500 tokens 时 status 仍为 running", after_450.status == TaskStatus.RUNNING)

    # 超预算（再消耗 100 → 550 > 500）
    sched.update_vruntime(t_budget.task_id, 100)
    after_550 = sched.get_task(t_budget.task_id)
    check("550/500 tokens 时自动 preempt", after_550.status == TaskStatus.PREEMPTED)
    check("preempt_reason 包含 budget_exceeded", "token_budget_exceeded" in (after_550.preempt_reason or ""))
    sched.close()

    # ═══════════════════════════════════════════════════════════════════════════════
    # 测试组 6：手动 preempt 与 complete
    # ═══════════════════════════════════════════════════════════════════════════════
    section("6. 手动 preempt 和 complete")

    _db6 = fresh_db()
    sched = Scheduler(db_path=_db6)

    t_preempt = AgentTask(name="preempt_me", nice=0)
    sched.submit(t_preempt)
    sched.pick_next()
    result = sched.preempt(t_preempt.task_id, reason="manual_test")
    check("手动 preempt 返回任务", result is not None)
    check("preempt 后 status = PREEMPTED", result.status == TaskStatus.PREEMPTED)
    check("preempt_reason 正确", result.preempt_reason == "manual_test")

    # 完成后状态不可再抢占
    sched.complete(t_preempt.task_id)  # 从 PREEMPTED 到 COMPLETED（允许）
    already_done = AgentTask(name="done_task", nice=0)
    sched.submit(already_done)
    sched.pick_next()
    sched.complete(already_done.task_id)
    re_preempt = sched.preempt(already_done.task_id, reason="late_preempt")
    check("已完成任务 preempt 无效（返回原任务）",
          re_preempt is not None and re_preempt.status == TaskStatus.COMPLETED)

    # complete 验证
    t_comp = AgentTask(name="complete_me", nice=0)
    sched.submit(t_comp)
    sched.pick_next()
    result_c = sched.complete(t_comp.task_id, tokens_final=200)
    check("complete 返回任务", result_c is not None)
    check("complete 后 status = COMPLETED", result_c.status == TaskStatus.COMPLETED)
    check("complete 后 ended_at 已设置", result_c.ended_at is not None)
    sched.close()

    # ═══════════════════════════════════════════════════════════════════════════════
    # 测试组 7：get_stats()
    # ═══════════════════════════════════════════════════════════════════════════════
    section("7. Scheduler.get_stats() — /proc/sched_debug 等价")

    _db7 = fresh_db()
    sched = Scheduler(db_path=_db7)
    # ta: 先 submit+pick → RUNNING
    ta = AgentTask(name="stat_a", nice=0)
    sched.submit(ta)
    sched.pick_next()  # ta → RUNNING

    # tc: submit + pick + complete（在 tb 之前 submit，pick 时 tc vruntime 最小）
    tc = AgentTask(name="stat_c", nice=0)
    sched.submit(tc)
    sched.pick_next()  # tc → RUNNING（ta 已在 running 不影响 pending 队列）
    sched.complete(tc.task_id)  # tc → COMPLETED

    # tb: 最后 submit，保持 PENDING
    tb = AgentTask(name="stat_b", nice=0)
    sched.submit(tb)

    stats = sched.get_stats()
    check("get_stats 有 nr_running 字段", "nr_running" in stats)
    check("nr_running >= 1", stats["nr_running"] >= 1)
    check("nr_pending >= 1", stats["nr_pending"] >= 1,
          f"nr_pending={stats['nr_pending']}")
    check("nr_completed >= 1", stats["nr_completed"] >= 1)
    check("total_tokens >= 0", stats["total_tokens"] >= 0)
    check("recent_events 是列表", isinstance(stats["recent_events"], list))
    sched.close()

    # ═══════════════════════════════════════════════════════════════════════════════
    # 测试组 8：CGroupManager
    # ═══════════════════════════════════════════════════════════════════════════════
    section("8. CGroupManager — cgroup v2 资源组")

    _db8 = fresh_db()
    cgm = CGroupManager(db_path=_db8)

    # 内置 cgroup 存在
    fg = cgm.get_cgroup("foreground")
    check("内置 foreground cgroup 存在", fg is not None)
    check("foreground quota=100000", fg is not None and fg.token_quota == 100_000)
    check("foreground weight=1024", fg is not None and fg.weight == 1024)

    bg = cgm.get_cgroup("background")
    check("内置 background cgroup 存在", bg is not None)
    check("background quota=50000", bg is not None and bg.token_quota == 50_000)

    # 创建自定义 cgroup
    custom = cgm.create_cgroup("custom_test", token_quota=30000, weight=512, max_agents=3)
    check("自定义 cgroup 创建成功", custom is not None)
    check("自定义 cgroup quota=30000", custom.token_quota == 30000)

    # 幂等性
    custom2 = cgm.create_cgroup("custom_test", token_quota=99999)
    check("create_cgroup 幂等（重复创建返回原 cgroup）", custom2.token_quota == 30000)

    # check_quota — 正常允许
    result = cgm.check_quota("foreground", new_agent=True)
    check("foreground check_quota allowed=True（初始状态）", result["allowed"])

    # check_quota — 配额耗尽
    cgm._conn.execute(
        "UPDATE agent_cgroups SET token_used=100001 WHERE cgroup_name='foreground'"
    )
    cgm._conn.commit()
    result_over = cgm.check_quota("foreground")
    check("foreground 超配额时 check_quota allowed=False", not result_over["allowed"])
    check("拒绝原因包含 quota exhausted", "exhausted" in result_over["reason"])
    # 重置
    cgm._conn.execute(
        "UPDATE agent_cgroups SET token_used=0 WHERE cgroup_name='foreground'"
    )
    cgm._conn.commit()

    # check_quota — 不存在的 cgroup
    result_miss = cgm.check_quota("nonexistent_cgroup")
    check("不存在 cgroup check_quota allowed=False", not result_miss["allowed"])

    # charge_tokens
    cgm.charge_tokens("background", 1000)
    bg_after = cgm.get_cgroup("background")
    check("charge_tokens 正确扣减", bg_after is not None and bg_after.token_used >= 1000)

    # reset_quota
    cgm.reset_quota("background")
    bg_reset = cgm.get_cgroup("background")
    check("reset_quota 后 token_used=0", bg_reset is not None and bg_reset.token_used == 0)

    # add_to_cgroup（与 cgm 共享同一 _db8）
    sched_for_cg = Scheduler(db_path=_db8)
    t_cg = AgentTask(name="cgroup_task", nice=0, cgroup_name="foreground")
    sched_for_cg.submit(t_cg)
    ok = cgm.add_to_cgroup(t_cg.task_id, "background")
    check("add_to_cgroup 成功", ok)
    t_after_move = sched_for_cg.get_task(t_cg.task_id)
    check("add_to_cgroup 后 task.cgroup_name 更新", t_after_move is not None and t_after_move.cgroup_name == "background")
    sched_for_cg.close()

    # agent count
    count = cgm.get_agent_count("background", active_only=True)
    check("get_agent_count 返回整数", isinstance(count, int))

    # 删除自定义 cgroup（清空活跃 agent 后）
    can_delete = cgm.delete_cgroup("custom_test")
    check("无活跃 agent 的自定义 cgroup 可删除", can_delete)

    # 内置 cgroup 不可删
    cannot_delete = cgm.delete_cgroup("foreground")
    check("内置 cgroup 不可删除", not cannot_delete)

    cgm.close()

    # ═══════════════════════════════════════════════════════════════════════════════
    # 测试组 9：AgentMonitor
    # ═══════════════════════════════════════════════════════════════════════════════
    section("9. AgentMonitor — /proc/sched_debug + watchdog")

    # 测试组 9 使用独立数据库（monitor/scheduler/cgroup 共享此 db）
    _db9 = fresh_db()

    # 提交几个任务作为监控基础
    sched_m = Scheduler(db_path=_db9)
    t_m1 = AgentTask(name="monitor_task_1", nice=-5)
    t_m2 = AgentTask(name="monitor_task_2", nice=5)
    sched_m.submit(t_m1)
    sched_m.submit(t_m2)
    sched_m.pick_next()  # t_m1 进入 running（vruntime 小）
    sched_m.update_vruntime(t_m1.task_id, 300)
    sched_m.close()

    monitor = AgentMonitor(db_path=_db9, timeout_seconds=1)

    # sched_debug()
    debug = monitor.sched_debug()
    check("sched_debug 返回 dict", isinstance(debug, dict))
    check("sched_debug 包含 scheduler_stats", "scheduler_stats" in debug)
    check("sched_debug 包含 runqueue", "runqueue" in debug)
    check("sched_debug 包含 cgroup_stats", "cgroup_stats" in debug)
    check("sched_debug 包含 top_consumers", "top_consumers" in debug)
    check("sched_debug 包含 timestamp", "timestamp" in debug)

    # proc_task()
    proc = monitor.proc_task(t_m1.task_id)
    check("proc_task 返回 dict", proc is not None)
    check("proc_task 包含 se.vruntime", "se.vruntime" in proc)
    check("proc_task 包含 se.sum_exec_tokens", "se.sum_exec_tokens" in proc)
    check("proc_task token_used >= 300", proc["se.sum_exec_tokens"] >= 300)
    check("proc_task nice=-5", proc["nice"] == -5)
    check("proc_task prio=115 (120+(-5))", proc["prio"] == 115)

    # proc_task 不存在
    missing = monitor.proc_task("nonexistent-task-id")
    check("proc_task 不存在返回 None", missing is None)

    # cgroup_stats()
    cgstats = monitor.cgroup_stats()
    check("cgroup_stats 返回列表", isinstance(cgstats, list))
    check("cgroup_stats 包含 foreground", any(c["name"] == "foreground" for c in cgstats))

    # summary()
    summ = monitor.summary()
    check("summary 返回字符串", isinstance(summ, str))
    check("summary 包含 Memory OS Agent Scheduler", "Memory OS Agent Scheduler" in summ)
    check("summary 包含 Tasks:", "Tasks:" in summ)

    # detect_timeouts — 注入一个 "started_at 很早的 running 任务"
    import sqlite3 as _sqlite3
    old_start = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
    db_conn = _sqlite3.connect(str(_db9))
    db_conn.row_factory = _sqlite3.Row
    db_conn.execute(
        """INSERT INTO agent_tasks
           (task_id, name, agent_type, nice, vruntime, token_used, token_budget,
            status, cgroup_name, created_at, started_at, ended_at, preempt_reason,
            session_id, project, extra)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("timeout-test-1234", "timeout_task", "worker", 0, 0.0, 100, 20000,
         "running", "background", old_start, old_start, None, None, "", "", None),
    )
    db_conn.commit()
    db_conn.close()

    # timeout_seconds=1，注入的任务 started 10s 前，必然超时
    preempted_list = monitor.detect_timeouts()
    check("detect_timeouts 检测到超时任务", len(preempted_list) >= 1)
    check("超时任务包含正确 task name", any("timeout_task" in p["name"] for p in preempted_list))

    # 确认任务被标记为 preempted
    sched_check = Scheduler(db_path=_db9)
    timeout_task = sched_check.get_task("timeout-test-1234")
    check("超时任务 status = PREEMPTED", timeout_task is not None and timeout_task.status == TaskStatus.PREEMPTED)
    check("preempt_reason 包含 timeout", "timeout" in (timeout_task.preempt_reason or ""))
    sched_check.close()

    monitor.close()

    # ═══════════════════════════════════════════════════════════════════════════════
    # 测试组 10：公平性端到端验证
    # ═══════════════════════════════════════════════════════════════════════════════
    section("10. 公平性端到端：nice 调度顺序验证")

    _db10 = fresh_db()
    sched = Scheduler(db_path=_db10)

    # 提交 5 个不同 nice 的任务，验证 pick_next 顺序
    tasks_nice = [
        AgentTask(name="idle_task",    nice=19),
        AgentTask(name="normal_task",  nice=0),
        AgentTask(name="high_task",    nice=-10),
        AgentTask(name="rt_task",      nice=-20),
        AgentTask(name="low_task",     nice=10),
    ]
    for t in tasks_nice:
        sched.submit(t)

    # 先选 vruntime 最小的（都从 0 开始，tie break 按 created_at），
    # 验证所有任务都能被调度到（无饿死）
    picked_names = []
    for _ in range(5):
        p = sched.pick_next()
        if p:
            picked_names.append(p.name)
            sched.complete(p.task_id)

    check("5 个不同 nice 任务全部被调度", len(picked_names) == 5,
          f"picked={picked_names}")
    check("无任务被饿死", set(t.name for t in tasks_nice) == set(picked_names))
    sched.close()

    # ═══════════════════════════════════════════════════════════════════════════════
    # 测试组 11：多 cgroup 隔离
    # ═══════════════════════════════════════════════════════════════════════════════
    section("11. 多 cgroup 配额隔离")

    _db11 = fresh_db()
    cgm = CGroupManager(db_path=_db11)

    # 将 background 配额扣到刚好耗尽（>= quota）
    cgm.charge_tokens("background", 50001)  # 50001 > 50000 → is_throttled = True
    bg = cgm.get_cgroup("background")
    check("background 已消耗 >= 50000", bg is not None and bg.token_used >= 50000)

    # foreground 不受影响
    fg = cgm.get_cgroup("foreground")
    check("foreground token_used 不受 background 影响", fg is not None and fg.token_used < 50000)

    # background 配额检查应受限
    result_bg = cgm.check_quota("background")
    check("background 耗尽后 check_quota=False", not result_bg["allowed"],
          f"allowed={result_bg['allowed']}, reason={result_bg['reason']}")

    # foreground 配额检查不受影响
    result_fg = cgm.check_quota("foreground")
    check("foreground 配额检查不受 background 影响", result_fg["allowed"])

    cgm.reset_quota("background")
    cgm.close()

    # ═══════════════════════════════════════════════════════════════════════════════
    # 测试组 13：AIMD 自适应配额调整
    # ═══════════════════════════════════════════════════════════════════════════════
    section("13. AIMD Allocator — 自适应配额调整（TCP AIMD 类比）")

    _db13 = fresh_db()
    cgm_aimd = CGroupManager(db_path=_db13)

    # 初始状态：foreground quota = 100000
    fg_initial = cgm_aimd.get_cgroup("foreground")
    check("foreground 初始 quota = 100000", fg_initial is not None and fg_initial.token_quota == 100_000)

    # 测试1：空闲时加性增（queue_depth == 0）
    updated = cgm_aimd.aimd_adjust("foreground", queue_depth=0, threshold=3, increase_rate=0.10)
    check("空闲时 AIMD 加性增返回非 None", updated is not None)
    expected_increased = int(100_000 * 1.10)  # 110000
    check(f"空闲时 quota 加性增 10% → {expected_increased}",
          updated is not None and updated.token_quota == expected_increased,
          f"actual={updated.token_quota if updated else 'None'}")

    # 测试2：正常区间（1 <= queue_depth <= threshold）不调整
    quota_before = cgm_aimd.get_cgroup("foreground").token_quota
    result_normal = cgm_aimd.aimd_adjust("foreground", queue_depth=2, threshold=3)
    check("正常区间不调整 quota",
          result_normal is not None and result_normal.token_quota == quota_before,
          f"quota_before={quota_before} after={result_normal.token_quota if result_normal else 'None'}")

    # 测试3：过载时乘性减（queue_depth > threshold）
    quota_before_overload = cgm_aimd.get_cgroup("foreground").token_quota
    overloaded = cgm_aimd.aimd_adjust("foreground", queue_depth=5, threshold=3)
    expected_halved = int(quota_before_overload * 0.5)
    check(f"过载时 AIMD 乘性减 → {expected_halved}",
          overloaded is not None and overloaded.token_quota == expected_halved,
          f"quota_before={quota_before_overload} actual={overloaded.token_quota if overloaded else 'None'}")

    # 测试4：min_quota 下限保护
    # 将 background（50000）连续乘性减 20 次，确保不低于 min_quota=5000
    for _ in range(20):
        cgm_aimd.aimd_adjust("background", queue_depth=10, threshold=3, min_quota=5_000)
    bg_after = cgm_aimd.get_cgroup("background")
    check("乘性减不低于 min_quota 5000",
          bg_after is not None and bg_after.token_quota >= 5_000,
          f"actual={bg_after.token_quota if bg_after else 'None'}")

    # 测试5：max_quota 上限保护
    # 将 system（20000）连续加性增 100 次，确保不超过 max_quota=500000
    for _ in range(100):
        cgm_aimd.aimd_adjust("system", queue_depth=0, increase_rate=0.10, max_quota=500_000)
    sys_after = cgm_aimd.get_cgroup("system")
    check("加性增不超过 max_quota 500000",
          sys_after is not None and sys_after.token_quota <= 500_000,
          f"actual={sys_after.token_quota if sys_after else 'None'}")

    # 测试6：无限配额组（unlimited）不参与 AIMD
    unlimited_before = cgm_aimd.get_cgroup("unlimited").token_quota
    result_unlimited = cgm_aimd.aimd_adjust("unlimited", queue_depth=0)
    unlimited_after = cgm_aimd.get_cgroup("unlimited").token_quota
    check("无限配额组 AIMD 调整为 no-op",
          unlimited_before == unlimited_after,
          f"before={unlimited_before} after={unlimited_after}")

    # 测试7：不存在的 cgroup 返回 None
    result_none = cgm_aimd.aimd_adjust("nonexistent_cgroup", queue_depth=0)
    check("不存在的 cgroup 返回 None", result_none is None)

    cgm_aimd.close()

    # ═══════════════════════════════════════════════════════════════════════════════
    # 测试组 12：dmesg 桥接（store.db 存在时）
    # ═══════════════════════════════════════════════════════════════════════════════
    section("12. dmesg 桥接到 store.db（可选集成）")

    store_db_path = Path(_TMPDIR) / "store.db"
    # 创建模拟的 store.db 并建 dmesg 表
    import sqlite3 as _sqlite3
    store_conn = _sqlite3.connect(str(store_db_path))
    store_conn.execute("""
        CREATE TABLE IF NOT EXISTS dmesg (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            level TEXT NOT NULL,
            subsystem TEXT NOT NULL,
            message TEXT NOT NULL,
            session_id TEXT,
            project TEXT,
            extra TEXT
        )
    """)
    store_conn.commit()
    store_conn.close()

    # 触发一个超预算 preempt（会写 dmesg WARN）
    sched = make_scheduler()
    t_dmesg = AgentTask(name="dmesg_test_task", nice=0, token_budget=50)
    sched.submit(t_dmesg)
    sched.pick_next()
    sched.update_vruntime(t_dmesg.task_id, 100)  # 触发超预算
    sched.close()

    # 检查 dmesg 表是否有写入
    store_conn2 = _sqlite3.connect(str(store_db_path))
    rows = store_conn2.execute(
        "SELECT * FROM dmesg WHERE subsystem='sched'"
    ).fetchall()
    store_conn2.close()
    check("dmesg 桥接写入了调度事件", len(rows) >= 1)
    check("dmesg 包含 WARN 级别", any(r[2] == "WARN" for r in rows))

    # ═══════════════════════════════════════════════════════════════════════════════
    # 最终报告
    # ═══════════════════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 60}")
    print(f"  测试结果：{passed}/{total} 通过，{failed} 失败")
    print(f"{'═' * 60}")

    # 清理 tmpfs
    shutil.rmtree(_TMPDIR, ignore_errors=True)

    if failed > 0:
        sys.exit(1)
    else:
        print("  ✅ 所有测试通过！sched/ 子系统验证完成。")
        sys.exit(0)
