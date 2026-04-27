#!/usr/bin/env python3
"""
test_sched_crud.py — 迭代87: Scheduler CRUD 测试
OS 类比：CFS runqueue + task_struct 管理验证
"""
import sys, os, json, tempfile, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# tmpfs 测试隔离
_tmp_dir = tempfile.mkdtemp(prefix="sched_crud_")
os.environ["MEMORY_OS_DIR"] = _tmp_dir
os.environ["MEMORY_OS_DB"] = str(Path(_tmp_dir) / "store.db")

from store import (
    open_db, ensure_schema,
    sched_create_task, sched_update_task, sched_get_tasks,
    sched_get_task, sched_delete_task, sched_append_log,
    sched_link_decision, sched_dump_tasks, sched_restore_tasks,
    insert_chunk,
)
from schema import MemoryChunk


if __name__ == "__main__":
    passed = 0
    failed = 0

    def test(name, condition, detail=""):
        global passed, failed
        if condition:
            passed += 1
            print(f"  ✓ {name}")
        else:
            failed += 1
            print(f"  ✗ {name} — {detail}")


    PROJECT = "test:sched_crud"
    SESSION = "test-session-001"


    def setup_db():
        conn = open_db()
        ensure_schema(conn)
        return conn


    # ═══════════════════════════════════════════════════════════
    print("=" * 60)
    print("迭代87: Scheduler CRUD Tests")
    print("=" * 60)

    # ── T1: Schema 创建 ──
    print("\n[T1] Schema 创建")
    conn = setup_db()
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    test("T1a: scheduler_tasks 表存在", "scheduler_tasks" in tables)
    test("T1b: scheduler_task_decisions 表存在", "scheduler_task_decisions" in tables)

    # ── T2: 创建任务 ──
    print("\n[T2] 创建任务")
    tid1 = sched_create_task(conn, PROJECT, SESSION, "实现 scheduler CRUD", priority=50)
    test("T2a: 返回 task_id", tid1 and len(tid1) == 36, f"got: {tid1}")
    tid2 = sched_create_task(conn, PROJECT, SESSION, "编写测试套件", priority=30,
                             dependencies=[tid1])
    test("T2b: 支持依赖", tid2 and len(tid2) == 36)
    tid3 = sched_create_task(conn, PROJECT, SESSION, "更新文档", priority=10)
    test("T2c: 多任务创建", tid3 and len(tid3) == 36)

    # ── T3: 查询任务 ──
    print("\n[T3] 查询任务")
    tasks = sched_get_tasks(conn, PROJECT)
    test("T3a: 查询所有返回3个", len(tasks) == 3, f"got: {len(tasks)}")
    test("T3b: 按优先级排序（高→低）",
         tasks[0]["priority"] >= tasks[-1]["priority"])

    pending = sched_get_tasks(conn, PROJECT, status="pending")
    test("T3c: 按状态过滤", len(pending) == 3)

    task1 = sched_get_task(conn, tid1)
    test("T3d: 获取单个任务", task1 and task1["task_name"] == "实现 scheduler CRUD")
    test("T3e: 默认 status=pending", task1["status"] == "pending")
    test("T3f: 默认 oom_adj=-800（受保护）", task1["oom_adj"] == -800)

    # ── T4: 更新任务 ──
    print("\n[T4] 更新任务")
    sched_update_task(conn, tid1, status="running", priority=80)
    task1 = sched_get_task(conn, tid1)
    test("T4a: status 更新为 running", task1["status"] == "running")
    test("T4b: priority 更新为 80", task1["priority"] == 80)

    # 忽略非法字段
    sched_update_task(conn, tid1, illegal_field="hack")
    task1 = sched_get_task(conn, tid1)
    test("T4c: 非法字段被忽略", "illegal_field" not in task1)

    # ── T5: 执行日志 ──
    print("\n[T5] 执行日志追加")
    sched_append_log(conn, tid1, "开始实现 CRUD", "store_core.py 已修改")
    sched_append_log(conn, tid1, "测试通过", "15/15 pass")
    task1 = sched_get_task(conn, tid1)
    test("T5a: 日志条目数=2", len(task1["execution_log"]) == 2)
    test("T5b: 日志包含 action", task1["execution_log"][0]["action"] == "开始实现 CRUD")
    test("T5c: 日志包含 result", task1["execution_log"][0]["result"] == "store_core.py 已修改")
    test("T5d: 日志包含时间戳", "ts" in task1["execution_log"][0])

    # ── T6: 决策关联 ──
    print("\n[T6] 决策关联")
    # 先创建一个 decision chunk
    chunk = MemoryChunk(
        project=PROJECT, source_session=SESSION,
        chunk_type="decision",
        content="选择 SQLite 而非 Redis", summary="使用 SQLite 作为存储",
        tags=["decision"], importance=0.8,
    )
    insert_chunk(conn, chunk.__dict__)
    dec_id = chunk.id
    sched_link_decision(conn, tid1, dec_id, "enabler")
    # 查询关联
    rows = conn.execute(
        "SELECT decision_type FROM scheduler_task_decisions WHERE task_id=? AND decision_id=?",
        (tid1, dec_id)
    ).fetchall()
    test("T6a: 关联创建成功", len(rows) == 1)
    test("T6b: decision_type=enabler", rows[0][0] == "enabler")

    # 幂等性
    sched_link_decision(conn, tid1, dec_id, "enabler")
    rows = conn.execute(
        "SELECT COUNT(*) FROM scheduler_task_decisions WHERE task_id=? AND decision_id=?",
        (tid1, dec_id)
    ).fetchall()
    test("T6c: 幂等（不重复插入）", rows[0][0] == 1)

    # ── T7: dump/restore 端到端 ──
    print("\n[T7] Dump/Restore 端到端")
    dump = sched_dump_tasks(conn, PROJECT, SESSION)
    test("T7a: dump 包含 active_tasks", "active_tasks" in dump)
    test("T7b: active_tasks 包含 running 任务", any(
        t["status"] == "running" for t in dump["active_tasks"]))
    test("T7c: running_count=1", dump["running_count"] == 1)
    test("T7d: pending_count=2", dump["pending_count"] == 2)
    test("T7e: decisions 关联恢复", len(dump["decisions"]) >= 1)

    # restore 格式化
    restore_text = sched_restore_tasks(conn, dump, PROJECT, SESSION)
    test("T7f: restore 输出非空", len(restore_text) > 0)
    test("T7g: restore 包含运行中", "▶" in restore_text or "running" in restore_text)
    test("T7h: restore 包含待办", "⏳" in restore_text or "pending" in restore_text)

    # ── T8: 删除任务 ──
    print("\n[T8] 删除任务")
    sched_delete_task(conn, tid3)
    task3 = sched_get_task(conn, tid3)
    test("T8a: 删除后查询返回 None", task3 is None)
    # 确认关联也被删除
    all_tasks = sched_get_tasks(conn, PROJECT)
    test("T8b: 剩余 2 个任务", len(all_tasks) == 2)

    # ── T9: 完成任务流程 ──
    print("\n[T9] 完成任务流程")
    sched_update_task(conn, tid1, status="completed")
    sched_update_task(conn, tid2, status="running")
    dump2 = sched_dump_tasks(conn, PROJECT, SESSION)
    test("T9a: completed_count=1", dump2["completed_count"] == 1)
    test("T9b: running_count=1 (tid2)", dump2["running_count"] == 1)

    # ── T10: 空项目场景 ──
    print("\n[T10] 空项目场景")
    empty_dump = sched_dump_tasks(conn, "nonexistent:project")
    test("T10a: 空项目 active_tasks=[]", empty_dump["active_tasks"] == [])
    empty_restore = sched_restore_tasks(conn, empty_dump, "nonexistent:project", SESSION)
    test("T10b: 空 restore 返回空字符串", empty_restore == "")

    # ── T11: 性能 ──
    print("\n[T11] 性能")
    t0 = time.perf_counter()
    for i in range(100):
        sched_create_task(conn, PROJECT, SESSION, f"perf_task_{i}", priority=i)
    elapsed_create = (time.perf_counter() - t0) * 1000
    test(f"T11a: 100 task 创建 < 500ms", elapsed_create < 500,
         f"got: {elapsed_create:.1f}ms")

    t0 = time.perf_counter()
    for _ in range(100):
        sched_get_tasks(conn, PROJECT, limit=20)
    elapsed_query = (time.perf_counter() - t0) * 1000
    test(f"T11b: 100 次查询 < 200ms", elapsed_query < 200,
         f"got: {elapsed_query:.1f}ms")

    t0 = time.perf_counter()
    for _ in range(50):
        sched_dump_tasks(conn, PROJECT, SESSION)
    elapsed_dump = (time.perf_counter() - t0) * 1000
    test(f"T11c: 50 次 dump < 500ms", elapsed_dump < 500,
         f"got: {elapsed_dump:.1f}ms")

    print(f"\n  PERF  create: {elapsed_create/100:.2f}ms/call  "
          f"query: {elapsed_query/100:.2f}ms/call  "
          f"dump: {elapsed_dump/50:.2f}ms/call")

    conn.close()

    # ── 清理 ──
    import shutil
    shutil.rmtree(_tmp_dir, ignore_errors=True)

    print(f"\n{'=' * 60}")
    print(f"Scheduler CRUD Tests: {passed} passed, {failed} failed")
    if failed == 0:
        print("ALL PASSED ✓")
    print(f"{'=' * 60}")

    sys.exit(0 if failed == 0 else 1)
