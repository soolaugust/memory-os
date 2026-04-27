"""
store_sched.py — Scheduler CRUD

从 store_core.py 拆分（迭代87 功能集）。
包含：sched_create_task, sched_update_task, sched_get_tasks, sched_get_task,
      sched_delete_task, sched_append_log, sched_link_decision,
      sched_dump_tasks, sched_restore_tasks。

OS 类比：Linux CFS runqueue management (2007)
"""
import json
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from store_vfs import open_db, ensure_schema, STORE_DB

# ── 迭代87：Scheduler CRUD（OS 类比：CFS runqueue management）──

def sched_create_task(conn: sqlite3.Connection, project: str, session_id: str,
                      task_name: str, priority: int = 0, dependencies: list = None,
                      due_at: str = None) -> str:
    """
    创建调度任务。OS 类比：fork() + sched_setattr() — 创建新任务并设置调度属性。
    返回 task_id。
    """
    import uuid as _uuid
    now = datetime.now(timezone.utc).isoformat()
    task_id = str(_uuid.uuid4())
    conn.execute("""
        INSERT INTO scheduler_tasks
        (id, project, session_id, task_name, status, priority,
         created_at, updated_at, due_at, dependencies, execution_log, swap_context, oom_adj)
        VALUES (?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?, '[]', NULL, -800)
    """, (task_id, project, session_id, task_name, priority,
          now, now, due_at, json.dumps(dependencies or [])))
    conn.commit()
    return task_id


def sched_update_task(conn: sqlite3.Connection, task_id: str, **kwargs) -> bool:
    """
    更新任务属性。OS 类比：sched_setattr() — 修改任务调度参数。
    支持字段：status, priority, due_at, execution_log, swap_context, oom_adj, task_name。
    """
    allowed = {"status", "priority", "due_at", "execution_log",
               "swap_context", "oom_adj", "task_name"}
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return False
    updates["updated_at"] = datetime.now(timezone.utc).isoformat()
    set_clause = ", ".join(f"{k}=?" for k in updates)
    values = list(updates.values()) + [task_id]
    conn.execute(f"UPDATE scheduler_tasks SET {set_clause} WHERE id=?", values)
    conn.commit()
    return conn.total_changes > 0


def sched_get_tasks(conn: sqlite3.Connection, project: str,
                    status: str = None, limit: int = 50) -> list:
    """
    查询任务列表。OS 类比：/proc/sched_debug — 读取调度队列状态。
    """
    if status:
        rows = conn.execute("""
            SELECT id, project, session_id, task_name, status, priority,
                   created_at, updated_at, due_at, dependencies,
                   execution_log, swap_context, oom_adj
            FROM scheduler_tasks
            WHERE project=? AND status=?
            ORDER BY priority DESC, created_at ASC
            LIMIT ?
        """, (project, status, limit)).fetchall()
    else:
        rows = conn.execute("""
            SELECT id, project, session_id, task_name, status, priority,
                   created_at, updated_at, due_at, dependencies,
                   execution_log, swap_context, oom_adj
            FROM scheduler_tasks
            WHERE project=?
            ORDER BY priority DESC, created_at ASC
            LIMIT ?
        """, (project, limit)).fetchall()
    cols = ["id", "project", "session_id", "task_name", "status", "priority",
            "created_at", "updated_at", "due_at", "dependencies",
            "execution_log", "swap_context", "oom_adj"]
    result = []
    for row in rows:
        d = dict(zip(cols, row))
        for jf in ("dependencies", "execution_log"):
            if isinstance(d[jf], str):
                try:
                    d[jf] = json.loads(d[jf])
                except Exception:
                    pass
        if isinstance(d["swap_context"], str):
            try:
                d["swap_context"] = json.loads(d["swap_context"])
            except Exception:
                pass
        result.append(d)
    return result


def sched_get_task(conn: sqlite3.Connection, task_id: str) -> Optional[dict]:
    """获取单个任务详情。"""
    row = conn.execute("""
        SELECT id, project, session_id, task_name, status, priority,
               created_at, updated_at, due_at, dependencies,
               execution_log, swap_context, oom_adj
        FROM scheduler_tasks WHERE id=?
    """, (task_id,)).fetchone()
    if not row:
        return None
    cols = ["id", "project", "session_id", "task_name", "status", "priority",
            "created_at", "updated_at", "due_at", "dependencies",
            "execution_log", "swap_context", "oom_adj"]
    d = dict(zip(cols, row))
    for jf in ("dependencies", "execution_log"):
        if isinstance(d[jf], str):
            try:
                d[jf] = json.loads(d[jf])
            except Exception:
                pass
    if isinstance(d["swap_context"], str):
        try:
            d["swap_context"] = json.loads(d["swap_context"])
        except Exception:
            pass
    return d


def sched_delete_task(conn: sqlite3.Connection, task_id: str) -> bool:
    """删除任务及其决策关联。OS 类比：do_exit() — 清理任务资源。"""
    conn.execute("DELETE FROM scheduler_task_decisions WHERE task_id=?", (task_id,))
    conn.execute("DELETE FROM scheduler_tasks WHERE id=?", (task_id,))
    conn.commit()
    return True


def sched_append_log(conn: sqlite3.Connection, task_id: str,
                     action: str, result: str = None) -> None:
    """追加执行日志条目。OS 类比：ftrace — 记录任务执行轨迹。"""
    now = datetime.now(timezone.utc).isoformat()
    row = conn.execute(
        "SELECT execution_log FROM scheduler_tasks WHERE id=?", (task_id,)
    ).fetchone()
    if not row:
        return
    try:
        log = json.loads(row[0]) if row[0] else []
    except Exception:
        log = []
    log.append({"ts": now, "action": action, "result": result})
    # 保留最近 50 条
    if len(log) > 50:
        log = log[-50:]
    conn.execute(
        "UPDATE scheduler_tasks SET execution_log=?, updated_at=? WHERE id=?",
        (json.dumps(log), now, task_id)
    )
    conn.commit()


def sched_link_decision(conn: sqlite3.Connection, task_id: str,
                        decision_id: str, decision_type: str = "enabler") -> None:
    """关联任务和决策 chunk。OS 类比：task_struct->mm_struct 映射。"""
    conn.execute("""
        INSERT OR IGNORE INTO scheduler_task_decisions
        (decision_id, task_id, decision_type) VALUES (?, ?, ?)
    """, (decision_id, task_id, decision_type))
    conn.commit()


def sched_dump_tasks(conn: sqlite3.Connection, project: str,
                     session_id: str = None) -> dict:
    """
    导出任务快照用于 swap_out。OS 类比：CRIU dump — 序列化进程状态。
    返回 {active_tasks, pending_tasks, completed_count, decisions}。
    """
    # 活跃任务（running + pending + blocked）
    active = sched_get_tasks(conn, project, status=None, limit=100)
    active = [t for t in active if t["status"] in ("running", "pending", "blocked")]

    # 统计完成数
    completed_count = conn.execute(
        "SELECT COUNT(*) FROM scheduler_tasks WHERE project=? AND status='completed'",
        (project,)
    ).fetchone()[0]

    # 关联的 decisions
    decision_ids = []
    for task in active:
        rows = conn.execute(
            "SELECT decision_id, decision_type FROM scheduler_task_decisions WHERE task_id=?",
            (task["id"],)
        ).fetchall()
        for did, dtype in rows:
            decision_ids.append({"decision_id": did, "type": dtype, "task_id": task["id"]})

    return {
        "active_tasks": [{
            "id": t["id"],
            "name": t["task_name"],
            "status": t["status"],
            "priority": t["priority"],
            "log_tail": (t.get("execution_log") or [])[-5:],
        } for t in active[:10]],  # Top-10 by priority
        "pending_count": sum(1 for t in active if t["status"] == "pending"),
        "running_count": sum(1 for t in active if t["status"] == "running"),
        "blocked_count": sum(1 for t in active if t["status"] == "blocked"),
        "completed_count": completed_count,
        "decisions": decision_ids[:20],
    }


def sched_restore_tasks(conn: sqlite3.Connection, dump: dict,
                        project: str, session_id: str) -> str:
    """
    从 swap dump 恢复任务上下文，返回格式化字符串用于注入 additionalContext。
    OS 类比：CRIU restore — 从快照重建进程状态。
    """
    if not dump or not dump.get("active_tasks"):
        return ""

    lines = []
    lines.append(f"\u3010\u4efb\u52a1\u8c03\u5ea6\u6062\u590d \u00b7 {dump.get('running_count', 0)} \u8fd0\u884c\u4e2d / "
                 f"{dump.get('pending_count', 0)} \u5f85\u529e / "
                 f"{dump.get('completed_count', 0)} \u5df2\u5b8c\u6210\u3011")

    for task in dump["active_tasks"][:5]:
        status_icon = {"running": "\u25b6", "pending": "\u23f3", "blocked": "\U0001f512"}.get(
            task["status"], "?")
        lines.append(f"  {status_icon} [{task['status']}] {task['name']} (P{task['priority']})")
        # 注入最近执行日志
        if task.get("log_tail"):
            for entry in task["log_tail"][-3:]:
                lines.append(f"    - {entry.get('action', '')}: {entry.get('result', '')}"[:80])

    return "\n".join(lines)
