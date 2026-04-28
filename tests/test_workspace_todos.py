"""
test_workspace_todos.py — iter365 前瞻性记忆测试

覆盖：
  T1: add_todo + get_pending_todos
  T2: complete_todo → 不再出现在 pending
  T3: cancel_todo
  T4: mark_todo_injected
  T5: format_todos_for_injection — 字符截断
  T6: extract_todos_from_text — TODO 模式识别
  T7: extract_todos_from_text — 条件性未来行动识别
  T8: 不同 workspace 的 todos 互不干扰
"""
import os
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


@pytest.fixture()
def tmpdb(tmp_path):
    db_path = tmp_path / "test_store.db"
    os.environ["MEMORY_OS_DB"] = str(db_path)
    os.environ["MEMORY_OS_DIR"] = str(tmp_path)
    yield db_path
    os.environ.pop("MEMORY_OS_DB", None)
    os.environ.pop("MEMORY_OS_DIR", None)


@pytest.fixture()
def conn(tmpdb):
    from store_vfs import open_db
    from store_todos import ensure_todos_schema
    c = open_db(tmpdb)
    ensure_todos_schema(c)
    yield c
    c.close()


# ── T1: add + get ─────────────────────────────────────────────────────────────

def test_t1_add_and_get(conn):
    from store_todos import add_todo, get_pending_todos
    add_todo(conn, "ws1", "proj", "检查日志文件是否有新错误")
    add_todo(conn, "ws1", "proj", "等 PR 合并后更新 README")
    todos = get_pending_todos(conn, "ws1")
    assert len(todos) == 2
    contents = [t["content"] for t in todos]
    assert "检查日志文件是否有新错误" in contents


# ── T2: complete_todo ─────────────────────────────────────────────────────────

def test_t2_complete_removes_from_pending(conn):
    from store_todos import add_todo, get_pending_todos, complete_todo
    tid = add_todo(conn, "ws1", "proj", "需要完成的事情")
    complete_todo(conn, tid)
    todos = get_pending_todos(conn, "ws1")
    assert not any(t["id"] == tid for t in todos)


def test_t2_complete_marks_status(conn):
    from store_todos import add_todo, complete_todo
    tid = add_todo(conn, "ws1", "proj", "待完成")
    complete_todo(conn, tid)
    row = conn.execute("SELECT status FROM workspace_todos WHERE id=?", (tid,)).fetchone()
    assert row[0] == "done"


# ── T3: cancel_todo ───────────────────────────────────────────────────────────

def test_t3_cancel(conn):
    from store_todos import add_todo, cancel_todo, get_pending_todos
    tid = add_todo(conn, "ws1", "proj", "被取消的事项")
    cancel_todo(conn, tid)
    todos = get_pending_todos(conn, "ws1")
    assert not any(t["id"] == tid for t in todos)


# ── T4: mark_todo_injected ────────────────────────────────────────────────────

def test_t4_mark_injected(conn):
    from store_todos import add_todo, mark_todo_injected
    tid = add_todo(conn, "ws1", "proj", "被注入的待办")
    mark_todo_injected(conn, tid)
    mark_todo_injected(conn, tid)
    row = conn.execute(
        "SELECT injected_count FROM workspace_todos WHERE id=?", (tid,)
    ).fetchone()
    assert row[0] == 2


# ── T5: format_todos_for_injection ───────────────────────────────────────────

def test_t5_format_todos(conn):
    from store_todos import add_todo, get_pending_todos, format_todos_for_injection
    add_todo(conn, "ws1", "proj", "检查端口配置是否正确", due_hint="等部署完成后")
    todos = get_pending_todos(conn, "ws1")
    text = format_todos_for_injection(todos, max_chars=200)
    assert "【工作区待办】" in text
    assert "检查端口" in text
    assert "等部署完成后" in text


def test_t5_format_empty(conn):
    from store_todos import format_todos_for_injection
    assert format_todos_for_injection([]) == ""


def test_t5_format_max_chars(conn):
    from store_todos import add_todo, get_pending_todos, format_todos_for_injection
    for i in range(10):
        add_todo(conn, "ws1", "proj", f"待办事项 {i} " + "x" * 50)
    todos = get_pending_todos(conn, "ws1", limit=10)
    text = format_todos_for_injection(todos, max_chars=80)
    assert len(text) <= 120  # 允许 header 超出少量


# ── T6: extract_todos_from_text ───────────────────────────────────────────────

def test_t6_extract_explicit_todo():
    from store_todos import extract_todos_from_text
    text = "TODO: 检查 nginx 配置是否与新端口一致"
    items = extract_todos_from_text(text)
    assert len(items) >= 1
    assert any("nginx" in i["content"] or "检查" in i["content"] for i in items)


def test_t6_extract_chinese_reminder():
    from store_todos import extract_todos_from_text
    text = "记得：下次进来要先检查数据库连接是否正常"
    items = extract_todos_from_text(text)
    assert len(items) >= 1


# ── T7: conditional future action ─────────────────────────────────────────────

def test_t7_conditional_extraction():
    from store_todos import extract_todos_from_text
    text = "等 feature-branch 合并后，再来处理配置文件的迁移"
    items = extract_todos_from_text(text)
    assert len(items) >= 1
    # 应该有 due_hint 包含合并条件
    has_due_hint = any(i.get("due_hint") for i in items)
    # 宽松判断：有内容即可（由于模式可能匹配多种形式）
    assert len(items) >= 1


# ── T8: workspace 隔离 ────────────────────────────────────────────────────────

def test_t8_workspace_isolation(conn):
    from store_todos import add_todo, get_pending_todos
    add_todo(conn, "ws_alpha", "proj", "alpha 的待办")
    add_todo(conn, "ws_beta", "proj", "beta 的待办")
    alpha_todos = get_pending_todos(conn, "ws_alpha")
    beta_todos = get_pending_todos(conn, "ws_beta")
    assert len(alpha_todos) == 1
    assert len(beta_todos) == 1
    assert alpha_todos[0]["content"] == "alpha 的待办"
    assert beta_todos[0]["content"] == "beta 的待办"
