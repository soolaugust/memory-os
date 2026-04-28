"""
test_episodes.py — iter364 情节时间线测试

覆盖：
  E1: write_episode + get_recent_episodes (project)
  E2: get_recent_episodes by workspace_id
  E3: mark_episode_injected
  E4: build_episode_summary — 各字段组合
  E5: format_episodes_for_injection — 字符截断
  E6: 多 episode 按 ended_at DESC 排序
  E7: empty episodes → format 返回空字符串
"""
import json
import os
import sys
import time
from datetime import datetime, timezone
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
    from store_episodes import ensure_episodes_schema
    c = open_db(tmpdb)
    ensure_episodes_schema(c)
    yield c
    c.close()


# ── E1: write + get by project ────────────────────────────────────────────────

def test_e1_write_and_get_by_project(conn):
    from store_episodes import write_episode, get_recent_episodes
    write_episode(conn, "sess1", "proj_a", "做了功能 X", chunks_created=3)
    write_episode(conn, "sess2", "proj_a", "修复了 bug Y", chunks_created=1)
    eps = get_recent_episodes(conn, "proj_a", limit=10)
    assert len(eps) == 2
    assert eps[0]["session_id"] == "sess2"  # 最近的在前


def test_e1_different_projects_isolated(conn):
    from store_episodes import write_episode, get_recent_episodes
    write_episode(conn, "s1", "proj_a", "A 的工作")
    write_episode(conn, "s2", "proj_b", "B 的工作")
    eps_a = get_recent_episodes(conn, "proj_a")
    assert len(eps_a) == 1
    assert eps_a[0]["session_id"] == "s1"


# ── E2: get by workspace_id ───────────────────────────────────────────────────

def test_e2_get_by_workspace_id(conn):
    from store_episodes import write_episode, get_recent_episodes
    write_episode(conn, "s1", "proj_a", "workspace A 工作",
                  workspace_id="ws_aaa")
    write_episode(conn, "s2", "proj_a", "workspace B 工作",
                  workspace_id="ws_bbb")
    eps = get_recent_episodes(conn, "proj_a", workspace_id="ws_aaa")
    assert len(eps) == 1
    assert eps[0]["session_id"] == "s1"


# ── E3: mark_episode_injected ─────────────────────────────────────────────────

def test_e3_mark_injected(conn):
    from store_episodes import write_episode, mark_episode_injected
    write_episode(conn, "s1", "proj", "工作")
    mark_episode_injected(conn, "s1")
    mark_episode_injected(conn, "s1")
    row = conn.execute(
        "SELECT injected_count FROM session_episodes WHERE session_id='s1'"
    ).fetchone()
    assert row[0] == 2


# ── E4: build_episode_summary ─────────────────────────────────────────────────

def test_e4_build_summary_with_files(tmp_path):
    from store_episodes import build_episode_summary
    msg = "完成了功能 X 的实现，修复了相关的单元测试。"
    s = build_episode_summary(msg, chunks_created=5,
                               files_modified=["hooks/loader.py", "tests/test_loader.py"],
                               tools_used={"Edit": 4, "Bash": 2})
    assert "完成了功能 X" in s or "loader.py" in s
    assert len(s) <= 400


def test_e4_build_summary_empty_msg():
    from store_episodes import build_episode_summary
    s = build_episode_summary("", chunks_created=0, files_modified=[], tools_used={})
    assert len(s) > 0  # 不为空


# ── E5: format_episodes_for_injection ────────────────────────────────────────

def test_e5_format_episodes(conn):
    from store_episodes import write_episode, get_recent_episodes, format_episodes_for_injection
    write_episode(conn, "s1", "proj", "实现了登录功能", chunks_created=2,
                  files_modified=["auth.py"])
    eps = get_recent_episodes(conn, "proj")
    text = format_episodes_for_injection(eps, max_chars=300)
    assert "【历史 Session 轨迹】" in text
    assert "登录" in text


def test_e5_format_respects_max_chars(conn):
    from store_episodes import write_episode, get_recent_episodes, format_episodes_for_injection
    for i in range(10):
        write_episode(conn, f"s{i}", "proj",
                      "x" * 150, chunks_created=i)
    eps = get_recent_episodes(conn, "proj", limit=10)
    text = format_episodes_for_injection(eps, max_chars=100)
    assert len(text) <= 150  # 允许一点超出（header 本身）


# ── E6: ordering ──────────────────────────────────────────────────────────────

def test_e6_ordering(conn):
    from store_episodes import write_episode, get_recent_episodes
    import time
    write_episode(conn, "old", "proj", "旧 session", ended_at="2026-01-01T00:00:00+00:00")
    time.sleep(0.01)
    write_episode(conn, "new", "proj", "新 session", ended_at="2026-04-28T00:00:00+00:00")
    eps = get_recent_episodes(conn, "proj")
    assert eps[0]["session_id"] == "new"


# ── E7: empty → empty string ──────────────────────────────────────────────────

def test_e7_empty_episodes(conn):
    from store_episodes import format_episodes_for_injection
    assert format_episodes_for_injection([]) == ""
