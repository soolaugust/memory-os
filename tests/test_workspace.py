"""
test_workspace.py — iter363 Workspace-aware memory tests

覆盖：
  W1: resolve_workspace — 首次创建 + 重复进入（entry_count 递增）
  W2: activate_workspace — 无数据时返回空
  W3: link_chunk_to_workspace — 关联 + 查询
  W4: upsert_workspace_file — hash 变更检测（相同 hash → skip）
  W5: scan_and_store — 增量扫描（已有文件不重复处理）
  W6: workspace_scanner — docker-compose.yml 端口提取
  W7: workspace_scanner — .env 文件端口提取
  W8: workspace_scanner — package.json 脚本端口提取
  W9: get_workspace_knowledge — chunk_type 过滤
  W10: unlink_chunk_from_workspace — 非 pinned 可删除，pinned 不删
  W11: list_workspaces — 多工作区排序
  W12: get_workspace_by_path — 路径查找
  W13: loader.py workspace block — 有 file_facts 时注入到 context
"""
import json
import os
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

# ── path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

# 使用 tmpfs 隔离（MEMORY_OS_DIR 环境变量由下面 fixture 注入）


@pytest.fixture()
def tmpdb(tmp_path):
    """提供隔离的 in-memory DB（via tmp file）。"""
    db_path = tmp_path / "test_store.db"
    os.environ["MEMORY_OS_DB"] = str(db_path)
    os.environ["MEMORY_OS_DIR"] = str(tmp_path)
    yield db_path
    os.environ.pop("MEMORY_OS_DB", None)
    os.environ.pop("MEMORY_OS_DIR", None)


@pytest.fixture()
def conn(tmpdb):
    from store_vfs import open_db
    from store_workspace import ensure_workspace_schema
    c = open_db(tmpdb)
    ensure_workspace_schema(c)
    # 创建 memory_chunks 表（workspace_knowledge 需要 FK）
    c.execute("""
        CREATE TABLE IF NOT EXISTS memory_chunks (
            id TEXT PRIMARY KEY, created_at TEXT, updated_at TEXT,
            project TEXT, source_session TEXT, chunk_type TEXT,
            content TEXT, summary TEXT, tags TEXT,
            importance REAL DEFAULT 0.5, retrievability REAL DEFAULT 1.0,
            last_accessed TEXT, feishu_url TEXT,
            access_count INTEGER DEFAULT 0, oom_adj INTEGER DEFAULT 0,
            lru_gen INTEGER DEFAULT 0
        )
    """)
    c.commit()
    yield c
    c.close()


def _insert_chunk(conn, chunk_id, chunk_type="decision", summary="test summary", importance=0.7):
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT OR IGNORE INTO memory_chunks
        (id, created_at, updated_at, project, source_session, chunk_type,
         content, summary, importance, last_accessed)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """, (chunk_id, now, now, "test_project", "s1", chunk_type,
          f"content of {chunk_id}", summary, importance, now))
    conn.commit()


# ── W1: resolve_workspace ─────────────────────────────────────────────────────

def test_w1_resolve_workspace_creates(conn):
    from store_workspace import resolve_workspace, get_workspace_by_path
    ws_id = resolve_workspace(conn, "/projects/myapp")
    assert ws_id and len(ws_id) == 16
    ws = get_workspace_by_path(conn, "/projects/myapp")
    assert ws is not None
    assert ws["path"] == "/projects/myapp"
    assert ws["entry_count"] == 1


def test_w1_resolve_workspace_increments_entry_count(conn):
    from store_workspace import resolve_workspace, get_workspace_by_path
    resolve_workspace(conn, "/projects/myapp")
    resolve_workspace(conn, "/projects/myapp")
    resolve_workspace(conn, "/projects/myapp")
    ws = get_workspace_by_path(conn, "/projects/myapp")
    assert ws["entry_count"] == 3


def test_w1_different_paths_different_ids(conn):
    from store_workspace import resolve_workspace
    id1 = resolve_workspace(conn, "/projects/app1")
    id2 = resolve_workspace(conn, "/projects/app2")
    assert id1 != id2


# ── W2: activate_workspace ────────────────────────────────────────────────────

def test_w2_activate_empty_workspace(conn):
    from store_workspace import resolve_workspace, activate_workspace
    ws_id = resolve_workspace(conn, "/projects/empty")
    result = activate_workspace(conn, ws_id)
    assert result["workspace_id"] == ws_id
    assert result["kb_chunks"] == []
    assert result["file_facts"] == []


def test_w2_activate_nonexistent(conn):
    from store_workspace import activate_workspace
    result = activate_workspace(conn, "nonexistent000000")
    assert result["kb_chunks"] == []
    assert result["file_facts"] == []


# ── W3: link_chunk_to_workspace ───────────────────────────────────────────────

def test_w3_link_and_query(conn):
    from store_workspace import resolve_workspace, link_chunk_to_workspace, get_workspace_knowledge
    _insert_chunk(conn, "c1", "decision", "port decision")
    ws_id = resolve_workspace(conn, "/projects/app")
    link_chunk_to_workspace(conn, ws_id, "c1", source="conversation")
    chunks = get_workspace_knowledge(conn, ws_id)
    assert len(chunks) == 1
    assert chunks[0]["id"] == "c1"


def test_w3_link_idempotent(conn):
    from store_workspace import resolve_workspace, link_chunk_to_workspace, get_workspace_knowledge
    _insert_chunk(conn, "c2")
    ws_id = resolve_workspace(conn, "/projects/app")
    link_chunk_to_workspace(conn, ws_id, "c2")
    link_chunk_to_workspace(conn, ws_id, "c2")
    chunks = get_workspace_knowledge(conn, ws_id)
    assert len(chunks) == 1  # 幂等


# ── W4: upsert_workspace_file ─────────────────────────────────────────────────

def test_w4_upsert_new_file(conn, tmp_path):
    from store_workspace import resolve_workspace, upsert_workspace_file, get_workspace_files
    f = tmp_path / "docker-compose.yml"
    f.write_text("version: '3'\n")
    ws_id = resolve_workspace(conn, str(tmp_path))
    changed = upsert_workspace_file(conn, ws_id, str(f), [{"type": "port", "port": 8080}])
    assert changed is True
    files = get_workspace_files(conn, ws_id)
    assert len(files) == 1
    assert files[0]["facts"][0]["port"] == 8080


def test_w4_same_hash_skip(conn, tmp_path):
    from store_workspace import resolve_workspace, upsert_workspace_file
    f = tmp_path / "docker-compose.yml"
    f.write_text("version: '3'\n")
    ws_id = resolve_workspace(conn, str(tmp_path))
    upsert_workspace_file(conn, ws_id, str(f), [{"type": "port", "port": 8080}])
    changed = upsert_workspace_file(conn, ws_id, str(f), [{"type": "port", "port": 8080}])
    assert changed is False  # 文件未变更，跳过


def test_w4_content_change_triggers_update(conn, tmp_path):
    from store_workspace import resolve_workspace, upsert_workspace_file
    f = tmp_path / ".env"
    f.write_text("PORT=3000\n")
    ws_id = resolve_workspace(conn, str(tmp_path))
    upsert_workspace_file(conn, ws_id, str(f), [{"type": "port", "port": 3000}])
    f.write_text("PORT=4000\n")  # 内容变更
    changed = upsert_workspace_file(conn, ws_id, str(f), [{"type": "port", "port": 4000}])
    assert changed is True


# ── W5: scan_and_store ────────────────────────────────────────────────────────

def test_w5_scan_and_store_docker_compose(conn, tmp_path):
    from store_workspace import resolve_workspace
    from workspace_scanner import scan_and_store
    dc = tmp_path / "docker-compose.yml"
    dc.write_text("""version: '3'
services:
  backend:
    ports:
      - "8000:8000"
  frontend:
    ports:
      - "3000:80"
""")
    ws_id = resolve_workspace(conn, str(tmp_path))
    result = scan_and_store(conn, ws_id, str(tmp_path))
    assert result["scanned"] >= 1
    assert result["updated"] >= 1
    assert result["facts_total"] >= 2


def test_w5_scan_no_update_on_repeat(conn, tmp_path):
    from store_workspace import resolve_workspace
    from workspace_scanner import scan_and_store
    dc = tmp_path / "docker-compose.yml"
    dc.write_text("version: '3'\nservices:\n  api:\n    ports:\n      - '5000:5000'\n")
    ws_id = resolve_workspace(conn, str(tmp_path))
    scan_and_store(conn, ws_id, str(tmp_path))
    result2 = scan_and_store(conn, ws_id, str(tmp_path))
    assert result2["updated"] == 0  # 未变更，不重复处理


# ── W6: docker-compose 端口提取 ───────────────────────────────────────────────

def test_w6_docker_compose_port_extraction():
    from workspace_scanner import extract_file_facts
    import tempfile, os
    content = """version: '3'
services:
  backend:
    ports:
      - "8000:8000"
      - "9000:9000"
  redis:
    ports:
      - "6379:6379"
"""
    with tempfile.NamedTemporaryFile(suffix="docker-compose.yml",
                                    mode="w", delete=False) as f:
        f.write(content)
        fpath = f.name
    try:
        facts = extract_file_facts(fpath)
        port_facts = [f for f in facts if f.get("type") == "port"]
        ports = {f["host_port"] for f in port_facts}
        assert 8000 in ports
        assert 9000 in ports
        assert 6379 in ports
    finally:
        os.unlink(fpath)


# ── W7: .env 端口提取 ─────────────────────────────────────────────────────────

def test_w7_env_file_port_extraction(tmp_path):
    from workspace_scanner import extract_file_facts
    env_file = tmp_path / ".env"
    env_file.write_text("PORT=3000\nDATABASE_URL=postgres://localhost:5432/mydb\nSECRET=abc\n")
    facts = extract_file_facts(str(env_file))
    port_facts = [f for f in facts if f.get("port")]
    ports = {f["port"] for f in port_facts}
    assert 3000 in ports
    assert 5432 in ports


# ── W8: package.json 脚本端口提取 ────────────────────────────────────────────

def test_w8_package_json_port_extraction(tmp_path):
    from workspace_scanner import extract_file_facts
    pkg = tmp_path / "package.json"
    pkg.write_text(json.dumps({
        "name": "my-frontend",
        "scripts": {
            "dev": "vite --port 5173",
            "start": "node server.js",
        }
    }))
    facts = extract_file_facts(str(pkg))
    port_facts = [f for f in facts if f.get("type") == "port"]
    ports = {f.get("port") for f in port_facts}
    assert 5173 in ports
    name_facts = [f for f in facts if f.get("type") == "project_name"]
    assert name_facts[0]["name"] == "my-frontend"


# ── W9: get_workspace_knowledge chunk_type 过滤 ───────────────────────────────

def test_w9_filter_by_chunk_type(conn):
    from store_workspace import resolve_workspace, link_chunk_to_workspace, get_workspace_knowledge
    _insert_chunk(conn, "d1", "decision", "decision summary")
    _insert_chunk(conn, "c1", "design_constraint", "constraint summary")
    ws_id = resolve_workspace(conn, "/projects/filtered")
    link_chunk_to_workspace(conn, ws_id, "d1")
    link_chunk_to_workspace(conn, ws_id, "c1")

    decisions = get_workspace_knowledge(conn, ws_id, chunk_type="decision")
    assert len(decisions) == 1
    assert decisions[0]["id"] == "d1"

    constraints = get_workspace_knowledge(conn, ws_id, chunk_type="design_constraint")
    assert len(constraints) == 1
    assert constraints[0]["id"] == "c1"

    all_chunks = get_workspace_knowledge(conn, ws_id)
    assert len(all_chunks) == 2


# ── W10: unlink_chunk_from_workspace ─────────────────────────────────────────

def test_w10_unlink_non_pinned(conn):
    from store_workspace import (resolve_workspace, link_chunk_to_workspace,
                                  unlink_chunk_from_workspace, get_workspace_knowledge)
    _insert_chunk(conn, "u1")
    ws_id = resolve_workspace(conn, "/projects/unlink")
    link_chunk_to_workspace(conn, ws_id, "u1", pinned=False)
    unlink_chunk_from_workspace(conn, ws_id, "u1")
    assert get_workspace_knowledge(conn, ws_id) == []


def test_w10_pinned_cannot_be_unlinked(conn):
    from store_workspace import (resolve_workspace, link_chunk_to_workspace,
                                  unlink_chunk_from_workspace, get_workspace_knowledge)
    _insert_chunk(conn, "p1")
    ws_id = resolve_workspace(conn, "/projects/pinned")
    link_chunk_to_workspace(conn, ws_id, "p1", pinned=True)
    unlink_chunk_from_workspace(conn, ws_id, "p1")
    chunks = get_workspace_knowledge(conn, ws_id)
    assert len(chunks) == 1  # pinned — 不被删除


# ── W11: list_workspaces ──────────────────────────────────────────────────────

def test_w11_list_workspaces(conn):
    from store_workspace import resolve_workspace, list_workspaces
    import time
    resolve_workspace(conn, "/projects/alpha")
    time.sleep(0.01)
    resolve_workspace(conn, "/projects/beta")
    ws_list = list_workspaces(conn)
    assert len(ws_list) >= 2
    # 最近进入的在前
    paths = [w["path"] for w in ws_list]
    assert paths.index("/projects/beta") < paths.index("/projects/alpha")


# ── W12: get_workspace_by_path ───────────────────────────────────────────────

def test_w12_get_workspace_by_path(conn):
    from store_workspace import resolve_workspace, get_workspace_by_path
    resolve_workspace(conn, "/projects/found")
    ws = get_workspace_by_path(conn, "/projects/found")
    assert ws is not None
    assert ws["path"] == "/projects/found"


def test_w12_get_workspace_missing(conn):
    from store_workspace import get_workspace_by_path
    ws = get_workspace_by_path(conn, "/projects/notexist")
    assert ws is None


# ── W13: activate_workspace returns file_facts ────────────────────────────────

def test_w13_activate_returns_file_facts(conn, tmp_path):
    from store_workspace import resolve_workspace, upsert_workspace_file, activate_workspace
    ws_id = resolve_workspace(conn, str(tmp_path))
    f = tmp_path / ".env"
    f.write_text("PORT=8080\n")
    upsert_workspace_file(conn, ws_id, str(f), [
        {"type": "port", "port": 8080, "description": "PORT=8080 (port 8080)"}
    ])
    data = activate_workspace(conn, ws_id)
    assert len(data["file_facts"]) == 1
    assert data["file_facts"][0]["facts"][0]["port"] == 8080
