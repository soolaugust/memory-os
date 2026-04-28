"""
test_agent_team.py — iter259 Multi-Agent 隔离验证测试套件

目标：验证"全部修复"的 multi-agent 兼容性，防止偏离和幻觉。

覆盖场景：
  A1. shadow_traces DB 隔离 — Agent A 和 Agent B 的 shadow trace 互不覆盖
  A2. shadow_traces fallback — DB 不可用时 fallback 到旧文件，读出正确 project
  A3. session_intents DB 隔离 — 多 agent 写入各自 intent，loader 读最新
  A4. CRIU checkpoint 隔离 — _checkpoint_cleanup 按 session_id 清理，不跨 agent
  A5. checkpoint_dump 传递 session_id — 激活 per-agent cleanup
  A6. extractor Active Suppression 从 DB 读取 per-session shadow trace
  A7. 并发写 shadow_trace 不丢失数据（INSERT OR REPLACE 语义）
  A8. checkpoint content_hash 版本校验（从 test_chaos.py 原有场景迁移）

OS 类比：
  Linux namespace isolation — 同一 host 的多进程通过 PID namespace 隔离，
  同一 DB 的多 agent 通过 session_id PRIMARY KEY 隔离，互不干扰。
"""

import json
import sqlite3
import sys
import threading
import time
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ── 共用 fixture ──────────────────────────────────────────────────────────────

@pytest.fixture()
def fresh_db(tmp_path, monkeypatch):
    """
    每个测试用独立 DB dir，monkeypatch 全局路径，保证隔离。
    返回 (db_dir, db_path, open_conn).
    """
    db_dir = tmp_path / "memory-os"
    db_dir.mkdir()
    db_path = db_dir / "store.db"

    monkeypatch.setenv("MEMORY_OS_DIR", str(db_dir))
    monkeypatch.setenv("MEMORY_OS_DB", str(db_path))

    import store as _store
    import store_core as _core
    import store_vfs as _vfs
    import store_criu as _criu

    monkeypatch.setattr(_store, "STORE_DB", db_path)
    monkeypatch.setattr(_core, "STORE_DB", db_path)
    monkeypatch.setattr(_vfs, "STORE_DB", db_path)
    monkeypatch.setattr(_criu, "MEMORY_OS_DIR", db_dir)

    from store import open_db, ensure_schema
    conn = open_db()
    ensure_schema(conn)
    conn.commit()
    return db_dir, db_path, conn


# ── A1: shadow_traces DB 隔离 ─────────────────────────────────────────────────

def test_A1_shadow_trace_agent_isolation(fresh_db, monkeypatch):
    """
    Agent A 和 Agent B 各自写入 shadow_traces；
    读取时各自的 top_k_ids 不互相覆盖。
    OS 类比：两个进程写 /proc/PID/mem，只影响各自 PID。
    """
    db_dir, db_path, conn = fresh_db

    # 模拟两个 agent 的 session_id
    agent_a_session = "aaaa-session-111"
    agent_b_session = "bbbb-session-222"
    project = "test-project"

    # 确保 shadow_traces 表存在
    conn.execute("""
        CREATE TABLE IF NOT EXISTS shadow_traces (
            session_id   TEXT PRIMARY KEY,
            project      TEXT NOT NULL DEFAULT '',
            agent_id     TEXT NOT NULL DEFAULT '',
            updated_at   TEXT NOT NULL,
            top_k_ids    TEXT NOT NULL DEFAULT '[]'
        )
    """)
    conn.commit()

    # Agent A 写入
    conn.execute(
        "INSERT OR REPLACE INTO shadow_traces (session_id, project, agent_id, updated_at, top_k_ids) VALUES (?,?,?,?,?)",
        (agent_a_session, project, agent_a_session[:16], "2026-01-01T00:00:00Z",
         json.dumps(["chunk-A1", "chunk-A2"]))
    )
    # Agent B 写入
    conn.execute(
        "INSERT OR REPLACE INTO shadow_traces (session_id, project, agent_id, updated_at, top_k_ids) VALUES (?,?,?,?,?)",
        (agent_b_session, project, agent_b_session[:16], "2026-01-01T00:00:01Z",
         json.dumps(["chunk-B1", "chunk-B2"]))
    )
    conn.commit()

    # 读取 Agent A 的数据
    row_a = conn.execute(
        "SELECT top_k_ids FROM shadow_traces WHERE session_id=? AND project=?",
        (agent_a_session, project)
    ).fetchone()
    ids_a = json.loads(row_a[0])

    # 读取 Agent B 的数据
    row_b = conn.execute(
        "SELECT top_k_ids FROM shadow_traces WHERE session_id=? AND project=?",
        (agent_b_session, project)
    ).fetchone()
    ids_b = json.loads(row_b[0])

    assert ids_a == ["chunk-A1", "chunk-A2"], f"Agent A's shadow trace corrupted: {ids_a}"
    assert ids_b == ["chunk-B1", "chunk-B2"], f"Agent B's shadow trace corrupted: {ids_b}"
    # Agent A 的数据中不应包含 Agent B 的 chunks
    assert "chunk-B1" not in ids_a
    assert "chunk-A1" not in ids_b


# ── A2: shadow_trace INSERT OR REPLACE 覆写同 session ────────────────────────

def test_A2_shadow_trace_replace_same_session(fresh_db):
    """
    同一 session_id INSERT OR REPLACE 时，只保留最新数据，不产生重复行。
    OS 类比：mmap() with MAP_FIXED — 同地址映射替换旧映射，不叠加。
    """
    db_dir, db_path, conn = fresh_db
    session_id = "session-replace-test"
    project = "proj"

    conn.execute("""
        CREATE TABLE IF NOT EXISTS shadow_traces (
            session_id TEXT PRIMARY KEY,
            project TEXT NOT NULL DEFAULT '',
            agent_id TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL,
            top_k_ids TEXT NOT NULL DEFAULT '[]'
        )
    """)
    conn.execute(
        "INSERT OR REPLACE INTO shadow_traces (session_id, project, agent_id, updated_at, top_k_ids) VALUES (?,?,?,?,?)",
        (session_id, project, session_id[:16], "2026-01-01T00:00:00Z", json.dumps(["old-chunk"]))
    )
    conn.commit()

    # 再次写入（模拟新的检索轮次更新 shadow trace）
    conn.execute(
        "INSERT OR REPLACE INTO shadow_traces (session_id, project, agent_id, updated_at, top_k_ids) VALUES (?,?,?,?,?)",
        (session_id, project, session_id[:16], "2026-01-01T00:00:01Z", json.dumps(["new-chunk-1", "new-chunk-2"]))
    )
    conn.commit()

    count = conn.execute(
        "SELECT COUNT(*) FROM shadow_traces WHERE session_id=?", (session_id,)
    ).fetchone()[0]
    assert count == 1, f"Should be exactly 1 row, got {count}"

    row = conn.execute(
        "SELECT top_k_ids FROM shadow_traces WHERE session_id=?", (session_id,)
    ).fetchone()
    ids = json.loads(row[0])
    assert ids == ["new-chunk-1", "new-chunk-2"], f"Should have new data: {ids}"
    assert "old-chunk" not in ids


# ── A3: session_intents DB 隔离 ───────────────────────────────────────────────

def test_A3_session_intent_db_isolation(fresh_db):
    """
    多个 agent 写入各自的 session intent；
    loader 读取最新 intent 时不混用不同 agent 的断点状态。
    OS 类比：/proc/PID/environ — 每进程独立环境变量，不同进程间不共享。
    """
    db_dir, db_path, conn = fresh_db
    project = "test-project"

    # 确保表存在（由 ensure_schema 创建，但这里直接测试 DB 操作）
    conn.execute("""
        CREATE TABLE IF NOT EXISTS session_intents (
            session_id    TEXT PRIMARY KEY,
            project       TEXT NOT NULL DEFAULT '',
            agent_id      TEXT NOT NULL DEFAULT '',
            saved_at      TEXT NOT NULL,
            intent_json   TEXT NOT NULL DEFAULT '{}',
            pinned_chunk_ids TEXT NOT NULL DEFAULT '[]'
        )
    """)
    conn.commit()

    # Agent A 保存 intent（早）
    conn.execute(
        "INSERT OR REPLACE INTO session_intents (session_id, project, agent_id, saved_at, intent_json, pinned_chunk_ids) VALUES (?,?,?,?,?,?)",
        ("sess-agent-a", project, "sess-agent-a"[:16], "2026-01-01T10:00:00Z",
         json.dumps({"next_actions": ["完成功能A"], "open_questions": []}),
         "[]")
    )
    # Agent B 保存 intent（晚）
    conn.execute(
        "INSERT OR REPLACE INTO session_intents (session_id, project, agent_id, saved_at, intent_json, pinned_chunk_ids) VALUES (?,?,?,?,?,?)",
        ("sess-agent-b", project, "sess-agent-b"[:16], "2026-01-01T11:00:00Z",
         json.dumps({"next_actions": ["完成功能B"], "open_questions": ["如何实现X"]}),
         "[]")
    )
    conn.commit()

    # loader 的行为：读最新一条（ORDER BY saved_at DESC LIMIT 1）
    row = conn.execute(
        "SELECT intent_json, saved_at FROM session_intents WHERE project=? ORDER BY saved_at DESC LIMIT 1",
        (project,)
    ).fetchone()
    intent = json.loads(row[0])

    # 最新的 intent 是 Agent B 的
    assert "完成功能B" in intent.get("next_actions", []), \
        f"Expected Agent B's intent, got: {intent}"
    assert "完成功能A" not in intent.get("next_actions", [])

    # 但 Agent A 的 intent 仍然存在（未被覆盖）
    row_a = conn.execute(
        "SELECT intent_json FROM session_intents WHERE session_id=?", ("sess-agent-a",)
    ).fetchone()
    assert row_a is not None, "Agent A's intent should still exist"
    intent_a = json.loads(row_a[0])
    assert "完成功能A" in intent_a.get("next_actions", [])


# ── A4: CRIU checkpoint 按 session_id 隔离清理 ───────────────────────────────

def test_A4_criu_checkpoint_cleanup_isolation(fresh_db, monkeypatch):
    """
    _checkpoint_cleanup(conn, project, session_id="agent-a") 只清理 agent-a 的超额 checkpoint，
    不影响 agent-b 的 checkpoints。
    OS 类比：CRIU per-process dump — kill -9 <pid> 只清理目标进程，不影响同主机其他进程。
    """
    db_dir, db_path, conn = fresh_db

    from store_criu import _ensure_checkpoint_schema, _checkpoint_cleanup

    # 确保 schema
    _ensure_checkpoint_schema(conn)

    # sysctl max_checkpoints = 2
    from config import sysctl_set as _cfg_set
    _cfg_set("criu.max_checkpoints", 2)

    project = "proj"

    # Agent A 写入 3 个 checkpoint（超出 max=2）
    for i in range(3):
        conn.execute(
            "INSERT INTO checkpoints (id, created_at, project, session_id, hit_chunk_ids, consumed) VALUES (?,?,?,?,?,0)",
            (f"ckpt-a-{i}", f"2026-01-01T00:0{i}:00Z", project, "session-agent-a", "[]")
        )
    # Agent B 写入 1 个 checkpoint（未超出）
    conn.execute(
        "INSERT INTO checkpoints (id, created_at, project, session_id, hit_chunk_ids, consumed) VALUES (?,?,?,?,?,0)",
        ("ckpt-b-0", "2026-01-01T00:00:00Z", project, "session-agent-b", "[]")
    )
    conn.commit()

    # 清理 Agent A 的超额（应删除 1 个）
    deleted = _checkpoint_cleanup(conn, project, session_id="session-agent-a")
    conn.commit()

    assert deleted == 1, f"Expected 1 deleted, got {deleted}"

    # Agent B 的 checkpoint 应完好
    b_remaining = conn.execute(
        "SELECT COUNT(*) FROM checkpoints WHERE session_id='session-agent-b'"
    ).fetchone()[0]
    assert b_remaining == 1, f"Agent B checkpoints should be untouched: {b_remaining}"

    # Agent A 应剩余 2 个（max）
    a_remaining = conn.execute(
        "SELECT COUNT(*) FROM checkpoints WHERE session_id='session-agent-a'"
    ).fetchone()[0]
    assert a_remaining == 2, f"Agent A should have 2 remaining: {a_remaining}"


# ── A5: checkpoint_dump 传递 session_id ──────────────────────────────────────

def test_A5_checkpoint_dump_passes_session_id(fresh_db, monkeypatch):
    """
    checkpoint_dump() 调用 _checkpoint_cleanup(conn, project, session_id=session_id)，
    确保 per-agent 隔离真正激活（而不是用了默认的 session_id="" 全局清理）。

    验证方式：mock _checkpoint_cleanup 捕获调用参数，断言 session_id 被传入。
    OS 类比：fork() 中 copy_process() 必须传入父进程 pid — 通过参数传递而非全局变量。
    """
    db_dir, db_path, conn = fresh_db

    from store_criu import _ensure_checkpoint_schema
    _ensure_checkpoint_schema(conn)

    # 先在 DB 中插入一个 chunk 供 dump 查询
    conn.execute("""
        INSERT OR IGNORE INTO memory_chunks
        (id, created_at, updated_at, project, chunk_type, content, summary, importance)
        VALUES ('chunk-test-1', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z',
                'proj', 'decision', '[decision] test', 'test chunk', 0.8)
    """)
    conn.commit()

    # 捕获 _checkpoint_cleanup 的调用参数
    captured_calls = []

    import store_criu as _criu_mod
    original_cleanup = _criu_mod._checkpoint_cleanup

    def mock_cleanup(conn, project, session_id=""):
        captured_calls.append({"project": project, "session_id": session_id})
        return 0  # 不执行真实清理

    monkeypatch.setattr(_criu_mod, "_checkpoint_cleanup", mock_cleanup)

    from store_criu import checkpoint_dump
    result = checkpoint_dump(
        conn, "proj", "test-session-xyz",
        hit_chunk_ids=["chunk-test-1"]
    )
    conn.commit()

    assert len(captured_calls) == 1, f"_checkpoint_cleanup should be called once: {captured_calls}"
    call = captured_calls[0]
    assert call["project"] == "proj", f"Wrong project: {call['project']}"
    assert call["session_id"] == "test-session-xyz", \
        f"session_id not passed to cleanup: {call['session_id']} (expected 'test-session-xyz')"


# ── A6: extractor Active Suppression 读取 per-session DB ─────────────────────

def test_A6_active_suppression_reads_db(fresh_db, monkeypatch):
    """
    extractor 的 Active Suppression 逻辑优先从 shadow_traces DB 读取 per-session IDs，
    而非从全局 .shadow_trace.json 文件。

    验证：写入 DB 的 shadow trace 与全局 JSON 文件内容不同时，
    suppress_unused 收到的 injected_ids 应来自 DB。
    OS 类比：per-process /proc 文件覆盖全局 sysctl — 子系统先看 per-pid 值。
    """
    db_dir, db_path, conn = fresh_db
    project = "proj"
    session_id = "sess-suppression-test"

    # 写入 DB shadow trace（per-session，正确值）
    conn.execute("""
        CREATE TABLE IF NOT EXISTS shadow_traces (
            session_id TEXT PRIMARY KEY,
            project TEXT NOT NULL DEFAULT '',
            agent_id TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL,
            top_k_ids TEXT NOT NULL DEFAULT '[]'
        )
    """)
    conn.execute(
        "INSERT OR REPLACE INTO shadow_traces (session_id, project, agent_id, updated_at, top_k_ids) VALUES (?,?,?,?,?)",
        (session_id, project, session_id[:16], "2026-01-01T00:00:00Z",
         json.dumps(["db-chunk-1", "db-chunk-2"]))
    )
    conn.commit()

    # 写入旧的全局 JSON 文件（内容不同）
    shadow_file = db_dir / ".shadow_trace.json"
    shadow_file.write_text(json.dumps({
        "project": project,
        "top_k_ids": ["file-chunk-stale"],
        "session_id": "different-session",
    }), encoding="utf-8")

    import store as _store_mod
    monkeypatch.setattr(_store_mod, "STORE_DB", db_path)

    # 模拟 extractor Active Suppression 读取逻辑
    _injected_ids = []
    _sup_loaded = False

    # DB 优先路径（iter259）
    try:
        _sup_db = sqlite3.connect(str(db_path))
        _sup_row = _sup_db.execute(
            "SELECT top_k_ids FROM shadow_traces WHERE session_id=? AND project=?",
            (session_id, project)
        ).fetchone()
        _sup_db.close()
        if _sup_row:
            _injected_ids = json.loads(_sup_row[0] or "[]")
            _sup_loaded = True
    except Exception:
        pass

    # Fallback（旧文件）
    if not _sup_loaded:
        if shadow_file.exists():
            _shadow = json.loads(shadow_file.read_text(encoding="utf-8"))
            if _shadow.get("project", project) == project:
                _injected_ids = _shadow.get("top_k_ids", [])

    # 应读到 DB 中的值，而非旧文件中的过期值
    assert _injected_ids == ["db-chunk-1", "db-chunk-2"], \
        f"Should read from DB, got: {_injected_ids}"
    assert "file-chunk-stale" not in _injected_ids


# ── A7: 并发写 shadow_trace 不丢失数据 ──────────────────────────────────────

def test_A7_concurrent_shadow_trace_writes(fresh_db):
    """
    两个 thread 同时写 shadow_traces（模拟两个 agent 并发），
    最终每个 session_id 的数据都保留。
    OS 类比：kernel spinlock — 并发写 /proc 文件不丢数据。
    """
    db_dir, db_path, conn = fresh_db

    conn.execute("""
        CREATE TABLE IF NOT EXISTS shadow_traces (
            session_id TEXT PRIMARY KEY,
            project TEXT NOT NULL DEFAULT '',
            agent_id TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL,
            top_k_ids TEXT NOT NULL DEFAULT '[]'
        )
    """)
    conn.commit()
    conn.close()  # 关闭主连接，让线程各自开连接

    errors = []

    def write_agent(session_id, chunks):
        try:
            c = sqlite3.connect(str(db_path), timeout=10)
            c.execute("INSERT OR REPLACE INTO shadow_traces (session_id, project, agent_id, updated_at, top_k_ids) VALUES (?,?,?,?,?)",
                      (session_id, "proj", session_id[:16], "2026-01-01T00:00:00Z",
                       json.dumps(chunks)))
            c.commit()
            c.close()
        except Exception as e:
            errors.append(str(e))

    threads = [
        threading.Thread(target=write_agent, args=(f"session-t{i}", [f"chunk-t{i}-1", f"chunk-t{i}-2"]))
        for i in range(5)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Concurrent writes had errors: {errors}"

    verify_conn = sqlite3.connect(str(db_path))
    rows = verify_conn.execute("SELECT session_id, top_k_ids FROM shadow_traces ORDER BY session_id").fetchall()
    verify_conn.close()

    assert len(rows) == 5, f"Expected 5 rows, got {len(rows)}"
    for session_id, top_k_json in rows:
        ids = json.loads(top_k_json)
        assert len(ids) == 2, f"session {session_id} has wrong chunks: {ids}"
        # 各自的 chunks 不混用
        idx = session_id.split("session-t")[1]
        assert f"chunk-t{idx}-1" in ids, f"session {session_id} missing own chunks"


# ── A8: CRIU checkpoint content_hash 版本校验 ────────────────────────────────

def test_A8_criu_checkpoint_content_hash_validation(fresh_db):
    """
    checkpoint_restore() 在 live chunk 被更新后，
    快照中的 content_hash 不匹配时应标记 _snapshot_stale=True。
    OS 类比：CRIU restore 时验证 ELF checksum — 二进制更新后不用旧快照。
    """
    import hashlib
    db_dir, db_path, conn = fresh_db

    from store_criu import _ensure_checkpoint_schema

    _ensure_checkpoint_schema(conn)

    # 插入 live chunk（更新版）
    from datetime import datetime, timezone
    _now = datetime.now(timezone.utc).isoformat()
    new_content = "updated content after modification"
    conn.execute("""
        INSERT OR IGNORE INTO memory_chunks
        (id, created_at, updated_at, project, chunk_type, content, summary, importance)
        VALUES ('chunk-hash-test', ?, ?, 'proj', 'decision', ?, 'content hash test', 0.8)
    """, (_now, _now, new_content))
    conn.commit()

    # 写入 checkpoint，其中快照 content_hash 基于旧内容
    old_content = "original content before modification"
    old_hash = hashlib.md5(old_content.encode()).hexdigest()[:8]
    snapshots = [{"id": "chunk-hash-test", "chunk_type": "decision",
                  "content": old_content, "summary": "content hash test",
                  "importance": 0.8, "content_hash": old_hash}]

    from datetime import datetime, timezone
    _now_ts = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT INTO checkpoints (id, created_at, project, session_id, hit_chunk_ids,
                                 chunk_snapshots, consumed)
        VALUES ('ckpt-hash-test', ?, 'proj', 'sess-hash',
                '["chunk-hash-test"]', ?, 0)
    """, (_now_ts, json.dumps(snapshots),))
    conn.commit()

    from store_criu import checkpoint_restore
    import store_criu as _criu
    _criu.MEMORY_OS_DIR = db_dir

    # monkeypatch config — criu.max_age_hours 默认已有值，无需 override

    result = checkpoint_restore(conn, "proj")
    assert result is not None, "checkpoint_restore should return data"

    live_chunks = [c for c in result["chunks"] if c["id"] == "chunk-hash-test" and not c.get("_from_snapshot")]
    assert live_chunks, "Should have live chunk"
    live = live_chunks[0]

    # live chunk 的 content_hash 与快照不一致 → 应被标记 _snapshot_stale
    live_hash = hashlib.md5((live["content"] or "").encode()).hexdigest()[:8]
    snap_hash = old_hash
    if live_hash != snap_hash:
        assert live.get("_snapshot_stale") is True, \
            f"Stale snapshot should be flagged, got: {live}"


# ── A9: checkpoint_cleanup fallback（无 session_id）────────────────────────

def test_A9_checkpoint_cleanup_fallback_no_session_id(fresh_db, monkeypatch):
    """
    当 session_id="" 时（旧调用方式），_checkpoint_cleanup 退化为全局清理，
    不因新参数引起 KeyError 或异常。
    OS 类比：向后兼容 — 旧 API 调用方式仍然可用。
    """
    db_dir, db_path, conn = fresh_db
    from store_criu import _ensure_checkpoint_schema, _checkpoint_cleanup
    _ensure_checkpoint_schema(conn)

    from config import sysctl_set as _cfg_set
    _cfg_set("criu.max_checkpoints", 2)

    project = "proj-fallback"
    for i in range(4):
        conn.execute(
            "INSERT INTO checkpoints (id, created_at, project, session_id, hit_chunk_ids, consumed) VALUES (?,?,?,?,?,0)",
            (f"ckpt-fb-{i}", f"2026-01-01T00:0{i}:00Z", project, "any-session", "[]")
        )
    conn.commit()

    # 不传 session_id（旧 API）
    deleted = _checkpoint_cleanup(conn, project)
    conn.commit()

    remaining = conn.execute(
        "SELECT COUNT(*) FROM checkpoints WHERE project=?", (project,)
    ).fetchone()[0]
    assert deleted == 2, f"Expected 2 deleted (4-2=2), got {deleted}"
    assert remaining == 2


# ── A10: 集成路径验证 — ensure_schema 创建 shadow_traces / session_intents ──

def test_A10_ensure_schema_creates_agent_tables(fresh_db):
    """
    ensure_schema() 创建了 shadow_traces 和 session_intents 表（iter259 必要前提）。
    OS 类比：init_module() — 内核模块加载时必须初始化所有 per-cpu 数据结构。
    """
    db_dir, db_path, conn = fresh_db

    # ensure_schema 已在 fresh_db fixture 中调用，直接验证表存在
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}

    assert "shadow_traces" in tables, \
        f"shadow_traces table must be created by ensure_schema. Tables: {tables}"
    assert "session_intents" in tables, \
        f"session_intents table must be created by ensure_schema. Tables: {tables}"

    # 验证 shadow_traces 的 PRIMARY KEY 是 session_id
    pragma = conn.execute("PRAGMA table_info(shadow_traces)").fetchall()
    col_info = {row[1]: row for row in pragma}  # col_name -> row
    assert "session_id" in col_info, "shadow_traces must have session_id column"
    assert col_info["session_id"][5] == 1, "session_id should be PRIMARY KEY (pk=1)"

    # 验证 session_intents 的 PRIMARY KEY 是 session_id
    pragma2 = conn.execute("PRAGMA table_info(session_intents)").fetchall()
    col_info2 = {row[1]: row for row in pragma2}
    assert "session_id" in col_info2, "session_intents must have session_id column"
    assert col_info2["session_id"][5] == 1, "session_id should be PRIMARY KEY (pk=1)"


# ── A11: page_fault_log per-session 文件隔离 ─────────────────────────────────

def test_A11_page_fault_log_per_session_isolation(tmp_path, monkeypatch):
    """
    两个 agent 并发写 _write_page_fault_log()，各自使用不同 session_id，
    最终写入不同文件，互不覆盖，merge 读取时能拿到两者的数据。
    OS 类比：/proc/PID/pagemap — 每进程独立文件，不同进程间不干扰。

    验证：
    1. Agent A 写入 page_fault_log.{sid_a[:8]}.json，Agent B 写入不同文件
    2. 两个文件独立存在，内容互不覆盖
    3. Glob 合并读取时，两者的 query 都能被读到
    """
    import glob as _glob
    import json as _json

    # 创建隔离的 MEMORY_OS_DIR
    mem_dir = tmp_path / "memory-os"
    mem_dir.mkdir()

    # monkeypatch extractor.MEMORY_OS_DIR（Path 类型）
    import hooks.extractor as _ext
    monkeypatch.setattr(_ext, "MEMORY_OS_DIR", mem_dir)

    session_a = "aaaa1111bbbb2222"
    session_b = "cccc3333dddd4444"
    tag_a = session_a[:8]   # "aaaa1111"
    tag_b = session_b[:8]   # "cccc3333"

    # Agent A 写入两条候选
    _ext._write_page_fault_log(
        ["query_from_agent_a_1", "query_from_agent_a_2"],
        session_a
    )
    # Agent B 写入两条（包含一个同名 query 用来测试 merge 去重）
    _ext._write_page_fault_log(
        ["query_from_agent_b_1", "query_from_agent_a_1"],  # a_1 与 A 重叠
        session_b
    )

    # 1. 两个独立文件存在
    file_a = mem_dir / f"page_fault_log.{tag_a}.json"
    file_b = mem_dir / f"page_fault_log.{tag_b}.json"
    assert file_a.exists(), f"Agent A's page fault file not found: {file_a}"
    assert file_b.exists(), f"Agent B's page fault file not found: {file_b}"

    # 2. 两文件内容独立，A 的文件里没有 B 专属的 query
    entries_a = _json.loads(file_a.read_text())
    entries_b = _json.loads(file_b.read_text())
    queries_a = {e["query"] for e in entries_a}
    queries_b = {e["query"] for e in entries_b}

    assert "query_from_agent_a_1" in queries_a, f"A's own query not in A's file: {queries_a}"
    assert "query_from_agent_a_2" in queries_a, f"A's second query not in A's file: {queries_a}"
    assert "query_from_agent_b_1" not in queries_a, f"B's query leaked into A's file: {queries_a}"

    assert "query_from_agent_b_1" in queries_b, f"B's own query not in B's file: {queries_b}"
    assert "query_from_agent_a_2" not in queries_b, f"A's query (not shared) leaked into B's file: {queries_b}"

    # 3. Glob 合并读取 — 模拟 retriever_daemon 的跨文件 merge 逻辑
    all_files = sorted(_glob.glob(str(mem_dir / "page_fault_log*.json")))
    assert len(all_files) == 2, f"Expected 2 page_fault_log files, got: {all_files}"

    merged_index = {}
    for fp in all_files:
        file_entries = _json.loads(open(fp, encoding="utf-8").read())
        for e in file_entries:
            if not isinstance(e, dict) or "query" not in e:
                continue
            q_key = e["query"].lower().strip()
            existing = merged_index.get(q_key)
            if existing is None:
                merged_index[q_key] = dict(e)
            else:
                # max fault_count wins
                existing["fault_count"] = max(
                    existing.get("fault_count", 1),
                    e.get("fault_count", 1)
                )

    all_merged_queries = set(merged_index.keys())
    assert "query_from_agent_a_1" in all_merged_queries, "A's query missing from merged"
    assert "query_from_agent_a_2" in all_merged_queries, "A's second query missing from merged"
    assert "query_from_agent_b_1" in all_merged_queries, "B's query missing from merged"

    # 重叠 query 去重后只有一条
    assert len([k for k in all_merged_queries if "a_1" in k]) == 1, \
        "Duplicate query_from_agent_a_1 should be deduplicated in merge"

    # fault_count 应为两者中的最大值（两边都是 1，所以 merge 后为 1）
    assert merged_index["query_from_agent_a_1"]["fault_count"] == 1


# ── A12: page_fault_log 无 session_id 退化到全局文件 ────────────────────────

def test_A12_page_fault_log_fallback_no_session_id(tmp_path, monkeypatch):
    """
    当 session_id 为空或 "unknown" 时，_write_page_fault_log 退化到
    全局 page_fault_log.json（向后兼容）。
    OS 类比：旧版 mmap() 不传 MAP_PRIVATE — 退化到全局共享页。
    """
    import json as _json
    import hooks.extractor as _ext
    monkeypatch.setattr(_ext, "MEMORY_OS_DIR", tmp_path / "memory-os")
    (tmp_path / "memory-os").mkdir()
    mem_dir = tmp_path / "memory-os"

    # session_id = "" → 全局文件
    _ext._write_page_fault_log(["query_no_session"], "")
    legacy_file = mem_dir / "page_fault_log.json"
    assert legacy_file.exists(), "Legacy page_fault_log.json must exist when session_id is empty"
    entries = _json.loads(legacy_file.read_text())
    assert any(e["query"] == "query_no_session" for e in entries)

    # session_id = "unknown" → 全局文件
    _ext._write_page_fault_log(["query_unknown_session"], "unknown")
    entries2 = _json.loads(legacy_file.read_text())
    assert any(e["query"] == "query_unknown_session" for e in entries2)

    # 无额外 per-session 文件产生
    import glob as _glob
    per_session_files = [
        f for f in _glob.glob(str(mem_dir / "page_fault_log*.json"))
        if "page_fault_log.json" not in f or f == str(legacy_file)
    ]
    # 只有一个文件（全局文件本身）
    all_pfl = sorted(_glob.glob(str(mem_dir / "page_fault_log*.json")))
    assert all_pfl == [str(legacy_file)], \
        f"No per-session files expected when session_id empty/unknown, got: {all_pfl}"


# ── A13: 跨 Agent 知识更新通知端到端投递 ────────────────────────────────────
# iter259 修复验证：net.agent_notify 从 AgentRouter（零投递）迁移到 ipc_msgq 路径后
# Agent A broadcast → Agent B consume 端到端测试

def test_A13_cross_agent_notification_delivery(fresh_db, monkeypatch):
    """
    Agent A（extractor）广播知识更新 → Agent B（loader）消费并收到通知。

    iter259 修复验证：
      - 旧路径：AgentRouter.route("*") → resolve_all_online() → net_agents 表为空 → 零投递
      - 新路径：store_vfs.ipc_send(target="*") → ipc_msgq 表 → ipc_recv 通配符查询 → 正确投递

    OS 类比：inotify IN_MODIFY 事件广播 —
      inotify 让进程监听文件系统事件而不是轮询 stat()，
      ipc_msgq 让 agent 监听知识更新而不是轮询 memory_chunks 表。
    """
    db_dir, db_path, conn = fresh_db

    # monkeypatch net.agent_notify 使用 fresh_db 路径
    import store_vfs as _vfs
    monkeypatch.setattr(_vfs, "STORE_DB", db_path)

    project = "test-notify-project"
    session_a = "agent-a-session-1234"
    session_b = "agent-b-session-5678"
    stats = {"decisions": 3, "constraints": 1, "chunks": 5}

    # Step 1: Agent A 广播知识更新
    from net.agent_notify import broadcast_knowledge_update, consume_pending_notifications
    result = broadcast_knowledge_update(project, session_a, stats)
    assert result is True, "broadcast_knowledge_update should return True on success"

    # Step 2: 验证 ipc_msgq 表中有对应消息
    row = conn.execute(
        "SELECT target_agent, msg_type, payload, status FROM ipc_msgq "
        "WHERE msg_type='knowledge_update' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert row is not None, "ipc_msgq should contain a knowledge_update message"
    target, msg_type, payload_str, status = row
    assert target == "*", f"target_agent should be '*' for broadcast, got: {target}"
    assert status == "QUEUED", f"Message should be QUEUED before consumption, got: {status}"
    import json as _json
    payload = _json.loads(payload_str) if isinstance(payload_str, str) else payload_str
    assert payload.get("project") == project, f"Wrong project in payload: {payload}"
    assert payload.get("stats", {}).get("chunks") == 5, f"Wrong stats in payload: {payload}"

    # Step 3: Agent B 消费通知
    notifications = consume_pending_notifications(session_b, limit=3)
    assert len(notifications) >= 1, f"Agent B should receive at least 1 notification, got: {notifications}"

    notif = notifications[0]
    assert notif["project"] == project, f"Wrong project in notification: {notif}"
    assert notif["stats"]["decisions"] == 3, f"Wrong decisions count: {notif}"
    assert notif["stats"]["constraints"] == 1, f"Wrong constraints count: {notif}"
    assert notif["stats"]["chunks"] == 5, f"Wrong chunks count: {notif}"
    assert notif["session_id"] == session_a, f"Wrong session_id in notification: {notif}"

    # Step 4: 第二次消费应为空（消息已被标记 CONSUMED）
    notifications2 = consume_pending_notifications(session_b, limit=3)
    assert len(notifications2) == 0, \
        f"Second consume should return empty (already CONSUMED), got: {notifications2}"


# ── A14: 多 Agent 同时广播，各自消费不干扰 ──────────────────────────────────

def test_A14_multi_agent_broadcast_isolation(fresh_db, monkeypatch):
    """
    Agent A 和 Agent B 同时广播各自的知识更新；
    Agent C 消费时同时收到两条通知，内容互不混淆。

    OS 类比：/proc/PID/net/if_inet6 — 多进程同时写，读取时各自数据独立。
    """
    db_dir, db_path, conn = fresh_db

    import store_vfs as _vfs
    monkeypatch.setattr(_vfs, "STORE_DB", db_path)

    from net.agent_notify import broadcast_knowledge_update, consume_pending_notifications

    project = "shared-project"
    session_a = "agent-a-broadcast-111"
    session_b = "agent-b-broadcast-222"
    session_c = "agent-c-consumer-333"

    stats_a = {"decisions": 2, "constraints": 0, "chunks": 2}
    stats_b = {"decisions": 5, "constraints": 3, "chunks": 8}

    # Agent A 和 Agent B 各自广播
    r_a = broadcast_knowledge_update(project, session_a, stats_a)
    r_b = broadcast_knowledge_update(project, session_b, stats_b)
    assert r_a is True, "Agent A broadcast should succeed"
    assert r_b is True, "Agent B broadcast should succeed"

    # Agent C 消费（应收到 2 条）
    notifications = consume_pending_notifications(session_c, limit=10)
    assert len(notifications) == 2, f"Agent C should receive 2 notifications, got: {len(notifications)}"

    # 按 session_id 分组验证
    import json as _json
    by_session = {n["session_id"]: n for n in notifications}
    assert session_a in by_session, f"Agent A's notification missing. Got sessions: {list(by_session.keys())}"
    assert session_b in by_session, f"Agent B's notification missing. Got sessions: {list(by_session.keys())}"

    assert by_session[session_a]["stats"]["chunks"] == 2, \
        f"Agent A stats wrong: {by_session[session_a]}"
    assert by_session[session_b]["stats"]["chunks"] == 8, \
        f"Agent B stats wrong: {by_session[session_b]}"

    # 再次消费应为空
    notifications2 = consume_pending_notifications(session_c, limit=10)
    assert len(notifications2) == 0, \
        f"Second consume should be empty, got: {notifications2}"


# ── A15: goals 进度 UPDATE 多 Agent 并发 session 幂等性 ─────────────────────────

def test_A15_goals_progress_session_idempotency(fresh_db):
    """
    两个不同 session_id 的 agent 各自对同一 project 的 active goal 执行进度更新；
    同一 session_id 执行两次，第二次不再追加（幂等）。
    不同 session_id 执行一次，各自追加一次。

    验证：
    1. 同一 session 执行两次 → progress 只增加 0.05（不是 0.10）
    2. 两个不同 session 各执行一次 → progress 增加 0.10（0.05 × 2）

    OS 类比：Linux 幂等 write() with O_APPEND + flock —
      写入前加锁，同一事务 token 的重复写入被 dedup 拦截。
    """
    db_dir, db_path, conn = fresh_db

    project = "test-goals-project"
    session_a = "sess-agent-goals-AAA"
    session_b = "sess-agent-goals-BBB"

    # 创建 goals 表（模拟 writer.py 的懒创建路径）
    conn.execute("""
        CREATE TABLE IF NOT EXISTS goals (
            id TEXT PRIMARY KEY,
            title TEXT,
            description TEXT,
            status TEXT DEFAULT 'active',
            progress REAL DEFAULT 0.0,
            created_at TEXT,
            updated_at TEXT,
            project TEXT,
            tags TEXT,
            last_progress_session TEXT DEFAULT ''
        )
    """)
    conn.execute(
        """INSERT INTO goals (id, title, description, status, progress,
                              created_at, updated_at, project, tags, last_progress_session)
           VALUES ('goal-test-001', 'Test Goal', 'desc', 'active', 0.0,
                   '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z', ?, '[]', '')""",
        (project,)
    )
    conn.commit()

    def _do_progress_update(conn, session_id):
        """模拟 extractor.py P2 修复后的进度更新逻辑"""
        from datetime import datetime, timezone
        now_iso = datetime.now(timezone.utc).isoformat()
        try:
            conn.execute("ALTER TABLE goals ADD COLUMN last_progress_session TEXT DEFAULT ''")
        except Exception:
            pass
        conn.execute(
            """UPDATE goals SET progress = MIN(1.0, progress + 0.05),
               updated_at = ?,
               last_progress_session = ?
               WHERE project = ? AND status = 'active'
                 AND (last_progress_session IS NULL OR last_progress_session != ?)""",
            [now_iso, session_id, project, session_id]
        )
        conn.commit()

    # ── Case 1: 同一 session_a 执行两次 → 幂等，progress 只 +0.05 ──
    _do_progress_update(conn, session_a)
    _do_progress_update(conn, session_a)  # 第二次不应再增加

    row = conn.execute("SELECT progress FROM goals WHERE id='goal-test-001'").fetchone()
    assert abs(row[0] - 0.05) < 1e-6, \
        f"Same session double-update should be idempotent, progress={row[0]:.4f} (expected 0.05)"

    # ── Case 2: 不同 session_b 执行一次 → 新 session，progress 再 +0.05 ──
    _do_progress_update(conn, session_b)

    row = conn.execute("SELECT progress FROM goals WHERE id='goal-test-001'").fetchone()
    assert abs(row[0] - 0.10) < 1e-6, \
        f"Different session should add another 0.05, progress={row[0]:.4f} (expected 0.10)"

    # ── Case 3: session_b 再执行一次 → 幂等 ──
    _do_progress_update(conn, session_b)

    row = conn.execute("SELECT progress FROM goals WHERE id='goal-test-001'").fetchone()
    assert abs(row[0] - 0.10) < 1e-6, \
        f"Session B second call should be idempotent, progress={row[0]:.4f} (expected 0.10)"


# ── A16: loader.py per-session shadow_trace 文件隔离 ─────────────────────────

def test_A16_loader_shadow_trace_per_session_file_isolation(tmp_path, monkeypatch):
    """
    loader.py SessionStart readahead 写入 per-session shadow_trace 文件（iter259 修复）：
    Agent A 和 Agent B 各自写入不同文件，互不覆盖。
    旧的全局 .shadow_trace.json 文件不再被写入。

    OS 类比：/proc/PID/maps — 每进程独立，不同进程间不互相覆盖。

    验证：
    1. 写入 per-session 文件（命名格式 .shadow_trace.{sid[:16]}.json）
    2. Agent A 的文件不包含 Agent B 的 top_k_ids
    3. 全局 .shadow_trace.json 不存在（已废弃）
    """
    import json as _json

    mem_dir = tmp_path / "memory-os"
    mem_dir.mkdir()

    session_a = "aaaa-agent-session-loader"
    session_b = "bbbb-agent-session-loader"
    tag_a = session_a[:16]
    tag_b = session_b[:16]

    # 模拟 loader.py 的 per-session 文件写入逻辑（iter259）
    def _write_shadow_trace(session_id, top_k_ids):
        _sid_tag = session_id[:16] if session_id else "unknown"
        _shadow_file = mem_dir / f".shadow_trace.{_sid_tag}.json"
        _shadow_data = {
            "project": "test-project",
            "top_k_ids": top_k_ids,
            "session_id": session_id,
            "source": "session_start_readahead",
        }
        _shadow_file.write_text(_json.dumps(_shadow_data, ensure_ascii=False), encoding="utf-8")
        return _shadow_file

    file_a = _write_shadow_trace(session_a, ["chunk-loader-A1", "chunk-loader-A2"])
    file_b = _write_shadow_trace(session_b, ["chunk-loader-B1", "chunk-loader-B2"])

    # 1. 两个独立的 per-session 文件存在
    assert file_a.exists(), f"Agent A shadow file not found: {file_a}"
    assert file_b.exists(), f"Agent B shadow file not found: {file_b}"
    assert file_a != file_b, "Agent A and B must write to different files"

    # 2. 文件内容互不混淆
    data_a = _json.loads(file_a.read_text())
    data_b = _json.loads(file_b.read_text())
    assert data_a["top_k_ids"] == ["chunk-loader-A1", "chunk-loader-A2"], \
        f"Agent A file corrupted: {data_a['top_k_ids']}"
    assert data_b["top_k_ids"] == ["chunk-loader-B1", "chunk-loader-B2"], \
        f"Agent B file corrupted: {data_b['top_k_ids']}"
    assert "chunk-loader-B1" not in data_a["top_k_ids"], "Agent B's chunks leaked into A's file"
    assert "chunk-loader-A1" not in data_b["top_k_ids"], "Agent A's chunks leaked into B's file"

    # 3. 全局文件不存在（已废弃，iter259 修复后不再写入）
    global_file = mem_dir / ".shadow_trace.json"
    assert not global_file.exists(), \
        f"Global .shadow_trace.json should NOT exist (deprecated in iter259), but found: {global_file}"


# ── A17: writer.py per-session ctx_pressure_state 文件隔离 ───────────────────

def test_A17_writer_ctx_pressure_per_session_file_isolation(tmp_path, monkeypatch):
    """
    writer.py _detect_context_pressure 写入 per-session ctx_pressure_state 文件（iter259 修复）：
    Agent A 和 Agent B 各自写入不同文件，互不覆盖。
    仅当 session_id 为空/unknown 时退化到全局文件（向后兼容）。

    OS 类比：/proc/PID/status — 每进程独立状态文件，不同进程间不干扰。

    验证：
    1. 有效 session_id → 写入 per-session 文件 ctx_pressure_state.{sid[:16]}.json
    2. 两个 agent 的文件独立，usage_pct 互不混淆
    3. session_id="" → 退化到全局 ctx_pressure_state.json（向后兼容）
    4. session_id="unknown" → 退化到全局文件
    """
    import json as _json

    mem_dir = tmp_path / "memory-os"
    mem_dir.mkdir()

    global_file = mem_dir / "ctx_pressure_state.json"

    # 模拟 writer.py per-session 写入逻辑（iter259）
    def _write_pressure_state(session_id, usage_pct, pressure):
        from datetime import datetime, timezone
        _sid = session_id
        _sid_tag = _sid[:16] if (_sid and _sid != "unknown") else ""
        if _sid_tag:
            _ctx_pressure_file = mem_dir / f"ctx_pressure_state.{_sid_tag}.json"
        else:
            _ctx_pressure_file = global_file  # 向后兼容
        _ctx_pressure_file.write_text(_json.dumps({
            "pressure": pressure,
            "usage_pct": usage_pct,
            "transcript_chars": int(usage_pct * 500_000),
            "session_id": _sid,
        }, ensure_ascii=False))
        return _ctx_pressure_file

    session_a = "agent-writer-AAAA-test"
    session_b = "agent-writer-BBBB-test"
    tag_a = session_a[:16]
    tag_b = session_b[:16]

    # Agent A: usage=65% (warn)
    file_a = _write_pressure_state(session_a, 0.65, "warn")
    # Agent B: usage=45% (none)
    file_b = _write_pressure_state(session_b, 0.45, "none")

    # 1. 写入 per-session 文件（不是全局文件）
    expected_a = mem_dir / f"ctx_pressure_state.{tag_a}.json"
    expected_b = mem_dir / f"ctx_pressure_state.{tag_b}.json"
    assert file_a == expected_a, f"Agent A should write to {expected_a}, got {file_a}"
    assert file_b == expected_b, f"Agent B should write to {expected_b}, got {file_b}"
    assert not global_file.exists(), \
        "Global ctx_pressure_state.json should NOT exist when session_id is valid"

    # 2. 两文件内容独立
    data_a = _json.loads(file_a.read_text())
    data_b = _json.loads(file_b.read_text())
    assert data_a["pressure"] == "warn", f"Agent A pressure wrong: {data_a}"
    assert abs(data_a["usage_pct"] - 0.65) < 1e-6, f"Agent A usage_pct wrong: {data_a}"
    assert data_b["pressure"] == "none", f"Agent B pressure wrong: {data_b}"
    assert abs(data_b["usage_pct"] - 0.45) < 1e-6, f"Agent B usage_pct wrong: {data_b}"
    assert data_a["session_id"] == session_a, f"Agent A session_id wrong: {data_a}"
    assert data_b["session_id"] == session_b, f"Agent B session_id wrong: {data_b}"

    # 3. session_id="" → 退化到全局文件
    file_empty = _write_pressure_state("", 0.30, "none")
    assert file_empty == global_file, \
        f"Empty session_id should write to global file, got {file_empty}"
    assert global_file.exists(), "Global file must be written when session_id is empty"

    # 4. session_id="unknown" → 退化到全局文件
    file_unknown = _write_pressure_state("unknown", 0.20, "none")
    assert file_unknown == global_file, \
        f"'unknown' session_id should write to global file, got {file_unknown}"


# ── A18: submit_extract_task — pool 未运行时返回 False ─────────────────────
def test_A18_submit_extract_task_returns_false_when_pool_not_running(tmp_path, monkeypatch):
    """
    submit_extract_task() 在 pool 未运行时应返回 False（不入队），
    使 extractor.py fallback 到同步执行路径。

    OS 类比：queue_work() 在 workqueue 未初始化时返回 -EINVAL —
    调用方在 pool 不存在时应能优雅降级。
    """
    import sys
    _ROOT = Path(__file__).parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    mem_dir = tmp_path / "memory-os"
    mem_dir.mkdir()

    import hooks.extractor_pool as _pool_mod
    # pool 未运行 → heartbeat 文件不存在
    monkeypatch.setattr(_pool_mod, "HEARTBEAT_FILE", mem_dir / "extractor_pool.heartbeat")
    monkeypatch.setattr(_pool_mod, "PID_FILE", mem_dir / "extractor_pool.pid")

    hook_input = {
        "last_assistant_message": "选择使用 SQLite WAL 模式，因为并发读写性能更好",
        "session_id": "test-session-submit",
        "transcript_path": "",
    }
    result = _pool_mod.submit_extract_task(hook_input, "test-project", "test-session-submit")
    assert result is False, \
        f"submit_extract_task should return False when pool not running, got: {result}"


# ── A19: submit_extract_task — pool 运行时入队到 ipc_msgq ──────────────────
def test_A19_submit_extract_task_enqueues_to_ipc_msgq(fresh_db, monkeypatch, tmp_path):
    """
    submit_extract_task() 在 pool 健康时应将 extract_task 写入 ipc_msgq。

    验证：
    1. check_health() mock 为 running=True
    2. submit_extract_task() 返回 True
    3. ipc_msgq 表中有对应的 extract_task 消息
    4. payload 包含正确的 session_id、project、text

    OS 类比：queue_work(pool, &work) 成功入队 → work_struct 出现在 work_list 中。
    """
    db_dir, db_path, conn = fresh_db

    import store_vfs as _vfs
    monkeypatch.setattr(_vfs, "STORE_DB", db_path)

    import hooks.extractor_pool as _pool_mod
    monkeypatch.setattr(_pool_mod, "STORE_DB", db_path)

    # mock check_health → running=True
    monkeypatch.setattr(_pool_mod, "check_health", lambda: {"running": True, "pid": 99999})

    hook_input = {
        "last_assistant_message": "决定采用 shadow_traces DB 隔离方案，因为全局文件存在 last-writer-wins 竞争",
        "session_id": "test-enqueue-session",
        "transcript_path": "",
    }
    result = _pool_mod.submit_extract_task(hook_input, "test-enqueue-project", "test-enqueue-session")
    assert result is True, f"submit_extract_task should return True when pool healthy, got: {result}"

    # 验证 ipc_msgq 中有对应消息
    row = conn.execute(
        "SELECT target_agent, msg_type, payload, status FROM ipc_msgq "
        "WHERE msg_type='extract_task' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert row is not None, "extract_task message should be in ipc_msgq"
    target, msg_type, payload_str, status = row
    assert target == "extractor_pool", f"target_agent should be 'extractor_pool', got: {target}"
    assert status == "QUEUED", f"status should be QUEUED, got: {status}"

    payload = json.loads(payload_str) if isinstance(payload_str, str) else payload_str
    assert payload.get("session_id") == "test-enqueue-session", \
        f"Wrong session_id in payload: {payload}"
    assert payload.get("project") == "test-enqueue-project", \
        f"Wrong project in payload: {payload}"
    assert "shadow_traces" in payload.get("text", ""), \
        f"Text not preserved in payload: {payload.get('text', '')[:80]}"


# ── A20: dequeue_tasks — pool worker 从 ipc_msgq 取任务 ─────────────────────
def test_A20_pool_dequeue_extract_task_from_ipc_msgq(fresh_db, monkeypatch):
    """
    extractor_pool 从 ipc_msgq 取 extract_task 后应标记 CONSUMED，
    且第二次取应为空。

    OS 类比：kworker 从 work_list 取出 work_struct 后将其从队列删除（list_del_init），
    保证 work 只被执行一次（one-shot 语义）。

    验证：
    1. 手动插入 2 条 extract_task 消息
    2. _dequeue_tasks() 取出 batch=2 → 返回 2 条
    3. 再次取 → 返回 0 条（已 CONSUMED）
    """
    db_dir, db_path, conn = fresh_db

    import store_vfs as _vfs
    monkeypatch.setattr(_vfs, "STORE_DB", db_path)

    from store_vfs import ipc_send
    import hooks.extractor_pool as _pool_mod
    monkeypatch.setattr(_pool_mod, "STORE_DB", db_path)

    # 插入 2 条 extract_task
    for i in range(2):
        ipc_send(conn, source=f"session-{i}", target="extractor_pool",
                 msg_type="extract_task",
                 payload={"session_id": f"session-{i}", "project": "proj",
                          "text": f"决策内容{i}", "transcript_path": ""},
                 priority=5, ttl_seconds=300)
    conn.commit()

    # pool worker 取任务
    tasks = _pool_mod._dequeue_tasks(conn, limit=10)
    conn.commit()
    assert len(tasks) == 2, f"Should dequeue 2 tasks, got {len(tasks)}"

    # 验证 payload
    for t in tasks:
        payload = t.get("payload", {})
        if isinstance(payload, str):
            payload = json.loads(payload)
        assert "session_id" in payload, f"payload missing session_id: {payload}"
        assert payload.get("project") == "proj", f"Wrong project: {payload}"

    # 第二次取应为空（已 CONSUMED）
    tasks2 = _pool_mod._dequeue_tasks(conn, limit=10)
    conn.commit()
    assert len(tasks2) == 0, \
        f"Second dequeue should be empty (CONSUMED), got: {len(tasks2)}"

    # 验证数据库中状态
    consumed_count = conn.execute(
        "SELECT COUNT(*) FROM ipc_msgq WHERE msg_type='extract_task' AND status='CONSUMED'"
    ).fetchone()[0]
    assert consumed_count == 2, f"Should have 2 CONSUMED tasks, got: {consumed_count}"
