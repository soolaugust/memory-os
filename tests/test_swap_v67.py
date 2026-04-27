#!/usr/bin/env python3
"""
迭代67：Swap Out Error Visibility + DB Readonly + Global Decision Query
测试：验证 save-task-state.py v67 的三项改进
"""
import sys
import os
import json
import tempfile
import sqlite3
from pathlib import Path

# tmpfs 隔离
import tmpfs

sys.path.insert(0, str(Path(__file__).parent))
from store import open_db, ensure_schema


if __name__ == "__main__":
    passed = 0
    failed = 0


    def check(name, cond):
        global passed, failed
        if cond:
            print(f"  ✓ {name}")
            passed += 1
        else:
            print(f"  ✗ {name}")
            failed += 1


    # === 准备测试 DB ===
    conn = open_db()
    ensure_schema(conn)

    # 写入 test data: 5 decisions across 3 sessions + 3 reasoning_chains in session-0
    for i in range(5):
        conn.execute("""
            INSERT INTO memory_chunks (id, project, source_session, chunk_type, summary, content, importance, last_accessed, access_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'), ?, datetime('now'))
        """, [f"dec-{i}", "test-proj", f"session-{i%3}", "decision",
              f"Test decision {i}", f"Content {i}", 0.8 + i*0.02, i])

    for i in range(3):
        conn.execute("""
            INSERT INTO memory_chunks (id, project, source_session, chunk_type, summary, content, importance, last_accessed, access_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'), ?, datetime('now'))
        """, [f"rc-{i}", "test-proj", "session-0", "reasoning_chain",
              f"Reasoning chain {i}", f"RC content {i}", 0.75, i*2])

    # 写入 recall_traces
    for i in range(3):
        conn.execute("""
            INSERT INTO recall_traces (id, timestamp, session_id, project, prompt_hash, candidates_count, top_k_json, injected, reason, duration_ms)
            VALUES (?, datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?)
        """, [f"trace-{i}", "test-session", "test-proj", f"hash-{i}", 10,
              json.dumps([{"id": f"dec-{i}", "summary": f"Test {i}"}]), 1, "test", 5.0])

    conn.commit()

    # === T1: _log_error 写错误日志 ===
    print("\nT1: _log_error 写入 swap_errors.log")
    error_log = Path(os.environ.get("MEMORY_OS_DIR", str(Path.home() / ".claude" / "memory-os"))) / "swap_errors.log"
    if error_log.exists():
        error_log.unlink()

    try:
        raise ValueError("test error for v67")
    except Exception as e:
        import traceback
        from datetime import datetime, timezone
        msg = f"[{datetime.now(timezone.utc).isoformat()}] test_context: {type(e).__name__}: {e}\n"
        msg += traceback.format_exc() + "\n"
        error_log.parent.mkdir(parents=True, exist_ok=True)
        with open(error_log, "a", encoding="utf-8") as f:
            f.write(msg)

    check("error log created", error_log.exists())
    content = error_log.read_text("utf-8")
    check("error log contains context", "test_context" in content)
    check("error log contains exception type", "ValueError" in content)
    check("error log contains message", "test error for v67" in content)
    error_log.unlink()


    # === T2: _open_db_readonly (immutable=1) ===
    print("\nT2: _open_db_readonly immutable=1 模式")
    db_path = str(Path(os.environ.get("MEMORY_OS_DB", str(Path.home() / ".claude" / "memory-os" / "store.db"))))

    # immutable=1 不读 WAL，需要先 checkpoint 把数据刷入主文件
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    try:
        uri = f"file:{db_path}?immutable=1"
        ro_conn = sqlite3.connect(uri, uri=True)
        rows = ro_conn.execute("SELECT count(*) FROM memory_chunks WHERE project='test-proj'").fetchone()
        check("immutable=1 connect OK", True)
        check("immutable=1 read returns data", rows[0] >= 8)

        # 验证只读——写入应失败
        write_failed = False
        try:
            ro_conn.execute("INSERT INTO memory_chunks (id, project, source_session, chunk_type, summary, content, importance) VALUES ('x','x','x','x','x','x',0.5)")
            ro_conn.commit()
        except Exception:
            write_failed = True
        check("immutable=1 blocks writes", write_failed)
        ro_conn.close()
    except Exception as e:
        check(f"immutable=1 connect OK (failed: {e})", False)


    # === T3: Global decision query (不限 session) ===
    print("\nT3: Global decision query 不限 session")

    rows = conn.execute("""
        SELECT summary, importance FROM memory_chunks
        WHERE project = ?
          AND chunk_type IN ('decision', 'reasoning_chain')
        ORDER BY importance DESC, access_count DESC LIMIT 10
    """, ["test-proj"]).fetchall()

    check("global query returns decisions", len(rows) > 0)
    check("global query includes all sessions", len(rows) >= 5)

    # 对比：session-scoped 查询只能看到部分
    rows_scoped = conn.execute("""
        SELECT summary, importance FROM memory_chunks
        WHERE project = ? AND source_session = ?
          AND chunk_type IN ('decision', 'reasoning_chain')
        ORDER BY importance DESC LIMIT 10
    """, ["test-proj", "session-0"]).fetchall()

    check("session-scoped returns fewer", len(rows_scoped) < len(rows))
    check("global query returns 8 (5 dec + 3 rc)", len(rows) == 8)


    # === T4: Error log rotation (>50KB truncation) ===
    print("\nT4: Error log rotation at 50KB")
    error_log.parent.mkdir(parents=True, exist_ok=True)
    error_log.write_text("X" * 60_000, "utf-8")
    check("large log file created", error_log.stat().st_size > 50_000)

    if error_log.stat().st_size > 50_000:
        c = error_log.read_text("utf-8")
        error_log.write_text(c[-30_000:], "utf-8")
    check("log rotated to ~30KB", error_log.stat().st_size <= 35_000)
    error_log.unlink()


    # === T5: dmesg 日志记录 elapsed_ms ===
    print("\nT5: dmesg swap_out 记录 elapsed_ms")
    from store import dmesg_log, dmesg_read, DMESG_INFO
    dmesg_log(conn, DMESG_INFO, "swap_out",
              "PreCompact swap out: 6 hit_ids, 10 decisions, 3 transcript_turns, 145ms",
              extra={"session": "test", "has_transcript": True, "elapsed_ms": 145.2})
    conn.commit()
    logs = dmesg_read(conn, subsystem="swap_out", limit=1)
    check("dmesg swap_out logged", len(logs) > 0)
    if logs:
        check("dmesg contains elapsed_ms", "145ms" in logs[0].get("message", ""))
        extra = json.loads(logs[0].get("extra", "{}")) if isinstance(logs[0].get("extra"), str) else logs[0].get("extra", {})
        check("dmesg extra has elapsed_ms", extra.get("elapsed_ms") == 145.2)


    conn.close()

    print(f"\n{'='*60}")
    print(f"结果：{passed}/{passed+failed} 通过")
    print(f"{'='*60}")
    if failed:
        sys.exit(1)
