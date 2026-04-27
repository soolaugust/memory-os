#!/usr/bin/env python3
"""
迭代49：CRIU — Checkpoint/Restore 测试
OS 类比：CRIU checkpoint → restore → cleanup 全流程验证
"""
import json
import sqlite3
import uuid
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
import tmpfs  # noqa: F401 — tmpfs isolation (iter54), must precede store import
from store import (
    open_db, ensure_schema, insert_chunk, checkpoint_dump,
    checkpoint_restore, checkpoint_collect_hits, _ensure_checkpoint_schema,
    _checkpoint_cleanup,
)
from config import get as _sysctl

PASS = 0
FAIL = 0


def _test(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}")


def _fresh_db():
    """创建内存数据库用于测试。"""
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA journal_mode=WAL")
    ensure_schema(conn)
    _ensure_checkpoint_schema(conn)
    return conn


def _insert_test_chunk(conn, project, chunk_id=None, chunk_type="decision",
                       summary="test summary", importance=0.8):
    """插入测试 chunk。"""
    if chunk_id is None:
        chunk_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT OR REPLACE INTO memory_chunks
        (id, created_at, updated_at, project, source_session,
         chunk_type, content, summary, tags, importance,
         retrievability, last_accessed)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (chunk_id, now, now, project, "test-session",
          chunk_type, f"[{chunk_type}] {summary}", summary,
          "[]", importance, 0.3, now))
    conn.commit()
    return chunk_id


def _insert_test_trace(conn, project, session_id, chunk_ids):
    """插入测试 recall_trace。"""
    now = datetime.now(timezone.utc).isoformat()
    trace_id = str(uuid.uuid4())
    top_k = [{"id": cid, "score": 0.8} for cid in chunk_ids]
    conn.execute("""
        INSERT INTO recall_traces
        (id, timestamp, session_id, project, prompt_hash,
         candidates_count, top_k_json, injected, reason, duration_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (trace_id, now, session_id, project, "test-hash",
          len(chunk_ids), json.dumps(top_k), 1, "test", 1.0))
    conn.commit()


# ── Test 1: checkpoint_dump 基本功能 ──

if __name__ == "__main__":
    print("Test 1: checkpoint_dump 基本功能")
    conn = _fresh_db()
    project = "test-project"
    session_id = "sess-001"

    # 插入 chunks
    cid1 = _insert_test_chunk(conn, project, summary="BM25 算法实测延迟 3ms")
    cid2 = _insert_test_chunk(conn, project, summary="选择 FTS5 替代 Python BM25")
    cid3 = _insert_test_chunk(conn, project, summary="连接池在 hook subprocess 中无意义")

    result = checkpoint_dump(conn, project, session_id,
                             [cid1, cid2, cid3],
                             ["bm25", "fts5"],
                             ["检索优化"])
    conn.commit()

    _test("checkpoint_id 非空", result["checkpoint_id"] is not None)
    _test("saved_ids=3", result["saved_ids"] == 3)
    _test("checkpoint_id 格式正确", result["checkpoint_id"].startswith("ckpt-"))
    conn.close()

    # ── Test 2: checkpoint_restore 基本功能 ──
    print("\nTest 2: checkpoint_restore 基本功能")
    conn = _fresh_db()
    project = "test-project"

    cid1 = _insert_test_chunk(conn, project, summary="决策A")
    cid2 = _insert_test_chunk(conn, project, summary="决策B")

    checkpoint_dump(conn, project, "sess-001", [cid1, cid2], ["hint1"], ["topic1"])
    conn.commit()

    restored = checkpoint_restore(conn, project)
    conn.commit()

    _test("restore 成功", restored is not None)
    _test("chunks 数量=2", len(restored["chunks"]) == 2)
    _test("madvise_hints 包含 hint1", "hint1" in restored["madvise_hints"])
    _test("query_topics 包含 topic1", "topic1" in restored["query_topics"])
    _test("age_hours < 1", restored["age_hours"] < 1)

    # 迭代87: 不再消费 checkpoint，二次 restore 应仍返回数据
    restored2 = checkpoint_restore(conn, project)
    _test("二次 restore 仍返回数据（不消费）", restored2 is not None)
    conn.close()

    # ── Test 3: checkpoint 过期 ──
    print("\nTest 3: checkpoint 过期")
    conn = _fresh_db()
    project = "test-project"

    cid1 = _insert_test_chunk(conn, project, summary="旧决策")

    # 手动插入一个过期的 checkpoint
    old_time = (datetime.now(timezone.utc) - timedelta(hours=100)).isoformat()
    conn.execute("""
        INSERT INTO checkpoints (id, created_at, project, session_id,
                                 hit_chunk_ids, madvise_hints, query_topics)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, ("ckpt-old", old_time, project, "old-session",
          json.dumps([cid1]), "[]", "[]"))
    conn.commit()

    restored = checkpoint_restore(conn, project)
    _test("过期 checkpoint 返回 None", restored is None)

    # 确认 checkpoint 被标记为 consumed
    consumed = conn.execute(
        "SELECT consumed FROM checkpoints WHERE id = ?", ("ckpt-old",)
    ).fetchone()
    _test("过期 checkpoint 标记为 consumed", consumed[0] == 1)
    conn.close()

    # ── Test 4: FIFO 淘汰 ──
    print("\nTest 4: FIFO 淘汰（max_checkpoints=3）")
    conn = _fresh_db()
    project = "test-project"

    cid1 = _insert_test_chunk(conn, project, summary="chunk1")

    # 插入 5 个 checkpoint
    for i in range(5):
        checkpoint_dump(conn, project, f"sess-{i}", [cid1], [], [])
        conn.commit()
        time.sleep(0.01)  # 确保时间戳不同

    # 应该只保留最新 3 个
    count = conn.execute(
        "SELECT COUNT(*) FROM checkpoints WHERE project = ?", (project,)
    ).fetchone()[0]
    _test(f"FIFO 淘汰后剩余={count} (期望<=3)", count <= 3)
    conn.close()

    # ── Test 5: 空 hit_ids 不创建 checkpoint ──
    print("\nTest 5: 空 hit_ids 不创建 checkpoint")
    conn = _fresh_db()
    project = "test-project"

    result = checkpoint_dump(conn, project, "sess-empty", [], [], [])
    _test("空 hit_ids 返回 checkpoint_id=None", result["checkpoint_id"] is None)
    _test("空 hit_ids saved_ids=0", result["saved_ids"] == 0)
    conn.close()

    # ── Test 6: chunk 被淘汰后 restore 返回存活的 ──
    print("\nTest 6: chunk 被淘汰后 restore 返回存活的子集")
    conn = _fresh_db()
    project = "test-project"

    cid1 = _insert_test_chunk(conn, project, summary="存活chunk")
    cid2 = "deleted-chunk-id"  # 不在 memory_chunks 中

    checkpoint_dump(conn, project, "sess-partial", [cid1, cid2], [], [])
    conn.commit()

    restored = checkpoint_restore(conn, project)
    conn.commit()

    _test("部分存活 restore 成功", restored is not None)
    _test("只返回存活的 chunk（1个）", len(restored["chunks"]) == 1)
    _test("存活 chunk summary 正确", restored["chunks"][0]["summary"] == "存活chunk")
    conn.close()

    # ── Test 7: checkpoint_collect_hits 从 recall_traces 收集 ──
    print("\nTest 7: checkpoint_collect_hits 从 recall_traces 收集")
    conn = _fresh_db()
    project = "test-project"
    session_id = "sess-collect"

    cid1 = _insert_test_chunk(conn, project, summary="hit1")
    cid2 = _insert_test_chunk(conn, project, summary="hit2")
    cid3 = _insert_test_chunk(conn, project, summary="hit3")

    _insert_test_trace(conn, project, session_id, [cid1, cid2])
    _insert_test_trace(conn, project, session_id, [cid2, cid3])  # cid2 重复

    hits = checkpoint_collect_hits(conn, project, session_id, limit=10)
    _test("收集到 3 个唯一 hit", len(hits) == 3)
    _test("包含 cid1", cid1 in hits)
    _test("包含 cid2（去重）", cid2 in hits)
    _test("包含 cid3", cid3 in hits)
    conn.close()

    # ── Test 8: 多项目隔离 ──
    print("\nTest 8: 多项目隔离")
    conn = _fresh_db()

    cid_a = _insert_test_chunk(conn, "project-A", summary="项目A决策")
    cid_b = _insert_test_chunk(conn, "project-B", summary="项目B决策")

    checkpoint_dump(conn, "project-A", "sess-A", [cid_a], [], [])
    checkpoint_dump(conn, "project-B", "sess-B", [cid_b], [], [])
    conn.commit()

    restored_a = checkpoint_restore(conn, "project-A")
    conn.commit()
    restored_b = checkpoint_restore(conn, "project-B")
    conn.commit()

    _test("项目A restore 成功", restored_a is not None)
    _test("项目B restore 成功", restored_b is not None)
    _test("项目A 只含自己的 chunk", restored_a["chunks"][0]["summary"] == "项目A决策")
    _test("项目B 只含自己的 chunk", restored_b["chunks"][0]["summary"] == "项目B决策")
    conn.close()

    # ── Test 9: 去重验证 ──
    print("\nTest 9: hit_ids 去重")
    conn = _fresh_db()
    project = "test-project"

    cid1 = _insert_test_chunk(conn, project, summary="唯一chunk")

    # 传入重复 IDs
    result = checkpoint_dump(conn, project, "sess-dedup",
                             [cid1, cid1, cid1], [], [])
    conn.commit()

    _test("去重后 saved_ids=1", result["saved_ids"] == 1)

    restored = checkpoint_restore(conn, project)
    conn.commit()
    _test("restore 只含 1 个 chunk", len(restored["chunks"]) == 1)
    conn.close()

    # ── Test 10: 性能基准 ──
    print("\nTest 10: 性能基准")
    conn = _fresh_db()
    project = "test-perf"

    chunk_ids = []
    for i in range(10):
        cid = _insert_test_chunk(conn, project, summary=f"性能测试chunk{i}")
        chunk_ids.append(cid)

    t0 = time.time()
    for _ in range(100):
        checkpoint_dump(conn, project, "sess-perf", chunk_ids[:5], ["h1", "h2"], ["t1"])
        conn.commit()
    dump_ms = (time.time() - t0) / 100 * 1000

    # 创建一个未消费的用于 restore 测试
    checkpoint_dump(conn, project, "sess-perf-last", chunk_ids[:5], ["h1", "h2"], ["t1"])
    conn.commit()

    t0 = time.time()
    for _ in range(100):
        # 需要重新创建因为 restore 会标记 consumed
        # 直接 SQL 重置 consumed 标志
        conn.execute("UPDATE checkpoints SET consumed = 0 WHERE project = ?", (project,))
        conn.commit()
        checkpoint_restore(conn, project)
        conn.commit()
    restore_ms = (time.time() - t0) / 100 * 1000

    _test(f"dump 延迟 {dump_ms:.2f}ms < 5ms", dump_ms < 5)
    _test(f"restore 延迟 {restore_ms:.2f}ms < 5ms", restore_ms < 5)
    conn.close()

    # ── Test 11: sysctl tunable 验证 ──
    print("\nTest 11: sysctl tunable 验证")
    _test("criu.max_checkpoints 已注册", _sysctl("criu.max_checkpoints") == 3)
    _test("criu.max_age_hours 已注册", _sysctl("criu.max_age_hours") == 72)
    _test("criu.max_hit_ids 已注册", _sysctl("criu.max_hit_ids") == 10)
    _test("criu.restore_boost 已注册", _sysctl("criu.restore_boost") == 0.12)

    # ── 汇总 ──
    print(f"\n{'='*50}")
    print(f"CRIU Checkpoint/Restore 测试: {PASS}/{PASS+FAIL} 通过")
    if FAIL:
        print(f"❌ {FAIL} 个测试失败")
        sys.exit(1)
    else:
        print("✅ 全部通过")
