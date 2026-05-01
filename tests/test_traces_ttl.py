#!/usr/bin/env python3
"""
test_traces_ttl.py — recall_traces TTL 清理测试

覆盖：
  TTL1: 超过 30 天的 trace → damon_scan 后被删除
  TTL2: 30 天内的 trace → damon_scan 后保留
"""
import sys
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import tmpfs  # noqa

from store import open_db, ensure_schema
from store_mm import damon_scan


def _insert_chunk(conn, cid, project):
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT OR REPLACE INTO memory_chunks
        (id, project, source_session, chunk_type, summary, content,
         importance, stability, retrievability, info_class, tags,
         access_count, oom_adj, created_at, updated_at, last_accessed,
         feishu_url, lru_gen, raw_snippet, encode_context, session_type_history)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (cid, project, "test", "decision", "test chunk", "test chunk",
          0.6, 1.0, 0.5, "episodic", json.dumps([]),
          5, 0, now, now, now, None, 0, "test chunk", "{}", ""))
    conn.commit()


def _insert_trace(conn, project, session_id, days_ago):
    """插入指定天数前的 recall_trace。"""
    trace_id = "trace_" + uuid.uuid4().hex[:12]
    ts = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    conn.execute("""
        INSERT INTO recall_traces
        (id, project, session_id, prompt_hash, timestamp, top_k_json, injected,
         reason, duration_ms, agent_id, ftrace_json)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (trace_id, project, session_id, "hash", ts,
          json.dumps([]), 0, "test", 10.0, "agent", None))
    conn.commit()
    return trace_id


def test_ttl1_old_traces_deleted():
    """TTL1: 超过 30 天的 trace 在 damon_scan 后被删除。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"ttl1_{uuid.uuid4().hex[:6]}"
    cid = f"ttl1c_{uuid.uuid4().hex[:10]}"
    _insert_chunk(conn, cid, proj)

    # 插入 40 天前的旧 trace
    old_trace_id = _insert_trace(conn, proj, "old_sess", days_ago=40)

    # 确认插入成功
    count_before = conn.execute(
        "SELECT COUNT(*) FROM recall_traces WHERE project=? AND id=?",
        (proj, old_trace_id)
    ).fetchone()[0]
    assert count_before == 1, f"TTL1: trace 插入失败"

    result = damon_scan(conn, proj)
    conn.commit()

    # 旧 trace 应已被删除
    count_after = conn.execute(
        "SELECT COUNT(*) FROM recall_traces WHERE project=? AND id=?",
        (proj, old_trace_id)
    ).fetchone()[0]
    assert count_after == 0, f"TTL1: 超过 30 天的 trace 应被删除，但仍存在"

    traces_cleaned = result["actions"].get("traces_cleaned", 0)
    assert traces_cleaned >= 1, f"TTL1: actions.traces_cleaned 应 >= 1, got {traces_cleaned}"

    conn.execute("DELETE FROM memory_chunks WHERE project=?", (proj,))
    conn.commit()
    conn.close()
    print(f"  TTL1 PASS: old trace deleted, traces_cleaned={traces_cleaned}")


def test_ttl2_recent_traces_kept():
    """TTL2: 30 天内的 trace 在 damon_scan 后保留。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"ttl2_{uuid.uuid4().hex[:6]}"
    cid = f"ttl2c_{uuid.uuid4().hex[:10]}"
    _insert_chunk(conn, cid, proj)

    # 插入 5 天前的近期 trace
    recent_trace_id = _insert_trace(conn, proj, "new_sess", days_ago=5)

    damon_scan(conn, proj)
    conn.commit()

    # 近期 trace 应保留
    count_after = conn.execute(
        "SELECT COUNT(*) FROM recall_traces WHERE project=? AND id=?",
        (proj, recent_trace_id)
    ).fetchone()[0]
    assert count_after == 1, f"TTL2: 近期 trace 不应被删除"

    conn.execute("DELETE FROM memory_chunks WHERE project=?", (proj,))
    conn.execute("DELETE FROM recall_traces WHERE project=?", (proj,))
    conn.commit()
    conn.close()
    print("  TTL2 PASS: recent trace preserved")


if __name__ == "__main__":
    print("recall_traces TTL 清理测试")
    print("=" * 60)

    tests = [test_ttl1_old_traces_deleted, test_ttl2_recent_traces_kept]

    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  {t.__name__} FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n结果：{passed}/{passed+failed} 通过")
    if failed:
        import sys
        sys.exit(1)
