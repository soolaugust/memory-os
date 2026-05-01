#!/usr/bin/env python3
"""
test_degradation_priority.py — 降级机制优先级协调测试

覆盖：
  DG1: fast stale 触发 → 跳过 uncited 微减（避免双重惩罚）
  DG2: 未触发 fast stale → uncited 微减正常执行
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
from tools.citation_detector import (
    run_citation_detection,
    STALE_CONSEC_THRESHOLD,
    STALE_FAST_PENALTY,
    UNCITED_IMPORTANCE_DELTA,
    MIN_IMPORTANCE,
)


def _insert_chunk(conn, cid, project, summary, importance=0.6, chunk_type="decision"):
    now = datetime.now(timezone.utc)
    la = (now - timedelta(minutes=5)).isoformat()
    now_iso = now.isoformat()
    conn.execute("""
        INSERT OR REPLACE INTO memory_chunks
        (id, project, source_session, chunk_type, summary, content,
         importance, stability, retrievability, info_class, tags,
         access_count, oom_adj, created_at, updated_at, last_accessed,
         feishu_url, lru_gen, raw_snippet, encode_context, session_type_history)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (cid, project, "test", chunk_type, summary, summary,
          importance, 10.0, 0.5, "episodic", json.dumps([]),
          0, 0, now_iso, now_iso, la, None, 0, summary[:500], "{}", ""))
    conn.commit()


def _insert_recall_trace(conn, project, session_id, chunk_ids_summaries):
    trace_id = "trace_" + uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc).isoformat()
    top_k = [{"id": cid, "summary": s} for cid, s in chunk_ids_summaries]
    conn.execute("""
        INSERT INTO recall_traces
        (id, project, session_id, prompt_hash, timestamp, top_k_json, injected,
         reason, duration_ms, agent_id, ftrace_json)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (trace_id, project, session_id, "testhash", now,
          json.dumps(top_k), 1, "test", 10.0, session_id[:16], None))
    conn.commit()
    return trace_id


def test_dg1_fast_stale_prevents_double_penalty():
    """DG1: fast stale 已降级的 chunk 不再受 uncited 微减双重惩罚。"""
    conn = open_db()
    ensure_schema(conn)

    proj = f"dg_stale_{uuid.uuid4().hex[:6]}"
    session = f"sess_{uuid.uuid4().hex[:8]}"
    cid = f"dg1_{uuid.uuid4().hex[:12]}"
    summary = "Java Spring Framework dependency injection configuration"
    initial_imp = 0.60
    _insert_chunk(conn, cid, proj, summary, importance=initial_imp)

    # 插入 STALE_CONSEC_THRESHOLD 条跨 session 的 recall_trace（触发 fast stale）
    for i in range(STALE_CONSEC_THRESHOLD):
        _insert_recall_trace(conn, proj, f"sess_hist_{i}", [(cid, summary)])

    # 当前 session 也有一条 trace（确保 uncited 路径走到）
    _insert_recall_trace(conn, proj, session, [(cid, summary)])

    # 回复文本完全不相关
    unrelated_reply = ("The Linux kernel uses buddy allocator for physical page "
                       "allocation and slab allocator for kernel objects.")

    stats = run_citation_detection(unrelated_reply, proj, session, conn=conn)
    conn.commit()

    new_imp = conn.execute(
        "SELECT importance FROM memory_chunks WHERE id=?", (cid,)
    ).fetchone()[0]

    # fast stale 应触发（有 STALE_CONSEC_THRESHOLD 条历史 trace）
    assert stats.get("fast_stale_degraded", 0) >= 1, (
        f"DG1: fast stale 应触发, stats={stats}"
    )

    # 期望：仅 fast stale penalty（×(1-0.30)），不叠加 uncited -0.01
    expected_stale_only = initial_imp * (1.0 - STALE_FAST_PENALTY)
    expected_double = expected_stale_only + UNCITED_IMPORTANCE_DELTA  # 更低（双重惩罚）

    # new_imp 应接近 expected_stale_only，不应接近 expected_double
    assert new_imp >= expected_stale_only - 0.005, (
        f"DG1: importance 不应低于 fast stale only ({expected_stale_only:.3f})，"
        f"got {new_imp:.3f}（可能发生了双重惩罚）"
    )

    conn.execute("DELETE FROM memory_chunks WHERE project=?", (proj,))
    conn.commit()
    conn.close()
    print(f"  DG1 PASS: fast stale only imp={new_imp:.3f}, "
          f"expected~{expected_stale_only:.3f}, double_penalty_floor={expected_double:.3f}")


def test_dg2_no_stale_applies_uncited_delta():
    """DG2: 未触发 fast stale → uncited 微减正常执行。"""
    conn = open_db()
    ensure_schema(conn)

    proj = f"dg_normal_{uuid.uuid4().hex[:6]}"
    session = f"sess_{uuid.uuid4().hex[:8]}"
    cid = f"dg2_{uuid.uuid4().hex[:12]}"
    summary = "Python asyncio event loop coroutine scheduling"
    initial_imp = 0.60
    _insert_chunk(conn, cid, proj, summary, importance=initial_imp)

    # 只有 1 条 trace（不够触发 fast stale，STALE_CONSEC_THRESHOLD=3）
    _insert_recall_trace(conn, proj, session, [(cid, summary)])

    unrelated_reply = ("The Linux kernel uses buddy allocator for physical page "
                       "allocation and slab allocator for kernel objects.")

    stats = run_citation_detection(unrelated_reply, proj, session, conn=conn)
    conn.commit()

    new_imp = conn.execute(
        "SELECT importance FROM memory_chunks WHERE id=?", (cid,)
    ).fetchone()[0]

    # fast stale 不触发（只有 1 条 trace）
    assert stats.get("fast_stale_degraded", 0) == 0, (
        f"DG2: 不应触发 fast stale（trace 不足）, stats={stats}"
    )

    # uncited 微减应正常执行
    assert stats["uncited"] >= 1, f"DG2: 应有 uncited chunk, stats={stats}"
    assert new_imp < initial_imp, (
        f"DG2: uncited 微减应降低 importance, {initial_imp:.3f}→{new_imp:.3f}"
    )

    conn.execute("DELETE FROM memory_chunks WHERE project=?", (proj,))
    conn.commit()
    conn.close()
    print(f"  DG2 PASS: no stale, uncited delta applied, imp {initial_imp:.3f}→{new_imp:.3f}")


if __name__ == "__main__":
    print("降级机制优先级协调测试")
    print("=" * 60)

    tests = [test_dg1_fast_stale_prevents_double_penalty,
             test_dg2_no_stale_applies_uncited_delta]

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
