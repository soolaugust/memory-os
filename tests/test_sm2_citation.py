#!/usr/bin/env python3
"""
test_sm2_citation.py — Citation + SM-2 联动测试

覆盖：
  SM1: cited chunk → recall_quality=5 → stability × 1.2
  SM2: fast_stale_degraded chunk → recall_quality=1 → stability × 0.8
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
)


def _insert_chunk(conn, cid, project, summary, importance=0.6, stability=1.0,
                  chunk_type="decision"):
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
          importance, stability, 0.5, "episodic", json.dumps([]),
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


def test_sm1_cited_chunk_stability_increases():
    """SM1: cited chunk 触发 recall_quality=5 → stability × 1.2。"""
    conn = open_db()
    ensure_schema(conn)

    proj = f"sm1_{uuid.uuid4().hex[:6]}"
    session = f"sess_{uuid.uuid4().hex[:8]}"
    cid = f"sm1c_{uuid.uuid4().hex[:10]}"
    summary = "Linux page fault PTE validate allocate physical page buddy allocator"
    initial_stability = 5.0
    _insert_chunk(conn, cid, proj, summary, stability=initial_stability)
    _insert_recall_trace(conn, proj, session, [(cid, summary)])

    # reply 明确引用 chunk 内容
    reply = ("The Linux page fault handler validates PTE entries and allocates "
             "physical pages through the buddy allocator.")

    stats = run_citation_detection(reply, proj, session, conn=conn)
    conn.commit()

    assert stats["cited"] >= 1, f"SM1: 应检测到引用, stats={stats}"

    new_stability = conn.execute(
        "SELECT stability FROM memory_chunks WHERE id=?", (cid,)
    ).fetchone()[0]

    # SM-2 quality=5: S_new = S_old × 1.2
    expected_stability = initial_stability * 1.2
    assert new_stability > initial_stability, (
        f"SM1: cited 后 stability 应增加, {initial_stability:.3f}→{new_stability:.3f}"
    )
    assert abs(new_stability - expected_stability) < 0.5, (
        f"SM1: stability 应约为 {expected_stability:.3f}, got {new_stability:.3f}"
    )

    conn.execute("DELETE FROM memory_chunks WHERE project=?", (proj,))
    conn.commit()
    conn.close()
    print(f"  SM1 PASS: cited stability {initial_stability:.2f}→{new_stability:.2f}"
          f" (expected~{expected_stability:.2f})")


def test_sm2_stale_chunk_stability_decreases():
    """SM2: fast_stale_degraded chunk 触发 recall_quality=1 → stability × 0.8。"""
    conn = open_db()
    ensure_schema(conn)

    proj = f"sm2_{uuid.uuid4().hex[:6]}"
    session = f"sess_{uuid.uuid4().hex[:8]}"
    cid = f"sm2s_{uuid.uuid4().hex[:10]}"
    summary = "Java Spring Framework dependency injection configuration beans"
    initial_stability = 5.0
    _insert_chunk(conn, cid, proj, summary, stability=initial_stability)

    # 插入 STALE_CONSEC_THRESHOLD 条跨 session trace（触发 fast stale）
    for i in range(STALE_CONSEC_THRESHOLD):
        _insert_recall_trace(conn, proj, f"sess_hist_{i}", [(cid, summary)])

    # 当前 session 也有 trace
    _insert_recall_trace(conn, proj, session, [(cid, summary)])

    # reply 完全不相关
    unrelated_reply = ("The Linux kernel buddy allocator manages physical page frames "
                       "using power-of-two sized free lists for fast allocation.")

    stats = run_citation_detection(unrelated_reply, proj, session, conn=conn)
    conn.commit()

    assert stats.get("fast_stale_degraded", 0) >= 1, (
        f"SM2: 应触发 fast stale, stats={stats}"
    )

    new_stability = conn.execute(
        "SELECT stability FROM memory_chunks WHERE id=?", (cid,)
    ).fetchone()[0]

    # SM-2 quality=1: S_new = S_old × max(0.7, 1 + 0.1×(1-3)) = S_old × 0.8
    expected_stability = initial_stability * 0.8
    assert new_stability < initial_stability, (
        f"SM2: fast_stale 后 stability 应减少, {initial_stability:.3f}→{new_stability:.3f}"
    )
    assert abs(new_stability - expected_stability) < 0.5, (
        f"SM2: stability 应约为 {expected_stability:.3f}, got {new_stability:.3f}"
    )

    conn.execute("DELETE FROM memory_chunks WHERE project=?", (proj,))
    conn.commit()
    conn.close()
    print(f"  SM2 PASS: stale stability {initial_stability:.2f}→{new_stability:.2f}"
          f" (expected~{expected_stability:.2f})")


def test_sm3_uncited_chunk_stability_slight_decrease():
    """SM3: uncited chunk 触发 recall_quality=2 → stability × 0.9（轻微减弱）。"""
    conn = open_db()
    ensure_schema(conn)

    proj = f"sm3_{uuid.uuid4().hex[:6]}"
    session = f"sess_{uuid.uuid4().hex[:8]}"
    cid = f"sm3u_{uuid.uuid4().hex[:10]}"
    summary = "Python asyncio event loop scheduler coroutine"
    initial_stability = 5.0
    _insert_chunk(conn, cid, proj, summary, stability=initial_stability)

    # 只有 1 条 trace（不触发 fast stale）
    _insert_recall_trace(conn, proj, session, [(cid, summary)])

    # reply 完全不相关
    unrelated_reply = ("The Linux kernel buddy allocator manages physical page frames "
                       "using power-of-two sized free lists for fast allocation.")

    stats = run_citation_detection(unrelated_reply, proj, session, conn=conn)
    conn.commit()

    # 确认没有触发 fast stale（只有 1 条 trace）
    assert stats.get("fast_stale_degraded", 0) == 0, (
        f"SM3: 不应触发 fast stale（trace 不足）, stats={stats}"
    )
    assert stats["uncited"] >= 1, f"SM3: 应检测到 uncited, stats={stats}"

    new_stability = conn.execute(
        "SELECT stability FROM memory_chunks WHERE id=?", (cid,)
    ).fetchone()[0]

    # SM-2 quality=2: S_new = S_old × max(0.7, 1+0.1×(2-3)) = S_old × 0.9
    expected_stability = initial_stability * 0.9
    assert new_stability < initial_stability, (
        f"SM3: uncited 后 stability 应轻微减少, {initial_stability:.3f}→{new_stability:.3f}"
    )
    assert abs(new_stability - expected_stability) < 0.3, (
        f"SM3: stability 应约为 {expected_stability:.3f}, got {new_stability:.3f}"
    )

    conn.execute("DELETE FROM memory_chunks WHERE project=?", (proj,))
    conn.commit()
    conn.close()
    print(f"  SM3 PASS: uncited stability {initial_stability:.2f}→{new_stability:.2f}"
          f" (expected~{expected_stability:.2f})")


if __name__ == "__main__":
    print("Citation + SM-2 联动测试")
    print("=" * 60)

    tests = [test_sm1_cited_chunk_stability_increases,
             test_sm2_stale_chunk_stability_decreases,
             test_sm3_uncited_chunk_stability_slight_decrease]

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
        sys.exit(1)
