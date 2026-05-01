#!/usr/bin/env python3
"""
test_confidence_citation.py — confidence_score 与 citation 联动测试（iter476）

覆盖：
  CC1: cited chunk → confidence_score 微增（+0.01）
  CC2: uncited chunk → confidence_score 轻微减少（-0.005）
  CC3: confidence_score 上界保护（不超过 1.0）
  CC4: confidence_score 下界保护（不低于 0.10）
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
    CITED_CONFIDENCE_DELTA,
    UNCITED_CONFIDENCE_DELTA,
    MIN_CONFIDENCE,
    MAX_CONFIDENCE,
)


def _insert_chunk(conn, cid, project, summary, importance=0.6, stability=1.0,
                  confidence_score=0.7, chunk_type="decision"):
    now = datetime.now(timezone.utc)
    la = (now - timedelta(minutes=5)).isoformat()
    now_iso = now.isoformat()
    conn.execute("""
        INSERT OR REPLACE INTO memory_chunks
        (id, project, source_session, chunk_type, summary, content,
         importance, stability, retrievability, info_class, tags,
         access_count, oom_adj, created_at, updated_at, last_accessed,
         feishu_url, lru_gen, raw_snippet, encode_context, session_type_history,
         confidence_score)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (cid, project, "test", chunk_type, summary, summary,
          importance, stability, 0.5, "episodic", json.dumps([]),
          0, 0, now_iso, now_iso, la, None, 0, summary[:500], "{}", "",
          confidence_score))
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
    """, (trace_id, project, session_id, "hash", now,
          json.dumps(top_k), 1, "test", 10.0, session_id[:16], None))
    conn.commit()
    return trace_id


def _get_confidence(conn, cid):
    row = conn.execute(
        "SELECT confidence_score FROM memory_chunks WHERE id=?", (cid,)
    ).fetchone()
    return float(row[0]) if row and row[0] is not None else 0.7


def test_cc1_cited_chunk_confidence_increases():
    """CC1: cited chunk → confidence_score 微增。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"cc1_{uuid.uuid4().hex[:6]}"
    session = f"sess_{uuid.uuid4().hex[:8]}"
    cid = f"cc1c_{uuid.uuid4().hex[:10]}"
    summary = "Linux page fault PTE validate allocate physical page buddy allocator"
    initial_conf = 0.70
    _insert_chunk(conn, cid, proj, summary, confidence_score=initial_conf)
    _insert_recall_trace(conn, proj, session, [(cid, summary)])

    reply = ("The Linux page fault handler validates PTE entries and allocates "
             "physical pages through the buddy allocator.")

    stats = run_citation_detection(reply, proj, session, conn=conn)
    conn.commit()

    assert stats["cited"] >= 1, f"CC1: 应检测到引用"
    new_conf = _get_confidence(conn, cid)
    expected_conf = initial_conf + CITED_CONFIDENCE_DELTA

    assert new_conf > initial_conf, (
        f"CC1: cited 后 confidence_score 应增加, {initial_conf:.3f}→{new_conf:.3f}"
    )
    assert abs(new_conf - expected_conf) < 0.005, (
        f"CC1: 预期 {expected_conf:.3f}, got {new_conf:.3f}"
    )

    conn.execute("DELETE FROM memory_chunks WHERE project=?", (proj,))
    conn.commit()
    conn.close()
    print(f"  CC1 PASS: cited confidence {initial_conf:.3f}→{new_conf:.3f} "
          f"(expected {expected_conf:.3f})")


def test_cc2_uncited_chunk_confidence_decreases():
    """CC2: uncited chunk → confidence_score 轻微减少。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"cc2_{uuid.uuid4().hex[:6]}"
    session = f"sess_{uuid.uuid4().hex[:8]}"
    cid = f"cc2c_{uuid.uuid4().hex[:10]}"
    summary = "Python asyncio event loop coroutine scheduler"
    initial_conf = 0.70
    _insert_chunk(conn, cid, proj, summary, confidence_score=initial_conf)
    _insert_recall_trace(conn, proj, session, [(cid, summary)])

    # 完全不相关的 reply
    reply = ("The Linux kernel buddy allocator manages physical page frames "
             "using power-of-two sized free lists.")

    stats = run_citation_detection(reply, proj, session, conn=conn)
    conn.commit()

    assert stats["uncited"] >= 1, f"CC2: 应检测到 uncited"
    new_conf = _get_confidence(conn, cid)
    expected_conf = initial_conf + UNCITED_CONFIDENCE_DELTA  # 0.695

    assert new_conf < initial_conf, (
        f"CC2: uncited 后 confidence_score 应轻微减少, {initial_conf:.3f}→{new_conf:.3f}"
    )
    assert abs(new_conf - expected_conf) < 0.005, (
        f"CC2: 预期 {expected_conf:.3f}, got {new_conf:.3f}"
    )

    conn.execute("DELETE FROM memory_chunks WHERE project=?", (proj,))
    conn.commit()
    conn.close()
    print(f"  CC2 PASS: uncited confidence {initial_conf:.3f}→{new_conf:.3f} "
          f"(expected {expected_conf:.3f})")


def test_cc3_confidence_upper_bound():
    """CC3: confidence_score 不超过上界 1.0。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"cc3_{uuid.uuid4().hex[:6]}"
    session = f"sess_{uuid.uuid4().hex[:8]}"
    cid = f"cc3c_{uuid.uuid4().hex[:10]}"
    summary = "Linux page fault PTE validate allocate physical page"
    _insert_chunk(conn, cid, proj, summary, confidence_score=0.999)  # 接近上界
    _insert_recall_trace(conn, proj, session, [(cid, summary)])

    reply = "The Linux page fault handler validates PTE entries and allocates physical pages."
    run_citation_detection(reply, proj, session, conn=conn)
    conn.commit()

    new_conf = _get_confidence(conn, cid)
    assert new_conf <= MAX_CONFIDENCE, f"CC3: confidence_score 超过上界 {MAX_CONFIDENCE}, got {new_conf}"

    conn.execute("DELETE FROM memory_chunks WHERE project=?", (proj,))
    conn.commit()
    conn.close()
    print(f"  CC3 PASS: confidence upper bound respected (got {new_conf:.4f})")


def test_cc4_confidence_lower_bound():
    """CC4: confidence_score 不低于下界 0.10。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"cc4_{uuid.uuid4().hex[:6]}"
    session = f"sess_{uuid.uuid4().hex[:8]}"
    cid = f"cc4c_{uuid.uuid4().hex[:10]}"
    summary = "Java Spring beans configuration injection"
    _insert_chunk(conn, cid, proj, summary, confidence_score=0.101)  # 接近下界
    _insert_recall_trace(conn, proj, session, [(cid, summary)])

    reply = "The Linux buddy allocator manages physical page frames."
    run_citation_detection(reply, proj, session, conn=conn)
    conn.commit()

    new_conf = _get_confidence(conn, cid)
    assert new_conf >= MIN_CONFIDENCE, f"CC4: confidence_score 低于下界 {MIN_CONFIDENCE}, got {new_conf}"

    conn.execute("DELETE FROM memory_chunks WHERE project=?", (proj,))
    conn.commit()
    conn.close()
    print(f"  CC4 PASS: confidence lower bound respected (got {new_conf:.4f})")


if __name__ == "__main__":
    print("confidence_score 与 citation 联动测试（iter476）")
    print("=" * 60)

    tests = [
        test_cc1_cited_chunk_confidence_increases,
        test_cc2_uncited_chunk_confidence_decreases,
        test_cc3_confidence_upper_bound,
        test_cc4_confidence_lower_bound,
    ]

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
