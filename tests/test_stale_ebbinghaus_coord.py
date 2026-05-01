#!/usr/bin/env python3
"""
test_stale_ebbinghaus_coord.py — fast_stale + Ebbinghaus 协调测试（iter477）

覆盖：
  SEC1: chunk 已被 Ebbinghaus 衰减（updated_at 近期 + last_accessed 远期）
        → fast_stale 跳过 importance 惩罚（仍计入 fast_stale_degraded）
  SEC2: chunk 未被 Ebbinghaus 衰减（last_accessed 近期）
        → fast_stale 正常执行 importance 惩罚
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
from tools.citation_detector import run_citation_detection, STALE_CONSEC_THRESHOLD


def _insert_chunk_with_timestamps(conn, cid, project, summary,
                                   importance=0.6, stability=1.0,
                                   last_accessed_delta_hours=0,
                                   updated_at_delta_hours=0):
    """
    插入带自定义时间戳的 chunk。
    last_accessed_delta_hours > 0 → 过去多少小时前访问过。
    updated_at_delta_hours > 0 → updated_at 设为多少小时前（模拟最近有写入）。
    """
    now = datetime.now(timezone.utc)
    last_accessed = (now - timedelta(hours=last_accessed_delta_hours)).isoformat()
    updated_at = (now - timedelta(hours=updated_at_delta_hours)).isoformat()
    created_at = now.isoformat()
    conn.execute("""
        INSERT OR REPLACE INTO memory_chunks
        (id, project, source_session, chunk_type, summary, content,
         importance, stability, retrievability, info_class, tags,
         access_count, oom_adj, created_at, updated_at, last_accessed,
         feishu_url, lru_gen, raw_snippet, encode_context, session_type_history)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (cid, project, "test", "decision", summary, summary,
          importance, stability, 0.5, "episodic", json.dumps([]),
          0, 0, created_at, updated_at, last_accessed,
          None, 0, summary[:500], "{}", ""))
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


def test_sec1_ebbinghaus_decayed_skips_stale_importance():
    """
    SEC1: chunk 已被 Ebbinghaus 衰减（updated_at 近期 + last_accessed 远期）
    → fast_stale 跳过 importance 惩罚。
    """
    conn = open_db()
    ensure_schema(conn)

    proj = f"sec1_{uuid.uuid4().hex[:6]}"
    session = f"sess_{uuid.uuid4().hex[:8]}"
    cid = f"sec1c_{uuid.uuid4().hex[:10]}"
    summary = "Java Spring Framework dependency injection configuration"
    initial_importance = 0.60

    # 模拟 Ebbinghaus 已衰减状态：
    # - last_accessed = 48h 前（旧访问）
    # - updated_at = 2h 前（近期有写入，只能是 Ebbinghaus 操作）
    _insert_chunk_with_timestamps(
        conn, cid, proj, summary,
        importance=initial_importance,
        last_accessed_delta_hours=48,   # 48h 前访问
        updated_at_delta_hours=2,       # 2h 前被 Ebbinghaus 更新
    )

    # 插入 STALE_CONSEC_THRESHOLD 条 trace
    for i in range(STALE_CONSEC_THRESHOLD):
        _insert_recall_trace(conn, proj, f"sess_hist_{i}", [(cid, summary)])
    _insert_recall_trace(conn, proj, session, [(cid, summary)])

    unrelated_reply = ("The Linux kernel buddy allocator manages physical page frames "
                       "using power-of-two sized free lists for fast allocation.")

    stats = run_citation_detection(unrelated_reply, proj, session, conn=conn)
    conn.commit()

    # fast_stale 应触发（计入统计），但 importance 不应下降
    assert stats.get("fast_stale_degraded", 0) >= 1, (
        f"SEC1: 应计入 fast_stale_degraded, stats={stats}"
    )

    new_importance = conn.execute(
        "SELECT importance FROM memory_chunks WHERE id=?", (cid,)
    ).fetchone()[0]

    # Ebbinghaus 协调：importance 不应被 fast_stale 进一步惩罚
    assert new_importance >= initial_importance - 0.02, (
        f"SEC1: Ebbinghaus 协调应跳过 importance 惩罚, "
        f"{initial_importance:.3f}→{new_importance:.3f}"
    )

    conn.execute("DELETE FROM memory_chunks WHERE project=?", (proj,))
    conn.commit()
    conn.close()
    print(f"  SEC1 PASS: Ebbinghaus-decayed chunk skips stale penalty: "
          f"{initial_importance:.3f}→{new_importance:.3f}")


def test_sec2_recent_access_applies_stale_penalty():
    """
    SEC2: chunk 未被 Ebbinghaus 衰减（last_accessed 近期）
    → fast_stale 正常执行 importance 惩罚。
    """
    conn = open_db()
    ensure_schema(conn)

    proj = f"sec2_{uuid.uuid4().hex[:6]}"
    session = f"sess_{uuid.uuid4().hex[:8]}"
    cid = f"sec2c_{uuid.uuid4().hex[:10]}"
    summary = "Java Spring Framework dependency injection configuration"
    initial_importance = 0.60

    # 近期访问过（不是 Ebbinghaus 衰减状态）
    _insert_chunk_with_timestamps(
        conn, cid, proj, summary,
        importance=initial_importance,
        last_accessed_delta_hours=1,   # 1h 前访问（近期）
        updated_at_delta_hours=1,      # updated_at 也是近期
    )

    for i in range(STALE_CONSEC_THRESHOLD):
        _insert_recall_trace(conn, proj, f"sess_hist_{i}", [(cid, summary)])
    _insert_recall_trace(conn, proj, session, [(cid, summary)])

    unrelated_reply = ("The Linux kernel buddy allocator manages physical page frames "
                       "using power-of-two sized free lists for fast allocation.")

    stats = run_citation_detection(unrelated_reply, proj, session, conn=conn)
    conn.commit()

    assert stats.get("fast_stale_degraded", 0) >= 1, (
        f"SEC2: 应触发 fast_stale, stats={stats}"
    )

    new_importance = conn.execute(
        "SELECT importance FROM memory_chunks WHERE id=?", (cid,)
    ).fetchone()[0]

    # 近期访问，无 Ebbinghaus 保护 → 正常惩罚
    assert new_importance < initial_importance - 0.05, (
        f"SEC2: 无 Ebbinghaus 保护时应正常惩罚, "
        f"{initial_importance:.3f}→{new_importance:.3f}"
    )

    conn.execute("DELETE FROM memory_chunks WHERE project=?", (proj,))
    conn.commit()
    conn.close()
    print(f"  SEC2 PASS: non-Ebbinghaus chunk applies stale penalty: "
          f"{initial_importance:.3f}→{new_importance:.3f}")


if __name__ == "__main__":
    print("fast_stale + Ebbinghaus 协调测试（iter477）")
    print("=" * 60)

    tests = [
        test_sec1_ebbinghaus_decayed_skips_stale_importance,
        test_sec2_recent_access_applies_stale_penalty,
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
