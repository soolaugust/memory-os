#!/usr/bin/env python3
"""
test_stability_aware_stale.py — Stability-aware Fast Stale Penalty 测试（iter473）

覆盖：
  SAS1: high-stability chunk（stability=20）→ 实际惩罚最小（约 ×0.91 = 1 - 0.09）
  SAS2: zero-stability chunk（stability=0）→ 全额惩罚（×0.70 = 1 - 0.30）
  SAS3: mid-stability chunk（stability=10）→ 半额惩罚（约 ×0.85 = 1 - 0.15）
  SAS4: high-stability 的惩罚幅度小于 zero-stability（相对比较验证）
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
    MIN_IMPORTANCE,
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


def _run_stale_scenario(stability_val, label):
    """运行 fast_stale 场景，返回 (initial_imp, new_imp, actual_penalty_rate)。"""
    conn = open_db()
    ensure_schema(conn)

    proj = f"sas_{label}_{uuid.uuid4().hex[:6]}"
    session = f"sess_{uuid.uuid4().hex[:8]}"
    cid = f"sasc_{uuid.uuid4().hex[:10]}"
    summary = f"Java Spring Framework dependency injection {label}"
    initial_importance = 0.60
    _insert_chunk(conn, cid, proj, summary,
                  importance=initial_importance, stability=stability_val)

    # 插入 STALE_CONSEC_THRESHOLD 条跨 session trace
    for i in range(STALE_CONSEC_THRESHOLD):
        _insert_recall_trace(conn, proj, f"sess_hist_{i}", [(cid, summary)])

    # 当前 session trace
    _insert_recall_trace(conn, proj, session, [(cid, summary)])

    # 完全不相关的 reply
    unrelated_reply = ("The Linux kernel buddy allocator manages physical page frames "
                       "using power-of-two sized free lists for fast allocation.")

    stats = run_citation_detection(unrelated_reply, proj, session, conn=conn)
    conn.commit()

    assert stats.get("fast_stale_degraded", 0) >= 1, (
        f"SAS({label}): 应触发 fast_stale, stats={stats}"
    )

    new_importance = conn.execute(
        "SELECT importance FROM memory_chunks WHERE id=?", (cid,)
    ).fetchone()[0]

    conn.execute("DELETE FROM memory_chunks WHERE project=?", (proj,))
    conn.commit()
    conn.close()

    actual_penalty_rate = 1.0 - (new_importance / initial_importance)
    return initial_importance, new_importance, actual_penalty_rate


def test_sas1_high_stability_minimal_penalty():
    """SAS1: stability=20 → actual_penalty ≈ 0.30 × max(0.3, 1-20/20) = 0.30 × 0.30 = 0.09"""
    init_imp, new_imp, actual_rate = _run_stale_scenario(stability_val=20.0, label="high")

    # 预期: actual_penalty = 0.30 × max(0.3, 1 - 20/20) = 0.30 × 0.30 = 0.09
    expected_penalty = STALE_FAST_PENALTY * max(0.3, 1.0 - 20.0 / 20.0)  # = 0.09
    expected_new_imp = max(MIN_IMPORTANCE, init_imp * (1.0 - expected_penalty))

    assert new_imp > init_imp * 0.85, (
        f"SAS1: high-stability 惩罚应很轻, {init_imp:.3f}→{new_imp:.3f}"
    )
    assert abs(new_imp - expected_new_imp) < 0.02, (
        f"SAS1: 预期 {expected_new_imp:.3f}, got {new_imp:.3f}"
    )
    print(f"  SAS1 PASS: stability=20 → imp {init_imp:.3f}→{new_imp:.3f} "
          f"(penalty≈{actual_rate:.1%}, expected≈{expected_penalty:.1%})")


def test_sas2_zero_stability_full_penalty():
    """SAS2: stability=0 → actual_penalty = 0.30 × max(0.3, 1-0/20) = 0.30 × 1.0 = 0.30"""
    init_imp, new_imp, actual_rate = _run_stale_scenario(stability_val=0.0, label="zero")

    # 预期: actual_penalty = 0.30 × max(0.3, 1.0) = 0.30
    expected_penalty = STALE_FAST_PENALTY * max(0.3, 1.0 - 0.0 / 20.0)  # = 0.30
    expected_new_imp = max(MIN_IMPORTANCE, init_imp * (1.0 - expected_penalty))

    assert abs(new_imp - expected_new_imp) < 0.02, (
        f"SAS2: 预期 {expected_new_imp:.3f}, got {new_imp:.3f}"
    )
    print(f"  SAS2 PASS: stability=0 → imp {init_imp:.3f}→{new_imp:.3f} "
          f"(penalty≈{actual_rate:.1%}, expected≈{expected_penalty:.1%})")


def test_sas3_mid_stability_half_penalty():
    """SAS3: stability=10 → actual_penalty = 0.30 × max(0.3, 1-10/20) = 0.30 × 0.5 = 0.15"""
    init_imp, new_imp, actual_rate = _run_stale_scenario(stability_val=10.0, label="mid")

    expected_penalty = STALE_FAST_PENALTY * max(0.3, 1.0 - 10.0 / 20.0)  # = 0.15
    expected_new_imp = max(MIN_IMPORTANCE, init_imp * (1.0 - expected_penalty))

    assert abs(new_imp - expected_new_imp) < 0.02, (
        f"SAS3: 预期 {expected_new_imp:.3f}, got {new_imp:.3f}"
    )
    print(f"  SAS3 PASS: stability=10 → imp {init_imp:.3f}→{new_imp:.3f} "
          f"(penalty≈{actual_rate:.1%}, expected≈{expected_penalty:.1%})")


def test_sas4_high_stability_penalized_less_than_low():
    """SAS4: stability=20 的惩罚幅度显著小于 stability=0（相对比较）。"""
    _, new_high, rate_high = _run_stale_scenario(stability_val=20.0, label="h")
    _, new_low, rate_low = _run_stale_scenario(stability_val=0.0, label="l")

    assert rate_high < rate_low, (
        f"SAS4: 高 stability 的惩罚率应低于低 stability: high={rate_high:.1%} low={rate_low:.1%}"
    )
    assert new_high > new_low, (
        f"SAS4: 高 stability chunk 惩罚后 importance 应高于低 stability: {new_high:.3f} vs {new_low:.3f}"
    )
    print(f"  SAS4 PASS: high_stability penalty={rate_high:.1%} < low_stability penalty={rate_low:.1%}")


if __name__ == "__main__":
    print("Stability-aware Fast Stale Penalty 测试（iter473）")
    print("=" * 60)

    tests = [
        test_sas1_high_stability_minimal_penalty,
        test_sas2_zero_stability_full_penalty,
        test_sas3_mid_stability_half_penalty,
        test_sas4_high_stability_penalized_less_than_low,
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
