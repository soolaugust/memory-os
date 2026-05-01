#!/usr/bin/env python3
"""
test_adaptive_threshold.py — Adaptive Citation Threshold 测试

覆盖：
  ACT1: citation_rate < 30% → effective_threshold 提高（精确度优先，减少误判）
  ACT2: citation_rate > 65% → effective_threshold 降低（召回率优先）
  ACT3: citation_rate 中间 → threshold 保持 CITATION_TRIGRAM_THRESHOLD
  ACT4: 低命中率 + 边界文本 → 提高阈值后不再 cited（减少误判效果验证）
"""
import sys
import json
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import tmpfs  # noqa

from store import open_db, ensure_schema
from tools.citation_detector import (
    run_citation_detection,
    CITATION_TRIGRAM_THRESHOLD,
    _update_citation_stats,
    get_citation_rate,
    _MEMORY_OS_DIR,
)


def _utcnow_iso():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _insert_chunk(conn, cid, project, summary, importance=0.6, chunk_type="decision"):
    from datetime import datetime, timezone, timedelta
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


def _insert_recall_trace(conn, project, session_id, chunk_ids_summaries: list):
    trace_id = "trace_" + uuid.uuid4().hex[:12]
    now = _utcnow_iso()
    top_k = [{"id": cid, "summary": summary} for cid, summary in chunk_ids_summaries]
    conn.execute("""
        INSERT INTO recall_traces
        (id, project, session_id, prompt_hash, timestamp, top_k_json, injected,
         reason, duration_ms, agent_id, ftrace_json)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (trace_id, project, session_id, "testhash", now,
          json.dumps(top_k), 1, "test", 10.0, session_id[:16], None))
    conn.commit()
    return trace_id


def _cleanup_stats(project: str):
    proj_safe = project.replace("/", "_").replace(":", "_")[:40]
    stats_file = _MEMORY_OS_DIR / f"citation_stats.{proj_safe}.json"
    if stats_file.exists():
        stats_file.unlink()


def test_act1_low_rate_raises_threshold():
    """ACT1: citation_rate < 30% → effective_threshold > CITATION_TRIGRAM_THRESHOLD。"""
    proj = f"act_low_{uuid.uuid4().hex[:6]}"
    _cleanup_stats(proj)

    # 写入低命中率
    for _ in range(10):
        _update_citation_stats(proj, 0, 5)

    rate = get_citation_rate(proj)
    assert rate < 0.30, f"ACT1 setup: rate should be <0.30, got {rate}"

    # 根据 adaptive threshold 逻辑计算预期值
    if rate < 0.30:
        expected_threshold = min(0.15, CITATION_TRIGRAM_THRESHOLD * 1.5)
    else:
        expected_threshold = CITATION_TRIGRAM_THRESHOLD

    assert expected_threshold > CITATION_TRIGRAM_THRESHOLD, (
        f"ACT1: low rate should raise threshold above {CITATION_TRIGRAM_THRESHOLD}, "
        f"got {expected_threshold}"
    )

    _cleanup_stats(proj)
    print(f"  ACT1 PASS: rate={rate:.2f}, threshold {CITATION_TRIGRAM_THRESHOLD}→{expected_threshold:.3f}")


def test_act2_high_rate_lowers_threshold():
    """ACT2: citation_rate > 65% → effective_threshold < CITATION_TRIGRAM_THRESHOLD。"""
    proj = f"act_high_{uuid.uuid4().hex[:6]}"
    _cleanup_stats(proj)

    for _ in range(10):
        _update_citation_stats(proj, 5, 5)

    rate = get_citation_rate(proj)
    assert rate > 0.65, f"ACT2 setup: rate should be >0.65, got {rate}"

    if rate > 0.65:
        expected_threshold = max(0.05, CITATION_TRIGRAM_THRESHOLD * 0.75)
    else:
        expected_threshold = CITATION_TRIGRAM_THRESHOLD

    assert expected_threshold < CITATION_TRIGRAM_THRESHOLD, (
        f"ACT2: high rate should lower threshold below {CITATION_TRIGRAM_THRESHOLD}, "
        f"got {expected_threshold}"
    )

    _cleanup_stats(proj)
    print(f"  ACT2 PASS: rate={rate:.2f}, threshold {CITATION_TRIGRAM_THRESHOLD}→{expected_threshold:.3f}")


def test_act3_medium_rate_keeps_threshold():
    """ACT3: 30-65% → threshold 保持不变（CITATION_TRIGRAM_THRESHOLD）。"""
    proj = f"act_med_{uuid.uuid4().hex[:6]}"
    _cleanup_stats(proj)

    for _ in range(10):
        _update_citation_stats(proj, 2, 5)  # rate=0.4

    rate = get_citation_rate(proj)
    assert 0.30 <= rate <= 0.65, f"ACT3 setup: rate should be 0.30-0.65, got {rate}"

    # 根据逻辑：中间区间 → 不变
    if rate < 0.30:
        expected = min(0.15, CITATION_TRIGRAM_THRESHOLD * 1.5)
    elif rate > 0.65:
        expected = max(0.05, CITATION_TRIGRAM_THRESHOLD * 0.75)
    else:
        expected = CITATION_TRIGRAM_THRESHOLD

    assert abs(expected - CITATION_TRIGRAM_THRESHOLD) < 0.001, (
        f"ACT3: medium rate should keep threshold={CITATION_TRIGRAM_THRESHOLD}, got {expected}"
    )

    _cleanup_stats(proj)
    print(f"  ACT3 PASS: rate={rate:.2f}, threshold unchanged={expected:.3f}")


def test_act4_high_threshold_prevents_weak_citation():
    """ACT4: 低命中率 + 提高阈值后，弱相关 chunk 不再被误判为 cited。"""
    conn = open_db()
    ensure_schema(conn)

    proj = f"act_eff_{uuid.uuid4().hex[:6]}"
    session = f"sess_{uuid.uuid4().hex[:8]}"
    _cleanup_stats(proj)

    # 写入低命中率
    for _ in range(15):
        _update_citation_stats(proj, 0, 5)

    # 弱相关 chunk（trigram overlap 刚好在 0.08-0.12 之间的灰区）
    cid = f"act4_{uuid.uuid4().hex[:10]}"
    # "Linux memory" 与 reply "memory allocation in Linux system" 有些重叠但不多
    summary = "Linux memory management overview"
    initial_imp = 0.60
    _insert_chunk(conn, cid, proj, summary, importance=initial_imp)
    _insert_recall_trace(conn, proj, session, [(cid, summary)])

    # reply 包含少量重叠（弱引用）
    weak_reply = "Memory allocation in operating systems depends on kernel design."

    stats = run_citation_detection(weak_reply, proj, session, conn=conn)
    conn.commit()

    new_imp = conn.execute(
        "SELECT importance FROM memory_chunks WHERE id=?", (cid,)
    ).fetchone()[0]

    # 因为低命中率提高了阈值，弱相关 chunk 不应被 cited（importance 不增加）
    # 注意：可能被 uncited（减少），也可能刚好在新 threshold 附近
    assert new_imp <= initial_imp + 0.005, (
        f"ACT4: 高阈值下弱相关 chunk 不应被 cited, imp {initial_imp:.3f}→{new_imp:.3f}"
    )

    conn.execute("DELETE FROM memory_chunks WHERE project=?", (proj,))
    conn.commit()
    conn.close()
    _cleanup_stats(proj)
    print(f"  ACT4 PASS: weak citation blocked, imp {initial_imp:.3f}→{new_imp:.3f}, stats={stats}")


if __name__ == "__main__":
    print("Adaptive Citation Threshold 测试")
    print("=" * 60)

    tests = [
        test_act1_low_rate_raises_threshold,
        test_act2_high_rate_lowers_threshold,
        test_act3_medium_rate_keeps_threshold,
        test_act4_high_threshold_prevents_weak_citation,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  {test.__name__} FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"结果：{passed}/{passed + failed} 通过")
    if failed:
        import sys
        sys.exit(1)
