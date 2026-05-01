#!/usr/bin/env python3
"""
test_citation_detector.py — Citation Detection + Importance Online 更新测试

覆盖：
  CD1: 回复文本引用了 chunk summary → importance 微增
  CD2: 回复文本未引用 chunk → importance 微减
  CD3: 无 recall_trace 记录 → 跳过，不报错
  CD4: importance 边界保护：[MIN_IMPORTANCE, MAX_IMPORTANCE]
  CD5: semantic chunk 级联更新（cited source → __semantic__ importance 增）
  CD6: 空回复文本 → 跳过，不更新
  CD7: 多个 chunk，混合引用情况
  CD8: Fast Stale Detection — 连续未引用 N 次 → importance 快速降级
"""
import sys
import json
import uuid
import sqlite3
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import tmpfs  # noqa

from store import open_db, ensure_schema
from tools.citation_detector import (
    run_citation_detection,
    CITED_IMPORTANCE_DELTA,
    UNCITED_IMPORTANCE_DELTA,
    CITATION_TRIGRAM_THRESHOLD,
    MIN_IMPORTANCE,
    MAX_IMPORTANCE,
    STALE_CONSEC_THRESHOLD,
    STALE_FAST_PENALTY,
    SKIP_CITATION_TYPES,
)


def _utcnow():
    return datetime.now(timezone.utc)


def _insert_chunk(conn, cid, project, summary, content="", importance=0.6,
                  chunk_type="decision"):
    now = _utcnow()
    la = (now - timedelta(minutes=5)).isoformat()
    now_iso = now.isoformat()
    conn.execute("""
        INSERT OR REPLACE INTO memory_chunks
        (id, project, source_session, chunk_type, summary, content,
         importance, stability, retrievability, info_class, tags,
         access_count, oom_adj, created_at, updated_at, last_accessed,
         feishu_url, lru_gen, raw_snippet, encode_context, session_type_history)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (cid, project, "test", chunk_type, summary, content or summary,
          importance, 10.0, 0.5, "episodic", json.dumps([]),
          0, 0, now_iso, now_iso, la,
          None, 0, (content or summary)[:500], "{}", ""))
    conn.commit()


def _insert_recall_trace(conn, project, session_id, chunk_ids_summaries: list):
    """插入一条 recall_trace 记录，top_k_json 包含 chunk 列表。"""
    trace_id = "trace_" + uuid.uuid4().hex[:12]
    now = _utcnow().isoformat()
    top_k = [{"id": cid, "summary": summary}
             for cid, summary in chunk_ids_summaries]
    conn.execute("""
        INSERT INTO recall_traces
        (id, project, session_id, prompt_hash, timestamp, top_k_json, injected,
         reason, duration_ms, agent_id, ftrace_json)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (trace_id, project, session_id, "testhash", now,
          json.dumps(top_k), 1, "test", 10.0, session_id[:16], None))
    conn.commit()
    return trace_id


def _get_importance(conn, chunk_id):
    row = conn.execute("SELECT importance FROM memory_chunks WHERE id=?",
                       (chunk_id,)).fetchone()
    return row[0] if row else None


def _cleanup(conn, *projects):
    for p in projects:
        conn.execute("DELETE FROM memory_chunks WHERE project=?", (p,))
    conn.commit()


# ── CD1: 引用 → importance 微增 ────────────────────────────────────────────────

def test_cd1_cited_chunk_importance_increases():
    """CD1: 回复文本引用了 chunk summary → importance 增加 CITED_IMPORTANCE_DELTA。"""
    conn = open_db()
    ensure_schema(conn)

    proj = f"cd_proj_{uuid.uuid4().hex[:6]}"
    session = f"sess_{uuid.uuid4().hex[:8]}"
    cid = f"cd1_{uuid.uuid4().hex[:12]}"
    summary = "Linux page fault handling: validate PTE entry allocate physical page"
    initial_imp = 0.60

    _insert_chunk(conn, cid, proj, summary, importance=initial_imp)
    _insert_recall_trace(conn, proj, session, [(cid, summary)])

    # 回复文本明确包含 summary 中的关键词
    reply = ("The page fault handler validates the PTE entry and then allocates "
             "a physical page from the buddy allocator. This is the core mechanism "
             "in Linux page fault handling.")

    stats = run_citation_detection(reply, proj, session, conn=conn)
    conn.commit()

    new_imp = _get_importance(conn, cid)
    assert stats["cited"] >= 1, f"CD1: 应检测到引用, stats={stats}"
    assert new_imp > initial_imp, (
        f"CD1: 引用后 importance 应增加，{initial_imp:.3f} → {new_imp:.3f}"
    )
    assert abs(new_imp - (initial_imp + CITED_IMPORTANCE_DELTA)) < 0.005, (
        f"CD1: importance 增量应为 {CITED_IMPORTANCE_DELTA}, got {new_imp - initial_imp:.4f}"
    )

    _cleanup(conn, proj)
    conn.close()


# ── CD2: 未引用 → importance 微减 ─────────────────────────────────────────────

def test_cd2_uncited_chunk_importance_decreases():
    """CD2: 回复文本与 chunk 内容无关 → importance 微减。"""
    conn = open_db()
    ensure_schema(conn)

    proj = f"cd_proj_{uuid.uuid4().hex[:6]}"
    session = f"sess_{uuid.uuid4().hex[:8]}"
    cid = f"cd2_{uuid.uuid4().hex[:12]}"
    summary = "Linux page fault handling: validate PTE entry allocate physical page"
    initial_imp = 0.60

    _insert_chunk(conn, cid, proj, summary, importance=initial_imp)
    _insert_recall_trace(conn, proj, session, [(cid, summary)])

    # 回复文本与 chunk 完全无关
    reply = "Python asyncio event loop uses select() system call on Linux for I/O multiplexing."

    stats = run_citation_detection(reply, proj, session, conn=conn)
    conn.commit()

    new_imp = _get_importance(conn, cid)
    assert stats["uncited"] >= 1, f"CD2: 应检测到未引用, stats={stats}"
    assert new_imp < initial_imp, (
        f"CD2: 未引用后 importance 应减小，{initial_imp:.3f} → {new_imp:.3f}"
    )

    _cleanup(conn, proj)
    conn.close()


# ── CD3: 无 recall_trace → 跳过 ──────────────────────────────────────────────

def test_cd3_no_recall_trace_skipped():
    """CD3: 没有 recall_trace 记录时，run_citation_detection 正常返回，不崩溃。"""
    conn = open_db()
    ensure_schema(conn)

    proj = f"cd_empty_{uuid.uuid4().hex[:6]}"
    session = f"sess_{uuid.uuid4().hex[:8]}"

    stats = run_citation_detection(
        "Some reply text about Linux kernel.", proj, session, conn=conn
    )

    # 没有 recall_trace 应该 skipped，不崩溃
    assert stats.get("skipped", 0) >= 1 or (stats["cited"] == 0 and stats["uncited"] == 0), \
        f"CD3: 无 recall_trace 应跳过或返回空统计，got {stats}"

    conn.close()


# ── CD4: importance 边界保护 ──────────────────────────────────────────────────

def test_cd4_importance_bounds():
    """CD4: importance 被钳制在 [MIN_IMPORTANCE, MAX_IMPORTANCE]。"""
    conn = open_db()
    ensure_schema(conn)

    proj = f"cd_bound_{uuid.uuid4().hex[:6]}"
    session = f"sess_{uuid.uuid4().hex[:8]}"

    # 上界测试：importance 已接近 MAX
    cid_high = f"cd4h_{uuid.uuid4().hex[:12]}"
    summary_high = "Linux page fault: PTE validate allocate physical memory page"
    _insert_chunk(conn, cid_high, proj, summary_high, importance=MAX_IMPORTANCE - 0.005)
    _insert_recall_trace(conn, proj, session, [(cid_high, summary_high)])

    reply = ("Linux page fault handling validates PTE and allocates physical memory pages "
             "through the buddy allocator.")
    run_citation_detection(reply, proj, session, conn=conn)
    conn.commit()

    imp_high = _get_importance(conn, cid_high)
    assert imp_high <= MAX_IMPORTANCE, f"CD4: importance 不应超过 {MAX_IMPORTANCE}, got {imp_high}"

    # 下界测试：importance 已接近 MIN
    cid_low = f"cd4l_{uuid.uuid4().hex[:12]}"
    summary_low = "completely unrelated topic about java spring framework"
    _insert_chunk(conn, cid_low, proj, summary_low, importance=MIN_IMPORTANCE + 0.005)
    _insert_recall_trace(conn, proj, session, [(cid_low, summary_low)])

    unrelated_reply = "asyncio event loop select system call multiplexing"
    run_citation_detection(unrelated_reply, proj, session, conn=conn)
    conn.commit()

    imp_low = _get_importance(conn, cid_low)
    assert imp_low >= MIN_IMPORTANCE, f"CD4: importance 不应低于 {MIN_IMPORTANCE}, got {imp_low}"

    _cleanup(conn, proj)
    conn.close()


# ── CD5: semantic chunk 级联更新 ──────────────────────────────────────────────

def test_cd5_semantic_cascade():
    """CD5: cited chunk 的 source project 的 __semantic__ chunk importance 同向增加。"""
    conn = open_db()
    ensure_schema(conn)

    proj = f"cd_sem_{uuid.uuid4().hex[:6]}"
    session = f"sess_{uuid.uuid4().hex[:8]}"
    SEMANTIC_PROJECT = "__semantic__"

    # 源 chunk
    cid = f"cd5_{uuid.uuid4().hex[:12]}"
    summary = "Linux memory compaction: migrate anonymous pages reduce fragmentation"
    initial_src_imp = 0.70
    _insert_chunk(conn, cid, proj, summary, importance=initial_src_imp)
    _insert_recall_trace(conn, proj, session, [(cid, summary)])

    # 语义层 chunk（tags 包含 proj）
    sem_id = f"sem_cd5_{uuid.uuid4().hex[:12]}"
    sem_initial_imp = 0.72
    _insert_chunk(conn, sem_id, SEMANTIC_PROJECT,
                  "Linux memory compaction: migrate pages reduce fragmentation",
                  chunk_type="semantic_memory",
                  importance=sem_initial_imp)
    # 更新 tags 为包含 proj 的 JSON list
    conn.execute("UPDATE memory_chunks SET tags=? WHERE id=?",
                 (json.dumps([proj, f"other_proj_{uuid.uuid4().hex[:4]}"]), sem_id))
    conn.commit()

    # 引用该 chunk 的 reply
    reply = ("Linux memory compaction migrates anonymous pages to reduce memory "
             "fragmentation. This is critical for large allocation success rates.")

    stats = run_citation_detection(reply, proj, session, conn=conn)
    conn.commit()

    sem_imp = _get_importance(conn, sem_id)
    assert stats["cited"] >= 1, f"CD5: 源 chunk 应被引用, stats={stats}"
    assert sem_imp > sem_initial_imp, (
        f"CD5: semantic chunk importance 应随 cited 增加，{sem_initial_imp:.3f} → {sem_imp:.3f}"
    )

    _cleanup(conn, proj, SEMANTIC_PROJECT)
    conn.close()


# ── CD6: 空回复 → 跳过 ────────────────────────────────────────────────────────

def test_cd6_empty_reply_skipped():
    """CD6: 空或极短的回复文本 → 跳过，不报错，不更新。"""
    conn = open_db()
    ensure_schema(conn)

    proj = f"cd_empty_{uuid.uuid4().hex[:6]}"
    session = f"sess_{uuid.uuid4().hex[:8]}"
    cid = f"cd6_{uuid.uuid4().hex[:12]}"
    summary = "Linux page fault handling"
    initial_imp = 0.60
    _insert_chunk(conn, cid, proj, summary, importance=initial_imp)
    _insert_recall_trace(conn, proj, session, [(cid, summary)])

    # 空回复
    stats = run_citation_detection("", proj, session, conn=conn)
    assert stats["cited"] == 0 and stats["uncited"] == 0, \
        f"CD6: 空回复应跳过, got {stats}"

    # 极短回复（< 20 字符）
    stats2 = run_citation_detection("ok", proj, session, conn=conn)
    assert stats2["cited"] == 0 and stats2["uncited"] == 0, \
        f"CD6: 极短回复应跳过, got {stats2}"

    # importance 不应改变
    imp = _get_importance(conn, cid)
    assert abs(imp - initial_imp) < 0.001, \
        f"CD6: 空回复不应改变 importance, {initial_imp:.3f} → {imp:.3f}"

    _cleanup(conn, proj)
    conn.close()


# ── CD7: 混合引用 ─────────────────────────────────────────────────────────────

def test_cd7_mixed_citation():
    """CD7: 多个 chunk，部分被引用，部分未被引用。"""
    conn = open_db()
    ensure_schema(conn)

    proj = f"cd_mix_{uuid.uuid4().hex[:6]}"
    session = f"sess_{uuid.uuid4().hex[:8]}"

    cid_a = f"cd7a_{uuid.uuid4().hex[:10]}"
    cid_b = f"cd7b_{uuid.uuid4().hex[:10]}"
    summary_a = "Linux page fault PTE validate physical page allocation"
    summary_b = "Python asyncio event loop select poll epoll multiplexing"
    imp_a = imp_b = 0.60

    _insert_chunk(conn, cid_a, proj, summary_a, importance=imp_a)
    _insert_chunk(conn, cid_b, proj, summary_b, importance=imp_b)
    _insert_recall_trace(conn, proj, session, [
        (cid_a, summary_a),
        (cid_b, summary_b),
    ])

    # reply 只涉及 page fault，不涉及 asyncio
    reply = ("The Linux kernel page fault handler validates PTE entries and "
             "allocates physical pages through the buddy allocator system.")

    stats = run_citation_detection(reply, proj, session, conn=conn)
    conn.commit()

    new_imp_a = _get_importance(conn, cid_a)
    new_imp_b = _get_importance(conn, cid_b)

    assert stats["cited"] + stats["uncited"] >= 1, f"CD7: 应有 chunk 被处理, stats={stats}"
    # chunk_a 应被引用（importance 升高），chunk_b 不应被引用（importance 降低）
    assert new_imp_a >= imp_a, f"CD7: chunk_a 应被引用, imp {imp_a:.3f}→{new_imp_a:.3f}"
    assert new_imp_b <= imp_b, f"CD7: chunk_b 应未被引用, imp {imp_b:.3f}→{new_imp_b:.3f}"

    _cleanup(conn, proj)
    conn.close()


# ── CD8: Fast Stale Detection ─────────────────────────────────────────────────

def test_cd8_fast_stale_detection():
    """CD8: chunk 连续 N 次被检索但未被引用 → importance 快速降级。"""
    conn = open_db()
    ensure_schema(conn)

    proj = f"cd_stale_{uuid.uuid4().hex[:6]}"
    session = f"sess_{uuid.uuid4().hex[:8]}"
    cid = f"cd8_{uuid.uuid4().hex[:12]}"
    summary = "completely irrelevant topic about Java Spring Framework dependency injection"
    initial_imp = 0.60
    _insert_chunk(conn, cid, proj, summary, importance=initial_imp)

    # 插入 STALE_CONSEC_THRESHOLD 条 recall_trace，都包含该 chunk
    for i in range(STALE_CONSEC_THRESHOLD):
        _insert_recall_trace(conn, proj, f"sess_{i}", [(cid, summary)])

    # 回复文本与 chunk 完全无关
    unrelated_reply = ("The Linux kernel uses buddy allocator for physical page allocation "
                       "and slab allocator for small object allocation.")

    stats = run_citation_detection(unrelated_reply, proj, session, conn=conn)
    conn.commit()

    new_imp = _get_importance(conn, cid)
    # 应触发 fast stale（连续 N 次未引用）
    # iter473: stability-aware penalty — _insert_chunk 用 stability=10.0
    # actual_penalty = STALE_FAST_PENALTY × max(0.3, 1 - 10/20) = 0.30 × 0.5 = 0.15
    _chunk_stability = 10.0  # 与 _insert_chunk 中 stability=10.0 一致
    _stability_scale = max(0.3, 1.0 - _chunk_stability / 20.0)
    _actual_penalty = STALE_FAST_PENALTY * _stability_scale
    expected_after_stale = initial_imp * (1.0 - _actual_penalty)
    assert new_imp < initial_imp, (
        f"CD8: 快速降级后 importance 应小于初始值 {initial_imp:.3f}，got {new_imp:.3f}"
    )
    assert abs(new_imp - expected_after_stale) <= 0.05, (
        f"CD8: importance 应约为 {expected_after_stale:.3f}（penalty≈{_actual_penalty:.1%}），"
        f"initial={initial_imp:.3f} got={new_imp:.3f}"
    )
    assert stats.get("fast_stale_degraded", 0) >= 1, (
        f"CD8: stats 应记录 fast_stale_degraded≥1, got {stats}"
    )

    _cleanup(conn, proj)
    conn.close()


# ── CD9: 非知识类 chunk 跳过 ──────────────────────────────────────────────────

def test_cd9_skip_non_knowledge_chunks():
    """CD9: task_state/prompt_context 等非知识类 chunk 不参与 citation 检测和 stale 降级。"""
    conn = open_db()
    ensure_schema(conn)

    proj = f"cd_skip_{uuid.uuid4().hex[:6]}"
    session = f"sess_{uuid.uuid4().hex[:8]}"

    # task_state chunk（非知识）
    cid_task = f"cd9t_{uuid.uuid4().hex[:10]}"
    summary_task = "正在：实现 citation detector；下步：adaptive K"
    initial_imp_task = 0.70
    _insert_chunk(conn, cid_task, proj, summary_task,
                  importance=initial_imp_task, chunk_type="task_state")

    # decision chunk（知识，应正常参与）
    cid_dec = f"cd9d_{uuid.uuid4().hex[:10]}"
    summary_dec = "Linux page fault handling: PTE validate allocate physical page"
    initial_imp_dec = 0.60
    _insert_chunk(conn, cid_dec, proj, summary_dec,
                  importance=initial_imp_dec, chunk_type="decision")

    _insert_recall_trace(conn, proj, session, [
        (cid_task, summary_task),
        (cid_dec, summary_dec),
    ])

    # reply 包含 task_state 的文字（如果不跳过，task_state 会被误判为 cited）
    # 但也包含 decision chunk 的内容
    reply = ("The Linux page fault handler validates PTE entries and allocates "
             "physical memory pages. 正在：实现 citation detector 是当前任务。")

    stats = run_citation_detection(reply, proj, session, conn=conn)
    conn.commit()

    imp_task = _get_importance(conn, cid_task)
    imp_dec = _get_importance(conn, cid_dec)

    # task_state 不应参与 citation，importance 不变
    assert abs(imp_task - initial_imp_task) < 0.005, (
        f"CD9: task_state importance 不应改变，{initial_imp_task:.3f} → {imp_task:.3f}"
    )
    # decision chunk 应被引用，importance 增加
    assert imp_dec > initial_imp_dec, (
        f"CD9: decision chunk 应被引用，{initial_imp_dec:.3f} → {imp_dec:.3f}"
    )
    # skipped 计数应包含 task_state
    assert stats["skipped"] >= 1, f"CD9: task_state 应计入 skipped, stats={stats}"

    _cleanup(conn, proj)
    conn.close()


if __name__ == "__main__":
    print("Citation Detector 测试")
    print("=" * 60)

    tests = [
        test_cd1_cited_chunk_importance_increases,
        test_cd2_uncited_chunk_importance_decreases,
        test_cd3_no_recall_trace_skipped,
        test_cd4_importance_bounds,
        test_cd5_semantic_cascade,
        test_cd6_empty_reply_skipped,
        test_cd7_mixed_citation,
        test_cd8_fast_stale_detection,
        test_cd9_skip_non_knowledge_chunks,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  {test.__name__} ✓")
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
