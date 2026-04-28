#!/usr/bin/env python3
"""
迭代26 测试：/proc Virtual Filesystem + Eviction Consistency Fix

验证：
  T1: proc_stats() 返回完整结构（chunks/retrieval/staleness/health）
  T2: proc_stats() 空库返回 total=0 不报错
  T3: evict_lowest_retention() 使用 unified scorer（高 access_count 保留）
  T4: evict_lowest_retention() 正确跳过 protect_types
  T5: proc_stats() 的 by_project / by_type 分布正确
  T6: eviction 一致性：低 retention_score 的先被淘汰
"""
import sys
import os
import tempfile
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta

# 设置测试环境
os.environ["CLAUDE_CWD"] = "/tmp/test_proc"

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

import tmpfs  # noqa: F401 — tmpfs isolation (iter54), must precede store import
from store import (open_db, ensure_schema, insert_chunk, proc_stats,
                   evict_lowest_retention, get_chunk_count, get_project_chunk_count,
                   insert_trace, update_accessed)
from schema import MemoryChunk
from scorer import retention_score as _retention_score

_PASS = 0
_FAIL = 0
_TIMES = []


def _assert(cond, msg):
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  ✅ {msg}")
    else:
        _FAIL += 1
        print(f"  ❌ {msg}")


def _make_chunk(project, chunk_type, summary, importance=0.5,
                access_count=0, age_days=1):  # age_days=1 绕过10分钟 grace period
    """创建测试 chunk，可指定 age（通过调整 last_accessed）。"""
    ts = datetime.now(timezone.utc) - timedelta(days=age_days)
    ts_iso = ts.isoformat()
    chunk = MemoryChunk(
        project=project,
        chunk_type=chunk_type,
        content=f"[{chunk_type}] {summary}",
        summary=summary,
        tags=[chunk_type, project],
        importance=importance,
    )
    chunk.last_accessed = ts_iso
    chunk.created_at = ts_iso
    chunk.updated_at = ts_iso
    d = chunk.to_dict()
    d["access_count"] = access_count
    return d


def test_proc_stats_empty():
    """T2: proc_stats 空库不报错。"""
    print("\n[T2] proc_stats 空库")
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    try:
        conn = open_db(db_path)
        ensure_schema(conn)
        stats = proc_stats(conn)
        _assert(stats["chunks"]["total"] == 0, "total=0")
        _assert(stats["retrieval"]["total_queries"] == 0, "queries=0")
        _assert(stats["staleness"]["active_pct"] == 0.0, "active_pct=0")
        conn.close()
    finally:
        db_path.unlink(missing_ok=True)


def test_proc_stats_full():
    """T1+T5: proc_stats 返回完整结构且分布正确。"""
    print("\n[T1+T5] proc_stats 完整结构 + 分布")
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    try:
        conn = open_db(db_path)
        ensure_schema(conn)

        # 插入测试数据：2 个项目，3 种类型
        insert_chunk(conn, _make_chunk("proj_a", "decision", "决策A1", 0.8))
        insert_chunk(conn, _make_chunk("proj_a", "decision", "决策A2", 0.7))
        insert_chunk(conn, _make_chunk("proj_a", "reasoning_chain", "推理A1", 0.6))
        insert_chunk(conn, _make_chunk("proj_b", "task_state", "任务B1", 0.5))
        insert_chunk(conn, _make_chunk("proj_b", "conversation_summary", "摘要B1", 0.4, age_days=40))

        # 插入 recall_trace
        import uuid
        now_iso = datetime.now(timezone.utc).isoformat()
        insert_trace(conn, {
            "id": str(uuid.uuid4()), "timestamp": now_iso,
            "session_id": "test", "project": "proj_a",
            "prompt_hash": "abc", "candidates_count": 10,
            "top_k_json": [], "injected": 1, "reason": "test",
            "duration_ms": 2.5,
        })
        insert_trace(conn, {
            "id": str(uuid.uuid4()), "timestamp": now_iso,
            "session_id": "test", "project": "proj_a",
            "prompt_hash": "def", "candidates_count": 5,
            "top_k_json": [], "injected": 0, "reason": "skipped",
            "duration_ms": 1.0,
        })
        conn.commit()

        t0 = time.time()
        stats = proc_stats(conn)
        dt_ms = (time.time() - t0) * 1000
        _TIMES.append(dt_ms)

        # T1: 完整结构
        _assert("chunks" in stats, "has chunks")
        _assert("retrieval" in stats, "has retrieval")
        _assert("staleness" in stats, "has staleness")
        _assert("health" in stats, "has health")

        # T5: 分布正确
        _assert(stats["chunks"]["total"] == 5, f"total=5 (got {stats['chunks']['total']})")
        _assert(stats["chunks"]["by_project"].get("proj_a") == 3,
                f"proj_a=3 (got {stats['chunks']['by_project'].get('proj_a')})")
        _assert(stats["chunks"]["by_project"].get("proj_b") == 2,
                f"proj_b=2 (got {stats['chunks']['by_project'].get('proj_b')})")
        _assert(stats["chunks"]["by_type"].get("decision") == 2,
                f"decision=2 (got {stats['chunks']['by_type'].get('decision')})")

        # 召回统计
        _assert(stats["retrieval"]["total_queries"] == 2, "queries=2")
        _assert(stats["retrieval"]["hit_rate_pct"] == 50.0, f"hit_rate=50% (got {stats['retrieval']['hit_rate_pct']})")
        _assert(stats["retrieval"]["avg_latency_ms"] > 0, "avg_latency > 0")

        # staleness: 40天前的 chunk 应出现在 30d 统计中
        _assert(stats["staleness"]["not_accessed_30d"] >= 1,
                f"stale_30d >= 1 (got {stats['staleness']['not_accessed_30d']})")

        print(f"  ⏱  proc_stats latency: {dt_ms:.2f}ms")
        conn.close()
    finally:
        db_path.unlink(missing_ok=True)


def test_eviction_uses_unified_scorer():
    """T3+T6: eviction 使用 unified scorer，高 access_count 优先保留。"""
    print("\n[T3+T6] eviction 使用 unified scorer")
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    try:
        conn = open_db(db_path)
        ensure_schema(conn)

        # chunk_low: importance=0.3, access_count=0 → low retention
        # chunk_high_access: importance=0.3, access_count=20 → access_freq 提升 retention
        # chunk_medium: importance=0.5, access_count=0 → medium retention
        low = _make_chunk("proj", "decision", "低价值低频", 0.3, access_count=0)
        high_access = _make_chunk("proj", "decision", "低价值高频", 0.3, access_count=20)
        medium = _make_chunk("proj", "decision", "中等价值", 0.5, access_count=0)

        insert_chunk(conn, low)
        insert_chunk(conn, high_access)
        insert_chunk(conn, medium)
        # 更新 access_count
        conn.execute("UPDATE memory_chunks SET access_count=20 WHERE summary='低价值高频'")
        conn.commit()

        # 验证 retention_score 排序预期
        low_score = _retention_score(0.3, low["last_accessed"], 0.5, 0)
        high_access_score = _retention_score(0.3, high_access["last_accessed"], 0.5, 20)
        medium_score = _retention_score(0.5, medium["last_accessed"], 0.5, 0)
        print(f"  scores: low={low_score:.4f}, high_access={high_access_score:.4f}, medium={medium_score:.4f}")

        _assert(high_access_score > low_score,
                f"high_access retention > low retention ({high_access_score:.4f} > {low_score:.4f})")

        # 淘汰 1 条：应该是 low（retention 最低）
        t0 = time.time()
        evicted = evict_lowest_retention(conn, "proj", 1)
        dt_ms = (time.time() - t0) * 1000
        _TIMES.append(dt_ms)

        _assert(len(evicted) == 1, f"evicted 1 chunk (got {len(evicted)})")
        _assert(evicted[0] == low["id"],
                f"evicted lowest retention (id={low['id'][:8]})")

        # 验证 high_access 保留（access_count=20 提升了 retention）
        remaining = conn.execute("SELECT summary FROM memory_chunks ORDER BY summary").fetchall()
        remaining_summaries = {r[0] for r in remaining}
        _assert("低价值高频" in remaining_summaries, "高频 chunk 被保留")
        _assert("中等价值" in remaining_summaries, "中等 chunk 被保留")

        print(f"  ⏱  eviction latency: {dt_ms:.2f}ms")
        conn.close()
    finally:
        db_path.unlink(missing_ok=True)


def test_eviction_protect_types():
    """T4: eviction 跳过 protect_types。"""
    print("\n[T4] eviction protect_types")
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    try:
        conn = open_db(db_path)
        ensure_schema(conn)

        # task_state 应受保护
        ts = _make_chunk("proj", "task_state", "当前任务", 0.1)
        dec = _make_chunk("proj", "decision", "低决策", 0.2)
        insert_chunk(conn, ts)
        insert_chunk(conn, dec)
        conn.commit()

        evicted = evict_lowest_retention(conn, "proj", 1)
        _assert(len(evicted) == 1, "evicted 1")
        _assert(evicted[0] == dec["id"], "evicted decision, not task_state")

        # task_state 仍在
        remaining = conn.execute(
            "SELECT chunk_type FROM memory_chunks"
        ).fetchall()
        types = {r[0] for r in remaining}
        _assert("task_state" in types, "task_state protected")
        conn.close()
    finally:
        db_path.unlink(missing_ok=True)


if __name__ == "__main__":
    print("=" * 60)
    print("迭代26 测试：/proc Virtual Filesystem + Eviction Fix")
    print("=" * 60)

    test_proc_stats_empty()
    test_proc_stats_full()
    test_eviction_uses_unified_scorer()
    test_eviction_protect_types()

    avg_ms = sum(_TIMES) / len(_TIMES) if _TIMES else 0
    print(f"\n{'=' * 60}")
    print(f"结果：{_PASS} passed, {_FAIL} failed, avg {avg_ms:.2f}ms")
    print(f"{'=' * 60}")
    sys.exit(1 if _FAIL > 0 else 0)
