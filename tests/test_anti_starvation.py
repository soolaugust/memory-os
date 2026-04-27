#!/usr/bin/env python3
"""
迭代62 测试：Anti-Starvation — 反饥饿机制

OS 类比：CFS vruntime aging + O(1) scheduler dynamic priority boost
验证：
  1. saturation_penalty 对反复召回的 chunk 施加递增惩罚
  2. starvation_boost 对长期未访问的 chunk 施加递增加分
  3. retrieval_score 集成：anti-starvation 能打破 relevance 锁定
  4. chunk_recall_counts 从 recall_traces 正确统计
  5. 向后兼容：recall_count=0 时行为不变
  6. sysctl tunables 注册且范围正确
"""
import sys
import os
import math

# 测试隔离
sys.path.insert(0, os.path.dirname(__file__))
import tmpfs  # noqa: F401 — 自动设置临时目录

from scorer import (
    saturation_penalty, starvation_boost, retrieval_score,
    exploration_bonus, freshness_bonus, access_bonus,
)
from config import get as sysctl
import store
import json
from datetime import datetime, timezone, timedelta



if __name__ == "__main__":
    passed = 0
    failed = 0


    def ok(name):
        global passed
        passed += 1
        print(f"  \u2713 {name}")


    def fail(name, msg=""):
        global failed
        failed += 1
        print(f"  \u2717 {name}: {msg}")


    def test(name, condition, msg=""):
        if condition:
            ok(name)
        else:
            fail(name, msg)


    print("=" * 50)
    print("Anti-Starvation Tests (迭代62)")
    print("=" * 50)

    # ── 1. saturation_penalty 基础行为 ──
    print("\n[1] saturation_penalty 基础行为")

    # recall_count=0 → 无惩罚
    test("zero recall -> zero penalty",
         saturation_penalty(0) == 0.0)

    # recall_count > 0 → 递增惩罚
    p1 = saturation_penalty(1)
    p5 = saturation_penalty(5)
    p10 = saturation_penalty(10)
    p30 = saturation_penalty(30)
    test("penalty monotonically increases",
         0 < p1 < p5 < p10 < p30,
         f"p1={p1:.4f} p5={p5:.4f} p10={p10:.4f} p30={p30:.4f}")

    # penalty 有上限
    cap = sysctl("scorer.saturation_cap")
    test("penalty capped",
         saturation_penalty(1000) <= cap,
         f"p1000={saturation_penalty(1000):.4f} cap={cap}")

    # 具体数值验证（factor=0.04, log2(1+3)=2.0）
    factor = sysctl("scorer.saturation_factor")
    expected_p3 = factor * math.log2(1 + 3)
    test("recall_count=3 value correct",
         abs(saturation_penalty(3) - expected_p3) < 1e-10,
         f"got={saturation_penalty(3):.6f} expected={expected_p3:.6f}")

    # ── 2. starvation_boost 基础行为 ──
    print("\n[2] starvation_boost 基础行为")

    # access_count > 0 → 无加分
    test("accessed chunk -> zero boost",
         starvation_boost(1, 10.0) == 0.0)

    # age < min_age → 无加分
    min_age = sysctl("scorer.starvation_min_age_days")
    test("young chunk -> zero boost",
         starvation_boost(0, min_age * 0.5) == 0.0,
         f"min_age={min_age}")

    # age = min_age → 刚好开始加分（0.0）
    test("at min_age -> zero boost",
         starvation_boost(0, min_age) == 0.0)

    # age > min_age → 递增加分
    ramp = sysctl("scorer.starvation_ramp_days")
    b_mid = starvation_boost(0, min_age + ramp / 2)
    b_full = starvation_boost(0, min_age + ramp)
    b_over = starvation_boost(0, min_age + ramp * 2)
    test("boost ramps up linearly",
         0 < b_mid < b_full,
         f"mid={b_mid:.4f} full={b_full:.4f}")

    # 满额后不再增长
    boost_factor = sysctl("scorer.starvation_boost_factor")
    test("boost saturates at factor",
         abs(b_full - boost_factor) < 1e-10,
         f"full={b_full:.6f} factor={boost_factor}")
    test("boost no further increase past ramp",
         abs(b_over - boost_factor) < 1e-10,
         f"over={b_over:.6f}")

    # ── 3. retrieval_score 集成 ──
    print("\n[3] retrieval_score 集成")

    now_iso = datetime.now(timezone.utc).isoformat()
    old_iso = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()

    # 向后兼容：recall_count=0 时行为不变
    score_compat = retrieval_score(
        relevance=0.5, importance=0.8, last_accessed=now_iso,
        access_count=2, created_at=now_iso, chunk_id="test1", query_seed="q1"
    )
    score_explicit_0 = retrieval_score(
        relevance=0.5, importance=0.8, last_accessed=now_iso,
        access_count=2, created_at=now_iso, chunk_id="test1", query_seed="q1",
        recall_count=0
    )
    test("backward compat: recall_count default=0",
         abs(score_compat - score_explicit_0) < 1e-10,
         f"compat={score_compat:.6f} explicit={score_explicit_0:.6f}")

    # 饱和惩罚：高 recall_count 降低分数
    score_low_recall = retrieval_score(
        relevance=1.0, importance=0.5, last_accessed=now_iso,
        access_count=5, created_at=now_iso, chunk_id="hot1", query_seed="q",
        recall_count=0
    )
    score_high_recall = retrieval_score(
        relevance=1.0, importance=0.5, last_accessed=now_iso,
        access_count=5, created_at=now_iso, chunk_id="hot1", query_seed="q",
        recall_count=20
    )
    test("saturation penalty reduces score",
         score_high_recall < score_low_recall,
         f"low_recall={score_low_recall:.4f} high_recall={score_high_recall:.4f}")

    # 饥饿加分：access=0 + old → 加分明显
    score_starving = retrieval_score(
        relevance=0.15, importance=0.9, last_accessed=old_iso,
        access_count=0, created_at=old_iso, chunk_id="cold1", query_seed="q",
        recall_count=0
    )
    score_not_starving = retrieval_score(
        relevance=0.15, importance=0.9, last_accessed=old_iso,
        access_count=1, created_at=old_iso, chunk_id="cold1", query_seed="q",
        recall_count=0
    )
    test("starvation boost: access=0 gets more than access=1",
         score_starving > score_not_starving,
         f"starving={score_starving:.4f} not_starving={score_not_starving:.4f}")

    # ── 4. 核心场景：anti-starvation 打破 relevance 锁定 ──
    print("\n[4] 核心场景：打破 relevance 锁定")

    # 模拟：高 relevance 的热门 chunk vs 低 relevance 的饥饿 chunk
    # 热门 chunk: relevance=1.0, recall_count=20 (反复被选)
    score_hot = retrieval_score(
        relevance=1.0, importance=0.5, last_accessed=now_iso,
        access_count=5, created_at=now_iso, chunk_id="monopoly", query_seed="q",
        recall_count=20
    )
    # 饥饿 chunk: relevance=0.15, access=0, age=5天
    score_cold = retrieval_score(
        relevance=0.15, importance=0.9, last_accessed=old_iso,
        access_count=0, created_at=old_iso, chunk_id="starved", query_seed="q",
        recall_count=0
    )
    print(f"  hot_chunk score={score_hot:.4f} (rel=1.0, recall=20)")
    print(f"  cold_chunk score={score_cold:.4f} (rel=0.15, access=0, age=5d)")
    # Anti-starvation 目标：缩小差距，让 cold chunk 有竞争力
    gap_ratio = score_hot / max(score_cold, 0.001)
    test("gap ratio reduced (< 3x with anti-starvation)",
         gap_ratio < 3.0,
         f"gap_ratio={gap_ratio:.2f}x")

    # ── 5. chunk_recall_counts 从 recall_traces 统计 ──
    print("\n[5] chunk_recall_counts 统计")

    conn = store.open_db()
    store.ensure_schema(conn)

    # 插入测试 traces
    import uuid
    for i in range(5):
        store.insert_trace(conn, {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": "test",
            "project": "test_proj",
            "prompt_hash": "abc",
            "candidates_count": 3,
            "top_k_json": json.dumps([
                {"id": "chunk_a", "summary": "a", "score": 0.9},
                {"id": "chunk_b", "summary": "b", "score": 0.5},
            ]),
            "injected": 1,
            "reason": "test",
            "duration_ms": 1.0,
        })

    # chunk_a 出现额外 1 次
    store.insert_trace(conn, {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": "test",
        "project": "test_proj",
        "prompt_hash": "abc",
        "candidates_count": 3,
        "top_k_json": json.dumps([
            {"id": "chunk_a", "summary": "a", "score": 0.9},
            {"id": "chunk_c", "summary": "c", "score": 0.3},
        ]),
        "injected": 1,
        "reason": "test",
        "duration_ms": 1.0,
    })
    conn.commit()

    counts = store.chunk_recall_counts(conn, "test_proj", window=30)
    test("chunk_a count=6", counts.get("chunk_a", 0) == 6,
         f"got={counts.get('chunk_a', 0)}")
    test("chunk_b count=5", counts.get("chunk_b", 0) == 5,
         f"got={counts.get('chunk_b', 0)}")
    test("chunk_c count=1", counts.get("chunk_c", 0) == 1,
         f"got={counts.get('chunk_c', 0)}")
    test("unknown chunk count=0", counts.get("chunk_z", 0) == 0)

    # window 限制：只看最近 2 条
    counts_small = store.chunk_recall_counts(conn, "test_proj", window=2)
    test("window=2 limits scope",
         counts_small.get("chunk_a", 0) <= 2,
         f"got={counts_small.get('chunk_a', 0)}")

    # 不同项目隔离
    counts_other = store.chunk_recall_counts(conn, "other_proj", window=30)
    test("project isolation", len(counts_other) == 0)

    # injected=0 的 trace 不计入
    store.insert_trace(conn, {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": "test",
        "project": "test_proj",
        "prompt_hash": "xyz",
        "candidates_count": 1,
        "top_k_json": json.dumps([{"id": "chunk_skip", "summary": "s", "score": 0.1}]),
        "injected": 0,
        "reason": "skipped_same_hash",
        "duration_ms": 0,
    })
    conn.commit()
    counts2 = store.chunk_recall_counts(conn, "test_proj", window=30)
    test("non-injected trace excluded", counts2.get("chunk_skip", 0) == 0)

    conn.close()

    # ── 6. sysctl tunables 验证 ──
    print("\n[6] sysctl tunables")

    tunables = [
        ("scorer.saturation_factor", 0.04, float),
        ("scorer.saturation_cap", 0.25, float),
        ("scorer.starvation_boost_factor", 0.30, float),
        ("scorer.starvation_min_age_days", 0.5, float),
        ("scorer.starvation_ramp_days", 3.0, float),
    ]
    for key, expected, typ in tunables:
        val = sysctl(key)
        test(f"{key}={val}",
             isinstance(val, typ) and val == expected,
             f"got={val} expected={expected}")

    # ── 7. 性能测试 ──
    print("\n[7] 性能测试")

    import time
    N = 10000
    t0 = time.perf_counter()
    for _ in range(N):
        saturation_penalty(15)
    t1 = time.perf_counter()
    sp_ms = (t1 - t0) / N * 1000
    print(f"  saturation_penalty: {sp_ms:.4f}ms/call")
    test("saturation_penalty < 0.01ms", sp_ms < 0.01)

    t0 = time.perf_counter()
    for _ in range(N):
        starvation_boost(0, 3.0)
    t1 = time.perf_counter()
    sb_ms = (t1 - t0) / N * 1000
    print(f"  starvation_boost: {sb_ms:.4f}ms/call")
    test("starvation_boost < 0.01ms", sb_ms < 0.01)

    # retrieval_score with recall_count
    t0 = time.perf_counter()
    for _ in range(N):
        retrieval_score(
            relevance=0.5, importance=0.8, last_accessed=now_iso,
            access_count=3, created_at=old_iso, chunk_id="perf1", query_seed="q",
            recall_count=10
        )
    t1 = time.perf_counter()
    rs_ms = (t1 - t0) / N * 1000
    print(f"  retrieval_score (with anti-starvation): {rs_ms:.4f}ms/call")
    test("retrieval_score < 0.05ms", rs_ms < 0.05)

    # chunk_recall_counts
    conn = store.open_db()
    store.ensure_schema(conn)
    t0 = time.perf_counter()
    for _ in range(100):
        store.chunk_recall_counts(conn, "test_proj", window=30)
    t1 = time.perf_counter()
    crc_ms = (t1 - t0) / 100 * 1000
    conn.close()
    print(f"  chunk_recall_counts: {crc_ms:.2f}ms/call")
    test("chunk_recall_counts < 5ms", crc_ms < 5.0)

    # ── 汇总 ──
    print("\n" + "=" * 50)
    print(f"Anti-Starvation Tests: {passed} passed, {failed} failed, total {passed + failed}")
    if failed == 0:
        print("ALL PASSED \u2713")
    else:
        print(f"FAILURES: {failed}")
        sys.exit(1)
