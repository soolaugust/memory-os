#!/usr/bin/env python3
"""
迭代43 测试：ASLR — 检索结果多样性随机化
OS 类比：Linux ASLR (Address Space Layout Randomization, 2005)

测试矩阵：
  T1 exploration_bonus 基本计算（access_count=0 → 有扰动）
  T2 access_count ≥ threshold → bonus=0（高访问 chunk 稳定）
  T3 同 chunk+同 query → 确定性（相同 bonus）
  T4 同 chunk+不同 query → 不同 bonus（跨 query 多样性）
  T5 不同 chunk+同 query → 不同 bonus（跨 chunk 差异）
  T6 epsilon=0 → 全部 bonus=0（可完全禁用）
  T7 access_count 越高 bonus 越小（单调递减）
  T8 retrieval_score 集成（含 ASLR）
  T9 retrieval_score 无 chunk_id 时向后兼容
  T10 sysctl tunables 验证
  T11 性能测试（1000 次调用 < 0.05ms avg）
  T12 ASLR 产生排名变化（同 relevance/importance 不同 chunk 排名不固定）
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("CLAUDE_CWD", str(__import__("pathlib").Path(__file__).parent.parent.parent.parent.parent))

from scorer import exploration_bonus, retrieval_score
from config import get as sysctl_get
from datetime import datetime, timezone

_PASS = 0
_FAIL = 0


def _assert(name: str, condition: bool, detail: str = ""):
    global _PASS, _FAIL
    if condition:
        _PASS += 1
        print(f"  PASS  {name}")
    else:
        _FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def test_basic_bonus():
    """T1: access_count=0 → 有扰动"""
    bonus = exploration_bonus("chunk_abc", 0, "test query")
    _assert("T1.1 bonus > 0 for access_count=0", bonus > 0, f"got {bonus}")
    _assert("T1.2 bonus <= epsilon (0.08)", bonus <= 0.08, f"got {bonus}")


def test_high_access_no_bonus():
    """T2: access_count >= threshold → bonus=0"""
    threshold = sysctl_get("scorer.aslr_access_threshold")
    bonus = exploration_bonus("chunk_abc", threshold, "test query")
    _assert("T2.1 bonus=0 at threshold", bonus == 0.0, f"got {bonus}")
    bonus2 = exploration_bonus("chunk_abc", threshold + 10, "test query")
    _assert("T2.2 bonus=0 above threshold", bonus2 == 0.0, f"got {bonus2}")


def test_deterministic():
    """T3: 同 chunk + 同 query → 确定性"""
    b1 = exploration_bonus("chunk_123", 0, "query_a")
    b2 = exploration_bonus("chunk_123", 0, "query_a")
    _assert("T3.1 same chunk+query → same bonus", b1 == b2, f"{b1} != {b2}")


def test_cross_query_diversity():
    """T4: 同 chunk + 不同 query → 不同 bonus"""
    b1 = exploration_bonus("chunk_123", 0, "query_a")
    b2 = exploration_bonus("chunk_123", 0, "query_b")
    b3 = exploration_bonus("chunk_123", 0, "query_c")
    values = {b1, b2, b3}
    _assert("T4.1 different queries → different bonuses", len(values) >= 2,
            f"all same: {values}")


def test_cross_chunk_diversity():
    """T5: 不同 chunk + 同 query → 不同 bonus"""
    b1 = exploration_bonus("chunk_aaa", 0, "same_query")
    b2 = exploration_bonus("chunk_bbb", 0, "same_query")
    b3 = exploration_bonus("chunk_ccc", 0, "same_query")
    values = {b1, b2, b3}
    _assert("T5.1 different chunks → different bonuses", len(values) >= 2,
            f"all same: {values}")


def test_epsilon_zero_disables():
    """T6: epsilon=0 → 全部 bonus=0"""
    os.environ["MEMORY_OS_SCORER_ASLR_EPSILON"] = "0"
    from config import _invalidate_cache
    _invalidate_cache()

    bonus = exploration_bonus("chunk_abc", 0, "test_query")
    _assert("T6.1 epsilon=0 → bonus=0", bonus == 0.0, f"got {bonus}")

    del os.environ["MEMORY_OS_SCORER_ASLR_EPSILON"]
    _invalidate_cache()


def test_monotonic_decrease():
    """T7: access_count 越高 bonus 越小"""
    bonuses = []
    for ac in range(6):
        total = sum(exploration_bonus(f"chunk_{i}", ac, "test_q") for i in range(100))
        bonuses.append(total / 100)

    _assert("T7.1 bonus[0] > bonus[4]", bonuses[0] > bonuses[4],
            f"bonuses: {[f'{b:.4f}' for b in bonuses]}")
    _assert("T7.2 bonus[5] == 0 (at threshold)", bonuses[5] == 0.0,
            f"got {bonuses[5]}")


def test_retrieval_score_integration():
    """T8: retrieval_score 集成 ASLR"""
    now = _now_iso()
    score_no_aslr = retrieval_score(
        relevance=0.8, importance=0.7,
        last_accessed=now, access_count=0,
        created_at=now,
    )
    score_with_aslr = retrieval_score(
        relevance=0.8, importance=0.7,
        last_accessed=now, access_count=0,
        created_at=now,
        chunk_id="test_chunk_1", query_seed="test_query",
    )
    _assert("T8.1 ASLR adds bonus", score_with_aslr >= score_no_aslr,
            f"no_aslr={score_no_aslr:.4f} with_aslr={score_with_aslr:.4f}")


def test_backward_compat():
    """T9: 无 chunk_id 时完全向后兼容"""
    now = _now_iso()
    score_old = retrieval_score(
        relevance=0.8, importance=0.7,
        last_accessed=now, access_count=5,
        created_at=now,
    )
    score_new = retrieval_score(
        relevance=0.8, importance=0.7,
        last_accessed=now, access_count=5,
        created_at=now,
        chunk_id="", query_seed="anything",
    )
    _assert("T9.1 empty chunk_id → no ASLR effect",
            abs(score_old - score_new) < 1e-10,
            f"old={score_old:.6f} new={score_new:.6f}")


def test_sysctl():
    """T10: sysctl tunables"""
    _assert("T10.1 aslr_epsilon default", sysctl_get("scorer.aslr_epsilon") == 0.08)
    _assert("T10.2 aslr_access_threshold default",
            sysctl_get("scorer.aslr_access_threshold") == 5)


def test_performance():
    """T11: 性能测试"""
    N = 1000
    t0 = time.time()
    for i in range(N):
        exploration_bonus(f"chunk_{i}", i % 5, f"query_{i % 10}")
    elapsed = (time.time() - t0) * 1000
    avg = elapsed / N
    _assert(f"T11.1 {N}x exploration_bonus < 0.05ms avg", avg < 0.05,
            f"avg={avg:.4f}ms")
    print(f"  INFO  {N}x exploration_bonus: {elapsed:.1f}ms total, {avg:.4f}ms/call")


def test_rank_diversity():
    """T12: ASLR 产生排名变化"""
    now = _now_iso()
    chunks = [f"chunk_{i}" for i in range(10)]

    def rank_order(q_seed):
        scores = []
        for cid in chunks:
            s = retrieval_score(
                relevance=0.5, importance=0.6,
                last_accessed=now, access_count=1,
                created_at=now,
                chunk_id=cid, query_seed=q_seed,
            )
            scores.append((s, cid))
        scores.sort(reverse=True)
        return [cid for _, cid in scores]

    order_a = rank_order("query_alpha")
    order_b = rank_order("query_beta")

    _assert("T12.1 different query → different rank order",
            order_a != order_b,
            f"same order: {order_a[:5]}")


if __name__ == "__main__":
    t0 = time.time()
    print("=" * 60)
    print("迭代43：ASLR — 检索结果多样性随机化")
    print("=" * 60)

    test_basic_bonus()
    test_high_access_no_bonus()
    test_deterministic()
    test_cross_query_diversity()
    test_cross_chunk_diversity()
    test_epsilon_zero_disables()
    test_monotonic_decrease()
    test_retrieval_score_integration()
    test_backward_compat()
    test_sysctl()
    test_performance()
    test_rank_diversity()

    elapsed = (time.time() - t0) * 1000
    print("=" * 60)
    print(f"结果：{_PASS} PASS / {_FAIL} FAIL  ({elapsed:.0f}ms)")
    print("=" * 60)
    sys.exit(1 if _FAIL > 0 else 0)
