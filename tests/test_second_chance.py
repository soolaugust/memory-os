#!/usr/bin/env python3
"""
迭代 34 测试：Second Chance — freshness_bonus 新知识曝光公平性

验证：
  1. freshness_bonus 在 grace period 内正确衰减
  2. freshness_bonus 超过 grace period 后为 0
  3. 新 chunk (access_count=0) 通过 freshness_bonus 获得竞争力
  4. 旧 chunk (access_count>0) 通过 access_bonus 保持优势
  5. retrieval_score 正确传递 created_at 参数
  6. config.py 新 tunable 注册正确
  7. 回归：旧有 retrieval_score (无 created_at) 行为不变
  8. 边界：created_at 为空串时 bonus=0
  9. 公平性验证：新旧 chunk 在过渡期的评分交叉
  10. 性能：freshness_bonus < 0.1ms/call
"""
import sys
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).parent))

from scorer import (
    freshness_bonus,
    retrieval_score,
    access_bonus,
    recency_score,
    importance_with_decay,
)
from config import get as sysctl_get, _REGISTRY


def test_freshness_bonus_brand_new():
    """刚创建的 chunk（age=0）应获得 max_bonus。"""
    now = datetime.now(timezone.utc).isoformat()
    bonus = freshness_bonus(now)
    max_bonus = sysctl_get("scorer.freshness_bonus_max")
    assert abs(bonus - max_bonus) < 0.01, f"brand new bonus={bonus}, expected ~{max_bonus}"
    print(f"  PASS: brand new chunk bonus={bonus:.4f} (max={max_bonus})")


def test_freshness_bonus_half_grace():
    """grace period 一半时，bonus 应约为 max/2。"""
    grace_days = sysctl_get("scorer.freshness_grace_days")
    half_ago = (datetime.now(timezone.utc) - timedelta(days=grace_days / 2)).isoformat()
    bonus = freshness_bonus(half_ago)
    max_bonus = sysctl_get("scorer.freshness_bonus_max")
    expected = max_bonus * 0.5
    assert abs(bonus - expected) < 0.02, f"half-grace bonus={bonus}, expected ~{expected}"
    print(f"  PASS: half-grace bonus={bonus:.4f} (expected ~{expected:.4f})")


def test_freshness_bonus_expired():
    """超过 grace period 后，bonus 应为 0。"""
    grace_days = sysctl_get("scorer.freshness_grace_days")
    old = (datetime.now(timezone.utc) - timedelta(days=grace_days + 1)).isoformat()
    bonus = freshness_bonus(old)
    assert bonus == 0.0, f"expired bonus={bonus}, expected 0.0"
    print(f"  PASS: expired chunk bonus=0.0")


def test_freshness_bonus_empty_string():
    """created_at 为空串时，bonus 应为 0。"""
    bonus = freshness_bonus("")
    # 空串 → _age_days 返回 30.0（fallback），超过 grace_days
    assert bonus == 0.0, f"empty string bonus={bonus}, expected 0.0"
    print(f"  PASS: empty string created_at → bonus=0.0")


def test_new_chunk_competitiveness():
    """新 chunk (access_count=0) 应通过 freshness_bonus 获得与旧 chunk 可比的分数。"""
    now_iso = datetime.now(timezone.utc).isoformat()
    three_days_ago = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()

    # 新 chunk：刚创建，relevance=0.8, importance=0.8, access=0
    new_score = retrieval_score(
        relevance=0.8, importance=0.8,
        last_accessed=now_iso, access_count=0,
        created_at=now_iso,
    )

    # 旧 chunk：3天前创建，relevance=0.8, importance=0.8, access=5
    old_score = retrieval_score(
        relevance=0.8, importance=0.8,
        last_accessed=three_days_ago, access_count=5,
        created_at=three_days_ago,
    )

    # 新 chunk 分数不应比旧 chunk 差太多（差距 < 30%）
    ratio = new_score / old_score if old_score > 0 else 0
    assert ratio > 0.7, f"new_score={new_score:.4f} old_score={old_score:.4f} ratio={ratio:.2f}"
    print(f"  PASS: new vs old: {new_score:.4f} vs {old_score:.4f} (ratio={ratio:.2f})")


def test_retrieval_score_backward_compat():
    """不传 created_at 时，retrieval_score 行为与迭代33 一致。"""
    now_iso = datetime.now(timezone.utc).isoformat()

    score_without = retrieval_score(
        relevance=0.9, importance=0.85,
        last_accessed=now_iso, access_count=3,
    )

    score_with_empty = retrieval_score(
        relevance=0.9, importance=0.85,
        last_accessed=now_iso, access_count=3,
        created_at="",
    )

    assert abs(score_without - score_with_empty) < 1e-6, \
        f"backward compat: {score_without} != {score_with_empty}"
    print(f"  PASS: backward compat: score={score_without:.6f}")


def test_config_tunables_registered():
    """新 tunable 应正确注册在 _REGISTRY 中。"""
    assert "scorer.freshness_bonus_max" in _REGISTRY, "freshness_bonus_max not in registry"
    assert "scorer.freshness_grace_days" in _REGISTRY, "freshness_grace_days not in registry"

    max_bonus = sysctl_get("scorer.freshness_bonus_max")
    grace_days = sysctl_get("scorer.freshness_grace_days")
    assert max_bonus == 0.15, f"default max_bonus={max_bonus}"
    assert grace_days == 7, f"default grace_days={grace_days}"
    print(f"  PASS: tunables registered (max={max_bonus}, grace={grace_days})")


def test_freshness_decays_linearly():
    """验证 freshness_bonus 在 grace period 内线性衰减。"""
    grace_days = sysctl_get("scorer.freshness_grace_days")
    max_bonus = sysctl_get("scorer.freshness_bonus_max")

    points = [0, 1, 2, 3, 5, grace_days - 1]
    prev_bonus = max_bonus + 1
    for d in points:
        t = (datetime.now(timezone.utc) - timedelta(days=d)).isoformat()
        b = freshness_bonus(t)
        expected = max_bonus * (1.0 - d / grace_days)
        assert abs(b - expected) < 0.02, f"day={d}: bonus={b}, expected={expected}"
        assert b < prev_bonus or d == 0, f"day={d}: not monotonically decreasing"
        prev_bonus = b
    print(f"  PASS: linear decay verified at {len(points)} points")


def test_crossover_point():
    """验证新旧 chunk 评分交叉：freshness 衰减后，access_bonus 接力。"""
    grace_days = sysctl_get("scorer.freshness_grace_days")

    # 新 chunk 在 grace_days-1 时的 freshness bonus
    near_expired = (datetime.now(timezone.utc) - timedelta(days=grace_days - 1)).isoformat()
    fb_near_expired = freshness_bonus(near_expired)

    # 旧 chunk access_count=10 的 access_bonus
    ab_veteran = access_bonus(10)

    # 在 grace period 尾部，access_bonus 应该超过 freshness_bonus
    assert ab_veteran > fb_near_expired, \
        f"crossover: access_bonus={ab_veteran} should > freshness={fb_near_expired}"
    print(f"  PASS: crossover: access_bonus={ab_veteran:.4f} > freshness={fb_near_expired:.4f}")


def test_performance():
    """freshness_bonus 延迟 < 0.1ms per call。"""
    now_iso = datetime.now(timezone.utc).isoformat()
    n = 1000
    t0 = time.time()
    for _ in range(n):
        freshness_bonus(now_iso)
    elapsed = (time.time() - t0) * 1000
    per_call = elapsed / n
    assert per_call < 0.1, f"too slow: {per_call:.3f}ms/call"
    print(f"  PASS: {n}x freshness_bonus = {elapsed:.1f}ms ({per_call:.4f}ms/call)")


if __name__ == "__main__":
    tests = [
        test_freshness_bonus_brand_new,
        test_freshness_bonus_half_grace,
        test_freshness_bonus_expired,
        test_freshness_bonus_empty_string,
        test_new_chunk_competitiveness,
        test_retrieval_score_backward_compat,
        test_config_tunables_registered,
        test_freshness_decays_linearly,
        test_crossover_point,
        test_performance,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            print(f"[TEST] {t.__name__}")
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Second Chance tests: {passed}/{passed+failed} passed")
    if failed:
        sys.exit(1)
    print("ALL PASSED")
