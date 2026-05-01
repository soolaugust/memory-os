#!/usr/bin/env python3
"""
test_inject_sort.py — inject_score 加权排序测试（iter472）

覆盖：
  IS1: 同等 trigram_score，高 importance chunk 应排在低 importance chunk 之前
  IS2: 高 trigram_score 低 importance 与 低 trigram_score 高 importance 的平衡
  IS3: design_constraint chunk 不参与 inject_sort（始终在头部）
"""
import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

# 直接测试 inject_sort 排序逻辑（不需要完整的 retriever 环境）


def _inject_score(trigram_score: float, importance: float) -> float:
    """inject_score = trigram_score × sqrt(importance)，与 retriever.py 逻辑一致。"""
    return trigram_score * math.sqrt(max(0.01, importance))


def _apply_inject_sort(top_k: list) -> list:
    """
    模拟 retriever.py iter472 inject_sort 逻辑：
    design_constraint 保持原位，普通 chunk 按 inject_score 降序排列。
    """
    constraints = [(s, c) for s, c in top_k if c.get("chunk_type") == "design_constraint"]
    normal = [(s, c) for s, c in top_k if c.get("chunk_type") != "design_constraint"]
    if len(normal) < 2:
        return top_k
    normal_sorted = sorted(
        normal,
        key=lambda x: x[0] * math.sqrt(max(0.01, float(x[1].get("importance") or 0.5))),
        reverse=True,
    )
    return constraints + normal_sorted


def test_is1_same_score_high_importance_first():
    """IS1: 相同 trigram_score 时，高 importance 的 chunk 应排前面。"""
    top_k = [
        (0.5, {"id": "low_imp", "summary": "Low importance chunk", "importance": 0.10, "chunk_type": "decision"}),
        (0.5, {"id": "high_imp", "summary": "High importance chunk", "importance": 0.90, "chunk_type": "decision"}),
        (0.5, {"id": "mid_imp", "summary": "Mid importance chunk", "importance": 0.50, "chunk_type": "decision"}),
    ]

    sorted_top_k = _apply_inject_sort(top_k)
    ids_in_order = [c["id"] for _, c in sorted_top_k]

    assert ids_in_order[0] == "high_imp", (
        f"IS1: 最高 importance 应排第一, got {ids_in_order}"
    )
    assert ids_in_order[1] == "mid_imp", (
        f"IS1: 中等 importance 应排第二, got {ids_in_order}"
    )
    assert ids_in_order[2] == "low_imp", (
        f"IS1: 最低 importance 应排最后, got {ids_in_order}"
    )
    print(f"  IS1 PASS: same-score ordering: {ids_in_order}")


def test_is2_score_importance_balance():
    """IS2: inject_score = score × sqrt(importance) 平衡相关性和重要性。"""
    # chunk A: score=0.8, importance=0.1 → inject_score = 0.8 × sqrt(0.1) ≈ 0.253
    # chunk B: score=0.3, importance=0.9 → inject_score = 0.3 × sqrt(0.9) ≈ 0.285
    # chunk C: score=0.6, importance=0.5 → inject_score = 0.6 × sqrt(0.5) ≈ 0.424
    # Expected order: C > B > A
    top_k = [
        (0.8, {"id": "A", "summary": "High score low imp", "importance": 0.1, "chunk_type": "decision"}),
        (0.3, {"id": "B", "summary": "Low score high imp", "importance": 0.9, "chunk_type": "decision"}),
        (0.6, {"id": "C", "summary": "Mid score mid imp", "importance": 0.5, "chunk_type": "decision"}),
    ]

    sorted_top_k = _apply_inject_sort(top_k)
    ids_in_order = [c["id"] for _, c in sorted_top_k]

    # Verify inject_scores
    score_a = _inject_score(0.8, 0.1)  # ≈ 0.253
    score_b = _inject_score(0.3, 0.9)  # ≈ 0.285
    score_c = _inject_score(0.6, 0.5)  # ≈ 0.424

    assert score_c > score_b > score_a, (
        f"IS2: inject_score 计算错误: A={score_a:.3f} B={score_b:.3f} C={score_c:.3f}"
    )
    assert ids_in_order == ["C", "B", "A"], (
        f"IS2: 排序应为 C>B>A, got {ids_in_order} "
        f"(scores: A={score_a:.3f}, B={score_b:.3f}, C={score_c:.3f})"
    )
    print(f"  IS2 PASS: balanced ordering: {ids_in_order} "
          f"(inject_scores: A={score_a:.3f}, B={score_b:.3f}, C={score_c:.3f})")


def test_is3_design_constraint_not_reordered():
    """IS3: design_constraint chunk 不参与 inject_sort，始终排在普通 chunk 之前。"""
    top_k = [
        (0.9, {"id": "normal_high", "summary": "High score normal", "importance": 0.8, "chunk_type": "decision"}),
        (0.5, {"id": "constraint", "summary": "Design constraint", "importance": 0.2, "chunk_type": "design_constraint"}),
        (0.1, {"id": "normal_low", "summary": "Low score normal", "importance": 0.1, "chunk_type": "decision"}),
    ]

    sorted_top_k = _apply_inject_sort(top_k)
    first_id = sorted_top_k[0][1]["id"]

    assert first_id == "constraint", (
        f"IS3: design_constraint 应始终排第一, got first={first_id}"
    )
    # 普通 chunk 按 inject_score 排序
    normal_ids = [c["id"] for _, c in sorted_top_k if c.get("chunk_type") != "design_constraint"]
    assert normal_ids[0] == "normal_high", (
        f"IS3: 普通 chunk 中高 inject_score 的排前面, got {normal_ids}"
    )
    print(f"  IS3 PASS: constraint first, normal sorted: {[c['id'] for _, c in sorted_top_k]}")


def _apply_inject_sort_with_filter(top_k: list, min_ratio: float = 0.10) -> list:
    """模拟 iter472+iter475：排序 + min_ratio 过滤。"""
    constraints = [(s, c) for s, c in top_k if c.get("chunk_type") == "design_constraint"]
    normal = [(s, c) for s, c in top_k if c.get("chunk_type") != "design_constraint"]
    if len(normal) < 2:
        return top_k
    scored = [(s, c, s * math.sqrt(max(0.01, float(c.get("importance") or 0.5))))
              for s, c in normal]
    scored_sorted = sorted(scored, key=lambda x: x[2], reverse=True)
    if scored_sorted and min_ratio > 0:
        max_score = scored_sorted[0][2]
        threshold = max_score * min_ratio
        scored_sorted = [item for item in scored_sorted if item[2] >= threshold]
    return constraints + [(s, c) for s, c, _ in scored_sorted]


def test_is4_low_inject_score_filtered():
    """IS4: inject_score 远低于最高分的 chunk 被过滤（min_ratio=0.10）。"""
    # chunk A: score=0.9, importance=0.8 → inject_score = 0.9 × sqrt(0.8) ≈ 0.805
    # chunk B: score=0.8, importance=0.7 → inject_score = 0.8 × sqrt(0.7) ≈ 0.670
    # chunk C: score=0.01, importance=0.1 → inject_score = 0.01 × sqrt(0.1) ≈ 0.003
    # threshold = 0.805 × 0.10 = 0.0805 → chunk C (0.003) 被过滤
    top_k = [
        (0.9, {"id": "A", "summary": "High score", "importance": 0.8, "chunk_type": "decision"}),
        (0.8, {"id": "B", "summary": "Mid score", "importance": 0.7, "chunk_type": "decision"}),
        (0.01, {"id": "C", "summary": "Near-zero score low imp", "importance": 0.1, "chunk_type": "decision"}),
    ]

    filtered = _apply_inject_sort_with_filter(top_k, min_ratio=0.10)
    ids = [c["id"] for _, c in filtered]

    assert "C" not in ids, f"IS4: 低 inject_score 的 chunk C 应被过滤, got {ids}"
    assert "A" in ids and "B" in ids, f"IS4: 正常 chunk A/B 应保留, got {ids}"
    print(f"  IS4 PASS: low inject_score chunk filtered, remaining: {ids}")


def test_is5_all_chunks_above_threshold_kept():
    """IS5: 所有 chunk inject_score 差距不大时，全部保留。"""
    top_k = [
        (0.5, {"id": "X", "summary": "X", "importance": 0.6, "chunk_type": "decision"}),
        (0.4, {"id": "Y", "summary": "Y", "importance": 0.5, "chunk_type": "decision"}),
        (0.3, {"id": "Z", "summary": "Z", "importance": 0.4, "chunk_type": "decision"}),
    ]
    # inject_scores: X≈0.387, Y≈0.283, Z≈0.190
    # max=0.387, threshold=0.0387 — Z (0.190) > threshold → 全保留
    filtered = _apply_inject_sort_with_filter(top_k, min_ratio=0.10)
    ids = [c["id"] for _, c in filtered]
    assert set(ids) == {"X", "Y", "Z"}, f"IS5: 差距不大时应全保留, got {ids}"
    print(f"  IS5 PASS: all chunks kept when scores are close: {ids}")


if __name__ == "__main__":
    print("inject_score 加权排序测试（iter472+475）")
    print("=" * 60)

    tests = [
        test_is1_same_score_high_importance_first,
        test_is2_score_importance_balance,
        test_is3_design_constraint_not_reordered,
        test_is4_low_inject_score_filtered,
        test_is5_all_chunks_above_threshold_kept,
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
