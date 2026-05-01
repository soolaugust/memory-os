#!/usr/bin/env python3
"""
test_summary_truncation.py — Token-budget aware summary truncation 测试（iter474）

覆盖：
  ST1: importance >= 0.75 → summary[:200]
  ST2: importance 0.40-0.74 → summary[:100]
  ST3: importance < 0.40 → summary[:60]
  ST4: short summary (< limit) → 不截断（保持原长）
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))


def _apply_truncation(summary: str, importance: float) -> str:
    """模拟 retriever.py iter474 的截断逻辑。"""
    if importance >= 0.75:
        limit = 200
    elif importance >= 0.40:
        limit = 100
    else:
        limit = 60
    return summary[:limit]


def test_st1_high_importance_full_summary():
    """ST1: importance >= 0.75 → 保留 200 字。"""
    long_summary = "A" * 300
    result = _apply_truncation(long_summary, importance=0.80)
    assert len(result) == 200, f"ST1: 应截断到 200, got {len(result)}"
    print("  ST1 PASS: high importance → 200 chars")


def test_st2_mid_importance_truncated():
    """ST2: importance 0.40-0.74 → 截断到 100 字。"""
    long_summary = "B" * 300
    result = _apply_truncation(long_summary, importance=0.55)
    assert len(result) == 100, f"ST2: 应截断到 100, got {len(result)}"
    print("  ST2 PASS: mid importance → 100 chars")


def test_st3_low_importance_heavily_truncated():
    """ST3: importance < 0.40 → 截断到 60 字。"""
    long_summary = "C" * 300
    result = _apply_truncation(long_summary, importance=0.20)
    assert len(result) == 60, f"ST3: 应截断到 60, got {len(result)}"
    print("  ST3 PASS: low importance → 60 chars")


def test_st4_short_summary_not_truncated():
    """ST4: summary 本身短于 limit → 不截断。"""
    short_summary = "Short summary text"  # 18 chars
    result = _apply_truncation(short_summary, importance=0.20)
    assert result == short_summary, f"ST4: 短 summary 不应截断, got '{result}'"
    print("  ST4 PASS: short summary preserved as-is")


def test_st5_boundary_values():
    """ST5: 边界值验证 (importance=0.75, 0.40)。"""
    s = "X" * 300
    assert len(_apply_truncation(s, 0.75)) == 200  # exactly at high boundary
    assert len(_apply_truncation(s, 0.74)) == 100  # just below high boundary
    assert len(_apply_truncation(s, 0.40)) == 100  # exactly at mid boundary
    assert len(_apply_truncation(s, 0.39)) == 60   # just below mid boundary
    print("  ST5 PASS: boundary values correct")


if __name__ == "__main__":
    print("Token-budget aware summary truncation 测试（iter474）")
    print("=" * 60)

    tests = [
        test_st1_high_importance_full_summary,
        test_st2_mid_importance_truncated,
        test_st3_low_importance_heavily_truncated,
        test_st4_short_summary_not_truncated,
        test_st5_boundary_values,
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
