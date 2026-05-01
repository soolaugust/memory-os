#!/usr/bin/env python3
"""
test_adaptive_k.py — Adaptive K + Citation Stats 测试

覆盖：
  AK1: citation_rate 低 (<30%) → effective_top_k 缩小
  AK2: citation_rate 高 (>65%) → effective_top_k 扩大
  AK3: citation_rate 中间 (30-65%) → top_k 不变
  AK4: _update_citation_stats 写文件 + get_citation_rate 正确读取
  AK5: 滑动窗口 WINDOW_SIZE=20 上限
  AK6: 无 stats 文件时 get_citation_rate 返回默认 0.5
"""
import sys
import json
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import tmpfs  # noqa


def test_ak4_update_and_read_citation_stats():
    """AK4: _update_citation_stats 写文件，get_citation_rate 正确读回。"""
    from tools.citation_detector import _update_citation_stats, get_citation_rate, _MEMORY_OS_DIR

    proj = "ak_test_proj_ak4"
    proj_safe = proj.replace("/", "_").replace(":", "_")[:40]
    stats_file = _MEMORY_OS_DIR / f"citation_stats.{proj_safe}.json"

    # 清理
    if stats_file.exists():
        stats_file.unlink()

    # 写入：cited=3, total=5 → rate=0.6
    _update_citation_stats(proj, 3, 5)

    rate = get_citation_rate(proj)
    assert 0.55 <= rate <= 0.65, f"AK4: expected ~0.6, got {rate}"

    # 清理
    if stats_file.exists():
        stats_file.unlink()
    print(f"  AK4 PASS: citation_rate={rate:.3f}")


def test_ak5_sliding_window():
    """AK5: 滑动窗口保持 <= 20 条。"""
    from tools.citation_detector import _update_citation_stats, get_citation_rate, _MEMORY_OS_DIR

    proj = "ak_test_proj_ak5"
    proj_safe = proj.replace("/", "_").replace(":", "_")[:40]
    stats_file = _MEMORY_OS_DIR / f"citation_stats.{proj_safe}.json"

    if stats_file.exists():
        stats_file.unlink()

    # 写入 25 次（全部 cited=0，rate=0）
    for _ in range(25):
        _update_citation_stats(proj, 0, 1)

    data = json.loads(stats_file.read_text(encoding="utf-8"))
    window = data["window"]
    assert len(window) <= 20, f"AK5: window should be <=20, got {len(window)}"
    rate = get_citation_rate(proj)
    assert rate < 0.1, f"AK5: all-zero window should give low rate, got {rate}"

    if stats_file.exists():
        stats_file.unlink()
    print(f"  AK5 PASS: window_len={len(window)}, rate={rate:.3f}")


def test_ak6_no_stats_file_returns_default():
    """AK6: 无 stats 文件时 get_citation_rate 返回默认 0.5。"""
    from tools.citation_detector import get_citation_rate, _MEMORY_OS_DIR

    proj = "ak_nonexistent_proj_xyz_ak6"
    proj_safe = proj.replace("/", "_").replace(":", "_")[:40]
    stats_file = _MEMORY_OS_DIR / f"citation_stats.{proj_safe}.json"
    if stats_file.exists():
        stats_file.unlink()

    rate = get_citation_rate(proj)
    assert rate == 0.5, f"AK6: expected 0.5 default, got {rate}"
    print(f"  AK6 PASS: default rate={rate}")


def test_ak1_low_citation_rate_shrinks_top_k():
    """AK1: citation_rate < 30% → effective_top_k 比基准小。"""
    from tools.citation_detector import _update_citation_stats, _MEMORY_OS_DIR

    proj = "ak_test_low_ak1"
    proj_safe = proj.replace("/", "_").replace(":", "_")[:40]
    stats_file = _MEMORY_OS_DIR / f"citation_stats.{proj_safe}.json"

    if stats_file.exists():
        stats_file.unlink()

    # 写入低命中率（cited=0, total=5 → rate=0）
    for _ in range(10):
        _update_citation_stats(proj, 0, 5)

    rate_after = json.loads(stats_file.read_text(encoding="utf-8"))["citation_rate"]
    assert rate_after < 0.3, f"AK1 setup: rate should be <0.3, got {rate_after}"

    # 模拟 retriever 中的 adaptive_k 逻辑
    base_top_k = 5
    effective_top_k = base_top_k
    ak_min = max(2, base_top_k - 2)
    if rate_after < 0.30:
        effective_top_k = max(ak_min, effective_top_k - 1)

    assert effective_top_k < base_top_k, (
        f"AK1: low citation rate should shrink top_k, {base_top_k}→{effective_top_k}"
    )

    if stats_file.exists():
        stats_file.unlink()
    print(f"  AK1 PASS: rate={rate_after:.2f}, top_k {base_top_k}→{effective_top_k}")


def test_ak2_high_citation_rate_grows_top_k():
    """AK2: citation_rate > 65% → effective_top_k 比基准大。"""
    from tools.citation_detector import _update_citation_stats, _MEMORY_OS_DIR

    proj = "ak_test_high_ak2"
    proj_safe = proj.replace("/", "_").replace(":", "_")[:40]
    stats_file = _MEMORY_OS_DIR / f"citation_stats.{proj_safe}.json"

    if stats_file.exists():
        stats_file.unlink()

    # 写入高命中率（cited=5, total=5 → rate=1.0）
    for _ in range(10):
        _update_citation_stats(proj, 5, 5)

    rate_after = json.loads(stats_file.read_text(encoding="utf-8"))["citation_rate"]
    assert rate_after > 0.65, f"AK2 setup: rate should be >0.65, got {rate_after}"

    # 模拟 retriever adaptive_k 逻辑
    base_top_k = 5
    effective_top_k = base_top_k
    ak_max = min(10, base_top_k + 3)
    if rate_after > 0.65:
        effective_top_k = min(ak_max, effective_top_k + 2)

    assert effective_top_k > base_top_k, (
        f"AK2: high citation rate should grow top_k, {base_top_k}→{effective_top_k}"
    )

    if stats_file.exists():
        stats_file.unlink()
    print(f"  AK2 PASS: rate={rate_after:.2f}, top_k {base_top_k}→{effective_top_k}")


def test_ak3_medium_citation_rate_stable():
    """AK3: citation_rate 30-65% → top_k 不变。"""
    from tools.citation_detector import _update_citation_stats, _MEMORY_OS_DIR

    proj = "ak_test_med_ak3"
    proj_safe = proj.replace("/", "_").replace(":", "_")[:40]
    stats_file = _MEMORY_OS_DIR / f"citation_stats.{proj_safe}.json"

    if stats_file.exists():
        stats_file.unlink()

    # 写入中等命中率（cited=2, total=5 → rate=0.4）
    for _ in range(10):
        _update_citation_stats(proj, 2, 5)

    rate_after = json.loads(stats_file.read_text(encoding="utf-8"))["citation_rate"]
    assert 0.30 <= rate_after <= 0.65, f"AK3 setup: rate should be 0.3-0.65, got {rate_after}"

    # 模拟 retriever adaptive_k 逻辑
    base_top_k = 5
    effective_top_k = base_top_k
    if rate_after < 0.30:
        effective_top_k = max(2, effective_top_k - 1)
    elif rate_after > 0.65:
        effective_top_k = min(10, effective_top_k + 2)
    # else: 不变

    assert effective_top_k == base_top_k, (
        f"AK3: medium citation rate should keep top_k stable, {base_top_k}→{effective_top_k}"
    )

    if stats_file.exists():
        stats_file.unlink()
    print(f"  AK3 PASS: rate={rate_after:.2f}, top_k stable={effective_top_k}")


if __name__ == "__main__":
    print("Adaptive K 测试")
    print("=" * 60)

    tests = [
        test_ak4_update_and_read_citation_stats,
        test_ak5_sliding_window,
        test_ak6_no_stats_file_returns_default,
        test_ak1_low_citation_rate_shrinks_top_k,
        test_ak2_high_citation_rate_grows_top_k,
        test_ak3_medium_citation_rate_stable,
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
