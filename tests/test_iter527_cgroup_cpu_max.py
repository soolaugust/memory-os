"""
iter527: cgroup_cpu_max — Per-Chunk Recall Bandwidth Cap
OS 类比：Linux cgroup v2 cpu.max (Tejun Heo, 2015) — 硬性 CPU 带宽限制

测试 bandwidth_throttle() 函数和其在 retrieval_score 中的集成效果。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tmpfs  # noqa: E402 — 测试隔离
from scorer import bandwidth_throttle, retrieval_score, saturation_penalty
from config import get as sysctl


def test_below_threshold_returns_1():
    """recall_count 未超 bw_max_pct 时返回 1.0（不 throttle）"""
    # bw_max_pct=0.30, window=30 → 阈值 = 0.30 * 30 = 9 次
    assert bandwidth_throttle(0) == 1.0
    assert bandwidth_throttle(5) == 1.0
    assert bandwidth_throttle(9) == 1.0  # 9/30=0.30, 刚好等于阈值 → 不触发


def test_above_threshold_returns_throttle():
    """recall_count 超 bw_max_pct 时返回 bw_throttle 因子"""
    # 10/30=0.333 > 0.30 → 触发
    result = bandwidth_throttle(10)
    assert result == 0.15, f"expected 0.15, got {result}"
    # 15/30=0.50 > 0.30 → 触发
    assert bandwidth_throttle(15) == 0.15
    # 极端：30/30=1.0 > 0.30 → 触发
    assert bandwidth_throttle(30) == 0.15


def test_custom_window():
    """支持自定义 window 参数"""
    # window=10, max_pct=0.30 → 阈值 = 3 次
    assert bandwidth_throttle(3, window=10) == 1.0   # 3/10=0.30 刚好不触发
    assert bandwidth_throttle(4, window=10) == 0.15  # 4/10=0.40 > 0.30 → 触发


def test_zero_window_uses_default():
    """window=0 时使用默认 _BW_WINDOW（不崩溃，fallback 到默认窗口）"""
    # window=0 → 使用默认 _BW_WINDOW=30, 10/30=0.333>0.30 → throttle
    assert bandwidth_throttle(10, window=0) == 0.15
    # window=0 + 低 recall → 不触发
    assert bandwidth_throttle(5, window=0) == 1.0


def test_negative_recall_count():
    """负数 recall_count 安全返回 1.0"""
    assert bandwidth_throttle(-5) == 1.0


def test_integration_with_retrieval_score():
    """bandwidth_throttle 在 retrieval_score 中的乘法效果"""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()

    # 基线：recall_count=5（未超 bandwidth）
    score_normal = retrieval_score(
        relevance=0.8, importance=0.9, last_accessed=now,
        access_count=10, created_at=now, recall_count=5
    )

    # 超 bandwidth：recall_count=15（15/30=0.50 > 0.30）
    score_throttled = retrieval_score(
        relevance=0.8, importance=0.9, last_accessed=now,
        access_count=10, created_at=now, recall_count=15
    )

    # throttled 分数应该是 normal 的约 15%（因为 bw_throttle=0.15）
    # 注意 saturation_penalty 也会影响两者差异，所以不是精确 15%
    assert score_throttled < score_normal * 0.25, \
        f"throttled={score_throttled:.4f} should be << normal={score_normal:.4f}"
    assert score_throttled > 0, "throttled score should still be positive"


def test_throttle_harder_than_saturation():
    """bandwidth_throttle 的削减效果远强于 saturation_penalty"""
    # saturation_penalty at recall_count=15: ~0.16 (加法)
    sp = saturation_penalty(15)
    # bandwidth_throttle at recall_count=15: 0.15 (乘法，削减 85%)
    bw = bandwidth_throttle(15)

    # 对于一个 score=1.0 的 chunk：
    # saturation 只减少 0.16 → 0.84
    # bandwidth 乘以 0.15 → 0.15
    # bandwidth 效果远强于 saturation
    effective_sat = 1.0 - sp  # ~0.84
    effective_bw = 1.0 * bw   # 0.15
    assert effective_bw < effective_sat * 0.5, \
        f"bw={effective_bw:.2f} should be << sat={effective_sat:.2f}"


def test_monopoly_chunk_suppressed():
    """模拟生产场景：高 importance + 高 access + 高 recall 的 chunk 被有效抑制"""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()

    # 垄断 chunk：imp=0.95, access=89, recall=13/30 (43%)
    score_monopoly = retrieval_score(
        relevance=0.9, importance=0.95, last_accessed=now,
        access_count=89, created_at=now, recall_count=13
    )

    # 新鲜 chunk：imp=0.7, access=2, recall=1/30 (3%)
    score_fresh = retrieval_score(
        relevance=0.7, importance=0.70, last_accessed=now,
        access_count=2, created_at=now, recall_count=1
    )

    # 有了 bandwidth_throttle，垄断 chunk 应该不再高于新鲜 chunk
    assert score_monopoly < score_fresh, \
        f"monopoly={score_monopoly:.4f} should be < fresh={score_fresh:.4f}"


def test_boundary_precision():
    """精确边界测试：recall_count/window 刚好等于 bw_max_pct"""
    # 9/30 = 0.30 = bw_max_pct → 不触发（<= 不超）
    assert bandwidth_throttle(9, window=30) == 1.0
    # 10/30 = 0.333 > 0.30 → 触发
    assert bandwidth_throttle(10, window=30) == 0.15


def test_sysctl_tunables_exist():
    """验证 config.py 中新增的 tunables 可读"""
    assert sysctl("scorer.bw_max_pct") == 0.30
    assert sysctl("scorer.bw_throttle") == 0.15
    assert sysctl("scorer.bw_window") == 30


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
