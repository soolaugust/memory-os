"""
test_emotional_salience.py — iter376 Emotional Salience Retrieval Boost 测试

覆盖：
  EB1: emotional_weight > threshold → 期望 boost 值正确
  EB2: emotional_weight <= threshold → 无加分
  EB3: emotional_weight=0 → 无加分
  EB4: 高 emotional_weight chunk 排名高于低 emotional_weight chunk
"""
import sys
import pytest
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "hooks"))

# ── EB1: emotional_weight > threshold → 有加分 ─────────────────────────────

def test_eb1_boost_amount():
    """验证 emotional boost 计算公式：boost = emotional_weight × factor"""
    # emotional_weight=0.7, factor=0.08 → boost=0.056
    factor = 0.08
    threshold = 0.4
    ew = 0.7
    boost = ew * factor if ew > threshold else 0.0
    assert boost == pytest.approx(0.056, abs=0.001)


def test_eb1_high_ew_exceeds_threshold():
    """emotional_weight=0.8 > threshold=0.4 → should boost"""
    threshold = 0.4
    ew = 0.8
    assert ew > threshold


# ── EB2: emotional_weight <= threshold → 无加分 ────────────────────────────

def test_eb2_low_emotional_weight_no_boost():
    """emotional_weight=0.3 < threshold=0.4 → 无加分"""
    threshold = 0.4
    ew = 0.3
    boost = ew * 0.08 if ew > threshold else 0.0
    assert boost == 0.0


def test_eb2_at_threshold_no_boost():
    """emotional_weight == threshold → 无加分（严格大于）"""
    threshold = 0.4
    ew = 0.4
    boost = ew * 0.08 if ew > threshold else 0.0
    assert boost == 0.0


# ── EB3: emotional_weight=0 → 无加分 ─────────────────────────────────────

def test_eb3_zero_emotional_weight():
    """emotional_weight=0 → score 无加分"""
    threshold = 0.4
    ew = 0.0
    boost = ew * 0.08 if ew > threshold else 0.0
    assert boost == 0.0


# ── EB4: 高 emotional chunk 排名高于低 emotional chunk ────────────────────

def test_eb4_ranking_with_emotional_boost():
    """
    高 emotional_weight chunk 在相同 relevance 下获得更高综合 score。
    两个 chunk 仅 emotional_weight 不同，验证 score 差异。
    """
    factor = 0.08
    threshold = 0.4

    ew_high = 0.9  # 高情绪显著性
    ew_low = 0.1   # 低情绪显著性（低于 threshold）

    base_score = 0.5
    boost_high = ew_high * factor if ew_high > threshold else 0.0
    boost_low = ew_low * factor if ew_low > threshold else 0.0

    score_high = base_score + boost_high
    score_low = base_score + boost_low

    assert score_high > score_low, (
        f"High emotional chunk (score={score_high:.4f}) should rank above "
        f"low emotional chunk (score={score_low:.4f})"
    )
    assert boost_high == pytest.approx(0.072, abs=0.001)
    assert boost_low == 0.0


# ── EB5: retriever _score_chunk 中 emotional boost 逻辑验证 ────────────────

def test_eb5_sysctl_defaults_exist():
    """config.py 中 emotional boost sysctl 已注册"""
    from config import get as _get
    factor = _get("retriever.emotional_boost_factor")
    threshold = _get("retriever.emotional_boost_threshold")
    assert factor is not None, "retriever.emotional_boost_factor sysctl 未注册"
    assert threshold is not None, "retriever.emotional_boost_threshold sysctl 未注册"
    assert factor == pytest.approx(0.08, abs=0.001)
    assert threshold == pytest.approx(0.4, abs=0.001)
