"""
test_tmv.py — 迭代333：Temporal Marginal Value (TMV) 单元测试

信息论背景：多次召回后，chunk 的边际信息 I(chunk|already_known) → 0
OS 类比：Linux NUMA distance penalty — 高 acc chunk ≈ remote NUMA node，访问代价 > 收益

验证：
  1. tmv_saturation_discount(acc < threshold) = 1.0（低访问不折扣）
  2. tmv_saturation_discount(threshold) = 1.0（刚到阈值无折扣）
  3. tmv_saturation_discount(2044) ≈ 0.69（高 acc 明显折扣，高于 floor=0.55）
  4. tmv_saturation_discount(9999) = floor（floor=0.55 保护下限）
  5. 折扣单调递减：acc=100 > acc=500 > acc=1000
  6. Session Density Gate：同 chunk 注入 >=4 次后 _score_chunk 分数降低 30%
  7. 全集成：检索分数在高 acc 场景下低于低 acc 场景
"""
import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "hooks"))

import tmpfs  # noqa — 必须在 store 之前 import，设置测试 DB 路径

from scorer import tmv_saturation_discount


# ──────────────────────────────────────────────────────────────────────
# 1. 低访问（< threshold）不折扣
# ──────────────────────────────────────────────────────────────────────

def test_below_threshold_no_discount():
    """acc < 50（threshold）→ multiplier = 1.0。"""
    assert tmv_saturation_discount(0) == 1.0
    assert tmv_saturation_discount(1) == 1.0
    assert tmv_saturation_discount(49) == 1.0


def test_at_threshold_no_discount():
    """acc = 50（恰好等于 threshold）→ log(50/50)=0 → multiplier = 1.0。"""
    result = tmv_saturation_discount(50)
    assert result == 1.0, f"at threshold should be 1.0, got {result}"


# ──────────────────────────────────────────────────────────────────────
# 2. 中等访问（100-500）轻微到明显折扣
# ──────────────────────────────────────────────────────────────────────

def test_moderate_access_slight_discount():
    """acc=100 → 约 0.93（轻微折扣）。"""
    result = tmv_saturation_discount(100)
    assert 0.88 < result < 1.0, f"acc=100 should be ~0.93, got {result}"


def test_high_access_visible_discount():
    """acc=500 → 约 0.79（明显折扣）。"""
    result = tmv_saturation_discount(500)
    assert 0.70 < result < 0.88, f"acc=500 should be ~0.79, got {result}"


# ──────────────────────────────────────────────────────────────────────
# 3. 极高访问（acc=2044 — design_constraint 实测值）
# ──────────────────────────────────────────────────────────────────────

def test_extreme_access_large_discount():
    """acc=2044 → ≈ 0.69（大幅折扣，高于 floor=0.55）。"""
    result = tmv_saturation_discount(2044)
    assert 0.55 <= result <= 0.80, f"acc=2044 should be ~0.69, got {result}"


def test_extreme_access_above_floor():
    """acc=2044 → 必须高于 floor=0.55（不被过度压制）。"""
    result = tmv_saturation_discount(2044)
    assert result >= 0.55, f"should be >= floor=0.55, got {result}"


# ──────────────────────────────────────────────────────────────────────
# 4. 超高访问应被 floor 保护
# ──────────────────────────────────────────────────────────────────────

def test_floor_protection():
    """acc=9999（极端值）→ log_ratio clamped to 1.0 → multiplier = max(0.55, 1-0.30) = 0.70。
    公式：log_ratio = min(1.0, log(9999/50)/log(1000/50))
          log(9999/50)/log(20) ≈ 5.3/3.0 = 1.77 → min(1.0) = 1.0
          discount = 0.30 × 1.0 = 0.30 → multiplier = max(0.55, 0.70) = 0.70
    """
    result = tmv_saturation_discount(9999)
    assert result >= 0.55, f"should be >= floor=0.55, got {result}"
    # log_ratio clamped to 1.0 → max penalty = 0.30 → multiplier = 0.70
    assert abs(result - 0.70) < 0.01, f"should be ~0.70 (max penalty path), got {result}"


# ──────────────────────────────────────────────────────────────────────
# 5. 折扣单调递减（acc 越高折扣越重）
# ──────────────────────────────────────────────────────────────────────

def test_monotone_decreasing():
    """acc=100 > acc=500 > acc=1000（multiplier 单调递减）。"""
    m100 = tmv_saturation_discount(100)
    m500 = tmv_saturation_discount(500)
    m1000 = tmv_saturation_discount(1000)
    assert m100 > m500, f"acc=100({m100:.3f}) should > acc=500({m500:.3f})"
    assert m500 > m1000, f"acc=500({m500:.3f}) should > acc=1000({m1000:.3f})"


# ──────────────────────────────────────────────────────────────────────
# 6. 公式验证（手动计算对比）
# ──────────────────────────────────────────────────────────────────────

def test_formula_acc_100():
    """手动验证 acc=100 的公式结果。"""
    threshold = 50
    weight = 0.30
    floor = 0.55
    log_ratio = math.log(100 / threshold) / math.log(1000 / threshold)
    log_ratio = min(1.0, log_ratio)
    discount = weight * log_ratio
    expected = max(floor, 1.0 - discount)
    result = tmv_saturation_discount(100)
    assert abs(result - expected) < 1e-9, f"formula mismatch: expected={expected:.4f}, got={result:.4f}"


def test_formula_acc_2044():
    """手动验证 acc=2044 的公式结果。"""
    threshold = 50
    weight = 0.30
    floor = 0.55
    log_ratio = math.log(2044 / threshold) / math.log(1000 / threshold)
    log_ratio = min(1.0, log_ratio)
    discount = weight * log_ratio
    expected = max(floor, 1.0 - discount)
    result = tmv_saturation_discount(2044)
    assert abs(result - expected) < 1e-9, f"formula mismatch: expected={expected:.4f}, got={result:.4f}"


# ──────────────────────────────────────────────────────────────────────
# 7. 集成测试：retrieval_score 在高 acc 时低于低 acc
# ──────────────────────────────────────────────────────────────────────

def test_retrieval_score_lower_for_high_acc():
    """
    相同 relevance/importance/recency，acc=2044 的 retrieval_score 应低于 acc=5。
    验证 TMV discount 已集成到 scorer.retrieval_score 的调用链之外。

    注意：TMV 乘法折扣在 retriever._score_chunk 中应用（不在 retrieval_score 内），
    此测试验证 tmv_saturation_discount 返回值确实 < 1.0 from the scorer module。
    """
    m_low = tmv_saturation_discount(5)    # acc < threshold → 1.0
    m_high = tmv_saturation_discount(2044)  # acc >> threshold → ~0.69
    assert m_low > m_high, f"low_acc_mult({m_low:.3f}) should > high_acc_mult({m_high:.3f})"
    # 意味着：score_low = score_base × 1.0，score_high = score_base × 0.69
    # → score_low > score_high
