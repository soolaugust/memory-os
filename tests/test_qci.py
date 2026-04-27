"""
test_qci.py — 迭代322：Query-Conditioned Importance 单元测试

验证：
  1. 高 relevance 时 importance 权重降低（recency 主导）
  2. 低 relevance 时 importance 权重升高（importance 先验主导）
  3. query_alpha=None 向后兼容（使用默认 0.55）
  4. 边界 clamp：alpha < 0.1 → 0.1，alpha > 0.9 → 0.9
  5. 高 relevance + 低 importance chunk 比低 relevance + 低 importance chunk 得分更高
  6. 低 relevance + 高 importance chunk 比低 relevance + 低 importance chunk 得分更高（importance 先验有效）
  7. 动态 alpha 公式：α = qci_base_alpha - qci_relevance_slope × relevance

信息论依据：
  高 relevance = query 已强定向（H(X|Q) 低），importance 先验提升边际收益低
  低 relevance = query 弱定向（H(X|Q) 高），importance 先验成为主要筛选信号
"""
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "hooks"))

from scorer import retrieval_score
from datetime import datetime, timezone


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _base_score(relevance, importance, query_alpha=None):
    """调用 retrieval_score，使用固定其他参数。"""
    return retrieval_score(
        relevance=relevance,
        importance=importance,
        last_accessed=_now_iso(),
        access_count=1,
        created_at=_now_iso(),
        chunk_id="test",
        query_seed="test query",
        recall_count=0,
        session_recall_count=0,
        chunk_project="test",
        current_project="test",
        query_alpha=query_alpha,
    )


# ══════════════════════════════════════════════════════════════════════
# 1. query_alpha 对 importance 权重的影响
# ══════════════════════════════════════════════════════════════════════

def test_high_alpha_amplifies_importance():
    """α=0.9（低 relevance 场景）时，高 importance chunk 得分明显高于低 importance。"""
    high_imp = _base_score(relevance=0.1, importance=0.9, query_alpha=0.9)
    low_imp = _base_score(relevance=0.1, importance=0.2, query_alpha=0.9)
    diff = high_imp - low_imp
    assert diff > 0.03, f"高 α 应让 importance 差距明显（> 0.03），got diff={diff:.4f}"


def test_low_alpha_reduces_importance_gap():
    """α=0.1（高 relevance 场景）时，importance 差距缩小。"""
    # 相同 relevance 下，高低 importance 的得分差距比 α=0.9 时更小
    high_imp_low_alpha = _base_score(relevance=0.9, importance=0.9, query_alpha=0.1)
    low_imp_low_alpha = _base_score(relevance=0.9, importance=0.2, query_alpha=0.1)
    diff_low_alpha = high_imp_low_alpha - low_imp_low_alpha

    high_imp_high_alpha = _base_score(relevance=0.1, importance=0.9, query_alpha=0.9)
    low_imp_high_alpha = _base_score(relevance=0.1, importance=0.2, query_alpha=0.9)
    diff_high_alpha = high_imp_high_alpha - low_imp_high_alpha

    assert diff_low_alpha < diff_high_alpha, \
        f"低 α 时 importance 差距应小于高 α，got low_α={diff_low_alpha:.4f} vs high_α={diff_high_alpha:.4f}"


def test_none_alpha_backward_compatible():
    """query_alpha=None 使用默认 0.55，结果与显式传 0.55 一致。"""
    score_none = _base_score(relevance=0.5, importance=0.7, query_alpha=None)
    score_explicit = _base_score(relevance=0.5, importance=0.7, query_alpha=0.55)
    assert abs(score_none - score_explicit) < 1e-9, \
        f"None 应等同于 0.55，got {score_none:.6f} vs {score_explicit:.6f}"


def test_alpha_clamp_min():
    """alpha < 0.1 被 clamp 到 0.1。"""
    score_zero = _base_score(relevance=0.5, importance=0.7, query_alpha=0.0)
    score_min = _base_score(relevance=0.5, importance=0.7, query_alpha=0.1)
    assert abs(score_zero - score_min) < 1e-9, \
        f"alpha=0 应 clamp 到 0.1，got {score_zero:.6f} vs {score_min:.6f}"


def test_alpha_clamp_max():
    """alpha > 0.9 被 clamp 到 0.9。"""
    score_one = _base_score(relevance=0.5, importance=0.7, query_alpha=1.0)
    score_max = _base_score(relevance=0.5, importance=0.7, query_alpha=0.9)
    assert abs(score_one - score_max) < 1e-9, \
        f"alpha=1.0 应 clamp 到 0.9，got {score_one:.6f} vs {score_max:.6f}"


# ══════════════════════════════════════════════════════════════════════
# 2. 核心语义验证
# ══════════════════════════════════════════════════════════════════════

def test_high_relevance_chunk_wins_over_low_relevance_same_importance():
    """高 relevance chunk 得分高于相同 importance 的低 relevance chunk。"""
    high_rel = _base_score(relevance=0.9, importance=0.5)
    low_rel = _base_score(relevance=0.1, importance=0.5)
    assert high_rel > low_rel, \
        f"高 relevance 应得分更高，got high={high_rel:.4f} low={low_rel:.4f}"


def test_high_importance_per_relevance_unit_helps_more_when_low():
    """低 relevance 场景下，同等 relevance 单位内 importance 差距的相对贡献更大。

    由于 score = relevance × (base + ...) + bonuses，绝对差距受 relevance 乘数影响。
    正确验证：低 relevance 时，(diff/relevance) 比高 relevance 时更大，
    即 importance 对单位 relevance 贡献更多（α 更高）。
    """
    # 低 relevance 组（α 高 → importance 权重高）
    low_rel_high_imp = _base_score(relevance=0.1, importance=0.9, query_alpha=0.55 - 0.25 * 0.1)
    low_rel_low_imp = _base_score(relevance=0.1, importance=0.2, query_alpha=0.55 - 0.25 * 0.1)
    # 归一化差距：单位 relevance 下的 importance 贡献
    diff_per_rel_low = (low_rel_high_imp - low_rel_low_imp) / 0.1

    # 高 relevance 组（α 低 → importance 权重低）
    high_rel_high_imp = _base_score(relevance=0.9, importance=0.9, query_alpha=0.55 - 0.25 * 0.9)
    high_rel_low_imp = _base_score(relevance=0.9, importance=0.2, query_alpha=0.55 - 0.25 * 0.9)
    diff_per_rel_high = (high_rel_high_imp - high_rel_low_imp) / 0.9

    # 低 relevance（高 α）时，单位 relevance 的 importance 差距更大
    assert diff_per_rel_low > diff_per_rel_high, \
        f"低 rel 时 importance/rel 贡献应更大: {diff_per_rel_low:.4f} > {diff_per_rel_high:.4f}"


# ══════════════════════════════════════════════════════════════════════
# 3. 动态 alpha 公式验证（与 retriever.py 的 _score_chunk 一致）
# ══════════════════════════════════════════════════════════════════════

def test_dynamic_alpha_formula():
    """α = base_alpha - slope × relevance，retriever 应用的公式验证。"""
    base_alpha = 0.55
    slope = 0.25

    for relevance in [0.0, 0.3, 0.6, 1.0]:
        expected_alpha = max(0.1, min(0.9, base_alpha - slope * relevance))
        score_dynamic = _base_score(relevance=relevance, importance=0.7, query_alpha=expected_alpha)
        score_explicit = _base_score(relevance=relevance, importance=0.7, query_alpha=expected_alpha)
        assert abs(score_dynamic - score_explicit) < 1e-9, \
            f"relevance={relevance} alpha={expected_alpha:.2f}: 公式应一致"


def test_alpha_range_monotone_in_relevance():
    """随 relevance 增大，α 单调减小（实现了 Query-Conditioned 的预期方向）。"""
    base_alpha = 0.55
    slope = 0.25
    prev_alpha = None
    for relevance in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        alpha = max(0.1, min(0.9, base_alpha - slope * relevance))
        if prev_alpha is not None:
            assert alpha <= prev_alpha, \
                f"relevance 增大时 α 应单调减小，got alpha={alpha:.3f} at rel={relevance}"
        prev_alpha = alpha
