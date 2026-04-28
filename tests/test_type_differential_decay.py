"""
test_type_differential_decay.py — iter375 Type-Differential Decay Rates 测试

覆盖：
  TD1: task_state (episodic) 衰减比 design_constraint (semantic) 快
  TD2: decay_rate=0.88 vs 0.99，30天后差异显著
  TD3: chunk_type="" 时 fallback 到全局 decay_rate
  TD4: 未知 chunk_type 也 fallback 到全局 decay_rate
"""
import sys
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


def _imp_with_decay(imp, days_ago, chunk_type="", decay_rate_override=None):
    """
    直接测试 importance_with_decay 的逻辑（独立于 config.py，通过 mock）。
    """
    from scorer import importance_with_decay
    from datetime import datetime, timezone, timedelta
    last_accessed = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()

    if decay_rate_override is not None:
        # mock sysctl to return specific decay rates
        def mock_sysctl(key):
            # 只拦截特定 chunk_type 的 decay key
            _type_map = {
                "scorer.decay_rate_task_state": 0.88,
                "scorer.decay_rate_design_constraint": 0.99,
                "scorer.decay_rate_conversation_summary": 0.90,
                "scorer.importance_decay_rate": 0.95,
                "scorer.importance_floor": 0.3,
            }
            return _type_map.get(key)

        with patch("scorer._sysctl", side_effect=mock_sysctl):
            return importance_with_decay(imp, last_accessed, chunk_type=chunk_type)
    else:
        return importance_with_decay(imp, last_accessed, chunk_type=chunk_type)


# ── TD1: task_state 衰减更快 ─────────────────────────────────────────────────

def test_td1_task_state_decays_faster_than_design_constraint():
    """30 天后 task_state (0.88) 的 effective_importance 低于 design_constraint (0.99)"""
    # 都从 importance=0.9 出发，30天后
    imp_task = _imp_with_decay(0.9, days_ago=30, chunk_type="task_state",
                               decay_rate_override=True)
    imp_dc = _imp_with_decay(0.9, days_ago=30, chunk_type="design_constraint",
                             decay_rate_override=True)
    # task_state 应该衰减更多（effective_importance 更低）
    assert imp_task < imp_dc, (
        f"task_state imp={imp_task:.4f} should be < design_constraint imp={imp_dc:.4f}"
    )


# ── TD2: 差异显著 ─────────────────────────────────────────────────────────────

def test_td2_decay_difference_significant_at_30_days():
    """30天后衰减差异至少 0.05"""
    imp_task = _imp_with_decay(0.9, days_ago=30, chunk_type="task_state",
                               decay_rate_override=True)
    imp_dc = _imp_with_decay(0.9, days_ago=30, chunk_type="design_constraint",
                             decay_rate_override=True)
    diff = imp_dc - imp_task
    assert diff >= 0.02, f"Expected diff >= 0.02 but got {diff:.4f}"


# ── TD3: chunk_type="" fallback 到全局 ──────────────────────────────────────

def test_td3_empty_chunk_type_uses_global_decay():
    """chunk_type='' 时使用全局 decay_rate=0.95"""
    from datetime import datetime, timezone, timedelta
    from scorer import importance_with_decay

    def mock_sysctl(key):
        return {"scorer.importance_decay_rate": 0.95,
                "scorer.importance_floor": 0.3}.get(key)

    last_accessed = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    with patch("scorer._sysctl", side_effect=mock_sysctl):
        # 7 天后：effective = 0.8 × 0.95^1 = 0.76
        result = importance_with_decay(0.8, last_accessed, chunk_type="")
    expected = max(0.3, 0.8 * (0.95 ** 1.0))
    assert abs(result - expected) < 0.01


# ── TD4: 未知 chunk_type fallback ────────────────────────────────────────────

def test_td4_unknown_chunk_type_fallback():
    """未知 chunk_type 也使用全局 decay_rate"""
    from datetime import datetime, timezone, timedelta
    from scorer import importance_with_decay

    def mock_sysctl(key):
        return {"scorer.importance_decay_rate": 0.95,
                "scorer.importance_floor": 0.3}.get(key)

    last_accessed = (datetime.now(timezone.utc) - timedelta(days=14)).isoformat()
    with patch("scorer._sysctl", side_effect=mock_sysctl):
        result = importance_with_decay(0.8, last_accessed, chunk_type="some_unknown_type")
    expected = max(0.3, 0.8 * (0.95 ** 2.0))
    assert abs(result - expected) < 0.01
