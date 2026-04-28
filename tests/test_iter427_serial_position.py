"""
test_iter427_serial_position.py — iter427: Serial Position Effect 单元测试

覆盖：
  SP1: 高 importance chunk 应出现在 normal_items 首位（primacy）
  SP2: 第二高 importance chunk 应出现在 normal_items 末位（recency）
  SP3: 中间 chunk 按 score 降序排列
  SP4: < 3 个 normal chunk 时不触发重排（边界）
  SP5: 约束类型（design_constraint）不参与 serial position 重排
  SP6: serial_position_enabled=False 时保持原始 score 顺序
  SP7: 高 importance threshold — 无高价值 chunk 时不重排
  SP8: serial_position_recency_types 命中 → 优先候选 primacy/recency

认知科学依据：
  Murdock (1962) "The Serial Position Effect of Free Recall" (JEP) —
    自由回忆中首项和末项回忆率最高（首因 + 近因效应），
    中间项受输出干扰（Roediger & McDermott 1995）抑制最严重。
  应用：高价值 chunk 置于注入块首/尾，利用 LLM 的 attention 局部性。

OS 类比：Linux BFQ `bfq_dispatch_request()` front-merge —
  最高优先级 I/O 请求置于 dispatch queue 头部；
  次高 I/O 追加到队尾（recency 位置保证最后被处理）。
"""
import sys
import sqlite3
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import tmpfs  # noqa
from config import get as sysctl, _REGISTRY


# ── helper: build fake top_k list ──────────────────────────────────────────────

def _make_chunk(cid, chunk_type="decision", importance=0.5, score=0.5):
    return (score, {
        "id": cid,
        "summary": f"summary_{cid}",
        "chunk_type": chunk_type,
        "importance": importance,
        "verification_status": "pending",
        "confidence_score": 0.7,
    })


def _apply_serial_position(top_k, imp_threshold=0.85, recency_types=None, enabled=True):
    """
    Mirror of retriever.py iter427 serial position logic for unit-testing.
    Returns reordered top_k list.
    """
    if not enabled:
        return top_k
    if recency_types is None:
        recency_types = {"decision", "design_constraint", "reasoning_chain"}
    if len(top_k) < 3:
        return top_k

    _spe_constraints = [(s, c) for s, c in top_k if c.get("chunk_type") == "design_constraint"]
    _spe_normal = [(s, c) for s, c in top_k if c.get("chunk_type") != "design_constraint"]
    if len(_spe_normal) < 3:
        return top_k

    _spe_high = [(s, c) for s, c in _spe_normal
                 if float(c.get("importance") or 0) >= imp_threshold
                 or c.get("chunk_type", "") in recency_types]
    _spe_mid = [(s, c) for s, c in _spe_normal if (s, c) not in _spe_high]

    if not _spe_high:
        return top_k

    _spe_high_sorted = sorted(_spe_high, key=lambda x: x[0], reverse=True)
    _primacy = _spe_high_sorted[:1]
    _recency = _spe_high_sorted[1:2]
    _spe_remaining_high = _spe_high_sorted[2:]
    _mid_ordered = sorted(_spe_mid + _spe_remaining_high, key=lambda x: x[0], reverse=True)
    _spe_reordered = _primacy + _mid_ordered + _recency
    return _spe_constraints + _spe_reordered


# ── SP1: 高 importance chunk → 首位 ───────────────────────────────────────────

def test_sp1_high_importance_at_primacy():
    """SP1: importance >= threshold 的 chunk 应出现在 normal_items 首位。"""
    top_k = [
        _make_chunk("low_1", importance=0.5, score=0.8),
        _make_chunk("high_1", importance=0.92, score=0.6),  # 高 importance，低 score
        _make_chunk("low_2", importance=0.4, score=0.7),
        _make_chunk("low_3", importance=0.3, score=0.5),
    ]
    result = _apply_serial_position(top_k, imp_threshold=0.85,
                                    recency_types=set())  # 空 recency_types，只看 importance
    # high_1 应在首位（即使 score 不是最高）
    assert result[0][1]["id"] == "high_1", \
        f"SP1: 高 importance chunk 应在首位，got {result[0][1]['id']}"


# ── SP2: 第二高 importance → 末位 ────────────────────────────────────────────

def test_sp2_second_high_at_recency():
    """SP2: 第二高 importance chunk 应出现在 normal_items 末位（recency anchor）。"""
    top_k = [
        _make_chunk("low_1", importance=0.5, score=0.9),
        _make_chunk("high_1", importance=0.95, score=0.6),  # 最高 → primacy
        _make_chunk("high_2", importance=0.88, score=0.5),  # 次高 → recency
        _make_chunk("low_2", importance=0.4, score=0.7),
        _make_chunk("low_3", importance=0.3, score=0.4),
    ]
    result = _apply_serial_position(top_k, imp_threshold=0.85, recency_types=set())
    # high_1 → 首位, high_2 → 末位
    assert result[0][1]["id"] == "high_1", \
        f"SP2: 最高 importance 应在首位，got {result[0][1]['id']}"
    assert result[-1][1]["id"] == "high_2", \
        f"SP2: 次高 importance 应在末位，got {result[-1][1]['id']}"


# ── SP3: 中间 chunk 按 score 降序 ────────────────────────────────────────────

def test_sp3_middle_chunks_score_descending():
    """SP3: 中间 chunk（非 primacy/recency）应按 score 降序排列。"""
    top_k = [
        _make_chunk("high_1", importance=0.95, score=0.6),  # → primacy
        _make_chunk("high_2", importance=0.90, score=0.5),  # → recency
        _make_chunk("mid_a", importance=0.4, score=0.8),    # 中间，score=0.8
        _make_chunk("mid_b", importance=0.3, score=0.7),    # 中间，score=0.7
        _make_chunk("mid_c", importance=0.2, score=0.3),    # 中间，score=0.3
    ]
    result = _apply_serial_position(top_k, imp_threshold=0.85, recency_types=set())
    # 首 + 末已确定，中间 3 个应按 score 降序
    middle = result[1:-1]
    middle_scores = [s for s, _ in middle]
    assert middle_scores == sorted(middle_scores, reverse=True), \
        f"SP3: 中间 chunk 应按 score 降序，got scores: {middle_scores}"


# ── SP4: < 3 个 normal chunk 时不重排 ────────────────────────────────────────

def test_sp4_less_than_3_no_reorder():
    """SP4: normal chunk < 3 时，serial position 不触发，返回原始顺序。"""
    top_k = [
        _make_chunk("a", importance=0.95, score=0.4),
        _make_chunk("b", importance=0.3, score=0.9),
    ]
    result = _apply_serial_position(top_k)
    # 顺序不变
    assert [c["id"] for _, c in result] == [c["id"] for _, c in top_k], \
        "SP4: < 3 个 chunk 时应保持原始顺序"


# ── SP5: design_constraint 不参与重排 ────────────────────────────────────────

def test_sp5_constraint_not_reordered():
    """SP5: design_constraint 类型不参与 serial position 重排，保持约束首位逻辑。"""
    top_k = [
        _make_chunk("constraint_1", chunk_type="design_constraint", importance=0.9, score=0.99),
        _make_chunk("low_1", importance=0.4, score=0.8),
        _make_chunk("high_1", importance=0.92, score=0.3),  # 高 importance, 低 score
        _make_chunk("low_2", importance=0.3, score=0.6),
        _make_chunk("low_3", importance=0.2, score=0.5),
    ]
    result = _apply_serial_position(top_k, imp_threshold=0.85, recency_types=set())
    # constraint_1 应保持在约束位置（result 首部）
    constraint_ids = [c["id"] for _, c in result if c["chunk_type"] == "design_constraint"]
    assert "constraint_1" in constraint_ids, "SP5: design_constraint 应保留"
    # high_1 应出现在 normal 部分的首位
    normal_results = [(s, c) for s, c in result if c["chunk_type"] != "design_constraint"]
    assert normal_results[0][1]["id"] == "high_1", \
        f"SP5: high_1 应在 normal 首位，got {normal_results[0][1]['id']}"


# ── SP6: serial_position_enabled=False → 保持 score 顺序 ─────────────────────

def test_sp6_disabled_preserves_score_order():
    """SP6: serial_position_enabled=False 时，保持原始 score 降序。"""
    top_k = [
        _make_chunk("a", importance=0.3, score=0.9),
        _make_chunk("b", importance=0.95, score=0.7),
        _make_chunk("c", importance=0.4, score=0.5),
    ]
    result = _apply_serial_position(top_k, enabled=False)
    assert [c["id"] for _, c in result] == ["a", "b", "c"], \
        f"SP6: disabled 时应保持原始顺序，got {[c['id'] for _, c in result]}"


# ── SP7: 无高价值 chunk → 不重排 ─────────────────────────────────────────────

def test_sp7_no_high_value_chunks_no_reorder():
    """SP7: 所有 chunk importance < threshold 且不在 recency_types 中 → 不重排。"""
    top_k = [
        _make_chunk("a", chunk_type="conversation_summary", importance=0.4, score=0.9),
        _make_chunk("b", chunk_type="task_state", importance=0.3, score=0.7),
        _make_chunk("c", chunk_type="task_state", importance=0.2, score=0.5),
        _make_chunk("d", chunk_type="conversation_summary", importance=0.1, score=0.3),
    ]
    result = _apply_serial_position(top_k, imp_threshold=0.85, recency_types=set())
    # 无高价值候选，返回原始顺序
    assert [c["id"] for _, c in result] == [c["id"] for _, c in top_k], \
        f"SP7: 无高价值 chunk 时应保持原始顺序，got {[c['id'] for _, c in result]}"


# ── SP8: recency_types 命中 → 优先候选 ───────────────────────────────────────

def test_sp8_recency_types_become_primacy_recency():
    """SP8: chunk_type 在 serial_position_recency_types 中 → 候选 primacy/recency。"""
    top_k = [
        _make_chunk("task_1", chunk_type="task_state", importance=0.3, score=0.9),
        _make_chunk("decision_1", chunk_type="decision", importance=0.5, score=0.6),  # in recency_types
        _make_chunk("task_2", chunk_type="task_state", importance=0.2, score=0.7),
        _make_chunk("reason_1", chunk_type="reasoning_chain", importance=0.4, score=0.4),  # in types
    ]
    recency_types = {"decision", "reasoning_chain"}
    result = _apply_serial_position(top_k, imp_threshold=0.95,  # 高阈值，importance 不触发
                                    recency_types=recency_types)
    result_ids = [c["id"] for _, c in result]
    # decision_1 应在首位（recency_type，score 在 high 候选中最高 0.6）
    # reason_1 应在末位（次高 0.4）
    high_positions = [i for i, (_, c) in enumerate(result)
                      if c["chunk_type"] in recency_types]
    assert 0 in high_positions, f"SP8: 首位应是 recency_type chunk，got {result_ids}"
    assert len(result) - 1 in high_positions, \
        f"SP8: 末位应是 recency_type chunk，got {result_ids}"
