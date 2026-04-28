"""
test_temporal_priming.py — iter388: Temporal Priming 单元测试

覆盖：
  TP1: primed chunk（上次会话命中）→ score 增加 priming_boost
  TP2: 非 primed chunk → score 不变
  TP3: 不同 session_id → 不应用 priming（session 隔离）
  TP4: priming_enabled=False → 不应用 priming
  TP5: priming_boost=0.0 → score 不变
  TP6: 多个 primed chunks → 全部得到加分
  TP7: shadow_data 无 top_k_ids → 安全退化，score 不变
  TP8: session_id 为空 → 不应用 priming

认知科学依据：
  Tulving & Schacter (1990) Priming Effect —
  最近在同会话中被召回的记忆，在随后的检索中被激活的阈值降低（启动效应）。
  神经基础：海马-新皮层投射维持短期激活状态（working memory buffer），
  最近命中的 chunk 仍处于"激活窗口"，再次相关时更易浮现。
OS 类比：CPU 时间局部性 (temporal locality) — 最近访问的 cache line 比
  未访问的有更高命中概率（L2/L3 temporal prefetch）。
"""
import sys
import json
import pytest
from pathlib import Path
from typing import List, Tuple, Set, Optional

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import tmpfs  # noqa


def _apply_priming_boost(
    final: List[Tuple[float, dict]],
    shadow_data: Optional[dict],
    session_id: str,
    priming_enabled: bool = True,
    priming_boost: float = 0.08,
) -> List[Tuple[float, dict]]:
    """
    提取自 retriever.py iter388 priming boost 逻辑的纯函数（用于单元测试）。
    模拟 retriever 中的 priming 块逻辑。
    """
    if not priming_enabled:
        return final
    if not session_id:
        return final
    if shadow_data is None:
        return final
    if shadow_data.get("session_id") != session_id:
        return final
    primed_ids: Set[str] = set(shadow_data.get("top_k_ids") or [])
    if not primed_ids:
        return final
    return [
        (s + priming_boost if c.get("id") in primed_ids else s, c)
        for s, c in final
    ]


def _make_chunk(cid: str, summary: str = "test") -> dict:
    return {"id": cid, "summary": summary, "chunk_type": "decision"}


def _make_shadow(session_id: str, top_k_ids: List[str]) -> dict:
    return {"session_id": session_id, "top_k_ids": top_k_ids}


# ── TP1: primed chunk 获得加分 ─────────────────────────────────────────────

def test_tp1_primed_chunk_gets_boost():
    """primed chunk（上次会话命中）→ score 增加 priming_boost。"""
    chunk_primed = _make_chunk("primed_chunk")
    chunk_other = _make_chunk("other_chunk")
    final = [(0.5, chunk_primed), (0.5, chunk_other)]
    shadow = _make_shadow("sess_abc", ["primed_chunk"])

    result = _apply_priming_boost(final, shadow, "sess_abc",
                                   priming_enabled=True, priming_boost=0.08)
    scores = {c["id"]: s for s, c in result}

    assert abs(scores["primed_chunk"] - 0.58) < 0.001, \
        f"primed chunk 应得 +0.08，got {scores['primed_chunk']}"
    assert abs(scores["other_chunk"] - 0.50) < 0.001, \
        f"非 primed chunk 应不变，got {scores['other_chunk']}"


# ── TP2: 非 primed chunk score 不变 ────────────────────────────────────────

def test_tp2_non_primed_chunk_unchanged():
    """非 primed chunk → score 不变。"""
    chunk = _make_chunk("non_primed")
    final = [(0.7, chunk)]
    shadow = _make_shadow("sess_abc", ["some_other_id"])

    result = _apply_priming_boost(final, shadow, "sess_abc",
                                   priming_enabled=True, priming_boost=0.08)
    s, c = result[0]
    assert abs(s - 0.7) < 0.001, f"非 primed chunk score 不应变，got {s}"


# ── TP3: 不同 session_id → session 隔离 ───────────────────────────────────

def test_tp3_different_session_no_priming():
    """不同 session_id → 不应用 priming（session 隔离）。"""
    chunk = _make_chunk("chunk_x")
    final = [(0.6, chunk)]
    shadow = _make_shadow("sess_OLD", ["chunk_x"])  # 旧会话的 shadow

    # 当前会话是 sess_NEW，不同于 shadow 记录的 sess_OLD
    result = _apply_priming_boost(final, shadow, "sess_NEW",
                                   priming_enabled=True, priming_boost=0.08)
    s, c = result[0]
    assert abs(s - 0.6) < 0.001, \
        f"不同 session 不应应用 priming，got {s}"


# ── TP4: priming_enabled=False → 完全禁用 ─────────────────────────────────

def test_tp4_disabled_priming():
    """priming_enabled=False → 任何 chunk 都不加分。"""
    chunk = _make_chunk("any_chunk")
    final = [(0.5, chunk)]
    shadow = _make_shadow("sess_abc", ["any_chunk"])

    result = _apply_priming_boost(final, shadow, "sess_abc",
                                   priming_enabled=False, priming_boost=0.08)
    s, c = result[0]
    assert abs(s - 0.5) < 0.001, \
        f"禁用 priming 时不应加分，got {s}"


# ── TP5: priming_boost=0.0 → score 不变 ────────────────────────────────────

def test_tp5_zero_boost_no_change():
    """priming_boost=0.0 → score 不变（但 priming 逻辑仍运行）。"""
    chunk = _make_chunk("chunk_z")
    final = [(0.5, chunk)]
    shadow = _make_shadow("sess_abc", ["chunk_z"])

    result = _apply_priming_boost(final, shadow, "sess_abc",
                                   priming_enabled=True, priming_boost=0.0)
    s, c = result[0]
    assert abs(s - 0.5) < 0.001, \
        f"boost=0 时 score 不应变，got {s}"


# ── TP6: 多个 primed chunks 全部加分 ──────────────────────────────────────

def test_tp6_multiple_primed_chunks():
    """多个 primed chunks → 全部得到 priming_boost 加分。"""
    chunks = [
        _make_chunk("c1"),
        _make_chunk("c2"),
        _make_chunk("c3"),  # NOT primed
    ]
    final = [(0.4, chunks[0]), (0.5, chunks[1]), (0.6, chunks[2])]
    shadow = _make_shadow("sess_abc", ["c1", "c2"])  # c3 not primed

    result = _apply_priming_boost(final, shadow, "sess_abc",
                                   priming_enabled=True, priming_boost=0.10)
    scores = {c["id"]: s for s, c in result}

    assert abs(scores["c1"] - 0.50) < 0.001, f"c1 should get +0.10, got {scores['c1']}"
    assert abs(scores["c2"] - 0.60) < 0.001, f"c2 should get +0.10, got {scores['c2']}"
    assert abs(scores["c3"] - 0.60) < 0.001, f"c3 NOT primed should stay 0.60, got {scores['c3']}"


# ── TP7: shadow_data 无 top_k_ids → 安全退化 ──────────────────────────────

def test_tp7_no_top_k_ids_safe_fallback():
    """shadow_data 无 top_k_ids → 安全退化，score 不变。"""
    chunk = _make_chunk("chunk_a")
    final = [(0.7, chunk)]
    shadow = {"session_id": "sess_abc"}  # 无 top_k_ids 字段

    result = _apply_priming_boost(final, shadow, "sess_abc",
                                   priming_enabled=True, priming_boost=0.08)
    s, c = result[0]
    assert abs(s - 0.7) < 0.001, \
        f"无 top_k_ids 时应安全退化，got {s}"


# ── TP8: session_id 为空 → 不应用 priming ──────────────────────────────────

def test_tp8_empty_session_id_no_priming():
    """session_id 为空字符串 → 不应用 priming。"""
    chunk = _make_chunk("chunk_b")
    final = [(0.5, chunk)]
    shadow = _make_shadow("sess_abc", ["chunk_b"])

    result = _apply_priming_boost(final, shadow, "",
                                   priming_enabled=True, priming_boost=0.08)
    s, c = result[0]
    assert abs(s - 0.5) < 0.001, \
        f"空 session_id 时不应加分，got {s}"
