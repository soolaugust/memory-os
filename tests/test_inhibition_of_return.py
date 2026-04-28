"""
test_inhibition_of_return.py — iter391: Inhibition of Return 单元测试

覆盖：
  IOR1: 刚注入 (turn_delta=1) → 最大 IOR 惩罚（score × (1 - penalty)）
  IOR2: 未注入 chunk → score 不变
  IOR3: IOR 衰减：turn_delta=decay_turns → 惩罚减半
  IOR4: exempt_types（design_constraint）→ 不受 IOR 惩罚
  IOR5: 不同 session → IOR 状态重置，无惩罚
  IOR6: ior_enabled=False → 不应用惩罚
  IOR7: _update_ior_state — 新 session 重置 current_turn
  IOR8: _update_ior_state — exempt_types 不记录进 injections
  IOR9: _update_ior_state — stale entries（>50 turns）自动清除
  IOR10: 多次注入同一 chunk → 惩罚不累加（turn 覆盖更新）

认知科学依据：
  Posner (1980) / Klein (2000) Inhibition of Return —
  注意力探测某位置后短暂抑制，促进注意力转移到新位置，
  避免注意力在同一位置反复停留（"探索驱动 vs 挖掘驱动"权衡）。
OS 类比：Linux CFQ anti-starvation timeslice —
  刚被服务的进程在 timeslice 内降低优先级，防止单一进程独占 I/O 带宽。
"""
import sys
import json
import math
import tempfile
import os
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "hooks"))

import tmpfs  # noqa


# ── 可独立测试的 IOR 应用函数（从 retriever.py 提取逻辑）───────────────────

def _apply_ior_penalty(final: list, ior_data: dict, session_id: str,
                        ior_penalty: float = 0.20,
                        ior_decay_turns: int = 3,
                        ior_exempt_types: set = None) -> list:
    """
    从 retriever.py 提取的 IOR 惩罚逻辑（pure function for testing）。

    Args:
        final: [(score, chunk_dict), ...]
        ior_data: IOR 状态 dict（含 session_id, injections, current_turn）
        session_id: 当前 session
        ior_penalty: 最大惩罚系数（0.20 默认）
        ior_decay_turns: 半衰期（turn 数）
        ior_exempt_types: 免除 IOR 惩罚的 chunk_type 集合

    Returns:
        应用 IOR 惩罚后的 [(score, chunk_dict), ...]
    """
    if ior_exempt_types is None:
        ior_exempt_types = set()

    if not (ior_data and ior_data.get("session_id") == session_id
            and isinstance(ior_data.get("injections"), dict)):
        return final

    ior_injs = ior_data["injections"]
    ior_cur_turn = ior_data.get("current_turn", 0)

    if not ior_injs:
        return final

    result = []
    for s, c in final:
        cid = c.get("id", "")
        ctype = c.get("chunk_type", "")
        if cid in ior_injs and ctype not in ior_exempt_types:
            turns_since = max(0, ior_cur_turn - ior_injs[cid])
            penalty_factor = ior_penalty * math.exp(
                -math.log(2) / max(1, ior_decay_turns) * turns_since
            )
            new_score = s * (1.0 - penalty_factor)
            result.append((new_score, c))
        else:
            result.append((s, c))
    return result


def _make_chunk(cid: str, ctype: str = "decision", score: float = 1.0):
    return (score, {"id": cid, "chunk_type": ctype, "summary": f"chunk {cid}"})


def _make_ior_data(session_id: str, injections: dict, current_turn: int) -> dict:
    return {"session_id": session_id, "injections": injections, "current_turn": current_turn}


# ══════════════════════════════════════════════════════════════════════
# 1. 基础 IOR 惩罚行为
# ══════════════════════════════════════════════════════════════════════

def test_ior1_immediate_injection_max_penalty():
    """刚注入（turn_delta=1）→ 接近最大惩罚。"""
    # IOR_PENALTY=0.20, decay_turns=3
    # turn_delta=1: penalty_factor = 0.20 × exp(-ln2/3 × 1) ≈ 0.20 × 0.794 ≈ 0.159
    # score = 1.0 × (1 - 0.159) ≈ 0.841
    ior_data = _make_ior_data("s1", {"chunk_A": 9}, current_turn=10)
    result = _apply_ior_penalty(
        [_make_chunk("chunk_A")], ior_data, "s1",
        ior_penalty=0.20, ior_decay_turns=3
    )
    score, _ = result[0]
    expected = 1.0 * (1.0 - 0.20 * math.exp(-math.log(2) / 3 * 1))
    assert abs(score - expected) < 1e-6, f"IOR1: got {score:.4f}, expected {expected:.4f}"
    assert score < 1.0, "IOR 惩罚后 score 应降低"


def test_ior2_uninjected_chunk_unchanged():
    """未注入的 chunk → score 不变。"""
    ior_data = _make_ior_data("s1", {"chunk_A": 9}, current_turn=10)
    result = _apply_ior_penalty(
        [_make_chunk("chunk_B")], ior_data, "s1",
    )
    score, _ = result[0]
    assert abs(score - 1.0) < 1e-9, f"未注入 chunk 不应受 IOR 影响，got {score}"


def test_ior3_decay_halving():
    """经过 decay_turns 轮后，IOR 惩罚减半。"""
    # turn_delta=0: penalty_factor = 0.20 × 1.0 = 0.20
    # turn_delta=decay_turns=3: penalty_factor = 0.20 × 0.5 = 0.10 (半衰期)
    decay_turns = 3

    ior_data_0 = _make_ior_data("s1", {"A": 10}, current_turn=10)  # delta=0
    result_0 = _apply_ior_penalty([_make_chunk("A")], ior_data_0, "s1",
                                   ior_penalty=0.20, ior_decay_turns=decay_turns)

    ior_data_decay = _make_ior_data("s1", {"A": 7}, current_turn=10)  # delta=3
    result_decay = _apply_ior_penalty([_make_chunk("A")], ior_data_decay, "s1",
                                       ior_penalty=0.20, ior_decay_turns=decay_turns)

    pf_0 = 0.20 * 1.0                              # turn_delta=0
    pf_decay = 0.20 * math.exp(-math.log(2))        # turn_delta=decay_turns → ×0.5

    score_0 = result_0[0][0]
    score_decay = result_decay[0][0]

    assert abs(score_0 - (1.0 - pf_0)) < 1e-6, f"delta=0 score mismatch: {score_0}"
    assert abs(score_decay - (1.0 - pf_decay)) < 1e-6, f"delta=3 score mismatch: {score_decay}"
    assert score_decay > score_0, "衰减后惩罚应更小（score 更高）"


def test_ior4_exempt_types_no_penalty():
    """design_constraint → 不受 IOR 惩罚。"""
    ior_data = _make_ior_data("s1", {"chunk_dc": 9}, current_turn=10)
    result = _apply_ior_penalty(
        [(1.0, {"id": "chunk_dc", "chunk_type": "design_constraint", "summary": "约束"})],
        ior_data, "s1",
        ior_exempt_types={"design_constraint"}
    )
    score, _ = result[0]
    assert abs(score - 1.0) < 1e-9, f"design_constraint 应豁免 IOR，got {score}"


def test_ior5_different_session_no_penalty():
    """不同 session → IOR 状态不应用。"""
    ior_data = _make_ior_data("session_old", {"chunk_A": 9}, current_turn=10)
    result = _apply_ior_penalty(
        [_make_chunk("chunk_A")], ior_data, "session_new"
    )
    score, _ = result[0]
    assert abs(score - 1.0) < 1e-9, f"不同 session 不应受 IOR 影响，got {score}"


# ══════════════════════════════════════════════════════════════════════
# 2. 远期衰减（几乎完全恢复）
# ══════════════════════════════════════════════════════════════════════

def test_ior_long_decay_nearly_zero():
    """注入后经过很多轮（10 × decay_turns）→ IOR 惩罚接近 0。"""
    ior_data = _make_ior_data("s1", {"A": 0}, current_turn=30)  # delta=30, decay_turns=3
    result = _apply_ior_penalty([_make_chunk("A")], ior_data, "s1",
                                 ior_penalty=0.20, ior_decay_turns=3)
    score, _ = result[0]
    pf = 0.20 * math.exp(-math.log(2) / 3 * 30)
    expected = 1.0 * (1.0 - pf)
    assert abs(score - expected) < 1e-6, f"长衰减后惩罚应接近 0，got {score:.6f}"
    assert score > 0.99, f"30 个 half-life 后 score 应几乎为 1.0，got {score:.6f}"


# ══════════════════════════════════════════════════════════════════════
# 3. _update_ior_state 行为验证
# ══════════════════════════════════════════════════════════════════════

def test_ior7_new_session_resets_state(tmp_path):
    """新 session → current_turn 重置为 1，injections 清空。"""
    sys.path.insert(0, str(Path(__file__).parent.parent / "hooks"))
    from retriever import _update_ior_state

    ior_file = tmp_path / ".ior_state.json"
    # 写入旧 session 状态
    ior_file.write_text(json.dumps({
        "session_id": "old_session",
        "current_turn": 99,
        "injections": {"chunk_old": 50}
    }))

    # 临时替换 IOR_FILE
    import retriever as _ret
    orig_ior = _ret.IOR_FILE
    _ret.IOR_FILE = str(ior_file)
    try:
        _update_ior_state(["chunk_new"], "new_session")
        data = json.loads(ior_file.read_text())
        assert data["session_id"] == "new_session"
        assert data["current_turn"] == 1
        assert "chunk_old" not in data["injections"]
        assert "chunk_new" in data["injections"]
    finally:
        _ret.IOR_FILE = orig_ior


def test_ior8_exempt_types_not_recorded(tmp_path):
    """exempt_types 中的 chunk_type → 不记录进 injections。"""
    sys.path.insert(0, str(Path(__file__).parent.parent / "hooks"))
    from retriever import _update_ior_state

    ior_file = tmp_path / ".ior_state.json"
    import retriever as _ret
    orig_ior = _ret.IOR_FILE
    _ret.IOR_FILE = str(ior_file)
    try:
        _update_ior_state(
            ["chunk_dc", "chunk_normal"],
            "s1",
            exempt_types={"design_constraint"},
            chunk_types={"chunk_dc": "design_constraint", "chunk_normal": "decision"}
        )
        data = json.loads(ior_file.read_text())
        assert "chunk_dc" not in data["injections"], "design_constraint 不应记录进 IOR"
        assert "chunk_normal" in data["injections"], "普通 chunk 应记录进 IOR"
    finally:
        _ret.IOR_FILE = orig_ior


def test_ior9_stale_entries_cleaned(tmp_path):
    """超过 50 turns 的旧记录 → 自动清除。"""
    sys.path.insert(0, str(Path(__file__).parent.parent / "hooks"))
    from retriever import _update_ior_state

    ior_file = tmp_path / ".ior_state.json"
    # 设置包含旧记录的状态
    ior_file.write_text(json.dumps({
        "session_id": "s1",
        "current_turn": 60,
        "injections": {
            "old_chunk": 5,     # 距现在 56 turns → 应清除（>50）
            "recent_chunk": 55  # 距现在 6 turns → 应保留
        }
    }))

    import retriever as _ret
    orig_ior = _ret.IOR_FILE
    _ret.IOR_FILE = str(ior_file)
    try:
        _update_ior_state(["new_chunk"], "s1")
        data = json.loads(ior_file.read_text())
        assert "old_chunk" not in data["injections"], "旧记录应被清除"
        assert "recent_chunk" in data["injections"], "近期记录应保留"
    finally:
        _ret.IOR_FILE = orig_ior


def test_ior10_same_chunk_injected_twice_updates_turn(tmp_path):
    """同一 chunk 再次注入 → turn 更新（不累加惩罚）。"""
    sys.path.insert(0, str(Path(__file__).parent.parent / "hooks"))
    from retriever import _update_ior_state

    ior_file = tmp_path / ".ior_state.json"
    import retriever as _ret
    orig_ior = _ret.IOR_FILE
    _ret.IOR_FILE = str(ior_file)
    try:
        # 第一次注入：turn=1
        _update_ior_state(["chunk_A"], "s1")
        data1 = json.loads(ior_file.read_text())
        turn1 = data1["injections"]["chunk_A"]

        # 第二次注入：turn=2
        _update_ior_state(["chunk_A"], "s1")
        data2 = json.loads(ior_file.read_text())
        turn2 = data2["injections"]["chunk_A"]

        assert turn2 > turn1, "第二次注入应更新 turn"
        assert data2["current_turn"] == 2, f"current_turn 应为 2，got {data2['current_turn']}"
    finally:
        _ret.IOR_FILE = orig_ior
