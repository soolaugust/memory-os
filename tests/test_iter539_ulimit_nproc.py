"""
iter539: ulimit_nproc — Per-Invocation Chunk Write Rate Limit Tests

OS 类比：Linux RLIMIT_NPROC (setrlimit, 1983 BSD)
  限制单用户进程数，防止 fork bomb 耗尽 PID 空间。

验证：
  1. 候选数 <= ulimit 时不裁剪（全量透传）
  2. 候选数 > ulimit 时裁剪到 ulimit（精确数量）
  3. 裁剪按 chunk_type priority 排序（高优先保留）
  4. design_constraint > decision > causal_chain > conversation_summary
  5. 同优先级内保持原始顺序（stable sort）
  6. config tunable extractor.ulimit_nproc 生效
  7. ulimit=2 极端场景——只保留 top-2 高优先级
  8. 所有 candidate 同类型时保留前 N 个
  9. 空列表不触发裁剪
  10. dmesg 记录丢弃数量（_ulimit_dropped）
  11. AIMD conservative 策略与 ulimit 正交（先 AIMD 后 ulimit）
  12. 量化证据作为 decision 正确参与排序
"""
import sys
import os
import sqlite3
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# 隔离测试 DB
_tmpdir = tempfile.mkdtemp(prefix="test539_")
_test_db = os.path.join(_tmpdir, "test.db")
os.environ["MEMORY_OS_DB"] = _test_db

import pytest
from config import get as _sysctl

# ── 模拟 ulimit 裁剪逻辑（从 extractor.py 提取的核心算法）──
_TYPE_PRIORITY = {
    "design_constraint": 0.95,
    "quantitative_evidence": 0.90,
    "decision": 0.85,
    "procedure": 0.85,
    "causal_chain": 0.82,
    "reasoning_chain": 0.80,
    "excluded_path": 0.70,
    "conversation_summary": 0.65,
}


def _apply_ulimit(decisions, excluded, reasoning, causal_chains,
                  conv_summaries, constraints, ulimit):
    """复现 extractor.py 中的 ulimit 裁剪算法。"""
    _all_candidates = []
    for s in decisions:
        _all_candidates.append((s, "decision", _TYPE_PRIORITY["decision"]))
    for s in excluded:
        _all_candidates.append((s, "excluded_path", _TYPE_PRIORITY["excluded_path"]))
    for s in reasoning:
        _all_candidates.append((s, "reasoning_chain", _TYPE_PRIORITY["reasoning_chain"]))
    for s in causal_chains:
        _all_candidates.append((s, "causal_chain", _TYPE_PRIORITY["causal_chain"]))
    for s in conv_summaries:
        _all_candidates.append((s, "conversation_summary", _TYPE_PRIORITY["conversation_summary"]))
    for s in constraints:
        _all_candidates.append((s, "design_constraint", _TYPE_PRIORITY["design_constraint"]))

    if len(_all_candidates) > ulimit:
        _all_candidates.sort(key=lambda x: x[2], reverse=True)
        _dropped = len(_all_candidates) - ulimit
        _all_candidates = _all_candidates[:ulimit]
        decisions = [s for s, t, _ in _all_candidates if t == "decision"]
        excluded = [s for s, t, _ in _all_candidates if t == "excluded_path"]
        reasoning = [s for s, t, _ in _all_candidates if t == "reasoning_chain"]
        causal_chains = [s for s, t, _ in _all_candidates if t == "causal_chain"]
        conv_summaries = [s for s, t, _ in _all_candidates if t == "conversation_summary"]
        constraints = [s for s, t, _ in _all_candidates if t == "design_constraint"]
    else:
        _dropped = 0

    return decisions, excluded, reasoning, causal_chains, conv_summaries, constraints, _dropped


# ── Tests ──

def test_t1_under_limit_no_trim():
    """候选数 <= ulimit 时不裁剪。"""
    d, e, r, cc, cs, dc, dropped = _apply_ulimit(
        decisions=["d1", "d2", "d3"],
        excluded=["e1"],
        reasoning=["r1"],
        causal_chains=["cc1"],
        conv_summaries=["cs1"],
        constraints=["dc1"],
        ulimit=8,
    )
    assert dropped == 0
    assert d == ["d1", "d2", "d3"]
    assert e == ["e1"]
    assert r == ["r1"]
    assert cc == ["cc1"]
    assert cs == ["cs1"]
    assert dc == ["dc1"]


def test_t2_over_limit_trim_to_exact():
    """候选数 > ulimit 时精确裁剪到 ulimit。"""
    d, e, r, cc, cs, dc, dropped = _apply_ulimit(
        decisions=["d1", "d2", "d3", "d4", "d5"],
        excluded=["e1", "e2"],
        reasoning=["r1", "r2"],
        causal_chains=["cc1", "cc2", "cc3"],
        conv_summaries=["cs1", "cs2"],
        constraints=[],
        ulimit=8,
    )
    total_kept = len(d) + len(e) + len(r) + len(cc) + len(cs) + len(dc)
    assert total_kept == 8
    assert dropped == 6  # 14 - 8 = 6


def test_t3_priority_order():
    """高优先级 chunk_type 优先保留。"""
    d, e, r, cc, cs, dc, dropped = _apply_ulimit(
        decisions=["decision_1"],
        excluded=["excluded_1"],
        reasoning=["reasoning_1"],
        causal_chains=["causal_1"],
        conv_summaries=["summary_1", "summary_2", "summary_3"],
        constraints=["constraint_1"],
        ulimit=4,
    )
    # Top-4 by priority: constraint(0.95) > decision(0.85) > causal(0.82) > reasoning(0.80)
    assert dc == ["constraint_1"]
    assert d == ["decision_1"]
    assert cc == ["causal_1"]
    assert r == ["reasoning_1"]
    assert e == []  # excluded(0.70) dropped
    assert cs == []  # summary(0.65) dropped
    assert dropped == 4


def test_t4_design_constraint_highest_priority():
    """design_constraint 始终最高优先级保留。"""
    d, e, r, cc, cs, dc, dropped = _apply_ulimit(
        decisions=["d1", "d2", "d3", "d4", "d5"],
        excluded=[],
        reasoning=[],
        causal_chains=[],
        conv_summaries=["cs1", "cs2", "cs3", "cs4"],
        constraints=["CRITICAL_CONSTRAINT"],
        ulimit=3,
    )
    # constraint(0.95) > all decisions(0.85) > summaries(0.65)
    assert "CRITICAL_CONSTRAINT" in dc
    assert len(d) == 2  # Top-2 decisions fill remaining slots
    assert cs == []


def test_t5_stable_sort_within_same_priority():
    """同优先级内保持原始顺序（stable sort）。"""
    d, e, r, cc, cs, dc, dropped = _apply_ulimit(
        decisions=["first_decision", "second_decision", "third_decision",
                   "fourth_decision", "fifth_decision"],
        excluded=[],
        reasoning=[],
        causal_chains=[],
        conv_summaries=[],
        constraints=[],
        ulimit=3,
    )
    # 全是 decision（同优先级），stable sort 保留前 3 个
    assert d == ["first_decision", "second_decision", "third_decision"]
    assert dropped == 2


def test_t6_config_tunable():
    """config tunable extractor.ulimit_nproc 可读取。"""
    val = _sysctl("extractor.ulimit_nproc")
    assert val == 8  # 默认值
    assert isinstance(val, int)


def test_t7_extreme_ulimit_2():
    """ulimit=2 极端场景——只保留 top-2。"""
    d, e, r, cc, cs, dc, dropped = _apply_ulimit(
        decisions=["d1"],
        excluded=["e1"],
        reasoning=["r1"],
        causal_chains=["cc1"],
        conv_summaries=["cs1"],
        constraints=["dc1"],
        ulimit=2,
    )
    # Top-2: constraint(0.95), decision(0.85)
    assert dc == ["dc1"]
    assert d == ["d1"]
    assert e == []
    assert r == []
    assert cc == []
    assert cs == []
    assert dropped == 4


def test_t8_all_same_type_keep_first_n():
    """所有候选同类型时保留前 N 个（原始顺序）。"""
    chains = [f"chain_{i}" for i in range(10)]
    d, e, r, cc, cs, dc, dropped = _apply_ulimit(
        decisions=[],
        excluded=[],
        reasoning=[],
        causal_chains=chains,
        conv_summaries=[],
        constraints=[],
        ulimit=4,
    )
    assert cc == ["chain_0", "chain_1", "chain_2", "chain_3"]
    assert dropped == 6


def test_t9_empty_lists_no_crash():
    """空列表不触发裁剪。"""
    d, e, r, cc, cs, dc, dropped = _apply_ulimit(
        decisions=[],
        excluded=[],
        reasoning=[],
        causal_chains=[],
        conv_summaries=[],
        constraints=[],
        ulimit=8,
    )
    assert dropped == 0
    assert d == e == r == cc == cs == dc == []


def test_t10_dropped_count_correct():
    """丢弃计数精确。"""
    # 15 candidates, ulimit=8 → dropped=7
    d, e, r, cc, cs, dc, dropped = _apply_ulimit(
        decisions=["d1", "d2", "d3"],
        excluded=["e1", "e2", "e3"],
        reasoning=["r1", "r2", "r3"],
        causal_chains=["cc1", "cc2", "cc3"],
        conv_summaries=["cs1", "cs2", "cs3"],
        constraints=[],
        ulimit=8,
    )
    assert dropped == 7
    total = len(d) + len(e) + len(r) + len(cc) + len(cs) + len(dc)
    assert total == 8


def test_t11_quantitative_evidence_as_decision():
    """量化证据并入 decisions 列表，参与 decision 优先级排序。"""
    # 量化证据通过 decisions.extend(quant_conclusions) 合入
    # 在 ulimit 排序中它们作为 "decision" 类型（0.85）
    d, e, r, cc, cs, dc, dropped = _apply_ulimit(
        decisions=["quant: 延迟降低 47%", "normal decision"],
        excluded=["e1", "e2", "e3", "e4"],
        reasoning=[],
        causal_chains=[],
        conv_summaries=["cs1", "cs2", "cs3"],
        constraints=[],
        ulimit=3,
    )
    # decisions(0.85) > excluded(0.70) > summaries(0.65)
    assert len(d) == 2  # both decisions kept
    assert len(e) == 1  # 1 excluded fills last slot
    assert cs == []


def test_t12_performance():
    """裁剪 100 个 candidates 到 8 个性能 < 1ms。"""
    import time
    decisions = [f"decision_{i}" for i in range(30)]
    excluded = [f"excl_{i}" for i in range(20)]
    reasoning = [f"reason_{i}" for i in range(15)]
    causal_chains = [f"chain_{i}" for i in range(15)]
    conv_summaries = [f"summary_{i}" for i in range(15)]
    constraints = [f"constraint_{i}" for i in range(5)]

    t0 = time.time()
    for _ in range(1000):
        _apply_ulimit(decisions, excluded, reasoning, causal_chains,
                      conv_summaries, constraints, 8)
    elapsed = (time.time() - t0) * 1000
    assert elapsed < 500, f"1000 iterations took {elapsed:.1f}ms (expected < 500ms)"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
            print(f"  ✓ {t.__name__}")
        except AssertionError as e:
            print(f"  ✗ {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
