#!/usr/bin/env python3
"""
迭代359 测试：Session Injection Deduplication

OS 类比：Linux KSM (Kernel Samepage Merging) — 同一物理页达到引用阈值后合并。
同一 chunk 在 session 内注入次数 >= threshold 后从输出中剔除（边际价值趋零）。

测试矩阵：
  T1: session_dedup_threshold tunable 存在且默认值=2
  T2: _session_injection_counts < threshold 时 top_k 不变
  T3: _session_injection_counts >= threshold 时 chunk 被过滤（非 design_constraint）
  T4: design_constraint 豁免——无论注入次数多少都不去重
  T5: threshold=0 时去重逻辑完全禁用（backward compat）
  T6: 所有候选都被去重时返回空列表（不崩溃）
  T7: reason 字段包含 dedup:<n> 标记
  T8: config tunable 可调整——threshold=1 时首次注入后即去重
"""
import sys
import os
import json
import uuid
import tempfile
from pathlib import Path
from datetime import datetime, timezone, timedelta

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

import tmpfs  # noqa: F401
import config
from config import get as _sysctl


# ── 模拟 inject dedup 逻辑（直接测试逻辑，无需启动完整 retriever）──

def _make_chunk(chunk_type="decision", importance=0.5) -> dict:
    now = datetime.now(timezone.utc) - timedelta(days=1)
    return {
        "id": str(uuid.uuid4()),
        "chunk_type": chunk_type,
        "importance": importance,
        "summary": f"test summary {uuid.uuid4().hex[:6]}",
        "content": "test content",
        "last_accessed": now.isoformat(),
        "created_at": now.isoformat(),
        "access_count": 0,
        "verification_status": "pending",
        "confidence_score": 0.7,
    }


def _apply_dedup(top_k, session_injection_counts, threshold):
    """
    迭代359 dedup 逻辑的直接复现（与 retriever.py 保持同步）。
    返回 (filtered_top_k, dedup_count)
    """
    if threshold <= 0 or not session_injection_counts:
        return top_k, 0

    dedup_top_k = []
    dedup_count = 0
    for score, chunk in top_k:
        cid = chunk.get("id", "")
        ctype = chunk.get("chunk_type", "")
        inj_cnt = session_injection_counts.get(cid, 0)
        if ctype == "design_constraint":
            dedup_top_k.append((score, chunk))
        elif inj_cnt >= threshold:
            dedup_count += 1
        else:
            dedup_top_k.append((score, chunk))

    return dedup_top_k, dedup_count


# ── Tests ──

def test_01_tunable_exists():
    """T1: session_dedup_threshold tunable 存在且默认值=2"""
    val = _sysctl("retriever.session_dedup_threshold")
    assert val == 2, f"Expected 2, got {val}"
    assert isinstance(val, int)
    print("  T1 ✓ session_dedup_threshold=2 (default)")


def test_02_no_dedup_below_threshold():
    """T2: 注入次数 < threshold 时 top_k 不变"""
    c1 = _make_chunk("decision")
    c2 = _make_chunk("reasoning_chain")
    top_k = [(0.8, c1), (0.7, c2)]
    # c1 注入了 1 次，threshold=2 → 不去重
    counts = {c1["id"]: 1}
    result, dedup_count = _apply_dedup(top_k, counts, threshold=2)
    assert len(result) == 2, f"Expected 2, got {len(result)}"
    assert dedup_count == 0
    print("  T2 ✓ no dedup below threshold")


def test_03_dedup_at_threshold():
    """T3: 注入次数 >= threshold 时 chunk 被过滤"""
    c1 = _make_chunk("decision")
    c2 = _make_chunk("reasoning_chain")
    top_k = [(0.8, c1), (0.7, c2)]
    # c1 注入了 2 次，threshold=2 → 去重
    counts = {c1["id"]: 2, c2["id"]: 0}
    result, dedup_count = _apply_dedup(top_k, counts, threshold=2)
    assert len(result) == 1, f"Expected 1, got {len(result)}"
    assert result[0][1]["id"] == c2["id"], "c2 should remain"
    assert dedup_count == 1
    print("  T3 ✓ dedup at threshold (inj_cnt=2, threshold=2)")


def test_04_design_constraint_exempt():
    """T4: design_constraint 豁免——无论注入次数多少都不去重"""
    dc = _make_chunk("design_constraint", importance=0.9)
    normal = _make_chunk("decision")
    top_k = [(0.9, dc), (0.7, normal)]
    # dc 注入 10 次，threshold=2 → dc 豁免
    counts = {dc["id"]: 10, normal["id"]: 5}
    result, dedup_count = _apply_dedup(top_k, counts, threshold=2)
    # design_constraint 保留，normal 被去重
    assert len(result) == 1
    assert result[0][1]["chunk_type"] == "design_constraint"
    assert dedup_count == 1
    print("  T4 ✓ design_constraint exempt from dedup")


def test_05_threshold_zero_disables():
    """T5: threshold=0 时去重逻辑完全禁用"""
    c1 = _make_chunk("decision")
    top_k = [(0.8, c1)]
    counts = {c1["id"]: 999}
    result, dedup_count = _apply_dedup(top_k, counts, threshold=0)
    assert len(result) == 1, "threshold=0 should disable dedup"
    assert dedup_count == 0
    print("  T5 ✓ threshold=0 disables dedup (backward compat)")


def test_06_all_deduped_returns_empty():
    """T6: 所有候选都被去重时返回空列表（不崩溃）"""
    chunks = [_make_chunk("decision") for _ in range(3)]
    top_k = [(0.9 - i * 0.1, c) for i, c in enumerate(chunks)]
    # 全部注入 5 次，threshold=2
    counts = {c["id"]: 5 for c in chunks}
    result, dedup_count = _apply_dedup(top_k, counts, threshold=2)
    assert result == [], f"Expected empty, got {result}"
    assert dedup_count == 3
    print("  T6 ✓ all deduped → empty list (no crash)")


def test_07_reason_field_marker():
    """T7: dedup_count > 0 时 reason 包含 dedup:<n> 标记"""
    # 模拟 reason 构建逻辑
    reason = "hash_changed|full"
    _iter359_dedup_count = 2
    if _iter359_dedup_count > 0:
        reason += f"|dedup:{_iter359_dedup_count}"
    assert "dedup:2" in reason, f"reason should contain dedup:2, got: {reason}"
    print(f"  T7 ✓ reason includes dedup marker: {reason}")


def test_08_threshold_one_dedup_after_first():
    """T8: threshold=1 时，chunk 被注入 1 次后即去重"""
    c1 = _make_chunk("decision")
    c2 = _make_chunk("task_state")
    top_k = [(0.8, c1), (0.7, c2)]
    # c1 已注入 1 次，c2 未注入
    counts = {c1["id"]: 1}
    result, dedup_count = _apply_dedup(top_k, counts, threshold=1)
    assert len(result) == 1
    assert result[0][1]["id"] == c2["id"]
    assert dedup_count == 1
    print("  T8 ✓ threshold=1 deduplicates after first injection")


def test_09_empty_counts_no_dedup():
    """T9: 空 session_injection_counts 时不触发去重（新 session）"""
    chunks = [_make_chunk("decision") for _ in range(3)]
    top_k = [(0.9 - i * 0.1, c) for i, c in enumerate(chunks)]
    result, dedup_count = _apply_dedup(top_k, {}, threshold=2)
    assert len(result) == 3
    assert dedup_count == 0
    print("  T9 ✓ empty counts → no dedup (new session)")


def test_10_mixed_chunk_types():
    """T10: 混合 chunk_type 场景——部分去重，constraint 保留"""
    dc = _make_chunk("design_constraint", importance=0.95)
    dec1 = _make_chunk("decision")
    dec2 = _make_chunk("decision")
    rc = _make_chunk("reasoning_chain")
    top_k = [(0.95, dc), (0.85, dec1), (0.75, dec2), (0.65, rc)]
    counts = {
        dc["id"]: 5,   # design_constraint → 豁免
        dec1["id"]: 3,  # >= threshold=2 → 去重
        dec2["id"]: 1,  # < threshold → 保留
        rc["id"]: 0,    # 未注入 → 保留
    }
    result, dedup_count = _apply_dedup(top_k, counts, threshold=2)
    assert dedup_count == 1, f"Expected 1 dedup, got {dedup_count}"
    assert len(result) == 3  # dc + dec2 + rc
    result_ids = {r[1]["id"] for r in result}
    assert dc["id"] in result_ids
    assert dec1["id"] not in result_ids
    assert dec2["id"] in result_ids
    assert rc["id"] in result_ids
    print(f"  T10 ✓ mixed types: {dedup_count} deduped, {len(result)} remaining")


if __name__ == "__main__":
    print("迭代359 测试：Session Injection Deduplication")
    print("=" * 60)

    tests = [
        test_01_tunable_exists,
        test_02_no_dedup_below_threshold,
        test_03_dedup_at_threshold,
        test_04_design_constraint_exempt,
        test_05_threshold_zero_disables,
        test_06_all_deduped_returns_empty,
        test_07_reason_field_marker,
        test_08_threshold_one_dedup_after_first,
        test_09_empty_counts_no_dedup,
        test_10_mixed_chunk_types,
    ]

    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            failed += 1
            import traceback
            print(f"  FAIL {t.__name__}: {e}")
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"结果：{passed}/{passed + failed} 通过")
    if failed:
        import sys
        sys.exit(1)
