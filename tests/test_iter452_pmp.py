"""
test_iter452_pmp.py — iter452: Primary Memory Persistence 单元测试

覆盖：
  PM1: 近期同项目 recall_traces 注入次数 >= pmp_min_injections(3) → chunk stability 增加
  PM2: 注入次数 < pmp_min_injections → 无 PMP 加成
  PM3: pmp_enabled=False → 无任何加成
  PM4: importance < pmp_min_importance(0.40) → 不参与 PMP
  PM5: 更多注入次数 → 更大 pmp_factor → 更大加成（正比关系）
  PM6: stability 加成后不超过 365.0（cap 保护）
  PM7: pmp_boost 可配置（更大 boost → 更大加成）
  PM8: 返回值正确（pmp_boosted, total_examined 计数）
  PM9: 无 recall_traces 时正常返回（不崩溃，不加成）
  PM10: sleep_consolidate 在含有足够注入记录时包含 pmp_boosted key（集成测试）
  PM11: top_k_json 中 chunk id 作为 dict 和 str 两种格式都能正确解析
  PM12: session_window_hours 可配置（超出窗口的 recall_traces 不计入）

认知科学依据：
  Waugh & Norman (1965) "Primary memory" (Psychological Review) —
    工作记忆中被持续"rehearsal"的信息转入长时记忆；停止复述则快速消失。
  Rundus (1971) "Analysis of rehearsal processes in free recall" —
    复述次数与最终记忆保留率高度正相关（r=0.85）。
  Craik & Watkins (1973) "The role of rehearsal in short-term memory" —
    精细加工型复述（被集成进推理，而非单纯重复）有更好的长时记忆效果。

OS 类比：Linux page working set estimation (PG_referenced + PG_active) —
  页面在短时间内被多次访问，kswapd 将其提升到 active list，
  类比：session 内高频注入的 chunk = 短时内反复 referenced → 工作集热页 → sleep 时优先保护。
"""
import sys
import json
import sqlite3
import datetime
import unittest.mock as mock
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import tmpfs  # noqa

from store_vfs import (
    ensure_schema,
    apply_primary_memory_persistence,
    sleep_consolidate,
)
import config


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _utcnow():
    return datetime.datetime.now(datetime.timezone.utc)


def _insert_chunk(conn, cid, project="test", stability=5.0, importance=0.6):
    """Insert a chunk with controlled stability/importance."""
    now_iso = _utcnow().isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count,
            encode_context)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0.8, ?, 1, ?)""",
        (cid, project, "decision", f"content {cid}", f"summary {cid}",
         importance, stability, now_iso, now_iso, now_iso, "kernel_mm")
    )
    conn.commit()


def _get_stability(conn, cid: str) -> float:
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


def _insert_recall_trace(conn, project="test", chunk_ids=None, hours_ago=1.0):
    """Insert a recall trace with specified chunk IDs in top_k_json."""
    ts = (_utcnow() - datetime.timedelta(hours=hours_ago)).isoformat()
    if chunk_ids is None:
        chunk_ids = []
    top_k = [{"id": cid} for cid in chunk_ids]
    injected = len(chunk_ids)
    conn.execute(
        """INSERT INTO recall_traces
           (timestamp, session_id, project, prompt_hash, candidates_count,
            top_k_json, injected, reason, duration_ms)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (ts, "session_pmp_test", project, "hash_pmp", 10,
         json.dumps(top_k), injected, "test", 20.0)
    )
    conn.commit()


# ── PM1: 足够注入次数 → stability 增加 ─────────────────────────────────────────────

def test_pm1_sufficient_injections_boost_stability(conn):
    """PM1: recall_traces 中 chunk 被注入 >= pmp_min_injections(3) → stability 增加。"""
    pmp_min_injections = config.get("store_vfs.pmp_min_injections")  # 3

    _insert_chunk(conn, "pmp_chunk", importance=0.6, stability=5.0)

    # 插入 pmp_min_injections 条 recall_traces，每条都包含 pmp_chunk
    for i in range(pmp_min_injections):
        _insert_recall_trace(conn, project="test", chunk_ids=["pmp_chunk"], hours_ago=float(i + 1))

    stab_before = _get_stability(conn, "pmp_chunk")
    result = apply_primary_memory_persistence(conn, "test", gap_seconds=3600.0)
    stab_after = _get_stability(conn, "pmp_chunk")

    assert stab_after > stab_before, (
        f"PM1: 足够注入次数应触发 PMP，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["pmp_boosted"] >= 1, f"PM1: pmp_boosted 应 >= 1，got {result}"


# ── PM2: 注入次数不足 → 无 PMP 加成 ──────────────────────────────────────────────────

def test_pm2_insufficient_injections_no_boost(conn):
    """PM2: 注入次数 < pmp_min_injections(3) → 无 PMP 加成。"""
    _insert_chunk(conn, "pmp_low_count", importance=0.6, stability=5.0)

    # 只插入 2 条（< 3）
    _insert_recall_trace(conn, project="test", chunk_ids=["pmp_low_count"], hours_ago=1.0)
    _insert_recall_trace(conn, project="test", chunk_ids=["pmp_low_count"], hours_ago=2.0)

    stab_before = _get_stability(conn, "pmp_low_count")
    result = apply_primary_memory_persistence(conn, "test", gap_seconds=3600.0)
    stab_after = _get_stability(conn, "pmp_low_count")

    assert abs(stab_after - stab_before) < 0.001, (
        f"PM2: 注入次数不足不应触发 PMP，before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── PM3: pmp_enabled=False → 无加成 ──────────────────────────────────────────────────

def test_pm3_disabled_no_boost(conn):
    """PM3: store_vfs.pmp_enabled=False → 无任何 PMP 加成。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.pmp_enabled":
            return False
        return original_get(key, project=project)

    _insert_chunk(conn, "pmp_disabled", importance=0.6, stability=5.0)
    for i in range(5):
        _insert_recall_trace(conn, project="test", chunk_ids=["pmp_disabled"], hours_ago=float(i + 1))

    stab_before = _get_stability(conn, "pmp_disabled")
    with mock.patch.object(config, 'get', side_effect=patched_get):
        result = apply_primary_memory_persistence(conn, "test", gap_seconds=3600.0)
    stab_after = _get_stability(conn, "pmp_disabled")

    assert abs(stab_after - stab_before) < 0.001, (
        f"PM3: disabled 时不应有 PMP 加成，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["pmp_boosted"] == 0, f"PM3: pmp_boosted 应为 0，got {result}"


# ── PM4: importance 不足 → 不参与 PMP ────────────────────────────────────────────────

def test_pm4_low_importance_excluded(conn):
    """PM4: importance < pmp_min_importance(0.40) → 不参与 PMP。"""
    _insert_chunk(conn, "pmp_low_imp", importance=0.20, stability=5.0)  # < 0.40
    for i in range(5):
        _insert_recall_trace(conn, project="test", chunk_ids=["pmp_low_imp"], hours_ago=float(i + 1))

    stab_before = _get_stability(conn, "pmp_low_imp")
    result = apply_primary_memory_persistence(conn, "test", gap_seconds=3600.0)
    stab_after = _get_stability(conn, "pmp_low_imp")

    assert abs(stab_after - stab_before) < 0.001, (
        f"PM4: 低 importance chunk 不应参与 PMP，before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── PM5: 更多注入次数 → 更大加成 ─────────────────────────────────────────────────────

def test_pm5_more_injections_more_boost(conn):
    """PM5: 注入次数越多 → pmp_factor 越大 → 加成越大（正比关系）。"""
    pmp_ref_count = config.get("store_vfs.pmp_ref_count")  # 8

    # 场景 A：3 次注入（pmp_factor = 3/8 = 0.375）
    _insert_chunk(conn, "low_inj", project="proj_a", importance=0.6, stability=5.0)
    for i in range(3):
        _insert_recall_trace(conn, project="proj_a", chunk_ids=["low_inj"], hours_ago=float(i + 1))
    result_a = apply_primary_memory_persistence(conn, "proj_a", gap_seconds=3600.0)
    stab_low = _get_stability(conn, "low_inj")

    # 场景 B：8 次注入（pmp_factor = 1.0，最大加成）
    _insert_chunk(conn, "high_inj", project="proj_b", importance=0.6, stability=5.0)
    for i in range(pmp_ref_count):
        _insert_recall_trace(conn, project="proj_b", chunk_ids=["high_inj"], hours_ago=float(i + 1))
    result_b = apply_primary_memory_persistence(conn, "proj_b", gap_seconds=3600.0)
    stab_high = _get_stability(conn, "high_inj")

    assert stab_high > stab_low, (
        f"PM5: 更多注入次数应获得更大加成，stab_low={stab_low:.4f} stab_high={stab_high:.4f}"
    )


# ── PM6: stability 上限 365.0 ─────────────────────────────────────────────────────────

def test_pm6_stability_cap_365(conn):
    """PM6: PMP boost 后 stability 不超过 365.0。"""
    _insert_chunk(conn, "pmp_cap", importance=0.8, stability=364.9)
    for i in range(10):
        _insert_recall_trace(conn, project="test", chunk_ids=["pmp_cap"], hours_ago=float(i + 1))

    apply_primary_memory_persistence(conn, "test", gap_seconds=3600.0)
    stab_after = _get_stability(conn, "pmp_cap")

    assert stab_after <= 365.0, f"PM6: stability 不应超过 365.0，got {stab_after}"


# ── PM7: pmp_boost 可配置 ─────────────────────────────────────────────────────────────

def test_pm7_configurable_boost(conn):
    """PM7: pmp_boost=0.30 时加成比默认 0.10 更大。"""
    original_get = config.get

    _insert_chunk(conn, "boost_chunk", project="proj_boost", importance=0.6, stability=5.0)
    for i in range(8):
        _insert_recall_trace(conn, project="proj_boost", chunk_ids=["boost_chunk"],
                              hours_ago=float(i + 1))

    def patched_30(key, project=None):
        if key == "store_vfs.pmp_boost":
            return 0.30
        return original_get(key, project=project)

    stab_before = _get_stability(conn, "boost_chunk")
    with mock.patch.object(config, 'get', side_effect=patched_30):
        apply_primary_memory_persistence(conn, "proj_boost", gap_seconds=3600.0)
    stab_after_30 = _get_stability(conn, "boost_chunk")
    delta_30 = stab_after_30 - stab_before

    # 重置
    conn.execute("UPDATE memory_chunks SET stability=5.0 WHERE id='boost_chunk'")
    conn.commit()

    stab_before_default = _get_stability(conn, "boost_chunk")
    apply_primary_memory_persistence(conn, "proj_boost", gap_seconds=3600.0)
    stab_after_default = _get_stability(conn, "boost_chunk")
    delta_default = stab_after_default - stab_before_default

    assert delta_30 > delta_default, (
        f"PM7: pmp_boost=0.30 加成应大于默认 0.10，"
        f"delta_30={delta_30:.5f} delta_default={delta_default:.5f}"
    )


# ── PM8: 返回计数正确 ────────────────────────────────────────────────────────────────

def test_pm8_return_counts_correct(conn):
    """PM8: result dict 中 pmp_boosted 和 total_examined 计数正确。"""
    # 3 个满足条件的 chunk（注入 >= 3 次，importance >= 0.40）
    for i in range(3):
        _insert_chunk(conn, f"count_{i}", importance=0.6, stability=5.0)
    for i in range(4):
        _insert_recall_trace(conn, project="test",
                              chunk_ids=["count_0", "count_1", "count_2"],
                              hours_ago=float(i + 1))

    # 1 个 importance 不足（被排除）
    _insert_chunk(conn, "count_low_imp", importance=0.10, stability=5.0)
    for i in range(4):
        _insert_recall_trace(conn, project="test", chunk_ids=["count_low_imp"],
                              hours_ago=float(i + 5))

    result = apply_primary_memory_persistence(conn, "test", gap_seconds=3600.0)

    assert "pmp_boosted" in result, "PM8: result 应含 pmp_boosted key"
    assert "total_examined" in result, "PM8: result 应含 total_examined key"
    assert result["pmp_boosted"] >= 3, f"PM8: pmp_boosted 应 >= 3，got {result}"
    assert result["total_examined"] >= 3, f"PM8: total_examined 应 >= 3，got {result}"


# ── PM9: 无 recall_traces 时正常返回 ─────────────────────────────────────────────────

def test_pm9_no_recall_traces_no_crash(conn):
    """PM9: recall_traces 为空时正常返回，不崩溃，无加成。"""
    _insert_chunk(conn, "pmp_empty", importance=0.6, stability=5.0)
    stab_before = _get_stability(conn, "pmp_empty")

    try:
        result = apply_primary_memory_persistence(conn, "test", gap_seconds=3600.0)
    except Exception as e:
        pytest.fail(f"PM9: 无 recall_traces 时不应崩溃，got {e}")

    stab_after = _get_stability(conn, "pmp_empty")
    assert abs(stab_after - stab_before) < 0.001, (
        f"PM9: 无 recall_traces 时不应有加成，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["pmp_boosted"] == 0, f"PM9: pmp_boosted 应为 0，got {result}"


# ── PM10: sleep_consolidate 包含 pmp_boosted key（集成测试）─────────────────────────

def test_pm10_sleep_consolidate_contains_pmp_key(conn):
    """PM10: sleep_consolidate() 返回 result 中含 pmp_boosted key（sub-op 20 被执行）。"""
    _insert_chunk(conn, "sc_pmp", importance=0.6, stability=5.0)
    for i in range(4):
        _insert_recall_trace(conn, project="test", chunk_ids=["sc_pmp"],
                              hours_ago=float(i + 1))

    result = sleep_consolidate(conn, "test", gap_seconds=3600.0)

    assert "pmp_boosted" in result, (
        f"PM10: sleep_consolidate 应包含 pmp_boosted，got keys={list(result.keys())}"
    )


# ── PM11: top_k_json 支持 dict 和 str 格式的 chunk id ──────────────────────────────

def test_pm11_top_k_json_dict_and_str_formats(conn):
    """PM11: top_k_json 中 chunk id 作为 dict({"id": ...}) 和 str 两种格式都能正确解析。"""
    _insert_chunk(conn, "dict_fmt", importance=0.6, stability=5.0)
    _insert_chunk(conn, "str_fmt", importance=0.6, stability=5.0)

    now_ts = _utcnow().isoformat()
    # dict 格式：[{"id": "dict_fmt"}, ...]
    for i in range(3):
        ts = (_utcnow() - datetime.timedelta(hours=float(i + 1))).isoformat()
        conn.execute(
            """INSERT INTO recall_traces
               (timestamp, session_id, project, prompt_hash, candidates_count,
                top_k_json, injected, reason, duration_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (ts, "s_dict", "test", "h_dict", 5,
             json.dumps([{"id": "dict_fmt"}]), 1, "test", 10.0)
        )

    # str 格式：["str_fmt", ...]
    for i in range(3):
        ts = (_utcnow() - datetime.timedelta(hours=float(i + 4))).isoformat()
        conn.execute(
            """INSERT INTO recall_traces
               (timestamp, session_id, project, prompt_hash, candidates_count,
                top_k_json, injected, reason, duration_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (ts, "s_str", "test", "h_str", 5,
             json.dumps(["str_fmt"]), 1, "test", 10.0)
        )
    conn.commit()

    stab_dict_before = _get_stability(conn, "dict_fmt")
    stab_str_before = _get_stability(conn, "str_fmt")

    result = apply_primary_memory_persistence(conn, "test", gap_seconds=3600.0)

    stab_dict_after = _get_stability(conn, "dict_fmt")
    stab_str_after = _get_stability(conn, "str_fmt")

    assert stab_dict_after > stab_dict_before, (
        f"PM11: dict 格式的 chunk id 应被正确解析，获得 PMP 加成，"
        f"before={stab_dict_before:.4f} after={stab_dict_after:.4f}"
    )
    assert stab_str_after > stab_str_before, (
        f"PM11: str 格式的 chunk id 应被正确解析，获得 PMP 加成，"
        f"before={stab_str_before:.4f} after={stab_str_after:.4f}"
    )


# ── PM12: session_window_hours 限制（超出窗口的记录不计入）───────────────────────────

def test_pm12_session_window_excludes_old_traces(conn):
    """PM12: pmp_session_window_hours=1h 时，>1h 前的 recall_traces 不计入。"""
    original_get = config.get

    def patched_1h(key, project=None):
        if key == "store_vfs.pmp_session_window_hours":
            return 1.0  # 窗口缩小到 1 小时
        return original_get(key, project=project)

    _insert_chunk(conn, "window_chunk", importance=0.6, stability=5.0)

    # 3 条在窗口内（0.2h, 0.5h, 0.8h 前）
    for hours in [0.2, 0.5, 0.8]:
        _insert_recall_trace(conn, project="test", chunk_ids=["window_chunk"],
                              hours_ago=hours)

    # 3 条在窗口外（1.5h, 2h, 3h 前，超出 1h 窗口）
    for hours in [1.5, 2.0, 3.0]:
        _insert_recall_trace(conn, project="test", chunk_ids=["window_chunk"],
                              hours_ago=hours)

    stab_before = _get_stability(conn, "window_chunk")
    with mock.patch.object(config, 'get', side_effect=patched_1h):
        # gap_seconds=0 确保不扩大窗口
        result = apply_primary_memory_persistence(conn, "test", gap_seconds=0.0)
    stab_after = _get_stability(conn, "window_chunk")

    # 窗口内只有 3 条（== pmp_min_injections=3），恰好触发
    assert stab_after > stab_before, (
        f"PM12: 窗口内 3 条注入记录应触发 PMP，before={stab_before:.4f} after={stab_after:.4f}"
    )
