"""
test_iter450_pme.py — iter450: Predictive Memory Encoding 单元测试

覆盖：
  PM1: 近期同项目 recall_traces 注入次数 >= pme_min_queries(3) → 新 chunk stability 增加
  PM2: 注入次数 < pme_min_queries → 无 PME 加成
  PM3: pme_enabled=False → 无任何加成
  PM4: importance < pme_min_importance(0.45) → 不参与 PME
  PM5: 更多注入次数 → 更大 pme_factor → 更大加成（正比关系）
  PM6: stability 加成后不超过 365.0（cap 保护）
  PM7: pme_boost 可配置（更大 boost → 更大加成）
  PM8: 返回值正确（新 stability 而非旧值）
  PM9: 无 recall_traces 时正常返回（不崩溃）
  PM10: insert_chunk 在有充足 recall_traces 时自动触发 PME

认知科学依据：
  Roediger & Karpicke (2011) "The Critical Importance of Retrieval for Learning" —
    预期将来会被测试的知识，编码时更深度加工，记忆痕迹更强（Test-Expectancy Effect）。
  Szpunar et al. (2014) NatComm — 中间插入测试不仅强化当时记忆，还提升后续编码质量。

OS 类比：Linux MADV_SEQUENTIAL hint —
  应用程序提前告诉内核"这段内存将被顺序读取"，内核提前扩大 readahead 窗口；
  类比：同话题近期高检索频率 = 测试预期 → 新写入同话题 chunk 获得预测性加成。
"""
import sys
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
    apply_predictive_memory_encoding,
    insert_chunk,
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


def _insert_chunk_raw(conn, cid, project="test", stability=5.0, importance=0.6):
    """Insert a chunk with controlled stability/importance, bypassing PME."""
    now_iso = _utcnow().isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count,
            encode_context)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0.8, ?, 1, ?)""",
        (cid, project, "decision", f"content {cid}", f"summary {cid}",
         importance, stability, now_iso, now_iso, now_iso, "")
    )
    conn.commit()


def _get_stability(conn, cid: str) -> float:
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


def _insert_recall_trace(conn, project="test", injected=1, hours_ago=1.0):
    """Insert a recall trace simulating a past injection."""
    ts = (_utcnow() - datetime.timedelta(hours=hours_ago)).isoformat()
    conn.execute(
        """INSERT INTO recall_traces
           (timestamp, session_id, project, prompt_hash, candidates_count,
            top_k_json, injected, reason, duration_ms)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (ts, "session_pme_test", project, "hash_pme", 10, "[]", injected, "test", 20.0)
    )
    conn.commit()


# ── PM1: 足够注入次数 → stability 增加 ─────────────────────────────────────────────

def test_pm1_sufficient_queries_boost_stability(conn):
    """PM1: 近期 recall_traces 注入次数 >= pme_min_queries(3) → 新 chunk stability 增加。"""
    pme_min_queries = config.get("store_vfs.pme_min_queries")  # 3

    # 插入 pme_min_queries 条 recall_traces（injected > 0）
    for i in range(pme_min_queries):
        _insert_recall_trace(conn, project="test", injected=2, hours_ago=float(i + 1))

    _insert_chunk_raw(conn, "pme_chunk", importance=0.6, stability=5.0)

    stab_before = _get_stability(conn, "pme_chunk")
    new_stab = apply_predictive_memory_encoding(conn, "pme_chunk", "test",
                                                  chunk_type="decision",
                                                  base_stability=5.0)
    stab_after = _get_stability(conn, "pme_chunk")

    assert stab_after > stab_before, (
        f"PM1: 足够注入次数应触发 PME，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert new_stab > stab_before, f"PM1: 返回值应大于原始 stability，got {new_stab}"


# ── PM2: 注入次数不足 → 无 PME 加成 ──────────────────────────────────────────────────

def test_pm2_insufficient_queries_no_boost(conn):
    """PM2: 注入次数 < pme_min_queries(3) → 无 PME 加成。"""
    # 只插入 2 条（< 3）
    _insert_recall_trace(conn, project="test", injected=1, hours_ago=1.0)
    _insert_recall_trace(conn, project="test", injected=1, hours_ago=2.0)

    _insert_chunk_raw(conn, "pme_low_count", importance=0.6, stability=5.0)

    stab_before = _get_stability(conn, "pme_low_count")
    apply_predictive_memory_encoding(conn, "pme_low_count", "test",
                                      chunk_type="decision",
                                      base_stability=5.0)
    stab_after = _get_stability(conn, "pme_low_count")

    assert abs(stab_after - stab_before) < 0.001, (
        f"PM2: 注入次数不足不应触发 PME，before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── PM3: pme_enabled=False → 无加成 ──────────────────────────────────────────────────

def test_pm3_disabled_no_boost(conn):
    """PM3: store_vfs.pme_enabled=False → 无任何 PME 加成。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.pme_enabled":
            return False
        return original_get(key, project=project)

    for i in range(5):
        _insert_recall_trace(conn, project="test", injected=2, hours_ago=float(i + 1))

    _insert_chunk_raw(conn, "pme_disabled", importance=0.6, stability=5.0)

    stab_before = _get_stability(conn, "pme_disabled")
    with mock.patch.object(config, 'get', side_effect=patched_get):
        apply_predictive_memory_encoding(conn, "pme_disabled", "test",
                                          chunk_type="decision",
                                          base_stability=5.0)
    stab_after = _get_stability(conn, "pme_disabled")

    assert abs(stab_after - stab_before) < 0.001, (
        f"PM3: disabled 时不应有 PME 加成，before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── PM4: importance 不足 → 不参与 PME ────────────────────────────────────────────────

def test_pm4_low_importance_excluded(conn):
    """PM4: importance < pme_min_importance(0.45) → 不参与 PME。"""
    for i in range(5):
        _insert_recall_trace(conn, project="test", injected=2, hours_ago=float(i + 1))

    _insert_chunk_raw(conn, "pme_low_imp", importance=0.20, stability=5.0)  # < 0.45

    stab_before = _get_stability(conn, "pme_low_imp")
    apply_predictive_memory_encoding(conn, "pme_low_imp", "test",
                                      chunk_type="decision",
                                      base_stability=5.0)
    stab_after = _get_stability(conn, "pme_low_imp")

    assert abs(stab_after - stab_before) < 0.001, (
        f"PM4: 低 importance chunk 不应参与 PME，before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── PM5: 更多注入次数 → 更大加成 ─────────────────────────────────────────────────────

def test_pm5_more_queries_more_boost(conn):
    """PM5: 注入次数越多 → pme_factor 越大 → 加成越大（正比关系）。"""
    pme_ref_count = config.get("store_vfs.pme_ref_count")  # 10

    # 场景 A：3 次注入（pme_min_queries，pme_factor = 3/10 = 0.30）
    for i in range(3):
        _insert_recall_trace(conn, project="proj_a", injected=1, hours_ago=float(i + 1))
    _insert_chunk_raw(conn, "low_count", project="proj_a", importance=0.6, stability=5.0)
    apply_predictive_memory_encoding(conn, "low_count", "proj_a",
                                      chunk_type="decision", base_stability=5.0)
    stab_low = _get_stability(conn, "low_count")

    # 场景 B：10 次注入（pme_factor = 1.0，最大加成）
    for i in range(10):
        _insert_recall_trace(conn, project="proj_b", injected=1, hours_ago=float(i + 1))
    _insert_chunk_raw(conn, "high_count", project="proj_b", importance=0.6, stability=5.0)
    apply_predictive_memory_encoding(conn, "high_count", "proj_b",
                                      chunk_type="decision", base_stability=5.0)
    stab_high = _get_stability(conn, "high_count")

    assert stab_high > stab_low, (
        f"PM5: 更多注入次数应获得更大加成，stab_low={stab_low:.4f} stab_high={stab_high:.4f}"
    )


# ── PM6: stability 上限 365.0 ─────────────────────────────────────────────────────────

def test_pm6_stability_cap_365(conn):
    """PM6: PME boost 后 stability 不超过 365.0。"""
    for i in range(10):
        _insert_recall_trace(conn, project="test", injected=2, hours_ago=float(i + 1))

    _insert_chunk_raw(conn, "pme_cap", importance=0.8, stability=364.9)

    apply_predictive_memory_encoding(conn, "pme_cap", "test",
                                      chunk_type="decision", base_stability=364.9)
    stab_after = _get_stability(conn, "pme_cap")

    assert stab_after <= 365.0, f"PM6: stability 不应超过 365.0，got {stab_after}"


# ── PM7: pme_boost 可配置 ─────────────────────────────────────────────────────────────

def test_pm7_configurable_boost(conn):
    """PM7: pme_boost=0.30 时加成比默认 0.12 更大。"""
    original_get = config.get

    for i in range(10):
        _insert_recall_trace(conn, project="proj_boost", injected=1, hours_ago=float(i + 1))

    def patched_30(key, project=None):
        if key == "store_vfs.pme_boost":
            return 0.30
        return original_get(key, project=project)

    _insert_chunk_raw(conn, "boost_chunk", project="proj_boost", importance=0.6, stability=5.0)
    stab_before = _get_stability(conn, "boost_chunk")

    with mock.patch.object(config, 'get', side_effect=patched_30):
        apply_predictive_memory_encoding(conn, "boost_chunk", "proj_boost",
                                          chunk_type="decision", base_stability=5.0)
    stab_after_30 = _get_stability(conn, "boost_chunk")
    delta_30 = stab_after_30 - stab_before

    # 重置
    conn.execute("UPDATE memory_chunks SET stability=5.0 WHERE id='boost_chunk'")
    conn.commit()

    stab_before_default = _get_stability(conn, "boost_chunk")
    apply_predictive_memory_encoding(conn, "boost_chunk", "proj_boost",
                                      chunk_type="decision", base_stability=5.0)
    stab_after_default = _get_stability(conn, "boost_chunk")
    delta_default = stab_after_default - stab_before_default

    assert delta_30 > delta_default, (
        f"PM7: pme_boost=0.30 加成应大于默认 0.12，"
        f"delta_30={delta_30:.5f} delta_default={delta_default:.5f}"
    )


# ── PM8: 返回值正确 ────────────────────────────────────────────────────────────────────

def test_pm8_return_value_correct(conn):
    """PM8: apply_predictive_memory_encoding 返回新 stability 值（而非旧值）。"""
    for i in range(10):
        _insert_recall_trace(conn, project="test", injected=1, hours_ago=float(i + 1))

    _insert_chunk_raw(conn, "pme_return", importance=0.6, stability=5.0)
    stab_before = _get_stability(conn, "pme_return")

    returned = apply_predictive_memory_encoding(conn, "pme_return", "test",
                                                 chunk_type="decision", base_stability=5.0)
    stab_after = _get_stability(conn, "pme_return")

    assert returned == stab_after, (
        f"PM8: 返回值应等于写入的新 stability，returned={returned:.4f} db={stab_after:.4f}"
    )
    assert returned > stab_before, (
        f"PM8: 返回值应大于原始 stability，returned={returned:.4f} before={stab_before:.4f}"
    )


# ── PM9: 无 recall_traces 时正常返回 ─────────────────────────────────────────────────

def test_pm9_no_recall_traces_no_crash(conn):
    """PM9: recall_traces 为空时 apply_predictive_memory_encoding 正常返回，不崩溃。"""
    _insert_chunk_raw(conn, "pme_empty", importance=0.6, stability=5.0)

    stab_before = _get_stability(conn, "pme_empty")
    try:
        returned = apply_predictive_memory_encoding(conn, "pme_empty", "test",
                                                     chunk_type="decision", base_stability=5.0)
    except Exception as e:
        pytest.fail(f"PM9: 无 recall_traces 时不应崩溃，got {e}")

    stab_after = _get_stability(conn, "pme_empty")
    # 无注入记录 → 无加成
    assert abs(stab_after - stab_before) < 0.001, (
        f"PM9: 无 recall_traces 时不应有加成，before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── PM10: insert_chunk 在有充足 recall_traces 时自动触发 PME（集成测试）──────────────

def test_pm10_insert_chunk_triggers_pme(conn):
    """PM10: insert_chunk 在有足够近期注入记录时自动触发 PME。"""
    pme_min_queries = config.get("store_vfs.pme_min_queries")  # 3
    pme_min_importance = config.get("store_vfs.pme_min_importance")  # 0.45

    # 插入足够的 recall_traces
    for i in range(pme_min_queries + 2):
        _insert_recall_trace(conn, project="integ_test", injected=2, hours_ago=float(i + 1))

    now_iso = _utcnow().isoformat()
    # 通过 insert_chunk 创建 chunk（importance 高于阈值）
    chunk_d = {
        "id": "pme_integ",
        "project": "integ_test",
        "chunk_type": "decision",
        "content": "决策：使用 MGLRU 替换旧版 LRU",
        "summary": "MGLRU 替换 LRU 决策",
        "importance": 0.7,
        "stability": 5.0,
        "encode_context": "mglru, lru, memory",
        "tags": [],
        "created_at": now_iso,
        "updated_at": now_iso,
        "source_session": "test_session",
        "retrievability": 0.9,
        "last_accessed": now_iso,
    }
    insert_chunk(conn, chunk_d)

    stab_after = _get_stability(conn, "pme_integ")

    # 有足够注入记录 → PME 应触发 → stability > 5.0
    assert stab_after > 5.0, (
        f"PM10: insert_chunk 应在有充足 recall_traces 时触发 PME，"
        f"stability after insert = {stab_after:.4f}（期望 > 5.0）"
    )
