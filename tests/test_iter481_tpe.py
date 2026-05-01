"""
test_iter481_tpe.py — iter481: Testing Effect 单元测试

覆盖：
  TP1: chunk 出现在近期 recall_traces 中 → stability 额外加成
  TP2: chunk 未出现在 recall_traces 中 → 无 TPE 加成
  TP3: tpe_enabled=False → 无加成
  TP4: importance < tpe_min_importance(0.25) → 不参与 TPE
  TP5: 加成受 tpe_max_boost(0.15) 保护
  TP6: stability 上限 365.0 保护
  TP7: 直接调用 apply_testing_effect → tpe_boosted > 0
  TP8: recall_traces 窗口外的记录不触发 TPE

认知科学依据：
  Roediger & Karpicke (2006) Science — 1周后保留率：测试组64% vs 复习组40%；d≈1.0。
  Karpicke & Roediger (2008) PNAS: 4次检索比1次+3次复习保留率高2倍。

OS 类比：CPU TLB hit — 主动检索命中更新 LRU，降低 eviction 概率。
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

from store_vfs import ensure_schema, apply_testing_effect
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


def _insert_raw(conn, cid, importance=0.6, stability=5.0):
    now_iso = _utcnow().isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count,
            encode_context, session_type_history)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (cid, "test", "observation", "content " + cid, "summary", importance, stability,
         now_iso, now_iso, 0.5, now_iso, 0, "test_ctx", "")
    )
    conn.commit()


def _insert_recall_trace(conn, chunk_id, minutes_ago=1):
    """插入一条包含 chunk_id 的 recall_trace。"""
    import datetime as dt
    ts = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=minutes_ago)
    ts_iso = ts.isoformat()
    import json
    top_k = json.dumps([chunk_id])
    conn.execute(
        """INSERT INTO recall_traces
           (id, timestamp, session_id, project, prompt_hash,
            candidates_count, top_k_json, injected, reason, duration_ms)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (f"trace_{chunk_id}_{minutes_ago}", ts_iso, "sess1", "test",
         "hash1", 10, top_k, 1, "test", 50)
    )
    conn.commit()


def _get_stability(conn, cid: str) -> float:
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


# ── TP1: recall_traces 命中 → stability 额外加成 ────────────────────────────────────────────

def test_tp1_recall_traced_chunk_boosted(conn):
    """TP1: chunk 在近期 recall_traces 中出现 → TPE stability 加成。"""
    _insert_raw(conn, "tp1")
    _insert_recall_trace(conn, "tp1", minutes_ago=1)  # 1分钟内有召回记录

    stab_before = _get_stability(conn, "tp1")
    result = apply_testing_effect(conn, ["tp1"])
    stab_after = _get_stability(conn, "tp1")

    assert stab_after > stab_before, (
        f"TP1: 被检索到的 chunk 应获得 TPE 加成，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["tpe_boosted"] > 0, f"TP1: tpe_boosted 应 > 0，got {result}"


# ── TP2: 无 recall_traces → 无加成 ──────────────────────────────────────────────────────────

def test_tp2_no_recall_trace_no_boost(conn):
    """TP2: chunk 未出现在近期 recall_traces 中 → 无 TPE 加成。"""
    _insert_raw(conn, "tp2")
    # 不插入 recall_trace

    stab_before = _get_stability(conn, "tp2")
    result = apply_testing_effect(conn, ["tp2"])
    stab_after = _get_stability(conn, "tp2")

    assert abs(stab_after - stab_before) < 0.001, (
        f"TP2: 未被检索的 chunk 不应有 TPE 加成，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["tpe_boosted"] == 0, f"TP2: tpe_boosted 应为 0，got {result}"


# ── TP3: tpe_enabled=False → 无加成 ─────────────────────────────────────────────────────────

def test_tp3_disabled_no_boost(conn):
    """TP3: tpe_enabled=False → 无 TPE 加成（即使有 recall_traces）。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.tpe_enabled":
            return False
        return original_get(key, project=project)

    _insert_raw(conn, "tp3")
    _insert_recall_trace(conn, "tp3", minutes_ago=1)

    stab_before = _get_stability(conn, "tp3")
    with mock.patch.object(config, 'get', side_effect=patched_get):
        result = apply_testing_effect(conn, ["tp3"])
    stab_after = _get_stability(conn, "tp3")

    assert abs(stab_after - stab_before) < 0.001, (
        f"TP3: disabled 时不应有 TPE 加成，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["tpe_boosted"] == 0, f"TP3: tpe_boosted 应为 0"


# ── TP4: importance 不足 → 不参与 TPE ────────────────────────────────────────────────────────

def test_tp4_low_importance_no_boost(conn):
    """TP4: importance < tpe_min_importance(0.25) → 不参与 TPE。"""
    _insert_raw(conn, "tp4", importance=0.10)
    _insert_recall_trace(conn, "tp4", minutes_ago=1)

    stab_before = _get_stability(conn, "tp4")
    result = apply_testing_effect(conn, ["tp4"])
    stab_after = _get_stability(conn, "tp4")

    assert abs(stab_after - stab_before) < 0.001, (
        f"TP4: 低 importance 不应触发 TPE，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["tpe_boosted"] == 0


# ── TP5: 加成受 tpe_max_boost 保护 ──────────────────────────────────────────────────────────

def test_tp5_max_boost_cap(conn):
    """TP5: TPE 加成受 tpe_max_boost(0.15) 保护（即使 boost_factor 被放大）。"""
    tpe_max_boost = config.get("store_vfs.tpe_max_boost")  # 0.15
    base = 5.0

    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.tpe_boost_factor":
            return 2.0  # 远超 max_boost=0.15
        return original_get(key, project=project)

    _insert_raw(conn, "tp5", stability=base)
    _insert_recall_trace(conn, "tp5", minutes_ago=1)

    stab_before = _get_stability(conn, "tp5")
    with mock.patch.object(config, 'get', side_effect=patched_get):
        apply_testing_effect(conn, ["tp5"])
    stab_after = _get_stability(conn, "tp5")

    increment = stab_after - stab_before
    max_allowed = base * tpe_max_boost + 0.01
    assert increment <= max_allowed, (
        f"TP5: TPE 增量 {increment:.4f} 不应超过 max_boost 允许的 {max_allowed:.4f}"
    )
    assert stab_after > stab_before, f"TP5: 应有 TPE 加成"


# ── TP6: stability 上限 365.0 ────────────────────────────────────────────────────────────────

def test_tp6_stability_cap_365(conn):
    """TP6: TPE 加成后 stability 不超过 365.0。"""
    _insert_raw(conn, "tp6", stability=364.0)
    _insert_recall_trace(conn, "tp6", minutes_ago=1)

    apply_testing_effect(conn, ["tp6"])
    stab = _get_stability(conn, "tp6")
    assert stab <= 365.0, f"TP6: stability 不应超过 365.0，got {stab}"


# ── TP7: 直接调用返回 tpe_boosted > 0 ────────────────────────────────────────────────────────

def test_tp7_direct_function_boost(conn):
    """TP7: 直接调用 apply_testing_effect，返回 tpe_boosted > 0。"""
    _insert_raw(conn, "tp7")
    _insert_recall_trace(conn, "tp7", minutes_ago=2)

    stab_before = _get_stability(conn, "tp7")
    result = apply_testing_effect(conn, ["tp7"])
    stab_after = _get_stability(conn, "tp7")

    assert result["tpe_boosted"] > 0, f"TP7: tpe_boosted 应 > 0，got {result}"
    assert stab_after > stab_before, (
        f"TP7: 应有 TPE 加成，before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── TP8: 时间窗口外的 recall_traces 不触发 TPE ────────────────────────────────────────────────

def test_tp8_outside_window_no_boost(conn):
    """TP8: recall_traces 超过 tpe_lookback_minutes(5) 窗口 → 不触发 TPE。"""
    _insert_raw(conn, "tp8")
    _insert_recall_trace(conn, "tp8", minutes_ago=10)  # 10分钟前，超出5分钟窗口

    stab_before = _get_stability(conn, "tp8")
    result = apply_testing_effect(conn, ["tp8"])
    stab_after = _get_stability(conn, "tp8")

    assert abs(stab_after - stab_before) < 0.001, (
        f"TP8: 时间窗口外 recall 不应触发 TPE，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["tpe_boosted"] == 0, f"TP8: tpe_boosted 应为 0"
