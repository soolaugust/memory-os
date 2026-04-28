"""
test_iter449_qwr.py — iter449: Quiet Wakefulness Reactivation 单元测试

覆盖：
  QW1: gap in [10min, 8h), 近期 chunk → stability 获得轻度预巩固加成
  QW2: gap < 10min（连续会话）→ 跳过（skipped_reason='gap_too_short'）
  QW3: gap >= 8h（整夜睡眠）→ 跳过（skipped_reason='gap_too_long_use_sc'）
  QW4: qwr_enabled=False → 无任何加成（skipped_reason='disabled'）
  QW5: last_accessed > qwr_recent_hours → chunk 被排除（过期，不在近期窗口）
  QW6: importance < qwr_min_importance(0.55) → 不参与 QWR
  QW7: stability 加成后不超过 365.0（cap 保护）
  QW8: 高 importance chunk 比低 importance chunk 优先获得加成（当 max_chunks 有限时）
  QW9: qwr_boost_factor 可配置（更大 factor → 更大加成）
  QW10: 返回计数正确（qwr_boosted, total_examined, skipped_reason）
  QW11: sleep_consolidate 在 gap_seconds=3600 时调用 QWR 子操作（集成测试）
  QW12: sleep_consolidate 在 gap_seconds=0 时不触发 QWR

认知科学依据：
  Karlsson & Frank (2009) NatNeuro "Awake replay of remote experiences" —
    大鼠清醒安静期海马自发重放先前轨迹（awake sharp-wave ripples），独立于睡眠。
  Tambini et al. (2010) Neuron — 学习后 10min 安静休息，海马-新皮层功能连接增强，
    预测 24h 记忆保留率（r=0.62, p<0.01）。
  Dewar et al. (2012) Psych Sci — 无干扰"空闲期"（unfilled rest）有助于新学习巩固。

OS 类比：Linux incremental pdflush writeback —
  background flusher 每 30s 小批量写回 dirty pages（QWR = 轻量预巩固），
  防止积压到 fsync/sync（SC = 整夜睡眠全量巩固）。
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
    apply_quiet_wakefulness_reactivation,
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


def _insert_chunk(conn, cid, project="test", stability=5.0, importance=0.6,
                  last_accessed=None, created_at=None):
    """Insert a chunk with specific last_accessed for QWR testing."""
    if last_accessed is None:
        last_accessed = _utcnow()
    if created_at is None:
        created_at = last_accessed
    now_iso = _utcnow().isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count,
            encode_context)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0.8, ?, 1, ?)""",
        (cid, project, "decision", f"content {cid}", f"summary {cid}",
         importance, stability, created_at.isoformat(), now_iso,
         last_accessed.isoformat(), "")
    )
    conn.commit()


def _get_stability(conn, cid: str) -> float:
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


# ── QW1: 适当 gap → stability 加成 ──────────────────────────────────────────────

def test_qw1_valid_gap_boosts_stability(conn):
    """QW1: gap=3600s（1小时，在 [10min, 8h) 窗口内）+ 近期 chunk → stability 加成。"""
    recent_time = _utcnow() - datetime.timedelta(hours=1)  # 在 qwr_recent_hours(4h) 窗口内
    _insert_chunk(conn, "recent_chunk", importance=0.6, stability=5.0,
                  last_accessed=recent_time)

    stab_before = _get_stability(conn, "recent_chunk")
    result = apply_quiet_wakefulness_reactivation(conn, "test", gap_seconds=3600.0)
    stab_after = _get_stability(conn, "recent_chunk")

    assert stab_after > stab_before, (
        f"QW1: 近期 chunk 应获得 QWR 加成，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["qwr_boosted"] >= 1, f"QW1: qwr_boosted 应 >= 1，got {result}"
    assert result["skipped_reason"] is None, f"QW1: 不应跳过，got {result}"


# ── QW2: gap < 10min（连续会话）→ 跳过 ───────────────────────────────────────────

def test_qw2_too_short_gap_skipped(conn):
    """QW2: gap=300s（5 分钟 < qwr_min_gap_mins=10）→ 视为连续会话，跳过 QWR。"""
    recent_time = _utcnow() - datetime.timedelta(hours=1)
    _insert_chunk(conn, "cont_chunk", importance=0.6, stability=5.0,
                  last_accessed=recent_time)

    stab_before = _get_stability(conn, "cont_chunk")
    result = apply_quiet_wakefulness_reactivation(conn, "test", gap_seconds=300.0)
    stab_after = _get_stability(conn, "cont_chunk")

    assert abs(stab_after - stab_before) < 0.001, (
        f"QW2: 连续会话不应触发 QWR，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["skipped_reason"] == "gap_too_short", (
        f"QW2: skipped_reason 应为 'gap_too_short'，got {result}"
    )
    assert result["qwr_boosted"] == 0, f"QW2: qwr_boosted 应为 0，got {result}"


# ── QW3: gap >= 8h（整夜睡眠）→ 跳过，由 SC 处理 ────────────────────────────────

def test_qw3_too_long_gap_use_sc(conn):
    """QW3: gap=32400s（9 小时 >= qwr_sleep_threshold_hours=8h）→ 整夜睡眠，由 SC 处理，QWR 跳过。"""
    recent_time = _utcnow() - datetime.timedelta(hours=1)
    _insert_chunk(conn, "sleep_chunk", importance=0.6, stability=5.0,
                  last_accessed=recent_time)

    stab_before = _get_stability(conn, "sleep_chunk")
    result = apply_quiet_wakefulness_reactivation(conn, "test", gap_seconds=32400.0)
    stab_after = _get_stability(conn, "sleep_chunk")

    assert abs(stab_after - stab_before) < 0.001, (
        f"QW3: 整夜睡眠间隔不应触发 QWR（应由 SC 处理），"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["skipped_reason"] == "gap_too_long_use_sc", (
        f"QW3: skipped_reason 应为 'gap_too_long_use_sc'，got {result}"
    )


# ── QW4: qwr_enabled=False → 禁用 ────────────────────────────────────────────────

def test_qw4_disabled_no_boost(conn):
    """QW4: store_vfs.qwr_enabled=False → 无任何 QWR 加成，skipped_reason='disabled'。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.qwr_enabled":
            return False
        return original_get(key, project=project)

    recent_time = _utcnow() - datetime.timedelta(hours=1)
    _insert_chunk(conn, "dis_chunk", importance=0.6, stability=5.0,
                  last_accessed=recent_time)

    stab_before = _get_stability(conn, "dis_chunk")
    with mock.patch.object(config, 'get', side_effect=patched_get):
        result = apply_quiet_wakefulness_reactivation(conn, "test", gap_seconds=3600.0)
    stab_after = _get_stability(conn, "dis_chunk")

    assert abs(stab_after - stab_before) < 0.001, (
        f"QW4: disabled 时不应有 QWR 加成，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["skipped_reason"] == "disabled", (
        f"QW4: skipped_reason 应为 'disabled'，got {result}"
    )


# ── QW5: last_accessed > qwr_recent_hours → 被排除 ───────────────────────────────

def test_qw5_stale_chunk_excluded(conn):
    """QW5: last_accessed > qwr_recent_hours(4h) 的 chunk 不在近期窗口，不参与 QWR。"""
    qwr_recent_hours = config.get("store_vfs.qwr_recent_hours")  # 4.0

    # 超出近期窗口的 chunk（last_accessed 6 小时前，> qwr_recent_hours=4h）
    stale_time = _utcnow() - datetime.timedelta(hours=qwr_recent_hours + 2)
    _insert_chunk(conn, "stale_chunk", importance=0.6, stability=5.0,
                  last_accessed=stale_time)

    stab_before = _get_stability(conn, "stale_chunk")
    result = apply_quiet_wakefulness_reactivation(conn, "test", gap_seconds=3600.0)
    stab_after = _get_stability(conn, "stale_chunk")

    assert abs(stab_after - stab_before) < 0.001, (
        f"QW5: 超出近期窗口的 chunk 不应获得 QWR 加成，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── QW6: importance 不足 → 被排除 ────────────────────────────────────────────────

def test_qw6_low_importance_excluded(conn):
    """QW6: importance < qwr_min_importance(0.55) → 不参与 QWR。"""
    recent_time = _utcnow() - datetime.timedelta(hours=1)
    _insert_chunk(conn, "low_imp_chunk", importance=0.30,  # < 0.55
                  stability=5.0, last_accessed=recent_time)

    stab_before = _get_stability(conn, "low_imp_chunk")
    result = apply_quiet_wakefulness_reactivation(conn, "test", gap_seconds=3600.0)
    stab_after = _get_stability(conn, "low_imp_chunk")

    assert abs(stab_after - stab_before) < 0.001, (
        f"QW6: 低 importance chunk 不应参与 QWR，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── QW7: stability 上限 365.0 ─────────────────────────────────────────────────────

def test_qw7_stability_cap_365(conn):
    """QW7: QWR boost 后 stability 不超过 365.0（cap 保护）。"""
    recent_time = _utcnow() - datetime.timedelta(hours=0.5)
    _insert_chunk(conn, "cap_chunk", importance=0.8, stability=364.9,
                  last_accessed=recent_time)

    apply_quiet_wakefulness_reactivation(conn, "test", gap_seconds=3600.0)
    stab_after = _get_stability(conn, "cap_chunk")

    assert stab_after <= 365.0, f"QW7: stability 不应超过 365.0，got {stab_after}"


# ── QW8: max_chunks 限制时高 importance 优先 ──────────────────────────────────────

def test_qw8_high_importance_prioritized(conn):
    """QW8: 当 max_chunks=2 时，高 importance chunk 优先获得加成，低 importance chunk 被排除。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.qwr_max_chunks":
            return 2  # 只取前 2 个
        return original_get(key, project=project)

    recent_time = _utcnow() - datetime.timedelta(hours=0.5)
    # 高 importance（应进入 top-2）
    _insert_chunk(conn, "high_imp1", importance=0.90, stability=5.0,
                  last_accessed=recent_time)
    _insert_chunk(conn, "high_imp2", importance=0.85, stability=5.0,
                  last_accessed=recent_time)
    # 低 importance（应被排除，不进入 top-2）
    _insert_chunk(conn, "low_imp3", importance=0.56, stability=5.0,
                  last_accessed=recent_time)

    stab_high1_before = _get_stability(conn, "high_imp1")
    stab_high2_before = _get_stability(conn, "high_imp2")
    stab_low3_before = _get_stability(conn, "low_imp3")

    with mock.patch.object(config, 'get', side_effect=patched_get):
        result = apply_quiet_wakefulness_reactivation(conn, "test", gap_seconds=3600.0)

    stab_high1_after = _get_stability(conn, "high_imp1")
    stab_high2_after = _get_stability(conn, "high_imp2")
    stab_low3_after = _get_stability(conn, "low_imp3")

    assert stab_high1_after > stab_high1_before, (
        f"QW8: 高 importance high_imp1 应获得加成，"
        f"before={stab_high1_before:.4f} after={stab_high1_after:.4f}"
    )
    assert stab_high2_after > stab_high2_before, (
        f"QW8: 高 importance high_imp2 应获得加成，"
        f"before={stab_high2_before:.4f} after={stab_high2_after:.4f}"
    )
    assert abs(stab_low3_after - stab_low3_before) < 0.001, (
        f"QW8: 低 importance low_imp3 在 max_chunks=2 时应被排除，"
        f"before={stab_low3_before:.4f} after={stab_low3_after:.4f}"
    )
    assert result["qwr_boosted"] == 2, f"QW8: qwr_boosted 应为 2，got {result}"


# ── QW9: qwr_boost_factor 可配置 ─────────────────────────────────────────────────

def test_qw9_configurable_boost_factor(conn):
    """QW9: qwr_boost_factor=1.10 时加成比默认 1.03 更大。"""
    original_get = config.get
    recent_time = _utcnow() - datetime.timedelta(hours=1)
    _insert_chunk(conn, "factor_chunk", importance=0.7, stability=5.0,
                  last_accessed=recent_time)

    def patched_10(key, project=None):
        if key == "store_vfs.qwr_boost_factor":
            return 1.10
        return original_get(key, project=project)

    stab_before = _get_stability(conn, "factor_chunk")
    with mock.patch.object(config, 'get', side_effect=patched_10):
        apply_quiet_wakefulness_reactivation(conn, "test", gap_seconds=3600.0)
    stab_after_10 = _get_stability(conn, "factor_chunk")
    delta_10 = stab_after_10 - stab_before

    # 重置
    conn.execute("UPDATE memory_chunks SET stability=5.0 WHERE id='factor_chunk'")
    conn.commit()

    stab_before_default = _get_stability(conn, "factor_chunk")
    apply_quiet_wakefulness_reactivation(conn, "test", gap_seconds=3600.0)  # 默认 1.03
    stab_after_default = _get_stability(conn, "factor_chunk")
    delta_default = stab_after_default - stab_before_default

    assert delta_10 > delta_default, (
        f"QW9: qwr_boost_factor=1.10 加成应大于默认 1.03，"
        f"delta_10={delta_10:.5f} delta_default={delta_default:.5f}"
    )


# ── QW10: 返回计数正确 ────────────────────────────────────────────────────────────

def test_qw10_return_counts_correct(conn):
    """QW10: result dict 中 qwr_boosted, total_examined, skipped_reason 计数正确。"""
    recent_time = _utcnow() - datetime.timedelta(hours=1)

    # 3 个满足条件的近期 chunk（importance >= 0.55）
    _insert_chunk(conn, "r1", importance=0.7, stability=5.0, last_accessed=recent_time)
    _insert_chunk(conn, "r2", importance=0.6, stability=5.0,
                  last_accessed=recent_time - datetime.timedelta(minutes=10))
    _insert_chunk(conn, "r3", importance=0.65, stability=5.0,
                  last_accessed=recent_time - datetime.timedelta(minutes=20))

    # 1 个 importance 不足（不应被计数）
    _insert_chunk(conn, "r_low", importance=0.30, stability=5.0, last_accessed=recent_time)

    # 1 个超出近期窗口的 chunk（不应被计数）
    qwr_recent = config.get("store_vfs.qwr_recent_hours")
    stale_time = _utcnow() - datetime.timedelta(hours=qwr_recent + 2)
    _insert_chunk(conn, "r_stale", importance=0.8, stability=5.0, last_accessed=stale_time)

    result = apply_quiet_wakefulness_reactivation(conn, "test", gap_seconds=3600.0)

    assert "qwr_boosted" in result, "QW10: result 应含 qwr_boosted key"
    assert "total_examined" in result, "QW10: result 应含 total_examined key"
    assert "skipped_reason" in result, "QW10: result 应含 skipped_reason key"
    assert result["qwr_boosted"] >= 3, f"QW10: qwr_boosted 应 >= 3（r1, r2, r3 均满足），got {result}"
    assert result["total_examined"] >= 3, f"QW10: total_examined 应 >= 3，got {result}"
    assert result["skipped_reason"] is None, f"QW10: 成功触发时 skipped_reason 应为 None，got {result}"


# ── QW11: sleep_consolidate 在 gap_seconds=3600 时触发 QWR 子操作（集成测试）───────

def test_qw11_sleep_consolidate_triggers_qwr(conn):
    """QW11: sleep_consolidate(gap_seconds=3600) 包含 sub-op 18（QWR），result 含 qwr_boosted。"""
    recent_time = _utcnow() - datetime.timedelta(hours=1)
    _insert_chunk(conn, "sc_qwr", importance=0.7, stability=5.0, last_accessed=recent_time)

    stab_before = _get_stability(conn, "sc_qwr")
    result = sleep_consolidate(conn, "test", gap_seconds=3600.0)
    stab_after = _get_stability(conn, "sc_qwr")

    # sleep_consolidate 应在 result 中包含 qwr_boosted 键
    assert "qwr_boosted" in result, (
        f"QW11: sleep_consolidate 应包含 qwr_boosted，got keys={list(result.keys())}"
    )
    # QWR 应有加成效果
    assert stab_after >= stab_before, (
        f"QW11: sleep_consolidate(gap=3600) 后 stability 不应降低，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── QW12: sleep_consolidate 在 gap_seconds=0 时不触发 QWR ────────────────────────

def test_qw12_sleep_consolidate_no_qwr_when_gap_zero(conn):
    """QW12: sleep_consolidate(gap_seconds=0) 不触发 QWR，result 中不含 qwr_boosted 或为 0。"""
    recent_time = _utcnow() - datetime.timedelta(hours=1)
    _insert_chunk(conn, "sc_no_qwr", importance=0.7, stability=5.0, last_accessed=recent_time)

    result = sleep_consolidate(conn, "test", gap_seconds=0.0)

    # gap_seconds=0 → QWR 不被触发，qwr_boosted 应为 0 或不存在
    qwr_boosted = result.get("qwr_boosted", 0)
    assert qwr_boosted == 0, (
        f"QW12: gap_seconds=0 时 qwr_boosted 应为 0，got {result}"
    )
