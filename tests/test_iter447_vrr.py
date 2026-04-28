"""
test_iter447_vrr.py — iter447: Von Restorff Sleep Reactivation 单元测试

覆盖：
  VR1: encode_context 与所有邻居低 Jaccard（高孤立度 >= 0.60）→ stability 获得 sleep bonus
  VR2: encode_context 与邻居高度重叠（低孤立度 < 0.60）→ 无 bonus
  VR3: importance < vrr_min_importance(0.50) → 无 bonus
  VR4: vrr_enabled=False → 无任何 bonus
  VR5: 孤立度越高（avg_jaccard 越低）→ bonus 越大
  VR6: consolidation 后 stability 不超过 365.0
  VR7: vrr_scale 可配置（更大 scale → 更大 bonus）
  VR8: 邻居数 < 3 → 无 bonus（避免项目初期误判）
  VR9: 返回计数正确（vrr_boosted, total_examined）
  VR10: 高孤立度 chunk 比低孤立度 chunk 获得更大 stability 增量

认知科学依据：
  Von Restorff (1933) Isolation Effect — 孤立项目比同质项目记忆好 +40-60%。
  McDaniel & Einstein (1986) — 孤立效应在延迟测试中更显著，睡眠巩固选择性保护孤立记忆。
  Huang et al. (2004) — 睡眠后孤立记忆的 delayed recall 比清醒组高约 25%。

OS 类比：Linux huge page mlock + MADV_HUGEPAGE 双标注 —
  独特布局（MADV_HUGEPAGE）+ 锁定（mlock）= kswapd 跳过 + khugepaged 优先处理（双重保护）。
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
    apply_von_restorff_sleep_reactivation,
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
                  encode_context="", created_at=None):
    """Insert chunk with specific encode_context for isolation scoring."""
    if created_at is None:
        created_at = _utcnow()
    now_iso = _utcnow().isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count,
            encode_context)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0.8, ?, 1, ?)""",
        (cid, project, "decision", f"content {cid}", f"summary {cid}",
         importance, stability, created_at.isoformat(), now_iso,
         now_iso, encode_context)
    )
    conn.commit()


def _get_stability(conn, cid: str) -> float:
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


# ── VR1: 高孤立度 chunk → stability 增加 ────────────────────────────────────────

def test_vr1_isolated_chunk_boosted(conn):
    """VR1: encode_context 与邻居低 Jaccard（高孤立度）→ sleep bonus 触发。"""
    base_time = _utcnow() - datetime.timedelta(hours=2)

    # 邻居 chunk（高度同质，共享大量 entity）
    common_ctx = "auth_token, session_key, oauth_provider, user_id, access_scope, token_expiry"
    for i in range(6):
        _insert_chunk(conn, f"neighbor_{i}", encode_context=common_ctx,
                      importance=0.6, stability=5.0,
                      created_at=base_time + datetime.timedelta(minutes=i * 2))

    # 孤立 chunk（完全不同的 entity，插在邻居之间的时间序列中）
    isolated_ctx = "kernel_scheduler, cfs_rq, sched_entity, vruntime, load_balancer"
    _insert_chunk(conn, "isolated", encode_context=isolated_ctx,
                  importance=0.6, stability=5.0,
                  created_at=base_time + datetime.timedelta(minutes=5))

    stab_before = _get_stability(conn, "isolated")
    result = apply_von_restorff_sleep_reactivation(conn, "test")
    stab_after = _get_stability(conn, "isolated")

    assert stab_after > stab_before, (
        f"VR1: 高孤立度 chunk 应获得 VRR sleep bonus，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["vrr_boosted"] >= 1, f"VR1: vrr_boosted 应 >= 1，got {result}"


# ── VR2: 低孤立度（与邻居高重叠）→ 无 bonus ──────────────────────────────────────

def test_vr2_non_isolated_no_boost(conn):
    """VR2: encode_context 与邻居高度重叠（avg_jaccard 高）→ isolation_score < threshold → 无 bonus。"""
    base_time = _utcnow() - datetime.timedelta(hours=2)

    # 所有 chunk 共享相同 entity（包括目标 chunk）
    shared_ctx = "auth_token, session_key, oauth_provider, user_id, access_scope"
    for i in range(8):
        _insert_chunk(conn, f"similar_{i}", encode_context=shared_ctx,
                      importance=0.6, stability=5.0,
                      created_at=base_time + datetime.timedelta(minutes=i * 2))

    # 目标 chunk 也与邻居完全相同
    _insert_chunk(conn, "non_isolated", encode_context=shared_ctx,
                  importance=0.6, stability=5.0,
                  created_at=base_time + datetime.timedelta(minutes=9))

    stab_before = _get_stability(conn, "non_isolated")
    apply_von_restorff_sleep_reactivation(conn, "test")
    stab_after = _get_stability(conn, "non_isolated")

    assert abs(stab_after - stab_before) < 0.001, (
        f"VR2: 低孤立度 chunk 不应获得 VRR bonus，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── VR3: importance 不足 → 无 bonus ─────────────────────────────────────────────

def test_vr3_low_importance_no_boost(conn):
    """VR3: importance < vrr_min_importance(0.50) → 无 VRR bonus。"""
    base_time = _utcnow() - datetime.timedelta(hours=2)

    common_ctx = "auth_token, session_key, oauth_provider, user_id, access_scope"
    for i in range(6):
        _insert_chunk(conn, f"nb_imp_{i}", encode_context=common_ctx,
                      importance=0.6, created_at=base_time + datetime.timedelta(minutes=i * 2))

    # 低 importance 的孤立 chunk（< 0.50）
    isolated_ctx = "kernel_rcu, spinlock, atomic_ops, memory_barrier, cache_line"
    _insert_chunk(conn, "low_imp_isolated", encode_context=isolated_ctx,
                  importance=0.30,  # < 0.50
                  stability=5.0,
                  created_at=base_time + datetime.timedelta(minutes=5))

    stab_before = _get_stability(conn, "low_imp_isolated")
    apply_von_restorff_sleep_reactivation(conn, "test")
    stab_after = _get_stability(conn, "low_imp_isolated")

    assert abs(stab_after - stab_before) < 0.001, (
        f"VR3: 低 importance 孤立 chunk 不应获得 VRR bonus，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── VR4: vrr_enabled=False → 无 bonus ───────────────────────────────────────────

def test_vr4_disabled_no_boost(conn):
    """VR4: store_vfs.vrr_enabled=False → 无任何 VRR 加成。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.vrr_enabled":
            return False
        return original_get(key, project=project)

    base_time = _utcnow() - datetime.timedelta(hours=2)
    common_ctx = "auth_token, session_key, oauth_provider, user_id"
    for i in range(6):
        _insert_chunk(conn, f"d_nb_{i}", encode_context=common_ctx,
                      importance=0.6, created_at=base_time + datetime.timedelta(minutes=i * 2))

    isolated_ctx = "kernel_scheduler, cfs_rq, sched_entity, vruntime"
    _insert_chunk(conn, "d_isolated", encode_context=isolated_ctx,
                  importance=0.6, stability=5.0,
                  created_at=base_time + datetime.timedelta(minutes=5))

    stab_before = _get_stability(conn, "d_isolated")
    with mock.patch.object(config, 'get', side_effect=patched_get):
        result = apply_von_restorff_sleep_reactivation(conn, "test")
    stab_after = _get_stability(conn, "d_isolated")

    assert abs(stab_after - stab_before) < 0.001, (
        f"VR4: disabled 时不应有 VRR bonus，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["vrr_boosted"] == 0, f"VR4: vrr_boosted 应为 0，got {result}"


# ── VR5: 孤立度越高 → bonus 越大 ────────────────────────────────────────────────

def test_vr5_higher_isolation_more_boost(conn):
    """VR5: isolation_score 越高，sleep bonus 越大（线性正比于 isolation_score × vrr_scale）。"""
    base_time = _utcnow() - datetime.timedelta(hours=2)

    # 邻居 chunk：5 个 entity
    neighbor_tokens = ["A", "B", "C", "D", "E"]
    nb_ctx = ", ".join(neighbor_tokens)
    for i in range(6):
        _insert_chunk(conn, f"nb5_{i}", encode_context=nb_ctx,
                      importance=0.6,
                      created_at=base_time + datetime.timedelta(minutes=i * 2))

    # 高孤立度：与邻居完全不同（Jaccard=0 → isolation=1.0）
    high_isolated_ctx = "kernel_rcu, spinlock, atomic_ops, memory_barrier, tlb_flush"
    _insert_chunk(conn, "very_isolated", encode_context=high_isolated_ctx,
                  importance=0.6, stability=5.0,
                  created_at=base_time + datetime.timedelta(minutes=3))

    # 中等孤立度：与邻居有部分重叠（Jaccard ≈ 0.2）
    medium_isolated_ctx = "A, B, kernel_rcu, spinlock, atomic_ops"  # 2/8 overlap
    _insert_chunk(conn, "medium_isolated", encode_context=medium_isolated_ctx,
                  importance=0.6, stability=5.0,
                  created_at=base_time + datetime.timedelta(minutes=13))

    stab_high_before = _get_stability(conn, "very_isolated")
    stab_med_before = _get_stability(conn, "medium_isolated")
    apply_von_restorff_sleep_reactivation(conn, "test")
    stab_high_after = _get_stability(conn, "very_isolated")
    stab_med_after = _get_stability(conn, "medium_isolated")

    delta_high = stab_high_after - stab_high_before
    delta_med = stab_med_after - stab_med_before

    assert delta_high > delta_med, (
        f"VR5: 高孤立度 chunk 的 bonus 应大于中等孤立度，"
        f"delta_high={delta_high:.5f} delta_med={delta_med:.5f}"
    )


# ── VR6: consolidation 后 stability 不超过 365.0 ─────────────────────────────────

def test_vr6_stability_cap_365(conn):
    """VR6: VRR bonus 后 stability 不超过 365.0。"""
    base_time = _utcnow() - datetime.timedelta(hours=2)

    common_ctx = "auth_token, session_key, oauth_provider, user_id"
    for i in range(6):
        _insert_chunk(conn, f"cap_nb_{i}", encode_context=common_ctx,
                      importance=0.6, created_at=base_time + datetime.timedelta(minutes=i * 2))

    # 孤立 chunk，stability 接近上限
    isolated_ctx = "kernel_rcu, spinlock, atomic_ops, memory_barrier, tlb_flush"
    _insert_chunk(conn, "near_cap", encode_context=isolated_ctx,
                  importance=0.6, stability=364.9,
                  created_at=base_time + datetime.timedelta(minutes=5))

    apply_von_restorff_sleep_reactivation(conn, "test")
    stab_after = _get_stability(conn, "near_cap")

    assert stab_after <= 365.0, f"VR6: stability 不应超过 365.0，got {stab_after}"


# ── VR7: vrr_scale 可配置 ────────────────────────────────────────────────────────

def test_vr7_configurable_scale(conn):
    """VR7: vrr_scale=0.25 时加成比默认 0.10 更大。"""
    original_get = config.get
    base_time = _utcnow() - datetime.timedelta(hours=2)

    common_ctx = "auth_token, session_key, oauth_provider, user_id"
    for i in range(6):
        _insert_chunk(conn, f"sc_nb_{i}", encode_context=common_ctx,
                      importance=0.6, created_at=base_time + datetime.timedelta(minutes=i * 2))

    isolated_ctx = "kernel_rcu, spinlock, atomic_ops, memory_barrier, tlb_flush"
    _insert_chunk(conn, "scale_chunk", encode_context=isolated_ctx,
                  importance=0.6, stability=5.0,
                  created_at=base_time + datetime.timedelta(minutes=5))

    def patched_25(key, project=None):
        if key == "store_vfs.vrr_scale":
            return 0.25
        return original_get(key, project=project)

    stab_before = _get_stability(conn, "scale_chunk")
    with mock.patch.object(config, 'get', side_effect=patched_25):
        apply_von_restorff_sleep_reactivation(conn, "test")
    stab_after_25 = _get_stability(conn, "scale_chunk")
    delta_25 = stab_after_25 - stab_before

    # 重置
    conn.execute("UPDATE memory_chunks SET stability=5.0 WHERE id='scale_chunk'")
    conn.commit()

    stab_before_default = _get_stability(conn, "scale_chunk")
    apply_von_restorff_sleep_reactivation(conn, "test")  # 默认 scale=0.10
    stab_after_default = _get_stability(conn, "scale_chunk")
    delta_default = stab_after_default - stab_before_default

    assert delta_25 > delta_default, (
        f"VR7: vrr_scale=0.25 加成应大于默认 0.10，"
        f"delta_25={delta_25:.5f} delta_default={delta_default:.5f}"
    )


# ── VR8: 邻居数 < 3 → 无 bonus（避免项目初期误判）──────────────────────────────

def test_vr8_too_few_neighbors_no_boost(conn):
    """VR8: 邻居数 < 3 时 isolation_score = 0.0 → 无 VRR bonus（项目初期保护）。"""
    base_time = _utcnow() - datetime.timedelta(hours=2)

    # 只有 2 个邻居（< 3）
    common_ctx = "auth_token, session_key"
    _insert_chunk(conn, "few_nb_1", encode_context=common_ctx, importance=0.6,
                  created_at=base_time)
    _insert_chunk(conn, "few_nb_2", encode_context=common_ctx, importance=0.6,
                  created_at=base_time + datetime.timedelta(minutes=2))

    # 孤立 chunk，但邻居只有 2 个 → 应该没有 bonus
    isolated_ctx = "kernel_rcu, spinlock, atomic_ops, memory_barrier, tlb_flush"
    _insert_chunk(conn, "few_nb_isolated", encode_context=isolated_ctx,
                  importance=0.6, stability=5.0,
                  created_at=base_time + datetime.timedelta(minutes=1))

    stab_before = _get_stability(conn, "few_nb_isolated")
    result = apply_von_restorff_sleep_reactivation(conn, "test")
    stab_after = _get_stability(conn, "few_nb_isolated")

    # 总 chunk = 3，每个 chunk 的窗口内邻居最多 2 个 → 均不满足 >= 3 邻居要求
    assert abs(stab_after - stab_before) < 0.001, (
        f"VR8: 邻居不足时不应触发 VRR，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── VR9: 返回计数正确 ─────────────────────────────────────────────────────────────

def test_vr9_return_counts_correct(conn):
    """VR9: result dict 中 vrr_boosted 和 total_examined 计数正确。"""
    base_time = _utcnow() - datetime.timedelta(hours=2)

    # 同质背景：6 个 auth-related chunk
    auth_ctx = "auth_token, session_key, oauth_provider, user_id, access_scope, token_expiry"
    for i in range(6):
        _insert_chunk(conn, f"bg_{i}", encode_context=auth_ctx,
                      importance=0.6,
                      created_at=base_time + datetime.timedelta(minutes=i * 3))

    # 1 个高孤立度 chunk（完全不同）
    _insert_chunk(conn, "iso1", encode_context="kernel_rcu, spinlock, atomic_ops, tlb_flush, vma",
                  importance=0.6, stability=5.0,
                  created_at=base_time + datetime.timedelta(minutes=5))

    # 1 个 importance 不足的孤立 chunk（不应被计数）
    _insert_chunk(conn, "iso_low", encode_context="bpf_prog, xdp_rx, tc_egress, netfilter, nft",
                  importance=0.30,  # < 0.50
                  stability=5.0,
                  created_at=base_time + datetime.timedelta(minutes=11))

    result = apply_von_restorff_sleep_reactivation(conn, "test")

    assert "vrr_boosted" in result, "VR9: result 应含 vrr_boosted key"
    assert "total_examined" in result, "VR9: result 应含 total_examined key"
    # iso1 应被 boost（高孤立度 + importance 足够），iso_low 因 importance < 0.50 不被查询
    assert result["vrr_boosted"] >= 1, f"VR9: 至少 1 个孤立 chunk 应被 boost，got {result}"
    # total_examined 包含所有 importance >= 0.50 的 chunk（bg×6 + iso1 = 7）
    assert result["total_examined"] >= 7, f"VR9: total_examined 应 >= 7，got {result}"


# ── VR10: 高孤立度 vs 低孤立度 → 加成差异清晰 ─────────────────────────────────────

def test_vr10_isolation_gradient(conn):
    """VR10: 完全孤立 chunk 的 stability 增量 > 部分孤立 > 非孤立（三档梯度）。"""
    vrr_min_isolation = config.get("store_vfs.vrr_min_isolation")  # 0.60
    base_time = _utcnow() - datetime.timedelta(hours=2)

    # 背景：5 个 entity 的同质 chunk（邻居）
    bg_ctx = "P, Q, R, S, T"
    for i in range(8):
        _insert_chunk(conn, f"grad_bg_{i}", encode_context=bg_ctx,
                      importance=0.6,
                      created_at=base_time + datetime.timedelta(minutes=i * 2))

    # 完全孤立（Jaccard=0, isolation≈1.0）
    _insert_chunk(conn, "full_iso", encode_context="X1, X2, X3, X4, X5",
                  importance=0.6, stability=5.0,
                  created_at=base_time + datetime.timedelta(minutes=3))

    # 部分孤立（1/7 overlap → Jaccard ≈ 0.14, isolation ≈ 0.86）
    _insert_chunk(conn, "part_iso", encode_context="P, X6, X7, X8, X9",
                  importance=0.6, stability=5.0,
                  created_at=base_time + datetime.timedelta(minutes=9))

    # 非孤立（完全相同 → Jaccard=1.0, isolation=0.0 → 低于阈值）
    _insert_chunk(conn, "no_iso", encode_context=bg_ctx,
                  importance=0.6, stability=5.0,
                  created_at=base_time + datetime.timedelta(minutes=15))

    stab_full_before = _get_stability(conn, "full_iso")
    stab_part_before = _get_stability(conn, "part_iso")
    stab_no_before = _get_stability(conn, "no_iso")

    apply_von_restorff_sleep_reactivation(conn, "test")

    stab_full_after = _get_stability(conn, "full_iso")
    stab_part_after = _get_stability(conn, "part_iso")
    stab_no_after = _get_stability(conn, "no_iso")

    delta_full = stab_full_after - stab_full_before
    delta_part = stab_part_after - stab_part_before
    delta_no = stab_no_after - stab_no_before

    # 完全孤立 > 部分孤立（两者都应有 bonus，因为都 >= vrr_min_isolation=0.60）
    assert delta_full > delta_part, (
        f"VR10: 完全孤立 delta > 部分孤立 delta，"
        f"delta_full={delta_full:.5f} delta_part={delta_part:.5f}"
    )
    # 非孤立应无 bonus（isolation_score < vrr_min_isolation）
    assert abs(delta_no) < 0.001, (
        f"VR10: 非孤立 chunk 不应有 bonus，delta_no={delta_no:.5f}"
    )
