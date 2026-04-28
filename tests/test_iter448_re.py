"""
test_iter448_re.py — iter448: Retroactive Enhancement 单元测试

覆盖：
  RE1: 新 chunk（24h 内）与旧 chunk（entity 重叠 >= 3）→ 旧 chunk stability 增加
  RE2: 无新 chunk（全部超出 re_new_window_hours）→ 无逆行增强
  RE3: entity 重叠数 < re_min_overlap(3) → 无增强（语义联系不足）
  RE4: re_enabled=False → 无任何增强
  RE5: overlap_score 越高 → bonus 越大（正比关系）
  RE6: consolidation 后 stability 不超过 365.0
  RE7: re_scale 可配置（更大 scale → 更大 bonus）
  RE8: importance < re_min_importance(0.45) → chunk 不参与 RE（新旧均需达标）
  RE9: 同一旧 chunk 被多个新 chunk 关联时，取最大 bonus（max 去重，不重复叠加）
  RE10: 返回计数正确（re_boosted, total_examined）

认知科学依据：
  Mednick et al. (2011) PNAS "REM, not incubation, improves creativity" —
    新知识编码后睡眠，逆行增强与之关联的旧记忆（bidirectional consolidation）。
  Stickgold & Walker (2013) — 睡眠优先分诊：importance 高 + 关联新知识 → 优先逆行重放。

OS 类比：Linux page fault 触发的 backward readahead —
  访问 page_N（新知识）时，内核向后预取 page_{N-4..N-1}（旧相关页），
  类比：新 chunk 编码激活的记忆回路逆行激活历史相关 chunk（backward cache warmup）。
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
    apply_retroactive_enhancement,
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
    """Insert chunk with specific encode_context and created_at."""
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


# ── RE1: 新 chunk + 旧 chunk 高重叠 → 旧 chunk stability 增加 ──────────────────────

def test_re1_new_chunk_retroactively_enhances_old(conn):
    """RE1: 新 chunk（24h 内）与旧 chunk（entity 重叠 >= 3）→ 旧 chunk stability 逆行增强。"""
    re_window = config.get("store_vfs.re_new_window_hours")  # 24.0
    re_min_overlap = config.get("store_vfs.re_min_overlap")  # 3

    # 旧 chunk（超出新知识窗口，> 24h 前）
    old_time = _utcnow() - datetime.timedelta(hours=re_window + 10)
    shared_ctx = "kernel_mm, page_cache, rmap, vma, anon_vma, pte_chain, huge_page"
    _insert_chunk(conn, "old_chunk", encode_context=shared_ctx,
                  importance=0.6, stability=5.0, created_at=old_time)

    # 新 chunk（在新知识窗口内，< 24h 前）共享大量 entity
    new_time = _utcnow() - datetime.timedelta(hours=1)
    _insert_chunk(conn, "new_chunk", encode_context=shared_ctx,
                  importance=0.6, stability=5.0, created_at=new_time)

    stab_old_before = _get_stability(conn, "old_chunk")
    stab_new_before = _get_stability(conn, "new_chunk")
    result = apply_retroactive_enhancement(conn, "test")
    stab_old_after = _get_stability(conn, "old_chunk")
    stab_new_after = _get_stability(conn, "new_chunk")

    assert stab_old_after > stab_old_before, (
        f"RE1: 旧 chunk 应获得逆行增强，before={stab_old_before:.4f} after={stab_old_after:.4f}"
    )
    assert result["re_boosted"] >= 1, f"RE1: re_boosted 应 >= 1，got {result}"

    # 新 chunk 自身不应被 RE 直接修改（RE 只修改旧 chunk）
    assert abs(stab_new_after - stab_new_before) < 0.001, (
        f"RE1: 新 chunk 不应被 RE 直接修改，before={stab_new_before:.4f} after={stab_new_after:.4f}"
    )


# ── RE2: 无新 chunk → 无逆行增强 ────────────────────────────────────────────────────

def test_re2_no_new_chunks_no_enhancement(conn):
    """RE2: 全部 chunk 超出 re_new_window_hours → 无新 chunk → 无逆行增强。"""
    re_window = config.get("store_vfs.re_new_window_hours")  # 24.0

    # 两个旧 chunk（超出新知识窗口）
    old_time1 = _utcnow() - datetime.timedelta(hours=re_window + 5)
    old_time2 = _utcnow() - datetime.timedelta(hours=re_window + 10)
    shared_ctx = "kernel_mm, page_cache, rmap, vma, anon_vma"
    _insert_chunk(conn, "old1", encode_context=shared_ctx, importance=0.6,
                  stability=5.0, created_at=old_time1)
    _insert_chunk(conn, "old2", encode_context=shared_ctx, importance=0.6,
                  stability=5.0, created_at=old_time2)

    stab_old1_before = _get_stability(conn, "old1")
    stab_old2_before = _get_stability(conn, "old2")
    result = apply_retroactive_enhancement(conn, "test")
    stab_old1_after = _get_stability(conn, "old1")
    stab_old2_after = _get_stability(conn, "old2")

    assert abs(stab_old1_after - stab_old1_before) < 0.001, (
        f"RE2: 无新 chunk 时旧 chunk 不应被增强，"
        f"before={stab_old1_before:.4f} after={stab_old1_after:.4f}"
    )
    assert abs(stab_old2_after - stab_old2_before) < 0.001, (
        f"RE2: 无新 chunk 时旧 chunk 不应被增强，"
        f"before={stab_old2_before:.4f} after={stab_old2_after:.4f}"
    )
    assert result["re_boosted"] == 0, f"RE2: re_boosted 应为 0，got {result}"


# ── RE3: entity 重叠不足 → 无增强 ───────────────────────────────────────────────────

def test_re3_insufficient_overlap_no_enhancement(conn):
    """RE3: 新旧 chunk entity 重叠数 < re_min_overlap(3) → 语义联系不足 → 无逆行增强。"""
    re_window = config.get("store_vfs.re_new_window_hours")  # 24.0
    re_min_overlap = config.get("store_vfs.re_min_overlap")  # 3

    # 旧 chunk
    old_time = _utcnow() - datetime.timedelta(hours=re_window + 5)
    old_ctx = "auth_token, session_key, oauth_provider, user_id, access_scope"
    _insert_chunk(conn, "old_low_overlap", encode_context=old_ctx,
                  importance=0.6, stability=5.0, created_at=old_time)

    # 新 chunk（仅共享 2 个 entity，< re_min_overlap=3）
    new_time = _utcnow() - datetime.timedelta(hours=1)
    new_ctx = "auth_token, session_key, kernel_mm, page_cache, rmap"  # 2 shared
    _insert_chunk(conn, "new_low_overlap", encode_context=new_ctx,
                  importance=0.6, stability=5.0, created_at=new_time)

    stab_before = _get_stability(conn, "old_low_overlap")
    apply_retroactive_enhancement(conn, "test")
    stab_after = _get_stability(conn, "old_low_overlap")

    assert abs(stab_after - stab_before) < 0.001, (
        f"RE3: 重叠不足的旧 chunk 不应获得增强，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── RE4: re_enabled=False → 无增强 ──────────────────────────────────────────────────

def test_re4_disabled_no_enhancement(conn):
    """RE4: store_vfs.re_enabled=False → 无任何逆行增强。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.re_enabled":
            return False
        return original_get(key, project=project)

    re_window = config.get("store_vfs.re_new_window_hours")
    old_time = _utcnow() - datetime.timedelta(hours=re_window + 5)
    new_time = _utcnow() - datetime.timedelta(hours=1)
    shared_ctx = "kernel_mm, page_cache, rmap, vma, anon_vma, pte_chain, huge_page"

    _insert_chunk(conn, "d_old", encode_context=shared_ctx,
                  importance=0.6, stability=5.0, created_at=old_time)
    _insert_chunk(conn, "d_new", encode_context=shared_ctx,
                  importance=0.6, stability=5.0, created_at=new_time)

    stab_before = _get_stability(conn, "d_old")
    with mock.patch.object(config, 'get', side_effect=patched_get):
        result = apply_retroactive_enhancement(conn, "test")
    stab_after = _get_stability(conn, "d_old")

    assert abs(stab_after - stab_before) < 0.001, (
        f"RE4: disabled 时不应有逆行增强，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["re_boosted"] == 0, f"RE4: re_boosted 应为 0，got {result}"


# ── RE5: overlap_score 越高 → bonus 越大 ──────────────────────────────────────────

def test_re5_higher_overlap_more_bonus(conn):
    """RE5: 新旧 chunk entity 重叠度越高 → overlap_score 越大 → bonus 越大。"""
    re_window = config.get("store_vfs.re_new_window_hours")

    # 旧 chunk A（与新 chunk 完全重叠）
    old_time_a = _utcnow() - datetime.timedelta(hours=re_window + 5)
    high_ctx = "A, B, C, D, E, F, G"  # 7 entity
    _insert_chunk(conn, "old_high_overlap", encode_context=high_ctx,
                  importance=0.6, stability=5.0, created_at=old_time_a)

    # 旧 chunk B（与新 chunk 部分重叠：3/9 entity）
    old_time_b = _utcnow() - datetime.timedelta(hours=re_window + 10)
    low_ctx = "A, B, C, X1, X2, X3"  # 3 shared with new
    _insert_chunk(conn, "old_low_overlap", encode_context=low_ctx,
                  importance=0.6, stability=5.0, created_at=old_time_b)

    # 新 chunk（与 old_high 完全相同，与 old_low 只共享 3）
    new_time = _utcnow() - datetime.timedelta(hours=1)
    _insert_chunk(conn, "new_anchor", encode_context=high_ctx,
                  importance=0.6, stability=5.0, created_at=new_time)

    stab_high_before = _get_stability(conn, "old_high_overlap")
    stab_low_before = _get_stability(conn, "old_low_overlap")
    apply_retroactive_enhancement(conn, "test")
    stab_high_after = _get_stability(conn, "old_high_overlap")
    stab_low_after = _get_stability(conn, "old_low_overlap")

    delta_high = stab_high_after - stab_high_before
    delta_low = stab_low_after - stab_low_before

    assert delta_high > delta_low, (
        f"RE5: 高重叠旧 chunk 的 bonus 应大于低重叠，"
        f"delta_high={delta_high:.5f} delta_low={delta_low:.5f}"
    )
    assert delta_low > 0, (
        f"RE5: 低重叠旧 chunk（overlap >= 3）也应获得增强，delta_low={delta_low:.5f}"
    )


# ── RE6: consolidation 后 stability 不超过 365.0 ─────────────────────────────────

def test_re6_stability_cap_365(conn):
    """RE6: RE 增强后 stability 不超过 365.0。"""
    re_window = config.get("store_vfs.re_new_window_hours")
    shared_ctx = "kernel_mm, page_cache, rmap, vma, anon_vma, pte_chain, huge_page"

    # 旧 chunk stability 接近上限
    old_time = _utcnow() - datetime.timedelta(hours=re_window + 5)
    _insert_chunk(conn, "near_cap_old", encode_context=shared_ctx,
                  importance=0.6, stability=364.9, created_at=old_time)

    # 新 chunk 触发逆行增强
    new_time = _utcnow() - datetime.timedelta(hours=1)
    _insert_chunk(conn, "cap_new", encode_context=shared_ctx,
                  importance=0.6, stability=5.0, created_at=new_time)

    apply_retroactive_enhancement(conn, "test")
    stab_after = _get_stability(conn, "near_cap_old")

    assert stab_after <= 365.0, f"RE6: stability 不应超过 365.0，got {stab_after}"


# ── RE7: re_scale 可配置 ───────────────────────────────────────────────────────────

def test_re7_configurable_scale(conn):
    """RE7: re_scale=0.15 时加成比默认 0.06 更大。"""
    original_get = config.get
    re_window = config.get("store_vfs.re_new_window_hours")
    shared_ctx = "kernel_mm, page_cache, rmap, vma, anon_vma, pte_chain, huge_page"

    old_time = _utcnow() - datetime.timedelta(hours=re_window + 5)
    _insert_chunk(conn, "scale_old", encode_context=shared_ctx,
                  importance=0.6, stability=5.0, created_at=old_time)

    new_time = _utcnow() - datetime.timedelta(hours=1)
    _insert_chunk(conn, "scale_new", encode_context=shared_ctx,
                  importance=0.6, stability=5.0, created_at=new_time)

    def patched_15(key, project=None):
        if key == "store_vfs.re_scale":
            return 0.15
        return original_get(key, project=project)

    stab_before = _get_stability(conn, "scale_old")
    with mock.patch.object(config, 'get', side_effect=patched_15):
        apply_retroactive_enhancement(conn, "test")
    stab_after_15 = _get_stability(conn, "scale_old")
    delta_15 = stab_after_15 - stab_before

    # 重置
    conn.execute("UPDATE memory_chunks SET stability=5.0 WHERE id='scale_old'")
    conn.commit()

    stab_before_default = _get_stability(conn, "scale_old")
    apply_retroactive_enhancement(conn, "test")  # 默认 scale=0.06
    stab_after_default = _get_stability(conn, "scale_old")
    delta_default = stab_after_default - stab_before_default

    assert delta_15 > delta_default, (
        f"RE7: re_scale=0.15 加成应大于默认 0.06，"
        f"delta_15={delta_15:.5f} delta_default={delta_default:.5f}"
    )


# ── RE8: importance 不足 → 不参与 RE ────────────────────────────────────────────────

def test_re8_low_importance_excluded(conn):
    """RE8: importance < re_min_importance(0.45) 的 chunk 不参与 RE（新旧均需达标）。"""
    re_window = config.get("store_vfs.re_new_window_hours")
    re_min_importance = config.get("store_vfs.re_min_importance")  # 0.45
    shared_ctx = "kernel_mm, page_cache, rmap, vma, anon_vma, pte_chain, huge_page"

    old_time = _utcnow() - datetime.timedelta(hours=re_window + 5)
    new_time = _utcnow() - datetime.timedelta(hours=1)

    # 情况 A：旧 chunk importance 不足
    _insert_chunk(conn, "low_imp_old", encode_context=shared_ctx,
                  importance=0.20,  # < 0.45
                  stability=5.0, created_at=old_time)
    _insert_chunk(conn, "high_imp_new_a", encode_context=shared_ctx,
                  importance=0.6, stability=5.0, created_at=new_time)

    stab_old_before = _get_stability(conn, "low_imp_old")
    apply_retroactive_enhancement(conn, "test")
    stab_old_after = _get_stability(conn, "low_imp_old")

    assert abs(stab_old_after - stab_old_before) < 0.001, (
        f"RE8A: 低 importance 旧 chunk 不应被增强，"
        f"before={stab_old_before:.4f} after={stab_old_after:.4f}"
    )

    # 情况 B：新 chunk importance 不足（旧 chunk importance 足够）
    conn.execute("DELETE FROM memory_chunks WHERE project='test'")
    conn.commit()

    _insert_chunk(conn, "high_imp_old_b", encode_context=shared_ctx,
                  importance=0.6, stability=5.0, created_at=old_time)
    _insert_chunk(conn, "low_imp_new_b", encode_context=shared_ctx,
                  importance=0.20,  # < 0.45 → 低 importance 新 chunk 不触发 RE
                  stability=5.0, created_at=new_time)

    stab_old_b_before = _get_stability(conn, "high_imp_old_b")
    apply_retroactive_enhancement(conn, "test")
    stab_old_b_after = _get_stability(conn, "high_imp_old_b")

    assert abs(stab_old_b_after - stab_old_b_before) < 0.001, (
        f"RE8B: 低 importance 新 chunk 不应触发逆行增强，"
        f"before={stab_old_b_before:.4f} after={stab_old_b_after:.4f}"
    )


# ── RE9: 多个新 chunk 关联同一旧 chunk → 取最大 bonus（max 去重）────────────────────

def test_re9_max_bonus_dedup(conn):
    """RE9: 同一旧 chunk 被多个新 chunk 关联时，取最大 bonus（不重复叠加）。"""
    original_get = config.get
    re_window = config.get("store_vfs.re_new_window_hours")

    old_time = _utcnow() - datetime.timedelta(hours=re_window + 5)
    new_time = _utcnow() - datetime.timedelta(hours=1)

    # 旧 chunk
    old_ctx = "A, B, C, D, E, F, G, H, I, J"  # 10 entity
    _insert_chunk(conn, "shared_old", encode_context=old_ctx,
                  importance=0.6, stability=5.0, created_at=old_time)

    # 新 chunk 1：高重叠（10/10 = 1.0 Jaccard）→ large bonus
    _insert_chunk(conn, "new_high", encode_context=old_ctx,
                  importance=0.6, stability=5.0,
                  created_at=new_time)

    # 新 chunk 2：低重叠（3/13 = 0.23 Jaccard）→ small bonus
    _insert_chunk(conn, "new_low", encode_context="A, B, C, X1, X2, X3",
                  importance=0.6, stability=5.0,
                  created_at=new_time + datetime.timedelta(minutes=5))

    stab_before = _get_stability(conn, "shared_old")
    apply_retroactive_enhancement(conn, "test")
    stab_after = _get_stability(conn, "shared_old")
    delta = stab_after - stab_before

    # Expected: max_bonus = high_overlap × re_scale (0.06), not sum of both bonuses
    # re_scale=0.06, overlap_score_high ≈ 1.0 → max_bonus ≈ 0.06
    # If double-counted: bonus ≈ 0.06 + 0.14×0.06 ≈ 0.069
    # max only: bonus ≈ 0.06
    # Stability: 5.0 × (1 + 0.06) = 5.3
    # Check it's not significantly more than single max bonus
    re_scale = original_get("store_vfs.re_scale")
    expected_max = 5.0 * re_scale  # ≈ 0.30 upper bound for single bonus
    assert stab_after > stab_before, f"RE9: 旧 chunk 应获得增强，before={stab_before:.4f} after={stab_after:.4f}"
    # Ensure not excessively large (would happen if bonuses were summed instead of max'd)
    assert delta <= expected_max * 1.1, (  # 10% tolerance
        f"RE9: bonus 不应叠加多个新 chunk 的奖励，delta={delta:.5f} expected_max={expected_max:.5f}"
    )


# ── RE10: 返回计数正确 ────────────────────────────────────────────────────────────────

def test_re10_return_counts_correct(conn):
    """RE10: result dict 中 re_boosted 和 total_examined 计数正确。"""
    re_window = config.get("store_vfs.re_new_window_hours")

    old_time = _utcnow() - datetime.timedelta(hours=re_window + 5)
    new_time = _utcnow() - datetime.timedelta(hours=1)

    shared_ctx = "kernel_mm, page_cache, rmap, vma, anon_vma, pte_chain, huge_page"

    # 2 个旧 chunk（high importance，与新 chunk 高重叠）
    _insert_chunk(conn, "o1", encode_context=shared_ctx, importance=0.6,
                  stability=5.0, created_at=old_time)
    _insert_chunk(conn, "o2", encode_context="kernel_mm, page_cache, rmap, ext4, jbd2, btree, fsync",
                  importance=0.6, stability=5.0,
                  created_at=old_time - datetime.timedelta(hours=2))

    # 1 个低 importance 旧 chunk（不应被计数）
    _insert_chunk(conn, "o_low", encode_context=shared_ctx, importance=0.20,
                  stability=5.0, created_at=old_time - datetime.timedelta(hours=3))

    # 新 chunk（触发逆行增强）
    _insert_chunk(conn, "n1", encode_context=shared_ctx, importance=0.6,
                  stability=5.0, created_at=new_time)

    result = apply_retroactive_enhancement(conn, "test")

    assert "re_boosted" in result, "RE10: result 应含 re_boosted key"
    assert "total_examined" in result, "RE10: result 应含 total_examined key"
    assert result["re_boosted"] >= 1, f"RE10: 至少 1 个旧 chunk 应被增强，got {result}"
    # total_examined = 旧 chunk 数量（importance >= 0.45），o1 + o2 = 2（o_low 被过滤）
    assert result["total_examined"] >= 2, f"RE10: total_examined 应 >= 2，got {result}"
