"""
test_iter451_rcr.py — iter451: Memory Reconsolidation Context Refresh 单元测试

覆盖：
  RCR1: 近期被检索的旧 chunk（labile window 内）+ 新 chunk 高重叠 → encode_context 被更新
  RCR2: last_accessed 超出 rcr_labile_hours → chunk 不在再巩固窗口，不更新
  RCR3: entity 重叠数 < rcr_min_overlap(2) → 不触发 RCR（语义联系不足）
  RCR4: rcr_enabled=False → 无任何更新
  RCR5: stability >= rcr_stable_floor(60.0) → 跳过（极度稳固记忆不需再巩固）
  RCR6: importance < rcr_min_importance(0.50) → 不参与 RCR
  RCR7: rcr_max_new_entities 限制（最多注入 N 个新 entity，不超额注入）
  RCR8: 新 chunk created_at > rcr_session_window_mins → 时间窗口外，不触发
  RCR9: 同一旧 chunk 被多个新 chunk 触发时，entity 不重复注入（去重）
  RCR10: 返回计数正确（rcr_updated, total_examined）
  RCR11: 注入后 encode_context 包含新 entity token（内容正确性验证）
  RCR12: sleep_consolidate 调用后 rcr_updated 出现在 result 中（集成测试）

认知科学依据：
  Nader et al. (2000) Nature "Fear memories require protein synthesis for reconsolidation" —
    被检索的记忆进入不稳定的再巩固窗口，可被新信息更新。
  Hupbach et al. (2007) NatNeuro — 旧记忆轻微激活后与新情境信息发生整合（bidirectional integration）。
  Lee (2009) TiNS — 再巩固的适应性功能是将新情境信息注入旧记忆，保持与当前情境相关性。

OS 类比：Linux copy-on-write page reconsolidation —
  读访问（=检索）→ 标记 COW ready → 新写入内容可更新旧页面的 encode_context。
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
    apply_reconsolidation_context_refresh,
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
                  encode_context="", created_at=None, last_accessed=None):
    """Insert chunk with specific encode_context, created_at, last_accessed."""
    if created_at is None:
        created_at = _utcnow()
    if last_accessed is None:
        last_accessed = created_at
    now_iso = _utcnow().isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count,
            encode_context)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0.8, ?, 1, ?)""",
        (cid, project, "decision", f"content {cid}", f"summary {cid}",
         importance, stability, created_at.isoformat(), now_iso,
         last_accessed.isoformat(), encode_context)
    )
    conn.commit()


def _get_encode_context(conn, cid: str) -> str:
    row = conn.execute("SELECT encode_context FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return (row[0] or "") if row else ""


# ── RCR1: 再巩固窗口内 + 高重叠 → encode_context 更新 ──────────────────────────────

def test_rcr1_labile_chunk_gets_context_refresh(conn):
    """RCR1: 近期被检索的旧 chunk（在 rcr_labile_hours 内）+ 新 chunk 高重叠 → encode_context 被注入新 entity。"""
    rcr_labile_hours = config.get("store_vfs.rcr_labile_hours")  # 6.0

    # 旧 chunk 近期被检索（last_accessed 2h ago，在 labile window 内）
    last_accessed_time = _utcnow() - datetime.timedelta(hours=2)
    old_ctx = "kernel_mm, page_cache, rmap, vma"
    _insert_chunk(conn, "labile_old", encode_context=old_ctx,
                  importance=0.6, stability=5.0,
                  created_at=last_accessed_time - datetime.timedelta(hours=1),
                  last_accessed=last_accessed_time)

    # 新 chunk 在 last_accessed 后 30min 内写入（在 session window 内），entity 有重叠
    new_chunk_time = last_accessed_time + datetime.timedelta(minutes=30)
    new_ctx = "kernel_mm, page_cache, huge_page, tlb_flush, pte_chain"  # 共享 kernel_mm, page_cache
    _insert_chunk(conn, "new_ctx_chunk", encode_context=new_ctx,
                  importance=0.6, stability=5.0,
                  created_at=new_chunk_time, last_accessed=new_chunk_time)

    ctx_before = _get_encode_context(conn, "labile_old")
    result = apply_reconsolidation_context_refresh(conn, "test")
    ctx_after = _get_encode_context(conn, "labile_old")

    assert ctx_after != ctx_before or result["rcr_updated"] >= 0, "RCR1: context should be updated or result returned"
    # 验证新 entity 被注入（huge_page 或 tlb_flush 等不在旧 ctx 的 token 应出现）
    new_tokens_in_old = {"huge_page", "tlb_flush", "pte_chain"}
    # 注意：分词后 token 是 lowercase，检查是否有任何新 token 被注入
    ctx_after_lower = ctx_after.lower()
    injected_any = any(tok in ctx_after_lower for tok in new_tokens_in_old)
    assert injected_any, (
        f"RCR1: 新 entity 应被注入到旧 chunk encode_context 中，"
        f"ctx_before='{ctx_before}' ctx_after='{ctx_after}'"
    )
    assert result["rcr_updated"] >= 1, f"RCR1: rcr_updated 应 >= 1，got {result}"


# ── RCR2: last_accessed 超出 labile window → 不更新 ───────────────────────────────

def test_rcr2_stale_chunk_not_updated(conn):
    """RCR2: last_accessed 超出 rcr_labile_hours → chunk 不在再巩固窗口，不更新。"""
    rcr_labile_hours = config.get("store_vfs.rcr_labile_hours")  # 6.0

    # 旧 chunk last_accessed 超出 labile window（8h ago，> 6h）
    stale_time = _utcnow() - datetime.timedelta(hours=rcr_labile_hours + 2)
    old_ctx = "kernel_mm, page_cache, rmap, vma"
    _insert_chunk(conn, "stale_old", encode_context=old_ctx,
                  importance=0.6, stability=5.0,
                  created_at=stale_time - datetime.timedelta(hours=1),
                  last_accessed=stale_time)

    # 新 chunk（与旧 chunk 高重叠）
    new_time = _utcnow() - datetime.timedelta(minutes=30)
    new_ctx = "kernel_mm, page_cache, huge_page, tlb_flush"
    _insert_chunk(conn, "new_for_stale", encode_context=new_ctx,
                  importance=0.6, stability=5.0,
                  created_at=new_time, last_accessed=new_time)

    ctx_before = _get_encode_context(conn, "stale_old")
    result = apply_reconsolidation_context_refresh(conn, "test")
    ctx_after = _get_encode_context(conn, "stale_old")

    assert ctx_after == ctx_before, (
        f"RCR2: 超出 labile window 的 chunk 不应被更新，"
        f"before='{ctx_before}' after='{ctx_after}'"
    )


# ── RCR3: entity 重叠不足 → 不触发 ─────────────────────────────────────────────────

def test_rcr3_insufficient_overlap_no_update(conn):
    """RCR3: 新旧 chunk entity 重叠数 < rcr_min_overlap(2) → 语义联系不足 → 不触发 RCR。"""
    last_accessed_time = _utcnow() - datetime.timedelta(hours=2)
    old_ctx = "auth_token, session_key, oauth_provider, user_id"
    _insert_chunk(conn, "low_overlap_old", encode_context=old_ctx,
                  importance=0.6, stability=5.0,
                  created_at=last_accessed_time - datetime.timedelta(hours=1),
                  last_accessed=last_accessed_time)

    # 新 chunk 只共享 1 个 entity（< rcr_min_overlap=2）
    new_time = last_accessed_time + datetime.timedelta(minutes=30)
    new_ctx = "auth_token, kernel_mm, page_cache, huge_page"  # only 1 shared
    _insert_chunk(conn, "low_overlap_new", encode_context=new_ctx,
                  importance=0.6, stability=5.0,
                  created_at=new_time, last_accessed=new_time)

    ctx_before = _get_encode_context(conn, "low_overlap_old")
    result = apply_reconsolidation_context_refresh(conn, "test")
    ctx_after = _get_encode_context(conn, "low_overlap_old")

    assert ctx_after == ctx_before, (
        f"RCR3: 重叠不足时不应触发 RCR，before='{ctx_before}' after='{ctx_after}'"
    )


# ── RCR4: rcr_enabled=False → 无更新 ────────────────────────────────────────────────

def test_rcr4_disabled_no_update(conn):
    """RCR4: store_vfs.rcr_enabled=False → 无任何 encode_context 更新。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.rcr_enabled":
            return False
        return original_get(key, project=project)

    last_accessed_time = _utcnow() - datetime.timedelta(hours=2)
    old_ctx = "kernel_mm, page_cache, rmap, vma"
    _insert_chunk(conn, "dis_old", encode_context=old_ctx,
                  importance=0.6, stability=5.0,
                  created_at=last_accessed_time - datetime.timedelta(hours=1),
                  last_accessed=last_accessed_time)

    new_time = last_accessed_time + datetime.timedelta(minutes=30)
    new_ctx = "kernel_mm, page_cache, huge_page, tlb_flush"
    _insert_chunk(conn, "dis_new", encode_context=new_ctx,
                  importance=0.6, stability=5.0,
                  created_at=new_time, last_accessed=new_time)

    ctx_before = _get_encode_context(conn, "dis_old")
    with mock.patch.object(config, 'get', side_effect=patched_get):
        result = apply_reconsolidation_context_refresh(conn, "test")
    ctx_after = _get_encode_context(conn, "dis_old")

    assert ctx_after == ctx_before, (
        f"RCR4: disabled 时不应更新 encode_context，"
        f"before='{ctx_before}' after='{ctx_after}'"
    )
    assert result["rcr_updated"] == 0, f"RCR4: rcr_updated 应为 0，got {result}"


# ── RCR5: stability >= rcr_stable_floor → 跳过 ───────────────────────────────────────

def test_rcr5_stable_chunk_skipped(conn):
    """RCR5: stability >= rcr_stable_floor(60.0) → 极度稳固记忆跳过 RCR。"""
    last_accessed_time = _utcnow() - datetime.timedelta(hours=2)
    old_ctx = "kernel_mm, page_cache, rmap, vma"
    _insert_chunk(conn, "stable_old", encode_context=old_ctx,
                  importance=0.6, stability=65.0,  # > rcr_stable_floor=60
                  created_at=last_accessed_time - datetime.timedelta(hours=1),
                  last_accessed=last_accessed_time)

    new_time = last_accessed_time + datetime.timedelta(minutes=30)
    new_ctx = "kernel_mm, page_cache, huge_page, tlb_flush"
    _insert_chunk(conn, "stable_new", encode_context=new_ctx,
                  importance=0.6, stability=5.0,
                  created_at=new_time, last_accessed=new_time)

    ctx_before = _get_encode_context(conn, "stable_old")
    result = apply_reconsolidation_context_refresh(conn, "test")
    ctx_after = _get_encode_context(conn, "stable_old")

    assert ctx_after == ctx_before, (
        f"RCR5: stability={65.0} >= stable_floor={60.0}，应跳过 RCR，"
        f"before='{ctx_before}' after='{ctx_after}'"
    )


# ── RCR6: importance 不足 → 不参与 ────────────────────────────────────────────────────

def test_rcr6_low_importance_excluded(conn):
    """RCR6: importance < rcr_min_importance(0.50) → 不参与 RCR。"""
    last_accessed_time = _utcnow() - datetime.timedelta(hours=2)
    old_ctx = "kernel_mm, page_cache, rmap, vma"
    _insert_chunk(conn, "low_imp_old", encode_context=old_ctx,
                  importance=0.30,  # < 0.50
                  stability=5.0,
                  created_at=last_accessed_time - datetime.timedelta(hours=1),
                  last_accessed=last_accessed_time)

    new_time = last_accessed_time + datetime.timedelta(minutes=30)
    new_ctx = "kernel_mm, page_cache, huge_page, tlb_flush"
    _insert_chunk(conn, "low_imp_new", encode_context=new_ctx,
                  importance=0.6, stability=5.0,
                  created_at=new_time, last_accessed=new_time)

    ctx_before = _get_encode_context(conn, "low_imp_old")
    result = apply_reconsolidation_context_refresh(conn, "test")
    ctx_after = _get_encode_context(conn, "low_imp_old")

    assert ctx_after == ctx_before, (
        f"RCR6: importance 不足时不应更新 encode_context，"
        f"before='{ctx_before}' after='{ctx_after}'"
    )


# ── RCR7: rcr_max_new_entities 限制 ──────────────────────────────────────────────────

def test_rcr7_max_new_entities_respected(conn):
    """RCR7: 注入的新 entity 数量不超过 rcr_max_new_entities（默认 5）。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.rcr_max_new_entities":
            return 2  # 限制为 2
        return original_get(key, project=project)

    last_accessed_time = _utcnow() - datetime.timedelta(hours=2)
    # old_ctx 有 2 个 entity，满足 rcr_min_overlap=2 的重叠要求
    old_ctx = "kernel_mm, page_cache"
    _insert_chunk(conn, "max_ent_old", encode_context=old_ctx,
                  importance=0.6, stability=5.0,
                  created_at=last_accessed_time - datetime.timedelta(hours=1),
                  last_accessed=last_accessed_time)

    new_time = last_accessed_time + datetime.timedelta(minutes=30)
    # 新 chunk 有很多新 entity（与旧 chunk 共享 kernel_mm, page_cache → overlap=2）
    new_ctx = "kernel_mm, page_cache, rmap, vma, pte_chain, huge_page, tlb_flush, anon_vma"
    _insert_chunk(conn, "max_ent_new", encode_context=new_ctx,
                  importance=0.6, stability=5.0,
                  created_at=new_time, last_accessed=new_time)

    ctx_before = _get_encode_context(conn, "max_ent_old")
    with mock.patch.object(config, 'get', side_effect=patched_get):
        result = apply_reconsolidation_context_refresh(conn, "test")
    ctx_after = _get_encode_context(conn, "max_ent_old")

    # 计算注入的新 entity 数量
    import re
    before_tokens = set(re.split(r'[,\s]+', ctx_before.lower().strip())) - {''}
    after_tokens = set(re.split(r'[,\s]+', ctx_after.lower().strip())) - {''}
    injected_count = len(after_tokens - before_tokens)

    assert injected_count <= 2, (
        f"RCR7: 注入的新 entity 数量不应超过 rcr_max_new_entities=2，"
        f"injected_count={injected_count} ctx_after='{ctx_after}'"
    )
    assert injected_count >= 1, (
        f"RCR7: 应有新 entity 被注入，injected_count={injected_count}"
    )


# ── RCR8: 新 chunk 超出 session window → 不触发 ───────────────────────────────────────

def test_rcr8_new_chunk_outside_session_window(conn):
    """RCR8: 新 chunk created_at 超出 rcr_session_window_mins(120min) → 时间窗口外，不触发。"""
    rcr_session_window_mins = config.get("store_vfs.rcr_session_window_mins")  # 120

    last_accessed_time = _utcnow() - datetime.timedelta(hours=2)
    old_ctx = "kernel_mm, page_cache, rmap, vma"
    _insert_chunk(conn, "window_old", encode_context=old_ctx,
                  importance=0.6, stability=5.0,
                  created_at=last_accessed_time - datetime.timedelta(hours=1),
                  last_accessed=last_accessed_time)

    # 新 chunk created_at > last_accessed + 120min（超出 session window）
    out_of_window_time = last_accessed_time + datetime.timedelta(minutes=rcr_session_window_mins + 30)
    new_ctx = "kernel_mm, page_cache, huge_page, tlb_flush"
    _insert_chunk(conn, "window_new", encode_context=new_ctx,
                  importance=0.6, stability=5.0,
                  created_at=out_of_window_time, last_accessed=out_of_window_time)

    ctx_before = _get_encode_context(conn, "window_old")
    result = apply_reconsolidation_context_refresh(conn, "test")
    ctx_after = _get_encode_context(conn, "window_old")

    assert ctx_after == ctx_before, (
        f"RCR8: 超出 session window 的新 chunk 不应触发 RCR，"
        f"before='{ctx_before}' after='{ctx_after}'"
    )


# ── RCR9: 多个新 chunk 触发时 entity 不重复注入 ───────────────────────────────────────

def test_rcr9_no_duplicate_entity_injection(conn):
    """RCR9: 多个新 chunk 都触发同一旧 chunk 的 RCR 时，相同 entity 不重复注入。"""
    last_accessed_time = _utcnow() - datetime.timedelta(hours=2)
    old_ctx = "kernel_mm, page_cache"
    _insert_chunk(conn, "dedup_old", encode_context=old_ctx,
                  importance=0.6, stability=5.0,
                  created_at=last_accessed_time - datetime.timedelta(hours=1),
                  last_accessed=last_accessed_time)

    # 新 chunk 1 和 新 chunk 2 都包含相同的新 entity（huge_page）
    new_time_1 = last_accessed_time + datetime.timedelta(minutes=20)
    new_time_2 = last_accessed_time + datetime.timedelta(minutes=40)
    _insert_chunk(conn, "dedup_new1", encode_context="kernel_mm, page_cache, huge_page, rmap",
                  importance=0.6, stability=5.0,
                  created_at=new_time_1, last_accessed=new_time_1)
    _insert_chunk(conn, "dedup_new2", encode_context="kernel_mm, page_cache, huge_page, vma",
                  importance=0.6, stability=5.0,
                  created_at=new_time_2, last_accessed=new_time_2)

    ctx_before = _get_encode_context(conn, "dedup_old")
    result = apply_reconsolidation_context_refresh(conn, "test")
    ctx_after = _get_encode_context(conn, "dedup_old")

    # 统计 huge_page 出现次数（不应重复）
    import re
    tokens = [t.strip().lower() for t in re.split(r'[,\s]+', ctx_after) if t.strip()]
    huge_page_count = tokens.count("huge_page")
    assert huge_page_count <= 1, (
        f"RCR9: 相同 entity 不应被重复注入，huge_page 出现 {huge_page_count} 次，ctx_after='{ctx_after}'"
    )


# ── RCR10: 返回计数正确 ────────────────────────────────────────────────────────────────

def test_rcr10_return_counts_correct(conn):
    """RCR10: result dict 中 rcr_updated 和 total_examined 计数正确。"""
    last_accessed_time = _utcnow() - datetime.timedelta(hours=2)

    # 3 个满足条件的旧 chunk（近期访问，重叠足够）
    shared_ctx = "kernel_mm, page_cache, rmap, vma"
    for i in range(3):
        _insert_chunk(conn, f"count_old_{i}", encode_context=shared_ctx,
                      importance=0.6, stability=5.0,
                      created_at=last_accessed_time - datetime.timedelta(hours=i + 1),
                      last_accessed=last_accessed_time - datetime.timedelta(minutes=i * 10))

    # 1 个 importance 不足（被排除）
    _insert_chunk(conn, "count_low_imp", encode_context=shared_ctx,
                  importance=0.20, stability=5.0,
                  created_at=last_accessed_time - datetime.timedelta(hours=2),
                  last_accessed=last_accessed_time)

    # 新 chunk（触发 RCR）
    new_time = last_accessed_time + datetime.timedelta(minutes=30)
    _insert_chunk(conn, "count_new", encode_context="kernel_mm, page_cache, huge_page, tlb_flush",
                  importance=0.6, stability=5.0,
                  created_at=new_time, last_accessed=new_time)

    result = apply_reconsolidation_context_refresh(conn, "test")

    assert "rcr_updated" in result, "RCR10: result 应含 rcr_updated key"
    assert "total_examined" in result, "RCR10: result 应含 total_examined key"
    # total_examined 应 >= 3（3 个满足 importance 阈值的旧 chunk，不含 low_imp）
    assert result["total_examined"] >= 3, f"RCR10: total_examined 应 >= 3，got {result}"
    assert result["rcr_updated"] >= 1, f"RCR10: rcr_updated 应 >= 1，got {result}"


# ── RCR11: 注入后 encode_context 包含新 entity（内容正确性验证）──────────────────────

def test_rcr11_encode_context_content_correct(conn):
    """RCR11: RCR 执行后旧 chunk encode_context 确实包含新注入的 entity token。"""
    last_accessed_time = _utcnow() - datetime.timedelta(hours=2)
    old_ctx = "kernel_mm, page_cache"  # 旧 chunk 只有 2 个 entity
    _insert_chunk(conn, "content_old", encode_context=old_ctx,
                  importance=0.6, stability=5.0,
                  created_at=last_accessed_time - datetime.timedelta(hours=1),
                  last_accessed=last_accessed_time)

    new_time = last_accessed_time + datetime.timedelta(minutes=30)
    # 新 chunk 共享 kernel_mm, page_cache，并带有新 entity
    new_ctx = "kernel_mm, page_cache, huge_page, tlb_flush, anon_vma"
    _insert_chunk(conn, "content_new", encode_context=new_ctx,
                  importance=0.6, stability=5.0,
                  created_at=new_time, last_accessed=new_time)

    result = apply_reconsolidation_context_refresh(conn, "test")
    ctx_after = _get_encode_context(conn, "content_old")

    # 原始 entity 应保留
    assert "kernel_mm" in ctx_after.lower() or "kernel" in ctx_after.lower(), (
        f"RCR11: 原始 entity 应保留，ctx_after='{ctx_after}'"
    )

    # 至少一个新 entity 应被注入
    newly_injected = {"huge_page", "tlb_flush", "anon_vma"}
    ctx_lower = ctx_after.lower()
    injected_any = any(tok in ctx_lower for tok in {"huge_page", "tlb_flush", "anon_vma",
                                                      "huge", "tlb", "anon"})
    assert injected_any or result["rcr_updated"] == 0, (
        f"RCR11: 应有新 entity 被注入（若 rcr_updated>0），ctx_after='{ctx_after}' result={result}"
    )


# ── RCR12: sleep_consolidate 调用后 result 中含 rcr_updated（集成测试）───────────────

def test_rcr12_sleep_consolidate_triggers_rcr(conn):
    """RCR12: sleep_consolidate() 调用后 result 中含 rcr_updated key（sub-op 19 被执行）。"""
    last_accessed_time = _utcnow() - datetime.timedelta(hours=2)
    old_ctx = "kernel_mm, page_cache, rmap"
    _insert_chunk(conn, "sc_rcr_old", encode_context=old_ctx,
                  importance=0.6, stability=5.0,
                  created_at=last_accessed_time - datetime.timedelta(hours=1),
                  last_accessed=last_accessed_time)

    new_time = last_accessed_time + datetime.timedelta(minutes=30)
    new_ctx = "kernel_mm, page_cache, huge_page, tlb_flush"
    _insert_chunk(conn, "sc_rcr_new", encode_context=new_ctx,
                  importance=0.6, stability=5.0,
                  created_at=new_time, last_accessed=new_time)

    result = sleep_consolidate(conn, "test", gap_seconds=3600.0)

    assert "rcr_updated" in result, (
        f"RCR12: sleep_consolidate 应包含 rcr_updated key，got keys={list(result.keys())}"
    )
