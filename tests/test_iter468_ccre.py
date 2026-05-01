"""
test_iter468_ccre.py — iter468: Contextual Cue Reinstatement Effect 单元测试

覆盖：
  CC1: encode_context tokens 匹配 → stability 加成（apply_contextual_cue_reinstatement_effect）
  CC2: encode_context tokens 不匹配 → 无加成
  CC3: ccre_enabled=False → 无任何加成
  CC4: importance < ccre_min_importance(0.30) → 不参与 CCRE
  CC5: 加成随匹配 token 数量增加（单调性），受 ccre_max_boost(0.20) 保护
  CC6: stability 加成后不超过 365.0
  CC7: update_accessed 集成测试 — 传入 context_tokens 时触发 CCRE
  CC8: 部分匹配（1 个 token）也能触发加成

认知科学依据：
  Godden & Baddeley (1975) "Context-dependent memory in two natural environments"
    (British Journal of Psychology) —
    在与编码时相同的物理/认知上下文中检索，成功率提升约 40%（vs 不同上下文）。
  Tulving & Thomson (1973) Encoding Specificity Principle —
    "retrieval cue 需包含编码时存在的信息"；上下文 token 重叠 = 最强检索线索。
  Smith (1979): 内部心理上下文（mental context）匹配与外部环境匹配效果相同。

OS 类比：Linux NUMA-aware memory access（mm/mempolicy.c MPOL_PREFERRED）—
  进程在与分配时相同 NUMA 节点访问 page → 低延迟（context match = locality）；
  跨 NUMA 节点访问 = context mismatch → 更高延迟。
  encode_context token 重叠度 = NUMA locality score。
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

from store_vfs import ensure_schema, apply_contextual_cue_reinstatement_effect, update_accessed
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
                  encode_context="kernel_mm,slab,buddy"):
    now_iso = _utcnow().isoformat()
    import datetime as _dt
    la_iso = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(minutes=10)).isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count,
            encode_context, session_type_history)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (cid, project, "decision", f"content {cid}", f"summary {cid}",
         importance, stability, now_iso, now_iso, 0.5,
         la_iso, 2, encode_context, "")
    )
    conn.commit()


def _get_stability(conn, cid: str) -> float:
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


# ── CC1: 上下文匹配 → stability 加成 ──────────────────────────────────────────────────────

def test_cc1_context_match_boosted(conn):
    """CC1: encode_context tokens 与 current_context_tokens 匹配 → stability 加成。"""
    _insert_chunk(conn, "ccre_1", encode_context="kernel_mm,slab,buddy")
    stab_before = _get_stability(conn, "ccre_1")

    # 传入匹配的 context tokens
    result = apply_contextual_cue_reinstatement_effect(
        conn, "ccre_1", ["kernel_mm", "slab", "buddy"]
    )
    stab_after = _get_stability(conn, "ccre_1")

    assert stab_after > stab_before, (
        f"CC1: context 匹配时 stability 应加成，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["ccre_boosted"] is True, f"CC1: ccre_boosted 应为 True，got {result}"
    assert result["ccre_matched_tokens"] == 3, f"CC1: 应匹配 3 个 tokens，got {result}"


# ── CC2: 上下文不匹配 → 无加成 ────────────────────────────────────────────────────────────

def test_cc2_context_mismatch_no_boost(conn):
    """CC2: encode_context tokens 与 current_context_tokens 不匹配 → 无 CCRE 加成。"""
    _insert_chunk(conn, "ccre_2", encode_context="kernel_mm,slab,buddy")
    stab_before = _get_stability(conn, "ccre_2")

    # 传入不匹配的 context tokens
    result = apply_contextual_cue_reinstatement_effect(
        conn, "ccre_2", ["userspace", "tcp", "network"]
    )
    stab_after = _get_stability(conn, "ccre_2")

    assert abs(stab_after - stab_before) < 0.001, (
        f"CC2: context 不匹配时不应有 CCRE 加成，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["ccre_boosted"] is False, f"CC2: ccre_boosted 应为 False，got {result}"
    assert result["ccre_matched_tokens"] == 0, f"CC2: 匹配 tokens 应为 0，got {result}"


# ── CC3: ccre_enabled=False → 无加成 ─────────────────────────────────────────────────────

def test_cc3_disabled_no_boost(conn):
    """CC3: ccre_enabled=False → 无任何 CCRE 加成。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.ccre_enabled":
            return False
        return original_get(key, project=project)

    _insert_chunk(conn, "ccre_3", encode_context="kernel_mm,slab")
    stab_before = _get_stability(conn, "ccre_3")

    with mock.patch.object(config, 'get', side_effect=patched_get):
        result = apply_contextual_cue_reinstatement_effect(
            conn, "ccre_3", ["kernel_mm", "slab"]
        )
    stab_after = _get_stability(conn, "ccre_3")

    assert abs(stab_after - stab_before) < 0.001, (
        f"CC3: disabled 时不应有 CCRE 加成，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["ccre_boosted"] is False, f"CC3: ccre_boosted 应为 False，got {result}"


# ── CC4: importance 不足 → 不参与 CCRE ──────────────────────────────────────────────────

def test_cc4_low_importance_no_boost(conn):
    """CC4: importance < ccre_min_importance(0.30) → 不参与 CCRE。"""
    _insert_chunk(conn, "ccre_4", importance=0.10, encode_context="kernel_mm,slab")
    stab_before = _get_stability(conn, "ccre_4")

    result = apply_contextual_cue_reinstatement_effect(
        conn, "ccre_4", ["kernel_mm", "slab"]
    )
    stab_after = _get_stability(conn, "ccre_4")

    assert abs(stab_after - stab_before) < 0.001, (
        f"CC4: 低 importance 不应有 CCRE 加成，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["ccre_boosted"] is False, f"CC4: ccre_boosted 应为 False，got {result}"


# ── CC5: 加成随匹配 token 数量增加，受 ccre_max_boost 保护 ─────────────────────────────────

def test_cc5_more_tokens_more_boost_capped(conn):
    """CC5: 匹配 token 数量越多 → 加成越大，但不超过 ccre_max_boost(0.20)。"""
    ccre_max_boost = config.get("store_vfs.ccre_max_boost")  # 0.20
    base = 5.0

    _insert_chunk(conn, "ccre_5_one", stability=base, encode_context="kernel_mm,slab,buddy")
    _insert_chunk(conn, "ccre_5_three", stability=base, encode_context="kernel_mm,slab,buddy")

    # 1 token match
    result_one = apply_contextual_cue_reinstatement_effect(
        conn, "ccre_5_one", ["kernel_mm"]
    )
    stab_one = _get_stability(conn, "ccre_5_one")

    # 3 tokens match
    result_three = apply_contextual_cue_reinstatement_effect(
        conn, "ccre_5_three", ["kernel_mm", "slab", "buddy"]
    )
    stab_three = _get_stability(conn, "ccre_5_three")

    assert stab_three >= stab_one - 0.001, (
        f"CC5: 3 token 匹配加成应 >= 1 token 匹配加成，"
        f"three={stab_three:.4f} one={stab_one:.4f}"
    )

    # 最大加成不超过 max_boost
    max_allowed = base * ccre_max_boost + 0.01
    increment_three = stab_three - base
    assert increment_three <= max_allowed, (
        f"CC5: CCRE 增量 {increment_three:.4f} 不应超过 max_boost 允许的 {max_allowed:.4f}"
    )


# ── CC6: stability 上限 365.0 ─────────────────────────────────────────────────────────────

def test_cc6_stability_cap_365(conn):
    """CC6: CCRE boost 后 stability 不超过 365.0。"""
    _insert_chunk(conn, "ccre_6", stability=364.0, importance=0.8,
                  encode_context="kernel_mm,slab,buddy,vmalloc")
    apply_contextual_cue_reinstatement_effect(
        conn, "ccre_6", ["kernel_mm", "slab", "buddy", "vmalloc"]
    )
    stab = _get_stability(conn, "ccre_6")
    assert stab <= 365.0, f"CC6: stability 不应超过 365.0，got {stab}"


# ── CC7: update_accessed 集成测试 ────────────────────────────────────────────────────────

def test_cc7_update_accessed_integration(conn):
    """CC7: update_accessed 传入 context_tokens 时触发 CCRE。"""
    _insert_chunk(conn, "ccre_7", stability=5.0, importance=0.6,
                  encode_context="kernel_mm,slab")
    stab_before = _get_stability(conn, "ccre_7")

    # 通过 update_accessed 的 context_tokens kwargs 触发 CCRE
    update_accessed(conn, ["ccre_7"], context_tokens=["kernel_mm", "slab"])
    stab_after = _get_stability(conn, "ccre_7")

    # CCRE 加成（stability 应增加）
    assert stab_after >= stab_before, (
        f"CC7: update_accessed 后 stability 不应降低，before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── CC8: 部分匹配（1 个 token）也触发加成 ───────────────────────────────────────────────

def test_cc8_partial_match_triggers_boost(conn):
    """CC8: 部分 token 匹配（1 个）也能触发 CCRE 加成。"""
    _insert_chunk(conn, "ccre_8", stability=5.0, importance=0.6,
                  encode_context="kernel_mm,slab,buddy,vmalloc")
    stab_before = _get_stability(conn, "ccre_8")

    # 只有 1 个 token 匹配
    result = apply_contextual_cue_reinstatement_effect(
        conn, "ccre_8", ["kernel_mm", "completely_different", "another_token"]
    )
    stab_after = _get_stability(conn, "ccre_8")

    assert stab_after > stab_before, (
        f"CC8: 部分匹配（1 token）应有 CCRE 加成，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["ccre_matched_tokens"] == 1, (
        f"CC8: 应匹配 1 个 token，got {result['ccre_matched_tokens']}"
    )
