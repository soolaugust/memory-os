"""
test_iter444_cre.py — iter444: Contextual Reinstatement Effect 单元测试

覆盖：
  CR1: chunk.encode_context 与 session_active_entities 有 >= cre_min_overlap(2) 重叠 → stability 增加
  CR2: entity 重叠不足（< 2）→ 无 consolidation
  CR3: importance < cre_min_importance(0.40) → 无 consolidation
  CR4: cre_enabled=False → 无任何 consolidation
  CR5: 重叠比例越高 → bonus 越大（overlap_ratio 正比）
  CR6: consolidation 后 stability 不超过 365.0
  CR7: cre_bonus 可配置（更大 bonus → 更大加成）
  CR8: session_active_entities 太稀疏（< min_overlap 个 entity）→ 无 consolidation
  CR9: 返回计数正确（cre_consolidated, total_examined）
  CR10: 显式 session_accessed_ids 参数精确构建情境集合

认知科学依据：
  Smith (1979) "Remembering in and out of context" — 情境再现时提取成功率高 40-50%。
  Tulving (1983) Encoding Specificity Principle — 检索线索与编码情境越接近，提取效率越高。
  Godden & Baddeley (1975) — 同情境下学习和回忆的记忆表现比跨情境高 ~40%（环境依赖记忆）。

OS 类比：Linux NUMA-aware khugepaged —
  khugepaged 优先合并同一 NUMA node 内相邻热页为 2MB hugepage；
  session 活跃情境 = 当前 NUMA node，与活跃情境高度重叠的 chunk = 同 node 热页。

注：session 活跃情境通过 last_accessed >= now-2h 窗口构建。
    session chunks 使用 last_accessed_days_ago=0.03 (≈43min, within 2h window)
    target chunks  使用 last_accessed_days_ago=0.5  (12h, outside 2h window, not in session context)
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
    apply_contextual_reinstatement_consolidation,
)
import config

# session chunk: last_accessed within 2h window (session entities 构建来源)
SESSION_ACCESSED_DAYS = 0.03   # ~43 minutes ago → within 2h window
# target chunk: last_accessed outside 2h window (不贡献到 session entities)
TARGET_ACCESSED_DAYS = 0.5     # 12 hours ago → outside 2h window


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _ago_iso(days: float = 0.0) -> str:
    return (datetime.datetime.now(datetime.timezone.utc) -
            datetime.timedelta(days=days)).isoformat()


def _make_entities(*names) -> str:
    return ", ".join(names)


def _insert_chunk(conn, cid, project="test", stability=5.0, importance=0.6,
                  encode_context="", last_accessed_days_ago=TARGET_ACCESSED_DAYS):
    now = _now_iso()
    last_accessed = _ago_iso(days=last_accessed_days_ago)
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count,
            encode_context)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0.9, ?, 1, ?)""",
        (cid, project, "decision", f"content {cid}", f"summary {cid}",
         importance, stability, now, now, last_accessed, encode_context)
    )
    conn.commit()


def _insert_session_chunk(conn, cid, project="test", importance=0.7, encode_context=""):
    """Insert a session chunk (last_accessed within 2h window → contributes to session entities)."""
    _insert_chunk(conn, cid, project=project, importance=importance,
                  encode_context=encode_context,
                  last_accessed_days_ago=SESSION_ACCESSED_DAYS)


def _get_stability(conn, cid: str) -> float:
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


# ── CR1: entity 重叠足够 → stability 增加 ──────────────────────────────────────────

def test_cr1_contextual_chunk_consolidated(conn):
    """CR1: chunk 与 session_active_entities 有 >= cre_min_overlap(2) 重叠 → stability 增加。"""
    session_entities = ["auth_token", "session_key", "oauth_provider", "access_scope"]
    session_ec = _make_entities(*session_entities)

    # session 中最近访问的 chunk（within 2h window → 构成活跃情境）
    _insert_session_chunk(conn, "session1", encode_context=session_ec)
    _insert_session_chunk(conn, "session2", encode_context=session_ec)

    # 目标 chunk：与 session 有足够重叠（outside 2h window → 不污染 session 情境）
    target_ec = _make_entities("auth_token", "session_key", "extra_entity")
    _insert_chunk(conn, "target", importance=0.6, stability=5.0,
                  encode_context=target_ec)

    stab_before = _get_stability(conn, "target")
    result = apply_contextual_reinstatement_consolidation(conn, "test")
    stab_after = _get_stability(conn, "target")

    assert stab_after > stab_before, (
        f"CR1: 情境重叠足够的 chunk 应获得 sleep 巩固加成，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["cre_consolidated"] >= 1, f"CR1: cre_consolidated 应 >= 1，got {result}"


# ── CR2: entity 重叠不足 → 无 consolidation ────────────────────────────────────────

def test_cr2_insufficient_overlap_no_consolidation(conn):
    """CR2: chunk entity 与 session_active_entities 重叠数 < cre_min_overlap(2) → 无 consolidation。"""
    # session 情境：4 个 auth-related entities
    session_ec = _make_entities("auth_token", "session_key", "oauth_provider", "access_scope")
    _insert_session_chunk(conn, "session_ctx2", encode_context=session_ec)

    # 目标 chunk：只有 1 个 entity 重叠（< cre_min_overlap=2），且在 2h 外不污染情境
    target_ec = _make_entities("auth_token", "unrelated_X", "unrelated_Y")
    _insert_chunk(conn, "low_overlap", importance=0.6, stability=5.0,
                  encode_context=target_ec)

    stab_before = _get_stability(conn, "low_overlap")
    # cre_min_overlap=2 (default), chunk has only 1 overlap → no consolidation
    apply_contextual_reinstatement_consolidation(conn, "test")
    stab_after = _get_stability(conn, "low_overlap")

    assert abs(stab_after - stab_before) < 0.001, (
        f"CR2: entity 重叠不足时不应 consolidate，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── CR3: importance 不足 → 无 consolidation ─────────────────────────────────────────

def test_cr3_low_importance_no_consolidation(conn):
    """CR3: importance < cre_min_importance(0.40) → 无 consolidation。"""
    shared_ec = _make_entities("auth_token", "session_key", "oauth_provider")
    _insert_session_chunk(conn, "session_ctx3", encode_context=shared_ec)

    # 低 importance chunk（outside 2h window）
    _insert_chunk(conn, "low_imp", importance=0.20, stability=5.0,  # < 0.40
                  encode_context=shared_ec)

    stab_before = _get_stability(conn, "low_imp")
    apply_contextual_reinstatement_consolidation(conn, "test")
    stab_after = _get_stability(conn, "low_imp")

    assert abs(stab_after - stab_before) < 0.001, (
        f"CR3: 低 importance chunk 不应受 CRE 巩固，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── CR4: cre_enabled=False → 无 consolidation ───────────────────────────────────────

def test_cr4_disabled_no_consolidation(conn):
    """CR4: store_vfs.cre_enabled=False → 无任何 CRE consolidation。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.cre_enabled":
            return False
        return original_get(key, project=project)

    shared_ec = _make_entities("auth_token", "session_key", "oauth_provider", "access_scope")
    _insert_session_chunk(conn, "session_ctx4", encode_context=shared_ec)
    _insert_chunk(conn, "disabled_cre", importance=0.6, stability=5.0,
                  encode_context=shared_ec)

    stab_before = _get_stability(conn, "disabled_cre")
    with mock.patch.object(config, 'get', side_effect=patched_get):
        result = apply_contextual_reinstatement_consolidation(conn, "test")
    stab_after = _get_stability(conn, "disabled_cre")

    assert abs(stab_after - stab_before) < 0.001, (
        f"CR4: disabled 时不应 consolidate，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["cre_consolidated"] == 0, f"CR4: cre_consolidated 应为 0，got {result}"


# ── CR5: 重叠比例越高 → bonus 越大 ──────────────────────────────────────────────────

def test_cr5_higher_overlap_more_boost(conn):
    """CR5: 与 session_active_entities 重叠比例越高，bonus 越大。"""
    # session 活跃情境：5 个 entity（within 2h window）
    session_entities = ["A", "B", "C", "D", "E"]
    session_ec = _make_entities(*session_entities)
    _insert_session_chunk(conn, "session_ctx5", encode_context=session_ec)

    # 高重叠 chunk：4/4 = 1.0 重叠率（outside 2h window，不污染情境）
    high_ec = _make_entities("A", "B", "C", "D")
    _insert_chunk(conn, "high_overlap", importance=0.7, stability=5.0,
                  encode_context=high_ec)

    # 低重叠 chunk：2/5 = 0.4 重叠率
    low_ec = _make_entities("A", "B", "X", "Y", "Z")
    _insert_chunk(conn, "low_overlap_r", importance=0.7, stability=5.0,
                  encode_context=low_ec)

    stab_high_before = _get_stability(conn, "high_overlap")
    stab_low_before = _get_stability(conn, "low_overlap_r")
    apply_contextual_reinstatement_consolidation(conn, "test")
    stab_high_after = _get_stability(conn, "high_overlap")
    stab_low_after = _get_stability(conn, "low_overlap_r")

    delta_high = stab_high_after - stab_high_before
    delta_low = stab_low_after - stab_low_before

    assert delta_high > delta_low, (
        f"CR5: 高重叠比例 chunk 加成应多于低重叠，"
        f"delta_high={delta_high:.5f} delta_low={delta_low:.5f}"
    )


# ── CR6: consolidation 后 stability 不超过 365.0 ─────────────────────────────────────

def test_cr6_stability_cap_365(conn):
    """CR6: CRE consolidation 后 stability 不超过 365.0。"""
    shared_ec = _make_entities("A", "B", "C", "D")
    _insert_session_chunk(conn, "session_ctx6", encode_context=shared_ec)
    _insert_chunk(conn, "near_cap", importance=0.8, stability=364.9,
                  encode_context=shared_ec)

    apply_contextual_reinstatement_consolidation(conn, "test")
    stab_after = _get_stability(conn, "near_cap")

    assert stab_after <= 365.0, f"CR6: stability 不应超过 365.0，got {stab_after}"


# ── CR7: cre_bonus 可配置 ────────────────────────────────────────────────────────────

def test_cr7_configurable_bonus(conn):
    """CR7: cre_bonus=0.30 时加成比默认 0.10 更大。"""
    original_get = config.get
    shared_ec = _make_entities("A", "B", "C", "D")
    _insert_session_chunk(conn, "session_ctx7", encode_context=shared_ec)
    _insert_chunk(conn, "bonus_chunk", importance=0.7, stability=5.0,
                  encode_context=shared_ec)

    def patched_30(key, project=None):
        if key == "store_vfs.cre_bonus":
            return 0.30
        return original_get(key, project=project)

    stab_before = _get_stability(conn, "bonus_chunk")
    with mock.patch.object(config, 'get', side_effect=patched_30):
        apply_contextual_reinstatement_consolidation(conn, "test")
    stab_after_30 = _get_stability(conn, "bonus_chunk")
    delta_30 = stab_after_30 - stab_before

    # 重置
    conn.execute("UPDATE memory_chunks SET stability=5.0 WHERE id='bonus_chunk'")
    conn.commit()

    stab_before_default = _get_stability(conn, "bonus_chunk")
    apply_contextual_reinstatement_consolidation(conn, "test")  # 默认 bonus=0.10
    stab_after_default = _get_stability(conn, "bonus_chunk")
    delta_default = stab_after_default - stab_before_default

    assert delta_30 > delta_default, (
        f"CR7: cre_bonus=0.30 加成应大于默认 0.10，"
        f"delta_30={delta_30:.5f} delta_default={delta_default:.5f}"
    )


# ── CR8: session_active_entities 太稀疏 → 无 consolidation ──────────────────────────

def test_cr8_sparse_session_no_consolidation(conn):
    """CR8: session 情境 entity 数量 < cre_min_overlap → 无 CRE consolidation。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.cre_min_overlap":
            return 5  # 要求 5 个重叠，而 session 只有 1 个 entity
        return original_get(key, project=project)

    # session 只有 1 个 entity（< 5）
    sparse_session_ec = _make_entities("only_one")
    _insert_session_chunk(conn, "sparse_session", encode_context=sparse_session_ec)

    target_ec = _make_entities("only_one", "A", "B")
    _insert_chunk(conn, "sparse_target", importance=0.6, stability=5.0,
                  encode_context=target_ec)

    stab_before = _get_stability(conn, "sparse_target")
    with mock.patch.object(config, 'get', side_effect=patched_get):
        result = apply_contextual_reinstatement_consolidation(conn, "test")
    stab_after = _get_stability(conn, "sparse_target")

    # session 情境只有 1 entity，< min_overlap=5 → 直接返回空
    assert result["cre_consolidated"] == 0, (
        f"CR8: 稀疏 session 情境不应触发 CRE，got {result}"
    )


# ── CR9: 返回计数正确 ──────────────────────────────────────────────────────────────

def test_cr9_return_counts_correct(conn):
    """CR9: result dict 中 cre_consolidated 和 total_examined 计数正确。"""
    session_ec = _make_entities("alpha", "beta", "gamma", "delta", "epsilon")
    _insert_session_chunk(conn, "session_9a", encode_context=session_ec)
    _insert_session_chunk(conn, "session_9b", encode_context=session_ec)

    # 2 个满足条件的 chunk（与 session 足够重叠）
    _insert_chunk(conn, "c1", importance=0.6, stability=5.0,
                  encode_context=_make_entities("alpha", "beta", "zeta"))
    _insert_chunk(conn, "c2", importance=0.5, stability=4.0,
                  encode_context=_make_entities("gamma", "delta", "theta"))
    # 1 个无重叠
    _insert_chunk(conn, "c3", importance=0.6, stability=5.0,
                  encode_context=_make_entities("foo", "bar", "baz"))
    # 1 个 importance 不足
    _insert_chunk(conn, "c4", importance=0.20, stability=5.0,
                  encode_context=session_ec)

    result = apply_contextual_reinstatement_consolidation(conn, "test")

    assert "cre_consolidated" in result, "CR9: result 应含 cre_consolidated key"
    assert "total_examined" in result, "CR9: result 应含 total_examined key"
    assert result["cre_consolidated"] >= 2, (
        f"CR9: 应有 >= 2 个 chunk 被巩固，got {result}"
    )
    assert result["total_examined"] >= 3, (
        f"CR9: total_examined 应 >= 3，got {result}"
    )


# ── CR10: 显式 session_accessed_ids 精确构建情境 ────────────────────────────────────

def test_cr10_explicit_session_ids(conn):
    """CR10: 传入 session_accessed_ids 时，只用这些 chunk 构建情境集合（精确模式）。"""
    # session chunk（有 auth-related entities，last_accessed 在 2h 外，不会自动进入 session 情境）
    session_ec = _make_entities("auth_token", "session_key", "oauth_provider", "access_scope")
    _insert_chunk(conn, "s_id_1", encode_context=session_ec,
                  importance=0.7, last_accessed_days_ago=2.0)  # 2天前，在 2h 窗口外
    _insert_chunk(conn, "s_id_2", encode_context=session_ec,
                  importance=0.7, last_accessed_days_ago=2.0)

    # 目标 chunk：与 session 有重叠
    target_ec = _make_entities("auth_token", "session_key", "other")
    _insert_chunk(conn, "target_10", importance=0.6, stability=5.0,
                  encode_context=target_ec)

    # 不相关 chunk（近期访问，会进入默认 2h session 情境）
    unrelated_ec = _make_entities("X", "Y", "Z", "W")
    for i in range(5):
        _insert_session_chunk(conn, f"unrel_{i}", encode_context=unrelated_ec)

    stab_before = _get_stability(conn, "target_10")

    # 使用显式 session_accessed_ids（强制用 2 天前的 auth-related session chunks 构建情境）
    result = apply_contextual_reinstatement_consolidation(
        conn, "test", session_accessed_ids=["s_id_1", "s_id_2"]
    )
    stab_after = _get_stability(conn, "target_10")

    assert stab_after > stab_before, (
        f"CR10: 显式 session_accessed_ids 精确构建情境后，目标 chunk 应获得加成，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["cre_consolidated"] >= 1, f"CR10: cre_consolidated 应 >= 1，got {result}"
