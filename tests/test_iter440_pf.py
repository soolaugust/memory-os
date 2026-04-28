"""
test_iter440_pf.py — iter440: Proactive Facilitation 单元测试

覆盖：
  PF1: 与高 importance 强邻居有足够 entity 重叠的 stale chunk → stability 增加
  PF2: 无强邻居（项目无高 importance + 高访问 chunk）→ 无 facilitation
  PF3: entity 重叠数不足（< pf_min_overlap=3）→ 无 facilitation
  PF4: pf_enabled=False → 无任何 facilitation
  PF5: access_count >= 2 的候选 chunk 不受干预（不在 stale 扫描范围内）
  PF6: 锚点 chunk 自身（importance≥0.75 + access≥2）不被重复 facilitate（不是候选）
  PF7: facilitate 后 stability 不超过 365.0
  PF8: pf_anchor_min_importance 可配置（降低阈值时更多 chunk 被锚定）
  PF9: 返回计数正确（facilitated, total_examined）

认知科学依据：
  Ausubel (1963) Proactive Facilitation / advance organizer —
  已有稳固 schema 为新知识提供认知锚点，降低新知识的遗忘速率。
  与 PI（iter408）互补：PI=旧弱记忆干扰新知识（负迁移）；PF=旧强记忆锚定新知识（正迁移）。

OS 类比：Linux page cache 引用计数（refcount）—
  被多个 inode 共享引用的 page 有高 refcount，kswapd 优先保留。
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
    apply_proactive_facilitation,
)
import config


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
    """生成包含指定 entity 名称的 encode_context 字符串。"""
    return ", ".join(names)


def _insert_chunk(conn, cid, project="test", stability=5.0, importance=0.6,
                  access_count=0, encode_context="", last_accessed_days_ago=40.0):
    now = _now_iso()
    last_accessed = _ago_iso(days=last_accessed_days_ago)
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count,
            encode_context)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0.9, ?, ?, ?)""",
        (cid, project, "decision", f"content {cid}", f"summary {cid}",
         importance, stability, now, now, last_accessed, access_count, encode_context)
    )
    conn.commit()


def _get_stability(conn, cid: str) -> float:
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


# ── PF1: 与高 importance 强邻居有足够 entity 重叠 → stability 增加 ──────────────

def test_pf1_anchored_chunk_facilitated(conn):
    """PF1: 候选 chunk 与高 importance 强邻居有 >= pf_min_overlap(3) entity 重叠 → stability 增加。"""
    min_overlap = config.get("store_vfs.pf_min_overlap")  # 3
    anchor_min_imp = config.get("store_vfs.pf_anchor_min_importance")  # 0.75
    anchor_min_acc = config.get("store_vfs.pf_anchor_min_access")  # 2

    # 强邻居（anchor）：高 importance + 高 access_count + 共享 entity
    shared_entities = ["auth_token", "session_key", "oauth_provider", "access_scope"]
    anchor_ec = _make_entities(*shared_entities)
    _insert_chunk(conn, "anchor", importance=anchor_min_imp, access_count=anchor_min_acc,
                  encode_context=anchor_ec, last_accessed_days_ago=1.0)  # 近期访问（非 stale）

    # 候选 chunk：stale + 与 anchor 有 4 entity 重叠
    candidate_ec = _make_entities(*shared_entities)  # 完全重叠
    _insert_chunk(conn, "candidate", importance=0.5, access_count=0,
                  stability=5.0, encode_context=candidate_ec,
                  last_accessed_days_ago=40.0)  # stale

    stab_before = _get_stability(conn, "candidate")
    result = apply_proactive_facilitation(conn, "test", stale_days=30)
    stab_after = _get_stability(conn, "candidate")

    assert stab_after > stab_before, (
        f"PF1: 被锚定的候选 chunk 应获得 stability 修复，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["facilitated"] >= 1, f"PF1: facilitated 应 >= 1，got {result}"


# ── PF2: 无强邻居 → 无 facilitation ─────────────────────────────────────────────

def test_pf2_no_anchor_no_facilitation(conn):
    """PF2: 项目中无高 importance + 高访问的锚点 chunk → 无 facilitation。"""
    # 只有低 importance 的 chunk
    candidate_ec = _make_entities("auth_token", "session_key", "oauth_provider")
    _insert_chunk(conn, "no_anchor_cand", importance=0.5, access_count=0,
                  stability=5.0, encode_context=candidate_ec, last_accessed_days_ago=40.0)
    # "anchor" 但 importance 太低
    _insert_chunk(conn, "weak_anchor", importance=0.40, access_count=5,
                  encode_context=candidate_ec, last_accessed_days_ago=1.0)

    stab_before = _get_stability(conn, "no_anchor_cand")
    result = apply_proactive_facilitation(conn, "test", stale_days=30)
    stab_after = _get_stability(conn, "no_anchor_cand")

    assert abs(stab_after - stab_before) < 0.001, (
        f"PF2: 无强邻居时不应有 facilitation，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── PF3: entity 重叠不足 → 无 facilitation ────────────────────────────────────────

def test_pf3_insufficient_overlap_no_facilitation(conn):
    """PF3: entity 重叠数 < pf_min_overlap(3) → 无 facilitation。"""
    anchor_min_imp = config.get("store_vfs.pf_anchor_min_importance")  # 0.75
    anchor_min_acc = config.get("store_vfs.pf_anchor_min_access")  # 2

    # 强邻居
    anchor_ec = _make_entities("entity_A", "entity_B", "entity_C", "entity_D")
    _insert_chunk(conn, "strong_anchor", importance=anchor_min_imp, access_count=anchor_min_acc,
                  encode_context=anchor_ec, last_accessed_days_ago=1.0)

    # 候选：只有 1 entity 重叠（低于 pf_min_overlap=3）
    candidate_ec = _make_entities("entity_A", "entity_X", "entity_Y")  # only A overlaps
    _insert_chunk(conn, "low_overlap_cand", importance=0.5, access_count=0,
                  stability=5.0, encode_context=candidate_ec, last_accessed_days_ago=40.0)

    stab_before = _get_stability(conn, "low_overlap_cand")
    apply_proactive_facilitation(conn, "test", stale_days=30)
    stab_after = _get_stability(conn, "low_overlap_cand")

    assert abs(stab_after - stab_before) < 0.001, (
        f"PF3: entity 重叠不足时不应 facilitate，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── PF4: pf_enabled=False → 无 facilitation ──────────────────────────────────────

def test_pf4_disabled_no_facilitation(conn):
    """PF4: store_vfs.pf_enabled=False → 无任何 facilitation。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.pf_enabled":
            return False
        return original_get(key, project=project)

    shared_entities = ["auth", "token", "session", "oauth"]
    anchor_ec = _make_entities(*shared_entities)
    anchor_min_imp = config.get("store_vfs.pf_anchor_min_importance")
    anchor_min_acc = config.get("store_vfs.pf_anchor_min_access")

    _insert_chunk(conn, "anchor_4", importance=anchor_min_imp, access_count=anchor_min_acc,
                  encode_context=anchor_ec, last_accessed_days_ago=1.0)
    _insert_chunk(conn, "cand_4", importance=0.5, access_count=0,
                  stability=5.0, encode_context=anchor_ec, last_accessed_days_ago=40.0)

    stab_before = _get_stability(conn, "cand_4")
    with mock.patch.object(config, 'get', side_effect=patched_get):
        result = apply_proactive_facilitation(conn, "test", stale_days=30)
    stab_after = _get_stability(conn, "cand_4")

    assert abs(stab_after - stab_before) < 0.001, (
        f"PF4: disabled 时不应 facilitate，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["facilitated"] == 0, f"PF4: facilitated 应为 0，got {result}"


# ── PF5: access_count >= 2 的 chunk 不在候选范围 ─────────────────────────────────

def test_pf5_active_chunk_not_candidate(conn):
    """PF5: access_count >= 2 的 chunk 不在 stale 扫描范围，不受 PF 干预。"""
    anchor_min_imp = config.get("store_vfs.pf_anchor_min_importance")
    anchor_min_acc = config.get("store_vfs.pf_anchor_min_access")
    shared_ec = _make_entities("auth", "token", "session", "oauth")

    _insert_chunk(conn, "anchor_5", importance=anchor_min_imp, access_count=anchor_min_acc,
                  encode_context=shared_ec, last_accessed_days_ago=1.0)
    # 候选但 access_count >= 2（活跃，不在 stale 扫描范围）
    _insert_chunk(conn, "active_5", importance=0.5, access_count=3,
                  stability=5.0, encode_context=shared_ec, last_accessed_days_ago=40.0)

    stab_before = _get_stability(conn, "active_5")
    apply_proactive_facilitation(conn, "test", stale_days=30)
    stab_after = _get_stability(conn, "active_5")

    assert abs(stab_after - stab_before) < 0.001, (
        f"PF5: 活跃 chunk（access_count >= 2）不应受 PF 干预，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── PF6: 锚点 chunk 自身（高 imp + 高 access）不被重复 facilitate ─────────────────

def test_pf6_anchor_itself_not_facilitated(conn):
    """PF6: 锚点 chunk（access_count >= 2）不在候选范围，自身不受 PF 修复。"""
    anchor_min_imp = config.get("store_vfs.pf_anchor_min_importance")
    anchor_min_acc = config.get("store_vfs.pf_anchor_min_access")
    shared_ec = _make_entities("auth", "token", "session", "oauth")

    # 锚点：高 imp + 高 access + stale（模拟长期不访问但仍是强邻居）
    _insert_chunk(conn, "anchor_6", importance=anchor_min_imp, access_count=anchor_min_acc,
                  stability=5.0, encode_context=shared_ec, last_accessed_days_ago=40.0)

    stab_before = _get_stability(conn, "anchor_6")
    apply_proactive_facilitation(conn, "test", stale_days=30)
    stab_after = _get_stability(conn, "anchor_6")

    # 锚点的 access_count=2 → 不在 access_count < 2 候选范围 → 不被 facilitate
    assert abs(stab_after - stab_before) < 0.001, (
        f"PF6: 锚点 chunk（access_count >= 2）不应被 facilitate，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── PF7: facilitate 后 stability 不超过 365.0 ────────────────────────────────────

def test_pf7_stability_cap_365(conn):
    """PF7: Proactive Facilitation 后 stability 不超过 365.0。"""
    anchor_min_imp = config.get("store_vfs.pf_anchor_min_importance")
    anchor_min_acc = config.get("store_vfs.pf_anchor_min_access")
    shared_ec = _make_entities("auth", "token", "session", "oauth")

    _insert_chunk(conn, "anchor_7", importance=anchor_min_imp, access_count=anchor_min_acc,
                  encode_context=shared_ec, last_accessed_days_ago=1.0)
    _insert_chunk(conn, "near_cap_7", importance=0.5, access_count=0,
                  stability=364.9, encode_context=shared_ec, last_accessed_days_ago=40.0)

    apply_proactive_facilitation(conn, "test", stale_days=30)
    stab_after = _get_stability(conn, "near_cap_7")

    assert stab_after <= 365.0, f"PF7: stability 不应超过 365.0，got {stab_after}"


# ── PF8: pf_anchor_min_importance 可配置 ─────────────────────────────────────────

def test_pf8_configurable_anchor_importance(conn):
    """PF8: 降低 pf_anchor_min_importance 时，更低 importance 的 chunk 也可作为锚点。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.pf_anchor_min_importance":
            return 0.50  # 降低到 0.50
        return original_get(key, project=project)

    shared_ec = _make_entities("auth", "token", "session", "oauth")
    anchor_min_acc = config.get("store_vfs.pf_anchor_min_access")

    # 原 importance=0.55（低于默认 0.75，但高于 patched 0.50）
    _insert_chunk(conn, "weak_anchor_8", importance=0.55, access_count=anchor_min_acc,
                  encode_context=shared_ec, last_accessed_days_ago=1.0)
    _insert_chunk(conn, "cand_8", importance=0.4, access_count=0,
                  stability=5.0, encode_context=shared_ec, last_accessed_days_ago=40.0)

    stab_before = _get_stability(conn, "cand_8")

    # 默认阈值（0.75）：weak_anchor_8 不是锚点 → 无 facilitation
    result_default = apply_proactive_facilitation(conn, "test", stale_days=30)
    stab_default = _get_stability(conn, "cand_8")

    assert abs(stab_default - stab_before) < 0.001, (
        f"PF8: 默认阈值下 importance=0.55 不应作为锚点"
    )

    # 降低阈值（0.50）：weak_anchor_8 成为锚点 → facilitation 发生
    with mock.patch.object(config, 'get', side_effect=patched_get):
        result_patched = apply_proactive_facilitation(conn, "test", stale_days=30)
    stab_patched = _get_stability(conn, "cand_8")

    assert stab_patched > stab_before, (
        f"PF8: 降低阈值后 importance=0.55 应作为锚点，触发 facilitation，"
        f"before={stab_before:.4f} patched={stab_patched:.4f}"
    )


# ── PF9: 返回计数正确 ─────────────────────────────────────────────────────────────

def test_pf9_return_counts_correct(conn):
    """PF9: result dict 中 facilitated 和 total_examined 计数正确。"""
    anchor_min_imp = config.get("store_vfs.pf_anchor_min_importance")  # 0.75
    anchor_min_acc = config.get("store_vfs.pf_anchor_min_access")  # 2
    shared_ec = _make_entities("alpha", "beta", "gamma", "delta")
    unrelated_ec = _make_entities("foo", "bar", "baz")

    # 1 强邻居
    _insert_chunk(conn, "anchor_9", importance=anchor_min_imp, access_count=anchor_min_acc,
                  encode_context=shared_ec, last_accessed_days_ago=1.0)
    # 2 锚定候选（重叠足够）
    _insert_chunk(conn, "cand_9a", importance=0.5, access_count=0,
                  stability=5.0, encode_context=shared_ec, last_accessed_days_ago=40.0)
    _insert_chunk(conn, "cand_9b", importance=0.4, access_count=0,
                  stability=4.0, encode_context=shared_ec, last_accessed_days_ago=40.0)
    # 1 非锚定候选（无重叠）
    _insert_chunk(conn, "cand_9c", importance=0.5, access_count=0,
                  stability=5.0, encode_context=unrelated_ec, last_accessed_days_ago=40.0)

    result = apply_proactive_facilitation(conn, "test", stale_days=30)

    assert "facilitated" in result, "PF9: result 应含 facilitated key"
    assert "total_examined" in result, "PF9: result 应含 total_examined key"
    assert result["facilitated"] >= 2, f"PF9: 应有 >= 2 个 chunk 被 facilitate，got {result}"
    assert result["total_examined"] >= 3, f"PF9: total_examined 应 >= 3，got {result}"
