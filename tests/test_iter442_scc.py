"""
test_iter442_scc.py — iter442: Schema-Consistent Consolidation 单元测试

覆盖：
  SC1: 近期 chunk 与图式核有足够 entity 重叠 → stability 增加（快速图式同化）
  SC2: 无图式核（项目无高 access + 高 importance chunk）→ 无 consolidation
  SC3: entity 重叠不足（< scc_min_overlap=3）→ 无 consolidation
  SC4: scc_enabled=False → 无任何 consolidation
  SC5: 旧 chunk（created_at > scc_window_days 之前）不受 SCC（只对近期新知识）
  SC6: 图式核自身（access_count>=5）不被重复巩固（仍会被巩固但因是近期 chunk 可以）
  SC7: consolidation 后 stability 不超过 365.0
  SC8: scc_schema_min_importance 可配置（降低阈值时更多 chunk 成为图式核）
  SC9: 返回计数正确（schema_consolidated, total_examined）

认知科学依据：
  Bartlett (1932) Schema Theory —
    与已有图式高度一致的新信息被更快整合，睡眠巩固期获得额外强化。
  Tse et al. (2007) Science "Schemas and memory consolidation" —
    已有丰富图式后，新知识 1 天内完成系统巩固（而非无图式时需 3 天）。
  McClelland et al. (1995) Complementary Learning Systems —
    图式越强，新皮层整合越快（快速系统巩固路径）。

OS 类比：Linux readahead pattern detection —
  顺序访问模式（与已有 I/O 模式一致）的 page，readahead 窗口扩大 → 更快完成 I/O。
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
    apply_schema_consistent_consolidation,
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
    return ", ".join(names)


def _insert_chunk(conn, cid, project="test", stability=5.0, importance=0.6,
                  access_count=0, encode_context="",
                  created_days_ago=1.0, last_accessed_days_ago=1.0):
    """Insert chunk with controlled created_at (for SCC window filtering)."""
    created_at = _ago_iso(days=created_days_ago)
    last_accessed = _ago_iso(days=last_accessed_days_ago)
    now = _now_iso()
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count,
            encode_context)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0.9, ?, ?, ?)""",
        (cid, project, "decision", f"content {cid}", f"summary {cid}",
         importance, stability, created_at, now, last_accessed, access_count, encode_context)
    )
    conn.commit()


def _get_stability(conn, cid: str) -> float:
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


# ── SC1: 近期 chunk 与图式核有足够重叠 → stability 增加 ────────────────────────────

def test_sc1_schema_consistent_chunk_boosted(conn):
    """SC1: 近期写入的 chunk 与图式核有 >= scc_min_overlap(3) entity 重叠 → stability 增加。"""
    scc_schema_min_access = config.get("store_vfs.scc_schema_min_access")  # 5
    scc_schema_min_imp = config.get("store_vfs.scc_schema_min_importance")  # 0.80
    scc_window = config.get("store_vfs.scc_window_days")  # 7.0

    shared_entities = ["auth_token", "session_key", "oauth_provider", "access_scope"]
    schema_ec = _make_entities(*shared_entities)

    # 图式核：高访问 + 高重要性
    _insert_chunk(conn, "schema_core", importance=scc_schema_min_imp,
                  access_count=scc_schema_min_access,
                  encode_context=schema_ec, created_days_ago=30.0)

    # 近期新 chunk：与图式核有 4 entity 重叠，创建时间在 window 内
    new_ec = _make_entities(*shared_entities)
    _insert_chunk(conn, "new_chunk", importance=0.5, access_count=0,
                  stability=5.0, encode_context=new_ec,
                  created_days_ago=2.0)  # 2天前 < scc_window_days=7

    stab_before = _get_stability(conn, "new_chunk")
    result = apply_schema_consistent_consolidation(conn, "test")
    stab_after = _get_stability(conn, "new_chunk")

    assert stab_after > stab_before, (
        f"SC1: 图式一致性新 chunk 应获得 stability 加成，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["schema_consolidated"] >= 1, f"SC1: schema_consolidated 应 >= 1，got {result}"


# ── SC2: 无图式核 → 无 consolidation ─────────────────────────────────────────────

def test_sc2_no_schema_core_no_consolidation(conn):
    """SC2: 项目中无高 access + 高 importance 的图式核 → 无 SCC consolidation。"""
    shared_ec = _make_entities("auth_token", "session_key", "oauth_provider")

    # 只有低访问的 chunk（非图式核）
    _insert_chunk(conn, "weak_core", importance=0.80, access_count=2,  # < scc_schema_min_access=5
                  encode_context=shared_ec, created_days_ago=30.0)
    _insert_chunk(conn, "new_chunk_2", importance=0.5, access_count=0,
                  stability=5.0, encode_context=shared_ec, created_days_ago=2.0)

    stab_before = _get_stability(conn, "new_chunk_2")
    result = apply_schema_consistent_consolidation(conn, "test")
    stab_after = _get_stability(conn, "new_chunk_2")

    assert abs(stab_after - stab_before) < 0.001, (
        f"SC2: 无图式核时不应 consolidate，before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── SC3: entity 重叠不足 → 无 consolidation ──────────────────────────────────────

def test_sc3_insufficient_overlap_no_consolidation(conn):
    """SC3: entity 重叠数 < scc_min_overlap(3) → 无 SCC consolidation。"""
    scc_schema_min_access = config.get("store_vfs.scc_schema_min_access")
    scc_schema_min_imp = config.get("store_vfs.scc_schema_min_importance")

    # 图式核：4 entities
    schema_ec = _make_entities("entity_A", "entity_B", "entity_C", "entity_D")
    _insert_chunk(conn, "schema_core_3", importance=scc_schema_min_imp,
                  access_count=scc_schema_min_access,
                  encode_context=schema_ec, created_days_ago=30.0)

    # 近期 chunk：只有 1 entity 重叠（低于 scc_min_overlap=3）
    cand_ec = _make_entities("entity_A", "entity_X", "entity_Y")
    _insert_chunk(conn, "low_overlap_3", importance=0.5, access_count=0,
                  stability=5.0, encode_context=cand_ec, created_days_ago=2.0)

    stab_before = _get_stability(conn, "low_overlap_3")
    apply_schema_consistent_consolidation(conn, "test")
    stab_after = _get_stability(conn, "low_overlap_3")

    assert abs(stab_after - stab_before) < 0.001, (
        f"SC3: entity 重叠不足时不应 consolidate，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── SC4: scc_enabled=False → 无 consolidation ────────────────────────────────────

def test_sc4_disabled_no_consolidation(conn):
    """SC4: store_vfs.scc_enabled=False → 无任何 SCC consolidation。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.scc_enabled":
            return False
        return original_get(key, project=project)

    scc_min_acc = config.get("store_vfs.scc_schema_min_access")
    scc_min_imp = config.get("store_vfs.scc_schema_min_importance")
    shared_ec = _make_entities("auth", "token", "session", "oauth")

    _insert_chunk(conn, "schema_core_4", importance=scc_min_imp, access_count=scc_min_acc,
                  encode_context=shared_ec, created_days_ago=30.0)
    _insert_chunk(conn, "new_chunk_4", importance=0.5, access_count=0,
                  stability=5.0, encode_context=shared_ec, created_days_ago=2.0)

    stab_before = _get_stability(conn, "new_chunk_4")
    with mock.patch.object(config, 'get', side_effect=patched_get):
        result = apply_schema_consistent_consolidation(conn, "test")
    stab_after = _get_stability(conn, "new_chunk_4")

    assert abs(stab_after - stab_before) < 0.001, (
        f"SC4: disabled 时不应 consolidate，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["schema_consolidated"] == 0, f"SC4: schema_consolidated 应为 0，got {result}"


# ── SC5: 旧 chunk（超出时间窗口）不受 SCC ────────────────────────────────────────

def test_sc5_old_chunk_not_consolidated(conn):
    """SC5: created_at > scc_window_days 之前的 chunk 不受 SCC（只对近期新知识）。"""
    scc_min_acc = config.get("store_vfs.scc_schema_min_access")
    scc_min_imp = config.get("store_vfs.scc_schema_min_importance")
    scc_window = config.get("store_vfs.scc_window_days")  # 7.0
    shared_ec = _make_entities("auth", "token", "session", "oauth")

    _insert_chunk(conn, "schema_core_5", importance=scc_min_imp, access_count=scc_min_acc,
                  encode_context=shared_ec, created_days_ago=30.0)
    # 旧 chunk：超出 scc_window_days=7 的时间窗口
    _insert_chunk(conn, "old_chunk_5", importance=0.5, access_count=0,
                  stability=5.0, encode_context=shared_ec,
                  created_days_ago=scc_window + 1.0)  # 8 天前 > 7 天窗口

    stab_before = _get_stability(conn, "old_chunk_5")
    apply_schema_consistent_consolidation(conn, "test")
    stab_after = _get_stability(conn, "old_chunk_5")

    assert abs(stab_after - stab_before) < 0.001, (
        f"SC5: 超出时间窗口的旧 chunk 不应受 SCC，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── SC6: 图式核自身（高访问）如果也是近期写入，不受特殊保护但可以被 SCC ────────────

def test_sc6_schema_core_not_double_boosted(conn):
    """SC6: 图式核 chunk 如果是旧 chunk（created_days_ago > window），不在近期候选中。"""
    scc_min_acc = config.get("store_vfs.scc_schema_min_access")
    scc_min_imp = config.get("store_vfs.scc_schema_min_importance")
    scc_window = config.get("store_vfs.scc_window_days")  # 7.0
    shared_ec = _make_entities("auth", "token", "session", "oauth")

    # 图式核：旧 chunk（created 30 天前）
    _insert_chunk(conn, "schema_core_6", importance=scc_min_imp, access_count=scc_min_acc,
                  stability=10.0, encode_context=shared_ec,
                  created_days_ago=30.0)  # 远超 window

    stab_before = _get_stability(conn, "schema_core_6")
    apply_schema_consistent_consolidation(conn, "test")
    stab_after = _get_stability(conn, "schema_core_6")

    # 图式核是旧 chunk → 不在 window 内 → 不被 SCC
    assert abs(stab_after - stab_before) < 0.001, (
        f"SC6: 旧图式核 chunk 不应受 SCC（不在近期窗口内），"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── SC7: consolidation 后 stability 不超过 365.0 ─────────────────────────────────

def test_sc7_stability_cap_365(conn):
    """SC7: Schema-Consistent Consolidation 后 stability 不超过 365.0。"""
    scc_min_acc = config.get("store_vfs.scc_schema_min_access")
    scc_min_imp = config.get("store_vfs.scc_schema_min_importance")
    shared_ec = _make_entities("auth", "token", "session", "oauth")

    _insert_chunk(conn, "schema_core_7", importance=scc_min_imp, access_count=scc_min_acc,
                  encode_context=shared_ec, created_days_ago=30.0)
    _insert_chunk(conn, "near_cap_7", importance=0.5, access_count=0,
                  stability=364.9, encode_context=shared_ec, created_days_ago=2.0)

    apply_schema_consistent_consolidation(conn, "test")
    stab_after = _get_stability(conn, "near_cap_7")

    assert stab_after <= 365.0, f"SC7: stability 不应超过 365.0，got {stab_after}"


# ── SC8: scc_schema_min_importance 可配置 ────────────────────────────────────────

def test_sc8_configurable_schema_importance(conn):
    """SC8: 降低 scc_schema_min_importance 时，更低 importance 的 chunk 也可作为图式核。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.scc_schema_min_importance":
            return 0.60  # 降低到 0.60
        return original_get(key, project=project)

    scc_min_acc = config.get("store_vfs.scc_schema_min_access")
    shared_ec = _make_entities("auth", "token", "session", "oauth")

    # importance=0.70（低于默认 0.80，但高于 patched 0.60）
    _insert_chunk(conn, "medium_core_8", importance=0.70, access_count=scc_min_acc,
                  encode_context=shared_ec, created_days_ago=30.0)
    _insert_chunk(conn, "new_chunk_8", importance=0.4, access_count=0,
                  stability=5.0, encode_context=shared_ec, created_days_ago=2.0)

    stab_before = _get_stability(conn, "new_chunk_8")

    # 默认阈值（0.80）：importance=0.70 不是图式核 → 无 consolidation
    result_default = apply_schema_consistent_consolidation(conn, "test")
    stab_default = _get_stability(conn, "new_chunk_8")

    assert abs(stab_default - stab_before) < 0.001, (
        f"SC8: 默认阈值下 importance=0.70 不应作为图式核"
    )

    # 降低阈值（0.60）：importance=0.70 成为图式核 → consolidation 发生
    with mock.patch.object(config, 'get', side_effect=patched_get):
        result_patched = apply_schema_consistent_consolidation(conn, "test")
    stab_patched = _get_stability(conn, "new_chunk_8")

    assert stab_patched > stab_before, (
        f"SC8: 降低阈值后 importance=0.70 应作为图式核，触发 SCC，"
        f"before={stab_before:.4f} patched={stab_patched:.4f}"
    )


# ── SC9: 返回计数正确 ─────────────────────────────────────────────────────────────

def test_sc9_return_counts_correct(conn):
    """SC9: result dict 中 schema_consolidated 和 total_examined 计数正确。"""
    scc_min_acc = config.get("store_vfs.scc_schema_min_access")
    scc_min_imp = config.get("store_vfs.scc_schema_min_importance")
    shared_ec = _make_entities("alpha", "beta", "gamma", "delta")
    unrelated_ec = _make_entities("foo", "bar", "baz")

    # 1 图式核（旧 chunk）
    _insert_chunk(conn, "schema_core_9", importance=scc_min_imp, access_count=scc_min_acc,
                  encode_context=shared_ec, created_days_ago=30.0)
    # 2 近期与图式核重叠的 chunk → 被 consolidate
    _insert_chunk(conn, "new_9a", importance=0.5, access_count=0,
                  stability=5.0, encode_context=shared_ec, created_days_ago=2.0)
    _insert_chunk(conn, "new_9b", importance=0.4, access_count=0,
                  stability=4.0, encode_context=shared_ec, created_days_ago=3.0)
    # 1 近期无重叠的 chunk → 不被 consolidate
    _insert_chunk(conn, "new_9c", importance=0.5, access_count=0,
                  stability=5.0, encode_context=unrelated_ec, created_days_ago=2.0)

    result = apply_schema_consistent_consolidation(conn, "test")

    assert "schema_consolidated" in result, "SC9: result 应含 schema_consolidated key"
    assert "total_examined" in result, "SC9: result 应含 total_examined key"
    assert result["schema_consolidated"] >= 2, (
        f"SC9: 应有 >= 2 个 chunk 被 consolidate，got {result}"
    )
    assert result["total_examined"] >= 3, (
        f"SC9: total_examined 应 >= 3（近期 chunk 总数），got {result}"
    )
