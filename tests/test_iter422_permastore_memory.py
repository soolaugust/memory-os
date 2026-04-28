"""
test_iter422_permastore_memory.py — iter422: Permastore Memory 单元测试

覆盖：
  PM1: 非 permastore chunk → RI 可以降到 0.1 floor
  PM2: permastore chunk → RI 不能低于 stability × floor_factor
  PM3: permastore chunk → RIF 不能低于 stability × floor_factor
  PM4: permastore chunk → DF 不能低于 stability × floor_factor
  PM5: permastore_enabled=False → 所有 chunk 使用普通 0.1 floor
  PM6: age < min_age_days → 不是 permastore（即使 access_count 和 importance 足够高）
  PM7: access_count < min_access_count → 不是 permastore
  PM8: importance < min_importance → 不是 permastore
  PM9: 三个条件都满足 → permastore floor = stability × floor_factor
  PM10: permastore floor 本身 >= 0.1
  PM11: compute_permastore_floor 独立函数测试（各 threshold 边界）
  PM12: insert_chunk 触发 RI → permastore 旧 chunk 受到更高 floor 保护

认知科学依据：
  Bahrick (1979) Permastore — 充分强化+高重要性记忆抵抗遗忘。
  Conway et al. (1991) — 专业知识具有 permastore 特征（数十年保留）。

OS 类比：Linux mlock() — 重要页面锁定在 RAM，kswapd 无法驱逐。
"""
import sys
import sqlite3
import datetime
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import tmpfs  # noqa

from store_vfs import (
    ensure_schema,
    compute_permastore_floor,
    apply_retroactive_interference,
)
from store import insert_chunk
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


def _days_ago_iso(days: float) -> str:
    dt = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)
    return dt.isoformat()


def _insert_chunk_raw(conn, cid, encode_context="",
                       stability=2.0, project="test",
                       importance=0.6, created_at=None,
                       access_count=1):
    now = _now_iso()
    ca = created_at or now
    conn.execute(
        "INSERT OR REPLACE INTO memory_chunks "
        "(id, project, chunk_type, content, summary, importance, stability, "
        "created_at, updated_at, retrievability, encode_context, access_count) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0.9, ?, ?)",
        (cid, project, "decision", f"content {cid}", f"summary {cid}",
         importance, stability, ca, now, encode_context, access_count)
    )
    conn.commit()


def _get_stability(conn, cid: str) -> float:
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


# ── PM1: Non-permastore chunk → RI can reduce to 0.1 ─────────────────────────

def test_pm1_non_permastore_normal_floor(conn, monkeypatch):
    """非 permastore chunk → RI 可将 stability 降到接近 0.1。"""
    import unittest.mock as mock
    original_get = config.get
    def patched_get(key, project=None):
        if key == "store_vfs.ri_decay_factor":
            return 0.01  # Very aggressive: 1% remaining
        return original_get(key, project=project)

    # Young chunk, few accesses, low importance → not permastore
    _insert_chunk_raw(conn, "pm1_old", encode_context="auth,jwt,token",
                      stability=2.0, importance=0.5, access_count=1,
                      created_at=_days_ago_iso(1))  # only 1 day old
    _insert_chunk_raw(conn, "pm1_new", encode_context="auth,jwt,token,refresh",
                      stability=2.0, importance=0.7)

    with mock.patch.object(config, 'get', side_effect=patched_get):
        apply_retroactive_interference(conn, "pm1_new", "test", base_stability=2.0)
    conn.commit()

    stab_after = _get_stability(conn, "pm1_old")
    # With decay_factor=0.01, new_stab=0.02 → clamped to 0.1
    assert abs(stab_after - 0.1) < 0.02, \
        f"PM1: 非 permastore chunk 应降到 0.1 floor，got {stab_after:.4f}"


# ── PM2: Permastore chunk → RI protected by higher floor ─────────────────────

def test_pm2_permastore_ri_protected(conn, monkeypatch):
    """permastore chunk → RI 不能将 stability 降到低于 stability × floor_factor。"""
    import unittest.mock as mock
    original_get = config.get
    def patched_get(key, project=None):
        if key == "store_vfs.ri_decay_factor":
            return 0.01  # Very aggressive decay
        return original_get(key, project=project)

    # Permastore conditions: 40 days old, 15 accesses, importance=0.90
    _insert_chunk_raw(conn, "pm2_old", encode_context="auth,jwt,token",
                      stability=2.0, importance=0.90, access_count=15,
                      created_at=_days_ago_iso(40))
    _insert_chunk_raw(conn, "pm2_new", encode_context="auth,jwt,token,refresh",
                      stability=2.0, importance=0.7)

    with mock.patch.object(config, 'get', side_effect=patched_get):
        apply_retroactive_interference(conn, "pm2_new", "test", base_stability=2.0)
    conn.commit()

    stab_after = _get_stability(conn, "pm2_old")
    # floor = 2.0 × 0.80 = 1.6 (vs decay result 2.0 × 0.01 = 0.02)
    expected_floor = 2.0 * config.get("store_vfs.permastore_floor_factor")
    assert stab_after >= expected_floor - 0.01, \
        f"PM2: permastore chunk 不应低于 floor={expected_floor:.4f}，got {stab_after:.4f}"


# ── PM3: Permastore chunk → RIF protected ────────────────────────────────────

def test_pm3_permastore_rif_protected(conn, monkeypatch):
    """permastore chunk → RIF 不能低于 stability × floor_factor。"""
    import unittest.mock as mock
    from store_vfs import apply_retrieval_induced_forgetting

    original_get = config.get
    def patched_get(key, project=None):
        if key == "store_vfs.rif_decay_factor":
            return 0.01  # Very aggressive
        return original_get(key, project=project)

    # Permastore chunk
    _insert_chunk_raw(conn, "pm3_neighbor", encode_context="auth,jwt,token",
                      stability=3.0, importance=0.90, access_count=15,
                      created_at=_days_ago_iso(40))
    # Also insert the "retrieved" chunk so RIF can find pm3_neighbor as a neighbor
    _insert_chunk_raw(conn, "pm3_retrieved", encode_context="auth,jwt,token,session",
                      stability=2.0, importance=0.7)

    with mock.patch.object(config, 'get', side_effect=patched_get):
        apply_retrieval_induced_forgetting(
            conn, chunk_ids=["pm3_retrieved"],
            project="test",
        )
    conn.commit()

    stab_after = _get_stability(conn, "pm3_neighbor")
    # If inhibited: floor = 3.0 × 0.80 = 2.4 (vs normal 0.1 with decay=0.01)
    expected_floor = 3.0 * config.get("store_vfs.permastore_floor_factor")
    # stab_after could be 3.0 (not inhibited if no overlap) or >= floor if inhibited
    assert stab_after >= expected_floor - 0.01 or abs(stab_after - 3.0) < 0.001, \
        f"PM3: permastore chunk 不应低于 floor={expected_floor:.4f}，got {stab_after:.4f}"


# ── PM4: Permastore chunk → DF protected ─────────────────────────────────────

def test_pm4_permastore_df_protected(conn, monkeypatch):
    """permastore chunk → Directed Forgetting 不能低于 stability × floor_factor。"""
    import unittest.mock as mock
    from store_vfs import apply_directed_forgetting

    original_get = config.get
    def patched_get(key, project=None):
        if key == "store_vfs.df_penalty_cap":
            return 0.50  # Max penalty = 50% of stability
        return original_get(key, project=project)

    # Permastore conditions + deprecated content
    _insert_chunk_raw(conn, "pm4_chunk", encode_context="auth,jwt",
                      stability=2.0, importance=0.90, access_count=15,
                      created_at=_days_ago_iso(40))
    # Override content with deprecated marker
    conn.execute("UPDATE memory_chunks SET content=? WHERE id=?",
                 ("deprecated: this approach is obsolete", "pm4_chunk"))
    conn.commit()

    with mock.patch.object(config, 'get', side_effect=patched_get):
        apply_directed_forgetting(conn, "pm4_chunk")
    conn.commit()

    stab_after = _get_stability(conn, "pm4_chunk")
    # Penalty up to 0.50 → new_stability = max(floor, 2.0 - 0.50×2.0) = max(1.6, 1.0) = 1.6
    expected_floor = 2.0 * config.get("store_vfs.permastore_floor_factor")
    assert stab_after >= expected_floor - 0.01, \
        f"PM4: permastore chunk DF 不应低于 floor={expected_floor:.4f}，got {stab_after:.4f}"


# ── PM5: permastore_enabled=False → normal 0.1 floor ─────────────────────────

def test_pm5_disabled_normal_floor(conn, monkeypatch):
    """permastore_enabled=False → 所有 chunk 使用普通 0.1 floor。"""
    import unittest.mock as mock
    original_get = config.get
    def patched_get(key, project=None):
        if key == "store_vfs.permastore_enabled":
            return False
        if key == "scorer.ribot_enabled":
            return False  # iter431: 同时禁用 Ribot floor，隔离 permastore 测试
        if key == "store_vfs.ri_decay_factor":
            return 0.01
        if key == "store_vfs.permastore_min_importance":
            return 0.70  # lower threshold so importance=0.75 qualifies as permastore
        return original_get(key, project=project)

    # importance=0.75 < ri_protect_importance(0.85) → RI can target it
    # importance=0.75 >= permastore_min_importance(0.70) → would be permastore if enabled
    _insert_chunk_raw(conn, "pm5_old", encode_context="auth,jwt,token",
                      stability=2.0, importance=0.75, access_count=15,
                      created_at=_days_ago_iso(40))
    _insert_chunk_raw(conn, "pm5_new", encode_context="auth,jwt,token,refresh",
                      stability=2.0, importance=0.7)

    with mock.patch.object(config, 'get', side_effect=patched_get):
        apply_retroactive_interference(conn, "pm5_new", "test", base_stability=2.0)
    conn.commit()

    stab_after = _get_stability(conn, "pm5_old")
    # Permastore disabled → floor=0.1, so 2.0 × 0.01 = 0.02 → clamped to 0.1
    assert abs(stab_after - 0.1) < 0.02, \
        f"PM5: permastore 禁用 → 应降到 0.1 floor，got {stab_after:.4f}"


# ── PM6: age too young → not permastore ──────────────────────────────────────

def test_pm6_young_chunk_not_permastore(conn):
    """age < min_age_days → 不是 permastore，即使 access_count 和 importance 满足。"""
    # 5 days old (< min 30 days)
    floor = compute_permastore_floor(
        conn, "pm6_fake",  # doesn't exist, will return 0.1
        current_stability=5.0
    )
    # Non-existent chunk → returns 0.1
    assert floor == 0.1, f"PM6: chunk 不存在应返回 0.1，got {floor}"

    # Insert young chunk with high accesses and importance
    _insert_chunk_raw(conn, "pm6_young", encode_context="test",
                      stability=5.0, importance=0.90, access_count=20,
                      created_at=_days_ago_iso(5))  # only 5 days old
    floor = compute_permastore_floor(conn, "pm6_young", current_stability=5.0)
    assert floor == 0.1, f"PM6: 5天 chunk 不应是 permastore，floor={floor:.4f}"


# ── PM7: access_count too low → not permastore ───────────────────────────────

def test_pm7_low_access_not_permastore(conn):
    """access_count < min_access_count → 不是 permastore。"""
    # 40 days old, importance=0.90, but only 3 accesses (< min 10)
    _insert_chunk_raw(conn, "pm7_chunk", encode_context="test",
                      stability=5.0, importance=0.90, access_count=3,
                      created_at=_days_ago_iso(40))
    floor = compute_permastore_floor(conn, "pm7_chunk", current_stability=5.0)
    assert floor == 0.1, f"PM7: 访问少 chunk 不应是 permastore，floor={floor:.4f}"


# ── PM8: importance too low → not permastore ─────────────────────────────────

def test_pm8_low_importance_not_permastore(conn):
    """importance < min_importance → 不是 permastore。"""
    # 40 days old, 15 accesses, but importance=0.7 (< min 0.80)
    _insert_chunk_raw(conn, "pm8_chunk", encode_context="test",
                      stability=5.0, importance=0.70, access_count=15,
                      created_at=_days_ago_iso(40))
    floor = compute_permastore_floor(conn, "pm8_chunk", current_stability=5.0)
    assert floor == 0.1, f"PM8: 低 importance chunk 不应是 permastore，floor={floor:.4f}"


# ── PM9: All three conditions met → permastore floor ─────────────────────────

def test_pm9_all_conditions_met(conn):
    """三个条件都满足 → permastore floor = stability × floor_factor。"""
    # 40 days, 15 accesses, importance=0.90
    _insert_chunk_raw(conn, "pm9_chunk", encode_context="test",
                      stability=5.0, importance=0.90, access_count=15,
                      created_at=_days_ago_iso(40))
    floor = compute_permastore_floor(conn, "pm9_chunk", current_stability=5.0)
    expected = 5.0 * config.get("store_vfs.permastore_floor_factor")  # 5.0 × 0.80 = 4.0
    assert abs(floor - expected) < 0.001, \
        f"PM9: permastore floor 应为 {expected:.4f}，got {floor:.4f}"


# ── PM10: permastore floor >= 0.1 ────────────────────────────────────────────

def test_pm10_floor_never_below_01(conn):
    """permastore floor 本身 >= 0.1（即使 stability 很低）。"""
    # Low-stability permastore chunk
    _insert_chunk_raw(conn, "pm10_chunk", encode_context="test",
                      stability=0.05, importance=0.90, access_count=15,
                      created_at=_days_ago_iso(40))
    floor = compute_permastore_floor(conn, "pm10_chunk", current_stability=0.05)
    # 0.05 × 0.80 = 0.04 → max(0.1, 0.04) = 0.1
    assert floor >= 0.1, f"PM10: permastore floor 不应低于 0.1，got {floor:.4f}"


# ── PM11: compute_permastore_floor boundary conditions ───────────────────────

def test_pm11_boundary_conditions(conn):
    """compute_permastore_floor 各 threshold 边界验证。"""
    # Exactly at boundary (min_age=30 days)
    _insert_chunk_raw(conn, "pm11_exact", encode_context="test",
                      stability=2.0, importance=0.80, access_count=10,
                      created_at=_days_ago_iso(30))
    floor = compute_permastore_floor(conn, "pm11_exact", current_stability=2.0)
    expected = 2.0 * config.get("store_vfs.permastore_floor_factor")  # 2.0 × 0.80 = 1.6
    assert abs(floor - expected) < 0.01, \
        f"PM11: 恰好满足边界 → permastore floor={expected:.4f}，got {floor:.4f}"

    # Just below boundary (29.5 days)
    _insert_chunk_raw(conn, "pm11_below", encode_context="test",
                      stability=2.0, importance=0.80, access_count=10,
                      created_at=_days_ago_iso(29.5))
    floor2 = compute_permastore_floor(conn, "pm11_below", current_stability=2.0)
    assert floor2 == 0.1, \
        f"PM11: 低于 age 阈值 → 应为普通 floor=0.1，got {floor2:.4f}"


# ── PM12: insert_chunk RI → permastore old chunk protected ───────────────────

def test_pm12_insert_chunk_ri_protects_permastore(conn, monkeypatch):
    """insert_chunk 触发 RI → permastore 旧 chunk 受更高 floor 保护。"""
    import unittest.mock as mock
    original_get = config.get
    def patched_get(key, project=None):
        if key == "store_vfs.ri_decay_factor":
            return 0.01  # Very aggressive
        return original_get(key, project=project)

    # Permastore old chunk (40d, 15 acc, imp=0.90) with tokens matching new chunk
    _insert_chunk_raw(conn, "pm12_permastore", encode_context="auth,jwt,token",
                      stability=3.0, importance=0.90, access_count=15,
                      created_at=_days_ago_iso(40))
    # Non-permastore old chunk with same tokens (young, few accesses)
    _insert_chunk_raw(conn, "pm12_normal", encode_context="auth,jwt,token",
                      stability=3.0, importance=0.60, access_count=2,
                      created_at=_days_ago_iso(1))

    stab_permastore_before = _get_stability(conn, "pm12_permastore")
    stab_normal_before = _get_stability(conn, "pm12_normal")

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    chunk = {
        "id": "pm12_new",
        "created_at": now, "updated_at": now, "project": "test",
        "source_session": "s1", "chunk_type": "decision",
        "info_class": "semantic",
        "content": "New JWT auth approach with token refresh.",
        "summary": "new JWT auth",
        "tags": [], "importance": 0.7, "retrievability": 0.9,
        "last_accessed": now, "access_count": 1,
        "oom_adj": 0, "lru_gen": 0, "stability": 2.0,
        "raw_snippet": "", "encoding_context": {"tokens": ["auth", "jwt", "token", "refresh"]},
    }
    with mock.patch.object(config, 'get', side_effect=patched_get):
        insert_chunk(conn, chunk)
    conn.commit()

    stab_permastore_after = _get_stability(conn, "pm12_permastore")
    stab_normal_after = _get_stability(conn, "pm12_normal")

    expected_permastore_floor = stab_permastore_before * config.get("store_vfs.permastore_floor_factor")

    # Permastore chunk: floor = stability × 0.80
    assert stab_permastore_after >= expected_permastore_floor - 0.02, \
        f"PM12: permastore chunk 应受 floor 保护，floor={expected_permastore_floor:.4f} got={stab_permastore_after:.4f}"
    # Normal chunk: floor = 0.1 → may be near 0.1
    assert stab_normal_after <= stab_permastore_after + 0.1, \
        f"PM12: 正常 chunk stability 应 <= permastore chunk，normal={stab_normal_after:.4f} permastore={stab_permastore_after:.4f}"
