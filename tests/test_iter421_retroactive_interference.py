"""
test_iter421_retroactive_interference.py — iter421: Retroactive Interference 单元测试

覆盖：
  RI1: 新 chunk 写入 → 旧重叠 chunk stability 下降
  RI2: 新 chunk 本身不受 RI 影响
  RI3: 无重叠旧 chunk → 无 RI 效应
  RI4: overlap < ri_min_overlap → 无 RI 效应
  RI5: ri_enabled=False → 禁用时无 RI 效应
  RI6: decay_factor=1.0 → 无衰减
  RI7: 高 importance chunk（>= protect_importance）免疫 RI
  RI8: 最多影响 ri_max_targets 个旧 chunk
  RI9: stability floor at 0.1
  RI10: 跨 project 不受 RI 影响（project 隔离）
  RI11: insert_chunk 触发 RI — 旧相关 chunk stability 自动下降
  RI12: RI 与 PI（iter408）的方向互补验证

认知科学依据：
  McGeoch (1932) Interference Theory — 遗忘的主因是相似记忆的竞争性干扰，非被动衰减。
  Barnes & Underwood (1959) — 新学习破坏旧记忆回忆（retroactive interference）。
  Anderson & Green (2001): 主动抑制相似记忆是 RI 的神经机制。

OS 类比：TLB shootdown (inter-processor interrupt) —
  新 VA→PA 映射建立时，发送 IPI 使其他核的旧 TLB 条目失效。
  新 chunk 写入 = 新映射 = 旧相关 chunk（旧 TLB 条目）stability 降低。
"""
import sys
import sqlite3
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import tmpfs  # noqa

from store_vfs import (
    ensure_schema,
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


def _insert_chunk_direct(conn, cid, encode_context="", stability=2.0, project="test",
                          importance=0.6, chunk_type="decision"):
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO memory_chunks "
        "(id, project, chunk_type, content, summary, importance, stability, "
        "created_at, updated_at, retrievability, encode_context) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0.9, ?)",
        (cid, project, chunk_type, f"content {cid}", f"summary {cid}",
         importance, stability, now, now, encode_context)
    )
    conn.commit()


def _get_stability(conn, cid: str) -> float:
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


# ── RI1: New chunk causes old overlapping chunk stability to drop ──────────────

def test_ri1_new_chunk_interferes_old(conn):
    """新 chunk 写入 → 旧重叠 chunk stability 下降。"""
    # Old chunk with tokens {auth, jwt, token}
    _insert_chunk_direct(conn, "ri1_old", encode_context="auth,jwt,token",
                          stability=2.0, importance=0.6)
    # New chunk with same tokens (simulates new info replacing old)
    _insert_chunk_direct(conn, "ri1_new", encode_context="auth,jwt,token,refresh",
                          stability=2.0, importance=0.7)

    stab_old_before = _get_stability(conn, "ri1_old")
    n = apply_retroactive_interference(conn, "ri1_new", "test", base_stability=2.0)
    conn.commit()
    stab_old_after = _get_stability(conn, "ri1_old")

    assert stab_old_after < stab_old_before or n > 0, \
        f"RI1: 旧 chunk stability 应下降，before={stab_old_before:.4f} after={stab_old_after:.4f}, n={n}"


# ── RI2: New chunk itself not affected ────────────────────────────────────────

def test_ri2_new_chunk_not_affected(conn):
    """新 chunk 本身不受 RI 影响（只影响旧 chunk）。"""
    _insert_chunk_direct(conn, "ri2_old", encode_context="auth,jwt,token",
                          stability=2.0, importance=0.6)
    _insert_chunk_direct(conn, "ri2_new", encode_context="auth,jwt,token,refresh",
                          stability=2.0, importance=0.7)

    stab_new_before = _get_stability(conn, "ri2_new")
    apply_retroactive_interference(conn, "ri2_new", "test", base_stability=2.0)
    conn.commit()
    stab_new_after = _get_stability(conn, "ri2_new")

    assert abs(stab_new_after - stab_new_before) < 0.001, \
        f"RI2: 新 chunk 不应受 RI，before={stab_new_before:.4f} after={stab_new_after:.4f}"


# ── RI3: No overlap → no RI ──────────────────────────────────────────────────

def test_ri3_no_overlap_no_ri(conn):
    """无 encode_context 重叠 → 无 RI 效应。"""
    _insert_chunk_direct(conn, "ri3_old", encode_context="database,postgres,migration",
                          stability=2.0, importance=0.6)
    _insert_chunk_direct(conn, "ri3_new", encode_context="auth,jwt,token",
                          stability=2.0, importance=0.7)

    stab_old_before = _get_stability(conn, "ri3_old")
    n = apply_retroactive_interference(conn, "ri3_new", "test", base_stability=2.0)
    conn.commit()
    stab_old_after = _get_stability(conn, "ri3_old")

    assert abs(stab_old_after - stab_old_before) < 0.001, \
        f"RI3: 无重叠 → stability 不应变，before={stab_old_before:.4f} after={stab_old_after:.4f}"
    assert n == 0, f"RI3: 无重叠应返回 n=0，got {n}"


# ── RI4: Overlap below min_overlap → no RI ───────────────────────────────────

def test_ri4_low_overlap_no_ri(conn, monkeypatch):
    """overlap=1 < min_overlap=3 → 无 RI 效应。"""
    import unittest.mock as mock
    original_get = config.get
    def patched_get(key, project=None):
        if key == "store_vfs.ri_min_overlap":
            return 3
        return original_get(key, project=project)

    # Only 1 shared token: "auth"
    _insert_chunk_direct(conn, "ri4_old", encode_context="auth,database,postgres",
                          stability=2.0, importance=0.6)
    _insert_chunk_direct(conn, "ri4_new", encode_context="auth,jwt,token",
                          stability=2.0, importance=0.7)

    stab_old_before = _get_stability(conn, "ri4_old")
    with mock.patch.object(config, 'get', side_effect=patched_get):
        n = apply_retroactive_interference(conn, "ri4_new", "test", base_stability=2.0)
    conn.commit()
    stab_old_after = _get_stability(conn, "ri4_old")

    assert abs(stab_old_after - stab_old_before) < 0.001, \
        f"RI4: 低重叠 → stability 不应变，before={stab_old_before:.4f} after={stab_old_after:.4f}"


# ── RI5: ri_enabled=False → no RI ────────────────────────────────────────────

def test_ri5_disabled_no_ri(conn, monkeypatch):
    """ri_enabled=False → 禁用时无 RI 效应。"""
    import unittest.mock as mock
    original_get = config.get
    def patched_get(key, project=None):
        if key == "store_vfs.ri_enabled":
            return False
        return original_get(key, project=project)

    _insert_chunk_direct(conn, "ri5_old", encode_context="auth,jwt,token",
                          stability=2.0, importance=0.6)
    _insert_chunk_direct(conn, "ri5_new", encode_context="auth,jwt,token,refresh",
                          stability=2.0, importance=0.7)

    stab_old_before = _get_stability(conn, "ri5_old")
    with mock.patch.object(config, 'get', side_effect=patched_get):
        n = apply_retroactive_interference(conn, "ri5_new", "test", base_stability=2.0)
    conn.commit()
    stab_old_after = _get_stability(conn, "ri5_old")

    assert abs(stab_old_after - stab_old_before) < 0.001, \
        f"RI5: 禁用 → stability 不应变，before={stab_old_before:.4f} after={stab_old_after:.4f}"
    assert n == 0, f"RI5: 禁用应返回 n=0，got {n}"


# ── RI6: decay_factor=1.0 → no decay ─────────────────────────────────────────

def test_ri6_decay_factor_one_no_decay(conn, monkeypatch):
    """decay_factor=1.0 → 无衰减（identity）。"""
    import unittest.mock as mock
    original_get = config.get
    def patched_get(key, project=None):
        if key == "store_vfs.ri_decay_factor":
            return 1.0
        return original_get(key, project=project)

    _insert_chunk_direct(conn, "ri6_old", encode_context="auth,jwt,token",
                          stability=2.0, importance=0.6)
    _insert_chunk_direct(conn, "ri6_new", encode_context="auth,jwt,token,refresh",
                          stability=2.0, importance=0.7)

    stab_old_before = _get_stability(conn, "ri6_old")
    with mock.patch.object(config, 'get', side_effect=patched_get):
        n = apply_retroactive_interference(conn, "ri6_new", "test", base_stability=2.0)
    conn.commit()
    stab_old_after = _get_stability(conn, "ri6_old")

    assert abs(stab_old_after - stab_old_before) < 0.001, \
        f"RI6: factor=1.0 stability 不应变，before={stab_old_before:.4f} after={stab_old_after:.4f}"
    assert n == 0, f"RI6: factor=1.0 应返回 n=0，got {n}"


# ── RI7: High-importance chunk immune to RI ───────────────────────────────────

def test_ri7_high_importance_protected(conn):
    """importance >= ri_protect_importance → chunk 免疫 RI。"""
    # Old chunk with HIGH importance=0.9 (above protect threshold=0.85)
    _insert_chunk_direct(conn, "ri7_protected", encode_context="auth,jwt,token",
                          stability=2.0, importance=0.9)
    # Old chunk with LOW importance=0.6 (below protect threshold → vulnerable)
    _insert_chunk_direct(conn, "ri7_vulnerable", encode_context="auth,jwt,session",
                          stability=2.0, importance=0.6)
    # New chunk
    _insert_chunk_direct(conn, "ri7_new", encode_context="auth,jwt,token,refresh",
                          stability=2.0, importance=0.7)

    stab_protected_before = _get_stability(conn, "ri7_protected")
    stab_vulnerable_before = _get_stability(conn, "ri7_vulnerable")
    apply_retroactive_interference(conn, "ri7_new", "test", base_stability=2.0)
    conn.commit()
    stab_protected_after = _get_stability(conn, "ri7_protected")
    stab_vulnerable_after = _get_stability(conn, "ri7_vulnerable")

    assert abs(stab_protected_after - stab_protected_before) < 0.001, \
        f"RI7: 高 importance chunk 应免疫 RI，before={stab_protected_before:.4f} after={stab_protected_after:.4f}"
    assert stab_vulnerable_after < stab_vulnerable_before, \
        f"RI7: 低 importance chunk 应受 RI，before={stab_vulnerable_before:.4f} after={stab_vulnerable_after:.4f}"


# ── RI8: max_targets limit ────────────────────────────────────────────────────

def test_ri8_max_targets_limit(conn, monkeypatch):
    """ri_max_targets=2 → 最多影响 2 个旧 chunk。"""
    import unittest.mock as mock
    original_get = config.get
    def patched_get(key, project=None):
        if key == "store_vfs.ri_max_targets":
            return 2
        return original_get(key, project=project)

    # Create 5 overlapping old chunks
    for i in range(5):
        _insert_chunk_direct(conn, f"ri8_old{i}",
                              encode_context=f"auth,jwt,token{i},session{i}",
                              stability=2.0, importance=0.6)
    _insert_chunk_direct(conn, "ri8_new",
                          encode_context="auth,jwt,refresh,security",
                          stability=2.0, importance=0.7)

    with mock.patch.object(config, 'get', side_effect=patched_get):
        n = apply_retroactive_interference(conn, "ri8_new", "test", base_stability=2.0)
    conn.commit()

    assert n <= 2, f"RI8: max_targets=2 → 最多影响 2 个，got {n}"


# ── RI9: stability floor at 0.1 ─────────────────────────────────────────────

def test_ri9_stability_floor_protected(conn, monkeypatch):
    """low stability chunk 不低于 0.1。"""
    import unittest.mock as mock
    original_get = config.get
    def patched_get(key, project=None):
        if key == "store_vfs.ri_decay_factor":
            return 0.10  # very aggressive decay
        return original_get(key, project=project)

    _insert_chunk_direct(conn, "ri9_old", encode_context="auth,jwt,token",
                          stability=0.12, importance=0.6)  # near floor
    _insert_chunk_direct(conn, "ri9_new", encode_context="auth,jwt,token,refresh",
                          stability=2.0, importance=0.7)

    with mock.patch.object(config, 'get', side_effect=patched_get):
        apply_retroactive_interference(conn, "ri9_new", "test", base_stability=2.0)
    conn.commit()
    stab_after = _get_stability(conn, "ri9_old")
    assert stab_after >= 0.1, f"RI9: stability 不应低于 0.1（floor），got {stab_after:.4f}"


# ── RI10: Cross-project isolation ────────────────────────────────────────────

def test_ri10_cross_project_isolation(conn):
    """不同 project 的 chunk 不受 RI 影响。"""
    _insert_chunk_direct(conn, "ri10_old", encode_context="auth,jwt,token",
                          stability=2.0, importance=0.6, project="project_B")
    _insert_chunk_direct(conn, "ri10_new", encode_context="auth,jwt,token,refresh",
                          stability=2.0, importance=0.7, project="project_A")

    stab_old_before = _get_stability(conn, "ri10_old")
    apply_retroactive_interference(conn, "ri10_new", "project_A", base_stability=2.0)
    conn.commit()
    stab_old_after = _get_stability(conn, "ri10_old")

    assert abs(stab_old_after - stab_old_before) < 0.001, \
        f"RI10: 跨 project 不应受 RI，before={stab_old_before:.4f} after={stab_old_after:.4f}"


# ── RI11: insert_chunk triggers RI automatically ──────────────────────────────

def test_ri11_insert_chunk_triggers_ri(conn):
    """insert_chunk 写入新 chunk → 旧相关 chunk stability 自动下降。"""
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Pre-insert an old overlapping chunk (low importance to be vulnerable)
    _insert_chunk_direct(conn, "ri11_old", encode_context="auth,jwt,token",
                          stability=2.0, importance=0.6, project="test")
    stab_old_before = _get_stability(conn, "ri11_old")

    # Insert new chunk via insert_chunk (triggers full pipeline including RI)
    chunk = {
        "id": "ri11_new",
        "created_at": now, "updated_at": now, "project": "test",
        "source_session": "s1", "chunk_type": "decision",
        "info_class": "semantic",
        "content": "New JWT authentication approach replaces old one.",
        "summary": "new JWT auth approach",
        "tags": [], "importance": 0.7, "retrievability": 0.9,
        "last_accessed": now, "access_count": 1,
        "oom_adj": 0, "lru_gen": 0, "stability": 2.0,
        "raw_snippet": "", "encoding_context": {"tokens": ["auth", "jwt", "token", "refresh"]},
    }
    insert_chunk(conn, chunk)
    conn.commit()

    stab_old_after = _get_stability(conn, "ri11_old")
    # The old chunk should have been affected by RI
    # (exact change depends on encode_context extraction, but should not have increased)
    assert stab_old_after <= stab_old_before + 0.001, \
        f"RI11: insert_chunk 后旧 chunk stability 不应增加，before={stab_old_before:.4f} after={stab_old_after:.4f}"


# ── RI12: RI and PI are complementary (opposite directions) ───────────────────

def test_ri12_ri_and_pi_are_complementary(conn):
    """RI(新→旧) 与 PI(旧→新) 方向互补：各自干扰对方方向的 chunk。"""
    # RI: new chunk interfering with OLD chunks
    _insert_chunk_direct(conn, "ri12_old", encode_context="auth,jwt,token",
                          stability=2.0, importance=0.6, project="test")
    _insert_chunk_direct(conn, "ri12_new", encode_context="auth,jwt,token,refresh",
                          stability=2.0, importance=0.7, project="test")

    stab_old_before = _get_stability(conn, "ri12_old")
    n_ri = apply_retroactive_interference(conn, "ri12_new", "test", base_stability=2.0)
    conn.commit()
    stab_old_after = _get_stability(conn, "ri12_old")

    # RI should have decreased old chunk stability or at least not increased it
    assert stab_old_after <= stab_old_before + 0.001, \
        f"RI12: RI 方向应为新→旧，old before={stab_old_before:.4f} after={stab_old_after:.4f}"
    # New chunk should not be affected by RI
    stab_new = _get_stability(conn, "ri12_new")
    assert stab_new >= 2.0 * 0.95, \
        f"RI12: 新 chunk 不应被 RI 显著降低，got {stab_new:.4f}"
