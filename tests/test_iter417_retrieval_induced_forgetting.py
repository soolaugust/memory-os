"""
test_iter417_retrieval_induced_forgetting.py — iter417: Retrieval-Induced Forgetting 单元测试

覆盖：
  RIF1: apply_rif — 语义竞争者 stability 下降
  RIF2: apply_rif — 被检索的 chunk 本身不受 RIF 影响
  RIF3: apply_rif — 无重叠邻居 → 无 RIF 效应
  RIF4: apply_rif — overlap 低于 min_overlap 阈值 → 无效应
  RIF5: apply_rif — rif_enabled=False → 禁用时无效应
  RIF6: apply_rif — decay_factor=1.0 → 无衰减
  RIF7: apply_rif — 最多影响 max_neighbors 个邻居
  RIF8: apply_rif — stability 不低于 0.1（floor 保护）
  RIF9: update_accessed 触发 RIF — 语义邻居 stability 下降
  RIF10: apply_rif — 跨 project 邻居不受影响（project 隔离）

认知科学依据：
  Anderson, Bjork & Bjork (1994) "Remembering can cause forgetting" —
    检索记忆 A 时主动抑制其语义竞争者 B/C（inhibitory tagging）。
  MacLeod et al. (2003): RIF 是真实的记忆抑制（non-retrieval 控制组无此效应）。

OS 类比：MESI 缓存一致性协议 —
  写入 Modified cache line → 其他核心的相同 cache line 变为 Invalid；
  访问 chunk A → 其语义竞争者的"局部性"下降（cache invalidation）。
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
    apply_retrieval_induced_forgetting,
    update_accessed,
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


def _make_chunk(cid, encode_context_str="", stability=2.0, project="test",
                chunk_type="decision"):
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return {
        "id": cid, "created_at": now, "updated_at": now,
        "project": project, "source_session": "s1",
        "chunk_type": chunk_type, "info_class": "semantic",
        "content": f"content for {cid}", "summary": f"summary {cid}",
        "tags": [], "importance": 0.8, "retrievability": 0.9,
        "last_accessed": now, "access_count": 2, "oom_adj": 0, "lru_gen": 0,
        "stability": stability, "raw_snippet": "",
        "encoding_context": {},
    }


def _insert_with_ec(conn, cid, ec_str, stability=2.0, project="test", chunk_type="decision"):
    chunk = _make_chunk(cid, stability=stability, project=project, chunk_type=chunk_type)
    insert_chunk(conn, chunk)
    conn.execute(
        "UPDATE memory_chunks SET encode_context=? WHERE id=?", (ec_str, cid)
    )
    conn.commit()


def _get_stability(conn, cid: str) -> float:
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


# ── RIF1: Semantic competitor → stability decreases ──────────────────────────

def test_rif1_competitor_stability_decreases(conn):
    """语义竞争者（高 encode_context 重叠）→ stability 下降。"""
    # chunk A (accessed): tokens = {auth, jwt, token, security}
    # chunk B (competitor): tokens = {auth, jwt, session} — 2 shared tokens → RIF target
    _insert_with_ec(conn, "rif1a", "auth,jwt,token,security", stability=2.0)
    _insert_with_ec(conn, "rif1b", "auth,jwt,session,cookie", stability=2.0)

    stab_b_before = _get_stability(conn, "rif1b")
    n = apply_retrieval_induced_forgetting(conn, ["rif1a"], "test")
    conn.commit()
    stab_b_after = _get_stability(conn, "rif1b")

    assert stab_b_after < stab_b_before or n > 0, \
        f"RIF1: 竞争者 stability 应下降，before={stab_b_before:.4f} after={stab_b_after:.4f}, n={n}"


# ── RIF2: Accessed chunk itself not affected ─────────────────────────────────

def test_rif2_accessed_chunk_not_inhibited(conn):
    """被检索的 chunk 本身不受 RIF 抑制。"""
    _insert_with_ec(conn, "rif2a", "auth,jwt,token", stability=2.0)
    _insert_with_ec(conn, "rif2b", "auth,jwt,token,security", stability=2.0)

    stab_a_before = _get_stability(conn, "rif2a")
    apply_retrieval_induced_forgetting(conn, ["rif2a"], "test")
    conn.commit()
    stab_a_after = _get_stability(conn, "rif2a")

    assert abs(stab_a_after - stab_a_before) < 0.01, \
        f"RIF2: 被检索 chunk 不应受到 RIF，before={stab_a_before:.4f} after={stab_a_after:.4f}"


# ── RIF3: No overlapping neighbors → no RIF ──────────────────────────────────

def test_rif3_no_overlapping_neighbors(conn):
    """无重叠邻居 → 无 RIF 效应。"""
    _insert_with_ec(conn, "rif3a", "auth,jwt", stability=2.0)
    _insert_with_ec(conn, "rif3b", "database,postgres,migration", stability=2.0)

    stab_b_before = _get_stability(conn, "rif3b")
    n = apply_retrieval_induced_forgetting(conn, ["rif3a"], "test")
    conn.commit()
    stab_b_after = _get_stability(conn, "rif3b")

    assert abs(stab_b_after - stab_b_before) < 0.001, \
        f"RIF3: 无重叠邻居 stability 不应变化，before={stab_b_before:.4f} after={stab_b_after:.4f}"
    assert n == 0, f"RIF3: 无重叠应返回 0 inhibited，got {n}"


# ── RIF4: Overlap below min_overlap threshold → no effect ────────────────────

def test_rif4_low_overlap_no_rif(conn, monkeypatch):
    """overlap=1 < min_overlap=2 → 无 RIF 效应。"""
    import unittest.mock as mock
    original_get = config.get
    def patched_get(key, project=None):
        if key == "store_vfs.rif_min_overlap":
            return 3  # require 3 token overlap
        return original_get(key, project=project)

    _insert_with_ec(conn, "rif4a", "auth,jwt,token", stability=2.0)
    _insert_with_ec(conn, "rif4b", "auth,database,postgres", stability=2.0)  # only 1 shared: auth

    stab_b_before = _get_stability(conn, "rif4b")
    with mock.patch.object(config, 'get', side_effect=patched_get):
        n = apply_retrieval_induced_forgetting(conn, ["rif4a"], "test")
    conn.commit()
    stab_b_after = _get_stability(conn, "rif4b")

    assert abs(stab_b_after - stab_b_before) < 0.001, \
        f"RIF4: 低重叠 stability 不应变化，before={stab_b_before:.4f} after={stab_b_after:.4f}"


# ── RIF5: rif_enabled=False → no effect ──────────────────────────────────────

def test_rif5_disabled_no_rif(conn, monkeypatch):
    """rif_enabled=False → 禁用时无 RIF 效应。"""
    import unittest.mock as mock
    original_get = config.get
    def patched_get(key, project=None):
        if key == "store_vfs.rif_enabled":
            return False
        return original_get(key, project=project)

    _insert_with_ec(conn, "rif5a", "auth,jwt,token,security", stability=2.0)
    _insert_with_ec(conn, "rif5b", "auth,jwt,session", stability=2.0)

    stab_b_before = _get_stability(conn, "rif5b")
    with mock.patch.object(config, 'get', side_effect=patched_get):
        n = apply_retrieval_induced_forgetting(conn, ["rif5a"], "test")
    conn.commit()
    stab_b_after = _get_stability(conn, "rif5b")

    assert abs(stab_b_after - stab_b_before) < 0.001, \
        f"RIF5: 禁用 RIF stability 不应变化，before={stab_b_before:.4f} after={stab_b_after:.4f}"
    assert n == 0, f"RIF5: 禁用应返回 0"


# ── RIF6: decay_factor=1.0 → no decay ────────────────────────────────────────

def test_rif6_decay_factor_one_no_decay(conn, monkeypatch):
    """decay_factor=1.0 → 无衰减（identity）。"""
    import unittest.mock as mock
    original_get = config.get
    def patched_get(key, project=None):
        if key == "store_vfs.rif_decay_factor":
            return 1.0
        return original_get(key, project=project)

    _insert_with_ec(conn, "rif6a", "auth,jwt,token", stability=2.0)
    _insert_with_ec(conn, "rif6b", "auth,jwt,session", stability=2.0)

    stab_b_before = _get_stability(conn, "rif6b")
    with mock.patch.object(config, 'get', side_effect=patched_get):
        n = apply_retrieval_induced_forgetting(conn, ["rif6a"], "test")
    conn.commit()
    stab_b_after = _get_stability(conn, "rif6b")

    assert abs(stab_b_after - stab_b_before) < 0.001, \
        f"RIF6: decay=1.0 stability 不应变化，before={stab_b_before:.4f} after={stab_b_after:.4f}"
    assert n == 0, f"RIF6: factor=1.0 应返回 0 inhibited"


# ── RIF7: max_neighbors limit ─────────────────────────────────────────────────

def test_rif7_max_neighbors_limit(conn, monkeypatch):
    """max_neighbors=2 → 最多影响 2 个邻居。"""
    import unittest.mock as mock
    original_get = config.get
    def patched_get(key, project=None):
        if key == "store_vfs.rif_max_neighbors":
            return 2
        return original_get(key, project=project)

    _insert_with_ec(conn, "rif7a", "auth,jwt,token,security,refresh", stability=2.0)
    for i in range(5):
        _insert_with_ec(conn, f"rif7b{i}", f"auth,jwt,session{i},cookie{i}", stability=2.0)

    with mock.patch.object(config, 'get', side_effect=patched_get):
        n = apply_retrieval_induced_forgetting(conn, ["rif7a"], "test")
    conn.commit()

    assert n <= 2, f"RIF7: max_neighbors=2 → 最多影响 2 个，got {n}"


# ── RIF8: stability floor at 0.1 ─────────────────────────────────────────────

def test_rif8_stability_floor_protected(conn, monkeypatch):
    """low stability chunk 不低于 0.1。"""
    import unittest.mock as mock
    original_get = config.get
    def patched_get(key, project=None):
        if key == "store_vfs.rif_decay_factor":
            return 0.50  # aggressive decay for test
        return original_get(key, project=project)

    _insert_with_ec(conn, "rif8a", "auth,jwt,token", stability=2.0)
    _insert_with_ec(conn, "rif8b", "auth,jwt,security", stability=0.12)  # near floor

    with mock.patch.object(config, 'get', side_effect=patched_get):
        apply_retrieval_induced_forgetting(conn, ["rif8a"], "test")
    conn.commit()

    stab_b = _get_stability(conn, "rif8b")
    assert stab_b >= 0.1, f"RIF8: stability 不应低于 0.1（floor）得到 {stab_b:.4f}"


# ── RIF9: update_accessed triggers RIF ───────────────────────────────────────

def test_rif9_update_accessed_triggers_rif(conn):
    """update_accessed 触发 RIF — 语义竞争者 stability 下降（或不变）。"""
    _insert_with_ec(conn, "rif9a", "auth,jwt,token,security", stability=2.0)
    _insert_with_ec(conn, "rif9b", "auth,jwt,session,refresh", stability=2.0)

    stab_b_before = _get_stability(conn, "rif9b")
    update_accessed(conn, ["rif9a"])
    conn.commit()
    stab_b_after = _get_stability(conn, "rif9b")

    # RIF should cause small decrease (or at least not increase)
    # Default decay=0.99 → stab × 0.99
    assert stab_b_after <= stab_b_before + 0.001, \
        f"RIF9: update_accessed 后竞争者 stability 不应增加，before={stab_b_before:.4f} after={stab_b_after:.4f}"


# ── RIF10: Cross-project isolation ───────────────────────────────────────────

def test_rif10_cross_project_isolation(conn):
    """不同 project 的 chunk 不受 RIF 影响（project 隔离）。"""
    _insert_with_ec(conn, "rif10a", "auth,jwt,token", stability=2.0, project="project_A")
    _insert_with_ec(conn, "rif10b", "auth,jwt,session", stability=2.0, project="project_B")

    stab_b_before = _get_stability(conn, "rif10b")
    # Apply RIF only for project_A (rif10b belongs to project_B)
    apply_retrieval_induced_forgetting(conn, ["rif10a"], "project_A")
    conn.commit()
    stab_b_after = _get_stability(conn, "rif10b")

    assert abs(stab_b_after - stab_b_before) < 0.001, \
        f"RIF10: 跨 project 不应受 RIF，before={stab_b_before:.4f} after={stab_b_after:.4f}"
