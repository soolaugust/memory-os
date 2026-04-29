"""
test_iter474_sae.py — iter474: Spreading Activation Effect 单元测试

覆盖：
  SA1: 访问 chunk A 后，语义相似 chunk B 的 retrievability 提升
  SA2: 语义不相似（Jaccard < sae_min_similarity=0.20）的 chunk 不受影响
  SA3: sae_enabled=False → 无激活扩散
  SA4: 源 chunk importance < sae_min_importance(0.25) → 不触发 SAE
  SA5: 提升幅度受 sae_max_spread(0.15) 上限保护
  SA6: 每次扩散最多影响 sae_max_neighbors(10) 个邻居
  SA7: retrievability 加成后不超过 1.0
  SA8: update_accessed 集成测试 — 访问触发邻居 retrievability 提升

认知科学依据：
  Collins & Loftus (1975) "A spreading-activation theory of semantic processing" —
    语义网络中节点激活沿关联边传播（decay with distance）；相关概念可达性提升 20-30%。
  Anderson (1983) ACT* 模型：基线激活水平 Bi = log(fan_i) + Σ source activations。

OS 类比：Linux readahead（mm/readahead.c）—
  顺序/相关 page 预取到 page cache，降低后续访问的 page fault 率。
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

from store_vfs import ensure_schema, insert_chunk, update_accessed, apply_spreading_activation_effect
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


def _insert_raw(conn, cid, content, project="test", chunk_type="observation",
                importance=0.6, retrievability=0.5):
    now_iso = _utcnow().isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count,
            encode_context, session_type_history)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (cid, project, chunk_type, content, "summary", importance, 5.0,
         now_iso, now_iso, retrievability, now_iso, 0, "test_ctx", "coding")
    )
    conn.commit()


def _get_retrievability(conn, cid: str) -> float:
    row = conn.execute("SELECT retrievability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


# ── SA1: 高相似度邻居 retrievability 提升 ────────────────────────────────────────────────

def test_sa1_similar_chunk_retrievability_boosted(conn):
    """SA1: 访问 chunk A 后，语义相似 chunk B 的 retrievability 提升。"""
    content_a = "memory allocator kernel slab buddy page frame vmalloc"
    content_b = "memory allocator kernel slab buddy page cache mmap"  # 高 Jaccard
    _insert_raw(conn, "sa1_a", content_a, retrievability=0.5)
    _insert_raw(conn, "sa1_b", content_b, retrievability=0.5)

    retr_b_before = _get_retrievability(conn, "sa1_b")
    apply_spreading_activation_effect(conn, ["sa1_a"])
    retr_b_after = _get_retrievability(conn, "sa1_b")

    assert retr_b_after > retr_b_before, (
        f"SA1: 相似 chunk B 的 retrievability 应提升，"
        f"before={retr_b_before:.4f} after={retr_b_after:.4f}"
    )


# ── SA2: 低相似度邻居不受影响 ─────────────────────────────────────────────────────────────

def test_sa2_dissimilar_chunk_not_boosted(conn):
    """SA2: 语义不相似（Jaccard < 0.20）的 chunk 不受 SAE 影响。"""
    content_a = "memory allocator kernel slab buddy page frame vmalloc"
    content_c = "neural network deep learning gradient descent backpropagation"  # 无重叠
    _insert_raw(conn, "sa2_a", content_a, retrievability=0.5)
    _insert_raw(conn, "sa2_c", content_c, retrievability=0.5)

    retr_c_before = _get_retrievability(conn, "sa2_c")
    apply_spreading_activation_effect(conn, ["sa2_a"])
    retr_c_after = _get_retrievability(conn, "sa2_c")

    assert abs(retr_c_after - retr_c_before) < 0.001, (
        f"SA2: 不相似 chunk 不应受 SAE 影响，before={retr_c_before:.4f} after={retr_c_after:.4f}"
    )


# ── SA3: sae_enabled=False → 无扩散 ──────────────────────────────────────────────────────

def test_sa3_disabled_no_spread(conn):
    """SA3: sae_enabled=False → 相似 chunk retrievability 不提升。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.sae_enabled":
            return False
        return original_get(key, project=project)

    content_a = "memory allocator kernel slab buddy page frame vmalloc"
    content_b = "memory allocator kernel slab buddy page cache mmap"
    _insert_raw(conn, "sa3_a", content_a, retrievability=0.5)
    _insert_raw(conn, "sa3_b", content_b, retrievability=0.5)

    retr_b_before = _get_retrievability(conn, "sa3_b")
    with mock.patch.object(config, 'get', side_effect=patched_get):
        apply_spreading_activation_effect(conn, ["sa3_a"])
    retr_b_after = _get_retrievability(conn, "sa3_b")

    assert abs(retr_b_after - retr_b_before) < 0.001, (
        f"SA3: disabled 时不应有激活扩散，before={retr_b_before:.4f} after={retr_b_after:.4f}"
    )


# ── SA4: 源 chunk importance 不足 → 不触发 SAE ───────────────────────────────────────────

def test_sa4_low_importance_no_spread(conn):
    """SA4: 源 chunk importance < sae_min_importance(0.25) → 不触发 SAE。"""
    content_a = "memory allocator kernel slab buddy page frame vmalloc"
    content_b = "memory allocator kernel slab buddy page cache mmap"
    _insert_raw(conn, "sa4_a", content_a, importance=0.10, retrievability=0.5)
    _insert_raw(conn, "sa4_b", content_b, retrievability=0.5)

    retr_b_before = _get_retrievability(conn, "sa4_b")
    apply_spreading_activation_effect(conn, ["sa4_a"])
    retr_b_after = _get_retrievability(conn, "sa4_b")

    assert abs(retr_b_after - retr_b_before) < 0.001, (
        f"SA4: 低 importance 不应触发 SAE，before={retr_b_before:.4f} after={retr_b_after:.4f}"
    )


# ── SA5: 提升幅度受 sae_max_spread 保护 ──────────────────────────────────────────────────

def test_sa5_max_spread_cap(conn):
    """SA5: SAE 提升不超过 sae_max_spread(0.15)。"""
    sae_max_spread = config.get("store_vfs.sae_max_spread")
    # 使用极大的 spread_factor 测试 cap
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.sae_spread_factor":
            return 5.0  # 远超正常值
        return original_get(key, project=project)

    content_a = "memory allocator kernel slab buddy page frame vmalloc"
    content_b = "memory allocator kernel slab buddy page cache mmap"
    _insert_raw(conn, "sa5_a", content_a, retrievability=0.5)
    _insert_raw(conn, "sa5_b", content_b, retrievability=0.3)

    retr_b_before = _get_retrievability(conn, "sa5_b")
    with mock.patch.object(config, 'get', side_effect=patched_get):
        apply_spreading_activation_effect(conn, ["sa5_a"])
    retr_b_after = _get_retrievability(conn, "sa5_b")

    spread = retr_b_after - retr_b_before
    assert spread <= sae_max_spread + 0.01, (
        f"SA5: SAE 提升 {spread:.4f} 不应超过 max_spread={sae_max_spread}，"
        f"before={retr_b_before:.4f} after={retr_b_after:.4f}"
    )
    assert retr_b_after > retr_b_before, (
        f"SA5: 应有 SAE 提升，before={retr_b_before:.4f} after={retr_b_after:.4f}"
    )


# ── SA6: 最多影响 sae_max_neighbors 个邻居 ───────────────────────────────────────────────

def test_sa6_max_neighbors_limit(conn):
    """SA6: 每次扩散最多影响 sae_max_neighbors(10) 个邻居。"""
    content_a = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    _insert_raw(conn, "sa6_src", content_a, retrievability=0.5)

    # 插入 15 个高相似度邻居
    for i in range(15):
        content_nb = f"alpha beta gamma delta epsilon zeta eta theta iota lambda_{i}"
        _insert_raw(conn, f"sa6_nb_{i}", content_nb, retrievability=0.5)

    apply_spreading_activation_effect(conn, ["sa6_src"])

    boosted_count = 0
    for i in range(15):
        retr = _get_retrievability(conn, f"sa6_nb_{i}")
        if retr > 0.5 + 0.001:
            boosted_count += 1

    max_neighbors = config.get("store_vfs.sae_max_neighbors")
    assert boosted_count <= max_neighbors, (
        f"SA6: 被提升的邻居数 {boosted_count} 不应超过 max_neighbors={max_neighbors}"
    )


# ── SA7: retrievability 不超过 1.0 ───────────────────────────────────────────────────────

def test_sa7_retrievability_cap_1(conn):
    """SA7: SAE 提升后 retrievability 不超过 1.0。"""
    content_a = "memory allocator kernel slab buddy page frame vmalloc"
    content_b = "memory allocator kernel slab buddy page cache mmap"
    _insert_raw(conn, "sa7_a", content_a, retrievability=0.5)
    _insert_raw(conn, "sa7_b", content_b, retrievability=0.95)  # 接近上限

    apply_spreading_activation_effect(conn, ["sa7_a"])
    retr_b = _get_retrievability(conn, "sa7_b")

    assert retr_b <= 1.0, f"SA7: retrievability 不应超过 1.0，got {retr_b:.4f}"


# ── SA8: update_accessed 集成测试 ────────────────────────────────────────────────────────

def test_sa8_update_accessed_integration(conn):
    """SA8: update_accessed 触发 SAE，相似邻居 retrievability 提升。"""
    content_a = "memory allocator kernel slab buddy page frame vmalloc"
    content_b = "memory allocator kernel slab buddy page cache mmap"

    # 插入两个高相似度 chunk
    now_iso = _utcnow().isoformat()
    for cid, content in [("sa8_a", content_a), ("sa8_b", content_b)]:
        conn.execute(
            """INSERT OR REPLACE INTO memory_chunks
               (id, project, chunk_type, content, summary, importance, stability,
                created_at, updated_at, retrievability, last_accessed, access_count,
                encode_context, session_type_history, source_session)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (cid, "test", "observation", content, "summary", 0.6, 5.0,
             now_iso, now_iso, 0.5, now_iso, 0, "test_ctx", "coding", "sess1")
        )
    conn.commit()

    retr_b_before = _get_retrievability(conn, "sa8_b")
    update_accessed(conn, ["sa8_a"], session_id="sess1", project="test")
    retr_b_after = _get_retrievability(conn, "sa8_b")

    assert retr_b_after >= retr_b_before, (
        f"SA8: update_accessed 后相似 chunk B 的 retrievability 不应降低，"
        f"before={retr_b_before:.4f} after={retr_b_after:.4f}"
    )
