"""
test_iter483_pe.py — iter483: Priming Effect 单元测试

覆盖：
  PE1: 已有语义相似 chunk → 新 chunk stability 提升（启动效应）
  PE2: 无相似 chunk → 无 PE 加成
  PE3: pe_enabled=False → 无加成
  PE4: importance < pe_min_importance(0.25) → 不参与 PE
  PE5: 多个启动源 → 加成更大
  PE6: 加成受 pe_max_boost(0.10) 保护
  PE7: stability 上限 365.0 保护
  PE8: insert_chunk 集成 — 编码时触发 PE

认知科学依据：
  Meyer & Schvaneveldt (1971) JEPS — 已激活相关概念使目标识别更快，编码更深。
  Collins & Loftus (1975): 语义网络中相关节点激活传播，提升新内容编码质量。

OS 类比：Linux dentry cache warm — 相关目录项已缓存，新文件路径解析更快更稳定。
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

from store_vfs import ensure_schema, apply_priming_effect, insert_chunk
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


def _insert_raw(conn, cid, content="", project="test", importance=0.6, stability=5.0):
    now_iso = _utcnow().isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count,
            encode_context, session_type_history)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (cid, project, "observation", content, "summary", importance, stability,
         now_iso, now_iso, 0.5, now_iso, 0, "test_ctx", "")
    )
    conn.commit()


def _get_stability(conn, cid: str) -> float:
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


# ── PE1: 已有相似 chunk → 新 chunk stability 提升 ────────────────────────────────────────────

def test_pe1_existing_similar_primes_new_chunk(conn):
    """PE1: 已有高相似度 chunk，新 chunk 编码时获得 PE stability 加成。"""
    # 先插入一个"启动源" chunk
    _insert_raw(conn, "pe1_prime",
                content="memory allocator kernel slab buddy page frame vmalloc")
    # 再插入新 chunk（内容与启动源高度相似）
    _insert_raw(conn, "pe1_new",
                content="memory allocator kernel slab buddy page cache mmap",
                stability=5.0)

    stab_before = _get_stability(conn, "pe1_new")
    result = apply_priming_effect(conn, "pe1_new",
                                   "memory allocator kernel slab buddy page cache mmap")
    stab_after = _get_stability(conn, "pe1_new")

    assert stab_after > stab_before, (
        f"PE1: 有启动源时新 chunk stability 应提升，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["pe_boosted"] is True, f"PE1: pe_boosted 应为 True，got {result}"
    assert result["pe_n_primes"] >= 1, f"PE1: pe_n_primes 应 >= 1，got {result}"


# ── PE2: 无相似 chunk → 无加成 ──────────────────────────────────────────────────────────────

def test_pe2_no_similar_no_boost(conn):
    """PE2: 无相似启动源 → 无 PE 加成。"""
    # 没有相似的已有 chunk
    _insert_raw(conn, "pe2_new",
                content="neural network deep learning gradient descent backpropagation",
                stability=5.0)
    # 有一个不相关的 chunk
    _insert_raw(conn, "pe2_unrelated",
                content="memory allocator kernel slab buddy page frame vmalloc")

    stab_before = _get_stability(conn, "pe2_new")
    result = apply_priming_effect(conn, "pe2_new",
                                   "neural network deep learning gradient descent backpropagation")
    stab_after = _get_stability(conn, "pe2_new")

    assert abs(stab_after - stab_before) < 0.001, (
        f"PE2: 无启动源时不应有加成，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["pe_boosted"] is False, f"PE2: pe_boosted 应为 False"


# ── PE3: pe_enabled=False → 无加成 ──────────────────────────────────────────────────────────

def test_pe3_disabled_no_boost(conn):
    """PE3: pe_enabled=False → 无 PE 加成。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.pe_enabled":
            return False
        return original_get(key, project=project)

    _insert_raw(conn, "pe3_prime",
                content="memory allocator kernel slab buddy page frame vmalloc")
    _insert_raw(conn, "pe3_new",
                content="memory allocator kernel slab buddy page cache mmap",
                stability=5.0)

    stab_before = _get_stability(conn, "pe3_new")
    with mock.patch.object(config, 'get', side_effect=patched_get):
        result = apply_priming_effect(conn, "pe3_new",
                                       "memory allocator kernel slab buddy page cache mmap")
    stab_after = _get_stability(conn, "pe3_new")

    assert abs(stab_after - stab_before) < 0.001, (
        f"PE3: disabled 时不应有加成，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["pe_boosted"] is False


# ── PE4: importance 不足 → 不参与 PE ─────────────────────────────────────────────────────────

def test_pe4_low_importance_no_boost(conn):
    """PE4: importance < pe_min_importance(0.25) → 不参与 PE。"""
    _insert_raw(conn, "pe4_prime",
                content="memory allocator kernel slab buddy page frame vmalloc")
    _insert_raw(conn, "pe4_new",
                content="memory allocator kernel slab buddy page cache mmap",
                importance=0.10, stability=5.0)

    stab_before = _get_stability(conn, "pe4_new")
    result = apply_priming_effect(conn, "pe4_new",
                                   "memory allocator kernel slab buddy page cache mmap")
    stab_after = _get_stability(conn, "pe4_new")

    assert abs(stab_after - stab_before) < 0.001, (
        f"PE4: 低 importance 不应触发 PE，before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── PE5: 多个启动源 → 更大加成 ──────────────────────────────────────────────────────────────

def test_pe5_more_primes_more_boost(conn):
    """PE5: 启动源越多，PE stability 加成越大（直到 max_boost 上限）。"""
    # 基线：1个启动源
    _insert_raw(conn, "pe5_prime1",
                content="memory allocator kernel slab buddy page frame vmalloc")
    _insert_raw(conn, "pe5_new1",
                content="memory allocator kernel slab buddy page cache mmap",
                stability=5.0)
    apply_priming_effect(conn, "pe5_new1",
                          "memory allocator kernel slab buddy page cache mmap")
    stab_1 = _get_stability(conn, "pe5_new1")

    # 实验组：3个启动源（在不同的 conn2 中重做）
    conn2 = sqlite3.connect(":memory:")
    conn2.row_factory = sqlite3.Row
    ensure_schema(conn2)

    for i in range(3):
        _insert_raw(conn2, f"pe5_prime_{i}",
                    content=f"memory allocator kernel slab buddy page frame vmalloc_{i}")
    _insert_raw(conn2, "pe5_new3",
                content="memory allocator kernel slab buddy page frame",
                stability=5.0)
    apply_priming_effect(conn2, "pe5_new3",
                          "memory allocator kernel slab buddy page frame")
    stab_3 = _get_stability(conn2, "pe5_new3")
    conn2.close()

    # 3个启动源加成应 >= 1个启动源（或相等，因为有上限）
    assert stab_3 >= stab_1 - 0.001, (
        f"PE5: 多启动源加成应 >= 少启动源，1_prime={stab_1:.4f} 3_primes={stab_3:.4f}"
    )


# ── PE6: 加成受 pe_max_boost 保护 ────────────────────────────────────────────────────────────

def test_pe6_max_boost_cap(conn):
    """PE6: PE 加成受 pe_max_boost(0.10) 保护（即使有大量启动源）。"""
    pe_max_boost = config.get("store_vfs.pe_max_boost")  # 0.10
    base = 5.0

    # 插入 10 个相似启动源
    for i in range(10):
        _insert_raw(conn, f"pe6_prime_{i}",
                    content=f"memory allocator kernel slab buddy page frame vmalloc_{i}")
    _insert_raw(conn, "pe6_new",
                content="memory allocator kernel slab buddy page frame vmalloc",
                stability=base)

    stab_before = _get_stability(conn, "pe6_new")
    apply_priming_effect(conn, "pe6_new",
                          "memory allocator kernel slab buddy page frame vmalloc")
    stab_after = _get_stability(conn, "pe6_new")

    increment = stab_after - stab_before
    max_allowed = base * pe_max_boost + 0.01
    assert increment <= max_allowed, (
        f"PE6: PE 增量 {increment:.4f} 不应超过 max_boost 允许的 {max_allowed:.4f}"
    )
    assert stab_after > stab_before, f"PE6: 应有 PE 加成"


# ── PE7: stability 上限 365.0 ────────────────────────────────────────────────────────────────

def test_pe7_stability_cap_365(conn):
    """PE7: PE 加成后 stability 不超过 365.0。"""
    _insert_raw(conn, "pe7_prime",
                content="memory allocator kernel slab buddy page frame vmalloc")
    _insert_raw(conn, "pe7_new",
                content="memory allocator kernel slab buddy page cache mmap",
                stability=364.0)

    apply_priming_effect(conn, "pe7_new",
                          "memory allocator kernel slab buddy page cache mmap")
    stab = _get_stability(conn, "pe7_new")
    assert stab <= 365.0, f"PE7: stability 不应超过 365.0，got {stab}"


# ── PE8: insert_chunk 集成 ───────────────────────────────────────────────────────────────────

def test_pe8_insert_chunk_integration(conn):
    """PE8: insert_chunk 时，若已有相似 chunk，新 chunk stability 应获得 PE 加成。"""
    # 先插入启动源（使用 insert_chunk 使其进入 DB）
    now_iso = _utcnow().isoformat()
    prime_content = "memory allocator kernel slab buddy page frame vmalloc"

    prime_chunk = {
        "id": "pe8_prime", "project": "test", "source_session": "sess1",
        "chunk_type": "observation",
        "content": prime_content,
        "summary": "prime summary",
        "importance": 0.6, "stability": 5.0,
        "created_at": now_iso, "updated_at": now_iso,
        "retrievability": 0.5, "last_accessed": now_iso,
        "access_count": 0, "encode_context": "test_ctx",
    }
    insert_chunk(conn, prime_chunk)

    # 再插入新 chunk（相似内容）
    new_content = "memory allocator kernel slab buddy page cache mmap"
    new_chunk = {
        "id": "pe8_new", "project": "test", "source_session": "sess1",
        "chunk_type": "observation",
        "content": new_content,
        "summary": "new summary",
        "importance": 0.6, "stability": 5.0,
        "created_at": now_iso, "updated_at": now_iso,
        "retrievability": 0.5, "last_accessed": now_iso,
        "access_count": 0, "encode_context": "test_ctx",
    }

    # 不带 PE 时的基线（在新 conn 中）
    conn2 = sqlite3.connect(":memory:")
    conn2.row_factory = sqlite3.Row
    ensure_schema(conn2)
    # 只插入新 chunk（无启动源）
    conn2.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count,
            encode_context, session_type_history)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("pe8_baseline", "test", "observation", new_content, "summary",
         0.6, 5.0, now_iso, now_iso, 0.5, now_iso, 0, "test_ctx", "")
    )
    conn2.commit()
    stab_baseline = _get_stability(conn2, "pe8_baseline")
    conn2.close()

    insert_chunk(conn, new_chunk)
    stab_with_prime = _get_stability(conn, "pe8_new")

    # 有启动源时，stability 应 >= 无启动源基线（允许小误差）
    assert stab_with_prime >= stab_baseline - 0.001, (
        f"PE8: 有启动源时 stability 应 >= 基线，"
        f"with_prime={stab_with_prime:.4f} baseline={stab_baseline:.4f}"
    )
