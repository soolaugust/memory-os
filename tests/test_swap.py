#!/usr/bin/env python3
"""
迭代 33：Swap Tier 测试
OS 类比：Linux Swap / zswap — 冷数据交换分区

测试矩阵：
  T1  swap_out 基本功能（chunk 移到 swap 表，主表删除）
  T2  swap_in 基本功能（从 swap 恢复到主表）
  T3  swap_out + swap_in 往返一致性（数据完整性）
  T4  低 importance chunk 直接删除（不 swap）
  T5  swap 分区容量控制（超 max_chunks 淘汰最旧）
  T6  swap_fault 关键词匹配
  T7  swap_fault 无匹配返回空列表
  T8  kswapd 集成（evict 改用 swap_out）
  T9  proc_stats 含 swap 统计
  T10 config.py 新增 swap.* tunable
  T11 swap_in 不存在的 ID 返回 not_found
"""
import sys
import os
import json
import time

# 设置路径
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("CLAUDE_CWD", str(__import__("pathlib").Path(__file__).parent.parent.parent.parent.parent))

import tmpfs  # noqa: F401 — tmpfs isolation (iter54), must precede store import
from store import (
    open_db, ensure_schema, insert_chunk, get_chunks, get_chunk_count,
    swap_out, swap_in, swap_fault, get_swap_count, proc_stats,
    evict_lowest_retention, kswapd_scan, delete_chunks,
)
from config import get as sysctl_get
from schema import MemoryChunk

_TEST_PROJECT = "test_swap_project"
_PASS = 0
_FAIL = 0


def _make_chunk(summary: str, importance: float = 0.7,
                chunk_type: str = "decision", project: str = _TEST_PROJECT) -> dict:
    chunk = MemoryChunk(
        project=project,
        source_session="test_session",
        chunk_type=chunk_type,
        content=f"[{chunk_type}] {summary}",
        summary=summary,
        tags=[chunk_type, project],
        importance=importance,
        retrievability=0.3,
    )
    return chunk.to_dict()


def _assert(name: str, condition: bool, detail: str = ""):
    global _PASS, _FAIL
    if condition:
        _PASS += 1
        print(f"  PASS  {name}")
    else:
        _FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def test_swap_out_basic():
    """T1: swap_out 基本功能"""
    conn = open_db(":memory:")
    ensure_schema(conn)

    chunk = _make_chunk("BM25 hybrid tokenize 优于 unigram", importance=0.8)
    insert_chunk(conn, chunk)
    conn.commit()

    _assert("T1.1 chunk in main table", get_chunk_count(conn) == 1)

    result = swap_out(conn, [chunk["id"]])
    conn.commit()

    _assert("T1.2 swapped_count=1", result["swapped_count"] == 1)
    _assert("T1.3 main table empty", get_chunk_count(conn) == 0)
    _assert("T1.4 swap table has 1", get_swap_count(conn) == 1)

    conn.close()


def test_swap_in_basic():
    """T2: swap_in 基本功能"""
    conn = open_db(":memory:")
    ensure_schema(conn)

    chunk = _make_chunk("FTS5 检索延迟 < 5ms", importance=0.85)
    insert_chunk(conn, chunk)
    conn.commit()

    swap_out(conn, [chunk["id"]])
    conn.commit()
    _assert("T2.1 main empty after swap_out", get_chunk_count(conn) == 0)

    result = swap_in(conn, [chunk["id"]])
    conn.commit()

    _assert("T2.2 restored_count=1", result["restored_count"] == 1)
    _assert("T2.3 main table has 1", get_chunk_count(conn) == 1)
    _assert("T2.4 swap table empty", get_swap_count(conn) == 0)

    conn.close()


def test_roundtrip_integrity():
    """T3: swap_out + swap_in 往返一致性"""
    conn = open_db(":memory:")
    ensure_schema(conn)

    chunk = _make_chunk("kswapd 三级水位线预淘汰", importance=0.9)
    chunk_id = chunk["id"]
    original_summary = chunk["summary"]
    original_importance = chunk["importance"]
    insert_chunk(conn, chunk)
    conn.commit()

    swap_out(conn, [chunk_id])
    conn.commit()
    swap_in(conn, [chunk_id])
    conn.commit()

    chunks = get_chunks(conn, _TEST_PROJECT)
    _assert("T3.1 one chunk restored", len(chunks) == 1)
    restored = chunks[0]
    _assert("T3.2 summary preserved", restored["summary"] == original_summary)
    _assert("T3.3 importance preserved", restored["importance"] == original_importance,
            f"got {restored['importance']}")

    conn.close()


def test_low_importance_direct_delete():
    """T4: 低 importance chunk 直接删除不 swap"""
    conn = open_db(":memory:")
    ensure_schema(conn)

    # importance=0.3 低于 swap.min_importance_for_swap (0.5)
    chunk = _make_chunk("低价值噪声 chunk", importance=0.3)
    insert_chunk(conn, chunk)
    conn.commit()

    result = swap_out(conn, [chunk["id"]])
    conn.commit()

    _assert("T4.1 swapped_count=0 (too low importance)", result["swapped_count"] == 0)
    _assert("T4.2 main table empty (deleted)", get_chunk_count(conn) == 0)
    _assert("T4.3 swap table empty", get_swap_count(conn) == 0)

    conn.close()


def test_swap_capacity_control():
    """T5: swap 分区容量控制"""
    conn = open_db(":memory:")
    ensure_schema(conn)

    # 设置环境变量临时降低 max_chunks（min=10，所以设 12 来测试）
    os.environ["MEMORY_OS_SWAP_MAX_CHUNKS"] = "12"
    from config import _invalidate_cache
    _invalidate_cache()

    chunks = []
    for i in range(15):
        c = _make_chunk(f"swap capacity test chunk {i}", importance=0.7)
        insert_chunk(conn, c)
        chunks.append(c)
    conn.commit()

    # 一次性 swap out 全部 15 个
    ids = [c["id"] for c in chunks]
    result = swap_out(conn, ids)
    conn.commit()

    _assert("T5.1 all 15 swapped", result["swapped_count"] == 15)
    _assert("T5.2 overflow evicted from swap", result["evicted_from_swap"] == 3,
            f"got {result['evicted_from_swap']}")
    _assert("T5.3 swap table has 12", get_swap_count(conn) == 12)

    # 清理环境变量
    del os.environ["MEMORY_OS_SWAP_MAX_CHUNKS"]
    _invalidate_cache()
    conn.close()


def test_swap_fault_match():
    """T6: swap_fault 关键词匹配"""
    conn = open_db(":memory:")
    ensure_schema(conn)

    chunk = _make_chunk("BM25 hybrid tokenize 优于 unigram 分词", importance=0.8)
    insert_chunk(conn, chunk)
    conn.commit()

    swap_out(conn, [chunk["id"]])
    conn.commit()
    _assert("T6.1 chunk in swap", get_swap_count(conn) == 1)

    matches = swap_fault(conn, "BM25 tokenize", _TEST_PROJECT)
    _assert("T6.2 found match", len(matches) >= 1)
    if matches:
        _assert("T6.3 correct id", matches[0]["id"] == chunk["id"])
        _assert("T6.4 has hit_count", matches[0]["hit_count"] > 0)

    conn.close()


def test_swap_fault_no_match():
    """T7: swap_fault 无匹配返回空"""
    conn = open_db(":memory:")
    ensure_schema(conn)

    chunk = _make_chunk("memory compaction 碎片整理", importance=0.7)
    insert_chunk(conn, chunk)
    conn.commit()

    swap_out(conn, [chunk["id"]])
    conn.commit()

    matches = swap_fault(conn, "kubernetes deployment", _TEST_PROJECT)
    _assert("T7.1 no match", len(matches) == 0)

    conn.close()


def test_kswapd_uses_swap():
    """T8: kswapd 集成 — evict 改用 swap_out"""
    conn = open_db(":memory:")
    ensure_schema(conn)

    # 写入一些 chunk
    for i in range(5):
        c = _make_chunk(f"kswapd integration test {i}", importance=0.55)
        insert_chunk(conn, c)
    conn.commit()

    # 手动调用 evict_lowest_retention
    evicted = evict_lowest_retention(conn, _TEST_PROJECT, 2)
    conn.commit()

    _assert("T8.1 evicted 2 from main", len(evicted) == 2)
    _assert("T8.2 main has 3", get_chunk_count(conn) == 3,
            f"got {get_chunk_count(conn)}")
    # 被 evict 的 chunk 应该在 swap 中（importance > 0.5）
    swap_count = get_swap_count(conn)
    _assert("T8.3 evicted chunks in swap", swap_count == 2,
            f"got {swap_count}")

    conn.close()


def test_proc_stats_swap():
    """T9: proc_stats 含 swap 统计"""
    conn = open_db(":memory:")
    ensure_schema(conn)

    chunk = _make_chunk("proc stats swap test", importance=0.7)
    insert_chunk(conn, chunk)
    conn.commit()
    swap_out(conn, [chunk["id"]])
    conn.commit()

    stats = proc_stats(conn)
    _assert("T9.1 has swap key", "swap" in stats)
    _assert("T9.2 swap total=1", stats["swap"]["total"] == 1,
            f"got {stats['swap']}")

    conn.close()


def test_swap_config():
    """T10: config.py swap.* tunable"""
    _assert("T10.1 swap.max_chunks exists", sysctl_get("swap.max_chunks") == 100)
    _assert("T10.2 swap.min_importance_for_swap", sysctl_get("swap.min_importance_for_swap") == 0.5)
    _assert("T10.3 swap.fault_top_k", sysctl_get("swap.fault_top_k") == 2)


def test_swap_in_not_found():
    """T11: swap_in 不存在的 ID"""
    conn = open_db(":memory:")
    ensure_schema(conn)

    result = swap_in(conn, ["nonexistent-id-123"])
    _assert("T11.1 not_found=1", result["not_found"] == 1)
    _assert("T11.2 restored_count=0", result["restored_count"] == 0)

    conn.close()


if __name__ == "__main__":
    t0 = time.time()
    print("=" * 60)
    print("迭代 33：Swap Tier 测试")
    print("=" * 60)

    test_swap_out_basic()
    test_swap_in_basic()
    test_roundtrip_integrity()
    test_low_importance_direct_delete()
    test_swap_capacity_control()
    test_swap_fault_match()
    test_swap_fault_no_match()
    test_kswapd_uses_swap()
    test_proc_stats_swap()
    test_swap_config()
    test_swap_in_not_found()

    elapsed = (time.time() - t0) * 1000
    print("=" * 60)
    print(f"结果：{_PASS} PASS / {_FAIL} FAIL  ({elapsed:.0f}ms)")
    print("=" * 60)
    sys.exit(1 if _FAIL > 0 else 0)
