"""
迭代64：TLB v2 — Multi-Slot + chunk_version Selective Invalidation 测试

OS 类比：N-Way Set-Associative TLB + NFS Weak Cache Consistency (WKC)

测试内容：
  1. chunk_version 基础功能（bump/read）
  2. Multi-Slot TLB 读写和 LRU 淘汰
  3. TLB v1 → v2 格式兼容
  4. chunk_version 选择性失效（insert bump, update_accessed 不 bump）
  5. L1 + L2 命中逻辑
  6. 性能：chunk_version I/O < 1ms
"""
import sys
import os
import json
import time
import tempfile

# ── tmpfs 隔离 ──
_tmpdir = tempfile.mkdtemp(prefix="test_tlb_v2_")
os.environ["MEMORY_OS_DIR"] = _tmpdir
os.environ["MEMORY_OS_DB"] = os.path.join(_tmpdir, "store.db")

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "hooks"))

from store import (
    open_db, ensure_schema, insert_chunk, update_accessed,
    delete_chunks, bump_chunk_version, read_chunk_version,
    CHUNK_VERSION_FILE,
)
from datetime import datetime, timezone

_passed = 0
_failed = 0


def _ok(name, cond, detail=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  ✓ {name}")
    else:
        _failed += 1
        print(f"  ✗ {name} — {detail}")


def _make_chunk(chunk_id, chunk_type="decision", summary="test", importance=0.8):
    now = datetime.now(timezone.utc).isoformat()
    return {
        "id": chunk_id, "created_at": now, "updated_at": now,
        "project": "test", "source_session": "test",
        "chunk_type": chunk_type, "content": "", "summary": summary,
        "tags": "[]", "importance": importance, "retrievability": 0.5,
        "last_accessed": now, "feishu_url": None,
    }


# ── Test 1: chunk_version 基础功能 ──

if __name__ == "__main__":
    print("\n=== Test 1: chunk_version 基础功能 ===")
    # 初始状态应为 0
    v0 = read_chunk_version()
    _ok("initial version = 0", v0 == 0, f"got {v0}")

    # bump 后应为 1
    v1 = bump_chunk_version()
    _ok("bump returns 1", v1 == 1, f"got {v1}")

    v1r = read_chunk_version()
    _ok("read after bump = 1", v1r == 1, f"got {v1r}")

    # 连续 bump
    bump_chunk_version()
    bump_chunk_version()
    v3 = read_chunk_version()
    _ok("3x bump = 3", v3 == 3, f"got {v3}")


    # ── Test 2: insert_chunk 自动 bump ──
    print("\n=== Test 2: insert_chunk 自动 bump ===")
    # Reset version
    CHUNK_VERSION_FILE.write_text("0")
    conn = open_db()
    ensure_schema(conn)

    v_before = read_chunk_version()
    insert_chunk(conn, _make_chunk("test1"))
    conn.commit()
    v_after = read_chunk_version()
    _ok("insert_chunk bumps version", v_after > v_before,
        f"before={v_before}, after={v_after}")


    # ── Test 3: update_accessed 不 bump ──
    print("\n=== Test 3: update_accessed 不 bump ===")
    v_before = read_chunk_version()
    update_accessed(conn, ["test1"])
    conn.commit()
    v_after = read_chunk_version()
    _ok("update_accessed does NOT bump version", v_after == v_before,
        f"before={v_before}, after={v_after}")


    # ── Test 4: delete_chunks bump ──
    print("\n=== Test 4: delete_chunks bump ===")
    insert_chunk(conn, _make_chunk("test_del"))
    conn.commit()
    v_before = read_chunk_version()
    delete_chunks(conn, ["test_del"])
    conn.commit()
    v_after = read_chunk_version()
    _ok("delete_chunks bumps version", v_after > v_before,
        f"before={v_before}, after={v_after}")

    # delete 0 chunks should not bump
    v_before = read_chunk_version()
    delete_chunks(conn, [])
    v_after = read_chunk_version()
    _ok("delete 0 chunks does NOT bump", v_after == v_before)


    # ── Test 5: Multi-Slot TLB 读写 ──
    print("\n=== Test 5: Multi-Slot TLB 读写 ===")
    from retriever import _tlb_read, _tlb_write, _read_chunk_version, TLB_FILE

    # 写入第一个 slot
    _tlb_write("hash_a", "inj_1", 0.0)
    tlb = _tlb_read()
    _ok("TLB has slots dict", "slots" in tlb, f"keys={list(tlb.keys())}")
    _ok("slot hash_a exists", "hash_a" in tlb.get("slots", {}))
    _ok("slot hash_a injection_hash", tlb["slots"]["hash_a"]["injection_hash"] == "inj_1")

    # 写入第二个 slot
    _tlb_write("hash_b", "inj_2", 0.0)
    tlb = _tlb_read()
    _ok("multi-slot: hash_a still exists", "hash_a" in tlb.get("slots", {}))
    _ok("multi-slot: hash_b added", "hash_b" in tlb.get("slots", {}))

    # 更新已有 slot
    _tlb_write("hash_a", "inj_3", 0.0)
    tlb = _tlb_read()
    _ok("update slot: hash_a injection_hash updated", tlb["slots"]["hash_a"]["injection_hash"] == "inj_3")


    # ── Test 6: TLB LRU 淘汰 ──
    print("\n=== Test 6: TLB LRU 淘汰 ===")
    # 清空 TLB
    TLB_FILE.write_text("{}")
    # 写入 10 个 slot（超过默认 max_entries=8）
    for i in range(10):
        _tlb_write(f"lru_{i}", f"inj_{i}", 0.0)
    tlb = _tlb_read()
    slot_count = len(tlb.get("slots", {}))
    _ok("LRU eviction: slots <= 8", slot_count <= 8, f"got {slot_count}")
    # 最后写入的应该保留
    _ok("LRU: newest slot lru_9 kept", "lru_9" in tlb.get("slots", {}))
    # 最早写入的应该被淘汰
    _ok("LRU: oldest slot lru_0 evicted", "lru_0" not in tlb.get("slots", {}))
    _ok("LRU: oldest slot lru_1 evicted", "lru_1" not in tlb.get("slots", {}))


    # ── Test 7: TLB v1 → v2 格式兼容 ──
    print("\n=== Test 7: TLB v1 → v2 格式兼容 ===")
    # 写入 v1 格式
    TLB_FILE.write_text(json.dumps({
        "prompt_hash": "old_hash",
        "injection_hash": "old_inj",
        "db_mtime": 12345.6,
    }))
    tlb = _tlb_read()
    _ok("v1 compat: has slots", "slots" in tlb)
    _ok("v1 compat: old_hash migrated", "old_hash" in tlb.get("slots", {}))
    _ok("v1 compat: injection_hash preserved",
        tlb["slots"]["old_hash"]["injection_hash"] == "old_inj")
    _ok("v1 compat: chunk_version=-1 forces miss", tlb.get("chunk_version") == -1)


    # ── Test 8: chunk_version 存储在 TLB 中 ──
    print("\n=== Test 8: chunk_version 存储在 TLB 中 ===")
    CHUNK_VERSION_FILE.write_text("42")
    TLB_FILE.write_text("{}")
    _tlb_write("test_ver", "inj_ver", 0.0)
    tlb = _tlb_read()
    _ok("TLB stores chunk_version", tlb.get("chunk_version") == 42,
        f"got {tlb.get('chunk_version')}")


    # ── Test 9: _read_chunk_version 一致性 ──
    print("\n=== Test 9: _read_chunk_version 一致性 ===")
    CHUNK_VERSION_FILE.write_text("99")
    ver = _read_chunk_version()
    _ok("_read_chunk_version reads correctly", ver == 99, f"got {ver}")

    # 不存在时返回 0
    CHUNK_VERSION_FILE.unlink(missing_ok=True)
    ver = _read_chunk_version()
    _ok("_read_chunk_version returns 0 when missing", ver == 0, f"got {ver}")


    # ── Test 10: 性能 ──
    print("\n=== Test 10: 性能 ===")
    CHUNK_VERSION_FILE.write_text("100")
    t0 = time.time()
    for _ in range(1000):
        read_chunk_version()
    elapsed_ms = (time.time() - t0) * 1000
    avg_us = elapsed_ms / 1000 * 1000  # μs per call
    _ok(f"read_chunk_version 1000x = {elapsed_ms:.1f}ms ({avg_us:.0f}μs/call)",
        elapsed_ms < 500, f"elapsed={elapsed_ms:.1f}ms")

    t0 = time.time()
    for _ in range(100):
        bump_chunk_version()
    elapsed_ms = (time.time() - t0) * 1000
    avg_us = elapsed_ms / 100 * 1000
    _ok(f"bump_chunk_version 100x = {elapsed_ms:.1f}ms ({avg_us:.0f}μs/call)",
        elapsed_ms < 500, f"elapsed={elapsed_ms:.1f}ms")

    t0 = time.time()
    for i in range(100):
        _tlb_write(f"perf_{i}", f"inj_{i}", 0.0)
    elapsed_ms = (time.time() - t0) * 1000
    avg_us = elapsed_ms / 100 * 1000
    _ok(f"_tlb_write 100x = {elapsed_ms:.1f}ms ({avg_us:.0f}μs/call)",
        elapsed_ms < 500, f"elapsed={elapsed_ms:.1f}ms")


    # ── cleanup ──
    conn.close()
    import shutil
    shutil.rmtree(_tmpdir, ignore_errors=True)

    # ── summary ──
    print(f"\n{'='*50}")
    print(f"TLB v2 tests: {_passed} passed, {_failed} failed")
    if _failed > 0:
        sys.exit(1)
