#!/usr/bin/env python3
"""
迭代44 MGLRU (Multi-Gen LRU) 测试
验证：aging / promote / stats / evict 集成 / config tunable
"""
import os
import sys
import sqlite3
from pathlib import Path
from datetime import datetime, timezone, timedelta

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

os.environ["MEMORY_OS_MGLRU_MAX_GEN"] = "4"
os.environ["MEMORY_OS_MGLRU_AGING_INTERVAL_HOURS"] = "1"  # 测试时最小间隔1h（min=1）

import tmpfs  # noqa: F401 — tmpfs isolation (iter54), must precede store import
from store import (
    open_db, ensure_schema, get_chunks,
    mglru_aging, mglru_promote, mglru_stats, evict_lowest_retention,
)
from config import get as _cfg

PROJECT = "test_mglru"
PASSED = 0
FAILED = 0


def _fresh_db():
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    return conn


def _insert(conn, cid, importance=0.5, chunk_type="decision", age_days=0, access_count=0):
    now = datetime.now(timezone.utc)
    created = (now - timedelta(days=age_days)).isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, summary, content, importance, created_at, last_accessed, access_count, lru_gen, oom_adj)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0)""",
        (cid, PROJECT, chunk_type, f"test {cid}", f"content {cid}",
         importance, created, created, access_count),
    )
    conn.commit()


def check(name, condition):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  ✓ {name}")
    else:
        FAILED += 1
        print(f"  ✗ {name}")


# ── Test 1: config tunable 注册 ──

if __name__ == "__main__":
    print("Test 1: config tunable 注册")
    max_gen = _cfg("mglru.max_gen")
    check("max_gen 已注册且有值", max_gen is not None and isinstance(max_gen, int))
    check("max_gen 默认 4", _cfg("mglru.max_gen") == 4)

    interval = _cfg("mglru.aging_interval_hours")
    check("aging_interval_hours 已注册", interval is not None)
    # 环境变量覆盖测试
    check("env 覆盖 aging_interval_hours（clamped to min=1）", interval == 1)


    # ── Test 2: mglru_aging 基本功能 ──
    print("\nTest 2: mglru_aging 基本功能")
    conn = _fresh_db()
    for i in range(5):
        _insert(conn, f"chunk_{i}", importance=0.5 + i * 0.1)

    # 清除时间戳文件以允许 aging
    ts_file = Path.home() / ".claude" / "memory-os" / ".mglru_last_aging"
    if ts_file.exists():
        ts_file.unlink()

    result = mglru_aging(conn, PROJECT)
    check("aging 执行成功", result["aged"] is True)
    check("affected_count = 5", result["affected_count"] == 5)
    check("gen_distribution 含 gen1", "gen1" in result.get("gen_distribution", {}))

    # 验证 chunk 的 lru_gen 已递增
    gen_vals = conn.execute(
        "SELECT id, COALESCE(lru_gen, 0) FROM memory_chunks WHERE project=? ORDER BY id",
        (PROJECT,),
    ).fetchall()
    check("所有 chunk gen=1", all(g == 1 for _, g in gen_vals))


    # ── Test 3: 连续 aging 推进 gen ──
    print("\nTest 3: 连续 aging")
    if ts_file.exists():
        ts_file.unlink()
    result2 = mglru_aging(conn, PROJECT)
    check("第二次 aging 成功", result2["aged"] is True)

    gen_vals2 = conn.execute(
        "SELECT COALESCE(lru_gen, 0) FROM memory_chunks WHERE project=?",
        (PROJECT,),
    ).fetchall()
    check("所有 chunk gen=2", all(g[0] == 2 for g in gen_vals2))


    # ── Test 4: max_gen 上限 ──
    print("\nTest 4: max_gen 上限")
    for _ in range(10):
        if ts_file.exists():
            ts_file.unlink()
        mglru_aging(conn, PROJECT)

    gen_vals3 = conn.execute(
        "SELECT COALESCE(lru_gen, 0) FROM memory_chunks WHERE project=?",
        (PROJECT,),
    ).fetchall()
    max_gen_val = _cfg("mglru.max_gen")
    check(f"gen 不超过 max_gen={max_gen_val}", all(g[0] <= max_gen_val for g in gen_vals3))
    check(f"gen 等于 max_gen={max_gen_val}", all(g[0] == max_gen_val for g in gen_vals3))


    # ── Test 5: mglru_promote — 被访问 chunk 晋升 gen 0 ──
    print("\nTest 5: mglru_promote")
    promoted = mglru_promote(conn, ["chunk_0", "chunk_2"])
    check("promote 返回 2", promoted == 2)

    gen_after_promote = {}
    for row in conn.execute(
        "SELECT id, COALESCE(lru_gen, 0) FROM memory_chunks WHERE project=?",
        (PROJECT,),
    ).fetchall():
        gen_after_promote[row[0]] = row[1]

    check("chunk_0 gen=0（被 promote）", gen_after_promote.get("chunk_0") == 0)
    check("chunk_2 gen=0（被 promote）", gen_after_promote.get("chunk_2") == 0)
    check(f"chunk_1 gen={max_gen_val}（未被 promote）", gen_after_promote.get("chunk_1") == max_gen_val)


    # ── Test 6: mglru_stats ──
    print("\nTest 6: mglru_stats")
    stats = mglru_stats(conn, PROJECT)
    check("stats 包含 gen_distribution", "gen_distribution" in stats)
    check("stats 包含 hot_pct", "hot_pct" in stats)
    check("stats 包含 cold_pct", "cold_pct" in stats)
    check("total = 5", stats["total"] == 5)
    check("hot_pct = 40%", stats["hot_pct"] == 40.0)


    # ── Test 7: evict 优先淘汰 high gen ──
    print("\nTest 7: evict 优先淘汰 high gen（集成验证）")
    conn2 = _fresh_db()
    for i in range(3):
        _insert(conn2, f"hot_{i}", importance=0.8, access_count=5)
    for i in range(3):
        _insert(conn2, f"cold_{i}", importance=0.3, access_count=0)

    conn2.execute("UPDATE memory_chunks SET lru_gen = 0 WHERE id LIKE 'hot_%'")
    conn2.execute("UPDATE memory_chunks SET lru_gen = 4 WHERE id LIKE 'cold_%'")
    conn2.commit()

    evicted = evict_lowest_retention(conn2, PROJECT, count=2)
    conn2.commit()
    check("淘汰了 2 个 chunk", len(evicted) == 2)
    check("淘汰的都是 cold 前缀", all("cold" in eid for eid in evicted))


    # ── Test 8: promote 空列表 ──
    print("\nTest 8: promote 空列表")
    result_empty = mglru_promote(conn, [])
    check("空列表返回 0", result_empty == 0)


    # ── 结果 ──
    print(f"\n{'='*40}")
    print(f"MGLRU 测试: {PASSED} passed, {FAILED} failed / {PASSED + FAILED} total")
    if FAILED:
        sys.exit(1)
