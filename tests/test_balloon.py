#!/usr/bin/env python3
"""
迭代46 Memory Balloon — 弹性配额动态分配 测试
验证：balloon_quota / config tunable / kswapd集成 / cgroup集成 / 多项目分配
"""
import os
import sys
import sqlite3
from pathlib import Path
from datetime import datetime, timezone, timedelta

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

import tmpfs  # noqa: F401 — tmpfs isolation (iter54), must precede store import
from store import (
    open_db, ensure_schema, get_project_chunk_count,
    balloon_quota, kswapd_scan, cgroup_throttle_check,
)
from config import get as _cfg

PASSED = 0
FAILED = 0


def _fresh_db():
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    return conn


def _insert_chunks(conn, project, count, age_days=0, importance=0.5, access_count=0):
    now = datetime.now(timezone.utc)
    created = (now - timedelta(days=age_days)).isoformat()
    for i in range(count):
        conn.execute(
            """INSERT INTO memory_chunks
               (id, project, chunk_type, summary, content, importance,
                created_at, last_accessed, access_count, lru_gen, oom_adj)
               VALUES (?, ?, 'decision', ?, ?, ?, ?, ?, ?, 0, 0)""",
            (f"{project}_chunk_{i}_{age_days}d", project, f"test {i}",
             f"content {i}", importance, created, created, access_count),
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
    check("balloon.global_pool 已注册", _cfg("balloon.global_pool") == 1000)
    check("balloon.min_quota 已注册", _cfg("balloon.min_quota") == 30)
    check("balloon.max_quota 已注册", _cfg("balloon.max_quota") == 500)
    check("balloon.activity_window_days 已注册", _cfg("balloon.activity_window_days") == 14)


    # ── Test 2: 单项目时获得接近全部 pool ──
    print("\nTest 2: 单项目独占 pool")
    conn = _fresh_db()
    _insert_chunks(conn, "proj_a", 50, age_days=0, access_count=5)
    result = balloon_quota(conn, "proj_a")
    check("quota > min_quota", result["quota"] > _cfg("balloon.min_quota"))
    check("quota <= max_quota", result["quota"] <= _cfg("balloon.max_quota"))
    check("total_projects = 1", result["total_projects"] == 1)
    check("fallback = False", result["fallback"] is False)
    check("activity_score > 0", result["activity_score"] > 0)
    # 单项目应获得 min_quota + 全部 distributable
    expected_max = min(_cfg("balloon.max_quota"), _cfg("balloon.global_pool"))
    check(f"quota 接近 max_quota ({result['quota']})", result["quota"] >= min(expected_max, 500))


    # ── Test 3: 多项目按活跃度分配 ──
    print("\nTest 3: 多项目按活跃度加权分配")
    conn2 = _fresh_db()
    # 活跃项目：50个近期chunk，多次被访问
    _insert_chunks(conn2, "active_proj", 50, age_days=1, access_count=10)
    # 不活跃项目：30个旧chunk，零访问
    _insert_chunks(conn2, "stale_proj", 30, age_days=30, access_count=0)

    active_result = balloon_quota(conn2, "active_proj")
    stale_result = balloon_quota(conn2, "stale_proj")

    check("活跃项目 quota > 不活跃项目 quota",
          active_result["quota"] > stale_result["quota"])
    check("活跃项目 activity_score > 不活跃项目",
          active_result["activity_score"] > stale_result["activity_score"])
    check("不活跃项目 quota >= min_quota",
          stale_result["quota"] >= _cfg("balloon.min_quota"))
    check("两个项目 total_projects = 2", active_result["total_projects"] == 2)


    # ── Test 4: 空数据库 ──
    print("\nTest 4: 空数据库降级")
    conn3 = _fresh_db()
    result_empty = balloon_quota(conn3, "nonexistent")
    check("空库不报错", result_empty is not None)
    check("空库 total_projects = 0", result_empty["total_projects"] == 0)
    check("空库有合理 quota", result_empty["quota"] >= _cfg("balloon.min_quota"))


    # ── Test 5: kswapd_scan 集成 — 使用 balloon 动态配额 ──
    print("\nTest 5: kswapd_scan 集成 balloon")
    conn4 = _fresh_db()
    _insert_chunks(conn4, "ksw_proj", 20, age_days=1, access_count=3)
    ksw = kswapd_scan(conn4, "ksw_proj", incoming_count=1)
    check("kswapd 返回 quota 字段", "quota" in ksw)
    # 单项目 balloon 配额应远大于 20 chunks → ZONE_OK
    check("kswapd zone = OK（配额充足）", ksw["zone"] == "OK")
    check("kswapd quota 来自 balloon（> 0）", ksw["quota"] > 0)


    # ── Test 6: cgroup_throttle_check 集成 balloon ──
    print("\nTest 6: cgroup_throttle_check 集成 balloon")
    conn5 = _fresh_db()
    _insert_chunks(conn5, "cg_proj", 20, age_days=1, access_count=3)
    cg = cgroup_throttle_check(conn5, "cg_proj", incoming_count=1)
    check("cgroup 返回 quota 字段", "quota" in cg)
    check("cgroup zone = OK（配额充足）", cg["zone"] == "OK")
    check("cgroup quota 来自 balloon", cg["quota"] > 0)


    # ── Test 7: 三项目分配公平性 ──
    print("\nTest 7: 三项目分配公平性")
    conn6 = _fresh_db()
    # 高活跃：100 chunk，最近，多次访问
    _insert_chunks(conn6, "high", 100, age_days=0, access_count=20)
    # 中活跃：50 chunk，最近
    _insert_chunks(conn6, "mid", 50, age_days=3, access_count=3)
    # 低活跃：10 chunk，很旧
    _insert_chunks(conn6, "low", 10, age_days=60, access_count=0)

    q_high = balloon_quota(conn6, "high")["quota"]
    q_mid = balloon_quota(conn6, "mid")["quota"]
    q_low = balloon_quota(conn6, "low")["quota"]

    check("high > mid > low", q_high > q_mid > q_low)
    check("low >= min_quota", q_low >= _cfg("balloon.min_quota"))
    check("high <= max_quota", q_high <= _cfg("balloon.max_quota"))
    # 总分配不超过 global_pool
    total_allocated = q_high + q_mid + q_low
    check(f"总分配 {total_allocated} <= global_pool {_cfg('balloon.global_pool')}",
          total_allocated <= _cfg("balloon.global_pool"))


    # ── Test 8: 不存在的项目查询 ──
    print("\nTest 8: 不存在项目的 balloon 查询")
    conn7 = _fresh_db()
    _insert_chunks(conn7, "existing", 50, age_days=1, access_count=5)
    ghost = balloon_quota(conn7, "ghost_project")
    check("不存在项目有 quota", ghost["quota"] >= _cfg("balloon.min_quota"))
    check("不存在项目 activity_score 为保底值", ghost["activity_score"] == 1.0)


    # ── 结果 ──
    print(f"\n{'='*40}")
    print(f"Memory Balloon 测试: {PASSED} passed, {FAILED} failed / {PASSED + FAILED} total")
    if FAILED:
        sys.exit(1)
