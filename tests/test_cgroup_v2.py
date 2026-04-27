#!/usr/bin/env python3
"""
迭代40 测试：cgroup v2 memory.high — Soft Quota Throttling

测试 cgroup_throttle_check() 在不同水位下的行为：
  T1 低于 memory_high → 不 throttle
  T2 恰好在 memory_high 边界 → throttle 开始
  T3 memory_high 和 pages_min 之间 → throttle 线性加重
  T4 超过 pages_min → throttle 最重（factor=throttle_factor, oom_adj=max）
  T5 空项目 → 不 throttle
  T6 throttle 渐进性：越接近 pages_min，importance 衰减越重
  T7 incoming_count 影响水位
  T8 性能测试
"""
import sys
import os
import sqlite3
import json
import time
from pathlib import Path
from datetime import datetime, timezone

# 迭代46 兼容：固定 balloon 配额为 200（测试预设水位基于 quota=200）
os.environ["MEMORY_OS_BALLOON_MAX_QUOTA"] = "200"
os.environ["MEMORY_OS_BALLOON_MIN_QUOTA"] = "200"
os.environ["MEMORY_OS_BALLOON_GLOBAL_POOL"] = "200"

# 设置路径
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

import tmpfs  # noqa: F401 — tmpfs isolation (iter54), must precede store import
from store import open_db, ensure_schema, insert_chunk, cgroup_throttle_check, get_project_chunk_count


def _setup_db(chunk_count: int = 0, project: str = "test_cgroup_v2") -> sqlite3.Connection:
    """创建内存数据库并插入指定数量的 chunk。"""
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA journal_mode=WAL")
    ensure_schema(conn)

    now_iso = datetime.now(timezone.utc).isoformat()
    for i in range(chunk_count):
        conn.execute("""
            INSERT INTO memory_chunks
            (id, created_at, updated_at, project, source_session,
             chunk_type, content, summary, tags, importance,
             retrievability, last_accessed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            f"chunk_{i}", now_iso, now_iso, project, "test_session",
            "decision", f"test content {i}", f"test summary {i}",
            "[]", 0.7, 0.5, now_iso,
        ))
    conn.commit()
    return conn


def test_01_below_memory_high():
    """T1: 水位低于 memory_high → 不 throttle"""
    # 默认 quota=200, memory_high=85% → 170 chunks
    # 放入 100 chunks → 50% < 85%
    conn = _setup_db(100, "test_project")
    result = cgroup_throttle_check(conn, "test_project", incoming_count=1)
    conn.close()

    assert not result["throttled"], f"should not throttle at {result['watermark_pct']}%"
    assert result["zone"] == "OK"
    assert result["importance_factor"] == 1.0
    assert result["oom_adj_delta"] == 0
    print(f"  T1 PASS: watermark={result['watermark_pct']}% → zone=OK")


def test_02_at_memory_high_boundary():
    """T2: 水位恰好在 memory_high 边界 → throttle 开始"""
    # quota=200, memory_high=85% → 170 chunks
    # 放入 170 chunks + 1 incoming → 171/200 = 85.5% → throttle
    conn = _setup_db(170, "test_project")
    result = cgroup_throttle_check(conn, "test_project", incoming_count=1)
    conn.close()

    assert result["throttled"], f"should throttle at {result['watermark_pct']}%"
    assert result["zone"] == "THROTTLE"
    assert result["importance_factor"] < 1.0
    assert result["oom_adj_delta"] > 0
    print(f"  T2 PASS: watermark={result['watermark_pct']}% → zone=THROTTLE factor={result['importance_factor']}")


def test_03_between_high_and_min():
    """T3: 水位在 memory_high 和 pages_min 之间 → 中等 throttle"""
    # quota=200, memory_high=85%=170, pages_min=95%=190
    # 放入 180 chunks → 90% → 中间位置
    conn = _setup_db(180, "test_project")
    result = cgroup_throttle_check(conn, "test_project", incoming_count=0)
    conn.close()

    assert result["throttled"]
    assert result["zone"] == "THROTTLE"
    # importance_factor 应该在 throttle_factor(0.7) 和 1.0 之间
    assert 0.7 <= result["importance_factor"] <= 1.0, \
        f"factor should be between 0.7 and 1.0, got {result['importance_factor']}"
    print(f"  T3 PASS: watermark={result['watermark_pct']}% → factor={result['importance_factor']} oom_adj={result['oom_adj_delta']}")


def test_04_above_pages_min():
    """T4: 水位超过 pages_min → 最重 throttle"""
    # quota=200, pages_min=95%=190
    # 放入 192 chunks → 96% → 超过 pages_min
    conn = _setup_db(192, "test_project")
    result = cgroup_throttle_check(conn, "test_project", incoming_count=0)
    conn.close()

    assert result["throttled"]
    assert result["zone"] == "THROTTLE"
    # 超过 pages_min 时 overshoot=1.0，factor 应该接近 throttle_factor(0.7)
    assert result["importance_factor"] <= 0.71, \
        f"factor should be ~0.7 at 96%, got {result['importance_factor']}"
    # oom_adj_delta 应该接近 throttle_oom_adj(300)
    assert result["oom_adj_delta"] >= 280, \
        f"oom_adj should be ~300 at 96%, got {result['oom_adj_delta']}"
    print(f"  T4 PASS: watermark={result['watermark_pct']}% → factor={result['importance_factor']} oom_adj={result['oom_adj_delta']}")


def test_05_empty_project():
    """T5: 空项目 → 不 throttle"""
    conn = _setup_db(0, "empty_project")
    result = cgroup_throttle_check(conn, "empty_project", incoming_count=5)
    conn.close()

    assert not result["throttled"]
    assert result["zone"] == "OK"
    assert result["watermark_pct"] == 2.5  # 5/200 = 2.5%
    print(f"  T5 PASS: empty project → zone=OK watermark={result['watermark_pct']}%")


def test_06_progressive_throttle():
    """T6: throttle 渐进性 — 越接近 pages_min，衰减越重"""
    # 测试多个水位点，验证 importance_factor 单调递减
    factors = []
    for count in [170, 175, 180, 185, 190]:
        conn = _setup_db(count, "test_project")
        result = cgroup_throttle_check(conn, "test_project", incoming_count=0)
        conn.close()
        factors.append((count, result["importance_factor"], result.get("throttled", False)))

    print(f"  T6 progressive throttle:")
    for count, factor, throttled in factors:
        pct = count / 200 * 100
        print(f"    {count} chunks ({pct:.0f}%) → factor={factor} throttled={throttled}")

    # 170/200=85% 是边界，后续应该单调递减
    throttled_factors = [f for c, f, t in factors if t]
    for i in range(1, len(throttled_factors)):
        assert throttled_factors[i] <= throttled_factors[i-1], \
            f"factor should be monotonically decreasing: {throttled_factors}"
    print(f"  T6 PASS: factors monotonically decrease")


def test_07_incoming_count_affects_watermark():
    """T7: incoming_count 影响水位计算"""
    conn = _setup_db(165, "test_project")

    # incoming=1 → 166/200=83% < 85% → OK
    r1 = cgroup_throttle_check(conn, "test_project", incoming_count=1)
    # incoming=10 → 175/200=87.5% > 85% → THROTTLE
    r2 = cgroup_throttle_check(conn, "test_project", incoming_count=10)
    conn.close()

    assert not r1["throttled"], f"165+1=83% should be OK"
    assert r2["throttled"], f"165+10=87.5% should be THROTTLE"
    print(f"  T7 PASS: incoming=1 → OK({r1['watermark_pct']}%), incoming=10 → THROTTLE({r2['watermark_pct']}%)")


def test_08_performance():
    """T8: 性能测试：throttle 检查应 < 1ms"""
    conn = _setup_db(180, "test_project")

    times = []
    for _ in range(100):
        t0 = time.time()
        cgroup_throttle_check(conn, "test_project", incoming_count=1)
        times.append((time.time() - t0) * 1000)
    conn.close()

    avg_ms = sum(times) / len(times)
    p95_ms = sorted(times)[94]
    assert avg_ms < 1.0, f"avg should be < 1ms, got {avg_ms:.3f}ms"
    print(f"  T8 PASS: avg={avg_ms:.3f}ms p95={p95_ms:.3f}ms (100 iterations)")


if __name__ == "__main__":
    print("迭代40 测试：cgroup v2 memory.high — Soft Quota Throttling")
    print("=" * 60)

    tests = [
        test_01_below_memory_high,
        test_02_at_memory_high_boundary,
        test_03_between_high_and_min,
        test_04_above_pages_min,
        test_05_empty_project,
        test_06_progressive_throttle,
        test_07_incoming_count_affects_watermark,
        test_08_performance,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test.__name__}: {e}")
            failed += 1

    print("=" * 60)
    print(f"结果：{passed}/{passed + failed} 通过")

    if failed > 0:
        sys.exit(1)
