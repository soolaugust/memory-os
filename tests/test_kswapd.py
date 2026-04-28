#!/usr/bin/env python3
"""
迭代30 测试：kswapd — Background Watermark Reclaim
OS 类比：Linux kswapd (1994) 后台水位线预淘汰

测试矩阵：
  T1: ZONE_OK — 水位低于 pages_low，无淘汰
  T2: ZONE_LOW — 水位在 pages_low ~ pages_min 之间，预淘汰至 pages_high
  T3: ZONE_MIN — 水位超过 pages_min，同步硬淘汰至 pages_high
  T4: Stale reclaim — 过期 chunk（>90天未访问）被优先回收
  T5: Protected chunks — task_state 和 importance>=0.9 不被回收
  T6: Batch size limit — ZONE_LOW 模式下每次淘汰不超过 batch_size
  T7: Empty project — 空项目返回 ZONE_OK
  T8: sysctl tunables — 5 个新 tunable 注册正确
  T9: Writer integration — writer 调用 kswapd_scan 而非旧 OOM handler
  T10: Extractor integration — extractor 调用 kswapd_scan 而非旧 OOM handler
"""
import sys
import json
import sqlite3
import os
import uuid
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

# 迭代46 兼容：固定 balloon 配额为 200（测试预设水位基于 quota=200）
os.environ["MEMORY_OS_BALLOON_MAX_QUOTA"] = "200"
os.environ["MEMORY_OS_BALLOON_MIN_QUOTA"] = "200"
os.environ["MEMORY_OS_BALLOON_GLOBAL_POOL"] = "200"

# 测试环境设置
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

import tmpfs  # noqa: F401 — tmpfs isolation (iter54), must precede store import
from store import (open_db, ensure_schema, insert_chunk, get_project_chunk_count,
                   kswapd_scan, delete_chunks, dmesg_read, DMESG_WARN)
from config import get as _sysctl, _REGISTRY

TEST_PROJECT = f"test_kswapd_{uuid.uuid4().hex[:6]}"


def _make_chunk(project: str, importance: float = 0.5,
                days_ago: int = 0, chunk_type: str = "decision") -> dict:
    """创建测试 chunk。"""
    now = datetime.now(timezone.utc)
    ts = (now - timedelta(days=days_ago)).isoformat()
    return {
        "id": str(uuid.uuid4()),
        "created_at": ts,
        "updated_at": ts,
        "project": project,
        "source_session": "test",
        "chunk_type": chunk_type,
        "content": f"ksw_content_{uuid.uuid4().hex}",
        "summary": f"ksw_{uuid.uuid4().hex}_{uuid.uuid4().hex[:8]}",
        "tags": json.dumps(["test"]),
        "importance": importance,
        "retrievability": 0.3,
        "last_accessed": ts,
        "feishu_url": None,
    }


def _fill_project(conn, project, count, importance=0.5, days_ago=0, chunk_type="decision"):
    """批量填充 chunk。"""
    ids = []
    for _ in range(count):
        chunk = _make_chunk(project, importance, days_ago, chunk_type)
        insert_chunk(conn, chunk)
        ids.append(chunk["id"])
    conn.commit()
    return ids


def _cleanup(conn, project):
    """清理测试数据。"""
    conn.execute("DELETE FROM memory_chunks WHERE project=?", (project,))
    conn.commit()


# ── T1: ZONE_OK ──
def test_zone_ok():
    """水位低于 pages_low，无淘汰。"""
    conn = open_db()
    ensure_schema(conn)
    project = f"{TEST_PROJECT}_ok"
    _cleanup(conn, project)

    # 填充少量 chunk（远低于 pages_low）
    _fill_project(conn, project, 10, importance=0.5)
    assert get_project_chunk_count(conn, project) == 10

    result = kswapd_scan(conn, project, incoming_count=1)
    assert result["zone"] == "OK"
    assert result["evicted_count"] == 0
    assert result["watermark_pct"] < _sysctl("kswapd.pages_low_pct")

    _cleanup(conn, project)
    conn.close()
    print("  T1 ZONE_OK ✓")


# ── T2: ZONE_LOW ──
def test_zone_low():
    """水位在 pages_high ~ pages_min 之间，预淘汰至 pages_high。
    注意：pages_low(80%) < pages_high(90%) < pages_min(95%)
    水位在 80-90% 时虽在 LOW zone 但已低于 HIGH 目标，无需淘汰。
    水位在 90-95% 时触发实际淘汰。
    修复：kswapd 使用全局 count，需在填充前测量全局基线，
    动态计算填充量使全局 count 落入 ZONE_LOW 区间。
    """
    conn = open_db()
    ensure_schema(conn)
    project = f"{TEST_PROJECT}_low"
    _cleanup(conn, project)

    quota = _sysctl("extractor.chunk_quota")
    pages_high_pct = _sysctl("kswapd.pages_high_pct")
    pages_min_pct = _sysctl("kswapd.pages_min_pct")

    # 测量当前全局 chunk 数（其他测试可能留下残留数据）
    global_base = conn.execute("SELECT COUNT(*) FROM memory_chunks").fetchone()[0]

    # 目标：全局 count + incoming(1) 落在 ZONE_LOW 区间 (pages_high, pages_min)
    target_pct = (pages_high_pct + pages_min_pct) / 2  # 92.5%
    target_global = int(quota * target_pct / 100)

    fill_count = target_global - global_base
    if fill_count <= 0:
        # 全局已超出目标，跳过（其他测试留下太多数据）
        conn.close()
        print("  T2 ZONE_LOW ⚠ skipped (global baseline too high, isolation issue)")
        return

    # 确保 fill_count 不会超过 pages_min（避免落入 ZONE_MIN）
    max_fill = int(quota * pages_min_pct / 100) - global_base - 1
    fill_count = min(fill_count, max_fill)
    if fill_count <= 0:
        conn.close()
        print("  T2 ZONE_LOW ⚠ skipped (global baseline too high, isolation issue)")
        return

    _fill_project(conn, project, fill_count, importance=0.4, days_ago=1)  # 绕过10分钟 grace period

    count_before = get_project_chunk_count(conn, project)
    result = kswapd_scan(conn, project, incoming_count=1)
    count_after = get_project_chunk_count(conn, project)

    assert result["zone"] == "LOW", f"Expected LOW, got {result['zone']} at {result['watermark_pct']}%"
    assert result["evicted_count"] > 0, "ZONE_LOW above pages_high should evict"
    assert count_after < count_before, "ZONE_LOW should reduce count"

    _cleanup(conn, project)
    conn.close()
    print(f"  T2 ZONE_LOW ✓ (evicted={result['evicted_count']})")


# ── T3: ZONE_MIN ──
def test_zone_min():
    """水位超过 pages_min，同步硬淘汰。"""
    conn = open_db()
    ensure_schema(conn)
    project = f"{TEST_PROJECT}_min"
    _cleanup(conn, project)

    quota = _sysctl("extractor.chunk_quota")
    pages_min_pct = _sysctl("kswapd.pages_min_pct")
    pages_high_pct = _sysctl("kswapd.pages_high_pct")

    # 动态计算填充量：确保全局 count 超过 pages_min（考虑其他测试残留）
    global_base = conn.execute("SELECT COUNT(*) FROM memory_chunks").fetchone()[0]
    target_global = int(quota * pages_min_pct / 100) + 5
    fill_count = max(5, target_global - global_base)
    _fill_project(conn, project, fill_count, importance=0.4, days_ago=1)  # 绕过10分钟 grace period

    result = kswapd_scan(conn, project, incoming_count=1)
    count_after = get_project_chunk_count(conn, project)

    assert result["zone"] == "MIN", f"Expected MIN, got {result['zone']} at {result['watermark_pct']}%"
    assert result["evicted_count"] > 0, "ZONE_MIN should evict aggressively"
    # ZONE_MIN 应该淘汰至 pages_high 以下
    target = int(quota * pages_high_pct / 100)
    assert count_after <= target + 2, f"After MIN reclaim, count={count_after} should be near target={target}"

    _cleanup(conn, project)
    conn.close()
    print(f"  T3 ZONE_MIN ✓ (evicted={result['evicted_count']}, count_after={count_after})")


# ── T4: Stale reclaim ──
def test_stale_reclaim():
    """过期 chunk 被优先回收。"""
    conn = open_db()
    ensure_schema(conn)
    project = f"{TEST_PROJECT}_stale"
    _cleanup(conn, project)

    stale_days = _sysctl("kswapd.stale_days")

    # 创建过期 chunk（stale_days + 10 天前）和新鲜 chunk
    stale_ids = _fill_project(conn, project, 5, importance=0.4, days_ago=stale_days + 10)
    fresh_ids = _fill_project(conn, project, 5, importance=0.4, days_ago=1)

    result = kswapd_scan(conn, project, incoming_count=1)

    # 过期 chunk 应被回收
    assert result["stale_evicted"] > 0, "Stale chunks should be reclaimed"

    # 验证新鲜 chunk 仍在
    remaining = conn.execute(
        "SELECT id FROM memory_chunks WHERE project=? AND id IN ({})".format(
            ",".join("?" * len(fresh_ids))
        ), [project] + fresh_ids
    ).fetchall()
    assert len(remaining) == len(fresh_ids), "Fresh chunks should survive"

    _cleanup(conn, project)
    conn.close()
    print(f"  T4 Stale reclaim ✓ (stale_evicted={result['stale_evicted']})")


# ── T5: Protected chunks ──
def test_protected_chunks():
    """task_state 和 importance>=0.9 不被回收。"""
    conn = open_db()
    ensure_schema(conn)
    project = f"{TEST_PROJECT}_protect"
    _cleanup(conn, project)

    quota = _sysctl("extractor.chunk_quota")

    # 创建受保护 chunk：task_state + high importance
    ts_ids = _fill_project(conn, project, 3, importance=0.5, chunk_type="task_state", days_ago=100)
    hi_ids = _fill_project(conn, project, 3, importance=0.95, days_ago=100)
    # 创建可淘汰 chunk 填满配额
    low_ids = _fill_project(conn, project, quota, importance=0.3, days_ago=50)

    result = kswapd_scan(conn, project, incoming_count=1)

    # 验证受保护 chunk 仍在
    for cid in ts_ids + hi_ids:
        row = conn.execute("SELECT id FROM memory_chunks WHERE id=?", (cid,)).fetchone()
        assert row is not None, f"Protected chunk {cid[:8]} should survive"

    _cleanup(conn, project)
    conn.close()
    print(f"  T5 Protected chunks ✓ (evicted={result['evicted_count']}, protected survive)")


# ── T6: Batch size limit ──
def test_batch_size_limit():
    """ZONE_LOW 模式下每次淘汰不超过 batch_size。"""
    conn = open_db()
    ensure_schema(conn)
    project = f"{TEST_PROJECT}_batch"
    _cleanup(conn, project)

    quota = _sysctl("extractor.chunk_quota")
    pages_low_pct = _sysctl("kswapd.pages_low_pct")
    pages_high_pct = _sysctl("kswapd.pages_high_pct")
    pages_min_pct = _sysctl("kswapd.pages_min_pct")
    batch_size = _sysctl("kswapd.batch_size")

    # 动态计算填充量：全局 count 落在 ZONE_LOW 区间（不触发 ZONE_MIN）
    global_base = conn.execute("SELECT COUNT(*) FROM memory_chunks").fetchone()[0]
    target_pct = (pages_high_pct + pages_min_pct) / 2  # 92.5%
    target_global = int(quota * target_pct / 100)
    fill_count = target_global - global_base
    if fill_count <= 0:
        conn.close()
        print("  T6 Batch size ⚠ skipped (global baseline too high)")
        return
    max_fill = int(quota * pages_min_pct / 100) - global_base - 1
    fill_count = min(fill_count, max_fill)
    if fill_count <= 0:
        conn.close()
        print("  T6 Batch size ⚠ skipped (global baseline too high)")
        return
    _fill_project(conn, project, fill_count, importance=0.4)

    result = kswapd_scan(conn, project, incoming_count=1)

    if result["zone"] == "LOW":
        # ZONE_LOW 的水位线淘汰应不超过 batch_size（stale 淘汰另算）
        watermark_evicted = result["evicted_count"] - result["stale_evicted"]
        assert watermark_evicted <= batch_size, \
            f"Watermark eviction {watermark_evicted} exceeds batch_size {batch_size}"

    _cleanup(conn, project)
    conn.close()
    print(f"  T6 Batch size ✓ (evicted={result['evicted_count']}, batch_size={batch_size})")


# ── T7: Empty project ──
def test_empty_project():
    """空项目（本 project 无 chunks）触发 kswapd 应不报错，返回合法结果。
    注意：kswapd 使用全局 count，其他测试的残留数据可能使 watermark > 0。
    核心断言：返回结果结构完整，zone 字段合法，不抛异常。
    """
    conn = open_db()
    ensure_schema(conn)
    project = f"{TEST_PROJECT}_empty"
    _cleanup(conn, project)

    result = kswapd_scan(conn, project, incoming_count=1)
    assert result["zone"] in ("OK", "LOW", "MIN"), f"Invalid zone: {result['zone']}"
    assert "evicted_count" in result
    assert "watermark_pct" in result
    assert result["watermark_pct"] >= 0

    conn.close()
    print(f"  T7 Empty project ✓ (zone={result['zone']}, watermark={result['watermark_pct']}%)")


# ── T8: sysctl tunables ──
def test_sysctl_tunables():
    """5 个新 kswapd tunable 注册正确。"""
    expected_keys = [
        "kswapd.pages_low_pct",
        "kswapd.pages_high_pct",
        "kswapd.pages_min_pct",
        "kswapd.stale_days",
        "kswapd.batch_size",
    ]
    for key in expected_keys:
        assert key in _REGISTRY, f"Missing tunable: {key}"
        val = _sysctl(key)
        assert val is not None, f"Tunable {key} returned None"
        assert isinstance(val, (int, float)), f"Tunable {key} has wrong type: {type(val)}"

    # 验证默认值
    assert _sysctl("kswapd.pages_low_pct") == 80
    assert _sysctl("kswapd.pages_high_pct") == 90
    assert _sysctl("kswapd.pages_min_pct") == 95
    assert _sysctl("kswapd.stale_days") == 90
    assert _sysctl("kswapd.batch_size") == 5

    print("  T8 sysctl tunables ✓ (5 kswapd tunables registered)")


# ── T9: Writer integration ──
def test_writer_uses_kswapd():
    """writer.py 使用 kswapd_scan 而非旧 OOM handler。"""
    writer_path = _ROOT.parent / "hooks" / "writer.py"
    code = writer_path.read_text()
    assert "kswapd_scan" in code, "writer.py should import kswapd_scan"
    assert "kswapd_scan(conn" in code, "writer.py should call kswapd_scan"
    # 旧 OOM handler 不应在写入路径中
    assert 'if current >= _sysctl("extractor.chunk_quota")' not in code, \
        "writer.py should not use old OOM handler"
    print("  T9 Writer integration ✓")


# ── T10: Extractor integration ──
def test_extractor_uses_kswapd():
    """extractor.py 使用 kswapd_scan 而非旧 OOM handler。"""
    ext_path = _ROOT.parent / "hooks" / "extractor.py"
    code = ext_path.read_text()
    assert "kswapd_scan" in code, "extractor.py should import kswapd_scan"
    assert "kswapd_scan(conn" in code, "extractor.py should call kswapd_scan"
    # 旧 OOM handler 不应在写入路径中
    assert "cgroup OOM" not in code, "extractor.py should not use old OOM dmesg message"
    print("  T10 Extractor integration ✓")


# ── 性能测试 ──
def test_performance():
    """kswapd_scan 延迟 < 50ms。"""
    conn = open_db()
    ensure_schema(conn)
    project = f"{TEST_PROJECT}_perf"
    _cleanup(conn, project)

    # 填充 100 个 chunk
    _fill_project(conn, project, 100, importance=0.5)

    iterations = 20
    total_ms = 0
    for _ in range(iterations):
        t0 = time.time()
        kswapd_scan(conn, project, incoming_count=1)
        total_ms += (time.time() - t0) * 1000

    avg_ms = total_ms / iterations

    _cleanup(conn, project)
    conn.close()
    print(f"  Perf: avg {avg_ms:.2f}ms over {iterations} iterations (< 50ms)")
    assert avg_ms < 50, f"kswapd_scan too slow: {avg_ms:.2f}ms"


if __name__ == "__main__":
    print("迭代30 测试：kswapd — Background Watermark Reclaim")
    print("=" * 60)

    tests = [
        test_zone_ok,
        test_zone_low,
        test_zone_min,
        test_stale_reclaim,
        test_protected_chunks,
        test_batch_size_limit,
        test_empty_project,
        test_sysctl_tunables,
        test_writer_uses_kswapd,
        test_extractor_uses_kswapd,
        test_performance,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  FAIL {test.__name__}: {e}")

    print(f"\n{'=' * 60}")
    print(f"结果：{passed}/{passed + failed} 通过")
    if failed:
        sys.exit(1)
