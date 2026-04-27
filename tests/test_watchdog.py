#!/usr/bin/env python3
"""
迭代35 测试：Watchdog Timer — 自我修复与健康检测
OS 类比：Linux Watchdog (2003) + softlockup detector + hung_task detector
"""
import sys
import os
import json
import sqlite3
import tempfile
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

import tmpfs  # noqa: F401 — tmpfs isolation (iter54), must precede store import
from store import (
    open_db, ensure_schema, insert_chunk, watchdog_check,
    dmesg_log, DMESG_ERR, DMESG_INFO,
)
from schema import MemoryChunk
from config import get as _sysctl

_PASS = 0
_FAIL = 0


def _assert(cond, msg):
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  ✅ {msg}")
    else:
        _FAIL += 1
        print(f"  ❌ {msg}")


def _make_db():
    """创建临时数据库用于测试。"""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db_path = Path(tmp.name)
    conn = open_db(db_path)
    ensure_schema(conn)
    return conn, db_path


def _insert_test_chunk(conn, project="test-proj", summary="test chunk", chunk_type="decision",
                        importance=0.8):
    """插入测试 chunk。"""
    chunk = MemoryChunk(
        project=project,
        source_session="test-session",
        chunk_type=chunk_type,
        content=f"[{chunk_type}] {summary}",
        summary=summary,
        tags=[chunk_type],
        importance=importance,
        retrievability=0.5,
    )
    insert_chunk(conn, chunk.to_dict())
    conn.commit()


# ── Test 1: 健康系统返回 HEALTHY ──
def test_healthy_system():
    print("\nTest 1: 健康系统 → HEALTHY")
    conn, db_path = _make_db()
    try:
        for i in range(5):
            _insert_test_chunk(conn, summary=f"healthy decision {i}")

        result = watchdog_check(conn)
        _assert(result["status"] == "HEALTHY", f"status={result['status']} should be HEALTHY")
        _assert(len(result["checks"]) >= 4, f"checks={len(result['checks'])} should be >= 4")
        _assert(len(result["repairs"]) == 0, f"repairs={len(result['repairs'])} should be 0")
        _assert(result["duration_ms"] < 500, f"duration={result['duration_ms']}ms should be < 500")
    finally:
        conn.close()
        os.unlink(db_path)


# ── Test 2: FTS5 integrity-check 验证 ──
def test_fts5_integrity():
    print("\nTest 2: FTS5 integrity-check 正常通过")
    conn, db_path = _make_db()
    try:
        for i in range(10):
            _insert_test_chunk(conn, summary=f"fts5 test chunk number {i} about BM25 scoring")
        conn.commit()

        # 正常情况下 FTS5 integrity-check 应该通过
        result = watchdog_check(conn)
        fts_check = [c for c in result["checks"] if c["name"] == "fts5_consistency"]
        _assert(len(fts_check) == 1, "fts5_consistency check exists")
        _assert(fts_check[0]["status"] == "ok", f"fts5 status={fts_check[0]['status']} should be ok")

        # 验证 FTS5 搜索仍然工作
        from store import fts_search
        results = fts_search(conn, "BM25 scoring", "test-proj", top_k=5)
        _assert(len(results) > 0, f"FTS5 search works: {len(results)} results")
    finally:
        conn.close()
        os.unlink(db_path)


# ── Test 3: dmesg ERR 聚合检测 ──
def test_dmesg_err_detection():
    print("\nTest 3: dmesg ERR 聚合检测")
    conn, db_path = _make_db()
    try:
        for i in range(15):
            dmesg_log(conn, DMESG_ERR, "test", f"test error {i}")
        conn.commit()

        result = watchdog_check(conn)
        err_check = [c for c in result["checks"] if c["name"] == "dmesg_errors"]
        _assert(len(err_check) == 1, "dmesg_errors check exists")
        _assert(err_check[0]["status"] == "elevated", f"status={err_check[0]['status']} should be elevated")
    finally:
        conn.close()
        os.unlink(db_path)


# ── Test 4: sysctl 验证 ──
def test_sysctl_validation():
    print("\nTest 4: sysctl 参数验证")
    conn, db_path = _make_db()
    try:
        result = watchdog_check(conn)
        sysctl_check = [c for c in result["checks"] if c["name"] == "sysctl_valid"]
        _assert(len(sysctl_check) == 1, "sysctl_valid check exists")
        _assert(sysctl_check[0]["status"] == "ok", f"sysctl status={sysctl_check[0]['status']} should be ok")
    finally:
        conn.close()
        os.unlink(db_path)


# ── Test 5: 数据库完整性检查 ──
def test_db_integrity():
    print("\nTest 5: 数据库完整性检查")
    conn, db_path = _make_db()
    try:
        _insert_test_chunk(conn, summary="integrity test chunk for watchdog")
        result = watchdog_check(conn)
        integrity_check = [c for c in result["checks"] if c["name"] == "db_integrity"]
        _assert(len(integrity_check) == 1, "db_integrity check exists")
        _assert(integrity_check[0]["status"] == "ok", f"integrity={integrity_check[0]['status']} should be ok")
    finally:
        conn.close()
        os.unlink(db_path)


# ── Test 6: swap 膨胀检测与自愈 ──
def test_swap_bloat_repair():
    print("\nTest 6: swap 膨胀检测与裁剪")
    conn, db_path = _make_db()
    try:
        max_swap = _sysctl("swap.max_chunks")
        now = datetime.now(timezone.utc).isoformat()
        for i in range(max_swap + 20):
            conn.execute(
                "INSERT INTO swap_chunks (id, swapped_at, project, chunk_type, original_importance, compressed_data) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (f"swap-{i}", now, "test-proj", "decision", 0.5, "compressed_placeholder")
            )
        conn.commit()

        swap_before = conn.execute("SELECT COUNT(*) FROM swap_chunks").fetchone()[0]
        _assert(swap_before > max_swap, f"swap_before={swap_before} > max={max_swap}")

        result = watchdog_check(conn)
        trim_repairs = [r for r in result["repairs"] if r["action"] == "swap_trim"]
        _assert(len(trim_repairs) > 0, f"swap_trim repair found: {trim_repairs}")

        swap_after = conn.execute("SELECT COUNT(*) FROM swap_chunks").fetchone()[0]
        _assert(swap_after <= max_swap, f"swap_after={swap_after} <= max={max_swap}")
    finally:
        conn.close()
        os.unlink(db_path)


# ── Test 7: 空数据库不崩溃 ──
def test_empty_db():
    print("\nTest 7: 空数据库 watchdog 不崩溃")
    conn, db_path = _make_db()
    try:
        result = watchdog_check(conn)
        _assert(result["status"] in ("HEALTHY", "REPAIRED"), f"status={result['status']}")
        _assert(result["duration_ms"] >= 0, f"duration={result['duration_ms']} >= 0")
    finally:
        conn.close()
        os.unlink(db_path)


# ── Test 8: watchdog 性能 < 50ms ──
def test_performance():
    print("\nTest 8: watchdog 性能 < 50ms")
    conn, db_path = _make_db()
    try:
        for i in range(50):
            _insert_test_chunk(conn, summary=f"perf test chunk {i} about various topics")
        conn.commit()

        durations = []
        for _ in range(5):
            result = watchdog_check(conn)
            durations.append(result["duration_ms"])

        avg_ms = sum(durations) / len(durations)
        _assert(avg_ms < 50, f"avg={avg_ms:.2f}ms should be < 50ms")
    finally:
        conn.close()
        os.unlink(db_path)


# ── Test 9: 正常 ERR 数量不触发告警 ──
def test_low_err_no_alert():
    print("\nTest 9: 低 ERR 数量不触发告警")
    conn, db_path = _make_db()
    try:
        for i in range(3):
            dmesg_log(conn, DMESG_ERR, "test", f"minor error {i}")
        conn.commit()

        result = watchdog_check(conn)
        err_check = [c for c in result["checks"] if c["name"] == "dmesg_errors"]
        _assert(len(err_check) == 1, "dmesg_errors check exists")
        _assert(err_check[0]["status"] == "ok", f"status={err_check[0]['status']} should be ok (below threshold)")
    finally:
        conn.close()
        os.unlink(db_path)


# ── Test 10: FTS5 一致时不 rebuild ──
def test_fts5_consistent_no_rebuild():
    print("\nTest 10: FTS5 一致时不触发 rebuild")
    conn, db_path = _make_db()
    try:
        for i in range(5):
            _insert_test_chunk(conn, summary=f"consistent fts5 chunk {i}")
        conn.commit()

        result = watchdog_check(conn)
        rebuild_repairs = [r for r in result["repairs"] if r["action"] == "fts5_rebuild"]
        _assert(len(rebuild_repairs) == 0, "no unnecessary fts5_rebuild")

        fts_check = [c for c in result["checks"] if c["name"] == "fts5_consistency"]
        _assert(len(fts_check) == 1 and fts_check[0]["status"] == "ok",
                f"fts5 status={fts_check[0]['status'] if fts_check else 'missing'}")
    finally:
        conn.close()
        os.unlink(db_path)


if __name__ == "__main__":
    test_healthy_system()
    test_fts5_integrity()
    test_dmesg_err_detection()
    test_sysctl_validation()
    test_db_integrity()
    test_swap_bloat_repair()
    test_empty_db()
    test_performance()
    test_low_err_no_alert()
    test_fts5_consistent_no_rebuild()

    print(f"\n{'='*50}")
    print(f"Total: {_PASS + _FAIL}  ✅ {_PASS}  ❌ {_FAIL}")
    sys.exit(0 if _FAIL == 0 else 1)
