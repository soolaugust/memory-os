#!/usr/bin/env python3
"""
迭代63：TCP Slow Start + Trace GC 测试
OS 类比：TCP Slow Start (Van Jacobson, 1988) + logrotate (1996)
"""
import sys
import os
import json
import sqlite3
import tempfile
from pathlib import Path
from datetime import datetime, timezone, timedelta

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

import tmpfs  # noqa: F401 — tmpfs isolation (iter54)
import store
import config

# ── 测试工具 ─────────────────────────────────────────────────
_test_count = 0
_pass_count = 0


def _assert(condition, msg):
    global _test_count, _pass_count
    _test_count += 1
    if condition:
        _pass_count += 1
        print(f"  ✓ {msg}")
    else:
        print(f"  ✗ {msg}")


def _setup_db(tmp_dir):
    db_path = Path(tmp_dir) / "test_ss_gc.db"
    conn = store.open_db(db_path)
    store.ensure_schema(conn)
    return conn


def _insert_chunks(conn, project, n, chunk_type="decision"):
    base_time = datetime.now(timezone.utc)
    ids = []
    for i in range(n):
        cid = f"chunk_{project}_{i}_{chunk_type}"
        ts = (base_time - timedelta(hours=i)).isoformat()
        conn.execute(
            "INSERT OR REPLACE INTO memory_chunks "
            "(id, created_at, updated_at, project, source_session, chunk_type, "
            "content, summary, tags, importance, retrievability, last_accessed, access_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (cid, ts, ts, project, "test_session", chunk_type,
             f"content_{i}", f"summary_{i} decision about {chunk_type}",
             json.dumps([chunk_type, project]),
             0.8 - i * 0.01, 0.3, ts, i)
        )
        ids.append(cid)
    conn.commit()
    return ids


def _insert_traces(conn, project, chunk_ids_per_trace, n_traces, base_time=None):
    if base_time is None:
        base_time = datetime.now(timezone.utc)
    for i in range(n_traces):
        ts = (base_time - timedelta(hours=i)).isoformat()
        top_k = [{"id": cid, "summary": f"s_{cid}", "score": 0.8 - j * 0.1}
                 for j, cid in enumerate(chunk_ids_per_trace)]
        conn.execute(
            "INSERT INTO recall_traces "
            "(id, timestamp, session_id, project, prompt_hash, candidates_count, "
            "top_k_json, injected, reason, duration_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (f"trace_{project}_{i}_{base_time.timestamp():.0f}", ts, "test_session", project,
             f"hash_{i}", 10, json.dumps(top_k), 1, "test", 1.5)
        )
    conn.commit()


# ── Slow Start 测试 ─────────────────────────────────────────

def test_slow_start_exponential():
    """T1: cwnd < ssthresh 且命中率达标 → 指数增长。"""
    print("\nT1: Slow Start — 指数增长")
    with tempfile.TemporaryDirectory() as tmp:
        conn = _setup_db(tmp)
        project = "proj_ss1"
        ids = _insert_chunks(conn, project, 20)
        _insert_traces(conn, project, ids[:15], 10)  # 75% 命中率

        # cwnd=0.35 < ssthresh=0.6 → should do exponential growth
        store._aimd_save_cwnd(project, 0.35, 0.0, "init")
        result = store.aimd_window(conn, project)

        _assert(result["direction"] == "slow_start", f"direction={result['direction']}=slow_start")
        # 0.35 * 2.0 = 0.7, but capped at ssthresh=0.6
        expected = min(0.35 * config.get("aimd.slow_start_factor"), config.get("aimd.ssthresh"))
        _assert(abs(result["cwnd"] - expected) < 0.01,
                f"cwnd={result['cwnd']}≈{expected} (exponential, capped at ssthresh)")
        conn.close()


def test_slow_start_to_congestion_avoidance():
    """T2: cwnd >= ssthresh 且命中率达标 → 线性增长（Congestion Avoidance）。"""
    print("\nT2: Slow Start → Congestion Avoidance 切换")
    with tempfile.TemporaryDirectory() as tmp:
        conn = _setup_db(tmp)
        project = "proj_ss2"
        ids = _insert_chunks(conn, project, 20)
        _insert_traces(conn, project, ids[:15], 10)

        # cwnd=0.65 >= ssthresh=0.6 → should do linear increase
        store._aimd_save_cwnd(project, 0.65, 0.0, "init")
        result = store.aimd_window(conn, project)

        _assert(result["direction"] == "increase", f"direction={result['direction']}=increase")
        ai_step = config.get("aimd.additive_increase")
        expected = 0.65 + ai_step
        _assert(abs(result["cwnd"] - expected) < 0.01,
                f"cwnd={result['cwnd']}≈{expected} (linear AI)")
        conn.close()


def test_slow_start_rapid_recovery():
    """T3: Slow Start 从 0.3 恢复到 0.6 只需 2 步（vs 线性 6 步）。"""
    print("\nT3: Slow Start — 快速恢复 (0.3→0.6 in 2 steps)")
    with tempfile.TemporaryDirectory() as tmp:
        conn = _setup_db(tmp)
        project = "proj_ss3"
        ids = _insert_chunks(conn, project, 20)
        _insert_traces(conn, project, ids[:15], 10)

        # Step 1: 0.3 → 0.6 (0.3 * 2.0 = 0.6, capped at ssthresh)
        store._aimd_save_cwnd(project, 0.3, 0.0, "decrease")
        r1 = store.aimd_window(conn, project)
        _assert(r1["direction"] == "slow_start", f"step1 direction={r1['direction']}")
        _assert(r1["cwnd"] == 0.6, f"step1 cwnd={r1['cwnd']}=0.6 (ssthresh)")

        # Step 2: 0.6 → 0.65 (linear AI, cwnd >= ssthresh)
        r2 = store.aimd_window(conn, project)
        _assert(r2["direction"] == "increase", f"step2 direction={r2['direction']}")
        _assert(r2["cwnd"] > 0.6, f"step2 cwnd={r2['cwnd']}>0.6 (linear)")
        conn.close()


def test_slow_start_no_trigger_on_decrease():
    """T4: 命中率低时仍然执行 MD，不触发 Slow Start。"""
    print("\nT4: Slow Start — 低命中率不触发")
    with tempfile.TemporaryDirectory() as tmp:
        conn = _setup_db(tmp)
        project = "proj_ss4"
        base_time = datetime.now(timezone.utc)
        # 用同一个 base_time 避免时间漂移导致 recent_written 不稳定
        ids = []
        for i in range(30):
            cid = f"chunk_{project}_{i}_decision"
            ts = (base_time - timedelta(minutes=i * 10)).isoformat()
            conn.execute(
                "INSERT OR REPLACE INTO memory_chunks "
                "(id, created_at, updated_at, project, source_session, chunk_type, "
                "content, summary, tags, importance, retrievability, last_accessed, access_count) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (cid, ts, ts, project, "test_session", "decision",
                 f"content_{i}", f"summary_{i} decision",
                 json.dumps(["decision", project]),
                 0.8 - i * 0.01, 0.3, ts, i)
            )
            ids.append(cid)
        conn.commit()
        # 10 traces, 只命中 1 个 chunk → 命中率 = 1/30 = 3.3%
        _insert_traces(conn, project, ids[:1], 10, base_time=base_time)

        store._aimd_save_cwnd(project, 0.35, 0.0, "init")
        result = store.aimd_window(conn, project)

        _assert(result["direction"] == "decrease", f"direction={result['direction']}=decrease")
        _assert(result["cwnd"] < 0.35, f"cwnd={result['cwnd']}<0.35 (MD)")
        conn.close()


def test_slow_start_config_tunables():
    """T5: Slow Start sysctl tunables 注册正确。"""
    print("\nT5: Slow Start tunables")
    _assert(config.get("aimd.ssthresh") == 0.6, "ssthresh=0.6")
    _assert(config.get("aimd.slow_start_factor") == 2.0, "slow_start_factor=2.0")


# ── Trace GC 测试 ───────────────────────────────────────────

def test_gc_traces_age():
    """T6: GC 按时间淘汰 > max_age_days 的 trace。"""
    print("\nT6: GC — 时间淘汰")
    with tempfile.TemporaryDirectory() as tmp:
        conn = _setup_db(tmp)
        project = "proj_gc1"
        ids = _insert_chunks(conn, project, 5)

        # 插入 5 条新 trace + 5 条旧 trace (20天前)
        _insert_traces(conn, project, ids[:3], 5)
        old_time = datetime.now(timezone.utc) - timedelta(days=20)
        _insert_traces(conn, project, ids[:3], 5, base_time=old_time)

        total_before = conn.execute(
            "SELECT COUNT(*) FROM recall_traces WHERE project=?",
            (project,)).fetchone()[0]
        _assert(total_before == 10, f"before gc: {total_before}=10 traces")

        result = store.gc_traces(conn, project)
        _assert(result["deleted_age"] == 5, f"deleted_age={result['deleted_age']}=5")
        _assert(result["remaining"] == 5, f"remaining={result['remaining']}=5")
        conn.close()


def test_gc_traces_rows():
    """T7: GC 按容量淘汰 > max_rows 的 trace。"""
    print("\nT7: GC — 容量淘汰")
    with tempfile.TemporaryDirectory() as tmp:
        conn = _setup_db(tmp)
        project = "proj_gc2"
        ids = _insert_chunks(conn, project, 5)

        # config._coerce() clamps env var to registry range [50, 5000]
        # 设置 max_rows=50（最小合法值），插入 60 条 trace → 超出 10 条
        os.environ["MEMORY_OS_GC_TRACE_MAX_ROWS"] = "50"
        config._invalidate_cache()
        try:
            for batch in range(6):
                bt = datetime.now(timezone.utc) - timedelta(hours=batch)
                _insert_traces(conn, project, ids[:3], 10, base_time=bt)

            total_before = conn.execute(
                "SELECT COUNT(*) FROM recall_traces WHERE project=?",
                (project,)).fetchone()[0]
            _assert(total_before == 60, f"before gc: {total_before}=60 traces")

            result = store.gc_traces(conn, project)
            _assert(result["deleted_rows"] > 0, f"deleted_rows={result['deleted_rows']}>0")
            _assert(result["remaining"] <= 50, f"remaining={result['remaining']}<=50 (max_rows)")
        finally:
            os.environ.pop("MEMORY_OS_GC_TRACE_MAX_ROWS", None)
            config._invalidate_cache()
        conn.close()


def test_gc_traces_no_op():
    """T8: 无过期/溢出时 GC 无操作。"""
    print("\nT8: GC — 无操作")
    with tempfile.TemporaryDirectory() as tmp:
        conn = _setup_db(tmp)
        project = "proj_gc3"
        ids = _insert_chunks(conn, project, 5)
        _insert_traces(conn, project, ids[:3], 3)

        result = store.gc_traces(conn, project)
        _assert(result["deleted_age"] == 0, f"deleted_age=0 (no old traces)")
        _assert(result["deleted_rows"] == 0, f"deleted_rows=0 (under max)")
        _assert(result["remaining"] == 3, f"remaining=3")
        conn.close()


def test_gc_traces_all_projects():
    """T9: project=None 时清理所有项目。"""
    print("\nT9: GC — 全局清理")
    with tempfile.TemporaryDirectory() as tmp:
        conn = _setup_db(tmp)
        ids_a = _insert_chunks(conn, "proj_a", 5)
        ids_b = _insert_chunks(conn, "proj_b", 5)

        # 两个项目各 5 条旧 trace
        old_time = datetime.now(timezone.utc) - timedelta(days=20)
        _insert_traces(conn, "proj_a", ids_a[:3], 5, base_time=old_time)
        _insert_traces(conn, "proj_b", ids_b[:3], 5, base_time=old_time)
        # 各 3 条新 trace
        _insert_traces(conn, "proj_a", ids_a[:3], 3)
        _insert_traces(conn, "proj_b", ids_b[:3], 3)

        result = store.gc_traces(conn, project=None)
        _assert(result["deleted_age"] == 10, f"deleted_age={result['deleted_age']}=10 (both projects)")
        _assert(result["remaining"] == 6, f"remaining={result['remaining']}=6")
        conn.close()


def test_gc_config_tunables():
    """T10: Trace GC sysctl tunables 注册正确。"""
    print("\nT10: GC tunables")
    _assert(config.get("gc.trace_max_age_days") == 14, "max_age_days=14")
    _assert(config.get("gc.trace_max_rows") == 500, "max_rows=500")


def test_gc_performance():
    """T11: GC 性能 < 10ms。"""
    print("\nT11: GC 性能")
    import time
    with tempfile.TemporaryDirectory() as tmp:
        conn = _setup_db(tmp)
        project = "proj_perf"
        ids = _insert_chunks(conn, project, 10)
        _insert_traces(conn, project, ids[:5], 100)

        times = []
        for _ in range(50):
            t0 = time.time()
            store.gc_traces(conn, project)
            times.append((time.time() - t0) * 1000)

        avg_ms = sum(times) / len(times)
        _assert(avg_ms < 10.0, f"avg={avg_ms:.3f}ms < 10ms")
        conn.close()


if __name__ == "__main__":
    print("=" * 60)
    print("迭代63：TCP Slow Start + Trace GC 测试")
    print("=" * 60)

    test_slow_start_exponential()
    test_slow_start_to_congestion_avoidance()
    test_slow_start_rapid_recovery()
    test_slow_start_no_trigger_on_decrease()
    test_slow_start_config_tunables()
    test_gc_traces_age()
    test_gc_traces_rows()
    test_gc_traces_no_op()
    test_gc_traces_all_projects()
    test_gc_config_tunables()
    test_gc_performance()

    print(f"\n{'=' * 60}")
    print(f"结果：{_pass_count}/{_test_count} 通过")
    print(f"{'=' * 60}")
    sys.exit(0 if _pass_count == _test_count else 1)
