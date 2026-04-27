#!/usr/bin/env python3
"""
迭代50：TCP AIMD — Adaptive Extraction Window 测试
OS 类比：TCP Congestion Control AIMD (Jacobson/Karels, 1988)
"""
import sys
import os
import json
import sqlite3
import tempfile
from pathlib import Path
from datetime import datetime, timezone, timedelta

# 将 memory-os 根目录加入 path
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

import tmpfs  # noqa: F401 — tmpfs isolation (iter54), must precede store import
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
    """创建临时 DB 并初始化 schema。"""
    db_path = Path(tmp_dir) / "test_aimd.db"
    conn = store.open_db(db_path)
    store.ensure_schema(conn)
    return conn


def _insert_chunks(conn, project, n, chunk_type="decision", base_time=None):
    """批量插入测试 chunk。"""
    if base_time is None:
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


def _insert_traces(conn, project, chunk_ids_per_trace, n_traces):
    """插入 recall_traces，模拟 retriever 命中。"""
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
            (f"trace_{project}_{i}", ts, "test_session", project,
             f"hash_{i}", 10, json.dumps(top_k), 1, "test", 1.5)
        )
    conn.commit()


# ── 测试 ─────────────────────────────────────────────────────

def test_aimd_stats_no_data():
    """T1: 无历史数据时 aimd_stats 返回零值。"""
    print("\nT1: aimd_stats — 无历史数据")
    with tempfile.TemporaryDirectory() as tmp:
        conn = _setup_db(tmp)
        stats = store.aimd_stats(conn, "proj_empty")
        _assert(stats["recent_written"] == 0, "recent_written=0")
        _assert(stats["recent_hit"] == 0, "recent_hit=0")
        _assert(stats["hit_rate"] == 0.0, "hit_rate=0.0")
        _assert(stats["sample_traces"] == 0, "sample_traces=0")
        conn.close()


def test_aimd_stats_with_data():
    """T2: 有数据时计算正确的命中率。"""
    print("\nT2: aimd_stats — 有命中数据")
    with tempfile.TemporaryDirectory() as tmp:
        conn = _setup_db(tmp)
        project = "proj_t2"
        # 写入 10 个 chunk
        ids = _insert_chunks(conn, project, 10)
        # 只有前 3 个被 recall_traces 命中
        _insert_traces(conn, project, ids[:3], 5)
        stats = store.aimd_stats(conn, project)
        _assert(stats["recent_written"] >= 10, f"recent_written={stats['recent_written']}>=10")
        _assert(stats["recent_hit"] == 3, f"recent_hit=3 (actual={stats['recent_hit']})")
        _assert(0 < stats["hit_rate"] <= 0.5, f"hit_rate={stats['hit_rate']} in (0, 0.5]")
        _assert(stats["sample_traces"] == 5, f"sample_traces=5")
        conn.close()


def test_aimd_stats_high_hit_rate():
    """T3: 所有 chunk 都被命中 → 高命中率。"""
    print("\nT3: aimd_stats — 高命中率")
    with tempfile.TemporaryDirectory() as tmp:
        conn = _setup_db(tmp)
        project = "proj_t3"
        ids = _insert_chunks(conn, project, 5)
        _insert_traces(conn, project, ids, 10)  # 所有 chunk 都被命中
        stats = store.aimd_stats(conn, project)
        _assert(stats["hit_rate"] >= 0.5, f"hit_rate={stats['hit_rate']}>=0.5 (high)")
        conn.close()


def test_aimd_window_init():
    """T4: 数据不足时返回初始窗口。"""
    print("\nT4: aimd_window — 初始窗口（数据不足）")
    with tempfile.TemporaryDirectory() as tmp:
        conn = _setup_db(tmp)
        project = "proj_t4"
        result = store.aimd_window(conn, project)
        _assert(result["direction"] == "init", "direction=init")
        _assert(result["cwnd"] == config.get("aimd.cwnd_init"), f"cwnd={result['cwnd']}=init")
        _assert(result["policy"] in ("full", "moderate", "conservative"), f"policy={result['policy']}")
        conn.close()


def test_aimd_window_increase():
    """T5: 高命中率 → cwnd 增大（Additive Increase）。"""
    print("\nT5: aimd_window — Additive Increase")
    with tempfile.TemporaryDirectory() as tmp:
        conn = _setup_db(tmp)
        project = "proj_t5"
        # 写入足够多 chunk 和高命中 trace
        ids = _insert_chunks(conn, project, 20)
        _insert_traces(conn, project, ids[:15], 10)  # 15/20 = 75% 命中率

        # 先设置一个较低的初始 cwnd
        store._aimd_save_cwnd(project, 0.6, 0.0, "init")
        result = store.aimd_window(conn, project)

        _assert(result["direction"] == "increase", f"direction={result['direction']}=increase")
        _assert(result["cwnd"] > 0.6, f"cwnd={result['cwnd']}>0.6 (increased)")
        conn.close()


def test_aimd_window_decrease():
    """T6: 低命中率 → cwnd 减小（Multiplicative Decrease）。"""
    print("\nT6: aimd_window — Multiplicative Decrease")
    with tempfile.TemporaryDirectory() as tmp:
        conn = _setup_db(tmp)
        project = "proj_t6"
        # 写入 30 chunk 但只有 1 个被命中 → 很低命中率
        ids = _insert_chunks(conn, project, 30)
        _insert_traces(conn, project, ids[:1], 5)  # 1/30 = 3.3% 命中率

        # 先设置一个较高的初始 cwnd
        store._aimd_save_cwnd(project, 0.9, 0.0, "init")
        result = store.aimd_window(conn, project)

        _assert(result["direction"] == "decrease", f"direction={result['direction']}=decrease")
        _assert(result["cwnd"] < 0.9, f"cwnd={result['cwnd']}<0.9 (decreased)")
        conn.close()


def test_aimd_cwnd_clamp():
    """T7: cwnd 不超出 [min, max] 范围。"""
    print("\nT7: aimd_window — cwnd clamp")
    with tempfile.TemporaryDirectory() as tmp:
        conn = _setup_db(tmp)
        project = "proj_t7"
        ids = _insert_chunks(conn, project, 20)

        # 测试上限 clamp：高命中率 + 已接近上限
        _insert_traces(conn, project, ids[:18], 10)
        store._aimd_save_cwnd(project, 0.99, 0.0, "init")
        result = store.aimd_window(conn, project)
        _assert(result["cwnd"] <= config.get("aimd.cwnd_max"),
                f"cwnd={result['cwnd']}<=max({config.get('aimd.cwnd_max')})")

        # 测试下限 clamp：低命中率 + 已接近下限
        project2 = "proj_t7b"
        ids2 = _insert_chunks(conn, project2, 30)
        _insert_traces(conn, project2, ids2[:1], 5)
        store._aimd_save_cwnd(project2, 0.31, 0.0, "init")
        result2 = store.aimd_window(conn, project2)
        _assert(result2["cwnd"] >= config.get("aimd.cwnd_min"),
                f"cwnd={result2['cwnd']}>=min({config.get('aimd.cwnd_min')})")
        conn.close()


def test_cwnd_to_policy():
    """T8: cwnd 到 policy 映射正确。"""
    print("\nT8: _cwnd_to_policy 映射")
    _assert(store._cwnd_to_policy(0.9) == "full", "0.9 → full")
    _assert(store._cwnd_to_policy(0.7) == "full", "0.7 → full")
    _assert(store._cwnd_to_policy(0.6) == "moderate", "0.6 → moderate")
    _assert(store._cwnd_to_policy(0.5) == "moderate", "0.5 → moderate")
    _assert(store._cwnd_to_policy(0.4) == "conservative", "0.4 → conservative")
    _assert(store._cwnd_to_policy(0.3) == "conservative", "0.3 → conservative")


def test_aimd_persistence():
    """T9: AIMD state 持久化和恢复。"""
    print("\nT9: AIMD state 持久化")
    original_file = store._AIMD_STATE_FILE
    original_dir = store.MEMORY_OS_DIR
    try:
        with tempfile.TemporaryDirectory() as tmp:
            store._AIMD_STATE_FILE = Path(tmp) / "aimd_test.json"
            store.MEMORY_OS_DIR = Path(tmp)

            store._aimd_save_cwnd("proj_a", 0.85, 0.45, "increase")
            store._aimd_save_cwnd("proj_b", 0.42, 0.15, "decrease")

            cwnd_a = store._aimd_load_cwnd("proj_a", 0.7)
            cwnd_b = store._aimd_load_cwnd("proj_b", 0.7)
            cwnd_missing = store._aimd_load_cwnd("proj_c", 0.7)

            _assert(cwnd_a == 0.85, f"proj_a cwnd=0.85 (actual={cwnd_a})")
            _assert(cwnd_b == 0.42, f"proj_b cwnd=0.42 (actual={cwnd_b})")
            _assert(cwnd_missing == 0.7, f"proj_c default=0.7 (actual={cwnd_missing})")
    finally:
        store._AIMD_STATE_FILE = original_file
        store.MEMORY_OS_DIR = original_dir


def test_aimd_config_tunables():
    """T10: AIMD sysctl tunables 注册正确。"""
    print("\nT10: AIMD sysctl tunables")
    _assert(config.get("aimd.window_traces") == 30, "window_traces=30")
    _assert(config.get("aimd.cwnd_max") == 1.0, "cwnd_max=1.0")
    _assert(config.get("aimd.cwnd_min") == 0.3, "cwnd_min=0.3")
    _assert(config.get("aimd.cwnd_init") == 0.7, "cwnd_init=0.7")
    _assert(config.get("aimd.hit_rate_target") == 0.3, "hit_rate_target=0.3")
    _assert(config.get("aimd.additive_increase") == 0.05, "additive_increase=0.05")
    _assert(config.get("aimd.multiplicative_decrease") == 0.5, "multiplicative_decrease=0.5")


def test_proc_stats_aimd():
    """T11: proc_stats 包含 AIMD 数据。"""
    print("\nT11: proc_stats 包含 AIMD")
    original_file = store._AIMD_STATE_FILE
    original_dir = store.MEMORY_OS_DIR
    try:
        with tempfile.TemporaryDirectory() as tmp:
            store._AIMD_STATE_FILE = Path(tmp) / "aimd_test.json"
            store.MEMORY_OS_DIR = Path(tmp)

            store._aimd_save_cwnd("proj_x", 0.75, 0.40, "increase")

            conn = _setup_db(tmp)
            stats = store.proc_stats(conn)
            _assert("aimd" in stats, "proc_stats contains 'aimd'")
            _assert("proj_x" in stats["aimd"], "aimd contains proj_x")
            _assert(stats["aimd"]["proj_x"]["cwnd"] == 0.75, f"aimd.proj_x.cwnd=0.75")
            _assert(stats["aimd"]["proj_x"]["policy"] == "full", "aimd.proj_x.policy=full")
            conn.close()
    finally:
        store._AIMD_STATE_FILE = original_file
        store.MEMORY_OS_DIR = original_dir


def test_aimd_multi_project_isolation():
    """T12: 多项目 AIMD 隔离。"""
    print("\nT12: 多项目 AIMD 隔离")
    with tempfile.TemporaryDirectory() as tmp:
        conn = _setup_db(tmp)
        p1, p2 = "proj_high", "proj_low"
        # 项目1：高命中率
        ids1 = _insert_chunks(conn, p1, 10)
        _insert_traces(conn, p1, ids1[:8], 5)
        # 项目2：低命中率
        ids2 = _insert_chunks(conn, p2, 10, chunk_type="reasoning_chain")
        _insert_traces(conn, p2, ids2[:1], 5)

        stats1 = store.aimd_stats(conn, p1)
        stats2 = store.aimd_stats(conn, p2)

        _assert(stats1["hit_rate"] > stats2["hit_rate"],
                f"proj_high hr={stats1['hit_rate']} > proj_low hr={stats2['hit_rate']}")
        conn.close()


def test_aimd_performance():
    """T13: AIMD 计算性能 < 5ms。"""
    print("\nT13: AIMD 性能")
    import time
    with tempfile.TemporaryDirectory() as tmp:
        conn = _setup_db(tmp)
        project = "proj_perf"
        ids = _insert_chunks(conn, project, 50)
        _insert_traces(conn, project, ids[:10], 20)

        times = []
        for _ in range(100):
            t0 = time.time()
            store.aimd_window(conn, project)
            times.append((time.time() - t0) * 1000)

        avg_ms = sum(times) / len(times)
        _assert(avg_ms < 5.0, f"avg={avg_ms:.3f}ms < 5ms")
        conn.close()


if __name__ == "__main__":
    print("=" * 60)
    print("迭代50：TCP AIMD — Adaptive Extraction Window 测试")
    print("=" * 60)

    test_aimd_stats_no_data()
    test_aimd_stats_with_data()
    test_aimd_stats_high_hit_rate()
    test_aimd_window_init()
    test_aimd_window_increase()
    test_aimd_window_decrease()
    test_aimd_cwnd_clamp()
    test_cwnd_to_policy()
    test_aimd_persistence()
    test_aimd_config_tunables()
    test_proc_stats_aimd()
    test_aimd_multi_project_isolation()
    test_aimd_performance()

    print(f"\n{'=' * 60}")
    print(f"结果：{_pass_count}/{_test_count} 通过")
    print(f"{'=' * 60}")
    sys.exit(0 if _pass_count == _test_count else 1)
