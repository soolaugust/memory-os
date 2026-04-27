#!/usr/bin/env python3
"""
迭代65：Small Pool Bypass — 小库 AIMD 旁路测试
OS 类比：TCP Nagle's Algorithm (RFC 896, 1984)
  小数据量时拥塞控制开销 > 收益，直接全速传输。
"""
import sys
import os
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone, timedelta

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

import tmpfs  # noqa: F401 — tmpfs isolation, must precede store import
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


def _setup_db(n_chunks=10, project="test_small"):
    """创建独立 DB 并插入 n_chunks 个 chunk。"""
    import tempfile
    tmp = tempfile.mkdtemp(prefix="test_sp_")
    os.environ["MEMORY_OS_DIR"] = tmp
    os.environ["MEMORY_OS_DB"] = os.path.join(tmp, "test.db")
    store.MEMORY_OS_DIR = Path(tmp)
    store.STORE_DB = Path(tmp) / "test.db"
    store._AIMD_STATE_FILE = Path(tmp) / "aimd_state.json"
    store._CHUNK_VERSION_FILE = Path(tmp) / ".chunk_version"
    conn = store.open_db()
    store.ensure_schema(conn)
    now = datetime.now(timezone.utc)
    for i in range(n_chunks):
        cid = f"chunk_{project}_{i}"
        ts = (now - timedelta(hours=i)).isoformat()
        conn.execute(
            "INSERT OR REPLACE INTO memory_chunks "
            "(id, created_at, updated_at, project, source_session, chunk_type, "
            "content, summary, tags, importance, retrievability, last_accessed, access_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (cid, ts, ts, project, "test_session", "decision",
             f"content_{i}", f"test decision {i}",
             json.dumps(["decision", project]),
             0.8, 0.3, ts, 0)
        )
    conn.commit()
    return conn


# ── Test 1: Small pool → bypass (cwnd=max, policy=full) ──────
def test_small_pool_bypass():
    print("\n[Test 1] Small pool bypass: chunk_count < quota * small_pool_pct → full")
    conn = _setup_db(n_chunks=10, project="proj_small")
    result = store.aimd_window(conn, "proj_small")
    _assert(result["direction"] == "bypass_small_pool",
            f"direction=bypass_small_pool (got: {result['direction']})")
    _assert(result["cwnd"] == config.get("aimd.cwnd_max"),
            f"cwnd={config.get('aimd.cwnd_max')} (got: {result['cwnd']})")
    _assert(result["policy"] == "full",
            f"policy=full (got: {result['policy']})")
    _assert(result["stats"].get("bypass") is True,
            f"stats.bypass=True (got: {result['stats'].get('bypass')})")
    _assert(result["stats"].get("chunk_count") == 10,
            f"stats.chunk_count=10 (got: {result['stats'].get('chunk_count')})")
    conn.close()


# ── Test 2: Large pool → normal AIMD (no bypass) ─────────────
def test_large_pool_no_bypass():
    print("\n[Test 2] Large pool: chunk_count >= quota * small_pool_pct → normal AIMD")
    # balloon 单项目 quota=500, small_pool_pct=0.4 → threshold=200
    # 插入 210 chunks (> 200) → 不 bypass
    conn = _setup_db(n_chunks=210, project="proj_large")
    result = store.aimd_window(conn, "proj_large")
    _assert(result["direction"] != "bypass_small_pool",
            f"direction != bypass_small_pool (got: {result['direction']})")
    conn.close()


# ── Test 3: Threshold edge case (exactly at boundary) ────────
def test_threshold_edge():
    print("\n[Test 3] Threshold edge: chunk_count == threshold → NO bypass (>= not <)")
    # balloon 单项目 quota=500, threshold = 500 * 0.4 = 200
    conn = _setup_db(n_chunks=200, project="proj_edge")
    result = store.aimd_window(conn, "proj_edge")
    _assert(result["direction"] != "bypass_small_pool",
            f"at boundary → no bypass (got: {result['direction']})")
    conn.close()


# ── Test 4: Below threshold by 1 → bypass ────────────────────
def test_threshold_below():
    print("\n[Test 4] Threshold -1: chunk_count == threshold-1 → bypass")
    # threshold = 500 * 0.4 = 200, so 199 < 200 → bypass
    conn = _setup_db(n_chunks=199, project="proj_below")
    result = store.aimd_window(conn, "proj_below")
    _assert(result["direction"] == "bypass_small_pool",
            f"below boundary → bypass (got: {result['direction']})")
    conn.close()


# ── Test 5: Empty project → bypass ───────────────────────────
def test_empty_project():
    print("\n[Test 5] Empty project (0 chunks) → bypass")
    conn = store.open_db()
    store.ensure_schema(conn)
    result = store.aimd_window(conn, "proj_empty")
    _assert(result["direction"] == "bypass_small_pool",
            f"empty → bypass (got: {result['direction']})")
    conn.close()


# ── Test 6: Bypass ignores persisted low cwnd ────────────────
def test_bypass_ignores_persisted_cwnd():
    print("\n[Test 6] Bypass ignores persisted low cwnd")
    conn = _setup_db(n_chunks=10, project="proj_low_cwnd")
    # 人工写入一个很低的 cwnd
    state_file = store.MEMORY_OS_DIR / "aimd_state.json"
    state_data = {}
    if state_file.exists():
        try:
            state_data = json.loads(state_file.read_text())
        except Exception:
            pass
    state_data["proj_low_cwnd"] = {"cwnd": 0.3, "hit_rate": 0.1,
                                    "direction": "decrease",
                                    "updated_at": datetime.now(timezone.utc).isoformat()}
    state_file.write_text(json.dumps(state_data))

    result = store.aimd_window(conn, "proj_low_cwnd")
    _assert(result["cwnd"] == config.get("aimd.cwnd_max"),
            f"bypass overrides low cwnd: cwnd={result['cwnd']}")
    _assert(result["policy"] == "full",
            f"policy=full despite persisted cwnd=0.3 (got: {result['policy']})")
    conn.close()


# ── Test 7: sysctl override small_pool_pct ────────────────────
def test_sysctl_override():
    print("\n[Test 7] sysctl override small_pool_pct=0.01 → 50 chunks no longer bypass")
    # 默认 threshold = 200*0.4=80, 50 < 80 → bypass
    # 设置 pct=0.01 → threshold=2, 50 > 2 → no bypass
    conn = _setup_db(n_chunks=50, project="proj_sysctl")

    # 先验证默认情况下 bypass
    result1 = store.aimd_window(conn, "proj_sysctl")
    _assert(result1["direction"] == "bypass_small_pool",
            f"default pct → bypass (got: {result1['direction']})")

    # 用环境变量覆盖
    os.environ["MEMORY_OS_AIMD_SMALL_POOL_PCT"] = "0.01"
    try:
        # 需要清缓存
        config._invalidate_cache()
        result2 = store.aimd_window(conn, "proj_sysctl")
        _assert(result2["direction"] != "bypass_small_pool",
                f"pct=0.01 → no bypass (got: {result2['direction']})")
    finally:
        del os.environ["MEMORY_OS_AIMD_SMALL_POOL_PCT"]
        config._invalidate_cache()
    conn.close()


# ── Test 8: Performance — bypass path is fast ─────────────────
def test_bypass_performance():
    print("\n[Test 8] Performance: bypass path < 5ms")
    import time
    conn = _setup_db(n_chunks=10, project="proj_perf")
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        store.aimd_window(conn, "proj_perf")
        times.append((time.perf_counter() - t0) * 1000)
    avg_ms = sum(times) / len(times)
    p95_ms = sorted(times)[94]
    _assert(avg_ms < 5.0,
            f"avg={avg_ms:.2f}ms < 5ms (p95={p95_ms:.2f}ms)")
    conn.close()


# ── Test 9: Extractor integration — bypass → full extraction ──
def test_extractor_policy():
    print("\n[Test 9] Extractor integration: bypass → aimd_policy='full'")
    conn = _setup_db(n_chunks=5, project="proj_ext")
    result = store.aimd_window(conn, "proj_ext")
    # In extractor.py, aimd_policy is set from result["policy"]
    # Verify the contract
    _assert(result["policy"] in ("full", "moderate", "conservative"),
            f"policy is valid enum (got: {result['policy']})")
    _assert(result["policy"] == "full",
            f"small pool → full policy for extractor (got: {result['policy']})")
    conn.close()


# ── Test 10: Config tunable registered properly ──────────────
def test_config_tunable():
    print("\n[Test 10] Config tunable aimd.small_pool_pct registered")
    val = config.get("aimd.small_pool_pct")
    _assert(val == 0.4, f"default=0.4 (got: {val})")
    _assert(isinstance(val, float), f"type=float (got: {type(val).__name__})")
    all_tunables = config.sysctl_list()
    _assert("aimd.small_pool_pct" in all_tunables,
            "aimd.small_pool_pct in sysctl_list()")


# ── Run ──────────────────────────────────────────────────────
if __name__ == "__main__":
    test_small_pool_bypass()
    test_large_pool_no_bypass()
    test_threshold_edge()
    test_threshold_below()
    test_empty_project()
    test_bypass_ignores_persisted_cwnd()
    test_sysctl_override()
    test_bypass_performance()
    test_extractor_policy()
    test_config_tunable()

    print(f"\n{'='*60}")
    print(f"Small Pool Bypass: {_pass_count}/{_test_count} passed")
    if _pass_count == _test_count:
        print("ALL TESTS PASSED ✓")
    else:
        print(f"FAILURES: {_test_count - _pass_count}")
        sys.exit(1)
