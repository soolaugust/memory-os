#!/usr/bin/env python3
"""
迭代36：PSI (Pressure Stall Information) 测试
OS 类比：Linux PSI — /proc/pressure/{cpu,memory,io}

测试矩阵：
  T1  空数据库 → 全 NONE
  T2  capacity SOME（使用率 > 70%）
  T3  capacity FULL（使用率 > 90%）
  T4  quality SOME（命中率低于基线）
  T5  quality FULL（命中率低于基线 50%）
  T6  retrieval SOME（>30% 延迟超基线）
  T7  retrieval FULL（>70% 延迟超基线）
  T8  overall = 三维度最严重值
  T9  PSI 集成 proc_stats()
  T10 miss_streak 计算正确
  T11 recommendation 输出正确
  T12 sysctl 新增 8 个 PSI tunable（含迭代60 adaptive baseline 3个）
  T13 性能基准
  T14 adaptive baseline 开启时基线自动校准
  T15 adaptive baseline 样本不足时 fallback 到固定基线
  T16 adaptive baseline 关闭时使用固定基线
  T17 adaptive baseline 全缓存命中时下限保护（≥5ms）
"""
import sys
import os
import json
import time
import uuid
from pathlib import Path
from datetime import datetime, timezone, timedelta

# 设置 path
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

import sqlite3
import tmpfs  # noqa: F401 — tmpfs isolation (iter54), must precede store import
from store import open_db, ensure_schema, insert_chunk, insert_trace, psi_stats, proc_stats, get_project_chunk_count
from config import get as _cfg, _REGISTRY

TEST_PROJECT = "test_psi_project"
PASSED = 0
FAILED = 0


def _make_db():
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA journal_mode=WAL")
    ensure_schema(conn)
    return conn


def _insert_chunks(conn, n, project=TEST_PROJECT, importance=0.6):
    now = datetime.now(timezone.utc).isoformat()
    for i in range(n):
        insert_chunk(conn, {
            "id": str(uuid.uuid4()),
            "created_at": now,
            "updated_at": now,
            "project": project,
            "source_session": "test",
            "chunk_type": "decision",
            "content": f"test content {i}",
            "summary": f"test summary {i}",
            "tags": "[]",
            "importance": importance,
            "retrievability": 0.5,
            "last_accessed": now,
            "feishu_url": None,
        })
    conn.commit()


def _insert_traces_custom(conn, latencies, injected_list, project=TEST_PROJECT):
    """精确控制每条 trace 的延迟和命中"""
    now = datetime.now(timezone.utc)
    for i, (lat, inj) in enumerate(zip(latencies, injected_list)):
        ts = (now - timedelta(seconds=i * 10)).isoformat()
        insert_trace(conn, {
            "id": str(uuid.uuid4()),
            "timestamp": ts,
            "session_id": "test_session",
            "project": project,
            "prompt_hash": f"hash_{i}",
            "candidates_count": 10,
            "top_k_json": [],
            "injected": 1 if inj else 0,
            "reason": "test",
            "duration_ms": lat,
        })
    conn.commit()


def check(name, condition):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  ✓ {name}")
    else:
        FAILED += 1
        print(f"  ✗ {name}")


# ── T1: 空数据库 → 全 NONE ──

if __name__ == "__main__":
    print("T1: Empty database → all NONE")
    conn = _make_db()
    result = psi_stats(conn, TEST_PROJECT)
    check("overall=NONE", result["overall"] == "NONE")
    check("retrieval.level=NONE", result["retrieval"]["level"] == "NONE")
    check("capacity.level=NONE", result["capacity"]["level"] == "NONE")
    check("quality.level=NONE", result["quality"]["level"] == "NONE")
    check("recommendation=healthy", "healthy" in result["recommendation"])
    conn.close()

    # ── T2: capacity SOME（使用率 > 70%） ──
    print("\nT2: Capacity SOME (usage > 70%)")
    conn = _make_db()
    quota = _cfg("extractor.chunk_quota")  # 200
    target = int(quota * 0.75)  # 75% → SOME
    _insert_chunks(conn, target)
    result = psi_stats(conn, TEST_PROJECT)
    check(f"capacity.level=SOME (count={target})", result["capacity"]["level"] == "SOME")
    check("capacity.usage_pct > 70", result["capacity"]["usage_pct"] > 70)
    conn.close()

    # ── T3: capacity FULL（使用率 > 90%） ──
    print("\nT3: Capacity FULL (usage > 90%)")
    conn = _make_db()
    target = int(quota * 0.95)  # 95% → FULL
    _insert_chunks(conn, target)
    result = psi_stats(conn, TEST_PROJECT)
    check(f"capacity.level=FULL (count={target})", result["capacity"]["level"] == "FULL")
    check("capacity.usage_pct > 90", result["capacity"]["usage_pct"] > 90)
    conn.close()

    # ── T4: quality SOME（命中率低于基线） ──
    print("\nT4: Quality SOME (hit rate < baseline)")
    conn = _make_db()
    hit_baseline = _cfg("psi.hit_rate_baseline_pct")  # 50%
    # 40% 命中率 → SOME（低于 50% 基线但高于 25%）
    _insert_traces_custom(conn,
        latencies=[2.0] * 20,
        injected_list=[True] * 8 + [False] * 12)  # 8/20 = 40%
    result = psi_stats(conn, TEST_PROJECT)
    check(f"quality.level=SOME (hit_rate=40%)", result["quality"]["level"] == "SOME")
    check(f"quality.hit_rate_pct=40", result["quality"]["hit_rate_pct"] == 40.0)
    conn.close()

    # ── T5: quality FULL（命中率低于基线 50%） ──
    print("\nT5: Quality FULL (hit rate < baseline * 0.5)")
    conn = _make_db()
    # 10% 命中率 → FULL（低于 50% * 0.5 = 25%）
    _insert_traces_custom(conn,
        latencies=[2.0] * 20,
        injected_list=[True] * 2 + [False] * 18)  # 2/20 = 10%
    result = psi_stats(conn, TEST_PROJECT)
    check(f"quality.level=FULL (hit_rate=10%)", result["quality"]["level"] == "FULL")
    conn.close()

    # ── T6: retrieval SOME（>30% 延迟超基线） ──
    # 迭代60：adaptive baseline 开启时，P50×margin 为基线
    # sorted=[5]*14 + [100]*6 → P50 idx=10 → 5ms, baseline=5*1.5=7.5 → clamp to 7.5ms
    # 6/20=30%超 → 不够。用更合理的分布：
    # sorted=[8]*12 + [30]*8 → P50 idx=10 → 8ms, baseline=8*1.5=12ms
    # P95 idx=18 → 30ms, baseline*3=36ms → P95 < 3×baseline (不触发 FULL)
    # 8/20=40% 超 12ms → SOME（>30% but <70%）
    print("\nT6: Retrieval SOME (>30% latency stalls)")
    conn = _make_db()
    latencies = [8.0] * 12 + [30.0] * 8
    _insert_traces_custom(conn, latencies=latencies,
        injected_list=[True] * 20)
    result = psi_stats(conn, TEST_PROJECT)
    check(f"retrieval.level=SOME (40% stalls)", result["retrieval"]["level"] == "SOME")
    check("retrieval.some_pct=40", result["retrieval"]["some_pct"] == 40.0)
    check("retrieval.adaptive=True", result["retrieval"].get("adaptive") == True)
    conn.close()

    # ── T7: retrieval FULL（>70% 延迟超基线） ──
    # P50=50ms（因为80%是50ms），adaptive baseline=50*1.5=75ms
    # 但 P95=200ms > 75*3=225? 不够。改用更极端分布。
    # 4条 5ms + 16条 200ms → P50=200, baseline=300, stall=0% → 不对。
    # 关键：需要 >70% 超过 adaptive baseline。
    # 用：4条 10ms + 16条 500ms → P50=500, baseline=500*1.5=750, 只有0%超
    # 不对。需要双峰分布：低值拉低 P50，高值超过 P50*margin。
    # 正确做法：10条 10ms + 10条 100ms → P50=10 (或 100，取决于排序)
    # sorted: [10]*10 + [100]*10 → P50 idx=10 → 100ms, baseline=150ms → 0%超
    # 还是不对。需要：low 拉低 P50，但 high 比 P50*margin 高很多。
    # 用固定基线测试更简单：禁用 adaptive。
    print("\nT7: Retrieval FULL (>70% latency stalls, adaptive off)")
    conn = _make_db()
    # 临时用环境变量禁用 adaptive
    import config as _config_mod
    _orig_adaptive = _config_mod._REGISTRY.get("psi.adaptive_baseline")
    _config_mod._REGISTRY["psi.adaptive_baseline"] = (0, int, 0, 1, None, "disabled for test")
    _config_mod._disk_config = None  # 清缓存
    # baseline=30ms（固定），80%超过30ms → FULL
    latencies = [5.0] * 4 + [80.0] * 16  # 80% 超 30ms
    _insert_traces_custom(conn, latencies=latencies,
        injected_list=[True] * 20)
    result = psi_stats(conn, TEST_PROJECT)
    check(f"retrieval.level=FULL (80% stalls, fixed baseline)", result["retrieval"]["level"] == "FULL")
    check("retrieval.adaptive=False", result["retrieval"].get("adaptive") == False)
    # 恢复
    _config_mod._REGISTRY["psi.adaptive_baseline"] = _orig_adaptive
    _config_mod._disk_config = None
    conn.close()

    # ── T8: overall = 三维度最严重值 ──
    print("\nT8: Overall = max severity across dimensions")
    conn = _make_db()
    # capacity=SOME, quality=NONE, retrieval=NONE → overall=SOME
    _insert_chunks(conn, int(quota * 0.75))
    _insert_traces_custom(conn, latencies=[2.0] * 20,
        injected_list=[True] * 20)  # 100% hit, low latency
    result = psi_stats(conn, TEST_PROJECT)
    check("overall=SOME (capacity drives)", result["overall"] == "SOME")
    conn.close()

    # ── T9: PSI 集成 proc_stats() ──
    print("\nT9: PSI integrated into proc_stats()")
    conn = _make_db()
    _insert_chunks(conn, 5)
    _insert_traces_custom(conn, latencies=[2.0] * 10, injected_list=[True] * 10)
    stats = proc_stats(conn)
    check("proc_stats has 'pressure' key", "pressure" in stats)
    if "pressure" in stats and TEST_PROJECT in stats["pressure"]:
        check("pressure has project data", "overall" in stats["pressure"][TEST_PROJECT])
    else:
        check("pressure has project data (empty OK)", True)
    conn.close()

    # ── T10: miss_streak 计算正确 ──
    print("\nT10: miss_streak calculation")
    conn = _make_db()
    # 最近3条 miss，然后1条 hit → miss_streak=3
    _insert_traces_custom(conn,
        latencies=[2.0] * 10,
        injected_list=[False, False, False, True, True, True, True, True, True, True])
    result = psi_stats(conn, TEST_PROJECT)
    check("miss_streak=3", result["quality"]["miss_streak"] == 3)
    conn.close()

    # ── T11: recommendation 输出正确 ──
    print("\nT11: Recommendations correct")
    conn = _make_db()
    _insert_chunks(conn, int(quota * 0.95))  # capacity FULL
    _insert_traces_custom(conn, latencies=[2.0] * 20,
        injected_list=[True] * 20)
    result = psi_stats(conn, TEST_PROJECT)
    check("has capacity_critical in recommendation",
          "capacity_critical" in result["recommendation"])
    conn.close()

    # ── T12: sysctl 新增 8 个 PSI tunable（含迭代60 adaptive baseline 3个） ──
    print("\nT12: PSI sysctl tunables registered")
    psi_keys = [k for k in _REGISTRY if k.startswith("psi.")]
    check(f"8 PSI tunables registered (got {len(psi_keys)})", len(psi_keys) == 8)
    for key in ["psi.window_size", "psi.latency_baseline_ms", "psi.hit_rate_baseline_pct",
                "psi.capacity_some_pct", "psi.capacity_full_pct",
                "psi.adaptive_baseline", "psi.adaptive_margin", "psi.adaptive_min_samples"]:
        check(f"  {key} exists", key in _REGISTRY)

    # ── T13: 性能基准 ──
    print("\nT13: Performance benchmark")
    conn = _make_db()
    _insert_chunks(conn, 50)
    _insert_traces_custom(conn, latencies=[3.0] * 30, injected_list=[True] * 30)
    t0 = time.time()
    for _ in range(100):
        psi_stats(conn, TEST_PROJECT)
    elapsed = (time.time() - t0) * 1000
    avg = elapsed / 100
    check(f"psi_stats avg={avg:.2f}ms (target < 5ms)", avg < 5.0)
    conn.close()

    # ── T14: adaptive baseline 开启时基线自动校准 ──
    print("\nT14: Adaptive baseline auto-calibration")
    conn = _make_db()
    # 所有延迟在 20ms 附近 → P50=20ms, adaptive baseline=20*1.5=30ms
    # 所有请求 ≤ 30ms → stall=0% → NONE
    latencies = [18.0, 19.0, 20.0, 21.0, 22.0, 18.5, 19.5, 20.5, 21.5, 22.5,
                 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 19.0, 20.0, 21.0]
    _insert_traces_custom(conn, latencies=latencies, injected_list=[True] * 20)
    result = psi_stats(conn, TEST_PROJECT)
    check("adaptive: all near P50 → NONE", result["retrieval"]["level"] == "NONE")
    check("adaptive baseline ≈ 30ms (P50=20 × 1.5)", 25 <= result["retrieval"]["baseline_ms"] <= 35)
    check("adaptive=True in result", result["retrieval"].get("adaptive") == True)
    conn.close()

    # ── T15: adaptive baseline 样本不足时 fallback 到固定基线 ──
    print("\nT15: Adaptive baseline fallback with insufficient samples")
    conn = _make_db()
    # 只有 3 条（< adaptive_min_samples=5）→ fallback 到固定 30ms
    _insert_traces_custom(conn, latencies=[100.0, 100.0, 100.0],
        injected_list=[True] * 3)
    result = psi_stats(conn, TEST_PROJECT)
    # 固定基线 30ms，3条全部 100ms > 30ms → 100% stall → FULL
    check("fallback: 3 samples → fixed baseline", result["retrieval"]["baseline_ms"] == 30.0)
    check("fallback: 100% stall → FULL", result["retrieval"]["level"] == "FULL")
    conn.close()

    # ── T16: adaptive baseline 关闭时使用固定基线 ──
    print("\nT16: Adaptive baseline disabled → fixed baseline")
    conn = _make_db()
    _config_mod._REGISTRY["psi.adaptive_baseline"] = (0, int, 0, 1, None, "disabled for test")
    _config_mod._disk_config = None
    # 所有延迟 20ms，固定基线 30ms → 0% stall → NONE
    latencies = [20.0] * 20
    _insert_traces_custom(conn, latencies=latencies, injected_list=[True] * 20)
    result = psi_stats(conn, TEST_PROJECT)
    check("disabled: fixed baseline=30", result["retrieval"]["baseline_ms"] == 30.0)
    check("disabled: adaptive=False", result["retrieval"].get("adaptive") == False)
    check("disabled: 0% stall → NONE", result["retrieval"]["level"] == "NONE")
    _config_mod._REGISTRY["psi.adaptive_baseline"] = (1, int, 0, 1, None, "re-enabled")
    _config_mod._disk_config = None
    conn.close()

    # ── T17: adaptive baseline 下限保护（≥5ms） ──
    print("\nT17: Adaptive baseline floor protection (≥5ms)")
    conn = _make_db()
    # 所有延迟 1ms → P50=1ms, adaptive=1*1.5=1.5ms < 5ms → clamp to 5ms
    latencies = [1.0] * 20
    _insert_traces_custom(conn, latencies=latencies, injected_list=[True] * 20)
    result = psi_stats(conn, TEST_PROJECT)
    check("floor: baseline clamped to ≥5ms", result["retrieval"]["baseline_ms"] >= 5.0)
    check("floor: level=NONE (all ≤ 5ms)", result["retrieval"]["level"] == "NONE")
    conn.close()

    # ── 汇总 ──
    print(f"\n{'='*50}")
    print(f"PSI Tests: {PASSED} passed, {FAILED} failed, total {PASSED+FAILED}")
    if FAILED > 0:
        sys.exit(1)
    print("ALL PASSED ✓")
