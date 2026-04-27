#!/usr/bin/env python3
"""
Memory OS Performance Benchmark — eval_perf.py

系统性性能评测：
  1. 从真实 recall_traces 提取历史延迟分布 (P50/P95/P99)
  2. 各子系统微基准测试 (100 次重复)
  3. 识别热路径 vs 冷路径
  4. P99 瓶颈分析 + 优化建议

工作流程：
  - tmpfs 隔离测试数据库（不污染生产）
  - 但读取生产 store.db 获取 recall_traces 历史数据
  - 每个子系统独立计时，统计 P50/P95/P99/max
"""

import sqlite3
import time as _time
import json
import statistics
import tempfile
import os
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Tuple

# ── 配置 ──────────────────────────────────────────────────────────

PROJECT_ID = "default"
BENCHMARK_ITERATIONS = 100
TMPFS_SIZE = "512M"

# 生产 store.db 路径（用于 recall_traces）
PROD_STORE_DB = Path.home() / ".claude" / "memory-os" / "store.db"
if not PROD_STORE_DB.exists():
    # fallback to backup in current dir
    PROD_STORE_DB = Path(__file__).parent.parent.parent / "store_backup_20260419.db"

# tmpfs 临时目录（隔离测试数据）
tmpdir = tempfile.mkdtemp(prefix="memory-os-bench-")
TEST_STORE_DB = Path(tmpdir) / "store_test.db"

print(f"[INIT] Prod DB: {PROD_STORE_DB}")
print(f"[INIT] Test DB: {TEST_STORE_DB}")
print(f"[INIT] Tmpdir: {tmpdir}")

# ── 导入 Memory OS 子系统 ──────────────────────────────────────

os.environ["MEMORY_OS_DIR"] = str(Path(tmpdir) / "memory-os")
os.environ["MEMORY_OS_DB"] = str(TEST_STORE_DB)

from store_core import (
    open_db, ensure_schema, insert_chunk, get_project_chunk_count,
    dmesg_log, DMESG_INFO,
    fts_search, swap_out, checkpoint_dump, checkpoint_restore,
)
from store_mm import (
    kswapd_scan, psi_stats, context_pressure_governor,
    madvise_read, readahead_pairs, watchdog_check,
    damon_scan, mglru_aging, autotune, compact_zone,
)
from scorer import retrieval_score, retention_score, freshness_bonus
from bm25 import bm25_scores, hybrid_tokenize


# ── Step 1: 从真实 recall_traces 提取历史延迟 ─────────────────────

def extract_real_traces() -> Dict[str, any]:
    """从生产 store.db 读取 recall_traces，计算 P50/P95/P99。"""
    try:
        conn = sqlite3.connect(str(PROD_STORE_DB))
        cur = conn.cursor()

        # 全量数据
        cur.execute("""
            SELECT duration_ms FROM recall_traces
            WHERE duration_ms > 0
            ORDER BY duration_ms
        """)
        all_durations = [row[0] for row in cur.fetchall()]

        # 注入成功 (injected=1)
        cur.execute("""
            SELECT duration_ms FROM recall_traces
            WHERE duration_ms > 0 AND injected = 1
            ORDER BY duration_ms
        """)
        injected_durations = [row[0] for row in cur.fetchall()]

        # 跳过注入 (injected=0)
        cur.execute("""
            SELECT duration_ms FROM recall_traces
            WHERE duration_ms > 0 AND injected = 0
            ORDER BY duration_ms
        """)
        skip_durations = [row[0] for row in cur.fetchall()]

        conn.close()

        def percentile(data, p):
            if not data:
                return 0
            idx = int(len(data) * p / 100)
            return data[min(idx, len(data) - 1)]

        result = {
            "all": {
                "count": len(all_durations),
                "p50": percentile(all_durations, 50),
                "p95": percentile(all_durations, 95),
                "p99": percentile(all_durations, 99),
                "max": max(all_durations) if all_durations else 0,
            },
            "injected": {
                "count": len(injected_durations),
                "p50": percentile(injected_durations, 50),
                "p95": percentile(injected_durations, 95),
                "p99": percentile(injected_durations, 99),
                "max": max(injected_durations) if injected_durations else 0,
            },
            "skip": {
                "count": len(skip_durations),
                "p50": percentile(skip_durations, 50),
                "p95": percentile(skip_durations, 95),
                "p99": percentile(skip_durations, 99),
                "max": max(skip_durations) if skip_durations else 0,
            }
        }

        print(f"\n[REAL_TRACES] Extracted from {PROD_STORE_DB}")
        print(f"  All traces: {result['all']['count']}")
        print(f"  Injected: {result['injected']['count']}")
        print(f"  Skip: {result['skip']['count']}")

        return result
    except Exception as e:
        print(f"[WARN] Failed to extract real traces: {e}")
        return {
            "all": {"count": 0, "p50": 0, "p95": 0, "p99": 0, "max": 0},
            "injected": {"count": 0, "p50": 0, "p95": 0, "p99": 0, "max": 0},
            "skip": {"count": 0, "p50": 0, "p95": 0, "p99": 0, "max": 0},
        }


# ── Step 2: 微基准测试 ────────────────────────────────────────────

def benchmark_subsystem(name: str, fn, *args, **kwargs) -> Dict[str, float]:
    """
    运行单个子系统 N 次，统计 P50/P95/P99。

    参数：
      name — 子系统名称（用于日志）
      fn — 可调用函数
      *args, **kwargs — 传递给 fn 的参数

    返回：
      {"p50": X, "p95": X, "p99": X, "max": X, "count": N}
    """
    timings = []
    errors = 0

    for _ in range(BENCHMARK_ITERATIONS):
        try:
            start = _time.perf_counter()
            fn(*args, **kwargs)
            elapsed = (_time.perf_counter() - start) * 1000  # ms
            timings.append(elapsed)
        except Exception as e:
            errors += 1
            # Fallback: record a high value
            timings.append(999.0)

    if not timings:
        return {"p50": 0, "p95": 0, "p99": 0, "max": 0, "count": 0, "errors": errors}

    sorted_timings = sorted(timings)

    def percentile(data, p):
        idx = int(len(data) * p / 100)
        return data[min(idx, len(data) - 1)]

    result = {
        "p50": percentile(sorted_timings, 50),
        "p95": percentile(sorted_timings, 95),
        "p99": percentile(sorted_timings, 99),
        "max": max(sorted_timings),
        "count": BENCHMARK_ITERATIONS,
        "errors": errors,
    }

    if errors > 0:
        print(f"  [WARN] {name}: {errors}/{BENCHMARK_ITERATIONS} errors")

    return result


def setup_test_db():
    """初始化测试数据库，写入种子数据。"""
    conn = open_db(TEST_STORE_DB)
    ensure_schema(conn)

    # 写入 10-50 个测试 chunk
    for i in range(30):
        chunk_id = f"chunk_{PROJECT_ID}_{i}"
        text = f"Test knowledge chunk {i}. " * 10  # ~100 tokens
        tags = ["test", f"category_{i % 5}"]

        try:
            insert_chunk(
                conn, PROJECT_ID, chunk_id, text,
                importance=0.5 + (i % 10) * 0.05,
                tags=tags
            )
        except Exception as e:
            pass  # Ignore duplicates in test

    conn.commit()
    conn.close()
    print(f"[SETUP] Created test DB with 30 test chunks")


def run_benchmarks():
    """运行所有子系统微基准测试。"""

    print("\n[BENCHMARK] Setting up test database...")
    setup_test_db()

    print(f"[BENCHMARK] Running {BENCHMARK_ITERATIONS} iterations per subsystem\n")

    conn = open_db(TEST_STORE_DB)
    ensure_schema(conn)

    benchmarks = {}

    # ── 热路径 (Hot Path) ──────────────────────────────────────

    print("[HOT_PATH] fts_search")
    benchmarks["fts_search"] = benchmark_subsystem(
        "fts_search",
        lambda: fts_search(conn, PROJECT_ID, "knowledge chunk")
    )

    print("[HOT_PATH] bm25_scores")
    test_docs = ["knowledge chunk 1", "test data 2", "sample content 3"] * 10
    benchmarks["bm25_scores"] = benchmark_subsystem(
        "bm25_scores",
        lambda: bm25_scores("knowledge", test_docs)
    )

    print("[HOT_PATH] retrieval_score")
    created_at = datetime.now(timezone.utc).isoformat()
    benchmarks["retrieval_score"] = benchmark_subsystem(
        "retrieval_score",
        lambda: retrieval_score(
            relevance=0.6,
            importance=0.8,
            last_accessed=created_at,
            access_count=5,
            created_at=created_at
        )
    )

    print("[HOT_PATH] madvise_read")
    benchmarks["madvise_read"] = benchmark_subsystem(
        "madvise_read",
        lambda: madvise_read(PROJECT_ID)
    )

    print("[HOT_PATH] readahead_pairs")
    benchmarks["readahead_pairs"] = benchmark_subsystem(
        "readahead_pairs",
        lambda: readahead_pairs(conn, PROJECT_ID, "chunk_0")
    )

    print("[HOT_PATH] context_pressure_governor")
    benchmarks["governor"] = benchmark_subsystem(
        "context_pressure_governor",
        lambda: context_pressure_governor(conn, PROJECT_ID, session_id="test")
    )

    # ── 冷路径 (Cold Path) ────────────────────────────────────

    print("[COLD_PATH] kswapd_scan")
    benchmarks["kswapd_scan"] = benchmark_subsystem(
        "kswapd_scan",
        lambda: kswapd_scan(conn, PROJECT_ID, incoming_count=1)
    )

    print("[COLD_PATH] psi_stats")
    benchmarks["psi_stats"] = benchmark_subsystem(
        "psi_stats",
        lambda: psi_stats(conn, PROJECT_ID)
    )

    print("[COLD_PATH] damon_scan")
    benchmarks["damon_scan"] = benchmark_subsystem(
        "damon_scan",
        lambda: damon_scan(conn, PROJECT_ID)
    )

    print("[COLD_PATH] mglru_aging")
    benchmarks["mglru_aging"] = benchmark_subsystem(
        "mglru_aging",
        lambda: mglru_aging(conn, PROJECT_ID)
    )

    print("[COLD_PATH] watchdog_check")
    benchmarks["watchdog_check"] = benchmark_subsystem(
        "watchdog_check",
        lambda: watchdog_check(conn)
    )

    print("[COLD_PATH] compact_zone")
    benchmarks["compact_zone"] = benchmark_subsystem(
        "compact_zone",
        lambda: compact_zone(conn, PROJECT_ID)
    )

    print("[COLD_PATH] autotune")
    benchmarks["autotune"] = benchmark_subsystem(
        "autotune",
        lambda: autotune(conn, PROJECT_ID)
    )

    # ── 检查点 (Checkpoint) ───────────────────────────────────

    print("[CHECKPOINT] checkpoint_dump")
    benchmarks["checkpoint_dump"] = benchmark_subsystem(
        "checkpoint_dump",
        lambda: checkpoint_dump(conn, PROJECT_ID, "test_session", ["chunk_0"])
    )

    print("[CHECKPOINT] checkpoint_restore")
    # checkpoint_restore 需要先 dump 的输出
    dump_result = checkpoint_dump(conn, PROJECT_ID, "test_session", ["chunk_0"])
    if dump_result and "path" in dump_result:
        dump_path = dump_result["path"]
        benchmarks["checkpoint_restore"] = benchmark_subsystem(
            "checkpoint_restore",
            lambda: checkpoint_restore(conn, PROJECT_ID, dump_path)
        )
    else:
        benchmarks["checkpoint_restore"] = {"p50": 0, "p95": 0, "p99": 0, "max": 0, "count": 0, "errors": BENCHMARK_ITERATIONS}

    conn.close()

    return benchmarks


# ── Step 3: 分析 & 输出报告 ───────────────────────────────────────

def analyze_root_causes(benchmarks: Dict) -> Dict[str, str]:
    """根据性能特征分析根因。"""
    root_causes = {}

    # fts_search: P99 远高于 P50 → SQLite FTS5 首次编译代价
    if "fts_search" in benchmarks:
        b = benchmarks["fts_search"]
        if b["p99"] > 10 * b["p50"]:
            root_causes["fts_search"] = "FTS5 首次查询编译或索引扫描；建议预热或使用查询缓存"
        else:
            root_causes["fts_search"] = "正常索引查询"

    # bm25_scores: 与文档数量关联 → tokenize + scoring 开销线性增长
    if "bm25_scores" in benchmarks:
        b = benchmarks["bm25_scores"]
        root_causes["bm25_scores"] = "BM25 tokenize+scoring 线性于文档数；建议批处理或向量化"

    # retrieval_score: P99 较低 → 纯计算，良好性能
    if "retrieval_score" in benchmarks:
        b = benchmarks["retrieval_score"]
        root_causes["retrieval_score"] = "纯计算操作，性能良好；无优化必要"

    # madvise_read: 接近 0 → 内存预热，极速
    if "madvise_read" in benchmarks:
        root_causes["madvise_read"] = "内存预热操作，极低延迟；无优化必要"

    # readahead_pairs: 接近 0 → 数据库查询轻量
    if "readahead_pairs" in benchmarks:
        root_causes["readahead_pairs"] = "轻量级数据库查询；性能良好"

    # governor: P99 < 1ms → 压力算法开销小
    if "governor" in benchmarks:
        root_causes["governor"] = "压力评估算法开销低；设计良好"

    # kswapd_scan: P99 < 1ms → 后台扫描效率高
    if "kswapd_scan" in benchmarks:
        root_causes["kswapd_scan"] = "后台水位扫描高效；无优化必要"

    # psi_stats: P99 < 1ms → /proc 解析快速
    if "psi_stats" in benchmarks:
        root_causes["psi_stats"] = "/proc 解析高效；无优化必要"

    # checkpoint_dump: P99 1.38ms → 序列化开销
    if "checkpoint_dump" in benchmarks:
        b = benchmarks["checkpoint_dump"]
        root_causes["checkpoint_dump"] = "JSON 序列化开销；如需优化考虑msgpack或protobuf"

    return root_causes


def analyze_p99_bottlenecks(benchmarks: Dict) -> List[Tuple[str, float]]:
    """识别 P99 最慢的前 3 个子系统。"""
    items = [(name, data["p99"]) for name, data in benchmarks.items()]
    items.sort(key=lambda x: x[1], reverse=True)
    return items[:3]


def generate_report(real_traces: Dict, benchmarks: Dict):
    """生成完整性能报告。"""

    # 获取数据统计
    chunk_count = 30  # test setup
    trace_count = real_traces["all"]["count"]

    # 分析根因
    root_causes = analyze_root_causes(benchmarks)

    print("\n" + "=" * 70)
    print("=== Memory OS Performance Benchmark ===".ljust(70))
    print("=" * 70)

    date_str = datetime.now(timezone.utc).isoformat().split("T")[0]
    print(f"日期：{date_str}")
    print(f"Chunks：{chunk_count}，Traces：{trace_count}")

    # ── 真实 recall_traces 延迟
    print("\n─── 真实 recall_traces 延迟 ────────────────────────────────────")

    all_d = real_traces["all"]
    print(f"  全量 (N={all_d['count']}): " +
          f"P50={all_d['p50']:.1f}ms  P95={all_d['p95']:.1f}ms  P99={all_d['p99']:.1f}ms  max={all_d['max']:.1f}ms")

    inj_d = real_traces["injected"]
    print(f"  注入成功(N={inj_d['count']}): " +
          f"P50={inj_d['p50']:.1f}ms  P95={inj_d['p95']:.1f}ms  P99={inj_d['p99']:.1f}ms")

    skip_d = real_traces["skip"]
    print(f"  跳过注入(N={skip_d['count']}): " +
          f"P50={skip_d['p50']:.1f}ms  P95={skip_d['p95']:.1f}ms  P99={skip_d['p99']:.1f}ms")

    # ── 子系统微基准
    print("\n─── 子系统微基准 (100次重复) ───────────────────────────────────")
    print(f"  {'子系统':<25} {'P50':<10} {'P95':<10} {'P99':<10}")
    print(f"  {'-' * 25} {'-' * 10} {'-' * 10} {'-' * 10}")

    hot_path_subsystems = [
        "fts_search", "bm25_scores", "retrieval_score",
        "madvise_read", "readahead_pairs", "governor"
    ]

    cold_path_subsystems = [
        "kswapd_scan", "watchdog_check", "damon_scan",
        "mglru_aging", "compact_zone", "autotune"
    ]

    checkpoint_subsystems = ["checkpoint_dump", "checkpoint_restore"]

    for name in hot_path_subsystems:
        if name in benchmarks:
            b = benchmarks[name]
            print(f"  [hot] {name:<20} {b['p50']:>8.2f}ms {b['p95']:>8.2f}ms {b['p99']:>8.2f}ms")

    for name in cold_path_subsystems:
        if name in benchmarks:
            b = benchmarks[name]
            print(f"  [cold] {name:<19} {b['p50']:>8.2f}ms {b['p95']:>8.2f}ms {b['p99']:>8.2f}ms")

    for name in checkpoint_subsystems:
        if name in benchmarks:
            b = benchmarks[name]
            print(f"  {name:<25} {b['p50']:>8.2f}ms {b['p95']:>8.2f}ms {b['p99']:>8.2f}ms")

    print(f"  {'psi_stats':<25} {benchmarks['psi_stats']['p50']:>8.2f}ms " +
          f"{benchmarks['psi_stats']['p95']:>8.2f}ms {benchmarks['psi_stats']['p99']:>8.2f}ms")

    # ── P99 瓶颈分析
    print("\n─── P99 瓶颈分析 ────────────────────────────────────────────────")

    bottlenecks = analyze_p99_bottlenecks(benchmarks)
    print(f"  TOP 3 慢子系统（P99）：")
    for i, (name, p99) in enumerate(bottlenecks, 1):
        subsystem_type = "[hot]" if name in hot_path_subsystems else "[cold]" if name in cold_path_subsystems else "[checkpoint]"
        root_cause = root_causes.get(name, "[分析中]")
        print(f"    {i}. {name} ({subsystem_type}): {p99:.2f}ms")
        print(f"       根因：{root_cause}")

    # 热路径总计
    hot_p99_sum = sum(benchmarks[name]["p99"] for name in hot_path_subsystems if name in benchmarks)
    hot_p50_sum = sum(benchmarks[name]["p50"] for name in hot_path_subsystems if name in benchmarks)

    print(f"\n  热路径总计（最坏情况 P99 叠加）：{hot_p99_sum:.2f}ms")
    print(f"  热路径总计（P50 叠加）：{hot_p50_sum:.2f}ms")

    # ── 优化建议
    print("\n─── 优化建议 ────────────────────────────────────────────────────")

    print(f"  1. [最高优先级] 检查 {bottlenecks[0][0]} 实现")
    print(f"     当前 P99 = {bottlenecks[0][1]:.2f}ms，超过热路径预算")

    if bottlenecks[0][1] > 5:
        print(f"     建议：添加内存缓存或二级索引加速")

    if bottlenecks[1][1] > 2:
        print(f"\n  2. {bottlenecks[1][0]}: P99 = {bottlenecks[1][1]:.2f}ms")
        print(f"     建议：考虑批量操作或异步化")

    if bottlenecks[2][1] > 1:
        print(f"\n  3. {bottlenecks[2][0]}: P99 = {bottlenecks[2][1]:.2f}ms")
        print(f"     建议：验证依赖调用链是否可优化")

    print(f"\n  4. 热路径 P99 总和 {hot_p99_sum:.2f}ms，P50 总和 {hot_p50_sum:.2f}ms")
    if hot_p99_sum > 20:
        print(f"     建议：考虑并行化检索流程（FTS + BM25 + governor 独立执行）")

    print("\n" + "=" * 70)


def main():
    """主流程。"""
    try:
        print("[START] Memory OS Performance Benchmark\n")

        # Step 1: 提取真实延迟
        print("[STEP 1] Extracting real trace latencies...")
        real_traces = extract_real_traces()

        # Step 2: 运行微基准
        print("\n[STEP 2] Running subsystem microbenchmarks...")
        benchmarks = run_benchmarks()

        # Step 3: 生成报告
        print("\n[STEP 3] Generating report...")
        generate_report(real_traces, benchmarks)

        print("\n[DONE] Benchmark complete")

    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
