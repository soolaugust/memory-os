#!/usr/bin/env python3
"""
memory-os 性能报告 — 对应 OS profiler（1970s Unix gprof）

读取 recall_traces 表中的 duration_ms，输出性能分布摘要。

用法：
  python3 perf_report.py            # 最近 50 条
  python3 perf_report.py --all      # 全量
  python3 perf_report.py --n 20     # 最近 N 条
"""
import sys
import sqlite3
from pathlib import Path

DB = Path.home() / ".claude" / "memory-os" / "store.db"


def report(limit: int = 50):
    if not DB.exists():
        print("store.db 不存在")
        return

    conn = sqlite3.connect(str(DB))
    try:
        rows = conn.execute(
            """SELECT timestamp, injected, reason, duration_ms, candidates_count
               FROM recall_traces
               WHERE duration_ms > 0
               ORDER BY timestamp DESC
               LIMIT ?""",
            (limit,)
        ).fetchall()
        # B10: global injection ratio — from dmesg extra field
        import json as _json
        dmesg_rows = conn.execute(
            """SELECT extra FROM dmesg
               WHERE subsystem='retriever' AND extra IS NOT NULL
               ORDER BY id DESC LIMIT ?""",
            (limit,)
        ).fetchall()
        _src_global_total = 0
        _src_local_total = 0
        _src_sample_count = 0
        for (_extra,) in dmesg_rows:
            try:
                _ed = _json.loads(_extra)
                if "src_global" in _ed:
                    _src_global_total += _ed["src_global"]
                    _src_local_total += _ed["src_local"]
                    _src_sample_count += 1
            except Exception:
                pass
    except Exception as e:
        print(f"查询失败: {e}")
        conn.close()
        return
    conn.close()

    if not rows:
        print("无有效 duration_ms 记录（duration_ms=0 的旧记录已排除）")
        return

    durations = [r[3] for r in rows]
    injected_ms = [r[3] for r in rows if r[1] == 1]
    skipped_ms = [r[3] for r in rows if r[1] == 0]

    def stats(lst):
        if not lst:
            return "n/a"
        avg = sum(lst) / len(lst)
        mn = min(lst)
        mx = max(lst)
        p95 = sorted(lst)[int(len(lst) * 0.95)]
        return f"avg={avg:.1f}  min={mn:.1f}  p95={p95:.1f}  max={mx:.1f}  n={len(lst)}"

    print(f"=== retriever.py 执行时间分析 (最近 {len(rows)} 条有效记录) ===")
    print(f"全部:    {stats(durations)}")
    print(f"注入:    {stats(injected_ms)}")
    print(f"跳过:    {stats(skipped_ms)}")
    # B10: per-source injection stats
    if _src_sample_count > 0:
        _total_inj = _src_global_total + _src_local_total
        _global_pct = 100.0 * _src_global_total / _total_inj if _total_inj > 0 else 0
        print(f"来源:    local={_src_local_total} global={_src_global_total} "
              f"({_global_pct:.1f}% global) 样本={_src_sample_count} 次检索")
    print(f"\n目标: <50ms (内部执行时间) / ~130ms (含Python启动)")
    print(f"\n最近 10 条详情:")
    print(f"{'时间':20} {'injected':>9} {'reason':18} {'duration_ms':>12} {'candidates':>10}")
    print("-" * 75)
    for r in rows[:10]:
        ts = r[0][:19].replace("T", " ")
        inj = "✓" if r[1] else "-"
        print(f"{ts:20} {inj:>9} {r[2]:18} {r[3]:12.1f} {r[4]:10}")


if __name__ == "__main__":
    limit = 50
    if "--all" in sys.argv:
        limit = 99999
    elif "--n" in sys.argv:
        idx = sys.argv.index("--n")
        if idx + 1 < len(sys.argv):
            try:
                limit = int(sys.argv[idx + 1])
            except ValueError:
                pass
    report(limit)
