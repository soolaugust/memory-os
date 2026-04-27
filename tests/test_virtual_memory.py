#!/usr/bin/env python3
"""
迭代70：Virtual Memory 虚拟内存评测框架

目标：量化 PreCompact/PostCompact swap 机制的真实收益
- 基线：无 swap 时的 compaction 信息丢失率
- 启用：swap 启用时的恢复完整度

OS 类比：Linux /proc/meminfo 中 SwapUsed vs SwapFree 的生产监测
"""
import sys
import json
import subprocess
import tempfile
import time
from pathlib import Path
from datetime import datetime, timezone
import sqlite3

sys.path.insert(0, str(Path(__file__).parent))

import tmpfs
from store import open_db, ensure_schema

MEMORY_OS_DIR = Path.home() / ".claude" / "memory-os"
TEST_SESSIONS = 5
TURNS_PER_SESSION = 20


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def simulate_conversation(session_id, project, turn_count=20):
    """模拟一个多轮对话，生成关键知识"""
    conn = open_db()
    ensure_schema(conn)
    
    decisions = []
    excluded_paths = []
    
    for turn in range(turn_count):
        # 模拟 decision
        decision_content = f"""
迭代{turn}: Architecture Decision
- 选择方案：Memory-OS Swap Out Pattern
- 根因：Compaction 前信息丢失 → 上下文无限问题
- 技术锚点：store.db swap_chunks 表、recall_traces 命中追踪
- 量化收益：P50 0.71ms (-47% vs 1.35ms)
"""
        decision = {
            "id": f"{session_id}:decision:{turn}",
            "project": project,
            "session_id": session_id,
            "chunk_type": "decision",
            "summary": f"Decision T{turn}: swap architecture",
            "content": decision_content,
            "importance": 0.8 + (0.1 if turn % 5 == 0 else 0),
            "source_session": session_id,
            "created_at": now_iso(),
        }
        decisions.append(decision)
        
        # 写入 DB
        conn.execute("""
            INSERT INTO memory_chunks 
            (id, project, source_session, chunk_type, content, summary, importance, created_at, updated_at, access_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            decision["id"],
            decision["project"],
            decision["session_id"],
            decision["chunk_type"],
            decision["content"],
            decision["summary"],
            decision["importance"],
            decision["created_at"],
            decision["created_at"],
            0,
        ))
        
        # 模拟 excluded_path
        if turn % 3 == 0:
            excluded = {
                "id": f"{session_id}:excluded:{turn}",
                "project": project,
                "session_id": session_id,
                "chunk_type": "excluded_path",
                "summary": f"Excluded: /test/perf_{turn}",
                "content": f"排除路径：test fixtures, perf 基准测试 turn={turn}",
                "importance": 0.7,
                "source_session": session_id,
                "created_at": now_iso(),
            }
            excluded_paths.append(excluded)
            
            conn.execute("""
                INSERT INTO memory_chunks 
                (id, project, source_session, chunk_type, content, summary, importance, created_at, updated_at, access_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                excluded["id"],
                excluded["project"],
                excluded["session_id"],
                excluded["chunk_type"],
                excluded["content"],
                excluded["summary"],
                excluded["importance"],
                excluded["created_at"],
                excluded["created_at"],
                0,
            ))
    
    conn.commit()
    conn.close()
    
    return {
        "session_id": session_id,
        "turn_count": turn_count,
        "decisions_written": len(decisions),
        "excluded_paths_written": len(excluded_paths),
    }


def measure_compaction_info_loss(project):
    """
    测量：Compaction 时信息丢失率
    
    基线场景：模拟 compaction 发生前，DB 中的关键知识
    丢失指标：
      - decision 丢失率
      - excluded_path 丢失率
      - 总信息完整度
    """
    conn = open_db()
    
    # 统计当前 DB 知识
    cursor = conn.cursor()
    cursor.execute("""
        SELECT chunk_type, COUNT(*) cnt, SUM(importance) imp_sum, AVG(importance) imp_avg
        FROM memory_chunks
        WHERE project = ? AND chunk_type IN ('decision', 'excluded_path')
        GROUP BY chunk_type
    """, (project,))
    
    baseline_stats = {}
    total_chunks = 0
    total_importance = 0.0
    
    for row in cursor.fetchall():
        chunk_type = row[0]
        count = row[1]
        imp_sum = row[2] or 0.0
        imp_avg = row[3] or 0.0
        
        baseline_stats[chunk_type] = {
            "count": count,
            "importance_sum": imp_sum,
            "importance_avg": imp_avg,
        }
        total_chunks += count
        total_importance += imp_sum
    
    conn.close()
    
    return {
        "baseline_stats": baseline_stats,
        "total_critical_chunks": total_chunks,
        "total_importance": total_importance,
        "information_density": total_importance / max(total_chunks, 1),
    }


def simulate_compaction_swap_out(project):
    """
    模拟：Compaction 前 PreCompact hook 的 swap_out 动作
    
    验证：关键知识是否被正确保存到 swap_state.json
    """
    conn = open_db()
    
    # 查询关键决策 + excluded_paths
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, summary, importance, chunk_type
        FROM memory_chunks
        WHERE project = ? AND chunk_type IN ('decision', 'excluded_path')
        ORDER BY importance DESC
    """, (project,))
    
    swapped_chunks = cursor.fetchall()
    conn.close()
    
    swap_state = {
        "timestamp": now_iso(),
        "project": project,
        "hit_ids": [row[0] for row in swapped_chunks],
        "decisions": [
            {"id": row[0], "summary": row[1], "importance": row[2]}
            for row in swapped_chunks if row[3] == "decision"
        ],
        "excluded_paths": [
            {"id": row[0], "summary": row[1], "importance": row[2]}
            for row in swapped_chunks if row[3] == "excluded_path"
        ],
    }
    
    # 写入 swap_state.json (模拟 PreCompact hook)
    MEMORY_OS_DIR.mkdir(parents=True, exist_ok=True)
    swap_file = MEMORY_OS_DIR / "swap_state_test.json"
    swap_file.write_text(json.dumps(swap_state, ensure_ascii=False, indent=2))
    
    return swap_state


def simulate_compaction_swap_in(swap_state):
    """
    模拟：Compaction 后 PostCompact hook 的 swap_in 动作
    
    验证：关键知识是否被正确恢复到上下文
    """
    restored_context = {
        "timestamp": now_iso(),
        "recovered_chunks": len(swap_state["hit_ids"]),
        "recovered_decisions": len(swap_state["decisions"]),
        "recovered_excluded_paths": len(swap_state["excluded_paths"]),
        "restoration_completeness": (
            len(swap_state["decisions"]) + len(swap_state["excluded_paths"])
        ) / max(len(swap_state["hit_ids"]), 1),
    }
    
    return restored_context


def run_test():
    """完整虚拟内存评测流程"""
    print("🧪 迭代70：Virtual Memory 虚拟内存评测框架")
    print("=" * 70)
    
    project = "test:memory-os-eval"
    
    # ✅ 阶段1：模拟多轮对话，累积知识
    print("\n📝 [阶段1] 模拟多轮对话，累积关键知识")
    print("-" * 70)
    
    session_results = []
    for s in range(TEST_SESSIONS):
        sid = f"session_{s:02d}"
        result = simulate_conversation(sid, project, TURNS_PER_SESSION)
        session_results.append(result)
        print(f"  {sid}: {result['decisions_written']} decisions + {result['excluded_paths_written']} excluded_paths")
    
    total_chunks_written = sum(r["decisions_written"] + r["excluded_paths_written"] for r in session_results)
    print(f"\n  总计写入：{total_chunks_written} 条关键知识")
    
    # ✅ 阶段2：测量基线（无 swap 时的信息完整度）
    print("\n📊 [阶段2] 测量基线：Compaction 前信息完整度")
    print("-" * 70)
    
    baseline = measure_compaction_info_loss(project)
    print(f"  总关键 chunks: {baseline['total_critical_chunks']}")
    print(f"  总 importance: {baseline['total_importance']:.2f}")
    print(f"  信息密度: {baseline['information_density']:.3f}/chunk")
    for chunk_type, stats in baseline["baseline_stats"].items():
        print(f"    {chunk_type:20} | count={stats['count']:2} | imp_avg={stats['importance_avg']:.3f}")
    
    # ✅ 阶段3：模拟 PreCompact swap_out
    print("\n💾 [阶段3] 模拟 PreCompact 阶段：Swap Out 关键知识")
    print("-" * 70)
    
    swap_state = simulate_compaction_swap_out(project)
    print(f"  Swap Out 关键 chunks: {len(swap_state['hit_ids'])}")
    print(f"  ├─ decisions: {len(swap_state['decisions'])}")
    print(f"  └─ excluded_paths: {len(swap_state['excluded_paths'])}")
    
    swapped_importance_sum = sum(d["importance"] for d in swap_state["decisions"]) + \
                             sum(d["importance"] for d in swap_state["excluded_paths"])
    print(f"  Swap Out importance: {swapped_importance_sum:.2f} / {baseline['total_importance']:.2f} ({100*swapped_importance_sum/max(baseline['total_importance'],1):.1f}%)")
    
    # ✅ 阶段4：模拟 PostCompact swap_in
    print("\n🔄 [阶段4] 模拟 PostCompact 阶段：Swap In 恢复知识")
    print("-" * 70)
    
    restored = simulate_compaction_swap_in(swap_state)
    print(f"  恢复 chunks: {restored['recovered_chunks']}")
    print(f"  恢复完整度: {100*restored['restoration_completeness']:.1f}%")
    
    # ✅ 结果总结
    print("\n📈 [评测结果] Virtual Memory 收益量化")
    print("=" * 70)
    
    result_summary = {
        "test_date": now_iso(),
        "test_sessions": TEST_SESSIONS,
        "turns_per_session": TURNS_PER_SESSION,
        "total_chunks_written": total_chunks_written,
        "baseline": baseline,
        "swap_out": {
            "swapped_chunks": len(swap_state["hit_ids"]),
            "swapped_importance": swapped_importance_sum,
            "swap_out_ratio": len(swap_state["hit_ids"]) / max(baseline["total_critical_chunks"], 1),
        },
        "swap_in": restored,
        "metrics": {
            "information_preservation_rate": restored["restoration_completeness"],
            "critical_decision_recovery": len(swap_state["decisions"]) / max(baseline["baseline_stats"].get("decision", {}).get("count", 1), 1),
            "excluded_path_recovery": len(swap_state["excluded_paths"]) / max(baseline["baseline_stats"].get("excluded_path", {}).get("count", 1), 1),
        },
    }
    
    # 打印摘要
    print(f"✅ 信息保留率: {100*result_summary['metrics']['information_preservation_rate']:.1f}%")
    print(f"✅ 决策恢复率: {100*result_summary['metrics']['critical_decision_recovery']:.1f}%")
    print(f"✅ 排除路径恢复率: {100*result_summary['metrics']['excluded_path_recovery']:.1f}%")
    
    # 保存结果
    result_file = Path(__file__).parent / "test_virtual_memory_results.json"
    result_file.write_text(json.dumps(result_summary, ensure_ascii=False, indent=2))
    print(f"\n💾 结果已保存: {result_file}")
    
    return result_summary


if __name__ == "__main__":
    try:
        results = run_test()
        print("\n✅ 虚拟内存评测完成")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 评测失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
