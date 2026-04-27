#!/usr/bin/env python3
"""
Memory OS Swap Recovery Benchmark

场景 A：单 chunk 往返完整性
场景 B：压缩恢复完整度
场景 C：多轮 compaction 累积损失

用 tmpfs 隔离测试数据库（避免污染生产 store.db）
"""
import json
import os
import sys
import time
import sqlite3
import tempfile
import shutil
from datetime import datetime, timezone
from pathlib import Path

# Add memory-os to path
_MOS_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_MOS_ROOT))

# Import from memory-os
from store_core import (
    checkpoint_dump, checkpoint_restore,
    swap_out, swap_in, open_db, ensure_schema
)
from utils import resolve_project_id


class TmpfsDB:
    """临时内存 DB（用 tmpfs 隔离测试）"""
    def __init__(self):
        self.tmpdir = tempfile.mkdtemp(prefix="mos_eval_")
        self.db_path = Path(self.tmpdir) / "test_store.db"
        self.conn = None

    def init(self):
        """初始化测试数据库"""
        self.conn = sqlite3.connect(str(self.db_path))
        ensure_schema(self.conn)
        self.conn.commit()
        return self.conn

    def cleanup(self):
        """清理临时文件"""
        if self.conn:
            self.conn.close()
        shutil.rmtree(self.tmpdir, ignore_errors=True)


def _create_mock_chunks(count: int, project: str) -> list:
    """创建 mock chunk 数据"""
    chunks = []
    for i in range(count):
        chunk_id = f"chunk_{project}_{i}"
        chunks.append({
            "id": chunk_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "project": project,
            "source_session": "test_session",
            "chunk_type": ["decision", "reasoning_chain", "prompt_context", "code_snippet"][i % 4],
            "content": f"Mock content block {i}\n" * 10 + f"Data: {i * 100} bytes" * 5,
            "summary": f"Mock summary for chunk {i} - important decision #{i}",
            "tags": ["mock", "test"],
            "importance": 0.5 + (i % 5) * 0.1,
            "retrievability": 0.8,
            "last_accessed": datetime.now(timezone.utc).isoformat(),
            "access_count": i % 3,
        })
    return chunks


def _insert_chunks_to_db(conn: sqlite3.Connection, chunks: list) -> None:
    """插入 chunk 到数据库"""
    for c in chunks:
        conn.execute("""
            INSERT INTO memory_chunks
            (id, created_at, updated_at, project, source_session, chunk_type,
             content, summary, tags, importance, retrievability, last_accessed, access_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            c["id"], c["created_at"], c["updated_at"], c["project"],
            c["source_session"], c["chunk_type"], c["content"], c["summary"],
            json.dumps(c["tags"]), c["importance"], c["retrievability"],
            c["last_accessed"], c["access_count"]
        ))
    conn.commit()


def _count_chunk_chars(chunks: list) -> int:
    """计算 chunk 内容总字符数"""
    total = 0
    for c in chunks:
        total += len(c.get("content", "")) + len(c.get("summary", ""))
    return total


def scenario_a_chunk_roundtrip(db: TmpfsDB) -> dict:
    """
    场景 A：单 chunk 往返
    - 写入 10 个 chunks
    - 调用 checkpoint_dump + swap_out
    - 清除内存状态
    - 调用 checkpoint_restore + swap_in
    - 验证所有 chunk 内容完整恢复
    """
    results = {
        "scenario": "A-SingleChunkRoundtrip",
        "chunks_written": 0,
        "chunks_recovered": 0,
        "content_match": True,
        "errors": [],
        "elapsed_ms": 0,
    }

    t0 = time.monotonic()
    project = "scenario_a"
    session_id = "test_session_a"

    try:
        conn = db.init()

        # 写入 10 个 chunks
        chunks = _create_mock_chunks(10, project)
        _insert_chunks_to_db(conn, chunks)
        results["chunks_written"] = 10

        # 获取 chunk IDs
        rows = conn.execute(
            "SELECT id FROM memory_chunks WHERE project = ?", (project,)
        ).fetchall()
        chunk_ids = [r[0] for r in rows]

        # 保存原始内容用于比对
        original_content = {}
        for c in chunks:
            original_content[c["id"]] = c["content"] + c["summary"]

        # checkpoint_dump
        ckpt_result = checkpoint_dump(conn, project, session_id,
                                     hit_chunk_ids=chunk_ids)
        checkpoint_id = ckpt_result.get("checkpoint_id")

        # swap_out
        swap_result = swap_out(conn, chunk_ids)
        swapped_count = swap_result.get("swapped_count", 0)

        # 清空 memory_chunks（模拟内存清空）
        conn.execute("DELETE FROM memory_chunks WHERE project = ?", (project,))
        conn.commit()

        # swap_in
        swap_in_result = swap_in(conn, chunk_ids)
        restored_count = swap_in_result.get("restored_count", 0)

        # checkpoint_restore（如果需要）
        restored_data = checkpoint_restore(conn, project)
        if restored_data:
            results["checkpoint_restored"] = True
            results["checkpoint_chunks"] = len(restored_data.get("chunks", []))

        # 验证内容
        rows = conn.execute(
            "SELECT id, content, summary FROM memory_chunks WHERE project = ?",
            (project,)
        ).fetchall()

        for row in rows:
            cid, content, summary = row
            recovered = content + summary
            if cid in original_content:
                if recovered != original_content[cid]:
                    results["content_match"] = False
                    results["errors"].append(f"Content mismatch for {cid}")

        results["chunks_recovered"] = len(rows)

        conn.close()

    except Exception as e:
        results["errors"].append(f"Exception: {type(e).__name__}: {str(e)}")

    results["elapsed_ms"] = (time.monotonic() - t0) * 1000
    return results


def scenario_b_compression_info_retention(db: TmpfsDB) -> dict:
    """
    场景 B：压缩恢复完整度
    - 比较恢复前 N 个 chunks 的实际数据量 vs 原始数据量
    - 计算信息保留率 = 恢复 chunks 字数 / 原始 chunks 字数
    """
    results = {
        "scenario": "B-CompressionRetention",
        "original_summary_chars": 0,
        "original_content_chars": 0,
        "recovered_summary_chars": 0,
        "recovered_content_chars": 0,
        "total_chunks": 0,
        "recovered_chunks": 0,
        "summary_retention_pct": 0.0,
        "content_retention_pct": 0.0,
        "info_types": {},
        "errors": [],
        "elapsed_ms": 0,
    }

    t0 = time.monotonic()
    project = "scenario_b"
    session_id = "test_session_b"

    try:
        conn = db.init()

        # 创建工作集（20 个 chunks）
        chunks = _create_mock_chunks(20, project)
        _insert_chunks_to_db(conn, chunks)

        results["total_chunks"] = len(chunks)

        # 计算原始字符数（分 summary 和 content）
        for c in chunks:
            results["original_summary_chars"] += len(c.get("summary", ""))
            results["original_content_chars"] += len(c.get("content", ""))

        # 获取 chunk IDs 及类型统计
        rows = conn.execute(
            "SELECT id, chunk_type FROM memory_chunks WHERE project = ?", (project,)
        ).fetchall()
        chunk_ids = [r[0] for r in rows]
        type_counts = {}
        for r in rows:
            ctype = r[1]
            type_counts[ctype] = type_counts.get(ctype, 0) + 1
        results["info_types"] = type_counts

        # checkpoint_dump（保存所有 chunk IDs）
        checkpoint_dump(conn, project, session_id, hit_chunk_ids=chunk_ids)

        # checkpoint_restore（恢复精确工作集）
        ckpt = checkpoint_restore(conn, project)
        if ckpt and ckpt.get("chunks"):
            results["recovered_chunks"] = len(ckpt["chunks"])
            # 正确计算恢复的数据量（使用真实 content 字段）
            for c in ckpt["chunks"]:
                results["recovered_summary_chars"] += len(c.get("summary", ""))
                results["recovered_content_chars"] += len(c.get("content", ""))

        # 计算保留率
        if results["original_summary_chars"] > 0:
            results["summary_retention_pct"] = (
                results["recovered_summary_chars"] / results["original_summary_chars"] * 100
            )
        if results["original_content_chars"] > 0:
            results["content_retention_pct"] = (
                results["recovered_content_chars"] / results["original_content_chars"] * 100
            )

        conn.close()

    except Exception as e:
        results["errors"].append(f"Exception: {type(e).__name__}: {str(e)}")

    results["elapsed_ms"] = (time.monotonic() - t0) * 1000
    return results


def scenario_c_multi_compaction_loss(db: TmpfsDB) -> dict:
    """
    场景 C：多轮 compaction 累积损失
    - 模拟 3 次连续 compaction
    - 每次恢复后再 swap_out
    - 测量第 N 轮后的累积信息损失率
    """
    results = {
        "scenario": "C-MultiCompactionLoss",
        "rounds": 3,
        "initial_chars": 0,
        "round_retention": [
            {"round": 1, "chars": 0, "rate_pct": 0.0},
            {"round": 2, "chars": 0, "rate_pct": 0.0},
            {"round": 3, "chars": 0, "rate_pct": 0.0},
        ],
        "decay_slope_pct_per_round": 0.0,
        "errors": [],
        "elapsed_ms": 0,
    }

    t0 = time.monotonic()
    project = "scenario_c"
    base_session = "test_session_c"

    try:
        conn = db.init()

        # 初始工作集
        chunks = _create_mock_chunks(30, project)
        _insert_chunks_to_db(conn, chunks)

        initial_chars = _count_chunk_chars(chunks)
        results["initial_chars"] = initial_chars

        rows = conn.execute(
            "SELECT id FROM memory_chunks WHERE project = ?", (project,)
        ).fetchall()
        chunk_ids = [r[0] for r in rows]

        # 模拟 3 轮 compaction
        for round_num in range(1, 4):
            session_id = f"{base_session}_{round_num}"

            # checkpoint_dump
            checkpoint_dump(conn, project, session_id, hit_chunk_ids=chunk_ids)

            # checkpoint_restore
            ckpt = checkpoint_restore(conn, project)
            if ckpt and ckpt.get("chunks"):
                # 正确计算：使用完整 content + summary 字符数（与 _count_chunk_chars 一致）
                recovered_chars = sum(
                    len(c.get("content", "")) + len(c.get("summary", ""))
                    for c in ckpt["chunks"]
                )

                rate = (recovered_chars / initial_chars * 100) if initial_chars > 0 else 0
                results["round_retention"][round_num - 1]["chars"] = recovered_chars
                results["round_retention"][round_num - 1]["rate_pct"] = rate

                # swap_out 压缩（模拟下一轮准备）
                if round_num < 3:
                    swap_out(conn, chunk_ids[:len(chunk_ids)//2])

            conn.commit()

        # 计算衰减斜率
        rates = [r["rate_pct"] for r in results["round_retention"]]
        if len(rates) > 1:
            slope = (rates[-1] - rates[0]) / 2  # 平均每轮衰减
            results["decay_slope_pct_per_round"] = slope

        conn.close()

    except Exception as e:
        results["errors"].append(f"Exception: {type(e).__name__}: {str(e)}")

    results["elapsed_ms"] = (time.monotonic() - t0) * 1000
    return results


def generate_report(results_a, results_b, results_c) -> str:
    """生成最终报告"""
    report = []
    report.append("=" * 60)
    report.append("Memory OS Swap Recovery Benchmark Report")
    report.append("=" * 60)
    report.append("")

    # 场景 A
    report.append("场景 A：Chunk 往返完整性")
    report.append("-" * 40)
    if results_a["errors"]:
        report.append("❌ 错误:")
        for e in results_a["errors"]:
            report.append(f"  {e}")
    else:
        ok = "✅" if results_a["chunks_recovered"] == results_a["chunks_written"] and \
             results_a["content_match"] else "⚠️"
        report.append(f"  {results_a['chunks_written']}/{results_a['chunks_recovered']} chunks 恢复 {ok}")
        report.append(f"  内容完整度: {'完全匹配' if results_a['content_match'] else '不匹配'}")
        report.append(f"  耗时: {results_a['elapsed_ms']:.1f}ms")
    report.append("")

    # 场景 B
    report.append("场景 B：Compaction 信息保留率")
    report.append("-" * 40)
    if results_b["errors"]:
        report.append("❌ 错误:")
        for e in results_b["errors"]:
            report.append(f"  {e}")
    else:
        report.append(f"  工作集: {results_b['total_chunks']} chunks")
        if "info_types" in results_b and results_b["info_types"]:
            types_str = " ".join(f"{k}({v})" for k, v in results_b["info_types"].items())
            report.append(f"  信息类型: {types_str}")
        report.append(f"  原始 Summary: {results_b['original_summary_chars']} 字符")
        report.append(f"  原始 Content: {results_b['original_content_chars']} 字符")
        report.append(f"  恢复 Summary: {results_b['recovered_summary_chars']} 字符")
        report.append(f"  恢复 Content: {results_b['recovered_content_chars']} 字符")
        report.append(f"  Summary 保留率: {results_b['summary_retention_pct']:.1f}%")
        report.append(f"  Content 保留率: {results_b['content_retention_pct']:.1f}%")
        report.append(f"  恢复 chunks: {results_b['recovered_chunks']}/{results_b['total_chunks']}")
        report.append(f"  耗时: {results_b['elapsed_ms']:.1f}ms")
    report.append("")

    # 场景 C
    report.append("场景 C：多轮累积损失")
    report.append("-" * 40)
    if results_c["errors"]:
        report.append("❌ 错误:")
        for e in results_c["errors"]:
            report.append(f"  {e}")
    else:
        report.append(f"  初始工作集: {results_c['initial_chars']} 字符")
        for r in results_c["round_retention"]:
            report.append(f"  第{r['round']}轮: {r['chars']} 字符 ({r['rate_pct']:.1f}%)")
        report.append(f"  衰减斜率: {results_c['decay_slope_pct_per_round']:.2f}% 每轮")
        report.append(f"  耗时: {results_c['elapsed_ms']:.1f}ms")
    report.append("")

    # 关键发现
    report.append("关键发现")
    report.append("-" * 40)
    findings = []

    if not results_a["errors"] and results_a["content_match"]:
        findings.append("✅ 单 chunk 往返恢复完整，无数据丢失")
    elif not results_a["errors"]:
        findings.append("⚠️ 部分 chunk 恢复不完整或内容不匹配")

    if not results_b["errors"]:
        rate = results_b["summary_retention_pct"]
        if results_b["recovered_chunks"] == results_b["total_chunks"]:
            findings.append(f"✅ Checkpoint 全量恢复 ({results_b['recovered_chunks']}/{results_b['total_chunks']} chunks)")
        if rate > 80:
            findings.append(f"✅ Summary 保留率高 ({rate:.1f}%)")
        elif rate > 50:
            findings.append(f"⚠️ Summary 保留率中等 ({rate:.1f}%)")
        else:
            findings.append(f"⚠️ Summary 保留率低 ({rate:.1f}%)")

    if not results_c["errors"]:
        slope = results_c["decay_slope_pct_per_round"]
        if slope > -5:
            findings.append(f"✅ 多轮衰减缓慢，系统稳定")
        elif slope > -15:
            findings.append(f"⚠️ 多轮衰减中等，需监控")
        else:
            findings.append(f"❌ 多轮衰减快速，需优化")

    for f in findings:
        report.append(f"  {f}")

    report.append("")
    report.append("=" * 60)

    return "\n".join(report)


def main():
    print("🚀 Memory OS Swap Recovery Benchmark")
    print("  Starting evaluation...")
    print()

    db = TmpfsDB()

    try:
        # 运行三个场景
        print("📝 场景 A：单 chunk 往返...")
        results_a = scenario_a_chunk_roundtrip(db)
        print(f"   完成 ({results_a['elapsed_ms']:.1f}ms)")

        print("📝 场景 B：压缩恢复完整度...")
        results_b = scenario_b_compression_info_retention(db)
        print(f"   完成 ({results_b['elapsed_ms']:.1f}ms)")

        print("📝 场景 C：多轮累积损失...")
        results_c = scenario_c_multi_compaction_loss(db)
        print(f"   完成 ({results_c['elapsed_ms']:.1f}ms)")

        # 生成报告
        report = generate_report(results_a, results_b, results_c)
        print()
        print(report)

        # 保存结果 JSON
        output_file = _MOS_ROOT / "swap_recovery_benchmark.json"
        output_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "scenario_a": results_a,
            "scenario_b": results_b,
            "scenario_c": results_c,
        }
        output_file.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n📊 结果已保存到: {output_file}")

    finally:
        db.cleanup()
        print("✅ Benchmark 完成")


if __name__ == "__main__":
    main()
