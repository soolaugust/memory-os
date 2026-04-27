#!/usr/bin/env python3
"""
tools/backfill_entity_graph.py — 迭代318：知识图谱全量回填

对 DB 中所有现存 chunk 重新跑 summary 三元组抽取，填充 entity_edges。
这是一次性的迁移脚本，不影响日常运行。

OS 类比：ext3 htree 迁移 — tune2fs -E dir_index / e2fsck -D 对已有目录
  重建 B-tree 索引，使历史数据也能享受新索引能力。

用法：
  python tools/backfill_entity_graph.py [--dry-run] [--project PROJECT]
"""
import sys
import argparse
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "hooks"))

from store_vfs import open_db, ensure_schema
# extractor 函数
from extractor import extract_and_write_summary_triples, extract_summary_triples


def backfill(dry_run: bool = False, project_filter: str = None):
    conn = open_db()
    ensure_schema(conn)

    # 查全部 chunk
    if project_filter:
        rows = conn.execute(
            "SELECT id, summary, project, chunk_type FROM memory_chunks "
            "WHERE summary != '' AND project = ?",
            (project_filter,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, summary, project, chunk_type FROM memory_chunks "
            "WHERE summary != ''"
        ).fetchall()

    print(f"扫描 {len(rows)} 个 chunk ...")

    total_edges = 0
    chunk_with_edges = 0

    for cid, summary, project, chunk_type in rows:
        triples = extract_summary_triples(summary)
        if not triples:
            continue

        chunk_with_edges += 1
        if dry_run:
            print(f"  [{chunk_type}] {summary[:60]}")
            for f, r, t in triples:
                print(f"    → ({f}) --[{r}]--> ({t})")
            total_edges += len(triples)
        else:
            n = extract_and_write_summary_triples(summary, cid, project, conn)
            total_edges += n

    if not dry_run:
        conn.commit()

    # 统计结果
    after_edges = conn.execute("SELECT COUNT(*) FROM entity_edges").fetchone()[0]
    after_chunks_mapped = conn.execute(
        "SELECT COUNT(DISTINCT chunk_id) FROM entity_map"
    ).fetchone()[0]
    total_chunks = conn.execute("SELECT COUNT(*) FROM memory_chunks").fetchone()[0]

    print()
    print(f"{'[DRY RUN] ' if dry_run else ''}回填完成:")
    print(f"  有三元组的 chunk: {chunk_with_edges} / {len(rows)}")
    print(f"  新增/更新边数: {total_edges}")
    if not dry_run:
        print(f"  entity_edges 总数: {after_edges}")
        print(f"  entity_map 覆盖 chunk 数: {after_chunks_mapped} / {total_chunks} "
              f"({after_chunks_mapped/max(total_chunks,1)*100:.1f}%)")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="只打印，不写 DB")
    parser.add_argument("--project", default=None, help="限定项目 ID")
    args = parser.parse_args()
    backfill(dry_run=args.dry_run, project_filter=args.project)
