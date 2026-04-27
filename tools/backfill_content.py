#!/usr/bin/env python3
"""
tools/backfill_content.py — 迭代326：content 富化回填

问题：
  1. causal_chain 存量 86 个（iter324 只修了新写入路径），86 个平均 content=89 chars
  2. quantitative_evidence 存量 80 个，平均 content=103 chars
  FTS5 token 不足 → avg_acc=1.33 / 0.49，大量零访问。

修复策略：
  causal_chain：按 project + created_at 排序，±1 邻居聚合（同 iter324 逻辑）
  quantitative_evidence：按 project + created_at 排序，±1 邻居用 | 拼接

OS 类比：e2fsck inode rehash — 修复旧 inode 的 hash 字段（老格式写入时
  没有计算，新格式才加上），与新写入路径对齐。

用法：
  python tools/backfill_content.py              # dry-run
  python tools/backfill_content.py --apply      # 应用修改
  python tools/backfill_content.py --stats      # 仅统计
"""
import sys
import re
from pathlib import Path
from datetime import datetime, timezone

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from store import open_db, ensure_schema


# ── 判断 content 是否已经是"富" content（被 iter324/326 处理过的）──
def _is_rich_content(content: str, summary: str, chunk_type: str) -> bool:
    """已经是富 content 的标志：包含邻居拼接分隔符，且 content 长度是 summary 的 1.5 倍以上。"""
    if not content or not summary:
        return False
    # 富 content 必须包含邻居拼接分隔符（是核心标志）
    has_separator = (
        (chunk_type == "causal_chain" and " → " in content)
        or (chunk_type == "quantitative_evidence" and " | " in content)
        or (chunk_type == "conversation_summary" and " | " in content)
    )
    if not has_separator:
        # 极长 content 也视为已富化（人工写入的特殊 content）
        # 但必须是 summary 的 2 倍以上，避免误判短 summary
        if len(summary) > 0 and len(content) > len(summary) * 2 and len(content) > 150:
            return True
        return False
    # 有分隔符：还需要 content 显著长于 summary（排除 summary 本身含 → 的情况）
    return len(content) > len(summary) * 1.3


def _build_rich_content(summaries: list, idx: int, chunk_type: str, topic: str) -> str:
    """为 idx 位置的 chunk 构建富 content。"""
    parts = []
    if idx > 0:
        parts.append(summaries[idx - 1])
    parts.append(summaries[idx])
    if idx < len(summaries) - 1:
        parts.append(summaries[idx + 1])

    topic_tag = f"[{chunk_type}|{topic}]" if topic else f"[{chunk_type}]"

    if chunk_type == "causal_chain":
        return f"{topic_tag} {' → '.join(parts)}"[:400]
    else:  # quantitative_evidence, conversation_summary — 均用 | 拼接
        return f"{topic_tag} {' | '.join(parts)}"[:400]


def _extract_topic_from_content(content: str) -> str:
    """从已有 content 的前缀提取 topic（如 [causal_chain|topic] → topic）。"""
    m = re.match(r'^\[[\w_]+\|([^\]]+)\]', content)
    return m.group(1) if m else ""


def run(chunk_types: list = None, dry_run: bool = True) -> dict:
    """
    主入口：扫描指定类型的贫 content chunk，富化并批量更新。

    chunk_types: ["causal_chain", "quantitative_evidence"]（默认两者都处理）
    """
    if chunk_types is None:
        chunk_types = ["causal_chain", "quantitative_evidence", "conversation_summary"]

    conn = open_db()
    ensure_schema(conn)

    result = {ct: {"total": 0, "poor": 0, "updated": 0} for ct in chunk_types}

    for chunk_type in chunk_types:
        # 取所有该类型的 chunk，按 project + created_at 排序（还原写入时的批次顺序）
        rows = conn.execute(
            """SELECT id, summary, content, project,
                      COALESCE(
                        (SELECT value FROM json_each(tags) WHERE value NOT IN (?, 'global') LIMIT 1),
                        ''
                      ) as topic_hint
               FROM memory_chunks
               WHERE chunk_type = ?
               ORDER BY project, created_at""",
            (chunk_type, chunk_type)
        ).fetchall()

        result[chunk_type]["total"] = len(rows)

        # 按 project 分组，每个 project 内独立做邻居聚合
        from collections import defaultdict
        by_project = defaultdict(list)
        for row in rows:
            by_project[row[1]].append(row)  # row[1] = project? No, row index

        # 重建 by project
        by_project2 = defaultdict(list)
        for row in rows:
            cid, summary, content, project, topic_hint = row
            by_project2[project].append((cid, summary, content, topic_hint))

        updates = []
        for project, chunks in by_project2.items():
            summaries = [c[1] for c in chunks]
            for idx, (cid, summary, content, topic_hint) in enumerate(chunks):
                result[chunk_type]["total"] += 0  # already counted

                # 判断是否需要富化
                if _is_rich_content(content, summary, chunk_type):
                    continue  # 已经是富 content，跳过

                result[chunk_type]["poor"] += 1

                # 提取 topic（从已有 content 前缀 或 tags 中的 topic_hint）
                topic = _extract_topic_from_content(content) or topic_hint or ""

                rich = _build_rich_content(summaries, idx, chunk_type, topic)

                # 只有 rich 比原 content 显著更长才更新
                if len(rich) > len(content) + 10:
                    updates.append((rich, cid))

        result[chunk_type]["poor"] = len(updates) + (result[chunk_type]["poor"] - len(updates))

        if not dry_run and updates:
            now = datetime.now(timezone.utc).isoformat()
            BATCH = 500
            for i in range(0, len(updates), BATCH):
                batch = updates[i:i + BATCH]
                conn.executemany(
                    "UPDATE memory_chunks SET content=?, updated_at=? WHERE id=?",
                    [(content, now, cid) for content, cid in batch]
                )
                conn.commit()
            result[chunk_type]["updated"] = len(updates)

    conn.close()
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="content 富化回填 (iter326)")
    parser.add_argument("--apply", action="store_true", help="应用修改（默认 dry-run）")
    parser.add_argument("--type", choices=["causal_chain", "quantitative_evidence", "conversation_summary", "all"],
                        default="all", help="处理的 chunk 类型")
    parser.add_argument("--stats", action="store_true", help="仅打印当前 content 长度统计")
    args = parser.parse_args()

    if args.stats:
        conn = open_db()
        ensure_schema(conn)
        for ct in ["causal_chain", "quantitative_evidence", "conversation_summary"]:
            rows = conn.execute(
                """SELECT LENGTH(content) as clen, COUNT(*) as n,
                          AVG(COALESCE(access_count,0)) as avg_acc
                   FROM memory_chunks WHERE chunk_type=?
                   GROUP BY CASE WHEN LENGTH(content)<100 THEN '<100'
                                 WHEN LENGTH(content)<200 THEN '100-200'
                                 WHEN LENGTH(content)<300 THEN '200-300'
                                 ELSE '300+' END
                   ORDER BY clen""",
                (ct,)
            ).fetchall()
            print(f"\n── {ct} content 分布 ──")
            for r in rows:
                print(f"  len={r[0]:>4}  n={r[1]:>4}  avg_acc={r[2]:>6.2f}")
        conn.close()
        return

    types = (["causal_chain", "quantitative_evidence", "conversation_summary"]
             if args.type == "all" else [args.type])
    dry_run = not args.apply

    result = run(chunk_types=types, dry_run=dry_run)

    mode = "dry-run" if dry_run else "applied"
    print(f"\n── iter326 content 富化回填 [{mode}] ──")
    for ct, r in result.items():
        print(f"  {ct}:")
        print(f"    total: {r['total']}")
        print(f"    poor content (needs update): {r['poor']}")
        print(f"    updated: {r['updated']}")
    if dry_run:
        print("\n  [dry-run] 运行 --apply 以应用修改")


if __name__ == "__main__":
    main()
