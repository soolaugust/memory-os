#!/usr/bin/env python3
"""
tools/backfill_info_class.py — 迭代325：info_class 反事实回填

问题：iter319 实现了 episodic/semantic/operational/ephemeral 五层分类，
      但 _route_info_class 只在新写入 chunk 时调用，存量 531/545 chunks
      全部是 'world'（97.4%），导致：
        - sleep_consolidate 的 episodic→semantic 晋升从未触发
        - DAMON DEAD 的 ephemeral 快速淘汰策略不生效
        - stale reclaim 对 semantic chunks 的保护失效

修复：遍历所有 info_class='world' 的 chunk，
      重新运行 classify_memory_type(chunk_type, summary) → 批量更新。

OS 类比：e2fsck pass 2/3 — 扫描 inode bitmap 修复错误分配的文件类型标志
  (EXT2_S_IFREG/EXT2_S_IFDIR)，与 inode 的实际内容对齐。

用法：
  python tools/backfill_info_class.py             # dry-run（仅预览）
  python tools/backfill_info_class.py --apply     # 应用修改
  python tools/backfill_info_class.py --stats     # 仅打印当前分布
"""
import sys
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from store import open_db, ensure_schema
from store_vfs import classify_memory_type


def _get_current_distribution(conn) -> dict:
    """查询当前 info_class 分布。"""
    rows = conn.execute(
        "SELECT COALESCE(info_class,'world') as ic, COUNT(*) as n "
        "FROM memory_chunks GROUP BY ic ORDER BY n DESC"
    ).fetchall()
    return {row[0]: row[1] for row in rows}


def run(dry_run: bool = True) -> dict:
    """
    主入口：扫描所有 info_class='world' 的 chunk，重新分类。

    Returns:
        {
          "total_scanned": int,
          "updated": int,
          "by_type": {new_class: count},
          "before": dist_before,
          "after": dist_after,  # dry_run=True 时为预测值
        }
    """
    conn = open_db()
    ensure_schema(conn)

    dist_before = _get_current_distribution(conn)
    total = sum(dist_before.values())

    # 找出所有 world chunk（或 info_class IS NULL）
    rows = conn.execute(
        """SELECT id, chunk_type, summary
           FROM memory_chunks
           WHERE COALESCE(info_class, 'world') = 'world'
           ORDER BY created_at"""
    ).fetchall()

    candidates = []
    for row in rows:
        cid, ctype, summary = row[0], row[1], row[2] or ""
        new_class = classify_memory_type(ctype, summary)
        if new_class != "world":
            candidates.append((cid, ctype, new_class))

    by_type: dict = {}
    for _, ctype, new_class in candidates:
        key = f"{ctype}→{new_class}"
        by_type[key] = by_type.get(key, 0) + 1

    updated = 0
    if not dry_run and candidates:
        now = datetime.now(timezone.utc).isoformat()
        # 批量更新（分批提交，每 500 条一批）
        BATCH = 500
        for i in range(0, len(candidates), BATCH):
            batch = candidates[i:i + BATCH]
            conn.executemany(
                "UPDATE memory_chunks SET info_class=?, updated_at=? WHERE id=?",
                [(new_class, now, cid) for cid, _, new_class in batch]
            )
            conn.commit()
            updated += len(batch)

    dist_after = _get_current_distribution(conn)
    conn.close()

    return {
        "total_chunks": total,
        "world_before": dist_before.get("world", 0),
        "candidates": len(candidates),
        "updated": updated,
        "by_type": by_type,
        "before": dist_before,
        "after": dist_after,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="info_class backfill tool (iter325)")
    parser.add_argument("--apply", action="store_true", help="应用修改（默认 dry-run）")
    parser.add_argument("--stats", action="store_true", help="仅打印当前分布")
    args = parser.parse_args()

    if args.stats:
        conn = open_db()
        ensure_schema(conn)
        dist = _get_current_distribution(conn)
        conn.close()
        total = sum(dist.values())
        print(f"── info_class 分布（共 {total} chunks）──")
        for ic, n in sorted(dist.items(), key=lambda x: -x[1]):
            pct = n / total * 100 if total else 0
            print(f"  {ic:<15} {n:>5}  ({pct:.1f}%)")
        return

    dry_run = not args.apply
    result = run(dry_run=dry_run)

    mode = "dry-run" if dry_run else "applied"
    print(f"\n── iter325 info_class backfill [{mode}] ──")
    print(f"  total chunks:   {result['total_chunks']}")
    print(f"  world before:   {result['world_before']}")
    print(f"  candidates:     {result['candidates']}")
    print(f"  updated:        {result['updated']}")

    if result["by_type"]:
        print(f"\n  reclassification breakdown:")
        for mapping, n in sorted(result["by_type"].items(), key=lambda x: -x[1]):
            print(f"    {mapping:<45} {n:>4}")

    print(f"\n── before ──")
    total = result["total_chunks"]
    for ic, n in sorted(result["before"].items(), key=lambda x: -x[1]):
        pct = n / total * 100 if total else 0
        print(f"  {ic:<15} {n:>5}  ({pct:.1f}%)")

    if not dry_run:
        print(f"\n── after ──")
        after_total = sum(result["after"].values())
        for ic, n in sorted(result["after"].items(), key=lambda x: -x[1]):
            pct = n / after_total * 100 if after_total else 0
            print(f"  {ic:<15} {n:>5}  ({pct:.1f}%)")
    else:
        print(f"\n  [dry-run] 运行 --apply 以应用修改")


if __name__ == "__main__":
    main()
