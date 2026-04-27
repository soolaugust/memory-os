#!/usr/bin/env python3
"""
memory-os 隔夜 consolidation — ksmd 式高相似度 chunk 合并

迭代302：Overnight Consolidation（借鉴 GBrain overnight brain maintenance）

OS 类比：Linux KSM（Kernel Samepage Merging，内核同页合并）
  ksmd（kthread）扫描进程地址空间，识别内容相同/相近的匿名页，
  将其合并为一个共享物理页（Copy-on-Write），释放重复内存。
  memory-os 等价：扫描项目内高相似度 chunk，将其合并为一个更完整的 chunk，
  降低知识冗余，提升检索 precision。

与 extractor.py 的 merge_similar 区别：
  - merge_similar：写入时在线去重，阈值低（0.22），快速判断
  - consolidate：离线隔夜批量扫描，阈值高（0.85），深度合并 + 保留 ghost 引用

合并策略：
  1. 选择 importance 更高的 chunk 为 survivor
  2. 将 secondary 的 content 追加到 survivor（多角度表述聚合）
  3. survivor.stability = max(s1.stability, s2.stability) * 1.2（合并加固）
  4. secondary 降级为 ghost（content 替换为 ghost 指针，importance=0，
     oom_adj=500 优先驱逐），保留 id 防止外部引用断裂

用法：
  python3 tools/consolidate.py [--project PROJECT] [--dry-run] [--threshold 0.85]

可通过 cron 或手动触发。
"""
import sys
import os
import argparse
import sqlite3
import json
import re
from datetime import datetime, timezone

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from store import open_db, ensure_schema
from utils import resolve_project_id


def _jaccard_similarity(a: str, b: str) -> float:
    """
    轻量 Jaccard 相似度（基于 trigram 集合）。
    不依赖向量/embedding，纯字符串操作，< 1ms。
    OS 类比：CRC32 checksum 比较 — 快速判断两个数据块是否值得深度对比。
    """
    if not a or not b:
        return 0.0
    # trigram 集合
    def trigrams(s):
        s = re.sub(r'\s+', ' ', s.strip().lower())
        return set(s[i:i+3] for i in range(len(s)-2)) if len(s) >= 3 else set(s)

    ta, tb = trigrams(a), trigrams(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def consolidate_project(conn: sqlite3.Connection, project: str,
                        threshold: float = 0.85, dry_run: bool = False) -> dict:
    """
    扫描 project 内所有 chunk，合并相似度 >= threshold 的 chunk 对。
    返回统计信息字典。
    """
    # 只处理非 ghost、有实际内容的 chunk
    rows = conn.execute("""
        SELECT id, chunk_type, summary, content, importance,
               COALESCE(stability, 1.0), COALESCE(info_class, 'world'),
               created_at
        FROM memory_chunks
        WHERE project=?
          AND COALESCE(oom_adj, 0) <= 0
          AND length(COALESCE(summary,'')) > 10
          AND COALESCE(content,'') NOT LIKE 'GHOST→%'
        ORDER BY importance DESC, created_at ASC
    """, (project,)).fetchall()

    if len(rows) < 2:
        return {"scanned": len(rows), "merged": 0, "ghosted": 0}

    now_iso = datetime.now(timezone.utc).isoformat()
    merged = 0
    ghosted = 0
    survivor_ids = set()  # 本轮已被选为 survivor 的，不再作为 secondary

    # O(N²) 扫描，但 N 通常 < 500，可接受（ksmd 也是全扫描）
    for i, row_a in enumerate(rows):
        id_a, type_a, summary_a, content_a, imp_a, stab_a, ic_a, _ = row_a
        if id_a in survivor_ids:
            continue

        for row_b in rows[i+1:]:
            id_b, type_b, summary_b, content_b, imp_b, stab_b, ic_b, _ = row_b
            if id_b in survivor_ids:
                continue
            # 只合并同 chunk_type 或相近类型（避免跨语义合并）
            if type_a != type_b and not (
                {type_a, type_b} <= {"decision", "reasoning_chain", "causal_chain"}
            ):
                continue

            sim = _jaccard_similarity(summary_a, summary_b)
            if sim < threshold:
                continue

            # survivor = importance 更高的；相同时取 stability 更高的
            if imp_a >= imp_b or (imp_a == imp_b and stab_a >= stab_b):
                survivor_id, survivor_summary, survivor_content = id_a, summary_a, content_a
                survivor_imp, survivor_stab = imp_a, stab_a
                ghost_id, ghost_summary = id_b, summary_b
            else:
                survivor_id, survivor_summary, survivor_content = id_b, summary_b, content_b
                survivor_imp, survivor_stab = imp_b, stab_b
                ghost_id, ghost_summary = id_a, summary_a

            print(f"  [consolidate] sim={sim:.3f} survivor={survivor_id[:8]} "
                  f"ghost={ghost_id[:8]}")
            print(f"    survivor: {survivor_summary[:60]}")
            print(f"    ghost:    {ghost_summary[:60]}")

            if not dry_run:
                # 1. 合并内容到 survivor
                if ghost_summary not in survivor_content:
                    new_content = (survivor_content + "\n[merged] " + ghost_summary).strip()[:3000]
                else:
                    new_content = survivor_content
                new_stab = min(365.0, max(survivor_stab, stab_b) * 1.2)

                conn.execute("""
                    UPDATE memory_chunks
                    SET content=?, stability=?, updated_at=?
                    WHERE id=?
                """, (new_content, new_stab, now_iso, survivor_id))

                # 2. secondary 降级为 ghost
                ghost_content = f"GHOST→{survivor_id} [{ghost_summary[:80]}]"
                conn.execute("""
                    UPDATE memory_chunks
                    SET content=?, importance=0.0, oom_adj=500, updated_at=?
                    WHERE id=?
                """, (ghost_content, now_iso, ghost_id))

                merged += 1
                ghosted += 1

            survivor_ids.add(ghost_id)  # 避免 ghost 再被其他 chunk 配对

    if not dry_run:
        conn.commit()

    return {"scanned": len(rows), "merged": merged, "ghosted": ghosted}


def main():
    parser = argparse.ArgumentParser(description="memory-os overnight consolidation")
    parser.add_argument("--project", default=None, help="项目 ID（默认自动解析当前目录）")
    parser.add_argument("--dry-run", action="store_true", help="只显示配对，不实际合并")
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="相似度阈值（默认 0.85）")
    parser.add_argument("--all-projects", action="store_true", help="扫描所有项目")
    args = parser.parse_args()

    conn = open_db()
    ensure_schema(conn)

    if args.all_projects:
        projects = [r[0] for r in conn.execute(
            "SELECT DISTINCT project FROM memory_chunks WHERE project IS NOT NULL"
        ).fetchall()]
    else:
        project = args.project or resolve_project_id(os.getcwd())
        projects = [project]

    total = {"scanned": 0, "merged": 0, "ghosted": 0}
    for proj in projects:
        print(f"\n[project: {proj}]")
        stats = consolidate_project(conn, proj, threshold=args.threshold,
                                    dry_run=args.dry_run)
        print(f"  scanned={stats['scanned']} merged={stats['merged']} "
              f"ghosted={stats['ghosted']}"
              + (" [DRY RUN]" if args.dry_run else ""))
        for k in total:
            total[k] += stats[k]

    print(f"\n[total] scanned={total['scanned']} merged={total['merged']} "
          f"ghosted={total['ghosted']}")

    # ── 迭代331：Sleep Consolidation — episodic→semantic 晋升 + 情节衰减 ──
    # OS 类比：Linux kswapd + ksmd — 慢波睡眠期间执行记忆重组
    # 以前 sleep_consolidate 在 store_vfs.py 中实现，但从未被 sleeping hook 实际调用。
    # 修复：在 consolidate.py（sleeping hook 的入口）末尾调用 sleep_consolidate + episodic_decay_scan。
    if not args.dry_run:
        try:
            from store_vfs import sleep_consolidate as _sleep_con, episodic_decay_scan as _ep_scan
            ep_total = {"promoted": 0, "decayed": 0, "sc_merged": 0}
            for proj in projects:
                try:
                    # episodic_decay_scan：晋升 access_count>=2 的情节 chunk + 衰减陈旧情节
                    ep_result = _ep_scan(conn, project=proj, semantic_threshold=2, stale_days=30)
                    conn.commit()
                    ep_total["promoted"] += ep_result.get("promoted", 0)
                    ep_total["decayed"] += ep_result.get("decayed", 0)

                    # sleep_consolidate：合并高相似 chunk + stability 动态调整
                    sc_result = _sleep_con(conn, project=proj)
                    conn.commit()
                    ep_total["sc_merged"] += sc_result.get("merged", 0)
                except Exception as _ep_e:
                    print(f"  [sleep_consolidate] {proj} 跳过: {_ep_e}")
            print(f"\n[sleep_consolidate] promoted={ep_total['promoted']} "
                  f"decayed={ep_total['decayed']} merged={ep_total['sc_merged']}")
        except Exception as _sc_e:
            print(f"\n[sleep_consolidate] 跳过: {_sc_e}")

    conn.close()


if __name__ == "__main__":
    main()
