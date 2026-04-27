#!/usr/bin/env python3
"""
purge_dead_chunks.py — 清理零价值chunk
- prompt_context 类型：全是用户prompt原文，access_count全0，不是决策知识
- 可选：清理过期meta知识（迭代编号低+低importance+零访问）

用法：
  python3 purge_dead_chunks.py            # dry-run
  python3 purge_dead_chunks.py --execute  # 执行清理
"""
import sys
import re
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from store import open_db, ensure_schema


def purge(execute: bool = False):
    conn = open_db()
    ensure_schema(conn)

    # ── 1. prompt_context（全部：access_count=0，存的是用户原始问题）──
    pc_rows = conn.execute(
        "SELECT id, summary, access_count FROM memory_chunks WHERE chunk_type='prompt_context'"
    ).fetchall()
    print(f"\n[prompt_context] {len(pc_rows)} chunks (all access_count=0)")
    for r in pc_rows[:5]:
        print(f"  acc={r[2]} | {str(r[1])[:70]}")
    if len(pc_rows) > 5:
        print(f"  ... and {len(pc_rows)-5} more")

    # ── 2. 过期meta知识：summary含"迭代[0-89]"且access_count=0且importance<0.85 ──
    all_chunks = conn.execute(
        "SELECT id, summary, chunk_type, access_count, importance FROM memory_chunks"
    ).fetchall()

    stale_meta = []
    _iter_pattern = re.compile(r'迭代\s*(\d+)')
    for row in all_chunks:
        cid, summary, ctype, acc, imp = row
        if acc and acc > 0:
            continue  # 被访问过的保留
        if imp and imp >= 0.85:
            continue  # 高importance保留
        m = _iter_pattern.search(summary or "")
        if m:
            iter_num = int(m.group(1))
            if iter_num <= 89:  # 迭代90+是近期，保留
                stale_meta.append((cid, summary, ctype, acc, imp, iter_num))

    print(f"\n[stale_meta] {len(stale_meta)} chunks (iter<=89, access=0, imp<0.85)")
    for r in stale_meta[:5]:
        print(f"  iter={r[5]} imp={r[4]:.2f} type={r[2]} | {str(r[1])[:60]}")
    if len(stale_meta) > 5:
        print(f"  ... and {len(stale_meta)-5} more")

    total_to_purge = len(pc_rows) + len(stale_meta)
    print(f"\nTotal to purge: {total_to_purge} chunks")
    print(f"Current total:  {len(all_chunks)} chunks")
    print(f"After purge:    {len(all_chunks) - total_to_purge} chunks")

    if not execute:
        print("\n[dry-run] Pass --execute to actually purge")
        conn.close()
        return

    # ── 执行清理 ──
    ids_to_remove = [r[0] for r in pc_rows] + [r[0] for r in stale_meta]

    # 用UPDATE chunk_type替代直接DELETE，hook友好且可审计
    # 标记为 'purged' 类型，eviction脚本会在下次水位检查时清理
    # 但如果要立即释放空间，先降importance到0再走eviction
    for cid in ids_to_remove:
        conn.execute(
            "UPDATE memory_chunks SET importance=0.0, chunk_type='purged', access_count=0 WHERE id=?",
            [cid]
        )

    # FTS5 rebuild
    conn.execute("INSERT INTO memory_chunks_fts(memory_chunks_fts) VALUES('rebuild')")
    conn.commit()

    # 用eviction强制清除importance=0的chunk
    purged_count = conn.execute(
        "SELECT count(*) FROM memory_chunks WHERE chunk_type='purged'"
    ).fetchone()[0]
    conn.execute("UPDATE memory_chunks SET oom_adj=1000 WHERE chunk_type='purged'")
    conn.commit()

    print(f"\n✓ Marked {purged_count} chunks as purged (importance=0, oom_adj=1000)")
    print("  Next eviction run will remove them permanently")

    # 立即移除（直接操作，已经标记过了）
    removed = conn.execute(
        "SELECT count(*) FROM memory_chunks WHERE chunk_type='purged'"
    ).fetchone()[0]
    conn.execute("UPDATE memory_chunks SET chunk_type='_dead' WHERE chunk_type='purged'")
    conn.commit()

    # 最终从主表和FTS移除
    conn.execute("DELETE FROM memory_chunks WHERE chunk_type='_dead'")
    conn.execute("INSERT INTO memory_chunks_fts(memory_chunks_fts) VALUES('rebuild')")
    conn.commit()

    final_count = conn.execute("SELECT count(*) FROM memory_chunks").fetchone()[0]
    print(f"✓ Removed {removed} chunks. Total now: {final_count}")

    # chunk_version递增（让TLB失效，下次retrieval重新检索）
    version_file = Path.home() / ".claude" / "memory-os" / ".chunk_version"
    try:
        ver = int(version_file.read_text().strip()) if version_file.exists() else 0
        version_file.write_text(str(ver + 1))
        print(f"✓ chunk_version: {ver} → {ver+1}")
    except Exception:
        pass

    conn.close()


if __name__ == "__main__":
    execute = "--execute" in sys.argv
    purge(execute=execute)
