#!/usr/bin/env python3
"""
memory-os DB Hygiene — 迭代59：清理重复 chunk + 数据完整性检查

OS 类比：fsck (file system check) — 文件系统一致性检查与修复

用法：
  python3 db_hygiene.py              # dry-run 模式，只报告不修改
  python3 db_hygiene.py --fix        # 执行清理
  python3 db_hygiene.py --verify     # 清理后验证

修改的文件：~/.claude/memory-os/store.db（memory_chunks 表 + FTS5 索引）
回滚：python3 -c "import shutil; shutil.copy2('/tmp/store.db-backup-iter59', Path.home() / '.claude' / 'memory-os' / 'store.db')"
"""
import sys
import os
import sqlite3
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

STORE_DB = Path(os.environ.get("MEMORY_OS_DB", str(Path.home() / ".claude" / "memory-os" / "store.db")))


def find_duplicates(conn):
    """找出所有精确重复的 chunk 组（相同 summary + chunk_type）"""
    rows = conn.execute('''
        SELECT summary, chunk_type, COUNT(*) as cnt,
               GROUP_CONCAT(id, '|') as ids
        FROM memory_chunks
        GROUP BY summary, chunk_type
        HAVING cnt > 1
        ORDER BY cnt DESC
    ''').fetchall()
    return rows


def dedup_chunks(conn, dry_run=True):
    """
    去重：每组保留 access_count 最高（ties 时取最早创建的）那一条，删除其余。
    返回 (deleted_count, kept_ids)
    """
    dupes = find_duplicates(conn)
    if not dupes:
        return 0, []

    delete_ids = []
    keep_ids = []

    for summary, chunk_type, cnt, id_str in dupes:
        ids = id_str.split('|')
        rows = conn.execute(f'''
            SELECT id, access_count, created_at FROM memory_chunks
            WHERE id IN ({",".join("?" for _ in ids)})
            ORDER BY access_count DESC, created_at ASC
        ''', ids).fetchall()

        keep_ids.append(rows[0][0])
        for row in rows[1:]:
            delete_ids.append(row[0])

    if dry_run:
        return len(delete_ids), keep_ids

    for did in delete_ids:
        conn.execute("DELETE FROM memory_chunks WHERE id = ?", (did,))
    conn.commit()

    conn.execute("INSERT INTO memory_chunks_fts(memory_chunks_fts) VALUES('rebuild')")
    conn.commit()

    return len(delete_ids), keep_ids


def verify(conn):
    """验证数据完整性"""
    issues = []

    dupes = find_duplicates(conn)
    if dupes:
        issues.append(f"Still {len(dupes)} duplicate groups")

    main_count = conn.execute("SELECT COUNT(*) FROM memory_chunks").fetchone()[0]
    fts_count = conn.execute("SELECT COUNT(*) FROM memory_chunks_fts").fetchone()[0]
    if main_count != fts_count:
        issues.append(f"FTS5 count mismatch: main={main_count} fts={fts_count}")

    empty = conn.execute("SELECT COUNT(*) FROM memory_chunks WHERE summary IS NULL OR summary = ''").fetchone()[0]
    if empty:
        issues.append(f"{empty} chunks with empty summary")

    return issues


def report(conn):
    """打印当前状态报告"""
    total = conn.execute("SELECT COUNT(*) FROM memory_chunks").fetchone()[0]
    rows = conn.execute('''
        SELECT chunk_type, COUNT(*), ROUND(AVG(importance),2), ROUND(AVG(access_count),1)
        FROM memory_chunks GROUP BY chunk_type ORDER BY COUNT(*) DESC
    ''').fetchall()

    dupes = find_duplicates(conn)
    dup_count = sum(r[2] - 1 for r in dupes)

    print(f"Total chunks: {total}")
    print(f"Duplicate chunks (extra copies): {dup_count}")
    print(f"Unique chunks: {total - dup_count}")
    print(f"\nDistribution:")
    for r in rows:
        print(f"  {r[0]:25s} count={r[1]:2d} avg_imp={r[2]} avg_access={r[3]}")

    if dupes:
        print(f"\nDuplicate groups ({len(dupes)}):")
        for summary, chunk_type, cnt, _ in dupes:
            print(f"  [{cnt}x] ({chunk_type}) {summary[:70]}")


def main():
    args = sys.argv[1:]
    conn = sqlite3.connect(str(STORE_DB))

    if "--verify" in args:
        issues = verify(conn)
        if issues:
            print("ISSUES FOUND:")
            for i in issues:
                print(f"  - {i}")
            sys.exit(1)
        else:
            print("PASS: No issues found")
            total = conn.execute("SELECT COUNT(*) FROM memory_chunks").fetchone()[0]
            print(f"Total chunks: {total}")
            sys.exit(0)

    report(conn)

    if "--fix" in args:
        print("\n--- Executing cleanup ---")
        deleted, kept = dedup_chunks(conn, dry_run=False)
        print(f"Deleted {deleted} duplicate chunks")
        print(f"FTS5 index rebuilt")
        issues = verify(conn)
        if issues:
            print(f"POST-FIX ISSUES: {issues}")
        else:
            total = conn.execute("SELECT COUNT(*) FROM memory_chunks").fetchone()[0]
            print(f"PASS: {total} chunks, no duplicates, FTS5 consistent")
    else:
        print("\n(dry-run mode, use --fix to execute)")

    conn.close()


if __name__ == "__main__":
    main()
