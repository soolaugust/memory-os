#!/usr/bin/env python3
"""
test_db_hygiene.py — 迭代59：DB Hygiene + Extractor Full-type KSM Dedup 测试
OS 类比：fsck 自检测试 + KSM dedup 验证
"""
import sys
import os
import json
import sqlite3
from pathlib import Path

# tmpfs 隔离（必须在 store import 前）
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))
import tmpfs

from store import open_db, ensure_schema, insert_chunk, already_exists, merge_similar, dmesg_log, DMESG_DEBUG
from schema import MemoryChunk

# 导入 db_hygiene 工具
sys.path.insert(0, str(_ROOT / "tools"))
from db_hygiene import find_duplicates, dedup_chunks, verify

PASS = 0
FAIL = 0


def _result(name, ok, detail=""):
    global PASS, FAIL
    tag = "PASS" if ok else "FAIL"
    if not ok:
        FAIL += 1
    else:
        PASS += 1
    print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""))


def _fresh_db():
    """每个测试用全新的空 DB（隔离测试间数据）"""
    conn = open_db()
    ensure_schema(conn)
    conn.execute("DELETE FROM memory_chunks")
    conn.execute("INSERT INTO memory_chunks_fts(memory_chunks_fts) VALUES('rebuild')")
    conn.commit()
    return conn


def _make_chunk(summary, chunk_type="decision", project="test", importance=0.8):
    return MemoryChunk(
        project=project,
        source_session="test-session",
        chunk_type=chunk_type,
        content=f"[{chunk_type}] {summary}",
        summary=summary,
        tags=[chunk_type, project],
        importance=importance,
        retrievability=0.3,
    ).to_dict()


def test_find_duplicates():
    """find_duplicates 能正确检测重复组"""
    print("\n=== test_find_duplicates ===")
    conn = _fresh_db()

    for _ in range(3):
        insert_chunk(conn, _make_chunk("选择 React 而非 Vue", "decision"))
    insert_chunk(conn, _make_chunk("选择 React 而非 Vue", "reasoning_chain"))
    conn.commit()

    dupes = find_duplicates(conn)
    _result("检测到重复组", len(dupes) == 1, f"groups={len(dupes)}")
    if dupes:
        _result("重复数量正确", dupes[0][2] == 3, f"cnt={dupes[0][2]}")
    conn.close()


def test_dedup_dry_run():
    """dry-run 模式不修改数据"""
    print("\n=== test_dedup_dry_run ===")
    conn = _fresh_db()

    for _ in range(3):
        insert_chunk(conn, _make_chunk("dry-run 测试决策", "decision"))
    conn.commit()

    before = conn.execute("SELECT COUNT(*) FROM memory_chunks").fetchone()[0]
    deleted, kept = dedup_chunks(conn, dry_run=True)
    after = conn.execute("SELECT COUNT(*) FROM memory_chunks").fetchone()[0]

    _result("dry-run 报告删除数", deleted == 2, f"deleted={deleted}")
    _result("dry-run 不修改数据", before == after, f"before={before} after={after}")
    conn.close()


def test_dedup_fix():
    """fix 模式正确去重，保留 access_count 最高的"""
    print("\n=== test_dedup_fix ===")
    conn = _fresh_db()

    for i in range(3):
        insert_chunk(conn, _make_chunk("fix 测试决策", "decision"))
    conn.commit()

    ids = [r[0] for r in conn.execute(
        "SELECT id FROM memory_chunks WHERE summary='fix 测试决策' ORDER BY created_at"
    ).fetchall()]
    conn.execute("UPDATE memory_chunks SET access_count=10 WHERE id=?", (ids[1],))
    conn.commit()

    deleted, kept = dedup_chunks(conn, dry_run=False)
    _result("删除正确数量", deleted == 2, f"deleted={deleted}")

    remaining = conn.execute(
        "SELECT id, access_count FROM memory_chunks WHERE summary='fix 测试决策'"
    ).fetchall()
    _result("只保留 1 条", len(remaining) == 1, f"remaining={len(remaining)}")
    if remaining:
        _result("保留 access_count 最高的", remaining[0][1] == 10,
                f"access_count={remaining[0][1]}")
    conn.close()


def test_verify_clean():
    """清理后 verify 通过"""
    print("\n=== test_verify_clean ===")
    conn = _fresh_db()

    insert_chunk(conn, _make_chunk("唯一决策A", "decision"))
    insert_chunk(conn, _make_chunk("唯一决策B", "reasoning_chain"))
    conn.commit()

    issues = verify(conn)
    _result("无重复时 verify 通过", len(issues) == 0, f"issues={issues}")
    conn.close()


def test_verify_detects_dupes():
    """verify 能检测残留重复"""
    print("\n=== test_verify_detects_dupes ===")
    conn = _fresh_db()

    insert_chunk(conn, _make_chunk("重复检测", "decision"))
    insert_chunk(conn, _make_chunk("重复检测", "decision"))
    conn.commit()

    issues = verify(conn)
    _result("检测到重复", any("duplicate" in i.lower() for i in issues),
            f"issues={issues}")
    conn.close()


def test_already_exists_with_chunk_type():
    """already_exists 传 chunk_type 精确匹配"""
    print("\n=== test_already_exists_with_chunk_type ===")
    conn = _fresh_db()

    insert_chunk(conn, _make_chunk("类型精确匹配测试", "decision"))
    conn.commit()

    exists_same = already_exists(conn, "类型精确匹配测试", chunk_type="decision")
    _result("同类型检测到存在", exists_same)

    exists_diff = already_exists(conn, "类型精确匹配测试", chunk_type="reasoning_chain")
    _result("不同类型检测为不存在", not exists_diff)

    exists_any = already_exists(conn, "类型精确匹配测试")
    _result("不传类型时向后兼容检测到", exists_any)
    conn.close()


def test_extractor_write_chunk_dedup():
    """extractor _write_chunk 路径的全类型去重"""
    print("\n=== test_extractor_write_chunk_dedup ===")
    conn = _fresh_db()

    summary = "使用 bigram 分词而非 jieba"
    chunk_type = "decision"

    insert_chunk(conn, _make_chunk(summary, chunk_type))
    conn.commit()
    count_1 = conn.execute("SELECT COUNT(*) FROM memory_chunks").fetchone()[0]

    if already_exists(conn, summary, chunk_type=chunk_type):
        skipped = True
    else:
        insert_chunk(conn, _make_chunk(summary, chunk_type))
        conn.commit()
        skipped = False

    count_2 = conn.execute("SELECT COUNT(*) FROM memory_chunks").fetchone()[0]
    _result("重复写入被阻止", skipped)
    _result("chunk 数量未增加", count_1 == count_2, f"before={count_1} after={count_2}")

    if already_exists(conn, summary, chunk_type="reasoning_chain"):
        cross_skipped = True
    else:
        insert_chunk(conn, _make_chunk(summary, "reasoning_chain"))
        conn.commit()
        cross_skipped = False

    count_3 = conn.execute("SELECT COUNT(*) FROM memory_chunks").fetchone()[0]
    _result("不同类型的相同 summary 允许写入", not cross_skipped)
    _result("跨类型写入正确增加", count_3 == count_2 + 1, f"after_cross={count_3}")
    conn.close()


def test_fts5_consistency_after_dedup():
    """去重后 FTS5 索引一致"""
    print("\n=== test_fts5_consistency_after_dedup ===")
    conn = _fresh_db()

    for _ in range(5):
        insert_chunk(conn, _make_chunk("FTS5一致性测试", "decision"))
    insert_chunk(conn, _make_chunk("独立记录", "reasoning_chain"))
    conn.commit()

    dedup_chunks(conn, dry_run=False)

    main_count = conn.execute("SELECT COUNT(*) FROM memory_chunks").fetchone()[0]
    fts_count = conn.execute("SELECT COUNT(*) FROM memory_chunks_fts").fetchone()[0]
    _result("FTS5 行数与主表一致", main_count == fts_count,
            f"main={main_count} fts={fts_count}")
    _result("主表无重复", main_count == 2, f"count={main_count}")
    conn.close()


if __name__ == "__main__":
    test_find_duplicates()
    test_dedup_dry_run()
    test_dedup_fix()
    test_verify_clean()
    test_verify_detects_dupes()
    test_already_exists_with_chunk_type()
    test_extractor_write_chunk_dedup()
    test_fts5_consistency_after_dedup()

    print(f"\n{'='*50}")
    print(f"Total: {PASS + FAIL}  PASS: {PASS}  FAIL: {FAIL}")
    if FAIL > 0:
        sys.exit(1)
    print("ALL PASS")
