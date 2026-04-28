"""
test_chunk_coalescing.py — iter374 Chunk Coalescing (Slab Allocator) 测试

覆盖：
  CC1: 同主题 ≥3 小 chunk → 合并为1（anchor保留，其余删除）
  CC2: 同主题 2 chunk（不足 min_group=3）→ 不合并
  CC3: summary 超过 max_summary_len → 不纳入合并（大 chunk 不合并）
  CC4: anchor 保留最高 importance
  CC5: 合并后 composite content 包含所有 summary
"""
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

import pytest

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


@pytest.fixture()
def tmpdb(tmp_path):
    db_path = tmp_path / "test_store.db"
    os.environ["MEMORY_OS_DB"] = str(db_path)
    os.environ["MEMORY_OS_DIR"] = str(tmp_path)
    yield db_path
    os.environ.pop("MEMORY_OS_DB", None)
    os.environ.pop("MEMORY_OS_DIR", None)


@pytest.fixture()
def conn(tmpdb):
    from store_vfs import open_db, ensure_schema
    c = open_db(tmpdb)
    ensure_schema(c)
    yield c
    c.close()


def _insert(conn, chunk_id, summary, importance=0.5, chunk_type="conversation_summary",
            project="proj"):
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT OR IGNORE INTO memory_chunks
        (id, chunk_type, summary, importance, oom_adj, created_at, updated_at,
         project, source_session, content, retrievability)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (chunk_id, chunk_type, summary, importance, 0,
          now, now, project, "sess1", summary, 1.0))
    conn.commit()


# ── CC1: 同主题 ≥3 小 chunk → 合并为 1 ────────────────────────────────────────

def test_cc1_coalesce_three_small_chunks(conn):
    """3 个同主题小 chunk → 合并为 1，其余 2 个删除"""
    from store_vfs import coalesce_small_chunks
    _insert(conn, "c1", "端口配置讨论了3000", importance=0.5)
    _insert(conn, "c2", "端口配置询问了端口", importance=0.4)
    _insert(conn, "c3", "端口配置确认了端口", importance=0.6)

    result = coalesce_small_chunks(conn, "proj", min_group=3, max_summary_len=60)
    assert result == 1  # 1 个合并组

    # 只剩 1 个 chunk（anchor）
    rows = conn.execute(
        "SELECT id FROM memory_chunks WHERE chunk_type='conversation_summary' AND project='proj'"
    ).fetchall()
    assert len(rows) == 1


# ── CC2: 同主题 2 chunk（不足 min_group）→ 不合并 ──────────────────────────────

def test_cc2_less_than_min_group_no_coalesce(conn):
    """只有 2 个同主题 chunk（< min_group=3）→ 不合并"""
    from store_vfs import coalesce_small_chunks
    _insert(conn, "d1", "缓存策略用了Redis", importance=0.5)
    _insert(conn, "d2", "缓存策略选择了LRU", importance=0.5)

    result = coalesce_small_chunks(conn, "proj", min_group=3, max_summary_len=60)
    assert result == 0  # 不合并

    # 仍然有 2 个 chunk
    rows = conn.execute(
        "SELECT id FROM memory_chunks WHERE chunk_type='conversation_summary' AND project='proj'"
    ).fetchall()
    assert len(rows) == 2


# ── CC3: summary 超过 max_summary_len → 不纳入合并 ────────────────────────────

def test_cc3_long_summary_excluded(conn):
    """summary > max_summary_len 的 chunk 不被合并"""
    from store_vfs import coalesce_small_chunks
    long_summary = "A" * 80  # > 60
    _insert(conn, "e1", long_summary, importance=0.5)
    _insert(conn, "e2", long_summary + "B", importance=0.5)
    _insert(conn, "e3", long_summary + "C", importance=0.5)

    # 虽然有 3 个，但都超过 max_summary_len，所以前缀匹配后不会进入合并
    # 实际上：summary 超长 → WHERE LENGTH(summary) <= max_summary_len 过滤掉
    result = coalesce_small_chunks(conn, "proj", min_group=3, max_summary_len=60)
    assert result == 0

    # 3 个 chunk 仍然存在
    rows = conn.execute(
        "SELECT id FROM memory_chunks WHERE project='proj'"
    ).fetchall()
    assert len(rows) == 3


# ── CC4: anchor 保留最高 importance ──────────────────────────────────────────

def test_cc4_anchor_highest_importance(conn):
    """合并后 anchor 的 importance = max 值"""
    from store_vfs import coalesce_small_chunks
    _insert(conn, "f1", "数据库设计用了 SQLite", importance=0.5)
    _insert(conn, "f2", "数据库设计选了轻量", importance=0.9)  # 最高 importance
    _insert(conn, "f3", "数据库设计确认方案", importance=0.3)

    result = coalesce_small_chunks(conn, "proj", min_group=3, max_summary_len=60)
    assert result == 1

    row = conn.execute(
        "SELECT importance FROM memory_chunks WHERE project='proj'"
    ).fetchone()
    assert abs(row[0] - 0.9) < 0.01  # 最高 importance 保留


# ── CC5: composite content 包含所有原始 summary ────────────────────────────────

def test_cc5_composite_content_has_all_summaries(conn):
    """合并后的 content 包含所有原始 summary"""
    from store_vfs import coalesce_small_chunks
    s1 = "API设计用了REST风格"
    s2 = "API设计采用JSON格式"
    s3 = "API设计确认了版本"
    _insert(conn, "g1", s1, importance=0.5)
    _insert(conn, "g2", s2, importance=0.5)
    _insert(conn, "g3", s3, importance=0.5)

    result = coalesce_small_chunks(conn, "proj", min_group=3, max_summary_len=60)
    assert result == 1

    row = conn.execute(
        "SELECT content FROM memory_chunks WHERE project='proj'"
    ).fetchone()
    content = row[0]
    # composite content 应该包含所有 summary
    assert s1 in content
    assert s2 in content
    assert s3 in content
