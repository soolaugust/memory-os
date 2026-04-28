"""
test_reconsolidate.py — iter382: Reconsolidate Stop Hook 单元测试

覆盖：
  RC1: reconsolidate() 对匹配 query 的 chunk importance 上调
  RC2: reconsolidate() 上调量与 query 匹配深度（Jaccard）成正比
  RC3: reconsolidate() importance 不超过 max_importance=0.98 上限
  RC4: reconsolidate() 对空 recalled_ids 返回 0（安全）
  RC5: reconsolidate() 对空 query 返回 0（安全）
  RC6: reconsolidate() 不影响不在 recalled_ids 中的 chunk
  RC7: reconsolidate() 返回更新 chunk 数量

认知科学依据：Nader et al. (2000) 再巩固理论 — 记忆每次被检索后进入不稳定窗口，
随后以更新形式重新巩固（re-stabilization），重复匹配的召回 → importance 增强（LTP）。
"""
import sys
import sqlite3
import pytest
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import tmpfs  # noqa

from store_vfs import open_db, ensure_schema, reconsolidate
from store import insert_chunk


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _make_chunk(cid, summary, chunk_type="decision", project="test",
                importance=0.6, access_count=3):
    now = datetime.now(timezone.utc).isoformat()
    return {
        "id": cid,
        "created_at": now,
        "updated_at": now,
        "project": project,
        "source_session": "s1",
        "chunk_type": chunk_type,
        "info_class": "semantic",
        "content": f"[{chunk_type}] {summary}",
        "summary": summary,
        "tags": [chunk_type],
        "importance": importance,
        "retrievability": 0.8,
        "last_accessed": now,
        "access_count": access_count,
        "oom_adj": 0,
        "lru_gen": 0,
        "stability": importance * 2.0,
        "raw_snippet": "",
        "encoding_context": {},
    }


# ── RC1: 匹配 query → importance 上调 ─────────────────────────────────────────

def test_rc1_matching_query_boosts_importance(conn):
    """匹配 query 的 chunk importance 应上调。"""
    original_imp = 0.6
    insert_chunk(conn, _make_chunk("rc1", "BM25 检索召回 FTS5 索引",
                                   importance=original_imp))
    conn.commit()

    n = reconsolidate(conn, ["rc1"], query="BM25 检索 FTS5", project="test")
    conn.commit()

    assert n >= 1, f"应更新 ≥ 1 个 chunk，got {n}"
    row = conn.execute("SELECT importance FROM memory_chunks WHERE id='rc1'").fetchone()
    assert row["importance"] > original_imp, \
        f"importance 应上调，got {row['importance']} (original={original_imp})"


# ── RC2: 匹配深度越高，boost 越大 ─────────────────────────────────────────────

def test_rc2_deeper_match_larger_boost(conn):
    """更高的 Jaccard 重叠度 → 更大的 importance 提升。"""
    original_imp = 0.6
    # chunk A: 与 query 高度重叠
    insert_chunk(conn, _make_chunk("rc2a", "BM25 算法 FTS5 检索召回优化",
                                   importance=original_imp))
    # chunk B: 与 query 低度重叠（只有一个词）
    insert_chunk(conn, _make_chunk("rc2b", "PostgreSQL 数据库配置",
                                   importance=original_imp))
    conn.commit()

    query = "BM25 算法 FTS5 检索优化 召回 相关性排序"
    reconsolidate(conn, ["rc2a", "rc2b"], query=query, project="test")
    conn.commit()

    row_a = conn.execute("SELECT importance FROM memory_chunks WHERE id='rc2a'").fetchone()
    row_b = conn.execute("SELECT importance FROM memory_chunks WHERE id='rc2b'").fetchone()

    # chunk A 与 query 重叠更多，boost 应更大
    boost_a = row_a["importance"] - original_imp
    boost_b = row_b["importance"] - original_imp
    assert boost_a >= boost_b, \
        f"高重叠 chunk 应有更大 boost，a_boost={boost_a:.4f} b_boost={boost_b:.4f}"


# ── RC3: importance 不超过 max_importance 上限 ────────────────────────────────

def test_rc3_importance_capped_at_max(conn):
    """reconsolidate 不应让 importance 超过 max_importance=0.98。"""
    insert_chunk(conn, _make_chunk("rc3", "BM25 检索 FTS5 索引 score 排名",
                                   importance=0.97))  # 已接近上限
    conn.commit()

    reconsolidate(conn, ["rc3"], query="BM25 检索 FTS5 索引", project="test",
                  max_importance=0.98)
    conn.commit()

    row = conn.execute("SELECT importance FROM memory_chunks WHERE id='rc3'").fetchone()
    assert row["importance"] <= 0.98, \
        f"importance 不应超过 0.98，got {row['importance']}"


# ── RC4: 空 recalled_ids → 返回 0 ────────────────────────────────────────────

def test_rc4_empty_recalled_ids(conn):
    """空 recalled_ids 安全返回 0。"""
    assert reconsolidate(conn, [], query="BM25", project="test") == 0


# ── RC5: 空 query → 返回 0 ───────────────────────────────────────────────────

def test_rc5_empty_query(conn):
    """空 query 安全返回 0。"""
    insert_chunk(conn, _make_chunk("rc5", "BM25 检索"))
    conn.commit()
    assert reconsolidate(conn, ["rc5"], query="", project="test") == 0


# ── RC6: 不在 recalled_ids 中的 chunk 不受影响 ───────────────────────────────

def test_rc6_unreferenced_chunk_unchanged(conn):
    """不在 recalled_ids 中的 chunk importance 不应改变。"""
    original_imp = 0.6
    insert_chunk(conn, _make_chunk("rc6_recalled", "BM25 检索排名",
                                   importance=original_imp))
    insert_chunk(conn, _make_chunk("rc6_other", "BM25 检索排名",  # 相同 summary
                                   importance=original_imp))
    conn.commit()

    reconsolidate(conn, ["rc6_recalled"], query="BM25 检索 排名", project="test")
    conn.commit()

    row = conn.execute("SELECT importance FROM memory_chunks WHERE id='rc6_other'").fetchone()
    assert row["importance"] == original_imp, \
        f"未被召回的 chunk 不应被修改，got {row['importance']}"


# ── RC7: 返回更新 chunk 数量 ──────────────────────────────────────────────────

def test_rc7_returns_updated_count(conn):
    """reconsolidate 返回实际更新的 chunk 数量。"""
    insert_chunk(conn, _make_chunk("rc7a", "BM25 检索召回"))
    insert_chunk(conn, _make_chunk("rc7b", "FTS5 索引查询"))
    conn.commit()

    n = reconsolidate(conn, ["rc7a", "rc7b"], query="BM25 FTS5 检索", project="test")
    # 至少有一个 chunk 与 query 有重叠，应返回 ≥ 1
    assert n >= 1, f"应更新 ≥ 1 个 chunk，got {n}"
