"""
test_cognitive_memory.py — 迭代311：认知记忆三机制单元测试

验证三个认知科学机制：
  A. Reconsolidation (Nader et al. 2000): 召回加深 → importance 上升
  B. Active Suppression (Anderson & Green 2001): 注入未用 → importance 下降
  C. Sleep Consolidation (Walker & Stickgold 2004): session 结束自动维护

OS 类比：
  A = ARC cache T2 晋升（频繁命中 → 淘汰优先级降低）
  B = vm.swappiness 主动换出（冷页面提前推出 RAM）
  C = KSM + pdflush（session 结束时合并相似页 + 稳定性衰减）
"""
import sys
import os
import sqlite3
import pytest
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "hooks"))

import tmpfs  # noqa

from store_vfs import (
    open_db, ensure_schema,
    reconsolidate, suppress_unused, sleep_consolidate,
)


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _insert_chunk(conn, cid, summary, chunk_type="decision",
                  importance=0.7, project="test",
                  last_accessed_days_ago=1, access_count=1, stability=7.0):
    now = datetime.now(timezone.utc)
    last_accessed = (now - timedelta(days=last_accessed_days_ago)).isoformat()
    created_at = (now - timedelta(days=last_accessed_days_ago + 1)).isoformat()
    conn.execute("""
        INSERT OR REPLACE INTO memory_chunks
            (id, summary, content, chunk_type, importance, project,
             last_accessed, access_count, created_at, updated_at,
             stability, retrievability, info_class, source_session,
             lru_gen, oom_adj)
        VALUES (?, ?, '', ?, ?, ?, ?, ?, ?, ?, ?, 0.5, 'world', 'test', 0, 0)
    """, (cid, summary, chunk_type, importance, project,
          last_accessed, access_count, created_at, created_at, stability))
    conn.commit()


# ══════════════════════════════════════════════════════════════════════════════
# A. Reconsolidation Tests
# ══════════════════════════════════════════════════════════════════════════════

def test_reconsolidate_boosts_importance(conn):
    """召回的 chunk 经过 reconsolidate 后 importance 上升。"""
    _insert_chunk(conn, "c1", "retriever BM25 检索模块", importance=0.6)

    # query 与 summary 有显著重叠 → 应获得较高 boost
    n = reconsolidate(conn, ["c1"], query="retriever BM25 检索", project="test")

    assert n == 1, f"应更新 1 个 chunk，实际: {n}"
    row = conn.execute("SELECT importance FROM memory_chunks WHERE id='c1'").fetchone()
    assert row["importance"] > 0.6, \
        f"importance 应上升，实际 {row['importance']}"


def test_reconsolidate_deeper_query_boosts_more(conn):
    """query 与 summary 重叠越高，importance boost 越大。"""
    _insert_chunk(conn, "c_high", "retriever BM25 检索模块性能优化", importance=0.6)
    _insert_chunk(conn, "c_low", "retriever BM25 检索模块性能优化", importance=0.6)

    # 高重叠 query（词汇完全命中）
    reconsolidate(conn, ["c_high"], query="retriever BM25 检索模块性能优化", project="test")
    # 低重叠 query（只有 1/4 词命中）
    reconsolidate(conn, ["c_low"], query="系统设计文档", project="test")

    high = conn.execute("SELECT importance FROM memory_chunks WHERE id='c_high'").fetchone()["importance"]
    low = conn.execute("SELECT importance FROM memory_chunks WHERE id='c_low'").fetchone()["importance"]

    assert high > low, \
        f"高重叠 query boost 应大于低重叠，got high={high} low={low}"


def test_reconsolidate_capped_at_max_importance(conn):
    """已经接近 max_importance 的 chunk 不会超过上限。"""
    _insert_chunk(conn, "c1", "检索模块", importance=0.97)

    reconsolidate(conn, ["c1"], query="检索模块", project="test", max_importance=0.98)

    row = conn.execute("SELECT importance FROM memory_chunks WHERE id='c1'").fetchone()
    assert row["importance"] <= 0.98, \
        f"importance 不应超过 max_importance=0.98，实际: {row['importance']}"


def test_reconsolidate_empty_input(conn):
    """空输入安全返回 0，不抛异常。"""
    n = reconsolidate(conn, [], query="anything", project="test")
    assert n == 0

    n = reconsolidate(conn, ["c1"], query="", project="test")
    assert n == 0


def test_reconsolidate_nonexistent_chunk(conn):
    """不存在的 chunk_id 不报错，返回 0。"""
    n = reconsolidate(conn, ["nonexistent_id"], query="some query", project="test")
    assert n == 0


# ══════════════════════════════════════════════════════════════════════════════
# B. Active Suppression Tests
# ══════════════════════════════════════════════════════════════════════════════

def test_suppress_unused_decreases_importance(conn):
    """chunk 被注入但回复中未使用 → importance 下降。"""
    _insert_chunk(conn, "c1", "存储层持久化设计 PostgreSQL", importance=0.7)

    # 回复中完全不涉及 chunk summary 的内容
    response = "我来帮你分析一下这个问题。根据已有信息，建议采用分层架构。"
    n = suppress_unused(conn, ["c1"], assistant_response=response, project="test")

    assert n == 1, f"应抑制 1 个 chunk，实际: {n}"
    row = conn.execute("SELECT importance FROM memory_chunks WHERE id='c1'").fetchone()
    assert row["importance"] < 0.7, \
        f"unused chunk importance 应下降，实际: {row['importance']}"


def test_suppress_skips_if_used_in_response(conn):
    """chunk 关键词出现在回复中 → 不应被抑制。"""
    _insert_chunk(conn, "c1", "retriever BM25 召回性能", importance=0.7)

    # 回复中提到了 retriever 和 BM25
    response = "对于 retriever 模块，BM25 召回性能在当前配置下表现良好，无需优化。"
    n = suppress_unused(conn, ["c1"], assistant_response=response, project="test")

    assert n == 0, f"chunk 已被使用，不应被抑制，实际 suppressed={n}"
    row = conn.execute("SELECT importance FROM memory_chunks WHERE id='c1'").fetchone()
    assert row["importance"] == pytest.approx(0.7, abs=0.001), \
        f"使用的 chunk importance 不应变化，实际: {row['importance']}"


def test_suppress_respects_min_importance(conn):
    """suppress 不会把 importance 降到 min_importance 以下。"""
    _insert_chunk(conn, "c1", "过时的存储模块文档", importance=0.08)

    response = "当前任务与记忆管理无关。"
    suppress_unused(conn, ["c1"], assistant_response=response,
                    project="test", penalty=0.05, min_importance=0.05)

    row = conn.execute("SELECT importance FROM memory_chunks WHERE id='c1'").fetchone()
    assert row["importance"] >= 0.05, \
        f"importance 不应低于 min_importance=0.05，实际: {row['importance']}"


def test_suppress_empty_input(conn):
    """空输入安全返回 0，不抛异常。"""
    n = suppress_unused(conn, [], assistant_response="some response", project="test")
    assert n == 0

    n = suppress_unused(conn, ["c1"], assistant_response="", project="test")
    assert n == 0


# ══════════════════════════════════════════════════════════════════════════════
# C. Sleep Consolidation Tests
# ══════════════════════════════════════════════════════════════════════════════

def test_sleep_consolidate_returns_dict(conn):
    """sleep_consolidate 安全返回 dict，不抛异常。"""
    result = sleep_consolidate(conn, project="test", session_id="test_session")
    assert isinstance(result, dict), f"应返回 dict，got {type(result)}"
    assert "merged" in result
    assert "boosted" in result
    assert "decayed" in result


def test_sleep_consolidate_merges_similar_chunks(conn):
    """Jaccard 相似度 >= threshold 的 chunk 被合并（victim 降为 ghost）。"""
    # 插入两个高度相似的 chunk
    _insert_chunk(conn, "c_orig", "BM25 检索算法优化方案分析", importance=0.8, access_count=3)
    _insert_chunk(conn, "c_dup",  "BM25 检索算法优化方案分析", importance=0.7, access_count=1)
    # 插入一个差异较大的 chunk（不应被合并）
    _insert_chunk(conn, "c_diff", "完全不相关的文档内容 XYZ", importance=0.6, access_count=2)

    result = sleep_consolidate(conn, project="test", session_id="sess",
                               similarity_threshold=0.72)

    assert result["merged"] >= 1, f"应合并至少 1 对，实际: {result['merged']}"

    # victim（c_dup，importance 更低）应被标记为 ghost（importance=0）
    victim = conn.execute(
        "SELECT importance, oom_adj FROM memory_chunks WHERE id='c_dup'"
    ).fetchone()
    assert victim["importance"] == 0, \
        f"被合并的 victim 应 importance=0，实际: {victim['importance']}"
    assert victim["oom_adj"] == 500, "victim 应标记 oom_adj=500（ghost）"

    # 差异较大的 chunk 不受影响
    diff = conn.execute(
        "SELECT importance FROM memory_chunks WHERE id='c_diff'"
    ).fetchone()
    assert diff["importance"] == pytest.approx(0.6, abs=0.05), \
        "不相似 chunk 不应被影响"


def test_sleep_consolidate_boosts_active_chunks(conn):
    """近期高访问 chunk stability × boost。"""
    # access_count >= 2，last_accessed 在最近 7 天内
    _insert_chunk(conn, "c_active", "活跃模块", access_count=5, stability=7.0,
                  last_accessed_days_ago=1)
    # access_count < 2，不应被 boost
    _insert_chunk(conn, "c_idle", "闲置模块", access_count=1, stability=7.0,
                  last_accessed_days_ago=1)

    result = sleep_consolidate(conn, project="test", session_id="sess",
                               stability_boost=1.15, active_days=7)

    assert result["boosted"] >= 1, f"应 boost >= 1 个 chunk，实际: {result['boosted']}"

    active = conn.execute(
        "SELECT stability FROM memory_chunks WHERE id='c_active'"
    ).fetchone()
    assert active["stability"] > 7.0, \
        f"活跃 chunk stability 应上升，实际: {active['stability']}"


def test_sleep_consolidate_decays_stale_chunks(conn):
    """长期未访问 chunk stability × decay。"""
    # last_accessed 在 31 天前（超过 stale_days=30）
    _insert_chunk(conn, "c_stale", "过时模块", access_count=0, stability=14.0,
                  last_accessed_days_ago=31)
    # 近期访问 chunk 不受 decay
    _insert_chunk(conn, "c_recent", "近期模块", access_count=3, stability=14.0,
                  last_accessed_days_ago=1)

    result = sleep_consolidate(conn, project="test", session_id="sess",
                               stability_decay=0.92, stale_days=30)

    assert result["decayed"] >= 1, f"应 decay >= 1 个 chunk，实际: {result['decayed']}"

    stale = conn.execute(
        "SELECT stability FROM memory_chunks WHERE id='c_stale'"
    ).fetchone()
    assert stale["stability"] < 14.0, \
        f"过时 chunk stability 应下降，实际: {stale['stability']}"


def test_sleep_consolidate_empty_db(conn):
    """空 DB 安全运行，返回全 0 dict。"""
    result = sleep_consolidate(conn, project="test", session_id="sess")
    assert result["merged"] == 0
    assert result["boosted"] == 0
    assert result["decayed"] == 0


def test_sleep_consolidate_max_merges_limit(conn):
    """max_merges 限制最大合并数量（防止大 DB 时性能退化）。"""
    # 插入 10 对高度相似 chunk
    for i in range(10):
        _insert_chunk(conn, f"c_orig_{i}", f"模块设计方案文档记录 {i}", importance=0.8)
        _insert_chunk(conn, f"c_dup_{i}",  f"模块设计方案文档记录 {i}", importance=0.6)

    result = sleep_consolidate(conn, project="test", session_id="sess",
                               similarity_threshold=0.80, max_merges=3)

    assert result["merged"] <= 3, \
        f"max_merges=3 时不应超过 3 次合并，实际: {result['merged']}"
