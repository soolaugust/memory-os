"""
test_sleep_consolidate.py — 迭代327：sleep_consolidate / episodic_decay_scan 单元测试

验证：
  1. semantic_threshold=2（iter327 降低后）：access_count>=2 的情节 chunk 触发晋升
  2. semantic_threshold=2：access_count=1 的情节 chunk 不触发晋升
  3. promote_to_semantic 写入 episodic_consolidations 记录
  4. 晋升后原情节 chunk info_class 变为 'world'，importance 降低
  5. 晋升后新语义 chunk info_class='semantic'，summary 包含 '[语义化]' 前缀
  6. stale 情节 chunk（access_count<2 且 last_accessed > stale_days）被衰减
  7. sleep_consolidate 正确调用 episodic_decay_scan
  8. 不满足阈值的情节 chunk 不被晋升（保护）
"""
import sys
import sqlite3
import pytest
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "hooks"))

import tmpfs  # noqa

from store_vfs import ensure_schema, insert_chunk, episodic_decay_scan, sleep_consolidate


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _old_iso(days: int):
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


def _make_chunk(cid, chunk_type, summary, info_class="episodic", access_count=0,
                last_accessed=None, project="test", importance=0.80):
    now = _now_iso()
    return {
        "id": cid,
        "created_at": now,
        "updated_at": now,
        "project": project,
        "source_session": "s1",
        "chunk_type": chunk_type,
        "info_class": info_class,
        "content": f"[{chunk_type}] {summary}",
        "summary": summary,
        "tags": [chunk_type],
        "importance": importance,
        "retrievability": 0.2,
        "last_accessed": last_accessed or now,
        "access_count": access_count,
        "oom_adj": 0,
        "lru_gen": 0,
        "stability": importance * 2.0,
        "raw_snippet": "",
        "encoding_context": {},
    }


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


# ══════════════════════════════════════════════════════════════════════
# 1. episodic_decay_scan — 晋升路径
# ══════════════════════════════════════════════════════════════════════

def test_promote_episodic_with_access_count_2(conn):
    """access_count=2 >= semantic_threshold=2 → 触发晋升。"""
    chunk = _make_chunk("e1", "causal_chain", "因为 FTS5 token 不足导致召回率低", access_count=2)
    insert_chunk(conn, chunk)
    conn.commit()

    result = episodic_decay_scan(conn, project="test", semantic_threshold=2)
    assert result["promoted"] >= 1, f"access_count=2 应触发晋升，got promoted={result['promoted']}"
    assert len(result["new_semantic_ids"]) >= 1


def test_no_promote_episodic_with_access_count_1(conn):
    """access_count=1 < semantic_threshold=2 → 不触发晋升。"""
    chunk = _make_chunk("e2", "causal_chain", "因为 A 导致了 B", access_count=1)
    insert_chunk(conn, chunk)
    conn.commit()

    result = episodic_decay_scan(conn, project="test", semantic_threshold=2)
    assert result["promoted"] == 0, f"access_count=1 不应晋升，got promoted={result['promoted']}"


def test_promoted_chunk_info_class_downgraded(conn):
    """晋升后，原情节 chunk info_class 变为 'world'。"""
    chunk = _make_chunk("e3", "reasoning_chain", "这是因为 A 的根本原因是 B", access_count=3)
    insert_chunk(conn, chunk)
    conn.commit()

    episodic_decay_scan(conn, project="test", semantic_threshold=2)
    conn.commit()

    row = conn.execute("SELECT info_class, importance FROM memory_chunks WHERE id='e3'").fetchone()
    assert row["info_class"] == "world", f"原情节 chunk 应降级为 world，got {row['info_class']}"
    assert row["importance"] < 0.80, f"原情节 chunk importance 应降低，got {row['importance']}"


def test_new_semantic_chunk_created(conn):
    """晋升后，新语义 chunk 应存在且 info_class='semantic'。"""
    chunk = _make_chunk("e4", "causal_chain", "由于内存压力导致 OOM 触发", access_count=3)
    insert_chunk(conn, chunk)
    conn.commit()

    result = episodic_decay_scan(conn, project="test", semantic_threshold=2)
    conn.commit()

    assert result["new_semantic_ids"], "应有新语义 chunk ID"
    new_id = result["new_semantic_ids"][0]
    row = conn.execute(
        "SELECT info_class, summary FROM memory_chunks WHERE id=?", (new_id,)
    ).fetchone()
    assert row is not None, "新语义 chunk 应存在"
    assert row["info_class"] == "semantic"
    assert "[语义化]" in row["summary"]


def test_episodic_consolidations_recorded(conn):
    """晋升后 episodic_consolidations 表应有记录。"""
    chunk = _make_chunk("e5", "causal_chain", "根因是 X 导致了 Y 失败", access_count=4)
    insert_chunk(conn, chunk)
    conn.commit()

    episodic_decay_scan(conn, project="test", semantic_threshold=2)
    conn.commit()

    n = conn.execute("SELECT COUNT(*) FROM episodic_consolidations").fetchone()[0]
    assert n >= 1, f"episodic_consolidations 应有记录，got {n}"


# ══════════════════════════════════════════════════════════════════════
# 2. episodic_decay_scan — 衰减路径
# ══════════════════════════════════════════════════════════════════════

def test_stale_episodic_decayed(conn):
    """access_count<2 且 last_accessed > stale_days 的情节 chunk 被衰减。"""
    old_ts = _old_iso(20)  # 20天前
    chunk = _make_chunk("e6", "causal_chain", "旧的因果链，从未被访问",
                        access_count=0, last_accessed=old_ts, importance=0.80)
    insert_chunk(conn, chunk)
    conn.commit()

    result = episodic_decay_scan(conn, project="test", stale_days=14, semantic_threshold=2)
    conn.commit()

    assert result["decayed"] >= 1, f"stale chunk 应被衰减，got decayed={result['decayed']}"
    row = conn.execute("SELECT importance FROM memory_chunks WHERE id='e6'").fetchone()
    assert row["importance"] < 0.80, f"衰减后 importance 应降低，got {row['importance']}"


def test_recent_episodic_not_decayed(conn):
    """最近 7 天内 last_accessed 的情节 chunk 不被衰减。"""
    chunk = _make_chunk("e7", "causal_chain", "最近访问的因果链", access_count=0, importance=0.80)
    insert_chunk(conn, chunk)
    conn.commit()

    result = episodic_decay_scan(conn, project="test", stale_days=14, semantic_threshold=2)

    row = conn.execute("SELECT importance FROM memory_chunks WHERE id='e7'").fetchone()
    assert row["importance"] == 0.80, f"最近 chunk 不应被衰减，got {row['importance']}"


# ══════════════════════════════════════════════════════════════════════
# 3. sleep_consolidate 集成
# ══════════════════════════════════════════════════════════════════════

def test_sleep_consolidate_calls_episodic_scan(conn):
    """sleep_consolidate 执行后 result 包含 episodic_promoted 字段。"""
    chunk = _make_chunk("e8", "causal_chain", "因为 B 的根因，导致了 C 的失败", access_count=2)
    insert_chunk(conn, chunk)
    conn.commit()

    result = sleep_consolidate(conn, project="test")
    assert "episodic_promoted" in result, "sleep_consolidate 结果应包含 episodic_promoted"
    assert "episodic_decayed" in result


def test_sleep_consolidate_merges_similar_chunks(conn):
    """高相似 chunk 被合并（Jaccard ≥ 0.72）。"""
    # 两个几乎相同的 summary
    s1 = "FTS5 检索性能：P50 < 10ms，P99 < 50ms，满足要求"
    s2 = "FTS5 检索性能：P50 < 10ms，P99 < 50ms，达到目标"
    chunk1 = _make_chunk("m1", "decision", s1, info_class="semantic", importance=0.80)
    chunk2 = _make_chunk("m2", "decision", s2, info_class="semantic", importance=0.75)
    insert_chunk(conn, chunk1)
    insert_chunk(conn, chunk2)
    conn.commit()

    result = sleep_consolidate(conn, project="test", similarity_threshold=0.72)
    conn.commit()

    assert result["merged"] >= 1, f"高相似 chunk 应被合并，got merged={result['merged']}"
    # victim 的 importance 应变为 0
    victim = conn.execute(
        "SELECT importance FROM memory_chunks WHERE id='m2'"
    ).fetchone()
    assert victim["importance"] == 0, f"victim 的 importance 应为 0，got {victim['importance']}"
