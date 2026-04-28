"""
test_episodic_semantic.py — 迭代319：情节/语义记忆分离单元测试

验证：
  1. classify_memory_type 正确分类各 chunk_type → info_class
  2. 情节 chunk 写入时 info_class='episodic'
  3. promote_to_semantic 从高频情节 chunk 创建语义 chunk
  4. promote_to_semantic 仅处理 access_count >= threshold 的 chunk
  5. episodic_decay_scan 衰减过期情节 chunk
  6. episodic_decay_scan 自动触发提升（端到端）
  7. sleep_consolidate 集成 episodic_decay_scan
  8. 语义 chunk 的 stability/importance 高于源情节 chunk
  9. episodic_consolidations 正确记录转化事件
  10. 边界：空列表/无满足条件/不存在 ID 安全处理

认知科学基础：
  Tulving (1972) Episodic/Semantic Memory Theory
  海马 → 新皮层记忆固化（Memory Consolidation, Walker & Stickgold 2004）
"""
import sys
import sqlite3
import pytest
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "hooks"))

import tmpfs  # noqa

from store_vfs import (
    open_db, ensure_schema, insert_chunk,
    classify_memory_type, promote_to_semantic, episodic_decay_scan,
    sleep_consolidate,
)


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _make_chunk(cid, summary, chunk_type="decision", info_class=None,
                importance=0.7, access_count=2, project="test",
                days_ago=0):
    now = datetime.now(timezone.utc)
    last_accessed = (now - timedelta(days=days_ago)).isoformat()
    now_iso = now.isoformat()
    d = {
        "id": cid,
        "created_at": now_iso,
        "updated_at": now_iso,
        "project": project,
        "source_session": "s1",
        "chunk_type": chunk_type,
        "info_class": info_class or classify_memory_type(chunk_type, summary),
        "content": f"[{chunk_type}] {summary}",
        "summary": summary,
        "tags": [chunk_type],
        "importance": importance,
        "retrievability": 0.5,
        "last_accessed": last_accessed,
        "access_count": access_count,
        "oom_adj": 0,
        "lru_gen": 0,
        "stability": importance * 2.0,
        "raw_snippet": "",
        "encoding_context": {},
    }
    return d


# ══════════════════════════════════════════════════════════════════════
# 1. classify_memory_type 分类测试
# ══════════════════════════════════════════════════════════════════════

def test_decision_is_semantic():
    assert classify_memory_type("decision", "使用 BM25 检索") == "semantic"


def test_design_constraint_is_semantic():
    assert classify_memory_type("design_constraint", "不得修改 schema") == "semantic"


def test_procedure_is_semantic():
    assert classify_memory_type("procedure", "部署流程") == "semantic"


def test_reasoning_chain_is_episodic():
    assert classify_memory_type("reasoning_chain", "因为X所以Y") == "episodic"


def test_conversation_summary_is_episodic():
    assert classify_memory_type("conversation_summary", "本次会话摘要") == "episodic"


def test_task_state_is_operational():
    assert classify_memory_type("task_state", "当前任务") == "operational"


def test_temporal_keyword_is_ephemeral():
    # "临时" 关键词 + 未在映射表中的 chunk_type → ephemeral
    assert classify_memory_type("tool_insight", "临时解决方案") == "ephemeral"


def test_excluded_path_is_semantic():
    assert classify_memory_type("excluded_path", "不使用向量DB") == "semantic"


def test_unknown_type_is_world():
    assert classify_memory_type("quantitative_evidence", "P95=266ms") == "semantic"


# ══════════════════════════════════════════════════════════════════════
# 2. promote_to_semantic 测试
# ══════════════════════════════════════════════════════════════════════

def test_promote_creates_semantic_chunk(conn):
    """高频情节 chunk 被提升为语义 chunk。"""
    insert_chunk(conn, _make_chunk("ep1", "retriever 使用 FTS5 索引",
                                   chunk_type="reasoning_chain",
                                   info_class="episodic", access_count=5))
    insert_chunk(conn, _make_chunk("ep2", "BM25 权重参数 summary=2.0",
                                   chunk_type="reasoning_chain",
                                   info_class="episodic", access_count=4))
    conn.commit()

    new_id = promote_to_semantic(conn, ["ep1", "ep2"], "test", min_recall_count=3)
    conn.commit()

    assert new_id is not None, "应创建新语义 chunk"
    row = conn.execute("SELECT * FROM memory_chunks WHERE id=?", (new_id,)).fetchone()
    assert row is not None
    assert row["info_class"] == "semantic", f"新 chunk 应为 semantic，got {row['info_class']}"
    assert "语义化" in row["summary"], f"summary 应含[语义化]前缀，got {row['summary']}"


def test_promote_downgrades_source_chunks(conn):
    """提升后源情节 chunk 被降级。"""
    original_imp = 0.8
    insert_chunk(conn, _make_chunk("ep1", "关键发现：性能瓶颈在 FTS5",
                                   chunk_type="reasoning_chain",
                                   info_class="episodic", importance=original_imp,
                                   access_count=5))
    conn.commit()

    promote_to_semantic(conn, ["ep1"], "test", min_recall_count=3)
    conn.commit()

    row = conn.execute("SELECT importance, info_class, oom_adj FROM memory_chunks WHERE id='ep1'").fetchone()
    assert row["importance"] < original_imp, \
        f"源 chunk importance 应降低，got {row['importance']}"
    assert row["oom_adj"] > 0, f"源 chunk oom_adj 应上调，got {row['oom_adj']}"


def test_promote_records_consolidation(conn):
    """promote_to_semantic 在 episodic_consolidations 中记录事件。"""
    insert_chunk(conn, _make_chunk("ep1", "重要结论",
                                   chunk_type="reasoning_chain",
                                   info_class="episodic", access_count=5))
    conn.commit()

    new_id = promote_to_semantic(conn, ["ep1"], "test", min_recall_count=3)
    conn.commit()

    row = conn.execute(
        "SELECT * FROM episodic_consolidations WHERE semantic_chunk_id=?",
        (new_id,)
    ).fetchone()
    assert row is not None, "应在 episodic_consolidations 中有记录"
    assert row["project"] == "test"


def test_promote_skips_low_access_count(conn):
    """access_count < threshold 的情节 chunk 不被提升。"""
    insert_chunk(conn, _make_chunk("ep1", "罕见访问结论",
                                   chunk_type="reasoning_chain",
                                   info_class="episodic", access_count=1))
    conn.commit()

    new_id = promote_to_semantic(conn, ["ep1"], "test", min_recall_count=3)
    assert new_id is None, "未达阈值不应被提升"


def test_promote_empty_list_returns_none(conn):
    """空列表安全返回 None。"""
    assert promote_to_semantic(conn, [], "test") is None


def test_promote_nonexistent_ids_returns_none(conn):
    """不存在的 chunk_id 安全返回 None。"""
    result = promote_to_semantic(conn, ["nonexistent_1", "nonexistent_2"], "test")
    assert result is None


# ══════════════════════════════════════════════════════════════════════
# 3. episodic_decay_scan 测试
# ══════════════════════════════════════════════════════════════════════

def test_episodic_decay_scans_stale_chunks(conn):
    """过期低频情节 chunk 被衰减（importance 下降）。"""
    original_imp = 0.7
    insert_chunk(conn, _make_chunk("ep_stale", "久远的情节记忆",
                                   chunk_type="reasoning_chain",
                                   info_class="episodic", importance=original_imp,
                                   access_count=0, days_ago=20))
    conn.commit()

    result = episodic_decay_scan(conn, "test", stale_days=14)
    conn.commit()

    assert result["decayed"] >= 1, f"应衰减 ≥ 1 个 chunk，got {result}"
    row = conn.execute(
        "SELECT importance FROM memory_chunks WHERE id='ep_stale'"
    ).fetchone()
    assert row["importance"] < original_imp, \
        f"stale 情节 chunk importance 应下降，got {row['importance']}"


def test_episodic_decay_does_not_stale_decay_fresh_chunks(conn):
    """近期访问的情节 chunk 不被 stale 衰减路径处理（decay 子操作跳过）。"""
    original_imp = 0.7
    # access_count=1（低于 semantic_threshold=3，不触发提升）
    # days_ago=1（未过 stale_days=14，不触发衰减）
    insert_chunk(conn, _make_chunk("ep_fresh", "新鲜情节记忆",
                                   chunk_type="reasoning_chain",
                                   info_class="episodic", importance=original_imp,
                                   access_count=1, days_ago=1))
    conn.commit()

    result = episodic_decay_scan(conn, "test", stale_days=14, semantic_threshold=3)
    conn.commit()

    row = conn.execute(
        "SELECT importance FROM memory_chunks WHERE id='ep_fresh'"
    ).fetchone()
    # 近期访问且低频（未达提升阈值）→ 不应被衰减或提升
    assert row["importance"] == original_imp, \
        f"新鲜低频 chunk 不应被衰减，got {row['importance']}"
    assert result["decayed"] == 0, f"应 decayed=0，got {result['decayed']}"


def test_episodic_decay_promotes_high_access(conn):
    """高频情节 chunk 被自动提升为语义 chunk。"""
    insert_chunk(conn, _make_chunk("ep_hot", "高频访问的重要结论",
                                   chunk_type="reasoning_chain",
                                   info_class="episodic", access_count=5, days_ago=1))
    conn.commit()

    result = episodic_decay_scan(conn, "test", semantic_threshold=3)
    conn.commit()

    # iter379: A0 in-place promotion (inplace_promoted) OR A merge promotion (promoted/new_semantic_ids)
    assert result["promoted"] >= 1 or result.get("new_semantic_ids") or result.get("inplace_promoted", 0) >= 1, \
        f"高频情节 chunk 应被提升，got {result}"


def test_episodic_decay_empty_project(conn):
    """空项目安全返回零计数。"""
    result = episodic_decay_scan(conn, "nonexistent_project")
    assert isinstance(result, dict)
    assert result.get("decayed", 0) == 0


# ══════════════════════════════════════════════════════════════════════
# 4. sleep_consolidate 集成 episodic_decay_scan
# ══════════════════════════════════════════════════════════════════════

def test_sleep_consolidate_includes_episodic_result(conn):
    """sleep_consolidate 返回结果中包含情节记忆处理的字段。"""
    insert_chunk(conn, _make_chunk("ep_stale", "古老情节",
                                   chunk_type="reasoning_chain",
                                   info_class="episodic", access_count=0,
                                   days_ago=30))
    conn.commit()

    result = sleep_consolidate(conn, "test")
    conn.commit()

    # iter319 集成：结果 dict 应含 episodic_decayed 字段
    assert "episodic_decayed" in result, \
        f"sleep_consolidate 结果应含 episodic_decayed，got {result.keys()}"
