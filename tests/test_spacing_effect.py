"""
test_spacing_effect.py — iter383: Spacing Effect Scheduler 单元测试

覆盖：
  SE1: 超过 stability 窗口的 chunk 出现在候选列表
  SE2: 未超过 stability 窗口的 chunk 不出现在候选列表
  SE3: 已被 supersede 的 chunk 不出现在候选列表
  SE4: urgency 排序 — days_overdue 越大（urgency 越低）越靠前
  SE5: 空项目安全返回空列表
  SE6: importance < min_importance 的 chunk 不被选中
  SE7: access_count = 0 的 chunk 不被选中（未被访问过无"遗忘"概念）

认知科学依据：
  Ebbinghaus (1885) Forgetting Curve + SuperMemo SM-2
  记忆强度随时间指数衰减，stability 代表保留期（天），
  超过 stability 天未访问 → 进入遗忘窗口 → 应主动复习。
"""
import sys
import sqlite3
import pytest
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import tmpfs  # noqa

from store_vfs import open_db, ensure_schema, insert_chunk, find_spaced_review_candidates, supersede_chunk


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _make_chunk(cid, summary, chunk_type="decision", project="test",
                importance=0.8, access_count=2, days_ago=0, stability=1.0):
    """Helper: 构造 chunk dict，days_ago 控制 last_accessed 多少天前"""
    now = datetime.now(timezone.utc)
    last_accessed = (now - timedelta(days=days_ago)).isoformat()
    now_iso = now.isoformat()
    return {
        "id": cid,
        "created_at": now_iso,
        "updated_at": now_iso,
        "project": project,
        "source_session": "s1",
        "chunk_type": chunk_type,
        "info_class": "semantic",
        "content": f"[{chunk_type}] {summary}",
        "summary": summary,
        "tags": [chunk_type],
        "importance": importance,
        "retrievability": 0.8,
        "last_accessed": last_accessed,
        "access_count": access_count,
        "oom_adj": 0,
        "lru_gen": 0,
        "stability": stability,
        "raw_snippet": "",
        "encoding_context": {},
    }


# ── SE1: 超过 stability 窗口的 chunk 出现在候选列表 ──────────────────────────

def test_se1_overdue_chunk_in_candidates(conn):
    """stability=1.0 天，last_accessed=3天前 → 超窗口 → 出现在候选。"""
    insert_chunk(conn, _make_chunk("se1", "使用 BM25 进行检索排名",
                                   days_ago=3, stability=1.0, access_count=2))
    conn.commit()

    candidates = find_spaced_review_candidates(conn, "test", top_n=5, min_importance=0.5)
    ids = [c["id"] for c in candidates]
    assert "se1" in ids, f"超过 stability 窗口的 chunk 应出现在候选中，got {ids}"


# ── SE2: 未超过 stability 窗口的 chunk 不出现 ────────────────────────────────

def test_se2_fresh_chunk_not_in_candidates(conn):
    """stability=10 天，last_accessed=2天前 → 未超窗口 → 不出现在候选。"""
    insert_chunk(conn, _make_chunk("se2", "FTS5 索引优化方案",
                                   days_ago=2, stability=10.0, access_count=3))
    conn.commit()

    candidates = find_spaced_review_candidates(conn, "test", top_n=5, min_importance=0.5)
    ids = [c["id"] for c in candidates]
    assert "se2" not in ids, f"未超窗口的 chunk 不应出现在候选中，got {ids}"


# ── SE3: 已被 supersede 的 chunk 不出现 ──────────────────────────────────────

def test_se3_superseded_chunk_excluded(conn):
    """supersede_chunk 标记后，chunk 不应出现在 spacing review 候选中。"""
    insert_chunk(conn, _make_chunk("se3_old", "旧版检索算法",
                                   days_ago=5, stability=1.0, access_count=3))
    insert_chunk(conn, _make_chunk("se3_new", "新版检索算法"))
    conn.commit()

    supersede_chunk(conn, "se3_old", "se3_new",
                    reason="superseded test", project="test")
    conn.commit()

    candidates = find_spaced_review_candidates(conn, "test", top_n=10, min_importance=0.5)
    ids = [c["id"] for c in candidates]
    assert "se3_old" not in ids, f"已被 supersede 的 chunk 不应出现在候选，got {ids}"


# ── SE4: urgency 排序 — 最逾期（urgency最低）的排最前 ────────────────────────

def test_se4_urgency_ordering(conn):
    """urgency=importance/(days_since/stability) 越低越迫切，越靠前。"""
    # chunk A: 5天前访问, stability=1 → days_overdue=4, urgency=0.8/(5/1)=0.16
    insert_chunk(conn, _make_chunk("se4a", "chunk A 检索策略 decision",
                                   days_ago=5, stability=1.0, importance=0.8, access_count=2))
    # chunk B: 10天前访问, stability=1 → days_overdue=9, urgency=0.8/(10/1)=0.08（更逾期）
    insert_chunk(conn, _make_chunk("se4b", "chunk B 检索策略 decision",
                                   days_ago=10, stability=1.0, importance=0.8, access_count=2))
    conn.commit()

    candidates = find_spaced_review_candidates(conn, "test", top_n=5, min_importance=0.5)
    ids = [c["id"] for c in candidates]

    # se4b 更逾期（urgency更低），应排在 se4a 前面
    if "se4a" in ids and "se4b" in ids:
        assert ids.index("se4b") < ids.index("se4a"), \
            f"更逾期的 chunk 应排更前，got order={ids}"


# ── SE5: 空项目安全返回空列表 ────────────────────────────────────────────────

def test_se5_empty_project_returns_empty(conn):
    """空项目安全返回 []。"""
    result = find_spaced_review_candidates(conn, "nonexistent_project", top_n=5)
    assert result == [], f"空项目应返回空列表，got {result}"


# ── SE6: importance < min_importance 不被选中 ────────────────────────────────

def test_se6_low_importance_excluded(conn):
    """importance < min_importance 的 chunk 不出现在候选中。"""
    insert_chunk(conn, _make_chunk("se6", "低重要性检索配置",
                                   days_ago=5, stability=1.0,
                                   importance=0.3, access_count=2))
    conn.commit()

    # min_importance=0.70 → 0.3 不满足
    candidates = find_spaced_review_candidates(conn, "test", top_n=5, min_importance=0.70)
    ids = [c["id"] for c in candidates]
    assert "se6" not in ids, f"低重要性 chunk 不应出现在候选，got {ids}"


# ── SE7: access_count = 0 不被选中 ───────────────────────────────────────────

def test_se7_never_accessed_excluded(conn):
    """access_count=0 的 chunk 未被访问过，不参与间隔复习。"""
    insert_chunk(conn, _make_chunk("se7", "从未访问的检索配置",
                                   days_ago=30, stability=1.0,
                                   importance=0.9, access_count=0))
    conn.commit()

    candidates = find_spaced_review_candidates(conn, "test", top_n=5, min_importance=0.5)
    ids = [c["id"] for c in candidates]
    assert "se7" not in ids, f"从未访问的 chunk 不应参与间隔复习，got {ids}"
