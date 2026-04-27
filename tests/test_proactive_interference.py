"""
test_proactive_interference.py — 迭代317：前摄干扰控制单元测试

验证：
  1. detect_conflict 能识别语义矛盾（新旧知识互相否定）
  2. detect_conflict 不误判相似（但不矛盾）的chunk
  3. supersede_chunk 创建版本对，旧chunk降权
  4. 检索时旧版本chunk得分低于新版本
  5. knowledge_versions 表正确记录演化关系
  6. 边界：空输入/不存在ID安全返回

认知科学基础：
  前摄干扰（Proactive Interference）— 旧知识干扰新知识的学习和检索
  Bartlett 1932图式同化 — 新知识依附已有框架，但框架更新时必须明确标记旧框架失效
  OS 类比：Linux kernel module versioning — 加载新模块版本时标记旧版本为 MODULE_STATE_GOING
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
    detect_conflict, supersede_chunk, get_superseded_ids,
)


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _make_chunk(cid, summary, chunk_type="decision", importance=0.7,
                project="test", access_count=2):
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    return {
        "id": cid,
        "created_at": now,
        "updated_at": now,
        "project": project,
        "source_session": "s1",
        "chunk_type": chunk_type,
        "info_class": "world",
        "content": f"[{chunk_type}] {summary}",
        "summary": summary,
        "tags": [chunk_type, project],
        "importance": importance,
        "retrievability": 0.5,
        "last_accessed": now,
        "access_count": access_count,
        "oom_adj": 0,
        "lru_gen": 0,
        "stability": 1.4,
        "raw_snippet": "",
        "encoding_context": {},
    }


# ══════════════════════════════════════════════════════════════════════
# 1. detect_conflict 基础测试
# ══════════════════════════════════════════════════════════════════════

def test_detect_conflict_finds_negation(conn):
    """新chunk用否定词明确反对旧chunk → 应检测为冲突。"""
    insert_chunk(conn, _make_chunk("c_old", "使用 BM25 检索，性能足够"))
    conn.commit()

    # 明确否定旧决策
    conflicts = detect_conflict(conn, "不使用 BM25 检索，改用向量检索", "decision", "test")
    assert len(conflicts) >= 1, f"应检测到冲突，got {conflicts}"
    assert "c_old" in conflicts, f"c_old 应在冲突列表中，got {conflicts}"


def test_detect_conflict_finds_replace_pattern(conn):
    """'改用/换成/替代' 关键词触发冲突检测。"""
    insert_chunk(conn, _make_chunk("c_old", "采用 Redis 缓存方案"))
    conn.commit()

    conflicts = detect_conflict(conn, "放弃 Redis，改用本地内存缓存", "decision", "test")
    assert "c_old" in conflicts, f"应检测到 Redis 替换冲突，got {conflicts}"


def test_detect_conflict_no_false_positive_similar(conn):
    """仅相似但无矛盾的chunk不被标记为冲突。"""
    insert_chunk(conn, _make_chunk("c_ok", "BM25 检索模块性能良好"))
    conn.commit()

    # 表达相似但没有否定关系
    conflicts = detect_conflict(conn, "BM25 检索模块继续使用", "decision", "test")
    assert len(conflicts) == 0, f"相似非矛盾chunk不应检测为冲突，got {conflicts}"


def test_detect_conflict_finds_opposite_conclusion(conn):
    """旧决策说'推荐X'，新chunk说'不推荐X' → 冲突。"""
    insert_chunk(conn, _make_chunk("c_old", "推荐使用 PostgreSQL 作为主存储"))
    conn.commit()

    conflicts = detect_conflict(conn, "不推荐 PostgreSQL，选择 SQLite 更轻量", "decision", "test")
    assert len(conflicts) >= 1, f"应检测到推荐/不推荐冲突，got {conflicts}"


def test_detect_conflict_empty_db(conn):
    """空DB中检测冲突，安全返回空列表。"""
    conflicts = detect_conflict(conn, "使用 BM25 检索", "decision", "test")
    assert conflicts == [], f"空DB应返回空列表，got {conflicts}"


def test_detect_conflict_different_type_no_conflict(conn):
    """不同 chunk_type 之间不判断冲突（跨类型语义不可比）。"""
    insert_chunk(conn, _make_chunk("c_design", "使用异步IO设计", chunk_type="design_constraint"))
    conn.commit()

    # decision 类型新chunk，不应与 design_constraint 类型旧chunk判冲突
    conflicts = detect_conflict(conn, "不使用异步IO", "decision", "test")
    assert len(conflicts) == 0, f"不同 chunk_type 不应判为冲突，got {conflicts}"


# ══════════════════════════════════════════════════════════════════════
# 2. supersede_chunk 版本对测试
# ══════════════════════════════════════════════════════════════════════

def test_supersede_creates_version_pair(conn):
    """supersede_chunk 在 knowledge_versions 创建版本对记录。"""
    insert_chunk(conn, _make_chunk("c_old", "旧决策：使用 BM25"))
    conn.commit()

    new_id = supersede_chunk(conn, "c_old", "c_new",
                              "新决策：使用向量检索替代 BM25",
                              project="test", session_id="sess1")

    row = conn.execute(
        "SELECT * FROM knowledge_versions WHERE old_chunk_id='c_old'"
    ).fetchone()
    assert row is not None, "knowledge_versions 应有版本对记录"
    assert row["new_chunk_id"] == "c_new", f"new_chunk_id 应为 c_new，got {row['new_chunk_id']}"
    assert row["reason"] is not None


def test_supersede_downgrades_old_chunk(conn):
    """supersede_chunk 将旧chunk的 importance 降权。"""
    insert_chunk(conn, _make_chunk("c_old", "旧方案", importance=0.85))
    conn.commit()

    supersede_chunk(conn, "c_old", "c_new", "新方案替代旧方案",
                    project="test", session_id="sess1")
    conn.commit()

    row = conn.execute(
        "SELECT importance, oom_adj FROM memory_chunks WHERE id='c_old'"
    ).fetchone()
    assert row["importance"] < 0.85, \
        f"旧chunk importance 应降权，got {row['importance']}"
    assert row["oom_adj"] > 0, \
        f"旧chunk oom_adj 应上调（更易淘汰），got {row['oom_adj']}"


def test_supersede_nonexistent_old_id(conn):
    """不存在的 old_id 安全处理，不抛异常。"""
    result = supersede_chunk(conn, "nonexistent", "c_new",
                              "新决策", project="test", session_id="sess")
    assert result is None or result == "c_new"


# ══════════════════════════════════════════════════════════════════════
# 3. get_superseded_ids — 检索时过滤旧版本
# ══════════════════════════════════════════════════════════════════════

def test_get_superseded_ids_returns_old_chunks(conn):
    """get_superseded_ids 返回已被取代的旧chunk ID集合。"""
    insert_chunk(conn, _make_chunk("c_old", "旧决策"))
    conn.commit()

    supersede_chunk(conn, "c_old", "c_new", "新决策取代旧决策",
                    project="test", session_id="sess")
    conn.commit()

    superseded = get_superseded_ids(conn, project="test")
    assert "c_old" in superseded, f"c_old 应在已取代集合中，got {superseded}"
    assert "c_new" not in superseded, f"c_new 不应在已取代集合中，got {superseded}"


def test_get_superseded_ids_empty_returns_empty_set(conn):
    """无版本对时返回空集合。"""
    superseded = get_superseded_ids(conn, project="test")
    assert isinstance(superseded, (set, frozenset, list))
    assert len(superseded) == 0


# ══════════════════════════════════════════════════════════════════════
# 4. 端到端：旧版本chunk在检索候选中被降权
# ══════════════════════════════════════════════════════════════════════

def test_superseded_chunk_lower_score(conn):
    """被取代的旧chunk得分低于新chunk（通过 importance 下降体现）。"""
    insert_chunk(conn, _make_chunk("c_old", "方案A：使用同步IO", importance=0.80))
    insert_chunk(conn, _make_chunk("c_new", "方案B：使用异步IO替代同步IO", importance=0.80))
    conn.commit()

    supersede_chunk(conn, "c_old", "c_new", "异步IO替代同步IO",
                    project="test", session_id="sess")
    conn.commit()

    old_imp = conn.execute(
        "SELECT importance FROM memory_chunks WHERE id='c_old'"
    ).fetchone()["importance"]
    new_imp = conn.execute(
        "SELECT importance FROM memory_chunks WHERE id='c_new'"
    ).fetchone()["importance"]

    assert new_imp > old_imp, \
        f"新版本 importance ({new_imp}) 应 > 旧版本 ({old_imp})"
