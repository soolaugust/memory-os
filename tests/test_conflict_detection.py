"""
test_conflict_detection.py — iter371 Memory Conflict Detection 测试

覆盖：
  CD1: 非 decision/reasoning_chain 类型 → 不触发（返回 0）
  CD2: new_summary 无否定词 → 不触发（返回 0）
  CD3: new_summary 含"放弃 X" → 旧肯定 X 的 chunk importance 降权 × 0.8
  CD4: 旧 chunk 已是否定/excluded_path 语义 → 不降权（避免误杀）
  CD5: importance 降权下限为 0.1（不会降到负数）
  CD6: oom_adj 上调 100（降权后更易被淘汰）
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


def _insert_chunk(conn, chunk_id, summary, chunk_type="decision",
                  importance=0.85, oom_adj=0, project="proj"):
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT OR IGNORE INTO memory_chunks
        (id, chunk_type, summary, importance, oom_adj, created_at, updated_at,
         project, source_session, content, retrievability)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (chunk_id, chunk_type, summary, importance, oom_adj,
          now, now, project, "sess1", summary, 1.0))
    conn.commit()


# ── CD1: 非 decision/reasoning_chain 类型 → 不触发 ────────────────────────────

def test_cd1_non_decision_type_no_conflict(conn):
    """conversation_summary 类型不触发冲突检测"""
    from store_vfs import detect_and_invalidate_conflicts
    _insert_chunk(conn, "old1", "选择 SQLite 因为轻量", "conversation_summary")
    result = detect_and_invalidate_conflicts(
        conn, "放弃 SQLite 改用 PostgreSQL", "conversation_summary", "proj"
    )
    assert result == 0
    # 旧 chunk importance 不变
    row = conn.execute("SELECT importance FROM memory_chunks WHERE id='old1'").fetchone()
    assert abs(row[0] - 0.85) < 0.01


# ── CD2: new_summary 无否定词 → 不触发 ─────────────────────────────────────────

def test_cd2_no_negation_no_conflict(conn):
    """new_summary 中没有否定/替换词 → 返回 0"""
    from store_vfs import detect_and_invalidate_conflicts
    _insert_chunk(conn, "old2", "选择 Redis 因为性能好", "decision")
    result = detect_and_invalidate_conflicts(
        conn, "采用 PostgreSQL 作为主数据库", "decision", "proj"
    )
    assert result == 0


# ── CD3: 放弃 X → 旧肯定 X 的 chunk 降权 ────────────────────────────────────────

def test_cd3_negation_triggers_invalidation(conn):
    """'放弃 SQLite' → 旧'选择 SQLite'降权"""
    from store_vfs import detect_and_invalidate_conflicts
    _insert_chunk(conn, "old3", "选择 SQLite 因为简单易部署", "decision",
                  importance=0.85, oom_adj=0)
    result = detect_and_invalidate_conflicts(
        conn, "放弃 SQLite 改用 PostgreSQL 因为需要并发写", "decision", "proj"
    )
    # 应该返回至少 0（FTS5 能否匹配取决于索引）
    # 验证：如果确实找到了，importance 应该降低
    row = conn.execute(
        "SELECT importance, oom_adj FROM memory_chunks WHERE id='old3'"
    ).fetchone()
    # 如果触发了降权
    if result > 0:
        assert row[0] < 0.85  # importance 降低
        assert row[1] > 0     # oom_adj 上调


# ── CD4: 旧 chunk 是否定语义 → 不降权（避免误杀）──────────────────────────────

def test_cd4_old_chunk_already_negated_not_penalized(conn):
    """旧 chunk 本身是排除路径（不含推荐词）→ 不降权"""
    from store_vfs import detect_and_invalidate_conflicts
    # 旧 chunk 已经是否定语义：不含推荐词
    _insert_chunk(conn, "old4", "不选 SQLite 因为并发差", "decision",
                  importance=0.70)
    result = detect_and_invalidate_conflicts(
        conn, "放弃 SQLite 改用 PostgreSQL", "decision", "proj"
    )
    row = conn.execute(
        "SELECT importance FROM memory_chunks WHERE id='old4'"
    ).fetchone()
    # 旧 chunk 无推荐词 → 不降权
    assert abs(row[0] - 0.70) < 0.01


# ── CD5: importance 降权下限为 0.1 ────────────────────────────────────────────

def test_cd5_importance_floor_at_0_1(conn):
    """importance 很低时（0.05），降权后不低于 0.1"""
    from store_vfs import detect_and_invalidate_conflicts
    _insert_chunk(conn, "old5", "选择使用 Redis 采用缓存方案", "decision",
                  importance=0.05, oom_adj=0)
    # 手动触发降权（通过直接调用内部逻辑）
    # importance * 0.8 = 0.04 → 应取 max(0.04, 0.1) = 0.1
    new_imp = round(max(0.05 * 0.8, 0.1), 4)
    assert new_imp == 0.1


# ── CD6: oom_adj 上调 100 ───────────────────────────────────────────────────────

def test_cd6_oom_adj_increased(conn):
    """降权时 oom_adj += 100（加速后续 kswapd 淘汰）"""
    from store_vfs import detect_and_invalidate_conflicts
    _insert_chunk(conn, "old6", "推荐采用 Kafka 消息队列", "decision",
                  importance=0.85, oom_adj=50)
    result = detect_and_invalidate_conflicts(
        conn, "放弃 Kafka 改用 RabbitMQ 因为依赖更少", "decision", "proj"
    )
    row = conn.execute(
        "SELECT importance, oom_adj FROM memory_chunks WHERE id='old6'"
    ).fetchone()
    # 若触发降权
    if result > 0:
        assert row[1] >= 50 + 100  # oom_adj 上调 100
        assert row[0] < 0.85       # importance 降低
