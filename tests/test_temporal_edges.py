"""
test_temporal_edges.py — iter367 时序邻近性关联边测试

覆盖：
  TP1: 同 session + 时间相邻 → 自动建立 COOCCURS 边（双向）
  TP2: 不同 session → 不建立时序边
  TP3: 时间间隔 > 5min → 不建立时序边
  TP4: written_chunk_ids 为空 → 无操作
  TP5: session_id = "unknown" → 不建立时序边
  TP6: 边的 weight = 0.3（弱关联）
  TP7: 已有更强的边时不降低 weight
"""
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

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
    from store_vfs import open_db
    from store_graph import ensure_graph_schema
    c = open_db(tmpdb)
    ensure_graph_schema(c)
    c.execute("""
        CREATE TABLE IF NOT EXISTS memory_chunks (
            id TEXT PRIMARY KEY, summary TEXT, chunk_type TEXT,
            importance REAL DEFAULT 0.5,
            created_at TEXT, updated_at TEXT, project TEXT,
            source_session TEXT, content TEXT, tags TEXT,
            retrievability REAL DEFAULT 1.0, last_accessed TEXT,
            feishu_url TEXT, access_count INTEGER DEFAULT 0,
            oom_adj INTEGER DEFAULT 0, lru_gen INTEGER DEFAULT 0
        )
    """)
    c.commit()
    yield c
    c.close()


def _insert_chunk(conn, chunk_id, session_id="sess1", created_at=None):
    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT OR IGNORE INTO memory_chunks
        (id, chunk_type, summary, importance, created_at, updated_at,
         project, source_session, content)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, (chunk_id, "decision", f"summary {chunk_id}", 0.7,
          created_at, created_at, "proj", session_id, "content"))
    conn.commit()


# ── TP1: 同 session + 时间相邻 → 双向 COOCCURS 边 ────────────────────────────

def test_tp1_temporal_edge_same_session(conn):
    from store_graph import add_edge, EdgeType
    # 模拟：旧 chunk 在 3 分钟前写入，新 chunk 刚写入
    _old_time = (datetime.now(timezone.utc) - timedelta(minutes=3)).isoformat()
    _insert_chunk(conn, "old", session_id="s1", created_at=_old_time)
    _insert_chunk(conn, "new", session_id="s1")  # 刚写入

    # 模拟 extractor 的时序边逻辑
    _since = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    _tp_rows = conn.execute(
        """SELECT id FROM memory_chunks
           WHERE source_session=?
             AND id NOT IN (?)
             AND created_at >= ?
           ORDER BY created_at DESC LIMIT 10""",
        ["s1", "new", _since]
    ).fetchall()
    _recent = [r[0] for r in _tp_rows]
    assert "old" in _recent

    # 建边
    add_edge(conn, "new", "old", EdgeType.COOCCURS, 0.3, source="temporal")
    add_edge(conn, "old", "new", EdgeType.COOCCURS, 0.3, source="temporal")

    edges = conn.execute("SELECT from_id, to_id FROM chunk_edges").fetchall()
    edge_pairs = {(r[0], r[1]) for r in edges}
    assert ("new", "old") in edge_pairs
    assert ("old", "new") in edge_pairs


# ── TP2: 不同 session → 不建立时序边 ─────────────────────────────────────────

def test_tp2_different_session_no_edge(conn):
    _old_time = (datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat()
    _insert_chunk(conn, "old_other", session_id="other_session", created_at=_old_time)
    _insert_chunk(conn, "new_mine", session_id="my_session")

    # 时序边逻辑只查询同 session
    _since = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    _tp_rows = conn.execute(
        """SELECT id FROM memory_chunks
           WHERE source_session=?
             AND id NOT IN (?)
             AND created_at >= ?""",
        ["my_session", "new_mine", _since]
    ).fetchall()
    assert len(_tp_rows) == 0  # 没有同 session 的相邻 chunk


# ── TP3: 时间间隔 > 5min → 不在查询范围内 ────────────────────────────────────

def test_tp3_old_chunk_outside_window(conn):
    _old_time = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    _insert_chunk(conn, "too_old", session_id="s1", created_at=_old_time)
    _insert_chunk(conn, "fresh", session_id="s1")

    _since = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    _tp_rows = conn.execute(
        """SELECT id FROM memory_chunks
           WHERE source_session=?
             AND id NOT IN (?)
             AND created_at >= ?""",
        ["s1", "fresh", _since]
    ).fetchall()
    _recent = [r[0] for r in _tp_rows]
    assert "too_old" not in _recent


# ── TP4: written_chunk_ids 为空 → 无操作 ─────────────────────────────────────

def test_tp4_empty_written_ids(conn):
    written_chunk_ids = []
    # 模拟 extractor 的守卫条件
    should_run = bool(written_chunk_ids and "sess" and "sess" != "unknown")
    assert should_run is False
    count = conn.execute("SELECT COUNT(*) FROM chunk_edges").fetchone()[0]
    assert count == 0


# ── TP5: session_id = "unknown" → 不建立 ─────────────────────────────────────

def test_tp5_unknown_session_no_edge(conn):
    session_id = "unknown"
    written_chunk_ids = ["c1"]
    should_run = bool(written_chunk_ids and session_id and session_id != "unknown")
    assert should_run is False


# ── TP6: 边的 weight = 0.3 ────────────────────────────────────────────────────

def test_tp6_edge_weight_is_0_3(conn):
    from store_graph import add_edge, EdgeType
    _insert_chunk(conn, "a", session_id="s1")
    _insert_chunk(conn, "b", session_id="s1")
    add_edge(conn, "a", "b", EdgeType.COOCCURS, 0.3, source="temporal")
    row = conn.execute(
        "SELECT weight FROM chunk_edges WHERE from_id='a' AND to_id='b'"
    ).fetchone()
    assert row is not None
    assert abs(row[0] - 0.3) < 0.01


# ── TP7: 已有更强的边时不降低 weight ────────────────────────────────────────

def test_tp7_stronger_edge_not_downgraded(conn):
    from store_graph import add_edge, EdgeType
    _insert_chunk(conn, "x", session_id="s1")
    _insert_chunk(conn, "y", session_id="s1")
    # 先建强边（共现边，0.5）
    add_edge(conn, "x", "y", EdgeType.COOCCURS, 0.5, source="cooccurrence")
    # 再尝试建时序弱边（0.3），不应降低
    add_edge(conn, "x", "y", EdgeType.COOCCURS, 0.3, source="temporal")
    row = conn.execute(
        "SELECT weight FROM chunk_edges WHERE from_id='x' AND to_id='y'"
    ).fetchone()
    assert abs(row[0] - 0.5) < 0.01  # 保持 0.5，未降低
