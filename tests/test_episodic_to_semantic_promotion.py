"""
test_episodic_to_semantic_promotion.py — iter379 Episodic-to-Semantic Promotion 测试

覆盖：
  ES1: access_count < 5 → chunk 保持 episodic，不升级
  ES2: access_count >= 5, 可巩固类型 → 原地升级为 semantic，stability × 1.5，oom_adj -= 50
  ES3: 不可巩固类型（entity_stub）即使 access_count >= 5 也不升级
  ES4: 升级后 info_class='semantic'，不重复升级
  ES5: sleep_consolidate() 返回 episodic_inplace_promoted 计数
"""
import sys
import sqlite3
import tempfile
from pathlib import Path
from datetime import datetime, timezone

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


def _make_conn(tmpdir: Path) -> sqlite3.Connection:
    """Helper: create a temp store.db and ensure schema."""
    from store import open_db, ensure_schema
    db_path = tmpdir / "store.db"
    import os
    orig = os.environ.get("MEMORY_OS_DIR", "")
    os.environ["MEMORY_OS_DIR"] = str(tmpdir)
    conn = sqlite3.connect(str(db_path))
    ensure_schema(conn)
    # Restore
    if orig:
        os.environ["MEMORY_OS_DIR"] = orig
    return conn


def _insert_chunk(conn: sqlite3.Connection, chunk_id: str, project: str,
                  chunk_type: str, info_class: str, access_count: int,
                  stability: float = 1.0, oom_adj: int = 0) -> None:
    """Helper: insert a minimal memory chunk directly."""
    now_iso = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT INTO memory_chunks
           (id, created_at, updated_at, project, source_session,
            chunk_type, info_class, content, summary, tags,
            importance, retrievability, stability, last_accessed,
            access_count, oom_adj, lru_gen, raw_snippet, encoding_context)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (chunk_id, now_iso, now_iso, project, "test-session",
         chunk_type, info_class, f"[{chunk_type}] test content", f"test summary {chunk_id}",
         "[]", 0.75, 0.5, stability, now_iso,
         access_count, oom_adj, 0, "", "{}")
    )
    conn.commit()


# ── ES1: access_count < threshold → stays episodic ───────────────────────────

def test_es1_below_threshold_stays_episodic():
    """access_count < 5 → inplace_promoted=0（不触发原地升级路径）。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        conn = _make_conn(tmpdir)
        project = "test:es1"

        # access_count=4 — below hard threshold=5, so inplace path skips it
        _insert_chunk(conn, "c_es1_1", project, "reasoning_chain", "episodic",
                      access_count=4, stability=2.0, oom_adj=0)

        from store_vfs import episodic_decay_scan
        # Pass semantic_hard_threshold=5 explicitly; semantic_threshold=100 to disable merge path
        result = episodic_decay_scan(conn, project, semantic_threshold=100,
                                     semantic_hard_threshold=5)

        assert result["inplace_promoted"] == 0, \
            f"Expected no inplace promotion, got {result['inplace_promoted']}"
        conn.close()


# ── ES2: access_count >= 5 → in-place upgrade to semantic ────────────────────

def test_es2_above_threshold_upgrades_inplace():
    """access_count >= 5, reasoning_chain → 原地升级为 semantic，stability × 1.5。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        conn = _make_conn(tmpdir)
        project = "test:es2"
        init_stability = 2.0
        init_oom = 100

        _insert_chunk(conn, "c_es2_1", project, "reasoning_chain", "episodic",
                      access_count=5, stability=init_stability, oom_adj=init_oom)

        from store_vfs import episodic_decay_scan
        result = episodic_decay_scan(conn, project, semantic_hard_threshold=5)

        assert result["inplace_promoted"] == 1, \
            f"Expected 1 inplace promotion, got {result['inplace_promoted']}"

        row = conn.execute(
            "SELECT info_class, stability, oom_adj FROM memory_chunks WHERE id='c_es2_1'"
        ).fetchone()
        assert row[0] == "semantic", f"Expected 'semantic', got {row[0]}"
        # stability × 1.5
        assert abs(row[1] - init_stability * 1.5) < 0.01, \
            f"Expected stability={init_stability * 1.5}, got {row[1]}"
        # oom_adj -= 50
        assert row[2] == init_oom - 50, f"Expected oom_adj={init_oom - 50}, got {row[2]}"
        conn.close()


def test_es2_multiple_consolidatable_types():
    """多种可巩固类型 (conversation_summary, causal_chain) 都能升级。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        conn = _make_conn(tmpdir)
        project = "test:es2b"

        for i, ctype in enumerate(["conversation_summary", "causal_chain", "decision"]):
            _insert_chunk(conn, f"c_es2b_{i}", project, ctype, "episodic",
                          access_count=6, stability=1.0)

        from store_vfs import episodic_decay_scan
        result = episodic_decay_scan(conn, project, semantic_hard_threshold=5)

        assert result["inplace_promoted"] == 3, \
            f"Expected 3 inplace promotions, got {result['inplace_promoted']}"

        rows = conn.execute(
            "SELECT info_class FROM memory_chunks WHERE project=? AND id LIKE 'c_es2b_%'",
            (project,)
        ).fetchall()
        for row in rows:
            assert row[0] == "semantic", f"Expected 'semantic', got {row[0]}"
        conn.close()


# ── ES3: non-consolidatable type → not upgraded ───────────────────────────────

def test_es3_non_consolidatable_type_not_upgraded():
    """entity_stub 类型即使 access_count >= 5 也不升级（inplace_promoted=0）。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        conn = _make_conn(tmpdir)
        project = "test:es3"

        _insert_chunk(conn, "c_es3_1", project, "entity_stub", "episodic",
                      access_count=10)

        from store_vfs import episodic_decay_scan
        # semantic_threshold=100 disables merge path; only test A0 (inplace) behavior
        result = episodic_decay_scan(conn, project, semantic_threshold=100,
                                     semantic_hard_threshold=5)

        assert result["inplace_promoted"] == 0, \
            f"Expected no inplace promotion for entity_stub, got {result['inplace_promoted']}"
        conn.close()


# ── ES4: already semantic → not double-promoted ───────────────────────────────

def test_es4_already_semantic_not_touched():
    """已经是 semantic 的 chunk 不重复升级。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        conn = _make_conn(tmpdir)
        project = "test:es4"

        _insert_chunk(conn, "c_es4_1", project, "reasoning_chain", "semantic",
                      access_count=10, stability=5.0, oom_adj=-100)

        from store_vfs import episodic_decay_scan
        result = episodic_decay_scan(conn, project, semantic_hard_threshold=5)

        assert result["inplace_promoted"] == 0, \
            f"Expected 0 (already semantic), got {result['inplace_promoted']}"

        row = conn.execute(
            "SELECT info_class, stability, oom_adj FROM memory_chunks WHERE id='c_es4_1'"
        ).fetchone()
        # Unchanged
        assert row[0] == "semantic"
        assert abs(row[1] - 5.0) < 0.01
        assert row[2] == -100
        conn.close()


# ── ES5: sleep_consolidate returns inplace_promoted count ─────────────────────

def test_es5_sleep_consolidate_returns_inplace_count():
    """sleep_consolidate() 返回 episodic_inplace_promoted 字段。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        conn = _make_conn(tmpdir)
        project = "test:es5"

        # Insert 2 eligible chunks
        _insert_chunk(conn, "c_es5_1", project, "reasoning_chain", "episodic",
                      access_count=7, stability=1.5)
        _insert_chunk(conn, "c_es5_2", project, "conversation_summary", "episodic",
                      access_count=5, stability=2.0)
        # Insert 1 below threshold
        _insert_chunk(conn, "c_es5_3", project, "decision", "episodic",
                      access_count=3)

        from store_vfs import sleep_consolidate
        result = sleep_consolidate(conn, project=project, session_id="test-es5")

        assert "episodic_inplace_promoted" in result, \
            f"sleep_consolidate() should return 'episodic_inplace_promoted'. Got: {list(result.keys())}"
        assert result["episodic_inplace_promoted"] == 2, \
            f"Expected 2 inplace promoted, got {result['episodic_inplace_promoted']}"
        conn.close()


# ── ES6: stability cap at 200 ─────────────────────────────────────────────────

def test_es6_stability_capped_at_200():
    """stability × 1.5 上限为 200。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        conn = _make_conn(tmpdir)
        project = "test:es6"

        _insert_chunk(conn, "c_es6_1", project, "decision", "episodic",
                      access_count=8, stability=150.0)  # 150 × 1.5 = 225 > 200

        from store_vfs import episodic_decay_scan
        result = episodic_decay_scan(conn, project, semantic_hard_threshold=5)

        assert result["inplace_promoted"] == 1
        row = conn.execute(
            "SELECT stability FROM memory_chunks WHERE id='c_es6_1'"
        ).fetchone()
        assert row[0] <= 200.0, f"stability should be capped at 200, got {row[0]}"
        conn.close()
