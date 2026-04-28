"""
test_schema_anchoring.py — iter380 Schema Anchoring 测试

覆盖：
  SA1: 端口/服务配置 summary → 写入 web_service_config schema 绑定
  SA2: 错误/异常 summary → 写入 error_pattern schema 绑定
  SA3: 无匹配模式的通用 summary → 不写入任何 schema 绑定
  SA4: schema_spread_activate() → FTS5 命中 chunk 后激活同 schema 其他 chunk
  SA5: existing_ids 排除 — 已命中的 chunk 不被重复加入激活结果
  SA6: anchor_chunk_schema() 幂等 — 多次写入同 chunk 不重复绑定 (INSERT OR IGNORE)
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
    if orig:
        os.environ["MEMORY_OS_DIR"] = orig
    return conn


def _insert_chunk(conn: sqlite3.Connection, chunk_id: str, project: str,
                  summary: str, chunk_type: str = "decision",
                  importance: float = 0.75) -> None:
    """Helper: insert a minimal memory chunk."""
    now_iso = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT INTO memory_chunks
           (id, created_at, updated_at, project, source_session,
            chunk_type, info_class, content, summary, tags,
            importance, retrievability, stability, last_accessed,
            access_count, oom_adj, lru_gen, raw_snippet, encoding_context)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (chunk_id, now_iso, now_iso, project, "test-session",
         chunk_type, "world", f"content for {chunk_id}", summary,
         "[]", importance, 0.5, 1.0, now_iso,
         0, 0, 0, "", "{}")
    )
    conn.commit()


# ── SA1: port/service summary → web_service_config schema ────────────────────

def test_sa1_port_summary_binds_web_service_config():
    """端口配置 summary 自动绑定到 web_service_config schema。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        conn = _make_conn(tmpdir)
        project = "test:sa1"

        from store_vfs import anchor_chunk_schema
        # Port configuration pattern
        n = anchor_chunk_schema(conn, "c_sa1_1", "后端服务端口 port=8080 启动", project)
        assert n > 0, f"Expected at least 1 schema binding, got {n}"

        # Check schema_anchors table
        rows = conn.execute(
            "SELECT schema_name FROM schema_anchors WHERE chunk_id='c_sa1_1' AND project=?",
            (project,)
        ).fetchall()
        schema_names = {r[0] for r in rows}
        assert "web_service_config" in schema_names, \
            f"Expected 'web_service_config' in schemas, got {schema_names}"
        conn.close()


def test_sa1b_localhost_port_pattern():
    """localhost:port 格式触发 web_service_config。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        conn = _make_conn(tmpdir)
        project = "test:sa1b"

        from store_vfs import anchor_chunk_schema
        n = anchor_chunk_schema(conn, "c_sa1b", "前端服务运行在 localhost:3000", project)
        assert n > 0

        rows = conn.execute(
            "SELECT schema_name FROM schema_anchors WHERE chunk_id='c_sa1b' AND project=?",
            (project,)
        ).fetchall()
        assert any(r[0] == "web_service_config" for r in rows), \
            f"Expected web_service_config, got {[r[0] for r in rows]}"
        conn.close()


# ── SA2: error/exception summary → error_pattern schema ──────────────────────

def test_sa2_error_summary_binds_error_pattern():
    """错误/异常 summary 绑定到 error_pattern schema。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        conn = _make_conn(tmpdir)
        project = "test:sa2"

        from store_vfs import anchor_chunk_schema
        n = anchor_chunk_schema(conn, "c_sa2_1", "AttributeError: NoneType has no attribute 'id'", project)
        assert n > 0

        rows = conn.execute(
            "SELECT schema_name FROM schema_anchors WHERE chunk_id='c_sa2_1' AND project=?",
            (project,)
        ).fetchall()
        schema_names = {r[0] for r in rows}
        assert "error_pattern" in schema_names, \
            f"Expected 'error_pattern', got {schema_names}"
        conn.close()


# ── SA3: generic summary → no schema binding ─────────────────────────────────

def test_sa3_generic_summary_no_binding():
    """普通 summary 不触发任何 schema 绑定（无匹配模式）。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        conn = _make_conn(tmpdir)
        project = "test:sa3"

        from store_vfs import anchor_chunk_schema
        n = anchor_chunk_schema(conn, "c_sa3_1",
                                 "用户询问了关于数据处理流程的问题", project)
        # Generic summary without any schema keywords
        assert n == 0, f"Expected 0 schema bindings, got {n}"

        rows = conn.execute(
            "SELECT COUNT(*) FROM schema_anchors WHERE chunk_id='c_sa3_1' AND project=?",
            (project,)
        ).fetchone()
        assert rows[0] == 0, f"Expected no rows, got {rows[0]}"
        conn.close()


# ── SA4: schema_spread_activate — activates same-schema chunks ───────────────

def test_sa4_schema_spread_activates_related():
    """命中 chunk → schema_spread_activate 激活同 schema 的其他 chunk。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        conn = _make_conn(tmpdir)
        project = "test:sa4"

        from store_vfs import anchor_chunk_schema, schema_spread_activate

        # Insert 3 chunks in web_service_config schema
        _insert_chunk(conn, "c_hit", project,
                      "后端 API 服务端口 port=8080", importance=0.8)
        _insert_chunk(conn, "c_related1", project,
                      "前端服务监听 localhost:3000", importance=0.75)
        _insert_chunk(conn, "c_related2", project,
                      "数据库连接 postgres://localhost:5432/mydb", importance=0.7)
        _insert_chunk(conn, "c_unrelated", project,
                      "用户讨论了代码重构计划", importance=0.6)

        # Bind chunks to schemas
        anchor_chunk_schema(conn, "c_hit", "后端 API 服务端口 port=8080", project)
        anchor_chunk_schema(conn, "c_related1", "前端服务监听 localhost:3000", project)
        anchor_chunk_schema(conn, "c_unrelated", "用户讨论了代码重构计划", project)
        conn.commit()

        # FTS5 "hit" → activate related chunks in same schema
        result = schema_spread_activate(conn, ["c_hit"], project=project)

        # c_related1 should be activated (same web_service_config schema)
        assert "c_related1" in result, \
            f"Expected c_related1 in activation result, got {list(result.keys())}"
        # c_hit itself should NOT appear (already in hit_chunk_ids)
        assert "c_hit" not in result, "c_hit should be excluded (already hit)"
        # c_unrelated should NOT appear (different schema)
        assert "c_unrelated" not in result, \
            f"c_unrelated should not be activated (no shared schema)"
        conn.close()


# ── SA5: existing_ids exclusion ──────────────────────────────────────────────

def test_sa5_existing_ids_excluded():
    """existing_ids 中的 chunk 不出现在激活结果中。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        conn = _make_conn(tmpdir)
        project = "test:sa5"

        from store_vfs import anchor_chunk_schema, schema_spread_activate

        _insert_chunk(conn, "c_a", project, "后端端口 port=8080", importance=0.8)
        _insert_chunk(conn, "c_b", project, "前端服务 localhost:3000", importance=0.75)
        _insert_chunk(conn, "c_c", project, "API 端口 0.0.0.0:9090", importance=0.7)

        anchor_chunk_schema(conn, "c_a", "后端端口 port=8080", project)
        anchor_chunk_schema(conn, "c_b", "前端服务 localhost:3000", project)
        anchor_chunk_schema(conn, "c_c", "API 端口 0.0.0.0:9090", project)
        conn.commit()

        # c_b is already in existing_ids
        result = schema_spread_activate(
            conn, ["c_a"], project=project,
            existing_ids={"c_b"}
        )

        # c_b should be excluded
        assert "c_b" not in result, \
            f"c_b should be excluded via existing_ids, but got {list(result.keys())}"
        conn.close()


# ── SA6: idempotency — no duplicate schema_anchors rows ──────────────────────

def test_sa6_anchor_chunk_schema_idempotent():
    """多次调用 anchor_chunk_schema 不产生重复绑定行 (INSERT OR IGNORE)。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        conn = _make_conn(tmpdir)
        project = "test:sa6"

        from store_vfs import anchor_chunk_schema

        summary = "后端服务端口 port=8080"
        # Call 3 times
        anchor_chunk_schema(conn, "c_sa6_1", summary, project)
        anchor_chunk_schema(conn, "c_sa6_1", summary, project)
        anchor_chunk_schema(conn, "c_sa6_1", summary, project)
        conn.commit()

        rows = conn.execute(
            "SELECT schema_name FROM schema_anchors WHERE chunk_id='c_sa6_1' AND project=?",
            (project,)
        ).fetchall()
        schema_names = [r[0] for r in rows]
        # No duplicates
        assert len(schema_names) == len(set(schema_names)), \
            f"Duplicate schema_anchors rows: {schema_names}"
        conn.close()


# ── SA7: design_decision schema ───────────────────────────────────────────────

def test_sa7_design_decision_schema():
    """'选择X因为Y' pattern → design_decision schema。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        conn = _make_conn(tmpdir)
        project = "test:sa7"

        from store_vfs import anchor_chunk_schema
        n = anchor_chunk_schema(conn, "c_sa7_1",
                                 "选择 PostgreSQL 因为需要事务支持", project)
        assert n > 0

        rows = conn.execute(
            "SELECT schema_name FROM schema_anchors WHERE chunk_id='c_sa7_1' AND project=?",
            (project,)
        ).fetchall()
        schema_names = {r[0] for r in rows}
        assert "design_decision" in schema_names, \
            f"Expected 'design_decision', got {schema_names}"
        conn.close()
