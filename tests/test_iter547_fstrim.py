"""
iter547: fstrim — Auxiliary Table Dead Block TRIM
OS 类比：Linux fstrim / FITRIM ioctl (Lukas Czerner, 2010, kernel 2.6.37)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tmpfs  # noqa: E402, F401 — 测试隔离
import json
import sqlite3
import pytest
from datetime import datetime, timezone, timedelta
from store_vfs import open_db, ensure_schema
from store_mm import fstrim


PROJECT = "test:fstrim"


def _setup_db():
    """Create DB with schema and clean ALL test data for isolation."""
    conn = open_db()
    ensure_schema(conn)
    conn.execute("DELETE FROM memory_chunks WHERE project LIKE 'test:%'")
    conn.execute("DELETE FROM entity_edges WHERE project LIKE 'test:%' OR project IS NULL")
    conn.execute("DELETE FROM entity_map WHERE project LIKE 'test:%' OR project = ''")
    conn.execute("DELETE FROM chunk_coactivation WHERE project LIKE 'test:%'")
    conn.execute("DELETE FROM chunk_pins WHERE project LIKE 'test:%'")
    try:
        conn.execute("DELETE FROM shm_segments")
    except Exception:
        pass
    try:
        conn.execute("DELETE FROM trigger_conditions WHERE project LIKE 'test:%'")
    except Exception:
        pass
    try:
        conn.execute("DELETE FROM schema_anchors WHERE project LIKE 'test:%'")
    except Exception:
        pass
    try:
        conn.execute("DELETE FROM episodic_consolidations WHERE project LIKE 'test:%'")
    except Exception:
        pass
    conn.commit()
    return conn


def _insert_chunk(conn, chunk_id, project=PROJECT, chunk_type="decision",
                  importance=0.7, access_count=1, chunk_state="ACTIVE"):
    """Insert a test chunk."""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO memory_chunks "
        "(id, created_at, updated_at, project, chunk_type, summary, content, "
        "importance, access_count, oom_adj, chunk_state, lru_gen, last_accessed) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (chunk_id, now, now, project, chunk_type,
         f"test chunk {chunk_id}", f"content for {chunk_id}",
         importance, access_count, 0, chunk_state, 0, now)
    )
    conn.commit()


# ──────────────────────────────────────────────
# T1: entity_edges stale refs trimmed
# ──────────────────────────────────────────────
def test_entity_edges_stale_trimmed():
    conn = _setup_db()
    _insert_chunk(conn, "alive-1")
    # Insert edges: one alive, one stale
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO entity_edges (id, from_entity, relation, to_entity, source_chunk_id, project, created_at) "
        "VALUES (?,?,?,?,?,?,?)",
        ("edge-alive", "A", "relates", "B", "alive-1", PROJECT, now)
    )
    conn.execute(
        "INSERT INTO entity_edges (id, from_entity, relation, to_entity, source_chunk_id, project, created_at) "
        "VALUES (?,?,?,?,?,?,?)",
        ("edge-stale", "C", "relates", "D", "deleted-chunk-999", PROJECT, now)
    )
    conn.commit()
    result = fstrim(conn)
    assert result["trimmed"]["entity_edges"] == 1
    remaining = conn.execute("SELECT COUNT(*) FROM entity_edges WHERE id IN ('edge-alive','edge-stale')").fetchone()[0]
    assert remaining == 1  # only alive edge survives
    conn.close()


# ──────────────────────────────────────────────
# T2: entity_map stale refs trimmed
# ──────────────────────────────────────────────
def test_entity_map_stale_trimmed():
    conn = _setup_db()
    _insert_chunk(conn, "alive-2")
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO entity_map (entity_name, chunk_id, project, updated_at) "
        "VALUES (?,?,?,?)",
        ("entity-alive", "alive-2", PROJECT, now)
    )
    conn.execute(
        "INSERT OR REPLACE INTO entity_map (entity_name, chunk_id, project, updated_at) "
        "VALUES (?,?,?,?)",
        ("entity-stale", "deleted-chunk-888", PROJECT, now)
    )
    conn.commit()
    result = fstrim(conn)
    assert result["trimmed"]["entity_map"] >= 1
    alive = conn.execute("SELECT COUNT(*) FROM entity_map WHERE entity_name='entity-alive' AND project=?", (PROJECT,)).fetchone()[0]
    assert alive == 1
    stale = conn.execute("SELECT COUNT(*) FROM entity_map WHERE entity_name='entity-stale' AND project=?", (PROJECT,)).fetchone()[0]
    assert stale == 0
    conn.close()


# ──────────────────────────────────────────────
# T3: chunk_coactivation both-sides check
# ──────────────────────────────────────────────
def test_coactivation_stale_trimmed():
    conn = _setup_db()
    _insert_chunk(conn, "alive-3a")
    _insert_chunk(conn, "alive-3b")
    now = datetime.now(timezone.utc).isoformat()
    # Both alive — should survive
    conn.execute(
        "INSERT INTO chunk_coactivation (chunk_a, chunk_b, project, coact_count, last_coact) "
        "VALUES (?,?,?,?,?)",
        ("alive-3a", "alive-3b", PROJECT, 1, now)
    )
    # One side dead — should be trimmed
    conn.execute(
        "INSERT INTO chunk_coactivation (chunk_a, chunk_b, project, coact_count, last_coact) "
        "VALUES (?,?,?,?,?)",
        ("alive-3a", "deleted-x", PROJECT, 1, now)
    )
    # Both dead — should be trimmed
    conn.execute(
        "INSERT INTO chunk_coactivation (chunk_a, chunk_b, project, coact_count, last_coact) "
        "VALUES (?,?,?,?,?)",
        ("deleted-y", "deleted-z", PROJECT, 1, now)
    )
    conn.commit()
    result = fstrim(conn)
    assert result["trimmed"]["chunk_coactivation"] == 2
    remaining = conn.execute("SELECT COUNT(*) FROM chunk_coactivation WHERE project=?", (PROJECT,)).fetchone()[0]
    assert remaining == 1  # only both-alive survives
    conn.close()


# ──────────────────────────────────────────────
# T4: chunk_pins stale refs trimmed
# ──────────────────────────────────────────────
def test_chunk_pins_stale_trimmed():
    conn = _setup_db()
    _insert_chunk(conn, "alive-4")
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO chunk_pins (chunk_id, project, pin_type, pinned_at) "
        "VALUES (?,?,?,?)",
        ("alive-4", PROJECT, "soft", now)
    )
    conn.execute(
        "INSERT OR REPLACE INTO chunk_pins (chunk_id, project, pin_type, pinned_at) "
        "VALUES (?,?,?,?)",
        ("deleted-pin", PROJECT, "hard", now)
    )
    conn.commit()
    result = fstrim(conn)
    assert result["trimmed"]["chunk_pins"] >= 1
    alive = conn.execute("SELECT COUNT(*) FROM chunk_pins WHERE chunk_id='alive-4'").fetchone()[0]
    assert alive == 1
    stale = conn.execute("SELECT COUNT(*) FROM chunk_pins WHERE chunk_id='deleted-pin'").fetchone()[0]
    assert stale == 0
    conn.close()


# ──────────────────────────────────────────────
# T5: shm_segments stale refs trimmed
# ──────────────────────────────────────────────
def test_shm_segments_stale_trimmed():
    conn = _setup_db()
    _insert_chunk(conn, "alive-5")
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO shm_segments (chunk_id, owner_agent, state, created_at, updated_at) "
        "VALUES (?,?,?,?,?)",
        ("alive-5", "agent-1", "SHARED", now, now)
    )
    conn.execute(
        "INSERT OR REPLACE INTO shm_segments (chunk_id, owner_agent, state, created_at, updated_at) "
        "VALUES (?,?,?,?,?)",
        ("deleted-shm", "agent-2", "SHARED", now, now)
    )
    conn.commit()
    result = fstrim(conn)
    assert result["trimmed"]["shm_segments"] >= 1
    conn.close()


# ──────────────────────────────────────────────
# T6: trigger_conditions stale refs trimmed
# ──────────────────────────────────────────────
def test_trigger_conditions_stale_trimmed():
    conn = _setup_db()
    _insert_chunk(conn, "alive-6")
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO trigger_conditions (id, chunk_id, project, session_id, trigger_pattern, trigger_type, created_at) "
        "VALUES (?,?,?,?,?,?,?)",
        ("trig-alive", "alive-6", PROJECT, "s1", "pattern", "keyword", now)
    )
    conn.execute(
        "INSERT INTO trigger_conditions (id, chunk_id, project, session_id, trigger_pattern, trigger_type, created_at) "
        "VALUES (?,?,?,?,?,?,?)",
        ("trig-stale", "deleted-trig", PROJECT, "s1", "pattern", "keyword", now)
    )
    conn.commit()
    result = fstrim(conn)
    assert result["trimmed"]["trigger_conditions"] == 1
    remaining = conn.execute("SELECT COUNT(*) FROM trigger_conditions WHERE id IN ('trig-alive','trig-stale')").fetchone()[0]
    assert remaining == 1
    conn.close()


# ──────────────────────────────────────────────
# T7: schema_anchors stale refs trimmed
# ──────────────────────────────────────────────
def test_schema_anchors_stale_trimmed():
    conn = _setup_db()
    _insert_chunk(conn, "alive-7")
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO schema_anchors (chunk_id, schema_name, project, confidence, created_at) "
        "VALUES (?,?,?,?,?)",
        ("alive-7", "schema-a", PROJECT, 0.8, now)
    )
    conn.execute(
        "INSERT OR REPLACE INTO schema_anchors (chunk_id, schema_name, project, confidence, created_at) "
        "VALUES (?,?,?,?,?)",
        ("deleted-sa", "schema-b", PROJECT, 0.8, now)
    )
    conn.commit()
    result = fstrim(conn)
    assert result["trimmed"]["schema_anchors"] >= 1
    conn.close()


# ──────────────────────────────────────────────
# T8: episodic_consolidations fully-stale trimmed
# ──────────────────────────────────────────────
def test_episodic_consolidations_stale_trimmed():
    conn = _setup_db()
    _insert_chunk(conn, "alive-8")
    _insert_chunk(conn, "sem-8")
    now = datetime.now(timezone.utc).isoformat()
    # One with alive source — should survive
    conn.execute(
        "INSERT INTO episodic_consolidations (semantic_chunk_id, source_chunk_ids, project, trigger_count, created_at) "
        "VALUES (?,?,?,?,?)",
        ("sem-8", json.dumps(["alive-8"]), PROJECT, 1, now)
    )
    # One with all-dead sources — should be trimmed
    conn.execute(
        "INSERT INTO episodic_consolidations (semantic_chunk_id, source_chunk_ids, project, trigger_count, created_at) "
        "VALUES (?,?,?,?,?)",
        ("sem-dead", json.dumps(["deleted-ep1", "deleted-ep2"]), PROJECT, 1, now)
    )
    conn.commit()
    result = fstrim(conn)
    assert result["trimmed"]["episodic_consolidations"] == 1
    remaining = conn.execute("SELECT COUNT(*) FROM episodic_consolidations WHERE project=?", (PROJECT,)).fetchone()[0]
    assert remaining == 1
    conn.close()


# ──────────────────────────────────────────────
# T9: empty DB — no errors, all zeros
# ──────────────────────────────────────────────
def test_empty_db_no_errors():
    conn = _setup_db()
    result = fstrim(conn)
    assert result["total_trimmed"] == 0
    assert result["duration_ms"] >= 0
    for v in result["trimmed"].values():
        assert v == 0
    conn.close()


# ──────────────────────────────────────────────
# T10: alive chunks are never trimmed
# ──────────────────────────────────────────────
def test_alive_refs_preserved():
    conn = _setup_db()
    _insert_chunk(conn, "alive-10a")
    _insert_chunk(conn, "alive-10b")
    now = datetime.now(timezone.utc).isoformat()
    # All refs point to alive chunks
    conn.execute(
        "INSERT INTO entity_edges (id, from_entity, relation, to_entity, source_chunk_id, project, created_at) "
        "VALUES (?,?,?,?,?,?,?)",
        ("edge-ok", "X", "r", "Y", "alive-10a", PROJECT, now)
    )
    conn.execute(
        "INSERT OR REPLACE INTO entity_map (entity_name, chunk_id, project, updated_at) "
        "VALUES (?,?,?,?)",
        ("ent-ok", "alive-10a", PROJECT, now)
    )
    conn.execute(
        "INSERT INTO chunk_coactivation (chunk_a, chunk_b, project, coact_count, last_coact) "
        "VALUES (?,?,?,?,?)",
        ("alive-10a", "alive-10b", PROJECT, 1, now)
    )
    conn.commit()
    result = fstrim(conn)
    # Nothing should be trimmed
    assert result["trimmed"]["entity_edges"] == 0
    assert result["trimmed"]["entity_map"] == 0
    assert result["trimmed"]["chunk_coactivation"] == 0
    conn.close()


# ──────────────────────────────────────────────
# T11: total_trimmed is sum of all tables
# ──────────────────────────────────────────────
def test_total_trimmed_is_sum():
    conn = _setup_db()
    _insert_chunk(conn, "alive-11")
    now = datetime.now(timezone.utc).isoformat()
    # Insert stale refs in multiple tables
    conn.execute(
        "INSERT INTO entity_edges (id, from_entity, relation, to_entity, source_chunk_id, project, created_at) "
        "VALUES (?,?,?,?,?,?,?)",
        ("edge-s1", "A", "r", "B", "dead-11a", PROJECT, now)
    )
    conn.execute(
        "INSERT OR REPLACE INTO entity_map (entity_name, chunk_id, project, updated_at) "
        "VALUES (?,?,?,?)",
        ("ent-s1", "dead-11b", PROJECT, now)
    )
    conn.execute(
        "INSERT OR REPLACE INTO chunk_pins (chunk_id, project, pin_type, pinned_at) "
        "VALUES (?,?,?,?)",
        ("dead-11c", PROJECT, "soft", now)
    )
    conn.commit()
    result = fstrim(conn)
    expected_total = sum(result["trimmed"].values())
    assert result["total_trimmed"] == expected_total
    assert result["total_trimmed"] >= 3
    conn.close()


# ──────────────────────────────────────────────
# T12: idempotent — second run returns 0
# ──────────────────────────────────────────────
def test_idempotent():
    conn = _setup_db()
    _insert_chunk(conn, "alive-12")
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO entity_edges (id, from_entity, relation, to_entity, source_chunk_id, project, created_at) "
        "VALUES (?,?,?,?,?,?,?)",
        ("edge-idem", "A", "r", "B", "dead-idem", PROJECT, now)
    )
    conn.commit()
    r1 = fstrim(conn)
    assert r1["total_trimmed"] >= 1
    r2 = fstrim(conn)
    assert r2["total_trimmed"] == 0
    conn.close()


# ──────────────────────────────────────────────
# T13: performance — 1000 stale records < 500ms
# ──────────────────────────────────────────────
def test_performance():
    conn = _setup_db()
    _insert_chunk(conn, "alive-perf")
    now = datetime.now(timezone.utc).isoformat()
    # Insert 1000 stale entity_map entries
    for i in range(1000):
        conn.execute(
            "INSERT OR REPLACE INTO entity_map (entity_name, chunk_id, project, updated_at) "
            "VALUES (?,?,?,?)",
            (f"perf-ent-{i}", f"dead-perf-{i}", PROJECT, now)
        )
    conn.commit()
    result = fstrim(conn)
    assert result["trimmed"]["entity_map"] == 1000
    assert result["duration_ms"] < 500
    conn.close()


# ──────────────────────────────────────────────
# T14: config disabled — no trimming
# ──────────────────────────────────────────────
def test_config_fstrim_enabled():
    """fstrim function itself doesn't check config — that's loader's job.
    Verify function always works when called directly."""
    conn = _setup_db()
    _insert_chunk(conn, "alive-14")
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO entity_edges (id, from_entity, relation, to_entity, source_chunk_id, project, created_at) "
        "VALUES (?,?,?,?,?,?,?)",
        ("edge-cfg", "A", "r", "B", "dead-cfg", PROJECT, now)
    )
    conn.commit()
    result = fstrim(conn)
    assert result["total_trimmed"] >= 1
    conn.close()
