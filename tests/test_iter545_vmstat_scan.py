"""
iter545: vmstat_scan — Scan Efficiency Accounting & Dark Page Demotion
OS 类比：Linux /proc/vmstat pgscan_kswapd/pgsteal_kswapd (Mel Gorman, 2004)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tmpfs  # noqa: E402, F401 — 测试隔离
import json
import sqlite3
import pytest
from datetime import datetime, timezone, timedelta
from store_vfs import open_db, ensure_schema, insert_chunk
from store_mm import vmstat_scan
from store_core import bump_chunk_version


PROJECT = "test:vmstat"


def _setup_db():
    """Create DB with schema and clean ALL test data for isolation."""
    conn = open_db()
    ensure_schema(conn)
    # Clean up all test-related data for full isolation
    conn.execute("DELETE FROM memory_chunks WHERE project LIKE 'test:%' OR project LIKE 'proj%' OR project LIKE 'other:%'")
    conn.execute("DELETE FROM recall_traces WHERE project LIKE 'test:%' OR project LIKE 'proj%' OR project LIKE 'other:%'")
    conn.commit()
    return conn


def _insert_chunk(conn, chunk_id, project=PROJECT, chunk_type="decision",
                  importance=0.7, access_count=0, oom_adj=0):
    """Insert a test chunk."""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO memory_chunks "
        "(id, created_at, updated_at, project, chunk_type, summary, content, "
        "importance, access_count, oom_adj, chunk_state, lru_gen) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (chunk_id, now, now, project, chunk_type,
         f"test chunk {chunk_id}", f"content for {chunk_id}",
         importance, access_count, oom_adj, "ACTIVE", 0)
    )
    conn.commit()


def _insert_trace(conn, project=PROJECT, candidates=10, top_k_items=None):
    """Insert a recall_trace record."""
    import uuid
    now = datetime.now(timezone.utc).isoformat()
    top_k_json = json.dumps(top_k_items or [])
    conn.execute(
        "INSERT INTO recall_traces "
        "(id, timestamp, session_id, project, prompt_hash, candidates_count, top_k_json, injected) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (str(uuid.uuid4()), now, "test-session", project,
         str(uuid.uuid4())[:8], candidates, top_k_json, 1 if top_k_items else 0)
    )
    conn.commit()


# ── Phase 1: Scan/Steal Accounting Tests ──

class TestScanAccounting:
    def test_empty_traces(self):
        """No traces → zero counters."""
        conn = _setup_db()
        result = vmstat_scan(conn, PROJECT)
        assert result["pgscan"] == 0
        assert result["pgsteal"] == 0
        assert result["scan_efficiency"] == 0.0
        conn.close()

    def test_basic_scan_steal(self):
        """Candidates=10, 2 items in top_k → efficiency=0.2."""
        conn = _setup_db()
        _insert_trace(conn, candidates=10, top_k_items=[
            {"id": "chunk-a", "score": 0.8, "chunk_type": "decision"},
            {"id": "chunk-b", "score": 0.6, "chunk_type": "causal_chain"},
        ])
        result = vmstat_scan(conn, PROJECT)
        assert result["pgscan"] == 10
        assert result["pgsteal"] == 2
        assert result["scan_efficiency"] == 0.2
        conn.close()

    def test_multiple_traces_accumulate(self):
        """Multiple traces accumulate pgscan and pgsteal."""
        conn = _setup_db()
        _insert_trace(conn, candidates=10, top_k_items=[
            {"id": "chunk-a", "score": 0.8, "chunk_type": "decision"},
        ])
        _insert_trace(conn, candidates=14, top_k_items=[
            {"id": "chunk-b", "score": 0.7, "chunk_type": "decision"},
            {"id": "chunk-c", "score": 0.5, "chunk_type": "causal_chain"},
        ])
        _insert_trace(conn, candidates=8, top_k_items=[])  # no injection
        result = vmstat_scan(conn, PROJECT)
        assert result["pgscan"] == 32  # 10+14+8
        assert result["pgsteal"] == 3  # 1+2+0
        assert 0.09 < result["scan_efficiency"] < 0.10  # 3/32 ≈ 0.094
        conn.close()

    def test_project_isolation(self):
        """Only counts traces for the given project."""
        conn = _setup_db()
        _insert_trace(conn, project=PROJECT, candidates=10, top_k_items=[
            {"id": "chunk-x", "score": 0.9, "chunk_type": "decision"},
        ])
        _insert_trace(conn, project="other:project", candidates=20, top_k_items=[
            {"id": "chunk-y", "score": 0.8, "chunk_type": "decision"},
        ])
        result = vmstat_scan(conn, PROJECT)
        assert result["pgscan"] == 10
        assert result["pgsteal"] == 1
        conn.close()


# ── Phase 2: Dark Page Detection Tests ──

class TestDarkPageDetection:
    def test_insufficient_traces_skips_detection(self):
        """Fewer than min_traces_dark → no dark page detection."""
        conn = _setup_db()
        _insert_chunk(conn, "chunk-1")
        # Only 2 traces (default min_traces_dark=5)
        _insert_trace(conn, candidates=5, top_k_items=[])
        _insert_trace(conn, candidates=5, top_k_items=[])
        result = vmstat_scan(conn, PROJECT)
        assert result["dark_pages_total"] == 0
        assert result["dark_pages_demoted"] == 0
        conn.close()

    def test_dark_page_identified(self):
        """Chunk never in any top_k after min_traces → dark page."""
        conn = _setup_db()
        _insert_chunk(conn, "dark-1", importance=0.5, access_count=0)
        _insert_chunk(conn, "surfaced-1", importance=0.7, access_count=2)
        # Insert enough traces (>=5) with surfaced-1 appearing
        for _ in range(6):
            _insert_trace(conn, candidates=10, top_k_items=[
                {"id": "surfaced-1", "score": 0.8, "chunk_type": "decision"},
            ])
        result = vmstat_scan(conn, PROJECT)
        assert result["dark_pages_total"] >= 1  # dark-1 is dark
        conn.close()

    def test_surfaced_chunk_not_dark(self):
        """Chunk that appeared in at least one top_k is NOT dark."""
        conn = _setup_db()
        _insert_chunk(conn, "surfaced-1", importance=0.7)
        for i in range(6):
            items = [{"id": "surfaced-1", "score": 0.8, "chunk_type": "decision"}] if i == 3 else []
            _insert_trace(conn, candidates=10, top_k_items=items)
        result = vmstat_scan(conn, PROJECT)
        # surfaced-1 appeared once → not dark
        assert result["dark_pages_total"] == 0
        conn.close()


# ── Phase 3: Demotion Tests ──

class TestDemotion:
    def _setup_with_dark_pages(self):
        """Create DB with dark pages and enough traces."""
        conn = _setup_db()
        # Dark pages (never in top_k, access=0, importance<0.9, oom_adj=0)
        _insert_chunk(conn, "dark-1", importance=0.5, access_count=0, oom_adj=0)
        _insert_chunk(conn, "dark-2", importance=0.6, access_count=0, oom_adj=0)
        _insert_chunk(conn, "dark-3", importance=0.4, access_count=0, oom_adj=0)
        # Surfaced chunk
        _insert_chunk(conn, "active-1", importance=0.8, access_count=5)
        # Insert 6 traces (>= min_traces_dark=5) with only active-1
        for _ in range(6):
            _insert_trace(conn, candidates=10, top_k_items=[
                {"id": "active-1", "score": 0.9, "chunk_type": "decision"},
            ])
        return conn

    def test_dark_pages_demoted(self):
        """Dark pages get oom_adj set to demote_adj."""
        conn = self._setup_with_dark_pages()
        result = vmstat_scan(conn, PROJECT)
        assert result["dark_pages_demoted"] >= 1
        # Verify oom_adj changed
        row = conn.execute(
            "SELECT oom_adj FROM memory_chunks WHERE id='dark-1'"
        ).fetchone()
        assert row[0] == 400  # default demote_adj
        conn.close()

    def test_max_demote_per_scan_respected(self):
        """Cannot demote more than max_demote_per_scan per call."""
        conn = _setup_db()
        # Create 10 dark pages
        for i in range(10):
            _insert_chunk(conn, f"dark-{i}", importance=0.4, access_count=0, oom_adj=0)
        for _ in range(6):
            _insert_trace(conn, candidates=10, top_k_items=[])
        result = vmstat_scan(conn, PROJECT)
        # Default max_demote_per_scan=5
        assert result["dark_pages_demoted"] <= 5
        conn.close()

    def test_protected_chunks_skipped(self):
        """Chunks with oom_adj < 0 are not demoted."""
        conn = _setup_db()
        _insert_chunk(conn, "protected-1", importance=0.5, access_count=0, oom_adj=-500)
        for _ in range(6):
            _insert_trace(conn, candidates=10, top_k_items=[])
        result = vmstat_scan(conn, PROJECT)
        assert result["dark_pages_skipped"] >= 1
        # Verify oom_adj unchanged
        row = conn.execute(
            "SELECT oom_adj FROM memory_chunks WHERE id='protected-1'"
        ).fetchone()
        assert row[0] == -500
        conn.close()

    def test_accessed_chunks_skipped(self):
        """Chunks with access_count > 0 are not demoted (may just need longer window)."""
        conn = _setup_db()
        _insert_chunk(conn, "accessed-1", importance=0.5, access_count=3, oom_adj=0)
        for _ in range(6):
            _insert_trace(conn, candidates=10, top_k_items=[])
        result = vmstat_scan(conn, PROJECT)
        row = conn.execute(
            "SELECT oom_adj FROM memory_chunks WHERE id='accessed-1'"
        ).fetchone()
        assert row[0] == 0  # unchanged
        conn.close()

    def test_high_importance_skipped(self):
        """Chunks with importance >= 0.9 are not demoted."""
        conn = _setup_db()
        _insert_chunk(conn, "important-1", importance=0.95, access_count=0, oom_adj=0)
        for _ in range(6):
            _insert_trace(conn, candidates=10, top_k_items=[])
        result = vmstat_scan(conn, PROJECT)
        row = conn.execute(
            "SELECT oom_adj FROM memory_chunks WHERE id='important-1'"
        ).fetchone()
        assert row[0] == 0  # unchanged
        conn.close()

    def test_already_demoted_skipped(self):
        """Chunks already at or above demote_adj are not re-demoted."""
        conn = _setup_db()
        _insert_chunk(conn, "already-demoted", importance=0.5, access_count=0, oom_adj=400)
        for _ in range(6):
            _insert_trace(conn, candidates=10, top_k_items=[])
        result = vmstat_scan(conn, PROJECT)
        # Should be skipped, not double-penalized
        row = conn.execute(
            "SELECT oom_adj FROM memory_chunks WHERE id='already-demoted'"
        ).fetchone()
        assert row[0] == 400  # unchanged
        conn.close()

    def test_chunk_version_bumped_on_demotion(self):
        """chunk_version is bumped when demotions occur (TLB invalidation)."""
        from store_vfs import CHUNK_VERSION_FILE
        conn = _setup_db()
        _insert_chunk(conn, "dark-bump", importance=0.4, access_count=0, oom_adj=0)
        for _ in range(6):
            _insert_trace(conn, candidates=10, top_k_items=[])
        # Get version before
        try:
            v_before = int(CHUNK_VERSION_FILE.read_text().strip()) if CHUNK_VERSION_FILE.exists() else 0
        except (ValueError, OSError):
            v_before = 0
        vmstat_scan(conn, PROJECT)
        try:
            v_after = int(CHUNK_VERSION_FILE.read_text().strip()) if CHUNK_VERSION_FILE.exists() else 0
        except (ValueError, OSError):
            v_after = 0
        assert v_after > v_before
        conn.close()


# ── Integration / Edge Cases ──

class TestEdgeCases:
    def test_null_project_scans_all(self):
        """project=None scans all projects."""
        conn = _setup_db()
        # Clean ALL traces for this test (project=None scans globally)
        conn.execute("DELETE FROM recall_traces")
        conn.execute("DELETE FROM memory_chunks WHERE project IN ('proj1','proj2')")
        conn.commit()
        _insert_chunk(conn, "p1-chunk", project="proj1", importance=0.5)
        _insert_chunk(conn, "p2-chunk", project="proj2", importance=0.5)
        _insert_trace(conn, project="proj1", candidates=5, top_k_items=[
            {"id": "p1-chunk", "score": 0.8, "chunk_type": "decision"},
        ])
        _insert_trace(conn, project="proj2", candidates=5, top_k_items=[])
        # 4 more traces to reach 6 total (>= min_traces_dark=5)
        for _ in range(4):
            _insert_trace(conn, project="proj1", candidates=5, top_k_items=[])
        result = vmstat_scan(conn, project=None)
        assert result["pgscan"] == 30  # 6 traces × 5 candidates
        conn.close()

    def test_duration_ms_recorded(self):
        """duration_ms is always present and non-negative."""
        conn = _setup_db()
        result = vmstat_scan(conn, PROJECT)
        assert "duration_ms" in result
        assert result["duration_ms"] >= 0
        conn.close()
