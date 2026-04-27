#!/usr/bin/env python3
"""
test_mount_walk.py — 迭代81: VFS Mount Point Resolution 测试

OS 类比：Linux VFS lookup_mnt() — 从子目录 CWD 向上遍历找到正确的 project_id
验证 save-task-state.py 和 resume-task-state.py 的 mount walk 修复
"""
import tmpfs  # noqa: F401 — must be first to isolate test DB

import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Add both hook dirs and memory-os to path
sys.path.insert(0, str(Path(__file__).parent))
_HOOKS_DIR = Path.home() / ".claude" / "hooks"
sys.path.insert(0, str(_HOOKS_DIR))

from store import open_db, ensure_schema
from utils import resolve_project_id


def _setup_test_db_with_project(project_id, n_chunks=5, n_traces=3):
    """Create a test DB with chunks and traces for a specific project."""
    conn = open_db()
    ensure_schema(conn)

    # Clean slate — avoid cross-test pollution in shared tmpfs DB
    conn.execute("DELETE FROM memory_chunks")
    conn.execute("DELETE FROM recall_traces")
    conn.commit()

    for i in range(n_chunks):
        conn.execute("""
            INSERT INTO memory_chunks
            (project, source_session, chunk_type, content, summary, tags, importance,
             retrievability, last_accessed, access_count, oom_adj, lru_gen)
            VALUES (?, 'test-sess', 'decision', ?, ?, '[]', 0.9,
                    1.0, datetime('now'), ?, 0, 0)
        """, (project_id, f"content-{i}", f"decision-summary-{i}", i))

    for i in range(n_traces):
        top_k = json.dumps([{"id": f"chunk-{i}", "summary": f"s-{i}", "score": 0.8}])
        conn.execute("""
            INSERT INTO recall_traces
            (timestamp, session_id, project, prompt_hash, candidates_count,
             top_k_json, injected, reason, duration_ms)
            VALUES (datetime('now'), 'test-sess', ?, ?, 10, ?, 1, 'hash_changed', 5.0)
        """, (project_id, f"hash-{i}", top_k))

    conn.commit()
    return conn


class TestMountWalkSave(unittest.TestCase):
    """Test _resolve_project_with_mount_walk in save-task-state.py"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.root_dir = Path(self.tmpdir) / "project-root"
        self.sub_dir = self.root_dir / "src" / "submodule"
        self.sub_dir.mkdir(parents=True)

        self.root_project = resolve_project_id(str(self.root_dir))
        self.sub_project = resolve_project_id(str(self.sub_dir))

        # Verify they're different
        self.assertNotEqual(self.root_project, self.sub_project,
                            "Root and sub project IDs must differ for test to be valid")

        self.conn = _setup_test_db_with_project(self.root_project, n_chunks=5, n_traces=3)

    def tearDown(self):
        self.conn.close()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _get_mount_walk_fn(self):
        """Import the function from save-task-state.py"""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "save_task_state", str(_HOOKS_DIR / "save-task-state.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod._resolve_project_with_mount_walk

    def test_direct_match(self):
        """Root dir CWD resolves directly — no walk needed."""
        fn = self._get_mount_walk_fn()
        result = fn(str(self.root_dir), self.conn)
        self.assertEqual(result, self.root_project)

    def test_subdir_walks_up(self):
        """Sub-directory CWD walks up to find root project."""
        fn = self._get_mount_walk_fn()
        result = fn(str(self.sub_dir), self.conn)
        self.assertEqual(result, self.root_project)

    def test_deep_subdir(self):
        """Very deep sub-directory still finds root."""
        fn = self._get_mount_walk_fn()
        deep = self.sub_dir / "a" / "b" / "c" / "d"
        deep.mkdir(parents=True)
        result = fn(str(deep), self.conn)
        self.assertEqual(result, self.root_project)

    def test_unrelated_dir_fallback(self):
        """Completely unrelated dir falls back to most active project."""
        fn = self._get_mount_walk_fn()
        unrelated = tempfile.mkdtemp()
        try:
            result = fn(unrelated, self.conn)
            # Should fallback to the only project in DB
            self.assertEqual(result, self.root_project)
        finally:
            os.rmdir(unrelated)

    def test_zzz_empty_db_returns_original(self):
        """Empty DB returns original CWD's project_id. (zzz prefix: runs last to avoid data pollution)"""
        fn = self._get_mount_walk_fn()
        # Delete all chunks via the test conn
        self.conn.execute("DELETE FROM memory_chunks")
        self.conn.commit()
        result = fn(str(self.sub_dir), self.conn)
        # Should return the sub_dir's own project ID (no data anywhere)
        self.assertEqual(result, self.sub_project)


class TestMountWalkResume(unittest.TestCase):
    """Test _resolve_project_with_mount_walk in resume-task-state.py"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.root_dir = Path(self.tmpdir) / "project-root"
        self.sub_dir = self.root_dir / "nested" / "deep"
        self.sub_dir.mkdir(parents=True)

        self.root_project = resolve_project_id(str(self.root_dir))
        self.conn = _setup_test_db_with_project(self.root_project, n_chunks=3)

    def tearDown(self):
        self.conn.close()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _get_mount_walk_fn(self):
        """Import the function from resume-task-state.py"""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "resume_task_state", str(_HOOKS_DIR / "resume-task-state.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod._resolve_project_with_mount_walk

    def test_subdir_resolves_to_root(self):
        """Resume from subdir CWD finds root project."""
        fn = self._get_mount_walk_fn()
        result = fn(str(self.sub_dir), self.conn)
        self.assertEqual(result, self.root_project)

    def test_direct_match(self):
        """Direct root CWD resolves immediately."""
        fn = self._get_mount_walk_fn()
        result = fn(str(self.root_dir), self.conn)
        self.assertEqual(result, self.root_project)


class TestMountWalkMultiProject(unittest.TestCase):
    """Test mount walk with multiple projects in DB."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.proj_a = Path(self.tmpdir) / "project-a"
        self.proj_b = Path(self.tmpdir) / "project-b"
        self.proj_a.mkdir()
        self.proj_b.mkdir()

        self.pid_a = resolve_project_id(str(self.proj_a))
        self.pid_b = resolve_project_id(str(self.proj_b))

        self.conn = open_db()
        ensure_schema(self.conn)

        # Clean slate
        self.conn.execute("DELETE FROM memory_chunks")
        self.conn.execute("DELETE FROM recall_traces")
        self.conn.commit()

        # Project A: 10 chunks
        for i in range(10):
            self.conn.execute("""
                INSERT INTO memory_chunks
                (project, source_session, chunk_type, content, summary, tags, importance,
                 retrievability, last_accessed, access_count, oom_adj, lru_gen)
                VALUES (?, 'sa', 'decision', ?, ?, '[]', 0.9, 1.0, datetime('now'), 0, 0, 0)
            """, (self.pid_a, f"ca-{i}", f"summary-a-{i}"))

        # Project B: 3 chunks
        for i in range(3):
            self.conn.execute("""
                INSERT INTO memory_chunks
                (project, source_session, chunk_type, content, summary, tags, importance,
                 retrievability, last_accessed, access_count, oom_adj, lru_gen)
                VALUES (?, 'sb', 'decision', ?, ?, '[]', 0.9, 1.0, datetime('now'), 0, 0, 0)
            """, (self.pid_b, f"cb-{i}", f"summary-b-{i}"))
        self.conn.commit()

    def tearDown(self):
        self.conn.close()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _get_mount_walk_fn(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "save_task_state", str(_HOOKS_DIR / "save-task-state.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod._resolve_project_with_mount_walk

    def test_subdir_of_a_resolves_to_a(self):
        """Subdir of project A resolves to A, not B."""
        fn = self._get_mount_walk_fn()
        sub_a = self.proj_a / "src" / "lib"
        sub_a.mkdir(parents=True)
        result = fn(str(sub_a), self.conn)
        self.assertEqual(result, self.pid_a)

    def test_subdir_of_b_resolves_to_b(self):
        """Subdir of project B resolves to B, not A."""
        fn = self._get_mount_walk_fn()
        sub_b = self.proj_b / "test"
        sub_b.mkdir(parents=True)
        result = fn(str(sub_b), self.conn)
        self.assertEqual(result, self.pid_b)

    def test_unrelated_dir_picks_most_active(self):
        """Unrelated dir fallback picks the project with most chunks."""
        fn = self._get_mount_walk_fn()
        unrelated = tempfile.mkdtemp()
        try:
            result = fn(unrelated, self.conn)
            # Project A has 10 chunks > B's 3
            self.assertEqual(result, self.pid_a)
        finally:
            os.rmdir(unrelated)


class TestSwapIntegrationWithMountWalk(unittest.TestCase):
    """End-to-end test: swap out from subdir CWD produces valid hit_ids."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.root_dir = Path(self.tmpdir) / "workspace"
        self.sub_dir = self.root_dir / "aios" / "memory-os"
        self.sub_dir.mkdir(parents=True)

        self.root_project = resolve_project_id(str(self.root_dir))
        self.conn = _setup_test_db_with_project(self.root_project, n_chunks=8, n_traces=5)

    def tearDown(self):
        self.conn.close()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_collect_swap_state_from_subdir(self):
        """_collect_swap_state with subdir CWD returns actual hit_ids and decisions."""
        # Close test connection first so _open_db_readonly can open the DB
        self.conn.close()

        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "save_task_state", str(_HOOKS_DIR / "save-task-state.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Patch STORE_DB to use test DB (same path as store.open_db uses)
        from store import STORE_DB as store_db_path
        mod.STORE_DB = Path(store_db_path)
        mod.LATEST_JSON = Path(self.tmpdir) / "latest.json"  # non-existent
        mod.MEMORY_OS_DIR = mod.STORE_DB.parent

        stdin_data = {
            "cwd": str(self.sub_dir),  # subdir CWD — the bug trigger
            "session_id": "test-sess",
        }
        swap = mod._collect_swap_state(stdin_data)

        # Before fix: 0 hit_ids, 0 decisions
        # After fix: should have data from root project
        self.assertGreater(len(swap["hit_chunk_ids"]), 0,
                           f"Expected hit_ids > 0 (was 0 before mount walk fix)")
        self.assertGreater(len(swap["decisions"]), 0,
                           f"Expected decisions > 0 (was 0 before mount walk fix)")

        # Re-open conn for tearDown
        self.conn = open_db()


if __name__ == "__main__":
    unittest.main(verbosity=2)
