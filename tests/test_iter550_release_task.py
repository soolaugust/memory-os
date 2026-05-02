"""
iter550: release_task — Per-Session Runtime State Cleanup

OS 类比：Linux release_task() (kernel/exit.c) — 进程退出时清理 per-process 运行时状态。
systemd-tmpfiles --clean 补充清理 /tmp/ 和 /run/ 中的累积文件。

测试覆盖：
- Phase 1: shadow_file_gc (4 tests)
- Phase 2: shadow_db_dedup (3 tests)
- Phase 3: session_episodes_gc (2 tests)
- Phase 4: checkpoint_gc (2 tests)
- Cross-cutting (3 tests): config, idempotent, performance
"""
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

# ── tmpfs isolation ──
sys.path.insert(0, str(Path(__file__).parent.parent))
import tmpfs  # noqa: E402,F401 — must import before store

from store_core import open_db, ensure_schema, MEMORY_OS_DIR, _ensure_checkpoint_schema
from store_mm import release_task


@pytest.fixture
def conn():
    """Create an isolated DB connection with schema."""
    db = open_db()
    ensure_schema(db)
    _ensure_checkpoint_schema(db)
    # Ensure shadow_traces table exists
    db.execute("""
        CREATE TABLE IF NOT EXISTS shadow_traces (
            session_id   TEXT PRIMARY KEY,
            project      TEXT NOT NULL DEFAULT '',
            agent_id     TEXT NOT NULL DEFAULT '',
            updated_at   TEXT NOT NULL,
            top_k_ids    TEXT NOT NULL DEFAULT '[]'
        )
    """)
    # Ensure session_episodes table exists
    db.execute("""
        CREATE TABLE IF NOT EXISTS session_episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            workspace_id TEXT,
            updated_at TEXT NOT NULL,
            injected INTEGER DEFAULT 0,
            summary TEXT
        )
    """)
    db.commit()
    yield db
    db.close()


# ── Phase 1: shadow_file_gc ──────────────────────────────────────────

class TestShadowFileGC:
    """Phase 1: 清理超龄 .shadow_trace.*.json 文件"""

    def test_removes_old_files(self, conn):
        """超龄文件被删除"""
        # 创建一个 "旧" shadow trace 文件
        old_file = MEMORY_OS_DIR / ".shadow_trace.old-session-1234.json"
        old_file.write_text(json.dumps({"project": "test", "top_k_ids": ["a"]}))
        # 设置 mtime 为 48h 前
        old_time = time.time() - 48 * 3600
        os.utime(str(old_file), (old_time, old_time))

        result = release_task(conn, "test")
        assert result["phases"]["shadow_file_gc"]["removed"] >= 1
        assert not old_file.exists()

    def test_keeps_recent_files(self, conn):
        """近期文件保留"""
        recent_file = MEMORY_OS_DIR / ".shadow_trace.recent-sess-5678.json"
        recent_file.write_text(json.dumps({"project": "test", "top_k_ids": ["b"]}))
        # mtime 是现在（< 24h 默认阈值）

        result = release_task(conn, "test")
        assert recent_file.exists()

    def test_no_files_no_error(self, conn):
        """没有 shadow 文件时不报错"""
        result = release_task(conn, "test")
        assert "shadow_file_gc" in result["phases"]
        assert result["phases"]["shadow_file_gc"].get("error") is None

    def test_mixed_age_files(self, conn):
        """混合新旧文件，只删旧的"""
        old1 = MEMORY_OS_DIR / ".shadow_trace.mix-old-aaaa.json"
        old2 = MEMORY_OS_DIR / ".shadow_trace.mix-old-bbbb.json"
        new1 = MEMORY_OS_DIR / ".shadow_trace.mix-new-cccc.json"

        for f in [old1, old2, new1]:
            f.write_text(json.dumps({"ids": []}))

        old_time = time.time() - 72 * 3600
        os.utime(str(old1), (old_time, old_time))
        os.utime(str(old2), (old_time, old_time))
        # new1 keeps current mtime

        result = release_task(conn, "test")
        assert result["phases"]["shadow_file_gc"]["removed"] >= 2
        assert not old1.exists()
        assert not old2.exists()
        assert new1.exists()


# ── Phase 2: shadow_db_dedup ─────────────────────────────────────────

class TestShadowDBDedup:
    """Phase 2: shadow_traces 表去重"""

    def test_deduplicates_identical_content(self, conn):
        """相同 top_k_ids 只保留最新 N 条"""
        same_ids = json.dumps(["chunk_a", "chunk_b"])
        now = datetime.now(timezone.utc)
        # 插入 5 条相同内容
        for i in range(5):
            ts = (now - timedelta(hours=i)).isoformat()
            conn.execute(
                "INSERT INTO shadow_traces (session_id, project, agent_id, updated_at, top_k_ids) "
                "VALUES (?, ?, ?, ?, ?)",
                (f"sess-{i}", "test", "agent", ts, same_ids)
            )
        conn.commit()

        result = release_task(conn, "test")
        phase = result["phases"]["shadow_db_dedup"]
        assert phase["before"] == 5
        # 默认 max_per_content=2, 所以删除 3 条
        assert phase["removed"] == 3
        assert phase["after"] == 2

        # 验证保留的是最新 2 条
        remaining = conn.execute(
            "SELECT session_id FROM shadow_traces ORDER BY updated_at DESC"
        ).fetchall()
        assert len(remaining) == 2
        assert remaining[0][0] == "sess-0"  # 最新
        assert remaining[1][0] == "sess-1"  # 次新

    def test_different_content_not_merged(self, conn):
        """不同 top_k_ids 内容不合并"""
        now = datetime.now(timezone.utc)
        for i in range(3):
            conn.execute(
                "INSERT INTO shadow_traces (session_id, project, agent_id, updated_at, top_k_ids) "
                "VALUES (?, ?, ?, ?, ?)",
                (f"unique-{i}", "test", "agent", now.isoformat(), json.dumps([f"chunk_{i}"]))
            )
        conn.commit()

        result = release_task(conn, "test")
        assert result["phases"]["shadow_db_dedup"]["removed"] == 0

    def test_empty_table(self, conn):
        """空表不报错"""
        # 清除可能遗留的数据
        conn.execute("DELETE FROM shadow_traces")
        conn.commit()
        result = release_task(conn, "test")
        assert result["phases"]["shadow_db_dedup"]["before"] == 0
        assert result["phases"]["shadow_db_dedup"]["removed"] == 0


# ── Phase 3: session_episodes_gc ─────────────────────────────────────

class TestSessionEpisodesGC:
    """Phase 3: 已注入旧 episodes 清理"""

    def test_removes_old_injected(self, conn):
        """超龄 + 已注入的 episodes 被删除"""
        old_time = (datetime.now(timezone.utc) - timedelta(hours=96)).isoformat()
        conn.execute(
            "INSERT INTO session_episodes (session_id, updated_at, injected, summary) "
            "VALUES (?, ?, ?, ?)",
            ("old-sess", old_time, 1, "old episode")
        )
        conn.commit()

        result = release_task(conn, "test")
        assert result["phases"]["session_episodes_gc"]["removed"] == 1

    def test_keeps_recent_and_uninjected(self, conn):
        """近期 + 未注入的 episodes 保留"""
        now_time = datetime.now(timezone.utc).isoformat()
        old_time = (datetime.now(timezone.utc) - timedelta(hours=96)).isoformat()
        # 近期已注入 → 保留
        conn.execute(
            "INSERT INTO session_episodes (session_id, updated_at, injected, summary) "
            "VALUES (?, ?, ?, ?)",
            ("recent-injected", now_time, 1, "recent")
        )
        # 旧的未注入 → 保留
        conn.execute(
            "INSERT INTO session_episodes (session_id, updated_at, injected, summary) "
            "VALUES (?, ?, ?, ?)",
            ("old-uninjected", old_time, 0, "old but not injected")
        )
        conn.commit()

        result = release_task(conn, "test")
        assert result["phases"]["session_episodes_gc"]["removed"] == 0
        remaining = conn.execute("SELECT COUNT(*) FROM session_episodes").fetchone()[0]
        assert remaining == 2


# ── Phase 4: checkpoint_gc ───────────────────────────────────────────

class TestCheckpointGC:
    """Phase 4: 超龄已消费 checkpoints 清理"""

    def test_removes_old_consumed(self, conn):
        """超龄 + 已消费的 checkpoints 被删除"""
        old_time = (datetime.now(timezone.utc) - timedelta(hours=72)).isoformat()
        conn.execute(
            "INSERT INTO checkpoints (id, created_at, project, session_id, hit_chunk_ids, consumed) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("cp-old", old_time, "test", "sess1", "[]", 1)
        )
        conn.commit()

        result = release_task(conn, "test")
        assert result["phases"]["checkpoint_gc"]["removed"] == 1

    def test_keeps_unconsumed(self, conn):
        """未消费的 checkpoints 保留（不论年龄）"""
        old_time = (datetime.now(timezone.utc) - timedelta(hours=72)).isoformat()
        conn.execute(
            "INSERT INTO checkpoints (id, created_at, project, session_id, hit_chunk_ids, consumed) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("cp-active", old_time, "test", "sess2", "[\"chunk1\"]", 0)
        )
        conn.commit()

        result = release_task(conn, "test")
        assert result["phases"]["checkpoint_gc"]["removed"] == 0


# ── Cross-cutting ────────────────────────────────────────────────────

class TestCrossCutting:
    """跨阶段验证"""

    def test_config_tunables(self, conn):
        """配置参数可读取"""
        from config import get as _cfg
        assert int(_cfg("release_task.shadow_file_max_age_hours")) == 24
        assert int(_cfg("release_task.shadow_db_max_per_content")) == 2
        assert int(_cfg("release_task.episodes_max_age_hours")) == 72
        assert int(_cfg("release_task.checkpoint_max_age_hours")) == 48

    def test_idempotent(self, conn):
        """连续两次调用第二次无操作"""
        # 第一次有数据清理
        same_ids = json.dumps(["a", "b"])
        now = datetime.now(timezone.utc)
        for i in range(5):
            ts = (now - timedelta(hours=i)).isoformat()
            conn.execute(
                "INSERT INTO shadow_traces (session_id, project, agent_id, updated_at, top_k_ids) "
                "VALUES (?, ?, ?, ?, ?)",
                (f"idem-{i}", "test", "agent", ts, same_ids)
            )
        conn.commit()

        r1 = release_task(conn, "test")
        assert r1["total_cleaned"] > 0

        r2 = release_task(conn, "test")
        assert r2["total_cleaned"] == 0  # 已无可清理

    def test_performance(self, conn):
        """执行时间 < 500ms"""
        # 插入大量测试数据
        same_ids = json.dumps(["x", "y", "z"])
        now = datetime.now(timezone.utc)
        for i in range(50):
            ts = (now - timedelta(hours=i % 24)).isoformat()
            conn.execute(
                "INSERT INTO shadow_traces (session_id, project, agent_id, updated_at, top_k_ids) "
                "VALUES (?, ?, ?, ?, ?)",
                (f"perf-{i}", "test", "agent", ts, same_ids)
            )
        conn.commit()

        t0 = time.time()
        result = release_task(conn, "test")
        elapsed = (time.time() - t0) * 1000

        assert elapsed < 500, f"release_task took {elapsed:.1f}ms (> 500ms)"
        assert result["duration_ms"] < 500
