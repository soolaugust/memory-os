#!/usr/bin/env python3
"""
Phase 2 完整测试集

验证：
- VFSItem 序列化 ✓
- SQLiteBackend 搜索和读取 ✓
- KnowledgeVFS 缓存和去重 ✓
- 硬 deadline 保障 ✓
"""
import pytest
import time
import tempfile
from pathlib import Path
from datetime import datetime, timezone

from vfs_core import VFSItem, VFSMetadata, VFSItemType, VFSSource
from vfs_backend_sqlite import SQLiteBackend
from vfs import KnowledgeVFS, DentryCache, InodeCache


class TestVFSItem:
    """测试 VFSItem 序列化"""

    def test_from_chunk_conversion(self):
        """测试从 chunk dict 转换为 VFSItem"""
        chunk_dict = {
            "id": "chunk-001",
            "chunk_type": "decision",
            "content": "使用 BM25 作为检索引擎",
            "summary": "选择 BM25",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "last_accessed": datetime.now(timezone.utc).isoformat(),
            "importance": 0.85,
            "retrievability": 0.45,
            "access_count": 5,
            "source_session": "sess-001",
            "tags": '["bm25"]',
            "project": "git:abc123",
        }

        item = VFSItem.from_chunk(chunk_dict, score=0.92)

        assert item.id == "chunk-001"
        assert item.type == "decision"
        assert item.score == 0.92
        assert item.path == "/memory-os/chunk-001"
        assert item.metadata.importance == 0.85

    def test_serialization_roundtrip(self):
        """测试序列化和反序列化"""
        original_item = VFSItem(
            id="test-001",
            type="decision",
            source="memory-os",
            content="Test content",
            summary="Test summary",
            metadata=VFSMetadata(
                created_at=datetime.now(timezone.utc).isoformat(),
                updated_at=datetime.now(timezone.utc).isoformat(),
                last_accessed=datetime.now(timezone.utc).isoformat(),
                importance=0.8,
                retrievability=0.6,
                access_count=3,
                source_session="sess",
                scope="project",
                tags=["test"],
                project="proj",
                content_hash="hash123",
            ),
            path="/memory-os/test-001",
            score=0.95,
        )

        # 序列化到 JSON
        json_str = original_item.to_json()
        assert isinstance(json_str, str)
        assert "test-001" in json_str

        # 从 JSON 反序列化
        restored_item = VFSItem.from_json(json_str)

        assert restored_item.id == original_item.id
        assert restored_item.type == original_item.type
        assert restored_item.score == original_item.score
        assert restored_item.metadata.importance == original_item.metadata.importance


class TestDentryCache:
    """测试 dentry 缓存"""

    def test_dentry_cache_hit(self):
        """测试缓存命中"""
        cache = DentryCache(ttl_secs=300)
        item = VFSItem(
            id="test",
            type="decision",
            source="memory-os",
            content="test",
            summary="test",
            metadata=VFSMetadata(
                created_at="2026-01-01T00:00:00Z",
                updated_at="2026-01-01T00:00:00Z",
                last_accessed="2026-01-01T00:00:00Z",
                importance=0.5,
                retrievability=0.5,
                access_count=0,
                source_session="",
                scope="project",
                tags=[],
                project="",
                content_hash="",
            ),
            path="/memory-os/test",
            score=1.0,
        )

        cache.put("/path/to/item", item)
        cached = cache.get("/path/to/item")

        assert cached is not None
        assert cached.id == "test"

    def test_dentry_cache_ttl_expiry(self):
        """测试 TTL 过期"""
        cache = DentryCache(ttl_secs=1)  # 1 秒 TTL
        item = VFSItem(
            id="test",
            type="decision",
            source="memory-os",
            content="test",
            summary="test",
            metadata=VFSMetadata(
                created_at="2026-01-01T00:00:00Z",
                updated_at="2026-01-01T00:00:00Z",
                last_accessed="2026-01-01T00:00:00Z",
                importance=0.5,
                retrievability=0.5,
                access_count=0,
                source_session="",
                scope="project",
                tags=[],
                project="",
                content_hash="",
            ),
            path="/memory-os/test",
            score=1.0,
        )

        cache.put("/path/to/item", item)
        time.sleep(1.1)
        cached = cache.get("/path/to/item")

        assert cached is None


class TestSQLiteBackend:
    """测试 SQLite 后端"""

    def test_search_returns_results(self):
        """测试搜索返回结果 — 使用 tmpfs 隔离 DB，不依赖生产数据"""
        import sqlite3 as _sq
        import os as _os
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent.parent))

        # 创建独立的 tmpdir，不依赖任何环境变量（防止前置 test 污染）
        tmpdir = Path(tempfile.mkdtemp(prefix="test_vfs_search_"))
        db_path = tmpdir / "store.db"
        _os.environ["MEMORY_OS_DIR"] = str(tmpdir)
        _os.environ["MEMORY_OS_DB"] = str(db_path)

        # 直接用 sqlite3 初始化 DB（绕过模块缓存，避免 open_db() 用旧路径）
        import sqlite3 as _sq3
        from datetime import datetime, timezone
        conn = _sq3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""CREATE TABLE IF NOT EXISTS memory_chunks (
            id TEXT PRIMARY KEY, chunk_type TEXT, summary TEXT, content TEXT,
            project TEXT, importance REAL, retrievability REAL, access_count INTEGER,
            source_session TEXT, tags TEXT, lru_gen INTEGER DEFAULT 0,
            oom_adj INTEGER DEFAULT 0, created_at TEXT, updated_at TEXT, last_accessed TEXT
        )""")
        conn.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS memory_chunks_fts
            USING fts5(rowid_ref UNINDEXED, summary, content)""")
        now = datetime.now(timezone.utc).isoformat()
        conn.execute("""INSERT OR IGNORE INTO memory_chunks
            (id, chunk_type, summary, content, project, importance, retrievability,
             access_count, source_session, created_at, updated_at, last_accessed)
            VALUES ('test-bm25-001','decision',
              'BM25 retrieval engine selection',
              'BM25 is an efficient full-text ranking algorithm for sparse retrieval',
              'test', 0.85, 0.7, 3, 'sess-test', ?, ?, ?)""", (now, now, now))
        conn.execute("""INSERT OR IGNORE INTO memory_chunks
            (id, chunk_type, summary, content, project, importance, retrievability,
             access_count, source_session, created_at, updated_at, last_accessed)
            VALUES ('test-bm25-002','reasoning_chain',
              'BM25 vs vector retrieval tradeoffs',
              'BM25 matches keywords exactly; vector search captures semantic similarity',
              'test', 0.75, 0.6, 2, 'sess-test', ?, ?, ?)""", (now, now, now))
        # 插入 FTS 数据（rowid_ref 关联 memory_chunks.rowid）
        conn.execute("""INSERT OR IGNORE INTO memory_chunks_fts (rowid_ref, summary, content)
            SELECT rowid, summary, content FROM memory_chunks
            WHERE id IN ('test-bm25-001', 'test-bm25-002')""")
        conn.commit()
        conn.close()

        backend = SQLiteBackend(db_path=db_path, readonly=False)
        results = backend.search("BM25", top_k=5)

        assert len(results) > 0, f"Expected results for 'BM25', got 0 (db={db_path})"
        assert all(isinstance(item, VFSItem) for item in results)
        # Results should be sorted by score (descending)
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_read_by_path(self):
        """测试按路径读取"""
        db_path = Path.home() / ".claude" / "memory-os" / "store.db"
        if not db_path.exists():
            pytest.skip("Store DB not found")

        backend = SQLiteBackend(db_path=db_path, readonly=True)

        # 先搜索获取一个有效的 path
        results = backend.search("BM25", top_k=1)
        if not results:
            pytest.skip("No search results")

        path = results[0].path
        item = backend.read(path)

        assert item is not None
        assert item.id == results[0].id
        assert item.type == results[0].type


class TestKnowledgeVFS:
    """测试 KnowledgeVFS 核心"""

    def test_vfs_initialization(self):
        """测试 VFS 初始化"""
        vfs = KnowledgeVFS()
        assert vfs is not None
        assert vfs.dentry_cache is not None
        assert vfs.inode_cache is not None

    def test_vfs_search_within_deadline(self):
        """测试搜索在硬 deadline 内完成"""
        db_path = Path.home() / ".claude" / "memory-os" / "store.db"
        if not db_path.exists():
            pytest.skip("Store DB not found")

        vfs = KnowledgeVFS()
        start = time.time()
        results = vfs.search("BM25", top_k=3, deadline_ms=100)
        elapsed_ms = (time.time() - start) * 1000

        assert elapsed_ms < 100, f"Search took {elapsed_ms}ms, exceeds 100ms deadline"
        assert len(results) > 0

    def test_cache_hit_latency(self):
        """测试缓存命中延迟 < 1ms"""
        db_path = Path.home() / ".claude" / "memory-os" / "store.db"
        if not db_path.exists():
            pytest.skip("Store DB not found")

        vfs = KnowledgeVFS()

        # 第一次搜索（缓存 miss）
        results1 = vfs.search("BM25", top_k=1, deadline_ms=100)
        if not results1:
            pytest.skip("No search results")

        # 第二次读取（缓存 hit）
        path = results1[0].path
        start = time.time()
        item = vfs.read(path)
        elapsed_ms = (time.time() - start) * 1000

        assert item is not None
        assert elapsed_ms < 1.0, f"Cached read took {elapsed_ms}ms, exceeds 1ms target"

    def test_deduplication(self):
        """测试去重"""
        vfs = KnowledgeVFS()
        results = vfs.search("decision", top_k=10, deadline_ms=100)

        # 检查 summary 去重
        summaries = set()
        for item in results:
            summary_hash = hash(item.summary)
            assert summary_hash not in summaries, f"Duplicate summary: {item.summary}"
            summaries.add(summary_hash)

    def test_stats_collection(self):
        """测试统计收集"""
        vfs = KnowledgeVFS()
        vfs.search("test", top_k=1)
        vfs.read("/memory-os/nonexistent")

        stats = vfs.stats()
        assert stats["searches"] == 1
        assert stats["reads"] == 1


# ── 性能基准（Phase 2F 验证）──────────────────────────────────────
def benchmark_vfs():
    """VFS 性能基准"""
    db_path = Path.home() / ".claude" / "memory-os" / "store.db"
    if not db_path.exists():
        print("⚠ Store DB not found, skipping benchmark")
        return

    vfs = KnowledgeVFS()

    # Benchmark: 搜索延迟
    queries = ["BM25", "decision", "extractor", "TLB", "optimization"]
    search_times = []

    for query in queries:
        start = time.time()
        results = vfs.search(query, top_k=5, deadline_ms=100)
        elapsed_ms = (time.time() - start) * 1000
        search_times.append(elapsed_ms)

    print("\n📊 VFS Performance Benchmark:")
    print(f"  Average search latency: {sum(search_times) / len(search_times):.1f}ms")
    print(f"  Max search latency: {max(search_times):.1f}ms")
    print(f"  All within 100ms deadline: {all(t < 100 for t in search_times)}")

    # Benchmark: 缓存命中
    if results:
        path = results[0].path
        start = time.time()
        item = vfs.read(path)
        elapsed_ms = (time.time() - start) * 1000
        print(f"  Cached read latency: {elapsed_ms:.3f}ms")

    stats = vfs.stats()
    print(f"  Dentry cache: {stats['dentry_cache']['valid']}/{stats['dentry_cache']['total']} valid")
    print(f"  Inode cache: {stats['inode_cache']['valid']}/{stats['inode_cache']['total']} valid")


if __name__ == "__main__":
    # 运行测试
    print("Running Phase 2 test suite...\n")
    pytest.main([__file__, "-v"])

    # 性能基准
    print("\n" + "=" * 60)
    benchmark_vfs()
