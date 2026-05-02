"""
iter571: mmap_populate — Probabilistic Cold Page Promotion

OS 类比：Linux MAP_POPULATE / madvise(MADV_WILLNEED) (Linus Torvalds, 2002,
  mm/mmap.c + mm/readahead.c) — mmap(MAP_POPULATE) 主动预填充 cold pages
  到 working set，打破 cold→no_access→cold 死锁循环。

测试矩阵：
  - 周期性触发：counter % interval == 0 时返回 cold chunk
  - 非触发周期：counter % interval != 0 时返回 None
  - importance 门槛过滤：imp < threshold 不返回
  - top_k_ids 排除：已在 top_k 中的不返回
  - ghost chunk 跳过：importance=0 不返回
  - 按 importance 降序选择最高价值 dark page
  - exclude_types 过滤
  - disabled 配置
  - 空 DB 不报错
  - 幂等性：同一 cold chunk 被访问后不再返回
  - project 过滤：只返回当前 project 或 global 的 chunks
  - 性能基线
"""
import os
import sys
import sqlite3
import time
import uuid

import pytest

# ── path setup ─────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("MEMORY_OS_STORE", ":memory:")

from store_mm import mmap_populate


# ── fixtures ───────────────────────────────────────────────────

def _make_db():
    """Create an in-memory DB with required tables."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE memory_chunks (
            id TEXT PRIMARY KEY,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            project TEXT,
            source_session TEXT,
            chunk_type TEXT,
            content TEXT,
            summary TEXT,
            tags TEXT,
            importance REAL,
            retrievability REAL,
            access_count INTEGER DEFAULT 0,
            oom_adj INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    return conn


def _insert_chunk(conn, chunk_id=None, summary="test", project="test_proj",
                  importance=0.8, access_count=0, chunk_type="decision"):
    cid = chunk_id or str(uuid.uuid4())
    conn.execute(
        "INSERT INTO memory_chunks (id, summary, project, importance, "
        "chunk_type, content, access_count) VALUES (?, ?, ?, ?, ?, '', ?)",
        (cid, summary, project, importance, chunk_type, access_count),
    )
    return cid


# ── tests ──────────────────────────────────────────────────────

def test_triggers_on_interval():
    """counter % interval == 0 时返回 cold chunk。"""
    conn = _make_db()
    cid = _insert_chunk(conn, importance=0.9, access_count=0)
    conn.commit()

    # interval=3, counter=3 → 3%3==0 → triggers
    result = mmap_populate(conn, "test_proj", set(), session_recall_count=3)
    assert result is not None
    assert result["id"] == cid


def test_no_trigger_off_interval():
    """counter % interval != 0 时返回 None。"""
    conn = _make_db()
    _insert_chunk(conn, importance=0.9, access_count=0)
    conn.commit()

    # interval=3, counter=1 → 1%3!=0 → no trigger
    result = mmap_populate(conn, "test_proj", set(), session_recall_count=1)
    assert result is None


def test_counter_zero_triggers():
    """counter=0 → 0%interval==0 → triggers（首次召回曝光）。"""
    conn = _make_db()
    cid = _insert_chunk(conn, importance=0.9, access_count=0)
    conn.commit()

    result = mmap_populate(conn, "test_proj", set(), session_recall_count=0)
    assert result is not None


def test_imp_threshold_filters():
    """importance < threshold 的 chunks 不返回。"""
    conn = _make_db()
    _insert_chunk(conn, importance=0.3, access_count=0)  # below default 0.5
    conn.commit()

    result = mmap_populate(conn, "test_proj", set(), session_recall_count=0)
    assert result is None


def test_excludes_top_k_ids():
    """已在 top_k 中的 chunk 不返回。"""
    conn = _make_db()
    cid = _insert_chunk(conn, importance=0.9, access_count=0)
    conn.commit()

    result = mmap_populate(conn, "test_proj", {cid}, session_recall_count=0)
    assert result is None


def test_ghost_skipped():
    """importance=0 的 ghost chunk 不返回。"""
    conn = _make_db()
    _insert_chunk(conn, importance=0.0, access_count=0)
    conn.commit()

    result = mmap_populate(conn, "test_proj", set(), session_recall_count=0)
    assert result is None


def test_selects_highest_importance():
    """多个候选时，选择 importance 最高的。"""
    conn = _make_db()
    _insert_chunk(conn, summary="low", importance=0.6, access_count=0)
    cid_high = _insert_chunk(conn, summary="high", importance=0.95, access_count=0)
    _insert_chunk(conn, summary="mid", importance=0.7, access_count=0)
    conn.commit()

    result = mmap_populate(conn, "test_proj", set(), session_recall_count=0)
    assert result is not None
    assert result["id"] == cid_high


def test_skips_accessed_chunks():
    """access_count > 0 的 chunk 不是 cold page，不返回。"""
    conn = _make_db()
    _insert_chunk(conn, importance=0.9, access_count=5)  # not cold
    conn.commit()

    result = mmap_populate(conn, "test_proj", set(), session_recall_count=0)
    assert result is None


def test_exclude_types():
    """exclude_types 配置的 chunk_type 被过滤。"""
    conn = _make_db()
    _insert_chunk(conn, importance=0.9, access_count=0,
                  chunk_type="prompt_context")
    _insert_chunk(conn, importance=0.9, access_count=0,
                  chunk_type="conversation_summary")
    conn.commit()

    # 默认排除 prompt_context,conversation_summary
    result = mmap_populate(conn, "test_proj", set(), session_recall_count=0)
    assert result is None


def test_non_excluded_type_returned():
    """非 exclude_types 的类型正常返回。"""
    conn = _make_db()
    cid = _insert_chunk(conn, importance=0.9, access_count=0,
                        chunk_type="quantitative_evidence")
    conn.commit()

    result = mmap_populate(conn, "test_proj", set(), session_recall_count=0)
    assert result is not None
    assert result["id"] == cid


def test_disabled():
    """enabled=False 时不触发。"""
    conn = _make_db()
    _insert_chunk(conn, importance=0.9, access_count=0)
    conn.commit()

    # Monkey-patch config
    import config
    original = config._REGISTRY.get("mmap_populate.enabled")
    config._REGISTRY["mmap_populate.enabled"] = (False, bool, None, None, None, "")
    try:
        result = mmap_populate(conn, "test_proj", set(), session_recall_count=0)
        assert result is None
    finally:
        if original:
            config._REGISTRY["mmap_populate.enabled"] = original
        else:
            del config._REGISTRY["mmap_populate.enabled"]


def test_empty_db():
    """空 DB 不报错，返回 None。"""
    conn = _make_db()
    result = mmap_populate(conn, "test_proj", set(), session_recall_count=0)
    assert result is None


def test_project_filter():
    """只返回当前 project 或 global 的 chunks。"""
    conn = _make_db()
    _insert_chunk(conn, importance=0.9, access_count=0, project="other_proj")
    cid_global = _insert_chunk(conn, importance=0.85, access_count=0, project="global")
    conn.commit()

    result = mmap_populate(conn, "test_proj", set(), session_recall_count=0)
    assert result is not None
    assert result["id"] == cid_global


def test_current_project_included():
    """当前 project 的 chunks 也可以被选中。"""
    conn = _make_db()
    cid = _insert_chunk(conn, importance=0.9, access_count=0, project="test_proj")
    conn.commit()

    result = mmap_populate(conn, "test_proj", set(), session_recall_count=0)
    assert result is not None
    assert result["id"] == cid


def test_interval_periodicity():
    """验证周期性：只在 0, 3, 6, 9... 时触发（interval=3）。"""
    conn = _make_db()
    _insert_chunk(conn, importance=0.9, access_count=0)
    conn.commit()

    triggers = []
    for i in range(12):
        r = mmap_populate(conn, "test_proj", set(), session_recall_count=i)
        if r is not None:
            triggers.append(i)

    assert triggers == [0, 3, 6, 9]


def test_performance():
    """100 chunks 扫描 < 50ms。"""
    conn = _make_db()
    for i in range(100):
        _insert_chunk(conn, importance=0.5 + i * 0.004, access_count=0)
    conn.commit()

    t0 = time.time()
    for _ in range(10):
        mmap_populate(conn, "test_proj", set(), session_recall_count=0)
    elapsed = (time.time() - t0) * 1000 / 10

    assert elapsed < 50, f"mmap_populate took {elapsed:.1f}ms (limit 50ms)"
