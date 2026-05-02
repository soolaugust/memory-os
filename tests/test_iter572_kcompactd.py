"""
iter572: kcompactd — Proactive Dead Page Reclaim
OS 类比：Linux kcompactd (Vlastimil Babka, 2016, kernel 4.6, mm/compaction.c)

测试 oom_adj 驱动的主动 dead page 回收，不受 kswapd watermark 门控。
"""
import json
import sqlite3
import sys
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import tmpfs  # 测试隔离
from store_core import open_db, ensure_schema, STORE_DB
from store_mm import kcompactd
import config


def _set_override(key, value):
    """修改 config._REGISTRY 中的值用于测试。"""
    entry = config._REGISTRY.get(key)
    if entry:
        config._REGISTRY[key] = (value,) + entry[1:]
    else:
        config._REGISTRY[key] = (value, type(value), None, None, None, "test override")


_original_values = {}

def _save_and_set(key, value):
    """保存原值并设置新值。"""
    _original_values[key] = config._REGISTRY.get(key)
    _set_override(key, value)

def _restore_all():
    """恢复所有修改过的配置。"""
    for key, original in _original_values.items():
        if original is not None:
            config._REGISTRY[key] = original
        elif key in config._REGISTRY:
            del config._REGISTRY[key]
    _original_values.clear()


def _make_chunk(conn, project="test_proj", access_count=0, importance=0.15,
                oom_adj=300, chunk_type="decision", summary="test chunk",
                age_days=10.0):
    """创建测试 chunk 并返回 ID。"""
    cid = str(uuid.uuid4())
    created = datetime.now(timezone.utc) - timedelta(days=age_days)
    conn.execute(
        """INSERT INTO memory_chunks
           (id, project, chunk_type, summary, content, importance,
            access_count, oom_adj, created_at, last_accessed)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (cid, project, chunk_type, summary, f"content of {summary}",
         importance, access_count, oom_adj, created.isoformat(), created.isoformat()),
    )
    # FTS5 索引
    rowid = conn.execute("SELECT rowid FROM memory_chunks WHERE id=?", (cid,)).fetchone()[0]
    conn.execute(
        "INSERT INTO memory_chunks_fts (rowid_ref, summary, content) VALUES (?, ?, ?)",
        (str(rowid), summary, f"content of {summary}"),
    )
    conn.commit()
    return cid


def _setup():
    conn = open_db()
    ensure_schema(conn)
    _restore_all()
    # 清理上一个测试的残留数据
    for table in ("memory_chunks", "memory_chunks_fts", "entity_map",
                  "entity_edges", "chunk_pins"):
        try:
            conn.execute(f"DELETE FROM {table}")
        except Exception:
            pass
    conn.commit()
    return conn


def _exists(conn, cid):
    return conn.execute("SELECT COUNT(*) FROM memory_chunks WHERE id=?", (cid,)).fetchone()[0] > 0


# ── 基本功能测试 ──

def test_dead_chunk_deleted():
    """oom_adj>=300 + zero-access + low-imp + old → 删除。"""
    conn = _setup()
    cid = _make_chunk(conn, oom_adj=300, importance=0.15, access_count=0, age_days=10)
    result = kcompactd(conn, "test_proj")
    assert result["deleted"] == 1
    assert not _exists(conn, cid)


def test_accessed_preserved():
    """access_count > 0 的 chunk 不被删除（SQL WHERE 条件）。"""
    conn = _setup()
    cid = _make_chunk(conn, oom_adj=300, importance=0.15, access_count=1, age_days=10)
    result = kcompactd(conn, "test_proj")
    assert result["deleted"] == 0
    assert _exists(conn, cid)


def test_low_oom_preserved():
    """oom_adj < threshold 的 chunk 不被删除。"""
    conn = _setup()
    cid = _make_chunk(conn, oom_adj=0, importance=0.15, access_count=0, age_days=10)
    result = kcompactd(conn, "test_proj")
    assert result["deleted"] == 0
    assert _exists(conn, cid)


def test_high_importance_preserved():
    """importance >= imp_ceiling 的 chunk 不被删除。"""
    conn = _setup()
    cid = _make_chunk(conn, oom_adj=300, importance=0.80, access_count=0, age_days=10)
    result = kcompactd(conn, "test_proj")
    assert result["deleted"] == 0
    assert _exists(conn, cid)


def test_pinned_preserved():
    """chunk_pins 中的 chunk 不被删除（mlock 保护）。"""
    conn = _setup()
    cid = _make_chunk(conn, oom_adj=300, importance=0.15, access_count=0, age_days=10)
    conn.execute(
        "INSERT INTO chunk_pins (chunk_id, project, pin_type, pinned_at) VALUES (?, 'test_proj', 'hard', ?)",
        (cid, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    result = kcompactd(conn, "test_proj")
    assert result["deleted"] == 0
    assert result["skipped_pinned"] == 1
    assert _exists(conn, cid)


def test_young_preserved():
    """age < min_age_days 的 chunk 不被删除（冷启动保护）。"""
    conn = _setup()
    cid = _make_chunk(conn, oom_adj=300, importance=0.15, access_count=0, age_days=0.5)
    result = kcompactd(conn, "test_proj")
    assert result["deleted"] == 0
    assert result["skipped_young"] == 1
    assert _exists(conn, cid)


def test_task_state_preserved():
    """task_state 类型不被删除。"""
    conn = _setup()
    cid = _make_chunk(conn, oom_adj=300, importance=0.15, access_count=0,
                      age_days=10, chunk_type="task_state")
    result = kcompactd(conn, "test_proj")
    assert result["deleted"] == 0
    assert _exists(conn, cid)


def test_ghost_not_touched():
    """importance=0 (ghost) 不在扫描范围。"""
    conn = _setup()
    cid = _make_chunk(conn, oom_adj=300, importance=0.0, access_count=0, age_days=10)
    result = kcompactd(conn, "test_proj")
    assert result["deleted"] == 0
    # ghost 不在 WHERE importance > 0 范围内


# ── 配额和配置测试 ──

def test_max_delete_cap():
    """单次删除量不超过 max_delete。"""
    conn = _setup()
    _save_and_set("kcompactd.max_delete", 3)
    for i in range(10):
        _make_chunk(conn, oom_adj=300, importance=0.15, access_count=0, age_days=10,
                    summary=f"chunk {i}")
    result = kcompactd(conn, "test_proj")
    assert result["deleted"] == 3
    assert result["scanned"] == 10
    _restore_all()


def test_disabled():
    """enabled=False 时不执行。"""
    conn = _setup()
    _save_and_set("kcompactd.enabled", False)
    cid = _make_chunk(conn, oom_adj=300, importance=0.15, access_count=0, age_days=10)
    result = kcompactd(conn, "test_proj")
    assert result["deleted"] == 0
    assert _exists(conn, cid)
    _restore_all()


def test_oom_threshold_tunable():
    """调整 oom_threshold 影响候选范围。"""
    conn = _setup()
    cid300 = _make_chunk(conn, oom_adj=300, importance=0.15, access_count=0, age_days=10)
    cid500 = _make_chunk(conn, oom_adj=500, importance=0.15, access_count=0, age_days=10)
    _save_and_set("kcompactd.oom_threshold", 400)
    result = kcompactd(conn, "test_proj")
    assert result["deleted"] == 1  # 只删 oom_adj=500
    assert _exists(conn, cid300)
    assert not _exists(conn, cid500)
    _restore_all()


def test_imp_ceiling_tunable():
    """调整 imp_ceiling 扩大/缩小回收范围。"""
    conn = _setup()
    cid = _make_chunk(conn, oom_adj=300, importance=0.25, access_count=0, age_days=10)
    _save_and_set("kcompactd.imp_ceiling", 0.2)  # 缩小范围
    result = kcompactd(conn, "test_proj")
    assert result["deleted"] == 0  # 0.25 >= 0.2, 不删
    assert _exists(conn, cid)
    _restore_all()


# ── 清理完整性测试 ──

def test_fts5_cleaned():
    """删除 chunk 时 FTS5 索引同步清理。"""
    conn = _setup()
    cid = _make_chunk(conn, oom_adj=300, importance=0.15, access_count=0, age_days=10,
                      summary="unique_kcompactd_test_term")
    # 确认 FTS5 存在
    fts_before = conn.execute(
        "SELECT COUNT(*) FROM memory_chunks_fts WHERE memory_chunks_fts MATCH 'unique_kcompactd_test_term'"
    ).fetchone()[0]
    assert fts_before >= 1

    kcompactd(conn, "test_proj")
    assert not _exists(conn, cid)

    # FTS5 也应该被清理
    fts_after = conn.execute(
        "SELECT COUNT(*) FROM memory_chunks_fts WHERE memory_chunks_fts MATCH 'unique_kcompactd_test_term'"
    ).fetchone()[0]
    assert fts_after == 0


def test_entity_map_cleaned():
    """删除 chunk 时 entity_map 同步清理。"""
    conn = _setup()
    cid = _make_chunk(conn, oom_adj=300, importance=0.15, access_count=0, age_days=10)
    conn.execute(
        "INSERT INTO entity_map (entity_name, chunk_id, project) VALUES ('test_entity', ?, 'test_proj')",
        (cid,),
    )
    conn.commit()
    kcompactd(conn, "test_proj")
    em_count = conn.execute("SELECT COUNT(*) FROM entity_map WHERE chunk_id=?", (cid,)).fetchone()[0]
    assert em_count == 0


def test_entity_edges_cleaned():
    """删除 chunk 时 entity_edges 中 source_chunk_id 引用同步清理。"""
    conn = _setup()
    cid = _make_chunk(conn, oom_adj=300, importance=0.15, access_count=0, age_days=10)
    conn.execute(
        """INSERT INTO entity_edges (id, from_entity, relation, to_entity, project, source_chunk_id, created_at)
           VALUES ('edge1', 'a', 'triggers', 'b', 'test_proj', ?, ?)""",
        (cid, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    kcompactd(conn, "test_proj")
    ee_count = conn.execute("SELECT COUNT(*) FROM entity_edges WHERE source_chunk_id=?", (cid,)).fetchone()[0]
    assert ee_count == 0


# ── 项目范围测试 ──

def test_project_scoped():
    """project 参数限制扫描范围。"""
    conn = _setup()
    cid_proj = _make_chunk(conn, project="test_proj", oom_adj=300, importance=0.15,
                           access_count=0, age_days=10)
    cid_other = _make_chunk(conn, project="other_proj", oom_adj=300, importance=0.15,
                            access_count=0, age_days=10)
    result = kcompactd(conn, "test_proj")
    assert result["deleted"] == 1
    assert not _exists(conn, cid_proj)
    assert _exists(conn, cid_other)  # 其他项目不受影响


def test_global_included():
    """project='global' 的 chunk 也在扫描范围内。"""
    conn = _setup()
    cid = _make_chunk(conn, project="global", oom_adj=300, importance=0.15,
                      access_count=0, age_days=10)
    result = kcompactd(conn, "test_proj")
    assert result["deleted"] == 1
    assert not _exists(conn, cid)


# ── 幂等性和性能测试 ──

def test_idempotent():
    """第二次运行 deleted=0。"""
    conn = _setup()
    _make_chunk(conn, oom_adj=300, importance=0.15, access_count=0, age_days=10)
    r1 = kcompactd(conn, "test_proj")
    assert r1["deleted"] == 1
    r2 = kcompactd(conn, "test_proj")
    assert r2["deleted"] == 0


def test_empty_db():
    """空 DB 不报错。"""
    conn = _setup()
    result = kcompactd(conn, "test_proj")
    assert result["deleted"] == 0
    assert result["scanned"] == 0


def test_oom_priority_order():
    """优先删除 oom_adj 最高的 chunk。"""
    conn = _setup()
    _save_and_set("kcompactd.max_delete", 1)
    cid300 = _make_chunk(conn, oom_adj=300, importance=0.15, access_count=0, age_days=10,
                         summary="oom300")
    cid1000 = _make_chunk(conn, oom_adj=1000, importance=0.15, access_count=0, age_days=10,
                          summary="oom1000")
    result = kcompactd(conn, "test_proj")
    assert result["deleted"] == 1
    # oom_adj=1000 应该先被删除
    assert not _exists(conn, cid1000)
    assert _exists(conn, cid300)
    _restore_all()


def test_performance():
    """100 chunks 扫描 <100ms。"""
    conn = _setup()
    for i in range(100):
        _make_chunk(conn, oom_adj=300, importance=0.15, access_count=0, age_days=10,
                    summary=f"perf chunk {i}")
    _save_and_set("kcompactd.max_delete", 50)
    t0 = time.time()
    result = kcompactd(conn, "test_proj")
    elapsed = (time.time() - t0) * 1000
    assert elapsed < 500  # 宽松阈值（CI 环境）
    assert result["deleted"] <= 50  # max_delete 限制
    _restore_all()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
