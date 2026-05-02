"""
iter509: rmap_sweep — Reverse Mapping Stale Reference Scrubber

OS 类比：Linux rmap (Rik van Riel, 2002)
  page frame 释放后清除所有 PTE 反向映射，防止 use-after-free。

测试验证：
  T1  全部有效引用 → 不修改
  T2  部分 stale → scrub 保留有效部分
  T3  全部 stale → 整条 trace 删除
  T4  混合场景（部分 trace 全 stale + 部分 trace 部分 stale + 部分 trace 干净）
  T5  空 recall_traces → 零操作
  T6  per-project 隔离：只清理指定 project
  T7  非标准格式条目保留
  T8  大批量（>500 条）正确处理分批删除
  T9  rmap 后 readahead_pairs 不产生 ghost pair
  T10 性能：300 traces 扫描 < 50ms
  T11 全局扫描（project=None）
"""
import sys
import os
import json
import time
import uuid

# tmpfs 隔离
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import tmpfs  # noqa: F401 — 必须在 store import 之前

from store_mm import gc_traces, rmap_sweep, readahead_pairs
from store_vfs import open_db, ensure_schema

# 每次测试使用唯一 project，避免数据泄漏
_test_counter = 0

def _unique_project():
    global _test_counter
    _test_counter += 1
    return f"test:rmap_{_test_counter}"


def _setup_db():
    conn = open_db()
    ensure_schema(conn)
    return conn


def _insert_chunk(conn, chunk_id=None, project="test:rmap"):
    cid = chunk_id or str(uuid.uuid4())
    conn.execute(
        "INSERT OR IGNORE INTO memory_chunks (id, project, chunk_type, summary, content, importance, created_at, last_accessed) "
        "VALUES (?, ?, 'decision', 'test summary', 'test content', 0.8, datetime('now'), datetime('now'))",
        (cid, project))
    conn.commit()
    return cid


def _insert_trace(conn, chunk_ids, project="test:rmap", injected=1):
    tid = str(uuid.uuid4())
    top_k = [{"id": cid, "summary": f"chunk {cid[:8]}", "score": 0.8} for cid in chunk_ids]
    conn.execute(
        "INSERT INTO recall_traces (id, project, session_id, prompt_hash, top_k_json, injected, timestamp) "
        "VALUES (?, ?, 'test_session', 'hash123', ?, ?, datetime('now'))",
        (tid, project, json.dumps(top_k), injected))
    conn.commit()
    return tid


def test_01_all_valid_no_change():
    """T1: 全部有效引用 → 不修改。"""
    conn = _setup_db()
    proj = _unique_project()
    c1 = _insert_chunk(conn, project=proj)
    c2 = _insert_chunk(conn, project=proj)
    _insert_trace(conn, [c1, c2], project=proj)

    result = rmap_sweep(conn, proj)
    assert result["scrubbed_traces"] == 0
    assert result["deleted_traces"] == 0
    assert result["stale_refs_removed"] == 0
    conn.close()


def test_02_partial_stale_scrubbed():
    """T2: 部分 stale → scrub 保留有效部分。"""
    conn = _setup_db()
    proj = _unique_project()
    c1 = _insert_chunk(conn, project=proj)
    ghost = "ghost-" + str(uuid.uuid4())
    _insert_trace(conn, [c1, ghost], project=proj)

    result = rmap_sweep(conn, proj)
    assert result["scrubbed_traces"] == 1
    assert result["stale_refs_removed"] == 1
    assert result["deleted_traces"] == 0

    row = conn.execute("SELECT top_k_json FROM recall_traces WHERE project=?", (proj,)).fetchone()
    data = json.loads(row[0])
    assert len(data) == 1
    assert data[0]["id"] == c1
    conn.close()


def test_03_all_stale_deleted():
    """T3: 全部 stale → 整条 trace 删除。"""
    conn = _setup_db()
    proj = _unique_project()
    ghost1 = "ghost-" + str(uuid.uuid4())
    ghost2 = "ghost-" + str(uuid.uuid4())
    _insert_trace(conn, [ghost1, ghost2], project=proj)

    result = rmap_sweep(conn, proj)
    assert result["deleted_traces"] == 1
    assert result["stale_refs_removed"] == 2

    count = conn.execute("SELECT COUNT(*) FROM recall_traces WHERE project=?", (proj,)).fetchone()[0]
    assert count == 0
    conn.close()


def test_04_mixed_scenario():
    """T4: 混合场景 — 干净 + 部分 stale + 全 stale。"""
    conn = _setup_db()
    proj = _unique_project()
    c1 = _insert_chunk(conn, project=proj)
    c2 = _insert_chunk(conn, project=proj)
    ghost = "ghost-" + str(uuid.uuid4())

    _insert_trace(conn, [c1, c2], project=proj)
    _insert_trace(conn, [c1, ghost], project=proj)
    _insert_trace(conn, [ghost, "ghost2-" + str(uuid.uuid4())], project=proj)

    result = rmap_sweep(conn, proj)
    assert result["total_scanned"] == 3
    assert result["scrubbed_traces"] == 1
    assert result["deleted_traces"] == 1
    assert result["stale_refs_removed"] == 3

    remaining = conn.execute("SELECT COUNT(*) FROM recall_traces WHERE project=?", (proj,)).fetchone()[0]
    assert remaining == 2
    conn.close()


def test_05_empty_traces():
    """T5: 空 recall_traces → 零操作。"""
    conn = _setup_db()
    proj = _unique_project()
    result = rmap_sweep(conn, proj)
    assert result["scrubbed_traces"] == 0
    assert result["deleted_traces"] == 0
    assert result["stale_refs_removed"] == 0
    assert result["total_scanned"] == 0
    conn.close()


def test_06_project_isolation():
    """T6: per-project 隔离 — 只清理指定 project。"""
    conn = _setup_db()
    proj = _unique_project()
    other = _unique_project()
    ghost = "ghost-" + str(uuid.uuid4())

    _insert_trace(conn, [ghost], project=proj)
    _insert_trace(conn, [ghost], project=other)

    result = rmap_sweep(conn, proj)
    assert result["deleted_traces"] == 1

    other_count = conn.execute(
        "SELECT COUNT(*) FROM recall_traces WHERE project=?", (other,)).fetchone()[0]
    assert other_count == 1
    conn.close()


def test_07_non_standard_items_preserved():
    """T7: top_k_json 中非标准格式条目保留。"""
    conn = _setup_db()
    proj = _unique_project()
    c1 = _insert_chunk(conn, project=proj)
    ghost = "ghost-" + str(uuid.uuid4())
    tid = str(uuid.uuid4())

    top_k = [
        {"id": c1, "summary": "valid", "score": 0.9},
        {"id": ghost, "summary": "ghost", "score": 0.5},
        "some_legacy_format",
    ]
    conn.execute(
        "INSERT INTO recall_traces (id, project, session_id, prompt_hash, top_k_json, injected, timestamp) "
        "VALUES (?, ?, 'test', 'hash', ?, 1, datetime('now'))",
        (tid, proj, json.dumps(top_k)))
    conn.commit()

    result = rmap_sweep(conn, proj)
    assert result["stale_refs_removed"] == 1

    row = conn.execute("SELECT top_k_json FROM recall_traces WHERE id=?", (tid,)).fetchone()
    data = json.loads(row[0])
    assert len(data) == 2
    assert data[0]["id"] == c1
    assert data[1] == "some_legacy_format"
    conn.close()


def test_08_large_batch_delete():
    """T8: 大批量（>500 条）正确处理分批删除。"""
    conn = _setup_db()
    proj = _unique_project()
    ghost = "ghost-" + str(uuid.uuid4())

    for _ in range(600):
        _insert_trace(conn, [ghost], project=proj)

    result = rmap_sweep(conn, proj)
    assert result["deleted_traces"] == 600
    assert result["stale_refs_removed"] == 600

    remaining = conn.execute("SELECT COUNT(*) FROM recall_traces WHERE project=?", (proj,)).fetchone()[0]
    assert remaining == 0
    conn.close()


def test_09_readahead_clean_after_rmap():
    """T9: rmap 后 readahead_pairs 不产生 ghost pair。"""
    conn = _setup_db()
    proj = _unique_project()
    c1 = _insert_chunk(conn, project=proj)
    c2 = _insert_chunk(conn, project=proj)
    ghost = "ghost-" + str(uuid.uuid4())

    for _ in range(3):
        _insert_trace(conn, [c1, ghost], project=proj)
    for _ in range(3):
        _insert_trace(conn, [c1, c2], project=proj)

    result = rmap_sweep(conn, proj)
    assert result["stale_refs_removed"] == 3

    pairs_after = readahead_pairs(conn, proj, [c1])
    if c1 in pairs_after:
        partner_ids = [p[0] for p in pairs_after[c1]]
        assert ghost not in partner_ids
    conn.close()


def test_10_performance():
    """T10: 性能 — 300 traces 扫描 < 50ms。"""
    conn = _setup_db()
    proj = _unique_project()
    c1 = _insert_chunk(conn, project=proj)
    ghost = "ghost-" + str(uuid.uuid4())

    for i in range(300):
        if i % 3 == 0:
            _insert_trace(conn, [c1, ghost], project=proj)
        elif i % 3 == 1:
            _insert_trace(conn, [ghost], project=proj)
        else:
            _insert_trace(conn, [c1], project=proj)

    t0 = time.perf_counter()
    result = rmap_sweep(conn, proj)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert result["total_scanned"] == 300
    assert elapsed_ms < 50, f"rmap_sweep took {elapsed_ms:.1f}ms (limit: 50ms)"
    conn.close()


def test_11_global_sweep():
    """T11: project=None 时全局扫描所有项目。"""
    conn = _setup_db()
    proj_a = _unique_project()
    proj_b = _unique_project()
    ghost = "ghost-" + str(uuid.uuid4())
    _insert_trace(conn, [ghost], project=proj_a)
    _insert_trace(conn, [ghost], project=proj_b)

    result = rmap_sweep(conn, project=None)
    assert result["deleted_traces"] >= 2
    assert result["stale_refs_removed"] >= 2
    conn.close()
