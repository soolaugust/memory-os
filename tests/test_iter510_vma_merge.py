"""
iter510: vma_merge — Recall Trace Deduplication 测试

OS 类比：Linux vma_merge() — 相邻 VMA 属性相同时自动合并，减少碎片。
"""
import sys
import os
import json
import uuid
import time
from pathlib import Path

# tmpfs 隔离（必须在 store import 之前）
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
import tmpfs  # noqa: F401,E402

from store_mm import vma_merge
from store_core import open_db, ensure_schema, dmesg_log, DMESG_INFO


def _setup_db():
    conn = open_db()
    ensure_schema(conn)
    return conn


def _insert_trace(conn, project, chunk_ids, ts=None, injected=1):
    """插入一条 recall_trace"""
    top_k = [{"id": cid, "summary": f"chunk_{cid[:8]}", "score": 0.8}
             for cid in chunk_ids]
    if ts is None:
        ts = f"2026-05-02T{10 + len(chunk_ids):02d}:00:00+00:00"
    conn.execute(
        "INSERT INTO recall_traces (id, project, session_id, prompt_hash, "
        "top_k_json, injected, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (str(uuid.uuid4()), project, "test-session", "hash123",
         json.dumps(top_k), injected, ts)
    )
    conn.commit()


def _count_traces(conn, project=None):
    if project:
        return conn.execute(
            "SELECT COUNT(*) FROM recall_traces WHERE project=?",
            (project,)).fetchone()[0]
    return conn.execute("SELECT COUNT(*) FROM recall_traces").fetchone()[0]


# ── T1: 完全重复的 traces 被合并 ──
def test_exact_merge_removes_duplicates():
    conn = _setup_db()
    proj = "test-exact-merge"
    ids = [str(uuid.uuid4()) for _ in range(3)]

    # 插入 5 条完全相同的 traces（不同时间戳）
    for i in range(5):
        _insert_trace(conn, proj, ids, ts=f"2026-05-02T{10+i:02d}:00:00+00:00")

    assert _count_traces(conn, proj) == 5

    result = vma_merge(conn, proj)
    assert result["exact_merged"] == 4  # 保留 1，删除 4
    assert result["fuzzy_merged"] == 0
    assert _count_traces(conn, proj) == 1
    conn.close()


# ── T2: 不同的 traces 不会被合并 ──
def test_distinct_traces_preserved():
    conn = _setup_db()
    proj = "test-distinct"

    for i in range(4):
        unique_ids = [str(uuid.uuid4()) for _ in range(3)]
        _insert_trace(conn, proj, unique_ids, ts=f"2026-05-02T{10+i:02d}:00:00+00:00")

    assert _count_traces(conn, proj) == 4

    result = vma_merge(conn, proj)
    assert result["exact_merged"] == 0
    assert result["fuzzy_merged"] == 0
    assert result["remaining"] == 4
    conn.close()


# ── T3: 模糊合并（Jaccard >= 0.8）──
def test_fuzzy_merge_high_jaccard():
    conn = _setup_db()
    proj = "test-fuzzy"

    base_ids = [str(uuid.uuid4()) for _ in range(5)]

    # trace 1: 全部 5 个 IDs
    _insert_trace(conn, proj, base_ids, ts="2026-05-02T10:00:00+00:00")
    # trace 2: 4/5 相同 + 1 个不同 → Jaccard = 4/6 = 0.67 < 0.8 → 不合并
    ids_67 = base_ids[:4] + [str(uuid.uuid4())]
    _insert_trace(conn, proj, ids_67, ts="2026-05-02T09:00:00+00:00")

    result = vma_merge(conn, proj)
    assert result["fuzzy_merged"] == 0  # Jaccard 0.67 < 0.8
    assert _count_traces(conn, proj) == 2

    # 现在测试高 Jaccard：5/5 相同 + 1 个额外 → Jaccard = 5/6 = 0.83 >= 0.8
    conn2 = _setup_db()
    proj2 = "test-fuzzy-high"
    _insert_trace(conn2, proj2, base_ids, ts="2026-05-02T10:00:00+00:00")
    ids_83 = base_ids + [str(uuid.uuid4())]  # superset
    _insert_trace(conn2, proj2, ids_83, ts="2026-05-02T09:00:00+00:00")

    result2 = vma_merge(conn2, proj2)
    assert result2["fuzzy_merged"] == 1
    assert _count_traces(conn2, proj2) == 1
    conn.close()
    conn2.close()


# ── T4: 空 DB / 单条 trace 不崩溃 ──
def test_empty_and_single_trace():
    conn = _setup_db()
    proj = "test-empty"

    # 空 DB
    result = vma_merge(conn, proj)
    assert result["exact_merged"] == 0
    assert result["fuzzy_merged"] == 0
    assert result["remaining"] == 0

    # 单条 trace
    _insert_trace(conn, proj, [str(uuid.uuid4())])
    result = vma_merge(conn, proj)
    assert result["exact_merged"] == 0
    assert result["remaining"] == 1
    conn.close()


# ── T5: max_merge_per_scan 批量限制 ──
def test_max_merge_limit():
    conn = _setup_db()
    proj = "test-limit"
    ids = [str(uuid.uuid4()) for _ in range(3)]

    # 插入 200 条完全相同的 traces
    for i in range(200):
        _insert_trace(conn, proj, ids, ts=f"2026-05-02T{i % 24:02d}:{i % 60:02d}:00+00:00")

    assert _count_traces(conn, proj) == 200

    # 默认 max_merge_per_scan=100
    result = vma_merge(conn, proj)
    assert result["exact_merged"] == 100  # 被限制为 100
    assert _count_traces(conn, proj) == 100  # 200 - 100 = 100

    # 再跑一次清理剩余
    result2 = vma_merge(conn, proj)
    assert result2["exact_merged"] == 99  # 100 - 1(keeper) = 99
    assert _count_traces(conn, proj) == 1
    conn.close()


# ── T6: project 隔离 ──
def test_project_isolation():
    conn = _setup_db()
    proj_a = "test-proj-a"
    proj_b = "test-proj-b"
    ids = [str(uuid.uuid4()) for _ in range(3)]

    for i in range(5):
        _insert_trace(conn, proj_a, ids, ts=f"2026-05-02T{10+i:02d}:00:00+00:00")
    for i in range(3):
        _insert_trace(conn, proj_b, ids, ts=f"2026-05-02T{10+i:02d}:00:00+00:00")

    # 只清理 proj_a
    result = vma_merge(conn, proj_a)
    assert result["exact_merged"] == 4
    assert _count_traces(conn, proj_a) == 1
    assert _count_traces(conn, proj_b) == 3  # 不受影响
    conn.close()


# ── T7: 无 project 参数时清理全局 ──
def test_global_merge():
    conn = _setup_db()
    proj_g1 = f"test-global-{uuid.uuid4().hex[:8]}"
    proj_g2 = f"test-global-{uuid.uuid4().hex[:8]}"
    ids = [str(uuid.uuid4()) for _ in range(3)]

    before = _count_traces(conn)  # 其他测试可能有残留

    for proj in [proj_g1, proj_g2]:
        for i in range(4):
            _insert_trace(conn, proj, ids, ts=f"2026-05-02T{10+i:02d}:00:00+00:00")

    assert _count_traces(conn) == before + 8

    result = vma_merge(conn, project=None)
    # 所有同 ID set 的 traces 被合并（跨 project 也合并）
    assert result["exact_merged"] >= 6
    # 验证这 8 条中只剩 1 条
    remaining_g = (
        _count_traces(conn, proj_g1) + _count_traces(conn, proj_g2)
    )
    assert remaining_g <= 2  # 最多各保留 1 条（实际可能只保留全局 1 条）
    conn.close()


# ── T8: injected=0 的 trace 也被合并 ──
def test_non_injected_traces_included():
    conn = _setup_db()
    proj = "test-non-injected"
    ids = [str(uuid.uuid4()) for _ in range(3)]

    _insert_trace(conn, proj, ids, ts="2026-05-02T10:00:00+00:00", injected=1)
    _insert_trace(conn, proj, ids, ts="2026-05-02T09:00:00+00:00", injected=0)

    result = vma_merge(conn, proj)
    assert result["exact_merged"] == 1
    assert _count_traces(conn, proj) == 1
    conn.close()


# ── T9: 保留最新的 trace ──
def test_keeps_newest_trace():
    conn = _setup_db()
    proj = "test-keep-newest"
    ids = [str(uuid.uuid4()) for _ in range(3)]

    _insert_trace(conn, proj, ids, ts="2026-05-02T08:00:00+00:00")
    _insert_trace(conn, proj, ids, ts="2026-05-02T12:00:00+00:00")  # newest
    _insert_trace(conn, proj, ids, ts="2026-05-02T10:00:00+00:00")

    result = vma_merge(conn, proj)
    assert result["exact_merged"] == 2

    # 验证保留的是最新的
    row = conn.execute(
        "SELECT timestamp FROM recall_traces WHERE project=?",
        (proj,)
    ).fetchone()
    assert "12:00:00" in row[0]
    conn.close()


# ── T10: 性能测试 ──
def test_performance():
    conn = _setup_db()
    proj = "test-perf"

    # 插入 300 条（100 个不同 ID set × 3 重复）
    for group in range(100):
        ids = [str(uuid.uuid4()) for _ in range(3)]
        for rep in range(3):
            _insert_trace(conn, proj, ids,
                          ts=f"2026-05-02T{group % 24:02d}:{rep*20:02d}:00+00:00")

    assert _count_traces(conn, proj) == 300

    t0 = time.time()
    result = vma_merge(conn, proj)
    elapsed = (time.time() - t0) * 1000

    assert result["exact_merged"] >= 100  # 至少合并了重复的
    assert elapsed < 200  # < 200ms for 300 traces
    print(f"  vma_merge perf: {elapsed:.1f}ms for 300 traces, merged={result['exact_merged']}+{result['fuzzy_merged']}")
    conn.close()


# ── T11: top_k_json 为空的 trace 不崩溃 ──
def test_null_top_k_json():
    conn = _setup_db()
    proj = "test-null"

    # 手动插入一条 NULL top_k_json
    conn.execute(
        "INSERT INTO recall_traces (id, project, session_id, prompt_hash, "
        "top_k_json, injected, timestamp) VALUES (?, ?, ?, ?, NULL, 1, ?)",
        (str(uuid.uuid4()), proj, "s1", "h1", "2026-05-02T10:00:00+00:00")
    )
    # 正常 trace
    ids = [str(uuid.uuid4()) for _ in range(3)]
    _insert_trace(conn, proj, ids)
    conn.commit()

    result = vma_merge(conn, proj)
    assert result["total_scanned"] >= 1  # NULL 的不被 WHERE 选中
    conn.close()


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for t in tests:
        name = t.__name__
        try:
            t()
            print(f"  PASS {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL {name}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{'='*40}")
    print(f"iter510 vma_merge: {passed}/{passed+failed} passed")
    if failed:
        sys.exit(1)
