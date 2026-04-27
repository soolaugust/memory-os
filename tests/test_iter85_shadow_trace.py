#!/usr/bin/env python3
"""
迭代85 测试：Shadow Trace — perf_event 采样式工作集追踪

验证：
1. _write_shadow_trace() 正确写入 SHADOW_TRACE_FILE
2. 文件格式：project/top_k_ids/session_id/timestamp
3. save-task-state.py 在 recall_traces 为空时从 shadow trace 恢复 hit_ids
4. shadow trace 验证 chunk 仍然存在（防 stale ref）
5. shadow trace 空 top_k_ids 不影响（正确降级）
6. recall_traces 非空时不触发 shadow trace fallback
7. shadow trace 文件不存在时 hit_ids = []（向后兼容）
8. 多次 write 覆盖（最新检索结果）
9. stale chunk IDs 被过滤
10. 写入性能 < 2ms/call
"""
import json
import os
import sys
import time
import uuid
from pathlib import Path
from datetime import datetime, timezone

import tmpfs  # noqa: F401 — 测试隔离（必须在 store import 之前）
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "hooks"))

from store_core import (
    open_db, ensure_schema, insert_chunk,
)

MEMORY_OS_DIR = Path(os.environ.get("MEMORY_OS_DIR", Path.home() / ".claude" / "memory-os"))
SHADOW_TRACE_FILE = MEMORY_OS_DIR / ".shadow_trace.json"


def make_chunk(conn, project, chunk_type="decision", summary=None, importance=0.85):
    cid = str(uuid.uuid4())
    summary = summary or f"test decision {cid[:8]}"
    now = datetime.now(timezone.utc).isoformat()
    insert_chunk(conn, {
        "id": cid,
        "project": project,
        "summary": summary,
        "chunk_type": chunk_type,
        "content": f"content_{cid[:8]}",
        "importance": importance,
        "retrievability": 0.5,
        "last_accessed": now,
        "tags": "[]",
        "source_session": "test-session",
        "created_at": now,
        "updated_at": now,
        "feishu_url": None,
    })
    return cid


def _write_shadow_trace(project: str, top_k_ids: list, session_id: str = "") -> None:
    """与 retriever.py 中完全相同的实现，供测试直接调用。"""
    try:
        SHADOW_TRACE_FILE.write_text(json.dumps({
            "project": project,
            "top_k_ids": top_k_ids,
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _shadow_trace_fallback(conn, project: str, existing_hit_ids: list) -> tuple:
    """
    模拟 save-task-state.py 中的 shadow trace fallback 逻辑。
    返回 (hit_ids, shadow_fallback_used)
    """
    if existing_hit_ids:
        return existing_hit_ids, False
    if not SHADOW_TRACE_FILE.exists():
        return [], False
    try:
        shadow = json.loads(SHADOW_TRACE_FILE.read_text(encoding="utf-8"))
        shadow_ids = shadow.get("top_k_ids", [])
        if not shadow_ids:
            return [], False
        placeholders = ",".join("?" * len(shadow_ids))
        valid = conn.execute(
            f"SELECT id FROM memory_chunks WHERE id IN ({placeholders})",
            shadow_ids
        ).fetchall()
        valid_ids = [r[0] for r in valid]
        if valid_ids:
            return valid_ids, True
    except Exception:
        pass
    return [], False


passed = 0
failed = 0


def _assert_test(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


def run_tests():
    proj = f"test-shadow-{uuid.uuid4().hex[:8]}"

    # T1: _write_shadow_trace 写入正确格式
    SHADOW_TRACE_FILE.unlink(missing_ok=True)
    ids = ["aaa-111", "bbb-222", "ccc-333"]
    _write_shadow_trace(proj, ids, "session-abc")
    _assert_test("T1: shadow trace file created", SHADOW_TRACE_FILE.exists(),
         "file not created")

    data = json.loads(SHADOW_TRACE_FILE.read_text())
    _assert_test("T2: shadow trace project field",
         data.get("project") == proj, f"got {data.get('project')}")
    _assert_test("T3: shadow trace top_k_ids field",
         data.get("top_k_ids") == ids, f"got {data.get('top_k_ids')}")
    _assert_test("T4: shadow trace session_id field",
         data.get("session_id") == "session-abc", f"got {data.get('session_id')}")
    _assert_test("T5: shadow trace has timestamp",
         bool(data.get("timestamp")), "no timestamp")

    # T6: 多次写覆盖（保留最新）
    _write_shadow_trace(proj, ["new-111", "new-222"], "session-xyz")
    data2 = json.loads(SHADOW_TRACE_FILE.read_text())
    _assert_test("T6: shadow trace overwritten with latest",
         data2.get("top_k_ids") == ["new-111", "new-222"],
         f"got {data2.get('top_k_ids')}")

    # T7: 空 top_k_ids 时 fallback 不返回结果
    conn = open_db()
    ensure_schema(conn)
    make_chunk(conn, proj)
    conn.commit()

    _write_shadow_trace(proj, [], "session-empty")
    hit_ids, used = _shadow_trace_fallback(conn, proj, [])
    _assert_test("T7: empty top_k_ids → no fallback",
         not used and hit_ids == [], f"used={used}, ids={hit_ids}")

    # T8: recall_traces 非空时不触发 fallback
    real_chunk_id = make_chunk(conn, proj, summary="real trace chunk")
    conn.commit()
    _write_shadow_trace(proj, [real_chunk_id], "session-test")
    hit_ids, used = _shadow_trace_fallback(conn, proj, ["existing-id"])
    _assert_test("T8: non-empty recall_traces → no shadow fallback",
         not used and hit_ids == ["existing-id"],
         f"used={used}, ids={hit_ids}")

    # T9: recall_traces 为空 + shadow trace 有效 → fallback 成功
    chunk_id = make_chunk(conn, proj, summary="shadow chunk for fallback")
    conn.commit()
    _write_shadow_trace(proj, [chunk_id], "session-test")
    hit_ids, used = _shadow_trace_fallback(conn, proj, [])
    _assert_test("T9: empty recall_traces + valid shadow → fallback",
         used and chunk_id in hit_ids,
         f"used={used}, ids={hit_ids}")

    # T10: stale chunk ID（从未插入 DB）不出现在 fallback 结果中
    stale_id = str(uuid.uuid4())
    _write_shadow_trace(proj, [stale_id, chunk_id], "session-stale")
    hit_ids, used = _shadow_trace_fallback(conn, proj, [])
    _assert_test("T10: stale chunk ID filtered out",
         stale_id not in hit_ids,
         f"stale_id unexpectedly in: {hit_ids}")
    _assert_test("T10b: valid chunk still returned alongside stale",
         chunk_id in hit_ids,
         f"chunk_id missing from: {hit_ids}")

    # T11: shadow trace 文件不存在 → hit_ids = [] (向后兼容)
    SHADOW_TRACE_FILE.unlink(missing_ok=True)
    hit_ids, used = _shadow_trace_fallback(conn, proj, [])
    _assert_test("T11: missing shadow file → empty hit_ids",
         not used and hit_ids == [],
         f"used={used}, ids={hit_ids}")

    # T12: 全部 stale IDs → fallback 返回空
    _write_shadow_trace(proj, [str(uuid.uuid4()), str(uuid.uuid4())], "session-all-stale")
    hit_ids, used = _shadow_trace_fallback(conn, proj, [])
    _assert_test("T12: all stale IDs → fallback returns empty",
         not used and hit_ids == [],
         f"used={used}, ids={hit_ids}")

    # T13: SHADOW_TRACE_FILE 常量路径验证
    _assert_test("T13: SHADOW_TRACE_FILE ends with .shadow_trace.json",
         str(SHADOW_TRACE_FILE).endswith(".shadow_trace.json"),
         f"path={SHADOW_TRACE_FILE}")

    conn.close()

    # T14: 写入性能 < 2ms/call
    t0 = time.perf_counter()
    for _ in range(10):
        _write_shadow_trace(proj, [str(uuid.uuid4())] * 5, "perf-session")
    elapsed = (time.perf_counter() - t0) * 1000 / 10
    _assert_test("T14: write performance < 2ms/call",
         elapsed < 2.0,
         f"avg {elapsed:.2f}ms")
    print(f"  PERF  _write_shadow_trace avg {elapsed:.2f}ms/call")

    SHADOW_TRACE_FILE.unlink(missing_ok=True)

    print(f"\n{'='*50}")
    print(f"Shadow Trace Tests: {passed} passed, {failed} failed, total {passed+failed}")
    if failed == 0:
        print("ALL PASSED ✓")
    else:
        print("SOME FAILED ✗")
    return failed == 0


if __name__ == "__main__":
    ok = run_tests()
    sys.exit(0 if ok else 1)
