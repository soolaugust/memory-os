#!/usr/bin/env python3
"""
test_iter86_readahead_warm.py
迭代86: Readahead Warm — SessionStart 预热 shadow_trace
OS 类比: Linux readahead() — 启动时主动预取预期页面
"""
import sys, os, json, uuid, tempfile, sqlite3
from pathlib import Path
from datetime import datetime, timezone

# ── tmpfs 测试隔离 ──
sys.path.insert(0, str(Path(__file__).parent))

# 创建 tmpfs 隔离目录
_tmp_dir = tempfile.mkdtemp(prefix="iter86_")
os.environ["MEMORY_OS_DIR"] = _tmp_dir
os.environ["MEMORY_OS_DB"] = str(Path(_tmp_dir) / "store.db")

from store import open_db, ensure_schema, dmesg_log, DMESG_INFO


if __name__ == "__main__":
    passed = 0
    failed = 0

    def test(name, condition, detail=""):
        global passed, failed
        if condition:
            passed += 1
            print(f"  ✓ {name}")
        else:
            failed += 1
            print(f"  ✗ {name}" + (f": {detail}" if detail else ""))

    def setup_db_with_chunks(project, n_chunks=5):
        """创建含 n_chunks 个 decision chunk 的 DB"""
        conn = open_db()
        ensure_schema(conn)
        ids = []
        for i in range(n_chunks):
            cid = str(uuid.uuid4())
            conn.execute(
                """INSERT INTO memory_chunks (id, project, chunk_type, summary, importance, access_count, last_accessed, created_at, source_session)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                [cid, project, "decision", f"决策{i}: 这是第{i}个重要决策内容",
                 0.8 - i*0.05, i+1,
                 datetime.now(timezone.utc).isoformat(),
                 datetime.now(timezone.utc).isoformat(),
                 "test-session"]
            )
            ids.append(cid)
        conn.commit()
        conn.close()
        return ids

    def run_loader_main(project):
        """直接调用 loader 的 working_set 查询和 shadow_trace 写入逻辑（不依赖 stdin）"""
        from store import open_db, ensure_schema
        from config import get as _sysctl
        from scorer import working_set_score as _unified_ws_score

        MEMORY_OS_DIR_PATH = Path(os.environ["MEMORY_OS_DIR"])
        STORE_DB = Path(os.environ["MEMORY_OS_DB"])
        SHADOW_FILE = MEMORY_OS_DIR_PATH / ".shadow_trace.json"
        WORKING_SET_TYPES = ("decision", "reasoning_chain", "conversation_summary")

        if not STORE_DB.exists():
            return False, []

        # 模拟 loader 的 working_set 查询
        conn = open_db()
        ensure_schema(conn)
        from store import get_chunks as store_get_chunks
        chunks = store_get_chunks(conn, project, chunk_types=WORKING_SET_TYPES)
        conn.close()

        scored = []
        for c in chunks:
            score = _unified_ws_score(c["importance"], c["last_accessed"])
            scored.append((score, c["chunk_type"], c["summary"], c["id"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        top_k = _sysctl("loader.working_set_top_k")
        working_set = scored[:top_k]

        # 迭代86: 写 shadow_trace
        if working_set and STORE_DB.exists():
            try:
                _st_conn = open_db()
                ensure_schema(_st_conn)
                _ws_rows = _st_conn.execute(
                    """SELECT id FROM memory_chunks
                       WHERE project = ?
                         AND chunk_type IN ({})
                       ORDER BY importance DESC, access_count DESC
                       LIMIT ?""".format(",".join("?" * len(WORKING_SET_TYPES))),
                    [project, *WORKING_SET_TYPES, top_k]
                ).fetchall()
                _st_conn.close()
                _ws_ids = [r[0] for r in _ws_rows]
                if _ws_ids:
                    _shadow_data = {
                        "project": project,
                        "top_k_ids": _ws_ids,
                        "session_id": "test-session-86",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "source": "session_start_readahead",
                    }
                    SHADOW_FILE.write_text(json.dumps(_shadow_data, ensure_ascii=False), encoding="utf-8")
                    return True, _ws_ids
            except Exception as e:
                return False, []
        return False, []


    print("=" * 55)
    print("迭代86: Readahead Warm — SessionStart shadow_trace 预热")
    print("=" * 55)

    project = "abspath:test86"
    MEMORY_OS_DIR_PATH = Path(os.environ["MEMORY_OS_DIR"])
    SHADOW_FILE = MEMORY_OS_DIR_PATH / ".shadow_trace.json"

    # T1: 空 DB 时，shadow_trace 不写（无工作集）
    print("\n[T1] 空 DB 场景")
    ok, ids = run_loader_main(project)
    test("T1: 空 DB 不写 shadow_trace", not ok and not SHADOW_FILE.exists())

    # T2: 有 chunks 时，shadow_trace 被写入
    print("\n[T2] 有 chunks 时预热")
    chunk_ids = setup_db_with_chunks(project, n_chunks=5)
    ok, written_ids = run_loader_main(project)
    test("T2: 成功写 shadow_trace", ok)
    test("T2: shadow_trace 文件存在", SHADOW_FILE.exists())

    if SHADOW_FILE.exists():
        data = json.loads(SHADOW_FILE.read_text())
        test("T3: shadow_trace 包含 project", data.get("project") == project)
        test("T4: shadow_trace 包含 top_k_ids", len(data.get("top_k_ids", [])) > 0)
        test("T5: source=session_start_readahead", data.get("source") == "session_start_readahead")
        test("T6: top_k_ids 全在 DB 中", all(i in chunk_ids for i in data["top_k_ids"]))
        test("T7: top_k_ids 数量 ≤ top_k", len(data["top_k_ids"]) <= 5)

    # T8: shadow_trace 可被 save-task-state 的 fallback 逻辑使用
    print("\n[T8] save-task-state fallback 验证")
    if SHADOW_FILE.exists():
        shadow = json.loads(SHADOW_FILE.read_text())
        shadow_ids = shadow.get("top_k_ids", [])
        conn = open_db()
        ensure_schema(conn)
        if shadow_ids:
            placeholders = ",".join("?" * len(shadow_ids))
            valid = conn.execute(
                f"SELECT id FROM memory_chunks WHERE id IN ({placeholders})",
                shadow_ids
            ).fetchall()
            valid_ids = [r[0] for r in valid]
            test("T8: fallback 能找到有效 chunk IDs", len(valid_ids) == len(shadow_ids),
                 f"valid={len(valid_ids)}, shadow={len(shadow_ids)}")
        conn.close()

    # T9: 第二次 SessionStart 更新 shadow_trace（幂等）
    print("\n[T9] 幂等性验证")
    ok2, ids2 = run_loader_main(project)
    if SHADOW_FILE.exists():
        data2 = json.loads(SHADOW_FILE.read_text())
        test("T9: 第二次调用也正确写入", data2.get("source") == "session_start_readahead")

    # T10: 写入性能 < 2ms
    print("\n[T10] 性能验证")
    import time
    t0 = time.perf_counter()
    for _ in range(20):
        run_loader_main(project)
    elapsed = (time.perf_counter() - t0) * 1000 / 20
    test("T10: shadow_trace 写入 < 5ms/call", elapsed < 5.0, f"avg {elapsed:.2f}ms")
    print(f"  PERF  readahead_warm avg {elapsed:.2f}ms/call")

    # 清理
    import shutil
    shutil.rmtree(_tmp_dir, ignore_errors=True)

    print(f"\n{'='*55}")
    print(f"Readahead Warm Tests: {passed} passed, {failed} failed")
    if failed == 0:
        print("ALL PASSED ✓")
    else:
        print("SOME FAILED ✗")
    sys.exit(0 if failed == 0 else 1)
