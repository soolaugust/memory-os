#!/usr/bin/env python3
"""
test_per_type_stale_threshold.py — per-type stale threshold 测试（iter483）

覆盖：
  PT1: decision chunk 连续 3 次 stale → 被降级（普通阈值 = 3）
  PT2: design_constraint 连续 3 次 stale → 不降级（需要 6 次）
  PT3: design_constraint 连续 6 次 stale → 被降级（双倍阈值 = 6）
  PT4: 边界值：decision 连续 2 次 → 不降级，3 次 → 降级
  PT5: 边界值：design_constraint 连续 5 次 → 不降级，6 次 → 降级

测试策略：
  - 直接写入 recall_traces 构造 stale 场景
  - 调用 _check_and_apply_fast_stale_with_ids 验证降级行为
"""
import sys
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import tmpfs  # noqa

from store import open_db, ensure_schema
from tools.citation_detector import _check_and_apply_fast_stale_with_ids, STALE_CONSEC_THRESHOLD


def _insert_chunk(conn, cid, project, chunk_type="decision", importance=0.7, stability=2.0):
    now = datetime.now(timezone.utc)
    last_accessed = (now - timedelta(days=5)).isoformat()  # 5天前，确保不触发 ebbinghaus 协调
    updated_at = (now - timedelta(days=5)).isoformat()     # 同上
    now_iso = now.isoformat()
    conn.execute("""
        INSERT OR REPLACE INTO memory_chunks
        (id, project, source_session, chunk_type, summary, content,
         importance, stability, retrievability, info_class, tags,
         access_count, oom_adj, created_at, updated_at, last_accessed,
         feishu_url, lru_gen, raw_snippet, encode_context, session_type_history)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (cid, project, "test", chunk_type, "stale test", "stale test",
          importance, stability, 0.5, "episodic", json.dumps([]),
          5, 0, now_iso, updated_at, last_accessed,
          None, 0, "stale test", "{}", ""))
    conn.commit()


def _insert_stale_traces(conn, project, cid, n_traces):
    """插入 n_traces 条 recall_trace，每条都包含 cid（模拟被检索但未引用）。"""
    now = datetime.now(timezone.utc)
    for i in range(n_traces):
        ts = (now - timedelta(minutes=n_traces - i)).isoformat()
        top_k = json.dumps([{"id": cid, "score": 0.8}])
        rid = f"rt_{uuid.uuid4().hex[:12]}"
        conn.execute("""
            INSERT INTO recall_traces
            (id, session_id, project, timestamp, prompt_hash, top_k_json, injected)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (rid, f"sess_{i}", project, ts, "hash_test", top_k, 1))
    conn.commit()


def test_pt1_decision_3_stale_degraded():
    """PT1: decision chunk 连续 3 次 stale → 被降级。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"pt1_{uuid.uuid4().hex[:8]}"
    cid = f"pt1c_{uuid.uuid4().hex[:6]}"

    assert STALE_CONSEC_THRESHOLD == 3, f"PT1 前提：STALE_CONSEC_THRESHOLD=3, got {STALE_CONSEC_THRESHOLD}"

    _insert_chunk(conn, cid, proj, chunk_type="decision", importance=0.7)
    _insert_stale_traces(conn, proj, cid, n_traces=3)

    old_imp = conn.execute("SELECT importance FROM memory_chunks WHERE id=?", (cid,)).fetchone()[0]
    degraded_ids, count = _check_and_apply_fast_stale_with_ids(conn, [cid], proj)
    conn.commit()
    new_imp = conn.execute("SELECT importance FROM memory_chunks WHERE id=?", (cid,)).fetchone()[0]

    assert cid in degraded_ids or new_imp < old_imp, (
        f"PT1: decision 连续 3 次 stale 应被降级, imp: {old_imp}→{new_imp}"
    )
    print(f"  PT1 PASS: decision 3 stale → degraded, imp: {old_imp:.3f}→{new_imp:.3f}")
    conn.close()


def test_pt2_design_constraint_3_stale_not_degraded():
    """PT2: design_constraint 连续 3 次 stale → 不降级（阈值=6）。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"pt2_{uuid.uuid4().hex[:8]}"
    cid = f"pt2c_{uuid.uuid4().hex[:6]}"

    _insert_chunk(conn, cid, proj, chunk_type="design_constraint", importance=0.7)
    _insert_stale_traces(conn, proj, cid, n_traces=3)

    old_imp = conn.execute("SELECT importance FROM memory_chunks WHERE id=?", (cid,)).fetchone()[0]
    degraded_ids, count = _check_and_apply_fast_stale_with_ids(conn, [cid], proj)
    conn.commit()
    new_imp = conn.execute("SELECT importance FROM memory_chunks WHERE id=?", (cid,)).fetchone()[0]

    assert abs(new_imp - old_imp) < 0.01, (
        f"PT2: design_constraint 连续 3 次 stale 不应降级（阈值=6），"
        f"imp: {old_imp:.3f}→{new_imp:.3f}"
    )
    print(f"  PT2 PASS: design_constraint 3 stale → NOT degraded (threshold=6), imp unchanged: {new_imp:.3f}")
    conn.close()


def test_pt3_design_constraint_6_stale_degraded():
    """PT3: design_constraint 连续 6 次 stale → 被降级（双倍阈值）。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"pt3_{uuid.uuid4().hex[:8]}"
    cid = f"pt3c_{uuid.uuid4().hex[:6]}"

    _insert_chunk(conn, cid, proj, chunk_type="design_constraint", importance=0.7)
    _insert_stale_traces(conn, proj, cid, n_traces=6)

    old_imp = conn.execute("SELECT importance FROM memory_chunks WHERE id=?", (cid,)).fetchone()[0]
    degraded_ids, count = _check_and_apply_fast_stale_with_ids(conn, [cid], proj)
    conn.commit()
    new_imp = conn.execute("SELECT importance FROM memory_chunks WHERE id=?", (cid,)).fetchone()[0]

    assert cid in degraded_ids or new_imp < old_imp, (
        f"PT3: design_constraint 连续 6 次 stale 应被降级, imp: {old_imp}→{new_imp}"
    )
    print(f"  PT3 PASS: design_constraint 6 stale → degraded, imp: {old_imp:.3f}→{new_imp:.3f}")
    conn.close()


def test_pt4_decision_boundary_2_vs_3():
    """PT4: decision 连续 2 次 → 不降级，3 次 → 降级。"""
    conn = open_db()
    ensure_schema(conn)

    # 2次 - 不降级
    proj2 = f"pt4a_{uuid.uuid4().hex[:6]}"
    cid2 = f"pt4a_{uuid.uuid4().hex[:6]}"
    _insert_chunk(conn, cid2, proj2, chunk_type="decision", importance=0.7)
    _insert_stale_traces(conn, proj2, cid2, n_traces=2)
    old2 = float(conn.execute("SELECT importance FROM memory_chunks WHERE id=?", (cid2,)).fetchone()[0])
    _check_and_apply_fast_stale_with_ids(conn, [cid2], proj2)
    conn.commit()
    new2 = float(conn.execute("SELECT importance FROM memory_chunks WHERE id=?", (cid2,)).fetchone()[0])
    assert abs(new2 - old2) < 0.01, f"PT4: decision 2次不应降级, {old2:.3f}→{new2:.3f}"

    # 3次 - 降级
    proj3 = f"pt4b_{uuid.uuid4().hex[:6]}"
    cid3 = f"pt4b_{uuid.uuid4().hex[:6]}"
    _insert_chunk(conn, cid3, proj3, chunk_type="decision", importance=0.7)
    _insert_stale_traces(conn, proj3, cid3, n_traces=3)
    old3 = float(conn.execute("SELECT importance FROM memory_chunks WHERE id=?", (cid3,)).fetchone()[0])
    degraded_ids, _ = _check_and_apply_fast_stale_with_ids(conn, [cid3], proj3)
    conn.commit()
    new3 = float(conn.execute("SELECT importance FROM memory_chunks WHERE id=?", (cid3,)).fetchone()[0])
    assert cid3 in degraded_ids or new3 < old3, f"PT4: decision 3次应降级, {old3:.3f}→{new3:.3f}"

    conn.close()
    print(f"  PT4 PASS: decision 2次未降级({new2:.3f}), 3次降级({old3:.3f}→{new3:.3f})")


def test_pt5_design_constraint_boundary_5_vs_6():
    """PT5: design_constraint 连续 5 次 → 不降级，6 次 → 降级。"""
    conn = open_db()
    ensure_schema(conn)

    # 5次 - 不降级
    proj5 = f"pt5a_{uuid.uuid4().hex[:6]}"
    cid5 = f"pt5a_{uuid.uuid4().hex[:6]}"
    _insert_chunk(conn, cid5, proj5, chunk_type="design_constraint", importance=0.7)
    _insert_stale_traces(conn, proj5, cid5, n_traces=5)
    old5 = float(conn.execute("SELECT importance FROM memory_chunks WHERE id=?", (cid5,)).fetchone()[0])
    _check_and_apply_fast_stale_with_ids(conn, [cid5], proj5)
    conn.commit()
    new5 = float(conn.execute("SELECT importance FROM memory_chunks WHERE id=?", (cid5,)).fetchone()[0])
    assert abs(new5 - old5) < 0.01, f"PT5: design_constraint 5次不应降级, {old5:.3f}→{new5:.3f}"

    # 6次 - 降级
    proj6 = f"pt5b_{uuid.uuid4().hex[:6]}"
    cid6 = f"pt5b_{uuid.uuid4().hex[:6]}"
    _insert_chunk(conn, cid6, proj6, chunk_type="design_constraint", importance=0.7)
    _insert_stale_traces(conn, proj6, cid6, n_traces=6)
    old6 = float(conn.execute("SELECT importance FROM memory_chunks WHERE id=?", (cid6,)).fetchone()[0])
    degraded_ids, _ = _check_and_apply_fast_stale_with_ids(conn, [cid6], proj6)
    conn.commit()
    new6 = float(conn.execute("SELECT importance FROM memory_chunks WHERE id=?", (cid6,)).fetchone()[0])
    assert cid6 in degraded_ids or new6 < old6, f"PT5: design_constraint 6次应降级, {old6:.3f}→{new6:.3f}"

    conn.close()
    print(f"  PT5 PASS: design_constraint 5次未降级({new5:.3f}), 6次降级({old6:.3f}→{new6:.3f})")


if __name__ == "__main__":
    print("per-type stale threshold 测试（iter483）")
    print("=" * 60)

    tests = [
        test_pt1_decision_3_stale_degraded,
        test_pt2_design_constraint_3_stale_not_degraded,
        test_pt3_design_constraint_6_stale_degraded,
        test_pt4_decision_boundary_2_vs_3,
        test_pt5_design_constraint_boundary_5_vs_6,
    ]

    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  {t.__name__} FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n结果：{passed}/{passed+failed} 通过")
    if failed:
        sys.exit(1)
