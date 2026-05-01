#!/usr/bin/env python3
"""
test_second_chance.py — Second-Chance Diversity Sampling 测试（iter471）

覆盖：
  SC1: stability>=5 且 0.05<importance<0.20 → 候选集命中
  SC2: importance>=0.20 → 不在候选集（高 importance 不需要 second-chance）
  SC3: task_state → 不在候选集（非知识类跳过）
  SC4: stability<5 → 不在候选集（历史不重要，不值得救活）
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


def _insert_chunk(conn, cid, project, summary, importance=0.6, stability=1.0,
                  chunk_type="decision"):
    now = datetime.now(timezone.utc)
    la = (now - timedelta(days=1)).isoformat()
    now_iso = now.isoformat()
    conn.execute("""
        INSERT OR REPLACE INTO memory_chunks
        (id, project, source_session, chunk_type, summary, content,
         importance, stability, retrievability, info_class, tags,
         access_count, oom_adj, created_at, updated_at, last_accessed,
         feishu_url, lru_gen, raw_snippet, encode_context, session_type_history)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (cid, project, "test", chunk_type, summary, summary,
          importance, stability, 0.5, "episodic", json.dumps([]),
          5, 0, now_iso, now_iso, la, None, 0, summary[:500], "{}", ""))
    conn.commit()


def _query_sc(conn, project):
    """复现 retriever.py iter471 second-chance 查询逻辑。"""
    return conn.execute(
        """SELECT id FROM memory_chunks
           WHERE project=?
             AND COALESCE(stability, 0) >= 5.0
             AND importance < 0.20
             AND importance > 0.05
             AND chunk_type NOT IN ('task_state','prompt_context',
                 'conversation_summary','session_summary','goal')
           ORDER BY RANDOM() LIMIT 5""",
        (project,)
    ).fetchall()


def test_sc1_eligible_chunk_found():
    """SC1: stability>=5 且 0.05<importance<0.20 → 查询可命中。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"sc1_{uuid.uuid4().hex[:6]}"
    cid = f"sc1c_{uuid.uuid4().hex[:10]}"
    _insert_chunk(conn, cid, proj, "Old important kernel memory knowledge",
                  importance=0.12, stability=8.0, chunk_type="decision")

    ids = [r[0] for r in _query_sc(conn, proj)]
    assert cid in ids, f"SC1: 符合条件的 chunk 应在候选集中, ids={ids}"

    conn.execute("DELETE FROM memory_chunks WHERE project=?", (proj,))
    conn.commit()
    conn.close()
    print("  SC1 PASS: eligible chunk found in second-chance candidates")


def test_sc2_high_importance_excluded():
    """SC2: importance>=0.20 → 不在候选集（不需要救活）。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"sc2_{uuid.uuid4().hex[:6]}"
    cid = f"sc2c_{uuid.uuid4().hex[:10]}"
    _insert_chunk(conn, cid, proj, "Still important chunk",
                  importance=0.50, stability=8.0)

    ids = [r[0] for r in _query_sc(conn, proj)]
    assert cid not in ids, f"SC2: 高 importance 不应在候选集"

    conn.execute("DELETE FROM memory_chunks WHERE project=?", (proj,))
    conn.commit()
    conn.close()
    print("  SC2 PASS: high importance chunk excluded")


def test_sc3_skip_types_excluded():
    """SC3: task_state → 不在候选集。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"sc3_{uuid.uuid4().hex[:6]}"
    cid = f"sc3c_{uuid.uuid4().hex[:10]}"
    _insert_chunk(conn, cid, proj, "正在执行任务 A",
                  importance=0.10, stability=8.0, chunk_type="task_state")

    ids = [r[0] for r in _query_sc(conn, proj)]
    assert cid not in ids, f"SC3: task_state 不应在候选集"

    conn.execute("DELETE FROM memory_chunks WHERE project=?", (proj,))
    conn.commit()
    conn.close()
    print("  SC3 PASS: task_state excluded")


def test_sc4_low_stability_excluded():
    """SC4: stability<5 → 不在候选集（历史不重要）。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"sc4_{uuid.uuid4().hex[:6]}"
    cid = f"sc4c_{uuid.uuid4().hex[:10]}"
    _insert_chunk(conn, cid, proj, "Low stability transient chunk",
                  importance=0.10, stability=2.0)

    ids = [r[0] for r in _query_sc(conn, proj)]
    assert cid not in ids, f"SC4: 低 stability 不应在候选集"

    conn.execute("DELETE FROM memory_chunks WHERE project=?", (proj,))
    conn.commit()
    conn.close()
    print("  SC4 PASS: low stability excluded")


if __name__ == "__main__":
    print("Second-Chance Diversity Sampling 测试")
    print("=" * 60)

    tests = [
        test_sc1_eligible_chunk_found,
        test_sc2_high_importance_excluded,
        test_sc3_skip_types_excluded,
        test_sc4_low_stability_excluded,
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
