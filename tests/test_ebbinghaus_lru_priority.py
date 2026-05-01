#!/usr/bin/env python3
"""
test_ebbinghaus_lru_priority.py — Ebbinghaus 冷 chunk 优先衰减测试（iter486）

覆盖：
  LP1: lru_gen 高（冷）的 chunk 在 max_chunks 限制下先被衰减
  LP2: lru_gen=0（热）的 chunk 在 max_chunks 不足时被跳过
  LP3: 无 max_chunks 限制时，冷热 chunk 都被衰减（排序不影响结果）
  LP4: 相同 lru_gen 时，last_accessed 更早的先被衰减（次级排序）

测试策略：
  - 插入不同 lru_gen 的 chunk
  - 设置 max_chunks=1，验证只有冷 chunk（高 lru_gen）被衰减
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
from store_mm import apply_ebbinghaus_decay


def _insert_chunk(conn, cid, project, importance=0.7, stability=2.0,
                  days_ago=10.0, lru_gen=0):
    now = datetime.now(timezone.utc)
    last_accessed = (now - timedelta(days=days_ago)).isoformat()
    now_iso = now.isoformat()
    conn.execute("""
        INSERT OR REPLACE INTO memory_chunks
        (id, project, source_session, chunk_type, summary, content,
         importance, stability, retrievability, info_class, tags,
         access_count, oom_adj, created_at, updated_at, last_accessed,
         feishu_url, lru_gen, raw_snippet, encode_context, session_type_history,
         confidence_score, verification_status)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (cid, project, "test", "decision", "lru test", "lru test",
          importance, stability, 0.5, "episodic", json.dumps([]),
          5, 0, now_iso, now_iso, last_accessed,
          None, lru_gen, "lru test", "{}", "",
          0.70, "pending"))
    conn.commit()


def _get_importance(conn, cid):
    row = conn.execute("SELECT importance FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else None


def test_lp1_cold_chunk_decayed_first():
    """LP1: max_chunks=1 时，lru_gen=5（冷）被衰减，lru_gen=0（热）被跳过。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"lp1_{uuid.uuid4().hex[:8]}"
    cid_cold = f"lp1cold_{uuid.uuid4().hex[:6]}"
    cid_hot = f"lp1hot_{uuid.uuid4().hex[:6]}"

    _insert_chunk(conn, cid_cold, proj, importance=0.7, lru_gen=5, days_ago=10.0)
    _insert_chunk(conn, cid_hot, proj, importance=0.7, lru_gen=0, days_ago=10.0)

    imp_cold_before = _get_importance(conn, cid_cold)
    imp_hot_before = _get_importance(conn, cid_hot)

    # max_chunks=1 → 只处理第一个（按 lru_gen DESC → 冷 chunk 先）
    apply_ebbinghaus_decay(conn, proj, max_chunks=1)
    conn.commit()

    imp_cold_after = _get_importance(conn, cid_cold)
    imp_hot_after = _get_importance(conn, cid_hot)

    assert imp_cold_after < imp_cold_before, (
        f"LP1: 冷 chunk 应被衰减，{imp_cold_before:.3f}→{imp_cold_after:.3f}"
    )
    assert abs(imp_hot_after - imp_hot_before) < 0.01, (
        f"LP1: 热 chunk 应被跳过（max_chunks=1），{imp_hot_before:.3f}→{imp_hot_after:.3f}"
    )
    conn.close()
    print(f"  LP1 PASS: cold(lru=5) decayed {imp_cold_before:.3f}→{imp_cold_after:.3f}, "
          f"hot(lru=0) skipped {imp_hot_after:.3f}")


def test_lp2_hot_chunk_skipped_when_budget_exhausted():
    """LP2: 3个 chunk（lru_gen=3,2,0），max_chunks=2 → lru_gen=3,2 被衰减，0 跳过。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"lp2_{uuid.uuid4().hex[:8]}"
    cid3 = f"lp2g3_{uuid.uuid4().hex[:6]}"
    cid2 = f"lp2g2_{uuid.uuid4().hex[:6]}"
    cid0 = f"lp2g0_{uuid.uuid4().hex[:6]}"

    _insert_chunk(conn, cid3, proj, importance=0.7, lru_gen=3, days_ago=10.0)
    _insert_chunk(conn, cid2, proj, importance=0.7, lru_gen=2, days_ago=10.0)
    _insert_chunk(conn, cid0, proj, importance=0.7, lru_gen=0, days_ago=10.0)

    imp3_before = _get_importance(conn, cid3)
    imp2_before = _get_importance(conn, cid2)
    imp0_before = _get_importance(conn, cid0)

    apply_ebbinghaus_decay(conn, proj, max_chunks=2)
    conn.commit()

    imp3_after = _get_importance(conn, cid3)
    imp2_after = _get_importance(conn, cid2)
    imp0_after = _get_importance(conn, cid0)

    assert imp3_after < imp3_before, f"LP2: lru_gen=3 应衰减"
    assert imp2_after < imp2_before, f"LP2: lru_gen=2 应衰减"
    assert abs(imp0_after - imp0_before) < 0.01, (
        f"LP2: lru_gen=0 应跳过, {imp0_before:.3f}→{imp0_after:.3f}"
    )
    conn.close()
    print(f"  LP2 PASS: gen3={imp3_before:.3f}→{imp3_after:.3f}, "
          f"gen2={imp2_before:.3f}→{imp2_after:.3f}, gen0={imp0_after:.3f}(skipped)")


def test_lp3_all_chunks_decay_with_large_budget():
    """LP3: max_chunks 足够大时，冷热 chunk 都被衰减。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"lp3_{uuid.uuid4().hex[:8]}"
    cid_cold = f"lp3c_{uuid.uuid4().hex[:6]}"
    cid_hot = f"lp3h_{uuid.uuid4().hex[:6]}"

    _insert_chunk(conn, cid_cold, proj, importance=0.7, lru_gen=5, days_ago=10.0)
    _insert_chunk(conn, cid_hot, proj, importance=0.7, lru_gen=0, days_ago=10.0)

    imp_cold_before = _get_importance(conn, cid_cold)
    imp_hot_before = _get_importance(conn, cid_hot)

    apply_ebbinghaus_decay(conn, proj, max_chunks=100)
    conn.commit()

    imp_cold_after = _get_importance(conn, cid_cold)
    imp_hot_after = _get_importance(conn, cid_hot)

    assert imp_cold_after < imp_cold_before, "LP3: 冷 chunk 应衰减"
    assert imp_hot_after < imp_hot_before, "LP3: 热 chunk 也应衰减（budget 足够）"
    conn.close()
    print(f"  LP3 PASS: both decay — cold {imp_cold_before:.3f}→{imp_cold_after:.3f}, "
          f"hot {imp_hot_before:.3f}→{imp_hot_after:.3f}")


def test_lp4_same_lru_gen_ordered_by_last_accessed():
    """LP4: 相同 lru_gen 时，last_accessed 更早的先被衰减（max_chunks=1）。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"lp4_{uuid.uuid4().hex[:8]}"
    cid_older = f"lp4old_{uuid.uuid4().hex[:6]}"
    cid_newer = f"lp4new_{uuid.uuid4().hex[:6]}"

    _insert_chunk(conn, cid_older, proj, importance=0.7, lru_gen=2, days_ago=20.0)  # 更老
    _insert_chunk(conn, cid_newer, proj, importance=0.7, lru_gen=2, days_ago=5.0)   # 较新

    imp_older_before = _get_importance(conn, cid_older)
    imp_newer_before = _get_importance(conn, cid_newer)

    apply_ebbinghaus_decay(conn, proj, max_chunks=1)
    conn.commit()

    imp_older_after = _get_importance(conn, cid_older)
    imp_newer_after = _get_importance(conn, cid_newer)

    assert imp_older_after < imp_older_before, (
        f"LP4: older chunk 应先被衰减, {imp_older_before:.3f}→{imp_older_after:.3f}"
    )
    assert abs(imp_newer_after - imp_newer_before) < 0.01, (
        f"LP4: newer chunk 应跳过, {imp_newer_before:.3f}→{imp_newer_after:.3f}"
    )
    conn.close()
    print(f"  LP4 PASS: same gen, older {imp_older_before:.3f}→{imp_older_after:.3f} "
          f"decays first, newer {imp_newer_after:.3f} skipped")


if __name__ == "__main__":
    print("Ebbinghaus 冷 chunk 优先衰减测试（iter486）")
    print("=" * 60)

    tests = [
        test_lp1_cold_chunk_decayed_first,
        test_lp2_hot_chunk_skipped_when_budget_exhausted,
        test_lp3_all_chunks_decay_with_large_budget,
        test_lp4_same_lru_gen_ordered_by_last_accessed,
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
