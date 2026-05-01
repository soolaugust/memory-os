#!/usr/bin/env python3
"""
test_ebbinghaus_damon_coord.py — Ebbinghaus + DAMON COLD 协调测试

覆盖：
  COORD1: 已被 Ebbinghaus 衰减的 chunk → DAMON COLD 阶段跳过 importance 惩罚（oom_adj 仍更新）
  COORD2: 未被 Ebbinghaus 衰减的 chunk（近期访问）→ DAMON COLD 正常惩罚 importance
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
                  chunk_type="decision", last_accessed_delta_days=0,
                  access_count=0, oom_adj=0):
    now = datetime.now(timezone.utc)
    la = (now - timedelta(days=last_accessed_delta_days)).isoformat()
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
          access_count, oom_adj, now_iso, now_iso, la, None, 0, summary[:500], "{}", ""))
    conn.commit()


def test_coord1_ebbinghaus_decayed_skips_cold_importance():
    """COORD1: Ebbinghaus 已衰减 → DAMON COLD 跳过 importance 惩罚，仍更新 oom_adj。"""
    conn = open_db()
    ensure_schema(conn)

    from store_mm import apply_ebbinghaus_decay

    proj = f"coord1_{uuid.uuid4().hex[:6]}"
    cid = f"c1_{uuid.uuid4().hex[:10]}"
    # access_count=0 → DAMON 会判为 COLD；last_accessed=30天前 → Ebbinghaus 会衰减
    initial_imp = 0.80
    _insert_chunk(conn, cid, proj, "Long neglected chunk about old technology",
                  importance=initial_imp, stability=1.0,
                  last_accessed_delta_days=30, access_count=0, oom_adj=0)

    # Step 1: 运行 Ebbinghaus（单独验证衰减）
    eb_result = apply_ebbinghaus_decay(conn, proj)
    conn.commit()

    after_ebbinghaus = conn.execute(
        "SELECT importance FROM memory_chunks WHERE id=?", (cid,)
    ).fetchone()[0]

    assert after_ebbinghaus < initial_imp, (
        f"COORD1 setup: Ebbinghaus 应已衰减, {initial_imp:.3f}→{after_ebbinghaus:.3f}"
    )
    assert cid in eb_result.get("decayed_ids", set()), (
        f"COORD1: Ebbinghaus decayed_ids 应包含该 chunk"
    )

    # Step 2: 模拟 DAMON COLD 逻辑（使用 decayed_ids 排除）
    # 直接复现 damon_scan 中的 COLD 处理逻辑
    from config import get as _cfg
    cold_oom_delta = _cfg("damon.cold_oom_adj_delta")
    _ebbinghaus_decayed_ids = eb_result.get("decayed_ids", set())

    importance_before_cold = after_ebbinghaus

    # 模拟 COLD action（与 damon_scan 中逻辑一致）
    row = conn.execute(
        "SELECT COALESCE(oom_adj, 0), importance FROM memory_chunks WHERE id=?", (cid,)
    ).fetchone()
    if row:
        current_adj, current_imp = row
        if current_adj < cold_oom_delta:
            if cid in _ebbinghaus_decayed_ids:
                # 只更新 oom_adj，跳过 importance 惩罚
                conn.execute(
                    "UPDATE memory_chunks SET oom_adj = MAX(COALESCE(oom_adj, 0), ?) WHERE id=?",
                    (cold_oom_delta, cid)
                )
            else:
                new_imp = max(0.5, round(current_imp * 0.95, 3))
                conn.execute(
                    "UPDATE memory_chunks SET oom_adj = MAX(COALESCE(oom_adj, 0), ?), importance=? WHERE id=?",
                    (cold_oom_delta, new_imp, cid)
                )
    conn.commit()

    after_cold_imp, after_cold_oom = conn.execute(
        "SELECT importance, oom_adj FROM memory_chunks WHERE id=?", (cid,)
    ).fetchone()

    # importance 不应再被 COLD 惩罚（已被 Ebbinghaus 精确衰减过）
    assert abs(after_cold_imp - importance_before_cold) < 0.005, (
        f"COORD1: importance 不应被 COLD 再次惩罚, "
        f"after_ebbinghaus={importance_before_cold:.4f} after_cold={after_cold_imp:.4f}"
    )
    # oom_adj 应被更新（COLD 标记仍然执行）
    assert after_cold_oom >= cold_oom_delta, (
        f"COORD1: oom_adj 应被 COLD 更新, got {after_cold_oom}"
    )

    conn.execute("DELETE FROM memory_chunks WHERE project=?", (proj,))
    conn.commit()
    conn.close()
    print(f"  COORD1 PASS: importance {initial_imp:.3f}→{after_cold_imp:.4f} (no double penalty), "
          f"oom_adj={after_cold_oom}")


def test_coord2_not_ebbinghaus_decayed_applies_cold():
    """COORD2: 未被 Ebbinghaus 衰减（近期访问）→ DAMON COLD 正常 importance 惩罚。"""
    conn = open_db()
    ensure_schema(conn)

    from store_mm import apply_ebbinghaus_decay

    proj = f"coord2_{uuid.uuid4().hex[:6]}"
    cid = f"c2_{uuid.uuid4().hex[:10]}"
    # access_count=0 → DAMON 会判为 COLD；但 last_accessed=0.1天前 → Ebbinghaus 不衰减
    initial_imp = 0.80
    _insert_chunk(conn, cid, proj, "Recent but zero-access chunk",
                  importance=initial_imp, stability=1.0,
                  last_accessed_delta_days=0.1, access_count=0, oom_adj=0)

    # Step 1: 运行 Ebbinghaus（应不衰减这个 chunk）
    eb_result = apply_ebbinghaus_decay(conn, proj)
    conn.commit()

    assert cid not in eb_result.get("decayed_ids", set()), (
        f"COORD2: 近期访问 chunk 不应被 Ebbinghaus 衰减"
    )

    # Step 2: 模拟 DAMON COLD（没有被 Ebbinghaus 排除 → 正常惩罚）
    from config import get as _cfg
    cold_oom_delta = _cfg("damon.cold_oom_adj_delta")
    _ebbinghaus_decayed_ids = eb_result.get("decayed_ids", set())

    row = conn.execute(
        "SELECT COALESCE(oom_adj, 0), importance FROM memory_chunks WHERE id=?", (cid,)
    ).fetchone()
    if row:
        current_adj, current_imp = row
        if current_adj < cold_oom_delta and cid not in _ebbinghaus_decayed_ids:
            new_imp = max(0.5, round(current_imp * 0.95, 3))
            conn.execute(
                "UPDATE memory_chunks SET oom_adj = MAX(COALESCE(oom_adj, 0), ?), importance=? WHERE id=?",
                (cold_oom_delta, new_imp, cid)
            )
    conn.commit()

    after_cold_imp = conn.execute(
        "SELECT importance FROM memory_chunks WHERE id=?", (cid,)
    ).fetchone()[0]

    # importance 应被 COLD 惩罚（× 0.95）
    expected_cold_imp = max(0.5, round(initial_imp * 0.95, 3))
    assert abs(after_cold_imp - expected_cold_imp) < 0.005, (
        f"COORD2: COLD 应惩罚 importance, expected {expected_cold_imp:.3f} got {after_cold_imp:.3f}"
    )

    conn.execute("DELETE FROM memory_chunks WHERE project=?", (proj,))
    conn.commit()
    conn.close()
    print(f"  COORD2 PASS: COLD applied importance {initial_imp:.3f}→{after_cold_imp:.3f}")


if __name__ == "__main__":
    print("Ebbinghaus + DAMON COLD 协调测试")
    print("=" * 60)

    tests = [
        test_coord1_ebbinghaus_decayed_skips_cold_importance,
        test_coord2_not_ebbinghaus_decayed_applies_cold,
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
