#!/usr/bin/env python3
"""
test_iter491_dynamic_cutoff.py — Ebbinghaus stability-aware cutoff 测试（iter491）

覆盖：
  DC1: stability < 2.0 → cutoff=0.5天 → 0.6天前访问的 chunk 应被衰减
  DC2: stability ∈ [2.0, 5.0) → cutoff=1.0天 → 0.6天前访问的 chunk 不应被衰减
  DC3: stability ∈ [5.0, 10.0) → cutoff=3.0天 → 2天前访问的 chunk 不应被衰减
  DC4: stability >= 10.0 → cutoff=7.0天 → 5天前访问的 chunk 不应被衰减
  DC5: stability >= 10.0 → cutoff=7.0天 → 8天前访问的 chunk 应被衰减
  DC6: 不同 stability 的 chunk 混合 → 只有达到各自保护期的 chunk 被衰减
"""
import sys
import json
import uuid
import math
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import tmpfs  # noqa

from store import open_db, ensure_schema
from store_mm import apply_ebbinghaus_decay


def _insert_chunk(conn, cid, project, stability=1.0, days_ago=0.6,
                  chunk_type="decision", importance=0.7, confidence=0.8):
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
    """, (cid, project, "test", chunk_type, "dc test", "dc test",
          importance, stability, 0.5, "episodic", json.dumps([]),
          0, 0, now_iso, now_iso, last_accessed,
          None, 2, "dc test", "{}", "",
          confidence, "pending"))
    conn.commit()


def _get_importance(conn, cid):
    row = conn.execute("SELECT importance FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else None


def test_dc1_low_stability_decays_after_half_day():
    """DC1: stability=1.0（<2.0）→ cutoff=0.5天 → 0.6天前访问的 chunk 应被衰减。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"dc1_{uuid.uuid4().hex[:8]}"
    cid = f"dc1c_{uuid.uuid4().hex[:6]}"

    initial_imp = 0.7
    _insert_chunk(conn, cid, proj, stability=1.0, days_ago=0.6, importance=initial_imp)

    result = apply_ebbinghaus_decay(conn, proj, max_chunks=10)
    conn.commit()

    new_imp = _get_importance(conn, cid)
    assert new_imp < initial_imp, (
        f"DC1: stability=1.0, last_accessed=0.6d ago → 超过 0.5d cutoff，应被衰减\n"
        f"  initial={initial_imp:.4f}, new={new_imp:.4f}"
    )
    expected = max(0.05, initial_imp * math.exp(-0.6 / 1.0))
    assert abs(new_imp - expected) < 0.01, (
        f"DC1: decay 幅度偏差太大 expected≈{expected:.4f}, got {new_imp:.4f}"
    )
    conn.close()
    print(f"  DC1 PASS: stability=1.0, 0.6d → decayed {initial_imp:.4f}→{new_imp:.4f} (cutoff=0.5d)")


def test_dc2_mid_stability_protected_under_one_day():
    """DC2: stability=3.0（∈[2.0,5.0)）→ cutoff=1.0天 → 0.6天前访问不应被衰减。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"dc2_{uuid.uuid4().hex[:8]}"
    cid = f"dc2c_{uuid.uuid4().hex[:6]}"

    initial_imp = 0.7
    _insert_chunk(conn, cid, proj, stability=3.0, days_ago=0.6, importance=initial_imp)

    apply_ebbinghaus_decay(conn, proj, max_chunks=10)
    conn.commit()

    new_imp = _get_importance(conn, cid)
    assert new_imp == initial_imp, (
        f"DC2: stability=3.0, last_accessed=0.6d ago → 未达 1.0d cutoff，不应被衰减\n"
        f"  initial={initial_imp:.4f}, new={new_imp:.4f}"
    )
    conn.close()
    print(f"  DC2 PASS: stability=3.0, 0.6d → protected (cutoff=1.0d, still {new_imp:.4f})")


def test_dc3_high_stability_protected_under_three_days():
    """DC3: stability=6.0（∈[5.0,10.0)）→ cutoff=3.0天 → 2天前访问不应被衰减。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"dc3_{uuid.uuid4().hex[:8]}"
    cid = f"dc3c_{uuid.uuid4().hex[:6]}"

    initial_imp = 0.7
    _insert_chunk(conn, cid, proj, stability=6.0, days_ago=2.0, importance=initial_imp)

    apply_ebbinghaus_decay(conn, proj, max_chunks=10)
    conn.commit()

    new_imp = _get_importance(conn, cid)
    assert new_imp == initial_imp, (
        f"DC3: stability=6.0, last_accessed=2.0d ago → 未达 3.0d cutoff，不应被衰减\n"
        f"  initial={initial_imp:.4f}, new={new_imp:.4f}"
    )
    conn.close()
    print(f"  DC3 PASS: stability=6.0, 2.0d → protected (cutoff=3.0d, still {new_imp:.4f})")


def test_dc4_very_high_stability_protected_under_seven_days():
    """DC4: stability=12.0（>=10.0）→ cutoff=7.0天 → 5天前访问不应被衰减。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"dc4_{uuid.uuid4().hex[:8]}"
    cid = f"dc4c_{uuid.uuid4().hex[:6]}"

    initial_imp = 0.7
    _insert_chunk(conn, cid, proj, stability=12.0, days_ago=5.0, importance=initial_imp)

    apply_ebbinghaus_decay(conn, proj, max_chunks=10)
    conn.commit()

    new_imp = _get_importance(conn, cid)
    assert new_imp == initial_imp, (
        f"DC4: stability=12.0, last_accessed=5.0d ago → 未达 7.0d cutoff，不应被衰减\n"
        f"  initial={initial_imp:.4f}, new={new_imp:.4f}"
    )
    conn.close()
    print(f"  DC4 PASS: stability=12.0, 5.0d → protected (cutoff=7.0d, still {new_imp:.4f})")


def test_dc5_very_high_stability_decays_after_seven_days():
    """DC5: stability=12.0（>=10.0）→ cutoff=7.0天 → 8天前访问应被衰减。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"dc5_{uuid.uuid4().hex[:8]}"
    cid = f"dc5c_{uuid.uuid4().hex[:6]}"

    initial_imp = 0.7
    _insert_chunk(conn, cid, proj, stability=12.0, days_ago=8.0, importance=initial_imp)

    apply_ebbinghaus_decay(conn, proj, max_chunks=10)
    conn.commit()

    new_imp = _get_importance(conn, cid)
    assert new_imp < initial_imp, (
        f"DC5: stability=12.0, last_accessed=8.0d ago → 超过 7.0d cutoff，应被衰减\n"
        f"  initial={initial_imp:.4f}, new={new_imp:.4f}"
    )
    expected = max(0.10, initial_imp * math.exp(-8.0 / 12.0))  # stability_high_floor=0.10
    assert abs(new_imp - expected) < 0.01, (
        f"DC5: decay 幅度偏差 expected≈{expected:.4f}, got {new_imp:.4f}"
    )
    conn.close()
    print(f"  DC5 PASS: stability=12.0, 8.0d → decayed {initial_imp:.4f}→{new_imp:.4f} (cutoff=7.0d)")


def test_dc6_mixed_stability_selective_decay():
    """DC6: 不同 stability 混合 → 只有达到各自保护期的 chunk 被衰减。

    chunk_a: stability=1.5, 0.6d → 超过 0.5d cutoff → 应衰减
    chunk_b: stability=3.0, 0.6d → 未达 1.0d cutoff → 不衰减
    chunk_c: stability=6.0, 2.0d → 未达 3.0d cutoff → 不衰减
    chunk_d: stability=12.0, 8.0d → 超过 7.0d cutoff → 应衰减
    """
    conn = open_db()
    ensure_schema(conn)
    proj = f"dc6_{uuid.uuid4().hex[:8]}"
    initial_imp = 0.7

    specs = [
        ("a", 1.5, 0.6, True),   # should decay
        ("b", 3.0, 0.6, False),  # protected
        ("c", 6.0, 2.0, False),  # protected
        ("d", 12.0, 8.0, True),  # should decay
    ]
    cids = {}
    for name, stab, days, _ in specs:
        cid = f"dc6{name}_{uuid.uuid4().hex[:6]}"
        cids[name] = cid
        _insert_chunk(conn, cid, proj, stability=stab, days_ago=days, importance=initial_imp)

    apply_ebbinghaus_decay(conn, proj, max_chunks=20)
    conn.commit()

    for name, stab, days, should_decay in specs:
        cid = cids[name]
        new_imp = _get_importance(conn, cid)
        if should_decay:
            assert new_imp < initial_imp, (
                f"DC6/{name}: stability={stab}, {days}d → 应衰减但未衰减 ({new_imp:.4f})"
            )
        else:
            assert new_imp == initial_imp, (
                f"DC6/{name}: stability={stab}, {days}d → 不应衰减但被衰减 ({new_imp:.4f})"
            )

    conn.close()
    print(f"  DC6 PASS: mixed stability → selective decay (a↓, b=, c=, d↓)")


if __name__ == "__main__":
    print("Ebbinghaus stability-aware cutoff 测试（iter491）")
    print("=" * 60)

    tests = [
        test_dc1_low_stability_decays_after_half_day,
        test_dc2_mid_stability_protected_under_one_day,
        test_dc3_high_stability_protected_under_three_days,
        test_dc4_very_high_stability_protected_under_seven_days,
        test_dc5_very_high_stability_decays_after_seven_days,
        test_dc6_mixed_stability_selective_decay,
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
