#!/usr/bin/env python3
"""
test_per_type_confidence_decay.py — per-type confidence decay factor 测试（iter488）

覆盖：
  TC1: task_state chunk → confidence 衰减最快（FACTOR=1.0）
  TC2: design_constraint chunk → confidence 衰减最慢（FACTOR=6.0）
  TC3: decision chunk → confidence 衰减中速（FACTOR=2.0，默认）
  TC4: task_state 衰减率 > decision > design_constraint（三者排序）
  TC5: design_constraint confidence 在 10天后仍保持高水平
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


def _insert_chunk(conn, cid, project, chunk_type="decision", importance=0.6,
                  stability=3.0, days_ago=10.0, confidence=0.80):
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
    """, (cid, project, "test", chunk_type, "tc test", "tc test",
          importance, stability, 0.5, "episodic", json.dumps([]),
          5, 0, now_iso, now_iso, last_accessed,
          None, 0, "tc test", "{}", "",
          confidence, "pending"))
    conn.commit()


def _get_confidence(conn, cid):
    row = conn.execute("SELECT confidence_score FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else None


def test_tc1_reasoning_chain_fast_decay():
    """TC1: reasoning_chain → confidence 衰减较快（FACTOR=1.5）。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"tc1_{uuid.uuid4().hex[:8]}"
    cid = f"tc1c_{uuid.uuid4().hex[:6]}"
    initial_conf = 0.80
    stability = 3.0
    days = 10.0

    _insert_chunk(conn, cid, proj, chunk_type="reasoning_chain",
                  stability=stability, days_ago=days, confidence=initial_conf)
    apply_ebbinghaus_decay(conn, proj, max_chunks=10)
    conn.commit()

    new_conf = _get_confidence(conn, cid)
    # FACTOR=1.5: exp(-10/4.5) ≈ 0.108, × 0.80 ≈ 0.086 → floor 0.10
    expected = max(0.10, initial_conf * math.exp(-days / (stability * 1.5)))
    assert abs(new_conf - expected) < 0.005, (
        f"TC1: reasoning_chain conf 应快速衰减到 ≈{expected:.3f}, got {new_conf:.3f}"
    )
    conn.close()
    print(f"  TC1 PASS: reasoning_chain conf {initial_conf:.3f}→{new_conf:.3f} (factor=1.5, expected≈{expected:.3f})")


def test_tc2_design_constraint_slow_decay():
    """TC2: design_constraint → confidence 衰减最慢（FACTOR=6.0）。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"tc2_{uuid.uuid4().hex[:8]}"
    cid = f"tc2c_{uuid.uuid4().hex[:6]}"
    initial_conf = 0.80
    stability = 3.0
    days = 10.0

    _insert_chunk(conn, cid, proj, chunk_type="design_constraint",
                  stability=stability, days_ago=days, confidence=initial_conf)
    apply_ebbinghaus_decay(conn, proj, max_chunks=10)
    conn.commit()

    new_conf = _get_confidence(conn, cid)
    # FACTOR=6.0: exp(-10/18) ≈ 0.576, × 0.80 ≈ 0.461
    expected = max(0.10, initial_conf * math.exp(-days / (stability * 6.0)))
    assert abs(new_conf - expected) < 0.005, (
        f"TC2: design_constraint conf 应缓慢衰减到 ≈{expected:.3f}, got {new_conf:.3f}"
    )
    conn.close()
    print(f"  TC2 PASS: design_constraint conf {initial_conf:.3f}→{new_conf:.3f} (factor=6.0, expected≈{expected:.3f})")


def test_tc3_decision_default_decay():
    """TC3: decision → confidence 中速衰减（FACTOR=2.0）。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"tc3_{uuid.uuid4().hex[:8]}"
    cid = f"tc3c_{uuid.uuid4().hex[:6]}"
    initial_conf = 0.80
    stability = 3.0
    days = 10.0

    _insert_chunk(conn, cid, proj, chunk_type="decision",
                  stability=stability, days_ago=days, confidence=initial_conf)
    apply_ebbinghaus_decay(conn, proj, max_chunks=10)
    conn.commit()

    new_conf = _get_confidence(conn, cid)
    # FACTOR=2.0: exp(-10/6) ≈ 0.189, × 0.80 ≈ 0.151
    expected = max(0.10, initial_conf * math.exp(-days / (stability * 2.0)))
    assert abs(new_conf - expected) < 0.005, (
        f"TC3: decision conf 中速衰减 ≈{expected:.3f}, got {new_conf:.3f}"
    )
    conn.close()
    print(f"  TC3 PASS: decision conf {initial_conf:.3f}→{new_conf:.3f} (factor=2.0, expected≈{expected:.3f})")


def test_tc4_decay_ordering_reasoning_gt_decision_gt_constraint():
    """TC4: reasoning_chain 衰减率 > decision > design_constraint（3种排序）。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"tc4_{uuid.uuid4().hex[:8]}"
    stability = 3.0
    days = 10.0
    initial_conf = 0.80

    cids = {}
    for ctype in ["reasoning_chain", "decision", "design_constraint"]:
        cid = f"tc4_{ctype[:4]}_{uuid.uuid4().hex[:6]}"
        cids[ctype] = cid
        _insert_chunk(conn, cid, proj, chunk_type=ctype,
                      stability=stability, days_ago=days, confidence=initial_conf)

    apply_ebbinghaus_decay(conn, proj, max_chunks=10)
    conn.commit()

    confs = {ctype: _get_confidence(conn, cid) for ctype, cid in cids.items()}

    assert confs["reasoning_chain"] <= confs["decision"], (
        f"TC4: reasoning_chain({confs['reasoning_chain']:.3f}) 应 <= decision({confs['decision']:.3f})"
    )
    assert confs["decision"] <= confs["design_constraint"], (
        f"TC4: decision({confs['decision']:.3f}) 应 <= design_constraint({confs['design_constraint']:.3f})"
    )
    conn.close()
    print(f"  TC4 PASS: reasoning_chain={confs['reasoning_chain']:.3f} ≤ "
          f"decision={confs['decision']:.3f} ≤ "
          f"design_constraint={confs['design_constraint']:.3f}")


def test_tc5_design_constraint_high_confidence_preserved():
    """TC5: design_constraint confidence 10天后仍 > 0.40（规范知识保持高可信）。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"tc5_{uuid.uuid4().hex[:8]}"
    cid = f"tc5c_{uuid.uuid4().hex[:6]}"

    _insert_chunk(conn, cid, proj, chunk_type="design_constraint",
                  stability=3.0, days_ago=10.0, confidence=0.80)

    apply_ebbinghaus_decay(conn, proj, max_chunks=10)
    conn.commit()

    new_conf = _get_confidence(conn, cid)
    assert new_conf > 0.40, (
        f"TC5: design_constraint 10天后 confidence 应 > 0.40（高可信保留），got {new_conf:.3f}"
    )
    conn.close()
    print(f"  TC5 PASS: design_constraint 10d later confidence={new_conf:.3f} > 0.40")


if __name__ == "__main__":
    print("per-type confidence decay factor 测试（iter488）")
    print("=" * 60)

    tests = [
        test_tc1_reasoning_chain_fast_decay,
        test_tc2_design_constraint_slow_decay,
        test_tc3_decision_default_decay,
        test_tc4_decay_ordering_reasoning_gt_decision_gt_constraint,
        test_tc5_design_constraint_high_confidence_preserved,
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
