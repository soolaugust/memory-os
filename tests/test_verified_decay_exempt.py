#!/usr/bin/env python3
"""
test_verified_decay_exempt.py — verified chunk 豁免 Ebbinghaus decay 测试（iter484）

覆盖：
  VE1: verification_status='verified' → importance/confidence 不衰减
  VE2: verification_status='pending' → 正常衰减（对照组）
  VE3: verification_status='disputed' → 正常衰减（非 verified 不豁免）
  VE4: verified chunk 长时间未访问（30天）→ 仍不衰减
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


def _insert_chunk(conn, cid, project, importance=0.8, confidence=0.75,
                  stability=2.0, days_ago=10.0, verification_status="pending"):
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
    """, (cid, project, "test", "decision", "verified test", "verified test",
          importance, stability, 0.5, "episodic", json.dumps([]),
          5, 0, now_iso, now_iso, last_accessed,
          None, 0, "verified test", "{}", "",
          confidence, verification_status))
    conn.commit()


def _get_fields(conn, cid):
    row = conn.execute(
        "SELECT importance, confidence_score FROM memory_chunks WHERE id=?", (cid,)
    ).fetchone()
    return {"importance": float(row[0]), "confidence_score": float(row[1] or 0.7)}


def test_ve1_verified_chunk_not_decayed():
    """VE1: verified chunk → importance/confidence 不衰减。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"ve1_{uuid.uuid4().hex[:8]}"
    cid = f"ve1c_{uuid.uuid4().hex[:6]}"

    _insert_chunk(conn, cid, proj, importance=0.8, confidence=0.75,
                  days_ago=10.0, verification_status="verified")

    before = _get_fields(conn, cid)
    apply_ebbinghaus_decay(conn, proj, max_chunks=10)
    conn.commit()
    after = _get_fields(conn, cid)

    assert abs(after["importance"] - before["importance"]) < 0.001, (
        f"VE1: verified chunk importance 不应衰减, {before['importance']:.3f}→{after['importance']:.3f}"
    )
    assert abs(after["confidence_score"] - before["confidence_score"]) < 0.001, (
        f"VE1: verified chunk confidence 不应衰减, {before['confidence_score']:.3f}→{after['confidence_score']:.3f}"
    )
    conn.close()
    print(f"  VE1 PASS: verified chunk exempt — imp={after['importance']:.3f}, conf={after['confidence_score']:.3f} (unchanged)")


def test_ve2_pending_chunk_decays():
    """VE2: verification_status='pending' → 正常衰减（对照组）。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"ve2_{uuid.uuid4().hex[:8]}"
    cid = f"ve2c_{uuid.uuid4().hex[:6]}"

    _insert_chunk(conn, cid, proj, importance=0.8, confidence=0.75,
                  days_ago=10.0, verification_status="pending")

    before = _get_fields(conn, cid)
    apply_ebbinghaus_decay(conn, proj, max_chunks=10)
    conn.commit()
    after = _get_fields(conn, cid)

    assert after["importance"] < before["importance"], (
        f"VE2: pending chunk 应衰减, {before['importance']:.3f}→{after['importance']:.3f}"
    )
    conn.close()
    print(f"  VE2 PASS: pending chunk decays — imp: {before['importance']:.3f}→{after['importance']:.3f}")


def test_ve3_disputed_chunk_decays():
    """VE3: verification_status='disputed' → 正常衰减（非 verified 不豁免）。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"ve3_{uuid.uuid4().hex[:8]}"
    cid = f"ve3c_{uuid.uuid4().hex[:6]}"

    _insert_chunk(conn, cid, proj, importance=0.8, confidence=0.75,
                  days_ago=10.0, verification_status="disputed")

    before = _get_fields(conn, cid)
    apply_ebbinghaus_decay(conn, proj, max_chunks=10)
    conn.commit()
    after = _get_fields(conn, cid)

    assert after["importance"] < before["importance"], (
        f"VE3: disputed chunk 应衰减, {before['importance']:.3f}→{after['importance']:.3f}"
    )
    conn.close()
    print(f"  VE3 PASS: disputed chunk decays — imp: {before['importance']:.3f}→{after['importance']:.3f}")


def test_ve4_verified_long_ago_still_exempt():
    """VE4: verified chunk 30天未访问 → 仍不衰减。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"ve4_{uuid.uuid4().hex[:8]}"
    cid = f"ve4c_{uuid.uuid4().hex[:6]}"

    _insert_chunk(conn, cid, proj, importance=0.8, confidence=0.75,
                  stability=0.5, days_ago=30.0, verification_status="verified")

    before = _get_fields(conn, cid)
    apply_ebbinghaus_decay(conn, proj, max_chunks=10)
    conn.commit()
    after = _get_fields(conn, cid)

    assert abs(after["importance"] - before["importance"]) < 0.001, (
        f"VE4: verified chunk 30天后仍不应衰减, {before['importance']:.3f}→{after['importance']:.3f}"
    )
    conn.close()
    print(f"  VE4 PASS: verified chunk stable after 30 days — imp={after['importance']:.3f}")


if __name__ == "__main__":
    print("verified chunk 豁免 Ebbinghaus decay 测试（iter484）")
    print("=" * 60)

    tests = [
        test_ve1_verified_chunk_not_decayed,
        test_ve2_pending_chunk_decays,
        test_ve3_disputed_chunk_decays,
        test_ve4_verified_long_ago_still_exempt,
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
