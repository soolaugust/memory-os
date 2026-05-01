#!/usr/bin/env python3
"""
test_confidence_decay.py — confidence_score Ebbinghaus 时间衰减测试（iter478）

覆盖：
  CD1: 长时间未访问的 chunk → confidence_score 衰减
  CD2: confidence 衰减速率慢于 importance（CONFIDENCE_DECAY_FACTOR=2.0）
  CD3: confidence_score 下界保护（不低于 0.10）
  CD4: 近期访问的 chunk（< 1天）→ confidence 不衰减
  CD5: 高稳定性 chunk → confidence 衰减更慢
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


def _insert_chunk(conn, cid, project, importance=0.6, stability=2.0,
                  confidence_score=0.8, last_accessed_days_ago=10.0,
                  chunk_type="decision"):
    now = datetime.now(timezone.utc)
    last_accessed = (now - timedelta(days=last_accessed_days_ago)).isoformat()
    now_iso = now.isoformat()
    conn.execute("""
        INSERT OR REPLACE INTO memory_chunks
        (id, project, source_session, chunk_type, summary, content,
         importance, stability, retrievability, info_class, tags,
         access_count, oom_adj, created_at, updated_at, last_accessed,
         feishu_url, lru_gen, raw_snippet, encode_context, session_type_history,
         confidence_score)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (cid, project, "test", chunk_type,
          "Test chunk summary", "Test chunk content",
          importance, stability, 0.5, "episodic", json.dumps([]),
          5, 0, now_iso, now_iso, last_accessed,
          None, 0, "Test chunk summary"[:500], "{}", "",
          confidence_score))
    conn.commit()


def _get_chunk(conn, cid):
    row = conn.execute(
        "SELECT importance, confidence_score, updated_at FROM memory_chunks WHERE id=?",
        (cid,)
    ).fetchone()
    if not row:
        return None
    return {"importance": float(row[0] or 0.5),
            "confidence_score": float(row[1] or 0.7),
            "updated_at": row[2]}


def test_cd1_confidence_decays_with_time():
    """CD1: 长时间未访问 → confidence_score 应衰减。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"cd1_{uuid.uuid4().hex[:6]}"
    cid = f"cd1c_{uuid.uuid4().hex[:10]}"
    initial_conf = 0.80
    # 10天前访问，stability=2.0 → 显著衰减
    _insert_chunk(conn, cid, proj, importance=0.6, stability=2.0,
                  confidence_score=initial_conf, last_accessed_days_ago=10.0)

    result = apply_ebbinghaus_decay(conn, proj, max_chunks=10)
    conn.commit()

    chunk = _get_chunk(conn, cid)
    new_conf = chunk["confidence_score"]

    assert new_conf < initial_conf, (
        f"CD1: confidence 应随时间衰减, {initial_conf:.3f}→{new_conf:.3f}"
    )

    # 验证衰减量符合公式：new = old × exp(-10 / (2.0 × 2.0))
    expected_conf = initial_conf * math.exp(-10.0 / (2.0 * 2.0))
    assert abs(new_conf - max(0.10, expected_conf)) < 0.005, (
        f"CD1: 衰减量偏差过大, expected≈{expected_conf:.3f}, got {new_conf:.3f}"
    )

    conn.execute("DELETE FROM memory_chunks WHERE project=?", (proj,))
    conn.commit()
    conn.close()
    print(f"  CD1 PASS: confidence decayed {initial_conf:.3f}→{new_conf:.3f} "
          f"(expected≈{max(0.10, expected_conf):.3f})")


def test_cd2_confidence_decays_slower_than_importance():
    """CD2: confidence 衰减速率慢于 importance（FACTOR=2.0）。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"cd2_{uuid.uuid4().hex[:6]}"
    cid = f"cd2c_{uuid.uuid4().hex[:10]}"
    initial_imp = 0.70
    initial_conf = 0.80
    stability = 3.0
    days = 6.0

    _insert_chunk(conn, cid, proj, importance=initial_imp, stability=stability,
                  confidence_score=initial_conf, last_accessed_days_ago=days)

    apply_ebbinghaus_decay(conn, proj, max_chunks=10)
    conn.commit()

    chunk = _get_chunk(conn, cid)

    # importance 衰减：exp(-6/3) = exp(-2) ≈ 0.135
    expected_imp = max(0.05, initial_imp * math.exp(-days / stability))
    # confidence 衰减：exp(-6/(3×2)) = exp(-1) ≈ 0.368
    expected_conf = max(0.10, initial_conf * math.exp(-days / (stability * 2.0)))

    imp_drop = initial_imp - chunk["importance"]
    conf_drop = initial_conf - chunk["confidence_score"]

    assert imp_drop > conf_drop, (
        f"CD2: importance 降幅({imp_drop:.3f}) 应大于 confidence 降幅({conf_drop:.3f})"
    )
    print(f"  CD2 PASS: imp_drop={imp_drop:.3f} > conf_drop={conf_drop:.3f} "
          f"(imp: {initial_imp:.3f}→{chunk['importance']:.3f}, "
          f"conf: {initial_conf:.3f}→{chunk['confidence_score']:.3f})")

    conn.execute("DELETE FROM memory_chunks WHERE project=?", (proj,))
    conn.commit()
    conn.close()


def test_cd3_confidence_lower_bound():
    """CD3: confidence_score 衰减后不低于 0.10。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"cd3_{uuid.uuid4().hex[:6]}"
    cid = f"cd3c_{uuid.uuid4().hex[:10]}"

    # 极低初始 confidence，长时间未访问 → 应钳制在 0.10
    _insert_chunk(conn, cid, proj, importance=0.6, stability=0.5,
                  confidence_score=0.15, last_accessed_days_ago=30.0)

    apply_ebbinghaus_decay(conn, proj, max_chunks=10)
    conn.commit()

    chunk = _get_chunk(conn, cid)
    assert chunk["confidence_score"] >= 0.10, (
        f"CD3: confidence 下界保护失败, got {chunk['confidence_score']:.4f}"
    )
    print(f"  CD3 PASS: confidence lower bound respected (got {chunk['confidence_score']:.4f})")

    conn.execute("DELETE FROM memory_chunks WHERE project=?", (proj,))
    conn.commit()
    conn.close()


def test_cd4_recent_access_no_confidence_decay():
    """CD4: 近期访问 chunk（0.5天前）→ confidence 不衰减。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"cd4_{uuid.uuid4().hex[:6]}"
    cid = f"cd4c_{uuid.uuid4().hex[:10]}"
    initial_conf = 0.75

    # 0.5天前访问 → 间隔保护（<1天不衰减）
    _insert_chunk(conn, cid, proj, importance=0.6, stability=2.0,
                  confidence_score=initial_conf, last_accessed_days_ago=0.5)

    apply_ebbinghaus_decay(conn, proj, max_chunks=10)
    conn.commit()

    chunk = _get_chunk(conn, cid)
    assert abs(chunk["confidence_score"] - initial_conf) < 0.001, (
        f"CD4: 近期访问 chunk confidence 不应衰减, "
        f"{initial_conf:.3f}→{chunk['confidence_score']:.3f}"
    )
    print(f"  CD4 PASS: recent chunk confidence unchanged: {chunk['confidence_score']:.3f}")

    conn.execute("DELETE FROM memory_chunks WHERE project=?", (proj,))
    conn.commit()
    conn.close()


def test_cd5_high_stability_confidence_decays_slower():
    """CD5: 高稳定性 chunk confidence 衰减比低稳定性更慢。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"cd5_{uuid.uuid4().hex[:6]}"

    cid_high = f"cd5h_{uuid.uuid4().hex[:8]}"
    cid_low = f"cd5l_{uuid.uuid4().hex[:8]}"
    initial_conf = 0.80
    days = 8.0

    # 高稳定性：stability=10
    _insert_chunk(conn, cid_high, proj, importance=0.6, stability=10.0,
                  confidence_score=initial_conf, last_accessed_days_ago=days)
    # 低稳定性：stability=1
    _insert_chunk(conn, cid_low, proj, importance=0.6, stability=1.0,
                  confidence_score=initial_conf, last_accessed_days_ago=days)

    apply_ebbinghaus_decay(conn, proj, max_chunks=10)
    conn.commit()

    high_chunk = _get_chunk(conn, cid_high)
    low_chunk = _get_chunk(conn, cid_low)

    assert high_chunk["confidence_score"] > low_chunk["confidence_score"], (
        f"CD5: 高稳定性 confidence({high_chunk['confidence_score']:.3f}) "
        f"应大于低稳定性({low_chunk['confidence_score']:.3f})"
    )
    print(f"  CD5 PASS: high-stab conf={high_chunk['confidence_score']:.3f} > "
          f"low-stab conf={low_chunk['confidence_score']:.3f}")

    conn.execute("DELETE FROM memory_chunks WHERE project=?", (proj,))
    conn.commit()
    conn.close()


if __name__ == "__main__":
    print("confidence_score Ebbinghaus 时间衰减测试（iter478）")
    print("=" * 60)

    tests = [
        test_cd1_confidence_decays_with_time,
        test_cd2_confidence_decays_slower_than_importance,
        test_cd3_confidence_lower_bound,
        test_cd4_recent_access_no_confidence_decay,
        test_cd5_high_stability_confidence_decays_slower,
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
