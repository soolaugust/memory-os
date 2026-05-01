#!/usr/bin/env python3
"""
test_citation_confidence_boost.py — 引用时 confidence 增强测试（iter485）

覆盖：
  CB1: cited chunk → confidence_score += 0.05
  CB2: uncited chunk → confidence_score -= 0.005（轻微减少）
  CB3: confidence 上界保护：1.00 封顶
  CB4: confidence 增强量显著大于微减量（引用 > 未引用）
  CB5: 多次引用累积：confidence 逐步接近上限

测试策略：
  - 直接调用 _update_chunk_confidence（逻辑镜像）
  - 验证增强幅度符合 iter485 定义的 +0.05
"""
import sys
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import tmpfs  # noqa

from store import open_db, ensure_schema
from tools.citation_detector import (
    _update_chunk_confidence,
    CITED_CONFIDENCE_DELTA,
    UNCITED_CONFIDENCE_DELTA,
    MIN_CONFIDENCE,
    MAX_CONFIDENCE,
)


def _insert_chunk(conn, cid, project, confidence=0.70):
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT OR REPLACE INTO memory_chunks
        (id, project, source_session, chunk_type, summary, content,
         importance, stability, retrievability, info_class, tags,
         access_count, oom_adj, created_at, updated_at, last_accessed,
         feishu_url, lru_gen, raw_snippet, encode_context, session_type_history,
         confidence_score)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (cid, project, "test", "decision", "cb test", "cb test",
          0.7, 2.0, 0.5, "episodic", json.dumps([]),
          5, 0, now, now, now, None, 0, "cb test", "{}", "", confidence))
    conn.commit()


def test_cb1_citation_boosts_confidence():
    """CB1: 引用 chunk → confidence += 0.05。"""
    assert abs(CITED_CONFIDENCE_DELTA - 0.05) < 0.001, (
        f"CB1 前提: CITED_CONFIDENCE_DELTA 应为 0.05, got {CITED_CONFIDENCE_DELTA}"
    )

    conn = open_db()
    ensure_schema(conn)
    proj = f"cb1_{uuid.uuid4().hex[:8]}"
    cid = f"cb1c_{uuid.uuid4().hex[:6]}"
    initial_conf = 0.60

    _insert_chunk(conn, cid, proj, confidence=initial_conf)
    new_conf = _update_chunk_confidence(conn, cid, CITED_CONFIDENCE_DELTA)
    conn.commit()

    expected = min(MAX_CONFIDENCE, initial_conf + CITED_CONFIDENCE_DELTA)
    assert abs(new_conf - expected) < 0.001, (
        f"CB1: confidence 应增强到 {expected:.3f}, got {new_conf:.3f}"
    )
    conn.close()
    print(f"  CB1 PASS: cited → confidence {initial_conf:.3f}→{new_conf:.3f} (+{CITED_CONFIDENCE_DELTA})")


def test_cb2_uncited_decreases_confidence():
    """CB2: 未引用 chunk → confidence -= 0.005。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"cb2_{uuid.uuid4().hex[:8]}"
    cid = f"cb2c_{uuid.uuid4().hex[:6]}"
    initial_conf = 0.70

    _insert_chunk(conn, cid, proj, confidence=initial_conf)
    new_conf = _update_chunk_confidence(conn, cid, UNCITED_CONFIDENCE_DELTA)
    conn.commit()

    expected = max(MIN_CONFIDENCE, initial_conf + UNCITED_CONFIDENCE_DELTA)
    assert abs(new_conf - expected) < 0.001, (
        f"CB2: uncited confidence 应小幅减少到 {expected:.3f}, got {new_conf:.3f}"
    )
    conn.close()
    print(f"  CB2 PASS: uncited → confidence {initial_conf:.3f}→{new_conf:.3f} ({UNCITED_CONFIDENCE_DELTA})")


def test_cb3_confidence_upper_bound():
    """CB3: confidence 不超过 1.00（上界保护）。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"cb3_{uuid.uuid4().hex[:8]}"
    cid = f"cb3c_{uuid.uuid4().hex[:6]}"

    _insert_chunk(conn, cid, proj, confidence=0.98)
    new_conf = _update_chunk_confidence(conn, cid, CITED_CONFIDENCE_DELTA)  # 0.98 + 0.05 > 1.0
    conn.commit()

    assert new_conf <= MAX_CONFIDENCE, (
        f"CB3: confidence 上界 {MAX_CONFIDENCE}, got {new_conf:.3f}"
    )
    assert new_conf == MAX_CONFIDENCE, (
        f"CB3: 超出上界应钳制到 {MAX_CONFIDENCE}, got {new_conf:.3f}"
    )
    conn.close()
    print(f"  CB3 PASS: confidence clamped to {MAX_CONFIDENCE} (from 0.98 + {CITED_CONFIDENCE_DELTA})")


def test_cb4_cited_boost_larger_than_uncited_drop():
    """CB4: 引用增强量 (+0.05) > 未引用减少量 (0.005)，保证被引用知识净受益。"""
    assert CITED_CONFIDENCE_DELTA > abs(UNCITED_CONFIDENCE_DELTA), (
        f"CB4: cited delta ({CITED_CONFIDENCE_DELTA}) 应大于 uncited delta ({abs(UNCITED_CONFIDENCE_DELTA)})"
    )
    # 10次引用 vs 10次未引用的净效果
    net_cite = CITED_CONFIDENCE_DELTA * 10
    net_uncite = UNCITED_CONFIDENCE_DELTA * 10
    assert net_cite > 0, f"CB4: 10次引用应正向增强, got {net_cite}"
    assert net_uncite < 0, f"CB4: 10次未引用应减少, got {net_uncite}"
    print(f"  CB4 PASS: cited=+{CITED_CONFIDENCE_DELTA}/call > uncited={UNCITED_CONFIDENCE_DELTA}/call "
          f"(10x: +{net_cite:.2f} vs {net_uncite:.3f})")


def test_cb5_cumulative_citations_raise_confidence():
    """CB5: 多次引用累积 → confidence 逐步提升。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"cb5_{uuid.uuid4().hex[:8]}"
    cid = f"cb5c_{uuid.uuid4().hex[:6]}"

    _insert_chunk(conn, cid, proj, confidence=0.50)

    confs = [0.50]
    for _ in range(8):  # 8次引用
        c = _update_chunk_confidence(conn, cid, CITED_CONFIDENCE_DELTA)
        confs.append(float(c))
    conn.commit()

    assert confs[-1] > confs[0], (
        f"CB5: 8次引用后 confidence 应提升，{confs[0]:.2f}→{confs[-1]:.2f}"
    )
    assert confs[-1] <= MAX_CONFIDENCE, "CB5: confidence 不超过上界"
    conn.close()
    print(f"  CB5 PASS: 8 citations → confidence {confs[0]:.2f}→{confs[-1]:.2f}")


if __name__ == "__main__":
    print("引用时 confidence 增强测试（iter485）")
    print("=" * 60)

    tests = [
        test_cb1_citation_boosts_confidence,
        test_cb2_uncited_decreases_confidence,
        test_cb3_confidence_upper_bound,
        test_cb4_cited_boost_larger_than_uncited_drop,
        test_cb5_cumulative_citations_raise_confidence,
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
