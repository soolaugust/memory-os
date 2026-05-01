#!/usr/bin/env python3
"""
test_confidence_filter.py — confidence_score 阈值过滤测试（iter482）

覆盖：
  CF1: confidence_score < 0.15 → score 归零（极低可信 chunk 不注入）
  CF2: confidence_score >= 0.15 → score 正常计算（不受阈值影响）
  CF3: design_constraint 类型豁免过滤（即使 confidence < 0.15 也保留分数）
  CF4: 边界值 confidence=0.15 → 不过滤（等于阈值时不触发）
  CF5: confidence=0.14 vs 0.15 → 0.14 被过滤，0.15 正常

测试策略：
  - 直接构造 chunk dict，模拟 _score_chunk 后的过滤逻辑
  - 验证 score=0.0 意味着 chunk 不会出现在 top-K 中
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
from store_vfs import insert_chunk


def _apply_conf_filter(score: float, chunk: dict) -> float:
    """镜像 retriever.py iter482 confidence filter 逻辑。"""
    _conf = float(chunk.get("confidence_score") or 0.7)
    if (_conf < 0.15 and chunk.get("chunk_type") != "design_constraint"):
        return 0.0
    return score


def test_cf1_very_low_confidence_score_zeroed():
    """CF1: confidence < 0.15 → score 归零。"""
    chunk = {"confidence_score": 0.10, "chunk_type": "decision"}
    result = _apply_conf_filter(0.8, chunk)
    assert result == 0.0, f"CF1: conf=0.10 应归零, got {result}"
    print(f"  CF1 PASS: confidence=0.10 → score=0.0 (filtered)")


def test_cf2_normal_confidence_not_filtered():
    """CF2: confidence >= 0.15 → score 不变。"""
    for conf in [0.15, 0.50, 0.70, 0.90]:
        chunk = {"confidence_score": conf, "chunk_type": "decision"}
        base_score = 0.75
        result = _apply_conf_filter(base_score, chunk)
        assert result == base_score, (
            f"CF2: conf={conf} 不应被过滤, got {result} != {base_score}"
        )
    print(f"  CF2 PASS: confidence ∈ [0.15, 1.0] → score unchanged")


def test_cf3_design_constraint_exempt():
    """CF3: design_constraint 豁免，即使 confidence < 0.15 也保留分数。"""
    chunk = {"confidence_score": 0.05, "chunk_type": "design_constraint"}
    base_score = 0.6
    result = _apply_conf_filter(base_score, chunk)
    assert result == base_score, (
        f"CF3: design_constraint 应豁免过滤，got {result} != {base_score}"
    )
    print(f"  CF3 PASS: design_constraint exempt from confidence filter")


def test_cf4_boundary_015_not_filtered():
    """CF4: confidence=0.15（边界值）→ 不触发过滤（<0.15 才触发）。"""
    chunk = {"confidence_score": 0.15, "chunk_type": "task_state"}
    base_score = 0.5
    result = _apply_conf_filter(base_score, chunk)
    assert result == base_score, (
        f"CF4: confidence=0.15 不应被过滤（边界：< 0.15 才触发），got {result}"
    )
    print(f"  CF4 PASS: confidence=0.15 boundary → not filtered")


def test_cf5_014_vs_015_boundary():
    """CF5: confidence=0.14 被过滤，confidence=0.15 正常通过。"""
    chunk_14 = {"confidence_score": 0.14, "chunk_type": "decision"}
    chunk_15 = {"confidence_score": 0.15, "chunk_type": "decision"}
    base_score = 0.7

    score_14 = _apply_conf_filter(base_score, chunk_14)
    score_15 = _apply_conf_filter(base_score, chunk_15)

    assert score_14 == 0.0, f"CF5: conf=0.14 应归零, got {score_14}"
    assert score_15 == base_score, f"CF5: conf=0.15 应正常, got {score_15}"
    print(f"  CF5 PASS: 0.14 → 0.0 (filtered), 0.15 → {score_15} (pass)")


def test_cf6_integration_insert_and_filter():
    """CF6: 插入极低 confidence chunk，模拟检索时被过滤不出现。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"cf6_{uuid.uuid4().hex[:8]}"
    cid_low = f"cf6low_{uuid.uuid4().hex[:6]}"
    cid_ok = f"cf6ok_{uuid.uuid4().hex[:6]}"
    now = datetime.now(timezone.utc).isoformat()

    # 直接写入两个 chunk，一个极低 confidence（模拟经过大量衰减的场景）
    for cid, conf, chunk_type in [
        (cid_low, 0.10, "decision"),
        (cid_ok, 0.70, "decision"),
    ]:
        conn.execute("""
            INSERT OR REPLACE INTO memory_chunks
            (id, project, source_session, chunk_type, summary, content,
             importance, stability, retrievability, info_class, tags,
             access_count, oom_adj, created_at, updated_at, last_accessed,
             feishu_url, lru_gen, raw_snippet, encode_context, session_type_history,
             confidence_score)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (cid, proj, "test", chunk_type, "cf6 test chunk", "cf6 test chunk",
              0.6, 2.0, 0.5, "episodic", json.dumps([]),
              5, 0, now, now, now, None, 0, "cf6 test", "{}", "", conf))
    conn.commit()

    # 验证 DB 中两个 chunk 都存在
    rows = conn.execute(
        "SELECT id, confidence_score FROM memory_chunks WHERE project=?", (proj,)
    ).fetchall()
    assert len(rows) == 2, f"CF6: 应插入 2 个 chunk，got {len(rows)}"

    # 模拟 retriever 过滤：conf < 0.15 的 chunk 分数归零
    base_score = 0.8
    scores = {}
    for cid, conf in rows:
        chunk = {"confidence_score": conf, "chunk_type": "decision"}
        scores[cid] = _apply_conf_filter(base_score, chunk)

    assert scores[cid_low] == 0.0, (
        f"CF6: conf=0.10 chunk 应被过滤（score=0）, got {scores[cid_low]}"
    )
    assert scores[cid_ok] == base_score, (
        f"CF6: conf=0.70 chunk 应正常, got {scores[cid_ok]}"
    )

    conn.close()
    print(f"  CF6 PASS: low-conf chunk filtered (score=0), normal chunk retained")


if __name__ == "__main__":
    print("confidence_score 阈值过滤测试（iter482）")
    print("=" * 60)

    tests = [
        test_cf1_very_low_confidence_score_zeroed,
        test_cf2_normal_confidence_not_filtered,
        test_cf3_design_constraint_exempt,
        test_cf4_boundary_015_not_filtered,
        test_cf5_014_vs_015_boundary,
        test_cf6_integration_insert_and_filter,
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
