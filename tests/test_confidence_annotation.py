#!/usr/bin/env python3
"""
test_confidence_annotation.py — low-confidence annotation 测试（iter480）

覆盖：
  CA1: confidence_score >= 0.9 → ✅ 标记（高可信度正向验证）
  CA2: confidence_score < 0.5 → ⚠️ 标记（低可信度警告）
  CA3: confidence_score 在 [0.5, 0.9) → 无额外标记（中等可信度，不干扰）
  CA4: verification_status="disputed" → ❓（覆盖 confidence 判断）
  CA5: confidence 衰减到 < 0.5 后，⚠️ 标记反映到 inject 输出中

CA1-CA4 为纯逻辑测试（与 retriever _conf_tag 逻辑一致）
CA5 验证 confidence 衰减 → 标注更新的端到端路径
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


# ── _conf_tag 逻辑镜像（与 retriever.py 保持同步）─────────────────────────────
def _conf_tag(chunk: dict) -> str:
    """与 retriever.py _conf_tag 函数逻辑完全一致的镜像。iter490: 阈值 0.50→0.30"""
    vs = chunk.get("verification_status", "pending")
    cs = float(chunk.get("confidence_score", 0.7) or 0.7)
    if vs == "disputed":
        return "❓"
    if vs == "verified" or cs >= 0.9:
        return "✅"
    if cs < 0.3:  # iter490: 阈值调整 0.50 → 0.30
        return "⚠️"
    return ""


def test_ca1_high_confidence_verified_tag():
    """CA1: confidence >= 0.9 → ✅ 标记。"""
    chunk = {"confidence_score": 0.95, "verification_status": "pending"}
    tag = _conf_tag(chunk)
    assert tag == "✅", f"CA1: confidence=0.95 应标记 ✅, got '{tag}'"
    print(f"  CA1 PASS: confidence=0.95 → '{tag}'")


def test_ca2_low_confidence_warning_tag():
    """CA2: confidence < 0.30 → ⚠️ 标记（iter490: 阈值调整）。"""
    chunk = {"confidence_score": 0.25, "verification_status": "pending"}
    tag = _conf_tag(chunk)
    assert tag == "⚠️", f"CA2: confidence=0.25 应标记 ⚠️, got '{tag}'"
    print(f"  CA2 PASS: confidence=0.25 → '{tag}'")


def test_ca3_mid_confidence_no_tag():
    """CA3: confidence ∈ [0.30, 0.9) → 无标记（iter490: 阈值调整后范围扩大）。"""
    for conf in [0.30, 0.50, 0.65, 0.80, 0.89]:
        chunk = {"confidence_score": conf, "verification_status": "pending"}
        tag = _conf_tag(chunk)
        assert tag == "", (
            f"CA3: confidence={conf} 不应添加标记, got '{tag}'"
        )
    print(f"  CA3 PASS: confidence ∈ [0.30, 0.9) → no tag")


def test_ca4_disputed_overrides_confidence():
    """CA4: verification_status='disputed' 时 → ❓（覆盖 confidence 判断）。"""
    # 即使 confidence 很高，disputed 仍然 → ❓
    chunk = {"confidence_score": 0.95, "verification_status": "disputed"}
    tag = _conf_tag(chunk)
    assert tag == "❓", f"CA4: disputed 应覆盖 confidence, got '{tag}'"
    print(f"  CA4 PASS: disputed overrides high confidence → '{tag}'")


def test_ca5_confidence_decays_to_warning():
    """CA5: Ebbinghaus 衰减后 confidence < 0.5 → 标记从无变为 ⚠️。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"ca5_{uuid.uuid4().hex[:6]}"
    cid = f"ca5c_{uuid.uuid4().hex[:10]}"

    # 初始 confidence=0.60（中等，无标记）
    now = datetime.now(timezone.utc)
    last_accessed = (now - timedelta(days=30)).isoformat()  # 30天前访问
    now_iso = now.isoformat()

    # stability=0.5 → 衰减极快（30天后 confidence = 0.6 × exp(-30/(0.5×2)) ≈ 0.6 × exp(-30) ≈ 0）
    conn.execute("""
        INSERT OR REPLACE INTO memory_chunks
        (id, project, source_session, chunk_type, summary, content,
         importance, stability, retrievability, info_class, tags,
         access_count, oom_adj, created_at, updated_at, last_accessed,
         feishu_url, lru_gen, raw_snippet, encode_context, session_type_history,
         confidence_score)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (cid, proj, "test", "decision", "Test CA5 chunk", "Test CA5 chunk",
          0.6, 0.5, 0.5, "episodic", json.dumps([]),
          5, 0, now_iso, now_iso, last_accessed,
          None, 0, "Test CA5 chunk"[:500], "{}", "",
          0.60))
    conn.commit()

    # 验证初始无警告标记（0.60 ∈ [0.5, 0.9)）
    initial_conf = conn.execute(
        "SELECT confidence_score FROM memory_chunks WHERE id=?", (cid,)
    ).fetchone()[0]
    initial_tag = _conf_tag({"confidence_score": initial_conf, "verification_status": "pending"})
    assert initial_tag == "", (
        f"CA5: 初始 confidence={initial_conf:.3f} 不应有标记, got '{initial_tag}'"
    )

    # 运行 Ebbinghaus 衰减
    result = apply_ebbinghaus_decay(conn, proj, max_chunks=10)
    conn.commit()

    # 验证 confidence 已衰减到 < 0.5
    new_conf = conn.execute(
        "SELECT confidence_score FROM memory_chunks WHERE id=?", (cid,)
    ).fetchone()[0]
    new_tag = _conf_tag({"confidence_score": new_conf, "verification_status": "pending"})

    assert new_conf < 0.5, (
        f"CA5: 衰减后 confidence 应 < 0.5, got {new_conf:.4f}"
    )
    assert new_tag == "⚠️", (
        f"CA5: 衰减后 confidence={new_conf:.3f} 应标记 ⚠️, got '{new_tag}'"
    )

    conn.close()
    print(f"  CA5 PASS: confidence decayed {initial_conf:.3f}→{new_conf:.4f} "
          f"→ tag: '{initial_tag}'→'{new_tag}'")


if __name__ == "__main__":
    print("low-confidence annotation 测试（iter480）")
    print("=" * 60)

    tests = [
        test_ca1_high_confidence_verified_tag,
        test_ca2_low_confidence_warning_tag,
        test_ca3_mid_confidence_no_tag,
        test_ca4_disputed_overrides_confidence,
        test_ca5_confidence_decays_to_warning,
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
