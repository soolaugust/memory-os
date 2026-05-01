#!/usr/bin/env python3
"""
test_stability_warmstart.py — stability warm-start 测试（iter479）

覆盖：
  SW1: importance >= 0.5 → 孤立 chunk 的 stability 基础值为 2.0（不受 schema 干扰）
  SW2: importance < 0.5 → 孤立 chunk stability 基础值为 1.0
  SW3: 显式传入 stability=5.0 时不被 warm-start 覆盖（基础值为 5.0）
  SW4: importance 恰好等于 0.5（边界值）→ warm-start 触发，稳定性高于 importance=0.49

测试策略：
  - 每个 chunk 使用独立的项目（避免 schema scaffolding 的跨 chunk 依赖影响）
  - DOP=0 的内容（无因果/结构/对比词）→ final_stability ≈ base_stability
  - 用已知 DOP=0 的简单内容，验证 final_stability 是否在预期范围内
"""
import sys
import uuid
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import tmpfs  # noqa

from store import open_db, ensure_schema
from store_vfs import insert_chunk


def _make_chunk(cid, project, importance, stability=1.0, explicit_stability=False):
    """创建测试 chunk，DOP=0（简单内容无深度加工特征）。"""
    now = datetime.now(timezone.utc).isoformat()
    d = {
        "id": cid,
        "project": project,
        "source_session": "test",
        "chunk_type": "decision",
        "info_class": "episodic",
        # DOP=0 的内容：无因果/结构/对比/精细阐述词
        "content": "abc def ghi jkl",
        "summary": "abc def ghi jkl",
        "tags": [],
        "importance": importance,
        "retrievability": 0.5,
        "last_accessed": now,
        "access_count": 0,
        "oom_adj": 0,
        "lru_gen": 0,
        "created_at": now,
        "updated_at": now,
    }
    if explicit_stability:
        d["stability"] = stability
    return d


def _get_stability(conn, cid):
    row = conn.execute(
        "SELECT stability FROM memory_chunks WHERE id=?", (cid,)
    ).fetchone()
    return float(row[0]) if row and row[0] is not None else None


def test_sw1_high_importance_warm_start_base():
    """SW1: importance=0.75，孤立项目（无 schema 干扰），stability ≈ 2.0。"""
    conn = open_db()
    ensure_schema(conn)
    # 孤立项目：每个 chunk 单独一个 project，避免 schema scaffolding 跨 chunk 干扰
    proj = f"sw1_{uuid.uuid4().hex[:12]}"
    cid = f"sw1c_{uuid.uuid4().hex[:10]}"

    d = _make_chunk(cid, proj, importance=0.75)
    insert_chunk(conn, d)
    conn.commit()

    stab = _get_stability(conn, cid)
    # DOP=0 时：final_stability = base_stability（schema bonus ≈ 0 in empty project）
    # warm-start: base = 2.0 → expect ~2.0
    assert stab >= 1.8, (
        f"SW1: importance=0.75 warm-start → stability 应 >= 1.8 (≈2.0), got {stab:.4f}"
    )
    assert stab < 3.0, (
        f"SW1: stability 不应远超 2.0, got {stab:.4f}"
    )

    conn.close()
    print(f"  SW1 PASS: importance=0.75 → stability={stab:.4f} (warm-start base=2.0)")


def test_sw2_low_importance_no_warm_start():
    """SW2: importance=0.30，孤立项目，stability ≈ 1.0（无 warm-start）。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"sw2_{uuid.uuid4().hex[:12]}"
    cid = f"sw2c_{uuid.uuid4().hex[:10]}"

    d = _make_chunk(cid, proj, importance=0.30)
    insert_chunk(conn, d)
    conn.commit()

    stab = _get_stability(conn, cid)
    # DOP=0, no schema bonus → stability ≈ 1.0
    assert stab >= 0.8, (
        f"SW2: importance=0.30 → stability 应 >= 0.8, got {stab:.4f}"
    )
    assert stab < 1.8, (
        f"SW2: importance=0.30 不触发 warm-start → stability 应 < 1.8, got {stab:.4f}"
    )

    conn.close()
    print(f"  SW2 PASS: importance=0.30 → stability={stab:.4f} (no warm-start)")


def test_sw3_explicit_stability_preserved():
    """SW3: 显式传入 stability=5.0，不被 warm-start 覆盖，最终 stability > 2.0。"""
    conn = open_db()
    ensure_schema(conn)
    proj_explicit = f"sw3e_{uuid.uuid4().hex[:10]}"
    proj_warmstart = f"sw3w_{uuid.uuid4().hex[:10]}"
    cid_e = f"sw3ec_{uuid.uuid4().hex[:8]}"
    cid_w = f"sw3wc_{uuid.uuid4().hex[:8]}"

    # 显式 stability=5.0 → DOP base=5.0
    d_explicit = _make_chunk(cid_e, proj_explicit, importance=0.80,
                             stability=5.0, explicit_stability=True)
    insert_chunk(conn, d_explicit)

    # warm-start base=2.0
    d_warmstart = _make_chunk(cid_w, proj_warmstart, importance=0.80)
    insert_chunk(conn, d_warmstart)
    conn.commit()

    stab_e = _get_stability(conn, cid_e)
    stab_w = _get_stability(conn, cid_w)

    # 显式 5.0 base 应明显高于 warm-start 2.0 base
    assert stab_e > stab_w, (
        f"SW3: explicit stability=5.0({stab_e:.3f}) 应 > warm-start 2.0({stab_w:.3f})"
    )
    assert stab_e >= 4.5, (
        f"SW3: explicit stability=5.0 基础，最终值应 >= 4.5, got {stab_e:.3f}"
    )

    conn.close()
    print(f"  SW3 PASS: explicit={stab_e:.3f} > warm-start={stab_w:.3f}")


def test_sw4_boundary_importance_05_vs_049():
    """SW4: importance=0.50（触发）vs 0.49（不触发），孤立项目对比。"""
    conn = open_db()
    ensure_schema(conn)
    proj_above = f"sw4a_{uuid.uuid4().hex[:10]}"
    proj_below = f"sw4b_{uuid.uuid4().hex[:10]}"
    cid_above = f"sw4ac_{uuid.uuid4().hex[:8]}"
    cid_below = f"sw4bc_{uuid.uuid4().hex[:8]}"

    d_above = _make_chunk(cid_above, proj_above, importance=0.50)  # 触发
    d_below = _make_chunk(cid_below, proj_below, importance=0.49)  # 不触发

    insert_chunk(conn, d_above)
    insert_chunk(conn, d_below)
    conn.commit()

    stab_above = _get_stability(conn, cid_above)
    stab_below = _get_stability(conn, cid_below)

    # 0.50 应触发 warm-start(2.0)，0.49 不触发(1.0)
    assert stab_above > stab_below, (
        f"SW4: 0.50({stab_above:.3f}) 应 > 0.49({stab_below:.3f})"
    )
    # 差值应接近 1.0（2.0 - 1.0）
    assert stab_above - stab_below >= 0.8, (
        f"SW4: warm-start 差距应 >= 0.8, got {stab_above - stab_below:.3f}"
    )

    conn.close()
    print(f"  SW4 PASS: 0.50 stability={stab_above:.3f} > 0.49 stability={stab_below:.3f} "
          f"(diff={stab_above - stab_below:.3f})")


if __name__ == "__main__":
    print("stability warm-start 测试（iter479）")
    print("=" * 60)

    tests = [
        test_sw1_high_importance_warm_start_base,
        test_sw2_low_importance_no_warm_start,
        test_sw3_explicit_stability_preserved,
        test_sw4_boundary_importance_05_vs_049,
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
