#!/usr/bin/env python3
"""
test_ebbinghaus_decay.py — Ebbinghaus Time-Decay importance 测试

覆盖：
  EB1: 长期未访问 chunk → importance 下降（e^(-Δt/stability) 衰减）
  EB2: 近期访问 chunk（< 1天）→ 不衰减
  EB3: stability 高的 chunk 衰减慢（stability=10 vs stability=1）
  EB4: oom_adj < 0（pinned）→ 跳过衰减
  EB5: SKIP_CITATION_TYPES（task_state）→ 跳过衰减
  EB6: importance 已在 0.05 下界 → 不再衰减
"""
import sys
import json
import math
import uuid
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import tmpfs  # noqa

from store import open_db, ensure_schema


def _insert_chunk(conn, cid, project, summary, importance=0.6, stability=1.0,
                  chunk_type="decision", last_accessed_delta_days=0, oom_adj=0):
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
          1, oom_adj, now_iso, now_iso, la, None, 0, summary[:500], "{}", ""))
    conn.commit()


def test_eb1_long_absent_chunk_decays():
    """EB1: 30天未访问 + stability=1.0 → importance 大幅下降。"""
    conn = open_db()
    ensure_schema(conn)

    from store_mm import apply_ebbinghaus_decay

    proj = f"eb1_{uuid.uuid4().hex[:6]}"
    cid = f"eb1c_{uuid.uuid4().hex[:10]}"
    initial_imp = 0.80
    stability = 1.0
    delta_days = 30.0

    _insert_chunk(conn, cid, proj, "Linux kernel memory management", importance=initial_imp,
                  stability=stability, last_accessed_delta_days=delta_days)

    result = apply_ebbinghaus_decay(conn, proj)
    conn.commit()

    new_imp = conn.execute(
        "SELECT importance FROM memory_chunks WHERE id=?", (cid,)
    ).fetchone()[0]

    expected = max(0.05, initial_imp * math.exp(-delta_days / stability))
    assert new_imp < initial_imp, f"EB1: importance 应衰减, {initial_imp:.3f}→{new_imp:.3f}"
    assert abs(new_imp - expected) < 0.01, (
        f"EB1: 期望衰减至 {expected:.4f}, got {new_imp:.4f}"
    )
    assert result["decayed"] >= 1, f"EB1: decayed 应 >= 1, got {result}"

    conn.execute("DELETE FROM memory_chunks WHERE project=?", (proj,))
    conn.commit()
    conn.close()
    print(f"  EB1 PASS: {initial_imp:.3f}→{new_imp:.3f} (expected~{expected:.3f}), "
          f"decay_factor={math.exp(-delta_days/stability):.3f}")


def test_eb2_recent_chunk_not_decayed():
    """EB2: 近期访问 chunk（0.5天前）→ 不衰减（间隔保护）。"""
    conn = open_db()
    ensure_schema(conn)

    from store_mm import apply_ebbinghaus_decay

    proj = f"eb2_{uuid.uuid4().hex[:6]}"
    cid = f"eb2c_{uuid.uuid4().hex[:10]}"
    initial_imp = 0.70

    _insert_chunk(conn, cid, proj, "recent topic", importance=initial_imp,
                  stability=1.0, last_accessed_delta_days=0.4)  # 9.6小时前

    result = apply_ebbinghaus_decay(conn, proj)
    conn.commit()

    new_imp = conn.execute(
        "SELECT importance FROM memory_chunks WHERE id=?", (cid,)
    ).fetchone()[0]

    assert abs(new_imp - initial_imp) < 0.005, (
        f"EB2: 近期访问 chunk 不应衰减, {initial_imp:.3f}→{new_imp:.3f}"
    )

    conn.execute("DELETE FROM memory_chunks WHERE project=?", (proj,))
    conn.commit()
    conn.close()
    print(f"  EB2 PASS: recent chunk not decayed, imp={new_imp:.3f}")


def test_eb3_high_stability_decays_slower():
    """EB3: stability=10 的 chunk 衰减比 stability=1 的慢。"""
    conn = open_db()
    ensure_schema(conn)

    from store_mm import apply_ebbinghaus_decay

    proj = f"eb3_{uuid.uuid4().hex[:6]}"
    cid_slow = f"eb3s_{uuid.uuid4().hex[:10]}"
    cid_fast = f"eb3f_{uuid.uuid4().hex[:10]}"
    initial_imp = 0.80
    delta_days = 7.0

    _insert_chunk(conn, cid_slow, proj, "stable chunk summary", importance=initial_imp,
                  stability=10.0, last_accessed_delta_days=delta_days)
    _insert_chunk(conn, cid_fast, proj, "unstable chunk summary", importance=initial_imp,
                  stability=1.0, last_accessed_delta_days=delta_days)

    apply_ebbinghaus_decay(conn, proj)
    conn.commit()

    imp_slow = conn.execute("SELECT importance FROM memory_chunks WHERE id=?",
                            (cid_slow,)).fetchone()[0]
    imp_fast = conn.execute("SELECT importance FROM memory_chunks WHERE id=?",
                            (cid_fast,)).fetchone()[0]

    assert imp_slow > imp_fast, (
        f"EB3: high stability 衰减应更慢, slow={imp_slow:.3f} fast={imp_fast:.3f}"
    )

    # 验证数学正确性
    expected_slow = max(0.05, initial_imp * math.exp(-delta_days / 10.0))
    expected_fast = max(0.05, initial_imp * math.exp(-delta_days / 1.0))
    assert abs(imp_slow - expected_slow) < 0.01, (
        f"EB3: slow 期望 {expected_slow:.4f}, got {imp_slow:.4f}"
    )
    assert abs(imp_fast - expected_fast) < 0.01, (
        f"EB3: fast 期望 {expected_fast:.4f}, got {imp_fast:.4f}"
    )

    conn.execute("DELETE FROM memory_chunks WHERE project=?", (proj,))
    conn.commit()
    conn.close()
    print(f"  EB3 PASS: slow_stab=10 imp={imp_slow:.3f}, fast_stab=1 imp={imp_fast:.3f}")


def test_eb4_pinned_chunk_not_decayed():
    """EB4: oom_adj < 0（pinned）→ 跳过衰减。"""
    conn = open_db()
    ensure_schema(conn)

    from store_mm import apply_ebbinghaus_decay

    proj = f"eb4_{uuid.uuid4().hex[:6]}"
    cid = f"eb4c_{uuid.uuid4().hex[:10]}"
    initial_imp = 0.80

    _insert_chunk(conn, cid, proj, "pinned chunk", importance=initial_imp,
                  stability=1.0, last_accessed_delta_days=30, oom_adj=-10)

    apply_ebbinghaus_decay(conn, proj)
    conn.commit()

    new_imp = conn.execute(
        "SELECT importance FROM memory_chunks WHERE id=?", (cid,)
    ).fetchone()[0]

    assert abs(new_imp - initial_imp) < 0.005, (
        f"EB4: pinned chunk 不应衰减, {initial_imp:.3f}→{new_imp:.3f}"
    )

    conn.execute("DELETE FROM memory_chunks WHERE project=?", (proj,))
    conn.commit()
    conn.close()
    print(f"  EB4 PASS: pinned chunk not decayed, imp={new_imp:.3f}")


def test_eb5_skip_citation_types():
    """EB5: task_state chunk → 跳过衰减（非知识类 chunk 不遗忘）。"""
    conn = open_db()
    ensure_schema(conn)

    from store_mm import apply_ebbinghaus_decay

    proj = f"eb5_{uuid.uuid4().hex[:6]}"
    cid = f"eb5c_{uuid.uuid4().hex[:10]}"
    initial_imp = 0.80

    _insert_chunk(conn, cid, proj, "正在完成任务 A", importance=initial_imp,
                  stability=1.0, last_accessed_delta_days=30, chunk_type="task_state")

    apply_ebbinghaus_decay(conn, proj)
    conn.commit()

    new_imp = conn.execute(
        "SELECT importance FROM memory_chunks WHERE id=?", (cid,)
    ).fetchone()[0]

    assert abs(new_imp - initial_imp) < 0.005, (
        f"EB5: task_state 不应衰减, {initial_imp:.3f}→{new_imp:.3f}"
    )

    conn.execute("DELETE FROM memory_chunks WHERE project=?", (proj,))
    conn.commit()
    conn.close()
    print(f"  EB5 PASS: task_state not decayed, imp={new_imp:.3f}")


def test_eb6_floor_protection():
    """EB6: importance 已在下界 0.05 → 不再衰减。"""
    conn = open_db()
    ensure_schema(conn)

    from store_mm import apply_ebbinghaus_decay

    proj = f"eb6_{uuid.uuid4().hex[:6]}"
    cid = f"eb6c_{uuid.uuid4().hex[:10]}"
    floor_imp = 0.05

    _insert_chunk(conn, cid, proj, "already at floor", importance=floor_imp,
                  stability=1.0, last_accessed_delta_days=30)

    result = apply_ebbinghaus_decay(conn, proj)
    conn.commit()

    new_imp = conn.execute(
        "SELECT importance FROM memory_chunks WHERE id=?", (cid,)
    ).fetchone()[0]

    assert new_imp >= floor_imp - 0.001, (
        f"EB6: importance 不应低于 0.05, got {new_imp:.4f}"
    )

    conn.execute("DELETE FROM memory_chunks WHERE project=?", (proj,))
    conn.commit()
    conn.close()
    print(f"  EB6 PASS: floor protection, imp={new_imp:.4f} >= 0.05")


def test_eb7_high_stability_floor_protection():
    """EB7: stability >= 5 → 下界为 0.10（高于全局 0.05），防止死锁衰减。"""
    conn = open_db()
    ensure_schema(conn)

    from store_mm import apply_ebbinghaus_decay

    proj = f"eb7_{uuid.uuid4().hex[:6]}"
    # 高稳定 chunk（stability=10，长期未访问）
    cid_high = f"eb7h_{uuid.uuid4().hex[:10]}"
    # 低稳定 chunk（stability=0.5，长期未访问）
    cid_low = f"eb7l_{uuid.uuid4().hex[:10]}"
    initial_imp = 0.10  # 刚好在两者下界之间（高 stab 下界=0.10，低 stab 下界=0.05）

    _insert_chunk(conn, cid_high, proj, "Highly stable kernel knowledge",
                  importance=initial_imp, stability=10.0,
                  last_accessed_delta_days=365)  # 365天未访问

    _insert_chunk(conn, cid_low, proj, "Unstable transient knowledge",
                  importance=initial_imp, stability=0.5,
                  last_accessed_delta_days=365)

    apply_ebbinghaus_decay(conn, proj)
    conn.commit()

    imp_high = conn.execute("SELECT importance FROM memory_chunks WHERE id=?",
                            (cid_high,)).fetchone()[0]
    imp_low = conn.execute("SELECT importance FROM memory_chunks WHERE id=?",
                           (cid_low,)).fetchone()[0]

    # 高稳定 chunk 不应低于 0.10（下界保护）
    assert imp_high >= 0.10 - 0.005, (
        f"EB7: high stability chunk 不应低于 0.10, got {imp_high:.4f}"
    )
    # 低稳定 chunk 应衰减到更低（接近 0.05）
    assert imp_low < 0.10, (
        f"EB7: low stability chunk 应衰减至 0.10 以下, got {imp_low:.4f}"
    )
    # 高稳定 chunk 的 importance 应 >= 低稳定 chunk（同样初始值）
    assert imp_high >= imp_low, (
        f"EB7: high stability floor 保护，high={imp_high:.4f} low={imp_low:.4f}"
    )

    conn.execute("DELETE FROM memory_chunks WHERE project=?", (proj,))
    conn.commit()
    conn.close()
    print(f"  EB7 PASS: high_stab imp={imp_high:.4f} (floor=0.10), "
          f"low_stab imp={imp_low:.4f} (floor=0.05)")


if __name__ == "__main__":
    print("Ebbinghaus Time-Decay 测试")
    print("=" * 60)

    tests = [
        test_eb1_long_absent_chunk_decays,
        test_eb2_recent_chunk_not_decayed,
        test_eb3_high_stability_decays_slower,
        test_eb4_pinned_chunk_not_decayed,
        test_eb5_skip_citation_types,
        test_eb6_floor_protection,
        test_eb7_high_stability_floor_protection,
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
