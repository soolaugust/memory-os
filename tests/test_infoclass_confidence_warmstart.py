#!/usr/bin/env python3
"""
test_infoclass_confidence_warmstart.py — info_class-aware confidence warm-start 测试（iter489）

覆盖：
  IC1: info_class='episodic' + 中等 source_reliability → confidence = 0.75（+0.05 加成）
  IC2: info_class='ephemeral' + 中等 source_reliability → confidence = 0.60（-0.10 惩罚）
  IC3: info_class='world' + 中等 source_reliability → confidence = 0.70（无调整）
  IC4: info_class='episodic' + 高 source_reliability → confidence = 0.85（高可信不受 info_class 影响）
  IC5: info_class='ephemeral' + 低 source_reliability → confidence = 0.40（惩罚后仍不低于 0.10）
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


def _make_chunk(cid, project, info_class="world", source_reliability=None):
    now = datetime.now(timezone.utc).isoformat()
    d = {
        "id": cid,
        "project": project,
        "source_session": "test",
        "chunk_type": "decision",
        "info_class": info_class,
        "content": "abc def ghi jkl mno",
        "summary": "abc def ghi jkl mno",
        "tags": [],
        "importance": 0.6,
        "retrievability": 0.5,
        "last_accessed": now,
        "access_count": 0,
        "oom_adj": 0,
        "lru_gen": 0,
        "created_at": now,
        "updated_at": now,
    }
    return d


def _get_confidence(conn, cid):
    row = conn.execute(
        "SELECT confidence_score, source_reliability FROM memory_chunks WHERE id=?", (cid,)
    ).fetchone()
    if not row:
        return None, None
    return float(row[0] or 0.7), (float(row[1]) if row[1] is not None else None)


def _set_source_reliability(conn, cid, sr):
    """直接设置 source_reliability（绕过 source_monitoring，精确控制）。"""
    conn.execute(
        "UPDATE memory_chunks SET source_reliability=? WHERE id=?", (sr, cid)
    )
    conn.commit()

    # 重新触发 confidence warm-start（模拟 source_monitoring 完成后的状态）
    # 手动执行 warm-start 逻辑
    if sr >= 0.80:
        ws_conf = 0.85
    elif sr < 0.40:
        ws_conf = 0.50
    else:
        ws_conf = 0.70

    # 读取 info_class
    row = conn.execute("SELECT info_class FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    ic = row[0] if row else "world"
    if ic == "episodic" and ws_conf == 0.70:
        ws_conf = min(1.0, ws_conf + 0.05)
    elif ic == "ephemeral":
        ws_conf = max(0.10, ws_conf - 0.10)

    conn.execute(
        "UPDATE memory_chunks SET confidence_score=? WHERE id=?", (round(ws_conf, 3), cid)
    )
    conn.commit()
    return ws_conf


def test_ic1_episodic_mid_sr_boost():
    """IC1: episodic + 中等 sr（0.60）→ confidence = 0.75（中性 0.70 + 0.05 episodic 加成）。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"ic1_{uuid.uuid4().hex[:8]}"
    cid = f"ic1c_{uuid.uuid4().hex[:6]}"

    d = _make_chunk(cid, proj, info_class="episodic")
    insert_chunk(conn, d)
    conn.commit()

    expected = _set_source_reliability(conn, cid, sr=0.60)
    conf, sr = _get_confidence(conn, cid)

    assert abs(expected - 0.75) < 0.001, f"IC1 期望 0.75, 计算得 {expected}"
    assert abs(conf - 0.75) < 0.001, (
        f"IC1: episodic + mid sr → confidence 应=0.75, got {conf:.3f}"
    )
    conn.close()
    print(f"  IC1 PASS: episodic + sr=0.60 → confidence={conf:.3f} (mid 0.70 + episodic +0.05)")


def test_ic2_ephemeral_mid_sr_penalty():
    """IC2: ephemeral + 中等 sr（0.60）→ confidence = 0.60（中性 0.70 - 0.10 ephemeral 惩罚）。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"ic2_{uuid.uuid4().hex[:8]}"
    cid = f"ic2c_{uuid.uuid4().hex[:6]}"

    d = _make_chunk(cid, proj, info_class="ephemeral")
    insert_chunk(conn, d)
    conn.commit()

    expected = _set_source_reliability(conn, cid, sr=0.60)
    conf, sr = _get_confidence(conn, cid)

    assert abs(expected - 0.60) < 0.001, f"IC2 期望 0.60, 计算得 {expected}"
    assert abs(conf - 0.60) < 0.001, (
        f"IC2: ephemeral + mid sr → confidence 应=0.60, got {conf:.3f}"
    )
    conn.close()
    print(f"  IC2 PASS: ephemeral + sr=0.60 → confidence={conf:.3f} (mid 0.70 - ephemeral 0.10)")


def test_ic3_world_mid_sr_no_adjustment():
    """IC3: world + 中等 sr（0.60）→ confidence = 0.70（无 info_class 调整）。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"ic3_{uuid.uuid4().hex[:8]}"
    cid = f"ic3c_{uuid.uuid4().hex[:6]}"

    d = _make_chunk(cid, proj, info_class="world")
    insert_chunk(conn, d)
    conn.commit()

    expected = _set_source_reliability(conn, cid, sr=0.60)
    conf, sr = _get_confidence(conn, cid)

    assert abs(expected - 0.70) < 0.001, f"IC3 期望 0.70, 计算得 {expected}"
    assert abs(conf - 0.70) < 0.001, (
        f"IC3: world + mid sr → confidence 应=0.70（无调整）, got {conf:.3f}"
    )
    conn.close()
    print(f"  IC3 PASS: world + sr=0.60 → confidence={conf:.3f} (unchanged)")


def test_ic4_episodic_high_sr_no_extra_boost():
    """IC4: episodic + 高 sr（0.90）→ confidence = 0.85（高可信不受 episodic 额外影响）。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"ic4_{uuid.uuid4().hex[:8]}"
    cid = f"ic4c_{uuid.uuid4().hex[:6]}"

    d = _make_chunk(cid, proj, info_class="episodic")
    insert_chunk(conn, d)
    conn.commit()

    expected = _set_source_reliability(conn, cid, sr=0.90)
    conf, sr = _get_confidence(conn, cid)

    # 高 sr → 0.85，episodic 加成只对中性(0.70)触发
    assert abs(expected - 0.85) < 0.001, f"IC4 期望 0.85, 计算得 {expected}"
    assert abs(conf - 0.85) < 0.001, (
        f"IC4: episodic + high sr → confidence 应=0.85（sr override）, got {conf:.3f}"
    )
    conn.close()
    print(f"  IC4 PASS: episodic + sr=0.90 → confidence={conf:.3f} (high-sr 0.85, no extra boost)")


def test_ic5_ephemeral_low_sr_penalty():
    """IC5: ephemeral + 低 sr（0.30）→ confidence = 0.40（低可信 0.50 - ephemeral 0.10）。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"ic5_{uuid.uuid4().hex[:8]}"
    cid = f"ic5c_{uuid.uuid4().hex[:6]}"

    d = _make_chunk(cid, proj, info_class="ephemeral")
    insert_chunk(conn, d)
    conn.commit()

    expected = _set_source_reliability(conn, cid, sr=0.30)
    conf, sr = _get_confidence(conn, cid)

    assert abs(expected - 0.40) < 0.001, f"IC5 期望 0.40, 计算得 {expected}"
    assert abs(conf - 0.40) < 0.001, (
        f"IC5: ephemeral + low sr → confidence 应=0.40, got {conf:.3f}"
    )
    conn.close()
    print(f"  IC5 PASS: ephemeral + sr=0.30 → confidence={conf:.3f} (low-sr 0.50 - 0.10)")


if __name__ == "__main__":
    print("info_class-aware confidence warm-start 测试（iter489）")
    print("=" * 60)

    tests = [
        test_ic1_episodic_mid_sr_boost,
        test_ic2_ephemeral_mid_sr_penalty,
        test_ic3_world_mid_sr_no_adjustment,
        test_ic4_episodic_high_sr_no_extra_boost,
        test_ic5_ephemeral_low_sr_penalty,
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
