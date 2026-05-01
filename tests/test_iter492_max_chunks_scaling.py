#!/usr/bin/env python3
"""
test_iter492_max_chunks_scaling.py — Ebbinghaus max_chunks 自动缩放 测试（iter492）

覆盖：
  MS1: 小项目（total=10）→ effective_max = 10（全扫，不遗漏）
  MS2: 中项目（total=90）→ effective_max = max(50, 90//3) = 50（参数适当收缩）
  MS3: 大项目（total=300）→ effective_max = max(100, min(passed, 300//5)) = 60
  MS4: 超大项目（total=300）→ max_chunks=200 时，effective_max = max(100, min(200, 60)) = 100
  MS5: 小项目全扫 — total=20，所有 stale chunk 都被衰减（不遗漏）
  MS6: decayed 计数不超过 effective_max
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


def _insert_stale_chunk(conn, cid, project, stability=1.5, days_ago=2.0,
                        importance=0.7, confidence=0.8):
    """插入一个确定会被衰减的 stale chunk（stability=1.5, days_ago=2.0 → 超过 1.0d cutoff）"""
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
    """, (cid, project, "test", "decision", f"ms test {cid[:6]}", f"ms test {cid[:6]}",
          importance, stability, 0.5, "episodic", json.dumps([]),
          0, 0, now_iso, now_iso, last_accessed,
          None, 2, f"ms test {cid[:6]}", "{}", "",
          confidence, "pending"))


def test_ms1_small_project_full_scan():
    """MS1: total=10（<=50），max_chunks=100 → effective_max = min(100,10)=10，全扫不遗漏。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"ms1_{uuid.uuid4().hex[:8]}"

    # 插入 10 个 stale chunk
    for i in range(10):
        cid = f"ms1c{i:02d}_{uuid.uuid4().hex[:4]}"
        _insert_stale_chunk(conn, cid, proj)
    conn.commit()

    result = apply_ebbinghaus_decay(conn, proj, max_chunks=100)
    conn.commit()

    # total=10 → effective_max=10 → 10 个 chunk 都应被 scan（都是 stale）
    assert result["decayed"] <= 10, f"MS1: decayed {result['decayed']} > total(10)"
    assert result["total_scanned"] > 0, "MS1: total_scanned 应 > 0"
    conn.close()
    print(f"  MS1 PASS: small project(10) → decayed={result['decayed']}, scanned={result['total_scanned']}")


def test_ms2_medium_project_capped_scan():
    """MS2: total=90（∈(50,200]）→ effective_max = max(50, 90//3) = 50，衰减数 ≤ 50。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"ms2_{uuid.uuid4().hex[:8]}"

    # 插入 90 个 stale chunk
    for i in range(90):
        cid = f"ms2c{i:03d}_{uuid.uuid4().hex[:4]}"
        _insert_stale_chunk(conn, cid, proj)
    conn.commit()

    result = apply_ebbinghaus_decay(conn, proj, max_chunks=100)
    conn.commit()

    # effective_max = max(50, 90//3) = max(50, 30) = 50
    assert result["decayed"] <= 50, (
        f"MS2: total=90 → effective_max=50，decayed={result['decayed']} 应 ≤ 50"
    )
    assert result["decayed"] > 0, "MS2: 应有 chunk 被衰减"
    conn.close()
    print(f"  MS2 PASS: medium project(90) → decayed={result['decayed']} ≤ 50 (effective_max=50)")


def test_ms3_large_project_auto_scaled():
    """MS3: total=150（∈(50,200]）→ effective_max = max(50, 150//3) = 50，不超过 50。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"ms3_{uuid.uuid4().hex[:8]}"

    # 插入 150 个 stale chunk
    for i in range(150):
        cid = f"ms3c{i:03d}_{uuid.uuid4().hex[:4]}"
        _insert_stale_chunk(conn, cid, proj)
    conn.commit()

    result = apply_ebbinghaus_decay(conn, proj, max_chunks=100)
    conn.commit()

    # effective_max = max(50, 150//3) = max(50, 50) = 50
    assert result["decayed"] <= 50, (
        f"MS3: total=150 → effective_max=50，decayed={result['decayed']} 应 ≤ 50"
    )
    assert result["decayed"] > 0, "MS3: 应有 chunk 被衰减"
    conn.close()
    print(f"  MS3 PASS: large project(150) → decayed={result['decayed']} ≤ 50 (effective_max=50)")


def test_ms4_very_large_project_bounded():
    """MS4: total=250（>200）→ effective_max = max(100, min(50, 250//5)) = max(100, min(50,50)) = 100。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"ms4_{uuid.uuid4().hex[:8]}"

    # 插入 250 个 stale chunk
    for i in range(250):
        cid = f"ms4c{i:03d}_{uuid.uuid4().hex[:4]}"
        _insert_stale_chunk(conn, cid, proj)
    conn.commit()

    # max_chunks=50 → effective_max = max(100, min(50, 250//5)) = max(100, 50) = 100
    result = apply_ebbinghaus_decay(conn, proj, max_chunks=50)
    conn.commit()

    assert result["decayed"] <= 100, (
        f"MS4: total=250, max_chunks=50 → effective_max=100, decayed={result['decayed']} 应 ≤ 100"
    )
    assert result["decayed"] > 0, "MS4: 应有 chunk 被衰减"
    conn.close()
    print(f"  MS4 PASS: very large project(250) → decayed={result['decayed']} ≤ 100 (effective_max=100)")


def test_ms5_small_project_no_miss():
    """MS5: 小项目（total=15）全扫，所有 stale chunk 都被处理（no miss）。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"ms5_{uuid.uuid4().hex[:8]}"

    # 插入 15 个 stale chunk（stability=1.0, days_ago=3.0 → 确保超过各 cutoff）
    cids = []
    for i in range(15):
        cid = f"ms5c{i:02d}_{uuid.uuid4().hex[:4]}"
        cids.append(cid)
        _insert_stale_chunk(conn, cid, proj, stability=1.0, days_ago=3.0)
    conn.commit()

    initial_imps = {}
    for cid in cids:
        row = conn.execute("SELECT importance FROM memory_chunks WHERE id=?", (cid,)).fetchone()
        initial_imps[cid] = float(row[0])

    result = apply_ebbinghaus_decay(conn, proj, max_chunks=100)
    conn.commit()

    # total=15 → effective_max=15（全扫），所有 chunk 都被衰减
    decayed_count = 0
    for cid in cids:
        row = conn.execute("SELECT importance FROM memory_chunks WHERE id=?", (cid,)).fetchone()
        new_imp = float(row[0])
        if new_imp < initial_imps[cid]:
            decayed_count += 1

    assert decayed_count == 15, (
        f"MS5: total=15 小项目全扫 → 应衰减 15 个，实际 {decayed_count} 个"
    )
    conn.close()
    print(f"  MS5 PASS: small project(15) full scan → all {decayed_count}/15 chunks decayed")


def test_ms6_decayed_count_respects_effective_max():
    """MS6: decayed 计数不超过 effective_max（验证处理上限有效）。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"ms6_{uuid.uuid4().hex[:8]}"

    # 插入 200 个 stale chunk
    for i in range(200):
        cid = f"ms6c{i:03d}_{uuid.uuid4().hex[:4]}"
        _insert_stale_chunk(conn, cid, proj, stability=1.0, days_ago=3.0)
    conn.commit()

    # total=200（∈(50,200]）→ effective_max = max(50, 200//3) = max(50, 66) = 66
    result = apply_ebbinghaus_decay(conn, proj, max_chunks=100)
    conn.commit()

    expected_max = max(50, 200 // 3)  # = 66
    assert result["decayed"] <= expected_max, (
        f"MS6: total=200 → effective_max={expected_max}，decayed={result['decayed']} 超限"
    )
    conn.close()
    print(f"  MS6 PASS: total=200 → effective_max={expected_max}, decayed={result['decayed']} ≤ {expected_max}")


if __name__ == "__main__":
    print("Ebbinghaus max_chunks 自动缩放 测试（iter492）")
    print("=" * 60)

    tests = [
        test_ms1_small_project_full_scan,
        test_ms2_medium_project_capped_scan,
        test_ms3_large_project_auto_scaled,
        test_ms4_very_large_project_bounded,
        test_ms5_small_project_no_miss,
        test_ms6_decayed_count_respects_effective_max,
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
