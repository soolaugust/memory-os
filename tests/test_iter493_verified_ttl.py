#!/usr/bin/env python3
"""
test_iter493_verified_ttl.py — Verified Status TTL 过期机制测试（iter493）

覆盖：
  VT1: verified chunk 在 TTL 内访问 → 不过期（保持 verified）
  VT2: verified chunk 超过 TTL 未访问（low stability）→ 重置为 pending
  VT3: verified chunk 超过标准 TTL 但 stability>=5.0（high stability）→ 不过期（TTL 更长）
  VT4: verified chunk 超过 high_stability TTL → 过期，重置为 pending
  VT5: pending chunk 不受影响
  VT6: 过期后的 chunk 不再豁免 Ebbinghaus 衰减（端到端）
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
from store_vfs import expire_stale_verified, insert_chunk
from store_mm import apply_ebbinghaus_decay


def _insert_verified_chunk(conn, cid, project, stability=1.0,
                           last_accessed_days_ago=0, verified=True):
    now = datetime.now(timezone.utc)
    last_accessed = (now - timedelta(days=last_accessed_days_ago)).isoformat()
    now_iso = now.isoformat()
    vstatus = "verified" if verified else "pending"
    conn.execute("""
        INSERT OR REPLACE INTO memory_chunks
        (id, project, source_session, chunk_type, summary, content,
         importance, stability, retrievability, info_class, tags,
         access_count, oom_adj, created_at, updated_at, last_accessed,
         feishu_url, lru_gen, raw_snippet, encode_context, session_type_history,
         confidence_score, verification_status)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (cid, project, "test", "decision", f"vt test {cid[:6]}", f"vt test {cid[:6]}",
          0.8, stability, 0.5, "episodic", json.dumps([]),
          0, 0, now_iso, now_iso, last_accessed,
          None, 2, f"vt test {cid[:6]}", "{}", "",
          0.8, vstatus))
    conn.commit()


def _get_vstatus(conn, cid):
    row = conn.execute(
        "SELECT verification_status FROM memory_chunks WHERE id=?", (cid,)
    ).fetchone()
    return row[0] if row else None


def test_vt1_within_ttl_not_expired():
    """VT1: verified chunk 在 TTL 内（last_accessed=1天前, TTL=30天）→ 保持 verified。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"vt1_{uuid.uuid4().hex[:8]}"
    cid = f"vt1c_{uuid.uuid4().hex[:6]}"

    # stability=1.0（low）, last_accessed=1天前，TTL默认30天 → 未过期
    _insert_verified_chunk(conn, cid, proj, stability=1.0, last_accessed_days_ago=1)

    result = expire_stale_verified(conn, proj)
    conn.commit()

    vstatus = _get_vstatus(conn, cid)
    assert vstatus == "verified", (
        f"VT1: last_accessed=1d < TTL=30d → 应保持 verified, got '{vstatus}'"
    )
    assert result["expired"] == 0, f"VT1: 不应有 chunk 过期, got expired={result['expired']}"
    conn.close()
    print(f"  VT1 PASS: within TTL (1d < 30d) → still verified, expired={result['expired']}")


def test_vt2_low_stability_exceeds_ttl():
    """VT2: verified chunk 超过 TTL（last_accessed=35天前, stability=1.0）→ 重置为 pending。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"vt2_{uuid.uuid4().hex[:8]}"
    cid = f"vt2c_{uuid.uuid4().hex[:6]}"

    # stability=1.0（low）, last_accessed=35天前，TTL=30天 → 过期
    _insert_verified_chunk(conn, cid, proj, stability=1.0, last_accessed_days_ago=35)

    result = expire_stale_verified(conn, proj)
    conn.commit()

    vstatus = _get_vstatus(conn, cid)
    assert vstatus == "pending", (
        f"VT2: last_accessed=35d > TTL=30d → 应重置为 pending, got '{vstatus}'"
    )
    assert result["expired"] >= 1, f"VT2: 应有 1 个 chunk 过期, got {result['expired']}"
    conn.close()
    print(f"  VT2 PASS: exceeded TTL (35d > 30d, stability=1.0) → reset to pending, expired={result['expired']}")


def test_vt3_high_stability_within_extended_ttl():
    """VT3: high stability（>=5.0）超过标准 TTL 但未超过 extended TTL → 保持 verified。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"vt3_{uuid.uuid4().hex[:8]}"
    cid = f"vt3c_{uuid.uuid4().hex[:6]}"

    # stability=6.0（high）, last_accessed=35天前, standard TTL=30d, extended TTL=90d → 未过期
    _insert_verified_chunk(conn, cid, proj, stability=6.0, last_accessed_days_ago=35)

    result = expire_stale_verified(conn, proj)
    conn.commit()

    vstatus = _get_vstatus(conn, cid)
    assert vstatus == "verified", (
        f"VT3: stability=6.0, last_accessed=35d → extended TTL=90d，应保持 verified, got '{vstatus}'"
    )
    assert result["expired"] == 0, f"VT3: 不应有 chunk 过期, got {result['expired']}"
    conn.close()
    print(f"  VT3 PASS: high-stability within extended TTL (35d < 90d) → still verified")


def test_vt4_high_stability_exceeds_extended_ttl():
    """VT4: high stability 超过 extended TTL（last_accessed=95天前）→ 重置为 pending。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"vt4_{uuid.uuid4().hex[:8]}"
    cid = f"vt4c_{uuid.uuid4().hex[:6]}"

    # stability=6.0（high）, last_accessed=95天前, extended TTL=90d → 过期
    _insert_verified_chunk(conn, cid, proj, stability=6.0, last_accessed_days_ago=95)

    result = expire_stale_verified(conn, proj)
    conn.commit()

    vstatus = _get_vstatus(conn, cid)
    assert vstatus == "pending", (
        f"VT4: stability=6.0, last_accessed=95d > extended TTL=90d → 应重置为 pending, got '{vstatus}'"
    )
    assert result["expired"] >= 1, f"VT4: 应有 1 个 chunk 过期, got {result['expired']}"
    conn.close()
    print(f"  VT4 PASS: exceeded extended TTL (95d > 90d, stability=6.0) → reset to pending")


def test_vt5_pending_chunk_unaffected():
    """VT5: pending chunk 不受 expire_stale_verified 影响。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"vt5_{uuid.uuid4().hex[:8]}"
    cid = f"vt5c_{uuid.uuid4().hex[:6]}"

    # pending chunk，100天前访问
    _insert_verified_chunk(conn, cid, proj, stability=1.0,
                           last_accessed_days_ago=100, verified=False)

    result = expire_stale_verified(conn, proj)
    conn.commit()

    vstatus = _get_vstatus(conn, cid)
    assert vstatus == "pending", (
        f"VT5: pending chunk 不应被 expire 函数修改, got '{vstatus}'"
    )
    assert result["total_verified"] == 0, f"VT5: 无 verified chunk，total_verified 应=0"
    conn.close()
    print(f"  VT5 PASS: pending chunk unaffected by expire_stale_verified")


def test_vt6_expired_verified_then_ebbinghaus_decays():
    """VT6: verified → pending（TTL 过期）→ 下一轮 Ebbinghaus 衰减有效。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"vt6_{uuid.uuid4().hex[:8]}"
    cid = f"vt6c_{uuid.uuid4().hex[:6]}"

    # verified chunk，超过 TTL
    _insert_verified_chunk(conn, cid, proj, stability=1.0, last_accessed_days_ago=35)

    initial_imp = conn.execute(
        "SELECT importance FROM memory_chunks WHERE id=?", (cid,)
    ).fetchone()[0]

    # Step 1: 过期 verified → pending
    expire_stale_verified(conn, proj)
    conn.commit()

    vstatus_after_expire = _get_vstatus(conn, cid)
    assert vstatus_after_expire == "pending", "VT6: 应已过期为 pending"

    # Step 2: Ebbinghaus 衰减（verified chunk 豁免，pending 不豁免）
    apply_ebbinghaus_decay(conn, proj, max_chunks=10)
    conn.commit()

    new_imp = conn.execute(
        "SELECT importance FROM memory_chunks WHERE id=?", (cid,)
    ).fetchone()[0]

    assert float(new_imp) < float(initial_imp), (
        f"VT6: 过期后的 chunk 应被 Ebbinghaus 衰减，"
        f"initial={initial_imp:.4f}, new={new_imp:.4f}"
    )
    conn.close()
    print(f"  VT6 PASS: verified→pending(TTL)→Ebbinghaus: {float(initial_imp):.4f}→{float(new_imp):.4f}")


if __name__ == "__main__":
    print("Verified Status TTL 过期机制测试（iter493）")
    print("=" * 60)

    tests = [
        test_vt1_within_ttl_not_expired,
        test_vt2_low_stability_exceeds_ttl,
        test_vt3_high_stability_within_extended_ttl,
        test_vt4_high_stability_exceeds_extended_ttl,
        test_vt5_pending_chunk_unaffected,
        test_vt6_expired_verified_then_ebbinghaus_decays,
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
