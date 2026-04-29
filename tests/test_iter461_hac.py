"""
test_iter461_hac.py — iter461: Hebbian Co-Activation Consolidation 单元测试

覆盖：
  HA1: 共激活 >= hac_min_coact(2) 次的 chunk 对在 sleep 时 stability 加成
  HA2: 共激活次数 < hac_min_coact → 无 HAC 加成
  HA3: hac_enabled=False → 无任何加成
  HA4: importance < hac_min_importance(0.35) → 不参与 HAC
  HA5: 加成系数 = hac_boost_factor(1.05)，受 hac_max_boost(0.15) 保护
  HA6: stability 加成后不超过 365.0（cap 保护）
  HA7: record_coactivation 正确累积 coact_count
  HA8: update_accessed() 集成测试：多次共同访问后 sleep_consolidate 触发 HAC

认知科学依据：
  Hebb (1949) "The Organization of Behavior" — "Cells that fire together, wire together"
    海马 Hebbian 可塑性：两个神经元同时激活 → 突触连接增强（LTP）。
  Zeithamova et al. (2012) "Hippocampal and ventral medial prefrontal activation" —
    睡眠期间共激活记忆对通过 SWR replay 相互巩固（r=0.61）。

OS 类比：Linux THP (Transparent Huge Pages) promotion —
  khugepaged 将同一 2MB PMD 内频繁共同访问的 pages 透明合并为 huge page，
  降低 TLB miss 率（共激活 → 协同晋升到更稳定的存储层）。
"""
import sys
import sqlite3
import datetime
import unittest.mock as mock
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import tmpfs  # noqa

from store_vfs import (
    ensure_schema,
    record_coactivation,
    apply_hebbian_coactivation_consolidation,
    update_accessed,
    sleep_consolidate,
)
import config


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _utcnow():
    return datetime.datetime.now(datetime.timezone.utc)


def _insert_chunk(conn, cid, project="test", stability=5.0, importance=0.6,
                  chunk_type="decision"):
    now_iso = _utcnow().isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count,
            encode_context, session_type_history)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (cid, project, chunk_type, f"content {cid}", f"summary {cid}",
         importance, stability, now_iso, now_iso, 0.5,
         now_iso, 2, "kernel_mm", "")
    )
    conn.commit()


def _get_stability(conn, cid: str) -> float:
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


def _get_coact_count(conn, chunk_a: str, chunk_b: str, project: str) -> int:
    a, b = sorted([chunk_a, chunk_b])
    row = conn.execute(
        "SELECT coact_count FROM chunk_coactivation WHERE chunk_a=? AND chunk_b=? AND project=?",
        (a, b, project)
    ).fetchone()
    return int(row[0]) if row else 0


# ── HA1: 共激活 >= min_coact 次 → sleep 时 stability 加成 ─────────────────────────────

def test_ha1_coactivated_chunks_boosted_in_sleep(conn):
    """HA1: 共激活 >= hac_min_coact(2) 次的 chunk 对在 sleep_consolidate 时 stability 加成。"""
    min_coact = config.get("store_vfs.hac_min_coact")  # 2
    _insert_chunk(conn, "hac_a", stability=5.0, importance=0.6)
    _insert_chunk(conn, "hac_b", stability=5.0, importance=0.6)

    now_iso = _utcnow().isoformat()
    # 记录 min_coact 次共激活
    for _ in range(min_coact):
        record_coactivation(conn, ["hac_a", "hac_b"], "test", now_iso=now_iso)
    conn.commit()

    stab_before = _get_stability(conn, "hac_a")
    result = apply_hebbian_coactivation_consolidation(conn, "test")
    stab_after = _get_stability(conn, "hac_a")

    assert stab_after > stab_before, (
        f"HA1: 共激活 {min_coact} 次后 stability 应加成，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["hac_boosted"] >= 1, f"HA1: hac_boosted 应 >= 1，got {result}"


# ── HA2: 共激活次数不足 → 无加成 ──────────────────────────────────────────────────────

def test_ha2_insufficient_coact_no_boost(conn):
    """HA2: 共激活次数 < hac_min_coact(2) → 无 HAC 加成。"""
    _insert_chunk(conn, "hac_c", stability=5.0, importance=0.6)
    _insert_chunk(conn, "hac_d", stability=5.0, importance=0.6)

    now_iso = _utcnow().isoformat()
    record_coactivation(conn, ["hac_c", "hac_d"], "test", now_iso=now_iso)  # 只有 1 次
    conn.commit()

    stab_before = _get_stability(conn, "hac_c")
    result = apply_hebbian_coactivation_consolidation(conn, "test")
    stab_after = _get_stability(conn, "hac_c")

    assert abs(stab_after - stab_before) < 0.01, (
        f"HA2: 共激活不足时不应有 HAC 加成，before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── HA3: hac_enabled=False → 无加成 ─────────────────────────────────────────────────

def test_ha3_disabled_no_boost(conn):
    """HA3: hac_enabled=False → 无任何 HAC 加成。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.hac_enabled":
            return False
        return original_get(key, project=project)

    min_coact = config.get("store_vfs.hac_min_coact")
    _insert_chunk(conn, "hac_e", stability=5.0, importance=0.6)
    _insert_chunk(conn, "hac_f", stability=5.0, importance=0.6)

    now_iso = _utcnow().isoformat()
    for _ in range(min_coact):
        record_coactivation(conn, ["hac_e", "hac_f"], "test", now_iso=now_iso)
    conn.commit()

    stab_before = _get_stability(conn, "hac_e")
    with mock.patch.object(config, 'get', side_effect=patched_get):
        result = apply_hebbian_coactivation_consolidation(conn, "test")
    stab_after = _get_stability(conn, "hac_e")

    assert abs(stab_after - stab_before) < 0.01, (
        f"HA3: disabled 时不应有 HAC 加成，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["hac_boosted"] == 0, f"HA3: hac_boosted 应为 0，got {result}"


# ── HA4: importance 不足 → 不参与 HAC ────────────────────────────────────────────────

def test_ha4_low_importance_no_boost(conn):
    """HA4: importance < hac_min_importance(0.35) → 不参与 HAC 加成。"""
    min_coact = config.get("store_vfs.hac_min_coact")
    _insert_chunk(conn, "hac_low_g", stability=5.0, importance=0.10)  # < 0.35
    _insert_chunk(conn, "hac_low_h", stability=5.0, importance=0.10)

    now_iso = _utcnow().isoformat()
    for _ in range(min_coact):
        record_coactivation(conn, ["hac_low_g", "hac_low_h"], "test", now_iso=now_iso)
    conn.commit()

    stab_before = _get_stability(conn, "hac_low_g")
    apply_hebbian_coactivation_consolidation(conn, "test")
    stab_after = _get_stability(conn, "hac_low_g")

    assert abs(stab_after - stab_before) < 0.01, (
        f"HA4: 低 importance 时不应有 HAC 加成，before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── HA5: 加成系数 = hac_boost_factor(1.05)，受 hac_max_boost(0.15) 保护 ──────────────

def test_ha5_boost_factor_capped(conn):
    """HA5: hac_boost_factor=1.05，加成受 hac_max_boost(0.15) 上限保护。"""
    hac_boost_factor = config.get("store_vfs.hac_boost_factor")  # 1.05
    hac_max_boost = config.get("store_vfs.hac_max_boost")         # 0.15
    min_coact = config.get("store_vfs.hac_min_coact")
    base = 5.0

    _insert_chunk(conn, "hac_f5_a", stability=base, importance=0.6)
    _insert_chunk(conn, "hac_f5_b", stability=base, importance=0.6)

    now_iso = _utcnow().isoformat()
    for _ in range(min_coact):
        record_coactivation(conn, ["hac_f5_a", "hac_f5_b"], "test", now_iso=now_iso)
    conn.commit()

    apply_hebbian_coactivation_consolidation(conn, "test")
    stab_after = _get_stability(conn, "hac_f5_a")

    expected_max = min(365.0, base * (1.0 + hac_max_boost))
    assert stab_after <= expected_max + 0.001, (
        f"HA5: HAC 加成不应超过 max_boost={hac_max_boost}，"
        f"expected_max={expected_max:.4f} got={stab_after:.4f}"
    )
    assert stab_after > base, f"HA5: 应有 HAC 加成，base={base:.4f} got={stab_after:.4f}"


# ── HA6: stability 上限 365.0 ─────────────────────────────────────────────────────────

def test_ha6_stability_cap_365(conn):
    """HA6: HAC boost 后 stability 不超过 365.0。"""
    min_coact = config.get("store_vfs.hac_min_coact")
    _insert_chunk(conn, "hac_cap_a", stability=364.5, importance=0.8)
    _insert_chunk(conn, "hac_cap_b", stability=364.5, importance=0.8)

    now_iso = _utcnow().isoformat()
    for _ in range(min_coact):
        record_coactivation(conn, ["hac_cap_a", "hac_cap_b"], "test", now_iso=now_iso)
    conn.commit()

    apply_hebbian_coactivation_consolidation(conn, "test")
    stab_after = _get_stability(conn, "hac_cap_a")

    assert stab_after <= 365.0, f"HA6: stability 不应超过 365.0，got {stab_after}"


# ── HA7: record_coactivation 正确累积 coact_count ─────────────────────────────────────

def test_ha7_coact_count_accumulates(conn):
    """HA7: record_coactivation 多次调用后 coact_count 正确累加。"""
    _insert_chunk(conn, "hac_cnt_a", stability=5.0, importance=0.6)
    _insert_chunk(conn, "hac_cnt_b", stability=5.0, importance=0.6)

    now_iso = _utcnow().isoformat()
    for i in range(5):
        record_coactivation(conn, ["hac_cnt_a", "hac_cnt_b"], "test", now_iso=now_iso)
    conn.commit()

    count = _get_coact_count(conn, "hac_cnt_a", "hac_cnt_b", "test")
    assert count == 5, f"HA7: coact_count 应为 5，got {count}"


# ── HA8: update_accessed 集成测试 ─────────────────────────────────────────────────────

def test_ha8_update_accessed_integration(conn):
    """HA8: update_accessed 多次共同访问后 coact_count 递增，sleep 时触发 HAC。"""
    min_coact = config.get("store_vfs.hac_min_coact")  # 2
    _insert_chunk(conn, "hac_int_a", stability=5.0, importance=0.6)
    _insert_chunk(conn, "hac_int_b", stability=5.0, importance=0.6)

    # 通过 update_accessed 共同访问 min_coact 次
    for _ in range(min_coact):
        update_accessed(conn, ["hac_int_a", "hac_int_b"])
    conn.commit()

    count = _get_coact_count(conn, "hac_int_a", "hac_int_b", "test")
    assert count >= min_coact, (
        f"HA8: update_accessed 后 coact_count 应 >= {min_coact}，got {count}"
    )

    stab_before = _get_stability(conn, "hac_int_a")
    apply_hebbian_coactivation_consolidation(conn, "test")
    stab_after = _get_stability(conn, "hac_int_a")

    assert stab_after >= stab_before, (
        f"HA8: HAC 后 stability 不应降低，before={stab_before:.4f} after={stab_after:.4f}"
    )
