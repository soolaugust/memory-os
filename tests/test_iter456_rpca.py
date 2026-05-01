"""
test_iter456_rpca.py — iter456: Retrieval Practice vs. Restudy Consolidation Asymmetry 单元测试

覆盖：
  RP1: access_source='retrieval' + importance >= 0.40 → stability 增加（主动检索加成）
  RP2: access_source='restudy' + importance >= 0.40 → stability 轻微增加（被动重读小加成）
  RP3: retrieval 加成 > restudy 加成（主动检索效益更高，Roediger & Karpicke 2006 核心）
  RP4: rpca_enabled=False → 无任何加成
  RP5: importance < rpca_min_importance(0.40) → 不参与 RPCA
  RP6: stability 加成后不超过 365.0（cap 保护）
  RP7: rpca_retrieval_bonus 可配置（更大 bonus → 更大加成）
  RP8: update_accessed() 集成测试：retrieval source 触发 RPCA stability 加成

认知科学依据：
  Roediger & Karpicke (2006) Psychological Science "Test-Enhanced Learning" —
    主动检索（retrieval practice）比被动重读（restudy）在延迟测试中保留率高约 50%。
  Karpicke & Roediger (2008) Science — retrieval practice 长时记忆优势 1.5-2 倍。

OS 类比：Linux page fault (demand fault = retrieval) → immediate active LRU list promotion;
  readahead prefetch (restudy) → inactive list first, needs second access to promote.
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
    apply_retrieval_practice_consolidation_asymmetry,
    update_accessed,
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
                  chunk_type="decision", retrievability=0.5, access_count=2):
    """Insert a chunk with controlled state for RPCA testing."""
    now_iso = _utcnow().isoformat()
    import datetime as _dt
    la_iso = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(minutes=10)).isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count,
            encode_context)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (cid, project, chunk_type, f"content {cid}", f"summary {cid}",
         importance, stability, now_iso, now_iso, retrievability,
         la_iso, access_count, "kernel_mm")
    )
    conn.commit()


def _get_stability(conn, cid: str) -> float:
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


# ── RP1: retrieval source → stability 增加 ───────────────────────────────────────────────

def test_rp1_retrieval_boosts_stability(conn):
    """RP1: access_source='retrieval' + importance >= 0.40 → stability 增加。"""
    _insert_chunk(conn, "rpca_ret", stability=5.0, importance=0.6)

    stab_before = _get_stability(conn, "rpca_ret")
    result = apply_retrieval_practice_consolidation_asymmetry(
        conn, ["rpca_ret"], "test",
        access_source_map={"rpca_ret": "retrieval"}
    )
    stab_after = _get_stability(conn, "rpca_ret")

    assert stab_after > stab_before, (
        f"RP1: retrieval source 应触发 RPCA 加成，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["rpca_boosted"] >= 1, f"RP1: rpca_boosted 应 >= 1，got {result}"


# ── RP2: restudy source → 轻微加成 ────────────────────────────────────────────────────────

def test_rp2_restudy_small_boost(conn):
    """RP2: access_source='restudy' + importance >= 0.40 → 轻微 stability 加成（< retrieval）。"""
    _insert_chunk(conn, "rpca_rest", stability=5.0, importance=0.6)

    stab_before = _get_stability(conn, "rpca_rest")
    result = apply_retrieval_practice_consolidation_asymmetry(
        conn, ["rpca_rest"], "test",
        access_source_map={"rpca_rest": "restudy"}
    )
    stab_after = _get_stability(conn, "rpca_rest")

    rpca_restudy_bonus = config.get("store_vfs.rpca_restudy_bonus")  # 0.02
    expected_stab = 5.0 * (1.0 + rpca_restudy_bonus)

    assert stab_after > stab_before, (
        f"RP2: restudy source 应有轻微 stability 加成，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert abs(stab_after - expected_stab) < 0.01, (
        f"RP2: restudy 加成应约等于 rpca_restudy_bonus={rpca_restudy_bonus:.2f}，"
        f"expected={expected_stab:.4f} got={stab_after:.4f}"
    )
    assert result["rpca_boosted"] >= 1, f"RP2: rpca_boosted 应 >= 1，got {result}"


# ── RP3: retrieval 加成 > restudy 加成 ────────────────────────────────────────────────────

def test_rp3_retrieval_greater_than_restudy(conn):
    """RP3: 相同条件下 retrieval 加成 > restudy 加成（主动检索效益更高）。"""
    _insert_chunk(conn, "ret_cmp", stability=5.0, importance=0.6)
    _insert_chunk(conn, "rest_cmp", stability=5.0, importance=0.6)

    stab_before_ret = _get_stability(conn, "ret_cmp")
    stab_before_rest = _get_stability(conn, "rest_cmp")

    apply_retrieval_practice_consolidation_asymmetry(
        conn, ["ret_cmp"], "test",
        access_source_map={"ret_cmp": "retrieval"}
    )
    apply_retrieval_practice_consolidation_asymmetry(
        conn, ["rest_cmp"], "test",
        access_source_map={"rest_cmp": "restudy"}
    )

    stab_after_ret = _get_stability(conn, "ret_cmp")
    stab_after_rest = _get_stability(conn, "rest_cmp")

    delta_ret = stab_after_ret - stab_before_ret
    delta_rest = stab_after_rest - stab_before_rest

    assert delta_ret > delta_rest, (
        f"RP3: retrieval 加成应大于 restudy 加成，"
        f"delta_ret={delta_ret:.5f} delta_rest={delta_rest:.5f}"
    )


# ── RP4: rpca_enabled=False → 无加成 ─────────────────────────────────────────────────────

def test_rp4_disabled_no_boost(conn):
    """RP4: store_vfs.rpca_enabled=False → 无任何 RPCA 加成。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.rpca_enabled":
            return False
        return original_get(key, project=project)

    _insert_chunk(conn, "rpca_dis", stability=5.0, importance=0.6)
    stab_before = _get_stability(conn, "rpca_dis")

    with mock.patch.object(config, 'get', side_effect=patched_get):
        result = apply_retrieval_practice_consolidation_asymmetry(
            conn, ["rpca_dis"], "test",
            access_source_map={"rpca_dis": "retrieval"}
        )
    stab_after = _get_stability(conn, "rpca_dis")

    assert abs(stab_after - stab_before) < 0.001, (
        f"RP4: disabled 时不应有 RPCA 加成，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["rpca_boosted"] == 0, f"RP4: rpca_boosted 应为 0，got {result}"


# ── RP5: importance 不足 → 不参与 RPCA ────────────────────────────────────────────────────

def test_rp5_low_importance_excluded(conn):
    """RP5: importance < rpca_min_importance(0.40) → 不参与 RPCA。"""
    _insert_chunk(conn, "rpca_low_imp", stability=5.0, importance=0.20)  # < 0.40
    stab_before = _get_stability(conn, "rpca_low_imp")

    result = apply_retrieval_practice_consolidation_asymmetry(
        conn, ["rpca_low_imp"], "test",
        access_source_map={"rpca_low_imp": "retrieval"}
    )
    stab_after = _get_stability(conn, "rpca_low_imp")

    assert abs(stab_after - stab_before) < 0.001, (
        f"RP5: 低 importance 不应触发 RPCA，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["rpca_boosted"] == 0, f"RP5: rpca_boosted 应为 0，got {result}"


# ── RP6: stability 上限 365.0 ─────────────────────────────────────────────────────────────

def test_rp6_stability_cap_365(conn):
    """RP6: RPCA boost 后 stability 不超过 365.0。"""
    _insert_chunk(conn, "rpca_cap", stability=364.9, importance=0.8)

    apply_retrieval_practice_consolidation_asymmetry(
        conn, ["rpca_cap"], "test",
        access_source_map={"rpca_cap": "retrieval"}
    )
    stab_after = _get_stability(conn, "rpca_cap")

    assert stab_after <= 365.0, f"RP6: stability 不应超过 365.0，got {stab_after}"


# ── RP7: rpca_retrieval_bonus 可配置 ─────────────────────────────────────────────────────

def test_rp7_configurable_retrieval_bonus(conn):
    """RP7: rpca_retrieval_bonus=0.25 时加成比默认 0.10 更大。"""
    original_get = config.get

    def patched_25(key, project=None):
        if key == "store_vfs.rpca_retrieval_bonus":
            return 0.25
        return original_get(key, project=project)

    _insert_chunk(conn, "rpca_scale", stability=5.0, importance=0.6)
    stab_before = _get_stability(conn, "rpca_scale")

    with mock.patch.object(config, 'get', side_effect=patched_25):
        apply_retrieval_practice_consolidation_asymmetry(
            conn, ["rpca_scale"], "test",
            access_source_map={"rpca_scale": "retrieval"}
        )
    stab_after_25 = _get_stability(conn, "rpca_scale")
    delta_25 = stab_after_25 - stab_before

    # 重置
    conn.execute("UPDATE memory_chunks SET stability=5.0 WHERE id='rpca_scale'")
    conn.commit()

    apply_retrieval_practice_consolidation_asymmetry(
        conn, ["rpca_scale"], "test",
        access_source_map={"rpca_scale": "retrieval"}
    )
    stab_after_default = _get_stability(conn, "rpca_scale")
    delta_default = stab_after_default - 5.0

    assert delta_25 > delta_default, (
        f"RP7: rpca_retrieval_bonus=0.25 加成应大于默认 0.10，"
        f"delta_25={delta_25:.5f} delta_default={delta_default:.5f}"
    )


# ── RP8: update_accessed() 集成测试 ─────────────────────────────────────────────────────

def test_rp8_update_accessed_integration(conn):
    """RP8: update_accessed() 对 retrieval source chunk 触发 RPCA stability 加成。"""
    _insert_chunk(conn, "rpca_integ", stability=5.0, importance=0.6,
                  retrievability=0.5, access_count=2)

    stab_before = _get_stability(conn, "rpca_integ")

    # 传入 access_source_map 参数触发 RPCA
    update_accessed(conn, ["rpca_integ"],
                    access_source_map={"rpca_integ": "retrieval"})

    stab_after = _get_stability(conn, "rpca_integ")

    assert stab_after > stab_before, (
        f"RP8: update_accessed 对 retrieval source chunk 应触发 RPCA 加成，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )
