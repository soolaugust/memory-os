"""
test_iter432_cumulative_interference.py — iter432: Cumulative Interference Effect 单元测试

覆盖：
  CI1: N_same_type >= ci_min_n → factor > 1.0
  CI2: N_same_type < ci_min_n → factor = 1.0（无干扰）
  CI3: factor 随 N_same_type 对数增长
  CI4: factor 上限为 ci_max_factor（2.0）
  CI5: cumulative_interference_enabled=False → factor 始终 1.0
  CI6: 保护类型（design_constraint/procedure）不受干扰影响
  CI7: decay_stability_by_type_with_ci — 高密度类型衰减更快
  CI8: 自定义 ci_scale 可通过 sysctl 配置
  CI9: N_median 正确影响 factor（N_same_type=N_median → factor=1+scale）
  CI10: 豁免类型使用普通衰减率，非豁免类型使用 effective_decay

认知科学依据：
  Underwood (1957) "Interference and forgetting" —
    过去学习的材料（干扰列表数）与 24h 遗忘量 r=0.92 的极强正相关。
  Jenkins & Dallenbach (1924): 干扰减少 → 遗忘减少（睡眠实验）。

OS 类比：Linux CPU cache set-associativity conflict —
  同一 cache set 中的 cache line 越多，每条 line 的平均留存时间越短。
"""
import sys
import sqlite3
import math
import datetime
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import tmpfs  # noqa

from store_vfs import (
    ensure_schema,
    compute_cumulative_interference_factor,
    decay_stability_by_type_with_ci,
)
import config


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _days_ago_iso(days: float) -> str:
    dt = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)
    return dt.isoformat()


def _insert_chunk(conn, cid, chunk_type="decision", project="test",
                  stability=2.0, importance=0.6, access_count=0,
                  days_old=60.0):
    """插入长期未访问的 chunk（触发 decay 条件：last_accessed < cutoff, access_count < 2）。"""
    old_time = _days_ago_iso(days_old)
    now = _now_iso()
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0.9, ?, ?)""",
        (cid, project, chunk_type, f"content {cid}", f"summary {cid}",
         importance, stability, old_time, now, old_time, access_count)
    )
    conn.commit()


def _get_stability(conn, cid: str) -> float:
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


# ── CI1: N >= min → factor > 1.0 ────────────────────────────────────────────

def test_ci1_high_density_factor_greater_than_one():
    """CI1: N_same_type >= ci_min_n(5) → factor > 1.0。"""
    factor = compute_cumulative_interference_factor(n_same_type=10, n_median=10)
    assert factor > 1.0, f"CI1: N=10 应有干扰因子 > 1.0，got {factor}"
    # factor = 1 + 0.30 * log(11) / log(11) = 1.30
    expected = 1.0 + 0.30 * math.log(1 + 10) / math.log(1 + 10)
    assert abs(factor - expected) < 0.01, f"CI1: factor 应约为 {expected:.3f}，got {factor}"


# ── CI2: N < min → factor = 1.0 ─────────────────────────────────────────────

def test_ci2_low_density_no_interference():
    """CI2: N_same_type < ci_min_n(5) → factor = 1.0（无干扰）。"""
    factor_3 = compute_cumulative_interference_factor(n_same_type=3, n_median=10)
    factor_0 = compute_cumulative_interference_factor(n_same_type=0, n_median=10)
    assert factor_3 == 1.0, f"CI2: N=3 (< min=5) 应无干扰 factor=1.0，got {factor_3}"
    assert factor_0 == 1.0, f"CI2: N=0 应无干扰 factor=1.0，got {factor_0}"


# ── CI3: factor 随 N 对数增长 ────────────────────────────────────────────────

def test_ci3_factor_increases_logarithmically():
    """CI3: factor 随 N_same_type 对数增长（N=5 < N=20 < N=100）。"""
    f5 = compute_cumulative_interference_factor(5, 10)
    f20 = compute_cumulative_interference_factor(20, 10)
    f100 = compute_cumulative_interference_factor(100, 10)
    assert f5 < f20, f"CI3: N=5 factor({f5:.3f}) 应 < N=20 factor({f20:.3f})"
    assert f20 < f100, f"CI3: N=20 factor({f20:.3f}) 应 < N=100 factor({f100:.3f})"


# ── CI4: factor 上限 ci_max_factor ──────────────────────────────────────────

def test_ci4_factor_capped_at_max():
    """CI4: factor 不超过 ci_max_factor（默认 2.0）。"""
    # N_same_type = 10000，远超 n_median=10
    factor = compute_cumulative_interference_factor(10000, 10)
    max_factor = config.get("scorer.ci_max_factor")
    assert factor <= max_factor, f"CI4: factor 不应超过 {max_factor}，got {factor}"
    assert factor == max_factor, f"CI4: 极高 N 时 factor 应达到上限 {max_factor}，got {factor}"


# ── CI5: enabled=False → factor 始终 1.0 ─────────────────────────────────────

def test_ci5_disabled_no_interference():
    """CI5: cumulative_interference_enabled=False → factor 始终 1.0。"""
    import unittest.mock as mock
    original_get = config.get
    def patched_get(key, project=None):
        if key == "scorer.cumulative_interference_enabled":
            return False
        return original_get(key, project=project)

    with mock.patch.object(config, 'get', side_effect=patched_get):
        f = compute_cumulative_interference_factor(100, 10)

    assert f == 1.0, f"CI5: 禁用时 factor 应为 1.0，got {f}"


# ── CI6: 保护类型不受干扰 ─────────────────────────────────────────────────────

def test_ci6_protected_types_not_affected(conn):
    """CI6: design_constraint 和 procedure 豁免累积干扰。"""
    # 插入大量 design_constraint（触发高 N_same_type）
    for i in range(20):
        _insert_chunk(conn, f"dc_{i}", chunk_type="design_constraint",
                     stability=5.0, days_old=60)
    # 插入大量 decision（非保护类型）
    for i in range(20):
        _insert_chunk(conn, f"dec_{i}", chunk_type="decision",
                     stability=5.0, days_old=60)

    result = decay_stability_by_type_with_ci(conn, "test", stale_days=30)
    ci_factors = result.get("ci_factors", {})

    # design_constraint 的 factor 应为 1.0（豁免）
    dc_factor = ci_factors.get("design_constraint", 1.0)
    assert dc_factor == 1.0, f"CI6: design_constraint 应豁免，factor={dc_factor}"

    # decision 的 factor 应 > 1.0（N=20 >= min=5）
    dec_factor = ci_factors.get("decision", 1.0)
    assert dec_factor > 1.0, f"CI6: decision (N=20) 应有 CI factor > 1.0，got {dec_factor}"


# ── CI7: 高密度类型衰减更快 ──────────────────────────────────────────────────

def test_ci7_high_density_decays_faster(conn):
    """CI7: decay_with_ci 中，高密度类型（decision×20）比低密度（causal_chain×6）衰减更快。"""
    # decision: 20 chunks (高密度)
    for i in range(20):
        _insert_chunk(conn, f"dec_{i}", chunk_type="decision",
                     stability=5.0, days_old=60)
    # causal_chain: 6 chunks (低密度，刚过 min_n=5)
    for i in range(6):
        _insert_chunk(conn, f"cc_{i}", chunk_type="causal_chain",
                     stability=5.0, days_old=60)

    decay_stability_by_type_with_ci(conn, "test", stale_days=30)

    # 高密度 decision 的 stability 应比低密度 causal_chain 更低
    dec_stabs = [_get_stability(conn, f"dec_{i}") for i in range(20)]
    cc_stabs = [_get_stability(conn, f"cc_{i}") for i in range(6)]

    avg_dec = sum(dec_stabs) / len(dec_stabs)
    avg_cc = sum(cc_stabs) / len(cc_stabs)

    assert avg_dec < avg_cc, (
        f"CI7: 高密度 decision avg_stability({avg_dec:.3f}) 应 < "
        f"低密度 causal_chain ({avg_cc:.3f})"
    )


# ── CI8: 自定义 ci_scale ─────────────────────────────────────────────────────

def test_ci8_custom_ci_scale():
    """CI8: 自定义 ci_scale=0.60 时，factor 约为默认 scale=0.30 的 2 倍（对数线性）。"""
    import unittest.mock as mock
    original_get = config.get

    def patched_get_60(key, project=None):
        if key == "scorer.ci_scale":
            return 0.60
        return original_get(key, project=project)

    def patched_get_30(key, project=None):
        if key == "scorer.ci_scale":
            return 0.30
        return original_get(key, project=project)

    n = 20
    with mock.patch.object(config, 'get', side_effect=patched_get_60):
        f60 = compute_cumulative_interference_factor(n, 10)
    with mock.patch.object(config, 'get', side_effect=patched_get_30):
        f30 = compute_cumulative_interference_factor(n, 10)

    # factor_60 - 1.0 ≈ 2 × (factor_30 - 1.0)
    delta60 = f60 - 1.0
    delta30 = f30 - 1.0
    assert delta60 > delta30, f"CI8: scale=0.60 factor({f60:.3f}) 应 > scale=0.30 ({f30:.3f})"
    assert abs(delta60 / delta30 - 2.0) < 0.1, \
        f"CI8: scale 翻倍 → delta 应翻倍，ratio={delta60/delta30:.3f}"


# ── CI9: N_median 影响 factor ─────────────────────────────────────────────────

def test_ci9_n_median_normalization():
    """CI9: N_same_type=N_median 时，factor = 1 + ci_scale（scale=0.30 → factor≈1.30）。"""
    scale = config.get("scorer.ci_scale")  # 0.30
    # N_same_type = N_median = 10 → factor = 1 + 0.30 * log(11)/log(11) = 1.30
    factor = compute_cumulative_interference_factor(n_same_type=10, n_median=10)
    expected = 1.0 + scale
    assert abs(factor - expected) < 0.01, \
        f"CI9: N=N_median=10 时 factor 应为 {expected:.3f}，got {factor}"


# ── CI10: 豁免类型使用原始衰减率，非豁免使用 effective_decay ──────────────────

def test_ci10_exempt_vs_nonexempt_decay(conn):
    """CI10: 验证豁免类型（design_constraint）stability 比非豁免类型（decision）衰减更少。"""
    # 插入 20 个 decision（高密度非豁免）
    for i in range(20):
        _insert_chunk(conn, f"dec_{i}", chunk_type="decision",
                     stability=3.0, days_old=60)
    # 插入 20 个 design_constraint（高密度豁免）
    for i in range(20):
        _insert_chunk(conn, f"dc_{i}", chunk_type="design_constraint",
                     stability=3.0, days_old=60)

    decay_stability_by_type_with_ci(conn, "test", stale_days=30)

    dec_after = [_get_stability(conn, f"dec_{i}") for i in range(20)]
    dc_after = [_get_stability(conn, f"dc_{i}") for i in range(20)]

    avg_dec = sum(dec_after) / len(dec_after)
    avg_dc = sum(dc_after) / len(dc_after)

    # design_constraint 衰减更慢（豁免干扰），stability 更高
    assert avg_dc > avg_dec, (
        f"CI10: design_constraint avg({avg_dc:.3f}) 应 > decision ({avg_dec:.3f})"
    )
