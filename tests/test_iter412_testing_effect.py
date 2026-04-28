"""
test_iter412_testing_effect.py — iter412: Testing Effect 单元测试

覆盖：
  TE1: 低 stability + 中等间隔 → R_at_recall 低 → quality 被 boost
  TE2: 高 stability + 中等间隔 → R_at_recall 高（容易检索）→ quality 不变
  TE3: 极低 retrievability (R≈0) → 最大 quality bonus
  TE4: quality 上限为 5（不超限）
  TE5: testing_effect_enabled=False → 关闭后 quality 不 boost
  TE6: testing_effect_scale=0 → 无 bonus（scale 清零）
  TE7: 短间隔（gap < 1hr）→ quality=3 + difficulty bonus（但已很新，R 高，bonus 小）
  TE8: gap=0（刚插入）→ R_at_recall≈1.0 → bonus=0
  TE9: 非常旧且 stability 小 → 极高难度 → quality boost 到 5
  TE10: 批量混合 stability → 各 chunk 得到不同 quality

认知科学依据：
  Roediger & Karpicke (2006) Test-Enhanced Learning —
    主动检索比被动重读提升 +50% 长期保留率。
  Bjork (1994) Desirable Difficulties —
    需要努力的检索形成更强的记忆痕迹（elaborative encoding 更深）。

OS 类比：Linux L3 cache miss → aggressive LRU promotion —
  L1 命中（容易检索）不改变 LRU position；L3 miss（困难检索）→ cache line 晋升到 L1/L2。
"""
import sys
import sqlite3
import math
import pytest
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import tmpfs  # noqa

from store_vfs import ensure_schema, update_accessed
from store import insert_chunk
import config


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _make_chunk(cid, stability=2.0, last_accessed_ago_secs=0, project="test"):
    now = datetime.now(timezone.utc)
    la = (now - timedelta(seconds=last_accessed_ago_secs)).isoformat()
    now_iso = now.isoformat()
    return {
        "id": cid,
        "created_at": now_iso,
        "updated_at": now_iso,
        "project": project,
        "source_session": "s1",
        "chunk_type": "decision",
        "info_class": "semantic",
        "content": f"chunk {cid}",
        "summary": f"chunk {cid}",
        "tags": [],
        "importance": 0.8,
        "retrievability": 0.9,
        "last_accessed": la,
        "access_count": 2,
        "oom_adj": 0,
        "lru_gen": 0,
        "stability": stability,
        "raw_snippet": "",
        "encoding_context": {},
    }


def _get_stability(conn, cid: str) -> float:
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


def _r_at_recall(gap_secs: float, stability_days: float) -> float:
    """Compute expected retrievability at time of recall."""
    gap_hours = gap_secs / 3600.0
    return math.exp(-gap_hours / max(0.01, stability_days * 24.0))


# ── TE1: Low stability + medium gap → difficulty boost ──────────────────────

def test_te1_low_stability_medium_gap_boosted(conn):
    """低 stability=0.5 + 6hr 间隔 → R_at_recall 低 → quality 被 boost → 较大 stability 增益。"""
    # stability=0.5, gap=6hr: R = exp(-6/(0.5×24)) = exp(-0.5) ≈ 0.607, difficulty=0.393
    # base quality=4 (medium zone), scale=2.0 → bonus=round(0.393×2)=1 → effective_quality=5
    chunk = _make_chunk("te1", stability=0.5, last_accessed_ago_secs=6*3600)
    insert_chunk(conn, chunk)
    conn.commit()

    stab_before = _get_stability(conn, "te1")
    update_accessed(conn, ["te1"])
    conn.commit()

    stab = _get_stability(conn, "te1")
    ratio = stab / stab_before if stab_before else 0
    # quality=5 → ×1.2 (testing effect boosted from 4→5)
    assert ratio >= 1.15, f"TE1: 低 stability + 6hr 间隔应触发 testing effect boost，ratio={ratio:.4f}"


def test_te1_high_stability_same_gap_no_boost(conn):
    """高 stability=10.0 + 6hr 间隔 → R_at_recall 高 → 无 testing effect boost。"""
    # stability=10.0, gap=6hr: R = exp(-6/(10×24)) = exp(-0.025) ≈ 0.975, difficulty≈0.025
    # bonus=round(0.025×2)=0 → effective_quality=4 → ×1.1
    chunk = _make_chunk("te1h", stability=10.0, last_accessed_ago_secs=6*3600)
    insert_chunk(conn, chunk)
    conn.commit()

    stab_before = _get_stability(conn, "te1h")
    update_accessed(conn, ["te1h"])
    conn.commit()

    stab = _get_stability(conn, "te1h")
    ratio = stab / stab_before if stab_before else 0
    # quality=4 → ×1.1 (no testing effect boost since R≈0.975)
    assert abs(ratio - 1.1) < 0.05, f"TE1h: 高 stability 无 boost，ratio 应≈1.1，got ratio={ratio:.4f}"


# ── TE2: High stability + medium gap → no boost ──────────────────────────────

def test_te2_high_stability_no_boost(conn):
    """stability=5.0 + 6hr → R_at_recall ≈ 0.95 → difficulty ≈ 0.05 → bonus=0。"""
    chunk = _make_chunk("te2", stability=5.0, last_accessed_ago_secs=6*3600)
    insert_chunk(conn, chunk)
    conn.commit()

    stab_before = _get_stability(conn, "te2")
    update_accessed(conn, ["te2"])
    conn.commit()

    stab = _get_stability(conn, "te2")
    ratio = stab / stab_before if stab_before else 0
    # R_at_recall = exp(-6/120) = exp(-0.05) ≈ 0.951, difficulty=0.049, bonus=0
    # quality=4 → ×1.1
    assert abs(ratio - 1.1) < 0.05, f"TE2: 高 stability 无 boost，ratio 应≈1.1，got {ratio:.4f}"


# ── TE3: Extreme low retrievability → max quality bonus ──────────────────────

def test_te3_extreme_low_retrievability_max_bonus(conn):
    """stability=0.1 + 12hr → R≈exp(-50)≈0 → 最大 difficulty → max bonus。"""
    # R = exp(-12/(0.1×24)) = exp(-5) ≈ 0.0067 → difficulty ≈ 0.993 → bonus=round(1.99)=2
    # base quality=4 (medium zone) → effective_quality=min(5,6)=5 → ×1.2
    chunk = _make_chunk("te3", stability=0.1, last_accessed_ago_secs=12*3600)
    insert_chunk(conn, chunk)
    conn.commit()

    stab_before = _get_stability(conn, "te3")
    update_accessed(conn, ["te3"])
    conn.commit()

    stab = _get_stability(conn, "te3")
    ratio = stab / stab_before if stab_before else 0
    assert ratio >= 1.15, f"TE3: 极低 R_at_recall 应触发最大 bonus，ratio={ratio:.4f}"


# ── TE4: Quality capped at 5 ─────────────────────────────────────────────────

def test_te4_quality_capped_at_5(conn):
    """gap >= 24hr → base quality=5 + testing bonus → capped at 5 → ×1.2。"""
    # Long gap already gives quality=5, testing bonus can't exceed 5
    chunk = _make_chunk("te4", stability=0.5, last_accessed_ago_secs=3*86400)
    insert_chunk(conn, chunk)
    conn.commit()

    stab_before = _get_stability(conn, "te4")
    update_accessed(conn, ["te4"])
    conn.commit()

    stab = _get_stability(conn, "te4")
    ratio = stab / stab_before if stab_before else 0
    # max quality=5 → ×1.2
    assert abs(ratio - 1.2) < 0.05, f"TE4: quality capped at 5 → ×1.2，got ratio={ratio:.4f}"


# ── TE5: testing_effect_enabled=False ────────────────────────────────────────

def test_te5_disabled_testing_effect(conn, monkeypatch):
    """testing_effect_enabled=False → 禁用后 quality 不被 boost，仅由 gap 决定。"""
    import unittest.mock as mock
    original_get = config.get
    def patched_get(key, project=None):
        if key == "recon.testing_effect_enabled":
            return False
        return original_get(key, project=project)

    chunk = _make_chunk("te5", stability=0.5, last_accessed_ago_secs=6*3600)
    insert_chunk(conn, chunk)
    conn.commit()

    stab_before = _get_stability(conn, "te5")

    with mock.patch.object(config, 'get', side_effect=patched_get):
        update_accessed(conn, ["te5"])
    conn.commit()

    stab = _get_stability(conn, "te5")
    ratio = stab / stab_before if stab_before else 0
    # Without testing effect: gap=6hr → medium zone → quality=4 → ×1.1
    assert abs(ratio - 1.1) < 0.05, \
        f"TE5: 禁用 testing effect，ratio 应≈1.1（仅 gap 决定），got {ratio:.4f}"


# ── TE6: testing_effect_scale=0 ──────────────────────────────────────────────

def test_te6_zero_scale_no_bonus(conn, monkeypatch):
    """testing_effect_scale=0 → bonus=0，quality 不变。"""
    import unittest.mock as mock
    original_get = config.get
    def patched_get(key, project=None):
        if key == "recon.testing_effect_scale":
            return 0.0
        return original_get(key, project=project)

    chunk = _make_chunk("te6", stability=0.5, last_accessed_ago_secs=6*3600)
    insert_chunk(conn, chunk)
    conn.commit()

    stab_before = _get_stability(conn, "te6")

    with mock.patch.object(config, 'get', side_effect=patched_get):
        update_accessed(conn, ["te6"])
    conn.commit()

    stab = _get_stability(conn, "te6")
    ratio = stab / stab_before if stab_before else 0
    # scale=0 → bonus=0 → quality=4 → ×1.1
    assert abs(ratio - 1.1) < 0.05, f"TE6: scale=0 无 bonus，ratio 应≈1.1，got {ratio:.4f}"


# ── TE7: Short gap → R near 1 → small or zero bonus ─────────────────────────

def test_te7_short_gap_small_bonus(conn):
    """短间隔（30s）→ gap < 1hr → quality=3（无增益）+ testing_effect_bonus 小。"""
    # gap=30s: R = exp(-0.0083/stability×24) ≈ very close to 1 → difficulty ≈ 0 → bonus=0
    chunk = _make_chunk("te7", stability=2.0, last_accessed_ago_secs=30)
    insert_chunk(conn, chunk)
    conn.commit()

    stab_before = _get_stability(conn, "te7")
    update_accessed(conn, ["te7"])
    conn.commit()

    stab = _get_stability(conn, "te7")
    ratio = stab / stab_before if stab_before else 0
    # Short gap: base quality=3, R≈1.0 → difficulty≈0 → bonus=0 → ×1.0
    assert abs(ratio - 1.0) < 0.05, f"TE7: 30s 间隔 quality=3 + no bonus → ratio≈1.0，got {ratio:.4f}"


# ── TE8: explicit recall_quality → testing effect bypassed ───────────────────

def test_te8_explicit_quality_bypasses_testing_effect(conn):
    """显式 recall_quality 优先，testing effect 不介入。"""
    # stability=0.5, gap=6hr would trigger testing effect if None
    # but with explicit quality=3, should get exactly ×1.0
    chunk = _make_chunk("te8", stability=0.5, last_accessed_ago_secs=6*3600)
    insert_chunk(conn, chunk)
    conn.commit()

    stab_before = _get_stability(conn, "te8")
    update_accessed(conn, ["te8"], recall_quality=3)  # explicit quality=3
    conn.commit()

    stab = _get_stability(conn, "te8")
    ratio = stab / stab_before if stab_before else 0
    # Explicit quality=3 → ×1.0, testing effect not applied
    assert abs(ratio - 1.0) < 0.05, \
        f"TE8: 显式 quality=3 绕过 testing effect，ratio 应≈1.0，got {ratio:.4f}"


# ── TE9: Testing effect + long gap → maximum boost ───────────────────────────

def test_te9_long_gap_low_stab_max_quality(conn):
    """stability=0.3 + 3 days gap → R≈0 → 最大 testing effect + long gap = quality=5。"""
    chunk = _make_chunk("te9", stability=0.3, last_accessed_ago_secs=3*86400)
    insert_chunk(conn, chunk)
    conn.commit()

    stab_before = _get_stability(conn, "te9")
    update_accessed(conn, ["te9"])
    conn.commit()

    stab = _get_stability(conn, "te9")
    ratio = stab / stab_before if stab_before else 0
    # Long gap: base quality=5, testing boost can't exceed 5 → ×1.2
    assert abs(ratio - 1.2) < 0.05, f"TE9: 长间隔+低 stability → quality=5 → ×1.2，got {ratio:.4f}"


# ── TE10: Batch with mixed stability → different quality per chunk ────────────

def test_te10_batch_mixed_stability(conn):
    """批量更新：stability 不同的 chunk 因 testing effect 获得不同增益。"""
    # low_stab: gap=6hr, stability=0.5 → difficulty high → quality=5 → ×1.2
    # high_stab: gap=6hr, stability=10.0 → difficulty low → quality=4 → ×1.1
    c_low = _make_chunk("te10_low", stability=0.5, last_accessed_ago_secs=6*3600)
    c_high = _make_chunk("te10_high", stability=10.0, last_accessed_ago_secs=6*3600)
    insert_chunk(conn, c_low)
    insert_chunk(conn, c_high)
    conn.commit()

    stab_low_before = _get_stability(conn, "te10_low")
    stab_high_before = _get_stability(conn, "te10_high")
    update_accessed(conn, ["te10_low", "te10_high"])
    conn.commit()

    stab_low = _get_stability(conn, "te10_low")
    stab_high = _get_stability(conn, "te10_high")
    ratio_low = stab_low / stab_low_before if stab_low_before else 0
    ratio_high = stab_high / stab_high_before if stab_high_before else 0

    assert ratio_low > ratio_high, (
        f"TE10: 低 stability chunk 应比高 stability chunk 获得更大 stability 增益，"
        f"low_ratio={ratio_low:.4f} > high_ratio={ratio_high:.4f}"
    )
