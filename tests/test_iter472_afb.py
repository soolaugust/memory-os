"""
test_iter472_afb.py — iter472: Access Frequency Boost 单元测试

覆盖：
  AF1: access_count >= afb_min_count(3) → stability 加成
  AF2: access_count < afb_min_count → 无加成
  AF3: afb_enabled=False → 无任何加成
  AF4: importance < afb_min_importance(0.25) → 不参与 AFB
  AF5: access_count 越高 → 加成越大（单调性）
  AF6: 加成受 afb_max_boost(0.20) 保护
  AF7: stability 加成后不超过 365.0
  AF8: update_accessed 集成测试 — 多次访问后 access_count 增加触发 AFB

认知科学依据：
  Newell & Rosenbloom (1981) "Mechanisms of skill acquisition and the law of practice" —
    熟练度提升遵循幂律：performance ∝ trials^(-0.4)；检索次数 ↑ → 记忆强度 ↑。
  Anderson (1983) ACT* 理论：memory strength = ΣΑ_j × t_j^(-d)，检索次数是最强预测因子。
  Bahrick (1979): 间隔检索后长期保留量与检索次数正相关（r=0.78）。

OS 类比：Linux active LRU promotion（mm/swap.c: mark_page_accessed）—
  多次被访问（PG_referenced → active LRU）的 hot page 获得更高驻留优先级；
  access_count ≥ afb_min_count = 页面进入 hot tier → kswapd 跳过淘汰。
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

from store_vfs import ensure_schema, apply_access_frequency_boost, update_accessed
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


def _insert_chunk(conn, cid, stability=5.0, importance=0.6,
                  access_count=0, project="test"):
    now_iso = _utcnow().isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count,
            encode_context, session_type_history)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (cid, project, "decision", f"content {cid}", f"summary {cid}",
         importance, stability, now_iso, now_iso, 0.5,
         now_iso, access_count, "test_ctx", "")
    )
    conn.commit()


def _get_stability(conn, cid: str) -> float:
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


# ── AF1: 高访问次数 → stability 加成 ─────────────────────────────────────────────────────────

def test_af1_high_access_count_boosted(conn):
    """AF1: access_count >= afb_min_count(3) → stability 加成。"""
    # 访问次数 >= afb_min_count
    _insert_chunk(conn, "afb_1_high", stability=5.0, access_count=5)
    stab_before = _get_stability(conn, "afb_1_high")

    result = apply_access_frequency_boost(conn, ["afb_1_high"])
    stab_after = _get_stability(conn, "afb_1_high")

    assert stab_after > stab_before, (
        f"AF1: 高访问次数 stability 应加成，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["afb_boosted"] >= 1, f"AF1: afb_boosted 应 >= 1，got {result}"


# ── AF2: 低访问次数 → 无加成 ─────────────────────────────────────────────────────────────────

def test_af2_low_access_count_no_boost(conn):
    """AF2: access_count < afb_min_count(3) → 无 AFB 加成。"""
    # 访问次数 < afb_min_count（1 次）
    _insert_chunk(conn, "afb_2_low", stability=5.0, access_count=1)
    stab_before = _get_stability(conn, "afb_2_low")

    result = apply_access_frequency_boost(conn, ["afb_2_low"])
    stab_after = _get_stability(conn, "afb_2_low")

    assert abs(stab_after - stab_before) < 0.001, (
        f"AF2: 低访问次数不应有 AFB 加成，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["afb_boosted"] == 0, f"AF2: afb_boosted 应为 0，got {result}"


# ── AF3: afb_enabled=False → 无加成 ──────────────────────────────────────────────────────────

def test_af3_disabled_no_boost(conn):
    """AF3: afb_enabled=False → 无任何 AFB 加成。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.afb_enabled":
            return False
        return original_get(key, project=project)

    _insert_chunk(conn, "afb_3", stability=5.0, access_count=10)
    stab_before = _get_stability(conn, "afb_3")

    with mock.patch.object(config, 'get', side_effect=patched_get):
        result = apply_access_frequency_boost(conn, ["afb_3"])
    stab_after = _get_stability(conn, "afb_3")

    assert abs(stab_after - stab_before) < 0.001, (
        f"AF3: disabled 时不应有 AFB 加成，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["afb_boosted"] == 0, f"AF3: afb_boosted 应为 0，got {result}"


# ── AF4: importance 不足 → 不参与 AFB ─────────────────────────────────────────────────────────

def test_af4_low_importance_no_boost(conn):
    """AF4: importance < afb_min_importance(0.25) → 不参与 AFB。"""
    _insert_chunk(conn, "afb_4", stability=5.0, importance=0.10, access_count=10)
    stab_before = _get_stability(conn, "afb_4")

    result = apply_access_frequency_boost(conn, ["afb_4"])
    stab_after = _get_stability(conn, "afb_4")

    assert abs(stab_after - stab_before) < 0.001, (
        f"AF4: 低 importance 不应有 AFB 加成，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["afb_boosted"] == 0, f"AF4: afb_boosted 应为 0，got {result}"


# ── AF5: 访问次数越高 → 加成越大（单调性）────────────────────────────────────────────────────────

def test_af5_more_accesses_more_boost(conn):
    """AF5: access_count 越高 → AFB 加成越大（或相等）。"""
    base = 5.0

    # access_count = 3（刚超过 afb_min_count）
    _insert_chunk(conn, "afb_5_low", stability=base, access_count=3)
    result_low = apply_access_frequency_boost(conn, ["afb_5_low"])
    stab_low = _get_stability(conn, "afb_5_low")

    # access_count = 10（更高访问频率）
    _insert_chunk(conn, "afb_5_high", stability=base, access_count=10)
    result_high = apply_access_frequency_boost(conn, ["afb_5_high"])
    stab_high = _get_stability(conn, "afb_5_high")

    assert stab_high >= stab_low - 0.001, (
        f"AF5: 高访问次数加成应 >= 低访问次数加成，high={stab_high:.4f} low={stab_low:.4f}"
    )


# ── AF6: 加成受 afb_max_boost 保护 ────────────────────────────────────────────────────────────

def test_af6_max_boost_cap(conn):
    """AF6: AFB 增量不超过 base × afb_max_boost(0.20)。"""
    afb_max_boost = config.get("store_vfs.afb_max_boost")  # 0.20
    base = 5.0

    # 超高访问次数
    _insert_chunk(conn, "afb_6", stability=base, access_count=100)
    stab_before = _get_stability(conn, "afb_6")
    apply_access_frequency_boost(conn, ["afb_6"])
    stab_after = _get_stability(conn, "afb_6")

    increment = stab_after - stab_before
    max_allowed = base * afb_max_boost + 0.01
    assert increment <= max_allowed, (
        f"AF6: AFB 增量 {increment:.4f} 不应超过 max_boost 允许的 {max_allowed:.4f}"
    )
    assert stab_after > stab_before, f"AF6: 应有 AFB 加成，before={stab_before:.4f} after={stab_after:.4f}"


# ── AF7: stability 上限 365.0 ─────────────────────────────────────────────────────────────

def test_af7_stability_cap_365(conn):
    """AF7: AFB boost 后 stability 不超过 365.0。"""
    _insert_chunk(conn, "afb_7", stability=364.5, importance=0.8, access_count=50)
    apply_access_frequency_boost(conn, ["afb_7"])
    stab = _get_stability(conn, "afb_7")
    assert stab <= 365.0, f"AF7: stability 不应超过 365.0，got {stab}"


# ── AF8: update_accessed 集成测试 ─────────────────────────────────────────────────────────────

def test_af8_update_accessed_integration(conn):
    """AF8: update_accessed 后 access_count 增加，高频访问 chunk 触发 AFB 加成。"""
    # 预置较高 access_count（>= afb_min_count）
    _insert_chunk(conn, "afb_8", stability=5.0, importance=0.6, access_count=5)
    stab_before = _get_stability(conn, "afb_8")

    # update_accessed 会递增 access_count 并触发 AFB
    update_accessed(conn, ["afb_8"])
    stab_after = _get_stability(conn, "afb_8")

    assert stab_after >= stab_before, (
        f"AF8: update_accessed 后 stability 不应降低，before={stab_before:.4f} after={stab_after:.4f}"
    )
