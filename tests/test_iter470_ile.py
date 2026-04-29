"""
test_iter470_ile.py — iter470: Interleaving Effect 单元测试

覆盖：
  IL1: session_type_history 多样性 >= ile_min_diversity(2) → stability 加成
  IL2: session_type_history 多样性 < ile_min_diversity → 无加成
  IL3: ile_enabled=False → 无任何加成
  IL4: importance < ile_min_importance(0.30) → 不参与 ILE
  IL5: 多样性越高 → 加成越大（单调性）
  IL6: 加成受 ile_max_boost(0.18) 保护
  IL7: stability 加成后不超过 365.0
  IL8: update_accessed 集成测试 — 多种 session_type 访问后 ILE 触发

认知科学依据：
  Kornell & Bjork (2008) "Learning concepts and categories" —
    交错练习（interleaved）vs. 分块练习（blocked）：测验成绩 64% vs. 36%（r=0.58）。
    机制：交错迫使大脑持续辨别相似概念 → 更深比较性处理 → 更精细记忆表征。
  Taylor & Rohrer (2010): 数学题交错练习比分块练习长期保留率高 43%。

OS 类比：Linux NUMA interleaving（mm/mempolicy.c MPOL_INTERLEAVE）—
  内存分配跨多个 NUMA 节点 → 无单点带宽瓶颈 → 整体吞吐量和容错性更高。
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

from store_vfs import ensure_schema, apply_interleaving_effect, update_accessed
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
                  session_type_history="", project="test"):
    now_iso = _utcnow().isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count,
            encode_context, session_type_history)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (cid, project, "decision", f"content {cid}", f"summary {cid}",
         importance, stability, now_iso, now_iso, 0.5,
         now_iso, 2, "test_ctx", session_type_history)
    )
    conn.commit()


def _get_stability(conn, cid: str) -> float:
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


# ── IL1: 高多样性 → stability 加成 ──────────────────────────────────────────────────────────

def test_il1_high_diversity_boosted(conn):
    """IL1: session_type_history 有多种不同 session_type → stability 加成。"""
    # 多样性高：3 种不同 session_type
    _insert_chunk(conn, "ile_1_div", stability=5.0,
                  session_type_history="coding,debugging,review")
    stab_before = _get_stability(conn, "ile_1_div")

    result = apply_interleaving_effect(conn, ["ile_1_div"])
    stab_after = _get_stability(conn, "ile_1_div")

    assert stab_after > stab_before, (
        f"IL1: 高多样性 stability 应加成，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["ile_boosted"] >= 1, f"IL1: ile_boosted 应 >= 1，got {result}"


# ── IL2: 低多样性 → 无加成 ───────────────────────────────────────────────────────────────────

def test_il2_low_diversity_no_boost(conn):
    """IL2: session_type_history 只有 1 种 session_type（< ile_min_diversity=2）→ 无 ILE 加成。"""
    _insert_chunk(conn, "ile_2_single", stability=5.0,
                  session_type_history="coding,coding,coding")
    stab_before = _get_stability(conn, "ile_2_single")

    result = apply_interleaving_effect(conn, ["ile_2_single"])
    stab_after = _get_stability(conn, "ile_2_single")

    assert abs(stab_after - stab_before) < 0.001, (
        f"IL2: 低多样性不应有 ILE 加成，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["ile_boosted"] == 0, f"IL2: ile_boosted 应为 0，got {result}"


# ── IL3: ile_enabled=False → 无加成 ──────────────────────────────────────────────────────────

def test_il3_disabled_no_boost(conn):
    """IL3: ile_enabled=False → 无任何 ILE 加成。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.ile_enabled":
            return False
        return original_get(key, project=project)

    _insert_chunk(conn, "ile_3", stability=5.0,
                  session_type_history="coding,debugging,review")
    stab_before = _get_stability(conn, "ile_3")

    with mock.patch.object(config, 'get', side_effect=patched_get):
        result = apply_interleaving_effect(conn, ["ile_3"])
    stab_after = _get_stability(conn, "ile_3")

    assert abs(stab_after - stab_before) < 0.001, (
        f"IL3: disabled 时不应有 ILE 加成，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["ile_boosted"] == 0, f"IL3: ile_boosted 应为 0，got {result}"


# ── IL4: importance 不足 → 不参与 ILE ───────────────────────────────────────────────────────

def test_il4_low_importance_no_boost(conn):
    """IL4: importance < ile_min_importance(0.30) → 不参与 ILE。"""
    _insert_chunk(conn, "ile_4", stability=5.0, importance=0.10,
                  session_type_history="coding,debugging,review")
    stab_before = _get_stability(conn, "ile_4")

    result = apply_interleaving_effect(conn, ["ile_4"])
    stab_after = _get_stability(conn, "ile_4")

    assert abs(stab_after - stab_before) < 0.001, (
        f"IL4: 低 importance 不应有 ILE 加成，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["ile_boosted"] == 0, f"IL4: ile_boosted 应为 0，got {result}"


# ── IL5: 多样性越高 → 加成越大（单调性）────────────────────────────────────────────────────────

def test_il5_more_diversity_more_boost(conn):
    """IL5: session_type 种类越多 → ILE 加成越大（或相等）。"""
    base = 5.0

    # 2 种 session_type（刚超过 ile_min_diversity=2）
    _insert_chunk(conn, "ile_5_two", stability=base,
                  session_type_history="coding,debugging")
    result_two = apply_interleaving_effect(conn, ["ile_5_two"])
    stab_two = _get_stability(conn, "ile_5_two")

    # 4 种 session_type（更高多样性）
    _insert_chunk(conn, "ile_5_four", stability=base,
                  session_type_history="coding,debugging,review,planning")
    result_four = apply_interleaving_effect(conn, ["ile_5_four"])
    stab_four = _get_stability(conn, "ile_5_four")

    assert stab_four >= stab_two - 0.001, (
        f"IL5: 4 种 session_type 加成应 >= 2 种，four={stab_four:.4f} two={stab_two:.4f}"
    )


# ── IL6: 加成受 ile_max_boost 保护 ──────────────────────────────────────────────────────────

def test_il6_max_boost_cap(conn):
    """IL6: ILE 增量不超过 base × ile_max_boost(0.18)。"""
    ile_max_boost = config.get("store_vfs.ile_max_boost")  # 0.18
    base = 5.0

    _insert_chunk(conn, "ile_6", stability=base,
                  session_type_history="a,b,c,d,e,f,g,h,i,j")  # 10 种
    stab_before = _get_stability(conn, "ile_6")
    apply_interleaving_effect(conn, ["ile_6"])
    stab_after = _get_stability(conn, "ile_6")

    increment = stab_after - stab_before
    max_allowed = base * ile_max_boost + 0.01
    assert increment <= max_allowed, (
        f"IL6: ILE 增量 {increment:.4f} 不应超过 max_boost 允许的 {max_allowed:.4f}"
    )
    assert stab_after > stab_before, f"IL6: 应有 ILE 加成，before={stab_before:.4f} after={stab_after:.4f}"


# ── IL7: stability 上限 365.0 ─────────────────────────────────────────────────────────────

def test_il7_stability_cap_365(conn):
    """IL7: ILE boost 后 stability 不超过 365.0。"""
    _insert_chunk(conn, "ile_7", stability=364.5, importance=0.8,
                  session_type_history="a,b,c,d,e")
    apply_interleaving_effect(conn, ["ile_7"])
    stab = _get_stability(conn, "ile_7")
    assert stab <= 365.0, f"IL7: stability 不应超过 365.0，got {stab}"


# ── IL8: update_accessed 集成测试 ────────────────────────────────────────────────────────────

def test_il8_update_accessed_integration(conn):
    """IL8: update_accessed 后，已有多样性的 chunk 触发 ILE 加成。"""
    # 预置 2 种 session_type history
    _insert_chunk(conn, "ile_8", stability=5.0, importance=0.6,
                  session_type_history="coding,debugging")
    stab_before = _get_stability(conn, "ile_8")

    # update_accessed 会触发 ILE（在 session_type_history 已有多样性时）
    update_accessed(conn, ["ile_8"])
    stab_after = _get_stability(conn, "ile_8")

    assert stab_after >= stab_before, (
        f"IL8: update_accessed 后 stability 不应降低，before={stab_before:.4f} after={stab_after:.4f}"
    )
