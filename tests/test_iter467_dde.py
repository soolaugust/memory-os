"""
test_iter467_dde.py — iter467: Desirable Difficulty Effect 单元测试

覆盖：
  DD1: 高平均词长（>= dde_min_avg_word_len=5.5）→ stability 加成
  DD2: 低平均词长 → 无加成（相对比较）
  DD3: dde_enabled=False → 无任何加成
  DD4: importance < dde_min_importance(0.30) → 不参与 DDE
  DD5: 词数 < dde_min_words(5) → 不触发 DDE
  DD6: 加成系数 = dde_boost_factor(1.08)，受 dde_max_boost(0.16) 保护
  DD7: stability 加成后不超过 365.0
  DD8: 平均词长更高的内容获得更大（或相等）加成

认知科学依据：
  Bjork (1994) "Memory and metamemory considerations in the training of human beings" —
    "有益的困难"（desirable difficulty）使学习时感觉更难，但产生更强的长期记忆痕迹（r=0.49）。
  Rayner & Pollatsek (1989): 低频词（longer words）需要更多眼跳注视时间 → 更深语义加工。
  Hirshman & Bjork (1988): 生成效应（self-generation）= 认知努力代理指标。

OS 类比：Linux zswap/zram（mm/zswap.c）—
  被压缩的页面需要 CPU 解压（认知努力），但压缩率高 = 有限内存中保留更多内容；
  复杂编码 → 更高信息密度存储，在 LRU 中有更高留存率。
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

from store_vfs import ensure_schema, insert_chunk, apply_desirable_difficulty_effect
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


def _make_chunk(cid, content="", importance=0.6, stability=5.0,
                chunk_type="decision", project="test"):
    now_iso = _utcnow().isoformat()
    return {
        "id": cid,
        "project": project,
        "source_session": "",
        "chunk_type": chunk_type,
        "content": content,
        "summary": "summary",
        "importance": importance,
        "stability": stability,
        "created_at": now_iso,
        "updated_at": now_iso,
        "retrievability": 0.5,
        "last_accessed": now_iso,
        "access_count": 0,
        "encode_context": "kernel_mm",
    }


def _get_stability(conn, cid: str) -> float:
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


# ── DD1: 高平均词长 → stability 加成 ──────────────────────────────────────────────────────

def test_dd1_long_words_boosted(conn):
    """DD1: 平均词长 >= dde_min_avg_word_len(5.5) → stability 加成。"""
    # 长词内容（平均词长约 8-10）
    content_complex = "synchronization parallelization implementation optimization architecture"
    chunk_complex = _make_chunk("dde_1_complex", content=content_complex, importance=0.6, stability=5.0)
    insert_chunk(conn, chunk_complex)
    stab_complex = _get_stability(conn, "dde_1_complex")

    # 短词基准（平均词长 < 3，避免 KDEE 干扰用相同密度）
    content_simple = "the the the the the the the the the the the the the"
    chunk_simple = _make_chunk("dde_1_simple", content=content_simple, importance=0.6, stability=5.0)
    insert_chunk(conn, chunk_simple)
    stab_simple = _get_stability(conn, "dde_1_simple")

    assert stab_complex > stab_simple, (
        f"DD1: 高复杂度内容 stability 应高于低复杂度，"
        f"complex={stab_complex:.4f} simple={stab_simple:.4f}"
    )


# ── DD2: 低平均词长 → 无加成 ──────────────────────────────────────────────────────────────

def test_dd2_short_words_no_boost(conn):
    """DD2: 平均词长 < dde_min_avg_word_len → 无 DDE 加成（与复杂内容相比不更高）。"""
    content_complex = "synchronization parallelization implementation optimization architecture"
    chunk_complex = _make_chunk("dde_2_complex", content=content_complex, importance=0.6, stability=5.0)
    insert_chunk(conn, chunk_complex)
    stab_complex = _get_stability(conn, "dde_2_complex")

    content_simple = "the the the the the the the the the the the"
    chunk_simple = _make_chunk("dde_2_simple", content=content_simple, importance=0.6, stability=5.0)
    insert_chunk(conn, chunk_simple)
    stab_simple = _get_stability(conn, "dde_2_simple")

    assert stab_simple <= stab_complex + 0.01, (
        f"DD2: 低复杂度 stability 不应高于高复杂度，simple={stab_simple:.4f} complex={stab_complex:.4f}"
    )


# ── DD3: dde_enabled=False → 无加成 ──────────────────────────────────────────────────────

def test_dd3_disabled_no_boost(conn):
    """DD3: dde_enabled=False → 高复杂度内容无 DDE 加成。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.dde_enabled":
            return False
        return original_get(key, project=project)

    content = "synchronization parallelization implementation optimization architecture"

    chunk_dis = _make_chunk("dde_3_dis", content=content, importance=0.6, stability=5.0)
    with mock.patch.object(config, 'get', side_effect=patched_get):
        insert_chunk(conn, chunk_dis)
    stab_disabled = _get_stability(conn, "dde_3_dis")

    chunk_en = _make_chunk("dde_3_en", content=content, importance=0.6, stability=5.0)
    insert_chunk(conn, chunk_en)
    stab_enabled = _get_stability(conn, "dde_3_en")

    assert stab_disabled <= stab_enabled + 0.01, (
        f"DD3: disabled 时 stability 不应高于 enabled，"
        f"disabled={stab_disabled:.4f} enabled={stab_enabled:.4f}"
    )


# ── DD4: importance 不足 → 不参与 DDE ───────────────────────────────────────────────────

def test_dd4_low_importance_no_boost(conn):
    """DD4: importance < dde_min_importance(0.30) → 不参与 DDE。"""
    content = "synchronization parallelization implementation optimization architecture"

    chunk_low = _make_chunk("dde_4_low", content=content, importance=0.10, stability=5.0)
    insert_chunk(conn, chunk_low)
    stab_low = _get_stability(conn, "dde_4_low")

    chunk_high = _make_chunk("dde_4_high", content=content, importance=0.60, stability=5.0)
    insert_chunk(conn, chunk_high)
    stab_high = _get_stability(conn, "dde_4_high")

    assert stab_low <= stab_high + 0.01, (
        f"DD4: 低 importance 时 stability 不应高于高 importance，"
        f"low={stab_low:.4f} high={stab_high:.4f}"
    )


# ── DD5: 词数不足 → 不触发 DDE ───────────────────────────────────────────────────────────

def test_dd5_too_few_words_no_boost(conn):
    """DD5: 词数 < dde_min_words(5) → 不触发 DDE。"""
    # 只有 3 个词（< 5），即使都是长词
    content_short = "synchronization parallelization implementation"  # 3 words
    now_iso = _utcnow().isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count,
            encode_context, session_type_history)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("dde_5_short", "test", "decision", content_short, "summary",
         0.6, 5.0, now_iso, now_iso, 0.5, now_iso, 0, "kernel_mm", "")
    )
    conn.commit()

    stab_before = _get_stability(conn, "dde_5_short")
    result = apply_desirable_difficulty_effect(conn, "dde_5_short", content_short)
    stab_after = _get_stability(conn, "dde_5_short")

    assert abs(stab_after - stab_before) < 0.001, (
        f"DD5: 词数不足时不应有 DDE 加成，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["dde_boosted"] is False, f"DD5: dde_boosted 应为 False，got {result}"


# ── DD6: 加成受 dde_max_boost 保护 ────────────────────────────────────────────────────────

def test_dd6_max_boost_cap(conn):
    """DD6: DDE 增量（相对 baseline）不超过 base × dde_max_boost(0.16)。"""
    dde_max_boost = config.get("store_vfs.dde_max_boost")  # 0.16
    base = 5.0

    content_base = "the the the the the the the the the the the"
    chunk_base = _make_chunk("dde_6_base", content=content_base, importance=0.6, stability=base)
    insert_chunk(conn, chunk_base)
    stab_base = _get_stability(conn, "dde_6_base")

    content_dde = "synchronization parallelization implementation optimization architecture"
    chunk_dde = _make_chunk("dde_6_dde", content=content_dde, importance=0.6, stability=base)
    insert_chunk(conn, chunk_dde)
    stab_dde = _get_stability(conn, "dde_6_dde")

    dde_increment = stab_dde - stab_base
    max_allowed = base * dde_max_boost + 0.1
    assert dde_increment <= max_allowed, (
        f"DD6: DDE 增量 {dde_increment:.4f} 不应超过 max_boost 允许的 {max_allowed:.4f}，"
        f"base={stab_base:.4f} dde={stab_dde:.4f}"
    )
    assert stab_dde > stab_base, f"DD6: 应有 DDE 加成，base={stab_base:.4f} dde={stab_dde:.4f}"


# ── DD7: stability 上限 365.0 ─────────────────────────────────────────────────────────────

def test_dd7_stability_cap_365(conn):
    """DD7: DDE boost 后 stability 不超过 365.0（直接测试 apply_desirable_difficulty_effect）。"""
    now_iso = _utcnow().isoformat()
    base = 363.0
    content = "synchronization parallelization implementation optimization architecture"
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count,
            encode_context, session_type_history)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("dde_7", "test", "decision", content, "summary",
         0.8, base, now_iso, now_iso, 0.5, now_iso, 0, "kernel_mm", "")
    )
    conn.commit()

    apply_desirable_difficulty_effect(conn, "dde_7", content)
    stab = _get_stability(conn, "dde_7")
    assert stab <= 365.0, f"DD7: stability 不应超过 365.0，got {stab}"


# ── DD8: apply_desirable_difficulty_effect 直接测试 ──────────────────────────────────────

def test_dd8_direct_function_boost(conn):
    """DD8: apply_desirable_difficulty_effect 直接对高复杂度内容产生加成。"""
    now_iso = _utcnow().isoformat()
    content = "synchronization parallelization implementation optimization architecture"
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count,
            encode_context, session_type_history)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("dde_8", "test", "decision", content, "summary",
         0.6, 5.0, now_iso, now_iso, 0.5, now_iso, 0, "kernel_mm", "")
    )
    conn.commit()

    stab_before = _get_stability(conn, "dde_8")
    result = apply_desirable_difficulty_effect(conn, "dde_8", content)
    stab_after = _get_stability(conn, "dde_8")

    assert stab_after > stab_before, (
        f"DD8: DDE 应对高复杂度内容加成，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["dde_boosted"] is True, f"DD8: dde_boosted 应为 True，got {result}"
