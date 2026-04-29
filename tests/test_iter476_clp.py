"""
test_iter476_clp.py — iter476: Cognitive Load Penalty 单元测试

覆盖：
  CL1: 词数 > clp_max_tokens(200) → stability 降低
  CL2: 词数 <= clp_max_tokens → 无惩罚
  CL3: clp_enabled=False → 无惩罚
  CL4: importance < clp_min_importance(0.30) → 不参与 CLP
  CL5: 惩罚幅度随词数增加而增大（词数越多惩罚越大）
  CL6: 惩罚受 clp_max_penalty(0.15) 上限保护
  CL7: stability 惩罚后不低于 0.1（下限保护）
  CL8: 直接调用 apply_cognitive_load_penalty → clp_penalized=True

认知科学依据：
  Miller (1956) "The magical number seven, plus or minus two" —
    工作记忆容量 7±2 chunks；超出时编码质量下降。
  Sweller (1988) Cognitive Load Theory: 内在负荷过高 → 有效编码降低。
  Paas & van Merriënboer (1994): 高负荷材料的长期记忆保留率反而更低。
  与 DDE 互补：短且复杂 = 有益困难 → stability+；长且复杂 = 认知超载 → stability−。

OS 类比：CPU context switch overhead（kernel/sched/core.c）—
  超线程数过多时调度开销超过并行收益；TLB flush 频率 ∝ 活跃进程数 → effective IPC 下降。
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

from store_vfs import ensure_schema, apply_cognitive_load_penalty
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


def _insert_raw(conn, cid, content, importance=0.6, stability=5.0, chunk_type="observation"):
    now_iso = _utcnow().isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count,
            encode_context, session_type_history)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (cid, "test", chunk_type, content, "summary", importance, stability,
         now_iso, now_iso, 0.5, now_iso, 0, "test_ctx", "")
    )
    conn.commit()


def _make_long_content(word_count: int) -> str:
    """生成指定词数的内容（用不重复词避免 KDEE 干扰，用低情绪词）。"""
    base_words = ["the", "a", "an", "is", "was", "are", "be", "been", "being",
                  "has", "have", "had", "do", "does", "did", "will", "would",
                  "could", "should", "may", "might", "shall", "can", "need"]
    words = []
    for i in range(word_count):
        words.append(base_words[i % len(base_words)])
    return " ".join(words)


def _get_stability(conn, cid: str) -> float:
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


# ── CL1: 词数超过阈值 → stability 降低 ───────────────────────────────────────────────────

def test_cl1_long_content_penalized(conn):
    """CL1: 词数 > clp_max_tokens(200) → stability 降低。"""
    long_content = _make_long_content(300)  # 超过 200 词
    _insert_raw(conn, "cl1", long_content, stability=5.0)

    stab_before = _get_stability(conn, "cl1")
    result = apply_cognitive_load_penalty(conn, "cl1", long_content)
    stab_after = _get_stability(conn, "cl1")

    assert stab_after < stab_before, (
        f"CL1: 超长内容 stability 应降低，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["clp_penalized"] is True, f"CL1: clp_penalized 应为 True，got {result}"


# ── CL2: 词数未超阈值 → 无惩罚 ───────────────────────────────────────────────────────────

def test_cl2_short_content_no_penalty(conn):
    """CL2: 词数 <= clp_max_tokens(200) → 无 CLP 惩罚。"""
    short_content = _make_long_content(100)  # 100 词，远未超阈值
    _insert_raw(conn, "cl2", short_content, stability=5.0)

    stab_before = _get_stability(conn, "cl2")
    result = apply_cognitive_load_penalty(conn, "cl2", short_content)
    stab_after = _get_stability(conn, "cl2")

    assert abs(stab_after - stab_before) < 0.001, (
        f"CL2: 短内容不应有 CLP 惩罚，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["clp_penalized"] is False, f"CL2: clp_penalized 应为 False，got {result}"


# ── CL3: clp_enabled=False → 无惩罚 ──────────────────────────────────────────────────────

def test_cl3_disabled_no_penalty(conn):
    """CL3: clp_enabled=False → 超长内容无 CLP 惩罚。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.clp_enabled":
            return False
        return original_get(key, project=project)

    long_content = _make_long_content(300)
    _insert_raw(conn, "cl3", long_content, stability=5.0)

    stab_before = _get_stability(conn, "cl3")
    with mock.patch.object(config, 'get', side_effect=patched_get):
        result = apply_cognitive_load_penalty(conn, "cl3", long_content)
    stab_after = _get_stability(conn, "cl3")

    assert abs(stab_after - stab_before) < 0.001, (
        f"CL3: disabled 时不应有惩罚，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["clp_penalized"] is False, f"CL3: clp_penalized 应为 False"


# ── CL4: importance 不足 → 不参与 CLP ────────────────────────────────────────────────────

def test_cl4_low_importance_no_penalty(conn):
    """CL4: importance < clp_min_importance(0.30) → 不参与 CLP。"""
    long_content = _make_long_content(300)
    _insert_raw(conn, "cl4", long_content, importance=0.10, stability=5.0)

    stab_before = _get_stability(conn, "cl4")
    result = apply_cognitive_load_penalty(conn, "cl4", long_content)
    stab_after = _get_stability(conn, "cl4")

    assert abs(stab_after - stab_before) < 0.001, (
        f"CL4: 低 importance 不应触发 CLP，before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── CL5: 词数越多惩罚越大 ─────────────────────────────────────────────────────────────────

def test_cl5_more_words_more_penalty(conn):
    """CL5: 词数越多，stability 降低越多（惩罚单调递增）。"""
    content_300 = _make_long_content(300)
    content_500 = _make_long_content(500)

    _insert_raw(conn, "cl5_300", content_300, stability=5.0)
    _insert_raw(conn, "cl5_500", content_500, stability=5.0)

    apply_cognitive_load_penalty(conn, "cl5_300", content_300)
    apply_cognitive_load_penalty(conn, "cl5_500", content_500)

    stab_300 = _get_stability(conn, "cl5_300")
    stab_500 = _get_stability(conn, "cl5_500")

    assert stab_500 <= stab_300 + 0.001, (
        f"CL5: 500 词惩罚应 >= 300 词惩罚，stab_300={stab_300:.4f} stab_500={stab_500:.4f}"
    )


# ── CL6: 惩罚受 clp_max_penalty 上限保护 ─────────────────────────────────────────────────

def test_cl6_max_penalty_cap(conn):
    """CL6: CLP 惩罚不超过 base × clp_max_penalty(0.15)。"""
    clp_max_penalty = config.get("store_vfs.clp_max_penalty")  # 0.15
    base = 5.0

    # 极长内容（1000 词）
    very_long_content = _make_long_content(1000)
    _insert_raw(conn, "cl6", very_long_content, stability=base)

    stab_before = _get_stability(conn, "cl6")
    apply_cognitive_load_penalty(conn, "cl6", very_long_content)
    stab_after = _get_stability(conn, "cl6")

    penalty = stab_before - stab_after
    max_allowed = base * clp_max_penalty + 0.01
    assert penalty <= max_allowed, (
        f"CL6: 惩罚 {penalty:.4f} 不应超过 max_penalty 允许的 {max_allowed:.4f}，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert stab_after < stab_before, f"CL6: 应有 CLP 惩罚，before={stab_before:.4f} after={stab_after:.4f}"


# ── CL7: stability 下限 0.1 保护 ─────────────────────────────────────────────────────────

def test_cl7_stability_floor_01(conn):
    """CL7: CLP 惩罚后 stability 不低于 0.1（下限保护）。"""
    # 极低 stability + 极长内容
    very_long_content = _make_long_content(1000)
    _insert_raw(conn, "cl7", very_long_content, stability=0.15)

    apply_cognitive_load_penalty(conn, "cl7", very_long_content)
    stab = _get_stability(conn, "cl7")
    assert stab >= 0.1, f"CL7: stability 不应低于 0.1，got {stab:.4f}"


# ── CL8: 直接调用 apply_cognitive_load_penalty ───────────────────────────────────────────

def test_cl8_direct_function_penalty(conn):
    """CL8: apply_cognitive_load_penalty 直接对超长内容产生惩罚，返回 clp_penalized=True。"""
    long_content = _make_long_content(400)
    _insert_raw(conn, "cl8", long_content, stability=5.0)

    stab_before = _get_stability(conn, "cl8")
    result = apply_cognitive_load_penalty(conn, "cl8", long_content)
    stab_after = _get_stability(conn, "cl8")

    assert stab_after < stab_before, (
        f"CL8: 直接调用应产生惩罚，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["clp_penalized"] is True, f"CL8: clp_penalized 应为 True，got {result}"
    assert result["clp_token_count"] >= 400, (
        f"CL8: clp_token_count 应 >= 400，got {result['clp_token_count']}"
    )
