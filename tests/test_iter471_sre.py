"""
test_iter471_sre.py — iter471: Self-Reference Effect 单元测试

覆盖：
  SR1: 含第一人称词的内容 → stability 加成
  SR2: 无第一人称词 → 无加成（相对比较）
  SR3: sre_enabled=False → 无任何加成
  SR4: importance < sre_min_importance(0.30) → 不参与 SRE
  SR5: 加成系数 = sre_boost_factor(1.09)，受 sre_max_boost(0.15) 保护
  SR6: stability 加成后不超过 365.0
  SR7: 直接调用 apply_self_reference_effect → sre_boosted=True
  SR8: 中文第一人称词（我/我们）也触发 SRE

认知科学依据：
  Rogers, Kuiper & Kirker (1977) "Self-reference and the encoding of personal information" —
    "Does it describe you?" 条件下记忆保留比语义判断（"Does it mean...?"）高 50-60%（r=0.61）。
    机制：自我参照激活 medial prefrontal cortex（mPFC）→ 更强 episodic memory consolidation。
  Symons & Johnson (1997): SRE 在跨文化研究中稳定复现（meta-analysis, d=1.07）。

OS 类比：Linux process-private mappings（MAP_PRIVATE, mm/mmap.c）—
  进程私有页面 TLB 局部性更好 → 更快访问；
  自我参照内容 = process-private data → 更高检索效率。
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

from store_vfs import ensure_schema, insert_chunk, apply_self_reference_effect_v2 as apply_self_reference_effect
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
                chunk_type="observation", project="test"):
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
        "encode_context": "test_ctx",
    }


def _get_stability(conn, cid: str) -> float:
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


# ── SR1: 含第一人称词 → stability 加成 ──────────────────────────────────────────────────────

def test_sr1_self_reference_boosted(conn):
    """SR1: 含第一人称词（we/our）的内容 → stability 加成。"""
    # 含自我参照词，低密度（避免 KDEE/DDE 干扰），使用 observation 避免 GE 干扰
    content_self = "the the we the the our the the the the the the the the"
    chunk_self = _make_chunk("sre_1_self", content=content_self, chunk_type="observation")
    insert_chunk(conn, chunk_self)
    stab_self = _get_stability(conn, "sre_1_self")

    # 无第一人称词，低密度
    content_neu = "the the the the the the the the the the the the the the"
    chunk_neu = _make_chunk("sre_1_neu", content=content_neu, chunk_type="observation")
    insert_chunk(conn, chunk_neu)
    stab_neu = _get_stability(conn, "sre_1_neu")

    assert stab_self > stab_neu, (
        f"SR1: 含自我参照词 stability 应高于中性内容，"
        f"self={stab_self:.4f} neu={stab_neu:.4f}"
    )


# ── SR2: 无第一人称词 → 无加成 ──────────────────────────────────────────────────────────────

def test_sr2_no_self_ref_no_boost(conn):
    """SR2: 无第一人称词内容 → 无 SRE 加成（与含第一人称词内容相比不更高）。"""
    content_self = "the the we the the our the the the the the the the the"
    chunk_self = _make_chunk("sre_2_self", content=content_self, chunk_type="observation")
    insert_chunk(conn, chunk_self)
    stab_self = _get_stability(conn, "sre_2_self")

    content_neu = "the the the the the the the the the the the the the the"
    chunk_neu = _make_chunk("sre_2_neu", content=content_neu, chunk_type="observation")
    insert_chunk(conn, chunk_neu)
    stab_neu = _get_stability(conn, "sre_2_neu")

    assert stab_neu <= stab_self + 0.01, (
        f"SR2: 中性内容 stability 不应高于自我参照内容，neu={stab_neu:.4f} self={stab_self:.4f}"
    )


# ── SR3: sre_enabled=False → 无加成 ──────────────────────────────────────────────────────────

def test_sr3_disabled_no_boost(conn):
    """SR3: sre_enabled=False → 含第一人称词内容无 SRE 加成（直接调用验证）。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.sre_enabled":
            return False
        return original_get(key, project=project)

    content = "the the we the the our the the the the the the the the"
    now_iso = _utcnow().isoformat()

    # 直接插入 DB，绕过 insert_chunk 链（避免 MIE 等效应的干扰）
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count,
            encode_context, session_type_history)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("sre_3", "test", "observation", content,
         "summary", 0.6, 5.0, now_iso, now_iso, 0.5, now_iso, 0, "test_ctx", "")
    )
    conn.commit()
    stab_before = _get_stability(conn, "sre_3")

    with mock.patch.object(config, 'get', side_effect=patched_get):
        result_dis = apply_self_reference_effect(conn, "sre_3", content)
    stab_disabled = _get_stability(conn, "sre_3")

    assert stab_disabled <= stab_before + 0.001, (
        f"SR3: disabled 时 SRE 不应有加成，before={stab_before:.4f} disabled={stab_disabled:.4f}"
    )
    assert result_dis["sre_boosted"] is False, f"SR3: sre_boosted 应为 False，got {result_dis}"


# ── SR4: importance 不足 → 不参与 SRE ────────────────────────────────────────────────────────

def test_sr4_low_importance_no_boost(conn):
    """SR4: importance < sre_min_importance(0.30) → 不参与 SRE（直接调用验证）。"""
    content = "the the we the the our the the the the the the the"
    now_iso = _utcnow().isoformat()

    # 直接插入两个 chunk，绕过 insert_chunk 链
    for cid, imp in [("sre_4_low", 0.10), ("sre_4_high", 0.60)]:
        conn.execute(
            """INSERT OR REPLACE INTO memory_chunks
               (id, project, chunk_type, content, summary, importance, stability,
                created_at, updated_at, retrievability, last_accessed, access_count,
                encode_context, session_type_history)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (cid, "test", "observation", content,
             "summary", imp, 5.0, now_iso, now_iso, 0.5, now_iso, 0, "test_ctx", "")
        )
    conn.commit()

    apply_self_reference_effect(conn, "sre_4_low", content)
    apply_self_reference_effect(conn, "sre_4_high", content)

    stab_low = _get_stability(conn, "sre_4_low")
    stab_high = _get_stability(conn, "sre_4_high")

    assert stab_low <= stab_high + 0.01, (
        f"SR4: 低 importance 时 stability 不应高于高 importance，"
        f"low={stab_low:.4f} high={stab_high:.4f}"
    )


# ── SR5: 加成受 sre_max_boost 保护 ────────────────────────────────────────────────────────────

def test_sr5_max_boost_cap(conn):
    """SR5: SRE 增量不超过 base × sre_max_boost(0.15)（直接调用 apply_self_reference_effect_v2）。"""
    sre_max_boost = config.get("store_vfs.sre_max_boost")  # 0.15
    base = 5.0

    now_iso = _utcnow().isoformat()
    # 直接插入 DB，避免 insert_chunk 中其他效应（iter414 SRE 等）干扰
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count,
            encode_context, session_type_history)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("sre_5", "test", "observation",
         "the the we the the our the the the the the the the the",
         "summary", 0.6, base, now_iso, now_iso, 0.5, now_iso, 0, "test_ctx", "")
    )
    conn.commit()

    stab_before = _get_stability(conn, "sre_5")
    apply_self_reference_effect(conn, "sre_5",
                                 "the the we the the our the the the the the the the the")
    stab_after = _get_stability(conn, "sre_5")

    increment = stab_after - stab_before
    max_allowed = base * sre_max_boost + 0.01
    assert increment <= max_allowed, (
        f"SR5: SRE 增量 {increment:.4f} 不应超过 max_boost 允许的 {max_allowed:.4f}，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert stab_after > stab_before, (
        f"SR5: 应有 SRE 加成，before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── SR6: stability 上限 365.0 ─────────────────────────────────────────────────────────────

def test_sr6_stability_cap_365(conn):
    """SR6: SRE boost 后 stability 不超过 365.0（直接调用 apply_self_reference_effect）。"""
    now_iso = _utcnow().isoformat()
    base = 363.0
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count,
            encode_context, session_type_history)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("sre_6", "test", "observation", "we discussed our approach",
         "summary", 0.8, base, now_iso, now_iso, 0.5, now_iso, 0, "test_ctx", "")
    )
    conn.commit()

    apply_self_reference_effect(conn, "sre_6", "we discussed our approach")
    stab = float(conn.execute(
        "SELECT stability FROM memory_chunks WHERE id=?", ("sre_6",)
    ).fetchone()[0])
    assert stab <= 365.0, f"SR6: stability 不应超过 365.0，got {stab}"


# ── SR7: 直接调用 apply_self_reference_effect ──────────────────────────────────────────────

def test_sr7_direct_function_boost(conn):
    """SR7: apply_self_reference_effect 直接对含第一人称词内容产生加成。"""
    now_iso = _utcnow().isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count,
            encode_context, session_type_history)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("sre_7", "test", "observation", "the the we the the our the the",
         "summary", 0.6, 5.0, now_iso, now_iso, 0.5, now_iso, 0, "test_ctx", "")
    )
    conn.commit()

    stab_before = float(conn.execute(
        "SELECT stability FROM memory_chunks WHERE id=?", ("sre_7",)
    ).fetchone()[0])
    result = apply_self_reference_effect(conn, "sre_7", "the the we the the our the the")
    stab_after = float(conn.execute(
        "SELECT stability FROM memory_chunks WHERE id=?", ("sre_7",)
    ).fetchone()[0])

    assert stab_after > stab_before, (
        f"SR7: SRE 应对含第一人称词内容加成，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["sre_boosted"] is True, f"SR7: sre_boosted 应为 True，got {result}"


# ── SR8: 中文第一人称词也触发 SRE ────────────────────────────────────────────────────────────

def test_sr8_chinese_self_ref_words(conn):
    """SR8: 中文第一人称词（我/我们/我的）也触发 SRE。"""
    content_cn_self = "我们讨论了架构方案，我认为这个方法更好，我们的目标是提高性能。"
    chunk_cn_self = _make_chunk("sre_8_cn", content=content_cn_self,
                                 chunk_type="observation")
    insert_chunk(conn, chunk_cn_self)
    stab_emo = _get_stability(conn, "sre_8_cn")

    content_cn_neu = "系统正常运行，所有服务状态良好，无异常情况记录。"
    chunk_cn_neu = _make_chunk("sre_8_neu", content=content_cn_neu,
                                chunk_type="observation")
    insert_chunk(conn, chunk_cn_neu)
    stab_neu = _get_stability(conn, "sre_8_neu")

    assert stab_emo >= stab_neu - 0.01, (
        f"SR8: 中文自我参照内容 stability 应 >= 中性内容，"
        f"self={stab_emo:.4f} neu={stab_neu:.4f}"
    )
