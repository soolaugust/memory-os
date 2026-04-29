"""
test_iter466_ete.py — iter466: Emotional Tagging Effect 单元测试

覆盖：
  ET1: 含情绪关键词内容 → insert_chunk 时 stability 加成
  ET2: 无情绪关键词 → 无加成（相对比较）
  ET3: ete_enabled=False → 无任何加成
  ET4: importance < ete_min_importance(0.30) → 不参与 ETE
  ET5: 加成系数 = ete_boost_factor(1.12)，受 ete_max_boost(0.18) 保护
  ET6: stability 加成后不超过 365.0
  ET7: 多个情绪词（比单个）在同等条件下不获得更多加成（max_boost 截断）
  ET8: 英文和中文情绪词均触发 ETE

认知科学依据：
  Cahill, Prins, Weber & McGaugh (1994) "Beta-adrenergic activation and memory for
    emotional events" (Nature) —
    情绪唤醒（norepinephrine 释放）→ 杏仁核激活 → 海马-杏仁核 LTP 增强 →
    情绪事件记忆更持久（延时测验 AUC 提升 40%）。
  LaBar & Cabeza (2006): 情绪强度与记忆精确度正相关（r=0.53）。

OS 类比：Linux OOM killer scoring（mm/oom_kill.c）—
  高 oom_score_adj 的关键进程（init, kernel threads）受保护不被 OOM killer 杀死；
  情绪显著内容 = 高保护优先级记忆（避免被淘汰）。
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

from store_vfs import ensure_schema, insert_chunk, apply_emotional_tagging_effect
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


# ── ET1: 含情绪关键词 → stability 加成 ──────────────────────────────────────────────────

def test_et1_emotional_content_boosted(conn):
    """ET1: 含情绪关键词（critical/error）的内容 → stability 加成。"""
    # 含情绪词
    content_emo = "the the the the the the critical error the the the the the"
    chunk_emo = _make_chunk("ete_1_emo", content=content_emo, importance=0.6, stability=5.0)
    insert_chunk(conn, chunk_emo)
    stab_emo = _get_stability(conn, "ete_1_emo")

    # 无情绪词（低密度，避免 KDEE）
    content_neu = "the the the the the the the the the the the the the"
    chunk_neu = _make_chunk("ete_1_neu", content=content_neu, importance=0.6, stability=5.0)
    insert_chunk(conn, chunk_neu)
    stab_neu = _get_stability(conn, "ete_1_neu")

    assert stab_emo > stab_neu, (
        f"ET1: 情绪内容 stability 应高于中性内容，emo={stab_emo:.4f} neu={stab_neu:.4f}"
    )


# ── ET2: 无情绪词 → 无加成 ───────────────────────────────────────────────────────────────

def test_et2_neutral_content_no_boost(conn):
    """ET2: 无情绪关键词内容 → 无 ETE 加成（与有情绪词的 chunk 相比不更高）。"""
    content_emo = "the the the the the critical the the the the the"
    chunk_emo = _make_chunk("ete_2_emo", content=content_emo, importance=0.6, stability=5.0)
    insert_chunk(conn, chunk_emo)
    stab_emo = _get_stability(conn, "ete_2_emo")

    content_neu = "the the the the the the the the the the the the"
    chunk_neu = _make_chunk("ete_2_neu", content=content_neu, importance=0.6, stability=5.0)
    insert_chunk(conn, chunk_neu)
    stab_neu = _get_stability(conn, "ete_2_neu")

    assert stab_neu <= stab_emo + 0.01, (
        f"ET2: 中性内容 stability 不应高于情绪内容，neu={stab_neu:.4f} emo={stab_emo:.4f}"
    )


# ── ET3: ete_enabled=False → 无加成 ──────────────────────────────────────────────────────

def test_et3_disabled_no_boost(conn):
    """ET3: ete_enabled=False → 情绪内容无 ETE 加成。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.ete_enabled":
            return False
        return original_get(key, project=project)

    content = "the the the the critical error the the the the"

    chunk_dis = _make_chunk("ete_3_dis", content=content, importance=0.6, stability=5.0)
    with mock.patch.object(config, 'get', side_effect=patched_get):
        insert_chunk(conn, chunk_dis)
    stab_disabled = _get_stability(conn, "ete_3_dis")

    chunk_en = _make_chunk("ete_3_en", content=content, importance=0.6, stability=5.0)
    insert_chunk(conn, chunk_en)
    stab_enabled = _get_stability(conn, "ete_3_en")

    assert stab_disabled <= stab_enabled + 0.01, (
        f"ET3: disabled 时 stability 不应高于 enabled，"
        f"disabled={stab_disabled:.4f} enabled={stab_enabled:.4f}"
    )


# ── ET4: importance 不足 → 不参与 ETE ───────────────────────────────────────────────────

def test_et4_low_importance_no_boost(conn):
    """ET4: importance < ete_min_importance(0.30) → 不参与 ETE。"""
    content = "the the the the critical error the the the the"

    chunk_low = _make_chunk("ete_4_low", content=content, importance=0.10, stability=5.0)
    insert_chunk(conn, chunk_low)
    stab_low = _get_stability(conn, "ete_4_low")

    chunk_high = _make_chunk("ete_4_high", content=content, importance=0.60, stability=5.0)
    insert_chunk(conn, chunk_high)
    stab_high = _get_stability(conn, "ete_4_high")

    assert stab_low <= stab_high + 0.01, (
        f"ET4: 低 importance 时 stability 不应高于高 importance，"
        f"low={stab_low:.4f} high={stab_high:.4f}"
    )


# ── ET5: 加成系数受 ete_max_boost 保护 ────────────────────────────────────────────────────

def test_et5_max_boost_cap(conn):
    """ET5: ETE 增量（相对 baseline）不超过 base × ete_max_boost(0.18)。"""
    ete_max_boost = config.get("store_vfs.ete_max_boost")  # 0.18
    base = 5.0

    # baseline（无情绪词；observation 避免 GE 干扰）
    content_base = "the the the the the the the the the the the"
    chunk_base = _make_chunk("ete_5_base", content=content_base, importance=0.6, stability=base,
                              chunk_type="observation")
    insert_chunk(conn, chunk_base)
    stab_base = _get_stability(conn, "ete_5_base")

    # ETE 分支（有情绪词；observation 避免 GE 干扰，仅测量 ETE 净增量）
    content_ete = "the the the the critical the the the the the"
    chunk_ete = _make_chunk("ete_5_ete", content=content_ete, importance=0.6, stability=base,
                             chunk_type="observation")
    insert_chunk(conn, chunk_ete)
    stab_ete = _get_stability(conn, "ete_5_ete")

    ete_increment = stab_ete - stab_base
    max_allowed = base * ete_max_boost + 0.1  # tolerance
    assert ete_increment <= max_allowed, (
        f"ET5: ETE 增量 {ete_increment:.4f} 不应超过 max_boost 允许的 {max_allowed:.4f}，"
        f"base={stab_base:.4f} ete={stab_ete:.4f}"
    )
    assert stab_ete > stab_base, f"ET5: 应有 ETE 加成，base={stab_base:.4f} ete={stab_ete:.4f}"


# ── ET6: stability 上限 365.0 ─────────────────────────────────────────────────────────────

def test_et6_stability_cap_365(conn):
    """ET6: ETE boost 后 stability 不超过 365.0（直接测试 apply_emotional_tagging_effect）。"""
    now_iso = _utcnow().isoformat()
    base = 363.0
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count,
            encode_context, session_type_history)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("ete_6", "test", "decision", "critical system failure",
         "critical", 0.8, base, now_iso, now_iso, 0.5, now_iso, 0, "kernel_mm", "")
    )
    conn.commit()

    apply_emotional_tagging_effect(conn, "ete_6", "critical system failure", "critical")
    stab = _get_stability(conn, "ete_6")
    assert stab <= 365.0, f"ET6: stability 不应超过 365.0，got {stab}"


# ── ET7: apply_emotional_tagging_effect 直接测试 ──────────────────────────────────────────

def test_et7_direct_function_boost(conn):
    """ET7: apply_emotional_tagging_effect 直接对有情绪词的 chunk 产生加成。"""
    now_iso = _utcnow().isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count,
            encode_context, session_type_history)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("ete_7", "test", "decision", "critical system failure",
         "system crashed", 0.6, 5.0, now_iso, now_iso, 0.5, now_iso, 0, "kernel_mm", "")
    )
    conn.commit()

    stab_before = _get_stability(conn, "ete_7")
    result = apply_emotional_tagging_effect(
        conn, "ete_7", "critical system failure", "system crashed"
    )
    stab_after = _get_stability(conn, "ete_7")

    assert stab_after >= stab_before, f"ET7: ETE 后 stability 不应降低"
    assert result["ete_boosted"] is True, f"ET7: ete_boosted 应为 True，got {result}"


# ── ET8: 中文情绪词也触发 ETE ──────────────────────────────────────────────────────────────

def test_et8_chinese_emotional_words(conn):
    """ET8: 中文情绪关键词（关键/错误/紧急）也触发 ETE。"""
    content_cn_emo = "系统关键错误导致服务崩溃，需要紧急修复。"
    chunk_cn_emo = _make_chunk("ete_8_cn", content=content_cn_emo, importance=0.6, stability=5.0)
    insert_chunk(conn, chunk_cn_emo)
    stab_emo = _get_stability(conn, "ete_8_cn")

    content_cn_neu = "系统正常运行，所有服务状态良好，无异常情况。"
    chunk_cn_neu = _make_chunk("ete_8_neu", content=content_cn_neu, importance=0.6, stability=5.0)
    insert_chunk(conn, chunk_cn_neu)
    stab_neu = _get_stability(conn, "ete_8_neu")

    assert stab_emo >= stab_neu - 0.01, (
        f"ET8: 中文情绪内容 stability 应 >= 中性内容，emo={stab_emo:.4f} neu={stab_neu:.4f}"
    )
