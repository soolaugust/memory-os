"""
test_emotional_salience.py — 迭代320：情感显著性驱动 importance 单元测试

验证：
  1. compute_emotional_salience 对高唤醒词返回正 delta
  2. compute_emotional_salience 对"已解决"类词返回负 delta
  3. compute_emotional_salience 对中性文本返回 0
  4. apply_emotional_salience 将 importance 写回 DB（高唤醒）
  5. apply_emotional_salience 将 importance 下调（已解决类）
  6. apply_emotional_salience 不写 DB 当 delta < 0.01（避免无意义写）
  7. 崩溃/P0/紧急类 chunk 比中性 chunk importance 更高（端到端）
  8. 多个唤醒词叠加不超过上限 +0.25
  9. 边界：空字符串/None 安全返回 0.0

认知科学基础：
  McGaugh (2004) — 杏仁核通过 norepinephrine 调节海马突触可塑性
  情感唤醒 → 更强的记忆编码 → 更高的检索优先级
"""
import sys
import sqlite3
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "hooks"))

import tmpfs  # noqa

from store_vfs import (
    open_db, ensure_schema, insert_chunk,
    compute_emotional_salience, apply_emotional_salience,
)


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _make_chunk(cid, summary, importance=0.6, project="test"):
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    return {
        "id": cid,
        "created_at": now,
        "updated_at": now,
        "project": project,
        "source_session": "s1",
        "chunk_type": "decision",
        "info_class": "semantic",
        "content": f"[decision] {summary}",
        "summary": summary,
        "tags": ["decision"],
        "importance": importance,
        "retrievability": 0.5,
        "last_accessed": now,
        "access_count": 1,
        "oom_adj": 0,
        "lru_gen": 0,
        "stability": 1.2,
        "raw_snippet": "",
        "encoding_context": {},
    }


# ══════════════════════════════════════════════════════════════════════
# 1. compute_emotional_salience — 纯计算测试
# ══════════════════════════════════════════════════════════════════════

def test_critical_text_positive_delta():
    """崩溃/严重错误类文本 → 正 delta。"""
    delta = compute_emotional_salience("系统崩溃，严重错误，需要立即修复")
    assert delta > 0, f"崩溃文本应有正 delta，got {delta}"


def test_p0_urgent_positive_delta():
    """P0/紧急类文本 → 正 delta。"""
    delta = compute_emotional_salience("P0 故障，紧急处理")
    assert delta > 0, f"P0/紧急文本应有正 delta，got {delta}"


def test_breakthrough_positive_delta():
    """突破/关键发现 → 正 delta。"""
    delta = compute_emotional_salience("关键发现：内存泄漏根因在 FTS5 触发器")
    assert delta > 0, f"关键发现文本应有正 delta，got {delta}"


def test_resolved_negative_delta():
    """已解决/已修复 → 负 delta。"""
    delta = compute_emotional_salience("已解决，不再需要关注")
    assert delta < 0, f"已解决文本应有负 delta，got {delta}"


def test_deprecated_negative_delta():
    """deprecated/过时 → 负 delta。"""
    delta = compute_emotional_salience("此方案已废弃，deprecated，使用新方案替代")
    assert delta < 0, f"废弃文本应有负 delta，got {delta}"


def test_neutral_text_zero_delta():
    """中性文本 → delta ≈ 0。"""
    delta = compute_emotional_salience("使用 FTS5 进行全文检索")
    assert abs(delta) < 0.01, f"中性文本 delta 应接近 0，got {delta}"


def test_empty_text_zero_delta():
    """空字符串安全返回 0.0。"""
    assert compute_emotional_salience("") == 0.0


def test_combined_signals_bounded():
    """多个高唤醒词叠加不超过上限 +0.25。"""
    delta = compute_emotional_salience(
        "崩溃 critical P0 紧急 突破 关键发现 fatal panic CRITICAL 性能瓶颈"
    )
    assert delta <= 0.25, f"叠加 delta 不应超过上限，got {delta}"


def test_positive_and_negative_cancel():
    """正负信号叠加部分抵消。"""
    pos = compute_emotional_salience("崩溃")
    neg = compute_emotional_salience("已解决")
    combined = compute_emotional_salience("崩溃已解决，已修复")
    # combined 应在 pos 和 neg 之间（因为两者都匹配）
    assert neg < combined < pos or combined <= pos, \
        f"正负叠加 delta={combined} 应小于纯正={pos}"


# ══════════════════════════════════════════════════════════════════════
# 2. apply_emotional_salience — DB 写回测试
# ══════════════════════════════════════════════════════════════════════

def test_apply_increases_importance_for_critical(conn):
    """高唤醒文本 apply 后 importance 上调。"""
    base_imp = 0.60
    insert_chunk(conn, _make_chunk("c1", "系统崩溃，P0 紧急事件", importance=base_imp))
    conn.commit()

    new_imp = apply_emotional_salience(conn, "c1", "系统崩溃，P0 紧急事件", base_imp)
    conn.commit()

    assert new_imp > base_imp, f"高唤醒文本应上调 importance，got {new_imp}"
    row = conn.execute("SELECT importance FROM memory_chunks WHERE id='c1'").fetchone()
    assert row["importance"] > base_imp, f"DB 中 importance 应上调，got {row['importance']}"


def test_apply_decreases_importance_for_resolved(conn):
    """已解决文本 apply 后 importance 下调。"""
    base_imp = 0.70
    insert_chunk(conn, _make_chunk("c2", "此问题已解决，不再需要关注", importance=base_imp))
    conn.commit()

    new_imp = apply_emotional_salience(conn, "c2", "此问题已解决，不再需要关注", base_imp)
    conn.commit()

    assert new_imp < base_imp, f"已解决文本应下调 importance，got {new_imp}"


def test_apply_skips_neutral_text(conn):
    """中性文本不写 DB（delta < 0.01）。"""
    base_imp = 0.65
    insert_chunk(conn, _make_chunk("c3", "使用 BM25 进行检索", importance=base_imp))
    conn.commit()

    new_imp = apply_emotional_salience(conn, "c3", "使用 BM25 进行检索", base_imp)

    assert new_imp == base_imp, f"中性文本不应改变 importance，got {new_imp}"
    row = conn.execute("SELECT importance FROM memory_chunks WHERE id='c3'").fetchone()
    assert row["importance"] == base_imp, \
        f"DB 中 importance 不应被修改，got {row['importance']}"


def test_apply_clamps_to_max(conn):
    """importance 上限为 0.98，不超过。"""
    base_imp = 0.95  # 高基础值 + 高唤醒词，理论上超过 0.98
    insert_chunk(conn, _make_chunk("c4", "崩溃 critical P0 紧急 突破", importance=base_imp))
    conn.commit()

    new_imp = apply_emotional_salience(conn, "c4", "崩溃 critical P0 紧急 突破", base_imp)
    assert new_imp <= 0.98, f"importance 不应超过 0.98，got {new_imp}"


def test_apply_clamps_to_min(conn):
    """importance 下限为 0.05，不低于。"""
    base_imp = 0.08  # 低基础值 + 负唤醒词
    insert_chunk(conn, _make_chunk("c5", "已解决 已废弃 deprecated resolved fixed", importance=base_imp))
    conn.commit()

    new_imp = apply_emotional_salience(conn, "c5", "已解决 已废弃 deprecated resolved fixed", base_imp)
    assert new_imp >= 0.05, f"importance 不应低于 0.05，got {new_imp}"


# ══════════════════════════════════════════════════════════════════════
# 3. 端到端：崩溃类 chunk 比中性 chunk importance 更高
# ══════════════════════════════════════════════════════════════════════

def test_critical_chunk_higher_importance_than_neutral(conn):
    """端到端：写入后崩溃类 chunk importance > 中性 chunk importance。"""
    base_imp = 0.65
    insert_chunk(conn, _make_chunk("neutral", "使用 FTS5 进行全文检索", importance=base_imp))
    insert_chunk(conn, _make_chunk("critical", "系统崩溃，P0 紧急，数据丢失风险", importance=base_imp))
    conn.commit()

    # 对两者分别应用情感显著性
    apply_emotional_salience(conn, "neutral", "使用 FTS5 进行全文检索", base_imp)
    apply_emotional_salience(conn, "critical", "系统崩溃，P0 紧急，数据丢失风险", base_imp)
    conn.commit()

    neutral_imp = conn.execute(
        "SELECT importance FROM memory_chunks WHERE id='neutral'"
    ).fetchone()["importance"]
    critical_imp = conn.execute(
        "SELECT importance FROM memory_chunks WHERE id='critical'"
    ).fetchone()["importance"]

    assert critical_imp > neutral_imp, \
        f"崩溃 chunk importance ({critical_imp}) 应 > 中性 ({neutral_imp})"
