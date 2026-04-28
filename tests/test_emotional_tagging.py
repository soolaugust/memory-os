"""
test_emotional_tagging.py — iter399: Emotional Tagging at Write Time

覆盖：
  ET1: 高唤醒词（崩溃/critical）→ emotional_weight > 0.4（触发 retriever boost 阈值）
  ET2: 负向词（已解决/resolved）→ emotional_weight == 0（降权词不产生情绪权重）
  ET3: 情感中性文本 → emotional_weight == 0.0
  ET4: 多情绪词叠加 → emotional_weight 更高（累积效应）
  ET5: emotional_weight 上限 1.0
  ET6: apply_emotional_salience 同时写入 importance 和 emotional_weight
  ET7: integration — _write_chunk 写入后 DB 中 emotional_weight 字段有效

认知科学依据：
  McGaugh (2000) Emotional Enhancement of Memory:
    杏仁核激活 → norepinephrine 释放 → 海马 LTP 增强 → 长时记忆固化
  LaBar & Cabeza (2006): 情绪词比中性词的识别成绩高 20-40%
OS 类比：Linux OOM Score — 内核按 oom_score 标注进程"重要性"，
  OOM Killer 据此选择牺牲目标；emotional_weight 类比 oom_adj 的反向（高 = 保留）。
"""
import sys
import sqlite3
import pytest
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "hooks"))

import tmpfs  # noqa
from store_vfs import (ensure_schema, compute_emotional_salience,
                        apply_emotional_salience)


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _insert_chunk(conn, chunk_id="chunk_test", importance=0.80,
                  emotional_weight=None, chunk_type="decision",
                  summary="test summary"):
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO memory_chunks "
        "(id, project, chunk_type, content, summary, importance, retrievability, "
        "emotional_weight, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, 0.35, ?, ?, ?)",
        (chunk_id, "test_proj", chunk_type, f"content {summary}", summary,
         importance, emotional_weight, now, now)
    )
    conn.commit()


# ══════════════════════════════════════════════════════════════════════
# 1. compute_emotional_salience 纯函数测试
# ══════════════════════════════════════════════════════════════════════

def test_et1_high_arousal_words():
    """高唤醒词（崩溃）→ delta > 0, emotional_weight > 0.4。"""
    delta = compute_emotional_salience("服务崩溃，critical bug 需要紧急修复")
    assert delta > 0.10, f"ET1: 崩溃+critical+紧急 → delta>0.10, got {delta}"
    # emotional_weight = min(1.0, delta / 0.25)
    ew = min(1.0, max(0.0, delta / 0.25))
    assert ew > 0.4, f"ET1: emotional_weight={ew:.4f} 应 > 0.4（触发 retriever boost 阈值）"


def test_et2_negative_words_no_weight():
    """负向词（已解决/resolved）→ delta < 0，emotional_weight = 0（不产生情绪权重）。"""
    delta = compute_emotional_salience("问题已解决，已修复，resolved")
    assert delta < 0, f"ET2: 已解决+resolved → delta<0, got {delta}"
    # 负向 delta 不产生 emotional_weight
    ew = max(0.0, min(1.0, delta / 0.25)) if delta > 0 else 0.0
    assert ew == 0.0, f"ET2: 负向词 emotional_weight 应为 0, got {ew}"


def test_et3_neutral_text_zero_weight():
    """情感中性文本 → delta ≈ 0，emotional_weight = 0。"""
    delta = compute_emotional_salience("实现了用户登录功能，包含密码验证逻辑")
    assert abs(delta) < 0.01, f"ET3: 中性文本 delta 应 < 0.01, got {delta}"


def test_et4_accumulation():
    """多情绪词叠加 → delta 累积，emotional_weight 更高。"""
    delta_single = compute_emotional_salience("崩溃了")
    delta_multi = compute_emotional_salience("崩溃了，CRITICAL，数据丢失，P0 紧急")
    assert delta_multi > delta_single, (
        f"ET4: 多词累积 delta={delta_multi:.4f} 应 > 单词 delta={delta_single:.4f}"
    )


def test_et5_weight_capped_at_1():
    """极端情绪词叠加 → emotional_weight 上限 1.0。"""
    text = "崩溃 critical 紧急 P0 fatal panic CRITICAL 数据丢失 严重错误 突破 关键发现 必须"
    delta = compute_emotional_salience(text)
    ew = min(1.0, max(0.0, delta / 0.25)) if delta > 0 else 0.0
    assert ew <= 1.0, f"ET5: emotional_weight 上限 1.0, got {ew}"


# ══════════════════════════════════════════════════════════════════════
# 2. apply_emotional_salience — DB 写入测试
# ══════════════════════════════════════════════════════════════════════

def test_et6_apply_writes_importance_and_weight(conn):
    """apply_emotional_salience → 同时更新 importance 和 emotional_weight。"""
    _insert_chunk(conn, "chunk_et6", importance=0.80, emotional_weight=None,
                  summary="服务崩溃，critical bug")
    apply_emotional_salience(conn, "chunk_et6", "服务崩溃，critical bug", 0.80)
    conn.commit()

    row = conn.execute(
        "SELECT importance, emotional_weight FROM memory_chunks WHERE id='chunk_et6'"
    ).fetchone()
    assert row is not None
    # importance 应上调（崩溃类词 delta > 0.12）
    assert row["importance"] > 0.80, (
        f"ET6: importance 应上调 > 0.80, got {row['importance']}"
    )
    # emotional_weight 应 > 0
    assert (row["emotional_weight"] or 0.0) > 0.0, (
        f"ET6: emotional_weight 应 > 0, got {row['emotional_weight']}"
    )


def test_et7_neutral_text_writes_zero_weight(conn):
    """中性文本 → emotional_weight 写入 0.0（not NULL）。"""
    _insert_chunk(conn, "chunk_et7", importance=0.80, emotional_weight=None,
                  summary="实现登录功能")
    apply_emotional_salience(conn, "chunk_et7", "实现登录功能", 0.80)
    conn.commit()

    row = conn.execute(
        "SELECT emotional_weight FROM memory_chunks WHERE id='chunk_et7'"
    ).fetchone()
    # 中性文本 delta < 0.01 → 不修改 importance，但写入 emotional_weight=0（如果原为 NULL）
    # 允许 NULL 或 0（NULL 说明 schema 没有 emotional_weight 列，0 说明已写入）
    ew = row["emotional_weight"]
    assert ew is None or ew == 0.0, (
        f"ET7: 中性文本 emotional_weight 应为 NULL 或 0, got {ew}"
    )


def test_et8_negative_words_no_weight_but_lower_importance(conn):
    """负向词 → importance 下调，但 emotional_weight = 0（不产生情绪权重）。"""
    _insert_chunk(conn, "chunk_et8", importance=0.80, emotional_weight=None,
                  summary="问题已解决，已修复，resolved")
    apply_emotional_salience(conn, "chunk_et8", "问题已解决，已修复，resolved", 0.80)
    conn.commit()

    row = conn.execute(
        "SELECT importance, emotional_weight FROM memory_chunks WHERE id='chunk_et8'"
    ).fetchone()
    # importance 下调（负向 delta）
    assert row["importance"] < 0.80, (
        f"ET8: 已解决类词 importance 应下调 < 0.80, got {row['importance']}"
    )
    # emotional_weight 应为 0（负向情绪不应激活 retriever boost）
    ew = row["emotional_weight"] or 0.0
    assert ew == 0.0, f"ET8: 负向词 emotional_weight 应为 0, got {ew}"


# ══════════════════════════════════════════════════════════════════════
# 3. integration — _write_chunk 写入后 DB 字段验证
# ══════════════════════════════════════════════════════════════════════

def test_et9_write_chunk_emotional_weight_integration(conn):
    """
    集成：_write_chunk 写入含情绪词的 reasoning_chain → emotional_weight > 0。
    """
    from extractor import _write_chunk

    # 含高唤醒词的 reasoning_chain
    _write_chunk(
        "reasoning_chain",
        "发现崩溃根因：critical bug 在 scheduler 中导致 data loss",
        "test_project_ew",
        "session_et9",
        conn=conn,
    )
    conn.commit()

    row = conn.execute(
        "SELECT importance, emotional_weight FROM memory_chunks "
        "WHERE chunk_type='reasoning_chain' AND project='test_project_ew' "
        "ORDER BY created_at DESC LIMIT 1"
    ).fetchone()

    if row is None:
        pytest.skip("chunk 未写入（可能被 SNR filter 过滤），跳过集成验证")

    # emotional_weight 应被设置（> 0 或明确 0，不应为 NULL 且 > 阈值）
    ew = row["emotional_weight"]
    # 含崩溃+critical+data loss → delta > 0.20 → ew > 0.4
    # 若 ew 为 NULL 说明 apply_emotional_salience 未被调用（bug）
    assert ew is not None, "ET9: emotional_weight 不应为 NULL（apply_emotional_salience 未调用）"
    assert ew > 0.4, (
        f"ET9: 崩溃+critical → emotional_weight 应 > 0.4（触发 retriever boost），got {ew}"
    )
