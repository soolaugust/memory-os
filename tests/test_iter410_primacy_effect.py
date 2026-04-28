"""
test_iter410_primacy_effect.py — iter410: Primacy Effect (Murdock 1962)

覆盖：
  PR1: compute_primacy_rank — 最早创建的 chunk 返回低 rank（接近 0.0）
  PR2: compute_primacy_rank — 最晚创建的 chunk 返回高 rank（接近 1.0）
  PR3: compute_primacy_rank — 项目 chunk 数 < min_total_chunks 返回 1.0
  PR4: compute_primacy_rank — 空/None 输入安全返回 1.0
  PR5: primacy_stability_bonus — rank < 0.10 返回完整加成（base × 0.15）
  PR6: primacy_stability_bonus — rank [0.10, 0.20) 线性衰减
  PR7: primacy_stability_bonus — rank >= 0.20 返回 0.0
  PR8: primacy_stability_bonus — None/invalid 安全返回 0.0
  PR9: apply_primacy_effect — 早期 chunk stability 被提升（项目够大时）
  PR10: apply_primacy_effect — 晚期 chunk stability 不变

认知科学依据：
  Murdock (1962) Serial Position Effect: 序列最早项目记忆效果最好（首位效应）。
  Rundus (1971): rehearsal hypothesis — 最早项目被复述次数最多。

OS 类比：Linux boot-time kernel parameters —
  内核启动时设置的参数比运行时 sysctl 更持久（是系统的基础 schema）。
"""
import sys
import sqlite3
import pytest
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "hooks"))

import tmpfs  # noqa
from store_vfs import (
    ensure_schema,
    compute_primacy_rank,
    primacy_stability_bonus,
    apply_primacy_effect,
)


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _ts(days_ago=0):
    dt = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return dt.isoformat()


def _insert(conn, chunk_id, days_ago=0, project="test"):
    created = _ts(days_ago=days_ago)
    conn.execute(
        "INSERT OR REPLACE INTO memory_chunks "
        "(id, project, chunk_type, content, summary, importance, retrievability, "
        "stability, created_at, updated_at) "
        "VALUES (?, ?, 'decision', 'content', 'summary', 0.7, 0.5, 1.0, ?, ?)",
        (chunk_id, project, created, _ts())
    )
    conn.commit()


def _insert_n_chunks(conn, n, project="test"):
    """插入 n 个 chunk，间隔 1 天（从 n 天前到 1 天前）。"""
    for i in range(n):
        _insert(conn, f"ch_{i}", days_ago=n - i, project=project)


# ══════════════════════════════════════════════════════════════════════
# 1. compute_primacy_rank 测试
# ══════════════════════════════════════════════════════════════════════

def test_pr1_earliest_chunk_low_rank(conn):
    """最早创建的 chunk 应返回低 rank（接近 0.0）。"""
    _insert_n_chunks(conn, 25)  # 25 个 chunk（> min_total=20）
    # ch_0 是最早的（25 天前）
    rank = compute_primacy_rank(conn, "ch_0", "test")
    assert rank < 0.10, f"PR1: 最早 chunk rank 应 < 0.10，got {rank:.4f}"


def test_pr2_latest_chunk_high_rank(conn):
    """最晚创建的 chunk 应返回高 rank（接近 1.0）。"""
    _insert_n_chunks(conn, 25)
    # ch_24 是最晚的（1 天前）
    rank = compute_primacy_rank(conn, "ch_24", "test")
    assert rank >= 0.90, f"PR2: 最晚 chunk rank 应 >= 0.90，got {rank:.4f}"


def test_pr3_small_project_returns_one(conn):
    """项目 chunk 数 < min_total_chunks 返回 1.0（不应用首位效应）。"""
    _insert_n_chunks(conn, 10)  # 只有 10 个 < 20
    rank = compute_primacy_rank(conn, "ch_0", "test", min_total_chunks=20)
    assert rank == 1.0, f"PR3: 小项目应返回 1.0，got {rank:.4f}"


def test_pr4_empty_inputs_safe(conn):
    """空/None 输入安全返回 1.0。"""
    assert compute_primacy_rank(conn, "", "test") == 1.0
    assert compute_primacy_rank(conn, None, "test") == 1.0
    assert compute_primacy_rank(conn, "chunk_x", "") == 1.0
    assert compute_primacy_rank(conn, "chunk_x", None) == 1.0


# ══════════════════════════════════════════════════════════════════════
# 2. primacy_stability_bonus 测试
# ══════════════════════════════════════════════════════════════════════

def test_pr5_early_rank_full_bonus():
    """rank < 0.10 → 完整加成 base × 0.15。"""
    bonus = primacy_stability_bonus(0.05, 1.0)
    assert 0.13 <= bonus <= 0.16, f"PR5: 早期 rank bonus 应约 0.15，got {bonus:.4f}"


def test_pr6_mid_rank_interpolated():
    """rank [0.10, 0.20) → 线性衰减。"""
    bonus_10 = primacy_stability_bonus(0.10, 1.0)
    bonus_15 = primacy_stability_bonus(0.15, 1.0)
    bonus_19 = primacy_stability_bonus(0.19, 1.0)
    # 单调递减
    assert bonus_10 >= bonus_15 >= bonus_19, (
        f"PR6: rank 递增时 bonus 应单调递减: {bonus_10:.4f} >= {bonus_15:.4f} >= {bonus_19:.4f}"
    )
    # bonus_10 应接近 0.15（刚进入衰减区间）
    assert bonus_10 <= 0.155, f"PR6: rank=0.10 时 bonus 不超过完整值，got {bonus_10:.4f}"


def test_pr7_late_rank_no_bonus():
    """rank >= 0.20 → 无加成。"""
    assert primacy_stability_bonus(0.20, 1.0) == 0.0
    assert primacy_stability_bonus(0.50, 1.0) == 0.0
    assert primacy_stability_bonus(1.0, 1.0) == 0.0


def test_pr8_invalid_inputs_safe():
    """None/invalid 输入安全返回 0.0。"""
    assert primacy_stability_bonus(None, 1.0) == 0.0
    assert primacy_stability_bonus(0.05, None) == 0.0
    assert primacy_stability_bonus("bad", 1.0) == 0.0


# ══════════════════════════════════════════════════════════════════════
# 3. apply_primacy_effect 集成测试
# ══════════════════════════════════════════════════════════════════════

def test_pr9_early_chunk_boosted(conn):
    """早期 chunk stability 被提升（项目够大时）。"""
    _insert_n_chunks(conn, 25)  # 25 个 chunk
    orig = conn.execute(
        "SELECT stability FROM memory_chunks WHERE id='ch_0'"
    ).fetchone()[0]
    new_s = apply_primacy_effect(conn, "ch_0", "test", base_stability=orig, min_total_chunks=20)
    conn.commit()
    assert new_s > orig, f"PR9: 早期 chunk stability 应被提升，got {new_s:.4f} vs {orig:.4f}"


def test_pr10_late_chunk_no_change(conn):
    """晚期 chunk stability 不变（rank >= 0.20）。"""
    _insert_n_chunks(conn, 25)
    # ch_24 rank 约 0.96，远超 0.20
    orig = conn.execute(
        "SELECT stability FROM memory_chunks WHERE id='ch_24'"
    ).fetchone()[0]
    new_s = apply_primacy_effect(conn, "ch_24", "test", base_stability=orig, min_total_chunks=20)
    assert new_s == orig, f"PR10: 晚期 chunk stability 不变，got {new_s:.4f} vs {orig:.4f}"
