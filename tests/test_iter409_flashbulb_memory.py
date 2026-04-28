"""
test_iter409_flashbulb_memory.py — iter409: Flashbulb Memory (Brown & Kulik 1977)

覆盖：
  FB1: flashbulb_stability_bonus — 高情绪唤醒(≥0.70) 返回 base×0.30 加成
  FB2: flashbulb_stability_bonus — 中等情绪 [0.50, 0.70) 线性插值
  FB3: flashbulb_stability_bonus — 弱情绪 [0.30, 0.50) 小加成
  FB4: flashbulb_stability_bonus — 低情绪 (<0.30) 返回 0.0
  FB5: flashbulb_stability_bonus — 加成上限 base×0.30
  FB6: flashbulb_stability_bonus — None/invalid 输入安全返回 0.0
  FB7: apply_flashbulb_effect — 高情绪 chunk stability 被提升
  FB8: apply_flashbulb_effect — 零情绪 chunk stability 不变
  FB9: apply_flashbulb_effect — 空/None chunk_id 安全返回 base
  FB10: flashbulb_bonus 单调性：情绪越高，bonus 越大

认知科学依据：
  Brown & Kulik (1977) "Flashbulb memories":
    高情绪唤醒事件（如重大历史事件）形成极鲜明、持久的记忆。
  McGaugh (2000) Memory consolidation: 杏仁核 norepinephrine 增强海马编码强度。

OS 类比：Linux mlockall(MCL_CURRENT | MCL_FUTURE) —
  高优先级进程将所有内存页锁入 RAM，抵抗 kswapd 驱逐；
  情绪性记忆 = mlockall chunk = 衰减抵抗力最强。
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
from store_vfs import (
    ensure_schema,
    flashbulb_stability_bonus,
    apply_flashbulb_effect,
)


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _insert_with_ew(conn, chunk_id, emotional_weight=0.0, project="test"):
    now = _now()
    conn.execute(
        "INSERT OR REPLACE INTO memory_chunks "
        "(id, project, chunk_type, content, summary, importance, retrievability, "
        "stability, emotional_weight, created_at, updated_at) "
        "VALUES (?, ?, 'decision', 'content', 'summary', 0.7, 0.5, 1.0, ?, ?, ?)",
        (chunk_id, project, emotional_weight, now, now)
    )
    conn.commit()


# ══════════════════════════════════════════════════════════════════════
# 1. flashbulb_stability_bonus 纯函数测试
# ══════════════════════════════════════════════════════════════════════

def test_fb1_strong_emotion_max_bonus():
    """高情绪唤醒(≥0.70) → base × 0.30 加成。"""
    bonus = flashbulb_stability_bonus(0.80, 1.0)
    assert 0.28 <= bonus <= 0.32, f"FB1: 高情绪 bonus 应约 0.30，got {bonus:.4f}"


def test_fb2_medium_emotion_interpolated():
    """中等情绪 (0.50-0.70) → 线性插值 0.15→0.30。"""
    bonus_50 = flashbulb_stability_bonus(0.50, 1.0)
    bonus_60 = flashbulb_stability_bonus(0.60, 1.0)
    bonus_70 = flashbulb_stability_bonus(0.70, 1.0)
    # 应单调递增
    assert bonus_50 <= bonus_60 <= bonus_70, (
        f"FB2: 中等情绪 bonus 应单调递增: {bonus_50:.4f} <= {bonus_60:.4f} <= {bonus_70:.4f}"
    )
    # ew=0.50 时 bonus 应接近 0.15
    assert 0.12 <= bonus_50 <= 0.18, f"FB2: ew=0.50 bonus 应约 0.15，got {bonus_50:.4f}"


def test_fb3_weak_emotion_small_bonus():
    """弱情绪 [0.30, 0.50) → 小加成 (0, 0.15)。"""
    bonus = flashbulb_stability_bonus(0.40, 1.0)
    assert 0.0 < bonus < 0.15, f"FB3: 弱情绪 bonus 应在 (0, 0.15)，got {bonus:.4f}"


def test_fb4_low_emotion_no_bonus():
    """低情绪 (<0.30) → 无加成。"""
    bonus = flashbulb_stability_bonus(0.20, 1.0)
    assert bonus == 0.0, f"FB4: 低情绪无加成，got {bonus:.4f}"
    bonus0 = flashbulb_stability_bonus(0.0, 1.0)
    assert bonus0 == 0.0, f"FB4: 零情绪无加成，got {bonus0:.4f}"


def test_fb5_bonus_capped_at_30_pct():
    """加成上限 base × 0.30。"""
    bonus = flashbulb_stability_bonus(1.0, 2.0)
    assert bonus <= 2.0 * 0.30 + 1e-9, f"FB5: bonus 上限 base×0.30，got {bonus:.4f}"


def test_fb6_invalid_inputs_safe():
    """None/invalid 输入安全返回 0.0。"""
    assert flashbulb_stability_bonus(None, 1.0) == 0.0
    assert flashbulb_stability_bonus(0.8, None) == 0.0
    assert flashbulb_stability_bonus("bad", 1.0) == 0.0
    assert flashbulb_stability_bonus(0.8, "bad") == 0.0
    assert flashbulb_stability_bonus(0.0, 1.0) == 0.0
    assert flashbulb_stability_bonus(0.8, 0.0) == 0.0


def test_fb10_monotonic_with_emotion():
    """情绪越高，bonus 越大（单调性）。"""
    bonuses = [flashbulb_stability_bonus(ew, 1.0) for ew in [0.1, 0.3, 0.5, 0.7, 0.9]]
    for i in range(len(bonuses) - 1):
        assert bonuses[i] <= bonuses[i+1], (
            f"FB10: bonus 应单调递增，got bonuses={[f'{b:.4f}' for b in bonuses]}"
        )


# ══════════════════════════════════════════════════════════════════════
# 2. apply_flashbulb_effect 集成测试
# ══════════════════════════════════════════════════════════════════════

def test_fb7_high_emotion_chunk_stability_boosted(conn):
    """高情绪 chunk 的 stability 被提升。"""
    _insert_with_ew(conn, "fb7_chunk", emotional_weight=0.85)
    original_s = conn.execute(
        "SELECT stability FROM memory_chunks WHERE id='fb7_chunk'"
    ).fetchone()[0]
    new_s = apply_flashbulb_effect(conn, "fb7_chunk", base_stability=original_s)
    conn.commit()
    db_s = conn.execute(
        "SELECT stability FROM memory_chunks WHERE id='fb7_chunk'"
    ).fetchone()[0]
    assert new_s > original_s, f"FB7: 高情绪 chunk stability 应被提升，got {new_s:.4f} vs {original_s:.4f}"
    assert db_s > original_s, f"FB7: DB stability 应被更新，got {db_s:.4f}"


def test_fb8_zero_emotion_no_change(conn):
    """零情绪 chunk stability 不变。"""
    _insert_with_ew(conn, "fb8_chunk", emotional_weight=0.0)
    original_s = conn.execute(
        "SELECT stability FROM memory_chunks WHERE id='fb8_chunk'"
    ).fetchone()[0]
    new_s = apply_flashbulb_effect(conn, "fb8_chunk", base_stability=original_s)
    assert new_s == original_s, f"FB8: 零情绪 stability 不变，got {new_s:.4f} vs {original_s:.4f}"


def test_fb9_empty_chunk_id_safe(conn):
    """空/None chunk_id 安全返回 base_stability。"""
    assert apply_flashbulb_effect(conn, "", base_stability=1.0) == 1.0
    assert apply_flashbulb_effect(conn, None, base_stability=1.5) == 1.5
