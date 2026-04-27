"""
test_sm2.py — 迭代323：SM-2 Ebbinghaus 间隔重复精确化单元测试

验证：
  1. quality=5 时 stability 增益最大（× 1.2）
  2. quality=3 时 stability 不变（× 1.0，中性）
  3. quality=0 时 stability 降低（× 0.7，遗忘惩罚）
  4. stability 上限为 365.0 天
  5. stability 初始值 1.0 时，quality=5 后变为 1.2
  6. recall_quality=None 时默认 quality=4（× 1.1，轻微加固）
  7. 多次 quality=5 累积效应（指数增长直到 365 上限）
  8. quality 边界 clamp：负数→0，>5→5
  9. 向后兼容：不传 recall_quality 与默认 quality=4 行为一致

Wozniak (1987) SM-2 算法：
  S_new = S_old × (1 + 0.1 × (quality - 3))
  quality ∈ {0..5}: 0=完全忘记, 3=中等, 5=完美回忆
"""
import sys
import sqlite3
import pytest
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "hooks"))

import tmpfs  # noqa

from store_vfs import ensure_schema, insert_chunk, update_accessed


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _make_chunk(cid, stability=1.0, project="test"):
    now = _now_iso()
    return {
        "id": cid,
        "created_at": now,
        "updated_at": now,
        "project": project,
        "source_session": "s1",
        "chunk_type": "decision",
        "info_class": "semantic",
        "content": f"[decision] chunk {cid}",
        "summary": f"chunk {cid} summary",
        "tags": ["decision"],
        "importance": 0.7,
        "retrievability": 0.5,
        "last_accessed": now,
        "access_count": 1,
        "oom_adj": 0,
        "lru_gen": 0,
        "stability": stability,
        "raw_snippet": "",
        "encoding_context": {},
    }


def _get_stability(conn, cid):
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return row["stability"] if row else None


# ══════════════════════════════════════════════════════════════════════
# 1. SM-2 factor 验证
# ══════════════════════════════════════════════════════════════════════

def test_quality5_max_gain(conn):
    """quality=5 → stability × 1.2（最大增益）。"""
    insert_chunk(conn, _make_chunk("c1", stability=1.0))
    conn.commit()
    update_accessed(conn, ["c1"], recall_quality=5)
    conn.commit()
    s = _get_stability(conn, "c1")
    assert abs(s - 1.2) < 1e-6, f"quality=5 → ×1.2，got {s}"


def test_quality3_neutral(conn):
    """quality=3 → stability 不变（× 1.0，中性）。"""
    insert_chunk(conn, _make_chunk("c2", stability=2.5))
    conn.commit()
    update_accessed(conn, ["c2"], recall_quality=3)
    conn.commit()
    s = _get_stability(conn, "c2")
    assert abs(s - 2.5) < 1e-6, f"quality=3 → ×1.0，got {s}"


def test_quality0_forgetting(conn):
    """quality=0 → stability × 0.7（遗忘惩罚，下限保护）。"""
    insert_chunk(conn, _make_chunk("c3", stability=2.0))
    conn.commit()
    update_accessed(conn, ["c3"], recall_quality=0)
    conn.commit()
    s = _get_stability(conn, "c3")
    assert abs(s - 1.4) < 1e-6, f"quality=0 → ×0.7 → 1.4，got {s}"


def test_quality4_mild_gain(conn):
    """quality=4 → stability × 1.1（轻微加固）。"""
    insert_chunk(conn, _make_chunk("c4", stability=1.0))
    conn.commit()
    update_accessed(conn, ["c4"], recall_quality=4)
    conn.commit()
    s = _get_stability(conn, "c4")
    assert abs(s - 1.1) < 1e-6, f"quality=4 → ×1.1，got {s}"


def test_default_quality_is_4(conn):
    """recall_quality=None 默认 quality=4 → × 1.1。"""
    insert_chunk(conn, _make_chunk("c5", stability=1.0))
    conn.commit()
    update_accessed(conn, ["c5"], recall_quality=None)
    conn.commit()
    s = _get_stability(conn, "c5")
    assert abs(s - 1.1) < 1e-6, f"默认 quality=4 → ×1.1，got {s}"


def test_no_quality_param_backward_compat(conn):
    """不传 recall_quality 与 recall_quality=None 行为一致（向后兼容）。"""
    insert_chunk(conn, _make_chunk("c6", stability=1.0))
    insert_chunk(conn, _make_chunk("c7", stability=1.0))
    conn.commit()
    update_accessed(conn, ["c6"])           # 不传 recall_quality
    update_accessed(conn, ["c7"], recall_quality=None)  # 显式 None
    conn.commit()
    s6 = _get_stability(conn, "c6")
    s7 = _get_stability(conn, "c7")
    assert abs(s6 - s7) < 1e-6, f"不传与 None 应一致，got {s6} vs {s7}"


# ══════════════════════════════════════════════════════════════════════
# 2. 上限与边界
# ══════════════════════════════════════════════════════════════════════

def test_stability_capped_at_365(conn):
    """stability 上限为 365.0 天。"""
    insert_chunk(conn, _make_chunk("c8", stability=360.0))
    conn.commit()
    update_accessed(conn, ["c8"], recall_quality=5)  # 360 × 1.2 = 432 → clamp to 365
    conn.commit()
    s = _get_stability(conn, "c8")
    assert s == 365.0, f"stability 上限应为 365，got {s}"


def test_quality_clamp_below_zero(conn):
    """quality < 0 被 clamp 到 0。"""
    insert_chunk(conn, _make_chunk("c9", stability=2.0))
    conn.commit()
    update_accessed(conn, ["c9"], recall_quality=-1)  # clamp to 0 → × 0.7
    conn.commit()
    s = _get_stability(conn, "c9")
    assert abs(s - 1.4) < 1e-6, f"quality=-1 clamp 到 0 → ×0.7，got {s}"


def test_quality_clamp_above_5(conn):
    """quality > 5 被 clamp 到 5。"""
    insert_chunk(conn, _make_chunk("c10", stability=1.0))
    conn.commit()
    update_accessed(conn, ["c10"], recall_quality=10)  # clamp to 5 → × 1.2
    conn.commit()
    s = _get_stability(conn, "c10")
    assert abs(s - 1.2) < 1e-6, f"quality=10 clamp 到 5 → ×1.2，got {s}"


# ══════════════════════════════════════════════════════════════════════
# 3. 累积效应
# ══════════════════════════════════════════════════════════════════════

def test_repeated_quality5_compounds(conn):
    """多次 quality=5 累积增长（指数直到 365 上限）。"""
    insert_chunk(conn, _make_chunk("c11", stability=1.0))
    conn.commit()

    s = 1.0
    for i in range(5):
        update_accessed(conn, ["c11"], recall_quality=5)
        conn.commit()
        s = min(365.0, s * 1.2)

    actual = _get_stability(conn, "c11")
    assert abs(actual - s) < 1e-4, f"5次 quality=5 后 stability 应为 {s:.4f}，got {actual}"


def test_quality5_beats_quality3_after_multiple_recalls(conn):
    """多次 quality=5 的 stability 显著高于 quality=3 的 stability。"""
    insert_chunk(conn, _make_chunk("high_q", stability=1.0))
    insert_chunk(conn, _make_chunk("mid_q", stability=1.0))
    conn.commit()

    for _ in range(5):
        update_accessed(conn, ["high_q"], recall_quality=5)
        update_accessed(conn, ["mid_q"], recall_quality=3)
        conn.commit()

    s_high = _get_stability(conn, "high_q")
    s_mid = _get_stability(conn, "mid_q")
    assert s_high > s_mid, f"quality=5 积累应 > quality=3：{s_high:.4f} > {s_mid:.4f}"


# ══════════════════════════════════════════════════════════════════════
# 4. 批量操作
# ══════════════════════════════════════════════════════════════════════

def test_batch_update_applies_same_quality(conn):
    """批量更新时所有 chunk 应用相同 recall_quality。"""
    for i in range(3):
        insert_chunk(conn, _make_chunk(f"b{i}", stability=1.0))
    conn.commit()

    update_accessed(conn, ["b0", "b1", "b2"], recall_quality=5)
    conn.commit()

    for i in range(3):
        s = _get_stability(conn, f"b{i}")
        assert abs(s - 1.2) < 1e-6, f"批量 quality=5 → ×1.2，b{i} got {s}"
