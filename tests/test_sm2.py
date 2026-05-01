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

from store_vfs import ensure_schema, update_accessed


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _insert_raw(conn, cid, stability=1.0, project="test", last_accessed=None):
    """直接 SQL 插入，bypass insert_chunk 的写入效应，只测 SM-2 核心逻辑。
    last_accessed 默认设为 10 分钟前，确保 gap > IOR window(300s)，避免 IOR 干扰 SM-2 ratio。
    """
    import datetime as _dt
    now = _now_iso()
    # 默认 10min 前，超出 IOR inhibition_window_secs=300s，不触发 IOR penalty
    la = last_accessed or (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(minutes=10)).isoformat()
    conn.execute("""
        INSERT OR REPLACE INTO memory_chunks
            (id, created_at, updated_at, project, source_session,
             chunk_type, info_class, content, summary, tags,
             importance, retrievability, last_accessed, access_count,
             oom_adj, lru_gen, stability, raw_snippet, encoding_context)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (cid, now, now, project, "s1",
          "decision", "semantic",
          f"[decision] chunk {cid}", f"chunk {cid} summary", "[]",
          0.7, 0.5, la, 1,
          0, 0, stability, "", "{}"))
    conn.commit()


def _get_stability(conn, cid):
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return row["stability"] if row else None


# ══════════════════════════════════════════════════════════════════════
# 1. SM-2 factor 验证
# ══════════════════════════════════════════════════════════════════════

def test_quality5_max_gain(conn):
    """quality=5 → stability × 1.2（最大增益）。"""
    _insert_raw(conn, "c1", stability=1.0)
    s_before = _get_stability(conn, "c1")
    update_accessed(conn, ["c1"], recall_quality=5, _sm2_only=True)
    conn.commit()
    s = _get_stability(conn, "c1")
    ratio = s / s_before if s_before else 0
    assert abs(ratio - 1.2) < 1e-4, f"quality=5 → ×1.2，got ratio={ratio:.6f} (s={s:.4f}, base={s_before:.4f})"


def test_quality3_neutral(conn):
    """quality=3 → stability 不变（× 1.0，中性）。"""
    _insert_raw(conn, "c2", stability=2.5)
    s_before = _get_stability(conn, "c2")
    update_accessed(conn, ["c2"], recall_quality=3, _sm2_only=True)
    conn.commit()
    s = _get_stability(conn, "c2")
    ratio = s / s_before if s_before else 0
    assert abs(ratio - 1.0) < 1e-4, f"quality=3 → ×1.0，got ratio={ratio:.6f}"


def test_quality0_forgetting(conn):
    """quality=0 → stability × 0.7（遗忘惩罚，下限保护）。"""
    _insert_raw(conn, "c3", stability=2.0)
    s_before = _get_stability(conn, "c3")
    update_accessed(conn, ["c3"], recall_quality=0, _sm2_only=True)
    conn.commit()
    s = _get_stability(conn, "c3")
    ratio = s / s_before if s_before else 0
    assert abs(ratio - 0.7) < 1e-4, f"quality=0 → ×0.7，got ratio={ratio:.6f}"


def test_quality4_mild_gain(conn):
    """quality=4 → stability × 1.1（轻微加固）。"""
    _insert_raw(conn, "c4", stability=1.0)
    s_before = _get_stability(conn, "c4")
    update_accessed(conn, ["c4"], recall_quality=4, _sm2_only=True)
    conn.commit()
    s = _get_stability(conn, "c4")
    ratio = s / s_before if s_before else 0
    assert abs(ratio - 1.1) < 1e-4, f"quality=4 → ×1.1，got ratio={ratio:.6f}"


def test_default_quality_is_4(conn):
    """recall_quality=None + 6hr gap → iter389 再巩固窗口推断 quality=4 → × 1.1。"""
    import datetime as _dt
    la = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(hours=6)).isoformat()
    _insert_raw(conn, "c5", stability=1.0, last_accessed=la)
    s_before = _get_stability(conn, "c5")
    update_accessed(conn, ["c5"], recall_quality=None, _sm2_only=True)
    conn.commit()
    s = _get_stability(conn, "c5")
    ratio = s / s_before if s_before else 0
    assert abs(ratio - 1.1) < 0.01, f"6hr gap → quality=4 → ×1.1，got ratio={ratio:.4f}"


def test_no_quality_param_backward_compat(conn):
    """不传 recall_quality 与 recall_quality=None 行为一致（向后兼容）。"""
    import datetime as _dt
    la = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(hours=6)).isoformat()
    _insert_raw(conn, "c6", stability=1.0, last_accessed=la)
    _insert_raw(conn, "c7", stability=1.0, last_accessed=la)
    s6_before = _get_stability(conn, "c6")
    s7_before = _get_stability(conn, "c7")
    update_accessed(conn, ["c6"], _sm2_only=True)
    conn.commit()
    update_accessed(conn, ["c7"], recall_quality=None, _sm2_only=True)
    conn.commit()
    s6 = _get_stability(conn, "c6")
    s7 = _get_stability(conn, "c7")
    ratio6 = s6 / s6_before if s6_before else 0
    ratio7 = s7 / s7_before if s7_before else 0
    assert abs(ratio6 - ratio7) < 0.01, \
        f"不传与 None SM-2 ratio 应一致，ratio6={ratio6:.4f} ratio7={ratio7:.4f}"


# ══════════════════════════════════════════════════════════════════════
# 2. 上限与边界
# ══════════════════════════════════════════════════════════════════════

def test_stability_capped_at_365(conn):
    """stability 上限为 365.0 天。"""
    _insert_raw(conn, "c8", stability=360.0)
    update_accessed(conn, ["c8"], recall_quality=5, _sm2_only=True)  # 360 × 1.2 = 432 → clamp to 365
    conn.commit()
    s = _get_stability(conn, "c8")
    assert s == 365.0, f"stability 上限应为 365，got {s}"


def test_quality_clamp_below_zero(conn):
    """quality < 0 被 clamp 到 0。"""
    _insert_raw(conn, "c9", stability=2.0)
    s_before = _get_stability(conn, "c9")
    update_accessed(conn, ["c9"], recall_quality=-1, _sm2_only=True)  # clamp to 0 → × 0.7
    conn.commit()
    s = _get_stability(conn, "c9")
    ratio = s / s_before if s_before else 0
    assert abs(ratio - 0.7) < 1e-4, f"quality=-1 clamp 到 0 → ×0.7，got ratio={ratio:.6f}"


def test_quality_clamp_above_5(conn):
    """quality > 5 被 clamp 到 5。"""
    _insert_raw(conn, "c10", stability=1.0)
    s_before = _get_stability(conn, "c10")
    update_accessed(conn, ["c10"], recall_quality=10, _sm2_only=True)  # clamp to 5 → × 1.2
    conn.commit()
    s = _get_stability(conn, "c10")
    ratio = s / s_before if s_before else 0
    assert abs(ratio - 1.2) < 1e-4, f"quality=10 clamp 到 5 → ×1.2，got ratio={ratio:.6f}"


# ══════════════════════════════════════════════════════════════════════
# 3. 累积效应
# ══════════════════════════════════════════════════════════════════════

def test_repeated_quality5_compounds(conn):
    """多次 quality=5 累积增长（指数直到 365 上限）。"""
    _insert_raw(conn, "c11", stability=1.0)
    s = _get_stability(conn, "c11")  # = 1.0（raw insert，无写入效应）
    for i in range(5):
        update_accessed(conn, ["c11"], recall_quality=5, _sm2_only=True)
        conn.commit()
        s = min(365.0, s * 1.2)

    actual = _get_stability(conn, "c11")
    assert abs(actual - s) < 1e-4, f"5次 quality=5 后 stability 应为 {s:.4f}，got {actual}"


def test_quality5_beats_quality3_after_multiple_recalls(conn):
    """多次 quality=5 的 stability 显著高于 quality=3 的 stability。"""
    _insert_raw(conn, "high_q", stability=1.0)
    _insert_raw(conn, "mid_q", stability=1.0)

    for _ in range(5):
        update_accessed(conn, ["high_q"], recall_quality=5, _sm2_only=True)
        update_accessed(conn, ["mid_q"], recall_quality=3, _sm2_only=True)
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
        _insert_raw(conn, f"b{i}", stability=1.0)

    s_before = {f"b{i}": _get_stability(conn, f"b{i}") for i in range(3)}
    update_accessed(conn, ["b0", "b1", "b2"], recall_quality=5, _sm2_only=True)
    conn.commit()

    for i in range(3):
        s = _get_stability(conn, f"b{i}")
        base = s_before[f"b{i}"]
        ratio = s / base if base else 0
        assert abs(ratio - 1.2) < 1e-4, f"批量 quality=5 → ×1.2，b{i} got ratio={ratio:.6f}"
