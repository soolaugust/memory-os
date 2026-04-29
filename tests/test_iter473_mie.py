"""
test_iter473_mie.py — iter473: Memory Interference Effect 单元测试

覆盖：
  MI1: 同类型 chunk 在 24h 内词汇重叠 >= 0.30 → 新 chunk stability 降低
  MI2: 词汇重叠 < mie_min_overlap(0.30) → 无干扰惩罚
  MI3: mie_enabled=False → 无惩罚
  MI4: importance < mie_min_importance(0.30) → 不参与 MIE
  MI5: 不同 chunk_type 不互相干扰
  MI6: 时间窗口外的 chunk 不触发干扰
  MI7: 惩罚受 mie_max_penalty(0.12) 上限保护
  MI8: 直接调用 apply_memory_interference_effect → mie_penalized=True

认知科学依据：
  McGeoch (1932) 倒摄干扰（RI）— 新学内容干扰旧记忆；相似度越高干扰越强。
  Underwood (1957) 前摄干扰（PI）— 旧习惯干扰新编码（词汇重叠 > 30% = 强干扰区）。
  量化：相似材料学习后，对旧材料的回忆下降 10-20%（McGeoch interference function）。

OS 类比：Linux cache thrashing（mm/vmscan.c thrash_count）—
  working set > available memory 时 page 不断换入换出，effective throughput 下降。
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

from store_vfs import ensure_schema, insert_chunk, apply_memory_interference_effect
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
                chunk_type="observation", project="test", source_session="sess1"):
    now_iso = _utcnow().isoformat()
    return {
        "id": cid,
        "project": project,
        "source_session": source_session,
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


def _insert_raw(conn, cid, content, chunk_type="observation", importance=0.6,
                stability=5.0, project="test", created_at=None):
    now_iso = created_at or _utcnow().isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count,
            encode_context, session_type_history)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (cid, project, chunk_type, content, "summary", importance, stability,
         now_iso, now_iso, 0.5, now_iso, 0, "test_ctx", "")
    )
    conn.commit()


# ── MI1: 高词汇重叠 → stability 降低 ─────────────────────────────────────────────────────

def test_mi1_high_overlap_penalized(conn):
    """MI1: 同类型 chunk 在时间窗口内词汇重叠 >= 0.30 → 新 chunk stability 降低。"""
    # 先插入一个"旧" chunk（相似内容）
    content_old = "memory allocator kernel page frame pool buddy system"
    _insert_raw(conn, "mi1_old", content_old, chunk_type="observation")

    # 再用 apply 直接测试相似内容的惩罚
    content_new = "memory allocator kernel page frame pool slab system"  # 高度重叠
    _insert_raw(conn, "mi1_new", content_new, chunk_type="observation", stability=5.0)
    stab_before = _get_stability(conn, "mi1_new")

    result = apply_memory_interference_effect(conn, "mi1_new", content_new, "test", "observation")
    stab_after = _get_stability(conn, "mi1_new")

    assert stab_after < stab_before, (
        f"MI1: 高词汇重叠时 stability 应降低，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["mie_penalized"] is True, f"MI1: mie_penalized 应为 True，got {result}"


# ── MI2: 词汇重叠不足 → 无惩罚 ───────────────────────────────────────────────────────────

def test_mi2_low_overlap_no_penalty(conn):
    """MI2: 词汇重叠 < mie_min_overlap(0.30) → 无干扰惩罚。"""
    content_old = "memory allocator kernel page frame pool buddy system slab vmalloc"
    _insert_raw(conn, "mi2_old", content_old, chunk_type="observation")

    content_new = "database query optimizer index btree postgresql transaction"  # 几乎无重叠
    _insert_raw(conn, "mi2_new", content_new, chunk_type="observation", stability=5.0)
    stab_before = _get_stability(conn, "mi2_new")

    result = apply_memory_interference_effect(conn, "mi2_new", content_new, "test", "observation")
    stab_after = _get_stability(conn, "mi2_new")

    assert abs(stab_after - stab_before) < 0.001, (
        f"MI2: 低词汇重叠时不应有惩罚，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["mie_penalized"] is False, f"MI2: mie_penalized 应为 False，got {result}"


# ── MI3: mie_enabled=False → 无惩罚 ─────────────────────────────────────────────────────

def test_mi3_disabled_no_penalty(conn):
    """MI3: mie_enabled=False → 高重叠内容无 MIE 惩罚。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.mie_enabled":
            return False
        return original_get(key, project=project)

    content_old = "memory allocator kernel page frame pool buddy system"
    _insert_raw(conn, "mi3_old", content_old, chunk_type="observation")

    content_new = "memory allocator kernel page frame pool slab system"
    _insert_raw(conn, "mi3_new", content_new, chunk_type="observation", stability=5.0)
    stab_before = _get_stability(conn, "mi3_new")

    with mock.patch.object(config, 'get', side_effect=patched_get):
        result = apply_memory_interference_effect(conn, "mi3_new", content_new, "test", "observation")
    stab_after = _get_stability(conn, "mi3_new")

    assert abs(stab_after - stab_before) < 0.001, (
        f"MI3: disabled 时不应有惩罚，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["mie_penalized"] is False, f"MI3: mie_penalized 应为 False"


# ── MI4: importance 不足 → 不参与 MIE ────────────────────────────────────────────────────

def test_mi4_low_importance_no_penalty(conn):
    """MI4: importance < mie_min_importance(0.30) → 不参与 MIE。"""
    content_old = "memory allocator kernel page frame pool buddy system"
    _insert_raw(conn, "mi4_old", content_old, chunk_type="observation")

    content_new = "memory allocator kernel page frame pool slab system"
    _insert_raw(conn, "mi4_low", content_new, chunk_type="observation",
                importance=0.10, stability=5.0)
    stab_before = _get_stability(conn, "mi4_low")

    result = apply_memory_interference_effect(conn, "mi4_low", content_new, "test", "observation")
    stab_after = _get_stability(conn, "mi4_low")

    assert abs(stab_after - stab_before) < 0.001, (
        f"MI4: 低 importance 时不应有惩罚，before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── MI5: 不同 chunk_type 不互相干扰 ──────────────────────────────────────────────────────

def test_mi5_different_type_no_interference(conn):
    """MI5: 不同 chunk_type 的 chunk 不互相干扰（跨类型不触发 MIE）。"""
    content_old = "memory allocator kernel page frame pool buddy system"
    _insert_raw(conn, "mi5_old", content_old, chunk_type="decision")

    # 新 chunk 是 observation 类型，与 decision 类型的旧 chunk 不干扰
    content_new = "memory allocator kernel page frame pool slab system"
    _insert_raw(conn, "mi5_new", content_new, chunk_type="observation", stability=5.0)
    stab_before = _get_stability(conn, "mi5_new")

    result = apply_memory_interference_effect(conn, "mi5_new", content_new, "test", "observation")
    stab_after = _get_stability(conn, "mi5_new")

    assert abs(stab_after - stab_before) < 0.001, (
        f"MI5: 不同类型不应相互干扰，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["mie_penalized"] is False, f"MI5: mie_penalized 应为 False，got {result}"


# ── MI6: 时间窗口外的 chunk 不触发干扰 ───────────────────────────────────────────────────

def test_mi6_outside_window_no_interference(conn):
    """MI6: 时间窗口（mie_window_hours=24h）外的旧 chunk 不触发 MIE。"""
    import datetime as dt
    # 插入 48 小时前的旧 chunk
    old_time = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=48)).isoformat()
    content_old = "memory allocator kernel page frame pool buddy system"
    _insert_raw(conn, "mi6_old", content_old, chunk_type="observation", created_at=old_time)

    content_new = "memory allocator kernel page frame pool slab system"
    _insert_raw(conn, "mi6_new", content_new, chunk_type="observation", stability=5.0)
    stab_before = _get_stability(conn, "mi6_new")

    result = apply_memory_interference_effect(conn, "mi6_new", content_new, "test", "observation")
    stab_after = _get_stability(conn, "mi6_new")

    assert abs(stab_after - stab_before) < 0.001, (
        f"MI6: 时间窗口外不应触发干扰，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["mie_penalized"] is False, f"MI6: mie_penalized 应为 False"


# ── MI7: 惩罚受 mie_max_penalty 上限保护 ─────────────────────────────────────────────────

def test_mi7_max_penalty_cap(conn):
    """MI7: MIE 惩罚不超过 base × mie_max_penalty(0.12)（直接调用验证）。"""
    mie_max_penalty = config.get("store_vfs.mie_max_penalty")  # 0.12
    base = 5.0

    content_old = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    _insert_raw(conn, "mi7_old", content_old, chunk_type="observation")

    # 完全相同内容（最大重叠）
    content_new = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    _insert_raw(conn, "mi7_new", content_new, chunk_type="observation", stability=base)
    stab_before = _get_stability(conn, "mi7_new")

    apply_memory_interference_effect(conn, "mi7_new", content_new, "test", "observation")
    stab_after = _get_stability(conn, "mi7_new")

    penalty = stab_before - stab_after
    max_allowed_penalty = base * mie_max_penalty + 0.01
    assert penalty <= max_allowed_penalty, (
        f"MI7: 惩罚 {penalty:.4f} 不应超过 max_penalty 允许的 {max_allowed_penalty:.4f}，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert stab_after < stab_before, f"MI7: 应有 MIE 惩罚，before={stab_before:.4f} after={stab_after:.4f}"


# ── MI8: 直接调用 apply_memory_interference_effect ───────────────────────────────────────

def test_mi8_direct_function_penalty(conn):
    """MI8: apply_memory_interference_effect 对高重叠内容直接产生惩罚。"""
    content_old = "memory allocator kernel page buddy slab vmalloc mmap pagefault cache"
    _insert_raw(conn, "mi8_old", content_old, chunk_type="observation")

    content_new = "memory allocator kernel page buddy slab vmalloc mmap pagefault swap"
    _insert_raw(conn, "mi8_new", content_new, chunk_type="observation", stability=5.0)
    stab_before = _get_stability(conn, "mi8_new")

    result = apply_memory_interference_effect(conn, "mi8_new", content_new, "test", "observation")
    stab_after = _get_stability(conn, "mi8_new")

    assert stab_after < stab_before, (
        f"MI8: 直接调用应产生惩罚，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["mie_penalized"] is True, f"MI8: mie_penalized 应为 True，got {result}"
    assert result["mie_overlap"] >= config.get("store_vfs.mie_min_overlap"), (
        f"MI8: mie_overlap {result['mie_overlap']:.4f} 应 >= min_overlap"
    )
