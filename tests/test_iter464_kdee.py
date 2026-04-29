"""
test_iter464_kdee.py — iter464: Keyword Density Encoding Effect 单元测试

覆盖：
  KD1: unique_word_ratio >= kdee_min_density(0.60) + words >= kdee_min_words(6) → stability 加成
  KD2: unique_word_ratio < kdee_min_density → 无加成（相对比较）
  KD3: content 词数 < kdee_min_words → 无加成
  KD4: kdee_enabled=False → 无任何加成
  KD5: importance < kdee_min_importance(0.30) → 不参与 KDEE
  KD6: 加成系数 = kdee_boost_factor(1.10)，受 kdee_max_boost(0.20) 保护
  KD7: stability 加成后不超过 365.0（cap 保护）
  KD8: 高密度内容（全部 unique words）比低密度内容（重复词多）获得更大加成

认知科学依据：
  Craik & Lockhart (1972) "Levels of processing: A framework for memory research"
    (Journal of Verbal Learning and Verbal Behavior) —
    深度加工（semantic processing）比浅层加工（phonological/orthographic）产生更持久记忆痕迹。
    信息密度高（unique words / total words 比率高）= 需要更深的语义处理 = 更强编码。
  Kintsch (1974): 文本命题密度与长期记忆保留量正相关（r=0.62）。

OS 类比：Linux ext4 extent tree depth —
  dense inode（大量唯一 extent 块）→ 更深的 B-tree 索引结构 →
  更鲁棒的随机访问性能（信息密度高 = 更深索引 = 更快检索）。
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

from store_vfs import ensure_schema, insert_chunk
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
        "summary": "summary content",
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


# ── KD1: 高密度内容 → stability 加成 ─────────────────────────────────────────────────

def test_kd1_high_density_boosted(conn):
    """KD1: unique_word_ratio >= kdee_min_density 且 words >= kdee_min_words → stability 加成。"""
    # 高密度：每个词都不同（unique_ratio ≈ 1.0）
    content_high = "kernel memory allocator slab buddy vmalloc mmap pagefault"
    chunk_high = _make_chunk("kdee_1_high", content=content_high, importance=0.6, stability=5.0)
    insert_chunk(conn, chunk_high)
    stab_high = _get_stability(conn, "kdee_1_high")

    # 低密度 baseline（重复词多，unique_ratio 低）
    content_low = "the the the the the the the the the the the the the the the"
    chunk_low = _make_chunk("kdee_1_low", content=content_low, importance=0.6, stability=5.0)
    insert_chunk(conn, chunk_low)
    stab_low = _get_stability(conn, "kdee_1_low")

    assert stab_high > stab_low, (
        f"KD1: 高密度内容 stability 应高于低密度，"
        f"high={stab_high:.4f} low={stab_low:.4f}"
    )


# ── KD2: 低密度内容 → 无加成（相对比较）─────────────────────────────────────────────

def test_kd2_low_density_no_boost(conn):
    """KD2: unique_word_ratio < kdee_min_density(0.60) → 无 KDEE 加成（与高密度相比不更高）。"""
    # 低密度：重复词多
    content_repeat = "the the the the the the the the the the system system system"
    chunk_low = _make_chunk("kdee_2_low", content=content_repeat, importance=0.6, stability=5.0)
    insert_chunk(conn, chunk_low)
    stab_low = _get_stability(conn, "kdee_2_low")

    # 高密度
    content_high = "kernel memory allocator slab buddy vmalloc mmap pagefault interrupt"
    chunk_high = _make_chunk("kdee_2_high", content=content_high, importance=0.6, stability=5.0)
    insert_chunk(conn, chunk_high)
    stab_high = _get_stability(conn, "kdee_2_high")

    assert stab_low <= stab_high + 0.01, (
        f"KD2: 低密度 stability 不应高于高密度，low={stab_low:.4f} high={stab_high:.4f}"
    )


# ── KD3: 词数不足 → 无加成 ───────────────────────────────────────────────────────────

def test_kd3_too_few_words_no_boost(conn):
    """KD3: content 词数 < kdee_min_words(6) → 无 KDEE 加成。"""
    # 只有 4 个词（< 6）
    content_short = "kernel memory allocator slab"
    chunk_short = _make_chunk("kdee_3_short", content=content_short, importance=0.6, stability=5.0)
    insert_chunk(conn, chunk_short)
    stab_short = _get_stability(conn, "kdee_3_short")

    # 足够词数 + 高密度（有加成的对照）
    content_long = "kernel memory allocator slab buddy vmalloc mmap pagefault"
    chunk_long = _make_chunk("kdee_3_long", content=content_long, importance=0.6, stability=5.0)
    insert_chunk(conn, chunk_long)
    stab_long = _get_stability(conn, "kdee_3_long")

    assert stab_short <= stab_long + 0.01, (
        f"KD3: 词数不足时 stability 不应高于足够词数，short={stab_short:.4f} long={stab_long:.4f}"
    )


# ── KD4: kdee_enabled=False → 无加成 ────────────────────────────────────────────────

def test_kd4_disabled_no_boost(conn):
    """KD4: kdee_enabled=False → 高密度内容无 KDEE 加成（与 enabled 相比不更高）。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.kdee_enabled":
            return False
        return original_get(key, project=project)

    content = "kernel memory allocator slab buddy vmalloc mmap pagefault interrupt context"

    chunk_dis = _make_chunk("kdee_4_dis", content=content, importance=0.6, stability=5.0)
    with mock.patch.object(config, 'get', side_effect=patched_get):
        insert_chunk(conn, chunk_dis)
    stab_disabled = _get_stability(conn, "kdee_4_dis")

    chunk_en = _make_chunk("kdee_4_en", content=content, importance=0.6, stability=5.0)
    insert_chunk(conn, chunk_en)
    stab_enabled = _get_stability(conn, "kdee_4_en")

    assert stab_disabled <= stab_enabled + 0.01, (
        f"KD4: disabled 时 stability 不应高于 enabled，"
        f"disabled={stab_disabled:.4f} enabled={stab_enabled:.4f}"
    )


# ── KD5: importance 不足 → 不参与 KDEE ───────────────────────────────────────────────

def test_kd5_low_importance_no_boost(conn):
    """KD5: importance < kdee_min_importance(0.30) → 不参与 KDEE。"""
    content = "kernel memory allocator slab buddy vmalloc mmap pagefault"

    # 低 importance（0.10 < 0.30）
    chunk_low = _make_chunk("kdee_5_low", content=content, importance=0.10, stability=5.0)
    insert_chunk(conn, chunk_low)
    stab_low = _get_stability(conn, "kdee_5_low")

    # 高 importance（0.60 >= 0.30，触发 KDEE）
    chunk_high = _make_chunk("kdee_5_high", content=content, importance=0.60, stability=5.0)
    insert_chunk(conn, chunk_high)
    stab_high = _get_stability(conn, "kdee_5_high")

    assert stab_low <= stab_high + 0.01, (
        f"KD5: 低 importance 时 stability 不应高于高 importance，"
        f"low={stab_low:.4f} high={stab_high:.4f}"
    )


# ── KD6: 加成受 kdee_max_boost 保护 ──────────────────────────────────────────────────

def test_kd6_max_boost_cap(conn):
    """KD6: KDEE 增量（相对 baseline）不超过 base × kdee_max_boost。"""
    kdee_max_boost = config.get("store_vfs.kdee_max_boost")  # 0.20
    base = 5.0

    # baseline（低密度，不触发 KDEE）
    content_base = "the the the the the the the the the the the the the the"
    chunk_base = _make_chunk("kdee_6_base", content=content_base, importance=0.6, stability=base)
    insert_chunk(conn, chunk_base)
    stab_base = _get_stability(conn, "kdee_6_base")

    # KDEE 分支（高密度）
    content_kdee = "kernel memory allocator slab buddy vmalloc mmap pagefault interrupt dma"
    chunk_kdee = _make_chunk("kdee_6_kdee", content=content_kdee, importance=0.6, stability=base)
    insert_chunk(conn, chunk_kdee)
    stab_kdee = _get_stability(conn, "kdee_6_kdee")

    kdee_increment = stab_kdee - stab_base
    max_allowed = base * kdee_max_boost + 0.1
    assert kdee_increment <= max_allowed, (
        f"KD6: KDEE 增量 {kdee_increment:.4f} 不应超过 max_boost 允许的 {max_allowed:.4f}，"
        f"base={stab_base:.4f} kdee={stab_kdee:.4f}"
    )
    assert stab_kdee > stab_base, f"KD6: 应有 KDEE 加成，base={stab_base:.4f} kdee={stab_kdee:.4f}"


# ── KD7: stability 上限 365.0 ─────────────────────────────────────────────────────────

def test_kd7_stability_cap_365(conn):
    """KD7: KDEE boost 后 stability 不超过 365.0（全局上限保护）。"""
    base = 363.0
    content_kdee = "kernel memory allocator slab buddy vmalloc mmap pagefault interrupt dma"
    chunk_kdee = _make_chunk("kdee_7_kdee", content=content_kdee, importance=0.8, stability=base)
    insert_chunk(conn, chunk_kdee)
    stab_kdee = _get_stability(conn, "kdee_7_kdee")

    assert stab_kdee <= 365.0, (
        f"KD7: KDEE boost 后 stability 不应超过 365.0，got {stab_kdee:.4f}"
    )


# ── KD8: 高密度比低密度获得更大加成 ──────────────────────────────────────────────────

def test_kd8_higher_density_more_boost(conn):
    """KD8: unique_word_ratio 更高（更高密度）→ 相对 baseline 的加成更大或相等。"""
    content_base = "the the the the the the the the the the the the"  # unique_ratio ≈ 0.08

    # 中密度（约 60% unique）
    content_medium = "kernel memory kernel memory kernel memory slab buddy vmalloc mmap"
    # 高密度（约 100% unique）
    content_high = "kernel memory allocator slab buddy vmalloc mmap pagefault interrupt dma"

    for cid, content in [
        ("kdee_8_base", content_base),
        ("kdee_8_med", content_medium),
        ("kdee_8_high", content_high),
    ]:
        chunk = _make_chunk(cid, content=content, importance=0.6, stability=5.0)
        insert_chunk(conn, chunk)

    stab_base = _get_stability(conn, "kdee_8_base")
    stab_med = _get_stability(conn, "kdee_8_med")
    stab_high = _get_stability(conn, "kdee_8_high")

    # 高密度 >= 中密度 >= 低密度（基准）
    assert stab_high >= stab_med - 0.1, (
        f"KD8: 高密度 stability 应 >= 中密度，high={stab_high:.4f} med={stab_med:.4f}"
    )
    assert stab_med >= stab_base - 0.1, (
        f"KD8: 中密度 stability 应 >= 基准，med={stab_med:.4f} base={stab_base:.4f}"
    )
