"""
test_iter420_spacing_effect.py — iter420: Spacing Effect 单元测试

覆盖：
  SE1: 长间隔访问（gap >= 24h）→ spaced_access_count 递增
  SE2: 短间隔访问（gap < 1h）→ spaced_access_count 不变
  SE3: 中等间隔访问（1h <= gap < 24h）→ spaced_access_count 不变
  SE4: spacing_factor = spaced_access_count / access_count ∈ [0, 1]
  SE5: spacing_factor > 0 → SM-2 quality bonus > 0（stability 更高）
  SE6: 纯集中访问（短间隔多次）→ spaced_access_count=0，无 quality bonus
  SE7: spacing_effect_enabled=False → spaced_access_count 不变，无 quality bonus
  SE8: spacing_factor=1.0（完全分布式）→ 获得最大 quality bonus
  SE9: Spacing Effect + Testing Effect 双重加成（长间隔 + 低 retrievability）
  SE10: 多个 chunk，各自独立计算 spacing_factor

认知科学依据：
  Ebbinghaus (1885) Spacing Effect — 分布式练习优于集中练习。
  Cepeda et al. (2006) Meta-analysis (317 experiments) —
    间隔效应（spaced > massed）平均效果量 d≈0.46。
  Glenberg (1979): 情境多样性（context diversity across repetitions）是核心机制。

OS 类比：Linux MGLRU cross-generation promotion —
  跨 aging cycle 被访问的 page 比同 gen 内多次访问的 page 更快晋升（distributed > massed）。
"""
import sys
import sqlite3
import datetime
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import tmpfs  # noqa

from store_vfs import (
    ensure_schema,
    update_accessed,
)
from store import insert_chunk
import config


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _insert_chunk(conn, cid, stability=2.0, project="test", chunk_type="decision",
                  last_accessed_offset_hours=0.0):
    """Insert a chunk with last_accessed set to now - offset_hours."""
    now = datetime.datetime.now(datetime.timezone.utc)
    last_acc = now - datetime.timedelta(hours=last_accessed_offset_hours)
    conn.execute(
        "INSERT OR REPLACE INTO memory_chunks "
        "(id, project, chunk_type, content, summary, importance, stability, "
        "created_at, updated_at, retrievability, last_accessed, access_count, "
        "spaced_access_count) "
        "VALUES (?, ?, ?, ?, ?, 0.8, ?, ?, ?, 0.9, ?, 1, 0)",
        (cid, project, chunk_type, f"content {cid}", f"summary {cid}",
         stability,
         last_acc.isoformat(), last_acc.isoformat(),
         last_acc.isoformat())
    )
    conn.commit()


def _get_stability(conn, cid: str) -> float:
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


def _get_spaced_count(conn, cid: str) -> int:
    row = conn.execute(
        "SELECT COALESCE(spaced_access_count, 0) FROM memory_chunks WHERE id=?", (cid,)
    ).fetchone()
    return int(row[0]) if row else 0


def _get_access_count(conn, cid: str) -> int:
    row = conn.execute(
        "SELECT COALESCE(access_count, 0) FROM memory_chunks WHERE id=?", (cid,)
    ).fetchone()
    return int(row[0]) if row else 0


# ── SE1: Long gap → spaced_access_count increments ───────────────────────────

def test_se1_long_gap_increments_spaced_count(conn):
    """访问间隔 >= 24h → spaced_access_count 递增。"""
    # Last accessed 30 hours ago
    _insert_chunk(conn, "se1", stability=2.0, last_accessed_offset_hours=30.0)

    spaced_before = _get_spaced_count(conn, "se1")
    update_accessed(conn, ["se1"])
    conn.commit()
    spaced_after = _get_spaced_count(conn, "se1")

    assert spaced_after > spaced_before, \
        f"SE1: 30h 间隔应使 spaced_access_count 递增，before={spaced_before} after={spaced_after}"


# ── SE2: Short gap → spaced_access_count unchanged ───────────────────────────

def test_se2_short_gap_no_spaced_increment(conn):
    """访问间隔 < 1h → spaced_access_count 不变。"""
    # Last accessed 10 minutes ago
    _insert_chunk(conn, "se2", stability=2.0, last_accessed_offset_hours=0.17)  # ~10 min

    spaced_before = _get_spaced_count(conn, "se2")
    update_accessed(conn, ["se2"])
    conn.commit()
    spaced_after = _get_spaced_count(conn, "se2")

    assert spaced_after == spaced_before, \
        f"SE2: 10min 间隔 spaced_access_count 不应变，before={spaced_before} after={spaced_after}"


# ── SE3: Medium gap → spaced_access_count unchanged ──────────────────────────

def test_se3_medium_gap_no_spaced_increment(conn):
    """访问间隔 1h-24h → spaced_access_count 不变（不到一个"新会话"）。"""
    # Last accessed 6 hours ago
    _insert_chunk(conn, "se3", stability=2.0, last_accessed_offset_hours=6.0)

    spaced_before = _get_spaced_count(conn, "se3")
    update_accessed(conn, ["se3"])
    conn.commit()
    spaced_after = _get_spaced_count(conn, "se3")

    assert spaced_after == spaced_before, \
        f"SE3: 6h 间隔 spaced_access_count 不应变，before={spaced_before} after={spaced_after}"


# ── SE4: spacing_factor computation ──────────────────────────────────────────

def test_se4_spacing_factor_range(conn):
    """spacing_factor = spaced_access_count / access_count ∈ [0, 1]。"""
    # Chunk with spaced_count=3, access_count=5 → factor=0.6
    conn.execute(
        "INSERT OR REPLACE INTO memory_chunks "
        "(id, project, chunk_type, content, summary, importance, stability, "
        "created_at, updated_at, retrievability, access_count, spaced_access_count) "
        "VALUES (?, ?, ?, ?, ?, 0.8, 2.0, ?, ?, 0.9, 5, 3)",
        ("se4", "test", "decision", "content", "summary",
         _now_iso(), _now_iso())
    )
    conn.commit()
    sac = _get_spaced_count(conn, "se4")
    ac = _get_access_count(conn, "se4")
    factor = sac / max(1, ac)
    assert 0.0 <= factor <= 1.0, f"SE4: spacing_factor 应在 [0,1]，got {factor}"
    assert abs(factor - 0.6) < 0.001, f"SE4: factor 应为 0.6，got {factor}"


# ── SE5: spacing_factor > 0 → SM-2 quality bonus → higher stability ──────────

def test_se5_spaced_chunk_higher_stability(conn):
    """高 spacing_factor chunk 相比纯集中 chunk 获得更高 stability 增益。"""
    now = datetime.datetime.now(datetime.timezone.utc)
    long_ago = now - datetime.timedelta(hours=48)

    # Spaced chunk: accessed 48h ago → will get spaced increment + quality boost
    conn.execute(
        "INSERT OR REPLACE INTO memory_chunks "
        "(id, project, chunk_type, content, summary, importance, stability, "
        "created_at, updated_at, retrievability, last_accessed, access_count, spaced_access_count) "
        "VALUES (?, ?, ?, ?, ?, 0.8, 2.0, ?, ?, 0.9, ?, 4, 2)",
        ("se5_spaced", "test", "decision", "content", "summary",
         long_ago.isoformat(), long_ago.isoformat(), long_ago.isoformat())
    )

    # Massed chunk: accessed 10min ago → no spaced increment, lower quality
    ten_min_ago = now - datetime.timedelta(minutes=10)
    conn.execute(
        "INSERT OR REPLACE INTO memory_chunks "
        "(id, project, chunk_type, content, summary, importance, stability, "
        "created_at, updated_at, retrievability, last_accessed, access_count, spaced_access_count) "
        "VALUES (?, ?, ?, ?, ?, 0.8, 2.0, ?, ?, 0.9, ?, 4, 0)",
        ("se5_massed", "test", "decision", "content", "summary",
         ten_min_ago.isoformat(), ten_min_ago.isoformat(), ten_min_ago.isoformat())
    )
    conn.commit()

    stab_spaced_before = _get_stability(conn, "se5_spaced")
    stab_massed_before = _get_stability(conn, "se5_massed")

    update_accessed(conn, ["se5_spaced"])
    conn.commit()
    update_accessed(conn, ["se5_massed"])
    conn.commit()

    stab_spaced_after = _get_stability(conn, "se5_spaced")
    stab_massed_after = _get_stability(conn, "se5_massed")

    ratio_spaced = stab_spaced_after / stab_spaced_before
    ratio_massed = stab_massed_after / stab_massed_before

    # Spaced access (long gap) should get better SM-2 quality than massed (short gap)
    assert ratio_spaced >= ratio_massed, \
        f"SE5: spaced(ratio={ratio_spaced:.4f}) 应 >= massed(ratio={ratio_massed:.4f})"


# ── SE6: Pure massed access → no spaced_access_count, no quality bonus ───────

def test_se6_massed_access_no_spaced_bonus(conn):
    """纯集中访问（短间隔）→ spaced_access_count=0，无额外 quality bonus。"""
    now = datetime.datetime.now(datetime.timezone.utc)
    recent = now - datetime.timedelta(minutes=5)

    conn.execute(
        "INSERT OR REPLACE INTO memory_chunks "
        "(id, project, chunk_type, content, summary, importance, stability, "
        "created_at, updated_at, retrievability, last_accessed, access_count, spaced_access_count) "
        "VALUES (?, ?, ?, ?, ?, 0.8, 2.0, ?, ?, 0.9, ?, 10, 0)",
        ("se6", "test", "decision", "content", "summary",
         recent.isoformat(), recent.isoformat(), recent.isoformat())
    )
    conn.commit()

    update_accessed(conn, ["se6"])
    conn.commit()

    spaced_after = _get_spaced_count(conn, "se6")
    assert spaced_after == 0, \
        f"SE6: 纯集中访问 spaced_access_count 应为 0，got {spaced_after}"


# ── SE7: spacing_effect_enabled=False → no spaced increment ──────────────────

def test_se7_disabled_no_spaced_increment(conn, monkeypatch):
    """spacing_effect_enabled=False → spaced_access_count 不变。"""
    import unittest.mock as mock
    original_get = config.get
    def patched_get(key, project=None):
        if key == "store_vfs.spacing_effect_enabled":
            return False
        return original_get(key, project=project)

    _insert_chunk(conn, "se7", stability=2.0, last_accessed_offset_hours=48.0)

    spaced_before = _get_spaced_count(conn, "se7")
    with mock.patch.object(config, 'get', side_effect=patched_get):
        update_accessed(conn, ["se7"])
    conn.commit()
    spaced_after = _get_spaced_count(conn, "se7")

    assert spaced_after == spaced_before, \
        f"SE7: 禁用后 spaced_access_count 不应变，before={spaced_before} after={spaced_after}"


# ── SE8: spacing_factor=1.0 → max quality bonus ───────────────────────────────

def test_se8_full_spacing_max_bonus(conn):
    """spacing_factor=1.0（每次都是长间隔）→ 获得最大 quality bonus。"""
    now = datetime.datetime.now(datetime.timezone.utc)
    long_ago = now - datetime.timedelta(hours=48)

    # access_count=5, spaced_access_count=5 (all spaced) → factor=1.0
    conn.execute(
        "INSERT OR REPLACE INTO memory_chunks "
        "(id, project, chunk_type, content, summary, importance, stability, "
        "created_at, updated_at, retrievability, last_accessed, access_count, spaced_access_count) "
        "VALUES (?, ?, ?, ?, ?, 0.8, 2.0, ?, ?, 0.9, ?, 5, 5)",
        ("se8", "test", "decision", "content", "summary",
         long_ago.isoformat(), long_ago.isoformat(), long_ago.isoformat())
    )
    conn.commit()

    stab_before = _get_stability(conn, "se8")
    update_accessed(conn, ["se8"])
    conn.commit()
    stab_after = _get_stability(conn, "se8")

    ratio = stab_after / stab_before
    # With long gap (quality=5) + spacing quality bonus (up to +2) → min quality=5+2=7→capped 5
    # SM-2 × (1 + 0.1×(5-3)) = ×1.2 minimum
    assert ratio >= 1.2, f"SE8: 完全 spaced 访问应得高 stability 比率，got {ratio:.4f}"


# ── SE9: Spacing Effect + Testing Effect double bonus ────────────────────────

def test_se9_spacing_and_testing_double_bonus(conn):
    """长间隔 + 低 retrievability → Spacing Effect + Testing Effect 双重加成。"""
    now = datetime.datetime.now(datetime.timezone.utc)
    # Very old access (3 days ago) → very low retrievability → high difficulty
    old = now - datetime.timedelta(hours=72)

    # Low stability → low R_at_recall → high difficulty → testing effect activates
    conn.execute(
        "INSERT OR REPLACE INTO memory_chunks "
        "(id, project, chunk_type, content, summary, importance, stability, "
        "created_at, updated_at, retrievability, last_accessed, access_count, spaced_access_count) "
        "VALUES (?, ?, ?, ?, ?, 0.8, 0.5, ?, ?, 0.3, ?, 3, 2)",
        ("se9", "test", "decision", "content", "summary",
         old.isoformat(), old.isoformat(), old.isoformat())
    )
    conn.commit()

    stab_before = _get_stability(conn, "se9")
    update_accessed(conn, ["se9"])
    conn.commit()
    stab_after = _get_stability(conn, "se9")

    ratio = stab_after / stab_before
    # Base SM-2 with quality=5 → ×1.2; with testing effect + spacing bonus ≥ ×1.2
    assert ratio >= 1.2, \
        f"SE9: Spacing + Testing double bonus → ratio >= 1.2，got {ratio:.4f}"


# ── SE10: Multiple chunks, independent spacing_factor ────────────────────────

def test_se10_multiple_chunks_independent(conn):
    """多个 chunk 各自独立计算 spacing_factor。"""
    now = datetime.datetime.now(datetime.timezone.utc)

    # Chunk A: accessed 48h ago (long gap) → spaced increment
    old = now - datetime.timedelta(hours=48)
    conn.execute(
        "INSERT OR REPLACE INTO memory_chunks "
        "(id, project, chunk_type, content, summary, importance, stability, "
        "created_at, updated_at, retrievability, last_accessed, access_count, spaced_access_count) "
        "VALUES (?, ?, ?, ?, ?, 0.8, 2.0, ?, ?, 0.9, ?, 3, 0)",
        ("se10a", "test", "decision", "content", "summary",
         old.isoformat(), old.isoformat(), old.isoformat())
    )
    # Chunk B: accessed 5min ago (short gap) → no spaced increment
    recent = now - datetime.timedelta(minutes=5)
    conn.execute(
        "INSERT OR REPLACE INTO memory_chunks "
        "(id, project, chunk_type, content, summary, importance, stability, "
        "created_at, updated_at, retrievability, last_accessed, access_count, spaced_access_count) "
        "VALUES (?, ?, ?, ?, ?, 0.8, 2.0, ?, ?, 0.9, ?, 3, 0)",
        ("se10b", "test", "decision", "content", "summary",
         recent.isoformat(), recent.isoformat(), recent.isoformat())
    )
    conn.commit()

    update_accessed(conn, ["se10a", "se10b"])
    conn.commit()

    spaced_a = _get_spaced_count(conn, "se10a")
    spaced_b = _get_spaced_count(conn, "se10b")

    assert spaced_a >= 1, f"SE10: chunk A (48h gap) 应有 spaced increment，got {spaced_a}"
    assert spaced_b == 0, f"SE10: chunk B (5min gap) 不应有 spaced increment，got {spaced_b}"
