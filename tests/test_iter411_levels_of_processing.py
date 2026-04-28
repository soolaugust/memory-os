"""
test_iter411_levels_of_processing.py — iter411: Levels of Processing (Craik & Lockhart 1972)

覆盖：
  LOP1: compute_encoding_depth — 多实体(>=8) 返回 1.0
  LOP2: compute_encoding_depth — 中等实体(5-7) 返回 0.7
  LOP3: compute_encoding_depth — 少量实体(3-4) 返回 0.4
  LOP4: compute_encoding_depth — 极少实体(1-2) 返回 0.1
  LOP5: compute_encoding_depth — 空/None 返回 0.0
  LOP6: depth_stability_bonus — 高深度(≥0.80) 返回 base × 0.15
  LOP7: depth_stability_bonus — 中等深度 线性插值
  LOP8: depth_stability_bonus — 低深度 返回 0.0
  LOP9: depth_stability_bonus — None/invalid 安全返回 0.0
  LOP10: apply_depth_effect — 多实体 chunk stability 被提升
  LOP11: apply_depth_effect — 无实体 chunk stability 不变

认知科学依据：
  Craik & Lockhart (1972) Levels of Processing: 语义网络密度 → 更深加工 → 更强记忆。
  Hyde & Jenkins (1973): 语义导向任务产生更好的记忆保留。

OS 类比：Linux NUMA-aware page allocation —
  本地 NUMA 节点访问延迟最低，类比：语义网络连接越多，检索"延迟"越低。
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
    compute_encoding_depth,
    depth_stability_bonus,
    apply_depth_effect,
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


def _insert_with_ctx(conn, chunk_id, encode_context="", project="test"):
    now = _now()
    conn.execute(
        "INSERT OR REPLACE INTO memory_chunks "
        "(id, project, chunk_type, content, summary, importance, retrievability, "
        "stability, encode_context, created_at, updated_at) "
        "VALUES (?, ?, 'decision', 'content', 'summary', 0.7, 0.5, 1.0, ?, ?, ?)",
        (chunk_id, project, encode_context, now, now)
    )
    conn.commit()


# ══════════════════════════════════════════════════════════════════════
# 1. compute_encoding_depth 纯函数测试
# ══════════════════════════════════════════════════════════════════════

def test_lop1_many_entities_full_depth():
    """>=8 实体 → depth = 1.0。"""
    ctx = "redis,cache,performance,cluster,lru,eviction,replication,sentinel"
    assert compute_encoding_depth(ctx) == 1.0, f"LOP1: 8个实体应返回 1.0"


def test_lop2_medium_entities_point7():
    """5-7 实体 → depth = 0.7。"""
    ctx = "redis,cache,performance,cluster,lru"
    assert compute_encoding_depth(ctx) == 0.7, f"LOP2: 5个实体应返回 0.7"
    ctx7 = "redis,cache,performance,cluster,lru,eviction,replication"
    assert compute_encoding_depth(ctx7) == 0.7, f"LOP2: 7个实体应返回 0.7"


def test_lop3_few_entities_point4():
    """3-4 实体 → depth = 0.4。"""
    assert compute_encoding_depth("redis,cache,performance") == 0.4, "LOP3: 3实体=0.4"
    assert compute_encoding_depth("redis,cache,performance,cluster") == 0.4, "LOP3: 4实体=0.4"


def test_lop4_minimal_entities_point1():
    """1-2 实体 → depth = 0.1。"""
    assert compute_encoding_depth("redis") == 0.1, "LOP4: 1实体=0.1"
    assert compute_encoding_depth("redis,cache") == 0.1, "LOP4: 2实体=0.1"


def test_lop5_empty_context_zero():
    """空/None encode_context → 0.0。"""
    assert compute_encoding_depth("") == 0.0
    assert compute_encoding_depth(None) == 0.0
    assert compute_encoding_depth("   ") == 0.0


# ══════════════════════════════════════════════════════════════════════
# 2. depth_stability_bonus 测试
# ══════════════════════════════════════════════════════════════════════

def test_lop6_high_depth_max_bonus():
    """高深度(>=0.80) → base × 0.15。"""
    bonus = depth_stability_bonus(1.0, 1.0)
    assert 0.13 <= bonus <= 0.16, f"LOP6: 高深度 bonus 应约 0.15，got {bonus:.4f}"


def test_lop7_medium_depth_interpolated():
    """中等深度(0.50-0.80) → 线性插值。"""
    bonus_50 = depth_stability_bonus(0.50, 1.0)
    bonus_65 = depth_stability_bonus(0.65, 1.0)
    bonus_80 = depth_stability_bonus(0.80, 1.0)
    assert bonus_50 <= bonus_65 <= bonus_80, (
        f"LOP7: bonus 应单调递增: {bonus_50:.4f} <= {bonus_65:.4f} <= {bonus_80:.4f}"
    )
    # depth=0.50 时 bonus 应接近 0.08
    assert 0.06 <= bonus_50 <= 0.10, f"LOP7: depth=0.50 bonus 应约 0.08，got {bonus_50:.4f}"


def test_lop8_low_depth_no_bonus():
    """低深度(< 0.20) → 无加成。"""
    assert depth_stability_bonus(0.0, 1.0) == 0.0
    assert depth_stability_bonus(0.15, 1.0) == 0.0


def test_lop9_invalid_inputs_safe():
    """None/invalid 输入安全返回 0.0。"""
    assert depth_stability_bonus(None, 1.0) == 0.0
    assert depth_stability_bonus(1.0, None) == 0.0
    assert depth_stability_bonus("bad", 1.0) == 0.0
    assert depth_stability_bonus(0.0, 1.0) == 0.0
    assert depth_stability_bonus(1.0, 0.0) == 0.0


# ══════════════════════════════════════════════════════════════════════
# 3. apply_depth_effect 集成测试
# ══════════════════════════════════════════════════════════════════════

def test_lop10_rich_context_chunk_boosted(conn):
    """多实体 chunk stability 被提升。"""
    _insert_with_ctx(conn, "lop10_chunk",
                     "redis,cache,performance,cluster,lru,eviction,replication,sentinel")
    orig = conn.execute(
        "SELECT stability FROM memory_chunks WHERE id='lop10_chunk'"
    ).fetchone()[0]
    new_s = apply_depth_effect(conn, "lop10_chunk", base_stability=orig)
    conn.commit()
    db_s = conn.execute(
        "SELECT stability FROM memory_chunks WHERE id='lop10_chunk'"
    ).fetchone()[0]
    assert new_s > orig, f"LOP10: 多实体 chunk stability 应被提升，got {new_s:.4f} vs {orig:.4f}"
    assert db_s > orig, f"LOP10: DB stability 应被更新，got {db_s:.4f}"


def test_lop11_empty_context_no_change(conn):
    """无 encode_context chunk stability 不变。"""
    now = _now()
    conn.execute(
        "INSERT OR REPLACE INTO memory_chunks "
        "(id, project, chunk_type, content, summary, importance, retrievability, "
        "stability, created_at, updated_at) "
        "VALUES ('lop11_chunk', 'test', 'decision', 'content', 'summary', 0.7, 0.5, 1.0, ?, ?)",
        (now, now)
    )
    conn.commit()
    orig = conn.execute(
        "SELECT stability FROM memory_chunks WHERE id='lop11_chunk'"
    ).fetchone()[0]
    new_s = apply_depth_effect(conn, "lop11_chunk", base_stability=orig)
    assert new_s == orig, f"LOP11: 无实体 chunk stability 不变，got {new_s:.4f} vs {orig:.4f}"
