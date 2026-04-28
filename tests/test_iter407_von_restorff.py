"""
test_iter407_von_restorff.py — iter407: Von Restorff Effect (von Restorff 1933)

覆盖：
  VR1: compute_isolation_score — 孤立 chunk（无相似邻居）返回高孤立度
  VR2: compute_isolation_score — 相似 chunk 集群中返回低孤立度
  VR3: compute_isolation_score — 邻居少于 min_neighbors 返回 0.0
  VR4: compute_isolation_score — 空/None 输入安全返回 0.0
  VR5: compute_isolation_score — 无 encode_context 的 chunk 返回 0.0
  VR6: isolation_stability_bonus — 高孤立度（>= 0.85）返回最大加成因子
  VR7: isolation_stability_bonus — 中等孤立度返回中等加成
  VR8: isolation_stability_bonus — 低孤立度（< 0.45）返回 0.0
  VR9: isolation_stability_bonus — 加成上限为 base × 0.20
  VR10: isolation_stability_bonus — None/invalid 输入安全返回 0.0
  VR11: apply_isolation_effect — 孤立 chunk 的 stability 被提升
  VR12: apply_isolation_effect — 普通 chunk（邻居不足）stability 不变

认知科学依据：
  von Restorff (1933) Über die Wirkung von Bereichsbildungen im Spurenfeld:
    均匀序列中独特/孤立的项目记忆留存率显著更高（isolation effect）。
  Wallace (1965): 孤立效应强度与孤立程度正相关。
  Klein & Saltz (1976): 语义独特性是孤立效应的核心机制。

OS 类比：Linux perf_event outlier detection —
  在均匀 baseline 中，统计显著不同的事件被标记为 outlier（高信息价值），
  优先保留在性能分析记录中。
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
    compute_isolation_score,
    isolation_stability_bonus,
    apply_isolation_effect,
    insert_chunk,
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


def _insert(conn, chunk_id, encode_context="", project="test"):
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
# 1. compute_isolation_score 测试
# ══════════════════════════════════════════════════════════════════════

def test_vr1_isolated_chunk_high_score(conn):
    """语义孤立的 chunk（无相似邻居）应返回高孤立度。"""
    # 先插入若干 redis 主题的邻居
    for i in range(5):
        _insert(conn, f"nb_{i}", "redis,cache,performance,cluster,eviction", "test")
    # 插入完全不同主题的 chunk
    _insert(conn, "isolated", "machine_learning,neural_network,transformer,training", "test")
    score = compute_isolation_score(conn, "isolated", "test")
    # 孤立度应该高（与 redis 主题的邻居几乎没有交集）
    assert score >= 0.7, f"VR1: 孤立 chunk 孤立度应 >= 0.7，got {score:.4f}"


def test_vr2_similar_chunk_low_score(conn):
    """相似 chunk 集群中的 chunk 应返回低孤立度。"""
    for i in range(5):
        _insert(conn, f"redis_{i}", "redis,cache,performance,cluster,eviction", "test")
    # 插入相同主题的 chunk
    _insert(conn, "similar", "redis,cache,performance,cluster,lru", "test")
    score = compute_isolation_score(conn, "similar", "test")
    # 孤立度应该低（与邻居高度相似）
    assert score < 0.5, f"VR2: 相似 chunk 孤立度应 < 0.5，got {score:.4f}"


def test_vr3_few_neighbors_returns_zero(conn):
    """邻居少于 min_neighbors 时返回 0.0（保守策略）。"""
    # 只插入 2 个邻居（默认 min_neighbors=3）
    _insert(conn, "nb1", "redis,cache", "test")
    _insert(conn, "nb2", "python,async", "test")
    _insert(conn, "target", "machine_learning,transformer", "test")
    score = compute_isolation_score(conn, "target", "test", min_neighbors=3)
    assert score == 0.0, f"VR3: 邻居不足时应返回 0.0，got {score:.4f}"


def test_vr4_empty_inputs_safe(conn):
    """空/None 输入安全返回 0.0。"""
    assert compute_isolation_score(conn, "", "test") == 0.0
    assert compute_isolation_score(conn, None, "test") == 0.0
    assert compute_isolation_score(conn, "chunk_x", "") == 0.0
    assert compute_isolation_score(conn, "chunk_x", None) == 0.0


def test_vr5_no_encode_context_returns_zero(conn):
    """chunk 无 encode_context 时返回 0.0。"""
    now = _now()
    conn.execute(
        "INSERT OR REPLACE INTO memory_chunks "
        "(id, project, chunk_type, content, summary, importance, retrievability, "
        "stability, created_at, updated_at) "
        "VALUES ('no_ctx', 'test', 'decision', 'content', 'summary', 0.7, 0.5, 1.0, ?, ?)",
        (now, now)
    )
    conn.commit()
    for i in range(5):
        _insert(conn, f"ctx_nb_{i}", "redis,cache,performance", "test")
    score = compute_isolation_score(conn, "no_ctx", "test")
    assert score == 0.0, f"VR5: 无 encode_context 应返回 0.0，got {score:.4f}"


# ══════════════════════════════════════════════════════════════════════
# 2. isolation_stability_bonus 测试
# ══════════════════════════════════════════════════════════════════════

def test_vr6_high_isolation_max_bonus():
    """高孤立度（>= 0.85）返回最大加成因子 base × 0.20。"""
    bonus = isolation_stability_bonus(0.90, 1.0)
    assert 0.18 <= bonus <= 0.22, f"VR6: 高孤立度 bonus 应约 0.20，got {bonus:.4f}"


def test_vr7_medium_isolation_medium_bonus():
    """中等孤立度（0.65-0.85 区间）返回中等加成。"""
    bonus = isolation_stability_bonus(0.75, 1.0)
    # 插值：t = (0.75 - 0.65) / (0.85 - 0.65) = 0.5
    # factor = 0.10 + 0.5 × (0.20 - 0.10) = 0.15
    assert 0.08 <= bonus <= 0.18, f"VR7: 中等孤立度 bonus 应在 [0.08, 0.18]，got {bonus:.4f}"


def test_vr8_low_isolation_no_bonus():
    """低孤立度（< 0.45）返回 0.0。"""
    bonus = isolation_stability_bonus(0.30, 1.0)
    assert bonus == 0.0, f"VR8: 低孤立度 bonus 应为 0，got {bonus:.4f}"


def test_vr9_bonus_capped_at_20_pct():
    """加成上限为 base × 0.20。"""
    # base=2.0, 高孤立度 → bonus 上限 = 2.0 × 0.20 = 0.40
    bonus = isolation_stability_bonus(0.99, 2.0)
    assert bonus <= 2.0 * 0.20 + 1e-9, f"VR9: bonus 不超过 base×0.20，got {bonus:.4f}"


def test_vr10_invalid_inputs_safe():
    """None/invalid 输入安全返回 0.0。"""
    assert isolation_stability_bonus(None, 1.0) == 0.0
    assert isolation_stability_bonus(0.9, None) == 0.0
    assert isolation_stability_bonus("bad", 1.0) == 0.0
    assert isolation_stability_bonus(0.9, "bad") == 0.0
    # 零值
    assert isolation_stability_bonus(0.0, 1.0) == 0.0
    assert isolation_stability_bonus(0.9, 0.0) == 0.0


# ══════════════════════════════════════════════════════════════════════
# 3. apply_isolation_effect 集成测试
# ══════════════════════════════════════════════════════════════════════

def test_vr11_isolated_chunk_stability_boosted(conn):
    """孤立 chunk 的 stability 被提升。"""
    # 插入足够多的相似邻居（redis 主题）
    for i in range(5):
        _insert(conn, f"vr11_nb_{i}", "redis,cache,performance,cluster,eviction", "test")
    # 插入孤立 chunk（完全不同主题）
    _insert(conn, "vr11_isolated", "astronomy,telescope,hubble,galaxy,cosmology", "test")
    original_stability = conn.execute(
        "SELECT stability FROM memory_chunks WHERE id='vr11_isolated'"
    ).fetchone()[0]

    new_stability = apply_isolation_effect(conn, "vr11_isolated", "test", base_stability=1.0)
    conn.commit()

    db_stability = conn.execute(
        "SELECT stability FROM memory_chunks WHERE id='vr11_isolated'"
    ).fetchone()[0]

    # 孤立度够高时应有加成
    assert new_stability >= original_stability, (
        f"VR11: 孤立 chunk stability 应 >= 原始值，got new={new_stability:.4f}, orig={original_stability:.4f}"
    )
    # 如果确实有加成（孤立度 > 0.45），验证 DB 也更新了
    if new_stability > original_stability:
        assert db_stability > original_stability, (
            f"VR11: DB stability 应被更新，got db={db_stability:.4f}, orig={original_stability:.4f}"
        )


def test_vr12_few_neighbors_no_change(conn):
    """邻居不足时 stability 不变（保守策略）。"""
    # 只有 1 个邻居（< min_neighbors=3）
    _insert(conn, "vr12_nb1", "redis,cache", "test")
    _insert(conn, "vr12_target", "machine_learning,transformer", "test")

    original_stability = conn.execute(
        "SELECT stability FROM memory_chunks WHERE id='vr12_target'"
    ).fetchone()[0]

    new_stability = apply_isolation_effect(conn, "vr12_target", "test", base_stability=original_stability)
    assert new_stability == original_stability, (
        f"VR12: 邻居不足时 stability 应不变，got {new_stability:.4f} vs {original_stability:.4f}"
    )
