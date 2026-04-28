"""
test_retroactive_interference.py — iter405: Retroactive Interference (RI)

覆盖：
  RI1: compute_recency_penalty — 旧 chunk + 多新 chunk 返回 penalty > 0
  RI2: compute_recency_penalty — 新 chunk（age < 7天）不受惩罚
  RI3: compute_recency_penalty — 无新 chunk 时 penalty = 0
  RI4: compute_recency_penalty — penalty 上限 0.15
  RI5: compute_recency_penalty — 相似度为 0 时无惩罚
  RI6: compute_recency_penalty — age 越大、count 越多 → penalty 越大
  RI7: get_newer_same_topic_count — 找到更新且相似的 chunk
  RI8: get_newer_same_topic_count — 无更新 chunk 时返回 (0, 0.0)
  RI9: get_newer_same_topic_count — overlap_threshold 过滤低相似度
  RI10: get_newer_same_topic_count — 空/None 输入安全
  RI11: compute_recency_penalty range [0.0, 0.15]
  RI12: compute_recency_penalty — None/invalid 输入安全返回 0.0

认知科学依据：
  Underwood (1957) Proactive Inhibition and Forgetting:
    学习 List B 后，回忆 List A 成功率下降（RI）。
    新记忆和旧记忆在同一语义领域竞争检索路径。
  McGeoch & Irion (1952) The Psychology of Human Learning:
    干扰效应强度 × 材料相似度（相似度越高 → 干扰越强）。

OS 类比：Linux MGLRU generation demotion —
  年龄较大的 pages 在新 pages 涌入时面临更大驱逐压力（recency bias）；
  chunk age 越大、同主题新 chunk 越多 → recency_penalty 越大 → 检索分下降。
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
    compute_recency_penalty,
    get_newer_same_topic_count,
)


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _iso(days_ago=0, minutes_ago=0):
    dt = datetime.now(timezone.utc) - timedelta(days=days_ago, minutes=minutes_ago)
    return dt.isoformat()


def _insert_chunk(conn, chunk_id, encode_context="", created_at=None, project="test"):
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO memory_chunks "
        "(id, project, chunk_type, content, summary, importance, retrievability, "
        "stability, encode_context, created_at, updated_at) "
        "VALUES (?, ?, 'decision', 'content', 'summary', 0.7, 0.5, 1.0, ?, ?, ?)",
        (chunk_id, project, encode_context, created_at or now, now)
    )
    conn.commit()


# ══════════════════════════════════════════════════════════════════════
# 1. compute_recency_penalty 纯函数测试
# ══════════════════════════════════════════════════════════════════════

def test_ri1_old_chunk_with_newer_gets_penalty():
    """旧 chunk（age > 7天）+ 有新 chunk → penalty > 0。"""
    penalty = compute_recency_penalty(
        chunk_age_days=30.0,
        newer_same_topic_count=3,
        similarity=0.5,
    )
    assert penalty > 0.0, f"RI1: 旧 chunk + 3 新 chunk penalty 应 > 0，got {penalty}"


def test_ri2_young_chunk_no_penalty():
    """新 chunk（age < 7天）不受 RI 惩罚。"""
    penalty = compute_recency_penalty(
        chunk_age_days=3.0,  # < 7 天阈值
        newer_same_topic_count=10,
        similarity=1.0,
    )
    assert penalty == 0.0, f"RI2: 新 chunk（age=3天）不应受惩罚，got {penalty}"


def test_ri3_no_newer_chunk_no_penalty():
    """无新 chunk 时 penalty = 0。"""
    penalty = compute_recency_penalty(
        chunk_age_days=60.0,
        newer_same_topic_count=0,
        similarity=0.8,
    )
    assert penalty == 0.0, f"RI3: 无新 chunk 时 penalty 应为 0，got {penalty}"


def test_ri4_penalty_capped_at_max():
    """penalty 上限 0.15。"""
    penalty = compute_recency_penalty(
        chunk_age_days=365.0,   # 极旧
        newer_same_topic_count=100,  # 极多新 chunk
        similarity=1.0,             # 完全相同主题
    )
    assert penalty <= 0.15, f"RI4: penalty 上限 0.15，got {penalty}"


def test_ri5_zero_similarity_no_penalty():
    """相似度为 0（完全不同主题）时无惩罚。"""
    penalty = compute_recency_penalty(
        chunk_age_days=60.0,
        newer_same_topic_count=5,
        similarity=0.0,
    )
    assert penalty == 0.0, f"RI5: 相似度 0 时 penalty 应为 0，got {penalty}"


def test_ri6_older_more_count_higher_penalty():
    """age 越大、count 越多 → penalty 越大（单调性）。"""
    p1 = compute_recency_penalty(20.0, 2, 0.5)
    p2 = compute_recency_penalty(60.0, 5, 0.5)
    assert p2 > p1, f"RI6: 更旧({p2:.4f}) 应 > 较新({p1:.4f})"


def test_ri11_penalty_range():
    """compute_recency_penalty 输出范围 [0.0, 0.15]。"""
    for age in [0, 3, 7, 10, 30, 60, 365]:
        for count in [0, 1, 3, 10]:
            for sim in [0.0, 0.3, 0.5, 1.0]:
                p = compute_recency_penalty(age, count, sim)
                assert 0.0 <= p <= 0.15, (
                    f"RI11: age={age}, count={count}, sim={sim} → penalty={p:.4f} 应在 [0.0, 0.15]"
                )


def test_ri12_invalid_inputs_safe():
    """None/invalid 输入安全返回 0.0。"""
    assert compute_recency_penalty(None, 3, 0.5) == 0.0
    assert compute_recency_penalty(30.0, None, 0.5) == 0.0
    assert compute_recency_penalty(30.0, 3, None) == 0.0
    assert compute_recency_penalty("bad", 3, 0.5) == 0.0


# ══════════════════════════════════════════════════════════════════════
# 2. get_newer_same_topic_count 测试
# ══════════════════════════════════════════════════════════════════════

def test_ri7_finds_newer_similar_chunks(conn):
    """找到更新且 encode_context 相似的 chunk。"""
    # 旧 chunk（30天前）
    _insert_chunk(conn, "old_chunk", "redis,cache,performance,cluster",
                  created_at=_iso(days_ago=30))
    # 新 chunk（今天，相同主题）
    _insert_chunk(conn, "new_chunk1", "redis,cache,performance,tuning",
                  created_at=_iso(minutes_ago=5))
    _insert_chunk(conn, "new_chunk2", "redis,cache,eviction,lru",
                  created_at=_iso(minutes_ago=10))

    count, avg_overlap = get_newer_same_topic_count(conn, "old_chunk", "test")
    assert count >= 2, f"RI7: 应找到 >= 2 个更新同主题 chunk，got {count}"
    assert avg_overlap > 0.20, f"RI7: 平均重叠度应 > 0.20，got {avg_overlap}"


def test_ri8_no_newer_chunks_returns_zero(conn):
    """无更新 chunk 时返回 (0, 0.0)。"""
    _insert_chunk(conn, "ri8_only", "redis,cache,performance",
                  created_at=_iso(days_ago=30))
    count, avg_overlap = get_newer_same_topic_count(conn, "ri8_only", "test")
    assert count == 0, f"RI8: 无更新 chunk 应返回 count=0，got {count}"
    assert avg_overlap == 0.0, f"RI8: avg_overlap 应为 0.0，got {avg_overlap}"


def test_ri9_low_similarity_filtered_out(conn):
    """低相似度（< threshold）的新 chunk 不计入 count。"""
    _insert_chunk(conn, "ri9_old", "redis,cache,performance",
                  created_at=_iso(days_ago=30))
    # 完全不同主题的新 chunk
    _insert_chunk(conn, "ri9_new", "machine_learning,neural_network,training",
                  created_at=_iso(minutes_ago=5))

    count, _ = get_newer_same_topic_count(
        conn, "ri9_old", "test", overlap_threshold=0.25
    )
    assert count == 0, (
        f"RI9: 完全不同主题的新 chunk 不应被计入，got count={count}"
    )


def test_ri10_empty_inputs_safe(conn):
    """空/None 输入安全返回 (0, 0.0)。"""
    c, a = get_newer_same_topic_count(conn, "", "test")
    assert c == 0 and a == 0.0

    c2, a2 = get_newer_same_topic_count(conn, "chunk_x", "")
    assert c2 == 0 and a2 == 0.0

    c3, a3 = get_newer_same_topic_count(conn, None, "test")
    assert c3 == 0 and a3 == 0.0


def test_ri_integration(conn):
    """全链路：旧 chunk + 多新同主题 chunk → penalty > 0（集成验证）。"""
    # 旧 chunk（30天前，redis 主题）
    _insert_chunk(conn, "ri_old", "cache,cluster,performance,redis,setup",
                  created_at=_iso(days_ago=30))
    # 多个新 chunk（今天，redis 主题）
    for i in range(3):
        _insert_chunk(conn, f"ri_new_{i}", f"cache,performance,redis,tuning_{i}",
                      created_at=_iso(minutes_ago=i+1))

    count, avg_overlap = get_newer_same_topic_count(conn, "ri_old", "test")
    penalty = compute_recency_penalty(30.0, count, avg_overlap)

    assert count >= 3, f"RI_integration: 应找到 3 个新 chunk，got {count}"
    assert penalty > 0.0, f"RI_integration: 30天旧 chunk + 3个新同主题 chunk 应有 penalty > 0，got {penalty}"
    assert penalty <= 0.15, f"RI_integration: penalty 不超过 0.15，got {penalty}"
