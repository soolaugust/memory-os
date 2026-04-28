"""
test_retrieval_induced_reconsolidation.py — iter395: Retrieval-Induced Reconsolidation

覆盖：
  RIR1: 高唤醒情绪记忆（emotional_weight > 0.4）再巩固效果更强
  RIR2: 情绪中性记忆（emotional_weight = 0）使用基础 boost
  RIR3: 首次召回（access_count ≤ 3）boost × 1.5（测试效果最强窗口）
  RIR4: 高频召回（access_count > 10）boost × 0.7（边际递减）
  RIR5: Jaccard 重叠度影响 boost 量（高重叠 > 低重叠）
  RIR6: importance 上限 0.98
  RIR7: Co-Retrieval Association Strengthening — entity_edge confidence 提升
  RIR8: 单 chunk 不触发 Co-Retrieval（至少 2 个才触发）
  RIR9: emotional_weight = 0.5 的 multiplier 正确（×1.25）
  RIR10: 空输入安全返回 0

认知科学依据：
  Nader et al. (2000) Memory Reconsolidation: reconsolidation requires protein synthesis
    记忆被提取后进入 labile state，这个窗口内记忆是可修改的。
  Roediger & Karpicke (2006) Test-Enhanced Learning: 首次成功检索带来最大记忆固化。
  McGaugh (2000) Emotional Enhancement of Memory: 杏仁核激活增强 LTP，情绪记忆再巩固效果更强。
  Hebb (1949) "neurons that fire together, wire together": 共同激活强化关联。

OS 类比：
  Linux ARC Cache — T2 晋升强度按热度梯度差异化。
  CPU Hardware Prefetcher — 学习共同命中的 cache line 对，构建 stride 预测表。
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
from store_vfs import (ensure_schema, reconsolidate, insert_chunk,
                       insert_edge)


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _insert_chunk(conn, chunk_id, summary="test chunk", importance=0.70,
                  emotional_weight=0.0, access_count=1, project="test_proj"):
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO memory_chunks "
        "(id, project, chunk_type, content, summary, importance, retrievability, "
        "emotional_weight, access_count, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, 0.80, ?, ?, ?, ?)",
        (chunk_id, project, "decision", f"content {summary}", summary,
         importance, emotional_weight, access_count, now, now)
    )
    conn.commit()


# ══════════════════════════════════════════════════════════════════════
# 1. Emotional Multiplier 测试
# ══════════════════════════════════════════════════════════════════════

def test_rir1_emotional_weight_amplifies_reconsolidation(conn):
    """高情绪记忆（ew=0.8）再巩固 boost 更大。"""
    # 两个 chunk：一个高情绪，一个中性，同样的 query 匹配度
    _insert_chunk(conn, "rir1_emotional", summary="崩溃 critical bug",
                  importance=0.70, emotional_weight=0.8, access_count=1)
    _insert_chunk(conn, "rir1_neutral", summary="崩溃 critical bug",
                  importance=0.70, emotional_weight=0.0, access_count=1)

    reconsolidate(conn, ["rir1_emotional", "rir1_neutral"],
                  query="崩溃 critical bug", project="test_proj")
    conn.commit()

    emotional_row = conn.execute(
        "SELECT importance FROM memory_chunks WHERE id='rir1_emotional'"
    ).fetchone()
    neutral_row = conn.execute(
        "SELECT importance FROM memory_chunks WHERE id='rir1_neutral'"
    ).fetchone()

    assert emotional_row["importance"] > neutral_row["importance"], (
        f"RIR1: 高情绪记忆 importance={emotional_row['importance']:.4f} "
        f"应 > 中性记忆 {neutral_row['importance']:.4f}"
    )
    # 两者都应上调（都被召回，都有 Jaccard 重叠）
    assert emotional_row["importance"] > 0.70, "RIR1: 高情绪记忆应上调"
    assert neutral_row["importance"] > 0.70, "RIR1: 中性记忆也应上调"


def test_rir2_neutral_emotion_uses_base_boost(conn):
    """中性情绪（ew=0.0）使用基础 boost，不额外放大。"""
    _insert_chunk(conn, "rir2_neutral", summary="实现用户登录功能",
                  importance=0.70, emotional_weight=0.0, access_count=2)

    n = reconsolidate(conn, ["rir2_neutral"],
                      query="实现用户登录功能", project="test_proj")
    conn.commit()
    assert n >= 1

    row = conn.execute(
        "SELECT importance FROM memory_chunks WHERE id='rir2_neutral'"
    ).fetchone()
    # 基础 boost 约 0.03 × 1.5（access_count=2 ≤ 3 → ×1.5）× 重叠率
    # 重叠率高（几乎完全匹配）→ boost ≈ 0.045 左右
    assert row["importance"] > 0.70, f"RIR2: 中性记忆应上调 > 0.70, got {row['importance']}"
    assert row["importance"] < 0.78, f"RIR2: 中性记忆 boost 不应过大 < 0.78, got {row['importance']}"


def test_rir9_emotional_weight_multiplier_calculation(conn):
    """ew=0.5 → multiplier = 1 + 0.5×0.5 = 1.25（仅当 ew > 0.4 时触发）。"""
    _insert_chunk(conn, "rir9_ew05", summary="严重错误 critical",
                  importance=0.70, emotional_weight=0.5, access_count=2)
    _insert_chunk(conn, "rir9_baseline", summary="严重错误 critical",
                  importance=0.70, emotional_weight=0.0, access_count=2)

    reconsolidate(conn, ["rir9_ew05", "rir9_baseline"],
                  query="严重错误 critical", project="test_proj")
    conn.commit()

    ew05_row = conn.execute(
        "SELECT importance FROM memory_chunks WHERE id='rir9_ew05'"
    ).fetchone()
    baseline_row = conn.execute(
        "SELECT importance FROM memory_chunks WHERE id='rir9_baseline'"
    ).fetchone()

    # ew=0.5 → multiplier 1.25，所以 boost_ew05 ≈ baseline_boost × 1.25
    boost_ew05 = ew05_row["importance"] - 0.70
    boost_baseline = baseline_row["importance"] - 0.70
    assert boost_baseline > 0, "RIR9: baseline should be boosted"
    ratio = boost_ew05 / boost_baseline
    # ratio should be approximately 1.25 (±10%)
    assert 1.10 <= ratio <= 1.45, (
        f"RIR9: ew=0.5 的 boost 比例应约 1.25，got {ratio:.3f} "
        f"(boost_ew05={boost_ew05:.4f}, boost_baseline={boost_baseline:.4f})"
    )


# ══════════════════════════════════════════════════════════════════════
# 2. Frequency Gradient 测试
# ══════════════════════════════════════════════════════════════════════

def test_rir3_first_retrieval_strongest(conn):
    """首次召回（access_count≤3）boost×1.5，比高频召回更强。"""
    _insert_chunk(conn, "rir3_new", summary="分布式系统架构设计",
                  importance=0.70, emotional_weight=0.0, access_count=1)
    _insert_chunk(conn, "rir3_old", summary="分布式系统架构设计",
                  importance=0.70, emotional_weight=0.0, access_count=15)

    reconsolidate(conn, ["rir3_new", "rir3_old"],
                  query="分布式系统架构设计", project="test_proj")
    conn.commit()

    new_row = conn.execute(
        "SELECT importance FROM memory_chunks WHERE id='rir3_new'"
    ).fetchone()
    old_row = conn.execute(
        "SELECT importance FROM memory_chunks WHERE id='rir3_old'"
    ).fetchone()

    assert new_row["importance"] > old_row["importance"], (
        f"RIR3: 首次召回({new_row['importance']:.4f}) 应 > 高频召回({old_row['importance']:.4f})"
    )


def test_rir4_high_frequency_diminishing_returns(conn):
    """高频召回（access_count>10）boost×0.7，比中等频率小。"""
    _insert_chunk(conn, "rir4_medium", summary="数据库连接池配置",
                  importance=0.70, emotional_weight=0.0, access_count=5)
    _insert_chunk(conn, "rir4_high", summary="数据库连接池配置",
                  importance=0.70, emotional_weight=0.0, access_count=20)

    reconsolidate(conn, ["rir4_medium", "rir4_high"],
                  query="数据库连接池配置", project="test_proj")
    conn.commit()

    medium_row = conn.execute(
        "SELECT importance FROM memory_chunks WHERE id='rir4_medium'"
    ).fetchone()
    high_row = conn.execute(
        "SELECT importance FROM memory_chunks WHERE id='rir4_high'"
    ).fetchone()

    assert medium_row["importance"] > high_row["importance"], (
        f"RIR4: 中等频率({medium_row['importance']:.4f}) 应 > 高频({high_row['importance']:.4f})"
    )


# ══════════════════════════════════════════════════════════════════════
# 3. Jaccard 重叠度影响
# ══════════════════════════════════════════════════════════════════════

def test_rir5_jaccard_overlap_affects_boost(conn):
    """高重叠 summary 获得更大 boost。"""
    _insert_chunk(conn, "rir5_high_overlap",
                  summary="Redis 缓存连接超时 timeout 配置",
                  importance=0.70, emotional_weight=0.0, access_count=2)
    _insert_chunk(conn, "rir5_low_overlap",
                  summary="用户登录注册功能实现",
                  importance=0.70, emotional_weight=0.0, access_count=2)

    reconsolidate(conn, ["rir5_high_overlap", "rir5_low_overlap"],
                  query="Redis 缓存连接超时 timeout", project="test_proj")
    conn.commit()

    high_row = conn.execute(
        "SELECT importance FROM memory_chunks WHERE id='rir5_high_overlap'"
    ).fetchone()
    low_row = conn.execute(
        "SELECT importance FROM memory_chunks WHERE id='rir5_low_overlap'"
    ).fetchone()

    assert high_row["importance"] > low_row["importance"], (
        f"RIR5: 高重叠({high_row['importance']:.4f}) 应 > 低重叠({low_row['importance']:.4f})"
    )


# ══════════════════════════════════════════════════════════════════════
# 4. importance 上限测试
# ══════════════════════════════════════════════════════════════════════

def test_rir6_importance_capped_at_max(conn):
    """importance 上限 0.98，不超过 max_importance。"""
    _insert_chunk(conn, "rir6_near_max",
                  summary="崩溃 critical fatal panic data loss",
                  importance=0.97, emotional_weight=1.0, access_count=1)

    reconsolidate(conn, ["rir6_near_max"],
                  query="崩溃 critical fatal panic data loss",
                  project="test_proj", max_importance=0.98)
    conn.commit()

    row = conn.execute(
        "SELECT importance FROM memory_chunks WHERE id='rir6_near_max'"
    ).fetchone()
    assert row["importance"] <= 0.98, (
        f"RIR6: importance 不应超过 0.98，got {row['importance']}"
    )


# ══════════════════════════════════════════════════════════════════════
# 5. Co-Retrieval Association Strengthening 测试
# ══════════════════════════════════════════════════════════════════════

def test_rir7_co_retrieval_strengthens_entity_edge(conn):
    """同次召回的两个 chunk，其 entity_edge confidence 提升 0.02。"""
    # 插入两个 chunk
    _insert_chunk(conn, "rir7_chunk_a", summary="Redis 缓存配置", project="test_proj")
    _insert_chunk(conn, "rir7_chunk_b", summary="Redis 连接池超时", project="test_proj")

    # 插入 entity_map 绑定
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO entity_map (entity_name, chunk_id, project, updated_at) "
        "VALUES (?, ?, ?, ?)",
        ("Redis", "rir7_chunk_a", "test_proj", now)
    )
    conn.execute(
        "INSERT INTO entity_map (entity_name, chunk_id, project, updated_at) "
        "VALUES (?, ?, ?, ?)",
        ("Redis_pool", "rir7_chunk_b", "test_proj", now)
    )
    # 插入 entity_edge
    insert_edge(conn, "Redis", "uses", "Redis_pool",
                project="test_proj", confidence=0.70)
    conn.commit()

    # 获取原始 confidence
    original_conf = conn.execute(
        "SELECT confidence FROM entity_edges "
        "WHERE from_entity='Redis' AND to_entity='Redis_pool'"
    ).fetchone()
    assert original_conf is not None
    orig_val = original_conf["confidence"]

    # 触发同次召回
    reconsolidate(conn, ["rir7_chunk_a", "rir7_chunk_b"],
                  query="Redis 缓存配置", project="test_proj")
    conn.commit()

    # confidence 应提升 0.02
    new_conf = conn.execute(
        "SELECT confidence FROM entity_edges "
        "WHERE from_entity='Redis' AND to_entity='Redis_pool'"
    ).fetchone()
    assert new_conf["confidence"] > orig_val, (
        f"RIR7: entity_edge confidence 应提升，{orig_val:.4f} → {new_conf['confidence']:.4f}"
    )
    assert new_conf["confidence"] <= 0.99, "RIR7: confidence 上限 0.99"


def test_rir8_single_chunk_no_co_retrieval(conn):
    """单 chunk 不触发 Co-Retrieval（至少 2 个 chunk 才触发边 confidence 提升）。"""
    _insert_chunk(conn, "rir8_solo", summary="Redis 配置", project="test_proj")
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO entity_map (entity_name, chunk_id, project, updated_at) "
        "VALUES (?, ?, ?, ?)",
        ("Redis", "rir8_solo", "test_proj", now)
    )
    # 插入一条边
    insert_edge(conn, "Redis", "related_to", "Cache",
                project="test_proj", confidence=0.60)
    conn.commit()

    orig = conn.execute(
        "SELECT confidence FROM entity_edges "
        "WHERE from_entity='Redis' AND to_entity='Cache'"
    ).fetchone()["confidence"]

    # 只有一个 chunk — 不触发 Co-Retrieval
    reconsolidate(conn, ["rir8_solo"], query="Redis", project="test_proj")
    conn.commit()

    new_conf = conn.execute(
        "SELECT confidence FROM entity_edges "
        "WHERE from_entity='Redis' AND to_entity='Cache'"
    ).fetchone()["confidence"]

    # 单 chunk 不触发 Co-Retrieval，confidence 不变（Co-Retrieval 只处理 recalled chunk 集合内部的边）
    # 注：Redis→Cache 的 Cache entity 不在本次召回 chunk 集合中，所以不触发
    assert new_conf == orig, f"RIR8: 单 chunk 不应触发 Co-Retrieval，expected {orig}, got {new_conf}"


# ══════════════════════════════════════════════════════════════════════
# 6. 边界条件
# ══════════════════════════════════════════════════════════════════════

def test_rir10_empty_input_safe(conn):
    """空输入安全返回 0。"""
    n1 = reconsolidate(conn, [], query="test")
    n2 = reconsolidate(conn, ["nonexistent"], query="")
    assert n1 == 0, "RIR10: 空 chunk_ids 应返回 0"
    assert n2 == 0, "RIR10: 空 query 应返回 0"
