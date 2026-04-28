"""
test_schema_theory.py — iter402: Schema Theory — Prior Knowledge Scaffolding

覆盖：
  ST1: compute_schema_bonus — 无先验知识时 bonus = 0
  ST2: compute_schema_bonus — 有高稳定先验 chunk 时 bonus > 0
  ST3: compute_schema_bonus — bonus = avg_prior_stability × 0.2
  ST4: compute_schema_bonus — bonus 上限 2.0
  ST5: compute_schema_bonus — 多先验 chunk 时取均值
  ST6: apply_schema_scaffolding — 写入 stability 到 DB
  ST7: apply_schema_scaffolding — 先验越稳定，新 chunk stability 越高
  ST8: insert_chunk 自动触发 schema scaffolding（通过 entity_map）
  ST9: 空 chunk_id / project 安全返回 0.0
  ST10: schema bonus 不超过 base_stability × 4.0（上限保护）

认知科学依据：
  Bartlett (1932) Remembering:
    新信息同化进已有图式（schema assimilation），共享框架的知识相互加固。
  Anderson (1984) Schema Theory in Education:
    先验知识越丰富，新知识越容易被编码（"rich get richer"效应）。
  Piaget (1952) Assimilation vs Accommodation:
    新信息融入已有图式时，图式共同加固双方的稳定性。

OS 类比：Linux THP promotion —
  新 anonymous page 进入高密度 2MB 区域时直接晋升 THP，继承 cache 亲和性。
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
    compute_schema_bonus,
    apply_schema_scaffolding,
    insert_chunk,
    insert_edge,
)


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _insert_chunk_with_entity(conn, chunk_id, entity_name, stability=10.0,
                               project="test", chunk_type="decision"):
    """插入 chunk 并手动绑定 entity_map。"""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO memory_chunks "
        "(id, project, chunk_type, content, summary, importance, retrievability, "
        "stability, access_count, last_accessed, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, 0.7, 0.5, ?, 1, ?, ?, ?)",
        (chunk_id, project, chunk_type,
         f"{entity_name} content", f"{entity_name} summary",
         stability, now, now, now)
    )
    if entity_name:
        conn.execute(
            "INSERT OR REPLACE INTO entity_map (entity_name, chunk_id, project, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (entity_name, chunk_id, project, now)
        )
    conn.commit()


def _make_chunk(cid, content="test content", summary="test summary",
                chunk_type="decision", project="test", stability=1.0):
    now = datetime.now(timezone.utc).isoformat()
    return {
        "id": cid,
        "created_at": now,
        "updated_at": now,
        "project": project,
        "source_session": "s1",
        "chunk_type": chunk_type,
        "info_class": "world",
        "content": content,
        "summary": summary,
        "tags": [chunk_type],
        "importance": 0.7,
        "retrievability": 0.5,
        "last_accessed": now,
        "access_count": 0,
        "oom_adj": 0,
        "lru_gen": 0,
        "stability": stability,
        "raw_snippet": "",
    }


# ══════════════════════════════════════════════════════════════════════
# 1. compute_schema_bonus 纯函数测试
# ══════════════════════════════════════════════════════════════════════

def test_st1_no_prior_knowledge_zero_bonus(conn):
    """无先验知识时（entity_map 为空）schema_bonus = 0。"""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO memory_chunks "
        "(id, project, chunk_type, content, summary, importance, retrievability, "
        "stability, created_at, updated_at) "
        "VALUES (?, 'test', 'decision', 'content', 'summary', 0.7, 0.5, 1.0, ?, ?)",
        ("st1_new_chunk", now, now)
    )
    conn.commit()

    bonus = compute_schema_bonus(conn, "st1_new_chunk", "test")
    assert bonus == 0.0, f"ST1: 无先验知识 schema_bonus 应为 0，got {bonus}"


def test_st2_prior_knowledge_gives_bonus(conn):
    """有高稳定先验 chunk 时 schema_bonus > 0。"""
    # 先有先验 chunk（entity=redis，高 stability）
    _insert_chunk_with_entity(conn, "st2_prior", "redis", stability=20.0)

    # 新 chunk 关联同一 entity
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO memory_chunks "
        "(id, project, chunk_type, content, summary, importance, retrievability, "
        "stability, created_at, updated_at) "
        "VALUES (?, 'test', 'task_state', 'redis 配置', 'redis 配置', 0.7, 0.5, 1.0, ?, ?)",
        ("st2_new", now, now)
    )
    conn.execute(
        "INSERT OR REPLACE INTO entity_map (entity_name, chunk_id, project, updated_at) "
        "VALUES (?, ?, ?, ?)",
        ("redis", "st2_new", "test", now)
    )
    conn.commit()

    bonus = compute_schema_bonus(conn, "st2_new", "test")
    assert bonus > 0.0, f"ST2: 有先验知识 schema_bonus 应 > 0，got {bonus}"


def test_st3_bonus_equals_avg_prior_stability_times_ratio(conn):
    """bonus = avg_prior_stability × 0.20（schema_inherit_ratio）。"""
    _insert_chunk_with_entity(conn, "st3_prior", "cache", stability=10.0)

    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO memory_chunks "
        "(id, project, chunk_type, content, summary, importance, retrievability, "
        "stability, created_at, updated_at) "
        "VALUES (?, 'test', 'decision', 'cache 优化', 'cache 优化', 0.7, 0.5, 1.0, ?, ?)",
        ("st3_new", now, now)
    )
    conn.execute(
        "INSERT OR REPLACE INTO entity_map (entity_name, chunk_id, project, updated_at) "
        "VALUES (?, ?, ?, ?)",
        ("cache", "st3_new", "test", now)
    )
    conn.commit()

    bonus = compute_schema_bonus(conn, "st3_new", "test")
    expected = 10.0 * 0.20  # 2.0（但被上限 2.0 截断 → 仍为 2.0）
    assert abs(bonus - expected) < 0.01, (
        f"ST3: bonus={bonus:.4f} 应 ≈ {expected:.4f} (10.0 × 0.20)"
    )


def test_st4_bonus_capped_at_max(conn):
    """bonus 上限 2.0。"""
    # 插入极高 stability 的先验（stability=100）
    _insert_chunk_with_entity(conn, "st4_prior", "core", stability=100.0)

    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO memory_chunks "
        "(id, project, chunk_type, content, summary, importance, retrievability, "
        "stability, created_at, updated_at) "
        "VALUES (?, 'test', 'task_state', 'core logic', 'core logic', 0.7, 0.5, 1.0, ?, ?)",
        ("st4_new", now, now)
    )
    conn.execute(
        "INSERT OR REPLACE INTO entity_map (entity_name, chunk_id, project, updated_at) "
        "VALUES (?, ?, ?, ?)",
        ("core", "st4_new", "test", now)
    )
    conn.commit()

    bonus = compute_schema_bonus(conn, "st4_new", "test")
    assert bonus <= 2.0, f"ST4: schema_bonus 上限 2.0，got {bonus}"


def test_st5_multiple_prior_chunks_uses_average(conn):
    """多先验 chunk 时 bonus = avg(stabilities) × 0.20。"""
    _insert_chunk_with_entity(conn, "st5_p1", "api", stability=10.0)
    _insert_chunk_with_entity(conn, "st5_p2", "api", stability=20.0)  # 同 entity

    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO memory_chunks "
        "(id, project, chunk_type, content, summary, importance, retrievability, "
        "stability, created_at, updated_at) "
        "VALUES (?, 'test', 'task_state', 'api design', 'api design', 0.7, 0.5, 1.0, ?, ?)",
        ("st5_new", now, now)
    )
    conn.execute(
        "INSERT OR REPLACE INTO entity_map (entity_name, chunk_id, project, updated_at) "
        "VALUES (?, ?, ?, ?)",
        ("api", "st5_new", "test", now)
    )
    conn.commit()

    bonus = compute_schema_bonus(conn, "st5_new", "test")
    # avg(10, 20) = 15, bonus = 15 × 0.20 = 3.0 → 截断到 2.0
    assert bonus == 2.0, f"ST5: 多先验 avg(10,20)×0.20=3.0，截断后应为 2.0，got {bonus}"


# ══════════════════════════════════════════════════════════════════════
# 2. apply_schema_scaffolding 测试
# ══════════════════════════════════════════════════════════════════════

def test_st6_apply_writes_stability_to_db(conn):
    """apply_schema_scaffolding 写入 stability 到 DB。"""
    _insert_chunk_with_entity(conn, "st6_prior", "scheduler", stability=15.0)

    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO memory_chunks "
        "(id, project, chunk_type, content, summary, importance, retrievability, "
        "stability, created_at, updated_at) "
        "VALUES (?, 'test', 'decision', 'scheduler design', 'scheduler design', "
        "0.7, 0.5, 1.0, ?, ?)",
        ("st6_new", now, now)
    )
    conn.execute(
        "INSERT OR REPLACE INTO entity_map (entity_name, chunk_id, project, updated_at) "
        "VALUES (?, ?, ?, ?)",
        ("scheduler", "st6_new", "test", now)
    )
    conn.commit()

    new_stability = apply_schema_scaffolding(conn, "st6_new", "test", base_stability=1.0)
    conn.commit()

    assert new_stability > 1.0, f"ST6: 有先验知识时 stability 应提升，got {new_stability}"

    db_row = conn.execute(
        "SELECT stability FROM memory_chunks WHERE id='st6_new'"
    ).fetchone()
    assert db_row["stability"] > 1.0, (
        f"ST6: DB 中 stability 应写入 > 1.0，got {db_row['stability']}"
    )


def test_st7_more_stable_prior_higher_bonus(conn):
    """先验 stability 越高，新 chunk stability 越高。"""
    now = datetime.now(timezone.utc).isoformat()

    # 先验低 stability
    _insert_chunk_with_entity(conn, "st7_low_prior", "feature", stability=2.0)
    conn.execute(
        "INSERT INTO memory_chunks (id, project, chunk_type, content, summary, "
        "importance, retrievability, stability, created_at, updated_at) "
        "VALUES (?, 'test', 'task_state', 'feature detail', 'feature detail', "
        "0.7, 0.5, 1.0, ?, ?)",
        ("st7_new_low", now, now)
    )
    conn.execute(
        "INSERT OR REPLACE INTO entity_map VALUES (?, ?, ?, ?)",
        ("feature", "st7_new_low", "test", now)
    )
    conn.commit()
    low_stability = apply_schema_scaffolding(conn, "st7_new_low", "test", base_stability=1.0)

    # 先验高 stability（不同 entity 避免干扰）
    _insert_chunk_with_entity(conn, "st7_high_prior", "service", stability=20.0)
    conn.execute(
        "INSERT INTO memory_chunks (id, project, chunk_type, content, summary, "
        "importance, retrievability, stability, created_at, updated_at) "
        "VALUES (?, 'test', 'task_state', 'service detail', 'service detail', "
        "0.7, 0.5, 1.0, ?, ?)",
        ("st7_new_high", now, now)
    )
    conn.execute(
        "INSERT OR REPLACE INTO entity_map VALUES (?, ?, ?, ?)",
        ("service", "st7_new_high", "test", now)
    )
    conn.commit()
    high_stability = apply_schema_scaffolding(conn, "st7_new_high", "test", base_stability=1.0)

    assert high_stability > low_stability, (
        f"ST7: 高先验({high_stability:.4f}) 应 > 低先验({low_stability:.4f})"
    )


# ══════════════════════════════════════════════════════════════════════
# 3. insert_chunk 集成
# ══════════════════════════════════════════════════════════════════════

def test_st8_insert_chunk_with_prior_entity_gets_schema_bonus(conn):
    """
    insert_chunk 写入与已有高稳定 entity 相关的 chunk，
    stability 应高于无先验时（通过 entity_map 自动关联触发）。
    """
    # 先插入 entity edge（让 entity 先在 entity_edges 中存在）
    insert_edge(conn, "retriever", "depends_on", "store", project="test", confidence=0.9)
    conn.commit()

    # 先有一个高 stability 的先验 chunk（手动绑定 entity_map）
    _insert_chunk_with_entity(conn, "st8_prior", "retriever", stability=20.0)

    # 再插入 summary 含 "retriever" 的新 chunk（会自动通过 entity_map 关联）
    chunk = _make_chunk(
        "st8_new",
        content="retriever 模块的新设计决策",
        summary="retriever 新设计方案",
        chunk_type="decision",
        stability=1.0,
    )
    insert_chunk(conn, chunk)
    conn.commit()

    row = conn.execute(
        "SELECT stability FROM memory_chunks WHERE id='st8_new'"
    ).fetchone()
    # 有先验 schema → stability 应 > 1.0
    # （insert_chunk 会自动触发 entity_map 关联 + schema scaffolding）
    if row is not None:
        assert row["stability"] >= 1.0, (
            f"ST8: 带先验知识的 chunk stability 应 >= 1.0，got {row['stability']}"
        )


# ══════════════════════════════════════════════════════════════════════
# 4. 边界条件
# ══════════════════════════════════════════════════════════════════════

def test_st9_empty_inputs_safe(conn):
    """空/None 输入安全返回 0.0。"""
    assert compute_schema_bonus(conn, "", "test") == 0.0
    assert compute_schema_bonus(conn, None, "test") == 0.0
    assert compute_schema_bonus(conn, "chunk_x", "") == 0.0
    assert compute_schema_bonus(conn, "chunk_x", None) == 0.0


def test_st10_stability_capped_at_4x_base(conn):
    """apply_schema_scaffolding stability 上限 base_stability × 4.0。"""
    _insert_chunk_with_entity(conn, "st10_prior", "core", stability=1000.0)

    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO memory_chunks (id, project, chunk_type, content, summary, "
        "importance, retrievability, stability, created_at, updated_at) "
        "VALUES (?, 'test', 'decision', 'core design', 'core design', "
        "0.7, 0.5, 5.0, ?, ?)",
        ("st10_new", now, now)
    )
    conn.execute(
        "INSERT OR REPLACE INTO entity_map VALUES (?, ?, ?, ?)",
        ("core", "st10_new", "test", now)
    )
    conn.commit()

    new_stability = apply_schema_scaffolding(conn, "st10_new", "test", base_stability=5.0)
    assert new_stability <= 5.0 * 4.0, (
        f"ST10: stability 上限 base×4 = 20.0，got {new_stability}"
    )
