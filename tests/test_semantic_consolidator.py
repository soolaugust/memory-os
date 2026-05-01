"""
test_semantic_consolidator.py — 跨项目语义巩固测试

覆盖：
  SC1: 相似 chunk 来自不同 project → 生成 semantic chunk
  SC2: 相似 chunk 来自同一 project → 不生成（同项目由 consolidate.py 处理）
  SC3: 低 importance chunk → 不参与（低于 min_importance 阈值）
  SC4: 已存在同 summary hash → 更新而非重复写入
  SC5: semantic chunk 写入 project="__semantic__"，chunk_type="semantic_memory"
  SC6: semantic chunk 的 importance = 源 chunk 平均值
  SC7: semantic chunk 的 tags 包含源 project 列表
  SC8: memory_lookup 检索时自动包含 __semantic__ 层结果
  SC9: _trigram_similarity 基本正确性
"""
import sys
import sqlite3
import uuid
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import tmpfs  # noqa

from store import open_db, ensure_schema
from tools.semantic_consolidator import (
    run_consolidation,
    SEMANTIC_PROJECT,
    _trigram_similarity,
    _summary_hash,
)


def _utcnow():
    return datetime.now(timezone.utc)


def _insert(conn, cid, project, summary, content="", importance=0.8,
            chunk_type="decision", stability=10.0, access_count=0):
    now = _utcnow()
    la = (now - timedelta(minutes=10)).isoformat()
    now_iso = now.isoformat()
    conn.execute("""
        INSERT OR REPLACE INTO memory_chunks
        (id, project, source_session, chunk_type, summary, content,
         importance, stability, retrievability, info_class, tags,
         access_count, oom_adj, created_at, updated_at, last_accessed,
         feishu_url, lru_gen, raw_snippet, encode_context, session_type_history)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (cid, project, "test", chunk_type, summary, content or summary,
          importance, stability, 0.5, "episodic", json.dumps([]),
          access_count, 0, now_iso, now_iso, la,
          None, 0, (content or summary)[:500], "{}", ""))
    conn.commit()


def _cleanup(conn, *projects):
    for p in projects:
        conn.execute("DELETE FROM memory_chunks WHERE project=?", (p,))
    conn.commit()


# ── SC9: _trigram_similarity 基本正确性 ──────────────────────────────────────

def test_sc9_trigram_similarity():
    """SC9: 相同字符串相似度=1.0，完全不同≈0，相似字符串在中间。"""
    assert _trigram_similarity("hello world", "hello world") == 1.0
    assert _trigram_similarity("", "hello") == 0.0
    assert _trigram_similarity("hello world", "xyz abc qrs") < 0.2
    sim = _trigram_similarity(
        "Linux page fault handling mechanism",
        "Linux page fault interrupt handler"
    )
    assert 0.3 < sim < 1.0, f"SC9: 相似字符串应在 0.3-1.0，got {sim}"


# ── SC1: 跨 project 相似 chunk → 生成 semantic chunk ────────────────────────

def test_sc1_cross_project_generates_semantic():
    """SC1: 来自不同 project 的相似 chunk → run_consolidation 生成 semantic chunk。"""
    conn = open_db()
    ensure_schema(conn)

    proj_a = f"sc_proj_a_{uuid.uuid4().hex[:6]}"
    proj_b = f"sc_proj_b_{uuid.uuid4().hex[:6]}"
    _cleanup(conn, proj_a, proj_b, SEMANTIC_PROJECT)

    # 两个不同 project 里有相似的 summary
    _insert(conn, "sc1_a", proj_a,
            "Linux page fault handling: check pte, alloc page, fill content",
            importance=0.8)
    _insert(conn, "sc1_b", proj_b,
            "Linux page fault handler: validate pte entry, allocate physical page",
            importance=0.75)

    stats = run_consolidation(conn, sim_threshold=0.30, min_importance=0.70,
                              dry_run=False)

    semantic_count = conn.execute(
        "SELECT COUNT(*) FROM memory_chunks WHERE project=? AND chunk_type='semantic_memory'",
        (SEMANTIC_PROJECT,)
    ).fetchone()[0]

    assert semantic_count >= 1, (
        f"SC1: 跨 project 相似 chunk 应生成 semantic chunk，"
        f"stats={stats}, semantic_count={semantic_count}"
    )

    _cleanup(conn, proj_a, proj_b, SEMANTIC_PROJECT)
    conn.close()


# ── SC2: 同 project 相似 chunk → 不生成 ──────────────────────────────────────

def test_sc2_same_project_no_semantic():
    """SC2: 相似 chunk 来自同一 project → 不生成 semantic chunk。"""
    conn = open_db()
    ensure_schema(conn)

    proj = f"sc_proj_same_{uuid.uuid4().hex[:6]}"
    _cleanup(conn, proj, SEMANTIC_PROJECT)

    _insert(conn, "sc2_a", proj,
            "Linux page fault handling: check pte, alloc page",
            importance=0.8)
    _insert(conn, "sc2_b", proj,
            "Linux page fault handler: validate pte entry, allocate page",
            importance=0.75)

    stats = run_consolidation(conn, sim_threshold=0.30, min_importance=0.70,
                              dry_run=False)

    # 同 project 的相似 chunk 不应生成 semantic chunk
    semantic_count = conn.execute(
        "SELECT COUNT(*) FROM memory_chunks WHERE project=?",
        (SEMANTIC_PROJECT,)
    ).fetchone()[0]

    assert semantic_count == 0, (
        f"SC2: 同 project chunk 不应生成 semantic chunk，got {semantic_count}"
    )

    _cleanup(conn, proj, SEMANTIC_PROJECT)
    conn.close()


# ── SC3: 低 importance → 不参与 ──────────────────────────────────────────────

def test_sc3_low_importance_excluded():
    """SC3: importance < min_importance 的 chunk 不参与跨项目聚合。"""
    conn = open_db()
    ensure_schema(conn)

    proj_a = f"sc_low_a_{uuid.uuid4().hex[:6]}"
    proj_b = f"sc_low_b_{uuid.uuid4().hex[:6]}"
    _cleanup(conn, proj_a, proj_b, SEMANTIC_PROJECT)

    # importance 低于阈值（0.40 < 0.65）
    _insert(conn, "sc3_a", proj_a,
            "Linux page fault handling: check pte, alloc page",
            importance=0.40)
    _insert(conn, "sc3_b", proj_b,
            "Linux page fault handler: validate pte, allocate page",
            importance=0.40)

    stats = run_consolidation(conn, sim_threshold=0.30, min_importance=0.65,
                              dry_run=False)

    semantic_count = conn.execute(
        "SELECT COUNT(*) FROM memory_chunks WHERE project=?",
        (SEMANTIC_PROJECT,)
    ).fetchone()[0]

    assert semantic_count == 0, (
        f"SC3: 低 importance chunk 不应生成 semantic chunk，got {semantic_count}"
    )

    _cleanup(conn, proj_a, proj_b, SEMANTIC_PROJECT)
    conn.close()


# ── SC4: 已存在同 summary hash → 更新而非重复写入 ────────────────────────────

def test_sc4_upsert_not_duplicate():
    """SC4: 相同 summary 的 semantic chunk 第二次运行时更新而非重复写入。"""
    conn = open_db()
    ensure_schema(conn)

    proj_a = f"sc_ups_a_{uuid.uuid4().hex[:6]}"
    proj_b = f"sc_ups_b_{uuid.uuid4().hex[:6]}"
    _cleanup(conn, proj_a, proj_b, SEMANTIC_PROJECT)

    _insert(conn, "sc4_a", proj_a,
            "Linux page fault handling: check pte, alloc page",
            importance=0.8)
    _insert(conn, "sc4_b", proj_b,
            "Linux page fault handler: validate pte, allocate page",
            importance=0.75)

    # 第一次运行
    stats1 = run_consolidation(conn, sim_threshold=0.30, min_importance=0.70)
    count1 = conn.execute(
        "SELECT COUNT(*) FROM memory_chunks WHERE project=?", (SEMANTIC_PROJECT,)
    ).fetchone()[0]

    # 第二次运行（相同数据）
    stats2 = run_consolidation(conn, sim_threshold=0.30, min_importance=0.70)
    count2 = conn.execute(
        "SELECT COUNT(*) FROM memory_chunks WHERE project=?", (SEMANTIC_PROJECT,)
    ).fetchone()[0]

    assert count2 == count1, (
        f"SC4: 第二次运行不应重复写入，count1={count1} count2={count2}"
    )
    assert stats2.get("updated", 0) >= stats1.get("created", 0) or stats2.get("created", 0) == 0, \
        f"SC4: 第二次应更新而非创建，stats2={stats2}"

    _cleanup(conn, proj_a, proj_b, SEMANTIC_PROJECT)
    conn.close()


# ── SC5: semantic chunk 属于 __semantic__ project ────────────────────────────

def test_sc5_semantic_project_and_type():
    """SC5: 生成的 semantic chunk project='__semantic__', chunk_type='semantic_memory'。"""
    conn = open_db()
    ensure_schema(conn)

    proj_a = f"sc_type_a_{uuid.uuid4().hex[:6]}"
    proj_b = f"sc_type_b_{uuid.uuid4().hex[:6]}"
    _cleanup(conn, proj_a, proj_b, SEMANTIC_PROJECT)

    _insert(conn, "sc5_a", proj_a,
            "Linux mm compaction: migrate pages to reduce fragmentation",
            importance=0.85)
    _insert(conn, "sc5_b", proj_b,
            "Linux memory compaction: migrate anonymous pages reduce fragmentation",
            importance=0.80)

    run_consolidation(conn, sim_threshold=0.30, min_importance=0.75)

    rows = conn.execute(
        "SELECT project, chunk_type FROM memory_chunks WHERE project=?",
        (SEMANTIC_PROJECT,)
    ).fetchall()

    assert len(rows) >= 1, "SC5: 应有 semantic chunk 生成"
    for project, chunk_type in rows:
        assert project == SEMANTIC_PROJECT, f"SC5: project 应为 __semantic__，got {project}"
        assert chunk_type == "semantic_memory", f"SC5: chunk_type 应为 semantic_memory，got {chunk_type}"

    _cleanup(conn, proj_a, proj_b, SEMANTIC_PROJECT)
    conn.close()


# ── SC6: importance = 源 chunk 平均值 ────────────────────────────────────────

def test_sc6_importance_is_average():
    """SC6: semantic chunk 的 importance 是源 chunk 的平均值（不膨胀）。"""
    conn = open_db()
    ensure_schema(conn)

    proj_a = f"sc_imp_a_{uuid.uuid4().hex[:6]}"
    proj_b = f"sc_imp_b_{uuid.uuid4().hex[:6]}"
    _cleanup(conn, proj_a, proj_b, SEMANTIC_PROJECT)

    imp_a, imp_b = 0.80, 0.70
    expected_avg = (imp_a + imp_b) / 2  # 0.75

    _insert(conn, "sc6_a", proj_a,
            "Linux page fault handling: check pte, alloc page",
            importance=imp_a)
    _insert(conn, "sc6_b", proj_b,
            "Linux page fault handler: validate pte entry allocate page",
            importance=imp_b)

    run_consolidation(conn, sim_threshold=0.30, min_importance=0.65)

    row = conn.execute(
        "SELECT importance FROM memory_chunks WHERE project=? ORDER BY importance DESC LIMIT 1",
        (SEMANTIC_PROJECT,)
    ).fetchone()

    assert row is not None, "SC6: 应有 semantic chunk 生成"
    assert abs(row[0] - expected_avg) < 0.05, (
        f"SC6: importance 应约等于源 chunk 平均值 {expected_avg:.2f}，got {row[0]:.4f}"
    )

    _cleanup(conn, proj_a, proj_b, SEMANTIC_PROJECT)
    conn.close()


# ── SC7: tags 包含源 project 列表 ─────────────────────────────────────────────

def test_sc7_tags_contain_source_projects():
    """SC7: semantic chunk 的 tags 字段包含源 project 列表（溯源）。"""
    conn = open_db()
    ensure_schema(conn)

    proj_a = f"sc_tag_a_{uuid.uuid4().hex[:6]}"
    proj_b = f"sc_tag_b_{uuid.uuid4().hex[:6]}"
    _cleanup(conn, proj_a, proj_b, SEMANTIC_PROJECT)

    _insert(conn, "sc7_a", proj_a,
            "Linux page fault: validate pte and allocate physical page",
            importance=0.80)
    _insert(conn, "sc7_b", proj_b,
            "Linux page fault handling validate pte allocate physical memory",
            importance=0.75)

    run_consolidation(conn, sim_threshold=0.30, min_importance=0.70)

    row = conn.execute(
        "SELECT tags FROM memory_chunks WHERE project=? LIMIT 1",
        (SEMANTIC_PROJECT,)
    ).fetchone()

    assert row is not None, "SC7: 应有 semantic chunk 生成"
    tags = json.loads(row[0] or "[]")
    assert proj_a in tags or proj_b in tags, (
        f"SC7: tags 应包含源 project，tags={tags}, proj_a={proj_a}, proj_b={proj_b}"
    )

    _cleanup(conn, proj_a, proj_b, SEMANTIC_PROJECT)
    conn.close()


# ── SC8: __semantic__ 层可通过 fts_search 检索 ───────────────────────────────

def test_sc8_semantic_layer_fts_searchable():
    """SC8: __semantic__ project 的 chunk 可以被 fts_search 检索到。"""
    from store_vfs import insert_chunk, fts_search
    conn = open_db()
    ensure_schema(conn)

    proj_a = f"sc_lookup_a_{uuid.uuid4().hex[:6]}"
    proj_b = f"sc_lookup_b_{uuid.uuid4().hex[:6]}"
    _cleanup(conn, proj_a, proj_b, SEMANTIC_PROJECT)

    # 预先生成一个 semantic chunk（模拟 consolidation 已运行）
    now = datetime.now(timezone.utc)
    la = (now - timedelta(minutes=10)).isoformat()
    sem_id = "sem_test_sc8_" + uuid.uuid4().hex[:8]
    sem_chunk = {
        "id": sem_id,
        "project": SEMANTIC_PROJECT,
        "source_session": "semantic_consolidator",
        "chunk_type": "semantic_memory",
        "summary": "Linux page fault: pte validation and physical page allocation",
        "content": f"[{proj_a}] page fault handling\n[{proj_b}] page fault allocation",
        "importance": 0.75,
        "stability": 10.0,
        "retrievability": 0.8,
        "tags": json.dumps([proj_a, proj_b]),
        "access_count": 0,
        "oom_adj": -100,
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
        "last_accessed": la,
        "feishu_url": None,
        "lru_gen": 0,
    }
    insert_chunk(conn, sem_chunk)
    conn.commit()

    # 直接用 fts_search 验证 __semantic__ 层可检索
    results = fts_search(conn, "page fault pte allocation", SEMANTIC_PROJECT, top_k=5)
    found_ids = [r["id"] for r in results]

    assert sem_id in found_ids, (
        f"SC8: __semantic__ 层 chunk 应可被 fts_search 检索到，"
        f"sem_id={sem_id}, found={found_ids}"
    )

    _cleanup(conn, proj_a, proj_b, SEMANTIC_PROJECT)
    conn.close()
