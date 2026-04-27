"""
test_ghost_reaper.py — 迭代335：Ghost Reaper 单元测试

信息论背景：ghost chunk 携带 0 信息（importance=0，已合并），
  但占用 FTS5 result slot，产生 false recall（SNR 降低）。
OS 类比：Linux zombie process reaping — wait4() 回收 zombie，释放进程表项。

验证：
  1. 无 ghost → reap_ghosts() 返回 reaped=0
  2. importance=0 + merged summary → 识别为 ghost 并删除
  3. importance=0 但 summary 无合并标记 → 按 oom_adj>=500 判断
  4. importance > 0 → 不被删除（保护正常低分 chunk）
  5. dry_run=True → 统计但不删除
  6. project 过滤 → 只删除指定 project 的 ghost
  7. FTS5 软过滤：importance=0 chunk 不出现在 fts_search 结果中
  8. reap 后 fts_search 结果不含被删 ghost
"""
import sys
import sqlite3
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "hooks"))

import tmpfs  # noqa — 设置测试 DB 路径

from store_vfs import reap_ghosts, fts_search
from store import open_db, ensure_schema


def _make_chunk(conn, chunk_id, importance, summary, project="test",
                chunk_type="decision", oom_adj=0, access_count=0):
    """插入测试 chunk 到数据库。"""
    conn.execute(
        """INSERT INTO memory_chunks
           (id, summary, content, chunk_type, importance, last_accessed,
            access_count, created_at, project, oom_adj, info_class, lru_gen)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            chunk_id,
            summary,
            f"content of {chunk_id}",
            chunk_type,
            importance,
            "2026-04-01T00:00:00+00:00",
            access_count,
            "2026-04-01T00:00:00+00:00",
            project,
            oom_adj,
            "world",
            2,
        ),
    )


def _setup_db():
    """创建测试数据库并返回连接。"""
    conn = open_db()
    ensure_schema(conn)
    return conn


# ──────────────────────────────────────────────────────────────────────
# 1. 无 ghost → reaped=0
# ──────────────────────────────────────────────────────────────────────

def test_no_ghosts():
    """无 ghost chunk 时 reap_ghosts 返回 reaped=0。"""
    conn = _setup_db()
    _make_chunk(conn, "c_normal1", importance=0.80, summary="正常决策 chunk")
    _make_chunk(conn, "c_normal2", importance=0.60, summary="另一个决策")
    conn.commit()

    result = reap_ghosts(conn, project="test")
    assert result["reaped_count"] == 0, f"no ghosts, got {result['reaped_count']}"
    conn.close()


# ──────────────────────────────────────────────────────────────────────
# 2. importance=0 + merged summary → 删除
# ──────────────────────────────────────────────────────────────────────

def test_ghost_by_merged_summary():
    """importance=0 + summary=[merged→...] → 识别并删除。"""
    conn = _setup_db()
    survivor_id = str(uuid.uuid4())
    ghost_id = str(uuid.uuid4())
    _make_chunk(conn, survivor_id, importance=0.85, summary="幸存 chunk 决策")
    _make_chunk(conn, ghost_id, importance=0.0,
                summary=f"[merged→{survivor_id}] 被合并的旧 chunk", oom_adj=500)
    conn.commit()

    result = reap_ghosts(conn, project="test")
    assert result["reaped_count"] >= 1, f"should reap ghost, got {result}"
    assert ghost_id in result["ghost_ids"], f"ghost_id should be in ghost_ids"

    # 验证数据库中已删除
    remaining = conn.execute(
        "SELECT id FROM memory_chunks WHERE id=?", (ghost_id,)
    ).fetchone()
    assert remaining is None, f"ghost should be deleted from DB"
    conn.close()


# ──────────────────────────────────────────────────────────────────────
# 3. importance=0 + oom_adj>=500（无 merged 标记）→ 也视为 ghost
# ──────────────────────────────────────────────────────────────────────

def test_ghost_by_oom_adj():
    """importance=0 + oom_adj>=500 → 也被识别为 ghost（即使无 merged 前缀）。"""
    conn = _setup_db()
    ghost_id = str(uuid.uuid4())
    _make_chunk(conn, ghost_id, importance=0.0, summary="旧的内容", oom_adj=500)
    conn.commit()

    result = reap_ghosts(conn, project="test")
    assert ghost_id in result["ghost_ids"], f"should detect ghost by oom_adj: {result}"
    conn.close()


# ──────────────────────────────────────────────────────────────────────
# 4. importance > 0 → 不删除（正常低分 chunk 保护）
# ──────────────────────────────────────────────────────────────────────

def test_normal_low_imp_not_reaped():
    """importance=0.01（极低但非零）→ 不被当作 ghost 删除。"""
    conn = _setup_db()
    low_imp_id = str(uuid.uuid4())
    _make_chunk(conn, low_imp_id, importance=0.01, summary="低重要性但非零 chunk", oom_adj=0)
    conn.commit()

    result = reap_ghosts(conn, project="test")
    assert low_imp_id not in result["ghost_ids"], f"low imp (>0) should not be reaped"

    remaining = conn.execute(
        "SELECT id FROM memory_chunks WHERE id=?", (low_imp_id,)
    ).fetchone()
    assert remaining is not None, f"low imp (>0) chunk should still exist"
    conn.close()


# ──────────────────────────────────────────────────────────────────────
# 5. dry_run=True → 统计但不删除
# ──────────────────────────────────────────────────────────────────────

def test_dry_run():
    """dry_run=True → 返回待删数量但不实际删除。"""
    conn = _setup_db()
    ghost_id = str(uuid.uuid4())
    _make_chunk(conn, ghost_id, importance=0.0,
                summary="[merged→abc123] dry run ghost", oom_adj=500)
    conn.commit()

    result = reap_ghosts(conn, project="test", dry_run=True)
    assert result["dry_run"] is True
    assert result["reaped_count"] >= 1, f"dry_run should count ghost: {result}"
    assert ghost_id in result["ghost_ids"]

    # 验证未被删除
    remaining = conn.execute(
        "SELECT id FROM memory_chunks WHERE id=?", (ghost_id,)
    ).fetchone()
    assert remaining is not None, f"dry_run should NOT delete ghost"
    conn.close()


# ──────────────────────────────────────────────────────────────────────
# 6. project 过滤 → 只删指定 project
# ──────────────────────────────────────────────────────────────────────

def test_project_filter():
    """project='proj_a' → 只删 proj_a 的 ghost，不删 proj_b 的 ghost。"""
    conn = _setup_db()
    ghost_a = str(uuid.uuid4())
    ghost_b = str(uuid.uuid4())
    _make_chunk(conn, ghost_a, importance=0.0,
                summary="[merged→x] ghost in proj_a", oom_adj=500, project="proj_a")
    _make_chunk(conn, ghost_b, importance=0.0,
                summary="[merged→y] ghost in proj_b", oom_adj=500, project="proj_b")
    conn.commit()

    result = reap_ghosts(conn, project="proj_a")
    assert ghost_a in result["ghost_ids"], f"proj_a ghost should be reaped: {result}"
    assert ghost_b not in result["ghost_ids"], f"proj_b ghost should NOT be reaped"

    # proj_b ghost 仍存在
    remaining_b = conn.execute(
        "SELECT id FROM memory_chunks WHERE id=?", (ghost_b,)
    ).fetchone()
    assert remaining_b is not None, f"proj_b ghost should still exist"
    conn.close()


# ──────────────────────────────────────────────────────────────────────
# 7. FTS5 软过滤：importance=0 chunk 不出现在 fts_search 结果中
# ──────────────────────────────────────────────────────────────────────

def test_fts_ghost_filter():
    """FTS5 查询中 importance > 0.0 过滤 — ghost chunk 不出现在结果中。"""
    conn = _setup_db()
    ghost_id = str(uuid.uuid4())
    normal_id = str(uuid.uuid4())

    # ghost: importance=0，summary 包含搜索词 "优化 性能"
    _make_chunk(conn, ghost_id, importance=0.0,
                summary="优化性能的决策 [merged→abc] 废弃", oom_adj=500, project="fts_test")
    # normal: importance=0.8，summary 包含相同词
    _make_chunk(conn, normal_id, importance=0.8,
                summary="优化性能 决策记录", project="fts_test")
    conn.commit()

    results = fts_search(conn, "优化性能", "fts_test", top_k=10)
    result_ids = [r["id"] for r in results]

    assert ghost_id not in result_ids, \
        f"ghost (importance=0) should NOT appear in fts_search results: {result_ids}"
    # normal chunk 应该出现（如果 FTS5 能匹配）
    conn.close()


# ──────────────────────────────────────────────────────────────────────
# 8. reap 后 fts_search 不含被删 ghost
# ──────────────────────────────────────────────────────────────────────

def test_fts_after_reap():
    """reap_ghosts 删除后，fts_search 不再返回被删 ghost（FTS5 索引已清理）。"""
    conn = _setup_db()
    ghost_id = str(uuid.uuid4())
    _make_chunk(conn, ghost_id, importance=0.0,
                summary="[merged→xyz] 性能优化历史版本", oom_adj=500,
                project="reap_test")
    conn.commit()

    # reap 前验证 ghost 存在于 DB（importance filter 会让它在 fts 中不可见）
    before = conn.execute(
        "SELECT id FROM memory_chunks WHERE id=?", (ghost_id,)
    ).fetchone()
    assert before is not None

    # 执行 reap
    result = reap_ghosts(conn, project="reap_test")
    conn.commit()
    assert result["reaped_count"] >= 1

    # reap 后 ghost 不在 DB
    after = conn.execute(
        "SELECT id FROM memory_chunks WHERE id=?", (ghost_id,)
    ).fetchone()
    assert after is None, f"ghost should be deleted after reap"

    # fts_search 不含 ghost（物理删除 + FTS5 trigger 清理）
    fts_results = fts_search(conn, "性能优化历史", "reap_test", top_k=10)
    fts_ids = [r["id"] for r in fts_results]
    assert ghost_id not in fts_ids, f"ghost should not appear in fts after reap"
    conn.close()


# ──────────────────────────────────────────────────────────────────────
# 9. projects_stats 统计正确
# ──────────────────────────────────────────────────────────────────────

def test_projects_stats():
    """reap_ghosts 返回正确的 projects_stats 按 project 分组统计。"""
    conn = _setup_db()
    for i in range(3):
        _make_chunk(conn, f"g_proj1_{i}", importance=0.0,
                    summary=f"[merged→base] ghost {i}", oom_adj=500, project="stats_proj1")
    for i in range(2):
        _make_chunk(conn, f"g_proj2_{i}", importance=0.0,
                    summary=f"[merged→base] ghost {i}", oom_adj=500, project="stats_proj2")
    conn.commit()

    # 全局 reap（不限 project）
    result = reap_ghosts(conn, project=None)
    stats = result["projects_stats"]
    assert stats.get("stats_proj1", 0) >= 3, f"stats_proj1 should have 3 ghosts: {stats}"
    assert stats.get("stats_proj2", 0) >= 2, f"stats_proj2 should have 2 ghosts: {stats}"
    conn.close()
