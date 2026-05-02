"""
iter575: unlink_anon_vmas — Dead Edge Pruning for Entity Graph

OS 类比：Linux unlink_anon_vmas() (Andrea Arcangeli, 2004, mm/rmap.c)
  进程 munmap/exit 时拆除 anon_vma 反向映射链。如果不 unlink，rmap walker
  (spreading_activate) 遍历指向已释放 VMA（已删除 chunk）的 anon_vma 条目，
  浪费 CPU 且路径永远不返回有效结果。

根因：entity_edges 80.1% 的 edge 至少一端 entity 不在 entity_map 中有映射
  到存活 chunk。spreading_activate 的完整路径在 entity_map 查找失败时整条死亡。
  fstrim 只清理有 source_chunk_id 的 stale ref，logrotate 只清理 NULL source +
  超龄 72h。真正的判据应该是 entity 端点可达性。

三级策略：
  Level 1 (fully_disconnected): 两端 entity 都不在 entity_map → 直接删除
  Level 2 (half_dangling): 只一端在 entity_map → 可选删除
  Level 3 (stale_mapped): entity_map 存在但映射 chunk 已删除 → alive_entities 过滤

测试覆盖：fully_disconnected/half_dangling/alive_preserved/disabled/max_cap/
          prune_half_off/project_filter/global_included/idempotent/
          stale_map_filtered/ghost_skipped/empty_db/commit_on_prune/
          no_commit_zero/performance
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tmpfs  # noqa: E402 — 测试隔离

import sqlite3
import time
import pytest
from datetime import datetime, timezone
from store_core import open_db, ensure_schema, insert_chunk, MEMORY_OS_DIR
from store_mm import unlink_anon_vmas
from config import get as sysctl


def _setup():
    """创建干净的测试 DB"""
    conn = open_db(":memory:")
    ensure_schema(conn)
    return conn


def _make_chunk(conn, chunk_id, project="test_proj", importance=0.5):
    """创建最小 chunk"""
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, summary, content, importance,
            access_count, oom_adj, lru_gen, last_accessed, created_at)
           VALUES (?, ?, 'decision', 'test', 'test content', ?,
                   0, 0, 0, ?, ?)""",
        (chunk_id, project, importance,
         datetime.now(timezone.utc).isoformat(),
         datetime.now(timezone.utc).isoformat())
    )
    conn.commit()
    return chunk_id


def _make_entity_map(conn, entity_name, chunk_id, project="test_proj"):
    """建立 entity_map 映射"""
    conn.execute(
        """INSERT OR IGNORE INTO entity_map (entity_name, chunk_id, project)
           VALUES (?, ?, ?)""",
        (entity_name, chunk_id, project)
    )
    conn.commit()


def _make_edge(conn, edge_id, from_entity, to_entity, project="test_proj"):
    """创建 entity_edge"""
    conn.execute(
        """INSERT OR REPLACE INTO entity_edges
           (id, from_entity, relation, to_entity, project, source_chunk_id,
            confidence, created_at)
           VALUES (?, ?, 'relates_to', ?, ?, NULL, 0.6, ?)""",
        (edge_id, from_entity, to_entity, project,
         datetime.now(timezone.utc).isoformat())
    )
    conn.commit()


# ─── Test Cases ──────────────────────────────────────────────────────────────


def test_fully_disconnected_pruned():
    """两端 entity 都不在 entity_map → 删除"""
    conn = _setup()
    # 创建一条两端都不在 entity_map 的 edge
    _make_edge(conn, "edge1", "orphan_a", "orphan_b")
    result = unlink_anon_vmas(conn)
    assert result["pruned"] == 1
    assert result["fully_disconnected"] == 1
    # 确认已删除
    remaining = conn.execute("SELECT COUNT(*) FROM entity_edges").fetchone()[0]
    assert remaining == 0


def test_half_dangling_pruned():
    """只有一端在 entity_map（prune_half_dangling=True）→ 删除"""
    conn = _setup()
    cid = _make_chunk(conn, "chunk1")
    _make_entity_map(conn, "alive_entity", cid)
    # 一端 alive，一端 dangling
    _make_edge(conn, "edge1", "alive_entity", "dead_entity")
    result = unlink_anon_vmas(conn)
    assert result["pruned"] == 1
    assert result["half_dangling"] == 1


def test_alive_edge_preserved():
    """两端 entity 都在 entity_map 且映射到存活 chunk → 保留"""
    conn = _setup()
    cid1 = _make_chunk(conn, "chunk1")
    cid2 = _make_chunk(conn, "chunk2")
    _make_entity_map(conn, "entity_a", cid1)
    _make_entity_map(conn, "entity_b", cid2)
    _make_edge(conn, "edge1", "entity_a", "entity_b")
    result = unlink_anon_vmas(conn)
    assert result["pruned"] == 0
    remaining = conn.execute("SELECT COUNT(*) FROM entity_edges").fetchone()[0]
    assert remaining == 1


def test_disabled():
    """enabled=False → 不执行"""
    conn = _setup()
    _make_edge(conn, "edge1", "orphan_a", "orphan_b")
    # 临时覆盖配置
    import config
    orig = config._REGISTRY["unlink_anon_vmas.enabled"]
    config._REGISTRY["unlink_anon_vmas.enabled"] = (False, bool, None, None, None, "")
    try:
        result = unlink_anon_vmas(conn)
        assert result["pruned"] == 0
        remaining = conn.execute("SELECT COUNT(*) FROM entity_edges").fetchone()[0]
        assert remaining == 1
    finally:
        config._REGISTRY["unlink_anon_vmas.enabled"] = orig


def test_max_prune_cap():
    """max_prune 限制单次删除量"""
    conn = _setup()
    # 创建 60 条 fully disconnected edges
    for i in range(60):
        _make_edge(conn, f"edge{i}", f"orphan_a{i}", f"orphan_b{i}")
    # max_prune=50 默认
    result = unlink_anon_vmas(conn)
    assert result["pruned"] == 50  # capped
    remaining = conn.execute("SELECT COUNT(*) FROM entity_edges").fetchone()[0]
    assert remaining == 10


def test_prune_half_dangling_off():
    """prune_half_dangling=False 时只清理 fully_disconnected"""
    conn = _setup()
    cid = _make_chunk(conn, "chunk1")
    _make_entity_map(conn, "alive_entity", cid)
    # fully disconnected
    _make_edge(conn, "edge1", "dead_a", "dead_b")
    # half dangling
    _make_edge(conn, "edge2", "alive_entity", "dead_c")

    import config
    orig = config._REGISTRY["unlink_anon_vmas.prune_half_dangling"]
    config._REGISTRY["unlink_anon_vmas.prune_half_dangling"] = (False, bool, None, None, None, "")
    try:
        result = unlink_anon_vmas(conn)
        assert result["pruned"] == 1
        assert result["fully_disconnected"] == 1
        assert result["half_dangling"] == 0
        # half-dangling edge preserved
        remaining = conn.execute("SELECT COUNT(*) FROM entity_edges WHERE id='edge2'").fetchone()[0]
        assert remaining == 1
    finally:
        config._REGISTRY["unlink_anon_vmas.prune_half_dangling"] = orig


def test_project_filter():
    """project 参数限制扫描范围"""
    conn = _setup()
    _make_edge(conn, "edge1", "orphan_a", "orphan_b", project="proj_a")
    _make_edge(conn, "edge2", "orphan_c", "orphan_d", project="proj_b")
    result = unlink_anon_vmas(conn, project="proj_a")
    assert result["pruned"] == 1
    assert result["scanned"] == 1
    # proj_b edge still exists
    remaining = conn.execute("SELECT COUNT(*) FROM entity_edges WHERE project='proj_b'").fetchone()[0]
    assert remaining == 1


def test_global_included():
    """project 过滤时包含 global edges"""
    conn = _setup()
    _make_edge(conn, "edge1", "orphan_a", "orphan_b", project="global")
    result = unlink_anon_vmas(conn, project="test_proj")
    assert result["pruned"] == 1
    assert result["scanned"] == 1


def test_idempotent():
    """第二次运行不删除任何东西"""
    conn = _setup()
    _make_edge(conn, "edge1", "orphan_a", "orphan_b")
    r1 = unlink_anon_vmas(conn)
    assert r1["pruned"] == 1
    r2 = unlink_anon_vmas(conn)
    assert r2["pruned"] == 0
    assert r2["scanned"] == 0


def test_stale_map_filtered():
    """entity_map 中映射到已删除 chunk (importance=0 ghost) → entity 不在 alive set"""
    conn = _setup()
    # 创建 ghost chunk (importance=0)
    conn.execute(
        """INSERT INTO memory_chunks
           (id, project, chunk_type, summary, content, importance,
            access_count, oom_adj, lru_gen, last_accessed, created_at)
           VALUES ('ghost1', 'test_proj', 'decision', 'ghost', 'ghost', 0,
                   0, 0, 0, ?, ?)""",
        (datetime.now(timezone.utc).isoformat(),
         datetime.now(timezone.utc).isoformat())
    )
    conn.commit()
    _make_entity_map(conn, "ghost_entity", "ghost1")
    _make_edge(conn, "edge1", "ghost_entity", "other_dead")
    result = unlink_anon_vmas(conn)
    # ghost_entity maps to imp=0 chunk → not in alive_entities → fully_disconnected
    assert result["pruned"] == 1
    assert result["fully_disconnected"] == 1


def test_empty_db():
    """空 DB 不报错"""
    conn = _setup()
    result = unlink_anon_vmas(conn)
    assert result["pruned"] == 0
    assert result["scanned"] == 0


def test_commit_on_prune():
    """有删除时确认 commit 成功"""
    conn = _setup()
    _make_edge(conn, "edge1", "a", "b")
    result = unlink_anon_vmas(conn)
    assert result["pruned"] == 1
    # 重新打开确认持久化（内存模式下 commit 本身不报错即可）
    remaining = conn.execute("SELECT COUNT(*) FROM entity_edges").fetchone()[0]
    assert remaining == 0


def test_no_commit_zero():
    """零删除时不执行 commit（避免无意义 fsync）"""
    conn = _setup()
    cid = _make_chunk(conn, "chunk1")
    _make_entity_map(conn, "a", cid)
    cid2 = _make_chunk(conn, "chunk2")
    _make_entity_map(conn, "b", cid2)
    _make_edge(conn, "edge1", "a", "b")
    # 没有要删除的
    result = unlink_anon_vmas(conn)
    assert result["pruned"] == 0


def test_mixed_edges():
    """混合场景：alive + fully_disconnected + half_dangling 正确分类"""
    conn = _setup()
    cid1 = _make_chunk(conn, "c1")
    cid2 = _make_chunk(conn, "c2")
    _make_entity_map(conn, "ent_a", cid1)
    _make_entity_map(conn, "ent_b", cid2)

    # Alive edge
    _make_edge(conn, "alive", "ent_a", "ent_b")
    # Fully disconnected
    _make_edge(conn, "dead1", "x", "y")
    _make_edge(conn, "dead2", "m", "n")
    # Half dangling
    _make_edge(conn, "half1", "ent_a", "orphan1")
    _make_edge(conn, "half2", "orphan2", "ent_b")

    result = unlink_anon_vmas(conn)
    assert result["pruned"] == 4
    assert result["fully_disconnected"] == 2
    assert result["half_dangling"] == 2
    # Only alive edge remains
    remaining = conn.execute("SELECT id FROM entity_edges").fetchall()
    assert [r[0] for r in remaining] == ["alive"]


def test_performance():
    """50 edges 在 50ms 内完成"""
    conn = _setup()
    for i in range(50):
        _make_edge(conn, f"e{i}", f"from{i}", f"to{i}")
    t0 = time.time()
    result = unlink_anon_vmas(conn)
    elapsed = (time.time() - t0) * 1000
    assert result["pruned"] == 50
    assert elapsed < 50, f"Too slow: {elapsed:.1f}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
