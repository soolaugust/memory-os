"""
test_iter511_page_idle.py — page_idle 空闲页面追踪测试

OS 类比：Linux /sys/kernel/mm/page_idle/bitmap (Vladimir Davydov, 2015)
测试 page_idle_mark / page_idle_clear / page_idle_scan 三阶段机制。
"""
import sys
import os
import json
import tempfile

# tmpfs 隔离（迭代54）
_tmpdir = tempfile.mkdtemp(prefix="test_page_idle_")
os.environ["MEMORY_OS_DIR"] = _tmpdir
os.environ["MEMORY_OS_DB"] = os.path.join(_tmpdir, "test.db")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from store import open_db, ensure_schema, insert_chunk
from store_mm import (page_idle_mark, page_idle_clear, page_idle_scan,
                      _page_idle_load, _page_idle_save, _PAGE_IDLE_FILE,
                      MEMORY_OS_DIR)
from schema import MemoryChunk
import pytest


@pytest.fixture
def conn():
    """创建测试 DB 连接并初始化 schema，每次清空数据。"""
    c = open_db()
    ensure_schema(c)
    # 清空数据确保隔离
    c.execute("DELETE FROM memory_chunks")
    try:
        c.execute("DELETE FROM memory_chunks_fts")
    except Exception:
        pass
    c.commit()
    yield c
    c.close()


@pytest.fixture(autouse=True)
def clean_bitmap():
    """每个测试前清理 bitmap 文件。"""
    if _PAGE_IDLE_FILE.exists():
        _PAGE_IDLE_FILE.unlink()
    yield
    if _PAGE_IDLE_FILE.exists():
        _PAGE_IDLE_FILE.unlink()


def _insert_test_chunk(conn, chunk_id, project="test_proj", importance=0.7,
                       chunk_type="decision", oom_adj=0):
    """插入测试 chunk。"""
    chunk = MemoryChunk(
        id=chunk_id,
        project=project,
        chunk_type=chunk_type,
        summary=f"Test chunk {chunk_id}",
        content=f"Content for {chunk_id}",
        importance=importance,
    )
    insert_chunk(conn, chunk.to_dict())
    if oom_adj != 0:
        conn.execute("UPDATE memory_chunks SET oom_adj = ? WHERE id = ?",
                     (oom_adj, chunk_id))
    conn.commit()


# ── T1: mark 基本功能 — 所有 chunk 被标记为 idle ──

def test_mark_basic(conn):
    """首次 mark 应将所有非 task_state chunk 标记为 idle（rounds=1）。"""
    _insert_test_chunk(conn, "c1", "proj_a")
    _insert_test_chunk(conn, "c2", "proj_a")
    _insert_test_chunk(conn, "c3", "proj_a", chunk_type="task_state")  # 不参与

    result = page_idle_mark(conn, "proj_a")
    assert result["marked"] == 2  # c1, c2（task_state 排除）
    assert result["carried_over"] == 0  # 首次标记无延续

    bitmap = _page_idle_load()
    assert bitmap["proj_a"]["c1"] == 1
    assert bitmap["proj_a"]["c2"] == 1
    assert "c3" not in bitmap["proj_a"]


# ── T2: mark 连续轮次 — idle_rounds 递增 ──

def test_mark_increments_rounds(conn):
    """连续 mark 时 idle_rounds 应递增。"""
    _insert_test_chunk(conn, "c1", "proj_a")

    page_idle_mark(conn, "proj_a")
    page_idle_mark(conn, "proj_a")
    page_idle_mark(conn, "proj_a")

    bitmap = _page_idle_load()
    assert bitmap["proj_a"]["c1"] == 3


# ── T3: clear 基本功能 — 被访问的 chunk 从 bitmap 移除 ──

def test_clear_removes_accessed(conn):
    """clear 应将被命中的 chunk 从 idle bitmap 移除。"""
    _insert_test_chunk(conn, "c1", "proj_a")
    _insert_test_chunk(conn, "c2", "proj_a")

    page_idle_mark(conn, "proj_a")
    cleared = page_idle_clear(["c1"], "proj_a")

    assert cleared == 1
    bitmap = _page_idle_load()
    assert "c1" not in bitmap["proj_a"]
    assert bitmap["proj_a"]["c2"] == 1


# ── T4: clear 后重新 mark — rounds 重置为 1 ──

def test_clear_resets_rounds(conn):
    """被 clear 后下次 mark 应重新从 1 开始。"""
    _insert_test_chunk(conn, "c1", "proj_a")

    page_idle_mark(conn, "proj_a")  # rounds=1
    page_idle_mark(conn, "proj_a")  # rounds=2
    page_idle_clear(["c1"], "proj_a")  # 移除
    page_idle_mark(conn, "proj_a")  # 重新标记，rounds=1

    bitmap = _page_idle_load()
    assert bitmap["proj_a"]["c1"] == 1


# ── T5: scan 不满足阈值 — 不执行降级 ──

def test_scan_below_threshold(conn):
    """idle_rounds < demote_rounds 时不应执行降级。"""
    _insert_test_chunk(conn, "c1", "proj_a", importance=0.8)

    page_idle_mark(conn, "proj_a")  # rounds=1
    page_idle_mark(conn, "proj_a")  # rounds=2

    result = page_idle_scan(conn, "proj_a")
    assert result["demoted"] == 0
    assert result["deleted"] == 0

    # importance 不变
    row = conn.execute("SELECT importance FROM memory_chunks WHERE id='c1'").fetchone()
    assert row[0] == 0.8


# ── T6: scan 达到 demote_rounds — 执行降级 ──

def test_scan_demotes_at_threshold(conn):
    """idle_rounds >= demote_rounds 时应降级 importance。"""
    _insert_test_chunk(conn, "c1", "proj_a", importance=0.8)

    # 模拟连续 3 轮 idle（demote_rounds 默认 3）
    page_idle_mark(conn, "proj_a")
    page_idle_mark(conn, "proj_a")
    page_idle_mark(conn, "proj_a")

    result = page_idle_scan(conn, "proj_a")
    assert result["demoted"] == 1
    assert result["deleted"] == 0

    # importance 应降级：0.8 * 0.7 = 0.56
    row = conn.execute("SELECT importance, oom_adj FROM memory_chunks WHERE id='c1'").fetchone()
    assert abs(row[0] - 0.56) < 0.01
    assert row[1] == 200  # oom_adj += 200


# ── T7: scan 达到 delete_rounds + importance 很低 — 删除 ──

def test_scan_deletes_at_delete_threshold(conn):
    """idle_rounds >= delete_rounds 且降级后 importance < 0.2 时应删除。"""
    _insert_test_chunk(conn, "c1", "proj_a", importance=0.2)  # 0.2*0.7=0.14 < 0.2

    # 模拟连续 5 轮 idle（delete_rounds 默认 5）
    for _ in range(5):
        page_idle_mark(conn, "proj_a")

    result = page_idle_scan(conn, "proj_a")
    assert result["deleted"] == 1

    # chunk 应被删除
    row = conn.execute("SELECT COUNT(*) FROM memory_chunks WHERE id='c1'").fetchone()
    assert row[0] == 0


# ── T8: scan 保护 mlock chunk — oom_adj <= -500 不处理 ──

def test_scan_protects_mlock(conn):
    """oom_adj <= -500 的 chunk 不应被降级。"""
    _insert_test_chunk(conn, "c1", "proj_a", importance=0.5, oom_adj=-800)

    for _ in range(5):
        page_idle_mark(conn, "proj_a")

    result = page_idle_scan(conn, "proj_a")
    assert result["demoted"] == 0
    assert result["deleted"] == 0

    # importance 不变
    row = conn.execute("SELECT importance FROM memory_chunks WHERE id='c1'").fetchone()
    assert row[0] == 0.5


# ── T9: scan 保护 design_constraint — 只降级不删除 ──

def test_scan_protects_design_constraint(conn):
    """design_constraint 类型不删除，即使 importance 很低。"""
    _insert_test_chunk(conn, "c1", "proj_a", importance=0.15,
                       chunk_type="design_constraint")

    for _ in range(5):
        page_idle_mark(conn, "proj_a")

    result = page_idle_scan(conn, "proj_a")
    assert result["demoted"] == 1  # 降级
    assert result["deleted"] == 0  # 不删除

    # chunk 仍存在
    row = conn.execute("SELECT COUNT(*) FROM memory_chunks WHERE id='c1'").fetchone()
    assert row[0] == 1


# ── T10: 多项目隔离 — 不同项目的 bitmap 互不干扰 ──

def test_project_isolation(conn):
    """不同项目的 idle tracking 应完全隔离。"""
    _insert_test_chunk(conn, "c1", "proj_a")
    _insert_test_chunk(conn, "c2", "proj_b")

    page_idle_mark(conn, "proj_a")
    page_idle_mark(conn, "proj_b")

    # 清除 proj_a 的 c1
    page_idle_clear(["c1"], "proj_a")

    bitmap = _page_idle_load()
    assert "c1" not in bitmap.get("proj_a", {})
    assert bitmap["proj_b"]["c2"] == 1  # proj_b 不受影响


# ── T11: 空 DB 时 mark — 不崩溃 ──

def test_mark_empty_db(conn):
    """空 DB 时 mark 应正常返回 0。"""
    result = page_idle_mark(conn, "nonexistent_proj")
    assert result["marked"] == 0
    assert result["carried_over"] == 0


# ── T12: clear 空 bitmap — 不崩溃 ──

def test_clear_empty_bitmap(conn):
    """无 bitmap 文件时 clear 应返回 0。"""
    result = page_idle_clear(["c1"], "proj_a")
    assert result == 0


# ── T13: scan 空 bitmap — 不崩溃 ──

def test_scan_empty_bitmap(conn):
    """无 bitmap 文件时 scan 应返回空结果。"""
    result = page_idle_scan(conn, "proj_a")
    assert result["scanned"] == 0
    assert result["demoted"] == 0


# ── T14: 端到端流程 — mark → 部分 clear → mark → scan ──

def test_end_to_end(conn):
    """完整的 idle tracking 生命周期。"""
    _insert_test_chunk(conn, "active", "proj_a", importance=0.8)
    _insert_test_chunk(conn, "idle1", "proj_a", importance=0.6)
    _insert_test_chunk(conn, "idle2", "proj_a", importance=0.4)

    # 会话 1：全部标记 idle
    page_idle_mark(conn, "proj_a")

    # 会话 1 期间：active 被检索命中
    page_idle_clear(["active"], "proj_a")

    # 会话 2：mark（active 重新标记为 1，idle1/idle2 递增为 2）
    page_idle_mark(conn, "proj_a")

    # 会话 2 期间：active 再次被命中
    page_idle_clear(["active"], "proj_a")

    # 会话 3：mark（active → 1，idle1/idle2 → 3）
    page_idle_mark(conn, "proj_a")

    # scan：idle1/idle2 达到 demote_rounds=3，应被降级
    result = page_idle_scan(conn, "proj_a")
    assert result["demoted"] == 2  # idle1 和 idle2
    assert result["deleted"] == 0

    # 验证 active 不被降级
    row = conn.execute("SELECT importance FROM memory_chunks WHERE id='active'").fetchone()
    assert row[0] == 0.8  # 未变

    # idle1: 0.6 * 0.7 = 0.42
    row = conn.execute("SELECT importance FROM memory_chunks WHERE id='idle1'").fetchone()
    assert abs(row[0] - 0.42) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
