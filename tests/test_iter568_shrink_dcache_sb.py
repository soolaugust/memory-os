"""
iter568: shrink_dcache_sb — Immediate Fragment Reclaim (No Age Gate)

OS 类比：Linux shrink_dcache_sb() (Al Viro, 2003, fs/dcache.c)
  当超级块 unmount/remount 时，立即释放该 sb 下所有不活跃 dentry，
  不需要等 LRU aging 周期。

测试：对 _vfs_write_protect()==True 且 access_count==0 的 chunk 立即删除，
无 age 门控（与 R4 的 1 天冷启动保护区别）。
"""
import os
import sys
import sqlite3
from datetime import datetime, timezone, timedelta

# ── tmpfs 测试隔离 ──
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tmpfs  # noqa: F401

from store_mm import shrink_dcache_sb
from store_vfs import open_db, ensure_schema


def _make_chunk(conn, summary, chunk_type="decision", importance=0.80,
                access_count=0, oom_adj=0, project="test_proj",
                age_days=0.0):
    """Helper: 创建测试 chunk。"""
    import uuid
    cid = str(uuid.uuid4())
    created = (datetime.now(timezone.utc) - timedelta(days=age_days)).isoformat()
    conn.execute(
        """INSERT INTO memory_chunks
           (id, summary, content, chunk_type, importance, access_count,
            oom_adj, project, source_session, created_at, last_accessed)
           VALUES (?, ?, '', ?, ?, ?, ?, ?, 'test_session', ?, ?)""",
        (cid, summary, chunk_type, importance, access_count,
         oom_adj, project, created, created)
    )
    conn.commit()
    return cid


def _exists(conn, cid):
    return conn.execute("SELECT 1 FROM memory_chunks WHERE id=?", (cid,)).fetchone() is not None


# ── 核心功能测试 ──

def test_pipe_fragment_deleted():
    """表格行碎片（| xxx | yyy |）应被立即删除，无 age 门控。"""
    conn = open_db()
    ensure_schema(conn)
    cid = _make_chunk(conn, "| 根因 | 6 个碎片 | 说明 |", age_days=0.0)
    assert _exists(conn, cid)
    result = shrink_dcache_sb(conn, "test_proj")
    assert result["deleted"] >= 1
    assert cid in result["deleted_ids"]
    assert not _exists(conn, cid)
    conn.close()


def test_pipe_start_fragment_deleted():
    """以 | 开头的碎片应被删除。"""
    conn = open_db()
    ensure_schema(conn)
    cid = _make_chunk(conn, "| 生产效果 | 垄断 chunk score 1.009 → 0.031", age_days=0.01)
    result = shrink_dcache_sb(conn, "test_proj")
    assert cid in result["deleted_ids"]
    assert not _exists(conn, cid)
    conn.close()


def test_bracket_start_fragment_deleted():
    """以 ) 或 ] 开头的碎片应被删除。"""
    conn = open_db()
    ensure_schema(conn)
    cid1 = _make_chunk(conn, ") 后续执行步骤的残留碎片文本片段", age_days=0.0)
    cid2 = _make_chunk(conn, "] 某个列表项的结尾残留碎片文本内容", age_days=0.0)
    result = shrink_dcache_sb(conn, "test_proj")
    assert cid1 in result["deleted_ids"]
    assert cid2 in result["deleted_ids"]
    conn.close()


def test_short_summary_deleted():
    """极短 summary（< 8 字符）应被删除。"""
    conn = open_db()
    ensure_schema(conn)
    cid = _make_chunk(conn, "abc", age_days=5.0)
    result = shrink_dcache_sb(conn, "test_proj")
    assert cid in result["deleted_ids"]
    conn.close()


def test_valid_chunk_preserved():
    """合法 chunk（不被 _vfs_write_protect 拒绝）应保留。"""
    conn = open_db()
    ensure_schema(conn)
    cid = _make_chunk(conn, "选择 BM25 替代向量检索因为 chunk 数量不足以训练 embedding", age_days=0.0)
    result = shrink_dcache_sb(conn, "test_proj")
    assert cid not in result["deleted_ids"]
    assert _exists(conn, cid)
    conn.close()


def test_accessed_chunk_preserved():
    """有访问记录的 chunk 即使 summary 是碎片也不删除。"""
    conn = open_db()
    ensure_schema(conn)
    cid = _make_chunk(conn, "| 表格行碎片 | 但有访问 |", access_count=3, age_days=0.0)
    result = shrink_dcache_sb(conn, "test_proj")
    assert cid not in result["deleted_ids"]
    assert _exists(conn, cid)
    conn.close()


def test_pinned_chunk_preserved():
    """被 mlock (chunk_pins) 的 chunk 不删除。"""
    conn = open_db()
    ensure_schema(conn)
    cid = _make_chunk(conn, "| 被 pin 的碎片 | 不删除 |", age_days=0.0)
    # 添加 pin (需要 pinned_at 字段)
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO chunk_pins (chunk_id, pin_type, project, pinned_at) VALUES (?, 'hard', 'test_proj', ?)",
        (cid, now)
    )
    conn.commit()
    result = shrink_dcache_sb(conn, "test_proj")
    assert cid not in result["deleted_ids"]
    assert _exists(conn, cid)
    conn.close()


def test_protected_oom_preserved():
    """oom_adj <= -500 的 chunk 不删除（用户显式保护）。"""
    conn = open_db()
    ensure_schema(conn)
    cid = _make_chunk(conn, "| 被保护的碎片 | 不删除 |", oom_adj=-500, age_days=0.0)
    result = shrink_dcache_sb(conn, "test_proj")
    assert cid not in result["deleted_ids"]
    assert _exists(conn, cid)
    conn.close()


def test_task_state_preserved():
    """task_state 类型不删除（控制面数据）。"""
    conn = open_db()
    ensure_schema(conn)
    cid = _make_chunk(conn, "| 控制面 | 不删除 |", chunk_type="task_state", age_days=0.0)
    result = shrink_dcache_sb(conn, "test_proj")
    assert cid not in result["deleted_ids"]
    assert _exists(conn, cid)
    conn.close()


def test_max_delete_cap():
    """max_delete 限制单次最大删除数。"""
    conn = open_db()
    ensure_schema(conn)
    # 创建 5 个碎片
    cids = []
    for i in range(5):
        cid = _make_chunk(conn, f"| 碎片{i} | 被删除 |", age_days=0.0)
        cids.append(cid)
    # 设置 max_delete=3
    os.environ["MEMORY_OS_SHRINK_DCACHE_SB_MAX_DELETE"] = "3"
    try:
        result = shrink_dcache_sb(conn, "test_proj")
        assert result["deleted"] == 3
    finally:
        del os.environ["MEMORY_OS_SHRINK_DCACHE_SB_MAX_DELETE"]
    conn.close()


def test_disabled():
    """shrink_dcache_sb.enabled=False 时不执行。"""
    conn = open_db()
    ensure_schema(conn)
    cid = _make_chunk(conn, "| 碎片 | 不应删除 |", age_days=0.0)
    os.environ["MEMORY_OS_SHRINK_DCACHE_SB_ENABLED"] = "false"
    try:
        result = shrink_dcache_sb(conn, "test_proj")
        assert result["deleted"] == 0
        assert _exists(conn, cid)
    finally:
        del os.environ["MEMORY_OS_SHRINK_DCACHE_SB_ENABLED"]
    conn.close()


def test_no_age_gate():
    """与 R4 不同：age=0 的碎片也立即删除（无冷启动保护）。"""
    conn = open_db()
    ensure_schema(conn)
    # age=0 (刚创建)
    cid_new = _make_chunk(conn, "| 刚创建 | 零天龄 | 应删除 |", age_days=0.0)
    # age=30 days
    cid_old = _make_chunk(conn, "| 很旧 | 三十天 | 也应删除 |", age_days=30.0)
    result = shrink_dcache_sb(conn, "test_proj")
    assert cid_new in result["deleted_ids"]
    assert cid_old in result["deleted_ids"]
    conn.close()


def test_colon_suffix_fragment_deleted():
    """以冒号结尾的标题碎片应被删除。"""
    conn = open_db()
    ensure_schema(conn)
    cid = _make_chunk(conn, "这是一个标题碎片的示例内容：", age_days=0.0)
    result = shrink_dcache_sb(conn, "test_proj")
    assert cid in result["deleted_ids"]
    conn.close()


def test_idempotent():
    """二次执行不会报错，也不会删除额外数据。"""
    conn = open_db()
    ensure_schema(conn)
    cid = _make_chunk(conn, "| 碎片 | 应被删除 |", age_days=0.0)
    result1 = shrink_dcache_sb(conn, "test_proj")
    assert result1["deleted"] == 1
    result2 = shrink_dcache_sb(conn, "test_proj")
    assert result2["deleted"] == 0
    conn.close()


def test_global_project_scan():
    """project=None 时扫描全库。"""
    conn = open_db()
    ensure_schema(conn)
    cid1 = _make_chunk(conn, "| 项目A碎片 | 内容 |", project="proj_a", age_days=0.0)
    cid2 = _make_chunk(conn, "| 项目B碎片 | 内容 |", project="proj_b", age_days=0.0)
    result = shrink_dcache_sb(conn, None)
    assert cid1 in result["deleted_ids"]
    assert cid2 in result["deleted_ids"]
    conn.close()


def test_performance():
    """100 chunks 扫描应 < 50ms。"""
    import time
    conn = open_db()
    ensure_schema(conn)
    for i in range(100):
        _make_chunk(conn, f"正常 chunk {i} 选择了某个技术方案因为性能更好", age_days=0.0)
    t0 = time.time()
    result = shrink_dcache_sb(conn, "test_proj")
    elapsed = (time.time() - t0) * 1000
    assert elapsed < 50, f"Too slow: {elapsed:.1f}ms"
    assert result["deleted"] == 0  # 正常 chunks 不被删除
    conn.close()


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
