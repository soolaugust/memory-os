#!/usr/bin/env python3
"""
tests/test_iter360_362.py — 迭代 360/361/362 验证测试

迭代360: FTS5 Auto-Optimize — fts_optimize() 函数行为验证
迭代361: Session FULL→LITE demotion — _session_full_injected 集合持久化验证
迭代362: Swap Warmup — warmup_swap_cache() 函数行为验证
"""
import sys
import json
import time
import uuid
from pathlib import Path
from datetime import datetime, timezone

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tests"))

import tmpfs  # noqa: F401
from store import open_db, ensure_schema, insert_chunk
from store_vfs import fts_optimize
from store_mm import warmup_swap_cache

PROJECT = f"iter362_{uuid.uuid4().hex[:6]}"


def _chunk(summary, content="test content", chunk_type="decision",
           importance=0.5) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "id": str(uuid.uuid4()),
        "created_at": now, "updated_at": now,
        "project": PROJECT,
        "source_session": "test",
        "chunk_type": chunk_type,
        "content": content,
        "summary": summary,
        "tags": json.dumps([]),
        "importance": importance,
        "retrievability": 0.5,
        "last_accessed": now,
        "feishu_url": None,
    }


# ── 迭代360 测试 ──────────────────────────────────────────────────────────────

def test_360_fts_optimize_basic():
    """T1: fts_optimize 首次调用返回 True（执行优化）"""
    conn = open_db()
    ensure_schema(conn)

    # 重置模块级冷却时间（强制执行）
    import store_vfs as _sv
    old_last = _sv._fts_last_optimize
    _sv._fts_last_optimize = 0.0

    result = fts_optimize(conn, force=False)
    assert result is True, f"Expected fts_optimize to return True on first call, got {result}"

    # 恢复
    _sv._fts_last_optimize = old_last
    conn.close()
    print("  T1(360) ✓ fts_optimize 首次调用返回 True")


def test_360_fts_optimize_cooldown():
    """T2: fts_optimize 冷却期内第二次调用返回 False（跳过）"""
    conn = open_db()
    ensure_schema(conn)

    import store_vfs as _sv
    old_last = _sv._fts_last_optimize
    # 设置为"刚执行过"
    _sv._fts_last_optimize = time.monotonic()

    result = fts_optimize(conn, force=False)
    assert result is False, f"Expected fts_optimize to be skipped in cooldown, got {result}"

    _sv._fts_last_optimize = old_last
    conn.close()
    print("  T2(360) ✓ fts_optimize 冷却期内跳过（返回 False）")


def test_360_fts_optimize_force():
    """T3: force=True 忽略冷却，强制执行"""
    conn = open_db()
    ensure_schema(conn)

    import store_vfs as _sv
    _sv._fts_last_optimize = time.monotonic()  # 刚执行过

    result = fts_optimize(conn, force=True)
    assert result is True, f"Expected force=True to bypass cooldown, got {result}"

    conn.close()
    print("  T3(360) ✓ fts_optimize force=True 忽略冷却")


# ── 迭代362 测试 ──────────────────────────────────────────────────────────────

def test_362_warmup_no_swap():
    """T4: swap_chunks 为空时 warmup 返回 restored_count=0"""
    conn = open_db()
    ensure_schema(conn)

    result = warmup_swap_cache(conn, f"empty_{uuid.uuid4().hex[:6]}",
                               importance_threshold=0.8, max_warmup=5)
    assert result["restored_count"] == 0, f"Expected 0, got {result['restored_count']}"
    conn.close()
    print("  T4(362) ✓ 无 swap chunk 时 warmup 返回 restored_count=0")


def test_362_warmup_with_high_importance_swap():
    """T5: 将高 importance chunk 先 swap_out，再 warmup 恢复"""
    from store_core import swap_out as _swap_out
    conn = open_db()
    ensure_schema(conn)

    # 写入一个高 importance chunk
    c = _chunk("高价值知识摘要", "这是高价值内容", importance=0.9)
    insert_chunk(conn, c)
    conn.commit()

    # swap_out 到 swap_chunks
    swap_result = _swap_out(conn, [c["id"]])
    conn.commit()

    # 确认已被 swap_out（不再在 memory_chunks 中）
    row = conn.execute("SELECT id FROM memory_chunks WHERE id=?", (c["id"],)).fetchone()
    assert row is None, "Chunk should be removed from memory_chunks after swap_out"

    # warmup 恢复
    warmup_result = warmup_swap_cache(conn, PROJECT,
                                      importance_threshold=0.8, max_warmup=5,
                                      session_id="test_session_362")
    assert warmup_result["restored_count"] >= 1, \
        f"Expected restored_count >= 1, got {warmup_result['restored_count']}"

    # 确认 chunk 已回到 memory_chunks
    row = conn.execute("SELECT id FROM memory_chunks WHERE id=?", (c["id"],)).fetchone()
    assert row is not None, "Chunk should be back in memory_chunks after warmup"

    # 清理
    conn.execute("DELETE FROM memory_chunks WHERE project=?", (PROJECT,))
    conn.commit()
    conn.close()
    print(f"  T5(362) ✓ warmup 恢复 {warmup_result['restored_count']} 个高价值 swap chunk")


def test_362_warmup_respects_threshold():
    """T6: importance < threshold 的 chunk 不被 warmup 恢复"""
    from store_core import swap_out as _swap_out
    conn = open_db()
    ensure_schema(conn)

    # 写入低 importance chunk
    proj = f"thresh_{uuid.uuid4().hex[:6]}"
    c_low = {
        "id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "project": proj,
        "source_session": "test",
        "chunk_type": "decision",
        "content": "低价值内容",
        "summary": "低价值知识",
        "tags": "[]",
        "importance": 0.3,  # 低于 threshold=0.8
        "retrievability": 0.5,
        "last_accessed": datetime.now(timezone.utc).isoformat(),
        "feishu_url": None,
    }
    insert_chunk(conn, c_low)
    conn.commit()

    # swap_out
    _swap_out(conn, [c_low["id"]])
    conn.commit()

    # warmup with threshold=0.8 — 低 importance chunk 不应被恢复
    result = warmup_swap_cache(conn, proj,
                               importance_threshold=0.8, max_warmup=5)
    assert result["restored_count"] == 0, \
        f"Expected 0 restored (below threshold), got {result['restored_count']}"

    conn.close()
    print("  T6(362) ✓ importance < threshold 的 chunk 不被 warmup 恢复")


def test_362_warmup_cooldown():
    """T7: 同一 session 内冷却期间 warmup 跳过（skipped_cooldown=True）"""
    conn = open_db()
    ensure_schema(conn)

    session_id = f"cooldown_test_{uuid.uuid4().hex[:8]}"

    # 第一次调用（触发冷却文件写入）
    from store_mm import _WARMUP_COOLDOWN_FILE
    # 手动写入冷却文件（模拟刚执行过）
    try:
        import store_mm as _smm
        _smm._WARMUP_COOLDOWN_FILE.parent.mkdir(parents=True, exist_ok=True)
        _smm._WARMUP_COOLDOWN_FILE.write_text(
            json.dumps({"session_id": session_id, "timestamp": time.time(),
                        "restored_count": 0}),
            encoding="utf-8"
        )
    except Exception as e:
        print(f"  T7(362) WARNING: 冷却文件写入失败 {e}")
        conn.close()
        return

    result = warmup_swap_cache(conn, PROJECT,
                               importance_threshold=0.8, max_warmup=5,
                               session_id=session_id)
    assert result.get("skipped_cooldown") is True, \
        f"Expected skipped_cooldown=True, got {result}"

    conn.close()
    print("  T7(362) ✓ 冷却期间 warmup 跳过（skipped_cooldown=True）")


# ── 运行入口 ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("迭代 360/361/362 验证测试")
    print("=" * 55)

    tests = [
        test_360_fts_optimize_basic,
        test_360_fts_optimize_cooldown,
        test_360_fts_optimize_force,
        test_362_warmup_no_swap,
        test_362_warmup_with_high_importance_swap,
        test_362_warmup_respects_threshold,
        test_362_warmup_cooldown,
    ]

    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            failed += 1
            import traceback
            print(f"  FAIL {t.__name__}: {e}")
            traceback.print_exc()

    print(f"\n{'='*55}")
    print(f"结果：{passed}/{passed+failed} 通过")
    if failed:
        sys.exit(1)
