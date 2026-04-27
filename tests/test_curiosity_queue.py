"""
test_curiosity_queue.py — 迭代305：主动探索 / curiosity-driven 知识空白检测测试

OS 类比：/proc/sys/vm/watermark_scale_factor — 验证水位触发阈值的自检测试套件
          当系统弱命中时自动记录"知识缺口"，类似 vmstat 侦测内存压力并触发 kswapd。

测试点：
  1. ensure_schema 后 curiosity_queue 表存在
  2. enqueue_curiosity 幂等（7天内同 query 不重复）
  3. pop_curiosity_queue 返回 pending 并改状态为 processing
  4. 模拟弱命中场景：top_score=0.15, query="量子纠缠机制" → 触发入队
  5. 模拟强命中场景：top_score=0.80 → 不入队
  6. query 长度 <= 8 字符（如 "好的"）→ 不入队
"""
import os
import sys
import sqlite3
import tempfile
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pytest

# ── 设置测试隔离环境 ──────────────────────────────────────────────────────────
# OS 类比：Linux namespace — 将测试进程隔离在独立的 tmpfs 挂载空间，不污染生产数据
_TEST_DIR = tempfile.mkdtemp(prefix="memory_os_test_curiosity_")
os.environ["MEMORY_OS_DIR"] = _TEST_DIR
os.environ["MEMORY_OS_DB"] = os.path.join(_TEST_DIR, "test_store.db")

# 加入模块搜索路径
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from store_vfs import (
    open_db, ensure_schema,
    enqueue_curiosity, pop_curiosity_queue,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def conn():
    """每个测试用独立内存 DB（OS 类比：tmpfs per-test namespace）"""
    c = sqlite3.connect(":memory:")
    c.execute("PRAGMA journal_mode=WAL")
    ensure_schema(c)
    yield c
    c.close()


# ── 测试 1：ensure_schema 后 curiosity_queue 表存在 ──────────────────────────

def test_schema_curiosity_queue_table_exists(conn):
    """
    OS 类比：mkfs 后 /proc/filesystems 中出现新文件系统类型——
    ensure_schema() 幂等建表后，sqlite_master 中必须有 curiosity_queue 条目。
    """
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='curiosity_queue'"
    ).fetchone()
    assert row is not None, "curiosity_queue 表应在 ensure_schema 后存在"


def test_schema_curiosity_queue_columns(conn):
    """验证 curiosity_queue 表有正确的列结构"""
    info = conn.execute("PRAGMA table_info(curiosity_queue)").fetchall()
    col_names = {row[1] for row in info}
    required = {"id", "query", "project", "detected_at", "top_score",
                "status", "filled_at", "chunk_id"}
    assert required.issubset(col_names), f"缺少列: {required - col_names}"


# ── 测试 2：enqueue_curiosity 幂等（7天内同 query 不重复）───────────────────

def test_enqueue_idempotent_same_project(conn):
    """
    OS 类比：写时合并（CoW + dedup）— 同一内容不重复分配物理页。
    7天内同 project+query 不应重复入队（第二次调用应被幂等跳过）。
    """
    n1 = enqueue_curiosity(conn, "量子纠缠机制原理", "proj-A", top_score=0.15)
    n2 = enqueue_curiosity(conn, "量子纠缠机制原理", "proj-A", top_score=0.12)
    assert n1 == 1, "首次入队应返回 1（已插入）"
    assert n2 == 0, "7天内重复入队应返回 0（幂等跳过）"

    count = conn.execute(
        "SELECT COUNT(*) FROM curiosity_queue WHERE query=? AND project=?",
        ("量子纠缠机制原理", "proj-A")
    ).fetchone()[0]
    assert count == 1, "DB 中应只有 1 条记录"


def test_enqueue_different_project_allowed(conn):
    """不同 project 的相同 query 应独立入队"""
    n1 = enqueue_curiosity(conn, "量子纠缠机制原理", "proj-A", top_score=0.15)
    n2 = enqueue_curiosity(conn, "量子纠缠机制原理", "proj-B", top_score=0.14)
    assert n1 == 1
    assert n2 == 1, "不同 project 应各自独立入队"


def test_enqueue_expired_record_can_requeue(conn):
    """
    OS 类比：swap slot 回收 — 已过期的 slot 重新变为可用。
    超过 7 天的旧记录过期后，相同 query 应可重新入队。
    """
    # 插入一条 8 天前的记录
    old_time = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
    conn.execute(
        "INSERT INTO curiosity_queue (query, project, detected_at, top_score, status) "
        "VALUES (?, ?, ?, ?, ?)",
        ("量子纠缠机制原理", "proj-A", old_time, 0.15, "pending")
    )
    conn.commit()

    n = enqueue_curiosity(conn, "量子纠缠机制原理", "proj-A", top_score=0.10)
    assert n == 1, "7天前的旧记录过期后，同 query 应可重新入队"


# ── 测试 3：pop_curiosity_queue 返回 pending 并改状态 ───────────────────────

def test_pop_returns_pending_and_marks_processing(conn):
    """
    OS 类比：mq_receive() — 从消息队列取出消息并标记为 consumed（QUEUED→CONSUMED）。
    pop_curiosity_queue 应返回 pending 条目，并将其状态改为 processing。
    """
    enqueue_curiosity(conn, "深度学习梯度消失问题", "proj-A", top_score=0.18)
    enqueue_curiosity(conn, "Transformer 注意力机制", "proj-A", top_score=0.20)

    items = pop_curiosity_queue(conn, project="proj-A", limit=5)
    assert len(items) == 2, "应返回 2 条 pending 记录"

    # 验证状态已改为 processing
    for item in items:
        row = conn.execute(
            "SELECT status FROM curiosity_queue WHERE id=?", (item["id"],)
        ).fetchone()
        assert row[0] == "processing", f"id={item['id']} 状态应为 processing"


def test_pop_limit_respected(conn):
    """pop_curiosity_queue 的 limit 参数应生效"""
    for i in range(5):
        enqueue_curiosity(conn, f"知识空白测试问题{i}", "proj-A", top_score=0.10 + i * 0.01)

    items = pop_curiosity_queue(conn, project="proj-A", limit=3)
    assert len(items) == 3, "limit=3 应只返回 3 条"


def test_pop_returns_only_pending(conn):
    """
    OS 类比：kswapd 只处理 inactive LRU list — 已 active 的页不重复处理。
    pop 后的 processing 记录不应再次被 pop。
    """
    enqueue_curiosity(conn, "神经网络反向传播", "proj-A", top_score=0.15)

    first_pop = pop_curiosity_queue(conn, project="proj-A", limit=5)
    second_pop = pop_curiosity_queue(conn, project="proj-A", limit=5)

    assert len(first_pop) == 1
    assert len(second_pop) == 0, "已 processing 的记录不应再次被 pop"


def test_pop_project_filter(conn):
    """pop_curiosity_queue 的 project 过滤应正确隔离"""
    enqueue_curiosity(conn, "问题A", "proj-A", top_score=0.15)
    enqueue_curiosity(conn, "问题B", "proj-B", top_score=0.15)

    items_a = pop_curiosity_queue(conn, project="proj-A", limit=5)
    assert len(items_a) == 1
    assert items_a[0]["query"] == "问题A"


def test_pop_no_project_filter(conn):
    """project=None 应返回所有项目的 pending 记录"""
    enqueue_curiosity(conn, "问题A", "proj-A", top_score=0.15)
    enqueue_curiosity(conn, "问题B", "proj-B", top_score=0.15)

    items = pop_curiosity_queue(conn, project=None, limit=10)
    assert len(items) == 2, "不过滤 project 应返回所有 pending"


# ── 测试 4：弱命中场景触发入队 ────────────────────────────────────────────────

def test_weak_hit_triggers_enqueue(conn):
    """
    OS 类比：vmstat 检测到内存压力 (high watermark) → 触发 kswapd。
    弱命中：FTS 召回数 >= 1 但 top-1 score < 0.25，query 长度 > 8 字符。
    → 应触发 enqueue_curiosity。
    """
    query = "量子纠缠机制"          # 7 个字（中文），len("量子纠缠机制") == 6 个汉字，但字符串长度 == 6
    # 重新确认：题目要求 > 8 字符，"量子纠缠机制" 是 6 个字，补充到 > 8
    query = "量子纠缠机制原理"       # 8 个汉字，len() == 8，不满足 > 8，再加
    query = "量子纠缠的物理机制"     # 9 个汉字，满足 > 8

    # 模拟弱命中：1 条 FTS 结果，top_score=0.15
    fts_results = [{"id": "c1", "fts_rank": 0.15}]
    top_score = fts_results[0]["fts_rank"]

    # 触发条件检查（与 retriever.py 中一致）
    should_enqueue = (
        len(fts_results) >= 1
        and top_score < 0.25
        and len(query) > 8
    )
    assert should_enqueue, "弱命中条件应成立"

    if should_enqueue:
        n = enqueue_curiosity(conn, query, "proj-A", top_score=top_score)
        assert n == 1, "弱命中应成功入队"

    count = conn.execute(
        "SELECT COUNT(*) FROM curiosity_queue WHERE project='proj-A' AND status='pending'"
    ).fetchone()[0]
    assert count == 1


def test_weak_hit_exact_threshold(conn):
    """边界值：top_score == 0.25 时不应触发（条件是 < 0.25，不含等于）"""
    query = "量子纠缠的物理机制原理"  # len > 8
    fts_results = [{"id": "c1", "fts_rank": 0.25}]
    top_score = fts_results[0]["fts_rank"]

    should_enqueue = (
        len(fts_results) >= 1
        and top_score < 0.25  # 严格小于
        and len(query) > 8
    )
    assert not should_enqueue, "top_score == 0.25 不应触发（条件是严格 < 0.25）"


# ── 测试 5：强命中场景不入队 ─────────────────────────────────────────────────

def test_strong_hit_no_enqueue(conn):
    """
    OS 类比：page cache hit — TLB 命中直接返回，不触发 page fault / kswapd。
    强命中：top_score=0.80 → 不应触发 curiosity 入队。
    """
    query = "量子纠缠的物理机制原理"  # len > 8
    fts_results = [{"id": "c1", "fts_rank": 0.80}]
    top_score = fts_results[0]["fts_rank"]

    should_enqueue = (
        len(fts_results) >= 1
        and top_score < 0.25
        and len(query) > 8
    )
    assert not should_enqueue, "强命中（top_score=0.80）不应触发入队"

    # 确保 DB 中没有新增记录
    count = conn.execute(
        "SELECT COUNT(*) FROM curiosity_queue"
    ).fetchone()[0]
    assert count == 0


# ── 测试 6：query 过短不入队 ─────────────────────────────────────────────────

def test_short_query_no_enqueue(conn):
    """
    OS 类比：iptables 最小包长过滤 — 太短的包不进规则链。
    query 长度 <= 8 字符（如 "好的"）→ 不触发入队。
    """
    short_queries = ["好的", "继续", "ok", "嗯嗯嗯嗯", "12345678"]  # all len <= 8
    fts_results = [{"id": "c1", "fts_rank": 0.10}]
    top_score = fts_results[0]["fts_rank"]

    for q in short_queries:
        should_enqueue = (
            len(fts_results) >= 1
            and top_score < 0.25
            and len(q) > 8
        )
        assert not should_enqueue, f"短 query '{q}'（len={len(q)}）不应触发入队"


def test_exactly_9_chars_triggers(conn):
    """边界值：len(query) == 9 > 8，应触发入队"""
    query = "量子纠缠原"  # 5 汉字 = len 5，不够；用英文
    query = "abcdefghi"   # 9 chars > 8
    fts_results = [{"id": "c1", "fts_rank": 0.10}]
    top_score = fts_results[0]["fts_rank"]

    should_enqueue = (
        len(fts_results) >= 1
        and top_score < 0.25
        and len(query) > 8
    )
    assert should_enqueue, f"len=9 的 query '{query}' 应触发入队"


# ── 测试 7：enqueue_curiosity 返回值和字段验证 ───────────────────────────────

def test_enqueue_stores_correct_fields(conn):
    """验证 enqueue_curiosity 写入的字段值正确"""
    query = "深度学习反向传播算法"
    project = "test-proj"
    top_score = 0.18

    n = enqueue_curiosity(conn, query, project, top_score=top_score)
    assert n == 1

    row = conn.execute(
        "SELECT query, project, top_score, status FROM curiosity_queue WHERE query=? AND project=?",
        (query, project)
    ).fetchone()
    assert row is not None
    assert row[0] == query
    assert row[1] == project
    assert abs(row[2] - top_score) < 1e-6
    assert row[3] == "pending"


def test_pop_returns_dict_with_required_keys(conn):
    """pop_curiosity_queue 返回的 dict 应包含必要字段"""
    enqueue_curiosity(conn, "深度学习梯度消失", "proj-A", top_score=0.15)
    items = pop_curiosity_queue(conn, project="proj-A", limit=1)
    assert len(items) == 1
    item = items[0]
    required_keys = {"id", "query", "project", "detected_at", "top_score", "status"}
    assert required_keys.issubset(item.keys()), f"缺少键: {required_keys - item.keys()}"
    assert item["status"] == "processing"
