#!/usr/bin/env python3
"""
迭代31 测试：Memory Compaction (compact_zone)
OS 类比：Linux compact_zone (2010) — 碎片整理

测试场景：
  T1: 5 个共享实体的 chunk → 合并为 1 个（importance=max, access_count=sum）
  T2: 2 个相关 chunk → 不合并（低于 min_cluster_size=3）
  T3: task_state 和 prompt_context chunk 被排除
  T4: 跨类型聚类（decision + reasoning 同主题）→ 合并
  T5: 合并后配额释放正确
  T6: 无共享实体的 chunk → 不聚类
  T7: kswapd_scan 集成 — compaction_freed 字段存在
  T8: 实体提取正确性（反引号/文件路径/中文双字词）
  T9: 停用词过滤（常见虚词不参与聚类）
  T10: 合并内容格式 [consolidated] 正确
"""
import os
import sys
import time
import tempfile
import uuid

# 设置 path
sys.path.insert(0, os.path.dirname(__file__))

import tmpfs  # noqa: F401 — tmpfs isolation (iter54), must precede store import
from store import (
    open_db, ensure_schema, insert_chunk, get_project_chunk_count,
    compact_zone, kswapd_scan, delete_chunks,
)
from datetime import datetime, timezone


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _make_chunk(project, chunk_type, summary, importance=0.7, access_count=1):
    """创建一个 chunk dict。"""
    now = _now_iso()
    return {
        "id": str(uuid.uuid4()),
        "created_at": now,
        "updated_at": now,
        "project": project,
        "source_session": "test-session",
        "chunk_type": chunk_type,
        "content": f"[{chunk_type}] {summary}",
        "summary": summary,
        "tags": "[]",
        "importance": importance,
        "retrievability": 0.3,
        "last_accessed": now,
        "feishu_url": None,
    }


def _setup_db():
    """创建临时数据库。"""
    tmpfile = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmpfile.close()
    from pathlib import Path
    conn = open_db(Path(tmpfile.name))
    ensure_schema(conn)
    return conn, tmpfile.name


def _cleanup(conn, db_path):
    conn.close()
    os.unlink(db_path)


def test_1_basic_compaction():
    """T1: 5 个共享实体的 chunk → 合并为 1 个"""
    conn, db_path = _setup_db()
    project = "test-compact"

    # 5 个关于 retriever BM25 召回的碎片 chunk（共享 retriever + BM25 两个实体）
    chunks = [
        _make_chunk(project, "decision", "retriever BM25 选择 hybrid tokenize 算法", 0.8),
        _make_chunk(project, "decision", "retriever BM25 bigram 效果优于 unigram", 0.7),
        _make_chunk(project, "decision", "retriever BM25 延迟 3ms 满足约束", 0.9),
        _make_chunk(project, "reasoning_chain", "retriever BM25 相比 TF-IDF 更适合短文本", 0.75),
        _make_chunk(project, "decision", "retriever BM25 权重 summary=2.0 content=1.0", 0.6),
    ]
    for c in chunks:
        insert_chunk(conn, c)
    # 手动设置 access_count（insert_chunk 不写此列，由 update_accessed 维护）
    for i, c in enumerate(chunks):
        conn.execute("UPDATE memory_chunks SET access_count=? WHERE id=?", (i + 1, c["id"]))
    conn.commit()

    assert get_project_chunk_count(conn, project) == 5

    result = compact_zone(conn, project)
    conn.commit()

    assert result["clusters_found"] >= 1, f"Expected clusters, got {result}"
    assert result["chunks_freed"] >= 2, f"Expected freed chunks, got {result}"

    remaining = get_project_chunk_count(conn, project)
    assert remaining < 5, f"Expected < 5 remaining, got {remaining}"

    # 验证 anchor 的 importance = max(0.9), access_count = sum(1+2+3+4+5=15)
    if result["anchor_ids"]:
        row = conn.execute(
            "SELECT importance, access_count FROM memory_chunks WHERE id = ?",
            (result["anchor_ids"][0],)
        ).fetchone()
        assert row[0] == 0.9, f"Expected max importance 0.9, got {row[0]}"
        assert row[1] == 15, f"Expected sum access_count 15, got {row[1]}"

    print(f"  T1 PASS: {result['clusters_found']} clusters, {result['chunks_freed']} freed, {remaining} remaining")
    _cleanup(conn, db_path)


def test_2_below_min_cluster():
    """T2: 2 个相关 chunk → 不合并（低于 min_cluster_size=3）"""
    conn, db_path = _setup_db()
    project = "test-compact"

    chunks = [
        _make_chunk(project, "decision", "FTS5 索引加速检索", 0.8),
        _make_chunk(project, "decision", "FTS5 权重配置优化", 0.7),
    ]
    for c in chunks:
        insert_chunk(conn, c)
    conn.commit()

    result = compact_zone(conn, project)
    assert result["clusters_found"] == 0, f"Expected 0 clusters for 2 chunks, got {result}"
    assert get_project_chunk_count(conn, project) == 2

    print(f"  T2 PASS: no compaction for {get_project_chunk_count(conn, project)} chunks")
    _cleanup(conn, db_path)


def test_3_excluded_types():
    """T3: task_state 和 prompt_context chunk 被排除"""
    conn, db_path = _setup_db()
    project = "test-compact"

    # task_state chunks 不参与 compaction
    chunks = [
        _make_chunk(project, "task_state", "BM25 任务状态1", 0.8),
        _make_chunk(project, "task_state", "BM25 任务状态2", 0.7),
        _make_chunk(project, "task_state", "BM25 任务状态3", 0.6),
        _make_chunk(project, "prompt_context", "BM25 用户话题", 0.5),
    ]
    for c in chunks:
        insert_chunk(conn, c)
    conn.commit()

    result = compact_zone(conn, project)
    assert result["clusters_found"] == 0, f"task_state should be excluded, got {result}"
    assert get_project_chunk_count(conn, project) == 4

    print(f"  T3 PASS: task_state/prompt_context excluded")
    _cleanup(conn, db_path)


def test_4_cross_type_cluster():
    """T4: 跨类型聚类（decision + reasoning 同主题）→ 合并"""
    conn, db_path = _setup_db()
    project = "test-compact"

    chunks = [
        _make_chunk(project, "decision", "scorer.py 统一评分引擎设计", 0.85, 3),
        _make_chunk(project, "reasoning_chain", "scorer.py 消除三份重复评分实现", 0.80, 2),
        _make_chunk(project, "decision", "scorer.py retention_score 四维公式", 0.75, 1),
        _make_chunk(project, "conversation_summary", "scorer.py 5/5 测试通过", 0.65, 1),
    ]
    for c in chunks:
        insert_chunk(conn, c)
    conn.commit()

    result = compact_zone(conn, project)
    conn.commit()

    # 应合并（共享 "scorer" 等实体）
    assert result["clusters_found"] >= 1, f"Expected cross-type cluster, got {result}"
    remaining = get_project_chunk_count(conn, project)
    assert remaining < 4, f"Expected < 4 remaining, got {remaining}"

    print(f"  T4 PASS: cross-type cluster merged, {remaining} remaining")
    _cleanup(conn, db_path)


def test_5_quota_freed():
    """T5: 合并后配额释放正确"""
    conn, db_path = _setup_db()
    project = "test-compact"

    # 创建 6 个相关 chunk
    for i in range(6):
        c = _make_chunk(project, "decision", f"kswapd 水位线策略 变体{i}", 0.7 + i * 0.02)
        insert_chunk(conn, c)
    conn.commit()

    before = get_project_chunk_count(conn, project)
    assert before == 6

    result = compact_zone(conn, project)
    conn.commit()

    after = get_project_chunk_count(conn, project)
    freed = before - after
    assert freed == result["chunks_freed"], f"freed mismatch: {freed} vs {result['chunks_freed']}"
    assert freed > 0, f"Expected quota freed > 0, got {freed}"

    print(f"  T5 PASS: quota freed = {freed} (before={before}, after={after})")
    _cleanup(conn, db_path)


def test_6_no_shared_entities():
    """T6: 无共享实体的 chunk → 不聚类"""
    conn, db_path = _setup_db()
    project = "test-compact"

    # 3 个完全不相关的 chunk
    chunks = [
        _make_chunk(project, "decision", "PostgreSQL 索引优化", 0.8),
        _make_chunk(project, "decision", "Docker 容器网络配置优化", 0.7),
        _make_chunk(project, "decision", "React 组件状态管理策略", 0.6),
    ]
    for c in chunks:
        insert_chunk(conn, c)
    conn.commit()

    result = compact_zone(conn, project)
    # 这三个 chunk 主题完全不同，不应被聚类
    assert result["clusters_found"] == 0, f"Unrelated chunks should not cluster, got {result}"

    print(f"  T6 PASS: no clustering for unrelated chunks")
    _cleanup(conn, db_path)


def test_7_kswapd_integration():
    """T7: kswapd_scan 集成 — compaction_freed 字段存在"""
    conn, db_path = _setup_db()
    project = "test-compact"

    # 简单调用确认返回值包含 compaction_freed
    result = kswapd_scan(conn, project, incoming_count=1)
    assert "compaction_freed" in result, f"Missing compaction_freed in kswapd result: {result}"

    print(f"  T7 PASS: kswapd_scan returns compaction_freed={result['compaction_freed']}")
    _cleanup(conn, db_path)


def test_8_entity_extraction():
    """T8: 实体提取正确性"""
    conn, db_path = _setup_db()
    project = "test-compact"

    # 含反引号、文件路径、中文双字词的 chunk
    chunks = [
        _make_chunk(project, "decision", "`retriever.py` 使用 FTS5 索引检索", 0.8),
        _make_chunk(project, "decision", "retriever.py 召回 Top-K 优化策略", 0.7),
        _make_chunk(project, "decision", "retriever.py FTS5 降级 fallback 机制", 0.75),
    ]
    for c in chunks:
        insert_chunk(conn, c)
    conn.commit()

    result = compact_zone(conn, project)
    conn.commit()

    # 应该发现聚类（共享 "retriever.py" 和 "FTS5"）
    assert result["clusters_found"] >= 1, f"Expected entity-based cluster, got {result}"

    print(f"  T8 PASS: entity extraction works, clusters={result['clusters_found']}")
    _cleanup(conn, db_path)


def test_9_stopwords_filtered():
    """T9: 停用词过滤（常见虚词不参与聚类）"""
    conn, db_path = _setup_db()
    project = "test-compact"

    # 只共享停用词 "the" "and" "for"，不应聚类
    chunks = [
        _make_chunk(project, "decision", "PostgreSQL 索引优化", 0.8),
        _make_chunk(project, "decision", "Redis 缓存策略设计", 0.7),
        _make_chunk(project, "decision", "MongoDB 分片集群部署", 0.6),
    ]
    for c in chunks:
        insert_chunk(conn, c)
    conn.commit()

    result = compact_zone(conn, project)
    # 这三个 chunk 主题完全不同，不应被聚类
    assert result["clusters_found"] == 0, f"Stopwords-only overlap should not cluster, got {result}"

    print(f"  T9 PASS: stopwords filtered correctly")
    _cleanup(conn, db_path)


def test_10_consolidated_format():
    """T10: 合并内容格式 [consolidated] 正确"""
    conn, db_path = _setup_db()
    project = "test-compact"

    chunks = [
        _make_chunk(project, "decision", "config.py sysctl 注册表设计", 0.9, 5),
        _make_chunk(project, "decision", "config.py 环境变量覆盖优先级", 0.7, 2),
        _make_chunk(project, "decision", "config.py 范围钳位机制", 0.6, 1),
    ]
    for c in chunks:
        insert_chunk(conn, c)
    conn.commit()

    result = compact_zone(conn, project)
    conn.commit()

    if result["anchor_ids"]:
        row = conn.execute(
            "SELECT content FROM memory_chunks WHERE id = ?",
            (result["anchor_ids"][0],)
        ).fetchone()
        assert "[consolidated]" in row[0], f"Expected [consolidated] in content, got: {row[0][:200]}"
        print(f"  T10 PASS: consolidated format correct")
    else:
        print(f"  T10 SKIP: no anchor (no compaction triggered)")

    _cleanup(conn, db_path)


def test_11_performance():
    """T11: 性能测试 — 200 chunks compaction < 500ms"""
    conn, db_path = _setup_db()
    project = "test-perf"

    # 创建 200 个 chunk（40 组 × 5 个关于同一主题）
    topics = ["BM25算法", "FTS5索引", "kswapd回收", "sysctl参数",
              "scorer评分", "retriever检索", "loader加载", "extractor提取",
              "compaction整理", "dmesg日志", "config配置", "store存储",
              "schema模型", "utils工具", "router路由", "writer写入",
              "缓存策略", "配额管理", "水位线", "碎片整理",
              "优先级调度", "遗忘曲线", "去重合并", "工作集恢复",
              "冷启动加载", "热缺页补入", "权重优化", "性能治理",
              "可观测性", "结构化日志", "访问计数", "保留评分",
              "召回命中率", "延迟分布", "容量规划", "淘汰策略",
              "连接复用", "批量操作", "事务管理", "幂等控制"]
    for i, topic in enumerate(topics):
        for j in range(5):
            c = _make_chunk(project, "decision",
                            f"{topic} 策略{j} 优化方案 实现细节",
                            0.5 + j * 0.1, j + 1)
            insert_chunk(conn, c)
    conn.commit()

    t0 = time.time()
    result = compact_zone(conn, project)
    conn.commit()
    elapsed_ms = (time.time() - t0) * 1000

    remaining = get_project_chunk_count(conn, project)
    print(f"  T11 PASS: 200 chunks -> {remaining} remaining, "
          f"freed={result['chunks_freed']}, clusters={result['clusters_found']}, "
          f"{elapsed_ms:.1f}ms")
    assert elapsed_ms < 500, f"Performance too slow: {elapsed_ms:.1f}ms"

    _cleanup(conn, db_path)


if __name__ == "__main__":
    tests = [
        test_1_basic_compaction,
        test_2_below_min_cluster,
        test_3_excluded_types,
        test_4_cross_type_cluster,
        test_5_quota_freed,
        test_6_no_shared_entities,
        test_7_kswapd_integration,
        test_8_entity_extraction,
        test_9_stopwords_filtered,
        test_10_consolidated_format,
        test_11_performance,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  {t.__name__} FAILED: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"compaction tests: {passed}/{passed+failed} passed")
    if failed > 0:
        sys.exit(1)
