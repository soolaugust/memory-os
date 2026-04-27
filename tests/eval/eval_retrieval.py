#!/usr/bin/env python3
"""
Memory OS Retrieval Quality Benchmark — 检索精度评估 harness

迭代目标：量化 retriever 系统的 precision@3, recall@3, MRR。
测试数据：基于真实 store.db chunks（42 个 chunks）构造 15 个 ground truth 对。
"""

import sys
import json
import sqlite3
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone

# 添加 memory-os 路径
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from store import (
    open_db, ensure_schema, fts_search, get_chunks, get_project_chunk_count
)
from scorer import retrieval_score, recency_score
from bm25 import hybrid_tokenize, bm25_scores, normalize


# ── Test Dataset: 15 ground truth (query → expected_chunks) ──

def build_ground_truth():
    """
    构造 15 个测试用例，覆盖：
    1. 精确匹配（关键词直接出现在 summary）
    2. 语义相关（近义词/同概念）
    3. 负例（完全不相关）
    4. 中文 query
    5. 英文 query

    返回：[
        {
            "query": "...",
            "description": "...",
            "expected_chunk_ids": [list of IDs],
            "category": "exact|semantic|negative|chinese|english"
        },
        ...
    ]
    """
    return [
        # ── 分类1：精确匹配（chunk summary 中的关键词直接出现）──
        {
            "query": "BM25 延迟 3ms",
            "description": "精确匹配：BM25 algorithm latency test",
            "expected_chunk_ids": ["e5b31de1-3dab-41fd-bb41-ed18d26d7a44"],
            "category": "exact_chinese"
        },
        {
            "query": "优先级判断 11",
            "description": "精确匹配：priority ranking with 11",
            "expected_chunk_ids": ["6d5ddc6d-f342-4710-bdbc-c9a9f33329fb"],
            "category": "exact_chinese"
        },
        {
            "query": "Hook 合并 103 34 次",
            "description": "精确匹配：Hook merging 103→34 times",
            "expected_chunk_ids": ["b06c3a96-8c96-4099-a384-cf04205d2cbf"],
            "category": "exact_chinese"
        },
        {
            "query": "P50 9ms 知识 37条",
            "description": "精确匹配：P50 latency 9ms with 37 knowledge items",
            "expected_chunk_ids": ["e61f04ad-77e9-46cf-ac9d-863498290929"],
            "category": "exact_chinese"
        },

        # ── 分类2：语义相关（近义词/同概念）──
        {
            "query": "响应时间延迟 47%",
            "description": "语义：响应时间/延迟改进百分比",
            "expected_chunk_ids": ["f337826f-ddd5-4ad4-b5dc-0c74bfb9c513"],
            "category": "semantic"
        },
        {
            "query": "缓存层级 KV 上下文 SQLite",
            "description": "语义：memory hierarchy levels from KV to storage",
            "expected_chunk_ids": ["76f8fe21-b3de-47da-a46a-2b3087c30f36"],
            "category": "semantic"
        },
        {
            "query": "工作集 恢复 利用率 33% 50%",
            "description": "语义：working set recovery and utilization metrics",
            "expected_chunk_ids": ["0048a525-ec63-4cc8-9e3a-b4bf28e38d87"],
            "category": "semantic"
        },

        # ── 分类3：多结果匹配（召回范围广）──
        {
            "query": "知识库 上下文 注入",
            "description": "多结果：knowledge injection into context",
            "expected_chunk_ids": ["63ad8093-2c99-41e6-bb0d-ba38aa9d08c0", "1ae234aa-631a-4a82-99bb-c3b7e37d2b50"],
            "category": "broad"
        },
        {
            "query": "测试 通过 验证",
            "description": "语义：testing and verification",
            "expected_chunk_ids": ["4cd00c39-5f35-4c37-8e97-dddd7d461382", "e60b2381-1c8d-4e4b-b9c3-7e8f9b1a2c3d"],
            "category": "semantic"
        },

        # ── 分类4：英文 query ──
        {
            "query": "swap partition compressed data",
            "description": "英文：swap management with compression",
            "expected_chunk_ids": [],  # 无直接匹配，用于验证 recall=0 场景
            "category": "english_no_match"
        },
        {
            "query": "decision chunk importance access count",
            "description": "英文：decision chunks 评分相关",
            "expected_chunk_ids": ["6d5ddc6d-f342-4710-bdbc-c9a9f33329fb", "e5b31de1-3dab-41fd-bb41-ed18d26d7a44"],
            "category": "english"
        },

        # ── 分类5：否定/负例（完全不相关）──
        {
            "query": "天气 温度 降雨量 彩虹",
            "description": "负例：weather query with no memory-os relevance",
            "expected_chunk_ids": [],
            "category": "negative"
        },
        {
            "query": "烹饪 菜谱 美食 厨师",
            "description": "负例：cooking query",
            "expected_chunk_ids": [],
            "category": "negative"
        },

        # ── 分类6：中文长查询 ──
        {
            "query": "Memory-OS 迭代过程中实现的 Hook 合并优化以及知识利用率提升的具体数字",
            "description": "中文长查询：Memory-OS iterations and optimizations",
            "expected_chunk_ids": ["b06c3a96-8c96-4099-a384-cf04205d2cbf", "0048a525-ec63-4cc8-9e3a-b4bf28e38d87", "ddd4338f-74a3-4cfa-951a-c3de28cc5b3c"],
            "category": "chinese_long"
        },

        # ── 分类7：技术缩写 ──
        {
            "query": "FTS5 索引 SQLite",
            "description": "技术术语：FTS5 and SQLite indexing",
            "expected_chunk_ids": ["ee353cfb-325c-4104-bd57-7fab13d1db0f"],
            "category": "tech_terms"
        },
    ]


# ── Baseline Scorer（纯 importance 排序，不用 BM25）──

def baseline_retrieval_score(chunk: dict) -> float:
    """
    简单 baseline：只用 importance 排序，不含相关性。
    用于对比 BM25+scorer 的提升。
    """
    importance = float(chunk.get("importance", 0))
    access_count = chunk.get("access_count", 0) or 0
    # 简单：importance + access_bonus
    return importance + min(0.2, access_count * 0.05)


# ── Evaluation Metrics ──

def compute_metrics(ground_truth, retrieved_ids, k=3):
    """
    计算单个测试用例的 precision@k, recall@k, MRR。

    Args:
        ground_truth: list of expected chunk IDs
        retrieved_ids: list of retrieved chunk IDs (ordered by score descending)
        k: top-k 参数（默认 3）

    Returns:
        {"precision@3": float, "recall@3": float, "mrr": float, "hit@3": int}
    """
    if not ground_truth:
        # 如果 ground truth 为空（负例），则 recall 无定义，只统计 false positive
        top_k_ids = set(retrieved_ids[:k])
        fp_count = len(top_k_ids)  # false positive
        return {
            "precision@3": 0.0,
            "recall@3": None,  # N/A
            "mrr": 0.0,
            "hit@3": 0,
            "fp_count": fp_count,
        }

    expected_set = set(ground_truth)
    top_k_ids = set(retrieved_ids[:k])
    hits = expected_set & top_k_ids

    precision_k = len(hits) / k if k > 0 else 0.0
    recall_k = len(hits) / len(expected_set) if expected_set else 0.0

    # MRR: Mean Reciprocal Rank — 第一个命中的排名倒数
    mrr = 0.0
    for i, rid in enumerate(retrieved_ids, 1):
        if rid in expected_set:
            mrr = 1.0 / i
            break

    return {
        "precision@3": precision_k,
        "recall@3": recall_k,
        "mrr": mrr,
        "hit@3": len(hits),
    }


# ── Main Benchmark ──

def benchmark():
    """
    执行完整 benchmark：
    1. 读取 store.db chunks
    2. 对 15 个 ground truth 调用 retriever
    3. 计算 precision/recall/MRR
    4. 输出对比报告（BM25 vs baseline）
    """
    db_path = Path.home() / ".claude" / "memory-os" / "store.db"
    if not db_path.exists():
        print(f"Error: {db_path} not found")
        return

    print("\n=== Memory OS Retrieval Quality Benchmark ===\n")

    # 连接数据库
    conn = open_db()
    ensure_schema(conn)

    # 获取项目 ID — 优先使用 eval_dataset.json 中指定的 project
    eval_dataset_path = _ROOT / "eval_dataset.json"
    if eval_dataset_path.exists():
        _ds = json.loads(eval_dataset_path.read_text())
        project = _ds.get("project", "")
        _dynamic_tests = _ds.get("tests", [])
    else:
        project = ""
        _dynamic_tests = []

    if not project:
        cursor = conn.cursor()
        cursor.execute("SELECT project, COUNT(*) as n FROM memory_chunks GROUP BY project ORDER BY n DESC LIMIT 1")
        result = cursor.fetchone()
        project = result[0] if result else "default"

    print(f"Using project: {project}\n")

    # 读取所有 chunks
    all_chunks = get_chunks(conn, project)
    chunk_by_id = {c["id"]: c for c in all_chunks}

    # 动态数据集优先，fallback 到 build_ground_truth()
    ground_truth_list = _dynamic_tests if _dynamic_tests else build_ground_truth()

    print(f"Dataset: {len(all_chunks)} chunks, {len(ground_truth_list)} test queries\n")
    print(f"Chunks by type:")
    type_counts = defaultdict(int)
    for c in all_chunks:
        type_counts[c.get("chunk_type", "unknown")] += 1
    for ct, cnt in sorted(type_counts.items()):
        print(f"  {ct}: {cnt}")
    print()

    # ── 策略1：FTS5 + BM25 + Scorer（生产）──
    bm25_results = {
        "precision_at_3": [],
        "recall_at_3": [],
        "mrr": [],
        "hit_counts": [],
        "query_details": [],
    }

    # ── 策略2：Baseline（纯 importance）──
    baseline_results = {
        "precision_at_3": [],
        "recall_at_3": [],
        "mrr": [],
        "hit_counts": [],
        "query_details": [],
    }

    print("Running retrievals...\n")
    worst_queries = []  # 追踪最差性能的 queries

    for gt in ground_truth_list:
        query = gt["query"]
        expected_ids = gt["expected_chunk_ids"]
        category = gt.get("category", "unknown")

        # ── BM25 + FTS5 检索 ──
        try:
            # 方法1：用 fts_search 直接召回
            fts_results = fts_search(conn, query, project, top_k=10)
            bm25_top_k_ids = [r["id"] for r in fts_results[:3]]
        except Exception as e:
            # fallback：Python BM25
            try:
                chunks = get_chunks(conn, project)
                search_texts = [f"{c['summary']} {c['content']}" for c in chunks]
                scores = bm25_scores(query, search_texts)
                normalized = normalize(scores)
                scored_chunks = list(zip(normalized, chunks))
                scored_chunks.sort(key=lambda x: x[0], reverse=True)
                bm25_top_k_ids = [c["id"] for s, c in scored_chunks[:3] if s > 0]
            except Exception:
                bm25_top_k_ids = []

        # ── Baseline：纯 importance 排序 ──
        chunks = get_chunks(conn, project)
        scored = [(baseline_retrieval_score(c), c) for c in chunks]
        scored.sort(key=lambda x: x[0], reverse=True)
        baseline_top_k_ids = [c["id"] for s, c in scored[:3]]

        # 计算指标
        bm25_metrics = compute_metrics(expected_ids, bm25_top_k_ids, k=3)
        baseline_metrics = compute_metrics(expected_ids, baseline_top_k_ids, k=3)

        # 记录结果
        for metric_name in ["precision_at_3", "recall_at_3", "mrr", "hit_counts"]:
            key = metric_name.replace("_at_3", "@3").replace("_counts", "@3")
            if key in bm25_metrics:
                val = bm25_metrics[key]
                if val is not None:
                    bm25_results[metric_name].append(val)

        for metric_name in ["precision_at_3", "recall_at_3", "mrr", "hit_counts"]:
            key = metric_name.replace("_at_3", "@3").replace("_counts", "@3")
            if key in baseline_metrics:
                val = baseline_metrics[key]
                if val is not None:
                    baseline_results[metric_name].append(val)

        detail = {
            "query": query[:80],
            "category": category,
            "expected": expected_ids,
            "bm25_top_3": bm25_top_k_ids,
            "baseline_top_3": baseline_top_k_ids,
            "bm25_recall": bm25_metrics.get("recall@3"),
            "baseline_recall": baseline_metrics.get("recall@3"),
        }
        bm25_results["query_details"].append(detail)
        baseline_results["query_details"].append(detail)

        # 追踪最差性能
        if bm25_metrics.get("recall@3", 0) == 0 and expected_ids:
            worst_queries.append(detail)

    conn.close()

    # ── 输出报告 ──
    def safe_avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    print("=" * 70)
    print("RESULTS\n")

    print("BM25 + Scorer (Production):")
    print(f"  Precision@3: {safe_avg(bm25_results['precision_at_3']):.3f}")
    print(f"  Recall@3:    {safe_avg(bm25_results['recall_at_3']):.3f}")
    print(f"  MRR:         {safe_avg(bm25_results['mrr']):.3f}")
    print(f"  Avg Hit:     {safe_avg(bm25_results['hit_counts']):.2f}")
    print()

    print("Baseline (Importance Only):")
    print(f"  Precision@3: {safe_avg(baseline_results['precision_at_3']):.3f}")
    print(f"  Recall@3:    {safe_avg(baseline_results['recall_at_3']):.3f}")
    print(f"  MRR:         {safe_avg(baseline_results['mrr']):.3f}")
    print(f"  Avg Hit:     {safe_avg(baseline_results['hit_counts']):.2f}")
    print()

    # 计算提升
    bm25_recall = safe_avg(bm25_results['recall_at_3'])
    baseline_recall = safe_avg(baseline_results['recall_at_3'])
    improvement = (bm25_recall - baseline_recall) / max(baseline_recall, 0.001) * 100

    print(f"Improvement: {improvement:+.1f}% recall@3")
    print()

    # 按类型分组统计
    print("By Category:")
    category_stats = defaultdict(lambda: {"bm25": [], "baseline": []})
    for detail in bm25_results["query_details"]:
        cat = detail["category"]
        category_stats[cat]["bm25"].append(detail["bm25_recall"] or 0)
        category_stats[cat]["baseline"].append(detail["baseline_recall"] or 0)

    for cat in sorted(category_stats.keys()):
        stats = category_stats[cat]
        bm25_avg = safe_avg(stats["bm25"])
        baseline_avg = safe_avg(stats["baseline"])
        n = len(stats["bm25"])
        print(f"  {cat:20s}: BM25={bm25_avg:.2f} baseline={baseline_avg:.2f} (N={n})")
    print()

    # 最差 queries
    if worst_queries:
        print(f"Worst Queries (recall=0, N={len(worst_queries)}):")
        for i, q in enumerate(worst_queries[:5], 1):
            print(f"  {i}. {q['query'][:70]}")
            print(f"     expected={q['expected'][:2] if q['expected'] else 'None'}")
            print(f"     got={q['bm25_top_3'][:2] if q['bm25_top_3'] else 'None'}")
        print()

    # 输出 JSON 结果（便于后续分析）
    result_file = Path(__file__).parent / "eval_results.json"
    with open(result_file, "w") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset": {
                "chunk_count": len(all_chunks),
                "test_count": len(ground_truth_list),
                "type_distribution": dict(type_counts),
            },
            "bm25": {
                "precision_at_3": safe_avg(bm25_results['precision_at_3']),
                "recall_at_3": bm25_recall,
                "mrr": safe_avg(bm25_results['mrr']),
                "avg_hit": safe_avg(bm25_results['hit_counts']),
            },
            "baseline": {
                "precision_at_3": safe_avg(baseline_results['precision_at_3']),
                "recall_at_3": baseline_recall,
                "mrr": safe_avg(baseline_results['mrr']),
                "avg_hit": safe_avg(baseline_results['hit_counts']),
            },
            "improvement_pct": improvement,
        }, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {result_file}\n")


if __name__ == "__main__":
    benchmark()
