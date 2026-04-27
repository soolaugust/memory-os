#!/usr/bin/env python3
"""
eval_v2.py — AIOS Memory-OS 三层评估框架

设计原则（第一性原理）：
  1. 不循环论证：query 不从 chunk summary 生成，用人类自然语言
  2. 三层评估：检索质量 → 语义差距 → 端到端行为
  3. 可复现：固定测试集，不依赖随机抽样
  4. 量化：每层输出一个核心指标，可跨版本追踪

L1 - 检索质量（Retrieval Quality）：
  用人工构造的自然语言查询测 recall@k / MRR / NDCG@k
  对标 LongMemEval 的 "information extraction" 维度

L2 - 语义差距量化（Semantic Gap）：
  专门构造同义词/近义/换述查询（BM25 应该失败的场景）
  量化 BM25 的实际天花板，为是否加 embedding 提供数据

L3 - 矛盾检测（Contradiction Detection）：
  扫描所有 decision chunk，用 BM25 找相似对，
  人工标注矛盾 → 量化矛盾密度

迭代 103：首版评估框架
"""

import sys
import json
import sqlite3
import math
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from store import open_db, ensure_schema, fts_search, get_chunks
from scorer import retrieval_score
from bm25 import hybrid_tokenize, bm25_scores, normalize


# ═══════════════════════════════════════════════════════════════════
# L1: Retrieval Quality — 人工构造的自然语言查询
# ═══════════════════════════════════════════════════════════════════

# 这些查询是人用自然语言写的，不是从 chunk summary 提取的。
# 每个查询模拟真实场景：用户在编码时想知道某个决策/约束/背景。
# expected_keywords 用于在当前 store.db 中动态匹配 ground truth。
L1_QUERIES = [
    # ── 决策类：为什么选了 X 而不是 Y ──
    {
        "query": "为什么放弃 chromadb 改用 FTS5",
        "intent": "查找弃用 chromadb 的决策和原因",
        "expected_keywords": ["chromadb", "FTS5", "BM25"],
        "category": "decision_recall",
    },
    {
        "query": "检索用什么优先级分类",
        "intent": "查找 retriever 的查询优先级机制",
        "expected_keywords": ["SKIP", "LITE", "FULL", "优先级", "nice"],
        "category": "decision_recall",
    },
    {
        "query": "chunk 最多能存多少条",
        "intent": "查找 chunk quota 配置",
        "expected_keywords": ["quota", "200", "chunk_quota"],
        "category": "config_recall",
    },
    {
        "query": "设计约束怎么保护不被淘汰",
        "intent": "查找 design_constraint 的 mlock/oom_adj 保护机制",
        "expected_keywords": ["oom_adj", "mlock", "design_constraint", "-800", "约束"],
        "category": "decision_recall",
    },
    {
        "query": "新 chunk 写入时怎么去重",
        "intent": "查找 dedup/merge 机制",
        "expected_keywords": ["dedup", "find_similar", "merge", "去重"],
        "category": "mechanism_recall",
    },

    # ── 架构类：系统怎么工作 ──
    {
        "query": "VFS 两级缓存是什么",
        "intent": "查找 KnowledgeVFS 的 dentry+inode cache 设计",
        "expected_keywords": ["VFS", "dentry", "inode", "cache", "缓存"],
        "category": "architecture_recall",
    },
    {
        "query": "检索延迟目标是多少毫秒",
        "intent": "查找性能目标/deadline",
        "expected_keywords": ["deadline", "50ms", "100ms", "延迟", "latency"],
        "category": "config_recall",
    },
    {
        "query": "kswapd 的三级水位线怎么设的",
        "intent": "查找内存管理的水位线配置",
        "expected_keywords": ["kswapd", "watermark", "水位", "pages_high", "pages_low"],
        "category": "mechanism_recall",
    },

    # ── 历史类：之前做过什么 ──
    {
        "query": "迭代 98 做了什么功能",
        "intent": "查找 iter98 的设计约束系统",
        "expected_keywords": ["iter98", "design_constraint", "约束", "constraint"],
        "category": "history_recall",
    },
    {
        "query": "跨会话目标追踪怎么实现的",
        "intent": "查找 Goal Persistence 机制",
        "expected_keywords": ["目标", "goal", "persistence", "跨会话"],
        "category": "mechanism_recall",
    },

    # ── 排除类（负例）：不应返回任何结果 ──
    {
        "query": "React hooks 的 useEffect 清理函数",
        "intent": "完全不相关的前端话题",
        "expected_keywords": [],
        "category": "negative",
    },
    {
        "query": "Kubernetes pod 亲和性调度",
        "intent": "不相关的运维话题",
        "expected_keywords": [],
        "category": "negative",
    },
]


# ═══════════════════════════════════════════════════════════════════
# L2: Semantic Gap — 同义词/近义/换述查询
# 每对 (original, paraphrase)：original 应该能命中，paraphrase 是同义但换了词
# ═══════════════════════════════════════════════════════════════════

L2_SEMANTIC_PAIRS = [
    {
        "original": "放弃 chromadb",
        "paraphrase": "弃用向量数据库",
        "intent": "测试 'chromadb' vs '向量数据库' 的语义等价",
    },
    {
        "original": "检索优先级 SKIP LITE FULL",
        "paraphrase": "查询分类三个级别",
        "intent": "测试 'SKIP/LITE/FULL' vs '三个级别' 的语义等价",
    },
    {
        "original": "kswapd 淘汰低价值 chunk",
        "paraphrase": "自动清理不重要的记忆",
        "intent": "测试 'kswapd' vs '自动清理' 的语义等价",
    },
    {
        "original": "设计约束 oom_adj -800",
        "paraphrase": "重要规则不能被删除",
        "intent": "测试 'oom_adj -800' vs '不能被删除' 的语义等价",
    },
    {
        "original": "VFS dentry cache inode cache",
        "paraphrase": "知识文件系统的两层缓存",
        "intent": "测试 'VFS dentry/inode' vs '知识文件系统两层缓存'",
    },
    {
        "original": "vDSO 快速路径 3ms",
        "paraphrase": "零开销的快速跳过机制",
        "intent": "测试 'vDSO' vs '快速跳过'",
    },
    {
        "original": "BM25 中文 bigram 分词",
        "paraphrase": "全文搜索的中文切词方式",
        "intent": "测试 'BM25 bigram' vs '全文搜索切词'",
    },
    {
        "original": "freshness_bonus grace_days",
        "paraphrase": "新知识的保护期加分",
        "intent": "测试 'freshness_bonus' vs '新知识保护期'",
    },
]


# ═══════════════════════════════════════════════════════════════════
# L3: Contradiction Detection — 扫描决策矛盾
# ═══════════════════════════════════════════════════════════════════

def find_potential_contradictions(conn, project, similarity_threshold=3):
    """
    用 BM25 找语义相似的 decision 对，标记可能矛盾的候选。

    策略：对每个 decision chunk，用其 summary 查 FTS5，
    找到相似度 > threshold 的其他 decision chunk 作为候选对。

    返回：[{"chunk_a": {...}, "chunk_b": {...}, "similarity": float}]
    """
    decisions = conn.execute("""
        SELECT id, summary, content, importance, created_at
        FROM memory_chunks
        WHERE project = ? AND chunk_type = 'decision'
        ORDER BY created_at
    """, [project]).fetchall()

    candidates = []
    seen_pairs = set()

    for i, (aid, asummary, acontent, aimp, acreated) in enumerate(decisions):
        # 用 summary 的前 50 字符作为查询
        query_text = (asummary or "")[:50]
        if len(query_text) < 5:
            continue

        results = fts_search(conn, query_text, project, top_k=5,
                           chunk_types=("decision",))

        for r in results:
            bid = r["id"]
            if bid == aid:
                continue
            pair_key = tuple(sorted([aid, bid]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            # 简单的矛盾信号检测：
            # 如果两个 decision 涉及同一主题但时间差 > 1 天，可能是更新
            bsummary = r["summary"]
            similarity = r.get("fts_rank", 0)

            if similarity >= similarity_threshold:
                candidates.append({
                    "chunk_a": {
                        "id": aid[:12],
                        "summary": asummary[:80],
                        "created": acreated[:10] if acreated else "?",
                    },
                    "chunk_b": {
                        "id": bid[:12],
                        "summary": bsummary[:80],
                        "created": r.get("created_at", "")[:10],
                    },
                    "similarity": round(similarity, 3),
                })

    return candidates


# ═══════════════════════════════════════════════════════════════════
# 评估执行器
# ═══════════════════════════════════════════════════════════════════

def ndcg_at_k(relevant_ids, retrieved_ids, k):
    """计算 NDCG@k (Normalized Discounted Cumulative Gain)"""
    if not relevant_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid in relevant_set:
            dcg += 1.0 / math.log2(i + 2)  # i+2 因为 log2(1)=0
    # iDCG: 理想情况下前 min(k, |relevant|) 个都命中
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant_ids))))
    return dcg / idcg if idcg > 0 else 0.0


def find_chunks_by_keywords(conn, project, keywords):
    """用关键词列表在 FTS5 中搜索，返回匹配的 chunk IDs"""
    if not keywords:
        return []

    all_ids = set()
    for kw in keywords:
        results = fts_search(conn, kw, project, top_k=5)
        for r in results:
            all_ids.add(r["id"])
    return list(all_ids)


def run_l1_retrieval(conn, project, k=5):
    """
    L1: 检索质量评估

    对每个人工查询：
    1. 先用 expected_keywords 动态定位 ground truth chunks
    2. 用自然语言 query 走 fts_search
    3. 计算 recall@k, MRR, NDCG@k
    """
    print("\n" + "=" * 70)
    print("L1: Retrieval Quality (Natural Language Queries)")
    print("=" * 70 + "\n")

    results = []
    category_stats = defaultdict(lambda: {"recall": [], "mrr": [], "ndcg": []})

    for q in L1_QUERIES:
        query = q["query"]
        keywords = q["expected_keywords"]
        category = q["category"]

        # 动态定位 ground truth
        gt_ids = find_chunks_by_keywords(conn, project, keywords)

        # 执行检索
        fts_results = fts_search(conn, query, project, top_k=k)
        retrieved_ids = [r["id"] for r in fts_results]

        # 计算指标
        if category == "negative":
            # 负例：检查 false positive
            fp = len(retrieved_ids)
            result = {
                "query": query,
                "category": category,
                "fp_count": fp,
                "recall@k": None,
                "mrr": None,
                "ndcg@k": None,
                "status": "PASS" if fp == 0 else f"FP={fp}",
            }
        elif not gt_ids:
            result = {
                "query": query,
                "category": category,
                "gt_count": 0,
                "recall@k": None,
                "mrr": None,
                "ndcg@k": None,
                "status": "NO_GT (keywords not found in store)",
            }
        else:
            gt_set = set(gt_ids)
            hits = gt_set & set(retrieved_ids[:k])
            recall = len(hits) / len(gt_set) if gt_set else 0
            mrr = 0.0
            for i, rid in enumerate(retrieved_ids, 1):
                if rid in gt_set:
                    mrr = 1.0 / i
                    break
            ndcg = ndcg_at_k(gt_ids, retrieved_ids, k)

            result = {
                "query": query,
                "category": category,
                "gt_count": len(gt_ids),
                "retrieved": len(retrieved_ids),
                "hits": len(hits),
                "recall@k": round(recall, 3),
                "mrr": round(mrr, 3),
                "ndcg@k": round(ndcg, 3),
                "status": "HIT" if recall > 0 else "MISS",
            }

            category_stats[category]["recall"].append(recall)
            category_stats[category]["mrr"].append(mrr)
            category_stats[category]["ndcg"].append(ndcg)

        results.append(result)

        # 打印
        status = result["status"]
        symbol = "✅" if status in ("HIT", "PASS") else "❌" if status == "MISS" else "⚠️"
        print(f"  {symbol} [{category:20s}] {query[:45]:45s} → {status}")
        if result.get("recall@k") is not None:
            print(f"     recall={result['recall@k']} mrr={result['mrr']} ndcg={result['ndcg@k']} gt={result['gt_count']}")

    # 汇总
    all_recalls = []
    all_mrrs = []
    all_ndcgs = []
    for cat_data in category_stats.values():
        all_recalls.extend(cat_data["recall"])
        all_mrrs.extend(cat_data["mrr"])
        all_ndcgs.extend(cat_data["ndcg"])

    def safe_avg(lst):
        return sum(lst) / len(lst) if lst else 0

    summary = {
        "total_queries": len(L1_QUERIES),
        "evaluated": len(all_recalls),
        "avg_recall@k": round(safe_avg(all_recalls), 3),
        "avg_mrr": round(safe_avg(all_mrrs), 3),
        "avg_ndcg@k": round(safe_avg(all_ndcgs), 3),
        "by_category": {},
    }
    for cat, data in sorted(category_stats.items()):
        summary["by_category"][cat] = {
            "avg_recall": round(safe_avg(data["recall"]), 3),
            "avg_mrr": round(safe_avg(data["mrr"]), 3),
            "count": len(data["recall"]),
        }

    print(f"\n  {'─' * 60}")
    print(f"  L1 Summary: Recall@{k}={summary['avg_recall@k']} | MRR={summary['avg_mrr']} | NDCG@{k}={summary['avg_ndcg@k']}")
    print(f"  ({summary['evaluated']} queries evaluated, {len(L1_QUERIES) - summary['evaluated']} negative/no-gt)\n")

    for cat, data in sorted(summary["by_category"].items()):
        print(f"    {cat:25s}: recall={data['avg_recall']:.3f} mrr={data['avg_mrr']:.3f} (N={data['count']})")

    return {"queries": results, "summary": summary}


def run_l2_semantic_gap(conn, project, k=5):
    """
    L2: 语义差距量化

    对每对 (original, paraphrase)：
    - original 查询应该命中
    - paraphrase 是同义但换了词，测试 BM25 是否也能命中
    - gap = original_recall - paraphrase_recall
    """
    print("\n" + "=" * 70)
    print("L2: Semantic Gap Analysis (BM25 Ceiling)")
    print("=" * 70 + "\n")

    results = []
    gaps = []

    for pair in L2_SEMANTIC_PAIRS:
        original = pair["original"]
        paraphrase = pair["paraphrase"]

        # original 查询
        orig_results = fts_search(conn, original, project, top_k=k)
        orig_ids = set(r["id"] for r in orig_results)

        # paraphrase 查询
        para_results = fts_search(conn, paraphrase, project, top_k=k)
        para_ids = set(r["id"] for r in para_results)

        # 计算重叠
        if orig_ids:
            overlap = len(orig_ids & para_ids) / len(orig_ids)
        else:
            overlap = None  # original 也没命中，说明查询本身就找不到

        gap = 1.0 - overlap if overlap is not None else None

        result = {
            "original": original,
            "paraphrase": paraphrase,
            "intent": pair["intent"],
            "orig_count": len(orig_ids),
            "para_count": len(para_ids),
            "overlap": round(overlap, 3) if overlap is not None else None,
            "gap": round(gap, 3) if gap is not None else None,
        }
        results.append(result)

        if gap is not None:
            gaps.append(gap)

        # 打印
        if overlap is None:
            symbol = "⚠️"
            status = "ORIG_MISS"
        elif gap == 0:
            symbol = "✅"
            status = "FULL_OVERLAP"
        elif gap < 0.5:
            symbol = "🟡"
            status = f"PARTIAL gap={gap:.0%}"
        else:
            symbol = "❌"
            status = f"HIGH_GAP gap={gap:.0%}"

        print(f"  {symbol} orig=\"{original[:30]:30s}\" → para=\"{paraphrase[:30]:30s}\" {status}")

    avg_gap = sum(gaps) / len(gaps) if gaps else 0
    full_overlap = sum(1 for g in gaps if g == 0)
    high_gap = sum(1 for g in gaps if g >= 0.5)

    summary = {
        "total_pairs": len(L2_SEMANTIC_PAIRS),
        "evaluated": len(gaps),
        "avg_semantic_gap": round(avg_gap, 3),
        "full_overlap_count": full_overlap,
        "high_gap_count": high_gap,
        "verdict": "BM25_SUFFICIENT" if avg_gap < 0.3 else "EMBEDDING_NEEDED" if avg_gap > 0.6 else "BORDERLINE",
    }

    print(f"\n  {'─' * 60}")
    print(f"  L2 Summary: Avg Semantic Gap = {summary['avg_semantic_gap']:.1%}")
    print(f"  Full overlap: {full_overlap}/{len(gaps)} | High gap (>50%): {high_gap}/{len(gaps)}")
    print(f"  Verdict: {summary['verdict']}")

    return {"pairs": results, "summary": summary}


def run_l3_contradiction(conn, project):
    """
    L3: 矛盾检测

    扫描 decision chunks，找语义相似的对，标记可能的矛盾。
    输出矛盾候选对数量和密度。
    """
    print("\n" + "=" * 70)
    print("L3: Contradiction Detection (Decision Consistency)")
    print("=" * 70 + "\n")

    # 获取 decision 总数
    total = conn.execute(
        "SELECT COUNT(*) FROM memory_chunks WHERE project = ? AND chunk_type = 'decision'",
        [project]
    ).fetchone()[0]

    candidates = find_potential_contradictions(conn, project, similarity_threshold=2)

    print(f"  Total decisions: {total}")
    print(f"  Similar pairs found: {len(candidates)}")
    print(f"  Contradiction density: {len(candidates)/max(total,1):.1%} (pairs/decisions)\n")

    # 显示 top 10 候选对
    sorted_candidates = sorted(candidates, key=lambda c: c["similarity"], reverse=True)
    for i, c in enumerate(sorted_candidates[:10], 1):
        a = c["chunk_a"]
        b = c["chunk_b"]
        print(f"  {i:2d}. sim={c['similarity']:.2f}")
        print(f"      A [{a['created']}]: {a['summary'][:65]}")
        print(f"      B [{b['created']}]: {b['summary'][:65]}")
        print()

    summary = {
        "total_decisions": total,
        "similar_pairs": len(candidates),
        "density": round(len(candidates) / max(total, 1), 3),
        "top_candidates": sorted_candidates[:10],
        "note": "Pairs need human review to confirm actual contradictions",
    }

    return {"candidates": sorted_candidates, "summary": summary}


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║   AIOS Memory-OS Evaluation Framework v2               ║")
    print("║   三层评估：检索质量 → 语义差距 → 矛盾检测             ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    conn = open_db()
    ensure_schema(conn)

    # 自动选择最大 project
    row = conn.execute("""
        SELECT project, COUNT(*) as n
        FROM memory_chunks
        GROUP BY project ORDER BY n DESC LIMIT 1
    """).fetchone()
    project = row[0] if row else "default"
    total = row[1] if row else 0

    print(f"Project: {project}")
    print(f"Total chunks: {total}")

    # 类型分布
    types = conn.execute("""
        SELECT chunk_type, COUNT(*) FROM memory_chunks
        WHERE project = ? GROUP BY chunk_type ORDER BY COUNT(*) DESC
    """, [project]).fetchall()
    for ct, cnt in types:
        print(f"  {ct}: {cnt}")

    # 运行三层评估
    l1 = run_l1_retrieval(conn, project, k=5)
    l2 = run_l2_semantic_gap(conn, project, k=5)
    l3 = run_l3_contradiction(conn, project)

    # 输出总结
    print("\n" + "=" * 70)
    print("OVERALL ASSESSMENT")
    print("=" * 70)

    l1_recall = l1["summary"]["avg_recall@k"]
    l2_gap = l2["summary"]["avg_semantic_gap"]
    l3_density = l3["summary"]["density"]

    print(f"""
  L1 Retrieval Quality:  Recall@5 = {l1_recall:.1%}  MRR = {l1['summary']['avg_mrr']:.3f}
  L2 Semantic Gap:       Avg Gap  = {l2_gap:.1%}    Verdict: {l2['summary']['verdict']}
  L3 Contradiction:      Density  = {l3_density:.1%}    ({l3['summary']['similar_pairs']} candidate pairs)

  Health Score: {_health_score(l1_recall, l2_gap, l3_density)}/100
""")

    # 保存结果
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "project": project,
        "total_chunks": total,
        "l1_retrieval": l1,
        "l2_semantic_gap": l2,
        "l3_contradiction": l3,
        "health_score": _health_score(l1_recall, l2_gap, l3_density),
    }

    out_path = _ROOT / "eval_v2_results.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=str))
    print(f"  Results saved to: {out_path}")

    conn.close()
    return output


def _health_score(recall, semantic_gap, contradiction_density):
    """
    综合健康分（0-100）：
    - L1 recall 权重 50%（核心功能）
    - L2 semantic gap 权重 30%（天花板）
    - L3 contradiction density 权重 20%（数据质量）
    """
    l1_score = recall * 100  # 0-100
    l2_score = (1 - semantic_gap) * 100  # gap 越小越好
    l3_score = max(0, (1 - contradiction_density * 10)) * 100  # density < 10% 满分
    return round(l1_score * 0.5 + l2_score * 0.3 + l3_score * 0.2)


if __name__ == "__main__":
    main()
