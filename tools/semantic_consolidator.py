#!/usr/bin/env python3
"""
tools/semantic_consolidator.py — 跨项目语义巩固（海马体 Replay）

OS 类比：Linux 慢波睡眠期间的海马体→新皮层记忆巩固
  海马体（情节记忆，project-local）在睡眠期间 replay，
  反复激活的模式抽象进入新皮层（语义记忆，project="__semantic__"），
  成为跨情境可激活的通用知识。

核心逻辑：
  1. 扫描所有 project 的高 importance chunk（importance >= threshold）
  2. 用 trigram Jaccard 找跨 project 相似 chunk（同一知识在不同场景出现）
  3. 合并相似 chunk 的 summary，写入 project="__semantic__" 的 semantic_memory chunk
  4. 有去重（summary hash）和更新机制，不重复写入

语义记忆层特性：
  - project="__semantic__"：不属于任何具体项目
  - chunk_type="semantic_memory"：被 retriever 和 loader 自动包含
  - info_class="semantic"：使用慢速衰减曲线（stability 衰减率 0.97）
  - importance 取源 chunk 平均值，不人为膨胀

用法：
  python3 tools/semantic_consolidator.py [--dry-run] [--threshold 0.70] [--min-importance 0.65]

可通过 cron 或 SessionStop hook 触发（建议每天一次）。
"""
import sys
import os
import re
import uuid
import json
import hashlib
import argparse
import sqlite3
from datetime import datetime, timezone
from typing import Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from store import open_db, ensure_schema

# 语义记忆的特殊项目 ID
SEMANTIC_PROJECT = "__semantic__"

# 不参与跨项目聚合的内部项目
_EXCLUDED_PROJECTS = {SEMANTIC_PROJECT, "__test__", ""}


def _trigram_similarity(a: str, b: str) -> float:
    """
    trigram Jaccard 相似度，无需 embedding。
    OS 类比：CRC32 快速数据块比较。
    """
    if not a or not b:
        return 0.0

    def trigrams(s: str):
        s = re.sub(r'\s+', ' ', s.strip().lower())
        return set(s[i:i+3] for i in range(len(s) - 2)) if len(s) >= 3 else set(s)

    ta, tb = trigrams(a), trigrams(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _summary_hash(summary: str) -> str:
    """生成 summary 的归一化 hash，用于去重。"""
    normalized = re.sub(r'\s+', ' ', summary.strip().lower())
    return hashlib.md5(normalized.encode()).hexdigest()[:16]


def _load_candidate_chunks(conn: sqlite3.Connection,
                            min_importance: float) -> list[dict]:
    """
    加载所有 project 的高 importance chunk 作为候选。
    排除 ghost chunk、__semantic__ project、以及 test 项目。
    """
    rows = conn.execute("""
        SELECT id, project, chunk_type, summary, content, importance,
               COALESCE(stability, 1.0) as stability,
               COALESCE(access_count, 0) as access_count,
               COALESCE(info_class, 'world') as info_class
        FROM memory_chunks
        WHERE importance >= ?
          AND project IS NOT NULL
          AND project NOT IN ('__semantic__', '')
          AND project NOT LIKE '__test__%'
          AND COALESCE(oom_adj, 0) <= 0
          AND COALESCE(content, '') NOT LIKE 'GHOST→%'
          AND length(COALESCE(summary, '')) > 15
        ORDER BY importance DESC, access_count DESC
    """, (min_importance,)).fetchall()

    return [
        {
            "id": r[0], "project": r[1], "chunk_type": r[2],
            "summary": r[3], "content": r[4], "importance": r[5],
            "stability": r[6], "access_count": r[7], "info_class": r[8],
        }
        for r in rows
    ]


def _load_existing_semantic_hashes(conn: sqlite3.Connection) -> dict[str, str]:
    """
    加载已有 semantic chunk 的 summary hash → chunk_id 映射，用于去重和更新。
    """
    rows = conn.execute("""
        SELECT id, summary FROM memory_chunks
        WHERE project = ? AND chunk_type = 'semantic_memory'
    """, (SEMANTIC_PROJECT,)).fetchall()

    return {_summary_hash(r[1]): r[0] for r in rows if r[1]}


def _find_cross_project_clusters(chunks: list[dict],
                                  sim_threshold: float) -> list[list[dict]]:
    """
    在候选 chunk 中找出跨 project 的相似簇。

    条件：
      - 相似度 >= sim_threshold
      - 来自不同 project（同 project 的相似 chunk 由 consolidate.py 处理）
      - 每个簇至少包含 2 个不同 project 的 chunk

    OS 类比：ksmd 扫描不同进程的匿名页，找内容相同的候选合并。
    """
    n = len(chunks)
    # 并查集：记录每个 chunk 属于哪个簇
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(n):
        for j in range(i + 1, n):
            # 必须来自不同 project
            if chunks[i]["project"] == chunks[j]["project"]:
                continue
            sim = _trigram_similarity(chunks[i]["summary"], chunks[j]["summary"])
            if sim >= sim_threshold:
                union(i, j)

    # 按 root 聚合
    from collections import defaultdict
    clusters_raw = defaultdict(list)
    for i in range(n):
        clusters_raw[find(i)].append(chunks[i])

    # 过滤：只保留跨 project 的簇（>= 2 个不同 project）
    result = []
    for cluster in clusters_raw.values():
        projects_in_cluster = {c["project"] for c in cluster}
        if len(projects_in_cluster) >= 2:
            result.append(cluster)

    return result


def _build_semantic_chunk(cluster: list[dict]) -> dict:
    """
    从跨 project 相似簇构建语义 chunk。

    策略：
      - summary: 取 importance 最高的 chunk 的 summary（最权威的表述）
      - content: 聚合各 project 的 summary（多角度印证）
      - importance: 取簇内平均（不人为膨胀）
      - stability: 取簇内最高值 × 1.1（跨项目印证加固，有上限）
      - source_projects: 记录来源 project 列表（溯源）
    """
    # 按 importance 降序排列
    cluster_sorted = sorted(cluster, key=lambda c: c["importance"], reverse=True)
    primary = cluster_sorted[0]

    # 去重 summary（避免完全相同的条目）
    seen_hashes = set()
    unique_summaries = []
    for c in cluster_sorted:
        h = _summary_hash(c["summary"])
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_summaries.append(f"[{c['project']}] {c['summary']}")

    merged_content = "\n".join(unique_summaries)[:2000]
    avg_importance = sum(c["importance"] for c in cluster) / len(cluster)
    max_stability = min(200.0, max(c["stability"] for c in cluster) * 1.1)
    source_projects = sorted({c["project"] for c in cluster})

    now_iso = datetime.now(timezone.utc).isoformat()
    return {
        "id": "sem_" + uuid.uuid4().hex[:16],
        "project": SEMANTIC_PROJECT,
        "source_session": "semantic_consolidator",
        "chunk_type": "semantic_memory",
        "summary": primary["summary"],
        "content": merged_content,
        "importance": round(avg_importance, 4),
        "stability": round(max_stability, 2),
        "retrievability": 0.8,
        "info_class": "semantic",
        "tags": json.dumps(source_projects),
        "access_count": sum(c["access_count"] for c in cluster),
        "oom_adj": -100,   # 语义记忆有更高保留优先级
        "created_at": now_iso,
        "updated_at": now_iso,
        "last_accessed": now_iso,
        "feishu_url": None,
        "lru_gen": 0,
        "raw_snippet": merged_content[:500],
        "encode_context": json.dumps({"source_projects": source_projects,
                                       "cluster_size": len(cluster)}),
        "session_type_history": "",
    }


def _upsert_semantic_chunk(conn: sqlite3.Connection, chunk: dict,
                            existing_hashes: dict[str, str],
                            dry_run: bool) -> str:
    """
    写入或更新 semantic chunk。
    如果 summary hash 已存在则更新 content/stability，否则新建。
    返回 "created" / "updated" / "skipped"
    """
    h = _summary_hash(chunk["summary"])

    if h in existing_hashes:
        existing_id = existing_hashes[h]
        if not dry_run:
            conn.execute("""
                UPDATE memory_chunks
                SET content=?, stability=?, importance=?, access_count=?,
                    updated_at=?, tags=?, encode_context=?
                WHERE id=?
            """, (chunk["content"], chunk["stability"], chunk["importance"],
                  chunk["access_count"], chunk["updated_at"],
                  chunk["tags"], chunk["encode_context"], existing_id))
        return "updated"

    if not dry_run:
        conn.execute("""
            INSERT OR REPLACE INTO memory_chunks
            (id, project, source_session, chunk_type, summary, content,
             importance, stability, retrievability, info_class, tags,
             access_count, oom_adj, created_at, updated_at, last_accessed,
             feishu_url, lru_gen, raw_snippet, encode_context, session_type_history)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            chunk["id"], chunk["project"], chunk["source_session"],
            chunk["chunk_type"], chunk["summary"], chunk["content"],
            chunk["importance"], chunk["stability"], chunk["retrievability"],
            chunk["info_class"], chunk["tags"], chunk["access_count"],
            chunk["oom_adj"], chunk["created_at"], chunk["updated_at"],
            chunk["last_accessed"], chunk["feishu_url"], chunk["lru_gen"],
            chunk["raw_snippet"], chunk["encode_context"],
            chunk["session_type_history"],
        ))
        # 同步写入 FTS5（复用 store_vfs 的 insert_chunk_fts 如果存在，否则手动）
        try:
            from store_vfs import _insert_fts as _fts
            _fts(conn, chunk["id"], chunk["summary"], chunk["content"] or "")
        except Exception:
            pass  # FTS5 写入失败不阻塞主流程

    return "created"


def run_consolidation(conn: sqlite3.Connection,
                      sim_threshold: float = 0.55,
                      min_importance: float = 0.65,
                      dry_run: bool = False) -> dict:
    """
    执行一次跨项目语义巩固。

    sim_threshold 故意设低（0.55）：
      trigram Jaccard 对长文本偏保守，0.55 约等于"主题相关"而非"内容相同"。
      这正是我们想要的：同一领域知识在不同 project 的不同表述。
    """
    stats = {"candidates": 0, "clusters": 0, "created": 0, "updated": 0}

    chunks = _load_candidate_chunks(conn, min_importance)
    stats["candidates"] = len(chunks)

    if len(chunks) < 2:
        return stats

    clusters = _find_cross_project_clusters(chunks, sim_threshold)
    stats["clusters"] = len(clusters)

    existing_hashes = _load_existing_semantic_hashes(conn)

    for cluster in clusters:
        semantic_chunk = _build_semantic_chunk(cluster)
        action = _upsert_semantic_chunk(conn, semantic_chunk,
                                         existing_hashes, dry_run)
        if action == "created":
            stats["created"] += 1
            # 更新 hash 缓存防止本轮重复写入
            existing_hashes[_summary_hash(semantic_chunk["summary"])] = semantic_chunk["id"]
        elif action == "updated":
            stats["updated"] += 1

        if dry_run or True:  # 总是打印（方便 cron 日志）
            projects = sorted({c["project"] for c in cluster})
            print(f"  [{action}] sim_cluster={len(cluster)} projects={projects}")
            print(f"    summary: {semantic_chunk['summary'][:80]}")

    if not dry_run:
        conn.commit()

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="memory-os 跨项目语义巩固（海马体 Replay）"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="只分析，不写入")
    parser.add_argument("--threshold", type=float, default=0.55,
                        help="trigram 相似度阈值（默认 0.55）")
    parser.add_argument("--min-importance", type=float, default=0.65,
                        help="候选 chunk 最低 importance（默认 0.65）")
    parser.add_argument("--show-semantic", action="store_true",
                        help="打印现有 __semantic__ 层的 chunk")
    args = parser.parse_args()

    conn = open_db()
    ensure_schema(conn)

    if args.show_semantic:
        rows = conn.execute("""
            SELECT id, summary, importance, stability, tags, updated_at
            FROM memory_chunks WHERE project=?
            ORDER BY importance DESC
        """, (SEMANTIC_PROJECT,)).fetchall()
        print(f"[__semantic__] {len(rows)} chunks:")
        for r in rows:
            print(f"  {r[0][:12]} imp={r[2]:.2f} stab={r[3]:.1f} "
                  f"tags={r[4]} | {r[1][:70]}")
        conn.close()
        return

    print(f"[semantic_consolidator] threshold={args.threshold} "
          f"min_importance={args.min_importance}"
          + (" [DRY RUN]" if args.dry_run else ""))

    stats = run_consolidation(
        conn,
        sim_threshold=args.threshold,
        min_importance=args.min_importance,
        dry_run=args.dry_run,
    )

    print(f"\n[result] candidates={stats['candidates']} clusters={stats['clusters']} "
          f"created={stats['created']} updated={stats['updated']}"
          + (" [DRY RUN]" if args.dry_run else ""))

    conn.close()


if __name__ == "__main__":
    main()
