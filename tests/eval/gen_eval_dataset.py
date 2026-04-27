#!/usr/bin/env python3
"""
gen_eval_dataset.py — 从当前 store.db 生成 eval ground truth 数据集

每次 store.db 变化时重新运行，保持 eval 数据集与数据库同步。
输出：eval_dataset.json（供 eval_retrieval.py 加载）

策略：
  1. 找最大 project（chunks 最多）
  2. 对每个 chunk，用其关键词构造 query
  3. 验证 FTS5 能找到该 chunk（确保 ground truth 可达）
  4. 输出 JSON 格式的测试集
"""

import sys
import json
import re
from pathlib import Path
from datetime import datetime, timezone

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from store import open_db, fts_search
from bm25 import hybrid_tokenize


def extract_query_from_chunk(chunk: dict) -> str:
    """从 chunk 摘要提取最具代表性的查询关键词"""
    summary = chunk.get("summary", "")
    content = chunk.get("content", "")[:200]
    text = summary + " " + content

    # 策略1：[xxx/iterN] Title — Subtitle 格式
    # 例：[memory-os/iter88] OOM Killer V9 — 主动杀死不产出价值的知识
    m = re.match(r'\[[\w\-/]+/iter(\d+)\]\s*(.+?)(?:\s*[—\-–]\s*(.+))?$', summary.strip())
    if m:
        title = m.group(2).strip()
        subtitle = (m.group(3) or "").strip()
        # 取 title 的关键词 + subtitle 开头
        tech = re.findall(r'[A-Za-z][A-Za-z0-9]{2,}|\d+', title)
        cn = re.findall(r'[\u4e00-\u9fff]{2,4}', subtitle)
        q = " ".join(tech[:3]) + (" " + " ".join(cn[:3]) if cn else "")
        if len(q) >= 5:
            return q.strip()

    # 策略2：[tag] 描述 格式
    m2 = re.match(r'\[([^\]]+)\]\s*(.+)', summary.strip())
    if m2:
        tag = m2.group(1).strip()
        rest = m2.group(2).strip()
        # 取 rest 中的关键技术词
        tech = re.findall(r'[A-Za-z][A-Za-z0-9]{2,}', rest)
        cn = re.findall(r'[\u4e00-\u9fff]{2,5}', rest)
        q = " ".join(tech[:3])
        if cn:
            q += " " + " ".join(cn[:2])
        if len(q) >= 5:
            return q.strip()
        # fallback: 取 rest 前 30 字符
        return rest[:30].strip()

    # 策略3：提取英文技术词 + 中文关键词
    tech_words = re.findall(
        r'Recall@\d+|BM25|FTS5|VFS|LRU|TLB|OOM|AIMD|CFS|NUMA|iter\d+|[A-Z][A-Za-z]{2,}\d*',
        text
    )
    cn_words = re.findall(r'[\u4e00-\u9fff]{3,8}', summary)
    if tech_words or cn_words:
        q = " ".join(dict.fromkeys(tech_words[:3]))  # deduplicate
        if cn_words:
            q += " " + " ".join(cn_words[:2])
        return q.strip()

    # 策略4: 直接截取摘要前 35 字符（最安全的 fallback）
    return re.sub(r'\s+', ' ', summary[:35]).strip()


def build_eval_dataset(conn, project: str, target_count: int = 20) -> list:
    """从指定 project 构建 eval 数据集，优先 memory-os 相关 chunks"""
    # 优先取 memory-os 相关 chunks（iter 标记的）
    rows = conn.execute("""
        SELECT id, chunk_type, summary, content, importance, retrievability
        FROM memory_chunks
        WHERE project = ?
          AND (
            summary LIKE '%iter%' OR summary LIKE '%BM25%' OR summary LIKE '%VFS%'
            OR summary LIKE '%memory-os%' OR summary LIKE '%检索%' OR summary LIKE '%recall%'
            OR summary LIKE '%Recall%' OR summary LIKE '%FTS%' OR summary LIKE '%scorer%'
            OR summary LIKE '%extractor%' OR summary LIKE '%retriever%'
            OR summary LIKE '%capabilities%' OR summary LIKE '%decisions%'
            OR summary LIKE '%OOM%' OR summary LIKE '%TLB%' OR summary LIKE '%LRU%'
          )
        ORDER BY importance DESC, retrievability DESC
        LIMIT 80
    """, [project]).fetchall()

    # 如果不够，补充其他 chunks
    if len(rows) < target_count:
        extra = conn.execute("""
            SELECT id, chunk_type, summary, content, importance, retrievability
            FROM memory_chunks
            WHERE project = ?
            ORDER BY importance DESC
            LIMIT 100
        """, [project]).fetchall()
        existing_ids = {r[0] for r in rows}
        for r in extra:
            if r[0] not in existing_ids:
                rows.append(r)

    dataset = []
    used_ids = set()
    skipped = 0

    for row in rows:
        if len(dataset) >= target_count:
            break

        chunk_id = row[0]
        chunk_type = row[1]
        summary = row[2] or ""
        content = row[3] or ""
        importance = row[4] or 0.0

        # 跳过低质量内容
        if len(summary) < 10:
            skipped += 1
            continue

        # 跳过 excluded_path（不应该被检索到）
        if chunk_type in ("excluded_path",):
            skipped += 1
            continue

        # 跳过已使用的 ID
        if chunk_id in used_ids:
            continue

        # 生成 query
        chunk_dict = {"summary": summary, "content": content, "chunk_type": chunk_type}
        query = extract_query_from_chunk(chunk_dict)

        if not query or len(query) < 3:
            skipped += 1
            continue

        # 验证 FTS5 能找到（只保留可达的 ground truth）
        try:
            results = fts_search(conn, query, project, top_k=5)
            result_ids = [r["id"] for r in results]
            if chunk_id not in result_ids:
                # FTS5 找不到 → 降级：检查是否在 top-10
                results10 = fts_search(conn, query, project, top_k=10)
                result_ids10 = [r["id"] for r in results10]
                if chunk_id not in result_ids10:
                    skipped += 1
                    continue
        except Exception:
            skipped += 1
            continue

        # 确定 category
        if re.search(r'[A-Z]{3,}|\biter\d+\b', query):
            category = "exact_tech"
        elif re.search(r'[\u4e00-\u9fff]', query):
            category = "exact_chinese"
        else:
            category = "semantic"

        dataset.append({
            "query": query,
            "description": f"{chunk_type}: {summary[:60]}",
            "expected_chunk_ids": [chunk_id],
            "category": category,
            "importance": importance,
        })
        used_ids.add(chunk_id)

    return dataset


def main():
    conn = open_db()

    # 找最大的相关 project
    rows = conn.execute("""
        SELECT project, COUNT(*) as n
        FROM memory_chunks
        GROUP BY project ORDER BY n DESC
    """).fetchall()

    print("Projects in store.db:")
    for r in rows:
        print(f"  {r[0]}: {r[1]} chunks")

    # 优先选 chunk 数量 >= 30 的项目
    best_project = None
    best_count = 0
    for proj, n in rows:
        if n > best_count:
            best_project = proj
            best_count = n

    print(f"\nSelected project: {best_project} ({best_count} chunks)")

    # 构建数据集
    dataset = build_eval_dataset(conn, best_project, target_count=20)
    print(f"\nGenerated {len(dataset)} test cases")

    # 输出 JSON
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": best_project,
        "chunk_count": best_count,
        "test_count": len(dataset),
        "tests": dataset,
    }

    out_path = _ROOT / "eval_dataset.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"Saved to: {out_path}")

    # 显示样本
    print("\nSample test cases:")
    for t in dataset[:5]:
        print(f"  [{t['category']}] query='{t['query']}'")
        print(f"    expected={t['expected_chunk_ids'][0][:8]} | {t['description'][:55]}")

    conn.close()
    return output


if __name__ == "__main__":
    main()
