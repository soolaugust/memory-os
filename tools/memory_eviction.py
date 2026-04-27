#!/usr/bin/env python3
"""
memory-os ARC 语义淘汰 — ARC + Access Frequency

OS 类比：
- LRU（迭代1-10）：按 importance × recency 淘汰
- ARC（迭代11）：加入 uniqueness 维度，语义冗余的 chunk 优先淘汰
- ARC v2（迭代19b）：加入 access_frequency 维度，高频访问 chunk 获得保护

原理：
  retention_score = importance × 0.4 + recency × 0.15 + uniqueness × 0.25 + access_freq × 0.2
  access_freq = min(1.0, log2(1 + access_count) / 4)

  uniqueness = 1 - max_similarity_to_same_type_chunks
  （同类型 chunk 间 BM25 相似度越高，uniqueness 越低，越容易被淘汰）

触发条件：
- chunk_count > MAX_CHUNKS (默认 50)

驱逐策略：
- 驱逐至 TARGET_CHUNKS (默认 40)，即驱逐约 20%
- 永远不驱逐 importance >= 0.9 的 chunk（核心决策，受保护）
- 永远不驱逐最近 7 天写入的 chunk（新鲜知识）
- 冗余组内保留 importance 最高的一条

用法：
  python3 memory_eviction.py            # 检查并驱逐
  python3 memory_eviction.py --dry      # 只打印，不写
  python3 memory_eviction.py --status   # 只显示当前状态（含 uniqueness）
"""
import sys
import re
import math
import sqlite3
import shutil
from datetime import datetime, timezone
from pathlib import Path

# Unified Scorer（迭代20）
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
from scorer import retention_score as _unified_retention_score
from scorer import recency_score as _unified_recency_score
from scorer import access_frequency as _unified_access_freq
from store import open_db, ensure_schema, delete_chunks as store_delete_chunks, get_chunk_count

DB = Path.home() / ".claude" / "memory-os" / "store.db"
BAK = DB.parent / "store.db.eviction.bak"
MAX_CHUNKS = 50
TARGET_CHUNKS = 40
IMPORTANCE_FLOOR = 0.9
RECENCY_PROTECT_DAYS = 7


# ── tokenization（与 retriever.py 共用逻辑）─────────────────────

def _tokenize(text: str) -> list:
    tokens = []
    for m in re.finditer(r'[a-zA-Z0-9_][-a-zA-Z0-9_.]*', text):
        tokens.append(m.group().lower())
    chinese = re.sub(r'[^\u4e00-\u9fff]', '', text)
    for i in range(len(chinese) - 1):
        tokens.append(chinese[i:i + 2])
    return tokens


# ── pairwise BM25 similarity ────────────────────────────────────

def _pairwise_similarity(docs: list) -> list:
    """
    计算每个文档与同组其他文档的最大 BM25 相似度。
    返回与 docs 等长的 float list（0-1），值越高说明越冗余。
    """
    if len(docs) <= 1:
        return [0.0] * len(docs)

    tokenized = [_tokenize(d) for d in docs]
    dl = [len(t) for t in tokenized]
    avg_dl = sum(dl) / len(dl) if dl else 1.0
    N = len(docs)

    df = {}
    for td in tokenized:
        for t in set(td):
            df[t] = df.get(t, 0) + 1

    max_sims = []
    k1, b = 1.5, 0.75
    for qi in range(N):
        query_tokens = tokenized[qi]
        if not query_tokens:
            max_sims.append(0.0)
            continue
        best = 0.0
        for di in range(N):
            if di == qi:
                continue
            tf_map = {}
            for t in tokenized[di]:
                tf_map[t] = tf_map.get(t, 0) + 1
            score = 0.0
            for qt in query_tokens:
                if qt not in df:
                    continue
                tf = tf_map.get(qt, 0)
                idf = max(0.0, math.log((N - df[qt] + 0.5) / (df[qt] + 0.5)))
                num = tf * (k1 + 1)
                den = tf + k1 * (1 - b + b * dl[di] / avg_dl)
                score += idf * (num / den if den else 0.0)
            if score > best:
                best = score
        max_sims.append(best)

    mx = max(max_sims) if max_sims else 0.0
    if mx > 0:
        max_sims = [s / mx for s in max_sims]
    return max_sims


# ── scoring（迭代20: 已迁移至 scorer.py Unified Scorer）──────────


def _is_protected(importance: float, created_at: str) -> bool:
    if importance >= IMPORTANCE_FLOOR:
        return True
    try:
        dt = datetime.fromisoformat(created_at)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - dt).total_seconds() / 86400
        if age_days <= RECENCY_PROTECT_DAYS:
            return True
    except Exception:
        pass
    return False


# ── uniqueness computation ───────────────────────────────────────

def _compute_uniqueness(rows: list) -> dict:
    """
    按 chunk_type 分组计算 uniqueness。
    返回 {chunk_id: uniqueness_score}
    """
    type_groups = {}
    for r in rows:
        rid, summary, ctype, *_ = r
        ctype = ctype or "unknown"
        if ctype not in type_groups:
            type_groups[ctype] = []
        type_groups[ctype].append((rid, summary or ""))

    result = {}
    for ctype, group in type_groups.items():
        if len(group) <= 1:
            for item in group:
                result[item[0]] = 1.0
            continue
        ids = [g[0] for g in group]
        docs = [g[1] for g in group]
        sims = _pairwise_similarity(docs)
        for i, rid in enumerate(ids):
            result[rid] = round(1.0 - sims[i], 4)
    return result


# ── main ─────────────────────────────────────────────────────────

def run(dry_run: bool = False, status_only: bool = False):
    if not DB.exists():
        print("store.db 不存在，跳过")
        return

    conn = open_db()
    ensure_schema(conn)
    rows = conn.execute(
        "SELECT id, summary, chunk_type, importance, last_accessed, created_at, "
        "COALESCE(access_count, 0) "
        "FROM memory_chunks ORDER BY created_at"
    ).fetchall()
    conn.close()

    total = len(rows)
    print(f"当前 chunk_count: {total}  (max={MAX_CHUNKS}, target={TARGET_CHUNKS})")

    uniqueness_map = _compute_uniqueness(rows)

    if status_only:
        scored = []
        for r in rows:
            rid, summary, ctype, imp, la, ca, ac = r
            imp = imp or 0.5
            la = la or ca or ""
            uniq = uniqueness_map.get(rid, 1.0)
            score = _unified_retention_score(imp, la, uniq, ac)
            protected = _is_protected(imp, ca or "")
            scored.append((score, protected, ctype, summary, uniq, ac))
        scored.sort(key=lambda x: x[0])
        print("\n保留分数排行（低→高，前10）：")
        for s, prot, ct, sm, uniq, ac in scored[:10]:
            tag = "[保护]" if prot else "      "
            print(f"  {tag} {s:.4f} uniq={uniq:.2f} ac={ac} [{ct}] {sm[:50]}")
        return

    if total <= MAX_CHUNKS:
        print(f"未超阈值（{total} ≤ {MAX_CHUNKS}），无需驱逐")
        return

    n_to_evict = total - TARGET_CHUNKS
    if n_to_evict <= 0:
        print("无需驱逐")
        return

    candidates = []
    protected_ids = set()
    for r in rows:
        rid, summary, ctype, imp, la, ca, ac = r
        imp = imp or 0.5
        la = la or ca or ""
        ca = ca or ""
        uniq = uniqueness_map.get(rid, 1.0)
        score = _unified_retention_score(imp, la, uniq, ac)
        if _is_protected(imp, ca):
            protected_ids.add(rid)
        else:
            candidates.append((score, rid, ctype, summary, uniq, ac))

    candidates.sort(key=lambda x: x[0])
    to_evict = candidates[:n_to_evict]

    print(f"\n计划驱逐 {len(to_evict)} 个 chunk（受保护={len(protected_ids)}）：")
    for s, rid, ct, sm, uniq, ac in to_evict:
        print(f"  {s:.4f} uniq={uniq:.2f} ac={ac} [{ct}] {sm[:50]}")

    if dry_run:
        print("\n[dry-run] 未实际写入")
        return

    shutil.copy2(DB, BAK)

    evict_ids = [rid for _, rid, _, _, _, _ in to_evict]
    conn = open_db()
    ensure_schema(conn)
    deleted = store_delete_chunks(conn, evict_ids)
    conn.commit()
    remaining = get_chunk_count(conn)
    conn.close()

    print(f"\n驱逐完成：删除 {deleted} 个，剩余 {remaining} 个")
    print(f"备份已写入: {BAK}")


if __name__ == "__main__":
    dry = "--dry" in sys.argv
    status = "--status" in sys.argv
    run(dry_run=dry, status_only=status)
