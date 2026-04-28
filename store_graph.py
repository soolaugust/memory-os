"""
store_graph.py — 知识图谱关联扩散（iter366）

OS 类比：
  - chunk_edges 表 = /proc/[pid]/maps 的 vm_area_struct 链表 — 描述地址空间中各区域的邻接关系
  - 1-hop 扩散 = 页表 radix tree walk — 从一个虚拟地址出发，找到相邻映射区域
  - BM25 命中 = TLB 精确命中 — 已找到目标页；扩散 = prefetch adjacent pages

人的记忆类比：语义网络激活扩散（Spreading Activation，Collins & Loftus 1975）
  - 人的联想不是关键词检索，而是图遍历
  - 听到"docker"→ 自动想到"上次改端口的决策" → 想到"相关的 nginx 约束"
  - 每次 BM25 命中 = 一个激活节点，扩散邻边补充关联知识

设计约束：
  - 边的推断必须轻量（纯规则 + 共现，不调 LLM）
  - 扩散深度最多 1-hop（防止噪声爆炸）
  - 每次检索最多补充 2 个扩散节点（token budget 保护）
"""
import json
import re
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from store_vfs import _safe_add_column


# ── Relation types ─────────────────────────────────────────────────────────────

class EdgeType:
    CAUSES = "causes"              # A 导致 B（因果）
    REQUIRES = "requires"          # A 依赖 B
    CONTRADICTS = "contradicts"    # A 与 B 矛盾（排除关系）
    RELATED = "related"            # 一般相关（共现）
    SUPERSEDES = "supersedes"      # A 取代 B（新决策覆盖旧决策）
    IMPLEMENTS = "implements"      # A 是 B 的实现
    COOCCURS = "cooccurs"          # 同 session 同 topic 出现（弱关联）


# ── Schema ────────────────────────────────────────────────────────────────────

def ensure_graph_schema(conn: sqlite3.Connection) -> None:
    """幂等 schema 初始化。"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunk_edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            from_id TEXT NOT NULL,
            to_id TEXT NOT NULL,
            relation_type TEXT NOT NULL,
            weight REAL DEFAULT 1.0,       -- 边强度 [0, 1]
            created_at TEXT NOT NULL,
            source TEXT DEFAULT 'rule',    -- 'rule' | 'cooccurrence' | 'manual'
            UNIQUE(from_id, to_id, relation_type)
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_edges_from ON chunk_edges(from_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_edges_to ON chunk_edges(to_id)"
    )
    conn.commit()


# ── Write edges ───────────────────────────────────────────────────────────────

def add_edge(
    conn: sqlite3.Connection,
    from_id: str,
    to_id: str,
    relation_type: str,
    weight: float = 1.0,
    source: str = "rule",
) -> bool:
    """
    添加一条有向边（幂等，重复调用更新 weight 取最大值）。
    返回 True 表示新建，False 表示已存在（更新）。
    """
    ensure_graph_schema(conn)
    existing = conn.execute(
        "SELECT id, weight FROM chunk_edges WHERE from_id=? AND to_id=? AND relation_type=?",
        (from_id, to_id, relation_type)
    ).fetchone()

    now = datetime.now(timezone.utc).isoformat()
    if existing:
        if weight > existing[1]:
            conn.execute(
                "UPDATE chunk_edges SET weight=? WHERE id=?",
                (weight, existing[0])
            )
            conn.commit()
        return False
    else:
        conn.execute("""
            INSERT INTO chunk_edges (from_id, to_id, relation_type, weight, created_at, source)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (from_id, to_id, relation_type, weight, now, source))
        conn.commit()
        return True


def add_cooccurrence_edges(
    conn: sqlite3.Connection,
    chunk_ids: list,
    weight: float = 0.5,
) -> int:
    """
    为同一 session 同时写入的 chunk 建立共现边（双向）。
    OS 类比：同 PID 分配的内存页面具有更高的 NUMA 亲和性。
    """
    ensure_graph_schema(conn)
    if len(chunk_ids) < 2:
        return 0
    count = 0
    for i, a in enumerate(chunk_ids):
        for b in chunk_ids[i + 1:]:
            if add_edge(conn, a, b, EdgeType.COOCCURS, weight, "cooccurrence"):
                count += 1
            if add_edge(conn, b, a, EdgeType.COOCCURS, weight, "cooccurrence"):
                count += 1
    return count


# ── Rule-based edge inference ─────────────────────────────────────────────────

# 因果关联模式（summary 文本中的关键词对）
_CAUSAL_PATTERNS = [
    (r'(.{5,40})\s*(?:导致|造成|引发)\s*(.{5,40})', EdgeType.CAUSES, 0.8),
    (r'(.{5,40})\s*(?:因此|所以)\s*(.{5,40})', EdgeType.CAUSES, 0.7),
]

# 依赖关联模式
_REQUIRES_PATTERNS = [
    (r'(?:需要|依赖|前提)[是为]?\s*(.{5,40})', EdgeType.REQUIRES, 0.75),
    (r'(.{5,40})\s*(?:requires|depends on)\s*(.{5,40})', EdgeType.REQUIRES, 0.75),
]

# 矛盾/覆盖模式
_CONTRADICTS_PATTERNS = [
    (r'(?:不选|放弃|排除)\s*(.{5,40})', EdgeType.CONTRADICTS, 0.9),
    (r'(.{5,40})\s*(?:而非|不用)\s*(.{5,40})', EdgeType.CONTRADICTS, 0.85),
]


def infer_edges_from_summaries(
    conn: sqlite3.Connection,
    chunks: list,
) -> int:
    """
    从 chunk 列表的 summary 文本中推断关联边（纯规则，不调 LLM）。
    chunks: [{"id": ..., "summary": ..., "chunk_type": ...}, ...]
    返回新增边数。
    """
    if len(chunks) < 2:
        return 0
    ensure_graph_schema(conn)
    count = 0

    # 构建 summary → chunk_id 映射（用于关联发现）
    id_by_summary = {c["summary"][:30]: c["id"] for c in chunks if c.get("summary")}

    for c in chunks:
        s = c.get("summary", "")
        if not s:
            continue
        # 决策 supersedes 排除路径
        if c.get("chunk_type") == "decision":
            for other in chunks:
                if other["id"] == c["id"]:
                    continue
                if other.get("chunk_type") == "excluded_path":
                    # 同 session 的决策与排除路径建立 supersedes 关系
                    add_edge(conn, c["id"], other["id"], EdgeType.SUPERSEDES, 0.7, "rule")
                    count += 1

        # 因果链 causes 决策
        if c.get("chunk_type") == "causal_chain":
            for other in chunks:
                if other["id"] == c["id"]:
                    continue
                if other.get("chunk_type") in ("decision", "reasoning_chain"):
                    add_edge(conn, c["id"], other["id"], EdgeType.CAUSES, 0.6, "rule")
                    count += 1

        # 设计约束 requires 相关决策
        if c.get("chunk_type") == "design_constraint":
            for other in chunks:
                if other["id"] == c["id"]:
                    continue
                if other.get("chunk_type") == "decision":
                    add_edge(conn, c["id"], other["id"], EdgeType.REQUIRES, 0.65, "rule")
                    count += 1

    return count


# ── Retrieval: 1-hop expansion ────────────────────────────────────────────────

def expand_with_neighbors(
    conn: sqlite3.Connection,
    seed_chunk_ids: list,
    top_n: int = 2,
    min_weight: float = 0.5,
    exclude_types: Optional[list] = None,
) -> list:
    """
    从 BM25 命中的种子 chunk 出发，做 1-hop 扩散，返回补充 chunk 列表。

    OS 类比：prefetch_page() — 根据当前访问的页地址，预取相邻 VMA 的页面。
    人的联想类比：语义网络扩散 — 从已激活节点沿强关联边扩散到邻节点。

    Args:
        seed_chunk_ids: BM25 已命中的 chunk IDs（不重复返回这些）
        top_n: 最多返回多少个扩散节点（token budget 保护）
        min_weight: 最低边强度阈值
        exclude_types: 不想扩散到的 chunk 类型（如 entity_stub）

    Returns:
        [{"id", "summary", "chunk_type", "importance", "edge_type", "weight"}, ...]
    """
    if not seed_chunk_ids:
        return []
    ensure_graph_schema(conn)

    # 找所有出边和入边的邻居（双向扩散）
    placeholders = ",".join("?" * len(seed_chunk_ids))
    neighbor_rows = conn.execute(f"""
        SELECT DISTINCT
            CASE WHEN from_id IN ({placeholders}) THEN to_id ELSE from_id END AS neighbor_id,
            relation_type,
            weight
        FROM chunk_edges
        WHERE (from_id IN ({placeholders}) OR to_id IN ({placeholders}))
          AND weight >= ?
        ORDER BY weight DESC
    """, (*seed_chunk_ids, *seed_chunk_ids, *seed_chunk_ids, min_weight)).fetchall()

    # 过滤掉种子节点本身
    seed_set = set(seed_chunk_ids)
    candidates = {}
    for neighbor_id, rel_type, weight in neighbor_rows:
        if neighbor_id in seed_set:
            continue
        if neighbor_id not in candidates or weight > candidates[neighbor_id]["weight"]:
            candidates[neighbor_id] = {"edge_type": rel_type, "weight": weight}

    if not candidates:
        return []

    # 排序：weight 降序，取 top_n
    sorted_ids = sorted(candidates, key=lambda x: -candidates[x]["weight"])[:top_n * 3]

    # 获取 chunk 详情
    id_placeholders = ",".join("?" * len(sorted_ids))
    chunk_rows = conn.execute(f"""
        SELECT id, summary, chunk_type, importance
        FROM memory_chunks
        WHERE id IN ({id_placeholders})
    """, sorted_ids).fetchall()

    results = []
    for row in chunk_rows:
        cid, summary, chunk_type, importance = row
        if exclude_types and chunk_type in exclude_types:
            continue
        meta = candidates.get(cid, {})
        results.append({
            "id": cid,
            "summary": summary,
            "chunk_type": chunk_type,
            "importance": importance,
            "edge_type": meta.get("edge_type", EdgeType.RELATED),
            "weight": meta.get("weight", 0.5),
        })
        if len(results) >= top_n:
            break

    return results


# ── Stats ─────────────────────────────────────────────────────────────────────

def graph_stats(conn: sqlite3.Connection) -> dict:
    """返回知识图谱的基本统计。"""
    ensure_graph_schema(conn)
    total = conn.execute("SELECT COUNT(*) FROM chunk_edges").fetchone()[0]
    by_type = conn.execute(
        "SELECT relation_type, COUNT(*) FROM chunk_edges GROUP BY relation_type"
    ).fetchall()
    return {
        "total_edges": total,
        "by_type": {r[0]: r[1] for r in by_type},
    }
