#!/usr/bin/env python3
"""
tools/citation_detector.py — 使用反馈信号：Citation Detection + Importance Online 更新

OS 类比：CPU performance counters + branch misprediction feedback
  - 现代 CPU 通过 PMU（Performance Monitoring Unit）追踪 branch 预测是否命中
  - 命中 → 更新 branch predictor history（强化这个预测路径）
  - 未命中 → 弱化这个预测路径

  memory-os 等价：
  - 检索器注入了 N 个 chunk（相当于做了 N 次"预测：这条记忆此刻有用"）
  - Claude 回复实际引用了其中 K 个（K 次预测命中）
  - 命中的 chunk: importance 微增（+0.02，有用信号）
  - 未命中的 chunk: importance 微减（-0.01，预测过度注入信号）
  - 同时联动更新关联的 __semantic__ 层 chunk

设计原则（来自 chaos-governance + 人类记忆研究）：
  - 小步更新（±0.02）：避免单次反馈导致importance剧烈震荡
  - 信噪比优先：只有 trigram overlap ≥ 阈值才算引用（过滤噪声）
  - 级联到语义层：semantic chunk的importance随源chunk同向移动（衰减系数0.5）
  - 上下界保护：importance 钳制在 [0.05, 0.99]

调用方式：
  python3 tools/citation_detector.py  ← 从 stdin 读取 Stop hook JSON
  run_citation_detection(reply_text, project, session_id)  ← 直接调用

Stop hook input format:
  {
    "last_assistant_message": "...",
    "session_id": "...",
    "transcript_path": "..."
  }
"""
import sys
import os
import re
import json
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from store import open_db, ensure_schema, dmesg_log, DMESG_INFO
from utils import resolve_project_id

_MEMORY_OS_DIR = Path.home() / ".claude" / "memory-os"

# ── 调参常量 ─────────────────────────────────────────────────────────────────
# 小步更新，避免震荡（±0.02 / ±0.01）
CITED_IMPORTANCE_DELTA = +0.02    # 被引用：微增
UNCITED_IMPORTANCE_DELTA = -0.01  # 未被引用：微减（只在连续未引用时生效）
SEMANTIC_CASCADE_FACTOR = 0.5     # 语义层级联衰减系数
MIN_IMPORTANCE = 0.05             # importance 下界
MAX_IMPORTANCE = 0.99             # importance 上界

# Confidence Score 联动更新（iter476/iter485）
# iter485: 引用时 confidence 增强——Retrieval Practice Effect (Roediger & Karpicke 2006)
#   成功检索并使用的知识，其可信度（epistemological confidence）同步提升；
#   +0.05 是"被引用验证"的有意义增强（vs iter476 微增 +0.01 的轻触）。
# OS 类比：Linux page hot boost — 频繁 map 的 page 在 MGLRU 中提升 generation，
#   且通过 madvise(MADV_WILLNEED) 锁入 hot tier，相当于提升 page 可信度评分。
CITED_CONFIDENCE_DELTA = +0.05    # 被引用：confidence 增强（成功检索强化可信度）
UNCITED_CONFIDENCE_DELTA = -0.005 # 未被引用：confidence 轻微减少
MIN_CONFIDENCE = 0.10             # confidence 下界
MAX_CONFIDENCE = 1.00             # confidence 上界

# trigram overlap 阈值：reply文字与 chunk summary/content 的相似度 ≥ 此值 → 视为引用
CITATION_TRIGRAM_THRESHOLD = 0.08  # 故意设低（reply通常是总结/重述，不直接复制原文）

# 最多影响多少个 chunk
MAX_AFFECTED_CHUNKS = 15

# Fast Stale Detection 参数
STALE_CONSEC_THRESHOLD = 3    # 连续未引用次数阈值
STALE_FAST_PENALTY = 0.30     # 快速降级系数：importance *= (1 - penalty)
STALE_LOOKBACK_TRACES = 8     # 向前看多少条 trace 统计连续未引用

# 非知识类 chunk_type — 跳过 citation 检测和 fast stale 降级
# 这些是系统记录（task 状态、对话历史），Claude 回复不会引用其文本内容
# OS 类比：内核页面（kernel page）不参与用户空间 page reclaim
SKIP_CITATION_TYPES = frozenset({
    "task_state", "prompt_context", "conversation_summary",
    "session_summary", "goal",
})


def _trigrams(s: str) -> set:
    """生成 trigram 集合（与 semantic_consolidator 保持一致）。"""
    s = re.sub(r'\s+', ' ', s.strip().lower())
    if len(s) < 3:
        return set(s)
    return {s[i:i+3] for i in range(len(s) - 2)}


def _overlap_score(text: str, chunk_summary: str, chunk_content: str = "") -> float:
    """
    计算 reply text 与 chunk 的 trigram overlap。
    取 summary 和 content 的最大值（content 通常更详细）。
    """
    t_reply = _trigrams(text)
    if not t_reply:
        return 0.0

    def jaccard(a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    t_summary = _trigrams(chunk_summary)
    score = jaccard(t_reply, t_summary)

    if chunk_content and len(chunk_content) < 1000:
        t_content = _trigrams(chunk_content[:500])
        score = max(score, jaccard(t_reply, t_content))

    return score


def _get_recent_recall_traces(conn: sqlite3.Connection, project: str,
                               session_id: str) -> list[dict]:
    """
    获取当前 session 最近 1 条 recall_trace，返回被注入的 chunk 列表。

    精度优化：只取当前 session 最新 1 条（对应当前轮的注入），避免跨 session 污染。
    OS 类比：CPU TLB flush per-process — 进程切换时 TLB 条目只属于当前 ASID，
    不会误用其他进程的地址翻译结果。

    Fast Stale Detection 使用 project 级别全量扫描（见 _check_and_apply_fast_stale），
    职责分离：citation precision（当前轮）vs stale breadth（历史趋势）。
    """
    rows = conn.execute(
        """SELECT id, top_k_json, timestamp FROM recall_traces
           WHERE project=? AND session_id=?
           ORDER BY timestamp DESC LIMIT 1""",
        (project, session_id)
    ).fetchall()

    # fallback: 若当前 session 无记录，尝试最近 1 条无 session 隔离的 trace
    # 兼容老版本 trace（session_id 为空）
    if not rows:
        rows = conn.execute(
            """SELECT id, top_k_json, timestamp FROM recall_traces
               WHERE project=?
                 AND (session_id IS NULL OR session_id='')
               ORDER BY timestamp DESC LIMIT 1""",
            (project,)
        ).fetchall()

    all_chunks = []
    seen_ids = set()
    for trace_id, top_k_json, ts in rows:
        if not top_k_json:
            continue
        try:
            items = json.loads(top_k_json) if isinstance(top_k_json, str) else top_k_json
            for item in (items or []):
                cid = item.get("id") if isinstance(item, dict) else item
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    all_chunks.append({
                        "id": cid,
                        "summary": item.get("summary", "") if isinstance(item, dict) else "",
                        "content": item.get("content", "") if isinstance(item, dict) else "",
                        "trace_id": trace_id,
                    })
        except (json.JSONDecodeError, TypeError):
            continue

    return all_chunks[:MAX_AFFECTED_CHUNKS]


def _enrich_chunks_from_db(conn: sqlite3.Connection,
                            chunks: list[dict]) -> list[dict]:
    """
    补全 chunk 的 summary/content（recall_trace 可能只存了 id）。
    """
    ids_missing = [c["id"] for c in chunks if not c.get("summary")]
    if not ids_missing:
        return chunks

    placeholders = ",".join("?" * len(ids_missing))
    rows = conn.execute(
        f"SELECT id, summary, content, importance FROM memory_chunks WHERE id IN ({placeholders})",
        ids_missing
    ).fetchall()
    db_map = {r[0]: {"summary": r[1] or "", "content": r[2] or "", "importance": r[3]} for r in rows}

    result = []
    for c in chunks:
        if not c.get("summary") and c["id"] in db_map:
            c = {**c, **db_map[c["id"]]}
        result.append(c)
    return result


def _update_chunk_importance(conn: sqlite3.Connection, chunk_id: str,
                              delta: float, reason: str) -> float | None:
    """
    原子更新单个 chunk 的 importance，返回新值。
    钳制在 [MIN_IMPORTANCE, MAX_IMPORTANCE]。
    """
    row = conn.execute(
        "SELECT importance FROM memory_chunks WHERE id=?", (chunk_id,)
    ).fetchone()
    if not row:
        return None

    old_imp = row[0] or 0.5
    new_imp = max(MIN_IMPORTANCE, min(MAX_IMPORTANCE, old_imp + delta))

    if abs(new_imp - old_imp) < 0.001:  # 已在边界，无需更新
        return new_imp

    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "UPDATE memory_chunks SET importance=?, updated_at=? WHERE id=?",
        (new_imp, now, chunk_id)
    )
    return new_imp


def _update_chunk_confidence(conn: sqlite3.Connection, chunk_id: str,
                              delta: float) -> float | None:
    """
    原子更新单个 chunk 的 confidence_score（iter476）。
    钳制在 [MIN_CONFIDENCE, MAX_CONFIDENCE]。

    心理学：Source monitoring framework (Johnson 1993) — 被引用的记忆
    获得正向验证（episodic tagging），未被引用的轻微减弱可信度。
    OS 类比：Linux ECC memory — 被频繁访问且无错误的 page → ECC status=clean；
      长期未访问的 page → ECC status=unknown（需重新验证）。
    """
    row = conn.execute(
        "SELECT confidence_score FROM memory_chunks WHERE id=?", (chunk_id,)
    ).fetchone()
    if not row:
        return None

    old_conf = float(row[0] or 0.7)
    new_conf = max(MIN_CONFIDENCE, min(MAX_CONFIDENCE, old_conf + delta))

    if abs(new_conf - old_conf) < 0.001:
        return new_conf

    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "UPDATE memory_chunks SET confidence_score=?, updated_at=? WHERE id=?",
        (new_conf, now, chunk_id)
    )
    return new_conf


def _cascade_to_semantic(conn: sqlite3.Connection, source_project: str,
                          delta: float) -> int:
    """
    将 importance 变化级联到包含 source_project 的 __semantic__ chunks。
    OS 类比：shared page reference count update — 语义记忆被多个 project 引用，
    任一 project 的引用变化都影响共享页的热度。
    """
    # 找到 tags 中包含 source_project 的 semantic chunk
    rows = conn.execute(
        """SELECT id, importance, tags FROM memory_chunks
           WHERE project='__semantic__' AND chunk_type='semantic_memory'
           AND tags LIKE ?""",
        (f'%{source_project}%',)
    ).fetchall()

    updated = 0
    cascaded_delta = delta * SEMANTIC_CASCADE_FACTOR
    now = datetime.now(timezone.utc).isoformat()

    for sem_id, sem_imp, tags in rows:
        # 再次确认 tags 确实包含 source_project（防止 substring match 误判）
        try:
            tag_list = json.loads(tags or "[]")
            if source_project not in tag_list:
                continue
        except (json.JSONDecodeError, TypeError):
            continue

        new_imp = max(MIN_IMPORTANCE, min(MAX_IMPORTANCE,
                                          (sem_imp or 0.5) + cascaded_delta))
        if abs(new_imp - (sem_imp or 0.5)) < 0.001:
            continue
        conn.execute(
            "UPDATE memory_chunks SET importance=?, updated_at=? WHERE id=?",
            (new_imp, now, sem_id)
        )
        updated += 1

    return updated


def _check_and_apply_fast_stale(conn: sqlite3.Connection,
                                uncited_ids: list[str], project: str) -> int:
    """向后兼容包装，内部调用 _check_and_apply_fast_stale_with_ids。"""
    _, count = _check_and_apply_fast_stale_with_ids(conn, uncited_ids, project)
    return count


def _check_and_apply_fast_stale_with_ids(conn: sqlite3.Connection,
                                          uncited_ids: list[str],
                                          project: str) -> tuple[set, int]:
    """
    Fast Stale Detection — 连续被检索但从未被引用 → 快速降级 importance。

    OS 类比：Linux DAMON dead_region detection — 反复未访问的内存区域
    主动标记为冷页（MADV_COLD），不等 kswapd 慢速时钟滴答。

    心理学对应：记忆的"检索诱导遗忘"(Retrieval-Induced Forgetting, RIF) —
    被激活但未能进入工作记忆的项目，随后的检索概率下降（Anderson 1994）。

    扫描策略：project 级别全量扫描最近 STALE_LOOKBACK_TRACES 条 recall_trace，
    统计任意 chunk_id 在最近 traces 中的连续出现次数（不限于当前 session）。
    uncited_ids 参数用于优先检查已知未引用的 chunk；此外对 project 内所有
    在最近 traces 中高频出现的 chunk 也做检测（跨 session 的 stale 信号）。
    超过 STALE_CONSEC_THRESHOLD → importance × (1 - STALE_FAST_PENALTY)。
    """
    degraded = 0
    degraded_ids: set[str] = set()
    now = datetime.now(timezone.utc).isoformat()

    # 获取最近 STALE_LOOKBACK_TRACES 条 recall_trace 的 top_k_json（project 级别）
    recent_traces = conn.execute(
        """SELECT top_k_json FROM recall_traces
           WHERE project=?
           ORDER BY timestamp DESC LIMIT ?""",
        (project, STALE_LOOKBACK_TRACES)
    ).fetchall()

    if len(recent_traces) < STALE_CONSEC_THRESHOLD:
        return set(), 0  # 数据不足，不做判断

    # 收集候选 IDs：优先 uncited_ids，同时扫描最近 traces 中出现的所有 chunk
    candidate_ids = set(uncited_ids or [])
    for (top_k_json,) in recent_traces:
        if not top_k_json:
            continue
        try:
            items = json.loads(top_k_json) if isinstance(top_k_json, str) else top_k_json
            for item in (items or []):
                if isinstance(item, dict):
                    cid = item.get("id", "")
                    if cid:
                        candidate_ids.add(cid)
                elif isinstance(item, str) and item:
                    candidate_ids.add(item)
        except (json.JSONDecodeError, TypeError):
            continue

    if not candidate_ids:
        return set(), 0

    # 统计每个 candidate_id 在最近 traces 中的连续出现次数
    for cid in candidate_ids:
        consec_count = 0
        for (top_k_json,) in recent_traces:
            if not top_k_json:
                break
            try:
                items = json.loads(top_k_json) if isinstance(top_k_json, str) else top_k_json
                ids_in_trace = set()
                for item in (items or []):
                    if isinstance(item, dict):
                        ids_in_trace.add(item.get("id", ""))
                    elif isinstance(item, str):
                        ids_in_trace.add(item)
                if cid in ids_in_trace:
                    consec_count += 1
                else:
                    break  # 不连续，停止计数
            except (json.JSONDecodeError, TypeError):
                break

        # iter483: per-type stale threshold — design_constraint 需要更多连续 stale 信号
        # 心理学：Schema Consistency Effect (Brewer & Nakamura 1984) — 与 schema 一致的知识
        #   更难被遗忘；设计约束是 schema-level 知识，应对偶发未引用有更强抵抗力。
        # OS 类比：Linux huge page (THP) 的 shrinker — THP 的 reclaim 开销远高于普通页，
        #   需要更大的内存压力才触发 THP 拆分；类似地，设计约束需要更多 stale 信号才降级。
        row = conn.execute(
            "SELECT importance, chunk_type, stability, updated_at, last_accessed"
            " FROM memory_chunks WHERE id=?", (cid,)
        ).fetchone()
        if row is None or row[0] is None:
            continue
        _ctype = row[1] or ""
        # 跳过非知识类 chunk（系统记录不应被遗忘机制惩罚）
        if _ctype in SKIP_CITATION_TYPES:
            continue
        # 决定该 chunk_type 的阈值
        _stale_threshold = (STALE_CONSEC_THRESHOLD * 2
                            if _ctype == "design_constraint"
                            else STALE_CONSEC_THRESHOLD)
        if consec_count >= _stale_threshold:
            # 连续 N 次出现在检索结果但未被引用 → 快速降级
            old_imp = row[0]
            chunk_stability = float(row[2] or 0.0)
            chunk_updated_at = row[3] or ""
            chunk_last_accessed = row[4] or ""

            # iter477: Ebbinghaus 协调 — 若 chunk 在过去 24h 内已被 Ebbinghaus 衰减，
            # 跳过 fast_stale 的 importance 惩罚（避免双重遗忘惩罚）。
            # 检测条件：updated_at 在过去 24h 内（说明最近有写入）
            #           AND last_accessed 超过 24h 前（说明写入不是因为访问）
            # → 最近的 updated_at 只能是 Ebbinghaus decay 或其他 DAMON 操作导致
            # OS 类比：Linux kswapd 发现 page 已被 DAMON 标记为 COLD → 跳过本轮 direct reclaim
            _skip_for_ebbinghaus = False
            try:
                _ebbinghaus_cutoff_24h = (
                    datetime.now(timezone.utc) - timedelta(hours=24)
                ).isoformat()
                # updated_at 在 24h 内（说明近期有写入）
                _recently_updated = chunk_updated_at > _ebbinghaus_cutoff_24h
                # last_accessed 超过 24h 前（说明访问不在近期）
                _long_since_access = (chunk_last_accessed and
                                      chunk_last_accessed < _ebbinghaus_cutoff_24h)
                if _recently_updated and _long_since_access:
                    _skip_for_ebbinghaus = True
            except Exception:
                pass

            if _skip_for_ebbinghaus:
                # Ebbinghaus 已处理，只做 degraded_ids 记录（SM-2 quality=1 还会执行）
                # 不执行 importance 惩罚
                degraded_ids.add(cid)
                degraded += 1
                continue

            # Stability-aware penalty scaling（iter473）：
            # 高 stability chunk（历史频繁被引用）对偶发 stale 信号更具抵抗力。
            # actual_penalty = STALE_FAST_PENALTY × max(0.3, 1 - stability / 20)
            # stability=0  → penalty=0.30（全额）
            # stability=10 → penalty=0.15（半额）
            # stability=20+ → penalty=0.09（最小 30% of base）
            # OS 类比：Linux MGLRU — page 在高代（generation=0）比低代更难被驱逐；
            #   高 stability 的 chunk 相当于在 gen=0 的 hot page，需要更强的
            #   reclaim pressure 才能降代。
            _stability_scale = max(0.3, 1.0 - chunk_stability / 20.0)
            actual_penalty = STALE_FAST_PENALTY * _stability_scale

            new_imp = max(MIN_IMPORTANCE,
                          old_imp * (1.0 - actual_penalty))
            if new_imp < old_imp - 0.01:  # 变化量足够才更新
                conn.execute(
                    "UPDATE memory_chunks SET importance=?, updated_at=? WHERE id=?",
                    (new_imp, now, cid)
                )
                degraded += 1
                degraded_ids.add(cid)

    return degraded_ids, degraded


def run_citation_detection(reply_text: str, project: str,
                            session_id: str, conn=None) -> dict:
    """
    核心入口：执行 citation detection + importance online 更新。

    Args:
        reply_text: Claude 的最后一条 assistant 回复文本
        project: 当前项目 ID
        session_id: 当前 session ID
        conn: 可选的 DB 连接（传入则复用，否则自建）

    Returns:
        stats dict: {cited, uncited, semantic_updated, skipped}
    """
    stats = {"cited": 0, "uncited": 0, "semantic_updated": 0, "skipped": 0, "spreading_activated": 0,
             "fast_stale_degraded": 0}

    if not reply_text or len(reply_text) < 20:
        return stats

    owns_conn = conn is None
    if owns_conn:
        try:
            conn = open_db()
            ensure_schema(conn)
        except Exception:
            return stats

    try:
        # 0. Adaptive Citation Threshold — 根据历史 citation rate 动态调整阈值
        # OS 类比：Linux page cache 的 readahead 根据命中率调整 ra_pages：
        #   命中率低 → 缩小 readahead（减少无效预取）→ 阈值提高（减少误判）
        #   命中率高 → 扩大 readahead（更激进预取）→ 阈值降低（捕捉更多引用）
        # 心理学：感知精确度 vs 召回率的 ROC 权衡 — 噪声环境下需要更严格的判断阈值
        _prior_rate = get_citation_rate(project)
        if _prior_rate < 0.30:
            # 低命中率：提高阈值，减少误判（精确度优先）
            effective_threshold = min(0.15, CITATION_TRIGRAM_THRESHOLD * 1.5)
        elif _prior_rate > 0.65:
            # 高命中率：降低阈值，捕捉更多引用（召回率优先）
            effective_threshold = max(0.05, CITATION_TRIGRAM_THRESHOLD * 0.75)
        else:
            effective_threshold = CITATION_TRIGRAM_THRESHOLD

        # 1. 获取最近注入的 chunk 列表
        retrieved_chunks = _get_recent_recall_traces(conn, project, session_id)

        cited_ids = []
        uncited_ids = []
        chunk_projects = {}

        if not retrieved_chunks:
            stats["skipped"] = 1
            # Fast Stale Detection 是 project 级别扫描，即使当前 session 无 recall_trace
            # 也要运行（跨 session 的历史 stale 信号）
            # OS 类比：DAMON 扫描整个 address space，不限于当前 syscall 的内存区域
            _fast_stale_count = _check_and_apply_fast_stale(conn, [], project)
            if _fast_stale_count > 0:
                stats["fast_stale_degraded"] = _fast_stale_count
                if owns_conn:
                    conn.commit()
            return stats

        # 2. 补全 summary/content（recall_trace 可能只有 id）
        retrieved_chunks = _enrich_chunks_from_db(conn, retrieved_chunks)

        # 3. 对每个 chunk 计算 citation score
        # 查询 chunk 所属 project 和 chunk_type（用于 semantic 级联 + 类型过滤）
        cids = [c["id"] for c in retrieved_chunks]
        chunk_types = {}  # chunk_id → chunk_type
        if cids:
            placeholders = ",".join("?" * len(cids))
            proj_rows = conn.execute(
                f"SELECT id, project, chunk_type FROM memory_chunks WHERE id IN ({placeholders})",
                cids
            ).fetchall()
            chunk_projects = {r[0]: r[1] for r in proj_rows}
            chunk_types = {r[0]: (r[2] or "") for r in proj_rows}

        for chunk in retrieved_chunks:
            cid = chunk["id"]
            summary = chunk.get("summary", "")
            content = chunk.get("content", "")

            # 跳过非知识类 chunk（系统记录，Claude 回复不会引用其文本）
            # OS 类比：内核保留页（kernel reserved pages）不参与用户空间 page reclaim
            ctype = chunk_types.get(cid) or chunk.get("chunk_type", "")
            if ctype in SKIP_CITATION_TYPES:
                stats["skipped"] += 1
                continue

            if not summary:
                stats["skipped"] += 1
                continue

            score = _overlap_score(reply_text, summary, content)

            if score >= effective_threshold:
                cited_ids.append(cid)
            else:
                uncited_ids.append(cid)

        # 4. 更新 importance
        project_deltas = {}  # project → net delta（用于 semantic 级联）

        for cid in cited_ids:
            new_imp = _update_chunk_importance(conn, cid, CITED_IMPORTANCE_DELTA,
                                               "citation_detected")
            if new_imp is not None:
                stats["cited"] += 1
                p = chunk_projects.get(cid, project)
                project_deltas[p] = project_deltas.get(p, 0) + CITED_IMPORTANCE_DELTA
            # iter476: cited → confidence_score 微增（被引用 = 知识被验证）
            _update_chunk_confidence(conn, cid, CITED_CONFIDENCE_DELTA)

        # 4.1 SM-2 quality=5 for cited chunks — 完美引用 → stability 强化
        # 心理学：间隔重复质量信号 (Wozniak 1987) — 引用即为"成功回忆"，
        # S_new = S_old × (1 + 0.1 × (5 - 3)) = S_old × 1.2
        # OS 类比：MMU Accessed bit 置位后 kswapd 晋升 page generation（MGLRU）
        if cited_ids:
            try:
                from store_vfs import update_accessed as _update_accessed
                # _sm2_only=True: 只执行 SM-2 stability 更新，跳过 IOR/PEME/spacing 等
                # 二次效应（避免与 citation importance 更新的语义冲突）
                _update_accessed(conn, cited_ids, recall_quality=5, _sm2_only=True)
            except Exception:
                pass

        # 4.5 Fast Stale Detection — 先于 uncited 微减执行，避免双重惩罚
        # OS 类比：Linux DAMON dead_region → madvise(MADV_COLD) — 反复未访问的页面
        # 主动标记为冷页，不等 kswapd 慢速扫描。
        # 心理学对应：记忆的"检索失败"信号 — 反复呈现但从不激活 → 快速遗忘
        #
        # 优先级协调：fast stale 惩罚（×0.7 ≈ -30%）>> uncited 微减（-0.01 ≈ -1%）
        # 若 chunk 已触发 fast stale，跳过其 uncited 微减（更大的惩罚已包含更小的）
        # OS 类比：Linux OOM killer 选择进程时 oom_score_adj 高的优先 — 不同惩罚机制
        # 有明确优先级，不允许同时叠加（避免 thundering herd）。
        stale_degraded_ids, _fast_stale_count = _check_and_apply_fast_stale_with_ids(
            conn, uncited_ids, project
        )
        stats["fast_stale_degraded"] = _fast_stale_count

        # 4.6 SM-2 quality=1 for stale-degraded chunks — 连续检索失败 → stability 惩罚
        # 心理学：retrieval failure signal — 反复呈现但未能进入工作记忆
        # S_new = S_old × (1 + 0.1 × (1 - 3)) = S_old × 0.8（稳定性降低）
        # OS 类比：DAMON dead_region → page demoted to cold tier（降代惩罚）
        if stale_degraded_ids:
            try:
                from store_vfs import update_accessed as _update_accessed
                # _sm2_only=True: 只执行 SM-2 stability 更新，跳过二次效应
                _update_accessed(conn, list(stale_degraded_ids), recall_quality=1,
                                 _sm2_only=True)
            except Exception:
                pass

        _uncited_sm2_ids = []  # 收集需要 SM-2 quality=2 的 uncited IDs
        for cid in uncited_ids:
            # 跳过已被 fast stale 处理的 chunk（避免双重惩罚）
            if cid in stale_degraded_ids:
                p = chunk_projects.get(cid, project)
                project_deltas[p] = project_deltas.get(p, 0) + UNCITED_IMPORTANCE_DELTA
                stats["uncited"] += 1  # 仍计入 uncited 统计（用于 citation rate 计算）
                continue
            new_imp = _update_chunk_importance(conn, cid, UNCITED_IMPORTANCE_DELTA,
                                               "citation_absent")
            if new_imp is not None:
                stats["uncited"] += 1
                p = chunk_projects.get(cid, project)
                project_deltas[p] = project_deltas.get(p, 0) + UNCITED_IMPORTANCE_DELTA
                _uncited_sm2_ids.append(cid)
            # iter476: uncited → confidence_score 轻微减少
            _update_chunk_confidence(conn, cid, UNCITED_CONFIDENCE_DELTA)

        # 4.7 SM-2 quality=2 for uncited chunks（轻微稳定性惩罚）
        # 心理学：被激活但未被有效编码的记忆轻微减弱稳定性 (Anderson 1994 RIF)
        # S_new = S_old × max(0.7, 1 + 0.1 × (2 - 3)) = S_old × 0.9（轻微减弱）
        # 区别：fast_stale quality=1 (×0.8)，uncited quality=2 (×0.9)，引用 quality=5 (×1.2)
        # OS 类比：Linux MGLRU — 访问过但分数低的页面降半代（不是降整代）
        if _uncited_sm2_ids:
            try:
                from store_vfs import update_accessed as _update_accessed
                _update_accessed(conn, _uncited_sm2_ids, recall_quality=2, _sm2_only=True)
            except Exception:
                pass

        # 5. 级联到 __semantic__ 层
        for p, net_delta in project_deltas.items():
            if abs(net_delta) > 0.005:  # 变化量足够大才级联
                updated = _cascade_to_semantic(conn, p, net_delta)
                stats["semantic_updated"] += updated

        # 5.5 Spreading Activation — 引用诱发联想促进（RIF Facilitation）
        # 心理学依据：Anderson (1983) ACT-R 激活扩散理论 — 被激活的节点沿网络边
        #   向邻居传播激活量；Collins & Loftus (1975) 语义网络激活扩散模型；
        #   与 RIF（检索诱发遗忘，对竞争者惩罚）对立：对关联者给予促进。
        # OS 类比：Linux page prefetch via readahead — 访问页 P 时预取其相邻页；
        #   chunk_coactivation 表记录"历史上一起被检索的 chunk 对"，
        #   引用 chunk A → 其共激活邻居 B 的 retrievability 微增（更容易被下次检索）。
        # 实现：查 chunk_coactivation，cited_ids 的邻居 retrievability += 0.03，钳制 [0, 1]
        # 上界：最多激活 10 个邻居（避免引发级联广播）
        if cited_ids:
            try:
                _MAX_SPREAD = 10
                _SPREAD_DELTA = 0.03
                _placeholders = ",".join("?" * len(cited_ids))
                _neighbors = conn.execute(
                    f"""SELECT DISTINCT
                          CASE WHEN chunk_a IN ({_placeholders}) THEN chunk_b ELSE chunk_a END AS neighbor_id,
                          MAX(coact_count) AS strength
                        FROM chunk_coactivation
                        WHERE (chunk_a IN ({_placeholders}) OR chunk_b IN ({_placeholders}))
                          AND project=?
                        GROUP BY neighbor_id
                        ORDER BY strength DESC
                        LIMIT ?""",
                    (*cited_ids, *cited_ids, *cited_ids, project, _MAX_SPREAD)
                ).fetchall()
                _spread_now = datetime.now(timezone.utc).isoformat()
                _spread_count = 0
                for _neighbor_id, _strength in _neighbors:
                    if _neighbor_id in set(cited_ids):
                        continue  # 已经是 cited，跳过
                    _nrow = conn.execute(
                        "SELECT retrievability FROM memory_chunks WHERE id=?",
                        (_neighbor_id,)
                    ).fetchone()
                    if not _nrow:
                        continue
                    _old_ret = float(_nrow[0] or 0.5)
                    _new_ret = min(1.0, _old_ret + _SPREAD_DELTA)
                    if _new_ret - _old_ret > 0.001:
                        conn.execute(
                            "UPDATE memory_chunks SET retrievability=?, updated_at=? WHERE id=?",
                            (_new_ret, _spread_now, _neighbor_id)
                        )
                        _spread_count += 1
                if _spread_count > 0:
                    stats["spreading_activated"] = _spread_count
            except Exception:
                pass  # spreading activation 失败不阻塞主流程

        if owns_conn and (stats["cited"] + stats["uncited"] > 0):
            conn.commit()
            dmesg_log(conn, DMESG_INFO, "citation_detector",
                      f"cited={stats['cited']} uncited={stats['uncited']} "
                      f"semantic_updated={stats['semantic_updated']}"
                      f"{(' spread=' + str(stats.get('spreading_activated', 0))) if stats.get('spreading_activated') else ''}",
                      session_id=session_id, project=project)
            conn.commit()

        # 6. 更新 per-project citation stats（供 retriever Adaptive K 使用）
        # OS 类比：CPU branch predictor history register — 记录最近 N 次预测命中率，
        # 用于调整下次预取窗口大小（readahead_max_sectors）。
        # 写文件（不走 DB）：retriever 快速路径只读文件，避免 DB 查询增加延迟。
        total = stats["cited"] + stats["uncited"]
        if total > 0:
            _update_citation_stats(project, stats["cited"], total)

    except Exception as e:
        import traceback
        sys.stderr.write(f"[citation_detector] error: {e}\n{traceback.format_exc()}\n")
    finally:
        if owns_conn and conn:
            conn.close()

    return stats


def _update_citation_stats(project: str, cited: int, total: int) -> None:
    """
    维护 per-project 滚动 citation stats 文件。
    OS 类比：/proc/vmstat 中的 pgfault/pgmajfault — 内核维护的累积计数器，
    用于 readahead 和 swap 预测算法。

    文件格式: {
        "project": "...",
        "window": [0, 1, 1, 0, ...],  # 滑动窗口（1=cited, 0=uncited），最近 WINDOW_SIZE 条
        "citation_rate": 0.6,
        "updated_at": "..."
    }
    """
    WINDOW_SIZE = 20
    MAX_STATS_FILES = 30    # 最多保留 30 个 project 的 stats 文件
    TTL_DAYS = 7            # 7 天未更新的 stats 文件视为过期
    try:
        _MEMORY_OS_DIR.mkdir(parents=True, exist_ok=True)
        # project 名称可能含路径字符，取 hash 作文件名后缀
        proj_safe = project.replace("/", "_").replace(":", "_")[:40]
        stats_file = _MEMORY_OS_DIR / f"citation_stats.{proj_safe}.json"

        # 读取现有 stats
        window = []
        if stats_file.exists():
            try:
                data = json.loads(stats_file.read_text(encoding="utf-8"))
                window = data.get("window", [])
            except Exception:
                window = []

        # 追加本轮结果（1=每个 cited，0=每个 uncited）
        # 以 cited/total 比例追加（避免 total 很大时单次撑满 window）
        if total > 0:
            # 简化：追加一个 [cited/total] 比例值（float），而非逐 chunk 展开
            window.append(round(cited / total, 3))
        window = window[-WINDOW_SIZE:]  # 保留最近 WINDOW_SIZE 条

        citation_rate = sum(window) / len(window) if window else 0.5

        stats_file.write_text(json.dumps({
            "project": project,
            "window": window,
            "citation_rate": round(citation_rate, 3),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }, ensure_ascii=False), encoding="utf-8")

        # TTL 清理：删除过期文件，限制总数上限（防止文件爆炸）
        # OS 类比：tmpwatch / systemd-tmpfiles — 定期清理过期临时文件
        try:
            import time as _time
            cutoff = _time.time() - TTL_DAYS * 86400
            all_stats = sorted(
                _MEMORY_OS_DIR.glob("citation_stats.*.json"),
                key=lambda p: p.stat().st_mtime
            )
            # 删除过期文件
            for p in all_stats:
                if p.stat().st_mtime < cutoff:
                    p.unlink(missing_ok=True)
            # 若仍超上限，删除最旧的（LRU 淘汰）
            all_stats = [p for p in all_stats if p.exists()]
            if len(all_stats) > MAX_STATS_FILES:
                for p in all_stats[:len(all_stats) - MAX_STATS_FILES]:
                    p.unlink(missing_ok=True)
        except Exception:
            pass
    except Exception:
        pass  # 写失败不影响主流程


def get_citation_rate(project: str) -> float:
    """
    读取 per-project citation rate（供 retriever Adaptive K 使用）。
    OS 类比：readahead 读 /proc/vmstat pgcache_hit/pgcache_miss 比率。
    返回 [0, 1] 浮点，默认 0.5（无数据时中立）。
    """
    try:
        proj_safe = project.replace("/", "_").replace(":", "_")[:40]
        stats_file = _MEMORY_OS_DIR / f"citation_stats.{proj_safe}.json"
        if stats_file.exists():
            data = json.loads(stats_file.read_text(encoding="utf-8"))
            return float(data.get("citation_rate", 0.5))
    except Exception:
        pass
    return 0.5


def main():
    """Stop hook 入口：从 stdin 读取 hook input JSON。"""
    try:
        raw = sys.stdin.read()
        hook_input = json.loads(raw) if raw.strip() else {}
    except Exception:
        hook_input = {}

    reply_text = hook_input.get("last_assistant_message", "")
    session_id = (hook_input.get("session_id", "")
                  or os.environ.get("CLAUDE_SESSION_ID", "")
                  or "unknown")

    try:
        project = resolve_project_id()
    except Exception:
        project = hook_input.get("cwd", "default")

    stats = run_citation_detection(reply_text, project, session_id)

    # Stop hook: 不向 stdout 写任何东西（避免干扰 extractor 的输出流）
    # 只写 stderr 用于调试
    if stats.get("cited", 0) + stats.get("uncited", 0) > 0:
        sys.stderr.write(
            f"[citation_detector] project={project} "
            f"cited={stats['cited']} uncited={stats['uncited']} "
            f"semantic_updated={stats['semantic_updated']}\n"
        )

    sys.exit(0)


if __name__ == "__main__":
    main()
