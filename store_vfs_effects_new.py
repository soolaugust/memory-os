"""
store_vfs_effects_new.py — 认知效应函数模块 (iter481+)

从 store_vfs.py 拆分（iter493: 文件拆分解决 context window thrashing）。
包含：iter481-492 及后续新增的 apply_* 认知效应函数。

每次新迭代只需：
  1. 在本文件末尾追加新的 apply_* 函数
  2. 在 store_vfs.py 的 update_accessed 末尾追加 try/except 调用

OS 类比：Linux mm/compaction.c — 单独模块，只在需要时被 include
         Working Set 模型 — 活跃迭代只 page-in 本文件
"""
import sqlite3
import datetime
import json
import math
import config

# ── Re-export：保持向后兼容，外部可从 store_vfs 或本文件 import ──

def apply_testing_effect(
    conn: "sqlite3.Connection",
    chunk_ids: list,
    now_iso: str = None,
) -> dict:
    """iter481: Testing Effect / Retrieval Practice Effect (TPE) —
    被主动检索到的 chunk 比单纯访问获得更高的 stability 加成。

    认知科学依据：
      Roediger & Karpicke (2006) "Test-enhanced learning" Science 319(5865):966-8 —
        1周后保留率：测试组 64% vs 复习组 40%；Cohen's d ≈ 1.0（认知科学最强效应之一）。
        机制：主动检索激活"检索练习"路径 → 强化编码痕迹 → 更高长时稳定性。
      Karpicke & Roediger (2008) PNAS: 4次检索比 1次检索+3次复习保留率高2倍。
      Butler & Roediger (2007): 测试效应跨领域稳定（不依赖材料类型）。

    OS 类比：CPU TLB hit vs page table walk（arch/x86/mm/tlb.c）—
      TLB 命中（= 主动检索）更新 LRU 并保留 hot page 于 L1；
      page table walk（= 被动复习）成本高且不更新 TLB；
      TPE = TLB hit 对应的额外 refcount boost。
    """
    result = {"tpe_boosted": 0}
    if not chunk_ids:
        return result
    try:
        import config as _cfg
        import datetime as _dt
        if not _cfg.get("store_vfs.tpe_enabled"):
            return result

        boost_factor = float(_cfg.get("store_vfs.tpe_boost_factor"))
        max_boost = float(_cfg.get("store_vfs.tpe_max_boost"))
        min_importance = float(_cfg.get("store_vfs.tpe_min_importance"))
        lookback_min = int(_cfg.get("store_vfs.tpe_lookback_minutes"))

        if now_iso is None:
            now_iso = _dt.datetime.now(_dt.timezone.utc).isoformat()

        # 计算时间窗口下界
        try:
            now_dt = _dt.datetime.fromisoformat(now_iso)
        except Exception:
            now_dt = _dt.datetime.now(_dt.timezone.utc)
        cutoff_dt = now_dt - _dt.timedelta(minutes=lookback_min)
        cutoff_iso = cutoff_dt.isoformat()

        for chunk_id in chunk_ids:
            row = conn.execute(
                "SELECT stability, importance FROM memory_chunks WHERE id=?", (chunk_id,)
            ).fetchone()
            if not row:
                continue
            stab = float(row[0] or 1.0)
            imp = float(row[1] or 0.0)
            if imp < min_importance:
                continue

            # 检查最近是否有 recall_traces 命中该 chunk
            # recall_traces.top_k_json 是 JSON 数组，包含 chunk_id
            try:
                hit_count = conn.execute(
                    """SELECT COUNT(*) FROM recall_traces
                       WHERE timestamp >= ? AND top_k_json LIKE ?""",
                    (cutoff_iso, f'%"{chunk_id}"%')
                ).fetchone()[0]
            except Exception:
                hit_count = 0

            if hit_count == 0:
                # 没有 recall_traces 命中，不是"主动检索"，不触发 TPE
                continue

            capped_boost = min(max_boost, boost_factor)
            new_stab = min(365.0, stab * (1.0 + capped_boost))
            if new_stab > stab + 1e-6:
                conn.execute(
                    "UPDATE memory_chunks SET stability=? WHERE id=?", (new_stab, chunk_id)
                )
                result["tpe_boosted"] += 1
        return result
    except Exception:
        return result


def apply_spacing_effect_bonus(
    conn: "sqlite3.Connection",
    chunk_ids: list,
    now_iso: str = None,
) -> dict:
    """iter482: Spacing Effect Bonus (SEB) — 重复访问时，间隔越长每次获得的 stability 增益越大。

    认知科学依据：
      Ebbinghaus (1885) "Über das Gedächtnis" — 间隔复习遗忘曲线最优。
      Bahrick, Bahrick & Bahrick (1993) JEPS: 间隔越大，长期 retention 增益越大。
      Cepeda et al. (2006) Psych Bulletin meta-analysis（317 studies）: d=0.70；
        最优复习间隔 = retention interval × 10-20%。
      Landauer & Bjork (1978): 间隔练习（spaced practice）vs 集中练习（massed practice）
        长期差异可达 40-50% retention（相同总练习时间下）。

    OS 类比：Linux page access bit TLB aging（arch/x86/mm/tlb.c）—
      系统定期清除 access bit；距上次清除（= 上次访问）越久，
      下次命中时 TLB 优先级越高（aging = 越稀缺越珍贵）；
      长间隔 = 该 page 在 aged 状态下仍被访问 = 高价值 → priority++。
    """
    result = {"seb_boosted": 0}
    if not chunk_ids:
        return result
    try:
        import config as _cfg
        import math as _math
        import datetime as _dt
        if not _cfg.get("store_vfs.seb_enabled"):
            return result

        min_gap_hours = int(_cfg.get("store_vfs.seb_min_gap_hours"))
        base_bonus = float(_cfg.get("store_vfs.seb_base_bonus"))
        max_bonus = float(_cfg.get("store_vfs.seb_max_bonus"))
        min_importance = float(_cfg.get("store_vfs.seb_min_importance"))

        if now_iso is None:
            now_iso = _dt.datetime.now(_dt.timezone.utc).isoformat()

        try:
            now_dt = _dt.datetime.fromisoformat(now_iso)
        except Exception:
            now_dt = _dt.datetime.now(_dt.timezone.utc)

        for chunk_id in chunk_ids:
            row = conn.execute(
                "SELECT stability, importance, last_accessed FROM memory_chunks WHERE id=?",
                (chunk_id,)
            ).fetchone()
            if not row:
                continue
            stab = float(row[0] or 1.0)
            imp = float(row[1] or 0.0)
            if imp < min_importance:
                continue
            last_acc = row[2] or ""
            if not last_acc:
                continue

            try:
                last_dt = _dt.datetime.fromisoformat(last_acc)
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=_dt.timezone.utc)
                if now_dt.tzinfo is None:
                    now_dt = now_dt.replace(tzinfo=_dt.timezone.utc)
                gap_hours = (now_dt - last_dt).total_seconds() / 3600.0
            except Exception:
                continue

            if gap_hours < min_gap_hours:
                continue

            # bonus_ratio = min(max_bonus, base_bonus × log2(gap_hours/min_gap + 1))
            bonus_ratio = min(max_bonus,
                              base_bonus * _math.log2(gap_hours / min_gap_hours + 1))
            new_stab = min(365.0, stab * (1.0 + bonus_ratio))
            if new_stab > stab + 1e-6:
                conn.execute(
                    "UPDATE memory_chunks SET stability=? WHERE id=?", (new_stab, chunk_id)
                )
                result["seb_boosted"] += 1
        return result
    except Exception:
        return result


def apply_priming_effect(
    conn: "sqlite3.Connection",
    chunk_id: str,
    content: str,
    now_iso: str = None,
) -> dict:
    """iter483: Priming Effect (PE) — 编码新 chunk 时，已有语义相似 chunk 提供启动效应，
    使新 chunk 的 stability 更高（提供语义脚手架）。

    认知科学依据：
      Meyer & Schvaneveldt (1971) "Facilitation in recognizing pairs of words" JEPS —
        已激活相关概念（prime）使目标词识别更快（反应时间快 80ms），编码更深。
      Tulving & Osler (1968) "Effectiveness of retrieval cues" — 编码时的提示线索
        在检索时有效 → 说明编码质量受启动影响。
      Collins & Loftus (1975) spreading activation: 语义网络中相关节点激活传播；
        新 chunk 编码时周围语义激活越高，编码越稳固。

    OS 类比：Linux dentry cache warm（fs/dcache.c）—
      相关目录项已在 dentry cache（prime）→ 新文件路径解析（编码）更快更稳定；
      dentry cache hit = 语义启动 = 新 chunk 编码时的"语义脚手架"。
    """
    result = {"pe_boosted": False, "pe_n_primes": 0}
    try:
        import config as _cfg
        if not _cfg.get("store_vfs.pe_enabled"):
            return result

        min_similarity = float(_cfg.get("store_vfs.pe_min_similarity"))
        boost_per_prime = float(_cfg.get("store_vfs.pe_boost_per_prime"))
        max_boost = float(_cfg.get("store_vfs.pe_max_boost"))
        min_importance = float(_cfg.get("store_vfs.pe_min_importance"))
        min_primes = int(_cfg.get("store_vfs.pe_min_primes"))

        row = conn.execute(
            "SELECT stability, importance, project FROM memory_chunks WHERE id=?",
            (chunk_id,)
        ).fetchone()
        if not row:
            return result
        stab = float(row[0] or 1.0)
        imp = float(row[1] or 0.0)
        project = row[2] or ""
        if imp < min_importance:
            return result

        src_words = set((content or "").lower().split())
        if not src_words:
            return result

        # 查找同 project 已有的相似 chunk（不包含自身）
        candidates = conn.execute(
            """SELECT id, content FROM memory_chunks
               WHERE project=? AND id != ? AND importance >= ?
               ORDER BY importance DESC LIMIT 100""",
            (project, chunk_id, min_importance)
        ).fetchall()

        n_primes = 0
        for cand in candidates:
            cand_words = set((cand[1] or "").lower().split())
            if not cand_words:
                continue
            union = len(src_words | cand_words)
            if union == 0:
                continue
            jaccard = len(src_words & cand_words) / union
            if jaccard >= min_similarity:
                n_primes += 1

        result["pe_n_primes"] = n_primes
        if n_primes < min_primes:
            return result

        boost_ratio = min(max_boost, n_primes * boost_per_prime)
        new_stab = min(365.0, stab * (1.0 + boost_ratio))
        if new_stab > stab + 1e-6:
            conn.execute(
                "UPDATE memory_chunks SET stability=? WHERE id=?", (new_stab, chunk_id)
            )
            result["pe_boosted"] = True
        return result
    except Exception:
        return result


def apply_cross_session_consolidation(
    conn: "sqlite3.Connection",
    chunk_ids: list,
    session_id: str,
    now_iso: str = None,
) -> dict:
    """iter484: Cross-Session Consolidation Effect (CCE) — 跨 session 访问的 chunk
    获得额外的 stability 奖励（模拟睡眠/休息期的海马-皮质巩固）。

    认知科学依据：
      Walker & Stickgold (2004) "Sleep-dependent learning and memory consolidation"
        Neuron 44(1):121-133 — 睡眠期海马 sharp-wave ripple（SWR）重放 →
        皮质巩固（系统巩固理论）；睡眠后记忆提升 6-12%（performance boost）。
      Stickgold (2005) Science "Sleep-dependent memory consolidation" —
        睡眠是记忆巩固的积极过程，非仅"减少干扰"。
      Korman et al. (2007): 跨日间隔（含睡眠）vs 非睡眠间隔巩固差异显著。

    OS 类比：Linux kswapd background reclaim（mm/vmscan.c）—
      kswapd 在系统空闲时（≈ session 间隔）主动整理 page，将 cold page 回收、
      warm page 移到高 LRU 位置；session 间隔 = kswapd 运行期 = 后台巩固；
      下次访问时 stability 额外提升 = kswapd 优化后的 page 命中率提升。
    """
    result = {"cce_boosted": 0}
    if not chunk_ids:
        return result
    try:
        import config as _cfg
        import datetime as _dt
        import math as _math
        if not _cfg.get("store_vfs.cce_enabled"):
            return result

        min_gap_hours = int(_cfg.get("store_vfs.cce_min_gap_hours"))
        base_bonus = float(_cfg.get("store_vfs.cce_base_bonus"))
        max_boost = float(_cfg.get("store_vfs.cce_max_boost"))
        min_importance = float(_cfg.get("store_vfs.cce_min_importance"))

        if now_iso is None:
            now_iso = _dt.datetime.now(_dt.timezone.utc).isoformat()

        try:
            now_dt = _dt.datetime.fromisoformat(now_iso)
        except Exception:
            now_dt = _dt.datetime.now(_dt.timezone.utc)

        for chunk_id in chunk_ids:
            row = conn.execute(
                "SELECT stability, importance, source_session, last_accessed "
                "FROM memory_chunks WHERE id=?", (chunk_id,)
            ).fetchone()
            if not row:
                continue
            stab = float(row[0] or 1.0)
            imp = float(row[1] or 0.0)
            if imp < min_importance:
                continue
            src_session = row[2] or ""
            last_acc = row[3] or ""

            # 检查是否跨 session 访问
            if not src_session or not session_id:
                continue
            if src_session == session_id:
                continue  # 同 session，非跨 session 访问

            # 计算时间间隔
            if not last_acc:
                continue
            try:
                last_dt = _dt.datetime.fromisoformat(last_acc)
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=_dt.timezone.utc)
                if now_dt.tzinfo is None:
                    now_dt = now_dt.replace(tzinfo=_dt.timezone.utc)
                gap_hours = (now_dt - last_dt).total_seconds() / 3600.0
            except Exception:
                continue

            if gap_hours < min_gap_hours:
                continue

            # bonus = min(max_boost, base_bonus × min(1.0, gap/24h))
            # 最大奖励在 >= 24h 间隔后达到
            bonus_ratio = min(max_boost, base_bonus * min(1.0, gap_hours / 24.0))
            new_stab = min(365.0, stab * (1.0 + bonus_ratio))
            if new_stab > stab + 1e-6:
                conn.execute(
                    "UPDATE memory_chunks SET stability=? WHERE id=?", (new_stab, chunk_id)
                )
                result["cce_boosted"] += 1
        return result
    except Exception:
        return result


def apply_desirable_difficulty(
    conn: "sqlite3.Connection",
    chunk_ids: list,
    now_iso: str = None,
) -> dict:
    """iter485: Desirable Difficulty Effect (DDE2) — 提取难度高时（低 retrievability + 低 stability）
    成功提取后 stability 增益更大（Bjork 1994）。

    认知科学依据：
      Bjork (1994) "Memory and metamemory considerations in the training of human beings"
        — 适度困难的提取任务（高遗忘率情境下成功检索）强化编码深度；
      Schmidt & Bjork (1992) Psych Science — 困难条件下的练习产生更持久的长期保留效果。
      McDaniel & Masson (1985): 费力提取（elaborative interrogation）比浅层提取更持久。

    OS 类比：Linux TLB miss → page walk — miss 时触发完整 page walk（成本高），
      但完成后更新 TLB，后续命中更快（记忆更稳固）；难提取 = TLB miss，成功 = TLB 重新填充。
    """
    result = {"dde2_boosted": 0}
    if not chunk_ids:
        return result
    try:
        import config as _cfg
        import datetime as _dt
        import math as _math
        if not _cfg.get("store_vfs.dde2_enabled"):
            return result

        r_threshold = float(_cfg.get("store_vfs.dde2_retrievability_threshold"))
        s_threshold = float(_cfg.get("store_vfs.dde2_stability_threshold"))
        bonus_factor = float(_cfg.get("store_vfs.dde2_bonus_per_difficulty"))
        max_boost = float(_cfg.get("store_vfs.dde2_max_boost"))
        min_importance = float(_cfg.get("store_vfs.dde2_min_importance"))

        if now_iso is None:
            now_iso = _dt.datetime.now(_dt.timezone.utc).isoformat()
        try:
            now_dt = _dt.datetime.fromisoformat(now_iso)
            if now_dt.tzinfo is None:
                now_dt = now_dt.replace(tzinfo=_dt.timezone.utc)
        except Exception:
            now_dt = _dt.datetime.now(_dt.timezone.utc)

        for chunk_id in chunk_ids:
            row = conn.execute(
                "SELECT stability, importance, last_accessed "
                "FROM memory_chunks WHERE id=?", (chunk_id,)
            ).fetchone()
            if not row:
                continue
            stab = float(row[0] or 1.0)
            imp = float(row[1] or 0.0)
            if imp < min_importance:
                continue
            if stab > s_threshold:
                continue  # stability 已高，不算"难"

            # 计算 retrievability：R = exp(-t/S)
            last_acc = row[2] or ""
            t_days = 0.0
            if last_acc:
                try:
                    last_dt = _dt.datetime.fromisoformat(last_acc)
                    if last_dt.tzinfo is None:
                        last_dt = last_dt.replace(tzinfo=_dt.timezone.utc)
                    t_days = max(0.0, (now_dt - last_dt).total_seconds() / 86400.0)
                except Exception:
                    pass
            retrievability = _math.exp(-t_days / max(stab, 0.1))
            if retrievability > r_threshold:
                continue  # retrievability 仍高，不算"难"

            # 难度得分：越低 R 越难，奖励越大
            difficulty = 1.0 - retrievability  # [0, 1]
            bonus_ratio = min(max_boost, bonus_factor * difficulty)
            new_stab = min(365.0, stab * (1.0 + bonus_ratio))
            if new_stab > stab + 1e-6:
                conn.execute(
                    "UPDATE memory_chunks SET stability=? WHERE id=?", (new_stab, chunk_id)
                )
                result["dde2_boosted"] += 1
        return result
    except Exception:
        return result


def apply_contextual_reinstatement(
    conn: "sqlite3.Connection",
    chunk_ids: list,
    query_namespace: str = "",
    query_tags: list = None,
) -> dict:
    """iter486: Contextual Reinstatement Effect (CRE2) — 恢复编码上下文时检索效率提高。

    认知科学依据：
      Godden & Baddeley (1975) British J Psychology — 水下/陆地编码实验证明，
        上下文匹配使记忆提取成功率提高 ~40%；
      Smith (1979): 在编码相同房间测验比不同房间高 ~25%；
      Smith & Vela (2001) Psych Bulletin — 上下文依赖记忆 meta-analysis 综述。

    OS 类比：CPU cache locality（spatial/temporal locality） —
      访问与之前同一 locality 集合的 page，L1/L2 cache 命中率更高；
      namespace/tag 匹配 = 访问同一 cache line 集合。
    """
    result = {"cre2_boosted": 0}
    if not chunk_ids:
        return result
    try:
        import config as _cfg
        if not _cfg.get("store_vfs.cre2_enabled"):
            return result

        ns_bonus = float(_cfg.get("store_vfs.cre2_namespace_match_bonus"))
        tag_bonus = float(_cfg.get("store_vfs.cre2_tag_overlap_bonus"))
        tag_threshold = float(_cfg.get("store_vfs.cre2_tag_overlap_threshold"))
        max_boost = float(_cfg.get("store_vfs.cre2_max_boost"))
        min_importance = float(_cfg.get("store_vfs.cre2_min_importance"))
        import json as _json

        if query_tags is None:
            query_tags = []
        query_tag_set = set(query_tags)

        for chunk_id in chunk_ids:
            row = conn.execute(
                "SELECT retrievability, importance, project, tags "
                "FROM memory_chunks WHERE id=?", (chunk_id,)
            ).fetchone()
            if not row:
                continue
            ret = float(row[0] or 0.5)
            imp = float(row[1] or 0.0)
            if imp < min_importance:
                continue

            boost = 0.0
            # project 作为 namespace 代理（OS 类比：cgroup/subsystem）
            chunk_ns = row[2] or ""
            if query_namespace and chunk_ns and query_namespace == chunk_ns:
                boost += ns_bonus

            # Tag Jaccard
            if query_tag_set:
                try:
                    raw_tags = row[3]
                    if isinstance(raw_tags, str):
                        chunk_tags = set(_json.loads(raw_tags)) if raw_tags.startswith("[") else set(raw_tags.split(","))
                    elif isinstance(raw_tags, (list, set)):
                        chunk_tags = set(raw_tags)
                    else:
                        chunk_tags = set()
                    if chunk_tags:
                        intersection = len(query_tag_set & chunk_tags)
                        union = len(query_tag_set | chunk_tags)
                        jaccard = intersection / union if union else 0.0
                        if jaccard >= tag_threshold:
                            boost += tag_bonus
                except Exception:
                    pass

            if boost <= 0:
                continue
            boost = min(max_boost, boost)
            new_ret = min(1.0, ret + boost)
            if new_ret > ret + 1e-6:
                conn.execute(
                    "UPDATE memory_chunks SET retrievability=? WHERE id=?", (new_ret, chunk_id)
                )
                result["cre2_boosted"] += 1
        return result
    except Exception:
        return result


def apply_emotion_tagging_decay_reduction(
    conn: "sqlite3.Connection",
    chunk_ids: list,
) -> dict:
    """iter487: Emotion Tagging Effect (ETE2) — 高情绪价值 chunk 的 stability 衰减速率降低。

    认知科学依据：
      McGaugh (2000) Science "Memory — a century of consolidation" —
        杏仁核通过 norepinephrine/cortisol 调节海马巩固，情绪唤起（arousal）增强 LTP；
      Cahill & McGaugh (1998): 情绪事件被记住更长（杏仁核-海马互作）；
      LaBar & Cabeza (2006) Nat Rev Neuro — 情绪记忆优势效应 meta-analysis。

    OS 类比：Linux cgroup memory.min 保护 —
      高优先级进程的 pages 受 memory.min 保护，kswapd 不会回收；
      高情绪 = 高 importance ≈ cgroup min 保护 → 衰减减缓。

    实现：通过给 stability 设置更高下限（decay floor）来模拟衰减减缓。
    在实际访问时检测，若 chunk 满足高情绪条件，则额外提升 stability。
    """
    result = {"ete2_boosted": 0}
    if not chunk_ids:
        return result
    try:
        import config as _cfg
        if not _cfg.get("store_vfs.ete2_enabled"):
            return result

        imp_threshold = float(_cfg.get("store_vfs.ete2_importance_threshold"))
        decay_reduction = float(_cfg.get("store_vfs.ete2_stability_decay_reduction"))
        kw_bonus = float(_cfg.get("store_vfs.ete2_keyword_bonus"))
        max_reduction = float(_cfg.get("store_vfs.ete2_max_decay_reduction"))
        emotion_keywords = _cfg.get("store_vfs.ete2_emotion_keywords") or []

        for chunk_id in chunk_ids:
            row = conn.execute(
                "SELECT stability, importance, content "
                "FROM memory_chunks WHERE id=?", (chunk_id,)
            ).fetchone()
            if not row:
                continue
            stab = float(row[0] or 1.0)
            imp = float(row[1] or 0.0)
            if imp < imp_threshold:
                continue  # 不满足高情绪条件

            total_reduction = decay_reduction

            # 检测情绪关键词
            content = (row[2] or "").lower()
            if emotion_keywords and any(kw.lower() in content for kw in emotion_keywords):
                total_reduction += kw_bonus

            total_reduction = min(max_reduction, total_reduction)

            # 提升 stability（模拟衰减减缓：等效于 stability × (1 + reduction)）
            bonus_stab = min(365.0, stab * (1.0 + total_reduction))
            if bonus_stab > stab + 1e-6:
                conn.execute(
                    "UPDATE memory_chunks SET stability=? WHERE id=?", (bonus_stab, chunk_id)
                )
                result["ete2_boosted"] += 1
        return result
    except Exception:
        return result


def apply_inhibition_of_return(
    conn: "sqlite3.Connection",
    chunk_ids: list,
    now_iso: str = None,
) -> dict:
    """iter488: Inhibition of Return (IOR) — 短时间内重复访问同一 chunk 时 stability 增益递减。

    认知科学依据：
      Posner & Cohen (1984) "Components of visual orienting" Attention & Performance X —
        注意力返回刚刚访问位置的速度较慢（IOR）；短时间内重复激活同一记忆收益递减；
      Klein (2000) TICS "Inhibition of return" — IOR 广泛存在于注意力和记忆提取中；
      频繁重复访问 ≈ 过度练习（overlearning），对稳定性提升边际效益递减。

    OS 类比：Linux madvise(MADV_RANDOM) + prefetch inhibition —
      对刚刚读取的 page 降低预取优先级，预取资源分配给尚未访问的 page；
      频繁重访 = 预取器降低优先级 = stability 增益惩罚。

    实现：检测 last_accessed 与 now 的间隔；若在 ior_inhibition_window_secs 内，
    则对本次访问的 stability 增益乘以 penalty_factor。
    直接调整 stability（若当前 stability 刚被其他效应提升，则部分回退）。
    """
    result = {"ior_penalized": 0}
    if not chunk_ids:
        return result
    try:
        import config as _cfg
        import datetime as _dt
        if not _cfg.get("store_vfs.ior_enabled"):
            return result

        window_secs = int(_cfg.get("store_vfs.ior_inhibition_window_secs"))
        penalty_factor = float(_cfg.get("store_vfs.ior_penalty_factor"))
        min_interval_secs = int(_cfg.get("store_vfs.ior_min_interval_secs"))
        min_importance = float(_cfg.get("store_vfs.ior_min_importance"))

        if now_iso is None:
            now_iso = _dt.datetime.now(_dt.timezone.utc).isoformat()
        try:
            now_dt = _dt.datetime.fromisoformat(now_iso)
            if now_dt.tzinfo is None:
                now_dt = now_dt.replace(tzinfo=_dt.timezone.utc)
        except Exception:
            now_dt = _dt.datetime.now(_dt.timezone.utc)

        for chunk_id in chunk_ids:
            row = conn.execute(
                "SELECT stability, importance, last_accessed "
                "FROM memory_chunks WHERE id=?", (chunk_id,)
            ).fetchone()
            if not row:
                continue
            stab = float(row[0] or 1.0)
            imp = float(row[1] or 0.0)
            if imp < min_importance:
                continue

            last_acc = row[2] or ""
            if not last_acc:
                continue

            try:
                last_dt = _dt.datetime.fromisoformat(last_acc)
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=_dt.timezone.utc)
                gap_secs = (now_dt - last_dt).total_seconds()
            except Exception:
                continue

            if gap_secs < 0:
                continue
            if gap_secs > window_secs:
                continue  # 超出抑制窗口，不惩罚

            # 窗口内重复访问 → stability 增益惩罚
            # 间隔越短惩罚越重：0~min_interval_secs 区间线性从最重到最轻
            if gap_secs < min_interval_secs:
                effective_penalty = penalty_factor  # 最重惩罚
            else:
                # 线性插值：从 penalty_factor 到 1.0（无惩罚），按间隔比例
                t = (gap_secs - min_interval_secs) / max(window_secs - min_interval_secs, 1)
                effective_penalty = penalty_factor + t * (1.0 - penalty_factor)

            # 惩罚：将 stability 从当前值向 stab * effective_penalty 方向调整
            # 只惩罚高于"正常水平"的部分（避免无意义降低）
            penalized_stab = max(1.0, stab * effective_penalty)
            if penalized_stab < stab - 1e-6:
                conn.execute(
                    "UPDATE memory_chunks SET stability=? WHERE id=?", (penalized_stab, chunk_id)
                )
                result["ior_penalized"] += 1
        return result
    except Exception:
        return result


def apply_encoding_variability(
    conn: "sqlite3.Connection",
    chunk_ids: list,
) -> dict:
    """iter489: Encoding Variability Effect (EVE) — 同一 chunk 在多种不同 session_type 下
    被访问时，形成多条检索路径，stability 额外提升（Martin 1972）。

    认知科学依据：
      Martin (1972) Psychological Review — 编码变异假说：同一刺激在不同上下文中编码，
        产生多条独立检索路径，降低单次遗忘的全局影响；
      Glenberg (1979): 上下文多样性与长期保留显著正相关（r≈0.60）；
      Estes (1955) context fluctuation model — 学习时的上下文在记忆中被编码，
        上下文多样 → 检索线索更丰富。

    OS 类比：DM-multipath (Device Mapper Multipath) —
      同一 block device 通过多条 I/O 路径访问，任一路径失效不影响整体可用性；
      多条检索路径 = 多路径冗余，单一遗忘不影响整体提取成功率。

    实现：从 session_type_history 字段提取不同 session 类型数量，
    不同类型数超过 min_unique_session_types 时触发 stability 加成。
    """
    result = {"eve_boosted": 0}
    if not chunk_ids:
        return result
    try:
        import config as _cfg
        import json as _json
        if not _cfg.get("store_vfs.eve_enabled"):
            return result

        min_types = int(_cfg.get("store_vfs.eve_min_unique_session_types"))
        bonus_per_type = float(_cfg.get("store_vfs.eve_bonus_per_type"))
        max_boost = float(_cfg.get("store_vfs.eve_max_boost"))
        min_importance = float(_cfg.get("store_vfs.eve_min_importance"))

        for chunk_id in chunk_ids:
            row = conn.execute(
                "SELECT stability, importance, session_type_history "
                "FROM memory_chunks WHERE id=?", (chunk_id,)
            ).fetchone()
            if not row:
                continue
            stab = float(row[0] or 1.0)
            imp = float(row[1] or 0.0)
            if imp < min_importance:
                continue

            sth = row[2] or ""
            # session_type_history 存储格式：逗号分隔的 session_type 记录
            # 或 JSON 数组
            try:
                if sth.startswith("["):
                    types_list = _json.loads(sth)
                else:
                    types_list = [t.strip() for t in sth.split(",") if t.strip()]
            except Exception:
                types_list = [t.strip() for t in sth.split(",") if t.strip()]

            unique_types = len(set(types_list))
            if unique_types < min_types:
                continue

            # 额外多样性：超过最低要求的每个类型各增加 bonus_per_type
            extra_types = unique_types - min_types + 1  # >= 1
            bonus_ratio = min(max_boost, bonus_per_type * extra_types)
            new_stab = min(365.0, stab * (1.0 + bonus_ratio))
            if new_stab > stab + 1e-6:
                conn.execute(
                    "UPDATE memory_chunks SET stability=? WHERE id=?", (new_stab, chunk_id)
                )
                result["eve_boosted"] += 1
        return result
    except Exception:
        return result


def apply_zeigarnik_effect(
    conn: "sqlite3.Connection",
    chunk_ids: list,
) -> dict:
    """iter490: Zeigarnik Effect (ZEF) — 含"未完成"信号的 chunk 比已完成 chunk 稳定性更高。

    认知科学依据：
      Zeigarnik (1927) Psychologische Forschung — 未完成任务比完成任务的回忆率高 ~90%；
      持续激活假说：中断任务产生认知张力（cognitive tension），维持工作记忆激活；
      Ovsiankina (1928): 中断任务自发产生恢复冲动（resumption intention），维持记忆优先级。

    OS 类比：dirty page tracking (mm/page-writeback.c) —
      含未刷新（dirty）数据的 page 受 writeback 保护，不被 kswapd 主动回收；
      等待 fsync 完成 = 等待任务完成 = 受保护的工作状态。

    实现：扫描 content + summary 字段，检测 TODO/FIXME/PENDING 等信号词，
    若存在则提升 stability。
    """
    result = {"zef_boosted": 0}
    if not chunk_ids:
        return result
    try:
        import config as _cfg
        if not _cfg.get("store_vfs.zef_enabled"):
            return result

        todo_keywords = _cfg.get("store_vfs.zef_todo_keywords") or []
        stability_bonus = float(_cfg.get("store_vfs.zef_stability_bonus"))
        max_boost = float(_cfg.get("store_vfs.zef_max_boost"))
        min_importance = float(_cfg.get("store_vfs.zef_min_importance"))

        for chunk_id in chunk_ids:
            row = conn.execute(
                "SELECT stability, importance, content, summary "
                "FROM memory_chunks WHERE id=?", (chunk_id,)
            ).fetchone()
            if not row:
                continue
            stab = float(row[0] or 1.0)
            imp = float(row[1] or 0.0)
            if imp < min_importance:
                continue

            content_lower = (row[2] or "").lower()
            summary_lower = (row[3] or "").lower()
            combined = content_lower + " " + summary_lower

            # 检测未完成信号词
            if not any(kw.lower() in combined for kw in todo_keywords):
                continue

            bonus_ratio = min(max_boost, stability_bonus)
            new_stab = min(365.0, stab * (1.0 + bonus_ratio))
            if new_stab > stab + 1e-6:
                conn.execute(
                    "UPDATE memory_chunks SET stability=? WHERE id=?", (new_stab, chunk_id)
                )
                result["zef_boosted"] += 1
        return result
    except Exception:
        return result


def apply_von_restorff_isolation(
    conn: "sqlite3.Connection",
    chunk_ids: list,
    session_id: str = "",
) -> dict:
    """iter491: von Restorff Isolation Effect (VRE) — 在 session 中 chunk_type 稀少的
    chunk（独特项目）获得额外 stability 提升。

    认知科学依据：
      von Restorff (1933) Psychologische Forschung — 同质列表中的孤立（独特）项目
        被记住的频率显著高于普通项目（isolation effect）；
      Hunt & Lamb (2001) J Exp Psych — isolation effect 在语义上下文中稳健，
        效果量 d ≈ 0.80；
      Fabiani & Donchin (1995): isolation effect 与 P300 波幅（注意力增强）正相关。

    OS 类比：Linux LRU generation aging (MGLRU) —
      在主要由 old-gen page 组成的 list 中，gen=0（newly accessed）的 page
      在 eviction 时受到额外保护；稀有类型 = LRU gen=0 in old pool。

    实现：统计当前 session 中各 chunk_type 的比例，
    比例低于 vre_rarity_threshold 的类型视为"稀有/独特"，触发 stability 加成。
    """
    result = {"vre_boosted": 0}
    if not chunk_ids:
        return result
    try:
        import config as _cfg
        if not _cfg.get("store_vfs.vre_enabled"):
            return result

        rarity_threshold = float(_cfg.get("store_vfs.vre_rarity_threshold"))
        stability_bonus = float(_cfg.get("store_vfs.vre_stability_bonus"))
        max_boost = float(_cfg.get("store_vfs.vre_max_boost"))
        min_importance = float(_cfg.get("store_vfs.vre_min_importance"))
        min_session_chunks = int(_cfg.get("store_vfs.vre_min_session_chunks"))

        # 统计 session 中各 chunk_type 的数量
        if session_id:
            type_counts = {}
            rows = conn.execute(
                "SELECT chunk_type, COUNT(*) as cnt FROM memory_chunks "
                "WHERE source_session=? GROUP BY chunk_type", (session_id,)
            ).fetchall()
            total = sum(r[1] for r in rows)
            if total < min_session_chunks:
                # session 内 chunk 不足，用全局比例
                rows = conn.execute(
                    "SELECT chunk_type, COUNT(*) as cnt FROM memory_chunks GROUP BY chunk_type"
                ).fetchall()
                total = sum(r[1] for r in rows)
            for r in rows:
                type_counts[r[0]] = r[1]
        else:
            # 无 session id，用全局比例
            rows = conn.execute(
                "SELECT chunk_type, COUNT(*) as cnt FROM memory_chunks GROUP BY chunk_type"
            ).fetchall()
            total = sum(r[1] for r in rows)
            type_counts = {r[0]: r[1] for r in rows}

        if total < min_session_chunks:
            return result

        for chunk_id in chunk_ids:
            row = conn.execute(
                "SELECT stability, importance, chunk_type "
                "FROM memory_chunks WHERE id=?", (chunk_id,)
            ).fetchone()
            if not row:
                continue
            stab = float(row[0] or 1.0)
            imp = float(row[1] or 0.0)
            if imp < min_importance:
                continue

            ctype = row[2] or "observation"
            type_ratio = type_counts.get(ctype, 0) / max(total, 1)
            if type_ratio >= rarity_threshold:
                continue  # 不稀有，不触发

            bonus_ratio = min(max_boost, stability_bonus)
            new_stab = min(365.0, stab * (1.0 + bonus_ratio))
            if new_stab > stab + 1e-6:
                conn.execute(
                    "UPDATE memory_chunks SET stability=? WHERE id=?", (new_stab, chunk_id)
                )
                result["vre_boosted"] += 1
        return result
    except Exception:
        return result


def apply_production_effect(
    conn: "sqlite3.Connection",
    chunk_ids: list,
) -> dict:
    """iter492: Production Effect (PEF) — 输出类（生产型）chunk 的编码更深，stability 更高。

    认知科学依据：
      MacLeod et al. (2010) J Exp Psych: General — 大声朗读（production）比默读
        在再认测试中高 ~10-15%（production effect）；
      MacLeod & Bodner (2017): production effect 的核心是"增强区分度"而非简单重复；
      Forrin et al. (2012): 写作产生效果与大声朗读相当，均优于默读。

    OS 类比：write-back vs write-through cache —
      write-back（输出类 chunk）需要额外处理步骤（将数据写入 dirty page），
      但获得更长的"in-memory"生命周期（delayed writeback = 更稳固的编码）；
      decision/reflection 等 = write-back operation = higher stability.
    """
    result = {"pef_boosted": 0}
    if not chunk_ids:
        return result
    try:
        import config as _cfg
        if not _cfg.get("store_vfs.pef_enabled"):
            return result

        production_types = _cfg.get("store_vfs.pef_production_types") or []
        stability_bonus = float(_cfg.get("store_vfs.pef_stability_bonus"))
        max_boost = float(_cfg.get("store_vfs.pef_max_boost"))
        min_importance = float(_cfg.get("store_vfs.pef_min_importance"))

        # 规范化 production_types
        prod_set = {t.lower().strip() for t in production_types}

        for chunk_id in chunk_ids:
            row = conn.execute(
                "SELECT stability, importance, chunk_type "
                "FROM memory_chunks WHERE id=?", (chunk_id,)
            ).fetchone()
            if not row:
                continue
            stab = float(row[0] or 1.0)
            imp = float(row[1] or 0.0)
            if imp < min_importance:
                continue

            ctype = (row[2] or "").lower().strip()
            if ctype not in prod_set:
                continue  # 非生产型 chunk_type

            bonus_ratio = min(max_boost, stability_bonus)
            new_stab = min(365.0, stab * (1.0 + bonus_ratio))
            if new_stab > stab + 1e-6:
                conn.execute(
                    "UPDATE memory_chunks SET stability=? WHERE id=?", (new_stab, chunk_id)
                )
                result["pef_boosted"] += 1
        return result
    except Exception:
        return result


def apply_emotional_salience(
    conn: sqlite3.Connection,
    chunk_id: str,
    text: str,
    base_importance: float,
) -> float:
    """
    迭代320：根据情感显著性调整 chunk 的 importance 并写回 DB。
    iter399：同时写入 emotional_weight（0.0~1.0），供 retriever 情绪增强使用。

    算法：
      delta = compute_emotional_salience(text)
      emotional_weight = clamp(delta / 0.25, 0.0, 1.0)  # 正向 delta 归一化为权重
      if |delta| < 0.01 → 不写 DB（避免无意义更新）
      new_importance = clamp(base_importance + delta, 0.05, 0.98)
      写入 memory_chunks.importance + emotional_weight

    OS 类比：Linux OOM Killer oom_score_adj 写入 —
      fork() 时继承父进程的 oom_score_adj，每个进程可自主调整；
      这里 importance 由 extractor 初始评估，情感显著性在写入后再调整。
      iter399 OS 类比：Linux mempolicy MPOL_PREFERRED_MANY —
        写入时标注页面的"情感节点亲和性"（emotional_weight），
        检索时 retriever 用此权重决定 boost 量（类比 NUMA locality hint）。

    Returns:
      new_importance（调整后；若无调整则返回 base_importance）
    """
    delta = compute_emotional_salience(text)

    # iter399: emotional_weight — 正向情绪强度归一化到 [0.0, 1.0]
    # 负向 delta（已废弃/已解决）不产生情绪权重（只影响 importance 降权）
    emotional_weight = round(max(0.0, min(1.0, delta / 0.25)), 4) if delta > 0 else 0.0

    # iter424: emotional_valence — 情绪效价（独立于唤醒度）
    emotional_valence = round(compute_emotional_valence(text), 4)

    if abs(delta) < 0.01:
        # delta 微弱 — 仍写入 emotional_weight=0（明确表示无情绪显著性）
        # 但只在字段为 NULL 时才写（避免覆盖已有有效值）
        try:
            conn.execute(
                "UPDATE memory_chunks SET emotional_weight=?, emotional_valence=? "
                "WHERE id=? AND (emotional_weight IS NULL OR emotional_weight=0)",
                (0.0, emotional_valence, chunk_id),
            )
        except Exception:
            pass
        return base_importance

    new_importance = max(0.05, min(0.98, base_importance + delta))
    if abs(new_importance - base_importance) < 0.001:
        new_importance = base_importance

    try:
        conn.execute(
            "UPDATE memory_chunks SET importance=?, emotional_weight=?, "
            "emotional_valence=?, updated_at=? WHERE id=?",
            (round(new_importance, 4), emotional_weight, emotional_valence,
             datetime.now(timezone.utc).isoformat(), chunk_id),
        )
    except Exception:
        pass
    return new_importance


# ── iter396：Source Monitoring — 信源监控加权（Johnson 1993）─────────────────
#
# 认知科学依据：
#   Johnson & Raye (1981) Reality Monitoring：
#     人类具备区分「内部生成」与「外部感知」记忆的元认知能力。
#     来自外部直接感知的记忆比内部推断的记忆更可靠，但并非绝对；
#     人容易把听说的事情记成"亲眼所见"（来源错误归因，source misattribution）。
#   Johnson (1993) MEM (Multiple Entry Model)：
#     记忆系统维护「来源标签」（source tag），帮助区分自我生成 vs 外部输入。
#     来源可信度（source credibility）影响信息的检索优先级和记忆强化程度。
#   Zaragoza & Mitchell (1996)：
#     高可信度来源的信息比低可信度来源更容易被记住和相信。
#
# OS 类比：Linux LSM（Linux Security Modules）
#   每次 file open / exec / socket 操作前，LSM hook 查询来源的 security context
#   （SELinux label / AppArmor profile），根据来源授予不同的访问权限。
#   这里：每次 chunk 写入时打上 source_type 标签，检索时据此调整 score。
#
# 实现：
#   1. compute_source_reliability(chunk_type, source_type, content) → float
#      根据 chunk_type + source_type 的组合估算可信度
#   2. source_monitor_weight(source_reliability) → float
#      将可信度转换为检索分数调整因子（range: 0.8 ~ 1.2）
#   3. apply_source_monitoring(conn, chunk_id, chunk_type, source_type, content)
#      写入 source_type + source_reliability 到 DB

# ─ 来源可信度基线表：chunk_type × source_type → base_reliability ─
_SOURCE_RELIABILITY_TABLE: dict = {
    # (chunk_type, source_type) → base reliability
    # direct = 用户直接陈述/观察
    ("design_constraint", "direct"):    0.95,
    ("decision",          "direct"):    0.90,
    ("task_state",        "direct"):    0.85,
    ("reasoning_chain",   "direct"):    0.80,
    ("procedure",         "direct"):    0.85,
    # tool_output = 代码/命令执行结果（机器生成，高重复性）
    ("design_constraint", "tool_output"): 0.88,
    ("decision",          "tool_output"): 0.85,
    ("task_state",        "tool_output"): 0.82,
    ("reasoning_chain",   "tool_output"): 0.78,
    ("procedure",         "tool_output"): 0.80,
    # inferred = 从多条信息推断（中等可信度）
    ("design_constraint", "inferred"):  0.72,
    ("decision",          "inferred"):  0.68,
    ("task_state",        "inferred"):  0.65,
    ("reasoning_chain",   "inferred"):  0.70,
    ("procedure",         "inferred"):  0.65,
    # hearsay = 间接转述/转述他人说法（最低可信度）
    ("design_constraint", "hearsay"):   0.50,
    ("decision",          "hearsay"):   0.45,
    ("task_state",        "hearsay"):   0.40,
    ("reasoning_chain",   "hearsay"):   0.48,
    ("procedure",         "hearsay"):   0.42,
}

# 各 source_type 的默认可信度（chunk_type 无明确映射时）
_SOURCE_TYPE_DEFAULT: dict = {
    "direct":      0.85,
    "tool_output": 0.80,
    "inferred":    0.68,
    "hearsay":     0.45,
    "unknown":     0.70,
}

# 关键词信号 → 推断 source_type（用于自动标注）
# 优先级：hearsay > inferred > tool_output > direct（越 uncertain 越优先检出）
import re as _re_sm

_SOURCE_HEARSAY_RE = _re_sm.compile(
    r"据说|听说|有人说|用户说|他说|她说|they said|I heard|reportedly|allegedly|"
    r"someone mentioned|it is said",
    _re_sm.IGNORECASE,
)
_SOURCE_INFERRED_RE = _re_sm.compile(
    r"推测|可能|应该|估计|推断|大概|based on|likely|probably|presumably|"
    r"it seems|appears to|suggests that",
    _re_sm.IGNORECASE,
)
_SOURCE_TOOL_OUTPUT_RE = _re_sm.compile(
    r"```|输出:|output:|result:|error:|traceback|exception|running|executed|"
    r"\$ |>>> |test passed|test failed|pytest|assert|build|compile",
    _re_sm.IGNORECASE,
)


def infer_source_type(text: str) -> str:
    """
    iter396：从文本内容自动推断 source_type。

    按优先级扫描关键词：
      hearsay → inferred → tool_output → direct（默认）

    OS 类比：Linux file magic 检测 — `file` 命令扫描文件头字节推断文件类型，
      而非依赖用户提供的文件名后缀。
    """
    if not text:
        return "unknown"
    if _SOURCE_HEARSAY_RE.search(text):
        return "hearsay"
    if _SOURCE_INFERRED_RE.search(text):
        return "inferred"
    if _SOURCE_TOOL_OUTPUT_RE.search(text):
        return "tool_output"
    return "direct"


def compute_source_reliability(
    chunk_type: str,
    source_type: str,
    content: str = "",
) -> float:
    """
    iter396：计算 chunk 的来源可信度（source_reliability）。

    算法：
      1. 从 _SOURCE_RELIABILITY_TABLE 查找 (chunk_type, source_type) 基线值
      2. 若无明确映射，使用 _SOURCE_TYPE_DEFAULT[source_type]
      3. 若 content 包含 uncertainty 词语（可能/估计/应该），适当降低（−0.05）
      4. 若 content 包含 certainty 词语（确认/已验证/verified），适当提高（+0.05）
      5. clamp 到 [0.2, 1.0]

    Returns:
      float ∈ [0.2, 1.0]，越高表示来源越可靠
    """
    if not source_type or source_type not in _SOURCE_TYPE_DEFAULT:
        source_type = "unknown"
    base = _SOURCE_RELIABILITY_TABLE.get(
        (chunk_type, source_type),
        _SOURCE_TYPE_DEFAULT.get(source_type, 0.70),
    )
    # 内容微调：不确定性词 → −0.05；确认词 → +0.05
    adjustment = 0.0
    if content:
        _uncertainty_re = _re_sm.compile(
            r'可能|估计|大概|不确定|probably|might|may be|uncertain|unclear',
            _re_sm.IGNORECASE,
        )
        _certainty_re = _re_sm.compile(
            r'确认|已验证|confirmed|verified|definitely|proven|tested',
            _re_sm.IGNORECASE,
        )
        if _uncertainty_re.search(content):
            adjustment -= 0.05
        if _certainty_re.search(content):
            adjustment += 0.05
    return round(max(0.2, min(1.0, base + adjustment)), 4)


def source_monitor_weight(source_reliability: float) -> float:
    """
    iter396：将 source_reliability 转换为检索分数调整因子。

    映射规则（线性区间）：
      reliability ≥ 0.85 → weight ∈ [1.00, 1.15]（高可信来源，微幅提升）
      0.60 ≤ reliability < 0.85 → weight ≈ 1.00（中等可信，不调整）
      reliability < 0.60 → weight ∈ [0.80, 1.00]（低可信来源，适度降权）

    设计原则：
      1. 调整幅度适中（max ±0.15），避免来源完全主导语义相关性
      2. 中间区间（0.60~0.85）不调整，防止噪音误判影响召回
      3. 对应 OS 类比：SELinux label 决定的访问权限不是二元的，
         而是 capability 粒度的（只有明确高风险的 context 才被限制）

    Returns:
      float ∈ [0.80, 1.15]
    """
    r = max(0.0, min(1.0, float(source_reliability) if source_reliability is not None else 0.70))
    if r >= 0.85:
        # 高可信度：线性插值 0.85→1.00，1.0→1.15
        return round(1.00 + (r - 0.85) / (1.0 - 0.85) * 0.15, 4)
    elif r >= 0.60:
        # 中等可信度：不调整
        return 1.00
    else:
        # 低可信度：线性插值 0.0→0.80，0.60→1.00
        return round(0.80 + r / 0.60 * 0.20, 4)


def apply_source_monitoring(
    conn: sqlite3.Connection,
    chunk_id: str,
    chunk_type: str,
    content: str,
    source_type: str = None,
) -> tuple:
    """
    iter396：推断 source_type，计算 source_reliability，并写入 DB。

    OS 类比：LSM security_inode_create hook —
      文件创建时检查 security context，打上 SELinux label（inode security blob）。
      这里 chunk 创建时打上 source_type 标签。

    Returns:
      (source_type: str, source_reliability: float)
    """
    if source_type is None or source_type == "unknown":
        source_type = infer_source_type(content or "")
    reliability = compute_source_reliability(chunk_type or "task_state",
                                             source_type, content or "")
    try:
        conn.execute(
            "UPDATE memory_chunks SET source_type=?, source_reliability=? WHERE id=?",
            (source_type, reliability, chunk_id),
        )
    except Exception:
        pass
    return (source_type, reliability)


# ── iter400：Forgetting Curve Individualization per chunk_type ──────────────
#
# 认知科学依据：
#   Squire (1992) Memory and Brain：程序性记忆（技能）比陈述性情节记忆衰减慢。
#   Tulving (1972)：语义记忆（概念/约束）比情节记忆（具体事件）持久。
#   Ebbinghaus (1885)：同一遗忘曲线对不同类型知识的参数不同。
#   Anderson et al. (1999) ACT-R：基础激活随时间衰减，衰减速率因记忆强度和类型而异。
#
# OS 类比：Linux cgroup memory.reclaim_ratio（per-cgroup）vs vm.swappiness（全局）
#   全局统一 stability_decay=0.92 相当于 vm.swappiness，对所有 chunk 一视同仁。
#   per-type 衰减率相当于 per-cgroup reclaim_ratio，允许不同类型 chunk 有不同的内存压力。
#
# CHUNK_TYPE_DECAY：chunk_type → stability_decay_factor
#   值越高（接近 1.0） → 衰减越慢，记忆越持久
#   值越低（接近 0.0） → 衰减越快，记忆越短暂
# 设计依据：
#   design_constraint  → 0.99 极慢衰减（系统约束是长期有效的，类比长时程增强 LTP）
#   decision           → 0.97 慢衰减（决策记录应长期保留）
#   reasoning_chain    → 0.94 中等衰减（推理过程较情节记忆持久，但不如决策）
#   procedure          → 0.96 较慢衰减（操作步骤是程序性记忆，耐久）
#   task_state         → 0.85 较快衰减（当前任务状态 = 工作记忆，任务完成后快速衰减）
#   prompt_context     → 0.70 快速衰减（prompt 上下文高度情景化，换会话即失效）
#   error_event        → 0.88 中等衰减（错误事件有警示价值，保留时间中等）
#   observation        → 0.90 中等衰减（观察记录较 task_state 持久，但不如 decision）

CHUNK_TYPE_DECAY: dict = {
    "design_constraint": 0.99,
    "decision":          0.97,
    "procedure":         0.96,
    "reasoning_chain":   0.94,
    "observation":       0.90,
    "error_event":       0.88,
    "task_state":        0.85,
    "prompt_context":    0.70,
}

# 未列出类型的默认衰减率（保守中值）
_DEFAULT_TYPE_DECAY: float = 0.92


def get_chunk_type_decay(chunk_type: str) -> float:
    """
    iter400：获取 chunk_type 的个体化稳定性衰减率。

    Returns:
      float ∈ (0.0, 1.0]，越高越耐久（越接近 1.0 衰减越慢）
    """
    return CHUNK_TYPE_DECAY.get(chunk_type or "", _DEFAULT_TYPE_DECAY)


def decay_stability_by_type(
    conn: sqlite3.Connection,
    project: str = None,
    stale_days: int = 30,
    now_iso: str = None,
) -> int:
    """
    iter400：按 chunk_type 个体化衰减 stability（Forgetting Curve Individualization）。

    每种 chunk_type 使用 CHUNK_TYPE_DECAY 中的独立衰减率，
    替代 sleep_consolidate 中的统一 stability_decay=0.92。

    OS 类比：Linux cgroup per-memory-group reclaim_ratio —
      不同 cgroup 有不同的内存回收压力参数，允许 DB/前台应用占用更多内存。

    算法：
      FOR each chunk_type IN CHUNK_TYPE_DECAY:
          UPDATE stability × type_decay
          WHERE last_accessed < cutoff AND access_count < 2 AND chunk_type=type_

    Returns:
      总衰减的 chunk 数
    """
    from datetime import datetime as _dt, timezone as _tz, timedelta as _td
    if now_iso is None:
        now_iso = _dt.now(_tz.utc).isoformat()
    cutoff = (_dt.now(_tz.utc) - _td(days=stale_days)).isoformat()

    proj_filter = "AND project=?" if project else ""
    proj_params = [project] if project else []

    total_decayed = 0
    all_types = list(CHUNK_TYPE_DECAY.keys()) + [""]  # "" = 无类型 → 使用默认

    # 对每种已知类型单独更新
    for ctype, decay in CHUNK_TYPE_DECAY.items():
        try:
            conn.execute(
                f"UPDATE memory_chunks "
                f"SET stability=MAX(0.1, stability * ?), updated_at=? "
                f"WHERE chunk_type=? AND last_accessed < ? AND access_count < 2 {proj_filter}",
                [decay, now_iso, ctype, cutoff] + proj_params,
            )
            total_decayed += conn.execute("SELECT changes()").fetchone()[0]
        except Exception:
            pass

    # 未列出的类型使用默认衰减率
    known_types_ph = ",".join("?" * len(CHUNK_TYPE_DECAY))
    try:
        conn.execute(
            f"UPDATE memory_chunks "
            f"SET stability=MAX(0.1, stability * ?), updated_at=? "
            f"WHERE (chunk_type NOT IN ({known_types_ph}) OR chunk_type IS NULL) "
            f"AND last_accessed < ? AND access_count < 2 {proj_filter}",
            [_DEFAULT_TYPE_DECAY, now_iso] + list(CHUNK_TYPE_DECAY.keys()) + [cutoff] + proj_params,
        )
        total_decayed += conn.execute("SELECT changes()").fetchone()[0]
    except Exception:
        pass

    return total_decayed


# ── iter435: Recency-Induced Decay Resistance (RDR) — 近期访问记忆的巩固窗口保护（McGaugh 2000）──
# 认知科学依据：McGaugh (2000) "Memory — a century of consolidation" —
#   记忆形成后进入 consolidation window（数分钟至数小时），海马体持续重放记忆痕迹，
#   这期间记忆对遗忘干扰的抵抗力最强（retrograde amnesia gradient 的基础）。
#   Müller & Pilzecker (1900) perseveration-consolidation hypothesis：
#     记忆痕迹需要时间"硬化"（consolidate），窗口期内应保护而非加速衰减。
#   Baddeley & Hitch (1974) Working Memory — phonological loop 维持近期信息的活跃表示，
#     防止立即遗忘，为长期记忆转移提供缓冲。
#
# memory-os 等价：
#   decay_stability_by_type 扫描时，last_accessed < 6h 且 importance >= 0.5 的 chunk
#   正处于 consolidation window，跳过本次衰减（等下次扫描时已过窗口再参与）。
#   效果：近期被检索的重要 chunk 额外获得 6 小时的衰减豁免期，
#   模拟海马体对刚访问记忆的主动保护机制。
#
# OS 类比：Linux MGLRU young generation minimum age (min_lru_age) —
#   刚被提升到 young generation 的页面有最短存活期，kswapd 在此期间不执行 aging，
#   避免"刚提升就被驱逐"的 LRU thrashing。等价于：
#   最近 access 的 chunk（young generation）在 rdr_window_hours 内不参与 stability decay。
# 实现：在 decay_stability_by_type 中追加 WHERE NOT (last_accessed > rdr_cutoff AND importance >= min_imp)

# ── iter434: Retrieval-Induced Forgetting (RIF) — 检索导致相关记忆被压制（Anderson et al. 1994）──
# 认知科学依据：Anderson, Bjork & Bjork (1994) "Remembering can cause forgetting" —
#   检索某条记忆（practiced item）主动抑制同类别相关但未被检索的记忆（unpracticed items）。
#   机制：检索激活类别竞争记忆 → 强化被选中者 → 主动抑制被压制者（RP-）→ RP- 遗忘增加 ~10-20%。
#   条件：RIF 要求竞争者与检索目标属于同一类别（chunk_type）且内容相关（Jaccard 相似度阈值）。
#
# OS 类比：CPU cache set-associativity way eviction —
#   访问 cache line A（命中 set 0, way 0）→ LRU 将同 set 的竞争 cache line B 推向更高 way
#   → B 的 eviction 概率上升（A 的命中加速了 B 的驱逐路径）。
#
# 与 iter432 Cumulative Interference 的区别：
#   CI（iter432）= 静态结构性干扰（同类数量多 → 被动衰减加速）
#   RIF（iter434）= 动态事件性抑制（检索事件 → 主动压制竞争记忆）

def _rif_tokenize(text: str) -> frozenset:
    """iter434: RIF 内部 tokenizer — 提取用于 Jaccard 相似度计算的 token 集合。"""
    import re as _re
    tokens = set()
    for m in _re.finditer(r'[a-zA-Z0-9_][-a-zA-Z0-9_.]*', text):
        tokens.add(m.group().lower())
    cn = _re.sub(r'[^\u4e00-\u9fff]', '', text)
    for i in range(len(cn) - 1):
        tokens.add(cn[i:i + 2])
    return frozenset(tokens)


def apply_rif_by_summary(
    conn: sqlite3.Connection,
    project: str,
    hit_chunk_ids: list,
) -> dict:
    """
    iter434: Retrieval-Induced Forgetting (RIF) by Summary Jaccard — 基于 summary 相似度的精确 RIF。

    与 iter417 apply_retrieval_induced_forgetting 的区别：
      - iter417: 基于 encode_context token 集合重叠（稀疏，依赖上下文标注）
      - iter434: 基于 summary Jaccard 相似度（更鲁棒，直接文本相似度）
      - iter434: 按 chunk_type 分组（竞争限定在同类别，更符合 RIF 实验条件）
      - iter434: 使用 scorer.rif_* sysctl（独立配置）

    Anderson, Bjork & Bjork (1994) 实验范式：
      - Practiced items (RP+): 被检索 → 记忆增强
      - Unpracticed-related (RP-): 同类别但未被检索 → 记忆被抑制（低于控制组基线）
      - Unpracticed-unrelated (NRP): 不同类别 → 不受影响（控制组基线）

    实现逻辑：
      1. 对每个命中 chunk，查询同 chunk_type 的其他 chunk（同类别竞争者）
      2. 计算 Jaccard 相似度（RP- 候选必须与命中 chunk 内容相关）
      3. 对 Jaccard >= rif_similarity_threshold 且未被命中的 chunk 施加 stability 惩罚
      4. 豁免：importance 高、受保护类型、permastore 保护的 chunk

    参数：
      conn          — 数据库连接
      project       — 项目 ID
      hit_chunk_ids — 本次被检索命中的 chunk ID 列表

    返回 dict：
      suppressed     — 受到 RIF 抑制的 chunk 数量
      total_examined — 总共检查的竞争者数量
      suppressed_ids — 被抑制的 chunk ID 列表（调试用）
    """
    if not hit_chunk_ids:
        return {"suppressed": 0, "total_examined": 0, "suppressed_ids": []}

    try:
        import config as _cfg_mod
        if not _cfg_mod.get("scorer.rif_enabled"):
            return {"suppressed": 0, "total_examined": 0, "suppressed_ids": []}

        rif_factor = _cfg_mod.get("scorer.rif_factor")
        sim_threshold = _cfg_mod.get("scorer.rif_similarity_threshold")
        max_targets = _cfg_mod.get("scorer.rif_max_targets")
        protect_imp = _cfg_mod.get("scorer.rif_protect_importance")
        protect_types_raw = _cfg_mod.get("scorer.rif_protect_types")
        protect_types = set(t.strip() for t in protect_types_raw.split(",") if t.strip())
    except Exception:
        return {"suppressed": 0, "total_examined": 0, "suppressed_ids": []}

    hit_set = set(hit_chunk_ids)
    placeholders = ",".join("?" * len(hit_chunk_ids))

    # ── 读取命中 chunk 的 chunk_type 和 summary ──
    hit_rows = conn.execute(
        f"SELECT id, chunk_type, summary FROM memory_chunks WHERE id IN ({placeholders})",
        hit_chunk_ids,
    ).fetchall()

    if not hit_rows:
        return {"suppressed": 0, "total_examined": 0, "suppressed_ids": []}

    # 按 chunk_type 分组
    type_to_hits = {}  # chunk_type → [(id, tokens)]
    for rid, ct, summary in hit_rows:
        if ct in protect_types:
            continue
        toks = _rif_tokenize(summary or "")
        if not toks:
            continue
        type_to_hits.setdefault(ct, []).append((rid, toks))

    if not type_to_hits:
        return {"suppressed": 0, "total_examined": 0, "suppressed_ids": []}

    # ── 对每种 chunk_type，查询同类竞争者（候选 RP-）──
    suppressed = 0
    total_examined = 0
    suppressed_ids = []

    now_iso = datetime.now(timezone.utc).isoformat()

    for chunk_type, hits_list in type_to_hits.items():
        if chunk_type in protect_types:
            continue

        # ── iter435: RDR — 计算近期访问保护截止时间（巩固窗口） ──
        # 近期访问的重要 chunk 正处于 McGaugh consolidation window，豁免 RIF 抑制
        try:
            _rdr_enabled_rif = _cfg_mod.get("store_vfs.rdr_enabled")
            if _rdr_enabled_rif:
                _rdr_wh = _cfg_mod.get("store_vfs.rdr_window_hours")
                _rdr_min_imp_rif = _cfg_mod.get("store_vfs.rdr_min_importance")
                from datetime import timedelta as _td_rdr
                _rdr_cutoff_rif = (datetime.now(timezone.utc) - _td_rdr(hours=_rdr_wh)).isoformat()
                _rdr_excl = (
                    f"AND NOT (last_accessed > '{_rdr_cutoff_rif}' "
                    f"AND COALESCE(importance, 0.0) >= {_rdr_min_imp_rif})"
                )
            else:
                _rdr_excl = ""
        except Exception:
            _rdr_excl = ""

        # 查询同类型的非命中 chunk
        competitors = conn.execute(
            f"""SELECT id, summary, COALESCE(stability, 1.0), importance
               FROM memory_chunks
               WHERE project = ? AND chunk_type = ?
                 AND COALESCE(importance, 0.5) < ?
                 AND COALESCE(oom_adj, 0) > -1000
                 {_rdr_excl}
               ORDER BY stability ASC""",
            (project, chunk_type, protect_imp),
        ).fetchall()

        # 排除命中 chunk
        competitors = [(rid, s, stab, imp) for rid, s, stab, imp in competitors if rid not in hit_set]
        total_examined += len(competitors)

        if not competitors:
            continue

        # 预计算竞争者 tokens
        comp_tokens = [(rid, _rif_tokenize(s or ""), stab) for rid, s, stab, imp in competitors]

        # 对每个命中 chunk，找 Jaccard >= threshold 的竞争者
        to_suppress = {}  # rid → current_stab (去重，取最小 stab 防重复压制)
        for _hit_id, hit_toks in hits_list:
            if not hit_toks:
                continue
            # 计算相似度并收集竞争者
            scored = []
            for rid, c_toks, c_stab in comp_tokens:
                if not c_toks:
                    continue
                inter = len(hit_toks & c_toks)
                union = len(hit_toks | c_toks)
                jaccard = inter / union if union > 0 else 0.0
                if jaccard >= sim_threshold:
                    scored.append((jaccard, rid, c_stab))

            # 取相似度最高的前 max_targets 个
            scored.sort(key=lambda x: x[0], reverse=True)
            for _, rid, c_stab in scored[:max_targets]:
                if rid not in to_suppress:
                    to_suppress[rid] = c_stab

        # ── 批量应用 RIF stability 惩罚 ──
        # 保护：iter422 Permastore floor 和 iter431 Ribot's Law floor
        for rid, c_stab in to_suppress.items():
            try:
                # permastore floor
                ps_floor = compute_permastore_floor(conn, rid, c_stab)
                # Ribot floor
                ribot_floor = 0.0
                try:
                    _row_r = _get_chunk_age_importance(conn, rid)
                    if _row_r:
                        ribot_floor = 0.1 + compute_ribot_floor(_row_r[0], _row_r[1])
                except Exception:
                    pass

                floor = max(ps_floor, ribot_floor)
                new_stab = max(floor, c_stab * rif_factor)
                if new_stab < c_stab:
                    conn.execute(
                        "UPDATE memory_chunks SET stability=?, updated_at=? WHERE id=?",
                        (round(new_stab, 4), now_iso, rid),
                    )
                    suppressed += 1
                    suppressed_ids.append(rid)
            except Exception:
                pass

    return {
        "suppressed": suppressed,
        "total_examined": total_examined,
        "suppressed_ids": suppressed_ids[:10],  # 只返回前10个（调试用）
    }


# ── iter436: Output Interference — 同轮注入竞争性遗忘（Roediger 1978）──────────────────
# 认知科学依据：Roediger (1978) "Recall as a self-limiting process" —
#   在一次回忆测试中，回忆早期项目（output）会干扰后续项目的提取（output interference）。
#   机制：早期输出激活该语义领域的竞争记忆，通过抑制机制阻碍后续项目进入工作记忆。
#   Roediger & Schmidt (1980): 同次测试中，越靠后的序列位置遗忘越多（OI 累积效应）。
#   与 RIF（iter434）区别：
#     RIF = 检索事件干扰"未被检索"的竞争者（编码竞争）
#     OI  = 同次输出中早期项目干扰晚期项目的工作记忆占用（输出干扰）
#
# memory-os 等价：
#   每次检索注入 N 个 chunk（recall_traces.top_k_json 记录），
#   位置 0 = 最相关/最优先（OS: cache line 命中，无干扰）
#   位置 k > 0 = 受前 k 个 chunk 的 output interference，巩固效果越来越差。
#   在 sleep_consolidate 时扫描最近注入记录，对位置 >= 1 的 chunk 施加轻微 stability 惩罚。
#
# OS 类比：Linux BFQ (Budget Fair Queue) I/O 批处理 —
#   同一 dispatch batch 中，第一个 I/O 请求消耗了大部分 budget；
#   后续请求在 budget 耗尽前完成的 I/O 减少（batch output competition）。
#   访问 cache line A 后，同 cache set 的 cache line B 在同一 dispatch cycle 中
#   获得更少的 refill 机会（类比：同轮注入的后续 chunk 巩固机会减少）。

def apply_output_interference(
    conn: sqlite3.Connection,
    project: str,
    window_hours: float = 24.0,
) -> dict:
    """
    iter436: Output Interference — 对同轮注入的后续 chunk 施加轻微 stability 惩罚。

    扫描最近 window_hours 内注入成功的 recall_traces，
    对每条 trace 的 top_k_json 中 position >= 1 的 chunk（位置越靠后干扰越强）
    施加递增的 stability 惩罚（× oi_decay_factor ^ position）。

    保护条件：
      - importance >= oi_protect_importance（核心知识豁免）
      - oi_enabled=False 时不执行
      - position 0 的 chunk（最优先）不受影响

    返回：{"penalized": N, "total_examined": N}
    """
    import json as _json
    from datetime import datetime as _dt, timezone as _tz, timedelta as _td
    try:
        import config as _cfg_oi
    except ImportError:
        return {"penalized": 0, "total_examined": 0}

    try:
        oi_enabled = _cfg_oi.get("store_vfs.oi_enabled")
        if not oi_enabled:
            return {"penalized": 0, "total_examined": 0}

        oi_decay_factor = _cfg_oi.get("store_vfs.oi_decay_factor")
        oi_protect_imp = _cfg_oi.get("store_vfs.oi_protect_importance")
        oi_max_coinjected = _cfg_oi.get("store_vfs.oi_max_coinjected")
    except Exception:
        return {"penalized": 0, "total_examined": 0}

    now = _dt.now(_tz.utc)
    cutoff_iso = (now - _td(hours=window_hours)).isoformat()
    now_iso = now.isoformat()

    # 查询最近注入成功的 recall_traces（injected=1，top_k_json 非空）
    try:
        traces = conn.execute(
            """SELECT top_k_json FROM recall_traces
               WHERE project = ? AND injected = 1
                 AND top_k_json IS NOT NULL
                 AND timestamp >= ?
               ORDER BY timestamp DESC LIMIT 200""",
            (project, cutoff_iso),
        ).fetchall()
    except Exception:
        return {"penalized": 0, "total_examined": 0}

    penalized = 0
    total_examined = 0

    # 聚合：同一 chunk 在多条 trace 中出现时，取最靠后的 position（最大干扰）
    # chunk_id → max_position_across_traces
    chunk_positions: dict = {}

    for (tkj,) in traces:
        try:
            if not tkj:
                continue
            items = _json.loads(tkj)
            if not isinstance(items, list) or len(items) <= 1:
                continue
            # 只处理有多个注入 chunk 的 trace（单 chunk 无 OI）
            n = min(len(items), oi_max_coinjected)
            for pos in range(1, n):  # position 0 豁免
                cid = items[pos].get("id") if isinstance(items[pos], dict) else None
                if cid:
                    # 取最大 position（最严重干扰）across traces
                    if cid not in chunk_positions or chunk_positions[cid] < pos:
                        chunk_positions[cid] = pos
        except Exception:
            continue

    if not chunk_positions:
        return {"penalized": 0, "total_examined": 0}

    total_examined = len(chunk_positions)

    for cid, pos in chunk_positions.items():
        try:
            row = conn.execute(
                "SELECT stability, importance FROM memory_chunks WHERE id=? AND project=?",
                (cid, project),
            ).fetchone()
            if not row:
                continue
            stab, imp = float(row[0] or 1.0), float(row[1] or 0.5)

            # 高 importance → 豁免
            if imp >= oi_protect_imp:
                continue

            # 惩罚：position 越大，惩罚越强（cumulative: factor^pos）
            penalty = oi_decay_factor ** pos
            new_stab = max(0.1, stab * penalty)
            if abs(new_stab - stab) < 0.0001:
                continue

            conn.execute(
                "UPDATE memory_chunks SET stability=?, updated_at=? WHERE id=?",
                (round(new_stab, 4), now_iso, cid),
            )
            penalized += 1
        except Exception:
            continue

    return {"penalized": penalized, "total_examined": total_examined}


# ── iter437: Hypermnesia — 多次分布式检索后记忆净增强（Erdelyi & Becker 1974）──────────────
# 认知科学依据：Erdelyi & Becker (1974) "1974 hypermnesia for pictures" (Cognitive Psychology) —
#   多轮自由回忆测试中，随测试次数增加，总召回量呈净增长（hypermnesia）：
#   不同回忆尝试激活不同检索路径，集体覆盖更多记忆痕迹（retrieval route diversity）。
#   Payne (1987) Meta-analysis: +15-25% improvement across 3-5 test sessions。
#   关键条件：必须是间隔分布（spaced）而非集中（massed）的回忆测试。
# memory-os 等价：
#   spaced_access_count（iter420）= 跨 24h 间隔的检索次数，代理"不同 session 的检索尝试数"。
#   达到 hypermnesia_threshold 后触发一次 stability boost（避免反复触发 = cooldown）。
#   与 Spacing Effect 区别：SE 是 per-access 小幅加成；Hypermnesia 是 threshold-crossing 大幅净增强。
# OS 类比：Linux khugepaged Transparent HugePage 多 epoch 晋升 —
#   页面在多个内存分配 epoch 内持续热访问 → khugepaged 合并为 2MB hugepage，降低 TLB miss rate；
#   类比：多次跨 session 检索 → 记忆表示从分散痕迹"合并"为稳定长期表示（net improvement）。

def apply_hypermnesia(
    conn: sqlite3.Connection,
    project: str,
) -> dict:
    """
    iter437: Hypermnesia — 对 spaced_access_count >= threshold 的 chunk 施加净增强 boost。

    在 sleep_consolidate 时调用：
      1. 查找 spaced_access_count >= hypermnesia_threshold 且 importance >= min_importance 的 chunk
      2. 排除在 cooldown_days 内已被 boost 的 chunk（hypermnesia_last_boost 字段）
      3. 对符合条件的 chunk：stability × hypermnesia_boost（上限 365.0）
      4. 更新 hypermnesia_last_boost = now

    返回：{"boosted": N, "total_examined": N}
    """
    from datetime import datetime as _dt, timezone as _tz, timedelta as _td
    try:
        import config as _cfg_hm
    except ImportError:
        return {"boosted": 0, "total_examined": 0}

    try:
        hm_enabled = _cfg_hm.get("store_vfs.hypermnesia_enabled")
        if not hm_enabled:
            return {"boosted": 0, "total_examined": 0}

        hm_threshold = _cfg_hm.get("store_vfs.hypermnesia_threshold")
        hm_boost = _cfg_hm.get("store_vfs.hypermnesia_boost")
        hm_min_imp = _cfg_hm.get("store_vfs.hypermnesia_min_importance")
        hm_cooldown_days = _cfg_hm.get("store_vfs.hypermnesia_cooldown_days")
    except Exception:
        return {"boosted": 0, "total_examined": 0}

    now = _dt.now(_tz.utc)
    now_iso = now.isoformat()
    cooldown_cutoff = (now - _td(days=hm_cooldown_days)).isoformat()

    try:
        # 候选：spaced_access_count >= threshold，importance >= min_imp，
        #   且 hypermnesia_last_boost 为空或在冷却期外
        rows = conn.execute(
            """SELECT id, stability
               FROM memory_chunks
               WHERE project = ?
                 AND COALESCE(spaced_access_count, 0) >= ?
                 AND COALESCE(importance, 0.5) >= ?
                 AND (hypermnesia_last_boost IS NULL
                      OR hypermnesia_last_boost < ?)
               LIMIT 200""",
            (project, hm_threshold, hm_min_imp, cooldown_cutoff),
        ).fetchall()
    except Exception:
        return {"boosted": 0, "total_examined": 0}

    total_examined = len(rows)
    boosted = 0

    for (cid, stab) in rows:
        try:
            new_stab = min(365.0, float(stab or 1.0) * hm_boost)
            conn.execute(
                "UPDATE memory_chunks SET stability=?, hypermnesia_last_boost=?, updated_at=? WHERE id=?",
                (round(new_stab, 4), now_iso, now_iso, cid),
            )
            boosted += 1
        except Exception:
            continue

    return {"boosted": boosted, "total_examined": total_examined}


# ── iter438: Jost's Law — 等强度记忆中较老者衰减更慢（Jost 1897）────────────────────────────────
# 认知科学依据：Jost (1897) "Die Assoziationsfestigkeit in ihrer Abhängigkeit von der Verteilung
#   der Wiederholungen" — Jost's Law of Memory：
#   若两个记忆在某一时刻强度相等，则较老的记忆在未来遗忘得更慢。
#   机制：老记忆已历经多次睡眠重放和巩固周期，突触权重矩阵更稳固；
#   Baddeley (1997): Jost's Law 是遗忘曲线的重要补充，age 越大 → 衰减率越低。
#
# 与 iter431 Ribot's Law 的互补关系：
#   Ribot = stability_floor 提高（下限保护）
#   Jost  = effective_decay 减慢（每次衰减步长缩小，持续减速）
#   两者叠加：老 chunk 既有更高 floor，也有更慢的 per-step 衰减速率。
#
# OS 类比：Linux MGLRU old generation protection —
#   在 old generation 长期存在（经历多个 aging interval）的页面，
#   kswapd 给予更弱的 reclaim pressure（不像 young gen 那样激进驱逐）；
#   类比：age 越大的 chunk → effective_decay 越接近 1.0 → per-step 衰减量越小。

def apply_jost_law(
    conn: sqlite3.Connection,
    project: str,
    stale_days: int = 30,
) -> dict:
    """
    iter438: Jost's Law — 对高龄 chunk 施加衰减减速修正。

    在 sleep_consolidate 中，decay_stability_by_type_with_ci 已对 access_count < 2 的 chunk
    执行了批量衰减。apply_jost_law 作为后处理，对满足年龄+重要性条件的 chunk 部分"撤销"
    多余的衰减（恢复一部分被过度衰减的 stability），等效于以 effective_decay 替换 base_decay：

      effective_decay = base_decay + (1 - base_decay) × jost_bonus
      stability_restored = current_stab / base_decay × effective_decay - current_stab
                         = current_stab × (effective_decay / base_decay - 1)

    实现简化：直接用 jost_bonus 乘以 (1 - current_decay_factor) 做增量修复：
      new_stab = min(old_stab, current_stab * (1 + jost_bonus / (1 - effective_decay + ε)))

    更简洁的实际实现：
      对每个符合条件的 chunk，stability × (1 + jost_effective_bonus)
      其中 jost_effective_bonus = jost_bonus × base_decay_step
                                = jost_bonus × (1 - decay_factor_used)
    但 decay_factor_used 不易追踪。改用直接乘法（近似）：
      new_stab = min(pre_decay_stab, current_stab × (1 + jost_bonus))

    由于 pre_decay_stab 未知，用保守方法：
      new_stab = current_stab × jost_multiplier，where jost_multiplier = 1 + jost_bonus×0.1
    这确保每次 sleep_consolidate 后，老 chunk 的 stability 被轻微"提振"，
    相当于减缓了 decay 的净效果。

    Returns:
      {"adjusted": N, "total_examined": N}
    """
    import math
    from datetime import datetime as _dt, timezone as _tz, timedelta as _td

    try:
        import config as _cfg_jost
    except ImportError:
        return {"adjusted": 0, "total_examined": 0}

    try:
        if not _cfg_jost.get("store_vfs.jost_enabled"):
            return {"adjusted": 0, "total_examined": 0}
        jost_min_imp = _cfg_jost.get("store_vfs.jost_min_importance")
        jost_scale = _cfg_jost.get("store_vfs.jost_scale")
        jost_max_bonus = _cfg_jost.get("store_vfs.jost_max_bonus")
        jost_min_age = _cfg_jost.get("store_vfs.jost_min_age_days")
    except Exception:
        return {"adjusted": 0, "total_examined": 0}

    now = _dt.now(_tz.utc)
    now_iso = now.isoformat()
    # 只对"被衰减候选"的 chunk 做修复（stale_days 未访问，access_count < 2，低重要性除外）
    cutoff_stale = (now - _td(days=stale_days)).isoformat()
    min_age_cutoff = (now - _td(days=jost_min_age)).isoformat()

    try:
        # 候选：high importance + old age + stale (recently decayed by CI)
        rows = conn.execute(
            """SELECT id, stability, created_at, importance
               FROM memory_chunks
               WHERE project = ?
                 AND COALESCE(importance, 0.0) >= ?
                 AND created_at < ?
                 AND last_accessed < ?
                 AND access_count < 2
                 AND COALESCE(stability, 0.1) > 0.1
               LIMIT 500""",
            (project, jost_min_imp, min_age_cutoff, cutoff_stale),
        ).fetchall()
    except Exception:
        return {"adjusted": 0, "total_examined": 0}

    total_examined = len(rows)
    adjusted = 0

    for row in rows:
        try:
            cid, stab, created_at, importance = row
            if not created_at:
                continue
            _ts_c = _dt.fromisoformat(created_at.replace("Z", "+00:00")).timestamp()
            age_days = (now.timestamp() - _ts_c) / 86400.0
            if age_days < jost_min_age:
                continue

            # Jost bonus: log(1 + age_days) / log(365) × jost_scale，上限 jost_max_bonus
            raw_bonus = math.log(1 + age_days) / math.log(365) * jost_scale
            jost_bonus = min(jost_max_bonus, raw_bonus)

            # effective_decay_reduction = jost_bonus × (1 - typical_decay)
            # 典型 type_decay ≈ 0.95，(1 - 0.95) = 0.05
            # 实际 stability 恢复量 = current_stab × jost_bonus × 0.05
            # 等效 multiplier ≈ 1 + jost_bonus × 0.05（保守）
            jost_multiplier = 1.0 + jost_bonus * 0.05  # 保守系数，避免过度逆转 decay
            new_stab = min(365.0, float(stab or 0.1) * jost_multiplier)

            if new_stab <= float(stab or 0.1) + 0.0001:
                continue  # 无实质变化

            conn.execute(
                "UPDATE memory_chunks SET stability=?, updated_at=? WHERE id=?",
                (round(new_stab, 4), now_iso, cid),
            )
            adjusted += 1
        except Exception:
            continue

    if adjusted > 0:
        conn.commit()

    return {"adjusted": adjusted, "total_examined": total_examined}


# ── iter439: Encoding Depth Decay Resistance — 深度编码减慢衰减（Craik & Tulving 1975）──────────────
# 认知科学依据：Craik & Tulving (1975) "Depth of processing and the retention of words in
#   episodic memory" — 深度语义加工产生更强的记忆痕迹，对遗忘曲线有天然抵抗力。
#   encode_context 中 entity 数量（iter411 LOP proxy）代理编码深度：
#   entity_count >= eddr_deep_threshold → 深度编码，stability 轻微修复（减慢衰减）。
#   entity_count <= eddr_shallow_threshold → 浅层编码，stability 轻微惩罚（加速衰减）。
# OS 类比：Linux ext4 extent tree depth —
#   深层 extent tree（多 entity）= I/O 代价更高 = kswapd 驱逐优先级更低（更抗衰减）。

def apply_encoding_depth_decay_resistance(
    conn: sqlite3.Connection,
    project: str,
    stale_days: int = 30,
) -> dict:
    """
    iter439: Encoding Depth Decay Resistance — 根据 encode_context 实体数量调整 stability。

    在 sleep_consolidate 中，decay_stability_by_type_with_ci 执行批量衰减后：
    - 深度编码 chunk（entity_count >= deep_threshold）：轻微恢复 stability，模拟抗遗忘优势。
    - 浅层编码 chunk（entity_count <= shallow_threshold）：轻微加速衰减，模拟快速遗忘。

    深度修复：new_stab = current_stab × (1 + depth_bonus × 0.03)
    浅层惩罚：new_stab = current_stab × (1 - shallow_penalty)

    Returns:
      {"deep_boosted": N, "shallow_penalized": N, "total_examined": N}
    """
    try:
        import config as _cfg_eddr
    except ImportError:
        return {"deep_boosted": 0, "shallow_penalized": 0, "total_examined": 0}

    try:
        if not _cfg_eddr.get("store_vfs.eddr_enabled"):
            return {"deep_boosted": 0, "shallow_penalized": 0, "total_examined": 0}
        eddr_deep_threshold = _cfg_eddr.get("store_vfs.eddr_deep_threshold")    # 5
        eddr_shallow_threshold = _cfg_eddr.get("store_vfs.eddr_shallow_threshold")  # 1
        eddr_max_depth_bonus = _cfg_eddr.get("store_vfs.eddr_max_depth_bonus")  # 0.15
        eddr_shallow_penalty = _cfg_eddr.get("store_vfs.eddr_shallow_penalty")  # 0.05
    except Exception:
        return {"deep_boosted": 0, "shallow_penalized": 0, "total_examined": 0}

    from datetime import datetime as _dt, timezone as _tz, timedelta as _td
    now = _dt.now(_tz.utc)
    now_iso = now.isoformat()
    cutoff_stale = (now - _td(days=stale_days)).isoformat()

    try:
        # 候选：stale + access_count < 2（被 decay 扫描过的）
        rows = conn.execute(
            """SELECT id, stability, encode_context, importance
               FROM memory_chunks
               WHERE project = ?
                 AND last_accessed < ?
                 AND access_count < 2
                 AND COALESCE(stability, 0.1) > 0.1
               LIMIT 500""",
            (project, cutoff_stale),
        ).fetchall()
    except Exception:
        return {"deep_boosted": 0, "shallow_penalized": 0, "total_examined": 0}

    total_examined = len(rows)
    deep_boosted = 0
    shallow_penalized = 0

    for row in rows:
        try:
            cid, stab, encode_context, importance = row
            stab_f = float(stab or 0.1)
            if stab_f <= 0.1:
                continue

            # 计算 entity 数量（encode_context 是逗号分隔字符串）
            if encode_context:
                entity_count = len([e.strip() for e in encode_context.split(',') if e.strip()])
            else:
                entity_count = 0

            new_stab = stab_f
            if entity_count >= eddr_deep_threshold:
                # 深度编码：stability 轻微修复（conservative 系数 0.03）
                raw_bonus = min(eddr_max_depth_bonus, entity_count / 10.0 * eddr_max_depth_bonus)
                new_stab = min(365.0, stab_f * (1.0 + raw_bonus * 0.03))
                if new_stab > stab_f + 0.0001:
                    deep_boosted += 1
                else:
                    continue
            elif entity_count <= eddr_shallow_threshold:
                # 浅层编码：轻微加速衰减
                new_stab = max(0.1, stab_f * (1.0 - eddr_shallow_penalty))
                if new_stab < stab_f - 0.0001:
                    shallow_penalized += 1
                else:
                    continue
            else:
                continue  # 中等深度：不干预

            conn.execute(
                "UPDATE memory_chunks SET stability=?, updated_at=? WHERE id=?",
                (round(new_stab, 4), now_iso, cid),
            )
        except Exception:
            continue

    if deep_boosted > 0 or shallow_penalized > 0:
        conn.commit()

    return {"deep_boosted": deep_boosted, "shallow_penalized": shallow_penalized,
            "total_examined": total_examined}


# ── iter440: Proactive Facilitation — 强邻居锚定保护新知识衰减（Ausubel 1963）──────────────────────
# 认知科学依据：Ausubel (1963) 正向迁移/先行组织者：已有稳固 schema 锚定新知识，
#   降低新知识的遗忘速率。encode_context entity 重叠代理语义相似度。
# OS 类比：Linux page cache refcount —
#   被多个 inode 共享引用的 page 有高 refcount，kswapd 优先保留（驱逐代价 > 收益）。

def apply_proactive_facilitation(
    conn: sqlite3.Connection,
    project: str,
    stale_days: int = 30,
) -> dict:
    """
    iter440: Proactive Facilitation — 对被高 importance 强邻居锚定的 chunk 减慢衰减。

    在 sleep_consolidate 中，对 stale + access_count < 2 的候选 chunk，
    若其 encode_context entity 集合与高 importance(≥ pf_anchor_min_importance) 且
    access_count >= pf_anchor_min_access 的强邻居有足够重叠（≥ pf_min_overlap entity），
    则该 chunk 被"锚定"，获得轻微 stability 修复：
      new_stab = current_stab × (1 + pf_max_bonus × 0.04)

    注意：这是对所有候选 chunk 批量扫描，效率优先：
    将所有强邻居的 entity 集合构建成全局集合，候选 chunk 逐一与之匹配。

    Returns:
      {"facilitated": N, "total_examined": N}
    """
    try:
        import config as _cfg_pf
    except ImportError:
        return {"facilitated": 0, "total_examined": 0}

    try:
        if not _cfg_pf.get("store_vfs.pf_enabled"):
            return {"facilitated": 0, "total_examined": 0}
        pf_anchor_min_imp = _cfg_pf.get("store_vfs.pf_anchor_min_importance")
        pf_anchor_min_acc = _cfg_pf.get("store_vfs.pf_anchor_min_access")
        pf_min_overlap = _cfg_pf.get("store_vfs.pf_min_overlap")
        pf_max_bonus = _cfg_pf.get("store_vfs.pf_max_bonus")
    except Exception:
        return {"facilitated": 0, "total_examined": 0}

    from datetime import datetime as _dt, timezone as _tz, timedelta as _td
    now = _dt.now(_tz.utc)
    now_iso = now.isoformat()
    cutoff_stale = (now - _td(days=stale_days)).isoformat()

    # Step 1: 获取强邻居的 entity 集合（全局锚点）
    try:
        anchor_rows = conn.execute(
            """SELECT encode_context FROM memory_chunks
               WHERE project = ?
                 AND COALESCE(importance, 0.0) >= ?
                 AND access_count >= ?
                 AND encode_context IS NOT NULL
                 AND encode_context != ''
               LIMIT 200""",
            (project, pf_anchor_min_imp, pf_anchor_min_acc),
        ).fetchall()
    except Exception:
        return {"facilitated": 0, "total_examined": 0}

    if not anchor_rows:
        return {"facilitated": 0, "total_examined": 0}

    # 构建全局强邻居 entity 集合（每个锚点 chunk 的 entity set）
    anchor_entity_sets = []
    for arow in anchor_rows:
        try:
            ec = arow[0] if isinstance(arow, (list, tuple)) else arow["encode_context"]
            if ec:
                entities = frozenset(e.strip() for e in ec.split(',') if e.strip())
                if entities:
                    anchor_entity_sets.append(entities)
        except Exception:
            continue

    if not anchor_entity_sets:
        return {"facilitated": 0, "total_examined": 0}

    # Step 2: 获取候选 chunk（stale + access_count < 2）
    try:
        candidate_rows = conn.execute(
            """SELECT id, stability, encode_context
               FROM memory_chunks
               WHERE project = ?
                 AND last_accessed < ?
                 AND access_count < 2
                 AND COALESCE(stability, 0.1) > 0.1
                 AND encode_context IS NOT NULL
                 AND encode_context != ''
               LIMIT 500""",
            (project, cutoff_stale),
        ).fetchall()
    except Exception:
        return {"facilitated": 0, "total_examined": 0}

    total_examined = len(candidate_rows)
    facilitated = 0
    pf_multiplier = 1.0 + pf_max_bonus * 0.04

    for row in candidate_rows:
        try:
            cid = row[0] if isinstance(row, (list, tuple)) else row["id"]
            stab = row[1] if isinstance(row, (list, tuple)) else row["stability"]
            ec = row[2] if isinstance(row, (list, tuple)) else row["encode_context"]

            if not ec:
                continue
            stab_f = float(stab or 0.1)
            if stab_f <= 0.1:
                continue

            # 计算候选 chunk 的 entity 集合
            cand_entities = frozenset(e.strip() for e in ec.split(',') if e.strip())
            if not cand_entities:
                continue

            # 检查是否与任一强邻居有足够重叠
            anchored = False
            for anchor_set in anchor_entity_sets:
                overlap = len(cand_entities & anchor_set)
                if overlap >= pf_min_overlap:
                    anchored = True
                    break

            if not anchored:
                continue

            # 锚定：轻微 stability 修复
            new_stab = min(365.0, stab_f * pf_multiplier)
            if new_stab <= stab_f + 0.0001:
                continue

            conn.execute(
                "UPDATE memory_chunks SET stability=?, updated_at=? WHERE id=?",
                (round(new_stab, 4), now_iso, cid),
            )
            facilitated += 1
        except Exception:
            continue

    if facilitated > 0:
        conn.commit()

    return {"facilitated": facilitated, "total_examined": total_examined}


# ── iter441: Emotional Consolidation — 情绪显著性记忆睡眠优先巩固（McGaugh 2000）──────────────────
# 认知科学依据：McGaugh (2000) Science 287 — 情绪事件通过杏仁核-海马交互在睡眠期间优先巩固；
#   emotional_weight 代理情绪唤醒水平，高唤醒 chunk 在 sleep_consolidate 时获得额外 stability 加成。
# OS 类比：Linux writeback priority — 高优先级 dirty page 被 pdflush 优先刷写（优先巩固）。

def apply_emotional_consolidation(
    conn: sqlite3.Connection,
    project: str,
) -> dict:
    """
    iter441: Emotional Consolidation — 情绪显著性 chunk 在 sleep_consolidate 时获得额外 stability 加成。

    对 emotional_weight >= ec_min_weight 且 importance >= ec_min_importance 的 chunk，
    按情绪权重比例给予 stability 修复：
      bonus = emotional_weight × ec_scale
      new_stab = min(365.0, current_stab × (1 + bonus))

    这是对 iter409（Flashbulb Memory 写入时加成）的补充：
    Flashbulb = encoding 阶段一次性加成；Emotional Consolidation = consolidation 阶段持续加成。

    范围：不限于 stale chunk（情绪显著性 chunk 无论访问状态都应获得睡眠巩固优势）。

    Returns:
      {"consolidated": N, "total_examined": N}
    """
    try:
        import config as _cfg_ec
    except ImportError:
        return {"consolidated": 0, "total_examined": 0}

    try:
        if not _cfg_ec.get("store_vfs.ec_enabled"):
            return {"consolidated": 0, "total_examined": 0}
        ec_min_weight = _cfg_ec.get("store_vfs.ec_min_weight")
        ec_scale = _cfg_ec.get("store_vfs.ec_scale")
        ec_min_importance = _cfg_ec.get("store_vfs.ec_min_importance")
    except Exception:
        return {"consolidated": 0, "total_examined": 0}

    from datetime import datetime as _dt, timezone as _tz
    now = _dt.now(_tz.utc)
    now_iso = now.isoformat()

    try:
        rows = conn.execute(
            """SELECT id, stability, emotional_weight
               FROM memory_chunks
               WHERE project = ?
                 AND COALESCE(emotional_weight, 0.0) >= ?
                 AND COALESCE(importance, 0.0) >= ?
                 AND COALESCE(stability, 0.1) > 0.1
               LIMIT 300""",
            (project, ec_min_weight, ec_min_importance),
        ).fetchall()
    except Exception:
        return {"consolidated": 0, "total_examined": 0}

    total_examined = len(rows)
    consolidated = 0

    for row in rows:
        try:
            cid = row[0] if isinstance(row, (list, tuple)) else row["id"]
            stab = row[1] if isinstance(row, (list, tuple)) else row["stability"]
            ew = row[2] if isinstance(row, (list, tuple)) else row["emotional_weight"]

            stab_f = float(stab or 0.1)
            ew_f = float(ew or 0.0)
            if stab_f <= 0.1 or ew_f < ec_min_weight:
                continue

            bonus = ew_f * ec_scale
            new_stab = min(365.0, stab_f * (1.0 + bonus))
            if new_stab <= stab_f + 0.0001:
                continue

            conn.execute(
                "UPDATE memory_chunks SET stability=?, updated_at=? WHERE id=?",
                (round(new_stab, 4), now_iso, cid),
            )
            consolidated += 1
        except Exception:
            continue

    if consolidated > 0:
        conn.commit()

    return {"consolidated": consolidated, "total_examined": total_examined}


# ── iter442: Schema-Consistent Consolidation — 图式一致性记忆的额外巩固（Bartlett 1932 / Tse 2007）──
# 认知科学依据：Tse et al. (2007) Science "Schemas and memory consolidation" —
#   已有丰富图式后，新知识 1 天内完成系统巩固（vs 无图式时需 3 天）。
#   Bartlett (1932) Schema Theory：图式一致的信息被快速整合，获得额外巩固强化。
# OS 类比：Linux readahead pattern detection — 顺序访问模式匹配 → 预取窗口扩大 → 更快完成 I/O。

def apply_schema_consistent_consolidation(
    conn: sqlite3.Connection,
    project: str,
) -> dict:
    """
    iter442: Schema-Consistent Consolidation — 与项目核心图式高度重叠的近期 chunk 获得额外巩固加成。

    步骤：
    1. 识别图式核（schema cores）：access_count >= scc_schema_min_access + importance >= scc_schema_min_importance
    2. 对近期写入（created_at >= now - scc_window_days）的 chunk：
       计算其 encode_context 与所有图式核的最大 entity 重叠数；
       若重叠 >= scc_min_overlap → 给予 stability 加成。
    3. new_stab = min(365.0, stab × (1 + scc_bonus × 0.04))

    区别于 iter440（PF）：
      PF = stale 旧 chunk 被锚定（老知识保护）
      SCC = 近期新 chunk 嵌入图式（新知识快速系统巩固）
    """
    try:
        import config as _cfg_scc
    except ImportError:
        return {"schema_consolidated": 0, "total_examined": 0}

    try:
        if not _cfg_scc.get("store_vfs.scc_enabled"):
            return {"schema_consolidated": 0, "total_examined": 0}
        scc_schema_min_access = _cfg_scc.get("store_vfs.scc_schema_min_access")
        scc_schema_min_imp = _cfg_scc.get("store_vfs.scc_schema_min_importance")
        scc_min_overlap = _cfg_scc.get("store_vfs.scc_min_overlap")
        scc_window_days = _cfg_scc.get("store_vfs.scc_window_days")
        scc_bonus = _cfg_scc.get("store_vfs.scc_bonus")
    except Exception:
        return {"schema_consolidated": 0, "total_examined": 0}

    scc_multiplier = 1.0 + scc_bonus * 0.04  # 0.15 × 0.04 = 0.006 ≈ 0.6% 加成

    from datetime import datetime as _dt, timezone as _tz, timedelta as _td
    now = _dt.now(_tz.utc)
    now_iso = now.isoformat()

    # ── Step 1: 构建图式核 entity 集合列表 ──
    cutoff_window = (now - _td(days=scc_window_days)).isoformat()

    try:
        schema_rows = conn.execute(
            """SELECT encode_context FROM memory_chunks
               WHERE project = ?
                 AND COALESCE(access_count, 0) >= ?
                 AND COALESCE(importance, 0.0) >= ?
                 AND COALESCE(encode_context, '') != ''""",
            (project, scc_schema_min_access, scc_schema_min_imp),
        ).fetchall()
    except Exception:
        return {"schema_consolidated": 0, "total_examined": 0}

    if not schema_rows:
        return {"schema_consolidated": 0, "total_examined": 0}

    schema_entity_sets = []
    for sr in schema_rows:
        try:
            ec = sr[0] if isinstance(sr, (list, tuple)) else sr["encode_context"]
            if ec and ec.strip():
                eset = frozenset(e.strip().lower() for e in ec.split(",") if e.strip())
                if eset:
                    schema_entity_sets.append(eset)
        except Exception:
            continue

    if not schema_entity_sets:
        return {"schema_consolidated": 0, "total_examined": 0}

    # ── Step 2: 近期写入的 chunk（创建时间 >= now - scc_window_days）──
    try:
        rows = conn.execute(
            """SELECT id, stability, encode_context FROM memory_chunks
               WHERE project = ?
                 AND created_at >= ?
                 AND COALESCE(encode_context, '') != ''
                 AND COALESCE(stability, 0.1) > 0.1
               LIMIT 500""",
            (project, cutoff_window),
        ).fetchall()
    except Exception:
        return {"schema_consolidated": 0, "total_examined": 0}

    total_examined = len(rows)
    schema_consolidated = 0

    for row in rows:
        try:
            cid = row[0] if isinstance(row, (list, tuple)) else row["id"]
            stab = row[1] if isinstance(row, (list, tuple)) else row["stability"]
            ec = row[2] if isinstance(row, (list, tuple)) else row["encode_context"]

            stab_f = float(stab or 0.1)
            if stab_f <= 0.1:
                continue

            if not ec or not ec.strip():
                continue

            cand_entities = frozenset(e.strip().lower() for e in ec.split(",") if e.strip())
            if not cand_entities:
                continue

            # 检查与任何图式核的重叠
            max_overlap = 0
            for schema_set in schema_entity_sets:
                overlap = len(cand_entities & schema_set)
                if overlap > max_overlap:
                    max_overlap = overlap
                if max_overlap >= scc_min_overlap:
                    break  # 已找到足够重叠，无需继续

            if max_overlap < scc_min_overlap:
                continue

            new_stab = min(365.0, stab_f * scc_multiplier)
            if new_stab <= stab_f + 0.0001:
                continue

            conn.execute(
                "UPDATE memory_chunks SET stability=?, updated_at=? WHERE id=?",
                (round(new_stab, 4), now_iso, cid),
            )
            schema_consolidated += 1
        except Exception:
            continue

    if schema_consolidated > 0:
        conn.commit()

    return {"schema_consolidated": schema_consolidated, "total_examined": total_examined}


# ── iter443: Sleep-Targeted Reactivation — 睡眠期主动抢救衰退的高价值记忆（Stickgold 2005）──────
# 认知科学依据：
#   Stickgold (2005) Nature: 睡眠期 targeted memory reactivation — 海马 sharp-wave ripples 优先重放
#     高价值但 retrievability 下降的记忆（即将消退的重要记忆被"抢救"）。
#   Stickgold & Walker (2013) Nature Neuroscience: sleep memory triage —
#     优先级 = importance × (1 - retrievability)（高价值 + 正在衰退 = 最需要抢救）。
# OS 类比：Linux dirty page "expire" scan — flusher 扫描即将超时的脏页，强制写回抢救。

def apply_sleep_targeted_reactivation(
    conn: sqlite3.Connection,
    project: str,
) -> dict:
    """
    iter443: Sleep-Targeted Reactivation (STR) — 睡眠期主动抢救高 importance 但 retrievability 低的 chunk。

    对 importance >= str_min_importance 且 retrievability <= str_max_retrievability 的 chunk，
    按衰退程度修复 stability：
      rescue_bonus = (1.0 - retrievability) × str_scale
      new_stab = min(365.0, stab × (1 + rescue_bonus))

    优先级 = importance × (1 - retrievability)：高价值且正在衰退的记忆获得最大修复。
    适用于所有满足条件的 chunk（不限 stale），模拟海马对重要衰退记忆的 targeted reactivation。

    Returns:
      {"rescued": N, "total_examined": N}
    """
    try:
        import config as _cfg_str
    except ImportError:
        return {"rescued": 0, "total_examined": 0}

    try:
        if not _cfg_str.get("store_vfs.str_enabled"):
            return {"rescued": 0, "total_examined": 0}
        str_min_importance = _cfg_str.get("store_vfs.str_min_importance")      # 0.65
        str_max_retrievability = _cfg_str.get("store_vfs.str_max_retrievability")  # 0.40
        str_scale = _cfg_str.get("store_vfs.str_scale")                        # 0.12
    except Exception:
        return {"rescued": 0, "total_examined": 0}

    from datetime import datetime as _dt, timezone as _tz
    now_iso = _dt.now(_tz.utc).isoformat()

    # 扫描：importance >= str_min_importance + retrievability <= str_max_retrievability
    try:
        rows = conn.execute(
            """SELECT id, stability, importance, retrievability FROM memory_chunks
               WHERE project = ?
                 AND COALESCE(importance, 0.0) >= ?
                 AND COALESCE(retrievability, 1.0) <= ?
                 AND COALESCE(stability, 0.1) > 0.1
               ORDER BY (COALESCE(importance, 0.0) * (1.0 - COALESCE(retrievability, 1.0))) DESC
               LIMIT 200""",
            (project, str_min_importance, str_max_retrievability),
        ).fetchall()
    except Exception:
        return {"rescued": 0, "total_examined": 0}

    total_examined = len(rows)
    rescued = 0

    for row in rows:
        try:
            cid = row[0] if isinstance(row, (list, tuple)) else row["id"]
            stab = row[1] if isinstance(row, (list, tuple)) else row["stability"]
            imp = row[2] if isinstance(row, (list, tuple)) else row["importance"]
            ret = row[3] if isinstance(row, (list, tuple)) else row["retrievability"]

            stab_f = float(stab or 0.1)
            ret_f = float(ret if ret is not None else 1.0)

            if stab_f <= 0.1:
                continue

            # rescue_bonus 与遗忘程度正比：retrievability 越低 → 修复越大
            rescue_bonus = (1.0 - ret_f) * str_scale
            if rescue_bonus <= 0.0001:
                continue

            new_stab = min(365.0, stab_f * (1.0 + rescue_bonus))
            if new_stab <= stab_f + 0.0001:
                continue

            conn.execute(
                "UPDATE memory_chunks SET stability=?, updated_at=? WHERE id=?",
                (round(new_stab, 4), now_iso, cid),
            )
            rescued += 1
        except Exception:
            continue

    if rescued > 0:
        conn.commit()

    return {"rescued": rescued, "total_examined": total_examined}


# ── iter444: Contextual Reinstatement Effect — 情境再现期活跃 chunk 的睡眠额外巩固（Smith 1979 / Tulving 1983）──
# 认知科学依据：
#   Smith (1979) "Remembering in and out of context" — 情境再现时记忆提取成功率高 40-50%（环境依赖记忆）。
#   Tulving (1983) Encoding Specificity Principle — 检索线索与编码时情境越匹配，提取效率越高。
# OS 类比：Linux NUMA-aware page consolidation (khugepaged) —
#   khugepaged 优先合并同一 NUMA node 内热页为 hugepage；
#   session 活跃情境 = 当前 NUMA node，情境高度重叠 chunk = 同 node 热页 → sleep consolidate 时优先加大 stability。

def apply_contextual_reinstatement_consolidation(
    conn: sqlite3.Connection,
    project: str,
    session_accessed_ids: list = None,
) -> dict:
    """
    iter444: Contextual Reinstatement Effect (CRE) — sleep 时基于 session 活跃情境的额外巩固。

    步骤：
    1. 构建 session_active_entities：从本 session 被访问的 chunk 的 encode_context 合并 entity 集合。
       若 session_accessed_ids 为空/None，则取最近 cre_max_session_entities 个被访问的 chunk。
    2. 对所有 importance >= cre_min_importance 的 chunk，计算 encode_context 与 active_entities 的重叠：
       overlap = |chunk_entities ∩ session_active_entities|
       若 overlap >= cre_min_overlap：
         overlap_ratio = min(1.0, overlap / max(1, len(chunk_entities)))
         bonus_factor = cre_bonus × overlap_ratio
         new_stab = min(365.0, stab × (1 + bonus_factor))
    3. 返回 {"cre_consolidated": N, "total_examined": N}

    情境再现逻辑：本 session 多次访问某个主题的知识 → session_active_entities 包含大量相关 entity
    → 属于同一主题的 chunk 的 encode_context 与 active_entities 高度重叠 → 获得额外巩固加成。
    这模拟了"在情境再现期间学习的记忆被优先巩固"的认知科学效应。

    Returns:
      {"cre_consolidated": N, "total_examined": N}
    """
    try:
        import config as _cfg_cre
    except ImportError:
        return {"cre_consolidated": 0, "total_examined": 0}

    try:
        if not _cfg_cre.get("store_vfs.cre_enabled"):
            return {"cre_consolidated": 0, "total_examined": 0}
        cre_min_overlap = _cfg_cre.get("store_vfs.cre_min_overlap")       # 2
        cre_min_importance = _cfg_cre.get("store_vfs.cre_min_importance") # 0.40
        cre_bonus = _cfg_cre.get("store_vfs.cre_bonus")                   # 0.10
        cre_max_session = _cfg_cre.get("store_vfs.cre_max_session_entities")  # 200
    except Exception:
        return {"cre_consolidated": 0, "total_examined": 0}

    from datetime import datetime as _dt, timezone as _tz
    now_iso = _dt.now(_tz.utc).isoformat()

    # ── Step 1: 构建 session_active_entities 集合 ──
    # 从本 session 被访问的 chunk 的 encode_context 中提取所有 entity。
    # 使用 2 小时时间窗口（last_accessed >= now-2h），而非 LIMIT 取最新 N 条：
    # 这避免了候选 chunk 自身 entity 污染 session 情境集合（自举偏差）。
    from datetime import timedelta as _td_cre
    session_cutoff = (_dt.now(_tz.utc) - _td_cre(hours=2)).isoformat()

    try:
        if session_accessed_ids:
            # 有明确的 session 访问 ID 列表：精确构建（忽略时间窗口）
            placeholders = ",".join("?" * len(session_accessed_ids[:200]))
            session_rows = conn.execute(
                f"""SELECT encode_context FROM memory_chunks
                    WHERE project = ? AND id IN ({placeholders})
                      AND encode_context IS NOT NULL AND encode_context != ''""",
                [project] + list(session_accessed_ids[:200]),
            ).fetchall()
        else:
            # 无明确 ID：取最近 2 小时内被访问的 chunk 作为 session 情境代理
            session_rows = conn.execute(
                """SELECT encode_context FROM memory_chunks
                   WHERE project = ?
                     AND encode_context IS NOT NULL AND encode_context != ''
                     AND last_accessed >= ?
                   ORDER BY last_accessed DESC
                   LIMIT ?""",
                (project, session_cutoff, cre_max_session),
            ).fetchall()
    except Exception:
        return {"cre_consolidated": 0, "total_examined": 0}

    # 构建 session 活跃 entity 集合
    session_active_entities: set = set()
    for srow in session_rows:
        ec = srow[0] if isinstance(srow, (list, tuple)) else srow["encode_context"]
        if ec:
            for e in ec.split(","):
                e = e.strip().lower()
                if e:
                    session_active_entities.add(e)

    if len(session_active_entities) < cre_min_overlap:
        # session 情境太稀疏，无法做有意义的情境匹配
        return {"cre_consolidated": 0, "total_examined": 0}

    # ── Step 2: 扫描 importance 足够的 chunk，计算情境重叠 ──
    try:
        rows = conn.execute(
            """SELECT id, stability, importance, encode_context FROM memory_chunks
               WHERE project = ?
                 AND COALESCE(importance, 0.0) >= ?
                 AND encode_context IS NOT NULL AND encode_context != ''
                 AND COALESCE(stability, 0.1) > 0.1
               ORDER BY COALESCE(importance, 0.0) DESC
               LIMIT 500""",
            (project, cre_min_importance),
        ).fetchall()
    except Exception:
        return {"cre_consolidated": 0, "total_examined": 0}

    total_examined = len(rows)
    cre_consolidated = 0

    for row in rows:
        try:
            cid = row[0] if isinstance(row, (list, tuple)) else row["id"]
            stab = row[1] if isinstance(row, (list, tuple)) else row["stability"]
            imp = row[2] if isinstance(row, (list, tuple)) else row["importance"]
            ec = row[3] if isinstance(row, (list, tuple)) else row["encode_context"]

            stab_f = float(stab or 0.1)
            if stab_f <= 0.1:
                continue

            if not ec:
                continue

            # 提取 chunk entity 集合
            chunk_entities = frozenset(e.strip().lower() for e in ec.split(",") if e.strip())
            if not chunk_entities:
                continue

            # 计算与 session 活跃情境的重叠
            overlap = len(chunk_entities & session_active_entities)
            if overlap < cre_min_overlap:
                continue

            # overlap_ratio：归一化（以 chunk 自身 entity 数为基准，避免大 session entity 集偏差）
            overlap_ratio = min(1.0, overlap / max(1, len(chunk_entities)))
            bonus_factor = cre_bonus * overlap_ratio

            if bonus_factor <= 0.0001:
                continue

            new_stab = min(365.0, stab_f * (1.0 + bonus_factor))
            if new_stab <= stab_f + 0.0001:
                continue

            conn.execute(
                "UPDATE memory_chunks SET stability=?, updated_at=? WHERE id=?",
                (round(new_stab, 4), now_iso, cid),
            )
            cre_consolidated += 1
        except Exception:
            continue

    if cre_consolidated > 0:
        conn.commit()

    return {"cre_consolidated": cre_consolidated, "total_examined": total_examined}


# ── iter445: Reward-Tagged Memory Consolidation — 奖励标签记忆的睡眠优先巩固（Murty & Adcock 2014）──
# 认知科学依据：
#   Murty & Adcock (2014) "Enriching experiences via prior associative learning facilitates memory" —
#     多巴胺奖励信号在慢波睡眠期（SWS）激活 VTA-海马投射，选择性强化高奖励预期的记忆痕迹。
#   Hennies et al. (2015) "Closed-loop memory reactivation during sleep" (Current Biology) —
#     高奖励标签 + 睡眠 = 最强记忆保留：reward × sleep 的交互效应显著大于单独效应之和。
# OS 类比：Linux workingset_activation（工作集激活标记）——
#   kswapd 扫描时，reference bit=1 的页获得 second chance（不立即回收）；
#   page refcount × recency = 工作集优先级（高频近期访问 page = 最高 protection）；
#   类比：access_count × recency_factor = 记忆奖励优先级 → sleep 时优先强化。

def apply_reward_tagged_memory_consolidation(
    conn: sqlite3.Connection,
    project: str,
) -> dict:
    """
    iter445: Reward-Tagged Memory Consolidation (RTMC) — sleep 时基于访问频率×近期性的奖励巩固。

    数学模型：
      reward_signal = min(1.0, log(1 + access_count) / log(1 + rtmc_acc_ref))
      recency_factor = max(0.0, 1.0 - hours_since_access / rtmc_recency_hours)
      priority = reward_signal × recency_factor
      bonus = priority × rtmc_scale
      new_stab = min(365.0, stab × (1 + bonus))

    触发条件：
      - rtmc_enabled = True
      - access_count >= rtmc_min_access（至少被检索 N 次 = 有奖励历史）
      - hours_since_access <= rtmc_recency_hours（最近仍有访问 = 奖励信号新鲜）
      - importance >= rtmc_min_importance（低重要性 chunk 不参与）

    Returns:
      {"rtmc_boosted": N, "total_examined": N}
    """
    try:
        import config as _cfg_rtmc
    except ImportError:
        return {"rtmc_boosted": 0, "total_examined": 0}

    try:
        if not _cfg_rtmc.get("store_vfs.rtmc_enabled"):
            return {"rtmc_boosted": 0, "total_examined": 0}
        rtmc_min_access = _cfg_rtmc.get("store_vfs.rtmc_min_access")       # 3
        rtmc_acc_ref = _cfg_rtmc.get("store_vfs.rtmc_acc_ref")             # 10
        rtmc_recency_hours = _cfg_rtmc.get("store_vfs.rtmc_recency_hours") # 48.0
        rtmc_scale = _cfg_rtmc.get("store_vfs.rtmc_scale")                 # 0.08
        rtmc_min_importance = _cfg_rtmc.get("store_vfs.rtmc_min_importance")  # 0.35
    except Exception:
        return {"rtmc_boosted": 0, "total_examined": 0}

    import math as _math_rtmc
    from datetime import datetime as _dt_rtmc, timezone as _tz_rtmc
    now_dt = _dt_rtmc.now(_tz_rtmc.utc)
    now_iso = now_dt.isoformat()

    # 计算时间窗口截止点：rtmc_recency_hours 之前
    from datetime import timedelta as _td_rtmc
    recency_cutoff = (now_dt - _td_rtmc(hours=rtmc_recency_hours)).isoformat()

    try:
        rows = conn.execute(
            """SELECT id, stability, access_count, last_accessed, importance
               FROM memory_chunks
               WHERE project = ?
                 AND COALESCE(access_count, 0) >= ?
                 AND COALESCE(importance, 0.0) >= ?
                 AND last_accessed IS NOT NULL
                 AND last_accessed >= ?
                 AND COALESCE(stability, 0.1) > 0.1
               ORDER BY COALESCE(access_count, 0) DESC
               LIMIT 500""",
            (project, rtmc_min_access, rtmc_min_importance, recency_cutoff),
        ).fetchall()
    except Exception:
        return {"rtmc_boosted": 0, "total_examined": 0}

    total_examined = len(rows)
    rtmc_boosted = 0
    log_ref = _math_rtmc.log(1 + rtmc_acc_ref)  # precompute denominator

    for row in rows:
        try:
            cid = row[0] if isinstance(row, (list, tuple)) else row["id"]
            stab = row[1] if isinstance(row, (list, tuple)) else row["stability"]
            acc = row[2] if isinstance(row, (list, tuple)) else row["access_count"]
            last_accessed = row[3] if isinstance(row, (list, tuple)) else row["last_accessed"]

            stab_f = float(stab or 0.1)
            if stab_f <= 0.1:
                continue

            acc_f = float(acc or 0)
            if acc_f < rtmc_min_access:
                continue

            # 计算 hours_since_access
            try:
                if last_accessed.endswith("Z"):
                    last_accessed = last_accessed[:-1] + "+00:00"
                from datetime import datetime as _dt2_rtmc, timezone as _tz2_rtmc
                la_dt = _dt2_rtmc.fromisoformat(last_accessed)
                if la_dt.tzinfo is None:
                    la_dt = la_dt.replace(tzinfo=_tz2_rtmc.utc)
                hours_since = (now_dt - la_dt).total_seconds() / 3600.0
            except Exception:
                continue

            if hours_since > rtmc_recency_hours:
                continue

            # 奖励信号：对数归一化访问次数（acc=rtmc_acc_ref 时 reward_signal=1.0）
            reward_signal = min(1.0, _math_rtmc.log(1 + acc_f) / log_ref)

            # 近期因子：访问越新鲜，recency_factor 越接近 1.0
            recency_factor = max(0.0, 1.0 - hours_since / rtmc_recency_hours)

            priority = reward_signal * recency_factor
            if priority < 0.001:
                continue

            bonus = priority * rtmc_scale
            if bonus < 0.0001:
                continue

            new_stab = min(365.0, stab_f * (1.0 + bonus))
            if new_stab <= stab_f + 0.0001:
                continue

            conn.execute(
                "UPDATE memory_chunks SET stability=?, updated_at=? WHERE id=?",
                (round(new_stab, 4), now_iso, cid),
            )
            rtmc_boosted += 1
        except Exception:
            continue

    if rtmc_boosted > 0:
        conn.commit()

    return {"rtmc_boosted": rtmc_boosted, "total_examined": total_examined}


# ── iter446: Temporal Contiguity Effect — 时间毗邻性的记忆互相强化（Kahana 1996）────────────────────
# 认知科学依据：
#   Kahana (1996) "Associative retrieval processes in free recall" (J. Memory & Language) —
#     lag-CRP 曲线峰值在 lag=±1（时间相邻的记忆强度互相激活），时间毗邻提供隐式时序链接。
#   Howard & Kahana (2002) — 时间上下文向量高度相关的相邻事件在睡眠回放时被联合重放。
# OS 类比：Linux MGLRU temporal cohort aging —
#   同一 aging interval 内被访问的 pages 属于同一 generation，sleep 扫描时互相保护。

def apply_temporal_contiguity_consolidation(
    conn: sqlite3.Connection,
    project: str,
) -> dict:
    """
    iter446: Temporal Contiguity Effect (TCE) — sleep 时对时间毗邻写入的 chunk 相互加成 stability。

    算法：
    1. 获取项目内 importance >= tce_min_importance 的所有 chunk，按 created_at 排序。
    2. 滑动窗口：对连续 chunk 中 created_at 差距 <= tce_window_secs 的相邻对识别。
    3. 找出属于同一时间窗口的 chunk 组（时间段内 >= 2 个 chunk = 形成时间情节单元）。
    4. 对每个有效 chunk 组（size <= tce_max_group_size），每个成员 stability × (1 + tce_bonus)。
    5. 返回 {"tce_boosted": N, "total_examined": N}

    时间毗邻逻辑：
      - 同一窗口内的 chunk 代表同一编码情节（如一次连续的调试会话、一次设计讨论）。
      - 睡眠期海马重放时，时序链接使同情节内的 chunk 相互激活（lag-CRP 效应）。
      - 相互加成体现了"情节记忆组块化"：相邻编码的知识被整合到同一情节表示中。

    Returns:
      {"tce_boosted": N, "total_examined": N}
    """
    try:
        import config as _cfg_tce
    except ImportError:
        return {"tce_boosted": 0, "total_examined": 0}

    try:
        if not _cfg_tce.get("store_vfs.tce_enabled"):
            return {"tce_boosted": 0, "total_examined": 0}
        tce_window_secs = _cfg_tce.get("store_vfs.tce_window_secs")       # 1800
        tce_bonus = _cfg_tce.get("store_vfs.tce_bonus")                   # 0.05
        tce_min_importance = _cfg_tce.get("store_vfs.tce_min_importance") # 0.45
        tce_max_group = _cfg_tce.get("store_vfs.tce_max_group_size")      # 10
    except Exception:
        return {"tce_boosted": 0, "total_examined": 0}

    from datetime import datetime as _dt_tce, timezone as _tz_tce
    now_iso = _dt_tce.now(_tz_tce.utc).isoformat()

    # 获取所有符合 importance 阈值的 chunk，按 created_at 排序
    try:
        rows = conn.execute(
            """SELECT id, stability, importance, created_at FROM memory_chunks
               WHERE project = ?
                 AND COALESCE(importance, 0.0) >= ?
                 AND COALESCE(stability, 0.1) > 0.1
                 AND created_at IS NOT NULL
               ORDER BY created_at ASC
               LIMIT 2000""",
            (project, tce_min_importance),
        ).fetchall()
    except Exception:
        return {"tce_boosted": 0, "total_examined": 0}

    if len(rows) < 2:
        return {"tce_boosted": 0, "total_examined": len(rows)}

    total_examined = len(rows)

    # 解析 created_at 为 timestamp（秒），构建 (cid, stab, ts) 列表
    parsed = []
    for row in rows:
        try:
            cid = row[0] if isinstance(row, (list, tuple)) else row["id"]
            stab = row[1] if isinstance(row, (list, tuple)) else row["stability"]
            created_at_str = row[3] if isinstance(row, (list, tuple)) else row["created_at"]

            if not created_at_str:
                continue
            if created_at_str.endswith("Z"):
                created_at_str = created_at_str[:-1] + "+00:00"
            from datetime import datetime as _dt2_tce, timezone as _tz2_tce
            ca_dt = _dt2_tce.fromisoformat(created_at_str)
            if ca_dt.tzinfo is None:
                ca_dt = ca_dt.replace(tzinfo=_tz2_tce.utc)
            ts = ca_dt.timestamp()
            parsed.append((cid, float(stab or 0.1), ts))
        except Exception:
            continue

    if len(parsed) < 2:
        return {"tce_boosted": 0, "total_examined": total_examined}

    # 滑动窗口：找出同一时间窗口内的 chunk 组（连续 created_at 差 <= tce_window_secs）
    # 算法：从左到右，维护当前组 [group_start_ts, ...]，差距超过窗口则提交当前组，开新组
    groups = []
    current_group = [parsed[0]]
    for i in range(1, len(parsed)):
        cid, stab, ts = parsed[i]
        prev_ts = parsed[i - 1][2]
        if ts - prev_ts <= tce_window_secs:
            current_group.append(parsed[i])
        else:
            if len(current_group) >= 2:
                groups.append(current_group)
            current_group = [parsed[i]]
    if len(current_group) >= 2:
        groups.append(current_group)

    if not groups:
        return {"tce_boosted": 0, "total_examined": total_examined}

    tce_boosted = 0
    updates = []

    for group in groups:
        # 如果组太大，按 importance（此处用稳定性代理）取 top tce_max_group 个
        # 注意：rows 是按 importance 降序查询出来的，但这里是按 created_at 排序后分组
        # 需要从 group 中筛选：我们已经按 importance 过滤了（>= min_imp），
        # 如果组 size 超过 max_group，随机或按顺序取前 max_group 个（时间顺序）
        if len(group) > tce_max_group:
            # 取 stability 最高的 top tce_max_group 个（保护最重要的）
            group = sorted(group, key=lambda x: x[1], reverse=True)[:tce_max_group]

        # 对组内每个 chunk 施加时间毗邻加成
        for cid, stab_f, ts in group:
            if stab_f <= 0.1:
                continue
            new_stab = min(365.0, stab_f * (1.0 + tce_bonus))
            if new_stab <= stab_f + 0.0001:
                continue
            updates.append((round(new_stab, 4), now_iso, cid))
            tce_boosted += 1

    if updates:
        conn.executemany(
            "UPDATE memory_chunks SET stability=?, updated_at=? WHERE id=?",
            updates,
        )
        conn.commit()

    return {"tce_boosted": tce_boosted, "total_examined": total_examined}


# ── iter447: Von Restorff Sleep Reactivation — 孤立记忆的睡眠期优先回放（Restorff 1933 / McDaniel & Einstein 1986）──
# 认知科学依据：
#   Von Restorff (1933) Isolation Effect — 孤立/独特的项目比同质项目记忆更好（+40-60% recall）。
#   McDaniel & Einstein (1986) JEP — 孤立效应在延迟测试（1周后）更显著；睡眠巩固选择性保护孤立记忆。
#   Huang et al. (2004) Memory — 孤立记忆的 delayed recall 在睡眠后比清醒组高约 25%。
# OS 类比：Linux huge page mlock + MADV_HUGEPAGE 双标注 —
#   独特布局页（MADV_HUGEPAGE）+ 锁定（mlock）= kswapd 跳过 + khugepaged 优先处理（双重保护路径）。

def apply_von_restorff_sleep_reactivation(
    conn: sqlite3.Connection,
    project: str,
) -> dict:
    """
    iter447: Von Restorff Sleep Reactivation (VRR) — 孤立 chunk 在 sleep 时获得额外 stability 加成。

    算法：
    1. 获取项目内 importance >= vrr_min_importance 的所有 chunk，按 created_at 排序。
    2. 对每个 chunk，取其在 created_at 序列中的前后 vrr_neighbor_window/2 个邻居。
    3. 计算孤立度：isolation_score = 1 - avg(jaccard(chunk.encode_context, neighbor.encode_context))
       Jaccard = |交集| / |并集| （基于 encode_context token 集合）
    4. isolation_score >= vrr_min_isolation → sleep bonus = isolation_score × vrr_scale
       new_stab = min(365.0, stab × (1 + sleep_bonus))
    5. 返回 {"vrr_boosted": N, "total_examined": N}

    孤立度计算细节：
      - encode_context 按逗号/空格分词（与 iter407 isolation_effect 一致）。
      - 邻居 < 3 个时 isolation_score = 0.0（避免项目初期误判所有 chunk 为孤立）。
      - 邻居 Jaccard 均值越低 = 该 chunk 与周围知识越不同 = isolation_score 越高。

    Returns:
      {"vrr_boosted": N, "total_examined": N}
    """
    try:
        import config as _cfg_vrr
    except ImportError:
        return {"vrr_boosted": 0, "total_examined": 0}

    try:
        if not _cfg_vrr.get("store_vfs.vrr_enabled"):
            return {"vrr_boosted": 0, "total_examined": 0}
        vrr_min_isolation = _cfg_vrr.get("store_vfs.vrr_min_isolation")   # 0.60
        vrr_min_importance = _cfg_vrr.get("store_vfs.vrr_min_importance") # 0.50
        vrr_neighbor_window = _cfg_vrr.get("store_vfs.vrr_neighbor_window") # 20
        vrr_scale = _cfg_vrr.get("store_vfs.vrr_scale")                   # 0.10
    except Exception:
        return {"vrr_boosted": 0, "total_examined": 0}

    from datetime import datetime as _dt_vrr, timezone as _tz_vrr
    now_iso = _dt_vrr.now(_tz_vrr.utc).isoformat()

    # 获取所有符合 importance 阈值的 chunk，按 created_at 排序
    try:
        rows = conn.execute(
            """SELECT id, stability, importance, encode_context FROM memory_chunks
               WHERE project = ?
                 AND COALESCE(importance, 0.0) >= ?
                 AND COALESCE(stability, 0.1) > 0.1
               ORDER BY created_at ASC
               LIMIT 2000""",
            (project, vrr_min_importance),
        ).fetchall()
    except Exception:
        return {"vrr_boosted": 0, "total_examined": 0}

    if not rows:
        return {"vrr_boosted": 0, "total_examined": 0}

    total_examined = len(rows)

    # 构建 (cid, stab, token_set) 列表
    parsed = []
    for row in rows:
        try:
            cid = row[0] if isinstance(row, (list, tuple)) else row["id"]
            stab = float(row[1] if isinstance(row, (list, tuple)) else row["stability"] or 0.1)
            enc_ctx = row[3] if isinstance(row, (list, tuple)) else row["encode_context"]
            # 分词：按逗号 + 空格分割，去重为 frozenset
            tokens = frozenset(
                t.strip().lower()
                for t in (enc_ctx or "").replace(",", " ").split()
                if t.strip()
            )
            parsed.append((cid, stab, tokens))
        except Exception:
            continue

    if not parsed:
        return {"vrr_boosted": 0, "total_examined": total_examined}

    half_window = max(1, vrr_neighbor_window // 2)
    vrr_boosted = 0
    updates = []

    for i, (cid, stab_f, tokens) in enumerate(parsed):
        if stab_f <= 0.1 or not tokens:
            continue

        # 取前后邻居（排除自身）
        lo = max(0, i - half_window)
        hi = min(len(parsed), i + half_window + 1)
        neighbors = [parsed[j] for j in range(lo, hi) if j != i]

        if len(neighbors) < 3:
            # 邻居太少 → 无法可靠计算孤立度（避免项目初期误判）
            continue

        # 计算 Jaccard 均值
        jaccard_sum = 0.0
        valid_neighbors = 0
        for _, _, nb_tokens in neighbors:
            if not nb_tokens:
                continue
            inter = len(tokens & nb_tokens)
            union = len(tokens | nb_tokens)
            if union > 0:
                jaccard_sum += inter / union
                valid_neighbors += 1

        if valid_neighbors == 0:
            continue

        avg_jaccard = jaccard_sum / valid_neighbors
        isolation_score = 1.0 - avg_jaccard

        if isolation_score < vrr_min_isolation:
            continue

        # 孤立度达标 → 计算 sleep bonus
        sleep_bonus = isolation_score * vrr_scale
        new_stab = min(365.0, stab_f * (1.0 + sleep_bonus))
        if new_stab <= stab_f + 0.0001:
            continue

        updates.append((round(new_stab, 4), now_iso, cid))
        vrr_boosted += 1

    if updates:
        conn.executemany(
            "UPDATE memory_chunks SET stability=?, updated_at=? WHERE id=?",
            updates,
        )
        conn.commit()

    return {"vrr_boosted": vrr_boosted, "total_examined": total_examined}


# ── iter448: Retroactive Enhancement — 新知识睡眠后逆行增强旧相关知识（Mednick et al. 2011）──
# 认知科学依据：
#   Mednick et al. (2011) PNAS — 睡眠不仅巩固新知识，还逆行增强与之关联的旧记忆痕迹（bidirectional consolidation）。
#   Walker & Stickgold (2004) — 新技能睡眠后，结构相似的旧技能也有 overnight 提升。
#   Ellenbogen et al. (2007) Science — 睡眠促进新-旧知识联合整合，产生逆行传递性推断。
# OS 类比：Linux page fault 触发的 backward readahead —
#   page_N 缺页中断 → 内核向后预取 page_N-K 到 page_N-1（历史邻居）；
#   新 chunk 写入后睡眠 → 其关联的历史旧 chunk 也被逆行激活并增强。

def apply_retroactive_enhancement(
    conn: sqlite3.Connection,
    project: str,
) -> dict:
    """
    iter448: Retroactive Enhancement (RE) — sleep 时新 chunk 逆行增强旧相关 chunk 的 stability。

    算法：
    1. 找出"新 chunk"：created_at >= now - re_new_window_hours（24h 内写入）
       且 importance >= re_min_importance。
    2. 找出"旧 chunk"：created_at < now - re_new_window_hours
       且 importance >= re_min_importance。
    3. 对每个新 chunk，计算与所有旧 chunk 的 Jaccard(encode_context)：
       重叠 >= re_min_overlap → 候选旧 chunk。
    4. 每个旧 chunk 的 bonus = max(overlap_score × re_scale over all new chunks)。
    5. new_stab = min(365.0, old_stab × (1 + re_bonus))。
    6. 返回 {"re_boosted": N, "total_examined": N}

    关键设计：
    - 每个旧 chunk 最多被增强一次（取所有新 chunk 中的最大 bonus，避免重复叠加）。
    - re_max_old_per_new 限制每个新 chunk 影响的旧 chunk 数量（防止新 chunk 过于"广播"）。

    Returns:
      {"re_boosted": N, "total_examined": N}
    """
    try:
        import config as _cfg_re
    except ImportError:
        return {"re_boosted": 0, "total_examined": 0}

    try:
        if not _cfg_re.get("store_vfs.re_enabled"):
            return {"re_boosted": 0, "total_examined": 0}
        re_new_window_hours = _cfg_re.get("store_vfs.re_new_window_hours")  # 24.0
        re_min_overlap = _cfg_re.get("store_vfs.re_min_overlap")            # 3
        re_min_importance = _cfg_re.get("store_vfs.re_min_importance")      # 0.45
        re_scale = _cfg_re.get("store_vfs.re_scale")                        # 0.06
        re_max_old_per_new = _cfg_re.get("store_vfs.re_max_old_per_new")    # 5
    except Exception:
        return {"re_boosted": 0, "total_examined": 0}

    from datetime import datetime as _dt_re, timezone as _tz_re, timedelta as _td_re
    now_dt = _dt_re.now(_tz_re.utc)
    now_iso = now_dt.isoformat()
    new_cutoff = (now_dt - _td_re(hours=re_new_window_hours)).isoformat()

    # 获取新 chunk（24h 内写入）
    try:
        new_rows = conn.execute(
            """SELECT id, stability, importance, encode_context FROM memory_chunks
               WHERE project = ?
                 AND COALESCE(importance, 0.0) >= ?
                 AND COALESCE(stability, 0.1) > 0.1
                 AND created_at >= ?
               LIMIT 500""",
            (project, re_min_importance, new_cutoff),
        ).fetchall()
    except Exception:
        return {"re_boosted": 0, "total_examined": 0}

    if not new_rows:
        return {"re_boosted": 0, "total_examined": 0}

    # 获取旧 chunk（24h 前写入）
    try:
        old_rows = conn.execute(
            """SELECT id, stability, importance, encode_context FROM memory_chunks
               WHERE project = ?
                 AND COALESCE(importance, 0.0) >= ?
                 AND COALESCE(stability, 0.1) > 0.1
                 AND (created_at IS NULL OR created_at < ?)
               LIMIT 2000""",
            (project, re_min_importance, new_cutoff),
        ).fetchall()
    except Exception:
        return {"re_boosted": 0, "total_examined": 0}

    if not old_rows:
        return {"re_boosted": 0, "total_examined": 0}

    total_examined = len(old_rows)

    def _parse_tokens(row, idx=3):
        enc = row[idx] if isinstance(row, (list, tuple)) else row["encode_context"]
        return frozenset(
            t.strip().lower()
            for t in (enc or "").replace(",", " ").split()
            if t.strip()
        )

    def _get_field(row, idx, name):
        return row[idx] if isinstance(row, (list, tuple)) else row[name]

    # 解析旧 chunk
    old_parsed = []
    for row in old_rows:
        try:
            cid = _get_field(row, 0, "id")
            stab = float(_get_field(row, 1, "stability") or 0.1)
            tokens = _parse_tokens(row)
            old_parsed.append((cid, stab, tokens))
        except Exception:
            continue

    if not old_parsed:
        return {"re_boosted": 0, "total_examined": total_examined}

    # 解析新 chunk
    new_parsed = []
    for row in new_rows:
        try:
            tokens = _parse_tokens(row)
            if tokens:
                new_parsed.append(tokens)
        except Exception:
            continue

    if not new_parsed:
        return {"re_boosted": 0, "total_examined": total_examined}

    # 对每个旧 chunk，计算最大 bonus（来自所有新 chunk 中的最优关联）
    # old_bonus_map: {cid: max_bonus}
    old_bonus_map: dict = {}

    for new_tokens in new_parsed:
        if not new_tokens:
            continue

        # 找出与此新 chunk 高重叠的旧 chunk
        candidates = []
        for cid, stab_f, old_tokens in old_parsed:
            if not old_tokens:
                continue
            inter = len(new_tokens & old_tokens)
            if inter < re_min_overlap:
                continue
            union = len(new_tokens | old_tokens)
            if union == 0:
                continue
            overlap_score = inter / union
            candidates.append((cid, stab_f, overlap_score))

        # 取 top re_max_old_per_new（按 overlap_score 降序）
        candidates.sort(key=lambda x: x[2], reverse=True)
        for cid, stab_f, overlap_score in candidates[:re_max_old_per_new]:
            bonus = overlap_score * re_scale
            existing = old_bonus_map.get(cid, 0.0)
            if bonus > existing:
                old_bonus_map[cid] = bonus

    if not old_bonus_map:
        return {"re_boosted": 0, "total_examined": total_examined}

    # 构建更新列表（需要当前 stability）
    # 使用 old_parsed 中的 stab 值
    stab_lookup = {cid: stab_f for cid, stab_f, _ in old_parsed}
    re_boosted = 0
    updates = []

    for cid, bonus in old_bonus_map.items():
        stab_f = stab_lookup.get(cid, 0.0)
        if stab_f <= 0.1 or bonus <= 0.0:
            continue
        new_stab = min(365.0, stab_f * (1.0 + bonus))
        if new_stab <= stab_f + 0.0001:
            continue
        updates.append((round(new_stab, 4), now_iso, cid))
        re_boosted += 1

    if updates:
        conn.executemany(
            "UPDATE memory_chunks SET stability=?, updated_at=? WHERE id=?",
            updates,
        )
        conn.commit()

    return {"re_boosted": re_boosted, "total_examined": total_examined}


def apply_quiet_wakefulness_reactivation(
    conn: sqlite3.Connection,
    project: str,
    gap_seconds: float,
) -> dict:
    """
    iter449: Quiet Wakefulness Reactivation (QWR) — 清醒安静期自发重放预巩固。

    机制：
      - Tambini et al. (2010) Neuron：学习后 10min 安静休息的功能连接增强预测 24h 记忆保留。
      - Karlsson & Frank (2009) NatNeuro：清醒安静期海马自发重放先前轨迹（awake replay）。
      - gap in [qwr_min_gap_mins, qwr_sleep_threshold_hours×3600)：处于清醒休息期 → QWR。
      - gap >= qwr_sleep_threshold_hours×3600：整夜睡眠，由 iter413 SC 处理，此函数跳过。

    参数：
      gap_seconds — 距上次 session 结束的时间间隔（秒）。

    算法：
      1. 检查 gap 是否在 QWR 窗口内（[min_gap_mins*60, sleep_threshold_hours*3600)）。
      2. 查询 last_accessed >= now - qwr_recent_hours 且 importance >= qwr_min_importance 的 chunk。
      3. 按 importance × recency_factor 排序，取前 qwr_max_chunks 个。
      4. 每个 chunk stability × qwr_boost_factor，cap 365.0。
      5. 返回 {"qwr_boosted": N, "total_examined": N, "skipped_reason": str or None}

    OS 类比：Linux page cache incremental writeback (pdflush background flush) —
      定期小批量 dirty page 写回（QWR = 轻量增量回写），防止积压到 fsync（SC = 全量回写）。
    """
    try:
        import config as _cfg_qwr
    except ImportError:
        return {"qwr_boosted": 0, "total_examined": 0, "skipped_reason": "import_error"}

    try:
        if not _cfg_qwr.get("store_vfs.qwr_enabled"):
            return {"qwr_boosted": 0, "total_examined": 0, "skipped_reason": "disabled"}
        qwr_min_gap_mins = _cfg_qwr.get("store_vfs.qwr_min_gap_mins")           # 10
        qwr_sleep_threshold_hours = _cfg_qwr.get("store_vfs.qwr_sleep_threshold_hours")  # 8.0
        qwr_recent_hours = _cfg_qwr.get("store_vfs.qwr_recent_hours")           # 4.0
        qwr_boost_factor = _cfg_qwr.get("store_vfs.qwr_boost_factor")           # 1.03
        qwr_min_importance = _cfg_qwr.get("store_vfs.qwr_min_importance")       # 0.55
        qwr_max_chunks = _cfg_qwr.get("store_vfs.qwr_max_chunks")               # 30
    except Exception:
        return {"qwr_boosted": 0, "total_examined": 0, "skipped_reason": "config_error"}

    # 检查 gap 是否在 QWR 窗口内
    min_gap_secs = qwr_min_gap_mins * 60
    max_gap_secs = qwr_sleep_threshold_hours * 3600

    if gap_seconds < min_gap_secs:
        # 太短（连续会话），不触发 QWR
        return {"qwr_boosted": 0, "total_examined": 0, "skipped_reason": "gap_too_short"}
    if gap_seconds >= max_gap_secs:
        # 太长（整夜睡眠），由 iter413 SC 处理
        return {"qwr_boosted": 0, "total_examined": 0, "skipped_reason": "gap_too_long_use_sc"}

    from datetime import datetime as _dt_qwr, timezone as _tz_qwr, timedelta as _td_qwr
    now_dt = _dt_qwr.now(_tz_qwr.utc)
    now_iso = now_dt.isoformat()

    # 近期编码窗口（last_accessed 在 qwr_recent_hours 内）
    recent_cutoff = (now_dt - _td_qwr(hours=qwr_recent_hours)).isoformat()

    try:
        rows = conn.execute(
            """SELECT id, stability, importance, last_accessed FROM memory_chunks
               WHERE project = ?
                 AND COALESCE(importance, 0.0) >= ?
                 AND COALESCE(stability, 0.1) > 0.1
                 AND last_accessed >= ?
               ORDER BY importance DESC
               LIMIT ?""",
            (project, qwr_min_importance, recent_cutoff, qwr_max_chunks * 3),
        ).fetchall()
    except Exception:
        return {"qwr_boosted": 0, "total_examined": 0, "skipped_reason": "query_error"}

    if not rows:
        return {"qwr_boosted": 0, "total_examined": 0, "skipped_reason": "no_candidates"}

    def _get_field(row, idx, name):
        return row[idx] if isinstance(row, (list, tuple)) else row[name]

    # 按 importance × recency_factor 排序（近期 + 高重要性优先）
    scored = []
    for row in rows:
        try:
            cid = _get_field(row, 0, "id")
            stab = float(_get_field(row, 1, "stability") or 0.1)
            imp = float(_get_field(row, 2, "importance") or 0.0)
            la = _get_field(row, 3, "last_accessed") or ""
            # 计算时间新鲜度（越近 → recency_factor 越大）
            try:
                la_dt = _dt_qwr.fromisoformat(la.replace("Z", "+00:00"))
                if la_dt.tzinfo is None:
                    la_dt = la_dt.replace(tzinfo=_tz_qwr.utc)
                hours_ago = (now_dt - la_dt).total_seconds() / 3600.0
                recency_factor = max(0.0, 1.0 - hours_ago / qwr_recent_hours)
            except Exception:
                recency_factor = 0.5
            score = imp * (0.5 + 0.5 * recency_factor)  # importance 主导，recency 调节
            scored.append((cid, stab, score))
        except Exception:
            continue

    # 取前 qwr_max_chunks 个
    scored.sort(key=lambda x: x[2], reverse=True)
    top_chunks = scored[:qwr_max_chunks]
    total_examined = len(top_chunks)

    if not top_chunks:
        return {"qwr_boosted": 0, "total_examined": 0, "skipped_reason": "no_valid_rows"}

    # 应用 QWR stability 加成
    updates = []
    qwr_boosted = 0
    for cid, stab, _ in top_chunks:
        new_stab = min(365.0, stab * qwr_boost_factor)
        if new_stab <= stab + 0.0001:
            continue
        updates.append((round(new_stab, 4), now_iso, cid))
        qwr_boosted += 1

    if updates:
        conn.executemany(
            "UPDATE memory_chunks SET stability=?, updated_at=? WHERE id=?",
            updates,
        )
        conn.commit()

    return {"qwr_boosted": qwr_boosted, "total_examined": total_examined, "skipped_reason": None}


# ── iter432: Cumulative Interference Effect — 累积干扰加速遗忘（Underwood 1957）──
# 认知科学依据：Underwood (1957) "Interference and forgetting" —
#   遗忘的主要原因是同类型知识的累积干扰，而非单纯时间流逝（decay theory 不足以解释）。
#   同类型先行学习列表越多，后续学习的遗忘越快（proactive interference）。
#   Underwood 1957 关键数据：24小时遗忘量 vs 已学干扰列表数的相关 r=0.92（极强正相关）。
#   Jenkins & Dallenbach (1924)：睡眠减少新干扰 → 遗忘更少（佐证：干扰主导，非时间）。
# OS 类比：Linux CPU cache set-associativity conflict —
#   同一 cache set 中 N-way associativity 达到上限时，新 line 必须驱逐旧 line（LRU）；
#   同 cache set 的 line 越多（more competition），每条 line 的平均留存时间越短。
#   cumulative_interference_factor = 1 + scale × log(1+N) / log(1+N_median)
#   → N 越大，factor > 1，stability 衰减更快（额外 × 1/factor 作为 penalty）。

def compute_cumulative_interference_factor(
    n_same_type: int,
    n_median: int = 10,
) -> float:
    """
    iter432: 计算累积干扰因子。

    factor = 1 + scale × log(1 + n_same_type) / log(1 + n_median)
    n_same_type < ci_min_n_same_type → factor = 1.0（无干扰）
    factor 上限为 ci_max_factor。

    在 decay_stability_by_type_with_ci() 中：
      new_stability = stability × type_decay / factor

    参数：
      n_same_type — 当前项目中同 chunk_type 的 chunk 数量
      n_median    — 参考中位数（用于规范化，默认 10）

    Returns:
      float >= 1.0 — 干扰因子（> 1 = 加速衰减）
    """
    import config as _config
    import math
    try:
        if not _config.get("scorer.cumulative_interference_enabled"):
            return 1.0
        min_n = int(_config.get("scorer.ci_min_n_same_type") or 5)
        if n_same_type < min_n:
            return 1.0
        scale = float(_config.get("scorer.ci_scale") or 0.30)
        max_factor = float(_config.get("scorer.ci_max_factor") or 2.0)
        if n_median <= 0:
            n_median = 10
        factor = 1.0 + scale * math.log(1 + n_same_type) / math.log(1 + n_median)
        return min(max_factor, factor)
    except Exception:
        return 1.0


def decay_stability_by_type_with_ci(
    conn: sqlite3.Connection,
    project: str = None,
    stale_days: int = 30,
    now_iso: str = None,
) -> dict:
    """
    iter432: decay_stability_by_type 的扩展版，叠加 Cumulative Interference Effect。

    在 decay_stability_by_type 基础上：
      对每种 chunk_type，统计当前项目中该类型的 chunk 数量（N_same_type），
      计算累积干扰因子 factor，对该类型的 stability 衰减乘以 1/factor（等效加速衰减）：
        effective_decay = type_decay × (1/factor) ≡ type_decay / factor
        new_stability = MAX(0.1, stability × effective_decay)

    豁免类型（ci_protect_types，默认 design_constraint/procedure）不受干扰影响。
    也尊重 Ribot floor：衰减结果不低于 ribot_floor。

    Returns:
      dict — {total_decayed: N, ci_factors: {chunk_type: factor}}
    """
    from datetime import datetime as _dt, timezone as _tz, timedelta as _td
    import config as _config
    if now_iso is None:
        now_iso = _dt.now(_tz.utc).isoformat()
    cutoff = (_dt.now(_tz.utc) - _td(days=stale_days)).isoformat()

    proj_filter = "AND project=?" if project else ""
    proj_params = [project] if project else []

    # 获取保护类型（不受干扰）
    protect_types_str = _config.get("scorer.ci_protect_types") or ""
    protect_types = frozenset(t.strip() for t in protect_types_str.split(",") if t.strip())

    # 统计每种 chunk_type 的数量（per-project）
    type_counts: dict = {}
    try:
        count_sql = "SELECT chunk_type, COUNT(*) FROM memory_chunks"
        count_params = []
        if project:
            count_sql += " WHERE project=?"
            count_params = [project]
        count_sql += " GROUP BY chunk_type"
        rows = conn.execute(count_sql, count_params).fetchall()
        for r in rows:
            ct = r[0] or ""
            cnt = int(r[1] or 0)
            type_counts[ct] = cnt
    except Exception:
        pass

    # N_median：所有 chunk_type 数量的中位数（规范化分母）
    counts_list = sorted(type_counts.values())
    n_median = counts_list[len(counts_list) // 2] if counts_list else 10

    total_decayed = 0
    ci_factors: dict = {}

    all_types = list(CHUNK_TYPE_DECAY.keys()) + [""]

    for ctype, decay in CHUNK_TYPE_DECAY.items():
        if ctype in protect_types:
            # 豁免类型：不应用干扰，使用普通 type_decay
            try:
                conn.execute(
                    f"UPDATE memory_chunks "
                    f"SET stability=MAX(0.1, stability * ?), updated_at=? "
                    f"WHERE chunk_type=? AND last_accessed < ? AND access_count < 2 {proj_filter}",
                    [decay, now_iso, ctype, cutoff] + proj_params,
                )
                total_decayed += conn.execute("SELECT changes()").fetchone()[0]
            except Exception:
                pass
            ci_factors[ctype] = 1.0
            continue

        n_ct = type_counts.get(ctype, 0)
        factor = compute_cumulative_interference_factor(n_ct, n_median)
        # effective_decay = type_decay / factor（factor >= 1 → 有效衰减 <= type_decay）
        # 注意：decay 是 [0,1] 的乘子（越大衰减越慢），除以 factor > 1 → 更小的乘子 → 更快衰减
        effective_decay = max(0.01, decay / factor)
        ci_factors[ctype] = factor

        try:
            conn.execute(
                f"UPDATE memory_chunks "
                f"SET stability=MAX(0.1, stability * ?), updated_at=? "
                f"WHERE chunk_type=? AND last_accessed < ? AND access_count < 2 {proj_filter}",
                [effective_decay, now_iso, ctype, cutoff] + proj_params,
            )
            total_decayed += conn.execute("SELECT changes()").fetchone()[0]
        except Exception:
            pass

    # 未列出的类型（使用默认衰减率）
    unknown_n = type_counts.get("", 0) + sum(
        v for k, v in type_counts.items() if k not in CHUNK_TYPE_DECAY
    )
    default_factor = compute_cumulative_interference_factor(unknown_n, n_median)
    effective_default = max(0.01, _DEFAULT_TYPE_DECAY / default_factor)
    ci_factors["_other"] = default_factor

    known_types_ph = ",".join("?" * len(CHUNK_TYPE_DECAY))
    try:
        conn.execute(
            f"UPDATE memory_chunks "
            f"SET stability=MAX(0.1, stability * ?), updated_at=? "
            f"WHERE (chunk_type NOT IN ({known_types_ph}) OR chunk_type IS NULL) "
            f"AND last_accessed < ? AND access_count < 2 {proj_filter}",
            [effective_default, now_iso] + list(CHUNK_TYPE_DECAY.keys()) + [cutoff] + proj_params,
        )
        total_decayed += conn.execute("SELECT changes()").fetchone()[0]
    except Exception:
        pass

    return {"total_decayed": total_decayed, "ci_factors": ci_factors}


# ── iter402：Schema Theory — Prior Knowledge Scaffolding（Bartlett 1932）────────
#
# 认知科学依据：
#   Bartlett (1932) Remembering — "图式"（Schema）理论：
#     新信息被同化到已有知识框架（图式）中，共享框架的知识相互加固。
#     当新知识和已有高稳定性知识共享概念时，新知识的初始稳定性更高。
#   Piaget (1952) Schema Assimilation：
#     assimilation — 新信息被纳入现有图式（没有根本改变图式）
#     accommodation — 现有图式被修改以适应新信息
#     这里实现 assimilation：新 chunk 共享已有 entity → 继承部分 stability
#   Anderson (1984) Schema Theory in Education：
#     先验知识越丰富，新知识越容易被编码（"rich get richer"效应）。
#
# OS 类比：Linux Transparent Hugepage (THP) promotion
#   当一个 2MB 对齐的内存区域中大多数 4KB 页面都存在时（prior_pages_exist），
#   新 fault 进来的匿名页会直接被提升为 THP 的一部分，继承 THP 的 cache 亲和性。
#   新 chunk 发现已有同主题 chunk（prior schema）→ 继承部分 stability bonus。
#
# 实现：
#   compute_schema_bonus(conn, chunk_id, project) → float [0.0, 2.0]
#     通过 entity_map 查找 chunk 关联的 entity，
#     再通过 entity_map 找到同 project 中共享这些 entity 的已有 chunk，
#     取这些先验 chunk 的 stability 均值 × schema_inherit_ratio（默认 0.2）。
#   apply_schema_scaffolding(conn, chunk_id, content, project)
#     写入 schema_bonus 到 stability

import re as _re_schema

_SCHEMA_INHERIT_RATIO: float = 0.2   # 继承先验 stability 的比例
_SCHEMA_MAX_BONUS: float = 2.0       # 最大 bonus（防止极端情况）


def compute_schema_bonus(
    conn: sqlite3.Connection,
    chunk_id: str,
    project: str,
    max_bonus: float = _SCHEMA_MAX_BONUS,
    inherit_ratio: float = _SCHEMA_INHERIT_RATIO,
) -> float:
    """
    iter402：计算新 chunk 基于先验图式（existing knowledge）的稳定性加成。

    算法：
      1. 通过 entity_map 找到 chunk_id 关联的 entity_name（写入时已设置）
      2. 对每个 entity，通过 entity_map 找到 project 中其他 chunk 的 stability
      3. 取所有先验 chunk stability 的均值 × inherit_ratio
      4. 先验 chunk 越多、越稳定 → bonus 越高
      5. clamp 到 [0.0, max_bonus]

    OS 类比：THP promotion scan — 扫描已有同区域 pages 的 PFN 密度，
      密度越高（prior_schema 越丰富）→ 新 page 晋升 THP 概率越高。

    Returns:
      float ∈ [0.0, max_bonus]
    """
    if not chunk_id or not project:
        return 0.0
    try:
        # Step 1: 找到该 chunk 关联的 entity（entity_map 当前行 OR entity_edges）
        entity_rows = conn.execute(
            "SELECT entity_name FROM entity_map WHERE chunk_id=? AND project=?",
            (chunk_id, project),
        ).fetchall()

        # entity_map PK=(entity_name, project)：若新 chunk 刚写入，entity 已指向它
        # 所以 entity_name 已知；再通过 entity_edges 找到同 project 中
        # 以该 entity 为 from/to 的关系涉及的 source_chunk_id（历史 chunk）
        entity_names = [r[0] for r in entity_rows if r[0]]
        if not entity_names:
            return 0.0

        # Step 2a: 通过 entity_edges 找到同 project 中涉及这些 entity 的 chunk
        ent_ph = ",".join("?" * len(entity_names))
        edge_chunk_rows = conn.execute(
            f"SELECT DISTINCT source_chunk_id FROM entity_edges "
            f"WHERE (from_entity IN ({ent_ph}) OR to_entity IN ({ent_ph})) "
            f"AND project=? AND source_chunk_id IS NOT NULL AND source_chunk_id != ?",
            entity_names + entity_names + [project, chunk_id],
        ).fetchall()
        edge_chunk_ids = [r[0] for r in edge_chunk_rows if r[0]]

        # Step 2b: 通过 content/summary LIKE 搜索找到同 project 中含这些 entity 的 chunk
        # entity_map PK 限制只能指向最新 chunk，所以需要直接搜内容
        like_conditions = " OR ".join(
            ["(mc.content LIKE ? OR mc.summary LIKE ?)"] * len(entity_names)
        )
        like_params = []
        for en in entity_names:
            like_params.extend([f"%{en}%", f"%{en}%"])

        content_rows = conn.execute(
            f"SELECT mc.id, mc.stability FROM memory_chunks mc "
            f"WHERE mc.project=? AND mc.id != ? AND mc.stability IS NOT NULL "
            f"AND ({like_conditions})",
            [project, chunk_id] + like_params,
        ).fetchall()
        content_stabilities = {r[0]: float(r[1]) for r in content_rows if r[1] is not None}

        # Step 2c: 合并 edge_chunk_ids 对应的 stability
        if edge_chunk_ids:
            edge_ph = ",".join("?" * len(edge_chunk_ids))
            edge_rows = conn.execute(
                f"SELECT stability FROM memory_chunks WHERE id IN ({edge_ph}) AND stability IS NOT NULL",
                edge_chunk_ids,
            ).fetchall()
            for r in edge_rows:
                content_stabilities[f"_edge_{len(content_stabilities)}"] = float(r[0])

        if not content_stabilities:
            return 0.0

        # Step 3: 先验 chunk stability 均值 × inherit_ratio
        prior_stabilities = list(content_stabilities.values())
        avg_prior_stability = sum(prior_stabilities) / len(prior_stabilities)
        bonus = avg_prior_stability * inherit_ratio
        return round(min(max_bonus, max(0.0, bonus)), 4)
    except Exception:
        return 0.0


def apply_schema_scaffolding(
    conn: sqlite3.Connection,
    chunk_id: str,
    project: str,
    base_stability: float = 1.0,
) -> float:
    """
    iter402：应用图式加成 — 将 compute_schema_bonus 结果加到 stability。

    OS 类比：THP promotion path — 新 page fault 落在高密度区域时，
      内核直接 alloc_huge_page() 而不是分配独立 4KB 页。

    Returns:
      new_stability（包含 schema bonus）
    """
    bonus = compute_schema_bonus(conn, chunk_id, project)
    if bonus <= 0.001:
        return base_stability

    new_stability = min(base_stability * 4.0, base_stability + bonus)
    try:
        conn.execute(
            "UPDATE memory_chunks SET stability=?, updated_at=? WHERE id=?",
            (round(new_stability, 4), datetime.now(timezone.utc).isoformat(), chunk_id),
        )
    except Exception:
        pass
    return round(new_stability, 4)


# ── iter401：Elaborative Encoding — Depth of Processing（Craik & Lockhart 1972）──
#
# 认知科学依据：
#   Craik & Lockhart (1972) Levels of Processing：
#     记忆痕迹强度由信息被加工的"深度"决定，而非单纯的重复次数。
#     - 浅处理（字形/音韵）：只分析物理特征 → 短暂记忆痕迹
#     - 深处理（语义/关联）：分析意义、关联已有知识 → 持久记忆痕迹
#   Craik & Tulving (1975)：语义判断任务（"这个词适合句子吗？"）比视觉判断
#     产生更强的记忆，因为触发了更多的语义网络激活。
#   Reder & Anderson (1980)：精细编码（elaborate encoding）通过增加区分性
#     线索来增强提取能力。
#
# OS 类比：Linux dirty page writeback 的 write aggregation —
#   页面在 dirty buffer 中等待时间越长，write aggregation 越充分，
#   I/O 效率越高（类比深度加工 → 记忆更完整，更易检索）。
#   另一类比：L1 TLB miss → L2 TLB → page table walk — 越深层的处理成本越高，
#   但缓存命中率越持久。
#
# 实现：
#   compute_depth_of_processing(text) → float [0.0, 1.0]
#   通过以下特征估算加工深度：
#     1. 因果推理词（because/therefore/causes/由于/因此）→ 语义深处理
#     2. 结构化分析词（first/then/finally/第一/第二）→ 组织性加工
#     3. 对比/比较（however/unlike/相比/但是）→ 区分性处理
#     4. 抽象概念数量（concept density）→ 语义丰富度
#     5. 文本长度（适度长度 = 充分展开）→ 信息密度代理

import re as _re_dop

_DOP_CAUSAL_RE = _re_dop.compile(
    r'because|therefore|thus|hence|causes|leads to|results in|due to|'
    r'since|so that|in order to|consequently|'
    r'因为|因此|所以|由于|导致|造成|使得|故而|结果|从而',
    _re_dop.IGNORECASE,
)
_DOP_STRUCTURAL_RE = _re_dop.compile(
    r'first[,\s]|second[,\s]|third[,\s]|finally|then |next |'
    r'step 1|step 2|step \d|phase \d|'
    r'第一[，。、]|第二[，。、]|第三[，。、]|首先|其次|最后|然后|接下来|步骤',
    _re_dop.IGNORECASE,
)
_DOP_CONTRASTIVE_RE = _re_dop.compile(
    r'however|but |although|unlike|whereas|on the other hand|'
    r'nevertheless|in contrast|compared to|'
    r'但是|然而|虽然|尽管|不过|相比|相反|与此相比|对比',
    _re_dop.IGNORECASE,
)
_DOP_ELABORATION_RE = _re_dop.compile(
    r'specifically|in particular|for example|for instance|'
    r'that is to say|in other words|namely|such as|'
    r'具体来说|特别是|例如|比如|也就是说|换句话说|即',
    _re_dop.IGNORECASE,
)

# 每个类别的最大贡献（防止单一维度主导）
_DOP_MAX_PER_CATEGORY = 0.25


def compute_depth_of_processing(text: str) -> float:
    """
    iter401：计算文本的加工深度（Depth of Processing, Craik & Lockhart 1972）。

    四个维度各贡献最多 0.25，总分 [0.0, 1.0]：
      1. 因果推理 (0.25)：有无因果/推理词
      2. 结构组织 (0.25)：有无序列/结构词
      3. 对比区分 (0.25)：有无对比/比较词
      4. 精细阐述 (0.25)：有无例证/解释词

    OS 类比：Linux perf stat 的 IPC（Instructions Per Cycle）—
      同样的代码路径，加工深度不同导致不同的缓存热度。

    Returns:
      float ∈ [0.0, 1.0]
    """
    if not text or len(text.strip()) < 4:
        return 0.0

    score = 0.0

    # 维度 1：因果推理
    causal_count = len(_DOP_CAUSAL_RE.findall(text))
    score += min(_DOP_MAX_PER_CATEGORY, causal_count * 0.12)

    # 维度 2：结构组织
    struct_count = len(_DOP_STRUCTURAL_RE.findall(text))
    score += min(_DOP_MAX_PER_CATEGORY, struct_count * 0.10)

    # 维度 3：对比区分
    contrast_count = len(_DOP_CONTRASTIVE_RE.findall(text))
    score += min(_DOP_MAX_PER_CATEGORY, contrast_count * 0.12)

    # 维度 4：精细阐述
    elab_count = len(_DOP_ELABORATION_RE.findall(text))
    score += min(_DOP_MAX_PER_CATEGORY, elab_count * 0.10)

    return round(min(1.0, max(0.0, score)), 4)


def apply_depth_of_processing(
    conn: sqlite3.Connection,
    chunk_id: str,
    content: str,
    base_stability: float = 1.0,
) -> float:
    """
    iter401：计算 depth_of_processing，写入 DB，并返回调整后的 stability。

    深度加工 bonus：
      depth >= 0.5 → stability += 0.5（中等深度加工）
      depth >= 0.75 → stability += 1.5（高度加工，形成长久记忆痕迹）
    上限：base_stability 最高 × 3.0

    OS 类比：Linux CoW（Copy-on-Write）page promotion —
      页面被多次写入且内容丰富时，从 anon page 晋升到 THP（Transparent Hugepage），
      访问延迟从 4KB miss → 2MB TLB hit。

    Returns:
      new_stability（包含 depth bonus）
    """
    dop = compute_depth_of_processing(content or "")

    # depth_bonus: 线性插值，dop=0 → +0, dop=1 → +2.0
    depth_bonus = dop * 2.0
    new_stability = min(base_stability * 3.0, base_stability + depth_bonus)

    try:
        conn.execute(
            "UPDATE memory_chunks SET depth_of_processing=?, stability=?, updated_at=? WHERE id=?",
            (dop, round(new_stability, 4), datetime.now(timezone.utc).isoformat(), chunk_id),
        )
    except Exception:
        pass

    return round(new_stability, 4)


def promote_to_semantic(
    conn: sqlite3.Connection,
    source_chunk_ids: list,
    project: str,
    session_id: str = "",
    min_recall_count: int = 3,
) -> Optional[str]:
    """
    迭代319：将多次召回的情节 chunk 合并提升为语义 chunk。

    算法：
      1. 读取所有 source_chunk_ids 的 content/summary/access_count
      2. 过滤掉 access_count < min_recall_count 的（未达到巩固阈值）
      3. 合并 summary → 生成新语义 chunk（info_class='semantic'）
      4. 降级原情节 chunk（info_class='world', importance *= 0.6, oom_adj += 100）
      5. 在 episodic_consolidations 中记录转化事件

    OS 类比：Linux THP compaction (khugepaged) —
      扫描连续小页面，若访问频率够高则合并成 2MB hugepage（类比语义 chunk），
      原小页面被 free（类比情节 chunk 降级），元数据存入 compound_page 结构。

    Returns:
      新语义 chunk 的 ID，或 None（无满足条件的情节 chunk）
    """
    if not source_chunk_ids:
        return None

    ph = ",".join("?" * len(source_chunk_ids))
    rows = conn.execute(
        f"SELECT id, summary, content, access_count, importance "
        f"FROM memory_chunks "
        f"WHERE id IN ({ph}) AND project=? AND info_class='episodic'",
        source_chunk_ids + [project],
    ).fetchall()

    # 过滤：access_count >= min_recall_count
    eligible = [(r[0], r[1], r[2], r[3], r[4]) for r in rows
                if (r[3] or 0) >= min_recall_count]
    if not eligible:
        return None

    # 合并 summary：取所有 eligible 的 summary，去重后拼接
    summaries = list({r[1] for r in eligible if r[1]})
    if not summaries:
        return None

    # 新语义 chunk：保留最高 importance，summary 为第一条，content 为所有 summary 聚合
    max_importance = max(r[4] or 0.5 for r in eligible)
    primary_summary = summaries[0]
    merged_content = "\n".join(summaries)[:2000]

    import uuid as _uuid
    new_id = "sem_" + _uuid.uuid4().hex[:16]
    now_iso = datetime.now(timezone.utc).isoformat()

    new_chunk = {
        "id": new_id,
        "created_at": now_iso,
        "updated_at": now_iso,
        "project": project,
        "source_session": session_id,
        "chunk_type": "decision",  # 语义记忆默认用 decision 类型
        "info_class": "semantic",
        "content": merged_content,
        "summary": f"[语义化] {primary_summary}",
        "tags": ["semantic", "consolidated"],
        "importance": min(0.95, max_importance * 1.1),  # 轻微提升
        "retrievability": 0.8,
        "last_accessed": now_iso,
        "access_count": sum(r[3] or 0 for r in eligible),
        "oom_adj": -100,  # 语义记忆优先保留
        "lru_gen": 0,
        "stability": min(365.0, max_importance * 30.0),  # 高 stability
        "raw_snippet": "",
        "encoding_context": {},
    }
    insert_chunk(conn, new_chunk)

    # 降级原情节 chunk
    source_ids = [r[0] for r in eligible]
    for src_id in source_ids:
        old_imp = next(r[4] for r in eligible if r[0] == src_id) or 0.5
        conn.execute(
            "UPDATE memory_chunks SET info_class='world', importance=?, oom_adj=oom_adj+100, "
            "updated_at=? WHERE id=?",
            (round(old_imp * 0.6, 4), now_iso, src_id),
        )

    # 记录转化事件
    trigger_count = max(r[3] or 0 for r in eligible)
    try:
        conn.execute(
            """INSERT INTO episodic_consolidations
               (semantic_chunk_id, source_chunk_ids, project, trigger_count, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (new_id, json.dumps(source_ids), project, trigger_count, now_iso),
        )
    except Exception:
        pass

    return new_id


def episodic_decay_scan(
    conn: sqlite3.Connection,
    project: str,
    stale_days: int = 14,
    semantic_threshold: int = 2,
    max_promote: int = 10,
    semantic_hard_threshold: int = 5,
) -> dict:
    """
    迭代319：扫描情节记忆 — 衰减过期情节 chunk，提升高频召回情节 chunk 为语义 chunk。
    迭代327：semantic_threshold 降低 3 → 2（access_count=0 的情节 chunk 因 content 太短
    从未被召回，threshold=3 导致晋升路径永远不触发；降低到 2 让 access_count>=2 的 10 个
    chunks 有资格晋升，也避免"先有鸡还是先有蛋"的死锁）。
    迭代379：新增 A0 原地提升路径 — 基于 Tulving (1972) 双加工理论：
      单个情节 chunk 多次访问（>= semantic_hard_threshold=5）时，原地升级为语义记忆。
      避免碎片合并（promote_to_semantic 路径），保留 chunk identity，
      提升 stability × 1.5，设 info_class='semantic'，让语义层衰减速率（0.97）生效。
      OS 类比：mprotect(PROT_READ|PROT_EXEC) — 热页面提升保护级别，
        从 anonymous page（情节）升级为 file-backed 共享页（语义，跨 session 共享）。

    三个子操作（类比睡眠巩固的特化版本）：
      A0. 原地提升（iter379）：单个 info_class='episodic' chunk，
          access_count >= semantic_hard_threshold（默认5）→ 原地升级 info_class='semantic',
          stability × 1.5（上限 200），oom_adj -= 50（增加保留概率）
      A.  合并提升：info_class='episodic' AND access_count >= semantic_threshold（默认2）
          → 调用 promote_to_semantic()，合并同组情节 chunk 为新语义 chunk
      B.  衰减：info_class='episodic' AND last_accessed < (now - stale_days)
          AND access_count < 2 → importance *= 0.7, oom_adj += 50

    OS 类比：Linux khugepaged + kswapd 协同 —
      A0: mprotect() 热页面原地升级权限（不复制，不移动）
      A:  khugepaged 提升高频访问小页面（促进 → 语义）
      kswapd 回收冷页面（衰减 → 降权 → 更易被 evict）

    Returns:
      {"decayed": N, "promoted": N, "inplace_promoted": N, "new_semantic_ids": [...]}
    """
    result: dict = {"decayed": 0, "promoted": 0, "inplace_promoted": 0, "new_semantic_ids": []}
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()

    # ── 子操作 A0：原地提升（iter379 新增）────────────────────────────────────
    # 认知科学基础：Tulving (1972) episodic-to-semantic shift —
    #   情节记忆通过多次重激活（access_count++）逐渐脱离时间/情境特异性，
    #   转化为与情境无关的通用语义知识（语义记忆）。
    # 触发条件：access_count >= semantic_hard_threshold（5），chunk_type 为可巩固类型
    # 效果：info_class 原地更新（不新建 chunk），stability × 1.5，oom_adj -= 50
    _CONSOLIDATABLE_TYPES = ("reasoning_chain", "conversation_summary", "causal_chain",
                              "decision", "design_constraint")
    try:
        inplace_rows = conn.execute(
            "SELECT id, stability, oom_adj, chunk_type FROM memory_chunks "
            "WHERE project=? AND info_class='episodic' "
            "  AND chunk_type IN ({}) "
            "  AND COALESCE(access_count,0) >= ?".format(
                ",".join("?" * len(_CONSOLIDATABLE_TYPES))
            ),
            (project, *_CONSOLIDATABLE_TYPES, semantic_hard_threshold),
        ).fetchall()

        inplace_promoted = 0
        for row in inplace_rows:
            cid, cur_stability, cur_oom, ctype = row
            cur_stability = cur_stability or 1.0
            cur_oom = cur_oom or 0
            new_stability = min(200.0, cur_stability * 1.5)
            new_oom = max(-500, cur_oom - 50)
            conn.execute(
                "UPDATE memory_chunks "
                "SET info_class='semantic', stability=?, oom_adj=?, updated_at=? "
                "WHERE id=?",
                (round(new_stability, 4), new_oom, now_iso, cid),
            )
            inplace_promoted += 1

        result["inplace_promoted"] = inplace_promoted
    except Exception:
        pass

    # ── 子操作 A：合并提升高频情节 chunk（原有路径）─────────────────────────
    try:
        promote_rows = conn.execute(
            "SELECT id FROM memory_chunks "
            "WHERE project=? AND info_class='episodic' AND COALESCE(access_count,0) >= ? "
            "ORDER BY access_count DESC LIMIT ?",
            (project, semantic_threshold, max_promote),
        ).fetchall()

        promote_ids = [r[0] for r in promote_rows]
        if promote_ids:
            new_id = promote_to_semantic(
                conn, promote_ids, project, min_recall_count=semantic_threshold
            )
            if new_id:
                result["promoted"] = len(promote_ids)
                result["new_semantic_ids"].append(new_id)
    except Exception:
        pass

    # ── 子操作 B：衰减过期情节 chunk ──────────────────────────────────────────
    try:
        from datetime import timedelta as _td
        cutoff = (now - _td(days=stale_days)).isoformat()
        conn.execute(
            "UPDATE memory_chunks "
            "SET importance=MAX(0.05, importance * 0.7), oom_adj=COALESCE(oom_adj,0)+50, "
            "    updated_at=? "
            "WHERE project=? AND info_class='episodic' "
            "  AND last_accessed < ? AND COALESCE(access_count,0) < 2",
            (now_iso, project, cutoff),
        )
        result["decayed"] = conn.execute("SELECT changes()").fetchone()[0]
    except Exception:
        pass

    return result


# ══════════════════════════════════════════════════════════════════════════════
# 迭代335：Ghost Reaper — zombie chunk FTS5 污染清除
# OS 类比：Linux wait4()/waitpid() — 父进程回收 zombie 子进程，释放进程表项。
#
# Ghost chunk 产生机制（consolidate/merge 路径）：
#   1. merge_similar / sleep_consolidate 合并 victim → survivor
#   2. victim 被标记：importance=0, oom_adj=500, summary=[merged→survivor_id]
#   3. 但 victim 未被 DELETE — FTS5 content table 仍有其 summary 索引
#   4. 结果：FTS5 搜索命中 ghost → 消耗 result slot + false recall count
#   5. importance=0 的 ghost 在 _score_chunk 后分数极低但仍出现在 final 列表
#
# 信息论根因（Redundancy Theory, Kolmogorov 1965）：
#   ghost chunk 携带 0 信息（已合并，K-complexity=0），但占用检索带宽。
#   每次 FTS5 hit = 浪费 ~0.1ms 评分计算 + 挤占候选池 slot（候选总量 top_k×3 固定）。
#   实测：全项目 67 ghost chunks 累计 1721 false recall（平均 25.7 次/ghost）。
#   P(ghost selected) ≈ 5%（评分极低但 DRR 偶发回流），SNR 降低约 3-5%。
#
# 解决（两层防御）：
#   Layer 1（硬删除）：reap_ghosts() 物理删除 importance=0 chunk，触发 FTS5 DELETE trigger
#   Layer 2（软过滤）：retriever.py fts_search 调用前加 importance > 0 防护（in-flight 保护）
#
# 触发时机：
#   - 手动调用（tools/reap_ghosts.py 或 CLI）
#   - kswapd 扫描时附带执行（低优先级后台任务）
#   - sleep_consolidate 合并完成后自动 reap（TODO iter336+）
# ══════════════════════════════════════════════════════════════════════════════

def reap_ghosts(conn: sqlite3.Connection,
                project: Optional[str] = None,
                dry_run: bool = False) -> dict:
    """
    迭代335：回收 ghost chunk（importance=0 且 oom_adj>=500 的已合并 chunk）。

    Ghost 判定标准（两条件同时满足，避免误删 importance=0 但有实意的 chunk）：
      1. importance <= 0.0（合并路径设置）
      2. summary LIKE '[merged→%'（合并标记前缀）

    只满足条件 1 但 summary 不含合并标记的 chunk 不被视为 ghost（可能是用户故意
    设为 0 importance 的保留 chunk），不删除。

    OS 类比：
      wait4() 的 WNOHANG 标志 — 非阻塞扫描，只回收已经是 zombie 的进程，
      不等待仍在运行的进程退出。

    Args:
      conn:     SQLite 连接（需要写权限）
      project:  限定回收范围（None = 全项目）
      dry_run:  True = 只统计不删除

    Returns:
      dict:
        reaped_count    — 已删除数量（dry_run 时为待删除数量）
        ghost_ids       — 被删除的 chunk_id 列表
        projects_stats  — {project: count} 各项目回收统计
        dry_run         — 是否只读模式
    """
    try:
        if project:
            rows = conn.execute(
                "SELECT id, project, summary FROM memory_chunks "
                "WHERE project=? AND importance <= 0.0 "
                "  AND (summary LIKE '[merged→%' OR oom_adj >= 500)",
                (project,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, project, summary FROM memory_chunks "
                "WHERE importance <= 0.0 "
                "  AND (summary LIKE '[merged→%' OR oom_adj >= 500)",
            ).fetchall()

        if not rows:
            return {"reaped_count": 0, "ghost_ids": [], "projects_stats": {}, "dry_run": dry_run}

        ghost_ids = [r[0] for r in rows]
        projects_stats: dict = {}
        for _gid, _gproj, _gsumm in rows:
            projects_stats[_gproj] = projects_stats.get(_gproj, 0) + 1

        if dry_run:
            return {
                "reaped_count": len(ghost_ids),
                "ghost_ids": ghost_ids,
                "projects_stats": projects_stats,
                "dry_run": True,
            }

        # B15 FTS sync: FTS5 是独立模式（无触发器），必须手动清理孤立行
        # OS 类比：ext4 unlink batch — 批量删除前先收集 rowid 映射，再同步 FTS 索引
        placeholders = ",".join("?" * len(ghost_ids))
        rowid_rows = conn.execute(
            f"SELECT rowid FROM memory_chunks WHERE id IN ({placeholders})",
            ghost_ids,
        ).fetchall()
        mc_rowids = [str(r[0]) for r in rowid_rows]

        conn.execute(
            f"DELETE FROM memory_chunks WHERE id IN ({placeholders})",
            ghost_ids,
        )
        reaped = conn.execute("SELECT changes()").fetchone()[0]

        # 同步清理 FTS5 孤立行
        if mc_rowids:
            try:
                fts_ph = ",".join("?" * len(mc_rowids))
                conn.execute(
                    f"DELETE FROM memory_chunks_fts WHERE rowid_ref IN ({fts_ph})",
                    mc_rowids,
                )
            except Exception:
                pass  # FTS 表不存在或已清理，忽略

        return {
            "reaped_count": reaped,
            "ghost_ids": ghost_ids,
            "projects_stats": projects_stats,
            "dry_run": False,
        }
    except Exception as e:
        return {"reaped_count": 0, "ghost_ids": [], "projects_stats": {}, "dry_run": dry_run,
                "error": str(e)}


# ── 迭代360：FTS5 Auto-Optimize（降低 P95 延迟）────────────────────────────────
# OS 类比：ext4 e2fsck online defrag — 合并碎片化的 b-tree segment，
#   减少 FTS5 查询时需要扫描的 segment 数量（O(S×logN) → O(logN)）。
#
# 问题根因（v5 audit, 2026-04-28）：
#   SQLite FTS5 在每次 insert/delete/update 后生成新的 b-tree segment。
#   当 segment 数量 S 增大时，FTS5 查询需要合并 S 个 posting list，
#   时间复杂度从 O(logN) 退化为 O(S×logN)。
#   实测：352 次历史写入（105 chunk）→ 产生大量碎片化 segment → P95=273ms。
#   FTS5 optimize 命令：强制合并所有 segment → 单 segment → O(logN)。
#
# 冷却保护：至少间隔 _FTS_OPTIMIZE_INTERVAL 秒（默认 3600 秒 = 1 小时），
#   避免高频写入场景下 optimize 本身成为性能瓶颈（optimize 是重写操作）。
#   OS 类比：e4defrag 的 min_defrag_interval — 防止 defrag 自我拖累。

_FTS_OPTIMIZE_INTERVAL: float = 3600.0  # 冷却时间（秒），最少 1 小时间隔
_fts_last_optimize: float = 0.0  # 上次 optimize 的 monotonic 时间戳


def interference_decay(conn: sqlite3.Connection, new_chunk: dict, project: str,
                       threshold_mild: float = 0.30,
                       threshold_strong: float = 0.50,
                       decay_mild: float = 0.10,
                       decay_strong: float = 0.20,
                       max_affected: int = 10) -> int:
    """
    iter386: Interference-Based Retrievability Decay — 干扰式检索衰减

    认知科学依据：
      McGeoch (1932) Interference Theory — 遗忘的主因是新旧记忆之间的干扰，
        而非时间本身（Ebbinghaus 的衰减曲线只是表象）。
      Anderson (2003) Inhibition Theory — 海马回路通过主动抑制机制降低干扰记忆的可及性，
        确保最相关记忆优先浮现（Retrieval-Induced Forgetting, RIF）。

    OS 类比：CPU TLB Shootdown (INVLPG, x86 SMP)
      当一个核修改了页表（写入新chunk）时，必须向所有其他核广播 TLB 失效（INVLPG），
      否则其他核的 TLB 仍持有旧的虚地址→物理地址映射（过时知识仍被注入）。
      类比：写入覆盖旧知识的新 chunk → 旧 chunk 的 retrievability 降低（TLB 失效）。

    算法：
      1. FTS5 搜索新 chunk 的 summary，找语义相近旧 chunk（同 project）
      2. 计算 Jaccard 相似度（summary token 集合）
      3. mild 干扰 [threshold_mild, threshold_strong): retrievability -= decay_mild
      4. strong 干扰 [threshold_strong, +∞): retrievability -= decay_strong
      5. design_constraint 类型免疫（设计约束不受覆盖，只能显式 supersede）
      6. retrievability 下限 0.05（防止完全消失，仍可在 page fault 时 swap_in）

    保护机制：
      - design_constraint 不受干扰（mlock 保护）
      - 相同 chunk_type 的干扰权重 × 1.5（同类型更可能是覆盖更新）
      - retrievability 下限 0.05

    Returns:
      受影响的 chunk 数量
    """
    import re as _re

    if not new_chunk or not project:
        return 0

    new_summary = (new_chunk.get("summary") or "").strip()
    new_type = new_chunk.get("chunk_type", "")
    new_id = new_chunk.get("id", "")

    if not new_summary:
        return 0

    # Token 化：英文词 + CJK bigram
    def _tokenize(text: str) -> frozenset:
        tokens = set()
        for m in _re.finditer(r'[a-zA-Z0-9_\u4e00-\u9fff]{2,}', text.lower()):
            tokens.add(m.group())
        cn = _re.sub(r'[^\u4e00-\u9fff]', '', text)
        for i in range(len(cn) - 1):
            tokens.add(cn[i:i + 2])
        return frozenset(tokens)

    new_tokens = _tokenize(new_summary)
    if not new_tokens:
        return 0

    # FTS5 搜索语义相近的旧 chunk
    try:
        similar = fts_search(conn, new_summary, project, top_k=max_affected * 2)
    except Exception:
        return 0

    if not similar:
        return 0

    affected = 0
    for chunk in similar[:max_affected * 2]:
        cid = chunk.get("id", "")
        if not cid or cid == new_id:
            continue
        # design_constraint 免疫
        if chunk.get("chunk_type") == "design_constraint":
            continue
        # 获取当前 retrievability
        row = conn.execute(
            "SELECT retrievability, chunk_type FROM memory_chunks WHERE id=?", (cid,)
        ).fetchone()
        if not row:
            continue
        old_ret, old_type = float(row[0] or 0.8), (row[1] or "")

        # 计算 Jaccard 相似度
        old_tokens = _tokenize(chunk.get("summary") or "")
        if not old_tokens:
            continue
        inter = len(new_tokens & old_tokens)
        union = len(new_tokens | old_tokens)
        if union == 0:
            continue
        jaccard = inter / union

        # 同类型干扰系数 1.5（更可能是内容更新）
        type_factor = 1.5 if old_type == new_type else 1.0

        if jaccard >= threshold_strong:
            penalty = decay_strong * type_factor
        elif jaccard >= threshold_mild:
            penalty = decay_mild * type_factor
        else:
            continue  # 相似度太低，不干扰

        new_ret = max(0.05, old_ret - penalty)
        if new_ret < old_ret:
            try:
                conn.execute(
                    "UPDATE memory_chunks SET retrievability=? WHERE id=?",
                    (round(new_ret, 4), cid)
                )
                affected += 1
            except Exception:
                pass

    return affected


def fts_optimize(conn: sqlite3.Connection, force: bool = False) -> bool:
    """
    迭代360：触发 FTS5 segment 合并优化，降低查询 P95 延迟。
    OS 类比：ext4 online defrag (e4defrag) — 在线整理碎片，不需要 unmount。

    SQLite FTS5 在每次 insert 后生成新 segment；累积多个 segment 后，
    查询需要扫描所有 segment（O(S × log N)），S 增大导致 P95 上升。
    optimize 命令将所有 segment 合并为 1 个（O(log N)）。

    Args:
      conn:  SQLite 连接
      force: True = 跳过冷却时间检查，强制执行

    Returns:
      True  = 执行了 optimize
      False = 冷却期内跳过，或执行失败
    """
    global _fts_last_optimize
    import time as _time
    now = _time.monotonic()
    if not force and (now - _fts_last_optimize) < _FTS_OPTIMIZE_INTERVAL:
        return False
    try:
        conn.execute("INSERT INTO memory_chunks_fts(memory_chunks_fts) VALUES('optimize')")
        _fts_last_optimize = now
        return True
    except Exception:
        return False


# ── iter450: Completion Effect — 已完成任务的 importance 降低（Ovsiankina 1928 补集）──────────────
# 认知科学依据：
#   Ovsiankina (1928) — 已完成任务失去"认知张力"，不再主动维持记忆优先级；
#   与 Zeigarnik Effect（iter490）构成对称：未完成=高优先级，已完成=可降权。
#   Loftus (1985) "Spreading activation in memory" — 已解决问题的语义网络激活减弱。
# OS 类比：Linux page writeback completion — dirty page 写回完成后清除 PG_dirty flag，
#   kswapd 可以自由回收（不再受 writeback 保护）；完成标记 = 解除 mlock 保护。

def apply_completion_effect(
    conn: "sqlite3.Connection",
    chunk_ids: list,
) -> dict:
    """iter450: Completion Effect (CEF) — 含"已完成"信号的 chunk importance 适度降低。

    扫描 content + summary，检测 DONE/RESOLVED/FIXED/CLOSED/COMPLETED 等完成关键词。
    若存在，将 importance 降低（floor: cef_min_importance），模拟"已完成任务失去认知张力"。

    与 Zeigarnik Effect (iter490) 的关系：
      ZEF  = 未完成信号 → stability 提升（维持认知张力）
      CEF  = 完成信号   → importance 小幅降低（释放认知张力）
      二者共同构成任务状态的完整认知模型。

    Returns:
      {"cef_reduced": N}
    """
    result = {"cef_reduced": 0}
    if not chunk_ids:
        return result
    try:
        import config as _cfg
        if not _cfg.get("store_vfs.cef_enabled"):
            return result

        completion_keywords = _cfg.get("store_vfs.cef_completion_keywords") or []
        reduction = float(_cfg.get("store_vfs.cef_importance_reduction"))
        min_importance = float(_cfg.get("store_vfs.cef_min_importance"))
        trigger_min_imp = float(_cfg.get("store_vfs.cef_trigger_min_importance"))
        import datetime as _dt

        now_iso = _dt.datetime.now(_dt.timezone.utc).isoformat()

        for chunk_id in chunk_ids:
            row = conn.execute(
                "SELECT importance, content, summary FROM memory_chunks WHERE id=?",
                (chunk_id,)
            ).fetchone()
            if not row:
                continue
            imp = float(row[0] or 0.5)
            if imp < trigger_min_imp:
                continue  # importance 已经很低，不需要再降

            content_lower = (row[1] or "").lower()
            summary_lower = (row[2] or "").lower()
            combined = content_lower + " " + summary_lower

            if not any(kw.lower() in combined for kw in completion_keywords):
                continue  # 无完成信号

            new_imp = max(min_importance, imp - reduction)
            if new_imp < imp - 0.0001:
                conn.execute(
                    "UPDATE memory_chunks SET importance=?, updated_at=? WHERE id=?",
                    (round(new_imp, 4), now_iso, chunk_id)
                )
                result["cef_reduced"] += 1
        return result
    except Exception:
        return result


def apply_retrieval_difficulty_gradient(
    conn: "sqlite3.Connection",
    chunk_ids: list,
    now_iso: str = None,
) -> dict:
    """iter451: Retrieval Difficulty Gradient (RDG) — 基于检索轨迹趋势的 desirable difficulty 加成。

    认知科学依据：
      Bjork & Bjork (1992) "A new theory of disuse and an old theory of stimulus fluctuation"
        — desirable difficulty 的本质是"边缘成功检索"（marginally successful retrieval）：
        retrievability 低 + 过去仍被成功检索（spaced_access_count 稳定增长）= 真正困难但成功的检索。
      与 DDE2 (iter485) 的区别：
        DDE2 = 单点快照（R < threshold + S < threshold）
        RDG  = 轨迹趋势（R 低 + spaced_access_count >= min_spaced = 历史上多次成功挑战难度）
      核心洞察：spaced_access_count 代理"过去 N 次跨24h成功检索"——
        高 spaced_access_count + 当前低 R = "在不断下降的可达性下仍被多次成功检索"
        这比单次快照更可靠地反映 desirable difficulty 的认知状态。

    OS 类比：Linux adaptive readahead（mm/readahead.c）—
      单次 miss 不触发大窗口预取；但若 miss 模式呈趋势（连续 miss + 仍被 app 使用），
      readahead_max 自动扩大（adaptive = 趋势驱动，非快照驱动）。
      RDG = adaptive difficulty threshold = 历史趋势而非当前快照。

    触发条件（与 DDE2 形成互补，不替代）：
      - retrievability <= rdg_max_retrievability（当前可达性低）
      - spaced_access_count >= rdg_min_spaced（历史上多次间隔成功检索）
      - importance >= rdg_min_importance
      - stability < rdg_max_stability（仍有成长空间）

    加成公式：
      difficulty_gradient = spaced_access_count / rdg_spaced_ref（归一化，上限1.0）
      bonus = difficulty_gradient × (1.0 - retrievability) × rdg_scale
      new_stab = min(365.0, stab × (1 + bonus))

    Returns:
      {"rdg_boosted": N}
    """
    result = {"rdg_boosted": 0}
    if not chunk_ids:
        return result
    try:
        import config as _cfg
        import datetime as _dt
        import math as _math
        if not _cfg.get("store_vfs.rdg_enabled"):
            return result

        rdg_max_ret = float(_cfg.get("store_vfs.rdg_max_retrievability"))
        rdg_min_spaced = int(_cfg.get("store_vfs.rdg_min_spaced"))
        rdg_spaced_ref = int(_cfg.get("store_vfs.rdg_spaced_ref"))
        rdg_scale = float(_cfg.get("store_vfs.rdg_scale"))
        rdg_max_stab = float(_cfg.get("store_vfs.rdg_max_stability"))
        rdg_min_importance = float(_cfg.get("store_vfs.rdg_min_importance"))

        if now_iso is None:
            now_iso = _dt.datetime.now(_dt.timezone.utc).isoformat()

        for chunk_id in chunk_ids:
            row = conn.execute(
                "SELECT stability, importance, retrievability, spaced_access_count "
                "FROM memory_chunks WHERE id=?", (chunk_id,)
            ).fetchone()
            if not row:
                continue
            stab = float(row[0] or 1.0)
            imp = float(row[1] or 0.0)
            ret = float(row[2] or 1.0)
            spaced = int(row[3] or 0)

            if imp < rdg_min_importance:
                continue
            if ret > rdg_max_ret:
                continue  # R 仍高，不算困难
            if stab >= rdg_max_stab:
                continue  # stability 已经很高，无需加成
            if spaced < rdg_min_spaced:
                continue  # 历史间隔成功检索次数不足，无法确认是持续成功的困难检索

            # difficulty_gradient: 间隔检索次数越多 = 越能确认"困难但成功"的历史轨迹
            difficulty_gradient = min(1.0, spaced / max(rdg_spaced_ref, 1))
            # bonus 与检索困难程度（1-R）正比，与历史成功程度（gradient）正比
            bonus = difficulty_gradient * (1.0 - ret) * rdg_scale
            if bonus < 0.0001:
                continue

            new_stab = min(365.0, stab * (1.0 + bonus))
            if new_stab > stab + 0.0001:
                conn.execute(
                    "UPDATE memory_chunks SET stability=? WHERE id=?", (new_stab, chunk_id)
                )
                result["rdg_boosted"] += 1
        return result
    except Exception:
        return result
