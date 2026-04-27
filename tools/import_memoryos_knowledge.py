#!/usr/bin/env python3
"""
import_memoryos_knowledge.py — 将 memory-os 自身架构知识导入 global store
iter132: 解决 FTS5 self-referential miss 问题

问题根因：DB 中没有 memory-os 自身的实现知识（kswapd/retriever/extractor/PSI/TLB等），
导致 memory-os 开发相关 query 始终触发 BM25 fallback（全表扫描），
而不是精确的 FTS5 bigram 匹配。

解法：将 memory-os 架构知识作为 design_constraint + decision + reasoning_chain 写入 global tier，
让 FTS5 能匹配 kswapd/retriever/extractor 等核心关键词。

OS 类比：Linux kernel.ko 加载 — 把内核模块的符号表注册到全局符号表，
         其他模块就能通过符号解析找到对应实现（而非盲目搜索整个内存）。
"""

import os, sys, json, hashlib
from pathlib import Path
from datetime import datetime, timezone

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(str(Path(__file__).resolve().parent.parent))

from store_core import open_db, ensure_schema, insert_chunk, already_exists

os.environ.setdefault("MEMORY_OS_DIR", str(Path.home() / ".claude" / "memory-os"))

PROJECT_ID = "global"
DRY_RUN = "--dry-run" in sys.argv

stats = {"imported": 0, "skipped_dup": 0}


def make_chunk(chunk_type, summary, content, importance=0.75, tags=None):
    chunk_id = f"memoryos-{hashlib.md5(summary.encode()).hexdigest()[:12]}"
    now = datetime.now(timezone.utc).isoformat()
    return {
        "id": chunk_id,
        "created_at": now,
        "updated_at": now,
        "project": PROJECT_ID,
        "source_session": "import:memoryos_architecture",
        "chunk_type": chunk_type,
        "content": content[:2000],
        "summary": summary[:120],
        "tags": json.dumps(tags or [], ensure_ascii=False),
        "importance": importance,
        "retrievability": 1.0,
        "embedding": "[]",
        "access_count": 0,
        "last_accessed": now,
        "lru_gen": 0,
        "oom_adj": 0,
    }


# ── memory-os 架构知识库 ──────────────────────────────────────────────────────
# 按子系统分类，每条包含：英文术语 + 中文术语 + OS类比名称
# 确保 FTS5 bigram 可以命中相关查询

KNOWLEDGE_CHUNKS = [

    # ── 1. 整体架构 ──────────────────────────────────────────────────────────
    ("design_constraint",
     "memory-os 架构定位：用 OS 内核哲学管理 AI Agent 的 context window 资源",
     """memory-os 是 Claude Code 的内存管理层：
- context window ≈ 物理内存（RAM）
- store.db ≈ 磁盘/swap 分区
- hooks 执行链 ≈ 系统调用路径（syscall）
- session ≈ 进程（process）
- chunk ≈ 内存页（page）
- chunk_type ≈ 内存分区（zone/segment）
核心四层：SessionStart(loader) / UserPromptSubmit(retriever+writer) / PostToolUse(compressor+profiler) / Stop(extractor+CRIU)
所有知识存储在 ~/.claude/memory-os/store.db，项目隔离通过 project 字段实现。""",
     0.90),

    ("design_constraint",
     "memory-os 检索管道（retriever）：FTS5→scorer→madvise→router，三优先级 SKIP/LITE/FULL",
     """retriever.py 是 UserPromptSubmit hook 的核心：
1. vDSO Fast Path：SKIP（确认词）→ TLB v2 命中 → 完整检索
2. 三优先级调度（nice值类比）：
   - SKIP(nice 19)：确认词/闲聊，零 I/O 直接返回
   - LITE(nice 0)：普通 query，只跑 FTS5
   - FULL(nice -20)：长 query/多实体/缺页，FTS5+scorer+router
3. FTS5 检索：SQLite FTS5 虚拟表，bigram CJK 分词，O(log N)
4. BM25 fallback：FTS5 无结果时全表扫描+Python BM25
5. DRR Fair Queuing：防止单一 chunk_type 独占 Top-K
6. 注入格式：json {"hookSpecificOutput": {"additionalContext": "..."}}
关键文件：hooks/retriever.py（1985行）""",
     0.90),

    ("design_constraint",
     "memory-os 提取管道（extractor）：Stop hook 从对话提取 decision/constraint/reasoning_chain",
     """extractor.py 是 Stop hook 的知识提取器：
- 从最后 N 轮对话中提取知识 chunk
- chunk_type 分类：decision/excluded_path/reasoning_chain/conversation_summary/design_constraint/quantitative_evidence/causal_chain/procedure
- importance 由提取信号强度决定（design_constraint=0.95, decision=0.85）
- 去重：already_exists() + find_similar() Jaccard 相似度
- KSM 合并：相似 chunk 合并，追加 summary 到 content
- AIMD 流控：命中率低时收缩提取窗口（cwnd）
- cgroup throttle：接近 quota 时降低新写入 importance
关键文件：hooks/extractor.py""",
     0.85),

    # ── 2. 内存管理子系统 ────────────────────────────────────────────────────
    ("decision",
     "kswapd 三水位线淘汰机制：quota 80%/90%/95% 触发 ZONE_LOW/MIN/OOM 换出",
     """kswapd_scan() 在 SessionStart 时执行：
- ZONE_OK（<80%）：不淘汰
- ZONE_LOW（80-90%）：后台预淘汰，按 retention_score 换出低分 chunk 到 swap_chunks
- ZONE_MIN（90-95%）：批量换出，batch_size=5
- ZONE_OOM（>95%）：强制同步淘汰，OOM killer 清理最低分 chunk
淘汰策略：oom_adj 正值优先淘汰；oom_adj <= -1000 绝对保护（不可淘汰）
相关 sysctl：kswapd.pages_low_pct=80, pages_high_pct=90, pages_min_pct=95, stale_days=90
关键文件：store.py:kswapd_scan()""",
     0.85),

    ("decision",
     "swap 换出/换入机制：低频 chunk 写入 swap_chunks，缺页时 demand paging 换入",
     """swap_out()：将 chunk 压缩序列化写入 swap_chunks 表，从主表删除
swap_in()：将 swap_chunks 解压恢复到 memory_chunks
swap_fault()：BM25 搜索 swap 中是否有匹配的被换出 chunk
page_fault_log.json：extractor 写入知识缺口信号，retriever 消费并触发 demand paging
swap_fault 触发条件：retriever 正常检索 Top-K 为空时（priority=FULL，未超 deadline）
相关 sysctl：swap.max_chunks=100, fault_top_k=2, min_importance_for_swap=0.5
关键文件：store_swap.py, hooks/retriever.py:_read_page_fault_log()""",
     0.85),

    ("decision",
     "MGLRU 多代 LRU：chunk 按 lru_gen 分代管理，老代优先淘汰",
     """MGLRU（Multi-Gen LRU）在 mglru_aging() 中执行：
- lru_gen=0 是最新代（youngest），gen值越大越老（oldest）
- mglru_promote()：检索命中时将 chunk 的 lru_gen 重置为 0（promote to newest gen）
- mglru_aging()：SessionStart 时按间隔递增全部 chunk 的 lru_gen
- evict_lowest_retention()：按 lru_gen DESC 排序，先淘汰最老代
- max_gen=4，超过不再递增
- aging_interval_hours=6，防止频繁推进
相关 sysctl：mglru.max_gen=4, mglru.aging_interval_hours=6
关键文件：store_vfs.py, store.py:mglru_aging()""",
     0.80),

    ("decision",
     "DAMON 内存访问监控：扫描 cold/dead chunk 主动回收",
     """damon_scan() 主动扫描未访问 chunk：
- COLD：创建 > damon.cold_age_days(14) 且 access_count=0 → 提高 oom_adj（加速淘汰）
- DEAD：创建 > damon.dead_age_days(30) 且 access_count=0 且 importance < dead_importance_max(0.65)
  → swap out 或直接删除（importance 极低时）
- 受 soft/hard pin 保护：hard pin 的 chunk 完全跳过
- max_actions_per_scan=10，避免单次扫描过久
关键文件：store.py:damon_scan()""",
     0.80),

    ("decision",
     "OOM Score 机制：oom_adj 控制 chunk 淘汰优先级（-1000 绝对保护到 +1000 优先淘汰）",
     """oom_adj 参考 Linux /proc/PID/oom_score_adj（-1000~+1000）：
- oom_adj = -1000：绝对保护，任何淘汰路径跳过（kswapd ZONE_MIN 除外）
- oom_adj = -800：强保护，自动分配给 design_constraint / quantitative_evidence
- oom_adj = 0：正常（大多数 decision/reasoning_chain）
- oom_adj = +300 (cgroup throttle)：超配额时新写入自动获得
- oom_adj = +500 (OOM auto disposable)：prompt_context 自动分配
OOM killer：evict_lowest_retention() 在 oom_adj 基础上计算 retention_score，正值让 score 下降
关键文件：store.py, store_vfs.py:evict_lowest_retention()""",
     0.85),

    # ── 3. 检索子系统 ────────────────────────────────────────────────────────
    ("decision",
     "FTS5 全文索引：SQLite FTS5 bigram CJK 分词，O(log N) 检索替代 O(N) 全表扫描",
     """FTS5 实现（store_vfs.py）：
- 独立模式（非 content-sync）：存储 _cjk_tokenize 处理后的文本
- _cjk_tokenize：CJK 单字 + bigram + 英文 Porter stemming
- _normalize_structured_summary：展开 [topic]/>/ 分隔符
- fts_search()：FTS5 MATCH 查询 + JOIN memory_chunks 还原完整 chunk
- Always-merge global（iter123）：始终追加搜 global project，合并去重排序
- Schema version 124：修复历史 UUID rowid_ref 污染（FTS5 JOIN CAST(rowid_ref AS INTEGER)）
FTS5 失效条件：DB 中没有包含查询词 bigram 的 chunk（此时触发 BM25 fallback）
关键文件：store_vfs.py:_ensure_fts5(), fts_search()""",
     0.85),

    ("decision",
     "TLB v2 检索缓存：multi-slot prompt_hash + chunk_version 避免重复检索相同结果",
     """TLB（Translation Lookaside Buffer，iter57→64）：
- 缓存文件：~/.claude/memory-os/.last_tlb.json
- 格式：{chunk_version: int, slots: {prompt_hash: {injection_hash: str}}}
- L1 命中：prompt_hash + chunk_version 完全匹配 → 零 I/O 退出
- L2 命中：chunk_version 匹配 + 任意 slot injection_hash 一致 → 跳过
- 失效条件：chunk_version 变化（新增/删除 chunk），而非 db_mtime（写元数据不触发）
- chunk_version：在 ~/.claude/memory-os/.chunk_version 中，bump_chunk_version() 递增
- vDSO Fast Path（iter61）：TLB 检查在 heavy import 之前执行（节省 ~27ms import 开销）
关键文件：hooks/retriever.py:_tlb_read(), _tlb_write()""",
     0.85),

    ("decision",
     "PSI 压力感知调度：retriever 检测系统压力，FULL→LITE 动态降级",
     """PSI（Pressure Stall Information，iter36）：
- psi_stats()：从 recall_traces 统计最近 N 次检索的延迟/命中率/容量
- three维度：retrieval(延迟) / quality(命中率) / capacity(容量使用率)
- overall=FULL 时：priority 从 FULL 降级到 LITE，关闭 knowledge_router
- PSI 自适应基线（iter60）：用滑动窗口 P50×margin 替代固定 latency_baseline_ms
- PSI Noise Floor：skipped_same_hash 记录 duration_ms=0，不污染 PSI 基线
关键 sysctl：psi.window_size=20, latency_baseline_ms=30.0, hit_rate_baseline_pct=50.0
关键文件：store.py:psi_stats(), hooks/retriever.py""",
     0.80),

    ("decision",
     "DRR Fair Queuing：防止单一 chunk_type 独占 Top-K，保证类型多样性",
     """DRR（Deficit Round Robin，iter50）：
- _drr_select()：从已排序候选集选 Top-K，每个 chunk_type 最多占 drr_max_same_type=2 个槽
- overflow 回流：其他类型不足时从 overflow 补齐（保证填满 top_k）
- drr_enabled 默认 True，False 退化为纯 score 排序
- 实测问题：93% chunk 是 decision → DRR 前 Top-K 全是 decision
- DRR-aware forced constraints（iter130）：计算自然 top_k 中已有约束数，动态调整 forced 配额
关键文件：hooks/retriever.py:_drr_select()""",
     0.80),

    ("decision",
     "Hybrid FTS5+BM25 召回（iter126）：FTS5 结果不足时 BM25 补充长尾，降权避免劣质挤占",
     """混合召回策略（iter126）：
1. FTS5 先跑（精确 bigram 匹配，O(log N)）
2. 如果 FTS5 结果 < hybrid_fts_min_count(3)，BM25 全表扫描补充差额
3. BM25 补充 chunk 降权：relevance × 0.6（discount）
4. 合并后统一用 _unified_retrieval_score 排序
iter131 BM25 Global Discount：BM25 fallback（FTS5 完全失败时）中，
   global 项目 chunk 的 relevance × bm25_global_discount(0.4)，
   防止跨项目高 importance chunk 通过词汇偶然重叠排名第一
关键文件：hooks/retriever.py（use_fts 分支）""",
     0.85),

    # ── 4. 写入子系统 ────────────────────────────────────────────────────────
    ("decision",
     "writer hook：防抖去重批量写入，debounce 300s 避免同内容重复写",
     """writer.py 是 UserPromptSubmit hook 的写入器：
- debounce：300s 内相同 session 不重复写入（用 .last_write_time 文件判断）
- 写入内容：当前 prompt_context 快照（chunk_type='prompt_context'）
- importance=0.3，oom_adj=+500（低优先级，容易被淘汰）
- 与 extractor 分工：writer 写实时快照，extractor 写提炼后的知识
关键文件：hooks/writer.py""",
     0.75),

    ("decision",
     "Anti-Starvation 反饥饿机制（iter62）：饱和惩罚 + 饥饿加分防召回同质化",
     """scorer.py 的 anti-starvation 机制：
- 饱和惩罚（saturation penalty）：recall_count 越高，score 越低
  penalty = saturation_factor × log2(1 + recall_count)，上限 saturation_cap=0.25
- 饥饿加分（starvation boost）：access_count=0 的老 chunk 获得加分
  条件：age > starvation_min_age_days(0.5)，线性增长到 starvation_boost_factor(0.30)
- chunk_recall_counts()：统计最近 30 次检索中每个 chunk 的召回次数
目的：防止热门 chunk 被永久锁定在 Top-K，让零访问的新/冷知识有曝光机会
关键文件：scorer.py, hooks/retriever.py""",
     0.80),

    # ── 5. 调度子系统 ────────────────────────────────────────────────────────
    ("decision",
     "query 优先级分类（SKIP/LITE/FULL）：依据 query 特征分配检索资源",
     """_classify_query_priority() 决策逻辑：
0. sched_ext 自定义规则（用户可通过 sysctl.json 添加正则规则，优先于内置规则）
1. 有缺页信号 → FULL（demand paging 优先）
2. 技术实体 >= 2 → FULL（多维度检索）
2.5. 通用知识 query（"什么是 GIL"）→ SKIP（不注入项目知识，避免噪音）
3. prompt 匹配 SKIP 模式且无技术信号 → SKIP
4. prompt 极短且无技术信号 → SKIP
5. query < 200 字 → LITE
6. 默认 → FULL
技术信号：反引号代码/文件路径/中文技术词/error,bug/def,class,import/大写缩写
关键文件：hooks/retriever.py:_classify_query_priority()""",
     0.85),

    ("decision",
     "Deadline I/O Scheduler（iter41）：50ms soft deadline + 200ms hard deadline 分阶段超时",
     """Deadline scheduler 保证检索不超时：
- deadline_ms=50.0（soft）：超过时跳过低优先级阶段（madvise/readahead/router）
- deadline_hard_ms=200.0（hard）：超过时立即返回已有结果，跳过后续所有增强
- 阶段优先级（高→低）：FTS5 > scorer > madvise > readahead > swap_fault > router
- soft 超时后执行 graceful degradation（有什么结果返回什么）
- hard 超时后执行 emergency return（直接输出 FTS5+scorer 结果）
根因：WAL checkpoint 时 writer 持有写锁，reader 连接等待 → 超时
关键文件：hooks/retriever.py:_check_deadline()""",
     0.80),

    ("decision",
     "Context Pressure Governor（iter55）：对话轮次/compaction次数感知，动态缩放注入窗口",
     """governor 监测 context 压力并缩放 max_chars：
- LOW（turns<=5）：scale=1.5（多注入，提升信息密度）
- NORMAL（5<turns<15）：scale=1.0（正常注入）
- HIGH（turns>=15 或 compact>=2）：scale=0.6（精简注入）
- CRITICAL（turns>=25 或 compact>=4）：scale=0.3（最小注入）
- 触发源：recall_traces 中的 turns 计数 + compaction 事件
- consecutive_decay：超过 1h 未更新则重置 consecutive_high（防历史锁死）
关键文件：store.py:context_pressure_governor(), hooks/retriever.py""",
     0.80),

    # ── 6. 配置子系统 ────────────────────────────────────────────────────────
    ("design_constraint",
     "memory-os sysctl 注册表：所有可调参数统一在 config.py _REGISTRY 注册，运行时可调",
     """config.py 是 memory-os 的 /proc/sys/ 虚拟文件系统等价物：
- get(key, project=None)：读取 tunable，优先级：环境变量 > namespace > sysctl.json > 默认值
- sysctl_set(key, value, project=None)：运行时修改并持久化到 sysctl.json
- Namespaces（iter37）：per-project 配置覆盖，不影响全局值
- sched_ext（iter47）：用户可添加自定义调度规则（正则+优先级），优先于内置逻辑
主要参数分类：retriever.*/scheduler.*/scorer.*/extractor.*/kswapd.*/swap.*/mglru.*/damon.*
关键文件：config.py""",
     0.90),

    # ── 7. VFS 层 ────────────────────────────────────────────────────────────
    ("design_constraint",
     "memory-os VFS 统一数据访问层：所有 hook 通过 store.py/store_vfs.py 接口访问 DB，禁止直接 SQL",
     """VFS（Virtual File System，iter21）设计原则：
- store.py：对外暴露的统一接口（open_db, ensure_schema, get_chunks, fts_search 等）
- store_vfs.py：核心 CRUD + FTS5 + evict（从 store_core 拆分）
- store_swap.py：swap_out/swap_in/swap_fault
- store_criu.py：checkpoint/restore
- store_mm.py：内存管理（kswapd/damon/mglru/balloon）
- store_proc.py：进程信息（psi_stats/context_pressure_governor）
只读连接（iter84）：retriever 在检索阶段使用 immutable=1 连接，零锁竞争
Write-Back Phase（iter84）：输出后用写连接批量 flush（dmesg+update_accessed+trace+commit）
关键文件：store.py, store_vfs.py""",
     0.90),

    # ── 8. 关键迭代历史 ──────────────────────────────────────────────────────
    ("reasoning_chain",
     "memory-os 关键迭代节点：iter23 FTS5→iter57 TLB→iter61 vDSO→iter84 Read-Only→iter97 CJK bigram→iter139 autotune双向",
     """memory-os 演进关键节点（用于理解设计决策脉络）：
- iter21：VFS 统一数据访问层（store.py 抽象）
- iter23：FTS5 全文索引（O(N)→O(logN) 检索）
- iter28：Scheduler Nice Levels（SKIP/LITE/FULL 三优先级）
- iter29：dmesg Ring Buffer（结构化事件日志）
- iter33：Swap Fault（demand paging）
- iter36：PSI 压力感知（动态降级）
- iter41：Deadline I/O Scheduler（分阶段超时）
- iter44：MGLRU（多代 LRU）
- iter50：DRR Fair Queuing（类型多样性）
- iter57→64：TLB v2（multi-slot + chunk_version）
- iter61：vDSO Fast Path（Lazy Import，SKIP/TLB 在 heavy import 前）
- iter84：Read-Only Fast Path（immutable 连接 + Write-Back Phase）
- iter97：CJK bigram FTS5（单字+bigram 双索引）
- iter123：Always-merge global（global 知识始终参与 FTS5 排序）
- iter126：Hybrid FTS5+BM25（L1/L2 多级缓存）
- iter129：autotune 逻辑修复（高命中率不再错误地增加 top_k）
- iter130：DRR-aware forced constraints（防约束膨胀）
- iter131：BM25 Global Discount（BM25 fallback 路径跨项目噪音抑制）
- iter132：FTS5 自知识导入（import_memoryos_knowledge.py 解决 self-referential miss）
- iter135：LITE + FTS5 miss 早退（防 BM25 fallback 噪音注入到无关 query）
- iter136：autotune deadline_ms 自引用修复（p95 排除 deadline_skip/hard_deadline 轨迹，防正反馈膨胀）
- iter137：autotune chunk_quota 上限（autotune.chunk_quota_max=400，防高命中率无限推高 quota）
- iter138：kswapd.pages_low_pct 回弹（capacity<70% 时向默认 80 恢复，防单向衰减锁死低水位）
- iter139：autotune deadline_ms 双向调节（p95<baseline 时收缩 deadline 向默认 50ms，防历史膨胀锁死高位）""",
     0.85),

    # ── 9. 关键度量指标 ──────────────────────────────────────────────────────
    ("quantitative_evidence",
     "memory-os 核心性能指标：FTS5<10ms, BM25 fallback<50ms, P50<30ms, vDSO SKIP<1ms",
     """memory-os 实测关键指标：
检索延迟：
  - vDSO SKIP（确认词）：<1ms（零 import）
  - TLB L1/L2 命中：<3ms（只读文件 stat）
  - FTS5 完整检索：P50 ~10ms（含 import ~27ms）
  - BM25 全表扫描 fallback：20-50ms（72 chunks）
  - PSI 压力告警阈值：>30ms（latency_baseline_ms）
检索质量（A/B 测试 iter20-21）：
  - BM25 vs 纯重要性排序：Recall@3 +147%（58.3% vs 23.6%），MRR +320%
  - memory-os 辅助 vs 无记忆：8/12 胜，平均得分 3.55 vs 2.12（+68%）
  - Session Recall@3：94.2%（最新基准）
SKIP/TLB 占比（理论）：SKIP ~42%，TLB hit ~40% → 80% 请求零完整检索
关键文件：benchmarks/, eval_*.py""",
     0.90),
]


def import_all():
    conn = open_db()
    ensure_schema(conn)

    for args in KNOWLEDGE_CHUNKS:
        if len(args) == 3:
            chunk_type, summary, content = args
            importance = 0.75
        else:
            chunk_type, summary, content, importance = args

        stats["total"] = len(KNOWLEDGE_CHUNKS)
        if already_exists(conn, summary, chunk_type):
            stats["skipped_dup"] += 1
            print(f"  [skip] {summary[:60]}...")
            continue

        chunk = make_chunk(chunk_type, summary, content, importance)
        if DRY_RUN:
            print(f"  [dry-run] would insert [{chunk_type}] {summary[:60]}...")
        else:
            insert_chunk(conn, chunk)
            stats["imported"] += 1
            print(f"  [ok] [{chunk_type}] imp={importance} {summary[:60]}...")

    if not DRY_RUN:
        conn.commit()
    conn.close()

    print(f"\n{'[DRY-RUN] ' if DRY_RUN else ''}导入完成: {stats}")


if __name__ == "__main__":
    print(f"memory-os 自知识导入 (iter132) {'[DRY-RUN]' if DRY_RUN else ''}")
    print(f"目标 DB：{os.environ.get('MEMORY_OS_DB', Path.home() / '.claude/memory-os/store.db')}")
    print()
    import_all()
