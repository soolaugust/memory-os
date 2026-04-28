"""
memory-os Config — sysctl Runtime Tunables Registry

迭代 27：OS 类比 — Linux sysctl (1993)
迭代 37：OS 类比 — Linux Namespaces (2002-2013)

sysctl 背景（迭代27）：
  早期 Linux 内核参数用 #define 分散在各子系统源码中，
  修改需要重编译内核。sysctl (1993) 引入 /proc/sys/ 虚拟文件系统，
  将内核参数统一注册为可读写的虚拟文件，管理员运行时即可调参：
    sysctl vm.swappiness=60
    sysctl net.core.somaxconn=1024

Namespaces 背景（迭代37）：
  Linux Namespaces (2002 mount ns → 2013 user ns) 让每个容器看到
  独立的资源视图（PID/NET/MNT/UTS/IPC/USER），同一物理主机上的
  不同容器可以有不同的进程号空间、网络栈、文件系统挂载等。
  Docker/K8s 的核心隔离机制就是 namespace。

  memory-os 当前问题：
    sysctl 是全局的——所有项目共享同一套 tunable 值。
    但不同项目特征差异大：
      - 大项目（1000+ chunk）需要 quota=500、激进 kswapd
      - 小项目（<50 chunk）默认 quota=200 足够
      - 某些项目需要更宽松的 scheduler（不 SKIP 短查询）
    全局配置无法满足多租户差异化需求。

  解决：
    per-project namespace 覆盖层——sysctl.json 支持 namespaces 字段：
      {"namespaces": {"git:abc123": {"extractor.chunk_quota": 500}}}
    get(key, project=None) 新增 project 参数：
      优先级：环境变量 > namespace(project) > global sysctl.json > 默认值
    sysctl_set(key, value, project=None)：无 project 时写全局，有则写 namespace。
    ns_list(project) / ns_clear(project)：管理 namespace。

解决（迭代27）：
  1. _REGISTRY: dict — 所有 tunable 的单点注册表（名称/默认值/类型/范围/描述）
  2. get(key) — 优先级：环境变量 > sysctl.json 配置文件 > 默认值
  3. sysctl_list() — 返回所有当前值（≈ sysctl -a）
  4. sysctl_set(key, value) — 运行时修改 + 持久化到 sysctl.json
"""
# ── 迭代157：__future__ annotations — 消除 typing 模块 import 开销 ────────────
# OS 类比：Linux lazy symbol resolution (ELF RTLD_LAZY) — 符号引用推迟到首次调用时解析，
#   而非 dlopen 时立即解析所有符号（RTLD_NOW）。
#   Python __future__ annotations (PEP 563, 3.7+) 让所有注解变为字符串（惰性求值），
#   不需要在模块加载时实际执行 Any/Optional 的名称查找。
#   config 是 retriever.py 在 heavy import 之前就加载的轻量级模块，
#   每次 hook 调用都付一次 typing import 成本（~3.9ms）。
#   __future__ annotations 完全消除这个成本：注解变字符串，typing 名称无需在运行时存在。
from __future__ import annotations
import json
import os
from datetime import datetime, timezone

# ── 迭代158：Replace pathlib.Path with os.path — 消除 pathlib import 开销 (~7ms) ──
# OS 类比：Linux kernel 的 vfs_stat() 直接调用 sys_stat() syscall，不经 glibc 路径抽象层，
#   消除中间层开销。pathlib.Path 是 os.path 的面向对象封装层，
#   而 os 模块在 Python 启动时已预加载（在 sys.modules 中），import os.path 近乎零成本。
#   pathlib 需要额外 ~6.88ms 加载。config.py 是 retriever.py 在 vDSO Stage 0+1 快速路径
#   之前就加载的轻量模块，每次 hook 调用都付这个成本。
#   替换策略：MEMORY_OS_DIR/SYSCTL_FILE 改为 str，所有文件操作改用 os.path/open()。

# 迭代90：Check os.environ first (set by conftest or tmpfs), fallback to default
_mem_dir_env = os.environ.get("MEMORY_OS_DIR")
MEMORY_OS_DIR: str = _mem_dir_env if _mem_dir_env else os.path.join(os.path.expanduser("~"), ".claude", "memory-os")
SYSCTL_FILE: str = os.path.join(MEMORY_OS_DIR, "sysctl.json")

# ── Tunable Registry ─────────────────────────────────────────────
# 每项：(default, type, min, max, env_key, description)
# env_key: 对应的环境变量名（向后兼容已有的 env 覆盖）
# min/max: None 表示不限

_REGISTRY: dict = {
    # ── retriever ──
    "retriever.top_k": (5, int, 1, 20, None,
        "默认召回 Top-K 条数（迭代72：3→5 提升知识覆盖率）"),
    "retriever.top_k_fault": (7, int, 1, 30, None,
        "有缺页信号时的扩大召回 Top-K（迭代72：5→7）"),
    "retriever.max_context_chars": (800, int, 100, 5000, None,
        "召回注入的最大字符数（迭代72：600→800 支撑 Top-5）"),
    "retriever.max_context_chars_fault": (1000, int, 100, 5000, None,
        "缺页场景的最大注入字符数（迭代72：800→1000 支撑 Top-7）"),
    "retriever.min_score_threshold": (0.30, float, 0.0, 1.0, None,
        "最低注入分数阈值（迭代86→87：0.15→0.30，A/B评测T4残留干扰BM25 0.29仍通过旧阈值）"),
    "retriever.generic_query_min_threshold": (0.85, float, 0.0, 1.0, None,
        "通用知识 query 的注入阈值（迭代90：0.70→0.85，GIL题评分0.79仍通过0.70）"),

    # ── writer ──
    "writer.debounce_secs": (300, int, 0, 3600, None,
        "写入防抖窗口（秒）"),

    # ── extractor ──
    "extractor.chunk_quota": (200, int, 10, 10000, "MEMORY_OS_CHUNK_QUOTA",
        "每项目 chunk 配额上限"),
    "extractor.min_length": (10, int, 3, 50, None,
        "提取 chunk 的最小字符长度"),
    "extractor.max_summary": (120, int, 50, 500, None,
        "提取 chunk 摘要的最大字符长度"),
    "extractor.max_input_chars": (12000, int, 1000, 50000, None,
        "extractor 处理的最大输入字符数"),

    # ── loader ──
    "loader.max_age_secs": (86400, int, 3600, 604800, None,
        "latest.json 有效期（秒），超过则不注入"),
    "loader.max_context_chars": (800, int, 200, 5000, None,
        "SessionStart 注入的最大字符数"),
    "loader.working_set_top_k": (5, int, 1, 20, None,
        "工作集恢复的 Top-K 条数"),
    "loader.restore_working_set": (True, bool, None, None, None,
        "iter378: 是否在 SessionStart 时恢复持久化工作集（.ws_{project}.json）"),
    "loader.ws_max_restore": (20, int, 5, 100, None,
        "iter378: 从 .ws_{project}.json 最多恢复多少个 chunk（按 access_count 排序）"),

    # ── knowledge_router ──
    "router.top_k_per_source": (3, int, 1, 20, None,
        "每个知识源的 Top-K 条数"),
    "router.min_score": (0.01, float, 0.0, 1.0, None,
        "最低 BM25 分数阈值"),
    "router.cache_ttl_secs": (300, int, 0, 3600, None,
        "进程内缓存 TTL（秒）"),
    "router.scatter_shortcircuit_score": (0.75, float, 0.0, 1.0, None,
        "Scatter-Gather 短路触发分数阈值（高质量结果 score >= 此值时短路）"),
    "router.scatter_shortcircuit_count": (3, int, 1, 10, None,
        "Scatter-Gather 短路触发最少结果数（>= N 条高质量结果时短路）"),
    "working_set.max_chunks": (200, int, 50, 2000, None,
        "Per-Agent Working Set 最大 chunk 数（超出时 LRU 驱逐）"),
    "working_set.flush_dirty_on_exit": (True, bool, None, None, None,
        "Session 结束时是否 flush dirty chunks 回 store.db"),
    "prefetch.enabled": (True, bool, None, None, None,
        "是否启用 PreTool 预取引擎"),
    "prefetch.max_chunks": (10, int, 1, 50, None,
        "每次 PreTool 预取的最大 chunk 数"),
    "prefetch.timeout_ms": (80, int, 10, 500, None,
        "预取操作超时毫秒数（不阻塞主路径）"),

    # ── scheduler（迭代28）──
    "scheduler.skip_max_chars": (8, int, 3, 30, None,
        "query 短于此长度且无技术信号时 SKIP（nice 19）"),
    "scheduler.lite_max_chars": (200, int, 50, 1000, None,
        "query 短于此长度且无缺页信号时 LITE（nice 0，跳过 router）"),
    "scheduler.min_entity_count_for_full": (2, int, 1, 10, None,
        "query 含 >= N 个技术实体时强制 FULL（nice -20）"),

    # ── Query Truncation（迭代62）──
    "retriever.max_query_chars": (300, int, 50, 2000, None,
        "检索 query 最大字符数。超长 prompt 截断以防 FTS5 性能退化（1600字→300ms+）"),

    # ── scorer ──
    "scorer.importance_decay_rate": (0.95, float, 0.5, 1.0, None,
        "importance 遗忘曲线衰减率（每7天，全局默认）"),
    "scorer.importance_floor": (0.3, float, 0.0, 0.9, None,
        "importance 衰减下限"),
    # ── iter375: Type-Differential Decay Rates ──
    # 人类记忆情节/语义双系统（Tulving 1972）:情节记忆衰减快，语义记忆衰减慢
    # OS 类比：Linux MGLRU — younger generation pages age faster
    # decay_rate 越大 = 衰减越慢（0.99 ≈ 很慢，0.85 ≈ 较快）
    "scorer.decay_rate_task_state":           (0.88, float, 0.5, 1.0, None,
        "task_state 专属衰减率（情节记忆，衰减快）"),
    "scorer.decay_rate_conversation_summary": (0.90, float, 0.5, 1.0, None,
        "conversation_summary 专属衰减率（情节记忆）"),
    "scorer.decay_rate_decision":             (0.97, float, 0.5, 1.0, None,
        "decision 专属衰减率（语义记忆，衰减慢）"),
    "scorer.decay_rate_design_constraint":    (0.99, float, 0.5, 1.0, None,
        "design_constraint 专属衰减率（几乎不衰减）"),
    "scorer.decay_rate_reasoning_chain":      (0.95, float, 0.5, 1.0, None,
        "reasoning_chain 专属衰减率（语义记忆）"),
    "scorer.decay_rate_quantitative_evidence":(0.96, float, 0.5, 1.0, None,
        "quantitative_evidence 专属衰减率"),
    "scorer.decay_rate_causal_chain":         (0.95, float, 0.5, 1.0, None,
        "causal_chain 专属衰减率"),
    "scorer.decay_rate_excluded_path":        (0.93, float, 0.5, 1.0, None,
        "excluded_path 专属衰减率（中速衰减）"),
    "scorer.decay_rate_procedure":            (0.96, float, 0.5, 1.0, None,
        "procedure 专属衰减率（程序记忆，慢速衰减）"),
    "scorer.access_bonus_cap": (0.2, float, 0.0, 1.0, None,
        "access_bonus 上限"),
    "scorer.freshness_bonus_max": (0.15, float, 0.0, 0.5, None,
        "新 chunk 的初始曝光加分上限（Second Chance）"),
    "scorer.freshness_grace_days": (7, int, 1, 30, None,
        "freshness_bonus 的 grace period 天数，超过后 bonus=0"),

    # ── iter433: Reminiscence Bump Effect — 项目形成期记忆强化（Conway & Howe 1990）──────────────
    # 认知科学依据：Conway & Howe (1990); Rubin et al. (1998) "A model of the autobiographical memory" —
    #   人类自传体记忆中，15-25 岁（"形成期"）的事件比其他阶段记忆得更清晰（+50%~+100% recall rate），
    #   即使间隔 60 年也保持优势（不受普通遗忘曲线约束）。
    #   机制：形成期事件被编码进"核心自我叙事"（core self-narrative），
    #     与身份认同绑定，获得额外的记忆巩固路径（hippocampal + cortical dual encoding）。
    # 应用：chunk 在项目生命周期中的相对创建位置 position_pct <= bump_pct（默认 15%）
    #   且 importance >= bump_min_importance → initial_stability × bump_factor（+30%）。
    #   与 Primacy Effect（iter410）的区别：
    #     Primacy：编码顺序的绝对位置效应（最早的 N 条）
    #     Reminiscence Bump：项目生命周期的相对时间窗口效应（前 bump_pct% 的时间段内写入的 chunk）
    # OS 类比：Linux early_boot firmware parameters / BIOS/UEFI cmdline —
    #   早期引导阶段设置的核心参数（kernel cmdline、ACPI 表）在整个运行期保持不变，
    #   比运行时 sysctl 更稳定（boot-immutable vs runtime-mutable）。
    #   memory-os 中：项目创生期写入的 chunk = 启动参数，形成项目的"认知框架"。
    "store_vfs.bump_enabled": (True, bool, None, None, None,
        "iter433: 是否启用 Reminiscence Bump Effect：项目形成期 chunk 获得 stability 加成"),
    "store_vfs.bump_pct": (0.15, float, 0.02, 0.50, None,
        "iter433: 项目形成期时间窗口（占项目总年龄的比例，默认前 15%）"),
    "store_vfs.bump_min_importance": (0.55, float, 0.0, 1.0, None,
        "iter433: 应用 Reminiscence Bump 的最低 importance 阈值（低重要性早期 chunk 不受保护）"),
    "store_vfs.bump_factor": (1.30, float, 1.0, 2.0, None,
        "iter433: 形成期 chunk stability 加成系数（initial_stability × factor，默认 1.30）"),
    "store_vfs.bump_min_project_age_days": (7.0, float, 1.0, 90.0, None,
        "iter433: 应用 Reminiscence Bump 的最短项目年龄（天），项目太新时禁用（避免误判）"),

    # ── iter434: Retrieval-Induced Forgetting (RIF) — 检索导致相关记忆被压制（Anderson et al. 1994）──
    # 认知科学依据：Anderson, Bjork & Bjork (1994) "Remembering can cause forgetting" —
    #   检索某条记忆（practiced item）会主动抑制同类别中相关但未被检索的记忆（unpracticed items）。
    #   机制：检索激活该类别所有竞争记忆 → 强化被选中者（RP+）→ 主动抑制被压制者（RP-）→ RP- 遗忘增加。
    #   效果：测验后被练习项目增强记忆，相关未练习项目遗忘更多（比基线低 ~10-20%）。
    #   条件：RIF 要求被抑制者与被检索者属于同一类别（category-based competition）。
    #
    # memory-os 等价：
    #   检索命中 chunk_A → 对同类型（chunk_type）且内容相似（Jaccard >= threshold）的未命中 chunk_B
    #   施加轻微 stability 惩罚（× rif_factor < 1.0）。
    #   体现竞争抑制：被频繁检索的 chunk 越来越强，其竞争者越来越弱 → 系统自然专注核心知识。
    #
    # OS 类比：CPU cache set-associativity conflict eviction —
    #   访问 cache line A（命中 set 0, way 0）→ 通过 LRU 策略将同 set 的竞争 cache line B 推向更高 way
    #   → 再次访问 B 的概率降低（等价于 RIF：A 的命中加速了 B 的驱逐路径）。
    "scorer.rif_enabled": (True, bool, None, None, None,
        "iter434: 是否启用 Retrieval-Induced Forgetting — 检索 chunk 时轻微抑制同类相关但未命中的 chunk"),
    "scorer.rif_factor": (0.95, float, 0.80, 1.0, None,
        "iter434: RIF 抑制系数（被抑制 chunk stability × rif_factor，默认 0.95 = 5% 下降）"),
    "scorer.rif_similarity_threshold": (0.25, float, 0.0, 1.0, None,
        "iter434: RIF 触发的最低 Jaccard 相似度（0.25 = 至少 25% 词汇重叠才视为竞争者）"),
    "scorer.rif_max_targets": (3, int, 1, 20, None,
        "iter434: 每个命中 chunk 最多抑制的竞争者数量（按 Jaccard 降序，取前 N 个）"),
    "scorer.rif_protect_importance": (0.85, float, 0.0, 1.0, None,
        "iter434: importance >= 此值的 chunk 豁免 RIF 抑制（核心知识不被竞争压制）"),
    "scorer.rif_protect_types": ("design_constraint,procedure", str, None, None, None,
        "iter434: 豁免 RIF 的 chunk_type 列表（逗号分隔；保护类型即使相似也不被抑制）"),

    # ── iter432: Cumulative Interference Effect — 累积干扰加速遗忘（Underwood 1957）──
    # 认知科学依据：Underwood (1957) "Interference and forgetting" —
    #   遗忘的主要原因是同类型知识的累积干扰（proactive interference from prior lists），
    #   而非单纯时间流逝（decay theory）。同类型知识越多，每个 chunk 的遗忘越快。
    #   Jenkins & Dallenbach (1924): 睡眠期间比清醒期间遗忘更少，因为睡眠减少了新干扰。
    #   Underwood 1957 回归分析：24小时遗忘量与已学干扰列表数的相关 r=0.92（极强）。
    # 应用：同项目同 chunk_type 数量越多（N_same_type）→ 单个 chunk 的 stability 衰减更快。
    #   cumulative_interference_factor = 1 + scale × log(1+N) / log(1+N_median)
    #   在 episodic_decay_scan 中额外乘以 1/factor（>1 = 加速衰减）。
    # OS 类比：Linux CPU cache set-associativity conflict —
    #   同一 cache set 中的 cache line 越多，每条 line 的平均留存时间越短
    #   （more ways used = higher conflict miss rate = faster eviction）。
    "scorer.cumulative_interference_enabled": (True, bool, None, None, None,
        "iter432: 是否启用累积干扰效应——同类型 chunk 越多，每个 chunk 的 stability 衰减越快"),
    "scorer.ci_scale": (0.30, float, 0.0, 1.0, None,
        "iter432: 累积干扰强度系数：factor = 1 + scale × log(1+N)/log(1+N_med)，越大干扰越强"),
    "scorer.ci_max_factor": (2.0, float, 1.0, 5.0, None,
        "iter432: 累积干扰因子上限（最多让 stability 衰减 1/max_factor 倍，防止过激）"),
    "scorer.ci_protect_types": ("design_constraint,procedure", str, None, None, None,
        "iter432: 豁免累积干扰的 chunk_type 列表（逗号分隔；核心知识不受数量压制）"),
    "scorer.ci_min_n_same_type": (5, int, 1, 50, None,
        "iter432: 触发干扰效应的最低同类型 chunk 数量（N_same_type < 此值时不应用干扰）"),

    # ── iter431: Ribot's Law — 远期记忆稳定性梯度（Ribot 1882）──────────────────
    # 认知科学依据：Théodule Ribot (1882) "Diseases of Memory" —
    #   越早形成的记忆越能抵抗损伤（retrograde amnesia gradient）。
    #   脑损伤患者失去近期记忆，但保留远期（远古）的记忆——因为远期记忆已被"新皮层化"
    #   （hippocampal → neocortical transfer，系统巩固理论）。
    # 应用：chunk 年龄（age_days）越大 + importance >= ribot_min_importance →
    #   Ebbinghaus 遗忘曲线的 stability_floor 随年龄对数增长。
    #   floor_bonus = min(ribot_max_bonus, log(1+age_days)/log(365) × ribot_scale)
    #   年龄 1 年（365天）→ floor_bonus 达到 ribot_max_bonus。
    # OS 类比：Linux ext4 journal aging —
    #   长时间存在的 inode（ancient inodes）在 extent tree 中有更稳定的布局，
    #   碎片整理操作会优先保留而非移动 ancient extents（structural stability gradient）。
    "scorer.ribot_enabled": (True, bool, None, None, None,
        "iter431: 是否启用 Ribot's Law — 年龄越大的 chunk stability_floor 越高"),
    "scorer.ribot_min_importance": (0.60, float, 0.0, 1.0, None,
        "iter431: 应用 Ribot's Law 的最低 importance 阈值（低重要性老 chunk 不受保护）"),
    "scorer.ribot_scale": (0.20, float, 0.0, 1.0, None,
        "iter431: Ribot 稳定性梯度系数：floor_bonus = log(1+age_days)/log(365) × scale，"
        "age=365d 时 bonus=0.20，age=30d 时 bonus≈0.09"),
    "scorer.ribot_max_bonus": (0.25, float, 0.0, 0.5, None,
        "iter431: Ribot floor_bonus 上限（防止超长历史 chunk 的 floor 过高）"),
    "scorer.ribot_min_age_days": (30, int, 7, 365, None,
        "iter431: 开始应用 Ribot's Law 的最小年龄（天），默认 30 天"),

    # ── dmesg（迭代29）──
    "dmesg.ring_buffer_size": (500, int, 10, 5000, None,
        "dmesg 环形缓冲区最大条目数"),

    # ── kswapd watermarks（迭代30）──
    "kswapd.pages_low_pct": (80, int, 50, 95, None,
        "low watermark 百分比：低于此水位触发后台预淘汰"),
    "kswapd.pages_high_pct": (90, int, 60, 99, None,
        "high watermark 百分比：高于此水位停止淘汰（安全区）"),
    "kswapd.pages_min_pct": (95, int, 80, 100, None,
        "min watermark 百分比：高于此水位触发同步硬淘汰（OOM）"),
    "kswapd.stale_days": (90, int, 14, 365, None,
        "超过此天数未访问的 chunk 标记为可回收（stale page）"),
    "kswapd.batch_size": (5, int, 1, 50, None,
        "每次 kswapd 扫描最多淘汰的 chunk 数"),

    # ── compaction（迭代31）──
    "compaction.min_cluster_size": (3, int, 2, 20, None,
        "触发 compaction 的最小聚类大小（同主题 chunk 数）"),
    "compaction.max_merge_per_run": (10, int, 1, 50, None,
        "每次 compaction 最多合并的聚类数"),
    "compaction.entity_overlap_min": (2, int, 1, 10, None,
        "聚类所需的最小共享实体数"),

    # ── madvise（迭代32）──
    "madvise.boost_factor": (0.15, float, 0.0, 0.5, None,
        "hint 匹配的 chunk 召回加分（叠加在 retrieval_score 上）"),
    "madvise.max_hints": (10, int, 3, 30, None,
        "每个项目最多保留的 hint 关键词数"),
    "madvise.ttl_secs": (1800, int, 300, 7200, None,
        "hint 有效期（秒），超过则忽略"),

    # ── swap（迭代33）──
    "swap.max_chunks": (100, int, 10, 1000, None,
        "swap 分区最大 chunk 数（超出时删除最旧 swap 条目）"),
    "swap.min_importance_for_swap": (0.5, float, 0.0, 1.0, None,
        "低于此 importance 的 chunk 直接删除而非 swap out"),
    "swap.fault_top_k": (2, int, 1, 10, None,
        "swap fault 时最多 swap in 的 chunk 数"),

    # ── iter430: Spontaneous Recovery — 自发恢复（Pavlov 1927）──────────────────
    # 认知科学依据：Pavlov (1927) — 条件反射被抑制后经过休息可自发恢复（不需额外强化）。
    #   Rescorla (1997): 恢复程度与休息时间正相关。
    #   应用：被 kswapd 驱逐到 swap 的高历史访问 chunk 经过一段时间后可自发恢复。
    # OS 类比：Linux zswap 解压缩 + MGLRU active 列表晋升 —
    #   swap 分区中的页面在满足热度条件时被自动提升回 active 列表。
    "swap.sr_enabled": (True, bool, None, None, None,
        "iter430: 是否启用 Spontaneous Recovery — swap 中高历史价值 chunk 的自发恢复"),
    "swap.sr_min_swap_days": (3.0, float, 0.5, 90.0, None,
        "iter430: 在 swap 中至少 N 天才触发自发恢复（防止抖动）"),
    "swap.sr_min_access_count": (3, int, 1, 50, None,
        "iter430: 历史访问次数阈值：>= N 次才视为'曾经重要'的 chunk"),
    "swap.sr_min_importance": (0.65, float, 0.3, 1.0, None,
        "iter430: importance 阈值：>= 此值的 chunk 才参与自发恢复"),
    "swap.sr_recovery_boost": (1.15, float, 1.0, 2.0, None,
        "iter430: stability 恢复系数 — swap in 时 stability × boost（默认 1.15 ≈ 15% 提升）"),
    "swap.sr_max_recover_per_run": (5, int, 1, 50, None,
        "iter430: 每次 SessionStart 最多恢复的 chunk 数量（限制 swap in I/O）"),

    # ── OOM Score（迭代38）──
    "oom.auto_protect_quant": (-500, int, -1000, 0, None,
        "量化证据 chunk 的自动 oom_adj（负值=保护）"),
    "oom.auto_disposable_ctx": (500, int, 0, 1000, None,
        "prompt_context chunk 的自动 oom_adj（正值=优先淘汰）"),

    # ── cgroup v2 memory.high（迭代40）──
    "cgroup.memory_high_pct": (85, int, 50, 95, None,
        "软配额水位百分比：超过时 throttle 新写入（降 importance + 加 oom_adj）"),
    "cgroup.throttle_factor": (0.7, float, 0.3, 1.0, None,
        "throttle 区间内新写入 importance 的衰减因子（乘法）"),
    "cgroup.throttle_oom_adj": (300, int, 0, 1000, None,
        "throttle 区间内新写入 chunk 的自动 oom_adj（正值=加速回收）"),

    # ── COW 预扫描（迭代39）──
    "extractor.cow_prescan_chars": (3000, int, 500, 10000, None,
        "COW 预扫描采样字符数（只扫描消息前 N 个字符）"),

    # ── iter392：Generation Effect — 主动生成增强 ──
    # 认知科学：Slamecka & Graf (1978) Generation Effect —
    #   自己生成的内容（vs 被动阅读）记忆留存率显著更高（+50%~+80%）。
    #   主动生成触发更深度认知加工（elaborative encoding），形成更强记忆痕迹。
    # 应用：reasoning_chain / decision 是 agent 主动生成的推理产物，
    #   写入时给予 stability 额外乘子，使其在 Ebbinghaus 曲线下衰减更慢。
    "extractor.generation_boost_enabled": (True, bool, None, None, None,
        "是否对 agent 主动生成类 chunk（reasoning_chain/decision）应用 generation effect stability 加成（iter392）"),
    "extractor.generation_boost_factor": (1.2, float, 1.0, 2.0, None,
        "生成效应稳定性加成系数：reasoning_chain/decision 的 stability 初始值乘以此系数（iter392，默认 1.2）"),
    "extractor.generation_boost_types": ("reasoning_chain,decision,causal_chain", str, None, None, None,
        "应用 generation_boost 的 chunk_type 集合（逗号分隔，iter392）"),

    # ── iter406: Generation Effect — Lexical Marker Detection ──────────────────
    "store_vfs.generation_effect_enabled": (True, bool, None, None, None,
        "是否启用 iter406 Generation Effect：检测内容中的推理/假设/元认知标记，对主动生成内容提升 stability"),
    "store_vfs.generation_effect_source_direct_bypass": (True, bool, None, None, None,
        "source_type='direct' 时跳过生成效应检测（人直接输入=被动录入，非 agent 生成），默认 True"),

    # ── iter407: Von Restorff Effect — Isolation Stability Bonus ─────────────────
    "store_vfs.isolation_effect_enabled": (True, bool, None, None, None,
        "是否启用 iter407 Von Restorff Effect：孤立 chunk（encode_context 语义独特）得到 stability 加成"),
    "store_vfs.isolation_context_window": (20, int, 5, 100, None,
        "iter407: 计算语义孤立度时对比的最近邻居数量（基于 created_at 排序）"),
    "store_vfs.isolation_min_neighbors": (3, int, 1, 20, None,
        "iter407: 邻居少于此数时返回 0.0 孤立度（避免项目初期误判所有 chunk 为孤立）"),

    # ── iter408: Proactive Interference — 旧知识干扰新知识写入 ─────────────────────
    "store_vfs.pi_enabled": (True, bool, None, None, None,
        "是否启用 iter408 Proactive Interference：旧强记忆干扰新 chunk 写入时的 initial stability"),
    "store_vfs.pi_search_k": (5, int, 1, 20, None,
        "iter408: 计算 PI 时检索的语义邻居数量"),
    "store_vfs.pi_strong_acc_threshold": (3, int, 1, 50, None,
        "iter408: 视为'强旧记忆'的 access_count 阈值（≥ 此值才产生 PI 压力）"),
    "store_vfs.pi_max_penalty": (0.10, float, 0.0, 0.30, None,
        "iter408: PI 最大惩罚（从 base_stability 中减去的上限，默认 0.10）"),

    # ── iter409: Flashbulb Memory — 情绪性内容写入时 stability 加强 ────────────────
    "store_vfs.flashbulb_enabled": (True, bool, None, None, None,
        "是否启用 iter409 Flashbulb Memory：emotional_weight 高的 chunk 写入时 stability 加强"),
    "store_vfs.flashbulb_strong_threshold": (0.70, float, 0.3, 1.0, None,
        "iter409: 强情绪唤醒阈值（≥ 此值获得最大加成 base×0.30）"),
    "store_vfs.flashbulb_medium_threshold": (0.50, float, 0.1, 0.9, None,
        "iter409: 中等情绪唤醒阈值（[medium, strong) 区间线性插值加成）"),

    # ── iter410: Primacy Effect — 首位效应（Murdock 1962 Serial Position Effect）──
    "store_vfs.primacy_enabled": (True, bool, None, None, None,
        "是否启用 iter410 Primacy Effect：项目最早创建的 chunk 获得 stability 首位加成"),
    "store_vfs.primacy_min_total": (20, int, 5, 200, None,
        "iter410: 项目 chunk 总数低于此值时不应用首位效应（避免新项目误判）"),
    "store_vfs.primacy_core_pct": (0.10, float, 0.02, 0.30, None,
        "iter410: 最早 N% 的 chunk 获得完整首位加成（默认最早 10%）"),

    # ── iter411: Levels of Processing — 编码深度（Craik & Lockhart 1972）─────────
    "store_vfs.lop_enabled": (True, bool, None, None, None,
        "是否启用 iter411 Levels of Processing：encode_context 实体密度代理编码深度 → stability 加成"),

    # ── iter414: Self-Reference Effect — 自我参照内容的记忆优势 ─────────────────────
    # 认知科学依据：Rogers et al. (1977) Self-Reference Effect —
    #   以"与自我相关"方式加工的信息比语义加工的记忆更强（self-referential processing 激活 PFC + hippocampus）。
    #   Symons & Johnson (1997) Meta-analysis: self-reference advantage ≈ +0.5 SD vs semantic encoding。
    #   在 memory-os 中：chunk 内容含第一人称标记（I/we/our/my）或 agent 自身推理产物，
    #   代理"自我参照"加工，initial stability 获得加成。
    # OS 类比：Linux process 自身页（stack/heap/text）在 TLB 中有最高局部性 —
    #   process 直接引用的 page（自我参照）命中率最高，类比 self-referential chunk 的检索优势。
    "store_vfs.self_ref_enabled": (True, bool, None, None, None,
        "是否启用 iter414 Self-Reference Effect：含第一人称标记的 chunk 获得 stability 加成"),
    "store_vfs.self_ref_bonus_cap": (0.25, float, 0.0, 0.50, None,
        "iter414: Self-Reference Effect stability 加成上限（作为 base × 此系数，默认 0.25）"),

    # ── iter415: Encoding Variability — 多情境编码的记忆鲁棒性 ────────────────────
    # 认知科学依据：Estes (1955) Encoding Variability Theory; Bjork & Bjork (1992) New Theory of Disuse —
    #   同一记忆在多个不同情境下编码 → 更多检索线索 → 在多样化情境下均可提取（retrieval robustness）。
    #   Glenberg (1979): 分布式练习效果部分来自情境多样性（context diversification across repetitions）。
    # 实现：encode_context token 数量随 iter404 语义启动而增长；token 数超过初始值越多，
    #   代表访问情境越多样，在 update_accessed 时给予轻微 stability 加成。
    # OS 类比：Linux 共享库被 N 个进程引用 → page cache 引用计数高 → 驱逐优先级低（多情境引用 = 更稳定）。
    "store_vfs.encoding_variability_enabled": (True, bool, None, None, None,
        "是否启用 iter415 Encoding Variability：encode_context 增长（多情境访问）→ stability 加成"),
    "store_vfs.encoding_variability_scale": (0.05, float, 0.0, 0.20, None,
        "iter415: 每个新增 encode_context token 的 stability 加成系数（默认 0.05，上限 base × 0.15）"),

    # ── iter416: Zeigarnik Effect — 未完成任务的记忆优势 ──────────────────────────────
    # 认知科学依据：Zeigarnik (1927) — 未完成任务比已完成任务被记忆得更好（+90% recall superiority）。
    #   Lewin (1935) Tension System Theory — 未完成任务在认知系统中维持"心理张力"，
    #   保持记忆激活直到任务完成（类比未释放的 futex 锁）。
    #   Ovsiankina (1928) — 被中断的任务在有机会时自发恢复（resumption tendency）。
    # 应用：chunk 内容含 TODO/FIXME/pending/unresolved 信号词 → 代表"未完成"认知任务，
    #   给予 stability 加成，防止被 kswapd 过早驱逐（这些信息最需要在下次会话中恢复）。
    # OS 类比：Linux futex waitqueue / O_SYNC dirty page —
    #   待处理的 I/O 请求保留在内核等待队列，不被 swapd 驱逐；
    #   未完成写入的 dirty page 被 writeback 守护进程跟踪，优先处理。
    "store_vfs.zeigarnik_enabled": (True, bool, None, None, None,
        "是否启用 iter416 Zeigarnik Effect：含未完成任务信号词的 chunk 获得 stability 加成"),
    "store_vfs.zeigarnik_bonus_cap": (0.20, float, 0.0, 0.50, None,
        "iter416: Zeigarnik Effect stability 加成上限（作为 base × 此系数，默认 0.20）"),

    # ── iter417: Retrieval-Induced Forgetting — 检索引发的竞争性抑制 ─────────────────
    # 认知科学依据：Anderson, Bjork & Bjork (1994) "Remembering can cause forgetting" —
    #   检索一个记忆时，与之竞争的语义邻居记忆受到主动抑制（inhibitory tagging）。
    #   抑制强度与语义相似度正相关（高相似 = 强竞争 = 更多抑制）。
    #   MacLeod et al. (2003): RIF 是真实的记忆抑制（non-retrieval 控制组无此效应）。
    # 应用：update_accessed 时，对语义邻居（高 encode_context token 重叠但未被检索）
    #   应用轻微 stability 衰减，模拟竞争性抑制，促进检索多样性。
    # OS 类比：MESI 缓存一致性协议 —
    #   一个核写入 cache line（Modified状态）→ 其他核的相同 cache line 被 Invalidated；
    #   一个 chunk 被"激活"→ 其语义竞争者的局部性降低（类比 cache invalidation）。
    "store_vfs.rif_enabled": (True, bool, None, None, None,
        "是否启用 iter417 Retrieval-Induced Forgetting：检索时对语义竞争者施加轻微 stability 衰减"),
    "store_vfs.rif_decay_factor": (0.99, float, 0.90, 1.00, None,
        "iter417: RIF stability 衰减因子（neighbor stability × 此值），默认 0.99（轻微 1% 衰减）"),
    "store_vfs.rif_min_overlap": (2, int, 1, 10, None,
        "iter417: 触发 RIF 所需的最小 encode_context token 重叠数（2 token 重叠 = 语义竞争者）"),
    "store_vfs.rif_max_neighbors": (5, int, 1, 20, None,
        "iter417: 每次检索最多影响的语义邻居数量（按 overlap 降序取前 N）"),

    # ── iter418: Directed Forgetting — 主动弃置过时知识 ──────────────────────────────
    # 认知科学依据：MacLeod (1998) Directed Forgetting —
    #   主动指令"忘记"某信息时，记忆对该信息的保留显著下降（inhibition account）。
    #   Johnson (1994): 认知系统主动抑制不再有用的记忆，释放认知资源。
    # 应用：chunk 内容含 deprecated/obsolete/replaced by 等信号词 → 主动减少 stability，
    #   加速 kswapd 自然淘汰（不强制删除，而是降低其竞争力）。
    # OS 类比：Linux madvise(MADV_DONTNEED) —
    #   显式通知内核该内存区域不再需要，内核加速回收（但不立即释放，等 kswapd 处理）。
    "store_vfs.df_enabled": (True, bool, None, None, None,
        "是否启用 iter418 Directed Forgetting：含过时信号词的 chunk 获得 stability 惩罚"),
    "store_vfs.df_penalty_cap": (0.15, float, 0.0, 0.50, None,
        "iter418: Directed Forgetting stability 惩罚上限（从 base 减去 base × 此系数，默认 0.15）"),

    # ── iter422: Permastore Memory — 充分强化后的记忆永久保护（Bahrick 1979）──────────────
    # 认知科学依据：Bahrick (1979) — 充分访问+高重要性的记忆达到"permastore"状态，
    #   即使经过数十年不复习，仍能保留约 80% 的可访问性。
    #   Conway et al. (1991): 专业知识具有 permastore 特征。
    # 应用：chunk 满足 age>=30d + access_count>=10 + importance>=0.80 →
    #   RI/RIF/DF 只能降低到 stability×floor_factor(0.80)（而非普通 floor=0.1）。
    # OS 类比：Linux mlock() — 重要页面锁定在 RAM，kswapd 无法驱逐。
    "store_vfs.permastore_enabled": (True, bool, None, None, None,
        "是否启用 iter422 Permastore Memory：充分访问+高重要性 chunk 的 stability 受更高 floor 保护"),
    "store_vfs.permastore_min_age_days": (30, int, 7, 365, None,
        "iter422: 进入 permastore 所需的最小 chunk 年龄（天），默认 30 天"),
    "store_vfs.permastore_min_access_count": (10, int, 3, 100, None,
        "iter422: 进入 permastore 所需的最小访问次数，默认 10 次"),
    "store_vfs.permastore_min_importance": (0.80, float, 0.3, 1.0, None,
        "iter422: 进入 permastore 所需的最低 importance，默认 0.80"),
    "store_vfs.permastore_floor_factor": (0.80, float, 0.3, 1.0, None,
        "iter422: permastore chunk 的 stability 下限系数（stability × factor），默认 0.80"),

    # ── iter421: Retroactive Interference — 新学习干扰旧记忆回忆 ─────────────────────────
    # 认知科学依据：McGeoch (1932) Interference Theory; Barnes & Underwood (1959) —
    #   新学习的信息（新 chunk）干扰对旧相关信息的回忆（retroactive interference）。
    #   RI 与 PI（iter408）互补：PI=旧→新，RI=新→旧。
    #   Anderson & Green (2001): 主动抑制相似记忆是 RI 的神经机制。
    # 应用：insert_chunk 时，对同项目中 encode_context 高度重叠的低 importance 旧 chunk
    #   施加轻微 stability 衰减（× ri_decay_factor=0.98），模拟新记忆干扰旧记忆。
    #   高重要性（>= ri_protect_importance=0.85）的 chunk 免疫 RI。
    # OS 类比：TLB shootdown — 新 VA→PA 映射建立时，发送 IPI 使其他核的旧 TLB 条目失效。
    "store_vfs.ri_enabled": (True, bool, None, None, None,
        "是否启用 iter421 Retroactive Interference：新 chunk 写入时对语义邻居旧 chunk 施加轻微 stability 衰减"),
    "store_vfs.ri_min_overlap": (2, int, 1, 10, None,
        "iter421: 触发 RI 的最小 encode_context token 重叠数（默认 2）"),
    "store_vfs.ri_decay_factor": (0.98, float, 0.90, 1.00, None,
        "iter421: RI stability 衰减因子（旧 chunk stability × 此值，默认 0.98，轻微 2% 衰减）"),
    "store_vfs.ri_max_targets": (3, int, 1, 10, None,
        "iter421: 每次 insert_chunk 最多影响的旧 chunk 数量（按重叠度降序取前 N）"),
    "store_vfs.ri_protect_importance": (0.85, float, 0.5, 1.0, None,
        "iter421: importance >= 此值的 chunk 免疫 RI（高重要性核心知识受保护）"),

    # ── iter420: Spacing Effect — 分布式练习的记忆优势（间隔效应）────────────────────────
    # 认知科学依据：Ebbinghaus (1885) Spacing Effect; Cepeda et al. (2006) Review (300+ studies) —
    #   分布式练习（相同次数的学习，分散在多个时间间隔）比集中练习产生更强的长时记忆保留。
    #   Glenberg (1979): 情境多样性（context diversity across repetitions）是间隔效应的核心机制。
    #   间隔效应与 iter412 Testing Effect 相互增强（间隔越长 → 难度越高 → 双重加成）。
    # 应用：update_accessed 时，若访问间隔 >= medium_gap_hours(24h)，spaced_access_count+1。
    #   spacing_factor = spaced_access_count / max(1, access_count)；
    #   SM-2 quality += round(spacing_factor × spacing_quality_scale)（最大 +2）。
    # OS 类比：Linux MGLRU cross-generation promotion —
    #   跨 aging cycle 的 page 访问比同 gen 内多次访问更快晋升（distributed > massed）。
    "store_vfs.spacing_effect_enabled": (True, bool, None, None, None,
        "是否启用 iter420 Spacing Effect：访问间隔 >= 24h 时递增 spaced_access_count，影响 SM-2 质量"),
    "store_vfs.spacing_quality_scale": (2.0, float, 0.0, 4.0, None,
        "iter420: Spacing Effect SM-2 quality 加成系数：quality_bonus = round(spacing_factor × scale)，最大 +2"),

    # ── iter419: Associative Memory — 新知识借助强关联记忆的编码优势 ────────────────────
    # 认知科学依据：Ebbinghaus (1885) Paired Associates Learning;
    #   Collins & Loftus (1975) Spreading Activation — 新知识与已有强记忆共享节点时
    #   形成更强的记忆痕迹（associative encoding advantage, 类比"锚点记忆"）。
    #   Anderson & Reder (1999): 高连接度节点的新关联比孤立节点更易编码（fan effect 逆向）。
    # 应用：写入新 chunk 时，如果其 encode_context 与已有高 importance 的 chunk 重叠 ≥ 阈值，
    #   给予 stability 加成（借助已有强记忆结构"搭架"）。
    # OS 类比：Linux huge pages (THP) — small page adjacent to huge page shares same TLB entry
    #   and benefits from the huge page's TLB locality (associative memory locality)。
    "store_vfs.am_enabled": (True, bool, None, None, None,
        "是否启用 iter419 Associative Memory：新 chunk 与高 importance 旧 chunk 共享实体 → stability 加成"),
    "store_vfs.am_min_overlap": (2, int, 1, 10, None,
        "iter419: 触发关联记忆加成的最小 encode_context token 重叠数"),
    "store_vfs.am_min_importance": (0.75, float, 0.3, 1.0, None,
        "iter419: 触发关联记忆加成的锚点 chunk 的最低 importance 阈值"),
    "store_vfs.am_bonus_cap": (0.15, float, 0.0, 0.40, None,
        "iter419: 关联记忆加成上限（base × 此系数，默认 0.15）"),

    # ── Deadline I/O Scheduler（迭代41）──
    "retriever.deadline_ms": (50.0, float, 5.0, 200.0, None,
        "检索截止时间（ms），超过时跳过低优先级阶段（从30ms调整为50ms，适应VFS+PSI开销）"),
    "retriever.deadline_hard_ms": (200.0, float, 20.0, 500.0, None,
        "硬截止时间（ms），超过时立即返回已有结果（从80ms调整为200ms，避免WAL争用下的空结果）"),

    # ── ASLR 检索多样性（迭代43）──
    "scorer.aslr_epsilon": (0.08, float, 0.0, 0.3, None,
        "ASLR 随机扰动幅度上限（乘以 1-access_ratio 后叠加到 retrieval_score）"),
    "scorer.aslr_access_threshold": (5, int, 1, 50, None,
        "ASLR 生效阈值：access_count 低于此值的 chunk 才获得随机扰动"),

    # ── Anti-Starvation（迭代62）──
    "scorer.saturation_factor": (0.04, float, 0.0, 0.15, None,
        "饱和惩罚系数：penalty = factor × log2(1 + recall_count)，越大惩罚越重"),
    "scorer.saturation_cap": (0.25, float, 0.05, 0.50, None,
        "饱和惩罚上限：防止热门知识被完全压制"),

    # ── 迭代333：TMV Multiplicative Saturation Discount ──
    # OS 类比：Linux NUMA distance penalty — access_count 极高的 chunk 类似 remote NUMA node，
    #   边际信息价值趋近于零（agent 已经"内化"），需要乘法折扣而非加法惩罚。
    "scorer.tmv_acc_threshold": (50, int, 10, 500, None,
        "TMV 饱和折扣起始阈值：access_count 超过此值开始应用乘法折扣"),
    "scorer.tmv_discount_weight": (0.30, float, 0.0, 0.60, None,
        "TMV 折扣强度：score × (1 - discount_weight × log(acc/threshold) / log(1000/threshold))，"
        "acc=2044/threshold=50 时 discount=0.30×(log(41)/log(20))≈0.39，最大降权 39%"),
    "scorer.tmv_discount_floor": (0.55, float, 0.30, 0.95, None,
        "TMV 折扣下限乘子：无论 acc 多高，score 不低于原始的此比例（防止 design_constraint 被过度压制）"),
    "scorer.tmv_session_density_gate": (4, int, 2, 10, None,
        "Session 密度门控：同一 chunk 在本 session 被注入 >= 此次数时，额外乘以 0.7（防止信息茧房）"),
    "scorer.starvation_boost_factor": (0.30, float, 0.05, 0.60, None,
        "饥饿加分系数：access_count=0 的老 chunk 最大加分值"),
    "scorer.starvation_min_age_days": (0.5, float, 0.0, 7.0, None,
        "饥饿加分最小年龄：低于此天数不加分（freshness_bonus 仍在生效）"),
    "scorer.starvation_ramp_days": (3.0, float, 0.5, 14.0, None,
        "饥饿加分线性增长区间：从 min_age 到 min_age+ramp_days 线性增长到满额"),

    # ── Memory Balloon（迭代46）──
    "balloon.global_pool": (1000, int, 100, 10000, None,
        "全局 chunk 总量池（所有项目共享），各项目配额从此池中动态分配"),
    "balloon.min_quota": (30, int, 10, 500, None,
        "每个项目的最低保障配额（即使不活跃也不低于此值）"),
    "balloon.max_quota": (500, int, 50, 5000, None,
        "单项目配额上限（即使活跃度最高也不超过此值）"),
    "balloon.activity_window_days": (14, int, 3, 90, None,
        "活跃度计算时间窗口（天），只统计此窗口内的写入/访问活动"),

    # ── MGLRU（迭代44）──
    "mglru.max_gen": (4, int, 2, 10, None,
        "MGLRU 最大代数（gen 0=youngest, max_gen=oldest，超过则不再递增）"),
    "mglru.aging_interval_hours": (6, int, 1, 168, None,
        "两次 aging 之间的最小间隔（小时），防止频繁 /clear 导致过度老化"),

    # ── DAMON（迭代42）──
    "damon.cold_age_days": (14, int, 3, 90, None,
        "chunk 创建超过此天数且 access_count=0 标记为 COLD"),
    "damon.dead_age_days": (30, int, 7, 180, None,
        "chunk 创建超过此天数且 access_count=0 且低 importance 标记为 DEAD"),
    "damon.dead_importance_max": (0.65, float, 0.3, 0.9, None,
        "DEAD 分类的 importance 上限（低于此值的零访问 chunk 被视为 DEAD）"),
    "damon.cold_oom_adj_delta": (200, int, 50, 500, None,
        "COLD chunk 的 oom_adj 增量（加速未来 kswapd 淘汰）"),
    "damon.max_actions_per_scan": (10, int, 1, 50, None,
        "每次 DAMON scan 最多执行的动作数（swap + mark + protect）"),

    # ── sched_ext（迭代47）──
    "scheduler.ext_enabled": (True, bool, None, None, None,
        "是否启用 sched_ext 自定义规则（False 时只使用内置分类器）"),
    "scheduler.ext_max_rules": (20, int, 1, 100, None,
        "sched_ext 自定义规则的最大数量"),

    # ── readahead（迭代48）──
    "readahead.min_cooccurrence": (2, int, 1, 10, None,
        "共现计数阈值：两个 chunk 在 recall_traces 中至少共同出现 N 次才建立 readahead pair"),
    "readahead.prefetch_bonus": (0.10, float, 0.01, 0.50, None,
        "readahead prefetch 加分：命中 pair 的 chunk 获得此固定加分"),
    "readahead.max_prefetch": (2, int, 1, 5, None,
        "每次检索最多 prefetch 的额外 chunk 数量"),
    "readahead.window_traces": (50, int, 10, 200, None,
        "分析共现模式时回看最近 N 条 recall_traces"),

    # ── TCP AIMD — Adaptive Extraction Window（迭代50）──
    "aimd.window_traces": (30, int, 10, 200, None,
        "AIMD 计算窗口：回看最近 N 条 recall_traces 统计命中率"),
    "aimd.cwnd_max": (1.0, float, 0.3, 1.0, None,
        "AIMD cwnd 上限（1.0 = 全速提取，所有信号匹配都写入）"),
    "aimd.cwnd_min": (0.3, float, 0.1, 0.8, None,
        "AIMD cwnd 下限（保底提取能力，不会完全停止提取）"),
    "aimd.cwnd_init": (0.7, float, 0.2, 1.0, None,
        "AIMD cwnd 初始值（新项目/无历史数据时的默认窗口）"),
    "aimd.hit_rate_target": (0.3, float, 0.1, 0.8, None,
        "AIMD 目标命中率：高于此值 cwnd 线性增加，低于此值 cwnd 指数减少"),
    "aimd.additive_increase": (0.05, float, 0.01, 0.2, None,
        "AIMD 加法增大步长：命中率达标时 cwnd += AI 步长"),
    "aimd.multiplicative_decrease": (0.5, float, 0.2, 0.9, None,
        "AIMD 乘法减小因子：命中率不达标时 cwnd *= MD 因子"),
    "aimd.ssthresh": (0.6, float, 0.3, 1.0, None,
        "AIMD Slow Start 阈值：cwnd < ssthresh 时指数恢复（每次翻倍），>= 时线性恢复"),
    "aimd.slow_start_factor": (2.0, float, 1.2, 4.0, None,
        "AIMD Slow Start 指数增长因子：cwnd = cwnd * factor（直到 ssthresh）"),
    "aimd.small_pool_pct": (0.4, float, 0.1, 0.8, None,
        "Small Pool Bypass: chunk 数 < quota×此比例时跳过 AIMD（cwnd=max, policy=full）"),

    # ── Trace GC — recall_traces 生命周期管理（迭代63）──
    "gc.trace_max_age_days": (14, int, 3, 90, None,
        "recall_traces 最大保留天数，超过后 GC 清理"),
    "gc.trace_max_rows": (500, int, 50, 5000, None,
        "recall_traces 最大保留行数，超过后按时间淘汰"),

    # ── CRIU Checkpoint/Restore（迭代49）──
    "criu.max_checkpoints": (3, int, 1, 10, None,
        "每个项目保留的最大 checkpoint 数量（FIFO 淘汰最旧）"),
    "criu.max_age_hours": (72, int, 6, 720, None,
        "checkpoint 过期时间（小时），超过则不恢复"),
    "criu.max_hit_ids": (50, int, 3, 200, None,
        "checkpoint 保存的最近命中 chunk ID 数量（iter89: 10→50，支持大工作集）"),
    "criu.restore_boost": (0.12, float, 0.0, 0.5, None,
        "checkpoint 恢复时命中 chunk 的评分加权（叠加在 working_set_score 上）"),

    # ── Autotune（迭代51）──
    "autotune.enabled": (True, bool, None, None, None,
        "是否启用参数自动调优（SessionStart 时运行）"),
    "autotune.min_traces": (10, int, 3, 100, None,
        "触发 autotune 所需的最少 recall_traces 数量（样本不足时跳过）"),
    "autotune.step_pct": (10, int, 5, 30, None,
        "每次自动调整的最大幅度百分比（保守调参，避免振荡）"),
    "autotune.cooldown_hours": (6, int, 1, 168, None,
        "两次 autotune 之间的最小间隔（小时），防止频繁调参"),
    "autotune.hit_rate_low_pct": (20, int, 5, 50, None,
        "命中率低于此阈值时收缩 top_k / quota（减少噪声写入）"),
    "autotune.hit_rate_high_pct": (50, int, 30, 80, None,
        "命中率高于此阈值时 quota 适度扩大（迭代129：top_k 不再随高命中率增加，已修复逻辑反转）"),
    # ── Autotune top_k 上限（迭代129）──
    # OS 类比：TCP AIMD cwnd_max — 拥塞窗口有上限，防止 cwnd 无限增长
    # 根因：旧逻辑命中率高→扩大 top_k（方向错误），且无上限保护，
    #       导致 top_k 从默认 5 被推到 12，与 design_constraint 膨胀叠加造成 injected=14+
    # 修复后 top_k 只在命中率低时增加，且受此上限保护
    "autotune.top_k_max": (6, int, 3, 15, None,
        "autotune 允许调整的 retriever.top_k 上限（迭代129：防止高命中率反向推高 top_k）"),
    # ── Autotune deadline 上限（迭代136）──
    # OS 类比：TCP SYN_RETRIES max — 限制重试上限，防止指数退避无限膨胀
    # 根因：deadline_skip 轨迹的 duration_ms ≈ 当前 deadline_ms（自引用），
    #       p95 包含这些轨迹 → p95 > 2×baseline → autotune 推高 deadline_ms,
    #       → 新轨迹 duration 更高 → p95 再升 → 正反馈循环（每次 +10%）
    "autotune.deadline_max_ms": (100.0, float, 50.0, 300.0, None,
        "autotune 允许调整的 retriever.deadline_ms 上限（迭代136：防止 deadline_skip 自强化循环膨胀，default=100ms）"),
    # ── Autotune chunk_quota 上限（迭代137）──
    # OS 类比：Linux cgroup memory.max — 硬限制，cgroup 内存不能超过此值
    #   autotune 在高命中率（>50%）时每 6 小时 +10% quota，无上限则无限增长：
    #   gitroot 实测 200→389（约 19 次 +10%），balloon.max_quota=500 是全局上限
    #   但 autotune 不检查 balloon 上限，可一直增到 10000（extractor.chunk_quota 上限）
    # 修复：autotune.chunk_quota_max 作为 autotune 调参的软性上限
    #   不同于 balloon.max_quota（全局硬限），这是 autotune 不应超越的"合理范围"
    #   default=400：保留生产使用的合理增长空间，阻止无限推高
    "autotune.chunk_quota_max": (400, int, 50, 5000, None,
        "autotune 允许调整的 extractor.chunk_quota 上限（迭代137：防止高命中率无限推高 quota，default=400）"),
    # ── Autotune kswapd 水位回弹（迭代138）──
    # OS 类比：Linux vm.watermark_boost_factor — 内存压力消退后 watermark 自动恢复正常水位
    #   策略4 单方向降低 pages_low_pct（容量>90%时），但无回弹机制
    #   abspath:7e3095aef7a6 实测：80→72→64.8→58.3，capacity 恢复后 pages_low 永久偏低
    #   pages_low 过低 = kswapd 在 58% 就开始淘汰，不必要地频繁 eviction
    # 修复（迭代138）：capacity < 70% 时，pages_low 每次向默认值(80)回弹 step_pct%
    #   恢复路径（step_pct=10%）：58→63→69→76→80（4个 autotune 周期，24小时）
    # 无需额外 sysctl：复用已有 autotune.step_pct 控制回弹步长

    # ── DRR Fair Queuing（迭代50）──
    "retriever.drr_enabled": (True, bool, None, None, None,
        "是否启用 DRR 类型多样性保障（False 时退化为纯 score 排序）"),
    "retriever.drr_max_same_type": (2, int, 1, 10, None,
        "单一 chunk_type 在 Top-K 中的最大占比（绝对值，超出让位给其他类型）"),

    # ── Query-Conditioned Importance（迭代322）──
    # OS 类比：Linux CPUFreq P-state — 根据负载动态调整处理器频率
    # α_eff = qci_base_alpha - qci_relevance_slope × relevance
    #   relevance=1.0 → α_eff=0.30（recency 主导），relevance=0.0 → α_eff=0.55（importance 主导）
    "retriever.qci_base_alpha": (0.55, float, 0.1, 0.9, None,
        "QCI 基础 α：relevance=0 时的 importance 权重（迭代322，默认 0.55）"),
    "retriever.qci_relevance_slope": (0.25, float, 0.0, 0.5, None,
        "QCI slope：每单位 relevance 降低 α 的幅度（迭代322，默认 0.25）"),

    # ── MMR 边际信息量过滤（迭代321）──
    # OS 类比：Linux multiqueue block I/O merge — 物理地址相邻的请求合并，避免重复 I/O
    # MMR 在 DRR 之后对内容语义去冗余，λ 越大越偏 relevance，越小越偏 diversity
    "retriever.mmr_enabled": (True, bool, None, None, None,
        "是否启用 MMR 内容去冗余（在 DRR 之后对 summary 语义去重，迭代321）"),
    "retriever.mmr_lambda": (0.6, float, 0.0, 1.0, None,
        "MMR λ 参数：λ=1.0 纯 relevance，λ=0.0 纯 diversity，默认 0.6 略偏 relevance"),

    # ── Hybrid FTS5+BM25（迭代126）──
    # OS 类比：L1/L2 多级缓存协议 — L1(FTS5)命中不足时查 L2(BM25)补充长尾
    "retriever.hybrid_fts_min_count": (3, int, 1, 20, None,
        "FTS5 结果少于此值时触发 BM25 补充召回（迭代126：默认3，等于 top_k 保障下限）"),

    # ── 迭代334：IWCSI — Importance-Weighted Cold-Start Injection ──
    # OS 类比：Linux DAMON damos_action=PAGE_PROMOTE — 强制曝光 cold region 打破死锁
    # 信息论依据：零召回高imp chunk 期望信息增益最高（I = importance × 1.0），
    #   但语义鸿沟导致 FTS5 永不命中 → IWCSI 是 cold-start SNR 修复机制
    "retriever.cold_start_enabled": (True, bool, None, None, None,
        "是否启用 IWCSI 冷启动注入（FULL 模式 + positive 不足时强制曝光高 imp 零召回 chunk）"),
    "retriever.cold_start_imp_threshold": (0.75, float, 0.5, 1.0, None,
        "IWCSI 触发的 importance 下限：只强制曝光 importance >= 此值的零召回 chunk"),
    "retriever.cold_start_max_inject": (1, int, 1, 3, None,
        "IWCSI 每次最多强制注入的 chunk 数量（默认1，避免挤占其他类型）"),
    # ── iter376: Emotional Salience Retrieval Boost ──────────────────────────
    # OS 类比：Linux OOM Score 情绪加权 — 高情绪显著性记忆优先保留，类比 oom_adj=-800
    # 认知科学依据：McGaugh (2000) 情绪增强记忆巩固（amygdala-hippocampus interaction）
    #   情绪事件（高 arousal）触发杏仁核激活，通过 norepinephrine 增强海马编码强度。
    #   在 memory-os 中：emotional_weight > 0 的 chunk 代表高情绪显著性知识，
    #   检索时应优先，类比高 oom_adj 进程保留在内存中不被 kswapd 淘汰。
    "retriever.emotional_boost_factor": (0.08, float, 0.0, 0.5, None,
        "情绪显著性加分系数：score += emotional_weight * factor（emotional_weight > threshold 时）"),
    "retriever.emotional_boost_threshold": (0.4, float, 0.0, 1.0, None,
        "情绪显著性加分触发阈值：emotional_weight > 此值时才加分，防止低情绪度噪音"),

    # ── 迭代335：Ghost Reaper — zombie chunk FTS5 污染清除 ──
    # OS 类比：Linux wait4() — 回收 zombie 进程，释放进程表项
    # ghost chunk = importance=0 且 summary=[merged→...] 的已合并 chunk
    # 仍在 FTS5 索引中，消耗 result slot 并产生 false recall
    "retriever.ghost_filter_enabled": (True, bool, None, None, None,
        "是否在 fts_search 中过滤 importance=0 的 ghost chunk（Layer 2 软过滤防护）"),

    # ── iter388: Temporal Priming — 时间性启动效应 ──
    # 认知科学依据：Tulving & Schacter (1990) Priming Effect —
    #   最近在同会话中被召回的记忆，在随后的检索中被激活的阈值降低（启动效应）。
    #   神经基础：海马-新皮层投射维持短期激活状态（working memory buffer），
    #   最近命中的 chunk 仍处于"激活窗口"，再次相关时更易浮现。
    # OS 类比：CPU 时间局部性 (temporal locality) — 最近访问的 cache line 比
    #   未访问的有更高命中概率（L2/L3 temporal prefetch）。
    "retriever.priming_enabled": (True, bool, None, None, None,
        "是否启用会话内时间性启动效应：同会话最近召回的 chunk 得 priming_boost 加分（iter388）"),
    "retriever.priming_boost": (0.08, float, 0.0, 0.30, None,
        "启动效应加分幅度：同会话最近召回的 chunk score += priming_boost（iter388，默认 0.08）"),

    # ── iter389: Reconsolidation Window — 再巩固窗口 ──────────────────────────
    # 认知科学依据：Walker & Stickgold (2004) Memory Reconsolidation —
    #   记忆在每次被激活后进入不稳定的"可塑窗口"，然后重新巩固（reconsolidation）。
    #   间隔越长的重复激活，巩固效果越强（spacing effect, Ebbinghaus 1885）。
    #   短间隔内反复命中（< 1小时）= 工作记忆内刷新，不触发长时记忆巩固。
    #   长间隔后命中（> 1天）= 真正的间隔回忆（spaced retrieval），SM-2 质量最高。
    # OS 类比：Linux MGLRU page aging —
    #   刚被访问的页（youngest generation）再次访问不触发 generation 晋升（短时局部性），
    #   但跨 aging interval 后再次访问会晋升到 younger generation（真正的热页）。
    # 在 update_accessed() 中：根据 now - last_accessed 动态推断 SM-2 quality，
    #   替代之前固定 quality=4 的简化假设，实现真正的 spacing effect。
    "recon.short_gap_hours": (1.0, float, 0.0, 24.0, None,
        "再巩固短间隔阈值（小时）：gap < 此值时 SM-2 quality=3（无增益，仅更新访问时间）"),
    "recon.medium_gap_hours": (24.0, float, 1.0, 168.0, None,
        "再巩固中间隔阈值（小时）：short<=gap<medium 时 SM-2 quality=4（轻微加固）"),
    "recon.long_gap_quality": (5, int, 3, 5, None,
        "gap >= medium_gap_hours 时的 SM-2 quality（默认5=最大巩固，间隔回忆效果最强）"),
    "recon.enabled": (True, bool, None, None, None,
        "是否启用再巩固窗口动态 SM-2 quality（False 时回退到固定 quality=4）"),

    # ── iter412: Testing Effect — 高难度检索强化记忆巩固 ─────────────────────────
    # 认知科学依据：Roediger & Karpicke (2006) "Test-Enhanced Learning" —
    #   主动检索（而非被动重读）显著提升长期保留率（+50%）。
    #   Bjork (1994) "Desirable Difficulties" — 需要努力的检索（retrieval difficulty 高）
    #   形成更强、更持久的记忆痕迹（elaborative encoding 更深）。
    #   Kornell et al. (2011) — 难但成功的检索比容易的检索巩固效果更强。
    # 实现：R_at_recall = exp(-gap_hours / (stability × 24))，
    #   difficulty = max(0, 1 - R_at_recall)，
    #   quality_bonus = round(difficulty × scale)（仅在 recall_quality=None 时生效）
    # OS 类比：Linux L3 cache miss → aggressive LRU promotion —
    #   L1 命中（容易检索）不改变 LRU 位置；L3 miss（困难检索）→ 强制 cache line 晋升到 L1/L2
    "recon.testing_effect_enabled": (True, bool, None, None, None,
        "是否启用 iter412 Testing Effect：低 retrievability 时的检索难度 → 增加 SM-2 quality bonus"),
    "recon.testing_effect_scale": (2.0, float, 0.0, 4.0, None,
        "Testing Effect 难度-质量转换系数：quality_bonus = round(difficulty × scale)，最大 +2（iter412）"),

    # ── iter413: Sleep Consolidation — 离线记忆巩固 ──────────────────────────
    # 认知科学依据：Stickgold (2005) "Sleep-dependent memory consolidation" —
    #   NREM 睡眠中海马体重放最近学习的记忆，将其转移到新皮层（系统巩固理论）。
    #   Walker & Stickgold (2004) — 学习后睡眠使次日表现提升 20-30%。
    #   Diekelmann & Born (2010) — SWS 期间的主动系统巩固降低干扰敏感性。
    # 实现：SessionStart 时对上一 session（过去 24hr）访问的高重要性 chunk 应用轻微 stability 加成
    # OS 类比：Linux pdflush/writeback daemon — session 间隙（idle period）后台巩固 dirty pages，
    #   类比海马-新皮层离线重放（sleep replay）将 working memory → long-term storage
    "consolidation.enabled": (True, bool, None, None, None,
        "是否启用 iter413 Sleep Consolidation：SessionStart 时对上一 session 的高重要性 chunk 应用离线巩固"),
    "consolidation.boost_factor": (1.06, float, 1.0, 1.30, None,
        "离线巩固稳定性加成系数：stability × boost_factor（iter413，保守值 1.06 ≈ 6%）"),
    "consolidation.min_importance": (0.70, float, 0.3, 1.0, None,
        "触发离线巩固的重要性阈值：importance >= 此值的 chunk 才参与 sleep replay（iter413）"),
    "consolidation.window_hours": (24, int, 1, 168, None,
        "离线巩固的时间窗口（小时）：只对过去 N 小时内被访问的 chunk 进行巩固（iter413）"),
    "consolidation.max_chunks": (50, int, 5, 500, None,
        "每次 SessionStart 最多巩固的 chunk 数量（iter413，按 importance 排序取前 N 个）"),

    # ── iter428: Event Segmentation — Session Boundary Consolidation Gate ───────────────────
    # 认知科学依据：Zacks et al. (2007) Event Segmentation Theory (Psychological Science) —
    #   人类将连续经验分割为离散事件单元，边界处记忆编码最强（boundary advantage）。
    #   Radvansky & Copeland (2006) "Walking through doorways causes forgetting" —
    #   穿越事件边界（空间/时间）触发短暂记忆抑制（doorway effect）：
    #   旧情境末尾的信息被短暂压制（约 5 分钟），新情境开始后的信息获得额外编码加成。
    # OS 类比：ext4 jbd2 journal commit boundary —
    #   新 epoch 首批写入的 page（刚越过 commit point）= 最高一致性保证（boundary boost）；
    #   commit 前的 dirty page（旧 epoch 末尾）= 不稳定窗口（doorway penalty）。
    "consolidation.boundary_enabled": (True, bool, None, None, None,
        "是否启用 iter428 Event Segmentation：session boundary 处分叉 sleep consolidation 逻辑"),
    "consolidation.boundary_multiplier": (1.5, float, 1.0, 3.0, None,
        "iter428: boundary boost 乘子 — boundary_proximity=+1.0 时 stability × (boost_factor + (multiplier-1)×proximity)，"
        "默认 1.5：boundary chunk 比普通 chunk 多 +50% sleep consolidation 加成"),
    "consolidation.boundary_grace_secs": (300, int, 30, 3600, None,
        "iter428: session 开始后多少秒内写入的 chunk 被视为 boundary boost 候选（默认 5 分钟）"),
    "consolidation.doorway_penalty": (0.05, float, 0.0, 0.3, None,
        "iter428: doorway effect stability 惩罚系数（boundary_proximity < -0.5 时应用，"
        "默认 0.05 = 最多 5% stability 惩罚，模拟 Radvansky 2006 doorway forgetting）"),

    # ── iter429: Enactment Effect — 行动编码加成 ─────────────────────────────
    # 认知科学依据：Engelkamp & Zimmer (1989) Subject-Performed Tasks (SPT) —
    #   亲自执行动作（SPT）比仅语言描述（VT）的记忆留存率高约 40%；
    #   行动编码激活运动皮层 + 语义系统双路径，形成更强的多模态痕迹。
    # OS 类比：Linux writeback — 写操作（exec/write syscall）创建比读操作（read）
    #   更深的 page cache dirty state，需要更多 I/O 才能清除。
    # 检测：chunk 的 source_type='tool_result' 或 content 包含工具调用签名
    "store_vfs.enactment_enabled": (True, bool, None, None, None,
        "是否启用 iter429 Enactment Effect：agent 工具调用产生的 chunk 获得 stability 加成"),
    "store_vfs.enactment_boost": (1.4, float, 1.0, 3.0, None,
        "iter429: 行动编码 stability 乘子 — 执行工具调用的 chunk stability × enactment_boost，"
        "默认 1.4（对应 SPT 比 VT 高约 40% 的留存率优势，Engelkamp 1989）"),
    "store_vfs.enactment_cap": (365.0, float, 1.0, 365.0, None,
        "iter429: enactment effect 后 stability 上限（避免超过遗忘曲线最大值）"),
    "store_vfs.enactment_tool_types": (
        "Bash,Edit,Write,NotebookEdit",
        str, None, None, None,
        "iter429: 触发行动编码加成的工具名列表（逗号分隔），"
        "这些工具产生副作用（写磁盘/执行命令），比 Read/Glob 等只读工具有更强行动编码"),

    # ── iter390: Prospective Memory — 展望记忆触发 ───────────────────────────
    # 认知科学依据：Einstein & McDaniel (1990) Prospective Memory —
    #   意图性记忆（"记得在X时做Y"）需要在未来条件满足时主动提取。
    #   extractor 检测展望意图信号 → 注册 trigger_conditions；
    #   retriever 在 query 匹配时注入关联 chunk（提醒效果）。
    # OS 类比：Linux inotify — 注册事件监听，条件满足时唤醒等待进程。
    "prospective.enabled": (True, bool, None, None, None,
        "是否启用展望记忆触发（extractor 检测意图 + retriever 注入提醒，iter390）"),
    "prospective.max_inject": (2, int, 1, 5, None,
        "每次检索最多注入的展望记忆 chunk 数量（避免占满 Top-K，默认 2）"),
    "prospective.score_boost": (0.8, float, 0.3, 1.0, None,
        "展望记忆触发注入的初始评分（较高以确保注入，但低于 design_constraint）"),

    # ── iter391: Inhibition of Return — 返回抑制动态衰减 ─────────────────────
    # 认知科学依据：Posner (1980) Inhibition of Return —
    #   注意力访问一个位置后，有 ~300ms 的返回抑制（IOR）；对记忆系统同样适用：
    #   Klein (2000) IOR in memory search — 最近被检索的项目有短暂的检索抑制，
    #   防止搜索固着在同一位置，促进广度探索。
    # OS 类比：Linux CFQ fair queuing anti-starvation —
    #   刚被服务的请求在 timeslice 内被降优先级，让其他等待队列获得服务机会。
    # 实现：session 级 IOR 状态（chunk_id → last_inject_turn），
    #   score *= (1 - ior_penalty × exp(-ior_decay_rate × turns_since_inject))
    "retriever.ior_enabled": (True, bool, None, None, None,
        "是否启用 IOR 返回抑制（最近注入的 chunk 获得短暂的分数惩罚，iter391）"),
    "retriever.ior_penalty": (0.20, float, 0.0, 0.5, None,
        "IOR 峰值惩罚系数：刚被注入的 chunk 分数 × (1 - ior_penalty)（iter391，默认 0.20）"),
    "retriever.ior_decay_turns": (3, int, 1, 20, None,
        "IOR 半衰期（检索轮次）：经过此轮次后惩罚衰减到一半（iter391，默认 3 轮）"),
    "retriever.ior_exempt_types": ("design_constraint", str, None, None, None,
        "IOR 豁免的 chunk_type（逗号分隔，这些类型不受返回抑制影响，iter391）"),

    # ── iter393：Semantic Distance Decay in Spreading Activation ──
    # 认知科学：Collins & Loftus (1975) Spreading Activation Theory —
    #   激活从锚点节点沿语义图扩散，激活量随语义距离（路径长度）衰减。
    #   距离越远，激活越低，形成自然的语义相关性梯度。
    # OS 类比：NUMA 局部性 — 本节点内存访问延迟低，跨 2 个 NUMA 节点的访问
    #   延迟呈指数增长（L1→L2→L3→DRAM→remote DRAM 约 3-10 倍梯度）。
    "retriever.sa_distance_decay_enabled": (True, bool, None, None, None,
        "是否对 spreading activation 应用语义距离衰减（iter393，默认启用）"),
    "retriever.sa_distance_decay_factor": (0.6, float, 0.1, 1.0, None,
        "每跳语义距离衰减系数：hop_distance 跳的激活分乘以此系数的 hop 次方（iter393，默认 0.6）"),
    "retriever.sa_max_hops": (2, int, 1, 4, None,
        "spreading activation 最大跳数（iter393，默认 2 跳；跳数越多计算越贵）"),

    # ── iter423: Fan Effect — IDF加权 Spreading Activation（Anderson 1974）──
    # 认知科学依据：Anderson (1974) Fan Effect —
    #   与一个概念关联的事实越多（fan-out 越大），检索每条具体事实越慢越难。
    #   高扇出节点（如"authentication"关联50个chunk）的激活传播效率低于
    #   低扇出节点（如"port_8080"只关联1-2个chunk）。
    # 实现：spreading activation 中，高 degree entity 的边贡献乘以 IDF 权重（降权）：
    #   IDF_weight = log(1 + median_degree / (1 + entity_degree))，归一化到 [0,1]
    #   entity_degree = 该 entity 在 entity_edges 中的总边数（in + out）
    #   degree >= fan_min_degree 时才应用惩罚（低扇出 entity 不惩罚）
    # OS 类比：Linux CPU cache set-associativity conflict —
    #   太多 cache line 映射到同一 set（高扇出）→ 频繁 eviction → 命中率下降。
    #   Fan Effect 惩罚 = 降低高扇出 entity 的"缓存命中率"（activation strength）。
    "retriever.fan_effect_enabled": (True, bool, None, None, None,
        "是否启用 Fan Effect IDF 加权（iter423：高扇出 entity 激活权重降低）"),
    "retriever.fan_effect_min_degree": (3, int, 1, 50, None,
        "iter423: Fan Effect 触发的最低 entity degree 阈值（低于此值的 entity 不惩罚）"),
    "retriever.fan_effect_idf_weight": (0.5, float, 0.0, 1.0, None,
        "iter423: IDF 权重混合系数：edge_score × (1 - fan_effect_idf_weight × (1 - idf_factor))，"
        "0=不惩罚，1=完全 IDF 权重"),

    # ── iter424: Mood-Congruent Memory — 情绪效价一致性检索增强（Bower 1981）──
    # 认知科学依据：Bower (1981) "Mood and memory" —
    #   人在某种情绪状态（情绪诱导实验）下，更容易回忆起与该情绪一致的记忆。
    #   正面情绪 → 优先检索正面内容；负面情绪 → 优先检索负面/危机内容。
    #   Bower (1981) Associative Network Theory：情绪节点（mood nodes）与记忆节点相连，
    #   情绪激活会扩散到同效价的记忆，降低其检索阈值。
    #   Matt et al. (1992) Meta-analysis: MCM effect is robust across recall + recognition tasks。
    # 应用：query 包含情绪效价词（崩溃/突破）→ 推断用户当前情绪状态 →
    #   chunk.emotional_valence 与 query 效价方向一致 → score += mcm_boost × |valence_match|。
    # OS 类比：Linux NUMA-aware page placement —
    #   进程有 preferred NUMA node（情绪状态），访问同 node 的 page（同效价 chunk）延迟最低。
    "retriever.mcm_enabled": (True, bool, None, None, None,
        "是否启用 iter424 Mood-Congruent Memory：query 情绪效价与 chunk 效价一致时检索加分"),
    "retriever.mcm_boost": (0.05, float, 0.0, 0.20, None,
        "iter424: 情绪效价一致时的 score boost（默认 +0.05，query_valence × chunk_valence > 0 时生效）"),
    "retriever.mcm_valence_threshold": (0.3, float, 0.0, 1.0, None,
        "iter424: query/chunk 情绪效价触发阈值（|valence| >= 此值才参与 MCM 匹配，避免弱情绪噪音）"),

    # ── iter394：Contextual Similarity Boost — 编码情境检索增强 ──
    # 认知科学：Tulving (1983) Encoding Specificity Principle +
    #   Godden & Baddeley (1975) Context-Dependent Memory —
    #   检索时的任务情境（session_type: debug/design/refactor/qa）与编码时越相似，
    #   记忆提取成功率越高。水下学习 → 水下更易回忆（情境再现效应）。
    # OS 类比：NUMA-aware scheduling — 进程在同一 NUMA 节点上运行时，
    #   访问该节点分配的内存延迟最低（情境局部性 ≈ NUMA 局部性）。
    # 实现：检索时从 query 提取 session_type/task_verbs，
    #   与 chunk.encoding_context.session_type/task_verbs 比对，
    #   匹配时加 context_type_boost（+0.05）+ task_verbs overlap boost（+0.03）。
    "retriever.context_type_boost_enabled": (True, bool, None, None, None,
        "是否启用 session_type 情境匹配 boost（iter394，默认启用）"),
    "retriever.context_type_boost": (0.05, float, 0.0, 0.15, None,
        "session_type 精确匹配时的 score boost（iter394，默认 +0.05）"),
    "retriever.task_verbs_boost": (0.03, float, 0.0, 0.10, None,
        "task_verbs Jaccard 交集加权 boost 上限（iter394，默认 +0.03）"),

    # ── BM25 Fallback Global Discount（迭代131）──
    # OS 类比：Linux NUMA Aware Scheduling — 当 local node 内存不足强制 cross-node 分配时，
    #   调度器施加 NUMA fault penalty（migratable page cost），阻止低相关性跨节点抢占。
    #   memory-os 对应：BM25 全表扫描时所有 72 global chunk 参与竞争，
    #   高 importance global chunk（kernel patch design_constraint, imp=0.95）
    #   通过偶发词汇重叠（如"记忆"）获得 relevance 虚高分，
    #   NUMA penalty(global)=0.05 无法阻止其排名第一。
    #   bm25_global_discount：BM25 fallback 路径中 global 项目 chunk 的 relevance 折扣系数
    #   default=0.4 — 远强于 FTS5 路径的 0.05 惩罚，匹配 BM25 不确定性高的事实
    "retriever.bm25_global_discount": (0.4, float, 0.1, 1.0, None,
        "BM25 全表扫描 fallback 路径中 global 项目 chunk 的 relevance 折扣（迭代131，默认0.4）"),

    # ── design_constraint 注入上限（迭代128）──
    # OS 类比：Linux mlock RLIMIT_MEMLOCK — 限制进程可以锁定的内存总量，
    # 防止单个进程无限 mlock 耗尽系统内存（所有 design_constraint 强制注入）。
    # 默认 3：确保最相关的约束能注入，但不会因约束数量增长导致注入膨胀。
    "retriever.max_forced_constraints": (3, int, 1, 16, None,
        "design_constraint 强制注入的最大数量（迭代128：防止约束膨胀，按 BM25 相关性择优注入）"),

    # ── Proactive Swap Probe（迭代355）──
    # OS 类比：Linux MGLRU (Multi-Generation LRU) 主动提升 swap 热页
    # 即使 FTS5 已有结果，仍检查 swap 中高 importance chunk 是否更相关
    "retriever.proactive_swap_enabled": (True, bool, None, None, None,
        "主动 swap 探针：即使 top_k 非空，仍检查 swap 中高 importance 的 chunk（迭代355）"),
    "retriever.proactive_swap_imp_threshold": (0.80, float, 0.5, 1.0, None,
        "proactive swap 探针的 importance 阈值：只恢复 importance >= 此值的 chunk"),
    "retriever.proactive_swap_max_restore": (3, int, 1, 10, None,
        "每次查询最多从 swap 恢复多少个 chunk（限制写连接切换开销）"),

    # ── Pin Decay + Cap（迭代356）──
    # OS 类比：Linux memcg pin_user_pages_lock + RLIMIT_MEMLOCK
    # 问题：chunk_pins 无过期机制，45% pin rate（47/105）阻塞 LRU 驱逐空间
    "pin.decay_enabled": (True, bool, None, None, None,
        "Pin 衰减开关：长期未访问的 soft pin 自动解除（迭代356）"),
    "pin.decay_days": (30, int, 7, 180, None,
        "soft pin 衰减阈值（天）：soft pin 的 chunk 超过 N 天未访问则自动解除"),
    "pin.cap_pct": (15, int, 5, 50, None,
        "项目 pin 上限（%）：pinned chunk 占项目总量不超过此比例（hard+soft 合计）"),
    "pin.cap_apply_on_pin": (True, bool, None, None, None,
        "新增 pin 时立即检查 cap，超限则驱逐最旧 soft pin（类比 RLIMIT_MEMLOCK enforcement）"),

    # ── Cross-Session KSM（迭代358）──
    # OS 类比：Linux KSM (Kernel Samepage Merging) ksmd 线程周期扫描
    # 问题：项目的 17 个 session 各自从 cold start 重新加载相同 chunk，
    # 无法共享跨 session 热点知识（KSM 缺失导致 knowledge locality 低）
    "ksm.enabled": (True, bool, None, None, None,
        "跨 Session KSM：扫描多 session working set 共享热点 chunk（迭代358）"),
    "ksm.min_access_count": (3, int, 1, 20, None,
        "chunk 在单 session 中的最低访问次数（才被视为热点候选）"),
    "ksm.min_sessions": (2, int, 2, 10, None,
        "chunk 必须出现在至少 N 个 session 才被提升（防止单 session 噪音）"),

    # ── TLB v2（迭代64）──
    "retriever.tlb_max_entries": (8, int, 1, 64, None,
        "TLB 最大 slot 数量（类比 CPU TLB 通常 64-1024 entries）"),

    # ── Memory Zones（迭代82）──
    "retriever.exclude_types": ("prompt_context", str, None, None, None,
        "逗号分隔的 chunk_type 列表，从检索候选中排除（OS 类比：Linux ZONE_RESERVED）"),

    # ── iter427：Serial Position Effect（Murdock 1962）──
    # OS 类比：BFQ front-merge — 高优先级 I/O 置于 dispatch queue 首/尾位置
    "retriever.serial_position_enabled": (True, bool, None, None, None,
        "是否启用序列位置效应注入顺序优化（Murdock 1962 primacy+recency），"
        "将高价值 chunk 置于注入块首/尾，避免 LLM 输出干扰效应。"),
    "retriever.serial_position_imp_threshold": (0.85, float, 0.0, 1.0, None,
        "importance >= 此值的 chunk 视为 primacy/recency 候选（默认 0.85）"),
    "retriever.serial_position_recency_types": ("decision,design_constraint,reasoning_chain",
        str, None, None, None,
        "逗号分隔的 chunk_type 列表，这些类型的 chunk 优先候选 primacy/recency 位置"),

    # ── 迭代359：Session Injection Deduplication ──
    "retriever.session_dedup_threshold": (2, int, 1, 10, None,
        "同一 session 内 chunk 被注入 >= 此次数后从输出中去重（OS 类比：copy-on-write lazy page dedup，"
        "只有同一页被重复 mapped 达到阈值才触发物理页合并）。design_constraint 类型豁免。"),

    # ── Context Pressure Governor（迭代55）──
    "governor.turns_low": (5, int, 1, 20, None,
        "低压阈值：对话轮次 ≤ 此值时判定为 LOW（上下文充裕）"),
    "governor.turns_high": (15, int, 5, 50, None,
        "高压阈值：对话轮次 ≥ 此值时判定为 HIGH（接近 compaction）"),
    "governor.turns_critical": (25, int, 10, 80, None,
        "临界阈值：对话轮次 ≥ 此值时判定为 CRITICAL（极高 compaction 风险）"),
    "governor.compact_high": (2, int, 1, 10, None,
        "compaction 次数 ≥ 此值时判定为 HIGH"),
    "governor.compact_critical": (4, int, 2, 20, None,
        "compaction 次数 ≥ 此值时判定为 CRITICAL"),
    "governor.recent_compact_secs": (120, int, 30, 600, None,
        "compaction 后此秒数内视为高压（刚溢出，需要精简注入）"),
    "governor.scale_low": (1.5, float, 1.0, 3.0, None,
        "LOW 压力缩放因子（> 1.0 多注入，提升信息密度）"),
    "governor.scale_high": (0.6, float, 0.2, 1.0, None,
        "HIGH 压力缩放因子（< 1.0 精简注入）"),
    "governor.scale_critical": (0.3, float, 0.1, 0.8, None,
        "CRITICAL 压力缩放因子（最小注入，仅保留最关键信息）"),
    "governor.window_hours": (2.0, float, 0.5, 24.0, None,
        "信号时间窗口：只统计最近 N 小时内的 compaction/turns（防跨 session 累积误判）"),
    "governor.consecutive_decay_hours": (1.0, float, 0.25, 12.0, None,
        "consecutive_high 衰减窗口：超过此时间未更新则 reset（防历史锁死）"),

    # ── PSI（迭代36）──
    "psi.window_size": (20, int, 5, 100, None,
        "PSI 计算窗口：最近 N 次检索的统计样本数"),
    "psi.latency_baseline_ms": (30.0, float, 1.0, 200.0, None,
        "检索延迟固定基线（ms），超过此值视为 stall。adaptive_baseline 开启时仅作 fallback"),
    "psi.hit_rate_baseline_pct": (50.0, float, 10.0, 90.0, None,
        "命中率基线（%），低于此值视为 quality stall"),
    "psi.capacity_some_pct": (70, int, 40, 90, None,
        "容量压力 SOME 阈值（%），使用率超过此值开始感受压力"),
    "psi.capacity_full_pct": (90, int, 70, 100, None,
        "容量压力 FULL 阈值（%），使用率超过此值为严重压力"),

    # ── PSI Adaptive Baseline（迭代60）──
    "psi.adaptive_baseline": (1, int, 0, 1, None,
        "启用自适应延迟基线（1=开启，0=固定基线）。开启后用滑动窗口 P50×margin 替代固定 latency_baseline_ms"),
    "psi.adaptive_margin": (1.5, float, 1.1, 3.0, None,
        "自适应基线 margin 系数。实际基线 = P50 × margin。1.5 表示允许 50% 的延迟波动"),
    "psi.adaptive_min_samples": (5, int, 3, 20, None,
        "自适应基线最小样本数。样本不足时 fallback 到固定 latency_baseline_ms"),
}

# ── 磁盘配置缓存（进程内只读一次）──
_disk_config: Optional[dict] = None


def _load_disk_config() -> dict:
    """加载 sysctl.json 配置文件（懒加载，进程内缓存）。"""
    global _disk_config
    if _disk_config is not None:
        return _disk_config
    if os.path.exists(SYSCTL_FILE):
        try:
            with open(SYSCTL_FILE, encoding="utf-8") as _f:
                _disk_config = json.loads(_f.read())
        except Exception:
            _disk_config = {}
    else:
        _disk_config = {}
    return _disk_config


def _invalidate_cache():
    """强制重新加载磁盘配置（sysctl_set 后调用）。"""
    global _disk_config
    _disk_config = None


def get(key: str, project: str = None) -> Any:
    """
    获取 tunable 值。
    优先级：环境变量 > namespace(project) > global sysctl.json > 默认值。

    迭代27 OS 类比：sysctl vm.swappiness 的读取路径 —
      先查 /proc/sys/vm/swappiness（运行时覆盖），再用编译时默认值。
    迭代37 OS 类比：Linux Namespaces —
      容器内进程看到的 /proc/sys/ 是 namespace 隔离后的视图，
      每个容器可以有独立的 sysctl 值（net.core.somaxconn 等）。
      get(key, project) 就是在 project namespace 视图中读取 tunable。
    """
    if key not in _REGISTRY:
        raise KeyError(f"sysctl: unknown tunable '{key}'")

    default, typ, lo, hi, env_key, desc = _REGISTRY[key]

    # 1. 环境变量（最高优先级，全局生效）
    env_names = []
    if env_key:
        env_names.append(env_key)
    env_names.append(f"MEMORY_OS_{key.upper().replace('.', '_')}")

    for env_name in env_names:
        env_val = os.environ.get(env_name)
        if env_val is not None:
            try:
                return _coerce(env_val, typ, lo, hi)
            except (ValueError, TypeError):
                break

    # 2. Per-project namespace 覆盖（迭代37 Namespaces）
    if project:
        disk = _load_disk_config()
        ns = disk.get("namespaces", {})
        if isinstance(ns, dict):
            proj_ns = ns.get(project, {})
            if isinstance(proj_ns, dict) and key in proj_ns:
                try:
                    return _coerce(proj_ns[key], typ, lo, hi)
                except (ValueError, TypeError):
                    pass

    # 3. sysctl.json 全局配置
    disk = _load_disk_config()
    if key in disk:
        try:
            return _coerce(disk[key], typ, lo, hi)
        except (ValueError, TypeError):
            pass

    # 4. 默认值
    return default


def sysctl_set(key: str, value: Any, project: str = None) -> None:
    """
    运行时修改 tunable 并持久化到 sysctl.json。
    OS 类比（迭代27）：sysctl -w vm.swappiness=60
    OS 类比（迭代37）：ip netns exec <ns> sysctl -w ...
      在指定 namespace 内设置 sysctl 值。

    project=None → 写入全局配置
    project=<id> → 写入 per-project namespace 覆盖
    """
    if key not in _REGISTRY:
        raise KeyError(f"sysctl: unknown tunable '{key}'")

    default, typ, lo, hi, env_key, desc = _REGISTRY[key]
    coerced = _coerce(value, typ, lo, hi)

    os.makedirs(MEMORY_OS_DIR, exist_ok=True)
    disk = _load_disk_config()

    if project:
        # 迭代37：写入 per-project namespace
        if "namespaces" not in disk:
            disk["namespaces"] = {}
        if project not in disk["namespaces"]:
            disk["namespaces"][project] = {}
        disk["namespaces"][project][key] = coerced
    else:
        disk[key] = coerced

    with open(SYSCTL_FILE, 'w', encoding='utf-8') as _f:
        _f.write(json.dumps(disk, ensure_ascii=False, indent=2))
    _invalidate_cache()


def sysctl_list(project: str = None) -> dict:
    """
    返回所有 tunable 的当前值（≈ sysctl -a）。
    迭代37：传入 project 时返回该 namespace 视图下的值。
    OS 类比：nsenter --target <pid> sysctl -a — 在指定 namespace 内列出所有参数。
    """
    result = {}
    for key in sorted(_REGISTRY.keys()):
        result[key] = {
            "value": get(key, project=project),
            "default": _REGISTRY[key][0],
            "type": _REGISTRY[key][1].__name__,
            "range": [_REGISTRY[key][2], _REGISTRY[key][3]],
            "description": _REGISTRY[key][5],
        }
    return result


# ── 迭代37：Namespace Management — Per-Project 配置隔离 ──────────

def ns_list(project: str) -> dict:
    """
    迭代37：列出指定项目 namespace 中的所有覆盖值。
    OS 类比：ip netns identify <pid> + nsenter sysctl -a
      查看容器内哪些 sysctl 被覆盖了。

    返回 dict：{key: value} 只包含被覆盖的 tunable（不含继承的全局值）。
    空 dict 表示该项目使用全局默认配置。
    """
    disk = _load_disk_config()
    ns = disk.get("namespaces", {})
    if not isinstance(ns, dict):
        return {}
    proj_ns = ns.get(project, {})
    if not isinstance(proj_ns, dict):
        return {}
    # 验证并返回合法的覆盖值
    result = {}
    for key, val in proj_ns.items():
        if key in _REGISTRY:
            default, typ, lo, hi, env_key, desc = _REGISTRY[key]
            try:
                result[key] = _coerce(val, typ, lo, hi)
            except (ValueError, TypeError):
                pass
    return result


def ns_clear(project: str) -> int:
    """
    迭代37：清除指定项目的 namespace（恢复使用全局配置）。
    OS 类比：ip netns delete <name> — 销毁 namespace，进程回到 init namespace。

    返回清除的覆盖项数量。
    """
    disk = _load_disk_config()
    ns = disk.get("namespaces", {})
    if not isinstance(ns, dict) or project not in ns:
        return 0
    count = len(ns[project]) if isinstance(ns[project], dict) else 0
    del ns[project]
    disk["namespaces"] = ns
    os.makedirs(MEMORY_OS_DIR, exist_ok=True)
    with open(SYSCTL_FILE, 'w', encoding='utf-8') as _f:
        _f.write(json.dumps(disk, ensure_ascii=False, indent=2))
    _invalidate_cache()
    return count


def ns_list_all() -> dict:
    """
    迭代37：列出所有已创建的 namespace（≈ ip netns list）。
    返回 dict：{project_id: {覆盖的 tunable 数量}}
    """
    disk = _load_disk_config()
    ns = disk.get("namespaces", {})
    if not isinstance(ns, dict):
        return {}
    return {proj: len(overrides) if isinstance(overrides, dict) else 0
            for proj, overrides in ns.items()}


def _coerce(value: Any, typ: type, lo: Any, hi: Any) -> Any:
    """类型转换 + 范围校验。"""
    if typ is bool:
        # bool 特殊处理：JSON 的 true/false、字符串 "true"/"false"、0/1
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes")
        return bool(value)
    v = typ(value)
    if lo is not None and v < lo:
        v = lo
    if hi is not None and v > hi:
        v = hi
    return v


# ── 迭代47：sched_ext — Extensible Scheduler（可编程调度策略）──────────
#
# OS 类比：Linux sched_ext (Extensible Scheduler, Linux 6.12, 2024, Tejun Heo/Meta)
#
# Linux sched_ext 背景：
#   CFS/EEVDF 是内核硬编码的调度器——所有工作负载用同一套策略。
#   不同场景（延迟敏感 vs 吞吐优先 vs 大小核调度）需要不同策略，
#   但修改调度器需要改内核代码+重编译+重启。
#   sched_ext (Linux 6.12, 2024) 让用户通过 BPF 程序在用户态编写
#   自定义调度策略：
#     - 内核提供 struct_ops callback（enqueue/dequeue/dispatch/tick 等）
#     - 用户态 BPF 程序注册 callback 实现自定义逻辑
#     - 运行时动态加载/卸载，无需重编译内核
#     - fallback: BPF 程序 panic/timeout 时自动回退到内置 CFS
#
#   已有的 sched_ext 调度器（用户态实现）：
#     - scx_rusty: Rust 实现的 NUMA-aware scheduler
#     - scx_lavd: 延迟 vs 吞吐自适应
#     - scx_simple: 教学用最简实现
#
# memory-os 当前问题：
#   retriever.py 的 _classify_query_priority() 是硬编码规则：
#     - SKIP 模式列表是 Python 正则
#     - LITE/FULL 边界是字符数阈值
#     - 无法运行时扩展——新场景需要改 retriever.py 代码
#   等价于 Linux 只有 CFS 没有 sched_ext 的状态。
#
# 解决：
#   sysctl.json 新增 scheduler_ext_rules 数组，每条规则：
#     {pattern: "regex", priority: "SKIP|LITE|FULL", scope: "global|project_id"}
#   retriever.py 分类器优先评估自定义规则（用户态 BPF 策略），
#   无匹配时 fallback 到内置逻辑（内核态 CFS 默认策略）。
#   规则管理 API：sched_ext_add/sched_ext_remove/sched_ext_list/sched_ext_stats

_SCHED_EXT_KEY = "scheduler_ext_rules"


def sched_ext_list(project: str = None) -> list:
    """
    列出当前生效的 sched_ext 规则。
    OS 类比：bpftool struct_ops list — 列出已加载的 BPF 调度策略。

    参数：
      project — 只返回 global + 该 project scope 的规则（None = 全部）

    返回规则列表（按优先级排序：project scope > global scope）。
    """
    disk = _load_disk_config()
    rules = disk.get(_SCHED_EXT_KEY, [])
    if not isinstance(rules, list):
        return []

    valid = []
    for r in rules:
        if not isinstance(r, dict) or "pattern" not in r or "priority" not in r:
            continue
        priority = r["priority"].upper()
        if priority not in ("SKIP", "LITE", "FULL"):
            continue
        scope = r.get("scope", "global")
        if project and scope != "global" and scope != project:
            continue
        valid.append({
            "pattern": r["pattern"],
            "priority": priority,
            "scope": scope,
            "reason": r.get("reason", ""),
            "hits": r.get("hits", 0),
        })

    # 排序：project scope 优先于 global（更具体的规则先匹配）
    valid.sort(key=lambda x: (0 if x["scope"] != "global" else 1))
    return valid


def sched_ext_add(pattern: str, priority: str, scope: str = "global",
                  reason: str = "") -> dict:
    """
    添加一条 sched_ext 规则。
    OS 类比：bpftool struct_ops register — 注册新的 BPF 调度策略。

    参数：
      pattern  — 正则表达式（匹配 query 文本）
      priority — "SKIP" / "LITE" / "FULL"
      scope    — "global" 或 project_id（限定生效范围）
      reason   — 规则说明（可选，等价于 BPF 程序的 description）

    返回 dict：
      added — bool
      rule_count — 当前规则总数
      error — 错误信息（如果有）

    验证：
      - 正则必须可编译
      - priority 必须合法
      - 规则总数不超过 max_rules
    """
    # ── 迭代161：Lazy re import — 消除 config 模块级 re import 对 Stage 0+1 的污染 ──
    # OS 类比：dlopen(RTLD_LAZY) — 符号解析推迟到第一次调用时，不在 dlopen 时预绑定
    # sched_ext_add 只在 Stage 2（main() 已调用后）被调用，安全延迟 import
    import re as _re
    priority = priority.upper()
    if priority not in ("SKIP", "LITE", "FULL"):
        return {"added": False, "rule_count": 0,
                "error": f"invalid priority '{priority}', must be SKIP/LITE/FULL"}

    # 验证正则可编译（等价于 BPF verifier 检查程序安全性）
    try:
        _re.compile(pattern)
    except _re.error as e:
        return {"added": False, "rule_count": 0,
                "error": f"invalid regex: {e}"}

    max_rules = get("scheduler.ext_max_rules")

    os.makedirs(MEMORY_OS_DIR, exist_ok=True)
    disk = _load_disk_config()
    rules = disk.get(_SCHED_EXT_KEY, [])
    if not isinstance(rules, list):
        rules = []

    if len(rules) >= max_rules:
        return {"added": False, "rule_count": len(rules),
                "error": f"max rules reached ({max_rules})"}

    # 去重：相同 pattern + scope 不重复添加
    for r in rules:
        if isinstance(r, dict) and r.get("pattern") == pattern and r.get("scope", "global") == scope:
            return {"added": False, "rule_count": len(rules),
                    "error": "duplicate rule (same pattern+scope)"}

    rules.append({
        "pattern": pattern,
        "priority": priority,
        "scope": scope,
        "reason": reason,
        "hits": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })

    disk[_SCHED_EXT_KEY] = rules
    with open(SYSCTL_FILE, 'w', encoding='utf-8') as _f:
        _f.write(json.dumps(disk, ensure_ascii=False, indent=2))
    _invalidate_cache()

    return {"added": True, "rule_count": len(rules)}


def sched_ext_remove(pattern: str, scope: str = "global") -> dict:
    """
    移除一条 sched_ext 规则。
    OS 类比：bpftool struct_ops unregister — 卸载 BPF 调度策略。

    返回 dict：
      removed — bool
      rule_count — 剩余规则数
    """
    disk = _load_disk_config()
    rules = disk.get(_SCHED_EXT_KEY, [])
    if not isinstance(rules, list):
        return {"removed": False, "rule_count": 0}

    new_rules = [r for r in rules
                 if not (isinstance(r, dict) and r.get("pattern") == pattern
                         and r.get("scope", "global") == scope)]
    removed = len(rules) - len(new_rules)
    disk[_SCHED_EXT_KEY] = new_rules
    with open(SYSCTL_FILE, 'w', encoding='utf-8') as _f:
        _f.write(json.dumps(disk, ensure_ascii=False, indent=2))
    _invalidate_cache()

    return {"removed": removed > 0, "rule_count": len(new_rules)}


def sched_ext_match(query: str, project: str = None) -> Optional[dict]:
    """
    评估 query 是否匹配任何 sched_ext 规则。
    OS 类比：sched_ext 的 ops.enqueue() callback — BPF 程序决定任务入队策略。

    评估顺序（首条匹配即返回）：
      1. project-scope 规则（最具体）
      2. global-scope 规则

    返回匹配的规则 dict（含 priority/reason/pattern），None = 无匹配（fallback 到内置策略）。
    匹配时自动递增 hits 计数。
    """
    # ── 迭代161：Lazy re import — Stage 2-only 函数，不污染 Stage 0+1 冷启动路径 ──
    import re as _re
    if not get("scheduler.ext_enabled", project=project):
        return None

    disk = _load_disk_config()
    rules = disk.get(_SCHED_EXT_KEY, [])
    if not isinstance(rules, list) or not rules:
        return None

    # 按 scope 排序：project-specific > global
    sorted_rules = sorted(
        enumerate(rules),
        key=lambda x: (0 if isinstance(x[1], dict) and x[1].get("scope", "global") == project else 1)
    )

    for idx, rule in sorted_rules:
        if not isinstance(rule, dict):
            continue
        pattern = rule.get("pattern", "")
        priority = rule.get("priority", "").upper()
        scope = rule.get("scope", "global")

        if priority not in ("SKIP", "LITE", "FULL"):
            continue
        if scope != "global" and scope != project:
            continue

        try:
            if _re.search(pattern, query, _re.IGNORECASE):
                # 命中：递增 hits（异步持久化，不阻塞）
                try:
                    rules[idx]["hits"] = rule.get("hits", 0) + 1
                    disk[_SCHED_EXT_KEY] = rules
                    with open(SYSCTL_FILE, 'w', encoding='utf-8') as _f:
                        _f.write(json.dumps(disk, ensure_ascii=False, indent=2))
                    _invalidate_cache()
                except Exception:
                    pass

                return {
                    "priority": priority,
                    "pattern": pattern,
                    "scope": scope,
                    "reason": rule.get("reason", ""),
                    "hits": rule.get("hits", 0) + 1,
                }
        except Exception:
            continue  # 正则执行异常 → 跳过该规则（等价于 BPF panic → fallback）

    return None
