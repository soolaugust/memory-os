"""
store_vfs_schema.py — VFS schema 定义 + 工具函数

从 store_vfs.py 提取（iter493: 文件拆分，解决 context thrashing）。
包含：imports, globals, ensure_schema, _safe_add_column, tokenize helpers, _ensure_fts5

OS 类比：Linux mm/Makefile 模块化 — mm/vmalloc.c / mm/slab.c 各自独立编译。
"""
"""
store_vfs.py — VFS 统一数据访问层（核心 CRUD + FTS5 + evict）

从 store_core.py 拆分（迭代21-64 功能集）。
包含：数据库连接、schema 迁移、FTS5 全文索引、CRUD 操作、去重/合并、淘汰。

OS 类比：Linux VFS (Virtual File System, 1992) + ext3 htree (2002)
"""
import json
import os
import re
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, List
import config

# ── 迭代154：Module-level bm25 imports — 消除 _fts5_escape/_cjk_tokenize 的内联 import 开销 ──
# OS 类比：ELF 动态链接器 GOT/PLT — 符号在模块加载时一次性绑定，调用时直接查表，
#   不像 dlopen(RTLD_LAZY) 每次调用时重新解析（per-call 解析 ≈ 内联 import）。
#
# 问题：_fts5_escape() 每次调用都执行 `from bm25 import ENGLISH_STOPWORDS, _porter_stem`
#   Python 内部实际是 sys.modules 查找 + 属性读取，首次还会触发 bm25 模块初始化。
#   _cjk_tokenize() 同样在 try 块中内联 import（每次都执行 sys.modules 查找）。
#   实测：FTS5 检索路径 _fts5_escape + _cjk_tokenize 合计 ~11.6ms 首次，
#   其中 bm25 import 贡献 ~3ms（bm25 未加载时）或 ~0.1ms（已缓存）。
#
# 修复：将 bm25 符号提升到 store_vfs 模块级 import。
#   bm25 模块本身很轻（纯 Python math + re，无重型依赖），~2ms 加载。
#   模块级 import 只付一次，_fts5_escape/_cjk_tokenize 调用时直接使用全局名，
#   消除 per-call sys.modules 查找 + 属性绑定开销。
try:
    from bm25 import ENGLISH_STOPWORDS as _BM25_STOPWORDS, _porter_stem as _bm25_stem
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False
    _BM25_STOPWORDS = frozenset()
    def _bm25_stem(w): return w

# tmpfs 隔离（迭代54）：环境变量覆盖，测试用临时目录，不污染生产数据
# OS 类比：Linux tmpfs (2000) — /dev/shm 内存文件系统，进程退出自动销毁
MEMORY_OS_DIR = Path(os.environ["MEMORY_OS_DIR"]) if os.environ.get("MEMORY_OS_DIR") else Path.home() / ".claude" / "memory-os"
STORE_DB = Path(os.environ["MEMORY_OS_DB"]) if os.environ.get("MEMORY_OS_DB") else MEMORY_OS_DIR / "store.db"
CHUNK_VERSION_FILE = MEMORY_OS_DIR / ".chunk_version"  # 迭代64: chunk_version for TLB v2

def open_db(db_path: Path = None) -> sqlite3.Connection:
    """
    打开 store.db 连接，统一 PRAGMA 策略。
    OS 类比：VFS 的 mount() — 一处配置挂载选项，所有后续 I/O 继承。

    WAL mode + synchronous=NORMAL：
    - WAL 允许读写并发（reader 不阻塞 writer）
    - NORMAL 在断电时最多丢失最后一个未 checkpoint 的 WAL 帧
    """
    if db_path is None:
        db_path = STORE_DB
    MEMORY_OS_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    # 迭代112: busy_timeout — WAL 写锁竞争时自动等待而非立即 SQLITE_BUSY
    # OS 类比：Linux futex FUTEX_WAIT_BITSET — 而非 FUTEX_TRYLOCK 立即失败
    # 根因：writer(async:false) + retriever(async:false) 并发时写锁竞争 → P95=266ms
    # 修复：让持有锁的连接快速释放时，等待方自动重试（最多等 150ms）
    conn.execute("PRAGMA busy_timeout=150")
    return conn

def ensure_schema(conn: sqlite3.Connection) -> None:
    """
    幂等 schema 迁移 — 一处定义，所有 hook 共用。
    OS 类比：VFS 的 super_operations.fill_super() — 挂载时检查并升级 on-disk format。

    策略：CREATE TABLE IF NOT EXISTS + ALTER TABLE ADD COLUMN（忽略已存在错误）。
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_chunks (
            id TEXT PRIMARY KEY,
            created_at TEXT,
            updated_at TEXT,
            project TEXT,
            source_session TEXT,
            chunk_type TEXT,
            content TEXT,
            summary TEXT,
            tags TEXT,
            importance REAL,
            retrievability REAL,
            last_accessed TEXT,
            feishu_url TEXT
        )
    """)
    _safe_add_column(conn, "memory_chunks", "access_count", "INTEGER DEFAULT 0")
    # 迭代38：oom_adj — per-chunk 淘汰优先级（-1000 绝对保护 ↔ +1000 优先淘汰）
    _safe_add_column(conn, "memory_chunks", "oom_adj", "INTEGER DEFAULT 0")
    # 迭代44：lru_gen — MGLRU 多代追踪（0=youngest, max_gen=oldest）
    _safe_add_column(conn, "memory_chunks", "lru_gen", "INTEGER DEFAULT 0")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS recall_traces (
            id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            session_id TEXT NOT NULL,
            project TEXT NOT NULL,
            prompt_hash TEXT NOT NULL,
            candidates_count INTEGER,
            top_k_json TEXT,
            injected INTEGER DEFAULT 0,
            reason TEXT,
            duration_ms REAL DEFAULT 0
        )
    """)
    # iter259: agent_id — 多 Agent 隔离（session_id 前16字符派生）
    _safe_add_column(conn, "recall_traces", "agent_id", "TEXT DEFAULT ''")
    # 迭代65：ftrace_json — 阶段级性能追踪数据（JSON）
    _safe_add_column(conn, "recall_traces", "ftrace_json", "TEXT")

    # ── 迭代29：dmesg 环形缓冲区（OS 类比：/dev/kmsg ring buffer）──
    conn.execute("""
        CREATE TABLE IF NOT EXISTS dmesg (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            level TEXT NOT NULL,
            subsystem TEXT NOT NULL,
            message TEXT NOT NULL,
            session_id TEXT,
            project TEXT,
            extra TEXT
        )
    """)

    # ── 迭代33：swap_chunks 表（OS 类比：Linux swap 分区）──
    conn.execute("""
        CREATE TABLE IF NOT EXISTS swap_chunks (
            id TEXT PRIMARY KEY,
            swapped_at TEXT NOT NULL,
            project TEXT NOT NULL,
            chunk_type TEXT,
            original_importance REAL,
            access_count_at_swap INTEGER DEFAULT 0,
            compressed_data TEXT NOT NULL
        )
    """)

    # ── 迭代100：IPC 共享内存段（OS 类比：System V shm + MESI 缓存一致性）──
    # 跨 Agent 共享知识的核心数据结构
    conn.execute("""
        CREATE TABLE IF NOT EXISTS shm_segments (
            chunk_id TEXT NOT NULL,
            owner_agent TEXT NOT NULL,
            shared_with TEXT NOT NULL DEFAULT '*',
            version INTEGER DEFAULT 1,
            state TEXT DEFAULT 'SHARED',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (chunk_id, owner_agent),
            FOREIGN KEY (chunk_id) REFERENCES memory_chunks(id)
        )
    """)
    # state: MESI 协议 — Modified/Exclusive/Shared/Invalid
    # shared_with: '*' = 全局共享, 逗号分隔 agent_id 列表

    # ── 迭代100：IPC 消息队列（OS 类比：POSIX mq_send/mq_receive）──
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ipc_msgq (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_agent TEXT NOT NULL,
            target_agent TEXT NOT NULL DEFAULT '*',
            msg_type TEXT NOT NULL,
            payload TEXT NOT NULL,
            priority INTEGER DEFAULT 0,
            status TEXT DEFAULT 'QUEUED',
            created_at TEXT NOT NULL,
            consumed_at TEXT,
            ttl_seconds INTEGER DEFAULT 3600
        )
    """)
    # msg_type: knowledge_update | cache_invalidate | task_handoff | heartbeat
    # status: QUEUED → DELIVERED → CONSUMED | EXPIRED

    # ── 迭代100：置信度追踪（OS 类比：ECC — Error Correcting Code）──
    _safe_add_column(conn, "memory_chunks", "confidence_score", "REAL DEFAULT 0.7")
    _safe_add_column(conn, "memory_chunks", "evidence_chain", "TEXT")
    _safe_add_column(conn, "memory_chunks", "verification_status", "TEXT DEFAULT 'pending'")
    # recall_traces 反馈列
    _safe_add_column(conn, "recall_traces", "user_feedback", "TEXT")
    _safe_add_column(conn, "recall_traces", "feedback_ts", "TEXT")

    # ── 迭代104：chunk_pins — 项目级 pin（OS 类比：VMA per-process mlock）──
    # 同一 chunk 在不同 project 中有独立的 pin 状态：
    #   pinned project → kswapd/damon/stale_reclaim 跳过淘汰该 chunk
    #   未 pin project → 正常淘汰评分
    # 类比：Linux MAP_LOCKED 语义仅对调用 mlock() 的进程有效，
    #        同一物理页在其他进程的 VMA 中仍可被 swap out。
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunk_pins (
            chunk_id  TEXT NOT NULL,
            project   TEXT NOT NULL,
            pin_type  TEXT NOT NULL DEFAULT 'soft',
            pinned_at TEXT NOT NULL,
            PRIMARY KEY (chunk_id, project)
        )
    """)
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunk_pins_project ON chunk_pins(project)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunk_pins_chunk ON chunk_pins(chunk_id)")
    except Exception:
        pass

    # ── 迭代148：Missing Indexes — B-tree 索引补全 ──────────────────────
    # OS 类比：Linux ext3 htree (2002, Daniel Phillips) — 目录条目 B-tree 索引
    #   没有 htree 时，大目录的文件查找是 O(N) 线性扫描 inode；
    #   htree 将目录条目组织成 B-tree，使 readdir/lookup 从 O(N) 降到 O(log N)。
    #
    # memory-os 问题：
    #   memory_chunks 主表缺少所有业务索引，而 project = ? 在全库出现 67 次。
    #   kswapd/DAMON/PSI/balloon/stale_reclaim/autotune 等所有子系统都以 project
    #   为最高频过滤条件，但每次查询都触发全表扫描。
    #   SQLite 对无索引的 WHERE project = ? 的时间复杂度是 O(N)（N=chunk 总数）。
    #
    # 索引设计（复合索引，最高选择性列在左）：
    #   1. (project)                     — 基础过滤（所有子系统共用）
    #   2. (project, chunk_type)          — get_chunks/compact_zone/find_similar
    #   3. (project, importance DESC)     — evict_lowest_retention（按重要性淘汰）
    #   4. (project, last_accessed)       — stale_reclaim/DAMON COLD/DEAD 分类
    #   5. recall_traces (project, timestamp) — AIMD/PSI/autotune/GC 时间窗口查询
    #   6. recall_traces (project, injected)  — hit_rate 统计（高频 COUNT）
    #   7. swap_chunks (project)          — gc_orphan_swap/balloon/kswapd 水位
    #
    # 注意：SQLite 对于 "importance < ?" 这种范围查询，
    #       idx_mc_project_importance 让优化器只扫描匹配 project 的子集（B-tree 层级裁剪）。
    try:
        # memory_chunks 核心索引
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mc_project ON memory_chunks(project)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mc_project_type ON memory_chunks(project, chunk_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mc_project_importance ON memory_chunks(project, importance DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mc_project_last_accessed ON memory_chunks(project, last_accessed)")
        # 迭代149：summary 索引 — already_exists/find_similar 精确去重加速
        # OS 类比：Linux VFS inode 哈希表 (inode_hashtable, 1992) —
        #   VFS 通过 (sb, ino) 哈希快速定位 inode 而非线性遍历 inode cache。
        #   already_exists 按 summary 精确匹配：O(N) 全表扫描 → O(log N) B-tree。
        #   already_exists 每次 chunk 写入前必调用，是写路径的最高频查询之一。
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mc_summary ON memory_chunks(summary)")
        # recall_traces 索引（AIMD/PSI/autotune 全都查这张表）
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rt_project_ts ON recall_traces(project, timestamp DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rt_project_injected ON recall_traces(project, injected)")
        # swap_chunks 索引（gc_orphan_swap/kswapd 水位）
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sc_project ON swap_chunks(project)")
    except Exception:
        pass

    # ── 迭代300：info_class — 五层路由（扩展自三层）──
    # OS 类比：Linux 进程地址空间分区（text/data/bss/stack/heap）——
    #   不同区域有不同的保护属性和 eviction 策略。
    #
    # 迭代319：扩展为五类（认知科学双记忆系统，Tulving 1972）：
    #   semantic  : 经多次验证的通用规律（语义记忆）——高 stability，慢衰减，优先保留
    #   episodic  : 特定会话的具体事件（情节记忆）——低 stability，快衰减，可转化
    #   world     : 关于外部世界的事实（原三层，默认，中等保留策略）
    #   operational: agent 操作配置（中等价值，项目内持久）
    #   ephemeral : 临时会话状态（低价值，优先驱逐）
    #
    # semantic vs world 区别：semantic 是"多次验证后提升的知识"，有明确的转化来源；
    #   world 是写入时就被判定为通用知识，没有经过验证路径。
    _safe_add_column(conn, "memory_chunks", "info_class", "TEXT DEFAULT 'world'")

    # ── 迭代319：episodic_consolidations — 情节→语义转化记录 ──────────────────
    # OS 类比：Linux huge page compaction (THP) — 连续小页面合并为大页面，
    #   元数据记录哪些小页面被合并（类比：哪些 episodic chunk 触发了语义化）。
    # 每条记录 = 一次 episodic→semantic 转化事件。
    conn.execute("""
        CREATE TABLE IF NOT EXISTS episodic_consolidations (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            semantic_chunk_id  TEXT NOT NULL,
            source_chunk_ids   TEXT NOT NULL,  -- JSON array of episodic chunk IDs
            project      TEXT NOT NULL,
            trigger_count INTEGER DEFAULT 0,    -- 触发转化时的召回次数
            created_at   TEXT NOT NULL
        )
    """)
    try:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ec_project ON episodic_consolidations(project)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ec_semantic ON episodic_consolidations(semantic_chunk_id)"
        )
    except Exception:
        pass

    # ── 迭代301：stability — Ebbinghaus 记忆稳定性（单位：天）──
    # OS 类比：Linux MGLRU lru_gen — 代龄越高越不活跃；
    #   但 stability 是反向的：越高越稳固，越难被 evict。
    #   初始值 = importance * 2.0；每次检索命中 *= 2.0（间隔重复加固）
    #   eviction_score = age_days / stability（越小越不驱逐）
    _safe_add_column(conn, "memory_chunks", "stability", "REAL DEFAULT 1.0")

    # ── iter399：emotional_weight — 写入时情绪显著性权重（McGaugh 2000）──
    # OS 类比：Linux mempolicy MPOL_PREFERRED_MANY — 写入时标注页面的"情感节点亲和性"，
    #   检索时 retriever 用此权重决定 boost 量（类比 NUMA locality hint）。
    # emotional_weight ∈ [0.0, 1.0]：0=情感中性，1=极高唤醒（崩溃/critical 类词）
    _safe_add_column(conn, "memory_chunks", "emotional_weight", "REAL DEFAULT 0.0")

    # ── iter424：emotional_valence — 情绪效价（Bower 1981 Mood-Congruent Memory）──
    # OS 类比：Linux NUMA node distance matrix — 每个 page 有 home node（positive/negative），
    #   访问时按 node distance 决定延迟（同 node = 低延迟 = valence 一致 = 检索优势）。
    # emotional_valence ∈ [-1.0, +1.0]：
    #   +1 = 正面情绪（突破/发现/成功），-1 = 负面情绪（错误/崩溃/失败），0 = 中性
    # Mood-Congruent Memory：query 情绪效价与 chunk 情绪效价一致时检索加分。
    _safe_add_column(conn, "memory_chunks", "emotional_valence", "REAL DEFAULT 0.0")

    # ── iter401：depth_of_processing — 加工深度（Craik & Lockhart 1972）──
    # OS 类比：Linux page writeback dirty throttle — 页面在 dirty state 停留时间越长，
    #   获得的 write aggregation 越充分，落盘后数据更完整（类比深度加工 → 更稳固的记忆痕迹）。
    # 认知科学依据：加工层次理论 — 语义加工（深处理）比音韵/字形加工（浅处理）
    #   形成更持久的记忆痕迹，因为语义处理触发了更多的关联激活。
    # depth_of_processing ∈ [0.0, 1.0]：
    #   0.0 = 浅处理（简单陈述，无推理/因果/结构）
    #   1.0 = 深处理（丰富的因果推理、结构化分析、多概念关联）
    # 影响：写入时 stability += depth_bonus（深处理的 chunk 初始稳定性更高）
    _safe_add_column(conn, "memory_chunks", "depth_of_processing", "REAL DEFAULT 0.5")

    # ── iter400：chunk_type_decay — 个体化遗忘速率（Ebbinghaus 1885 + 记忆类型差异）──
    # OS 类比：Linux cgroup memory.reclaim_ratio — 不同 cgroup 有不同的内存回收速率，
    #   而非全局统一的 vm.swappiness。
    # 认知科学依据：Squire (1992) / Tulving (1972) 记忆类型理论：
    #   程序性记忆（如技能/约束）比情节记忆（如任务状态）衰减更慢。
    #   design_constraint/decision → 衰减极慢（类比肌肉记忆）
    #   task_state/reasoning_chain → 衰减较快（类比工作记忆/情节记忆）
    # 字段：存储在 sysctl / 配置层，每次 idle_consolidation 查询该表确定 per-type 衰减率。
    # （该 iter 不新增 DB 列，而是影响算法行为）

    # ── iter396：source_type / source_reliability — 信源监控（Johnson 1993）──
    # OS 类比：Linux LSM (Linux Security Modules) — 每次文件访问/进程创建前，
    #   LSM hook 检查来源的"域"（SELinux context / AppArmor label），
    #   来源不同 → 不同信任级别 → 不同访问权限。
    # source_type ∈ {direct, tool_output, inferred, hearsay, unknown}：
    #   direct      = 用户直接陈述/观察（第一手信源，最高可信度）
    #   tool_output = 代码运行/命令输出（机器生成，高可重复性，取决于工具可靠性）
    #   inferred    = 从多条信息推断（合理推断，中等可信度）
    #   hearsay     = 间接转述/用户描述他人说的（可信度最低）
    #   unknown     = 来源不明（默认值）
    # source_reliability ∈ [0.0, 1.0]：
    #   写入时由 compute_source_reliability() 估算；
    #   检索时作为 retrieval_score 的加权因子（source_monitor_weight()）。
    _safe_add_column(conn, "memory_chunks", "source_type", "TEXT DEFAULT 'unknown'")
    _safe_add_column(conn, "memory_chunks", "source_reliability", "REAL DEFAULT 0.7")

    # ── iter403：encode_context — 编码时上下文关键词（Tulving 1974）──
    # OS 类比：Linux NUMA-aware memory allocation — 进程倾向从本地 node 取页；
    #   编码时 context = home node；检索时 context 越接近 = NUMA距离越小 = 命中率越高。
    # encode_context TEXT：编码时的上下文关键词集合（逗号分隔的 token 列表）。
    # 写入时从 content + summary + tags + chunk_type 中提取关键词集。
    # 检索时计算 context overlap（Jaccard）→ 调整检索分（context cue boost）。
    _safe_add_column(conn, "memory_chunks", "encode_context", "TEXT DEFAULT ''")

    # ── 迭代306：raw_snippet — 写入时保真原始片段（≤500字）──
    # OS 类比：Linux page cache 保存原始 disk block，VFS 层面不压缩；
    #   读取时 on-demand 合并（类比 copy-on-read 模式）。
    # raw_snippet 不参与 FTS5 索引（避免膨胀），仅在 retriever 注入时按需附加。
    _safe_add_column(conn, "memory_chunks", "raw_snippet", "TEXT DEFAULT ''")

    # ── 迭代315：encoding_context — 情境感知注入（Encoding Specificity）──
    # OS 类比：Linux perf_event context — 记录性能事件时附带 CPU/task 上下文。
    # 存储 chunk 写入时的情境特征 JSON，检索时与 query_context 比对计算匹配度。
    _safe_add_column(conn, "memory_chunks", "encoding_context", "TEXT DEFAULT '{}'")

    # ── iter415: original_ec_count — encode_context 初始 token 数（Encoding Variability）──
    # 存储 chunk 写入时的 encode_context token 数量，用于检测后续多情境富化。
    # OS 类比：page 首次 mapped-in 的引用计数基线；后续跨进程引用增量代表多情境共享。
    _safe_add_column(conn, "memory_chunks", "original_ec_count", "INTEGER DEFAULT 0")

    # ── iter420: spaced_access_count — 间隔访问计数（Spacing Effect）──
    # 认知科学依据：Ebbinghaus (1885) Spacing Effect / Cepeda et al. (2006) Review —
    #   分布在多个间隔时间段的练习（spaced practice）比集中练习（massed practice）
    #   产生更强的长时记忆保留（间隔效应）。
    # 存储 chunk 被"间隔访问"的次数（gap >= medium_gap_hours = 24h，代表新的"学习会话"）。
    # 每次 update_accessed 时如果访问间隔 >= 24h，则递增此计数。
    # spacing_factor = spaced_access_count / max(1, access_count) ∈ [0,1]：
    #   1.0 = 完全分布式（每次都有足够间隔），0.0 = 全部集中（massed）
    # OS 类比：Linux MGLRU cross-generation promotion —
    #   跨 aging cycle 被访问的 page（distributed access）比在同一 gen 内被访问的
    #   page（massed access）更快晋升到 younger generation（真正的热页）。
    _safe_add_column(conn, "memory_chunks", "spaced_access_count", "INTEGER DEFAULT 0")

    # ── iter437: hypermnesia_last_boost — 上次 Hypermnesia boost 时间（冷却期追踪）──
    # OS 类比：khugepaged scan_sleep_millisecs — 两次 hugepage 合并之间的最小休眠间隔，
    #   防止 khugepaged 频繁唤醒消耗 CPU（hypermnesia cooldown 防止反复触发）。
    _safe_add_column(conn, "memory_chunks", "hypermnesia_last_boost", "TEXT")

    # ── iter456: access_source — 检索来源标记（RPCA：主动检索 vs 被动重读）──
    # 认知科学依据：Roediger & Karpicke (2006) Retrieval Practice vs. Restudy —
    #   主动检索（retrieval）产生的记忆巩固效益比被动重读（restudy）高约 50%。
    # access_source ∈ {'retrieval', 'restudy'}：
    #   'retrieval' = 用户 query 主动命中（默认，通过 FTS5/BM25 检索召回）
    #   'restudy'   = 被动曝光（loader注入、preload 等非主动检索路径）
    # OS 类比：Linux page fault type — demand paging（retrieval，主动缺页）vs
    #   prefetch/readahead（restudy，内核预读，未被 CPU 实际访问确认）。
    _safe_add_column(conn, "memory_chunks", "access_source", "TEXT DEFAULT 'retrieval'")

    # ── Task13：row_version — Optimistic Locking（CAS）──
    # OS 类比：Linux seqlock / atomic_cmpxchg — 读取 sequence number 后写入时验证未变化。
    # 多 agent 并发写：每次 update 递增 row_version，CAS 检查版本防止 ABA 问题。
    _safe_add_column(conn, "memory_chunks", "row_version", "INTEGER DEFAULT 1")

    # ── Task12：chunk_state — Lifecycle FSM（ACTIVE/COLD/DEAD/SWAP/GHOST）──
    # OS 类比：Linux page state machine — PG_active/PG_referenced/PG_lru/PG_swapcache
    #   ACTIVE  = PG_active + PG_referenced   — 最近访问，热数据
    #   COLD    = PG_lru (inactive list)       — 可回收候选，7-30天无访问
    #   DEAD    = DAMON DEAD region            — 极少访问，可 swap_out 或 evict
    #   SWAP    = PG_swapcache                 — 已 swap_out，在 swap_chunks
    #   GHOST   = 待 GC，evict 前短暂标记态
    _safe_add_column(conn, "memory_chunks", "chunk_state", "TEXT DEFAULT 'ACTIVE'")
    try:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mc_state ON memory_chunks(chunk_state)"
        )
    except Exception:
        pass

    # ── iter428：boundary_proximity — 事件边界距离（Zacks et al. 2007 Event Segmentation）──
    # OS 类比：ext4 jbd2 journal commit boundary —
    #   刚越过 journal commit 的 page（新 epoch 首批写入）稳定性最高；
    #   commit 前的 dirty page（旧 epoch 末尾）处于"不稳定窗口"（doorway effect）。
    # boundary_proximity ∈ [-1.0, +1.0]：
    #   +1.0 = 本 session 刚开始时写入（刚越过 session boundary → encoding boost）
    #    0.0 = 中性（会话中间写入，无边界效应）
    #   -1.0 = 上一 session 末尾写入（doorway effect → 短暂 retrieval penalty）
    _safe_add_column(conn, "memory_chunks", "boundary_proximity", "REAL DEFAULT 0.0")
    _safe_add_column(conn, "memory_chunks", "session_type_history", "TEXT DEFAULT ''")  # iter459 CIE

    # ── iter461: HAC — chunk_coactivation 表（Hebbian 共激活次数追踪）──
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunk_coactivation (
            chunk_a     TEXT NOT NULL,
            chunk_b     TEXT NOT NULL,
            project     TEXT NOT NULL,
            coact_count INTEGER DEFAULT 1,
            last_coact  TEXT,
            PRIMARY KEY (chunk_a, chunk_b, project)
        )
    """)
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_coact_a ON chunk_coactivation(chunk_a, project)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_coact_b ON chunk_coactivation(chunk_b, project)")
    except Exception:
        pass

    # ── 迭代317：knowledge_versions — 前摄干扰控制（Proactive Interference）──
    # OS 类比：Linux kernel module versioning — 加载新模块版本时标记旧版本为
    #   MODULE_STATE_GOING，确保旧版本不再被新请求调用。
    # Bartlett 1932 图式同化：新知识依附已有框架，框架更新时必须明确标记旧框架失效。
    # 每条记录 = 一次知识演化事件：old_chunk 被 new_chunk 取代。
    conn.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_versions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            old_chunk_id TEXT NOT NULL,
            new_chunk_id TEXT NOT NULL,
            reason      TEXT,
            project     TEXT,
            session_id  TEXT,
            created_at  TEXT NOT NULL
        )
    """)
    try:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_kv_old ON knowledge_versions(old_chunk_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_kv_project ON knowledge_versions(project)"
        )
    except Exception:
        pass

    # ── 迭代304：entity_edges — 知识图谱关系边（OS 类比：Linux 内核模块依赖图）──
    # 每条边 = (from_entity) --[relation]--> (to_entity)，
    # 类比内核 module_kobject 依赖表：kmod 加载前检查依赖链，
    # 边缺失 → 加载失败（知识断链 → 检索回答残缺）。
    #
    # relation 类型：
    #   uses        — X 使用/采用/基于 Y
    #   depends_on  — X 依赖/需要 Y
    #   part_of     — X 是 Y 的一部分/子模块
    #   implements  — X 实现了 Y
    #   related_to  — 其他关联
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entity_edges (
            id TEXT PRIMARY KEY,
            from_entity TEXT NOT NULL,
            relation TEXT NOT NULL,
            to_entity TEXT NOT NULL,
            project TEXT,
            source_chunk_id TEXT,
            confidence REAL DEFAULT 0.7,
            created_at TEXT NOT NULL
        )
    """)
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ee_from ON entity_edges(from_entity)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ee_to ON entity_edges(to_entity)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ee_project ON entity_edges(project)")
    except Exception:
        pass

    # ── 迭代310：entity_map — chunk_id ↔ entity_name 映射（Spreading Activation 锚点）──
    # OS 类比：Linux /proc/modules 中每个 module 的 kobject 指针 —
    #   entity_map 是 chunk 到 entity 的"地址翻译表"，
    #   spreading activation 通过它从 FTS5 命中的 chunk 找到对应 entity，
    #   再沿 entity_edges 扩散邻居，类比 TLB walk（chunk→entity→邻居entity→邻居chunk）。
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entity_map (
            entity_name TEXT NOT NULL,
            chunk_id    TEXT NOT NULL,
            project     TEXT NOT NULL DEFAULT '',
            updated_at  TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (entity_name, project)
        )
    """)
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_em_chunk ON entity_map(chunk_id)")
    except Exception:
        pass

    # ── iter404：priming_state — 语义启动状态表（Collins & Loftus 1975）──
    # OS 类比：Linux page readahead / ra_state —
    #   访问一个 page 触发相邻 pages 预取进 page cache（readahead window）；
    #   类似地，检索一个 chunk 时，相关 entity 被"启动"（primed），
    #   后续短时间内（prime_half_life ~ 30min）相关 chunk 检索分提升。
    # prime_strength ∈ [0.0, 1.0]：当前启动强度（随时间指数衰减）
    conn.execute("""
        CREATE TABLE IF NOT EXISTS priming_state (
            entity_name TEXT NOT NULL,
            project     TEXT NOT NULL DEFAULT '',
            primed_at   TEXT NOT NULL,
            prime_strength REAL NOT NULL DEFAULT 1.0,
            PRIMARY KEY (entity_name, project)
        )
    """)
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_prime_project ON priming_state(project, primed_at)")
    except Exception:
        pass

    # ── 迭代305：curiosity_queue — 知识空白探索队列（OS 类比：kswapd 水位触发）──
    # 当 retriever 检测到「弱命中」（FTS 有结果但 top-1 分数 < WMARK_LOW=0.25）时，
    # 说明 DB 里「有相关内容但不够用」——把 query 写入此队列。
    # deep-sleep 阶段消费队列，主动补充知识。
    #
    # OS 类比：Linux /proc/sys/vm/watermark_scale_factor +
    #   kswapd shrink_node() — 检测到 free pages < WMARK_LOW 时异步回收：
    #     WMARK_LOW（0.25）: FTS5 top-1 score 低于此值 → 判定为「知识低水位」
    #     kswapd（deep-sleep consumer）: 消费 curiosity_queue，主动填充知识空白
    #     status 生命周期: pending → processing → filled/dismissed
    #       等价于: 页面回收任务 → kswapd 领取 → 完成 swap-in 或 discard
    conn.execute("""
        CREATE TABLE IF NOT EXISTS curiosity_queue (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            query       TEXT NOT NULL,
            project     TEXT NOT NULL,
            detected_at TEXT NOT NULL,
            top_score   REAL,
            status      TEXT DEFAULT 'pending',
            filled_at   TEXT,
            chunk_id    TEXT
        )
    """)
    # 索引设计：
    #   (project, status) — pop_curiosity_queue 按 project+pending 过滤（最高频路径）
    #   (project, query)  — enqueue_curiosity 幂等检查（7天内同 query）
    try:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cq_project_status "
            "ON curiosity_queue(project, status)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cq_project_query_time "
            "ON curiosity_queue(project, query, detected_at)"
        )
    except Exception:
        pass

    # ── iter380：schema_anchors — Bartlett (1932) Schema Theory ─────────────
    # OS 类比：Linux SLUB Allocator kmem_cache — 相似对象共享结构模板（kmem_cache），
    #   新对象写入时自动归属对应 cache；检索时 cache 整体激活，批量命中。
    #
    # schema_anchors 记录 chunk → schema 的绑定关系：
    #   chunk 写入时，扫描 summary 匹配预定义 schema 规则 → 写入绑定行
    #   retriever 命中 chunk 后，查 schema_anchors → 激活同 schema 的其他 chunk
    #   类比：kmem_cache 命中后，同 cache 的相邻 slab 自动预热到 L2
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_anchors (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id    TEXT NOT NULL,
            schema_name TEXT NOT NULL,
            project     TEXT NOT NULL,
            confidence  REAL DEFAULT 0.8,
            created_at  TEXT NOT NULL,
            UNIQUE(chunk_id, schema_name)
        )
    """)
    try:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sa_schema_project "
            "ON schema_anchors(schema_name, project)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sa_chunk "
            "ON schema_anchors(chunk_id)"
        )
    except Exception:
        pass

    # ── 迭代23：FTS5 全文索引（OS 类比：ext3 htree）──
    # content-sync 模式：FTS5 表引用主表数据，通过触发器保持同步
    # 搜索 summary + content 两个字段
    _ensure_fts5(conn)

