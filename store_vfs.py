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

def _safe_add_column(conn: sqlite3.Connection, table: str, column: str, col_type: str) -> None:
    """幂等加列：已存在则静默跳过。"""
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
    except Exception:
        pass

def _normalize_structured_summary(text: str) -> str:
    """
    iter108：结构化 summary 归一化 — 在 FTS5 索引前展开标签/路径分隔符。

    目标：把 "[capabilities] 锁/并发分析协议 > 迁移到新项目/子系统时需确认"
    转为  "capabilities 锁 并发分析协议 迁移到新项目 子系统时需确认"
    让 FTS5 能按语义词命中，而不是被 []、>、/ 等 ASCII 符号截断。

    处理规则：
      - [topic] 前缀 → 去掉括号，保留内容
      - [规则/Category] 前缀 → 同上
      - X > Y → "X Y"（去掉 >）
      - X/Y 路径分隔 → "X Y"（仅非文件路径的 /）
      - (xxx) / （xxx） → 保留内容去括号
    OS 类比：search indexer 对结构化字段做 field normalization 再建倒排索引。
    """
    if not text:
        return text
    # 去掉方括号，保留内容
    result = re.sub(r'\[([^\]]{1,40})\]', r' \1 ', text)
    # > 分隔符 → 空格
    result = result.replace('>', ' ')
    # / 在非文件路径上下文中 → 空格（文件路径含 . 不替换）
    result = re.sub(r'(?<![.\w])/(?![.\w])', ' ', result)
    # 全角括号
    result = result.replace('（', ' ').replace('）', ' ')
    # 多余空白合并
    result = re.sub(r'\s+', ' ', result).strip()
    return result


def _cjk_tokenize(text: str) -> str:
    """
    迭代97：CJK 单字分词预处理。
    在每个 CJK 字符前后插入空格，让 unicode61 tokenizer 按单字分词。

    迭代100：同时追加 CJK bigram（相邻字对）到末尾，提升精确短语匹配精度。

    迭代99：英文部分追加 stemmed 形式，使 FTS5 索引能匹配查询侧的 stemmed token。
    例如 "analyzing" 额外追加 "analyz"，查询 "analysis" stem→"analys" 时更接近命中。

    OS 类比：inverted index 中同时存 unigram + bigram posting list — 单字用于召回，
    bigram 用于精排（phrase match score 更高）。
    """
    if not text:
        return ''
    # 单字：每个 CJK 字前后加空格
    result = re.sub(r'([\u4e00-\u9fff\u3400-\u4dbf])', r' \1 ', text)
    # bigram：提取连续 CJK 字对，追加到末尾
    cjk_chars = re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]', text)
    bigrams = [cjk_chars[i] + cjk_chars[i+1] for i in range(len(cjk_chars) - 1)]
    if bigrams:
        result = result + ' ' + ' '.join(bigrams)
    # 迭代99：英文 stemming — 追加 stemmed 形式到文本末尾
    # 迭代154：使用模块级 _BM25_STOPWORDS / _bm25_stem，消除 per-call inline import
    if _BM25_AVAILABLE:
        eng_words = re.findall(r'[a-zA-Z]{3,}', text)
        stemmed_extra = []
        for w in eng_words:
            low = w.lower()
            if low not in _BM25_STOPWORDS:
                s = _bm25_stem(low)
                if s != low:
                    stemmed_extra.append(s)
        if stemmed_extra:
            result = result + ' ' + ' '.join(set(stemmed_extra))
    return result


def _ensure_fts5(conn: sqlite3.Connection) -> None:
    """
    迭代23：幂等创建 FTS5 虚拟表。
    OS 类比：ext3 的 htree 是在 mkfs/tune2fs 时创建索引结构。

    迭代97：改为非 content-sync 模式（独立存储 CJK 预处理文本）。
    - 旧 content-sync 模式：FTS5 直接引用主表原始文本，CJK 单字查询无效
    - 新独立模式：FTS5 存储经 _cjk_tokenize 处理的文本，支持单字精准匹配
    - insert_chunk / delete_chunks 中手动维护 FTS 索引（不依赖触发器）

    迁移：检测旧 content-sync 表并自动迁移为新格式。
    """
    # 检测 FTS5 表是否存在 + 是否为旧 content-sync 格式
    exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_chunks_fts'"
    ).fetchone()

    if exists:
        # 迭代97：检测是否为旧 content-sync 格式（有触发器 = 旧版）
        old_trigger = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND name='memory_chunks_ai'"
        ).fetchone()
        if old_trigger:
            # 旧 content-sync 格式：迁移为新独立格式
            # 1. 删除旧触发器
            for t in ('memory_chunks_ai', 'memory_chunks_ad', 'memory_chunks_au'):
                conn.execute(f"DROP TRIGGER IF EXISTS {t}")
            # 2. 删除旧 FTS5 表
            conn.execute("DROP TABLE IF EXISTS memory_chunks_fts")
            # 3. 重新创建（见下方）
        else:
            # 迭代100：检测是否为旧单字格式（需升级为 bigram 格式）
            # 标志：fts_schema_version 表存在且 version < 100
            ver_row = conn.execute(
                "SELECT version FROM fts_schema_version WHERE name='memory_chunks_fts'"
            ).fetchone() if conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='fts_schema_version'"
            ).fetchone() else None
            if ver_row is None or ver_row[0] < 100:
                # 迭代100 迁移：删除旧 FTS5，重建含 bigram 的新格式
                conn.execute("DROP TABLE IF EXISTS memory_chunks_fts")
                # 继续执行下方创建逻辑
            elif ver_row[0] < 124:
                # 迭代124：检测 UUID rowid_ref 污染（生产 bug）
                # 根因：历史版本的 insert_chunk 在某个代码版本中误写入 UUID 字符串
                # 而非 str(integer_rowid)，导致 fts_search JOIN CAST(rowid_ref AS INTEGER)
                # 始终返回 0/garbage，所有 FTS5 查询失效，系统静默降级到 BM25 全表扫描。
                # 检测方式：取一条 rowid_ref，如果包含 '-' 则是 UUID，需要重建。
                # OS 类比：fsck inode corruption check — 检测元数据损坏并触发磁盘重建。
                sample = conn.execute(
                    "SELECT rowid_ref FROM memory_chunks_fts LIMIT 1"
                ).fetchone()
                is_uuid_corrupted = sample and '-' in str(sample[0])
                if is_uuid_corrupted:
                    # UUID 污染：清空 FTS5 并全量重建（DROP 虚拟表会有问题，用 DELETE 代替）
                    conn.execute("DELETE FROM memory_chunks_fts")
                    # 继续执行下方重建逻辑（表已存在，跳过 CREATE VIRTUAL TABLE）
                    _fts_needs_rebuild = True
                else:
                    # version 100 且无 UUID 污染：仅升级 version 到 124
                    conn.execute(
                        "INSERT OR REPLACE INTO fts_schema_version (name, version) VALUES ('memory_chunks_fts', 124)"
                    )
                    return
            else:
                # 已经是迭代124 格式，无需操作
                return

    # 创建新 FTS5 虚拟表（独立模式，存储 CJK 预处理后的文本）
    # OS 类比：ext4 的 htree 独立 B-tree，不引用 inode 原始数据
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS memory_chunks_fts USING fts5(
            rowid_ref UNINDEXED,
            summary,
            content
        )
    """)

    # 全量重建：从主表读取，经 CJK 预处理后写入 FTS5
    # OS 类比：fsck 重建 htree — 扫描所有 inode，重写 B-tree 索引
    # 迭代124：统一使用 str(rowid)（整数转字符串），确保 fts_search JOIN CAST 可正确还原。
    # 历史 bug：旧版本写入了 UUID 字符串（chunk.id）而非 str(rowid)，导致 CAST→0/garbage。
    rows = conn.execute(
        "SELECT rowid, summary, content FROM memory_chunks WHERE summary != ''"
    ).fetchall()
    for rowid, summary, content in rows:
        conn.execute(
            "INSERT INTO memory_chunks_fts(rowid_ref, summary, content) VALUES (?, ?, ?)",
            (str(rowid), _cjk_tokenize(_normalize_structured_summary(summary or '')),
             _cjk_tokenize(_normalize_structured_summary(content or '')))
        )

    # 迭代124：记录 FTS schema 版本（124 = 修复 UUID 污染后的干净重建版本）
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fts_schema_version (
            name TEXT PRIMARY KEY,
            version INTEGER NOT NULL
        )
    """)
    conn.execute(
        "INSERT OR REPLACE INTO fts_schema_version (name, version) VALUES ('memory_chunks_fts', 124)"
    )

    # ── 迭代87：Scheduler Tables（OS 类比：CFS runqueue + task_struct）──
    # 迭代87：任务调度队列（task_struct）
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scheduler_tasks (
            id TEXT PRIMARY KEY,
            project TEXT NOT NULL,
            session_id TEXT NOT NULL,
            task_name TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            priority INTEGER DEFAULT 0,
            created_at TEXT,
            updated_at TEXT,
            due_at TEXT,
            dependencies TEXT,
            execution_log TEXT,
            swap_context TEXT,
            oom_adj INTEGER DEFAULT -800
        )
    """)
    # 迭代87：任务-决策关联表（task 依赖的核心 decision）
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scheduler_task_decisions (
            decision_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            decision_type TEXT,
            FOREIGN KEY (decision_id) REFERENCES memory_chunks(id),
            FOREIGN KEY (task_id) REFERENCES scheduler_tasks(id),
            PRIMARY KEY (decision_id, task_id)
        )
    """)
    # ── 迭代99：Hook 事务日志（OS 类比：ext4 journal — 崩溃恢复的 WAL）──
    # 记录每次 Stop hook 的事务状态，支持崩溃后诊断部分写入
    conn.execute("""
        CREATE TABLE IF NOT EXISTS hook_txn_log (
            txn_id       TEXT PRIMARY KEY,
            hook         TEXT NOT NULL DEFAULT 'extractor',
            status       TEXT NOT NULL DEFAULT 'pending',
            chunk_count  INTEGER DEFAULT 0,
            session_id   TEXT NOT NULL DEFAULT '',
            project      TEXT NOT NULL DEFAULT '',
            started_at   TEXT NOT NULL,
            committed_at TEXT,
            error        TEXT
        )
    """)
    # iter259: agent_id 维度
    _safe_add_column(conn, "hook_txn_log", "agent_id", "TEXT DEFAULT ''")

    # ── iter259：session_intents 表 — 替代单文件 session_intent.json（并发安全）──
    # 多 Agent 场景下，session_intent.json 是单文件，最后写者覆盖之前写者。
    # 改为 DB 表，每个 session_id 独立一行，互不干扰。
    # OS 类比：per-process /proc/PID/status，而不是全局单文件 /proc/intent
    conn.execute("""
        CREATE TABLE IF NOT EXISTS session_intents (
            session_id   TEXT PRIMARY KEY,
            project      TEXT NOT NULL DEFAULT '',
            agent_id     TEXT NOT NULL DEFAULT '',
            saved_at     TEXT NOT NULL,
            intent_json  TEXT NOT NULL DEFAULT '{}',
            pinned_chunk_ids TEXT NOT NULL DEFAULT '[]'
        )
    """)
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_si_project ON session_intents(project)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_si_saved ON session_intents(saved_at DESC)")
    except Exception:
        pass

    # ── iter259：shadow_traces 表 — 替代单文件 .shadow_trace.json（并发安全）──
    # 多 Agent 场景下，.shadow_trace.json 是单文件，并发写入会相互覆盖。
    # OS 类比：per-process page table，而不是共享全局 page table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS shadow_traces (
            session_id   TEXT PRIMARY KEY,
            project      TEXT NOT NULL DEFAULT '',
            agent_id     TEXT NOT NULL DEFAULT '',
            updated_at   TEXT NOT NULL,
            top_k_ids    TEXT NOT NULL DEFAULT '[]'
        )
    """)
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sht_project ON shadow_traces(project)")
    except Exception:
        pass

    # ── iter259：tool_patterns — 工具调用序列学习（OS 类比：perf_event ring buffer）──
    # extractor 写入 tool_patterns，retriever 查询，但之前 ensure_schema 未创建该表。
    # 修复：在此统一创建，防止悬空查询（OperationalError: no such table）。
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tool_patterns (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_hash    TEXT UNIQUE,
            tool_sequence   TEXT NOT NULL,
            context_keywords TEXT DEFAULT '[]',
            frequency       INTEGER DEFAULT 1,
            avg_duration_ms REAL DEFAULT 0,
            success_rate    REAL DEFAULT 1.0,
            first_seen      TEXT,
            last_seen       TEXT,
            project         TEXT
        )
    """)
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tp_project ON tool_patterns(project)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tp_hash ON tool_patterns(pattern_hash)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tp_freq ON tool_patterns(frequency DESC)")
    except Exception:
        pass

    # ── iter390: trigger_conditions — 展望记忆触发条件 ──────────────────────────
    # 认知科学依据：Einstein & McDaniel (1990) Prospective Memory —
    #   意图性记忆：在未来某个时刻执行某个动作的意图（"下次打开 X 时记得..."）。
    #   触发模式：特定信号（context cue）激活相关延迟意图记忆。
    # OS 类比：Linux inotify/fanotify — 注册文件系统事件监听，触发条件满足时唤醒等待进程。
    # trigger_conditions 存储 extractor 检测到的"将来触发"意图，
    # retriever 在匹配到 trigger_pattern 时注入关联 chunk。
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trigger_conditions (
            id          TEXT PRIMARY KEY,
            chunk_id    TEXT NOT NULL,
            project     TEXT NOT NULL,
            session_id  TEXT NOT NULL DEFAULT '',
            trigger_pattern TEXT NOT NULL,
            trigger_type TEXT NOT NULL DEFAULT 'keyword',
            created_at  TEXT NOT NULL,
            fired_count INTEGER DEFAULT 0,
            last_fired  TEXT,
            expires_at  TEXT
        )
    """)
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tc_project ON trigger_conditions(project)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tc_chunk ON trigger_conditions(chunk_id)")
    except Exception:
        pass

    # ── iter259：entity_edges 补充 agent_id 复合唯一约束 ──
    # 多 Agent 场景下同一实体对可被不同 agent 提取，需按 agent 隔离唯一性
    _safe_add_column(conn, "entity_edges", "agent_id", "TEXT DEFAULT ''")

    conn.commit()


# ── iter390: trigger_conditions CRUD ────────────────────────────────────────

def insert_trigger(conn: sqlite3.Connection, trigger: dict) -> None:
    """写入一条 trigger_conditions 记录。"""
    conn.execute("""
        INSERT OR REPLACE INTO trigger_conditions
        (id, chunk_id, project, session_id, trigger_pattern, trigger_type,
         created_at, fired_count, last_fired, expires_at)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """, (
        trigger["id"],
        trigger["chunk_id"],
        trigger["project"],
        trigger.get("session_id", ""),
        trigger["trigger_pattern"],
        trigger.get("trigger_type", "keyword"),
        trigger["created_at"],
        trigger.get("fired_count", 0),
        trigger.get("last_fired"),
        trigger.get("expires_at"),
    ))


def query_triggers(conn: sqlite3.Connection, project: str,
                   query_text: str, max_triggers: int = 3) -> list:
    """
    查询与 query_text 匹配的 trigger_conditions，返回相关 chunk_id 列表。
    OS 类比：inotify_read() — 读取待处理的文件系统事件（触发条件已满足）。

    匹配逻辑：trigger_pattern 是关键词/正则，query_text 中包含时触发。
    返回 [(chunk_id, trigger_id, trigger_pattern), ...] 最多 max_triggers 条。
    """
    import re as _re
    now_iso = datetime.now(timezone.utc).isoformat()
    rows = conn.execute(
        "SELECT id, chunk_id, trigger_pattern, trigger_type FROM trigger_conditions "
        "WHERE project=? AND (expires_at IS NULL OR expires_at > ?) "
        "ORDER BY fired_count ASC, created_at DESC LIMIT 50",
        (project, now_iso),
    ).fetchall()

    matched = []
    for row in rows:
        tid, cid, pattern, ttype = row[0], row[1], row[2], row[3]
        try:
            if ttype == "regex":
                if _re.search(pattern, query_text, _re.IGNORECASE):
                    matched.append((cid, tid, pattern))
            else:
                # keyword: pattern is a simple keyword/phrase
                if pattern.lower() in query_text.lower():
                    matched.append((cid, tid, pattern))
        except Exception:
            continue
        if len(matched) >= max_triggers:
            break
    return matched


def fire_trigger(conn: sqlite3.Connection, trigger_id: str) -> None:
    """记录 trigger 已触发（更新 fired_count + last_fired）。"""
    now_iso = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "UPDATE trigger_conditions SET fired_count=fired_count+1, last_fired=? WHERE id=?",
        (now_iso, trigger_id),
    )


def _fts5_escape(query: str) -> str:
    """
    将自然语言查询转为 FTS5 安全的 MATCH 表达式。

    迭代97：配合新 FTS5 独立模式（CJK 单字分词存储）。
    迭代100：改为 bigram 优先策略，与存储侧 _cjk_tokenize v100 对应。
    迭代103：查询侧同义词扩展（Query Expansion），打击自然语言↔技术术语的语义差距。

    策略：
      - 英文词/数字/标识符：提取词元，OR 连接
      - CJK：优先提取 bigram（相邻字对），少于 2 个 bigram 时补单字
      - 同义词扩展：对匹配到的概念追加技术术语/自然语言等价词

    OS 类比：ext4 htree 查询 + 搜索引擎的 Query Expansion (QE)。
    """
    tokens = []
    seen: set = set()

    # 迭代99：英文词 + Porter stemming + stopword 过滤（与 bm25.py 对称）
    # 迭代154：使用模块级 _BM25_STOPWORDS / _bm25_stem，消除 per-call inline import
    for m in re.finditer(r'[a-zA-Z0-9_][-a-zA-Z0-9_.]*', query):
        token = m.group().lower().strip('.-_')
        if len(token) >= 2 and token not in seen and token not in _BM25_STOPWORDS:
            stemmed = _bm25_stem(token)
            seen.add(token)
            seen.add(stemmed)
            tokens.append(f'"{stemmed}"')

    # CJK：bigram 优先
    cjk_chars = re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]', query)
    bigrams = [cjk_chars[i] + cjk_chars[i+1] for i in range(len(cjk_chars) - 1)]
    if bigrams:
        for bg in bigrams:
            if bg not in seen:
                seen.add(bg)
                tokens.append(f'"{bg}"')
    else:
        # 无 bigram（单个 CJK 字或为空）：退回单字
        for c in cjk_chars:
            if c not in seen:
                seen.add(c)
                tokens.append(f'"{c}"')

    # 迭代103：同义词扩展 — 查询侧 Query Expansion
    # 对已提取的 token 和原始 query 文本做概念匹配，追加等价术语
    syn_tokens = _synonym_expand(query, seen)
    tokens.extend(syn_tokens)

    if not tokens:
        return ""
    return " OR ".join(tokens)


# ── 迭代103：同义词/概念映射表 ──────────────────────────────────────
# OS 类比：DNS CNAME — 一个规范名可以有多个别名指向同一资源。
# 双向映射：技术术语 ↔ 自然语言描述。
# 触发条件：query 中出现左侧任一词/短语时，追加右侧所有词到 FTS5 查询。
#
# 格式：每条规则 = (trigger_patterns, expansion_terms)
#   trigger_patterns: 正则列表，匹配 query 原文
#   expansion_terms: 要追加到 FTS5 查询的术语列表
#
# 迭代153：Pre-compiled Synonym Patterns — AOT 正则预编译
# OS 类比：GCC AOT compilation vs JIT — 在模块加载时预编译所有正则，
#   消除首次 _synonym_expand 调用的 re.compile 延迟（实测 8.8ms → <1ms）。
#   Python re.search("pattern", ...) 在内部维护 512 条 LRU 编译缓存（CPython），
#   229 个触发模式首次调用时全部 cache miss → 每条 ~0.04ms，合计 ~8.8ms。
#   预编译将 229 条 re.compile() 提前到模块 import 时执行（只付一次）。

_SYNONYM_RULES = [
    # 内存管理
    (["自动清理", "自动淘汰", "自动删除", "清理记忆", "清理知识"],
     ["kswapd", "evict", "淘汰"]),
    (["kswapd", "页面回收", "内存回收"],
     ["淘汰", "清理", "evict", "watermark"]),
    # 迭代125：淘汰/驱逐机制更多触发词（原规则只覆盖"自动淘汰"，查询"淘汰策略"无法触发）
    (["淘汰策略", "淘汰机制", "驱逐", "eviction", "回收策略", "缓存驱逐",
      "chunk.*淘汰", "淘汰.*chunk", "何时.*淘汰", "怎么.*淘汰", "如何.*淘汰"],
     ["kswapd", "evict", "watermark", "retention", "oom_adj", "swap_out"]),

    # 保护/不可删除
    (["不能被删除", "不可删除", "保护.*规则", "重要.*保护", "不可淘汰", "重要.*不能", "规则.*不.*删"],
     ["oom_adj", "mlock", "design_constraint", "设计约束", "保护", "不可淘汰"]),
    (["oom_adj", "mlock", "设计约束", "design.constraint"],
     ["保护", "不可淘汰", "强制注入"]),

    # 检索优先级
    (["查询分类", "查询.*级别", "优先级分类", "检索分类"],
     ["SKIP", "LITE", "FULL", "nice", "优先级"]),
    (["SKIP.*LITE.*FULL", "nice.*level"],
     ["优先级", "分类", "查询类型"]),

    # VFS/缓存
    (["知识文件系统", "文件系统.*缓存", "两层缓存", "两级缓存"],
     ["VFS", "dentry", "inode", "cache"]),
    (["VFS", "dentry.*cache", "inode.*cache"],
     ["缓存", "文件系统", "知识"]),

    # BM25/搜索
    (["全文搜索", "全文检索", "中文.*切词", "中文.*分词"],
     ["BM25", "FTS5", "bigram", "tokenize"]),
    (["BM25", "FTS5", "bigram"],
     ["全文", "搜索", "分词", "检索"]),

    # 快速路径
    (["快速跳过", "零开销", "极速.*路径"],
     ["vDSO", "fast.*path", "快速路径"]),
    (["vDSO", "fast.path", "快速路径"],
     ["跳过", "零开销", "极速"]),

    # 新知识保护期
    (["保护期", "新.*加分", "新知识.*保护"],
     ["freshness", "grace", "bonus"]),
    (["freshness.bonus", "grace.days"],
     ["保护期", "新知识", "加分", "衰减"]),

    # chunk/quota
    (["最多.*存.*多少", "存储上限", "配额", "容量限制"],
     ["quota", "chunk_quota", "200"]),
    (["quota", "chunk_quota"],
     ["上限", "配额", "最多", "容量"]),

    # 迭代/版本 — "迭代 98" 需要扩展为 "iter98" "iter" "迭代"
    (["迭代", "第.*次迭代", "版本.*\\d+"],
     ["iter", "迭代"]),
    (["iter\\d+", "iter "],
     ["迭代"]),

    # 去重
    (["去重", "重复.*检测", "合并.*相似"],
     ["dedup", "find_similar", "merge", "already_exists"]),
    (["dedup", "find_similar", "merge_similar"],
     ["去重", "重复", "合并"]),

    # 向量数据库
    (["向量数据库", "向量.*索引", "embedding.*数据库"],
     ["chromadb", "vector", "embedding"]),
    (["chromadb"],
     ["向量", "数据库", "embedding"]),

    # 迭代125：检索/召回语义（查询"检索效果/召回率"应扩展到 FTS/BM25 相关术语）
    (["检索效果", "召回率", "检索质量", "召回精度", "检索优化"],
     ["recall", "precision", "BM25", "fts_rank", "FTS5", "retrieval"]),
    (["recall.*rate", "retrieval.*quality"],
     ["召回", "检索", "精度"]),

    # 迭代125：session/会话 相关
    (["会话恢复", "上次会话", "session.*恢复", "工作集恢复"],
     ["loader", "working_set", "checkpoint", "CRIU", "session_start"]),
    (["CRIU", "checkpoint.*restore", "working.set"],
     ["会话恢复", "工作集", "loader"]),

    # 迭代125：知识导入/wiki 相关
    (["知识导入", "wiki.*导入", "import.*知识", "导入.*知识库"],
     ["import_knowledge", "procedure", "incremental_import"]),
    (["import_knowledge", "incremental_import"],
     ["导入", "知识库", "wiki", "procedure"]),

    # ── 迭代133：memory-os 自知识同义词扩展 ─────────────────────────────────
    # 确保 iter132 导入的 memory-os 架构 chunk 能被自然语言查询命中

    # retriever 检索管道
    (["检索管道", "检索流程", "召回流程", "检索器", "retriever"],
     ["retriever", "FTS5", "BM25", "SKIP", "LITE", "FULL", "检索"]),
    (["检索优先级", "query.*优先级", "查询.*调度"],
     ["SKIP", "LITE", "FULL", "nice", "scheduler", "优先级"]),

    # extractor 提取管道
    (["提取器", "提取管道", "extractor", "知识提取", "stop.*hook"],
     ["extractor", "提取", "decision", "reasoning_chain", "chunk"]),
    (["如何.*提取", "怎么.*提取知识", "哪些.*被提取", "提取.*规则"],
     ["extractor", "chunk_type", "importance", "AIMD", "提取"]),

    # PSI 压力感知
    (["系统压力", "检索压力", "压力感知", "PSI", "psi"],
     ["PSI", "压力", "降级", "FULL", "LITE", "latency"]),
    (["动态降级", "自动降级", "检索降级", "性能降级"],
     ["PSI", "downgrade", "降级", "压力", "psi_stats"]),

    # TLB 检索缓存
    (["检索缓存", "结果缓存", "缓存命中", "TLB", "tlb"],
     ["TLB", "prompt_hash", "chunk_version", "injection_hash", "缓存"]),
    (["prompt.*hash.*缓存", "chunk.*version", "检索结果.*重复"],
     ["TLB", "tlb_read", "tlb_write", "chunk_version", "缓存"]),

    # DRR 公平调度
    (["类型多样", "多样性", "召回多样", "防.*独占", "DRR", "drr"],
     ["DRR", "drr_select", "chunk_type", "多样性", "公平"]),
    (["单一类型.*独占", "decision.*占满", "类型.*不平衡"],
     ["DRR", "max_same_type", "overflow", "多样性"]),

    # madvise 预热
    (["预热加分", "hint.*加分", "预热.*检索", "madvise"],
     ["madvise", "hint", "boost", "prefetch", "预热"]),
    (["读取提示", "访问提示", "预期.*访问"],
     ["madvise", "hint", "madvise_read", "预热"]),

    # Anti-Starvation 反饥饿
    (["饥饿.*加分", "饱和.*惩罚", "反饥饿", "召回.*同质", "anti.*starvation"],
     ["starvation", "saturation", "recall_count", "anti-starvation", "反饥饿"]),
    (["热门.*chunk.*独占", "总是召回.*相同", "新知识.*没被召回"],
     ["starvation", "boost", "saturation", "penalty", "饥饿"]),

    # Deadline Scheduler 超时控制
    (["检索超时", "检索截止", "时间预算", "deadline", "soft.*deadline", "hard.*deadline"],
     ["deadline", "deadline_ms", "deadline_hard_ms", "超时", "截止"]),
    (["检索太慢", "检索.*延迟", "超过.*时限"],
     ["deadline", "psi", "latency", "deadline_skipped", "超时"]),

    # Context Pressure Governor
    (["注入窗口", "对话.*压力", "压缩.*感知", "governor", "context.*pressure"],
     ["governor", "context_pressure", "scale", "turns", "compact", "注入"]),
    (["注入.*太多", "注入.*太少", "对话轮次.*多", "context.*满了"],
     ["governor", "scale", "CRITICAL", "HIGH", "LOW", "压力"]),

    # Readahead 预取
    (["预取", "协同访问", "共现.*预取", "readahead"],
     ["readahead", "prefetch", "cooccurrence", "pair", "预取"]),
    (["一起被召回", "频繁.*同时.*出现", "共现对"],
     ["readahead", "readahead_pairs", "cooccurrence", "预取"]),

    # ASLR 随机扰动
    (["随机扰动", "探索.*检索", "多样化.*召回", "ASLR", "aslr"],
     ["ASLR", "aslr_epsilon", "random", "扰动", "多样"]),

    # chunk_type 类型
    (["chunk.*类型", "知识.*分类", "哪种.*类型", "chunk_type"],
     ["decision", "design_constraint", "reasoning_chain", "excluded_path",
      "procedure", "conversation_summary", "quantitative_evidence"]),
    (["设计约束", "系统约束", "design.*constraint", "强制.*注入.*约束"],
     ["design_constraint", "mlock", "forced", "oom_adj", "约束"]),

    # vDSO 快速路径
    (["vdso", "vDSO", "快速.*退出", "零.*import"],
     ["vDSO", "fast_exit", "SKIP", "lazy_import", "快速路径"]),
    (["import.*开销", "启动.*慢", "冷启动.*延迟"],
     ["vDSO", "lazy_import", "fast_path", "import", "开销"]),

    # MGLRU 多代 LRU
    (["多代.*LRU", "lru.*代", "lru_gen", "MGLRU", "mglru"],
     ["MGLRU", "lru_gen", "aging", "promote", "老化"]),
    (["chunk.*老化", "旧.*chunk.*淘汰", "代数.*管理"],
     ["MGLRU", "lru_gen", "mglru_aging", "evict", "老化"]),

    # DAMON 访问监控
    (["冷.*chunk", "死.*chunk", "长期未访问", "DAMON", "damon"],
     ["DAMON", "cold", "dead", "access_count", "监控"]),
    (["access_count.*0", "从未被访问", "零访问"],
     ["DAMON", "cold", "starvation", "boost", "访问"]),

    # Swap 换入换出
    (["swap.*换出", "换入.*换出", "被换出.*找回", "swap.*fault"],
     ["swap_out", "swap_in", "swap_fault", "demand_paging", "换出"]),
    (["demand.*paging", "按需.*加载", "缺页.*补入", "page.*fault"],
     ["swap_fault", "page_fault_log", "demand_paging", "缺页"]),

    # ── 迭代332：通用语义 Query Expansion — 自然语言问法桥接规则 ─────────────────
    # 目标：将"如何/怎么/为什么"问句前缀 + 动词/名词 映射到对应技术关键词，
    # 填补 GBrain 语义检索与 memory-os BM25 词汇匹配之间的语义鸿沟。
    # 测试基线：Jaccard overlap ~0.19（自然语言 vs 技术关键词对）

    # ─── 类别 A：优化/性能类问句 ───
    # "如何优化检索速度" / "怎么加快" / "性能提升" → 技术关键词
    (["如何.*优化", "怎么.*优化", "优化.*方法", "如何.*加速", "怎么.*加快",
      "如何.*提升.*性能", "性能.*提升", "性能.*优化", "速度.*慢.*怎么",
      "how.*optim", "how.*speed.*up", "improve.*performance"],
     ["optimize", "latency", "deadline", "fast", "performance", "ms",
      "优化", "加速", "提升", "PSI", "vDSO"]),

    # ─── 类别 B：召回/检索效果类问句 ───
    # "为什么召回率低" / "检索不到" / "找不到相关内容" → recall/retrieval 关键词
    (["为什么.*召回", "召回率.*低", "检索.*效果差", "找不到.*相关",
      "检索.*不准", "为什么.*找不到", "没有.*相关.*结果",
      "why.*recall.*low", "retrieval.*quality.*poor", "不准确"],
     ["recall", "FTS5", "BM25", "fts_rank", "precision", "threshold",
      "召回", "检索", "min_score", "候选集"]),

    # ─── 类别 C：去重/合并/冗余问句 ───
    # "如何减少重复" / "合并相似内容" / "知识冗余" → dedup 关键词
    (["如何.*减少.*重复", "怎么.*去重", "合并.*相似", "知识.*冗余",
      "重复.*内容.*怎么", "避免.*重复.*知识",
      "how.*dedup", "how.*merge.*similar", "reduce.*redundan"],
     ["dedup", "find_similar", "merge", "Jaccard", "already_exists",
      "去重", "合并", "相似度", "sleep_consolidate"]),

    # ─── 类别 D：重要性/权重/优先级问句 ───
    # "怎么设置重要性" / "importance 怎么计算" / "哪些知识更重要" → importance 关键词
    (["怎么.*设置.*重要", "如何.*调整.*权重", "importance.*怎么", "哪些.*更重要",
      "知识.*重要性", "优先.*保留.*哪些", "怎么.*决定.*保留",
      "how.*set.*importance", "how.*calculate.*score"],
     ["importance", "weight", "score", "oom_adj", "stability",
      "重要性", "权重", "保留", "importance_override"]),

    # ─── 类别 E：存储/容量/上限问句 ───
    # "能存多少" / "存储容量" / "达到上限" → quota 关键词
    (["能.*存.*多少", "存储.*容量", "知识.*上限", "最多.*多少.*条",
      "达到.*上限.*怎么", "知识库.*满了",
      "how.*much.*store", "storage.*limit", "capacity.*limit"],
     ["quota", "chunk_quota", "max", "limit", "evict", "200",
      "上限", "配额", "容量", "淘汰"]),

    # ─── 类别 F：写入/保存/提取失败问句 ───
    # "知识没有被保存" / "为什么没有提取到" / "提取失败" → extractor 关键词
    (["知识.*没有.*保存", "为什么.*没.*提取", "提取.*失败", "没.*写入",
      "内容.*没有.*记录", "为什么.*不.*提取",
      "why.*not.*extract", "why.*not.*save", "knowledge.*lost"],
     ["extractor", "already_exists", "throttle", "cwnd", "AIMD",
      "提取", "去重", "流控", "写入失败"]),

    # ─── 类别 G：速度/延迟/超时问句 ───
    # "检索太慢" / "hook 超时" / "为什么这么慢" → deadline/latency 关键词
    (["检索.*太慢", "太慢.*了", "为什么.*慢", "响应.*慢",
      "hook.*超时", "超过.*时间", "延迟.*高",
      "why.*slow", "too.*slow", "latency.*high", "timeout"],
     ["deadline", "deadline_ms", "hard_deadline", "latency", "psi",
      "超时", "延迟", "deadline_skipped", "import"]),

    # ─── 类别 H：注入/上下文/输出问句 ───
    # "注入了什么" / "上下文里有什么" / "为什么注入了不相关内容" → injection 关键词
    # iter332修复：扩展目标词匹配实际DB词汇（噪音/无关/context window）
    (["注入.*什么", "为什么.*注入", "上下文.*有什么", "注入.*不相关",
      "注入.*不相关内容", "为什么.*出现.*上下文", "context.*里.*什么",
      "what.*inject", "why.*inject.*irrelevant", "context.*noise",
      "不相关.*内容", "无关.*内容"],
     ["inject", "additionalContext", "min_score", "threshold", "DRR",
      "注入", "上下文", "噪音", "过滤", "无关", "不相关", "边际", "MMR"]),

    # ─── 类别 I：中英文通用概念等价 ───
    # 核心技术词的中英文双向桥接（FTS5 无 stemming，需显式映射）
    (["优化", "提升", "改进", "加速"],
     ["optimize", "improve", "faster", "performance", "speed"]),
    (["optimize", "improve", "enhance", "accelerate"],
     ["优化", "提升", "性能", "加速", "改进"]),

    (["召回", "检索", "查找", "搜索"],
     ["recall", "retrieve", "search", "FTS5", "BM25"]),
    (["recall", "retrieve", "retrieval", "search"],
     ["召回", "检索", "查找", "FTS5"]),

    (["删除", "清理", "淘汰", "移除"],
     ["delete", "evict", "remove", "clean", "purge"]),
    (["delete", "evict", "remove", "purge"],
     ["删除", "清理", "淘汰", "移除", "clean", "delet"]),

    (["保存", "记录", "存储", "写入"],
     ["save", "store", "write", "insert", "persist"]),
    (["save", "store", "write", "insert", "persist"],
     ["保存", "记录", "存储", "写入"]),

    (["重要", "关键", "核心", "优先"],
     ["important", "critical", "priority", "key", "essential"]),
    (["important", "critical", "priority", "essential"],
     ["重要", "关键", "核心", "优先"]),

    # ─── 类别 J：因果/原因/解释问句 ───
    # "为什么会X" / "原因是什么" / "根因" / "导致" → causal_chain/reasoning_chain 关键词
    (["为什么.*会", "原因.*是什么", "怎么导致", "什么.*导致", "导致.*问题",
      "根本原因", "根因",
      "what.*cause", "why.*happen", "root.*cause"],
     ["causal_chain", "reasoning_chain", "根因", "导致", "因为",
      "原因", "causal", "因果"]),

    # ─── 类别 K：比较/对比问句 ───
    # "X 和 Y 有什么区别" / "哪个更好" → 两者的关键词都扩展
    (["有什么区别", "区别.*是什么", "对比.*两者", "哪个.*更好",
      "what.*difference", "compare.*between", "vs.*which"],
     ["difference", "compare", "versus", "区别", "对比", "优劣"]),

    # ─── 类别 L：如何查看/监控/调试问句 ───
    # "怎么查看" / "如何监控" / "怎么调试" / "如何调试" → 日志/工具关键词
    (["怎么.*查看", "如何.*监控", "怎么.*调试", "如何.*调试", "怎么.*检查",
      "如何.*诊断", "查看.*状态", "调试.*问题", "debug.*问题",
      "how.*check", "how.*monitor", "how.*debug", "how.*diagnos"],
     ["dmesg", "log", "stats", "trace", "recall_traces", "psi",
      "日志", "监控", "调试", "statistics", "debug"]),
]

# ── 迭代153/154：Synonym Patterns — 懒编译（first-call JIT）策略 ─────────────
# OS 类比：Linux JIT BPF verifier (4.8, 2016) — 首次调用时编译到机器码，
#   后续调用直接执行已编译的 native code（类比 Python LRU compile cache 命中）。
#
# 迭代153 的 AOT 方案在 per-process hook 模型下是 net-negative（见分析）：
#   每个进程都独立 import store_vfs → AOT 编译 229 条正则 ~16ms（每次都付）
#   而 _synonym_expand 在一次检索中只被调用 1 次，节省 ~8.8ms 首次编译
#   净效果：+16ms - 8.8ms = +7.2ms（更慢）
#
# 迭代154 改为懒编译：
#   _SYNONYM_RULES_COMPILED = None 作为哨兵（模块加载时 0ms）
#   _synonym_expand 首次调用时编译（付 ~8.8ms 一次）
#   同一进程多次调用（如测试场景）后续直接复用
#   per-process 节约：import 从 ~32ms 降回 ~16ms，首次调用 ~8.8ms（与旧 AOT 相同）
#   本质：推迟到第一次需要时再编译，不改变 per-process 总成本，但消除无用的 import 延迟
#
# 注意：如果未来改为 daemon/socket 模式（跨请求进程复用），懒编译自动变为"只付一次"优化。
_SYNONYM_RULES_COMPILED = None  # 懒编译哨兵，None = 尚未编译

# ── 迭代155：Synonym Prescan — Bloom filter 风格快速退出 ─────────────────────
# OS 类比：Linux Bloom filter in network packet classification (iptables hashlimit, 2003)
#   iptables 在做完整规则表 walk 之前，先用 Bloom filter 做 O(1) 预筛：
#   如果 Bloom filter 说"不可能命中"，直接跳过，不进入 O(N) 规则遍历。
#   memory-os 等价问题：
#     _synonym_expand 每次都要先触发 _ensure_synonym_compiled()（~9ms JIT），
#     再做 60 条规则匹配（~2.5ms），但大多数 query 根本没有同义词触发词。
#     P50 query 只有 22 字，短 query 极少含 kswapd/淘汰/BM25 等术语。
#   解决：预先提取所有同义词触发模式中的"简单关键词"（无正则元字符的子串）
#     构建 frozenset —— O(1) 成员检测，模块加载时 0ms（纯字符串操作）。
#     _synonym_expand 先检查 query 是否含任何触发关键词：
#       未命中 → return [] 立即退出（0.002ms），跳过 9ms JIT + 2.5ms 匹配
#       命中 → 继续完整流程（行为与 iter154 完全相同）
#   预期效果：P50（22 char query）中 ~60-70% 不含任何触发词 → 节省 ~9ms
#   误判代价：极低——prescan 只做快速退出（false negative 导致漏扩展），
#     不会误扩展（false positive 最多多做一次完整匹配，无害）。
def _build_synonym_trigger_keywords() -> frozenset:
    """
    从 _SYNONYM_RULES 提取所有触发模式中的简单关键词（无正则元字符的子串）。
    模块加载时调用一次（纯字符串操作，~0ms），不触发 re.compile。
    OS 类比：iptables Bloom filter 预构建 — 规则加载时建立 bit array，不在包处理时构建。
    """
    keywords: set = set()
    for triggers, _expansions in _SYNONYM_RULES:
        for t in triggers:
            # 提取 CJK 连续子串（≥2字）作为触发词
            cn_chunks = re.findall(r'[\u4e00-\u9fff]{2,}', t)
            for chunk in cn_chunks:
                keywords.add(chunk[:2])  # bigram 前缀足够区分
                if len(chunk) >= 4:
                    keywords.add(chunk[2:4])  # 第二个 bigram
            # 提取英文词（≥4字，跳过正则元字符前缀的词）
            for m in re.finditer(r'[a-zA-Z]{4,}', t):
                w = m.group().lower()
                keywords.add(w)
    return frozenset(keywords)


_SYNONYM_TRIGGER_KEYWORDS: frozenset = _build_synonym_trigger_keywords()


def _ensure_synonym_compiled():
    """
    按需编译同义词规则（懒编译，只在首次 _synonym_expand 调用时执行）。
    OS 类比：Linux JIT BPF — bpf() 系统调用时才触发 JIT 编译，不在 load time 编译。
    """
    global _SYNONYM_RULES_COMPILED
    if _SYNONYM_RULES_COMPILED is not None:
        return
    compiled = []
    for triggers, expansions in _SYNONYM_RULES:
        cpats = []
        for t in triggers:
            try:
                cpats.append(re.compile(t, re.IGNORECASE))
            except re.error:
                cpats.append(None)  # fallback to string contains
        compiled.append((cpats, triggers, expansions))
    _SYNONYM_RULES_COMPILED = compiled


def _synonym_expand(query: str, seen: set) -> list:
    """
    对 query 做同义词扩展，返回额外的 FTS5 token 列表。
    只追加 seen 中不存在的新 token，避免重复。
    迭代153+154：懒编译策略 — 首次调用时编译同义词正则，消除 import 时的 AOT 开销。
    迭代155：Prescan 快速退出 — Bloom filter 风格，O(1) 检测 query 是否含任何触发词，
      未命中则直接返回 []，跳过 ~9ms JIT 编译 + ~2.5ms 规则匹配。
    """
    query_lower = query.lower()

    # ── 迭代155：Prescan — Bloom filter 快速退出 ──────────────────────────────
    # OS 类比：iptables hashlimit Bloom filter — O(1) 预筛，不命中直接 accept，不走规则表
    # 特殊情况："迭代 N" 模式不在 _SYNONYM_TRIGGER_KEYWORDS 中（数字），需单独检测
    _has_iter_pattern = '迭代' in query
    if not _has_iter_pattern:
        # 提取 query 的 CJK bigrams + 英文词（与 _build_synonym_trigger_keywords 对称）
        _q_cjk = re.findall(r'[\u4e00-\u9fff]{2,}', query)
        _q_bigrams = set()
        for _chunk in _q_cjk:
            _q_bigrams.add(_chunk[:2])
            if len(_chunk) >= 4:
                _q_bigrams.add(_chunk[2:4])
        _q_eng = set(m.lower() for m in re.findall(r'[a-zA-Z]{4,}', query_lower))
        _q_tokens = _q_bigrams | _q_eng
        # 快速交集检测（frozenset.__and__ 是 O(min(|A|,|B|)) ≈ O(query tokens count)）
        if not (_q_tokens & _SYNONYM_TRIGGER_KEYWORDS):
            return []  # prescan miss: 0.002ms，跳过 ~11.5ms 后续处理

    _ensure_synonym_compiled()
    extra = []

    # 特殊处理："迭代 N" / "迭代N" → "iterN"（高频模式，规则表无法覆盖）
    for m in re.finditer(r'迭代\s*(\d+)', query):
        iter_token = f'iter{m.group(1)}'
        if iter_token not in seen:
            seen.add(iter_token)
            extra.append(f'"{iter_token}"')

    for compiled_patterns, raw_triggers, expansions in _SYNONYM_RULES_COMPILED:
        matched = False
        for i, pat in enumerate(compiled_patterns):
            if pat is not None:
                if pat.search(query_lower) or pat.search(query):
                    matched = True
                    break
            else:
                # re.error fallback — use string contains
                if raw_triggers[i].lower() in query_lower:
                    matched = True
                    break
        if matched:
            for term in expansions:
                term_lower = term.lower()
                if term_lower not in seen and len(term_lower) >= 2:
                    seen.add(term_lower)
                    # 英文术语加引号，CJK 术语转 bigram
                    if re.match(r'^[a-zA-Z0-9_]+$', term):
                        extra.append(f'"{term_lower}"')
                    else:
                        # CJK: 生成 bigram
                        cjk = re.findall(r'[\u4e00-\u9fff]', term)
                        if len(cjk) >= 2:
                            for i in range(len(cjk) - 1):
                                bg = cjk[i] + cjk[i+1]
                                if bg not in seen:
                                    seen.add(bg)
                                    extra.append(f'"{bg}"')
                        elif cjk:
                            for c in cjk:
                                if c not in seen:
                                    seen.add(c)
                                    extra.append(f'"{c}"')
    return extra

def fts_search(conn: sqlite3.Connection, query: str, project: str,
               top_k: int = 10, chunk_types: tuple = None) -> List[dict]:
    """
    迭代23：FTS5 全文搜索 — 替代全表扫描 + Python BM25。
    OS 类比：htree 的 ext3_htree_fill_tree() → O(log N) 目录查找。

    BM25 由 SQLite FTS5 内置函数 bm25() 计算（C 实现），
    权重参数：summary 权重 2.0, content 权重 1.0。

    迭代172：将 2 次 FTS5 查询（project + global）合并为 1 次 IN(project, 'global')。
    OS 类比：readv() vs 2×read() — 向量化 I/O 减少系统调用次数。
    节省：~0.63ms（消除第二次 FTS5 query + sort，单次 IN 查询由 SQLite 优化器处理）。

    返回与 get_chunks() 相同格式的 dict 列表，额外带 fts_rank 字段。
    """
    match_expr = _fts5_escape(query)
    if not match_expr:
        return []

    # FTS5 bm25() 返回负值（越小越相关），取负后变为正值排序
    # 权重参数按列顺序：rowid_ref(UNINDEXED)=0, summary=2.0, content=1.0
    # 迭代97：非 content-sync 模式，通过 rowid_ref 关联主表
    def _run_fts(project_filter):
        """project_filter: None=全库, str=单项目, list/tuple=多项目"""
        sql = """
            SELECT mc.id, mc.summary, mc.content, mc.importance, mc.last_accessed,
                   mc.chunk_type, COALESCE(mc.access_count, 0), mc.created_at,
                   -bm25(memory_chunks_fts, 0, 2.0, 1.0) AS fts_rank,
                   COALESCE(mc.lru_gen, 0), mc.project,
                   mc.verification_status, mc.confidence_score,
                   COALESCE(mc.retrievability, 1.0),
                   COALESCE(mc.source_reliability, 0.7)
            FROM memory_chunks_fts
            JOIN memory_chunks mc ON mc.rowid = CAST(memory_chunks_fts.rowid_ref AS INTEGER)
            WHERE memory_chunks_fts MATCH ?
              AND mc.summary != ''
              AND mc.importance > 0.0
        """
        # ── 迭代335：Ghost Filter (Layer 2) — FTS5 查询内过滤 importance=0 的 ghost chunk ──
        # OS 类比：Linux page allocator MIGRATE_TYPES 过滤 — 分配器跳过 MIGRATE_RESERVE 类型页
        # ghost chunk 由 consolidate/merge 路径产生（importance=0, summary=[merged→...]），
        # 但未被物理删除 → 仍在 FTS5 索引中 → FTS5 命中 ghost 消耗 result slot + 计算开销。
        # AND mc.importance > 0.0 是低成本的 B-tree 过滤（importance 列已索引），
        # 无需单独的 ghost_filter_enabled sysctl 开关（始终启用，无负面影响）。
        params = [match_expr]
        if project_filter is not None:
            if isinstance(project_filter, (list, tuple)):
                # iter172: 合并多个 project 为一次 IN 查询
                placeholders = ",".join("?" * len(project_filter))
                sql += f" AND mc.project IN ({placeholders})"
                params.extend(project_filter)
            else:
                sql += " AND mc.project = ?"
                params.append(project_filter)
        if chunk_types:
            placeholders = ",".join("?" * len(chunk_types))
            sql += f" AND mc.chunk_type IN ({placeholders})"
            params.extend(chunk_types)
        sql += " ORDER BY fts_rank DESC LIMIT ?"
        params.append(top_k)
        try:
            return conn.execute(sql, params).fetchall()
        except Exception:
            return []

    # ── 迭代123 + iter172：Always-merge global — 单次 IN 查询 ──
    # OS 类比：readv() vectorized I/O — 两个缓冲区的 I/O 合并为一次系统调用，
    #   减少 kernel/userspace 切换次数（每次 syscall ~1μs overhead）。
    #   迭代123 将 project + global 分两次查询（2×FTS5），
    #   iter172 改为单次 IN(project, 'global') 查询（1×FTS5），节省 ~0.63ms。
    #
    # 语义保持：
    #   1. IN(project, 'global') 等价于旧的 project_query UNION global_query（去重由 id 唯一保证）
    #   2. ORDER BY fts_rank DESC LIMIT top_k 在合并结果上执行（全局最优）
    #   3. 历史孤儿 fallback 路径保留（全库搜索，project=None）

    # Step 1+2（iter172合并）：单次搜 project + global
    if project is None or project == "global":
        # project 本身是 global 或未指定：直接用单 project 查询
        rows = _run_fts(project)
    else:
        # iter172: 将 project + "global" 合并为一次 IN 查询
        rows = _run_fts([project, "global"])

    # Step 3: 历史孤儿 fallback：project ID 变化后的旧 chunk 补救
    # 仅当合并后仍不足 top_k 一半时触发全库搜索
    if len(rows) < max(1, top_k // 2):
        all_rows = _run_fts(None)
        seen_ids = {r[0] for r in rows}
        for r in all_rows:
            if r[0] not in seen_ids:
                rows.append(r)
                seen_ids.add(r[0])
            if len(rows) >= top_k:
                break

    result = []
    for rid, summary, content, importance, last_accessed, chunk_type, access_count, created_at, fts_rank, lru_gen, chunk_project, verification_status, confidence_score, retrievability, source_reliability in rows:
        result.append({
            "id": rid,
            "summary": summary or "",
            "content": content or "",
            "importance": importance if importance is not None else 0.5,
            "last_accessed": last_accessed or "",
            "chunk_type": chunk_type or "task_state",
            "access_count": access_count or 0,
            "created_at": created_at or "",
            "fts_rank": fts_rank,
            "lru_gen": lru_gen or 0,
            "project": chunk_project or "",  # 迭代111: NUMA distance scoring
            "verification_status": verification_status,
            "confidence_score": confidence_score,
            "retrievability": retrievability if retrievability is not None else 1.0,  # iter369
            "source_reliability": float(source_reliability) if source_reliability is not None else 0.7,  # iter396
        })
    return result

# ── CRUD 操作 ─────────────────────────────────────────────────

def get_chunks(conn: sqlite3.Connection, project: str,
               chunk_types: tuple = None) -> list:
    """
    查询项目的所有 chunk，返回 dict 列表。
    OS 类比：VFS 的 readdir() — 统一接口读取不同文件系统的目录项。
    """
    # 迭代94: 同时检索 global 层（跨项目共享知识）
    projects = [project] if project == "global" else [project, "global"]
    proj_ph = ",".join("?" * len(projects))
    if chunk_types:
        placeholders = ",".join("?" * len(chunk_types))
        query = f"""SELECT id, summary, content, importance, last_accessed,
                           chunk_type, COALESCE(access_count, 0), created_at, project,
                           verification_status, confidence_score, COALESCE(lru_gen, 0)
                    FROM memory_chunks
                    WHERE project IN ({proj_ph}) AND chunk_type IN ({placeholders})
                    AND summary != ''"""
        rows = conn.execute(query, (*projects, *chunk_types)).fetchall()
    else:
        rows = conn.execute(
            f"""SELECT id, summary, content, importance, last_accessed,
                      chunk_type, COALESCE(access_count, 0), created_at, project,
                      verification_status, confidence_score, COALESCE(lru_gen, 0)
               FROM memory_chunks
               WHERE project IN ({proj_ph}) AND summary != ''""",
            tuple(projects),
        ).fetchall()
    result = []
    for rid, summary, content, importance, last_accessed, chunk_type, access_count, created_at, chunk_project, verification_status, confidence_score, lru_gen in rows:
        result.append({
            "id": rid,
            "summary": summary or "",
            "content": content or "",
            "importance": importance if importance is not None else 0.5,
            "last_accessed": last_accessed or "",
            "chunk_type": chunk_type or "task_state",
            "access_count": access_count or 0,
            "created_at": created_at or "",
            "project": chunk_project or "",  # 迭代111: NUMA distance scoring
            "verification_status": verification_status,
            "confidence_score": confidence_score,
            "lru_gen": lru_gen,
        })
    return result

def insert_chunk(conn: sqlite3.Connection, chunk_dict: dict) -> None:
    """
    插入或替换一条 chunk。
    OS 类比：VFS 的 write() — 统一写入接口。

    迭代97：同步维护 FTS5 索引（非 content-sync 模式，手动写入预处理文本）。
    """
    d = chunk_dict
    tags = json.dumps(d["tags"], ensure_ascii=False) if isinstance(d.get("tags"), list) else d.get("tags", "[]")
    # 如果已存在（REPLACE 路径），先从 FTS5 删除旧记录
    existing_rowid = conn.execute(
        "SELECT rowid FROM memory_chunks WHERE id=?", (d["id"],)
    ).fetchone()
    if existing_rowid:
        conn.execute(
            "DELETE FROM memory_chunks_fts WHERE rowid_ref=?",
            (str(existing_rowid[0]),)
        )

    # 迭代306：raw_snippet 截断到 500 字（防止超长写入）
    raw_snippet = (d.get("raw_snippet") or "")[:500]
    # 迭代315：encoding_context 序列化为 JSON 字符串
    enc_ctx = d.get("encoding_context", {})
    if isinstance(enc_ctx, dict):
        enc_ctx_str = json.dumps(enc_ctx, ensure_ascii=False)
    else:
        enc_ctx_str = enc_ctx if isinstance(enc_ctx, str) else "{}"
    conn.execute("""
        INSERT OR REPLACE INTO memory_chunks
        (id, created_at, updated_at, project, source_session,
         chunk_type, info_class, content, summary, tags, importance,
         retrievability, last_accessed, feishu_url, access_count, oom_adj, lru_gen,
         stability, raw_snippet, encoding_context)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        d["id"], d["created_at"], d["updated_at"], d["project"], d["source_session"],
        d["chunk_type"], d.get("info_class", "world"),
        d["content"], d["summary"], tags,
        d["importance"], d["retrievability"], d["last_accessed"], d.get("feishu_url"),
        d.get("access_count", 0), d.get("oom_adj", 0), d.get("lru_gen", 0),
        d.get("stability", 1.0), raw_snippet, enc_ctx_str,
    ))
    # 迭代97：写入 FTS5（CJK 预处理）
    # iter142：使用 new_rowid 重新清理 FTS（防止 INSERT OR REPLACE 保留 rowid 时残留旧条目）
    new_rowid = conn.execute(
        "SELECT rowid FROM memory_chunks WHERE id=?", (d["id"],)
    ).fetchone()
    if new_rowid and d.get("summary"):
        try:
            # 先清理该 rowid 的所有现有 FTS 条目（幂等保护，防止 race/double-insert）
            conn.execute(
                "DELETE FROM memory_chunks_fts WHERE rowid_ref=?",
                (str(new_rowid[0]),)
            )
            # iter108：结构化 summary 先归一化再 CJK 分词，修复 [topic]/>/ 截断 FTS5 检索
            fts_summary = _cjk_tokenize(_normalize_structured_summary(d.get("summary") or ""))
            fts_content = _cjk_tokenize(_normalize_structured_summary(d.get("content") or ""))
            conn.execute(
                "INSERT INTO memory_chunks_fts(rowid_ref, summary, content) VALUES (?, ?, ?)",
                (str(new_rowid[0]), fts_summary, fts_content)
            )
        except Exception:
            pass  # FTS5 表可能尚未创建（ensure_schema 未调用时）
    bump_chunk_version()  # 迭代64: TLB v2 — 新 chunk 写入递增版本

    # 迭代310：entity_map 自动关联 — 将 chunk 与 entity_edges 中的 entity 绑定
    # OS 类比：Linux dentry cache — 路径名→inode 的反向映射，insert_chunk 时顺便建立
    # 策略：用 summary 子串匹配 entity_edges 中已知的 entity 名，写入 entity_map
    # 这样 spreading_activate 才能从 FTS5 命中的 chunk 找到对应 entity，沿图扩散
    try:
        chunk_id = d["id"]
        project = d.get("project", "")
        summary_lower = (d.get("summary") or "").lower()
        if summary_lower and project:
            # 取该 project 的所有已知 entity（from_entity 和 to_entity）
            entity_rows = conn.execute(
                "SELECT DISTINCT from_entity FROM entity_edges WHERE project=? "
                "UNION SELECT DISTINCT to_entity FROM entity_edges WHERE project=?",
                (project, project)
            ).fetchall()
            now_str = datetime.now(timezone.utc).isoformat()
            for (ent,) in entity_rows:
                if not ent:
                    continue
                # entity 名（去下划线/中划线，小写）出现在 summary 中则建立映射
                ent_normalized = ent.lower().replace("_", " ").replace("-", " ")
                if ent_normalized in summary_lower or ent.lower() in summary_lower:
                    conn.execute(
                        """INSERT OR REPLACE INTO entity_map
                           (entity_name, chunk_id, project, updated_at)
                           VALUES (?, ?, ?, ?)""",
                        (ent, chunk_id, project, now_str)
                    )
    except Exception:
        pass  # entity_map 写入失败不阻塞主流程

    # iter396：Source Monitoring — 自动推断 source_type，写入 source_reliability
    # 仅在 chunk dict 未明确提供 source_type 时自动推断
    try:
        _sm_chunk_id = d["id"]
        _sm_source_type = d.get("source_type")  # 允许外部显式指定
        _sm_chunk_type = d.get("chunk_type", "task_state")
        _sm_content = (d.get("content") or "") + " " + (d.get("summary") or "")
        apply_source_monitoring(conn, _sm_chunk_id, _sm_chunk_type,
                                _sm_content, _sm_source_type)
    except Exception:
        pass  # source monitoring 失败不阻塞主流程

    # iter401：Elaborative Encoding — 写入时计算加工深度，调整初始 stability
    # 深度加工（因果/结构/对比/精细阐述）→ 更高初始 stability
    try:
        _dop_chunk_id = d["id"]
        _dop_content = (d.get("content") or "") + " " + (d.get("summary") or "")
        _dop_base_stability = d.get("stability", 1.0)
        _dop_new_stability = apply_depth_of_processing(
            conn, _dop_chunk_id, _dop_content, _dop_base_stability
        )
    except Exception:
        _dop_new_stability = d.get("stability", 1.0)
        pass  # depth_of_processing 写入失败不阻塞主流程

    # iter402：Schema Theory — 图式先验加成（Bartlett 1932）
    # entity_map 已建立后再查 prior schema（entity_map 由上方 entity_map 自动关联步骤填充）
    # 先验 chunk stability 均值 × 0.2 作为 schema bonus
    try:
        _schema_chunk_id = d["id"]
        _schema_project = d.get("project", "")
        if _schema_project:
            apply_schema_scaffolding(conn, _schema_chunk_id, _schema_project,
                                     base_stability=_dop_new_stability)
    except Exception:
        pass  # schema scaffolding 失败不阻塞主流程

    # iter403：Cue-Dependent Forgetting — 提取编码上下文，写入 encode_context 字段
    # 编码时的上下文线索 = content + summary + tags + chunk_type 中提取的关键词集
    try:
        _cdf_content = (d.get("content") or "") + " " + (d.get("summary") or "")
        _cdf_tags = d.get("tags", [])
        if isinstance(_cdf_tags, str):
            import json as _json_cdf
            try:
                _cdf_tags = _json_cdf.loads(_cdf_tags)
            except Exception:
                _cdf_tags = []
        _cdf_chunk_type = d.get("chunk_type", "")
        _cdf_encode_ctx = extract_encode_context(
            _cdf_content, tags=_cdf_tags, chunk_type=_cdf_chunk_type
        )
        if _cdf_encode_ctx:
            conn.execute(
                "UPDATE memory_chunks SET encode_context=? WHERE id=?",
                (_cdf_encode_ctx, d["id"])
            )
    except Exception:
        pass  # encode_context 写入失败不阻塞主流程

    # iter404：Semantic Priming — 检索后启动相关 entity
    # 当 chunk 被插入时，它的 entity 被 primed（以支持后续 spreading）
    # 这里只在 insert_chunk 时做轻量 priming（full priming 在检索时触发）
    try:
        _pr_content = (d.get("content") or "") + " " + (d.get("summary") or "")
        _pr_ctx = extract_encode_context(_pr_content, chunk_type=d.get("chunk_type", ""))
        if _pr_ctx and d.get("project"):
            prime_entities(conn, _pr_ctx.split(","), d["project"], prime_strength=0.3)
    except Exception:
        pass

    # iter406：Generation Effect — 主动生成内容 stability 加成（McDaniel & Einstein 1986）
    # 检测内容中的生成标记（推理人称/假设检验/元认知），计算 generation score，
    # score 越高 → stability 增量越大（补充 iter401 结构深度 + iter392 类型加成）
    try:
        _ge_content = d.get("content") or ""
        _ge_summary = d.get("summary") or ""
        _ge_source_type = d.get("source_type")
        _ge_base_stability = d.get("stability", 1.0)
        apply_generation_effect(
            conn, d["id"], _ge_content, _ge_summary,
            source_type=_ge_source_type,
            base_stability=_ge_base_stability,
        )
    except Exception:
        pass  # generation effect 写入失败不阻塞主流程

    # iter407: Von Restorff Effect — 孤立 chunk 得到 stability bonus（von Restorff 1933）
    # 在均匀背景中，语义独特/孤立的 chunk 比普通 chunk 有更强的记忆留存率
    try:
        _vr_project = d.get("project", "")
        _vr_base_stability = d.get("stability", 1.0)
        if _vr_project:
            apply_isolation_effect(conn, d["id"], _vr_project, base_stability=_vr_base_stability)
    except Exception:
        pass  # von restorff 效应写入失败不阻塞主流程

    # iter408: Proactive Interference — 旧强记忆干扰新 chunk 的 initial stability（Underwood 1957）
    # 项目中已有高相似度+高 access_count 的 chunk → 新 chunk stability 降低
    try:
        _pi_project = d.get("project", "")
        _pi_base_stability = d.get("stability", 1.0)
        if _pi_project:
            apply_proactive_interference(conn, d["id"], _pi_project, base_stability=_pi_base_stability)
    except Exception:
        pass  # proactive interference 写入失败不阻塞主流程

    # iter409: Flashbulb Memory — 高情绪唤醒 chunk 的 initial stability 加强（Brown & Kulik 1977）
    # emotional_weight > 0 → stability bonus（与 iter376 检索时加分互补，这里是写入时固化增强）
    try:
        _fb_base_stability = d.get("stability", 1.0)
        apply_flashbulb_effect(conn, d["id"], base_stability=_fb_base_stability)
    except Exception:
        pass  # flashbulb effect 写入失败不阻塞主流程

    # iter410: Primacy Effect — 项目最早创建的 chunk 是基础 schema，stability 首位加成（Murdock 1962）
    # boot-time parameters 类比：项目初期确立的知识比后来的更持久（rehearsal hypothesis）
    try:
        _pr_project = d.get("project", "")
        _pr_base_stability = d.get("stability", 1.0)
        if _pr_project:
            apply_primacy_effect(conn, d["id"], _pr_project, base_stability=_pr_base_stability)
    except Exception:
        pass  # primacy effect 写入失败不阻塞主流程

    # iter411: Levels of Processing — encode_context 实体数量代理编码深度（Craik & Lockhart 1972）
    # 更多语义实体 = 更丰富语义网络 = 更深加工 = 更强 stability
    try:
        _lop_base_stability = d.get("stability", 1.0)
        apply_depth_effect(conn, d["id"], base_stability=_lop_base_stability)
    except Exception:
        pass  # depth effect 写入失败不阻塞主流程

    # iter414: Self-Reference Effect — 含自我参照标记的 chunk 获得 stability 加成（Rogers et al. 1977）
    # 自我参照加工激活 PFC + hippocampus 双路径，形成更强记忆痕迹
    try:
        _sr_base_stability = d.get("stability", 1.0)
        apply_self_reference_effect(conn, d["id"], base_stability=_sr_base_stability)
    except Exception:
        pass  # self-reference effect 写入失败不阻塞主流程

    # iter416: Zeigarnik Effect — 未完成任务信号词 → stability 加成（Zeigarnik 1927）
    try:
        _zg_base_stability = d.get("stability", 1.0)
        apply_zeigarnik_effect(conn, d["id"], base_stability=_zg_base_stability)
    except Exception:
        pass  # zeigarnik effect 写入失败不阻塞主流程

    # iter418: Directed Forgetting — 过时/已完成信号词 → stability 惩罚（MacLeod 1998）
    try:
        _df_base_stability = d.get("stability", 1.0)
        apply_directed_forgetting(conn, d["id"], base_stability=_df_base_stability)
    except Exception:
        pass  # directed forgetting 写入失败不阻塞主流程

    # iter419: Associative Memory — 与强关联 chunk 共享实体 → stability 加成（Ebbinghaus 1885）
    try:
        _am_project = d.get("project", "")
        _am_base_stability = d.get("stability", 1.0)
        if _am_project:
            apply_associative_memory_bonus(
                conn, d["id"], _am_project, base_stability=_am_base_stability
            )
    except Exception:
        pass  # associative memory 写入失败不阻塞主流程

    # iter421: Retroactive Interference — 新知识干扰旧相关 chunk 的稳定性（McGeoch 1932）
    try:
        _ri_project = d.get("project", "")
        _ri_base_stability = d.get("stability", 1.0)
        if _ri_project:
            apply_retroactive_interference(
                conn, d["id"], _ri_project, base_stability=_ri_base_stability
            )
    except Exception:
        pass  # retroactive interference 写入失败不阻塞主流程

    # iter415: store original encode_context token count for variability tracking
    # This count is used by apply_encoding_variability at access time to measure enrichment
    try:
        _ec_str = d.get("encoding_context", {})
        if isinstance(_ec_str, dict):
            import json as _json
            _ec_str = _json.dumps(_ec_str)
        # Read current encode_context from DB (may have been set by extract_encode_context)
        _ec_row = conn.execute(
            "SELECT encode_context FROM memory_chunks WHERE id=?", (d["id"],)
        ).fetchone()
        if _ec_row:
            _ec_val = _ec_row[0] if isinstance(_ec_row, (list, tuple)) else _ec_row["encode_context"]
            _orig_tokens = [t.strip() for t in (_ec_val or "").split(",") if t.strip()]
            _orig_count = len(_orig_tokens)
            # Store original_ec_count in DB (if column exists; safe fallback)
            try:
                conn.execute(
                    "UPDATE memory_chunks SET original_ec_count=? WHERE id=?",
                    (_orig_count, d["id"])
                )
            except Exception:
                pass  # column may not exist in older schemas
    except Exception:
        pass  # iter415 init 失败不阻塞主流程


# ── iter403：Cue-Dependent Forgetting — Context-Sensitive Retrieval（Tulving 1974）──
#
# 认知科学依据：
#   Tulving & Thomson (1973) Encoding Specificity Principle：
#     编码时的上下文（cues）与检索时的上下文（retrieval cues）重叠度越高，
#     检索成功率越高。这是记忆最重要的规律之一。
#   Godden & Baddeley (1975) Context-Dependent Memory：
#     水下学的词在水下测试效果最好（环境上下文匹配），
#     陆上学的词在陆上测试效果最好。
#   Estes (1955) Stimulus Fluctuation Model：
#     记忆提取受"编码时 context"与"检索时 context"的重叠度（θ）决定。
#
# OS 类比：Linux NUMA-aware memory allocation —
#   进程倾向于从本地 NUMA node（编码时 context = home node）分配内存；
#   当进程的运行 node（检索时 context）越接近 home node，
#   内存访问延迟越低（命中率越高）。
#   context_overlap ≈ NUMA distance 的倒数：overlap = 1 → local node（最优）。
#
# 实现：
#   extract_encode_context(text, tags, chunk_type) → str（逗号分隔关键词）
#     写入时从 content/summary/tags 提取关键词集，存入 encode_context 字段。
#   compute_context_overlap(encode_ctx, retrieve_ctx) → float [0.0, 1.0]
#     Jaccard 相似度：|A∩B| / |A∪B|。
#   context_cue_weight(overlap) → float [0.85, 1.20]
#     overlap → 检索分权重：高重叠 → 提升检索优先级（+20%），低重叠 → 轻微降权。
#   apply_context_cue_boost(chunk, retrieve_context) → float
#     对 fts_search/retriever 返回的 chunk score 应用 context cue weight。

import re as _re_cdf

# 停用词（中英文，过滤掉无语义的功能词）
_CDF_STOPWORDS = frozenset({
    # 英文
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "on",
    "at", "by", "for", "with", "from", "and", "or", "but", "not", "it",
    "this", "that", "these", "those", "i", "we", "you", "he", "she", "they",
    # 中文虚词/功能词
    "的", "了", "在", "是", "有", "和", "也", "不", "这", "那", "但", "而",
    "都", "就", "以", "为", "于", "中", "上", "下", "个", "我", "你", "他",
})

_CDF_TOKEN_RE = _re_cdf.compile(r'[a-zA-Z][a-zA-Z0-9_\-]*|[\u4e00-\u9fff]{2,}')


def extract_encode_context(
    text: str,
    tags: list = None,
    chunk_type: str = "",
    max_tokens: int = 50,
) -> str:
    """
    iter403：从 content/summary/tags/chunk_type 提取编码上下文关键词。

    OS 类比：NUMA node affinity setup — 进程创建时记录 preferred node（home node），
      之后分配内存时优先从该 node 取。

    Returns:
      逗号分隔的关键词字符串（小写，去停用词，最多 max_tokens 个）
    """
    if not text:
        return ""
    # 提取 tokens
    tokens = set()
    for tok in _CDF_TOKEN_RE.findall(text.lower()):
        if tok not in _CDF_STOPWORDS and len(tok) >= 2:
            tokens.add(tok)
    # 加入 tags（标签是高权重上下文信号）
    if tags:
        for tag in tags:
            if isinstance(tag, str):
                t = tag.lower().strip()
                if t and t not in _CDF_STOPWORDS:
                    tokens.add(t)
    # 加入 chunk_type（类型本身也是 context signal）
    if chunk_type:
        tokens.add(chunk_type.lower())
    # 限制数量（取前 max_tokens，按字母序稳定）
    sorted_tokens = sorted(tokens)[:max_tokens]
    return ",".join(sorted_tokens)


def compute_context_overlap(
    encode_context: str,
    retrieve_context: str,
) -> float:
    """
    iter403：计算编码时上下文与检索时上下文的 Jaccard 重叠度。

    OS 类比：NUMA distance 计算 —
      两个 node 之间的距离越小，内存访问越快。
      overlap = 1 - normalized_distance：1.0 = 同一 node（最优）。

    Returns:
      float ∈ [0.0, 1.0]，0.0 = 无重叠，1.0 = 完全相同
    """
    if not encode_context or not retrieve_context:
        return 0.0
    try:
        enc_set = set(t.strip() for t in encode_context.split(",") if t.strip())
        ret_set = set(t.strip() for t in retrieve_context.split(",") if t.strip())
        if not enc_set or not ret_set:
            return 0.0
        intersection = len(enc_set & ret_set)
        union = len(enc_set | ret_set)
        return round(intersection / union, 4) if union > 0 else 0.0
    except Exception:
        return 0.0


def context_cue_weight(overlap: float) -> float:
    """
    iter403：将 context overlap 映射到检索分权重。

    分段函数（OS 类比：NUMA access latency tiers）：
      - overlap >= 0.50 → weight ∈ [1.10, 1.20]（高上下文匹配，类比 local node，延迟最低）
      - overlap ∈ [0.20, 0.50) → weight = 1.0（中等匹配，不调整，类比远端 node 但可访问）
      - overlap < 0.20 → weight ∈ [0.85, 1.0)（低匹配，轻微降权，类比跨 NUMA 域）

    设计原则：
      - 高匹配给正向激励（最多 +20%），强调上下文相关性
      - 低匹配给轻微惩罚（最多 -15%），避免跨 context 污染
      - 中间区域中性（避免噪声波动影响检索）

    Returns:
      float ∈ [0.85, 1.20]
    """
    try:
        r = max(0.0, min(1.0, float(overlap) if overlap is not None else 0.0))
    except (TypeError, ValueError):
        r = 0.0

    if r >= 0.50:
        # 高重叠：linear 插值 [1.10, 1.20]
        weight = 1.10 + (r - 0.50) / 0.50 * 0.10
    elif r >= 0.20:
        # 中等：不调整
        weight = 1.0
    else:
        # 低重叠：linear 插值 [0.85, 1.0)
        weight = 0.85 + r / 0.20 * 0.15

    return round(min(1.20, max(0.85, weight)), 4)


def apply_context_cue_boost(
    conn: sqlite3.Connection,
    chunk_id: str,
    retrieve_context: str,
    base_score: float = 1.0,
) -> float:
    """
    iter403：查询 chunk 的 encode_context，计算与 retrieve_context 的 overlap，
    返回 context_cue_weight 调整后的 score。

    OS 类比：NUMA-aware scheduler load balancing —
      检索时调度器偏向从与查询 context 最近的 NUMA node 上的 chunk 取结果。

    Returns:
      float：调整后的 score（base_score × context_cue_weight）
    """
    if not retrieve_context or not chunk_id:
        return base_score
    try:
        row = conn.execute(
            "SELECT encode_context FROM memory_chunks WHERE id=?", (chunk_id,)
        ).fetchone()
        if row is None or not row[0]:
            return base_score
        encode_ctx = row[0]
        overlap = compute_context_overlap(encode_ctx, retrieve_context)
        weight = context_cue_weight(overlap)
        return round(base_score * weight, 4)
    except Exception:
        return base_score


# ── iter404：Semantic Priming — Spreading Activation with Temporal Decay
#             （Collins & Loftus 1975 / Meyer & Schvaneveldt 1971）────────────
#
# 认知科学依据：
#   Collins & Loftus (1975) Spreading Activation Theory:
#     语义网络中，激活从当前节点沿关联链向邻居扩散，
#     扩散强度随网络距离衰减（activation × confidence^hops）。
#     时间维度：启动效应持续约数十分钟（Meyer & Schvaneveldt 1971），
#     随时间指数衰减（prime_strength × exp(-λ × t)）。
#   Meyer & Schvaneveldt (1971) Semantic Priming:
#     "Bread"→"Butter" 反应更快（短暂语义启动），
#     但 "Bread"→"Doctor" 无启动（无语义关联）。
#   Anderson (1983) ACT*:
#     工作记忆中活跃的概念持续向关联记忆扩散激活，
#     扩散在短暂窗口（~30分钟）内有效，之后衰减到基线。
#
# OS 类比：Linux page readahead（ra_state + readahead window）
#   顺序访问 file pages 时，内核维护 readahead window；
#   访问 page N → prefetch [N+1, N+ra_size] 进 page cache；
#   类比：检索 chunk A → prime chunk A 的相关 entities → 后续相关 chunk 有 cache 优势。
#   prime_half_life ≈ ra_lookahead_time：超过此时间，预取失效（evicted from cache）。
#
# 实现：
#   prime_entities(conn, entity_names, project, prime_strength=1.0)
#     写入/更新 priming_state 表（upsert，取 max(existing, new) strength）
#   get_active_primes(conn, project, now_iso=None) → {entity_name: current_strength}
#     读取 priming_state，按时间衰减计算当前强度（>0.05 才算 active）
#   compute_priming_boost(conn, chunk_id, project, now_iso=None) → float [0.0, 0.30]
#     通过 entity_map 找到 chunk 关联 entity，查询当前 prime 强度，返回 boost
#   clear_stale_primes(conn, project, min_strength=0.05)
#     清理已衰减到阈值以下的 priming 条目（GC）

import math as _math_priming

_PRIME_HALF_LIFE_MINUTES: float = 30.0   # 启动效应半衰期（分钟）
_PRIME_MAX_BOOST: float = 0.30           # 最大启动加成
_PRIME_MIN_STRENGTH: float = 0.05        # 低于此强度视为已失效


def prime_entities(
    conn: sqlite3.Connection,
    entity_names: list,
    project: str,
    prime_strength: float = 1.0,
    now_iso: str = None,
) -> int:
    """
    iter404：将一组 entity 写入 priming_state（启动它们）。

    OS 类比：readahead_cache_miss_trigger() — 缺页触发预取，将邻居 pages 标记为 readahead。

    Returns:
      int：实际写入/更新的 entity 数量
    """
    if not entity_names or not project:
        return 0
    if now_iso is None:
        now_iso = datetime.now(timezone.utc).isoformat()
    prime_strength = max(0.0, min(1.0, float(prime_strength)))
    count = 0
    try:
        for ent in entity_names:
            if not ent or not ent.strip():
                continue
            ent = ent.strip()
            # 若已有 prime 且更强，保留更强的（取 max）
            existing = conn.execute(
                "SELECT prime_strength, primed_at FROM priming_state "
                "WHERE entity_name=? AND project=?",
                (ent, project)
            ).fetchone()
            if existing:
                # 计算 existing 的当前有效强度（已衰减）
                _ex_strength = existing[0]
                try:
                    _ex_ts = datetime.fromisoformat(existing[1].replace("Z", "+00:00")).timestamp()
                    _now_ts = datetime.fromisoformat(now_iso.replace("Z", "+00:00")).timestamp()
                    _elapsed_min = (_now_ts - _ex_ts) / 60.0
                    _lambda = _math_priming.log(2) / _PRIME_HALF_LIFE_MINUTES
                    _current = _ex_strength * _math_priming.exp(-_lambda * _elapsed_min)
                except Exception:
                    _current = 0.0
                if prime_strength > _current:
                    conn.execute(
                        "UPDATE priming_state SET prime_strength=?, primed_at=? "
                        "WHERE entity_name=? AND project=?",
                        (prime_strength, now_iso, ent, project)
                    )
            else:
                conn.execute(
                    "INSERT INTO priming_state (entity_name, project, primed_at, prime_strength) "
                    "VALUES (?, ?, ?, ?)",
                    (ent, project, now_iso, prime_strength)
                )
            count += 1
    except Exception:
        pass
    return count


def get_active_primes(
    conn: sqlite3.Connection,
    project: str,
    now_iso: str = None,
    min_strength: float = _PRIME_MIN_STRENGTH,
) -> dict:
    """
    iter404：返回 project 中当前活跃的 entity → current_prime_strength 映射。

    OS 类比：readahead_state.ra_pages — 返回当前 readahead window 中仍有效的 pages。

    Returns:
      {entity_name: current_strength}，只包含 current_strength > min_strength 的条目
    """
    if not project:
        return {}
    if now_iso is None:
        now_iso = datetime.now(timezone.utc).isoformat()
    try:
        _now_ts = datetime.fromisoformat(now_iso.replace("Z", "+00:00")).timestamp()
    except Exception:
        return {}
    _lambda = _math_priming.log(2) / _PRIME_HALF_LIFE_MINUTES
    result = {}
    try:
        rows = conn.execute(
            "SELECT entity_name, prime_strength, primed_at FROM priming_state WHERE project=?",
            (project,)
        ).fetchall()
        for ent, strength, primed_at in rows:
            if not ent or not strength:
                continue
            try:
                _primed_ts = datetime.fromisoformat(primed_at.replace("Z", "+00:00")).timestamp()
                _elapsed_min = (_now_ts - _primed_ts) / 60.0
                current = float(strength) * _math_priming.exp(-_lambda * _elapsed_min)
            except Exception:
                current = 0.0
            if current > min_strength:
                result[ent] = round(current, 4)
    except Exception:
        pass
    return result


def compute_priming_boost(
    conn: sqlite3.Connection,
    chunk_id: str,
    project: str,
    now_iso: str = None,
) -> float:
    """
    iter404：计算 chunk 当前受到的语义启动加成（semantic priming boost）。

    算法：
      1. 通过 encode_context 找到 chunk 的 entity/keyword 集合
      2. 与 active primes 取交集
      3. boost = avg(matching prime strengths) × _PRIME_MAX_BOOST

    OS 类比：readahead_cache_hit() — 访问的 page 在 readahead window 内 → cache hit，
      节省一次 disk I/O（类比：primed entity match → 检索 score 提升）。

    Returns:
      float ∈ [0.0, _PRIME_MAX_BOOST]
    """
    if not chunk_id or not project:
        return 0.0
    try:
        # 获取当前活跃 primes
        active_primes = get_active_primes(conn, project, now_iso=now_iso)
        if not active_primes:
            return 0.0

        # 获取 chunk 的 encode_context（关键词集合）
        row = conn.execute(
            "SELECT encode_context FROM memory_chunks WHERE id=?", (chunk_id,)
        ).fetchone()
        if row is None or not row[0]:
            return 0.0

        chunk_tokens = set(t.strip() for t in row[0].split(",") if t.strip())
        if not chunk_tokens:
            return 0.0

        # 匹配 prime entities 与 chunk tokens（entity name 子串匹配或精确匹配）
        matching_strengths = []
        for prime_ent, strength in active_primes.items():
            prime_lower = prime_ent.lower()
            # 精确匹配 OR 子串匹配（entity "redis" 匹配 token "redis-cluster"）
            if prime_lower in chunk_tokens or any(
                prime_lower in tok or tok in prime_lower
                for tok in chunk_tokens
                if len(tok) >= 3
            ):
                matching_strengths.append(strength)

        if not matching_strengths:
            return 0.0

        avg_strength = sum(matching_strengths) / len(matching_strengths)
        boost = round(avg_strength * _PRIME_MAX_BOOST, 4)
        return min(_PRIME_MAX_BOOST, max(0.0, boost))
    except Exception:
        return 0.0


def clear_stale_primes(
    conn: sqlite3.Connection,
    project: str = None,
    min_strength: float = _PRIME_MIN_STRENGTH,
    now_iso: str = None,
) -> int:
    """
    iter404：清理已衰减到阈值以下的 priming 条目（GC）。

    OS 类比：invalidate_readahead_pages() — 清理 readahead window 中已过期的预取 pages。

    Returns:
      int：删除的条目数
    """
    if now_iso is None:
        now_iso = datetime.now(timezone.utc).isoformat()
    try:
        _now_ts = datetime.fromisoformat(now_iso.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0
    _lambda = _math_priming.log(2) / _PRIME_HALF_LIFE_MINUTES
    # 计算在 min_strength 时的最大有效时间（分钟）
    # min_strength = 1.0 × exp(-λ × t_max) → t_max = -ln(min_strength) / λ
    try:
        t_max_min = -_math_priming.log(min_strength) / _lambda  # minutes
        cutoff_ts = _now_ts - t_max_min * 60
        cutoff_iso = datetime.fromtimestamp(cutoff_ts, tz=timezone.utc).isoformat()
    except Exception:
        return 0

    try:
        where = "WHERE primed_at < ?"
        params = [cutoff_iso]
        if project is not None:
            where += " AND project=?"
            params.append(project)
        cursor = conn.execute(f"DELETE FROM priming_state {where}", params)
        return cursor.rowcount
    except Exception:
        return 0


# ── iter405：Retroactive Interference (RI) — Recency Penalty for Stale Chunks
#             （Underwood 1957 / McGeoch & Irion 1952）──────────────────────────
#
# 认知科学依据：
#   Underwood (1957) Proactive Inhibition and Forgetting:
#     在学习 List B 之后，回忆 List A 的成功率下降（retroactive interference）。
#     新记忆和旧记忆在同一语义领域竞争检索路径，新的倾向于"覆盖"旧的。
#   McGeoch & Irion (1952) The Psychology of Human Learning:
#     干扰效应强度 × 新旧材料的相似度；相似度越高，干扰越强。
#   Anderson & Neely (1996) Interference and Inhibition:
#     抑制（inhibition）是主动过程，不只是竞争失败的被动结果。
#
# OS 类比：Linux MGLRU generation demotion —
#   进入系统的新 page 从 youngest generation 开始；
#   老一代（older generation）的 page 在 aging scan 中随新 page 的涌入逐渐降代；
#   当内存紧张时，老一代 page 被优先驱逐（recency bias）。
#   chunk age 越大、同主题新 chunk 越多 → recency_penalty 越大 → 检索分下降。
#
# 实现：
#   compute_recency_penalty(chunk_age_days, newer_same_topic_count, similarity) → float [0.0, 0.15]
#     - chunk_age_days：chunk 的年龄（天数）
#     - newer_same_topic_count：同主题（encode_context 重叠）且更新的 chunk 数量
#     - similarity：encode_context Jaccard 与最近同主题 chunk 的平均重叠
#     返回检索分罚分（0.0 = 无干扰，0.15 = 最大干扰）
#
#   get_newer_same_topic_count(conn, chunk_id, project, overlap_threshold=0.30) → int
#     查找同 project 中更新、且 encode_context 与当前 chunk 高度重叠的 chunk 数量

_RI_MAX_PENALTY: float = 0.15     # 最大干扰罚分（降低检索分）
_RI_AGE_THRESHOLD_DAYS: float = 7.0  # 7 天以上的 chunk 才可能受 RI 影响
_RI_COUNT_SATURATION: int = 5     # 5 个以上新 chunk 后干扰饱和


def compute_recency_penalty(
    chunk_age_days: float,
    newer_same_topic_count: int,
    similarity: float = 0.5,
) -> float:
    """
    iter405：计算旧 chunk 因新内容涌入而受到的 retroactive interference 惩罚。

    公式：penalty = min(_RI_MAX_PENALTY,
                        age_factor × count_factor × similarity_factor)

      age_factor：年龄越大 → 越容易被干扰（line 0 to 1 over 30 days）
        age_factor = min(1.0, (age - threshold) / 30.0)  if age > threshold else 0.0
      count_factor：新 chunk 越多 → 干扰越强（saturate at _RI_COUNT_SATURATION）
        count_factor = min(1.0, newer_count / _RI_COUNT_SATURATION)
      similarity_factor：相似度越高 → 干扰越强
        similarity_factor = similarity

    设计：只有当 age > 7天、有 >= 1 个新 chunk 存在、且有一定相似度时才产生惩罚。

    OS 类比：MGLRU aging pressure = generation_age × page_count × access_recency

    Returns:
      float ∈ [0.0, _RI_MAX_PENALTY]
    """
    try:
        age = max(0.0, float(chunk_age_days))
        count = max(0, int(newer_same_topic_count))
        sim = max(0.0, min(1.0, float(similarity)))
    except (TypeError, ValueError):
        return 0.0

    if age <= _RI_AGE_THRESHOLD_DAYS or count == 0 or sim < 0.10:
        return 0.0

    age_factor = min(1.0, (age - _RI_AGE_THRESHOLD_DAYS) / 30.0)
    count_factor = min(1.0, count / _RI_COUNT_SATURATION)
    penalty = age_factor * count_factor * sim * _RI_MAX_PENALTY
    return round(min(_RI_MAX_PENALTY, max(0.0, penalty)), 4)


def get_newer_same_topic_count(
    conn: sqlite3.Connection,
    chunk_id: str,
    project: str,
    overlap_threshold: float = 0.25,
    now_iso: str = None,
) -> tuple:
    """
    iter405：查找同 project 中比当前 chunk 更新、且 encode_context 重叠度 >= threshold 的 chunk 数量。

    OS 类比：mglru_scan_newer_pages() — 统计 younger generation 中的相关 pages 数量。

    Returns:
      (newer_count: int, avg_overlap: float)
    """
    if not chunk_id or not project:
        return 0, 0.0
    try:
        row = conn.execute(
            "SELECT created_at, encode_context FROM memory_chunks WHERE id=?",
            (chunk_id,)
        ).fetchone()
        if row is None or not row[0]:
            return 0, 0.0
        chunk_created_at, chunk_enc_ctx = row[0], row[1] or ""
        if not chunk_enc_ctx:
            return 0, 0.0

        chunk_tokens = set(t.strip() for t in chunk_enc_ctx.split(",") if t.strip())
        if not chunk_tokens:
            return 0, 0.0

        # 查找更新的 chunk（created_at > chunk_created_at）
        newer_rows = conn.execute(
            "SELECT encode_context FROM memory_chunks "
            "WHERE project=? AND id != ? AND created_at > ? AND encode_context IS NOT NULL",
            (project, chunk_id, chunk_created_at)
        ).fetchall()

        if not newer_rows:
            return 0, 0.0

        overlaps = []
        for (enc_ctx,) in newer_rows:
            if not enc_ctx:
                continue
            newer_tokens = set(t.strip() for t in enc_ctx.split(",") if t.strip())
            if not newer_tokens:
                continue
            union = len(chunk_tokens | newer_tokens)
            if union > 0:
                jaccard = len(chunk_tokens & newer_tokens) / union
                if jaccard >= overlap_threshold:
                    overlaps.append(jaccard)

        count = len(overlaps)
        avg_overlap = sum(overlaps) / count if count > 0 else 0.0
        return count, round(avg_overlap, 4)
    except Exception:
        return 0, 0.0


# ── iter406：Generation Effect — 自生成内容 stability 加成（McDaniel & Einstein 1986）
#            ────────────────────────────────────────────────────────────────────────────
#
# 认知科学依据：
#   Slamecka & Graf (1978) The Generation Effect: Delineation of a Phenomenon:
#     被试自己生成的词汇（相对于阅读词汇）回忆率高 20-50%，即"生成效应"。
#     生成行为本身（无论结果）强化了记忆痕迹。
#   McDaniel & Einstein (1986) Bizarre Imagery as an Effective Memory Aid:
#     生成效应与精细阐述（elaboration）协同——不只是"生成"，
#     而是"在主动建构意义的过程中生成"，是决定 stability 的关键因子。
#   Jacoby (1978) On Interpreting the Effects of Repetition:
#     主动加工（active processing）vs 被动加工（passive processing）：
#     主动生成：推理、假设检验、类比构建 → 记忆强度更高
#     被动接受：直接复制、引用、简单整理 → 记忆强度较低
#
# OS 类比：Linux Write-Allocate 缓存策略 (write-allocate + write-back, 1974)
#   Write-Allocate（写分配）：CPU 写 miss 时，将整个 cache line 从 DRAM 读入，
#     在 cache 中修改后标记 dirty，等 writeback 时才写回 DRAM。
#     效果：写入触发完整 cache line 的加载和激活，该 line 进入 active 状态，
#     后续访问命中率显著提升（vs Write-No-Allocate 直写穿透）。
#   类比：agent 主动生成的内容相当于触发 Write-Allocate——
#     不只是被动写入（Write-No-Allocate），而是在生成过程中激活并构建完整 cache line；
#     生成标记密度越高 → Write-Allocate 程度越高 → 初始 stability 越高。
#
# 与 iter392（type-based）和 iter401（structural depth）的区别：
#   iter392：基于 chunk_type（reasoning_chain/decision/causal_chain）的粗粒度加成
#   iter401：基于结构性标记（因果词、对比词、层级结构）的深度加工检测
#   iter406：基于词汇层面的"主动生成标记"密度——
#     推理人称（"我认为"/"因此"/"我的理解是"）
#     假设检验（"如果...那么"/"假设"/"验证"）
#     元认知（"这说明"/"这意味着"/"关键在于"）
#     这三类标记直接指示了 agent 处于"主动建构"状态，而非"被动整理"状态。
#
# 实现：
#   compute_generation_score(content, summary, source_type) → float [0.0, 1.0]
#     检测内容中"主动生成"词汇标记密度，返回 generation score。
#   generation_stability_bonus(generation_score, base_stability) → float
#     将 generation score 映射为 stability 增量：
#       score >= 0.7 → bonus = base × 0.35（强生成，类比 Slamecka: +50%）
#       score 0.4-0.7 → bonus = base × 0.15（中等生成）
#       score 0.2-0.4 → bonus = base × 0.05（弱生成信号）
#       score < 0.2 → bonus = 0（无生成标记，被动内容）
#   apply_generation_effect(conn, chunk_id, content, summary, source_type, base_stability)
#     查找、计算并写入 stability 更新

# ── 生成标记词典（三层：推理人称 / 假设检验 / 元认知）──
# 中英文各有独立的识别规则
_GEN_REASONING_PERSON_ZH = frozenset([
    "我认为", "我觉得", "我的理解", "我推断", "我判断", "我估计",
    "在我看来", "据我分析", "从我的角度", "基于上述",
])
_GEN_REASONING_PERSON_EN = frozenset([
    "i think", "i believe", "i infer", "in my view", "as i see it",
    "i conclude", "i estimate", "my understanding", "i reason",
])
_GEN_HYPOTHETICAL_ZH = frozenset([
    "如果", "假设", "假如", "倘若", "若", "要是",
    "验证", "检验", "测试下", "实验", "推测",
])
_GEN_HYPOTHETICAL_EN = frozenset([
    "if we", "suppose", "hypothesis", "assuming", "let's verify", "let me check",
    "hypothetically", "let's test", "what if", "assume that",
])
_GEN_METACOG_ZH = frozenset([
    "这说明", "这意味着", "关键在于", "核心是", "本质是",
    "因此可以", "由此得出", "综上所述", "总结来看", "换句话说",
    "值得注意", "需要强调", "重要的是", "这表明", "这证明",
])
_GEN_METACOG_EN = frozenset([
    "this means", "therefore", "thus", "hence", "this implies",
    "in summary", "in conclusion", "the key insight", "this suggests",
    "it follows that", "importantly", "crucially", "this demonstrates",
    "as a result", "consequently",
])

# 最小内容长度（太短的内容不做生成检测，避免噪音）
_GEN_MIN_CHARS: int = 30
# 生成分层阈值
_GEN_STRONG_THRESHOLD: float = 0.7
_GEN_MEDIUM_THRESHOLD: float = 0.4
_GEN_WEAK_THRESHOLD: float = 0.2
# 最大稳定性加成（生成效应增量上限）
_GEN_MAX_STABILITY_BONUS_FACTOR: float = 0.35  # base × 0.35，即最多 +35%


def compute_generation_score(
    content: str,
    summary: str = "",
    source_type: str = None,
) -> float:
    """
    iter406：计算内容的"主动生成"标记密度，返回 generation score ∈ [0.0, 1.0]。

    检测三类生成标记：
      1. 推理人称（agent 以第一人称推理）
      2. 假设/验证（agent 主动构建假设并检验）
      3. 元认知（agent 反思、总结、得出结论）

    source_type 快速路径：
      "direct"（直接人类输入）→ 0.0（非生成内容，被动接收）
      "tool_output"（工具输出）→ 0.1 cap（工具输出为主，agent 生成为辅）
      None/"inferred"/"hearsay" → 正常检测

    Returns:
      float ∈ [0.0, 1.0]
    """
    if not content and not summary:
        return 0.0

    # source_type 快速判断
    if source_type == "direct":
        return 0.0  # 直接人类输入：非生成内容
    cap = 1.0
    if source_type == "tool_output":
        cap = 0.1  # 工具输出：agent 生成成分极少

    text = ((content or "") + " " + (summary or "")).lower().strip()
    if len(text) < _GEN_MIN_CHARS:
        return 0.0

    # 统计各类标记命中数
    reasoning_hits = sum(1 for m in _GEN_REASONING_PERSON_ZH if m in text)
    reasoning_hits += sum(1 for m in _GEN_REASONING_PERSON_EN if m in text)
    hypo_hits = sum(1 for m in _GEN_HYPOTHETICAL_ZH if m in text)
    hypo_hits += sum(1 for m in _GEN_HYPOTHETICAL_EN if m in text)
    meta_hits = sum(1 for m in _GEN_METACOG_ZH if m in text)
    meta_hits += sum(1 for m in _GEN_METACOG_EN if m in text)

    # 分层权重：元认知 > 推理人称 > 假设（从确定度排序）
    # 元认知标记代表 agent 已得出结论，生成效应最强
    # 推理人称代表 agent 正在推理，次之
    # 假设标记代表 agent 在探索，生成效应相对最弱
    # 归一化到 [0.0, 1.0]：每层最多贡献 1/3
    meta_contribution = min(1.0, meta_hits / 3.0) * 0.45
    reasoning_contribution = min(1.0, reasoning_hits / 2.0) * 0.35
    hypo_contribution = min(1.0, hypo_hits / 3.0) * 0.20

    raw_score = meta_contribution + reasoning_contribution + hypo_contribution
    return round(min(1.0, min(cap, raw_score)), 4)


def generation_stability_bonus(
    generation_score: float,
    base_stability: float,
) -> float:
    """
    iter406：将 generation score 映射为 stability 增量。

    设计原则（Slamecka & Graf 1978）：
      强生成（>= 0.7）：回忆率提升 ~50% → stability bonus = base × 0.35
      中等生成（0.4-0.7）：回忆率提升 ~15% → stability bonus = base × 0.15
      弱生成（0.2-0.4）：回忆率提升 ~5% → stability bonus = base × 0.05
      无生成（< 0.2）：0 增量

    上限保护：total stability 不超过 base × 1.5（防止叠加后 stability 爆炸）

    Returns:
      float — stability 增量（非绝对值，需加到 base_stability 上）
    """
    try:
        generation_score = float(generation_score)
        base_stability = float(base_stability)
    except (TypeError, ValueError):
        return 0.0
    if generation_score <= 0.0 or base_stability <= 0.0:
        return 0.0
    try:
        score = float(generation_score)
        base = float(base_stability)
    except (TypeError, ValueError):
        return 0.0

    if score >= _GEN_STRONG_THRESHOLD:
        factor = _GEN_MAX_STABILITY_BONUS_FACTOR
    elif score >= _GEN_MEDIUM_THRESHOLD:
        # 线性插值：[0.4, 0.7) → factor [0.05, 0.35]
        t = (score - _GEN_MEDIUM_THRESHOLD) / (_GEN_STRONG_THRESHOLD - _GEN_MEDIUM_THRESHOLD)
        factor = 0.05 + t * (_GEN_MAX_STABILITY_BONUS_FACTOR - 0.05)
    elif score >= _GEN_WEAK_THRESHOLD:
        # 弱生成：[0.2, 0.4) → factor [0.0, 0.05]
        t = (score - _GEN_WEAK_THRESHOLD) / (_GEN_MEDIUM_THRESHOLD - _GEN_WEAK_THRESHOLD)
        factor = t * 0.05
    else:
        return 0.0

    bonus = base * factor
    # 上限：total stability 不超过 base × 1.5
    max_bonus = base * 0.50
    return round(min(bonus, max_bonus), 4)


def apply_generation_effect(
    conn: sqlite3.Connection,
    chunk_id: str,
    content: str,
    summary: str = "",
    source_type: str = None,
    base_stability: float = 1.0,
) -> float:
    """
    iter406：计算生成效应并更新 chunk 的 stability。

    Returns:
      float — 更新后的 stability（= base + bonus）
    """
    if not chunk_id:
        return base_stability
    try:
        score = compute_generation_score(content, summary, source_type)
        bonus = generation_stability_bonus(score, base_stability)
        new_stability = base_stability + bonus
        if bonus > 0.001:
            conn.execute(
                "UPDATE memory_chunks SET stability=? WHERE id=?",
                (new_stability, chunk_id)
            )
        return new_stability
    except Exception:
        return base_stability


# ── iter407: Von Restorff Effect — 孤立 chunk 的 stability 加成（von Restorff 1933）──────
# 认知科学依据：von Restorff (1933) "Über die Wirkung von Bereichsbildungen im Spurenfeld"
#   在均匀背景中，孤立/突出的项目（与背景在质上不同）比普通项目保留率显著更高。
#   Klein & Saltz (1976): isolation 效应在语义上独特的项目中最强（不仅限于物理外观）。
#   Wallace (1965): 效应强度与孤立程度正相关（越独特 → 记忆越好）。
#
# OS 类比：Linux perf_event outlier detection / NUMA distant access warning
#   perf stat 输出中，远离均值的异常值被标记和报警；
#   NUMA 拓扑中，与主工作集差异大的内存地址访问触发 distant-node access penalty。
#   memory-os 类比：在语义空间中"孤立"的 chunk（与同项目邻居语义距离大）
#   → 稀有信息 → 更值得保留 → stability bonus。
#
# 认知机制：孤立效应（von Restorff）的神经机制是 LTP（长时程增强）差异激活：
#   孤立项目打破了神经激活的均匀背景，引发更强的海马体编码，形成更持久的突触权重。
#
# 实现策略：
#   encode_context（逗号分隔的关键词串）作为语义代理向量
#   Jaccard 相似度计算 chunk 与同项目邻居的平均语义相似度
#   isolation_score = 1.0 - avg_similarity（孤立度）
#   只在邻居数 >= 3 时计算（数据不足时保守处理，不给 bonus）

def _parse_ec_to_set(ctx_str: str) -> frozenset:
    """将 encode_context 字符串（逗号分隔）解析为词集合。"""
    if not ctx_str:
        return frozenset()
    return frozenset(w.strip().lower() for w in ctx_str.split(',') if w.strip())


def _jaccard_ec(a: frozenset, b: frozenset) -> float:
    """计算两个词集合的 Jaccard 相似度。"""
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union > 0 else 0.0


def compute_isolation_score(
    conn: sqlite3.Connection,
    chunk_id: str,
    project: str,
    context_window: int = 20,
    min_neighbors: int = 3,
) -> float:
    """
    iter407: 计算 chunk 在同项目中的语义孤立度（Von Restorff Effect）。

    孤立度 = 1.0 - 平均语义相似度（与最近 context_window 个邻居的平均 Jaccard）

    Args:
      conn: SQLite 连接
      chunk_id: 目标 chunk ID
      project: 项目标识
      context_window: 考察的邻居数量（最近创建的 N 个 chunk，排除自己）
      min_neighbors: 最少需要多少邻居才计算（< min 时返回 0.0，数据不足）

    Returns:
      float ∈ [0.0, 1.0]：孤立度。0.0=完全不孤立，1.0=完全孤立（无语义重叠）
    """
    if not chunk_id or not project:
        return 0.0
    try:
        # 获取目标 chunk 的 encode_context
        row = conn.execute(
            "SELECT encode_context FROM memory_chunks WHERE id=? AND project=?",
            (chunk_id, project)
        ).fetchone()
        if not row:
            return 0.0
        target_ctx = _parse_ec_to_set(row[0] or "")
        if not target_ctx:
            # 无 encode_context → 无法计算相似度，返回 0（保守处理）
            return 0.0

        # 获取同项目中最近的 context_window 个 chunk（排除自己）
        neighbors = conn.execute(
            """SELECT encode_context FROM memory_chunks
               WHERE project=? AND id != ? AND encode_context IS NOT NULL
                 AND encode_context != ''
               ORDER BY created_at DESC
               LIMIT ?""",
            (project, chunk_id, context_window)
        ).fetchall()

        if len(neighbors) < min_neighbors:
            return 0.0  # 数据不足，保守处理

        # 计算平均 Jaccard 相似度
        similarities = []
        for (nb_ctx_str,) in neighbors:
            nb_set = _parse_ec_to_set(nb_ctx_str or "")
            if nb_set:
                sim = _jaccard_ec(target_ctx, nb_set)
                similarities.append(sim)

        if not similarities:
            return 0.0

        avg_sim = sum(similarities) / len(similarities)
        isolation_score = max(0.0, 1.0 - avg_sim)
        return min(1.0, isolation_score)

    except Exception:
        return 0.0


def isolation_stability_bonus(
    isolation_score: float,
    base_stability: float,
) -> float:
    """
    iter407: Von Restorff Isolation Bonus — 孤立度越高 stability bonus 越大。

    设计（Wallace 1965 效应强度正相关于孤立程度）：
      isolation >= 0.85（极孤立）: factor = 0.20 → bonus = base × 0.20
      isolation [0.65, 0.85): linear interp 0.10 → 0.20
      isolation [0.45, 0.65): linear interp 0.00 → 0.10
      isolation < 0.45（不突出）: 0

    上限: base × 0.20（比 Generation Effect 小，因为孤立是被动属性，生成是主动行为）

    OS 类比：perf 异常值标记 — 异常程度越大，标记权重越高，优先级越高。
    """
    try:
        isolation_score = float(isolation_score)
        base_stability = float(base_stability)
    except (TypeError, ValueError):
        return 0.0
    if isolation_score <= 0.0 or base_stability <= 0.0:
        return 0.0

    _VRSTOFF_STRONG = 0.85
    _VRSTOFF_MED    = 0.65
    _VRSTOFF_WEAK   = 0.45
    _VRSTOFF_MAX_FACTOR = 0.20
    _VRSTOFF_MED_FACTOR = 0.10

    if isolation_score >= _VRSTOFF_STRONG:
        factor = _VRSTOFF_MAX_FACTOR
    elif isolation_score >= _VRSTOFF_MED:
        t = (isolation_score - _VRSTOFF_MED) / (_VRSTOFF_STRONG - _VRSTOFF_MED)
        factor = _VRSTOFF_MED_FACTOR + t * (_VRSTOFF_MAX_FACTOR - _VRSTOFF_MED_FACTOR)
    elif isolation_score >= _VRSTOFF_WEAK:
        t = (isolation_score - _VRSTOFF_WEAK) / (_VRSTOFF_MED - _VRSTOFF_WEAK)
        factor = 0.0 + t * _VRSTOFF_MED_FACTOR
    else:
        return 0.0

    bonus = base_stability * factor
    # 上限保护
    max_bonus = base_stability * _VRSTOFF_MAX_FACTOR
    return min(bonus, max_bonus)


def apply_isolation_effect(
    conn: sqlite3.Connection,
    chunk_id: str,
    project: str,
    base_stability: float = 1.0,
    context_window: int = 20,
) -> float:
    """
    iter407: 计算孤立效应并更新 chunk 的 stability。

    Returns:
      float — 更新后的 stability（= base + bonus）
    """
    if not chunk_id or not project:
        return base_stability
    try:
        isolation = compute_isolation_score(
            conn, chunk_id, project,
            context_window=context_window,
        )
        bonus = isolation_stability_bonus(isolation, base_stability)
        new_stability = base_stability + bonus
        if bonus > 0.001:
            conn.execute(
                "UPDATE memory_chunks SET stability=? WHERE id=?",
                (new_stability, chunk_id)
            )
        return new_stability
    except Exception:
        return base_stability


# ── iter408: Proactive Interference — 旧知识干扰新知识写入（Underwood 1957）─────
#
# 认知科学依据：
#   Underwood (1957) "Proactive Inhibition and Forgetting":
#     学习新材料时，先前学习的相似材料产生"前摄抑制"（Proactive Interference）。
#     已学材料越多、越相似 → 新材料的初始记忆强度越低。
#   Porter & Duncan (1953): PI 效应与已有材料的数量正相关。
#   Postman & Underwood (1973) interference theory review:
#     PI 是遗忘的主要机制之一（与 RI 并列）。
#
# 与 iter405 RI（Retroactive Interference）的对称性：
#   RI（iter405）: 新 chunk 写入后，旧 chunk 检索分降低（新干扰旧）
#   PI（iter408）: 旧 chunk 存在时，新 chunk 写入时 stability 降低（旧干扰新）
#
#   RI + PI = 完整的干扰理论（Miller 1956 双向干扰）
#
# OS 类比：Linux TLB Shootdown Cost
#   修改一个被多核共享的 PTE（page table entry）时，内核必须向持有该
#   TLB entry 的所有 CPU 发送 IPI（inter-processor interrupt），强制
#   其 flush TLB（TLB shootdown）。共享该 PTE 的 CPU 越多，shootdown 开销越大。
#   类比：与新 chunk 语义重叠的旧 chunk 越多、越"活跃"（高 access_count），
#   新 chunk 写入时面临的"认知阻力"越大 → initial stability 越低。
#
# 实现：
#   compute_pi_penalty(conn, chunk_id, project) → float [0.0, 0.10]
#     1. 找最近邻居（最相似的 search_k=5 个 chunk）
#     2. 计算平均 Jaccard 相似度 avg_sim
#     3. 统计其中 access_count >= strong_acc_threshold 的"强旧记忆"数量
#     4. penalty = avg_sim × (strong_count / search_k) × max_penalty
#   apply_proactive_interference(conn, chunk_id, project, base_stability)
#     计算 penalty，更新 DB stability
#
# 保护规则：
#   1. design_constraint 类型豁免（约束永远应被记住）
#   2. 新 chunk 高 importance (> 0.85) → penalty 减半
#   3. 最大 penalty = base × 0.10（保守，避免过度惩罚新知识）


def compute_pi_penalty(
    conn: sqlite3.Connection,
    chunk_id: str,
    project: str,
    search_k: int = 5,
    strong_acc_threshold: int = 3,
    max_penalty: float = 0.10,
) -> float:
    """
    iter408: 计算新 chunk 面临的 Proactive Interference 惩罚系数。

    Returns:
      float ∈ [0.0, max_penalty] — PI 导致的 stability 降低量（非比例，绝对值）
      返回值直接从 base_stability 中减去。

    公式：
      avg_sim = 与最近 search_k 个邻居的平均 Jaccard 相似度
      strong_ratio = 其中 access_count >= threshold 的邻居比例
      penalty = avg_sim × strong_ratio × max_penalty
    """
    if not chunk_id or not project:
        return 0.0
    try:
        row = conn.execute(
            "SELECT encode_context FROM memory_chunks WHERE id=? AND project=?",
            (chunk_id, project)
        ).fetchone()
        if not row:
            return 0.0
        target_ctx = _parse_ec_to_set(row[0] or "")
        if not target_ctx:
            return 0.0

        # 找最近的 search_k 个邻居（不含自身）
        neighbors = conn.execute(
            """SELECT id, encode_context, access_count
               FROM memory_chunks
               WHERE project=? AND id != ?
                 AND encode_context IS NOT NULL AND encode_context != ''
               ORDER BY created_at DESC
               LIMIT ?""",
            (project, chunk_id, search_k)
        ).fetchall()

        if not neighbors:
            return 0.0

        similarities = []
        strong_count = 0
        for nb_row in neighbors:
            nb_id = nb_row[0] if isinstance(nb_row, (list, tuple)) else nb_row["id"]
            nb_ctx_str = nb_row[1] if isinstance(nb_row, (list, tuple)) else nb_row["encode_context"]
            nb_acc = nb_row[2] if isinstance(nb_row, (list, tuple)) else nb_row["access_count"]
            nb_ctx = _parse_ec_to_set(nb_ctx_str or "")
            if nb_ctx:
                sim = _jaccard_ec(target_ctx, nb_ctx)
                similarities.append(sim)
                if sim > 0.0 and (nb_acc or 0) >= strong_acc_threshold:
                    strong_count += 1

        if not similarities:
            return 0.0

        avg_sim = sum(similarities) / len(similarities)
        strong_ratio = strong_count / len(similarities)
        penalty = avg_sim * strong_ratio * max_penalty
        return min(penalty, max_penalty)
    except Exception:
        return 0.0


def apply_proactive_interference(
    conn: sqlite3.Connection,
    chunk_id: str,
    project: str,
    base_stability: float = 1.0,
    search_k: int = 5,
    strong_acc_threshold: int = 3,
    max_penalty: float = 0.10,
) -> float:
    """
    iter408: 计算 PI 惩罚并更新 chunk 的 stability。

    保护规则：
      - design_constraint 豁免（永久知识不受 PI）
      - high importance (>0.85) → penalty 减半
      - penalty < 0.001 → 跳过 DB 写入

    Returns:
      float — 更新后的 stability
    """
    if not chunk_id or not project:
        return base_stability
    try:
        # 检查保护条件
        row = conn.execute(
            "SELECT chunk_type, importance FROM memory_chunks WHERE id=?",
            (chunk_id,)
        ).fetchone()
        if not row:
            return base_stability

        chunk_type = row[0] if isinstance(row, (list, tuple)) else row["chunk_type"]
        importance = row[1] if isinstance(row, (list, tuple)) else row["importance"]

        # design_constraint 豁免
        if chunk_type == "design_constraint":
            return base_stability

        penalty = compute_pi_penalty(
            conn, chunk_id, project,
            search_k=search_k,
            strong_acc_threshold=strong_acc_threshold,
            max_penalty=max_penalty,
        )

        # 高 importance 新知识抗 PI（减半惩罚）
        if (importance or 0.0) > 0.85:
            penalty *= 0.5

        new_stability = max(0.1, base_stability - penalty)
        if penalty > 0.001:
            conn.execute(
                "UPDATE memory_chunks SET stability=? WHERE id=?",
                (new_stability, chunk_id)
            )
        return new_stability
    except Exception:
        return base_stability


# ── iter409: Flashbulb Memory — 情绪性内容的写入时 stability 加强（Brown & Kulik 1977）
#
# 认知科学依据：
#   Brown & Kulik (1977) "Flashbulb memories":
#     高情绪唤醒事件（例如 JFK 遇刺）形成极其鲜明、持久、细节丰富的记忆。
#     与普通记忆相比，flashbulb 记忆的消退曲线更平缓。
#   McGaugh (2000) "Memory — a century of consolidation" (Science):
#     情绪唤醒触发杏仁核激活 → norepinephrine 释放 → 增强海马编码强度（amygdala modulation）。
#   Cahill et al. (1994, Nature): β-肾上腺素受体阻断剂阻断了情绪增强效应
#     → 直接神经生理证据支持 norepinephrine 机制。
#
# 与 iter376 的区别：
#   iter376（emotional_boost_factor）：retrieval 时加分——情绪 chunk 更容易被检索到
#   iter409（flashbulb_stability_bonus）：insert 时加强——情绪 chunk 的初始 stability 更高
#   → iter376 是检索优先级，iter409 是记忆固化强度，互补而非冗余
#
# OS 类比：Linux mlockall(MCL_CURRENT | MCL_FUTURE)
#   高优先级进程调用 mlockall 将所有（当前和未来）内存页锁定在 RAM 中，
#   无法被 kswapd 驱逐。情绪性记忆 = 被 mlockall 的内存 = 衰减抵抗力最强。
#
# 实现：
#   flashbulb_stability_bonus(emotional_weight, base_stability) → bonus
#     strong (≥ 0.70): +30% of base (cap: base × 0.30)
#     medium (0.50-0.70): interp 15→30%
#     weak (0.30-0.50): interp 0→15%
#     < 0.30: no bonus
#   apply_flashbulb_effect(conn, chunk_id, base_stability) → new_stability


def flashbulb_stability_bonus(emotional_weight: float, base_stability: float) -> float:
    """
    iter409: 将 emotional_weight 映射为 stability bonus（Brown & Kulik 1977）。

    设计（McGaugh 2000 杏仁核 norepinephrine 效应梯度）：
      strong (≥ 0.70): factor = 0.30（极强情绪唤醒，如系统崩溃/重大决策）
      medium [0.50, 0.70): 线性插值 0.15 → 0.30
      weak   [0.30, 0.50): 线性插值 0.00 → 0.15
      < 0.30: factor = 0（无情绪显著性，不加分）

    cap: base × 0.30（上限，防止 stability 异常膨胀）
    """
    try:
        emotional_weight = float(emotional_weight)
        base_stability = float(base_stability)
    except (TypeError, ValueError):
        return 0.0
    if emotional_weight <= 0.0 or base_stability <= 0.0:
        return 0.0

    _FB_STRONG = 0.70
    _FB_MEDIUM = 0.50
    _FB_WEAK   = 0.30
    _FB_MAX_FACTOR = 0.30
    _FB_MED_FACTOR = 0.15

    if emotional_weight >= _FB_STRONG:
        factor = _FB_MAX_FACTOR
    elif emotional_weight >= _FB_MEDIUM:
        t = (emotional_weight - _FB_MEDIUM) / (_FB_STRONG - _FB_MEDIUM)
        factor = _FB_MED_FACTOR + t * (_FB_MAX_FACTOR - _FB_MED_FACTOR)
    elif emotional_weight >= _FB_WEAK:
        t = (emotional_weight - _FB_WEAK) / (_FB_MEDIUM - _FB_WEAK)
        factor = 0.0 + t * _FB_MED_FACTOR
    else:
        return 0.0

    bonus = base_stability * factor
    max_bonus = base_stability * _FB_MAX_FACTOR
    return min(bonus, max_bonus)


def apply_flashbulb_effect(
    conn: sqlite3.Connection,
    chunk_id: str,
    base_stability: float = 1.0,
) -> float:
    """
    iter409: 读取 chunk 的 emotional_weight，计算 flashbulb bonus 并更新 stability。

    Returns:
      float — 更新后的 stability（= base + bonus）
    """
    if not chunk_id:
        return base_stability
    try:
        row = conn.execute(
            "SELECT emotional_weight FROM memory_chunks WHERE id=?",
            (chunk_id,)
        ).fetchone()
        if not row:
            return base_stability
        ew = row[0] if isinstance(row, (list, tuple)) else row["emotional_weight"]
        bonus = flashbulb_stability_bonus(ew or 0.0, base_stability)
        new_stability = base_stability + bonus
        if bonus > 0.001:
            conn.execute(
                "UPDATE memory_chunks SET stability=? WHERE id=?",
                (new_stability, chunk_id)
            )
        return new_stability
    except Exception:
        return base_stability


# ── iter410: Primacy Effect — 首位效应（Murdock 1962 Serial Position Effect）────
#
# 认知科学依据：
#   Murdock (1962) "The serial position effect of free recall" (JEP):
#     在一系列项目中，最早出现的项目（primacy）和最近出现的项目（recency）
#     记忆效果最好，中间项目最差。
#   Primacy Effect 机制（Rundus 1971）：
#     最早的项目在工作记忆中停留时间最长，被 rehearsed（复述）次数最多，
#     形成更强的长时记忆痕迹（elaborative rehearsal hypothesis）。
#   在工程知识场景中：
#     项目最初建立的 chunk（架构决策/设计约束/技术选型）是后续所有工作的
#     认知 schema（Bartlett 1932）。它们被参考、验证、依赖的次数最多，
#     相当于被 rehearsed 最多次。
#
# OS 类比：Linux boot-time kernel parameters
#   内核启动时通过 cmdline 设置的参数（如 hugepages=1024、pcie_aspm=off）
#   比 sysctl 运行时参数更持久：它们在所有子系统初始化之前就生效，
#   是系统的基础 schema，后续配置都在它之上构建。
#   对应：项目最早创建的 chunk = boot-time parameters = 基础 schema = 更持久。
#
# 实现约束：
#   1. min_total_chunks=20 阈值 — 项目少于 20 个 chunk 时不应用（避免新项目所有 chunk 都加成）
#   2. primacy_pct=0.10 — 最早的 10% 的 chunk 获得完整 primacy bonus
#   3. 延伸区间 [0.10, 0.20) — 线性衰减到 0
#   4. 上限 base × 0.15（保守，首位效应是相对效应）


def compute_primacy_rank(
    conn: sqlite3.Connection,
    chunk_id: str,
    project: str,
    min_total_chunks: int = 20,
) -> float:
    """
    iter410: 计算 chunk 在项目中按创建时间排名的百分位 [0.0, 1.0]。

    0.0 = 最早创建（primacy 最强），1.0 = 最晚创建。
    若项目 chunk 总数 < min_total_chunks，返回 1.0（不触发 primacy 加成）。
    """
    if not chunk_id or not project:
        return 1.0
    try:
        total = conn.execute(
            "SELECT COUNT(*) FROM memory_chunks WHERE project=?", (project,)
        ).fetchone()[0]
        if total < min_total_chunks:
            return 1.0  # 项目还太小，不应用首位效应

        # 获取 chunk 的创建时间排名（升序）
        rank_row = conn.execute(
            """SELECT COUNT(*) FROM memory_chunks
               WHERE project=? AND created_at < (
                   SELECT created_at FROM memory_chunks WHERE id=? AND project=?
               )""",
            (project, chunk_id, project)
        ).fetchone()
        if not rank_row:
            return 1.0
        rank = rank_row[0]  # 比该 chunk 更早的 chunk 数量（0-based）
        return rank / total  # 百分位：0.0 = 最早
    except Exception:
        return 1.0


def primacy_stability_bonus(primacy_rank: float, base_stability: float) -> float:
    """
    iter410: 将 primacy_rank 映射为 stability 加成（首位效应）。

    设计（Rundus 1971 rehearsal hypothesis）：
      rank < 0.10（最早 10%）: factor = 0.15（完整首位加成）
      rank [0.10, 0.20)：线性衰减 0.15 → 0.0
      rank ≥ 0.20：factor = 0（不在首位区间）

    cap: base × 0.15
    """
    try:
        primacy_rank = float(primacy_rank)
        base_stability = float(base_stability)
    except (TypeError, ValueError):
        return 0.0
    if base_stability <= 0.0:
        return 0.0

    _PRIMACY_CORE = 0.10    # 最早 10% 获得完整加成
    _PRIMACY_TAIL = 0.20    # 10-20% 线性衰减
    _PRIMACY_MAX_FACTOR = 0.15

    if primacy_rank < _PRIMACY_CORE:
        factor = _PRIMACY_MAX_FACTOR
    elif primacy_rank < _PRIMACY_TAIL:
        t = 1.0 - (primacy_rank - _PRIMACY_CORE) / (_PRIMACY_TAIL - _PRIMACY_CORE)
        factor = t * _PRIMACY_MAX_FACTOR
    else:
        return 0.0

    bonus = base_stability * factor
    max_bonus = base_stability * _PRIMACY_MAX_FACTOR
    return min(bonus, max_bonus)


def apply_primacy_effect(
    conn: sqlite3.Connection,
    chunk_id: str,
    project: str,
    base_stability: float = 1.0,
    min_total_chunks: int = 20,
) -> float:
    """
    iter410: 计算首位效应并更新 chunk 的 stability。

    Returns:
      float — 更新后的 stability（= base + primacy_bonus）
    """
    if not chunk_id or not project:
        return base_stability
    try:
        rank = compute_primacy_rank(conn, chunk_id, project, min_total_chunks=min_total_chunks)
        bonus = primacy_stability_bonus(rank, base_stability)
        new_stability = base_stability + bonus
        if bonus > 0.001:
            conn.execute(
                "UPDATE memory_chunks SET stability=? WHERE id=?",
                (new_stability, chunk_id)
            )
        return new_stability
    except Exception:
        return base_stability


# ── iter411: Levels of Processing — 编码深度效应（Craik & Lockhart 1972）─────────
#
# 认知科学依据：
#   Craik & Lockhart (1972) "Levels of processing: A framework for memory research" (JVLVB):
#     记忆强度由编码时的加工深度决定，而非存储容量：
#     - 浅层加工（phonological）：重复发音，不理解语义 → 弱记忆
#     - 中层加工（syntactic）：理解语法结构 → 中等记忆
#     - 深层加工（semantic）：与已有语义网络建立丰富联系 → 强记忆
#   Hyde & Jenkins (1973): 语义导向任务（"它是有生命的吗？"）比
#     结构导向任务（"它有字母 e 吗？"）产生更好的记忆保留。
#
#   在 memory-os 中，encode_context（逗号分隔实体列表）是语义网络节点密度的代理指标：
#     实体越多 → 与更多概念建立了语义联系 → 更深的加工 → 更强的记忆痕迹
#
# OS 类比：Linux NUMA-aware page allocation
#   NUMA-local 页面访问延迟最低（本地 memory bank），类比语义本地性：
#   与本项目语义网络连接越多（实体越多），访问该知识的"延迟"越低（检索更容易）。
#   深层加工 = NUMA-local allocation = 低访问延迟 = 更不容易被 swap out。
#
# 实现约束：
#   1. 基于 encode_context 实体数量（已有字段，轻量）
#   2. 非线性分级：8+→1.0, 5-7→0.7, 3-4→0.4, 1-2→0.1, 0→0.0
#   3. 加成上限 base × 0.15（保守，因实体数量是间接代理，不是直接测量）
#   4. 不依赖 DB 查询（纯函数），可用于写入前预计算


def compute_encoding_depth(encode_context: str) -> float:
    """
    iter411: 基于 encode_context 实体数量计算编码深度分 [0.0, 1.0]。

    分级（Hyde & Jenkins 1973 语义网络密度代理）：
      entity_count >= 8: depth = 1.0（极丰富语义网络，深层加工）
      entity_count 5-7:  depth = 0.7（丰富）
      entity_count 3-4:  depth = 0.4（中等）
      entity_count 1-2:  depth = 0.1（浅层）
      entity_count = 0:  depth = 0.0（无语义编码）
    """
    if not encode_context:
        return 0.0
    try:
        entities = [e.strip() for e in encode_context.split(',') if e.strip()]
        count = len(entities)
        if count == 0:
            return 0.0
        elif count <= 2:
            return 0.1
        elif count <= 4:
            return 0.4
        elif count <= 7:
            return 0.7
        else:
            return 1.0
    except Exception:
        return 0.0


def depth_stability_bonus(depth: float, base_stability: float) -> float:
    """
    iter411: 将编码深度分映射为 stability 加成。

    设计（Craik & Lockhart 1972 深度 → 保留强度）：
      depth >= 0.80: factor = 0.15（极深层加工）
      depth [0.50, 0.80): 线性插值 0.08 → 0.15
      depth [0.20, 0.50): 线性插值 0.00 → 0.08
      depth < 0.20: factor = 0（浅层/无加工）

    cap: base × 0.15（保守上限，间接代理指标）
    """
    try:
        depth = float(depth)
        base_stability = float(base_stability)
    except (TypeError, ValueError):
        return 0.0
    if depth <= 0.0 or base_stability <= 0.0:
        return 0.0

    _LOP_DEEP   = 0.80
    _LOP_MED    = 0.50
    _LOP_WEAK   = 0.20
    _LOP_MAX_FACTOR = 0.15
    _LOP_MED_FACTOR = 0.08

    if depth >= _LOP_DEEP:
        factor = _LOP_MAX_FACTOR
    elif depth >= _LOP_MED:
        t = (depth - _LOP_MED) / (_LOP_DEEP - _LOP_MED)
        factor = _LOP_MED_FACTOR + t * (_LOP_MAX_FACTOR - _LOP_MED_FACTOR)
    elif depth >= _LOP_WEAK:
        t = (depth - _LOP_WEAK) / (_LOP_MED - _LOP_WEAK)
        factor = 0.0 + t * _LOP_MED_FACTOR
    else:
        return 0.0

    bonus = base_stability * factor
    max_bonus = base_stability * _LOP_MAX_FACTOR
    return min(bonus, max_bonus)


def apply_depth_effect(
    conn: sqlite3.Connection,
    chunk_id: str,
    base_stability: float = 1.0,
) -> float:
    """
    iter411: 读取 chunk 的 encode_context，计算编码深度加成并更新 stability。

    Returns:
      float — 更新后的 stability（= base + depth_bonus）
    """
    if not chunk_id:
        return base_stability
    try:
        row = conn.execute(
            "SELECT encode_context FROM memory_chunks WHERE id=?",
            (chunk_id,)
        ).fetchone()
        if not row:
            return base_stability
        ec = row[0] if isinstance(row, (list, tuple)) else row["encode_context"]
        depth = compute_encoding_depth(ec or "")
        bonus = depth_stability_bonus(depth, base_stability)
        new_stability = base_stability + bonus
        if bonus > 0.001:
            conn.execute(
                "UPDATE memory_chunks SET stability=? WHERE id=?",
                (new_stability, chunk_id)
            )
        return new_stability
    except Exception:
        return base_stability


# ── iter414: Self-Reference Effect — 自我参照内容的记忆优势（Rogers et al. 1977）──
# 认知科学依据：Rogers et al. (1977), Symons & Johnson (1997 meta-analysis):
#   以"与自我相关"方式加工的信息比语义加工的记忆更强（+0.5 SD）。
# OS 类比：Linux process 自身页（stack/heap/text）在 TLB 中有最高局部性。

_SELF_REF_MARKERS = frozenset([
    "i ", "i'm", "i've", "i'll", "i'd", "we ", "we're", "we've", "we'll",
    "our ", "my ", "myself", "ourselves", "me ", "us ", "let me", "let's",
    # Chinese self-reference markers
    "我", "我们", "我的", "我们的", "自己",
])


def compute_self_reference_score(content: str, chunk_type: str = "") -> float:
    """
    iter414: 计算内容的自我参照分数 [0.0, 1.0]。

    检测内容中第一人称标记的密度，以及 agent 主动生成的 chunk 类型加成。

    Returns:
      float — 自我参照分数 [0.0, 1.0]
    """
    if not content:
        return 0.0
    try:
        content_lower = content.lower()
        # Count self-reference marker occurrences
        total_matches = 0
        for marker in _SELF_REF_MARKERS:
            # Simple substring count (fast, no regex)
            idx = 0
            while True:
                pos = content_lower.find(marker, idx)
                if pos == -1:
                    break
                total_matches += 1
                idx = pos + len(marker)

        # Normalize by content length (per 100 chars)
        content_words = max(1, len(content.split()))
        density = total_matches / content_words

        # chunk_type bonus: agent-generated types get extra self-reference weight
        type_bonus = 0.0
        if chunk_type in ("reasoning_chain", "decision", "causal_chain", "procedure"):
            type_bonus = 0.2  # agent's own reasoning = inherently self-referential

        raw_score = min(1.0, density * 5.0 + type_bonus)  # density 0.2 → raw=1.0 + bonus
        return min(1.0, raw_score)
    except Exception:
        return 0.0


def self_ref_stability_bonus(score: float, base_stability: float, bonus_cap: float = 0.25) -> float:
    """
    iter414: 根据自我参照分数计算 stability 加成。

    bonus = base × bonus_cap × score
    capped at base × bonus_cap

    Args:
      score: self-reference score [0.0, 1.0]
      base_stability: chunk 的基础 stability
      bonus_cap: 最大加成比例（默认 0.25 = base × 25%）

    Returns:
      float — stability 加成量
    """
    if score <= 0.0 or base_stability <= 0.0:
        return 0.0
    max_bonus = base_stability * bonus_cap
    return min(max_bonus, max_bonus * score)


def apply_self_reference_effect(
    conn: sqlite3.Connection,
    chunk_id: str,
    base_stability: float = 1.0,
) -> float:
    """
    iter414: 读取 chunk 内容，计算自我参照加成并更新 stability。

    Returns:
      float — 更新后的 stability
    """
    if not chunk_id:
        return base_stability
    import config as _config
    if not _config.get("store_vfs.self_ref_enabled"):
        return base_stability
    try:
        row = conn.execute(
            "SELECT content, chunk_type FROM memory_chunks WHERE id=?",
            (chunk_id,)
        ).fetchone()
        if not row:
            return base_stability
        content = row[0] if isinstance(row, (list, tuple)) else row["content"]
        chunk_type = row[1] if isinstance(row, (list, tuple)) else row["chunk_type"]
        bonus_cap = _config.get("store_vfs.self_ref_bonus_cap")
        score = compute_self_reference_score(content or "", chunk_type or "")
        bonus = self_ref_stability_bonus(score, base_stability, bonus_cap)
        new_stability = base_stability + bonus
        if bonus > 0.001:
            conn.execute(
                "UPDATE memory_chunks SET stability=? WHERE id=?",
                (new_stability, chunk_id)
            )
        return new_stability
    except Exception:
        return base_stability


# ── iter415: Encoding Variability — 多情境编码的记忆鲁棒性（Estes 1955）──────────
# 认知科学依据：多情境编码 → 更多检索线索 → retrieval robustness。
# OS 类比：共享库被 N 个进程引用 → 高引用计数 → 不易被 kswapd 驱逐。


def compute_context_enrichment(current_ec: str, original_ec_count: int) -> int:
    """
    iter415: 计算 encode_context 的富化程度（新增 token 数）。

    Returns:
      int — 超过原始 token 数的新增 token 数量（>= 0）
    """
    if not current_ec:
        return 0
    current_tokens = [t.strip() for t in current_ec.split(",") if t.strip()]
    enrichment = max(0, len(current_tokens) - original_ec_count)
    return enrichment


def encoding_variability_bonus(enrichment_count: int, base_stability: float,
                                scale: float = 0.05) -> float:
    """
    iter415: 根据 encode_context 富化程度计算 stability 加成。

    bonus = base × min(0.15, enrichment_count × scale)
    capped at base × 0.15

    Args:
      enrichment_count: 超过初始 token 数的新增 token 数量
      base_stability: 当前 stability
      scale: 每个新增 token 的加成系数（默认 0.05）

    Returns:
      float — stability 加成量
    """
    if enrichment_count <= 0 or base_stability <= 0.0:
        return 0.0
    max_factor = 0.15  # cap at base × 15%
    factor = min(max_factor, enrichment_count * scale)
    return base_stability * factor


def apply_encoding_variability(
    conn: sqlite3.Connection,
    chunk_id: str,
    current_stability: float = None,
) -> float:
    """
    iter415: 检查 encode_context 富化程度，给予 stability 加成。

    只在 update_accessed 时调用（不在 insert_chunk 时调用，因为初始状态无富化）。

    Returns:
      float — 更新后的 stability（如无富化则返回 current_stability）
    """
    if not chunk_id:
        return current_stability or 0.0
    import config as _config
    if not _config.get("store_vfs.encoding_variability_enabled"):
        return current_stability or 0.0
    try:
        row = conn.execute(
            "SELECT stability, encode_context, COALESCE(original_ec_count, 0) AS orig_count "
            "FROM memory_chunks WHERE id=?",
            (chunk_id,)
        ).fetchone()
        if not row:
            return current_stability or 0.0

        stab = float(row[0] if isinstance(row, (list, tuple)) else row["stability"]) or 1.0
        ec = row[1] if isinstance(row, (list, tuple)) else row["encode_context"]
        orig_count = int(row[2] if isinstance(row, (list, tuple)) else row["orig_count"])

        if current_stability is not None:
            stab = current_stability

        scale = _config.get("store_vfs.encoding_variability_scale")
        enrichment = compute_context_enrichment(ec or "", orig_count)
        bonus = encoding_variability_bonus(enrichment, stab, scale)
        new_stability = min(365.0, stab + bonus)
        if bonus > 0.001:
            conn.execute(
                "UPDATE memory_chunks SET stability=? WHERE id=?",
                (new_stability, chunk_id)
            )
        return new_stability
    except Exception:
        return current_stability or 0.0


# ── iter416: Zeigarnik Effect — 未完成任务的记忆优势（Zeigarnik 1927）──────────────
# 认知科学依据：Zeigarnik (1927) — 未完成任务 recall superiority ≈ +90% vs completed tasks。
#   Lewin (1935) Tension System Theory — 未完成任务维持认知系统"张力"，保持记忆激活。
# OS 类比：Linux futex waitqueue — pending I/O 保留在内核队列，不被 swapd 驱逐。

_ZEIGARNIK_MARKERS = frozenset([
    "todo", "fixme", "hack", "xxx", "wip", "pending", "unresolved",
    "incomplete", "not done", "need to", "needs to", "need to check",
    "investigate", "to be done", "to do", "follow up", "follow-up",
    "open issue", "open question", "tbd", "tbf", "tbr", "revisit",
    "blocked on", "waiting for", "in progress",
    # Chinese pending markers
    "待", "待确认", "待完成", "待处理", "未完成", "未解决", "需要确认",
    "需要调查", "跟进", "待跟进", "后续", "TODO", "FIXME",
])


def compute_zeigarnik_score(content: str, chunk_type: str = "") -> float:
    """
    iter416: 计算内容的 Zeigarnik 未完成任务分数 [0.0, 1.0]。

    检测内容中未完成任务信号词的存在，以及 task_state chunk_type 加成。

    Returns:
      float — Zeigarnik 分数 [0.0, 1.0]
    """
    if not content:
        return 0.0
    try:
        content_lower = content.lower()
        total_matches = 0
        for marker in _ZEIGARNIK_MARKERS:
            if marker.lower() in content_lower:
                total_matches += 1

        # Normalize: 1 match = 0.4, 2+ matches = higher, capped at 0.8 from content
        content_score = min(0.8, total_matches * 0.4) if total_matches > 0 else 0.0

        # chunk_type bonus: task_state chunks are inherently about pending tasks
        type_bonus = 0.0
        if chunk_type == "task_state":
            type_bonus = 0.2  # task_state = tracking incomplete workflow

        return min(1.0, content_score + type_bonus)
    except Exception:
        return 0.0


def zeigarnik_stability_bonus(score: float, base_stability: float,
                               bonus_cap: float = 0.20) -> float:
    """
    iter416: 根据 Zeigarnik score 计算 stability 加成。

    bonus = score × base × bonus_cap（线性比例，最大为 base × cap）
    """
    if score <= 0.0 or base_stability <= 0.0:
        return 0.0
    return min(base_stability * bonus_cap, score * base_stability * bonus_cap)


def apply_zeigarnik_effect(
    conn: sqlite3.Connection,
    chunk_id: str,
    base_stability: float = 1.0,
) -> float:
    """
    iter416: 检测 chunk 的未完成任务信号，给予 stability 加成。

    在 insert_chunk 管线中调用（Self-Reference Effect 之后）。

    Returns:
      float — 更新后的 stability
    """
    if not chunk_id:
        return base_stability
    import config as _config
    if not _config.get("store_vfs.zeigarnik_enabled"):
        return base_stability
    try:
        row = conn.execute(
            "SELECT content, chunk_type, stability FROM memory_chunks WHERE id=?",
            (chunk_id,)
        ).fetchone()
        if not row:
            return base_stability

        content = row[0] if isinstance(row, (list, tuple)) else row["content"]
        chunk_type = row[1] if isinstance(row, (list, tuple)) else row["chunk_type"]
        stab = float(row[2] if isinstance(row, (list, tuple)) else row["stability"]) or 1.0

        bonus_cap = _config.get("store_vfs.zeigarnik_bonus_cap")
        score = compute_zeigarnik_score(content or "", chunk_type or "")
        bonus = zeigarnik_stability_bonus(score, stab, bonus_cap)
        new_stability = min(365.0, stab + bonus)
        if bonus > 0.001:
            conn.execute(
                "UPDATE memory_chunks SET stability=? WHERE id=?",
                (new_stability, chunk_id)
            )
        return new_stability
    except Exception:
        return base_stability


# ── iter422: Permastore Memory — 充分强化后的记忆永久保护（Bahrick 1979）───────────────
# 认知科学依据：Bahrick (1979) Permastore — 充分暴露+高重要性的记忆达到"永久存储"状态：
#   即使经过数十年不复习，仍能保留约 80% 的可访问性（vs 普通记忆的完全遗忘）。
#   Conway et al. (1991): 专业知识（expert knowledge）具有 permastore 特征。
# 应用：满足条件的 chunk（age>=30d, access_count>=10, importance>=0.80）进入 permastore 状态；
#   RI/RIF/DF 对这些 chunk 只能将 stability 降低到 stability×floor_factor(0.80)，
#   而非普通的硬 floor=0.1，保护核心知识不被干扰效应过度压制。
# OS 类比：Linux mlock() + MADV_WILLNEED —
#   重要页面（内核代码、共享库 .text 段）mlock 锁定在 RAM，
#   即使系统内存极度紧张，kswapd 也无法驱逐这些页面（硬保护下限）。

def compute_permastore_floor(
    conn: sqlite3.Connection,
    chunk_id: str,
    current_stability: float,
) -> float:
    """
    iter422: 计算 chunk 的 stability 下限（Permastore Memory）。

    如果 chunk 满足 permastore 条件（age >= min_age_days, access_count >= min_acc,
    importance >= min_importance），返回 current_stability × floor_factor（> 普通 0.1）。
    否则返回普通 floor=0.1。

    在 RI/RIF/DF 函数中替代硬编码的 floor=0.1。

    Returns:
      float — 该 chunk 的 stability 下限
    """
    import config as _config
    if not _config.get("store_vfs.permastore_enabled"):
        return 0.1  # disabled: use normal floor
    try:
        min_age_days = _config.get("store_vfs.permastore_min_age_days")
        min_acc = _config.get("store_vfs.permastore_min_access_count")
        min_imp = _config.get("store_vfs.permastore_min_importance")
        floor_factor = _config.get("store_vfs.permastore_floor_factor")

        row = conn.execute(
            "SELECT created_at, access_count, importance FROM memory_chunks WHERE id=?",
            (chunk_id,)
        ).fetchone()
        if not row:
            return 0.1

        created_at = row[0] if isinstance(row, (list, tuple)) else row["created_at"]
        access_count = int(row[1] if isinstance(row, (list, tuple)) else row["access_count"]) or 0
        importance = float(row[2] if isinstance(row, (list, tuple)) else row["importance"]) or 0.0

        if not created_at:
            return 0.1

        # Compute age in days
        try:
            from datetime import datetime as _dt, timezone as _tz
            _created_ts = _dt.fromisoformat(created_at.replace("Z", "+00:00")).timestamp()
            _now_ts = _dt.now(_tz.utc).timestamp()
            age_days = (_now_ts - _created_ts) / 86400.0
        except Exception:
            return 0.1

        if age_days >= min_age_days and access_count >= min_acc and importance >= min_imp:
            # Permastore: floor is a fraction of current stability
            return max(0.1, current_stability * floor_factor)

        return 0.1
    except Exception:
        return 0.1


# ── iter417: Retrieval-Induced Forgetting — 检索引发的竞争性抑制（Anderson et al. 1994）──
# 认知科学依据：Anderson, Bjork & Bjork (1994) "Remembering can cause forgetting" —
#   检索一个记忆时主动抑制其语义竞争者（inhibitory tagging），
#   抑制强度 ∝ 语义相似度（高相似 = 强竞争 = 更多抑制）。
# OS 类比：MESI 缓存一致性协议 —
#   写入 Modified cache line → 其他核心的相同 cache line 变为 Invalid。
#   访问 chunk A → 其语义竞争者 B 的"有效性"下降（类比 cache invalidation）。


def apply_retrieval_induced_forgetting(
    conn: sqlite3.Connection,
    chunk_ids: list,
    project: str,
) -> int:
    """
    iter417: 对被检索 chunk 的语义竞争者施加轻微 stability 衰减。

    在 update_accessed 调用后，对未被检索但与检索 chunk 高度重叠的语义邻居
    施加 RIF 抑制（stability × decay_factor）。

    Args:
      conn: SQLite 连接
      chunk_ids: 本次被检索的 chunk ID 列表
      project: 项目 ID（限定 RIF 范围，跨项目不产生干扰）

    Returns:
      int — 受到 RIF 抑制的邻居数量
    """
    if not chunk_ids or not project:
        return 0
    import config as _config
    if not _config.get("store_vfs.rif_enabled"):
        return 0
    try:
        decay_factor = _config.get("store_vfs.rif_decay_factor")
        min_overlap = _config.get("store_vfs.rif_min_overlap")
        max_neighbors = _config.get("store_vfs.rif_max_neighbors")

        if decay_factor >= 1.0:
            return 0  # no-op if factor is 1.0

        # Get encode_context tokens for each accessed chunk
        placeholders = ",".join(["?"] * len(chunk_ids))
        acc_rows = conn.execute(
            f"SELECT id, encode_context FROM memory_chunks WHERE id IN ({placeholders})",
            chunk_ids,
        ).fetchall()

        # Collect all tokens from accessed chunks
        accessed_token_sets = {}
        for row in acc_rows:
            cid = row[0] if isinstance(row, (list, tuple)) else row["id"]
            ec = row[1] if isinstance(row, (list, tuple)) else row["encode_context"]
            tokens = frozenset(t.strip() for t in (ec or "").split(",") if t.strip())
            accessed_token_sets[cid] = tokens

        if not accessed_token_sets:
            return 0

        # Get candidate neighbors: same project, not in accessed set, has encode_context
        accessed_set = set(chunk_ids)
        candidates = conn.execute(
            "SELECT id, encode_context, stability FROM memory_chunks "
            "WHERE project=? AND encode_context IS NOT NULL AND stability > 0.1 "
            "AND id NOT IN ({})".format(",".join(["?"] * len(accessed_set))),
            [project] + list(accessed_set),
        ).fetchall()

        # Compute overlap for each candidate
        neighbor_overlaps = []
        for row in candidates:
            cid = row[0] if isinstance(row, (list, tuple)) else row["id"]
            ec = row[1] if isinstance(row, (list, tuple)) else row["encode_context"]
            stab = float(row[2] if isinstance(row, (list, tuple)) else row["stability"])
            c_tokens = frozenset(t.strip() for t in (ec or "").split(",") if t.strip())
            if not c_tokens:
                continue
            # Find max overlap with any accessed chunk
            max_overlap = max(
                len(c_tokens & acc_tokens)
                for acc_tokens in accessed_token_sets.values()
            )
            if max_overlap >= min_overlap:
                neighbor_overlaps.append((cid, stab, max_overlap))

        if not neighbor_overlaps:
            return 0

        # Sort by overlap descending, take top N
        neighbor_overlaps.sort(key=lambda x: -x[2])
        to_inhibit = neighbor_overlaps[:max_neighbors]

        # Apply RIF decay (iter422: permastore floor)
        inhibited = 0
        for n_cid, n_stab, _ in to_inhibit:
            _ps_floor = compute_permastore_floor(conn, n_cid, n_stab)
            new_stab = max(_ps_floor, n_stab * decay_factor)
            if abs(new_stab - n_stab) > 0.001:
                conn.execute(
                    "UPDATE memory_chunks SET stability=? WHERE id=?",
                    (new_stab, n_cid)
                )
                inhibited += 1
        return inhibited
    except Exception:
        return 0


# ── iter418: Directed Forgetting — 主动弃置过时知识（MacLeod 1998）──────────────
# 认知科学依据：MacLeod (1998) Directed Forgetting — 主动指令"忘记"使记忆加速衰减。
# OS 类比：Linux madvise(MADV_DONTNEED) — 通知内核不再需要该内存区域，加速回收。

_DIRECTED_FORGETTING_MARKERS = frozenset([
    "deprecated", "obsolete", "outdated", "old version", "replaced by",
    "superseded", "no longer", "not anymore", "was removed", "has been removed",
    "has been replaced", "legacy", "remove this", "to be removed", "will be removed",
    "already done", "completed", "resolved", "closed", "done", "finished",
    # Chinese deprecated markers
    "已废弃", "已过时", "已替换", "已完成", "已解决", "已关闭", "已删除",
    "不再使用", "替换为", "被替换", "旧版本",
])


def compute_directed_forgetting_score(content: str, chunk_type: str = "") -> float:
    """
    iter418: 计算内容的"主动遗忘"分数 [0.0, 1.0]。

    检测过时/已完成/已废弃信号词，返回应被主动弃置的程度。

    Returns:
      float — directed forgetting 分数 [0.0, 1.0]
    """
    if not content:
        return 0.0
    try:
        content_lower = content.lower()
        total_matches = 0
        for marker in _DIRECTED_FORGETTING_MARKERS:
            if marker.lower() in content_lower:
                total_matches += 1

        # 1 match = 0.5 score (significant signal), 2+ = capped at 1.0
        return min(1.0, total_matches * 0.5) if total_matches > 0 else 0.0
    except Exception:
        return 0.0


def directed_forgetting_penalty(score: float, base_stability: float,
                                 penalty_cap: float = 0.15) -> float:
    """
    iter418: 根据 directed forgetting score 计算 stability 惩罚量。

    penalty = score × base × penalty_cap（线性比例，最大为 base × cap）
    """
    if score <= 0.0 or base_stability <= 0.0:
        return 0.0
    return min(base_stability * penalty_cap, score * base_stability * penalty_cap)


def apply_directed_forgetting(
    conn: sqlite3.Connection,
    chunk_id: str,
    base_stability: float = 1.0,
) -> float:
    """
    iter418: 检测 chunk 的过时/完成信号，给予 stability 惩罚（加速自然淘汰）。

    在 insert_chunk 管线中调用（Zeigarnik Effect 之后）。

    Returns:
      float — 更新后的 stability
    """
    if not chunk_id:
        return base_stability
    import config as _config
    if not _config.get("store_vfs.df_enabled"):
        return base_stability
    try:
        row = conn.execute(
            "SELECT content, chunk_type, stability FROM memory_chunks WHERE id=?",
            (chunk_id,)
        ).fetchone()
        if not row:
            return base_stability

        content = row[0] if isinstance(row, (list, tuple)) else row["content"]
        chunk_type = row[1] if isinstance(row, (list, tuple)) else row["chunk_type"]
        stab = float(row[2] if isinstance(row, (list, tuple)) else row["stability"]) or 1.0

        penalty_cap = _config.get("store_vfs.df_penalty_cap")
        score = compute_directed_forgetting_score(content or "", chunk_type or "")
        penalty = directed_forgetting_penalty(score, stab, penalty_cap)
        _ps_floor = compute_permastore_floor(conn, chunk_id, stab)
        new_stability = max(_ps_floor, stab - penalty)  # iter422: permastore floor
        if penalty > 0.001:
            conn.execute(
                "UPDATE memory_chunks SET stability=? WHERE id=?",
                (new_stability, chunk_id)
            )
        return new_stability
    except Exception:
        return base_stability


# ── iter419: Associative Memory — 新知识借助强关联记忆的编码优势 ────────────────────
# 认知科学依据：Ebbinghaus (1885) Paired Associates; Collins & Loftus (1975) —
#   新知识与已有强记忆共享节点时形成更强记忆痕迹（associative encoding advantage）。
# OS 类比：Linux huge pages — small page adjacent to huge page shares TLB entry (associative locality)。


def apply_associative_memory_bonus(
    conn: sqlite3.Connection,
    chunk_id: str,
    project: str,
    base_stability: float = 1.0,
) -> float:
    """
    iter419: 写入新 chunk 时，若与已有高重要性 chunk 共享 encode_context token，
    给予 stability 加成（关联记忆锚点效应）。

    在 insert_chunk 管线中调用（Directed Forgetting 之后）。

    Returns:
      float — 更新后的 stability
    """
    if not chunk_id or not project:
        return base_stability
    import config as _config
    if not _config.get("store_vfs.am_enabled"):
        return base_stability
    try:
        # Get new chunk's encode_context tokens
        new_row = conn.execute(
            "SELECT encode_context, stability FROM memory_chunks WHERE id=?",
            (chunk_id,)
        ).fetchone()
        if not new_row:
            return base_stability

        new_ec = new_row[0] if isinstance(new_row, (list, tuple)) else new_row["encode_context"]
        stab = float(new_row[1] if isinstance(new_row, (list, tuple)) else new_row["stability"]) or 1.0

        new_tokens = frozenset(t.strip() for t in (new_ec or "").split(",") if t.strip())
        if not new_tokens:
            return stab  # no tokens, no associative bonus

        min_overlap = _config.get("store_vfs.am_min_overlap")
        min_imp = _config.get("store_vfs.am_min_importance")
        bonus_cap = _config.get("store_vfs.am_bonus_cap")

        # Find existing high-importance chunks in same project (excluding self)
        anchors = conn.execute(
            "SELECT id, encode_context, importance FROM memory_chunks "
            "WHERE project=? AND id!=? AND importance >= ? AND encode_context IS NOT NULL",
            (project, chunk_id, min_imp)
        ).fetchall()

        # Find max overlap with any anchor chunk
        max_overlap = 0
        for anchor_row in anchors:
            a_ec = anchor_row[1] if isinstance(anchor_row, (list, tuple)) else anchor_row["encode_context"]
            a_tokens = frozenset(t.strip() for t in (a_ec or "").split(",") if t.strip())
            if not a_tokens:
                continue
            overlap = len(new_tokens & a_tokens)
            if overlap > max_overlap:
                max_overlap = overlap

        if max_overlap < min_overlap:
            return stab  # no sufficient overlap with strong anchors

        # Compute bonus: overlap-scaled, capped at base × bonus_cap
        # More overlap = stronger associative encoding
        overlap_factor = min(1.0, (max_overlap - min_overlap + 1) / 4.0)  # scale: 0.25 per extra overlap
        bonus = stab * bonus_cap * overlap_factor
        new_stability = min(365.0, stab + bonus)
        if bonus > 0.001:
            conn.execute(
                "UPDATE memory_chunks SET stability=? WHERE id=?",
                (new_stability, chunk_id)
            )
        return new_stability
    except Exception:
        return base_stability


# ── iter421: Retroactive Interference — 新学习干扰旧记忆回忆 ────────────────────
# 认知科学依据：McGeoch (1932) Interference Theory; Barnes & Underwood (1959) —
#   新学习的信息（新 chunk）干扰对旧相关信息（高重叠旧 chunk）的回忆。
#   RI 与 PI（iter408）互补：PI = 旧→新，RI = 新→旧。
#   McGeoch: 遗忘的主因是相似记忆的竞争性干扰，而非 Ebbinghaus 的被动衰减。
#   Anderson & Green (2001): 主动抑制相似记忆是 RI 的神经机制。
# 应用：insert_chunk 时，对同项目中 encode_context 高度重叠的低importance旧 chunk
#   施加轻微 stability 衰减（× ri_decay_factor=0.98），模拟新记忆干扰旧记忆。
#   高重要性（importance >= ri_protect_importance=0.85）的 chunk 免疫 RI（核心知识受保护）。
# OS 类比：TLB shootdown (inter-processor interrupt) —
#   当一个核建立新的 VA→PA 映射时，发送 IPI 使其他所有核的相同 VA TLB 条目失效。
#   新 chunk 写入 = 新映射建立 = 旧相关 chunk（旧 VA 条目）需要被"失效"（stability 降低）。

def apply_retroactive_interference(
    conn: sqlite3.Connection,
    new_chunk_id: str,
    project: str,
    base_stability: float = 1.0,
) -> int:
    """
    iter421: 写入新 chunk 后，对同项目中 encode_context 高重叠的旧 chunk 施加轻微 stability 衰减。

    在 insert_chunk 管线中调用（Associative Memory 之后）。

    Returns:
      int — 被干扰的 chunk 数量
    """
    if not new_chunk_id or not project:
        return 0
    import config as _config
    if not _config.get("store_vfs.ri_enabled"):
        return 0
    try:
        min_overlap = _config.get("store_vfs.ri_min_overlap")
        decay_factor = _config.get("store_vfs.ri_decay_factor")
        max_targets = _config.get("store_vfs.ri_max_targets")
        protect_imp = _config.get("store_vfs.ri_protect_importance")

        if decay_factor >= 1.0:
            return 0  # no-op

        # Get new chunk's encode_context tokens
        new_row = conn.execute(
            "SELECT encode_context FROM memory_chunks WHERE id=?",
            (new_chunk_id,)
        ).fetchone()
        if not new_row:
            return 0
        new_ec = new_row[0] if isinstance(new_row, (list, tuple)) else new_row["encode_context"]
        new_tokens = frozenset(t.strip() for t in (new_ec or "").split(",") if t.strip())
        if not new_tokens:
            return 0

        # Find existing chunks in same project (not self, not high-importance anchors)
        candidates = conn.execute(
            "SELECT id, encode_context, stability FROM memory_chunks "
            "WHERE project=? AND id!=? AND importance < ? AND encode_context IS NOT NULL",
            (project, new_chunk_id, protect_imp)
        ).fetchall()

        # Compute overlap for each candidate
        overlapping = []
        for cand in candidates:
            c_id = cand[0] if isinstance(cand, (list, tuple)) else cand["id"]
            c_ec = cand[1] if isinstance(cand, (list, tuple)) else cand["encode_context"]
            c_stab = float(cand[2] if isinstance(cand, (list, tuple)) else cand["stability"]) or 1.0
            c_tokens = frozenset(t.strip() for t in (c_ec or "").split(",") if t.strip())
            overlap = len(new_tokens & c_tokens)
            if overlap >= min_overlap:
                overlapping.append((c_id, c_stab, overlap))

        if not overlapping:
            return 0

        # Sort by overlap descending, take top max_targets
        overlapping.sort(key=lambda x: x[2], reverse=True)
        overlapping = overlapping[:max_targets]

        inhibited = 0
        for c_id, c_stab, _ in overlapping:
            _ps_floor = compute_permastore_floor(conn, c_id, c_stab)
            new_stab = max(_ps_floor, c_stab * decay_factor)  # iter422: permastore floor
            if abs(new_stab - c_stab) > 1e-6:
                conn.execute(
                    "UPDATE memory_chunks SET stability=? WHERE id=?",
                    (new_stab, c_id)
                )
                inhibited += 1
        return inhibited
    except Exception:
        return 0


# ── iter413: Sleep Consolidation — 离线记忆巩固（Stickgold 2005）───────────────
# 认知科学依据：NREM 睡眠中海马体重放最近学习的记忆，将其转移到新皮层。
# OS 类比：Linux pdflush/writeback daemon — session 间 idle period 内后台巩固 dirty pages。


def run_sleep_consolidation(
    conn: sqlite3.Connection,
    project: str,
    now_iso: str = None,
) -> dict:
    """
    iter413: Sleep Consolidation — SessionStart 时对上一 session 的高重要性 chunk 应用离线巩固。

    Stickgold (2005): NREM 睡眠中海马重放最近学习的记忆 → stability 提升 20-30%。
    memory-os 保守实现：× boost_factor（默认 1.06）模拟 "light sleep consolidation"。

    选择标准：
      1. importance >= consolidation.min_importance（只有重要记忆值得 replay）
      2. last_accessed within consolidation.window_hours（只对上一 session 访问的 chunk）
      3. 按 importance 降序取前 max_chunks 个

    返回 dict: {"consolidated": int, "project": str, "boost_factor": float}
    """
    import config as _config
    if not _config.get("consolidation.enabled"):
        return {"consolidated": 0, "project": project, "boost_factor": 1.0}
    if not project:
        return {"consolidated": 0, "project": project, "boost_factor": 1.0}

    if now_iso is None:
        now_iso = datetime.now(timezone.utc).isoformat()
    boost_factor = _config.get("consolidation.boost_factor")
    min_importance = _config.get("consolidation.min_importance")
    window_hours = _config.get("consolidation.window_hours")
    max_chunks = _config.get("consolidation.max_chunks")

    try:
        from datetime import timedelta as _td
        _now_dt = datetime.fromisoformat(now_iso.replace("Z", "+00:00"))
        _cutoff = (_now_dt - _td(hours=window_hours)).isoformat()

        # 选取：high-importance + recently accessed + this project
        rows = conn.execute(
            "SELECT id, stability FROM memory_chunks "
            "WHERE project=? AND importance >= ? AND last_accessed >= ? "
            "  AND COALESCE(stability, 0) < 365.0 "
            "ORDER BY importance DESC LIMIT ?",
            (project, min_importance, _cutoff, max_chunks)
        ).fetchall()

        if not rows:
            return {"consolidated": 0, "project": project, "boost_factor": boost_factor}

        consolidated = 0
        for row in rows:
            cid = row[0] if isinstance(row, (list, tuple)) else row["id"]
            stab = float(row[1] if isinstance(row, (list, tuple)) else row["stability"]) or 1.0
            new_stab = min(365.0, stab * boost_factor)
            conn.execute(
                "UPDATE memory_chunks SET stability=? WHERE id=?",
                (new_stab, cid)
            )
            consolidated += 1

        return {"consolidated": consolidated, "project": project, "boost_factor": boost_factor}
    except Exception:
        return {"consolidated": 0, "project": project, "boost_factor": boost_factor}


# ── 迭代100：IPC 共享内存 API（OS 类比：shmget/shmat/shmdt + MESI 协议）────────

def shm_attach(conn: sqlite3.Connection, chunk_id: str, agent_id: str,
               shared_with: str = "*") -> None:
    """将 chunk 挂载到共享内存段，多 Agent 可见。等价于 shmat()。"""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT OR REPLACE INTO shm_segments
        (chunk_id, owner_agent, shared_with, version, state, created_at, updated_at)
        VALUES (?, ?, ?, 1, 'SHARED', ?, ?)
    """, (chunk_id, agent_id, shared_with, now, now))


def shm_detach(conn: sqlite3.Connection, chunk_id: str, agent_id: str) -> None:
    """从共享内存段卸载。等价于 shmdt()。"""
    conn.execute(
        "DELETE FROM shm_segments WHERE chunk_id=? AND owner_agent=?",
        (chunk_id, agent_id))


def shm_list(conn: sqlite3.Connection, agent_id: str = None,
             project: str = None) -> list:
    """列出当前可见的共享内存段。"""
    sql = """
        SELECT s.chunk_id, s.owner_agent, s.shared_with, s.version, s.state,
               m.summary, m.chunk_type, m.importance
        FROM shm_segments s
        JOIN memory_chunks m ON s.chunk_id = m.id
        WHERE s.state != 'INVALID'
    """
    params = []
    if agent_id:
        sql += " AND (s.shared_with = '*' OR s.shared_with LIKE ?)"
        params.append(f"%{agent_id}%")
    if project:
        sql += " AND m.project IN (?, 'global')"
        params.append(project)
    return [dict(zip(
        ["chunk_id", "owner_agent", "shared_with", "version", "state",
         "summary", "chunk_type", "importance"], r))
        for r in conn.execute(sql, params).fetchall()]


def shm_invalidate(conn: sqlite3.Connection, chunk_id: str) -> int:
    """MESI Invalidate — 标记所有 Agent 缓存失效。修改 chunk 时调用。"""
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.execute("""
        UPDATE shm_segments SET state='INVALID', updated_at=?
        WHERE chunk_id=? AND state != 'INVALID'
    """, (now, chunk_id))
    return cur.rowcount


def shm_promote(conn: sqlite3.Connection, chunk_id: str, agent_id: str,
                project: str = None) -> None:
    """将高价值 chunk 提升到共享内存（global promotion）。"""
    now = datetime.now(timezone.utc).isoformat()
    # 如果指定 project，同时标记 chunk 为 global
    if project:
        conn.execute(
            "UPDATE memory_chunks SET project='global' WHERE id=? AND project=?",
            (chunk_id, project))
    shm_attach(conn, chunk_id, agent_id, shared_with="*")


# ── 迭代100：IPC 消息队列 API（OS 类比：POSIX mq_send/mq_receive）────────

def ipc_send(conn: sqlite3.Connection, source: str, target: str,
             msg_type: str, payload: dict, priority: int = 0,
             ttl_seconds: int = 3600) -> int:
    """发送 IPC 消息。返回消息 ID。"""
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.execute("""
        INSERT INTO ipc_msgq (source_agent, target_agent, msg_type, payload,
                              priority, status, created_at, ttl_seconds)
        VALUES (?, ?, ?, ?, ?, 'QUEUED', ?, ?)
    """, (source, target, msg_type, json.dumps(payload, ensure_ascii=False),
          priority, now, ttl_seconds))
    return cur.lastrowid


def ipc_recv(conn: sqlite3.Connection, agent_id: str,
             msg_type: str = None, limit: int = 10) -> list:
    """接收 IPC 消息。标记为 CONSUMED。等价于 mq_receive()。"""
    now = datetime.now(timezone.utc).isoformat()
    sql = """
        SELECT id, source_agent, msg_type, payload, priority, created_at
        FROM ipc_msgq
        WHERE (target_agent = ? OR target_agent = '*')
          AND status = 'QUEUED'
    """
    params = [agent_id]
    if msg_type:
        sql += " AND msg_type = ?"
        params.append(msg_type)
    sql += " ORDER BY priority DESC, created_at ASC LIMIT ?"
    params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    msgs = []
    for r in rows:
        msgs.append({
            "id": r[0], "source": r[1], "msg_type": r[2],
            "payload": json.loads(r[3]) if r[3] else {},
            "priority": r[4], "created_at": r[5],
        })
        conn.execute(
            "UPDATE ipc_msgq SET status='CONSUMED', consumed_at=? WHERE id=?",
            (now, r[0]))
    return msgs


def ipc_broadcast_knowledge_update(conn: sqlite3.Connection, agent_id: str,
                                    project: str, stats: dict) -> int:
    """广播知识更新通知（修改 chunk 后调用）。"""
    return ipc_send(conn, agent_id, "*", "knowledge_update",
                    {"project": project, **stats}, priority=5)


def ipc_cleanup_expired(conn: sqlite3.Connection) -> int:
    """清理过期消息。loader SessionStart 时调用。"""
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.execute("""
        DELETE FROM ipc_msgq
        WHERE status = 'QUEUED'
          AND datetime(created_at, '+' || ttl_seconds || ' seconds') < datetime(?)
    """, (now,))
    return cur.rowcount


# ── 迭代100：可验证性 API（OS 类比：ECC + TCB — Trusted Computing Base）────────

def update_confidence(conn: sqlite3.Connection, chunk_id: str,
                      delta: float, reason: str,
                      verification_status: str = None) -> float:
    """更新 chunk 置信度。返回新值。"""
    row = conn.execute(
        "SELECT confidence_score, verification_status FROM memory_chunks WHERE id=?",
        (chunk_id,)).fetchone()
    if not row:
        return 0.0
    old = row[0] or 0.7
    new_conf = max(0.05, min(0.99, old + delta))
    new_status = verification_status or row[1] or "pending"
    conn.execute("""
        UPDATE memory_chunks
        SET confidence_score=?, verification_status=?, updated_at=?
        WHERE id=?
    """, (new_conf, new_status, datetime.now(timezone.utc).isoformat(), chunk_id))
    return new_conf


def get_confidence_stats(conn: sqlite3.Connection, project: str) -> dict:
    """获取项目级置信度统计。"""
    rows = conn.execute("""
        SELECT verification_status, COUNT(*), AVG(confidence_score)
        FROM memory_chunks WHERE project IN (?, 'global')
        GROUP BY verification_status
    """, (project,)).fetchall()
    return {r[0] or "pending": {"count": r[1], "avg_conf": round(r[2] or 0.7, 3)}
            for r in rows}


# ── 迭代64：chunk_version — TLB Selective Invalidation ────────────────────────
# OS 类比：Linux inode generation number + NFS Weak Cache Consistency (WKC)

def bump_chunk_version() -> int:
    """递增并返回新版本号。仅在 chunk 新增/删除时调用。"""
    try:
        CHUNK_VERSION_FILE.parent.mkdir(parents=True, exist_ok=True)
        ver = 0
        if CHUNK_VERSION_FILE.exists():
            try:
                ver = int(CHUNK_VERSION_FILE.read_text().strip())
            except (ValueError, OSError):
                ver = 0
        ver += 1
        CHUNK_VERSION_FILE.write_text(str(ver))
        return ver
    except Exception:
        return 0


def read_chunk_version() -> int:
    """读取当前 chunk_version。用于 TLB 缓存失效判断。"""
    try:
        if CHUNK_VERSION_FILE.exists():
            return int(CHUNK_VERSION_FILE.read_text().strip())
    except (ValueError, OSError):
        pass
    return 0

def update_accessed(conn: sqlite3.Connection, chunk_ids: list,
                    now_iso: str = None, recall_quality: int = None) -> None:
    """
    批量更新 last_accessed + access_count 自增。
    iter106: 同时执行 auto-verification — access_count 达到阈值后自动升 verified。
    iter323: SM-2 Ebbinghaus 精确化 — stability × (1 + 0.1 × (quality-3))。
    OS 类比：MMU Accessed bit 置位 + kswapd 扫描计数 + ECC 自动修正。

    Ebbinghaus spacing effect 背景（iter301）：
      心理学研究表明，知识被重复检索的间隔越长，每次重复后的记忆稳定性增益越大。
      memory-os 简化模型：每次命中 stability *= 2.0，上限 365 天（一年）。
      stability 高的 chunk 在 eviction 评分中受保护：
        eviction_score = age_days / stability（越大越优先被驱逐）
      结果：高频被用的知识越来越稳固，长期未被访问的知识自然衰减至被驱逐。
    """
    if not chunk_ids:
        return
    if now_iso is None:
        now_iso = datetime.now(timezone.utc).isoformat()
    placeholders = ",".join("?" * len(chunk_ids))
    # iter389: Read last_accessed + stability BEFORE update (needed for reconsolidation gap calc + iter412 Testing Effect)
    _pre_access_rows = conn.execute(
        f"SELECT id, last_accessed, COALESCE(stability,1.0) FROM memory_chunks WHERE id IN ({placeholders})",
        chunk_ids,
    ).fetchall()
    _pre_access_map = {row[0]: row[1] for row in _pre_access_rows}
    _pre_stability_map = {row[0]: float(row[2]) for row in _pre_access_rows}
    conn.execute(
        f"UPDATE memory_chunks SET last_accessed=?, access_count=COALESCE(access_count,0)+1 "
        f"WHERE id IN ({placeholders})",
        [now_iso] + chunk_ids,
    )
    # iter323: SM-2 Ebbinghaus 精确化 — 替代 stability *= 2.0 的粗糙模型
    # Wozniak (1987) SM-2 算法：S_new = S_old × (1 + 0.1 × (quality - 3))
    #   quality ∈ {0..5}：0=完全忘记，3=勉强回忆，5=完美回忆
    #   quality=5 → ×1.2（最大增益），quality=3 → ×1.0（中性），quality<3 → 降低 stability
    #   旧 ×2.0 = 固定 quality=13 的极端假设，导致 stability 过快饱和
    #
    # iter389: Reconsolidation Window — Walker & Stickgold (2004) 再巩固窗口
    #   gap < 1hr   → quality=3（短时工作记忆刷新，无长时巩固效果）
    #   1hr ≤ gap < 24hr → quality=4（中等间隔，轻微加固）
    #   gap ≥ 24hr  → quality=5（真正的间隔回忆，最大巩固效果）
    #   显式 recall_quality 参数优先（调用方已推断质量时不被覆盖）
    #
    # OS 类比：Linux MGLRU page aging —
    #   短间隔访问（< aging_interval）不晋升 generation；跨 aging 访问 → generation 晋升
    # iter389: Reconsolidation Window — dynamic SM-2 quality inference
    if recall_quality is not None:
        # explicit quality override — skip reconsolidation window
        _rq = max(0, min(5, recall_quality))
        _sm2_factor = max(0.7, 1.0 + 0.1 * (_rq - 3))
        conn.execute(
            f"UPDATE memory_chunks "
            f"SET stability=MIN(365.0, COALESCE(stability,1.0)*?) "
            f"WHERE id IN ({placeholders})",
            [_sm2_factor] + chunk_ids,
        )
    else:
        import config as _config
        _recon_enabled = _config.get("recon.enabled")
        if _recon_enabled:
            # 动态计算：使用 pre-update last_accessed 推断 quality（避免 N+1 重查）
            _short_gap_secs = _config.get("recon.short_gap_hours") * 3600.0
            _medium_gap_secs = _config.get("recon.medium_gap_hours") * 3600.0
            _long_q = _config.get("recon.long_gap_quality")
            _now_ts = datetime.fromisoformat(now_iso.replace("Z", "+00:00")).timestamp()
            # iter412: Testing Effect — 高难度检索强化记忆巩固
            # Roediger & Karpicke (2006): 难检索 → R_at_recall 低 → quality_bonus 高
            # OS 类比：L3 cache miss → aggressive LRU promotion to L1/L2
            _testing_effect_enabled = _config.get("recon.testing_effect_enabled")
            _testing_effect_scale = _config.get("recon.testing_effect_scale")
            # iter420: Spacing Effect — 分布式练习 quality 加成
            # Ebbinghaus (1885) / Cepeda et al. (2006): spaced > massed practice
            # OS 类比：MGLRU cross-generation promotion — distributed access > massed access
            _spacing_effect_enabled = _config.get("store_vfs.spacing_effect_enabled")
            _spacing_quality_scale = _config.get("store_vfs.spacing_quality_scale")

            # 构建 per-chunk quality map（使用 pre-update last_accessed），默认 quality=4
            _quality_map = {cid: 4 for cid in chunk_ids}
            _spaced_increment_ids = []  # chunks that qualify for spaced_access_count increment
            for cid, la in _pre_access_map.items():
                if la:
                    try:
                        _la_ts = datetime.fromisoformat(la.replace("Z", "+00:00")).timestamp()
                        _gap = _now_ts - _la_ts
                        if _gap < _short_gap_secs:
                            _quality_map[cid] = 3
                        elif _gap < _medium_gap_secs:
                            _quality_map[cid] = 4
                        else:
                            _quality_map[cid] = _long_q
                            # iter420: gap >= medium_gap_hours (24h) → this is a "new session"
                            # Increment spaced_access_count for this chunk
                            if _spacing_effect_enabled:
                                _spaced_increment_ids.append(cid)
                        # iter412: Testing Effect — boost quality if retrieval was difficult
                        if _testing_effect_enabled and _testing_effect_scale > 0:
                            import math as _math
                            _stab = _pre_stability_map.get(cid, 1.0)
                            # R_at_recall = exp(-gap_hours / (stability × 24))
                            _gap_hours = _gap / 3600.0
                            _r_at_recall = _math.exp(-_gap_hours / max(0.01, _stab * 24.0))
                            _difficulty = max(0.0, 1.0 - _r_at_recall)
                            _q_bonus = round(_difficulty * _testing_effect_scale)
                            if _q_bonus > 0:
                                _quality_map[cid] = min(5, _quality_map[cid] + _q_bonus)
                    except Exception:
                        pass
            # iter420: Spacing Effect — increment spaced_access_count for long-gap accesses
            if _spacing_effect_enabled and _spaced_increment_ids:
                _sp_ph = ",".join("?" * len(_spaced_increment_ids))
                conn.execute(
                    f"UPDATE memory_chunks SET spaced_access_count=COALESCE(spaced_access_count,0)+1 "
                    f"WHERE id IN ({_sp_ph})",
                    _spaced_increment_ids,
                )
            # iter420: Spacing Effect — add quality bonus based on spacing_factor
            if _spacing_effect_enabled and _spacing_quality_scale > 0:
                # Read spaced_access_count and access_count after increment
                _sp_rows = conn.execute(
                    f"SELECT id, COALESCE(spaced_access_count,0), COALESCE(access_count,1) "
                    f"FROM memory_chunks WHERE id IN ({placeholders})",
                    chunk_ids,
                ).fetchall()
                for _sp_row in _sp_rows:
                    _cid = _sp_row[0]
                    _sac = int(_sp_row[1])  # spaced_access_count
                    _ac = max(1, int(_sp_row[2]))   # access_count
                    _spacing_factor = _sac / _ac  # ∈ [0, 1]
                    if _spacing_factor > 0:
                        _sq_bonus = round(_spacing_factor * _spacing_quality_scale)
                        if _sq_bonus > 0:
                            _quality_map[_cid] = min(5, _quality_map.get(_cid, 4) + _sq_bonus)
            # 按 quality 分组批量更新（避免 N 次单独 SQL）
            from collections import defaultdict as _defaultdict
            _by_quality = _defaultdict(list)
            for cid in chunk_ids:
                _by_quality[_quality_map.get(cid, 4)].append(cid)
            for _q, _cids in _by_quality.items():
                _sm2_f = max(0.7, 1.0 + 0.1 * (_q - 3))
                _ph = ",".join("?" * len(_cids))
                conn.execute(
                    f"UPDATE memory_chunks SET stability=MIN(365.0,COALESCE(stability,1.0)*?) "
                    f"WHERE id IN ({_ph})",
                    [_sm2_f] + _cids,
                )
        else:
            # fallback: fixed quality=4
            _sm2_factor = 1.1  # quality=4 → ×1.1
            conn.execute(
                f"UPDATE memory_chunks "
                f"SET stability=MIN(365.0, COALESCE(stability,1.0)*?) "
                f"WHERE id IN ({placeholders})",
                [_sm2_factor] + chunk_ids,
            )
    # iter106: Auto-Verification — access_count >= 3 且 pending → verified
    # 逻辑：多次被实际召回说明有效，自动升级 verification_status
    # OS 类比：ECC 多次读取一致 → 标记页面为 clean
    AUTO_VERIFY_THRESHOLD = 3
    conn.execute(
        f"UPDATE memory_chunks SET verification_status='verified' "
        f"WHERE id IN ({placeholders}) "
        f"  AND verification_status='pending' "
        f"  AND COALESCE(access_count,0) >= ?",
        chunk_ids + [AUTO_VERIFY_THRESHOLD],
    )

    # iter404: Semantic Priming — 访问 chunk 时，prime 其 encode_context 中的实体
    # OS 类比：readahead_trigger() — 访问 page N，将相邻 pages 标记进 readahead window
    try:
        ec_rows = conn.execute(
            f"SELECT id, encode_context, project FROM memory_chunks "
            f"WHERE id IN ({placeholders}) AND encode_context IS NOT NULL AND encode_context != ''",
            chunk_ids,
        ).fetchall()
        for _cid, _ec, _proj in ec_rows:
            if _ec and _proj:
                _tokens = [t.strip() for t in _ec.split(",") if t.strip()]
                if _tokens:
                    prime_entities(conn, _tokens, _proj, prime_strength=0.8, now_iso=now_iso)
    except Exception:
        pass

    # iter415: Encoding Variability — 多情境访问 → encode_context 富化 → stability 加成
    # Estes (1955): 多情境编码提升检索鲁棒性（更多检索线索）
    # OS 类比：共享库被 N 个进程引用 → 高引用计数 → 不易被 kswapd 驱逐
    try:
        for _cid in chunk_ids:
            apply_encoding_variability(conn, _cid)
    except Exception:
        pass

    # iter417: Retrieval-Induced Forgetting — 检索引发语义竞争者 stability 轻微衰减
    # Anderson et al. (1994): 检索记忆 A 抑制其语义竞争者 B/C
    # OS 类比：MESI 协议 — 写入 cache line 使其他核的相同 line 变为 Invalid
    try:
        # Get project from the accessed chunks (use first hit)
        _rif_proj_row = conn.execute(
            f"SELECT project FROM memory_chunks WHERE id IN ({placeholders}) LIMIT 1",
            chunk_ids,
        ).fetchone()
        if _rif_proj_row:
            _rif_proj = _rif_proj_row[0] if isinstance(_rif_proj_row, (list, tuple)) else _rif_proj_row["project"]
            apply_retrieval_induced_forgetting(conn, chunk_ids, _rif_proj)
    except Exception:
        pass

def insert_trace(conn: sqlite3.Connection, trace_dict: dict) -> None:
    """写入 recall_traces 记录。迭代65：新增 ftrace_json 阶段级追踪。"""
    d = trace_dict
    top_k = json.dumps(d["top_k_json"], ensure_ascii=False) if isinstance(d.get("top_k_json"), list) else d.get("top_k_json", "[]")
    ftrace = d.get("ftrace_json")
    ftrace_str = json.dumps(ftrace, ensure_ascii=False) if isinstance(ftrace, dict) else ftrace
    conn.execute("""
        INSERT INTO recall_traces
        (id, timestamp, session_id, project, prompt_hash,
         candidates_count, top_k_json, injected, reason, duration_ms, ftrace_json)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (
        d["id"], d["timestamp"], d["session_id"], d["project"],
        d["prompt_hash"], d["candidates_count"], top_k,
        d["injected"], d["reason"], round(d.get("duration_ms", 0), 2),
        ftrace_str,
    ))

def find_similar(conn: sqlite3.Connection, summary: str, chunk_type: str,
                 threshold: float = 0.22, project: str = None) -> Optional[str]:
    """
    Jaccard token 相似度查重，返回最相似 chunk 的 id 或 None。
    OS 类比：KSM (Kernel Same-page Merging) — 内容相同的物理页合并。

    iter105: threshold 默认从 0.5 降到 0.28。
    原因：中文 bigram 分词粒度细，同义表述的 Jaccard 实测约 0.25-0.35，
    0.5 阈值实际上从未触发（等于没有去重）。0.28 在实测中可命中语义相近句。
    project 参数：限制去重范围到同 project，避免跨项目误合并。
    """
    if project:
        rows = conn.execute(
            "SELECT id, summary FROM memory_chunks WHERE chunk_type=? AND summary!='' AND project=?",
            (chunk_type, project)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, summary FROM memory_chunks WHERE chunk_type=? AND summary!=''",
            (chunk_type,)
        ).fetchall()
    if not rows:
        return None

    def _tok(text):
        tokens = set()
        for m in re.finditer(r'[a-zA-Z0-9_][-a-zA-Z0-9_.]*', text):
            tokens.add(m.group().lower())
        cn = re.sub(r'[^\u4e00-\u9fff]', '', text)
        for i in range(len(cn) - 1):
            tokens.add(cn[i:i + 2])
        return tokens

    q_set = _tok(summary)
    if not q_set:
        return None

    best_score, best_id = 0.0, None
    for rid, existing_summary in rows:
        d_set = _tok(existing_summary)
        if not d_set:
            continue
        intersection = len(q_set & d_set)
        union = len(q_set | d_set)
        jaccard = intersection / union if union > 0 else 0.0
        if jaccard > best_score:
            best_score = jaccard
            best_id = rid
    return best_id if best_score >= threshold else None

def already_exists(conn: sqlite3.Connection, summary: str, chunk_type: str = None) -> bool:
    """全局去重：相同 summary 不重复写入。
    chunk_type 为 None 时检查所有已知类型，指定时只检查该类型。

    iter107: 跨项目全局去重 — [规则/...] 前缀的 summary 不受 project 限制，
    同一规则文本在任意 project 中写过一次即全局阻断。
    根因：sleep session 在不同 project ID 下运行，旧去重逻辑只在同 project 内查重，
    导致同一规则内容在每个新 project 下各写一份，造成 DB 膨胀（每晚 +200 条）。
    OS 类比：全局页面表 (global page table) — 共享内核页面只映射一次，不因进程不同而重复分配。
    """
    # iter107：[规则/...] 跨项目全局去重（忽略 project 字段）
    if re.match(r'^\[规则[/／]', summary.strip()):
        row = conn.execute(
            "SELECT id FROM memory_chunks WHERE summary=?",
            (summary,)
        ).fetchone()
        return row is not None

    if chunk_type:
        row = conn.execute(
            "SELECT id FROM memory_chunks WHERE summary=? AND chunk_type=?",
            (summary, chunk_type),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT id FROM memory_chunks WHERE summary=? "
            "AND chunk_type IN ('decision','excluded_path','reasoning_chain','conversation_summary','prompt_context','design_constraint')",
            (summary,)
        ).fetchone()
    return row is not None


def detect_and_invalidate_conflicts(
    conn: sqlite3.Connection,
    new_summary: str,
    chunk_type: str,
    project: str,
) -> int:
    """
    iter371: Memory Conflict Detection — MESI 缓存一致性协议类比

    认知科学背景：
      前向干扰（Retroactive Interference, McGeoch & McDonald 1931）—
      新记忆的写入会干扰已有的旧记忆，使旧记忆的提取可靠性下降。
      例如：旧知识"使用 X 方案"，新知识"放弃 X 采用 Y"—— 旧知识应降权。

    OS 类比：MESI 缓存一致性协议（Intel 1984）—
      当 CPU 核心修改缓存行（M state），其他核心持有该行的副本
      从 Shared(S) 降级为 Invalid(I)，下次访问触发 cache miss 重新从主存加载。
      这里：新写入 chunk 触发对语义矛盾旧 chunk 的 Invalid 降权。

    策略：
      1. 只对 decision/reasoning_chain 类型触发（其他类型不存在明确语义矛盾）
      2. 从 new_summary 中提取"被否定实体"：放弃/不选/废弃/replaced by/not using 后的词
      3. FTS5 搜索旧 chunk 中包含这些关键词的记录（同 project + 同 chunk_type）
      4. 对语义矛盾的旧 chunk：importance *= 0.8，oom_adj += 100（降权但不删除）
      5. 返回失效的 chunk 数

    示例：
      new: "放弃 SQLite 改用 PostgreSQL"
      → 搜索含 "SQLite" 的旧 decision chunk
      → 找到 "选择 SQLite 因为简单" → importance *= 0.8, oom_adj += 100
    """
    if chunk_type not in ("decision", "reasoning_chain"):
        return 0
    if not new_summary or len(new_summary) < 5:
        return 0

    import re as _re

    # 提取被否定/替换的实体关键词
    NEGATION_PATTERNS = [
        r'(?:放弃|不选|不用|废弃|替换|不再用|弃用|移除|删除)\s*([\w\u4e00-\u9fff_\-.]{2,20})',
        r'(?:replaced?|abandoned?|rejected?|removed?|deprecated?)\s+([\w_\-.]{2,20})',
        r'(?:不选择|不采用|不推荐)\s*([\w\u4e00-\u9fff_\-.]{2,20})',
        r'而非\s*([\w\u4e00-\u9fff_\-.]{2,20})',
        r'not\s+using\s+([\w_\-.]{2,20})',
        r'instead\s+of\s+([\w_\-.]{2,20})',
    ]

    negated_entities = []
    for pat in NEGATION_PATTERNS:
        for m in _re.finditer(pat, new_summary, _re.IGNORECASE | _re.UNICODE):
            entity = m.group(1).strip()
            if len(entity) >= 2:
                negated_entities.append(entity)

    if not negated_entities:
        return 0

    invalidated = 0
    now_iso = datetime.now(timezone.utc).isoformat()

    for entity in negated_entities[:3]:  # 最多处理3个实体（避免过度失效）
        try:
            # FTS5 搜索包含该实体的旧 chunk（同 project + 同类型）
            fts_query = entity.replace('"', '""')
            rows = conn.execute(
                """SELECT mc.id, mc.importance, mc.oom_adj, mc.summary
                   FROM memory_chunks mc
                   JOIN memory_chunks_fts fts ON mc.id = fts.rowid
                   WHERE fts.summary MATCH ?
                     AND mc.project = ?
                     AND mc.chunk_type = ?
                     AND mc.summary != ?
                   LIMIT 5""",
                (f'"{fts_query}"', project, chunk_type, new_summary)
            ).fetchall()
        except Exception:
            # FTS5 可能不可用，降级到 LIKE 查询
            try:
                rows = conn.execute(
                    """SELECT id, importance, oom_adj, summary
                       FROM memory_chunks
                       WHERE summary LIKE ?
                         AND project = ?
                         AND chunk_type = ?
                         AND summary != ?
                       LIMIT 5""",
                    (f"%{entity}%", project, chunk_type, new_summary)
                ).fetchall()
            except Exception:
                rows = []

        for row in rows:
            cid, imp, oom, old_summary = row
            # 验证旧 chunk 与新 chunk 确实语义矛盾：旧 chunk 应该是"推荐"该实体的
            # 避免把"放弃 X"之类的旧 excluded_path chunk 也降权
            AFFIRMATION_SIGNALS = _re.compile(
                r'(?:选择|采用|推荐|使用|用|基于|保留|保持|'
                r'decided?|chosen?|using|adopted?|recommended?)',
                _re.IGNORECASE
            )
            if not AFFIRMATION_SIGNALS.search(old_summary):
                continue

            new_imp = round(max(imp * 0.8, 0.1), 4)
            new_oom = min((oom or 0) + 100, 800)
            try:
                conn.execute(
                    "UPDATE memory_chunks SET importance=?, oom_adj=?, updated_at=? WHERE id=?",
                    (new_imp, new_oom, now_iso, cid)
                )
                invalidated += 1
            except Exception:
                pass

    return invalidated


def merge_similar(conn: sqlite3.Connection, summary: str, chunk_type: str,
                  importance: float, project: str = None) -> bool:
    """
    KSM merge：如果已存在相似 chunk，更新其 importance 并追加新内容到 content。
    返回 True 表示已合并（调用方不需要 INSERT）。

    iter105: threshold 提升到 0.65（减少误合并），合并时追加新 summary 到 content
    让 chunk 随时间积累不同角度的表述，提升检索召回率。
    OS 类比：KSM 合并相同物理页，但保留 COW — 写时才分离。
    """
    similar_id = find_similar(conn, summary, chunk_type, threshold=0.22, project=project)
    if not similar_id:
        return False
    now_iso = datetime.now(timezone.utc).isoformat()
    # 追加新 summary 到 content（不同角度表述的聚合，提升 FTS5 召回覆盖）
    row = conn.execute(
        "SELECT content FROM memory_chunks WHERE id=?", (similar_id,)
    ).fetchone()
    existing_content = row[0] if row else ""
    # 只在新 summary 与现有 content 不重叠时追加（避免完全重复）
    if summary not in existing_content:
        new_content = (existing_content + "\n" + summary).strip()[:2000]
    else:
        new_content = existing_content
    conn.execute(
        "UPDATE memory_chunks SET importance=MAX(importance, ?), last_accessed=?, updated_at=?, content=? WHERE id=?",
        (importance, now_iso, now_iso, new_content, similar_id),
    )
    return True

def coalesce_small_chunks(
    conn: sqlite3.Connection,
    project: str,
    min_group: int = 3,
    max_summary_len: int = 60,
    chunk_type: str = "conversation_summary",
    topic_prefix_len: int = 4,
) -> int:
    """
    iter374: Chunk Coalescing — Slab Allocator 合并碎片化小 chunk。

    人的记忆类比：Chunking (Miller 1956) — 人类将相关小记忆片段合并为有意义的组块，
      降低工作记忆负担，提升整体记忆容量。
    OS 类比：Linux Slab Allocator (Bonwick 1994) — 相同大小的对象归入同一 slab，
      避免碎片化，提高内存利用率。
      memory_chunks 中 conversation_summary 类型往往因为短会话产生大量碎片：
        chunk_1: "用户讨论了端口配置"  (imp=0.5)
        chunk_2: "用户询问了端口号"    (imp=0.5)
        chunk_3: "用户确认了3000端口" (imp=0.5)
      → 合并为一个高质量复合 chunk（max importance，content = 所有 summary 拼接）。

    触发条件（同时满足）：
      1. chunk_type 为 conversation_summary（可配置）
      2. summary 长度 <= max_summary_len（小 chunk 特征）
      3. summary 前 topic_prefix_len 字相同（同一主题组）
      4. 同组 chunk 数量 >= min_group

    合并策略：
      - 保留最高 importance 的 chunk（anchor）
      - anchor.content = 所有 chunk summary 拼接
      - anchor.importance = max(all importance)
      - 删除其余 chunk
      - 递增 chunk_version（触发 TLB 失效）

    Returns: 合并产生的 composite chunk 数量（即触发的合并组数）
    """
    try:
        # 查找所有符合条件的小 chunk（summary 短 + 指定类型 + 同项目）
        rows = conn.execute(
            """SELECT id, summary, importance, content, created_at
               FROM memory_chunks
               WHERE project = ? AND chunk_type = ?
                 AND LENGTH(summary) <= ?
               ORDER BY summary, created_at""",
            (project, chunk_type, max_summary_len),
        ).fetchall()

        if not rows:
            return 0

        # 按 topic_prefix 分组（前 N 字作为主题键）
        groups: dict = {}
        for row_id, summary, importance, content, created_at in rows:
            prefix = (summary or "")[:topic_prefix_len].strip()
            if not prefix:
                continue
            groups.setdefault(prefix, []).append({
                "id": row_id,
                "summary": summary,
                "importance": importance,
                "content": content or "",
                "created_at": created_at,
            })

        coalesced = 0
        now_iso = datetime.now(timezone.utc).isoformat()

        for prefix, members in groups.items():
            if len(members) < min_group:
                continue  # 不足 min_group，不合并

            # 按 importance 降序选 anchor（最高 importance 的 chunk 保留）
            members_sorted = sorted(members, key=lambda x: x["importance"], reverse=True)
            anchor = members_sorted[0]
            rest = members_sorted[1:]

            # 构建复合 content = 所有不重复 summary 拼接
            all_summaries = [anchor["summary"]] + [m["summary"] for m in rest]
            seen: set = set()
            unique_summaries = []
            for s in all_summaries:
                s_strip = s.strip()
                if s_strip and s_strip not in seen:
                    seen.add(s_strip)
                    unique_summaries.append(s_strip)
            composite_content = "\n".join(unique_summaries)[:2000]

            # 更新 anchor
            max_imp = anchor["importance"]
            conn.execute(
                """UPDATE memory_chunks
                   SET content=?, importance=?, updated_at=?
                   WHERE id=?""",
                (composite_content, max_imp, now_iso, anchor["id"]),
            )

            # 删除其余（包括 FTS5 同步）
            rest_ids = [m["id"] for m in rest]
            if rest_ids:
                placeholders = ",".join("?" * len(rest_ids))
                # 获取 rowids for FTS5 清理
                rowids = [r[0] for r in conn.execute(
                    f"SELECT rowid FROM memory_chunks WHERE id IN ({placeholders})",
                    rest_ids,
                ).fetchall()]
                conn.execute(
                    f"DELETE FROM memory_chunks WHERE id IN ({placeholders})",
                    rest_ids,
                )
                for rowid in rowids:
                    try:
                        conn.execute(
                            "DELETE FROM memory_chunks_fts WHERE rowid_ref=?",
                            (str(rowid),),
                        )
                    except Exception:
                        pass

            coalesced += 1

        if coalesced > 0:
            # 递增 chunk_version，触发 TLB 失效
            try:
                _cv_path = os.path.join(
                    os.environ.get("MEMORY_OS_DIR",
                                   os.path.join(os.path.expanduser("~"), ".claude", "memory-os")),
                    ".chunk_version",
                )
                try:
                    with open(_cv_path, encoding="utf-8") as _f:
                        _cv = int(_f.read().strip())
                except Exception:
                    _cv = 0
                with open(_cv_path, "w", encoding="utf-8") as _f:
                    _f.write(str(_cv + 1))
            except Exception:
                pass

        return coalesced

    except Exception:
        return 0


def delete_chunks(conn: sqlite3.Connection, chunk_ids: list) -> int:
    """
    批量删除 chunk，返回实际删除数。
    OS 类比：VFS 的 unlink() — 统一删除接口。
    迭代97：同步删除 FTS5 记录。
    """
    if not chunk_ids:
        return 0
    # 先获取要删除的 rowid（用于 FTS5 清理）
    placeholders = ",".join("?" * len(chunk_ids))
    rowids = [r[0] for r in conn.execute(
        f"SELECT rowid FROM memory_chunks WHERE id IN ({placeholders})", chunk_ids
    ).fetchall()]

    count = conn.execute(
        f"DELETE FROM memory_chunks WHERE id IN ({placeholders})",
        chunk_ids,
    ).rowcount

    # 迭代97：同步删除 FTS5 记录
    if rowids:
        for rowid in rowids:
            try:
                conn.execute("DELETE FROM memory_chunks_fts WHERE rowid_ref=?", (str(rowid),))
            except Exception:
                pass

    if count > 0:
        bump_chunk_version()  # 迭代64: TLB v2
    return count

def get_chunk_count(conn: sqlite3.Connection) -> int:
    """返回当前 chunk 总数。"""
    return conn.execute("SELECT COUNT(*) FROM memory_chunks").fetchone()[0]

def get_project_chunk_count(conn: sqlite3.Connection, project: str) -> int:
    """
    迭代25：返回指定项目的 chunk 数量。
    OS 类比：cgroup 的 memory.usage_in_bytes — 查询当前资源占用。
    """
    return conn.execute(
        "SELECT COUNT(*) FROM memory_chunks WHERE project=?", (project,)
    ).fetchone()[0]

def evict_lowest_retention(conn: sqlite3.Connection, project: str,
                           count: int, protect_types: tuple = ("task_state",)) -> list:
    """
    迭代25+26：按 unified retention_score 淘汰指定项目中 score 最低的 N 条 chunk。
    OS 类比：cgroup OOM handler — 资源超配额时按优先级回收。

    迭代26 修复：使用 scorer.py 的 retention_score() 替代 ad-hoc ORDER BY。
    迭代104：hard pin 的 chunk 不参与 kswapd 硬淘汰（soft pin 不保护此路径）。

    策略：
    - 跳过 protect_types（task_state 是当前会话状态，不可淘汰）
    - 按 unified retention_score 升序（最低分先淘汰）
    - 返回被淘汰的 chunk id 列表
    """
    if count <= 0:
        return []

    from scorer import retention_score as _retention_score

    # 迭代104：hard pin 的 chunk 不被 kswapd 硬淘汰
    hard_pinned = get_pinned_ids(conn, project, pin_type="hard")

    protect_placeholders = ",".join("?" * len(protect_types))
    # 取所有候选 chunk（需要 Python 端计算 retention_score）
    # 限制候选集为 count * 5 以避免大表全扫描
    # 迭代38：排除 oom_adj <= -1000 的 chunk（OOM_SCORE_ADJ_MIN = 绝对保护）
    candidate_limit = max(count * 5, 50)
    # 迭代44：MGLRU — 优先从最老代淘汰（gen DESC），同代内按 importance/recency 排序
    # 迭代301：加入 stability 和 info_class
    # iter_multiagent P1：排除最近 10 分钟内写入的 chunk（cross-agent grace period）。
    # 根因：多 agent 共享同一 project 时，Agent B 的 kswapd 可能淘汰 Agent A 刚写入的
    # 低 retention 新 chunk（如 conversation_summary，importance=0.65）。
    # 修复：created_at >= datetime('now', '-10 minutes') 的 chunk 不参与 kswapd 硬淘汰。
    # OS 类比：Linux cgroup v2 memory.min — 保护新分配的页面不在 grace period 内被回收。
    rows = conn.execute(
        f"""SELECT id, importance, last_accessed, COALESCE(access_count, 0),
                   COALESCE(oom_adj, 0), COALESCE(lru_gen, 0),
                   COALESCE(stability, 1.0), COALESCE(info_class, 'world')
            FROM memory_chunks
            WHERE project=? AND chunk_type NOT IN ({protect_placeholders})
              AND COALESCE(oom_adj, 0) > -1000
              AND (created_at IS NULL OR datetime(created_at) < datetime('now', '-10 minutes'))
            ORDER BY COALESCE(lru_gen, 0) DESC, importance ASC, last_accessed ASC
            LIMIT ?""",
        (project, *protect_types, candidate_limit),
    ).fetchall()

    if not rows:
        return []

    # 用 Unified Scorer retention_score 精确排序
    # 迭代38：oom_adj 作为修正因子
    # 迭代300：info_class ephemeral 额外降分（更容易被淘汰）
    # 迭代301：Ebbinghaus stability 保护因子
    from datetime import datetime as _dt
    _now_ts = _dt.now(timezone.utc).isoformat()
    scored = []
    for row in rows:
        rid, importance, last_accessed, access_count, oom_adj, lru_gen, stability, info_class = (
            row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]
        )
        if rid in hard_pinned:
            continue  # 迭代104：hard pin 跳过 kswapd 硬淘汰
        score = _retention_score(
            importance=importance if importance is not None else 0.5,
            last_accessed=last_accessed or "",
            uniqueness=0.5,
            access_count=access_count or 0,
        )
        # OOM 修正：oom_adj 正值让 score 下降（更容易被淘汰）
        oom_modifier = oom_adj / 2000.0
        score = max(0.0, score - oom_modifier)
        # 迭代300：ephemeral 额外降分 0.15（更优先被驱逐）
        if info_class == "ephemeral":
            score = max(0.0, score - 0.15)
        # 迭代301：Ebbinghaus stability 保护
        # retention_score 加上 stability_bonus = ln(stability+1) * 0.05，上限 0.2
        import math as _math
        stability_bonus = min(0.20, _math.log(stability + 1.0) * 0.05)
        score = min(1.0, score + stability_bonus)
        scored.append((score, rid))

    # 按 retention_score 升序，取最低的 count 条
    scored.sort(key=lambda x: x[0])
    evicted_ids = [rid for _, rid in scored[:count]]
    # 迭代33：swap out 替代直接删除（保留冷知识，可恢复）
    from store_swap import swap_out  # deferred import to avoid circular dependency
    swap_out(conn, evicted_ids)
    return evicted_ids


# ── 迭代104：chunk_pins — 项目级 pin API ─────────────────────────────
# OS 类比：Linux mlock()/munlock() per-VMA 内存锁定接口
#
# pin_type:
#   'hard' — kswapd/damon/stale_reclaim 全部跳过（类比 mlock + MAP_LOCKED）
#   'soft' — 仅跳过 stale/damon DEAD 清理，kswapd ZONE_MIN 仍可淘汰
#            （类比 MADV_WILLNEED：建议保留但非强制）

def pin_chunk(conn: sqlite3.Connection, chunk_id: str, project: str,
              pin_type: str = "soft") -> bool:
    """
    迭代104：将 chunk 锁定到指定项目（project-scoped mlock）。
    OS 类比：mlock(addr, len) — 将页面锁定在当前进程地址空间，阻止被 swap out。

    pin_type:
      'hard' — 任何淘汰路径均跳过该 chunk（kswapd ZONE_MIN 除外，但 stale/damon 完全保护）
      'soft' — 仅保护 stale reclaim 和 DAMON DEAD 清理，不干预 kswapd watermark 淘汰

    返回 True 表示成功写入（新 pin 或 upsert），False 表示 chunk 不存在。
    """
    from datetime import datetime, timezone
    # 验证 chunk 存在
    if not conn.execute("SELECT 1 FROM memory_chunks WHERE id=?", (chunk_id,)).fetchone():
        return False
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO chunk_pins (chunk_id, project, pin_type, pinned_at)
           VALUES (?, ?, ?, ?)""",
        (chunk_id, project, pin_type, now),
    )
    return True


def unpin_chunk(conn: sqlite3.Connection, chunk_id: str, project: str) -> bool:
    """
    迭代104：解除 chunk 在指定项目中的 pin。
    OS 类比：munlock(addr, len) — 解除内存锁定，页面重新可被 swap out。

    返回 True 表示成功删除，False 表示原本未 pin。
    """
    rowcount = conn.execute(
        "DELETE FROM chunk_pins WHERE chunk_id=? AND project=?",
        (chunk_id, project),
    ).rowcount
    return rowcount > 0


def is_pinned(conn: sqlite3.Connection, chunk_id: str, project: str) -> Optional[str]:
    """
    迭代104：查询 chunk 在指定项目中的 pin 状态。
    OS 类比：/proc/[pid]/smaps 中的 Locked: 字段。

    返回 pin_type ('hard'/'soft') 或 None（未 pin）。
    """
    row = conn.execute(
        "SELECT pin_type FROM chunk_pins WHERE chunk_id=? AND project=?",
        (chunk_id, project),
    ).fetchone()
    return row[0] if row else None


def get_pinned_chunks(conn: sqlite3.Connection, project: str,
                      pin_type: str = None) -> list:
    """
    迭代104：列出项目中所有 pinned chunk 的 ID。
    OS 类比：获取进程 mlock 区域列表（/proc/[pid]/smaps 的 VmLck 汇总）。

    pin_type=None 返回全部，pin_type='hard'/'soft' 按类型过滤。
    返回 {chunk_id, pin_type, pinned_at, summary, chunk_type} 字典列表。
    """
    if pin_type:
        rows = conn.execute(
            """SELECT cp.chunk_id, cp.pin_type, cp.pinned_at,
                      mc.summary, mc.chunk_type, mc.importance
               FROM chunk_pins cp
               JOIN memory_chunks mc ON mc.id = cp.chunk_id
               WHERE cp.project = ? AND cp.pin_type = ?
               ORDER BY cp.pinned_at DESC""",
            (project, pin_type),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT cp.chunk_id, cp.pin_type, cp.pinned_at,
                      mc.summary, mc.chunk_type, mc.importance
               FROM chunk_pins cp
               JOIN memory_chunks mc ON mc.id = cp.chunk_id
               WHERE cp.project = ?
               ORDER BY cp.pin_type DESC, cp.pinned_at DESC""",
            (project,),
        ).fetchall()
    return [
        {"chunk_id": r[0], "pin_type": r[1], "pinned_at": r[2],
         "summary": r[3], "chunk_type": r[4], "importance": r[5]}
        for r in rows
    ]


def get_pinned_ids(conn: sqlite3.Connection, project: str,
                   pin_type: str = None) -> set:
    """
    迭代104：返回项目中所有 pinned chunk_id 的集合（高效查询用）。
    OS 类比：内核的 locked_vm 计数 + mlock 位图。
    供 kswapd/damon/stale_reclaim 批量排除使用。
    """
    if pin_type:
        rows = conn.execute(
            "SELECT chunk_id FROM chunk_pins WHERE project=? AND pin_type=?",
            (project, pin_type),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT chunk_id FROM chunk_pins WHERE project=?",
            (project,),
        ).fetchall()
    return {r[0] for r in rows}


# ── 迭代356：Pin Decay + Pin Cap ────────────────────────────────────────────
# OS 类比：Linux RLIMIT_MEMLOCK + memcg pin_user_pages 上限
#
# 问题根因（v5 audit, 2026-04-28）：
#   chunk_pins 表无过期机制。chunk 被 pin 后永久不可被 LRU 驱逐。
#   实测 47/105 chunk（45%）处于 pin 状态，仅剩 55% 参与正常 LRU 循环，
#   导致高 importance chunk 被迫 swap out（LRU 逼出），同时 swap 无法恢复（swap dead zone）。
#
# 修复方案：
#   pin_decay()  — 扫描 soft pin，last_accessed 超过 decay_days 天的自动解除
#   _enforce_pin_cap() — 新增 pin 时检查上限，超限时驱逐最旧 soft pin

def pin_decay(conn: sqlite3.Connection, project: str,
              decay_days: int = None) -> int:
    """
    迭代356：Soft pin 衰减 — 长期未访问的 soft pin 自动解除。

    OS 类比：munlock_vma_pages_range() 在进程 exit_mm 时解除 mlock 区域；
    这里模拟一个周期性 GC：soft pin 超过 decay_days 天未访问 → 解除 pin，
    允许重新参与 LRU eviction 和 swap out。

    Hard pin（design_constraint 等核心架构知识）不受衰减影响。

    返回解除的 pin 数量。
    """
    from config import get as _cfg
    if not _cfg("pin.decay_enabled"):
        return 0
    if decay_days is None:
        decay_days = _cfg("pin.decay_days")

    # 找出 soft pin 且 chunk 的 last_accessed 超过 decay_days 天的条目
    # 若 last_accessed 为 NULL，以 pinned_at 为准
    stale_rows = conn.execute(
        """SELECT cp.chunk_id
           FROM chunk_pins cp
           LEFT JOIN memory_chunks mc ON mc.id = cp.chunk_id
           WHERE cp.project = ?
             AND cp.pin_type = 'soft'
             AND (
               datetime(COALESCE(mc.last_accessed, cp.pinned_at)) <
               datetime('now', ? || ' days')
             )""",
        (project, f"-{decay_days}"),
    ).fetchall()

    if not stale_rows:
        return 0

    stale_ids = [r[0] for r in stale_rows]
    for cid in stale_ids:
        conn.execute(
            "DELETE FROM chunk_pins WHERE chunk_id=? AND project=? AND pin_type='soft'",
            (cid, project),
        )
    return len(stale_ids)


def enforce_pin_cap(conn: sqlite3.Connection, project: str,
                    cap_pct: int = None) -> int:
    """
    迭代356：Pin 上限执行 — 超过 cap_pct% 时驱逐最旧 soft pin。

    OS 类比：RLIMIT_MEMLOCK 在 mlock() 时检查当前 locked_vm，
    超限则 EAGAIN（或主动释放）。

    策略：
      1. 计算当前项目总 chunk 数
      2. 计算当前 pin 数量（hard + soft）
      3. 若 pin_count > cap_pct% × total → 按 pinned_at ASC 驱逐最旧 soft pin
      注：hard pin 不被驱逐（架构约束必须保留）

    返回驱逐的 soft pin 数量。
    """
    from config import get as _cfg
    if not _cfg("pin.cap_apply_on_pin"):
        return 0
    if cap_pct is None:
        cap_pct = _cfg("pin.cap_pct")

    total = conn.execute(
        "SELECT COUNT(*) FROM memory_chunks WHERE project=?", (project,)
    ).fetchone()[0]
    if total == 0:
        return 0

    cap_limit = max(1, int(total * cap_pct / 100))
    pin_count = conn.execute(
        "SELECT COUNT(*) FROM chunk_pins WHERE project=?", (project,)
    ).fetchone()[0]

    if pin_count <= cap_limit:
        return 0

    # 超限：驱逐最旧的 soft pin（按 pinned_at 升序）
    excess = pin_count - cap_limit
    oldest_soft = conn.execute(
        """SELECT cp.chunk_id FROM chunk_pins cp
           WHERE cp.project = ? AND cp.pin_type = 'soft'
           ORDER BY cp.pinned_at ASC
           LIMIT ?""",
        (project, excess),
    ).fetchall()

    if not oldest_soft:
        return 0

    evicted = 0
    for row in oldest_soft:
        conn.execute(
            "DELETE FROM chunk_pins WHERE chunk_id=? AND project=? AND pin_type='soft'",
            (row[0], project),
        )
        evicted += 1
    return evicted


# ── 迭代304：知识图谱关系边 API ────────────────────────────────────────────
# OS 类比：Linux 内核模块依赖图（module_kobject + sysfs /sys/module/<mod>/holders/）
#   insert_edge  ≈ modprobe 写入依赖条目
#   query_neighbors ≈ modinfo --field=depends / /sys/module/<mod>/holders/

def insert_edge(conn: sqlite3.Connection,
                from_entity: str,
                relation: str,
                to_entity: str,
                project: str = None,
                source_chunk_id: str = None,
                confidence: float = 0.7) -> str:
    """
    迭代304：幂等插入关系边。
    (from_entity, relation, to_entity) 三元组相同则更新 confidence；
    否则插入新边，返回 edge_id。

    OS 类比：Linux sysfs kobject_add() — 同一对象重复 add 只更新属性，
      不会创建重复 sysfs 节点（幂等性由 kset_find_obj 保证）。
    """
    import hashlib
    # 用三元组生成确定性 id（幂等键）
    key = f"{from_entity}|{relation}|{to_entity}"
    edge_id = "ee_" + hashlib.sha1(key.encode()).hexdigest()[:16]

    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO entity_edges (id, from_entity, relation, to_entity, project,
                                   source_chunk_id, confidence, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET confidence=excluded.confidence
        """,
        (edge_id, from_entity, relation, to_entity, project,
         source_chunk_id, confidence, now),
    )

    # 迭代310：insert_edge 时反向建立 entity_map（新增 entity 对应已有 chunks）
    # OS 类比：Linux inotify 反向 dentry — 新路径名与已有 inode 建立关联时更新 dcache
    try:
        if project:
            now_str = now
            for ent in (from_entity, to_entity):
                if not ent:
                    continue
                ent_lower = ent.lower()
                ent_normalized = ent_lower.replace("_", " ").replace("-", " ")
                # 在 memory_chunks 中找 summary 包含该 entity 的 chunk
                rows = conn.execute(
                    "SELECT id FROM memory_chunks WHERE project=? "
                    "AND (LOWER(summary) LIKE ? OR LOWER(summary) LIKE ?)",
                    (project, f"%{ent_lower}%", f"%{ent_normalized}%")
                ).fetchall()
                for (cid,) in rows:
                    conn.execute(
                        """INSERT OR REPLACE INTO entity_map
                           (entity_name, chunk_id, project, updated_at)
                           VALUES (?, ?, ?, ?)""",
                        (ent, cid, project, now_str)
                    )
    except Exception:
        pass  # entity_map 反向映射失败不阻塞主流程

    return edge_id


def query_neighbors(conn: sqlite3.Connection,
                    entity: str,
                    project: str = None,
                    direction: str = 'both') -> list:
    """
    迭代304：查询实体的邻居（关系边）。
    direction: 'out' = 以 entity 为 from（出边），
               'in'  = 以 entity 为 to（入边），
               'both'= 双向（默认）。

    返回：[(relation, neighbor_entity, confidence)]

    OS 类比：sysfs /sys/module/<mod>/holders/（入边）
           + /sys/module/<mod>/depends（出边）
           — 双向查询模块依赖关系。
    """
    results = []
    proj_filter = "AND project=?" if project else ""

    if direction in ('out', 'both'):
        # 出边：from_entity = entity
        params = [entity] + ([project] if project else [])
        rows = conn.execute(
            f"SELECT relation, to_entity, confidence FROM entity_edges "
            f"WHERE from_entity=? {proj_filter} ORDER BY confidence DESC",
            params,
        ).fetchall()
        results.extend(rows)

    if direction in ('in', 'both'):
        # 入边：to_entity = entity（返回时用 '-' 前缀标记方向）
        params = [entity] + ([project] if project else [])
        rows = conn.execute(
            f"SELECT relation, from_entity, confidence FROM entity_edges "
            f"WHERE to_entity=? {proj_filter} ORDER BY confidence DESC",
            params,
        ).fetchall()
        results.extend(rows)

    return results


# ── 迭代310：Spreading Activation（OS 类比：CPU cache prefetch + L2 warm-up）──
#
# OS 类比：CPU prefetcher 在 L1 miss 后，顺序预取相邻 cache line 到 L2，
#   使得后续访问几乎零延迟。Spreading Activation 在 FTS5 命中 chunk A 后，
#   沿 entity_edges 一/二跳扩散邻居到候选集（带 decay 权重），
#   类比 prefetch 把"可能相关"的 chunk 提前加入评分池。
#
# Collins & Loftus (1975) Spreading Activation Theory：
#   激活从一个节点沿关系边传播，每跳乘以 decay 系数（0 < decay < 1）。
#   激活强度 = 边的置信度 × decay^跳数。

def spreading_activate(
    conn,
    hit_chunk_ids: list,
    project: str = None,
    decay: float = 0.7,
    max_hops: int = 2,
    existing_ids: set = None,
    max_activation_bonus: float = 0.4,
    edge_half_life_days: float = 90.0,
    distance_decay_enabled: bool = None,
    distance_decay_factor: float = None,
) -> dict:
    """
    迭代310：从 FTS5 命中的 chunk 出发，沿 entity_edges 扩散激活邻居 chunk。

    算法：
      1. 将 hit_chunk_ids 中每个 chunk 映射到其 entity_name（通过 entity_map）
      2. 对每个 entity，查询 entity_edges 一跳邻居（带 confidence）
      3. 对一跳邻居的 entity，再查二跳邻居（max_hops 控制深度）
      4. 将邻居 entity 映射回 chunk_id（通过 entity_map）
      5. 计算激活分：effective_confidence × decay^跳数，上限 max_activation_bonus
      6. 跳过 existing_ids 中已有的 chunk

    iter387: Temporal Edge Decay — 关联强度时间衰减
      认知科学：Collins & Loftus (1975) Spreading Activation Model —
        关联强度随时间衰减（忘却导致联想路径弱化），频繁激活的路径强化（LTP）。
      OS 类比：ARP Cache TTL — 过期条目 confidence 降低，直到 GC 或刷新。
      effective_confidence = confidence × exp(-λ × days_since_created)
        λ = ln(2) / edge_half_life_days（默认 90 天半衰期）
        90 天后 confidence 折半，365 天后折至约 7%，防止旧关联路径持续污染激活。
      edge_half_life_days=0 表示禁用时间衰减（保持原行为）。

    iter393: Semantic Distance Decay — 语义距离衰减
      认知科学：Collins & Loftus (1975) — 激活从锚点沿语义图扩散时，随距离衰减：
        "cat" → "animal"（1跳，强激活）→ "mammal"（2跳，弱激活）
        距离越远语义相关性越低，激活量应按距离梯度衰减而非等权传播。
      OS 类比：NUMA 局部性 — 同节点访问快，跨 2 节点延迟呈指数增长（不是线性）。
      实现：每跳额外乘以 distance_decay_factor（独立于 edge confidence 的 decay），
        hop=1 时：score × distance_decay_factor^1
        hop=2 时：score × distance_decay_factor^2
        (distance_decay_factor < 1.0，典型值 0.6，2 跳约为 0.36)
      distance_decay_enabled=False 时退化到旧行为（只有 confidence-weighted decay）。

    Returns:
      {chunk_id: activation_score} — 仅包含新增的邻居 chunk
    """
    if existing_ids is None:
        existing_ids = set()

    hit_set = set(hit_chunk_ids) | existing_ids

    # Step 1: chunk_id → entity_name（通过 entity_map）
    if not hit_chunk_ids:
        return {}

    ph = ",".join("?" * len(hit_chunk_ids))
    proj_filter = "AND project=?" if project else ""
    params = list(hit_chunk_ids) + ([project] if project else [])
    try:
        entity_rows = conn.execute(
            f"SELECT entity_name, chunk_id FROM entity_map "
            f"WHERE chunk_id IN ({ph}) {proj_filter}",
            params,
        ).fetchall()
    except Exception:
        return {}

    if not entity_rows:
        return {}

    # BFS 沿 entity_edges 扩散
    # frontier: {entity_name: accumulated_score}
    frontier = {row[0]: 1.0 for row in entity_rows}  # seed entities 激活强度 = 1.0
    visited_entities = set(frontier.keys())
    activation: dict = {}  # chunk_id → best_activation_score

    # iter387: Temporal Edge Decay 参数
    import math as _math
    _edge_decay_enabled = edge_half_life_days > 0
    if _edge_decay_enabled:
        _edge_lambda = _math.log(2) / edge_half_life_days  # λ = ln(2)/T½
    from datetime import datetime as _dt, timezone as _tz
    _now_ts = _dt.now(_tz.utc).timestamp()

    # iter393: Semantic Distance Decay 参数
    # 从 sysctl 读取（调用者可通过参数覆盖，不传则从 config 读）
    if distance_decay_enabled is None:
        try:
            from config import get as _cget393
            distance_decay_enabled = _cget393("retriever.sa_distance_decay_enabled")
        except Exception:
            distance_decay_enabled = True
    if distance_decay_factor is None:
        try:
            from config import get as _cget393f
            distance_decay_factor = float(_cget393f("retriever.sa_distance_decay_factor") or 0.6)
        except Exception:
            distance_decay_factor = 0.6

    def _effective_confidence(conf: float, created_at_str: str) -> float:
        """iter387: 计算时间衰减后的有效 confidence。"""
        if not _edge_decay_enabled or not created_at_str:
            return conf
        try:
            # 解析 created_at（ISO 8601，可能含 +00:00 或无时区）
            ca = created_at_str.replace("Z", "+00:00")
            created_ts = _dt.fromisoformat(ca).timestamp()
            days_old = (_now_ts - created_ts) / 86400.0
            if days_old <= 0:
                return conf
            # exponential decay: conf × e^(-λ×days)
            decayed = conf * _math.exp(-_edge_lambda * days_old)
            return max(decayed, 0.01)  # 最低 0.01，防止完全失活
        except Exception:
            return conf

    # ── iter423: Fan Effect — IDF加权 Spreading Activation（Anderson 1974）──
    # entity degree 越高（扇出越大），该 entity 传播的激活权重越低。
    # 使用懒加载缓存：第一次遇到 entity 时查询其 degree，后续复用。
    _fan_effect_enabled = False
    _fan_min_degree = 3
    _fan_idf_weight = 0.5
    _entity_degree_cache: dict = {}  # entity_name → degree
    try:
        from config import get as _cfg_fan
        _fan_effect_enabled = _cfg_fan("retriever.fan_effect_enabled")
        _fan_min_degree = _cfg_fan("retriever.fan_effect_min_degree")
        _raw_idf_w = _cfg_fan("retriever.fan_effect_idf_weight")
        _fan_idf_weight = float(_raw_idf_w) if _raw_idf_w is not None else 0.5
    except Exception:
        pass

    def _fan_idf_factor(entity: str, degree: int, median_deg: float) -> float:
        """iter423: 计算 Fan Effect IDF 折扣系数。degree 越高，返回值越低（最低 0.1）。"""
        if not _fan_effect_enabled or degree < _fan_min_degree:
            return 1.0
        # IDF = log(1 + median / (1 + degree)) / log(1 + median/1)
        # 归一化：fan_min_degree 时 ≈ 1.0，degree→∞ 时 → 0.0
        import math as _m
        idf_raw = _m.log(1.0 + max(1.0, median_deg) / (1.0 + degree))
        idf_norm_max = _m.log(1.0 + max(1.0, median_deg))
        idf = idf_raw / idf_norm_max if idf_norm_max > 0 else 1.0
        idf = max(0.1, min(1.0, idf))
        # Mix: edge_score × (1 - idf_weight × (1 - idf))
        return 1.0 - _fan_idf_weight * (1.0 - idf)

    _fan_median_degree: float = 1.0  # 用于归一化，首次 BFS 后更新

    for hop in range(1, max_hops + 1):
        if decay ** hop < 0.05:  # 激活衰减至 5% 以下时停止
            break

        # iter393: 语义距离衰减系数（distance_decay_factor ^ hop）
        # hop=1: 0.6^1=0.60, hop=2: 0.6^2=0.36
        # OS 类比：NUMA 访问延迟 — 每跨一个 NUMA node，延迟约乘以 1.5-3×
        _dist_decay = (distance_decay_factor ** hop) if distance_decay_enabled else 1.0

        next_frontier = {}

        # ── iter423: Fan Effect — 批量查询当前 frontier entities 的 degree ──
        if _fan_effect_enabled and frontier:
            _uncached = [e for e in frontier if e not in _entity_degree_cache]
            if _uncached:
                try:
                    _uc_ph = ",".join("?" * len(_uncached))
                    _deg_rows = conn.execute(
                        f"SELECT entity, COUNT(*) as deg FROM ("
                        f"  SELECT from_entity as entity FROM entity_edges WHERE from_entity IN ({_uc_ph})"
                        f"  UNION ALL"
                        f"  SELECT to_entity as entity FROM entity_edges WHERE to_entity IN ({_uc_ph})"
                        f") GROUP BY entity",
                        _uncached + _uncached,
                    ).fetchall()
                    for _dr in _deg_rows:
                        _entity_degree_cache[_dr[0]] = int(_dr[1])
                except Exception:
                    pass
            # Update median degree for normalization
            if _entity_degree_cache:
                _sorted_degs = sorted(_entity_degree_cache.values())
                _mid = len(_sorted_degs) // 2
                _fan_median_degree = float(_sorted_degs[_mid]) if _sorted_degs else 1.0

        for entity, parent_score in frontier.items():
            proj_params = [entity] + ([project] if project else [])
            # iter423: Fan Effect — 获取 entity 的扇出惩罚系数
            _entity_deg = _entity_degree_cache.get(entity, 0)
            _fan_factor = _fan_idf_factor(entity, _entity_deg, _fan_median_degree)
            try:
                edges = conn.execute(
                    f"SELECT CASE WHEN from_entity=? THEN to_entity ELSE from_entity END as neighbor, "
                    f"confidence, created_at FROM entity_edges "
                    f"WHERE (from_entity=? OR to_entity=?) {proj_filter} "
                    f"ORDER BY confidence DESC LIMIT 20",
                    [entity, entity, entity] + ([project] if project else []),
                ).fetchall()
            except Exception:
                continue

            for neighbor, confidence, created_at in edges:
                if neighbor in visited_entities:
                    continue
                # iter387: 应用时间衰减
                eff_conf = _effective_confidence(confidence, created_at)
                # iter393: 每跳乘以语义距离衰减（_dist_decay = factor^hop）
                # iter423: 乘以 Fan Effect IDF 惩罚（高扇出 entity 激活权重降低）
                edge_score = parent_score * eff_conf * decay * _dist_decay * _fan_factor
                if edge_score < 0.05:
                    continue
                if neighbor not in next_frontier or next_frontier[neighbor] < edge_score:
                    next_frontier[neighbor] = edge_score

        if not next_frontier:
            break

        # neighbor entity → chunk_id
        neighbor_list = list(next_frontier.keys())
        ne_ph = ",".join("?" * len(neighbor_list))
        ne_params = neighbor_list + ([project] if project else [])
        try:
            chunk_rows = conn.execute(
                f"SELECT entity_name, chunk_id FROM entity_map "
                f"WHERE entity_name IN ({ne_ph}) {proj_filter}",
                ne_params,
            ).fetchall()
        except Exception:
            chunk_rows = []

        for ent_name, cid in chunk_rows:
            if cid in hit_set:
                continue
            score = min(next_frontier[ent_name], max_activation_bonus)
            if cid not in activation or activation[cid] < score:
                activation[cid] = score

        visited_entities.update(next_frontier.keys())
        frontier = next_frontier

    return activation


# ── iter380：Schema Anchoring — Bartlett (1932) Schema Theory ────────────────
#
# 认知科学：Bartlett (1932) Schema Theory — 人的记忆不是存储原始事实，
#   而是将新信息嵌入已有 schema（知识结构框架）中存储。
#   检索时，激活 schema → 框架内的所有关联知识一起浮现。
#
# OS 类比：Linux SLUB Allocator kmem_cache —
#   相同类型的对象共享 kmem_cache（schema），
#   新对象（chunk）写入时归属对应 cache；
#   内存压力时，cache 整体作为回收单元（schema-level eviction）；
#   cache 命中时，同 slab 的相邻对象自动预热（schema spreading）。
#
# 实现：
#   anchor_chunk_schema(conn, chunk_id, summary, project)
#     — 写入时扫描 summary 匹配预定义 schema 规则，写入 schema_anchors 绑定行
#   schema_spread_activate(conn, hit_chunk_ids, project)
#     — 检索时：命中 chunk → 查 schema_anchors → 激活同 schema 的其他 chunk

# 预定义 Schema 规则（基于实际使用场景，按特异性排序）
# 格式：(schema_name, [关键词正则], confidence)
_SCHEMA_RULES = [
    # web_service_config — 服务端口/URL/主机/协议配置（解决"忘记端口"核心问题）
    ("web_service_config",
     [r'\b(?:port|端口|listen|bind)\b.*?\d{2,5}',
      r'\b\d{2,5}\s*(?:port|端口)',
      r'(?:localhost|127\.0\.0\.1|0\.0\.0\.0):\d{2,5}',
      r'(?:http|https|ws|grpc|tcp)://[^\s]+:\d{2,5}',
      r'(?:前端|backend|server|service)\s*(?:端口|port)\s*[=:]\s*\d{2,5}'],
     0.90),
    # auth_config — 认证/授权配置
    ("auth_config",
     [r'\b(?:token|api.?key|secret|password|credential|auth)\b.{0,30}(?:=|:)\s*\S+',
      r'(?:bearer|jwt|oauth|api.?key|密钥|认证|鉴权)',
      r'\b(?:GITHUB_TOKEN|OPENAI_API_KEY|AWS_SECRET)\b'],
     0.85),
    # performance_constraint — 性能约束/指标
    ("performance_constraint",
     [r'\d+(?:\.\d+)?\s*(?:ms|μs|us|s)\b.*(?:latency|延迟|timeout|超时)',
      r'(?:p99|p95|p50|avg)\s*[=<>:]\s*\d+\s*ms',
      r'(?:throughput|qps|rps|tps)\s*[=<>:]\s*\d+',
      r'(?:性能|延迟|吞吐).{0,20}(?:上限|限制|要求|不超过)'],
     0.80),
    # dependency_config — 依赖版本/包管理
    ("dependency_config",
     [r'(?:requirements|package\.json|pyproject|Cargo\.toml)',
      r'\b(?:pip install|npm install|cargo add|go get)\b',
      r'[a-z][a-z0-9_-]+==\d+\.\d+',
      r'"[a-z][a-z0-9_-]+":\s*"\d+\.\d+'],
     0.75),
    # error_pattern — 错误/异常/崩溃模式
    ("error_pattern",
     [r'(?:错误|error|exception|crash|bug|失败|failure)\s*[:：]\s*.{5,}',
      r'(?:fix|修复|resolved|fixed)\s*.{5,}(?:bug|error|crash|issue)',
      r'(?:AttributeError|KeyError|TypeError|RuntimeError|ValueError)',
      r'(?:segment fault|segfault|oom killer|memory leak)'],
     0.80),
    # design_decision — 架构/设计决策
    ("design_decision",
     [r'(?:选择|决定|采用|放弃|不用)\s*.{3,30}\s*(?:因为|原因|而非)',
      r'(?:设计决策|architectural decision|trade.?off)',
      r'(?:替代方案|alternative)\s*.{3,}被?放弃',
      r'(?:不推荐|deprecated|不使用)\s*.{5,}'],
     0.75),
    # database_config — 数据库/存储配置
    ("database_config",
     [r'(?:sqlite|postgres|mysql|redis|mongodb)\s*(?::|\s+at\s+)\s*\S+',
      r'(?:db|database|数据库)\s*(?:path|路径|host|port)\s*[=:]\s*\S+',
      r'(?:store\.db|\.db|\.sqlite)'],
     0.80),
]

# 预编译正则（模块级，避免每次调用重新编译）
_COMPILED_SCHEMA_RULES = [
    (name, [re.compile(p, re.IGNORECASE) for p in patterns], conf)
    for name, patterns, conf in _SCHEMA_RULES
]


def anchor_chunk_schema(
    conn: sqlite3.Connection,
    chunk_id: str,
    summary: str,
    project: str,
) -> int:
    """
    iter380：写入 chunk 时，扫描 summary 匹配 schema 规则，写入 schema_anchors 绑定行。

    算法：
      1. 遍历 _COMPILED_SCHEMA_RULES，对 summary 做正则匹配
      2. 任意规则命中 → INSERT OR IGNORE INTO schema_anchors
      3. 返回写入的绑定数（0=无命中）

    性能：< 1ms（纯正则，无 LLM，模块级预编译）
    OS 类比：kmem_cache_alloc — 新对象分配时，根据 size/align 归属正确的 kmem_cache
    """
    if not summary or not chunk_id:
        return 0

    now_iso = datetime.now(timezone.utc).isoformat()
    written = 0

    for schema_name, compiled_patterns, confidence in _COMPILED_SCHEMA_RULES:
        for pat in compiled_patterns:
            if pat.search(summary):
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO schema_anchors "
                        "(chunk_id, schema_name, project, confidence, created_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (chunk_id, schema_name, project, confidence, now_iso),
                    )
                    written += 1
                except Exception:
                    pass
                break  # 同一 schema 内，第一个命中即可，不重复插入

    return written


def schema_spread_activate(
    conn: sqlite3.Connection,
    hit_chunk_ids: list,
    project: str,
    max_per_schema: int = 3,
    activation_score: float = 0.25,
    existing_ids: set = None,
) -> dict:
    """
    iter380：从 FTS5 命中的 chunk 出发，通过 schema_anchors 激活同 schema 的其他 chunk。

    算法：
      1. 查 hit_chunk_ids 所属的所有 schema（schema_anchors）
      2. 对每个 schema，查询同 project 下同 schema 的其他 chunk（排除已有的）
      3. 按 importance DESC 排序，每个 schema 最多取 max_per_schema 个
      4. 返回 {chunk_id: activation_score}

    与 spreading_activate 的区别：
      spreading_activate 沿 entity_edges 图扩散（Collins & Loftus 1975）
      schema_spread_activate 沿 schema 框架激活（Bartlett 1932）—— 更高层次的语义聚合

    OS 类比：SLUB allocator partial list — kmem_cache 命中后，
      同 slab 的 partial list 中的对象自动成为候选（schema-level prefetch）
    """
    if not hit_chunk_ids:
        return {}
    if existing_ids is None:
        existing_ids = set()

    all_excluded = set(hit_chunk_ids) | existing_ids

    # Step 1: 找到命中 chunk 所属的 schema
    ph = ",".join("?" * len(hit_chunk_ids))
    try:
        schema_rows = conn.execute(
            f"SELECT DISTINCT schema_name FROM schema_anchors "
            f"WHERE chunk_id IN ({ph}) AND project=?",
            list(hit_chunk_ids) + [project],
        ).fetchall()
    except Exception:
        return {}

    if not schema_rows:
        return {}

    schema_names = [r[0] for r in schema_rows]
    result: dict = {}

    # Step 2: 对每个 schema，激活同 schema 的其他 chunk
    for schema_name in schema_names:
        try:
            related = conn.execute(
                "SELECT sa.chunk_id, mc.importance "
                "FROM schema_anchors sa "
                "JOIN memory_chunks mc ON mc.id = sa.chunk_id "
                "WHERE sa.schema_name=? AND sa.project=? "
                "ORDER BY mc.importance DESC "
                "LIMIT ?",
                (schema_name, project, max_per_schema + len(all_excluded)),
            ).fetchall()
        except Exception:
            continue

        count = 0
        for cid, importance in related:
            if cid in all_excluded:
                continue
            if cid not in result or result[cid] < activation_score:
                result[cid] = activation_score
            count += 1
            if count >= max_per_schema:
                break

    return result


# ── 迭代305：Curiosity Queue API（OS 类比：kswapd 水位触发 + 任务队列）────────
#
# OS 类比概述：
#   Linux kswapd 在 free pages < WMARK_LOW 时唤醒，异步回收内存；
#   类似地，retriever 在 FTS top-1 score < 0.25（知识低水位）时，
#   将 query 写入 curiosity_queue，deep-sleep 阶段异步填充知识空白。
#
#   enqueue_curiosity  ≡ wakeup_kswapd()  — 水位低时触发，幂等防重复
#   pop_curiosity_queue ≡ kswapd_do_work() — 取出任务并标记"正在处理"

def enqueue_curiosity(conn: sqlite3.Connection,
                      query: str, project: str,
                      top_score: float = None) -> int:
    """
    迭代305：将「弱命中 query」入队到 curiosity_queue。
    幂等：同 project+query 7天内已有记录则跳过，返回 0；否则插入，返回 1。

    OS 类比：wakeup_kswapd(zone, order) — 检测到水位不足时唤醒 kswapd，
      若 kswapd 已在运行（任务已入队），不重复触发（幂等语义）。
      7天 TTL = 知识填充周期（类比 kswapd 的 watermark_boost_factor 衰减窗口）。

    参数：
      conn      — DB 连接（调用方持有）
      query     — 触发弱命中的原始查询字符串
      project   — 所属项目 ID
      top_score — 触发时的 FTS top-1 分数（记录诊断用）

    返回：
      1 = 成功入队（新记录）
      0 = 幂等跳过（7天内已有相同 project+query）
    """
    now = datetime.now(timezone.utc)
    cutoff = (now - timedelta(days=7)).isoformat()
    now_iso = now.isoformat()

    # 幂等检查：7天内同 project+query 已存在则跳过
    # OS 类比：page_is_in_reclaim() — 页面已在回收队列中，不重复加入
    existing = conn.execute(
        "SELECT id FROM curiosity_queue "
        "WHERE project=? AND query=? AND detected_at >= ? AND status IN ('pending','processing')",
        (project, query, cutoff)
    ).fetchone()
    if existing:
        return 0  # 幂等跳过

    conn.execute(
        "INSERT INTO curiosity_queue (query, project, detected_at, top_score, status) "
        "VALUES (?, ?, ?, ?, 'pending')",
        (query, project, now_iso, top_score)
    )
    return 1


def pop_curiosity_queue(conn: sqlite3.Connection,
                        project: str = None,
                        limit: int = 5) -> List[dict]:
    """
    迭代305：从 curiosity_queue 取出最多 limit 条 pending 记录，
    并将其状态原子更新为 processing，返回条目列表。

    OS 类比：kswapd_do_work() + lru_deactivate_folio() —
      kswapd 从 inactive LRU list 取出 folio（pending→processing），
      标记后进行异步 swap-out；若其间进程访问该页（填充完成），
      状态变为 filled（swap-in 完成），否则 dismissed（放弃回收）。

    参数：
      conn    — DB 连接（调用方持有）
      project — 限定项目（None = 全库）
      limit   — 最多取出数量（默认 5，类比 kswapd 每轮 nr_to_reclaim）

    返回：
      list of dict，每条含 {id, query, project, detected_at, top_score, status}
      返回时 status 已改为 "processing"
    """
    # 查询 pending 条目
    # OS 类比：isolate_lru_folios() — 从 inactive LRU 隔离一批页面用于回收
    if project is not None:
        rows = conn.execute(
            "SELECT id, query, project, detected_at, top_score, status "
            "FROM curiosity_queue "
            "WHERE project=? AND status='pending' "
            "ORDER BY detected_at ASC LIMIT ?",
            (project, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, query, project, detected_at, top_score, status "
            "FROM curiosity_queue "
            "WHERE status='pending' "
            "ORDER BY detected_at ASC LIMIT ?",
            (limit,)
        ).fetchall()

    if not rows:
        return []

    # 批量更新状态为 processing
    # OS 类比：SetPageReclaim(folio) — 设置页面回收标志，阻止其他路径重复处理
    ids = [r[0] for r in rows]
    placeholders = ",".join("?" * len(ids))
    conn.execute(
        f"UPDATE curiosity_queue SET status='processing' WHERE id IN ({placeholders})",
        ids
    )

    return [
        {
            "id": r[0],
            "query": r[1],
            "project": r[2],
            "detected_at": r[3],
            "top_score": r[4],
            "status": "processing",  # 返回更新后的状态
        }
        for r in rows
    ]


# ══════════════════════════════════════════════════════════════════════════════
# 迭代311：认知科学三机制 — Reconsolidation / Active Suppression / Sleep Consolidation
# 设计哲学：人类记忆不是静态存储再提取，而是动态演化的活跃系统。
#   每次召回都是一次重写（再巩固），
#   不被使用的记忆被主动抑制（而非被动衰减），
#   睡眠期间自动整合高频知识（巩固转移）。
# ══════════════════════════════════════════════════════════════════════════════


def reconsolidate(
    conn: sqlite3.Connection,
    recalled_chunk_ids: list,
    query: str,
    project: str = None,
    boost: float = 0.03,
    max_importance: float = 0.98,
) -> int:
    """
    迭代311-A（iter395 扩展）：再巩固（Reconsolidation，Nader et al. 2000）
    每次 chunk 被召回后，根据 query 匹配深度小幅上调 importance。

    神经科学背景：
      记忆每次被检索后进入"不稳定窗口"（labile state），
      随后以更新的形式重新巩固（re-stabilization）。
      重复且深度匹配的召回 → importance 上升（长期增强，LTP）。

    iter395 扩展 — Retrieval-Induced Reconsolidation（取回触发差异化强化）：
      1. Emotional Multiplier (McGaugh 2000)：情绪记忆再巩固效果更强
         emotional_weight > 0.4 → boost × (1 + emotional_weight × 0.5)
         根据：杏仁核激活增强海马突触可塑性（LTP），高情绪词汇的记忆在
         每次提取后更新时获得额外的 norepinephrine 加固。

      2. Frequency Gradient (Roediger & Karpicke 2006 Testing Effect)：
         首次被召回的强化效果 > 高频反复召回
         access_count ≤ 3 → boost × 1.5（测试效果最强窗口）
         access_count > 10 → boost × 0.7（已高度巩固，边际效益递减）
         根据：间隔效应研究表明首次成功检索带来最大的记忆固化增益。

      3. Co-Retrieval Association Strengthening（同次召回关联强化）：
         同一次检索中被一起召回的 chunk 对，其 entity_edge confidence += 0.02
         根据：Hebb (1949) "neurons that fire together, wire together"
         类比：CPU 的 hardware prefetcher 学习 memory access pattern —
         常一起命中的 cache line 对被记入 stride predictor。

    OS 类比：Linux ARC（Adaptive Replacement Cache）— 被反复命中的页面
      从 T1（最近访问）晋升到 T2（频繁访问），淘汰优先级降低。
      iter395 新增：T2 晋升的强度按页面的"热度梯度"差异化。

    Returns:
      更新的 chunk 数量
    """
    if not recalled_chunk_ids or not query:
        return 0

    # 提取 query 词集（英文词 + 中文双字）
    import re as _re
    query_words: set = set()
    for m in _re.finditer(r'[a-zA-Z\u4e00-\u9fff][a-zA-Z0-9\u4e00-\u9fff]{1,}', query.lower()):
        query_words.add(m.group())
    cn = _re.sub(r'[^\u4e00-\u9fff]', '', query)
    for i in range(len(cn) - 1):
        query_words.add(cn[i:i + 2])

    if not query_words:
        return 0

    ph = ",".join("?" * len(recalled_chunk_ids))
    proj_filter = "AND project=?" if project else ""
    params = recalled_chunk_ids + ([project] if project else [])

    try:
        rows = conn.execute(
            f"SELECT id, summary, importance, "
            f"COALESCE(emotional_weight, 0.0), COALESCE(access_count, 0) "
            f"FROM memory_chunks "
            f"WHERE id IN ({ph}) {proj_filter}",
            params,
        ).fetchall()
    except Exception:
        # fallback: older schema without emotional_weight
        try:
            rows = conn.execute(
                f"SELECT id, summary, importance, 0.0, COALESCE(access_count, 0) "
                f"FROM memory_chunks "
                f"WHERE id IN ({ph}) {proj_filter}",
                params,
            ).fetchall()
        except Exception:
            return 0

    now_iso = datetime.now(timezone.utc).isoformat()
    updated = 0
    for row in rows:
        cid, summary, importance, emotional_weight, access_count = (
            row[0], row[1] or "", row[2] or 0.5, float(row[3] or 0.0), int(row[4] or 0)
        )
        # 计算 summary 词集与 query 的 Jaccard 重叠
        s_words: set = set()
        for m in _re.finditer(r'[a-zA-Z\u4e00-\u9fff][a-zA-Z0-9\u4e00-\u9fff]{1,}', summary.lower()):
            s_words.add(m.group())
        scn = _re.sub(r'[^\u4e00-\u9fff]', '', summary)
        for i in range(len(scn) - 1):
            s_words.add(scn[i:i + 2])

        if not s_words:
            overlap_ratio = 0.0
        else:
            overlap_ratio = len(query_words & s_words) / len(query_words | s_words)

        # 至少给最低 boost（被召回本身就是强化信号）
        actual_boost = boost * (0.3 + 0.7 * overlap_ratio)

        # iter395-1: Emotional Multiplier — 情绪记忆再巩固效果更强
        # emotional_weight > 0.4 → boost 乘以 (1 + ew × 0.5)
        # 最大乘数 1.5（ew=1.0 → ×1.5）
        if emotional_weight > 0.4:
            _em_multiplier = 1.0 + emotional_weight * 0.5
            actual_boost *= _em_multiplier

        # iter395-2: Frequency Gradient — 首次召回效果最强，高频边际递减
        # access_count ≤ 3  → ×1.5（测试效果最强窗口，Roediger 2006）
        # 4 ≤ count ≤ 10   → ×1.0（正常强化）
        # count > 10        → ×0.7（已高度巩固，边际递减）
        if access_count <= 3:
            actual_boost *= 1.5
        elif access_count > 10:
            actual_boost *= 0.7

        new_importance = min(importance + actual_boost, max_importance)

        if new_importance > importance + 0.001:  # 避免浮点噪音触发无意义写入
            try:
                conn.execute(
                    "UPDATE memory_chunks SET importance=?, updated_at=? WHERE id=?",
                    (round(new_importance, 4), now_iso, cid),
                )
                updated += 1
            except Exception:
                pass

    # iter395-3: Co-Retrieval Association Strengthening（Hebb 1949）
    # 同一次检索中被一起召回的 chunk 对，增强其 entity_edge confidence
    # 只在至少 2 个 chunk 被召回时触发
    if len(recalled_chunk_ids) >= 2:
        try:
            # 批量提升同次召回 chunk 之间的 entity_edge confidence
            # 找到 recalled chunk 中涉及的 entity_edges（from/to 均在召回集中的边）
            ph2 = ",".join("?" * len(recalled_chunk_ids))
            # 通过 entity_map 找到 recalled_chunk 对应的 entity
            recall_entities = conn.execute(
                f"SELECT entity_name FROM entity_map "
                f"WHERE chunk_id IN ({ph2})" + (" AND project=?" if project else ""),
                recalled_chunk_ids + ([project] if project else []),
            ).fetchall()
            recall_entity_set = {r[0] for r in recall_entities}
            if len(recall_entity_set) >= 2:
                # 提升这些 entity 之间的边 confidence（co-firing → strengthen links）
                _ent_ph = ",".join("?" * len(recall_entity_set))
                _ent_list = list(recall_entity_set)
                conn.execute(
                    f"UPDATE entity_edges "
                    f"SET confidence = MIN(0.99, confidence + 0.02) "
                    f"WHERE from_entity IN ({_ent_ph}) AND to_entity IN ({_ent_ph})",
                    _ent_list + _ent_list,
                )
        except Exception:
            pass  # entity_map/entity_edges 失败不阻塞主流程

    return updated


def find_spaced_review_candidates(
    conn: sqlite3.Connection,
    project: str,
    top_n: int = 5,
    min_importance: float = 0.70,
) -> list:
    """
    iter383：间隔效应主动复习候选 — Spacing Effect Scheduler

    认知科学依据：Ebbinghaus (1885) Spacing Effect + SuperMemo SM-2。
      知识点的记忆强度随时间指数衰减（遗忘曲线），最优复习时机在强度降至
      阈值之前。"间隔复习"比"集中复习"更能建立长期记忆（Cepeda et al. 2006）。
      公式：next_review_at = last_accessed + stability × 86400 秒
      如果 now > next_review_at → chunk 进入"即将遗忘窗口"，应主动复习。

    OS 类比：Linux pdflush（内核 2.5 引入）— 不等到内存压力才 writeback，
      而是根据 dirty_expire_interval 定期扫描并主动刷出 dirty pages。
      这里等价：不等用户查询才检索，而是在 session 开始时主动推送"即将遗忘"的知识。

    候选条件（AND）：
      1. importance >= min_importance（重要知识，值得主动复习）
      2. access_count >= 1（曾被访问过，有访问历史的才有"遗忘"概念）
      3. now - last_accessed > stability × 86400（超过稳定窗口，进入遗忘区间）
      4. chunk_type IN ('decision','design_constraint','reasoning_chain','procedure')
      5. 未被 supersede（不在 knowledge_versions.old_chunk_id 中）

    排序（优先级）：
      urgency = importance / (days_since_last_access / stability)
      urgency 越低 → 越过期，越优先复习

    Returns:
      list of dict: [{id, summary, chunk_type, importance, last_accessed, stability, urgency}]
    """
    import datetime as _dt
    now_iso = _dt.datetime.now(_dt.timezone.utc).isoformat()

    try:
        rows = conn.execute(
            """
            SELECT mc.id, mc.summary, mc.chunk_type, mc.importance,
                   mc.last_accessed, COALESCE(mc.stability, 1.0) AS stability
            FROM memory_chunks mc
            WHERE mc.project = ?
              AND COALESCE(mc.importance, 0) >= ?
              AND COALESCE(mc.access_count, 0) >= 1
              AND mc.chunk_type IN ('decision','design_constraint','reasoning_chain','procedure')
              AND mc.last_accessed IS NOT NULL
              AND (julianday(?) - julianday(mc.last_accessed)) * 86400
                  > COALESCE(mc.stability, 1.0) * 86400
              AND mc.id NOT IN (
                SELECT old_chunk_id FROM knowledge_versions WHERE project=?
              )
            ORDER BY mc.importance DESC
            LIMIT 50
            """,
            (project, min_importance, now_iso, project),
        ).fetchall()
    except Exception:
        return []

    candidates = []
    for row in rows:
        cid, summary, ctype, importance, last_accessed, stability = row
        try:
            _la = _dt.datetime.fromisoformat(last_accessed.replace("Z", "+00:00"))
            _now = _dt.datetime.now(_dt.timezone.utc)
            days_since = (_now - _la.replace(tzinfo=_dt.timezone.utc) if _la.tzinfo is None
                          else _now - _la).total_seconds() / 86400
        except Exception:
            days_since = 9999.0

        if days_since <= 0 or stability <= 0:
            continue
        urgency = importance / (days_since / stability)  # 小 → 更迫切

        candidates.append({
            "id": cid,
            "summary": summary or "",
            "chunk_type": ctype or "",
            "importance": importance or 0.7,
            "last_accessed": last_accessed,
            "stability": stability,
            "urgency": round(urgency, 4),
            "days_overdue": round(days_since - stability, 2),
        })

    # 按 urgency 升序（urgency 小 = 更迫切），取 top_n
    candidates.sort(key=lambda x: x["urgency"])
    return candidates[:top_n]


def suppress_unused(
    conn: sqlite3.Connection,
    injected_chunk_ids: list,
    assistant_response: str,
    project: str = None,
    penalty: float = 0.025,
    min_importance: float = 0.05,
    min_overlap_to_skip: float = 0.04,
) -> int:
    """
    迭代311-B：主动抑制（Active Suppression，Anderson & Green 2001）
    chunk 被注入但 LLM 未使用时，主动下调 importance。

    神经科学背景：
      前额叶皮层通过抑制性神经元主动压制不相关记忆的提取。
      Think/No-Think 实验：有意不去想某件事，大脑会主动抑制该记忆
      的海马激活，导致后续回忆成功率下降。

    OS 类比：Linux vm.swappiness — 主动将"冷"页面推出 RAM，
      而不是等 OOM 才被动淘汰。swappiness 越高，越积极换出不常用页面。

    算法：
      1. 对每个被注入的 chunk，检测其 summary 关键词是否出现在 LLM 回复中
      2. 重叠度 < min_overlap_to_skip → 判定为"未被使用"
      3. importance -= penalty，下限 min_importance

    Returns:
      被抑制的 chunk 数量
    """
    if not injected_chunk_ids or not assistant_response:
        return 0

    import re as _re
    # 提取 LLM 回复词集
    resp_lower = assistant_response.lower()
    resp_words: set = set()
    for m in _re.finditer(r'[a-zA-Z\u4e00-\u9fff][a-zA-Z0-9\u4e00-\u9fff]{1,}', resp_lower):
        resp_words.add(m.group())
    rcn = _re.sub(r'[^\u4e00-\u9fff]', '', assistant_response)
    for i in range(len(rcn) - 1):
        resp_words.add(rcn[i:i + 2])

    ph = ",".join("?" * len(injected_chunk_ids))
    proj_filter = "AND project=?" if project else ""
    params = injected_chunk_ids + ([project] if project else [])

    try:
        rows = conn.execute(
            f"SELECT id, summary, importance FROM memory_chunks "
            f"WHERE id IN ({ph}) {proj_filter}",
            params,
        ).fetchall()
    except Exception:
        return 0

    now_iso = datetime.now(timezone.utc).isoformat()
    suppressed = 0
    for row in rows:
        cid, summary, importance = row[0], row[1] or "", row[2] or 0.5

        # 计算 summary 词集与 LLM 回复的重叠
        s_words: set = set()
        for m in _re.finditer(r'[a-zA-Z\u4e00-\u9fff][a-zA-Z0-9\u4e00-\u9fff]{1,}', summary.lower()):
            s_words.add(m.group())
        scn = _re.sub(r'[^\u4e00-\u9fff]', '', summary)
        for i in range(len(scn) - 1):
            s_words.add(scn[i:i + 2])

        if not s_words or not resp_words:
            overlap = 0.0
        else:
            overlap = len(s_words & resp_words) / len(s_words | resp_words)

        if overlap < min_overlap_to_skip:
            new_importance = max(importance - penalty, min_importance)
            if new_importance < importance - 0.001:
                try:
                    conn.execute(
                        "UPDATE memory_chunks SET importance=?, updated_at=? WHERE id=?",
                        (round(new_importance, 4), now_iso, cid),
                    )
                    suppressed += 1
                except Exception:
                    pass

    return suppressed


def sleep_consolidate(
    conn: sqlite3.Connection,
    project: str,
    session_id: str = "",
    similarity_threshold: float = 0.72,
    stability_boost: float = 1.15,
    stability_decay: float = 0.92,
    active_days: int = 7,
    stale_days: int = 30,
    max_merges: int = 20,
) -> dict:
    """
    迭代311-C：睡眠巩固（Sleep Consolidation，Walker & Stickgold 2004）
    session 结束时自动触发：合并高相似 chunk + stability 动态调整。

    神经科学背景：
      慢波睡眠（SWS）期间，海马将当日编码的情景记忆"回放"给新皮层，
      实现从海马依赖（短期）到皮层依赖（长期）的记忆转移（consolidation）。
      高频激活的记忆获得更强的皮层表征（长期增强，LTP）；
      低频记忆连接弱化（长期抑制，LTD）。

    OS 类比：Linux KSM（Kernel Samepage Merging）+ pdflush
      ksmd 在后台扫描合并相同页面（↔ 合并相似 chunk）；
      pdflush 将 dirty page 按优先级写回磁盘（↔ stability 回写）。

    三个子操作：
      1. 合并高相似 chunk（复用 Jaccard trigram）— 减少冗余
      2. 本 session 高访问 chunk stability × stability_boost（活跃记忆加固）
      3. 长期未访问 chunk stability × stability_decay（不活跃记忆弱化）

    Returns:
      {"merged": N, "boosted": N, "decayed": N}
    """
    import re as _re
    from datetime import datetime as _dt, timezone as _tz, timedelta as _td

    result = {"merged": 0, "boosted": 0, "decayed": 0}
    now = _dt.now(_tz.utc)
    now_iso = now.isoformat()

    proj_filter = "AND project=?" if project else ""
    proj_params = [project] if project else []

    # ── 子操作 1：合并高相似 chunk ────────────────────────────────────────────
    def _trigrams(s: str) -> set:
        s = _re.sub(r'\s+', ' ', s.strip().lower())
        return set(s[i:i + 3] for i in range(len(s) - 2)) if len(s) >= 3 else set(s)

    def _jaccard(a: str, b: str) -> float:
        ta, tb = _trigrams(a), _trigrams(b)
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    try:
        rows = conn.execute(
            f"SELECT id, summary, importance, stability FROM memory_chunks "
            f"WHERE chunk_type NOT IN ('prompt_context','task_state') {proj_filter} "
            f"ORDER BY importance DESC LIMIT 500",
            proj_params,
        ).fetchall()

        merged_ids: set = set()
        merge_count = 0
        for i in range(len(rows)):
            if merge_count >= max_merges:
                break
            if rows[i][0] in merged_ids:
                continue
            for j in range(i + 1, len(rows)):
                if merge_count >= max_merges:
                    break
                if rows[j][0] in merged_ids:
                    continue
                sim = _jaccard(rows[i][1] or "", rows[j][1] or "")
                if sim >= similarity_threshold:
                    # survivor = 高 importance 的那个（rows[i] 因 ORDER BY imp DESC 排前）
                    survivor_id, victim_id = rows[i][0], rows[j][0]
                    victim_imp = rows[j][2] or 0.3
                    survivor_imp = rows[i][2] or 0.5
                    # survivor importance 轻微提升（吸收了 victim 的信号）
                    new_imp = min(0.98, max(survivor_imp, victim_imp) * 1.02)
                    # victim 降为 ghost（importance=0, oom_adj=500）
                    conn.execute(
                        "UPDATE memory_chunks SET importance=0, oom_adj=500, "
                        "summary=?, updated_at=? WHERE id=?",
                        (f"[merged→{survivor_id}] {rows[j][1] or ''}"[:200],
                         now_iso, victim_id),
                    )
                    conn.execute(
                        "UPDATE memory_chunks SET importance=?, updated_at=? WHERE id=?",
                        (round(new_imp, 4), now_iso, survivor_id),
                    )
                    merged_ids.add(victim_id)
                    merge_count += 1
        result["merged"] = merge_count
    except Exception:
        pass

    # ── 子操作 2：本 session 高访问 chunk stability × boost ──────────────────
    try:
        # 本 session 被访问过的 chunk（last_accessed 在 session 期间）
        # 简化：取 last_accessed 在最近 active_days 天内 且 access_count >= 2 的
        cutoff_active = (now - _td(days=active_days)).isoformat()
        conn.execute(
            f"UPDATE memory_chunks SET stability=MIN(365.0, stability * ?), updated_at=? "
            f"WHERE last_accessed >= ? AND access_count >= 2 {proj_filter}",
            [stability_boost, now_iso, cutoff_active] + proj_params,
        )
        result["boosted"] = conn.execute(
            f"SELECT changes()"
        ).fetchone()[0]
    except Exception:
        pass

    # ── 子操作 3：长期未访问 chunk stability × decay（iter400：per-type 个体化衰减）──
    # iter400：以 chunk_type 个体化衰减率替代统一的 stability_decay 参数。
    # 认知科学依据：程序性记忆（design_constraint/procedure）衰减慢；
    #   工作记忆/情节记忆（task_state/prompt_context）衰减快。
    # OS 类比：Linux cgroup memory.reclaim_ratio — per-group 内存回收压力。
    try:
        cutoff_stale = (now - _td(days=stale_days)).isoformat()
        # iter400: per-type 衰减（覆盖全局 stability_decay 参数）
        _decayed = decay_stability_by_type(conn, project, stale_days=stale_days,
                                           now_iso=now_iso)
        result["decayed"] = _decayed
    except Exception:
        # fallback: 使用全局统一衰减率（兼容旧 schema）
        try:
            cutoff_stale = (now - _td(days=stale_days)).isoformat()
            conn.execute(
                f"UPDATE memory_chunks SET stability=MAX(0.1, stability * ?), updated_at=? "
                f"WHERE last_accessed < ? AND access_count < 2 {proj_filter}",
                [stability_decay, now_iso, cutoff_stale] + proj_params,
            )
            result["decayed"] = conn.execute("SELECT changes()").fetchone()[0]
        except Exception:
            pass

    # ── 子操作 4（迭代319）：情节 chunk 巩固扫描 ──────────────────────────────
    # OS 类比：khugepaged 在 kswapd 回收后，再扫描高频访问小页面尝试合并
    try:
        ep_result = episodic_decay_scan(conn, project, stale_days=stale_days)
        result["episodic_decayed"] = ep_result.get("decayed", 0)
        result["episodic_promoted"] = ep_result.get("promoted", 0)
        result["episodic_inplace_promoted"] = ep_result.get("inplace_promoted", 0)  # iter379
        result["new_semantic_ids"] = ep_result.get("new_semantic_ids", [])
    except Exception:
        pass

    return result


# ── 迭代315：情境感知注入 — 编码情境提取 ─────────────────────────
# OS 类比：Linux perf_event context — 记录性能事件时附带 CPU/task 上下文，
#   使后续分析能区分「在什么场景下发生的」，而不只是「发生了什么」。
# 认知科学依据：Encoding Specificity (Tulving 1973) — 检索线索与编码时线索
#   重叠越高，记忆提取成功率越高。

def extract_encoding_context(text: str) -> dict:
    """迭代315: 从文本提取编码情境特征（纯正则，不调LLM）。

    返回 dict:
      session_type: debug/design/review/refactor/qa/unknown
      entities:     核心实体词列表（≤8个）
      task_verbs:   动作类词列表（≤5个）
    """
    import re as _re
    # session_type 关键词规则
    _TYPE_RULES = [
        ("debug",    r'调试|报错|错误|traceback|exception|bug|fix|error|failed'),
        ("design",   r'设计|架构|方案|决策|接口|API|schema|interface'),
        ("review",   r'审查|review|PR|merge|代码审核|LGTM'),
        ("refactor", r'重构|重写|迁移|refactor|cleanup|rename'),
        ("qa",       r'测试|验证|test|assert|pytest|passed|failed'),
    ]
    session_type = "unknown"
    for stype, pat in _TYPE_RULES:
        if _re.search(pat, text, _re.IGNORECASE):
            session_type = stype
            break

    # entities：反引号内容 + 英文驼峰/下划线词
    entities = []
    seen = set()
    for m in _re.finditer(r'`([^`]{2,30})`', text[:2000]):
        w = m.group(1).strip()
        if w not in seen:
            seen.add(w)
            entities.append(w)
    _STOP = frozenset({
        'the', 'and', 'for', 'with', 'from', 'this', 'that', 'are', 'was',
        'has', 'not', 'but', 'can', 'will', 'use', 'new', 'get', 'set',
        'add', 'run',
    })
    for m in _re.finditer(r'\b([A-Z][a-zA-Z0-9]{2,20}|[a-z][a-z0-9_]{3,20})\b', text[:2000]):
        w = m.group(1)
        if w not in seen and w.lower() not in _STOP:
            seen.add(w)
            entities.append(w)
        if len(entities) >= 10:
            break

    # task_verbs：中文动作词
    task_verbs = []
    _VERB_PATS = [
        r'(?:修复|调试|排查|诊断|定位)',   # debug类
        r'(?:设计|规划|构建|重构|迁移)',    # design类
        r'(?:实现|添加|删除|更新|升级)',    # impl类
        r'(?:测试|验证|检查|确认|评估)',    # qa类
        r'(?:优化|改进|提升|加速|减少)',    # perf类
    ]
    for pat in _VERB_PATS:
        for m in _re.finditer(pat, text[:2000]):
            v = m.group(0)
            if v not in task_verbs:
                task_verbs.append(v)

    return {
        "session_type": session_type,
        "entities": entities[:8],
        "task_verbs": task_verbs[:5],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 迭代317：前摄干扰控制（Proactive Interference Control）
# 认知科学基础：Proactive Interference (PI) — Müller & Pilzecker 1900，Bartlett 1932
#   旧知识干扰新知识的学习和检索：当新知识与旧知识语义矛盾时，
#   必须明确将旧知识"失效标记"（而不是等待自然衰减），
#   否则检索时两者并存，导致矛盾信息同时注入 LLM 上下文。
#
# OS 类比：Linux kernel module versioning（MODULE_STATE_GOING）：
#   insmod 新版本模块时，旧版本被标记为 GOING，不再接受新请求；
#   这里等价于：旧 chunk 的 importance 降权 + oom_adj 上调，使其在检索排序中沉底。
# ══════════════════════════════════════════════════════════════════════════════

# 冲突检测：否定/替换模式关键词
# 语义：含这些关键词的新 chunk 表示"否定/替换"已有知识
_CONFLICT_NEGATION_PATTERNS = [
    # 直接否定
    r'不使用', r'不采用', r'不推荐', r'不选择', r'不再使用', r'不再采用',
    # 替换/放弃
    r'放弃', r'改用', r'换成', r'替代', r'替换', r'迁移到', r'迁移至',
    # 否定建议/结论
    r'不推荐', r'不建议', r'反对', r'否定',
]

_CONFLICT_NEG_RE = re.compile('|'.join(_CONFLICT_NEGATION_PATTERNS))


def _extract_key_entities(text: str) -> set:
    """
    从文本中提取关键实体词（英文标识符 + CJK bigram）。
    用于 detect_conflict 的词集交集比对。
    """
    entities: set = set()
    # 英文词（含下划线、点，如 BM25 / PostgreSQL / redis_client）
    for m in re.finditer(r'[a-zA-Z][a-zA-Z0-9_.]{1,}', text):
        w = m.group().strip('._').lower()
        if len(w) >= 2:
            entities.add(w)
    # CJK bigram
    cjk = re.sub(r'[^\u4e00-\u9fff]', '', text)
    for i in range(len(cjk) - 1):
        entities.add(cjk[i:i + 2])
    return entities


def detect_conflict(
    conn: sqlite3.Connection,
    new_summary: str,
    chunk_type: str,
    project: str,
) -> list:
    """
    迭代317：检测 new_summary 与 DB 中同类型 chunk 的语义冲突。

    冲突判定逻辑：
      1. new_summary 含否定/替换关键词（否则直接返回 []）
      2. new_summary 与已有 chunk 有实体词交集（即谈论同一对象）
      3. 两条规则同时满足 → 判定为冲突

    只在同 project + 同 chunk_type 内检测（跨类型语义不可比）。

    Returns:
      冲突的旧 chunk ID 列表（可能为空）
    """
    # 快速路径：new_summary 不含否定/替换词 → 不可能冲突
    if not _CONFLICT_NEG_RE.search(new_summary):
        return []

    # 提取 new_summary 的关键实体
    new_entities = _extract_key_entities(new_summary)
    if not new_entities:
        return []

    # 查询同 project + 同 chunk_type 的已有 chunk
    try:
        rows = conn.execute(
            "SELECT id, summary FROM memory_chunks "
            "WHERE project=? AND chunk_type=? AND summary != ''",
            (project, chunk_type),
        ).fetchall()
    except Exception:
        return []

    conflicts = []
    for row in rows:
        cid, existing_summary = row[0], row[1] or ""
        if not existing_summary:
            continue
        existing_entities = _extract_key_entities(existing_summary)
        # 实体词交集 ≥ 1 → 谈论同一对象 → 冲突
        if new_entities & existing_entities:
            conflicts.append(cid)

    return conflicts


def supersede_chunk(
    conn: sqlite3.Connection,
    old_id: str,
    new_id: str,
    reason: str,
    project: str,
    session_id: str = "",
) -> Optional[str]:
    """
    迭代317：将 old_id chunk 标记为被 new_id 取代。

    操作：
      1. 在 knowledge_versions 中写入版本对记录
      2. 旧 chunk importance *= 0.5（降权），oom_adj += 200（更易淘汰）
      3. 若 old_id 不存在，安全返回 new_id（不抛异常）

    Returns:
      new_id（成功）或 None（仅当 old_id 存在但操作异常时）
    """
    now_iso = datetime.now(timezone.utc).isoformat()

    # 检查 old_id 是否存在
    row = conn.execute(
        "SELECT importance, oom_adj FROM memory_chunks WHERE id=?",
        (old_id,),
    ).fetchone()

    # 无论旧 chunk 是否存在，都写入版本对记录（new_id 为知识演化的声明）
    try:
        conn.execute(
            """INSERT INTO knowledge_versions
               (old_chunk_id, new_chunk_id, reason, project, session_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (old_id, new_id, reason, project, session_id, now_iso),
        )
    except Exception:
        pass

    if row is not None:
        old_importance = row[0] if row[0] is not None else 0.5
        old_oom_adj = row[1] if row[1] is not None else 0
        new_importance = round(old_importance * 0.5, 4)
        new_oom_adj = old_oom_adj + 200
        try:
            conn.execute(
                "UPDATE memory_chunks SET importance=?, oom_adj=?, updated_at=? WHERE id=?",
                (new_importance, new_oom_adj, now_iso, old_id),
            )
        except Exception:
            return None

    return new_id


def get_superseded_ids(
    conn: sqlite3.Connection,
    project: str = None,
) -> set:
    """
    迭代317：返回已被取代的旧 chunk ID 集合。

    用途：检索时排除旧版本 chunk，防止矛盾知识注入 LLM 上下文。

    OS 类比：Linux kernel module_state_going_list — 获取所有 MODULE_STATE_GOING
      的模块 ID，供 module_find_or_load() 跳过。

    Returns:
      set of old_chunk_id strings
    """
    try:
        if project:
            rows = conn.execute(
                "SELECT DISTINCT old_chunk_id FROM knowledge_versions WHERE project=?",
                (project,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT DISTINCT old_chunk_id FROM knowledge_versions"
            ).fetchall()
        return {r[0] for r in rows}
    except Exception:
        return set()


# ══════════════════════════════════════════════════════════════════════════════
# 迭代319：情节记忆 vs 语义记忆分离（Episodic/Semantic Memory Separation）
# 认知科学基础：Tulving (1972) 双记忆系统
#   情节记忆（Episodic）：特定时间/情境的事件记忆（"上次会话里决定了X"）
#     - 高时效性，会话结束后快速衰减
#     - 来源：reasoning_chain, conversation_summary, causal_chain
#   语义记忆（Semantic）：去情境化的通用知识（"X系统使用Y算法"）
#     - 高稳定性，慢衰减，可跨会话复用
#     - 来源：decision, design_constraint, procedure
#   转化路径：情节记忆被多次召回（>=3次）→ 自动提升为语义记忆
#     类比：海马短期情节记忆 → 新皮层长期语义存储（记忆固化，consolidation）
#
# OS 类比：Linux huge page compaction + THP（Transparent Huge Pages）
#   小页面（情节 chunk）被频繁访问后合并成大页面（语义 chunk），
#   保留合并记录（episodic_consolidations），原小页面标记为"可回收"。
# ══════════════════════════════════════════════════════════════════════════════

# chunk_type → info_class 映射表
_CHUNK_TYPE_INFO_CLASS: dict = {
    # 情节记忆：高时效性，会话内有效
    "reasoning_chain": "episodic",
    "conversation_summary": "episodic",
    "causal_chain": "episodic",
    # 语义记忆：去情境化通用知识
    "decision": "semantic",
    "design_constraint": "semantic",
    "procedure": "semantic",
    "quantitative_evidence": "semantic",
    # 操作配置：项目内持久
    "task_state": "operational",
    "prompt_context": "operational",
    # 其余 → world（中等保留）
}


def classify_memory_type(chunk_type: str, summary: str) -> str:
    """
    迭代319：根据 chunk_type + summary 特征，推断 info_class。

    优先级：
      1. 内容含"临时/本次/暂时"关键词 → ephemeral（覆盖 chunk_type 映射）
      2. chunk_type 直接映射
      3. excluded_path → semantic
      4. 默认 world

    OS 类比：Linux mm/vma.c vm_area_struct.vm_flags —
      每个 VMA 在 mmap 时被赋予 VM_READ/VM_WRITE/VM_EXEC/VM_SHARED 标志，
      决定该区域的回收策略（shared+dirty → writeback，anon → swap）。

    Returns: 'episodic' | 'semantic' | 'world' | 'operational' | 'ephemeral'
    """
    # chunk_type 直接映射（情节/语义/operational 类型不受内容关键词覆盖）
    if chunk_type in _CHUNK_TYPE_INFO_CLASS:
        return _CHUNK_TYPE_INFO_CLASS[chunk_type]

    # 内容关键词：含"临时"/"本次"/"这次"关键词 → ephemeral
    # 仅作用于未被 chunk_type 映射的类型（避免覆盖明确分类的 episodic/semantic）
    if re.search(r'临时|本次|这次|暂时|测试用', summary):
        return "ephemeral"

    # excluded_path：记录"不做的选择" — 通常是多次验证后的稳定决策 → semantic
    if chunk_type == "excluded_path":
        return "semantic"

    return "world"


# ══════════════════════════════════════════════════════════════════════════════
# 迭代320：情感显著性驱动 importance（Emotional Salience）
# 认知科学基础：McGaugh (2004) "The amygdala modulates the consolidation of
#   memories of emotionally arousing experiences" — 情感唤醒激活杏仁核，
#   杏仁核通过 norepinephrine 调节海马突触可塑性，增强记忆编码。
#   结果：情感标记越强，记忆越优先被固化、越难被遗忘。
#
# 实际映射：
#   高情感唤醒词（紧急/严重/失败/崩溃/突破）→ importance 上调
#   负面情感词（已解决/废弃/过时）→ 轻微下调（"关闭"事件降权）
#   中性词 → 不改变
#
# OS 类比：Linux OOM Killer 的 /proc/[pid]/oom_score_adj —
#   进程可以声明自己的重要性，内核在 OOM 时优先 kill 分数低的进程；
#   情感显著性相当于 chunk 自我声明的"存活优先级"。
# ══════════════════════════════════════════════════════════════════════════════

# 情感唤醒词典（高唤醒 → 正调整，低唤醒 → 负调整）
# 格式：(patterns, delta)
_EMOTIONAL_SALIENCE_RULES: list = [
    # 高唤醒正向：突破/发现/关键
    (r'突破|关键发现|重要发现|核心|必须|严格要求', +0.10),
    # 高唤醒负向：错误/崩溃/失败/紧急
    (r'崩溃|严重错误|critical.*bug|P0|紧急|fatal|panic|CRITICAL', +0.12),
    (r'failed|failure|exception|traceback|ERROR|死锁|data.*loss|数据丢失', +0.08),
    # 英文高唤醒正向
    (r'breakthrough|critical|must|important|key insight|major', +0.08),
    # 情感中性但高价值
    (r'性能瓶颈|bottleneck|O\(N\)|O\(n\^2\)|slow|latency.*high', +0.06),
    # 低唤醒：已解决/已关闭/完成（降权让位新知识）
    (r'已解决|已修复|已关闭|不再需要|obsolete|deprecated|过时|已废弃', -0.08),
    (r'resolved|fixed|closed|no longer|wont.fix|done.*already', -0.06),
]

# 预编译正则（模块加载时一次性）
_EMOTIONAL_SALIENCE_RE: list = [
    (re.compile(pat, re.IGNORECASE), delta)
    for pat, delta in _EMOTIONAL_SALIENCE_RULES
]


def compute_emotional_salience(text: str) -> float:
    """
    迭代320：计算文本的情感显著性分数（delta importance）。

    扫描 text 中的情感唤醒词，累积 delta：
      正向词（紧急/关键/崩溃）→ 累积正 delta
      负向词（已解决/废弃）→ 累积负 delta

    OS 类比：Linux OOM Killer oom_score 计算 — 综合多个维度（内存使用、
      进程优先级、用户调整）计算最终 OOM 分数，越高越被优先杀死。
      这里是情感信号的累积聚合。

    Returns:
      float delta，范围 [-0.20, +0.25]
      delta = 0.0 表示情感中性
    """
    if not text:
        return 0.0
    delta = 0.0
    for pat, d in _EMOTIONAL_SALIENCE_RE:
        if pat.search(text):
            delta += d
    return max(-0.20, min(0.25, delta))


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

    if abs(delta) < 0.01:
        # delta 微弱 — 仍写入 emotional_weight=0（明确表示无情绪显著性）
        # 但只在字段为 NULL 时才写（避免覆盖已有有效值）
        try:
            conn.execute(
                "UPDATE memory_chunks SET emotional_weight=? WHERE id=? AND (emotional_weight IS NULL OR emotional_weight=0)",
                (0.0, chunk_id),
            )
        except Exception:
            pass
        return base_importance

    new_importance = max(0.05, min(0.98, base_importance + delta))
    if abs(new_importance - base_importance) < 0.001:
        new_importance = base_importance

    try:
        conn.execute(
            "UPDATE memory_chunks SET importance=?, emotional_weight=?, updated_at=? WHERE id=?",
            (round(new_importance, 4), emotional_weight,
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

        # 物理删除 — FTS5 content 表通过 DELETE trigger 自动清理索引
        # 参考：schema.py 中 FTS5 设置了 content='memory_chunks' + DELETE trigger
        placeholders = ",".join("?" * len(ghost_ids))
        conn.execute(
            f"DELETE FROM memory_chunks WHERE id IN ({placeholders})",
            ghost_ids,
        )
        reaped = conn.execute("SELECT changes()").fetchone()[0]

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
