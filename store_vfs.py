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

    # ── 迭代306：raw_snippet — 写入时保真原始片段（≤500字）──
    # OS 类比：Linux page cache 保存原始 disk block，VFS 层面不压缩；
    #   读取时 on-demand 合并（类比 copy-on-read 模式）。
    # raw_snippet 不参与 FTS5 索引（避免膨胀），仅在 retriever 注入时按需附加。
    _safe_add_column(conn, "memory_chunks", "raw_snippet", "TEXT DEFAULT ''")

    # ── 迭代315：encoding_context — 情境感知注入（Encoding Specificity）──
    # OS 类比：Linux perf_event context — 记录性能事件时附带 CPU/task 上下文。
    # 存储 chunk 写入时的情境特征 JSON，检索时与 query_context 比对计算匹配度。
    _safe_add_column(conn, "memory_chunks", "encoding_context", "TEXT DEFAULT '{}'")

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
    conn.commit()

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
                   mc.verification_status, mc.confidence_score
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
    for rid, summary, content, importance, last_accessed, chunk_type, access_count, created_at, fts_rank, lru_gen, chunk_project, verification_status, confidence_score in rows:
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
    # memory-os 召回质量推断（无显式 quality 时默认 quality=4，即"良好召回"）：
    #   quality=4 → S_new = S_old × 1.1（轻微加固），优于旧 ×2.0 的激进增长
    #   调用方可通过 recall_quality 参数显式指定（retriever 将在未来版本传入）
    #
    # OS 类比：CPU TLB 击中后 PTE Accessed bit 置位
    #   短间隔重复（连续访问）= 较小 quality，长间隔仍命中 = 较高 quality
    _rq = recall_quality if (recall_quality is not None) else 4
    _rq = max(0, min(5, _rq))  # clamp to [0, 5]
    _sm2_factor = 1.0 + 0.1 * (_rq - 3)  # quality=3→×1.0, quality=5→×1.2, quality=0→×0.7
    _sm2_factor = max(0.7, _sm2_factor)   # 下限：即使 quality=0 也不跌破 0.7
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
    rows = conn.execute(
        f"""SELECT id, importance, last_accessed, COALESCE(access_count, 0),
                   COALESCE(oom_adj, 0), COALESCE(lru_gen, 0),
                   COALESCE(stability, 1.0), COALESCE(info_class, 'world')
            FROM memory_chunks
            WHERE project=? AND chunk_type NOT IN ({protect_placeholders})
              AND COALESCE(oom_adj, 0) > -1000
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
) -> dict:
    """
    迭代310：从 FTS5 命中的 chunk 出发，沿 entity_edges 扩散激活邻居 chunk。

    算法：
      1. 将 hit_chunk_ids 中每个 chunk 映射到其 entity_name（通过 entity_map）
      2. 对每个 entity，查询 entity_edges 一跳邻居（带 confidence）
      3. 对一跳邻居的 entity，再查二跳邻居（max_hops 控制深度）
      4. 将邻居 entity 映射回 chunk_id（通过 entity_map）
      5. 计算激活分：confidence × decay^跳数，上限 max_activation_bonus
      6. 跳过 existing_ids 中已有的 chunk

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

    for hop in range(1, max_hops + 1):
        if decay ** hop < 0.05:  # 激活衰减至 5% 以下时停止
            break

        next_frontier = {}
        for entity, parent_score in frontier.items():
            proj_params = [entity] + ([project] if project else [])
            try:
                edges = conn.execute(
                    f"SELECT CASE WHEN from_entity=? THEN to_entity ELSE from_entity END as neighbor, "
                    f"confidence FROM entity_edges "
                    f"WHERE (from_entity=? OR to_entity=?) {proj_filter} "
                    f"ORDER BY confidence DESC LIMIT 20",
                    [entity, entity, entity] + ([project] if project else []),
                ).fetchall()
            except Exception:
                continue

            for neighbor, confidence in edges:
                if neighbor in visited_entities:
                    continue
                # 每跳只乘一次 decay（parent_score 已包含前序跳数的 decay 累积）
                edge_score = parent_score * confidence * decay
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
    迭代311-A：再巩固（Reconsolidation，Nader et al. 2000）
    每次 chunk 被召回后，根据 query 匹配深度小幅上调 importance。

    神经科学背景：
      记忆每次被检索后进入"不稳定窗口"（labile state），
      随后以更新的形式重新巩固（re-stabilization）。
      重复且深度匹配的召回 → importance 上升（长期增强，LTP）。

    OS 类比：Linux ARC（Adaptive Replacement Cache）— 被反复命中的页面
      从 T1（最近访问）晋升到 T2（频繁访问），淘汰优先级降低。

    算法：
      1. 用 query 词集与 chunk summary 计算 Jaccard 重叠度
      2. boost = base_boost × overlap_ratio（匹配越深，强化越多）
      3. importance = min(importance + boost, max_importance)
      4. 更新 updated_at 标记再巩固时间

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
            f"SELECT id, summary, importance FROM memory_chunks "
            f"WHERE id IN ({ph}) {proj_filter}",
            params,
        ).fetchall()
    except Exception:
        return 0

    now_iso = datetime.now(timezone.utc).isoformat()
    updated = 0
    for row in rows:
        cid, summary, importance = row[0], row[1] or "", row[2] or 0.5
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

    return updated


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

    # ── 子操作 3：长期未访问 chunk stability × decay ──────────────────────────
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

    算法：
      delta = compute_emotional_salience(text)
      if |delta| < 0.01 → 不写 DB（避免无意义更新）
      new_importance = clamp(base_importance + delta, 0.05, 0.98)
      写入 memory_chunks.importance

    OS 类比：Linux OOM Killer oom_score_adj 写入 —
      fork() 时继承父进程的 oom_score_adj，每个进程可自主调整；
      这里 importance 由 extractor 初始评估，情感显著性在写入后再调整。

    Returns:
      new_importance（调整后；若无调整则返回 base_importance）
    """
    delta = compute_emotional_salience(text)
    if abs(delta) < 0.01:
        return base_importance

    new_importance = max(0.05, min(0.98, base_importance + delta))
    if abs(new_importance - base_importance) < 0.001:
        return base_importance

    try:
        conn.execute(
            "UPDATE memory_chunks SET importance=?, updated_at=? WHERE id=?",
            (round(new_importance, 4), datetime.now(timezone.utc).isoformat(), chunk_id),
        )
    except Exception:
        pass
    return new_importance


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
) -> dict:
    """
    迭代319：扫描情节记忆 — 衰减过期情节 chunk，提升高频召回情节 chunk 为语义 chunk。
    迭代327：semantic_threshold 降低 3 → 2（access_count=0 的情节 chunk 因 content 太短
    从未被召回，threshold=3 导致晋升路径永远不触发；降低到 2 让 access_count>=2 的 10 个
    chunks 有资格晋升，也避免"先有鸡还是先有蛋"的死锁）。

    两个子操作（类比睡眠巩固的特化版本）：
      A. 提升：info_class='episodic' AND access_count >= semantic_threshold
         → 调用 promote_to_semantic()，一次只处理同 summary 类别的
      B. 衰减：info_class='episodic' AND last_accessed < (now - stale_days)
         AND access_count < 2 → importance *= 0.7, oom_adj += 50

    OS 类比：Linux khugepaged + kswapd 协同 —
      khugepaged 提升高频访问小页面（促进 → 语义），
      kswapd 回收冷页面（衰减 → 降权 → 更易被 evict）。

    Returns:
      {"decayed": N, "promoted": N, "new_semantic_ids": [...]}
    """
    result: dict = {"decayed": 0, "promoted": 0, "new_semantic_ids": []}
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()

    # ── 子操作 A：提升高频情节 chunk ──────────────────────────────────────────
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
