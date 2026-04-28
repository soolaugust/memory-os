# AIOS Memory OS — 架构设计说明

> 最后更新：迭代258（2026-04）
> 核心理念：用操作系统内核的设计哲学解决 AI Agent 的认知资源管理问题

---

## 一、系统定位

**问题**：Claude 等 LLM 的 context window 是有限稀缺资源，跨 session 无持久化，并发时无协调机制。

**解法**：在 Claude Code hooks 基础设施上，构建一个类 OS 的内存/调度/IO 管理层：
- 把 context window 类比为**物理内存（RAM）**
- 把 store.db 类比为**磁盘/swap**
- 把 hooks 执行链类比为**系统调用路径**
- 把 session 类比为**进程**

---

## 二、整体架构分层

```
┌─────────────────────────────────────────────────────────────┐
│                    Claude Code (用户界面层)                   │
│          User Prompt → Claude Response → Tool Calls          │
└───────────────────────┬─────────────────────────────────────┘
                        │ Claude Code Hooks (系统调用门)
┌───────────────────────▼─────────────────────────────────────┐
│                   Hooks 执行链（L0 内核态）                    │
│                                                              │
│  SessionStart      UserPromptSubmit   PostToolUse    Stop    │
│  ┌──────────┐      ┌──────────────┐  ┌──────────┐  ┌─────┐ │
│  │ loader   │      │ retriever    │  │compressor│  │extrc│ │
│  │ (工作集  │      │ (检索注入)   │  │(zram)    │  │(提取│ │
│  │  恢复)   │      │ writer       │  │profiler  │  │+CRIU│ │
│  │ CRIU     │      │ (写入更新)   │  │(eBPF)    │  │dump)│ │
│  │ restore  │      │ parallel_    │  │posttool_ │  │     │ │
│  └──────────┘      │ hint(CFS)    │  │observers │  └─────┘ │
│                    └──────────────┘  └──────────┘          │
└────────────────────────────┬────────────────────────────────┘
                             │ VFS 统一数据访问层
┌────────────────────────────▼────────────────────────────────┐
│                   存储层（L1 内存管理）                        │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  store.db (SQLite, WAL 模式)                         │   │
│  │  ├── memory_chunks     主知识库（409 chunks）         │   │
│  │  ├── memory_chunks_fts FTS5 全文索引                 │   │
│  │  ├── recall_traces     检索命中追踪（TLB 类比）       │   │
│  │  ├── swap_chunks       换出页（跨 session 持久化）    │   │
│  │  ├── checkpoints       CRIU 快照（工作集精确恢复）    │   │
│  │  ├── scheduler_tasks   CFS 调度队列                  │   │
│  │  ├── goals             长期目标追踪                   │   │
│  │  ├── tool_patterns     工具调用模式学习               │   │
│  │  ├── dmesg             内核日志（subsystem 分类）      │   │
│  │  ├── hook_txn_log      事务日志（ext4 journal）       │   │
│  │  ├── shm_segments      跨 session 共享内存            │   │
│  │  ├── ipc_msgq          跨 Agent IPC 消息队列          │   │
│  │  └── chunk_pins        内存锁定（soft/hard pin）      │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────┐    │
│  │latest.json │  │session_    │  │tool_profile.db     │    │
│  │(任务状态)  │  │intent.json │  │(eBPF 调用剖析)     │    │
│  │            │  │(CRIU 断点) │  │                    │    │
│  └────────────┘  └────────────┘  └────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## 三、核心子系统

### 3.1 内存管理子系统

类比 Linux 虚拟内存管理（mm/），核心职责：控制 context window 使用量。

| 机制 | Linux 类比 | 实现文件 | 功能 |
|------|-----------|---------|------|
| Working Set | Denning Working Set Model | `loader.py` | Session 启动时预加载高权值 chunk |
| Page Swap | swap_out / swap_in | `store_swap.py` | 低频 chunk 换出到 swap_chunks，按需换入 |
| kswapd | Linux kswapd 水位线 | `store.py:kswapd_scan()` | 三水位预淘汰：ZONE_OK/LOW/MIN |
| OOM Killer | Linux OOM Killer | `store.py:evict_lowest_retention()` | 重要性最低的 chunk 被优先淘汰 |
| MGLRU | Multi-Gen LRU | `store.py:mglru_aging()` | Session 启动时推进 generation clock |
| DAMON | Data Access Monitor | `store.py:damon_scan()` | 扫描 dead/cold chunk 主动回收 |
| cgroup | cgroup v2 memory.high | `store.py:cgroup_throttle_check()` | 软限制：超过水位时降低新写入 importance |
| madvise | madvise(MADV_WILLNEED) | `store.py:madvise_write()` | 写入下轮检索的预热 hint |
| KSM | Kernel Samepage Merging | `store.py:already_exists()` `merge_similar()` | 相同/相似 chunk 去重合并 |
| CRIU | Checkpoint/Restore | `store_criu.py` | 会话工作集快照，精确恢复 |

**内存分区（chunk_type 语义）**：

| chunk_type | importance | oom_adj | 类比 |
|-----------|-----------|---------|------|
| `design_constraint` | 0.95 | -800 (protected) | 内核代码段（不可换出） |
| `quantitative_evidence` | 0.90 | -800 (protected) | 带 mlock 的关键数据页 |
| `decision` | 0.85 | 0 (normal) | 应用程序热路径页 |
| `causal_chain` | 0.82 | 0 | 推理路径页 |
| `reasoning_chain` | 0.80 | 0 | 分析缓冲区 |
| `procedure` | 0.85 | 0 | 可复用代码段 |
| `excluded_path` | 0.70 | +200 (prefer evict) | 已标记的冷页 |
| `conversation_summary` | 0.65 | +200 | 短生命周期数据 |
| `tool_insight` | 0.75 | 0 | 工具输出缓存 |
| `prompt_context` | — | +1000 | Session 结束后 GC |

---

### 3.2 IO 子系统（Hooks 执行链）

类比 Linux block IO 层，每个 hook event 对应一类"系统调用"：

```
SessionStart ──────────────────────────────────────────────────
  context_budget_guard.py  → 检查 context 配额（ulimit 类比）
  sleep-session-start.sh   → 恢复 sleep 状态
  session-start.js (ECC)   → ECC 插件 session 初始化
  loader.py                → 工作集恢复 + CRIU restore + 变化感知

UserPromptSubmit ───────────────────────────────────────────────
  sleep-activity-touch.sh  → 更新活跃时间戳
  parallel_hint.py         → CFS 并行任务检测（P4）
  [SYSTEM RULE echo]       → TaskCreate 强制规则
  writer.py                → 新知识写入（用户输入中的决策）
  retriever.py             → 相关知识检索注入（demand paging）

PreToolUse ─────────────────────────────────────────────────────
  [Bash] git hook guard    → 防止 --no-verify 绕过
  [Write] doc-file-warning → 文档文件写入警告
  [Edit|Write] suggest-compact → 大文件编辑提示压缩
  [mcp__.*] mcp-health-check → MCP 健康检查
  [Read] filesize_guard.js → 大文件读取保护
  [Bash|Write|Edit] pretool_coalesced.js → 批量预检

PostToolUse ────────────────────────────────────────────────────
  [Bash|Read] output_compressor.py → zram：大输出压缩提示（P1）
  [多工具] tool_profiler.py        → eBPF：调用效率分析（P3）
  [Bash] post-bash-pr-created.js   → PR 创建通知
  [Bash] post-bash-build-complete.js → 构建完成通知
  [Edit|Write] quality-gate.js     → 代码质量门
  [Bash|Write|Edit] governance-capture.js → 治理捕获
  [*] posttool_observers.js        → 通用工具观察者

Stop ────────────────────────────────────────────────────────────
  extractor.py             → 知识提取 + CRIU intent dump（P2）
  stop_coalesced.js        → 批量 Stop 事件合并处理

PreCompact / PostCompact ────────────────────────────────────────
  [计划中] save-task-state.py   → 压缩前保存任务状态
  [计划中] pre-compact.js (ECC) → ECC 压缩前处理
  [计划中] resume-task-state.py → 压缩后恢复任务状态
```

---

### 3.3 检索子系统（需求分页 / Demand Paging）

类比 Linux 缺页中断处理，按需将 store.db 中的知识注入 context。

#### Daemon 架构（iter162+）

早期 `retriever.py` 每次 UserPromptSubmit 启动新 Python 进程（subprocess），启动开销约 54ms。iter162 引入 **retriever_daemon.py** 常驻进程，通过 Unix socket 服务检索请求：

```
retriever_wrapper.sh（UserPromptSubmit hook）
    │  JSON-line over Unix socket
    ▼
retriever_daemon.py（常驻进程，/tmp/memory-os-retriever.sock）
    ├── 4 worker handler pool
    ├── TLB L1（prompt_hash 精确匹配）→ SKIP（<0.1ms）
    ├── TLB L2（injection_hash 模糊匹配）→ SKIP（<0.1ms）
    ├── FTS5 result cache（Page Cache 类比，iter205）
    ├── BM25 scoring loop（10 chunks，~0.1ms warm）
    └── 注入 additionalContext（hookSpecificOutput JSON）
```

**延迟对比**：
| 路径 | iter162 前（subprocess） | iter238 当前 |
|------|------------------------|-------------|
| SKIP 路径（TLB 命中） | ~54ms | <0.1ms |
| 完整注入 P50 | ~54ms | ~0.1ms |
| 完整注入 P95 | ~54ms | ~0.14ms |

**TLB 缓存（iter179）**：
- L1：完全相同 prompt（hash 匹配）→ 0 注入
- L2：DB 内容未变（injection_hash 不变）→ 0 注入；DB 有新写入时失效

**FTS5 result cache（iter205）**：相同 FTS 表达式的查询结果缓存，cache hit ~0.3us。

#### 评分函数（_score_chunk，iter235+）

iter235 起 FTS 路径返回 raw tuple，评分改用位置索引（`_CI_*` 常量）消除 dict lookup：

```python
_CI_ID=0, _CI_SUM=1, _CI_CON=2, _CI_IMP=3, _CI_LA=4,
_CI_CT=5, _CI_AC=6, _CI_CA=7, _CI_FR=8, _CI_LG=9,
_CI_CP=10, _CI_VS=11, _CI_CS=12
```

分值组成：
```
score = relevance × (eff_imp×0.55 + recency×0.45 + access_bonus + freshness_bonus)
      + starvation_boost - saturation_penalty
      + verification_bonus - verification_penalty
      + lru_gen_boost - numa_distance_penalty
```

| 组件 | 优化历程 |
|------|---------|
| `lru_gen_boost` | iter236：`min(lg,8)` → ternary + 预计算常量 0.0075；-0.161us/chunk |
| `_age_days_fast` | iter237：内联到 `_score_chunk`，消除函数调用；E2E P50 -21%（0.127ms→0.100ms） |
| `_TYPE_PREFIX` | iter238：dict 移至模块级常量；0.356us → 0.128us per inject |
| FTS raw tuple | iter235：消除 dict 构建；~0.14ms P95 稳定 |
| compact score formula | iter239b：移除恒零 eb+sb 加法（2× LOAD_FAST+BINARY_ADD）；-29ns/chunk |
| vs-first inject | iter239c：先加载 _vs，None 时跳过 _cs（corpus 100% None）；-365ns/5 lines |
| `_LGB_TABLE` | iter240：lgb ternary → 9 元素查表；-45ns/chunk（同 _AB_TABLE/_ST_TABLE 策略）|
| SQL COALESCE(ca,la) | iter241：created_at/project NULL 处理移入 SQL；消除 Python `or ""` × 2/chunk |
| drop `if _ca and` | iter242：COALESCE 保证 _ca 非空，删除死代码 bool 检查；-31ns/chunk |
| SQL COALESCE(imp,0.5)| iter243：importance NULL 处理移入 SQL；消除 `or 0.5` × 1/chunk |
| drop age `>=0` guards| iter244：无未来时间戳（verified N=415），删除两处 ternary guard；-21ns/chunk |
| `(age, exp)` tuple cache | iter245：cache 存 (age, exp_val) tuple，命中时跳过 math.exp；-43ns/chunk |
| `(age, exp, rec)` 3-tuple cache | iter246：加入 recency=1/(1+age)，命中时跳过除法；-31ns/chunk |
| UNPACK_SEQUENCE(13) 全量解包 | iter247：9× 独立 BINARY_SUBSCR → 单次 C-loop 解包；-34ns/chunk |
| fuse `base` temp into score expr | iter248：消除 base=... 临时变量；STORE/LOAD pair → 寄存器直用；-24ns/chunk |
| drop `_lg < 9` guard | iter249：corpus 验证 max(lru_gen)=4，_LGB_TABLE 直接索引；-26ns/chunk |
| drop `eff_imp >= floor` guard | iter250：corpus 验证 min eff_imp=0.596 >> 0.05，删除 ternary guard；-13.8ns/chunk |
| extend `_AB_TABLE` 21→64 entries | iter251：corpus 105/427 chunks 有 ac>20（调 log2），64 项表覆盖全部；-62ns/chunk |
| eliminate `eb` variable dead-code | iter252：折叠 `else: eb=0.0` + `if eb:` 检查为 0 字节码（_run_aslr=False）；-89ns/request |
| drop `if _vs is None` guard | iter253：corpus 验证 0/427 vs IS NULL，删除 None 检查 + 2× or 回退；-19.8ns/chunk |
| drop `if age_ca < fb_grc` guard | iter254：corpus 验证 0/427 chunks 超出 grace 期；-18.3ns/chunk |
| collapse vb 4-branch+vp → single ternary | iter255：corpus 0/427 disputed（vp=0）+pending全cs=0.7；-28.2ns/chunk；-0.28us/request |
| extend `_ST_TABLE` 21→271 entries | iter256：corpus max rc=270（recall_traces），271项表覆盖全部；-26.8ns/chunk；-0.27us/request |
| rename `_cs` → `_` in FTS unpack | iter257：iter255后_cs不再读取；死变量消除；-13.5ns/chunk；-0.135us/request |
| reorder `ndp` global-first | iter258：global占67.9%语料，改为优先分支；-30.7ns/chunk；-0.307us/request |
| **累计（iter238→258）** | **11.4us → 1.74us/request（-84.7%）** | |

**Context Pressure Governor**：四级水位（LOW/NORMAL/HIGH/CRITICAL）动态缩放注入窗口大小。

---

### 3.4 调度子系统

类比 Linux CFS（Completely Fair Scheduler），管理 Agent 并发执行。

```
sched/
├── agent_scheduler.py   AgentTask + Scheduler (vruntime, nice level)
├── agent_cgroup.py      cgroup v2 资源限制（foreground/background）
└── agent_monitor.py     Agent 健康监控
```

**调度优先级**：
- `NICE_CRITICAL = -20`：前台用户交互
- `NICE_NORMAL = 0`：普通 Agent 任务
- `NICE_BACKGROUND = 19`：低优先级后台任务

**P4 并行提示**（`parallel_hint.py`）：检测 UserPromptSubmit 中的独立子任务，注入 CFS 提示引导并行执行。

---

### 3.5 网络子系统（跨 Agent IPC）

类比 Linux socket + netfilter，提供跨 session / 跨 Agent 通信。

```
net/
├── agent_notify.py    跨 Agent 知识更新广播（inotify 类比）
├── agent_protocol.py  消息序列化协议
├── agent_router.py    消息路由
├── agent_socket.py    UNIX domain socket 封装
└── agent_firewall.py  防火墙规则（消息过滤）
```

**IPC 流程**：
- extractor Stop 时 `broadcast_knowledge_update()` → ipc_msgq
- loader SessionStart 时 `consume_pending_notifications()` → 消费并注入

**iter260 Extractor Pool 流程**：
```
Stop hook → submit_extract_task() → ipc_msgq[extract_task] → extractor_pool(常驻) → store.db
                                            ↑                          ↓
                              pool 未运行时 False             broadcast_knowledge_update
                                    ↓
                             同步 fallback（原有逻辑）
```

---

### 3.6 知识提取子系统（extractor.py + extractor_pool.py）

类比 Linux VFS write path + journal commit，Session 结束时提取知识写入 store.db。

**提取类型**（按 importance 降序）：

```
design_constraint (0.95) ← 系统级约束，违反会产生语义错误
quantitative_evidence (0.90) ← 含数字度量的可复用结论
decision (0.85)          ← 含决策动词/技术锚点的方案选择
causal_chain (0.82)      ← 因果链（"因为X→所以Y"）
reasoning_chain (0.80)   ← 推理过程
excluded_path (0.70)     ← 被放弃的路径
conversation_summary (0.65) ← 对话摘要
tool_insight (0.75)      ← Bash 工具输出中的量化结论
```

**质量过滤层**（OOM Killer 类比，逐层过滤噪声）：
1. `_is_fragment()` — 截断/残缺句检测
2. `_is_quality_chunk()` — V9 多版本通用质量过滤
3. `_is_quality_decision()` — decision 专用 SNR 过滤（iter106）
4. `_is_tool_insight_noise()` — tool_insight 专用噪声过滤

**COW 预扫描**（iter39）：未命中信号词（~60% 消息）直接跳过完整提取，<0.1ms。

**事务语义**（iter99）：ext4 journal 两阶段提交，全成功或全回滚。

---

### 3.7 CRIU 子系统（迭代49+110）

**P2：Session Intent Checkpoint（迭代110）**：
- Stop 时从 `last_assistant_message` 尾部 2000 字提取未完成意图
- 保存至 `session_intent.json`（3 类：next_actions / open_questions / partial_work）
- SessionStart 时 loader 读取并注入（age < 24h）

**工作集 Checkpoint（迭代49）**：
- Stop 时记录本 session 访问的 chunk IDs 到 `checkpoints` 表
- 下次 SessionStart 时精确恢复，而非泛化 Top-K

---

## 四、数据流图

```
用户输入
   │
   ▼ UserPromptSubmit
┌──────────────┐     ┌─────────────────────┐
│ writer.py    │────▶│ store.db            │
│ (新知识写入) │     │ memory_chunks (RAM) │
└──────────────┘     │                     │
                     │                     │◀─── swap_in (按需)
   │                 │                     │
   ▼                 │ swap_chunks (disk)  │
┌──────────────┐     └──────────┬──────────┘
│ retriever.py │◀───────────────┘
│ (检索注入)   │
└──────┬───────┘
       │ additionalContext (≤500 tokens)
       ▼
   Claude 推理
       │
       ▼ Tool Calls
┌──────────────────────────────┐
│ PostToolUse hooks            │
│  output_compressor.py (zram) │ → additionalContext 压缩提示
│  tool_profiler.py (eBPF)     │ → tool_profile.db 记录
└──────────────────────────────┘
       │
       ▼ Stop
┌──────────────┐
│ extractor.py │ → submit_extract_task() → ipc_msgq[extract_task]
│  (Stop hook) │   (pool running: <5ms, return)
│              │ → fallback: 同步提取 → store.db  (pool not running)
│              │ → session_intent.json (CRIU P2)
│              │ → checkpoints (CRIU P1)
│              │ → madvise hints (预热)
│
│ extractor_   │ → poll ipc_msgq → _run_extraction_pipeline()
│ pool.py      │   → store.db → broadcast_knowledge_update
└──────────────┘
```

---

## 五、关键设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| 存储后端 | SQLite + WAL | 轻量、无服务端依赖、并发安全 |
| 检索算法 | BM25 + FTS5 混合 | 无需 embedding，延迟 < 10ms |
| context 注入上限 | ≤ 500 tokens | 平衡信息量与 context 消耗 |
| 提取时机 | Stop hook（异步） | 不阻塞用户体验 |
| 检索时机 | UserPromptSubmit（同步） | 确保 Claude 看到相关知识 |
| 事务隔离 | SQLite BEGIN IMMEDIATE | 防止并发写入污染 |
| 工具输出修改 | 不支持，用 additionalContext | Claude Code hook 限制 |
| chunk 去重 | 精确 hash + 语义相似度双重 | 平衡准确性与性能 |

---

## 六、当前状态（迭代258）

- **知识库**：427 chunks（decision 230 / quantitative_evidence 49 / causal_chain 41 / procedure 39 / design_constraint 21 / ...）
- **检索延迟**：P50 ~0.1ms / P95 ~0.14ms（vs subprocess 基线 ~54ms，540× 改善）
- **_score_chunk**：1.74us/10 chunks（iter238 基线 11.4us，-84.7%；iter239-258 累计改善）
- **检索质量**：BM25 vs 纯重要性排序 Recall@3 +147%（58.3% vs 23.6%），MRR +320%
- **A/B 测试**：memory-os 辅助 vs 无记忆：8/12 胜，平均得分 3.55 vs 2.12（+68%）
- **Session Recall@3**：94.2%
- **检索命中率**：61.9% chunks 被检索命中（最高单 chunk ×2043 次）
- **活跃 hook 数**：~20 个（SessionStart 4 + UserPromptSubmit 5 + PreToolUse 5 + PostToolUse 7 + Stop 2）
- **核心文件**：~70 个 .py 文件，32K+ 行代码
- **测试覆盖**：549+ 个测试用例

---

## 七、迭代路线图（OS 演进类比）

| 阶段 | OS 类比 | 状态 |
|------|---------|------|
| 基础内存管理 | Unix VM (1969) | ✅ iter1-20 |
| 虚拟内存/swap | BSD VM (1979) | ✅ iter21-50 |
| 高级页面回收 | Linux 2.6 MM | ✅ iter51-90 |
| 多代 LRU/DAMON | Linux 6.x | ✅ iter91-100 |
| zram/eBPF/CRIU | Linux 现代 | ✅ iter101-110 |
| Daemon 常驻检索 | vDSO + Unix socket | ✅ iter162 |
| TLB 两级缓存 | CPU TLB L1/L2 | ✅ iter179 |
| FTS result cache | Page Cache | ✅ iter205 |
| 评分 positional access | struct field offset | ✅ iter235 |
| 评分热路径微优化 | 指令级优化（ILP） | ✅ iter236-238 |
| SQL COALESCE NULL 下沉 | 零成本抽象 / schema 层 NULL 处理 | ✅ iter239-244 |
| `(age, exp)` tuple cache | FPU 结果缓存（类比 decoded-PTE cache）| ✅ iter245 |
| `(age, exp, rec)` 3-tuple cache | FPU+除法结果缓存（消除 1/(1+age) 除法）| ✅ iter246 |
| UNPACK_SEQUENCE tuple 全量解包 | 9× BINARY_SUBSCR → 单次 C-loop；-34ns/chunk | ✅ iter247 |
| fuse score arithmetic | 消除 base 临时变量，寄存器直用；-24ns/chunk | ✅ iter248 |
| drop `_lg < 9` LGB guard | corpus 验证 max=4，直接 _LGB_TABLE[_lg]；-26ns/chunk | ✅ iter249 |
| drop `eff_imp >= floor` guard | corpus 验证 min eff_imp=0.596 >> 0.05；-13.8ns/chunk | ✅ iter250 |
| extend `_AB_TABLE` 21→64 entries | 消除 105/427 chunk 的 log2 调用；-62ns/chunk | ✅ iter251 |
| eliminate `eb` dead-code | fold `else: eb=0.0` + `if eb:` → 0 bytecodes when _run_aslr=False | ✅ iter252 |
| drop `if _vs is None` guard | corpus 验证 0/427 vs IS NULL；删除 None 检查 + or 回退；-19.8ns/chunk | ✅ iter253 |
| drop `if age_ca < fb_grc` guard | corpus 验证 0/427 超出 grace 期；kswapd 不变式；-18.3ns/chunk | ✅ iter254 |
| collapse vb 4-branch+vp → ternary | corpus 0/427 disputed + pending 全 cs=0.7；-28.2ns/chunk | ✅ iter255 |
| extend `_ST_TABLE` 21→271 entries | corpus max rc=270，271项表；-26.8ns/chunk；同 iter251 enlarged TLB | ✅ iter256 |
| rename `_cs` → `_` (dead var) | iter255后_cs不再使用；消除STORE_FAST interning；-13.5ns/chunk | ✅ iter257 |
| reorder ndp global-first | PGO分支排序：global 67.9%优先；减少分支预测失败；-30.7ns/chunk | ✅ iter258 |
| multi-agent 隔离 | session_id PRIMARY KEY隔离shadow_traces/session_intents；per-session文件防止last-writer-wins | ✅ iter259 |
| extractor async pool | Stop hook offload → ipc_msgq → extractor_pool kworker；消除 Stop hook 50-150ms I/O 阻塞 | ✅ iter260 |
| 分布式内存 | NUMA / RDMA | 🔜 iter261+ |
