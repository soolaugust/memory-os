# AIOS Memory OS

**用操作系统内核的设计哲学，解决 AI Agent 的认知资源管理问题。**

---

## 是什么？（一句话）

让 AI 拥有持久记忆——像操作系统管理计算机一样，帮 AI 管理它的知识与记忆。

---

## 为什么需要它？

使用 AI 助手时，你可能遇到过这三个问题：

| 问题 | 现象 |
|------|------|
| **每次对话结束就彻底遗忘** | 你今天教给 AI 的所有东西，明天对话时它全不记得了，每次都要重新解释背景 |
| **对话窗口塞不下所有信息** | AI 每次能看到的内容有限，就像一张桌子大小固定，放不下所有需要的资料 |
| **多个 AI 之间无法共享知识** | 让多个 AI 协作时，它们各自为战，学到的东西无法互通，每个都要单独培训 |

核心原因：**AI 本质上是「无状态」的，它天生缺乏跨时间的记忆能力。**

---

## 解题思路：向操作系统学习

计算机早已解决了同样的问题——操作系统用内存、硬盘、调度器管理有限资源，Memory OS 把这套哲学搬到 AI 认知管理上：

| 操作系统概念 | Memory OS 对应 |
|-------------|---------------|
| 内存（运行时工作空间） | 对话窗口——AI 当前能看到的内容 |
| 硬盘（长期存储） | 知识库——跨会话保存的知识与记忆 |
| 缺页自动加载 | 智能检索注入——问到什么就调取相关记忆 |
| 任务调度 | 多助手协调——多个 AI 共享知识协作 |
| 进度存档恢复 | 会话快照——保存工作状态，下次无缝接续 |

---

## 工作流程

```
你说话
  → 系统自动调取相关记忆注入对话
  → AI 基于完整上下文回答
  → 对话结束，自动提取新知识保存到知识库
```

整个过程全自动，无需手动管理任何记忆。

---

## 知识管理

系统像图书馆一样管理知识：分类、保鲜、去重、淘汰。

| 机制 | 类比 | 作用 |
|------|------|------|
| 重要知识优先保留 | 珍贵文献特殊保管 | 设计决策、关键数据永久保留，闲置信息及时清理 |
| 冷热淘汰 | 书架定期清理 | 长期未用的知识自动移出，为新知识腾出空间 |
| 紧急腾空 | 仓库满了清理最不重要的货 | 空间耗尽时自动删除最不重要的旧知识 |
| 多代分层管理 | 图书馆新旧书分区 | 按访问频率分层，热门知识随时可取 |
| 会话快照存档 | 存档游戏进度 | 对话结束自动保存工作状态，下次无缝续接 |

知识按重要程度分级：核心决策（永不淘汰）→ 量化结论 → 操作方案 → 背景说明 → 临时记录（用后即清）。

---

## 检索速度

从「去仓库翻箱子」到「桌上随手就拿到」：

| | 以前 | 现在 |
|--|------|------|
| 方式 | 每次临时重新打开"记忆仓库"，加载索引、建立连接 | 专属检索助手时刻待命，两级缓存随时调取 |
| 延迟 P50 | **54 毫秒** | **0.1 毫秒** |
| 提升 | — | **快了 540 倍** |

---

## 核心成果

| 指标 | 数值 |
|------|------|
| 检索速度提升 | 540×（54ms → 0.1ms） |
| 记忆命中率提升 | +147%（vs 无记忆管理） |
| 首条命中精准度 | +320% MRR |
| 实际对比得分 | +68%（用户满意度对照测试） |
| 跨会话记忆保留 | 94% |
| 沉淀知识条数 | 427 条（258 轮迭代累积） |
| 检索热路径性能 | 1.74 μs/次（迭代 258，较基线 -84.7%） |

---

## 路线图

- ✅ **基础记忆管理**（iter 1–100）：知识保存、冷热淘汰、多优先级分级保护
- ✅ **常驻检索助手**（iter 162–205）：两级缓存、常用结果预存、极速调取
- ✅ **检索精度持续打磨**（iter 235–258）：数据驱动优化、258 轮迭代、提速 84.7%
- 🔜 **分布式多助手共享记忆**（iter 259+）：多个 AI 共享同一知识图谱，协作无边界

---

## 痛点 → 解法

### 痛点 1：Claude 没有跨 session 记忆，每次从零开始

每次新开对话，之前的决策、踩过的坑、架构约束全部丢失。需要大量"热身"时间重建上下文，重复犯同样的错误。

**解法：知识持久化 + 自动检索注入**

- Session 结束时，自动从对话中提取决策/推理链/设计约束/量化结论，写入 `store.db`
- 新 session 开始时，检索相关知识注入 context（`additionalContext`），无需用户手动粘贴

**实测效果：**
```
知识库现有：409 chunks（decision 230 / quantitative_evidence 49 / causal_chain 41 / procedure 39 / ...）
BM25 检索 vs 纯重要性排序：Recall@3 +147%（58.3% vs 23.6%），MRR +320%
A/B 测试：memory-os 辅助 vs 无记忆：8/12 胜，平均得分 3.55 vs 2.12（+68%）
Session Recall@3：94.2%
检索命中率：61.9% chunks 实际被检索命中（最高单 chunk ×2043 次）
```

---

### 痛点 2：检索延迟高，每次 prompt 都要等待

早期基于 subprocess 的检索 P50 约 54ms，每次用户输入都有明显卡顿。

**解法：Daemon 架构 + 三级缓存**

- **retriever_daemon.py**：常驻进程，通过 Unix socket 服务检索请求，消除 Python import 启动成本
- **FTS5 result cache**：相同 FTS 表达式结果缓存，cache hit ~0.3us
- **TLB 两级缓存**：L1（prompt_hash 精确匹配）+ L2（injection_hash 模糊匹配），命中直接返回空响应

**实测延迟（iter238，2026-04）：**
```
SKIP 路径（确认词/无关 prompt）：<0.1ms（TLB 命中，无注入）
完整检索+注入 P50：             ~0.1ms
完整检索+注入 P95：             ~0.14ms
历史对比：subprocess 时代 P50 ~54ms → daemon 后 ~0.1ms（降低 540×）
```

---

### 痛点 3：Context window 频繁撑满，触发强制压缩

长任务中 Claude 的 context 会被大量工具输出占满，触发 compaction，丢失关键推理链。

**解法：多层 context 压缩与换出**

- **zram（output_compressor.py）**：大输出自动注入压缩提示，引导 Claude 优先关注结果区和错误行
  - Bash 输出 >3KB / Read 输出 >4KB 触发
  - 不截断内容，通过 `additionalContext` 导引注意力分配
- **Context Pressure Governor**：四级水位（LOW/NORMAL/HIGH/CRITICAL）动态缩放注入窗口
- **Swap 换出**：低频知识自动换出到 `swap_chunks`，按需换入，降低 context 占用

---

### 痛点 4：Session 中断后丢失"正在做的事"

Context compaction 或 session 超时后，Claude 不知道上次做到哪里，需要用户重新描述。

**解法：CRIU 断点恢复**

- Stop 时从最后一条 assistant 消息提取未完成意图（next_actions / open_questions）
- 保存到 `session_intent.json`，下次 SessionStart 自动注入断点恢复上下文
- 有效期 24h，过期自动失效

---

### 痛点 5：架构约束散落在历史对话中，容易被遗忘违反

"这里不能用 mock"、"必须先调用 ensure_schema"——这类约束不在代码里，只在某次 review 的对话里，很容易被新 session 的 Claude 违反。

**解法：设计约束专项提取与高保护存储**

- 自动识别约束句式（22 种模式：`不能/禁止/注意/⚠️/WARNING:/否则会...`）
- `design_constraint` 类型：`importance=0.95`，`oom_adj=-800`（最高保护，不被淘汰）
- 每次 UserPromptSubmit 自动检索并注入，相关约束优先出现在 context 头部

**实测：**
```
当前知识库：21 个设计约束，最高命中 ×2043 次（检索管道设计约束）
```

---

### 痛点 6：重复读同一个文件，重复执行同一条命令

Claude 在复杂任务中常重复调用相同工具，浪费 context token。

**解法：工具调用效率分析（tool_profiler.py）**

- PostToolUse 后记录每次调用（tool_name / output_bytes / duration_ms）
- 检测低效模式：同文件 Read ≥3 次 / 同命令 Bash ≥2 次
- 检测到时注入提示建议缓存或精确查询

---

### 痛点 7：面对多个独立子任务仍然串行执行

**解法：CFS 并行任务检测（parallel_hint.py）**

- UserPromptSubmit 时检测并行信号：显式并行词 / 列表型任务（3+ 项）/ 对比分析
- 检测到时注入提示：`[CFS] 检测到 N 个独立子任务，可用 Agent tool 并行执行`

---

## 架构总览

```
Claude Code
    ↕ hooks (syscall 门)
┌──────────────────────────────────────┐
│  hooks/                              │
│  ├── loader.py        (SessionStart) │  Working Set 恢复 + CRIU restore
│  ├── retriever_wrapper.sh (UserPmt)  │  → retriever_daemon.py (常驻进程)
│  ├── writer.py        (UserPrompt)   │  知识写入
│  ├── extractor.py     (Stop)         │  知识提取 + CRIU dump
│  ├── output_compressor.py (Post)     │  zram 大输出压缩提示
│  ├── tool_profiler.py (Post)         │  工具调用效率分析
│  └── parallel_hint.py (UserPmt)      │  CFS 并行任务提示
└──────────────────────────────────────┘
    ↕ VFS 统一数据层
┌──────────────────────────────────────┐
│  ~/.claude/memory-os/store.db        │
│  memory_chunks / swap_chunks         │
│  checkpoints / dmesg                 │
│  FTS5 全文索引（bigram CJK）          │
└──────────────────────────────────────┘
```

详细架构说明见 [ARCHITECTURE.md](./ARCHITECTURE.md)。

---

## 核心指标

| 指标 | 数值 |
|------|------|
| 检索延迟 P50（TLB 命中） | ~0.1ms |
| 检索延迟 P95（完整注入） | ~0.14ms |
| vs subprocess 基线改善 | 540× 更快 |
| 检索召回率（BM25 vs baseline） | +147% Recall@3 |
| 排名质量（MRR） | +320% |
| A/B 回答质量提升 | +68%（3.55 vs 2.12 分） |
| Session Recall@3 | 94.2% |
| 知识库规模 | 409 chunks / 8 类型 |
| 测试覆盖 | 549+ tests |

---

## 设计哲学

每一个功能都有对应的 Linux 内核机制类比：

| 功能 | Linux 类比 | 迭代 |
|------|-----------|------|
| 知识检索注入 | Demand Paging（缺页中断） | iter1 |
| 工作集预加载 | Denning Working Set Model | iter18 |
| 知识淘汰 | kswapd + OOM Killer | iter25,38 |
| 跨 session 恢复 | CRIU Checkpoint/Restore | iter49 |
| 参数自优化 | TCP AIMD + Auto-tuning | iter50,51 |
| 多代 LRU | MGLRU（Linux 6.x） | iter44 |
| 访问模式监控 | DAMON | iter42 |
| 工具输出压缩 | zram | iter110 |
| 并行任务提示 | CFS Work-Stealing | iter110 |
| Daemon 常驻检索 | vDSO + Unix socket | iter162 |
| TLB 两级缓存 | CPU TLB L1/L2 | iter179 |
| FTS 结果缓存 | Page Cache | iter205 |
| 评分 positional access | struct field offset | iter235 |

---

## 快速部署

### 前置条件

- Python 3.12+
- SQLite（内置，无需安装）
- `nc`（netcat）、`flock`（通常已预装）
- Claude Code CLI

### 安装步骤

```bash
# 1. 克隆仓库
git clone <repo-url> ~/codes/aios/memory-os
cd ~/codes/aios/memory-os

# 2. 初始化数据目录（首次运行时 store.py 自动创建）
mkdir -p ~/.claude/memory-os

# 3. 在 ~/.claude/settings.json 的 hooks 字段中添加以下配置
# （将 /path/to/memory-os 替换为实际路径）
```

**hooks 配置（settings.json）**：

```json
{
  "hooks": {
    "SessionStart": [
      {"type": "command", "command": "python3 /path/to/memory-os/hooks/loader.py", "timeout": 10}
    ],
    "UserPromptSubmit": [
      {"type": "command", "command": "bash /path/to/memory-os/hooks/retriever_wrapper.sh", "timeout": 10, "async": false},
      {"type": "command", "command": "python3 /path/to/memory-os/hooks/writer.py", "timeout": 10, "async": false},
      {"type": "command", "command": "python3 /path/to/memory-os/hooks/parallel_hint.py", "timeout": 3, "async": false}
    ],
    "PostToolUse": [
      {"matcher": "Bash|Read", "hooks": [{"type": "command", "command": "python3 /path/to/memory-os/hooks/output_compressor.py", "timeout": 5}]},
      {"matcher": "*", "hooks": [{"type": "command", "command": "python3 /path/to/memory-os/hooks/tool_profiler.py", "timeout": 5, "async": true}]}
    ],
    "Stop": [
      {"type": "command", "command": "python3 /path/to/memory-os/hooks/extractor.py", "timeout": 10, "async": true}
    ]
  }
}
```

### 验证安装

```bash
# 验证 loader（SessionStart）
echo '{"session_id":"test","transcript_path":"/dev/null","cwd":"'$(pwd)'"}' \
  | python3 hooks/loader.py
# 预期：输出 {"hookSpecificOutput": ...} 或空（无历史时）

# 验证 retriever daemon 启动
echo '{"session_id":"test","prompt":"test query","cwd":"'$(pwd)'"}' \
  | bash hooks/retriever_wrapper.sh
# 预期：daemon 自动启动，返回 {} 或注入内容，<3s 内完成

# 验证 daemon 已在运行
ls /tmp/memory-os-retriever.sock && echo "daemon running"

# 运行测试套件
python3 -m pytest -q --tb=short
```

### daemon 管理

retriever daemon 在首次请求时自动启动，无需手动管理。

```bash
# 查看 daemon 日志
tail -f ~/.claude/memory-os/daemon.log

# 手动重启 daemon（更新代码后）
pkill -f retriever_daemon.py

# daemon 会在下次 retriever_wrapper.sh 调用时自动重新启动
```

---

## 依赖

Python 3.12+，SQLite（内置），`nc`，`flock`。无需 GPU，无需外部 API。
