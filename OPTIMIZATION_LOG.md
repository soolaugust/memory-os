# memory-os 优化日志

## 2026-04-30 第一轮迭代（token效率/信噪比/OS架构/人类记忆/混乱治理）

### 已完成优化

| # | 改动 | 文件 | 效益 |
|---|------|------|------|
| 1 | 去除 UserPromptSubmit 噪声 tokens | writer_retriever_merged.py | 每轮 -25 tokens |
| 2 | extractor_pool min_length 语义修正 | extractor_pool.py | 避免错误 fallback |
| 3 | _db_vacuum 自适应节流（6h/24h/72h） | context_budget_guard.py | 减少无谓 vacuum |
| 4 | 维护函数合并节流文件 | context_budget_guard.py | -2次 SessionStart I/O |
| 5 | priming_state 改 LFU 淘汰 | context_budget_guard.py | 保护热实体不被驱逐 |

## 2026-04-30 第二轮迭代

| # | 改动 | 文件/配置 | 效益 |
|---|------|------|------|
| 6 | 删除 SYSTEM RULE echo hook | settings.json | 每轮 -60 tokens |
| 7 | thrashing_detector SQLite 懒开 | thrashing_detector.py | 低压力时跳过 DB |
| 8 | thrashing_detector matcher 精确化 | settings.json | MCP/Agent 调用不触发 |
| 9 | tool_profiler synchronous=OFF | tool_profiler.py | 302ms → 54ms（-82%） |
| 10 | posttool_guard 合并两同步 hook | posttool_guard.py + settings.json | 76ms → 43ms（-44%） |
| 11 | writer.py 改 async=true | settings.json | 用户不再等待 111ms/轮 |

## 2026-04-30 第三轮迭代

| # | 改动 | 文件 | 效益 |
|---|------|------|------|
| 12 | session_episodes freshness gate（<24h全文/24h-7d简短/7d+跳过） | store_episodes.py | 减少过期 session 历史噪音注入 |
| 13 | 跨Agent同步信噪比过滤（仅注入 decisions/constraints，跳过纯chunk计数） | hooks/loader.py | 消除无召回价值的 "+12个chunk" 噪音 |
| 14 | CRIU restore 应用 WORKING_SET_TYPES 过滤（移除 causal_chain 注入） | hooks/loader.py | 不再注入已修复的 bug 记录 |
| 15 | _db_vacuum 加入 session_episodes 截断（保留最新100条） | hooks/context_budget_guard.py | 防止 session 历史无限积累 |

## 性能基准（优化后）

| 场景 | 优化前 | 优化后 |
|------|--------|--------|
| UserPromptSubmit 阻塞延迟 | ~150ms (writer 111ms + retriever) | ~2ms (retriever daemon socket) |
| PostToolUse(Bash/Read) 阻塞 | ~76ms (compressor+thrashing) | ~43ms (posttool_guard) |
| tool_profiler (async) | 302ms 后台 | 54ms 后台 |
| SYSTEM RULE 注入 | 每轮 +60 tokens | 0 tokens |
| SessionStart vacuum | 固定 24h | 自适应 6h/24h/72h |
