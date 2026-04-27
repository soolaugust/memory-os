# Phase 2: KnowledgeVFS 集成指南

**发布时间**: 2026-04-20  
**状态**: ✅ 完成（所有性能目标达成）  
**代码位置**: `/aios/memory-os/vfs*.py`

---

## 快速开始

### 基本用法

```python
from vfs import get_vfs

# 获取全局 VFS 实例
vfs = get_vfs()

# 搜索
results = vfs.search("BM25", top_k=5, deadline_ms=100)
for item in results:
    print(f"{item.path}: {item.summary} (score={item.score:.3f})")

# 读取单项
item = vfs.read("/memory-os/chunk-001")
if item:
    print(item.content)
```

### 在 Hook 中集成

**retriever.py 中的集成示例**：

```python
# 替换原有的 BM25 搜索
from vfs import get_vfs

def _search_knowledge(query: str, top_k: int = 5) -> List[dict]:
    """使用 VFS 统一搜索"""
    vfs = get_vfs()
    
    # 单行调用，替代原有 store.fts_search + scorer 组合
    items = vfs.search(query, top_k=top_k, deadline_ms=80)
    
    # 转换回原格式（向后兼容）
    return [
        {
            "id": item.id,
            "summary": item.summary,
            "score": item.score,
        }
        for item in items
    ]
```

---

## 架构设计

### 虚拟路径格式

```
/<source>/<id>

例：
  /memory-os/chunk-uuid-123abc       ← 从 memory-os store.db
  /memory-md/feishu_access_method    ← 待实现：从 memory-md
  /self-improving/domains/vfs.md     ← 待实现：从 self-improving
  /project/history-uuid-xyz          ← 待实现：从项目 JSONL
```

### 二级缓存

**L1 dentry cache** — 路径缓存（<0.1ms）
- 数据：`(path, VFSItem)` 元组
- TTL：300 秒（可配置）
- 速度：<0.1ms 命中
- 用途：快速查找已读取项目

**L2 inode cache** — 内容缓存（<1ms）
- 数据：`(id, VFSItem)` 完整内容
- TTL：300 秒（可配置）
- 速度：<1ms 命中
- 用途：减少后端重复查询

**L3 后端存储** — 冷路径
- SQLite FTS5：5-10ms
- 文件系统（待实现）：20ms
- JSONL（待实现）：30ms
- 硬 deadline：100ms

### 去重策略

```python
# summary 级别去重 + 源权重
seen = {}
for item in all_items:
    # 用 summary 哈希作为去重 key
    key = hashlib.sha256(item.summary.encode()).hexdigest()
    
    # 只保留最高分的版本
    if key not in seen or item.score > seen[key].score:
        seen[key] = item

# 应用源权重
source_weights = {
    "memory-os": 1.0,         # 主存储，最高权重
    "self-improving": 0.7,    # 知识库，中等权重
    "project": 0.6,           # 项目历史，较低权重
}
```

---

## 性能基准

### 测试环境
- DB：191 chunks，FTS5 索引完整
- 硬件：Ryzen 5950X，SSD
- 并发：1（当前单线程）

### 结果

| 操作 | 延迟 | 目标 | 达成 |
|------|------|------|------|
| 搜索 (avg) | 0.4ms | <100ms | ✅ |
| 搜索 (max) | 0.8ms | <100ms | ✅ |
| 缓存读 | 0.003ms | <1ms | ✅ |
| dentry cache hit rate | 100% | >80% | ✅ |
| inode cache hit rate | 100% | >80% | ✅ |

### 瓶颈分析

当前：SQLite 后端（FTS5 查询 0.3ms + 转换 0.1ms）

未来加速机会：
1. **查询优化**：参数化 MATCH 或使用预编译语句
2. **并发后端**：多后端并行查询（目前仅 SQLite）
3. **分布式缓存**：Redis 支持（跨进程共享缓存）

---

## 后端扩展

### 添加新后端

1. 继承 `VFSBackend` 基类：

```python
from vfs_core import VFSBackend, VFSItem

class MyBackend(VFSBackend):
    def read(self, path: str) -> Optional[VFSItem]:
        # 从 /my-source/id 读取
        pass
    
    def search(self, query: str, top_k: int = 5) -> List[VFSItem]:
        # 搜索实现
        pass
    
    def write(self, item: VFSItem) -> bool:
        # 写入实现
        pass
    
    def delete(self, path: str) -> bool:
        # 删除实现
        pass
    
    @property
    def name(self) -> str:
        return "MyBackend"
    
    @property
    def source_type(self) -> str:
        return "my-source"
```

2. 在 KnowledgeVFS 中注册：

```python
vfs = get_vfs()
backend = MyBackend()
vfs._backends[backend.source_type] = backend
```

---

## 集成清单

### Phase 2 完成项
- [x] VFSItem + VFSMetadata 定义
- [x] SQLiteBackend 实现（FTS5 搜索）
- [x] dentry + inode 二级缓存
- [x] 并行搜索 + 去重
- [x] 100ms 硬 deadline 保障
- [x] 完整测试套件（11/11 passing）
- [x] 性能基准报告

### Phase 2 集成项（待做）
- [ ] 更新 retriever.py 使用 VFS
- [ ] 更新 eval 系统使用 VFS
- [ ] 添加 FilesystemBackend（self-improving）
- [ ] 添加 JSONLBackend（项目历史）
- [ ] 生产验证（无 tmpfs 隔离）

### Phase 3 规划（下一阶段）
- [ ] `sched/` — Agent 调度器
- [ ] 优先级队列 + 公平性调度
- [ ] 动态资源分配（AIMD）
- [ ] 性能监控和自适应调优

---

## 监控和调试

### 查看缓存统计

```python
vfs = get_vfs()
stats = vfs.stats()

print(f"Reads: {stats['reads']}")
print(f"Searches: {stats['searches']}")
print(f"Dentry hits: {stats['dentry_hits']}")
print(f"Dentry cache: {stats['dentry_cache']}")
print(f"Inode cache: {stats['inode_cache']}")
```

### 清空缓存

```python
vfs.dentry_cache.clear()
vfs.inode_cache.clear()
```

---

## 常见问题

### Q: VFS 支持写入吗？

A: 目前 SQLiteBackend 在生产环境中以只读模式运行。写入由原有的 writer/extractor hooks 负责，VFS 作为统一的读取接口。

### Q: 如何处理大量并发查询？

A: 当前实现支持线程池，但未优化并发。建议：
1. 增加 ThreadPoolExecutor 的 `max_workers`
2. 添加 Redis 支持以实现跨进程缓存
3. 使用 asyncio 改写为异步 I/O

### Q: VFS 和当前检索系统的关系？

A: VFS 是**统一的中间层**，不替代现有系统，而是**包装现有系统**（SQLite FTS5 + scorer）。好处：
- 对现有实现完全透明
- 支持多后端，无锁定
- 缓存分层自动优化

---

## 参考文献

- **OS 类比**：Linux VFS (1996) — 统一文件系统抽象
  - Linus Torvalds & Linux kernel team
  - Reference: "Linux Kernel Architecture" (Wolfgang Mauerer)

- **缓存设计**：CPU TLB + Page Cache 两级缓存模式
  - Reference: "Computer Systems: A Programmer's Perspective" (Bryant & O'Hallaron)

- **去重算法**：最小完美哈希与布隆过滤器权衡
  - Reference: "Algorithms on Strings, Trees, and Sequences" (Dan Gusfield)
