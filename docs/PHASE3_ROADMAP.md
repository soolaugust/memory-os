# Phase 3-4 Roadmap: Agent Orchestration & Hook Fusion

**Generated:** 2026-04-20 after Phase 2 VFS completion  
**Context:** KnowledgeVFS unified layer now enables stateful scheduling and orchestration

---

## Phase 3: Dynamic Resource Scheduler (sched/)

**Goal:** Prevent agent starvation, guarantee deadline adherence for VFS queries

### Design

```
┌─────────────────────────────────────────────┐
│     Agent Team (memory-os hooks)            │
│  - extractor (high priority)                │
│  - retriever (normal priority)              │
│  - eval (low priority)                      │
└────────┬────────────────────────────────────┘
         │ submit(priority, deadline, min_quota)
         ▼
┌─────────────────────────────────────────────┐
│     Priority Queue Scheduler (sched/)       │
│  - AIMD quota allocation                    │
│  - Deadline enforcement (hard/soft)         │
│  - Adaptive backpressure                    │
└────────┬────────────────────────────────────┘
         │ grant(quota_ms) or deny()
         ▼
┌─────────────────────────────────────────────┐
│     KnowledgeVFS + Backends                 │
│  - dentry/inode cache hit (0.003ms)         │
│  - SQLite backend (0.4ms avg)               │
│  - Future: Filesystem, JSONL backends       │
└─────────────────────────────────────────────┘
```

### Key Components

1. **PriorityQueue**
   - Per-agent state: (priority, deadline, quota_used_ms, quota_remaining_ms)
   - Insert: O(log N) heapq
   - Pop: O(log N) by deadline

2. **AIMD Allocator**
   - Increase phase: quota += 10% when idle (no queue depth)
   - Decrease phase: quota *= 0.5 when overload (queue_depth > threshold)
   - Prevents oscillation with exponential backoff

3. **Monitoring**
   - Latency histogram: [p50, p99, max] per agent
   - Resource utilization: (quota_used / quota_allocated) per agent
   - Queue depth: max depth seen, average wait time

4. **Integration Points**
   - retriever.py: call scheduler.acquire(priority=normal, deadline_ms=80) before vfs.search()
   - extractor.py: call scheduler.acquire(priority=high, deadline_ms=200) for checkpoint writes
   - eval.py: call scheduler.acquire(priority=low, deadline_ms=5000) for large batch evals

### Success Metrics

| Metric | Target | Method |
|--------|--------|--------|
| Deadline adherence | >99% | Track deadline_exceeded events |
| Mean latency | <100ms (w/o contention) | histogram p50 |
| Tail latency | <150ms (p99) | histogram p99 |
| Quota utilization | 70-90% | (used/allocated) per agent |
| Scheduler overhead | <1% | measure scheduler.acquire() + grant() time |

---

## Phase 4: Hook Orchestration & Event Stream Fusion

**Goal:** Coordinate multi-hook workflows (extractor→retriever→eval) with transaction semantics

### Motivation

Current problem:
- Hooks run independently, no coordination
- Extractor writes → retriever reads → eval scores in blind sequence
- No rollback on eval failure (extractor+retriever already committed)
- No deduplication across extraction batches

### Design

```
Extractor Hook
  │ emits: ChunkExtracted(id, content, source)
  │
  └─→ Dedup Filter
       ├─ check SHA256(content) in VFS
       ├─ if duplicate → emit DuplicateDetected(id, prior_id, score_delta)
       └─ if new → pass through
  │
  └─→ Priority Scheduler
       ├─ request: priority=high, deadline=200ms
       └─ grant: quota=10ms (proceed) or deny (backpressure)
  │
  └─→ KnowledgeVFS
       └─ write(VFSItem)

       ▼

Retriever Hook
  │ emits: QueryReceived(query, top_k, deadline_ms=80)
  │
  └─→ Priority Scheduler
       ├─ request: priority=normal, deadline=80ms
       └─ grant: acquire resource slot
  │
  └─→ KnowledgeVFS.search()
       ├─ L1 dentry cache hit (0.003ms) → done
       ├─ L2 inode cache hit (0.1ms) → done
       └─ L3 backend query (0.4ms) → done
  │
  └─→ Scorer (enhanced w/ source weights from VFS metadata)
       └─ emit: QueryCompleted(results, latency_ms)

       ▼

Eval Hook
  │ emits: EvalBatch(queries, retriever_results, ground_truth)
  │
  └─→ Priority Scheduler
       ├─ request: priority=low, deadline=5000ms
       └─ grant: acquire batch-sized slot
  │
  └─→ Compute Recall@K, NDCG
       └─ emit: EvalResult(metrics, failures)

Observability Layer
  │ subscribes to all events (non-blocking)
  │
  ├─→ Duplicate Detection Dashboard (DedupStats)
  ├─→ Latency Monitor (LatencyHistogram)
  ├─→ Resource Utilization (SchedulerStats)
  └─→ Eval Metrics Tracker (RecallTrend)
```

### Coordination Semantics

1. **Transaction-like atomicity** (without true ACID):
   - Extraction → VFS write → emit ChunkWritten (then retriever can read)
   - Retriever → VFS search → emit QueryAnswered (then eval can score)
   - Eval → compute → emit EvalComplete (allows next extractor batch)

2. **Deduplication ordering**:
   - New chunk arrives → check VFS SHA256 → if found, skip or merge score
   - Prevents duplicate extractor work + retriever confusion

3. **Backpressure**:
   - Scheduler.deny() signals extractor to wait (avoid checkpoint bloat)
   - Scheduler.acquire() blocks retriever if under high load

### Implementation Strategy

1. **Event bus** — simple list of subscribers per event type
2. **Hook decorators** — wrap entry/exit with emit(EventType)
3. **Checkpoint fusion** — merge extractor+retriever checkpoints (single VFS write)
4. **Monitoring endpoints** — expose /metrics for grafana/prometheus

### Success Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| Duplicate catch rate | >95% | SHA256 dedup effectiveness |
| Multi-hook latency (E→R→Eval) | <500ms p99 | end-to-end pipeline |
| Checkpoint merge rate | >80% | % of batches merged successfully |
| Backpressure trigger rate | 5-10% | healthy level of resource feedback |

---

## Integration Readiness Checklist

### Phase 2 ✅ Complete
- [x] VFSItem + VFSMetadata serialization
- [x] SQLiteBackend with FTS5
- [x] dentry + inode L1/L2 cache
- [x] Parallel search, deduplication, 100ms deadline
- [x] Performance baselines verified

### Phase 3 (Next)
- [ ] sched/scheduler.py — PriorityQueue + AIMD
- [ ] sched/monitor.py — LatencyHistogram + ResourceTracker
- [ ] Integration test: multiple agents under contention
- [ ] Verify deadline adherence >99%

### Phase 4 (After Phase 3)
- [ ] Hook event bus implementation
- [ ] Hook decorators for emit(EventType)
- [ ] Dedup filter on extractor output
- [ ] Multi-hook checkpoint fusion
- [ ] Observability dashboard

### Future (Phase 5+)
- [ ] Add FilesystemBackend for self-improving
- [ ] Add JSONLBackend for project history
- [ ] Distributed cache (Redis) for cross-process sharing
- [ ] Async I/O rewrite for concurrent agents
- [ ] ML-based quota prediction (AIMD → learned allocation)

---

## Decision: Sequential Execution Order

**Why not parallel?**
- Phase 3 unlocks Phase 4 coordination (scheduler must exist before hook orchestration)
- Phase 4 depends on Phase 3's resource guarantees to work reliably
- Each phase has clear integration points (test before proceeding)

**Why not skip to Phase 4?**
- Without scheduler, Phase 4 orchestration can still deadlock under load
- Better to establish resource fairness first, then layer coordination

**Estimated effort:**
- Phase 3: 2-3 sessions (scheduler core + monitoring)
- Phase 4: 2-3 sessions (event bus + integration)
- Total: 1-2 weeks of steady development

---

## Current State Summary

| Component | Status | Perf | Notes |
|-----------|--------|------|-------|
| VFS Core | ✅ Phase 2 Complete | 0.4ms search, 0.003ms cached | 11/11 tests passing |
| SQLiteBackend | ✅ Phase 2 Complete | 5-10ms FTS5 | Readonly mode in prod |
| Caches (L1/L2) | ✅ Phase 2 Complete | <0.1ms hit | TTL 300s |
| Scheduler | 🔲 Phase 3 Ready | TBD | Design complete, awaiting implementation |
| Hook orchestration | 🔲 Phase 4 Ready | TBD | Event bus design ready |
| Backends (FS/JSONL) | 🔲 Phase 2+ Ready | TBD | Blocked on Phase 3 completion |

---

## Next Session Entry Point

Start Phase 3 by:
1. Create `sched/scheduler.py` with PriorityQueue + AIMD logic
2. Implement monitor.py for latency tracking
3. Write integration tests (multiple agents, simulated load)
4. Verify deadline adherence metric

Then update Task #9 status to track Phase 2 Integration completion.
