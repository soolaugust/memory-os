<div align="center">

# AIOS Memory OS

**Persistent memory for AI agents — designed like a kernel, not a database.**

[![Python](https://img.shields.io/badge/python-3.12%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![SQLite](https://img.shields.io/badge/storage-SQLite%20WAL-lightgrey?logo=sqlite)](https://sqlite.org/)
[![Tests](https://img.shields.io/badge/tests-44%20passing-brightgreen)](#testing)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Iterations](https://img.shields.io/badge/iterations-362%2B-orange)](#roadmap)

[English](./README.en.md) · [中文](./README.md)

</div>

---

## The Problem

Every time you start a new conversation with an AI assistant, it starts from scratch. Every decision, every discovered pitfall, every architectural constraint — gone. You re-explain context. It re-learns the same lessons. And if you run multiple agents in parallel, they have no way to share what they've learned.

This is not a model limitation. It's a missing infrastructure layer.

---

## The Solution

Memory OS applies **operating system memory management philosophy** to AI cognitive resource management. The same principles that let Linux handle millions of processes with limited RAM now give AI agents persistent, retrievable, multi-agent-shared memory.

| OS Concept | Memory OS Equivalent |
|---|---|
| RAM (runtime working space) | Context window — what the AI sees right now |
| Disk (persistent storage) | Knowledge base — facts that survive across sessions |
| Demand paging | Smart retrieval — fetch relevant memories on-demand |
| Process scheduling | Multi-agent coordination — multiple AIs share one knowledge graph |
| CRIU checkpoint/restore | Session snapshots — save state, resume seamlessly |
| kworker thread pool | Async extraction pool — I/O offloaded from the critical path |

---

## How It Works

```
You speak
  → System retrieves relevant memories → injects into context
  → AI responds with full context
  → Session ends → decisions/insights auto-extracted → persisted to store.db
  → Next session starts → working set restored automatically
```

Zero manual memory management. The whole pipeline runs inside Claude Code hooks.

---

## Key Metrics

| Metric | Value |
|---|---|
| Retrieval latency P50 (TLB hit) | **~0.1 ms** |
| vs. subprocess baseline | **540× faster** (54 ms → 0.1 ms) |
| Recall@3 improvement (BM25 vs baseline) | **+147%** |
| MRR improvement | **+320%** |
| A/B answer quality uplift | **+68%** (3.55 vs 2.12) |
| Cross-session recall | **94.2%** |
| Knowledge base size | **427 chunks / 8 types** |
| Hot-path retrieval | **1.74 µs/op** (iter 258, −84.7% from baseline) |
| Total iterations | **362+** |
| **Token injection per call** | **~44 tokens** (avg 178 chars) |
| **Token net ROI per call** | **~+256 tokens** saved (inject 44, save ~300 re-explanation) |
| FULL→LITE demotion savings (iter 361) | **~62 tokens/repeat** (69.6% reduction on re-injection) |
| Session dedup excluded chunks (iter 359) | chunks injected ≥2× automatically excluded |
| Same-hash TLB bypass | **zero tokens** overhead on repeated prompts |

---

## Architecture Overview

```
Claude Code
    ↕  hooks (syscall boundary)
┌─────────────────────────────────────────┐
│  hooks/                                  │
│  ├── loader.py        (SessionStart)     │  Working Set restore + CRIU restore
│  ├── retriever_wrapper.sh (UserPrompt)   │→ retriever_daemon.py (persistent proc)
│  ├── writer.py        (UserPrompt)       │  Knowledge write + task state
│  ├── extractor.py     (Stop)             │  Knowledge extraction + CRIU dump
│  ├── extractor_pool.py                   │  Async kworker pool (iter 260)
│  ├── output_compressor.py (PostToolUse)  │  zram — large output compression hints
│  ├── tool_profiler.py (PostToolUse)      │  eBPF-style tool call efficiency
│  └── parallel_hint.py (UserPrompt)       │  CFS parallel task detection
└─────────────────────────────────────────┘
    ↕  VFS unified data layer
┌─────────────────────────────────────────┐
│  ~/.claude/memory-os/store.db            │
│  memory_chunks / swap_chunks             │
│  checkpoints / dmesg                     │
│  FTS5 full-text index (bigram CJK)       │
└─────────────────────────────────────────┘
    ↕  IPC (ipc_msgq)
┌─────────────────────────────────────────┐
│  net/agent_notify.py                     │  Cross-agent knowledge broadcast
│  extractor_pool (kworker pool)           │  Async extraction worker (iter 260)
└─────────────────────────────────────────┘
```

Full architecture details: [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md)

---

## Design Philosophy

Every subsystem maps to a Linux kernel mechanism:

| Feature | Linux Analogy | Iteration |
|---|---|---|
| Knowledge retrieval injection | Demand paging (page fault) | iter 1 |
| Working set preload | Denning Working Set Model | iter 18 |
| Knowledge eviction | kswapd + OOM Killer | iter 25, 38 |
| Session restore | CRIU Checkpoint/Restore | iter 49 |
| Congestion control | TCP AIMD + auto-tuning | iter 50, 51 |
| Multi-generation LRU | MGLRU (Linux 6.x) | iter 44 |
| Access pattern monitoring | DAMON | iter 42 |
| Output compression hints | zram | iter 110 |
| Persistent retrieval daemon | vDSO + Unix socket | iter 162 |
| Two-level TLB cache | CPU TLB L1/L2 | iter 179 |
| FTS result cache | Page cache | iter 205 |
| Multi-agent isolation | Linux namespace (PID/mount) | iter 259 |
| Async extraction pool | kworker thread pool + pdflush | iter 260 |
| FTS5 auto-optimize | ext4 online defrag | iter 360 |
| FULL→LITE injection demotion | page cache dirty bit fast-path | iter 361 |
| Proactive swap warmup | MGLRU proactive reclaim | iter 362 |

---

## Problems Solved

<details>
<summary><strong>Problem 1: No cross-session memory — starting from zero every time</strong></summary>

Every new conversation loses all previous decisions, pitfalls, and architectural constraints. Significant "warm-up" time is wasted rebuilding context.

**Solution: Knowledge persistence + automatic retrieval injection**

- At session end, decisions / reasoning chains / design constraints / quantitative evidence are auto-extracted from the conversation and written to `store.db`
- At session start, relevant knowledge is retrieved and injected into context (`additionalContext`) — no manual copy-paste required

**Measured results:**
```
Knowledge base: 427 chunks (decision 230 / quantitative_evidence 49 / causal_chain 41 / procedure 39 / ...)
BM25 retrieval vs. importance-rank baseline: Recall@3 +147% (58.3% vs 23.6%), MRR +320%
A/B test (memory-os assisted vs. none): 8/12 wins, avg score 3.55 vs 2.12 (+68%)
Session Recall@3: 94.2%
Retrieval hit rate: 61.9% of chunks actually retrieved (top chunk hit ×2043)
```
</details>

<details>
<summary><strong>Problem 2: High retrieval latency — noticeable lag on every prompt</strong></summary>

Early subprocess-based retrieval: P50 ~54 ms. Every user keystroke had visible stutter.

**Solution: Daemon architecture + three-level cache**

- **`retriever_daemon.py`**: Persistent process served via Unix socket, eliminates Python import startup cost
- **FTS5 result cache**: Same FTS expression cached, hit ~0.3 µs
- **Two-level TLB**: L1 (exact prompt hash) + L2 (injection hash fuzzy match), direct empty response on hit

**Measured latency (iter 238, 2026-04):**
```
SKIP path (ack/irrelevant prompts):  <0.1 ms  (TLB hit, no injection)
Full retrieval + injection P50:      ~0.1 ms
Full retrieval + injection P95:      ~0.14 ms
Historical: subprocess era P50 ~54 ms → daemon ~0.1 ms (540× reduction)
```
</details>

<details>
<summary><strong>Problem 3: Context window fills up, forced compaction loses reasoning chains</strong></summary>

In long tasks, tool output floods the context, triggers compaction, and key reasoning chains are lost.

**Solution: Multi-layer context compression and swap**

- **zram (`output_compressor.py`)**: Large outputs trigger attention-guidance hints via `additionalContext` (Bash >3KB / Read >4KB)
- **Context Pressure Governor**: Four watermark levels (LOW/NORMAL/HIGH/CRITICAL) dynamically shrink injection window
- **Swap**: Low-frequency knowledge swapped out to `swap_chunks`, restored on demand
</details>

<details>
<summary><strong>Problem 4: Session interruption loses "what I was doing"</strong></summary>

After context compaction or session timeout, Claude doesn't know where it left off.

**Solution: CRIU session checkpoint**

- On Stop, unfinished intent extracted from last assistant message (next_actions / open_questions)
- Saved to `session_intents` DB table, auto-injected at next SessionStart
- 24h TTL, expires automatically
</details>

<details>
<summary><strong>Problem 5: Architectural constraints scattered in history, easily violated</strong></summary>

"Don't mock the database here", "must call ensure_schema first" — these constraints live in old conversations, not in code. New sessions easily violate them.

**Solution: Design constraint extraction with highest protection**

- Auto-detects constraint patterns (22 patterns: `must not / forbidden / WARNING: / otherwise ...`)
- `design_constraint` type: `importance=0.95`, `oom_adj=-800` (highest protection, never evicted)
- Auto-injected at every UserPromptSubmit, constraints appear at the top of context

**Measured: 21 active design constraints, top constraint hit ×2043 times.**
</details>

<details>
<summary><strong>Problem 6: Repeated tool calls — reading the same file multiple times</strong></summary>

Claude in complex tasks often re-reads the same files or re-runs the same commands, wasting context tokens.

**Solution: Tool call efficiency profiler (`tool_profiler.py`)**

- PostToolUse records each call (tool_name / output_bytes / duration_ms)
- Detects inefficiency: same-file Read ≥3 times / same Bash command ≥2 times
- Injects suggestion hint when detected
</details>

<details>
<summary><strong>Problem 7: Serial execution of independent sub-tasks</strong></summary>

**Solution: CFS parallel task detection (`parallel_hint.py`)**

- Detects parallelism signals at UserPromptSubmit: explicit parallel words / list tasks (3+ items) / comparison analysis
- Injects: `[CFS] Detected N independent sub-tasks — consider Agent tool for parallel execution`
</details>

<details>
<summary><strong>Problem 8: Multiple agents overwriting each other's memory (iter 259)</strong></summary>

When multiple Claude sessions run concurrently, last-writer-wins race conditions corrupt shared files (`.shadow_trace.json`, `ctx_pressure_state.json`).

**Solution: Per-session isolation (Linux namespace analogy)**

- `shadow_traces` and `session_intents` tables use `session_id` as PRIMARY KEY — each agent's state is isolated
- Per-session files: `.shadow_trace.{sid[:16]}.json`, `ctx_pressure_state.{sid[:16]}.json`
- Verified by 20-test suite in `tests/test_agent_team.py`
</details>

<details>
<summary><strong>Problem 9: Stop hook blocks on I/O-heavy transcript parsing (iter 260)</strong></summary>

`extractor.py` spends 50–150 ms reading and parsing transcript files synchronously in the Stop hook critical path.

**Solution: Async extraction worker pool (kworker analogy)**

- Stop hook calls `submit_extract_task()` → writes to `ipc_msgq` → returns immediately (<5 ms)
- `extractor_pool.py` persistent daemon polls `ipc_msgq`, runs full pipeline in `ThreadPoolExecutor(3)`
- Graceful fallback: if pool not running → synchronous execution (original behavior)

```
Stop hook  →  ipc_msgq[extract_task]  →  extractor_pool (kworker ×3)  →  store.db
  <5ms              persist                   async parallel                 broadcast
```
</details>

<details>
<summary><strong>Problem 10: Repeated injection wastes tokens — full context re-attached every call (iter 359, 361)</strong></summary>

Without deduplication, the same chunk is injected with its full `raw_snippet` on every prompt in a long session. Content already in the model's working memory is re-transmitted, wasting tokens.

**Solution: Three-layer token budget enforcement**

- **FULL→LITE demotion (iter 361):** Once a chunk has been injected in full format (summary + raw_snippet) in this session, subsequent injections use LITE format (summary only) — raw text has zero marginal value after the first time.
- **Session dedup (iter 359):** Chunks injected ≥ `session_dedup_threshold` (default: 2) times are excluded entirely from context output.
- **Same-hash TLB bypass:** Identical prompt hashes return the cached result immediately — zero DB queries, zero new tokens injected.

**Measured (validated by `tests/test_token_budget.py`):**
```
Injection cost:             ~44 tokens/call  (avg 178 chars)
FULL→LITE saving:           ~62 tokens/repeat (69.6% reduction per re-injected chunk)
User re-explanation saved:  ~300 tokens/call  (no need to re-describe context)
Net token ROI:              ~+256 tokens/call
Context cap enforced:       ≤ 800 chars       (max_context_chars sysctl)
```
</details>

---

## Roadmap

| Phase | Status |
|---|---|
| Basic memory management (iter 1–100) — persist, evict, prioritize | ✅ Done |
| Persistent retrieval daemon + multi-level cache (iter 162–205) | ✅ Done |
| Data-driven precision tuning — 258 iterations, −84.7% latency (iter 235–258) | ✅ Done |
| Multi-agent isolation — per-session namespacing, IPC broadcast (iter 259) | ✅ Done |
| Async extraction pool — Stop hook offload, kworker pool (iter 260) | ✅ Done |
| Token budget optimization — FULL→LITE demotion, session dedup, swap warmup (iter 359–362) | ✅ Done |
| Distributed multi-agent shared memory — NUMA/RDMA analogy (iter 363+) | 🔜 Planned |

---

## Quick Start

### Prerequisites

- Python 3.12+
- SQLite (built-in, no installation needed)
- `nc` (netcat) and `flock` (usually pre-installed on Linux/macOS)
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-org/memory-os ~/codes/aios/memory-os
cd ~/codes/aios/memory-os

# 2. Create data directory (store.py auto-creates schema on first run)
mkdir -p ~/.claude/memory-os

# 3. Add hooks to ~/.claude/settings.json
```

**`~/.claude/settings.json` hooks configuration:**

```json
{
  "hooks": {
    "SessionStart": [
      {
        "type": "command",
        "command": "python3 /path/to/memory-os/hooks/loader.py",
        "timeout": 10
      }
    ],
    "UserPromptSubmit": [
      {
        "type": "command",
        "command": "bash /path/to/memory-os/hooks/retriever_wrapper.sh",
        "timeout": 10,
        "async": false
      },
      {
        "type": "command",
        "command": "python3 /path/to/memory-os/hooks/writer.py",
        "timeout": 10,
        "async": false
      },
      {
        "type": "command",
        "command": "python3 /path/to/memory-os/hooks/parallel_hint.py",
        "timeout": 3,
        "async": false
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Bash|Read",
        "hooks": [{
          "type": "command",
          "command": "python3 /path/to/memory-os/hooks/output_compressor.py",
          "timeout": 5
        }]
      },
      {
        "matcher": "*",
        "hooks": [{
          "type": "command",
          "command": "python3 /path/to/memory-os/hooks/tool_profiler.py",
          "timeout": 5,
          "async": true
        }]
      }
    ],
    "Stop": [
      {
        "type": "command",
        "command": "python3 /path/to/memory-os/hooks/extractor.py",
        "timeout": 10,
        "async": true
      }
    ]
  }
}
```

### Verify Installation

Run from the project root:

```bash
# Test SessionStart hook
echo '{"session_id":"test","transcript_path":"/dev/null","cwd":"'$(pwd)'"}' \
  | python3 hooks/loader.py
# Expected: {"hookSpecificOutput": ...} or empty (no history yet)

# Test retriever daemon startup
echo '{"session_id":"test","prompt":"test query","cwd":"'$(pwd)'"}' \
  | bash hooks/retriever_wrapper.sh
# Expected: daemon starts automatically, returns {} or injected content within 3s

# Confirm daemon is running
ls /tmp/memory-os-retriever.sock && echo "daemon running"

# Run the test suite
python3 -m pytest tests/test_agent_team.py tests/test_chaos.py -q
```

### Daemon Management

The retriever daemon starts automatically on the first request — no manual startup needed.

```bash
# View daemon logs
tail -f ~/.claude/memory-os/daemon.log

# Restart daemon after code update
pkill -f retriever_daemon.py
# It restarts automatically on the next retriever_wrapper.sh call

# Start extractor pool (optional, iter 260 async extraction)
bash hooks/extractor_pool_wrapper.sh start
bash hooks/extractor_pool_wrapper.sh status
```

---

## Testing

```bash
# Core multi-agent isolation tests (A1–A20)
python3 -m pytest tests/test_agent_team.py -v

# Chaos / fault tolerance tests
python3 -m pytest tests/test_chaos.py -v

# All stable tests
python3 -m pytest tests/test_agent_team.py tests/test_chaos.py -q
```

The test suite covers:
- Per-session `shadow_traces` and `session_intents` DB isolation
- CRIU checkpoint cleanup by `session_id`
- Concurrent multi-agent shadow trace writes (no data loss)
- Cross-agent IPC notification delivery end-to-end
- Extractor pool task queue (submit / dequeue / CONSUMED semantics)
- Goals progress idempotency across multiple sessions

---

## Dependencies

| Dependency | Purpose |
|---|---|
| Python 3.12+ | Core runtime |
| SQLite (built-in) | Primary store, FTS5 full-text index |
| `nc` (netcat) | Unix socket communication with retriever daemon |
| `flock` | Single-instance daemon startup lock |

No GPU required. No external API required. Everything runs locally.

---

## Contributing

Contributions are welcome. Each subsystem is isolated behind a clean interface — hooks call into `store.py` / `store_vfs.py` / `store_criu.py` through a VFS layer, making individual components testable in isolation.

Before submitting a PR, run:
```bash
python3 -m pytest tests/test_agent_team.py tests/test_chaos.py -q
```

---

<div align="center">

Built with the philosophy: *if the OS solved it, we can learn from it.*

</div>
