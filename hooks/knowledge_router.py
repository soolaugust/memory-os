#!/usr/bin/env python3
"""
KnowledgeRouter — 统一知识寻址层
对应 OS 历史节点：Unix 文件系统（统一命名空间，1969）

职责：
- 给任意 query，路由到所有知识系统并返回统一格式结果
- 不合并存储，只建统一接口（抽象层）
- 可作为库被 retriever.py / loader.py / 其他 hook 调用

支持的知识系统：
  L1  memory-os store.db     — 决策/排除路径/推理链（BM25）
  L2  MEMORY.md              — 跨会话持久记忆索引（关键词匹配）
  L3  self-improving/        — 行为规则/工作流/corrections（文件扫描）
  L4  metamemory             — 共享知识库（mm CLI，按需）

路由优先级：L1 > L2 > L3 > L4（速度由快到慢，短路策略）

v4 升级（迭代24）：Per-Request Connection Scope
  OS 类比：task_struct.files_struct — fd table 随 task 生命周期
  route() 和 _search_memory_os() 接受外部 conn 参数，
  复用调用方（retriever.py）的连接，避免重复 open_db()/close()。
  仅在 standalone 调用时自行管理连接（向后兼容）。

历史：
  v3 迭代19：PCID 跨进程持久缓存（mtime 校验 invalidation）
"""
import sys
import re
import json
import time
import sqlite3
import subprocess
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
from utils import resolve_project_id
from store import open_db, ensure_schema, fts_search as _fts_search
from bm25 import hybrid_tokenize as _tokenize, bm25_normalized as _bm25_norm  # 迭代22: Shared Library（非 SQLite 源仍用）
from config import get as _sysctl  # 迭代27: sysctl Runtime Tunables

MEMORY_OS_DIR = Path.home() / ".claude" / "memory-os"
STORE_DB = MEMORY_OS_DIR / "store.db"
# MEMORY.md 搜索路径列表（按优先级）
MEMORY_MD_PATHS = [
    Path.home() / ".claude" / "projects" / "-home-mi-ssd-codes-claude-workspace" / "memory" / "MEMORY.md",
    Path.home() / ".claude" / "MEMORY.md",
    Path.home() / "MEMORY.md",
]
SELF_IMPROVING_DIR = Path.home() / "self-improving"

# 迭代27：常量迁移至 config.py sysctl 注册表（运行时可调）
# 原硬编码：_sysctl("router.top_k_per_source")=3, _sysctl("router.min_score")=0.01, _CACHE_TTL_S=300
_PCID_CACHE_DIR = MEMORY_OS_DIR  # 磁盘缓存目录

# ── 进程内 TTL 缓存（L1: 同进程内极速命中）────────
_cache: dict = {}  # key -> (timestamp, data)


def _cache_get(key: str):
    entry = _cache.get(key)
    if entry and (time.time() - entry[0]) < _sysctl("router.cache_ttl_secs"):
        return entry[1]
    return None


def _cache_set(key: str, data):
    _cache[key] = (time.time(), data)


# ── PCID 磁盘持久缓存（L2: 跨进程存活）──────────
# OS 类比：PCID 让 TLB 项跨 context switch 存活
# 校验策略：缓存文件内嵌源文件 mtime 签名，mtime 变化 = cache invalidation


def _pcid_cache_path(key: str) -> Path:
    return _PCID_CACHE_DIR / f".pcid_{key}.json"


def _pcid_load(key: str) -> dict:
    """加载磁盘缓存，返回 dict 或 None（miss/过期/损坏）。"""
    p = _pcid_cache_path(key)
    if not p.exists():
        return None
    try:
        raw = p.read_text(encoding="utf-8")
        return json.loads(raw)
    except Exception:
        return None


def _pcid_save(key: str, payload: dict) -> None:
    """持久化缓存到磁盘。"""
    p = _pcid_cache_path(key)
    try:
        _PCID_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _get_mtime(path: Path) -> float:
    """安全获取文件 mtime，不存在返回 0。"""
    try:
        return path.stat().st_mtime
    except Exception:
        return 0.0


def _get_dir_mtime_sig(dir_path: Path, patterns: list) -> dict:
    """
    计算目录下多个 glob pattern 匹配文件的 mtime 签名。
    返回 {relative_path: mtime} dict，作为缓存 invalidation 依据。
    OS 类比：inode mtime 检测文件变更。
    """
    sig = {}
    if not dir_path.exists():
        return sig
    for pattern in patterns:
        for fp in sorted(dir_path.glob(pattern)):
            try:
                sig[str(fp.relative_to(dir_path))] = fp.stat().st_mtime
            except Exception:
                pass
    return sig


# ─────────────────────────────────────────────────────────────
# 统一结果格式
# ─────────────────────────────────────────────────────────────

def _result(source: str, chunk_type: str, summary: str, score: float,
            content: str = "", path: str = "") -> dict:
    return {
        "source": source,       # "memory_os" | "memory_md" | "self_improving" | "metamemory"
        "chunk_type": chunk_type,
        "summary": summary[:120],
        "score": round(score, 4),
        "content": content[:300] if content else "",
        "path": path,
    }


    # ── BM25 已迁移至 bm25.py（迭代22 Shared Library）──


# ─────────────────────────────────────────────────────────────
# L1：memory-os store.db
# ─────────────────────────────────────────────────────────────

def _search_memory_os(query: str, project: str, conn: sqlite3.Connection = None) -> list:
    """
    v6 迭代24：接受外部 conn（Per-Request Connection Scope）。
    OS 类比：task_struct.files_struct — 复用调用方的 fd，不自行 open。
    若无外部 conn 则自行打开（向后兼容 standalone 调用）。
    保留 FTS5 + Python BM25 fallback 双路径。
    """
    if not STORE_DB.exists():
        return []
    own_conn = conn is None
    try:
        if own_conn:
            conn = open_db()
            ensure_schema(conn)

        # 主路径：FTS5 索引检索
        try:
            fts_results = _fts_search(conn, query, project, top_k=_sysctl("router.top_k_per_source"))
        except Exception:
            fts_results = []

        if fts_results:
            results = []
            max_rank = max(r["fts_rank"] for r in fts_results) if fts_results else 1.0
            if max_rank <= 0:
                max_rank = 1.0
            for r in fts_results:
                score = r["fts_rank"] / max_rank
                if score >= _sysctl("router.min_score"):
                    results.append(_result("memory_os", r["chunk_type"], r["summary"], score, r["content"]))
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:_sysctl("router.top_k_per_source")]

        # Fallback：全表扫描 + Python BM25
        rows = conn.execute(
            "SELECT id, summary, content, chunk_type, importance FROM memory_chunks "
            "WHERE project=? AND summary!=''", (project,)
        ).fetchall()
        if not rows:
            return []
        docs = [f"{r[1]} {r[2]}" for r in rows]
        scores = _bm25_norm(query, docs)
        results = []
        for i, row in enumerate(rows):
            if scores[i] >= _sysctl("router.min_score"):
                results.append(_result("memory_os", row[3], row[1], scores[i], row[2]))
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:_sysctl("router.top_k_per_source")]
    except Exception:
        return []
    finally:
        if own_conn and conn:
            conn.close()


# ─────────────────────────────────────────────────────────────
# L2：MEMORY.md（关键词匹配，只解析索引行）
# ─────────────────────────────────────────────────────────────

def _load_memory_md_index() -> list:
    """
    加载 MEMORY.md 索引行（PCID 两级缓存）。
    L1: 进程内 TTL 缓存（同一 hook 进程内重复调用）
    L2: 磁盘 PCID 缓存（跨 hook 进程调用，mtime 校验）
    """
    # L1: 进程内缓存（极速）
    cached = _cache_get("md_index")
    if cached is not None:
        return cached

    # L2: PCID 磁盘缓存（跨进程）
    # 构建当前 mtime 签名
    current_mtimes = {}
    for md_path in MEMORY_MD_PATHS:
        if md_path.exists():
            current_mtimes[str(md_path)] = _get_mtime(md_path)

    pcid = _pcid_load("md_index")
    if pcid and pcid.get("mtimes") == current_mtimes:
        # PCID hit: mtime 未变，直接使用缓存数据
        entries = [tuple(e) for e in pcid.get("data", [])]
        _cache_set("md_index", entries)
        return entries

    # PCID miss: 重新扫描（冷路径）
    entries = []  # [(line, md_path), ...]
    for md_path in MEMORY_MD_PATHS:
        if not md_path.exists():
            continue
        try:
            lines = md_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        for l in lines:
            if l.strip().startswith("- ["):
                entries.append((l.strip(), str(md_path)))

    # 写回 L1 + L2
    _cache_set("md_index", entries)
    _pcid_save("md_index", {"mtimes": current_mtimes, "data": entries})
    return entries


def _search_memory_md(query: str) -> list:
    entries = _load_memory_md_index()
    if not entries:
        return []
    entry_lines = [e[0] for e in entries]
    scores = _bm25_norm(query, entry_lines)
    results = []
    for i, (line, path) in enumerate(entries):
        if scores[i] >= _sysctl("router.min_score"):
            m = re.search(r'—\s*(.+)$', line)
            summary = m.group(1).strip() if m else line
            results.append(_result("memory_md", "reference", summary, scores[i], path=path))
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:_sysctl("router.top_k_per_source")]


# ─────────────────────────────────────────────────────────────
# L3：self-improving/（文件名 + 前15行 BM25）
# ─────────────────────────────────────────────────────────────

def _clean_md_lines(lines: list) -> list:
    """
    过滤 markdown 文件中的噪声行，只保留实质内容行：
    - 去掉 frontmatter（--- 之间的区域）
    - 去掉纯分隔符（---, ***, ===）
    - 去掉注释行（<!-- ... -->）
    - 去掉过短行（< 6字符）
    - 去掉纯标点/符号行
    """
    result = []
    in_frontmatter = False
    frontmatter_count = 0
    for line in lines:
        stripped = line.strip()
        # frontmatter 检测（文件开头的 --- 块）
        if stripped == "---" and frontmatter_count < 2:
            in_frontmatter = not in_frontmatter
            frontmatter_count += 1
            continue
        if in_frontmatter:
            continue
        # 注释行
        if stripped.startswith("<!--") or stripped.endswith("-->"):
            continue
        # 纯分隔符
        if re.match(r'^[-=*`#>]{2,}$', stripped):
            continue
        # 过短
        if len(stripped) < 6:
            continue
        result.append(stripped)
    return result


def _load_self_improving_index() -> tuple:
    """
    加载 self-improving/ 文件索引（PCID 两级缓存）。
    返回 (docs: list[str], file_paths: list[Path], summaries: list[str])。

    L1: 进程内 TTL 缓存（< 0.1ms）
    L2: PCID 磁盘缓存（< 1ms，跨 hook 进程，mtime 签名校验）
    冷路径：全量 glob + 文件读取（10-50ms）

    v3 升级：PCID 消除跨进程冷启动，预期将 hash_changed P99 从 554ms 降至 <30ms。
    """
    # L1: 进程内缓存
    cached = _cache_get("si_index")
    if cached is not None:
        return cached

    if not SELF_IMPROVING_DIR.exists():
        empty = ([], [], [])
        _cache_set("si_index", empty)
        return empty

    # 构建当前 mtime 签名（glob 模式匹配的所有文件）
    si_patterns = ["memory.md", "corrections.md", "domains/*.md", "projects/*.md"]
    current_sig = _get_dir_mtime_sig(SELF_IMPROVING_DIR, si_patterns)

    # L2: PCID 磁盘缓存
    pcid = _pcid_load("si_index")
    if pcid and pcid.get("mtime_sig") == current_sig:
        # PCID hit: 文件未变更，反序列化缓存数据
        docs = pcid["docs"]
        file_map = [Path(p) for p in pcid["file_paths"]]
        summaries = pcid["summaries"]
        result = (docs, file_map, summaries)
        _cache_set("si_index", result)
        return result

    # PCID miss: 全量扫描（冷路径）
    candidates = []
    for pattern in si_patterns:
        candidates.extend(sorted(SELF_IMPROVING_DIR.glob(pattern)))

    docs, file_map, summaries = [], [], []
    for fp in candidates:
        try:
            raw_lines = fp.read_text(encoding="utf-8").splitlines()
            clean_lines = _clean_md_lines(raw_lines)
            if not clean_lines:
                continue
            doc_text = f"{fp.stem} {' '.join(clean_lines[:10])}"
            docs.append(doc_text)
            file_map.append(fp)
            summary_line = next(
                (l for l in clean_lines if not l.startswith("#")), clean_lines[0]
            )
            summaries.append(summary_line[:100])
        except Exception:
            pass

    result = (docs, file_map, summaries)
    # 写回 L1 + L2
    _cache_set("si_index", result)
    _pcid_save("si_index", {
        "mtime_sig": current_sig,
        "docs": docs,
        "file_paths": [str(fp) for fp in file_map],
        "summaries": summaries,
    })
    return result


def _search_self_improving(query: str) -> list:
    docs, file_map, summaries = _load_self_improving_index()
    if not docs:
        return []
    scores = _bm25_norm(query, docs)
    results = []
    for i, fp in enumerate(file_map):
        if scores[i] >= _sysctl("router.min_score"):
            results.append(_result(
                "self_improving", "rule", summaries[i], scores[i], path=str(fp)
            ))
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:_sysctl("router.top_k_per_source")]


# ─────────────────────────────────────────────────────────────
# L4：metamemory（mm CLI，最慢，默认关）
# ─────────────────────────────────────────────────────────────

def _search_metamemory(query: str, timeout: int = 3) -> list:
    try:
        r = subprocess.run(
            ["mm", "search", query, "--limit", "3", "--format", "json"],
            capture_output=True, text=True, timeout=timeout
        )
        if r.returncode != 0:
            return []
        data = json.loads(r.stdout)
        results = []
        for item in data[:_sysctl("router.top_k_per_source")]:
            title = item.get("title", "")
            content = item.get("content", "")[:200]
            results.append(_result("metamemory", "knowledge", title[:100], 0.5, content=content))
        return results
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────

def route(query: str, project: Optional[str] = None,
          sources: Optional[list] = None,
          include_metamemory: bool = False,
          timeout_ms: int = 100,
          conn: sqlite3.Connection = None) -> list:
    """
    统一知识查询接口（v2 并行检索 + 短路策略）。

    参数：
      query             — 自然语言查询
      project           — project_id（None 时自动推导）
      sources           — 限定来源列表（None 表示全部 L1-L3）
      include_metamemory — 是否包含 metamemory（慢，默认关）
      timeout_ms        — 总超时毫秒（默认 100ms）
      conn              — 外部 db 连接（迭代24 Per-Request Connection Scope）

    返回：
      按 score 降序排列的结果列表，每项：
        {source, chunk_type, summary, score, content, path}

    v4 升级（迭代24）：
    - Per-Request Connection Scope：接受外部 conn，复用 retriever.py 的连接
    - OS 类比：task_struct.files_struct（fd table 随 task 生命周期复用）

    历史：
    - v3 迭代19：PCID 跨进程持久缓存（mtime 校验 invalidation）
    - v2 迭代10：并行检索 + 短路策略 + 源权重归一化
    """
    if project is None:
        project = resolve_project_id()

    active_sources = list(sources) if sources else ["memory_os", "memory_md", "self_improving"]
    if include_metamemory and "metamemory" not in active_sources:
        active_sources.append("metamemory")

    # 源权重（归一化不同语料库的 BM25 分数差异）
    SOURCE_WEIGHT = {
        "memory_os": 1.0,
        "memory_md": 0.8,
        "self_improving": 0.7,
        "metamemory": 0.6,
    }

    _search_fn = {
        "memory_os": lambda: _search_memory_os(query, project, conn=conn),
        "memory_md": lambda: _search_memory_md(query),
        "self_improving": lambda: _search_self_improving(query),
        "metamemory": lambda: _search_metamemory(query),
    }

    all_results: list = []

    # 判断是否含慢源（metamemory 需要 subprocess 调用 mm CLI）
    has_slow_source = "metamemory" in active_sources
    fast_sources = [s for s in active_sources if s != "metamemory"]

    if not has_slow_source:
        # ── v3 快速路径：全串行执行（避免 ThreadPoolExecutor 开销）──
        # OS 类比：单核系统上的协作式调度，不需要抢占式调度器的开销
        for src in fast_sources:
            fn = _search_fn.get(src)
            if not fn:
                continue
            try:
                results = fn()
                weight = SOURCE_WEIGHT.get(src, 0.7)
                for r in results:
                    r["score"] = round(r["score"] * weight, 4)
                all_results.extend(results)
            except Exception:
                pass
    else:
        # ── 含慢源：快速源串行 + 慢源线程池 ──
        # 快速源串行（< 5ms 总计）
        for src in fast_sources:
            fn = _search_fn.get(src)
            if not fn:
                continue
            try:
                results = fn()
                weight = SOURCE_WEIGHT.get(src, 0.7)
                for r in results:
                    r["score"] = round(r["score"] * weight, 4)
                all_results.extend(results)
            except Exception:
                pass

        # 短路判断：快速源已有足够高质量结果 → skip 慢源
        high_quality = [r for r in all_results if r["score"] >= 0.8]
        skip_slow = len(high_quality) >= 3

        # 慢源（metamemory）线程池隔离
        if not skip_slow:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            timeout_s = timeout_ms / 1000.0
            slow_sources = [s for s in active_sources if s == "metamemory"]
            with ThreadPoolExecutor(max_workers=len(slow_sources)) as pool:
                futures = {pool.submit(_search_fn[s]): s for s in slow_sources if s in _search_fn}
                for future in as_completed(futures, timeout=timeout_s):
                    src = futures[future]
                    try:
                        results = future.result()
                        weight = SOURCE_WEIGHT.get(src, 0.7)
                        for r in results:
                            r["score"] = round(r["score"] * weight, 4)
                        all_results.extend(results)
                    except Exception:
                        pass

    # 全局去重：同 summary 保留最高分
    seen: dict = {}
    for r in all_results:
        key = r["summary"].lower().strip()
        if key not in seen or r["score"] > seen[key]["score"]:
            seen[key] = r

    deduped = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
    return deduped[: _sysctl("router.top_k_per_source") * 2]  # 最多 6 条


def format_for_context(results: list) -> str:
    """将路由结果格式化为 context 注入文本。"""
    if not results:
        return ""
    _PREFIX = {
        "decision": "[决策]", "excluded_path": "[排除]",
        "reasoning_chain": "[推理]", "rule": "[规则]",
        "reference": "[索引]", "knowledge": "[知识]", "task_state": "",
    }
    lines = ["【知识路由召回】"]
    for r in results:
        prefix = _PREFIX.get(r["chunk_type"], "")
        src_tag = f"({r['source']})"
        line = f"{prefix} {r['summary']} {src_tag}".strip()
        lines.append(f"- {line}")
    return "\n".join(lines)


if __name__ == "__main__":
    import os
    os.environ.setdefault("CLAUDE_CWD", str(Path.home() / "ssd/codes/claude-workspace"))
    query_arg = " ".join(sys.argv[1:]) or "AI OS 演化路径"
    results = route(query_arg)
    print(f"Query: {query_arg}")
    print(f"Results: {len(results)}")
    for r in results:
        print(f"  [{r['source']}|{r['chunk_type']}] score={r['score']:.4f} | {r['summary'][:80]}")
