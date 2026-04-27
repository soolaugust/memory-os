#!/usr/bin/env python3
"""
Phase 2+: FilesystemBackend — self-improving + memory-md 文件系统后端

OS 类比：tmpfs / ext4 对 VFS 的实现 — 将文件系统挂载到 VFS 命名空间。
  - read(path)   ≈ vfs_read → file.f_op->read
  - search(query) ≈ 遍历 inode + BM25 评分（内存版本 FTS5 等价）

支持的虚拟路径：
  /self-improving/<relative/path.md>   ← ~/self-improving/
  /memory-md/<relative/path.md>        ← ~/.claude/memory-os/ 的 .md 文件

iter165: Corpus cache — 进程内文档语料库缓存（inode cache 类比）
  OS 类比：Linux dentry/inode cache — VFS 层缓存目录项和 inode，
    避免每次路径解析都走到具体文件系统（ext4/tmpfs）的磁盘读取。
    文件修改时 inode mtime 变化 → page cache 失效 → 下次读才回到磁盘。
  memory-os 对应：缓存扫描到的 (doc_text, fp, sections) 语料库，
    cache key = max(fp.stat().st_mtime) over all .md candidates。
    stat() N=40 文件仅需 0.18ms（远小于 file read + tokenize 的 9ms）。
    cache hit: 跳过 file I/O + section split，直接进入 BM25 评分循环。
    BM25 评分是 query-dependent，不可缓存，但 doc tokenization 可缓存。

iter166: Corpus tuple 预计算 mtime_iso + summary（inode 属性预填充）
  OS 类比：inode prefetch — 文件系统在 readdir() 时顺便填充 inode 属性，
    避免后续 stat() 每个文件都走一次系统调用。
  memory-os 对应：corpus build 时已有 stat()，顺便计算 mtime_iso 和 summary，
    存入 corpus tuple，_section_to_item 直接读取缓存值，跳过 stat() + summary 提取。
    corpus tuple: (tf, dl, fp, mtime_iso, rel_path, item_id, summary, idx, body, header)
    _section_to_item 省去：stat() × N_results + sha256(content) × N_results。

iter177: Glob candidates cache — 进程内 glob 结果缓存（dentry cache 类比）
  OS 类比：Linux dentry cache — 路径名→inode 映射缓存，readdir() 结果缓存在 dcache，
    同一目录的后续 openat()/stat() 无需重走目录树遍历。
  问题：_glob_candidates() 每次 _get_corpus() 调用都执行 glob（目录遍历），
    SI: `~/self-improving/**/*.md` glob = 0.66ms（N=40 文件，递归遍历）
    MM: `~/.claude/projects/*/memory/*.md` glob = 1.15ms（跨 projects/ 子目录）
    corpus cache hit 路径也无法避免此开销。
  解法：缓存 glob 结果（candidates list），以 watch_dirs 的 max(st_mtime_ns) 为 key。
    watch_dirs = base_dir 及其直接子目录（SI: 5 dirs, MM: 7 dirs）。
    stat N_dirs (~5-7) << glob N_files (~12-40)：从 0.66-1.15ms 降至 0.02-0.03ms。
  失效：watch_dirs 任一 mtime_ns 变化（文件新增/删除会改变父目录 mtime）。
  OS 类比：dcache 失效 — 目录 inode mtime 变化时 dcache 对应 dentry 失效重建。
"""
import hashlib
import threading
import time
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Tuple

from vfs_core import VFSBackend, VFSItem, VFSMetadata, VFSSource


# ── BM25 轻量实现（不依赖外部库）──────────────────────────────────────────
def _tokenize(text: str) -> List[str]:
    """简单分词：小写 + 去标点"""
    return re.findall(r"[a-zA-Z\u4e00-\u9fff]+", text.lower())


def _bm25_score(query: str, doc: str, k1: float = 1.5, b: float = 0.75,
                avgdl: float = 50.0) -> float:
    """单文档 BM25 分数（简化版，不需要语料库统计）"""
    q_terms = _tokenize(query)
    d_terms = _tokenize(doc)
    dl = len(d_terms)
    if dl == 0 or not q_terms:
        return 0.0

    # 词频统计
    tf: Dict[str, int] = {}
    for t in d_terms:
        tf[t] = tf.get(t, 0) + 1

    score = 0.0
    for term in q_terms:
        f = tf.get(term, 0)
        if f == 0:
            continue
        # BM25 TF 部分（省略 IDF，假设 IDF=1 以简化）
        tf_norm = (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
        score += tf_norm

    return score


def _precompute_tf(doc: str) -> Tuple[Dict[str, int], int]:
    """预计算文档 TF 字典和文档长度（iter165 corpus cache 优化）。

    corpus cache 中存储 (tf_dict, dl) 而非原始 doc_text，
    避免每次 query 都重新 tokenize doc（O(doc_len) → O(1) lookup）。
    """
    d_terms = _tokenize(doc)
    dl = len(d_terms)
    tf: Dict[str, int] = {}
    for t in d_terms:
        tf[t] = tf.get(t, 0) + 1
    return tf, dl


def _bm25_score_tf(q_terms: List[str], tf: Dict[str, int], dl: int,
                   k1: float = 1.5, b: float = 0.75, avgdl: float = 50.0) -> float:
    """从预计算 TF 计算 BM25 分数（iter165 — 跳过 doc tokenization）。

    OS 类比：CPU cache hit — TLB/L1 命中时无需走 page table walk，
      直接从缓存取物理地址。这里 tf/dl 就是缓存的"doc 物理表示"。
    """
    if dl == 0 or not q_terms:
        return 0.0
    score = 0.0
    for term in q_terms:
        f = tf.get(term, 0)
        if f == 0:
            continue
        tf_norm = (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
        score += tf_norm
    return score


# ── 文件系统后端基类 ────────────────────────────────────────────────────────
class FilesystemBackendBase(VFSBackend):
    """文件系统后端基类

    子类只需指定 base_dir 和 source_type。
    """

    def __init__(self, base_dir: Path, source: str, min_score: float = 0.1,
                 max_content_kb: int = 64):
        """
        Args:
            base_dir: 根目录
            source: VFS 来源标识（如 "self-improving"）
            min_score: 最低 BM25 分数阈值
            max_content_kb: 单文件最大读取大小（KB）
        """
        self.base_dir = base_dir
        self._source = source
        self.min_score = min_score
        self.max_content_bytes = max_content_kb * 1024
        # ── iter165: 进程内文档语料库缓存 ──────────────────────────────
        # OS 类比：Linux inode cache — 文件 inode 对象缓存在内存，
        #   避免每次 stat()/read() 都走到磁盘。mtime 变化时自动失效。
        # _corpus_cache: (mtime_key, corpus_list)
        #   mtime_key = max(fp.stat().st_mtime) over all candidates (0.18ms)
        #   corpus_list = [(doc_text, fp, idx, body, header), ...]
        self._corpus_cache: Tuple[float, List] = (0.0, [])
        self._corpus_lock = threading.Lock()
        # ── iter177: Glob candidates cache + watch_dirs cache ────────────
        # OS 类比：Linux dentry cache — readdir() 结果缓存，目录 mtime 变化才失效。
        #   _glob_candidates() 每次调用都执行 OS 目录遍历（SI: 0.66ms, MM: 1.15ms）。
        #   缓存 glob 结果，以 watch_dirs max(st_mtime_ns) 为 key（~0.02-0.03ms）。
        # _glob_cache: (dir_mtime_key, candidates_list)
        #   dir_mtime_key = max(d.stat().st_mtime_ns for d in _watch_dirs())
        # _watch_dirs_cached: 永久缓存 _watch_dirs() 结果（daemon 60s 内不会新增 project 目录）
        #   消除 MemoryMdBackend._watch_dirs() 中的 iterdir() 开销（0.27ms → ~0ms after 1st call）
        self._glob_cache: Tuple[int, List] = (0, [])
        self._glob_cache_lock = threading.Lock()
        self._watch_dirs_memo: List = []  # lazy init: populated on first _watch_dirs() call

    def read(self, path: str) -> Optional[VFSItem]:
        """按虚拟路径读取文件

        Args:
            path: /<source>/<relative/path.md>
        """
        parts = path.strip("/").split("/", 1)
        if len(parts) < 2 or parts[0] != self._source:
            return None

        rel_path = parts[1]
        filepath = self.base_dir / rel_path

        if not filepath.exists() or not filepath.is_file():
            return None

        return self._file_to_item(filepath, score=1.0)

    def _glob_candidates(self) -> List[Path]:
        """返回此后端的候选文件列表（子类可覆盖以限制 glob 范围）。"""
        return sorted(self.base_dir.glob("**/*.md"))

    def _watch_dirs(self) -> List[Path]:
        """返回用于 glob cache 失效检测的目录列表（子类可覆盖）。
        默认：base_dir + 直接子目录（一级，stat N~5 dirs << glob N~40 files）。
        iter177: 结果永久缓存在 _watch_dirs_memo（daemon 生命周期内目录结构不变）。
        OS 类比：inotify watch_list — 只监视目录 mtime，不监视每个 inode。
        """
        if self._watch_dirs_memo:
            return self._watch_dirs_memo
        dirs = [self.base_dir]
        try:
            dirs += [d for d in self.base_dir.iterdir() if d.is_dir()]
        except Exception:
            pass
        self._watch_dirs_memo = dirs
        return dirs

    def _get_dir_mtime_key(self) -> int:
        """iter177: 计算 watch_dirs 的 max(st_mtime_ns)，用作 glob/corpus 缓存 key。
        OS 类比：inotify read() — 读取目录变化事件，~0.02-0.04ms。
        """
        watch = self._watch_dirs()
        try:
            return max((d.stat().st_mtime_ns for d in watch if d.exists()), default=0)
        except Exception:
            return 0

    def _glob_candidates_cached(self) -> Tuple[int, List[Path]]:
        """iter177: 缓存 glob 结果，以 watch_dirs max(st_mtime_ns) 为失效 key。
        返回 (dir_mtime_key, candidates) — key 可被 _get_corpus 复用，避免重复 stat。
        cache hit: ~0.02-0.04ms（stat N_dirs + lock lookup）
        cache miss: 0.66-1.15ms（glob）→ 写回缓存
        OS 类比：dcache hit — 路径名→inode 映射命中，无需走目录树。
        """
        dir_mtime_key = self._get_dir_mtime_key()

        with self._glob_cache_lock:
            cached_key, cached_list = self._glob_cache
            if cached_key == dir_mtime_key and cached_list:
                return dir_mtime_key, cached_list

        # cache miss: 执行 glob（目录树遍历）
        candidates = self._glob_candidates()

        with self._glob_cache_lock:
            self._glob_cache = (dir_mtime_key, candidates)
        return dir_mtime_key, candidates

    def _get_corpus(self) -> list:
        """返回缓存的文档语料库（corpus cache hit/miss）。

        corpus = [(tf, dl, fp, mtime_iso, rel_path, item_id, summary, idx, body, header), ...]
        iter165: tf, dl 预计算（跳过 doc tokenization）
        iter166: mtime_iso, rel_path, item_id, summary 预计算（跳过 per-result stat + summary 提取）
        iter177: glob 结果缓存（0.66-1.15ms → 0.02ms）+ corpus key 改为 watch_dirs mtime_ns
                 corpus cache key = max(dir.stat().st_mtime_ns for dir in _watch_dirs())
                 (0.02-0.04ms，替代 max(fp.stat for N=40 files) = 0.18-0.21ms)

        iter166 — OS 类比：inode prefetch
          readdir() 顺便填充 inode 属性（stat），避免后续 N 次单独 stat()。
          cache hit path (iter177): watch_dirs stat ≈ 0.03ms + lock lookup ≈ 0.002ms
          cache miss path: glob + stat + read + split + precompute_tf + precompute_meta ≈ 9ms
        """
        if not self.base_dir.exists():
            return []

        # iter177: corpus cache key = watch_dirs max(mtime_ns)（0.02-0.04ms vs 0.18-0.21ms）
        # 复用 _glob_candidates_cached() 返回的 key，只调用一次 _get_dir_mtime_key()。
        # OS 类比：inotify — 监视目录 mtime，不逐一 stat 每个文件。
        mtime_key, candidates = self._glob_candidates_cached()

        with self._corpus_lock:
            cached_key, cached_corpus = self._corpus_cache
            if cached_key == mtime_key and cached_corpus:
                return cached_corpus  # cache hit

        if not candidates:
            return []

        # cache miss: build corpus (file I/O + section split + TF precompute + meta precompute)
        corpus = []
        for fp in candidates:
            try:
                stat = fp.stat()
                if stat.st_size == 0:
                    continue
                content = fp.read_bytes()[:self.max_content_bytes].decode("utf-8", errors="ignore")
                # iter166: 预计算 meta（一次 stat，多次复用）
                mtime_iso = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
                rel_path = str(fp.relative_to(self.base_dir))
                if len(content) >= 2048:
                    sections = self._split_sections(content)
                    for idx, (header, body) in enumerate(sections):
                        doc_text = f"{fp.stem} {header} {body[:400]}"
                        tf, dl = _precompute_tf(doc_text)
                        # iter166: 预计算 summary 和 item_id
                        if header and len(header) >= 5:
                            summary = header[:120]
                        else:
                            summary = self._extract_summary(body, fp.stem)
                        item_id = f"{rel_path}#{idx}" if idx > 0 else rel_path
                        corpus.append((tf, dl, fp, mtime_iso, rel_path, item_id, summary, idx, body, header))
                else:
                    doc_text = f"{fp.stem} {content[:500]}"
                    tf, dl = _precompute_tf(doc_text)
                    summary = self._extract_summary(content, fp.stem)
                    corpus.append((tf, dl, fp, mtime_iso, rel_path, rel_path, summary, 0, content, ""))
            except Exception:
                continue

        with self._corpus_lock:
            self._corpus_cache = (mtime_key, corpus)
        return corpus

    def search(self, query: str, top_k: int = 5) -> List[VFSItem]:
        """BM25 全文搜索（段落级分块索引，iter165+166 语料库 + TF/meta 预计算缓存）

        iter165 变更：
          v1: 缓存文档文本（减少文件 I/O）
          v2: 缓存预计算 TF dict（消除 doc tokenization，剩余只有 query tokenize）
        iter166 变更：
          corpus tuple 含预计算 mtime_iso/rel_path/item_id/summary，
          _section_to_item 跳过 stat() + sha256 + summary 提取。

        OS 类比：Linux page cache — 首次访问填充页缓存，后续命中无需磁盘 I/O；
          TF 预计算类比于 CPU 的 BTB (Branch Target Buffer) —
          分支目标地址预存储，下次执行无需重新解码。
        """
        corpus = self._get_corpus()
        if not corpus:
            return []

        # 只 tokenize query 一次（corpus hit 后 doc 不需要再 tokenize）
        q_terms = _tokenize(query)
        if not q_terms:
            return []

        # BM25 评分（query-dependent，不可缓存；doc TF 已预计算）
        # iter166: corpus tuple = (tf, dl, fp, mtime_iso, rel_path, item_id, summary, idx, body, header)
        scored = []
        for entry in corpus:
            tf, dl = entry[0], entry[1]
            score = _bm25_score_tf(q_terms, tf, dl)
            if score >= self.min_score:
                scored.append((score,) + entry[2:])  # (score, fp, mtime_iso, rel_path, item_id, summary, idx, body, header)

        # 按分数降序，去重（同一文件最多取 2 个不同章节进入 top_k）
        scored.sort(key=lambda x: x[0], reverse=True)
        items = []
        file_count: Dict[Path, int] = {}
        for entry in scored:
            if len(items) >= top_k:
                break
            score, fp = entry[0], entry[1]
            file_count[fp] = file_count.get(fp, 0) + 1
            if file_count[fp] > 2:
                continue  # 同文件最多 2 个章节，避免大文件独占结果
            try:
                mtime_iso, rel_path, item_id, summary, idx, body, header = entry[2:]
                item = self._section_to_item_fast(
                    score=score, mtime_iso=mtime_iso, rel_path=rel_path,
                    item_id=item_id, summary=summary, body=body)
                if item:
                    items.append(item)
            except Exception:
                continue

        return items

    def _split_sections(self, content: str) -> List[Tuple[str, str]]:
        """按 Markdown 标题分块，返回 [(header, body)] 列表。
        如果没有标题，整文件作为一个段落。
        """
        import re
        parts = re.split(r'^(?=#{1,3} )', content, flags=re.MULTILINE)
        sections = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            lines = part.splitlines()
            if not lines:
                continue
            header_line = lines[0].lstrip('#').strip() if lines[0].startswith('#') else ""
            body = '\n'.join(lines[1:]).strip() if len(lines) > 1 else part
            sections.append((header_line, body))
        return sections if sections else [("", content)]

    def _section_to_item(self, filepath: Path, score: float,
                          section_idx: int, section_text: str,
                          section_header: str) -> Optional[VFSItem]:
        """将文件的某个章节转换为 VFSItem。"""
        try:
            # 摘要：优先用章节标题，fallback 到正文首句
            if section_header and len(section_header) >= 5:
                summary = section_header[:120]
            else:
                summary = self._extract_summary(section_text, filepath.stem)

            content_to_use = section_text if section_text else ""
            content_hash = hashlib.sha256(content_to_use.encode("utf-8", errors="ignore")).hexdigest()

            mtime = filepath.stat().st_mtime
            mtime_iso = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
            rel_path = str(filepath.relative_to(self.base_dir))

            # 段落 ID：文件路径 + 章节索引
            item_id = f"{rel_path}#{section_idx}" if section_idx > 0 else rel_path

            return VFSItem(
                id=item_id,
                type="rule",
                source=self._source,
                content=content_to_use[:2048],  # 限制内容大小
                summary=summary,
                metadata=VFSMetadata(
                    created_at=mtime_iso,
                    updated_at=mtime_iso,
                    last_accessed=datetime.now(timezone.utc).isoformat(),
                    importance=0.5,
                    retrievability=score,
                    access_count=0,
                    source_session="",
                    scope="global",
                    tags=[self._source],
                    project="",
                    content_hash=content_hash,
                ),
                path=f"/{self._source}/{rel_path}",
                score=score,
            )
        except Exception:
            return None

    def _section_to_item_fast(self, score: float, mtime_iso: str, rel_path: str,
                               item_id: str, summary: str, body: str) -> Optional[VFSItem]:
        """iter166: 使用 corpus 预计算的 meta 构建 VFSItem（跳过 stat + sha256 + summary）。

        OS 类比：inode cache hit — 直接读缓存的 inode 属性，无需走磁盘。
        相比 _section_to_item：跳过 filepath.stat()（1 syscall）+ sha256（hash 计算）+ summary 提取。
        """
        try:
            content_to_use = body if body else ""
            return VFSItem(
                id=item_id,
                type="rule",
                source=self._source,
                content=content_to_use[:2048],
                summary=summary,
                metadata=VFSMetadata(
                    created_at=mtime_iso,
                    updated_at=mtime_iso,
                    last_accessed=datetime.now(timezone.utc).isoformat(),
                    importance=0.5,
                    retrievability=score,
                    access_count=0,
                    source_session="",
                    scope="global",
                    tags=[self._source],
                    project="",
                    content_hash="",  # iter166: skip sha256（search 路径不需要）
                ),
                path=f"/{self._source}/{rel_path}",
                score=score,
            )
        except Exception:
            return None

    def write(self, item: VFSItem) -> bool:
        """只读后端，不支持写入"""
        return False

    def delete(self, path: str) -> bool:
        """只读后端，不支持删除"""
        return False

    def _file_to_item(self, filepath: Path, score: float) -> Optional[VFSItem]:
        """将文件转换为 VFSItem"""
        try:
            content_bytes = filepath.read_bytes()[:self.max_content_bytes]
            content = content_bytes.decode("utf-8", errors="ignore")

            # 提取摘要（跳过 frontmatter，取第一个有意义行）
            summary = self._extract_summary(content, filepath.stem)
            content_hash = hashlib.sha256(content_bytes).hexdigest()

            mtime = filepath.stat().st_mtime
            mtime_iso = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
            rel_path = str(filepath.relative_to(self.base_dir))

            return VFSItem(
                id=rel_path,
                type="rule",
                source=self._source,
                content=content,
                summary=summary,
                metadata=VFSMetadata(
                    created_at=mtime_iso,
                    updated_at=mtime_iso,
                    last_accessed=datetime.now(timezone.utc).isoformat(),
                    importance=0.5,
                    retrievability=score,
                    access_count=0,
                    source_session="",
                    scope="global",
                    tags=[self._source],
                    project="",
                    content_hash=content_hash,
                ),
                path=f"/{self._source}/{rel_path}",
                score=score,
            )
        except Exception:
            return None

    @staticmethod
    def _extract_summary(content: str, fallback: str) -> str:
        """从 Markdown 内容提取摘要（跳过 frontmatter）"""
        in_frontmatter = False
        frontmatter_count = 0

        for line in content.splitlines():
            stripped = line.strip()

            # 跳过 frontmatter
            if stripped == "---" and frontmatter_count < 2:
                in_frontmatter = not in_frontmatter
                frontmatter_count += 1
                continue

            if in_frontmatter:
                continue

            # 跳过注释和纯符号行
            if stripped.startswith("<!--") or re.match(r"^[-=*`#>]{2,}$", stripped):
                continue

            # 跳过标题（但取标题文本）
            if stripped.startswith("#"):
                text = stripped.lstrip("#").strip()
                if len(text) >= 5:
                    return text[:120]
                continue

            if len(stripped) >= 10:
                return stripped[:120]

        return fallback[:120]


# ── 具体后端实现 ────────────────────────────────────────────────────────────
class SelfImprovingBackend(FilesystemBackendBase):
    """~/self-improving/ 文件系统后端

    挂载点：/self-improving/<path>
    来源权重：0.7（在 KnowledgeVFS 中配置）
    """

    def __init__(self, base_dir: Optional[Path] = None):
        super().__init__(
            base_dir=base_dir or (Path.home() / "self-improving"),
            source=VFSSource.SELF_IMPROVING.value,
            min_score=0.05,
        )

    @property
    def name(self) -> str:
        return "SelfImprovingBackend"

    @property
    def source_type(self) -> str:
        return VFSSource.SELF_IMPROVING.value


class MemoryMdBackend(FilesystemBackendBase):
    """~/.claude/projects/*/memory/ 目录中的记忆文件后端

    挂载点：/memory-md/<path>
    扫描：~/.claude/projects/*/memory/*.md（用户手写的跨会话记忆）
    来源权重：0.85（比 self-improving 高，因为是用户直接写的记忆）

    注意：只扫描 projects/*/memory/ 子目录，避免扫描 ~/.claude/ 全部 2000+ 文件。
    """

    def __init__(self, base_dir: Optional[Path] = None):
        # base_dir 指向 ~/.claude/projects/ 用于 glob 扫描
        # 但 FilesystemBackendBase 的 read() 从 base_dir/<rel_path> 读取
        super().__init__(
            base_dir=base_dir or (Path.home() / ".claude" / "projects"),
            source="memory-md",
            min_score=0.05,
        )

    def _glob_candidates(self) -> List[Path]:
        """只扫描 projects/*/memory/*.md，避免全量扫描。"""
        return sorted(self.base_dir.glob("*/memory/*.md"))

    def _watch_dirs(self) -> List[Path]:
        """iter177: 监视 projects/*/memory/ 目录（7 dirs, stat=0.03ms）。
        glob 模式 `*/memory/*.md` — 新增文件只在 memory/ 目录下，
        monitor memory/ dirs 即可检测增删，无需扫描 projects/ 全部子目录（313 dirs）。
        iter177: 结果永久缓存（_watch_dirs_memo）— daemon 60s 内不会新增 project 目录。
        OS 类比：inotify IN_CREATE/IN_DELETE — 只在目标目录注册事件，不递归。
        """
        if self._watch_dirs_memo:
            return self._watch_dirs_memo
        dirs = []
        try:
            for project_dir in self.base_dir.iterdir():
                if project_dir.is_dir():
                    memory_dir = project_dir / "memory"
                    if memory_dir.exists():
                        dirs.append(memory_dir)
        except Exception:
            pass
        result = dirs if dirs else [self.base_dir]
        self._watch_dirs_memo = result
        return result

    @property
    def name(self) -> str:
        return "MemoryMdBackend"

    @property
    def source_type(self) -> str:
        return "memory-md"


# ── 测试 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print("Testing FilesystemBackend...")

    # Test SelfImprovingBackend
    si = SelfImprovingBackend()
    if si.base_dir.exists():
        results = si.search("memory", top_k=3)
        print(f"✓ SelfImproving search: {len(results)} results")
        for r in results[:2]:
            print(f"  - [{r.score:.3f}] {r.summary[:60]}")
    else:
        print(f"⚠ self-improving dir not found: {si.base_dir}")

    # Test MemoryMdBackend
    mm = MemoryMdBackend()
    results2 = mm.search("BM25", top_k=3)
    print(f"✓ MemoryMd search: {len(results2)} results")
    for r in results2[:2]:
        print(f"  - [{r.score:.3f}] {r.summary[:60]}")

    print("\n✅ FilesystemBackend verified")
