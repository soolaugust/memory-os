#!/usr/bin/env python3
"""
KnowledgeVFS Backends — 三套后端的适配器实现

实现 VFSBackend 接口的三个具体类：
  1. SQLiteBackend — memory-os store.db（L1 主存储）
  2. FilesystemBackend — self-improving/（L3 知识库）
  3. ProjectBackend — 项目级 JSONL 存储（L4 历史）

OS 类比：ext4、NFS、FUSE 等具体文件系统对标 super_block 和 inode_operations。
"""

import sys
import json
import sqlite3
import time
import re
import hashlib
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime, timezone

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from knowledge_vfs import (
    VFSBackend, VFSItem, VFSItemType, VFSSource, VFSMetadata, VFSCache
)
from config import get as _sysctl
from bm25 import bm25_normalized as _bm25_norm
from store import open_db, ensure_schema, fts_search as _fts_search


# ─────────────────────────────────────────────────────────────
# SQLiteBackend — memory-os 存储
# ─────────────────────────────────────────────────────────────

class SQLiteBackend(VFSBackend):
    """
    SQLite 后端：连接 memory-os store.db
    支持 FTS5 全文搜索 + BM25 排序
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or (
            Path.home() / ".claude" / "memory-os" / "store.db"
        )
        self.cache = VFSCache()
        self._project_id: Optional[str] = None

    def name(self) -> str:
        return "memory-os"

    def _get_project_id(self) -> str:
        """自动推导 project_id（缓存）"""
        if self._project_id is None:
            from utils import resolve_project_id
            self._project_id = resolve_project_id()
        return self._project_id

    def read(self, path: str, recursive: bool = False) -> List[VFSItem]:
        """
        按路径读取 chunk
        路径格式：/memory-os/<chunk-id>
        """
        if not self.db_path.exists():
            return []

        # 解析路径
        parts = path.strip("/").split("/", 1)
        if len(parts) < 2:
            return []

        chunk_id = parts[1]
        project = self._get_project_id()

        try:
            conn = open_db()
            ensure_schema(conn)

            # 按 ID 读取
            row = conn.execute(
                "SELECT id, summary, content, chunk_type, importance, created_at, updated_at "
                "FROM memory_chunks WHERE id=? AND project=?",
                (chunk_id, project)
            ).fetchone()

            if not row:
                return []

            metadata = VFSMetadata(
                created_at=row[5],
                updated_at=row[6],
                importance=row[4] or 0,
                scope="session",
                source=path,
                tags=[],
                retrievability=1.0,
                mtime=time.time(),
                hash=hashlib.sha256(row[2].encode()).hexdigest()
            )

            item = VFSItem(
                id=row[0],
                type=VFSItemType(row[3]) if row[3] else VFSItemType.TRACE,
                content=row[2],
                summary=row[1][:120],
                source=VFSSource.MEMORY_OS,
                metadata=metadata,
                score=1.0,
                path=path
            )
            return [item]

        except Exception as e:
            print(f"SQLiteBackend.read error: {e}")
            return []
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def search(self, query: str, top_k: int = 3, timeout_ms: int = 100) -> List[VFSItem]:
        """全文搜索（FTS5 + BM25）"""
        if not self.db_path.exists():
            return []

        project = self._get_project_id()
        start_time = time.time()

        try:
            conn = open_db()
            ensure_schema(conn)

            # 主路径：FTS5 索引
            try:
                fts_results = _fts_search(conn, query, project, top_k=top_k)
            except Exception:
                fts_results = []

            if fts_results:
                items = []
                max_rank = max(
                    (r["fts_rank"] for r in fts_results), default=1.0
                )
                if max_rank <= 0:
                    max_rank = 1.0

                for r in fts_results:
                    score = r["fts_rank"] / max_rank
                    if score >= _sysctl("router.min_score"):
                        metadata = VFSMetadata(
                            created_at=r.get("created_at", ""),
                            updated_at=r.get("updated_at", ""),
                            importance=0,
                            scope="session",
                            source=f"/memory-os/{r['id']}",
                            retrievability=score,
                            mtime=time.time(),
                            hash=hashlib.sha256(r["content"].encode()).hexdigest()
                        )
                        item = VFSItem(
                            id=r["id"],
                            type=VFSItemType(r["chunk_type"]),
                            content=r["content"],
                            summary=r["summary"],
                            source=VFSSource.MEMORY_OS,
                            metadata=metadata,
                            score=score,
                            path=f"/memory-os/{r['id']}"
                        )
                        items.append(item)

                items.sort(key=lambda x: x.score, reverse=True)
                return items[:top_k]

            # Fallback：全表扫描 + Python BM25
            rows = conn.execute(
                "SELECT id, summary, content, chunk_type, importance, created_at, updated_at "
                "FROM memory_chunks WHERE project=? AND summary!=''",
                (project,)
            ).fetchall()

            if not rows:
                return []

            docs = [f"{r[1]} {r[2]}" for r in rows]
            scores = _bm25_norm(query, docs)
            items = []

            for i, row in enumerate(rows):
                if scores[i] >= _sysctl("router.min_score"):
                    metadata = VFSMetadata(
                        created_at=row[5],
                        updated_at=row[6],
                        importance=row[4] or 0,
                        scope="session",
                        source=f"/memory-os/{row[0]}",
                        retrievability=scores[i],
                        mtime=time.time(),
                        hash=hashlib.sha256(row[2].encode()).hexdigest()
                    )
                    item = VFSItem(
                        id=row[0],
                        type=VFSItemType(row[3]),
                        content=row[2],
                        summary=row[1][:120],
                        source=VFSSource.MEMORY_OS,
                        metadata=metadata,
                        score=scores[i],
                        path=f"/memory-os/{row[0]}"
                    )
                    items.append(item)

            items.sort(key=lambda x: x.score, reverse=True)
            return items[:top_k]

        except Exception as e:
            print(f"SQLiteBackend.search error: {e}")
            return []
        finally:
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > timeout_ms:
                print(f"SQLiteBackend.search exceeded {timeout_ms}ms: {elapsed_ms:.1f}ms")
            try:
                conn.close()
            except Exception:
                pass

    def write(self, item: VFSItem) -> str:
        """写入项目到数据库"""
        project = self._get_project_id()

        try:
            conn = open_db()
            ensure_schema(conn)

            # 简化版本：仅插入，生成新 ID
            import uuid
            new_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc).isoformat()

            conn.execute(
                "INSERT INTO memory_chunks (id, project, summary, content, chunk_type, importance, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    new_id, project,
                    item.summary[:120],
                    item.content,
                    item.type.value,
                    item.metadata.importance if item.metadata else 0,
                    now, now
                )
            )
            conn.commit()
            return new_id

        except Exception as e:
            print(f"SQLiteBackend.write error: {e}")
            return ""
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def delete(self, item_id: str, force: bool = False) -> bool:
        """删除项目"""
        project = self._get_project_id()

        try:
            conn = open_db()
            conn.execute(
                "DELETE FROM memory_chunks WHERE id=? AND project=?",
                (item_id, project)
            )
            conn.commit()
            return True
        except Exception as e:
            print(f"SQLiteBackend.delete error: {e}")
            return False
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def invalidate_cache(self):
        """失效缓存"""
        self.cache.invalidate("all")


# ─────────────────────────────────────────────────────────────
# FilesystemBackend — self-improving 知识库
# ─────────────────────────────────────────────────────────────

class FilesystemBackend(VFSBackend):
    """
    文件系统后端：遍历 ~/self-improving/
    支持 .md 文件的 BM25 搜索
    """

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or (Path.home() / "self-improving")
        self.cache = VFSCache()

    def name(self) -> str:
        return "self-improving"

    def _clean_md_lines(self, lines: list) -> list:
        """清洁 markdown 行"""
        result = []
        in_frontmatter = False
        frontmatter_count = 0

        for line in lines:
            stripped = line.strip()

            if stripped == "---" and frontmatter_count < 2:
                in_frontmatter = not in_frontmatter
                frontmatter_count += 1
                continue

            if in_frontmatter or stripped.startswith("<!--") or re.match(r'^[-=*`#>]{2,}$', stripped):
                continue

            if len(stripped) >= 6:
                result.append(stripped)

        return result

    def read(self, path: str, recursive: bool = False) -> List[VFSItem]:
        """
        读取文件
        路径格式：/self-improving/filepath
        """
        parts = path.strip("/").split("/", 1)
        if len(parts) < 2:
            return []

        filepath = Path(self.base_dir) / parts[1]

        if not filepath.exists() or not filepath.is_file():
            return []

        try:
            content = filepath.read_text(encoding="utf-8")
            lines = self._clean_md_lines(content.splitlines())
            summary = next(
                (l for l in lines if not l.startswith("#")), lines[0] if lines else ""
            )

            mtime = filepath.stat().st_mtime
            content_hash = hashlib.sha256(content.encode()).hexdigest()

            metadata = VFSMetadata(
                created_at=datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat(),
                updated_at=datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat(),
                importance=0,
                scope="global",
                source=path,
                tags=["self-improving"],
                retrievability=1.0,
                mtime=mtime,
                hash=content_hash
            )

            item = VFSItem(
                id=str(filepath.relative_to(self.base_dir)),
                type=VFSItemType.RULE,
                content=content,
                summary=summary[:120],
                source=VFSSource.SELF_IMPROVING,
                metadata=metadata,
                score=1.0,
                path=path
            )
            return [item]

        except Exception as e:
            print(f"FilesystemBackend.read error: {e}")
            return []

    def search(self, query: str, top_k: int = 3, timeout_ms: int = 100) -> List[VFSItem]:
        """全文搜索"""
        if not self.base_dir.exists():
            return []

        start_time = time.time()

        try:
            # 扫描所有 .md 文件
            candidates = list(self.base_dir.glob("**/*.md"))
            docs = []
            file_map = []
            summaries = []

            for fp in candidates:
                try:
                    content = fp.read_text(encoding="utf-8")
                    lines = self._clean_md_lines(content.splitlines())
                    if not lines:
                        continue

                    doc_text = f"{fp.stem} {' '.join(lines[:10])}"
                    docs.append(doc_text)
                    file_map.append(fp)

                    summary_line = next(
                        (l for l in lines if not l.startswith("#")), lines[0]
                    )
                    summaries.append(summary_line[:100])

                except Exception:
                    pass

            if not docs:
                return []

            # BM25 排序
            scores = _bm25_norm(query, docs)
            items = []

            for i, fp in enumerate(file_map):
                if scores[i] >= _sysctl("router.min_score"):
                    try:
                        content = fp.read_text(encoding="utf-8")
                        mtime = fp.stat().st_mtime
                        content_hash = hashlib.sha256(content.encode()).hexdigest()

                        metadata = VFSMetadata(
                            created_at=datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat(),
                            updated_at=datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat(),
                            importance=0,
                            scope="global",
                            source=str(fp.relative_to(self.base_dir)),
                            tags=["self-improving"],
                            retrievability=scores[i],
                            mtime=mtime,
                            hash=content_hash
                        )

                        rel_path = str(fp.relative_to(self.base_dir))
                        item = VFSItem(
                            id=rel_path,
                            type=VFSItemType.RULE,
                            content=content,
                            summary=summaries[i],
                            source=VFSSource.SELF_IMPROVING,
                            metadata=metadata,
                            score=scores[i],
                            path=f"/self-improving/{rel_path}"
                        )
                        items.append(item)
                    except Exception:
                        pass

            items.sort(key=lambda x: x.score, reverse=True)
            return items[:top_k]

        except Exception as e:
            print(f"FilesystemBackend.search error: {e}")
            return []
        finally:
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > timeout_ms:
                print(f"FilesystemBackend.search exceeded {timeout_ms}ms")

    def write(self, item: VFSItem) -> str:
        """写入文件"""
        filename = f"vfs_{int(time.time())}.md"
        filepath = self.base_dir / filename

        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            filepath.write_text(item.content, encoding="utf-8")
            return str(filepath.relative_to(self.base_dir))
        except Exception as e:
            print(f"FilesystemBackend.write error: {e}")
            return ""

    def delete(self, item_id: str, force: bool = False) -> bool:
        """删除文件"""
        filepath = self.base_dir / item_id

        try:
            if filepath.exists():
                filepath.unlink()
                return True
            return False
        except Exception as e:
            print(f"FilesystemBackend.delete error: {e}")
            return False

    def invalidate_cache(self):
        """失效缓存"""
        self.cache.invalidate("all")


# ─────────────────────────────────────────────────────────────
# ProjectBackend — 项目级历史存储
# ─────────────────────────────────────────────────────────────

class ProjectBackend(VFSBackend):
    """
    项目后端：.claude/projects/<project-id>/memory/
    支持 JSONL 和 .md 混合存储
    """

    def __init__(self, project_id: Optional[str] = None):
        from utils import resolve_project_id
        self.project_id = project_id or resolve_project_id()
        self.base_dir = (
            Path.home() / ".claude" / "projects" / self.project_id / "memory"
        )
        self.cache = VFSCache()

    def name(self) -> str:
        return "project"

    def read(self, path: str, recursive: bool = False) -> List[VFSItem]:
        """读取项目项"""
        parts = path.strip("/").split("/", 1)
        if len(parts) < 2:
            return []

        item_id = parts[1]
        jsonl_file = self.base_dir / "history.jsonl"

        if not jsonl_file.exists():
            return []

        try:
            for line in jsonl_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                data = json.loads(line)
                if data.get("id") == item_id:
                    metadata = VFSMetadata(
                        created_at=data.get("created_at", ""),
                        updated_at=data.get("updated_at", ""),
                        importance=data.get("importance", 0),
                        scope="project",
                        source=path,
                        tags=data.get("tags", []),
                        retrievability=1.0,
                        mtime=time.time(),
                        hash=data.get("hash", "")
                    )

                    item = VFSItem(
                        id=item_id,
                        type=VFSItemType(data.get("type", "trace")),
                        content=data.get("content", ""),
                        summary=data.get("summary", ""),
                        source=VFSSource.PROJECT,
                        metadata=metadata,
                        score=1.0,
                        path=path
                    )
                    return [item]

            return []

        except Exception as e:
            print(f"ProjectBackend.read error: {e}")
            return []

    def search(self, query: str, top_k: int = 3, timeout_ms: int = 100) -> List[VFSItem]:
        """搜索项目项"""
        jsonl_file = self.base_dir / "history.jsonl"

        if not jsonl_file.exists():
            return []

        start_time = time.time()

        try:
            items = []
            docs = []
            data_list = []

            for line in jsonl_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                data = json.loads(line)
                doc = f"{data.get('summary', '')} {data.get('content', '')}"
                docs.append(doc)
                data_list.append(data)

            if not docs:
                return []

            scores = _bm25_norm(query, docs)

            for i, data in enumerate(data_list):
                if scores[i] >= _sysctl("router.min_score"):
                    metadata = VFSMetadata(
                        created_at=data.get("created_at", ""),
                        updated_at=data.get("updated_at", ""),
                        importance=data.get("importance", 0),
                        scope="project",
                        source=f"/project/{data.get('id')}",
                        tags=data.get("tags", []),
                        retrievability=scores[i],
                        mtime=time.time(),
                        hash=data.get("hash", "")
                    )

                    item = VFSItem(
                        id=data.get("id", ""),
                        type=VFSItemType(data.get("type", "trace")),
                        content=data.get("content", ""),
                        summary=data.get("summary", ""),
                        source=VFSSource.PROJECT,
                        metadata=metadata,
                        score=scores[i],
                        path=f"/project/{data.get('id')}"
                    )
                    items.append(item)

            items.sort(key=lambda x: x.score, reverse=True)
            return items[:top_k]

        except Exception as e:
            print(f"ProjectBackend.search error: {e}")
            return []
        finally:
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > timeout_ms:
                print(f"ProjectBackend.search exceeded {timeout_ms}ms")

    def write(self, item: VFSItem) -> str:
        """追加写入项目"""
        import uuid
        jsonl_file = self.base_dir / "history.jsonl"

        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)

            new_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc).isoformat()

            record = {
                "id": new_id,
                "type": item.type.value,
                "summary": item.summary,
                "content": item.content,
                "importance": item.metadata.importance if item.metadata else 0,
                "scope": item.metadata.scope if item.metadata else "project",
                "tags": item.metadata.tags if item.metadata else [],
                "created_at": now,
                "updated_at": now,
                "hash": hashlib.sha256(item.content.encode()).hexdigest(),
            }

            with open(jsonl_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            return new_id

        except Exception as e:
            print(f"ProjectBackend.write error: {e}")
            return ""

    def delete(self, item_id: str, force: bool = False) -> bool:
        """标记删除项目（在 JSONL 中标记）"""
        jsonl_file = self.base_dir / "history.jsonl"

        if not jsonl_file.exists():
            return False

        try:
            lines = jsonl_file.read_text(encoding="utf-8").splitlines()
            updated_lines = []

            for line in lines:
                if not line.strip():
                    continue
                data = json.loads(line)
                if data.get("id") == item_id:
                    data["deleted_at"] = datetime.now(timezone.utc).isoformat()
                updated_lines.append(json.dumps(data, ensure_ascii=False))

            jsonl_file.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")
            return True

        except Exception as e:
            print(f"ProjectBackend.delete error: {e}")
            return False

    def invalidate_cache(self):
        """失效缓存"""
        self.cache.invalidate("all")


if __name__ == "__main__":
    print("KnowledgeVFS Backends module loaded")
