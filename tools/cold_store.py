#!/usr/bin/env python3
"""
cold_store.py — memory-os L4→L5 冷备份：store.db → mm (MetaMemory) API

设计对应 OS 原理：swap-out（内存页写回磁盘），将高价值 chunk 持久化到
跨 session 可搜索的外部知识系统。

功能：
  1. sync   — 将 store.db 中未同步的高价值 chunk 推送到 mm
  2. pull   — 从 mm 拉取本项目相关文档，补入 store.db（swap-in）
  3. status — 显示同步状态

用法：
  python3 cold_store.py sync              # 同步到 mm
  python3 cold_store.py sync --dry        # 预览，不写
  python3 cold_store.py pull              # 从 mm 拉取
  python3 cold_store.py status            # 查看状态
"""
import sys
import os
import json
import sqlite3
import subprocess
import hashlib
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
from utils import resolve_project_id

MEMORY_OS_DIR = Path.home() / ".claude" / "memory-os"
STORE_DB = MEMORY_OS_DIR / "store.db"
SYNC_STATE_FILE = MEMORY_OS_DIR / "cold_sync_state.json"

# 只同步高价值 chunk_type
SYNC_TYPES = {"decision", "reasoning_chain", "excluded_path"}
# importance 最低阈值（过低的不值得持久化）
MIN_IMPORTANCE = 0.65

# mm folder：冷备份归类
MM_FOLDER_NAME = "memory-os-cold"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_sync_state() -> dict:
    """返回 {chunk_id: {mm_doc_id, synced_at, content_hash}}"""
    if SYNC_STATE_FILE.exists():
        try:
            return json.loads(SYNC_STATE_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_sync_state(state: dict) -> None:
    MEMORY_OS_DIR.mkdir(parents=True, exist_ok=True)
    SYNC_STATE_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2)
    )


def _content_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


def _mm_cmd(args: list, input_data: str = None) -> tuple:
    """执行 mm CLI，返回 (success, output_dict_or_str)"""
    try:
        cmd = ["mm"] + args
        result = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=15,
            input=input_data,
        )
        output = result.stdout.strip()
        if result.returncode != 0:
            return False, result.stderr.strip() or output
        try:
            return True, json.loads(output)
        except Exception:
            return True, output
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except FileNotFoundError:
        return False, "mm CLI not found"
    except Exception as e:
        return False, str(e)


def _ensure_folder() -> str:
    """确保 mm 中存在 memory-os-cold 文件夹，返回 folder_id。"""
    ok, data = _mm_cmd(["folders"])
    if not ok:
        return ""

    # 递归搜索已有文件夹
    def _find_folder(node, name):
        if node.get("name") == name:
            return node["id"]
        for child in node.get("children", []):
            found = _find_folder(child, name)
            if found:
                return found
        return None

    folder_id = _find_folder(data, MM_FOLDER_NAME)
    if folder_id:
        return folder_id

    # 创建文件夹
    ok, result = _mm_cmd(["mkdir", MM_FOLDER_NAME])
    if ok and isinstance(result, dict):
        return result.get("id", "")
    return ""


def _get_syncable_chunks() -> list:
    """从 store.db 取出可同步的高价值 chunk。"""
    if not STORE_DB.exists():
        return []
    conn = sqlite3.connect(str(STORE_DB))
    placeholders = ",".join(f"'{t}'" for t in SYNC_TYPES)
    rows = conn.execute(f"""
        SELECT id, chunk_type, summary, content, importance, project,
               created_at, source_session, tags
        FROM memory_chunks
        WHERE chunk_type IN ({placeholders})
          AND importance >= ?
          AND summary != ''
        ORDER BY importance DESC, created_at DESC
    """, (MIN_IMPORTANCE,)).fetchall()
    conn.close()

    return [
        {
            "id": r[0], "chunk_type": r[1], "summary": r[2],
            "content": r[3], "importance": r[4], "project": r[5],
            "created_at": r[6], "source_session": r[7],
            "tags": r[8],
        }
        for r in rows
    ]


def _format_mm_content(chunk: dict) -> str:
    """构造 mm 文档内容（markdown 格式）。"""
    lines = [
        f"# [{chunk['chunk_type']}] {chunk['summary']}",
        "",
        f"**类型**: {chunk['chunk_type']}",
        f"**重要度**: {chunk['importance']}",
        f"**项目**: {chunk['project']}",
        f"**创建时间**: {chunk['created_at']}",
        f"**来源会话**: {chunk['source_session'][:8] if chunk['source_session'] else 'unknown'}",
        "",
        "## 内容",
        "",
        chunk["content"] or chunk["summary"],
    ]
    return "\n".join(lines)


# ── sync ─────────────────────────────────────────────────────

def cmd_sync(dry_run: bool = False):
    """将 store.db 高价值 chunk 同步到 mm。"""
    chunks = _get_syncable_chunks()
    if not chunks:
        print("没有可同步的 chunk")
        return

    sync_state = _load_sync_state()
    folder_id = "" if dry_run else _ensure_folder()
    if not dry_run and not folder_id:
        print("错误：无法创建/获取 mm 文件夹")
        return

    created, updated, skipped = 0, 0, 0

    for chunk in chunks:
        cid = chunk["id"]
        content = _format_mm_content(chunk)
        ch = _content_hash(content)

        existing = sync_state.get(cid)
        if existing and existing.get("content_hash") == ch:
            skipped += 1
            continue

        title = f"[{chunk['chunk_type']}] {chunk['summary'][:80]}"
        tags = chunk.get("tags", "")
        if isinstance(tags, str):
            try:
                tag_list = json.loads(tags)
            except Exception:
                tag_list = [tags] if tags else []
        else:
            tag_list = tags or []
        tags_str = ",".join(str(t) for t in tag_list) if tag_list else chunk["chunk_type"]

        if dry_run:
            action = "UPDATE" if existing else "CREATE"
            print(f"  [{action}] {title}")
            if existing:
                updated += 1
            else:
                created += 1
            continue

        if existing and existing.get("mm_doc_id"):
            # 更新已有文档
            ok, result = _mm_cmd(
                ["update", existing["mm_doc_id"], "--tags", tags_str],
                input_data=content,
            )
            if ok:
                sync_state[cid] = {
                    "mm_doc_id": existing["mm_doc_id"],
                    "synced_at": _now_iso(),
                    "content_hash": ch,
                }
                updated += 1
            else:
                print(f"  更新失败 [{cid[:8]}]: {result}")
        else:
            # 创建新文档
            ok, result = _mm_cmd(
                ["create", title, "--folder", folder_id, "--tags", tags_str],
                input_data=content,
            )
            if ok and isinstance(result, dict):
                mm_doc_id = result.get("id", "")
                sync_state[cid] = {
                    "mm_doc_id": mm_doc_id,
                    "synced_at": _now_iso(),
                    "content_hash": ch,
                }
                created += 1
            else:
                print(f"  创建失败 [{cid[:8]}]: {result}")

    if not dry_run:
        _save_sync_state(sync_state)

    tag = "[dry-run] " if dry_run else ""
    print(f"\n{tag}同步完成: 创建={created}, 更新={updated}, 跳过={skipped}")


# ── pull ─────────────────────────────────────────────────────

def cmd_pull():
    """从 mm 拉取本项目相关文档，补入 store.db（swap-in）。"""
    project = resolve_project_id()
    ok, results = _mm_cmd(["search", project])
    if not ok:
        print(f"mm 搜索失败: {results}")
        return

    docs = results if isinstance(results, list) else results.get("results", [])
    if not docs:
        print("mm 中无匹配文档")
        return

    if not STORE_DB.exists():
        print("store.db 不存在，跳过")
        return

    conn = sqlite3.connect(str(STORE_DB))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_chunks (
            id TEXT PRIMARY KEY, created_at TEXT, updated_at TEXT,
            project TEXT, source_session TEXT, chunk_type TEXT,
            content TEXT, summary TEXT, tags TEXT,
            importance REAL, retrievability REAL,
            last_accessed TEXT, feishu_url TEXT
        )
    """)

    imported = 0
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        title = doc.get("title", "")
        content = doc.get("content", "") or doc.get("snippet", "")
        if not title or not content:
            continue

        # 解析 chunk_type
        chunk_type = "decision"
        for ct in SYNC_TYPES:
            if f"[{ct}]" in title:
                chunk_type = ct
                break

        summary = title.replace(f"[{chunk_type}]", "").strip()[:120]

        # 按 summary 去重
        exists = conn.execute(
            "SELECT id FROM memory_chunks WHERE summary=?", (summary,)
        ).fetchone()
        if exists:
            continue

        import uuid
        chunk_id = str(uuid.uuid4())
        now = _now_iso()
        conn.execute("""
            INSERT INTO memory_chunks
            (id, created_at, updated_at, project, source_session,
             chunk_type, content, summary, tags, importance,
             retrievability, last_accessed, feishu_url)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            chunk_id, now, now, project, "cold_pull",
            chunk_type, content, summary,
            json.dumps([chunk_type, project, "cold_pull"]),
            0.75, 0.5, now, None,
        ))
        imported += 1

    conn.commit()
    conn.close()
    print(f"从 mm 导入 {imported} 个文档到 store.db")


# ── status ───────────────────────────────────────────────────

def cmd_status():
    """显示同步状态。"""
    chunks = _get_syncable_chunks()
    sync_state = _load_sync_state()

    synced = sum(1 for c in chunks if c["id"] in sync_state)
    unsynced = len(chunks) - synced

    print(f"可同步 chunk: {len(chunks)}")
    print(f"  已同步: {synced}")
    print(f"  未同步: {unsynced}")
    print(f"  同步类型: {', '.join(sorted(SYNC_TYPES))}")
    print(f"  最低 importance: {MIN_IMPORTANCE}")

    # mm 健康检查
    ok, data = _mm_cmd(["health"])
    if ok and isinstance(data, dict):
        print(f"\nmm 服务: {data.get('status', 'unknown')}")
        print(f"  文档数: {data.get('document_count', '?')}")
    else:
        print(f"\nmm 服务: 不可用 ({data})")

    if unsynced > 0:
        print(f"\n未同步列表:")
        for c in chunks:
            if c["id"] not in sync_state:
                print(f"  [{c['chunk_type']}] {c['summary'][:60]}")


# ── main ─────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]
    cmd = args[0] if args else "status"
    dry_run = "--dry" in args

    if cmd == "sync":
        cmd_sync(dry_run=dry_run)
    elif cmd == "pull":
        cmd_pull()
    elif cmd == "status":
        cmd_status()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
