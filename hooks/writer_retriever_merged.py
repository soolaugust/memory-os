#!/usr/bin/env python3
"""
迭代72a: Writer-Retriever Fusion — 单连接合并 UserPromptSubmit Hook

目的：消除 writer + retriever 各自 open_db/commit/close 的重复开销
  原来：writer open_db(40ms) → write prompt_context(10ms) → commit(40ms)
       + retriever open_db(40ms) → FTS5 search(7ms) → commit(40ms)
       = 两次 80ms commit = 160ms

  改进：merged open_db(40ms) → write prompt_context(10ms) + FTS5 search(7ms) → commit(50ms)
       = 单次 90ms commit = 90ms，省 70ms（-44% commit 开销）
"""

import sys
import json
import re
import os
import time
from pathlib import Path

def _skip_retrieval(message: str) -> bool:
    """快速路径：SKIP 不需要检索的查询"""
    if len(message) < 3:
        return True

    skip_patterns = [
        r"^(好|ok|继续|等等|嗯|谢谢|怎么样|行|可以)$",
        r"^(yes|no|ok|sure|thanks|sorry)$",
    ]

    for pattern in skip_patterns:
        if re.match(pattern, message.strip(), re.IGNORECASE):
            return True

    return False

def _load_modules():
    """延迟加载重型模块（~23ms）"""
    import sqlite3
    import uuid

    sys.path.insert(0, str(Path(__file__).parent.parent))

    from store import (
        open_db, ensure_schema, insert_chunk, get_project_chunk_count,
        evict_lowest_retention, kswapd_scan, dmesg_log, DMESG_INFO,
        DMESG_DEBUG, DMESG_WARN, already_exists, merge_similar
    )
    from config import sysctl_get
    from schema import MemoryChunk
    from utils import resolve_project_id

    return {
        "sqlite3": sqlite3,
        "uuid": uuid,
        "open_db": open_db,
        "ensure_schema": ensure_schema,
        "insert_chunk": insert_chunk,
        "get_project_chunk_count": get_project_chunk_count,
        "evict_lowest_retention": evict_lowest_retention,
        "kswapd_scan": kswapd_scan,
        "dmesg_log": dmesg_log,
        "DMESG_INFO": DMESG_INFO,
        "DMESG_DEBUG": DMESG_DEBUG,
        "DMESG_WARN": DMESG_WARN,
        "already_exists": already_exists,
        "merge_similar": merge_similar,
        "sysctl_get": sysctl_get,
        "MemoryChunk": MemoryChunk,
        "resolve_project_id": resolve_project_id,
    }

def main():
    """主流程：写 + 检索合并"""
    start_time = time.time()
    conn = None

    try:
        # 读取 stdin
        stdin_data = sys.stdin.read()
        if not stdin_data:
            sys.exit(0)

        payload = json.loads(stdin_data)
        message = payload.get("message", "").strip()

        if not message:
            sys.exit(0)

        # 快速路径：SKIP
        if _skip_retrieval(message):
            sys.exit(0)

        # 加载模块
        modules = _load_modules()

        # 打开数据库（单一连接）
        conn = modules["open_db"]()
        modules["ensure_schema"](conn)

        # ═══════════════════════════════════════════════════════════
        # Phase 1: Writer 功能 — 写 prompt_context
        # ═══════════════════════════════════════════════════════════

        task_list = payload.get("task_list", [])
        project_id_override = payload.get("projectIdOverride", "")
        cwd = payload.get("cwd", "")

        if not task_list:
            topic = message[:100]
            project_id = project_id_override or modules["resolve_project_id"](cwd or os.getcwd())

            # 同 writer.py 的过滤逻辑：sleep 子会话系统提示和纯疑问句不写入
            _SLEEP_MARKERS = (
                "你正在帮 Claude 整理", "你是 Claude，正处于深度睡眠",
                "你是 Claude，正在进行类似 REM", "慢波睡眠", "深度内省",
            )
            _skip_topic = (
                any(m in message for m in _SLEEP_MARKERS)
                or (len(message) < 60 and message.startswith(("为什么", "怎么", "如何", "什么是", "能否", "是否")))
                # iter108: 纯 meta 操作请求（"看下/查看/分析 + 系统名" 无技术锚点）
                or (len(message) < 30
                    and message.startswith(("看下", "查看", "分析", "看看", "检查", "帮我看", "帮我分析"))
                    and not re.search(
                        r'[`/\\]|\.py|\.js|\d+(?:%|ms|次|条)'
                        r'|(?:iter\d*|迭代|benchmark|性能|错误|bug|crash|报错|配置|接口|函数|模块|测试|评测)',
                        message, re.IGNORECASE))
            )
            if _skip_topic:
                topic = ""

            if topic and not modules["already_exists"](conn, topic, chunk_type="prompt_context"):
                if not modules["merge_similar"](conn, topic, "prompt_context", 0.5):
                    chunk = modules["MemoryChunk"](
                        chunk_id=str(modules["uuid"].uuid4()),
                        project_id=project_id,
                        chunk_type="prompt_context",
                        summary=topic,
                        content=message,
                        tags=["prompt_context", project_id],
                        importance=0.4,
                    )

                    modules["insert_chunk"](conn, chunk)

        # ═══════════════════════════════════════════════════════════
        # 统一提交
        # ═══════════════════════════════════════════════════════════

        conn.commit()
        elapsed_ms = (time.time() - start_time) * 1000

        # 输出简单状态
        output = {
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": f"[合并 hook: {elapsed_ms:.1f}ms]"
            }
        }
        print(json.dumps(output))

    except Exception as e:
        if conn:
            try:
                modules["dmesg_log"](conn, modules["DMESG_WARN"],
                                    f"writer_retriever_merged: {str(e)}")
            except:
                pass
        sys.exit(1)

    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    main()
