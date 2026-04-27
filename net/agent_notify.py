#!/usr/bin/env python3
"""
net/agent_notify.py — 跨Agent知识更新通知（高层IPC API）

OS 类比：Linux inotify(7) — 内核文件系统事件通知机制
  inotify 让进程监听文件系统事件（IN_CREATE/IN_MODIFY/IN_DELETE），
  而不是轮询 stat()。知识更新通知同理：extractor 写入后广播，
  而不是让每个 agent 轮询 store.db。

架构：
  extractor Stop hook（写入者）→ broadcast_knowledge_update()
  loader SessionStart hook（读取者）← consume_pending_notifications()

  iter259 修复：从 net.db AgentRouter（需 register()，从未调用）迁移到
  store_vfs.ipc_msgq 路径（ipc_send/ipc_recv，独立于 AgentRouter，直接可用）。

  根因：net.agent_notify 原实现依赖 AgentRouter.route("*") 广播，
  而 net_agents 表从未被 loader/extractor register()，导致 resolve_all_online()
  始终返回空列表 → reachable=False → 零投递。ipc_msgq 路径使用
  target_agent='*' 通配符查询，不依赖任何注册机制，直接可用。

消息格式（JSON payload）：
  {
    "type": "knowledge_update",
    "project": "<project_id>",
    "session_id": "<session_id>",
    "stats": {
      "decisions": N,
      "constraints": N,
      "chunks": N
    },
    "ts": "<ISO8601>"
  }
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List

# ── 常量 ──────────────────────────────────────────────────────────────────────
_NOTIFY_SOURCE_PREFIX = "extractor:"   # extractor 的 agent_id 前缀
_NOTIFY_CONSUMER_PREFIX = "loader:"    # loader 的 agent_id 前缀
_KNOWLEDGE_UPDATE_TYPE = "knowledge_update"
_BROADCAST_TARGET = "*"                # ipc_msgq 广播通配符


def broadcast_knowledge_update(project: str, session_id: str, stats: dict) -> bool:
    """
    extractor commit 后调用：广播本轮写入统计到所有 agent。

    OS 类比：inotify IN_MODIFY 事件广播 — 文件系统内容变更时通知所有订阅者。

    iter259 修复：使用 store_vfs.ipc_send（ipc_msgq 路径），不再依赖
    AgentRouter.route("*")（需先 register() 才能投递，hooks 从未调用）。

    参数：
      project    — 项目 ID（知识来源）
      session_id — 当前会话 ID
      stats      — 写入统计：{"decisions": N, "constraints": N, "chunks": N}

    返回：True = 广播成功，False = 失败（不影响主流程）
    """
    try:
        from store_vfs import open_db, ensure_schema, ipc_send
        conn = open_db()
        ensure_schema(conn)
        source_agent = f"{_NOTIFY_SOURCE_PREFIX}{session_id[:16]}"
        payload = {
            "type": _KNOWLEDGE_UPDATE_TYPE,
            "project": project,
            "session_id": session_id,
            "stats": stats,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        ipc_send(conn, source_agent, _BROADCAST_TARGET,
                 _KNOWLEDGE_UPDATE_TYPE, payload, priority=5, ttl_seconds=3600)
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False


def consume_pending_notifications(consumer_id: str, limit: int = 3) -> List[dict]:
    """
    loader SessionStart 时调用：消费其他 agent 的知识更新通知。

    OS 类比：inotify read(fd) — 从 inotify 文件描述符读取待处理事件。

    iter259 修复：使用 store_vfs.ipc_recv（ipc_msgq 路径），不再依赖
    AgentSocket.recv_all()（从 net_messages 查指定 target，而广播 target='*'
    不等于 loader agent_id，导致零命中）。

    参数：
      consumer_id — 消费者标识（session_id 或 agent_id）
      limit       — 最多消费条数（默认 3，避免注入过多）

    返回：通知列表，每项 {"project": str, "stats": dict, "ts": str}
    """
    try:
        from store_vfs import open_db, ensure_schema, ipc_recv
        agent_id = f"{_NOTIFY_CONSUMER_PREFIX}{consumer_id[:16]}"
        conn = open_db()
        ensure_schema(conn)
        messages = ipc_recv(conn, agent_id, msg_type=_KNOWLEDGE_UPDATE_TYPE, limit=limit * 2)
        conn.commit()
        conn.close()

        results = []
        for msg in messages:
            try:
                payload = msg.get("payload", {})
                if isinstance(payload, str):
                    payload = json.loads(payload)
                if payload.get("type") != _KNOWLEDGE_UPDATE_TYPE:
                    continue
                project = payload.get("project", "")
                stats = payload.get("stats", {})
                ts = payload.get("ts", "")
                if project and stats:
                    results.append({
                        "project": project,
                        "stats": stats,
                        "ts": ts,
                        "session_id": payload.get("session_id", ""),
                    })
                if len(results) >= limit:
                    break
            except (json.JSONDecodeError, AttributeError, TypeError):
                continue
        return results
    except Exception:
        return []
