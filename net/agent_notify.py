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

  通信通过 net/ 子系统（net.db）独立于 store.db，
  避免两个 hook 争用同一 SQLite 锁。

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
from typing import List, Tuple, Optional

# ── 常量 ──────────────────────────────────────────────────────────────────────
_NOTIFY_SOURCE_PREFIX = "extractor:"   # extractor 的 agent_id 前缀
_NOTIFY_CONSUMER_PREFIX = "loader:"    # loader 的 agent_id 前缀
_KNOWLEDGE_UPDATE_TYPE = "knowledge_update"
_BROADCAST_CONSUME_TARGET = "*"        # 广播目标


def broadcast_knowledge_update(project: str, session_id: str, stats: dict) -> bool:
    """
    extractor commit 后调用：广播本轮写入统计到所有 agent。

    OS 类比：inotify IN_MODIFY 事件广播 — 文件系统内容变更时通知所有订阅者。

    参数：
      project    — 项目 ID（知识来源）
      session_id — 当前会话 ID
      stats      — 写入统计：{"decisions": N, "constraints": N, "chunks": N}

    返回：True = 广播成功，False = 失败（不影响主流程）
    """
    try:
        from net.agent_socket import AgentSocket

        agent_id = f"{_NOTIFY_SOURCE_PREFIX}{session_id[:16]}"
        payload = json.dumps({
            "type": _KNOWLEDGE_UPDATE_TYPE,
            "project": project,
            "session_id": session_id,
            "stats": stats,
            "ts": datetime.now(timezone.utc).isoformat(),
        }, ensure_ascii=False)

        sock = AgentSocket(agent_id)
        try:
            msg = sock.broadcast(payload)
            return msg is not None
        finally:
            sock.close()
    except Exception:
        return False


def consume_pending_notifications(consumer_id: str, limit: int = 3) -> List[dict]:
    """
    loader SessionStart 时调用：消费其他 agent 的知识更新通知。

    OS 类比：inotify read(fd) — 从 inotify 文件描述符读取待处理事件。

    参数：
      consumer_id — 消费者标识（session_id 或 agent_id）
      limit       — 最多消费条数（默认 3，避免注入过多）

    返回：通知列表，每项 {"project": str, "stats": dict, "ts": str}
    """
    try:
        from net.agent_socket import AgentSocket
        from net.agent_protocol import BROADCAST_TARGET

        agent_id = f"{_NOTIFY_CONSUMER_PREFIX}{consumer_id[:16]}"
        sock = AgentSocket(agent_id)
        # listen 模式：接受来自任何源的广播消息
        sock.listen()
        results = []
        try:
            messages = sock.recv_all(limit=limit * 2)  # 多取一些备用，过滤后取 limit 条
            for msg in messages:
                try:
                    payload = json.loads(msg.payload)
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
                except (json.JSONDecodeError, AttributeError):
                    continue
        finally:
            sock.close()
        return results
    except Exception:
        return []
