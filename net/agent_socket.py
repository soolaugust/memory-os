"""
agent_socket.py -- Memory OS Agent Socket API

迭代 90: OS 类比 -- BSD Socket API (4.2BSD, 1983, Bill Joy)

BSD Socket 背景:
  4.2BSD (1983) 引入 socket(2) API, 统一了进程间通信接口:
    socket()   -> 创建套接字描述符 (fd)
    bind()     -> 绑定本地地址
    listen()   -> 监听连接请求 (TCP server)
    connect()  -> 发起连接 (TCP client)
    accept()   -> 接受连接 (TCP server)
    send()     -> 发送数据
    recv()     -> 接收数据
    close()    -> 关闭连接

  Socket 类型:
    SOCK_STREAM  (TCP) -- 可靠、有序、面向连接
    SOCK_DGRAM   (UDP) -- 不可靠、无序、无连接
    SOCK_RAW     (Raw) -- 直接操作 IP 层

Agent Socket 映射:
  AgentSocket    ~ struct socket
  connect()      ~ connect(2) -- 建立到目标 agent 的逻辑连接
  send()         ~ send(2)    -- 发送消息 (reliable=True 类比 TCP, False 类比 UDP)
  recv()         ~ recv(2)    -- 接收消息 (从 net_messages 表取)
  broadcast()    ~ sendto(2) with broadcast addr -- 广播消息
  close()        ~ close(2)   -- 关闭连接
  listen()       ~ listen(2)  -- 设置为监听模式 (接受任何来源的消息)

连接状态 (类比 TCP state machine, RFC 793):
  CLOSED       -> 初始状态
  LISTENING    -> 监听模式 (accept 任何消息)
  ESTABLISHED  -> 已建立连接 (点对点通信)
  CLOSED       -> 连接关闭
"""

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional, List, Dict, Any
from pathlib import Path

from .agent_protocol import (
    _open_net_db, _ensure_net_schema, net_dmesg_log,
    AgentMessage, MessageType, MessageStatus, DeliveryMode,
    BROADCAST_TARGET, DEFAULT_TTL,
    NET_DB_PATH,
)
from .agent_router import AgentRouter, AgentEndpoint
from .agent_firewall import AgentFirewall, Chain, Action


class SocketState(str, Enum):
    """
    Socket 状态 (类比 TCP state machine):
      CLOSED      ~ TCP CLOSED
      LISTENING   ~ TCP LISTEN
      ESTABLISHED ~ TCP ESTABLISHED
    """
    CLOSED = "closed"
    LISTENING = "listening"
    ESTABLISHED = "established"


class AgentSocket:
    """
    Agent 通信套接字 (类比 BSD Socket + TCP/UDP)

    使用模式:
      # 点对点 (类比 TCP client)
      sock = AgentSocket("agent-A")
      sock.connect("agent-B")
      sock.send("hello", reliable=True)
      msg = sock.recv(timeout_ms=5000)
      sock.close()

      # 监听模式 (类比 TCP server)
      sock = AgentSocket("agent-B")
      sock.listen()
      msg = sock.recv()  # 接收任何人发来的消息
      sock.send("reply", reliable=True)  # 回复最后一个发送者

      # 广播 (类比 UDP broadcast)
      sock = AgentSocket("agent-A")
      sock.broadcast("alert: system update")

    内部流程:
      send() -> OUTPUT 防火墙检查 -> 路由解析 -> INPUT 防火墙检查 -> 写入 net_messages
      recv() -> 从 net_messages 表读取 target=self 的消息 -> 标记为 READ
    """

    def __init__(self, agent_id: str, db_path: Path = None,
                 router: AgentRouter = None, firewall: AgentFirewall = None):
        """
        创建 Agent Socket (类比 socket(AF_INET, SOCK_STREAM, 0))

        参数:
          agent_id -- 本 agent 标识 (类比 bind 的本地地址)
          db_path  -- 数据库路径 (测试用覆盖)
          router   -- 共享路由器实例 (不传则新建)
          firewall -- 共享防火墙实例 (不传则新建)
        """
        self._agent_id = agent_id
        self._db_path = db_path or NET_DB_PATH
        self._conn = _open_net_db(self._db_path)
        _ensure_net_schema(self._conn)

        # 路由器和防火墙 (类比 socket 关联的 routing table + netfilter)
        self._router = router or AgentRouter(self._db_path)
        self._firewall = firewall or AgentFirewall(self._db_path)
        self._owns_router = router is None
        self._owns_firewall = firewall is None

        # 连接状态
        self._state = SocketState.CLOSED
        self._peer: Optional[str] = None  # 已连接的对端 agent_id
        self._conn_id: Optional[str] = None
        self._seq_counter = 0  # 发送序列号 (类比 TCP seq)

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def state(self) -> str:
        return self._state.value

    @property
    def peer(self) -> Optional[str]:
        return self._peer

    # ── 连接管理 (类比 connect/listen/close) ───────────────────────────────────

    def connect(self, target: str) -> bool:
        """
        建立与目标 Agent 的连接 (类比 connect(2) + TCP 三次握手)

        流程:
          1. 路由解析: 确认目标可达
          2. 建立连接记录 (conntrack entry)
          3. 状态迁移: CLOSED -> ESTABLISHED

        返回 True 表示连接成功, False 表示目标不可达
        """
        if self._state == SocketState.ESTABLISHED:
            # 已连接, 先断开 (类比 TCP RST + 重新 connect)
            self.close()

        # 路由解析
        route_result = self._router.route(self._agent_id, target)
        if not route_result["reachable"]:
            net_dmesg_log(self._conn, "WARN", "socket",
                          f"connect failed: {target} unreachable",
                          {"source": self._agent_id, "target": target})
            return False

        # 建立连接记录
        now = datetime.now(timezone.utc).isoformat()
        self._conn_id = str(uuid.uuid4())
        self._conn.execute("""
            INSERT INTO net_connections (conn_id, source, target, status, created_at, last_activity)
            VALUES (?, ?, ?, 'established', ?, ?)
        """, (self._conn_id, self._agent_id, target, now, now))
        self._conn.commit()

        self._peer = target
        self._state = SocketState.ESTABLISHED
        self._seq_counter = 0

        net_dmesg_log(self._conn, "INFO", "socket",
                      f"connection established: {self._agent_id} -> {target}",
                      {"conn_id": self._conn_id})
        return True

    def listen(self) -> None:
        """
        进入监听模式 (类比 listen(2) + bind(2))

        监听模式下接受任何来源的消息, 不限定对端.
        """
        self._state = SocketState.LISTENING
        self._peer = None
        net_dmesg_log(self._conn, "INFO", "socket",
                      f"listening: {self._agent_id}",
                      {"agent_id": self._agent_id})

    def close(self) -> None:
        """
        关闭连接 (类比 close(2) + TCP FIN)
        """
        if self._conn_id:
            now = datetime.now(timezone.utc).isoformat()
            self._conn.execute(
                "UPDATE net_connections SET status='closed', last_activity=? WHERE conn_id=?",
                (now, self._conn_id)
            )
            self._conn.commit()

        old_peer = self._peer
        self._state = SocketState.CLOSED
        self._peer = None
        self._conn_id = None
        self._seq_counter = 0

        if old_peer:
            net_dmesg_log(self._conn, "INFO", "socket",
                          f"connection closed: {self._agent_id} -> {old_peer}")

    # ── 发送 (类比 send/sendto/sendmsg) ───────────────────────────────────────

    def send(self, message: str, reliable: bool = True,
             msg_type: str = None, payload: Dict[str, Any] = None,
             target: str = None) -> Optional[AgentMessage]:
        """
        发送消息 (类比 send(2) / sendto(2))

        参数:
          message  -- 消息文本 (放入 payload["text"])
          reliable -- True=需要 ACK (TCP), False=不需要 (UDP)
          msg_type -- 消息类型 (默认: REQUEST if reliable else NOTIFICATION)
          payload  -- 自定义 payload (覆盖 message 文本)
          target   -- 目标 (不传则使用 connect 的 peer)

        返回: 发送的 AgentMessage, 或 None (发送失败)
        """
        # 确定目标
        actual_target = target or self._peer
        if not actual_target:
            if self._state == SocketState.LISTENING:
                net_dmesg_log(self._conn, "WARN", "socket",
                              "send failed: listening socket has no peer, use target param")
                return None
            net_dmesg_log(self._conn, "WARN", "socket",
                          "send failed: not connected and no target specified")
            return None

        # 确定消息类型
        if msg_type is None:
            msg_type = (MessageType.REQUEST.value if reliable
                        else MessageType.NOTIFICATION.value)

        # 构建消息
        self._seq_counter += 1
        actual_payload = payload or {"text": message}
        msg = AgentMessage(
            source=self._agent_id,
            target=actual_target,
            msg_type=msg_type,
            payload=actual_payload,
            ack_required=reliable,
            seq=self._seq_counter,
            priority=0,
        )

        # OUTPUT 防火墙检查
        output_verdict = self._firewall.check_output(msg)
        if output_verdict in (Action.DROP.value, Action.REJECT.value):
            msg.status = MessageStatus.REJECTED.value
            net_dmesg_log(self._conn, "WARN", "socket",
                          f"send blocked by OUTPUT firewall: {output_verdict}",
                          {"message_id": msg.id})
            return None

        # 路由解析 + 投递
        route_result = self._router.route(self._agent_id, actual_target)

        if not route_result["reachable"]:
            net_dmesg_log(self._conn, "WARN", "socket",
                          f"send failed: target {actual_target} unreachable")
            return None

        # 对每个目标执行 INPUT 防火墙检查并投递
        delivered_count = 0
        for target_ep in route_result["targets"]:
            # 创建投递副本 (广播/组播时每个目标一份)
            delivery_msg = AgentMessage(
                source=msg.source,
                target=target_ep.agent_id,
                msg_type=msg.msg_type,
                payload=dict(msg.payload),
                id=msg.id if len(route_result["targets"]) == 1 else str(uuid.uuid4()),
                timestamp=msg.timestamp,
                ttl=msg.ttl,
                ack_required=msg.ack_required,
                seq=msg.seq,
                priority=msg.priority,
                status=MessageStatus.DELIVERED.value,
                delivery_mode=route_result["mode"],
            )

            # INPUT 防火墙检查
            input_verdict = self._firewall.check_input(delivery_msg)
            if input_verdict in (Action.DROP.value, Action.REJECT.value):
                delivery_msg.status = MessageStatus.REJECTED.value
                continue

            # 写入消息队列
            self._persist_message(delivery_msg)
            delivered_count += 1

        # 更新连接统计
        if self._conn_id:
            self._conn.execute(
                "UPDATE net_connections SET messages_sent = messages_sent + 1, "
                "last_activity=? WHERE conn_id=?",
                (datetime.now(timezone.utc).isoformat(), self._conn_id)
            )
            self._conn.commit()

        if delivered_count > 0:
            return msg
        return None

    def broadcast(self, message: str, team: str = None) -> Optional[AgentMessage]:
        """
        广播消息 (类比 sendto with broadcast address / IP multicast)

        team=None: 全局广播 (target="*")
        team="xxx": 组播到指定团队 (target="team:xxx")
        """
        target = f"team:{team}" if team else BROADCAST_TARGET
        return self.send(
            message=message,
            reliable=False,
            msg_type=MessageType.NOTIFICATION.value,
            target=target,
        )

    # ── 接收 (类比 recv/recvfrom/recvmsg) ─────────────────────────────────────

    def recv(self, timeout_ms: int = None, msg_type: str = None) -> Optional[AgentMessage]:
        """
        接收消息 (类比 recv(2) / recvfrom(2))

        从 net_messages 表中取出发给本 agent 的最早一条未读消息.
        取出后标记为 READ (类比从 socket receive buffer 取走).

        参数:
          timeout_ms -- 超时 (毫秒). None=非阻塞, 立即返回. > 0 = 模拟阻塞等待.
                        注: 实际实现为轮询 (AI 环境无真正阻塞语义)
          msg_type   -- 只接收特定类型的消息 (过滤)

        返回: AgentMessage 或 None (无消息)
        """
        # 构建查询条件
        conditions = ["target=?", "status IN ('delivered', 'queued')"]
        params: list = [self._agent_id]

        if self._state == SocketState.ESTABLISHED and self._peer:
            # 已连接模式: 只接收来自 peer 的消息 (类比 connected socket)
            conditions.append("source=?")
            params.append(self._peer)

        if msg_type:
            conditions.append("msg_type=?")
            params.append(msg_type)

        where = " AND ".join(conditions)
        query = f"SELECT * FROM net_messages WHERE {where} ORDER BY priority ASC, timestamp ASC LIMIT 1"

        # 非阻塞或轮询
        if timeout_ms and timeout_ms > 0:
            import time
            deadline = time.monotonic() + timeout_ms / 1000.0
            poll_interval = min(0.05, timeout_ms / 1000.0)  # 50ms 轮询间隔
            while time.monotonic() < deadline:
                row = self._conn.execute(query, params).fetchone()
                if row:
                    return self._consume_message(row)
                time.sleep(poll_interval)
            return None
        else:
            row = self._conn.execute(query, params).fetchone()
            if row:
                return self._consume_message(row)
            return None

    def recv_all(self, limit: int = 100) -> List[AgentMessage]:
        """
        接收所有待读消息 (类比 recvmmsg / 批量读取)
        """
        conditions = ["target=?", "status IN ('delivered', 'queued')"]
        params: list = [self._agent_id]

        if self._state == SocketState.ESTABLISHED and self._peer:
            conditions.append("source=?")
            params.append(self._peer)

        where = " AND ".join(conditions)
        query = f"SELECT * FROM net_messages WHERE {where} ORDER BY priority ASC, timestamp ASC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        messages = []
        for row in rows:
            msg = self._consume_message(row)
            if msg:
                messages.append(msg)
        return messages

    def peek(self) -> int:
        """
        查看待读消息数量 (类比 ioctl(FIONREAD) / MSG_PEEK)
        """
        conditions = ["target=?", "status IN ('delivered', 'queued')"]
        params: list = [self._agent_id]

        if self._state == SocketState.ESTABLISHED and self._peer:
            conditions.append("source=?")
            params.append(self._peer)

        where = " AND ".join(conditions)
        row = self._conn.execute(
            f"SELECT COUNT(*) as cnt FROM net_messages WHERE {where}", params
        ).fetchone()
        return row["cnt"] if row else 0

    # ── 内部方法 ───────────────────────────────────────────────────────────────

    def _persist_message(self, msg: AgentMessage) -> None:
        """将消息写入数据库 (类比 sk_buff 入队列)"""
        d = msg.to_dict()
        self._conn.execute("""
            INSERT INTO net_messages
                (id, source, target, msg_type, payload, timestamp, ttl,
                 ack_required, seq, ack_id, priority, status, delivery_mode,
                 retry_count, max_retries)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            d["id"], d["source"], d["target"], d["msg_type"], d["payload"],
            d["timestamp"], d["ttl"], 1 if d["ack_required"] else 0,
            d["seq"], d["ack_id"], d["priority"], d["status"],
            d["delivery_mode"], d["retry_count"], d["max_retries"],
        ))
        self._conn.commit()

    def _consume_message(self, row: sqlite3.Row) -> Optional[AgentMessage]:
        """
        消费一条消息: 标记为 READ 并返回

        如果 ack_required=True, 自动发送 ACK (类比 TCP 自动 ACK)
        """
        msg = AgentMessage.from_row(row)

        # 标记为已读
        self._conn.execute(
            "UPDATE net_messages SET status=? WHERE id=?",
            (MessageStatus.READ.value, msg.id)
        )
        self._conn.commit()

        # 自动 ACK
        if msg.ack_required:
            ack = msg.create_ack()
            ack.status = MessageStatus.DELIVERED.value
            self._persist_message(ack)

        # 更新连接统计
        if self._conn_id:
            self._conn.execute(
                "UPDATE net_connections SET messages_received = messages_received + 1, "
                "last_activity=? WHERE conn_id=?",
                (datetime.now(timezone.utc).isoformat(), self._conn_id)
            )
            self._conn.commit()

        msg.status = MessageStatus.READ.value
        return msg

    # ── 连接信息 (类比 getsockname/getpeername/getsockopt) ─────────────────────

    def get_connection_info(self) -> Dict[str, Any]:
        """
        获取连接信息 (类比 getsockname + getpeername + netstat)
        """
        info = {
            "agent_id": self._agent_id,
            "state": self._state.value,
            "peer": self._peer,
            "conn_id": self._conn_id,
            "seq_counter": self._seq_counter,
            "pending_messages": self.peek(),
        }
        if self._conn_id:
            row = self._conn.execute(
                "SELECT * FROM net_connections WHERE conn_id=?",
                (self._conn_id,)
            ).fetchone()
            if row:
                info["messages_sent"] = row["messages_sent"]
                info["messages_received"] = row["messages_received"]
                info["created_at"] = row["created_at"]
                info["last_activity"] = row["last_activity"]
        return info

    def __del__(self):
        """析构时关闭连接"""
        try:
            if self._state != SocketState.CLOSED:
                self.close()
            if self._conn:
                self._conn.close()
            if self._owns_router and self._router:
                self._router.close()
            if self._owns_firewall and self._firewall:
                self._firewall.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        return (f"AgentSocket(agent_id={self._agent_id!r}, "
                f"state={self._state.value!r}, peer={self._peer!r})")
