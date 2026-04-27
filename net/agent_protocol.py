"""
agent_protocol.py -- Memory OS Agent 间通信消息协议

迭代 90: OS 类比 -- Linux 网络栈 L4/L7 (TCP/IP Protocol Suite, 1983)

TCP/IP 协议背景:
  TCP 段头部 (RFC 793, 1981):
    Source Port (16b) + Dest Port (16b) + Sequence Number (32b) +
    Acknowledgment Number (32b) + Flags (SYN/ACK/FIN/RST) + Window Size + ...
  每个 TCP 段都是自描述的: 谁发的、发给谁、第几个包、需不需要确认。

  IP 数据报头部 (RFC 791, 1981):
    TTL (Time To Live) -- 每经过一个路由器减 1, 到 0 则丢弃
    Protocol -- 上层协议 (TCP=6, UDP=17, ICMP=1)
    Source/Dest IP -- 网络层寻址

Agent 消息协议映射:
  AgentMessage  ~ TCP segment + IP datagram 的融合
    id          ~ IP Identification -- 唯一标识每个消息
    source      ~ IP Source Address -- 发送者 agent_id
    target      ~ IP Dest Address   -- 接收者 ("*" = broadcast, 类比 255.255.255.255)
    msg_type    ~ IP Protocol field -- request/response/notification/heartbeat
    payload     ~ TCP Payload       -- 消息体
    timestamp   ~ TCP Timestamp Option (RFC 1323) -- 用于 RTT 估算和排序
    ttl         ~ IP TTL            -- 转发跳数限制, 防环路
    ack_required ~ TCP ACK flag     -- 可靠传输: 需要 ACK
    seq         ~ TCP Sequence Number -- 消息序列号, 检测乱序/重复
    ack_id      ~ TCP Acknowledgment Number -- 确认哪条消息
    priority    ~ IP DSCP/TOS       -- 区分服务优先级

消息生命周期:
  1. send() 创建 AgentMessage, 写入 net_messages 表
  2. 路由器根据 target 决定投递路径 (单播/广播/组播)
  3. 防火墙检查 INPUT/OUTPUT/FORWARD 链
  4. 接收方 recv() 从队列取出, 如 ack_required=True 则发 ACK
  5. TTL 每跳减 1, 到 0 则丢弃 (dmesg 记录 TTL exceeded)
"""

import json
import sqlite3
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List
import os

# ── 环境配置 (与 sched/ 保持一致) ──────────────────────────────────────────────
NET_OS_DIR = (
    Path(os.environ["MEMORY_OS_DIR"])
    if os.environ.get("MEMORY_OS_DIR")
    else Path.home() / ".claude" / "memory-os"
)
NET_DB_PATH = (
    Path(os.environ["NET_DB"])
    if os.environ.get("NET_DB")
    else NET_OS_DIR / "net.db"
)

# ── 消息类型 (类比 IP Protocol Number) ──────────────────────────────────────────
BROADCAST_TARGET = "*"


class MessageType(str, Enum):
    """
    消息类型 (类比 IP 上层协议字段):
      REQUEST      ~ TCP SYN -- 发起请求, 期望对方响应
      RESPONSE     ~ TCP SYN-ACK/ACK -- 对 REQUEST 的回复
      NOTIFICATION ~ UDP datagram -- 单向通知, 无需回复
      HEARTBEAT    ~ ICMP Echo -- 保活探测
      ACK          ~ TCP ACK -- 确认收到消息
      ERROR        ~ ICMP Dest Unreachable -- 错误通知
    """
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    HEARTBEAT = "heartbeat"
    ACK = "ack"
    ERROR = "error"


class MessageStatus(str, Enum):
    """
    消息状态 (类比 TCP 段在内核中的生命周期):
      QUEUED    ~ sk_buff 在发送队列  -- 已创建, 待投递
      DELIVERED ~ sk_buff 到达接收方  -- 已投递到目标接收队列
      ACKED     ~ 收到 ACK           -- 已确认
      EXPIRED   ~ TTL=0 / 超时       -- 已过期
      REJECTED  ~ RST / ICMP unreachable -- 被防火墙或路由拒绝
      READ      ~ recv() 已读取      -- 接收方已取走
    """
    QUEUED = "queued"
    DELIVERED = "delivered"
    ACKED = "acked"
    EXPIRED = "expired"
    REJECTED = "rejected"
    READ = "read"


class DeliveryMode(str, Enum):
    """
    投递模式 (类比 IP 寻址方式):
      UNICAST   ~ 单播 (一对一)
      BROADCAST ~ 广播 (一对全部, target="*")
      MULTICAST ~ 组播 (一对组, target="team:xxx")
    """
    UNICAST = "unicast"
    BROADCAST = "broadcast"
    MULTICAST = "multicast"


# ── 默认 TTL (类比 Linux 默认 TTL=64, net.ipv4.ip_default_ttl) ────────────────
DEFAULT_TTL = 64
DEFAULT_PRIORITY = 0  # 普通优先级 (范围: -20 ~ 19, 类比 nice)


@dataclass
class AgentMessage:
    """
    Agent 间通信消息 (类比 TCP 段 + IP 数据报):

    核心字段:
      id           -- UUID, 消息唯一标识 (IP Identification)
      source       -- 发送者 agent_id (IP Source Address)
      target       -- 接收者 agent_id, "*"=broadcast, "team:xxx"=multicast
      msg_type     -- 消息类型 (MessageType enum)
      payload      -- 消息体 dict (TCP Payload)
      timestamp    -- ISO8601 UTC 创建时间 (TCP Timestamp)
      ttl          -- 存活跳数, 每跳减 1, 到 0 丢弃 (IP TTL)
      ack_required -- 是否需要 ACK 确认 (TCP reliable delivery)
      seq          -- 序列号, 单调递增 (TCP Sequence Number)
      ack_id       -- 确认的消息 ID (TCP Acknowledgment)
      priority     -- 优先级 -20~19 (IP DSCP/TOS)
      status       -- 消息状态 (MessageStatus)
      delivery_mode-- 投递模式 (DeliveryMode)
      retry_count  -- 已重试次数 (TCP retransmission count)
      max_retries  -- 最大重试次数 (TCP SYN retries)
    """
    source: str
    target: str
    msg_type: str = MessageType.NOTIFICATION.value
    payload: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    ttl: int = DEFAULT_TTL
    ack_required: bool = False
    seq: int = 0
    ack_id: Optional[str] = None
    priority: int = DEFAULT_PRIORITY
    status: str = MessageStatus.QUEUED.value
    delivery_mode: str = DeliveryMode.UNICAST.value
    retry_count: int = 0
    max_retries: int = 3

    def __post_init__(self):
        """校验并规范化字段"""
        # TTL 钳位 (类比 ip_default_ttl 范围 1-255)
        self.ttl = max(0, min(255, self.ttl))
        # Priority 钳位 (类比 nice -20~19)
        self.priority = max(-20, min(19, self.priority))
        # 自动推导 delivery_mode
        if self.target == BROADCAST_TARGET:
            self.delivery_mode = DeliveryMode.BROADCAST.value
        elif self.target.startswith("team:"):
            self.delivery_mode = DeliveryMode.MULTICAST.value
        else:
            self.delivery_mode = DeliveryMode.UNICAST.value

    def to_dict(self) -> Dict[str, Any]:
        """序列化为 dict (类比 sk_buff 序列化为网络字节序)"""
        d = asdict(self)
        d["payload"] = json.dumps(d["payload"]) if isinstance(d["payload"], dict) else d["payload"]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AgentMessage":
        """从 dict 反序列化 (类比 从网络字节序解包为 sk_buff)"""
        d = dict(d)  # shallow copy
        if isinstance(d.get("payload"), str):
            try:
                d["payload"] = json.loads(d["payload"])
            except (json.JSONDecodeError, TypeError):
                d["payload"] = {"raw": d["payload"]}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "AgentMessage":
        """从 sqlite3.Row 构建 (类比 netfilter skb_from_nlattr)"""
        d = dict(row)
        return cls.from_dict(d)

    def decrement_ttl(self) -> bool:
        """
        TTL 减 1, 返回是否仍存活.
        类比 ip_decrease_ttl() in net/ipv4/ip_forward.c
        """
        self.ttl -= 1
        if self.ttl <= 0:
            self.status = MessageStatus.EXPIRED.value
            return False
        return True

    def create_ack(self) -> "AgentMessage":
        """
        为当前消息生成 ACK 回复.
        类比 tcp_send_ack() -- 交换 source/target, 设 ack_id
        """
        return AgentMessage(
            source=self.target,
            target=self.source,
            msg_type=MessageType.ACK.value,
            payload={"acked_msg_id": self.id},
            ack_id=self.id,
            ack_required=False,
            priority=self.priority,
        )

    def create_response(self, payload: Dict[str, Any]) -> "AgentMessage":
        """
        为 REQUEST 创建 RESPONSE.
        类比 TCP 的 SYN-ACK -- 对 SYN 的响应
        """
        return AgentMessage(
            source=self.target,
            target=self.source,
            msg_type=MessageType.RESPONSE.value,
            payload=payload,
            ack_id=self.id,
            ack_required=self.ack_required,
            priority=self.priority,
        )

    def create_error(self, error_msg: str) -> "AgentMessage":
        """
        创建 ERROR 消息 (类比 ICMP Destination Unreachable)
        """
        return AgentMessage(
            source=self.target if self.target != BROADCAST_TARGET else "system",
            target=self.source,
            msg_type=MessageType.ERROR.value,
            payload={"error": error_msg, "original_msg_id": self.id},
            ack_required=False,
            priority=-10,  # 高优先级
        )


# ── 数据库操作 ─────────────────────────────────────────────────────────────────

def _open_net_db(db_path: Path = None) -> sqlite3.Connection:
    """打开 net.db, WAL 模式 (与 store_core.open_db 保持一致)"""
    if db_path is None:
        db_path = NET_DB_PATH
    NET_OS_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_net_schema(conn: sqlite3.Connection) -> None:
    """
    幂等建表 -- net/ 子系统全部表结构

    类比 Linux 网络子系统初始化:
      net_init() -> sock_init() -> proto_init() -> netfilter_init()
    """
    conn.executescript("""
        -- 消息表 (类比 sk_buff 队列)
        CREATE TABLE IF NOT EXISTS net_messages (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            target TEXT NOT NULL,
            msg_type TEXT NOT NULL DEFAULT 'notification',
            payload TEXT DEFAULT '{}',
            timestamp TEXT NOT NULL,
            ttl INTEGER DEFAULT 64,
            ack_required INTEGER DEFAULT 0,
            seq INTEGER DEFAULT 0,
            ack_id TEXT,
            priority INTEGER DEFAULT 0,
            status TEXT DEFAULT 'queued',
            delivery_mode TEXT DEFAULT 'unicast',
            retry_count INTEGER DEFAULT 0,
            max_retries INTEGER DEFAULT 3
        );

        -- 按 target+status 索引 (加速 recv 查询, 类比 socket receive queue hash)
        CREATE INDEX IF NOT EXISTS idx_net_msg_target_status
            ON net_messages(target, status);
        -- 按 source 索引 (加速发件箱查询)
        CREATE INDEX IF NOT EXISTS idx_net_msg_source
            ON net_messages(source);
        -- 按 timestamp 索引 (加速过期扫描, 类比 TCP retransmission timer)
        CREATE INDEX IF NOT EXISTS idx_net_msg_timestamp
            ON net_messages(timestamp);

        -- Agent 注册表 (类比 ARP 表 / DNS 记录)
        CREATE TABLE IF NOT EXISTS net_agents (
            agent_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            team TEXT DEFAULT '',
            endpoint TEXT DEFAULT '',
            status TEXT DEFAULT 'online',
            capabilities TEXT DEFAULT '[]',
            registered_at TEXT NOT NULL,
            last_heartbeat TEXT,
            metadata TEXT DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_net_agents_name
            ON net_agents(name);
        CREATE INDEX IF NOT EXISTS idx_net_agents_team
            ON net_agents(team);

        -- 防火墙规则表 (类比 iptables rules)
        CREATE TABLE IF NOT EXISTS net_firewall_rules (
            rule_id TEXT PRIMARY KEY,
            chain TEXT NOT NULL DEFAULT 'INPUT',
            priority INTEGER DEFAULT 0,
            action TEXT NOT NULL DEFAULT 'ACCEPT',
            source_pattern TEXT DEFAULT '*',
            target_pattern TEXT DEFAULT '*',
            msg_type_pattern TEXT DEFAULT '*',
            enabled INTEGER DEFAULT 1,
            description TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            hit_count INTEGER DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_net_fw_chain
            ON net_firewall_rules(chain, priority);

        -- 连接表 (类比 TCP 连接状态表 / conntrack)
        CREATE TABLE IF NOT EXISTS net_connections (
            conn_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            target TEXT NOT NULL,
            status TEXT DEFAULT 'established',
            created_at TEXT NOT NULL,
            last_activity TEXT,
            messages_sent INTEGER DEFAULT 0,
            messages_received INTEGER DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_net_conn_source
            ON net_connections(source);
        CREATE INDEX IF NOT EXISTS idx_net_conn_target
            ON net_connections(target);

        -- dmesg 桥接: 网络子系统日志
        CREATE TABLE IF NOT EXISTS net_dmesg (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            level TEXT NOT NULL DEFAULT 'INFO',
            subsystem TEXT NOT NULL DEFAULT 'net',
            message TEXT NOT NULL,
            extra TEXT DEFAULT '{}'
        );
    """)
    conn.commit()


def net_dmesg_log(conn: sqlite3.Connection, level: str, subsystem: str,
                  message: str, extra: dict = None) -> None:
    """
    网络子系统 dmesg 日志 (类比 net_dbg/net_info/net_warn/net_err 宏)

    同时尝试桥接到 store.db 的 dmesg (如果可用)
    """
    now = datetime.now(timezone.utc).isoformat()
    extra_json = json.dumps(extra or {})
    conn.execute(
        "INSERT INTO net_dmesg (timestamp, level, subsystem, message, extra) "
        "VALUES (?, ?, ?, ?, ?)",
        (now, level, subsystem, message, extra_json)
    )
    conn.commit()

    # 桥接到 store.db (尽力而为, 不影响主流程)
    try:
        import sys
        parent_dir = str(Path(__file__).parent.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        from store_core import dmesg_log as store_dmesg_log, open_db, ensure_schema
        sconn = open_db()
        ensure_schema(sconn)
        store_dmesg_log(sconn, level, f"net.{subsystem}", message, extra=extra)
        sconn.close()
    except Exception:
        pass  # store.db 不可用时静默降级
