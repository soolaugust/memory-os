"""
EXPERIMENTAL — 未接入生产。无 hook 注册，无实际调用者。
保留供未来 agent 通信需求参考。

net/ -- Memory OS Agent 间通信子系统 (网络栈)

迭代 90: OS 类比 -- Linux 网络栈 (BSD Socket + TCP/IP + Netfilter)

Linux 网络栈核心分层 (自上而下):
  L7  Application   -- HTTP/gRPC/...
  L4  Transport      -- TCP (可靠) / UDP (不可靠) / SCTP
  L3  Network        -- IP (路由、寻址、TTL)
  L2  Data Link      -- Ethernet / Wi-Fi (帧、MAC)
  L1  Physical       -- 电信号 / 光信号

Memory OS net/ 映射:
  agent_socket.py    ~ L4-L7  Socket API (send/recv/connect/listen/broadcast)
  agent_router.py    ~ L3     路由表 + DNS (寻址、路由决策、名称解析)
  agent_protocol.py  ~ L2-L3  消息协议 (AgentMessage 数据结构 + 序列化)
  agent_firewall.py  ~ Netfilter (INPUT/OUTPUT/FORWARD 链 + 规则匹配)

  物理层 (L1) = SendMessage tool / MetaBot HTTP API (不在此模块实现)

模块:
  agent_protocol.py  -- 消息协议和数据类型 (AgentMessage, MessageType, etc.)
  agent_router.py    -- 路由表和名称解析 (AgentRouter, AgentEndpoint)
  agent_firewall.py  -- 安全过滤/防火墙 (AgentFirewall, FirewallRule)
  agent_socket.py    -- Socket API (AgentSocket -- 应用层统一接口)
"""

from .agent_protocol import (
    AgentMessage, MessageType, MessageStatus, DeliveryMode,
    BROADCAST_TARGET, DEFAULT_TTL, DEFAULT_PRIORITY,
    _open_net_db, _ensure_net_schema, net_dmesg_log,
    NET_OS_DIR, NET_DB_PATH,
)
from .agent_router import (
    AgentRouter, AgentEndpoint, AgentStatus,
)
from .agent_firewall import (
    AgentFirewall, FirewallRule, Chain, Action,
)
from .agent_socket import (
    AgentSocket, SocketState,
)

__all__ = [
    # Protocol
    "AgentMessage", "MessageType", "MessageStatus", "DeliveryMode",
    "BROADCAST_TARGET", "DEFAULT_TTL", "DEFAULT_PRIORITY",
    # Router
    "AgentRouter", "AgentEndpoint", "AgentStatus",
    # Firewall
    "AgentFirewall", "FirewallRule", "Chain", "Action",
    # Socket
    "AgentSocket", "SocketState",
    # Internal (for advanced usage / testing)
    "_open_net_db", "_ensure_net_schema", "net_dmesg_log",
    "NET_OS_DIR", "NET_DB_PATH",
]
