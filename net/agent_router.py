"""
agent_router.py -- Memory OS Agent 路由表与名称解析

迭代 90: OS 类比 -- Linux IP 路由子系统 + DNS resolver

Linux 路由背景:
  路由表 (FIB - Forwarding Information Base):
    ip route show 输出每条路由: 目标网段、下一跳、出接口、metric
    内核查找顺序: 最长前缀匹配 (longest prefix match)
    路由缓存 (route cache): 加速已知目标的查找 (3.6 前 dst_cache, 之后 FIB nexthop)

  ARP 表 (Address Resolution Protocol):
    IP 地址 -> MAC 地址映射, 类比 agent_name -> agent_id 映射
    arp -a 查看当前 ARP 缓存

  DNS (Domain Name System):
    hostname -> IP 地址解析, 类比 agent 名称 -> agent endpoint 解析
    /etc/resolv.conf 配置 nameserver
    gethostbyname() / getaddrinfo() -- 用户空间 API

Agent 路由映射:
  AgentRouter     ~ Linux 路由子系统 (net/ipv4/fib_*.c)
  AgentEndpoint   ~ struct fib_nh (下一跳描述符)
  register()      ~ ip route add -- 注册路由条目
  unregister()    ~ ip route del -- 删除路由条目
  resolve()       ~ DNS resolve / ARP lookup -- 名称解析
  route()         ~ ip_route_output_flow() -- 查找转发路径
  get_routing_table() ~ ip route show -- 查看路由表
"""

import json
import sqlite3
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any
from pathlib import Path

from .agent_protocol import (
    _open_net_db, _ensure_net_schema, net_dmesg_log,
    AgentMessage, MessageStatus, DeliveryMode, BROADCAST_TARGET,
    NET_DB_PATH,
)


# ── Agent 状态 (类比 ARP 条目状态) ─────────────────────────────────────────────
class AgentStatus:
    ONLINE = "online"        # 类比 ARP REACHABLE
    OFFLINE = "offline"      # 类比 ARP STALE -> FAILED
    BUSY = "busy"            # 类比 TCP backlog full


@dataclass
class AgentEndpoint:
    """
    Agent 端点描述符 (类比 struct fib_nh + ARP entry)

    agent_id      -- 唯一标识 (IP 地址)
    name          -- 人类可读名称 (hostname)
    team          -- 所属团队 (子网/VLAN)
    endpoint      -- 物理传输地址 (MAC 地址 / 实际 API URL)
    status        -- 在线状态
    capabilities  -- 能力列表 (类比端口开放列表)
    last_heartbeat -- 最后心跳时间 (ARP 过期判断依据)
    metadata      -- 额外信息
    """
    agent_id: str
    name: str
    team: str = ""
    endpoint: str = ""
    status: str = AgentStatus.ONLINE
    capabilities: List[str] = field(default_factory=list)
    registered_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_heartbeat: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["capabilities"] = json.dumps(d["capabilities"])
        d["metadata"] = json.dumps(d["metadata"])
        return d

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "AgentEndpoint":
        d = dict(row)
        if isinstance(d.get("capabilities"), str):
            try:
                d["capabilities"] = json.loads(d["capabilities"])
            except (json.JSONDecodeError, TypeError):
                d["capabilities"] = []
        if isinstance(d.get("metadata"), str):
            try:
                d["metadata"] = json.loads(d["metadata"])
            except (json.JSONDecodeError, TypeError):
                d["metadata"] = {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class AgentRouter:
    """
    Agent 路由器 (类比 Linux FIB + ARP + DNS resolver)

    职责:
      1. Agent 注册/注销 (路由条目增删)
      2. 名称解析 (DNS-like: name -> AgentEndpoint)
      3. 路由决策 (单播/组播/广播路径)
      4. 心跳管理 (ARP 超时检测)
      5. 路由表查看 (ip route show)

    线程安全: 所有操作通过 SQLite 事务保证一致性
    """

    def __init__(self, db_path: Path = None):
        self._db_path = db_path or NET_DB_PATH
        self._conn = _open_net_db(self._db_path)
        _ensure_net_schema(self._conn)
        # 心跳过期时间 (类比 ARP gc_stale_time, 默认 300s)
        self._heartbeat_timeout_sec = 300

    def close(self):
        """关闭路由器 (类比 net_exit_net)"""
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── 注册/注销 (类比 ip route add/del + ARP entry) ──────────────────────────

    def register(self, agent_id: str, name: str, team: str = "",
                 endpoint: str = "", capabilities: List[str] = None,
                 metadata: Dict[str, Any] = None) -> AgentEndpoint:
        """
        注册 Agent (类比 ip route add + ARP 静态绑定)

        幂等: 如果 agent_id 已存在, 更新信息 (UPSERT)
        """
        now = datetime.now(timezone.utc).isoformat()
        caps_json = json.dumps(capabilities or [])
        meta_json = json.dumps(metadata or {})

        self._conn.execute("""
            INSERT INTO net_agents (agent_id, name, team, endpoint, status,
                                    capabilities, registered_at, last_heartbeat, metadata)
            VALUES (?, ?, ?, ?, 'online', ?, ?, ?, ?)
            ON CONFLICT(agent_id) DO UPDATE SET
                name=excluded.name, team=excluded.team, endpoint=excluded.endpoint,
                status='online', capabilities=excluded.capabilities,
                last_heartbeat=excluded.last_heartbeat, metadata=excluded.metadata
        """, (agent_id, name, team, endpoint, caps_json, now, now, meta_json))
        self._conn.commit()

        net_dmesg_log(self._conn, "INFO", "router",
                      f"agent registered: {name} ({agent_id}) team={team}",
                      {"agent_id": agent_id, "team": team})

        return AgentEndpoint(
            agent_id=agent_id, name=name, team=team, endpoint=endpoint,
            capabilities=capabilities or [], registered_at=now,
            last_heartbeat=now, metadata=metadata or {},
        )

    def unregister(self, agent_id: str) -> bool:
        """
        注销 Agent (类比 ip route del + ARP flush)

        不立即删除, 而是标记为 offline (类比 ARP STALE 状态)
        后续可通过 gc_stale_agents() 清理
        """
        cur = self._conn.execute(
            "UPDATE net_agents SET status='offline' WHERE agent_id=?",
            (agent_id,)
        )
        self._conn.commit()
        if cur.rowcount > 0:
            net_dmesg_log(self._conn, "INFO", "router",
                          f"agent unregistered: {agent_id}")
            return True
        return False

    def delete_agent(self, agent_id: str) -> bool:
        """硬删除 agent 记录 (类比 ARP flush + route del)"""
        cur = self._conn.execute(
            "DELETE FROM net_agents WHERE agent_id=?", (agent_id,)
        )
        self._conn.commit()
        return cur.rowcount > 0

    # ── 名称解析 (类比 DNS resolve / ARP lookup) ───────────────────────────────

    def resolve(self, name: str) -> Optional[AgentEndpoint]:
        """
        按名称解析 Agent (类比 gethostbyname / DNS A record lookup)

        查找优先级:
          1. 精确匹配 agent_id (类比 IP 直连路由)
          2. 精确匹配 name (类比 DNS A record)
          3. 返回 None (类比 NXDOMAIN)

        只返回 online 状态的 agent (类比 ARP REACHABLE)
        """
        # 先按 agent_id 精确匹配
        row = self._conn.execute(
            "SELECT * FROM net_agents WHERE agent_id=? AND status='online'",
            (name,)
        ).fetchone()
        if row:
            return AgentEndpoint.from_row(row)

        # 再按 name 匹配
        row = self._conn.execute(
            "SELECT * FROM net_agents WHERE name=? AND status='online'",
            (name,)
        ).fetchone()
        if row:
            return AgentEndpoint.from_row(row)

        return None

    def resolve_team(self, team: str) -> List[AgentEndpoint]:
        """
        解析团队内所有在线 Agent (类比 IGMP 组成员查询)
        """
        rows = self._conn.execute(
            "SELECT * FROM net_agents WHERE team=? AND status='online'",
            (team,)
        ).fetchall()
        return [AgentEndpoint.from_row(r) for r in rows]

    def resolve_all_online(self) -> List[AgentEndpoint]:
        """
        获取所有在线 Agent (类比 ARP -a, 广播目标列表)
        """
        rows = self._conn.execute(
            "SELECT * FROM net_agents WHERE status='online'"
        ).fetchall()
        return [AgentEndpoint.from_row(r) for r in rows]

    # ── 路由决策 (类比 ip_route_output_flow) ───────────────────────────────────

    def route(self, source: str, target: str) -> Dict[str, Any]:
        """
        计算从 source 到 target 的路由路径

        类比 ip_route_output_flow() + FIB lookup:
          1. target="*"       -> broadcast (返回所有在线 agent)
          2. target="team:X"  -> multicast (返回 team X 的所有 agent)
          3. target=agent_id  -> unicast (直接路由)

        返回:
          {
            "mode": "unicast|broadcast|multicast",
            "source": AgentEndpoint or None,
            "targets": [AgentEndpoint, ...],
            "hops": int,
            "reachable": bool,
          }
        """
        # 解析发送方
        source_ep = self.resolve(source)

        # 广播
        if target == BROADCAST_TARGET:
            targets = self.resolve_all_online()
            # 广播时排除发送方自身
            targets = [t for t in targets if t.agent_id != source]
            return {
                "mode": DeliveryMode.BROADCAST.value,
                "source": source_ep,
                "targets": targets,
                "hops": 1,
                "reachable": len(targets) > 0,
            }

        # 组播
        if target.startswith("team:"):
            team_name = target[5:]  # 去掉 "team:" 前缀
            targets = self.resolve_team(team_name)
            targets = [t for t in targets if t.agent_id != source]
            return {
                "mode": DeliveryMode.MULTICAST.value,
                "source": source_ep,
                "targets": targets,
                "hops": 1,
                "reachable": len(targets) > 0,
            }

        # 单播
        target_ep = self.resolve(target)
        return {
            "mode": DeliveryMode.UNICAST.value,
            "source": source_ep,
            "targets": [target_ep] if target_ep else [],
            "hops": 1,
            "reachable": target_ep is not None,
        }

    # ── 心跳管理 (类比 ARP timer / NUD state machine) ─────────────────────────

    def heartbeat(self, agent_id: str) -> bool:
        """
        更新 Agent 心跳 (类比 ARP 刷新 REACHABLE 状态)
        """
        now = datetime.now(timezone.utc).isoformat()
        cur = self._conn.execute(
            "UPDATE net_agents SET last_heartbeat=?, status='online' WHERE agent_id=?",
            (now, agent_id)
        )
        self._conn.commit()
        return cur.rowcount > 0

    def gc_stale_agents(self) -> List[str]:
        """
        垃圾回收: 将超时未心跳的 Agent 标记为 offline
        (类比 ARP gc: neigh_periodic_work -> neigh_cleanup)

        阈值: heartbeat_timeout_sec (默认 300s, 类比 gc_stale_time)
        """
        cutoff = (
            datetime.now(timezone.utc) - timedelta(seconds=self._heartbeat_timeout_sec)
        ).isoformat()

        rows = self._conn.execute(
            "SELECT agent_id FROM net_agents WHERE status='online' AND last_heartbeat < ?",
            (cutoff,)
        ).fetchall()
        stale_ids = [r["agent_id"] for r in rows]

        if stale_ids:
            placeholders = ",".join("?" * len(stale_ids))
            self._conn.execute(
                f"UPDATE net_agents SET status='offline' WHERE agent_id IN ({placeholders})",
                stale_ids
            )
            self._conn.commit()
            net_dmesg_log(self._conn, "WARN", "router",
                          f"gc: {len(stale_ids)} stale agents marked offline",
                          {"agent_ids": stale_ids})
        return stale_ids

    # ── 路由表查看 (类比 ip route show / arp -a) ───────────────────────────────

    def get_routing_table(self) -> Dict[str, Any]:
        """
        返回当前路由表 (类比 ip route show + arp -a)

        输出:
          {
            "agents": { agent_id: {name, team, status, ...}, ... },
            "teams": { team_name: [agent_id, ...], ... },
            "stats": { "total": N, "online": N, "offline": N },
          }
        """
        rows = self._conn.execute("SELECT * FROM net_agents").fetchall()
        agents = {}
        teams: Dict[str, List[str]] = {}
        online_count = 0
        offline_count = 0

        for r in rows:
            ep = AgentEndpoint.from_row(r)
            agents[ep.agent_id] = {
                "name": ep.name,
                "team": ep.team,
                "status": ep.status,
                "endpoint": ep.endpoint,
                "last_heartbeat": ep.last_heartbeat,
                "capabilities": ep.capabilities,
            }
            if ep.team:
                teams.setdefault(ep.team, []).append(ep.agent_id)
            if ep.status == AgentStatus.ONLINE:
                online_count += 1
            else:
                offline_count += 1

        return {
            "agents": agents,
            "teams": teams,
            "stats": {
                "total": len(agents),
                "online": online_count,
                "offline": offline_count,
            },
        }

    def get_agent(self, agent_id: str) -> Optional[AgentEndpoint]:
        """查询单个 agent (无论状态)"""
        row = self._conn.execute(
            "SELECT * FROM net_agents WHERE agent_id=?", (agent_id,)
        ).fetchone()
        return AgentEndpoint.from_row(row) if row else None
