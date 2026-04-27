"""
agent_firewall.py -- Memory OS Agent 防火墙 (安全过滤)

迭代 90: OS 类比 -- Linux Netfilter / iptables (1998, Rusty Russell)

Netfilter 背景:
  Linux 2.4 引入 Netfilter 框架, 在网络栈的 5 个 hook 点插入过滤逻辑:
    PREROUTING  -> 包进入路由决策前
    INPUT       -> 包发往本机 (目标是自己)
    FORWARD     -> 包需要转发 (目标不是自己)
    OUTPUT      -> 本机产生的包即将发出
    POSTROUTING -> 包离开路由决策后

  iptables 规则链 (chain):
    每条规则 = 匹配条件 + 动作 (ACCEPT/DROP/LOG/REJECT)
    规则按优先级顺序匹配, 第一条匹配则执行动作 (first-match-wins)
    链的默认策略 (policy): 所有规则都不匹配时的兜底动作

  conntrack (连接跟踪):
    跟踪每个连接的状态 (NEW/ESTABLISHED/RELATED/INVALID)
    有状态防火墙: 允许 ESTABLISHED 连接的回包, 无需额外规则

Agent 防火墙映射:
  AgentFirewall   ~ iptables + conntrack
  FirewallRule    ~ struct ipt_entry (单条规则)
  Chain           ~ INPUT/OUTPUT/FORWARD 三条链
  check()         ~ nf_hook_slow() -> ipt_do_table() (遍历规则链)
  add_rule()      ~ iptables -A chain -s src -d dst -j action
  default_policy  ~ iptables -P chain ACCEPT/DROP

简化设计:
  仅实现三条链: INPUT (接收) / OUTPUT (发送) / FORWARD (转发)
  匹配条件: source_pattern, target_pattern, msg_type_pattern (支持 * 通配符和 fnmatch)
  动作: ACCEPT (放行) / DROP (丢弃) / LOG (记录并放行) / REJECT (拒绝并回 ERROR)
"""

import json
import sqlite3
import uuid
import fnmatch
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List, Dict, Any
from pathlib import Path

from .agent_protocol import (
    _open_net_db, _ensure_net_schema, net_dmesg_log,
    AgentMessage, MessageStatus,
    NET_DB_PATH,
)


class Chain(str, Enum):
    """
    防火墙链 (类比 Netfilter hook points):
      INPUT   ~ NF_INET_LOCAL_IN   -- 消息发往本 agent
      OUTPUT  ~ NF_INET_LOCAL_OUT  -- 本 agent 发出的消息
      FORWARD ~ NF_INET_FORWARD    -- 需要转发的消息
    """
    INPUT = "INPUT"
    OUTPUT = "OUTPUT"
    FORWARD = "FORWARD"


class Action(str, Enum):
    """
    规则动作 (类比 iptables target):
      ACCEPT ~ NF_ACCEPT  -- 放行
      DROP   ~ NF_DROP    -- 静默丢弃 (不通知发送方)
      LOG    ~ LOG target -- 记录日志后放行
      REJECT ~ REJECT target -- 拒绝并发送 ERROR 回复 (类比 ICMP port unreachable)
    """
    ACCEPT = "ACCEPT"
    DROP = "DROP"
    LOG = "LOG"
    REJECT = "REJECT"


@dataclass
class FirewallRule:
    """
    单条防火墙规则 (类比 struct ipt_entry)

    rule_id          -- 规则唯一标识
    chain            -- 所属链 (INPUT/OUTPUT/FORWARD)
    priority         -- 优先级 (数字越小越先匹配, 类比 iptables 规则序号)
    action           -- 匹配后的动作 (ACCEPT/DROP/LOG/REJECT)
    source_pattern   -- 发送方匹配模式 (* = 任意, 支持 fnmatch)
    target_pattern   -- 接收方匹配模式
    msg_type_pattern -- 消息类型匹配模式
    enabled          -- 是否启用
    description      -- 规则描述
    hit_count        -- 命中次数 (类比 iptables -v 的 pkts 计数)
    """
    chain: str = Chain.INPUT.value
    priority: int = 0
    action: str = Action.ACCEPT.value
    source_pattern: str = "*"
    target_pattern: str = "*"
    msg_type_pattern: str = "*"
    enabled: bool = True
    description: str = ""
    rule_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    hit_count: int = 0

    def matches(self, msg: AgentMessage) -> bool:
        """
        检查消息是否匹配此规则 (类比 ipt_do_table 中的条件匹配)

        使用 fnmatch 做通配符匹配:
          "*" 匹配所有
          "agent-*" 匹配以 "agent-" 开头的
          "team:backend" 精确匹配
        """
        if not self.enabled:
            return False
        if not fnmatch.fnmatch(msg.source, self.source_pattern):
            return False
        if not fnmatch.fnmatch(msg.target, self.target_pattern):
            return False
        if not fnmatch.fnmatch(msg.msg_type, self.msg_type_pattern):
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "FirewallRule":
        d = dict(row)
        d["enabled"] = bool(d.get("enabled", 1))
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class AgentFirewall:
    """
    Agent 防火墙 (类比 Linux Netfilter + iptables + conntrack)

    核心逻辑:
      1. 每条链有独立的规则列表, 按 priority 排序
      2. check() 遍历对应链的规则, first-match-wins
      3. 无规则匹配时使用默认策略 (default_policy)
      4. 命中计数自动递增 (类比 iptables -v)

    默认策略:
      INPUT:   ACCEPT (默认接收所有消息)
      OUTPUT:  ACCEPT (默认允许发送所有消息)
      FORWARD: ACCEPT (默认允许转发)

    这与 Linux 默认策略一致: 新系统默认 ACCEPT, 管理员按需添加 DROP/REJECT 规则
    """

    # 默认策略 (类比 iptables -P chain target)
    DEFAULT_POLICIES = {
        Chain.INPUT.value: Action.ACCEPT.value,
        Chain.OUTPUT.value: Action.ACCEPT.value,
        Chain.FORWARD.value: Action.ACCEPT.value,
    }

    def __init__(self, db_path: Path = None):
        self._db_path = db_path or NET_DB_PATH
        self._conn = _open_net_db(self._db_path)
        _ensure_net_schema(self._conn)
        self._policies = dict(self.DEFAULT_POLICIES)

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── 规则管理 (类比 iptables -A/-D/-L) ──────────────────────────────────────

    def add_rule(self, chain: str, rule: FirewallRule = None, **kwargs) -> FirewallRule:
        """
        添加规则到指定链 (类比 iptables -A chain ...)

        可传入 FirewallRule 对象, 也可用 kwargs 快速构建:
          fw.add_rule("INPUT", source_pattern="malicious-*", action="DROP")
        """
        if rule is None:
            rule = FirewallRule(chain=chain, **kwargs)
        else:
            rule.chain = chain

        self._conn.execute("""
            INSERT INTO net_firewall_rules
                (rule_id, chain, priority, action, source_pattern, target_pattern,
                 msg_type_pattern, enabled, description, created_at, hit_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            rule.rule_id, rule.chain, rule.priority, rule.action,
            rule.source_pattern, rule.target_pattern, rule.msg_type_pattern,
            1 if rule.enabled else 0, rule.description,
            rule.created_at, rule.hit_count,
        ))
        self._conn.commit()

        net_dmesg_log(self._conn, "INFO", "firewall",
                      f"rule added: chain={chain} action={rule.action} "
                      f"src={rule.source_pattern} dst={rule.target_pattern}",
                      {"rule_id": rule.rule_id})
        return rule

    def remove_rule(self, rule_id: str) -> bool:
        """删除规则 (类比 iptables -D chain rule-spec)"""
        cur = self._conn.execute(
            "DELETE FROM net_firewall_rules WHERE rule_id=?", (rule_id,)
        )
        self._conn.commit()
        return cur.rowcount > 0

    def enable_rule(self, rule_id: str) -> bool:
        """启用规则"""
        cur = self._conn.execute(
            "UPDATE net_firewall_rules SET enabled=1 WHERE rule_id=?", (rule_id,)
        )
        self._conn.commit()
        return cur.rowcount > 0

    def disable_rule(self, rule_id: str) -> bool:
        """禁用规则"""
        cur = self._conn.execute(
            "UPDATE net_firewall_rules SET enabled=0 WHERE rule_id=?", (rule_id,)
        )
        self._conn.commit()
        return cur.rowcount > 0

    def get_rules(self, chain: str = None) -> List[FirewallRule]:
        """
        获取规则列表 (类比 iptables -L [chain])

        按 priority ASC 排序 (优先级数字小的先匹配)
        """
        if chain:
            rows = self._conn.execute(
                "SELECT * FROM net_firewall_rules WHERE chain=? ORDER BY priority ASC",
                (chain,)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM net_firewall_rules ORDER BY chain, priority ASC"
            ).fetchall()
        return [FirewallRule.from_row(r) for r in rows]

    def flush(self, chain: str = None) -> int:
        """
        清空规则 (类比 iptables -F [chain])
        """
        if chain:
            cur = self._conn.execute(
                "DELETE FROM net_firewall_rules WHERE chain=?", (chain,)
            )
        else:
            cur = self._conn.execute("DELETE FROM net_firewall_rules")
        self._conn.commit()
        return cur.rowcount

    # ── 策略管理 (类比 iptables -P chain target) ───────────────────────────────

    def set_policy(self, chain: str, action: str) -> None:
        """设置链的默认策略 (类比 iptables -P INPUT DROP)"""
        self._policies[chain] = action

    def get_policy(self, chain: str) -> str:
        """获取链的默认策略"""
        return self._policies.get(chain, Action.ACCEPT.value)

    # ── 核心: 包过滤 (类比 nf_hook_slow -> ipt_do_table) ──────────────────────

    def check(self, message: AgentMessage, chain: str) -> str:
        """
        检查消息是否被允许通过指定链

        类比 Netfilter 的 ipt_do_table():
          1. 按 priority 排序遍历链中的规则
          2. 第一条匹配的规则决定动作 (first-match-wins)
          3. 无规则匹配则使用默认策略

        返回: "ACCEPT", "DROP", "REJECT", "LOG"
        """
        rules = self.get_rules(chain)

        for rule in rules:
            if rule.matches(message):
                # 命中计数递增 (类比 iptables -v 的 pkts/bytes 计数)
                self._conn.execute(
                    "UPDATE net_firewall_rules SET hit_count = hit_count + 1 WHERE rule_id=?",
                    (rule.rule_id,)
                )
                self._conn.commit()

                if rule.action == Action.LOG.value:
                    # LOG: 记录后继续匹配 (类比 LOG target 不终止链遍历)
                    net_dmesg_log(
                        self._conn, "INFO", "firewall",
                        f"LOG: chain={chain} src={message.source} dst={message.target} "
                        f"type={message.msg_type} rule={rule.rule_id}",
                        {"message_id": message.id}
                    )
                    continue  # LOG 不终止, 继续匹配后续规则

                if rule.action == Action.DROP.value:
                    net_dmesg_log(
                        self._conn, "WARN", "firewall",
                        f"DROP: chain={chain} src={message.source} dst={message.target}",
                        {"message_id": message.id, "rule_id": rule.rule_id}
                    )

                if rule.action == Action.REJECT.value:
                    net_dmesg_log(
                        self._conn, "WARN", "firewall",
                        f"REJECT: chain={chain} src={message.source} dst={message.target}",
                        {"message_id": message.id, "rule_id": rule.rule_id}
                    )

                return rule.action

        # 无规则匹配, 使用默认策略
        return self._policies.get(chain, Action.ACCEPT.value)

    def check_input(self, message: AgentMessage) -> str:
        """检查 INPUT 链 (接收方过滤)"""
        return self.check(message, Chain.INPUT.value)

    def check_output(self, message: AgentMessage) -> str:
        """检查 OUTPUT 链 (发送方过滤)"""
        return self.check(message, Chain.OUTPUT.value)

    def check_forward(self, message: AgentMessage) -> str:
        """检查 FORWARD 链 (转发过滤)"""
        return self.check(message, Chain.FORWARD.value)

    # ── 统计 (类比 iptables -nvL) ──────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """
        防火墙统计 (类比 iptables -nvL 输出)
        """
        result = {"policies": dict(self._policies), "chains": {}}
        for chain_name in [Chain.INPUT.value, Chain.OUTPUT.value, Chain.FORWARD.value]:
            rules = self.get_rules(chain_name)
            result["chains"][chain_name] = {
                "policy": self._policies.get(chain_name, "ACCEPT"),
                "num_rules": len(rules),
                "total_hits": sum(r.hit_count for r in rules),
                "rules": [
                    {
                        "rule_id": r.rule_id,
                        "priority": r.priority,
                        "action": r.action,
                        "source": r.source_pattern,
                        "target": r.target_pattern,
                        "msg_type": r.msg_type_pattern,
                        "hits": r.hit_count,
                        "enabled": r.enabled,
                    }
                    for r in rules
                ],
            }
        return result
