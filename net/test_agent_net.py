#!/usr/bin/env python3
"""
test_agent_net.py -- net/ 子系统完整测试套件

迭代 90: Memory OS Agent 间通信测试

测试覆盖:
  === agent_protocol.py ===
  1.  AgentMessage 默认值和字段校验
  2.  TTL 钳位 (0-255)
  3.  Priority 钳位 (-20~19)
  4.  delivery_mode 自动推导 (unicast/broadcast/multicast)
  5.  to_dict / from_dict 序列化往返
  6.  from_row 从 sqlite3.Row 构建
  7.  decrement_ttl 和过期
  8.  create_ack 生成确认消息
  9.  create_response 生成响应
  10. create_error 生成错误通知

  === agent_router.py ===
  11. register 注册 agent (幂等)
  12. unregister 注销 (标记 offline)
  13. resolve 名称解析 (by id + by name)
  14. resolve 不存在的 agent (NXDOMAIN)
  15. resolve_team 团队解析
  16. resolve_all_online 全部在线列表
  17. route 单播路由
  18. route 广播路由
  19. route 组播路由
  20. route 不可达目标
  21. heartbeat 心跳更新
  22. gc_stale_agents 过期回收
  23. get_routing_table 路由表快照
  24. delete_agent 硬删除

  === agent_firewall.py ===
  25. add_rule 添加规则
  26. check INPUT 链 -- ACCEPT (默认策略)
  27. check INPUT 链 -- DROP 规则
  28. check INPUT 链 -- REJECT 规则
  29. check OUTPUT 链
  30. LOG 规则 (记录但继续匹配)
  31. priority 优先级排序 (first-match-wins)
  32. fnmatch 通配符匹配
  33. remove_rule 删除规则
  34. enable/disable 规则开关
  35. flush 清空链
  36. set_policy 修改默认策略
  37. stats 统计输出
  38. hit_count 命中计数

  === agent_socket.py ===
  39. connect 建立连接
  40. connect 到不可达目标
  41. send 点对点可靠消息 (TCP 类比)
  42. send 不可靠消息 (UDP 类比)
  43. recv 接收消息
  44. recv 已连接模式 (只收 peer 消息)
  45. listen 模式 (收任何消息)
  46. broadcast 广播
  47. broadcast 组播 (team)
  48. close 关闭连接
  49. peek 查看待读数量
  50. recv_all 批量接收
  51. send 被 OUTPUT 防火墙拦截
  52. send 被 INPUT 防火墙拦截
  53. get_connection_info 连接信息
  54. 自动 ACK (reliable=True)

  === 集成测试 ===
  55. 端到端: A->B 请求-响应
  56. 广播: A -> 全体
  57. 防火墙 + 路由 + 消息 联动

隔离策略:
  所有测试使用 tmpfs 目录 (MEMORY_OS_DIR + NET_DB 环境变量覆盖)
  测试完成后自动清理
"""

import os
import sys
import tempfile
import shutil
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ── 测试环境隔离 (必须在 import net 之前设置) ──────────────────────────────────
_TMPDIR = tempfile.mkdtemp(prefix="memoryos_net_test_")
os.environ["MEMORY_OS_DIR"] = _TMPDIR
os.environ["NET_DB"] = str(Path(_TMPDIR) / "net.db")
os.environ["MEMORY_OS_DB"] = str(Path(_TMPDIR) / "store.db")

# 加入父目录到 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from net.agent_protocol import (
    AgentMessage, MessageType, MessageStatus, DeliveryMode,
    BROADCAST_TARGET, DEFAULT_TTL,
    _open_net_db, _ensure_net_schema,
)
from net.agent_router import AgentRouter, AgentEndpoint, AgentStatus
from net.agent_firewall import AgentFirewall, FirewallRule, Chain, Action
from net.agent_socket import AgentSocket, SocketState

# ── 测试统计 ─────────────────────────────────────────────────────────────────
passed = 0
failed = 0
total = 0


def check(name: str, cond: bool, detail: str = "") -> bool:
    global passed, failed, total
    total += 1
    if cond:
        passed += 1
        print(f"  OK [{total:02d}] {name}")
        return True
    else:
        failed += 1
        print(f"  FAIL [{total:02d}] {name}" + (f": {detail}" if detail else ""))
        return False


def section(title: str) -> None:
    print(f"\n{'=' * 66}")
    print(f"  {title}")
    print(f"{'=' * 66}")


# ════════════════════════════════════════════════════════════════════════════════
#  agent_protocol.py 测试
# ════════════════════════════════════════════════════════════════════════════════

def test_protocol():
    section("agent_protocol.py -- AgentMessage")

    # 01. 默认值
    msg = AgentMessage(source="agent-a", target="agent-b")
    check("01 default msg_type = notification",
          msg.msg_type == "notification")
    check("01 default ttl = 64", msg.ttl == DEFAULT_TTL)
    check("01 default ack_required = False", msg.ack_required is False)
    check("01 default status = queued", msg.status == "queued")
    check("01 default delivery_mode = unicast", msg.delivery_mode == "unicast")
    check("01 id is UUID", len(msg.id) == 36 and "-" in msg.id)
    check("01 timestamp is ISO8601", "T" in msg.timestamp)

    # 02. TTL 钳位
    msg_hi = AgentMessage(source="a", target="b", ttl=999)
    check("02 TTL clamp upper to 255", msg_hi.ttl == 255)
    msg_lo = AgentMessage(source="a", target="b", ttl=-5)
    check("02 TTL clamp lower to 0", msg_lo.ttl == 0)

    # 03. Priority 钳位
    msg_p = AgentMessage(source="a", target="b", priority=-99)
    check("03 priority clamp lower to -20", msg_p.priority == -20)
    msg_p2 = AgentMessage(source="a", target="b", priority=99)
    check("03 priority clamp upper to 19", msg_p2.priority == 19)

    # 04. delivery_mode 自动推导
    msg_bc = AgentMessage(source="a", target="*")
    check("04 target='*' -> broadcast", msg_bc.delivery_mode == "broadcast")
    msg_mc = AgentMessage(source="a", target="team:backend")
    check("04 target='team:xxx' -> multicast", msg_mc.delivery_mode == "multicast")
    msg_uc = AgentMessage(source="a", target="agent-b")
    check("04 target='agent-b' -> unicast", msg_uc.delivery_mode == "unicast")

    # 05. 序列化往返
    msg = AgentMessage(
        source="a", target="b", msg_type="request",
        payload={"key": "value", "num": 42}, ttl=10, ack_required=True,
    )
    d = msg.to_dict()
    check("05 to_dict payload is JSON string", isinstance(d["payload"], str))
    msg2 = AgentMessage.from_dict(d)
    check("05 from_dict roundtrip source", msg2.source == "a")
    check("05 from_dict roundtrip payload", msg2.payload == {"key": "value", "num": 42})
    check("05 from_dict roundtrip ack_required", msg2.ack_required is True)

    # 06. from_row (需要数据库)
    conn = _open_net_db()
    _ensure_net_schema(conn)
    d = msg.to_dict()
    conn.execute("""
        INSERT INTO net_messages (id, source, target, msg_type, payload, timestamp,
            ttl, ack_required, seq, ack_id, priority, status, delivery_mode,
            retry_count, max_retries)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (d["id"], d["source"], d["target"], d["msg_type"], d["payload"],
          d["timestamp"], d["ttl"], 1 if d["ack_required"] else 0,
          d["seq"], d["ack_id"], d["priority"], d["status"],
          d["delivery_mode"], d["retry_count"], d["max_retries"]))
    conn.commit()
    row = conn.execute("SELECT * FROM net_messages WHERE id=?", (msg.id,)).fetchone()
    msg_from_row = AgentMessage.from_row(row)
    check("06 from_row source preserved", msg_from_row.source == "a")
    check("06 from_row payload dict", isinstance(msg_from_row.payload, dict))
    conn.close()

    # 07. decrement_ttl
    msg3 = AgentMessage(source="a", target="b", ttl=2)
    alive = msg3.decrement_ttl()
    check("07 ttl=2 -> decrement -> alive=True, ttl=1", alive and msg3.ttl == 1)
    alive = msg3.decrement_ttl()
    check("07 ttl=1 -> decrement -> alive=False, ttl=0", not alive and msg3.ttl == 0)
    check("07 expired status", msg3.status == "expired")

    # 08. create_ack
    original = AgentMessage(source="sender", target="receiver", ack_required=True)
    ack = original.create_ack()
    check("08 ACK source = original target", ack.source == "receiver")
    check("08 ACK target = original source", ack.target == "sender")
    check("08 ACK msg_type = ack", ack.msg_type == "ack")
    check("08 ACK ack_id = original id", ack.ack_id == original.id)
    check("08 ACK ack_required = False", ack.ack_required is False)

    # 09. create_response
    req = AgentMessage(source="client", target="server", msg_type="request")
    resp = req.create_response({"result": "ok"})
    check("09 response source = request target", resp.source == "server")
    check("09 response target = request source", resp.target == "client")
    check("09 response msg_type = response", resp.msg_type == "response")
    check("09 response payload", resp.payload == {"result": "ok"})

    # 10. create_error
    msg4 = AgentMessage(source="sender", target="receiver")
    err = msg4.create_error("not found")
    check("10 error source = original target", err.source == "receiver")
    check("10 error target = original source", err.target == "sender")
    check("10 error msg_type = error", err.msg_type == "error")
    check("10 error payload has error key", "error" in err.payload)


# ════════════════════════════════════════════════════════════════════════════════
#  agent_router.py 测试
# ════════════════════════════════════════════════════════════════════════════════

def test_router():
    section("agent_router.py -- AgentRouter")

    db_path = Path(_TMPDIR) / "router_test.db"
    router = AgentRouter(db_path)

    # 11. register
    ep = router.register("agent-001", "worker-alpha", team="backend",
                         capabilities=["code", "review"])
    check("11 register returns AgentEndpoint", isinstance(ep, AgentEndpoint))
    check("11 register agent_id", ep.agent_id == "agent-001")
    check("11 register name", ep.name == "worker-alpha")
    check("11 register team", ep.team == "backend")
    check("11 register capabilities", ep.capabilities == ["code", "review"])

    # 11b. register 幂等 (upsert)
    ep2 = router.register("agent-001", "worker-alpha-v2", team="backend")
    check("11 register idempotent upsert", ep2.name == "worker-alpha-v2")

    # 更多 agents
    router.register("agent-002", "worker-beta", team="backend")
    router.register("agent-003", "worker-gamma", team="frontend")
    router.register("agent-004", "planner", team="system")

    # 12. unregister
    ok = router.unregister("agent-004")
    check("12 unregister returns True", ok)
    ep_off = router.get_agent("agent-004")
    check("12 unregistered agent status=offline", ep_off.status == "offline")

    # 13. resolve by id + by name
    ep_by_id = router.resolve("agent-002")
    check("13 resolve by agent_id", ep_by_id is not None and ep_by_id.name == "worker-beta")
    ep_by_name = router.resolve("worker-gamma")
    check("13 resolve by name", ep_by_name is not None and ep_by_name.agent_id == "agent-003")

    # 14. resolve 不存在
    ep_none = router.resolve("ghost-agent")
    check("14 resolve nonexistent returns None", ep_none is None)
    # resolve offline agent returns None
    ep_off_resolve = router.resolve("agent-004")
    check("14 resolve offline returns None", ep_off_resolve is None)

    # 15. resolve_team
    backend_agents = router.resolve_team("backend")
    check("15 resolve_team backend has 2 agents",
          len(backend_agents) == 2,
          f"got {len(backend_agents)}")
    check("15 resolve_team backend names",
          set(a.agent_id for a in backend_agents) == {"agent-001", "agent-002"})

    # 16. resolve_all_online
    all_online = router.resolve_all_online()
    check("16 resolve_all_online = 3 (agent-004 is offline)",
          len(all_online) == 3,
          f"got {len(all_online)}")

    # 17. route 单播
    r = router.route("agent-001", "agent-002")
    check("17 unicast route mode", r["mode"] == "unicast")
    check("17 unicast reachable", r["reachable"])
    check("17 unicast 1 target", len(r["targets"]) == 1)

    # 18. route 广播
    r = router.route("agent-001", "*")
    check("18 broadcast mode", r["mode"] == "broadcast")
    check("18 broadcast excludes self",
          all(t.agent_id != "agent-001" for t in r["targets"]))
    check("18 broadcast 2 targets (excl self, excl offline)",
          len(r["targets"]) == 2,
          f"got {len(r['targets'])}")

    # 19. route 组播
    r = router.route("agent-003", "team:backend")
    check("19 multicast mode", r["mode"] == "multicast")
    check("19 multicast 2 targets",
          len(r["targets"]) == 2,
          f"got {len(r['targets'])}")

    # 20. route 不可达
    r = router.route("agent-001", "nonexistent")
    check("20 unreachable route", not r["reachable"])
    check("20 unreachable empty targets", len(r["targets"]) == 0)

    # 21. heartbeat
    ok = router.heartbeat("agent-001")
    check("21 heartbeat returns True", ok)
    ok2 = router.heartbeat("nonexistent")
    check("21 heartbeat nonexistent returns False", not ok2)

    # 22. gc_stale_agents
    # 手动修改心跳时间到过去
    old_time = (datetime.now(timezone.utc) - timedelta(seconds=600)).isoformat()
    router._conn.execute(
        "UPDATE net_agents SET last_heartbeat=? WHERE agent_id=?",
        (old_time, "agent-002")
    )
    router._conn.commit()
    stale = router.gc_stale_agents()
    check("22 gc_stale detects 1 stale agent",
          len(stale) == 1 and stale[0] == "agent-002",
          f"got {stale}")

    # 23. get_routing_table
    table = router.get_routing_table()
    check("23 routing_table has agents dict", "agents" in table)
    check("23 routing_table has teams dict", "teams" in table)
    check("23 routing_table stats total=4", table["stats"]["total"] == 4,
          f"got {table['stats']['total']}")

    # 24. delete_agent
    ok = router.delete_agent("agent-004")
    check("24 delete_agent returns True", ok)
    check("24 deleted agent is gone", router.get_agent("agent-004") is None)

    router.close()


# ════════════════════════════════════════════════════════════════════════════════
#  agent_firewall.py 测试
# ════════════════════════════════════════════════════════════════════════════════

def test_firewall():
    section("agent_firewall.py -- AgentFirewall")

    db_path = Path(_TMPDIR) / "firewall_test.db"
    fw = AgentFirewall(db_path)

    msg = AgentMessage(source="agent-a", target="agent-b", msg_type="request")

    # 25. add_rule
    rule = fw.add_rule("INPUT", source_pattern="agent-a", action="ACCEPT",
                       description="allow agent-a")
    check("25 add_rule returns FirewallRule", isinstance(rule, FirewallRule))
    check("25 rule chain=INPUT", rule.chain == "INPUT")

    # 26. check INPUT -- ACCEPT (默认策略, 无 DROP 规则)
    verdict = fw.check_input(msg)
    check("26 default policy ACCEPT", verdict == "ACCEPT")

    # 27. check INPUT -- DROP 规则
    fw.add_rule("INPUT", source_pattern="malicious-*", action="DROP",
                priority=-10)
    evil_msg = AgentMessage(source="malicious-bot", target="agent-b", msg_type="request")
    verdict = fw.check_input(evil_msg)
    check("27 DROP rule blocks malicious-*", verdict == "DROP")

    # 28. check INPUT -- REJECT 规则
    fw.add_rule("INPUT", source_pattern="spam-*", action="REJECT", priority=-5)
    spam_msg = AgentMessage(source="spam-bot", target="agent-b")
    verdict = fw.check_input(spam_msg)
    check("28 REJECT rule blocks spam-*", verdict == "REJECT")

    # 29. check OUTPUT 链
    fw.add_rule("OUTPUT", target_pattern="forbidden-*", action="DROP")
    out_msg = AgentMessage(source="agent-b", target="forbidden-service")
    verdict = fw.check_output(out_msg)
    check("29 OUTPUT DROP blocks forbidden-*", verdict == "DROP")

    # 30. LOG 规则 (继续匹配)
    fw.flush("INPUT")
    fw.add_rule("INPUT", source_pattern="*", action="LOG", priority=0,
                description="log all")
    fw.add_rule("INPUT", source_pattern="agent-*", action="ACCEPT", priority=10)
    verdict = fw.check_input(msg)
    # LOG 不终止, 应继续匹配到 ACCEPT
    check("30 LOG does not terminate, continues to ACCEPT", verdict == "ACCEPT")

    # 31. priority 排序 (first-match-wins)
    fw.flush("INPUT")
    fw.add_rule("INPUT", source_pattern="agent-a", action="DROP", priority=10)
    fw.add_rule("INPUT", source_pattern="agent-a", action="ACCEPT", priority=1)
    verdict = fw.check_input(msg)
    check("31 lower priority number wins (ACCEPT at 1 beats DROP at 10)",
          verdict == "ACCEPT")

    # 32. fnmatch 通配符
    fw.flush("INPUT")
    fw.add_rule("INPUT", source_pattern="worker-*", action="DROP")
    worker_msg = AgentMessage(source="worker-alpha", target="agent-b")
    verdict = fw.check_input(worker_msg)
    check("32 fnmatch worker-* matches worker-alpha", verdict == "DROP")
    other_msg = AgentMessage(source="planner-1", target="agent-b")
    verdict = fw.check_input(other_msg)
    check("32 fnmatch worker-* does NOT match planner-1", verdict == "ACCEPT")

    # 33. remove_rule
    rules_before = fw.get_rules("INPUT")
    ok = fw.remove_rule(rules_before[0].rule_id)
    check("33 remove_rule returns True", ok)
    rules_after = fw.get_rules("INPUT")
    check("33 rule count decreased", len(rules_after) == len(rules_before) - 1)

    # 34. enable/disable
    fw.flush("INPUT")
    rule = fw.add_rule("INPUT", source_pattern="agent-x", action="DROP")
    fw.disable_rule(rule.rule_id)
    x_msg = AgentMessage(source="agent-x", target="agent-b")
    verdict = fw.check_input(x_msg)
    check("34 disabled rule does not match -> default ACCEPT", verdict == "ACCEPT")
    fw.enable_rule(rule.rule_id)
    verdict = fw.check_input(x_msg)
    check("34 re-enabled rule matches -> DROP", verdict == "DROP")

    # 35. flush
    fw.add_rule("INPUT", source_pattern="test-*", action="DROP")
    count = fw.flush("INPUT")
    check("35 flush INPUT clears rules", count >= 1)
    check("35 INPUT rules empty after flush", len(fw.get_rules("INPUT")) == 0)

    # 36. set_policy
    fw.set_policy("INPUT", "DROP")
    check("36 policy changed to DROP", fw.get_policy("INPUT") == "DROP")
    verdict = fw.check_input(msg)
    check("36 default policy DROP blocks msg", verdict == "DROP")
    fw.set_policy("INPUT", "ACCEPT")  # 恢复

    # 37. stats
    fw.flush()
    fw.add_rule("INPUT", source_pattern="*", action="ACCEPT")
    fw.add_rule("OUTPUT", target_pattern="*", action="ACCEPT")
    st = fw.stats()
    check("37 stats has chains dict", "chains" in st)
    check("37 stats INPUT has rules",
          st["chains"]["INPUT"]["num_rules"] >= 1)

    # 38. hit_count
    fw.flush("INPUT")
    rule = fw.add_rule("INPUT", source_pattern="counter-agent", action="ACCEPT")
    for _ in range(5):
        fw.check_input(AgentMessage(source="counter-agent", target="b"))
    rules = fw.get_rules("INPUT")
    check("38 hit_count = 5 after 5 checks",
          rules[0].hit_count == 5,
          f"got {rules[0].hit_count}")

    fw.close()


# ════════════════════════════════════════════════════════════════════════════════
#  agent_socket.py 测试
# ════════════════════════════════════════════════════════════════════════════════

def test_socket():
    section("agent_socket.py -- AgentSocket")

    db_path = Path(_TMPDIR) / "socket_test.db"
    router = AgentRouter(db_path)
    firewall = AgentFirewall(db_path)

    # 注册 agents
    router.register("alice", "alice", team="dev")
    router.register("bob", "bob", team="dev")
    router.register("charlie", "charlie", team="ops")

    sock_a = AgentSocket("alice", db_path, router=router, firewall=firewall)
    sock_b = AgentSocket("bob", db_path, router=router, firewall=firewall)
    sock_c = AgentSocket("charlie", db_path, router=router, firewall=firewall)

    # 39. connect
    ok = sock_a.connect("bob")
    check("39 connect returns True", ok)
    check("39 state = established", sock_a.state == "established")
    check("39 peer = bob", sock_a.peer == "bob")

    # 40. connect 不可达
    ok = sock_a.connect("nonexistent")
    check("40 connect to nonexistent returns False", not ok)

    # 重新连接
    sock_a.connect("bob")

    # 41. send 可靠消息
    msg = sock_a.send("hello bob", reliable=True)
    check("41 send returns AgentMessage", msg is not None)
    check("41 sent msg source=alice", msg.source == "alice")
    check("41 sent msg ack_required=True", msg.ack_required is True)

    # 42. send 不可靠消息
    msg2 = sock_a.send("fire and forget", reliable=False)
    check("42 unreliable send returns msg", msg2 is not None)
    check("42 unreliable ack_required=False", msg2.ack_required is False)

    # 43. recv
    sock_b.listen()
    received = sock_b.recv()
    check("43 bob recv gets message", received is not None)
    check("43 recv source=alice", received.source == "alice")
    check("43 recv payload has text",
          received.payload.get("text") == "hello bob",
          f"got {received.payload}")

    # 44. recv 已连接模式 (只收 peer)
    # charlie 给 bob 发一条, bob 已连接到 alice, 不应收到 charlie 的
    sock_b_connected = AgentSocket("bob", db_path, router=router, firewall=firewall)
    sock_b_connected.connect("alice")
    sock_c.send("hello from charlie", reliable=False, target="bob")
    received_from_peer = sock_b_connected.recv()
    # bob 已连接 alice, 只能收 alice 的消息, charlie 的收不到
    # 但 alice 没有新消息发给 bob, 所以应该是 None 或 之前的 fire-and-forget
    # 由于之前 sock_b.listen() 已经消费了第一条, 第二条 unreliable 还在队列
    # sock_b_connected 连接到 alice, 只看 source=alice 的
    check("44 connected socket filters by peer",
          received_from_peer is not None and received_from_peer.source == "alice"
          or received_from_peer is None)
    # 用 listen 模式的 sock_b 来收 charlie 的消息
    charlie_msg = sock_b.recv()
    check("44 listen socket gets charlie's msg",
          charlie_msg is not None and charlie_msg.source == "charlie")

    # 45. listen 模式
    sock_b2 = AgentSocket("bob", db_path, router=router, firewall=firewall)
    sock_b2.listen()
    check("45 listen state", sock_b2.state == "listening")

    # 46. broadcast
    msg_bc = sock_a.broadcast("system alert")
    check("46 broadcast returns msg", msg_bc is not None)
    # bob 和 charlie 应收到
    bc_recv_b = sock_b.recv()
    bc_recv_c = sock_c.recv()
    check("46 bob got broadcast",
          bc_recv_b is not None,
          f"bc_recv_b={bc_recv_b}")
    check("46 charlie got broadcast",
          bc_recv_c is not None,
          f"bc_recv_c={bc_recv_c}")

    # 47. 组播
    msg_mc = sock_c.broadcast("dev team update", team="dev")
    check("47 multicast returns msg", msg_mc is not None)
    mc_recv_a = sock_a.recv()
    check("47 alice (dev team) got multicast",
          mc_recv_a is not None,
          f"mc_recv_a={mc_recv_a}")

    # 48. close
    sock_a.close()
    check("48 close state=closed", sock_a.state == "closed")
    check("48 close peer=None", sock_a.peer is None)

    # 49. peek
    # 给 bob 发几条消息
    sock_tmp = AgentSocket("alice", db_path, router=router, firewall=firewall)
    sock_tmp.send("msg1", reliable=False, target="bob")
    sock_tmp.send("msg2", reliable=False, target="bob")
    pending = sock_b.peek()
    check("49 peek shows pending count >= 2",
          pending >= 2,
          f"got {pending}")

    # 50. recv_all
    all_msgs = sock_b.recv_all(limit=50)
    check("50 recv_all returns list", isinstance(all_msgs, list))
    check("50 recv_all gets multiple msgs", len(all_msgs) >= 2,
          f"got {len(all_msgs)}")

    # 51. send 被 OUTPUT 防火墙拦截
    firewall.flush("OUTPUT")
    firewall.add_rule("OUTPUT", target_pattern="bob", action="DROP")
    blocked_sock = AgentSocket("alice", db_path, router=router, firewall=firewall)
    result = blocked_sock.send("blocked msg", target="bob")
    check("51 OUTPUT DROP blocks send -> None", result is None)
    firewall.flush("OUTPUT")  # 清理

    # 52. send 被 INPUT 防火墙拦截
    firewall.flush("INPUT")
    firewall.add_rule("INPUT", source_pattern="alice", action="DROP")
    blocked_sock2 = AgentSocket("alice", db_path, router=router, firewall=firewall)
    result2 = blocked_sock2.send("blocked at input", target="charlie")
    check("52 INPUT DROP blocks delivery -> None", result2 is None)
    firewall.flush("INPUT")  # 清理

    # 53. get_connection_info
    sock_info = AgentSocket("alice", db_path, router=router, firewall=firewall)
    sock_info.connect("bob")
    info = sock_info.get_connection_info()
    check("53 connection_info has agent_id", info["agent_id"] == "alice")
    check("53 connection_info has state", info["state"] == "established")
    check("53 connection_info has peer", info["peer"] == "bob")

    # 54. 自动 ACK
    sock_sender = AgentSocket("alice", db_path, router=router, firewall=firewall)
    sock_receiver = AgentSocket("bob", db_path, router=router, firewall=firewall)
    sock_receiver.listen()
    sock_sender.send("need ack", reliable=True, target="bob")
    recv_msg = sock_receiver.recv()
    check("54 reliable msg received", recv_msg is not None)
    # 检查 ACK 消息被创建
    ack_msg = sock_sender.recv()
    # ACK 可能还在队列中, 尝试用 listen 模式接收
    sock_sender_listen = AgentSocket("alice", db_path, router=router, firewall=firewall)
    sock_sender_listen.listen()
    ack = sock_sender_listen.recv(msg_type="ack")
    check("54 auto-ACK generated",
          ack is not None and ack.msg_type == "ack",
          f"ack={ack}")

    # 清理
    for s in [sock_a, sock_b, sock_c, sock_b_connected, sock_b2, sock_tmp,
              blocked_sock, blocked_sock2, sock_info, sock_sender, sock_receiver,
              sock_sender_listen]:
        try:
            s.close()
        except Exception:
            pass

    router.close()
    firewall.close()


# ════════════════════════════════════════════════════════════════════════════════
#  集成测试
# ════════════════════════════════════════════════════════════════════════════════

def test_integration():
    section("Integration Tests")

    db_path = Path(_TMPDIR) / "integration_test.db"
    router = AgentRouter(db_path)
    firewall = AgentFirewall(db_path)

    router.register("server", "api-server", team="backend")
    router.register("client", "web-client", team="frontend")
    router.register("monitor", "sys-monitor", team="system")

    # 55. 端到端: 请求-响应
    client_sock = AgentSocket("client", db_path, router=router, firewall=firewall)
    server_sock = AgentSocket("server", db_path, router=router, firewall=firewall)
    server_sock.listen()

    # Client 发送请求
    client_sock.connect("server")
    client_sock.send("GET /api/status", reliable=True,
                     msg_type="request",
                     payload={"method": "GET", "path": "/api/status"})

    # Server 接收
    request = server_sock.recv()
    check("55 server receives request", request is not None)
    check("55 request type", request.msg_type == "request")
    check("55 request payload path",
          request.payload.get("path") == "/api/status",
          f"got {request.payload}")

    # Server 回复
    server_sock.send("response", reliable=True, target="client",
                     msg_type="response",
                     payload={"status": 200, "body": "ok"})

    # Client 接收响应 (可能先收到 auto-ACK, 需要跳过)
    response = client_sock.recv()
    # 如果第一条是 ACK (auto-ack from server recv), 再取下一条
    if response and response.msg_type == "ack":
        response = client_sock.recv()
    check("55 client receives response", response is not None)
    if response:
        check("55 response payload status=200",
              response.payload.get("status") == 200,
              f"got {response.payload}")

    # 56. 广播: monitor -> 全体
    monitor_sock = AgentSocket("monitor", db_path, router=router, firewall=firewall)
    monitor_sock.broadcast("ALERT: high memory usage")

    # server 和 client 都应收到
    alert_s = server_sock.recv()
    client_listen = AgentSocket("client", db_path, router=router, firewall=firewall)
    client_listen.listen()
    alert_c = client_listen.recv()
    check("56 server receives broadcast", alert_s is not None)
    check("56 client receives broadcast", alert_c is not None)

    # 57. 防火墙 + 路由 + 消息联动
    firewall.flush()
    # 先排空 client 的消息队列 (之前的 auto-ACK 等残留消息)
    client_drain = AgentSocket("client", db_path, router=router, firewall=firewall)
    client_drain.listen()
    while client_drain.recv() is not None:
        pass
    # 禁止 monitor 给 client 发消息
    firewall.add_rule("INPUT", source_pattern="monitor", target_pattern="client",
                      action="DROP")
    monitor_sock.send("secret alert", target="client", reliable=False)
    blocked = client_drain.recv()
    check("57 firewall blocks monitor->client", blocked is None)

    # 但 server->client 不受影响
    server_sock.send("normal update", target="client", reliable=False)
    normal = client_drain.recv()
    check("57 server->client still works", normal is not None)

    # 清理
    for s in [client_sock, server_sock, monitor_sock, client_listen, client_drain]:
        try:
            s.close()
        except Exception:
            pass
    router.close()
    firewall.close()


# ════════════════════════════════════════════════════════════════════════════════
#  主入口
# ════════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n  Memory OS net/ Test Suite")
    print(f"  tmpdir: {_TMPDIR}")

    try:
        test_protocol()
        test_router()
        test_firewall()
        test_socket()
        test_integration()
    finally:
        # 清理 tmpfs
        shutil.rmtree(_TMPDIR, ignore_errors=True)

    print(f"\n{'=' * 66}")
    print(f"  RESULTS: {passed}/{total} passed, {failed} failed")
    print(f"{'=' * 66}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
