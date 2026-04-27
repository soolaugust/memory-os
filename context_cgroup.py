"""
context_cgroup.py — Context Window Budget Controller

迭代 B5：OS 类比 — Linux cgroup memory controller (2007, Paul Menage/LSFMM)

背景：
  Linux cgroup memory controller 解决的核心问题：
    多个进程共享有限物理内存，每个进程都想要更多，
    没有全局控制器时，一个进程的内存膨胀会导致其他进程 OOM。
  解法：
    memory.max  — 硬限制（超过就 OOM kill 或 reclaim）
    memory.high — 软限制（超过就 throttle，触发 reclaim）
    memory.stat — 各子系统内存占用统计

  AIOS 当前问题：
    Claude Code 的 system prompt 由多个"子系统"注入内容：
      - CLAUDE.md（配置指令）
      - Skills 列表（ECC + 本地）
      - MCP tool schemas（每个 server 注册所有 tool 的 JSON schema）
      - Agent types（内建 agent 描述）
      - Session summary hook
      - Memory-OS loader hook
      - UserPromptSubmit hook
    没有全局控制器时，任何子系统膨胀都会导致 "Prompt is too long"（= OOM）。

  解法：Context Cgroup Controller
    1. 扫描所有注入源，估算其 token 占用（memory.stat）
    2. 对比模型 context window 限制（memory.max）
    3. 当总占用超过阈值时，触发降级策略（memory reclaim/OOM kill）：
       - 第一级（memory.high）：禁用低优先级 skills/MCP servers
       - 第二级（memory.max）：只保留最小核心 + 报警

数据来源：
  - ~/.claude/skills/ 和 ECC cache 目录扫描
  - .mcp.json 文件中的 server 数量 × 估算 tool 数
  - CLAUDE.md 文件大小
  - settings.json hooks 配置

输出：
  - 预算报告（当前 vs 限制）
  - 自动降级动作（移除低优先级组件）
  - dmesg 日志记录
"""

import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── 常量 ──────────────────────────────────────────────────────

# 模型 context window token 估算（保守值）
# Claude Opus 4.6 context = 200K tokens
# 系统提示（含 tool schemas）的安全预算：不超过 40% = 80K tokens
# 1 token ≈ 4 chars（英文）/ ≈ 2 chars（中文）/ JSON schema ≈ 3 chars
# 粗略估算：80K tokens ≈ 240K chars（JSON schema 为主）
CONTEXT_MAX_CHARS = 200_000   # memory.max — 硬限制（chars）
CONTEXT_HIGH_CHARS = 150_000  # memory.high — 软限制，超过开始降级

# 各组件每个实例的估算 char 占用（基于实测）
CHAR_ESTIMATES = {
    "skill_entry": 80,          # 每个 skill 在列表中 ≈80 chars（名+描述）
    "mcp_tool_schema": 1500,    # 每个 MCP tool 的 JSON schema ≈1500 chars
    "agent_type": 300,          # Agent tool 中每个 agent 类型描述 ≈300 chars
    "claude_md_char": 1,        # CLAUDE.md 每 char = 1 char（直接计量）
    "hook_context": 500,        # 每个非 async hook 的 additionalContext ≈500 chars
}

# MCP server → 估算 tool 数量（基于实测）
MCP_TOOL_COUNTS = {
    "github": 22,
    "playwright": 20,
    "memory": 10,
    "serena": 30,
    "context7": 2,
    "exa": 2,
    "sequential-thinking": 1,
    "fetch": 1,
    "deepwiki": 5,
    "spec-workflow": 3,
    "wechat-article-reader": 1,
}

# 优先级分组（类比 cgroup 的 hierarchy）
# priority 越低越先被 reclaim
SKILL_PRIORITY = {
    # 核心（永不回收）
    "metabot": 100, "metamemory": 100, "metaskill": 100, "supermemory": 100,
    "code-review": 90, "search-first": 90,
    # 领域核心
    "cpu-expert": 80, "kernel-reviewer": 80, "sched-reviewer": 80,
    "send-patch": 80, "pe-mine": 80, "pe-reviewer": 80,
    "feishu": 75, "intraday-scan": 75,
    # 通用工具
    "deep-research": 70, "configure-ecc": 60, "skill-hub": 60,
    "article-writing": 50, "crosspost": 50, "pptx": 50,
    # 默认
    "_default": 30,
}

MCP_PRIORITY = {
    # 核心（永不回收）
    "serena": 100,
    "context7": 80,
    "fetch": 70,
    # 工具型
    "exa": 60,
    "sequential-thinking": 50,
    "deepwiki": 50,
    # 可替代（gh CLI / memory-os / 按需手动启用）
    "github": 20,
    "playwright": 20,
    "memory": 10,
    "spec-workflow": 30,
    "wechat-article-reader": 40,
    "_default": 30,
}

HOME = Path.home()
CLAUDE_DIR = HOME / ".claude"


@dataclass
class ComponentStat:
    """一个组件的资源统计（类比 memory.stat 的子项）"""
    name: str
    component_type: str   # "skill" | "mcp_server" | "claude_md" | "hook" | "agent_type" | "builtin"
    estimated_chars: int
    priority: int
    source: str           # 来源路径
    removable: bool = True


@dataclass
class BudgetReport:
    """预算报告（类比 memory.stat）"""
    timestamp: str
    total_chars: int
    max_chars: int = CONTEXT_MAX_CHARS
    high_chars: int = CONTEXT_HIGH_CHARS
    components: list = field(default_factory=list)
    pressure: str = "none"   # "none" | "some" | "full" — PSI 级别
    actions_taken: list = field(default_factory=list)

    @property
    def usage_pct(self) -> float:
        return (self.total_chars / self.max_chars * 100) if self.max_chars > 0 else 0

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "total_chars": self.total_chars,
            "max_chars": self.max_chars,
            "high_chars": self.high_chars,
            "usage_pct": round(self.usage_pct, 1),
            "pressure": self.pressure,
            "components_by_type": self._group_by_type(),
            "top_consumers": [
                {"name": c.name, "type": c.component_type, "chars": c.estimated_chars}
                for c in sorted(self.components, key=lambda x: x.estimated_chars, reverse=True)[:10]
            ],
            "actions_taken": self.actions_taken,
        }

    def _group_by_type(self) -> dict:
        groups = {}
        for c in self.components:
            t = c.component_type
            if t not in groups:
                groups[t] = {"count": 0, "total_chars": 0}
            groups[t]["count"] += 1
            groups[t]["total_chars"] += c.estimated_chars
        return groups


def _scan_skills() -> list[ComponentStat]:
    """扫描所有 skill 来源"""
    stats = []

    # 1. ~/.claude/skills/
    user_skills = CLAUDE_DIR / "skills"
    if user_skills.exists():
        for d in sorted(user_skills.iterdir()):
            if d.is_dir():
                name = d.name
                prio = SKILL_PRIORITY.get(name, SKILL_PRIORITY["_default"])
                stats.append(ComponentStat(
                    name=name,
                    component_type="skill",
                    estimated_chars=CHAR_ESTIMATES["skill_entry"],
                    priority=prio,
                    source=str(d),
                ))

    # 2. Workspace skills
    cwd = Path.cwd()
    ws_skills = cwd / ".claude" / "skills"
    if ws_skills.exists():
        for d in sorted(ws_skills.iterdir()):
            if d.is_dir():
                name = d.name
                prio = SKILL_PRIORITY.get(name, SKILL_PRIORITY["_default"])
                stats.append(ComponentStat(
                    name=f"ws:{name}",
                    component_type="skill",
                    estimated_chars=CHAR_ESTIMATES["skill_entry"],
                    priority=prio,
                    source=str(d),
                    removable=False,  # workspace skills 不自动删
                ))

    # 3. ECC marketplace skills/commands/agents (主加载源)
    ecc_mp = CLAUDE_DIR / "plugins" / "marketplaces" / "everything-claude-code"
    for subdir, comp_type, char_key in [
        ("skills", "skill", "skill_entry"),
        ("commands", "command", "skill_entry"),
        ("agents", "agent_type", "agent_type"),
    ]:
        mp_dir = ecc_mp / subdir
        if mp_dir.exists():
            for item in sorted(mp_dir.iterdir()):
                if (item.is_dir() and subdir == "skills") or (item.is_file() and item.suffix == ".md"):
                    name = item.stem if item.is_file() else item.name
                    prio = SKILL_PRIORITY.get(name, SKILL_PRIORITY["_default"])
                    stats.append(ComponentStat(
                        name=f"ecc-mp:{name}",
                        component_type=comp_type,
                        estimated_chars=CHAR_ESTIMATES[char_key],
                        priority=prio,
                        source=str(item),
                    ))

    # 4. ECC plugin cache skills
    ecc_cache = CLAUDE_DIR / "plugins" / "cache" / "everything-claude-code"
    if ecc_cache.exists():
        # 找到版本目录
        for org_dir in ecc_cache.iterdir():
            if org_dir.is_dir():
                for ver_dir in org_dir.iterdir():
                    skills_dir = ver_dir / "skills"
                    if skills_dir.exists():
                        for d in sorted(skills_dir.iterdir()):
                            if d.is_dir():
                                name = d.name
                                prio = SKILL_PRIORITY.get(name, SKILL_PRIORITY["_default"])
                                stats.append(ComponentStat(
                                    name=f"ecc:{name}",
                                    component_type="skill",
                                    estimated_chars=CHAR_ESTIMATES["skill_entry"],
                                    priority=prio,
                                    source=str(d),
                                ))

    # 4. ECC plugin cache commands (也注册为 slash commands / skills)
    if ecc_cache.exists():
        for org_dir in ecc_cache.iterdir():
            if org_dir.is_dir():
                for ver_dir in org_dir.iterdir():
                    cmds_dir = ver_dir / "commands"
                    if cmds_dir.exists():
                        for f in sorted(cmds_dir.iterdir()):
                            if f.is_file() and f.suffix == ".md":
                                name = f.stem
                                stats.append(ComponentStat(
                                    name=f"ecc-cmd:{name}",
                                    component_type="command",
                                    estimated_chars=CHAR_ESTIMATES["skill_entry"],
                                    priority=SKILL_PRIORITY.get(name, 20),
                                    source=str(f),
                                ))

    # 5. ECC plugin cache agents (注册为 Agent tool 的 subagent_type)
    if ecc_cache.exists():
        for org_dir in ecc_cache.iterdir():
            if org_dir.is_dir():
                for ver_dir in org_dir.iterdir():
                    agents_dir = ver_dir / "agents"
                    if agents_dir.exists():
                        for f in sorted(agents_dir.iterdir()):
                            if f.is_file() and f.suffix == ".md":
                                name = f.stem
                                stats.append(ComponentStat(
                                    name=f"ecc-agent:{name}",
                                    component_type="agent_type",
                                    estimated_chars=CHAR_ESTIMATES["agent_type"],
                                    priority=SKILL_PRIORITY.get(name, 20),
                                    source=str(f),
                                ))

    return stats


def _scan_mcp_servers() -> list[ComponentStat]:
    """扫描所有 MCP server 配置"""
    stats = []
    mcp_files = []

    # 1. 项目级 .mcp.json
    cwd = Path.cwd()
    proj_mcp = cwd / ".mcp.json"
    if proj_mcp.exists():
        mcp_files.append(("project", proj_mcp))

    # 2. ECC marketplace .mcp.json
    ecc_mcp = CLAUDE_DIR / "plugins" / "marketplaces" / "everything-claude-code" / ".mcp.json"
    if ecc_mcp.exists():
        mcp_files.append(("ecc", ecc_mcp))

    # 3. settings.json enabledMcpjsonServers
    settings = CLAUDE_DIR / "settings.json"
    if settings.exists():
        try:
            cfg = json.loads(settings.read_text())
            for srv_name in cfg.get("enabledMcpjsonServers", []):
                tool_count = MCP_TOOL_COUNTS.get(srv_name, 3)
                prio = MCP_PRIORITY.get(srv_name, MCP_PRIORITY["_default"])
                stats.append(ComponentStat(
                    name=srv_name,
                    component_type="mcp_server",
                    estimated_chars=tool_count * CHAR_ESTIMATES["mcp_tool_schema"],
                    priority=prio,
                    source=f"settings.json:enabledMcpjsonServers",
                    removable=True,
                ))
        except Exception:
            pass

    for source_label, mcp_file in mcp_files:
        try:
            data = json.loads(mcp_file.read_text())
            for srv_name in data.get("mcpServers", {}):
                tool_count = MCP_TOOL_COUNTS.get(srv_name, 3)
                prio = MCP_PRIORITY.get(srv_name, MCP_PRIORITY["_default"])
                stats.append(ComponentStat(
                    name=srv_name,
                    component_type="mcp_server",
                    estimated_chars=tool_count * CHAR_ESTIMATES["mcp_tool_schema"],
                    priority=prio,
                    source=str(mcp_file),
                    removable=(source_label != "project"),
                ))
        except Exception:
            pass

    return stats


def _scan_claude_md() -> list[ComponentStat]:
    """扫描 CLAUDE.md 大小"""
    stats = []
    cwd = Path.cwd()
    for label, path in [("project", cwd / "CLAUDE.md"), ("home", HOME / ".claude" / "CLAUDE.md")]:
        if path.exists():
            size = path.stat().st_size
            stats.append(ComponentStat(
                name=f"CLAUDE.md ({label})",
                component_type="claude_md",
                estimated_chars=size,
                priority=100,  # 配置文件不自动删
                source=str(path),
                removable=False,
            ))
    return stats


def _scan_builtin() -> list[ComponentStat]:
    """估算内建固定开销（tool schemas, agent types 等）"""
    # 这些是 Claude Code 内建的，无法移除，只做统计
    builtin_tools = 25  # Read, Write, Edit, Bash, Grep, Glob, Agent, etc.
    agent_types = 35    # 内建 agent 类型数（保守估计）

    return [
        ComponentStat(
            name="builtin_tools",
            component_type="builtin",
            estimated_chars=builtin_tools * CHAR_ESTIMATES["mcp_tool_schema"],
            priority=100,
            source="claude-code-builtin",
            removable=False,
        ),
        ComponentStat(
            name="agent_types",
            component_type="builtin",
            estimated_chars=agent_types * CHAR_ESTIMATES["agent_type"],
            priority=100,
            source="claude-code-builtin",
            removable=False,
        ),
        ComponentStat(
            name="system_prompt_base",
            component_type="builtin",
            estimated_chars=15000,  # 基础系统提示约 15K chars
            priority=100,
            source="claude-code-builtin",
            removable=False,
        ),
    ]


def scan() -> BudgetReport:
    """
    全量扫描，生成预算报告。
    类比：cat /proc/meminfo + cat /sys/fs/cgroup/memory.stat
    """
    components = []
    components.extend(_scan_skills())
    components.extend(_scan_mcp_servers())
    components.extend(_scan_claude_md())
    components.extend(_scan_builtin())

    total = sum(c.estimated_chars for c in components)

    if total > CONTEXT_MAX_CHARS:
        pressure = "full"
    elif total > CONTEXT_HIGH_CHARS:
        pressure = "some"
    else:
        pressure = "none"

    return BudgetReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        total_chars=total,
        components=components,
        pressure=pressure,
    )


def reclaim(report: BudgetReport, target_chars: Optional[int] = None,
            dry_run: bool = True) -> BudgetReport:
    """
    在超限时自动回收低优先级组件。
    类比：kswapd + OOM killer

    策略：
      1. 按 priority 升序排列可回收组件
      2. 逐个"驱逐"（移除 skill 目录 / 从 .mcp.json 删除 server）
      3. 直到 total_chars < target_chars

    dry_run=True 时只生成计划不执行。
    """
    if target_chars is None:
        target_chars = report.high_chars

    if report.total_chars <= target_chars:
        report.actions_taken.append("no_action: within budget")
        return report

    # 按优先级排序：低优先级的先回收
    removable = [c for c in report.components if c.removable]
    removable.sort(key=lambda c: c.priority)

    current = report.total_chars
    actions = []

    for comp in removable:
        if current <= target_chars:
            break

        action = {
            "type": "reclaim",
            "name": comp.name,
            "component_type": comp.component_type,
            "chars_freed": comp.estimated_chars,
            "priority": comp.priority,
            "source": comp.source,
            "dry_run": dry_run,
        }

        if not dry_run:
            success = _evict(comp)
            action["executed"] = success
            if success:
                current -= comp.estimated_chars
        else:
            current -= comp.estimated_chars

        actions.append(action)

    report.actions_taken = actions
    report.total_chars = current

    if current > target_chars:
        report.pressure = "full"  # 回收后仍超限
    elif current > report.high_chars:
        report.pressure = "some"
    else:
        report.pressure = "none"

    return report


def _evict(comp: ComponentStat) -> bool:
    """
    执行驱逐操作。
    类比：OOM killer 选择并终止进程释放内存。
    """
    try:
        if comp.component_type == "skill":
            source_path = Path(comp.source)
            if source_path.exists() and source_path.is_dir():
                # 移到 /tmp 备份而非直接删除（安全）
                backup = Path("/tmp") / "aios-reclaimed-skills" / comp.name
                backup.parent.mkdir(parents=True, exist_ok=True)
                if backup.exists():
                    shutil.rmtree(backup)
                shutil.move(str(source_path), str(backup))
                return True

        elif comp.component_type == "mcp_server":
            # 从 .mcp.json 中移除 server
            mcp_file = Path(comp.source)
            if mcp_file.exists():
                data = json.loads(mcp_file.read_text())
                servers = data.get("mcpServers", {})
                # 从 name 提取真实 server 名
                srv_name = comp.name
                if srv_name in servers:
                    del servers[srv_name]
                    mcp_file.write_text(json.dumps(data, indent=2, ensure_ascii=False))
                    return True

    except Exception:
        return False

    return False


# ── CLI 入口 ──────────────────────────────────────────────────

def main():
    """命令行工具：python3 context_cgroup.py [scan|reclaim|reclaim-execute]"""
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "scan"

    report = scan()

    if cmd == "scan":
        d = report.to_dict()
        print(json.dumps(d, indent=2, ensure_ascii=False))
        # 简洁摘要
        print(f"\n── Context Budget: {report.total_chars:,} / {report.max_chars:,} chars "
              f"({report.usage_pct:.0f}%) pressure={report.pressure} ──")
        groups = d["components_by_type"]
        for t, g in sorted(groups.items(), key=lambda x: x[1]["total_chars"], reverse=True):
            print(f"  {t:15s} {g['count']:3d} items  {g['total_chars']:>8,} chars")

    elif cmd in ("reclaim", "reclaim-execute"):
        dry = (cmd == "reclaim")
        report = reclaim(report, dry_run=dry)
        d = report.to_dict()
        label = "DRY RUN" if dry else "EXECUTED"
        print(f"── Reclaim ({label}) ──")
        print(f"Budget after: {report.total_chars:,} / {report.max_chars:,} chars "
              f"({report.usage_pct:.0f}%) pressure={report.pressure}")
        for a in report.actions_taken:
            if isinstance(a, dict):
                status = "✓" if a.get("executed", True) else "✗"
                print(f"  {status} {a['component_type']:12s} {a['name']:40s} "
                      f"-{a['chars_freed']:>6,} chars  prio={a['priority']}")
            else:
                print(f"  {a}")
    else:
        print(f"Usage: python3 context_cgroup.py [scan|reclaim|reclaim-execute]")
        sys.exit(1)


if __name__ == "__main__":
    main()
