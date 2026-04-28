"""
workspace_scanner.py — Filesystem-aware knowledge sensing (iter363)

OS 类比：inotify + /proc 文件系统读取
  - 项目文件 = 进程的 /proc/[pid]/maps — 描述地址空间布局
  - 扫描 = read_proc() — 一次性读取结构化状态

核心职责：
  扫描工程目录中的结构化配置文件（docker-compose.yml、.env、
  package.json、Makefile、pyproject.toml 等），
  提取端口、服务、命令等 facts，写入 workspace_files 表。

每次 SessionStart 时调用，仅对内容变更的文件重新扫描（hash 对比）。
"""
import json
import os
import re
from pathlib import Path
from typing import Optional

# ── Target files — 按重要性排序 ───────────────────────────────────────────────

# (glob_pattern, max_depth)
SCAN_TARGETS = [
    ("docker-compose.yml", 1),
    ("docker-compose.yaml", 1),
    ("docker-compose*.yml", 1),
    ("docker-compose*.yaml", 1),
    (".env", 1),
    (".env.local", 1),
    (".env.development", 1),
    (".env.production", 1),
    ("package.json", 1),
    ("pyproject.toml", 1),
    ("setup.cfg", 1),
    ("Makefile", 1),
    ("makefile", 1),
    ("GNUmakefile", 1),
    ("Procfile", 1),
    ("supervisord.conf", 1),
    ("nginx.conf", 2),
    ("*.nginx.conf", 2),
]


# ── Extractors ─────────────────────────────────────────────────────────────────

def _extract_docker_compose(content: str, file_path: str) -> list:
    """
    从 docker-compose.yml 提取服务端口和环境变量。
    不依赖 yaml 解析库（避免可选依赖），使用正则。
    """
    facts = []

    # 提取服务名和端口映射
    # 模式：  - "8080:80"  或  - 8080:80
    service_block_re = re.compile(r'^(\s{2})(\w[\w-]*):\s*$', re.MULTILINE)
    port_re = re.compile(r'-\s*["\']?(\d+):(\d+)["\']?')
    env_re = re.compile(r'-\s*(\w+)=(.+)')

    current_service = None
    in_ports = False
    in_env = False

    for line in content.splitlines():
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())

        # 服务名检测（2空格缩进 + 无 `-` 前缀）
        svc_match = re.match(r'^  (\w[\w-]*):\s*$', line)
        if svc_match and indent == 2:
            current_service = svc_match.group(1)
            in_ports = False
            in_env = False
            continue

        if stripped == 'ports:':
            in_ports = True
            in_env = False
            continue
        if stripped == 'environment:':
            in_env = True
            in_ports = False
            continue
        if stripped and not stripped.startswith('-') and ':' in stripped and indent <= 4:
            in_ports = False
            in_env = False

        if in_ports and current_service:
            pm = port_re.search(stripped)
            if pm:
                host_port, container_port = pm.group(1), pm.group(2)
                facts.append({
                    "type": "port",
                    "service": current_service,
                    "host_port": int(host_port),
                    "container_port": int(container_port),
                    "description": f"service '{current_service}' host:{host_port} → container:{container_port}",
                })

        if in_env and current_service:
            em = env_re.match(stripped)
            if em:
                key, val = em.group(1).strip(), em.group(2).strip()
                # 只记录 PORT/HOST/URL 类环境变量
                if any(kw in key.upper() for kw in ('PORT', 'HOST', 'URL', 'ADDR', 'BIND')):
                    facts.append({
                        "type": "env_var",
                        "service": current_service,
                        "key": key,
                        "value": val,
                        "description": f"{current_service}: {key}={val}",
                    })

    return facts


def _extract_env_file(content: str, file_path: str) -> list:
    """从 .env 文件提取端口和服务相关环境变量。"""
    facts = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if '=' not in line:
            continue
        key, _, val = line.partition('=')
        key = key.strip()
        val = val.strip().strip('"').strip("'")

        if any(kw in key.upper() for kw in ('PORT', 'HOST', 'URL', 'ADDR', 'BIND', 'ENDPOINT')):
            # 尝试提取纯数字端口
            port_num = None
            if val.isdigit():
                port_num = int(val)
            else:
                pm = re.search(r':(\d{2,5})(?:[/?]|$)', val)
                if pm:
                    port_num = int(pm.group(1))

            fact = {
                "type": "env_var",
                "key": key,
                "value": val,
                "description": f"{key}={val}",
            }
            if port_num:
                fact["type"] = "port"
                fact["port"] = port_num
                fact["description"] = f"{key}={val} (port {port_num})"
            facts.append(fact)

    return facts


def _extract_package_json(content: str, file_path: str) -> list:
    """从 package.json 提取 name、scripts（start/dev/serve 中的端口）。"""
    facts = []
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return facts

    name = data.get("name", "")
    if name:
        facts.append({"type": "project_name", "name": name, "description": f"project name: {name}"})

    scripts = data.get("scripts", {})
    for script_name, cmd in scripts.items():
        if script_name in ("start", "dev", "serve", "preview"):
            # 提取命令中的端口号 (--port 3000 / PORT=3000 / :3000)
            pm = re.search(r'(?:--port\s+|PORT=|:)(\d{2,5})', cmd)
            if pm:
                port = int(pm.group(1))
                facts.append({
                    "type": "port",
                    "script": script_name,
                    "port": port,
                    "command": cmd,
                    "description": f"npm run {script_name} → port {port}",
                })
            else:
                facts.append({
                    "type": "script",
                    "script": script_name,
                    "command": cmd,
                    "description": f"npm run {script_name}: {cmd}",
                })

    return facts


def _extract_makefile(content: str, file_path: str) -> list:
    """从 Makefile 提取常用 target 及其中的端口信息。"""
    facts = []
    interesting_targets = {"run", "start", "dev", "serve", "up", "docker-up", "backend", "frontend"}

    current_target = None
    for line in content.splitlines():
        # target 行
        tm = re.match(r'^([\w-]+)\s*:', line)
        if tm:
            current_target = tm.group(1).lower()
            continue

        if current_target in interesting_targets and line.startswith('\t'):
            cmd = line.strip()
            pm = re.search(r'(?:--port\s+|PORT=|:)(\d{2,5})', cmd)
            if pm:
                port = int(pm.group(1))
                facts.append({
                    "type": "port",
                    "target": current_target,
                    "port": port,
                    "command": cmd,
                    "description": f"make {current_target} → port {port}",
                })

    return facts


def _extract_pyproject(content: str, file_path: str) -> list:
    """从 pyproject.toml 提取项目名称和脚本入口。"""
    facts = []
    # 提取 name
    m = re.search(r'^\s*name\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
    if m:
        facts.append({"type": "project_name", "name": m.group(1),
                      "description": f"project name: {m.group(1)}"})

    # 提取 [tool.uvicorn] 或 port 配置
    pm = re.search(r'port\s*=\s*(\d+)', content, re.IGNORECASE)
    if pm:
        facts.append({"type": "port", "port": int(pm.group(1)),
                      "description": f"configured port: {pm.group(1)}"})

    return facts


def _extract_nginx(content: str, file_path: str) -> list:
    """从 nginx.conf 提取 listen 端口。"""
    facts = []
    for m in re.finditer(r'listen\s+(\d+)(?:\s+ssl)?', content):
        port = int(m.group(1))
        if port not in (80, 443):  # 标准端口不记录
            pass
        facts.append({
            "type": "port",
            "port": port,
            "source": "nginx",
            "description": f"nginx listen: {port}",
        })
    return facts


# ── Dispatcher ─────────────────────────────────────────────────────────────────

_EXTRACTORS = {
    "docker-compose": _extract_docker_compose,
    ".env": _extract_env_file,
    "package.json": _extract_package_json,
    "makefile": _extract_makefile,
    "pyproject.toml": _extract_pyproject,
    "nginx": _extract_nginx,
}


def extract_file_facts(file_path: str) -> list:
    """
    读取并解析单个文件，返回提取的 facts 列表。
    无法读取或识别的文件返回空列表。
    """
    path = Path(file_path)
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    name_lower = path.name.lower()

    if "docker-compose" in name_lower:
        return _extract_docker_compose(content, file_path)
    elif name_lower.startswith(".env"):
        return _extract_env_file(content, file_path)
    elif name_lower == "package.json":
        return _extract_package_json(content, file_path)
    elif name_lower in ("makefile", "gnumakefile"):
        return _extract_makefile(content, file_path)
    elif name_lower == "pyproject.toml":
        return _extract_pyproject(content, file_path)
    elif "nginx" in name_lower or name_lower.endswith(".nginx.conf"):
        return _extract_nginx(content, file_path)
    elif name_lower == "procfile":
        facts = []
        for line in content.splitlines():
            if ':' in line:
                proc, _, cmd = line.partition(':')
                pm = re.search(r'(?:--port\s+|PORT=\$?PORT|:)(\d{2,5})', cmd)
                if pm:
                    facts.append({
                        "type": "port", "process": proc.strip(),
                        "port": int(pm.group(1)),
                        "description": f"Procfile {proc.strip()} port: {pm.group(1)}",
                    })
        return facts

    return []


# ── Workspace scan entry point ─────────────────────────────────────────────────

def scan_workspace(cwd: str, max_files: int = 20) -> list:
    """
    扫描工作区目录，返回所有文件的 facts 列表（含文件路径信息）。

    Args:
        cwd: 工作区根目录
        max_files: 最多扫描文件数（防止大型仓库扫描过慢）

    Returns:
        [{"file_path": ..., "facts": [...]}, ...]
    """
    results = []
    base = Path(cwd)
    seen = set()
    count = 0

    for pattern, max_depth in SCAN_TARGETS:
        if count >= max_files:
            break
        # 仅扫描 1 或 2 层
        for depth in range(1, max_depth + 1):
            glob_prefix = "*/" * (depth - 1)
            for p in base.glob(glob_prefix + pattern):
                if p in seen:
                    continue
                seen.add(p)
                facts = extract_file_facts(str(p))
                if facts:
                    results.append({"file_path": str(p), "facts": facts})
                    count += 1
                if count >= max_files:
                    break

    return results


def scan_and_store(
    conn,  # sqlite3.Connection
    workspace_id: str,
    cwd: str,
    force: bool = False,
) -> dict:
    """
    扫描工作区文件并将 facts 写入 workspace_files 表。
    使用 upsert_workspace_file 的 hash 比对跳过未变更文件。

    Returns:
        {"scanned": N, "updated": N, "facts_total": N}
    """
    from store_workspace import upsert_workspace_file

    base = Path(cwd)
    seen = set()
    scanned = updated = facts_total = 0

    for pattern, max_depth in SCAN_TARGETS:
        for depth in range(1, max_depth + 1):
            glob_prefix = "*/" * (depth - 1)
            for p in base.glob(glob_prefix + pattern):
                if p in seen:
                    continue
                seen.add(p)
                scanned += 1

                if not force:
                    # 检查是否变更
                    from store_workspace import _file_hash
                    current_hash = _file_hash(str(p))
                    existing = conn.execute("""
                        SELECT file_hash FROM workspace_files
                        WHERE workspace_id = ? AND file_path = ?
                    """, (workspace_id, str(p))).fetchone()
                    if existing and existing[0] == current_hash:
                        continue

                facts = extract_file_facts(str(p))
                changed = upsert_workspace_file(conn, workspace_id, str(p), facts)
                if changed or force:
                    updated += 1
                    facts_total += len(facts)

    return {"scanned": scanned, "updated": updated, "facts_total": facts_total}
