#!/usr/bin/env python3
"""
memory-os 工具函数

resolve_project_id() 按优先级推导 project_id，解决同名目录冲突问题。

历史数据迁移策略：不迁移。旧 chunk 使用纯目录名（如 "claude-workspace"），
新 chunk 使用带 label 前缀的格式（如 "git:abc123def456"）。两者不会匹配，
旧 chunk 自然隔离，不会被新查询召回。如需召回旧记录，用户可手动触发写入更新。

迭代150：vDSO 快速路径 — Project ID File Cache
OS 类比：Linux vDSO (2.6, 2003) — 将高频 syscall (gettimeofday/clock_gettime) 映射到
用户空间共享页面，避免每次调用都陷入内核态切换。
问题：每个 hook 调用（loader/extractor/retriever/writer）都是独立的 Python 进程。
  - module-level `import subprocess` 成本 ~17ms（每次冷启动）
  - subprocess.run(['git', 'remote', 'get-url', 'origin']) 成本 ~10-20ms（进程 fork）
  - 合计每次 hook 调用 ~30-40ms 消耗在 project_id 解析上（不包含任何实际工作）
修复：文件缓存 + lazy import
  - 先查 ~/.claude/memory-os/.project_id_cache.json（文件读取 ~0.1ms）
  - 缓存命中：直接返回，跳过 subprocess import + git 调用
  - 缓存未命中：lazy import subprocess，执行 git 解析，写入缓存
  - 缓存失效：.git/config mtime 变化时（分支切换/remote 变更）
预期效果：P50 从 32ms → ~10ms，P95 FULL 从 88ms → ~50ms
"""
import os
import hashlib
import json
from pathlib import Path


# ── vDSO 缓存文件（OS 类比：/proc/sys/kernel/vsyscall_32 的 vDSO 页面地址）──
# MEMORY_OS_DIR 路径复用 store_vfs 的逻辑，但不 import（避免循环+成本）
_MEMORY_OS_DIR = Path(os.environ["MEMORY_OS_DIR"]) if os.environ.get("MEMORY_OS_DIR") else Path.home() / ".claude" / "memory-os"
_PROJECT_ID_CACHE_FILE = _MEMORY_OS_DIR / ".project_id_cache.json"


def _cache_load() -> dict:
    """从缓存文件读取。失败静默返回空字典。O(1) = 文件读 + JSON 解析。"""
    try:
        if _PROJECT_ID_CACHE_FILE.exists():
            return json.loads(_PROJECT_ID_CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _cache_save(cwd_key: str, project_id: str, git_config_mtime: float = 0.0) -> None:
    """将解析结果写入缓存文件。失败静默（不影响主流程）。"""
    try:
        _MEMORY_OS_DIR.mkdir(parents=True, exist_ok=True)
        data = _cache_load()
        data[cwd_key] = {
            "project_id": project_id,
            "git_config_mtime": git_config_mtime,
        }
        _PROJECT_ID_CACHE_FILE.write_text(
            json.dumps(data, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass


def _git_config_mtime(cwd: str) -> float:
    """获取 .git/config 的 mtime（缓存失效判断）。不存在返回 0.0。"""
    try:
        # 从 cwd 向上查找 .git 目录
        p = Path(cwd).resolve()
        for _ in range(10):  # 最多向上 10 层
            git_config = p / ".git" / "config"
            if git_config.exists():
                return git_config.stat().st_mtime
            parent = p.parent
            if parent == p:
                break
            p = parent
    except Exception:
        pass
    return 0.0


def resolve_project_id(cwd: str = None) -> str:
    """
    按优先级推导 project_id：
    1. git remote origin URL hash（全局唯一，跨机器稳定）
    2. git rev-parse --show-toplevel hash（同机器唯一，子目录不影响）
    3. 绝对路径 hash（同机器唯一，路径变化则丢失历史）
    4. 目录名（兜底，可能冲突 —— 与旧版行为一致）

    返回 "{label}:{hash}" 格式以便调试，label 标识来源。
    SQL 查询使用完整字符串精确匹配，label 前缀不影响过滤逻辑。

    迭代150：vDSO 快速路径 — 文件缓存，避免重复 git subprocess。
    OS 类比：vDSO gettimeofday — 直接读内存映射，不陷入内核。
    """
    if cwd is None:
        cwd = os.environ.get("CLAUDE_CWD", os.getcwd())

    # ── 快速路径：文件缓存检查（OS 类比：vDSO 用户空间直接返回）──
    cwd_key = hashlib.sha256(cwd.encode()).hexdigest()[:16]
    current_mtime = _git_config_mtime(cwd)

    cache = _cache_load()
    entry = cache.get(cwd_key)
    if entry and isinstance(entry, dict):
        cached_mtime = entry.get("git_config_mtime", 0.0)
        # .git/config 未变化（分支/remote 没切换）→ 直接返回缓存值
        if current_mtime > 0 and abs(current_mtime - cached_mtime) < 1.0:
            return entry["project_id"]
        # non-git 目录（mtime=0）→ 也用缓存（abspath hash 不变）
        if current_mtime == 0 and cached_mtime == 0 and entry.get("project_id"):
            return entry["project_id"]

    # ── 慢速路径：subprocess（OS 类比：vDSO miss → syscall）──
    # lazy import subprocess — 只在缓存未命中时才 import（节省 ~17ms 冷启动）
    import subprocess

    # 优先级1：git remote URL（全局唯一 + 跨机器稳定）
    try:
        r = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True, text=True, cwd=cwd, timeout=3,
        )
        if r.returncode == 0 and r.stdout.strip():
            url = r.stdout.strip()
            h = hashlib.sha256(url.encode()).hexdigest()[:12]
            result = f"git:{h}"
            _cache_save(cwd_key, result, current_mtime)
            return result
    except Exception:
        pass

    # 优先级2：git root path hash（同机器唯一，子目录稳定）
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, cwd=cwd, timeout=3,
        )
        if r.returncode == 0 and r.stdout.strip():
            root = r.stdout.strip()
            h = hashlib.sha256(root.encode()).hexdigest()[:12]
            result = f"gitroot:{h}"
            _cache_save(cwd_key, result, current_mtime)
            return result
    except Exception:
        pass

    # 优先级3：绝对路径 hash（同机器唯一）
    try:
        abspath = os.path.abspath(cwd)
        h = hashlib.sha256(abspath.encode()).hexdigest()[:12]
        result = f"abspath:{h}"
        _cache_save(cwd_key, result, 0.0)  # non-git: mtime=0
        return result
    except Exception:
        pass

    # 兜底：目录名（与旧版行为一致，可能冲突）
    result = Path(cwd).name or "unknown"
    _cache_save(cwd_key, result, 0.0)
    return result
