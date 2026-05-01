#!/usr/bin/env python3
"""
memory-os writer — UserPromptSubmit hook
WAL 风格：每次用户提交时检查点写入任务状态到 L4(sqlite) + L5(latest.json)
不调用 LLM，纯规则提取，目标 < 50ms
"""
import sys
import os
import re
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

# 将 memory-os 根目录加入 path
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
from schema import MemoryChunk
from utils import resolve_project_id
from store import open_db, ensure_schema, insert_chunk, get_project_chunk_count, evict_lowest_retention, kswapd_scan, dmesg_log, DMESG_INFO, DMESG_DEBUG, DMESG_WARN, already_exists, merge_similar
from config import get as _sysctl  # 迭代27: sysctl Runtime Tunables

MEMORY_OS_DIR = Path.home() / ".claude" / "memory-os"
LATEST_JSON = MEMORY_OS_DIR / "latest.json"
STORE_DB = MEMORY_OS_DIR / "store.db"

# ── 迭代109: Context Pressure Detection ──────────────────────────────────────
# 对话 history 持续增长，接近 context 上限时：
#   1. 提前触发知识提取（防止有价值信息在 session 中随 compact 丢失）
#   2. 通过 additionalContext 提示用户开新会话
# OS 类比：Linux PSI (Pressure Stall Information) — 资源压力早期预警
#
# 系数校准（2026-04-22 实测）：
#   transcript JSONL 5.9MB → 实际 context 172k chars
#   系数 = 172680 / 5985418 = 0.0289
#   context 构成: tool_result 53% + tool_use_input 33% + text 15%
#   不能用 file_size × 简单系数，需要解析内容
_CTX_WARN_THRESHOLD = 0.60      # 60%: 软警告，建议新会话
_CTX_CRITICAL_THRESHOLD = 0.80  # 80%: 强警告，强烈建议新会话
_CTX_ASSUMED_MAX = 500_000      # chars，实测 Claude 4.x context ~200k tokens ≈ 500k chars
_CTX_PRESSURE_STATE = MEMORY_OS_DIR / "ctx_pressure_state.json"

# 迭代27：常量迁移至 config.py sysctl 注册表（运行时可调）
# 原硬编码：_sysctl("writer.debounce_secs")=300, _sysctl("extractor.chunk_quota")=200


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _detect_context_pressure(hook_input: dict) -> dict:
    """
    迭代109: Context Pressure Detection
    通过读取 transcript_path 估算当前会话 conversation history 大小，
    返回 pressure 状态用于触发提前提取和换会话提示。

    OS 类比：
      Linux PSI (Pressure Stall Information, kernel 4.20, 2019)
      在资源真正耗尽前提供 pressure 信号，让应用层有时间应对。
      memory.pressure_level: low/medium/critical → 对应 none/warn/critical

    返回: {
        "pressure": "none" | "warn" | "critical",
        "transcript_chars": int,
        "usage_pct": float,
        "session_id": str,
    }
    """
    result = {"pressure": "none", "transcript_chars": 0, "usage_pct": 0.0}

    transcript_path = hook_input.get("transcript_path", "")
    if not transcript_path:
        return result

    path = Path(transcript_path)
    if not path.exists():
        return result

    try:
        # 实际解析 transcript JSONL，提取真正进入 context 的内容
        # 系数校准（2026-04-22 实测）：file_size 系数不可靠（tool result 使文件膨胀14x）
        # 必须解析 content 字段，累计 text + tool_use input + tool_result text
        file_size = path.stat().st_size
        # 快速路径：文件 < 500KB，context 压力不可能触发（~15k chars）
        if file_size < 500_000:
            return result

        # 只读文件末尾 3MB（最近约 90 turns）
        # 原理：Claude compact 后老 turns 丢失，最近 3MB ≈ 当前实际 context 窗口
        # 实测：75MB 文件 tail 3MB → 25ms，145k chars，91 turns
        TAIL_BYTES = 3_000_000
        read_offset = max(0, file_size - TAIL_BYTES)
        ctx_chars = 0
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            if read_offset > 0:
                f.seek(read_offset)
                f.readline()  # 跳过可能截断的首行
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                msg = entry.get("message", {})
                if not isinstance(msg, dict):
                    continue
                content = msg.get("content", "")
                if isinstance(content, str):
                    ctx_chars += len(content)
                elif isinstance(content, list):
                    for c in content:
                        if not isinstance(c, dict):
                            continue
                        t = c.get("type", "")
                        if t == "text":
                            ctx_chars += len(c.get("text", ""))
                        elif t == "tool_use":
                            ctx_chars += len(json.dumps(c.get("input", {})))
                        elif t == "tool_result":
                            rc = c.get("content", "")
                            if isinstance(rc, str):
                                ctx_chars += len(rc)
                            elif isinstance(rc, list):
                                for item in rc:
                                    if isinstance(item, dict) and item.get("type") == "text":
                                        ctx_chars += len(item.get("text", ""))

        usage_pct = ctx_chars / _CTX_ASSUMED_MAX
        result["transcript_chars"] = ctx_chars
        result["usage_pct"] = usage_pct

        if usage_pct >= _CTX_CRITICAL_THRESHOLD:
            result["pressure"] = "critical"
        elif usage_pct >= _CTX_WARN_THRESHOLD:
            result["pressure"] = "warn"

        # 持久化压力状态（iter259: per-session 文件，避免多 agent 互相覆盖）
        # OS 类比：/proc/PID/status — 每进程独立文件，不同进程间不干扰
        try:
            _sid = hook_input.get("session_id", "")
            _sid_tag = _sid[:16] if (_sid and _sid != "unknown") else ""
            if _sid_tag:
                _ctx_pressure_file = MEMORY_OS_DIR / f"ctx_pressure_state.{_sid_tag}.json"
            else:
                _ctx_pressure_file = _CTX_PRESSURE_STATE  # 向后兼容
            _ctx_pressure_file.write_text(json.dumps({
                "pressure": result["pressure"],
                "usage_pct": round(usage_pct, 3),
                "transcript_chars": ctx_chars,
                "updated_at": _now_iso(),
                "session_id": _sid,
            }, ensure_ascii=False))
        except Exception:
            pass

    except Exception:
        pass

    return result


def _build_context_pressure_notice(pressure_info: dict, project: str,
                                   session_id: str) -> str | None:
    """
    迭代109: 构造换会话提示，通过 additionalContext 注入到 Claude。
    只在 warn/critical 时返回提示字符串，otherwise None。
    """
    pressure = pressure_info.get("pressure", "none")
    if pressure == "none":
        return None

    usage_pct = pressure_info.get("usage_pct", 0) * 100
    chars = pressure_info.get("transcript_chars", 0)

    if pressure == "critical":
        return (
            f"【⚠️ Context 压力：{usage_pct:.0f}%（严重）】"
            f" 当前会话 history 已达约 {chars:,} chars，接近上限。"
            f" memory-os 已自动提取关键知识。"
            f" 建议：本轮完成后立即开新会话（/new 或 Ctrl+R），"
            f" 下一会话将自动恢复上下文。"
        )
    else:  # warn
        return (
            f"【⚡ Context 压力：{usage_pct:.0f}%（警告）】"
            f" 建议在 1-2 轮内开新会话，memory-os 已记录当前进度。"
        )


def _get_session_id(hook_input: dict = None) -> str:
    """迭代66：优先从 hook stdin 获取 session_id（权威来源），fallback 到环境变量"""
    if hook_input and hook_input.get("session_id"):
        return hook_input["session_id"]
    return os.environ.get("CLAUDE_SESSION_ID", "unknown")


def _should_skip_debounce(current_tasks: list = None, next_tasks: list = None) -> bool:
    """仅在无任务变化时才防抖；有新任务内容时强制写入。"""
    if not LATEST_JSON.exists():
        return False
    try:
        mtime = LATEST_JSON.stat().st_mtime
        if (time.time() - mtime) >= _sysctl("writer.debounce_secs"):
            return False  # 超过防抖窗口，正常写入
        # 在防抖窗口内：检查任务是否有变化
        if current_tasks or next_tasks:
            try:
                existing = json.loads(LATEST_JSON.read_text(encoding="utf-8"))
                existing_content = existing.get("content", "")
                # 如果有任务且和现有内容不同，跳过防抖
                new_has_tasks = bool(current_tasks or next_tasks)
                existing_has_no_tasks = "无活跃任务" in existing_content
                if new_has_tasks and existing_has_no_tasks:
                    return False
                # 比较任务列表是否变化
                import re
                existing_tasks = set(re.findall(r"^- (.+)$", existing_content, re.MULTILINE))
                new_tasks = set(current_tasks or []) | set(next_tasks or [])
                if new_tasks != existing_tasks:
                    return False
            except Exception:
                pass
        return True
    except Exception:
        return False


def _parse_task_list(hook_input: dict) -> tuple[list, list]:
    current_tasks = []
    next_tasks = []

    task_list = hook_input.get("task_list") or hook_input.get("tasks") or []
    if isinstance(task_list, str):
        try:
            task_list = json.loads(task_list)
        except Exception:
            task_list = []

    for task in task_list:
        if not isinstance(task, dict):
            continue
        status = task.get("status", "")
        subject = task.get("subject") or task.get("title") or task.get("description") or ""
        if not subject:
            continue
        if status == "in_progress":
            current_tasks.append(subject)
        elif status == "pending":
            next_tasks.append(subject)

    return current_tasks, next_tasks


def _extract_excluded_paths(current_tasks: list, next_tasks: list) -> list:
    keywords = ["排除", "放弃", "改用", "不用", "已废弃", "deprecated", "excluded"]
    excluded = []
    for text in current_tasks + next_tasks:
        lower = text.lower()
        for kw in keywords:
            if kw in lower:
                excluded.append(text.strip())
                break
    return excluded


def _extract_prompt_topic(prompt: str) -> str:
    """
    从 prompt 提取话题摘要（用于无 task_list 时的轻量写入）。
    取首句非空实质行，截断到 100 字。
    迭代88：质量门当 — 拦截模糊/短指令，避免写入无召回价值的 prompt_context。
    """
    if not prompt or len(prompt.strip()) < 10:
        return ""
    # 过滤1：sleep 子会话的系统提示 — 不是用户知识，不应入库
    _SLEEP_MARKERS = (
        "你正在帮 Claude 整理", "你是 Claude，正处于深度睡眠",
        "你是 Claude，正在进行类似 REM", "类比大脑默认模式网络",
        "慢波睡眠", "深度内省", "自由探索阶段",
    )
    if any(m in prompt for m in _SLEEP_MARKERS):
        return ""
    # 过滤2：纯疑问句（用户的问题是请求，不是知识）
    # 识别：以疑问词开头且无技术路径/代码块 → 不写入
    _stripped = prompt.strip()
    _QUESTION_STARTERS = ("为什么", "怎么", "如何", "什么是", "能否", "是否", "有没有", "请问")
    _is_pure_question = (
        any(_stripped.startswith(q) for q in _QUESTION_STARTERS)
        and len(_stripped) < 60
        and not re.search(r'[`/\\]|\.py|\.js|hook|config|store', _stripped)
    )
    # 过滤3（iter108）：纯 meta 操作请求 — "看下/查看/分析 + 系统名称" 不是知识
    # 例："看下目前aios还有什么问题"、"分析一下现在的状态"
    # 识别：以操作动词开头 + 长度 < 30 + 无技术锚点（文件/数字/代码标识符）
    _META_STARTERS = ("看下", "查看", "分析", "看看", "检查", "帮我看", "帮我分析", "帮我查")
    _is_meta_request = (
        any(_stripped.startswith(m) for m in _META_STARTERS)
        and len(_stripped) < 30
        and not re.search(
            r'[`/\\]|\.py|\.js|\d+(?:%|ms|次|条)'
            r'|(?:iter\d*|迭代|benchmark|性能|错误|bug|crash|报错|配置|接口|函数|模块|测试|评测)',
            _stripped, re.IGNORECASE)
    )
    if _is_meta_request:
        return ""
    for line in prompt.strip().splitlines():
        stripped = line.strip()
        if len(stripped) >= 5 and not stripped.startswith(('#', '```', '<!--', '---')):
            clean = re.sub(r'\*{1,3}|`{1,3}', '', stripped).strip()
            # ── 迭代88：prompt_context 质量门当 ──
            # OS 类比：admission control — 入口处拦截低价值请求，避免消耗后端资源
            # 拦截条件（全部满足 = 无价值）：
            #   1. 无技术关键词（文件路径/代码标识符/数字度量/技术术语）
            #   2. 中文实词少于 3 个（去掉助词/连词/代词后）
            has_tech = bool(
                re.search(r'[\w./]+\.(?:py|js|ts|json|db|sql|yaml|toml|sh|md)\b', clean)
                or re.search(r'`[^`]+`', clean)
                or re.search(r'\d+(?:\.\d+)?(?:%|ms|KB|MB|GB|次|条|个|行)', clean)
                or re.search(r'(?:函数|模块|类|文件|接口|表|字段|索引|配置|迭代|hook|API|DB|SQL|测试|评测|benchmark)', clean)
            )
            # 去掉常见虚词+日常动词后统计实际内容字数
            content_chars = re.sub(
                r'[，。！？、：；\u201c\u201d\u2018\u2019（）\s\d\W]|'
                r'(?:的|了|吧|啊|呢|嘛|吗|呗|哦|哈|是|在|和|与|或|也|都|就|而|且|不|要|能|会|把|被|让|给|从|对|到|等|着|过|来|去|'
                r'这|那|有|没|什么|怎么|为什么|如何|还|才|又|再|已|更|很|最|比|较|该|可|以|应|该|需|看|想|做|用|说|知道|觉得|'
                r'继续|推进|开始|确认|同意|全部|所有|一下|一些|这样|那样|系统|区别|效果|成果|现在|然后|之后|以后)',
                '', clean
            )
            if not has_tech and len(content_chars) < 6:
                return ""  # 太模糊，不写入
            if _is_pure_question:
                return ""  # 纯疑问句是请求而非知识
            return clean[:100]
    return ""


def _calc_importance(current_tasks: list, next_tasks: list, excluded_paths: list) -> float:
    """根据内容丰富度动态计算 importance（0.5-0.9）。
    基准 0.5；有当前任务 +0.2；有待执行 +0.1；有排除路径 +0.1；上限 0.9。
    """
    score = 0.5
    if current_tasks:
        score += 0.2
    if next_tasks:
        score += 0.1
    if excluded_paths:
        score += 0.1
    return min(score, 0.9)


def _build_chunk(current_tasks: list, next_tasks: list,
                 excluded_paths: list, project: str, session_id: str) -> MemoryChunk:
    content_parts = []
    if current_tasks:
        content_parts.append("当前任务：\n" + "\n".join(f"- {t}" for t in current_tasks))
    if next_tasks:
        content_parts.append("待执行：\n" + "\n".join(f"- {t}" for t in next_tasks))
    if excluded_paths:
        content_parts.append("已排除路径：\n" + "\n".join(f"- {e}" for e in excluded_paths))

    content = "\n\n".join(content_parts) if content_parts else "（无活跃任务）"

    summary_parts = []
    if current_tasks:
        summary_parts.append(f"正在：{current_tasks[0]}")
    if next_tasks:
        summary_parts.append(f"下步：{next_tasks[0]}")
    summary = "；".join(summary_parts)[:100]

    importance = _calc_importance(current_tasks, next_tasks, excluded_paths)

    return MemoryChunk(
        project=project,
        source_session=session_id,
        chunk_type="task_state",
        content=content,
        summary=summary,
        tags=["task_state", project],
        importance=importance,
        retrievability=0.2,
    )


def _write_latest_json(chunk: MemoryChunk) -> None:
    MEMORY_OS_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_JSON.write_text(chunk.to_json(), encoding="utf-8")


def _write_sqlite(chunk: MemoryChunk) -> None:
    """
    v10 迭代30：kswapd 水位线预淘汰（替代迭代25 的硬 OOM handler）。
    OS 类比：__alloc_pages_slowpath → 先唤醒 kswapd 检查水位，再分配。
    """
    conn = open_db()
    try:
        ensure_schema(conn)
        # 迭代30 kswapd：写入前检查水位
        ksw = kswapd_scan(conn, chunk.project, incoming_count=1)
        if ksw["evicted_count"] > 0:
            conn.commit()
            dmesg_log(conn, DMESG_WARN, "writer",
                      f"kswapd zone={ksw['zone']}: evicted={ksw['evicted_count']} stale={ksw['stale_evicted']} watermark={ksw['watermark_pct']}%",
                      session_id=chunk.source_session, project=chunk.project)
        insert_chunk(conn, chunk.to_dict())
        # 迭代144：Auto OOM Adj — 根据 chunk_type 自动设置 oom_adj
        # OS 类比：内核根据进程类型自动设置 oom_score_adj（systemd 给 journald 设 -900）
        # 根因：config.py 定义了 oom.auto_disposable_ctx=500 和 oom.auto_protect_quant=-500，
        #       但没有代码在写入时应用这些值，导致 prompt_context 的 oom_adj 始终为 0，
        #       不会比 decision/reasoning_chain 更优先淘汰——设计意图未落地。
        # 修复：insert_chunk 后立即按 chunk_type 设置 oom_adj（UPDATE 仅改单列，代价极低）
        try:
            from config import get as _cfg
            auto_oom = None
            if chunk.chunk_type == "prompt_context":
                auto_oom = _cfg("oom.auto_disposable_ctx")  # 默认 500：优先淘汰
            elif chunk.chunk_type == "quantitative_evidence":
                auto_oom = _cfg("oom.auto_protect_quant")   # 默认 -500：高保护
            if auto_oom is not None:
                conn.execute(
                    "UPDATE memory_chunks SET oom_adj=? WHERE id=? AND COALESCE(oom_adj,0)=0",
                    (auto_oom, chunk.id),
                )
        except Exception:
            pass  # auto oom_adj 分配失败不影响主写入流程
        # 迭代29 dmesg：写入记录
        dmesg_log(conn, DMESG_DEBUG, "writer",
                  f"type={chunk.chunk_type} imp={chunk.importance:.2f} summary={chunk.summary[:60]}",
                  session_id=chunk.source_session, project=chunk.project)
        conn.commit()
    finally:
        conn.close()


def _sync_scheduler_tasks(current_tasks, next_tasks, project, session_id):
    """
    迭代87：将 TodoWrite 任务同步到 scheduler_tasks 表。
    OS 类比：进程 fork 时在内核 task_struct 表中登记。
    幂等设计：按 (project, task_name, status) 去重。
    """
    try:
        from store_core import (open_db, ensure_schema, sched_create_task,
                                sched_update_task, sched_get_tasks)
        conn = open_db()
        ensure_schema(conn)

        existing = sched_get_tasks(conn, project, limit=200)
        existing_map = {(t["task_name"], t["status"]): t for t in existing}
        existing_names = {t["task_name"] for t in existing}

        # 同步 running 任务（current_tasks = in_progress）
        for i, name in enumerate(current_tasks):
            if name in existing_names:
                for t in existing:
                    if t["task_name"] == name and t["status"] != "running":
                        sched_update_task(conn, t["id"], status="running")
            else:
                sched_create_task(conn, project, session_id, name,
                                  priority=50 + (len(current_tasks) - i))

        # 同步 pending 任务
        for i, name in enumerate(next_tasks):
            if name not in existing_names:
                sched_create_task(conn, project, session_id, name,
                                  priority=30 + (len(next_tasks) - i))

        conn.close()
    except Exception:
        pass  # async hook, 不阻塞用户


def main():
    try:
        raw = sys.stdin.read()
        hook_input = json.loads(raw) if raw.strip() else {}
    except Exception:
        hook_input = {}

    project = resolve_project_id()
    session_id = _get_session_id(hook_input)
    current_tasks, next_tasks = _parse_task_list(hook_input)

    # ── 迭代109: Context Pressure Detection ──────────────────────────────
    # 在写入前检测 context 压力：
    #   none   → 正常 debounce 逻辑
    #   warn   → 输出提示，但不绕过 debounce
    #   critical → 绕过 debounce，强制写入 + 强提示
    pressure_info = _detect_context_pressure(hook_input)
    pressure_notice = _build_context_pressure_notice(pressure_info, project, session_id)
    is_critical = pressure_info["pressure"] == "critical"

    if not is_critical and _should_skip_debounce(current_tasks, next_tasks):
        # debounce 触发，但有压力通知时仍需输出
        if pressure_notice:
            print(json.dumps({
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": pressure_notice,
                }
            }, ensure_ascii=False))
        sys.exit(0)

    excluded_paths = _extract_excluded_paths(current_tasks, next_tasks)

    chunk = _build_chunk(current_tasks, next_tasks, excluded_paths, project, session_id)
    _write_latest_json(chunk)

    if current_tasks or next_tasks:
        _write_sqlite(chunk)
        # 迭代87：同步任务到 scheduler_tasks 表
        _sync_scheduler_tasks(current_tasks, next_tasks, project, session_id)
    else:
        # v4 迭代13：无 task_list 时从 prompt 提取话题写入轻量 chunk
        # v5 迭代58：KSM 去重 — 相同/相似 prompt_context 不重复写入
        prompt = hook_input.get("prompt", "")
        # 迭代93: 检测否定反馈信号，降低关联知识 importance
        _process_negative_feedback(prompt, project, session_id)
        # 迭代100: ECC 验证反馈闭环 — 更新 confidence_score
        _capture_verification_feedback(prompt, project, session_id)
        # Per-turn Citation Detection — 每轮反馈信号（比 Stop hook 快 1-3 轮）
        # OS 类比：PMU per-instruction branch feedback vs post-epoch batch update
        # 从 transcript 读取上一条 assistant reply，检测哪些注入 chunk 被实际引用
        try:
            _transcript_path = hook_input.get("transcript_path", "")
            if _transcript_path:
                from pathlib import Path as _Path
                import json as _json
                _tp = _Path(_transcript_path)
                if _tp.exists():
                    # 读末尾 32KB，找最近的 assistant message
                    _size = _tp.stat().st_size
                    _offset = max(0, _size - 32768)
                    with open(str(_tp), 'r', encoding='utf-8', errors='replace') as _f:
                        if _offset > 0:
                            _f.seek(_offset)
                        _tail = _f.read()
                    # transcript 是 JSONL，每行一个消息，找最后一条 assistant 行
                    _last_assistant = ""
                    for _line in reversed(_tail.splitlines()):
                        _line = _line.strip()
                        if not _line:
                            continue
                        try:
                            _msg = _json.loads(_line)
                            # 处理两种格式：{role, content} 或 {type: message, message: {role, content}}
                            _role = _msg.get("role", "") or (_msg.get("message") or {}).get("role", "")
                            if _role == "assistant":
                                _content = _msg.get("content", "") or (_msg.get("message") or {}).get("content", "")
                                if isinstance(_content, list):
                                    _texts = [c.get("text", "") for c in _content if isinstance(c, dict) and c.get("type") == "text"]
                                    _last_assistant = " ".join(_texts)
                                elif isinstance(_content, str):
                                    _last_assistant = _content
                                if _last_assistant:
                                    break
                        except (_json.JSONDecodeError, AttributeError, TypeError):
                            continue
                    if _last_assistant and len(_last_assistant) >= 20:
                        from tools.citation_detector import run_citation_detection
                        run_citation_detection(_last_assistant, project, session_id)
        except Exception:
            pass  # per-turn citation 失败不影响主流程
        # 迭代91: 检测长期目标表达并持久化
        _detect_and_persist_goal(prompt, project, session_id)
        topic = _extract_prompt_topic(prompt)
        if topic:
            # 迭代58 KSM dedup：exact match → skip, similar → merge, else → insert
            conn = open_db()
            try:
                ensure_schema(conn)
                if already_exists(conn, topic, chunk_type="prompt_context"):
                    dmesg_log(conn, DMESG_DEBUG, "writer",
                              f"ksm_skip: prompt_context exact dup '{topic[:40]}'",
                              session_id=session_id, project=project)
                    conn.commit()
                elif merge_similar(conn, topic, "prompt_context", 0.5):
                    dmesg_log(conn, DMESG_DEBUG, "writer",
                              f"ksm_merge: prompt_context similar '{topic[:40]}'",
                              session_id=session_id, project=project)
                    conn.commit()
                else:
                    conn.close()
                    conn = None
                    ctx_chunk = MemoryChunk(
                        project=project,
                        source_session=session_id,
                        chunk_type="prompt_context",
                        content=f"用户话题：{topic}",
                        summary=topic,
                        tags=["prompt_context", project],
                        importance=0.5,
                        retrievability=0.15,
                    )
                    _write_sqlite(ctx_chunk)
            finally:
                if conn:
                    conn.close()

    # ── 迭代109: 输出 additionalContext（压力通知）──────────────────────
    if pressure_notice:
        try:
            dmesg_log(open_db(), DMESG_WARN, "writer",
                      f"ctx_pressure: {pressure_info['pressure']} "
                      f"usage={pressure_info['usage_pct']*100:.0f}% "
                      f"chars={pressure_info['transcript_chars']:,}",
                      session_id=session_id, project=project)
        except Exception:
            pass
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": pressure_notice,
            }
        }, ensure_ascii=False))

    sys.exit(0)


def _detect_and_persist_goal(prompt: str, project: str, session_id: str) -> bool:
    """
    迭代91: Goal Persistence — 跨会话目标检测和持久化

    OS 类比：Linux cgroup 目标配置文件 — 跨会话的资源配额和目标声明持久保存在磁盘，
    进程重启后无需重新配置，系统自动恢复目标约束。

    检测用户 prompt 中的长期目标表达，写入 goals 表持久化。
    目标识别信号：
    - "最终想要/目标是/希望/打算..."
    - "通过...来..."
    - "解决...问题"
    """
    import hashlib
    from datetime import datetime, timezone

    # 目标信号模式
    GOAL_PATTERNS = [
        r"最终(想|是|要|希望|目的)",
        r"目标(是|：|:)",
        r"想(通过|借助|利用).+(来|实现|解决|完成)",
        r"(解决|修复|改善|优化|增强).+(问题|痛点|体验|能力)",
        r"(让|使|帮助|支持).+(能够|可以|具备)",
    ]

    import re
    matched = any(re.search(p, prompt) for p in GOAL_PATTERNS)
    if not matched:
        return False

    # 提取目标摘要（取 prompt 前 100 字作为摘要）
    goal_summary = prompt[:100].strip()
    goal_id = f"goal-{hashlib.md5(goal_summary.encode()).hexdigest()[:12]}"
    now = datetime.now(timezone.utc).isoformat()

    try:
        conn = open_db()
        ensure_schema(conn)
        # 检查是否已存在
        existing = conn.execute(
            "SELECT id FROM goals WHERE id = ?", [goal_id]
        ).fetchone()
        # iter_multiagent P2：添加 last_progress_session 列（幂等）
        try:
            conn.execute("ALTER TABLE goals ADD COLUMN last_progress_session TEXT DEFAULT ''")
        except Exception:
            pass  # 列已存在时静默跳过
        if not existing:
            conn.execute("""
                INSERT INTO goals (id, title, description, status, progress,
                                   created_at, updated_at, project, tags,
                                   last_progress_session)
                VALUES (?, ?, ?, 'active', 0.0, ?, ?, ?, ?, '')
            """, [goal_id, goal_summary, prompt[:500], now, now, project,
                  json.dumps(["user_goal", project])])
            dmesg_log(conn, DMESG_INFO, "writer",
                      f"goal_persist: new '{goal_summary[:50]}'",
                      session_id=session_id, project=project)
        else:
            # 更新 updated_at
            conn.execute("UPDATE goals SET updated_at=? WHERE id=?", [now, goal_id])
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False


def _process_negative_feedback(prompt: str, project: str, session_id: str) -> bool:
    """
    迭代93: Feedback Loop — 否定反馈信号处理

    OS 类比：Linux 进程优先级惩罚（nice +19）— 消耗 CPU 超限的进程被调度器降级，
    防止其继续占用资源。这里"占用资源"是指占据 Top-K 检索结果位置。

    检测用户 prompt 中的否定/纠错信号，对关联的上一条知识降低 importance，
    防止错误知识继续被高频召回。

    否定信号：
    - "不对"、"不是"、"不行"、"错了"、"重新"、"重来"
    - "这不是"、"不是这个"、"不是我想要的"
    - "no"、"wrong"、"not right"、"redo"、"again"
    """
    NEGATIVE_PATTERNS = [
        r"^(不对|不是|不行|错了|错误|重新|重来|再来)",
        r"(不是(我想|我要|这个)|这不(是|对|行))",
        r"^(no[,\s]|wrong|not right|redo|again\b)",
        r"(你理解错了|你搞错了|不对的|答非所问)",
    ]
    import re, hashlib
    from datetime import datetime, timezone

    matched = any(re.search(p, prompt.strip(), re.IGNORECASE) for p in NEGATIVE_PATTERNS)
    if not matched:
        return False

    try:
        conn = open_db()
        ensure_schema(conn)
        # 找到最近的 prompt_context chunk（上一轮的话题）
        recent = conn.execute(
            """SELECT id, summary, importance FROM memory_chunks
               WHERE project = ? AND chunk_type = 'prompt_context'
               ORDER BY last_accessed DESC, created_at DESC LIMIT 1""",
            [project]
        ).fetchone()

        if recent:
            chunk_id, summary, cur_imp = recent
            # 降低 importance（最低 0.1）
            new_imp = max(0.1, cur_imp * 0.7)
            conn.execute(
                "UPDATE memory_chunks SET importance=?, oom_adj=MAX(oom_adj, 200) WHERE id=?",
                [new_imp, chunk_id]
            )
            dmesg_log(conn, DMESG_INFO, "writer",
                      f"neg_feedback: imp {cur_imp:.2f}→{new_imp:.2f} '{summary[:40]}'",
                      session_id=session_id, project=project)
            conn.commit()

        conn.close()
        return True
    except Exception:
        return False


def _capture_verification_feedback(prompt: str, project: str, session_id: str) -> bool:
    """
    迭代100：Verification Feedback Loop — 隐式反馈捕捉。
    OS 类比：ECC error reporting — 硬件检测到位翻转后更新 CE/UE 计数器，
    操作系统据此决定是否 retire 该内存页。

    检测用户对上一轮检索结果的评价（正面/负面/部分），
    更新对应 chunks 的 confidence_score 和 verification_status。
    """
    import re as _re
    POSITIVE = [r"(很好|正确|对的|继续|谢谢|解决了|就是这|太好了|完全同意|有用)"]
    NEGATIVE = [r"^(不对|错了|不是|不行|你搞错|答非所问|错误)",
                r"^(no[,.\s]|wrong|not right|incorrect)"]
    PARTIAL = [r"(部分对|差不多|不完全|不太对|有些|勉强)"]

    feedback_type = None
    p = prompt.strip()
    if any(_re.search(pat, p, _re.I) for pat in POSITIVE):
        feedback_type = "useful"
    elif any(_re.search(pat, p, _re.I) for pat in NEGATIVE):
        feedback_type = "wrong"
    elif any(_re.search(pat, p, _re.I) for pat in PARTIAL):
        feedback_type = "partial"

    if not feedback_type:
        return False

    try:
        conn = open_db()
        ensure_schema(conn)
        # 找到最近一条 recall_trace（上一轮注入的检索结果）
        trace = conn.execute(
            """SELECT id, top_k_json FROM recall_traces
               WHERE project=? ORDER BY timestamp DESC LIMIT 1""",
            [project]).fetchone()
        if not trace:
            conn.close()
            return False

        trace_id, top_k_json = trace
        now = datetime.now(timezone.utc).isoformat()

        # 更新 recall_traces 反馈字段
        conn.execute(
            "UPDATE recall_traces SET user_feedback=?, feedback_ts=? WHERE id=?",
            [feedback_type, now, trace_id])

        # 解析被注入的 chunk IDs，更新其 confidence
        try:
            top_k = json.loads(top_k_json) if isinstance(top_k_json, str) else (top_k_json or [])
            chunk_ids = [item["id"] for item in top_k if isinstance(item, dict) and "id" in item]

            delta_map = {"useful": 0.10, "wrong": -0.25, "partial": -0.08}
            status_map = {"useful": "verified", "wrong": "disputed", "partial": None}
            delta = delta_map.get(feedback_type, 0)
            new_status = status_map.get(feedback_type)

            for cid in chunk_ids[:10]:  # 最多影响 10 个 chunks
                from store_vfs import update_confidence
                update_confidence(conn, cid, delta, f"feedback_{feedback_type}",
                                  verification_status=new_status)
        except (json.JSONDecodeError, TypeError, KeyError):
            pass

        dmesg_log(conn, DMESG_INFO, "writer",
                  f"verification_feedback: {feedback_type} on trace {trace_id[:8]}",
                  session_id=session_id, project=project)
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False


if __name__ == "__main__":
    main()
