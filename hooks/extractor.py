#!/usr/bin/env python3
"""
memory-os extractor — Stop hook
从 last_assistant_message 提取决策链 chunk + 对话摘要

v5 策略（迭代39 升级）：
- v4 全部能力保留（决策/排除/推理链/对比/因果/量化/conversation_summary）
- 新增 COW 预扫描：先做 O(1) 快速检测，无信号词时跳过完整提取
  （OS 类比：Linux fork() COW — 只在真正写入时才复制页面）
- 目标 < 150ms（不调用 LLM），无信号消息 < 0.5ms
"""
import sys
import os
import re
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
from schema import MemoryChunk
from utils import resolve_project_id
from store import open_db, ensure_schema, insert_chunk, already_exists, merge_similar, get_project_chunk_count, evict_lowest_retention, kswapd_scan, dmesg_log, DMESG_INFO, DMESG_WARN, DMESG_DEBUG, madvise_write, set_oom_adj, OOM_ADJ_PROTECTED, OOM_ADJ_PREFER, cgroup_throttle_check, checkpoint_dump, checkpoint_collect_hits, aimd_window, pin_chunk
from config import get as _sysctl  # 迭代27: sysctl Runtime Tunables

MEMORY_OS_DIR = Path.home() / ".claude" / "memory-os"
STORE_DB = MEMORY_OS_DIR / "store.db"

# 迭代27：常量迁移至 config.py sysctl 注册表（运行时可调）
# 原硬编码：_sysctl("extractor.min_length")=10, _sysctl("extractor.max_summary")=120, _sysctl("extractor.chunk_quota")=200

# ── 决策信号词 ──────────────────────────────────────────────
DECISION_SIGNALS = [
    r'(?:选择|决定|采用|推荐|最终方案|方案选定)[：:]\s*(.{10,120})',
    r'(?:推断|结论|因此)[：:]\s*(.{10,120})',
    r'(?:选择|决定)\s*(.{5,60})\s*(?:而非|不选|放弃)',
    r'\*\*(?:推荐|选择|决定|结论)[：:]?\*\*\s*(.{10,100})',
    r'→\s*(.{5,80})\s*(?:是正确|更合适|最优|更好)',
    # 方向性结论
    r'(?:方向\s*[A-Za-z])[：:]\s*(.{10,120})',
    r'(?:核心洞察|关键洞察)[：:]\s*(.{10,120})',
    # 英文
    r'(?:decided?|chosen?|adopted?|recommended?)[：:]\s*(.{10,100})',
    r'(?:conclusion|therefore)[：:]\s*(.{10,100})',
]

# ── 排除路径信号词 ────────────────────────────────────────────
EXCLUDED_SIGNALS = [
    r'(?:不用|放弃|排除|废弃|不推荐|不选|跳过)\s*(.{3,60})',
    r'(.{3,60})\s*(?:不适合|效果差|有问题|不可行|被放弃)',
    r'(?:deprecated|abandoned|rejected|skipped)[：:]?\s*(.{3,60})',
    r'\*\*不推荐\*\*[：:]?\s*(.{10,80})',
    # 带原因的排除
    r'(?:排除|不选)\s*(.{3,60})\s*[，,]\s*(?:因为|原因)',
]

# ── 推理链信号 ──────────────────────────────────────────────
REASONING_SIGNALS = [
    r'(?:根本原因|root cause)[：:]\s*(.{10,120})',
    r'(?:为什么|Why)[：:]\s*(.{10,120})',
    r'(?:核心问题|核心差距|真正差距)[：:]\s*(.{10,120})',
    r'(?:第一性原理)[：:]\s*(.{10,120})',
    # 扩充：更多推理标记（覆盖实际内容中高频但被漏掉的模式）
    r'(?:根本问题|根本差距|本质问题|根因)[在是]?[：:在于]?\s*(.{10,120})',
    r'(?:原因[：:是]|问题在于)[：:]?\s*(.{10,120})',
    r'(?:这是因为|发现根因|真正原因)[：:]?\s*(.{10,120})',
    r'(?:关键发现|核心发现|分析显示)[：:]\s*(.{10,120})',
    r'(?:诊断结论|性能瓶颈|瓶颈在于)[：:]?\s*(.{10,120})',
    # 英文推理标记
    r'(?:root cause|key finding|the reason|because)[:\s]+(.{10,120})',
    r'(?:analysis shows|discovered that|found that|turns out)[,:\s]+(.{10,120})',
    # 迭代120：覆盖现代中文 LLM 回复高频推理表达（A/B 分析显示完全缺失）
    # 这说明/这表明/这意味着 — 结论性推理句（最高频但缺失）
    r'(?:这说明|这表明|这意味着|这反映了)[：,，]?\s*(.{10,120})',
    # 关键在于/症结在于 — 核心问题诊断
    r'(?:关键在于|症结在于|核心在于|问题在于)\s*(.{10,120})',
    # 实质上/本质上 — 深层解释
    r'(?:实质上|本质上|根本上)[，,：:]?\s*(.{10,120})',
    # 因此 + 结论（单独使用，无前置因为的推理终结句）
    r'(?:^|\n)因此[，,：:]?\s*(.{10,100})',
    # 由此可见/可见 — 归纳推理
    r'(?:由此可见|由此可得|可以看出)[，,：:]?\s*(.{10,100})',
    # 证明了/验证了 + 核心结论
    r'(?:这证明了|这验证了|实验证明|测试证明)\s*(.{10,100})',
]

# ── v3 新增：对比句式（捕获隐式决策）────────────────────────
COMPARISON_SIGNALS = [
    # "X 而非 Y" / "X 而不是 Y" → 决策含完整对比
    r'(?:使用|用|采用|选)\s*(.{3,40})\s*(?:而非|而不是|不是)\s*(.{3,40})',
    # "相比 X，Y 更…" → 决策 "相比 X，Y 更…"
    r'(?:相比|比起|对比)\s*(.{3,30})[，,]\s*(.{3,60}?更.{2,30})',
    # "X 比 Y 好/快/稳定" → 决策 "X 比 Y …"
    r'(.{3,30})\s*比\s*(.{3,30})\s*(好|快|稳定|合适|简单|可靠|高效).{0,30}',
    # "不用 X 改用 Y" / "放弃 X 改用 Y"
    r'(?:不用|放弃|弃用)\s*(.{3,30})\s*(?:改用|换成|用)\s*(.{3,40})',
]

# ── v3 新增：因果链（保留 why 维度）─────────────────────────
# 迭代122：重构 CAUSAL_SIGNALS — 覆盖真实 LLM 因果表达模式
# 迭代127：修复低命中率 — 放宽约束覆盖真实 LLM 输出结构
#   问题1：模式[0]要求逗号分隔，"因为X所以Y"（无逗号）不匹配
#   问题2："这导致了..." 前缀"这"只有1字，旧最小3字限制阻断
#   问题3："原因：..." / "根因：..." 冒号式未覆盖
#   问题4："由于X，需要Y" 后半无"所以/因此"，旧双组模式不匹配
# OS 类比：信号处理器的 syscall 过滤表 — 太严的 seccomp 规则会阻断合法调用，
#   需要基于实测 trace 校准过滤粒度（strace → seccomp profile）。
# 新设计分两类：
#   A. 双组（cause + effect）：格式化为 "cause → effect"
#   B. 单组（完整因果句）：直接存储完整句子（含因果连接词上下文足够语义）
CAUSAL_SIGNALS = [
    # ── 类型A：双组（因 + 果）——正式书面因果 ──
    # "因为 X，所以 Y" / "由于 X，因此 Y"（迭代127：允许逗号可选）
    r'(?:因为|由于|原因是)\s*(.{5,60}?)[，,；;]?\s*(?:所以|因此|故|于是)\s*(.{5,60})',
    # "由于 X，Y"（迭代127新增：后半不要求"所以"，覆盖"由于限制，需要额外补充"）
    r'(?:由于|因为)\s*(.{5,60})[，,；;]\s*(.{5,60})',
    # "X 是因为 Y"
    r'(.{5,40})\s*是因为\s*(.{5,60})',
    # "之所以 X，是因为 Y"
    r'之所以\s*(.{5,40})[，,]\s*(?:是因为|因为)\s*(.{5,60})',

    # ── 类型B：单组（完整因果句）——LLM 高频表达 ──
    # 决策 + 原因（最常见："选择X，因为Y"）
    r'(.{5,80}?)[，,]\s*(?:因为|原因是|由于)\s*.{5,60}',
    # 迭代127新增：冒号式原因说明（"原因：X" / "根因：X" — 最常见 LLM 诊断格式）
    r'(?:原因|根因|根本原因|问题原因)[：:]\s*(.{10,100})',
    # 导致/造成/引发（迭代127：放宽前缀到1字，覆盖"这导致了..."）
    r'(.{1,60})\s*(?:导致了?|造成了?|引发了?|触发了?|引起了?)\s*.{5,60}',
    # 是由...导致/引发的（被动因果）
    r'(.{3,60})\s*是由\s*.{3,40}\s*(?:导致|引发|造成)的?',
    # X，根本原因是 Y（诊断性因果）
    r'(.{3,60})[，,]\s*根本原因(?:是|在于)\s*.{5,60}',
    # 因此/所以 + 结论（单向）
    r'(?:因此|所以|故此|于是)[，,]?\s*(.{10,80})',
    # 英文因果
    r'(.{5,60})\s*(?:because|due to|caused by|resulted in|leads to)\s*.{5,60}',
]

# ── v3 新增：量化证据模式 ────────────────────────────────────
QUANTITATIVE_PATTERN = re.compile(
    r'(?:'
    r'\d+(?:\.\d+)?%'                          # 百分比
    r'|\d+(?:\.\d+)?\s*(?:ms|s|秒|毫秒)'       # 时间
    r'|\d+(?:\.\d+)?\s*(?:MB|GB|KB|字节)'       # 大小
    r'|[<>≤≥]\s*\d+'                            # 不等式约束
    r'|\d+/\d+\s*(?:cases?|测试)'               # 测试结果
    r'|hit_rate[=:]\s*\d'                        # 指标
    r'|noise_rate[=:]\s*\d'
    r'|precision[=:]\s*\d'
    r')',
    re.IGNORECASE
)

# ── 迭代98+102：设计约束信号（design_constraint）────────────────
# 系统中"为什么不这样做"的约束知识——违反会产生语义错误但表面合理的修改
# 迭代102 扩展：从8个模式扩展到22个，覆盖工程中常见约束表达
CONSTRAINT_SIGNALS = [
    # ── 原有模式（迭代98）—— 为什么不能/why must not 提前，避免被短模式[0]抢占 ──
    r'(?:为什么.*不能.*?|why.*must not.*?)[：:]\s*(.{10,100})',  # 提前：.*? 允许冒号前有额外词语
    r'(?:不能|禁止|不允许|must not|should not)\s*(.{5,80})\s*(?:因为|，因为|because|，because)',
    r'(?:这样做会|这会导致|this would|will cause)\s*(.{5,80})',
    r'(?:会导致|会产生|会引发|会造成)\s*(.{5,80})',
    r'(?:破坏|违反|corrupt|violate)\s*(.{5,80})',
    r'(?:设计约束|invariant|不变量|design constraint)[：:]\s*(.{10,120})',
    r'(?:之所以|正是因为)\s*(.{5,60})\s*(?:绕过|skip)',
    r'(?:前提条件|prerequisite|assumption)[：:]\s*(.{10,100})',
    # ── 迭代102 新增：中文警告句式 ──
    r'(?:注意不要|小心不要|务必不要|千万不要|切勿)\s*(.{5,80})',
    r'(?:注意[：:]\s*)(.{10,120})',   # "注意：此处不能..." 标题式警告
    r'(?:警告[：:]\s*)(.{10,120})',   # "警告：..." markdown 警告块
    r'(?:危险[：:]\s*)(.{10,120})',   # "危险：..." 高危警告
    # ── 迭代102 新增：markdown 警告标记 ──
    r'(?:⚠️|⚠|🚫|❌)\s*(.{5,100})',  # emoji 警告前缀
    r'(?:WARNING:|CAUTION:|DANGER:|IMPORTANT:)\s*(.{10,120})',  # 英文警告标记
    r'(?:> \[!WARNING\]|> \[!CAUTION\]|> \[!DANGER\])[^\n]*\n+(.{10,120})',  # GitHub alert 格式
    # ── 迭代102 新增：英文约束句式 ──
    r'(?:never|avoid|do not|don\'t)\s+(.{5,80})\s+(?:because|as|since|—|--)',
    r'(?:always\s+(?:ensure|check|verify|call|use))\s+(.{5,80})\s+(?:before|first|or)',
    r'(?:requires?|must\s+(?:be|have|call|use))\s+(.{5,80})\s+(?:before|first|to)',
    # ── 迭代102 新增：前置条件 / 顺序约束 ──
    r'(?:只有.*才能|必须先.*再|先.*后才能)\s*(.{5,80})',
    r'(?:assert|ensure|guarantee|require)[（(](.{5,80})[）)]',  # assert(条件) 格式
    r'(?:不变式|invariant|precondition|postcondition)[：:]?\s*(.{10,100})',
    # ── 迭代102 新增：后果/副作用 ──
    r'(?:否则会|否则将|不然会|不然将)\s*(.{5,80})',
    r'(?:race condition|deadlock|memory leak|data corruption|undefined behavior|竞态|死锁|内存泄漏|数据损坏)\s*(?:will|would|可能|将会|的风险)?(.{0,60})',
]

# ── 上下文标记（帮助判断 chunk 属于哪个任务/主题）──────────
TOPIC_HEADER = re.compile(r'^#{1,3}\s+(.{5,60})$', re.MULTILINE)


def _extract_topic(text: str) -> str:
    """从最近的 markdown 标题提取话题。"""
    headers = TOPIC_HEADER.findall(text)
    if headers:
        return headers[-1].strip()[:60]
    return ""


def _extract_by_signals(text: str, patterns: list) -> list:
    results = []
    for pattern in patterns:
        for m in re.finditer(pattern, text):
            captured = m.group(1).strip()
            # 截断到第一个换行或句号
            captured = re.split(r'[\n。！？]', captured)[0].strip()
            # 去掉 markdown 格式符
            captured = re.sub(r'\*{1,3}|`{1,3}', '', captured).strip()
            # 迭代73：碎片过滤 — 截断的代码/表格/标题不入库
            if _is_fragment(captured):
                continue
            if len(captured) >= _sysctl("extractor.min_length"):
                results.append(captured[:_sysctl("extractor.max_summary")])
    return _deduplicate(results)


def _extract_structured_decisions(text: str) -> list:
    """
    从结构化 markdown 提取决策：
    - 标题下的第一段结论
    - 有序列表中的关键项（数字开头）
    - callout / blockquote 中的结论
    """
    results = []

    # 提取 blockquote 中的关键句（> **xxx**: yyy）
    blockquote = re.finditer(r'^>\s*\*\*(.{2,20})\*\*[：:]\s*(.{10,100})$', text, re.MULTILINE)
    for m in blockquote:
        label = m.group(1).strip()
        content = m.group(2).strip()
        if label in ('核心洞察', '关键预测', '结论', '推断', '最终方案', '核心命题'):
            results.append(f"{label}：{content}"[:_sysctl("extractor.max_summary")])

    # 提取有序列表中包含决策词的项
    ordered_items = re.finditer(r'^\d+[.。]\s+(.{10,100})$', text, re.MULTILINE)
    for m in ordered_items:
        item = m.group(1).strip()
        # 只保留含有决策信号的列表项
        if re.search(r'(?:选择|决定|推荐|结论|采用|方向|核心)', item):
            results.append(re.sub(r'\*{1,3}|`', '', item)[:_sysctl("extractor.max_summary")])

    return _deduplicate(results)


def _extract_comparisons(text: str) -> tuple:
    """
    v3 对比句式提取。返回 (decisions, exclusions)。
    从 "X 而非 Y" 类句式同时提取决策和排除路径。
    """
    decisions = []
    exclusions = []
    for pattern in COMPARISON_SIGNALS:
        for m in re.finditer(pattern, text):
            groups = m.groups()
            full_match = m.group(0).strip()
            full_clean = re.sub(r'\*{1,3}|`{1,3}', '', full_match).strip()
            # 对比句式产出完整决策（含对比上下文）
            if len(full_clean) >= _sysctl("extractor.min_length"):
                decisions.append(full_clean[:_sysctl("extractor.max_summary")])
            # "不用 X 改用 Y" 模式：第一个捕获组是排除项
            if len(groups) >= 2 and re.match(r'(?:不用|放弃|弃用)', pattern[:20]):
                excluded = re.sub(r'\*{1,3}|`{1,3}', '', groups[0]).strip()
                if len(excluded) >= 5:
                    exclusions.append(excluded[:_sysctl("extractor.max_summary")])
    return _deduplicate(decisions), _deduplicate(exclusions)


def _extract_causal_chains(text: str) -> list:
    """
    v3 因果链提取。返回 causal_chain chunks。
    迭代122：重构以支持两类模式：
      类型A（双组）：格式化为 "cause → effect"
      类型B（单组/完整句）：直接存储完整因果句（含触发词语义上下文）

    两类均需要 → 分隔符（类型A显式格式化，类型B用完整匹配替代）。
    """
    results = []
    for pattern in CAUSAL_SIGNALS:
        for m in re.finditer(pattern, text):
            groups = m.groups()
            if len(groups) >= 2:
                # 类型A：双组（因 + 果），格式化为 "cause → effect"
                cause = re.sub(r'\*{1,3}|`{1,3}', '', groups[0]).strip()
                effect = re.sub(r'\*{1,3}|`{1,3}', '', groups[1]).strip()
                cause = re.split(r'[\n]', cause)[0].strip()
                effect = re.split(r'[\n]', effect)[0].strip()
                if len(cause) >= 5 and len(effect) >= 5:
                    chain = f"{cause} → {effect}"
                    results.append(chain[:_sysctl("extractor.max_summary")])
            elif len(groups) == 1:
                # 类型B：单组（完整因果句）
                # 使用完整匹配（m.group(0)）保留触发词上下文，不只取捕获组
                full_match = m.group(0).strip()
                full_match = re.sub(r'\*{1,3}|`{1,3}', '', full_match)
                full_match = re.split(r'[\n]', full_match)[0].strip()
                # 须包含因果语义词（防止误匹配）
                # 迭代127：扩展语义词 + 加入"原因"系词（冒号式模式的全匹配包含"原因："）
                if re.search(r'(?:因为|由于|导致了?|造成了?|引发了?|触发了?|引起了?|'
                             r'根本原因|因此|所以|原因[：:]|根因[：:]|问题原因[：:]|'
                             r'because|due to|caused by|resulted in|leads to)',
                             full_match):
                    # 迭代127：min_length 从15→10→6（"这导致了性能下降"=8字，需≥6才能通过）
                    if len(full_match) >= 6:
                        results.append(full_match[:_sysctl("extractor.max_summary")])
    return _deduplicate(results)


def _extract_constraints(text: str) -> list:
    """
    迭代98：设计约束提取 — "为什么不这样做"的系统级约束知识。

    特征：
    - 隐性：正常代码里不写，只在 maintainer 解释/code review 时出现
    - 跨时间有效：架构级约束，长期稳定不过期
    - 高保护：违反会产生语义错误但表面合理的修改

    format: "路径/符号 不能 做 Y，因为会 Z"
    """
    _CONSTRAINT_SEMANTIC = re.compile(
        r'(?:不能|禁止|不允许|不应该|must not|should not|cannot|'
        r'会导致|会产生|会引发|会造成|导致|破坏|违反|corrupt|violate|'
        r'前提|必须先|只有.*才能|否则|不变量|invariant|precondition|'
        r'race condition|deadlock|memory leak|data corruption|'
        r'竞态|死锁|内存泄漏|数据损坏|'
        r'never|avoid|don\'t|always ensure|requires?.*before|must.*before|'
        r'unsafe|incorrect|incorrect|wrong|error|fail|危险|风险|'
        r'without.*lock|without.*holding|without.*acquiring|'
        r'因为|由于|以免|以防)',  # iter119: 因果说明也是约束知识的核心载体
        re.IGNORECASE
    )
    results = []
    for pattern in CONSTRAINT_SIGNALS:
        for m in re.finditer(pattern, text):
            full_match = m.group(0)  # 完整匹配（含触发词）
            captured = m.group(1).strip() if m.groups() else full_match.strip()
            # 截断到句号/换行
            captured = re.split(r'[\n。！？]', captured)[0].strip()
            # 去 markdown 格式
            captured = re.sub(r'\*{1,3}|`{1,3}', '', captured).strip()
            # iter119: 修复 "race condition: ..." 类 pattern — 去除 captured 开头的冒号残留
            # 原因：CONSTRAINT_SIGNALS pattern 22 匹配 "race condition: ..." 时
            # 捕获组从冒号后开始，但 re.finditer 可能包含前导冒号+空格
            captured = re.sub(r'^[：:\s]+', '', captured).strip()
            if len(captured) < 10 or len(captured) > _sysctl("extractor.max_summary"):
                continue
            # iter119: 碎片过滤 — 残缺句/截断句不入库
            if _is_fragment(captured):
                continue
            # iter119: 通用质量过滤 — 拦截状态快照/噪声行
            # 注意：约束知识允许以介词开头（如"在 X 里调用 Y 会导致 Z"），
            # 但 _is_quality_chunk 有介词开头过滤规则。对约束类 chunk，改为对
            # full_match 做质量验证（触发词提供足够上下文，不应因 captured 以介词
            # 开头而丢弃真正的约束知识）。
            # 对 full_match 做质量检验，同时 captured 必须非碎片且有实质内容
            if not _is_quality_chunk(full_match):
                continue
            # iter119: 约束语义门控 — 完整匹配文本（含触发词）必须含约束语义词
            # 宽泛模式（注意:/警告:/⚠️）容易误匹配调试输出，用完整上下文验证
            if not _CONSTRAINT_SEMANTIC.search(full_match):
                continue
            results.append(captured)
    return _deduplicate(results)


def _extract_quantitative_conclusions(text: str) -> list:
    """
    v3 量化证据保留。含数字度量的结论行自动提取。
    保留带性能数据、测试结果、指标的句子——这些是"可重建性"最低的信息。

    v6 迭代71：Generational GC 后过滤 — 排除纯验证/测试报告类句子。
    OS 类比：分代 GC 中的 young generation 过滤。
    这些句子虽含量化数据，但本质是过程性记录而非可复用决策。
    """
    # 迭代71：低价值模式排除（纯验证报告、纯测试通过数、纯回归报告）
    LOW_VALUE_QUANT = re.compile(
        r'^(?:'
        r'\d+/\d+\s*(?:通过|passed|全绿|green)'  # "33/33 通过"
        r'|验证[：:]\s*\d+/\d+'                    # "验证：33/33..."
        r'|回归.*全绿'                              # "回归全绿"
        r'|\d+\s*(?:passed|failed)'                # "11 passed"
        r'|测试[：:]\s*\d+'                         # "测试：11..."
        r')',
        re.IGNORECASE
    )
    # 代码行特征：Python/shell 关键字 + 函数调用/f-string 组合
    _CODE_LINE_RE = re.compile(
        r'\bfor\b.+\bin\b|\bprint\s*\(|f[\'"][^\'\"]*\{|'
        r'\bimport\b|\bdef\b|\bif\b.+:|\.append\(|sys\.path|'
        r'^\s*[a-z_]+\s*=\s*(?:open|conn|cursor)\b|'
        r'^\$\s*\w|^#\s*!/'  # shell shebang/变量
    )

    results = []
    for line in text.splitlines():
        stripped = line.strip()
        # 跳过表头行、纯分隔符、代码块标记
        if not stripped or stripped.startswith('```') or stripped.startswith('|---'):
            continue
        if stripped.startswith('#'):
            continue
        # V11: 跳过代码行（Python/shell 语法特征）— 防止 f-string/:25s 被误判为量化数据
        if _CODE_LINE_RE.search(stripped):
            continue
        # 必须包含量化证据
        if not QUANTITATIVE_PATTERN.search(stripped):
            continue
        # 必须包含结论性动词或标点（不是纯数据行）
        if not re.search(r'[：:→✅❌=]|(?:实测|验证|结果|通过|达到|降至|提升|稳定)', stripped):
            continue
        clean = re.sub(r'\*{1,3}|`{1,3}', '', stripped).strip()
        clean = re.sub(r'^[-*•]\s*', '', clean)  # 去列表符号
        # 迭代71：排除纯验证/测试报告
        if LOW_VALUE_QUANT.search(clean):
            continue
        if _sysctl("extractor.min_length") <= len(clean) <= _sysctl("extractor.max_summary"):
            results.append(clean)
    return _deduplicate(results)[:5]  # 量化结论最多5条


def _quant_semantic_concepts(summary: str) -> str:
    """
    迭代336：从量化证据 summary 中提取语义概念词，追加到 content 末尾。

    信息论根因（Encoding-Retrieval Mismatch, Tulving 1973）：
      quantitative_evidence summary 含数字/符号/迭代编号（如 "11.4us→1.35us"），
      但查询时用概念词（"如何优化性能"/"召回率提升"），形成语义鸿沟。
      FTS5 的 BM25 是词汇匹配引擎，无法跨越此鸿沟 → 14/18 (78%) 零召回。

    修复策略（Query Expansion 的反向：Document Expansion）：
      在写入时预计算并追加概念词，让 FTS5 能按概念词索引量化证据。
      类比：搜索引擎对商品标题自动追加品类词（"iPhone 15" → "手机 智能手机 苹果"）。

    OS 类比：Linux /proc/[pid]/wchan — 内核将进程等待通道名（符号地址）
      与人类可读的系统调用名映射，让 "ps" 输出人类可理解的状态而非裸地址。

    规则优先级：越具体的规则越先匹配，最多追加 3 类概念，避免语义污染。
    """
    import re as _re
    if not summary:
        return ""
    concepts: list = []
    s = summary

    # ── 类别 1：性能优化（数值降低方向）──
    # "X→Y" 且含时间单位/延迟词，判定为性能优化
    if _re.search(r'\d.*→.*\d', s) or '→' in s:
        # 尝试判断方向：提取箭头两侧的第一个数字
        _nums = _re.findall(r'[\d.]+', s)
        if len(_nums) >= 2:
            try:
                _before = float(_nums[0])
                _after = float(_nums[-1])
                if _before > _after and _after > 0:
                    # 数值降低 → 优化/加速
                    concepts.append("性能优化 速度提升 延迟降低 optimize latency improve")
                elif _after > _before:
                    # 数值升高 → 提升/增长
                    concepts.append("性能提升 改善 increase improve recall")
            except (ValueError, IndexError):
                concepts.append("性能优化 improve optimize 量化提升")

    # ── 类别 2：检索/召回类 ──
    if _re.search(r'召回|recall|FTS|BM25|检索|fts_rank|precision|hit.rate', s, _re.IGNORECASE):
        concepts.append("检索优化 召回率提升 search retrieve FTS5 BM25 recall precision")

    # ── 类别 3：启动/导入/延迟类 ──
    if _re.search(r'import|启动|冷启动|加载|startup|load|ms|us|μs|latency', s, _re.IGNORECASE):
        concepts.append("启动性能 冷启动 import overhead latency ms startup")

    # ── 类别 4：修复/Bug 类 ──
    if _re.search(r'修复|fix|bug|错误|error|crash|回归|regression', s, _re.IGNORECASE):
        concepts.append("修复 bug修复 fix repair regression")

    # ── 类别 5：内存/swap/淘汰类 ──
    if _re.search(r'内存|memory|swap|evict|淘汰|chunk|kswapd|oom', s, _re.IGNORECASE):
        concepts.append("内存优化 淘汰 eviction memory chunk kswapd")

    # ── 类别 6：迭代版本关联 ──
    _iter_match = _re.search(r'iter(\d+)', s, _re.IGNORECASE)
    if _iter_match:
        concepts.append(f"迭代优化 iter{_iter_match.group(1)} 版本改进")

    # 最多取前 3 类，去重，追加为 concept 注释行
    if not concepts:
        # fallback: 通用量化证据概念
        concepts.append("量化优化 性能改进 benchmark optimize improve")

    unique_concepts = list(dict.fromkeys(concepts))[:3]
    return " | ".join(unique_concepts)


def _is_quality_reasoning(summary: str) -> bool:
    """
    迭代113：reasoning_chain 专用质量门控 — 最小语义密度校验。

    reasoning_chain 表达的是因果关系/推理过程，必须含以下任一信号：
      A. 因果词：因为/由于/导致/→/所以/故/原因/根因/因此
      B. 发现词：发现/诊断/确认/问题是/根本/核心问题
      C. 推理标记：root cause / because / therefore / key finding
      D. 长度 ≥ 25 chars（长句通常包含足够上下文）

    过滤目标：纯名词短语（如 "enqueue 阶段"、"sub-sched 激活机制"）
    这类碎片是提取模式误匹配 "xxx阶段" / "xxx机制" 等后缀词产生的，
    独立存储时没有任何可复用的因果知识。

    OS 类比：TCP segment 的 payload 必须包含有效数据，不接受空 payload（纯 ACK 除外）。
    """
    if not summary:
        return False
    s = summary.strip()
    # 长句通常自带足够上下文
    if len(s) >= 25:
        return True
    # 含因果/推理/发现信号词
    if re.search(
        r'(?:因为|由于|导致|→|所以|故|原因|根因|因此|'
        r'发现|诊断|确认|问题是|根本|核心问题|'
        r'root cause|because|therefore|key finding|'
        r'理解有误|理解错误|误以为|实际上)',
        s
    ):
        return True
    return False


def _is_fragment(text: str) -> bool:
    """
    迭代73+78+80：碎片检测。排除截断的代码片段、表格行、markdown 标题等非完整句。
    OS 类比：TCP checksum — 数据完整性校验，丢弃损坏的 segment。
    迭代78：新增冒号碎片检测（leading/trailing ：/:）
    迭代80：新增架构层级标签碎片检测（/L4/L5...、/层级名...）
    """
    if not text or len(text) < 8:
        return True
    # 以特殊字符开头 = 截断的代码/表格/markdown/续行（含全角变体）
    if text[0] in ('_', '|', ')', ']', '}', '>', '+', '=', ':', '：', '）', '】', '》'):
        return True
    # 以逗号/顿号/分号开头 = 截断残缺句（上一句的后半段）
    if text[0] in (',', '，', '、', ';', '；'):
        return True
    # markdown 标题行本身不是摘要
    if text.startswith('#'):
        return True
    # 纯数字/符号行
    if re.match(r'^[\d\s.,:;/×\-+=%]+$', text):
        return True
    # 以 | 分隔的表格行
    if text.count('|') >= 2:
        return True
    # 迭代78：以冒号结尾 = 标题/标签碎片（"核心成果："）
    stripped = text.rstrip()
    if stripped.endswith(':') or stripped.endswith('：'):
        return True
    # 迭代80：/大写字母 或 /中文 开头 = 架构层级标签碎片（"/L4/L5..."、"/层级名..."）
    # 保留真正的文件路径（/home/...、/etc/...、/var/... 等小写开头）
    if text[0] == '/' and len(text) > 1 and re.match(r'^/[A-Z\u4e00-\u9fff]', text):
        return True
    return False


def _extract_conversation_summary(text: str, extra_texts: list = None) -> list:
    """
    v4+v5 对话摘要提取。从助手回复中提取核心行动/结论句。
    目标：捕获非决策类的有价值信息（解决了什么、做了什么、发现了什么）。

    提取策略（3 层）：
      S1 完成动作：已完成/已修复/已创建/已更新 + 对象
      S2 发现/诊断：发现/诊断/定位/确认 + 问题描述
      S3 markdown 总结标题下的首句（## 总结/## Summary 后的第一段）

    v5 迭代73：碎片过滤 — _is_fragment() 校验捕获结果完整性。
    增强1：extra_texts — transcript 尾部最近 5 轮 assistant 消息，也参与提取。
    """
    # 合并 last_assistant_message 和 transcript 尾部额外轮次
    all_texts = [text]
    if extra_texts:
        all_texts.extend(t for t in extra_texts if t and t != text)

    results = []

    # S1: 完成动作
    action_patterns = [
        r'(?:已完成|已修复|已创建|已更新|已实现|已添加|已删除|已重构|已迁移|已升级|已部署)[：:：]?\s*(.{5,100})',
        r'(?:完成了|修复了|创建了|更新了|实现了|添加了|重构了)[：:：]?\s*(.{5,100})',
        r'(?:successfully|completed|fixed|created|implemented|deployed)[：:：]?\s*(.{5,100})',
    ]
    # S2: 发现/诊断
    diag_patterns = [
        r'(?:发现|诊断|定位|确认|排查)[：:]\s*(.{10,100})',
        r'(?:问题是|原因是|根因是|bug 是)[：:]\s*(.{10,100})',
        r'(?:found|diagnosed|confirmed|identified)[：:]\s*(.{10,100})',
    ]

    # 增强1：对 last_assistant_message + transcript 尾部 5 轮消息都做 S1/S2/S3 提取
    for src_text in all_texts:
        for pat in action_patterns:
            for m in re.finditer(pat, src_text, re.IGNORECASE):
                captured = m.group(1).strip()
                captured = re.split(r'[\n。！？]', captured)[0].strip()
                captured = re.sub(r'\*{1,3}|`{1,3}', '', captured).strip()
                if _is_fragment(captured):
                    continue
                if _sysctl("extractor.min_length") <= len(captured) <= _sysctl("extractor.max_summary"):
                    results.append(captured)

        for pat in diag_patterns:
            for m in re.finditer(pat, src_text, re.IGNORECASE):
                captured = m.group(1).strip()
                captured = re.split(r'[\n。！？]', captured)[0].strip()
                captured = re.sub(r'\*{1,3}|`{1,3}', '', captured).strip()
                if _is_fragment(captured):
                    continue
                if _sysctl("extractor.min_length") <= len(captured) <= _sysctl("extractor.max_summary"):
                    results.append(captured)

        # S3: 总结标题下的首句
        summary_sections = re.finditer(
            r'^#{1,3}\s*(?:总结|Summary|结论|完成|结果|验证)[^\n]*\n+(.{10,200})',
            src_text, re.MULTILINE | re.IGNORECASE
        )
        for m in summary_sections:
            first_line = m.group(1).strip()
            first_line = re.split(r'[\n]', first_line)[0].strip()
            first_line = re.sub(r'\*{1,3}|`{1,3}', '', first_line).strip()
            first_line = re.sub(r'^[-*•]\s*', '', first_line)
            if _is_fragment(first_line):
                continue
            if _sysctl("extractor.min_length") <= len(first_line) <= _sysctl("extractor.max_summary"):
                results.append(first_line)

    return _deduplicate(results)[:6]  # 增强1后最多 6 条（原 3 × 最多 2 轮有效内容）


def _read_transcript_tail(transcript_path: str, max_bytes: int = 200 * 1024) -> list:
    """
    增强1：读取 transcript JSONL 尾部，返回最近 5 轮 assistant 消息文本列表。
    只读文件末尾 max_bytes 字节（seek，不全量读取），控制延迟 < 50ms。

    transcript 格式（每行一个 JSON）：
      assistant 轮次：type="assistant", message.role="assistant", content=[{type="text",text="..."}]
    """
    try:
        path = Path(transcript_path)
        if not path.exists() or path.stat().st_size == 0:
            return []
        file_size = path.stat().st_size
        read_size = min(max_bytes, file_size)
        with open(path, 'rb') as f:
            f.seek(-read_size, 2)
            raw = f.read(read_size)
        text = raw.decode('utf-8', errors='replace')
        lines = text.split('\n')
        # 第一行可能被截断，跳过
        if read_size < file_size:
            lines = lines[1:]

        results = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get('type') != 'assistant':
                continue
            msg = d.get('message', {})
            if not isinstance(msg, dict) or msg.get('role') != 'assistant':
                continue
            content = msg.get('content', [])
            if not isinstance(content, list):
                continue
            for c in content:
                if isinstance(c, dict) and c.get('type') == 'text':
                    t = c.get('text', '')
                    if t and len(t) >= 20:
                        results.append(t)
        # 返回最近 5 轮（列表末尾是最新的）
        return results[-5:]
    except Exception:
        return []


def _extract_from_tool_outputs(transcript_path: str, session_id: str,
                               project: str, conn: sqlite3.Connection) -> int:
    """
    增强2：从 transcript JSONL 尾部提取 Bash tool_result 关键结论。
    chunk_type = "tool_insight"（新类型）。

    策略：
    - 只读尾部 200KB（seek，不全量读取）
    - 找 user 类型条目中对应 Bash 工具的 tool_result
    - 提取含量化数据（通过/失败/性能指标/百分比变化）的行

    返回写入的 chunk 数（上限 5）。
    """
    _TOOL_INSIGHT_PATTERN = re.compile(
        r'(?:'
        # 测试通过/失败计数
        r'\d+/\d+\s*(?:通过|passed|failed|tests?)'
        r'|(?:PASSED|FAILED|ERROR)\s+\d+'
        r'|\d+\s+(?:passed|failed|error)'
        # 测试摘要行（"N passed, M failed"）
        r'|\d+\s+passed.*\d+\s+(?:failed|warning|error)'
        # 性能数据
        r'|P\d+\s*[=:]\s*\d+\s*(?:ms|s)'
        r'|\d+(?:\.\d+)?\s*ms\b'
        # 指标变化
        r'|\+\d+(?:\.\d+)?%|-\d+(?:\.\d+)?%'
        r'|(?:hit_rate|coverage|precision|recall)[=:]\s*\d'
        r')',
        re.IGNORECASE
    )

    try:
        path = Path(transcript_path)
        if not path.exists() or path.stat().st_size == 0:
            return 0
        file_size = path.stat().st_size
        read_size = min(200 * 1024, file_size)
        with open(path, 'rb') as f:
            f.seek(-read_size, 2)
            raw = f.read(read_size)
        text_raw = raw.decode('utf-8', errors='replace')
        lines = text_raw.split('\n')
        if read_size < file_size:
            lines = lines[1:]

        # 第一遍：收集 Bash tool_use id 集合
        bash_tool_ids: set = set()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get('type') != 'assistant':
                continue
            msg = d.get('message', {})
            if not isinstance(msg, dict):
                continue
            for c in (msg.get('content') or []):
                if isinstance(c, dict) and c.get('type') == 'tool_use':
                    if c.get('name') == 'Bash':
                        bash_tool_ids.add(c.get('id', ''))

        if not bash_tool_ids:
            return 0

        # 第二遍：提取 Bash 工具的 tool_result
        written = 0
        seen_summaries: set = set()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get('type') != 'user':
                continue
            msg = d.get('message', {})
            if not isinstance(msg, dict):
                continue
            for c in (msg.get('content') or []):
                if not isinstance(c, dict) or c.get('type') != 'tool_result':
                    continue
                if c.get('tool_use_id', '') not in bash_tool_ids:
                    continue
                # 提取输出文本
                raw_content = c.get('content', '')
                if isinstance(raw_content, str):
                    output_text = raw_content
                elif isinstance(raw_content, list):
                    parts = []
                    for item in raw_content:
                        if isinstance(item, dict) and item.get('type') == 'text':
                            parts.append(item.get('text', ''))
                    output_text = '\n'.join(parts)
                else:
                    continue

                if not output_text or len(output_text) < 20:
                    continue

                # 逐行扫描含量化数据的行
                for out_line in output_text.splitlines():
                    stripped = out_line.strip()
                    if not stripped or len(stripped) < 10:
                        continue
                    if not _TOOL_INSIGHT_PATTERN.search(stripped):
                        continue
                    # 去 ANSI 颜色码，去多余空白
                    clean = re.sub(r'\x1b\[[0-9;]*m', '', stripped)
                    clean = re.sub(r'\s+', ' ', clean).strip()
                    if len(clean) < 10 or len(clean) > 200:
                        continue
                    if not _is_quality_chunk(clean):
                        continue
                    # tool_insight 专用过滤：排除纯调试/系统输出噪音
                    # 这些行含量化数据但没有可复用决策价值
                    if _is_tool_insight_noise(clean):
                        continue
                    key = re.sub(r'\s+', '', clean.lower())
                    if key in seen_summaries:
                        continue
                    seen_summaries.add(key)
                    _write_chunk("tool_insight", clean, project, session_id,
                                 topic="", conn=conn,
                                 importance_override=0.75)
                    written += 1
                    if written >= 5:
                        return written
        return written
    except Exception:
        return 0


def _extract_tool_patterns(transcript_path: str, conn: sqlite3.Connection,
                           project: str, session_id: str,
                           context_text: str = "") -> int:
    """
    工具使用模式学习 — 从本轮 transcript 中提取工具调用序列并写入 tool_patterns 表。
    OS 类比：perf_event 采样 — 在会话结束时提取 CPU 热点调用链，供下轮预测和预热。

    策略：
    - 只读 transcript JSONL 尾部 200KB（seek，不全量读取）
    - 收集 assistant 类型条目中所有 tool_use 的 name 字段（按出现顺序）
    - 滑动窗口（3/4/5 个工具一组）切分为子序列
    - 对每个子序列：
        hash(JSON序列) → 已存在则 UPDATE frequency+1/last_seen，否则 INSERT
    - context_keywords：从 context_text（last_assistant_message）提取英文技术词和中文双字词

    返回写入/更新的 pattern 数（上限 30）。
    """
    import hashlib

    if not transcript_path:
        return 0

    try:
        path = Path(transcript_path)
        if not path.exists() or path.stat().st_size == 0:
            return 0
        file_size = path.stat().st_size
        read_size = min(200 * 1024, file_size)
        with open(path, 'rb') as f:
            f.seek(-read_size, 2)
            raw = f.read(read_size)
        text_raw = raw.decode('utf-8', errors='replace')
        lines = text_raw.split('\n')
        if read_size < file_size:
            lines = lines[1:]  # 跳过可能截断的首行

        # 按顺序收集本轮所有 tool_use 名称
        tool_names: list = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get('type') != 'assistant':
                continue
            msg = d.get('message', {})
            if not isinstance(msg, dict):
                continue
            for c in (msg.get('content') or []):
                if isinstance(c, dict) and c.get('type') == 'tool_use':
                    name = c.get('name', '')
                    if name:
                        tool_names.append(name)

        if len(tool_names) < 2:
            return 0  # 少于 2 个工具调用，无模式可提取

        # 提取 context_keywords（用于模式的上下文标注）
        keywords: list = []
        if context_text:
            seen_kw: set = set()
            # 英文技术词（驼峰/下划线/短横线）
            for m in re.finditer(r'\b([A-Z][a-zA-Z0-9_]{2,20}|[a-z][a-z0-9_]{3,20})\b', context_text[:3000]):
                w = m.group(1)
                if w.lower() not in _MADVISE_STOPWORDS and w not in seen_kw:
                    seen_kw.add(w)
                    keywords.append(w)
                    if len(keywords) >= 8:
                        break
            # 中文双字词（高频）
            if len(keywords) < 8:
                cn = re.sub(r'[^\u4e00-\u9fff]', '', context_text[:3000])
                freq: dict = {}
                for i in range(len(cn) - 1):
                    bg = cn[i:i+2]
                    freq[bg] = freq.get(bg, 0) + 1
                for bg, cnt in sorted(freq.items(), key=lambda x: -x[1]):
                    if cnt >= 2 and len(keywords) < 8:
                        keywords.append(bg)

        kw_json = json.dumps(keywords, ensure_ascii=False)

        now_iso = datetime.now(timezone.utc).isoformat()
        written = 0

        # 滑动窗口：3/4/5 个工具一组
        for window in (3, 4, 5):
            if len(tool_names) < window:
                continue
            for i in range(len(tool_names) - window + 1):
                seq = tool_names[i:i + window]
                seq_json = json.dumps(seq, ensure_ascii=False)
                # 修复：hash 只用 seq_json，不含 project
                # 原因：project ID 随路径/git remote 变化，同一序列在不同 project 各存
                # 一份导致频率被稀释（无法累积到推荐阈值），跨 project 学习失效。
                h = hashlib.md5(seq_json.encode()).hexdigest()

                existing = conn.execute(
                    "SELECT id, frequency FROM tool_patterns WHERE pattern_hash=?", (h,)
                ).fetchone()
                if existing:
                    conn.execute(
                        "UPDATE tool_patterns SET frequency=frequency+1, last_seen=? WHERE pattern_hash=?",
                        (now_iso, h)
                    )
                else:
                    conn.execute(
                        """INSERT INTO tool_patterns
                           (pattern_hash, tool_sequence, context_keywords, frequency,
                            avg_duration_ms, success_rate, first_seen, last_seen, project)
                           VALUES (?, ?, ?, 1, 0, 1.0, ?, ?, ?)""",
                        (h, seq_json, kw_json, now_iso, now_iso, project)
                    )
                    written += 1
                    if written >= 30:
                        conn.commit()
                        return written

        conn.commit()
        return written
    except Exception:
        return 0


def _extract_page_fault_candidates(text: str) -> list:
    """
    提取"推理中途需要但可能没有的知识"——用于缺页日志。
    识别模式分 3 级（v3 扩展）：
      P0 显式缺口：假设/待验证/需要确认
      P1 隐式缺口：之前/上次/历史中提到过
      P2 探索性缺口：TODO/需要了解
      P3 v3新增：上下文引用缺失（提到文件/函数但没读取内容）
    """
    candidates = []
    patterns = [
        # P0: 显式知识缺口
        r'(?:假设|待验证|需要确认|需要验证)[：:]\s*(.{5,80})',
        r'(?:需要查看|需要读|应该先看|先检查)\s*(.{5,60})',
        r'(?:我不确定|不清楚|需要了解|尚不明确)\s*(.{5,60})',
        # P1: 隐式引用（暗示之前有相关决策/讨论）
        r'(?:之前|上次|此前|历史上)(?:决定|选择|讨论|分析)(?:过|了)\s*(.{5,60})',
        r'(?:根据之前的|按照此前的)\s*(.{5,60})',
        # P2: 探索性
        r'TODO[：:]\s*(.{5,80})',
        r'(?:待调查|待研究|待确认)\s*(.{5,60})',
        # P3: v3 上下文引用（提到概念但可能没有完整上下文）
        r'(?:参考|见|详见|参见)\s*(.{5,60})',
        r'(?:上文|前文|之前提到的)\s*(.{5,60})',
        r'(?:还需要|另外需要|同时需要)\s*(?:了解|确认|检查)\s*(.{5,60})',
    ]
    for pat in patterns:
        for m in re.finditer(pat, text):
            candidates.append(m.group(1).strip()[:80])
    return _deduplicate(candidates)


def _deduplicate(items: list) -> list:
    seen = set()
    result = []
    for item in items:
        key = re.sub(r'\s+', '', item.lower())
        if key not in seen and len(key) > 3:
            seen.add(key)
            result.append(item)
    return result


def _is_quality_chunk(summary: str) -> bool:
    """
    写入前质量过滤——返回 False 则丢弃。
    拦截以下噪声：
    - ] [ - | 开头（截断/前缀污染/列表项/表格行）
    - 纯 markdown 符号
    - 过短（< 10字）
    - 以助词/连词开头（说明是句子中间的截断）
    - 元数据泄漏关键词
    - 占位符
    - v4 迭代17：markdown 表格行、纯指标/数据行
    - v7 迭代74：纯验证报告、纯性能数据、无主语截断句
    - v8 迭代79：纯状态快照、模糊方向声明（无具体技术锚点）
    """
    s = summary.strip()
    if len(s) < 10:
        return False
    if re.match(r'^[\[\]\-|]', s):
        return False
    if re.match(r'^[-=*`#>]{2,}$', s):
        return False
    if re.match(r'^[了的地得把被让向从以在对和与或]', s):
        return False
    # 表格行：含 3+ 个 | 分隔符
    if s.count('|') >= 3:
        return False
    # 纯数据/指标行：全是数字、符号、单位，没有中文动词
    if not re.search(r'[\u4e00-\u9fff]{2,}', s) and re.match(r'^[\d\s.%ms/=<>×+\-,()]+$', s):
        return False
    noise_kw = ["← project 字段", "(importance=", "Stop extractor", "路径被重复写入",
                "【相关历史", "hookSpecificOutput", "additionalContext",
                "chunk_count", "recall_traces"]
    if any(kw in s for kw in noise_kw):
        return False
    placeholders = {"方案 X 是最优解", "extractor 升级", "KnowledgeRouter"}
    if s in placeholders:
        return False
    # ── 迭代74：Promotion Filter — 拦截不可复用的过程性记录 ──
    # OS 类比：Generational GC promotion filter — young gen 短命对象不提升到 old gen
    # V1 纯验证/测试报告（"N/N 通过"、"回归全绿"、"ALL PASSED"）
    if re.match(r'^\d+/\d+\s*(?:通过|passed|全绿|green|新测试)', s, re.I):
        return False
    if re.match(r'^(?:验证|回归|测试|ALL\s*PASSED)[：:]\s*\d+', s, re.I):
        return False
    # V2 纯性能/延迟数据（性能/延迟前缀 + 无决策动词 = 纯报告）
    if re.match(r'^(?:性能|延迟|耗时|avg|p\d+|latency)[：:]', s, re.I):
        # 含决策动词的保留（如"性能：采用 X 后提升 3x"）
        if not re.search(r'(?:选择|决定|采用|推荐|因为|替代|改用)', s):
            return False
    if re.match(r'^[\w_]+\s*(?:延迟|耗时)[：:]\s*[\d.]+\s*(?:ms|s)', s):
        return False
    # V3 HTML/XML 标签泄漏
    if s.startswith('<') and re.match(r'^<[a-z-]+', s):
        return False
    # ── 迭代79：Seed Pruning — 拦截不可复用的状态快照和模糊声明 ──
    # OS 类比：do_exit() → exit_mmap() — 进程退出时释放不再需要的页面
    # V4 纯状态快照（"数据规模：N chunks..."、"当前状态：..."）
    # 这些是某时刻的 point-in-time 数据，随时间失效，不是可复用决策
    if re.match(r'^(?:数据规模|当前状态|系统状态|chunk\s*数|统计|现状)[：:]', s, re.I):
        if not re.search(r'(?:选择|决定|采用|推荐|因为|替代|改用|应该)', s):
            return False
    # V5 模糊方向声明（"X — Y" 格式，且无具体技术锚点）
    # 如 "精简重构 — Less is More" — 是战略口号，不是可执行决策
    # 具体技术锚点：文件路径、函数名、数字度量、具体工具/库名
    if re.search(r'^.{3,20}\s*[—–]\s*.{3,}$', s):
        has_anchor = bool(
            re.search(r'[\w./]+\.(?:py|js|ts|json|db|sql|yaml|toml)\b', s)  # 文件路径
            or re.search(r'\d+(?:\.\d+)?(?:%|ms|s|MB|GB|次|条|个)', s)      # 数字度量
            or re.search(r'`[^`]+`', s)                                      # 代码标识符
            or re.search(r'(?:→|->)\s*\d', s)                               # 量化变化
        )
        if not has_anchor:
            return False
    # ── 迭代116：ftrace/调试计数器行过滤 ──
    # OS 类比：ftrace ring buffer 中的 event 数据，只在 debug session 有意义
    # 模式：word_cnt=N word_cnt=N ... — 多个 word=数字 键值对，是内核调试输出
    # 目标：过滤 "sub_enq_cnt=0 sub_deq_cnt=0" 类调试行
    if re.match(r'^[\w_]+=\d+(?:\s+[\w_]+=\d+)+', s):
        return False
    # 纯数值单位换算行（"N ns = M s ≈ Xh Ym" — point-in-time 计算，无决策价值）
    if re.match(r'^\d[\d\s.]*(?:ns|ms|s)\s*[=≈]', s):
        return False
    # ── 迭代88：OOM Killer V9 — 主动杀死不产出价值的知识 ──
    # OS 类比：Linux OOM Killer (Andries Brouwer, 2000) — 选择性终止消耗资源但无产出的进程
    # V6 编号列表项作为独立 decision（"2. XXX"、"3. YYY"） → 上下文碎片
    # 编号项只有在列表内才有意义，独立存储时丢失上下文
    if re.match(r'^\d+\.\s', s):
        # 保留：含具体技术锚点的编号项（文件路径/数字度量/代码标识符/量化变化）
        has_anchor = bool(
            re.search(r'[\w./]+\.(?:py|js|ts|json|db|sql|yaml|toml)\b', s)
            or re.search(r'\d+(?:\.\d+)?(?:%|ms|s|MB|GB|次|条|个)', s)
            or re.search(r'`[^`]+`', s)
            or re.search(r'(?:→|->)\s*\d', s)
        )
        if not has_anchor:
            return False
    # V7 纯迭代完成报告（"迭代N xxx 完成/修复/通过"） → 是进度日志不是决策
    # "内容：迭代86 ..." 格式的保存建议摘要
    if re.match(r'^(?:内容：)?迭代\s*\d+', s):
        # 保留含具体技术决策动词的
        if not re.search(r'(?:选择|决定|采用|替代|改用|放弃|因为|根因)', s):
            return False
    # V8 指标快照（"命中率：当前 X%"、"P99=Xms"、"性能微调"） → point-in-time 数据
    if re.match(r'^(?:命中率|覆盖率|利用率|零访问率|候选池|性能微调)[：:]', s, re.I):
        if not re.search(r'(?:选择|决定|采用|替代|改用|因为|所以)', s):
            return False
    # V9 回归验证报告 — 非 V1 格式但本质相同（"回归验证: N/N 通过 ✅"）
    if re.search(r'(?:回归|验证|regression)\s*[:：]?\s*\d+/\d+', s, re.I):
        return False
    if re.search(r'\d+/\d+\s*(?:测试|tests?)\s*(?:通过|passed|✅|全绿)', s, re.I):
        return False
    # V9b 以 N/N 开头的测试计数（"15/15 新测试"、"38/38 新测试全绿"）
    if re.match(r'^\d+/\d+\s', s):
        return False
    # ── iter89: OOM V10 — 补充三类漏网碎片 ──
    # V10a 进度条碎片（含 ████ Unicode 块字符 — 视觉展示，无语义）
    if '█' in s or '░' in s:
        return False
    # V10b 字母列表项（"A. xxx"、"B. xxx" — 与编号列表项同理，脱离上下文无意义）
    if re.match(r'^[A-Z]\.\s', s) and len(s) < 40:
        return False
    # V10c 截断残缺句（末尾以引号/逗号结尾，且无技术锚点）
    if re.search(r'[",，]$', s):
        has_anchor = bool(
            re.search(r'[\w./]+\.(?:py|js|ts|json|db|sql)\b', s)
            or re.search(r'\d+(?:\.\d+)?(?:%|ms|s|MB|GB|次|条|个)', s)
            or re.search(r'`[^`]+`', s)
        )
        if not has_anchor:
            return False
    # V10d iter90：pytest 测试输出碎片（"====...passed..."、"::Test"、"PASSED [%]"）
    if any(pattern in s for pattern in [
        "passed in",      # "12 passed in 2.77s"
        "::Test",         # pytest test path
        "PASSED [",       # "PASSED [33%]"
        "FAILED [",       # "FAILED [20%]"
    ]):
        return False
    if re.match(r'^={2,}', s):  # "====== separator"
        return False
    return True


def _is_quality_decision(summary: str) -> bool:
    """
    iter106: decision 类型专用质量过滤（SNR 提升）。
    iter107: 新增前置排除规则，防止规则文档复制品绕过过滤器。

    前置排除（优先于所有通过条件）：
      X1. [规则/...] 前缀 — 来自 self-improving wiki/memory 的规则条目，是文档行而非决策
      X2. [纠正] 前缀 — correction 记录，属于 excluded_path 语义，不应写为 decision
      X3. 以 ** 开头的 markdown 强调行（脱离文档上下文无意义）

    通过条件（满足任一）：
      A. 含决策动词（选择/决定/采用/推荐/替代/改用/放弃/因为/所以/根因）
      B. 含具体技术锚点（文件路径/数字度量/代码标识符/量化变化）
      C. 含对比句式（X 而非 Y / 相比 X，Y 更…）

    OS 类比：Promotion Filter — young gen 对象只有达到晋升条件才进入 old gen。
    """
    s = summary.strip()

    # ── 前置排除（短路，直接拒绝）──────────────────────────────
    # X1. 规则文档行（[规则/Capabilities]、[规则/Rules]、[规则/Wiki Triggers] 等）
    if re.match(r'^\[规则[/／]', s):
        return False
    # X2. 纠正记录（应写为 excluded_path，不是 decision）
    if re.match(r'^\[纠正\]', s):
        return False
    # X3. 纯 markdown 强调行（"**xxx**: yyy" 独立存储时丢失上下文）
    if re.match(r'^\*\*[^*]{2,30}\*\*[：:]\s', s) and len(s) < 80:
        return False

    # ── 通过条件（满足任一即写入）─────────────────────────────
    # A. 决策动词
    if re.search(r'(?:选择|决定|采用|推荐|替代|改用|放弃|因为|所以|根因|不选|不用|废弃|最终方案)', s):
        return True
    # B. 具体技术锚点
    if re.search(r'[\w./]+\.(?:py|js|ts|json|db|sql|yaml|toml|sh|md)\b', s):  # 文件路径
        return True
    if re.search(r'\d+(?:\.\d+)?(?:%|ms|s|MB|GB|次|条|个|行|倍|x)', s):  # 数字度量
        return True
    if re.search(r'`[^`]+`', s):  # 代码标识符
        return True
    if re.search(r'(?:→|->)\s*\d', s):  # 量化变化
        return True
    # C. 对比句式
    if re.search(r'(?:而非|而不是|不是.*而是|相比.*更|比.*更好)', s):
        return True
    return False


    # ── _already_exists / _find_similar 已迁移至 store.py（迭代21 VFS）──


def _is_tool_insight_noise(text: str) -> bool:
    """
    tool_insight 专用过滤：排除含量化数据但无决策价值的系统/调试输出。
    OS 类比：dmesg 过滤 — 内核日志中 printk(KERN_DEBUG) 不写入 audit log。

    过滤目标（经实际噪音归纳）：
    - 纯测试通过/失败行（"N passed"、"N/N 通过"）
    - 延迟测量行（"X.Xms | n=N | q=..."）
    - 系统状态快照（"decisions=N excluded=N ..."）
    - 进度/百分比行（"+85000.0% recall@3"、"Worst Queries (recall=0)"）
    - memory-os 内部日志行（"Injection avg=...ms"、"hash_changed|full"）
    - syslog/journald 行（时间戳 + 进程名 + 消息格式）
    """
    # iter114: syslog/journald 行过滤（"Apr 21 15:05:13 host process[pid]: ..."）
    # OS 类比：auditd 过滤 printk() kern.debug 级别消息
    if re.match(r'^[A-Za-z]+\s+\d+\s+\d+:\d+:\d+\s+\S+\s+\S+\[\d+\]:', text):
        return True
    # N passed / N warnings / N failed（pytest 输出）
    if re.search(r'\d+\s+(?:passed|failed|warnings?|errors?)\b', text, re.I):
        return True
    # X.Xms | n=N | q=... （检索延迟行）
    if re.match(r'[\d.]+ms\s*\|\s*n=\d+', text):
        return True
    # decisions=N excluded=N ... （extractor dmesg 行）
    if re.match(r'decisions=\d+', text):
        return True
    # Injection avg=... （retriever 统计行）
    if re.match(r'Injection\s+avg=', text):
        return True
    # hash_changed|... / skipped_same_hash / priority= （retriever reason 行）
    if re.search(r'hash_changed\||\bskipped_same_hash\b|\bpriority=(?:FULL|LITE|SKIP)\b', text):
        return True
    # 召回traces总数: ... （统计快照）
    if '召回traces总数' in text or '有效注入' in text:
        return True
    # Worst Queries / Improvement: +X% recall （eval 报告）
    if re.match(r'(?:Worst Queries|Improvement:)', text):
        return True
    # ✅/✗ + 纯延迟验证（"✅ dump 延迟 Xms < Yms"）— 过程性验证，非决策
    if re.match(r'^[✅✗❌]\s+\w+\s+延迟\s+[\d.]+ms', text):
        return True
    # "测试: N/N 通过" 或 "测试结果：N/N 通过" 格式
    if re.search(r'测试(?:结果)?\s*[:：]\s*\d+[/／]\d+\s*(?:通过|passed)', text, re.I):
        return True
    # "N/N 通过，N 失败" 格式
    if re.search(r'\d+[/／]\d+\s*通过', text):
        return True
    # CRIU/测试类完成报告
    if re.search(r'(?:Checkpoint|Restore)\s+测试\s*:\s*\d+/\d+', text):
        return True
    # iter106: eval recall 统计行（"category : recall=0.9 (N=10)"、"Q: id recall=1.0"）
    if re.search(r'recall=[01]\.\d+(?:\s*\(N=\d+\))?', text):
        return True
    # iter106: 单字母分类标签行（"R: recall=0.0"、"Q: hash recall=..."）
    if re.match(r'^[A-Z]:\s+\w', text) and 'recall' in text:
        return True
    return False


# ── 迭代39：COW 预扫描 — 写入时惰性求值 ──────────────────────
# OS 类比：Linux fork() Copy-on-Write (1991)
#   fork() 不立即复制父进程的所有页面，而是共享并标记为只读。
#   只有当进程真正写入时才触发 page fault → 复制该页面。
#   大多数 fork+exec 场景中，子进程立即 exec 新程序，
#   父进程的页面从未被修改 → 节省了大量不必要的内存复制。
#
#   memory-os 等价问题：
#     extractor Stop hook 在每次会话结束时都执行完整提取流程：
#     6+ 种正则扫描 + already_exists 全表查询 + kswapd 水位检查。
#     但 ~60% 的消息是纯代码输出/简短确认/调试信息，不含任何有价值决策。
#     COW 预扫描：先做一次极轻量检测（单次正则，< 0.1ms），
#     只有检测到信号时才触发完整提取（"写入时复制"）。

# 合并所有信号词到一个预扫描正则（union of all signal patterns 的关键词）
_COW_PRESCAN = re.compile(
    r'(?:'
    # 决策信号关键词
    r'选择|决定|采用|推荐|最终方案|方案选定|推断|结论|因此'
    r'|decided?|chosen?|adopted?|recommended?|conclusion|therefore'
    # 排除信号关键词
    r'|不用|放弃|排除|废弃|不推荐|不选|跳过'
    r'|deprecated|abandoned|rejected|skipped'
    # 推理链信号（扩充：覆盖更多实际内容中的推理标记）
    r'|根本原因|root cause|核心问题|第一性原理'
    r'|根本问题|根因|本质问题|问题在于|原因[：:]|根本差距'
    r'|这是因为|发现根因|真正原因|关键发现|核心发现|分析显示'
    r'|诊断结论|性能瓶颈|瓶颈在于|key finding|the reason|analysis shows'
    # 迭代120：新增高频推理词
    r'|这说明|这表明|这意味着|关键在于|症结在于|实质上|本质上|由此可见|这证明了|这验证了'
    # 因果链（迭代122：扩充覆盖真实LLM因果表达）
    r'|因为.*所以|由于.*因此|是因为'
    r'|导致|造成|引发|触发|引起|是由.*导致'
    r'|because|due to|caused by|resulted in|leads to'
    # 对比句式
    r'|而非|而不是|相比|比起|改用|换成'
    # 量化证据
    r'|\d+(?:\.\d+)?%|\d+(?:\.\d+)?\s*ms|\d+/\d+\s*(?:cases?|测试)'
    # 完成动作
    r'|已完成|已修复|已创建|已更新|已实现|完成了|修复了'
    r'|successfully|completed|fixed|implemented'
    # 发现/诊断
    r'|发现|诊断|定位|确认.*问题|确认.*原因'
    r'|found|diagnosed|confirmed|identified'
    # 总结标题
    r'|##?\s*(?:总结|Summary|结论|完成|结果|验证)'
    # 迭代102：设计约束新增关键词
    r'|注意不要|小心不要|务必不要|千万不要|切勿'
    r'|⚠️|⚠|WARNING:|CAUTION:|DANGER:|IMPORTANT:'
    r'|never.*because|avoid.*because|don\'t.*because'
    r'|race condition|deadlock|memory leak|data corruption'
    r'|竞态|死锁|内存泄漏|数据损坏'
    r'|否则会|否则将|只有.*才能|必须先'
    r')',
    re.IGNORECASE
)


def _cow_prescan(text: str) -> bool:
    """
    迭代39：COW 预扫描 — 快速检测消息是否可能包含有价值内容。
    OS 类比：fork() 后 MMU 检查 PTE 的 Write bit，
    只有 Write bit 被触发时才执行 copy_page()。

    策略：
      对前 3000 字符执行单次正则匹配（union of all signal keywords）。
      命中 → 返回 True（触发完整提取 "copy-on-write"）
      未命中 → 返回 False（跳过提取，只保留 page_fault 和 madvise 写入）

    预期：~60-70% 的消息不含信号词，可以跳过整个提取流程。
    性能：< 0.1ms（单次正则匹配，无 I/O）。
    """
    # 只扫描前 N 字符（决策/结论通常在消息头部或中部）
    prescan_chars = _sysctl("extractor.cow_prescan_chars")
    sample = text[:prescan_chars]
    return bool(_COW_PRESCAN.search(sample))


def _calculate_confidence(chunk_type: str, summary: str) -> float:
    """
    迭代100：ECC 初始置信度评估 — 从提取特征自动推断。
    OS 类比：CPU confidence estimation — 预测器根据历史准确率调整分支预测信心。
    """
    import re as _re
    # 量化证据（最高置信）
    if _re.search(r'\d+(?:\.\d+)?(?:ms|%|MB|GB|次|条|个)', summary):
        return 0.90
    if chunk_type == "design_constraint":
        return 0.95
    if chunk_type == "excluded_path":
        return 0.80
    if chunk_type == "reasoning_chain":
        return 0.65
    if chunk_type == "conversation_summary":
        return 0.50
    if chunk_type == "decision":
        if any(w in summary for w in ("因为", "所以", "根因", "because")):
            return 0.85
        return 0.75
    return 0.70


def _route_info_class(chunk_type: str, summary: str) -> str:
    """
    迭代300/319：五层路由 — 根据 chunk_type 和内容特征判断 info_class。

    迭代319：委托 store_vfs.classify_memory_type()，统一路由逻辑（DRY）。
    保留本函数作为兼容入口，行为与旧版相同但增加了情节/语义分层。

    五层路由：
      semantic   → decision/design_constraint/procedure/excluded_path（多次验证通用知识）
      episodic   → reasoning_chain/conversation_summary/causal_chain（会话内情节事件）
      operational → task_state/prompt_context（agent 操作配置）
      ephemeral  → 含"临时"/"本次"关键词（临时状态）
      world      → 其余（默认，中等保留）

    OS 类比：Linux VFS 文件类型路由（S_ISREG/S_ISDIR/S_ISLNK）——
      不同文件类型有不同的 page cache 策略和 eviction 优先级。
    """
    from store_vfs import classify_memory_type as _classify
    return _classify(chunk_type, summary)


def _write_chunk(chunk_type: str, summary: str, project: str, session_id: str,
                 topic: str = "", conn: sqlite3.Connection = None,
                 importance_override: float = None,
                 _txn_managed: bool = False,
                 raw_snippet: str = "",
                 content_override: str = "") -> None:
    """v5 迭代21：委托 store.py VFS 统一数据访问层。
    _txn_managed=True 时跳过内部 commit（由外层事务统一管理）。
    迭代100：新增 confidence_score 自动评估。
    迭代300：新增 info_class 三层路由。
    迭代301：新增 stability 初始值（importance * 2.0）。
    迭代306：新增 raw_snippet（写入时保真原始片段，≤500字，可选）。
    """
    importance_map = {
        "decision": 0.85,
        "reasoning_chain": 0.80,
        "excluded_path": 0.70,
        "quantitative_evidence": 0.90,  # 量化证据：最不可重建，高保护
        "causal_chain": 0.82,           # 因果链：与 reasoning_chain 同级但独立
        "procedure": 0.85,              # 可复用操作步骤/协议（wiki import 来源）
    }
    importance = importance_override if importance_override is not None else importance_map.get(chunk_type, 0.70)
    retrievability = 0.2 if chunk_type in ("reasoning_chain", "causal_chain") else 0.35

    tags = [chunk_type, project]
    if topic:
        tags.append(topic[:30])

    # 迭代324：content_override 允许调用方传入更丰富的检索内容（如因果链聚合）
    if content_override:
        content = content_override
    elif topic:
        content = f"[{chunk_type}|{topic}] {summary}"
    else:
        content = f"[{chunk_type}] {summary}"

    # 迭代100：ECC 置信度
    confidence = _calculate_confidence(chunk_type, summary)

    # 迭代300：三层路由
    info_class = _route_info_class(chunk_type, summary)
    # 迭代301：Ebbinghaus stability 初始值
    stability = importance * 2.0

    # 迭代306：raw_snippet 截断到 500 字
    raw_snippet = (raw_snippet or "")[:500]

    # 迭代315：提取编码情境（Encoding Specificity, Tulving 1973）
    try:
        from store_vfs import extract_encoding_context as _extract_enc_ctx
        encoding_context = _extract_enc_ctx(summary)
    except Exception:
        encoding_context = {}

    chunk = MemoryChunk(
        project=project,
        source_session=session_id,
        chunk_type=chunk_type,
        info_class=info_class,
        content=content,
        summary=summary,
        raw_snippet=raw_snippet,
        tags=tags,
        importance=importance,
        retrievability=retrievability,
        stability=stability,
        encoding_context=encoding_context,
    )

    should_close = conn is None
    if conn is None:
        conn = open_db()
        ensure_schema(conn)

    try:
        # 迭代59：全类型 KSM Dedup — 传入 chunk_type 精确去重
        if already_exists(conn, summary, chunk_type=chunk_type):
            dmesg_log(conn, DMESG_DEBUG, "extractor",
                      f"ksm_skip: {chunk_type} exact dup '{summary[:40]}'",
                      session_id=session_id, project=project)
            if not _txn_managed:
                conn.commit()
            return
        if merge_similar(conn, summary, chunk_type, importance, project=project):
            dmesg_log(conn, DMESG_DEBUG, "extractor",
                      f"ksm_merge: {chunk_type} similar '{summary[:40]}'",
                      session_id=session_id, project=project)
            if not _txn_managed:
                conn.commit()
            return
        insert_chunk(conn, chunk.to_dict())

        # ── 迭代318：Summary 三元组自动抽取 ────────────────────────────────
        # 每次写入 chunk 时，从 summary 提取关系边写入 entity_edges，
        # 使 spreading_activation 能沿图扩散而不是空转。
        # OS 类比：Linux inode 写入时同步更新 dentry cache —
        #   不是等批处理，而是在写路径上顺手维护索引。
        try:
            _new_row = conn.execute(
                "SELECT id FROM memory_chunks WHERE summary=? AND chunk_type=? "
                "ORDER BY created_at DESC LIMIT 1",
                (summary, chunk_type),
            ).fetchone()
            _cid = _new_row[0] if _new_row else chunk.id
            extract_and_write_summary_triples(summary, _cid, project, conn)
        except Exception:
            pass  # 三元组抽取失败不影响主流程

        # ── 迭代320：情感显著性 importance 调整 ──────────────────────────────
        # 在写入后立即用情感唤醒词调整 importance，
        # 崩溃/关键/突破类信息自动上调，已解决/废弃类下调。
        # OS 类比：Linux OOM Killer oom_score_adj 写入 —
        #   fork 后进程可自我声明重要性，在 OOM 压力下决定存活顺序。
        try:
            _new_row2 = conn.execute(
                "SELECT id, importance FROM memory_chunks WHERE summary=? AND chunk_type=? "
                "ORDER BY created_at DESC LIMIT 1",
                (summary, chunk_type),
            ).fetchone()
            if _new_row2:
                _cid2, _cur_imp = _new_row2[0], _new_row2[1] or importance
                from store_vfs import apply_emotional_salience
                apply_emotional_salience(conn, _cid2, summary, _cur_imp)
        except Exception:
            pass  # 情感调整失败不影响主流程

        # 迭代100：IPC 广播知识更新（OS 类比：inotify — 文件变更通知）
        try:
            from store_vfs import ipc_broadcast_knowledge_update
            ipc_broadcast_knowledge_update(conn, session_id, project,
                                           {"chunk_type": chunk_type, "action": "insert"})
        except Exception:
            pass  # IPC 失败不阻塞提取主流程
        if not _txn_managed:
            conn.commit()
    finally:
        if should_close:
            conn.close()


def _write_page_fault_log(candidates: list, session_id: str) -> None:
    """
    写缺页日志——下轮 UserPromptSubmit 优先加载这些知识缺口。
    v3 闭环升级：
    - fault_count: 同一缺口出现次数（热缺页识别）
    - resolved: 是否已被 retriever 消费并成功补入
    - 重复缺口自增 fault_count 而非重复添加

    iter259: per-session 文件——每个 agent/session 写独立文件，消除并发 overwrite 竞态。
    OS 类比：/proc/PID/pagemap — 每进程独立文件，不同进程间互不干扰。
    命名：page_fault_log.<session_id[:8]>.json（session_id 有效时）
          page_fault_log.json（session_id 为空/"unknown" 时，向后兼容）
    """
    if not candidates:
        return
    # iter259: per-session file — 消除多 agent 并发写竞态
    _sid_tag = session_id[:8] if (session_id and session_id != "unknown") else ""
    if _sid_tag:
        log_path = MEMORY_OS_DIR / f"page_fault_log.{_sid_tag}.json"
    else:
        log_path = MEMORY_OS_DIR / "page_fault_log.json"
    existing = []
    if log_path.exists():
        try:
            existing = json.loads(log_path.read_text())
        except Exception:
            existing = []

    # 建索引：query → entry（用于去重和自增 fault_count）
    query_index = {}
    for entry in existing:
        if isinstance(entry, dict) and "query" in entry:
            # 兼容旧格式（无 fault_count 字段）
            if "fault_count" not in entry:
                entry["fault_count"] = 1
            if "resolved" not in entry:
                entry["resolved"] = False
            q_key = re.sub(r'\s+', '', entry["query"].lower())
            query_index[q_key] = entry

    now_iso = datetime.now(timezone.utc).isoformat()
    for c in candidates:
        q_key = re.sub(r'\s+', '', c.lower())
        if q_key in query_index:
            # 已存在：自增 fault_count，更新时间戳
            query_index[q_key]["fault_count"] += 1
            query_index[q_key]["ts"] = now_iso
            query_index[q_key]["resolved"] = False  # 重新出现说明未真正解决
        else:
            query_index[q_key] = {
                "query": c, "session_id": session_id,
                "ts": now_iso, "fault_count": 1, "resolved": False,
            }

    # 按 fault_count 降序排列（热缺页优先），只保留最近 20 条未解决的
    all_entries = list(query_index.values())
    unresolved = [e for e in all_entries if not e.get("resolved", False)]
    unresolved.sort(key=lambda e: e.get("fault_count", 1), reverse=True)
    merged = unresolved[:20]

    log_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2))


def _extract_topic_entities(text: str, decisions: list, excluded: list,
                            reasoning: list, summaries: list,
                            topic: str) -> list:
    """
    迭代32：从对话内容提取主题实体，用于 madvise hint。
    OS 类比：应用程序分析访问模式，告知内核预读区域。

    提取策略（4 层，按信号强度排序）：
      H1 显式话题：markdown 标题中的关键词
      H2 决策实体：从 decisions/reasoning 中提取被引号/反引号包裹的标识符
      H3 技术词：文件路径、函数名、类名等代码标识符
      H4 高频中文主题词：出现 ≥ 2 次的中文双字词

    返回去重后的实体列表（最多 max_hints 个）。
    """
    entities = []
    seen = set()

    def _add(word):
        key = word.lower().strip()
        if len(key) >= 2 and key not in seen and key not in _MADVISE_STOPWORDS:
            seen.add(key)
            entities.append(word.strip())

    # H1: 话题标题
    if topic:
        for w in re.findall(r'[a-zA-Z][a-zA-Z0-9_]{2,20}', topic):
            _add(w)
        cn = re.sub(r'[^\u4e00-\u9fff]', '', topic)
        for i in range(len(cn) - 1):
            _add(cn[i:i + 2])

    # H2: 决策/推理中的标识符
    all_summaries = decisions + excluded + reasoning + summaries
    for s in all_summaries:
        # 反引号内容
        for m in re.finditer(r'`([^`]{2,30})`', s):
            _add(m.group(1))
        # 英文技术词
        for m in re.finditer(r'\b([a-zA-Z][a-zA-Z0-9_]{2,20})\b', s):
            _add(m.group(1))

    # H3: 原文中的文件路径和代码标识符
    for m in re.finditer(r'[\w./]+\.(?:py|js|ts|md|json|db|sql|yaml|toml)\b', text[:5000]):
        _add(m.group(0))
    for m in re.finditer(r'`([^`]{2,30})`', text[:5000]):
        _add(m.group(1))

    # H4: 高频中文双字词（出现 ≥ 2 次）
    cn_text = re.sub(r'[^\u4e00-\u9fff]', '', text[:5000])
    bigram_count = {}
    for i in range(len(cn_text) - 1):
        bg = cn_text[i:i + 2]
        bigram_count[bg] = bigram_count.get(bg, 0) + 1
    for bg, cnt in sorted(bigram_count.items(), key=lambda x: -x[1]):
        if cnt >= 2:
            _add(bg)

    return entities


# madvise 停用词（排除常见虚词和通用编程词）
_MADVISE_STOPWORDS = frozenset({
    "the", "and", "for", "that", "this", "with", "from", "are", "was", "has",
    "not", "but", "can", "will", "all", "one", "its", "had", "been", "each",
    "which", "their", "than", "other", "into", "more", "some", "such", "when",
    "use", "used", "using", "new", "old", "set", "get", "add", "run", "see",
    "now", "way", "may", "also", "per", "via", "yet", "out", "how", "why",
    "def", "class", "import", "return", "true", "false", "none", "self",
    "的", "了", "在", "是", "和", "有", "不", "这", "到", "我", "们",
})


def _write_madvise_hints(text: str, decisions: list, excluded: list,
                         reasoning: list, summaries: list,
                         project: str, session_id: str, topic: str) -> None:
    """
    迭代32：写入 madvise hints（MADV_WILLNEED）。
    从本轮对话内容提取主题实体，作为下一轮检索的预热 hint。
    """
    hints = _extract_topic_entities(text, decisions, excluded, reasoning,
                                    summaries, topic)
    if hints:
        madvise_write(project, hints, session_id)


def _detect_and_write_entities(text: str, project: str, session_id: str,
                               conn) -> int:
    """
    迭代303：轻量 NER — 识别新出现实体并写入 entity_stub chunk。

    OS 类比：Linux inotify/dnotify — 目录事件驱动，新文件（实体）出现时自动建立
      inode stub，后续对该实体的访问直接 dentry cache 命中，无需重新解析路径。

    三类实体（不调 LLM，纯正则，目标 < 3ms）：
      T1 GitHub 仓库：org/repo 格式（如 garrytan/gbrain）
      T2 技术项目名：首字母大写或全大写的英文词（≥4字符），排除常用英文词
      T3 中文专有词：被引号/书名号/「」包围的中文词（2-10字）

    去重：already_exists 检查，只写首次出现的实体。
    info_class 固定为 'world'（实体是关于世界的事实）。
    importance 低（0.40），stability 初始低（0.5），靠后续引用自然加固。
    """
    if not text:
        return 0

    entities = set()

    # T1: GitHub 仓库（owner/repo 格式）
    for m in re.finditer(r'\b([a-zA-Z0-9_-]{2,30}/[a-zA-Z0-9_.-]{2,40})\b', text):
        candidate = m.group(1)
        # 排除文件路径（含多个 /）和常见假阳性，增加更严格的字母检查
        if candidate.count('/') == 1 and not candidate.endswith(('.py', '.js', '.ts', '.md')):
            left, right = candidate.split('/', 1)
            # 两侧必须包含英文字母（排除中文片段误匹配）
            if (re.search(r'[a-zA-Z]', left) and re.search(r'[a-zA-Z]', right)
                    and 2 <= len(left) <= 20 and 2 <= len(right) <= 40):
                entities.add(('github_repo', candidate))

    # T2: 技术项目名（首字母大写英文，≥4字符，非常用词）
    # iter328: 加严过滤 — 必须满足以下任一"技术性"特征：
    #   a. 驼峰（含内嵌大写，如 MemoryChunk, FTS5, BM25）
    #   b. 含数字（FTS5, iter328）
    #   c. 含下划线（snake_case 工具名）
    #   d. 全大写缩写（≥3字，如 BM25, FTS, OOM）
    # 排除：纯首字母大写普通英文词（Best, Note, True, Walker 等）
    _COMMON_EN = frozenset({
        'This', 'That', 'When', 'With', 'From', 'Into', 'Over', 'Under',
        'After', 'Before', 'During', 'While', 'Since', 'Until', 'Through',
        'About', 'Because', 'Though', 'Although', 'However', 'Therefore',
        'True', 'False', 'None', 'Note', 'Also', 'Even', 'Just', 'Like',
        'Step', 'Type', 'List', 'Dict', 'String', 'Class', 'Model', 'Data',
    })
    for m in re.finditer(r'\b([A-Z][a-zA-Z0-9_]{3,25})\b', text):
        word = m.group(1)
        if word in _COMMON_EN:
            continue
        # 技术性特征检测
        _has_tech = (
            bool(re.search(r'[A-Z]', word[1:]))   # 驼峰（内嵌大写）
            or bool(re.search(r'\d', word))         # 含数字
            or '_' in word                           # 含下划线
            or word.isupper() and len(word) >= 3    # 全大写缩写
        )
        if _has_tech:
            entities.add(('tech_entity', word))

    # T3: 中文专有词（引号/书名号/「」包围）
    # iter328: 加严过滤 — 必须包含至少一个英文字母或数字（技术术语特征），
    # 排除纯中文普通短语（"早饭通常有什么"、"不留无用代码"）
    for m in re.finditer(r'[「「《""]([^\u0000-\u007f「」《》""]{2,10})[」」》""]', text):
        cn_word = m.group(1).strip()
        if cn_word and re.search(r'[a-zA-Z0-9_]', cn_word):
            # 必须含英文/数字才认为是技术命名概念
            entities.add(('named_concept', cn_word))

    if not entities:
        return 0

    count = 0
    for etype, name in entities:
        summary = f"[entity:{etype}] {name}"
        # 去重：只写首次出现
        if already_exists(conn, summary, chunk_type="entity_stub"):
            continue
        _write_chunk(
            "entity_stub", summary, project, session_id,
            topic=etype, conn=conn,
            importance_override=0.40,
            _txn_managed=True,
        )
        count += 1

    return count


# ── 迭代304：关系三元组提取（知识图谱边）────────────────────────────────
# OS 类比：Linux modprobe 依赖解析 — 从 modules.dep 文本提取 "A: B C" 格式的
#   依赖关系，写入内核模块依赖图。纯文本正则，不调 LLM，目标 < 5ms。
#
# 支持模式：
#   uses        — "X 使用/采用/基于 Y"
#   depends_on  — "X 依赖/需要 Y"
#   part_of     — "X 是 Y 的一部分/子模块/子系统"
#   implements  — "X 实现了 Y"

# 实体词 token（中英文标识符，1-40字符）
_ENTITY_PAT = r'[\w\u4e00-\u9fff][\w\u4e00-\u9fff\-\.\/]{0,39}'

_RELATION_PATTERNS = [
    # uses: X 使用/采用/基于 Y
    (re.compile(
        rf'({_ENTITY_PAT})\s*(?:使用|采用|基于|调用|依托)\s*({_ENTITY_PAT})',
        re.UNICODE,
    ), 'uses'),
    # uses: X uses/utilizes/is built on Y (英文)
    (re.compile(
        rf'({_ENTITY_PAT})\s+(?:uses?|utilizes?|is built on|relies on)\s+({_ENTITY_PAT})',
        re.IGNORECASE | re.UNICODE,
    ), 'uses'),
    # depends_on: X 依赖/需要 Y
    (re.compile(
        rf'({_ENTITY_PAT})\s*(?:依赖|需要|要求)\s*({_ENTITY_PAT})',
        re.UNICODE,
    ), 'depends_on'),
    # depends_on: X depends on/requires Y (英文)
    (re.compile(
        rf'({_ENTITY_PAT})\s+(?:depends? on|requires?)\s+({_ENTITY_PAT})',
        re.IGNORECASE | re.UNICODE,
    ), 'depends_on'),
    # part_of: X 是 Y 的一部分/子模块/子系统/组成部分
    (re.compile(
        rf'({_ENTITY_PAT})\s+是\s+({_ENTITY_PAT})\s*的(?:一部分|子模块|子系统|组成部分|模块)',
        re.UNICODE,
    ), 'part_of'),
    # part_of: X is part of / a submodule of Y (英文)
    (re.compile(
        rf'({_ENTITY_PAT})\s+is\s+(?:part of|a submodule of|a module of)\s+({_ENTITY_PAT})',
        re.IGNORECASE | re.UNICODE,
    ), 'part_of'),
    # implements: X 实现了/实现 Y
    (re.compile(
        rf'({_ENTITY_PAT})\s+实现了?\s+({_ENTITY_PAT})',
        re.UNICODE,
    ), 'implements'),
    # implements: X implements Y (英文)
    (re.compile(
        rf'({_ENTITY_PAT})\s+implements?\s+({_ENTITY_PAT})',
        re.IGNORECASE | re.UNICODE,
    ), 'implements'),

    # ── 迭代318：补充关系模式 — 覆盖浓缩句/陈述句结构 ──────────────────
    # superseded_by: X 被 Y 替代/取代
    (re.compile(
        rf'({_ENTITY_PAT})\s*(?:被|改为|换为|替换为|迁移到)\s*({_ENTITY_PAT})',
        re.UNICODE,
    ), 'superseded_by'),
    # superseded_by: X replaced by / migrated to Y
    (re.compile(
        rf'({_ENTITY_PAT})\s+(?:replaced? by|migrated? to|superseded? by)\s+({_ENTITY_PAT})',
        re.IGNORECASE | re.UNICODE,
    ), 'superseded_by'),
    # writes_to / reads_from: X 写入/读取 Y
    (re.compile(
        rf'({_ENTITY_PAT})\s*(?:写入|写到|存入|持久化到)\s*({_ENTITY_PAT})',
        re.UNICODE,
    ), 'writes_to'),
    (re.compile(
        rf'({_ENTITY_PAT})\s*(?:读取|查询|检索自)\s*({_ENTITY_PAT})',
        re.UNICODE,
    ), 'reads_from'),
    # writes_to (英文): X writes to / persists to Y
    (re.compile(
        rf'({_ENTITY_PAT})\s+(?:writes? to|persists? to|stores? in|inserts? into)\s+({_ENTITY_PAT})',
        re.IGNORECASE | re.UNICODE,
    ), 'writes_to'),
    # calls: X 调用 Y（函数/模块级调用关系）
    (re.compile(
        rf'({_ENTITY_PAT})\s*调用\s*({_ENTITY_PAT})',
        re.UNICODE,
    ), 'calls'),
    (re.compile(
        rf'({_ENTITY_PAT})\s+calls?\s+({_ENTITY_PAT})',
        re.IGNORECASE | re.UNICODE,
    ), 'calls'),
    # related_to: X 与 Y 相关/关联
    (re.compile(
        rf'({_ENTITY_PAT})\s*与\s*({_ENTITY_PAT})\s*(?:相关|关联|配合|协同)',
        re.UNICODE,
    ), 'related_to'),
    # triggers: X 触发 Y
    (re.compile(
        rf'({_ENTITY_PAT})\s*触发\s*({_ENTITY_PAT})',
        re.UNICODE,
    ), 'triggers'),
    (re.compile(
        rf'({_ENTITY_PAT})\s+triggers?\s+({_ENTITY_PAT})',
        re.IGNORECASE | re.UNICODE,
    ), 'triggers'),
]

# 噪声实体过滤（太短、纯数字、常用停用词）
_ENTITY_STOPWORDS = frozenset({
    # 英文基础停用词
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'it', 'its', 'this', 'that', 'these', 'those', 'and', 'or', 'but',
    'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as',
    'true', 'false', 'none', 'null', 'not',
    # 英文扩展停用词
    'can', 'will', 'may', 'should', 'would', 'could', 'must', 'shall',
    'have', 'has', 'had', 'do', 'does', 'did', 'done', 'make', 'use',
    'get', 'set', 'add', 'new', 'old', 'all', 'any', 'each', 'some',
    'one', 'two', 'our', 'we', 'you', 'they', 'he', 'she',
    # 中文停用词（高频虚词）
    '选择', '决定', '推荐', '采用', '使用', '应该', '需要', '可以',
    '因为', '所以', '因此', '但是', '然而', '如果', '当然', '另外',
    '问题', '方案', '方法', '系统', '模块', '功能', '实现', '设计',
    '代码', '文件', '数据', '信息', '内容', '部分', '进行', '提供',
    '包括', '通过', '基于', '关于', '对于', '来自', '目前', '已经',
    '可能', '需要', '没有', '不是', '这个', '那个', '一个', '一种',
    '迭代', '版本', '测试', '验证', '修复', '添加', '删除', '更新',
    '注意', '警告', '结果', '分析', '总结', '说明', '解释', '描述',
})


def _is_noise_entity(s: str) -> bool:
    """过滤噪声实体：太短/纯数字/停用词/多词短语/过长。"""
    if len(s) < 2:
        return True
    if len(s) > 40:
        return True
    if s.isdigit():
        return True
    if s.lower() in _ENTITY_STOPWORDS:
        return True
    # 多词短语不是实体（超过2个空格分隔的词）
    if len(s.split()) > 2:
        return True
    # 纯中文词超过4字通常是句子片段，不是实体名
    cn_only = re.sub(r'[^\u4e00-\u9fff]', '', s)
    if len(cn_only) >= len(s) * 0.8 and len(cn_only) > 4:
        return True
    return False


def _extract_entity_relations(text: str, project: str, session_id: str, conn) -> int:
    """
    迭代304：从文本提取关系三元组并写入 entity_edges。
    纯正则，不调 LLM，目标 < 5ms。

    OS 类比：Linux modprobe 依赖解析 — 读取 modules.dep 文本，
      用正则提取 "module_a: module_b module_c" 依赖关系，
      构建内核模块加载顺序图（kmod_module_new_from_name + dep traversal）。

    返回写入的边数量。
    """
    if not text:
        return 0

    # 避免循环导入：延迟导入 insert_edge
    try:
        from store_vfs import insert_edge
    except ImportError:
        try:
            from store import insert_edge  # type: ignore
        except ImportError:
            return 0

    count = 0
    seen = set()  # 去重同一次提取中的重复三元组

    for pattern, relation in _RELATION_PATTERNS:
        for m in pattern.finditer(text):
            from_e = m.group(1).strip()
            to_e = m.group(2).strip()

            # 过滤噪声
            if _is_noise_entity(from_e) or _is_noise_entity(to_e):
                continue
            # 避免自指边（X uses X）
            if from_e.lower() == to_e.lower():
                continue

            triple = (from_e, relation, to_e)
            if triple in seen:
                continue
            seen.add(triple)

            try:
                insert_edge(conn, from_e, relation, to_e,
                            project=project, confidence=0.7)
                count += 1
            except Exception:
                pass  # 单边失败不影响整体

    return count


# ── 迭代318：Summary 专用三元组抽取 ─────────────────────────────────────────
# OS 类比：Linux /proc/net/dev — 针对网络接口统计的专用解析器，
#   不用通用的 sysfs 读取路径，因为格式不同，解析策略也不同。
#
# 问题：_RELATION_PATTERNS 是为长文本（assistant message）设计的；
#   chunk summary 是浓缩句（"BM25 检索性能足够"、"retriever 调用 FTS5"），
#   主语通常是第一个技术词，谓语在中间，宾语在最后。
#
# 策略：
#   1. 从 summary 中抽取候选技术实体（英文标识符 + 短 CJK 词）
#   2. 检测谓语关键词确定 relation
#   3. 将候选实体对 + relation 写入 entity_edges
#
# 实体候选标准：
#   - 英文词：长度 ≥ 3，非停用词，驼峰/下划线/全大写优先
#   - CJK 词：2-4 字，不在停用词表，通常是模块名/概念名

_SUMMARY_ENTITY_PAT = re.compile(
    r'(?:'
    r'[A-Z][a-zA-Z0-9_]{2,}'        # 驼峰/首字母大写（MemoryChunk, BM25）
    r'|[a-z][a-z0-9_]{3,}'           # 小写标识符（retriever, kswapd）
    r'|[A-Z]{2,}[0-9_]*[A-Z0-9]*'   # 全大写缩写（FTS5, BM25, VFS）
    r')',
    re.UNICODE,
)
# CJK 实体单独处理：只允许 2-3 字的技术词，且必须是纯汉字（不含数字/标点）
_SUMMARY_CJK_ENTITY_PAT = re.compile(r'[\u4e00-\u9fff]{2,3}')


# 谓语关键词 → relation 映射（短句专用）
_SUMMARY_PREDICATES = [
    # 使用/依赖
    (re.compile(r'(?:使用|采用|基于|调用|依赖|读取|查询|依托)'), 'uses'),
    (re.compile(r'(?:uses?|calls?|queries?|reads?|relies? on)', re.IGNORECASE), 'uses'),
    # 实现/包含
    (re.compile(r'(?:实现|包含|提供|支持)'), 'implements'),
    (re.compile(r'(?:implements?|provides?|supports?)', re.IGNORECASE), 'implements'),
    # 写入/存储
    (re.compile(r'(?:写入|存入|持久化|写到)'), 'writes_to'),
    (re.compile(r'(?:writes?|persists?|stores?)', re.IGNORECASE), 'writes_to'),
    # 替代/取代：主动句 "X 改用/换成 Y" 或 "X replaces Y"
    (re.compile(r'(?:改用|换成|迁移到|迁移至|替换为)'), 'superseded_by'),
    (re.compile(r'(?:replaces?|supersedes?|migrates? to)', re.IGNORECASE), 'superseded_by'),
    # 被动替代："X 被 Y 替代" — 谓语是"被 Y"整体（被+宾语），主动宾颠倒
    # 用独立正则直接抽取，不走通用谓语左右分割逻辑
    # 注意：此模式在下面 _PASSIVE_REPLACE_PAT 单独处理
    # 触发/唤醒
    (re.compile(r'(?:触发|唤醒|激活)'), 'triggers'),
    (re.compile(r'(?:triggers?|activates?|wakes?)', re.IGNORECASE), 'triggers'),
    # 依赖/需要
    (re.compile(r'(?:依赖|需要)'), 'depends_on'),
    (re.compile(r'(?:depends? on|requires?)', re.IGNORECASE), 'depends_on'),
]


def extract_summary_triples(summary: str) -> list:
    """
    迭代318：从 chunk summary 短句中抽取三元组列表。
    返回 [(from_entity, relation, to_entity), ...] 列表。
    不调 DB，纯文本处理，< 1ms。

    算法：
      1. 检测谓语关键词确定关系类型和分割点
      2. 以谓语为分界，左侧最后一个技术实体 = from，右侧第一个技术实体 = to
      3. 过滤噪声实体
    """
    if not summary or len(summary) < 5:
        return []

    triples = []
    seen = set()

    # ── 被动替代句专项处理："X 被 Y 替代/取代/替换" ─────────────────────────
    # 主动 to-entity 是 "被" 后面的词（Y），from-entity 是 "被" 前面的词（X）
    _PASSIVE_REPLACE_PAT = re.compile(
        r'([A-Z][a-zA-Z0-9_]{1,}|[a-z][a-z0-9_]{2,}|[A-Z]{2,}[0-9_]*)'
        r'\s+被\s+'
        r'([A-Z][a-zA-Z0-9_]{1,}|[a-z][a-z0-9_]{2,}|[A-Z]{2,}[0-9_]*)'
        r'\s*(?:替代|取代|替换)',
        re.UNICODE,
    )
    for m in _PASSIVE_REPLACE_PAT.finditer(summary):
        from_e = m.group(1).strip()
        to_e = m.group(2).strip()
        if not _is_noise_entity(from_e) and not _is_noise_entity(to_e):
            if from_e.lower() != to_e.lower():
                triple = (from_e, 'superseded_by', to_e)
                if triple not in seen:
                    seen.add(triple)
                    triples.append(triple)

    def _find_entities(text: str) -> list:
        """从文本中提取英文技术实体（不含 CJK 片段，CJK 实体另走严格路径）。"""
        candidates = _SUMMARY_ENTITY_PAT.findall(text)
        return [e for e in candidates if not _is_noise_entity(e)]

    for pred_re, relation in _SUMMARY_PREDICATES:
        for m in pred_re.finditer(summary):
            pred_start = m.start()
            pred_end = m.end()

            # 谓语左侧：找最近的技术实体（from）
            left_text = summary[:pred_start]
            left_entities = _find_entities(left_text)
            if not left_entities:
                continue
            from_e = left_entities[-1]  # 最近的

            # 谓语右侧：找第一个技术实体（to）
            right_text = summary[pred_end:]
            right_entities = _find_entities(right_text)
            if not right_entities:
                continue
            to_e = right_entities[0]  # 最近的

            # 两边都必须有英文标识符特征（至少一个含英文字母）
            # 防止纯中文片段噪声（"记录 → 的"）
            has_alpha_from = bool(re.search(r'[a-zA-Z]', from_e))
            has_alpha_to = bool(re.search(r'[a-zA-Z]', to_e))
            if not has_alpha_from and not has_alpha_to:
                continue  # 两边都是纯中文 → 跳过（质量差）

            # 过滤自指
            if from_e.lower() == to_e.lower():
                continue

            triple = (from_e, relation, to_e)
            if triple not in seen:
                seen.add(triple)
                triples.append(triple)

    return triples


def extract_and_write_summary_triples(
    summary: str,
    chunk_id: str,
    project: str,
    conn,
) -> int:
    """
    迭代318：从 chunk summary 提取三元组并写入 entity_edges。
    在 _write_chunk() 中调用，每次写 chunk 时自动触发。
    返回写入的边数。
    """
    if not summary or not project:
        return 0

    triples = extract_summary_triples(summary)
    if not triples:
        return 0

    try:
        from store_vfs import insert_edge
    except ImportError:
        try:
            from store import insert_edge  # type: ignore
        except ImportError:
            return 0

    count = 0
    for from_e, relation, to_e in triples:
        try:
            insert_edge(
                conn, from_e, relation, to_e,
                project=project,
                source_chunk_id=chunk_id,
                confidence=0.75,  # summary 来的比长文本更精确，稍高
            )
            count += 1
        except Exception:
            pass

    return count


def main():
    import time as _time
    _t_start = _time.time()

    try:
        raw = sys.stdin.read()
        hook_input = json.loads(raw) if raw.strip() else {}
    except Exception:
        hook_input = {}

    text = hook_input.get("last_assistant_message", "")
    if not text or len(text) < _sysctl("extractor.min_length"):
        sys.exit(0)

    # ── iter260: Async Pool Offload ──────────────────────────────────────────
    # OS 类比：queue_work(pool, &work) — Stop hook 提交 work_struct 到 kworker pool，
    #   立即返回（< 5ms），让 extractor_pool 常驻进程异步处理 I/O 密集的 transcript parsing。
    # pool 未运行（首次启动/崩溃）时退化到同步执行（fallback 路径，与旧行为等价）。
    try:
        project    = resolve_project_id()
        session_id = (hook_input.get("session_id", "")
                      or os.environ.get("CLAUDE_SESSION_ID", "")
                      or "unknown")
        from hooks.extractor_pool import submit_extract_task
        if submit_extract_task(hook_input, project, session_id):
            # 成功入队 → Stop hook 立即返回
            sys.exit(0)
        # pool 未运行 → fallback 到下方同步路径
    except Exception:
        pass  # import 失败 / 任何异常都 fallback 到同步执行
    # ── 同步 fallback 路径（pool 未运行时） ─────────────────────────────────

    # ── 时间片调度：长消息自适应截断（OS 类比：scheduler time-slice）
    # 超过阈值时，只处理前 N 个最可能含决策的段落（标题+代码块+短段落）
    MAX_CHARS = _sysctl("extractor.max_input_chars")
    if len(text) > MAX_CHARS:
        # 保留：标题行、含信号词的行、代码块首尾行
        signal_words = r'(?:选择|决定|推荐|结论|采用|排除|不用|放弃|根本原因|为什么|核心)'
        filtered_lines = []
        in_code = False
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith('```'):
                in_code = not in_code
                if len('\n'.join(filtered_lines)) < MAX_CHARS:
                    filtered_lines.append(line)
                continue
            if in_code:
                continue
            if (stripped.startswith('#') or re.search(signal_words, stripped)
                    or stripped.startswith('>') or stripped.startswith('|')):
                filtered_lines.append(line)
        text = '\n'.join(filtered_lines)[:MAX_CHARS]

    project = resolve_project_id()
    # 迭代66：优先从 hook stdin 获取 session_id（权威来源）
    session_id = (hook_input.get("session_id", "")
                  or os.environ.get("CLAUDE_SESSION_ID", "")
                  or "unknown")

    # 增强1+2：从 hook stdin 获取 transcript_path
    transcript_path = hook_input.get("transcript_path", "")

    # ── 迭代39：COW 预扫描 — 写入时惰性求值 ──
    # OS 类比：fork() COW — MMU 检查 Write bit，未触发则跳过 copy_page()
    # 快速检测消息是否可能包含有价值内容
    # 未命中时跳过完整提取流程，仅保留 page_fault 和 madvise 写入
    #
    # 增强3：长消息（> 500 字符）必然含有价值内容，跳过 prescan 直接进入完整提取
    # OS 类比：大分配请求（alloc_pages(order>=4)）绕过 per-CPU pageset 直接走 zone 分配
    _text_long = len(text) > 500
    cow_hit = _text_long or _cow_prescan(text)
    if not cow_hit:
        # COW miss：消息不含任何信号词，跳过完整提取
        # 仍然执行 page_fault 提取（知识缺口检测不依赖信号词）
        page_faults = _extract_page_fault_candidates(text)
        _write_page_fault_log(page_faults, session_id)
        # 写入 madvise hints（即使无提取物，也记录话题趋势）
        topic = _extract_topic(text)
        _write_madvise_hints(text, [], [], [], [], project, session_id, topic)
        # dmesg: COW skip
        try:
            conn = open_db()
            ensure_schema(conn)
            dmesg_log(conn, DMESG_DEBUG, "extractor",
                      f"cow_skip: text_len={len(text)} prescan=miss long_msg={_text_long} faults={len(page_faults)} {(_time.time()-_t_start)*1000:.1f}ms",
                      session_id=session_id, project=project)
            conn.commit()
            conn.close()
        except Exception:
            pass
        sys.exit(0)

    topic = _extract_topic(text)

    # ── 迭代50：TCP AIMD — 查询当前拥塞窗口决定提取策略 ──
    # OS 类比：TCP 发送端在发包前检查 cwnd，cwnd 决定可发送的数据量
    # 先打开 conn 查询 AIMD（需要读 recall_traces 统计命中率）
    aimd_policy = "full"  # 默认全速
    aimd_info = None
    try:
        _aimd_conn = open_db()
        ensure_schema(_aimd_conn)
        aimd_info = aimd_window(_aimd_conn, project)
        aimd_policy = aimd_info["policy"]
        _aimd_conn.close()
    except Exception:
        pass  # AIMD 失败不影响主流程（fallback 到 full）

    # ── 提取各类 chunk（v3 多模式提取）── COW hit: 触发完整提取
    decisions = (
        _extract_by_signals(text, DECISION_SIGNALS)
        + _extract_structured_decisions(text)
    )

    excluded = _extract_by_signals(text, EXCLUDED_SIGNALS)
    reasoning = _extract_by_signals(text, REASONING_SIGNALS)

    # v3 新增：对比句式（同时产出决策和排除）
    comp_decisions, comp_exclusions = _extract_comparisons(text)
    decisions.extend(comp_decisions)
    excluded.extend(comp_exclusions)

    # v3 新增：因果链
    # iter105: causal_chain 独立类型，不混入 reasoning_chain
    causal_chains = _extract_causal_chains(text)
    causal_chains = _deduplicate(causal_chains)

    # v3 新增：量化证据（作为 decision，importance 最高）
    quant_conclusions = _extract_quantitative_conclusions(text)
    decisions.extend(quant_conclusions)

    # 迭代98 新增：设计约束（系统级"为什么不这样做"知识）
    constraints = _extract_constraints(text)

    # 全局去重
    decisions = _deduplicate(decisions)
    excluded = _deduplicate(excluded)
    reasoning = _deduplicate(reasoning)
    constraints = _deduplicate(constraints)

    # v4 新增：对话摘要
    # 增强1：从 transcript 尾部额外 5 轮消息补充提取
    _transcript_extra = _read_transcript_tail(transcript_path) if transcript_path else []
    conv_summaries = _extract_conversation_summary(text, extra_texts=_transcript_extra)

    page_faults = _extract_page_fault_candidates(text)

    # ── 迭代50：AIMD 策略过滤 — 根据 cwnd 策略裁剪提取物 ──
    # OS 类比：TCP cwnd 限制发送窗口，cwnd 小时只发高优先级数据
    # conservative: 只保留 decision + reasoning_chain + 量化证据（最不可重建的信息）
    # moderate: 保留上述 + excluded_path，跳过 conversation_summary
    # full: 全部保留
    if aimd_policy == "conservative":
        # 只保留量化结论 + 非量化 decision + reasoning，丢弃 excluded 和 summaries
        excluded = []
        conv_summaries = []
    elif aimd_policy == "moderate":
        # 跳过 conversation_summary（最低信息密度）
        conv_summaries = []

    if not decisions and not excluded and not reasoning and not conv_summaries and not constraints and not causal_chains:
        # 仍然写缺页日志（即使本轮没有提取物，也可能有知识缺口）
        _write_page_fault_log(page_faults, session_id)
        sys.exit(0)

    # ── 迭代99：Hook 事务语义（OS 类比：ext4 journal 两阶段提交）──
    # Phase 1 (Prepare)：BEGIN IMMEDIATE — 独占写锁，防止并发写入污染
    # Phase 2 (Commit)：所有 chunk 写入成功后统一 COMMIT
    # Rollback：任意异常触发 ROLLBACK，保证原子性（全成功/全失败）
    # txn_log：记录事务状态供崩溃后诊断
    import uuid as _uuid
    _txn_id = _uuid.uuid4().hex[:16]

    # ── 批量写入（v5 迭代21：委托 store.py VFS 统一数据访问层）──
    conn = open_db()
    try:
        ensure_schema(conn)
        # 写入事务开始标记
        _txn_started_at = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """INSERT OR IGNORE INTO hook_txn_log
               (txn_id, hook, status, session_id, project, started_at)
               VALUES (?, 'extractor', 'pending', ?, ?, ?)""",
            (_txn_id, session_id, project, _txn_started_at)
        )
        conn.commit()

        # ── 迭代30：kswapd 水位线预淘汰（替代迭代25 的硬 OOM handler）──
        # OS 类比：kswapd 在 __alloc_pages_slowpath 前检查水位
        #   ZONE_OK  → 无需淘汰，直接写入
        #   ZONE_LOW → 预淘汰至 pages_high（后台回收，不阻塞分配）
        #   ZONE_MIN → 同步硬淘汰（direct reclaim，等价于旧 OOM handler）
        incoming_count = len(decisions) + len(excluded) + len(reasoning) + len(conv_summaries) + len(constraints) + len(causal_chains)
        ksw = kswapd_scan(conn, project, incoming_count)
        if ksw["evicted_count"] > 0:
            conn.commit()
            # dmesg：kswapd 淘汰事件
            dmesg_log(conn, DMESG_WARN, "extractor",
                      f"kswapd zone={ksw['zone']}: evicted={ksw['evicted_count']} stale={ksw['stale_evicted']} "
                      f"watermark={ksw['watermark_pct']}% quota={ksw['quota']} incoming={incoming_count}",
                      session_id=session_id, project=project)

        # ── 迭代40：cgroup v2 memory.high — Soft Quota Throttling ──
        # OS 类比：cgroup v2 memory.high (2015) — 超过软限制时 throttle 新写入
        # 检查水位是否在 memory_high 区间，如果是则降低新写入的 importance + 加 oom_adj
        throttle = cgroup_throttle_check(conn, project, incoming_count)
        throttle_active = throttle["throttled"]
        if throttle_active:
            dmesg_log(conn, DMESG_WARN, "extractor",
                      f"cgroup_throttle: zone={throttle['zone']} watermark={throttle['watermark_pct']}% "
                      f"factor={throttle['importance_factor']} oom_adj_delta={throttle['oom_adj_delta']}",
                      session_id=session_id, project=project)

        # 量化证据集合（用于 importance 提升判断）
        quant_set = set(quant_conclusions)
        # 迭代38：OOM Score — 量化证据自动高保护，临时摘要标记可优先淘汰
        written_chunk_ids = []  # 收集写入的 chunk id（用于 oom_adj 设置）
        quant_chunk_ids = []    # 量化证据 chunk ids
        throttled_chunk_ids = []  # 迭代40：被 throttle 的 chunk ids

        def _throttled_importance(base_imp: float) -> float:
            """迭代40：throttle 区间内 importance 乘以衰减因子。"""
            if throttle_active:
                return round(base_imp * throttle["importance_factor"], 3)
            return base_imp

        def _track_throttled_chunk(summary: str, chunk_type: str):
            """迭代40：记录被 throttle 的 chunk id（用于 oom_adj 设置）。"""
            if throttle_active and throttle["oom_adj_delta"] > 0:
                row = conn.execute(
                    "SELECT id FROM memory_chunks WHERE summary=? AND chunk_type=? ORDER BY created_at DESC LIMIT 1",
                    (summary, chunk_type)
                ).fetchone()
                if row:
                    throttled_chunk_ids.append(row[0])

        # ── 迭代326：quantitative_evidence content 富化 ──────────────────────────
        # 根因：quant_conclusions 平均 content 只有 103 chars（同 causal_chain 的
        # FTS5 token 不足问题）。修复：写入前先过滤出全部合格量化结论，
        # 然后为每个节点构建 "topic + 相邻量化结论" 的富 content。
        # 目标 content ≈ 200-350 chars，接近 decision（248 chars）的 FTS5 密度。
        # OS 类比：Linux huge page — 小页面（单条量化数据）合并成大页面（上下文丰富片段）
        _qualified_quant = [s for s in decisions if s in quant_set and _is_quality_chunk(s)]
        for _q_idx, summary in enumerate(decisions):
            if not _is_quality_chunk(summary):
                continue
            # iter105: 量化证据写成独立 chunk_type，不混入 decision
            if summary in quant_set:
                # 构建富 content：topic + 相邻量化结论（±1 邻居）
                _q_pos = _qualified_quant.index(summary) if summary in _qualified_quant else -1
                if _q_pos >= 0:
                    _q_parts = []
                    if _q_pos > 0:
                        _q_parts.append(_qualified_quant[_q_pos - 1])
                    _q_parts.append(summary)
                    if _q_pos < len(_qualified_quant) - 1:
                        _q_parts.append(_qualified_quant[_q_pos + 1])
                    _q_topic_tag = f"[quantitative_evidence|{topic}]" if topic else "[quantitative_evidence]"
                    _q_raw = f"{_q_topic_tag} {' | '.join(_q_parts)}"
                    # ── 迭代336：Document Expansion — 追加语义概念词 ──
                    # 信息论根因：quant summary 含数字/符号，查询用概念词 → encoding-retrieval mismatch
                    # 修复：写入时预计算概念词并追加到 content，让 FTS5 能按概念词索引量化证据
                    # OS 类比：Linux /proc/wchan — 将裸符号地址映射为人类可读的系统调用名
                    _q_concepts = _quant_semantic_concepts(summary)
                    _q_rich_content = (f"{_q_raw} [concepts: {_q_concepts}]" if _q_concepts
                                       else _q_raw)[:500]
                else:
                    # 无相邻节点时也追加概念词（保证 FTS5 可达）
                    _q_concepts = _quant_semantic_concepts(summary)
                    _q_rich_content = (f"[concepts: {_q_concepts}]" if _q_concepts else "")[:200]
                _write_chunk("quantitative_evidence", summary, project, session_id, topic, conn,
                             importance_override=0.90, _txn_managed=True,
                             content_override=_q_rich_content)
                row = conn.execute(
                    "SELECT id FROM memory_chunks WHERE summary=? AND chunk_type='quantitative_evidence' ORDER BY created_at DESC LIMIT 1",
                    (summary,)
                ).fetchone()
                if row:
                    quant_chunk_ids.append(row[0])
                else:
                    # iter106: SNR Promotion Filter — decision 需有决策动词/技术锚点/对比才写入
                    if not _is_quality_decision(summary):
                        dmesg_log(conn, DMESG_DEBUG, "extractor",
                                  f"snr_filter: decision dropped (no anchor) '{summary[:40]}'",
                                  session_id=session_id, project=project)
                        continue
                    imp = _throttled_importance(0.85)
                    _write_chunk("decision", summary, project, session_id, topic, conn,
                                 importance_override=imp, _txn_managed=True)
                    _track_throttled_chunk(summary, "decision")
        for summary in excluded:
            if _is_quality_chunk(summary):
                imp = _throttled_importance(0.70)
                _write_chunk("excluded_path", summary, project, session_id, topic, conn,
                             importance_override=imp, _txn_managed=True)
                _track_throttled_chunk(summary, "excluded_path")
        for summary in reasoning:
            if _is_quality_chunk(summary):
                # iter113: reasoning_chain 专用语义密度门控（必须含因果词或长度≥25）
                if not _is_quality_reasoning(summary):
                    dmesg_log(conn, DMESG_DEBUG, "extractor",
                              f"rsn_filter: reasoning_chain dropped (no causal signal) '{summary[:40]}'",
                              session_id=session_id, project=project)
                    continue
                imp = _throttled_importance(0.80)
                _write_chunk("reasoning_chain", summary, project, session_id, topic, conn,
                             importance_override=imp, _txn_managed=True)
                _track_throttled_chunk(summary, "reasoning_chain")
        # iter105: 因果链独立写入
        # ── 迭代324：causal_chain 写入前过滤，构建邻节点上下文 ─────────────────
        # OS 类比：Linux readahead + page clustering — 相邻 page 批量读入，
        # 避免每个 page 单独 I/O（等价于每个因果节点单独写入导致 content 碎片化）。
        # 根因：每个 causal_chain chunk 的 content = "[causal_chain] summary"（仅 ~89 字），
        # 而 decision 的 content 平均 248 字 — FTS5 token 不足，召回率极低（acc≈1.3 vs 43.8）。
        # 修复：对通过门控的因果节点，content 包含前一个+当前+后一个节点的聚合文本，
        # 保留因果推理的完整脉络，FTS5 可匹配到更丰富的语义 token。
        # summary 仍保留单节点（用于展示），content 作为检索索引（不展示）。
        _qualified_chains = []
        for _cc_summary in causal_chains:
            if not _is_quality_chunk(_cc_summary):
                continue
            has_arrow = '→' in _cc_summary
            has_causal_kw = bool(re.search(
                r'(?:因为|由于|导致了?|造成了?|引发了?|触发了?|引起了?|'
                r'根本原因|因此|所以|原因[：:]|根因[：:]|问题原因[：:]|'
                r'because|due to|caused by|resulted in|leads to)',
                _cc_summary
            ))
            if has_arrow:
                if len(_cc_summary.split('→')[0].strip()) < 3:
                    continue
            elif not has_causal_kw:
                continue
            _qualified_chains.append(_cc_summary)

        for _cc_idx, _cc_summary in enumerate(_qualified_chains):
            imp = _throttled_importance(0.82)
            # 构建聚合 content：前节点 + 当前 + 后节点（最多 ±1 邻居）
            # 聚合后 content ≈ 200-300 字，接近 decision 的 content 密度
            _ctx_parts = []
            if _cc_idx > 0:
                _ctx_parts.append(_qualified_chains[_cc_idx - 1])
            _ctx_parts.append(_cc_summary)
            if _cc_idx < len(_qualified_chains) - 1:
                _ctx_parts.append(_qualified_chains[_cc_idx + 1])
            _topic_tag = f"[causal_chain|{topic}]" if topic else "[causal_chain]"
            _rich_content = f"{_topic_tag} {' → '.join(_ctx_parts)}"[:400]
            _write_chunk("causal_chain", _cc_summary, project, session_id, topic, conn,
                         importance_override=imp, _txn_managed=True,
                         content_override=_rich_content)
            _track_throttled_chunk(_cc_summary, "causal_chain")
        # ── 迭代329：conversation_summary 写入前过滤，构建邻节点上下文 ──────────────
        # 根因：conversation_summary 平均 content=63 chars，FTS5 token 不足 → acc≈1.24。
        # 修复：同 iter324 causal_chain 策略，±1 邻居聚合，目标 content≈200 chars。
        # OS 类比：Linux readahead — 预读相邻 page，减少 random I/O。
        _qualified_summaries = [s for s in conv_summaries if _is_quality_chunk(s)]
        for _cs_idx, summary in enumerate(_qualified_summaries):
            imp = _throttled_importance(0.65)
            # 构建聚合 content：前节点 + 当前 + 后节点（±1 邻居）
            _cs_parts = []
            if _cs_idx > 0:
                _cs_parts.append(_qualified_summaries[_cs_idx - 1])
            _cs_parts.append(summary)
            if _cs_idx < len(_qualified_summaries) - 1:
                _cs_parts.append(_qualified_summaries[_cs_idx + 1])
            _cs_topic_tag = f"[conversation_summary|{topic}]" if topic else "[conversation_summary]"
            _cs_rich_content = f"{_cs_topic_tag} {' | '.join(_cs_parts)}"[:400]
            _write_chunk("conversation_summary", summary, project, session_id, topic, conn,
                         importance_override=imp, _txn_managed=True,
                         content_override=_cs_rich_content)
            _track_throttled_chunk(summary, "conversation_summary")

        # 迭代98：设计约束写入（importance=0.95，oom_adj=-800 高保护）
        constraint_chunk_ids = []
        for summary in constraints:
            if _is_quality_chunk(summary):
                _write_chunk("design_constraint", summary, project, session_id, topic, conn,
                             importance_override=0.95, _txn_managed=True)  # 约束知识高价值
                row = conn.execute(
                    "SELECT id FROM memory_chunks WHERE summary=? AND chunk_type='design_constraint' ORDER BY created_at DESC LIMIT 1",
                    (summary,)
                ).fetchone()
                if row:
                    constraint_chunk_ids.append(row[0])

        # 迭代38：为量化证据 chunk 设置 OOM_ADJ_PROTECTED（高保护）
        # 迭代104：soft pin 到当前 project（保护 stale reclaim + DAMON DEAD，不挡 kswapd ZONE_MIN）
        # OS 类比：systemd OOMPolicy=continue 标记关键服务不被 OOM Killer 杀死
        for cid in quant_chunk_ids:
            set_oom_adj(conn, cid, OOM_ADJ_PROTECTED)
            pin_chunk(conn, cid, project, pin_type="soft")  # 迭代104: 量化证据 → soft pin

        # 迭代98：为设计约束 chunk 设置 OOM_ADJ_PROTECTED（架构级知识最高保护）
        # 迭代104：同时 hard pin 到当前 project（VMA mlock 语义，跨 project 不互干扰）
        for cid in constraint_chunk_ids:
            set_oom_adj(conn, cid, OOM_ADJ_PROTECTED)
            pin_chunk(conn, cid, project, pin_type="hard")  # 迭代104: design_constraint → hard pin

        # 迭代40：为 throttled chunk 设置 oom_adj（加速未来回收）
        # OS 类比：cgroup v2 memory.high 下的分配会被计入 memory.stat.high 计数，
        # 这些页面在后续 kswapd 扫描中有更高的回收概率
        if throttled_chunk_ids and throttle["oom_adj_delta"] > 0:
            from store import batch_set_oom_adj
            batch_set_oom_adj(conn, throttled_chunk_ids, throttle["oom_adj_delta"])

        # 增强2：从 transcript Bash tool_result 提取量化结论（tool_insight 类型）
        tool_insight_count = 0
        if transcript_path and aimd_policy != "conservative":
            try:
                tool_insight_count = _extract_from_tool_outputs(
                    transcript_path, session_id, project, conn)
                conn.commit()
            except Exception:
                pass  # 失败不影响主流程

        # 工具使用模式学习 — perf_event 采样工具调用链
        # OS 类比：perf_event_open() 采样 CPU 调用栈，会话结束时 flush ring buffer
        tool_pattern_count = 0
        if transcript_path:
            try:
                tool_pattern_count = _extract_tool_patterns(
                    transcript_path, conn, project, session_id,
                    context_text=text)
            except Exception:
                pass  # 失败不影响主流程

        # 迭代303：Entity detection — NER 触发写入 entity_stub
        # OS 类比：Linux inotify — 文件系统事件触发，对每个新出现的"实体"建立 inode stub
        # 策略：轻量正则 NER（无 LLM 调用，< 2ms），识别三类实体：
        #   1. 项目/框架名（首字母大写英文词、含 - 的技术名词）
        #   2. GitHub 用户/仓库（xxx/yyy 格式）
        #   3. 中文专有词（首次在对话中出现，被「」/《》/""包围的词）
        # 只写入首次出现（already_exists 去重），避免刷写
        entity_stub_count = 0
        try:
            entity_stub_count = _detect_and_write_entities(
                text, project, session_id, conn)
        except Exception:
            pass  # entity detection 失败不影响主流程

        # 迭代304：Entity relations — 从文本提取关系三元组写入 entity_edges
        # OS 类比：modprobe modules.dep 解析 — 识别模块间依赖关系，建立加载顺序图
        # 策略：纯正则，不调 LLM，< 5ms
        edge_count = 0
        try:
            edge_count = _extract_entity_relations(
                text, project, session_id, conn)
        except Exception:
            pass  # edge extraction 失败不影响主流程

        # 迭代29 dmesg：提取汇总
        # 迭代50：AIMD 信息加入日志
        _dur = (_time.time() - _t_start) * 1000
        aimd_tag = ""
        if aimd_info:
            aimd_tag = f" aimd={aimd_policy}(cwnd={aimd_info['cwnd']:.2f} hr={aimd_info['hit_rate']:.2f} {aimd_info['direction']})"
        _long_tag = f" long_msg={_text_long} transcript_extra={len(_transcript_extra)}" if _transcript_extra or _text_long else ""
        dmesg_log(conn, DMESG_INFO, "extractor",
                  f"decisions={len(decisions)} excluded={len(excluded)} reasoning={len(reasoning)} causal={len(causal_chains)} summaries={len(conv_summaries)} constraints={len(constraints)} tool_insights={tool_insight_count} tool_patterns={tool_pattern_count} entities={entity_stub_count} edges={edge_count} faults={len(page_faults)} {_dur:.1f}ms{aimd_tag}{_long_tag}",
                  session_id=session_id, project=project)
        # ── 迭代99：原子提交 — 更新 txn_log 状态后统一 COMMIT ──
        # OS 类比：ext4 journal commit — 日志记录 committed 后才真正写入 data block
        _chunk_count = (len(decisions) + len(excluded) + len(reasoning)
                        + len(conv_summaries) + len(constraints)
                        + tool_insight_count + tool_pattern_count + entity_stub_count)
        conn.execute(
            """UPDATE hook_txn_log
               SET status='committed', chunk_count=?, committed_at=?
               WHERE txn_id=?""",
            (_chunk_count, datetime.now(timezone.utc).isoformat(), _txn_id)
        )
        conn.commit()  # 单一原子 COMMIT：txn_log + 所有 chunks

        # ── 迭代103：跨Agent知识广播（OS 类比：inotify IN_MODIFY 事件）──
        # commit 成功后广播本轮写入统计，其他 agent 的 loader 可在 SessionStart 消费
        if _chunk_count > 0:
            try:
                from net.agent_notify import broadcast_knowledge_update
                broadcast_knowledge_update(project, session_id, {
                    "decisions": len(decisions),
                    "constraints": len(constraints),
                    "chunks": _chunk_count,
                })
            except Exception:
                pass  # IPC 失败不影响主流程

    except Exception as _txn_err:
        # ── 迭代99：Rollback — 记录错误到 txn_log（在新连接中，因原连接可能已损坏）──
        try:
            conn.rollback()
            conn.execute(
                """UPDATE hook_txn_log SET status='failed', error=? WHERE txn_id=?""",
                (str(_txn_err)[:500], _txn_id)
            )
            conn.commit()
        except Exception:
            pass
        dmesg_log(conn, DMESG_WARN, "extractor",
                  f"txn_rollback: txn_id={_txn_id} err={type(_txn_err).__name__}:{str(_txn_err)[:80]}",
                  session_id=session_id, project=project)
    finally:
        conn.close()

    _write_page_fault_log(page_faults, session_id)

    # ── 迭代32：madvise — 写入检索 hint（MADV_WILLNEED）──
    # OS 类比：应用程序在完成一轮处理后，通过 madvise 告知内核下一轮可能访问的区域
    # extractor 分析本轮对话内容，提取主题实体作为 hint
    _write_madvise_hints(text, decisions, excluded, reasoning, conv_summaries + constraints,
                         project, session_id, topic)

    # ── 迭代49：CRIU checkpoint — 会话结束时保存精确工作集快照 ──
    # OS 类比：CRIU dump — 在进程终止前序列化完整状态
    # 收集本次会话的 retrieval 命中 chunk IDs + madvise hints
    try:
        ckpt_conn = open_db()
        ensure_schema(ckpt_conn)
        hit_ids = checkpoint_collect_hits(ckpt_conn, project, session_id)
        if hit_ids:
            # 读取当前 madvise hints 作为 checkpoint 的一部分
            from store import madvise_read
            current_hints = madvise_read(project)
            hint_keywords = [h.get("keyword", "") for h in current_hints] if current_hints else []

            # 从本轮提取的实体作为 query_topics
            topic_entities = []
            if topic:
                topic_entities.append(topic)
            all_summaries = decisions[:3] + reasoning[:2]
            for s in all_summaries:
                for m in re.finditer(r'`([^`]{2,30})`', s):
                    topic_entities.append(m.group(1))
            topic_entities = list(dict.fromkeys(topic_entities))[:5]

            ckpt_result = checkpoint_dump(ckpt_conn, project, session_id,
                                          hit_ids, hint_keywords, topic_entities)
            if ckpt_result.get("checkpoint_id"):
                dmesg_log(ckpt_conn, DMESG_INFO, "extractor",
                          f"criu_dump: ckpt={ckpt_result['checkpoint_id']} ids={ckpt_result['saved_ids']} cleaned={ckpt_result['cleaned']}",
                          session_id=session_id, project=project)
            ckpt_conn.commit()
        ckpt_conn.close()
    except Exception:
        pass  # checkpoint 失败不影响主流程

    # ── 迭代94: 跨项目知识晋升 + 迭代95: 目标进度追踪 ──
    try:
        _pr_conn = open_db()
        ensure_schema(_pr_conn)
        promoted = _promote_to_global(_pr_conn, project, session_id)
        # 目标进度：有新决策写入 → progress 微增
        # iter_multiagent P2：session 级幂等防双计 — 同一 session 只增加一次
        if len(decisions) > 0:
            now_iso = __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()
            try:
                _pr_conn.execute("ALTER TABLE goals ADD COLUMN last_progress_session TEXT DEFAULT ''")
            except Exception:
                pass
            _pr_conn.execute(
                """UPDATE goals SET progress = MIN(1.0, progress + 0.05),
                   updated_at = ?,
                   last_progress_session = ?
                   WHERE project = ? AND status = 'active'
                     AND (last_progress_session IS NULL OR last_progress_session != ?)""",
                [now_iso, session_id, project, session_id]
            )
        _pr_conn.commit()
        _pr_conn.close()
    except Exception:
        pass

    # ── LRU 语义淘汰（超阈值时自动触发）──
    try:
        import importlib.util
        _evict_path = Path(__file__).parent.parent / "tools" / "memory_eviction.py"
        if _evict_path.exists():
            spec = importlib.util.spec_from_file_location("memory_eviction", _evict_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.run(dry_run=False)
    except Exception:
        pass  # 淘汰失败不影响主流程

    # ── 冷备份同步（新 chunk 自动推到 mm）──
    try:
        _cold_path = Path(__file__).parent.parent / "tools" / "cold_store.py"
        if _cold_path.exists():
            spec = importlib.util.spec_from_file_location("cold_store", _cold_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.cmd_sync(dry_run=False)
    except Exception:
        pass  # 冷备份失败不影响主流程（mm 可能离线）

    # ── 会话结束 GC：清除 prompt_context chunks（临时短暂信号，无跨会话召回价值）──
    try:
        _gc_conn = open_db()
        ensure_schema(_gc_conn)
        _gc_deleted = _gc_conn.execute(
            "DELETE FROM memory_chunks WHERE chunk_type = 'prompt_context'"
        ).rowcount
        # iter114: tool_insight GC — bash 输出量化结论是 point-in-time 数据，
        # 跨会话召回率极低（100% 从未访问），与 prompt_context 同级清除。
        # 保留逻辑：access_count >= 1 的 tool_insight 说明曾被实际使用，保留。
        # OS 类比：tmpfs — 进程退出时自动释放 VMA 映射的临时文件系统内容。
        _gc_tool = _gc_conn.execute(
            "DELETE FROM memory_chunks WHERE chunk_type = 'tool_insight' AND COALESCE(access_count,0) = 0"
        ).rowcount
        _gc_deleted += _gc_tool
        # iter328: entity_stub GC — NER 提取的实体存根 100% zero-access（噪声率高）
        # 策略：只保留 access_count >= 1 的（曾被实际用于检索），其余清除。
        # OS 类比：dentries 的 d_count=0 时被 dentry_cache 的 LRU 回收。
        _gc_entity = _gc_conn.execute(
            "DELETE FROM memory_chunks WHERE chunk_type = 'entity_stub' AND COALESCE(access_count,0) = 0"
        ).rowcount
        _gc_deleted += _gc_entity
        _gc_conn.commit()
        if _gc_deleted > 0:
            dmesg_log(_gc_conn, DMESG_INFO, "extractor",
                      f"session_gc: deleted {_gc_deleted} temp chunks "
                      f"(prompt_context + {_gc_tool} tool_insight + {_gc_entity} entity_stub)",
                      session_id=session_id, project=project)
            _gc_conn.commit()
        _gc_conn.close()
    except Exception:
        pass  # GC 失败不影响主流程

    # ── 迭代110 P2: CRIU Session Intent Checkpoint ──────────────────────────
    # OS 类比：CRIU (Checkpoint/Restore in Userspace, 2012) — 在进程终止前
    #   序列化完整进程状态（寄存器、内存映射、文件描述符），下次 restore 时
    #   像什么都没发生一样继续。
    #
    # AIOS 类比：在 session 结束前，从 last_assistant_message 提取
    #   "incomplete intent" — Claude 在思考/执行过程中遇到的悬而未决的事项。
    #   下次 SessionStart 时注入，让新 session 像从断点继续一样工作。
    #
    # 提取目标（三类未完成信号）：
    #   I1 next_actions:  "接下来需要..." / "下一步..." / "还需要..."
    #   I2 open_questions: "需要验证..." / "待确认..." / "不确定..."
    #   I3 partial_work:  "正在..." / "目前..." / "已完成...但还需要..."
    try:
        _intent = _extract_session_intent(text)
        if _intent:
            # iter259: 从 DB shadow_traces 表读取（并发安全，替代单文件）
            _intent_chunk_ids: list = []
            _agent_id = session_id[:16] if session_id else ""
            try:
                _st_conn = open_db()
                ensure_schema(_st_conn)
                _st_row = _st_conn.execute(
                    "SELECT top_k_ids FROM shadow_traces WHERE session_id=? AND project=?",
                    (session_id, project)
                ).fetchone()
                if _st_row:
                    _intent_chunk_ids = json.loads(_st_row[0] or "[]")
                _st_conn.close()
            except Exception:
                # 兼容旧文件（逐步迁移期）
                try:
                    _shadow_file = MEMORY_OS_DIR / ".shadow_trace.json"
                    if _shadow_file.exists():
                        _st = json.loads(_shadow_file.read_text(encoding="utf-8"))
                        if _st.get("project", project) == project:
                            _intent_chunk_ids = _st.get("top_k_ids", [])
                except Exception:
                    pass

            # iter259: 写入 DB session_intents 表（替代单文件 session_intent.json）
            # OS 类比：per-process /proc/PID/status — 每个 session 独立一行
            _intent_conn = open_db()
            ensure_schema(_intent_conn)
            _intent_conn.execute(
                """INSERT OR REPLACE INTO session_intents
                   (session_id, project, agent_id, saved_at, intent_json, pinned_chunk_ids)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    session_id, project, _agent_id,
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(_intent, ensure_ascii=False),
                    json.dumps(_intent_chunk_ids, ensure_ascii=False),
                )
            )
            _intent_conn.commit()

            # 同时保留旧文件以向后兼容（只写最新 session，不再是唯一数据源）
            try:
                _intent_file = MEMORY_OS_DIR / "session_intent.json"
                _intent_file.write_text(
                    json.dumps({
                        "session_id": session_id,
                        "project": project,
                        "saved_at": datetime.now(timezone.utc).isoformat(),
                        "intent": _intent,
                        "pinned_chunk_ids": _intent_chunk_ids,
                    }, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )
            except Exception:
                pass

            # iter259: soft-pin 关联 chunk，防止被 kswapd 在 24h 有效期内淘汰
            if _intent_chunk_ids:
                try:
                    from store_vfs import pin_chunk as _pin_chunk
                    _pinned = 0
                    for _cid in _intent_chunk_ids:
                        if _pin_chunk(_intent_conn, _cid, project, pin_type="soft"):
                            _pinned += 1
                    if _pinned:
                        _intent_conn.commit()
                        dmesg_log(_intent_conn, DMESG_DEBUG, "extractor",
                                  f"intent_soft_pin: pinned {_pinned} chunks for 24h (CRIU intent restore)",
                                  session_id=session_id, project=project)
                        _intent_conn.commit()
                except Exception:
                    pass  # soft-pin 失败不影响 intent 保存
            _intent_conn.close()
    except Exception:
        pass  # Intent 保存失败不影响主流程

    # ── 迭代311-B：Active Suppression — 注入未用则下调 importance ─────────────
    # OS 类比：vm.swappiness 主动换出冷页面
    # shadow_trace 记录上次 retriever 注入的 chunk IDs，本次回复未用的下调
    # iter259：优先从 shadow_traces DB 表读取（并发安全，替代单文件）
    try:
        _injected_ids = []
        _sup_loaded = False
        # 优先从 DB shadow_traces 表读取（per-session 隔离）
        try:
            _sup_db = open_db()
            ensure_schema(_sup_db)
            _sup_row = _sup_db.execute(
                "SELECT top_k_ids FROM shadow_traces WHERE session_id=? AND project=?",
                (session_id, project)
            ).fetchone()
            _sup_db.close()
            if _sup_row:
                _injected_ids = json.loads(_sup_row[0] or "[]")
                _sup_loaded = True
        except Exception:
            pass
        # 兼容旧文件（DB 读取失败时 fallback）
        if not _sup_loaded:
            _shadow_file = MEMORY_OS_DIR / ".shadow_trace.json"
            if _shadow_file.exists():
                _shadow = json.loads(_shadow_file.read_text(encoding="utf-8"))
                _shadow_proj = _shadow.get("project", project)
                if _shadow_proj == project:
                    _injected_ids = _shadow.get("top_k_ids", [])
        if _injected_ids:
            from store_vfs import suppress_unused as _suppress_unused
            _sup_conn = open_db()
            ensure_schema(_sup_conn)
            _sup_n = _suppress_unused(
                _sup_conn, _injected_ids, assistant_response=text, project=project
            )
            if _sup_n:
                dmesg_log(_sup_conn, DMESG_DEBUG, "extractor",
                          f"suppress_unused: {_sup_n} chunks penalized (not referenced in response)",
                          session_id=session_id, project=project)
            _sup_conn.commit()
            _sup_conn.close()
    except Exception:
        pass  # suppress_unused 失败不影响主流程

    # ── 迭代311-C：Sleep Consolidation — session 结束自动维护记忆 ──────────────
    # OS 类比：pdflush writeback + KSM — 进程退出时合并相似页、稳定活跃页、淘汰陈旧页
    try:
        from store_vfs import sleep_consolidate as _sleep_consolidate
        _slp_conn = open_db()
        ensure_schema(_slp_conn)
        _slp_result = _sleep_consolidate(_slp_conn, project=project, session_id=session_id)
        _slp_any = any(
            v > 0 for k, v in _slp_result.items()
            if isinstance(v, (int, float)) and k != "new_semantic_ids"
        )
        if _slp_any or _slp_result.get("new_semantic_ids"):
            dmesg_log(_slp_conn, DMESG_INFO, "extractor",
                      f"sleep_consolidate: merged={_slp_result['merged']} "
                      f"boosted={_slp_result['boosted']} decayed={_slp_result['decayed']} "
                      f"ep_promoted={_slp_result.get('episodic_promoted',0)} "
                      f"ep_decayed={_slp_result.get('episodic_decayed',0)}",
                      session_id=session_id, project=project)
        _slp_conn.commit()
        _slp_conn.close()
    except Exception:
        pass  # sleep_consolidate 失败不影响主流程

    sys.exit(0)


if __name__ == "__main__":
    main()


def _extract_session_intent(text: str) -> dict:
    """
    迭代110 P2: CRIU Session Intent Extraction — 提取会话末尾的未完成意图。

    OS 类比：CRIU dump_task() — 序列化进程当前执行状态（PC、stack、open files）。
    这里序列化的是 Claude 的"执行状态"：
      - 下一步要做什么（next_actions）
      - 还有哪些问题未解决（open_questions）
      - 正在进行中的工作（partial_work）

    返回 {"next_actions": [...], "open_questions": [...], "partial_work": [...]}
    任何列表为空则该字段不存在。
    """
    import re as _re

    NEXT_ACTION_PATTERNS = [
        r'(?:接下来(?:需要|要|应该)|下一步(?:是|需要|要)|还需要|然后(?:需要|要))[：:]?\s*(.{5,80})',
        r'(?:next[,:\s]+(?:step|action|task|we need)[s]?)[：:\s]+(.{5,80})',
        r'(?:TODO|待做|待完成|后续)[：:]\s*(.{5,80})',
        r'(?:^|\n)\d+\.\s+(?:然后|接着|再|最后)\s*(.{5,60})',
    ]
    OPEN_QUESTION_PATTERNS = [
        r'(?:需要验证|待验证|待确认|需要确认|不确定|还不清楚)[：:]?\s*(.{5,80})',
        r'(?:需要查看|需要读|需要了解|需要检查)[：:]?\s*(.{5,60})',
        r'(?:question|need to verify|not sure|unclear)[:\s]+(.{5,80})',
        r'(?:假设|假定)\s*(.{5,60})\s*(?:待验证|需要确认)',
    ]
    PARTIAL_WORK_PATTERNS = [
        r'(?:正在|目前正在|当前正在)[：:]?\s*(.{5,60})',
        r'(?:已完成[^，。\n]*?，?但(?:还|仍)(?:需要|要|未))\s*(.{5,80})',
        r'(?:partially|in progress|working on)[:\s]+(.{5,80})',
    ]

    result = {}

    # 只扫描文本的最后 2000 字符（意图通常在消息末尾）
    sample = text[-2000:] if len(text) > 2000 else text

    def _extract_pattern(patterns, sample_text):
        items = []
        seen = set()
        for pat in patterns:
            for m in _re.finditer(pat, sample_text, _re.MULTILINE | _re.IGNORECASE):
                captured = m.group(1).strip()
                captured = _re.split(r'[\n。！？]', captured)[0].strip()
                captured = _re.sub(r'\*{1,3}|`{1,3}', '', captured).strip()
                key = _re.sub(r'\s+', '', captured.lower())
                if len(captured) >= 5 and key not in seen:
                    seen.add(key)
                    items.append(captured[:80])
        return items[:3]  # 每类最多 3 条

    next_actions = _extract_pattern(NEXT_ACTION_PATTERNS, sample)
    open_questions = _extract_pattern(OPEN_QUESTION_PATTERNS, sample)
    partial_work = _extract_pattern(PARTIAL_WORK_PATTERNS, sample)

    if next_actions:
        result["next_actions"] = next_actions
    if open_questions:
        result["open_questions"] = open_questions
    if partial_work:
        result["partial_work"] = partial_work

    return result


def _promote_to_global(conn, project: str, session_id: str) -> int:
    """
    迭代94: Cross-Project Knowledge Promotion — 跨项目知识晋升

    OS 类比：Linux 内核模块（.ko）的跨进程共享 — 内核代码段被所有进程共享，
    而不是每个进程各自复制一份。高价值知识同理：不应被项目边界割裂。

    将本项目中高重要性（importance >= 0.85）且被多次访问（access_count >= 3）
    的知识晋升到全局层（project="global"），使所有项目都能检索到。

    条件：
    - importance >= 0.85（顶层知识）
    - access_count >= 3（经过实战验证）
    - chunk_type in (decision, reasoning_chain)（方法论类知识，而非 prompt_context）
    - 全局层尚无相同 summary
    """
    from datetime import datetime, timezone
    import json as _json, uuid as _uuid

    if project == "global":
        return 0  # 防止循环

    try:
        candidates = conn.execute(
            """SELECT id, chunk_type, summary, content, importance, tags
               FROM memory_chunks
               WHERE project NOT IN ('global', 'test')
                 AND chunk_type IN ('procedure')
                 AND importance >= 0.92
                 AND access_count >= 5
               ORDER BY importance DESC, access_count DESC
               LIMIT 2"""
        ).fetchall()

        promoted = 0
        for row in candidates:
            src_id, ctype, summary, content, imp, tags = row
            # 检查全局层是否已有
            exists = conn.execute(
                "SELECT id FROM memory_chunks WHERE project='global' AND summary=?",
                [summary]
            ).fetchone()
            if exists:
                continue
            now = datetime.now(timezone.utc).isoformat()
            global_id = f"global-{_uuid.uuid4().hex[:12]}"
            conn.execute("""
                INSERT INTO memory_chunks
                (id, created_at, updated_at, project, source_session,
                 chunk_type, content, summary, tags, importance,
                 retrievability, last_accessed, access_count, lru_gen, oom_adj)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, [global_id, now, now, "global", f"promoted:{project}",
                  ctype, content, summary,
                  tags if isinstance(tags, str) else _json.dumps(["global", project]),
                  imp, 0.5, now, 0, 0, -400])
            promoted += 1

        if promoted > 0:
            dmesg_log(conn, DMESG_INFO, "extractor",
                      f"global_promote: {promoted} chunks from {project}",
                      session_id=session_id, project=project)
        return promoted
    except Exception:
        return 0
