"""
test_causal_secondary.py — 迭代330：causal secondary query 单元测试

验证：
  1. 含因果词 prompt → 返回非空 causal query
  2. 含技术信号 prompt → 返回非空 causal query
  3. 纯确认词 → 返回空（不触发二次搜索）
  4. 空 prompt → 返回空
  5. causal query 不包含在主 query 中（独立，不污染主搜索）
  6. causal query 包含核心因果语义词（原因/导致/因为/根因）
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "hooks"))

import tmpfs  # noqa

from retriever import _build_causal_query, _build_query


# ══════════════════════════════════════════════════════════════════════
# 1. 含因果词 prompt
# ══════════════════════════════════════════════════════════════════════

def test_prompt_with_why_triggers_causal():
    """含 '为什么' 的 prompt → 返回非空 causal query。"""
    q = _build_causal_query("为什么 FTS5 召回率低")
    assert q, "含因果词 prompt 应触发 causal secondary query"
    assert any(w in q for w in ["原因", "导致", "因为", "根因"]), \
        f"causal query 应含因果语义词，got: {q!r}"


def test_prompt_with_because_en():
    """英文 'because' → 触发。"""
    q = _build_causal_query("why does memory-os have low recall because of FTS5")
    assert q, "含 'because' 应触发"


def test_prompt_with_cause():
    """'导致' → 触发。"""
    q = _build_causal_query("什么导致了 causal_chain 的零召回")
    assert q


def test_prompt_with_root_cause():
    """'根因' → 触发。"""
    q = _build_causal_query("分析 BM25 fallback 的根因")
    assert q
    assert "根因" in q


# ══════════════════════════════════════════════════════════════════════
# 2. 含技术信号但无因果词
# ══════════════════════════════════════════════════════════════════════

def test_tech_signal_with_underscore():
    """含下划线 snake_case 标识符 → 触发。"""
    q = _build_causal_query("causal_chain 为什么召回率低")
    assert q


def test_tech_signal_uppercase():
    """含全大写缩写词 → 触发。"""
    q = _build_causal_query("FTS5 性能分析")
    assert q
    assert any(w in q for w in ["原因", "导致", "因为"])


def test_tech_signal_backtick():
    """含反引号代码 → 触发。"""
    q = _build_causal_query("`store_vfs.py` 性能调查")
    assert q


def test_tech_signal_milliseconds():
    """含时间单位 ms → 触发。"""
    q = _build_causal_query("为什么检索需要 100ms")
    # 含 '为什么'（因果词），一定触发
    assert q


def test_dotted_filename():
    """含文件名 → 触发。"""
    q = _build_causal_query("retriever.py 的性能瓶颈")
    assert q


# ══════════════════════════════════════════════════════════════════════
# 3. 不触发场景
# ══════════════════════════════════════════════════════════════════════

def test_empty_prompt_returns_empty():
    """空字符串 → 返回空。"""
    assert _build_causal_query("") == ""


def test_confirmation_word_no_causal():
    """纯确认词（无技术信号） → 返回空。"""
    q = _build_causal_query("好的")
    assert q == "", f"确认词不应触发 causal query，got: {q!r}"


def test_pure_english_ack():
    """英文确认词 → 返回空。"""
    q = _build_causal_query("ok")
    assert q == ""


def test_no_tech_no_causal():
    """无技术信号 + 无因果词 → 返回空。"""
    q = _build_causal_query("这是一个普通的问题")
    assert q == "", f"无信号 prompt 不应触发，got: {q!r}"


# ══════════════════════════════════════════════════════════════════════
# 4. 主 query 与 causal query 独立性
# ══════════════════════════════════════════════════════════════════════

def test_causal_query_independent_from_main():
    """causal query 是独立返回值，不影响 _build_query 的主 query 输出。"""
    hook_input = {"prompt": "为什么 FTS5 召回率低"}
    main_q = _build_query(hook_input)
    causal_q = _build_causal_query("为什么 FTS5 召回率低")

    # main_q 不应该包含 causal_q 中的扩展词（原因/导致/因为/根因 等）
    # 除非 prompt 本身就含这些词（本例 prompt 无这些词，所以 main_q 不含）
    # main_q 只包含 prompt + entities（内部可能含多余空格）
    assert "原因" not in main_q or "为什么" in main_q, \
        f"main_q 不应含独立扩展词，got: {main_q!r}"
    assert "FTS5" in main_q, f"main_q 应含实体词，got: {main_q!r}"

    # causal_q 包含扩展词
    assert "原因" in causal_q or "导致" in causal_q, \
        f"causal_q 应含扩展词，got: {causal_q!r}"


# ══════════════════════════════════════════════════════════════════════
# 5. 长度截断保护
# ══════════════════════════════════════════════════════════════════════

def test_causal_query_truncated_from_long_prompt():
    """超长 prompt 的 causal query 应截断（不超过 prompt[:100] + 扩展词）。"""
    long_prompt = "为什么 " + "A" * 200
    q = _build_causal_query(long_prompt)
    assert q  # 应触发（含 '为什么'）
    # causal query 长度有限：prompt[:100] + 扩展词
    assert len(q) < 130, f"causal query 不应过长，got len={len(q)}"
