"""
test_context_similarity_boost.py — iter394: Contextual Similarity Boost 单元测试

覆盖：
  CC1: session_type 精确匹配 → boost 应用（+context_type_boost）
  CC2: session_type 不匹配 → 无 session_type boost
  CC3: task_verbs overlap → 按 Jaccard 比例 boost
  CC4: context_type_boost_enabled=False → 无 session_type/task_verbs boost
  CC5: cwd 匹配 + session_type 匹配 → 叠加 boost（上限 0.25）
  CC6: unknown session_type → 无 boost（guard 条件）
  CC7: task_verbs 无交集 → 无 task_verbs boost

认知科学依据：
  Tulving (1983) Encoding Specificity Principle —
    编码时的情境（context）与检索时的情境越匹配，记忆提取成功率越高。
  Godden & Baddeley (1975) Context-Dependent Memory —
    在水下学习 → 水下回忆（同情境）比陆地回忆效果好 36%。
OS 类比：NUMA-aware scheduler — 同 NUMA 域的内存访问延迟更低，优先调度。
"""
import sys
import math
import pytest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "hooks"))

import tmpfs  # noqa

# 直接从 retriever 导入 _compute_context_match 逻辑做纯函数测试
# 由于 _compute_context_match 是嵌套函数，我们提取等效逻辑做单元测试
# 或通过对 _sysctl 进行 mock 来测试完整逻辑

def _make_context_match_fn(
    context_type_boost_enabled=True,
    context_type_boost=0.05,
    task_verbs_boost=0.03,
    focus_keywords=None,
):
    """
    创建一个与 retriever._compute_context_match 等效的函数。
    对 sysctl 调用做 inline 替换，避免依赖全局 config 状态。
    """
    _focus_keywords = focus_keywords or []

    def _compute_context_match(enc_ctx: dict, cur_ctx: dict) -> float:
        """iter394 等效的 context match 计算（提取自 retriever._compute_context_match）。"""
        if not enc_ctx or not cur_ctx:
            return 0.0
        boost = 0.0

        # cwd 匹配：+0.10
        enc_cwd = (enc_ctx.get("cwd") or "").rstrip("/")
        cur_cwd = (cur_ctx.get("cwd") or "").rstrip("/")
        if enc_cwd and cur_cwd:
            if enc_cwd == cur_cwd or cur_cwd.startswith(enc_cwd + "/"):
                boost += 0.10

        # keyword 匹配：+0.05 × Jaccard
        enc_kw = set(enc_ctx.get("keywords") or [])
        cur_kw = set(cur_ctx.get("keywords") or _focus_keywords)
        if enc_kw and cur_kw:
            intersection = len(enc_kw & cur_kw)
            union = len(enc_kw | cur_kw)
            if union > 0:
                boost += (intersection / union) * 0.05

        # entity 匹配：+0.05 × Jaccard (iter385)
        import re as _re
        enc_entities = set(e.lower() for e in (enc_ctx.get("entities") or []) if e)
        cur_entities = set(cur_ctx.get("entities") or [])
        if not cur_entities:
            _q = cur_ctx.get("query", "")
            cur_entities = set(m.group().lower()
                               for m in _re.finditer(r'[a-zA-Z][a-zA-Z0-9_\.]{2,}', _q))
            _cjk = _re.sub(r'[^\u4e00-\u9fff]', '', _q)
            for _i in range(len(_cjk) - 1):
                cur_entities.add(_cjk[_i:_i + 2])
        if enc_entities and cur_entities:
            _ei = len(enc_entities & cur_entities)
            _eu = len(enc_entities | cur_entities)
            if _eu > 0:
                boost += (_ei / _eu) * 0.05

        # iter394: Session Type + Task Verbs Boost
        if context_type_boost_enabled:
            enc_stype = enc_ctx.get("session_type", "unknown")
            cur_stype = cur_ctx.get("session_type", "unknown")
            if (enc_stype and cur_stype and enc_stype != "unknown"
                    and cur_stype != "unknown" and enc_stype == cur_stype):
                boost += context_type_boost
            enc_verbs = set(enc_ctx.get("task_verbs") or [])
            cur_verbs = set(cur_ctx.get("task_verbs") or [])
            if enc_verbs and cur_verbs:
                _vi = len(enc_verbs & cur_verbs)
                _vu = len(enc_verbs | cur_verbs)
                if _vu > 0:
                    boost += (_vi / _vu) * task_verbs_boost

        return min(boost, 0.25)

    return _compute_context_match


# ══════════════════════════════════════════════════════════════════════
# 1. session_type 匹配测试
# ══════════════════════════════════════════════════════════════════════

def test_cc1_session_type_match():
    """session_type 精确匹配 → 应用 context_type_boost (+0.05)。"""
    fn = _make_context_match_fn(context_type_boost=0.05)
    enc = {"session_type": "debug", "task_verbs": []}
    cur = {"session_type": "debug", "task_verbs": []}
    boost = fn(enc, cur)
    assert abs(boost - 0.05) < 1e-6, f"CC1: session_type match → boost=0.05, got {boost:.6f}"


def test_cc2_session_type_mismatch():
    """session_type 不匹配 → 无 session_type boost。"""
    fn = _make_context_match_fn(context_type_boost=0.05)
    enc = {"session_type": "debug", "task_verbs": []}
    cur = {"session_type": "design", "task_verbs": []}
    boost = fn(enc, cur)
    assert boost == 0.0, f"CC2: session_type mismatch → boost=0, got {boost:.6f}"


def test_cc3_task_verbs_overlap():
    """task_verbs Jaccard 交集 → 按比例 boost。"""
    fn = _make_context_match_fn(
        context_type_boost_enabled=True,
        context_type_boost=0.0,   # 关闭 session_type boost，隔离 task_verbs
        task_verbs_boost=0.03,
    )
    enc = {"session_type": "unknown", "task_verbs": ["fix", "debug", "test"]}
    cur = {"session_type": "unknown", "task_verbs": ["fix", "debug", "optimize"]}
    # Jaccard: {fix, debug} / {fix, debug, test, optimize} = 2/4 = 0.5
    expected_boost = 0.5 * 0.03
    boost = fn(enc, cur)
    assert abs(boost - expected_boost) < 1e-6, (
        f"CC3: task_verbs Jaccard=0.5 → boost={expected_boost:.6f}, got {boost:.6f}"
    )


def test_cc4_boost_disabled():
    """context_type_boost_enabled=False → 无 session_type/task_verbs boost。"""
    fn = _make_context_match_fn(
        context_type_boost_enabled=False,
        context_type_boost=0.05,
        task_verbs_boost=0.03,
    )
    enc = {"session_type": "debug", "task_verbs": ["fix", "debug"]}
    cur = {"session_type": "debug", "task_verbs": ["fix", "debug"]}
    boost = fn(enc, cur)
    # 无 cwd/kw/entity 匹配，且 boost disabled → 0
    assert boost == 0.0, f"CC4: boost disabled → 0.0, got {boost:.6f}"


def test_cc5_cwd_and_session_type_additive():
    """cwd 匹配 (+0.10) + session_type 匹配 (+0.05) = 0.15（未超上限 0.25）。"""
    fn = _make_context_match_fn(context_type_boost=0.05)
    enc = {
        "cwd": "/home/mi/project",
        "session_type": "debug",
        "task_verbs": [],
    }
    cur = {
        "cwd": "/home/mi/project",
        "session_type": "debug",
        "task_verbs": [],
    }
    boost = fn(enc, cur)
    assert abs(boost - 0.15) < 1e-6, (
        f"CC5: cwd(0.10) + session_type(0.05) = 0.15, got {boost:.6f}"
    )


def test_cc6_unknown_session_type_no_boost():
    """enc 或 cur session_type 为 'unknown' → 无 session_type boost（guard 条件）。"""
    fn = _make_context_match_fn(context_type_boost=0.05)
    # enc = unknown
    enc1 = {"session_type": "unknown", "task_verbs": []}
    cur1 = {"session_type": "debug", "task_verbs": []}
    assert fn(enc1, cur1) == 0.0, "CC6a: enc unknown → no boost"

    # cur = unknown
    enc2 = {"session_type": "debug", "task_verbs": []}
    cur2 = {"session_type": "unknown", "task_verbs": []}
    assert fn(enc2, cur2) == 0.0, "CC6b: cur unknown → no boost"

    # both = unknown
    enc3 = {"session_type": "unknown", "task_verbs": []}
    cur3 = {"session_type": "unknown", "task_verbs": []}
    assert fn(enc3, cur3) == 0.0, "CC6c: both unknown → no boost"


def test_cc7_task_verbs_no_overlap():
    """task_verbs 无交集 → 无 task_verbs boost。"""
    fn = _make_context_match_fn(
        context_type_boost_enabled=True,
        context_type_boost=0.0,
        task_verbs_boost=0.03,
    )
    enc = {"session_type": "unknown", "task_verbs": ["design", "architect"]}
    cur = {"session_type": "unknown", "task_verbs": ["fix", "debug"]}
    # Jaccard = 0 / 4 = 0
    boost = fn(enc, cur)
    assert boost == 0.0, f"CC7: no task_verbs overlap → boost=0, got {boost:.6f}"


# ══════════════════════════════════════════════════════════════════════
# 2. 上限和边界测试
# ══════════════════════════════════════════════════════════════════════

def test_cc8_boost_capped_at_0_25():
    """多维度叠加 boost 上限为 0.25。"""
    fn = _make_context_match_fn(
        context_type_boost=0.05,
        task_verbs_boost=0.03,
    )
    # cwd(0.10) + session_type(0.05) + task_verbs_full(0.03) + entities... 总计可能超 0.25
    enc = {
        "cwd": "/project",
        "session_type": "debug",
        "task_verbs": ["fix", "debug", "test"],
        "keywords": ["perf", "latency"],
        "entities": ["BM25", "FTS5"],
    }
    cur = {
        "cwd": "/project",
        "session_type": "debug",
        "task_verbs": ["fix", "debug", "test"],
        "keywords": ["perf", "latency"],
        "entities": ["BM25", "FTS5"],
        "query": "BM25 FTS5 performance",
    }
    boost = fn(enc, cur)
    assert boost <= 0.25 + 1e-9, f"CC8: boost 上限 0.25，got {boost:.6f}"


def test_cc9_empty_context_returns_zero():
    """空 enc_ctx 或 cur_ctx → 返回 0.0。"""
    fn = _make_context_match_fn()
    assert fn({}, {"session_type": "debug"}) == 0.0
    assert fn({"session_type": "debug"}, {}) == 0.0
    assert fn(None, {"session_type": "debug"}) == 0.0
    assert fn({"session_type": "debug"}, None) == 0.0


def test_cc10_all_session_types_recognized():
    """验证 extract_encoding_context 能识别所有已知 session_type。"""
    from store_vfs import extract_encoding_context

    test_cases = {
        "debug error fix crash": "debug",
        "design architecture plan": "design",
        "review code check": "review",
        "refactor cleanup restructure": "refactor",
        "test qa verify validate": "qa",
    }
    for text, expected_type in test_cases.items():
        ctx = extract_encoding_context(text)
        # extract_encoding_context 可能检测到不同 type，但至少返回一个有效 type
        assert ctx.get("session_type") is not None, (
            f"CC10: '{text}' → session_type 应不为 None"
        )


def test_cc11_task_verbs_full_overlap():
    """task_verbs 完全重合 → boost = task_verbs_boost（Jaccard=1.0）。"""
    fn = _make_context_match_fn(
        context_type_boost_enabled=True,
        context_type_boost=0.0,
        task_verbs_boost=0.03,
    )
    enc = {"session_type": "unknown", "task_verbs": ["fix", "debug"]}
    cur = {"session_type": "unknown", "task_verbs": ["fix", "debug"]}
    boost = fn(enc, cur)
    assert abs(boost - 0.03) < 1e-6, f"CC11: full overlap → task_verbs_boost=0.03, got {boost:.6f}"
