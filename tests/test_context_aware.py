"""
test_context_aware.py — 迭代315：情境感知注入单元测试

验证：
  1. extract_encoding_context 正确提取session_type/entities/task_verbs
  2. context_match_score 正确计算三维匹配分
  3. 相同情境 chunk 得分高于不同情境 chunk（核心验证）
  4. 空/None 输入安全返回0
"""
import pytest
from store_vfs import extract_encoding_context
from scorer import context_match_score, retrieval_score


# ── extract_encoding_context 测试 ────────────────────────────────


def test_extract_debug_session_type():
    """含「报错」的文本应识别为 debug。"""
    ctx = extract_encoding_context("系统启动时出现报错，traceback 如下：")
    assert ctx["session_type"] == "debug"


def test_extract_design_session_type():
    """含「设计」的文本识别为 design。"""
    ctx = extract_encoding_context("我们需要设计新的 API schema 接口。")
    assert ctx["session_type"] == "design"


def test_extract_qa_session_type():
    """含「测试」的文本识别为 qa。"""
    ctx = extract_encoding_context("运行 pytest 验证所有 assert 通过。")
    assert ctx["session_type"] == "qa"


def test_extract_refactor_session_type():
    """含「重构」的文本识别为 refactor。"""
    ctx = extract_encoding_context("对模块进行重构，cleanup 冗余代码。")
    assert ctx["session_type"] == "refactor"


def test_extract_unknown_session_type():
    """无关键词的文本识别为 unknown。"""
    ctx = extract_encoding_context("今天天气不错，适合写代码。")
    assert ctx["session_type"] == "unknown"


def test_extract_entities():
    """含反引号和驼峰词的文本正确提取实体。"""
    text = "优化 `BM25` 检索器，修改 `scorer.py` 中的 RetrievalScore 函数。"
    ctx = extract_encoding_context(text)
    entities = ctx["entities"]
    # 反引号内容应该被提取
    assert "BM25" in entities
    assert "scorer.py" in entities


def test_extract_task_verbs():
    """含中文动作词的文本正确提取 task_verbs。"""
    text = "修复 BM25 评分错误，同时优化检索性能，添加新的测试用例。"
    ctx = extract_encoding_context(text)
    verbs = ctx["task_verbs"]
    assert len(verbs) > 0
    # 修复属于 debug类，优化属于 perf类，添加属于 impl类
    assert any(v in ("修复", "优化", "添加") for v in verbs)


def test_extract_returns_correct_keys():
    """返回 dict 包含所有必需字段。"""
    ctx = extract_encoding_context("test text")
    assert "session_type" in ctx
    assert "entities" in ctx
    assert "task_verbs" in ctx


def test_extract_entities_max_8():
    """entities 最多返回 8 个。"""
    # 大量反引号词
    text = " ".join(f"`word{i}`" for i in range(20))
    ctx = extract_encoding_context(text)
    assert len(ctx["entities"]) <= 8


def test_extract_task_verbs_max_5():
    """task_verbs 最多返回 5 个。"""
    text = "修复错误，设计方案，实现功能，测试验证，优化性能，重构代码，删除旧逻辑，更新文档"
    ctx = extract_encoding_context(text)
    assert len(ctx["task_verbs"]) <= 5


# ── context_match_score 测试 ──────────────────────────────────────


def test_context_match_same_type():
    """相同 session_type（非 unknown）应得 0.08 分。"""
    q = {"session_type": "debug", "entities": [], "task_verbs": []}
    c = {"session_type": "debug", "entities": [], "task_verbs": []}
    score = context_match_score(q, c)
    assert abs(score - 0.08) < 1e-9


def test_context_match_different_type():
    """不同 session_type 不得 session_type 加分。"""
    q = {"session_type": "debug", "entities": [], "task_verbs": []}
    c = {"session_type": "design", "entities": [], "task_verbs": []}
    score = context_match_score(q, c)
    assert score == 0.0


def test_context_match_unknown_type_no_bonus():
    """均为 unknown 时 session_type 不得分。"""
    q = {"session_type": "unknown", "entities": [], "task_verbs": []}
    c = {"session_type": "unknown", "entities": [], "task_verbs": []}
    score = context_match_score(q, c)
    assert score == 0.0


def test_context_match_entity_overlap():
    """entities 完全重叠时得 0.12 分。"""
    q = {"session_type": "unknown", "entities": ["BM25", "scorer"], "task_verbs": []}
    c = {"session_type": "unknown", "entities": ["BM25", "scorer"], "task_verbs": []}
    score = context_match_score(q, c)
    assert abs(score - 0.12) < 1e-9


def test_context_match_entity_partial_overlap():
    """entities 部分重叠时按 Jaccard 计算。"""
    # q={A,B}, c={A,C} → intersection=1, union=3, jaccard=1/3
    q = {"session_type": "unknown", "entities": ["A", "B"], "task_verbs": []}
    c = {"session_type": "unknown", "entities": ["A", "C"], "task_verbs": []}
    score = context_match_score(q, c)
    expected = (1 / 3) * 0.12
    assert abs(score - expected) < 1e-9


def test_context_match_verb_overlap():
    """task_verbs 完全重叠时得 0.06 分。"""
    q = {"session_type": "unknown", "entities": [], "task_verbs": ["修复", "优化"]}
    c = {"session_type": "unknown", "entities": [], "task_verbs": ["修复", "优化"]}
    score = context_match_score(q, c)
    assert abs(score - 0.06) < 1e-9


def test_context_match_combined():
    """type + entities + verbs 全中时得到最高分（上限 0.20）。"""
    q = {"session_type": "debug", "entities": ["BM25", "scorer"], "task_verbs": ["修复", "优化"]}
    c = {"session_type": "debug", "entities": ["BM25", "scorer"], "task_verbs": ["修复", "优化"]}
    score = context_match_score(q, c)
    # 0.08 + 0.12 + 0.06 = 0.26，但 cap=0.20
    assert abs(score - 0.20) < 1e-9


def test_context_match_cap_at_020():
    """分数不超过 0.20 上限。"""
    q = {"session_type": "debug", "entities": ["A", "B", "C"], "task_verbs": ["修复", "优化"]}
    c = {"session_type": "debug", "entities": ["A", "B", "C"], "task_verbs": ["修复", "优化"]}
    score = context_match_score(q, c)
    assert score <= 0.20


def test_context_match_empty():
    """空/None 输入安全返回 0.0。"""
    assert context_match_score(None, None) == 0.0
    assert context_match_score({}, {}) == 0.0
    assert context_match_score(None, {"session_type": "debug"}) == 0.0
    assert context_match_score({"session_type": "debug"}, None) == 0.0


def test_context_match_empty_dict():
    """空 dict（所有字段缺失）返回 0.0。"""
    assert context_match_score({}, {}) == 0.0


# ── 端到端：相同情境 chunk 得分高于不同情境 chunk ─────────────────


def test_same_context_beats_different():
    """
    核心验证：同情境 chunk 的 retrieval_score > 不同情境 chunk。

    场景：当前在 debug session 查询 BM25 相关内容，
    chunk_A: 同为 debug session，含相同实体
    chunk_B: design session，不同实体
    两者基础分完全相同，仅 encoding_context 不同。
    """
    query_ctx = {
        "session_type": "debug",
        "entities": ["BM25", "scorer"],
        "task_verbs": ["修复"],
    }
    same_ctx = {
        "session_type": "debug",
        "entities": ["BM25", "scorer"],
        "task_verbs": ["修复"],
    }
    diff_ctx = {
        "session_type": "design",
        "entities": ["frontend", "ui"],
        "task_verbs": ["设计"],
    }

    base_kwargs = dict(
        relevance=0.5,
        importance=0.7,
        last_accessed="2025-01-01T00:00:00+00:00",
        access_count=1,
        created_at="2025-01-01T00:00:00+00:00",
    )

    score_same = retrieval_score(**base_kwargs,
                                 encoding_context=same_ctx,
                                 query_context=query_ctx)
    score_diff = retrieval_score(**base_kwargs,
                                 encoding_context=diff_ctx,
                                 query_context=query_ctx)

    assert score_same > score_diff, (
        f"同情境 chunk 评分应高于不同情境 chunk: {score_same:.4f} vs {score_diff:.4f}"
    )


def test_no_context_equals_baseline():
    """不传 encoding_context 时与 encoding_context=None 结果一致（向后兼容）。
    用 approx 容忍两次 _refresh_now() 之间的微小时间差导致的浮点差异。
    """
    base_kwargs = dict(
        relevance=0.5,
        importance=0.7,
        last_accessed="2025-01-01T00:00:00+00:00",
        access_count=1,
        created_at="2025-01-01T00:00:00+00:00",
    )
    score_no_ctx = retrieval_score(**base_kwargs)
    score_none_ctx = retrieval_score(**base_kwargs,
                                     encoding_context=None,
                                     query_context=None)
    assert score_no_ctx == pytest.approx(score_none_ctx, rel=1e-6)
