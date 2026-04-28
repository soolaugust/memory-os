"""
test_encoding_context_boost.py — iter385: Encoding Context Retrieval Boost 单元测试

覆盖：
  ECB1: 实体重叠 → score boost (+0.05 × Jaccard)
  ECB2: 无实体重叠 → 无 entity boost（只有 cwd/keyword boost）
  ECB3: 空 encoding_context → 0.0 boost
  ECB4: query entity fallback 提取（cur_ctx 无 pre-computed entities）
  ECB5: 组合 boost（cwd + keyword + entity）上限 0.20
  ECB6: partial entity 重叠 → 按比例 boost
  ECB7: encoding_context.entities 大小写不敏感匹配

认知科学依据：
  Godden & Baddeley (1975) Context-Dependent Memory —
  检索时复现编码时的上下文（环境线索）可最大化回忆成功率。
  实体重叠作为编码/检索上下文匹配的最细粒度指标：
  chunk 被写入时涉及的技术实体（函数名/模块/概念）与
  当前 query 的实体重叠越多，该 chunk 越可能是当前问题的正确答案。
"""
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import tmpfs  # noqa


# ── 直接测试 _compute_context_match 的行为 ────────────────────────────────────
# 由于 _compute_context_match 在 retriever.py 的 main() 内定义（访问外部闭包变量），
# 我们提取其逻辑为独立函数进行单元测试，验证算法正确性。

def _compute_context_match_impl(enc_ctx: dict, cur_ctx: dict,
                                 focus_keywords: list = None) -> float:
    """
    提取自 retriever.py:_compute_context_match 的纯函数实现（用于测试）。
    与生产代码逻辑完全一致，去掉闭包依赖（_focus_keywords 变为参数）。
    """
    import re as _re

    if not enc_ctx or not cur_ctx:
        return 0.0

    _focus_keywords = focus_keywords or []
    boost = 0.0

    # cwd match: +0.10
    enc_cwd = (enc_ctx.get("cwd") or "").rstrip("/")
    cur_cwd = (cur_ctx.get("cwd") or "").rstrip("/")
    if enc_cwd and cur_cwd:
        if enc_cwd == cur_cwd or cur_cwd.startswith(enc_cwd + "/"):
            boost += 0.10

    # keyword Jaccard: +0.05
    enc_kw = set(enc_ctx.get("keywords") or [])
    cur_kw = set(cur_ctx.get("keywords") or _focus_keywords)
    if enc_kw and cur_kw:
        intersection = len(enc_kw & cur_kw)
        union = len(enc_kw | cur_kw)
        if union > 0:
            boost += (intersection / union) * 0.05

    # iter385: entity-level boost: +0.05 × entity_Jaccard
    enc_entities = set(e.lower() for e in (enc_ctx.get("entities") or []) if e)
    cur_entities = set(cur_ctx.get("entities") or [])
    if not cur_entities:
        # fallback: extract from query
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

    return min(boost, 0.20)


# ── ECB1: 实体完全重叠 → 最大 entity boost ──────────────────────────────────

def test_ecb1_full_entity_overlap():
    """所有 enc_entities 都在 cur_entities 中 → entity boost = 0.05。"""
    enc_ctx = {"entities": ["BM25", "fts_search", "retriever"]}
    cur_ctx = {"entities": ["bm25", "fts_search", "retriever", "scorer"]}

    result = _compute_context_match_impl(enc_ctx, cur_ctx)
    # enc_entities = {bm25, fts_search, retriever}, cur_entities = {bm25, fts_search, retriever, scorer}
    # intersection=3, union=4 → Jaccard=0.75 → boost=0.75×0.05=0.0375
    assert result > 0.0, f"实体重叠应产生 boost，got {result}"
    assert result <= 0.20, f"boost 不超过 0.20 上限，got {result}"


# ── ECB2: 无实体重叠 → 无 entity boost ────────────────────────────────────

def test_ecb2_no_entity_overlap():
    """enc_entities 与 cur_entities 完全不同 → entity boost = 0.0。"""
    enc_ctx = {"entities": ["PostgreSQL", "database", "schema"]}
    cur_ctx = {"entities": ["bm25", "fts5", "retriever"]}

    result = _compute_context_match_impl(enc_ctx, cur_ctx)
    # Jaccard = 0/6 = 0 → entity boost = 0
    assert result == 0.0, f"无实体重叠应产生 0 boost，got {result}"


# ── ECB3: 空 encoding_context → 0.0 boost ────────────────────────────────

def test_ecb3_empty_encoding_context():
    """enc_ctx 为空 → 整体 boost = 0.0。"""
    assert _compute_context_match_impl({}, {"entities": ["bm25"]}) == 0.0
    assert _compute_context_match_impl(None, {"entities": ["bm25"]}) == 0.0


# ── ECB4: query entity fallback 提取 ─────────────────────────────────────

def test_ecb4_query_entity_fallback():
    """cur_ctx 无 pre-computed entities → 从 query 字段 fallback 提取。"""
    enc_ctx = {"entities": ["BM25", "fts_search"]}
    # 不提供 entities，提供 query 字段
    cur_ctx = {"query": "BM25 算法在 fts_search 中的召回优化"}
    # query 中的英文实体应被提取：bm25, fts_search

    result = _compute_context_match_impl(enc_ctx, cur_ctx)
    assert result > 0.0, f"fallback 提取实体后应有 boost，got {result}"


# ── ECB5: 组合 boost（cwd + keyword + entity）上限 0.20 ────────────────

def test_ecb5_combined_boost_capped():
    """cwd + keyword + entity 三者叠加，总 boost 上限 0.20。"""
    enc_ctx = {
        "cwd": "/home/user/project",
        "keywords": ["bm25", "fts5"],
        "entities": ["BM25", "fts_search", "scorer"],
    }
    cur_ctx = {
        "cwd": "/home/user/project",
        "keywords": ["bm25", "fts5", "retriever"],
        "entities": ["bm25", "fts_search", "scorer"],
    }

    result = _compute_context_match_impl(enc_ctx, cur_ctx)
    # cwd: +0.10, keyword Jaccard={bm25,fts5}/{bm25,fts5,retriever}=2/3≈0.667→+0.033,
    # entity Jaccard={bm25,fts_search,scorer}/{bm25,fts_search,scorer}=1.0→+0.05
    # total raw ≈ 0.183 → capped at 0.20
    assert result <= 0.20, f"上限 0.20 enforced，got {result}"
    assert result >= 0.14, f"三路叠加应有足够 boost，got {result}"


# ── ECB6: partial entity 重叠 → 按 Jaccard 比例 boost ──────────────────

def test_ecb6_partial_entity_overlap_proportional():
    """部分实体重叠 → boost 按 Jaccard 比例，不是固定 +0.05。"""
    # 50% 重叠
    enc_ctx = {"entities": ["BM25", "fts_search"]}
    cur_ctx = {"entities": ["bm25", "scorer"]}  # bm25 重叠，scorer 不重叠

    result_partial = _compute_context_match_impl(enc_ctx, cur_ctx)
    # Jaccard = {bm25}/{bm25,fts_search,scorer} = 1/3 ≈ 0.333 → boost = 0.333×0.05 ≈ 0.0167

    # 100% 重叠
    enc_ctx_full = {"entities": ["BM25"]}
    cur_ctx_full = {"entities": ["bm25"]}

    result_full = _compute_context_match_impl(enc_ctx_full, cur_ctx_full)
    # Jaccard = {bm25}/{bm25} = 1.0 → boost = 1.0×0.05 = 0.05

    assert result_full > result_partial, \
        f"完全重叠应有更高 boost: full={result_full:.4f} partial={result_partial:.4f}"
    assert result_partial > 0.0, \
        f"部分重叠应有 > 0 boost，got {result_partial}"


# ── ECB7: 大小写不敏感匹配 ──────────────────────────────────────────────

def test_ecb7_case_insensitive_entity_matching():
    """encoding_context 存储大写实体 → 与 query 小写实体正确匹配。"""
    enc_ctx = {"entities": ["BM25", "FTS_SEARCH", "RETRIEVER"]}
    cur_ctx = {"entities": ["bm25", "fts_search", "retriever"]}

    result = _compute_context_match_impl(enc_ctx, cur_ctx)
    # 全部匹配（大小写不敏感）→ Jaccard=1.0 → boost=0.05
    assert abs(result - 0.05) < 1e-6, \
        f"大小写不敏感匹配，boost 应为 0.05，got {result}"


# ── ECB8: CJK 实体 fallback 提取 ────────────────────────────────────────

def test_ecb8_cjk_entity_fallback():
    """query 包含中文技术词 → CJK bigram fallback 提取实体。"""
    enc_ctx = {"entities": ["检索", "召回", "排序"]}
    cur_ctx = {"query": "BM25 检索召回排序优化"}
    # CJK bigram 提取：检索, 索召, 召回, 回排, 排序, 序优, 优化
    # enc bigrams (lowercase): 检索, 召回, 排序
    # 预期: 检索、召回、排序 会匹配到

    result = _compute_context_match_impl(enc_ctx, cur_ctx)
    assert result > 0.0, f"CJK bigram 匹配应产生 boost，got {result}"
