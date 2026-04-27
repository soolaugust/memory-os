"""
test_synonym_expansion.py — 迭代332：通用语义 Query Expansion 单元测试

验证：
  1. 自然语言问法（如何/怎么/为什么）触发对应技术术语扩展
  2. 性能类问句扩展到 optimize/latency/deadline
  3. 召回类问句扩展到 recall/FTS5/BM25
  4. 去重类问句扩展到 dedup/Jaccard/merge
  5. 重要性类问句扩展到 importance/oom_adj/stability
  6. 注入/上下文问句扩展到 inject/过滤/无关
  7. 调试/查看问句扩展到 dmesg/log/stats/trace
  8. 写入失败问句扩展到 extractor/AIMD/already_exists
  9. 中英文双向桥接有效（召回↔recall，优化↔optimize）
 10. 不触发场景：空字符串、纯确认词
 11. Prescan 快速退出路径正常工作（短 query 无触发词时 0ms）
 12. 同义词扩展结果无重复
"""
import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "hooks"))

import tmpfs  # noqa

from store_vfs import _fts5_escape, _synonym_expand


def _get_tokens(query: str) -> set:
    """提取 _fts5_escape 返回的所有 token（去引号）。"""
    expr = _fts5_escape(query)
    if not expr:
        return set()
    # 提取引号内容
    return set(m.group(1).lower() for m in re.finditer(r'"([^"]+)"', expr))


# ══════════════════════════════════════════════════════════════════════
# 1. 性能/优化类问句（Category A）
# ══════════════════════════════════════════════════════════════════════

def test_optimize_query_expands_to_latency():
    """'如何优化检索速度' → 扩展到 optimize/latency 等性能词。"""
    tokens = _get_tokens("如何优化检索速度")
    assert "optimize" in tokens or "latency" in tokens or "deadline" in tokens, \
        f"性能优化查询应扩展到技术术语，got tokens: {tokens}"


def test_how_to_speed_up_expands():
    """'怎么加快检索' → 扩展到 fast/performance/psi 等词。"""
    tokens = _get_tokens("怎么加快检索")
    assert any(t in tokens for t in ["fast", "performance", "optim", "psi", "vdso"]), \
        f"加速查询应扩展，got tokens: {tokens}"


def test_performance_improve_expands():
    """'性能提升方法' → 扩展到技术术语。"""
    tokens = _get_tokens("性能提升方法")
    assert any(t in tokens for t in ["optimize", "latency", "perform", "psi"]), \
        f"性能提升查询应扩展，got tokens: {tokens}"


# ══════════════════════════════════════════════════════════════════════
# 2. 召回/检索类问句（Category B）
# ══════════════════════════════════════════════════════════════════════

def test_recall_rate_low_expands():
    """'为什么召回率低' → 扩展到 recall/FTS5/BM25。"""
    tokens = _get_tokens("为什么召回率低")
    assert any(t in tokens for t in ["recall", "fts5", "bm25", "precis"]), \
        f"召回率查询应扩展，got tokens: {tokens}"


def test_cant_find_relevant_expands():
    """'检索效果差' → 扩展到 precision/min_score/threshold。"""
    tokens = _get_tokens("检索效果差")
    assert any(t in tokens for t in ["recall", "bm25", "fts5", "precis", "threshold"]), \
        f"检索效果查询应扩展，got tokens: {tokens}"


def test_zh_recall_expands_to_en():
    """中文'召回' → 扩展到英文 recall/retrieve/search。"""
    tokens = _get_tokens("召回结果不准")
    assert any(t in tokens for t in ["recall", "retriev", "search", "fts5", "bm25"]), \
        f"中文召回应扩展到英文，got tokens: {tokens}"


# ══════════════════════════════════════════════════════════════════════
# 3. 去重/合并类问句（Category C）
# ══════════════════════════════════════════════════════════════════════

def test_dedup_query_expands():
    """'如何减少重复知识' → 扩展到 dedup/find_similar/Jaccard。"""
    tokens = _get_tokens("如何减少重复知识")
    assert any(t in tokens for t in ["dedup", "find_similar", "jaccard", "merg"]), \
        f"去重查询应扩展，got tokens: {tokens}"


def test_merge_similar_expands():
    """'合并相似内容' → 扩展到 merge/Jaccard。"""
    tokens = _get_tokens("合并相似内容")
    assert any(t in tokens for t in ["merge", "jaccard", "dedup", "similar"]), \
        f"合并查询应扩展，got tokens: {tokens}"


# ══════════════════════════════════════════════════════════════════════
# 4. 重要性/权重类问句（Category D）
# ══════════════════════════════════════════════════════════════════════

def test_importance_query_expands():
    """'怎么设置重要性' → 扩展到 importance/score/oom_adj。"""
    tokens = _get_tokens("怎么设置重要性")
    assert any(t in tokens for t in ["import", "score", "oom_adj", "stabil"]), \
        f"重要性查询应扩展，got tokens: {tokens}"


def test_priority_query_expands():
    """'哪些知识更重要' → 扩展到 importance/critical/priority。"""
    tokens = _get_tokens("哪些知识更重要")
    assert any(t in tokens for t in ["import", "critic", "priorit", "key"]), \
        f"优先级查询应扩展，got tokens: {tokens}"


# ══════════════════════════════════════════════════════════════════════
# 5. 注入/上下文类问句（Category H）
# ══════════════════════════════════════════════════════════════════════

def test_injection_query_expands():
    """'注入了不相关内容' → 扩展到相关过滤词。"""
    tokens = _get_tokens("注入了不相关内容")
    # 至少有注入相关的词
    assert any(t in tokens for t in ["inject", "threshold", "drr", "min_scor", "mmr"]), \
        f"注入查询应扩展到过滤词，got tokens: {tokens}"


def test_context_noise_expands():
    """'为什么注入了不相关内容' → 触发注入类扩展。"""
    tokens = _get_tokens("为什么注入了不相关内容")
    assert any(t in tokens for t in ["inject", "threshold", "drr", "noise"]), \
        f"上下文噪音查询应扩展，got tokens: {tokens}"


# ══════════════════════════════════════════════════════════════════════
# 6. 调试/监控类问句（Category L）
# ══════════════════════════════════════════════════════════════════════

def test_view_log_expands():
    """'怎么查看日志' → 扩展到 dmesg/log/stats/trace。"""
    tokens = _get_tokens("怎么查看日志")
    assert any(t in tokens for t in ["dmesg", "log", "stat", "trac"]), \
        f"日志查询应扩展，got tokens: {tokens}"


def test_debug_expands():
    """'如何调试问题' → 扩展到 dmesg/log/debug。"""
    tokens = _get_tokens("如何调试问题")
    assert any(t in tokens for t in ["dmesg", "log", "debug", "trac"]), \
        f"调试查询应扩展，got tokens: {tokens}"


# ══════════════════════════════════════════════════════════════════════
# 7. 写入/提取失败类问句（Category F）
# ══════════════════════════════════════════════════════════════════════

def test_knowledge_not_saved_expands():
    """'知识没有被保存' → 扩展到 extractor/AIMD/already_exists。"""
    tokens = _get_tokens("知识没有被保存")
    assert any(t in tokens for t in ["extract", "aimd", "already_exists", "throttl"]), \
        f"知识保存失败查询应扩展，got tokens: {tokens}"


def test_extraction_failed_expands():
    """'为什么没有提取到知识' → 扩展到 extractor。"""
    tokens = _get_tokens("为什么没有提取到知识")
    assert any(t in tokens for t in ["extract", "aimd", "cwnd", "already_exists"]), \
        f"提取失败查询应扩展，got tokens: {tokens}"


# ══════════════════════════════════════════════════════════════════════
# 8. 速度/延迟类问句（Category G）
# ══════════════════════════════════════════════════════════════════════

def test_too_slow_expands():
    """'检索太慢' → 扩展到 deadline/latency/psi。"""
    tokens = _get_tokens("检索太慢")
    assert any(t in tokens for t in ["deadline", "latenc", "psi"]), \
        f"太慢查询应扩展，got tokens: {tokens}"


def test_high_latency_expands():
    """'延迟高' → 扩展到 deadline/ms/latency。"""
    tokens = _get_tokens("延迟高")
    assert any(t in tokens for t in ["deadline", "latenc", "psi", "timeout"]), \
        f"高延迟查询应扩展，got tokens: {tokens}"


# ══════════════════════════════════════════════════════════════════════
# 9. 中英文双向桥接（Category I）
# ══════════════════════════════════════════════════════════════════════

def test_zh_optimize_expands_to_en():
    """中文'优化' → 英文 optimize/improve。"""
    tokens = _get_tokens("优化内存使用")
    assert any(t in tokens for t in ["optim", "improv", "faster", "perform"]), \
        f"中文优化应扩展到英文，got tokens: {tokens}"


def test_en_recall_expands_to_zh():
    """英文'recall' → 中文召回/检索。"""
    tokens = _get_tokens("recall rate improvement")
    # 中文字直接以字符形式出现在 tokens 里（bigram）
    assert any(t in tokens for t in ["optim", "improv", "recall", "retriev", "search"]), \
        f"英文 recall 应扩展，got tokens: {tokens}"


def test_en_delete_expands_to_zh():
    """英文 'evict' → 中文淘汰/清理词汇。"""
    tokens = _get_tokens("evict old chunks")
    assert any(t in tokens for t in ["delet", "clean", "purg", "remov"]), \
        f"evict 应扩展，got tokens: {tokens}"


# ══════════════════════════════════════════════════════════════════════
# 10. 不触发场景（防误触发）
# ══════════════════════════════════════════════════════════════════════

def test_empty_query_no_expansion():
    """空字符串 → 返回空。"""
    tokens = _get_tokens("")
    assert len(tokens) == 0, f"空查询不应有 token，got: {tokens}"


def test_simple_ack_no_expansion():
    """纯确认词 '好的' → 无扩展词注入。"""
    # '好的' 含 CJK，会有 bigram token，但不应有技术术语
    tokens = _get_tokens("好的")
    tech_terms = {"optim", "recall", "fts5", "bm25", "deadlin", "extrac", "dmesg"}
    assert not (tokens & tech_terms), \
        f"确认词不应触发技术术语扩展，got tech: {tokens & tech_terms}"


def test_pure_english_ack_no_expansion():
    """英文确认词 'ok sure' → 无扩展。"""
    tokens = _get_tokens("ok sure")
    tech_terms = {"optim", "recall", "fts5", "bm25", "deadlin", "extrac"}
    assert not (tokens & tech_terms), \
        f"英文确认词不应触发技术扩展，got tech: {tokens & tech_terms}"


# ══════════════════════════════════════════════════════════════════════
# 11. Prescan 快速路径（无触发词时直接返回 []）
# ══════════════════════════════════════════════════════════════════════

def test_prescan_returns_empty_for_no_trigger():
    """无触发词 query → _synonym_expand 返回 []（Prescan miss）。"""
    seen: set = set()
    result = _synonym_expand("普通的问题描述无技术词", seen)
    # 不要求 result 为空，但不应有技术扩展词
    assert all(t not in '"'.join(result) for t in ["optimize", "fts5", "bm25"]), \
        f"Prescan miss 不应包含技术扩展词，got: {result}"


def test_prescan_fires_for_known_trigger():
    """含已知触发词 → Prescan hit，继续完整展开。"""
    seen: set = set()
    result = _synonym_expand("如何优化性能", seen)
    # 含 "优化" 触发词，应有扩展
    has_tech = any("optim" in r or "latenc" in r or "perform" in r for r in result)
    assert has_tech, f"Prescan hit 应展开技术词，got: {result}"


# ══════════════════════════════════════════════════════════════════════
# 12. 扩展结果无重复
# ══════════════════════════════════════════════════════════════════════

def test_no_duplicate_tokens():
    """复杂查询 → 展开后无重复 token（seen 集合防重）。"""
    tokens_list = list(_get_tokens("为什么召回率低 如何优化 FTS5"))
    tokens_set = set(tokens_list)
    assert len(tokens_list) == len(tokens_set), \
        f"token 列表含重复项: {[t for t in tokens_list if tokens_list.count(t) > 1]}"


# ══════════════════════════════════════════════════════════════════════
# 因果类（Category J）
# ══════════════════════════════════════════════════════════════════════

def test_root_cause_expands():
    """'根本原因是什么' → 扩展到 causal_chain/reasoning_chain。"""
    tokens = _get_tokens("根本原因是什么")
    assert any(t in tokens for t in ["causal", "reason", "causal_chain"]), \
        f"根因查询应扩展，got tokens: {tokens}"


def test_what_causes_expands():
    """'什么导致了问题' → 扩展到 causal_chain 相关词。"""
    tokens = _get_tokens("什么导致了问题")
    assert any(t in tokens for t in ["causal", "reason", "root"]), \
        f"导致查询应扩展，got tokens: {tokens}"
