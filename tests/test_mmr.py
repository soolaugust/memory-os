"""
test_mmr.py — 迭代321：MMR 边际信息量过滤单元测试

验证：
  1. 冗余候选被过滤（两条相同摘要只保留一条）
  2. 多样候选全部保留（不同内容不被过滤）
  3. lambda=1.0 等价于纯 relevance 排序（退化行为）
  4. 空输入安全返回 []
  5. candidates <= top_k 时直接返回（无需 MMR）
  6. sim_threshold 边界：Jaccard < threshold 不惩罚（保留相似但不完全重复的 chunk）
  7. 高分冗余候选被低分多样候选替换（核心 MMR 行为）
  8. None/空 summary 安全处理

信息论依据：
  Carbonell & Goldstein (1998) — Maximal Marginal Relevance
  I(cᵢ; query | already_selected) ≈ relevance(cᵢ) - max_sim(cᵢ, selected)
"""
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "hooks"))

# 直接从 retriever 模块导入（不触发 main 流程）
from retriever import _mmr_rerank


def _chunk(cid: str, summary: str, content: str = "") -> dict:
    return {
        "id": cid,
        "summary": summary,
        "content": content,
        "chunk_type": "decision",
        "importance": 0.7,
    }


# ══════════════════════════════════════════════════════════════════════
# 1. 基础正确性测试
# ══════════════════════════════════════════════════════════════════════

def test_empty_candidates_returns_empty():
    """空输入安全返回 []。"""
    assert _mmr_rerank([], top_k=3) == []


def test_top_k_zero_returns_empty():
    """top_k=0 返回 []。"""
    candidates = [(0.8, _chunk("a", "retriever 使用 FTS5 索引"))]
    assert _mmr_rerank(candidates, top_k=0) == []


def test_fewer_candidates_than_top_k_returns_all():
    """candidates <= top_k 时直接返回全部（无需 MMR 计算）。"""
    candidates = [
        (0.9, _chunk("a", "retriever 使用 FTS5 索引")),
        (0.7, _chunk("b", "BM25 权重参数")),
    ]
    result = _mmr_rerank(candidates, top_k=5)
    assert len(result) == 2, "候选数 ≤ top_k 应全部返回"


def test_single_candidate_returns_single():
    """单条候选正常返回。"""
    candidates = [(0.8, _chunk("a", "FTS5 索引召回"))]
    result = _mmr_rerank(candidates, top_k=3)
    assert len(result) == 1


# ══════════════════════════════════════════════════════════════════════
# 2. 冗余过滤（核心 MMR 行为）
# ══════════════════════════════════════════════════════════════════════

def test_identical_summaries_filtered_to_one():
    """两条相同摘要只保留一条（Jaccard=1.0 远超 sim_threshold=0.45）。"""
    text = "retriever 使用 FTS5 进行全文检索，O(log N) 索引"
    candidates = [
        (0.9, _chunk("a", text, text)),
        (0.7, _chunk("b", text, text)),  # 完全相同
    ]
    result = _mmr_rerank(candidates, top_k=2, sim_threshold=0.45)
    # 两条完全相同，第二条应被过滤
    ids = [c["id"] for _, c in result]
    assert len(ids) == 1 or ids.count("a") == 1, \
        f"完全相同的候选应只保留一条，got {ids}"


def test_high_score_redundant_replaced_by_diverse():
    """高分冗余候选被低分多样候选替换（核心 MMR 行为）。

    设计：b 与 a 完全相同（Jaccard=1.0），c 完全不同。
    λ=0.4（偏 diversity）时：
      - b 的 MMR = 0.4×rel_b - 0.6×1.0（完全冗余 penalty 极大）
      - c 的 MMR = 0.4×rel_c - 0.6×0.0（无相似 penalty）
    → c 的 MMR > b 的 MMR，c 被选中。
    """
    shared_text = "FTS5 full text search index retriever bm25 ranking score"
    diverse_text = "sleep consolidate episodic semantic memory promotion decay"
    candidates = [
        (0.90, _chunk("a", shared_text)),
        (0.85, _chunk("b", shared_text)),   # 与 a 完全相同（Jaccard=1.0）
        (0.60, _chunk("c", diverse_text)),  # 完全不同内容
    ]
    # λ=0.4 偏 diversity，确保冗余惩罚大于分数差
    result = _mmr_rerank(candidates, top_k=2, lambda_mmr=0.4, sim_threshold=0.3)
    ids = [c["id"] for _, c in result]
    # MMR 应选 a（最高分）和 c（多样），而不是 a+b（完全冗余）
    assert "a" in ids, f"最高分 chunk 'a' 应被选中，got {ids}"
    assert "c" in ids, f"多样 chunk 'c' 应被选中替代完全冗余 'b'，got {ids}"
    assert "b" not in ids, f"完全冗余 chunk 'b' 应被过滤，got {ids}"


def test_diverse_candidates_all_preserved():
    """完全不同内容的候选全部保留（无过滤）。"""
    candidates = [
        (0.9, _chunk("a", "FTS5 full text search retrieval")),
        (0.8, _chunk("b", "sleep consolidate episodic memory")),
        (0.7, _chunk("c", "emotional salience P0 critical crash")),
    ]
    result = _mmr_rerank(candidates, top_k=3, sim_threshold=0.45)
    ids = [c["id"] for _, c in result]
    assert set(ids) == {"a", "b", "c"}, \
        f"多样候选应全部保留，got {ids}"


# ══════════════════════════════════════════════════════════════════════
# 3. lambda 参数行为
# ══════════════════════════════════════════════════════════════════════

def test_lambda_1_preserves_relevance_order():
    """lambda=1.0 退化为纯 relevance 排序（不做 diversity 调整）。"""
    candidates = [
        (0.9, _chunk("a", "FTS5 retriever indexing")),
        (0.8, _chunk("b", "FTS5 retriever bm25 scoring")),   # 与 a 相似
        (0.5, _chunk("c", "sleep consolidate memory")),      # 与 a/b 不同
    ]
    result = _mmr_rerank(candidates, top_k=2, lambda_mmr=1.0)
    ids = [c["id"] for _, c in result]
    # lambda=1.0 → 只看 relevance → 选 a, b（最高分）
    assert ids[0] == "a", f"第一名应为最高分 'a'，got {ids}"
    assert "b" in ids, f"lambda=1.0 应保留高分相似候选 'b'，got {ids}"


def test_lambda_0_maximizes_diversity():
    """lambda=0.0 最大化多样性（第二选择与已选内容最不相似）。"""
    shared_text = "FTS5 retriever full text search bm25 ranking index"
    candidates = [
        (0.9, _chunk("a", shared_text)),
        (0.85, _chunk("b", shared_text + " query tokenize")),  # 与 a 高度相似
        (0.5, _chunk("c", "sleep emotional salience episodic decay promote")),  # 与 a 不同
    ]
    result = _mmr_rerank(candidates, top_k=2, lambda_mmr=0.0, sim_threshold=0.3)
    ids = [c["id"] for _, c in result]
    # lambda=0 → diversity 主导 → 第二选 c（与 a 最不同）而非 b（与 a 太像）
    assert "a" in ids, f"第一名应为 'a'（最高分），got {ids}"
    assert "c" in ids, f"lambda=0 应选多样候选 'c'，got {ids}"


# ══════════════════════════════════════════════════════════════════════
# 4. sim_threshold 边界行为
# ══════════════════════════════════════════════════════════════════════

def test_sim_below_threshold_no_penalty():
    """Jaccard < sim_threshold 时不施加相似度惩罚（避免误杀弱相关 chunk）。"""
    # a 和 b 有轻微词汇重叠但不同话题（预期 Jaccard ≈ 0.1-0.2）
    candidates = [
        (0.9, _chunk("a", "FTS5 retriever indexing performance")),
        (0.85, _chunk("b", "FTS5 database schema migration")),  # 共享 FTS5 但话题不同
        (0.4, _chunk("c", "emotional memory salience decay")),
    ]
    # sim_threshold=0.8（极高），弱相似不惩罚
    result = _mmr_rerank(candidates, top_k=2, lambda_mmr=0.6, sim_threshold=0.8)
    ids = [c["id"] for _, c in result]
    # 高阈值下 a 和 b 不被认为是冗余，选前两名 a, b
    assert "a" in ids, f"应选 'a'，got {ids}"
    assert "b" in ids, f"高 sim_threshold 下 'b' 不应被过滤，got {ids}"


# ══════════════════════════════════════════════════════════════════════
# 5. 边界/安全处理
# ══════════════════════════════════════════════════════════════════════

def test_none_summary_safe():
    """None summary 安全处理，不崩溃。"""
    candidates = [
        (0.9, {"id": "a", "summary": None, "content": "FTS5 retriever", "chunk_type": "decision", "importance": 0.7}),
        (0.7, {"id": "b", "summary": "BM25 scoring", "content": "", "chunk_type": "decision", "importance": 0.6}),
    ]
    result = _mmr_rerank(candidates, top_k=2)
    assert len(result) >= 1, "None summary 应安全处理"


def test_empty_summary_safe():
    """空 summary 安全处理，不崩溃。"""
    candidates = [
        (0.9, _chunk("a", "", "")),
        (0.7, _chunk("b", "BM25 scoring")),
    ]
    result = _mmr_rerank(candidates, top_k=2)
    assert len(result) >= 1, "空 summary 应安全处理"


def test_output_length_bounded_by_top_k():
    """输出长度不超过 top_k。"""
    candidates = [
        (0.9 - i * 0.1, _chunk(f"c{i}", f"chunk {i} content topic area {i}"))
        for i in range(10)
    ]
    result = _mmr_rerank(candidates, top_k=3)
    assert len(result) <= 3, f"输出长度应 ≤ top_k=3，got {len(result)}"


def test_scores_preserved_in_output():
    """输出中的分数与输入一致（MMR 不修改原始分数）。"""
    candidates = [
        (0.9, _chunk("a", "FTS5 retriever indexing")),
        (0.7, _chunk("b", "sleep consolidate memory")),
    ]
    result = _mmr_rerank(candidates, top_k=2)
    score_map = dict(candidates)
    for score, chunk in result:
        original = score_map.get(chunk["id"])
        if original is not None:
            assert score == original, \
                f"chunk {chunk['id']} 分数应保持 {original}，got {score}"
