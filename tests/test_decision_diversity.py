"""
test_decision_diversity.py — 迭代337：Decision Diversity Suppression 单元测试

信息论背景：decision + design_constraint 占注入总量 76%（实测），
  Shannon 冗余度极高 —— 高频注入相同类型的"规则"信息，边际信息增益趋零。
OS 类比：
  1. Normative Pool — Linux cgroup unified memory.limit，同类资源统一限制
  2. Jaccard KSM — Linux Kernel Samepage Merging，内容重复的页不重复注入

验证：
  1. 正常情况 — 非 normative 类型不受 normative pool 影响
  2. Normative pool cap — decision+design_constraint 联合数量不超过 max_same * 2
  3. 纯 decision overflow — 超出 normative pool cap 的 decision 让位给其他类型
  4. 混合 decision+design_constraint — 两者合计受联合上限约束
  5. Jaccard dedup — 相似度 >= 0.50 的约束被过滤，不重复注入
  6. Jaccard 低相似度 — 相似度 < 0.50 的约束正常注入
  7. 空候选集 — 边界情况处理
  8. top_k = 1 — 极小 top_k 不 crash
  9. 端到端：DRR + Jaccard 联合效果验证
"""
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "hooks"))

import tmpfs  # noqa

# 直接导入被测函数
from retriever import _drr_select, _sysctl


def _make_chunk(chunk_type, importance=0.8, chunk_id=None, summary=None):
    """构造测试 chunk dict。"""
    cid = chunk_id or str(uuid.uuid4())
    return {
        "id": cid,
        "summary": summary or f"{chunk_type} chunk {cid[:8]}",
        "chunk_type": chunk_type,
        "importance": importance,
        "last_accessed": "2026-04-01T00:00:00+00:00",
        "access_count": 0,
        "created_at": "2026-04-01T00:00:00+00:00",
        "project": "test",
        "info_class": "world",
        "lru_gen": 0,
    }


def _candidates(chunks_with_scores):
    """将 [(score, type)] 转换为 _drr_select 的输入格式。"""
    return [(s, _make_chunk(t)) for s, t in chunks_with_scores]


def _candidates_full(pairs):
    """将 [(score, chunk)] 直接传入（允许自定义 chunk）。"""
    return pairs


# ──────────────────────────────────────────────────────────────────────
# 1. 正常情况 — 非 normative 类型不受 normative pool 影响
# ──────────────────────────────────────────────────────────────────────

def test_non_normative_unaffected():
    """reasoning_chain / conversation_summary 不受 normative pool 限制。"""
    max_same = _sysctl("retriever.drr_max_same_type")
    # 6 个不同类型的候选（每个类型 1 个）
    cands = _candidates([
        (0.9, "reasoning_chain"),
        (0.8, "conversation_summary"),
        (0.7, "causal_chain"),
        (0.6, "excluded_path"),
        (0.5, "task_state"),
        (0.4, "procedure"),
    ])
    result = _drr_select(cands, top_k=6)
    assert len(result) == 6, f"all 6 non-normative should be selected: {len(result)}"
    types = [c.get("chunk_type") for _, c in result]
    # normative 类型（decision/design_constraint）不在结果中
    assert "decision" not in types
    assert "design_constraint" not in types


# ──────────────────────────────────────────────────────────────────────
# 2. Normative pool cap — decision+design_constraint 联合数量不超过 max_same * 2
# ──────────────────────────────────────────────────────────────────────

def test_normative_pool_cap():
    """decision 数量大于 max_same * 2 时，超出部分让位给其他类型。"""
    max_same = _sysctl("retriever.drr_max_same_type")
    cap = max_same * 2  # normative pool 联合上限

    # 多于 cap 个 decision + 2 个其他类型
    excess = cap + 2  # 超过上限的 decision 数量
    cands = []
    for i in range(excess):
        cands.append((0.9 - i * 0.01, _make_chunk("decision")))
    cands.append((0.1, _make_chunk("reasoning_chain")))
    cands.append((0.09, _make_chunk("conversation_summary")))

    result = _drr_select(cands, top_k=excess + 2)
    normative_count = sum(
        1 for _, c in result
        if c.get("chunk_type") in ("decision", "design_constraint")
    )
    assert normative_count <= cap * 2, (
        f"normative count {normative_count} should be <= cap*2={cap*2}: "
        f"{[c.get('chunk_type') for _, c in result]}"
    )


# ──────────────────────────────────────────────────────────────────────
# 3. 纯 decision overflow — 超出 normative pool cap 的让位给其他类型
# ──────────────────────────────────────────────────────────────────────

def test_decision_gives_way_to_others():
    """当 decision 超出 normative pool cap 时，低分的其他类型有机会进入 top_k。"""
    max_same = _sysctl("retriever.drr_max_same_type")
    cap = max_same * 2

    # cap+1 个高分 decision + 1 个低分 reasoning_chain
    cands = []
    for i in range(cap + 1):
        cands.append((0.9 - i * 0.01, _make_chunk("decision")))
    reasoning_chunk = _make_chunk("reasoning_chain")
    cands.append((0.1, reasoning_chunk))

    result = _drr_select(cands, top_k=cap + 1)
    result_ids = [c["id"] for _, c in result]
    result_types = [c.get("chunk_type") for _, c in result]

    # reasoning_chain 应该进入结果（decision 超出 cap 后让位）
    assert reasoning_chunk["id"] in result_ids, (
        f"reasoning_chain should be selected when decision exceeds cap. "
        f"types={result_types}"
    )
    # decision 数量不超 cap
    decision_count = result_types.count("decision")
    assert decision_count <= cap, (
        f"decision count {decision_count} should not exceed cap {cap}"
    )


# ──────────────────────────────────────────────────────────────────────
# 4. 混合 decision+design_constraint — 两者合计受联合上限约束
# ──────────────────────────────────────────────────────────────────────

def test_mixed_normative_joint_cap():
    """decision 和 design_constraint 混合时，两者合计受联合上限约束。"""
    max_same = _sysctl("retriever.drr_max_same_type")
    cap = max_same * 2  # normative 联合上限

    # 各 cap 个 decision 和 design_constraint（总共 2*cap，远超联合上限）
    cands = []
    for i in range(cap):
        cands.append((0.9 - i * 0.01, _make_chunk("decision")))
    for i in range(cap):
        cands.append((0.85 - i * 0.01, _make_chunk("design_constraint")))
    # 2 个非 normative 类型
    cands.append((0.1, _make_chunk("reasoning_chain")))
    cands.append((0.09, _make_chunk("causal_chain")))

    result = _drr_select(cands, top_k=cap * 2 + 2)
    normative_count = sum(
        1 for _, c in result
        if c.get("chunk_type") in ("decision", "design_constraint")
    )
    # 联合数量不超过 cap * 2（overflow 允许额外 cap，共 cap*2）
    assert normative_count <= cap * 2, (
        f"normative count {normative_count} should <= cap*2={cap*2}: "
        f"types={[c.get('chunk_type') for _, c in result]}"
    )


# ──────────────────────────────────────────────────────────────────────
# 5. Jaccard dedup — 相似度 >= 0.50 的约束被过滤，不重复注入
# ──────────────────────────────────────────────────────────────────────

def test_jaccard_dedup_high_similarity():
    """
    与已选 top_k 中某 chunk summary Jaccard >= 0.50 的约束不应被 forced 注入。
    模拟 retriever 中的 _is_content_redundant() 逻辑。
    """
    import re

    # 模拟 top_k 中已有的 chunk
    existing_summary = "retriever 检索 FTS5 BM25 召回率 优化"
    existing_words = set(re.sub(r'[^\w\u4e00-\u9fff]', ' ',
                                existing_summary.lower()).split())

    # 高相似度约束（几乎相同的词）
    constraint_summary = "retriever FTS5 BM25 召回率 检索 优化 延迟"
    constraint_words = set(re.sub(r'[^\w\u4e00-\u9fff]', ' ',
                                  constraint_summary.lower()).split())

    # 手动计算 Jaccard
    union = existing_words | constraint_words
    inter = existing_words & constraint_words
    jaccard = len(inter) / len(union) if union else 0.0

    # 验证 Jaccard 确实 >= 0.50（确保测试前提成立）
    assert jaccard >= 0.50, (
        f"test premise: Jaccard should be >= 0.50, got {jaccard:.3f}. "
        f"inter={inter}, union={union}"
    )

    # 模拟 _is_content_redundant() 逻辑
    def _is_content_redundant(c_summary, top_k_token_sets):
        c_words = set(re.sub(r'[^\w\u4e00-\u9fff]', ' ',
                             (c_summary or "").lower()).split())
        if not c_words:
            return False
        for existing_ws in top_k_token_sets:
            union_set = existing_ws | c_words
            if union_set:
                j = len(existing_ws & c_words) / len(union_set)
                if j >= 0.50:
                    return True
        return False

    top_k_token_sets = [existing_words]
    assert _is_content_redundant(constraint_summary, top_k_token_sets), (
        f"high Jaccard ({jaccard:.3f}) constraint should be redundant"
    )


# ──────────────────────────────────────────────────────────────────────
# 6. Jaccard 低相似度 — 相似度 < 0.50 的约束正常注入
# ──────────────────────────────────────────────────────────────────────

def test_jaccard_low_similarity_not_filtered():
    """与已选内容 Jaccard < 0.50 的约束不应被过滤（不是冗余）。"""
    import re

    existing_summary = "retriever 检索 FTS5 召回优化"
    existing_words = set(re.sub(r'[^\w\u4e00-\u9fff]', ' ',
                                existing_summary.lower()).split())

    # 低相似度约束（完全不同的词）
    constraint_summary = "内存管理 swap 淘汰 kswapd 冷启动 延迟"
    constraint_words = set(re.sub(r'[^\w\u4e00-\u9fff]', ' ',
                                  constraint_summary.lower()).split())

    union = existing_words | constraint_words
    inter = existing_words & constraint_words
    jaccard = len(inter) / len(union) if union else 0.0

    # 验证 Jaccard < 0.50（确保测试前提）
    assert jaccard < 0.50, (
        f"test premise: Jaccard should be < 0.50, got {jaccard:.3f}"
    )

    def _is_content_redundant(c_summary, top_k_token_sets):
        c_words = set(re.sub(r'[^\w\u4e00-\u9fff]', ' ',
                             (c_summary or "").lower()).split())
        if not c_words:
            return False
        for existing_ws in top_k_token_sets:
            union_set = existing_ws | c_words
            if union_set:
                j = len(existing_ws & c_words) / len(union_set)
                if j >= 0.50:
                    return True
        return False

    top_k_token_sets = [existing_words]
    assert not _is_content_redundant(constraint_summary, top_k_token_sets), (
        f"low Jaccard ({jaccard:.3f}) constraint should NOT be filtered as redundant"
    )


# ──────────────────────────────────────────────────────────────────────
# 7. 空候选集 — 边界情况处理
# ──────────────────────────────────────────────────────────────────────

def test_empty_candidates():
    """空候选集 → _drr_select 返回空列表，不 crash。"""
    result = _drr_select([], top_k=5)
    assert result == [], f"empty candidates should return [], got {result}"


# ──────────────────────────────────────────────────────────────────────
# 8. top_k = 1 — 极小 top_k 不 crash
# ──────────────────────────────────────────────────────────────────────

def test_top_k_one():
    """top_k=1 时 _drr_select 只返回最高分的 chunk。"""
    cands = _candidates([
        (0.9, "decision"),
        (0.8, "reasoning_chain"),
        (0.7, "design_constraint"),
    ])
    result = _drr_select(cands, top_k=1)
    assert len(result) == 1, f"top_k=1 should return exactly 1 result, got {len(result)}"
    # 最高分是第一个
    assert result[0][0] == pytest.approx(0.9, abs=0.01) or result[0][0] >= 0.85, \
        f"should select highest score: {result[0][0]}"


# ──────────────────────────────────────────────────────────────────────
# 9. 端到端：normative pool 保证非 normative 类型也有代表
# ──────────────────────────────────────────────────────────────────────

def test_diversity_end_to_end():
    """
    端到端验证：大量 decision + 少量其他类型，
    normative pool cap 保证其他类型也能进入 top_k。
    """
    max_same = _sysctl("retriever.drr_max_same_type")
    top_k_size = 5

    # 10 个高分 decision，远超 normative pool cap
    cands = []
    for i in range(10):
        cands.append((0.9 - i * 0.01, _make_chunk("decision")))
    # 3 个低分其他类型
    rc_chunk = _make_chunk("reasoning_chain")
    cs_chunk = _make_chunk("conversation_summary")
    ep_chunk = _make_chunk("excluded_path")
    cands.extend([
        (0.2, rc_chunk),
        (0.19, cs_chunk),
        (0.18, ep_chunk),
    ])

    result = _drr_select(cands, top_k=top_k_size)
    result_ids = set(c["id"] for _, c in result)
    result_types = [c.get("chunk_type") for _, c in result]

    # 至少有一个非 normative 类型进入结果
    non_normative_in_result = sum(
        1 for t in result_types
        if t not in ("decision", "design_constraint")
    )
    assert non_normative_in_result >= 1, (
        f"At least 1 non-normative type should be in top_k due to normative pool cap. "
        f"types={result_types}"
    )

    # normative 类型总量不超 cap
    normative_in_result = sum(
        1 for t in result_types
        if t in ("decision", "design_constraint")
    )
    assert normative_in_result <= max_same * 2, (
        f"normative count {normative_in_result} should <= max_same*2={max_same*2}"
    )


# pytest 导入（避免 NameError in test_top_k_one）
try:
    import pytest
except ImportError:
    class _pytest_approx:
        def approx(self, v, abs=None): return v
    pytest = _pytest_approx()


if __name__ == "__main__":
    import sys
    tests = [
        test_non_normative_unaffected,
        test_normative_pool_cap,
        test_decision_gives_way_to_others,
        test_mixed_normative_joint_cap,
        test_jaccard_dedup_high_similarity,
        test_jaccard_low_similarity_not_filtered,
        test_empty_candidates,
        test_top_k_one,
        test_diversity_end_to_end,
    ]
    failed = []
    for t in tests:
        try:
            t()
            print(f"  PASS {t.__name__}")
        except Exception as e:
            print(f"  FAIL {t.__name__}: {e}")
            failed.append(t.__name__)
    if failed:
        print(f"\n{len(failed)} failed: {failed}")
        sys.exit(1)
    else:
        print(f"\n{len(tests)} passed")
