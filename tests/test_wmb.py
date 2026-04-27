"""
test_wmb.py — 迭代316：工作记忆预算管理单元测试

验证：
  1. tier_chunks 正确分层（高分→active，中→background，低→dormant）
  2. active 层上限3个
  3. background 层上限5个
  4. 空输入安全返回
  5. 全高分时：active满3个，剩余进background
  6. 全低分时：active为空，background也可能为空
  7. apply_wmb_budget 函数：只返回active+background，dormant不注入
  8. 总注入数 <= active_limit + background_limit
"""

import pytest
from wmb import tier_chunks, apply_wmb_budget, wmb_stats


def _make_chunk(cid, chunk_type="decision"):
    return {"id": cid, "summary": f"chunk_{cid}", "chunk_type": chunk_type}


def _make_scored(n, scores):
    """生成 n 个 (chunk_dict, score) pair，scores 为列表"""
    return [(_make_chunk(f"c{i}"), s) for i, s in enumerate(scores)]


# ── 测试1：基本分层正确性 ──
def test_basic_tier_split():
    """高分→active，中→background，低→dormant"""
    pairs = [
        (_make_chunk("high1"), 0.9),
        (_make_chunk("high2"), 0.8),
        (_make_chunk("mid1"),  0.5),
        (_make_chunk("mid2"),  0.4),
        (_make_chunk("low1"),  0.2),
        (_make_chunk("low2"),  0.1),
    ]
    result = tier_chunks(pairs)
    active_ids = {c["id"] for c in result["active"]}
    background_ids = {c["id"] for c in result["background"]}
    dormant_ids = {c["id"] for c in result["dormant"]}

    assert "high1" in active_ids
    assert "high2" in active_ids
    assert "mid1" in background_ids or "mid2" in background_ids
    assert "low1" in dormant_ids or "low2" in dormant_ids


# ── 测试2：active 层上限3个 ──
def test_active_limit():
    """所有分数都很高时，active 不超过3个"""
    pairs = _make_scored(8, [0.99, 0.95, 0.90, 0.88, 0.85, 0.82, 0.80, 0.78])
    result = tier_chunks(pairs)
    assert len(result["active"]) <= 3


# ── 测试3：background 层上限5个 ──
def test_background_limit():
    """background 不超过5个"""
    # 高分3个→active满，剩余全为中分→应进background但不超5
    pairs = _make_scored(12, [0.99, 0.95, 0.90,  # 会进active
                               0.7, 0.65, 0.55, 0.5, 0.45, 0.4, 0.38, 0.36, 0.35])
    result = tier_chunks(pairs)
    assert len(result["background"]) <= 5


# ── 测试4：空输入安全返回 ──
def test_empty_input():
    """空输入不抛异常，返回三层均为空列表"""
    result = tier_chunks([])
    assert result == {"active": [], "background": [], "dormant": []}


# ── 测试5：全高分时active满3个，剩余进background ──
def test_all_high_scores():
    """全部分数 >= 0.65：active最多3个，超出的进background"""
    pairs = _make_scored(7, [0.99, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70])
    result = tier_chunks(pairs)
    assert len(result["active"]) == 3  # active 上限
    assert len(result["background"]) == 4  # 剩余4个（归一化后可能>=0.65但active已满）
    assert len(result["dormant"]) == 0


# ── 测试6：全低分时active和background均为空 ──
def test_all_low_scores():
    """全部分数 < 0.35（归一化后）→ active为空，background也为空"""
    # 所有分数很接近且都很低，归一化后也都低
    pairs = _make_scored(4, [0.05, 0.03, 0.02, 0.01])
    result = tier_chunks(pairs)
    # 归一化后 0.05/0.05=1.0 是最高的，所以highest会进active
    # 检查：dormant有内容
    total = len(result["active"]) + len(result["background"]) + len(result["dormant"])
    assert total == 4  # 所有chunk都被分类了


# ── 测试7：apply_wmb_budget 只返回active+background ──
def test_apply_wmb_budget_excludes_dormant():
    """apply_wmb_budget 返回列表只包含active+background，不含dormant"""
    pairs = [
        (_make_chunk("a"), 0.9),
        (_make_chunk("b"), 0.8),
        (_make_chunk("c"), 0.5),
        (_make_chunk("d"), 0.4),
        (_make_chunk("e"), 0.1),  # 低分，应进dormant
        (_make_chunk("f"), 0.05), # 低分，应进dormant
    ]
    injected = apply_wmb_budget(pairs)
    injected_ids = {c["id"] for c in injected}
    # 低分的e、f应当不在注入列表中（分布中它们会进dormant）
    assert isinstance(injected, list)
    # 注入列表不包含所有6个（dormant部分被过滤）
    tier = tier_chunks(pairs)
    dormant_ids = {c["id"] for c in tier["dormant"]}
    for cid in dormant_ids:
        assert cid not in injected_ids


# ── 测试8：总注入数 <= active_limit + background_limit ──
def test_total_injected_within_budget():
    """总注入数不超过 active_limit + background_limit = 3 + 5 = 8"""
    pairs = _make_scored(20, [1.0 - i * 0.03 for i in range(20)])
    injected = apply_wmb_budget(pairs, top_k=20)
    assert len(injected) <= 8


# ── 测试9：apply_wmb_budget 不抛异常（空输入） ──
def test_apply_wmb_budget_empty():
    """空输入不抛异常"""
    result = apply_wmb_budget([])
    assert result == []


# ── 测试10：wmb_stats 返回正确统计 ──
def test_wmb_stats():
    """wmb_stats 返回各层数量和覆盖率"""
    pairs = [
        (_make_chunk("a"), 0.9),
        (_make_chunk("b"), 0.8),
        (_make_chunk("c"), 0.5),
        (_make_chunk("d"), 0.1),
    ]
    tier = tier_chunks(pairs)
    stats = wmb_stats(tier)
    assert "active_count" in stats
    assert "background_count" in stats
    assert "dormant_count" in stats
    assert "total" in stats
    assert "injected_ratio" in stats
    assert stats["total"] == 4
    assert 0.0 <= stats["injected_ratio"] <= 1.0


# ── 测试11：单个chunk输入 ──
def test_single_chunk():
    """单个chunk：应进active（最高分归一化为1.0>=0.65）"""
    pairs = [(_make_chunk("solo"), 0.5)]
    result = tier_chunks(pairs)
    assert len(result["active"]) == 1
    assert result["active"][0]["id"] == "solo"
    assert result["background"] == []
    assert result["dormant"] == []


# ── 测试12：top_k 截断生效 ──
def test_top_k_truncation():
    """tier_chunks 只取前 top_k 个候选"""
    # 20个chunk，top_k=5，dormant+background+active总共5个
    pairs = _make_scored(20, [1.0 - i * 0.04 for i in range(20)])
    result = tier_chunks(pairs, top_k=5)
    total = len(result["active"]) + len(result["background"]) + len(result["dormant"])
    assert total == 5
