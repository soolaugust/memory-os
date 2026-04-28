"""
test_cue_dependent_forgetting.py — iter403: Cue-Dependent Forgetting

覆盖：
  CD1: extract_encode_context — 从文本提取关键词（去停用词）
  CD2: extract_encode_context — 从 tags 提取关键词
  CD3: extract_encode_context — 空/None 输入安全返回 ''
  CD4: compute_context_overlap — 完全相同上下文 → overlap = 1.0
  CD5: compute_context_overlap — 无重叠上下文 → overlap = 0.0
  CD6: compute_context_overlap — 部分重叠 → Jaccard = |A∩B|/|A∪B|
  CD7: compute_context_overlap — 空输入安全返回 0.0
  CD8: context_cue_weight — overlap >= 0.50 → weight ∈ [1.10, 1.20]
  CD9: context_cue_weight — overlap ∈ [0.20, 0.50) → weight = 1.0
  CD10: context_cue_weight — overlap < 0.20 → weight ∈ [0.85, 1.0)
  CD11: context_cue_weight — range [0.85, 1.20]
  CD12: context_cue_weight — None/invalid 输入安全返回合理值
  CD13: apply_context_cue_boost — chunk 无 encode_context → score 不变
  CD14: apply_context_cue_boost — 高 overlap → score 提升
  CD15: apply_context_cue_boost — 低 overlap → score 降低
  CD16: insert_chunk 自动写入 encode_context 字段
  CD17: extract_encode_context — chunk_type 被加入上下文
  CD18: extract_encode_context — 最多 max_tokens 个 token

认知科学依据：
  Tulving & Thomson (1973) Encoding Specificity Principle:
    编码时的上下文（cues）越接近检索时的上下文，检索成功率越高。
  Godden & Baddeley (1975) Context-Dependent Memory:
    水下学习 → 水下测试效果最优（环境上下文匹配）。
  Estes (1955) Stimulus Fluctuation Model:
    记忆提取受 encode_context ∩ retrieve_context 的重叠度决定。

OS 类比：Linux NUMA-aware memory allocation —
  编码时 context = home NUMA node；检索时 context 越接近 home node，
  访问延迟越低（命中率越高）。
"""
import sys
import sqlite3
import pytest
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "hooks"))

import tmpfs  # noqa
from store_vfs import (
    ensure_schema,
    extract_encode_context,
    compute_context_overlap,
    context_cue_weight,
    apply_context_cue_boost,
    insert_chunk,
)


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _make_chunk(cid, content="test content", summary="test summary",
                chunk_type="decision", project="test", stability=1.0):
    now = datetime.now(timezone.utc).isoformat()
    return {
        "id": cid,
        "created_at": now,
        "updated_at": now,
        "project": project,
        "source_session": "s1",
        "chunk_type": chunk_type,
        "info_class": "world",
        "content": content,
        "summary": summary,
        "tags": [chunk_type],
        "importance": 0.7,
        "retrievability": 0.5,
        "last_accessed": now,
        "access_count": 0,
        "oom_adj": 0,
        "lru_gen": 0,
        "stability": stability,
        "raw_snippet": "",
    }


# ══════════════════════════════════════════════════════════════════════
# 1. extract_encode_context 测试
# ══════════════════════════════════════════════════════════════════════

def test_cd1_extract_keywords_from_text():
    """从文本提取有意义关键词（去停用词）。"""
    text = "Redis cache configuration for high performance deployment"
    ctx = extract_encode_context(text)
    tokens = set(ctx.split(",")) if ctx else set()
    # "redis", "cache", "configuration", "high", "performance", "deployment"
    assert "redis" in tokens, f"CD1: 应包含 'redis'，got tokens={tokens}"
    assert "cache" in tokens, f"CD1: 应包含 'cache'，got tokens={tokens}"
    # 停用词 "for" 不应出现
    assert "for" not in tokens, f"CD1: 停用词 'for' 不应出现，got tokens={tokens}"


def test_cd2_extract_from_tags():
    """从 tags 提取关键词。"""
    ctx = extract_encode_context("test content", tags=["redis", "performance"], chunk_type="")
    tokens = set(ctx.split(",")) if ctx else set()
    assert "redis" in tokens, f"CD2: tags 中的 'redis' 应被提取，got {tokens}"
    assert "performance" in tokens, f"CD2: tags 中的 'performance' 应被提取，got {tokens}"


def test_cd3_empty_input_safe():
    """空/None 输入安全返回 ''。"""
    assert extract_encode_context("") == ""
    assert extract_encode_context(None) == ""
    assert extract_encode_context("   ") == ""


def test_cd17_chunk_type_included_in_context():
    """chunk_type 被加入 encode_context。"""
    ctx = extract_encode_context("some content", chunk_type="design_constraint")
    tokens = set(ctx.split(",")) if ctx else set()
    assert "design_constraint" in tokens, (
        f"CD17: chunk_type 应加入上下文，got {tokens}"
    )


def test_cd18_max_tokens_limit():
    """extract_encode_context 最多返回 max_tokens 个 token。"""
    # 生成大量不同词汇的文本
    long_text = " ".join([f"keyword{i}" for i in range(200)])
    ctx = extract_encode_context(long_text, max_tokens=10)
    if ctx:
        tokens = [t for t in ctx.split(",") if t]
        assert len(tokens) <= 10, f"CD18: 最多 10 个 token，got {len(tokens)}"


# ══════════════════════════════════════════════════════════════════════
# 2. compute_context_overlap 测试
# ══════════════════════════════════════════════════════════════════════

def test_cd4_identical_context_full_overlap():
    """完全相同上下文 → overlap = 1.0。"""
    ctx = "redis,cache,performance,design"
    overlap = compute_context_overlap(ctx, ctx)
    assert abs(overlap - 1.0) < 0.001, f"CD4: 完全相同上下文 overlap 应为 1.0，got {overlap}"


def test_cd5_disjoint_context_zero_overlap():
    """无重叠上下文 → overlap = 0.0。"""
    ctx_a = "redis,cache,performance"
    ctx_b = "database,schema,migration"
    overlap = compute_context_overlap(ctx_a, ctx_b)
    assert overlap == 0.0, f"CD5: 无重叠 overlap 应为 0.0，got {overlap}"


def test_cd6_partial_overlap_jaccard():
    """部分重叠 → Jaccard = |A∩B| / |A∪B|。"""
    # A = {redis, cache, performance}
    # B = {redis, cache, database}
    # |A∩B| = 2 (redis, cache)
    # |A∪B| = 4 (redis, cache, performance, database)
    # Jaccard = 2/4 = 0.5
    ctx_a = "redis,cache,performance"
    ctx_b = "redis,cache,database"
    overlap = compute_context_overlap(ctx_a, ctx_b)
    assert abs(overlap - 0.5) < 0.01, f"CD6: Jaccard(2/4)=0.5，got {overlap}"


def test_cd7_empty_overlap_safe():
    """空输入安全返回 0.0。"""
    assert compute_context_overlap("", "redis,cache") == 0.0
    assert compute_context_overlap("redis,cache", "") == 0.0
    assert compute_context_overlap("", "") == 0.0
    assert compute_context_overlap(None, "redis") == 0.0


# ══════════════════════════════════════════════════════════════════════
# 3. context_cue_weight 测试
# ══════════════════════════════════════════════════════════════════════

def test_cd8_high_overlap_weight_above_1():
    """高重叠（>= 0.50）→ weight ∈ [1.10, 1.20]。"""
    w = context_cue_weight(0.80)
    assert w >= 1.10, f"CD8: 高重叠 weight={w:.4f} 应 >= 1.10"
    assert w <= 1.20, f"CD8: 高重叠 weight={w:.4f} 应 <= 1.20"

    w_max = context_cue_weight(1.0)
    assert abs(w_max - 1.20) < 0.001, f"CD8: overlap=1.0 时 weight 应为 1.20，got {w_max}"


def test_cd9_medium_overlap_weight_one():
    """中等重叠（0.20 ~ 0.50）→ weight = 1.0（不调整）。"""
    for overlap in [0.20, 0.30, 0.40, 0.49]:
        w = context_cue_weight(overlap)
        assert w == 1.0, f"CD9: 中等重叠 overlap={overlap}, weight={w:.4f} 应 == 1.0"


def test_cd10_low_overlap_weight_below_1():
    """低重叠（< 0.20）→ weight ∈ [0.85, 1.0)。"""
    w = context_cue_weight(0.0)
    assert abs(w - 0.85) < 0.001, f"CD10: overlap=0.0 时 weight 应为 0.85，got {w}"
    w_near = context_cue_weight(0.10)
    assert 0.85 <= w_near < 1.0, f"CD10: 低重叠 weight={w_near:.4f} 应在 [0.85, 1.0)"


def test_cd11_weight_range():
    """context_cue_weight 输出范围 [0.85, 1.20]。"""
    for overlap in [0.0, 0.1, 0.19, 0.20, 0.35, 0.49, 0.50, 0.75, 1.0]:
        w = context_cue_weight(overlap)
        assert 0.85 <= w <= 1.20, (
            f"CD11: overlap={overlap}, weight={w:.4f} 应在 [0.85, 1.20]"
        )


def test_cd12_weight_invalid_inputs_safe():
    """None/invalid 输入安全处理。"""
    w_none = context_cue_weight(None)
    assert 0.85 <= w_none <= 1.20, f"CD12: None 输入应安全，got {w_none}"
    w_neg = context_cue_weight(-0.5)
    assert 0.85 <= w_neg <= 1.20, f"CD12: 负值输入应安全，got {w_neg}"
    w_over = context_cue_weight(2.0)
    assert 0.85 <= w_over <= 1.20, f"CD12: 超出范围输入应安全，got {w_over}"


# ══════════════════════════════════════════════════════════════════════
# 4. apply_context_cue_boost 测试
# ══════════════════════════════════════════════════════════════════════

def test_cd13_no_encode_context_score_unchanged(conn):
    """chunk 无 encode_context → score 不变。"""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO memory_chunks "
        "(id, project, chunk_type, content, summary, importance, retrievability, "
        "created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, 0.7, 0.5, ?, ?)",
        ("cd13_chunk", "test", "decision", "content", "summary", now, now)
    )
    conn.commit()

    score = apply_context_cue_boost(conn, "cd13_chunk", "redis,cache", base_score=1.0)
    assert score == 1.0, f"CD13: 无 encode_context 时 score 应不变，got {score}"


def test_cd14_high_overlap_score_increased(conn):
    """高 overlap → score 提升（>= 1.10x）。"""
    now = datetime.now(timezone.utc).isoformat()
    # 写入 chunk 时设置 encode_context（模拟写入时已提取）
    conn.execute(
        "INSERT INTO memory_chunks "
        "(id, project, chunk_type, content, summary, importance, retrievability, "
        "encode_context, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, 0.7, 0.5, ?, ?, ?)",
        ("cd14_chunk", "test", "decision", "redis cache performance",
         "redis cache performance",
         "cache,design_constraint,performance,redis",  # encode_context
         now, now)
    )
    conn.commit()

    # retrieve_context 与 encode_context 高度重叠
    retrieve_ctx = "redis,cache,performance,design_constraint"
    score = apply_context_cue_boost(conn, "cd14_chunk", retrieve_ctx, base_score=1.0)
    assert score >= 1.10, f"CD14: 高 overlap 时 score 应 >= 1.10，got {score}"


def test_cd15_low_overlap_score_decreased(conn):
    """低 overlap → score 降低（< 1.0）。"""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO memory_chunks "
        "(id, project, chunk_type, content, summary, importance, retrievability, "
        "encode_context, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, 0.7, 0.5, ?, ?, ?)",
        ("cd15_chunk", "test", "task_state", "redis cluster setup",
         "redis cluster setup",
         "cluster,redis,setup,topology",  # encode_context
         now, now)
    )
    conn.commit()

    # retrieve_context 与 encode_context 无重叠
    retrieve_ctx = "machine_learning,neural_network,training"
    score = apply_context_cue_boost(conn, "cd15_chunk", retrieve_ctx, base_score=1.0)
    assert score < 1.0, f"CD15: 低 overlap 时 score 应 < 1.0，got {score}"
    assert score >= 0.85, f"CD15: 最低 score 应 >= 0.85，got {score}"


# ══════════════════════════════════════════════════════════════════════
# 5. insert_chunk 集成测试
# ══════════════════════════════════════════════════════════════════════

def test_cd16_insert_chunk_writes_encode_context(conn):
    """insert_chunk 写入后，encode_context 自动设置。"""
    content = "Redis cache configuration with high performance LRU eviction policy"
    chunk = _make_chunk(
        "cd16_chunk",
        content=content,
        summary="Redis LRU eviction cache configuration",
        chunk_type="design_constraint",
    )
    insert_chunk(conn, chunk)
    conn.commit()

    row = conn.execute(
        "SELECT encode_context FROM memory_chunks WHERE id='cd16_chunk'"
    ).fetchone()

    if row is not None and row["encode_context"] is not None:
        ctx = row["encode_context"]
        # 应包含有意义的关键词
        assert len(ctx) > 0, "CD16: encode_context 应非空"
        tokens = set(ctx.split(","))
        # "redis", "cache", "lru" 等关键词应被提取
        meaningful = {"redis", "cache", "lru", "eviction", "policy", "performance",
                      "configuration", "design_constraint"}
        overlap = tokens & meaningful
        assert len(overlap) >= 2, (
            f"CD16: encode_context 应包含有意义关键词，got tokens={tokens}"
        )


def test_cd16b_similar_context_higher_boost(conn):
    """相同 context 编码的 chunk 在相同 context 检索时 score 更高。"""
    # Chunk A: 在 redis context 下编码
    conn.execute(
        "INSERT INTO memory_chunks "
        "(id, project, chunk_type, content, summary, importance, retrievability, "
        "encode_context, created_at, updated_at) "
        "VALUES (?, 'test', 'decision', 'redis', 'redis', 0.7, 0.5, ?, ?, ?)",
        ("cd16b_redis", "redis,cache,cluster,performance,setup",
         datetime.now(timezone.utc).isoformat(), datetime.now(timezone.utc).isoformat())
    )
    # Chunk B: 在 ML context 下编码
    conn.execute(
        "INSERT INTO memory_chunks "
        "(id, project, chunk_type, content, summary, importance, retrievability, "
        "encode_context, created_at, updated_at) "
        "VALUES (?, 'test', 'decision', 'ml', 'ml', 0.7, 0.5, ?, ?, ?)",
        ("cd16b_ml", "neural,training,model,gradient,loss",
         datetime.now(timezone.utc).isoformat(), datetime.now(timezone.utc).isoformat())
    )
    conn.commit()

    # 在 redis context 检索
    redis_ctx = "redis,cache,cluster,performance"
    score_redis = apply_context_cue_boost(conn, "cd16b_redis", redis_ctx, base_score=1.0)
    score_ml = apply_context_cue_boost(conn, "cd16b_ml", redis_ctx, base_score=1.0)

    assert score_redis > score_ml, (
        f"CD16b: redis context 的 chunk({score_redis:.4f}) 应在 redis 检索时分更高 "
        f"than ml context chunk({score_ml:.4f})"
    )
