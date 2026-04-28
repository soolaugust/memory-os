"""
test_iter406_generation_effect.py — iter406: Generation Effect (Slamecka & Graf 1978)

覆盖：
  GE1: compute_generation_score — 强元认知内容返回高分
  GE2: compute_generation_score — 被动/描述性内容返回低分
  GE3: compute_generation_score — source_type="direct" 返回 0.0
  GE4: compute_generation_score — source_type="tool_output" 上限 0.1
  GE5: compute_generation_score — 空/None 输入返回 0.0
  GE6: generation_stability_bonus — 强分数返回 base × ~0.35
  GE7: generation_stability_bonus — 零分返回 0.0
  GE8: generation_stability_bonus — bonus 不超过 base × 0.50
  GE9: generation_stability_bonus — 单调性：分数越高 bonus 越大
  GE10: apply_generation_effect — 写入更新后的 stability 到 DB
  GE11: apply_generation_effect — 被动内容不改变 stability
  GE12: insert_chunk 集成 — 含推理语言的 chunk 得到更高 stability
  GE13: compute_generation_score 范围 [0.0, 1.0]
  GE14: None/无效输入全部安全返回

认知科学依据：
  Slamecka & Graf (1978) Generation Effect:
    自生成内容（主动推理/假设生成）比被动接收内容记忆保留率更高。
    生成行为激活更深的语义加工（Level of Processing, Craik & Lockhart 1972）。
  McDaniel & Einstein (1986):
    生成效应在复杂材料上更显著：主动构建的知识结构比誊写更持久。

OS 类比：CPU Write-Allocate cache policy —
  写缺失时先将 cache line 装载再修改（Write-Allocate，深度激活），
  vs. 直接写穿透（Write-No-Allocate，浅层激活）。
  主动生成 = Write-Allocate，stability 更高，淘汰阈值更低。
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
    compute_generation_score,
    generation_stability_bonus,
    apply_generation_effect,
    insert_chunk,
)


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _make_chunk(cid, content="test", summary="test summary",
                chunk_type="decision", project="test",
                stability=1.0, source_type=None):
    now = _now()
    d = {
        "id": cid, "created_at": now, "updated_at": now, "project": project,
        "source_session": "s1", "chunk_type": chunk_type, "info_class": "world",
        "content": content, "summary": summary, "tags": [chunk_type],
        "importance": 0.7, "retrievability": 0.5, "last_accessed": now,
        "access_count": 0, "oom_adj": 0, "lru_gen": 0, "stability": stability,
        "raw_snippet": "",
    }
    if source_type is not None:
        d["source_type"] = source_type
    return d


# ══════════════════════════════════════════════════════════════════════
# 1. compute_generation_score 纯函数测试
# ══════════════════════════════════════════════════════════════════════

def test_ge1_strong_metacognitive_content_high_score():
    """强元认知内容（多个推理标记）返回高分。"""
    content = (
        "这意味着我们需要重新设计缓存层。"
        "我认为根本原因在于 LRU 策略没有考虑访问频率。"
        "因此可以采用 LFU 替代，这说明频率比时间更重要。"
        "综上所述，这个决策是正确的。"
    )
    score = compute_generation_score(content)
    assert score >= 0.5, f"GE1: 强元认知内容应得高分，got {score:.3f}"


def test_ge2_passive_content_low_score():
    """被动/描述性内容（无推理标记）返回低分。"""
    content = (
        "Redis is an in-memory data structure store. "
        "It supports strings, hashes, lists, sets. "
        "The default port is 6379."
    )
    score = compute_generation_score(content)
    assert score <= 0.2, f"GE2: 被动描述内容应得低分，got {score:.3f}"


def test_ge3_source_type_direct_returns_zero():
    """source_type='direct' 的内容（人直接输入）返回 0.0。"""
    content = (
        "这意味着我认为我们需要重构这个模块。"
        "因此可以得出结论：现有设计有缺陷。"
        "综上所述，这是一个重要的决策。"
    )
    score = compute_generation_score(content, source_type="direct")
    assert score == 0.0, f"GE3: source_type='direct' 应返回 0.0，got {score}"


def test_ge4_source_type_tool_output_capped():
    """source_type='tool_output' 上限 0.1。"""
    content = (
        "这意味着我认为我们应该重新思考这个问题。"
        "综上所述，因此可以得出结论。"
        "这说明这个方案是正确的。"
    )
    score = compute_generation_score(content, source_type="tool_output")
    assert score <= 0.1, f"GE4: source_type='tool_output' 上限 0.1，got {score}"


def test_ge5_empty_input_returns_zero():
    """空/None 输入返回 0.0。"""
    assert compute_generation_score("") == 0.0
    assert compute_generation_score(None) == 0.0
    assert compute_generation_score("   ") == 0.0
    assert compute_generation_score("hi") == 0.0


def test_ge13_score_range():
    """compute_generation_score 输出范围 [0.0, 1.0]。"""
    test_cases = [
        "",
        "   ",
        "Redis port is 6379.",
        "我认为这意味着因此可以综上所述这说明这证明关键在于",
        "I think therefore I conclude this means in summary crucially",
        "this is a test " * 100,
        None,
    ]
    for content in test_cases:
        score = compute_generation_score(content)
        assert 0.0 <= score <= 1.0, (
            f"GE13: score 应在 [0.0, 1.0]，got {score} for content={repr(content)[:40]}"
        )


def test_ge14_none_invalid_inputs_safe():
    """None/无效输入全部安全返回。"""
    assert compute_generation_score(None) == 0.0
    assert compute_generation_score(None, summary=None) == 0.0
    assert compute_generation_score("content", source_type=None) >= 0.0
    assert generation_stability_bonus(None, 1.0) == 0.0
    assert generation_stability_bonus(0.5, None) == 0.0


# ══════════════════════════════════════════════════════════════════════
# 2. generation_stability_bonus 测试
# ══════════════════════════════════════════════════════════════════════

def test_ge6_strong_score_returns_large_bonus():
    """强分数（>= 0.7）返回接近 base × 0.35 的 bonus。"""
    bonus = generation_stability_bonus(generation_score=0.9, base_stability=1.0)
    assert bonus >= 0.30, f"GE6: 强分数应得大 bonus，got {bonus:.4f}"
    assert bonus <= 0.50, f"GE6: bonus 不超过 base × 0.50，got {bonus:.4f}"


def test_ge7_zero_score_returns_zero_bonus():
    """零分返回 0.0 bonus。"""
    bonus = generation_stability_bonus(generation_score=0.0, base_stability=1.0)
    assert bonus == 0.0, f"GE7: 零分应返回 0.0 bonus，got {bonus}"


def test_ge8_bonus_never_exceeds_half_base():
    """bonus 不超过 base_stability × 0.50。"""
    for base in [0.5, 1.0, 2.0]:
        for score in [0.0, 0.3, 0.5, 0.7, 1.0]:
            bonus = generation_stability_bonus(score, base)
            max_allowed = base * 0.50
            assert bonus <= max_allowed + 1e-9, (
                f"GE8: bonus={bonus:.4f} 超过 base({base})×0.50={max_allowed:.4f}, "
                f"score={score}"
            )


def test_ge9_monotonic_higher_score_higher_bonus():
    """单调性：分数越高，bonus 越大（或相等）。"""
    base = 1.0
    scores = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    bonuses = [generation_stability_bonus(s, base) for s in scores]
    for i in range(len(bonuses) - 1):
        assert bonuses[i] <= bonuses[i + 1] + 1e-9, (
            f"GE9: score 单调性违反：bonus[{scores[i]}]={bonuses[i]:.4f} > "
            f"bonus[{scores[i+1]}]={bonuses[i+1]:.4f}"
        )


# ══════════════════════════════════════════════════════════════════════
# 3. apply_generation_effect 测试
# ══════════════════════════════════════════════════════════════════════

def test_ge10_apply_writes_stability_to_db(conn):
    """apply_generation_effect 将更新后的 stability 写入 DB。"""
    now = _now()
    conn.execute(
        "INSERT INTO memory_chunks "
        "(id, project, chunk_type, content, summary, importance, retrievability, "
        "stability, created_at, updated_at) "
        "VALUES ('ge10', 'test', 'reasoning_chain', 'content', 'summary', "
        "0.7, 0.5, 1.0, ?, ?)",
        (now, now)
    )
    conn.commit()

    active_content = (
        "这意味着我们应该采用分片策略。"
        "我认为根本原因在于单节点内存上限。"
        "因此可以得出：水平扩展是必然选择。"
        "综上所述，这个架构决策是正确的。"
    )
    new_stability = apply_generation_effect(
        conn, "ge10", active_content, base_stability=1.0
    )
    conn.commit()

    row = conn.execute(
        "SELECT stability FROM memory_chunks WHERE id='ge10'"
    ).fetchone()
    assert row is not None
    assert row[0] > 1.0, f"GE10: 主动推理内容应增大 stability，got {row[0]}"
    assert new_stability > 1.0, f"GE10: 返回值 new_stability 应 > 1.0，got {new_stability}"


def test_ge11_passive_content_no_change(conn):
    """被动内容不改变 stability（bonus ≈ 0）。"""
    now = _now()
    conn.execute(
        "INSERT INTO memory_chunks "
        "(id, project, chunk_type, content, summary, importance, retrievability, "
        "stability, created_at, updated_at) "
        "VALUES ('ge11', 'test', 'decision', 'content', 'summary', "
        "0.7, 0.5, 1.0, ?, ?)",
        (now, now)
    )
    conn.commit()

    passive_content = "Redis is a key-value store. Default port: 6379."
    new_stability = apply_generation_effect(
        conn, "ge11", passive_content, base_stability=1.0
    )
    conn.commit()

    row = conn.execute(
        "SELECT stability FROM memory_chunks WHERE id='ge11'"
    ).fetchone()
    assert row[0] <= 1.05, f"GE11: 被动内容 stability 不应增加，got {row[0]}"


# ══════════════════════════════════════════════════════════════════════
# 4. insert_chunk 集成测试
# ══════════════════════════════════════════════════════════════════════

def test_ge12_insert_chunk_reasoning_gets_higher_stability(conn):
    """含推理语言的 chunk 通过 insert_chunk 得到更高 stability。"""
    passive = _make_chunk(
        "ge12_passive",
        content="Redis is a key-value store used for caching. Port is 6379.",
        summary="Redis 介绍",
        stability=1.0,
    )
    active = _make_chunk(
        "ge12_active",
        content=(
            "这意味着我们应该采用分片策略。"
            "我认为根本原因在于单节点内存上限。"
            "因此可以得出：水平扩展是必然选择。"
            "综上所述，这个架构决策是正确的。"
        ),
        summary="缓存架构决策",
        stability=1.0,
    )

    insert_chunk(conn, passive)
    insert_chunk(conn, active)
    conn.commit()

    passive_row = conn.execute(
        "SELECT stability FROM memory_chunks WHERE id='ge12_passive'"
    ).fetchone()
    active_row = conn.execute(
        "SELECT stability FROM memory_chunks WHERE id='ge12_active'"
    ).fetchone()

    assert passive_row is not None and active_row is not None
    active_stability = active_row[0]
    passive_stability = passive_row[0]

    assert active_stability >= passive_stability, (
        f"GE12: 推理 chunk stability({active_stability:.4f}) 应 >= "
        f"被动 chunk stability({passive_stability:.4f})"
    )


def test_ge_english_generation_markers():
    """英文推理标记也能被正确检测。"""
    content = (
        "I think the root cause is in the caching layer. "
        "This means we should redesign the eviction policy. "
        "Therefore, LFU is more appropriate here. "
        "In summary, the Write-Allocate strategy is the right choice. "
        "This suggests a fundamental rethinking of the architecture."
    )
    score = compute_generation_score(content)
    assert score >= 0.4, f"英文推理标记应得较高分，got {score:.3f}"
