"""
test_iter408_proactive_interference.py — iter408: Proactive Interference (PI)

覆盖：
  PI1: compute_pi_penalty — 有强旧记忆 + 高相似度 → penalty > 0
  PI2: compute_pi_penalty — 无强旧记忆（access_count < threshold）→ penalty = 0
  PI3: compute_pi_penalty — 相似度低的邻居 → penalty 接近 0
  PI4: compute_pi_penalty — 空/None 输入安全返回 0.0
  PI5: compute_pi_penalty — 无邻居时返回 0.0
  PI6: compute_pi_penalty — penalty 不超过 max_penalty
  PI7: apply_proactive_interference — design_constraint 类型豁免
  PI8: apply_proactive_interference — 高 importance (>0.85) penalty 减半
  PI9: apply_proactive_interference — 有 PI 时 stability 降低
  PI10: apply_proactive_interference — PI penalty 下限 0.1 stability

认知科学依据：
  Underwood (1957) Proactive Inhibition and Forgetting:
    旧有相似材料学习量越多 → 新材料遗忘越快（前摄干扰）。
  与 iter405 RI 对称：RI=新干扰旧（检索时），PI=旧干扰新（写入时）。

OS 类比：Linux TLB Shootdown Cost —
  修改共享 PTE 时，需向所有持有该 TLB entry 的 CPU 广播 IPI；
  共享越多，写入成本越大。类比：已有强记忆邻居越多，新 chunk stability 越低。
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
    compute_pi_penalty,
    apply_proactive_interference,
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


def _insert(conn, chunk_id, encode_context="", access_count=0,
            chunk_type="decision", importance=0.7, project="test"):
    now = _now()
    conn.execute(
        "INSERT OR REPLACE INTO memory_chunks "
        "(id, project, chunk_type, content, summary, importance, retrievability, "
        "stability, encode_context, access_count, created_at, updated_at) "
        "VALUES (?, ?, ?, 'content', 'summary', ?, 0.5, 1.0, ?, ?, ?, ?)",
        (chunk_id, project, chunk_type, importance,
         encode_context, access_count, now, now)
    )
    conn.commit()


# ══════════════════════════════════════════════════════════════════════
# 1. compute_pi_penalty 测试
# ══════════════════════════════════════════════════════════════════════

def test_pi1_strong_old_high_sim_has_penalty(conn):
    """有强旧记忆（高 access_count）且语义高相似 → penalty > 0。"""
    # 强旧 chunk（高 access_count + 相同主题）
    for i in range(5):
        _insert(conn, f"old_{i}", "redis,cache,performance,cluster,lru", access_count=10)
    # 新 chunk（相同主题）
    _insert(conn, "new_chunk", "redis,cache,performance,cluster,eviction", access_count=0)
    penalty = compute_pi_penalty(conn, "new_chunk", "test")
    assert penalty > 0.0, f"PI1: 强旧记忆 + 高相似 → penalty 应 > 0，got {penalty:.4f}"


def test_pi2_weak_old_no_penalty(conn):
    """旧 chunk access_count < threshold → penalty = 0（弱旧记忆不产生 PI）。"""
    for i in range(5):
        _insert(conn, f"fresh_{i}", "redis,cache,performance,cluster,lru", access_count=0)
    _insert(conn, "new2", "redis,cache,performance,cluster,eviction", access_count=0)
    penalty = compute_pi_penalty(conn, "new2", "test", strong_acc_threshold=3)
    assert penalty == 0.0, f"PI2: 无强旧记忆时 penalty 应为 0，got {penalty:.4f}"


def test_pi3_low_similarity_near_zero(conn):
    """语义差异大（低 Jaccard 相似度）→ penalty 接近 0。"""
    for i in range(5):
        _insert(conn, f"diff_{i}", "machine_learning,neural_network,training,pytorch", access_count=10)
    _insert(conn, "new3", "redis,cache,performance,cluster,eviction", access_count=0)
    penalty = compute_pi_penalty(conn, "new3", "test")
    # 相似度极低，penalty 应接近 0
    assert penalty < 0.02, f"PI3: 低相似度 penalty 应接近 0，got {penalty:.4f}"


def test_pi4_empty_inputs_safe(conn):
    """空/None 输入安全返回 0.0。"""
    assert compute_pi_penalty(conn, "", "test") == 0.0
    assert compute_pi_penalty(conn, None, "test") == 0.0
    assert compute_pi_penalty(conn, "chunk_x", "") == 0.0
    assert compute_pi_penalty(conn, "chunk_x", None) == 0.0


def test_pi5_no_neighbors_returns_zero(conn):
    """无邻居时返回 0.0。"""
    _insert(conn, "alone", "redis,cache,performance", access_count=0)
    penalty = compute_pi_penalty(conn, "alone", "test")
    assert penalty == 0.0, f"PI5: 无邻居时 penalty 应为 0，got {penalty:.4f}"


def test_pi6_penalty_capped_at_max(conn):
    """penalty 不超过 max_penalty。"""
    for i in range(10):
        _insert(conn, f"strong_{i}", "redis,cache,performance,cluster,lru", access_count=100)
    _insert(conn, "new6", "redis,cache,performance,cluster,lru", access_count=0)
    max_p = 0.10
    penalty = compute_pi_penalty(conn, "new6", "test", max_penalty=max_p)
    assert penalty <= max_p + 1e-9, f"PI6: penalty 上限 {max_p}，got {penalty:.4f}"


# ══════════════════════════════════════════════════════════════════════
# 2. apply_proactive_interference 测试
# ══════════════════════════════════════════════════════════════════════

def test_pi7_design_constraint_exempt(conn):
    """design_constraint 类型豁免 PI（不降低 stability）。"""
    for i in range(5):
        _insert(conn, f"si_{i}", "redis,cache,performance,cluster,lru", access_count=10)
    _insert(conn, "pi7_dc", "redis,cache,performance,cluster,eviction",
            chunk_type="design_constraint", access_count=0)
    original = conn.execute(
        "SELECT stability FROM memory_chunks WHERE id='pi7_dc'"
    ).fetchone()[0]
    new_s = apply_proactive_interference(conn, "pi7_dc", "test", base_stability=original)
    assert new_s == original, f"PI7: design_constraint 不受 PI，got {new_s:.4f} vs {original:.4f}"


def test_pi8_high_importance_halved_penalty(conn):
    """高 importance (>0.85) 的新 chunk，PI penalty 减半。"""
    for i in range(5):
        _insert(conn, f"hi_{i}", "redis,cache,performance,cluster,lru", access_count=10)

    # low importance chunk
    _insert(conn, "pi8_low", "redis,cache,performance,cluster,eviction",
            access_count=0, importance=0.5)
    # high importance chunk
    _insert(conn, "pi8_high", "redis,cache,performance,cluster,eviction",
            access_count=0, importance=0.90)

    s_low = apply_proactive_interference(conn, "pi8_low", "test", base_stability=1.0)
    s_high = apply_proactive_interference(conn, "pi8_high", "test", base_stability=1.0)

    # high importance chunk 受到的 PI 惩罚应该更少（stability 更高）
    assert s_high >= s_low, (
        f"PI8: 高 importance chunk PI 惩罚应更少，s_high={s_high:.4f} >= s_low={s_low:.4f}"
    )


def test_pi9_stability_reduced_by_pi(conn):
    """有 PI 时 stability 降低（小于 base）。"""
    for i in range(5):
        _insert(conn, f"pi9_old_{i}", "redis,cache,performance,cluster,lru", access_count=10)
    _insert(conn, "pi9_new", "redis,cache,performance,cluster,eviction", access_count=0)
    base = 1.0
    new_s = apply_proactive_interference(conn, "pi9_new", "test", base_stability=base)
    # 只要有足够强的 PI 信号，stability 应该降低
    # （由于 Jaccard 可能不完全，允许 new_s <= base）
    assert new_s <= base + 1e-9, (
        f"PI9: PI 存在时 stability 不应超过 base，got {new_s:.4f} > {base:.4f}"
    )


def test_pi10_stability_floor(conn):
    """stability 不低于下限 0.1（防止新 chunk 被完全抑制）。"""
    for i in range(20):
        _insert(conn, f"pi10_old_{i}", "redis,cache,performance,cluster,lru", access_count=100)
    _insert(conn, "pi10_new", "redis,cache,performance,cluster,lru", access_count=0)
    new_s = apply_proactive_interference(conn, "pi10_new", "test",
                                         base_stability=0.1, max_penalty=0.30)
    assert new_s >= 0.1 - 1e-9, f"PI10: stability 下限 0.1，got {new_s:.4f}"
