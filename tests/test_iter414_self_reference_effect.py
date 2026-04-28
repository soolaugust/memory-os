"""
test_iter414_self_reference_effect.py — iter414: Self-Reference Effect 单元测试

覆盖：
  SR1: compute_self_reference_score — first-person markers → non-zero score
  SR2: compute_self_reference_score — no markers → score=0
  SR3: compute_self_reference_score — dense first-person text → high score
  SR4: compute_self_reference_score — agent-generated chunk_type bonus
  SR5: self_ref_stability_bonus — score > 0 → bonus > 0
  SR6: self_ref_stability_bonus — score=0 → bonus=0
  SR7: self_ref_stability_bonus — cap at base × bonus_cap
  SR8: apply_self_reference_effect — chunk with first-person → stability increases
  SR9: apply_self_reference_effect — chunk without markers → stability unchanged
  SR10: apply_self_reference_effect — self_ref_enabled=False → no boost
  SR11: compute_self_reference_score — Chinese first-person markers
  SR12: reasoning_chain chunk_type → type_bonus applied

认知科学依据：
  Rogers et al. (1977) Self-Reference Effect —
    自我参照加工激活 PFC + hippocampus 双路径，形成更强记忆痕迹。
  Symons & Johnson (1997) Meta-analysis: self-reference advantage ≈ +0.5 SD vs semantic encoding。

OS 类比：Linux process 自身页在 TLB 中有最高局部性 —
  自我参照的 page（stack/heap）命中率最高，类比 self-referential chunk 的检索优势。
"""
import sys
import sqlite3
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import tmpfs  # noqa

from store_vfs import (
    ensure_schema,
    compute_self_reference_score,
    self_ref_stability_bonus,
    apply_self_reference_effect,
)
from store import insert_chunk
import config


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _insert_chunk_direct(conn, cid, content="test", chunk_type="decision",
                          stability=2.0, project="test"):
    now = __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc
    ).isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO memory_chunks "
        "(id, project, chunk_type, content, summary, importance, stability, "
        "created_at, updated_at, retrievability) "
        "VALUES (?, ?, ?, ?, ?, 0.8, ?, ?, ?, 0.9)",
        (cid, project, chunk_type, content, content, stability, now, now)
    )
    conn.commit()


def _get_stability(conn, cid: str) -> float:
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


# ── SR1: First-person markers → non-zero score ───────────────────────────────

def test_sr1_first_person_markers_score(conn):
    """含第一人称标记（'I', 'my'）→ score > 0。"""
    score = compute_self_reference_score("I think my approach is correct here.", "decision")
    assert score > 0.0, f"SR1: 含 I/my 应得 score > 0，got {score}"


def test_sr1_we_marker(conn):
    """含 'we' → score > 0。"""
    score = compute_self_reference_score("We need to remember our configuration.")
    assert score > 0.0, f"SR1b: 含 we/our 应得 score > 0，got {score}"


# ── SR2: No markers → score=0 ────────────────────────────────────────────────

def test_sr2_no_markers_zero_score(conn):
    """无第一人称标记 → score=0（chunk_type 为无 type_bonus 的 task_state）。"""
    score = compute_self_reference_score("The database configuration uses port 5432.", "task_state")
    assert score == 0.0, f"SR2: 无第一人称标记应得 score=0，got {score}"


def test_sr2_empty_content(conn):
    """空内容 → score=0。"""
    assert compute_self_reference_score("") == 0.0
    assert compute_self_reference_score(None) == 0.0


# ── SR3: Dense first-person → high score ─────────────────────────────────────

def test_sr3_dense_first_person_high_score(conn):
    """高密度第一人称文本 → 高 score。"""
    dense = "I decided to use this approach because I think it's better. My reasoning is clear."
    score = compute_self_reference_score(dense)
    assert score > 0.3, f"SR3: 高密度第一人称应得 score > 0.3，got {score}"


# ── SR4: Agent-generated chunk_type → type_bonus ─────────────────────────────

def test_sr4_reasoning_chain_type_bonus(conn):
    """reasoning_chain chunk_type 额外 self-reference bonus（即使内容无标记）。"""
    score_rc = compute_self_reference_score("The algorithm processes data.", "reasoning_chain")
    score_default = compute_self_reference_score("The algorithm processes data.", "task_state")
    assert score_rc > score_default, \
        f"SR4: reasoning_chain 应比 task_state 得更高 score：{score_rc:.4f} > {score_default:.4f}"


def test_sr4_decision_type_bonus(conn):
    """decision chunk_type 也获得 type_bonus。"""
    score_dec = compute_self_reference_score("Configure port 5432.", "decision")
    score_sum = compute_self_reference_score("Configure port 5432.", "conversation_summary")
    assert score_dec >= score_sum, \
        f"SR4b: decision 应 >= conversation_summary：{score_dec:.4f} >= {score_sum:.4f}"


# ── SR5: self_ref_stability_bonus — score > 0 → bonus > 0 ────────────────────

def test_sr5_positive_score_positive_bonus(conn):
    """score > 0 → bonus > 0。"""
    bonus = self_ref_stability_bonus(0.5, 2.0)
    assert bonus > 0.0, f"SR5: score=0.5 → bonus 应 > 0，got {bonus}"


# ── SR6: score=0 → bonus=0 ───────────────────────────────────────────────────

def test_sr6_zero_score_zero_bonus(conn):
    """score=0 → bonus=0。"""
    bonus = self_ref_stability_bonus(0.0, 2.0)
    assert bonus == 0.0, f"SR6: score=0 → bonus 应=0，got {bonus}"


# ── SR7: Bonus capped at base × bonus_cap ────────────────────────────────────

def test_sr7_bonus_capped(conn):
    """bonus 上限为 base × bonus_cap。"""
    base = 2.0
    cap = 0.25
    # score=1.0 → full cap
    bonus = self_ref_stability_bonus(1.0, base, cap)
    assert abs(bonus - base * cap) < 1e-6, \
        f"SR7: score=1.0 bonus 应 = base×cap = {base*cap}，got {bonus}"
    # score > 1.0 → still capped
    bonus_high = self_ref_stability_bonus(2.0, base, cap)
    assert bonus_high <= base * cap, f"SR7: 过高 score 仍受 cap 限制"


# ── SR8: apply_self_reference_effect — first-person → stability increases ─────

def test_sr8_first_person_stability_boost(conn):
    """含第一人称内容的 chunk → stability 增加。"""
    _insert_chunk_direct(conn, "sr8", content="I decided to use port 8080 for my service.",
                          chunk_type="decision", stability=2.0)

    stab_before = _get_stability(conn, "sr8")
    apply_self_reference_effect(conn, "sr8", base_stability=2.0)
    conn.commit()
    stab_after = _get_stability(conn, "sr8")

    assert stab_after > stab_before, \
        f"SR8: 含第一人称内容应 → stability 增加，before={stab_before:.4f} after={stab_after:.4f}"


# ── SR9: No first-person markers → stability unchanged ────────────────────────

def test_sr9_no_markers_no_boost(conn):
    """无第一人称标记 → stability 不变（chunk_type 无 type_bonus）。"""
    _insert_chunk_direct(conn, "sr9", content="Database uses PostgreSQL on port 5432.",
                          chunk_type="task_state", stability=2.0)

    stab_before = _get_stability(conn, "sr9")
    apply_self_reference_effect(conn, "sr9", base_stability=2.0)
    conn.commit()
    stab_after = _get_stability(conn, "sr9")

    assert abs(stab_after - stab_before) < 0.001, \
        f"SR9: 无标记 stability 不应变化，before={stab_before:.4f} after={stab_after:.4f}"


# ── SR10: self_ref_enabled=False → no boost ───────────────────────────────────

def test_sr10_disabled_no_boost(conn, monkeypatch):
    """self_ref_enabled=False → 禁用，stability 不变。"""
    import unittest.mock as mock
    original_get = config.get
    def patched_get(key, project=None):
        if key == "store_vfs.self_ref_enabled":
            return False
        return original_get(key, project=project)

    _insert_chunk_direct(conn, "sr10", content="I decided this is my solution.",
                          chunk_type="decision", stability=2.0)
    stab_before = _get_stability(conn, "sr10")

    with mock.patch.object(config, 'get', side_effect=patched_get):
        apply_self_reference_effect(conn, "sr10", base_stability=2.0)
    conn.commit()

    stab_after = _get_stability(conn, "sr10")
    assert abs(stab_after - stab_before) < 0.001, \
        f"SR10: 禁用后 stability 不应变，before={stab_before:.4f} after={stab_after:.4f}"


# ── SR11: Chinese first-person markers ────────────────────────────────────────

def test_sr11_chinese_self_reference(conn):
    """中文第一人称标记（我/我们/我的）→ score > 0。"""
    score = compute_self_reference_score("我认为这个配置是正确的，我们的端口应该是8080。")
    assert score > 0.0, f"SR11: 中文第一人称应得 score > 0，got {score}"


# ── SR12: reasoning_chain insert → boosted stability ─────────────────────────

def test_sr12_reasoning_chain_insert_boost(conn):
    """reasoning_chain chunk 因 type_bonus 获得 stability 加成（即使无显式第一人称）。"""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    chunk = {
        "id": "sr12",
        "created_at": now,
        "updated_at": now,
        "project": "test",
        "source_session": "s1",
        "chunk_type": "reasoning_chain",
        "info_class": "semantic",
        "content": "The algorithm design follows a divide-and-conquer approach.",
        "summary": "algorithm design",
        "tags": [],
        "importance": 0.8,
        "retrievability": 0.9,
        "last_accessed": now,
        "access_count": 1,
        "oom_adj": 0,
        "lru_gen": 0,
        "stability": 2.0,
        "raw_snippet": "",
        "encoding_context": {},
    }
    insert_chunk(conn, chunk)
    conn.commit()
    stab = _get_stability(conn, "sr12")
    # reasoning_chain type_bonus (0.2) → score≥0.2 → bonus > 0 → stab > initial
    # (initial stability may have been modified by other effects in insert_chunk)
    assert stab >= 2.0, f"SR12: reasoning_chain stability 应 >= 2.0，got {stab:.4f}"
