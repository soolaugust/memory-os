"""
test_iter416_zeigarnik_effect.py — iter416: Zeigarnik Effect 单元测试

覆盖：
  ZE1: compute_zeigarnik_score — TODO marker → score > 0
  ZE2: compute_zeigarnik_score — no markers → score=0
  ZE3: compute_zeigarnik_score — multiple markers → higher score
  ZE4: compute_zeigarnik_score — task_state type_bonus
  ZE5: compute_zeigarnik_score — Chinese markers (待/未完成/待处理)
  ZE6: zeigarnik_stability_bonus — score > 0 → bonus > 0
  ZE7: zeigarnik_stability_bonus — score=0 → bonus=0
  ZE8: zeigarnik_stability_bonus — capped at base × bonus_cap
  ZE9: apply_zeigarnik_effect — TODO chunk → stability increases
  ZE10: apply_zeigarnik_effect — no markers → stability unchanged
  ZE11: apply_zeigarnik_effect — zeigarnik_enabled=False → no boost
  ZE12: insert_chunk with TODO content → stability boosted at write time
  ZE13: FIXME/WIP/PENDING markers → score > 0
  ZE14: task_state chunk_type + no markers → score > 0 (type_bonus)

认知科学依据：
  Zeigarnik (1927) — 未完成任务 recall superiority ≈ +90% vs completed tasks。
  Lewin (1935) Tension System Theory — 未完成任务维持认知系统"张力"，保持记忆激活。
  Ovsiankina (1928) — 被中断的任务在有机会时自发恢复（resumption tendency）。

OS 类比：Linux futex waitqueue —
  pending I/O 请求保留在内核等待队列，不被 swapd 驱逐；
  未完成写入的 dirty page 被 writeback 守护进程跟踪。
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
    compute_zeigarnik_score,
    zeigarnik_stability_bonus,
    apply_zeigarnik_effect,
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
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
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


# ── ZE1: TODO marker → score > 0 ────────────────────────────────────────────

def test_ze1_todo_marker_score(conn):
    """含 TODO → score > 0。"""
    score = compute_zeigarnik_score("TODO: refactor this function to use the new API.")
    assert score > 0.0, f"ZE1: TODO 应得 score > 0，got {score}"


def test_ze1_fixme_marker(conn):
    """含 FIXME → score > 0。"""
    score = compute_zeigarnik_score("FIXME: this logic is broken when port is 0.")
    assert score > 0.0, f"ZE1b: FIXME 应得 score > 0，got {score}"


# ── ZE2: No markers → score=0 ──────────────────────────────────────────────

def test_ze2_no_markers_zero_score(conn):
    """无未完成标记（且非 task_state 类型）→ score=0。"""
    score = compute_zeigarnik_score(
        "The database uses PostgreSQL on port 5432.", "decision"
    )
    assert score == 0.0, f"ZE2: 无标记应得 score=0，got {score}"


def test_ze2_empty_content(conn):
    """空内容 → score=0。"""
    assert compute_zeigarnik_score("") == 0.0
    assert compute_zeigarnik_score(None) == 0.0


# ── ZE3: Multiple markers → higher score ────────────────────────────────────

def test_ze3_multiple_markers_higher_score(conn):
    """多个标记 → score 更高。"""
    single = compute_zeigarnik_score("TODO: investigate this issue.")
    multi = compute_zeigarnik_score("TODO: investigate. FIXME: this is pending. WIP.")
    assert multi >= single, f"ZE3: 多标记应得更高 score：{multi:.4f} >= {single:.4f}"


# ── ZE4: task_state type_bonus ──────────────────────────────────────────────

def test_ze4_task_state_type_bonus(conn):
    """task_state chunk_type 获得 type_bonus（即使内容无标记）。"""
    score_ts = compute_zeigarnik_score("Port configuration is 8080.", "task_state")
    score_dec = compute_zeigarnik_score("Port configuration is 8080.", "decision")
    assert score_ts > score_dec, \
        f"ZE4: task_state 应比 decision 得更高 score：{score_ts:.4f} > {score_dec:.4f}"


def test_ze4_task_state_no_markers_positive(conn):
    """task_state + 无标记 → score > 0（纯 type_bonus）。"""
    score = compute_zeigarnik_score("Processing authentication step.", "task_state")
    assert score > 0.0, f"ZE4b: task_state 应得 score > 0（type_bonus），got {score}"


# ── ZE5: Chinese markers ─────────────────────────────────────────────────────

def test_ze5_chinese_pending_markers(conn):
    """中文未完成标记（待/未完成/待处理）→ score > 0。"""
    score = compute_zeigarnik_score("待确认：这个配置是否正确，需要调查端口冲突。")
    assert score > 0.0, f"ZE5: 中文待确认应得 score > 0，got {score}"


def test_ze5_chinese_wip(conn):
    """中文 未完成/待跟进 → score > 0。"""
    score = compute_zeigarnik_score("未完成：数据库迁移还需要验证。后续跟进。")
    assert score > 0.0, f"ZE5b: 中文未完成/跟进应得 score > 0，got {score}"


# ── ZE6: zeigarnik_stability_bonus — score > 0 → bonus > 0 ──────────────────

def test_ze6_positive_score_positive_bonus(conn):
    """score=0.5 → bonus > 0。"""
    bonus = zeigarnik_stability_bonus(0.5, 2.0)
    assert bonus > 0.0, f"ZE6: score=0.5 → bonus 应>0，got {bonus}"


# ── ZE7: score=0 → bonus=0 ───────────────────────────────────────────────────

def test_ze7_zero_score_zero_bonus(conn):
    """score=0 → bonus=0。"""
    bonus = zeigarnik_stability_bonus(0.0, 2.0)
    assert bonus == 0.0, f"ZE7: score=0 → bonus=0，got {bonus}"


# ── ZE8: Bonus capped at base × bonus_cap ────────────────────────────────────

def test_ze8_bonus_capped(conn):
    """bonus 上限为 base × bonus_cap。"""
    base = 2.0
    cap = 0.20
    # score=1.0 → full cap
    bonus = zeigarnik_stability_bonus(1.0, base, cap)
    assert abs(bonus - base * cap) < 1e-6, \
        f"ZE8: score=1.0 → bonus = base×cap = {base*cap}，got {bonus}"
    # score > 1.0 → still capped
    bonus_high = zeigarnik_stability_bonus(2.0, base, cap)
    assert bonus_high <= base * cap, "ZE8: 过高 score 仍受 cap 限制"


# ── ZE9: apply_zeigarnik_effect — TODO chunk → stability increases ────────────

def test_ze9_todo_chunk_stability_boost(conn):
    """含 TODO 的 chunk → stability 增加。"""
    _insert_chunk_direct(
        conn, "ze9",
        content="TODO: investigate memory leak in the cache layer.",
        chunk_type="task_state", stability=2.0
    )
    stab_before = _get_stability(conn, "ze9")
    apply_zeigarnik_effect(conn, "ze9", base_stability=2.0)
    conn.commit()
    stab_after = _get_stability(conn, "ze9")
    assert stab_after > stab_before, \
        f"ZE9: TODO chunk → stability 应增加，before={stab_before:.4f} after={stab_after:.4f}"


# ── ZE10: No markers → stability unchanged ───────────────────────────────────

def test_ze10_no_markers_no_boost(conn):
    """无未完成标记（非 task_state 类型）→ stability 不变。"""
    _insert_chunk_direct(
        conn, "ze10",
        content="The service runs on port 8080 with TLS enabled.",
        chunk_type="design_constraint", stability=2.0
    )
    stab_before = _get_stability(conn, "ze10")
    apply_zeigarnik_effect(conn, "ze10", base_stability=2.0)
    conn.commit()
    stab_after = _get_stability(conn, "ze10")
    assert abs(stab_after - stab_before) < 0.001, \
        f"ZE10: 无标记 stability 不应变化，before={stab_before:.4f} after={stab_after:.4f}"


# ── ZE11: zeigarnik_enabled=False → no boost ─────────────────────────────────

def test_ze11_disabled_no_boost(conn, monkeypatch):
    """zeigarnik_enabled=False → 禁用，stability 不变。"""
    import unittest.mock as mock
    original_get = config.get
    def patched_get(key, project=None):
        if key == "store_vfs.zeigarnik_enabled":
            return False
        return original_get(key, project=project)

    _insert_chunk_direct(
        conn, "ze11",
        content="TODO: fix this critical bug in authentication flow.",
        chunk_type="task_state", stability=2.0
    )
    stab_before = _get_stability(conn, "ze11")
    with mock.patch.object(config, 'get', side_effect=patched_get):
        apply_zeigarnik_effect(conn, "ze11", base_stability=2.0)
    conn.commit()
    stab_after = _get_stability(conn, "ze11")
    assert abs(stab_after - stab_before) < 0.001, \
        f"ZE11: 禁用后 stability 不应变，before={stab_before:.4f} after={stab_after:.4f}"


# ── ZE12: insert_chunk with TODO → stability boosted at write time ────────────

def test_ze12_insert_chunk_todo_boosted(conn):
    """insert_chunk 写入含 TODO 的 task_state chunk → stability 在写入时被加成。"""
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    chunk = {
        "id": "ze12",
        "created_at": now, "updated_at": now, "project": "test",
        "source_session": "s1", "chunk_type": "task_state",
        "info_class": "semantic",
        "content": "TODO: complete the database migration script.",
        "summary": "pending migration task",
        "tags": [], "importance": 0.8, "retrievability": 0.9,
        "last_accessed": now, "access_count": 1,
        "oom_adj": 0, "lru_gen": 0, "stability": 2.0,
        "raw_snippet": "", "encoding_context": {},
    }
    insert_chunk(conn, chunk)
    conn.commit()

    stab = _get_stability(conn, "ze12")
    assert stab >= 2.0, f"ZE12: TODO task_state stability 应 >= 2.0，got {stab:.4f}"


# ── ZE13: FIXME/WIP/PENDING markers → score > 0 ──────────────────────────────

def test_ze13_various_markers_score(conn):
    """FIXME、WIP、pending markers 各自 → score > 0。"""
    for marker in ["FIXME", "WIP", "pending", "unresolved", "tbd", "revisit", "blocked on"]:
        score = compute_zeigarnik_score(f"This issue is {marker}: need to resolve.")
        assert score > 0.0, f"ZE13: '{marker}' 应得 score > 0，got {score}"


# ── ZE14: task_state + content markers → amplified score ─────────────────────

def test_ze14_task_state_with_content_markers(conn):
    """task_state + TODO content → score 比 task_state-only 更高。"""
    score_type_only = compute_zeigarnik_score("Port is 8080.", "task_state")
    score_both = compute_zeigarnik_score("TODO: Port config pending validation.", "task_state")
    assert score_both >= score_type_only, \
        f"ZE14: type+content 应 >= type_only：{score_both:.4f} >= {score_type_only:.4f}"
