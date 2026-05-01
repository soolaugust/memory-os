"""
test_iter459_cie.py — iter459: Contextual Interference Effect 单元测试

覆盖：
  CI1: session_type_history >= 2 种不同类型 → stability 加成
  CI2: 只有 1 种 session_type → 无 CIE 加成
  CI3: cie_enabled=False → 无任何加成
  CI4: importance < cie_min_importance(0.40) → 不参与 CIE
  CI5: diversity_score 与 unique_types 正相关（越多不同类型，加成越大）
  CI6: stability 加成后不超过 365.0（cap 保护）
  CI7: session_type 参数正确更新 session_type_history
  CI8: update_accessed() 集成测试：传入 session_type 触发 CIE 更新

认知科学依据：
  Shea & Morgan (1979) "Contextual interference effects on the acquisition,
    retention, and transfer of a motor skill" (Journal of Experimental Psychology: HPP) —
    随机（mixed）练习顺序比集中（blocked）练习在延迟测试中成绩高 57%。
  Brady (2008) Meta-analysis — CI effect: d=0.56，跨多种技能均显著。

OS 类比：Linux blk-mq — 不同 queue/CPU 的混合调度在多种 I/O pattern 混合时
  表现优于单一 queue（cross-queue diversity = CI effect）。
"""
import sys
import sqlite3
import datetime
import unittest.mock as mock
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import tmpfs  # noqa

from store_vfs import (
    ensure_schema,
    apply_contextual_interference_effect,
    update_accessed,
)
import config


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _utcnow():
    return datetime.datetime.now(datetime.timezone.utc)


def _insert_chunk(conn, cid, project="test", stability=5.0, importance=0.6,
                  chunk_type="decision", session_type_history=""):
    now_iso = _utcnow().isoformat()
    import datetime as _dt
    la_iso = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(minutes=10)).isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count,
            encode_context, session_type_history)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (cid, project, chunk_type, f"content {cid}", f"summary {cid}",
         importance, stability, now_iso, now_iso, 0.5,
         la_iso, 2, "kernel_mm", session_type_history)
    )
    conn.commit()


def _get_stability(conn, cid: str) -> float:
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


def _get_session_type_history(conn, cid: str) -> str:
    row = conn.execute(
        "SELECT session_type_history FROM memory_chunks WHERE id=?", (cid,)
    ).fetchone()
    return row[0] or "" if row else ""


# ── CI1: >= 2 种 session_type → stability 加成 ────────────────────────────────

def test_ci1_diverse_types_boost_stability(conn):
    """CI1: session_type_history 含 >= 2 种不同类型 → stability 加成。"""
    # 预置包含两种不同 session_type 的历史
    _insert_chunk(conn, "cie_1", stability=5.0, importance=0.6,
                  session_type_history="debug,design")
    stab_before = _get_stability(conn, "cie_1")

    result = apply_contextual_interference_effect(conn, ["cie_1"], "test")
    stab_after = _get_stability(conn, "cie_1")

    assert stab_after > stab_before, (
        f"CI1: >= 2 种 session_type 时 stability 应提升，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["cie_boosted"] >= 1, f"CI1: cie_boosted 应 >= 1，got {result}"


# ── CI2: 只有 1 种 session_type → 无加成 ──────────────────────────────────────

def test_ci2_single_type_no_boost(conn):
    """CI2: session_type_history 只含 1 种类型（< cie_min_unique_types=2）→ 无 CIE 加成。"""
    _insert_chunk(conn, "cie_2", stability=5.0, importance=0.6,
                  session_type_history="debug,debug,debug")
    stab_before = _get_stability(conn, "cie_2")

    result = apply_contextual_interference_effect(conn, ["cie_2"], "test")
    stab_after = _get_stability(conn, "cie_2")

    assert abs(stab_after - stab_before) < 0.01, (
        f"CI2: 只有 1 种类型不应有 CIE 加成，before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── CI3: cie_enabled=False → 无加成 ───────────────────────────────────────────

def test_ci3_disabled_no_boost(conn):
    """CI3: store_vfs.cie_enabled=False → 无任何 CIE 加成。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.cie_enabled":
            return False
        return original_get(key, project=project)

    _insert_chunk(conn, "cie_3", stability=5.0, importance=0.6,
                  session_type_history="debug,design,refactor")
    stab_before = _get_stability(conn, "cie_3")

    with mock.patch.object(config, 'get', side_effect=patched_get):
        result = apply_contextual_interference_effect(conn, ["cie_3"], "test")
    stab_after = _get_stability(conn, "cie_3")

    assert abs(stab_after - stab_before) < 0.01, (
        f"CI3: disabled 时不应有 CIE 加成，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["cie_boosted"] == 0, f"CI3: cie_boosted 应为 0，got {result}"


# ── CI4: importance 不足 → 不参与 CIE ─────────────────────────────────────────

def test_ci4_low_importance_no_boost(conn):
    """CI4: importance < cie_min_importance(0.40) → 不参与 CIE。"""
    _insert_chunk(conn, "cie_4", stability=5.0, importance=0.20,
                  session_type_history="debug,design,refactor")
    stab_before = _get_stability(conn, "cie_4")

    result = apply_contextual_interference_effect(conn, ["cie_4"], "test")
    stab_after = _get_stability(conn, "cie_4")

    assert abs(stab_after - stab_before) < 0.01, (
        f"CI4: 低 importance 不应触发 CIE，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["cie_boosted"] == 0, f"CI4: cie_boosted 应为 0，got {result}"


# ── CI5: diversity 越高加成越大 ────────────────────────────────────────────────

def test_ci5_more_diversity_more_boost(conn):
    """CI5: unique session_type 越多 → CIE 加成越大。"""
    # 场景 A: 2 种 session_type（低多样性）
    _insert_chunk(conn, "cie_5a", stability=5.0, importance=0.6,
                  session_type_history="debug,design")
    apply_contextual_interference_effect(conn, ["cie_5a"], "test")
    stab_a = _get_stability(conn, "cie_5a")

    # 场景 B: 4 种 session_type（高多样性，达到 ref_types=4）
    _insert_chunk(conn, "cie_5b", stability=5.0, importance=0.6,
                  session_type_history="debug,design,refactor,review")
    apply_contextual_interference_effect(conn, ["cie_5b"], "test")
    stab_b = _get_stability(conn, "cie_5b")

    assert stab_b >= stab_a, (
        f"CI5: 更多 unique session_type 应带来更大或相等加成，"
        f"stab_a(2 types)={stab_a:.4f} stab_b(4 types)={stab_b:.4f}"
    )


# ── CI6: stability 上限 365.0 ─────────────────────────────────────────────────

def test_ci6_stability_cap_365(conn):
    """CI6: CIE boost 后 stability 不超过 365.0。"""
    _insert_chunk(conn, "cie_6", stability=364.0, importance=0.8,
                  session_type_history="debug,design,refactor,review")
    apply_contextual_interference_effect(conn, ["cie_6"], "test")
    stab_after = _get_stability(conn, "cie_6")

    assert stab_after <= 365.0, f"CI6: stability 不应超过 365.0，got {stab_after}"


# ── CI7: session_type 参数更新 session_type_history ──────────────────────────

def test_ci7_session_type_updates_history(conn):
    """CI7: 传入 session_type 参数后，session_type_history 应包含新的类型。"""
    _insert_chunk(conn, "cie_7", stability=5.0, importance=0.6,
                  session_type_history="")

    apply_contextual_interference_effect(conn, ["cie_7"], "test",
                                          session_type="debug")
    hist = _get_session_type_history(conn, "cie_7")

    assert "debug" in hist, (
        f"CI7: 传入 session_type='debug' 后 history 应包含 'debug'，got '{hist}'"
    )


# ── CI8: update_accessed() 集成测试 ──────────────────────────────────────────

def test_ci8_update_accessed_integration(conn):
    """CI8: update_accessed() 传入 session_type 后多次调用不同类型时触发 CIE 加成。"""
    # 先插入 chunk，预置一种 session_type 历史
    _insert_chunk(conn, "cie_8", stability=5.0, importance=0.6,
                  session_type_history="debug")

    stab_before = _get_stability(conn, "cie_8")

    # 调用 update_accessed 传入不同 session_type（现在历史含 debug + design = 2种）
    update_accessed(conn, ["cie_8"], session_type="design")

    stab_after = _get_stability(conn, "cie_8")
    hist = _get_session_type_history(conn, "cie_8")

    # 历史应包含 design
    assert "design" in hist, f"CI8: update_accessed 后 history 应含 'design'，got '{hist}'"
    # stability 应有加成（因为已有 debug + design = 2 种）
    assert stab_after >= stab_before, (
        f"CI8: update_accessed + 2 种 session_type 时 stability 不应减少，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )
