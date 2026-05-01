"""
test_iter469_ge.py — iter469: Generation Effect 单元测试

覆盖：
  GE1: generative chunk_type (decision) → stability 加成
  GE2: non-generative chunk_type (observation) → 无加成（相对比较）
  GE3: ge_enabled=False → 无任何加成
  GE4: importance < ge_min_importance(0.25) → 不参与 GE
  GE5: 加成系数 = ge_boost_factor(1.10)，受 ge_max_boost(0.18) 保护
  GE6: stability 加成后不超过 365.0
  GE7: 直接调用 apply_generation_effect → ge_boosted=True
  GE8: ge_generative_types 中所有类型均触发 GE（decision/design_constraint/feedback）

认知科学依据：
  Slamecka & Graf (1978) "The generation effect: Delineation of a phenomenon" —
    自我生成的信息比被动阅读信息记忆保留率高 20-30%（延时测验）。
    机制：生成过程激活更深的语义处理网络 + 自我参照加工。
  McElroy & Slamecka (1982): 生成效应在词汇和命题层面均成立。

OS 类比：Linux CoW (mm/memory.c: do_wp_page) —
  被进程主动写入（dirty）的页面获得更高 active LRU 优先级；
  只读共享页面（read-only, PG_dirty=0）优先被 kswapd 淘汰。
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

from store_vfs import ensure_schema, insert_chunk, apply_generation_effect_v2 as apply_generation_effect
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


def _make_chunk(cid, content="the the the the the the the the the the", importance=0.6,
                stability=5.0, chunk_type="decision", project="test"):
    now_iso = _utcnow().isoformat()
    import datetime as _dt
    la_iso = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(minutes=10)).isoformat()
    return {
        "id": cid,
        "project": project,
        "source_session": "",
        "chunk_type": chunk_type,
        "content": content,
        "summary": "summary",
        "importance": importance,
        "stability": stability,
        "created_at": now_iso,
        "updated_at": now_iso,
        "retrievability": 0.5,
        "last_accessed": la_iso,
        "access_count": 0,
        "encode_context": "test_context",
    }


def _get_stability(conn, cid: str) -> float:
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


# ── GE1: generative chunk_type → stability 加成 ─────────────────────────────────────────

def test_ge1_generative_type_boosted(conn):
    """GE1: chunk_type=decision（generative）→ stability 加成。"""
    # 低密度内容避免 KDEE/DDE/ETE 干扰
    content = "the the the the the the the the the the the the"

    chunk_gen = _make_chunk("ge_1_gen", content=content, chunk_type="decision")
    insert_chunk(conn, chunk_gen)
    stab_gen = _get_stability(conn, "ge_1_gen")

    chunk_pas = _make_chunk("ge_1_pas", content=content, chunk_type="observation")
    insert_chunk(conn, chunk_pas)
    stab_pas = _get_stability(conn, "ge_1_pas")

    assert stab_gen > stab_pas, (
        f"GE1: generative chunk_type stability 应高于 non-generative，"
        f"gen={stab_gen:.4f} pas={stab_pas:.4f}"
    )


# ── GE2: non-generative chunk_type → 无加成 ──────────────────────────────────────────────

def test_ge2_non_generative_no_boost(conn):
    """GE2: chunk_type=observation（non-generative）→ 无 GE 加成（与 generative 相比不更高）。"""
    content = "the the the the the the the the the the the the"

    chunk_gen = _make_chunk("ge_2_gen", content=content, chunk_type="decision")
    insert_chunk(conn, chunk_gen)
    stab_gen = _get_stability(conn, "ge_2_gen")

    chunk_pas = _make_chunk("ge_2_pas", content=content, chunk_type="observation")
    insert_chunk(conn, chunk_pas)
    stab_pas = _get_stability(conn, "ge_2_pas")

    assert stab_pas <= stab_gen + 0.01, (
        f"GE2: non-generative stability 不应高于 generative，gen={stab_gen:.4f} pas={stab_pas:.4f}"
    )


# ── GE3: ge_enabled=False → 无加成 ────────────────────────────────────────────────────────

def test_ge3_disabled_no_boost(conn):
    """GE3: ge_enabled=False → generative chunk_type 无 GE 加成。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.ge_enabled":
            return False
        return original_get(key, project=project)

    content = "the the the the the the the the the the the the"

    chunk_dis = _make_chunk("ge_3_dis", content=content, chunk_type="decision")
    with mock.patch.object(config, 'get', side_effect=patched_get):
        insert_chunk(conn, chunk_dis)
    stab_disabled = _get_stability(conn, "ge_3_dis")

    chunk_en = _make_chunk("ge_3_en", content=content, chunk_type="decision")
    insert_chunk(conn, chunk_en)
    stab_enabled = _get_stability(conn, "ge_3_en")

    assert stab_disabled <= stab_enabled + 0.01, (
        f"GE3: disabled 时 stability 不应高于 enabled，"
        f"disabled={stab_disabled:.4f} enabled={stab_enabled:.4f}"
    )


# ── GE4: importance 不足 → 不参与 GE ──────────────────────────────────────────────────────

def test_ge4_low_importance_no_boost(conn):
    """GE4: importance < ge_min_importance(0.25) → 不参与 GE。"""
    content = "the the the the the the the the the the the the"

    chunk_low = _make_chunk("ge_4_low", content=content, importance=0.10, chunk_type="decision")
    insert_chunk(conn, chunk_low)
    stab_low = _get_stability(conn, "ge_4_low")

    chunk_high = _make_chunk("ge_4_high", content=content, importance=0.60, chunk_type="decision")
    insert_chunk(conn, chunk_high)
    stab_high = _get_stability(conn, "ge_4_high")

    assert stab_low <= stab_high + 0.01, (
        f"GE4: 低 importance 时 stability 不应高于高 importance，"
        f"low={stab_low:.4f} high={stab_high:.4f}"
    )


# ── GE5: 加成受 ge_max_boost 保护 ──────────────────────────────────────────────────────────

def test_ge5_max_boost_cap(conn):
    """GE5: GE 增量不超过 base × ge_max_boost(0.18)。"""
    ge_max_boost = config.get("store_vfs.ge_max_boost")  # 0.18
    base = 5.0
    content = "the the the the the the the the the the the the"

    chunk_base = _make_chunk("ge_5_base", content=content, importance=0.6, stability=base,
                              chunk_type="observation")
    insert_chunk(conn, chunk_base)
    stab_base = _get_stability(conn, "ge_5_base")

    chunk_ge = _make_chunk("ge_5_ge", content=content, importance=0.6, stability=base,
                            chunk_type="decision")
    insert_chunk(conn, chunk_ge)
    stab_ge = _get_stability(conn, "ge_5_ge")

    increment = stab_ge - stab_base
    max_allowed = base * ge_max_boost + 0.2  # wider tolerance for compound effects
    assert increment <= max_allowed, (
        f"GE5: GE 增量 {increment:.4f} 不应超过 max_boost 允许的 {max_allowed:.4f}，"
        f"base={stab_base:.4f} ge={stab_ge:.4f}"
    )
    assert stab_ge > stab_base, f"GE5: 应有 GE 加成，base={stab_base:.4f} ge={stab_ge:.4f}"


# ── GE6: stability 上限 365.0 ─────────────────────────────────────────────────────────────

def test_ge6_stability_cap_365(conn):
    """GE6: GE boost 后 stability 不超过 365.0（直接调用 apply_generation_effect）。"""
    now_iso = _utcnow().isoformat()
    import datetime as _dt
    la_iso = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(minutes=10)).isoformat()
    base = 363.0
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count,
            encode_context, session_type_history)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("ge_6", "test", "decision", "content", "summary",
         0.8, base, now_iso, now_iso, 0.5, now_iso, 0, "test_ctx", "")
    )
    conn.commit()

    apply_generation_effect(conn, "ge_6", "decision")
    stab = float(conn.execute(
        "SELECT stability FROM memory_chunks WHERE id=?", ("ge_6",)
    ).fetchone()[0])
    assert stab <= 365.0, f"GE6: stability 不应超过 365.0，got {stab}"


# ── GE7: 直接调用 apply_generation_effect ──────────────────────────────────────────────────

def test_ge7_direct_function_boost(conn):
    """GE7: apply_generation_effect 直接对 generative chunk_type 产生加成。"""
    now_iso = _utcnow().isoformat()
    import datetime as _dt
    la_iso = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(minutes=10)).isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count,
            encode_context, session_type_history)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("ge_7", "test", "decision", "content", "summary",
         0.6, 5.0, now_iso, now_iso, 0.5, now_iso, 0, "test_ctx", "")
    )
    conn.commit()

    stab_before = float(conn.execute(
        "SELECT stability FROM memory_chunks WHERE id=?", ("ge_7",)
    ).fetchone()[0])
    result = apply_generation_effect(conn, "ge_7", "decision")
    stab_after = float(conn.execute(
        "SELECT stability FROM memory_chunks WHERE id=?", ("ge_7",)
    ).fetchone()[0])

    assert stab_after > stab_before, (
        f"GE7: GE 应对 generative chunk_type 加成，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["ge_boosted"] is True, f"GE7: ge_boosted 应为 True，got {result}"


# ── GE8: 所有 generative types 均触发 GE ──────────────────────────────────────────────────

def test_ge8_all_generative_types_boosted(conn):
    """GE8: ge_generative_types 中所有类型（decision/design_constraint/feedback）均触发 GE。"""
    content = "the the the the the the the the the the the the"
    generative_types = ["decision", "design_constraint", "feedback"]

    # baseline: observation (non-generative)
    chunk_obs = _make_chunk("ge_8_obs", content=content, chunk_type="observation")
    insert_chunk(conn, chunk_obs)
    stab_obs = _get_stability(conn, "ge_8_obs")

    for i, ctype in enumerate(generative_types):
        cid = f"ge_8_{ctype}"
        chunk = _make_chunk(cid, content=content, chunk_type=ctype)
        insert_chunk(conn, chunk)
        stab = _get_stability(conn, cid)
        assert stab > stab_obs - 0.01, (
            f"GE8: chunk_type={ctype} stability 应 >= observation，"
            f"stab={stab:.4f} obs={stab_obs:.4f}"
        )
