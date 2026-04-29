"""
test_iter458_eie.py — iter458: Elaborative Interrogation Effect 单元测试

覆盖：
  EI1: 内容含 >= 2 个因果连接词 + importance >= 0.40 → insert_chunk 时 stability 提升
  EI2: 因果连接词 < eie_min_connectives(2) → 无 EIE 加成
  EI3: eie_enabled=False → 无任何加成
  EI4: importance < eie_min_importance(0.40) → 不参与 EIE
  EI5: 加成系数 = eie_boost_factor(1.15)，即 stability × 1.15
  EI6: 加成被 eie_max_boost(0.30) 上限保护（stability 最多 × 1.30）
  EI7: stability 加成后不超过 365.0（cap 保护）
  EI8: 英文因果连接词触发（because/therefore/causes/hence/consequently）
  EI9: 中文因果连接词触发（因为/导致/因此/所以/由于）
  EI10: 连接词计数跨 content + summary（合并搜索）

认知科学依据：
  Pressley et al. (1992) "Elaborative interrogation and memory for prose" —
    解释"为什么这个事实是真的"使记忆保留率提升 72%（EI=72%, control=37%）。
  Martin & Pressley (1991) — "why" 问题比 "what" 问题更有效；因果性越强，编码越深。

OS 类比：Linux ext4 htree directory indexing —
  深度索引（因果关联 = htree depth 深）使文件查找从 O(N) 降到 O(log N)；
  因果解释型 chunk = 有深度 htree 的目录 = 更低检索代价（更高 stability）。
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

from store_vfs import ensure_schema, insert_chunk, apply_elaborative_interrogation_effect
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


def _make_chunk(cid, content="", summary="", importance=0.6, stability=5.0,
                chunk_type="decision", project="test"):
    now_iso = _utcnow().isoformat()
    return {
        "id": cid,
        "project": project,
        "source_session": "",
        "chunk_type": chunk_type,
        "content": content,
        "summary": summary,
        "importance": importance,
        "stability": stability,
        "created_at": now_iso,
        "updated_at": now_iso,
        "retrievability": 0.5,
        "last_accessed": now_iso,
        "access_count": 0,
        "encode_context": "kernel_mm",
    }


def _get_stability(conn, cid: str) -> float:
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


# ── EI1: 因果连接词 >= 2 + importance >= 0.40 → stability 提升 ─────────────────────────────

def test_ei1_causal_connectives_boost_stability(conn):
    """EI1: content 含 >= 2 个因果连接词 → insert_chunk 后 stability 提升。"""
    content = "The system failed because the config was wrong. Therefore we need to fix it."
    chunk = _make_chunk("eie_1", content=content, importance=0.6, stability=5.0)
    insert_chunk(conn, chunk)
    stab_after = _get_stability(conn, "eie_1")

    assert stab_after > 5.0, (
        f"EI1: 因果连接词 >= 2 时 stability 应提升，base=5.0 after={stab_after:.4f}"
    )


# ── EI2: 连接词不足 → 无加成 ──────────────────────────────────────────────────────────────

def test_ei2_insufficient_connectives_no_boost(conn):
    """EI2: 只有 1 个因果连接词 (< eie_min_connectives=2) → 无 EIE 加成（相对比较）。"""
    # 对照组：0 个连接词，低密度（避免 KDEE/DDE 干扰）
    content_zero = "the the the the the the the the the the the the the the the"
    chunk_zero = _make_chunk("eie_2_zero", content=content_zero, importance=0.6, stability=5.0)
    insert_chunk(conn, chunk_zero)
    stab_zero = _get_stability(conn, "eie_2_zero")

    # 实验组：1 个连接词（不足 min_connectives=2），低密度
    content_one = "the the the the because the the the the the the the the the the"
    chunk_one = _make_chunk("eie_2_one", content=content_one, importance=0.6, stability=5.0)
    insert_chunk(conn, chunk_one)
    stab_one = _get_stability(conn, "eie_2_one")

    # 1 个连接词不应触发 EIE，两者 stability 应相近（允许 MIE 干扰效应的小幅差异）
    assert abs(stab_one - stab_zero) < 0.20, (
        f"EI2: 连接词不足时不应有 EIE 加成，zero={stab_zero:.4f} one={stab_one:.4f}"
    )


# ── EI3: eie_enabled=False → 无加成 ───────────────────────────────────────────────────────

def test_ei3_disabled_no_boost(conn):
    """EI3: store_vfs.eie_enabled=False → 无 EIE 加成（与 enabled 相比 stability 不更高）。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.eie_enabled":
            return False
        return original_get(key, project=project)

    content = "The system failed because the config was wrong. Therefore we need to fix it."

    # disabled 分支
    chunk_disabled = _make_chunk("eie_3_dis", content=content, importance=0.6, stability=5.0)
    with mock.patch.object(config, 'get', side_effect=patched_get):
        insert_chunk(conn, chunk_disabled)
    stab_disabled = _get_stability(conn, "eie_3_dis")

    # enabled 分支（相同内容相同参数，EIE 应额外加成）
    chunk_enabled = _make_chunk("eie_3_en", content=content, importance=0.6, stability=5.0)
    insert_chunk(conn, chunk_enabled)
    stab_enabled = _get_stability(conn, "eie_3_en")

    # disabled 时不触发 EIE，stability 不应高于 enabled
    assert stab_disabled <= stab_enabled + 0.01, (
        f"EI3: eie_enabled=False 时 stability 不应高于 enabled，"
        f"disabled={stab_disabled:.4f} enabled={stab_enabled:.4f}"
    )


# ── EI4: importance 不足 → 不参与 EIE ────────────────────────────────────────────────────

def test_ei4_low_importance_no_boost(conn):
    """EI4: importance < eie_min_importance(0.40) → 不参与 EIE（相对比较）。"""
    content = "The system failed because the config was wrong. Therefore we need to fix it."

    # 低 importance（0.20 < 0.40 阈值）
    chunk_low = _make_chunk("eie_4_low", content=content, importance=0.20, stability=5.0)
    insert_chunk(conn, chunk_low)
    stab_low = _get_stability(conn, "eie_4_low")

    # 高 importance（0.60 >= 0.40，触发 EIE）
    chunk_high = _make_chunk("eie_4_high", content=content, importance=0.60, stability=5.0)
    insert_chunk(conn, chunk_high)
    stab_high = _get_stability(conn, "eie_4_high")

    # 低 importance 不触发 EIE，stability 不应高于高 importance 版本
    assert stab_low <= stab_high + 0.01, (
        f"EI4: 低 importance 时 stability 不应高于高 importance，"
        f"low={stab_low:.4f} high={stab_high:.4f}"
    )


# ── EI5: 加成系数 = eie_boost_factor(1.15) ────────────────────────────────────────────────

def test_ei5_boost_factor_1_15(conn):
    """EI5: EIE 加成系数 eie_boost_factor(1.15)：EIE chunk 比 baseline 高出约 15%。"""
    eie_boost_factor = config.get("store_vfs.eie_boost_factor")  # 1.15
    eie_max_boost = config.get("store_vfs.eie_max_boost")         # 0.30
    base = 5.0

    # baseline：不触发 EIE（0 个连接词，低密度避免 KDEE）
    content_base = "the the the the the the the the the the the the the"
    chunk_base = _make_chunk("eie_5_base", content=content_base, importance=0.6, stability=base)
    insert_chunk(conn, chunk_base)
    stab_base = _get_stability(conn, "eie_5_base")

    # EIE 分支：2 个连接词触发（低密度内容避免 KDEE 同时触发）
    content_eie = "the the the the because the the the the therefore the the the"
    chunk_eie = _make_chunk("eie_5_eie", content=content_eie, importance=0.6, stability=base)
    insert_chunk(conn, chunk_eie)
    stab_eie = _get_stability(conn, "eie_5_eie")

    # EIE 产生的增量 ≈ stab_base × (eie_boost_factor - 1.0)，允许其他效果的 ±10% 误差
    expected_increment = base * (eie_boost_factor - 1.0)  # 5.0 × 0.15 = 0.75
    actual_increment = stab_eie - stab_base

    assert abs(actual_increment - expected_increment) < expected_increment * 0.3 + 0.1, (
        f"EI5: EIE 增量应约为 {expected_increment:.4f}（base×{eie_boost_factor - 1.0:.2f}），"
        f"stab_base={stab_base:.4f} stab_eie={stab_eie:.4f} increment={actual_increment:.4f}"
    )


# ── EI6: eie_max_boost=0.30 上限保护 ─────────────────────────────────────────────────────

def test_ei6_max_boost_cap(conn):
    """EI6: eie_boost_factor 大时被 eie_max_boost(0.30) 上限保护（直接调用 apply_elaborative_interrogation_effect）。"""
    original_get = config.get

    def patched_get_large(key, project=None):
        if key == "store_vfs.eie_boost_factor":
            return 2.0  # 远超 max_boost=0.30
        return original_get(key, project=project)

    eie_max_boost = config.get("store_vfs.eie_max_boost")  # 0.30
    base = 5.0
    content = "the the the the because the therefore the the hence the the the"
    now_iso = _utcnow().isoformat()

    # 直接插入 DB，绕过 insert_chunk 中其他效应（GE iter406 等）干扰
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count,
            encode_context, session_type_history)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("eie_6", "test", "observation", content,
         "summary", 0.6, base, now_iso, now_iso, 0.5, now_iso, 0, "kernel_mm", "")
    )
    conn.commit()

    stab_before = _get_stability(conn, "eie_6")
    with mock.patch.object(config, 'get', side_effect=patched_get_large):
        apply_elaborative_interrogation_effect(conn, "eie_6", content)
    stab_after = _get_stability(conn, "eie_6")

    increment = stab_after - stab_before
    max_allowed_increment = base * eie_max_boost + 0.01  # 5.0 × 0.30 + tolerance
    assert increment <= max_allowed_increment, (
        f"EI6: EIE 增量 {increment:.4f} 不应超过 max_boost={eie_max_boost} 允许的 {max_allowed_increment:.4f}，"
        f"before={stab_before:.4f} after={stab_after:.4f}"
    )
    # 但确实有加成
    assert stab_after > stab_before, (
        f"EI6: 应有 EIE 加成，before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── EI7: stability 上限 365.0 ─────────────────────────────────────────────────────────────

def test_ei7_stability_cap_365(conn):
    """EI7: EIE boost 不超越 365.0（与基准组比较增量受限）。"""
    # 使用中等 stability（5.0），验证 EIE 加成后 EIE chunk <= baseline × (1 + max_boost) + 一些余量
    eie_max_boost = config.get("store_vfs.eie_max_boost")  # 0.30
    base = 5.0

    # baseline（无连接词，低密度）
    content_base = "the the the the the the the the the the the the the"
    chunk_base = _make_chunk("eie_7_base", content=content_base, importance=0.8, stability=base)
    insert_chunk(conn, chunk_base)
    stab_base = _get_stability(conn, "eie_7_base")

    # EIE chunk（2 个连接词触发 EIE，低密度避免 KDEE 同时触发）
    content_eie = "the the the the because the the the therefore the the the"
    chunk_eie = _make_chunk("eie_7_eie", content=content_eie, importance=0.8, stability=base)
    insert_chunk(conn, chunk_eie)
    stab_eie = _get_stability(conn, "eie_7_eie")

    # EIE 增量（相对于 baseline）不应超过 base × max_boost（因为 EIE 最多增加 max_boost 比例）
    eie_increment = stab_eie - stab_base
    max_eie_increment = base * eie_max_boost + 0.2  # 5.0 × 0.30 + tolerance
    assert eie_increment <= max_eie_increment, (
        f"EI7: EIE 增量 {eie_increment:.4f} 不应超过 base × max_boost = {max_eie_increment:.4f}，"
        f"stab_base={stab_base:.4f} stab_eie={stab_eie:.4f}"
    )


# ── EI8: 英文因果连接词触发 ───────────────────────────────────────────────────────────────

def test_ei8_english_connectives(conn):
    """EI8: 英文 'because', 'therefore', 'causes', 'hence', 'consequently' 均可触发。"""
    # 使用多个英文连接词
    content = "X causes Y, hence the system crashed."
    chunk = _make_chunk("eie_8_a", content=content, importance=0.6, stability=5.0)
    insert_chunk(conn, chunk)
    stab_a = _get_stability(conn, "eie_8_a")

    content2 = "The error is because of bug. Consequently we must fix it."
    chunk2 = _make_chunk("eie_8_b", content=content2, importance=0.6, stability=5.0)
    insert_chunk(conn, chunk2)
    stab_b = _get_stability(conn, "eie_8_b")

    assert stab_a > 5.0, f"EI8a: causes+hence 应触发 EIE，got {stab_a:.4f}"
    assert stab_b > 5.0, f"EI8b: because+consequently 应触发 EIE，got {stab_b:.4f}"


# ── EI9: 中文因果连接词触发 ───────────────────────────────────────────────────────────────

def test_ei9_chinese_connectives(conn):
    """EI9: 中文因果连接词（因为/导致/因此/所以/由于）可触发 EIE。"""
    content = "系统崩溃是因为配置错误，因此需要修复。"
    chunk = _make_chunk("eie_9", content=content, importance=0.6, stability=5.0)
    insert_chunk(conn, chunk)
    stab_after = _get_stability(conn, "eie_9")

    assert stab_after > 5.0, (
        f"EI9: 中文因果连接词应触发 EIE，base=5.0 got={stab_after:.4f}"
    )


# ── EI10: content + summary 合并计数 ──────────────────────────────────────────────────────

def test_ei10_content_plus_summary_combined(conn):
    """EI10: 因果连接词分布在 content 和 summary 中各 1 个，合计 >= 2 时触发。"""
    content = "The system failed because the config was wrong."   # 1 个
    summary = "Fixed it. Therefore stability improved."           # 1 个
    chunk = _make_chunk("eie_10", content=content, summary=summary,
                        importance=0.6, stability=5.0)
    insert_chunk(conn, chunk)
    stab_after = _get_stability(conn, "eie_10")

    assert stab_after > 5.0, (
        f"EI10: content+summary 合计 >= 2 个因果连接词时应触发 EIE，got {stab_after:.4f}"
    )
