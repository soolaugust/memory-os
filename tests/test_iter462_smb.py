"""
test_iter462_smb.py — iter462: Source Monitoring Boost 单元测试

覆盖：
  SM1: source_session 非空 → insert_chunk 时 stability 加成
  SM2: source_session 空 → 无 SMB 加成（相对比较）
  SM3: smb_enabled=False → 无任何加成
  SM4: importance < smb_min_importance(0.30) → 不参与 SMB
  SM5: 加成系数 = smb_boost_factor(1.08)，受 smb_max_boost(0.12) 保护
  SM6: stability 加成后不超过 365.0（cap 保护）

认知科学依据：
  Johnson, Hashtroudi & Lindsay (1993) "Source monitoring" (Psychological Bulletin) —
    source monitoring = 区分"在哪里/什么情境下学到的"能力。
    有清晰来源的记忆（episodic + semantic 双重标签）比来源模糊的记忆遗忘更慢。
  Lindsay (2008): 来源清晰度与记忆精确度正相关（r=0.48），因为额外编码维度 = 更多检索线索。

OS 类比：Linux inode i_generation — 有 generation 追踪的 inode 在 fsck 后可更快恢复
  （溯源信息完整 = 更鲁棒的文件系统状态 = 更低恢复成本）。
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

from store_vfs import ensure_schema, insert_chunk
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


def _make_chunk(cid, source_session="", importance=0.6, stability=5.0,
                chunk_type="decision", project="test"):
    now_iso = _utcnow().isoformat()
    return {
        "id": cid,
        "project": project,
        "source_session": source_session,
        "chunk_type": chunk_type,
        "content": "The system encountered an issue. Nothing special here.",
        "summary": "system issue",
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


# ── SM1: source_session 非空 → stability 加成 ─────────────────────────────────────────

def test_sm1_sourced_chunk_boosted(conn):
    """SM1: source_session 非空（有明确来源）→ insert_chunk 时 stability 加成。"""
    # 有来源
    chunk_sourced = _make_chunk("smb_1_src", source_session="session_20260428_debug",
                                importance=0.6, stability=5.0)
    insert_chunk(conn, chunk_sourced)
    stab_sourced = _get_stability(conn, "smb_1_src")

    # 无来源（baseline）
    chunk_nosrc = _make_chunk("smb_1_nosrc", source_session="",
                              importance=0.6, stability=5.0)
    insert_chunk(conn, chunk_nosrc)
    stab_nosrc = _get_stability(conn, "smb_1_nosrc")

    assert stab_sourced > stab_nosrc, (
        f"SM1: 有来源的 chunk stability 应高于无来源，"
        f"sourced={stab_sourced:.4f} nosrc={stab_nosrc:.4f}"
    )


# ── SM2: source_session 空 → 无 SMB 加成 ─────────────────────────────────────────────

def test_sm2_no_source_no_boost(conn):
    """SM2: source_session 为空时，SMB 不触发（稳定性不高于有来源版本）。"""
    # 这与 SM1 等价：无来源 <= 有来源。此测试重点验证空来源不会异常加成。
    chunk_empty = _make_chunk("smb_2_empty", source_session="", importance=0.6, stability=5.0)
    insert_chunk(conn, chunk_empty)
    stab_empty = _get_stability(conn, "smb_2_empty")

    chunk_none = _make_chunk("smb_2_none", source_session=None, importance=0.6, stability=5.0)
    insert_chunk(conn, chunk_none)
    stab_none = _get_stability(conn, "smb_2_none")

    # 空字符串和 None 效果相同（均不触发 SMB）
    assert abs(stab_empty - stab_none) < 0.05, (
        f"SM2: 空字符串和 None 来源效果应相近，empty={stab_empty:.4f} none={stab_none:.4f}"
    )


# ── SM3: smb_enabled=False → 无加成 ──────────────────────────────────────────────────

def test_sm3_disabled_no_boost(conn):
    """SM3: smb_enabled=False → 有来源 chunk 无 SMB 加成（与 enabled 相比不更高）。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.smb_enabled":
            return False
        return original_get(key, project=project)

    source = "session_20260428_debug"
    chunk_dis = _make_chunk("smb_3_dis", source_session=source, importance=0.6, stability=5.0)
    with mock.patch.object(config, 'get', side_effect=patched_get):
        insert_chunk(conn, chunk_dis)
    stab_disabled = _get_stability(conn, "smb_3_dis")

    chunk_en = _make_chunk("smb_3_en", source_session=source, importance=0.6, stability=5.0)
    insert_chunk(conn, chunk_en)
    stab_enabled = _get_stability(conn, "smb_3_en")

    # disabled 时 stability 不应高于 enabled
    assert stab_disabled <= stab_enabled + 0.01, (
        f"SM3: disabled 时 stability 不应高于 enabled，"
        f"disabled={stab_disabled:.4f} enabled={stab_enabled:.4f}"
    )


# ── SM4: importance 不足 → 不参与 SMB ────────────────────────────────────────────────

def test_sm4_low_importance_no_boost(conn):
    """SM4: importance < smb_min_importance(0.30) → 不参与 SMB。"""
    source = "session_20260428_debug"

    # 低 importance（0.10 < 0.30）
    chunk_low = _make_chunk("smb_4_low", source_session=source, importance=0.10, stability=5.0)
    insert_chunk(conn, chunk_low)
    stab_low = _get_stability(conn, "smb_4_low")

    # 高 importance（0.60 >= 0.30，触发 SMB）
    chunk_high = _make_chunk("smb_4_high", source_session=source, importance=0.60, stability=5.0)
    insert_chunk(conn, chunk_high)
    stab_high = _get_stability(conn, "smb_4_high")

    # 低 importance 不触发 SMB，stability 不应高于高 importance 版本
    assert stab_low <= stab_high + 0.01, (
        f"SM4: 低 importance 时 stability 不应高于高 importance，"
        f"low={stab_low:.4f} high={stab_high:.4f}"
    )


# ── SM5: 加成系数受 smb_max_boost 保护 ────────────────────────────────────────────────

def test_sm5_max_boost_cap(conn):
    """SM5: smb_boost_factor 受 smb_max_boost(0.12) 上限保护（相对 baseline）。"""
    smb_max_boost = config.get("store_vfs.smb_max_boost")  # 0.12
    base = 5.0

    # baseline（空来源，不触发 SMB）
    chunk_base = _make_chunk("smb_5_base", source_session="", importance=0.6, stability=base)
    insert_chunk(conn, chunk_base)
    stab_base = _get_stability(conn, "smb_5_base")

    # SMB chunk（有来源，触发 SMB）
    chunk_smb = _make_chunk("smb_5_smb", source_session="session_debug", importance=0.6, stability=base)
    insert_chunk(conn, chunk_smb)
    stab_smb = _get_stability(conn, "smb_5_smb")

    smb_increment = stab_smb - stab_base
    max_allowed = base * smb_max_boost + 0.1  # tolerance
    assert smb_increment <= max_allowed, (
        f"SM5: SMB 增量 {smb_increment:.4f} 不应超过 max_boost 允许的 {max_allowed:.4f}，"
        f"base={stab_base:.4f} smb={stab_smb:.4f}"
    )
    assert stab_smb > stab_base, f"SM5: 应有 SMB 加成，base={stab_base:.4f} smb={stab_smb:.4f}"


# ── SM6: stability 上限 365.0 ─────────────────────────────────────────────────────────

def test_sm6_stability_cap_365(conn):
    """SM6: SMB boost 后 stability 不超过 365.0（全局上限保护）。"""
    base = 360.0  # 接近上限以测试 cap

    chunk_smb = _make_chunk("smb_6_smb", source_session="session_debug", importance=0.8, stability=base)
    insert_chunk(conn, chunk_smb)
    stab_smb = _get_stability(conn, "smb_6_smb")

    assert stab_smb <= 365.0, (
        f"SM6: SMB boost 后 stability 不应超过 365.0，got {stab_smb:.4f}"
    )
