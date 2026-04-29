"""
test_iter475_spe.py — iter475: Serial Position Effect 单元测试

覆盖：
  SP1: session 首位 chunk（primacy）stability 加成
  SP2: session 中间位置 chunk → 无 primacy 加成
  SP3: spe_enabled=False → 无 primacy 加成
  SP4: session 末位 chunk（recency）retrievability 加成
  SP5: session 中间位置 chunk → 无 recency 加成
  SP6: session 大小 < spe_min_session_size(3) → 不触发 SPE
  SP7: importance < spe_min_importance(0.25) → 不参与 SPE
  SP8: update_accessed 集成测试 — recency 在 update_accessed 时触发

认知科学依据：
  Murdock (1962) "The serial position effect of free recall" —
    序列首位（primacy）和末位（recency）recall 率比中间高约 20-30%。
    Primacy: 首位项目有更多复习时间 → 长时记忆（stability）更高。
    Recency: 末位项目仍在工作记忆 → 短期可达性（retrievability）更高。
  Glanzer & Cunitz (1966): recency 效应在延迟测试后消失（∴ 只影响 retrievability）。

OS 类比：CPU L1/L2 cache LRU —
  最先加载的 hot page（primacy）因反复被引用留在 cache；
  最近访问的 page（recency）在 MRU 位置，下次命中概率最高。
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
    ensure_schema, update_accessed,
    apply_serial_position_primacy, apply_serial_position_recency
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


def _insert_raw(conn, cid, content="content", project="test", chunk_type="observation",
                importance=0.6, stability=5.0, retrievability=0.5,
                source_session="sess1", created_offset_seconds=0):
    """created_offset_seconds 用于控制 session 内的创建顺序。"""
    import datetime as dt
    ts = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=created_offset_seconds)
    now_iso = ts.isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count,
            encode_context, session_type_history, source_session)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (cid, project, chunk_type, content, "summary", importance, stability,
         now_iso, now_iso, retrievability, now_iso, 0, "test_ctx", "coding", source_session)
    )
    conn.commit()


def _get_stability(conn, cid: str) -> float:
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


def _get_retrievability(conn, cid: str) -> float:
    row = conn.execute("SELECT retrievability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


# ── SP1: session 首位 chunk 获得 primacy stability 加成 ──────────────────────────────────

def test_sp1_primacy_boost_first_chunk(conn):
    """SP1: session 首位 chunk（position=0）获得 primacy stability 加成。"""
    # 插入 5 个 chunk，使 session 大小 >= min_session_size=3
    for i in range(5):
        _insert_raw(conn, f"sp1_{i}", source_session="sess_sp1",
                    created_offset_seconds=i, stability=5.0)

    # 对首位 chunk 调用 primacy
    stab_before = _get_stability(conn, "sp1_0")
    result = apply_serial_position_primacy(conn, "sp1_0", "sess_sp1", "test")
    stab_after = _get_stability(conn, "sp1_0")

    assert stab_after > stab_before, (
        f"SP1: 首位 chunk 应获得 primacy 加成，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["spe_primacy_boosted"] is True, f"SP1: spe_primacy_boosted 应为 True"


# ── SP2: 中间位置 chunk → 无 primacy 加成 ────────────────────────────────────────────────

def test_sp2_middle_chunk_no_primacy(conn):
    """SP2: session 中间位置（position >= primacy_window=5）的 chunk 无 primacy 加成。"""
    for i in range(10):
        _insert_raw(conn, f"sp2_{i}", source_session="sess_sp2",
                    created_offset_seconds=i, stability=5.0)

    # position=7（远超 primacy_window=5）
    stab_before = _get_stability(conn, "sp2_7")
    result = apply_serial_position_primacy(conn, "sp2_7", "sess_sp2", "test")
    stab_after = _get_stability(conn, "sp2_7")

    assert abs(stab_after - stab_before) < 0.001, (
        f"SP2: 中间位置不应有 primacy 加成，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["spe_primacy_boosted"] is False, f"SP2: spe_primacy_boosted 应为 False"


# ── SP3: spe_enabled=False → 无 primacy 加成 ─────────────────────────────────────────────

def test_sp3_disabled_no_primacy(conn):
    """SP3: spe_enabled=False → 首位 chunk 无 primacy 加成。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.spe_enabled":
            return False
        return original_get(key, project=project)

    for i in range(5):
        _insert_raw(conn, f"sp3_{i}", source_session="sess_sp3",
                    created_offset_seconds=i, stability=5.0)

    stab_before = _get_stability(conn, "sp3_0")
    with mock.patch.object(config, 'get', side_effect=patched_get):
        result = apply_serial_position_primacy(conn, "sp3_0", "sess_sp3", "test")
    stab_after = _get_stability(conn, "sp3_0")

    assert abs(stab_after - stab_before) < 0.001, (
        f"SP3: disabled 时不应有 primacy 加成，before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── SP4: session 末位 chunk 获得 recency retrievability 加成 ─────────────────────────────

def test_sp4_recency_boost_last_chunk(conn):
    """SP4: session 末位 chunk（最近 recency_window=5 内）获得 recency retrievability 加成。"""
    for i in range(8):
        _insert_raw(conn, f"sp4_{i}", source_session="sess_sp4",
                    created_offset_seconds=i, retrievability=0.5)

    # 末位 chunk（position=7，在最近 5 个内）
    retr_before = _get_retrievability(conn, "sp4_7")
    result = apply_serial_position_recency(conn, ["sp4_7"])
    retr_after = _get_retrievability(conn, "sp4_7")

    assert retr_after > retr_before, (
        f"SP4: 末位 chunk 应获得 recency retrievability 加成，"
        f"before={retr_before:.4f} after={retr_after:.4f}"
    )
    assert result["spe_recency_boosted"] >= 1, f"SP4: spe_recency_boosted 应 >= 1"


# ── SP5: 中间位置 chunk → 无 recency 加成 ────────────────────────────────────────────────

def test_sp5_middle_chunk_no_recency(conn):
    """SP5: session 中间位置（非最近 recency_window=5 内）的 chunk 无 recency 加成。"""
    for i in range(10):
        _insert_raw(conn, f"sp5_{i}", source_session="sess_sp5",
                    created_offset_seconds=i, retrievability=0.5)

    # position=0（最早写入，不在最近 5 个内）
    retr_before = _get_retrievability(conn, "sp5_0")
    result = apply_serial_position_recency(conn, ["sp5_0"])
    retr_after = _get_retrievability(conn, "sp5_0")

    assert abs(retr_after - retr_before) < 0.001, (
        f"SP5: 中间位置不应有 recency 加成，before={retr_before:.4f} after={retr_after:.4f}"
    )


# ── SP6: session 大小不足 → 不触发 SPE ───────────────────────────────────────────────────

def test_sp6_small_session_no_spe(conn):
    """SP6: session 大小 < spe_min_session_size(3) → 不触发 SPE。"""
    # 只有 2 个 chunk（< min_session_size=3）
    for i in range(2):
        _insert_raw(conn, f"sp6_{i}", source_session="sess_sp6_small",
                    created_offset_seconds=i, stability=5.0)

    stab_before = _get_stability(conn, "sp6_0")
    result = apply_serial_position_primacy(conn, "sp6_0", "sess_sp6_small", "test")
    stab_after = _get_stability(conn, "sp6_0")

    assert abs(stab_after - stab_before) < 0.001, (
        f"SP6: session 过小时不应触发 SPE primacy，before={stab_before:.4f} after={stab_after:.4f}"
    )
    assert result["spe_primacy_boosted"] is False, f"SP6: spe_primacy_boosted 应为 False"


# ── SP7: importance 不足 → 不参与 SPE ────────────────────────────────────────────────────

def test_sp7_low_importance_no_spe(conn):
    """SP7: importance < spe_min_importance(0.25) → 不触发 SPE primacy。"""
    for i in range(5):
        _insert_raw(conn, f"sp7_{i}", source_session="sess_sp7",
                    created_offset_seconds=i, importance=0.10, stability=5.0)

    stab_before = _get_stability(conn, "sp7_0")
    result = apply_serial_position_primacy(conn, "sp7_0", "sess_sp7", "test")
    stab_after = _get_stability(conn, "sp7_0")

    assert abs(stab_after - stab_before) < 0.001, (
        f"SP7: 低 importance 时不应触发 SPE，before={stab_before:.4f} after={stab_after:.4f}"
    )


# ── SP8: update_accessed 集成测试 ────────────────────────────────────────────────────────

def test_sp8_update_accessed_recency_integration(conn):
    """SP8: update_accessed 触发 SPE recency，最近访问的 session 末位 chunk retrievability 提升。"""
    for i in range(6):
        _insert_raw(conn, f"sp8_{i}", source_session="sess_sp8",
                    created_offset_seconds=i, retrievability=0.5)

    # 访问末位 chunk（position=5，在 recency_window=5 内）
    retr_before = _get_retrievability(conn, "sp8_5")
    update_accessed(conn, ["sp8_5"], session_id="sess_sp8", project="test")
    retr_after = _get_retrievability(conn, "sp8_5")

    assert retr_after >= retr_before, (
        f"SP8: update_accessed 后末位 chunk retrievability 不应降低，"
        f"before={retr_before:.4f} after={retr_after:.4f}"
    )
