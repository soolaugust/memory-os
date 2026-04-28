"""
test_iter446_tce.py — iter446: Temporal Contiguity Effect 单元测试

覆盖：
  TC1: 时间毗邻的 chunk（created_at 差 <= tce_window_secs=1800s）→ stability 双向加成
  TC2: 时间间距超出窗口（> 1800s）→ 无 TCE 加成（不属于同一情节单元）
  TC3: 只有 1 个 chunk 在某时间窗口 → 无加成（孤立不构成情节单元）
  TC4: tce_enabled=False → 无任何加成
  TC5: importance < tce_min_importance(0.45) → chunk 不参与 TCE
  TC6: 3 个时间毗邻 chunk → 全部获得加成（组大小 >= 2 即触发）
  TC7: 组大小超过 tce_max_group_size(10) → 只有 top stability 的前 10 个获得加成
  TC8: consolidation 后 stability 不超过 365.0
  TC9: tce_bonus 可配置（更大 bonus → 更大加成）
  TC10: 返回计数正确（tce_boosted, total_examined）

认知科学依据：
  Kahana (1996) "Associative retrieval processes in free recall" —
    lag-CRP 曲线峰值在 lag=±1，时间相邻编码的记忆强度互相激活。
  Howard & Kahana (2002) — 时间上下文向量高度相关的相邻事件联合回放。

OS 类比：Linux MGLRU temporal cohort aging —
  同一 aging interval 的 pages 属于同一 generation，sleep 扫描时同代互保。
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
    apply_temporal_contiguity_consolidation,
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


def _iso(dt):
    return dt.isoformat()


def _insert_chunk(conn, cid, project="test", stability=5.0, importance=0.6,
                  created_at=None):
    """Insert chunk with specific created_at timestamp."""
    if created_at is None:
        created_at = _utcnow()
    now_iso = _utcnow().isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count,
            encode_context)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0.8, ?, 1, ?)""",
        (cid, project, "decision", f"content {cid}", f"summary {cid}",
         importance, stability, created_at.isoformat(), now_iso,
         now_iso, "")
    )
    conn.commit()


def _get_stability(conn, cid: str) -> float:
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


# ── TC1: 时间毗邻 chunk → stability 双向加成 ────────────────────────────────────

def test_tc1_contiguous_chunks_boosted(conn):
    """TC1: created_at 差 <= tce_window_secs(1800s) 的 2 个 chunk 均获得 stability 加成。"""
    tce_window = config.get("store_vfs.tce_window_secs")  # 1800s = 30min
    base_time = _utcnow() - datetime.timedelta(hours=1)

    _insert_chunk(conn, "c1", stability=5.0, importance=0.6,
                  created_at=base_time)
    _insert_chunk(conn, "c2", stability=5.0, importance=0.6,
                  created_at=base_time + datetime.timedelta(seconds=600))  # 10分钟后，在窗口内

    stab_c1_before = _get_stability(conn, "c1")
    stab_c2_before = _get_stability(conn, "c2")
    result = apply_temporal_contiguity_consolidation(conn, "test")
    stab_c1_after = _get_stability(conn, "c1")
    stab_c2_after = _get_stability(conn, "c2")

    assert stab_c1_after > stab_c1_before, (
        f"TC1: 时间毗邻 c1 应获得 stability 加成，before={stab_c1_before:.4f} after={stab_c1_after:.4f}"
    )
    assert stab_c2_after > stab_c2_before, (
        f"TC1: 时间毗邻 c2 应获得 stability 加成，before={stab_c2_before:.4f} after={stab_c2_after:.4f}"
    )
    assert result["tce_boosted"] >= 2, f"TC1: tce_boosted 应 >= 2，got {result}"


# ── TC2: 时间间距超出窗口 → 无加成 ───────────────────────────────────────────────

def test_tc2_far_apart_chunks_no_boost(conn):
    """TC2: created_at 差 > tce_window_secs(1800s) → 不属于同一情节单元 → 无加成。"""
    tce_window = config.get("store_vfs.tce_window_secs")  # 1800
    base_time = _utcnow() - datetime.timedelta(hours=5)

    _insert_chunk(conn, "far1", stability=5.0, importance=0.6, created_at=base_time)
    # 间距 = tce_window + 60s = 超出窗口
    _insert_chunk(conn, "far2", stability=5.0, importance=0.6,
                  created_at=base_time + datetime.timedelta(seconds=tce_window + 60))

    stab_far1_before = _get_stability(conn, "far1")
    stab_far2_before = _get_stability(conn, "far2")
    apply_temporal_contiguity_consolidation(conn, "test")
    stab_far1_after = _get_stability(conn, "far1")
    stab_far2_after = _get_stability(conn, "far2")

    assert abs(stab_far1_after - stab_far1_before) < 0.001, (
        f"TC2: 超出时间窗口的 far1 不应获得加成，"
        f"before={stab_far1_before:.4f} after={stab_far1_after:.4f}"
    )
    assert abs(stab_far2_after - stab_far2_before) < 0.001, (
        f"TC2: 超出时间窗口的 far2 不应获得加成，"
        f"before={stab_far2_before:.4f} after={stab_far2_after:.4f}"
    )


# ── TC3: 时间窗口内只有 1 个 chunk → 无加成 ──────────────────────────────────────

def test_tc3_single_chunk_in_window_no_boost(conn):
    """TC3: 某时间窗口内只有 1 个 chunk → 孤立 → 不构成情节单元 → 无加成。"""
    base_time = _utcnow() - datetime.timedelta(hours=2)

    # 1 个孤立 chunk（前后时间均远离其他 chunk）
    _insert_chunk(conn, "lone", stability=5.0, importance=0.6, created_at=base_time)
    # 另一个 chunk 间距超出窗口（不与 lone 毗邻）
    tce_window = config.get("store_vfs.tce_window_secs")
    _insert_chunk(conn, "other", stability=5.0, importance=0.6,
                  created_at=base_time + datetime.timedelta(seconds=tce_window + 300))

    stab_lone_before = _get_stability(conn, "lone")
    result = apply_temporal_contiguity_consolidation(conn, "test")
    stab_lone_after = _get_stability(conn, "lone")

    assert abs(stab_lone_after - stab_lone_before) < 0.001, (
        f"TC3: 孤立 chunk 不应获得 TCE 加成，"
        f"before={stab_lone_before:.4f} after={stab_lone_after:.4f}"
    )


# ── TC4: tce_enabled=False → 无加成 ─────────────────────────────────────────────

def test_tc4_disabled_no_boost(conn):
    """TC4: store_vfs.tce_enabled=False → 无任何 TCE 加成。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.tce_enabled":
            return False
        return original_get(key, project=project)

    base_time = _utcnow() - datetime.timedelta(hours=1)
    _insert_chunk(conn, "d1", stability=5.0, importance=0.6, created_at=base_time)
    _insert_chunk(conn, "d2", stability=5.0, importance=0.6,
                  created_at=base_time + datetime.timedelta(seconds=300))

    stab_d1_before = _get_stability(conn, "d1")
    stab_d2_before = _get_stability(conn, "d2")

    with mock.patch.object(config, 'get', side_effect=patched_get):
        result = apply_temporal_contiguity_consolidation(conn, "test")

    stab_d1_after = _get_stability(conn, "d1")
    stab_d2_after = _get_stability(conn, "d2")

    assert abs(stab_d1_after - stab_d1_before) < 0.001, (
        f"TC4: disabled 时 d1 不应获得加成，before={stab_d1_before:.4f} after={stab_d1_after:.4f}"
    )
    assert abs(stab_d2_after - stab_d2_before) < 0.001, (
        f"TC4: disabled 时 d2 不应获得加成，before={stab_d2_before:.4f} after={stab_d2_after:.4f}"
    )
    assert result["tce_boosted"] == 0, f"TC4: tce_boosted 应为 0，got {result}"


# ── TC5: importance 不足 → 不参与 TCE ────────────────────────────────────────────

def test_tc5_low_importance_excluded(conn):
    """TC5: importance < tce_min_importance(0.45) 的 chunk 不参与 TCE（即使时间毗邻）。"""
    tce_min_imp = config.get("store_vfs.tce_min_importance")  # 0.45
    base_time = _utcnow() - datetime.timedelta(hours=1)

    # 低 importance chunk（时间毗邻，但低于 0.45 阈值）
    _insert_chunk(conn, "low_imp", stability=5.0, importance=0.20,  # < 0.45
                  created_at=base_time)
    _insert_chunk(conn, "low_imp2", stability=5.0, importance=0.30,  # < 0.45
                  created_at=base_time + datetime.timedelta(seconds=300))

    stab_li1_before = _get_stability(conn, "low_imp")
    stab_li2_before = _get_stability(conn, "low_imp2")
    apply_temporal_contiguity_consolidation(conn, "test")
    stab_li1_after = _get_stability(conn, "low_imp")
    stab_li2_after = _get_stability(conn, "low_imp2")

    assert abs(stab_li1_after - stab_li1_before) < 0.001, (
        f"TC5: 低 importance chunk 不应参与 TCE，"
        f"before={stab_li1_before:.4f} after={stab_li1_after:.4f}"
    )
    assert abs(stab_li2_after - stab_li2_before) < 0.001, (
        f"TC5: 低 importance chunk 不应参与 TCE，"
        f"before={stab_li2_before:.4f} after={stab_li2_after:.4f}"
    )


# ── TC6: 3 个时间毗邻 chunk → 全部获得加成 ───────────────────────────────────────

def test_tc6_three_contiguous_chunks_all_boosted(conn):
    """TC6: 3 个时间毗邻 chunk → 属于同一情节单元 → 全部获得 TCE 加成。"""
    base_time = _utcnow() - datetime.timedelta(hours=1)
    _insert_chunk(conn, "t1", stability=5.0, importance=0.6, created_at=base_time)
    _insert_chunk(conn, "t2", stability=5.0, importance=0.7,
                  created_at=base_time + datetime.timedelta(minutes=5))
    _insert_chunk(conn, "t3", stability=5.0, importance=0.5,
                  created_at=base_time + datetime.timedelta(minutes=10))

    stabs_before = [_get_stability(conn, f"t{i}") for i in range(1, 4)]
    result = apply_temporal_contiguity_consolidation(conn, "test")
    stabs_after = [_get_stability(conn, f"t{i}") for i in range(1, 4)]

    for i, (before, after) in enumerate(zip(stabs_before, stabs_after), 1):
        assert after > before, (
            f"TC6: t{i} 应获得 TCE 加成，before={before:.4f} after={after:.4f}"
        )
    assert result["tce_boosted"] >= 3, f"TC6: tce_boosted 应 >= 3，got {result}"


# ── TC7: 组大小超过 tce_max_group_size → 只有 top stability 获得加成 ──────────────

def test_tc7_large_group_capped(conn):
    """TC7: 组大小超过 tce_max_group_size(10) 时，只有 top stability 的前 10 个获得加成。"""
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.tce_max_group_size":
            return 3  # 限制为 3，便于测试
        return original_get(key, project=project)

    base_time = _utcnow() - datetime.timedelta(hours=1)
    # 插入 5 个时间毗邻 chunk（超过 patched max=3）
    # 按 stability 降序：c_h1(10.0) > c_h2(8.0) > c_h3(6.0) > c_l1(2.0) > c_l2(1.0)
    _insert_chunk(conn, "c_h1", stability=10.0, importance=0.6,
                  created_at=base_time)
    _insert_chunk(conn, "c_h2", stability=8.0, importance=0.6,
                  created_at=base_time + datetime.timedelta(seconds=120))
    _insert_chunk(conn, "c_h3", stability=6.0, importance=0.6,
                  created_at=base_time + datetime.timedelta(seconds=240))
    _insert_chunk(conn, "c_l1", stability=2.0, importance=0.6,
                  created_at=base_time + datetime.timedelta(seconds=360))
    _insert_chunk(conn, "c_l2", stability=1.0, importance=0.6,
                  created_at=base_time + datetime.timedelta(seconds=480))

    stab_h1_before = _get_stability(conn, "c_h1")
    stab_l2_before = _get_stability(conn, "c_l2")

    with mock.patch.object(config, 'get', side_effect=patched_get):
        result = apply_temporal_contiguity_consolidation(conn, "test")

    stab_h1_after = _get_stability(conn, "c_h1")
    stab_l2_after = _get_stability(conn, "c_l2")

    # top 3 stability 的 chunk（c_h1, c_h2, c_h3）应获得加成
    assert stab_h1_after > stab_h1_before, (
        f"TC7: 高 stability chunk 应获得 TCE 加成（在 top 3 中），"
        f"before={stab_h1_before:.4f} after={stab_h1_after:.4f}"
    )
    # 低 stability（c_l1=2.0, c_l2=1.0）不在 top 3 → 无加成
    assert abs(stab_l2_after - stab_l2_before) < 0.001, (
        f"TC7: 低 stability chunk 不应获得加成（不在 top 3 中），"
        f"before={stab_l2_before:.4f} after={stab_l2_after:.4f}"
    )
    assert result["tce_boosted"] == 3, f"TC7: tce_boosted 应 == 3（cap 为 3），got {result}"


# ── TC8: consolidation 后 stability 不超过 365.0 ─────────────────────────────────

def test_tc8_stability_cap_365(conn):
    """TC8: TCE 加成后 stability 不超过 365.0。"""
    base_time = _utcnow() - datetime.timedelta(hours=1)
    _insert_chunk(conn, "cap1", stability=364.9, importance=0.6, created_at=base_time)
    _insert_chunk(conn, "cap2", stability=5.0, importance=0.6,
                  created_at=base_time + datetime.timedelta(seconds=300))

    apply_temporal_contiguity_consolidation(conn, "test")

    assert _get_stability(conn, "cap1") <= 365.0, "TC8: stability 不应超过 365.0"
    assert _get_stability(conn, "cap2") <= 365.0, "TC8: stability 不应超过 365.0"


# ── TC9: tce_bonus 可配置 ────────────────────────────────────────────────────────

def test_tc9_configurable_bonus(conn):
    """TC9: tce_bonus=0.20 时加成比默认 0.05 更大。"""
    original_get = config.get
    base_time = _utcnow() - datetime.timedelta(hours=1)

    _insert_chunk(conn, "b1", stability=5.0, importance=0.6, created_at=base_time)
    _insert_chunk(conn, "b2", stability=5.0, importance=0.6,
                  created_at=base_time + datetime.timedelta(seconds=300))

    def patched_20(key, project=None):
        if key == "store_vfs.tce_bonus":
            return 0.20
        return original_get(key, project=project)

    stab_before = _get_stability(conn, "b1")
    with mock.patch.object(config, 'get', side_effect=patched_20):
        apply_temporal_contiguity_consolidation(conn, "test")
    stab_after_20 = _get_stability(conn, "b1")
    delta_20 = stab_after_20 - stab_before

    # 重置
    conn.execute("UPDATE memory_chunks SET stability=5.0 WHERE id='b1'")
    conn.commit()

    stab_before_default = _get_stability(conn, "b1")
    apply_temporal_contiguity_consolidation(conn, "test")  # 默认 bonus=0.05
    stab_after_default = _get_stability(conn, "b1")
    delta_default = stab_after_default - stab_before_default

    assert delta_20 > delta_default, (
        f"TC9: tce_bonus=0.20 应大于默认 0.05，"
        f"delta_20={delta_20:.5f} delta_default={delta_default:.5f}"
    )


# ── TC10: 返回计数正确 ─────────────────────────────────────────────────────────────

def test_tc10_return_counts_correct(conn):
    """TC10: result dict 中 tce_boosted 和 total_examined 计数正确。"""
    tce_window = config.get("store_vfs.tce_window_secs")
    base_time = _utcnow() - datetime.timedelta(hours=2)

    # 第 1 组：3 个时间毗邻 chunk → 全部被加成
    _insert_chunk(conn, "g1a", stability=5.0, importance=0.6, created_at=base_time)
    _insert_chunk(conn, "g1b", stability=5.0, importance=0.7,
                  created_at=base_time + datetime.timedelta(seconds=300))
    _insert_chunk(conn, "g1c", stability=5.0, importance=0.5,
                  created_at=base_time + datetime.timedelta(seconds=600))

    # 间隔：超出窗口（第 1 组和第 2 组分离）
    gap = base_time + datetime.timedelta(seconds=tce_window + 3600)

    # 第 2 组：2 个时间毗邻 chunk → 全部被加成
    _insert_chunk(conn, "g2a", stability=4.0, importance=0.6, created_at=gap)
    _insert_chunk(conn, "g2b", stability=4.0, importance=0.6,
                  created_at=gap + datetime.timedelta(seconds=200))

    # 孤立 chunk（无时间邻居）
    _insert_chunk(conn, "lone_chunk", stability=5.0, importance=0.6,
                  created_at=gap + datetime.timedelta(seconds=tce_window + 3600))

    # 低 importance → 不参与
    _insert_chunk(conn, "low", stability=5.0, importance=0.10,
                  created_at=base_time + datetime.timedelta(seconds=100))

    result = apply_temporal_contiguity_consolidation(conn, "test")

    assert "tce_boosted" in result, "TC10: result 应含 tce_boosted key"
    assert "total_examined" in result, "TC10: result 应含 total_examined key"
    assert result["tce_boosted"] >= 5, (
        f"TC10: 应有 >= 5 个 chunk 被加成（2 组共 5 个），got {result}"
    )
    # total_examined 包含所有符合 importance 阈值的 chunk（g1a+b+c+g2a+b+lone=6）
    assert result["total_examined"] >= 6, (
        f"TC10: total_examined 应 >= 6，got {result}"
    )
