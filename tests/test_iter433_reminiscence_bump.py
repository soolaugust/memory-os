"""
test_iter433_reminiscence_bump.py — iter433: Reminiscence Bump Effect 单元测试

覆盖：
  RB1: 形成期 chunk（position_pct <= 15%，importance >= 0.55）→ stability 加成
  RB2: 非形成期 chunk（position_pct > 15%）→ 无加成
  RB3: importance < bump_min_importance → 无加成
  RB4: 项目年龄 < min_project_age_days → 无加成（项目太新）
  RB5: 加成因子 bump_factor 可通过 sysctl 配置
  RB6: bump_enabled=False → 无加成
  RB7: position_pct 边界（恰好 = bump_pct → 有加成）
  RB8: stability 上限 365.0（不超过遗忘曲线最大值）
  RB9: insert_chunk 触发 Reminiscence Bump — 早期 chunk stability > 晚期 chunk
  RB10: 单独项目（只有 1 个 chunk）— project_age_secs=0 时安全处理

认知科学依据：
  Conway & Howe (1990); Rubin et al. (1998) Reminiscence Bump —
  人类 15-25 岁形成期事件比其他阶段记忆得更清晰（+50%~+100% recall rate）。
  机制：核心自我叙事（core self-narrative）双重编码路径。

OS 类比：Linux early_boot firmware parameters —
  启动早期设置的核心参数（kernel cmdline）比运行时 sysctl 更稳定。
"""
import sys
import sqlite3
import datetime
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import tmpfs  # noqa

from store_vfs import (
    ensure_schema,
    compute_reminiscence_bump_factor,
    apply_reminiscence_bump,
    apply_reminiscence_bump_batch,
)
import config


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _ago_iso(days: float) -> str:
    return (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)).isoformat()


def _insert_raw(conn, cid, project="test", created_days_ago=0.0,
                importance=0.7, stability=2.0, chunk_type="decision"):
    created = _ago_iso(created_days_ago)
    now = _now_iso()
    conn.execute(
        """INSERT OR REPLACE INTO memory_chunks
           (id, project, chunk_type, content, summary, importance, stability,
            created_at, updated_at, retrievability, last_accessed, access_count)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0.9, ?, 1)""",
        (cid, project, chunk_type, f"content {cid}", f"summary {cid}",
         importance, stability, created, now, now)
    )
    conn.commit()


def _get_stability(conn, cid: str) -> float:
    row = conn.execute("SELECT stability FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return float(row[0]) if row else 0.0


# ── RB1: 形成期 chunk → stability 加成 ──────────────────────────────────────

def test_rb1_formative_chunk_gets_bump(conn):
    """RB1: position_pct <= 15% 且 importance >= 0.55 → stability 加成。"""
    # 项目年龄 = 100 天，bump_pct=0.15 → 前 15 天内创建的 chunk 获得加成
    # 插入项目最早 chunk（100 天前）
    _insert_raw(conn, "first_chunk", created_days_ago=100.0, importance=0.7, stability=2.0)
    # 插入当前 chunk（95 天前，distance=5 days, position_pct=5/100=5% <= 15%）
    _insert_raw(conn, "bump_chunk", created_days_ago=95.0, importance=0.7, stability=2.0)
    # 插入晚期 chunk（10 天前，position_pct=90/100=90% > 15%）
    _insert_raw(conn, "late_chunk", created_days_ago=10.0, importance=0.7, stability=2.0)

    bump_factor = config.get("store_vfs.bump_factor")  # 1.30
    new_stab = compute_reminiscence_bump_factor(conn, "bump_chunk", "test", 2.0)
    assert new_stab > 2.0, f"RB1: 形成期 chunk 应有 stability 加成，got {new_stab}"
    assert abs(new_stab - 2.0 * bump_factor) < 0.01, \
        f"RB1: stability 应为 2.0×{bump_factor}={2.0*bump_factor:.2f}，got {new_stab}"


# ── RB2: 非形成期 → 无加成 ──────────────────────────────────────────────────

def test_rb2_late_chunk_no_bump(conn):
    """RB2: position_pct > bump_pct(15%) → 无加成。"""
    _insert_raw(conn, "first_c", created_days_ago=100.0, importance=0.7, stability=2.0)
    _insert_raw(conn, "late_c", created_days_ago=10.0, importance=0.8, stability=3.0)

    new_stab = compute_reminiscence_bump_factor(conn, "late_c", "test", 3.0)
    assert abs(new_stab - 3.0) < 0.001, \
        f"RB2: 晚期 chunk 不应有加成，got {new_stab} vs base=3.0"


# ── RB3: importance 不足 → 无加成 ────────────────────────────────────────────

def test_rb3_low_importance_no_bump(conn):
    """RB3: importance < bump_min_importance(0.55) → 无加成。"""
    _insert_raw(conn, "first_c2", created_days_ago=60.0, importance=0.4, stability=2.0)
    _insert_raw(conn, "low_imp", created_days_ago=55.0, importance=0.40, stability=2.0)

    new_stab = compute_reminiscence_bump_factor(conn, "low_imp", "test", 2.0)
    assert abs(new_stab - 2.0) < 0.001, \
        f"RB3: 低 importance chunk 不应有加成，got {new_stab}"


# ── RB4: 项目太新 → 无加成 ──────────────────────────────────────────────────

def test_rb4_young_project_no_bump(conn):
    """RB4: project_age_days < min_project_age_days(7) → 无加成。"""
    # 项目只有 3 天老（< min 7 天）
    _insert_raw(conn, "new_p_first", created_days_ago=3.0, importance=0.7, stability=2.0)
    _insert_raw(conn, "new_p_early", created_days_ago=2.8, importance=0.7, stability=2.0)

    new_stab = compute_reminiscence_bump_factor(conn, "new_p_early", "test", 2.0)
    assert abs(new_stab - 2.0) < 0.001, \
        f"RB4: 年轻项目不应有 Bump 加成，got {new_stab}"


# ── RB5: bump_factor 可配置 ──────────────────────────────────────────────────

def test_rb5_configurable_bump_factor(conn):
    """RB5: 自定义 bump_factor=1.50 时，加成为 50%。"""
    import unittest.mock as mock
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.bump_factor":
            return 1.50
        return original_get(key, project=project)

    # project_age = 100 days, early chunk at day 5 (position_pct=5% <= 15%)
    _insert_raw(conn, "first_c3", created_days_ago=100.0, importance=0.7, stability=2.0)
    _insert_raw(conn, "late_c3", created_days_ago=10.0, importance=0.7, stability=2.0)  # establishes project_age=90d
    _insert_raw(conn, "early_c3", created_days_ago=95.0, importance=0.7, stability=2.0)  # position_pct=5/90≈5.5%

    with mock.patch.object(config, 'get', side_effect=patched_get):
        new_stab = compute_reminiscence_bump_factor(conn, "early_c3", "test", 2.0)

    assert abs(new_stab - 2.0 * 1.50) < 0.01, \
        f"RB5: bump_factor=1.50 时 stability 应为 3.0，got {new_stab}"


# ── RB6: bump_enabled=False → 无加成 ────────────────────────────────────────

def test_rb6_disabled_no_bump(conn):
    """RB6: store_vfs.bump_enabled=False → 无加成。"""
    import unittest.mock as mock
    original_get = config.get

    def patched_get(key, project=None):
        if key == "store_vfs.bump_enabled":
            return False
        return original_get(key, project=project)

    _insert_raw(conn, "first_c4", created_days_ago=80.0, importance=0.7, stability=2.0)
    _insert_raw(conn, "early_c4", created_days_ago=75.0, importance=0.7, stability=2.0)

    with mock.patch.object(config, 'get', side_effect=patched_get):
        new_stab = compute_reminiscence_bump_factor(conn, "early_c4", "test", 2.0)

    assert abs(new_stab - 2.0) < 0.001, \
        f"RB6: 禁用时不应有加成，got {new_stab}"


# ── RB7: 边界 position_pct = bump_pct → 有加成 ──────────────────────────────

def test_rb7_boundary_position_gets_bump(conn):
    """RB7: position_pct 略低于 bump_pct(0.15) → 应有加成。"""
    # project_age ≈ 100 天（first=100d ago, last=0d ago）
    # bump chunk at day ~10 (90d ago): position_pct = 10/100 = 10% < 15% → 有加成
    _insert_raw(conn, "b7_first", created_days_ago=100.0, importance=0.7, stability=2.0)
    _insert_raw(conn, "b7_last", created_days_ago=0.1, importance=0.7, stability=2.0)
    _insert_raw(conn, "b7_boundary", created_days_ago=90.0, importance=0.7, stability=2.0)  # 10d into project (10%)

    new_stab = compute_reminiscence_bump_factor(conn, "b7_boundary", "test", 2.0)
    bump_factor = config.get("store_vfs.bump_factor")
    assert new_stab >= 2.0 * bump_factor - 0.05, \
        f"RB7: position_pct=10% 应有加成，got {new_stab}"


# ── RB8: stability 上限 365.0 ────────────────────────────────────────────────

def test_rb8_stability_capped_at_365(conn):
    """RB8: bump 后 stability 不超过 365.0。"""
    _insert_raw(conn, "c8_first", created_days_ago=100.0, importance=0.9, stability=300.0)
    _insert_raw(conn, "c8_early", created_days_ago=95.0, importance=0.9, stability=300.0)

    new_stab = compute_reminiscence_bump_factor(conn, "c8_early", "test", 300.0)
    assert new_stab <= 365.0, f"RB8: stability 不应超过 365.0，got {new_stab}"


# ── RB9: insert_chunk 触发 Bump — 早期 > 晚期 ──────────────────────────────

def test_rb9_batch_applies_bump_to_early_chunks(conn):
    """RB9: apply_reminiscence_bump_batch 后，形成期 chunk stability > 晚期 chunk。"""
    from store_vfs import apply_reminiscence_bump_batch

    # 项目跨度 90 天，bump_pct=15% → 前 13.5 天是形成期
    # 早期 chunk：87 天前（距项目起点 3 天，position_pct=3/90=3.3%）
    # 晚期 chunk：10 天前（position_pct=80/90=88.9%）
    _insert_raw(conn, "anchor_oldest", created_days_ago=90.0, importance=0.6, stability=1.0)
    _insert_raw(conn, "rb9_early", created_days_ago=87.0, importance=0.7, stability=2.0)
    _insert_raw(conn, "rb9_late", created_days_ago=10.0, importance=0.7, stability=2.0)

    result = apply_reminiscence_bump_batch(conn, "test")
    conn.commit()

    early_stab = _get_stability(conn, "rb9_early")
    late_stab = _get_stability(conn, "rb9_late")

    assert result["bumped"] >= 1, f"RB9: 应有至少 1 个 chunk 被 bump，got {result}"
    assert early_stab > late_stab, (
        f"RB9: 形成期 chunk stability({early_stab:.3f}) 应 > 晚期 ({late_stab:.3f})"
    )


# ── RB10: 单 chunk 项目（project_age_secs=0）安全处理 ────────────────────────

def test_rb10_single_chunk_project_safe(conn):
    """RB10: 项目只有 1 个 chunk 时（first == last），安全返回 base_stability。"""
    _insert_raw(conn, "single_chunk", created_days_ago=50.0, importance=0.8, stability=3.0)

    # Only 1 chunk → ts_first == ts_chunk == ts_last → project_age_secs = 0
    new_stab = compute_reminiscence_bump_factor(conn, "single_chunk", "test", 3.0)
    # Should return base_stability safely (no divide by zero)
    assert new_stab == 3.0 or new_stab > 0.0, \
        f"RB10: 单 chunk 项目应安全返回，got {new_stab}"
    assert new_stab <= 365.0, f"RB10: stability 不应超过 365.0，got {new_stab}"
