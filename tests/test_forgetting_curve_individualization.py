"""
test_forgetting_curve_individualization.py — iter400: Forgetting Curve Individualization

覆盖：
  FC1: get_chunk_type_decay — design_constraint 衰减率最低（最持久，≥ 0.98）
  FC2: get_chunk_type_decay — task_state 衰减率较低（< 0.90，较快遗忘）
  FC3: get_chunk_type_decay — 排序：design_constraint > decision > task_state > prompt_context
  FC4: get_chunk_type_decay — 未知 chunk_type 返回默认值（0.92）
  FC5: decay_stability_by_type — design_constraint stability 衰减比 task_state 少
  FC6: decay_stability_by_type — 只衰减 access_count < 2 的 chunk（已访问的受保护）
  FC7: decay_stability_by_type — 只衰减 last_accessed < cutoff 的 chunk
  FC8: sleep_consolidate 中 decayed 子操作使用 per-type 衰减
  FC9: 空 project 安全处理（不抛异常）
  FC10: CHUNK_TYPE_DECAY 表完整性验证（所有主要类型都在表中）

认知科学依据：
  Squire (1992) Memory and Brain:
    程序性记忆（design_constraint/procedure）比情节记忆（task_state）衰减慢。
  Tulving (1972) Episodic vs Semantic Memory:
    语义记忆（概念/约束）比情节记忆（具体事件）持久。
  Anderson et al. (1999) ACT-R:
    基础激活随时间衰减，衰减速率因记忆类型而异。

OS 类比：Linux cgroup memory.reclaim_ratio —
  不同 cgroup 有不同的内存回收速率，而非全局统一 vm.swappiness。
"""
import sys
import sqlite3
import pytest
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "hooks"))

import tmpfs  # noqa
from store_vfs import (
    ensure_schema,
    CHUNK_TYPE_DECAY,
    get_chunk_type_decay,
    decay_stability_by_type,
    sleep_consolidate,
)


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _insert_chunk(conn, chunk_id, chunk_type="decision", stability=10.0,
                  access_count=0, project="test", stale=True):
    """插入测试 chunk，stale=True 时设 last_accessed 为 60 天前。"""
    now = datetime.now(timezone.utc).isoformat()
    if stale:
        last_acc = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
    else:
        last_acc = now
    conn.execute(
        "INSERT OR REPLACE INTO memory_chunks "
        "(id, project, chunk_type, content, summary, importance, retrievability, "
        "stability, access_count, last_accessed, created_at, updated_at) "
        "VALUES (?, ?, ?, 'content', 'summary', 0.7, 0.5, ?, ?, ?, ?, ?)",
        (chunk_id, project, chunk_type, stability, access_count, last_acc, now, now)
    )
    conn.commit()


# ══════════════════════════════════════════════════════════════════════
# 1. get_chunk_type_decay 单元测试
# ══════════════════════════════════════════════════════════════════════

def test_fc1_design_constraint_slowest_decay():
    """design_constraint 衰减率最低（最持久，≥ 0.98）。"""
    decay = get_chunk_type_decay("design_constraint")
    assert decay >= 0.98, (
        f"FC1: design_constraint 衰减率={decay:.4f} 应 >= 0.98（类比长期增强 LTP）"
    )


def test_fc2_task_state_faster_decay():
    """task_state 衰减率低于 0.90（工作记忆快速衰减）。"""
    decay = get_chunk_type_decay("task_state")
    assert decay < 0.90, (
        f"FC2: task_state 衰减率={decay:.4f} 应 < 0.90（情节工作记忆快速遗忘）"
    )


def test_fc3_decay_ordering():
    """衰减率排序：design_constraint ≥ decision ≥ task_state ≥ prompt_context。"""
    dc = get_chunk_type_decay("design_constraint")
    decs = get_chunk_type_decay("decision")
    ts = get_chunk_type_decay("task_state")
    pc = get_chunk_type_decay("prompt_context")

    assert dc >= decs, f"FC3: design_constraint({dc}) 应 >= decision({decs})"
    assert decs > ts, f"FC3: decision({decs}) 应 > task_state({ts})"
    assert ts > pc, f"FC3: task_state({ts}) 应 > prompt_context({pc})"


def test_fc4_unknown_type_returns_default():
    """未知 chunk_type 返回默认值（0.92）。"""
    decay = get_chunk_type_decay("completely_unknown_type")
    assert abs(decay - 0.92) < 0.01, (
        f"FC4: 未知类型应返回默认 0.92，got {decay}"
    )
    decay_empty = get_chunk_type_decay("")
    assert 0.85 <= decay_empty <= 0.95, f"FC4: 空 chunk_type 应有合理默认值，got {decay_empty}"


# ══════════════════════════════════════════════════════════════════════
# 2. decay_stability_by_type 功能测试
# ══════════════════════════════════════════════════════════════════════

def test_fc5_design_constraint_less_decay_than_task_state(conn):
    """
    相同初始 stability，design_constraint 衰减后 stability 高于 task_state。
    """
    _insert_chunk(conn, "fc5_dc", chunk_type="design_constraint", stability=10.0)
    _insert_chunk(conn, "fc5_ts", chunk_type="task_state", stability=10.0)

    decay_stability_by_type(conn, project="test", stale_days=30)
    conn.commit()

    dc_row = conn.execute(
        "SELECT stability FROM memory_chunks WHERE id='fc5_dc'"
    ).fetchone()
    ts_row = conn.execute(
        "SELECT stability FROM memory_chunks WHERE id='fc5_ts'"
    ).fetchone()

    assert dc_row["stability"] > ts_row["stability"], (
        f"FC5: design_constraint stability({dc_row['stability']:.4f}) "
        f"应 > task_state stability({ts_row['stability']:.4f})"
    )


def test_fc6_accessed_chunks_not_decayed(conn):
    """access_count >= 2 的 chunk 不应被衰减（受活跃保护）。"""
    _insert_chunk(conn, "fc6_active", chunk_type="task_state",
                  stability=10.0, access_count=5)  # 高访问次数

    original_stability = conn.execute(
        "SELECT stability FROM memory_chunks WHERE id='fc6_active'"
    ).fetchone()["stability"]

    decay_stability_by_type(conn, project="test", stale_days=30)
    conn.commit()

    new_stability = conn.execute(
        "SELECT stability FROM memory_chunks WHERE id='fc6_active'"
    ).fetchone()["stability"]

    assert new_stability == original_stability, (
        f"FC6: 高访问 chunk 不应被衰减，original={original_stability}, new={new_stability}"
    )


def test_fc7_recent_chunks_not_decayed(conn):
    """last_accessed < 30天的 chunk 不应被衰减（cutoff 保护）。"""
    _insert_chunk(conn, "fc7_recent", chunk_type="task_state",
                  stability=10.0, stale=False)  # last_accessed = now

    original_stability = conn.execute(
        "SELECT stability FROM memory_chunks WHERE id='fc7_recent'"
    ).fetchone()["stability"]

    decay_stability_by_type(conn, project="test", stale_days=30)
    conn.commit()

    new_stability = conn.execute(
        "SELECT stability FROM memory_chunks WHERE id='fc7_recent'"
    ).fetchone()["stability"]

    assert new_stability == original_stability, (
        f"FC7: 最近访问的 chunk 不应被衰减，original={original_stability}, new={new_stability}"
    )


def test_fc5b_quantitative_decay_ratio(conn):
    """
    定量验证：design_constraint stability × 0.99，task_state × 0.85。
    """
    initial = 10.0
    _insert_chunk(conn, "fc5b_dc", chunk_type="design_constraint", stability=initial)
    _insert_chunk(conn, "fc5b_ts", chunk_type="task_state", stability=initial)
    _insert_chunk(conn, "fc5b_pc", chunk_type="prompt_context", stability=initial)

    decay_stability_by_type(conn, project="test", stale_days=30)
    conn.commit()

    dc_s = conn.execute(
        "SELECT stability FROM memory_chunks WHERE id='fc5b_dc'"
    ).fetchone()["stability"]
    ts_s = conn.execute(
        "SELECT stability FROM memory_chunks WHERE id='fc5b_ts'"
    ).fetchone()["stability"]
    pc_s = conn.execute(
        "SELECT stability FROM memory_chunks WHERE id='fc5b_pc'"
    ).fetchone()["stability"]

    # design_constraint: initial × 0.99 = 9.9
    assert abs(dc_s - initial * 0.99) < 0.01, (
        f"FC5b: design_constraint stability 应 ≈ {initial * 0.99:.2f}, got {dc_s:.4f}"
    )
    # task_state: initial × 0.85 = 8.5
    assert abs(ts_s - initial * 0.85) < 0.01, (
        f"FC5b: task_state stability 应 ≈ {initial * 0.85:.2f}, got {ts_s:.4f}"
    )
    # prompt_context: initial × 0.70 = 7.0
    assert abs(pc_s - initial * 0.70) < 0.01, (
        f"FC5b: prompt_context stability 应 ≈ {initial * 0.70:.2f}, got {pc_s:.4f}"
    )


# ══════════════════════════════════════════════════════════════════════
# 3. sleep_consolidate 集成测试
# ══════════════════════════════════════════════════════════════════════

def test_fc8_sleep_consolidate_uses_per_type_decay(conn):
    """
    sleep_consolidate 调用后，design_constraint stability 比 task_state 衰减少。
    """
    _insert_chunk(conn, "fc8_dc", chunk_type="design_constraint", stability=10.0)
    _insert_chunk(conn, "fc8_ts", chunk_type="task_state", stability=10.0)

    result = sleep_consolidate(conn, project="test", stale_days=30)
    conn.commit()

    assert "decayed" in result, "FC8: sleep_consolidate 应返回 decayed 计数"
    assert result["decayed"] >= 0

    dc_row = conn.execute(
        "SELECT stability FROM memory_chunks WHERE id='fc8_dc'"
    ).fetchone()
    ts_row = conn.execute(
        "SELECT stability FROM memory_chunks WHERE id='fc8_ts'"
    ).fetchone()

    assert dc_row["stability"] > ts_row["stability"], (
        f"FC8: sleep_consolidate 后 design_constraint({dc_row['stability']:.4f}) "
        f"应 > task_state({ts_row['stability']:.4f})"
    )


# ══════════════════════════════════════════════════════════════════════
# 4. 边界条件
# ══════════════════════════════════════════════════════════════════════

def test_fc9_empty_project_no_exception(conn):
    """空 project 参数不抛异常，安全降级。"""
    _insert_chunk(conn, "fc9_chunk", chunk_type="task_state", stability=5.0)
    try:
        n = decay_stability_by_type(conn, project=None, stale_days=30)
        assert isinstance(n, int), "FC9: 应返回 int"
    except Exception as e:
        pytest.fail(f"FC9: 空 project 不应抛异常，got {e}")


def test_fc10_chunk_type_decay_table_completeness():
    """CHUNK_TYPE_DECAY 包含所有主要 chunk_type。"""
    required_types = [
        "design_constraint", "decision", "procedure",
        "task_state", "prompt_context",
    ]
    for t in required_types:
        assert t in CHUNK_TYPE_DECAY, f"FC10: {t} 应在 CHUNK_TYPE_DECAY 中"

    # 所有值应在 (0.0, 1.0] 范围内
    for t, v in CHUNK_TYPE_DECAY.items():
        assert 0.0 < v <= 1.0, f"FC10: CHUNK_TYPE_DECAY[{t}]={v} 应在 (0.0, 1.0]"


def test_fc_stability_min_floor(conn):
    """stability 不应低于 0.1（MIN(0.1, ...) 下限保护）。"""
    # 设置极小 stability 后衰减
    now = datetime.now(timezone.utc).isoformat()
    last_acc = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO memory_chunks "
        "(id, project, chunk_type, content, summary, importance, retrievability, "
        "stability, access_count, last_accessed, created_at, updated_at) "
        "VALUES (?, ?, ?, 'c', 's', 0.5, 0.5, ?, 0, ?, ?, ?)",
        ("fc_floor_chunk", "test", "task_state", 0.15, last_acc, now, now)
    )
    conn.commit()

    decay_stability_by_type(conn, project="test", stale_days=30)
    conn.commit()

    row = conn.execute(
        "SELECT stability FROM memory_chunks WHERE id='fc_floor_chunk'"
    ).fetchone()
    assert row["stability"] >= 0.1, (
        f"FC: stability 不应低于 0.1，got {row['stability']}"
    )
