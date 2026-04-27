#!/usr/bin/env python3
"""
memory-os Context Pressure Governor 测试套件 — test_governor.py

迭代55：OS 类比 — Linux TCP Congestion Control (BBR) 验证
验证 governor 根据多信号正确判定压力等级并输出缩放因子。

测试维度：
1. 基本分级：LOW / NORMAL / HIGH / CRITICAL
2. 对话轮次驱动的压力升级
3. compaction 次数驱动的压力升级
4. 近期 compaction 时间窗口检测
5. 连续高压追踪（consecutive_high → CRITICAL 升级）
6. 缩放因子正确性（与 sysctl tunable 一致）
7. 持久状态读写
8. 边界条件（空数据库、无 dmesg 表）
9. retriever 集成（effective_max_chars 缩放）
"""
import sys
import os
import json
import time
import tempfile
import sqlite3
from pathlib import Path
from datetime import datetime, timezone, timedelta

# 将 memory-os 根目录加入 path
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

import tmpfs  # noqa: F401 — tmpfs isolation (iter54)
from store import (
    open_db, ensure_schema, dmesg_log, DMESG_INFO,
    context_pressure_governor, GOV_LOW, GOV_NORMAL, GOV_HIGH, GOV_CRITICAL,
    _governor_load_state, _governor_save_state, _GOVERNOR_STATE_FILE,
    MEMORY_OS_DIR,
)
from config import get as _cfg


def _setup():
    """创建测试 DB，清理前序测试残留"""
    conn = open_db()
    ensure_schema(conn)
    # 清理 dmesg 残留（防止跨测试 compaction 计数污染）
    try:
        conn.execute("DELETE FROM dmesg WHERE subsystem = 'swap_in'")
        conn.commit()
    except Exception:
        pass
    return conn


def _teardown(conn):
    conn.close()


def _insert_recall_traces(conn, project, session_id, count):
    """插入 N 条 recall_traces 模拟对话轮次"""
    import uuid
    for i in range(count):
        ts = (datetime.now(timezone.utc) - timedelta(minutes=count - i)).isoformat()
        conn.execute("""
            INSERT OR IGNORE INTO recall_traces
            (id, timestamp, project, session_id, prompt_hash, candidates_count, top_k_json, injected)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (str(uuid.uuid4()), ts, project, session_id,
              f"hash_{i}", 5, "[]", 1))
    conn.commit()


def _insert_swap_in_logs(conn, count, age_secs=0):
    """插入 N 条 swap_in dmesg 日志模拟 compaction 历史"""
    for i in range(count):
        ts = (datetime.now(timezone.utc) - timedelta(seconds=age_secs + i * 60)).isoformat()
        dmesg_log(conn, DMESG_INFO, "swap_in",
                  f"PostCompact swap in: test {i}",
                  extra={"session": "test"})
    conn.commit()


# ═══════════════════════════════════════════════
# 测试 1：LOW 压力 — 新会话，少轮次
# ═══════════════════════════════════════════════
def test_low_pressure():
    """新会话 (≤5 turns) 应判定为 LOW"""
    conn = _setup()
    project = "test_gov_low"
    session_id = "sess_low"

    # 清理 governor 状态
    _governor_save_state({})

    # 只有 3 轮对话
    _insert_recall_traces(conn, project, session_id, 3)

    result = context_pressure_governor(conn, project, session_id=session_id)
    assert result["level"] == GOV_LOW, f"期望 LOW，实际 {result['level']}"
    assert result["scale"] == _cfg("governor.scale_low"), \
        f"scale 应为 {_cfg('governor.scale_low')}，实际 {result['scale']}"
    assert result["turns"] == 3
    assert "fresh" in result["reason"]

    _teardown(conn)
    return True


# ═══════════════════════════════════════════════
# 测试 2：NORMAL 压力 — 中等轮次
# ═══════════════════════════════════════════════
def test_normal_pressure():
    """中等轮次 (>5, <15) 应判定为 NORMAL"""
    conn = _setup()
    project = "test_gov_normal"
    session_id = "sess_normal"
    _governor_save_state({})

    _insert_recall_traces(conn, project, session_id, 10)

    result = context_pressure_governor(conn, project, session_id=session_id)
    assert result["level"] == GOV_NORMAL, f"期望 NORMAL，实际 {result['level']}"
    assert result["scale"] == 1.0
    assert result["turns"] == 10

    _teardown(conn)
    return True


# ═══════════════════════════════════════════════
# 测试 3：HIGH 压力 — 高轮次
# ═══════════════════════════════════════════════
def test_high_pressure_turns():
    """高轮次 (≥15) 应判定为 HIGH"""
    conn = _setup()
    project = "test_gov_high"
    session_id = "sess_high"
    _governor_save_state({})

    _insert_recall_traces(conn, project, session_id, 18)

    result = context_pressure_governor(conn, project, session_id=session_id)
    assert result["level"] == GOV_HIGH, f"期望 HIGH，实际 {result['level']}"
    assert result["scale"] == _cfg("governor.scale_high")
    assert result["turns"] == 18

    _teardown(conn)
    return True


# ═══════════════════════════════════════════════
# 测试 4：HIGH 压力 — compaction 次数驱动
# ═══════════════════════════════════════════════
def test_high_pressure_compactions():
    """≥2 次 compaction 应判定为 HIGH"""
    conn = _setup()
    project = "test_gov_compact"
    session_id = "sess_compact"
    _governor_save_state({})

    # 只有 3 轮对话（本身是 LOW），但有 2 次 compaction
    _insert_recall_traces(conn, project, session_id, 3)
    _insert_swap_in_logs(conn, 2, age_secs=300)  # 5 分钟前

    result = context_pressure_governor(conn, project, session_id=session_id)
    assert result["level"] == GOV_HIGH, f"期望 HIGH，实际 {result['level']}"
    assert result["compactions"] == 2

    _teardown(conn)
    return True


# ═══════════════════════════════════════════════
# 测试 5：CRITICAL 压力 — 多次 compaction
# ═══════════════════════════════════════════════
def test_critical_pressure_compactions():
    """≥4 次 compaction 应判定为 CRITICAL"""
    conn = _setup()
    project = "test_gov_crit"
    session_id = "sess_crit"
    _governor_save_state({})

    _insert_recall_traces(conn, project, session_id, 5)
    _insert_swap_in_logs(conn, 5, age_secs=600)

    result = context_pressure_governor(conn, project, session_id=session_id)
    assert result["level"] == GOV_CRITICAL, f"期望 CRITICAL，实际 {result['level']}"
    assert result["scale"] == _cfg("governor.scale_critical")

    _teardown(conn)
    return True


# ═══════════════════════════════════════════════
# 测试 6：CRITICAL — consecutive_high 升级
# ═══════════════════════════════════════════════
def test_consecutive_high_escalation():
    """连续 3 次 HIGH → CRITICAL 升级"""
    conn = _setup()
    project = "test_gov_consec"
    session_id = "sess_consec"

    # 模拟 consecutive_high=3
    _governor_save_state({"consecutive_high": 3})

    _insert_recall_traces(conn, project, session_id, 5)

    result = context_pressure_governor(conn, project, session_id=session_id)
    assert result["level"] == GOV_CRITICAL, f"期望 CRITICAL，实际 {result['level']}"
    assert "consecutive_high=3" in result["reason"]

    _teardown(conn)
    return True


# ═══════════════════════════════════════════════
# 测试 7：缩放因子范围验证
# ═══════════════════════════════════════════════
def test_scale_ranges():
    """所有 scale 值在合理范围内"""
    scale_low = _cfg("governor.scale_low")
    scale_high = _cfg("governor.scale_high")
    scale_critical = _cfg("governor.scale_critical")

    assert 1.0 <= scale_low <= 3.0, f"scale_low={scale_low} 应在 [1.0, 3.0]"
    assert 0.2 <= scale_high <= 1.0, f"scale_high={scale_high} 应在 [0.2, 1.0]"
    assert 0.1 <= scale_critical <= 0.8, f"scale_critical={scale_critical} 应在 [0.1, 0.8]"
    assert scale_low > 1.0 > scale_high > scale_critical, \
        "scale 应递减：LOW > NORMAL > HIGH > CRITICAL"

    return True


# ═══════════════════════════════════════════════
# 测试 8：持久状态读写
# ═══════════════════════════════════════════════
def test_persistent_state():
    """governor 状态应正确持久化和恢复"""
    test_state = {
        "level": "HIGH",
        "scale": 0.6,
        "turns": 20,
        "compactions": 3,
        "consecutive_high": 2,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    _governor_save_state(test_state)
    loaded = _governor_load_state()

    assert loaded["level"] == "HIGH"
    assert loaded["consecutive_high"] == 2
    assert loaded["turns"] == 20

    # 清理
    _governor_save_state({})
    return True


# ═══════════════════════════════════════════════
# 测试 9：空数据库/初始状态
# ═══════════════════════════════════════════════
def test_empty_database():
    """空数据库应返回 LOW（无轮次、无 compaction）"""
    conn = _setup()
    _governor_save_state({})

    result = context_pressure_governor(conn, "nonexistent_project", session_id="new")
    assert result["level"] == GOV_LOW, f"空数据库应为 LOW，实际 {result['level']}"
    assert result["turns"] == 0
    assert result["compactions"] == 0

    _teardown(conn)
    return True


# ═══════════════════════════════════════════════
# 测试 10：effective_max_chars 缩放验证
# ═══════════════════════════════════════════════
def test_effective_chars_scaling():
    """验证 retriever 中 effective_max_chars 按 governor scale 缩放"""
    base_chars = _cfg("retriever.max_context_chars")  # 800 (iter72 updated from 600→800)

    # LOW: base * 1.5
    scale_low = _cfg("governor.scale_low")
    scaled_low = int(base_chars * scale_low)
    expected_low = int(base_chars * scale_low)
    assert scaled_low == expected_low, f"LOW 缩放后应为 {expected_low}，实际 {scaled_low}"

    # HIGH: base * 0.6
    scale_high = _cfg("governor.scale_high")
    scaled_high = int(base_chars * scale_high)
    expected_high = int(base_chars * scale_high)
    assert scaled_high == expected_high, f"HIGH 缩放后应为 {expected_high}，实际 {scaled_high}"

    # CRITICAL: base * 0.3
    scale_critical = _cfg("governor.scale_critical")
    scaled_critical = int(base_chars * scale_critical)
    expected_critical = int(base_chars * scale_critical)
    assert scaled_critical == expected_critical, f"CRITICAL 缩放后应为 {expected_critical}，实际 {scaled_critical}"
    assert scaled_critical >= 150, "CRITICAL 缩放后不应低于 150 下限"

    return True


# ═══════════════════════════════════════════════
# 测试 11：近期 compaction 时间窗口
# ═══════════════════════════════════════════════
def test_recent_compaction_window():
    """compaction 后 120 秒内应判定为 HIGH"""
    conn = _setup()
    project = "test_gov_recent"
    session_id = "sess_recent"
    _governor_save_state({})

    # 只有 1 次 compaction，但是刚刚发生（60 秒前）
    _insert_recall_traces(conn, project, session_id, 3)
    # 直接写 dmesg，时间戳设为 60 秒前
    ts_recent = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()
    conn.execute("""
        INSERT INTO dmesg (timestamp, level, subsystem, message, extra)
        VALUES (?, ?, 'swap_in', 'recent compaction test', '{}')
    """, (ts_recent, DMESG_INFO))
    conn.commit()

    result = context_pressure_governor(conn, project, session_id=session_id)
    assert result["level"] == GOV_HIGH, f"近期 compaction 应为 HIGH，实际 {result['level']}"
    assert "recent_compact" in result["reason"]
    assert 0 < result["secs_since_compact"] < 120

    _teardown(conn)
    return True


# ═══════════════════════════════════════════════
# 测试 12：consecutive_high 重置
# ═══════════════════════════════════════════════
def test_consecutive_high_reset():
    """压力降低时 consecutive_high 应重置为 0"""
    conn = _setup()
    project = "test_gov_reset"
    session_id = "sess_reset"

    # 先模拟 consecutive_high=2
    _governor_save_state({"consecutive_high": 2})

    # 只有 3 轮（LOW 压力），应重置 consecutive_high
    _insert_recall_traces(conn, project, session_id, 3)

    result = context_pressure_governor(conn, project, session_id=session_id)
    assert result["level"] == GOV_LOW, f"低压力应为 LOW，实际 {result['level']}"
    assert result["consecutive_high"] == 0, \
        f"consecutive_high 应重置为 0，实际 {result['consecutive_high']}"

    _teardown(conn)
    return True


# ═══════════════════════════════════════════════
# 测试 13：性能 — governor 延迟
# ═══════════════════════════════════════════════
def test_governor_performance():
    """governor 延迟应 < 5ms"""
    conn = _setup()
    project = "test_gov_perf"
    session_id = "sess_perf"
    _governor_save_state({})

    _insert_recall_traces(conn, project, session_id, 10)

    iterations = 100
    t0 = time.time()
    for _ in range(iterations):
        context_pressure_governor(conn, project, session_id=session_id)
    elapsed_ms = (time.time() - t0) * 1000

    avg_ms = elapsed_ms / iterations
    assert avg_ms < 5.0, f"governor 平均延迟应 < 5ms，实际 {avg_ms:.2f}ms"

    _teardown(conn)
    return avg_ms


# ═══════════════════════════════════════════════
# 测试 14：时间窗口 — 窗口外 compaction 不计数（迭代63）
# ═══════════════════════════════════════════════
def test_window_excludes_old_compactions():
    """窗口外（>2h 前）的 compaction 记录不应计入压力"""
    conn = _setup()
    project = "test_gov_window"
    session_id = "sess_window"
    _governor_save_state({})

    _insert_recall_traces(conn, project, session_id, 3)

    # 插入 10 条 swap_in 记录，时间戳在 5 小时前（超出 2h 窗口）
    for i in range(10):
        old_ts = (datetime.now(timezone.utc) - timedelta(hours=5, minutes=i)).isoformat()
        conn.execute("""
            INSERT INTO dmesg (timestamp, level, subsystem, message, extra)
            VALUES (?, ?, 'swap_in', ?, '{}')
        """, (old_ts, DMESG_INFO, f"old compaction {i}"))
    conn.commit()

    result = context_pressure_governor(conn, project, session_id=session_id)
    # 旧 compaction 不应计入 → compactions=0 → 3 turns → LOW
    assert result["compactions"] == 0, \
        f"窗口外 compaction 不应计入，实际 compactions={result['compactions']}"
    assert result["level"] == GOV_LOW, \
        f"无近期 compaction 且低轮次应为 LOW，实际 {result['level']}"

    _teardown(conn)
    return True


# ═══════════════════════════════════════════════
# 测试 15：时间窗口 — 窗口外 turns 不计数（迭代63）
# ═══════════════════════════════════════════════
def test_window_excludes_old_turns():
    """窗口外的 recall_traces 不应计入 turns"""
    conn = _setup()
    project = "test_gov_old_turns"
    session_id = "sess_old_turns"
    _governor_save_state({})

    import uuid
    # 插入 30 条旧的 recall_traces（6h 前），远超 turns_critical=25
    for i in range(30):
        old_ts = (datetime.now(timezone.utc) - timedelta(hours=6, minutes=i)).isoformat()
        conn.execute("""
            INSERT OR IGNORE INTO recall_traces
            (id, timestamp, project, session_id, prompt_hash, candidates_count, top_k_json, injected)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (str(uuid.uuid4()), old_ts, project, session_id,
              f"old_hash_{i}", 5, "[]", 1))
    conn.commit()

    result = context_pressure_governor(conn, project, session_id=session_id)
    # 旧 turns 不应计入 → turns=0 → LOW
    assert result["turns"] == 0, \
        f"窗口外 turns 不应计入，实际 turns={result['turns']}"
    assert result["level"] == GOV_LOW, \
        f"无近期活动应为 LOW，实际 {result['level']}"

    _teardown(conn)
    return True


# ═══════════════════════════════════════════════
# 测试 16：consecutive_high 衰减（迭代63）
# ═══════════════════════════════════════════════
def test_consecutive_high_time_decay():
    """超过 decay_hours 未更新时 consecutive_high 应自动 reset"""
    conn = _setup()
    project = "test_gov_decay"
    session_id = "sess_decay"

    # 模拟 2 小时前的 consecutive_high=5（远超 3 的升级阈值）
    old_ts = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    _governor_save_state({
        "consecutive_high": 5,
        "updated_at": old_ts,
    })

    # 只有 3 轮对话 — 本身是 LOW
    _insert_recall_traces(conn, project, session_id, 3)

    result = context_pressure_governor(conn, project, session_id=session_id)
    # consecutive_high 应被衰减 reset → 不应触发 CRITICAL
    assert result["level"] == GOV_LOW, \
        f"consecutive_high 衰减后应为 LOW，实际 {result['level']}"
    assert result["consecutive_high"] == 0, \
        f"consecutive_high 应 reset 为 0，实际 {result['consecutive_high']}"

    _teardown(conn)
    return True


# ═══════════════════════════════════════════════
# 测试 17：混合窗口 — 窗口内+窗口外共存（迭代63）
# ═══════════════════════════════════════════════
def test_mixed_window_compactions():
    """只有窗口内的 compaction 应被计入"""
    conn = _setup()
    project = "test_gov_mixed"
    session_id = "sess_mixed"
    _governor_save_state({})

    _insert_recall_traces(conn, project, session_id, 3)

    # 10 条旧（5h 前）+ 2 条新（10min 前）
    for i in range(10):
        old_ts = (datetime.now(timezone.utc) - timedelta(hours=5, minutes=i)).isoformat()
        conn.execute("""
            INSERT INTO dmesg (timestamp, level, subsystem, message, extra)
            VALUES (?, ?, 'swap_in', ?, '{}')
        """, (old_ts, DMESG_INFO, f"old compaction {i}"))

    for i in range(2):
        new_ts = (datetime.now(timezone.utc) - timedelta(minutes=2 + i)).isoformat()
        conn.execute("""
            INSERT INTO dmesg (timestamp, level, subsystem, message, extra)
            VALUES (?, ?, 'swap_in', ?, '{}')
        """, (new_ts, DMESG_INFO, f"new compaction {i}"))
    conn.commit()

    result = context_pressure_governor(conn, project, session_id=session_id)
    # 只有 2 条窗口内（10min burst）→ burst=2 ≥ compact_high=2 → HIGH
    assert result["level"] == GOV_HIGH, \
        f"2 次近期 compaction（burst）应为 HIGH，实际 {result['level']}"

    _teardown(conn)
    return True


# ═══════════════════════════════════════════════
# 运行所有测试
# ═══════════════════════════════════════════════
def run_all():
    tests = [
        ("LOW 压力 — 新会话", test_low_pressure),
        ("NORMAL 压力 — 中等轮次", test_normal_pressure),
        ("HIGH 压力 — 高轮次", test_high_pressure_turns),
        ("HIGH 压力 — compaction 驱动", test_high_pressure_compactions),
        ("CRITICAL 压力 — 多次 compaction", test_critical_pressure_compactions),
        ("CRITICAL — consecutive_high 升级", test_consecutive_high_escalation),
        ("缩放因子范围验证", test_scale_ranges),
        ("持久状态读写", test_persistent_state),
        ("空数据库/初始状态", test_empty_database),
        ("effective_max_chars 缩放", test_effective_chars_scaling),
        ("近期 compaction 时间窗口", test_recent_compaction_window),
        ("consecutive_high 重置", test_consecutive_high_reset),
        ("性能 — governor 延迟", test_governor_performance),
        ("时间窗口 — 窗口外 compaction 不计数（迭代63）", test_window_excludes_old_compactions),
        ("时间窗口 — 窗口外 turns 不计数（迭代63）", test_window_excludes_old_turns),
        ("consecutive_high 衰减（迭代63）", test_consecutive_high_time_decay),
        ("混合窗口 — 窗口内+窗口外共存（迭代63）", test_mixed_window_compactions),
    ]

    passed = 0
    failed = 0
    t0 = time.time()

    for name, fn in tests:
        try:
            result = fn()
            if result is True or isinstance(result, float):
                perf_info = f" ({result:.2f}ms/call)" if isinstance(result, float) else ""
                print(f"  ✅ {name}{perf_info}")
                passed += 1
            else:
                print(f"  ❌ {name}: returned {result}")
                failed += 1
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    elapsed = (time.time() - t0) * 1000
    print(f"\n{'='*50}")
    print(f"Governor: {passed}/{passed+failed} passed, {elapsed:.1f}ms")
    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
