"""
iter551: initcall_debug — Boot Subsystem Latency Instrumentation
OS 类比：Linux initcall_debug (Arjan van de Ven, 2008, kernel 2.6.24)

测试覆盖：
  1. _InitcallTimer 基本计时功能
  2. _InitcallTimer.probe() 异常抑制（fault isolation）
  3. initcall_debug() 分析：排序、top_n、blame_line
  4. initcall_debug() 空输入
  5. initcall_debug() 全失败场景
  6. Milestone-based timing 集成（模拟 loader.py 方式）
  7. Config tunable 门控
"""
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from store_mm import _InitcallTimer, initcall_debug


# ── Test 1: _InitcallTimer 基本计时 ──

def test_timer_basic():
    """_InitcallTimer 记录子系统名称和耗时。"""
    timer = _InitcallTimer()
    with timer.probe("subsys_a"):
        time.sleep(0.002)
    with timer.probe("subsys_b"):
        time.sleep(0.005)

    assert len(timer.timings) == 2
    assert timer.timings[0][0] == "subsys_a"
    assert timer.timings[1][0] == "subsys_b"
    # elapsed_ms > 0
    assert timer.timings[0][1] > 0
    assert timer.timings[1][1] > timer.timings[0][1]  # subsys_b slower
    # ok = True
    assert timer.timings[0][2] is True
    assert timer.timings[1][2] is True


# ── Test 2: probe() 异常抑制 ──

def test_timer_exception_suppression():
    """probe() 的 __exit__ suppress 异常，不影响后续子系统。"""
    timer = _InitcallTimer()
    with timer.probe("will_fail"):
        raise RuntimeError("simulated failure")
    # 不应该抛出异常
    with timer.probe("after_fail"):
        pass

    assert len(timer.timings) == 2
    assert timer.timings[0][0] == "will_fail"
    assert timer.timings[0][2] is False  # ok = False
    assert timer.timings[1][0] == "after_fail"
    assert timer.timings[1][2] is True


# ── Test 3: initcall_debug() 排序和 top_n ──

def test_initcall_debug_sorting():
    """initcall_debug 按耗时降序排列，返回 top_n。"""
    timings = [
        ("fast", 1.0, True),
        ("medium", 5.0, True),
        ("slow", 20.0, True),
        ("very_slow", 50.0, True),
        ("fastest", 0.1, True),
    ]
    result = initcall_debug(timings, top_n=3)

    assert result["subsystem_count"] == 5
    assert result["total_ms"] == 76.1
    assert len(result["top_slow"]) == 3
    assert result["top_slow"][0]["name"] == "very_slow"
    assert result["top_slow"][1]["name"] == "slow"
    assert result["top_slow"][2]["name"] == "medium"


# ── Test 4: initcall_debug() 空输入 ──

def test_initcall_debug_empty():
    """空 timings 返回零值结果。"""
    result = initcall_debug([])
    assert result["total_ms"] == 0
    assert result["subsystem_count"] == 0
    assert result["top_slow"] == []
    assert result["failed"] == []
    assert result["blame_line"] == ""
    assert result["below_1ms"] == 0


# ── Test 5: initcall_debug() 全失败场景 ──

def test_initcall_debug_all_failed():
    """所有子系统失败时，failed 列表正确填充。"""
    timings = [
        ("a", 2.0, False),
        ("b", 3.0, False),
    ]
    result = initcall_debug(timings)
    assert len(result["failed"]) == 2
    assert result["failed"][0]["name"] == "a"
    assert result["failed"][1]["name"] == "b"
    assert "FAILED" in result["blame_line"]


# ── Test 6: Milestone-based timing 集成 ──

def test_milestone_based_timing():
    """模拟 loader.py 的 milestone 方式计时。"""
    import time as _ict_time
    _ict_milestones = [("_boot_start", _ict_time.time())]

    _ict_milestones.append(("hook_analyzer", _ict_time.time()))
    time.sleep(0.003)
    _ict_milestones.append(("watchdog", _ict_time.time()))
    time.sleep(0.008)
    _ict_milestones.append(("autotune", _ict_time.time()))
    time.sleep(0.002)
    _ict_milestones.append(("_boot_end", _ict_time.time()))

    # 从 milestones 计算 timings（复制 loader.py 的逻辑）
    _ict_timings = []
    for i in range(len(_ict_milestones) - 1):
        name = _ict_milestones[i][0]
        if name.startswith("_"):
            continue
        elapsed_ms = (_ict_milestones[i + 1][1] - _ict_milestones[i][1]) * 1000
        _ict_timings.append((name, round(elapsed_ms, 2), True))

    result = initcall_debug(_ict_timings)
    assert result["subsystem_count"] == 3
    assert result["top_slow"][0]["name"] == "watchdog"  # slowest
    assert result["total_ms"] > 10  # at least 13ms from sleeps


# ── Test 7: Config tunable 门控 ──

def test_config_tunable():
    """initcall_debug.enabled 和 initcall_debug.top_n 存在于 config 注册表中。"""
    from config import get as _cfg
    assert _cfg("initcall_debug.enabled") is True
    assert _cfg("initcall_debug.top_n") == 5


# ── Test 8: below_1ms 计数 ──

def test_below_1ms_count():
    """below_1ms 正确统计耗时 < 1ms 的子系统。"""
    timings = [
        ("fast1", 0.1, True),
        ("fast2", 0.5, True),
        ("slow", 10.0, True),
        ("fast3", 0.9, True),
    ]
    result = initcall_debug(timings)
    assert result["below_1ms"] == 3


# ── Test 9: blame_line 格式 ──

def test_blame_line_format():
    """blame_line 不包含 < 1ms 的子系统。"""
    timings = [
        ("tiny", 0.3, True),
        ("big", 50.0, True),
    ]
    result = initcall_debug(timings)
    assert "big=50ms" in result["blame_line"]
    assert "tiny" not in result["blame_line"]  # < 1ms excluded from blame


# ── Run All ──

if __name__ == "__main__":
    tests = [
        test_timer_basic,
        test_timer_exception_suppression,
        test_initcall_debug_sorting,
        test_initcall_debug_empty,
        test_initcall_debug_all_failed,
        test_milestone_based_timing,
        test_config_tunable,
        test_below_1ms_count,
        test_blame_line_format,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
            print(f"  PASS: {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL: {t.__name__} — {e}")
        except Exception as e:
            print(f"  ERROR: {t.__name__} — {type(e).__name__}: {e}")

    print(f"\n{passed}/{len(tests)} tests passed")
    if passed < len(tests):
        sys.exit(1)
