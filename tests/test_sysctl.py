#!/usr/bin/env python3
"""
迭代27 测试：sysctl Runtime Tunables Registry
验证 config.py 的注册表、读取优先级、运行时修改、范围校验
"""
import json
import os
import sys
import time
import tempfile
from pathlib import Path

# 确保导入本地模块
sys.path.insert(0, str(Path(__file__).parent))

passed = 0
failed = 0


def _assert_test(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


def run_tests():
    global passed, failed
    t0 = time.time()

    # ── 1. 基本注册表读取（默认值）──
    import config
    # 清除可能存在的环境变量覆盖
    for key in list(os.environ.keys()):
        if key.startswith("MEMORY_OS_"):
            del os.environ[key]
    config._invalidate_cache()

    val = config.get("retriever.top_k")
    _assert_test("默认值 retriever.top_k=5", val == 5, f"got {val}")

    val = config.get("writer.debounce_secs")
    _assert_test("默认值 writer.debounce_secs=300", val == 300, f"got {val}")

    val = config.get("extractor.chunk_quota")
    _assert_test("默认值 extractor.chunk_quota=200", val == 200, f"got {val}")

    val = config.get("router.min_score")
    _assert_test("默认值 router.min_score=0.01", val == 0.01, f"got {val}")

    val = config.get("scorer.importance_decay_rate")
    _assert_test("默认值 scorer.importance_decay_rate=0.95", val == 0.95, f"got {val}")

    # ── 2. 环境变量覆盖（最高优先级）──
    os.environ["MEMORY_OS_RETRIEVER_TOP_K"] = "7"
    config._invalidate_cache()
    val = config.get("retriever.top_k")
    _assert_test("环境变量覆盖 retriever.top_k=7", val == 7, f"got {val}")
    del os.environ["MEMORY_OS_RETRIEVER_TOP_K"]

    # 向后兼容 env_key: MEMORY_OS_CHUNK_QUOTA
    os.environ["MEMORY_OS_CHUNK_QUOTA"] = "500"
    config._invalidate_cache()
    val = config.get("extractor.chunk_quota")
    _assert_test("向后兼容 MEMORY_OS_CHUNK_QUOTA=500", val == 500, f"got {val}")
    del os.environ["MEMORY_OS_CHUNK_QUOTA"]

    # ── 3. 范围校验（clamp）──
    os.environ["MEMORY_OS_RETRIEVER_TOP_K"] = "999"
    config._invalidate_cache()
    val = config.get("retriever.top_k")
    _assert_test("范围钳位 max=20", val == 20, f"got {val}")
    del os.environ["MEMORY_OS_RETRIEVER_TOP_K"]

    os.environ["MEMORY_OS_RETRIEVER_TOP_K"] = "0"
    config._invalidate_cache()
    val = config.get("retriever.top_k")
    _assert_test("范围钳位 min=1", val == 1, f"got {val}")
    del os.environ["MEMORY_OS_RETRIEVER_TOP_K"]

    # ── 4. sysctl_set + sysctl.json 持久化 ──
    # 使用临时文件避免污染真实配置
    original_file = config.SYSCTL_FILE
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    config.SYSCTL_FILE = tmp_path
    config._invalidate_cache()

    try:
        config.sysctl_set("retriever.top_k", 8)
        val = config.get("retriever.top_k")
        _assert_test("sysctl_set 运行时修改", val == 8, f"got {val}")

        # 验证持久化
        disk_data = json.loads(tmp_path.read_text())
        _assert_test("sysctl.json 持久化", disk_data.get("retriever.top_k") == 8, f"got {disk_data}")

        # 验证 sysctl_set 范围钳位
        config.sysctl_set("retriever.top_k", 100)
        val = config.get("retriever.top_k")
        _assert_test("sysctl_set 范围钳位", val == 20, f"got {val}")
    finally:
        config.SYSCTL_FILE = original_file
        config._invalidate_cache()
        try:
            tmp_path.unlink()
        except Exception:
            pass

    # ── 5. sysctl_list 完整性 ──
    config._invalidate_cache()
    listing = config.sysctl_list()
    _assert_test(f"sysctl_list 返回 {len(config._REGISTRY)} 个 tunable", len(listing) == len(config._REGISTRY), f"got {len(listing)}")

    # 检查每项都有必需字段
    all_have_fields = all(
        "value" in v and "default" in v and "type" in v and "range" in v and "description" in v
        for v in listing.values()
    )
    _assert_test("sysctl_list 字段完整", all_have_fields)

    # ── 6. 未知 key 抛异常 ──
    try:
        config.get("nonexistent.key")
        _assert_test("未知 key 抛 KeyError", False, "no exception raised")
    except KeyError:
        _assert_test("未知 key 抛 KeyError", True)

    # ── 7. scorer.py 集成验证 ──
    from scorer import importance_with_decay, access_bonus
    from datetime import datetime, timezone

    now_iso = datetime.now(timezone.utc).isoformat()
    # 使用默认 sysctl 值，验证 scorer 函数正常工作
    result = importance_with_decay(0.8, now_iso)
    _assert_test("scorer importance_with_decay 集成", 0.75 < result <= 0.85, f"got {result}")

    result = access_bonus(10)
    _assert_test("scorer access_bonus 集成", 0.0 < result <= 0.2, f"got {result}")

    # ── 8. 读取性能（sysctl 调用应 < 0.1ms）──
    t1 = time.time()
    for _ in range(1000):
        config.get("retriever.top_k")
    elapsed_ms = (time.time() - t1) * 1000
    avg_us = elapsed_ms  # 1000次总耗时 ms ≈ 每次 μs
    _assert_test(f"性能: 1000次 get() = {elapsed_ms:.1f}ms", elapsed_ms < 100, f"avg {avg_us:.0f}μs/call")

    # ── 汇总 ──
    total_ms = (time.time() - t0) * 1000
    print(f"\n{'='*50}")
    print(f"  {passed}/{passed+failed} passed, avg {total_ms:.1f}ms")
    if failed:
        print(f"  {failed} FAILED")
        sys.exit(1)
    else:
        print("  ALL PASSED")


if __name__ == "__main__":
    run_tests()
