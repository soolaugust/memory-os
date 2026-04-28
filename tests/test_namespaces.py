#!/usr/bin/env python3
"""
迭代37 测试：Namespaces — Per-Project Configuration Isolation
OS 类比：Linux Namespaces (2002-2013)

测试矩阵：
  T1 基础隔离：不同 project 读取不同 namespace 值
  T2 优先级链：env > namespace > global > default
  T3 ns_list：查看 project namespace 覆盖
  T4 ns_clear：清除 namespace 恢复全局
  T5 ns_list_all：列出所有 namespace
  T6 sysctl_set(project=...)：写入 per-project 覆盖
  T7 sysctl_list(project=...)：namespace 视图
  T8 范围钳位：namespace 值也受 min/max 约束
  T9 无 namespace 时回退全局
  T10 向后兼容：无 project 参数时行为不变
  T11 回归：sysctl 基础功能不受影响
"""
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(__file__))

import config

_original_sysctl_file = config.SYSCTL_FILE
_original_dir = config.MEMORY_OS_DIR


def _setup():
    """创建临时目录，隔离测试环境。"""
    tmpdir = tempfile.mkdtemp(prefix="memoryos_ns_test_")
    config.MEMORY_OS_DIR = tmpdir
    config.SYSCTL_FILE = os.path.join(tmpdir, "sysctl.json")
    config._invalidate_cache()
    return tmpdir


def _teardown(tmpdir):
    """恢复原始配置。"""
    config.MEMORY_OS_DIR = _original_dir
    config.SYSCTL_FILE = _original_sysctl_file
    config._invalidate_cache()
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


def test_basic_isolation():
    """T1: 不同 project 可以有不同的 tunable 值。"""
    tmpdir = _setup()
    try:
        # 默认值
        default_quota = config.get("extractor.chunk_quota")
        assert default_quota == 200

        # 为 project_a 设置 namespace 覆盖
        config.sysctl_set("extractor.chunk_quota", 500, project="project_a")
        # 为 project_b 设置不同值
        config.sysctl_set("extractor.chunk_quota", 100, project="project_b")

        # project_a 看到 500
        assert config.get("extractor.chunk_quota", project="project_a") == 500
        # project_b 看到 100
        assert config.get("extractor.chunk_quota", project="project_b") == 100
        # 无 project 看到全局默认
        assert config.get("extractor.chunk_quota") == 200
        # 未设 namespace 的 project_c 也看到全局默认
        assert config.get("extractor.chunk_quota", project="project_c") == 200

        print("  T1 PASS: basic isolation")
    finally:
        _teardown(tmpdir)


def test_priority_chain():
    """T2: 优先级链 env > namespace > global > default。"""
    tmpdir = _setup()
    try:
        key = "retriever.top_k"
        # default = 5 (iter72 updated from 3→5)
        assert config.get(key) == 5

        # 设置 global 覆盖
        config.sysctl_set(key, 7)
        assert config.get(key) == 7

        # 设置 namespace 覆盖（应优先于 global）
        config.sysctl_set(key, 10, project="proj_x")
        assert config.get(key, project="proj_x") == 10
        assert config.get(key) == 7  # 全局不受影响

        # 设置环境变量（应优先于 namespace）
        os.environ["MEMORY_OS_RETRIEVER_TOP_K"] = "15"
        config._invalidate_cache()
        assert config.get(key, project="proj_x") == 15  # env 优先
        assert config.get(key) == 15  # env 也优先于 global

        del os.environ["MEMORY_OS_RETRIEVER_TOP_K"]
        config._invalidate_cache()
        assert config.get(key, project="proj_x") == 10  # 回退到 namespace

        print("  T2 PASS: priority chain (env > ns > global > default)")
    finally:
        if "MEMORY_OS_RETRIEVER_TOP_K" in os.environ:
            del os.environ["MEMORY_OS_RETRIEVER_TOP_K"]
        _teardown(tmpdir)


def test_ns_list():
    """T3: ns_list 返回 project 的覆盖值。"""
    tmpdir = _setup()
    try:
        assert config.ns_list("proj_a") == {}

        config.sysctl_set("extractor.chunk_quota", 300, project="proj_a")
        config.sysctl_set("retriever.top_k", 5, project="proj_a")

        overrides = config.ns_list("proj_a")
        assert overrides == {"extractor.chunk_quota": 300, "retriever.top_k": 5}

        # 其他 project 不受影响
        assert config.ns_list("proj_b") == {}

        print("  T3 PASS: ns_list")
    finally:
        _teardown(tmpdir)


def test_ns_clear():
    """T4: ns_clear 清除 namespace 后恢复全局。"""
    tmpdir = _setup()
    try:
        config.sysctl_set("extractor.chunk_quota", 500, project="proj_a")
        assert config.get("extractor.chunk_quota", project="proj_a") == 500

        cleared = config.ns_clear("proj_a")
        assert cleared == 1

        # 清除后回退到全局默认
        assert config.get("extractor.chunk_quota", project="proj_a") == 200

        # 清除不存在的 namespace 返回 0
        assert config.ns_clear("nonexistent") == 0

        print("  T4 PASS: ns_clear")
    finally:
        _teardown(tmpdir)


def test_ns_list_all():
    """T5: ns_list_all 列出所有 namespace。"""
    tmpdir = _setup()
    try:
        assert config.ns_list_all() == {}

        config.sysctl_set("extractor.chunk_quota", 300, project="proj_a")
        config.sysctl_set("retriever.top_k", 5, project="proj_a")
        config.sysctl_set("extractor.chunk_quota", 100, project="proj_b")

        result = config.ns_list_all()
        assert result == {"proj_a": 2, "proj_b": 1}

        print("  T5 PASS: ns_list_all")
    finally:
        _teardown(tmpdir)


def test_sysctl_set_with_project():
    """T6: sysctl_set(project=...) 写入 per-project 覆盖。"""
    tmpdir = _setup()
    try:
        config.sysctl_set("kswapd.batch_size", 20, project="proj_a")

        # 验证磁盘上的 sysctl.json 结构
        with open(config.SYSCTL_FILE, encoding="utf-8") as _f:
            disk = json.load(_f)
        assert "namespaces" in disk
        assert "proj_a" in disk["namespaces"]
        assert disk["namespaces"]["proj_a"]["kswapd.batch_size"] == 20

        # 全局不受影响
        assert config.get("kswapd.batch_size") == 5  # default

        print("  T6 PASS: sysctl_set with project")
    finally:
        _teardown(tmpdir)


def test_sysctl_list_with_project():
    """T7: sysctl_list(project=...) 返回 namespace 视图。"""
    tmpdir = _setup()
    try:
        config.sysctl_set("extractor.chunk_quota", 500, project="proj_a")

        # namespace 视图
        ns_view = config.sysctl_list(project="proj_a")
        assert ns_view["extractor.chunk_quota"]["value"] == 500

        # 全局视图
        global_view = config.sysctl_list()
        assert global_view["extractor.chunk_quota"]["value"] == 200  # default

        # 其他 tunable 在 ns 视图中仍为全局值
        assert ns_view["retriever.top_k"]["value"] == global_view["retriever.top_k"]["value"]

        print("  T7 PASS: sysctl_list with project")
    finally:
        _teardown(tmpdir)


def test_namespace_range_clamp():
    """T8: namespace 值也受 min/max 范围钳位。"""
    tmpdir = _setup()
    try:
        # extractor.chunk_quota: range [10, 10000]
        config.sysctl_set("extractor.chunk_quota", 99999, project="proj_a")
        assert config.get("extractor.chunk_quota", project="proj_a") == 10000  # clamped to max

        config.sysctl_set("extractor.chunk_quota", 1, project="proj_b")
        assert config.get("extractor.chunk_quota", project="proj_b") == 10  # clamped to min

        print("  T8 PASS: namespace range clamp")
    finally:
        _teardown(tmpdir)


def test_fallback_no_namespace():
    """T9: 无 namespace 时回退到 global → default。"""
    tmpdir = _setup()
    try:
        # 设置全局覆盖
        config.sysctl_set("retriever.top_k", 8)

        # 没有 namespace 的 project 看到全局值
        assert config.get("retriever.top_k", project="any_project") == 8

        # 有 namespace 但没覆盖这个 key 的 project 也看到全局值
        config.sysctl_set("extractor.chunk_quota", 500, project="proj_a")
        assert config.get("retriever.top_k", project="proj_a") == 8  # fallback to global

        print("  T9 PASS: fallback to global when no namespace override")
    finally:
        _teardown(tmpdir)


def test_backward_compat():
    """T10: 向后兼容——无 project 参数时行为不变。"""
    tmpdir = _setup()
    try:
        # 和迭代27 完全相同的调用方式
        val = config.get("extractor.chunk_quota")
        assert val == 200

        config.sysctl_set("extractor.chunk_quota", 300)
        assert config.get("extractor.chunk_quota") == 300

        listing = config.sysctl_list()
        assert listing["extractor.chunk_quota"]["value"] == 300

        print("  T10 PASS: backward compatibility")
    finally:
        _teardown(tmpdir)


def test_regression_sysctl_basics():
    """T11: 回归——sysctl 基础功能不受迭代37影响。"""
    tmpdir = _setup()
    try:
        # 注册表完整性
        assert len(config._REGISTRY) >= 43  # 迭代36 有 43 个 tunable
        assert "extractor.chunk_quota" in config._REGISTRY
        assert "psi.window_size" in config._REGISTRY

        # 类型正确
        assert isinstance(config.get("extractor.chunk_quota"), int)
        assert isinstance(config.get("scorer.importance_decay_rate"), float)

        # 未知 key 报错
        try:
            config.get("nonexistent.key")
            assert False, "should raise KeyError"
        except KeyError:
            pass

        # sysctl_list 返回所有 key
        listing = config.sysctl_list()
        for key in config._REGISTRY:
            assert key in listing

        print("  T11 PASS: sysctl regression")
    finally:
        _teardown(tmpdir)


def test_performance():
    """性能测试：namespace 查找不应显著影响 get() 延迟。"""
    tmpdir = _setup()
    try:
        # 创建多个 namespace
        for i in range(20):
            config.sysctl_set("extractor.chunk_quota", 100 + i, project=f"proj_{i}")

        # 测量 get() with project 的延迟
        t0 = time.time()
        N = 1000
        for _ in range(N):
            config.get("extractor.chunk_quota", project="proj_10")
        elapsed = (time.time() - t0) * 1000
        avg_ms = elapsed / N

        assert avg_ms < 1.0, f"get() with namespace too slow: {avg_ms:.3f}ms"
        print(f"  PERF: {N}x get(project=...) = {elapsed:.1f}ms total, {avg_ms:.4f}ms/call")
    finally:
        _teardown(tmpdir)


if __name__ == "__main__":
    print("迭代37 Namespaces 测试")
    tests = [
        test_basic_isolation,
        test_priority_chain,
        test_ns_list,
        test_ns_clear,
        test_ns_list_all,
        test_sysctl_set_with_project,
        test_sysctl_list_with_project,
        test_namespace_range_clamp,
        test_fallback_no_namespace,
        test_backward_compat,
        test_regression_sysctl_basics,
        test_performance,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n结果：{passed}/{passed + failed} 通过")
    if failed:
        sys.exit(1)
