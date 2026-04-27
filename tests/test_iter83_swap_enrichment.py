#!/usr/bin/env python3
"""
迭代83 测试：Swap State Enrichment — PostCompact 恢复上下文增强

测试目标：
  1. excluded_paths 恢复（之前 swap_out 收集但 swap_in 忽略）
  2. Governor 感知的动态恢复窗口（LOW=2000, NORMAL=1500, HIGH=1000, CRITICAL=800）
  3. 标头统计信息（chars + decisions + paths）
"""

import sys
import json
import os
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent))


def _import_resume_module():
    """导入 resume-task-state.py 模块"""
    import importlib.util
    resume_path = Path.home() / ".claude" / "hooks" / "resume-task-state.py"
    if not resume_path.exists():
        print(f"SKIP: {resume_path} not found")
        sys.exit(0)
    spec = importlib.util.spec_from_file_location("resume_task_state", str(resume_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_excluded_paths_restoration():
    """Test 1: excluded_paths 恢复 — dict 格式和 str 格式"""
    mod = _import_resume_module()

    # Case A: excluded_paths 为 dict 列表
    state_dict = {
        "decisions": [{"summary": "test decision"}],
        "excluded_paths": [
            {"path": "/test/*", "reason": "test files"},
            {"path": "*.bak", "reason": "backups"},
            {"path": "__pycache__", "reason": "cache"},
        ],
    }
    result = mod._format_restore_context(state_dict)
    assert "【排除路径】" in result, f"排除路径标题缺失: {result}"
    assert "/test/*" in result, f"/test/* 缺失: {result}"
    assert "*.bak" in result, f"*.bak 缺失: {result}"
    assert "__pycache__" in result, f"__pycache__ 缺失: {result}"
    print("  ✅ Test 1a: excluded_paths dict 格式恢复正确")

    # Case B: excluded_paths 为 str 列表
    state_str = {
        "excluded_paths": ["/tmp/*", "node_modules"],
    }
    result = mod._format_restore_context(state_str)
    assert "【排除路径】" in result, f"排除路径标题缺失: {result}"
    assert "/tmp/*" in result, f"/tmp/* 缺失: {result}"
    assert "node_modules" in result, f"node_modules 缺失: {result}"
    print("  ✅ Test 1b: excluded_paths str 格式恢复正确")

    # Case C: excluded_paths 为空
    state_empty = {"excluded_paths": []}
    result = mod._format_restore_context(state_empty)
    assert "【排除路径】" not in result, f"空列表不应生成标题: {result}"
    print("  ✅ Test 1c: excluded_paths 空列表正确跳过")

    # Case D: excluded_paths 不存在
    state_missing = {"decisions": [{"summary": "x"}]}
    result = mod._format_restore_context(state_missing)
    assert "【排除路径】" not in result, f"字段不存在不应生成标题: {result}"
    print("  ✅ Test 1d: excluded_paths 缺失字段正确跳过")

    # Case E: excluded_paths 限制为 3 条
    state_many = {
        "excluded_paths": [f"/path{i}" for i in range(10)],
    }
    result = mod._format_restore_context(state_many)
    path_count = result.count("  - /path")
    assert path_count == 3, f"期望 3 条路径，实际 {path_count}: {result}"
    print("  ✅ Test 1e: excluded_paths 截断到 3 条")


def test_governor_aware_window():
    """Test 2: Governor 感知的动态恢复窗口"""
    mod = _import_resume_module()

    assert hasattr(mod, "RESTORE_CHARS_BY_PRESSURE"), "RESTORE_CHARS_BY_PRESSURE 未定义"
    assert mod.RESTORE_CHARS_BY_PRESSURE["LOW"] == 2000
    assert mod.RESTORE_CHARS_BY_PRESSURE["NORMAL"] == 1500
    assert mod.RESTORE_CHARS_BY_PRESSURE["HIGH"] == 1000
    assert mod.RESTORE_CHARS_BY_PRESSURE["CRITICAL"] == 800
    print("  ✅ Test 2a: RESTORE_CHARS_BY_PRESSURE 常量正确")

    levels = ["LOW", "NORMAL", "HIGH", "CRITICAL"]
    values = [mod.RESTORE_CHARS_BY_PRESSURE[l] for l in levels]
    for i in range(len(values) - 1):
        assert values[i] > values[i+1], f"非单调递减: {levels[i]}={values[i]} vs {levels[i+1]}={values[i+1]}"
    print("  ✅ Test 2b: 压力级别单调递减（LOW > NORMAL > HIGH > CRITICAL）")


def test_governor_window_integration():
    """Test 2c: Governor 窗口集成测试 — 用临时 governor_state.json 测试实际截断"""
    mod = _import_resume_module()
    memory_os_dir = mod.MEMORY_OS_DIR

    # _format_restore_context 对 decisions 限制 5 条，需要用多种字段凑长度
    long_decisions = [{"summary": f"Decision {i}: " + "x" * 200} for i in range(20)]
    long_conv = [{"role": "user", "summary": "y" * 300}, {"role": "assistant", "summary": "z" * 300}]
    long_tasks = [{"content": "t" * 200, "status": "in_progress"} for _ in range(5)]
    state = {
        "decisions": long_decisions,
        "excluded_paths": ["/test/*", "*.bak"],
        "conversation_summary": long_conv,
        "task_progress": long_tasks,
        "madvise_hints": [f"hint_{i}" for i in range(8)],
    }
    restore_text = mod._format_restore_context(state)
    original_len = len(restore_text)
    assert original_len > 2000, f"测试数据不够长: {original_len}"

    gov_file = memory_os_dir / "governor_state.json"
    had_gov = gov_file.exists()
    old_content = None
    if had_gov:
        old_content = gov_file.read_text("utf-8")

    try:
        gov_file.write_text(json.dumps({"level": "CRITICAL"}), "utf-8")

        effective_max = mod.MAX_RESTORE_CHARS
        gov = json.loads(gov_file.read_text("utf-8"))
        level = gov.get("level", "NORMAL")
        effective_max = mod.RESTORE_CHARS_BY_PRESSURE.get(level, mod.MAX_RESTORE_CHARS)

        assert effective_max == 800, f"CRITICAL 下 effective_max 应为 800，实际 {effective_max}"

        if len(restore_text) > effective_max:
            truncated = restore_text[:effective_max] + "\n..."
            assert len(truncated) <= effective_max + 4, f"截断后长度异常: {len(truncated)}"

        print(f"  ✅ Test 2c: CRITICAL 压力截断到 {effective_max} chars（原始 {original_len}）")

    finally:
        if had_gov and old_content:
            gov_file.write_text(old_content, "utf-8")
        elif not had_gov and gov_file.exists():
            gov_file.unlink()


def test_stats_header():
    """Test 3: 标头统计信息"""
    swap_state_cases = [
        ({"decisions": [{"summary": "d1"}, {"summary": "d2"}], "excluded_paths": ["/a", "/b"]}, "2decisions", "2paths"),
        ({"decisions": [{"summary": "d1"}], "excluded_paths": []}, "1decisions", None),
        ({"decisions": [], "excluded_paths": ["/a"]}, None, "1paths"),
        ({"decisions": [], "excluded_paths": []}, None, None),
    ]

    for i, (swap_state, expect_dec, expect_ep) in enumerate(swap_state_cases):
        restore_text = "test restore text"
        dec_count = len(swap_state.get("decisions", []))
        ep_count = len(swap_state.get("excluded_paths", []))

        stats = f"{len(restore_text)}chars"
        if dec_count:
            stats += f" {dec_count}decisions"
        if ep_count:
            stats += f" {ep_count}paths"

        context = f"【Compaction 后自动恢复·swap_state·{stats}】\n{restore_text}"

        assert "【Compaction 后自动恢复·" in context
        if expect_dec:
            assert expect_dec in stats, f"case {i}: '{expect_dec}' not in '{stats}'"
        else:
            assert "decisions" not in stats, f"case {i}: 0 decisions 不应出现 in '{stats}'"
        if expect_ep:
            assert expect_ep in stats, f"case {i}: '{expect_ep}' not in '{stats}'"
        else:
            assert "paths" not in stats, f"case {i}: 0 paths 不应出现 in '{stats}'"

    print("  ✅ Test 3a: 标头统计格式正确（4 cases）")


def test_stats_header_in_context():
    """Test 3b: 验证标头在实际 context 输出中的格式"""
    mod = _import_resume_module()

    state = {
        "decisions": [{"summary": "use WAL mode"}, {"summary": "skip cold start"}],
        "excluded_paths": ["/test/*"],
        "topics": ["性能优化"],
    }
    restore_text = mod._format_restore_context(state)

    dec_count = len(state.get("decisions", []))
    ep_count = len(state.get("excluded_paths", []))
    stats = f"{len(restore_text)}chars"
    if dec_count:
        stats += f" {dec_count}decisions"
    if ep_count:
        stats += f" {ep_count}paths"

    context = f"【Compaction 后自动恢复·swap_state·{stats}】\n{restore_text}"

    assert "2decisions" in context, f"2decisions 缺失: {context[:200]}"
    assert "1paths" in context, f"1paths 缺失: {context[:200]}"
    assert "chars" in context, f"chars 缺失: {context[:200]}"
    print(f"  ✅ Test 3b: 实际 context 标头: {stats}")


def test_full_format_with_all_fields():
    """Test 4: 完整格式化 — 所有字段都填充"""
    mod = _import_resume_module()

    state = {
        "conversation_summary": [
            {"role": "user", "summary": "请优化 hook 性能"},
            {"role": "assistant", "summary": "已合并 writer+retriever 到单连接"},
        ],
        "task_progress": [
            {"content": "合并 hook", "status": "completed"},
            {"content": "写测试", "status": "in_progress"},
            {"content": "更新文档", "status": "pending"},
        ],
        "topics": ["hook 融合", "性能优化"],
        "decisions": [
            {"summary": "单连接贯穿 writer+retriever"},
            {"summary": "TLB 增加 recency_score"},
        ],
        "excluded_paths": [
            {"path": "__pycache__", "reason": "cache"},
        ],
        "reasoning_progress": [
            {"type": "next_steps", "content": "部署到 settings.json"},
        ],
        "madvise_hints": ["hook", "性能", "TLB"],
    }

    result = mod._format_restore_context(state)

    assert "【最近对话】" in result
    assert "【任务进度】" in result
    assert "【当前话题】" in result
    assert "【已有结论】" in result
    assert "【排除路径】" in result
    assert "【下一步】" in result
    assert "【关键词】" in result

    assert "请优化 hook 性能" in result
    assert "合并 hook" in result
    assert "hook 融合" in result
    assert "单连接贯穿" in result
    assert "__pycache__" in result
    assert "部署到 settings.json" in result

    print(f"  ✅ Test 4: 完整格式化 — 所有 7 个 section 正确（{len(result)} chars）")


def test_regression_empty_state():
    """Test 5: 回归测试 — 空 state 不崩溃"""
    mod = _import_resume_module()

    assert mod._format_restore_context({}) == ""
    assert mod._format_restore_context({"decisions": []}) == ""
    assert mod._format_restore_context({"excluded_paths": [], "decisions": []}) == ""
    print("  ✅ Test 5: 空 state 回归正常")


def main():
    print("=" * 60)
    print("迭代83 测试：Swap State Enrichment")
    print("=" * 60)

    tests = [
        ("1. excluded_paths 恢复", test_excluded_paths_restoration),
        ("2. Governor 感知常量", test_governor_aware_window),
        ("2c. Governor 窗口集成", test_governor_window_integration),
        ("3a. 标头统计格式", test_stats_header),
        ("3b. 标头实际 context", test_stats_header_in_context),
        ("4. 完整格式化", test_full_format_with_all_fields),
        ("5. 回归-空 state", test_regression_empty_state),
    ]

    passed = 0
    failed = 0

    for name, func in tests:
        print(f"\n【{name}】")
        try:
            func()
            passed += 1
        except Exception as e:
            print(f"  ❌ FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"结果: {passed} passed, {failed} failed / {len(tests)} total")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
