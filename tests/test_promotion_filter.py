#!/usr/bin/env python3
"""
迭代74 测试：_is_quality_chunk() v7 Promotion Filter
拦截纯验证报告、纯性能数据、HTML 标签泄漏
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import tmpfs  # noqa: F401 — 测试隔离

from hooks.extractor import _is_quality_chunk


if __name__ == "__main__":
    passed = failed = 0

    def check(name, actual, expected):
        global passed, failed
        if actual == expected:
            passed += 1
            print(f"  PASS: {name}")
        else:
            failed += 1
            print(f"  FAIL: {name}: expected {expected}, got {actual}")


    print("=== V1: 纯验证/测试报告 ===")
    # 应被拦截（return False）
    check("pure_pass_count", _is_quality_chunk("9/9 通过，145ms"), False)
    check("pure_pass_all", _is_quality_chunk("33/33 新测试 + 18/18 回归全绿"), False)
    check("verification_prefix", _is_quality_chunk("验证：27/27 通过，回归全绿，psi_stats 0.05ms/call"), False)
    check("verification_cn", _is_quality_chunk("验证：33/33 新测试 + 18/18 sysctl 回归"), False)
    check("regression_prefix", _is_quality_chunk("回归：11/11 kswapd + 18/18 sysctl 全绿"), False)
    check("test_prefix", _is_quality_chunk("测试：42/42 PSI 全通过"), False)

    # 不应被拦截（含有决策信息的验证句）
    check("decision_with_num", _is_quality_chunk("选择 FTS5 替代 Python BM25 全表扫描"), True)
    check("action_with_test", _is_quality_chunk("已完成 FTS5 迁移，延迟从 200ms 降到 7ms"), True)
    check("conclusion_with_metric", _is_quality_chunk("选择 BM25 因为短文档效果更好，精度提升 15%"), True)

    print("\n=== V2: 纯性能/延迟数据 ===")
    # 应被拦截
    check("perf_prefix", _is_quality_chunk("性能：0.155ms/call（1000 候选 → Top-5）"), False)
    check("bypass_latency", _is_quality_chunk("bypass 延迟: 0.03ms（比正常 AIMD 路径快 10x+）"), False)
    check("perf_write", _is_quality_chunk("性能：100 次写入 0.9ms，50 条读取 0.1ms"), False)
    check("latency_prefix", _is_quality_chunk("延迟：7ms p95"), False)
    check("avg_latency", _is_quality_chunk("avg: 0.35ms/call 含 FTS5 查询"), False)

    # 不应被拦截（含决策的性能句）
    check("decision_perf", _is_quality_chunk("采用 FTS5 后延迟从 200ms 降到 7ms"), True)
    check("comparison_perf", _is_quality_chunk("BM25 比 TF-IDF 快 3x，选择 BM25"), True)

    print("\n=== V3: HTML/XML 标签泄漏 ===")
    # 应被拦截
    check("html_tag", _is_quality_chunk("<task-notification>"), False)
    check("xml_tag", _is_quality_chunk("<system-reminder>这是内部标记"), False)

    # 不应被拦截（正常使用 < 的句子）
    check("comparison_lt", _is_quality_chunk("目标 < 150ms 不调用 LLM"), True)
    check("generics", _is_quality_chunk("使用 Dict<str, int> 类型映射"), True)

    print("\n=== 既有规则回归 ===")
    # 确保既有规则仍有效
    check("short", _is_quality_chunk("太短了"), False)
    check("bracket_start", _is_quality_chunk("[决策] 这是一个选择"), False)
    check("dash_start", _is_quality_chunk("- 列表项内容"), False)
    check("particle_start", _is_quality_chunk("了什么什么的"), False)
    check("table_3pipe", _is_quality_chunk("col1 | col2 | col3 | col4"), False)
    check("noise_kw", _is_quality_chunk("hookSpecificOutput 格式说明"), False)
    check("valid_decision", _is_quality_chunk("选择 BM25 替代 TF-IDF 因为短文档效果更好"), True)
    check("valid_exclusion", _is_quality_chunk("放弃 chromadb 因为中文 BM25 效果差"), True)
    check("valid_reasoning", _is_quality_chunk("根本原因是 WAL checkpoint 阻塞了读操作"), True)


    print(f"\n{'='*50}")
    print(f"结果：{passed}/{passed+failed} 通过")
    if failed:
        print(f"FAILED: {failed} tests")
        sys.exit(1)
    else:
        print("ALL PASSED")
