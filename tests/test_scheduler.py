#!/usr/bin/env python3
"""
迭代28 测试：Scheduler Nice Levels — 查询优先级分类器
验证 _classify_query_priority() 对不同 query 的分级正确性
"""
import sys
import os
import time

# 设置环境
os.environ.setdefault("CLAUDE_CWD", str(__import__("pathlib").Path(__file__).parent.parent.parent.parent.parent))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "hooks"))
sys.path.insert(0, os.path.dirname(__file__))

from hooks.retriever import _classify_query_priority, _extract_key_entities, _SKIP_PATTERNS, _has_real_tech_signal


if __name__ == "__main__":
    passed = 0
    failed = 0

    def test(name, actual, expected):
        global passed, failed
        if actual == expected:
            passed += 1
            print(f"  ✅ {name}: {actual}")
        else:
            failed += 1
            print(f"  ❌ {name}: expected={expected}, got={actual}")


    print("=" * 60)
    print("迭代28 Scheduler Nice Levels 测试")
    print("=" * 60)

    # ── Test 1：SKIP 类（确认/闲聊，nice 19）──
    print("\n--- Test 1：SKIP 类（确认/闲聊）---")
    skip_cases = [
        ("好", "好", False, 0),
        ("好的", "好的", False, 0),
        ("嗯", "嗯", False, 0),
        ("ok", "ok", False, 0),
        ("继续", "继续", False, 0),
        ("确认", "确认", False, 0),
        ("谢谢", "谢谢", False, 0),
        ("是", "是", False, 0),
        ("收到", "收到", False, 0),
        ("lgtm", "lgtm", False, 0),
        ("sure", "sure", False, 0),
        ("yes", "yes", False, 0),
    ]
    for prompt, query, pf, ec in skip_cases:
        test(f"SKIP: '{prompt}'", _classify_query_priority(prompt, query, pf, ec), "SKIP")

    # ── Test 2：SKIP 不误杀技术 query ──
    print("\n--- Test 2：含技术信号不 SKIP ---")
    no_skip_cases = [
        ("好", "好 retriever.py", False, 0),
        ("继续", "继续 `BM25` 优化", False, 0),
        ("是", "是 API 的问题", False, 0),
    ]
    for prompt, query, pf, ec in no_skip_cases:
        result = _classify_query_priority(prompt, query, pf, ec)
        test(f"NOT SKIP: prompt='{prompt}' query='{query}'", result != "SKIP", True)

    # ── Test 3：FULL 类（有缺页信号，nice -20）──
    print("\n--- Test 3：FULL 类（缺页信号 → 不可降级）---")
    test("page_fault → FULL",
         _classify_query_priority("好", "好 some_fault_query", True, 0), "FULL")
    test("page_fault short → FULL",
         _classify_query_priority("ok", "ok fault", True, 0), "FULL")

    # ── Test 4：FULL 类（多技术实体，nice -20）──
    print("\n--- Test 4：FULL 类（多技术实体）---")
    test("2 entities → FULL",
         _classify_query_priority("修复 BM25 和 FTS5 的兼容性",
                                  "修复 BM25 和 FTS5 的兼容性 BM25 FTS5", False, 2), "FULL")
    test("3 entities → FULL",
         _classify_query_priority("优化 retriever scorer store 三个模块",
                                  "优化 retriever scorer store 三个模块 retriever scorer store", False, 3), "FULL")

    # ── Test 5：LITE 类（普通中等 query，nice 0）──
    print("\n--- Test 5：LITE 类（普通中等 query）---")
    test("medium query → LITE",
         _classify_query_priority("这个函数的作用是什么", "这个函数的作用是什么", False, 0), "LITE")
    test("normal question → LITE",
         _classify_query_priority("如何优化数据库查询", "如何优化数据库查询", False, 0), "LITE")

    # ── Test 6：FULL 类（长 query，超过 lite 阈值）──
    print("\n--- Test 6：FULL 类（长 query）---")
    long_query = "请分析 memory-os 的 retriever.py 和 knowledge_router.py 之间的数据流，特别是 FTS5 检索结果如何传递给 scorer.py 计算最终分数，以及 Per-Request Connection Scope 如何确保整个流程共享同一个 SQLite 连接。同时需要检查 PCID 磁盘缓存和进程内 TTL 缓存的一致性。" + " extra" * 20
    test("long query → FULL",
         _classify_query_priority(long_query, long_query, False, 1), "FULL")

    # ── Test 7：_SKIP_PATTERNS 正则匹配测试 ──
    print("\n--- Test 7：SKIP 正则模式匹配 ---")
    should_match = ["好", "好的", "好吧", "嗯嗯", "ok", "okay", "是", "是的", "继续", "确认", "谢谢", "lgtm", "sure", "yes"]
    should_not_match = ["好的请继续分析", "ok but check this", "是的但还有个bug", "修复bug"]
    for s in should_match:
        test(f"pattern match '{s}'", bool(_SKIP_PATTERNS.match(s.strip())), True)
    for s in should_not_match:
        test(f"pattern no match '{s}'", bool(_SKIP_PATTERNS.match(s.strip())), False)

    # ── Test 8：_TECH_SIGNAL 技术信号检测 ──
    print("\n--- Test 8：技术信号检测 ---")
    tech_cases = [
        ("`retriever`", True),
        ("retriever.py", True),
        ("BM25", True),
        ("FTS5", True),
        ("修复 error 的问题", True),
        ("函数调用", True),
        ("def main():", True),
        ("今天天气不错", False),
        ("你好呀", False),
    ]
    for text, expected in tech_cases:
        test(f"tech signal '{text}'", _has_real_tech_signal(text), expected)

    # ── Test 9：性能测试 ──
    print("\n--- Test 9：分类器性能 ---")
    t0 = time.time()
    N = 10000
    for _ in range(N):
        _classify_query_priority("好", "好", False, 0)
        _classify_query_priority("修复 BM25 bug", "修复 BM25 bug BM25", False, 1)
        _classify_query_priority("请分析 retriever.py 的性能", "请分析 retriever.py 的性能 retriever.py", True, 2)
    elapsed = (time.time() - t0) * 1000
    per_call = elapsed / (N * 3)
    test(f"{N*3} calls in {elapsed:.1f}ms, {per_call:.4f}ms/call", per_call < 0.1, True)

    # ── 结果汇总 ──
    print("\n" + "=" * 60)
    print(f"结果：{passed} passed, {failed} failed, 共 {passed+failed} tests")
    print(f"avg {per_call:.4f}ms/classify")
    if failed > 0:
        sys.exit(1)
    print("✅ 迭代28 全部通过")
