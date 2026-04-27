#!/usr/bin/env python3
"""
迭代73 测试：_is_fragment() 碎片检测 + _extract_conversation_summary() 质量提升
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import tmpfs  # noqa: F401 — 测试隔离

from hooks.extractor import _is_fragment, _extract_conversation_summary


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


    # ── _is_fragment 测试 ──
    print("=== _is_fragment() tests ===")

    # 应判定为碎片
    check("starts_with_underscore", _is_fragment("_at) — 7天 grace period"), True)
    check("starts_with_hash", _is_fragment("### 下次 compaction 时"), True)
    check("starts_with_pipe", _is_fragment("| 字段 | 值 |"), True)
    check("starts_with_paren", _is_fragment(") 的参数不对"), True)
    check("starts_with_bracket", _is_fragment("] 结束标记"), True)
    check("starts_with_brace", _is_fragment("} else {"), True)
    check("starts_with_gt", _is_fragment("> 引用块碎片"), True)
    check("pure_numbers", _is_fragment("123 / 456 = 0.27"), True)
    check("table_row", _is_fragment("col1 | col2 | col3"), True)
    check("too_short", _is_fragment("short"), True)
    check("empty", _is_fragment(""), True)

    # 不应判定为碎片（完整句子）
    check("valid_chinese", _is_fragment("已完成 FTS5 全文索引迁移，10/10 测试通过"), False)
    check("valid_english", _is_fragment("Successfully deployed the new authentication module"), False)
    check("valid_diagnosis", _is_fragment("发现根因是 WAL checkpoint 阻塞了读操作"), False)
    check("valid_decision", _is_fragment("选择 BM25 替代 TF-IDF 因为短文档效果更好"), False)
    check("valid_with_pipe", _is_fragment("管道处理需要单独的 buffer"), False)  # 只有1个|


    # ── _extract_conversation_summary 测试 ──
    print("\n=== _extract_conversation_summary() quality tests ===")

    # S1: 完成动作 — 正常匹配
    text1 = "已完成 FTS5 全文索引迁移，retriever 延迟从 200ms 降到 7ms"
    results1 = _extract_conversation_summary(text1)
    check("s1_action_match", len(results1) >= 1, True)
    check("s1_no_fragment", all(not _is_fragment(r) for r in results1), True)

    # S1: 碎片应被过滤
    text2 = "已完成 _at) — 修复了一些问题"
    results2 = _extract_conversation_summary(text2)
    # _at) 开头的捕获应被 _is_fragment 过滤
    for r in results2:
        check(f"s1_filtered_{r[:20]}", _is_fragment(r), False)

    # S3: markdown 标题下碎片应被过滤
    text3 = """## 总结
    ### 下次 compaction 时
    需要注意的事项"""
    results3 = _extract_conversation_summary(text3)
    # "### 下次 compaction 时" 不应出现在结果中
    for r in results3:
        check(f"s3_no_hash_prefix_{r[:20]}", not r.startswith('#'), True)

    # S3: 正常总结应匹配
    text4 = """## 总结
    迭代73完成了碎片检测过滤器的实现，提升了知识提取质量"""
    results4 = _extract_conversation_summary(text4)
    check("s3_valid_summary", len(results4) >= 1, True)

    # S2: 诊断匹配
    text5 = "根因是：WAL auto-checkpoint 在 synchronous=NORMAL 模式下阻塞了 commit"
    results5 = _extract_conversation_summary(text5)
    check("s2_diagnosis_match", len(results5) >= 1, True)

    # 混合测试：碎片和有效内容共存
    text6 = """已完成 | 未完成 | 状态
    已修复 retriever 的 TLB 缓存失效问题
    发现：_internal 变量泄露到全局作用域"""
    results6 = _extract_conversation_summary(text6)
    # 表格行和 _internal 开头的应被过滤
    for r in results6:
        check(f"mixed_quality_{r[:20]}", not _is_fragment(r), True)


    print(f"\n{'='*50}")
    print(f"结果：{passed}/{passed+failed} 通过")
    if failed:
        print(f"FAILED: {failed} tests")
        sys.exit(1)
    else:
        print("ALL PASSED")
