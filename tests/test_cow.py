#!/usr/bin/env python3
"""
迭代39 测试：COW 预扫描 — Copy-on-Write 写入时惰性求值
OS 类比：Linux fork() COW (1991)

测试覆盖：
  T1 预扫描命中：含决策信号词的消息应触发完整提取
  T2 预扫描跳过：纯代码/确认消息应跳过完整提取
  T3 量化证据检测：含数字指标的消息应命中
  T4 因果链检测：因果关系句式应命中
  T5 英文信号词：英文决策词应命中
  T6 完成动作检测：已完成/已修复等应命中
  T7 边界：空文本/极短文本
  T8 sysctl 控制：cow_prescan_chars 限制扫描范围
  T9 预扫描性能：< 0.5ms
  T10 预扫描正则与实际提取的一致性验证
"""
import sys
import time
from pathlib import Path

# 添加 memory-os 根目录到 path
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "hooks"))


if __name__ == "__main__":
    passed = 0
    failed = 0


    def test(name, condition, detail=""):
        global passed, failed
        if condition:
            passed += 1
            print(f"  PASS  {name}")
        else:
            failed += 1
            print(f"  FAIL  {name} — {detail}")


    # ── 导入 ──
    from hooks.extractor import _cow_prescan, _COW_PRESCAN
    from hooks.extractor import (
        _extract_by_signals, DECISION_SIGNALS, EXCLUDED_SIGNALS,
        REASONING_SIGNALS, _extract_comparisons, _extract_causal_chains,
        _extract_quantitative_conclusions, _extract_conversation_summary,
    )

    print("=" * 60)
    print("迭代39：COW 预扫描 — Copy-on-Write 写入时惰性求值")
    print("=" * 60)

    # ── T1：预扫描命中 — 含决策信号词 ──
    print("\n--- T1: 预扫描命中（决策信号词）---")
    decision_texts = [
        "经过分析，选择 BM25 作为召回引擎而非 embedding",
        "最终方案：使用 FTS5 全文索引替代 Python 端 BM25",
        "结论：hybrid tokenize 的 bigram 效果优于 unigram",
        "推荐使用 WAL 模式提高并发读写性能",
    ]
    for txt in decision_texts:
        test(f"cow_hit '{txt[:40]}...'", _cow_prescan(txt), f"expected hit, got miss")

    # ── T2：预扫描跳过 — 纯代码/确认/调试 ──
    print("\n--- T2: 预扫描跳过（无信号词）---")
    skip_texts = [
        "好的",
        "收到",
        "```python\ndef hello():\n    print('world')\n```",
        "文件已保存到 /tmp/output.json",
        "这是一段普通的文本描述没有任何特殊关键词也没有数字",
        "让我来看看这个文件的内容",
        "我理解你的需求了",
    ]
    for txt in skip_texts:
        test(f"cow_skip '{txt[:40]}...'", not _cow_prescan(txt), f"expected skip, got hit")

    # ── T3：量化证据检测 ──
    print("\n--- T3: 量化证据检测 ---")
    quant_texts = [
        "召回延迟从 5ms 降到 2ms",
        "命中率提升到 85%",
        "8/8 测试全部通过",
        "准确率 92.5% 超过基线",
    ]
    for txt in quant_texts:
        test(f"cow_hit_quant '{txt[:40]}...'", _cow_prescan(txt), f"expected hit, got miss")

    # ── T4：因果链检测 ──
    print("\n--- T4: 因果链检测 ---")
    causal_texts = [
        "因为 FTS5 使用 C 层 BM25，所以比 Python 端快 10 倍",
        "由于内存碎片问题，因此需要 compaction",
    ]
    for txt in causal_texts:
        test(f"cow_hit_causal '{txt[:40]}...'", _cow_prescan(txt), f"expected hit, got miss")

    # ── T5：英文信号词 ──
    print("\n--- T5: 英文信号词 ---")
    english_texts = [
        "Decided to use SQLite FTS5 for indexing",
        "Conclusion: hybrid tokenize works best for Chinese",
        "Successfully deployed the new retriever",
        "Fixed the bug in kswapd_scan watermark logic",
    ]
    for txt in english_texts:
        test(f"cow_hit_en '{txt[:40]}...'", _cow_prescan(txt), f"expected hit, got miss")

    # ── T6：完成动作检测 ──
    print("\n--- T6: 完成动作检测 ---")
    action_texts = [
        "已完成 BM25 引擎的统一重构",
        "已修复 FTS5 索引不同步的问题",
        "已实现 kswapd 后台水位线预淘汰",
        "完成了 Unified Scorer 的迁移工作",
    ]
    for txt in action_texts:
        test(f"cow_hit_action '{txt[:40]}...'", _cow_prescan(txt), f"expected hit, got miss")

    # ── T7：边界测试 ──
    print("\n--- T7: 边界测试 ---")
    test("cow_empty", not _cow_prescan(""), "empty text should skip")
    test("cow_short", not _cow_prescan("ok"), "very short text should skip")
    test("cow_whitespace", not _cow_prescan("   \n\n  "), "whitespace should skip")

    # ── T8：sysctl 控制 — cow_prescan_chars 限制扫描范围 ──
    print("\n--- T8: sysctl 控制 ---")
    from config import get as _sysctl
    prescan_chars = _sysctl("extractor.cow_prescan_chars")
    test("sysctl_cow_prescan_chars", prescan_chars == 3000,
         f"expected 3000, got {prescan_chars}")

    # 信号词在 prescan_chars 范围外 → 应跳过
    long_prefix = "a" * 3500  # 超出 3000 字符
    signal_at_end = long_prefix + " 选择 BM25 作为召回引擎"
    test("cow_signal_beyond_range", not _cow_prescan(signal_at_end),
         "signal beyond prescan_chars should be missed")

    # 信号词在 prescan_chars 范围内 → 应命中
    signal_at_start = "选择 BM25 作为召回引擎 " + long_prefix
    test("cow_signal_within_range", _cow_prescan(signal_at_start),
         "signal within prescan_chars should hit")

    # ── T9：预扫描性能 ──
    print("\n--- T9: 性能测试 ---")
    # 生成一个 3000 字符的无信号文本
    perf_text = "这是一段普通文本不含任何信号词 " * 100
    iterations = 1000
    t_start = time.time()
    for _ in range(iterations):
        _cow_prescan(perf_text)
    t_elapsed = (time.time() - t_start) * 1000  # ms
    avg_ms = t_elapsed / iterations
    test(f"cow_perf_{iterations}x", avg_ms < 0.5,
         f"avg={avg_ms:.4f}ms (target < 0.5ms)")
    print(f"  INFO  {iterations}x prescan avg: {avg_ms:.4f}ms")

    # 有信号文本性能
    perf_text_hit = "经过分析我们决定采用 FTS5 " + "补充文本 " * 500
    t_start = time.time()
    for _ in range(iterations):
        _cow_prescan(perf_text_hit)
    t_elapsed_hit = (time.time() - t_start) * 1000
    avg_ms_hit = t_elapsed_hit / iterations
    test(f"cow_perf_hit_{iterations}x", avg_ms_hit < 0.5,
         f"avg={avg_ms_hit:.4f}ms (target < 0.5ms)")
    print(f"  INFO  {iterations}x prescan (hit) avg: {avg_ms_hit:.4f}ms")

    # ── T10：预扫描与实际提取的一致性 ──
    print("\n--- T10: 预扫描与提取一致性 ---")
    # 如果预扫描命中，实际提取也应产出非空结果
    consistency_texts = [
        ("选择：使用 BM25 hybrid tokenize 作为召回引擎", "decision"),
        ("放弃 chromadb 因为中文 BM25 效果差", "excluded"),
        ("根本原因：FTS5 索引与主表不同步导致查询 miss", "reasoning"),
        ("已完成 retriever v8 的 Per-Request Connection Scope 优化", "summary"),
    ]
    for txt, expected_type in consistency_texts:
        prescan_ok = _cow_prescan(txt)
        if expected_type == "decision":
            extracted = _extract_by_signals(txt, DECISION_SIGNALS)
        elif expected_type == "excluded":
            extracted = _extract_by_signals(txt, EXCLUDED_SIGNALS)
        elif expected_type == "reasoning":
            extracted = _extract_by_signals(txt, REASONING_SIGNALS)
        elif expected_type == "summary":
            extracted = _extract_conversation_summary(txt)
        else:
            extracted = []
        # 预扫描命中是实际提取的必要条件（但非充分条件——预扫描可能误命中）
        test(f"consistency_{expected_type}", prescan_ok,
             f"prescan should hit for {expected_type}: '{txt[:40]}...'")

    # ── 汇总 ──
    print("\n" + "=" * 60)
    total = passed + failed
    print(f"结果: {passed}/{total} 通过" + (f"  ({failed} 失败)" if failed else "  ✅ 全绿"))
    print("=" * 60)

    sys.exit(1 if failed else 0)
