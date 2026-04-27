#!/usr/bin/env python3
"""
迭代88：OOM Killer V9 — 质量门当加强测试
OS 类比：Linux OOM Killer (Andries Brouwer, 2000) — 选择性终止消耗资源但无产出的进程

测试覆盖：
- V6 编号列表项碎片过滤
- V7 迭代完成报告过滤
- V8 指标快照过滤
- V9 回归验证报告过滤（扩展 V1）
- prompt_context 质量门当
- 白名单保留（含技术锚点的编号项、含决策动词的迭代报告）
"""
import sys, os, re, unittest

# tmpfs 隔离
sys.path.insert(0, os.path.dirname(__file__))
import tmpfs

# 导入被测函数
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'hooks'))
from extractor import _is_quality_chunk, _is_fragment
from writer import _extract_prompt_topic


class TestOOMKillerV9(unittest.TestCase):
    """V6-V9 新增过滤规则测试"""

    # ── V6: 编号列表项碎片 ──
    def test_v6_numbered_item_no_anchor_rejected(self):
        """编号列表项无技术锚点 → 拒绝"""
        cases = [
            "2. 迭代88: KV Cache 层注入（L1 极速路径，额外注入）",
            "3. 工作集恢复精度提升",
            "1. Hook 过载治理完成",
            "5. 未来方向探索",
        ]
        for s in cases:
            self.assertFalse(_is_quality_chunk(s), f"Should reject: {s!r}")

    def test_v6_numbered_item_with_file_path_kept(self):
        """编号项含文件路径 → 保留"""
        s = "2. retriever.py 新增 DRR 公平调度"
        self.assertTrue(_is_quality_chunk(s), f"Should keep: {s!r}")

    def test_v6_numbered_item_with_metric_kept(self):
        """编号项含数字度量 → 保留"""
        s = "3. 延迟从 200ms 降到 60ms"
        self.assertTrue(_is_quality_chunk(s), f"Should keep: {s!r}")

    def test_v6_numbered_item_with_code_id_kept(self):
        """编号项含代码标识符 → 保留"""
        s = "1. 新增 `_drr_select()` 函数实现公平调度"
        self.assertTrue(_is_quality_chunk(s), f"Should keep: {s!r}")

    def test_v6_numbered_item_with_quantized_change_kept(self):
        """编号项含量化变化 → 保留"""
        s = "4. 命中率 50%→75%"
        self.assertTrue(_is_quality_chunk(s), f"Should keep: {s!r}")

    # ── V7: 迭代完成报告 ──
    def test_v7_iteration_report_no_decision_rejected(self):
        """纯迭代完成报告无决策动词 → 拒绝"""
        cases = [
            "迭代86 SessionStart shadow trace 预热，冷启动修复，新测试通过",
            "内容：迭代86 SessionStart shadow trace 预热，冷启动 hit_ids 修复",
            "迭代70完成——PreCompact swap_out 信息恢复率提升",
            "迭代 83 完成三项改动",
        ]
        for s in cases:
            self.assertFalse(_is_quality_chunk(s), f"Should reject: {s!r}")

    def test_v7_iteration_with_decision_verb_kept(self):
        """迭代报告含决策动词 → 保留"""
        cases = [
            "迭代64 决定采用 chunk_version 替代 db_mtime 作为 TLB 失效信号",
            "迭代60 因为固定 baseline 导致 PSI 永久 FULL，改用自适应基线",
            "内容：迭代67 根因是 session_id=unknown 导致 swap out 失败",
        ]
        for s in cases:
            self.assertTrue(_is_quality_chunk(s), f"Should keep: {s!r}")

    # ── V8: 指标快照 ──
    def test_v8_metric_snapshot_rejected(self):
        """纯指标快照 → 拒绝"""
        cases = [
            "命中率：当前 50.5%（retriever 检索到内容并注入的比例）",
            "覆盖率：Top-5 覆盖了 80% 的活跃 chunks",
            "零访问率：35.3% 持续改善中",
            "候选池：规模 26 chunks",
            "性能微调：2.2ms/call，不影响 SessionStart 主流程",
        ]
        for s in cases:
            self.assertFalse(_is_quality_chunk(s), f"Should reject: {s!r}")

    def test_v8_metric_with_decision_kept(self):
        """指标含决策动词 → 保留"""
        cases = [
            "命中率：50%→75% 因为采用了 FTS5 替代 Python BM25",
            "性能微调：选择 immutable=1 所以 P99 从 400ms 降到 60ms",
        ]
        for s in cases:
            self.assertTrue(_is_quality_chunk(s), f"Should keep: {s!r}")

    # ── V9: 回归验证报告（扩展 V1）──
    def test_v9_regression_report_rejected(self):
        """回归验证报告 → 拒绝"""
        cases = [
            "回归验证: 209/209 测试通过 ✅",
            "回归: 11/11 kswapd + 18/18 sysctl 全绿",
            "验证：356/356 tests passed",
            "regression: 100/100 no new failures",
            "38/38 新测试全绿（test_sched_crud.py）",
            "15/15 新 _is_fragment 测试 + kswapd 11/11",
        ]
        for s in cases:
            self.assertFalse(_is_quality_chunk(s), f"Should reject: {s!r}")

    # ── 白名单回归：之前版本应该保留的 ──
    def test_existing_rules_still_work(self):
        """确保 V1-V5 规则未被破坏"""
        # 短文本 → 拒绝
        self.assertFalse(_is_quality_chunk("太短了"))
        # 截断 → 拒绝
        self.assertFalse(_is_quality_chunk("] 这是截断的"))
        # 正常决策 → 保留
        self.assertTrue(_is_quality_chunk("放弃 chromadb 因为中文 BM25 效果差，改用 SQLite FTS5"))
        # 有锚点的 — 格式 → 保留
        self.assertTrue(_is_quality_chunk("store.py 重构 — 消除 13 处重复代码"))
        # 无锚点的 — 格式 → 拒绝
        self.assertFalse(_is_quality_chunk("精简重构 — Less is More"))


class TestPromptContextQualityGate(unittest.TestCase):
    """迭代88：prompt_context 质量门当测试"""

    def test_vague_instruction_rejected(self):
        """模糊短指令 → 空字符串（不写入）"""
        cases = [
            "继续推进，而且需要对比的，这样才知道有没有这套系统的区别",
            "全部都要",
            "好的继续",
            "开始吧",
            "确认",
            "都做吧",
        ]
        for s in cases:
            result = _extract_prompt_topic(s)
            self.assertEqual(result, "", f"Should reject vague prompt: {s!r}, got: {result!r}")

    def test_tech_instruction_kept(self):
        """含技术关键词的指令 → 保留"""
        cases = [
            "修复 retriever.py 的 FTS5 降级 bug",
            "运行 test_kswapd.py 的 11 个测试",
            "把 extractor 的配额从 200 改到 500",
            "hook 触发次数太多了，从 103次/轮 降到 50次 以下",
            "迭代70 的 swap out 恢复率是多少",
        ]
        for s in cases:
            result = _extract_prompt_topic(s)
            self.assertNotEqual(result, "", f"Should keep tech prompt: {s!r}")

    def test_substantive_question_kept(self):
        """内容丰富的问题 → 保留"""
        s = "现在的成果能够用什么来源的模型benchmark工具看下效果么"
        result = _extract_prompt_topic(s)
        self.assertNotEqual(result, "", f"Should keep substantive question: {s!r}")

    def test_empty_prompt_rejected(self):
        """空/过短 prompt → 拒绝"""
        self.assertEqual(_extract_prompt_topic(""), "")
        self.assertEqual(_extract_prompt_topic("hi"), "")
        self.assertEqual(_extract_prompt_topic("   "), "")

    def test_long_prompt_truncated(self):
        """长 prompt 截断到 100 字"""
        long_text = "修复 store.py " + "A" * 200
        result = _extract_prompt_topic(long_text)
        self.assertLessEqual(len(result), 100)

    def test_markdown_header_skipped(self):
        """markdown 标题行跳过"""
        prompt = "# 标题\n\n修复 retriever.py 的 bug"
        result = _extract_prompt_topic(prompt)
        self.assertIn("retriever.py", result)


class TestIsFragmentRegression(unittest.TestCase):
    """确保 _is_fragment() 未被破坏"""

    def test_zone_reserved_not_fragment(self):
        """'ZONE_RESERVED 类型' 不以特殊字符开头 → 非碎片"""
        result = _is_fragment("ZONE_RESERVED 类型")
        self.assertFalse(result)

    def test_arch_label_fragment_caught(self):
        """/L4/L5 架构标签 → 碎片"""
        self.assertTrue(_is_fragment("/L4/L5 层级存储"))
        self.assertTrue(_is_fragment("/层级名 架构"))

    def test_file_path_not_fragment(self):
        """真实文件路径 → 非碎片"""
        self.assertFalse(_is_fragment("/home/mi/store.py 修改"))


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    failed = len(result.failures) + len(result.errors)
    print(f"\n{'='*60}")
    print(f"迭代88 OOM Killer V9: {result.testsRun} tests, {failed} failures")
    sys.exit(0 if failed == 0 else 1)
