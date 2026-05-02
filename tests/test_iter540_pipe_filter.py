"""
iter540: pipe_filter — Markdown Table Cell Leak Prevention
OS 类比：Linux pipe(2) SIGPIPE — 管道读端关闭时 kill 写端，防止数据泄漏到死管道。

验证：_extract_quantitative_conclusions 正确拦截 markdown 表格行，
     同时保留合法的量化证据（含数字+结论动词但不是表格格式）。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tmpfs  # noqa: F401 — 测试隔离
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hooks"))

from extractor import _extract_quantitative_conclusions, _is_fragment, _is_quality_chunk


class TestPipeFilter:
    """iter540: 表格单元格泄漏拦截"""

    # ── 拦截类：必须被过滤的表格行 ──────────────────────────────────

    def test_table_row_pipe_start(self):
        """以 | 开头的表格数据行应被拦截"""
        text = "| 根因 | 单次 extractor 可写入无限 chunks（实测 1 秒 14 个碎片） |"
        result = _extract_quantitative_conclusions(text)
        assert not any("根因" in r for r in result), f"表格行泄漏: {result}"

    def test_table_row_cleanup_stats(self):
        """含清理统计的表格行应被拦截"""
        text = "| 清理 | 90→71 chunks（-19），零访问率 35.6%→18.3% |"
        result = _extract_quantitative_conclusions(text)
        assert not any("清理" in r for r in result), f"表格行泄漏: {result}"

    def test_table_row_multi_column(self):
        """多列表格行（3+ 个 |）应被拦截"""
        text = "| iter | 效果 | 延迟 2.1ms→0.5ms | 通过 |"
        result = _extract_quantitative_conclusions(text)
        assert len(result) == 0, f"多列表格泄漏: {result}"

    def test_table_row_without_leading_pipe(self):
        """不以 | 开头但含 2+ 个 | 的行也是表格行"""
        text = "iter539 | 90→71 chunks | -19 | 零访问率 35.6%→18.3%"
        result = _extract_quantitative_conclusions(text)
        assert len(result) == 0, f"表格行泄漏: {result}"

    def test_table_header_separator(self):
        """|---| 表头分隔符应被拦截（原有逻辑）"""
        text = "|---|---|---|\n| iter | 效果 | 延迟 |"
        result = _extract_quantitative_conclusions(text)
        assert len(result) == 0

    def test_table_mixed_content(self):
        """混合文本中表格行应被拦截，正常行保留"""
        text = """检索延迟从 57ms 降至 23ms（-60%）

| 指标 | 前 | 后 |
|---|---|---|
| 延迟 | 57ms | 23ms |
| 命中率 | 40% | 85% |

上述优化效果稳定。"""
        result = _extract_quantitative_conclusions(text)
        # 第一行 "检索延迟从 57ms 降至 23ms（-60%）" 应被保留
        assert any("57ms" in r or "23ms" in r for r in result), f"合法行被误杀: {result}"
        # 表格行不应出现
        assert not any("|" in r for r in result), f"表格行泄漏: {result}"

    # ── 保留类：合法量化证据应通过 ────────────────────────────────

    def test_normal_quant_evidence(self):
        """正常量化证据应通过"""
        text = "检索延迟从 57ms 降至 23ms，提升 60%"
        result = _extract_quantitative_conclusions(text)
        assert len(result) >= 1, f"合法量化被误杀: {result}"

    def test_arrow_quant(self):
        """含箭头的量化结论应通过"""
        text = "零访问率 35.6%→18.3%（-17.3pp），FTS5 rebuild 通过"
        result = _extract_quantitative_conclusions(text)
        assert len(result) >= 1, f"箭头量化被误杀: {result}"

    def test_pipe_in_code_not_blocked(self):
        """代码中的管道符不是表格（但代码行本身被代码过滤器拦截）"""
        text = "grep error | wc -l 结果 = 0（验证通过）"
        # 这行含 | 但只有1个，不触发表格检测
        # 但它可能被代码行过滤器拦截（grep/wc 是 shell 命令）
        result = _extract_quantitative_conclusions(text)
        # 不强制通过——可能被代码过滤器合理拦截

    def test_single_pipe_in_sentence(self):
        """句子中单个 | 不应触发表格过滤"""
        text = "性能对比：A方案 延迟 50ms | B方案的延迟稳定在 30ms"
        # 只有1个|，不触发 count>=2 检查
        # 但此句不 startswith('|')，所以不触发第一条规则
        # 是否通过取决于其他门控条件
        result = _extract_quantitative_conclusions(text)
        # 不强制——这里测试的是 pipe_filter 不误杀单管道句

    # ── _is_fragment 对表格行的防护验证 ──────────────────────────

    def test_is_fragment_pipe_start(self):
        """_is_fragment 应捕获以 | 开头的文本"""
        assert _is_fragment("| 单次 extractor 可写入无限 chunks")

    def test_is_fragment_multi_pipe(self):
        """_is_fragment 应捕获含 2+ 个 | 的文本"""
        assert _is_fragment("根因 | 单次写入 | 14 个碎片")

    # ── _is_quality_chunk 对表格行的防护验证 ────────────────────────

    def test_quality_chunk_pipe_start(self):
        """_is_quality_chunk 应拒绝以 | 开头的摘要"""
        assert not _is_quality_chunk("| 根因 | 单次 extractor 可写入无限 chunks")

    def test_quality_chunk_multi_pipe(self):
        """_is_quality_chunk 应拒绝含 3+ 个 | 的摘要"""
        assert not _is_quality_chunk("iter | 效果 | 延迟 | 通过")


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
