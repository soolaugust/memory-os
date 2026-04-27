"""
test_quant_expansion.py — 迭代336：Quantitative Evidence Semantic Expansion 单元测试

信息论背景：Encoding-Retrieval Mismatch (Tulving 1973) —
  quantitative_evidence summary 含数字/符号（"11.4us→1.35us"），
  查询用概念词（"如何优化性能"），FTS5 词汇匹配无法跨越语义鸿沟 → 14/18 (78%) 零召回。

Document Expansion 修复：写入时预计算语义概念词追加到 content，
  让 FTS5 能按概念词索引量化证据。

OS 类比：Linux /proc/[pid]/wchan — 符号地址映射为人类可读的系统调用名。

验证：
  1. 性能优化方向（数值降低）→ 追加性能优化概念词
  2. 性能提升方向（数值升高）→ 追加 improve/increase 概念词
  3. 检索/召回类词汇 → 追加 FTS5/BM25/检索优化
  4. 启动/延迟类词汇 → 追加 latency/startup 概念词
  5. 修复/bug类词汇 → 追加 fix/repair
  6. 迭代版本词汇 → 追加 iterN 版本改进
  7. 无规则匹配 → fallback 通用量化优化词
  8. FTS5 集成：写入 quant chunk 后，按概念词查询能命中
"""
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "hooks"))

import tmpfs  # noqa

# 导入被测函数
from extractor import _quant_semantic_concepts
from store_vfs import fts_search, open_db, ensure_schema, insert_chunk


# ──────────────────────────────────────────────────────────────────────
# 1. 性能优化方向（数值降低：11.4 → 1.35）
# ──────────────────────────────────────────────────────────────────────

def test_perf_decrease_concepts():
    """数值降低方向（11.4us→1.35us）→ 追加性能优化类概念词。"""
    s = "BM25 pipeline 优化：11.4us→1.35us（↓88%）"
    result = _quant_semantic_concepts(s)
    assert result, "should generate concepts"
    # 必须包含性能优化相关词
    assert any(kw in result for kw in ["性能优化", "延迟降低", "optimize", "latency"]), \
        f"should contain perf/latency keywords: {result}"


# ──────────────────────────────────────────────────────────────────────
# 2. 性能提升方向（数值升高：0.5 → 0.9）
# ──────────────────────────────────────────────────────────────────────

def test_perf_increase_concepts():
    """数值升高方向（0.5→0.9）→ 追加 improve/increase/提升概念词。"""
    s = "召回率 Precision@5: 0.5→0.9（+80%）"
    result = _quant_semantic_concepts(s)
    assert result, "should generate concepts"
    # 数值升高 → improve / increase / 提升
    assert any(kw in result for kw in ["improve", "increase", "提升", "检索", "recall"]), \
        f"should contain improve keywords: {result}"


# ──────────────────────────────────────────────────────────────────────
# 3. 检索/召回类词汇
# ──────────────────────────────────────────────────────────────────────

def test_retrieval_concepts():
    """含 'recall'/'FTS5' → 追加检索优化概念词。"""
    s = "FTS5 hit_rate: 62% → BM25 fallback: 18%，检索质量提升"
    result = _quant_semantic_concepts(s)
    assert result, "should generate concepts"
    assert any(kw in result for kw in ["检索优化", "召回率", "FTS5", "BM25", "recall"]), \
        f"should contain retrieval keywords: {result}"


# ──────────────────────────────────────────────────────────────────────
# 4. 启动/延迟类词汇
# ──────────────────────────────────────────────────────────────────────

def test_latency_concepts():
    """含 'import'/'ms' → 追加启动性能概念词。"""
    s = "bm25 模块 import 时间: 32ms → 3ms（懒加载）"
    result = _quant_semantic_concepts(s)
    assert result, "should generate concepts"
    assert any(kw in result for kw in ["启动性能", "冷启动", "import", "latency", "ms", "startup"]), \
        f"should contain latency keywords: {result}"


# ──────────────────────────────────────────────────────────────────────
# 5. 修复/bug类词汇
# ──────────────────────────────────────────────────────────────────────

def test_fix_concepts():
    """含 'fix'/'修复' → 追加 fix/repair 概念词。"""
    s = "修复 UUID rowid 污染：FTS5 命中率 0% → 63%（iter124 bugfix）"
    result = _quant_semantic_concepts(s)
    assert result, "should generate concepts"
    assert any(kw in result for kw in ["修复", "fix", "repair", "bug", "regression"]), \
        f"should contain fix keywords: {result}"


# ──────────────────────────────────────────────────────────────────────
# 6. 迭代版本词汇
# ──────────────────────────────────────────────────────────────────────

def test_iter_concepts():
    """含 'iter238' → 追加 iter238 版本改进概念词。"""
    s = "累积 iter238→261：11.4us → 1.35us"
    result = _quant_semantic_concepts(s)
    assert result, "should generate concepts"
    assert "iter238" in result or "迭代优化" in result, \
        f"should contain iter238 or 迭代优化: {result}"


# ──────────────────────────────────────────────────────────────────────
# 7. 无规则匹配 → fallback
# ──────────────────────────────────────────────────────────────────────

def test_fallback_concepts():
    """纯数字 summary 无特定类别词 → fallback 通用量化优化词。"""
    s = "count: 42"
    result = _quant_semantic_concepts(s)
    assert result, "fallback should always return something"
    # fallback 包含通用量化词
    assert any(kw in result for kw in ["量化优化", "性能改进", "optimize", "improve", "benchmark"]), \
        f"fallback should contain generic optimize keywords: {result}"


# ──────────────────────────────────────────────────────────────────────
# 8. FTS5 集成：写入 quant chunk 后，按概念词查询能命中
# ──────────────────────────────────────────────────────────────────────

def test_fts_concept_match():
    """
    写入 quantitative_evidence chunk，content 含概念词，
    用概念词查询（"性能优化"）能在 fts_search 中命中。
    """
    from store import open_db, ensure_schema
    conn = open_db()
    ensure_schema(conn)

    chunk_id = str(uuid.uuid4())
    quant_summary = "FTS5 检索延迟 11.4us→1.35us（优化88%）"
    concept_str = _quant_semantic_concepts(quant_summary)

    # 模拟 extractor 写入：content 包含概念词
    content = f"[quantitative_evidence] {quant_summary} [concepts: {concept_str}]"
    conn.execute(
        """INSERT INTO memory_chunks
           (id, summary, content, chunk_type, importance, last_accessed,
            access_count, created_at, project, oom_adj, info_class, lru_gen)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            chunk_id,
            quant_summary,
            content,
            "quantitative_evidence",
            0.90,
            "2026-04-01T00:00:00+00:00",
            0,
            "2026-04-01T00:00:00+00:00",
            "fts_test",
            -500,  # oom_adj 保护
            "world",
            0,
        ),
    )
    # 手动写入 FTS5（模拟 insert_chunk 的 FTS5 路径）
    from store_vfs import _cjk_tokenize, _normalize_structured_summary
    rowid = conn.execute(
        "SELECT rowid FROM memory_chunks WHERE id=?", (chunk_id,)
    ).fetchone()[0]
    fts_summary = _cjk_tokenize(_normalize_structured_summary(quant_summary))
    fts_content = _cjk_tokenize(_normalize_structured_summary(content))
    conn.execute(
        "INSERT INTO memory_chunks_fts(rowid_ref, summary, content) VALUES (?, ?, ?)",
        (str(rowid), fts_summary, fts_content)
    )
    conn.commit()

    # 按概念词查询
    results = fts_search(conn, "性能优化 延迟", "fts_test", top_k=10)
    result_ids = [r["id"] for r in results]
    assert chunk_id in result_ids, (
        f"quant chunk should be found by concept query '性能优化 延迟'. "
        f"concept_str={concept_str!r}. results={result_ids}"
    )
    conn.close()


# ──────────────────────────────────────────────────────────────────────
# 9. 多类别 summary → 最多 3 类概念
# ──────────────────────────────────────────────────────────────────────

def test_max_3_concept_categories():
    """
    summary 同时触发多个类别（检索+延迟+版本），最多返回 3 类。
    概念词拼接不应超长（每类≤60字），防止 content 膨胀。
    """
    s = "iter238 召回率 FTS5 hit: 50ms→5ms import latency 修复"
    result = _quant_semantic_concepts(s)
    # 结果应是 " | " 分隔的最多 3 组
    parts = [p.strip() for p in result.split("|")]
    assert len(parts) <= 3, f"max 3 concept categories, got {len(parts)}: {result}"
    assert result, "should not be empty"
