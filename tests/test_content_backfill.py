"""
test_content_backfill.py — 迭代326：content 富化回填单元测试

验证：
  1. 贫 content 检测（content ≈ summary → 需要富化）
  2. 富 content 跳过（已有邻居拼接的不重复处理）
  3. causal_chain 富化：中间节点用 → 拼接前后邻居
  4. quantitative_evidence 富化：中间节点用 | 拼接前后邻居
  5. 单节点：无邻居时 content 只含当前（无分隔符）
  6. 首节点：只有后邻居
  7. 末节点：只有前邻居
  8. content 不超过 400 chars
  9. topic 正确嵌入 content 前缀
 10. 富化后 content 长度 > 原 content 长度
"""
import sys
import sqlite3
import pytest
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "hooks"))

import tmpfs  # noqa

from store_vfs import ensure_schema, insert_chunk


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _make_chunk(cid, chunk_type, summary, content, project="test"):
    now = _now_iso()
    return {
        "id": cid,
        "created_at": now,
        "updated_at": now,
        "project": project,
        "source_session": "s1",
        "chunk_type": chunk_type,
        "info_class": "episodic" if chunk_type == "causal_chain" else "semantic",
        "content": content,
        "summary": summary,
        "tags": [chunk_type],
        "importance": 0.90,
        "retrievability": 0.2,
        "last_accessed": now,
        "access_count": 0,
        "oom_adj": 0,
        "lru_gen": 0,
        "stability": 1.8,
        "raw_snippet": "",
        "encoding_context": {},
    }


# 复用 backfill 工具函数
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from backfill_content import _is_rich_content, _build_rich_content


# ══════════════════════════════════════════════════════════════════════
# 1. 贫/富 content 检测
# ══════════════════════════════════════════════════════════════════════

def test_poor_content_detected_causal():
    """causal_chain 贫 content（无 →，长度=summary+前缀）→ 需要富化。"""
    summary = "因为 A 导致了 B"
    content = f"[causal_chain] {summary}"
    assert not _is_rich_content(content, summary, "causal_chain"), \
        "贫 content 应被标记为需要富化"


def test_poor_content_detected_quant():
    """quantitative_evidence 贫 content（无 |）→ 需要富化。"""
    summary = "FTS5<10ms"
    content = f"[quantitative_evidence] {summary}"
    assert not _is_rich_content(content, summary, "quantitative_evidence")


def test_rich_content_skipped_causal():
    """已有 → 分隔符且 content 明显长于 summary → 跳过。"""
    summary = "因为 A 导致了 B"
    content = "[causal_chain] 原因是 X → 因为 A 导致了 B → 结果是 C"
    assert _is_rich_content(content, summary, "causal_chain"), \
        "富 content 不应被重复处理"


def test_rich_content_skipped_quant():
    """quantitative_evidence 已有 | 分隔符 → 跳过。"""
    summary = "FTS5<10ms"
    content = "[quantitative_evidence] BM25<50ms | FTS5<10ms | P50<30ms"
    assert _is_rich_content(content, summary, "quantitative_evidence")


# ══════════════════════════════════════════════════════════════════════
# 2. 富化内容构建
# ══════════════════════════════════════════════════════════════════════

def test_causal_middle_node_has_arrow_neighbors():
    """causal_chain 中间节点用 → 拼接前后邻居。"""
    summaries = ["原因是 X", "因为 A 导致了 B", "结果是 C"]
    content = _build_rich_content(summaries, 1, "causal_chain", "")
    assert "原因是 X" in content
    assert "因为 A 导致了 B" in content
    assert "结果是 C" in content
    assert " → " in content


def test_quant_middle_node_has_pipe_neighbors():
    """quantitative_evidence 中间节点用 | 拼接前后邻居。"""
    summaries = ["BM25<50ms", "FTS5<10ms", "P50<30ms"]
    content = _build_rich_content(summaries, 1, "quantitative_evidence", "")
    assert "BM25<50ms" in content
    assert "FTS5<10ms" in content
    assert "P50<30ms" in content
    assert " | " in content


def test_single_node_no_separator():
    """单节点：无邻居，content 只含自身（无 → 或 |）。"""
    summaries = ["唯一一条因果"]
    cc_content = _build_rich_content(summaries, 0, "causal_chain", "")
    assert "唯一一条因果" in cc_content
    assert " → " not in cc_content

    summaries2 = ["唯一量化数据"]
    qe_content = _build_rich_content(summaries2, 0, "quantitative_evidence", "")
    assert "唯一量化数据" in qe_content
    assert " | " not in qe_content


def test_first_node_only_next_neighbor():
    """首节点：只拼接后邻居（无前邻居）。"""
    summaries = ["首节点", "第二节点", "第三节点"]
    content = _build_rich_content(summaries, 0, "causal_chain", "")
    assert "首节点" in content
    assert "第二节点" in content
    assert "第三节点" not in content
    # 只有一个 →（self + next）
    assert content.count(" → ") == 1


def test_last_node_only_prev_neighbor():
    """末节点：只拼接前邻居（无后邻居）。"""
    summaries = ["第一节点", "第二节点", "末节点"]
    content = _build_rich_content(summaries, 2, "causal_chain", "")
    assert "第一节点" not in content
    assert "第二节点" in content
    assert "末节点" in content
    assert content.count(" → ") == 1


def test_content_within_400_chars():
    """富化后 content 不超过 400 chars（[:400] 截断保护）。"""
    long_summaries = ["A" * 150, "B" * 150, "C" * 150]
    for chunk_type in ["causal_chain", "quantitative_evidence"]:
        content = _build_rich_content(long_summaries, 1, chunk_type, "")
        assert len(content) <= 400, f"{chunk_type} content 超过 400: {len(content)}"


def test_topic_prefix_embedded():
    """topic 正确嵌入 content 前缀。"""
    summaries = ["因为 X 导致 Y"]
    content = _build_rich_content(summaries, 0, "causal_chain", "memory-os")
    assert "[causal_chain|memory-os]" in content

    content2 = _build_rich_content(summaries, 0, "quantitative_evidence", "perf")
    assert "[quantitative_evidence|perf]" in content2


def test_no_topic_uses_default_prefix():
    """无 topic 时使用默认前缀。"""
    summaries = ["因为 X 导致 Y"]
    content = _build_rich_content(summaries, 0, "causal_chain", "")
    assert content.startswith("[causal_chain]")


def test_rich_content_longer_than_poor():
    """富化后 content 长度 > 原贫 content。"""
    summaries = ["第一条量化数据：FTS5<10ms", "第二条：BM25<50ms", "第三条：P50<30ms"]
    for idx in range(3):
        poor = f"[quantitative_evidence] {summaries[idx]}"
        rich = _build_rich_content(summaries, idx, "quantitative_evidence", "")
        if idx in (0, 2):  # 首/末节点有 1 个邻居
            assert len(rich) > len(poor), f"idx={idx} 富化后应更长"
        else:  # 中间节点有 2 个邻居
            assert len(rich) > len(poor) + 10, f"idx={idx} 中间节点富化后应显著更长"


# ══════════════════════════════════════════════════════════════════════
# iter329: conversation_summary 富化测试
# ══════════════════════════════════════════════════════════════════════

def test_conv_summary_poor_content_detected():
    """conversation_summary 贫 content（无 |）→ 需要富化。"""
    summary = "已完成 iter326 causal_chain content 富化"
    content = f"[conversation_summary] {summary}"
    assert not _is_rich_content(content, summary, "conversation_summary"), \
        "贫 content 应被标记为需要富化"


def test_conv_summary_rich_content_skipped():
    """conversation_summary 已有 | 分隔符且 content 显著长于 summary → 跳过。"""
    summary = "已完成 iter326 causal_chain content 富化"
    content = "[conversation_summary] 上一步操作 | 已完成 iter326 causal_chain content 富化 | 下一步继续"
    assert _is_rich_content(content, summary, "conversation_summary"), \
        "富 content 不应被重复处理"


def test_conv_summary_middle_node_has_pipe_neighbors():
    """conversation_summary 中间节点用 | 拼接前后邻居。"""
    summaries = ["完成了 A 模块", "修复了 B 的 bug", "更新了 C 的配置"]
    content = _build_rich_content(summaries, 1, "conversation_summary", "")
    assert "完成了 A 模块" in content
    assert "修复了 B 的 bug" in content
    assert "更新了 C 的配置" in content
    assert " | " in content


def test_conv_summary_single_node_no_separator():
    """单节点 conversation_summary：无邻居，content 只含自身。"""
    summaries = ["完成了唯一操作"]
    content = _build_rich_content(summaries, 0, "conversation_summary", "")
    assert "完成了唯一操作" in content
    assert " | " not in content


def test_conv_summary_content_within_400_chars():
    """富化后 conversation_summary content 不超过 400 chars。"""
    long_summaries = ["A" * 140, "B" * 140, "C" * 140]
    content = _build_rich_content(long_summaries, 1, "conversation_summary", "")
    assert len(content) <= 400


def test_conv_summary_topic_prefix():
    """topic 正确嵌入 conversation_summary content 前缀。"""
    summaries = ["完成了某项任务"]
    content = _build_rich_content(summaries, 0, "conversation_summary", "memory-os")
    assert "[conversation_summary|memory-os]" in content


def test_conv_summary_richer_than_poor():
    """iter329：conversation_summary 富化后 content > 原贫 content。"""
    summaries = ["完成了 FTS5 索引重建", "修复了 causal_chain 召回低的问题", "更新了 backfill_content.py"]
    for idx in range(3):
        poor = f"[conversation_summary] {summaries[idx]}"
        rich = _build_rich_content(summaries, idx, "conversation_summary", "")
        if idx == 1:  # 中间节点
            assert len(rich) > len(poor) + 10, f"中间节点应显著更长"
        else:  # 首/末节点
            assert len(rich) > len(poor), f"idx={idx} 富化后应更长"
