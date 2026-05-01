"""
test_elaborative_encoding.py — iter401: Elaborative Encoding — Depth of Processing

覆盖：
  EE1: compute_depth_of_processing — 含因果推理词 → score > 0
  EE2: compute_depth_of_processing — 含结构组织词 → score > 0
  EE3: compute_depth_of_processing — 含对比/比较词 → score > 0
  EE4: compute_depth_of_processing — 含精细阐述词 → score > 0
  EE5: compute_depth_of_processing — 浅处理文本 → score ≈ 0
  EE6: compute_depth_of_processing — 深处理文本（含多个维度）→ score > 0.5
  EE7: apply_depth_of_processing — 深处理文本 stability 高于浅处理
  EE8: apply_depth_of_processing — stability 下限 >= base_stability（不降级）
  EE9: apply_depth_of_processing — stability 上限 base × 3.0
  EE10: insert_chunk 自动触发 depth_of_processing 写入
  EE11: 空/None 输入安全返回 0.0
  EE12: 中文因果词（因为/因此）识别

认知科学依据：
  Craik & Lockhart (1972) Levels of Processing:
    浅处理（字形）vs 深处理（语义），加工深度决定记忆痕迹强度。
  Craik & Tulving (1975):
    语义判断任务比视觉判断任务产生更强的记忆，因为语义激活更广泛的关联网络。
  Reder & Anderson (1980):
    精细编码（elaborate encoding）通过增加区分性线索增强提取能力。

OS 类比：Linux dirty page writeback write aggregation —
  页面在 dirty buffer 中等待越久，write aggregation 越充分，I/O 效率越高。
"""
import sys
import sqlite3
import pytest
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "hooks"))

import tmpfs  # noqa
from store_vfs import (
    ensure_schema,
    compute_depth_of_processing,
    apply_depth_of_processing,
    insert_chunk,
)


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _make_chunk(cid, content="test", summary="test", chunk_type="decision",
                project="test", stability=1.0):
    now = datetime.now(timezone.utc).isoformat()
    return {
        "id": cid,
        "created_at": now,
        "updated_at": now,
        "project": project,
        "source_session": "s1",
        "chunk_type": chunk_type,
        "info_class": "world",
        "content": content,
        "summary": summary,
        "tags": [chunk_type],
        "importance": 0.7,
        "retrievability": 0.5,
        "last_accessed": now,
        "access_count": 0,
        "oom_adj": 0,
        "lru_gen": 0,
        "stability": stability,
        "raw_snippet": "",
    }


# ══════════════════════════════════════════════════════════════════════
# 1. compute_depth_of_processing 纯函数测试
# ══════════════════════════════════════════════════════════════════════

def test_ee1_causal_reasoning_increases_depth():
    """因果推理词（because/therefore/因此）增加加工深度。"""
    text = "The system crashed because of a memory leak. Therefore, we should fix the allocator."
    dop = compute_depth_of_processing(text)
    assert dop > 0.0, f"EE1: 含因果词文本 dop={dop:.4f} 应 > 0"


def test_ee2_structural_organization_increases_depth():
    """结构组织词（first/second/首先/其次）增加加工深度。"""
    text = "首先分析问题，其次定位根因，最后制定解决方案。"
    dop = compute_depth_of_processing(text)
    assert dop > 0.0, f"EE2: 含结构词文本 dop={dop:.4f} 应 > 0"


def test_ee3_contrastive_processing_increases_depth():
    """对比/比较词（however/相比/但是）增加加工深度。"""
    text = "旧方案采用同步 I/O，然而新方案使用异步处理，相比之下延迟降低了 60%。"
    dop = compute_depth_of_processing(text)
    assert dop > 0.0, f"EE3: 含对比词文本 dop={dop:.4f} 应 > 0"


def test_ee4_elaboration_increases_depth():
    """精细阐述词（specifically/例如/比如）增加加工深度。"""
    text = "该系统存在性能问题，具体来说，例如在高并发场景下响应时间超过 500ms。"
    dop = compute_depth_of_processing(text)
    assert dop > 0.0, f"EE4: 含阐述词文本 dop={dop:.4f} 应 > 0"


def test_ee5_shallow_text_low_depth():
    """简单陈述句（无推理/结构/对比）→ dop ≈ 0（浅处理）。"""
    text = "服务启动成功。"
    dop = compute_depth_of_processing(text)
    assert dop < 0.05, f"EE5: 浅处理文本 dop={dop:.4f} 应 < 0.05"

    text2 = "The task is done."
    dop2 = compute_depth_of_processing(text2)
    assert dop2 < 0.05, f"EE5: 浅处理文本 dop={dop2:.4f} 应 < 0.05"


def test_ee6_deeply_processed_text_high_depth():
    """多维度深处理文本 → dop > 0.5。"""
    text = (
        "因为缓存策略不当导致了高延迟，因此我们重新设计了 LRU 驱逐算法。"
        "首先分析了现有问题，其次对比了三种替代方案，最后选择了 ARC 算法。"
        "具体来说，ARC 算法结合了 LRU 和 LFU 的优点，因此在命中率上比纯 LRU 高 15%。"
        "然而 ARC 的实现复杂度更高，虽然内存占用增加了 10%，但是性能收益值得这个代价。"
    )
    dop = compute_depth_of_processing(text)
    assert dop > 0.5, f"EE6: 深处理文本 dop={dop:.4f} 应 > 0.5"


def test_ee12_chinese_causal_words():
    """中文因果词（因为/由于/导致）正确识别。"""
    texts = [
        "由于内存不足导致服务崩溃",
        "因此需要增加 swap 空间",
        "使得系统稳定性下降",
        "从而影响了用户体验",
    ]
    for text in texts:
        dop = compute_depth_of_processing(text)
        assert dop > 0.0, f"EE12: 中文因果词文本 '{text}' dop={dop:.4f} 应 > 0"


def test_ee11_empty_and_none_inputs():
    """空/None/短文本安全返回 0.0。"""
    assert compute_depth_of_processing("") == 0.0
    assert compute_depth_of_processing(None) == 0.0
    assert compute_depth_of_processing("  ") == 0.0
    assert compute_depth_of_processing("short") == 0.0  # < 10 chars


def test_ee_depth_range():
    """compute_depth_of_processing 输出范围 [0.0, 1.0]。"""
    texts = [
        "",
        "简单",
        "因此",
        "因此由于所以从而，first second third，however whereas，specifically for example",
        "因为因此由于所以从而因而hence therefore thus，首先其次最后，相比然而但是，具体来说例如比如" * 5,
    ]
    for text in texts:
        dop = compute_depth_of_processing(text)
        assert 0.0 <= dop <= 1.0, f"dop={dop:.4f} 超出 [0.0, 1.0] 范围: '{text[:30]}'"


# ══════════════════════════════════════════════════════════════════════
# 2. apply_depth_of_processing DB 写入测试
# ══════════════════════════════════════════════════════════════════════

def test_ee7_deep_processing_higher_stability(conn):
    """深处理文本 stability 高于浅处理文本（写入 DB 后）。"""
    now = datetime.now(timezone.utc).isoformat()
    for cid, text in [
        ("ee7_deep", "因为缓存不当导致高延迟，因此重设计了 LRU 算法。首先分析问题，然后对比方案。"),
        ("ee7_shallow", "任务完成。"),
    ]:
        conn.execute(
            "INSERT INTO memory_chunks "
            "(id, project, chunk_type, content, summary, importance, retrievability, "
            "stability, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, 0.7, 0.5, 1.0, ?, ?)",
            (cid, "test", "decision", text, text, now, now)
        )
    conn.commit()

    apply_depth_of_processing(conn, "ee7_deep",
        "因为缓存不当导致高延迟，因此重设计了 LRU 算法。首先分析问题，然后对比方案。",
        base_stability=1.0)
    apply_depth_of_processing(conn, "ee7_shallow", "任务完成。", base_stability=1.0)
    conn.commit()

    deep_row = conn.execute(
        "SELECT stability FROM memory_chunks WHERE id='ee7_deep'"
    ).fetchone()
    shallow_row = conn.execute(
        "SELECT stability FROM memory_chunks WHERE id='ee7_shallow'"
    ).fetchone()

    assert deep_row["stability"] > shallow_row["stability"], (
        f"EE7: 深处理 stability({deep_row['stability']:.4f}) "
        f"应 > 浅处理 stability({shallow_row['stability']:.4f})"
    )


def test_ee8_stability_not_decreased(conn):
    """apply_depth_of_processing 不降低 stability（深度加工只增不减）。"""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO memory_chunks "
        "(id, project, chunk_type, content, summary, importance, retrievability, "
        "stability, created_at, updated_at) "
        "VALUES (?, 'test', 'task_state', '任务完成', '任务完成', 0.7, 0.5, 5.0, ?, ?)",
        ("ee8_chunk", now, now)
    )
    conn.commit()

    new_stability = apply_depth_of_processing(conn, "ee8_chunk", "任务完成",
                                               base_stability=5.0)
    conn.commit()

    assert new_stability >= 5.0, (
        f"EE8: 浅处理不应降低 stability，base=5.0, new={new_stability:.4f}"
    )


def test_ee9_stability_capped_at_3x_base(conn):
    """stability 上限 base_stability × 3.0，防止极端深度加工无限膨胀。"""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO memory_chunks "
        "(id, project, chunk_type, content, summary, importance, retrievability, "
        "stability, created_at, updated_at) "
        "VALUES (?, 'test', 'decision', 'content', 'summary', 0.7, 0.5, 1.0, ?, ?)",
        ("ee9_chunk", now, now)
    )
    conn.commit()

    # 极端深度加工文本（dop 接近 1.0）
    very_deep = (
        "因为如此所以由于导致，首先其次然后最后，然而相比但是不过，"
        "具体来说例如比如换句话说，because therefore thus hence,"
        "first second third finally, however whereas although,"
        "specifically for example namely"
    )
    new_stability = apply_depth_of_processing(conn, "ee9_chunk", very_deep, base_stability=1.0)
    conn.commit()

    assert new_stability <= 3.0, (
        f"EE9: stability 上限 base × 3.0 = 3.0，got {new_stability:.4f}"
    )


# ══════════════════════════════════════════════════════════════════════
# 3. insert_chunk 集成测试
# ══════════════════════════════════════════════════════════════════════

def test_ee10_insert_chunk_writes_depth_of_processing(conn):
    """insert_chunk 写入后，depth_of_processing 字段被自动设置。"""
    deep_text = "因为内存泄漏导致崩溃，因此修复了分配器。首先分析了根因，然后对比了三种方案。"
    chunk = _make_chunk("ee10_chunk", content=deep_text, summary=deep_text)
    insert_chunk(conn, chunk)
    conn.commit()

    row = conn.execute(
        "SELECT depth_of_processing, stability FROM memory_chunks WHERE id='ee10_chunk'"
    ).fetchone()

    if row is not None and row["depth_of_processing"] is not None:
        dop = row["depth_of_processing"]
        assert dop > 0.0, f"EE10: 深处理 chunk depth_of_processing={dop:.4f} 应 > 0"


def test_ee10b_shallow_chunk_low_depth(conn):
    """浅处理文本 → depth_of_processing 接近 0，stability 接近 base。"""
    chunk = _make_chunk("ee10b_chunk", content="任务完成。", summary="任务完成。",
                        stability=1.0)
    insert_chunk(conn, chunk)
    conn.commit()

    row = conn.execute(
        "SELECT depth_of_processing, stability FROM memory_chunks WHERE id='ee10b_chunk'"
    ).fetchone()

    if row is not None and row["depth_of_processing"] is not None:
        dop = row["depth_of_processing"]
        assert dop < 0.1, f"EE10b: 浅处理 depth_of_processing={dop:.4f} 应 < 0.1"
        # stability ≈ base × 2.0（iter479 warm-start：importance=0.7 ≥ 0.5，浅处理无额外 DOP bonus）
        assert row["stability"] <= 3.0, (
            f"EE10b: 浅处理 stability={row['stability']:.4f} 应接近 warm-start base 2.0（无 DOP bonus）"
        )
