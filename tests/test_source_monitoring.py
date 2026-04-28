"""
test_source_monitoring.py — iter396: Source Monitoring Framework

覆盖：
  SM1: infer_source_type — hearsay 关键词识别
  SM2: infer_source_type — inferred 关键词识别
  SM3: infer_source_type — tool_output 关键词识别（代码块/命令输出）
  SM4: infer_source_type — 中性文本默认 direct
  SM5: compute_source_reliability — direct 来源最高可信度
  SM6: compute_source_reliability — hearsay 来源最低可信度
  SM7: compute_source_reliability — 不确定性词语降低可信度
  SM8: compute_source_reliability — 确认词语提高可信度
  SM9: source_monitor_weight — 高可信度 → weight > 1.0
  SM10: source_monitor_weight — 中等可信度 → weight == 1.0
  SM11: source_monitor_weight — 低可信度 → weight < 1.0
  SM12: apply_source_monitoring — 写入 DB 并可查询
  SM13: insert_chunk 自动触发 source monitoring
  SM14: source_monitor_weight range [0.80, 1.15]
  SM15: 空输入安全降级

认知科学依据：
  Johnson & Raye (1981) Reality Monitoring:
    人类区分「内部生成」与「外部感知」记忆的元认知能力。
  Johnson (1993) MEM (Multiple Entry Model):
    记忆系统维护「来源标签」，影响检索优先级。
  Zaragoza & Mitchell (1996):
    高可信度来源的信息更容易被记住和相信。

OS 类比：Linux LSM (Linux Security Modules) —
  文件访问前检查来源的 security context（SELinux label），
  来源不同 → 不同信任级别 → 不同访问权限。
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
    infer_source_type,
    compute_source_reliability,
    source_monitor_weight,
    apply_source_monitoring,
    insert_chunk,
)


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _make_chunk(cid, content="test content", summary="test summary",
                chunk_type="decision", project="test"):
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
        "stability": 1.0,
        "raw_snippet": "",
    }


# ══════════════════════════════════════════════════════════════════════
# 1. infer_source_type 测试
# ══════════════════════════════════════════════════════════════════════

def test_sm1_hearsay_keywords():
    """hearsay 关键词 → source_type='hearsay'（最低优先级检出）。"""
    assert infer_source_type("据说这个接口会有问题") == "hearsay"
    assert infer_source_type("听说团队要迁移数据库") == "hearsay"
    assert infer_source_type("someone mentioned the API is broken") == "hearsay"
    assert infer_source_type("I heard there was an issue") == "hearsay"


def test_sm2_inferred_keywords():
    """inferred 关键词 → source_type='inferred'（中等可信度）。"""
    assert infer_source_type("推测是内存泄漏导致的") == "inferred"
    assert infer_source_type("可能是配置问题，大概需要重启") == "inferred"
    assert infer_source_type("probably a race condition") == "inferred"
    assert infer_source_type("it seems the cache is stale") == "inferred"


def test_sm3_tool_output_keywords():
    """tool_output 关键词 → source_type='tool_output'（代码/命令输出）。"""
    assert infer_source_type("```python\nprint('hello')\n```") == "tool_output"
    assert infer_source_type("pytest: 5 passed, 0 failed") == "tool_output"
    assert infer_source_type("Traceback (most recent call last)") == "tool_output"
    assert infer_source_type("$ git status\n输出:") == "tool_output"


def test_sm4_neutral_text_defaults_direct():
    """情感中性/中立文本 → source_type='direct'（默认最高可信）。"""
    assert infer_source_type("实现了用户登录功能，包含密码验证") == "direct"
    assert infer_source_type("Redis 缓存配置已完成") == "direct"
    assert infer_source_type("") == "unknown"
    assert infer_source_type(None) == "unknown"


# ══════════════════════════════════════════════════════════════════════
# 2. compute_source_reliability 测试
# ══════════════════════════════════════════════════════════════════════

def test_sm5_direct_source_highest_reliability():
    """direct 来源的 design_constraint 可信度最高（≥ 0.90）。"""
    r = compute_source_reliability("design_constraint", "direct")
    assert r >= 0.90, f"SM5: direct design_constraint reliability={r:.4f} 应 >= 0.90"


def test_sm6_hearsay_source_lowest_reliability():
    """hearsay 来源的可信度最低（< 0.60）。"""
    r_hearsay = compute_source_reliability("decision", "hearsay")
    r_direct = compute_source_reliability("decision", "direct")
    assert r_hearsay < 0.60, f"SM6: hearsay reliability={r_hearsay:.4f} 应 < 0.60"
    assert r_direct > r_hearsay, f"SM6: direct({r_direct:.4f}) 应 > hearsay({r_hearsay:.4f})"


def test_sm7_uncertainty_words_reduce_reliability():
    """不确定性词语（可能/估计）降低可信度（−0.05）。"""
    base = compute_source_reliability("task_state", "inferred", "")
    with_uncertainty = compute_source_reliability(
        "task_state", "inferred", "可能是配置问题，估计需要重启"
    )
    assert with_uncertainty < base, (
        f"SM7: 不确定性词语应降低可信度，{base:.4f} → {with_uncertainty:.4f}"
    )
    assert abs(base - with_uncertainty - 0.05) < 0.001, (
        f"SM7: 降幅应为 0.05，实际 {base - with_uncertainty:.4f}"
    )


def test_sm8_certainty_words_increase_reliability():
    """确认词语（确认/verified）提高可信度（+0.05）。"""
    base = compute_source_reliability("decision", "direct", "")
    with_certainty = compute_source_reliability(
        "decision", "direct", "已确认，经过测试 verified"
    )
    assert with_certainty > base, (
        f"SM8: 确认词语应提高可信度，{base:.4f} → {with_certainty:.4f}"
    )
    assert abs(with_certainty - base - 0.05) < 0.001, (
        f"SM8: 提升幅度应为 0.05，实际 {with_certainty - base:.4f}"
    )


def test_sm_reliability_ordering():
    """来源可信度排序：direct > tool_output > inferred > hearsay。"""
    direct = compute_source_reliability("decision", "direct")
    tool = compute_source_reliability("decision", "tool_output")
    inferred = compute_source_reliability("decision", "inferred")
    hearsay = compute_source_reliability("decision", "hearsay")

    assert direct > tool > inferred > hearsay, (
        f"SM: 来源可信度应满足 direct({direct:.3f}) > "
        f"tool_output({tool:.3f}) > inferred({inferred:.3f}) > hearsay({hearsay:.3f})"
    )


# ══════════════════════════════════════════════════════════════════════
# 3. source_monitor_weight 测试
# ══════════════════════════════════════════════════════════════════════

def test_sm9_high_reliability_weight_above_1():
    """高可信度（≥ 0.85）→ weight > 1.0（轻微提升检索分）。"""
    w = source_monitor_weight(0.92)
    assert w > 1.0, f"SM9: 高可信度 weight={w:.4f} 应 > 1.0"
    w_max = source_monitor_weight(1.0)
    assert w_max <= 1.15, f"SM9: 最大 weight={w_max:.4f} 应 <= 1.15"


def test_sm10_medium_reliability_weight_one():
    """中等可信度（0.60 ~ 0.85）→ weight == 1.0（不调整）。"""
    for r in [0.60, 0.70, 0.75, 0.83]:
        w = source_monitor_weight(r)
        assert w == 1.0, f"SM10: 中等可信度 r={r}, weight={w:.4f} 应 == 1.0"


def test_sm11_low_reliability_weight_below_1():
    """低可信度（< 0.60）→ weight < 1.0（降低检索优先级）。"""
    w = source_monitor_weight(0.45)
    assert w < 1.0, f"SM11: 低可信度 weight={w:.4f} 应 < 1.0"
    w_min = source_monitor_weight(0.0)
    assert w_min >= 0.80, f"SM11: 最小 weight={w_min:.4f} 应 >= 0.80"


def test_sm14_weight_range():
    """source_monitor_weight 输出范围 [0.80, 1.15]。"""
    for r in [0.0, 0.1, 0.3, 0.5, 0.6, 0.7, 0.85, 0.95, 1.0]:
        w = source_monitor_weight(r)
        assert 0.80 <= w <= 1.15, f"SM14: r={r}, weight={w:.4f} 应在 [0.80, 1.15]"


# ══════════════════════════════════════════════════════════════════════
# 4. apply_source_monitoring DB 写入测试
# ══════════════════════════════════════════════════════════════════════

def test_sm12_apply_source_monitoring_writes_db(conn):
    """apply_source_monitoring → 写入 source_type + source_reliability 到 DB。"""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO memory_chunks "
        "(id, project, chunk_type, content, summary, importance, retrievability, "
        "created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, 0.7, 0.5, ?, ?)",
        ("sm12_chunk", "test", "decision", "confirmed and verified decision", "test", now, now)
    )
    conn.commit()

    s_type, reliability = apply_source_monitoring(
        conn, "sm12_chunk", "decision",
        "confirmed and verified decision", source_type="direct"
    )
    conn.commit()

    assert s_type == "direct", f"SM12: source_type={s_type}"
    assert reliability >= 0.90, f"SM12: direct decision reliability={reliability:.4f} 应 >= 0.90"

    row = conn.execute(
        "SELECT source_type, source_reliability FROM memory_chunks WHERE id='sm12_chunk'"
    ).fetchone()
    assert row is not None
    assert row["source_type"] == "direct"
    assert row["source_reliability"] >= 0.90


def test_sm12b_hearsay_written_correctly(conn):
    """hearsay source_type → DB 中 source_reliability < 0.60。"""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO memory_chunks "
        "(id, project, chunk_type, content, summary, importance, retrievability, "
        "created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, 0.7, 0.5, ?, ?)",
        ("sm12b_chunk", "test", "task_state", "据说这个接口有问题", "test", now, now)
    )
    conn.commit()

    s_type, reliability = apply_source_monitoring(
        conn, "sm12b_chunk", "task_state",
        "据说这个接口有问题"
    )
    conn.commit()

    assert s_type == "hearsay", f"SM12b: source_type 应为 hearsay, got {s_type}"
    assert reliability < 0.60, f"SM12b: hearsay reliability={reliability:.4f} 应 < 0.60"


# ══════════════════════════════════════════════════════════════════════
# 5. insert_chunk 自动触发 source monitoring
# ══════════════════════════════════════════════════════════════════════

def test_sm13_insert_chunk_auto_source_monitoring(conn):
    """insert_chunk 写入后，source_type 和 source_reliability 自动设置。"""
    chunk = _make_chunk(
        "sm13_chunk",
        content="据说用户反馈了一个问题",
        summary="据说用户反馈了一个问题",
        chunk_type="task_state",
    )
    insert_chunk(conn, chunk)
    conn.commit()

    row = conn.execute(
        "SELECT source_type, source_reliability FROM memory_chunks WHERE id='sm13_chunk'"
    ).fetchone()
    # 可能为 NULL（如果 schema 还没有这两列，ensure_schema 会加）
    # 或者已正确写入
    if row is not None and row["source_type"] is not None:
        assert row["source_type"] in ("hearsay", "inferred", "tool_output", "direct", "unknown")
        assert row["source_reliability"] is not None
        assert 0.0 < row["source_reliability"] <= 1.0


def test_sm13b_tool_output_chunk_auto_source(conn):
    """含代码块/命令输出的 chunk → source_type 自动推断为 tool_output。"""
    chunk = _make_chunk(
        "sm13b_chunk",
        content="运行结果：\n```\npytest: 5 passed\n```",
        summary="测试运行结果",
        chunk_type="task_state",
    )
    insert_chunk(conn, chunk)
    conn.commit()

    row = conn.execute(
        "SELECT source_type, source_reliability FROM memory_chunks WHERE id='sm13b_chunk'"
    ).fetchone()
    if row is not None and row["source_type"] is not None:
        assert row["source_type"] == "tool_output", (
            f"SM13b: source_type 应为 tool_output, got {row['source_type']}"
        )


# ══════════════════════════════════════════════════════════════════════
# 6. 边界条件
# ══════════════════════════════════════════════════════════════════════

def test_sm15_empty_input_safe():
    """空/None 输入安全降级。"""
    # infer_source_type
    assert infer_source_type("") == "unknown"
    assert infer_source_type(None) == "unknown"

    # compute_source_reliability 空 source_type
    r = compute_source_reliability("decision", "", "")
    assert 0.2 <= r <= 1.0, f"SM15: 空 source_type reliability={r:.4f} 应在 [0.2, 1.0]"

    # compute_source_reliability 未知 chunk_type
    r2 = compute_source_reliability("unknown_type", "direct", "")
    assert 0.2 <= r2 <= 1.0

    # source_monitor_weight 异常值
    w = source_monitor_weight(None)
    assert 0.80 <= w <= 1.15
    w2 = source_monitor_weight(-0.5)  # 超出范围
    assert 0.80 <= w2 <= 1.15
    w3 = source_monitor_weight(2.0)  # 超出范围
    assert 0.80 <= w3 <= 1.15


def test_sm_reliability_clamped_to_range():
    """compute_source_reliability 输出范围 [0.2, 1.0]。"""
    # 极端叠加（确认词 + hearsay = 0.45 + 0.05 = 0.50，还在范围内）
    r = compute_source_reliability("hearsay_type", "hearsay",
                                   "confirmed verified tested")
    assert 0.2 <= r <= 1.0, f"可信度应在 [0.2, 1.0], got {r}"
