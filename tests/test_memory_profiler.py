#!/usr/bin/env python3
"""
迭代359/Task11 测试：memory_profile() — Per-Project Memory Profiler

OS 类比：/proc/[pid]/smaps — 进程级精细内存映射统计。

测试矩阵：
  T1: memory_profile() 空库返回完整 schema，不崩溃
  T2: summary 字段准确（total/pinned/zero_access/stale）
  T3: pin_analysis 正确统计 hard/soft pin
  T4: swap_analysis 反映 swap 情况
  T5: access_distribution 4 个桶计数正确
  T6: importance_distribution 4 个桶正确
  T7: type_breakdown 按类型分析正确
  T8: session_dedup.dedup_candidates 准确统计高频 chunk
  T9: recommendations 对问题库产生有效建议
  T10: project=None 全局汇总模式
"""
import sys
import os
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone, timedelta

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

import tmpfs  # noqa: F401
from store import open_db, ensure_schema, insert_chunk, memory_profile, pin_chunk


PROJECT = f"test_profiler_{uuid.uuid4().hex[:6]}"
PROJECT_B = f"test_profiler_b_{uuid.uuid4().hex[:6]}"


def _chunk(project, chunk_type="decision", importance=0.5,
           days_ago=1, access_count=0, retrievability=0.3) -> dict:
    now = datetime.now(timezone.utc) - timedelta(days=days_ago)
    ts = now.isoformat()
    cid = str(uuid.uuid4())
    return {
        "id": cid,
        "created_at": ts,
        "updated_at": ts,
        "project": project,
        "source_session": "test",
        "chunk_type": chunk_type,
        "content": f"content {cid[:8]}",
        "summary": f"summary {cid[:8]}",
        "tags": json.dumps(["test"]),
        "importance": importance,
        "retrievability": retrievability,
        "last_accessed": ts,
        "feishu_url": None,
        "access_count": access_count,
    }


def _setup():
    conn = open_db()
    ensure_schema(conn)
    conn.execute("DELETE FROM memory_chunks WHERE project IN (?, ?)", (PROJECT, PROJECT_B))
    conn.execute("DELETE FROM chunk_pins WHERE project IN (?, ?)", (PROJECT, PROJECT_B))
    try:
        conn.execute("DELETE FROM swap_chunks WHERE project IN (?, ?)", (PROJECT, PROJECT_B))
    except Exception:
        pass
    conn.commit()
    return conn


def _teardown(conn):
    conn.execute("DELETE FROM memory_chunks WHERE project IN (?, ?)", (PROJECT, PROJECT_B))
    conn.execute("DELETE FROM chunk_pins WHERE project IN (?, ?)", (PROJECT, PROJECT_B))
    try:
        conn.execute("DELETE FROM swap_chunks WHERE project IN (?, ?)", (PROJECT, PROJECT_B))
    except Exception:
        pass
    conn.commit()
    conn.close()


def test_01_empty_db():
    """T1: 空项目返回完整 schema，不崩溃"""
    conn = _setup()
    profile = memory_profile(conn, project=f"nonexistent_{uuid.uuid4().hex}")
    # 必须包含所有顶层 key
    required = ["project", "summary", "pin_analysis", "swap_analysis",
                "ksm_analysis", "access_distribution", "importance_distribution",
                "session_dedup", "type_breakdown", "recommendations"]
    for key in required:
        assert key in profile, f"Missing key: {key}"
    assert profile["summary"]["total"] == 0
    assert profile["recommendations"]  # 非空列表
    _teardown(conn)
    print("  T1 ✓ empty project returns complete schema")


def test_02_summary_accuracy():
    """T2: summary 字段准确"""
    conn = _setup()
    # 5 个 chunk：3 个活跃（1天前），2 个 stale（40天前）
    for i in range(3):
        c = _chunk(PROJECT, days_ago=1, access_count=i + 1)
        insert_chunk(conn, c)
    for i in range(2):
        c = _chunk(PROJECT, days_ago=40, access_count=0)
        insert_chunk(conn, c)
    conn.commit()

    profile = memory_profile(conn, project=PROJECT)
    s = profile["summary"]
    assert s["total"] == 5, f"total={s['total']}"
    assert s["stale_7d"] == 2, f"stale_7d={s['stale_7d']}"
    assert s["active_7d"] == 3, f"active_7d={s['active_7d']}"
    assert s["zero_access"] == 2, f"zero_access={s['zero_access']}"
    _teardown(conn)
    print(f"  T2 ✓ summary accurate (total={s['total']}, stale={s['stale_7d']})")


def test_03_pin_analysis():
    """T3: pin_analysis 正确统计 hard/soft pin"""
    conn = _setup()
    c_hard = _chunk(PROJECT, importance=0.9)
    c_soft = _chunk(PROJECT, importance=0.7)
    c_none = _chunk(PROJECT, importance=0.5)
    for c in (c_hard, c_soft, c_none):
        insert_chunk(conn, c)
    pin_chunk(conn, c_hard["id"], PROJECT, "hard")
    pin_chunk(conn, c_soft["id"], PROJECT, "soft")
    conn.commit()

    profile = memory_profile(conn, project=PROJECT)
    pa = profile["pin_analysis"]
    assert pa["hard"] == 1, f"hard={pa['hard']}"
    assert pa["soft"] == 1, f"soft={pa['soft']}"
    assert pa["total_pinned"] == 2
    assert pa["unpinned"] == 1
    assert pa["pin_rate_pct"] > 0
    _teardown(conn)
    print(f"  T3 ✓ pin_analysis: hard={pa['hard']}, soft={pa['soft']}")


def test_04_swap_analysis():
    """T4: swap_analysis 反映 swap 情况（基于 swap_chunks 表）"""
    conn = _setup()
    c = _chunk(PROJECT)
    insert_chunk(conn, c)
    conn.commit()

    profile = memory_profile(conn, project=PROJECT)
    sa = profile["swap_analysis"]
    assert "total_swapped" in sa
    assert "swap_ratio_pct" in sa
    assert "active_in_memory" in sa
    assert sa["active_in_memory"] == 1
    _teardown(conn)
    print(f"  T4 ✓ swap_analysis: swapped={sa['total_swapped']}, ratio={sa['swap_ratio_pct']}%")


def test_05_access_distribution():
    """T5: access_distribution 4 个桶计数正确"""
    conn = _setup()
    # zero: 2, low(1-5): 2, mid(6-20): 2, high(21+): 2
    for ac in [0, 0, 3, 5, 10, 15, 25, 100]:
        c = _chunk(PROJECT, access_count=ac)
        insert_chunk(conn, c)
    # 手动更新 access_count（因为 insert_chunk 可能不写 access_count 字段）
    for ac in [0, 0, 3, 5, 10, 15, 25, 100]:
        pass
    # 用 UPDATE 确保 access_count 正确写入
    chunks = conn.execute(
        "SELECT id FROM memory_chunks WHERE project=? ORDER BY created_at", (PROJECT,)
    ).fetchall()
    for i, (cid,) in enumerate(chunks):
        conn.execute("UPDATE memory_chunks SET access_count=? WHERE id=?",
                     ([0, 0, 3, 5, 10, 15, 25, 100][i], cid))
    conn.commit()

    profile = memory_profile(conn, project=PROJECT)
    ad = profile["access_distribution"]
    assert ad["zero"] == 2, f"zero={ad['zero']}"
    assert ad["low_1_5"] == 2, f"low_1_5={ad['low_1_5']}"
    assert ad["mid_6_20"] == 2, f"mid_6_20={ad['mid_6_20']}"
    assert ad["high_21plus"] == 2, f"high_21plus={ad['high_21plus']}"
    _teardown(conn)
    print(f"  T5 ✓ access_distribution: {ad}")


def test_06_importance_distribution():
    """T6: importance_distribution 4 个桶正确"""
    conn = _setup()
    for imp in [0.1, 0.2, 0.4, 0.5, 0.65, 0.75, 0.85, 0.95]:
        c = _chunk(PROJECT, importance=imp)
        insert_chunk(conn, c)
    conn.commit()

    profile = memory_profile(conn, project=PROJECT)
    id_ = profile["importance_distribution"]
    assert id_["low_0_0.3"] == 2
    assert id_["mid_0.3_0.6"] == 2
    assert id_["high_0.6_0.8"] == 2
    assert id_["critical_0.8plus"] == 2
    _teardown(conn)
    print(f"  T6 ✓ importance_distribution: {id_}")


def test_07_type_breakdown():
    """T7: type_breakdown 按类型分析"""
    conn = _setup()
    for chunk_type in ["decision", "decision", "reasoning_chain", "design_constraint"]:
        c = _chunk(PROJECT, chunk_type=chunk_type, importance=0.7)
        insert_chunk(conn, c)
    conn.commit()

    profile = memory_profile(conn, project=PROJECT)
    tb = profile["type_breakdown"]
    assert "decision" in tb
    assert tb["decision"]["count"] == 2
    assert tb["decision"]["pct"] == 50.0  # 2/4
    assert "reasoning_chain" in tb
    assert "design_constraint" in tb
    _teardown(conn)
    print(f"  T7 ✓ type_breakdown: {list(tb.keys())}")


def test_08_session_dedup_candidates():
    """T8: session_dedup.dedup_candidates 准确统计高频 chunk"""
    conn = _setup()
    # 高频 (access >= 3*threshold=6): 3 个
    for _ in range(3):
        c = _chunk(PROJECT, access_count=10)
        cid = c["id"]
        insert_chunk(conn, c)
        conn.execute("UPDATE memory_chunks SET access_count=10 WHERE id=?", (cid,))
    # 低频: 2 个
    for _ in range(2):
        c = _chunk(PROJECT, access_count=1)
        cid = c["id"]
        insert_chunk(conn, c)
        conn.execute("UPDATE memory_chunks SET access_count=1 WHERE id=?", (cid,))
    conn.commit()

    profile = memory_profile(conn, project=PROJECT)
    sd = profile["session_dedup"]
    assert sd["dedup_threshold"] == 2
    assert sd["dedup_candidates"] == 3, f"Expected 3, got {sd['dedup_candidates']}"
    assert sd["estimated_tokens_saved_per_call"] == 30  # 3 * 10
    _teardown(conn)
    print(f"  T8 ✓ session_dedup: candidates={sd['dedup_candidates']}, "
          f"tokens_saved={sd['estimated_tokens_saved_per_call']}/call")


def test_09_recommendations():
    """T9: recommendations 对问题库产生有效建议"""
    conn = _setup()
    # 插入大量零访问 chunk (> 30%)
    for i in range(7):
        c = _chunk(PROJECT, days_ago=30, access_count=0)
        insert_chunk(conn, c)
    for i in range(3):
        c = _chunk(PROJECT, days_ago=1, access_count=5)
        cid = c["id"]
        insert_chunk(conn, c)
        conn.execute("UPDATE memory_chunks SET access_count=5 WHERE id=?", (cid,))
    conn.commit()

    profile = memory_profile(conn, project=PROJECT)
    recs = profile["recommendations"]
    assert len(recs) > 0
    # 至少有一个 ⚠️ 建议（零访问率 > 30%）
    has_warning = any("⚠️" in r or "💡" in r for r in recs)
    assert has_warning, f"Expected warnings/tips, got: {recs}"
    _teardown(conn)
    print(f"  T9 ✓ recommendations: {recs[0][:60]}...")


def test_10_global_mode():
    """T10: project=None 全局汇总模式"""
    conn = _setup()
    # 插入两个项目的数据
    for ct in ["decision", "reasoning_chain"]:
        insert_chunk(conn, _chunk(PROJECT, chunk_type=ct))
        insert_chunk(conn, _chunk(PROJECT_B, chunk_type=ct))
    conn.commit()

    profile_a = memory_profile(conn, project=PROJECT)
    profile_all = memory_profile(conn, project=None)

    assert profile_all["project"] == "*all*"
    assert profile_all["summary"]["total"] >= profile_a["summary"]["total"]
    assert profile_all["summary"]["total"] >= 4  # 至少有我们插入的4个

    _teardown(conn)
    print(f"  T10 ✓ global mode: total={profile_all['summary']['total']} "
          f"(>= project_a={profile_a['summary']['total']})")


if __name__ == "__main__":
    print("迭代359/Task11 测试：memory_profile() — Per-Project Profiler")
    print("=" * 60)

    tests = [
        test_01_empty_db,
        test_02_summary_accuracy,
        test_03_pin_analysis,
        test_04_swap_analysis,
        test_05_access_distribution,
        test_06_importance_distribution,
        test_07_type_breakdown,
        test_08_session_dedup_candidates,
        test_09_recommendations,
        test_10_global_mode,
    ]

    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            failed += 1
            import traceback
            print(f"  FAIL {t.__name__}: {e}")
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"结果：{passed}/{passed+failed} 通过")
    if failed:
        import sys
        sys.exit(1)
