#!/usr/bin/env python3
"""
test_iter494_autotune_circuit_breaker.py — Autotune Circuit Breaker 测试（iter494）

覆盖：
  CB1: 首次 autotune（无历史）→ circuit 初始化为 closed, consecutive_bad=0
  CB2: 连续 N 次 hit_rate 下降（恶化）→ 触发熔断（circuit open），参数回滚
  CB3: 非恶化调整 → consecutive_bad 保持 0，circuit 不打开
  CB4: 熔断 open 状态下 → autotune 被跳过（skipped_reason=circuit_open）
  CB5: open 状态经 cb_open_hours 后 → 允许 half_open 试探
  CB6: half_open 试探成功 → circuit 关闭，恢复正常
"""
import sys
import json
import uuid
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import tmpfs  # noqa

from store import open_db, ensure_schema
from store_mm import autotune, _autotune_load_state, _autotune_save_state


def _mock_traces(conn, project, hit_rate_pct: float, n: int = 15):
    """插入 n 条模拟 recall_traces，命中率约为 hit_rate_pct%。"""
    hit_count = int(n * hit_rate_pct / 100)
    now = datetime.now(timezone.utc)
    for i in range(n):
        injected = 1 if i < hit_count else 0
        ts = (now - timedelta(minutes=n - i)).isoformat()
        conn.execute(
            """INSERT INTO recall_traces
               (project, session_id, timestamp, reason, injected, duration_ms, top_k_json, prompt_hash)
               VALUES (?,?,?,?,?,?,?,?)""",
            (project, "sess1", ts, "test", injected, 10.0, "[]", f"h{i:04d}"),
        )
    conn.commit()


def _mock_chunks(conn, project, n: int = 30):
    """插入 n 个 chunk 让 autotune 不被 small_pool bypass。"""
    now = datetime.now(timezone.utc).isoformat()
    for i in range(n):
        cid = f"cb_chunk_{i:03d}_{uuid.uuid4().hex[:4]}"
        conn.execute(
            """INSERT OR REPLACE INTO memory_chunks
               (id, project, source_session, chunk_type, summary, content,
                importance, stability, retrievability, info_class, tags,
                access_count, oom_adj, created_at, updated_at, last_accessed,
                feishu_url, lru_gen, raw_snippet, encode_context,
                session_type_history, confidence_score, verification_status)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (cid, project, "s1", "decision", f"cb {i}", f"cb {i}",
             0.7, 1.5, 0.5, "episodic", json.dumps([]),
             3, 0, now, now, now,
             None, 2, f"cb {i}", "{}", "", 0.8, "pending"),
        )
    conn.commit()


def _force_autotune_state(project: str, hit_rate_pct: float,
                           circuit: dict, hours_ago: float = 7.0):
    """直接写入 autotune_state，模拟历史状态。"""
    ts = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    _autotune_save_state(
        project,
        stats={"hit_rate_pct": hit_rate_pct, "sample_count": 15,
               "avg_latency_ms": 20.0, "p95_latency_ms": 30.0,
               "chunk_count": 30, "current_overrides": 0},
        adjustments=[],
        circuit=circuit,
    )
    # 修正时间戳（save_state 会用 now，需要手动覆盖）
    from store_mm import _AUTOTUNE_STATE_FILE
    import json as _json
    data = _json.loads(_AUTOTUNE_STATE_FILE.read_text())
    data[project]["timestamp"] = ts
    _AUTOTUNE_STATE_FILE.write_text(_json.dumps(data, ensure_ascii=False, indent=2))


def test_cb1_first_run_no_circuit():
    """CB1: 首次运行无历史 → circuit 初始化，consecutive_bad=0，state=closed。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"cb1_{uuid.uuid4().hex[:8]}"
    _mock_chunks(conn, proj, 50)
    _mock_traces(conn, proj, hit_rate_pct=60.0, n=20)
    conn.commit()

    result = autotune(conn, proj)
    conn.close()

    cb_state = result.get("circuit_state", "closed")
    cb_bad = result.get("consecutive_bad", 0)
    assert cb_state == "closed", f"CB1: 首次运行应为 closed, got {cb_state}"
    assert cb_bad == 0, f"CB1: consecutive_bad 应为 0, got {cb_bad}"
    print(f"  CB1 PASS: first run → circuit={cb_state}, consecutive_bad={cb_bad}")


def test_cb2_consecutive_bad_triggers_open():
    """CB2: 模拟连续 3 次 hit_rate 下降（有调整） → 第 3 次触发熔断（circuit open）。
    使用 hit_rate=10%（< hit_rate_low_pct=20%）确保 autotune 产生 top_k 扩大调整。
    历史 hit_rate=50%，本次 10%，下降 80% >> 10% 阈值。
    """
    conn = open_db()
    ensure_schema(conn)
    proj = f"cb2_{uuid.uuid4().hex[:8]}"
    _mock_chunks(conn, proj, 50)

    # 强制 N-1 次恶化历史（consecutive_bad = cb_consecutive_bad - 1 = 2）
    # 注意：param_snapshot 记录的是调整前的值（top_k=5，会被 autotune 扩大）
    _force_autotune_state(proj, hit_rate_pct=50.0,
                           circuit={"state": "closed", "consecutive_bad": 2,
                                    "param_snapshot": {"retriever.top_k": 5}},
                           hours_ago=7.0)

    # 第 3 次：hit_rate 从 50% 跌到 10%（下降 80% >> 10% 阈值）
    # hit_rate=10% < hit_rate_low_pct=20% → autotune 会扩大 top_k（产生调整）
    _mock_traces(conn, proj, hit_rate_pct=10.0, n=20)
    conn.commit()

    result = autotune(conn, proj)
    conn.close()

    cb_state = result.get("circuit_state", "closed")
    assert cb_state == "open", (
        f"CB2: 连续 3 次恶化应触发 open, got circuit_state={cb_state}, "
        f"adjustments={len(result.get('adjustments', []))}, "
        f"skipped_reason={result.get('skipped_reason')}"
    )
    print(f"  CB2 PASS: 3 consecutive bad → circuit open, adjustments={len(result.get('adjustments', []))}")


def test_cb3_good_adjustment_resets_counter():
    """CB3: 调整后 hit_rate 未下降（好调整）→ consecutive_bad 重置为 0。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"cb3_{uuid.uuid4().hex[:8]}"
    _mock_chunks(conn, proj, 50)

    # 历史：consecutive_bad=2（接近熔断边缘）
    _force_autotune_state(proj, hit_rate_pct=30.0,  # 低命中率（会触发 top_k 扩大）
                           circuit={"state": "closed", "consecutive_bad": 2,
                                    "param_snapshot": {}},
                           hours_ago=7.0)

    # 本次：hit_rate 从 30% 升到 50%（好转）
    _mock_traces(conn, proj, hit_rate_pct=50.0, n=20)
    conn.commit()

    result = autotune(conn, proj)
    conn.close()

    cb_bad = result.get("consecutive_bad", -1)
    cb_state = result.get("circuit_state", "closed")
    assert cb_bad == 0, f"CB3: 好调整应重置 consecutive_bad 到 0, got {cb_bad}"
    assert cb_state == "closed", f"CB3: 好调整后应 closed, got {cb_state}"
    print(f"  CB3 PASS: good adjustment resets counter → consecutive_bad={cb_bad}, state={cb_state}")


def test_cb4_open_skips_autotune():
    """CB4: circuit open 且未超 cb_open_hours → autotune 被跳过。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"cb4_{uuid.uuid4().hex[:8]}"
    _mock_chunks(conn, proj, 50)
    _mock_traces(conn, proj, hit_rate_pct=40.0, n=20)
    conn.commit()

    # 强制 circuit open（刚打开，1 小时前）
    _force_autotune_state(proj, hit_rate_pct=40.0,
                           circuit={"state": "open", "consecutive_bad": 3,
                                    "open_since": datetime.now(timezone.utc).isoformat()},
                           hours_ago=7.0)

    result = autotune(conn, proj)
    conn.close()

    skipped = result.get("skipped_reason", "")
    assert "circuit_open" in skipped, (
        f"CB4: circuit open 应被跳过, got skipped_reason='{skipped}'"
    )
    print(f"  CB4 PASS: circuit open → skipped ({skipped[:50]})")


def test_cb5_open_allows_half_open_after_timeout():
    """CB5: circuit open 超过 cb_open_hours → 允许 half_open 试探（不被跳过）。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"cb5_{uuid.uuid4().hex[:8]}"
    _mock_chunks(conn, proj, 50)
    _mock_traces(conn, proj, hit_rate_pct=55.0, n=20)
    conn.commit()

    # circuit open 超过 cb_open_hours（默认 24h，模拟 25h 前打开）
    open_since = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
    _force_autotune_state(proj, hit_rate_pct=40.0,
                           circuit={"state": "open", "consecutive_bad": 3,
                                    "open_since": open_since},
                           hours_ago=7.0)

    result = autotune(conn, proj)
    conn.close()

    skipped = result.get("skipped_reason", "")
    assert "circuit_open" not in skipped, (
        f"CB5: 超过 cb_open_hours 应允许 half_open, got skipped_reason='{skipped}'"
    )
    print(f"  CB5 PASS: after cb_open_hours → half_open probe allowed (skipped='{skipped}')")


def test_cb6_half_open_success_closes_circuit():
    """CB6: half_open 状态试探成功（hit_rate 未恶化）→ circuit 关闭。"""
    conn = open_db()
    ensure_schema(conn)
    proj = f"cb6_{uuid.uuid4().hex[:8]}"
    _mock_chunks(conn, proj, 50)

    # 历史状态：half_open
    _force_autotune_state(proj, hit_rate_pct=40.0,
                           circuit={"state": "half_open", "consecutive_bad": 3,
                                    "param_snapshot": {}},
                           hours_ago=7.0)

    # 本次：hit_rate 55%（没有下降）→ 试探成功
    _mock_traces(conn, proj, hit_rate_pct=55.0, n=20)
    conn.commit()

    result = autotune(conn, proj)
    conn.close()

    cb_state = result.get("circuit_state", "unknown")
    assert cb_state == "closed", (
        f"CB6: half_open 试探成功应关闭 circuit, got {cb_state}"
    )
    print(f"  CB6 PASS: half_open success → circuit closed")


if __name__ == "__main__":
    print("Autotune Circuit Breaker 测试（iter494）")
    print("=" * 60)

    tests = [
        test_cb1_first_run_no_circuit,
        test_cb2_consecutive_bad_triggers_open,
        test_cb3_good_adjustment_resets_counter,
        test_cb4_open_skips_autotune,
        test_cb5_open_allows_half_open_after_timeout,
        test_cb6_half_open_success_closes_circuit,
    ]

    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  {t.__name__} FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n结果：{passed}/{passed+failed} 通过")
    if failed:
        sys.exit(1)
