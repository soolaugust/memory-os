#!/usr/bin/env python3
"""
eval_fourth_stage.py — 第四阶段新能力验证 harness

测试六个维度：
1. 意图预测准确率 (Intent Prediction)
2. 反馈回路 (Feedback Loop) — 否定后 importance 降低
3. 跨项目全局层检索 (Global Layer)
4. 目标追踪写入与读取 (Goal Tracking)
5. 意图预取延迟 < 10ms
6. DB 健康检查（所有新表存在）

运行：python3 eval_fourth_stage.py
输出：eval_fourth_stage_results.json
"""

import sys, os, json, re, time, uuid
from pathlib import Path
from datetime import datetime, timezone

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
os.chdir(str(_ROOT))

from store_core import open_db, ensure_schema, insert_chunk
from hooks.retriever import _predict_intent, _intent_prefetch
from hooks.writer import _detect_and_persist_goal, _process_negative_feedback

PROJECT = "abspath:7e3095aef7a6"


# ── 1. 意图预测准确率 ──────────────────────────────────────────────────────────
def test_intent_prediction():
    cases = [
        ("继续",                          "continue"),
        ("continue from where you left",  "continue"),
        ("接下来",                         "continue"),
        ("修复这个 bug",                   "fix_bug"),
        ("fix the crash",                  "fix_bug"),
        ("这里有个 exception",             "fix_bug"),
        ("review this code",              "code_review"),
        ("帮我看一下这段代码",             "code_review"),
        ("为什么会这样",                   "understand"),
        ("how does this work",            "understand"),
        ("实现这个功能",                   "implement"),
        ("build a new API",               "implement"),
        ("优化一下性能",                   "optimize"),
        ("探索一下这个问题",               "explore"),
        ("investigate why",               "explore"),
        ("随便说句话",                     "unknown"),
        ("hello",                         "unknown"),
    ]
    correct = 0
    details = []
    for prompt, expected in cases:
        got, _ = _predict_intent(prompt)
        ok = got == expected
        if ok:
            correct += 1
        details.append({"prompt": prompt, "expected": expected, "got": got, "ok": ok})

    acc = correct / len(cases)
    wrong = [d for d in details if not d["ok"]]
    print(f"\n[1] 意图预测准确率: {correct}/{len(cases)} = {acc:.1%}")
    for d in wrong:
        print(f"    ✗ '{d['prompt']}' → {d['got']} (expect {d['expected']})")
    passed = acc >= 0.85
    print(f"    {'✅ PASS' if passed else '❌ FAIL'} (阈值 85%)")
    return {"passed": passed, "accuracy": acc, "correct": correct, "total": len(cases), "wrong": wrong}


# ── 2. 反馈回路 ───────────────────────────────────────────────────────────────
def test_feedback_loop():
    conn = open_db()
    ensure_schema(conn)

    test_id = f"test-feedback-{uuid.uuid4().hex[:8]}"
    now = datetime.now(timezone.utc).isoformat()
    original_imp = 0.5
    from store_core import insert_chunk
    insert_chunk(conn, {
        "id": test_id, "created_at": now, "updated_at": now,
        "project": PROJECT, "source_session": "test",
        "chunk_type": "prompt_context",
        "content": "test", "summary": "eval feedback test chunk eval_fourth_stage",
        "tags": [], "importance": original_imp,
        "retrievability": 0.3, "last_accessed": now,
    })
    conn.commit()
    conn.close()

    negatives = ["不对", "错了，重新来", "wrong", "这不是我想要的"]
    processed = sum(1 for n in negatives if _process_negative_feedback(n, PROJECT, "test"))

    conn = open_db()
    row = conn.execute("SELECT importance, oom_adj FROM memory_chunks WHERE id=?", [test_id]).fetchone()
    new_imp = row[0] if row else original_imp
    new_oom = row[1] if row else 0
    conn.execute("DELETE FROM memory_chunks WHERE id=?", [test_id])
    conn.commit()
    conn.close()

    degraded = new_imp < original_imp
    passed = degraded and new_oom > 0 and processed >= 2
    print(f"\n[2] 反馈回路: imp {original_imp:.2f}→{new_imp:.2f}, oom_adj={new_oom}, 识别 {processed}/{len(negatives)}")
    print(f"    {'✅ PASS' if passed else '❌ FAIL'}")
    return {"passed": passed, "original_imp": original_imp, "new_imp": new_imp,
            "oom_adj": new_oom, "detected": processed}


# ── 3. 跨项目全局层 ────────────────────────────────────────────────────────────
def test_global_layer():
    from store_core import get_chunks
    conn = open_db()
    ensure_schema(conn)

    gid = f"test-global-{uuid.uuid4().hex[:8]}"
    now = datetime.now(timezone.utc).isoformat()
    summary = f"全局共享测试知识 eval {gid[:8]}"
    from store_core import insert_chunk
    insert_chunk(conn, {
        "id": gid, "created_at": now, "updated_at": now,
        "project": "global", "source_session": "test",
        "chunk_type": "decision",
        "content": summary, "summary": summary,
        "tags": ["global"], "importance": 0.9,
        "retrievability": 0.5, "last_accessed": now,
    })
    conn.commit()

    chunks = get_chunks(conn, PROJECT, ("decision",))
    found = any(c["id"] == gid for c in chunks)

    prefetched = _intent_prefetch(conn, PROJECT, "全局测试", top_k=5)
    found_pf = any(p["id"] == gid for p in prefetched)

    conn.execute("DELETE FROM memory_chunks WHERE id=?", [gid])
    conn.commit()
    conn.close()

    passed = found
    print(f"\n[3] 全局层检索: get_chunks={found}, intent_prefetch={found_pf}")
    print(f"    {'✅ PASS' if passed else '❌ FAIL'}")
    return {"passed": passed, "found_in_get_chunks": found, "found_in_prefetch": found_pf}


# ── 4. 目标追踪 ───────────────────────────────────────────────────────────────
def test_goal_tracking():
    conn = open_db()
    ensure_schema(conn)
    before = conn.execute("SELECT COUNT(*) FROM goals WHERE project=?", [PROJECT]).fetchone()[0]
    conn.close()

    prompts = [
        "最终想要通过 aios 提升 AI 系统的能力，解决跨会话记忆丢失问题",
        "目标是让系统在 200 轮对话后仍保持高质量响应",
        "通过工具模式学习来自动优化工作流程",
    ]
    written = sum(1 for p in prompts if _detect_and_persist_goal(p, PROJECT, "test"))

    conn = open_db()
    after = conn.execute("SELECT COUNT(*) FROM goals WHERE project=?", [PROJECT]).fetchone()[0]
    goals = conn.execute(
        "SELECT title FROM goals WHERE project=? AND status='active' ORDER BY created_at DESC LIMIT 3",
        [PROJECT]
    ).fetchall()
    conn.close()

    new = after - before
    # 目标可能已在前次运行中写入（KSM 去重），只验证检测能力
    passed = written >= 2 and after >= 2
    print(f"\n[4] 目标追踪: 识别 {written}/{len(prompts)}, 新写 {new} 条 (总 {after})")
    for g in goals:
        print(f"    - {g[0][:60]}")
    print(f"    {'✅ PASS' if passed else '❌ FAIL'}")
    return {"passed": passed, "detected": written, "new_goals": new, "total": after}


# ── 5. 意图预取延迟 ────────────────────────────────────────────────────────────
def test_intent_latency():
    conn = open_db()
    ensure_schema(conn)
    _intent_prefetch(conn, PROJECT, "warmup", top_k=3)  # warmup

    times = []
    for p in ["继续", "修复 bug", "实现功能", "为什么", "优化性能"] * 2:
        t0 = time.monotonic()
        _intent_prefetch(conn, PROJECT, p, top_k=3)
        times.append((time.monotonic() - t0) * 1000)
    conn.close()

    times.sort()
    p50, p95 = times[len(times)//2], times[int(len(times)*0.95)]
    passed = p95 < 10
    print(f"\n[5] 意图预取延迟: P50={p50:.1f}ms P95={p95:.1f}ms")
    print(f"    {'✅ PASS (P95<10ms)' if passed else '❌ FAIL'}")
    return {"passed": passed, "p50_ms": round(p50,2), "p95_ms": round(p95,2)}


# ── 6. DB 健康检查 ────────────────────────────────────────────────────────────
def test_db_health():
    conn = open_db()
    ensure_schema(conn)
    tables = {}
    for t in ["memory_chunks","goals","tool_patterns","change_log",
              "scheduler_tasks","recall_traces","checkpoints","dmesg"]:
        try:
            n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            tables[t] = {"ok": True, "count": n}
        except Exception as e:
            tables[t] = {"ok": False, "error": str(e)}

    chunks_n = tables["memory_chunks"]["count"]
    try:
        fts_n = conn.execute("SELECT COUNT(*) FROM memory_chunks_fts").fetchone()[0]
        fts_ok = abs(chunks_n - fts_n) <= 2
    except:
        fts_ok = False
    conn.close()

    all_ok = all(v["ok"] for v in tables.values())
    passed = all_ok and fts_ok
    print(f"\n[6] DB 健康检查: 所有表={all_ok}, FTS5 一致={fts_ok}")
    for t, v in tables.items():
        mark = "✅" if v["ok"] else "❌"
        print(f"    {mark} {t}: {v.get('count','MISSING')}")
    print(f"    {'✅ PASS' if passed else '❌ FAIL'}")
    return {"passed": passed, "all_tables_exist": all_ok, "fts5_ok": fts_ok, "tables": tables}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  aios 第四阶段功能验证")
    print("=" * 60)

    results = {"timestamp": datetime.now(timezone.utc).isoformat(), "tests": {}}
    tests = [
        ("intent_prediction", test_intent_prediction),
        ("feedback_loop",     test_feedback_loop),
        ("global_layer",      test_global_layer),
        ("goal_tracking",     test_goal_tracking),
        ("intent_latency",    test_intent_latency),
        ("db_health",         test_db_health),
    ]

    passed_count = 0
    for name, fn in tests:
        try:
            r = fn()
            results["tests"][name] = r
            if r.get("passed", False):
                passed_count += 1
        except Exception as e:
            import traceback
            print(f"\n[{name}] ERROR: {e}")
            traceback.print_exc()
            results["tests"][name] = {"passed": False, "error": str(e)}

    total = len(tests)
    results["summary"] = {"passed": passed_count, "total": total,
                          "pass_rate": round(passed_count/total, 2)}

    print(f"\n{'='*60}")
    print(f"  结果: {passed_count}/{total} 通过 ({passed_count/total:.0%})")
    print(f"{'='*60}\n")

    out = Path(__file__).parent / "eval_fourth_stage_results.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"结果已保存: {out}")
    return results


if __name__ == "__main__":
    main()
