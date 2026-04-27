#!/usr/bin/env python3
"""
iter106 SNR Benchmark — 信噪比优化验证
测试三项改动：
  A. prompt_context 清除（DB 中不应再有此类型）
  B. _is_quality_decision 过滤：低质量 decision 被拒绝
  C. lru_gen_boost：gen=0 比 gen=8 高 0.06 分
  D. lru_gen 影响 retrieval 排序（gen=0 排前）

跑法：python3 eval_snr_iter106.py
"""
import sys
import os
import sqlite3
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "hooks"))

DB_PATH = Path.home() / ".claude" / "memory-os" / "store.db"

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = []


def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    results.append((name, condition, detail))
    print(f"{status}  {name}")
    if detail:
        print(f"       {detail}")


# ── A. prompt_context 清除验证 ──────────────────────────────
print("\n=== A. prompt_context 清除 ===")
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute("SELECT count(*) FROM memory_chunks WHERE chunk_type='prompt_context'")
pc_count = cur.fetchone()[0]
check("A1: DB 中无 prompt_context chunks", pc_count == 0, f"found={pc_count}")

# ── B. _is_quality_decision 过滤 ──────────────────────────
print("\n=== B. decision 写入阈值过滤 ===")
from extractor import _is_quality_decision

decision_tests = [
    # (summary, should_pass, label)
    ("选择 sqlite 而非 postgresql 因为嵌入式部署", True, "含决策动词+对比"),
    ("采用 BM25 替代全表扫描，P99 从 45ms→12ms", True, "含动词+度量"),
    ("store.py 中 insert_chunk 函数逻辑", True, "含文件路径"),
    ("性能提升 3x 达到 95ms", True, "含度量"),
    ("项目已完成初步探索阶段", False, "纯状态快照，无锚点"),
    ("这是一个重要的发现", False, "泛化陈述"),
    ("分析结果表明系统运行正常", False, "无具体锚点"),
    ("当前架构比较合理", False, "模糊方向声明"),
    ("因为这个设计更好", True, "含因果词"),
    ("`lru_gen` 字段驱动置换顺序", True, "含代码标识符"),
]

b_pass = 0
for summary, expected, label in decision_tests:
    got = _is_quality_decision(summary)
    ok = got == expected
    if ok:
        b_pass += 1
    check(f"B: {label[:30]}", ok, f"expected={expected}, got={got}")

check("B_total: 过滤准确率 ≥ 90%", b_pass >= 9, f"{b_pass}/{len(decision_tests)}")

# ── C. lru_gen_boost 数值验证 ─────────────────────────────
print("\n=== C. lru_gen_boost 数值 ===")
from scorer import lru_gen_boost, retrieval_score

check("C1: gen=0 → +0.06", abs(lru_gen_boost(0) - 0.06) < 0.001, f"got={lru_gen_boost(0):.4f}")
check("C2: gen=4 → +0.03", abs(lru_gen_boost(4) - 0.03) < 0.001, f"got={lru_gen_boost(4):.4f}")
check("C3: gen=8 →  0.00", abs(lru_gen_boost(8) - 0.00) < 0.001, f"got={lru_gen_boost(8):.4f}")
check("C4: gen=None → 0.0", lru_gen_boost(None) == 0.0, f"got={lru_gen_boost(None)}")

s0 = retrieval_score(0.8, 0.9, "2026-04-21T10:00:00", access_count=3, lru_gen=0)
s8 = retrieval_score(0.8, 0.9, "2026-04-21T10:00:00", access_count=3, lru_gen=8)
check("C5: gen=0 score > gen=8 score", s0 > s8, f"gen0={s0:.4f}, gen8={s8:.4f}, diff={s0-s8:.4f}")
check("C6: 差值精确等于 0.06", abs(s0 - s8 - 0.06) < 0.001, f"diff={s0-s8:.4f}")

# ── D. fts_search 返回 lru_gen 字段 ──────────────────────
print("\n=== D. fts_search lru_gen 字段 ===")
from store_vfs import fts_search

rows = fts_search(conn, "query expansion", "abspath:7e3095aef7a6", top_k=5)
check("D1: fts_search 返回结果", len(rows) > 0, f"rows={len(rows)}")
if rows:
    has_lru = all("lru_gen" in r for r in rows)
    check("D2: 所有结果含 lru_gen 字段", has_lru, str(rows[0].keys()))
    lru_vals = [r["lru_gen"] for r in rows]
    check("D3: lru_gen 值为整数", all(isinstance(v, int) for v in lru_vals), str(lru_vals))

# ── E. DB 整体 SNR 指标 ────────────────────────────────────
print("\n=== E. DB SNR 健康指标 ===")
cur.execute("SELECT count(*), sum(access_count), sum(case when access_count>0 then 1 else 0 end) FROM memory_chunks")
total, total_acc, ever_acc = cur.fetchone()
snr_ratio = ever_acc / max(total, 1)
check("E1: ever-accessed 比例 ≥ 35%", snr_ratio >= 0.35, f"{ever_acc}/{total} = {snr_ratio:.1%}")

cur.execute("SELECT count(*) FROM memory_chunks WHERE chunk_type='decision'")
dec_count = cur.fetchone()[0]
check("E2: decision chunks 数量合理 (≤ 350)", dec_count <= 350, f"count={dec_count}")

cur.execute("SELECT count(*) FROM memory_chunks WHERE chunk_type='design_constraint'")
dc_count = cur.fetchone()[0]
check("E3: design_constraint 存在", dc_count > 0, f"count={dc_count}")

conn.close()

# ── Summary ────────────────────────────────────────────────
print()
passed = sum(1 for _, ok, _ in results if ok)
total_r = len(results)
print(f"=== Summary: {passed}/{total_r} passed ({'%.0f' % (passed/total_r*100)}%) ===")
if passed < total_r:
    print("\nFailed cases:")
    for name, ok, detail in results:
        if not ok:
            print(f"  ❌ {name}: {detail}")
sys.exit(0 if passed == total_r else 1)
