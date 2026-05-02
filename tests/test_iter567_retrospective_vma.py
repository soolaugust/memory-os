"""
iter567: retrospective_vma_validate — Retroactive Write-Gate Enforcement

OS 类比：Linux ksm_scan (KSM, Red Hat 2009) — 后台扫描已有页面做回溯内容检查。
当 _vma_validate() 升级后，旧版漏入的碎片需要回溯扫描应用新规则。

测试 R4 规则：access_count=0 + age>=1d + _vma_validate(summary)=False → oom_adj=1000
"""
import os
import sys
import sqlite3
import time
from datetime import datetime, timezone, timedelta

# ── tmpfs 测试隔离 ──
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tmpfs  # noqa: F401 — import 即自动 mount tmpfs

from store_mm import oom_score_adj_rebalance, _retrospective_vma_validate
from store_vfs import open_db, ensure_schema


def _make_chunk(conn, summary, chunk_type="decision", importance=0.80,
                access_count=0, oom_adj=0, project="test_proj",
                age_days=2.0):
    """Helper: 创建一个测试 chunk。"""
    import uuid
    cid = str(uuid.uuid4())
    created = (datetime.now(timezone.utc) - timedelta(days=age_days)).isoformat()
    conn.execute(
        """INSERT INTO memory_chunks
           (id, summary, content, chunk_type, importance, access_count,
            oom_adj, project, source_session, created_at, last_accessed)
           VALUES (?, ?, '', ?, ?, ?, ?, ?, 'test_session', ?, ?)""",
        (cid, summary, chunk_type, importance, access_count,
         oom_adj, project, created, created)
    )
    conn.commit()
    return cid


def _get_oom(conn, cid):
    row = conn.execute("SELECT oom_adj FROM memory_chunks WHERE id=?", (cid,)).fetchone()
    return row[0] if row else None


# ── _retrospective_vma_validate 单元测试 ──

def test_vma_validate_table_row():
    """表格行碎片（| xxx | yyy |）应该返回 False。"""
    assert _retrospective_vma_validate("| 零访问率 | 42.7%→39.8%（-2.9pp） |") is False
    assert _retrospective_vma_validate("| 生产效果 | 垄断 chunk score 1.009 → 0.031 |") is False
    assert _retrospective_vma_validate("| 3192147e | git:78dc99a5695f | 1.000→0.032 | 96.8% |") is False


def test_vma_validate_valid_decision():
    """正常决策应该通过。"""
    assert _retrospective_vma_validate("选择 React 而非 Vue：社区活跃度更高，团队经验更丰富") is True
    assert _retrospective_vma_validate("冷启动保护：< 2 samples 不触发 throttle") is True
    assert _retrospective_vma_validate("memory-os 引用前必须验证路径存在") is True


def test_vma_validate_line_number_prefix():
    """行号前缀碎片。"""
    assert _retrospective_vma_validate("1260:- 性能：79 chunks 扫描 1.18ms") is False
    assert _retrospective_vma_validate("547:  if text[0] in set") is False


def test_vma_validate_status_report():
    """状态报告碎片。"""
    assert _retrospective_vma_validate("⚠️ DEGRADED: 9/10 passed") is False
    assert _retrospective_vma_validate("✅ PASS: all tests green") is False


def test_vma_validate_short_text():
    """过短文本。"""
    assert _retrospective_vma_validate("") is False
    assert _retrospective_vma_validate("abc") is False
    assert _retrospective_vma_validate(None) is False


def test_vma_validate_line_ref():
    """行号引用前缀。"""
    assert _retrospective_vma_validate("line 547: 检查条件不正确") is False
    assert _retrospective_vma_validate("L1260: 函数定义位置") is False


# ── R4 集成测试 ──

def test_r4_table_row_demoted():
    """R4: 零访问 + 年龄>=1d + 表格行碎片 → oom_adj 升至 1000。"""
    conn = open_db()
    ensure_schema(conn)
    cid = _make_chunk(conn, "| 生产效果 | 元数据/数据比 21.1x → 6.2x (-70.7%) |",
                      importance=0.80, age_days=2.0)
    result = oom_score_adj_rebalance(conn, "test_proj")
    assert result["r4_invalidated"] >= 1
    assert _get_oom(conn, cid) == 1000
    conn.close()


def test_r4_valid_decision_preserved():
    """R4: 有效决策不应被降级。"""
    conn = open_db()
    ensure_schema(conn)
    cid = _make_chunk(conn, "冷启动保护：< 2 samples 不触发 throttle",
                      importance=0.80, age_days=5.0)
    result = oom_score_adj_rebalance(conn, "test_proj")
    assert _get_oom(conn, cid) == 0  # 未变
    conn.close()


def test_r4_accessed_not_touched():
    """R4: 已访问的 chunk（即使 summary 是碎片）不应被 R4 降级。"""
    conn = open_db()
    ensure_schema(conn)
    cid = _make_chunk(conn, "| 表格行 | 但有人用过 |",
                      importance=0.80, access_count=3, age_days=5.0)
    result = oom_score_adj_rebalance(conn, "test_proj")
    # R4 不处理已访问的 chunk（acc != 0）
    assert _get_oom(conn, cid) != 1000
    conn.close()


def test_r4_young_not_touched():
    """R4: 年龄不足 1 天的 chunk 不应被 R4 降级（给新 chunk 冷启动时间）。"""
    conn = open_db()
    ensure_schema(conn)
    cid = _make_chunk(conn, "| 新表格行 | 刚写入 |",
                      importance=0.80, age_days=0.2)
    result = oom_score_adj_rebalance(conn, "test_proj")
    assert _get_oom(conn, cid) == 0  # 未变
    conn.close()


def test_r4_pinned_not_touched():
    """R4: mlock 保护的 chunk 不应被 R4 降级。"""
    conn = open_db()
    ensure_schema(conn)
    cid = _make_chunk(conn, "| pinned 表格行 | 受保护 |",
                      importance=0.80, age_days=5.0)
    # 添加 pin
    conn.execute("""CREATE TABLE IF NOT EXISTS chunk_pins
                    (chunk_id TEXT, project TEXT, pin_type TEXT,
                     pinned_at TEXT, PRIMARY KEY (chunk_id, project))""")
    conn.execute("INSERT OR REPLACE INTO chunk_pins VALUES (?, 'test_proj', 'hard', datetime('now'))",
                 (cid,))
    conn.commit()
    result = oom_score_adj_rebalance(conn, "test_proj")
    assert _get_oom(conn, cid) == 0  # 未变（pinned 跳过）
    conn.close()


def test_r4_already_max_oom_skipped():
    """R4: 已经 oom_adj=1000 的 chunk 不重复处理。"""
    conn = open_db()
    ensure_schema(conn)
    cid = _make_chunk(conn, "| 已标记 | 回收 |",
                      importance=0.80, oom_adj=1000, age_days=5.0)
    result = oom_score_adj_rebalance(conn, "test_proj")
    assert _get_oom(conn, cid) == 1000
    # 不应该计入 r4_invalidated（跳过了）
    conn.close()


def test_r4_highly_protected_not_touched():
    """R4: oom_adj <= -500 的用户显式保护 chunk 不被降级。"""
    conn = open_db()
    ensure_schema(conn)
    cid = _make_chunk(conn, "| 保护碎片 | 用户显式 |",
                      importance=0.80, oom_adj=-800, age_days=5.0)
    result = oom_score_adj_rebalance(conn, "test_proj")
    assert _get_oom(conn, cid) == -800  # 未变
    conn.close()


def test_r4_multiple_fragments():
    """R4: 多个碎片 chunk 同时被降级。"""
    conn = open_db()
    ensure_schema(conn)
    cid1 = _make_chunk(conn, "| 零访问率 | 42.7%→39.8% |", age_days=3.0)
    cid2 = _make_chunk(conn, "| 生产 | 1 chunk 修正 (1000→0) |", age_days=3.0)
    cid3 = _make_chunk(conn, "正常决策：选择方案 A", age_days=3.0)  # 正常
    result = oom_score_adj_rebalance(conn, "test_proj")
    assert _get_oom(conn, cid1) == 1000
    assert _get_oom(conn, cid2) == 1000
    assert _get_oom(conn, cid3) == 0  # 正常决策未变
    assert result["r4_invalidated"] >= 2
    conn.close()


def test_r4_idempotent():
    """R4: 幂等 — 第二次执行不应再产生新的调整。"""
    conn = open_db()
    ensure_schema(conn)
    cid = _make_chunk(conn, "| 碎片 | 幂等测试 |", age_days=3.0)
    r1 = oom_score_adj_rebalance(conn, "test_proj")
    assert r1["r4_invalidated"] >= 1
    r2 = oom_score_adj_rebalance(conn, "test_proj")
    assert r2["r4_invalidated"] == 0  # 幂等：已是 1000，不重复
    conn.close()


def test_r4_disabled_via_sysctl():
    """R4: 通过 sysctl 设 r4_min_age_days=999 实际禁用。"""
    conn = open_db()
    ensure_schema(conn)
    cid = _make_chunk(conn, "| 碎片 | 禁用测试 |", age_days=3.0)
    # 设置极大 age 门槛使 R4 不触发
    from config import sysctl_set
    sysctl_set("oom_rebalance.r4_min_age_days", 999.0)
    result = oom_score_adj_rebalance(conn, "test_proj")
    assert _get_oom(conn, cid) == 0  # 年龄不足 999d，不触发
    # 恢复
    sysctl_set("oom_rebalance.r4_min_age_days", 1.0)
    conn.close()


def test_r4_coexists_with_r1_r2_r3():
    """R4 与 R1/R2/R3 共存：各规则独立工作。"""
    conn = open_db()
    ensure_schema(conn)
    # R1 候选：高 oom + 已访问
    c_r1 = _make_chunk(conn, "R1 活跃 chunk", oom_adj=500, access_count=5, age_days=10.0)
    # R4 候选：零访问碎片
    c_r4 = _make_chunk(conn, "| R4 碎片 | 零访问 |", age_days=5.0)
    # 正常 chunk
    c_ok = _make_chunk(conn, "正常决策不应被任何规则触及", age_days=5.0)
    result = oom_score_adj_rebalance(conn, "test_proj")
    assert result["r1_demoted"] >= 1
    assert result["r4_invalidated"] >= 1
    assert _get_oom(conn, c_r1) == 0    # R1 降级
    assert _get_oom(conn, c_r4) == 1000  # R4 标记回收
    assert _get_oom(conn, c_ok) == 0     # 未变
    conn.close()


def test_performance():
    """R4 性能：100 chunks 扫描 < 50ms。"""
    conn = open_db()
    ensure_schema(conn)
    for i in range(50):
        _make_chunk(conn, f"| 碎片 {i} | 性能测试 |", age_days=3.0)
    for i in range(50):
        _make_chunk(conn, f"正常决策 {i}：选择方案 A 而非方案 B", age_days=3.0)
    t0 = time.time()
    result = oom_score_adj_rebalance(conn, "test_proj")
    elapsed = (time.time() - t0) * 1000
    print(f"  performance: scanned={result['scanned']} r4={result['r4_invalidated']} {elapsed:.1f}ms")
    assert elapsed < 50, f"Too slow: {elapsed:.1f}ms"
    # max_adjustments=20 限制单次最大调整数，所以 r4 <= 20
    assert result["r4_invalidated"] == 20  # 受 max_adjustments 限制
    conn.close()
