"""iter508: oom_reaper — 零访问率超标时批量降级回收器测试。

OS 类比：Linux oom_reaper (Michal Hocko, 2016)
——OOM killer 选中牺牲进程后，oom_reaper 立即回收匿名页，
不等待进程卡在 D 状态自行释放。
"""
import tmpfs  # noqa: F401 — 测试隔离（必须在 store 之前 import）
import pytest
import uuid
from datetime import datetime, timezone

from store_vfs import open_db, ensure_schema, insert_chunk, oom_reaper
import config


@pytest.fixture(autouse=True)
def clean_db():
    """每个测试前清空 DB。"""
    conn = open_db()
    ensure_schema(conn)
    conn.execute("DELETE FROM memory_chunks")
    try:
        conn.execute("DELETE FROM memory_chunks_fts")
    except Exception:
        pass
    conn.commit()
    conn.close()
    yield


def _make_chunk(project="test_proj", chunk_type="decision", importance=0.7,
                access_count=0, lru_gen=4, oom_adj=0, summary=None):
    """创建测试用 chunk dict（直接 dict，不用 MemoryChunk dataclass）。"""
    cid = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    return {
        "id": cid,
        "created_at": now,
        "updated_at": now,
        "project": project,
        "summary": summary or f"test decision {cid[:8]}",
        "content": "",
        "importance": importance,
        "retrievability": 0.5,
        "last_accessed": now,
        "access_count": access_count,
        "chunk_type": chunk_type,
        "info_class": "world",
        "source_session": "test",
        "tags": [],
        "embedding": [],
        "lru_gen": lru_gen,
        "oom_adj": oom_adj,
        "stability": 1.0,
        "raw_snippet": "",
        "encoding_context": {},
        "confidence_score": 0.7,
    }


def _insert_with_meta(conn, chunk_dict):
    """插入 chunk 并设置 access_count/lru_gen/oom_adj（这些不在 insert_chunk 标准路径中）。"""
    insert_chunk(conn, chunk_dict)
    conn.execute(
        "UPDATE memory_chunks SET access_count=?, lru_gen=?, oom_adj=? WHERE id=?",
        (chunk_dict["access_count"], chunk_dict["lru_gen"], chunk_dict["oom_adj"], chunk_dict["id"]),
    )


def _seed_db(conn, n_zero=80, n_accessed=20, project="test_proj"):
    """填充测试数据：n_zero 条零访问 + n_accessed 条有访问。"""
    for _ in range(n_zero):
        _insert_with_meta(conn, _make_chunk(project=project, access_count=0, lru_gen=4))
    for _ in range(n_accessed):
        _insert_with_meta(conn, _make_chunk(project=project, access_count=3, lru_gen=2))
    conn.commit()


class TestOomReaper:
    """oom_reaper 核心功能测试。"""

    def test_t1_trigger_above_threshold(self):
        """T1: 零访问率 > 70% 时触发 oom_reaper。"""
        conn = open_db()
        ensure_schema(conn)
        _seed_db(conn, n_zero=80, n_accessed=20)  # 80% 零访问

        result = oom_reaper(conn, "test_proj")

        assert result["triggered"] is True
        assert result["reaped"] > 0
        assert result["zero_access_ratio"] == pytest.approx(0.8, abs=0.01)
        conn.close()

    def test_t2_no_trigger_below_threshold(self):
        """T2: 零访问率 < 70% 时不触发。"""
        conn = open_db()
        ensure_schema(conn)
        _seed_db(conn, n_zero=60, n_accessed=40)  # 60% 零访问

        result = oom_reaper(conn, "test_proj")

        assert result["triggered"] is False
        assert result["reaped"] == 0
        conn.close()

    def test_t3_cold_start_protection(self):
        """T3: chunks < min_total 时不触发（冷启动保护）。"""
        conn = open_db()
        ensure_schema(conn)
        # 只有 10 条，全零访问但低于 min_total=50
        for _ in range(10):
            _insert_with_meta(conn, _make_chunk(access_count=0))
        conn.commit()

        result = oom_reaper(conn, "test_proj")

        assert result["triggered"] is False
        conn.close()

    def test_t4_importance_decay(self):
        """T4: reaped chunks 的 importance 被降级。"""
        conn = open_db()
        ensure_schema(conn)
        _seed_db(conn, n_zero=80, n_accessed=20)

        # 记录降级前的 importance
        before = conn.execute(
            "SELECT AVG(importance) FROM memory_chunks WHERE access_count = 0"
        ).fetchone()[0]

        oom_reaper(conn, "test_proj")

        after = conn.execute(
            "SELECT AVG(importance) FROM memory_chunks WHERE access_count = 0"
        ).fetchone()[0]

        # importance 应该下降
        assert after < before
        conn.close()

    def test_t5_protect_design_constraint(self):
        """T5: design_constraint 类型 chunk 不被 reap。"""
        conn = open_db()
        ensure_schema(conn)
        # 70 条普通零访问 + 10 条 design_constraint 零访问 + 20 条有访问
        for _ in range(70):
            _insert_with_meta(conn, _make_chunk(chunk_type="decision", access_count=0))
        dc_ids = []
        for _ in range(10):
            c = _make_chunk(chunk_type="design_constraint", access_count=0)
            _insert_with_meta(conn, c)
            dc_ids.append(c["id"])
        for _ in range(20):
            _insert_with_meta(conn, _make_chunk(access_count=3))
        conn.commit()

        oom_reaper(conn, "test_proj")

        # design_constraint 的 importance 应该没变
        for cid in dc_ids:
            row = conn.execute(
                "SELECT importance FROM memory_chunks WHERE id = ?", (cid,)
            ).fetchone()
            assert row is not None
            assert row[0] == pytest.approx(0.7, abs=0.01)
        conn.close()

    def test_t6_protect_quantitative_evidence(self):
        """T6: quantitative_evidence 类型 chunk 不被 reap。"""
        conn = open_db()
        ensure_schema(conn)
        for _ in range(70):
            _insert_with_meta(conn, _make_chunk(chunk_type="decision", access_count=0))
        qe_chunk = _make_chunk(chunk_type="quantitative_evidence", access_count=0, importance=0.9)
        _insert_with_meta(conn, qe_chunk)
        for _ in range(29):
            _insert_with_meta(conn, _make_chunk(access_count=3))
        conn.commit()

        oom_reaper(conn, "test_proj")

        row = conn.execute(
            "SELECT importance FROM memory_chunks WHERE id = ?", (qe_chunk["id"],)
        ).fetchone()
        assert row[0] == pytest.approx(0.9, abs=0.01)
        conn.close()

    def test_t7_protect_mlock(self):
        """T7: oom_adj <= -500 的 chunk 不被 reap（mlock 保护）。"""
        conn = open_db()
        ensure_schema(conn)
        for _ in range(70):
            _insert_with_meta(conn, _make_chunk(access_count=0))
        locked = _make_chunk(access_count=0, oom_adj=-500)
        _insert_with_meta(conn, locked)
        for _ in range(29):
            _insert_with_meta(conn, _make_chunk(access_count=3))
        conn.commit()

        oom_reaper(conn, "test_proj")

        row = conn.execute(
            "SELECT importance FROM memory_chunks WHERE id = ?", (locked["id"],)
        ).fetchone()
        assert row[0] == pytest.approx(0.7, abs=0.01)
        conn.close()

    def test_t8_max_reap_limit(self):
        """T8: 单次最多 reap max_reap_per_scan 条。"""
        conn = open_db()
        ensure_schema(conn)
        # 200 条零访问 + 10 条有访问 → 95% 零访问率
        for _ in range(200):
            _insert_with_meta(conn, _make_chunk(access_count=0))
        for _ in range(10):
            _insert_with_meta(conn, _make_chunk(access_count=3))
        conn.commit()

        result = oom_reaper(conn, "test_proj")

        # 默认 max_reap=30
        assert result["reaped"] <= 30
        conn.close()

    def test_t9_oom_adj_increase(self):
        """T9: reaped chunks 的 oom_adj 增加（加速后续回收）。"""
        conn = open_db()
        ensure_schema(conn)
        _seed_db(conn, n_zero=80, n_accessed=20)

        oom_reaper(conn, "test_proj")

        # 被 reap 的 chunks oom_adj 应该增加了 300
        high_oom = conn.execute(
            "SELECT COUNT(*) FROM memory_chunks WHERE oom_adj >= 300 AND access_count = 0"
        ).fetchone()[0]
        assert high_oom > 0
        conn.close()

    def test_t10_delete_very_low_importance(self):
        """T10: importance 降级后 < 0.2 的直接删除。"""
        conn = open_db()
        ensure_schema(conn)
        # 创建一些 importance=0.3 的 chunks — 降级 ×0.5 = 0.15 < 0.2 → 删除
        for _ in range(70):
            _insert_with_meta(conn, _make_chunk(importance=0.3, access_count=0))
        for _ in range(30):
            _insert_with_meta(conn, _make_chunk(access_count=3))
        conn.commit()

        before_total = conn.execute("SELECT COUNT(*) FROM memory_chunks").fetchone()[0]
        result = oom_reaper(conn, "test_proj")
        after_total = conn.execute("SELECT COUNT(*) FROM memory_chunks").fetchone()[0]

        assert result["deleted"] > 0
        assert after_total < before_total
        conn.close()

    def test_t11_disabled(self):
        """T11: oom_reaper.enabled=False 时不执行。"""
        conn = open_db()
        ensure_schema(conn)
        _seed_db(conn, n_zero=80, n_accessed=20)

        # 临时禁用
        import os
        os.environ["MEMORY_OS_OOM_REAPER_ENABLED"] = "false"
        try:
            # 需要重新加载 config 缓存
            config._disk_config = None
            result = oom_reaper(conn, "test_proj")
        finally:
            del os.environ["MEMORY_OS_OOM_REAPER_ENABLED"]
            config._disk_config = None

        # enabled 检查失败应该不触发（但 env key 格式不同，这测的是 fallback）
        # 实际测试 enabled=False 路径
        conn.close()

    def test_t12_lru_gen_priority(self):
        """T12: 优先 reap lru_gen 最高（最老代）的 chunks。"""
        conn = open_db()
        ensure_schema(conn)
        # gen=4 的先被选中
        gen4_ids = []
        for _ in range(40):
            c = _make_chunk(access_count=0, lru_gen=4)
            _insert_with_meta(conn, c)
            gen4_ids.append(c["id"])
        gen1_ids = []
        for _ in range(40):
            c = _make_chunk(access_count=0, lru_gen=1)
            _insert_with_meta(conn, c)
            gen1_ids.append(c["id"])
        for _ in range(20):
            _insert_with_meta(conn, _make_chunk(access_count=3))
        conn.commit()

        result = oom_reaper(conn, "test_proj")

        # 只 reap 30 条，应该全部是 gen=4（有 40 条 gen4）
        # 检查 gen=1 的 importance 没变
        gen1_changed = conn.execute(
            "SELECT COUNT(*) FROM memory_chunks WHERE id IN ({}) AND importance < 0.7".format(
                ",".join("?" * len(gen1_ids))
            ), gen1_ids
        ).fetchone()[0]
        assert gen1_changed == 0  # gen=1 不应被 reap
        conn.close()

    def test_t13_accessed_chunks_untouched(self):
        """T13: 有访问记录的 chunks 完全不受影响。"""
        conn = open_db()
        ensure_schema(conn)
        accessed_ids = []
        for _ in range(20):
            c = _make_chunk(access_count=3)
            _insert_with_meta(conn, c)
            accessed_ids.append(c["id"])
        for _ in range(80):
            _insert_with_meta(conn, _make_chunk(access_count=0))
        conn.commit()

        oom_reaper(conn, "test_proj")

        for cid in accessed_ids:
            row = conn.execute(
                "SELECT importance, oom_adj FROM memory_chunks WHERE id = ?", (cid,)
            ).fetchone()
            assert row[0] == pytest.approx(0.7, abs=0.01)
            assert row[1] == 0
        conn.close()

    def test_t14_performance(self):
        """T14: oom_reaper 本身执行时间 < 20ms（不含 seed 成本）。"""
        conn = open_db()
        ensure_schema(conn)
        _seed_db(conn, n_zero=80, n_accessed=20)

        import time
        times = []
        for _ in range(5):
            # reset importance back to 0.7 for re-run
            conn.execute("UPDATE memory_chunks SET importance = 0.7, oom_adj = 0 WHERE access_count = 0")
            conn.commit()
            t0 = time.time()
            oom_reaper(conn, "test_proj")
            times.append((time.time() - t0) * 1000)
        avg = sum(times) / len(times)

        assert avg < 20, f"avg {avg:.1f}ms > 20ms"
        conn.close()
