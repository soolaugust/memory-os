"""
iter549: vacuum — Database File Compaction
OS 类比：SSD Background GC / Firmware Compaction — VACUUM 收缩 SQLite 文件

Tests:
  1. 高碎片率触发 VACUUM — freelist >= threshold 时执行压缩
  2. 低碎片率不触发 — freelist < threshold 时跳过
  3. 小文件不触发 — DB < min_size_kb 时跳过
  4. 冷却期不触发 — 距上次 VACUUM < cooldown_hours 时跳过
  5. 冷却期过期后可再触发 — cooldown 到期后正常执行
  6. 文件不存在返回 db_not_found
  7. VACUUM 后文件实际收缩 — before > after
  8. VACUUM 后数据完整性 — 表/行数不变
  9. 幂等性 — 连续 VACUUM 第二次因低碎片跳过
  10. 冷却标记写入正确 — vacuum_last.json 包含 ts/freed_kb
  11. 性能 — VACUUM 5MB DB < 2000ms
  12. config tunables 生效 — 修改 threshold 影响触发逻辑
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tmpfs  # noqa: E402, F401 — 测试隔离
import json
import sqlite3
import time
import pytest
from pathlib import Path
from store_mm import vacuum
from config import get as _cfg, sysctl_set


def _create_fragmented_db(db_path: str, n_rows: int = 500, delete_pct: float = 0.7):
    """创建一个有大量碎片的测试 DB。

    策略：写入大量数据，然后删除大部分，制造 freelist。
    """
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE IF NOT EXISTS test_data (id INTEGER PRIMARY KEY, payload TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS memory_chunks (id TEXT PRIMARY KEY, content TEXT, chunk_state TEXT DEFAULT 'ACTIVE')")

    # 插入大量数据（每行约 1KB 的 payload）
    payload = "X" * 1000
    conn.executemany(
        "INSERT INTO test_data (payload) VALUES (?)",
        [(payload,) for _ in range(n_rows)]
    )
    # 插入一些 memory_chunks 来模拟真实场景
    for i in range(10):
        conn.execute(
            "INSERT INTO memory_chunks (id, content, chunk_state) VALUES (?, ?, 'ACTIVE')",
            (f"chunk_{i}", f"content_{i}" * 50)
        )
    conn.commit()

    # 删除大部分数据制造碎片
    delete_count = int(n_rows * delete_pct)
    conn.execute(f"DELETE FROM test_data WHERE id <= {delete_count}")
    conn.commit()
    conn.close()

    return db_path


def _get_freelist_pct(db_path: str) -> float:
    """获取 DB 的 freelist 百分比。"""
    conn = sqlite3.connect(db_path)
    page_count = conn.execute("PRAGMA page_count").fetchone()[0]
    freelist = conn.execute("PRAGMA freelist_count").fetchone()[0]
    conn.close()
    return (freelist / page_count * 100) if page_count > 0 else 0


def _cleanup_cooldown(db_path: str):
    """清理冷却标记文件。"""
    cooldown_file = Path(db_path).parent / "vacuum_last.json"
    if cooldown_file.exists():
        cooldown_file.unlink()


# ── Test 1: 高碎片率触发 VACUUM ──

def test_vacuum_triggers_on_high_fragmentation(tmp_path):
    """freelist >= threshold 时执行 VACUUM 压缩。"""
    db_path = str(tmp_path / "store.db")
    _create_fragmented_db(db_path, n_rows=500, delete_pct=0.8)
    _cleanup_cooldown(db_path)

    freelist_before = _get_freelist_pct(db_path)
    assert freelist_before >= 40, f"Setup failed: freelist only {freelist_before}%"

    result = vacuum(db_path)

    assert result["vacuumed"] is True
    assert result["freed_kb"] > 0
    assert result["freed_pct"] > 0
    assert "compacted" in result["reason"]


# ── Test 2: 低碎片率不触发 ──

def test_vacuum_skips_low_fragmentation(tmp_path):
    """freelist < threshold 时不执行 VACUUM。"""
    db_path = str(tmp_path / "store.db")
    # 创建 DB 但不删除数据（碎片率低）— 需要足够大超过 min_size
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE test_data (id INTEGER PRIMARY KEY, payload TEXT)")
    payload = "X" * 1000
    conn.executemany("INSERT INTO test_data (payload) VALUES (?)", [(payload,) for _ in range(600)])
    conn.commit()
    conn.close()
    _cleanup_cooldown(db_path)

    # 确保文件 >= min_size_kb 但碎片率低
    assert os.path.getsize(db_path) >= 512 * 1024

    result = vacuum(db_path)

    assert result["vacuumed"] is False
    assert "low_fragmentation" in result["reason"]


# ── Test 3: 小文件不触发 ──

def test_vacuum_skips_small_db(tmp_path):
    """DB 文件 < min_size_kb 时不触发。"""
    db_path = str(tmp_path / "store.db")
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE t (id INTEGER)")
    conn.execute("INSERT INTO t VALUES (1)")
    conn.commit()
    conn.close()
    _cleanup_cooldown(db_path)

    # 确保文件很小
    assert os.path.getsize(db_path) < 512 * 1024

    result = vacuum(db_path)

    assert result["vacuumed"] is False
    assert "small_db" in result["reason"]


# ── Test 4: 冷却期不触发 ──

def test_vacuum_skips_during_cooldown(tmp_path):
    """距上次 VACUUM < cooldown_hours 时跳过。"""
    db_path = str(tmp_path / "store.db")
    _create_fragmented_db(db_path, n_rows=500, delete_pct=0.8)

    # 写入一个最近的冷却标记
    cooldown_file = tmp_path / "vacuum_last.json"
    cooldown_file.write_text(json.dumps({"ts": time.time(), "freed_kb": 100}))

    result = vacuum(db_path)

    assert result["vacuumed"] is False
    assert "cooldown" in result["reason"]


# ── Test 5: 冷却期过期后可再触发 ──

def test_vacuum_runs_after_cooldown_expires(tmp_path):
    """cooldown 到期后正常执行。"""
    db_path = str(tmp_path / "store.db")
    _create_fragmented_db(db_path, n_rows=500, delete_pct=0.8)

    # 写入过期的冷却标记（25小时前）
    cooldown_file = tmp_path / "vacuum_last.json"
    cooldown_file.write_text(json.dumps({"ts": time.time() - 25 * 3600, "freed_kb": 50}))

    result = vacuum(db_path)

    assert result["vacuumed"] is True
    assert result["freed_kb"] > 0


# ── Test 6: 文件不存在 ──

def test_vacuum_handles_missing_file(tmp_path):
    """DB 文件不存在时返回 db_not_found。"""
    result = vacuum(str(tmp_path / "nonexistent.db"))

    assert result["vacuumed"] is False
    assert result["reason"] == "db_not_found"


# ── Test 7: VACUUM 后文件实际收缩 ──

def test_vacuum_actually_shrinks_file(tmp_path):
    """VACUUM 执行后 DB 文件大小减小。"""
    db_path = str(tmp_path / "store.db")
    _create_fragmented_db(db_path, n_rows=500, delete_pct=0.8)
    _cleanup_cooldown(db_path)

    before_size = os.path.getsize(db_path)
    result = vacuum(db_path)
    after_size = os.path.getsize(db_path)

    assert result["vacuumed"] is True
    assert after_size < before_size
    assert result["after_size_kb"] < result["before_size_kb"]


# ── Test 8: VACUUM 后数据完整性 ──

def test_vacuum_preserves_data_integrity(tmp_path):
    """VACUUM 后所有表数据完整无丢失。"""
    db_path = str(tmp_path / "store.db")
    _create_fragmented_db(db_path, n_rows=500, delete_pct=0.7)
    _cleanup_cooldown(db_path)

    # 记录 VACUUM 前的数据
    conn = sqlite3.connect(db_path)
    remaining_rows = conn.execute("SELECT COUNT(*) FROM test_data").fetchone()[0]
    chunk_count = conn.execute("SELECT COUNT(*) FROM memory_chunks").fetchone()[0]
    chunk_ids = [r[0] for r in conn.execute("SELECT id FROM memory_chunks ORDER BY id").fetchall()]
    conn.close()

    # 执行 VACUUM
    result = vacuum(db_path)
    assert result["vacuumed"] is True

    # 验证数据完整
    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM test_data").fetchone()[0] == remaining_rows
    assert conn.execute("SELECT COUNT(*) FROM memory_chunks").fetchone()[0] == chunk_count
    post_ids = [r[0] for r in conn.execute("SELECT id FROM memory_chunks ORDER BY id").fetchall()]
    assert post_ids == chunk_ids
    conn.close()


# ── Test 9: 幂等性 ──

def test_vacuum_idempotent(tmp_path):
    """连续 VACUUM：第一次压缩，第二次因碎片低或文件小而跳过。"""
    db_path = str(tmp_path / "store.db")
    _create_fragmented_db(db_path, n_rows=500, delete_pct=0.8)
    _cleanup_cooldown(db_path)

    # 第一次成功
    r1 = vacuum(db_path)
    assert r1["vacuumed"] is True

    # 第二次：清除 cooldown，碎片率已经很低或文件已收缩到小于阈值
    _cleanup_cooldown(db_path)
    r2 = vacuum(db_path)
    assert r2["vacuumed"] is False
    assert "low_fragmentation" in r2["reason"] or "small_db" in r2["reason"]


# ── Test 10: 冷却标记写入 ──

def test_vacuum_writes_cooldown_marker(tmp_path):
    """VACUUM 后写入 vacuum_last.json 包含 ts 和 freed_kb。"""
    db_path = str(tmp_path / "store.db")
    _create_fragmented_db(db_path, n_rows=500, delete_pct=0.8)
    _cleanup_cooldown(db_path)

    before_ts = time.time()
    result = vacuum(db_path)
    assert result["vacuumed"] is True

    cooldown_file = tmp_path / "vacuum_last.json"
    assert cooldown_file.exists()
    data = json.loads(cooldown_file.read_text())
    assert data["ts"] >= before_ts
    assert data["freed_kb"] > 0
    assert "freed_pct" in data


# ── Test 11: 性能 ──

def test_vacuum_performance(tmp_path):
    """VACUUM 5MB DB < 2000ms。"""
    db_path = str(tmp_path / "store.db")
    # 创建更大的 DB
    _create_fragmented_db(db_path, n_rows=2000, delete_pct=0.8)
    _cleanup_cooldown(db_path)

    t0 = time.time()
    result = vacuum(db_path)
    elapsed = (time.time() - t0) * 1000

    assert result["vacuumed"] is True
    assert elapsed < 2000, f"VACUUM took {elapsed:.0f}ms > 2000ms"
    assert result["duration_ms"] < 2000


# ── Test 12: config tunables 生效 ──

def test_vacuum_respects_config_threshold(tmp_path):
    """修改 vacuum.threshold_pct 影响触发逻辑。"""
    db_path = str(tmp_path / "store.db")
    _create_fragmented_db(db_path, n_rows=500, delete_pct=0.5)  # 约 50% 碎片
    _cleanup_cooldown(db_path)

    freelist = _get_freelist_pct(db_path)

    # 设置高阈值（高于实际碎片率）→ 不触发
    sysctl_set("vacuum.threshold_pct", 95.0)
    r1 = vacuum(db_path)
    assert r1["vacuumed"] is False

    # 恢复低阈值（低于实际碎片率）→ 触发
    sysctl_set("vacuum.threshold_pct", 10.0)
    _cleanup_cooldown(db_path)
    r2 = vacuum(db_path)
    assert r2["vacuumed"] is True

    # 恢复默认
    sysctl_set("vacuum.threshold_pct", 40.0)
